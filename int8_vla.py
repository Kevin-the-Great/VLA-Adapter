"""
真 INT8 静态量化推理模块 (VLA-Adapter)
======================================

纯 PyTorch 实现，不依赖 torchao。

量化配置:
  - 权重: Symmetric INT8, per-channel (每个 output channel 一个 scale)
  - 激活: Symmetric INT8, per-tensor (每层一个静态 scale)

流程:
  1. smooth_qwen()          对 Qwen2.5 做 SmoothQuant
  2. attach_minmax_hooks()  给所有目标 nn.Linear 挂 hook
  3. 跑校准 episode         hook 自动收集每层激活 min/max
  4. remove_hooks()         摘掉 hook
  5. convert_to_int8()      nn.Linear → StaticInt8Linear（用收集到的 scale）
  6. 正式推理               torch._int_mm 做真 INT8 矩阵乘法
"""

import os
import functools
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Part 1: Smooth Qwen2.5
# ============================================================

@torch.no_grad()
def smooth_ln_fcs(ln, fcs, act_scales, alpha=0.5):
    if not isinstance(fcs, list):
        fcs = [fcs]

    device = ln.weight.device
    dtype = ln.weight.dtype
    act_scales = act_scales.to(device=device, dtype=dtype)

    weight_scales = torch.cat(
        [fc.weight.abs().max(dim=0, keepdim=True)[0] for fc in fcs], dim=0
    )
    weight_scales = weight_scales.max(dim=0)[0].clamp(min=1e-5)

    scales = (
        (act_scales.pow(alpha) / weight_scales.pow(1 - alpha))
        .clamp(min=1e-5).to(device).to(dtype)
    )

    ln.weight.div_(scales)
    if hasattr(ln, 'bias') and ln.bias is not None:
        ln.bias.div_(scales)
    for fc in fcs:
        fc.weight.mul_(scales.view(1, -1))


@torch.no_grad()
def smooth_qwen(model, act_scales, alpha=0.5):
    """对 Qwen2.5 的所有 decoder layer 做 smooth。"""
    print(f"[smooth] Smooth Qwen2.5 (alpha={alpha})")

    qwen_layers = None
    for path_fn in [
        lambda m: m.llm_backbone.llm.model.layers,
        lambda m: m.language_model.model.layers,
        lambda m: m.model.layers,
    ]:
        try:
            qwen_layers = path_fn(model)
            print(f"  找到 {len(qwen_layers)} 个 decoder layers")
            break
        except (AttributeError, TypeError):
            continue

    if qwen_layers is None:
        raise RuntimeError("找不到 Qwen2.5 decoder layers!")

    smooth_count = 0
    for i, layer in enumerate(qwen_layers):
        attn = layer.self_attn

        q_key = next((k for k in act_scales if f"layers.{i}.self_attn.q_proj" in k), None)
        if q_key is not None:
            smooth_ln_fcs(
                layer.input_layernorm,
                [attn.q_proj, attn.k_proj, attn.v_proj],
                act_scales[q_key], alpha=alpha
            )
            smooth_count += 1

        gate_key = next((k for k in act_scales if f"layers.{i}.mlp.gate_proj" in k), None)
        if gate_key is not None:
            smooth_ln_fcs(
                layer.post_attention_layernorm,
                [layer.mlp.gate_proj, layer.mlp.up_proj],
                act_scales[gate_key], alpha=alpha
            )
            smooth_count += 1

    print(f"  Smooth 完成: {smooth_count} 组 LN-Linear 被修改")
    return smooth_count


# ============================================================
# Part 2: 判断哪些层需要量化
# ============================================================

def should_quantize(name):
    """
    量化: ViT (DINOv2, SigLIP) + Projector + Qwen2.5 的 Linear
    不量化: Policy Head, lm_head, Embedding
    """
    nl = name.lower()
    if any(kw in nl for kw in ['policy', 'bridge', 'action_head']):
        return False
    if 'lm_head' in nl:
        return False
    if 'embed' in nl and 'proj' not in nl:
        return False
    return True


# ============================================================
# Part 3: Hook 方案收集激活 min/max
# ============================================================

def attach_minmax_hooks(model):
    """
    给所有目标 nn.Linear 挂 hook，收集输入激活的 per-tensor min/max。
    和 generate_vla_act_scales.py 的原理完全一样，只是记录的是
    全局 min/max（两个数）而不是 per-channel 最大值（一个向量）。

    Returns:
        act_minmax: dict，key=层名，value={'min': float, 'max': float}
        hooks: list，用完调用 remove_hooks(hooks) 摘掉
    """
    act_minmax = {}

    def minmax_hook(m, x, y, name):
        if isinstance(x, tuple):
            x = x[0]
        with torch.no_grad():
            val = x.detach().float()
            batch_min = val.min().item()
            batch_max = val.max().item()
            if name not in act_minmax:
                act_minmax[name] = {'min': batch_min, 'max': batch_max}
            else:
                act_minmax[name]['min'] = min(act_minmax[name]['min'], batch_min)
                act_minmax[name]['max'] = max(act_minmax[name]['max'], batch_max)

    hooks = []
    count = 0
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if not should_quantize(name):
            continue
        hooks.append(
            module.register_forward_hook(
                functools.partial(minmax_hook, name=name)
            )
        )
        count += 1

    print(f"[int8] 挂了 {count} 个 hook (目标 nn.Linear 层)")
    return act_minmax, hooks


def remove_hooks(hooks):
    for h in hooks:
        h.remove()
    print(f"[int8] 移除了 {len(hooks)} 个 hook")


# ============================================================
# Part 4: StaticInt8Linear（推理阶段）
# ============================================================

class StaticInt8Linear(nn.Module):
    """
    真 INT8 Linear，推理时用 torch._int_mm。

    权重: per-channel symmetric INT8
    激活: per-tensor symmetric INT8，静态 scale（校准时确定，推理时固定）

    forward 流程:
      x_int8 = round(x / act_scale).clamp(-128, 127)
      y_int32 = torch._int_mm(x_int8, W_int8.T)
      y = y_int32 * (act_scale * weight_scale) + bias
    """

    _INT_MM_MIN_ROWS = 17  # CUTLASS kernel 要求 M >= 17

    def __init__(self, in_features, out_features,
                 weight_int8, weight_scale, act_scale, bias):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.register_buffer('weight_int8', weight_int8)   # (Co, Ci), int8
        self.register_buffer('weight_scale', weight_scale) # (Co,), float32
        self.register_buffer('act_scale', torch.tensor(act_scale, dtype=torch.float32))
        self.register_buffer(
            'output_scale',
            (act_scale * weight_scale).to(torch.float32)   # (Co,), 预计算
        )
        if bias is not None:
            self.register_buffer('bias', bias)
        else:
            self.bias = None

    def forward(self, x):
        orig_shape = x.shape
        orig_dtype = x.dtype

        x_2d = x.reshape(-1, x.shape[-1]) if x.dim() > 2 else x
        M = x_2d.shape[0]

        # 量化激活
        x_int8 = (x_2d.float() / self.act_scale.item()).round().clamp(-128, 127).to(torch.int8)

        # padding（torch._int_mm 要求 M >= 17）
        need_pad = M < self._INT_MM_MIN_ROWS
        if need_pad:
            x_int8 = F.pad(x_int8, (0, 0, 0, self._INT_MM_MIN_ROWS - M))

        # 真 INT8 矩阵乘法
        y_int32 = torch._int_mm(x_int8, self.weight_int8.t())

        if need_pad:
            y_int32 = y_int32[:M, :]

        # 还原到浮点
        y = y_int32.float() * self.output_scale.unsqueeze(0)

        if self.bias is not None:
            y = y + self.bias.float().unsqueeze(0)

        if len(orig_shape) > 2:
            y = y.reshape(*orig_shape[:-1], self.out_features)

        return y.to(orig_dtype)

    @classmethod
    def from_linear(cls, linear_module, act_scale):
        """
        从 nn.Linear + 激活 scale 直接构造 StaticInt8Linear。

        act_scale: 校准时收集的 per-tensor scale（一个 float）
        权重 scale: per-channel，从权重本身计算
        """
        weight_float = linear_module.weight.data.float()

        # per-channel 权重 scale
        weight_scale = weight_float.abs().max(dim=1)[0].clamp(min=1e-10) / 127.0

        # 量化权重
        weight_int8 = (
            (weight_float / weight_scale.unsqueeze(1))
            .round().clamp(-128, 127).to(torch.int8)
        )

        bias = linear_module.bias.data.clone() if linear_module.bias is not None else None

        return cls(
            in_features=linear_module.in_features,
            out_features=linear_module.out_features,
            weight_int8=weight_int8,
            weight_scale=weight_scale,
            act_scale=act_scale,
            bias=bias,
        )


# ============================================================
# Part 5: 转换
# ============================================================

def _replace_module(model, name, new_module):
    parts = name.rsplit('.', 1)
    if len(parts) == 2:
        parent = dict(model.named_modules())[parts[0]]
        setattr(parent, parts[1], new_module)
    else:
        setattr(model, parts[0], new_module)


def convert_to_int8(model, act_minmax):
    """
    校准完成后，把所有目标 nn.Linear 转换为 StaticInt8Linear。

    act_minmax: attach_minmax_hooks 收集到的 dict
                key=层名, value={'min': float, 'max': float}

    没有收集到数据的层（n_observed=0）跳过，保持 FP16。
    """
    converted = 0
    skipped = 0

    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if not should_quantize(name):
            continue
        if name not in act_minmax:
            print(f"  [skip] {name}: 没有校准数据，保持 FP16")
            skipped += 1
            continue

        stats = act_minmax[name]
        abs_max = max(abs(stats['min']), abs(stats['max']))
        act_scale = max(abs_max / 127.0, 1e-10)

        int8_layer = StaticInt8Linear.from_linear(module, act_scale)
        _replace_module(model, name, int8_layer)
        converted += 1

    print(f"[int8] 转换完成: {converted} 层 → StaticInt8Linear, {skipped} 层跳过（无数据）")
    return converted


# ============================================================
# Part 6: 统计和验证
# ============================================================

def print_model_stats(model):
    counts = {'nn.Linear': 0, 'StaticInt8Linear': 0}
    for _, module in model.named_modules():
        if isinstance(module, StaticInt8Linear):
            counts['StaticInt8Linear'] += 1
        elif isinstance(module, nn.Linear):
            counts['nn.Linear'] += 1
    print(f"  StaticInt8Linear (真INT8): {counts['StaticInt8Linear']}")
    print(f"  nn.Linear        (FP16):   {counts['nn.Linear']}")


def verify_int8_model(model, num_show=3):
    print(f"\n  === INT8 权重检查 (前 {num_show} 层) ===")
    count = 0
    for name, module in model.named_modules():
        if not isinstance(module, StaticInt8Linear):
            continue
        if count >= num_show:
            break
        w = module.weight_int8
        print(f"  [{count+1}] {name}")
        print(f"      shape={list(w.shape)}, dtype={w.dtype}")
        print(f"      权重范围: [{w.min().item()}, {w.max().item()}]")
        print(f"      act_scale={module.act_scale.item():.6f}")
        print(f"      weight_scale: [{module.weight_scale.min().item():.6f}, {module.weight_scale.max().item():.6f}]")
        count += 1


def check_int_mm():
    if not hasattr(torch, '_int_mm'):
        print(f"⚠ torch._int_mm 不可用，需要 PyTorch >= 2.3（当前 {torch.__version__}）")
        return False
    try:
        a = torch.randint(-128, 127, (32, 16), dtype=torch.int8, device='cuda')
        b = torch.randint(-128, 127, (16, 8), dtype=torch.int8, device='cuda')
        c = torch._int_mm(a, b)
        assert c.dtype == torch.int32
        return True
    except Exception as e:
        print(f"⚠ torch._int_mm 测试失败: {e}")
        return False