"""
真 INT8 静态量化推理模块 (VLA-Adapter)
======================================

★ 重写版: 不依赖 torchao 高级 API, 用纯 PyTorch 实现 ★

背景:
  torchao 0.4.0 没有暴露 Observer / 静态量化的底层 API
  (AffineQuantizedMinMaxObserver, PerAxis 等是 0.7+ 才有的)。
  但 PyTorch 2.4 自带 torch._int_mm() (真 INT8 矩阵乘法),
  所以我们自己实现 Observer 和量化逻辑, 完全不依赖 torchao。

和其他脚本的区别:
  - smoothquant_vla.py        → fake quant (FP16模拟INT8误差, 在 vla-adapter 环境跑)
  - export_int8_weights.py    → 导出 .npz 给 NeuroSim/FPGA (不需要 torchao)
  - 本脚本 (int8_vla)         → GPU 上跑真 INT8 推理 (在 int8-eval 环境跑)

流程:
  1. 加载 FP16 模型
  2. Smooth Qwen2.5 (用已有的 act_scales, alpha=0.5)
  3. 替换目标 Linear → ObservedLinear (带 observer)
  4. 跑校准数据 (observer 收集每层输入激活的 min/max)
  5. 冻结 scale, 转换为 StaticInt8Linear (真 INT8 权重 + 固定激活 scale)
  6. 推理时: 激活量化→INT8, 调用 torch._int_mm 做 INT8×INT8→INT32, 再还原

量化配置 (和 SmoothQuant 论文一致):
  - 权重: Symmetric INT8, per-channel (每个 output channel 一个 scale)
  - 激活: Symmetric INT8, per-tensor (每层一个 scale, 静态, 不是动态)

依赖:
  只需要 PyTorch >= 2.4 (有 torch._int_mm)
  不需要 torchao
"""

import os
import sys
import argparse
import time
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 配置
# ============================================================
DEFAULT_MODEL_PATH = "outputs/LIBERO-Object-Pro"
DEFAULT_ACT_SCALES = "act_scales/vla_adapter_object.pt"
DEFAULT_OUTPUT_PATH = "int8_models/vla_int8_static.pt"
DEFAULT_ALPHA = 0.5
DEFAULT_CALIB_SAMPLES = 500


# ============================================================
# 检查 torch._int_mm 是否可用
# ============================================================
def check_int_mm():
    """
    检查 torch._int_mm 是否可用。
    torch._int_mm 是 PyTorch 2.3+ 引入的真 INT8 矩阵乘法。

    签名: torch._int_mm(A: int8[M,K], B: int8[K,N]) → int32[M,N]

    限制: M 必须 >= 17 (CUTLASS kernel 的要求)。
    对于 batch=1 的推理, 需要 padding 到 17 行再算。

    注意: 这是个 "private" API (带下划线), 但在 PyTorch 社区广泛使用,
    torchao 底层也是调用这个函数。
    """
    if not hasattr(torch, '_int_mm'):
        print("⚠ torch._int_mm 不可用 (需要 PyTorch >= 2.3)")
        print("  当前 PyTorch 版本:", torch.__version__)
        return False

    # 快速测试 (注意: M 必须 >= 17, 这是 CUTLASS kernel 的限制)
    try:
        a = torch.randint(-128, 127, (32, 16), dtype=torch.int8, device='cuda')
        b = torch.randint(-128, 127, (16, 8), dtype=torch.int8, device='cuda')
        c = torch._int_mm(a, b)
        assert c.dtype == torch.int32
        assert c.shape == (32, 8)
        return True
    except Exception as e:
        print(f"⚠ torch._int_mm 测试失败: {e}")
        return False


# ============================================================
# Part 1: Smooth Qwen2.5
# ============================================================
@torch.no_grad()
def smooth_ln_fcs(ln, fcs, act_scales, alpha=0.5):
    """
    对一组 RMSNorm + Linear 做 smooth。

    数学公式:
        s = act_scales^alpha / weight_scales^(1-alpha)
        ln.weight /= s
        fc.weight *= s

    效果: 把激活的 outlier 转移到权重上, 让两边都更容易量化。
    """
    if not isinstance(fcs, list):
        fcs = [fcs]

    device = ln.weight.device
    dtype = ln.weight.dtype
    act_scales = act_scales.to(device=device, dtype=dtype)

    # 权重 scales: 所有下游 Linear 权重每列绝对值最大值
    weight_scales = torch.cat(
        [fc.weight.abs().max(dim=0, keepdim=True)[0] for fc in fcs],
        dim=0
    )
    weight_scales = weight_scales.max(dim=0)[0].clamp(min=1e-5)

    # smooth 因子
    scales = (
        (act_scales.pow(alpha) / weight_scales.pow(1 - alpha))
        .clamp(min=1e-5)
        .to(device)
        .to(dtype)
    )

    # 修改 RMSNorm
    ln.weight.div_(scales)
    if hasattr(ln, 'bias') and ln.bias is not None:
        ln.bias.div_(scales)

    # 修改 Linear
    for fc in fcs:
        fc.weight.mul_(scales.view(1, -1))


@torch.no_grad()
def smooth_qwen(model, act_scales, alpha=0.5):
    """
    对 Qwen2.5 的所有 decoder layer 做 smooth。

    每层有两组配对:
      1. input_layernorm       → q_proj, k_proj, v_proj
      2. post_attention_layernorm → gate_proj, up_proj
    """
    print(f"[Step 1] Smooth Qwen2.5 (alpha={alpha})")

    # 找到 Qwen2.5 的 decoder layers
    qwen_layers = None
    possible_paths = [
        lambda m: m.llm_backbone.llm.model.layers,
        lambda m: m.language_model.model.layers,
        lambda m: m.model.layers,
    ]

    for path_fn in possible_paths:
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

        # 配对 1: input_layernorm → q/k/v
        q_key = None
        for key in act_scales:
            if f"layers.{i}.self_attn.q_proj" in key:
                q_key = key
                break

        if q_key is not None:
            smooth_ln_fcs(
                layer.input_layernorm,
                [attn.q_proj, attn.k_proj, attn.v_proj],
                act_scales[q_key],
                alpha=alpha
            )
            smooth_count += 1

        # 配对 2: post_attention_layernorm → gate/up
        gate_key = None
        for key in act_scales:
            if f"layers.{i}.mlp.gate_proj" in key:
                gate_key = key
                break

        if gate_key is not None:
            smooth_ln_fcs(
                layer.post_attention_layernorm,
                [layer.mlp.gate_proj, layer.mlp.up_proj],
                act_scales[gate_key],
                alpha=alpha
            )
            smooth_count += 1

    print(f"  Smooth 完成: {smooth_count} 组 LN-Linear 被修改")
    return smooth_count


# ============================================================
# Part 2: 模型加载
# ============================================================
def load_vla_model(model_path):
    """加载 VLA-Adapter 模型。"""
    print(f"[Step 0] 加载模型: {model_path}")
    try:
        from prismatic import load as load_vla
        model = load_vla(model_path)
        model.eval()
        print(f"  用 prismatic.load() 加载成功")
        return model
    except ImportError:
        pass

    try:
        from transformers import AutoModelForVision2Seq
        model = AutoModelForVision2Seq.from_pretrained(
            model_path, torch_dtype=torch.float16, device_map="cpu"
        )
        model.eval()
        print(f"  用 AutoModelForVision2Seq 加载成功")
        return model
    except Exception:
        pass

    raise FileNotFoundError(f"无法加载模型: {model_path}")


# ============================================================
# Part 3: Observer (自己实现, 不依赖 torchao)
# ============================================================

class MinMaxObserver:
    """
    最简单的 observer: 记录 tensor 的全局 min 和 max。
    校准结束后, 用 min/max 算出量化的 scale。

    Symmetric INT8 的 scale 计算:
      scale = max(|min|, |max|) / 127
      zero_point = 0

    这和 SmoothQuant 论文里的做法一致:
    对称量化, 不需要 zero_point。
    """

    def __init__(self):
        self.min_val = None
        self.max_val = None
        self.n_observed = 0

    def observe(self, x):
        """记录一个 tensor 的 min/max。"""
        with torch.no_grad():
            flat = x.detach().float()
            batch_min = flat.min().item()
            batch_max = flat.max().item()

            if self.min_val is None:
                self.min_val = batch_min
                self.max_val = batch_max
            else:
                self.min_val = min(self.min_val, batch_min)
                self.max_val = max(self.max_val, batch_max)

            self.n_observed += 1

    def get_scale(self):
        """计算 symmetric INT8 的 scale。"""
        if self.min_val is None:
            raise RuntimeError("Observer 没有数据! 请先跑校准。")
        abs_max = max(abs(self.min_val), abs(self.max_val))
        scale = abs_max / 127.0
        return max(scale, 1e-10)  # 防止 scale=0


class PerChannelMinMaxObserver:
    """
    Per-channel observer: 每个 output channel 一个 min/max。
    用于权重量化。

    对 shape=(Co, Ci) 的权重, 记录每行的 min/max。
    """

    def __init__(self):
        self.row_abs_max = None  # shape=(Co,)

    def observe(self, weight):
        """记录权重每行的最大绝对值。权重不变, 只需要调用一次。"""
        with torch.no_grad():
            # shape=(Co, Ci) → 每行取 abs max
            self.row_abs_max = weight.detach().float().abs().max(dim=1)[0]

    def get_scales(self):
        """返回 per-channel scales, shape=(Co,)。"""
        if self.row_abs_max is None:
            raise RuntimeError("Observer 没有数据!")
        return (self.row_abs_max / 127.0).clamp(min=1e-10)


# ============================================================
# Part 4: ObservedLinear (校准阶段用)
# ============================================================

class ObservedLinear(nn.Module):
    """
    带 observer 的 Linear, 用于校准阶段。

    和普通 nn.Linear 一样做 FP16 计算, 但额外记录:
      - 输入激活的全局 min/max (per-tensor)
      - 权重的 per-channel min/max (只算一次)

    校准完成后, 通过 StaticInt8Linear.from_observed() 转换为真 INT8。
    """

    def __init__(self, in_features, out_features, weight, bias):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(weight, requires_grad=False)
        self.bias = nn.Parameter(bias, requires_grad=False) if bias is not None else None

        # Observer
        self.act_observer = MinMaxObserver()
        self.weight_observer = PerChannelMinMaxObserver()

        # 权重不变, 只需要 observe 一次
        self.weight_observer.observe(self.weight.data)

    def forward(self, x):
        # 记录激活统计 (每次 forward 都更新)
        self.act_observer.observe(x)
        # 计算还是 FP16 (校准阶段不改精度)
        return F.linear(x, self.weight, self.bias)

    @classmethod
    def from_float(cls, linear_module):
        """从普通 nn.Linear 创建 ObservedLinear。"""
        return cls(
            in_features=linear_module.in_features,
            out_features=linear_module.out_features,
            weight=linear_module.weight.data.clone(),
            bias=linear_module.bias.data.clone() if linear_module.bias is not None else None,
        )


# ============================================================
# Part 5: StaticInt8Linear (推理阶段用)
# ============================================================

class StaticInt8Linear(nn.Module):
    """
    真 INT8 Linear, 用于推理。

    内部存储:
      - weight_int8: (Co, Ci), dtype=int8, 真正的整数权重
      - weight_scale: (Co,), dtype=float32, 每行一个 scale
      - act_scale: scalar float, 输入激活的量化 scale
      - bias: (Co,), dtype=float16/float32 (如果有)

    推理计算流程:
      1. 输入 x (FP16, shape=...,Ci)
      2. x_int8 = round(x / act_scale), clamp to [-128,127]
      3. y_int32 = torch._int_mm(x_int8, weight_int8.T)   ← 真 INT8 kernel
      4. y_fp = y_int32 * (act_scale * weight_scale)       ← 还原到浮点
      5. y_fp += bias

    为什么需要 (act_scale * weight_scale)?
      x ≈ x_int8 * act_scale
      W ≈ W_int8 * weight_scale
      x @ W.T ≈ (x_int8 * act_scale) @ (W_int8 * weight_scale).T
             = act_scale * weight_scale * (x_int8 @ W_int8.T)
    """

    def __init__(self, in_features, out_features,
                 weight_int8, weight_scale, act_scale, bias):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # 注册为 buffer (不参与梯度, 但会被 state_dict 保存)
        self.register_buffer('weight_int8', weight_int8)      # (Co, Ci), int8
        self.register_buffer('weight_scale', weight_scale)    # (Co,), float32
        self.register_buffer('act_scale', torch.tensor(act_scale, dtype=torch.float32))
        if bias is not None:
            self.register_buffer('bias', bias)
        else:
            self.bias = None

        # 预计算还原系数: act_scale * weight_scale, shape=(Co,)
        # 推理时直接乘, 省一次计算
        self.register_buffer(
            'output_scale',
            (act_scale * weight_scale).to(torch.float32)  # (Co,)
        )

    # torch._int_mm 要求 M >= 17 (CUTLASS kernel 的限制)
    # 推理时 batch_size=1 很常见 (M=1), 所以需要 padding
    _INT_MM_MIN_ROWS = 17

    def forward(self, x):
        """
        真 INT8 推理 forward。

        关键限制: torch._int_mm(A, B) 要求 A 的行数 >= 17。
        VLA 推理时经常 batch=1 (M=1), 所以用 zero-padding 到 17 行,
        算完再切回来。padding 的行全是 0, 不影响结果。

        这是 PyTorch 社区的标准做法, torchao 内部也是这样处理的。
        """
        orig_shape = x.shape
        orig_dtype = x.dtype

        # 展平到 2D: (..., Ci) → (M, Ci)
        if x.dim() > 2:
            x_2d = x.reshape(-1, x.shape[-1])
        else:
            x_2d = x

        M = x_2d.shape[0]

        # Step 1: 量化激活 → INT8
        x_float = x_2d.float()
        act_scale_val = self.act_scale.item()
        x_int8 = (x_float / act_scale_val).round().clamp(-128, 127).to(torch.int8)

        # Step 2: Padding (如果 M < 17)
        need_pad = M < self._INT_MM_MIN_ROWS
        if need_pad:
            pad_rows = self._INT_MM_MIN_ROWS - M
            x_int8 = torch.nn.functional.pad(x_int8, (0, 0, 0, pad_rows))
            # pad 后 x_int8.shape = (17, Ci), 多出来的行全是 0

        # Step 3: INT8 × INT8 → INT32
        # torch._int_mm(A, B): A=(M,K) int8, B=(K,N) int8 → C=(M,N) int32
        # 我们的权重是 (Co, Ci), 需要转置: W.T = (Ci, Co)
        y_int32 = torch._int_mm(x_int8, self.weight_int8.t())

        # Step 4: 去掉 padding
        if need_pad:
            y_int32 = y_int32[:M, :]

        # Step 5: 还原到浮点
        # y_int32 的每列 j 需要乘以 output_scale[j]
        y_float = y_int32.float() * self.output_scale.unsqueeze(0)

        # Step 6: 加 bias
        if self.bias is not None:
            y_float = y_float + self.bias.float().unsqueeze(0)

        # 恢复形状和精度
        if len(orig_shape) > 2:
            y_float = y_float.reshape(*orig_shape[:-1], self.out_features)

        return y_float.to(orig_dtype)

    @classmethod
    def from_observed(cls, observed_module):
        """
        从 ObservedLinear 转换:
          1. 从 observer 取出 scale
          2. 把 FP16 权重转成真 INT8
        """
        # 激活 scale (per-tensor, 一个数)
        act_scale = observed_module.act_observer.get_scale()

        # 权重 scale (per-channel)
        weight_scale = observed_module.weight_observer.get_scales()  # (Co,)

        # 量化权重: W_int8 = round(W / scale), clamp
        weight_float = observed_module.weight.data.float()
        weight_int8 = (
            (weight_float / weight_scale.unsqueeze(1))
            .round()
            .clamp(-128, 127)
            .to(torch.int8)
        )

        return cls(
            in_features=observed_module.in_features,
            out_features=observed_module.out_features,
            weight_int8=weight_int8,
            weight_scale=weight_scale,
            act_scale=act_scale,
            bias=observed_module.bias.data.clone() if observed_module.bias is not None else None,
        )


# ============================================================
# Part 6: 层替换逻辑
# ============================================================

def should_quantize(name):
    """
    判断一个 nn.Linear 是否应该被量化。

    量化: ViT (DINOv2, SigLIP) + Projector + Qwen2.5 的 Linear
    不量化: Policy Head, lm_head, Embedding
    """
    nl = name.lower()

    # 跳过 policy head
    if any(kw in nl for kw in ['policy', 'bridge', 'action_head']):
        return False

    # 跳过 lm_head
    if 'lm_head' in nl:
        return False

    # 跳过 embedding
    if 'embed' in nl and 'proj' not in nl:
        return False

    return True


def _replace_module(model, name, new_module):
    """替换模型中指定名称的子模块。"""
    parts = name.rsplit('.', 1)
    if len(parts) == 2:
        parent_name, child_name = parts
        parent = dict(model.named_modules())[parent_name]
    else:
        child_name = parts[0]
        parent = model
    setattr(parent, child_name, new_module)


def replace_with_observed(model):
    """
    把模型中所有目标 nn.Linear 替换为 ObservedLinear。
    返回: 被替换的层数。
    """
    count = 0
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if not should_quantize(name):
            continue

        observed = ObservedLinear.from_float(module)
        _replace_module(model, name, observed)
        count += 1

    return count


def convert_observed_to_quantized(model):
    """
    校准完成后, 把所有 ObservedLinear 转换为 StaticInt8Linear。
    返回: 转换的层数。
    """
    count = 0
    for name, module in list(model.named_modules()):
        if not isinstance(module, ObservedLinear):
            continue

        quantized = StaticInt8Linear.from_observed(module)
        _replace_module(model, name, quantized)
        count += 1

    return count


# ============================================================
# Part 7: 校准
# ============================================================

def calibrate_with_libero(model, num_episodes=10, task_suite_name="libero_object",
                          seed=0, device="cuda:0"):
    """
    用 LIBERO 环境跑几个 episode 做校准。

    做的事: 正常 eval 循环, ObservedLinear 的 forward 自动收集激活范围。
    """
    print(f"\n[校准] 跑 {num_episodes} 个 episode")

    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[task_suite_name]()
    num_tasks = task_suite.get_num_tasks()

    model = model.to(device)
    model.eval()

    total_steps = 0
    episodes_per_task = max(1, num_episodes // num_tasks)
    remaining = num_episodes
    t0 = time.time()

    for task_id in range(num_tasks):
        if remaining <= 0:
            break

        task = task_suite.get_task(task_id)
        task_description = task.language

        env_args = {
            "bddl_file_name": os.path.join(
                task_suite.get_task_bddl_file_path(),
                task.problem_folder,
                task.bddl_file,
            ),
            "camera_heights": 256,
            "camera_widths": 256,
        }

        env = OffScreenRenderEnv(**env_args)
        env.seed(seed)

        n_ep = min(episodes_per_task, remaining)
        for ep in range(n_ep):
            obs = env.reset()
            done = False
            step = 0

            while not done and step < 300:
                try:
                    image = obs["agentview_image"]
                    with torch.no_grad():
                        action = model.predict_action(image, task_description)
                    if isinstance(action, torch.Tensor):
                        action = action.cpu().numpy()
                    obs, reward, done, info = env.step(action)
                    step += 1
                    total_steps += 1
                except Exception as e:
                    print(f"  ⚠ Task {task_id} Ep {ep} Step {step}: {e}")
                    break

            remaining -= 1

        env.close()

        if (task_id + 1) % 2 == 0:
            print(f"  进度: {num_episodes - remaining}/{num_episodes} ep, "
                  f"{total_steps} steps, {time.time()-t0:.0f}s")

    print(f"  校准完成: {num_episodes - remaining} ep, {total_steps} steps, "
          f"{time.time()-t0:.0f}s")
    return total_steps


def calibrate_with_random(model, num_batches=50, device="cuda:0"):
    """
    用随机数据做校准 (当 LIBERO 不可用时的 fallback)。

    ⚠ 只能验证代码流程, 不能用于正式实验!
    """
    print(f"\n[校准-fallback] 用随机数据 ({num_batches} batches)")
    print("  ⚠ 这只是 fallback, 正式实验请用 LIBERO 真实数据!")

    model = model.to(device)
    model.eval()

    count = 0
    for name, module in model.named_modules():
        if isinstance(module, ObservedLinear):
            in_feat = module.in_features
            for _ in range(num_batches):
                fake_input = torch.randn(1, in_feat, device=device, dtype=module.weight.dtype)
                with torch.no_grad():
                    module(fake_input)
            count += 1

    print(f"  校准完成: {count} 层, 每层 {num_batches} 个随机 batch")
    return count


# ============================================================
# Part 8: 保存和加载
# ============================================================

def save_quantized_model(model, path):
    """
    保存量化后的模型 state_dict。

    StaticInt8Linear 的 weight_int8 (int8), weight_scale, act_scale, output_scale
    都注册为 buffer, torch.save 可以直接处理。
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(model.state_dict(), path)
    size_mb = os.path.getsize(path) / (1024 * 1024)
    print(f"  保存到: {path} ({size_mb:.1f} MB)")


def load_quantized_model(model_path, quantized_state_path, act_scales_path, alpha=0.5):
    """
    加载已量化的模型。

    流程:
      1. 加载 FP16 模型骨架
      2. Smooth Qwen2.5
      3. 替换为 ObservedLinear (占位)
      4. 转换为 StaticInt8Linear (占位)
      5. 加载量化后的 state_dict

    需要先创建和量化时完全相同的模型结构, 才能加载 state_dict。
    """
    model = load_vla_model(model_path)
    act_scales = torch.load(act_scales_path, map_location="cpu")
    smooth_qwen(model, act_scales, alpha=alpha)
    replace_with_observed(model)
    convert_observed_to_quantized(model)
    state_dict = torch.load(quantized_state_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=False)
    return model


# ============================================================
# Part 9: 统计和验证
# ============================================================

def print_model_stats(model):
    """打印模型中各类型模块的统计。"""
    counts = {"nn.Linear": 0, "ObservedLinear": 0, "StaticInt8Linear": 0}
    for name, module in model.named_modules():
        if isinstance(module, StaticInt8Linear):
            counts["StaticInt8Linear"] += 1
        elif isinstance(module, ObservedLinear):
            counts["ObservedLinear"] += 1
        elif isinstance(module, nn.Linear):
            counts["nn.Linear"] += 1

    print(f"\n  === 模型统计 ===")
    print(f"  StaticInt8Linear (真INT8): {counts['StaticInt8Linear']}")
    print(f"  ObservedLinear   (校准中): {counts['ObservedLinear']}")
    print(f"  nn.Linear        (FP16):   {counts['nn.Linear']}")


def verify_observer_stats(model, num_show=5):
    """检查 observer 是否收集到了数据。"""
    print(f"\n  === Observer 统计 (前 {num_show} 层) ===")
    count = 0
    for name, module in model.named_modules():
        if not isinstance(module, ObservedLinear):
            continue
        if count >= num_show:
            break

        act_scale = module.act_observer.get_scale()
        w_scales = module.weight_observer.get_scales()

        print(f"\n  [{count+1}] {name}")
        print(f"      激活 scale: {act_scale:.6f} "
              f"(range: [{module.act_observer.min_val:.4f}, {module.act_observer.max_val:.4f}])")
        print(f"      权重 scale: min={w_scales.min().item():.6f}, "
              f"max={w_scales.max().item():.6f}, n_observed={module.act_observer.n_observed}")
        count += 1

    if count == 0:
        print("  ❌ 没有 ObservedLinear! 请先调用 replace_with_observed()")


def verify_int8_model(model, num_show=3):
    """检查 StaticInt8Linear 的量化结果。"""
    print(f"\n  === INT8 权重检查 (前 {num_show} 层) ===")
    count = 0
    for name, module in model.named_modules():
        if not isinstance(module, StaticInt8Linear):
            continue
        if count >= num_show:
            break

        w = module.weight_int8
        print(f"\n  [{count+1}] {name}")
        print(f"      shape: {list(w.shape)}, dtype: {w.dtype}")
        print(f"      权重范围: [{w.min().item()}, {w.max().item()}]")
        print(f"      act_scale: {module.act_scale.item():.6f}")
        print(f"      weight_scale: [{module.weight_scale.min().item():.6f}, "
              f"{module.weight_scale.max().item():.6f}]")
        count += 1


# ============================================================
# Part 10: 一键流程
# ============================================================

@torch.no_grad()
def full_pipeline(model_path, act_scales_path, output_path,
                  alpha=0.5, calib_episodes=10, device="cuda:0"):
    """
    完整流程: 加载 → smooth → 插入observer → 校准 → 转INT8 → 保存。
    """
    print("=" * 60)
    print("  真 INT8 静态量化 (纯 PyTorch, 无 torchao 依赖)")
    print("=" * 60)
    print(f"  模型:      {model_path}")
    print(f"  act_scales: {act_scales_path}")
    print(f"  输出:      {output_path}")
    print(f"  alpha:     {alpha}")
    print(f"  校准 ep:   {calib_episodes}")
    print(f"  PyTorch:   {torch.__version__}")
    print(f"  _int_mm:   {'可用' if check_int_mm() else '不可用'}")
    print("=" * 60)

    # Step 0: 加载
    model = load_vla_model(model_path)
    act_scales = torch.load(act_scales_path, map_location="cpu")
    print(f"  加载了 {len(act_scales)} 个 act_scales key")

    # Step 1: Smooth Qwen2.5
    smooth_qwen(model, act_scales, alpha=alpha)

    # Step 2: 插入 observer
    print(f"\n[Step 2] 插入 observer")
    n_observed = replace_with_observed(model)
    print(f"  替换了 {n_observed} 个 Linear → ObservedLinear")
    print_model_stats(model)

    # Step 3: 校准
    try:
        calibrate_with_libero(model, num_episodes=calib_episodes, device=device)
    except ImportError:
        print("  ⚠ LIBERO 未安装, 用随机数据 fallback")
        calibrate_with_random(model, device=device)

    verify_observer_stats(model)

    # Step 4: 转换为真 INT8
    print(f"\n[Step 4] 转换为真 INT8")
    model = model.cpu()
    n_quantized = convert_observed_to_quantized(model)
    print(f"  转换了 {n_quantized} 个 ObservedLinear → StaticInt8Linear")
    print_model_stats(model)
    verify_int8_model(model)

    # Step 5: 保存
    print(f"\n[Step 5] 保存量化模型")
    save_quantized_model(model, output_path)

    print(f"\n{'='*60}")
    print(f"  Done!")
    print(f"  下一步: python eval_int8.py --quantized-model {output_path}")
    print(f"{'='*60}")

    return model


# ============================================================
# Part 11: 集成指南
# ============================================================

INTEGRATION_GUIDE = """
============================================================
★ 推荐: 集成到你的 eval 脚本里 ★
============================================================

在 run_libero_eval 里加入以下代码:

```python
import torch
from int8_vla import (
    smooth_qwen,
    replace_with_observed,
    convert_observed_to_quantized,
    verify_observer_stats,
    verify_int8_model,
)

# ---------- 模型加载后, eval 之前 ----------

# 1. Smooth Qwen2.5
act_scales = torch.load("act_scales/vla_adapter_object.pt", map_location="cpu")
smooth_qwen(model, act_scales, alpha=0.5)

# 2. 插入 observer
n = replace_with_observed(model)
print(f"[int8] 插入 {n} 个 observer")

# 3. 校准阶段: 跑前 N 个 episode, observer 自动收集激活统计
CALIBRATION_EPISODES = 10
for ep in range(CALIBRATION_EPISODES):
    # ... 正常跑 eval, 但不记录结果 ...
    pass

# 4. 检查 observer
verify_observer_stats(model, num_show=3)

# 5. 转换为真 INT8
n = convert_observed_to_quantized(model)
print(f"[int8] 转换了 {n} 层为真 INT8")
verify_int8_model(model, num_show=3)

# 6. 正式 eval
# ---- 你的 eval 循环, 记录成功率 ----
```

好处:
  - 校准数据就是 eval 数据, 分布一致
  - 不需要单独的校准脚本
  - 不需要保存/加载中间文件
============================================================
"""


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="真 INT8 静态量化 (VLA-Adapter, 纯 PyTorch)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=INTEGRATION_GUIDE,
    )
    parser.add_argument("--mode", type=str, default="calibrate",
                        choices=["calibrate", "guide", "check"],
                        help="calibrate=完整流程, guide=打印集成指南, check=检查环境")
    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--act-scales", type=str, default=DEFAULT_ACT_SCALES)
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--calib-episodes", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    if args.mode == "check":
        print(f"PyTorch: {torch.__version__}")
        print(f"CUDA:    {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
        print(f"_int_mm: {'✓ 可用' if check_int_mm() else '✗ 不可用'}")
        return

    if args.mode == "guide":
        print(INTEGRATION_GUIDE)
        return

    if args.mode == "calibrate":
        full_pipeline(
            model_path=args.model_path,
            act_scales_path=args.act_scales,
            output_path=args.output,
            alpha=args.alpha,
            calib_episodes=args.calib_episodes,
            device=args.device,
        )


if __name__ == "__main__":
    main()