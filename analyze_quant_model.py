"""
analyze_quant_model.py
======================
直接从 state_dict 分析量化模型精度，无需加载完整模型。

用法：
    python analyze_quant_model.py \
        --pt_path int8_models/vla_int8_static.pt \
        --minmax_path int8_models/vla_int8_static.pt.minmax.pt
"""

import argparse
import os
import torch
from collections import defaultdict


# ──────────────────────────────────────────────────────────────
# 工具
# ──────────────────────────────────────────────────────────────

def dtype_str(dtype):
    return {
        torch.float32: "FP32", torch.float16: "FP16",
        torch.bfloat16: "BF16", torch.int8: "INT8", torch.int32: "INT32",
    }.get(dtype, str(dtype))


def precision_of(key, sd):
    t = sd.get(key)
    if t is None:
        return "NOT FOUND"
    return dtype_str(t.dtype)


def shape_of(key, sd):
    t = sd.get(key)
    if t is None:
        return ""
    return str(list(t.shape))


def find_act_scale(layer_path, minmax):
    if layer_path in minmax:
        v = minmax[layer_path]
        return max(abs(v['min']), abs(v['max'])) / 127.0, layer_path
    for k, v in minmax.items():
        if layer_path in k or k in layer_path:
            return max(abs(v['min']), abs(v['max'])) / 127.0, k
    return None, None


def weight_key(base, sd):
    k_int8 = base + ".weight_int8"
    k_fp   = base + ".weight"
    if k_int8 in sd:
        return k_int8, "INT8(真)"
    if k_fp in sd:
        return k_fp, dtype_str(sd[k_fp].dtype)
    return None, "NOT FOUND"


def flag(scale):
    if scale is None:
        return ""
    if scale > 5:
        return "  ⚠ outlier大"
    if scale > 2:
        return "  △ 偏大"
    return ""


def print_row(idx, comp, prec, act_s, shape, note=""):
    act_s_str = f"{act_s:.4f}" if act_s is not None else "   —  "
    print(f"  {str(idx):>5}  {comp:<26} {prec:<14} {act_s_str:>10}  {shape:<22}  {note}")


# ──────────────────────────────────────────────────────────────
# Qwen2.5
# ──────────────────────────────────────────────────────────────

def find_qwen_prefix(sd):
    for prefix in [
        "llm_backbone.llm.model.layers",
        "language_model.model.layers",
        "model.layers",
    ]:
        if any(k.startswith(prefix) for k in sd):
            return prefix
    return None


def analyze_qwen(sd, minmax):
    prefix = find_qwen_prefix(sd)
    if prefix is None:
        print("  ⚠ 未找到 Qwen decoder layers")
        return

    layer_indices = set()
    for k in sd:
        if k.startswith(prefix + "."):
            rest = k[len(prefix)+1:]
            idx_str = rest.split(".")[0]
            if idx_str.isdigit():
                layer_indices.add(int(idx_str))

    n_layers = max(layer_indices) + 1
    show = sorted({0, 1, 2, n_layers - 1})
    sep = "─" * 95

    print(f"\n{'═'*95}")
    print(f"  QWEN2.5 DECODER LAYERS  (共 {n_layers} 层，展示前3层 + 最后1层)")
    print(f"{'═'*95}")
    print(f"  {'层':>5}  {'组件':<26} {'权重精度':<14} {'act_scale':>10}  {'权重形状':<22}  备注")
    print(f"  {sep}")

    prev = -1
    for i in show:
        if prev != -1 and i != prev + 1:
            print(f"  {'':>5}  ... (layers {prev+1}–{i-1} 同上，省略)")
        prev = i
        base = f"{prefix}.{i}"

        # input LayerNorm
        ln1_key = f"{base}.input_layernorm.weight"
        print_row(i, "input_layernorm.weight",
                  precision_of(ln1_key, sd), None, shape_of(ln1_key, sd),
                  "γ 已被 smooth 除以 s")

        # Attention projections
        for proj in ["self_attn.q_proj", "self_attn.k_proj",
                     "self_attn.v_proj", "self_attn.o_proj"]:
            full = f"{base}.{proj}"
            wk, prec = weight_key(full, sd)
            shape = shape_of(wk, sd) if wk else ""
            act_s, _ = find_act_scale(full, minmax)
            print_row(i, proj, prec, act_s, shape, "smooth 后量化" + flag(act_s))

        # BMM / Softmax（融合在 SDPA 里，state_dict 里没有）
        print_row(i, "attn.BMM(QK^T)",    "BF16(SDPA内)", None, "N/A",
                  "scaled_dot_product_attention，FP路径")
        print_row(i, "attn.Softmax",       "BF16(SDPA内)", None, "N/A",
                  "融合在 SDPA 里，FP路径")
        print_row(i, "attn.BMM(Attn·V)",  "BF16(SDPA内)", None, "N/A",
                  "同上")

        # post-attention LayerNorm
        ln2_key = f"{base}.post_attention_layernorm.weight"
        print_row(i, "post_attn_layernorm.weight",
                  precision_of(ln2_key, sd), None, shape_of(ln2_key, sd),
                  "γ 已被 smooth 除以 s")

        # FFN
        for proj in ["mlp.gate_proj", "mlp.up_proj"]:
            full = f"{base}.{proj}"
            wk, prec = weight_key(full, sd)
            act_s, _ = find_act_scale(full, minmax)
            print_row(i, proj, prec, act_s,
                      shape_of(wk, sd) if wk else "", "smooth 后量化" + flag(act_s))

        print_row(i, "mlp.SiLU(gate)×up", "BF16", None, "N/A", "逐元素激活，FP路径")

        full = f"{base}.mlp.down_proj"
        wk, prec = weight_key(full, sd)
        act_s, _ = find_act_scale(full, minmax)
        print_row(i, "mlp.down_proj", prec, act_s,
                  shape_of(wk, sd) if wk else "",
                  "⚡ 未smooth，接收 SiLU 输出" + flag(act_s))

        print(f"  {sep}")


# ──────────────────────────────────────────────────────────────
# ViT（SigLIP / DINOv2）
# ──────────────────────────────────────────────────────────────

def analyze_vit(sd, minmax, blocks_prefix, label, n_show=2):
    block_indices = set()
    for k in sd:
        if k.startswith(blocks_prefix + "."):
            rest = k[len(blocks_prefix)+1:]
            idx_str = rest.split(".")[0]
            if idx_str.isdigit():
                block_indices.add(int(idx_str))
    if not block_indices:
        print(f"  ⚠ 未找到 {label} blocks")
        return

    n_blocks = max(block_indices) + 1
    show = sorted(set(list(range(min(n_show, n_blocks))) + [n_blocks - 1]))
    sep = "─" * 95

    print(f"\n{'═'*95}")
    print(f"  {label}  (共 {n_blocks} 个 block，展示前{n_show}个 + 最后1个)")
    print(f"{'═'*95}")
    print(f"  {'Block':>5}  {'组件':<26} {'权重精度':<14} {'act_scale':>10}  {'权重形状':<22}  备注")
    print(f"  {sep}")

    prev = -1
    for i in show:
        if prev != -1 and i != prev + 1:
            print(f"  {'':>5}  ... (blocks {prev+1}–{i-1} 同上，省略)")
        prev = i
        base = f"{blocks_prefix}.{i}"

        # norm1
        for norm_name in ["norm1", "layer_norm1"]:
            k = f"{base}.{norm_name}.weight"
            if k in sd:
                print_row(i, f"{norm_name}.weight", precision_of(k, sd),
                          None, shape_of(k, sd), "LayerNorm")
                break

        # attention fused qkv
        found = False
        for qkv_path in ["attn.qkv", "attn.in_proj"]:
            full = f"{base}.{qkv_path}"
            wk, prec = weight_key(full, sd)
            if wk:
                act_s, _ = find_act_scale(full, minmax)
                print_row(i, qkv_path, prec, act_s, shape_of(wk, sd),
                          "Naive W8A8" + flag(act_s))
                found = True
                break
        if not found:
            for q_path in ["attn.q", "attn.query", "attn.q_proj"]:
                full = f"{base}.{q_path}"
                wk, prec = weight_key(full, sd)
                if wk:
                    act_s, _ = find_act_scale(full, minmax)
                    print_row(i, q_path, prec, act_s, shape_of(wk, sd),
                              "Naive W8A8" + flag(act_s))
                    break

        for proj_path in ["attn.proj", "attn.out_proj", "attn.projection"]:
            full = f"{base}.{proj_path}"
            wk, prec = weight_key(full, sd)
            if wk:
                act_s, _ = find_act_scale(full, minmax)
                print_row(i, proj_path, prec, act_s, shape_of(wk, sd),
                          "Naive W8A8" + flag(act_s))
                break

        print_row(i, "attn.Softmax", "BF16", None, "N/A", "FP路径")

        # norm2
        for norm_name in ["norm2", "layer_norm2"]:
            k = f"{base}.{norm_name}.weight"
            if k in sd:
                print_row(i, f"{norm_name}.weight", precision_of(k, sd),
                          None, shape_of(k, sd), "LayerNorm")
                break

        # MLP
        for fc_path in ["mlp.fc1", "mlp.fc2", "mlp.gate_proj",
                         "mlp.up_proj", "mlp.down_proj"]:
            full = f"{base}.{fc_path}"
            wk, prec = weight_key(full, sd)
            if wk:
                act_s, _ = find_act_scale(full, minmax)
                print_row(i, fc_path, prec, act_s, shape_of(wk, sd),
                          "Naive W8A8" + flag(act_s))

        print(f"  {sep}")


# ──────────────────────────────────────────────────────────────
# Projector
# ──────────────────────────────────────────────────────────────

def analyze_projector(sd, minmax):
    sep = "─" * 95
    print(f"\n{'═'*95}")
    print(f"  PROJECTOR")
    print(f"{'═'*95}")
    print(f"  {'组件':<40} {'权重精度':<14} {'act_scale':>10}  {'权重形状':<22}  备注")
    print(f"  {sep}")

    found = False
    for k in sorted(sd.keys()):
        if not k.startswith("projector."):
            continue
        if not (k.endswith(".weight_int8") or k.endswith(".weight")):
            continue
        base = k.replace(".weight_int8", "").replace(".weight", "")
        prec = "INT8(真)" if k.endswith(".weight_int8") else dtype_str(sd[k].dtype)
        act_s, _ = find_act_scale(base, minmax)
        comp = base.replace("projector.", "")
        print(f"  {comp:<40} {prec:<14} "
              f"{f'{act_s:.4f}' if act_s else '   —  ':>10}  "
              f"{shape_of(k, sd):<22}  {flag(act_s).strip()}")
        found = True

    if not found:
        print("  ⚠ 未找到 projector 层")


# ──────────────────────────────────────────────────────────────
# 汇总
# ──────────────────────────────────────────────────────────────

def print_summary():
    print(f"\n{'═'*95}")
    print("  对照表（参考论文 Figure 6）")
    print(f"{'═'*95}")
    print("""
  操作                        论文(OPT)       你的模型              说明
  ─────────────────────────── ─────────────── ───────────────────── ────────────────────────────
  LayerNorm γ                 FP16            BF16                  smooth 后 γ 已除以 s
  Q / K / V proj              INT8            INT8(真) ✅            smooth 后量化
  BMM (QK^T)                  INT8            BF16 (SDPA内) ⚠       PyTorch SDPA 融合算子，无法拦截
  Softmax                     FP16            BF16 (SDPA内)          同上
  BMM (Attn·V)                INT8            BF16 (SDPA内) ⚠       同上
  O proj                      INT8            INT8(真) ✅
  post-LN γ                   FP16            BF16
  gate_proj / up_proj         INT8            INT8(真) ✅            smooth 后量化
  SiLU 激活                   FP16            BF16                  逐元素，FP路径
  down_proj                   INT8            INT8(真) ✅            ⚡ 未smooth，act_scale可能大

  ViT (SigLIP / DINOv2):
  QKV (fused)                 INT8            INT8(真) ✅            Naive W8A8，无smooth
  attn.proj                   INT8            INT8(真) ✅
  Softmax                     FP16            BF16
  MLP fc1 / fc2               INT8            INT8(真) ✅

  ⚠ Qwen2.5 使用 scaled_dot_product_attention (SDPA)，BMM 被融合进去无法单独量化。
    Q/K/V 输出 INT8 进入 SDPA 前会先 dequantize 成 BF16 做 attention 计算。
""")


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pt_path",     default="int8_models/vla_int8_static.pt")
    parser.add_argument("--minmax_path", default=None)
    args = parser.parse_args()

    minmax_path = args.minmax_path or (args.pt_path + ".minmax.pt")

    print(f"\n加载 state_dict: {args.pt_path}")
    sd = torch.load(args.pt_path, map_location="cpu")
    if not isinstance(sd, dict):
        print("⚠ 文件不是 state_dict，请确认保存方式")
        return
    print(f"  共 {len(sd)} 个 tensor")

    minmax = {}
    if os.path.exists(minmax_path):
        print(f"加载 minmax:     {minmax_path}")
        minmax = torch.load(minmax_path, map_location="cpu")
        print(f"  共 {len(minmax)} 层有 act_minmax 记录")
    else:
        print(f"⚠ 未找到 minmax 文件，act_scale 列将全部为 —")

    analyze_qwen(sd, minmax)
    
    for prefix in ["vision_backbone.fused_featurizer.blocks",
                   "vision_backbone.fused_featurizer.encoder.layers"]:
        if any(k.startswith(prefix) for k in sd):
            analyze_vit(sd, minmax, prefix, "SIGLIP (fused_featurizer)")
            break

    for prefix in ["vision_backbone.featurizer.blocks",
                   "vision_backbone.featurizer.encoder.layer"]:
        if any(k.startswith(prefix) for k in sd):
            analyze_vit(sd, minmax, prefix, "DINOV2 (featurizer)")
            break

    analyze_projector(sd, minmax)
    print_summary()


# 保留 run_analysis 接口，兼容之前的调用方式
def run_analysis(model):
    print("⚠ run_analysis(model) 已废弃，请直接运行脚本分析 pt 文件：")
    print("  python analyze_quant_model.py --pt_path int8_models/vla_int8_static.pt")


if __name__ == "__main__":
    main()