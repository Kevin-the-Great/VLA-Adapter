"""
阶段一: 校准 + 量化 + 保存 (只跑一次)
======================================

★ 重写版: 不依赖 torchao, 用纯 PyTorch ★

做的事:
  1. 加载 FP16 模型
  2. Smooth Qwen2.5
  3. 插入 observer
  4. 用 LIBERO eval 跑 10 个 episode 做校准 (observer 自动收集激活范围)
  5. 转成真 INT8 (StaticInt8Linear, 用 torch._int_mm)
  6. 保存到 .pt 文件

用法:
  conda activate int8-eval
  cd /path/to/VLA-Adapter

  python quantize_and_save.py \
      --model-path outputs/LIBERO-Object-Pro \
      --act-scales act_scales/vla_adapter_object.pt \
      --output int8_models/vla_int8_static.pt \
      --calib-episodes 10

产出:
  int8_models/vla_int8_static.pt   (INT8 权重 + 激活 scale)
  int8_models/quantize_config.json (量化配置, 加载时需要)

之后用 eval_int8.py 加载这个 .pt 跑 eval, 不用再校准了。

依赖:
  PyTorch >= 2.4 (有 torch._int_mm)
  不需要 torchao
"""

import os
import sys
import json
import time
import argparse

import torch

# ============================================================
# 配置 (根据你的 HPC 路径调整)
# ============================================================
DEFAULT_MODEL_PATH = "outputs/LIBERO-Object-Pro"
DEFAULT_ACT_SCALES = "act_scales/vla_adapter_object.pt"
DEFAULT_OUTPUT = "int8_models/vla_int8_static.pt"
DEFAULT_ALPHA = 0.5
DEFAULT_CALIB_EPISODES = 10
DEFAULT_TASK_SUITE = "libero_object"
DEFAULT_SEED = 0


# ============================================================
# Step 0: 检查依赖
# ============================================================
def check_dependencies():
    """检查 int8_vla 和 LIBERO 是否可用。"""
    errors = []

    print(f"  PyTorch: {torch.__version__}")

    if not hasattr(torch, '_int_mm'):
        errors.append("torch._int_mm 不可用! 需要 PyTorch >= 2.3")

    try:
        from libero.libero import benchmark
        print(f"  LIBERO:  OK")
    except ImportError:
        errors.append("LIBERO 未安装!")

    try:
        from int8_vla import smooth_qwen, replace_with_observed
        print(f"  int8_vla.py: OK")
    except ImportError:
        errors.append(
            "找不到 int8_vla.py!\n"
            "  请把它放到当前目录或 VLA-Adapter 目录下。"
        )

    if errors:
        print("\n  === 缺少依赖 ===")
        for e in errors:
            print(f"  ✗ {e}")
        sys.exit(1)


# ============================================================
# Step 1: 加载模型
# ============================================================
def load_model(model_path):
    """加载 VLA-Adapter 模型 (FP16)。"""
    print(f"\n[Step 1] 加载 FP16 模型: {model_path}")
    t0 = time.time()

    try:
        from prismatic import load as load_vla
        model = load_vla(model_path)
        model.eval()
        print(f"  用 prismatic.load() 加载, 耗时 {time.time()-t0:.1f}s")
        return model
    except Exception as e:
        print(f"  prismatic.load() 失败: {e}, 尝试 AutoModelForVision2Seq...")
        pass

    try:
        from transformers import AutoModelForVision2Seq
        model = AutoModelForVision2Seq.from_pretrained(
            model_path, torch_dtype=torch.float16, device_map="cpu",
            trust_remote_code=True)
        model.eval()
        print(f"  用 AutoModelForVision2Seq 加载, 耗时 {time.time()-t0:.1f}s")
        return model
    except Exception as e:
        raise FileNotFoundError(f"无法加载模型: {model_path}\n  错误: {e}")


# ============================================================
# Step 2-6: 各步骤
# ============================================================
def do_smooth(model, act_scales_path, alpha):
    from int8_vla import smooth_qwen
    print(f"\n[Step 2] Smooth Qwen2.5 (alpha={alpha})")
    act_scales = torch.load(act_scales_path, map_location="cpu")
    print(f"  加载了 {len(act_scales)} 个 act_scales key")
    return smooth_qwen(model, act_scales, alpha=alpha)


def do_insert_observers(model):
    from int8_vla import replace_with_observed, print_model_stats
    print(f"\n[Step 3] 插入 observer")
    n = replace_with_observed(model)
    print(f"  替换了 {n} 个 Linear → ObservedLinear")
    print_model_stats(model)
    return n


def do_calibration(model, num_episodes, task_suite_name, seed, device, model_path):
    """校准 = 用 LIBERO 环境跑几个 episode, ObservedLinear forward 时自动收集。"""
    from int8_vla import verify_observer_stats
    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv
    from transformers import AutoProcessor
    from PIL import Image
    import numpy as np
    import math

    def quat2axisangle(quat):
        """Convert quaternion to axis-angle. Copied from robosuite."""
        if quat[3] > 1.0:
            quat[3] = 1.0
        elif quat[3] < -1.0:
            quat[3] = -1.0
        den = np.sqrt(1.0 - quat[3] * quat[3])
        if math.isclose(den, 0.0):
            return np.zeros(3)
        return (quat[:3] * 2.0 * math.acos(quat[3])) / den

    def normalize_proprio_q99(proprio, norm_stats):
        """Normalize proprio using q99/q01 bounds (matches training)."""
        mask = norm_stats.get("mask", np.ones_like(norm_stats["q01"], dtype=bool))
        p_high = np.array(norm_stats["q99"])
        p_low = np.array(norm_stats["q01"])
        normalized = np.clip(
            np.where(mask, 2 * (proprio - p_low) / (p_high - p_low + 1e-8) - 1, proprio),
            a_min=-1.0, a_max=1.0,
        )
        return normalized

    # -- 加载 action_head 和 proprio_projector --
    from prismatic.models.action_heads import L1RegressionActionHead
    from prismatic.models.projectors import ProprioProjector

    print(f"\n[Step 4] 校准 (跑 {num_episodes} 个 episode)")

    # 加载 processor
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    print(f"  Processor 加载完成")

    # 加载 action_head
    LLM_DIM = 896
    ACTION_DIM = 7
    PROPRIO_DIM = 8
    action_head = L1RegressionActionHead(
        input_dim=LLM_DIM, hidden_dim=LLM_DIM, action_dim=ACTION_DIM,
        use_pro_version=True,
    ).to(torch.float16).to(device)
    ah_ckpt = os.path.join(model_path, "action_head--checkpoint.pt")
    if os.path.exists(ah_ckpt):
        sd = torch.load(ah_ckpt, map_location="cpu")
        sd = {k.replace("module.", ""): v for k, v in sd.items()}
        action_head.load_state_dict(sd)
        print(f"  Action head 加载: {ah_ckpt}")
    else:
        print(f"  ⚠ 未找到 action_head checkpoint, 用随机权重")
    action_head.eval()

    # 加载 proprio_projector
    proprio_projector = ProprioProjector(
        llm_dim=LLM_DIM, proprio_dim=PROPRIO_DIM,
    ).to(torch.float16).to(device)
    pp_ckpt = os.path.join(model_path, "proprio_projector--checkpoint.pt")
    if os.path.exists(pp_ckpt):
        sd = torch.load(pp_ckpt, map_location="cpu")
        sd = {k.replace("module.", ""): v for k, v in sd.items()}
        proprio_projector.load_state_dict(sd)
        print(f"  Proprio projector 加载: {pp_ckpt}")
    else:
        print(f"  ⚠ 未找到 proprio_projector checkpoint, 用随机权重")
    proprio_projector.eval()

    # 加载 dataset_statistics 用于 unnormalize
    ds_path = os.path.join(model_path, "dataset_statistics.json")
    if os.path.exists(ds_path):
        with open(ds_path) as f:
            model.norm_stats = json.load(f)
        print(f"  Dataset stats 加载完成")

    # 确定 unnorm_key
    if hasattr(model, 'norm_stats') and model.norm_stats:
        unnorm_key = list(model.norm_stats.keys())[0]
        print(f"  unnorm_key: {unnorm_key}")
    else:
        unnorm_key = None
        print(f"  ⚠ 无 norm_stats, unnorm_key=None")

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[task_suite_name]()
    num_tasks = task_suite.get_num_tasks()

    model = model.to(device)
    model.eval()

    total_steps = 0
    t0 = time.time()
    episodes_per_task = max(1, num_episodes // num_tasks)
    remaining = num_episodes

    for task_id in range(num_tasks):
        if remaining <= 0:
            break
        task = task_suite.get_task(task_id)
        task_description = task.language
        env_args = {
            "bddl_file_name": task_suite.get_task_bddl_file_path(task_id),
            "camera_heights": 256, "camera_widths": 256,
        }
        env = OffScreenRenderEnv(**env_args)
        env.seed(seed)

        n_ep = min(episodes_per_task, remaining)
        for ep in range(n_ep):
            obs = env.reset()
            done, step = False, 0
            while not done and step < 300:
                try:
                    # 1) 获取图像: 旋转180° + resize to 224x224 + 转 PIL
                    image = obs["agentview_image"]
                    image = image[::-1, ::-1]  # 旋转 180°, 与训练预处理一致
                    # Resize to 224x224 using same pipeline as training
                    import tensorflow as tf
                    img_encoded = tf.image.encode_jpeg(image)
                    img_decoded = tf.io.decode_image(img_encoded, expand_animations=False, dtype=tf.uint8)
                    img_resized = tf.image.resize(img_decoded, (224, 224), method="lanczos3", antialias=True)
                    image = tf.cast(tf.clip_by_value(tf.round(img_resized), 0, 255), tf.uint8).numpy()
                    pil_image = Image.fromarray(image).convert("RGB")

                    # 2) 用 processor 预处理
                    prompt = f"In: What action should the robot take to {task_description.lower()}?\nOut:"
                    inputs = processor(prompt, pil_image).to(device, dtype=torch.float16)

                    # 3) 获取 proprio: eef_pos(3) + axisangle(3) + gripper_qpos(2) = 8
                    proprio = np.concatenate([
                        obs["robot0_eef_pos"],
                        quat2axisangle(obs["robot0_eef_quat"].copy()),
                        obs["robot0_gripper_qpos"],
                    ])
                    # Normalize proprio using dataset stats
                    if unnorm_key and hasattr(model, 'norm_stats') and unnorm_key in model.norm_stats:
                        pstats = model.norm_stats[unnorm_key].get("proprio", None)
                        if pstats:
                            proprio = normalize_proprio_q99(proprio, pstats)

                    # 4) 调用 predict_action
                    with torch.no_grad():
                        action, _ = model.predict_action(
                            **inputs,
                            unnorm_key=unnorm_key,
                            do_sample=False,
                            proprio=proprio,
                            proprio_projector=proprio_projector,
                            action_head=action_head,
                        )
                    if isinstance(action, torch.Tensor):
                        action = action.cpu().numpy()
                    # action 可能是 (chunk_len, 7), 取第一个
                    if hasattr(action, 'shape') and len(action.shape) > 1:
                        action = action[0]
                    obs, reward, done, info = env.step(action)
                    step += 1
                    total_steps += 1
                except Exception as e:
                    import traceback
                    print(f"  ⚠ Task {task_id} Ep {ep} Step {step}: {e}")
                    traceback.print_exc()
                    break
            remaining -= 1
        env.close()

        print(f"  Task {task_id+1}/{num_tasks} "
              f"({num_episodes - remaining}/{num_episodes} ep, "
              f"{total_steps} steps, {time.time()-t0:.0f}s)")

    verify_observer_stats(model, num_show=5)
    return total_steps


def do_convert(model):
    from int8_vla import convert_observed_to_quantized, print_model_stats, verify_int8_model
    print(f"\n[Step 5] 转换为真 INT8")
    model = model.cpu()
    n = convert_observed_to_quantized(model)
    print(f"  转换了 {n} 个 ObservedLinear → StaticInt8Linear")
    print_model_stats(model)
    verify_int8_model(model, num_show=3)
    return n


def do_save(model, output_path, config):
    print(f"\n[Step 6] 保存")
    output_dir = os.path.dirname(output_path) or "."
    os.makedirs(output_dir, exist_ok=True)
    t0 = time.time()
    torch.save(model.state_dict(), output_path)
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  模型:  {output_path} ({size_mb:.1f} MB, {time.time()-t0:.1f}s)")
    config_path = os.path.join(output_dir, "quantize_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"  配置:  {config_path}")
    return output_path, config_path


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="阶段一: 校准 + 量化 + 保存 INT8 模型 (纯 PyTorch)")
    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--act-scales", type=str, default=DEFAULT_ACT_SCALES)
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT)
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--calib-episodes", type=int, default=DEFAULT_CALIB_EPISODES)
    parser.add_argument("--task-suite", type=str, default=DEFAULT_TASK_SUITE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    print("=" * 60)
    print("  阶段一: 校准 + 量化 + 保存 (纯 PyTorch)")
    print("=" * 60)
    print(f"  模型:       {args.model_path}")
    print(f"  act_scales: {args.act_scales}")
    print(f"  输出:       {args.output}")
    print(f"  alpha:      {args.alpha}")
    print(f"  校准 ep:    {args.calib_episodes}")
    print(f"  PyTorch:    {torch.__version__}")
    print("=" * 60)

    print("\n[Step 0] 检查依赖")
    check_dependencies()

    model = load_model(args.model_path)
    do_smooth(model, args.act_scales, args.alpha)
    do_insert_observers(model)

    total_steps = do_calibration(
        model, num_episodes=args.calib_episodes,
        task_suite_name=args.task_suite,
        seed=args.seed, device=args.device,
        model_path=args.model_path)

    do_convert(model)

    config = {
        "model_path": args.model_path,
        "act_scales_path": args.act_scales,
        "alpha": args.alpha,
        "calib_episodes": args.calib_episodes,
        "calib_steps": total_steps,
        "task_suite": args.task_suite,
        "seed": args.seed,
        "pytorch_version": torch.__version__,
    }
    do_save(model, args.output, config)

    print(f"\n{'='*60}")
    print(f"  阶段一完成! 下一步:")
    print(f"  python eval_int8.py --quantized-model {args.output}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()