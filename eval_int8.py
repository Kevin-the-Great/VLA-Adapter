"""
阶段二: 加载 INT8 模型 + 跑 eval
=================================

★ 重写版: 不依赖 torchao, 用纯 PyTorch ★

做的事:
  1. 加载 FP16 模型骨架
  2. 重建量化结构 (Smooth → ObservedLinear → StaticInt8Linear)
  3. 灌入保存的 INT8 权重
  4. 跑 LIBERO eval, 记录成功率

用法:
  conda activate int8-eval
  cd /path/to/VLA-Adapter

  python eval_int8.py \
      --quantized-model int8_models/vla_int8_static.pt \
      --device cuda:0

前提:
  先跑过 quantize_and_save.py 生成 .pt 文件

依赖:
  PyTorch >= 2.4
  不需要 torchao
"""

import os
import sys
import json
import time
import argparse
from collections import defaultdict

import torch
import numpy as np

# ============================================================
# 配置
# ============================================================
DEFAULT_QUANTIZED_MODEL = "int8_models/vla_int8_static.pt"
DEFAULT_TASK_SUITE = "libero_object"
DEFAULT_NUM_EPISODES = 20
DEFAULT_MAX_STEPS = 300
DEFAULT_SEED = 0


# ============================================================
# Step 1: 加载量化模型
# ============================================================
def load_quantized_model(quantized_model_path, device="cuda:0"):
    """
    加载已保存的 INT8 模型。

    流程:
      1. 读 quantize_config.json → 得到原始模型路径、act_scales 等
      2. 加载 FP16 模型骨架
      3. Smooth Qwen2.5 (和量化时一样)
      4. replace_with_observed → nn.Linear → ObservedLinear
      5. convert_to_quantized → ObservedLinear → StaticInt8Linear
      6. load_state_dict → 把保存的 INT8 权重灌进去

      步骤 3-5 只是搭"骨架", 数值会被步骤 6 覆盖。
    """
    from int8_vla import (
        smooth_qwen, replace_with_observed,
        convert_observed_to_quantized, print_model_stats,
    )

    # 读取配置
    config_dir = os.path.dirname(quantized_model_path) or "."
    config_path = os.path.join(config_dir, "quantize_config.json")

    if os.path.exists(config_path):
        with open(config_path) as f:
            config = json.load(f)
        model_path = config["model_path"]
        act_scales_path = config["act_scales_path"]
        alpha = config["alpha"]
        print(f"  从 {config_path} 读取配置:")
        print(f"    model_path:  {model_path}")
        print(f"    act_scales:  {act_scales_path}")
        print(f"    alpha:       {alpha}")
    else:
        print(f"  ⚠ 找不到 {config_path}, 用默认路径")
        model_path = "outputs/LIBERO-Object-Pro"
        act_scales_path = "act_scales/vla_adapter_object.pt"
        alpha = 0.5

    # 1. 加载 FP16 骨架
    print(f"\n  [1/5] 加载 FP16 模型骨架: {model_path}")
    t0 = time.time()
    try:
        from prismatic import load as load_vla
        model = load_vla(model_path)
        model.eval()
    except ImportError:
        from transformers import AutoModelForVision2Seq
        model = AutoModelForVision2Seq.from_pretrained(
            model_path, torch_dtype=torch.float16, device_map="cpu")
        model.eval()
    print(f"      加载完成 ({time.time()-t0:.1f}s)")

    # 2. Smooth
    print(f"  [2/5] Smooth Qwen2.5")
    act_scales = torch.load(act_scales_path, map_location="cpu")
    smooth_qwen(model, act_scales, alpha=alpha)

    # 3. 搭 ObservedLinear 骨架
    print(f"  [3/5] 搭建 ObservedLinear 骨架")
    n1 = replace_with_observed(model)
    print(f"      替换了 {n1} 个 Linear")

    # 4. 搭 StaticInt8Linear 骨架
    print(f"  [4/5] 搭建 StaticInt8Linear 骨架")
    n2 = convert_observed_to_quantized(model)
    print(f"      转换了 {n2} 个 ObservedLinear")

    # 5. 灌入 INT8 权重
    print(f"  [5/5] 加载 INT8 权重: {quantized_model_path}")
    t0 = time.time()
    state_dict = torch.load(quantized_model_path, map_location="cpu")
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"      加载完成 ({time.time()-t0:.1f}s)")

    if missing:
        print(f"      ⚠ missing keys: {len(missing)}")
        for k in missing[:5]:
            print(f"        - {k}")
    if unexpected:
        print(f"      ⚠ unexpected keys: {len(unexpected)}")
        for k in unexpected[:5]:
            print(f"        - {k}")

    print_model_stats(model)
    model = model.to(device)
    print(f"\n  模型已加载到 {device}, 准备 eval!")
    return model


# ============================================================
# Step 2: 跑 LIBERO eval
# ============================================================
def run_eval(model, task_suite_name, num_episodes, max_steps, seed, device):
    """在 LIBERO 上跑 eval, 记录每个 task 的成功率。"""
    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv

    print(f"\n{'='*60}")
    print(f"  LIBERO Eval (INT8)")
    print(f"{'='*60}")
    print(f"  task_suite:  {task_suite_name}")
    print(f"  episodes:    {num_episodes} per task")
    print(f"  max_steps:   {max_steps}")
    print(f"{'='*60}")

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[task_suite_name]()
    num_tasks = task_suite.get_num_tasks()

    model.eval()
    results = {}
    total_t0 = time.time()

    for task_id in range(num_tasks):
        task = task_suite.get_task(task_id)
        task_name = task.name
        task_description = task.language
        env_args = {
            "bddl_file_name": os.path.join(
                task_suite.get_task_bddl_file_path(),
                task.problem_folder, task.bddl_file),
            "camera_heights": 256, "camera_widths": 256,
        }
        env = OffScreenRenderEnv(**env_args)
        task_results = []
        task_t0 = time.time()

        for ep in range(num_episodes):
            env.seed(seed + ep)
            obs = env.reset()
            done, success, step = False, False, 0

            while not done and step < max_steps:
                try:
                    image = obs["agentview_image"]
                    with torch.no_grad():
                        action = model.predict_action(image, task_description)
                    if isinstance(action, torch.Tensor):
                        action = action.cpu().numpy()
                    obs, reward, done, info = env.step(action)
                    step += 1
                    if done or (info and info.get("success", False)):
                        success = True
                        break
                except Exception as e:
                    print(f"  ⚠ Task {task_id} Ep {ep} Step {step}: {e}")
                    break

            task_results.append(1 if success else 0)

        env.close()
        sr = np.mean(task_results) * 100
        results[task_name] = task_results
        print(f"  Task {task_id+1:2d}/{num_tasks}: "
              f"成功率 {sr:5.1f}% ({sum(task_results)}/{len(task_results)}) "
              f"[{task_name[:40]}] ({time.time()-task_t0:.0f}s)")

    # 总结
    all_results = [r for v in results.values() for r in v]
    avg_sr = np.mean(all_results) * 100

    print(f"\n{'='*60}")
    print(f"  总平均成功率: {avg_sr:.1f}%")
    print(f"  总 episode:   {len(all_results)}")
    print(f"  总耗时:       {time.time()-total_t0:.0f}s")
    print(f"{'='*60}")

    return results, avg_sr


# ============================================================
# Step 3: 保存结果
# ============================================================
def save_results(results, avg_sr, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    result_data = {
        "method": "static_int8_pure_pytorch",
        "average_success_rate": avg_sr,
        "per_task": {},
    }
    for task_name, task_results in results.items():
        result_data["per_task"][task_name] = {
            "success_rate": np.mean(task_results) * 100,
            "successes": sum(task_results),
            "episodes": len(task_results),
            "raw": task_results,
        }
    result_path = os.path.join(output_dir, "eval_int8_results.json")
    with open(result_path, "w") as f:
        json.dump(result_data, f, indent=2)
    print(f"  结果保存到: {result_path}")
    return result_path


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="阶段二: 加载 INT8 模型, 跑 LIBERO eval (纯 PyTorch)")
    parser.add_argument("--quantized-model", type=str, default=DEFAULT_QUANTIZED_MODEL)
    parser.add_argument("--task-suite", type=str, default=DEFAULT_TASK_SUITE)
    parser.add_argument("--num-episodes", type=int, default=DEFAULT_NUM_EPISODES)
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output-dir", type=str, default="int8_models")
    args = parser.parse_args()

    print("=" * 60)
    print("  阶段二: 加载 INT8 模型 + LIBERO eval")
    print("=" * 60)
    print(f"  量化模型: {args.quantized_model}")
    print(f"  task:     {args.task_suite}")
    print(f"  episodes: {args.num_episodes}/task")
    print(f"  device:   {args.device}")
    print(f"  PyTorch:  {torch.__version__}")
    print("=" * 60)

    # Step 1
    print(f"\n[Step 1] 加载量化模型")
    model = load_quantized_model(args.quantized_model, device=args.device)

    # Step 2
    results, avg_sr = run_eval(
        model,
        task_suite_name=args.task_suite,
        num_episodes=args.num_episodes,
        max_steps=args.max_steps,
        seed=args.seed,
        device=args.device,
    )

    # Step 3
    save_results(results, avg_sr, args.output_dir)


if __name__ == "__main__":
    main()