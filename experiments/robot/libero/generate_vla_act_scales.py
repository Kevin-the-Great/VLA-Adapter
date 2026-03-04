"""
generate_vla_act_scales.py

在 VLA-Adapter eval 的基础上挂 hook，收集每个 Linear 层的激活最大值。
输出一个 act_scales.pt 文件，用于后续 SmoothQuant 的 smooth 阶段。

用法（跟你平时跑 eval 一样，只是换了脚本名，加了 --output_path）：

CUDA_VISIBLE_DEVICES=0 python -u generate_vla_act_scales.py \
  --model_family openvla \
  --use_proprio True \
  --num_images_in_input 2 \
  --use_film False \
  --use_pro_version True \
  --load_in_8bit False \
  --pretrained_checkpoint outputs/LIBERO-Object-Pro \
  --task_suite_name libero_object \
  --output_path act_scales/vla_adapter_object.pt \
  --num_calibration_steps 512

也可以不跑完整 eval，只跑 512 步就停：
  --calibration_only True
"""

import json
import logging
import os
import sys
import functools
from collections import deque
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, Union

import draccus
import numpy as np
import torch
import torch.nn as nn
import tqdm
from libero.libero import benchmark

# Append paths
sys.path.append("../..")

from experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    get_libero_wrist_image,
    quat2axisangle,
    save_rollout_video,
)
from experiments.robot.openvla_utils import (
    get_action_head,
    get_noisy_action_projector,
    get_processor,
    get_proprio_projector,
    resize_image_for_policy,
)
from experiments.robot.robot_utils import (
    DATE_TIME,
    get_action,
    get_image_resize_size,
    get_model,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)
from prismatic.vla.constants import NUM_ACTIONS_CHUNK


# =====================================================
# ============ 校准核心代码（来自 SmoothQuant）============
# =====================================================

def attach_act_scales_hooks(model):
    """
    给模型的所有 nn.Linear 层挂 hook，记录每个通道的最大激活值。
    
    这段代码的逻辑跟 SmoothQuant 的 calibration.py 里的 get_act_scales() 完全一样，
    只是不需要自己喂数据——数据由 eval 流程提供。
    
    Returns:
        act_scales: dict，key 是层名，value 会在 forward 过程中被持续更新
        hooks: list，用完之后要 remove
    """
    act_scales = {}

    def stat_tensor(name, tensor):
        """记录这个 tensor 每个通道的最大绝对值"""
        hidden_dim = tensor.shape[-1]
        tensor = tensor.view(-1, hidden_dim).abs().detach()
        # .float() 避免 bfloat16 精度问题
        comming_max = torch.max(tensor, dim=0)[0].float().cpu()
        if name in act_scales:
            act_scales[name] = torch.max(act_scales[name], comming_max)
        else:
            act_scales[name] = comming_max

    def stat_input_hook(m, x, y, name):
        """hook 函数：每次 Linear 层被调用时，记录输入的最大值"""
        if isinstance(x, tuple):
            x = x[0]
        stat_tensor(name, x)

    hooks = []
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear):
            hooks.append(
                m.register_forward_hook(
                    functools.partial(stat_input_hook, name=name)
                )
            )

    print(f"[Calibration] Attached hooks to {len(hooks)} Linear layers in main model")
    return act_scales, hooks


def attach_action_head_hooks(action_head, act_scales, hooks):
    """
    给 action_head (Policy) 的所有 nn.Linear 层挂 hook。
    复用同一个 act_scales dict，key 加 "action_head." 前缀。
    """
    def stat_input_hook(m, x, y, name):
        if isinstance(x, tuple):
            x = x[0]
        hidden_dim = x.shape[-1]
        tensor = x.view(-1, hidden_dim).abs().detach()
        comming_max = torch.max(tensor, dim=0)[0].float().cpu()
        if name in act_scales:
            act_scales[name] = torch.max(act_scales[name], comming_max)
        else:
            act_scales[name] = comming_max

    count = 0
    for name, m in action_head.named_modules():
        if isinstance(m, nn.Linear):
            hooks.append(
                m.register_forward_hook(
                    functools.partial(stat_input_hook, name=f"action_head.{name}")
                )
            )
            count += 1

    print(f"[Calibration] Attached hooks to {count} Linear layers in action_head (Policy)")
    return act_scales, hooks


def remove_hooks(hooks):
    """移除所有 hook"""
    for h in hooks:
        h.remove()
    print(f"[Calibration] Removed {len(hooks)} hooks")


def save_act_scales(act_scales, output_path):
    """保存 act_scales 到 .pt 文件"""
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    torch.save(act_scales, output_path)
    print(f"[Calibration] Saved act_scales to {output_path}")
    print(f"[Calibration] Total layers: {len(act_scales)}")
    
    # 分别统计主模型和 Policy 的层数
    policy_keys = [k for k in act_scales if k.startswith("action_head.")]
    model_keys = [k for k in act_scales if not k.startswith("action_head.")]
    print(f"  Main model layers: {len(model_keys)}")
    print(f"  Policy (action_head) layers: {len(policy_keys)}")
    
    # 打印一些统计信息
    print(f"\n  --- Main model samples ---")
    for name, scale in list(sorted(act_scales.items()))[:5]:
        if not name.startswith("action_head."):
            print(f"  {name}: shape={scale.shape}, max={scale.max():.2f}, min={scale.min():.4f}")
    
    print(f"\n  --- Policy (action_head) samples ---")
    for name, scale in sorted(act_scales.items()):
        if name.startswith("action_head."):
            print(f"  {name}: max={scale.max():.2f}, min={scale.min():.4f}")
            break  # 只打印第一个作为样例
    
    # 打印 Policy 各层的 outlier 排名（方便决定量化策略）
    if policy_keys:
        print(f"\n  --- Policy outlier 排名 (Top 10) ---")
        policy_outliers = [(k, act_scales[k].max().item()) for k in policy_keys]
        policy_outliers.sort(key=lambda x: x[1], reverse=True)
        for name, max_val in policy_outliers[:10]:
            print(f"  {name}: max_outlier={max_val:.2f}")


# =====================================================
# ========= 以下是 eval 代码（基本照搬原版）=============
# =====================================================

class TaskSuite(str, Enum):
    LIBERO_SPATIAL = "libero_spatial"
    LIBERO_OBJECT = "libero_object"
    LIBERO_GOAL = "libero_goal"
    LIBERO_10 = "libero_10"
    LIBERO_90 = "libero_90"

TASK_MAX_STEPS = {
    TaskSuite.LIBERO_SPATIAL: 220,
    TaskSuite.LIBERO_OBJECT: 280,
    TaskSuite.LIBERO_GOAL: 300,
    TaskSuite.LIBERO_10: 520,
    TaskSuite.LIBERO_90: 400,
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


@dataclass
class GenerateConfig:
    # fmt: off
    # === 原版 eval 参数（跟 run_libero_eval.py 一样） ===
    model_family: str = "openvla"
    pretrained_checkpoint: Union[str, Path] = ""
    use_l1_regression: bool = True
    use_minivlm: bool = True
    num_diffusion_steps: int = 50
    use_film: bool = False
    num_images_in_input: int = 2
    use_proprio: bool = True
    center_crop: bool = True
    num_open_loop_steps: int = 8
    unnorm_key: Union[str, Path] = ""
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    task_suite_name: str = TaskSuite.LIBERO_SPATIAL
    num_steps_wait: int = 10
    num_trials_per_task: int = 50
    initial_states_path: str = "DEFAULT"
    env_img_res: int = 256
    run_id_note: Optional[str] = None
    local_log_dir: str = "./experiments/logs"
    use_wandb: bool = False
    wandb_entity: str = "your-wandb-entity"
    wandb_project: str = "your-wandb-project"
    seed: int = 7
    save_version: str = "vla-adapter"
    use_pro_version: bool = True
    phase: str = "Inference"
    
    # === 新增：校准专用参数 ===
    output_path: str = "act_scales/vla_adapter.pt"        # act_scales 保存路径
    num_calibration_steps: int = 512                       # 收集多少步激活数据
    calibration_only: bool = False                         # True = 收集完就停，不跑完整 eval
    # fmt: on


def initialize_model(cfg):
    """跟原版一样，加载模型"""
    model = get_model(cfg)
    model.set_version(cfg.save_version)
    
    proprio_projector = None
    if cfg.use_proprio:
        proprio_projector = get_proprio_projector(cfg, model.llm_dim, proprio_dim=8)

    action_head = None
    if cfg.use_l1_regression:
        action_head = get_action_head(cfg, model.llm_dim)

    processor = None
    if cfg.model_family == "openvla":
        processor = get_processor(cfg)
        # Check unnorm key
        unnorm_key = cfg.task_suite_name
        if unnorm_key not in model.norm_stats and f"{unnorm_key}_no_noops" in model.norm_stats:
            unnorm_key = f"{unnorm_key}_no_noops"
        assert unnorm_key in model.norm_stats, f"Action un-norm key {unnorm_key} not found!"
        cfg.unnorm_key = unnorm_key

    return model, action_head, proprio_projector, None, processor


@draccus.wrap()
def calibrate_vla(cfg: GenerateConfig) -> None:
    """
    主函数：加载模型 → 挂 hook → 跑 eval（收集激活） → 保存 act_scales
    """
    assert cfg.pretrained_checkpoint, "pretrained_checkpoint must not be empty!"
    
    # 1. 设置随机种子
    set_seed_everywhere(cfg.seed)
    
    # 2. 加载模型（跟平时 eval 完全一样）
    print("=" * 60)
    print("[Calibration] Step 1: Loading model...")
    print("=" * 60)
    model, action_head, proprio_projector, noisy_action_projector, processor = initialize_model(cfg)
    
    # 3. 挂 hook（主模型 + action_head）
    print("=" * 60)
    print("[Calibration] Step 2: Attaching hooks to all Linear layers...")
    print("=" * 60)
    act_scales, hooks = attach_act_scales_hooks(model)
    
    # 也给 action_head (Policy) 挂 hook
    if action_head is not None:
        act_scales, hooks = attach_action_head_hooks(action_head, act_scales, hooks)
    
    total_hook_count = len(hooks)
    print(f"[Calibration] Total hooks attached: {total_hook_count}")
    
    # 4. 跑 eval，让 hook 收集激活数据
    print("=" * 60)
    print(f"[Calibration] Step 3: Running eval to collect activations ({cfg.num_calibration_steps} steps)...")
    print("=" * 60)
    
    resize_size = get_image_resize_size(cfg)
    
    # 初始化 LIBERO
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    num_tasks = task_suite.n_tasks
    
    step_count = 0
    total_episodes = 0
    total_successes = 0
    
    for task_id in range(num_tasks):
        if step_count >= cfg.num_calibration_steps and cfg.calibration_only:
            break
            
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description = get_libero_env(task, cfg.model_family, resolution=cfg.env_img_res)
        
        # 每个 task 跑几个 episode
        trials_to_run = min(cfg.num_trials_per_task, 
                           max(1, (cfg.num_calibration_steps - step_count) // 20))
        if not cfg.calibration_only:
            trials_to_run = cfg.num_trials_per_task
        
        for episode_idx in range(trials_to_run):
            if step_count >= cfg.num_calibration_steps and cfg.calibration_only:
                break
            
            initial_state = initial_states[episode_idx % len(initial_states)]
            
            # 跑一个 episode
            env.reset()
            obs = env.set_init_state(initial_state)
            action_queue = deque(maxlen=cfg.num_open_loop_steps)
            
            max_steps = TASK_MAX_STEPS[cfg.task_suite_name]
            success = False
            t = 0
            
            while t < max_steps + cfg.num_steps_wait:
                if t < cfg.num_steps_wait:
                    obs, reward, done, info = env.step(get_libero_dummy_action(cfg.model_family))
                    t += 1
                    continue
                
                # 准备观测
                img = get_libero_image(obs)
                wrist_img = get_libero_wrist_image(obs)
                img_resized = resize_image_for_policy(img, resize_size)
                wrist_img_resized = resize_image_for_policy(wrist_img, resize_size)
                observation = {
                    "full_image": img_resized,
                    "wrist_image": wrist_img_resized,
                    "state": np.concatenate(
                        (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
                    ),
                }
                
                # 查询模型（这里会触发 forward，hook 就会记录激活）
                if len(action_queue) == 0:
                    actions = get_action(
                        cfg, model, observation, task_description,
                        processor=processor, action_head=action_head,
                        proprio_projector=proprio_projector,
                        noisy_action_projector=noisy_action_projector,
                        use_film=cfg.use_film, use_minivlm=cfg.use_minivlm
                    )
                    action_queue.extend(actions)
                    step_count += 1
                    
                    # 每 50 步打印一次进度
                    if step_count % 50 == 0:
                        policy_count = len([k for k in act_scales if k.startswith("action_head.")])
                        print(f"[Calibration] Collected {step_count}/{cfg.num_calibration_steps} forward passes "
                              f"(model keys: {len(act_scales) - policy_count}, policy keys: {policy_count})")
                
                action = action_queue.popleft()
                action = normalize_gripper_action(action, binarize=True)
                if cfg.model_family == "openvla":
                    action = invert_gripper_action(action)
                
                obs, reward, done, info = env.step(action.tolist())
                if done:
                    success = True
                    break
                t += 1
            
            total_episodes += 1
            if success:
                total_successes += 1
        
        env.close()
        del env
        
        print(f"[Calibration] Task {task_id+1}/{num_tasks} done. "
              f"Steps: {step_count}, Success rate: {total_successes}/{total_episodes}")
    
    # 5. 移除 hook
    remove_hooks(hooks)
    
    # 6. 保存结果
    print("=" * 60)
    print("[Calibration] Step 4: Saving act_scales...")
    print("=" * 60)
    save_act_scales(act_scales, cfg.output_path)
    
    # 打印总结
    print("=" * 60)
    print(f"[Calibration] Done!")
    print(f"  Forward passes collected: {step_count}")
    print(f"  Linear layers tracked: {len(act_scales)}")
    policy_count = len([k for k in act_scales if k.startswith("action_head.")])
    model_count = len(act_scales) - policy_count
    print(f"    Main model: {model_count} layers")
    print(f"    Policy (action_head): {policy_count} layers")
    print(f"  Output saved to: {cfg.output_path}")
    if total_episodes > 0:
        print(f"  Eval success rate: {total_successes}/{total_episodes} "
              f"({total_successes/total_episodes*100:.1f}%)")
    print("=" * 60)


if __name__ == "__main__":
    calibrate_vla()