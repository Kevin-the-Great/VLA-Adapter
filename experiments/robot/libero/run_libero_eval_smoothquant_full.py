"""
run_libero_eval_smoothquant_full.py

在原版 run_libero_eval.py 基础上，增加了 SmoothQuant fake quantization 功能。
唯一改动在 initialize_model() 函数里，模型加载后插入了 smooth + quantize 代码。

用法 (4 组实验):
    cd /hpc2hdd/home/kluo573/vlaadapter/VLA-Adapter
    export PYTHONPATH="/hpc2hdd/home/kluo573/vlaadapter/VLA-Adapter:/hpc2hdd/home/kluo573/vlaadapter/VLA-Adapter/LIBERO"
    mkdir -p eval_logs

    # 实验 1: FP16 baseline
    QUANT_MODE=fp16 CUDA_VISIBLE_DEVICES=0 python -u \
      experiments/robot/libero/run_libero_eval_smoothquant_full.py \
      --model_family openvla --use_proprio True --num_images_in_input 2 \
      --use_film False --use_pro_version True \
      --pretrained_checkpoint outputs/LIBERO-Spatial-Pro \
      --task_suite_name libero_spatial \
      > eval_logs/Spatial--fp16.log 2>&1 &

    # 实验 2: Naive W8A8 (Qwen2.5 only, 不 smooth)
    QUANT_MODE=naive_w8a8 CUDA_VISIBLE_DEVICES=0 python -u \
      experiments/robot/libero/run_libero_eval_smoothquant_full.py \
      --model_family openvla --use_proprio True --num_images_in_input 2 \
      --use_film False --use_pro_version True \
      --pretrained_checkpoint outputs/LIBERO-Spatial-Pro \
      --task_suite_name libero_spatial \
      > eval_logs/Spatial--naive_w8a8.log 2>&1 &

    # 实验 3: SmoothQuant W8A8 (Qwen2.5 only)
    QUANT_MODE=smoothquant CUDA_VISIBLE_DEVICES=0 python -u \
      experiments/robot/libero/run_libero_eval_smoothquant_full.py \
      --model_family openvla --use_proprio True --num_images_in_input 2 \
      --use_film False --use_pro_version True \
      --pretrained_checkpoint outputs/LIBERO-Spatial-Pro \
      --task_suite_name libero_spatial \
      > eval_logs/Spatial--smoothquant.log 2>&1 &

    # 实验 4: 全模型量化 (Qwen2.5 SmoothQuant + ViT/Projector Naive W8A8)
    QUANT_MODE=smoothquant QUANT_VISION=1 CUDA_VISIBLE_DEVICES=0 python -u \
      experiments/robot/libero/run_libero_eval_smoothquant_full.py \
      --model_family openvla --use_proprio True --num_images_in_input 2 \
      --use_film False --use_pro_version True \
      --pretrained_checkpoint outputs/LIBERO-Spatial-Pro \
      --task_suite_name libero_spatial \
      > eval_logs/Spatial--smoothquant_full.log 2>&1 &

    # 实验 5: 真 INT8 静态量化（首次跑，校准 + eval + 保存）
    QUANT_MODE=static_int8 CALIB_EPISODES=10 INT8_MODEL_PATH=int8_models/vla_int8_static.pt \
      CUDA_VISIBLE_DEVICES=0 python -u \
      experiments/robot/libero/run_libero_eval_smoothquant_full.py \
      --model_family openvla --use_proprio True --num_images_in_input 2 \
      --use_film False --use_pro_version True \
      --pretrained_checkpoint outputs/LIBERO-Spatial-Pro \
      --task_suite_name libero_spatial \
      > eval_logs/Spatial--static_int8.log 2>&1 &

    # 实验 5 之后：直接加载 INT8 权重跑 eval（跳过校准）
    QUANT_MODE=load_int8 INT8_MODEL_PATH=int8_models/vla_int8_static.pt \
      CUDA_VISIBLE_DEVICES=0 python -u \
      experiments/robot/libero/run_libero_eval_smoothquant_full.py \
      --model_family openvla --use_proprio True --num_images_in_input 2 \
      --use_film False --use_pro_version True \
      --pretrained_checkpoint outputs/LIBERO-Spatial-Pro \
      --task_suite_name libero_spatial \
      > eval_logs/Spatial--load_int8.log 2>&1 &
"""

import json
import logging
import os
import sys
from collections import deque
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, Union

import draccus
import numpy as np
import torch
import tqdm
from libero.libero import benchmark

import wandb

# Append current directory so that interpreter can find experiments.robot
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


# ===================================================================
# SmoothQuant 配置
# 通过环境变量控制，不需要改代码：
#
# QUANT_MODE: 控制 Qwen2.5 的量化方式
#   fp16          → 不量化 (baseline)
#   naive_w8a8    → 直接量化，不 smooth (fake quant)
#   smoothquant   → smooth + 量化 (fake quant)
#   static_int8   → smooth + 真 INT8 (用 torch._int_mm)
#
# QUANT_VISION: 控制 ViT + Projector 是否也量化
#   0 (默认)      → 不量化，保持 FP16
#   1             → Naive W8A8（不 smooth）
#
# CALIB_EPISODES: static_int8 模式下的校准 episode 数
#   默认 10       → 跑 10 个 episode 让 observer 收集激活范围，
#                   然后转换为真 INT8，再跑正式 eval
#
# 组合示例：
#   QUANT_MODE=fp16        QUANT_VISION=0              → 全 FP16 baseline        (实验1)
#   QUANT_MODE=naive_w8a8  QUANT_VISION=0              → 只量化 Qwen2.5 (naive)  (实验2)
#   QUANT_MODE=smoothquant QUANT_VISION=0              → 只量化 Qwen2.5 (smooth) (实验3)
#   QUANT_MODE=smoothquant QUANT_VISION=1              → 全模型量化               (实验4)
#   QUANT_MODE=static_int8 CALIB_EPISODES=10           → 真 INT8 静态量化         (实验5)
# ===================================================================
QUANT_MODE = os.environ.get("QUANT_MODE", "fp16")
QUANT_VISION = os.environ.get("QUANT_VISION", "0") == "1"
CALIB_EPISODES = int(os.environ.get("CALIB_EPISODES", "10"))
INT8_MODEL_PATH = os.environ.get("INT8_MODEL_PATH", "int8_models/vla_int8_static.pt")

# smoothquant 仓库路径（在 VLA-Adapter 目录下）
SMOOTHQUANT_ROOT = "/hpc2hdd/home/kluo573/vlaadapter/VLA-Adapter/smoothquant"
ACT_SCALES_PATH = "act_scales/vla_adapter_object.pt"
SMOOTH_ALPHA = 0.5

# 把 smoothquant 仓库和 VLA-Adapter 根目录加到 Python path
sys.path.insert(0, SMOOTHQUANT_ROOT)
sys.path.insert(0, "/hpc2hdd/home/kluo573/vlaadapter/VLA-Adapter")


# Define task suite constants
class TaskSuite(str, Enum):
    LIBERO_SPATIAL = "libero_spatial"
    LIBERO_OBJECT = "libero_object"
    LIBERO_GOAL = "libero_goal"
    LIBERO_10 = "libero_10"
    LIBERO_90 = "libero_90"


# Define max steps for each task suite
TASK_MAX_STEPS = {
    TaskSuite.LIBERO_SPATIAL: 220,
    TaskSuite.LIBERO_OBJECT: 280,
    TaskSuite.LIBERO_GOAL: 300,
    TaskSuite.LIBERO_10: 520,
    TaskSuite.LIBERO_90: 400,
}


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)



@dataclass
class GenerateConfig:
    # fmt: off

    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    model_family: str = "openvla"                    # Model family
    pretrained_checkpoint: Union[str, Path] = ""     # Pretrained checkpoint path
    use_l1_regression: bool = True                   # If True, uses continuous action head with L1 regression objective
    use_minivlm: bool = True                         # If True, uses minivlm
    num_diffusion_steps: int = 50                    # (When `diffusion==True`) Number of diffusion steps for inference
    use_film: bool = False                           # If True, uses FiLM to infuse language inputs into visual features
    num_images_in_input: int = 2                     # Number of images in the VLA input (default: 1)
    use_proprio: bool = True                         # Whether to include proprio state in input

    center_crop: bool = True                         # Center crop? (if trained w/ random crop image aug)
    num_open_loop_steps: int = 8                     # Number of actions to execute open-loop before requerying policy
    unnorm_key: Union[str, Path] = ""                # Action un-normalization key

    load_in_8bit: bool = False                       # (For OpenVLA only) Load with 8-bit quantization
    load_in_4bit: bool = False                       # (For OpenVLA only) Load with 4-bit quantization

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = TaskSuite.LIBERO_SPATIAL  # Task suite
    num_steps_wait: int = 10                         # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 50                    # Number of rollouts per task
    initial_states_path: str = "DEFAULT"             # "DEFAULT", or path to initial states JSON file
    env_img_res: int = 256                           # Resolution for environment images (not policy input resolution)

    #################################################################################################################
    # Utils
    #################################################################################################################
    run_id_note: Optional[str] = None                # Extra note to add to end of run ID for logging
    local_log_dir: str = "./experiments/logs"        # Local directory for eval logs

    use_wandb: bool = False                          # Whether to also log results in Weights & Biases
    wandb_entity: str = "your-wandb-entity"          # Name of WandB entity
    wandb_project: str = "your-wandb-project"        # Name of WandB project

    seed: int = 7                                    # Random Seed (for reproducibility)

    # fmt: on
    save_version: str = "vla-adapter"                # version of
    use_pro_version: bool = True                     # encourage to use the pro models we released.
    phase: str = "Inference"



def validate_config(cfg: GenerateConfig) -> None:
    """Validate configuration parameters."""
    assert cfg.pretrained_checkpoint is not None, "pretrained_checkpoint must not be None!"

    if "image_aug" in str(cfg.pretrained_checkpoint):
        assert cfg.center_crop, "Expecting `center_crop==True` because model was trained with image augmentations!"

    assert not (cfg.load_in_8bit and cfg.load_in_4bit), "Cannot use both 8-bit and 4-bit quantization!"

    # Validate task suite
    assert cfg.task_suite_name in [suite.value for suite in TaskSuite], f"Invalid task suite: {cfg.task_suite_name}"



def initialize_model(cfg: GenerateConfig):
    """Initialize model and associated components."""
    # Load model
    model = get_model(cfg)
    model.set_version(cfg.save_version)

    # ===========================================================
    # SmoothQuant / Static INT8 Quantization
    # ===========================================================
    print(f"\n[Quant] QUANT_MODE = {QUANT_MODE}, QUANT_VISION = {QUANT_VISION}")

    if QUANT_MODE == "load_int8":
        # -------------------------------------------------------
        # 加载已保存的 INT8 模型（跳过校准）：
        #   1. smooth（必须和校准时一样）
        #   2. 加载 INT8 state_dict
        # -------------------------------------------------------
        from int8_vla import smooth_qwen, convert_to_int8, StaticInt8Linear

        act_scales = torch.load(ACT_SCALES_PATH, map_location="cpu")
        smooth_qwen(model, act_scales, alpha=SMOOTH_ALPHA)

        print(f"[load_int8] 加载 INT8 权重: {INT8_MODEL_PATH}")
        # 先把结构转好，再灌权重
        act_minmax_dummy = torch.load(INT8_MODEL_PATH + ".minmax.pt", map_location="cpu")
        convert_to_int8(model, act_minmax_dummy)
        state_dict = torch.load(INT8_MODEL_PATH, map_location="cpu")
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"  ⚠ missing keys: {len(missing)}")
        if unexpected:
            print(f"  ⚠ unexpected keys: {len(unexpected)}")
        print(f"[load_int8] 加载完成，直接跑 eval")

    elif QUANT_MODE == "static_int8":
        # -------------------------------------------------------
        # 真 INT8 静态量化（首次跑，需要校准）：
        #   这里只做 smooth + 挂 hook。
        #   校准在 eval_libero() 里做（跑 CALIB_EPISODES 个 episode，
        #   hook 自动收集激活 min/max）。
        #   校准完摘 hook，转换为 StaticInt8Linear，保存。
        # -------------------------------------------------------
        from int8_vla import smooth_qwen, attach_minmax_hooks

        act_scales = torch.load(ACT_SCALES_PATH, map_location="cpu")
        smooth_qwen(model, act_scales, alpha=SMOOTH_ALPHA)
        print(f"[static_int8] Smooth applied (alpha={SMOOTH_ALPHA})")

        # 挂 hook，eval_libero() 里校准完再摘
        _act_minmax, _hooks = attach_minmax_hooks(model)
        # 把 hook 相关对象存到 model 上，方便 eval_libero() 取用
        model._int8_act_minmax = _act_minmax
        model._int8_hooks = _hooks
        print(f"[static_int8] Hooks attached, will calibrate for {CALIB_EPISODES} episodes")

    elif QUANT_MODE != "fp16":
        # -------------------------------------------------------
        # Fake Quant 路径（原有逻辑不变）
        # -------------------------------------------------------
        from smoothquant_vla_full import smooth_qwen, quantize_qwen, quantize_vision_projector

        if QUANT_MODE == "smoothquant":
            act_scales = torch.load(ACT_SCALES_PATH, map_location="cpu")
            smooth_qwen(model, act_scales, alpha=SMOOTH_ALPHA)
            print(f"[SmoothQuant] Smooth applied (alpha={SMOOTH_ALPHA})")

        quantize_qwen(model)
        print(f"[SmoothQuant] Qwen2.5 quantization applied! Mode={QUANT_MODE}")

        if QUANT_VISION:
            quantize_vision_projector(model)
            print("[SmoothQuant] ViT + Projector Naive W8A8 applied!")
        else:
            print("[SmoothQuant] ViT + Projector kept FP16")

    elif QUANT_VISION:
        from smoothquant_vla_full import quantize_vision_projector
        quantize_vision_projector(model)
        print("[SmoothQuant] Only ViT + Projector quantized (Naive W8A8)")
    else:
        print("[Quant] Mode=fp16, no quantization applied")
    # ===========================================================

    # Load proprio projector if needed
    proprio_projector = None
    if cfg.use_proprio:
        proprio_projector = get_proprio_projector(
            cfg,
            model.llm_dim,
            proprio_dim=8,
        )

    # Load action head if needed
    action_head = None
    if cfg.use_l1_regression:
        action_head = get_action_head(cfg, model.llm_dim)

    noisy_action_projector = None

    processor = None
    if cfg.model_family == "openvla":
        processor = get_processor(cfg)
        check_unnorm_key(cfg, model)

    return model, action_head, proprio_projector, noisy_action_projector, processor


def check_unnorm_key(cfg: GenerateConfig, model) -> None:
    """Check that the model contains the action un-normalization key."""
    unnorm_key = cfg.task_suite_name

    if unnorm_key not in model.norm_stats and f"{unnorm_key}_no_noops" in model.norm_stats:
        unnorm_key = f"{unnorm_key}_no_noops"

    assert unnorm_key in model.norm_stats, f"Action un-norm key {unnorm_key} not found in VLA `norm_stats`!"
    cfg.unnorm_key = unnorm_key



def setup_logging(cfg: GenerateConfig):
    """Set up logging to file and optionally to wandb."""
    vision_tag = "+vision" if QUANT_VISION else ""
    run_id = f"EVAL-{cfg.task_suite_name}-{QUANT_MODE}{vision_tag}-{DATE_TIME}"
    if cfg.run_id_note is not None:
        run_id += f"--{cfg.run_id_note}"

    os.makedirs(cfg.local_log_dir, exist_ok=True)
    local_log_filepath = os.path.join(cfg.local_log_dir, run_id + ".txt")
    log_file = open(local_log_filepath, "w")
    logger.info(f"Logging to local log file: {local_log_filepath}")

    if cfg.use_wandb:
        wandb.init(
            entity=cfg.wandb_entity,
            project=cfg.wandb_project,
            name=run_id,
        )

    return log_file, local_log_filepath, run_id



def log_message(message: str, log_file=None):
    """Log a message to console and optionally to a log file."""
    logger.info(message)
    if log_file:
        log_file.write(message + "\n")
        log_file.flush()



def load_initial_states(cfg: GenerateConfig, task_suite, task_id: int, log_file=None):
    """Load initial states for the given task."""
    initial_states = task_suite.get_task_init_states(task_id)

    if cfg.initial_states_path != "DEFAULT":
        with open(cfg.initial_states_path, "r") as f:
            all_initial_states = json.load(f)
        log_message(f"Using initial states from {cfg.initial_states_path}", log_file)
        return initial_states, all_initial_states
    else:
        log_message("Using default initial states", log_file)
        return initial_states, None



def prepare_observation(obs, resize_size):
    """Prepare observation for policy input."""
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

    return observation, img



def process_action(action, model_family):
    """Process action before sending to environment."""
    action = normalize_gripper_action(action, binarize=True)

    if model_family == "openvla":
        action = invert_gripper_action(action)

    return action



def run_episode(
    cfg: GenerateConfig,
    env,
    task_description: str,
    model,
    resize_size,
    processor=None,
    action_head=None,
    proprio_projector=None,
    noisy_action_projector=None,
    initial_state=None,
    log_file=None,
):
    """Run a single episode in the environment."""
    env.reset()

    if initial_state is not None:
        obs = env.set_init_state(initial_state)
    else:
        obs = env.get_observation()

    if cfg.num_open_loop_steps != NUM_ACTIONS_CHUNK:
        print(f"WARNING: cfg.num_open_loop_steps ({cfg.num_open_loop_steps}) does not match the NUM_ACTIONS_CHUNK "
               "{NUM_ACTIONS_CHUNK} constant defined in prismatic.vla.constants!")
    action_queue = deque(maxlen=cfg.num_open_loop_steps)

    t = 0
    replay_images = []
    max_steps = TASK_MAX_STEPS[cfg.task_suite_name]

    success = False
    try:
        while t < max_steps + cfg.num_steps_wait:
            if t < cfg.num_steps_wait:
                obs, reward, done, info = env.step(get_libero_dummy_action(cfg.model_family))
                t += 1
                continue

            observation, img = prepare_observation(obs, resize_size)
            replay_images.append(img)

            if len(action_queue) == 0:
                actions = get_action(
                    cfg,
                    model,
                    observation,
                    task_description,
                    processor=processor,
                    action_head=action_head,
                    proprio_projector=proprio_projector,
                    noisy_action_projector=noisy_action_projector,
                    use_film=cfg.use_film,
                    use_minivlm=cfg.use_minivlm
                )
                action_queue.extend(actions)

            action = action_queue.popleft()
            action = process_action(action, cfg.model_family)

            obs, reward, done, info = env.step(action.tolist())
            if done:
                success = True
                break
            t += 1

    except Exception as e:
        log_message(f"Episode error: {e}", log_file)

    return success, replay_images


# ===========================================================
# ★ 新增：static_int8 校准阶段 ★
# ===========================================================
def run_calibration(
    cfg: GenerateConfig,
    model,
    task_suite,
    resize_size,
    processor,
    action_head,
    proprio_projector,
    noisy_action_projector,
    num_calib_episodes: int,
):
    """
    校准阶段：跑 num_calib_episodes 个 episode，让 ObservedLinear
    的 observer 自动收集激活范围。不记录成功率，结果直接丢弃。

    episode 按 task 轮询分配：
      num_calib_episodes=10, num_tasks=10 → 每个 task 跑 1 个 episode
      num_calib_episodes=10, num_tasks=4  → task 0,1 各跑 3 个，task 2,3 各跑 2 个
    """
    print(f"\n[static_int8] ===== 校准阶段：跑 {num_calib_episodes} 个 episode =====")

    num_tasks = task_suite.n_tasks
    # 每个 task 分配的 episode 数（尽量均匀）
    base = num_calib_episodes // num_tasks
    remainder = num_calib_episodes % num_tasks
    episodes_per_task = [base + (1 if i < remainder else 0) for i in range(num_tasks)]

    total_done = 0
    for task_id in range(num_tasks):
        n_ep = episodes_per_task[task_id]
        if n_ep == 0:
            continue

        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description = get_libero_env(task, cfg.model_family, resolution=cfg.env_img_res)

        for ep in range(n_ep):
            initial_state = initial_states[ep % len(initial_states)]
            success, _ = run_episode(
                cfg, env, task_description, model, resize_size,
                processor, action_head, proprio_projector,
                noisy_action_projector, initial_state, log_file=None,
            )
            total_done += 1
            print(f"  [calibration] {total_done}/{num_calib_episodes} episodes done "
                  f"(task {task_id}, ep {ep}), success={success}")

            # 检查 observer 是否收到数据
            for name, module in model.named_modules():
                if hasattr(module, 'act_observer'):
                    print(f"  [debug] first ObservedLinear '{name}': n_observed={module.act_observer.n_observed}")
                    break

        env.close()
        del env

    print(f"[static_int8] 校准完成，共 {total_done} 个 episode")
# ===========================================================


def run_task(
    cfg: GenerateConfig,
    task_suite,
    task_id: int,
    model,
    resize_size,
    processor=None,
    action_head=None,
    proprio_projector=None,
    noisy_action_projector=None,
    total_episodes=0,
    total_successes=0,
    log_file=None,
    save_version=None
):
    """Run evaluation for a single task."""
    task = task_suite.get_task(task_id)

    initial_states, all_initial_states = load_initial_states(cfg, task_suite, task_id, log_file)

    env, task_description = get_libero_env(task, cfg.model_family, resolution=cfg.env_img_res)

    task_episodes, task_successes = 0, 0
    for episode_idx in tqdm.tqdm(range(cfg.num_trials_per_task)):
        log_message(f"\nTask: {task_description}", log_file)

        if cfg.initial_states_path == "DEFAULT":
            initial_state = initial_states[episode_idx]
        else:
            initial_states_task_key = task_description.replace(" ", "_")
            episode_key = f"demo_{episode_idx}"

            if not all_initial_states[initial_states_task_key][episode_key]["success"]:
                log_message(f"Skipping task {task_id} episode {episode_idx} due to failed expert demo!", log_file)
                continue

            initial_state = np.array(all_initial_states[initial_states_task_key][episode_key]["initial_state"])

        log_message(f"Starting episode {task_episodes + 1}...", log_file)

        success, replay_images = run_episode(
            cfg,
            env,
            task_description,
            model,
            resize_size,
            processor,
            action_head,
            proprio_projector,
            noisy_action_projector,
            initial_state,
            log_file,
        )

        task_episodes += 1
        total_episodes += 1
        if success:
            task_successes += 1
            total_successes += 1

        save_rollout_video(
            replay_images, total_episodes, success=success, task_description=task_description, log_file=log_file, save_version=save_version
        )

        log_message(f"Success: {success}", log_file)
        log_message(f"# episodes completed so far: {total_episodes}", log_file)
        log_message(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)", log_file)

    task_success_rate = float(task_successes) / float(task_episodes) if task_episodes > 0 else 0
    total_success_rate = float(total_successes) / float(total_episodes) if total_episodes > 0 else 0

    log_message(f"Current task success rate: {task_success_rate}", log_file)
    log_message(f"Current total success rate: {total_success_rate}", log_file)

    env.close()
    del env

    if cfg.use_wandb:
        wandb.log(
            {
                f"success_rate/{task_description}": task_success_rate,
                f"num_episodes/{task_description}": task_episodes,
            }
        )

    return total_episodes, total_successes



@draccus.wrap()
def eval_libero(cfg: GenerateConfig) -> float:
    """Main function to evaluate a trained policy on LIBERO benchmark tasks."""
    validate_config(cfg)
    set_seed_everywhere(cfg.seed)

    model, action_head, proprio_projector, noisy_action_projector, processor = initialize_model(cfg)

    resize_size = get_image_resize_size(cfg)
    log_file, local_log_filepath, run_id = setup_logging(cfg)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    num_tasks = task_suite.n_tasks

    log_message(f"Task suite: {cfg.task_suite_name}", log_file)
    log_message(f"Quantization mode: {QUANT_MODE}, Vision quantized: {QUANT_VISION}", log_file)

    # ===========================================================
    # ★ static_int8：校准阶段，校准完转换为真 INT8 ★
    # ===========================================================
    if QUANT_MODE == "static_int8":
        run_calibration(
            cfg, model, task_suite, resize_size,
            processor, action_head, proprio_projector, noisy_action_projector,
            num_calib_episodes=CALIB_EPISODES,
        )

        # 摘 hook
        from int8_vla import remove_hooks, convert_to_int8, verify_int8_model, print_model_stats
        remove_hooks(model._int8_hooks)

        # 转换为真 INT8
        print(f"\n[static_int8] ===== 转换为真 INT8 =====")
        n = convert_to_int8(model, model._int8_act_minmax)
        print_model_stats(model)
        verify_int8_model(model, num_show=3)
        log_message(f"[static_int8] Converted {n} layers to StaticInt8Linear", log_file)

        # 保存 INT8 权重 + minmax（load_int8 时用）
        os.makedirs(os.path.dirname(INT8_MODEL_PATH) or ".", exist_ok=True)
        torch.save(model.state_dict(), INT8_MODEL_PATH)
        torch.save(model._int8_act_minmax, INT8_MODEL_PATH + ".minmax.pt")
        size_mb = os.path.getsize(INT8_MODEL_PATH) / (1024 * 1024)
        print(f"[static_int8] 权重已保存: {INT8_MODEL_PATH} ({size_mb:.1f} MB)")
        log_message(f"[static_int8] Saved to {INT8_MODEL_PATH} ({size_mb:.1f} MB)", log_file)
    # ===========================================================

    # 正式 eval（所有 mode 都走这里）
    total_episodes, total_successes = 0, 0
    for task_id in tqdm.tqdm(range(num_tasks)):
        total_episodes, total_successes = run_task(
            cfg,
            task_suite,
            task_id,
            model,
            resize_size,
            processor,
            action_head,
            proprio_projector,
            noisy_action_projector,
            total_episodes,
            total_successes,
            log_file,
            cfg.save_version
        )

    final_success_rate = float(total_successes) / float(total_episodes) if total_episodes > 0 else 0

    log_message("Final results:", log_file)
    log_message(f"Quantization mode: {QUANT_MODE}, Vision quantized: {QUANT_VISION}", log_file)
    log_message(f"Total episodes: {total_episodes}", log_file)
    log_message(f"Total successes: {total_successes}", log_file)
    log_message(f"Overall success rate: {final_success_rate:.4f} ({final_success_rate * 100:.1f}%)", log_file)

    if cfg.use_wandb:
        wandb.log(
            {
                "success_rate/total": final_success_rate,
                "num_episodes/total": total_episodes,
            }
        )
        wandb.save(local_log_filepath)

    if log_file:
        log_file.close()

    return final_success_rate



if __name__ == "__main__":
    eval_libero()