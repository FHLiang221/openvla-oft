"""
run_libero_eval.py

Evaluates a trained policy in a LIBERO simulation benchmark task suite.
"""

import json
import logging
import os
import random
import sys
from collections import deque
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, Union

import draccus
import numpy as np
import tqdm
import yaml
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


# Define task suite constants
class TaskSuite(str, Enum):
    LIBERO_SPATIAL = "libero_spatial"
    LIBERO_OBJECT = "libero_object"
    LIBERO_GOAL = "libero_goal"
    LIBERO_10 = "libero_10"
    LIBERO_90 = "libero_90"


# Define max steps for each task suite
TASK_MAX_STEPS = {
    TaskSuite.LIBERO_SPATIAL: 220,  # longest training demo has 193 steps
    TaskSuite.LIBERO_OBJECT: 280,  # longest training demo has 254 steps
    TaskSuite.LIBERO_GOAL: 300,  # longest training demo has 270 steps
    TaskSuite.LIBERO_10: 520,  # longest training demo has 505 steps
    TaskSuite.LIBERO_90: 400,  # longest training demo has 373 steps
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
    use_diffusion: bool = False                      # If True, uses continuous action head with diffusion modeling objective (DDIM)
    num_diffusion_steps_train: int = 50              # (When `diffusion==True`) Number of diffusion steps used for training
    num_diffusion_steps_inference: int = 50          # (When `diffusion==True`) Number of diffusion steps used for inference
    use_film: bool = False                           # If True, uses FiLM to infuse language inputs into visual features
    num_images_in_input: int = 2                     # Number of images in the VLA input (default: 1)
    use_proprio: bool = True                         # Whether to include proprio state in input

    center_crop: bool = True                         # Center crop? (if trained w/ random crop image aug)
    num_open_loop_steps: int = 8                     # Number of actions to execute open-loop before requerying policy

    lora_rank: int = 32                              # Rank of LoRA weight matrix (MAKE SURE THIS MATCHES TRAINING!)

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
    
    #################################################################################################################
    # Holdout prompt evaluation parameters
    #################################################################################################################
    holdout_prompts_yaml: Optional[str] = None       # Path to YAML file with holdout prompts for evaluation

    # fmt: on


def validate_config(cfg: GenerateConfig) -> None:
    """Validate configuration parameters."""
    assert cfg.pretrained_checkpoint is not None, "pretrained_checkpoint must not be None!"

    if "image_aug" in str(cfg.pretrained_checkpoint):
        assert cfg.center_crop, "Expecting `center_crop==True` because model was trained with image augmentations!"

    assert not (cfg.load_in_8bit and cfg.load_in_4bit), "Cannot use both 8-bit and 4-bit quantization!"

    # Validate task suite
    assert cfg.task_suite_name in [suite.value for suite in TaskSuite], f"Invalid task suite: {cfg.task_suite_name}"


def load_holdout_prompts(holdout_prompts_yaml: str) -> dict:
    """Load holdout prompts from YAML file and structure for systematic evaluation."""
    if not holdout_prompts_yaml or not os.path.exists(holdout_prompts_yaml):
        return {}
    
    with open(holdout_prompts_yaml, 'r') as f:
        holdout_prompts = yaml.safe_load(f)
    
    print(f"🎯 Loaded holdout prompts from: {holdout_prompts_yaml}")
    for task, prompts in holdout_prompts.items():
        print(f"  {task}: {len(prompts)} adversarial prompts")
    
    return holdout_prompts


def get_holdout_task_description(original_task_description: str, holdout_prompts: dict) -> str:
    """Get a random holdout prompt for the given task, or return original if no holdout available."""
    # Normalize task description to match YAML keys
    normalized_task = original_task_description.lower().strip()
    
    if normalized_task in holdout_prompts and holdout_prompts[normalized_task]:
        # Randomly select one of the holdout prompts
        selected_prompt = random.choice(holdout_prompts[normalized_task])
        print(f"🔄 Using holdout prompt: '{selected_prompt}' (original: '{original_task_description}')")
        return selected_prompt
    
    # Return original if no holdout prompt available
    return original_task_description


def get_all_prompt_variants(original_task_description: str, holdout_prompts: dict) -> list:
    """Get all prompt variants for systematic evaluation: original + all adversarial prompts."""
    # Normalize task description to match YAML keys
    normalized_task = original_task_description.lower().strip()
    
    # Start with the original prompt
    variants = [("original", original_task_description)]
    
    # Add all adversarial prompts if available
    if normalized_task in holdout_prompts and holdout_prompts[normalized_task]:
        for i, adv_prompt in enumerate(holdout_prompts[normalized_task]):
            variants.append((f"adversarial_{i+1}", adv_prompt))
    
    return variants


def initialize_model(cfg: GenerateConfig):
    """Initialize model and associated components."""
    # Load model
    model = get_model(cfg)

    # Load proprio projector if needed
    proprio_projector = None
    if cfg.use_proprio:
        proprio_projector = get_proprio_projector(
            cfg,
            model.llm_dim,
            proprio_dim=8,  # 8-dimensional proprio for LIBERO
        )

    # Load action head if needed
    action_head = None
    if cfg.use_l1_regression or cfg.use_diffusion:
        action_head = get_action_head(cfg, model.llm_dim)

    # Load noisy action projector if using diffusion
    noisy_action_projector = None
    if cfg.use_diffusion:
        noisy_action_projector = get_noisy_action_projector(cfg, model.llm_dim)

    # Get OpenVLA processor if needed
    processor = None
    if cfg.model_family == "openvla":
        processor = get_processor(cfg)
        check_unnorm_key(cfg, model)

    return model, action_head, proprio_projector, noisy_action_projector, processor


def check_unnorm_key(cfg: GenerateConfig, model) -> None:
    """Check that the model contains the action un-normalization key."""
    # Initialize unnorm_key
    unnorm_key = cfg.task_suite_name

    # In some cases, the key must be manually modified (e.g. after training on a modified version of the dataset
    # with the suffix "_no_noops" in the dataset name)
    if unnorm_key not in model.norm_stats and f"{unnorm_key}_no_noops" in model.norm_stats:
        unnorm_key = f"{unnorm_key}_no_noops"

    assert unnorm_key in model.norm_stats, f"Action un-norm key {unnorm_key} not found in VLA `norm_stats`!"

    # Set the unnorm_key in cfg
    cfg.unnorm_key = unnorm_key


def setup_logging(cfg: GenerateConfig):
    """Set up logging to file and optionally to wandb."""
    # Create run ID
    run_id = f"EVAL-{cfg.task_suite_name}-{cfg.model_family}-{DATE_TIME}"
    if cfg.run_id_note is not None:
        run_id += f"--{cfg.run_id_note}"

    # Set up local logging
    os.makedirs(cfg.local_log_dir, exist_ok=True)
    local_log_filepath = os.path.join(cfg.local_log_dir, run_id + ".txt")
    log_file = open(local_log_filepath, "w")
    logger.info(f"Logging to local log file: {local_log_filepath}")

    # Initialize Weights & Biases logging if enabled
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
    # Get default initial states
    initial_states = task_suite.get_task_init_states(task_id)

    # If using custom initial states, load them from file
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
    # Get preprocessed images
    img = get_libero_image(obs)
    wrist_img = get_libero_wrist_image(obs)

    # Resize images to size expected by model
    img_resized = resize_image_for_policy(img, resize_size)
    wrist_img_resized = resize_image_for_policy(wrist_img, resize_size)

    # Prepare observations dict
    observation = {
        "full_image": img_resized,
        "wrist_image": wrist_img_resized,
        "state": np.concatenate(
            (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
        ),
    }

    return observation, img  # Return both processed observation and original image for replay


def process_action(action, model_family):
    """Process action before sending to environment."""
    # Normalize gripper action [0,1] -> [-1,+1] because the environment expects the latter
    action = normalize_gripper_action(action, binarize=True)

    # [OpenVLA] The dataloader flips the sign of the gripper action to align with other datasets
    # (0 = close, 1 = open), so flip it back (-1 = open, +1 = close) before executing the action
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
    # Reset environment
    env.reset()

    # Set initial state if provided
    if initial_state is not None:
        obs = env.set_init_state(initial_state)
    else:
        obs = env.get_observation()

    # Initialize action queue
    if cfg.num_open_loop_steps != NUM_ACTIONS_CHUNK:
        print(f"WARNING: cfg.num_open_loop_steps ({cfg.num_open_loop_steps}) does not match the NUM_ACTIONS_CHUNK "
              f"({NUM_ACTIONS_CHUNK}) constant defined in prismatic.vla.constants! For best performance (in terms of "
               "both speed and success rate), we recommend executing the full action chunk.")
    action_queue = deque(maxlen=cfg.num_open_loop_steps)

    # Setup
    t = 0
    replay_images = []
    max_steps = TASK_MAX_STEPS[cfg.task_suite_name]

    # Run episode
    success = False
    try:
        while t < max_steps + cfg.num_steps_wait:
            # Do nothing for the first few timesteps to let objects stabilize
            if t < cfg.num_steps_wait:
                obs, reward, done, info = env.step(get_libero_dummy_action(cfg.model_family))
                t += 1
                continue

            # Prepare observation
            observation, img = prepare_observation(obs, resize_size)
            replay_images.append(img)

            # If action queue is empty, requery model
            if len(action_queue) == 0:
                # Query model to get action
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
                )
                action_queue.extend(actions)

            # Get action from queue
            action = action_queue.popleft()

            # Process action
            action = process_action(action, cfg.model_family)

            # Execute action in environment
            obs, reward, done, info = env.step(action.tolist())
            if done:
                success = True
                break
            t += 1

    except Exception as e:
        log_message(f"Episode error: {e}", log_file)

    return success, replay_images


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
    holdout_prompts=None,
    run_id=None,
):
    """Run evaluation for a single task with all prompt variants."""
    # Get task
    task = task_suite.get_task(task_id)

    # Get initial states
    initial_states, all_initial_states = load_initial_states(cfg, task_suite, task_id, log_file)

    # Initialize environment and get original task description
    env, original_task_description = get_libero_env(task, cfg.model_family, resolution=cfg.env_img_res)

    # Get all prompt variants for systematic evaluation
    if holdout_prompts:
        prompt_variants = get_all_prompt_variants(original_task_description, holdout_prompts)
    else:
        prompt_variants = [("original", original_task_description)]

    log_message(f"\n{'='*80}", log_file)
    log_message(f"TASK {task_id}: {original_task_description}", log_file)
    log_message(f"Will evaluate {len(prompt_variants)} prompt variants, {cfg.num_trials_per_task} episodes each", log_file)
    log_message(f"Initial states available: {len(initial_states) if cfg.initial_states_path == 'DEFAULT' else 'custom'}", log_file)
    log_message(f"Prompt variants: {[v[0] for v in prompt_variants]}", log_file)
    log_message(f"{'='*80}", log_file)

    # Track results per prompt variant
    variant_results = {}
    task_total_episodes = 0
    task_total_successes = 0

    # Evaluate each prompt variant systematically
    for variant_type, task_description in prompt_variants:
        log_message(f"\n{'-'*60}", log_file)
        log_message(f"EVALUATING: {variant_type.upper()}", log_file)
        log_message(f"Prompt: {task_description}", log_file)
        log_message(f"{'-'*60}", log_file)

        variant_episodes = 0
        variant_successes = 0

        # Reset episode_idx for each variant to ensure same initial states are used
        episode_idx = 0
        with tqdm.tqdm(total=cfg.num_trials_per_task, desc=f"{variant_type}") as pbar:
            while variant_episodes < cfg.num_trials_per_task:
                # Handle initial state with bounds checking
                if cfg.initial_states_path == "DEFAULT":
                    # Use default initial state with bounds checking
                    if episode_idx >= len(initial_states):
                        log_message(f"ERROR: episode_idx {episode_idx} >= len(initial_states) {len(initial_states)}", log_file)
                        break
                    initial_state = initial_states[episode_idx]
                else:
                    # Get keys for fetching initial episode state from JSON
                    # IMPORTANT: Use original task description for lookup, not variant prompt
                    initial_states_task_key = original_task_description.replace(" ", "_")
                    episode_key = f"demo_{episode_idx}"

                    # Check if episode key exists
                    if (initial_states_task_key not in all_initial_states or 
                        episode_key not in all_initial_states[initial_states_task_key]):
                        log_message(f"Skipping task {task_id} episode {episode_idx} - episode key not found!", log_file)
                        episode_idx += 1
                        continue

                    # Skip episode if expert demonstration failed to complete the task
                    if not all_initial_states[initial_states_task_key][episode_key]["success"]:
                        log_message(f"Skipping task {task_id} episode {episode_idx} due to failed expert demo!", log_file)
                        episode_idx += 1
                        continue

                    # Get initial state
                    initial_state = np.array(all_initial_states[initial_states_task_key][episode_key]["initial_state"])
                
                episode_idx += 1

                log_message(f"Starting {variant_type} episode {variant_episodes + 1}...", log_file)

                # Run episode
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

                # Update counters
                variant_episodes += 1
                task_total_episodes += 1
                total_episodes += 1
                
                if success:
                    variant_successes += 1
                    task_total_successes += 1
                    total_successes += 1

                # Save replay video with task and variant info for better organization
                # Use format: task_id_variant_episode for logical grouping
                video_id = f"{task_id}_{variant_type}_{variant_episodes}"
                save_rollout_video(
                    replay_images, video_id, success=success, 
                    task_description=f"[Task {task_id}] [{variant_type}] {task_description}", log_file=log_file, run_id=run_id
                )

                # Log episode results
                log_message(f"Success: {success}", log_file)
                
                # Update progress bar
                pbar.update(1)

        # Calculate and store variant results
        variant_success_rate = float(variant_successes) / float(variant_episodes) if variant_episodes > 0 else 0
        variant_results[variant_type] = {
            "episodes": variant_episodes,
            "successes": variant_successes,
            "success_rate": variant_success_rate,
            "prompt": task_description
        }

        log_message(f"\n{variant_type.upper()} RESULTS:", log_file)
        log_message(f"Episodes: {variant_episodes}", log_file)
        log_message(f"Successes: {variant_successes}", log_file)
        log_message(f"Success Rate: {variant_success_rate:.4f} ({variant_success_rate * 100:.1f}%)", log_file)

    # Log overall task results
    task_overall_success_rate = float(task_total_successes) / float(task_total_episodes) if task_total_episodes > 0 else 0
    total_success_rate = float(total_successes) / float(total_episodes) if total_episodes > 0 else 0

    log_message(f"\n{'='*80}", log_file)
    log_message(f"TASK {task_id} SUMMARY:", log_file)
    for variant_type, results in variant_results.items():
        log_message(f"  {variant_type}: {results['successes']}/{results['episodes']} ({results['success_rate']*100:.1f}%)", log_file)
    log_message(f"  OVERALL: {task_total_successes}/{task_total_episodes} ({task_overall_success_rate*100:.1f}%)", log_file)
    log_message(f"TOTAL SUCCESS RATE: {total_successes}/{total_episodes} ({total_success_rate*100:.1f}%)", log_file)
    log_message(f"{'='*80}", log_file)

    # Log to wandb if enabled
    if cfg.use_wandb:
        wandb_data = {}
        # Log individual variant results
        for variant_type, results in variant_results.items():
            wandb_data[f"success_rate/{original_task_description}_{variant_type}"] = results["success_rate"]
            wandb_data[f"num_episodes/{original_task_description}_{variant_type}"] = results["episodes"]
        
        # Log overall task results
        wandb_data[f"success_rate/{original_task_description}_overall"] = task_overall_success_rate
        wandb_data[f"num_episodes/{original_task_description}_overall"] = task_total_episodes
        
        wandb.log(wandb_data)

    return total_episodes, total_successes


@draccus.wrap()
def eval_libero(cfg: GenerateConfig) -> float:
    """Main function to evaluate a trained policy on LIBERO benchmark tasks."""
    # Validate configuration
    validate_config(cfg)

    # Set random seed
    set_seed_everywhere(cfg.seed)

    # Initialize model and components
    model, action_head, proprio_projector, noisy_action_projector, processor = initialize_model(cfg)

    # Get expected image dimensions
    resize_size = get_image_resize_size(cfg)

    # Setup logging
    log_file, local_log_filepath, run_id = setup_logging(cfg)

    # Load holdout prompts if specified
    holdout_prompts = None
    if cfg.holdout_prompts_yaml:
        holdout_prompts = load_holdout_prompts(cfg.holdout_prompts_yaml)
        if holdout_prompts:
            log_message(f"Loaded holdout prompts from: {cfg.holdout_prompts_yaml}", log_file)
        else:
            log_message(f"No holdout prompts found in: {cfg.holdout_prompts_yaml}", log_file)

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    num_tasks = task_suite.n_tasks

    log_message(f"Task suite: {cfg.task_suite_name}", log_file)

    # Start evaluation
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
            holdout_prompts,
            run_id,
        )

    # Calculate final success rate
    final_success_rate = float(total_successes) / float(total_episodes) if total_episodes > 0 else 0

    # Log final results
    log_message("Final results:", log_file)
    log_message(f"Total episodes: {total_episodes}", log_file)
    log_message(f"Total successes: {total_successes}", log_file)
    log_message(f"Overall success rate: {final_success_rate:.4f} ({final_success_rate * 100:.1f}%)", log_file)

    # Log to wandb if enabled
    if cfg.use_wandb:
        wandb.log(
            {
                "success_rate/total": final_success_rate,
                "num_episodes/total": total_episodes,
            }
        )
        wandb.save(local_log_filepath)

    # Close log file
    if log_file:
        log_file.close()

    return final_success_rate


if __name__ == "__main__":
    eval_libero()





#Normal usage example:
"""
CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint "/home/freddie/project/openvla-oft/checkpoints/openvla-7b-full" \
  --task_suite_name libero_spatial \
  --center_crop True \
  --num_trials_per_task 50 \
  --use_l1_regression False \
  --use_diffusion False \
  --use_film False \
  --num_images_in_input 2 \
  --use_proprio False \
  --lora_rank 32 \
  --use_wandb True \
  --wandb_entity freddieliang-usc \
  --wandb_project openvla_repl \
  --run_id_note liberoSpatial_BASELINE_eval_adversarial_holdout \
  --holdout_prompts_yaml ./ERT_eval_tasks.yaml
"""
# one liner:
# CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py --pretrained_checkpoint "/home/freddie/project/openvla-oft/checkpoints/openvla-7b-full" --task_suite_name libero_spatial --center_crop True --num_trials_per_task 50 --use_l1_regression False --use_diffusion False --use_film False --num_images_in_input 2 --use_proprio False --lora_rank 32 --use_wandb True --wandb_entity freddieliang-usc --wandb_project openvla_repl --run_id_note liberoSpatial_BASELINE_eval_adversarial_holdout --holdout_prompts_yaml ./ERT_eval_tasks.yaml





# Example with holdout prompts:
"""
CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
      --pretrained_checkpoint "/home/freddie/project/openvla-oft/checkpoints/libero_spatial_oft_ERT64/openvla-7b+libero_spatial_no_no
  ops+b64+lr-0.0005+lora-r32+dropout-0.0--image_aug--liberoSpatial_oft_ERT64--5000_chkpt" \
      --task_suite_name libero_spatial \
      --center_crop True \
      --num_trials_per_task 50 \
      --use_l1_regression True \
      --use_diffusion False \
      --use_film False \
      --num_images_in_input 2 \
      --use_proprio True \
      --lora_rank 32 \
      --use_wandb True \
      --wandb_entity freddieliang-usc \
      --wandb_project openvla_repl \
      --run_id_note liberoSpatial_oft_eval_adversarial_holdout \
      --holdout_prompts_yaml ./ERT_eval_tasks.yaml
"""
# one liner ERT64 eval:
# CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py --pretrained_checkpoint ./checkpoints/libero_spatial_oft_ERT64/openvla-7b+libero_spatial_no_noops+b64+lr-0.0005+lora-r32+dropout-0.0--image_aug--liberoSpatial_oft_ERT64--5000_chkpt --task_suite_name libero_spatial --center_crop True --num_trials_per_task 50 --use_l1_regression True --use_diffusion False --use_film False --num_images_in_input 2 --use_proprio True --lora_rank 32 --use_wandb True --wandb_entity freddieliang-usc --wandb_project openvla_repl --run_id_note liberoSpatial_oft_eval_adversarial_holdout --holdout_prompts_yaml ./ERT_eval_tasks.yaml

# one liner QD eval:
# CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py --pretrained_checkpoint ./checkpoints/libero_spatial_oft_QD64_Repeat/openvla-7b+libero_spatial_no_noops+b64+lr-0.0005+lora-r32+dropout-0.0--image_aug--liberoSpatial_oft_QD64_Repeat--5000_chkpt --task_suite_name libero_spatial --center_crop True --num_trials_per_task 50 --use_l1_regression True --use_diffusion False --use_film False --num_images_in_input 2 --use_proprio True --lora_rank 32 --use_wandb True --wandb_entity freddieliang-usc --wandb_project openvla_repl --run_id_note liberoSpatial_oft_QD64_eval_QD_holdout --holdout_prompts_yaml ./QD_eval_tasks.yaml





# Moojink's evaluation example:
"""
CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
    --pretrained_checkpoint moojink/openvla-7b-oft-finetuned-libero-spatial \
    --task_suite_name libero_spatial \
    --center_crop True \
    --num_trials_per_task 50 \
    --use_l1_regression True \
    --use_diffusion False \
    --use_film False \
    --num_images_in_input 2 \
    --use_proprio True \
    --lora_rank 32 \
    --use_wandb True \
    --wandb_entity freddieliang-usc \
    --wandb_project openvla_repl \
    --run_id_note liberoSpatial_moojink_eval_adversarial_holdout \
    --holdout_prompts_yaml ./ERT_eval_tasks.yaml
"""
#one liner ERT eval:
# CUDA_VISIBLE_DEVICES=1 python experiments/robot/libero/run_libero_eval.py --pretrained_checkpoint moojink/openvla-7b-oft-finetuned-libero-spatial --task_suite_name libero_spatial --center_crop True --num_trials_per_task 50 --use_l1_regression True --use_diffusion False --use_film False --num_images_in_input 2 --use_proprio True --lora_rank 32 --use_wandb True --wandb_entity freddieliang-usc --wandb_project openvla_repl --run_id_note liberoSpatial_moojink_eval_adversarial_holdout --holdout_prompts_yaml ./ERT_eval_tasks.yaml

#one liner QD eval:
# CUDA_VISIBLE_DEVICES=1 python experiments/robot/libero/run_libero_eval.py --pretrained_checkpoint moojink/openvla-7b-oft-finetuned-libero-spatial --task_suite_name libero_spatial --center_crop True --num_trials_per_task 50 --use_l1_regression True --use_diffusion False --use_film False --num_images_in_input 2 --use_proprio True --lora_rank 32 --use_wandb True --wandb_entity freddieliang-usc --wandb_project openvla_repl --run_id_note liberoSpatial_moojink_eval_QD_holdout --holdout_prompts_yaml ./QD_eval_tasks.yaml