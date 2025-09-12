"""
run_simpler_env_eval.py

Evaluates a trained policy in SimplerEnv environments.
"""

import json
import logging
import os
import random
import sys
import time
from collections import deque
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, Union

import draccus
import numpy as np
import tqdm
import yaml
from PIL import Image

import wandb

# Append current directory so that interpreter can find experiments.robot
sys.path.append("../..")
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

# SimplerEnv imports
import simpler_env
from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict


# Define SimplerEnv task constants
class TaskSuite(str, Enum):
    PICK_TASKS = "pick"
    DRAWER_TASKS = "drawer"
    MOVE_NEAR_TASKS = "move_near"
    PLACE_TASKS = "place"
    ALL_TASKS = "all"
    OPENVLA_BENCHMARK = "openvla_benchmark"  # Matches OpenVLA paper evaluation


# Define available SimplerEnv tasks (based on tasks.md)
SIMPLER_ENV_TASKS = {
    TaskSuite.PICK_TASKS: [
        "google_robot_pick_horizontal_coke_can",
        "google_robot_pick_vertical_coke_can", 
        "google_robot_pick_standing_coke_can",
        "google_robot_pick_coke_can",
        "google_robot_pick_object"
    ],
    TaskSuite.DRAWER_TASKS: [
        # Opening drawers
        "google_robot_open_top_drawer",
        "google_robot_open_middle_drawer", 
        "google_robot_open_bottom_drawer",
        # Closing drawers  
        "google_robot_close_top_drawer",
        "google_robot_close_middle_drawer",
        "google_robot_close_bottom_drawer"
    ],
    TaskSuite.MOVE_NEAR_TASKS: [
        "google_robot_move_near"
    ],
    TaskSuite.PLACE_TASKS: [
        "google_robot_place_apple_in_closed_top_drawer",
        "google_robot_place_in_closed_drawer"
    ],
    TaskSuite.OPENVLA_BENCHMARK: [
        # In-distribution tasks (matching paper as closely as possible)
        "google_robot_pick_coke_can",                    # 1. Pick Coke Can
        "google_robot_move_near",                        # 2. Move Apple near Green Can (closest match)
        "google_robot_move_near",                        # 3. Move Blue Chip Bag near Apple (closest match)  
        "google_robot_pick_standing_coke_can",           # 4. Place Coke Can Upright (closest match)
        "google_robot_open_middle_drawer",               # 5. Open Middle Drawer
        
        # OOD tasks (best available matches in SimplerEnv)
        "google_robot_move_near",                        # 6. Move Orange near Brown Chip Bag (object variation)
        "google_robot_pick_object",                      # 7. Pick Pepsi Can (object variation) 
        "google_robot_pick_object",                      # 8. Pick Banana (object variation)
        "google_robot_pick_object",                      # 9. Pick Green Cup (object variation)
        "google_robot_place_in_closed_drawer",           # 10. Place Apple on Plate (placement variation)
        "google_robot_place_in_closed_drawer",           # 11. Place Banana in Pan (placement variation)
        "google_robot_move_near",                        # 12. Move Coke Can to Taylor Swift (spatial variation)
    ]
}

# Define max steps for SimplerEnv tasks (can be adjusted based on task complexity)
TASK_MAX_STEPS = {
    TaskSuite.PICK_TASKS: 200,
    TaskSuite.DRAWER_TASKS: 250,
    TaskSuite.MOVE_NEAR_TASKS: 200,
    TaskSuite.PLACE_TASKS: 300,
    TaskSuite.ALL_TASKS: 300,
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
    num_images_in_input: int = 1                     # Number of images in the VLA input (default: 1)
    use_proprio: bool = True                         # Whether to include proprio state in input

    center_crop: bool = True                         # Center crop? (if trained w/ random crop image aug)
    num_open_loop_steps: int = 8                     # Number of actions to execute open-loop before requerying policy

    lora_rank: int = 32                              # Rank of LoRA weight matrix (MAKE SURE THIS MATCHES TRAINING!)

    unnorm_key: Union[str, Path] = ""                # Action un-normalization key

    load_in_8bit: bool = False                       # (For OpenVLA only) Load with 8-bit quantization
    load_in_4bit: bool = False                       # (For OpenVLA only) Load with 4-bit quantization

    #################################################################################################################
    # SimplerEnv environment-specific parameters
    #################################################################################################################
    task_suite_name: str = TaskSuite.ALL_TASKS       # Task suite to evaluate
    task_name: Optional[str] = None                  # Specific task name (overrides task_suite_name if provided)
    num_steps_wait: int = 10                         # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 20                    # Number of rollouts per task
    
    # OpenVLA benchmark specific settings
    openvla_benchmark_mode: bool = False             # Use OpenVLA paper evaluation protocol
    compute_stderr: bool = False                     # Compute standard error for benchmark comparison
    env_img_res: int = 224                           # Resolution for environment images

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

    # Validate task suite or specific task
    if cfg.task_name is None:
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


def get_task_instruction(env, info, env_name: str) -> str:
    """Get task instruction from environment, similar to your collection script."""
    if "target_drawer" in info:  # any drawer task
        drawer = info["target_drawer"]
        obj = info.get("object_name")

        # recognise **all** place-drawer envs, even custom names
        if obj is None and "place" in env_name.lower():
            if getattr(env, "model_id", None):
                obj = env._get_instruction_obj_name(env.model_id)

        if obj:  # place-drawer
            return f"place {obj} into {drawer} drawer"

        # open-drawer / close-drawer
        return env.get_language_instruction()

    # pick / move-near / ...
    return env.get_language_instruction()


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
            proprio_dim=8,  # 8-dimensional proprio for SimplerEnv (EEF_state + gripper_state)
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
    # Initialize unnorm_key for SimplerEnv
    unnorm_key = "simpler_env_switch_dataset"

    assert unnorm_key in model.norm_stats, f"Action un-norm key {unnorm_key} not found in VLA `norm_stats`!"

    # Set the unnorm_key in cfg
    cfg.unnorm_key = unnorm_key


def setup_logging(cfg: GenerateConfig):
    """Set up logging to file and optionally to wandb."""
    # Create run ID
    task_name = cfg.task_name if cfg.task_name else cfg.task_suite_name
    run_id = f"EVAL-SimplerEnv-{task_name}-{cfg.model_family}-{DATE_TIME}"
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


def prepare_observation(obs, env, resize_size):
    """Prepare observation for policy input."""
    # Get RGB image from SimplerEnv observation
    img = get_image_from_maniskill2_obs_dict(env, obs)
    
    # Resize image to size expected by model
    img_resized = resize_image_for_policy(img, resize_size)
    
    # Get robot state - SimplerEnv uses base_pose which should be 7D or 8D
    robot_pose = obs["agent"]["base_pose"]
    
    # Ensure 8D state (pad if necessary to match training format)
    if len(robot_pose) == 7:
        robot_pose = np.append(robot_pose, 0.0)  # Pad to 8D
    
    # Split into EEF_state and gripper_state as expected by OpenVLA config
    eef_state = robot_pose[:6]  # First 6 elements (x, y, z, roll, pitch, yaw)
    gripper_state = robot_pose[6:7]  # 7th element (gripper)
    
    # Combine for full 8D proprioception
    full_state = np.concatenate([eef_state, gripper_state, [0.0]])  # Pad to 8D total
    
    # Prepare observations dict
    observation = {
        "image": img_resized,  # SimplerEnv uses single primary image
        "state": full_state,
    }

    return observation, img  # Return both processed observation and original image for replay


def process_action(action, model_family):
    """Process action before sending to environment."""
    # SimplerEnv expects 7D actions: [dx, dy, dz, drx, dry, drz, gripper]
    # Ensure action is 7D
    if len(action) > 7:
        action = action[:7]
    
    # Normalize gripper action [0,1] -> [-1,+1] if needed by environment
    # Note: This may need adjustment based on SimplerEnv's gripper action format
    action = normalize_gripper_action(action, binarize=False)

    # [OpenVLA] The dataloader might flip the sign of the gripper action
    if model_family == "openvla":
        action = invert_gripper_action(action)

    return action


def save_replay_video(replay_images, video_id, success=False, task_description="", log_file=None, run_id=None):
    """Save replay video of the episode."""
    # Create videos directory
    video_dir = f"./eval_videos/{run_id}" if run_id else "./eval_videos"
    os.makedirs(video_dir, exist_ok=True)
    
    # Save as gif or video
    if replay_images:
        success_str = "SUCCESS" if success else "FAIL"
        filename = f"{video_id}_{success_str}.gif"
        filepath = os.path.join(video_dir, filename)
        
        # Convert numpy arrays to PIL Images if needed
        pil_images = []
        for img in replay_images:
            if isinstance(img, np.ndarray):
                pil_images.append(Image.fromarray(img.astype(np.uint8)))
            else:
                pil_images.append(img)
        
        # Save as GIF
        if pil_images:
            pil_images[0].save(
                filepath,
                save_all=True,
                append_images=pil_images[1:],
                duration=100,  # 100ms per frame
                loop=0
            )
            
        log_message(f"Saved replay video: {filepath}", log_file)


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
    log_file=None,
):
    """Run a single episode in the SimplerEnv environment."""
    # Reset environment
    obs, info = env.reset()
    
    # Initialize action queue
    if cfg.num_open_loop_steps != NUM_ACTIONS_CHUNK:
        print(f"WARNING: cfg.num_open_loop_steps ({cfg.num_open_loop_steps}) does not match the NUM_ACTIONS_CHUNK "
              f"({NUM_ACTIONS_CHUNK}) constant defined in prismatic.vla.constants! For best performance (in terms of "
               "both speed and success rate), we recommend executing the full action chunk.")
    action_queue = deque(maxlen=cfg.num_open_loop_steps)

    # Setup
    t = 0
    replay_images = []
    max_steps = TASK_MAX_STEPS.get(cfg.task_suite_name, 300)

    # Run episode
    success = False
    try:
        while t < max_steps + cfg.num_steps_wait:
            # Do nothing for the first few timesteps to let objects stabilize
            if t < cfg.num_steps_wait:
                # Use zero action for stabilization
                zero_action = np.zeros(7, dtype=np.float32)
                obs, reward, terminated, truncated, info = env.step(zero_action)
                t += 1
                continue

            # Prepare observation
            observation, img = prepare_observation(obs, env, resize_size)
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
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Check for success
            done = terminated or truncated
            if done and terminated:  # terminated means task completed successfully
                success = True
                break
            elif done:  # truncated means timeout or failure
                break
                
            t += 1

    except Exception as e:
        log_message(f"Episode error: {e}", log_file)

    return success, replay_images


def run_task(
    cfg: GenerateConfig,
    task_name: str,
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
    """Run evaluation for a single SimplerEnv task with all prompt variants."""
    # Initialize environment
    env = simpler_env.make(task_name)
    
    # Get initial task description
    obs, info = env.reset()
    original_task_description = get_task_instruction(env, info, task_name)

    # Get all prompt variants for systematic evaluation
    if holdout_prompts:
        prompt_variants = get_all_prompt_variants(original_task_description, holdout_prompts)
    else:
        prompt_variants = [("original", original_task_description)]

    log_message(f"\n{'='*80}", log_file)
    log_message(f"TASK: {task_name}", log_file)
    log_message(f"Original instruction: {original_task_description}", log_file)
    log_message(f"Will evaluate {len(prompt_variants)} prompt variants, {cfg.num_trials_per_task} episodes each", log_file)
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

        with tqdm.tqdm(total=cfg.num_trials_per_task, desc=f"{task_name}_{variant_type}") as pbar:
            while variant_episodes < cfg.num_trials_per_task:
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

                # Save replay video
                video_id = f"{task_name}_{variant_type}_{variant_episodes}"
                save_replay_video(
                    replay_images, video_id, success=success, 
                    task_description=f"[{task_name}] [{variant_type}] {task_description}", 
                    log_file=log_file, run_id=run_id
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
    log_message(f"TASK {task_name} SUMMARY:", log_file)
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
            wandb_data[f"success_rate/{task_name}_{variant_type}"] = results["success_rate"]
            wandb_data[f"num_episodes/{task_name}_{variant_type}"] = results["episodes"]
        
        # Log overall task results
        wandb_data[f"success_rate/{task_name}_overall"] = task_overall_success_rate
        wandb_data[f"num_episodes/{task_name}_overall"] = task_total_episodes
        
        wandb.log(wandb_data)

    return total_episodes, total_successes


def get_tasks_to_evaluate(cfg: GenerateConfig):
    """Get list of tasks to evaluate based on configuration."""
    if cfg.openvla_benchmark_mode:
        # Use OpenVLA benchmark protocol: 12 tasks with 5 rollouts each
        cfg.num_trials_per_task = 5  # Override to match paper
        cfg.compute_stderr = True    # Enable stderr computation
        return SIMPLER_ENV_TASKS[TaskSuite.OPENVLA_BENCHMARK]
    elif cfg.task_name:
        # Evaluate specific task
        return [cfg.task_name]
    elif cfg.task_suite_name == TaskSuite.ALL_TASKS:
        # Evaluate all tasks
        all_tasks = []
        for task_list in SIMPLER_ENV_TASKS.values():
            if task_list != SIMPLER_ENV_TASKS[TaskSuite.OPENVLA_BENCHMARK]:  # Skip benchmark tasks in all mode
                all_tasks.extend(task_list)
        return all_tasks
    else:
        # Evaluate task suite
        return SIMPLER_ENV_TASKS.get(cfg.task_suite_name, [])


@draccus.wrap()
def eval_simpler_env(cfg: GenerateConfig) -> float:
    """Main function to evaluate a trained policy on SimplerEnv tasks."""
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

    # Get tasks to evaluate
    tasks_to_evaluate = get_tasks_to_evaluate(cfg)
    
    if not tasks_to_evaluate:
        log_message("No tasks to evaluate! Check task_name or task_suite_name.", log_file)
        return 0.0

    log_message(f"Tasks to evaluate: {tasks_to_evaluate}", log_file)

    # Start evaluation
    total_episodes, total_successes = 0, 0
    for task_name in tqdm.tqdm(tasks_to_evaluate, desc="Tasks"):
        total_episodes, total_successes = run_task(
            cfg,
            task_name,
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

    # Calculate standard error if requested (for benchmark comparison)
    stderr = 0.0
    if cfg.compute_stderr and total_episodes > 0:
        # Standard error = sqrt(p * (1-p) / n) where p = success rate, n = total episodes
        import math
        stderr = math.sqrt(final_success_rate * (1 - final_success_rate) / total_episodes)

    # Log final results
    log_message("Final results:", log_file)
    log_message(f"Total episodes: {total_episodes}", log_file)
    log_message(f"Total successes: {total_successes}", log_file)
    
    if cfg.compute_stderr:
        log_message(f"Overall success rate: {final_success_rate:.1%} ± {stderr:.1%}", log_file)
        log_message(f"OpenVLA Benchmark Format: {final_success_rate*100:.1f}±{stderr*100:.1f}%", log_file)
        
        # Compare to OpenVLA paper results
        log_message("\n" + "="*60, log_file)
        log_message("COMPARISON TO OPENVLA PAPER RESULTS:", log_file)
        log_message("RT-1-X:     33.3±6.1%", log_file)
        log_message("Octo:       26.7±5.8%", log_file) 
        log_message("RT-2-X:     78.3±5.4%", log_file)
        log_message("OpenVLA:    85.0±4.6%", log_file)
        log_message(f"Your Model: {final_success_rate*100:.1f}±{stderr*100:.1f}%", log_file)
        log_message("="*60, log_file)
    else:
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
    eval_simpler_env()


# Usage examples:

# OpenVLA Benchmark Mode (matches paper evaluation protocol):
"""
CUDA_VISIBLE_DEVICES=0 python experiments/robot/simpler_env/run_simpler_env_eval.py \
    --pretrained_checkpoint "./checkpoints/your_finetuned_model" \
    --openvla_benchmark_mode True \
    --use_l1_regression True \
    --use_proprio True \
    --lora_rank 32 \
    --use_wandb True \
    --wandb_entity your-entity \
    --wandb_project openvla-benchmark \
    --run_id_note "benchmark_comparison"
"""

# Evaluate specific task:
"""
CUDA_VISIBLE_DEVICES=0 python experiments/robot/simpler_env/run_simpler_env_eval.py \
    --pretrained_checkpoint "./checkpoints/your_finetuned_model" \
    --task_name "google_robot_pick_standing_coke_can" \
    --num_trials_per_task 20 \
    --use_l1_regression True \
    --use_proprio True \
    --lora_rank 32 \
    --use_wandb True \
    --wandb_entity your-entity \
    --wandb_project simpler-env-eval
"""

# Evaluate all pick tasks:
"""
CUDA_VISIBLE_DEVICES=0 python experiments/robot/simpler_env/run_simpler_env_eval.py \
    --pretrained_checkpoint "./checkpoints/your_finetuned_model" \
    --task_suite_name pick \
    --num_trials_per_task 20 \
    --use_l1_regression True \
    --use_proprio True \
    --lora_rank 32
"""

# Evaluate all tasks:
"""
CUDA_VISIBLE_DEVICES=0 python experiments/robot/simpler_env/run_simpler_env_eval.py \
    --pretrained_checkpoint "./checkpoints/your_finetuned_model" \
    --task_suite_name all \
    --num_trials_per_task 10 \
    --use_l1_regression True \
    --use_proprio True \
    --lora_rank 32
"""