"""
run_simpler_env_eval_hf.py

Evaluates a trained policy in SimplerEnv environments using direct HuggingFace loading approach.
Based on SimplerEnv-OpenVLA implementation that works with base OpenVLA models.
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
import imageio

import wandb

# HuggingFace imports for direct model loading
from transformers import AutoModelForVision2Seq, AutoProcessor
import torch
from transforms3d.euler import euler2axangle

# Append current directory so that interpreter can find experiments.robot
sys.path.append("../..")
from experiments.robot.robot_utils import (
    DATE_TIME,
    set_seed_everywhere,
)

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

    image_size: list[int] = None                     # Image size [H, W] - will default to [224, 224]
    action_scale: float = 1.0                        # Action scaling factor

    # Sticky gripper settings (for Google Robot setup)
    sticky_gripper_num_repeat: int = 15              # Number of steps to repeat gripper action

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

    # Validate task suite or specific task
    if cfg.task_name is None:
        assert cfg.task_suite_name in [suite.value for suite in TaskSuite], f"Invalid task suite: {cfg.task_suite_name}"


class OpenVLAInference:
    """OpenVLA inference class using direct HuggingFace loading (SimplerEnv-OpenVLA approach)."""
    
    def __init__(
        self,
        saved_model_path: str = "openvla/openvla-7b",
        unnorm_key: Optional[str] = None,
        policy_setup: str = "google_robot",
        image_size: list[int] = [224, 224],
        action_scale: float = 1.0,
        sticky_gripper_num_repeat: int = 15,
    ) -> None:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        
        # Policy setup configuration
        if policy_setup == "google_robot":
            unnorm_key = "fractal20220817_data" if unnorm_key is None else unnorm_key
            self.sticky_gripper_num_repeat = sticky_gripper_num_repeat
        else:
            raise NotImplementedError(f"Policy setup {policy_setup} not supported")
            
        self.policy_setup = policy_setup
        self.unnorm_key = unnorm_key

        print(f"*** policy_setup: {policy_setup}, unnorm_key: {unnorm_key} ***")
        
        # Load model and processor
        self.processor = AutoProcessor.from_pretrained(saved_model_path, trust_remote_code=True)
        self.vla = AutoModelForVision2Seq.from_pretrained(
            saved_model_path,
            attn_implementation="flash_attention_2",  # [Optional] Requires `flash_attn`
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).cuda()

        self.image_size = image_size
        self.action_scale = action_scale

        # Initialize sticky gripper state
        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        self.previous_gripper_action = None

    def reset(self) -> None:
        """Reset sticky gripper state for new episode."""
        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        self.previous_gripper_action = None

    def predict_action(self, image: np.ndarray, task_description: str) -> dict:
        """
        Predict action using OpenVLA model.
        
        Input:
            image: np.ndarray of shape (H, W, 3), uint8
            task_description: str, task description
        Output:
            raw_action: dict with keys:
                - 'world_vector': np.ndarray of shape (3,), xyz translation
                - 'rotation_delta': np.ndarray of shape (3,), rotation delta (roll, pitch, yaw)
                - 'open_gripper': np.ndarray of shape (1,), gripper action [0, 1]
        """
        assert image.dtype == np.uint8
        
        # Resize image
        import cv2 as cv
        image = cv.resize(image, tuple(self.image_size), interpolation=cv.INTER_AREA)
        
        # Convert to PIL Image
        image_pil = Image.fromarray(image)
        
        # Process with model
        inputs = self.processor(task_description, image_pil).to("cuda:0", dtype=torch.bfloat16)
        raw_actions = self.vla.predict_action(**inputs, unnorm_key=self.unnorm_key, do_sample=False)[None]
        
        raw_action = {
            "world_vector": np.array(raw_actions[0, :3]),
            "rotation_delta": np.array(raw_actions[0, 3:6]),
            "open_gripper": np.array(raw_actions[0, 6:7]),  # range [0, 1]; 1 = open; 0 = close
        }

        return raw_action

    def process_action_for_env(self, raw_action: dict) -> dict:
        """Process raw action for SimplerEnv environment."""
        action = {}
        action["world_vector"] = raw_action["world_vector"] * self.action_scale
        
        # Convert rotation delta to axis-angle representation
        action_rotation_delta = np.asarray(raw_action["rotation_delta"], dtype=np.float64)
        roll, pitch, yaw = action_rotation_delta
        action_rotation_ax, action_rotation_angle = euler2axangle(roll, pitch, yaw)
        action_rotation_axangle = action_rotation_ax * action_rotation_angle
        action["rot_axangle"] = action_rotation_axangle * self.action_scale

        # Process gripper action with sticky gripper logic for Google Robot
        if self.policy_setup == "google_robot":
            current_gripper_action = raw_action["open_gripper"]
            if self.previous_gripper_action is None:
                relative_gripper_action = np.array([0])
            else:
                relative_gripper_action = self.previous_gripper_action - current_gripper_action
            self.previous_gripper_action = current_gripper_action

            if np.abs(relative_gripper_action) > 0.5 and (not self.sticky_action_is_on):
                self.sticky_action_is_on = True
                self.sticky_gripper_action = relative_gripper_action

            if self.sticky_action_is_on:
                self.gripper_action_repeat += 1
                relative_gripper_action = self.sticky_gripper_action

            if self.gripper_action_repeat == self.sticky_gripper_num_repeat:
                self.sticky_action_is_on = False
                self.gripper_action_repeat = 0
                self.sticky_gripper_action = 0.0

            action["gripper"] = relative_gripper_action
        else:
            # Simple gripper action for other setups
            action["gripper"] = 2.0 * (raw_action["open_gripper"] > 0.5) - 1.0

        action["terminate_episode"] = np.array([0.0])
        return action


def initialize_model(cfg: GenerateConfig):
    """Initialize OpenVLA model using direct HuggingFace loading."""
    if cfg.image_size is None:
        cfg.image_size = [224, 224]
        
    model = OpenVLAInference(
        saved_model_path=cfg.pretrained_checkpoint,
        policy_setup="google_robot",  # SimplerEnv uses Google Robot setup
        image_size=cfg.image_size,
        action_scale=cfg.action_scale,
        sticky_gripper_num_repeat=cfg.sticky_gripper_num_repeat,
    )
    
    return model


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
    """Get task instruction from environment."""
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


def setup_logging(cfg: GenerateConfig):
    """Set up logging to file and optionally to wandb."""
    # Create run ID
    task_name = cfg.task_name if cfg.task_name else cfg.task_suite_name
    run_id = f"EVAL-SimplerEnv-HF-{task_name}-{cfg.model_family}-{DATE_TIME}"
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


def prepare_observation(obs, env):
    """Prepare observation for policy input."""
    # Get RGB image from SimplerEnv observation
    try:
        img = get_image_from_maniskill2_obs_dict(env, obs)
    except KeyError as e:
        print(f"DEBUG: KeyError in get_image_from_maniskill2_obs_dict: {e}")
        print(f"DEBUG: Available observation keys: {list(obs.keys())}")
        
        # Check if image structure exists but with different nesting
        if "image" in obs:
            print(f"DEBUG: 'image' key found. Structure: {list(obs['image'].keys())}")
            
            # Determine camera name
            camera_name = None
            if "google_robot" in env.robot_uid:
                camera_name = "overhead_camera"
            elif "widowx" in env.robot_uid:
                camera_name = "3rd_view_camera"
            
            # Try to extract image with expected camera name
            if camera_name and camera_name in obs["image"]:
                print(f"DEBUG: Found camera '{camera_name}'. Keys: {list(obs['image'][camera_name].keys())}")
                if "rgb" in obs["image"][camera_name]:
                    img = obs["image"][camera_name]["rgb"]
                else:
                    # Try other common image keys
                    for key in ["color", "image", "full_image"]:
                        if key in obs["image"][camera_name]:
                            img = obs["image"][camera_name][key]
                            break
                    else:
                        raise KeyError(f"No RGB image found in {camera_name}. Available: {list(obs['image'][camera_name].keys())}")
            else:
                # Try to get any available camera
                available_cameras = list(obs["image"].keys())
                print(f"DEBUG: Camera '{camera_name}' not found. Available cameras: {available_cameras}")
                if available_cameras:
                    first_camera = available_cameras[0]
                    if "rgb" in obs["image"][first_camera]:
                        img = obs["image"][first_camera]["rgb"]
                    else:
                        # Try other image keys
                        for key in ["color", "image", "full_image"]:
                            if key in obs["image"][first_camera]:
                                img = obs["image"][first_camera][key]
                                break
                        else:
                            raise KeyError(f"No RGB image found in camera {first_camera}")
                else:
                    raise KeyError("No cameras found in observation")
        else:
            # Fallback to render if no image structure
            print("DEBUG: No 'image' key found, falling back to render")
            if hasattr(env, 'render'):
                img = env.render()
            else:
                raise KeyError(f"Could not find image in observation and env has no render method")
    
    # Ensure image is numpy array and has correct format
    if not isinstance(img, np.ndarray):
        img = np.array(img)

    return img


def convert_env_action_to_simpler_env_format(action_dict: dict) -> np.ndarray:
    """Convert processed action dict to SimplerEnv 7D action format."""
    # SimplerEnv expects 7D actions: [dx, dy, dz, drx, dry, drz, gripper]
    action_7d = np.concatenate([
        action_dict["world_vector"],      # 3D translation
        action_dict["rot_axangle"],       # 3D rotation (axis-angle)
        action_dict["gripper"]            # 1D gripper
    ])
    
    # Ensure it's exactly 7D
    if len(action_7d) > 7:
        action_7d = action_7d[:7]
    elif len(action_7d) < 7:
        # Pad with zeros if needed
        action_7d = np.pad(action_7d, (0, 7 - len(action_7d)), mode='constant')
    
    return action_7d.astype(np.float32)


def save_replay_video(replay_images, video_id, success=False, task_description="", log_file=None, run_id=None, task_name=None):
    """Save replay video of the episode."""
    # Create videos directory with task-specific subfolder
    if run_id and task_name:
        video_dir = f"./eval_videos/{run_id}/{task_name}"
    elif run_id:
        video_dir = f"./eval_videos/{run_id}"
    else:
        video_dir = "./eval_videos"
    os.makedirs(video_dir, exist_ok=True)
    
    # Save as MP4 video
    if replay_images:
        success_str = "SUCCESS" if success else "FAIL"
        filename = f"{video_id}_{success_str}.mp4"
        filepath = os.path.join(video_dir, filename)
        
        # Convert to numpy arrays if needed and ensure uint8 format
        video_frames = []
        for img in replay_images:
            if isinstance(img, Image.Image):
                video_frames.append(np.array(img).astype(np.uint8))
            elif isinstance(img, np.ndarray):
                video_frames.append(img.astype(np.uint8))
            else:
                video_frames.append(np.array(img).astype(np.uint8))
        
        # Save as MP4 using imageio
        if video_frames:
            imageio.mimsave(
                filepath,
                video_frames,
                fps=10,  # 10 fps for good viewing speed
                quality=8,  # Good quality
                macro_block_size=1
            )
            
        log_message(f"Saved replay video: {filepath}", log_file)


def run_episode(
    cfg: GenerateConfig,
    env,
    task_description: str,
    model,
    log_file=None,
):
    """Run a single episode in the SimplerEnv environment."""
    # Reset environment and model
    obs, info = env.reset()
    model.reset()
    
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
            img = prepare_observation(obs, env)
            replay_images.append(img)

            # Get action from model (SimplerEnv-OpenVLA approach)
            raw_action = model.predict_action(img, task_description)
            action_dict = model.process_action_for_env(raw_action)
            
            # Convert to SimplerEnv format
            action = convert_env_action_to_simpler_env_format(action_dict)

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
                    log_file=log_file, run_id=run_id, task_name=task_name
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
    """Main function to evaluate a trained policy on SimplerEnv tasks using HuggingFace approach."""
    # Validate configuration
    validate_config(cfg)

    # Set random seed
    set_seed_everywhere(cfg.seed)

    # Initialize model
    model = initialize_model(cfg)

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

# Evaluate base OpenVLA model:
"""
CUDA_VISIBLE_DEVICES=0 python experiments/robot/simpler_env/run_simpler_env_eval_hf.py \
    --pretrained_checkpoint "openvla/openvla-7b" \
    --task_suite_name all \
    --num_trials_per_task 10
    
cd /project/fhliang/projects/openvla-oft && python experiments/robot/simpler_env/run_simpler_env_eval_hf.py --pretrained_checkpoint "openvla/openvla-7b" --task_suite_name all --num_trials_per_task 10
"""

# OpenVLA Benchmark Mode (matches paper evaluation protocol):
"""
CUDA_VISIBLE_DEVICES=0 python experiments/robot/simpler_env/run_simpler_env_eval_hf.py \
    --pretrained_checkpoint "openvla/openvla-7b" \
    --openvla_benchmark_mode True \
    --use_wandb True \
    --wandb_entity your-entity \
    --wandb_project openvla-benchmark \
    --run_id_note "base_model_benchmark"
"""

# Evaluate specific task:
"""
CUDA_VISIBLE_DEVICES=0 python experiments/robot/simpler_env/run_simpler_env_eval_hf.py \
    --pretrained_checkpoint "openvla/openvla-7b" \
    --task_name "google_robot_pick_standing_coke_can" \
    --num_trials_per_task 20
"""