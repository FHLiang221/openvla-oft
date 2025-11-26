#!/usr/bin/env python3
"""
dryrun_policy_adv.py
--------------------
Offline sanity check for adv_finetune checkpoints.

It loads the *base* OpenVLA model, applies the LoRA adapter saved by adv_finetune
(when merge_lora_during_training=False), reconstructs the L1RegressionActionHead
if present, and predicts a single 7-D action from a few frames + instruction.

Usage examples
--------------
# Point directly at a specific checkpoint directory (…--10000_chkpt)
python3 dryrun_policy_adv.py \
  --ckpt_dir ./checkpoints/success_oft/<run_id>--10000_chkpt \
  --vla_path openvla/openvla-7b \
  --npz /path/to/success_episode_000001.npz \
  --num_images_in_input 1 \
  --use_proprio

# Or use images
python3 dryrun_policy_adv.py \
  --ckpt_dir ./checkpoints/success_oft/<run_id>--10000_chkpt \
  --vla_path openvla/openvla-7b \
  --images_glob "samples/*.png" \
  --instruction "pick the standing coke can"

Notes
-----
- This script mirrors key logic in your adv_finetune training loop:
  * loads processor + VLA (trust_remote_code)
  * sets vision_backbone num_images_in_input
  * loads LoRA adapter from ckpt_dir/lora_adapter
  * if action_head--*_checkpoint.pt exists, reconstructs L1RegressionActionHead and uses it
  * extracts last_hidden_states and slices text tokens similar to training
- Masking of action token positions is approximated by taking the *last*
  NUM_ACTIONS_CHUNK * ACTION_DIM text tokens. This is sufficient for a **sanity check**,
  but not a substitute for the full training-time masking.
"""

import argparse, os, json
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image

import torch
from transformers import AutoProcessor, AutoModelForVision2Seq

# prismatic / openvla components (must be installed in the env you trained with)
from prismatic.models.action_heads import L1RegressionActionHead
from prismatic.vla.constants import ACTION_DIM, NUM_ACTIONS_CHUNK
from peft import PeftModel


def log(s): print(s, flush=True)


def load_frames_from_npz(npz_path: str, max_frames: int = 4) -> Tuple[List[Image.Image], Optional[np.ndarray], Optional[str]]:
    D = np.load(npz_path, allow_pickle=True)

    # image field names we might encounter
    for k in ("images", "image_primary"):
        if k in D:
            imgs = D[k]
            break
    else:
        raise RuntimeError(f"No images found in NPZ: {list(D.keys())}")

    # language/instruction
    language = None
    for k in ("language", "instruction"):
        if k in D:
            language = str(D[k])
            break

    # proprio/states (optional)
    proprio = None
    for k in ("states", "proprio"):
        if k in D:
            proprio = np.asarray(D[k])
            break

    # take first N frames
    T = imgs.shape[0]
    N = min(max_frames, T)
    frames = [Image.fromarray(imgs[i]).convert("RGB") for i in range(N)]

    if proprio is not None:
        proprio = proprio[:N]

    return frames, proprio, language


def load_frames_from_glob(pattern: str, max_frames: int = 4) -> List[Image.Image]:
    paths = sorted(Path().glob(pattern))
    if not paths:
        raise RuntimeError(f"No images matched: {pattern}")
    frames = []
    for p in paths[:max_frames]:
        frames.append(Image.open(p).convert("RGB"))
    return frames


def sanity_checks(a: np.ndarray,
                  max_xyz: float = 0.20,      # meters per step (tune to your data)
                  max_aa: float = 0.75,       # rad per step (axis-angle magnitude)
                  grip_range: Tuple[float,float] = (0.0, 1.0)):
    assert a.shape == (7,), f"Expected 7-D action, got {a.shape}"
    assert np.all(np.isfinite(a)), f"Action has NaN/Inf: {a}"
    # position deltas
    if np.linalg.norm(a[:3], ord=np.inf) > max_xyz:
        log(f"⚠️  Large Δxyz: {a[:3]} (>|{max_xyz}|)")
    # orientation delta (axis-angle magnitude)
    aa_mag = np.linalg.norm(a[3:6])
    if aa_mag > max_aa:
        log(f"⚠️  Large Δaxis-angle magnitude: {aa_mag:.3f} rad (> {max_aa})")
    # gripper
    if not (grip_range[0] - 1e-3 <= a[6] <= grip_range[1] + 1e-3):
        log(f"⚠️  Gripper out of range {grip_range}: {a[6]:.3f}")


def find_action_head_checkpoint(ckpt_dir: Path) -> Optional[Path]:
    # action_head--{...}_checkpoint.pt
    cand = sorted(ckpt_dir.glob("action_head--*_checkpoint.pt"))
    return cand[0] if cand else None


def predict_action_with_head(vla, processor, frames, instruction: str, proprio: Optional[np.ndarray], device="cpu") -> np.ndarray:
    """
    Heuristic inference mirroring training:
    - build batch via processor
    - forward with output_hidden_states=True
    - slice last_hidden_states to get text_hidden_states (skip visual patches)
    - take last NUM_ACTIONS_CHUNK*ACTION_DIM tokens and pass through L1 head
    - return the first 7-D action (current action)
    """
    # tokenize + image transform
    # Check model's expected number of images
    expected_num_images = vla.vision_backbone.get_num_images_in_input()
    log(f"Model expects {expected_num_images} images, got {len(frames)}")
    
    # Ensure we have the right number of images
    if len(frames) != expected_num_images:
        if len(frames) > expected_num_images:
            # Take the first N images
            frames = frames[:expected_num_images]
            log(f"Truncated to {expected_num_images} images")
        else:
            # Repeat the last image to reach expected count
            while len(frames) < expected_num_images:
                frames.append(frames[-1])
            log(f"Padded to {expected_num_images} images by repeating last frame")
    
    # Process each image individually with the same instruction, then concatenate
    if len(frames) == 1:
        # Single image case
        batch = processor(text=instruction, images=frames[0], return_tensors="pt")
    else:
        # Multiple images case - process each image with the same instruction
        all_inputs = [processor(text=instruction, images=frame, return_tensors="pt") for frame in frames]
        
        # Concatenate pixel values along the channel dimension
        # Each image has 6 channels (3 for SigLIP + 3 for DINOv2) when using fused vision backbone
        primary_pixel_values = all_inputs[0]["pixel_values"]
        additional_pixel_values = [inputs["pixel_values"] for inputs in all_inputs[1:]]
        concatenated_pixel_values = torch.cat([primary_pixel_values] + additional_pixel_values, dim=1)
        
        # Use the first input as base and replace pixel_values
        batch = all_inputs[0].copy()
        batch["pixel_values"] = concatenated_pixel_values
        
        # Debug: log tensor shapes
        log(f"Concatenated pixel_values shape: {concatenated_pixel_values.shape}")
        log(f"Expected channels per image: 6, Total images: {len(frames)}")
        log(f"Expected total channels: {6 * len(frames)}")
    
    batch = {k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()}

    # proprio (optional)
    proprio_tensor = None
    if proprio is not None:
        proprio_tensor = torch.tensor(proprio, dtype=torch.float32, device=device).unsqueeze(0)  # [1,N,8]

    vla.eval()
    try:
        with torch.no_grad(), torch.autocast(device_type="cuda" if device.startswith("cuda") else "cpu", dtype=torch.bfloat16 if device.startswith("cuda") else torch.float32):
            # Create dummy labels to avoid action mask processing errors
            # Use IGNORE_INDEX (-100) for all tokens to avoid loss computation
            dummy_labels = torch.full_like(batch["input_ids"], -100)
            
            out = vla(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                pixel_values=batch["pixel_values"].to(torch.bfloat16) if device.startswith("cuda") else batch["pixel_values"],
                labels=dummy_labels,  # Use dummy labels instead of None
                output_hidden_states=True,
                proprio=proprio_tensor,  # Use actual proprioception if provided
                proprio_projector=None,
                noisy_actions=None,
                noisy_action_projector=None,
                diffusion_timestep_embeddings=None,
                use_film=False,
            )
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            raise SystemExit(f"CUDA out of memory. Try reducing --max_frames or using CPU with --device cpu")
        else:
            raise SystemExit(f"Model forward pass failed: {e}")

    last_hidden = out.hidden_states[-1]           # [1, seq_len, dim]
    num_patches = vla.vision_backbone.get_num_patches() * vla.vision_backbone.get_num_images_in_input()
    text_hidden = last_hidden[:, num_patches:-1, :]   # drop image patches; -1 to align with training

    need = NUM_ACTIONS_CHUNK * ACTION_DIM
    if text_hidden.shape[1] < need:
        # left-pad with repeats of the first token so shape matches
        pad = text_hidden[:, :1, :].expand(1, need - text_hidden.shape[1], text_hidden.shape[2])
        action_tokens = torch.cat([pad, text_hidden], dim=1)
    else:
        action_tokens = text_hidden[:, -need:, :]

    # shape -> [B, need, H]
    # L1 head expects [B, NUM_ACTIONS_CHUNK*ACTION_DIM, H], returns [B, NUM_ACTIONS_CHUNK, ACTION_DIM]
    pred_actions = vla._action_head.predict_action(action_tokens.to(torch.bfloat16 if device.startswith("cuda") else action_tokens.dtype))
    a = pred_actions[0, 0, :7].float().cpu().numpy()
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", required=True, help="Checkpoint directory produced by adv_finetune (…--<step>_chkpt)")
    ap.add_argument("--vla_path", required=True, help="Base VLA HF path (e.g., openvla/openvla-7b)")
    ap.add_argument("--npz", help="Episode NPZ with frames ('images' or 'image_primary') and optional 'states'/'proprio'")
    ap.add_argument("--images_glob", help="Glob for image files (e.g., 'samples/*.png')")
    ap.add_argument("--instruction", default=None, help="Override instruction string")
    ap.add_argument("--max_frames", type=int, default=4)
    ap.add_argument("--num_images_in_input", type=int, default=1)
    ap.add_argument("--use_proprio", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    ckpt_dir = Path(args.ckpt_dir)
    adapter_dir = ckpt_dir / "lora_adapter"
    if not adapter_dir.is_dir():
        raise SystemExit(f"Expected LoRA adapter at: {adapter_dir}")

    # Load processor + base model
    try:
        log(f"Loading processor from {ckpt_dir}")
        processor = AutoProcessor.from_pretrained(ckpt_dir, trust_remote_code=True)
        log(f"Loading base model from {args.vla_path}")
        vla = AutoModelForVision2Seq.from_pretrained(args.vla_path, torch_dtype=torch.bfloat16 if args.device.startswith('cuda') else torch.float32, low_cpu_mem_usage=True, trust_remote_code=True)
        vla.vision_backbone.set_num_images_in_input(args.num_images_in_input)
    except Exception as e:
        raise SystemExit(f"Failed to load model: {e}")

    # Load LoRA adapter and keep it attached (no merge needed for sanity)
    log(f"Attaching LoRA adapter from {adapter_dir}")
    vla = PeftModel.from_pretrained(vla, str(adapter_dir))
    vla.to(args.device)

    # Load action head (if saved)
    ah_path = find_action_head_checkpoint(ckpt_dir)
    if ah_path is None:
        raise SystemExit(f"No action head checkpoint found in {ckpt_dir}. Expected a file like 'action_head--*_checkpoint.pt'.")

    log(f"Rebuilding L1RegressionActionHead from {ah_path.name}")
    # vla.config.llm_dim or vla.llm_dim (depends on class); use getattr with fallback
    llm_dim = getattr(vla, "llm_dim", None) or getattr(vla.base_model, "llm_dim", None) or getattr(vla.config, "llm_dim", None)
    if llm_dim is None:
        # last resort: infer from hidden state size by a dry pass with dummy inputs (rarely needed)
        raise SystemExit("Could not infer llm_dim from model. Ensure you're using the same codebase as training.")
    action_head = L1RegressionActionHead(input_dim=llm_dim, hidden_dim=llm_dim, action_dim=ACTION_DIM)
    
    try:
        state = torch.load(str(ah_path), map_location="cpu", weights_only=True)
        
        # Remove 'module.' prefix from state_dict keys if present (common with DataParallel/DistributedDataParallel)
        if any(key.startswith('module.') for key in state.keys()):
            state = {key.replace('module.', ''): value for key, value in state.items()}
        
        action_head.load_state_dict(state)
    except Exception as e:
        raise SystemExit(f"Failed to load action head checkpoint: {e}")
    
    action_head.to(args.device).eval()

    # tuck into model for convenience
    vla._action_head = action_head

    # Load frames + instruction (+proprio optional)
    frames, proprio, instr = None, None, None
    if args.npz:
        frames, proprio, instr = load_frames_from_npz(args.npz, max_frames=args.max_frames)
    elif args.images_glob:
        frames = load_frames_from_glob(args.images_glob, max_frames=args.max_frames)
    else:
        raise SystemExit("Provide --npz or --images_glob")

    instruction = args.instruction or instr or "pick the standing coke can"

    # Predict a single action (7-D) heuristically
    a = predict_action_with_head(vla, processor, frames, instruction, proprio if args.use_proprio else None, device=args.device)

    # Sanity checks
    log(f"Predicted 7-D action: {np.array2string(a, precision=4)}")
    sanity_checks(a)

    # Pretty print
    dx, dy, dz, rx, ry, rz, g = a
    out = {
        "instruction": instruction,
        "action": {
            "delta_xyz_m": [float(dx), float(dy), float(dz)],
            "delta_axis_angle_rad": [float(rx), float(ry), float(rz)],
            "gripper_abs": float(g),
        },
        "ckpt_dir": str(ckpt_dir),
        "adapter_dir": str(adapter_dir),
        "used_action_head": str(ah_path.name)
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    # reduce TF CUDA noise if TF gets imported through processor internals
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    main()
