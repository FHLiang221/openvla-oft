#!/usr/bin/env python3
# Save as: infer_actions_from_npz.py
# Usage:
#   python3 infer_actions_from_npz.py \
#     --npz /path/to/episode.npz \
#     --pretrained_checkpoint ./checkpoints/<run_id>--10000_chkpt_merged \
#     --out_npz /path/to/episode_actions.npz \
#     --num_images_in_input 1 --use_proprio
import argparse, json, numpy as np
from pathlib import Path
from types import SimpleNamespace

# OpenVLA-OFT helpers from your repo
from experiments.robot.openvla_utils import (
    get_action_head, get_processor, get_proprio_projector,
    get_vla, get_vla_action,
)
from prismatic.vla.constants import NUM_ACTIONS_CHUNK, PROPRIO_DIM

def load_npz_episode(p: Path):
    D = np.load(p, allow_pickle=True)
    imgs = D["images"] if "images" in D else D["image_primary"]
    lang = str(D["language"]) if "language" in D else str(D["instruction"])
    states = None
    if "states" in D: states = D["states"].astype(np.float32)
    elif "proprio" in D: states = D["proprio"].astype(np.float32)
    ts = D["timestamps_ns"] if "timestamps_ns" in D else None
    return imgs, lang, states, ts

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--pretrained_checkpoint", required=True,
                    help="Merged HF id or directory (if you trained with LoRA, merge once)")
    ap.add_argument("--out_npz", required=True, help="Where to save predicted actions")
    ap.add_argument("--num_images_in_input", type=int, default=1, choices=[1,2])
    ap.add_argument("--use_proprio", action="store_true", default=True)
    ap.add_argument("--center_crop", action="store_true", default=True)
    ap.add_argument("--unnorm_key", default="kinova_coke_pick_same_start")
    ap.add_argument("--horizon", type=int, default=-1, help="<=0 uses full NPZ length")
    ap.add_argument("--save_mode", choices=["first","chunk"], default="first",
                    help="'first' saves [T,7], 'chunk' saves [T,K,7]")
    args = ap.parse_args()

    cfg = SimpleNamespace(
        pretrained_checkpoint=args.pretrained_checkpoint,
        use_l1_regression=True, use_diffusion=False, use_film=False,
        num_images_in_input=args.num_images_in_input, use_proprio=args.use_proprio,
        load_in_8bit=False, load_in_4bit=False, center_crop=args.center_crop,
        num_open_loop_steps=NUM_ACTIONS_CHUNK, unnorm_key=args.unnorm_key,
    )

    # Load model & helpers
    vla = get_vla(cfg)
    processor = get_processor(cfg)
    action_head = get_action_head(cfg, llm_dim=vla.llm_dim)
    proprio_projector = get_proprio_projector(cfg, llm_dim=vla.llm_dim, proprio_dim=PROPRIO_DIM)

    # Data
    imgs, instruction, states, ts = load_npz_episode(Path(args.npz))
    T_avail = len(imgs)
    T = T_avail if args.horizon <= 0 else min(args.horizon, T_avail)

    # Inference
    first_actions = np.zeros((T, 7), dtype=np.float32)
    chunk_actions = np.zeros((T, NUM_ACTIONS_CHUNK, 7), dtype=np.float32) if args.save_mode == "chunk" else None

    for t in range(T):
        # Ensure image is in the correct format (numpy array with shape (H, W, 3) and dtype uint8)
        img_array = imgs[t]
        if img_array.dtype != np.uint8:
            img_array = img_array.astype(np.uint8)
        if len(img_array.shape) == 2:  # grayscale
            img_array = np.stack([img_array] * 3, axis=-1)
        elif img_array.shape[-1] == 1:  # single channel
            img_array = np.repeat(img_array, 3, axis=-1)
        
        s_t = states[t] if (states is not None and t < len(states)) else np.zeros((PROPRIO_DIM,), np.float32)
        obs = {"full_image": img_array, "state": s_t, "task_description": instruction}
        acts = get_vla_action(cfg, vla, processor, obs, instruction, action_head, proprio_projector)
        a = np.asarray(acts[0], dtype=np.float32)
        first_actions[t] = a
        if chunk_actions is not None:
            chunk = np.asarray(acts, dtype=np.float32)
            if chunk.shape[0] != NUM_ACTIONS_CHUNK:
                # pad or trim to fixed K
                K = min(NUM_ACTIONS_CHUNK, chunk.shape[0])
                chunk_actions[t, :K] = chunk[:K]
            else:
                chunk_actions[t] = chunk

    meta = {
        "source_npz": str(Path(args.npz).resolve()),
        "instruction": instruction,
        "num_images_in_input": args.num_images_in_input,
        "use_proprio": args.use_proprio,
        "unnorm_key": args.unnorm_key,
        "save_mode": args.save_mode,
        "NUM_ACTIONS_CHUNK": NUM_ACTIONS_CHUNK,
    }

    if args.save_mode == "first":
        np.savez_compressed(
            args.out_npz,
            pred_actions=first_actions,      # [T,7]
            timestamps_ns=ts if ts is not None else np.array([], dtype=np.int64),
            meta=np.string_(json.dumps(meta)),
        )
    else:
        np.savez_compressed(
            args.out_npz,
            pred_chunks=chunk_actions,       # [T,K,7]
            pred_actions=first_actions,      # convenience
            timestamps_ns=ts if ts is not None else np.array([], dtype=np.int64),
            meta=np.string_(json.dumps(meta)),
        )
    print(f"[ok] saved actions → {args.out_npz}")

if __name__ == "__main__":
    main()
