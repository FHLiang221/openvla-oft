#!/usr/bin/env python3
"""
npz_to_simpler_rlds_success.py
--------------------------------
Convert NPZ episodes produced by `rosbag2_to_rlds.py` into the
**SimplerEnvSuccessDataset** NPZ schema used by kpertsch/rlds_dataset_builder.

Input NPZ (per episode) from rosbag2_to_rlds.py (K = T steps) includes:
  - image_primary[K,H,W,3] uint8
  - actions[K,7] float32
  - proprio[K,8] float32                (optional but recommended)
  - timestamps_ns[K] int64              (unused here)
  - instruction (str)

Output NPZ (per episode) expected by SimplerEnvSuccessDataset builder:
  - episode_id: int64
  - env_name: str
  - language: str
  - images[T,224,224,3] uint8
  - actions[T,7] float32
  - states[T,8] float32
  - rewards[T] float32                  (0 ... 0, 1 on last step)
  - is_terminal[T] bool                 (False ... True on last step)
  - is_first[T] bool                    (True on first, else False)

Notes
- We keep T equal to the number of action/image pairs in the NPZ.
- Images are resized to 224x224 uint8 to match SimplerEnv collector.
- All episodes are treated as **successes** and saved to `successes/`.
"""

import argparse
import glob
import json
from pathlib import Path
from typing import List

import numpy as np

# ---- resizer (cv2 -> PIL) ----
def _resize(img: np.ndarray, size: int = 224) -> np.ndarray:
    if img.shape[:2] == (size, size):
        return img
    try:
        import cv2
        return cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
    except Exception:
        from PIL import Image
        return np.array(Image.fromarray(img).resize((size, size), Image.BILINEAR))

def convert_one(
    in_path: Path,
    out_success_dir: Path,
    episode_id: int,
    env_name: str,
    language_override: str = None,
    img_size: int = 224,
) -> Path:
    data = np.load(in_path, allow_pickle=True)
    # Required from rosbag2_to_rlds.py
    imgs = data["image_primary"]                    # [T,H,W,3] uint8
    acts = data["actions"].astype(np.float32)       # [T,7]
    proprio = data.get("proprio", None)             # [T,8] or None
    instr = str(data.get("instruction", ""))        # str
    if language_override is not None:
        instr = str(language_override)

    T = acts.shape[0]
    if imgs.shape[0] != T:
        # Bring to common min length
        T2 = min(imgs.shape[0], T)
        imgs = imgs[:T2]
        acts = acts[:T2]
        if proprio is not None:
            proprio = proprio[:T2]
        T = T2

    # Resize images
    images = np.stack([_resize(imgs[i], img_size).astype(np.uint8) for i in range(T)], axis=0)

    # States = proprio; if missing, synthesize zeros
    if proprio is None:
        states = np.zeros((T, 8), dtype=np.float32)
    else:
        states = proprio.astype(np.float32)

    # Rewards/flags
    rewards = np.zeros((T,), dtype=np.float32)
    if T > 0:
        rewards[-1] = 1.0
    is_terminal = np.zeros((T,), dtype=bool)
    if T > 0:
        is_terminal[-1] = True
    is_first = np.zeros((T,), dtype=bool)
    if T > 0:
        is_first[0] = True

    # Pack as SimplerEnvSuccessDataset episode
    out_success_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_success_dir / f"success_episode_{int(episode_id):06d}.npz"
    np.savez_compressed(
        out_path,
        episode_id=np.int64(episode_id),
        env_name=str(env_name),
        language=str(instr),
        images=images,            # [T,224,224,3] uint8
        actions=acts,             # [T,7] float32
        states=states,            # [T,8] float32
        rewards=rewards,          # [T] float32
        is_terminal=is_terminal,  # [T] bool
        is_first=is_first,        # [T] bool
    )
    return out_path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_npz_dir", required=True, help="Folder containing NPZ episodes from rosbag2_to_rlds.py")
    ap.add_argument("--pattern", default="episode_*.npz", help="Glob pattern for input NPZs")
    ap.add_argument("--out_base", required=True, help="Output base folder (will create successes/ under here)")
    ap.add_argument("--env_name", default="google_robot_pick_standing_coke_can")
    ap.add_argument("--start_id", type=int, default=1, help="Starting episode_id for naming")
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--language_override", default=None, help="If set, use this instruction for all episodes")
    args = ap.parse_args()

    in_dir = Path(args.in_npz_dir)
    out_base = Path(args.out_base)
    success_dir = out_base / "successes"

    files = sorted(in_dir.glob(args.pattern))
    if not files:
        raise SystemExit(f"No NPZ files matched: {in_dir}/{args.pattern}")

    written = []
    eid = args.start_id
    for f in files:
        p = convert_one(Path(f), success_dir, eid, env_name=args.env_name,
                        language_override=args.language_override, img_size=args.img_size)
        print(f"[ok] wrote {p}")
        written.append(str(p))
        eid += 1

    meta = {
        "dataset_base": str(out_base),
        "num_successes": len(written),
        "success_files_sample": written[:10] + (["..."] if len(written) > 10 else []),
        "env_name": args.env_name,
        "note": "Generated from rosbag2_to_rlds NPZ using npz_to_simpler_rlds_success.py"
    }
    (out_base / "metadata.json").write_text(json.dumps(meta, indent=2))

    print(f"\nAll done.\nSave location:\n  {success_dir}\nNow run TFDS build with your SimplerEnvSuccessDataset builder.")
    
if __name__ == "__main__":
    main()
