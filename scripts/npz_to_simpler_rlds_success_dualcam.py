#!/usr/bin/env python3
"""
npz_to_simpler_rlds_success_dualcam_npz.py
------------------------------------------
Convert dual-cam NPZ episodes (from rosbag2_to_rlds_npz_dualcam.py) into
a "SimplerEnvSuccessDataset"-style NPZ with success flags—keeping BOTH views.

Output fields per episode:
  - episode_id: int64
  - env_name: str
  - language: str
  - images_primary[T,224,224,3] uint8
  - images_wrist[T,224,224,3]   uint8
  - actions[T,7] float32
  - states[T,8]  float32
  - rewards[T]   float32 (0..0, 1 on last)
  - is_terminal[T] bool
  - is_first[T]    bool
  - images[T,224,224,3] uint8   # alias of images_primary for compatibility
"""

import argparse, json
from pathlib import Path
import numpy as np

def _resize(img: np.ndarray, size) -> np.ndarray:
    if size is None:
        return img
    if img.shape[:2] == (size, size):
        return img
    try:
        import cv2
        return cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
    except Exception:
        from PIL import Image
        return np.array(Image.fromarray(img).resize((size, size), Image.BILINEAR))

def _pick(D, *keys, default=None):
    for k in keys:
        if k in D: return D[k]
    return default

def load_and_align(path: Path, img_size: int):
    D = np.load(path, allow_pickle=True)

    img_p = _pick(D, "image_primary", "images", "full_image", "image_fov")
    img_w = _pick(D, "image_wrist", "wrist_images", "wrist_image")
    actions = _pick(D, "pred_actions", "actions")
    proprio = _pick(D, "proprio", "states", "state", default=None)
    lang    = _pick(D, "language_instruction", "language", "instruction", "task_description", default="")
    if img_p is None or img_w is None:
        raise KeyError("Missing primary or wrist images in NPZ")
    if actions is None:
        raise KeyError("Missing actions (pred_actions/actions) in NPZ")

    # cast types
    img_p = np.asarray(img_p, dtype=np.uint8)
    img_w = np.asarray(img_w, dtype=np.uint8)
    actions = np.asarray(actions, dtype=np.float32)
    proprio = (np.asarray(proprio, dtype=np.float32) if proprio is not None else None)

    # Align lengths to T = len(actions); use first T frames from each view.
    # (Dual-cam NPZ commonly has len(images) == len(actions)+1; we drop the trailing frame.)
    T_imgs = min(int(img_p.shape[0]), int(img_w.shape[0]))
    T = min(int(actions.shape[0]), T_imgs)
    if T <= 0:
        raise ValueError("Episode too short after alignment")

    img_p = img_p[:T]
    img_w = img_w[:T]
    actions = actions[:T]
    if proprio is None:
        states = np.zeros((T, 8), dtype=np.float32)
    else:
        states = proprio[:T].astype(np.float32)

    # Resize both views to 224
    img_p_res = np.stack([_resize(img_p[i], img_size) for i in range(T)], axis=0).astype(np.uint8)
    img_w_res = np.stack([_resize(img_w[i], img_size) for i in range(T)], axis=0).astype(np.uint8)

    return {
        "T": T,
        "images_primary": img_p_res,
        "images_wrist":   img_w_res,
        "actions":        actions,
        "states":         states,
        "language":       str(lang),
    }

def convert_one(in_path: Path, out_success_dir: Path, episode_id: int, env_name: str, img_size: int, language_override: str = None) -> Path:
    out_success_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_success_dir / f"success_episode_{int(episode_id):06d}.npz"
    
    if out_path.exists():
        print(f"[ok] already written {out_path}")
        return out_path

    ep = load_and_align(in_path, img_size)
    if language_override is not None:
        ep["language"] = str(language_override)

    T = ep["T"]
    rewards = np.zeros((T,), dtype=np.float32);  rewards[-1] = 1.0
    is_terminal = np.zeros((T,), dtype=bool);    is_terminal[-1] = True
    is_first = np.zeros((T,), dtype=bool);       is_first[0] = True

    np.savez_compressed(
        out_path,
        episode_id=np.int64(episode_id),
        env_name=str(env_name),
        language=str(ep["language"]),
        images_primary=ep["images_primary"],
        images_wrist=ep["images_wrist"],
        images=ep["images_primary"],      # alias for consumers expecting 'images'
        actions=ep["actions"],
        states=ep["states"],
        rewards=rewards,
        is_terminal=is_terminal,
        is_first=is_first,
    )
    return out_path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_npz_dir", required=True, help="Folder with dual-cam NPZ episodes")
    ap.add_argument("--pattern", default="*.npz")
    ap.add_argument("--out_base", required=True, help="Output base folder (creates successes/ under here)")
    ap.add_argument("--env_name", default="dualcam_robot_task")
    ap.add_argument("--start_id", type=int, default=1)
    ap.add_argument("--img_size", type=int, default=None)
    ap.add_argument("--language_override", default=None)
    ap.add_argument("--success_only", action="store_true", help="Keep NPZs whose 'success' flag is true")
    args = ap.parse_args()

    in_dir = Path(args.in_npz_dir)
    import os
    num_files = len(os.listdir(in_dir))
    files = sorted(in_dir.rglob(args.pattern))
    if not files:
        raise SystemExit(f"No NPZ files matched: {in_dir}/{args.pattern}")

    out_base = Path(args.out_base)
    success_dir = out_base / "successes"
    written = []
    eid = args.start_id
    print(f"Processing {len(files)} files")
    for f in files:
        if args.success_only:
            try:
                D = np.load(f, allow_pickle=True)
                succ = False
                for k in ("success", "is_success", "episode_success"):
                    if k in D and bool(np.array(D[k]).item()): succ = True; break
                if not succ:
                    continue
            except Exception:
                continue

        p = convert_one(f, success_dir, eid, env_name=args.env_name, img_size=args.img_size,
                        language_override=args.language_override)
        print(f"[ok] wrote {p}")
        written.append(str(p))
        eid += 1
    print(f"Processing {len(files)} files")

    meta = {
        "dataset_base": str(out_base),
        "num_successes": len(written),
        "success_files_sample": written[:10] + (["..."] if len(written) > 10 else []),
        "env_name": args.env_name,
        "note": "Dual-cam NPZ successes (images_primary + images_wrist) derived from rosbag2_to_rlds_npz_dualcam.py",
    }
    (out_base / "metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"\nAll done.\nSave location:\n  {success_dir}")
    
if __name__ == "__main__":
    main()
