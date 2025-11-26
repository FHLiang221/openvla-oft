
import argparse
import os
from typing import List, Dict, Any

import numpy as np
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds

# ------------- Configurable keys (tweak if your builder uses different names) -------------
KEY_IMAGE = ("observation", "image")
KEY_WRIST = ("observation", "wrist_image")
KEY_PROP  = ("observation", "proprio")
KEY_ACT   = ("action",)
KEY_REW   = ("reward",)
KEY_FST   = ("is_first",)
KEY_LST   = ("is_last",)
KEY_TERM  = ("is_terminal",)
KEY_LANG  = ("language_instruction",)

def _get_nested(d: Dict[str, Any], keypath: tuple):
    x = d
    for k in keypath:
        x = x[k]
    return x

def as_numpy_steps(episode: Dict[str, Any]) -> List[Dict[str, Any]]:
    #Convert the nested 'steps' dataset to a list of python dicts (numpy).
    # Some TFDS builders return a tf.data.Dataset (works with tfds.as_numpy),
    # others return an IterableDataset (already yields numpy-friendly dicts).
    steps_ds = episode["steps"]
    try:
        return [s for s in tfds.as_numpy(steps_ds)]
    except TypeError:
        # Fallback for IterableDataset
        return [s for s in steps_ds]

def stack_from_steps(steps: List[Dict[str, Any]]):
    images = np.stack([_get_nested(s, KEY_IMAGE) for s in steps], axis=0)
    wrist  = np.stack([_get_nested(s, KEY_WRIST) for s in steps], axis=0)
    prop   = np.stack([_get_nested(s, KEY_PROP)  for s in steps], axis=0)
    acts   = np.stack([_get_nested(s, KEY_ACT)   for s in steps], axis=0)
    rews   = np.array([_get_nested(s, KEY_REW)   for s in steps])
    is_first = np.array([_get_nested(s, KEY_FST) for s in steps], dtype=bool)
    is_last  = np.array([_get_nested(s, KEY_LST) for s in steps], dtype=bool)
    is_term  = np.array([_get_nested(s, KEY_TERM) for s in steps], dtype=bool)
    lang   = steps[0].get(KEY_LANG[-1], "") if len(steps) else ""
    return images, wrist, prop, acts, rews, is_first, is_last, is_term, lang

def quat_to_euler(qx, qy, qz, qw):
    # #Return roll, pitch, yaw (radians) from a quaternion (x,y,z,w).#
    # roll (x-axis rotation)
    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = 1.0 - 2.0 * (qx*qx + qy*qy)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # pitch (y-axis rotation)
    sinp = 2.0 * (qw * qy - qz * qx)
    sinp = np.clip(sinp, -1.0, 1.0)
    pitch = np.arcsin(sinp)

    # yaw (z-axis rotation)
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy*qy + qz*qz)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw

def grid_show_dual(images: np.ndarray, wrist: np.ndarray, max_frames: int = 16, title: str = "", save_path: str = None, start_frame: int = 0):
    # #Show a 2×K grid: top row = primary cam, bottom row = wrist cam.#
    T = min(len(images)-start_frame, max_frames)
    cols = T
    fig, axs = plt.subplots(2, cols, figsize=(cols * 2, 4))
    if cols == 1:
        axs = np.array([[axs[0]], [axs[1]]])
    for i in range(T):
        axs[0, i].imshow(images[i+start_frame])
        axs[0, i].axis("off")
        axs[0, i].set_title(f"t={i+start_frame}")
        axs[1, i].imshow(wrist[i+start_frame])
        axs[1, i].axis("off")
    if title:
        fig.suptitle(title)
    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")
    plt.show()

def plot_time_series(y: np.ndarray, ylabel: str, xlabel: str = "timestep", save_path: str = None):
    # #Generic 1D time series plot (one figure).#
    plt.figure()
    plt.plot(np.arange(len(y)), y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()

def plot_xyz(prop: np.ndarray, save_dir: str = None):
    # #Plot x,y,z vs time (three separate figures).#
    if prop.shape[1] < 3:
        return
    x, y, z = prop[:,0], prop[:,1], prop[:,2]
    sx = None if save_dir is None else os.path.join(save_dir, "x.png")
    sy = None if save_dir is None else os.path.join(save_dir, "y.png")
    sz = None if save_dir is None else os.path.join(save_dir, "z.png")
    plot_time_series(x, "x", save_path=sx)
    plot_time_series(y, "y", save_path=sy)
    plot_time_series(z, "z", save_path=sz)

def plot_rpy_from_quat(prop: np.ndarray, save_dir: str = None):
    #Plot roll, pitch, yaw decoded from quaternion (three separate figures).#
    if prop.shape[1] < 7:
        return
    qx, qy, qz, qw = prop[:,3], prop[:,4], prop[:,5], prop[:,6]
    rpy = np.array([quat_to_euler(qx[i], qy[i], qz[i], qw[i]) for i in range(len(qx))])
    roll, pitch, yaw = rpy[:,0], rpy[:,1], rpy[:,2]
    sr = None if save_dir is None else os.path.join(save_dir, "roll.png")
    sp = None if save_dir is None else os.path.join(save_dir, "pitch.png")
    sy = None if save_dir is None else os.path.join(save_dir, "yaw.png")
    plot_time_series(roll, "roll (rad)", save_path=sr)
    plot_time_series(pitch, "pitch (rad)", save_path=sp)
    plot_time_series(yaw, "yaw (rad)", save_path=sy)

def plot_gripper(prop: np.ndarray, save_dir: str = None):
    if prop.shape[1] < 8:
        return
    grip = prop[:,7]
    sg = None if save_dir is None else os.path.join(save_dir, "gripper.png")
    plot_time_series(grip, "gripper", save_path=sg)

def plot_rewards_flags(rews: np.ndarray, is_first: np.ndarray, is_last: np.ndarray, is_term: np.ndarray, save_dir: str = None):
    pr = None if save_dir is None else os.path.join(save_dir, "reward.png")
    p1 = None if save_dir is None else os.path.join(save_dir, "is_first.png")
    pl = None if save_dir is None else os.path.join(save_dir, "is_last.png")
    pt = None if save_dir is None else os.path.join(save_dir, "is_terminal.png")
    plot_time_series(rews, "reward", save_path=pr)
    plot_time_series(is_first.astype(np.int32), "is_first", save_path=p1)
    plot_time_series(is_last.astype(np.int32), "is_last", save_path=pl)
    plot_time_series(is_term.astype(np.int32), "is_terminal", save_path=pt)

def plot_actions(acts: np.ndarray, save_dir: str = None):
    #Plot each action dimension as its own figure.#
    D = acts.shape[1]
    for d in range(D):
        sp = None if save_dir is None else os.path.join(save_dir, f"action_{d}.png")
        plot_time_series(acts[:, d], f"action[{d}]", save_path=sp)

def main():
    import numpy as np
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="TFDS dataset name (e.g., simpler_env_success_dataset)")
    ap.add_argument("--data_dir", default=None, help="TFDS data_dir (defaults to ~/tensorflow_datasets)")
    ap.add_argument("--episode", type=int, default=0, help="Episode index to visualize")
    ap.add_argument("--max_frames", type=int, default=16, help="Max frames to display in the grid")
    ap.add_argument("--save_dir", default="/home/yachuanh/inspect_rlds", help="Directory to save plotted images")
    ap.add_argument("--start_frame", type=int, default=30, help="Start frame to display in the grid")
    args = ap.parse_args()

    data_dir = os.path.expanduser(args.data_dir) if args.data_dir is not None else None

    # Load episodes (outer dataset). We avoid shuffling to get stable indexing.
    ds = tfds.load(args.dataset, split="train", data_dir=data_dir, shuffle_files=False)

    # Select the Nth episode
    episode_ds = ds.skip(args.episode).take(1)
    episode_list = list(tfds.as_numpy(episode_ds))
    if not episode_list:
        raise ValueError(f"No episode at index {args.episode}.")
    episode = episode_list[0]

    # Prepare save directory
    save_dir = os.path.join(os.path.expanduser(args.save_dir), f"episode_{args.episode}")
    os.makedirs(save_dir, exist_ok=True)

    # Extract nested 'steps' into numpy arrays
    steps = as_numpy_steps(episode)
    if not steps:
        raise ValueError("Selected episode has zero steps.")
    images, wrist, prop, acts, rews, is_first, is_last, is_term, lang = stack_from_steps(steps)

    # Print some metadata / checks
    print(f"Episode {args.episode}: length={len(steps)}")
    print(f"Images shape: {images.shape}, Wrist shape: {wrist.shape}")
    print(f"Proprio shape: {prop.shape}, Actions shape: {acts.shape}")
    print(f"Instruction: {lang!r}")

    # Visualizations (save + show)
    grid_path = os.path.join(save_dir, f"grid_ep{args.episode}.png")
    grid_show_dual(images, wrist, max_frames=args.max_frames, title=f"Episode {args.episode}", save_path=grid_path, start_frame=args.start_frame)
    plot_xyz(prop, save_dir=save_dir)
    plot_rpy_from_quat(prop, save_dir=save_dir)
    plot_gripper(prop, save_dir=save_dir)
    plot_actions(acts, save_dir=save_dir)
    plot_rewards_flags(rews, is_first, is_last, is_term, save_dir=save_dir)
    plt.show()

if __name__ == "__main__":
    main()
