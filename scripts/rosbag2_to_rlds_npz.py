#!/usr/bin/env python3
"""
rosbag2_to_rlds.py  (NPZ + optional TFDS export)
------------------------------------------------
This converter turns ROS 2 bag(s) into:
  • NPZ per-episode files (default), and/or
  • TFDS-style TFRecord files (SequenceExample per episode).

Action sources:
  A) Preferred: integrate EEF Twist commands over image intervals
     - /servo_node/delta_twist_cmds : geometry_msgs/TwistStamped
     - /gripper_cmd                 : std_msgs/Float32 (absolute 0..1)
  B) Fallback: derive ΔEEF pose via TF between consecutive images
     - /tf, /tf_static              : tf2_msgs/TFMessage
     - base_frame, eef_frame        : computed chain T(base->eef)
     - gripper from joint_states    : finger joints mapped to 0..1

Robustness:
  - Uses message header.stamp when available (Image/JointState/TwistStamped)
  - TF lookup tolerates +/- --tf_slop_ms (default 150 ms) per edge.
  - Adds optional proprio: [x,y,z,r,p,y,0.0,gripper] at the start of each interval.

Outputs per episode (NPZ mode):
  image_primary[T-1,H,W,3]  uint8
  actions[T-1,7]            float32  (Δxyz, Δaxis-angle xyz, gripper_abs@t_{i+1})
  timestamps_ns[T-1]        int64
  instruction               str
  proprio[T-1,8]            float32  (if TF available)

Outputs per split (TFDS mode; one shard per split):
  TFRecord SequenceExample per episode with:
    context:
      instruction: bytes
      image_height: int64
      image_width: int64
      image_format: bytes ('jpeg')
      has_proprio: int64 (0/1)
      ep_length: int64 (T-1)
    feature_lists (length = T-1):
      image_primary: bytes (JPEG-encoded RGB)
      action: float_list (7)
      timestamp_ns: int64_list (1)
      is_first: int64_list (0/1)
      is_last: int64_list (0/1)
      proprio: float_list (8)  [only if has_proprio=1]

Usage:
  # NPZ (default) - individual bags
  python3 rosbag2_to_rlds.py --bags bag_ep01 bag_ep02 \
      --out_root datasets/rlds --dataset_name kinova_task --split train \
      --instruction "pick the standing coke can" \
      --base_frame j2n6s200_link_base --eef_frame j2n6s200_end_effector

  # NPZ (default) - folder containing multiple data folders
  python3 rosbag2_to_rlds.py --bags /path/to/data_folder \
      --out_root datasets/rlds --dataset_name kinova_task --split train \
      --instruction "pick the standing coke can" \
      --base_frame j2n6s200_link_base --eef_frame j2n6s200_end_effector

  # TFDS (TFRecords)
  python3 rosbag2_to_rlds.py --bags bag_ep01 bag_ep02 \
      --out_root datasets/tfds --dataset_name kinova_task --split train \
      --instruction "pick the standing coke can" \
      --base_frame j2n6s200_link_base --eef_frame j2n6s200_end_effector \
      --format tfds

  # Both NPZ + TFDS
  python3 rosbag2_to_rlds.py ... --format both
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Lazy TF import (only if TFDS export is requested)
_tf = None
def _ensure_tf():
    global _tf
    if _tf is None:
        import importlib
        _tf = importlib.import_module("tensorflow")

# Optional JPEG encoding via cv2, else fall back to TF encode_jpeg
def _encode_jpeg_rgb(img: np.ndarray, quality: int = 95) -> bytes:
    assert img.dtype == np.uint8 and img.ndim == 3 and img.shape[2] == 3
    try:
        import cv2
        # OpenCV expects BGR
        bgr = img[:, :, ::-1]
        ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
        if not ok:
            raise RuntimeError("cv2.imencode failed")
        return buf.tobytes()
    except Exception:
        _ensure_tf()
        t = _tf.convert_to_tensor(img)  # RGB uint8
        # quality ignored by tf.image.encode_jpeg in some versions; acceptable.
        return bytes(_tf.io.encode_jpeg(t).numpy())

# ---------------- Defaults ----------------
DEFAULT_CAMERA_TOPIC = "/camera/fov/color/image_raw"
DEFAULT_CAMERA_INFO_TOPIC = "/camera/fov/color/camera_info"
DEFAULT_TWIST_TOPIC = "/servo_node/delta_twist_cmds"
DEFAULT_GRIPPER_TOPIC = "/gripper_cmd"
DEFAULT_JOINT_STATES_TOPIC = "/joint_states"
DEFAULT_TF_TOPIC = "/tf"
DEFAULT_TF_STATIC_TOPIC = "/tf_static"

def _ns_to_sec(ns: int) -> float:
    return ns / 1e9

def _hdr_stamp_ns(msg) -> Optional[int]:
    try:
        return int(msg.header.stamp.sec) * 1_000_000_000 + int(msg.header.stamp.nanosec)
    except Exception:
        return None

def _nearest_before_or_last(ts_list: List[int], t_ns: int) -> Optional[int]:
    import bisect
    i = bisect.bisect_right(ts_list, t_ns) - 1
    return i if i >= 0 else None

def _nearest_after(ts_list: List[int], t_ns: int) -> Optional[int]:
    import bisect
    i = bisect.bisect_left(ts_list, t_ns)
    return i if i < len(ts_list) else None

def _quat_to_R(qx, qy, qz, qw) -> np.ndarray:
    w, x, y, z = float(qw), float(qx), float(qy), float(qz)
    R = np.array([
        [1-2*(y*y+z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1-2*(x*x+z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1-2*(x*x+y*y)]
    ], dtype=np.float64)
    return R

def _axis_angle_from_R(R: np.ndarray) -> np.ndarray:
    tr = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    angle = np.arccos(tr)
    if angle < 1e-8:
        return np.zeros(3, dtype=np.float32)
    rx = R[2,1] - R[1,2]
    ry = R[0,2] - R[2,0]
    rz = R[1,0] - R[0,1]
    axis = np.array([rx, ry, rz], dtype=np.float64)
    axis = axis / (2.0*np.sin(angle) + 1e-12)
    return (axis * angle).astype(np.float32)

def _pose_to_T(x, y, z, qx, qy, qz, qw) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = _quat_to_R(qx, qy, qz, qw)
    T[:3, 3] = [x, y, z]
    return T

def _decompose_T(T: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return T[:3, 3].astype(np.float32), T[:3, :3].astype(np.float64)

def _R_to_rpy_xyz(R: np.ndarray) -> Tuple[float, float, float]:
    r = float(np.arctan2(R[2,1], R[2,2]))
    p = float(np.arcsin(-R[2,0]))
    y = float(np.arctan2(R[1,0], R[0,0]))
    return r, p, y

# ---------------- Readers ----------------

def read_bag_rosbag2_py(bag_path: Path, topics: List[str]) -> Dict[str, List[Tuple[int, object]]]:
    from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message

    storage_options = StorageOptions(uri=str(bag_path), storage_id="sqlite3")
    converter_options = ConverterOptions("", "")

    reader = SequentialReader()
    reader.open(storage_options, converter_options)

    type_map = {}
    for conn in reader.get_all_topics_and_types():
        if conn.name in topics:
            type_map[conn.name] = get_message(conn.type)

    out = {t: [] for t in topics}
    while reader.has_next():
        (topic, data, _bag_t_ns) = reader.read_next()
        if topic not in type_map:
            continue
        try:
            msg_type = type_map[topic]
            msg = deserialize_message(data, msg_type)
            t_ns = _hdr_stamp_ns(msg)
            if t_ns is None:
                t_ns = int(_bag_t_ns)
            out[topic].append((int(t_ns), msg))
        except Exception:
            pass
    return out

def read_bag_rosbags(bag_path: Path, topics: List[str]) -> Dict[str, List[Tuple[int, object]]]:
    from rosbags.highlevel import AnyReader
    out = {t: [] for t in topics}
    with AnyReader([bag_path]) as reader:
        conns = {c.id: c for c in reader.connections if c.topic in topics}
        for conn, _bag_ts, raw in reader.messages(connections=conns.values()):
            msg = reader.deserialize(raw, conn.msgtype)
            t_ns = _hdr_stamp_ns(msg)
            if t_ns is None:
                t_ns = int(_bag_ts)
            out[conn.topic].append((int(t_ns), msg))
    return out

# ---------------- Episode extraction ----------------

@dataclass
class EpisodeData:
    image_times: List[int]
    images: List[np.ndarray]
    twist_times: List[int]
    twists: List[np.ndarray]
    grip_times: List[int]
    grips: List[float]
    tf_dynamic: List[Tuple[int, object]]
    tf_static: List[object]
    joint_times: List[int]
    joints: List[Dict[str, float]]
    instruction: str

def extract_episode(
    bag_dir: Path,
    camera_topic: str,
    camera_info_topic: Optional[str],
    twist_topic: str,
    gripper_topic: Optional[str],
    joint_states_topic: str,
    tf_topic: str,
    tf_static_topic: str,
    instruction: str,
) -> EpisodeData:
    topics = [camera_topic, twist_topic, joint_states_topic, tf_topic, tf_static_topic]
    if camera_info_topic:
        topics.append(camera_info_topic)
    if gripper_topic:
        topics.append(gripper_topic)

    try:
        data = read_bag_rosbag2_py(bag_dir, topics)
    except Exception:
        data = read_bag_rosbags(bag_dir, topics)

    if camera_topic not in data or len(data[camera_topic]) < 2:
        raise RuntimeError(f"No or insufficient images on topic {camera_topic}")

    images, image_times = [], []
    for t_ns, msg in data[camera_topic]:
        H, W = int(msg.height), int(msg.width)
        arr = np.frombuffer(bytes(msg.data), dtype=np.uint8)
        step = int(getattr(msg, 'step', W*3))
        if step == W:
            img = arr.reshape(H, W, 1); img = np.repeat(img, 3, axis=2)
        else:
            ch = step // W; img = arr.reshape(H, W, ch)
            enc = getattr(msg, 'encoding', 'rgb8').lower()
            if enc.startswith('rgb'): img = img[..., :3] if ch >= 3 else np.repeat(img, 3, axis=2)
            elif enc.startswith('bgr'): img = (img[..., :3])[:, :, ::-1] if ch >= 3 else np.repeat(img, 3, axis=2)
            elif enc in ('rgba8', 'bgra8'): img = img[..., :3];  img = img[:, :, ::-1] if enc=='bgra8' else img
            else: img = img[..., :3] if ch >= 3 else np.repeat(img, 3, axis=2)
        images.append(img.copy()); image_times.append(int(t_ns))

    # Twists
    twist_times, twists = [], []
    for t_ns, msg in data.get(twist_topic, []):
        tw = msg.twist if hasattr(msg, "twist") else msg
        v = [tw.linear.x, tw.linear.y, tw.linear.z, tw.angular.x, tw.angular.y, tw.angular.z]
        twist_times.append(int(t_ns)); twists.append(np.array(v, dtype=np.float32))
    order = np.argsort(twist_times); twist_times = [twist_times[i] for i in order]; twists = [twists[i] for i in order]

    # Gripper command (if any)
    grip_times, grips = [], []
    if gripper_topic and len(data.get(gripper_topic, [])) > 0:
        for t_ns, msg in data[gripper_topic]:
            val = float(msg.data) if hasattr(msg, "data") else float(msg)
            grip_times.append(int(t_ns)); grips.append(val)
        order = np.argsort(grip_times); grip_times = [grip_times[i] for i in order]; grips = [grips[i] for i in order]

    tf_dynamic = data.get(tf_topic, [])
    tf_static = [msg for _, msg in data.get(tf_static_topic, [])]

    joint_times, joints = [], []
    for t_ns, msg in data.get(joint_states_topic, []):
        pos_map = {n: p for n, p in zip(list(msg.name), list(msg.position))}
        joint_times.append(int(t_ns)); joints.append(pos_map)

    return EpisodeData(image_times, images, twist_times, twists, grip_times, grips, tf_dynamic, tf_static, joint_times, joints, instruction)

# ---------------- TF utilities ----------------

def _gather_tf_maps(tf_dynamic_msgs: List[Tuple[int, object]], tf_static_msgs: List[object]):
    from collections import defaultdict
    dyn_map = defaultdict(list); dyn_times = defaultdict(list); static_map = {}
    for _bag_t, tfmsg in tf_dynamic_msgs:
        for ts in getattr(tfmsg, 'transforms', []):
            p = ts.header.frame_id.strip('/'); c = ts.child_frame_id.strip('/')
            tr = ts.transform.translation; q = ts.transform.rotation
            t_ns = int(ts.header.stamp.sec)*1_000_000_000 + int(ts.header.stamp.nanosec)
            dyn_times[(p, c)].append(t_ns)
            dyn_map[(p, c)].append((tr.x, tr.y, tr.z, q.x, q.y, q.z, q.w))
    for tfmsg in tf_static_msgs:
        for ts in getattr(tfmsg, 'transforms', []):
            p = ts.header.frame_id.strip('/'); c = ts.child_frame_id.strip('/')
            tr = ts.transform.translation; q = ts.transform.rotation
            static_map[(p, c)] = (tr.x, tr.y, tr.z, q.x, q.y, q.z, q.w)
    for k in list(dyn_map.keys()):
        order = np.argsort(np.array(dyn_times[k]))
        dyn_times[k] = [dyn_times[k][i] for i in order]
        dyn_map[k] = [dyn_map[k][i] for i in order]
    return dyn_map, dyn_times, static_map

def _find_path(parents_children, src: str, dst: str) -> Optional[List[Tuple[str,str]]]:
    from collections import deque, defaultdict
    g = defaultdict(list)
    for (p,c) in parents_children: g[p].append(c)
    q = deque([src]); prev = {src: None}; via = {}
    while q:
        u = q.popleft()
        if u == dst: break
        for v in g.get(u, []):
            if v not in prev:
                prev[v] = u; via[v] = (u, v); q.append(v)
    if dst not in prev: return None
    path_edges = []; cur = dst
    while prev[cur] is not None:
        path_edges.append(via[cur]); cur = prev[cur]
    path_edges.reverse(); return path_edges

def _lookup_edge_T_at_time(p, c, dyn_map, dyn_times, static_map, t_ns: int, slop_ns: int) -> Optional[np.ndarray]:
    if (p, c) in dyn_map:
        ts = dyn_times[(p, c)]
        i_before = _nearest_before_or_last(ts, t_ns)
        i_after  = _nearest_after(ts, t_ns)
        cand = []
        if i_before is not None:
            dt = abs(t_ns - ts[i_before]); cand.append((dt, dyn_map[(p,c)][i_before]))
        if i_after is not None:
            dt = abs(ts[i_after] - t_ns); cand.append((dt, dyn_map[(p,c)][i_after]))
        if not cand: return None
        dt_min, pose = min(cand, key=lambda x: x[0])
        if dt_min > slop_ns: return None
        x,y,z,qx,qy,qz,qw = pose
        return _pose_to_T(x,y,z,qx,qy,qz,qw)
    elif (p, c) in static_map:
        x,y,z,qx,qy,qz,qw = static_map[(p, c)]
        return _pose_to_T(x,y,z,qx,qy,qz,qw)
    else:
        return None

def _lookup_T_at_time(path_edges, dyn_map, dyn_times, static_map, t_ns: int, slop_ns: int) -> Optional[np.ndarray]:
    T = np.eye(4, dtype=np.float64)
    for (p,c) in path_edges:
        Te = _lookup_edge_T_at_time(p, c, dyn_map, dyn_times, static_map, t_ns, slop_ns)
        if Te is None: return None
        T = T @ Te
    return T

# ---------------- Builders ----------------

def _build_gripper_series_from_joints(joint_times: List[int], joints: List[Dict[str,float]],
                                      finger_joints: List[str], finger_open: List[float], finger_closed: List[float]):
    grip_t, grip_v = [], []
    if len(joint_times) == 0: return grip_t, grip_v
    last = 0.0
    for t_ns, pos_map in zip(joint_times, joints):
        vals = []
        for i, name in enumerate(finger_joints):
            if name in pos_map and i < len(finger_open) and i < len(finger_closed):
                op, cl = finger_open[i], finger_closed[i]
                den = (cl - op) if abs(cl-op) > 1e-6 else 1.0
                frac = np.clip((pos_map[name] - op)/den, 0.0, 1.0)
                vals.append(float(frac))
        if vals:
            last = float(np.mean(vals))
        grip_t.append(int(t_ns)); grip_v.append(last)
    return grip_t, grip_v

def derive_actions_and_proprio_from_tf(
    ep: EpisodeData,
    base_frame: str,
    eef_frame: str,
    finger_joints: List[str],
    finger_open: List[float],
    finger_closed: List[float],
    slop_ns: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    base = base_frame.strip('/'); eef = eef_frame.strip('/')
    dyn_map, dyn_times, static_map = _gather_tf_maps(ep.tf_dynamic, ep.tf_static)
    edges = list(dyn_map.keys()) + list(static_map.keys())
    path_edges = _find_path(edges, base, eef)
    if path_edges is None: raise RuntimeError(f"No TF path from {base} to {eef}.")

    grip_t, grip_v = _build_gripper_series_from_joints(ep.joint_times, ep.joints, finger_joints, finger_open, finger_closed)

    T = len(ep.images)
    actions = np.zeros((T-1, 7), dtype=np.float32)
    proprio = np.zeros((T-1, 8), dtype=np.float32)
    image_primary = np.zeros((T-1, ep.images[0].shape[0], ep.images[0].shape[1], 3), dtype=np.uint8)
    timestamps_ns = np.zeros((T-1,), dtype=np.int64)

    poses_T = []
    for t_ns in ep.image_times:
        T_be = _lookup_T_at_time(path_edges, dyn_map, dyn_times, static_map, t_ns, slop_ns)
        if T_be is None:
            raise RuntimeError(f"Missing TF for {base}->{eef} near time {t_ns} (slop={slop_ns/1e6:.1f}ms).")
        poses_T.append(T_be)

    for i in range(T-1):
        t0 = ep.image_times[i]; t1 = ep.image_times[i+1]
        T0 = poses_T[i]; T1 = poses_T[i+1]
        p0, R0 = _decompose_T(T0); p1, R1 = _decompose_T(T1)
        dp = (p1 - p0).astype(np.float32); aa = _axis_angle_from_R(R0.T @ R1)

        # gripper at start/end
        g0 = 0.0; g1 = 0.0
        if len(grip_t) > 0:
            i0 = _nearest_before_or_last(grip_t, t0); g0 = grip_v[i0] if i0 is not None else g0
            i1 = _nearest_before_or_last(grip_t, t1); g1 = grip_v[i1] if i1 is not None else g0

        actions[i, :3] = dp; actions[i, 3:6] = aa; actions[i, 6] = float(g1)
        r, p, y = _R_to_rpy_xyz(R0)
        proprio[i, :6] = np.array([p0[0], p0[1], p0[2], r, p, y], dtype=np.float32)
        proprio[i, 6] = 0.0; proprio[i, 7] = float(g0)
        image_primary[i] = ep.images[i]; timestamps_ns[i] = t0

    return image_primary, actions, timestamps_ns, proprio

def integrate_twist_over_interval(
    t0_ns: int, t1_ns: int, twist_times: List[int], twists: List[np.ndarray], last_twist_idx_cache: int
) -> Tuple[np.ndarray, int]:
    if t1_ns <= t0_ns:
        return np.zeros(6, dtype=np.float32), last_twist_idx_cache
    dt = _ns_to_sec(t1_ns - t0_ns)
    start_idx = last_twist_idx_cache + 1; end_idx = start_idx; N = len(twist_times)
    while end_idx < N and twist_times[end_idx] <= t1_ns: end_idx += 1
    if end_idx - start_idx <= 0:
        idx = _nearest_before_or_last(twist_times, t1_ns)
        if idx is None: return np.zeros(6, dtype=np.float32), last_twist_idx_cache
        mean_twist = twists[idx]; return (mean_twist * dt).astype(np.float32), idx
    else:
        window_twists = np.stack(twists[start_idx:end_idx], axis=0)
        mean_twist = window_twists.mean(axis=0); return (mean_twist * dt).astype(np.float32), (end_idx - 1)

def build_rlds_arrays(
    ep: EpisodeData,
    mode: str,
    base_frame: str,
    eef_frame: str,
    finger_joints: List[str],
    finger_open: List[float],
    finger_closed: List[float],
    slop_ns: int,
) -> Dict[str, np.ndarray]:
    if len(ep.images) < 2:
        raise RuntimeError("Need at least 2 images to form actions.")

    if mode == "twist" and len(ep.twist_times) > 0:
        arrays: Dict[str, np.ndarray] = {}
        T = len(ep.images)
        actions = np.zeros((T - 1, 7), dtype=np.float32)
        image_primary = np.zeros((T - 1, ep.images[0].shape[0], ep.images[0].shape[1], 3), dtype=np.uint8)
        timestamps_ns = np.zeros((T - 1,), dtype=np.int64)
        last_tw_idx = -1; last_gr = 0.0
        for i in range(T - 1):
            t0 = ep.image_times[i]; t1 = ep.image_times[i + 1]
            d6, last_tw_idx = integrate_twist_over_interval(t0, t1, ep.twist_times, ep.twists, last_twist_idx_cache=last_tw_idx)
            g_abs = last_gr
            if len(ep.grip_times) > 0:
                idx = _nearest_before_or_last(ep.grip_times, t1)
                if idx is not None: g_abs = ep.grips[idx]
            last_gr = g_abs
            actions[i, :6] = d6; actions[i, 6] = g_abs
            image_primary[i] = ep.images[i]; timestamps_ns[i] = t0
        arrays["image_primary"] = image_primary
        arrays["actions"] = actions
        arrays["timestamps_ns"] = timestamps_ns
        arrays["instruction"] = np.array(ep.instruction)
        # optional proprio via TF
        if len(ep.tf_dynamic) > 0 or len(ep.tf_static) > 0:
            try:
                _, _, _, proprio = derive_actions_and_proprio_from_tf(
                    ep, base_frame, eef_frame, finger_joints, finger_open, finger_closed, slop_ns
                ); arrays["proprio"] = proprio
            except Exception:
                pass
        return arrays
    else:
        image_primary, actions, timestamps_ns, proprio = derive_actions_and_proprio_from_tf(
            ep, base_frame, eef_frame, finger_joints, finger_open, finger_closed, slop_ns
        )
        return {
            "image_primary": image_primary,
            "actions": actions,
            "timestamps_ns": timestamps_ns,
            "instruction": np.array(ep.instruction),
            "proprio": proprio,
        }

def write_episode_npz(out_dir: Path, idx: int, arrays: Dict[str, np.ndarray]) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    ep_path = out_dir / f"episode_{idx:05d}.npz"
    np.savez_compressed(ep_path, **arrays)
    return ep_path

# ---------------- TFDS writer ----------------

def _feat_int64(v: int):
    return _tf.train.Feature(int64_list=_tf.train.Int64List(value=[int(v)]))

def _feat_bytes(b: bytes):
    return _tf.train.Feature(bytes_list=_tf.train.BytesList(value=[b]))

def _feat_float_list(arr):
    return _tf.train.Feature(float_list=_tf.train.FloatList(value=[float(x) for x in arr]))

def _feat_int64_list(arr):
    return _tf.train.Feature(int64_list=_tf.train.Int64List(value=[int(x) for x in arr]))

def write_split_tfds(out_dir: Path, split: str, episodes: List[Dict[str, np.ndarray]], instruction: str, jpeg_quality: int = 95) -> Path:
    _ensure_tf()
    split_dir = out_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)
    tfrec_path = split_dir / f"{split}-00000-of-00001.tfrecord"
    with _tf.io.TFRecordWriter(str(tfrec_path)) as w:
        for ep in episodes:
            imgs = ep["image_primary"]
            H, W = imgs.shape[1], imgs.shape[2]
            Tm1 = imgs.shape[0]
            actions = ep["actions"]
            ts = ep["timestamps_ns"]
            has_prop = "proprio" in ep
            prop = ep.get("proprio", None)

            flists = {}
            # images
            img_feats = []
            for i in range(Tm1):
                jpeg = _encode_jpeg_rgb(imgs[i], quality=jpeg_quality)
                img_feats.append(_tf.train.Feature(bytes_list=_tf.train.BytesList(value=[jpeg])))
            flists["image_primary"] = _tf.train.FeatureList(feature=img_feats)
            # actions
            act_feats = [_feat_float_list(actions[i]) for i in range(Tm1)]
            flists["action"] = _tf.train.FeatureList(feature=act_feats)
            # timestamps
            ts_feats = [_feat_int64_list([int(ts[i])]) for i in range(Tm1)]
            flists["timestamp_ns"] = _tf.train.FeatureList(feature=ts_feats)
            # first/last flags
            first_feats = [_feat_int64_list([1 if i==0 else 0]) for i in range(Tm1)]
            last_feats  = [_feat_int64_list([1 if i==Tm1-1 else 0]) for i in range(Tm1)]
            flists["is_first"] = _tf.train.FeatureList(feature=first_feats)
            flists["is_last"]  = _tf.train.FeatureList(feature=last_feats)
            # proprio (optional)
            if has_prop:
                prop_feats = [_feat_float_list(prop[i]) for i in range(Tm1)]
                flists["proprio"] = _tf.train.FeatureList(feature=prop_feats)

            context = _tf.train.Features(feature={
                "instruction": _feat_bytes(instruction.encode("utf-8")),
                "image_height": _feat_int64(H),
                "image_width": _feat_int64(W),
                "image_format": _feat_bytes(b"jpeg"),
                "has_proprio": _feat_int64(1 if has_prop else 0),
                "ep_length": _feat_int64(Tm1),
            })
            ex = _tf.train.SequenceExample(
                context=context,
                feature_lists=_tf.train.FeatureLists(feature_list=flists)
            )
            w.write(ex.SerializeToString())

    # minimal dataset info
    info = {
        "name": out_dir.parent.name,
        "split": split,
        "num_episodes": len(episodes),
        "image_format": "jpeg",
        "action_shape": [7],
        "proprio_shape": [8],
    }
    with open(split_dir / "dataset_info.json", "w") as f:
        json.dump(info, f, indent=2)
    return tfrec_path

# ---------------- CLI ----------------

def parse_float_list(arg: str) -> List[float]:
    return [float(x.strip()) for x in arg.split(',')] if arg else []

def parse_str_list(arg: str) -> List[str]:
    return [x.strip() for x in arg.split(',')] if arg else []

def discover_data_folders(folder_path: Path) -> List[Path]:
    """
    Discover all data folders (rosbag directories) within the given folder path.
    A data folder is considered valid if it contains rosbag metadata files.
    """
    data_folders = []
    
    if not folder_path.exists():
        raise FileNotFoundError(f"Folder path does not exist: {folder_path}")
    
    if not folder_path.is_dir():
        raise ValueError(f"Path is not a directory: {folder_path}")
    
    # Look for rosbag directories - they typically contain metadata.db3 or metadata.yaml
    for item in folder_path.iterdir():
        if item.is_dir():
            # Check if this looks like a rosbag directory
            rosbag_indicators = ['metadata.db3', 'metadata.yaml', 'metadata.db']
            if any((item / indicator).exists() for indicator in rosbag_indicators):
                data_folders.append(item)
            else:
                # Also check for common rosbag file patterns
                rosbag_files = list(item.glob('*.db3')) + list(item.glob('*.yaml')) + list(item.glob('*.db'))
                if rosbag_files:
                    data_folders.append(item)
    
    # Sort for consistent ordering
    data_folders.sort()
    return data_folders

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bags", nargs="+", required=True, 
                   help="Individual bag paths or a single folder path containing multiple data folders")
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--dataset_name", required=True)
    ap.add_argument("--split", default="train", choices=["train", "val"])
    ap.add_argument("--instruction", required=True)
    # Topics
    ap.add_argument("--camera_topic", default=DEFAULT_CAMERA_TOPIC)
    ap.add_argument("--camera_info_topic", default=DEFAULT_CAMERA_INFO_TOPIC)
    ap.add_argument("--twist_topic", default=DEFAULT_TWIST_TOPIC)
    ap.add_argument("--gripper_topic", default=DEFAULT_GRIPPER_TOPIC)
    ap.add_argument("--joint_states_topic", default=DEFAULT_JOINT_STATES_TOPIC)
    ap.add_argument("--tf_topic", default=DEFAULT_TF_TOPIC)
    ap.add_argument("--tf_static_topic", default=DEFAULT_TF_STATIC_TOPIC)
    # TF / proprio options
    ap.add_argument("--base_frame", default="base_link")
    ap.add_argument("--eef_frame", default="tool_link")
    ap.add_argument("--finger_joints", default="finger_1_joint,finger_2_joint,finger_3_joint")
    ap.add_argument("--finger_open", default="0.0,0.0,0.0")
    ap.add_argument("--finger_closed", default="1.0,1.0,1.0")
    ap.add_argument("--tf_slop_ms", type=float, default=150.0, help="Max TF time mismatch allowed per edge (ms).")
    # Output
    ap.add_argument("--format", choices=["npz", "tfds", "both"], default="npz")
    ap.add_argument("--jpeg_quality", type=int, default=95)
    # Force mode
    ap.add_argument("--force_mode", choices=["twist", "tf"], default=None)
    args = ap.parse_args()

    out_ds_dir = Path(args.out_root) / args.dataset_name
    out_split_dir = out_ds_dir / args.split
    out_split_dir.mkdir(parents=True, exist_ok=True)

    finger_joints = parse_str_list(args.finger_joints)
    finger_open = parse_float_list(args.finger_open)
    finger_closed = parse_float_list(args.finger_closed)
    slop_ns = int(args.tf_slop_ms * 1e6)

    # Determine if we're processing individual bags or a folder of bags
    if len(args.bags) == 1:
        # Check if the single argument is a directory
        potential_folder = Path(args.bags[0])
        if potential_folder.is_dir():
            print(f"[discover] Scanning folder for data directories: {potential_folder}")
            bag_paths = discover_data_folders(potential_folder)
            if not bag_paths:
                raise RuntimeError(f"No data folders found in directory: {potential_folder}")
            print(f"[discover] Found {len(bag_paths)} data folders")
        else:
            # Single bag file/directory
            bag_paths = [potential_folder]
    else:
        # Multiple individual bag paths
        bag_paths = [Path(bag) for bag in args.bags]

    manifest = []
    tfds_buffer = []  # keep per-episode dicts before writing TFRecord
    failed_folders = []  # track failed processing attempts
    
    for i, bag_dir in enumerate(bag_paths):
        print(f"[convert] {i+1}/{len(bag_paths)}: {bag_dir}")
        
        try:
            ep = extract_episode(
                bag_dir=bag_dir,
                camera_topic=args.camera_topic,
                camera_info_topic=args.camera_info_topic,
                twist_topic=args.twist_topic,
                gripper_topic=args.gripper_topic,
                joint_states_topic=args.joint_states_topic,
                tf_topic=args.tf_topic,
                tf_static_topic=args.tf_static_topic,
                instruction=args.instruction,
            )
            mode = args.force_mode or ("twist" if len(ep.twist_times) > 0 else "tf")
            arrays = build_rlds_arrays(
                ep, mode, args.base_frame, args.eef_frame, finger_joints, finger_open, finger_closed, slop_ns
            )
        except Exception as e:
            error_msg = f"Failed to process {bag_dir}: {str(e)}"
            print(f"  ERROR: {error_msg}")
            failed_folders.append({
                "folder": str(bag_dir),
                "error": str(e),
                "error_type": type(e).__name__
            })
            continue

        if args.format in ("npz", "both"):
            ep_path = write_episode_npz(out_split_dir, i, arrays)
            manifest.append({
                "episode": ep_path.name,
                "num_steps": int(arrays["actions"].shape[0]),
                "mode": mode,
                "has_proprio": bool("proprio" in arrays),
            })
            print(f"  -> NPZ: steps={manifest[-1]['num_steps']}, proprio={manifest[-1]['has_proprio']} (tf_slop={args.tf_slop_ms}ms)")

        if args.format in ("tfds", "both"):
            tfds_buffer.append(arrays)

    # meta.json (NPZ path)
    if args.format in ("npz", "both"):
        meta_path = out_split_dir.parent / "meta.json"
        meta = {
            "dataset_name": args.dataset_name,
            "splits": {args.split: [m["episode"] for m in manifest]},
            "keys": ["image_primary", "actions", "timestamps_ns", "instruction"] + (["proprio"] if any(m.get("has_proprio", False) for m in manifest) else []),
            "action_format": "delta_xyz, delta_axisangle_xyz, gripper_abs",
            "proprio_format": "[x,y,z, roll,pitch,yaw, 0.0, gripper_abs] at start image",
            "note": "Timestamps from message headers. TF lookup tolerates +/- tf_slop_ms per edge.",
        }
        if meta_path.exists():
            with open(meta_path, "r") as f:
                old = json.load(f)
            old_splits = old.get("splits", {}); old_splits.update(meta["splits"]); old["splits"] = old_splits
            if "proprio" in meta["keys"] and "proprio" not in old.get("keys", []):
                old["keys"] = list(sorted(set(old.get("keys", []) + ["proprio"])))
            meta = old
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        print(f"[meta] {meta_path}")

    # TFDS writer
    if args.format in ("tfds", "both"):
        tfrec_path = write_split_tfds(out_ds_dir, args.split, tfds_buffer, instruction=args.instruction, jpeg_quality=args.jpeg_quality)
        print(f"[tfds] wrote {len(tfds_buffer)} episodes -> {tfrec_path}")

    # Summary statistics
    total_folders = len(bag_paths)
    processed_folders = len(manifest)
    failed_count = len(failed_folders)
    
    print(f"\n[SUMMARY] Processing completed:")
    print(f"  Total folders found: {total_folders}")
    print(f"  Successfully processed: {processed_folders}")
    print(f"  Failed to process: {failed_count}")
    
    if failed_folders:
        print(f"\n[FAILED FOLDERS] Details of {failed_count} failed processing attempts:")
        for i, failure in enumerate(failed_folders, 1):
            print(f"  {i}. {failure['folder']}")
            print(f"     Error: {failure['error_type']}: {failure['error']}")
        
        # Save failed folders to a log file
        failed_log_path = out_split_dir.parent / f"failed_folders_{args.split}.json"
        with open(failed_log_path, "w") as f:
            json.dump(failed_folders, f, indent=2)
        print(f"\n[LOG] Failed folders details saved to: {failed_log_path}")
    
    print(f"\n[done] format={args.format}  split_dir={out_split_dir}")

if __name__ == "__main__":
    main()
