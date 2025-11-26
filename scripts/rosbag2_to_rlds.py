#!/usr/bin/env python3
"""
bag_to_tfds_for_adv_finetune.py  (RLDS-style TFDS writer)
---------------------------------------------------------
Converts ROS 2 bag(s) with:
  /camera/camera/color/image_raw
  /camera/camera/color/camera_info
  /joint_states
  /tf, /tf_static
into **TFRecord (TFDS/RLDS-style)** with per-step fields aligned to the
"SimplerEnvSuccessDataset" schema (image PNG @ 224x224, proprio, action, flags).

FeatureLists (length = T steps):
  steps/observation/image              bytes (PNG-encoded RGB, shape 224x224x3)
  steps/observation/proprio            float[8]  ([x,y,z,qx,qy,qz,qw,grip])
  steps/action                         float[7]
  steps/discount                       float[1]  (default 1.0)
  steps/reward                         float[1]  (default 0 except last=1.0)
  steps/is_first                       int64[1]  (1 on first)
  steps/is_last                        int64[1]  (1 on last)
  steps/is_terminal                    int64[1]  (default equals is_last)
  steps/language_instruction           bytes     (same string each step)
  steps/language_embedding             float[512]  (optional; --add_language_embedding)

Context:
  episode_metadata/file_path           bytes  (bag path)
  episode_metadata/episode_id          int64  (sequential index)
  episode_metadata/env_name            bytes  (optional, via --env_name)

This matches the structure expected by the SimplerEnvSuccessDataset builder
(images as PNG 224x224, proprio 8D, action 7D, flags, language), so you can
mix/compare robot data with the SimplerEnv TFDS.  See the SimplerEnv builder
docs in your repo.
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---- PNG encoder (cv2 -> PIL -> tensorflow) and resizer ----
def encode_png_rgb(img: np.ndarray) -> bytes:
    assert img.dtype == np.uint8 and img.ndim == 3 and img.shape[2] == 3
    try:
        from io import BytesIO
        from PIL import Image
        bio = BytesIO()
        Image.fromarray(img).save(bio, format="PNG")
        return bio.getvalue()
    except Exception:
        pass
    try:
        import cv2
        ok, buf = cv2.imencode(".png", img[:, :, ::-1])  # cv2 expects BGR
        if ok:
            return buf.tobytes()
    except Exception:
        pass
    try:
        import tensorflow as tf
        t = tf.convert_to_tensor(img)  # RGB uint8
        return bytes(tf.io.encode_png(t).numpy())
    except Exception as e:
        raise RuntimeError(f"No PNG encoder available (cv2/PIL/tensorflow). Error: {e}")

def resize_to_224(img: np.ndarray) -> np.ndarray:
    if img.shape[0] == 224 and img.shape[1] == 224:
        return img
    try:
        from PIL import Image
        pil_img = Image.fromarray(img).resize((224,224), Image.BILINEAR)
        buf = pil_img.tobytes()
        return np.frombuffer(buf, dtype=np.uint8).reshape((224,224,3))
    except Exception:
        try:
            import cv2
            return cv2.resize(img, (224, 224), interpolation=cv2.INTER_LINEAR)
        except Exception as e:
            raise RuntimeError(f"No image resizer (PIL/cv2). Error: {e}")

# ------------ Defaults (match your topics) ------------
DEFAULT_CAMERA_TOPIC = "/camera/camera/color/image_raw"
DEFAULT_CAMERA_INFO_TOPIC = "/camera/camera/color/camera_info"
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

def _pose_to_T(x, y, z, qx, qy, qz, qw) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = _quat_to_R(qx, qy, qz, qw)
    T[:3, 3] = [x, y, z]
    return T

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

def _decompose_T(T: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return T[:3, 3].astype(np.float32), T[:3, :3].astype(np.float64)

# ------------ Bag readers (rosbag2_py or rosbags) ------------

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

# ------------ Episode extraction ------------

@dataclass
class EpisodeData:
    image_times: List[int]
    images: List[np.ndarray]
    tf_dynamic: List[Tuple[int, object]]
    tf_static: List[object]
    joint_times: List[int]
    joints: List[Dict[str, float]]
    instruction: str
    env_name: str

def extract_episode(
    bag_dir: Path,
    camera_topic: str,
    camera_info_topic: Optional[str],
    joint_states_topic: str,
    tf_topic: str,
    tf_static_topic: str,
    instruction: str,
    env_name: str,
) -> EpisodeData:
    topics = [camera_topic, joint_states_topic, tf_topic, tf_static_topic]
    if camera_info_topic:
        topics.append(camera_info_topic)

    try:
        data = read_bag_rosbag2_py(bag_dir, topics)
    except Exception:
        data = read_bag_rosbags(bag_dir, topics)

    if camera_topic not in data or len(data[camera_topic]) < 1:
        raise RuntimeError(f"No images on topic {camera_topic}")

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

    tf_dynamic = data.get(tf_topic, [])
    tf_static = [msg for _, msg in data.get(tf_static_topic, [])]

    joint_times, joints = [], []
    for t_ns, msg in data.get(joint_states_topic, []):
        pos_map = {n: p for n, p in zip(list(msg.name), list(msg.position))}
        joint_times.append(int(t_ns)); joints.append(pos_map)

    return EpisodeData(image_times, images, tf_dynamic, tf_static, joint_times, joints, instruction, env_name)

# ------------ TF utilities ------------

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

def _pose_to_T_from_tuple(pose_tuple):
    x,y,z,qx,qy,qz,qw = pose_tuple
    return _pose_to_T(x,y,z,qx,qy,qz,qw)

def _nearest_before_or_last(ts_list: List[int], t_ns: int) -> Optional[int]:
    import bisect
    i = bisect.bisect_right(ts_list, t_ns) - 1
    return i if i >= 0 else None

def _nearest_after(ts_list: List[int], t_ns: int) -> Optional[int]:
    import bisect
    i = bisect.bisect_left(ts_list, t_ns)
    return i if i < len(ts_list) else None

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
        return _pose_to_T_from_tuple(pose)
    elif (p, c) in static_map:
        return _pose_to_T_from_tuple(static_map[(p, c)])
    else:
        return None

def _lookup_T_at_time(path_edges, dyn_map, dyn_times, static_map, t_ns: int, slop_ns: int) -> Optional[np.ndarray]:
    T = np.eye(4, dtype=np.float64)
    for (p,c) in path_edges:
        Te = _lookup_edge_T_at_time(p, c, dyn_map, dyn_times, static_map, t_ns, slop_ns)
        if Te is None: return None
        T = T @ Te
    return T

def _R_to_rpy_xyz(R: np.ndarray) -> Tuple[float, float, float]:
    r = float(np.arctan2(R[2,1], R[2,2]))
    p = float(np.arcsin(-R[2,0]))
    y = float(np.arctan2(R[1,0], R[0,0]))
    return r, p, y

# ------------ Builders ------------

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
        if vals: last = float(np.mean(vals))
        grip_t.append(int(t_ns)); grip_v.append(last)
    return grip_t, grip_v

def derive_steps_from_tf(
    ep: EpisodeData,
    base_frame: str,
    eef_frame: str,
    finger_joints: List[str],
    finger_open: List[float],
    finger_closed: List[float],
    slop_ns: int,
) -> Dict[str, np.ndarray]:
    base = base_frame.strip('/'); eef = eef_frame.strip('/')
    dyn_map, dyn_times, static_map = _gather_tf_maps(ep.tf_dynamic, ep.tf_static)
    edges = list(dyn_map.keys()) + list(static_map.keys())
    path_edges = _find_path(edges, base, eef)
    if path_edges is None: raise RuntimeError(f"No TF path from {base} to {eef}.")

    grip_t, grip_v = _build_gripper_series_from_joints(ep.joint_times, ep.joints, finger_joints, finger_open, finger_closed)

    T = len(ep.images)
    actions = np.zeros((T, 7), dtype=np.float32)  # per-step (not T-1) to match SimplerEnv style
    proprio = np.zeros((T, 8), dtype=np.float32)
    images_224 = np.zeros((T, 224, 224, 3), dtype=np.uint8)
    timestamps_ns = np.zeros((T,), dtype=np.int64)
    is_first = np.zeros((T,), dtype=np.int64)
    is_last  = np.zeros((T,), dtype=np.int64)
    is_term  = np.zeros((T,), dtype=np.int64)
    rewards  = np.zeros((T,), dtype=np.float32)
    discount = np.ones((T,), dtype=np.float32)  # default 1.0

    # poses at each image timestamp
    poses_T = []
    for t_ns in ep.image_times:
        T_be = _lookup_T_at_time(path_edges, dyn_map, dyn_times, static_map, t_ns, slop_ns)
        if T_be is None:
            raise RuntimeError(f"Missing TF for {base}->{eef} near time {t_ns} (slop={slop_ns/1e6:.1f}ms).")
        poses_T.append(T_be)

    for i in range(T):
        # image + ts
        img = resize_to_224(ep.images[i])
        images_224[i] = img
        timestamps_ns[i] = ep.image_times[i]
        is_first[i] = 1 if i == 0 else 0
        is_last[i]  = 1 if i == (T - 1) else 0
        is_term[i]  = is_last[i]
        rewards[i]  = 1.0 if is_last[i] == 1 else 0.0

        # proprio at this step
        p, R = _decompose_T(poses_T[i])
        rpy = _R_to_rpy_xyz(R)  # not stored directly; we keep quaternion in 7D pose for SimplerEnv
        # Convert rotation to quaternion from R
        # Solve quaternion from rotation matrix
        # (stable method)
        qw = np.sqrt(max(0.0, 1 + R[0,0] + R[1,1] + R[2,2])) / 2.0
        qx = np.sign(R[2,1] - R[1,2]) * np.sqrt(max(0.0, 1 + R[0,0] - R[1,1] - R[2,2])) / 2.0
        qy = np.sign(R[0,2] - R[2,0]) * np.sqrt(max(0.0, 1 - R[0,0] + R[1,1] - R[2,2])) / 2.0
        qz = np.sign(R[1,0] - R[0,1]) * np.sqrt(max(0.0, 1 - R[0,0] - R[1,1] + R[2,2])) / 2.0

        g = 0.0
        if len(grip_t) > 0:
            gi = _nearest_before_or_last(grip_t, ep.image_times[i])
            if gi is not None: g = grip_v[gi]

        proprio[i, :] = np.array([p[0], p[1], p[2], qx, qy, qz, qw, g], dtype=np.float32)

        # action: use delta from previous step (i>0), else zeros
        if i > 0:
            p_prev, R_prev = _decompose_T(poses_T[i-1])
            dp = (p - p_prev).astype(np.float32)
            # axis-angle from R_prev^T R
            Rt = R_prev.T @ R
            tr = np.clip((np.trace(Rt) - 1.0) / 2.0, -1.0, 1.0)
            ang = np.arccos(tr)
            if ang < 1e-8:
                aa = np.zeros(3, dtype=np.float32)
            else:
                rx = Rt[2,1] - Rt[1,2]
                ry = Rt[0,2] - Rt[2,0]
                rz = Rt[1,0] - Rt[0,1]
                axis = np.array([rx, ry, rz], dtype=np.float64) / (2.0*np.sin(ang) + 1e-12)
                aa = (axis * ang).astype(np.float32)
            actions[i, :3] = dp
            actions[i, 3:6] = aa
            actions[i, 6] = g  # absolute gripper at current step
        else:
            actions[i, :] = 0.0

    return {
        "images_224": images_224,
        "actions": actions,
        "proprio": proprio,
        "timestamps_ns": timestamps_ns,
        "is_first": is_first,
        "is_last": is_last,
        "is_terminal": is_term,
        "rewards": rewards,
        "discount": discount,
    }

# ------------ TFRecord (TFDS) writer ------------

def _ensure_tf():
    import os, importlib
    os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
    os.environ.setdefault("KERAS_BACKEND", "tensorflow")
    # Reduce TF logs
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    return importlib.import_module("tensorflow")

def _feat_int64(v: int, tf):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[int(v)]))

def _feat_bytes(b: bytes, tf):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[b]))

def _feat_float_list(arr, tf):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[float(x) for x in np.asarray(arr).reshape(-1)]))

def _feat_int64_list(arr, tf):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[int(x) for x in np.asarray(arr).reshape(-1)]))

def write_split_tfds(out_dir: Path, split: str, episodes: List[Dict[str, np.ndarray]], instruction: str,
                     bag_paths: List[str], env_name: str, add_language_embedding: bool = False,
                     tfds_compliant: bool = True, tfds_version: str = "0.0.1") -> Path:
    tf = _ensure_tf()
    embedder = None
    if add_language_embedding:
        try:
            import tensorflow_hub as hub
            embedder = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")
        except Exception as e:
            print(f"[warn] failed to load TF-Hub USE: {e}. Skipping language_embedding.")
            embedder = None

    # Choose output layout. In TFDS-compliant mode, write under versioned dir and use TFDS file naming.
    if tfds_compliant:
        version_dir = out_dir / tfds_version
        version_dir.mkdir(parents=True, exist_ok=True)
        tfrec_path = version_dir / f"rlds-{split}.tfrecord-00000-of-00001"
    else:
        split_dir = out_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        tfrec_path = split_dir / f"{split}-00000-of-00001.tfrecord"
    with tf.io.TFRecordWriter(str(tfrec_path)) as w:
        for ep_idx, ep in enumerate(episodes):
            imgs = ep["images_224"]              # [T,224,224,3] uint8
            T = imgs.shape[0]
            actions = ep["actions"]              # [T,7] float32
            prop = ep["proprio"]                 # [T,8] float32
            ts = ep["timestamps_ns"]             # [T] int64
            is_first = ep["is_first"]
            is_last = ep["is_last"]
            is_term = ep["is_terminal"]
            rewards = ep["rewards"]
            discount = ep["discount"]

            # Per-step language
            li = instruction.encode("utf-8")
            if embedder is not None:
                try:
                    emb = embedder([instruction])[0].numpy().astype(np.float32)
                except Exception as e:
                    print(f"[warn] USE failed on episode {ep_idx}: {e}")
                    emb = None
            else:
                emb = None

            flists = {}

            # steps/observation/image -> PNG bytes
            img_feats = []
            for i in range(T):
                png = encode_png_rgb(imgs[i])
                img_feats.append(tf.train.Feature(bytes_list=tf.train.BytesList(value=[png])))
            flists["steps/observation/image"] = tf.train.FeatureList(feature=img_feats)

            # steps/observation/proprio
            flists["steps/observation/proprio"] = tf.train.FeatureList(
                feature=[_feat_float_list(prop[i], tf) for i in range(T)]
            )

            # steps/action
            flists["steps/action"] = tf.train.FeatureList(
                feature=[_feat_float_list(actions[i], tf) for i in range(T)]
            )

            # steps/discount, steps/reward
            flists["steps/discount"] = tf.train.FeatureList(
                feature=[_feat_float_list([discount[i]], tf) for i in range(T)]
            )
            flists["steps/reward"] = tf.train.FeatureList(
                feature=[_feat_float_list([rewards[i]], tf) for i in range(T)]
            )

            # flags
            flists["steps/is_first"] = tf.train.FeatureList(
                feature=[_feat_int64_list([is_first[i]], tf) for i in range(T)]
            )
            flists["steps/is_last"] = tf.train.FeatureList(
                feature=[_feat_int64_list([is_last[i]], tf) for i in range(T)]
            )
            flists["steps/is_terminal"] = tf.train.FeatureList(
                feature=[_feat_int64_list([is_term[i]], tf) for i in range(T)]
            )

            # language per step
            flists["steps/language_instruction"] = tf.train.FeatureList(
                feature=[_feat_bytes(li, tf) for _ in range(T)]
            )
            if emb is not None:
                flists["steps/language_embedding"] = tf.train.FeatureList(
                    feature=[_feat_float_list(emb, tf) for _ in range(T)]
                )

            # context metadata
            context = tf.train.Features(feature={
                "episode_metadata/file_path": _feat_bytes(bag_paths[ep_idx].encode("utf-8"), tf),
                "episode_metadata/episode_id": _feat_int64(ep_idx + 1, tf),
                "episode_metadata/env_name": _feat_bytes(env_name.encode("utf-8"), tf),
            })

            ex = tf.train.SequenceExample(
                context=context,
                feature_lists=tf.train.FeatureLists(feature_list=flists)
            )
            w.write(ex.SerializeToString())

    # dataset info
    if tfds_compliant:
        # TFDS ReadOnlyBuilder expects a dataset_info.json at the builder dir (version subdir)
        # Provide minimal fields: name, version, splits with shard lengths.
        # We approximate shard lengths by total number of steps across episodes.
        total_steps = int(sum(ep["actions"].shape[0] for ep in episodes))
        info = {
            "name": out_dir.name,
            "version": tfds_version,
            "splits": {
                split: {
                    "numShards": 1,
                    "shardLengths": [total_steps],
                }
            },
        }
        with open((out_dir / tfds_version) / "dataset_info.json", "w") as f:
            json.dump(info, f, indent=2)
    else:
        # Legacy lightweight info under split dir
        split_dir = out_dir / split
        info = {
            "name": out_dir.parent.name,
            "split": split,
            "num_episodes": len(episodes),
            "image_format": "png",
            "image_size": [224,224,3],
            "features": [
                "steps/observation/image", "steps/observation/proprio",
                "steps/action", "steps/discount", "steps/reward",
                "steps/is_first", "steps/is_last", "steps/is_terminal",
                "steps/language_instruction", "steps/language_embedding(optional)"
            ],
            "episode_metadata": ["file_path","episode_id","env_name"],
        }
        with open(split_dir / "dataset_info.json", "w") as f:
            json.dump(info, f, indent=2)
    return tfrec_path

# ------------ CLI ------------

def parse_float_list(arg: str) -> List[float]:
    return [float(x.strip()) for x in arg.split(',')] if arg else []

def parse_str_list(arg: str) -> List[str]:
    return [x.strip() for x in arg.split(',')] if arg else []

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bags", nargs="+", required=True, help="Paths to rosbag2 folder(s) (each is an episode)")
    ap.add_argument("--out_root", required=True, help="Output root folder for dataset")
    ap.add_argument("--dataset_name", required=True, help="Name for dataset (folder)")
    ap.add_argument("--split", default="train", choices=["train", "val"]) 
    ap.add_argument("--instruction", default="pick the standing coke can")
    ap.add_argument("--env_name", default="google_robot_pick_standing_coke_can")
    # Topics
    ap.add_argument("--camera_topic", default=DEFAULT_CAMERA_TOPIC)
    ap.add_argument("--camera_info_topic", default=DEFAULT_CAMERA_INFO_TOPIC)
    ap.add_argument("--joint_states_topic", default=DEFAULT_JOINT_STATES_TOPIC)
    ap.add_argument("--tf_topic", default=DEFAULT_TF_TOPIC)
    ap.add_argument("--tf_static_topic", default=DEFAULT_TF_STATIC_TOPIC)
    # TF / proprio options
    ap.add_argument("--base_frame", default="j2n6s200_link_base")
    ap.add_argument("--eef_frame", default="j2n6s200_end_effector")
    ap.add_argument("--finger_joints", default="j2n6s200_joint_finger_1,j2n6s200_joint_finger_2,j2n6s200_joint_finger_3")
    ap.add_argument("--finger_open", default="0.0,0.0,0.0")
    ap.add_argument("--finger_closed", default="1.0,1.0,1.0")
    ap.add_argument("--tf_slop_ms", type=float, default=200.0, help="Max TF time mismatch allowed per edge (ms).")
    # Optional language embedding
    ap.add_argument("--add_language_embedding", action="store_true", help="Compute USE embedding via TF-Hub")
    # TFDS compatibility options
    ap.add_argument("--tfds_compliant", action="store_true", help="Write TFDS-compatible directory structure (recommended)")
    ap.add_argument("--tfds_version", default="0.0.1", help="TFDS dataset version for ReadOnlyBuilder layout")
    args = ap.parse_args()

    out_ds_dir = Path(args.out_root) / args.dataset_name
    out_split_dir = out_ds_dir / args.split
    out_split_dir.mkdir(parents=True, exist_ok=True)

    finger_joints = parse_str_list(args.finger_joints)
    finger_open = parse_float_list(args.finger_open)
    finger_closed = parse_float_list(args.finger_closed)
    slop_ns = int(args.tf_slop_ms * 1e6)

    episodes_buf = []
    for i, bag in enumerate(args.bags):
        bag_dir = Path(bag)
        print(f"[convert] {i+1}/{len(args.bags)}: {bag_dir}")
        ep = extract_episode(
            bag_dir=bag_dir,
            camera_topic=args.camera_topic,
            camera_info_topic=args.camera_info_topic,
            joint_states_topic=args.joint_states_topic,
            tf_topic=args.tf_topic,
            tf_static_topic=args.tf_static_topic,
            instruction=args.instruction,
            env_name=args.env_name,
        )
        steps = derive_steps_from_tf(
            ep, args.base_frame, args.eef_frame, finger_joints, finger_open, finger_closed, slop_ns
        )
        episodes_buf.append(steps)
        print(f"  -> steps={steps['actions'].shape[0]} proprio=True (tf_slop={args.tf_slop_ms}ms)")

    tfrec_path = write_split_tfds(out_ds_dir, args.split, episodes_buf, instruction=args.instruction,
                                  bag_paths=[str(Path(b)) for b in args.bags], env_name=args.env_name,
                                  add_language_embedding=args.add_language_embedding,
                                  tfds_compliant=args.tfds_compliant, tfds_version=args.tfds_version)
    print(f"[tfds] wrote {len(episodes_buf)} episodes -> {tfrec_path}")
    print(f"[done] split_dir={out_split_dir}")

if __name__ == "__main__":
    main()