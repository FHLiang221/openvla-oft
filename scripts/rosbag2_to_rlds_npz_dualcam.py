300#!/usr/bin/env python3
"""
rosbag2 -> NPZ (dual camera, MULTI-OUTPUT) without ROS dependencies.

- One NPZ **per bag**, optionally split into multiple NPZs via --chunk_len
- Uses `rosbags` (no rclpy / no rosbag2_py / no tf_transformations)
- Primary camera anchors time; **match first**, then downsample to --target_fps
- Wrist image + joint_state + TF(base->eef) matched per kept frame
- Robust TF (nearest dynamic, fallback to static, or last-known within max age)
- Saves raw RGB (no resize/crop) for OpenVLA-OFT; RLDS-ready keys
- Output keys per NPZ: images, wrist_images, states, timestamps_ns, joints_ns, language, meta

pip install rosbags numpy
"""

import argparse, os, sys, glob, math, json
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import numpy as np

from rosbags.highlevel import AnyReader

# ---------- quaternion / transform helpers ----------
def quat_conj(q):
    x,y,z,w = q; return np.array([-x,-y,-z,w], np.float64)

def quat_mul(q1,q2):
    x1,y1,z1,w1 = q1; x2,y2,z2,w2 = q2
    return np.array([
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
    ], np.float64)

def quat_to_mat3(q):
    x,y,z,w = q
    xx,yy,zz = x*x, y*y, z*z
    xy,xz,yz = x*y, x*z, y*z
    wx,wy,wz = w*x, w*y, w*z
    return np.array([
        [1-2*(yy+zz), 2*(xy-wz),   2*(xz+wy)],
        [2*(xy+wz),   1-2*(xx+zz), 2*(yz-wx)],
        [2*(xz-wy),   2*(yz+wx),   1-2*(xx+yy)],
    ], np.float64)

def axis_angle_from_R(R):
    import math
    tr = (np.trace(R) - 1.0) * 0.5
    tr = max(min(tr, 1.0), -1.0)
    ang = math.acos(tr)
    if ang < 1e-8:
        return np.zeros(3, np.float32)
    rx = R[2,1] - R[1,2]
    ry = R[0,2] - R[2,0]
    rz = R[1,0] - R[0,1]
    axis = np.array([rx, ry, rz], np.float64) / (2.0*math.sin(ang) + 1e-12)
    return (axis * ang).astype(np.float32)

def compose(t1,q1,t2,q2):
    R1 = quat_to_mat3(q1)
    t = t1 + R1 @ t2
    q = quat_mul(q1, q2)
    q = q / max(1e-12, np.linalg.norm(q))
    return t, q

def inv_transform(t,q):
    R = quat_to_mat3(q)
    qinv = quat_conj(q)
    qinv = qinv / max(1e-12, np.linalg.norm(qinv))
    tinv = - R.T @ t
    return tinv, qinv

# ---------- TF store ----------
class TFStore:
    def __init__(self):
        self.static_edges: Dict[Tuple[str,str], Tuple[np.ndarray, np.ndarray]] = {}
        self.dynamic_edges: Dict[Tuple[str,str], List[Tuple[int, Tuple[np.ndarray, np.ndarray]]]] = {}
        self.frames = set()

    def add_static(self, parent, child, tvec, q):
        self.static_edges[(parent,child)] = (np.asarray(tvec,np.float64), self._norm_q(q))
        self.frames.update([parent,child])

    def add_dynamic(self, parent, child, t_ns, tvec, q):
        self.dynamic_edges.setdefault((parent,child), []).append(
            (int(t_ns),(np.asarray(tvec,np.float64), self._norm_q(q)))
        )
        self.frames.update([parent,child])

    def finalize(self):
        for k in self.dynamic_edges:
            self.dynamic_edges[k].sort(key=lambda x: x[0])

    def _norm_q(self, q):
        q = np.asarray(q, np.float64)
        return q / max(1e-12, np.linalg.norm(q))

    def _nearest_dyn(self, key, t_ns, slop_ns, max_age_ns):
        arr = self.dynamic_edges.get(key)
        if not arr: return None
        lo, hi = 0, len(arr)-1
        best_i, best_dt = None, None
        while lo <= hi:
            mid = (lo+hi)//2
            tmid = arr[mid][0]
            dt = abs(tmid - t_ns)
            if best_dt is None or dt < best_dt:
                best_dt, best_i = dt, mid
            if tmid < t_ns: lo = mid+1
            else: hi = mid-1
        if best_i is None: return None
        t0,(tvec,q) = arr[best_i]
        if best_dt <= slop_ns or best_dt <= max_age_ns:
            return (tvec,q, t0, best_dt)
        return None

    def _edge_transform(self, parent, child, t_ns, slop_ns, max_age_ns):
        key = (parent,child)
        nd = self._nearest_dyn(key, t_ns, slop_ns, max_age_ns)
        if nd: return nd[0], nd[1]
        if key in self.static_edges:
            return self.static_edges[key]
        return None

    def _neighbors(self):
        adj = {}
        def add(a,b):
            adj.setdefault(a,set()).add(b)
            adj.setdefault(b,set()).add(a)
        for (p,c) in self.static_edges.keys(): add(p,c)
        for (p,c) in self.dynamic_edges.keys(): add(p,c)
        return adj

    def _path_nodes(self, start, goal):
        adj = self._neighbors()
        if start not in adj or goal not in adj: return None
        from collections import deque
        dq = deque([start]); prev = {start: None}
        while dq:
            u = dq.popleft()
            if u == goal: break
            for v in adj.get(u, ()): 
                if v not in prev:
                    prev[v] = u; dq.append(v)
        if goal not in prev: return None
        seq=[]; v=goal
        while v is not None:
            seq.append(v); v = prev[v]
        return list(reversed(seq))

    def lookup_chain(self, base_frame, eef_frame, t_ns, slop_ns, max_age_ns):
        if base_frame == eef_frame:
            return np.zeros(3,np.float64), np.array([0,0,0,1],np.float64)
        nodes = self._path_nodes(base_frame, eef_frame)
        if not nodes: return None
        T_t = np.zeros(3,np.float64)
        T_q = np.array([0,0,0,1],np.float64)
        for i in range(len(nodes)-1):
            a,b = nodes[i], nodes[i+1]
            edge = self._edge_transform(a,b, t_ns, slop_ns, max_age_ns)
            if edge is None:
                rev = self._edge_transform(b,a, t_ns, slop_ns, max_age_ns)
                if rev is None: return None
                tb,qb = rev
                ta,qa = inv_transform(tb,qb)
            else:
                ta,qa = edge
            T_t, T_q = compose(T_t,T_q, ta,qa)
        return T_t, T_q

# ---------- nearest timestamp index ----------
def nearest_index(times: np.ndarray, t_ns: int, slop_ns: int) -> Optional[int]:
    if times.size == 0: return None
    lo,hi = 0,len(times)-1
    best_i,best_dt=None,None
    while lo <= hi:
        mid = (lo+hi)//2
        tm = times[mid]
        dt = abs(tm - t_ns)
        if best_dt is None or dt < best_dt:
            best_dt, best_i = dt, mid
        if tm < t_ns: lo = mid+1
        else: hi = mid-1
    return best_i if (best_dt is not None and best_dt <= slop_ns) else None

def build_downsampled_indices(t_ns_list, target_fps):
    if not t_ns_list: return []
    period = int(1e9/float(target_fps))
    keep=[]; last=None
    for i,t in enumerate(t_ns_list):
        if last is None or (t-last) >= period:
            keep.append(i); last=t
    return keep

def stack_or_object(imgs):
    shapes = {im.shape for im in imgs if im is not None}
    if len(shapes)==1:
        return np.stack(imgs, axis=0)
    return np.array(imgs, dtype=object)

def img_to_rgb_numpy(msg):
    H, W = int(getattr(msg, 'height', 0)), int(getattr(msg, 'width', 0))
    step = int(getattr(msg, 'step', 0))
    buf = memoryview(msg.data)
    arr = np.frombuffer(buf, dtype=np.uint8)
    if H>0 and W>0 and step>=W*3:
        row = arr.reshape(H, step)
        row = row[:, :W*3]
        rgb = row.reshape(H, W, 3)
        return rgb.copy()
    # fallback: best effort
    try:
        rgb = arr.reshape(H, W, -1)[:, :, :3]
        return rgb.copy()
    except Exception:
        return arr.copy()

def grip_from_map(mp: Dict[str,float], fnames, fo, fc) -> float:
    vals=[]
    for i,n in enumerate(fnames):
        if n in mp:
            den = max(1e-6, fc[i]-fo[i])
            vals.append(np.clip((mp[n]-fo[i])/den, 0.0, 1.0))
    return float(np.mean(vals)) if vals else 0.0

# ---------- per-bag processing ----------
def find_rosbag2_dirs(root: Path) -> List[Path]:
    # any subdir containing metadata.yaml (treat each as an episode)
    dirs = sorted(Path(p).parent for p in glob.glob(str(root / "**" / "metadata.yaml"), recursive=True))
    if root.is_dir() and (root / "metadata.yaml").exists():
        if root not in dirs: dirs.insert(0, root)
    return dirs

def process_one_bag(bag_dir: Path, args, out_dir: Path, index_offset: int) -> int:
    prim_times=[]; prim_imgs=[]
    wrist_times=[]; wrist_imgs=[]
    joints_times=[]; joints_maps=[]
    tvec_hist=[]; quat_hist=[]; grip_hist=[]
    last_wrist = None
    tf_store = TFStore()

    topics_needed = {
        args.primary_topic, args.primary_info, args.wrist_topic, args.wrist_info,
        args.joint_topic, args.tf_topic, args.tf_static_topic,
    }

    with AnyReader([bag_dir]) as reader:
        conns = [c for c in reader.connections if c.topic in topics_needed]
        for conn, t_ns, raw in reader.messages(connections=conns):
            msg = reader.deserialize(raw, conn.msgtype)

            if conn.topic == args.primary_topic:
                # prefer header stamp, fallback to bag time
                if hasattr(msg, 'header') and hasattr(msg.header, 'stamp'):
                    t = int(msg.header.stamp.sec)*1_000_000_000 + int(msg.header.stamp.nanosec)
                else:
                    t = int(t_ns)
                try:
                    rgb = img_to_rgb_numpy(msg)
                except Exception:
                    continue
                prim_times.append(t)
                prim_imgs.append(rgb)

            elif conn.topic == args.wrist_topic:
                if hasattr(msg, 'header') and hasattr(msg.header, 'stamp'):
                    t = int(msg.header.stamp.sec)*1_000_000_000 + int(msg.header.stamp.nanosec)
                else:
                    t = int(t_ns)
                try:
                    rgb = img_to_rgb_numpy(msg)
                except Exception:
                    continue
                wrist_times.append(t)
                wrist_imgs.append(rgb)

            elif conn.topic == args.joint_topic:
                if hasattr(msg, 'header') and hasattr(msg.header, 'stamp'):
                    t = int(msg.header.stamp.sec)*1_000_000_000 + int(msg.header.stamp.nanosec)
                else:
                    t = int(t_ns)
                try:
                    mp = {n: float(p) for n,p in zip(msg.name, msg.position)}
                except Exception:
                    continue
                joints_times.append(t)
                joints_maps.append(mp)

            elif conn.topic in (args.tf_topic, args.tf_static_topic):
                is_static = (conn.topic == args.tf_static_topic)
                try:
                    for ts in msg.transforms:
                        parent = ts.header.frame_id
                        child  = ts.child_frame_id
                        tvec = np.array([ts.transform.translation.x,
                                         ts.transform.translation.y,
                                         ts.transform.translation.z], np.float64)
                        q    = np.array([ts.transform.rotation.x,
                                         ts.transform.rotation.y,
                                         ts.transform.rotation.z,
                                         ts.transform.rotation.w], np.float64)
                        if is_static:
                            tf_store.add_static(parent, child, tvec, q)
                        else:
                            tmsg = int(ts.header.stamp.sec)*1_000_000_000 + int(ts.header.stamp.nanosec)
                            tf_store.add_dynamic(parent, child, tmsg, tvec, q)
                except Exception:
                    continue

    tf_store.finalize()

    def sort_by_time(times, vals):
        if not times: return np.array([],dtype=np.int64), []
        order = np.argsort(np.asarray(times, dtype=np.int64))
        return np.asarray(times, dtype=np.int64)[order], [vals[i] for i in order]

    prim_times, prim_imgs   = sort_by_time(prim_times, prim_imgs)
    wrist_times, wrist_imgs = sort_by_time(wrist_times, wrist_imgs)
    joints_times, joints_maps = sort_by_time(joints_times, joints_maps)

    print(f"[{bag_dir.name}] primary frames: {len(prim_times)}; matching at full rate before downsampling")

    slop_ns       = int(args.slop_ms * 1e6)
    wrist_slop_ns = int(args.wrist_slop_ms * 1e6)
    tf_slop_ns    = int(args.tf_slop_ms * 1e6)
    tf_max_age_ns = int(args.tf_max_age_ms * 1e6)

    fo = np.array(args.finger_open,   np.float64)
    fc = np.array(args.finger_closed, np.float64)
    fnames = list(args.finger_joints)

    images=[]; wrist_images=[]; states=[]
    kept_t=[]; matched_js_t=[]
    dropped = dict(no_wrist=0, no_js=0, no_tf=0)

    for ki in range(len(prim_times)):
        t_img = int(prim_times[ki]); rgb = prim_imgs[ki]

        wi = nearest_index(wrist_times, t_img, wrist_slop_ns) if len(wrist_times)>0 else None
        if args.require_both_cams and wi is None:
            dropped["no_wrist"] += 1; continue
        rgb_w = wrist_imgs[wi] if wi is not None else None

        ji = nearest_index(joints_times, t_img, slop_ns) if len(joints_times)>0 else None
        if ji is None:
            dropped["no_js"] += 1; continue
        mp = joints_maps[ji]; t_js = int(joints_times[ji])

        tf_res = tf_store.lookup_chain(args.base_frame, args.eef_frame, t_img, tf_slop_ns, tf_max_age_ns)
        if tf_res is None:
            dropped["no_tf"] += 1; continue
        tvec, q = tf_res

        # state vector: [px,py,pz, qw,qx,qy,qz, grip]
        g = grip_from_map(mp, fnames, fo, fc)
        state = np.array([
            tvec[0], tvec[1], tvec[2], q[3], q[0], q[1], q[2], g
        ], np.float32)

        # forward-fill wrist if allowed
        if rgb_w is None:
            if args.require_both_cams:
                dropped["no_wrist"] += 1; continue
            rgb_w = last_wrist
            if rgb_w is None:
                dropped["no_wrist"] += 1; continue
        else:
            last_wrist = rgb_w
        images.append(rgb)
        wrist_images.append(rgb_w)
        states.append(state)
        tvec_hist.append(tvec.copy()); quat_hist.append(q.copy()); grip_hist.append(float(g))
        kept_t.append(t_img)
        matched_js_t.append(t_js)

    # --- Downsample AFTER matching ---
    if images:
        ds_idx = build_downsampled_indices(kept_t, args.target_fps)
        if len(ds_idx) < len(images):
            images       = [images[i] for i in ds_idx]
            wrist_images = [wrist_images[i] for i in ds_idx]
            states       = [states[i] for i in ds_idx]
            tvec_hist    = [tvec_hist[i] for i in ds_idx]
            quat_hist    = [quat_hist[i] for i in ds_idx]
            grip_hist    = [grip_hist[i] for i in ds_idx]
            kept_t       = [kept_t[i] for i in ds_idx]
            matched_js_t = [matched_js_t[i] for i in ds_idx]
        print(f"[{bag_dir.name}] matched frames: {len(kept_t)}; kept {len(images)} after downsampling to {args.target_fps}Hz")

    if not images:
        print(f"[{bag_dir.name}] [WARN] No samples kept. Drops: {dropped}")
        return index_offset  # nothing written; index unchanged

    actions = np.zeros((max(0, len(tvec_hist)-1), 7), np.float32)
    if len(tvec_hist) >= 2:
        for i in range(len(tvec_hist)-1):
            dp = (np.asarray(tvec_hist[i+1]) - np.asarray(tvec_hist[i])).astype(np.float32)
            R0 = quat_to_mat3(np.asarray(quat_hist[i]))
            R1 = quat_to_mat3(np.asarray(quat_hist[i+1]))
            aa = axis_angle_from_R(R0.T @ R1)
            actions[i, :3] = dp
            actions[i, 3:6] = aa
            actions[i, 6]   = float(grip_hist[i+1])
    
    # slice into multiple NPZs if requested
    total = len(images)
    chunk = int(args.chunk_len)
    nout = math.ceil(total / float(chunk)) if chunk > 0 else 1

    for part in range(nout):
        s = part*chunk
        e = min(total, (part+1)*chunk) if chunk > 0 else total
        if e - s <= 0: continue

        outp = out_dir / f"{index_offset+part:06d}.npz"

        images_arr = stack_or_object(images[s:e])
        wrist_arr  = stack_or_object(wrist_images[s:e])
        states_arr = np.stack(states[s:e], axis=0).astype(np.float32)
        t_arr      = np.asarray(kept_t[s:e], dtype=np.int64)
        tjs_arr    = np.asarray(matched_js_t[s:e], dtype=np.int64)
        acts_arr   = actions[s:max(s, e-1), ...] if actions.size else np.zeros((0,7), np.float32)

        meta = {
            "bag": str(bag_dir),
            "index_start": int(index_offset),
            "index_part": int(part),
            "index_total_parts": int(nout),
            "base_frame": args.base_frame,
            "eef_frame": args.eef_frame,
            "target_fps": args.target_fps,
            "slop_ms": args.slop_ms,
            "wrist_slop_ms": args.wrist_slop_ms,
            "tf_slop_ms": args.tf_slop_ms,
            "tf_max_age_ms": args.tf_max_age_ms,
            "require_both_cams": bool(args.require_both_cams),
            "dropped": dropped,
            "slice": {"start": int(s), "end": int(e), "total_kept": int(total)},
        }

        np.savez_compressed(
            outp,
            images=images_arr,
            actions=acts_arr,                     # or 'actions=actions' if you removed chunking
            success=np.array(bool(args.success), dtype=bool),
            proprio=states_arr,                   # handy alias some tools expect
            wrist_images=wrist_arr,
            states=states_arr,
            timestamps_ns=t_arr,
            joints_ns=tjs_arr,
            language=str(args.language),
            meta=json.dumps(meta),
        )
        print(f"[{bag_dir.name}] → wrote {outp.name}  frames={e-s}  ({part+1}/{nout})")

    return index_offset + max(1, nout)

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--bags_root", required=True, help="Folder containing one or more rosbag2_* dirs (metadata.yaml)")
    ap.add_argument("--out_dir", required=True)

    # topics
    ap.add_argument("--primary_topic", type=str, default="/camera/camera/color/image_raw")
    ap.add_argument("--primary_info",  type=str, default="/camera/camera/color/camera_info")
    ap.add_argument("--wrist_topic",   type=str, default="/camera/wrist/color/image_raw")
    ap.add_argument("--wrist_info",    type=str, default="/camera/wrist/color/camera_info")
    ap.add_argument("--joint_topic",   type=str, default="/joint_states")
    ap.add_argument("--tf_topic",      type=str, default="/tf")
    ap.add_argument("--tf_static_topic", type=str, default="/tf_static")

    # frames
    ap.add_argument("--base_frame", type=str, default="j2n6s200_link_base")
    ap.add_argument("--eef_frame",  type=str, default="j2n6s200_end_effector")

    # timing / matching
    ap.add_argument("--target_fps", type=float, default=5.0, help="Downsample rate applied AFTER matching")
    ap.add_argument("--slop_ms", type=float, default=60.0, help="Tolerance for joints match")
    ap.add_argument("--wrist_slop_ms", type=float, default=60.0, help="Tolerance for wrist match")
    ap.add_argument("--tf_slop_ms", type=float, default=60.0, help="Tolerance for TF dynamic match")
    ap.add_argument("--tf_max_age_ms", type=float, default=150.0, help="Allow last-known TF up to this age")

    ap.add_argument("--require_both_cams", action="store_true", help="Drop frames without wrist image")

    # gripper mapping
    ap.add_argument("--finger_joints", nargs="+", default=["panda_finger_joint1","panda_finger_joint2"])
    ap.add_argument("--finger_open",   nargs="+", type=float, default=[0.04,0.04])
    ap.add_argument("--finger_closed", nargs="+", type=float, default=[0.0,0.0])

    # output / slicing
    ap.add_argument("--language", type=str, default="")
    ap.add_argument("--chunk_len", type=int, default=10000, help="Max frames per NPZ chunk (0 = no chunking)")
    ap.add_argument("--success", action="store_true", help="Mark episode success=True in metadata")

    args = ap.parse_args()
    bags_root = Path(args.bags_root)
    bag_paths  = find_rosbag2_dirs(bags_root)
    if not bag_paths:
        print(f"[ERR] No rosbag2 folders (metadata.yaml) found in {bags_root}", file=sys.stderr)
        sys.exit(1)


    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    idx = 0
    for bag in bag_paths:
        try:
            idx = process_one_bag(bag, args, out_dir, idx)
        except Exception as e:
            print(f"[ERR] Failed to process {bag}: {str(e)}")
            continue

if __name__ == "__main__":
    main()
