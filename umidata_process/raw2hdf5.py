"""
HDF5 文件结构示例：
$ h5ls example.hdf5
Group: left_hand
Dataset: left_hand/action, shape=(875, 8), dtype=float64
Group: left_hand/observations
Group: left_hand/observations/images
Dataset: left_hand/observations/images/wrist, shape=(875, 1080, 1920, 3), dtype=uint8
Dataset: left_hand/observations/state, shape=(875, 8), dtype=float64
Group: right_hand
Dataset: right_hand/action, shape=(875, 8), dtype=float64
Group: right_hand/observations
Group: right_hand/observations/images
Dataset: right_hand/observations/images/wrist, shape=(875, 1080, 1920, 3), dtype=uint8
Dataset: right_hand/observations/state, shape=(875, 8), dtype=float64"""

import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
import cv2
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed



# ================== 基础 IO ==================

def load_tum(path):
    """读取 TUM 格式文件 return timestamps(N,), values(N,D)"""
    data = np.loadtxt(path)
    if data.ndim == 1:
        data = data[None, :]
    ts = data[:, 0]
    values = data[:, 1:]
    return ts, values


def load_video(video_path):
    """读取整个视频为 RGB numpy 数组"""
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame[..., ::-1])  # BGR → RGB
    cap.release()
    return np.asarray(frames, dtype=np.uint8)


def align_by_timestamp(query_ts, ref_ts, ref_values):
    """
    最近邻时间对齐
    query_ts: (N,)
    ref_ts: (M,)
    ref_values: (M,D)
    """
    idx = np.searchsorted(ref_ts, query_ts, side="left")
    idx = np.clip(idx, 0, len(ref_ts) - 1)
    return ref_values[idx]


# ================== 单手处理 ==================

def process_one_hand(hand_path):
    """
    hand_path:
      session_xxx/left_hand_xxx 或 right_hand_xxx

    return:
      frames: (T,H,W,3)
      qpos: (T,7)
      clamp: (T,1)
    """

    # -------- 视频 --------
    rgb_dir = os.path.join(hand_path, "RGB_Images")
    video_path = os.path.join(rgb_dir, "video.mp4")
    ts_path = os.path.join(rgb_dir, "timestamps.csv")

    frames = load_video(video_path)
    ts_df = pd.read_csv(ts_path)
    video_ts = ts_df["aligned_stamp"].values

    # -------- 轨迹 --------
    traj_path = os.path.join(
        hand_path, "Merged_Trajectory", "merged_trajectory.txt"
    )
    traj_ts, traj_val = load_tum(traj_path)  # (N,7)

    # -------- 夹爪 --------
    clamp_path = os.path.join(
        hand_path, "Clamp_Data", "clamp_data_tum.txt"
    )
    clamp_ts, clamp_val = load_tum(clamp_path)  # (N,1)

    # -------- 时间对齐 --------
    aligned_qpos = align_by_timestamp(video_ts, traj_ts, traj_val)
    aligned_clamp = align_by_timestamp(video_ts, clamp_ts, clamp_val)

    return frames, aligned_qpos, aligned_clamp, video_ts


# ================== HDF5 写入 ==================

def write_hdf5(out_path, left_data, right_data, max_start_diff_ms=50):
    left_frames, left_qpos, left_clamp, left_ts = left_data
    right_frames, right_qpos, right_clamp, right_ts = right_data

    # ---------- 0. 起始时间戳 sanity check ----------
    start_diff_ms = abs(left_ts[0] - right_ts[0]) * 1000.0
    # 打印带 session / 文件信息
    print(f"[INFO] [{os.path.basename(out_path)}] start timestamp diff = {start_diff_ms:.2f} ms", flush=True)

    # 断言也带上 session / 文件信息
    assert start_diff_ms < max_start_diff_ms, (
        f"[{os.path.basename(out_path)}] Start timestamp diff too large: {start_diff_ms:.1f} ms"
    )

    # ---------- 1. 拼 state ----------
    left_state  = np.concatenate([left_qpos,  left_clamp],  axis=1)
    right_state = np.concatenate([right_qpos, right_clamp], axis=1)

    # ---------- 2. 裁到最短 ----------
    T = min(len(left_state), len(right_state))

    left_state   = left_state[:T]
    right_state  = right_state[:T]
    left_frames  = left_frames[:T]
    right_frames = right_frames[:T]

    # ---------- 3. action（在裁剪后算） ----------
    left_action  = np.vstack([left_state[1:],  left_state[-1:]])
    right_action = np.vstack([right_state[1:], right_state[-1:]])


    # ---------- 4. 写 HDF5 ----------
    with h5py.File(out_path, "w") as f:

        grp0 = f.create_group("left_hand")
        obs0 = grp0.create_group("observations")
        img0 = obs0.create_group("images")
        img0.create_dataset(
            "wrist", data=left_frames,
            compression="gzip", compression_opts=4
        )
        grp0.create_dataset("observations/state", data=left_state)
        grp0.create_dataset("action", data=left_action)

        grp1 = f.create_group("right_hand")
        obs1 = grp1.create_group("observations")
        img1 = obs1.create_group("images")
        img1.create_dataset(
            "wrist", data=right_frames,
            compression="gzip", compression_opts=4
        )
        grp1.create_dataset("observations/state", data=right_state)
        grp1.create_dataset("action", data=right_action)

    print(f"[OK] saved HDF5 → {out_path}")



# ================== 单 session（子进程） ==================

def process_session_to_hdf5(session_path, out_path):
    hands = os.listdir(session_path)
    left_hand = [h for h in hands if h.startswith("left_hand")][0]
    right_hand = [h for h in hands if h.startswith("right_hand")][0]

    left_path = os.path.join(session_path, left_hand)
    right_path = os.path.join(session_path, right_hand)

    print(f"[PID {os.getpid()}] {session_path}")

    left_data = process_one_hand(left_path)
    right_data = process_one_hand(right_path)

    write_hdf5(out_path, left_data, right_data, max_start_diff_ms=50)


# ================== 并行入口 ==================

def convert_all_parallel(root, max_workers=6):
    """
    root/
      session_001/
      session_002/
    """
    hdf5_root = os.path.join(root, "hdf5")
    os.makedirs(hdf5_root, exist_ok=True)

    sessions = sorted(
        [d for d in os.listdir(root) if d.startswith("session_")]
    )

    tasks = []
    for idx, s in enumerate(sessions):
        session_path = os.path.join(root, s)
        out_path = os.path.join(hdf5_root, f"{idx}.hdf5")
        tasks.append((session_path, out_path))

    print(f"Total sessions: {len(tasks)}")
    print(f"Using workers : {max_workers}")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(process_session_to_hdf5, sp, op)
            for sp, op in tasks
        ]

        for f in tqdm(as_completed(futures), total=len(futures)):
            f.result()  # 抛异常


# ================== main ==================

if __name__ == "__main__":
    #HDF5_USE_FILE_LOCKING=FALSE uv run 2_raw2hdf5_neareast_multcores.py 加锁

    ROOT = "/gemini-2/user/private/data/data_umi_fruit"
    convert_all_parallel(
        ROOT,
        max_workers=min(32, os.cpu_count() // 2),
    )
