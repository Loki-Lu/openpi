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
    """最近邻时间对齐"""
    idx = np.searchsorted(ref_ts, query_ts, side="left")
    idx = np.clip(idx, 0, len(ref_ts) - 1)
    return ref_values[idx]

# ================== 单手处理 ==================

def process_one_hand(hand_path, target_fps=30):
    """
    处理单手数据并进行降采样
    """
    # -------- 1. 视频与时间戳 --------
    rgb_dir = os.path.join(hand_path, "RGB_Images")
    video_path = os.path.join(rgb_dir, "video.mp4")
    ts_path = os.path.join(rgb_dir, "timestamps.csv")

    frames = load_video(video_path)
    ts_df = pd.read_csv(ts_path)
    video_ts = ts_df["aligned_stamp"].values

    # -------- 2. 计算原 FPS 并采样 --------
    duration = video_ts[-1] - video_ts[0]
    num_frames = len(video_ts)
    original_fps = num_frames / duration if duration > 0 else 0
    
    # 计算步长：如果原 60 目标 30，则 step=2
    step = max(1, int(round(original_fps / target_fps)))
    
    # 打印采样信息 (识别左右手)
    hand_type = "Left" if "left" in hand_path.lower() else "Right"
    new_fps = original_fps / step
    print(f"[{hand_type}] Raw: {original_fps:.2f}fps | Step: {step} | Result: {new_fps:.2f}fps")

    if step > 1:
        frames = frames[::step]
        video_ts = video_ts[::step]

    # -------- 3. 轨迹与夹爪 --------
    traj_path = os.path.join(hand_path, "Merged_Trajectory", "merged_trajectory.txt")
    traj_ts, traj_val = load_tum(traj_path)

    clamp_path = os.path.join(hand_path, "Clamp_Data", "clamp_data_tum.txt")
    clamp_ts, clamp_val = load_tum(clamp_path)

    # -------- 4. 时间对齐 (关键：基于采样后的 video_ts) --------
    aligned_qpos = align_by_timestamp(video_ts, traj_ts, traj_val)
    aligned_clamp = align_by_timestamp(video_ts, clamp_ts, clamp_val)

    return frames, aligned_qpos, aligned_clamp, video_ts

# ================== HDF5 写入 ==================

def write_hdf5(out_path, left_data, right_data, max_start_diff_ms=50):
    left_frames, left_qpos, left_clamp, left_ts = left_data
    right_frames, right_qpos, right_clamp, right_ts = right_data

    # 起始时间戳对齐检查
    start_diff_ms = abs(left_ts[0] - right_ts[0]) * 1000.0
    print(f"[INFO] {os.path.basename(out_path)} Time Diff: {start_diff_ms:.2f} ms")

    if start_diff_ms > max_start_diff_ms:
        print(f"[WARN] {os.path.basename(out_path)} Large sync error!")

    # 1. 拼接 state (pos + clamp)
    left_state  = np.concatenate([left_qpos,  left_clamp],  axis=1)
    right_state = np.concatenate([right_qpos, right_clamp], axis=1)

    # 2. 裁到最短长度
    T = min(len(left_state), len(right_state))
    left_state, right_state = left_state[:T], right_state[:T]
    left_frames, right_frames = left_frames[:T], right_frames[:T]

    # 3. 计算 action (下一帧的 state)
    left_action  = np.vstack([left_state[1:],  left_state[-1:]])
    right_action = np.vstack([right_state[1:], right_state[-1:]])

    # 4. 写入文件
    with h5py.File(out_path, "w") as f:
        for side, state, action, imgs in zip(
            ["left_hand", "right_hand"], 
            [left_state, right_state], 
            [left_action, right_action], 
            [left_frames, right_frames]
        ):
            grp = f.create_group(side)
            obs = grp.create_group("observations")
            img_grp = obs.create_group("images")
            img_grp.create_dataset("wrist", data=imgs, compression="gzip", compression_opts=4)
            grp.create_dataset("observations/state", data=state.astype(np.float32))
            grp.create_dataset("action", data=action.astype(np.float32))

# ================== 并行任务包装 ==================

def process_session_to_hdf5(session_path, out_path, target_fps):
    """子进程入口"""
    hands = os.listdir(session_path)
    try:
        left_hand = [h for h in hands if h.startswith("left_hand")][0]
        right_hand = [h for h in hands if h.startswith("right_hand")][0]
        
        left_data = process_one_hand(os.path.join(session_path, left_hand), target_fps)
        right_data = process_one_hand(os.path.join(session_path, right_hand), target_fps)
        
        write_hdf5(out_path, left_data, right_data)
    except Exception as e:
        print(f"[ERROR] Session {session_path} failed: {e}")

# ================== 主入口 ==================

def convert_all_parallel(root, target_fps=30, max_workers=6):
    """
    暴露 target_fps 参数，实现全流程降采样
    """
    hdf5_root = os.path.join(root, "hdf5")
    os.makedirs(hdf5_root, exist_ok=True)

    sessions = sorted([d for d in os.listdir(root) if d.startswith("session_")])
    tasks = []
    for idx, s in enumerate(sessions):
        tasks.append((os.path.join(root, s), os.path.join(hdf5_root, f"{idx}.hdf5")))

    print(f"Targeting {target_fps} FPS for {len(tasks)} sessions using {max_workers} workers...")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(process_session_to_hdf5, sp, op, target_fps)
            for sp, op in tasks
        ]
        for f in tqdm(as_completed(futures), total=len(futures), desc="Converting Sessions"):
            f.result()

if __name__ == "__main__":
    ROOT = "/gemini-2/user/private/data/data_umi_fruit"
    
    # 在这里直接指定你想要的 FPS
    convert_all_parallel(
        ROOT,
        target_fps=30, 
        max_workers=min(15, os.cpu_count() // 2),
    )