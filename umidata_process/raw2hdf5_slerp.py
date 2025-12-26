import os
import cv2
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R, Slerp

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
    return np.array(frames, dtype=np.uint8)

# ------------------ 插值函数 -------------------
def linear_interp(query_ts, ref_ts, ref_values):
    """对标量或向量数据做线性插值"""
    query_ts = np.array(query_ts)
    ref_ts = np.array(ref_ts)
    ref_values = np.array(ref_values)

    # 确保 ref_ts 单调递增
    if not np.all(np.diff(ref_ts) >= 0):
        idx_sort = np.argsort(ref_ts)
        ref_ts = ref_ts[idx_sort]
        ref_values = ref_values[idx_sort]

    if ref_values.ndim == 1:
        return np.interp(query_ts, ref_ts, ref_values)
    else:
        result = np.zeros((len(query_ts), ref_values.shape[1]), dtype=ref_values.dtype)
        for i in range(ref_values.shape[1]):
            result[:, i] = np.interp(query_ts, ref_ts, ref_values[:, i])
        return result

def slerp_interp(query_ts, ref_ts, ref_quats):
    """对四元数做球面插值 (Slerp)"""
    # clip query_ts 到 ref_ts 范围
    query_ts_clipped = np.clip(query_ts, ref_ts[0], ref_ts[-1])

    ref_rot = R.from_quat(ref_quats)
    slerp = Slerp(ref_ts, ref_rot)
    interp_rot = slerp(query_ts_clipped)
    return interp_rot.as_quat()

def align_trajectory(query_ts, traj_ts, traj_vals):
    """轨迹插值：前3列位置线性，后4列四元数Slerp"""
    pos = linear_interp(query_ts, traj_ts, traj_vals[:, :3])
    quat = slerp_interp(query_ts, traj_ts, traj_vals[:, 3:])
    return np.concatenate([pos, quat], axis=1)

def align_clamp(query_ts, clamp_ts, clamp_vals):
    """夹爪线性插值"""
    return linear_interp(query_ts, clamp_ts, clamp_vals).reshape(-1, 1)

# ------------------ 处理每只手 -------------------
def process_one_hand(hand_path):
    """
    hand_path:
      session_003/left_hand_250801xxxx
    返回：
      frames: (T,H,W,3)
      qpos: (T,7)
      clamp: (T,1)
    """
    # ----------- 视频 & timestamp ----------------
    rgb_dir = os.path.join(hand_path, "RGB_Images")
    video_path = os.path.join(rgb_dir, "video.mp4")
    ts_path = os.path.join(rgb_dir, "timestamps.csv")

    frames = load_video(video_path)
    ts_df = pd.read_csv(ts_path)
    video_ts = ts_df["aligned_stamp"].values  # ✅ 使用 aligned_stamp

    # ----------- merged trajectory ---------------
    traj_path = os.path.join(hand_path, "Merged_Trajectory", "merged_trajectory.txt")
    traj_ts, traj_val = load_tum(traj_path)  # traj_val shape = (N,7)

    # ----------- clamp data ----------------------
    clamp_path = os.path.join(hand_path, "Clamp_Data", "clamp_data_tum.txt")
    clamp_ts, clamp_val = load_tum(clamp_path)  # (N,1)

    # ----------- 时间对齐 ------------------------
    aligned_qpos = align_trajectory(video_ts, traj_ts, traj_val)  # (T,7)
    aligned_clamp = align_clamp(video_ts, clamp_ts, clamp_val)    # (T,1)

    return frames, aligned_qpos, aligned_clamp, video_ts

# ------------------ 写HDF5 -------------------
def write_hdf5(out_path, left_data, right_data, max_start_diff_ms=500):
    left_frames, left_qpos, left_clamp, left_ts = left_data
    right_frames, right_qpos, right_clamp, right_ts = right_data

    # ---------- 0. 起始时间戳 sanity check ----------
    start_diff_ms = abs(left_ts[0] - right_ts[0]) * 1000.0
    print(f"[INFO] start timestamp diff = {start_diff_ms:.2f} ms")

    assert start_diff_ms < max_start_diff_ms, (
        f"Start timestamp diff too large: {start_diff_ms:.1f} ms"
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


# ------------------ 处理每个 session -------------------
def process_session(session_path, hdf5_root):
    hands = os.listdir(session_path)
    left_hand = [h for h in hands if h.startswith("left_hand")][0]
    right_hand = [h for h in hands if h.startswith("right_hand")][0]

    left_path = os.path.join(session_path, left_hand)
    right_path = os.path.join(session_path, right_hand)

    print(f"Processing:")
    print("  Left :", left_path)
    print("  Right:", right_path)

    left_data = process_one_hand(left_path)
    right_data = process_one_hand(right_path)

    # ---------- 创建上一级 hdf5 文件夹 ----------
    os.makedirs(hdf5_root, exist_ok=True)

    # 找到已有的 hdf5 文件，按照顺序编号
    existing_files = [f for f in os.listdir(hdf5_root) if f.endswith(".hdf5")]
    existing_indices = [int(os.path.splitext(f)[0]) for f in existing_files if f.split('.')[0].isdigit()]
    next_index = max(existing_indices) + 1 if existing_indices else 0

    out_path = os.path.join(hdf5_root, f"{next_index}.hdf5")
    write_hdf5(out_path, left_data, right_data, max_start_diff_ms=500)

# ------------------ 遍历所有 session -------------------
def convert_all(root):
    hdf5_root = os.path.join(root, "hdf5_slerp")
    os.makedirs(hdf5_root, exist_ok=True)

    sessions = [d for d in os.listdir(root) if d.startswith("session_")]
    sessions.sort()

    for s in sessions:
        session_path = os.path.join(root, s)
        print(f"\n=== Processing {session_path} ===")
        process_session(session_path, hdf5_root)

# ------------------ 主程序 -------------------
if __name__ == "__main__":
    ROOT = "/gemini/user/private/data_process/organize_small"
    convert_all(ROOT)
