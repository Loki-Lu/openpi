#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FastUMI raw data -> LeRobot v2 dataset (手动指定任务文件夹 + 英文描述 + 多 session 支持)
Author: for tong
"""

from __future__ import annotations
import os
from pathlib import Path
import dataclasses
from typing import Dict, List, Tuple
import multiprocessing as mp
import time

# =============================================================================
# ============================== CONFIG =======================================
# =============================================================================

@dataclasses.dataclass
class Config:
    raw_root: Path = Path("/gemini/space/users/tong/data/lumin_merged224")  # 原始 task 根目录
    output_root: Path = Path("/gemini/space/users/tong/data/lerobot_multi_tasks_224")

    repo_id: str = "Loki0929/teleai_umi_multi_tasks"

    fps: int = 30
    image_size: int = 224
    resize_interpolation: int = 3  # cv2.INTER_AREA

    num_workers: int = 10
    image_writer_processes: int = 10
    image_writer_threads: int = 10

    verbose: bool = True


CFG = Config()
os.environ["HF_LEROBOT_HOME"] = str(CFG.output_root)

# =============================================================================
# ============================ IMPORTS ========================================
# =============================================================================

import cv2
import numpy as np
import pandas as pd
import tqdm

from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset

# =============================================================================
# =============================== IO HELPERS ==================================
# =============================================================================

def load_tum(path: Path):
    data = np.loadtxt(path)
    if data.ndim == 1:
        data = data[None, :]
    return data[:, 0], data[:, 1:]

def load_video(video_path: Path):
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame[..., ::-1])  # BGR -> RGB
    cap.release()
    return np.asarray(frames, dtype=np.uint8)

def align_by_timestamp(query_ts, ref_ts, ref_val):
    idx = np.searchsorted(ref_ts, query_ts, side="left")
    idx = np.clip(idx, 0, len(ref_ts) - 1)
    return ref_val[idx]

# =============================================================================
# ============================ RAW PROCESSING =================================
# =============================================================================

def process_one_hand(hand_path: Path, fps: int):
    # print(f"[INFO] Processing hand data in {hand_path}")
    rgb_dir = hand_path / "RGB_Images"
    frames = load_video(rgb_dir / "video.mp4")
    ts = pd.read_csv(rgb_dir / "timestamps.csv")["header_stamp"].values

    duration = ts[-1] - ts[0]
    raw_fps = len(ts) / duration
    step = max(1, int(round(raw_fps / fps)))

    if CFG.verbose:
        side = "Left" if "left" in hand_path.name.lower() else "Right"
        print(f"[{side}] {raw_fps:.2f} fps → step={step}")

    frames = frames[::step]
    ts = ts[::step]

    traj_ts, traj = load_tum(hand_path / "Merged_Trajectory/merged_trajectory.txt")
    clamp_ts, clamp = load_tum(hand_path / "Clamp_Data/clamp_data_tum.txt")

    traj = traj[:, :7]
    clamp = clamp[:, :1]

    qpos = align_by_timestamp(ts, traj_ts, traj)
    grip = align_by_timestamp(ts, clamp_ts, clamp)

    return frames, np.concatenate([qpos, grip], axis=1)

# =============================================================================
# ============================ EPISODE BUILDER =================================
# =============================================================================

def find_hand_dir(session_path: Path, prefix: str) -> Path:
    matches = [p for p in session_path.iterdir() if p.is_dir() and p.name.lower().startswith(prefix)]
    if len(matches) == 0:
        raise FileNotFoundError(f"No directory starts with '{prefix}' in {session_path}")
    if len(matches) > 1:
        raise RuntimeError(f"Multiple '{prefix}' dirs found: {matches}")
    return matches[0]

def build_episode(session_path: Path, task_desc: str):
    """
    支持多 session，处理单个 session 下的左右手
    """
    left_dir = find_hand_dir(session_path, "left_hand")
    right_dir = find_hand_dir(session_path, "right_hand")

    lf, ls = process_one_hand(left_dir, CFG.fps)
    rf, rs = process_one_hand(right_dir, CFG.fps)

    T = min(len(ls), len(rs))
    state = np.concatenate([ls[:T], rs[:T]], axis=1).astype(np.float32)
    action = np.vstack([state[1:], state[-1:]])

    def resize(frames):
        out = np.empty((T, CFG.image_size, CFG.image_size, 3), np.uint8)
        for i in range(T):
            out[i] = cv2.resize(frames[i], (CFG.image_size, CFG.image_size),
                                interpolation=CFG.resize_interpolation)
        return out

    return {
        "state": state,
        "action": action,
        "images": {
            "left_wrist": resize(lf[:T]),
            "right_wrist": resize(rf[:T]),
        },
        "length": T,
        "name": session_path.name,
        "task": task_desc,
    }

def build_episode_worker(args: Tuple[Path, str]):
    # print("[WORKER START]", args[0].name, flush=True)
    session_path, task_desc = args
    try:
        return build_episode(session_path, task_desc)
    except Exception as e:
        return {"__error__": True, "session": session_path.name, "msg": str(e)}

# =============================================================================
# ============================ DATASET CREATION =================================
# =============================================================================

def create_dataset() -> LeRobotDataset:
    state_names = [
        "x_l_abs","y_l_abs","z_l_abs","qx_l_abs","qy_l_abs","qz_l_abs","qw_l_abs","gripper_l_abs",
        "x_r_abs","y_r_abs","z_r_abs","qx_r_abs","qy_r_abs","qz_r_abs","qw_r_abs","gripper_r_abs",
    ]

    features = {
        "observation.state": {"dtype": "float32", "shape": (16,), "names": state_names},
        "action": {"dtype": "float32", "shape": (16,), "names": state_names},
        "observation.images.left_wrist": {"dtype": "video", "shape": (CFG.image_size, CFG.image_size, 3)},
        "observation.images.right_wrist": {"dtype": "video", "shape": (CFG.image_size, CFG.image_size, 3)},
    }

    path = HF_LEROBOT_HOME / CFG.repo_id
    if path.exists():
        import shutil
        shutil.rmtree(path)

    return LeRobotDataset.create(
        repo_id=CFG.repo_id,
        fps=CFG.fps,
        robot_type="fastumi",
        features=features,
        use_videos=True,
        image_writer_processes=CFG.image_writer_processes,
        image_writer_threads=CFG.image_writer_threads,
    )

# =============================================================================
# ================================ MAIN =======================================
# =============================================================================

def main():
    start_time = time.perf_counter()
    # 手动任务 + 英文描述
    tasks: Dict[str, str] = {
        "task1": "Make a sandwich",
        "task2": "Right hand grabs an eraser, left hand grabs a marker",
        "task3": "Left and right hands simultaneously pick up crumpled paper from the table",
        "task4": "Right hand picks up a can twice, left hand picks up a crumpled paper twice",
        "task5": "Right hand holds a cup, left hand puts a glue stick into the cup",
        "task6": "Make a sandwich",
        "task7": "Right hand grabs an eraser, left hand grabs a marker",
        "task8": "Right hand grabs an eraser, left hand grabs a marker",
        "task9": "Right hand picks up a can twice, left hand picks up a crumpled paper twice",
    }

    ctx = mp.get_context("spawn")

    # ✅ 只创建一次 dataset
    dataset = create_dataset()

    for task_name, desc in tasks.items():
        print(f"\n==============================")
        print(f"[TASK] {task_name}: {desc}")
        print(f"==============================")

        task_path = CFG.raw_root / task_name
        if not task_path.exists():
            print(f"[WARN] Task folder {task_path} not found, skipped.")
            continue

        # 收集当前 task 的 session
        task_sessions: List[Tuple[Path, str]] = []
        session_dirs = [
            p for p in task_path.iterdir()
            if p.is_dir() and p.name.lower().startswith("session_")
        ]

        for session in sorted(session_dirs):
            task_sessions.append((session, desc))

        if len(task_sessions) == 0:
            print(f"[WARN] No sessions found in {task_name}, skipped.")
            continue

        print(f"[INFO] {task_name}: {len(task_sessions)} sessions")

        # ✅ 每个 task 独立一个 Pool（但写同一个 dataset）
        with ctx.Pool(processes=CFG.num_workers) as pool:
            for ep in tqdm.tqdm(
                pool.imap_unordered(build_episode_worker, task_sessions),
                total=len(task_sessions),
                desc=f"Processing {task_name}",
            ):
                if isinstance(ep, dict) and ep.get("__error__", False):
                    print(f"[ERROR] {ep['session']} -> {ep['msg']}")
                    continue

                T = ep["length"]
                for i in range(T):
                    dataset.add_frame({
                        "observation.state": ep["state"][i],
                        "action": ep["action"][i],
                        "task": ep["task"],  # ✅ task 作为字段区分
                        "observation.images.left_wrist": ep["images"]["left_wrist"][i],
                        "observation.images.right_wrist": ep["images"]["right_wrist"][i],
                    })

                dataset.save_episode()
                if CFG.verbose:
                    print(f"[OK] {task_name} | {ep['name']}")

    # ✅ 所有 task 完成后再 consolidate
    try:
        dataset.consolidate()
    except Exception as e:
        print(f"[WARN] consolidate failed: {e}")

    print(f"\n[DONE] Output → {HF_LEROBOT_HOME / CFG.repo_id}\n")
    end_time = time.perf_counter()
    total_sec = end_time - start_time

    h = int(total_sec // 3600)
    m = int((total_sec % 3600) // 60)
    s = total_sec % 60

    print(
        f"\n[TIME] Total conversion time: "
        f"{h:d}h {m:d}m {s:.2f}s ({total_sec:.2f} seconds)\n"
    )



if __name__ == "__main__":
    main()
