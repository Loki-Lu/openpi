#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FastUMI raw data -> LeRobot v2 dataset (NO intermediate HDF5)
Multi-process version
Author: for tong
"""

from __future__ import annotations
import os
from pathlib import Path
import dataclasses
from typing import Optional
import multiprocessing as mp

# =============================================================================
# ============================== CONFIG =======================================
# =============================================================================

@dataclasses.dataclass
class Config:
    # -------- Paths --------
    raw_root: Path = Path("/gemini-1/space/tong/data/cube_sponge")
    output_root: Path = Path("/gemini-1/space/tong/data/lerobot_jnt")

    # -------- Dataset --------
    repo_id: str = "Loki0929/teleai_umi_jnt"
    task: str = "Grasp the cube and sponge into the bin"
    fps: int = 30  # target fps

    # -------- Performance --------
    num_workers: int = 60
    image_writer_processes: int = 6
    image_writer_threads: int = 4

    # -------- Image --------
    image_size: int = 224
    resize_interpolation: int = 3  # cv2.INTER_AREA

    # -------- Debug --------
    verbose: bool = True


CFG = Config()

# =============================================================================
# ============================ ENV & IMPORTS ==================================
# =============================================================================

os.environ["HF_LEROBOT_HOME"] = str(CFG.output_root)

import cv2
import numpy as np
import pandas as pd
import tqdm

from lerobot.common.datasets.lerobot_dataset import (
    HF_LEROBOT_HOME,
    LeRobotDataset,
)

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

    traj_ts, traj = load_tum(hand_path / "Merged_Trajectory/merged_trajectory_jnt.txt")
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
    matches = [p for p in session_path.iterdir()
               if p.is_dir() and p.name.lower().startswith(prefix)]
    if len(matches) == 0:
        raise FileNotFoundError(f"No directory starts with '{prefix}' in {session_path}")
    if len(matches) > 1:
        raise RuntimeError(f"Multiple '{prefix}' dirs found: {matches}")
    return matches[0]

def build_episode(session_path: Path):
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
            out[i] = cv2.resize(
                frames[i],
                (CFG.image_size, CFG.image_size),
                interpolation=CFG.resize_interpolation,
            )
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
    }

# Worker-safe episode builder for multiprocessing
def build_episode_worker(session_path: Path):
    try:
        return build_episode(session_path)
    except Exception as e:
        return {"__error__": True, "session": session_path.name, "msg": str(e)}

# =============================================================================
# =========================== LEROBOT DATASET =================================
# =============================================================================

def create_dataset() -> LeRobotDataset:
    state_names = [
        # 机械臂关节 + 夹爪
        "q1_l_abs","q2_l_abs","q3_l_abs","q4_l_abs","q5_l_abs","q6_l_abs","q7_l_abs","grip_l_abs",
        "q1_r_abs","q2_r_abs","q3_r_abs","q4_r_abs","q5_r_abs","q6_r_abs","q7_r_abs","grip_r_abs",
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
    dataset = create_dataset()

    sessions = sorted(p for p in CFG.raw_root.iterdir() if p.name.startswith("session_"))

    print(f"[INFO] Sessions={len(sessions)} | num_workers={CFG.num_workers}")

    ctx = mp.get_context("spawn")  # safer for cv2/ffmpeg
    with ctx.Pool(processes=CFG.num_workers) as pool:
        for ep in tqdm.tqdm(
            pool.imap_unordered(build_episode_worker, sessions),
            total=len(sessions),
            desc="Processing sessions",
        ):
            if isinstance(ep, dict) and ep.get("__error__", False):
                print(f"[ERROR] {ep['session']} -> {ep['msg']}")
                continue

            # write episode in main process
            T = ep["length"]
            for i in range(T):
                dataset.add_frame({
                    "observation.state": ep["state"][i],
                    "action": ep["action"][i],
                    "task": CFG.task,
                    "observation.images.left_wrist": ep["images"]["left_wrist"][i],
                    "observation.images.right_wrist": ep["images"]["right_wrist"][i],
                })

            dataset.save_episode()
            if CFG.verbose:
                print(f"[OK] {ep['name']}")

    try:
        dataset.consolidate()
    except Exception as e:
        print(f"[Warn] consolidate failed: {e}")
    print(f"\n[DONE] Output → {HF_LEROBOT_HOME / CFG.repo_id}\n")


if __name__ == "__main__":
    main()
