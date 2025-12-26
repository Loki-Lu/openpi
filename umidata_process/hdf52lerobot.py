#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to convert FastUMI HDF5 data to the LeRobot dataset v2.0 format, with multiprocessing.

Key improvements vs your draft
-----------------------------
- **Deterministic episode order**: natural sort across nested folders.
- **Parallel preprocess, serial write**: executor.map keeps output order == input order.
- **Writer settings synced to mode**: `use_videos = (mode == "video")`.
- **Consolidate at end**: `dataset.consolidate()` to finalize shards.
- **Safer defaults**: smaller writer parallelism by default; can be overridden.

Examples
--------
uv run convert_fastumi_data_to_lerobot_ordered.py \
  --raw-dir /root/dataset/unzip/sweep_trash/merged_sweep_trash_deltatcp \
  --repo-id myteam/sweep_trash \
  --task  "grab the small broom and sweep the trash into the dustpan" \
  --fps 20 \
  --output-dir /root/dataset/unzip/sweep_trash/merged_sweep_trash_deltatcp_rgb224 \
  --mode image \
  --workers 8
"""
from __future__ import annotations

import dataclasses
import os
import re
import shutil
from pathlib import Path
from typing import Literal, List, Dict, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor

import h5py
import numpy as np
import torch
import tqdm
# import tyro
import cv2  # 本地导入避免额外依赖

from lerobot.common.datasets.lerobot_dataset import (
    HF_LEROBOT_HOME,
    LeRobotDataset,
)
# from lerobot.common.datasets.push_dataset_to_hub._download_raw import (
#     download_raw,
# )

# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class DatasetConfig:
    use_videos: bool = False            # safer default for images
    tolerance_s: float = 1e-4
    image_writer_processes: int = 2     # avoid over-parallelism
    image_writer_threads: int = 4
    video_backend: str | None = None


DEFAULT_DATASET_CONFIG = DatasetConfig()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def natsort_key(p: Path):
    # natural sort by each path component, then basename tokens
    def tok(s: str):
        return [int(t) if t.isdigit() else t for t in re.findall(r"\d+|\D+", s)]
    parts = list(p.parts)
    key: List[object] = []
    for part in parts:
        key.extend(tok(part))
    return key

# ---------------------------------------------------------------------------
# Dataset creation utilities
# ---------------------------------------------------------------------------

def create_empty_dataset(
    repo_id: str,
    fps: int,
    robot_type: str = "fastumi",
    mode: Literal["video", "image"] = "video",
    *,
    has_velocity: bool = False,
    has_effort: bool = False,
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
) -> LeRobotDataset:
    """Initialise an empty LeRobot dataset ready to receive frames."""

    # FastUMI state definition: (x, y, z, qx, qy, qz, qw, gripper_width)
    # 双臂 UMI 状态定义（16 DoF）
    state_names = [
        # left arm
        "x_l", "y_l", "z_l", "qx_l", "qy_l", "qz_l", "qw_l",
        # right arm
        "x_r", "y_r", "z_r", "qx_r", "qy_r", "qz_r", "qw_r",
        # grippers
        "gripper_l", "gripper_r",
    ]

    cameras = ["front"]  # only a single RGB camera

    features: Dict[str, Dict[str, object]] = {
        "observation.state": {
            "dtype": "float32",
            "shape": (len(state_names),),
            "names": state_names,
        },
        "action": {
            "dtype": "float32",
            "shape": (len(state_names),),
            "names": state_names,
        },
    }

    if has_velocity:
        features["observation.velocity"] = {
            "dtype": "float32",
            "shape": (len(state_names),),
            "names": state_names,
        }

    if has_effort:
        features["observation.effort"] = {
            "dtype": "float32",
            "shape": (len(state_names),),
            "names": state_names,
        }

    for cam in cameras:
        features[f"observation.images.{cam}"] = {
            "dtype": mode,
            "shape": (224, 224, 3),  # (H, W, C) 224p RGB
            "names": ["height", "width", "channels"],
        }

    # Clear previous dataset folder if it already exists
    if (HF_LEROBOT_HOME / repo_id).exists():
        shutil.rmtree(HF_LEROBOT_HOME / repo_id)

    # Keep writer mode in sync with image/video choice
    use_videos = (mode == "video") if dataset_config.use_videos is None else dataset_config.use_videos

    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        robot_type=robot_type,
        features=features,
        use_videos=use_videos,
        tolerance_s=dataset_config.tolerance_s,
        image_writer_processes=dataset_config.image_writer_processes,
        image_writer_threads=dataset_config.image_writer_threads,
        video_backend=dataset_config.video_backend,
    )

# ---------------------------------------------------------------------------
# IO helpers (episode loading / preprocessing)
# ---------------------------------------------------------------------------

def _load_images(ep: h5py.File, cameras: List[str]) -> Dict[str, np.ndarray]:
    """加载并处理视频帧：若为原始像素直接取；若为JPEG字节则逐帧解码；然后统一 resize 到 (224,224)。"""
    imgs: Dict[str, np.ndarray] = {}
    for cam in cameras:
        ds = ep[f"/observations/images/{cam}"]
        if ds.ndim == 4:
            # 已是像素，注意 FastUMI 通常存 BGR；统一转 RGB
            arr = ds[:]                  # (T, H, W, 3)
            arr = arr[..., ::-1]         # BGR -> RGB
        else:
            # 存的是 JPEG bytes，需要逐帧解码
            decoded: List[np.ndarray] = []
            for buf in ds:
                bgr = cv2.imdecode(buf, 1)
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                decoded.append(rgb)
            arr = np.stack(decoded, axis=0)  # (T, H, W, 3)

        # resize 到 224×224
        T = arr.shape[0]
        resized = np.empty((T, 224, 224, 3), dtype=arr.dtype)
        for i in range(T):
            resized[i] = cv2.resize(arr[i], (224, 224), interpolation=cv2.INTER_AREA)
        imgs[cam] = resized
    return imgs


def _load_episode(ep_path: Path) -> Tuple[
    Dict[str, np.ndarray],  # imgs
    torch.Tensor,           # state (T,8)
    torch.Tensor,           # action (T,8)
    Optional[torch.Tensor], # velocity
    Optional[torch.Tensor], # effort
]:
    cameras = ["wrist"]
    with h5py.File(ep_path, "r") as ep:
        state = torch.from_numpy(ep["/observations/qpos"][:].astype(np.float32))  # (T, 8)
        action = torch.from_numpy(ep["/action"][:].astype(np.float32))            # (T, 8)
        velocity = None  # FastUMI 默认没有
        effort = None
        imgs = _load_images(ep, cameras)
    return imgs, state, action, velocity, effort


# ----------------- parallelizable unit -----------------

def _prepare_episode_for_write(ep_path: Path, task: str):
    """子进程执行：读取 hdf5、解码/缩放图片，返回可序列化的写入包。"""
    imgs, state, action, velocity, effort = _load_episode(ep_path)
    T = min(state.shape[0], action.shape[0], *(arr.shape[0] for arr in imgs.values()))
    pkg = {
        "task": task,
        "state": state[:T].numpy(),          # (T, 8) np.float32
        "action": action[:T].numpy(),        # (T, 8) np.float32
        "velocity": None if velocity is None else velocity[:T].numpy(),
        "effort": None if effort is None else effort[:T].numpy(),
        "images": {k: v[:T] for k, v in imgs.items()},  # 裁齐长度
        "length": int(T),
        "episode_name": ep_path.stem,
    }
    return pkg


def _write_episode_pkg(dataset: LeRobotDataset, pkg):
    """主进程串行写入一个 episode，避免并发写冲突。"""
    T = pkg["length"]
    for i in range(T):
        frame = {
            "observation.state": pkg["state"][i],
            "action": pkg["action"][i],
            "task": pkg["task"],
        }
        for cam, arr in pkg["images"].items():
            frame[f"observation.images.{cam}"] = arr[i]
        if pkg["velocity"] is not None:
            frame["observation.velocity"] = pkg["velocity"][i]
        if pkg["effort"] is not None:
            frame["observation.effort"] = pkg["effort"][i]
        dataset.add_frame(frame)
    dataset.save_episode()

# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def port_fastumi(
    *,
    raw_dir: Path,
    repo_id: str,
    task: str = "DEBUG",
    fps: int = 30,
    output_dir: Path | None = None,
    raw_repo_id: str | None = None,
    episodes: List[int] | None = None,
    push_to_hub: bool = False,
    mode: Literal["video", "image"] = "image",
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
    workers: int = 1,   # 并行 worker 数（仅预处理）
):
    """Convert FastUMI episodes to LeRobot dataset format with parallel preprocessing."""

    # # 1) Gather raw episodes
    # if not raw_dir.exists():
    #     if raw_repo_id is None:
    #         raise FileNotFoundError(
    #             f"{raw_dir} does not exist and --raw-repo-id was not provided."
    #         )
    #     download_raw(raw_dir, repo_id=raw_repo_id)

    hdf5_files: List[Path] = sorted(raw_dir.rglob("*.hdf5"), key=natsort_key)
    if not hdf5_files:
        raise RuntimeError(f"No .hdf5 files found under {raw_dir}")

    if episodes is None:
        ep_indices = list(range(len(hdf5_files)))
    else:
        ep_indices = episodes

    files_to_process = [hdf5_files[i] for i in ep_indices]

    # 2) Create empty dataset (writer settings synced to mode)
    cfg = dataclasses.replace(
        dataset_config,
        use_videos=(mode == "video"),
    )
    dataset = create_empty_dataset(
        repo_id=repo_id,
        fps=fps,
        robot_type="fastumi",
        mode=mode,
        has_velocity=False,
        has_effort=False,
        dataset_config=cfg,
    )

    # 3) Parallel preprocess + serial write (preserve order)
    if workers <= 1:
        for p in tqdm.tqdm(files_to_process, desc="Converting episodes (single)"):
            pkg = _prepare_episode_for_write(p, task)
            _write_episode_pkg(dataset, pkg)
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            # executor.map 保序：输出顺序与输入顺序一致
            it = ex.map(_prepare_episode_for_write, files_to_process, [task]*len(files_to_process), chunksize=1)
            for pkg in tqdm.tqdm(it, total=len(files_to_process), desc=f"Preprocessing (workers={workers})"):
                _write_episode_pkg(dataset, pkg)

    # 3.5) Consolidate for stable on-disk layout
    try:
        dataset.consolidate()
    except Exception as e:
        print(f"[Warn] consolidate failed: {e}")

    # 4) Copy to custom output (optional)
    if output_dir is not None:
        target = output_dir / repo_id
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(HF_LEROBOT_HOME / repo_id, target)
        print(f"[Info] Dataset copied to {target}")

    # 5) Push to Hub (optional)
    if push_to_hub:
        dataset.push_to_hub()

# ---------------------------------------------------------------------------
# Handy preset for a specific dataset (optional direct call)
# ---------------------------------------------------------------------------

def convert_pick_lid_data():
    """示例：专门用于转换 pick_spoon 数据的便捷函数"""
    port_fastumi(
        raw_dir=Path('/gemini/user/private/data_process/organize_small/hdf5'),
        repo_id="Loki/teleai_umi",
        task="Move the pottle to the right",
        fps=60,
        mode="image",
        push_to_hub=False,
        workers=100,
    )

# ---------------------------------------------------------------------------
# CLI (tyro)
# ---------------------------------------------------------------------------
@dataclasses.dataclass
class CLI:
    raw_dir: Path
    repo_id: str
    task: str = "DEBUG"
    fps: int = 30
    output_dir: Optional[Path] = None
    raw_repo_id: Optional[str] = None
    episodes: Optional[List[int]] = None
    push_to_hub: bool = False
    mode: Literal["video", "image"] = "image"
    workers: int = 1
    # dataset writer 内部并行（一般无需改动；有需要可在 CLI 中覆盖）
    image_writer_processes: int = DEFAULT_DATASET_CONFIG.image_writer_processes
    image_writer_threads: int = DEFAULT_DATASET_CONFIG.image_writer_threads
    video_backend: Optional[str] = None

    def run(self):
        cfg = DatasetConfig(
            use_videos=(self.mode == "video"),
            tolerance_s=DEFAULT_DATASET_CONFIG.tolerance_s,
            image_writer_processes=self.image_writer_processes,
            image_writer_threads=self.image_writer_threads,
            video_backend=self.video_backend,
        )
        port_fastumi(
            raw_dir=self.raw_dir,
            repo_id=self.repo_id,
            task=self.task,
            fps=self.fps,
            output_dir=self.output_dir,
            raw_repo_id=self.raw_repo_id,
            episodes=self.episodes,
            push_to_hub=self.push_to_hub,
            mode=self.mode,
            dataset_config=cfg,
            workers=self.workers,
        )

if __name__ == "__main__":
    # 方式一：直接跑内置示例
    convert_pick_lid_data()
    # 方式二：命令行
    # tyro.cli(CLI).run()
