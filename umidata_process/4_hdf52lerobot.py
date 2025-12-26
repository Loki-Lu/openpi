#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert FastUMI HDF5 double-arm data to LeRobot v2.0 dataset format.
Supports parallel preprocessing and serial writing to avoid HDF5 concurrency issues.


"""
from __future__ import annotations
import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
import dataclasses
import shutil
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Literal
from concurrent.futures import ProcessPoolExecutor

import h5py
import numpy as np
import torch
import tqdm
import cv2
import re
from lerobot.common.datasets.lerobot_dataset import (
    HF_LEROBOT_HOME,
    LeRobotDataset,
)


# ---------------------------------------------------------------------------
# Dataset configuration
# ---------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class DatasetConfig:
    use_videos: bool = False
    tolerance_s: float = 1e-4
    image_writer_processes: int = 2
    image_writer_threads: int = 4
    video_backend: str | None = None


DEFAULT_DATASET_CONFIG = DatasetConfig()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def natsort_key(p: Path):
    def tok(s: str):
        return [int(t) if t.isdigit() else t for t in re.findall(r"\d+|\D+", s)]
    key: List[object] = []
    for part in p.parts:
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
    """Create an empty LeRobot dataset with double-arm 16 DoF features."""

    state_names = [
        # left arm
        "x_l", "y_l", "z_l", "qx_l", "qy_l", "qz_l", "qw_l", "gripper_l",
        # right arm
        "x_r", "y_r", "z_r", "qx_r", "qy_r", "qz_r", "qw_r", "gripper_r",
    ]

    features: Dict[str, Dict[str, object]] = {
        "observation.state": {"dtype": "float32", "shape": (16,), "names": state_names},
        "action": {"dtype": "float32", "shape": (16,), "names": state_names},
    }

    if has_velocity:
        features["observation.velocity"] = {"dtype": "float32", "shape": (16,), "names": state_names}
    if has_effort:
        features["observation.effort"] = {"dtype": "float32", "shape": (16,), "names": state_names}

    # 修改 cameras 列表为左右手
    cameras = ["left_wrist", "right_wrist"]

    # 在 features 中注册左右手图片
    for cam in cameras:
        features[f"observation.images.{cam}"] = {"dtype": mode, "shape": (224, 224, 3), "names": ["H","W","C"]}

    # 清理旧目录
    if (HF_LEROBOT_HOME / repo_id).exists():
        shutil.rmtree(HF_LEROBOT_HOME / repo_id)

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
# Episode loading
# ---------------------------------------------------------------------------
def _load_images(images_group: h5py.Group, cameras: List[str]) -> Dict[str, np.ndarray]:
    """Load and resize images (supports raw pixels or JPEG bytes)."""
    imgs: Dict[str, np.ndarray] = {}
    for cam in cameras:
        ds = images_group[cam]  # 相对路径访问
        if ds.ndim == 4:
            arr = ds[:]             # (T,H,W,3)
            arr = arr[..., ::-1]    # BGR -> RGB
        else:
            decoded: List[np.ndarray] = []
            for buf in ds:
                bgr = cv2.imdecode(buf, 1)
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                decoded.append(rgb)
            arr = np.stack(decoded, axis=0)
        # resize
        T = arr.shape[0]
        resized = np.empty((T, 224, 224, 3), dtype=arr.dtype)
        for i in range(T):
            resized[i] = cv2.resize(arr[i], (224, 224), interpolation=cv2.INTER_AREA)
        imgs[cam] = resized
    return imgs


def _load_episode(ep_path: Path) -> Tuple[
    Dict[str, np.ndarray],
    torch.Tensor,
    torch.Tensor,
    Optional[torch.Tensor],
    Optional[torch.Tensor],
]:
    """Load a double-arm episode from HDF5 and merge states/actions to 16 DoF."""
    hands = ["left_hand", "right_hand"]
    cameras = ["wrist"]
    state_list, action_list = [], []
    imgs_dict: Dict[str, Dict[str, np.ndarray]] = {}

    with h5py.File(ep_path, "r") as ep:
        for hand in hands:
            state_list.append(torch.from_numpy(ep[f"{hand}/observations/state"][:].astype(np.float32)))
            action_list.append(torch.from_numpy(ep[f"{hand}/action"][:].astype(np.float32)))
            imgs_hand = _load_images(ep[f"{hand}/observations/images"], cameras)
            # 将 wrist 图片映射为 left_wrist / right_wrist
            for cam, arr in imgs_hand.items():
                imgs_dict[f"{hand.split('_')[0]}_{cam}"] = arr

    # 合并双臂状态/动作
    state = torch.cat(state_list, dim=1)
    action = torch.cat(action_list, dim=1)

    velocity = None
    effort = None

    return imgs_dict, state, action, velocity, effort



# ---------------------------------------------------------------------------
# Prepare & write
# ---------------------------------------------------------------------------
def _prepare_episode_for_write(ep_path: Path, task: str):
    imgs, state, action, velocity, effort = _load_episode(ep_path)
    T = min(state.shape[0], action.shape[0], *(v.shape[0] for v in imgs.values()))
    pkg = {
        "task": task,
        "state": state[:T].numpy(),
        "action": action[:T].numpy(),
        "velocity": None if velocity is None else velocity[:T].numpy(),
        "effort": None if effort is None else effort[:T].numpy(),
        "images": {k: v[:T] for k, v in imgs.items()},
        "length": int(T),
        "episode_name": ep_path.stem,
    }
    return pkg


def _write_episode_pkg(dataset: LeRobotDataset, pkg, repo_id: str):
    T = pkg["length"]
    for i in range(T):
        frame = {"observation.state": pkg["state"][i], "action": pkg["action"][i], "task": pkg["task"]}
        for cam, arr in pkg["images"].items():
            frame[f"observation.images.{cam}"] = arr[i]  # now cam could be left_wrist / right_wrist
        if pkg["velocity"] is not None:
            frame["observation.velocity"] = pkg["velocity"][i]
        if pkg["effort"] is not None:
            frame["observation.effort"] = pkg["effort"][i]
        dataset.add_frame(frame)
    dataset.save_episode()

    local_path = HF_LEROBOT_HOME / repo_id
    print(f"[Info] Episode '{pkg['episode_name']}' written to {local_path}")




# ---------------------------------------------------------------------------
# Main conversion
# ---------------------------------------------------------------------------
def port_fastumi(
    *,
    raw_dir: Path,
    repo_id: str,
    task: str = "DEBUG",
    fps: int = 60,
    output_dir: Optional[Path] = None,
    episodes: Optional[List[int]] = None,
    push_to_hub: bool = False,
    mode: Literal["video", "image"] = "image",
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
    workers: int = 1,
):
    hdf5_files: List[Path] = sorted(raw_dir.rglob("*.hdf5"), key=natsort_key)
    if not hdf5_files:
        raise RuntimeError(f"No .hdf5 files found under {raw_dir}")

    ep_indices = episodes if episodes is not None else list(range(len(hdf5_files)))
    files_to_process = [hdf5_files[i] for i in ep_indices]

    cfg = dataclasses.replace(dataset_config, use_videos=(mode == "video"))
    dataset = create_empty_dataset(repo_id=repo_id, fps=fps, mode=mode, dataset_config=cfg)

    if workers <= 1:
        for p in tqdm.tqdm(files_to_process, desc="Converting episodes (single)"):
            pkg = _prepare_episode_for_write(p, task)
            _write_episode_pkg(dataset, pkg, repo_id)
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            it = ex.map(_prepare_episode_for_write, files_to_process, [task]*len(files_to_process), chunksize=1)
            for pkg in tqdm.tqdm(it, total=len(files_to_process), desc=f"Preprocessing (workers={workers})"):
                _write_episode_pkg(dataset, pkg, repo_id)

    try:
        dataset.consolidate()
    except Exception as e:
        print(f"[Warn] consolidate failed: {e}")

    if output_dir is not None:
        target = output_dir / repo_id
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(HF_LEROBOT_HOME / repo_id, target)
        print(f"[Info] Dataset copied to {target}")

    if push_to_hub:
        dataset.push_to_hub()


# ---------------------------------------------------------------------------
# Example conversion
# ---------------------------------------------------------------------------
def convert_pick_lid_data():
    port_fastumi(
        raw_dir=Path('/gemini-2/user/private/data/data_umi_fruit/hdf5'),
        repo_id="Loki0929/teleai_umi",
        task="Put the eggplant and bananas into the basket.",
        fps=60,
        mode="image",
        push_to_hub=True,
        workers=40,
        output_dir=Path('/gemini-2/user/private/data/data_umi_fruit/lerobot'),
    )


if __name__ == "__main__":
    convert_pick_lid_data()
