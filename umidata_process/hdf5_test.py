import h5py
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

def check_hdf5_save(session_path, save_root="./check_frames"):
    """
    session_path: session 文件夹路径，例如：
        /gemini/user/private/data_process/organize_small/session_001
    save_root: 保存图片的根目录
    """
    os.makedirs(save_root, exist_ok=True)

    # 1. HDF5 文件路径
    hdf5_dir = os.path.join(session_path, "hdf5")
    h5_files = [f for f in os.listdir(hdf5_dir) if f.endswith(".hdf5")]
    if not h5_files:
        print("没有找到 HDF5 文件")
        return
    h5_path = os.path.join(hdf5_dir, h5_files[0])
    print("检查文件:", h5_path)

    # 2. 读取 HDF5 数据
    with h5py.File(h5_path, "r") as f:
        robot0_qpos = f["robot_0/observations/qpos"][:]
        robot0_action = f["robot_0/action"][:]
        robot0_clamp = robot0_action[:, -1:]
        robot0_frames = f["robot_0/observations/images/cam0"][:]

        robot1_qpos = f["robot_1/observations/qpos"][:]
        robot1_action = f["robot_1/action"][:]
        robot1_clamp = robot1_action[:, -1:]
        robot1_frames = f["robot_1/observations/images/cam0"][:]

    # 3. 打印前10帧数字信息
    print("\n=== 前10帧 Robot 0 ===")
    for i in range(min(10, robot0_qpos.shape[0])):
        print(f"Frame {i}: qpos={robot0_qpos[i]}, clamp={robot0_clamp[i]}")

    print("\n=== 前10帧 Robot 1 ===")
    for i in range(min(10, robot1_qpos.shape[0])):
        print(f"Frame {i}: qpos={robot1_qpos[i]}, clamp={robot1_clamp[i]}")

    # 4. 保存前5帧图片
    for robot_name, frames in zip(["robot_0", "robot_1"], [robot0_frames, robot1_frames]):
        save_dir = os.path.join(save_root, os.path.basename(session_path), robot_name)
        os.makedirs(save_dir, exist_ok=True)
        for i in range(min(5, frames.shape[0])):
            save_path = os.path.join(save_dir, f"frame_{i}.png")
            plt.imsave(save_path, frames[i])
            print(f"Saved {robot_name} frame {i} to {save_path}")

if __name__ == "__main__":
    session_path = "/gemini/user/private/data_process/organize_small/session_001"
    check_hdf5_save(session_path)
