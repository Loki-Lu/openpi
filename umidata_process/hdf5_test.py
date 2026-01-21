import h5py
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

def check_hdf5_save(session_path, save_root="./check_frames"):
    """
    session_path: session 文件夹路径
    save_root: 保存图片的根目录
    """
    os.makedirs(save_root, exist_ok=True)

    # 1. HDF5 文件路径
    hdf5_dir = os.path.join(session_path, "hdf5")
    # 确保路径存在
    if not os.path.exists(hdf5_dir):
        print(f"路径不存在: {hdf5_dir}")
        return

    h5_files = [f for f in os.listdir(hdf5_dir) if f.endswith(".hdf5")]
    if not h5_files:
        print("没有找到 HDF5 文件")
        return
    
    # 选取第一个找到的文件
    h5_path = os.path.join(hdf5_dir, h5_files[0])
    print("检查文件:", h5_path)

    # 2. 读取 HDF5 数据
    try:
        with h5py.File(h5_path, "r") as f:
            # 适配 left_hand
            left_qpos = f["left_hand/observations/state"][:]
            left_action = f["left_hand/action"][:]
            left_clamp = left_action[:, -1:] # 假设最后一列是夹爪
            left_frames = f["left_hand/observations/images/wrist"][:]

            # 适配 right_hand
            right_qpos = f["right_hand/observations/state"][:]
            right_action = f["right_hand/action"][:]
            right_clamp = right_action[:, -1:]
            right_frames = f["right_hand/observations/images/wrist"][:]
    except KeyError as e:
        print(f"错误：在 HDF5 中找不到键值 {e}")
        print("请检查文件结构是否为 left_hand/observations/state 等")
        return

    # 3. 打印前10帧数字信息
    print("\n=== 前10帧 Left Hand ===")
    for i in range(min(10, left_qpos.shape[0])):
        print(f"Frame {i}: qpos={left_qpos[i]}, clamp={left_clamp[i]}")

    print("\n=== 前10帧 Right Hand ===")
    for i in range(min(10, right_qpos.shape[0])):
        print(f"Frame {i}: qpos={right_qpos[i]}, clamp={right_clamp[i]}")

    # 4. 保存前5帧图片
    # 将内部变量放进列表方便循环
    robots = [
        ("left_hand", left_frames),
        ("right_hand", right_frames)
    ]

    for robot_name, frames in robots:
        # 创建保存目录
        save_dir = os.path.join(save_root, "check_results", robot_name)
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"\n正在保存 {robot_name} 的图片...")
        for i in range(min(5, frames.shape[0])):
            save_path = os.path.join(save_dir, f"frame_{i}.png")
            # 注意：如果图像很大，plt.imsave 可能会比较慢
            plt.imsave(save_path, frames[i])
            print(f"Saved {robot_name} frame {i} to {save_path}")

if __name__ == "__main__":
    # 根据你报错信息中的路径修改
    session_path = "/gemini-1/space/tong/data/water_trash" 
    check_hdf5_save(session_path)