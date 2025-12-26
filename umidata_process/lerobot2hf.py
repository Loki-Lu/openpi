#!/usr/bin/env python3
import os
import shutil
from huggingface_hub import HfApi, Repository

def upload_to_hf(local_lerobot_dir, repo_name, hf_token):
    """
    上传 LeRobot 数据集到 Hugging Face
    Args:
        local_lerobot_dir: 本地 LeRobot 数据集文件夹
        repo_name: HF Dataset 名称
        hf_token: Hugging Face 访问 token
    """
    api = HfApi()
    
    # 创建 dataset repo
    repo_url = api.create_repo(
        name=repo_name,
        token=hf_token,
        repo_type="dataset"
    )
    
    # 克隆仓库到本地
    tmp_dir = "/tmp/lerobot_hf_repo"
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    repo = Repository(local_dir=tmp_dir, clone_from=repo_url)

    # 拷贝 LeRobot 数据到 repo
    dest_dir = os.path.join(tmp_dir, "data")
    shutil.copytree(local_lerobot_dir, dest_dir)

    # 上传到 Hugging Face
    repo.push_to_hub(commit_message="Upload LeRobot dataset")
    print(f"[OK] Dataset uploaded to Hugging Face: {repo_url}")

if __name__ == "__main__":
    LOCAL_LEROBOT_DIR = "/gemini/user/private/data_process/organize_small/lerobot"
    HF_REPO_NAME = "my_lerobot_dataset"  # 修改为你想要的名字
    HF_TOKEN = "YOUR_HF_TOKEN"  # 替换为你的 Hugging Face token

    upload_to_hf(LOCAL_LEROBOT_DIR, HF_REPO_NAME, HF_TOKEN)
