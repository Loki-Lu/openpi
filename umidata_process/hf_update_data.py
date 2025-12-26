#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Robust uploader for large LeRobot datasets.

- Uses HfApi.upload_large_folder (resume-safe)
- No duplicate upload (hash-based)
- Handles network disconnects gracefully

Usage:
  python upload_lerobot_dataset.py \
    --folder /abs/path/to/lerobot/Loki0929/teleai_umi \
    --repo Loki0929/teleai_umi
"""

import argparse
import os
from huggingface_hub import HfApi
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Upload large LeRobot dataset to HuggingFace Hub")
    parser.add_argument(
        "--folder",
        type=str,
        required=True,
        help="Absolute path to local LeRobot dataset folder",
    )
    parser.add_argument(
        "--repo",
        type=str,
        required=True,
        help="HuggingFace repo id, e.g. Loki0929/teleai_umi",
    )
    parser.add_argument(
        "--repo-type",
        type=str,
        default="dataset",
        choices=["dataset"],
        help="Repo type (default: dataset)",
    )
    args = parser.parse_args()

    folder = Path(args.folder).expanduser().resolve()

    # ---- sanity checks ----
    if not folder.exists():
        raise RuntimeError(f"Folder does not exist: {folder}")
    if not folder.is_dir():
        raise RuntimeError(f"Not a directory: {folder}")

    required = ["episodes", "data", "dataset_info.json"]
    missing = [r for r in required if not (folder / r).exists()]
    if missing:
        print("[Warn] Folder does not look like a standard LeRobot dataset.")
        print("       Missing:", missing)

    print("=" * 80)
    print("Uploading LeRobot dataset")
    print("Local folder :", folder)
    print("HF repo      :", args.repo)
    print("=" * 80)

    api = HfApi()

    # ---- upload ----
    api.upload_large_folder(
        folder_path=str(folder),
        repo_id=args.repo,
        repo_type=args.repo_type,
    )

    print("\n✅ Upload finished successfully.")
    print("   Repo:", f"https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()
