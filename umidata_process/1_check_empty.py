"""
Usage examples:

1. Only check, no deletion (default mode):
   python 1_check_empty.py .py

2. Dry-run deletion mode (print sessions that would be deleted):
   python 1_check_empty.py --delete_bad_sessions --dry_run

3. Real deletion (must input confirmation):
   python 1_check_empty.py --delete_bad_sessions

4. Print specific missing / empty file paths:
   python 1_check_empty.py --return_issue_paths

You can combine options, e.g.:
   python 1_check_empty.py --delete_bad_sessions --dry_run --return_issue_paths
==========================================================
"""

import os
import json
import shutil
import numpy as np
import cv2
from tqdm import tqdm
import argparse

# ============================================================
# ---------------------- 命令行参数 --------------------------
# ============================================================

parser = argparse.ArgumentParser(description="UMI / Water data integrity check and optional session cleanup.")

parser.add_argument("--delete_bad_sessions", action="store_true",
                    help="Enable deletion of bad sessions (default: False)")
parser.add_argument("--dry_run", action="store_true",
                    help="Dry-run mode: just print sessions that would be deleted (default: False)")
parser.add_argument("--return_issue_paths", action="store_true",
                    help="Print specific missing/empty file paths (default: False)")

args = parser.parse_args()

DELETE_BAD_SESSIONS = args.delete_bad_sessions
DRY_RUN_ONLY = args.dry_run
RETURN_ISSUE_PATHS = args.return_issue_paths

print(f"\n[CONFIG] DELETE_BAD_SESSIONS={DELETE_BAD_SESSIONS}, DRY_RUN_ONLY={DRY_RUN_ONLY}, RETURN_ISSUE_PATHS={RETURN_ISSUE_PATHS}\n")

# ============================================================
# ---------------------- 配置区 ------------------------------
# ============================================================

ROOT = "/gemini-2/user/private/data/data_umi_fruit"  # 数据根目录
hands = [
    "left_hand_250801DR48FP25002070",
    "right_hand_250801DR48FP25002073",
]

required_files = [
    os.path.join("Clamp_Data", "clamp_data_tum.txt"),
    os.path.join("Merged_Trajectory", "merge_stats.txt"),
    os.path.join("Merged_Trajectory", "merged_trajectory.txt"),
    os.path.join("RGB_Images", "timestamps.csv"),
    os.path.join("RGB_Images", "raw_meta.json"),
    os.path.join("RGB_Images", "video.mp4"),
    os.path.join("SLAM_Poses", "slam_raw.txt"),
    os.path.join("Vive_Poses", "vive_data_tum.txt"),
    os.path.join("Vive_Poses", "offset_info.txt"),
]

# ============================================================
# ---------------------- 工具函数 ------------------------------
# ============================================================

def is_empty_or_nan(file_path):
    if not os.path.isfile(file_path) or os.path.getsize(file_path) == 0:
        return True

    ext = os.path.splitext(file_path)[1].lower()
    rel_path = os.path.join(os.path.basename(os.path.dirname(file_path)), os.path.basename(file_path))

    metadata_files = [
        os.path.join("Merged_Trajectory", "merge_stats.txt"),
        os.path.join("Vive_Poses", "offset_info.txt"),
    ]

    if rel_path in metadata_files:
        return False

    try:
        if ext == ".json":
            with open(file_path, "r") as f:
                data = json.load(f)
            return not bool(data)

        if ext in [".txt", ".csv"]:
            delimiter = "," if ext == ".csv" else None
            data = np.genfromtxt(
                file_path,
                delimiter=delimiter,
                skip_header=1,
                dtype=np.float64,
                encoding=None,
            )
            if data is None or data.size == 0:
                return True
            if np.all(np.isnan(data)):
                return True

    except Exception:
        return True

    return False

def is_video_empty(file_path):
    if not os.path.isfile(file_path) or os.path.getsize(file_path) == 0:
        return True
    try:
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            return True
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return frame_count == 0
    except Exception:
        return True

def require_delete_confirmation():
    print("\n" + "!" * 60)
    print("⚠️  WARNING: YOU ARE ABOUT TO DELETE DATA  ⚠️")
    print("This operation will PERMANENTLY DELETE entire sessions.")
    print("This action is IRREVERSIBLE.")
    print("!" * 60)

    confirm = input('\nType exactly "DELETE ALL BAD SESSIONS" to continue:\n> ')
    if confirm.strip() != "DELETE ALL BAD SESSIONS":
        print("\n❌ Confirmation failed. Aborting.")
        exit(1)
    print("\n✅ Confirmation accepted. Proceeding...\n")

# ============================================================
# ---------------------- 主逻辑 ------------------------------
# ============================================================

# 删除确认（仅在真实删除时）
if DELETE_BAD_SESSIONS and not DRY_RUN_ONLY:
    require_delete_confirmation()

missing_by_type = {f: 0 for f in required_files}
empty_by_type = {f: 0 for f in required_files}
missing_paths = []
empty_paths = []
total_files_checked = 0
total_missing_files = {hand: 0 for hand in hands}
total_empty_or_nan = {hand: 0 for hand in hands}

sessions = [s for s in os.listdir(ROOT) if os.path.isdir(os.path.join(ROOT, s))]
bad_sessions = []

print("🚀 Starting data integrity check...")

for session in tqdm(sessions, desc="Checking Sessions"):
    session_path = os.path.join(ROOT, session)
    session_has_issue = False

    for hand in hands:
        hand_path = os.path.join(session_path, hand)
        if not os.path.isdir(hand_path):
            continue

        for rel_file in required_files:
            file_path = os.path.join(hand_path, rel_file)
            total_files_checked += 1
            ext = os.path.splitext(file_path)[1].lower()

            is_missing = False
            is_empty = False

            if not os.path.isfile(file_path):
                is_missing = True
                session_has_issue = True
                total_missing_files[hand] += 1
                missing_by_type[rel_file] += 1
            else:
                if ext in [".txt", ".csv", ".json"]:
                    is_empty = is_empty_or_nan(file_path)
                elif ext == ".mp4":
                    is_empty = is_video_empty(file_path)
                if is_empty:
                    session_has_issue = True
                    total_empty_or_nan[hand] += 1
                    empty_by_type[rel_file] += 1

            if RETURN_ISSUE_PATHS:
                if is_missing:
                    missing_paths.append(file_path)
                elif is_empty:
                    empty_paths.append(file_path)

    if session_has_issue:
        bad_sessions.append(session_path)
        if DELETE_BAD_SESSIONS:
            if DRY_RUN_ONLY:
                print(f"[DRY-RUN] Will delete session: {session_path}")
            else:
                print(f"[DELETE] Removing session: {session_path}")
                shutil.rmtree(session_path, ignore_errors=True)

# ============================================================
# ---------------------- 输出统计 ------------------------------
# ============================================================

print("\n" + "=" * 50)
print("📊 DATA CHECK SUMMARY")
print("=" * 50)
print(f"Total sessions found: {len(sessions)}")
print(f"Total files checked: {total_files_checked}")

print("\n### 🖐 Summary by Hand ###")
for hand in hands:
    print(f"* {hand}")
    print(f"  - Missing files: {total_missing_files[hand]}")
    print(f"  - Empty / invalid files: {total_empty_or_nan[hand]}")

print("\n### 📁 Missing Files by Type ###")
for k, v in missing_by_type.items():
    if v > 0:
        print(f"  - {k}: {v}")

print("\n### 📁 Empty / Invalid Files by Type ###")
for k, v in empty_by_type.items():
    if v > 0:
        print(f"  - {k}: {v}")

print("\n### 🧹 Session Cleanup ###")
print(f"Bad sessions detected: {len(bad_sessions)}")

if DELETE_BAD_SESSIONS:
    if DRY_RUN_ONLY:
        print("Mode: DRY-RUN (no session deleted)")
    else:
        print("Mode: DELETE ENABLED (sessions removed)")
else:
    print("Mode: CHECK ONLY")

if RETURN_ISSUE_PATHS:
    print("\n--- Missing paths ---")
    for p in missing_paths:
        print(p)

    print("\n--- Empty / invalid paths ---")
    for p in empty_paths:
        print(p)

print("\n✅ Check complete.")
