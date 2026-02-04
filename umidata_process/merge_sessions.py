import shutil
from pathlib import Path
import re
import cv2
import multiprocessing as mp
from tqdm import tqdm

SRC_ROOT = Path("/gemini/space/users/tong/data/lumin")
DST_ROOT = Path("/gemini/space/users/tong/data/lumin_merged224")
DST_ROOT.mkdir(exist_ok=True)

TARGET_SIZE = (224, 224)
NUM_WORKERS = 16  

def process_video_resize(src_video: Path, dst_video: Path):
    """读取原视频，裁剪并调整大小为 224x224，保存为 mp4"""
    cap = cv2.VideoCapture(str(src_video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # 或者 'avc1'
    
    out = cv2.VideoWriter(str(dst_video), fourcc, fps, TARGET_SIZE)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # 调整大小
        resized = cv2.resize(frame, TARGET_SIZE, interpolation=cv2.INTER_AREA)
        out.write(resized)
    
    cap.release()
    out.release()

def process_session(args):
    sess_path, dst_path = args
    # 1. 先创建目标文件夹结构
    dst_path.mkdir(parents=True, exist_ok=True)
    
    # 2. 遍历 session 内部内容 (left_hand_..., right_hand_...)
    for hand_dir in sess_path.iterdir():
        if not hand_dir.is_dir():
            continue
            
        dst_hand_dir = dst_path / hand_dir.name
        dst_hand_dir.mkdir(exist_ok=True)
        
        # 递归复制除了视频以外的所有东西 (csv, txt 等)
        for item in hand_dir.rglob("*"):
            relative_path = item.relative_to(hand_dir)
            target_item = dst_hand_dir / relative_path
            
            if item.is_dir():
                target_item.mkdir(exist_ok=True)
            elif item.suffix.lower() == ".mp4":
                # 【核心修改】：如果是视频，不拷贝，直接进行 Resize 处理
                process_video_resize(item, target_item)
            else:
                # 其他小文件直接拷贝
                if not target_item.exists():
                    shutil.copy2(item, target_item)

def extract_timestamp(name: str):
    m = re.search(r"multi_sessions_(\d+)", name)
    return m.group(1) if m else name

def main():
    task_dirs = sorted(
        [d for d in SRC_ROOT.iterdir() if d.is_dir() and d.name.startswith("task")],
        key=lambda x: x.name
    )

    all_jobs = []

    for task_idx, task_dir in enumerate(task_dirs, start=1):
        dst_task = DST_ROOT / f"task{task_idx}"
        dst_task.mkdir(parents=True, exist_ok=True)

        session_infos = []
        for bg in task_dir.iterdir():
            if not bg.is_dir(): continue
            for ms in bg.iterdir():
                if not ms.is_dir(): continue
                ts = extract_timestamp(ms.name)
                for sess in ms.iterdir():
                    if sess.is_dir() and sess.name.startswith("session_"):
                        session_infos.append((ts, sess))

        session_infos.sort(key=lambda x: x[0])
        print(f"[Task {task_idx}] total sessions: {len(session_infos)}")

        for sess_idx, (_, sess_path) in enumerate(session_infos, start=1):
            new_name = f"session_{sess_idx:03d}"
            dst = dst_task / new_name
            all_jobs.append((sess_path, dst))

    # 使用进程池处理
    print(f"Starting parallel processing with {NUM_WORKERS} workers...")
    with mp.Pool(NUM_WORKERS) as pool:
        list(tqdm(pool.imap_unordered(process_session, all_jobs), total=len(all_jobs)))

if __name__ == "__main__":
    main()