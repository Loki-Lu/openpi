import os
import pandas as pd

def filter_sessions(root_dir, max_diff_ms=50.0):
    """
    快速扫描 root_dir 下的所有 session，筛选出左右手同步合格的文件。
    
    返回:
        valid_sessions (list): 包含合格 session 路径的列表
        corrupt_sessions (list): 包含错误原因和路径的列表
    """
    valid_sessions = []
    corrupt_sessions = []
    
    # 找到所有 session 文件夹
    sessions = sorted([d for d in os.listdir(root_dir) if d.startswith("session_")])
    
    for s in sessions:
        session_path = os.path.join(root_dir, s)
        try:
            # 1. 检查是否存在左右手文件夹
            hands = os.listdir(session_path)
            l_hand_dir = [h for h in hands if h.startswith("left_hand")]
            r_hand_dir = [h for h in hands if h.startswith("right_hand")]
            
            if not l_hand_dir or not r_hand_dir:
                corrupt_sessions.append({"path": s, "reason": "Missing hand folder"})
                continue
            
            # 2. 读取两只手的时间戳 csv
            l_ts_path = os.path.join(session_path, l_hand_dir[0], "RGB_Images", "timestamps.csv")
            r_ts_path = os.path.join(session_path, r_hand_dir[0], "RGB_Images", "timestamps.csv")
            
            if not os.path.exists(l_ts_path) or not os.path.exists(r_ts_path):
                corrupt_sessions.append({"path": s, "reason": "Missing timestamps.csv"})
                continue
            
            l_ts = pd.read_csv(l_ts_path)["aligned_stamp"].values[0]
            r_ts = pd.read_csv(r_ts_path)["aligned_stamp"].values[0]
            
            # 3. 计算起始差值 (单位: ms)
            diff_ms = abs(l_ts - r_ts) * 1000.0
            
            if diff_ms <= max_diff_ms:
                valid_sessions.append(session_path)
            else:
                corrupt_sessions.append({
                    "path": s, 
                    "reason": f"Sync diff too large ({diff_ms:.2f} ms)"
                })
                
        except Exception as e:
            corrupt_sessions.append({"path": s, "reason": f"Error: {str(e)}"})

    # 打印简要报告
    print(f"\n{'='*40}")
    print(f"Filter Report (Limit: {max_diff_ms} ms)")
    print(f"Total Sessions: {len(sessions)}")
    print(f"Valid Sessions: {len(valid_sessions)}")
    print(f"Corrupt/Skipped: {len(corrupt_sessions)}")
    for item in corrupt_sessions:
        print(f" - [SKIP] {item['path']}: {item['reason']}")
    print(f"{'='*40}\n")
    
    return valid_sessions, corrupt_sessions

# ================= 使用示例 =================
if __name__ == "__main__":
    ROOT = "/gemini-2/user/private/data/data_umi_fruit"
    
    # 得到筛选后的合格路径列表
    valid_paths, _ = filter_sessions(ROOT, max_diff_ms=50)
    
    # 之后你可以把这个 valid_paths 传给你的并行转换函数
    # tasks = [(p, out_p) for idx, p in enumerate(valid_paths)]