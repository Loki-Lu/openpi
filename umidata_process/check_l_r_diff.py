import os
import numpy as np
import pandas as pd
from tqdm import tqdm


# =============== 基础 IO（只读时间戳） ===============

def load_video_timestamps(hand_path):
    """
    只读取视频 timestamps.csv
    return: first_timestamp (float)
    """
    ts_path = os.path.join(hand_path, "RGB_Images", "timestamps.csv")
    if not os.path.exists(ts_path):
        raise FileNotFoundError(ts_path)

    df = pd.read_csv(ts_path)
    ts = df["aligned_stamp"].values
    if len(ts) == 0:
        raise RuntimeError(f"Empty timestamps: {ts_path}")

    return float(ts[0])


# =============== 单 session 检查 ===============

def check_one_session(session_path, max_start_diff_ms=50.0):
    hands = os.listdir(session_path)
    left_hand = [h for h in hands if h.startswith("left_hand")]
    right_hand = [h for h in hands if h.startswith("right_hand")]

    if not left_hand or not right_hand:
        raise RuntimeError(f"Missing hands in {session_path}")

    left_path = os.path.join(session_path, left_hand[0])
    right_path = os.path.join(session_path, right_hand[0])

    left_ts0 = load_video_timestamps(left_path)
    right_ts0 = load_video_timestamps(right_path)

    diff_ms = abs(left_ts0 - right_ts0) * 1000.0
    ok = diff_ms < max_start_diff_ms

    return {
        "session": os.path.basename(session_path),
        "left_ts0": left_ts0,
        "right_ts0": right_ts0,
        "diff_ms": diff_ms,
        "ok": ok,
    }


# =============== 扫描入口 ===============

def scan_all_sessions(root, max_start_diff_ms=50.0):
    sessions = sorted(
        d for d in os.listdir(root) if d.startswith("session_")
    )

    bad = []

    print(f"Total sessions: {len(sessions)}")
    print(f"Threshold     : {max_start_diff_ms:.1f} ms\n")

    for s in tqdm(sessions):
        session_path = os.path.join(root, s)
        try:
            info = check_one_session(session_path, max_start_diff_ms)
            if not info["ok"]:
                bad.append(info)
                print(
                    f"[BAD] {info['session']} | "
                    f"left={info['left_ts0']:.6f} "
                    f"right={info['right_ts0']:.6f} "
                    f"diff={info['diff_ms']:.2f} ms"
                )
        except Exception as e:
            print(f"[ERROR] {s}: {e}")

    print("\n========== SUMMARY ==========")
    if not bad:
        print("All sessions OK ✅")
    else:
        print(f"Bad sessions ({len(bad)}):")
        for b in bad:
            print(
                f"  {b['session']} -> {b['diff_ms']:.2f} ms"
            )

    return bad


# =============== main ===============

if __name__ == "__main__":
    ROOT = "/gemini-2/user/private/data/data_umi_fruit"
    scan_all_sessions(ROOT, max_start_diff_ms=50.0)
