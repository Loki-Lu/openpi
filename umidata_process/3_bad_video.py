import subprocess
from pathlib import Path

ROOT = Path("/gemini-1/space/tong/data/cube_sponge")

bad = []

def check_video(video_path: Path):
    proc = subprocess.run(
        ["ffprobe", "-v", "error", str(video_path)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    if proc.returncode != 0:
        return proc.stderr.strip()
    return None


for session in sorted(ROOT.glob("session_*")):
    for hand in session.iterdir():
        if not hand.is_dir():
            continue
        if not (hand.name.startswith("left_hand") or hand.name.startswith("right_hand")):
            continue

        video = hand / "RGB_Images" / "video.mp4"
        if not video.exists():
            print(f"[MISSING] {video}")
            continue

        err = check_video(video)
        if err:
            tag = "MOOV_MISSING" if "moov atom not found" in err else "BROKEN"
            print(f"[{tag}] {video}")
            bad.append((session.name, hand.name, video, err))

print("\n========== SUMMARY ==========")
print(f"Total broken videos: {len(bad)}")
