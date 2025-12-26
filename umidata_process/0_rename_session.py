import os
import argparse

def rename_sessions(root, dry_run=True, start_idx=1):
    sessions = sorted(
        d for d in os.listdir(root)
        if d.startswith("session_") and os.path.isdir(os.path.join(root, d))
    )

    print(f"Found {len(sessions)} sessions\n")

    width = 3  # 001

    for i, old in enumerate(sessions, start=start_idx):
        new = f"session_{i:0{width}d}"

        old_path = os.path.join(root, old)
        new_path = os.path.join(root, new)

        if old == new:
            print(f"[SKIP] {old}")
            continue

        if os.path.exists(new_path):
            raise RuntimeError(f"Target already exists: {new}")

        print(f"{old}  ->  {new}")

        if not dry_run:
            os.rename(old_path, new_path)

    print("\n[DRY RUN]" if dry_run else "\n[DONE]")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="dataset root")
    parser.add_argument("--dry-run", action="store_true", help="only print, do not rename")
    parser.add_argument("--start-idx", type=int, default=1, help="start index")
    args = parser.parse_args()

    rename_sessions(
        root=args.root,
        dry_run=args.dry_run,
        start_idx=args.start_idx,
    )
