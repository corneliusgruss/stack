"""Quick data verification for SCC."""
from pathlib import Path
from stack.data.iphone_loader import load_session

raw = Path("data/raw")
sessions = sorted(d for d in raw.iterdir() if d.is_dir() and d.name.startswith("session_"))
print(f"{len(sessions)} sessions found")

ok = 0
for s in sessions:
    sess = load_session(str(s))
    n_rgb = sess.num_rgb_frames
    n_poses = sess.num_poses
    status = "OK" if n_rgb >= n_poses else f"MISSING {n_poses - n_rgb} frames"
    print(f"  {s.name}: {n_poses} poses, {n_rgb} rgb — {status}")
    if n_rgb >= n_poses:
        ok += 1

print(f"\n{ok}/{len(sessions)} sessions complete")
