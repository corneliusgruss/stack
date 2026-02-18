"""
Process raw capture sessions through DROID-SLAM to generate 6DoF poses.

For sessions captured with the ultrawide camera (no ARKit poses),
this script runs DROID-SLAM offline to fill in poses.json.

Usage:
    # Process a single session
    stack-slam --session data/raw/session_2026-02-20_143000

    # Process all unprocessed sessions in a directory
    stack-slam --data-dir data/raw

    # With custom calibration scale
    stack-slam --session data/raw/session_... --scale 1.0

Requires: droid-slam, lietorch, pytorch with CUDA
See notebooks/run_slam.ipynb for Colab version.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# DROID-SLAM imports are deferred to avoid import errors on machines without GPU
DROID_AVAILABLE = False
try:
    import torch
    if torch.cuda.is_available():
        try:
            from droid_slam import Droid
            DROID_AVAILABLE = True
        except ImportError:
            pass
except ImportError:
    pass


def load_intrinsics(session_dir: Path) -> np.ndarray:
    """Load camera intrinsics from calib.txt.

    Returns [fx, fy, cx, cy] array. If calib.txt is missing,
    DROID-SLAM can auto-calibrate (opt_intr=True).
    """
    calib_file = session_dir / "calib.txt"
    if not calib_file.exists():
        return None

    parts = calib_file.read_text().strip().split()
    intrinsics = np.array([float(x) for x in parts[:4]], dtype=np.float64)
    return intrinsics


def load_rgb_paths(session_dir: Path) -> list[Path]:
    """Load sorted RGB frame paths from session."""
    rgb_dir = session_dir / "rgb"
    if not rgb_dir.exists():
        raise FileNotFoundError(f"No rgb/ directory in {session_dir}")
    paths = sorted(rgb_dir.glob("*.jpg"))
    if not paths:
        raise FileNotFoundError(f"No JPG files in {rgb_dir}")
    return paths


def load_metadata(session_dir: Path) -> dict:
    """Load session metadata."""
    meta_file = session_dir / "metadata.json"
    if not meta_file.exists():
        raise FileNotFoundError(f"No metadata.json in {session_dir}")
    with open(meta_file) as f:
        return json.load(f)


def check_session_needs_slam(session_dir: Path) -> bool:
    """Check if a session needs SLAM processing."""
    meta = load_metadata(session_dir)
    source = meta.get("captureSource", "iphone_arkit")
    if source == "iphone_arkit":
        return False  # ARKit already provides poses
    processed = meta.get("slamProcessed", False)
    if processed:
        return False  # Already processed
    return True


def run_droid_slam(
    session_dir: Path,
    scale_factor: float = 1.0,
    opt_intr: bool = False,
    stride: int = 1,
) -> np.ndarray:
    """Run DROID-SLAM on a session's RGB frames.

    Args:
        session_dir: Path to session directory.
        scale_factor: Metric scale correction factor.
        opt_intr: Whether to optimize intrinsics (use if no calib.txt).
        stride: Frame stride (1 = every frame, 2 = every other, etc.)

    Returns:
        (N, 4, 4) array of SE3 pose matrices.
    """
    if not DROID_AVAILABLE:
        raise RuntimeError(
            "DROID-SLAM not available. Install with:\n"
            "  pip install droid-slam\n"
            "Requires CUDA GPU. See notebooks/run_slam.ipynb for Colab."
        )

    import torch
    from droid_slam import Droid
    import cv2

    rgb_paths = load_rgb_paths(session_dir)
    intrinsics = load_intrinsics(session_dir)

    if intrinsics is None:
        print("No calib.txt found — enabling intrinsic optimization")
        opt_intr = True
        # Default ultrawide estimate for 480x360
        intrinsics = np.array([300.0, 300.0, 240.0, 180.0])

    print(f"Processing {len(rgb_paths)} frames from {session_dir.name}")
    print(f"Intrinsics: fx={intrinsics[0]:.1f} fy={intrinsics[1]:.1f} cx={intrinsics[2]:.1f} cy={intrinsics[3]:.1f}")

    # Initialize DROID-SLAM
    droid = Droid(
        image_size=[360, 480],  # H, W
        intrinsics=intrinsics,
        opt_intr=opt_intr,
        buffer=512,
        beta=0.3,
    )

    # Feed frames
    for i, path in enumerate(rgb_paths):
        if i % stride != 0:
            continue
        image = cv2.imread(str(path))
        if image is None:
            print(f"Warning: Could not read {path}")
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        timestamp = float(i) / 60.0  # Approximate timestamp

        droid.track(timestamp, image)

        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(rgb_paths)} frames")

    # Run global bundle adjustment
    print("Running global bundle adjustment...")
    traj = droid.terminate()  # Returns (N, 7) [tx, ty, tz, qx, qy, qz, qw]

    # Convert to 4x4 pose matrices
    from scipy.spatial.transform import Rotation

    num_poses = traj.shape[0]
    poses_4x4 = np.zeros((num_poses, 4, 4), dtype=np.float64)
    for i in range(num_poses):
        t = traj[i, :3]
        q = traj[i, 3:]  # [qx, qy, qz, qw]
        R = Rotation.from_quat(q).as_matrix()
        poses_4x4[i, :3, :3] = R
        poses_4x4[i, :3, 3] = t * scale_factor
        poses_4x4[i, 3, 3] = 1.0

    print(f"Got {num_poses} poses from DROID-SLAM")
    return poses_4x4


def apply_scale_correction(
    poses: np.ndarray,
    known_distance_slam: float,
    known_distance_real: float,
) -> np.ndarray:
    """Apply metric scale correction to SLAM poses.

    Monocular SLAM has arbitrary scale. Use a known distance
    (e.g., measured size of a stacking cube visible in the scene)
    to compute the correction factor.

    Args:
        poses: (N, 4, 4) pose matrices.
        known_distance_slam: Distance in SLAM coordinates.
        known_distance_real: True distance in meters.

    Returns:
        Scale-corrected poses.
    """
    scale = known_distance_real / known_distance_slam
    corrected = poses.copy()
    corrected[:, :3, 3] *= scale
    print(f"Scale correction: {scale:.4f} (SLAM {known_distance_slam:.4f} -> real {known_distance_real:.4f})")
    return corrected


def write_poses(session_dir: Path, poses_4x4: np.ndarray):
    """Write SLAM poses to session's poses.json in StackCapture format."""
    rgb_paths = load_rgb_paths(session_dir)

    # Build poses.json entries matching existing format
    poses_list = []
    for i in range(min(len(poses_4x4), len(rgb_paths))):
        # Use frame index as approximate timestamp
        timestamp = float(i) / 60.0

        poses_list.append({
            "timestamp": timestamp,
            "rgbIndex": i,
            "depth": None,  # No depth from monocular SLAM (could use DROID depth maps)
            "transform": poses_4x4[i].tolist(),
        })

    # Write poses.json
    poses_file = session_dir / "poses.json"
    with open(poses_file, "w") as f:
        json.dump(poses_list, f, indent=2)
    print(f"Wrote {len(poses_list)} poses to {poses_file}")

    # Update metadata
    meta_file = session_dir / "metadata.json"
    with open(meta_file) as f:
        meta = json.load(f)
    meta["slamProcessed"] = True
    meta["poseCount"] = len(poses_list)
    with open(meta_file, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Updated metadata: slamProcessed=true")


def process_session(
    session_dir: Path,
    scale_factor: float = 1.0,
    force: bool = False,
):
    """Full pipeline: check → run SLAM → write poses.

    Args:
        session_dir: Path to session directory.
        scale_factor: Metric scale correction.
        force: Re-process even if already processed.
    """
    session_dir = Path(session_dir)

    if not force and not check_session_needs_slam(session_dir):
        meta = load_metadata(session_dir)
        source = meta.get("captureSource", "iphone_arkit")
        if source == "iphone_arkit":
            print(f"Skipping {session_dir.name}: ARKit session (already has poses)")
        else:
            print(f"Skipping {session_dir.name}: already SLAM processed")
        return

    print(f"\n{'='*60}")
    print(f"Processing: {session_dir.name}")
    print(f"{'='*60}")

    poses = run_droid_slam(session_dir, scale_factor=scale_factor)
    write_poses(session_dir, poses)
    print(f"Done: {session_dir.name}")


def find_sessions(data_dir: Path) -> list[Path]:
    """Find all session directories in a data directory."""
    data_dir = Path(data_dir)
    sessions = sorted([
        d for d in data_dir.iterdir()
        if d.is_dir() and d.name.startswith("session_")
    ])
    return sessions


def main():
    parser = argparse.ArgumentParser(
        description="Process raw capture sessions through DROID-SLAM"
    )
    parser.add_argument(
        "--session", type=str, default=None,
        help="Path to a single session directory"
    )
    parser.add_argument(
        "--data-dir", type=str, default=None,
        help="Process all unprocessed sessions in this directory"
    )
    parser.add_argument(
        "--scale", type=float, default=1.0,
        help="Metric scale correction factor"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-process even if already processed"
    )
    args = parser.parse_args()

    if args.session:
        process_session(Path(args.session), scale_factor=args.scale, force=args.force)
    elif args.data_dir:
        sessions = find_sessions(Path(args.data_dir))
        print(f"Found {len(sessions)} sessions in {args.data_dir}")
        for session_dir in sessions:
            process_session(session_dir, scale_factor=args.scale, force=args.force)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
