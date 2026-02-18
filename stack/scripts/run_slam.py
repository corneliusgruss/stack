"""
Process raw capture sessions through COLMAP SfM to generate 6DoF poses.

For sessions captured with the ultrawide camera (no ARKit poses),
this script runs COLMAP offline to fill in poses.json.

Usage:
    # Process a single session
    python -m stack.scripts.run_slam --session data/raw/session_2026-02-17_220023

    # Process all unprocessed sessions
    python -m stack.scripts.run_slam --data-dir data/raw

    # Force reprocess
    python -m stack.scripts.run_slam --data-dir data/raw --force

Requires: colmap (brew install colmap), pycolmap
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation, Slerp

try:
    import pycolmap
except ImportError:
    print("ERROR: pycolmap not installed. Run: pip install pycolmap")
    sys.exit(1)


def check_colmap():
    """Verify COLMAP CLI is installed."""
    result = subprocess.run(
        ["colmap", "-h"], capture_output=True, text=True
    )
    if result.returncode != 0:
        print("ERROR: colmap not found. Install with: brew install colmap")
        sys.exit(1)


def load_metadata(session_dir: Path) -> dict:
    meta_file = session_dir / "metadata.json"
    if not meta_file.exists():
        raise FileNotFoundError(f"No metadata.json in {session_dir}")
    with open(meta_file) as f:
        return json.load(f)


def needs_processing(session_dir: Path) -> bool:
    meta = load_metadata(session_dir)
    if meta.get("captureSource", "iphone_arkit") == "iphone_arkit":
        return False
    return not meta.get("slamProcessed", False)


def run_colmap_sfm(session_dir: Path, subsample: int = 6) -> tuple[dict, object]:
    """Run COLMAP SfM on a session. Returns (sub_poses, reconstruction)."""
    rgb_dir = session_dir / "rgb"
    frame_paths = sorted(rgb_dir.glob("*.jpg"))
    n_total = len(frame_paths)

    # Subsample frames
    sub_indices = list(range(0, n_total, subsample))
    if sub_indices[-1] != n_total - 1:
        sub_indices.append(n_total - 1)

    # Work in temp directory for speed
    work = Path(tempfile.mkdtemp(prefix="colmap_"))
    try:
        (work / "images").mkdir()
        (work / "sparse").mkdir()

        # Copy subsampled frames
        for new_idx, orig_idx in enumerate(sub_indices):
            shutil.copy2(frame_paths[orig_idx], work / "images" / f"{new_idx:06d}.jpg")

        db = work / "database.db"
        env = os.environ.copy()

        # Feature extraction
        result = subprocess.run([
            "colmap", "feature_extractor",
            "--database_path", str(db),
            "--image_path", str(work / "images"),
            "--ImageReader.camera_model", "SIMPLE_RADIAL",
            "--ImageReader.single_camera", "1",
            "--SiftExtraction.max_image_size", "480",
            "--SiftExtraction.max_num_features", "4096",
        ], capture_output=True, text=True, env=env)
        if result.returncode != 0:
            raise RuntimeError(f"Feature extraction failed: {result.stderr[-300:]}")

        # Sequential matching
        result = subprocess.run([
            "colmap", "sequential_matcher",
            "--database_path", str(db),
            "--SequentialMatching.overlap", "10",
            "--SequentialMatching.loop_detection", "0",
        ], capture_output=True, text=True, env=env)
        if result.returncode != 0:
            raise RuntimeError(f"Sequential matching failed: {result.stderr[-300:]}")

        # Mapper
        result = subprocess.run([
            "colmap", "mapper",
            "--database_path", str(db),
            "--image_path", str(work / "images"),
            "--output_path", str(work / "sparse"),
            "--Mapper.init_min_num_inliers", "50",
            "--Mapper.ba_refine_focal_length", "1",
            "--Mapper.ba_refine_extra_params", "1",
        ], capture_output=True, text=True, env=env)
        if result.returncode != 0:
            raise RuntimeError(f"Mapper failed: {result.stderr[-300:]}")

        # Find best reconstruction
        recon_dirs = sorted((work / "sparse").iterdir())
        if not recon_dirs:
            raise RuntimeError("No reconstruction produced")

        best_dir = max(
            recon_dirs,
            key=lambda d: pycolmap.Reconstruction(str(d)).num_images()
            if (d / "images.bin").exists() else 0
        )
        recon = pycolmap.Reconstruction(str(best_dir))

        # Extract poses
        sub_poses = {}
        for img_id, img in recon.images.items():
            sub_idx = int(Path(img.name).stem)
            orig_idx = sub_indices[sub_idx]
            cfw = img.cam_from_world()
            R_cw = np.array(cfw.rotation.matrix())
            t_cw = np.array(cfw.translation)
            R_wc = R_cw.T
            t_wc = -R_wc @ t_cw
            pose = np.eye(4)
            pose[:3, :3] = R_wc
            pose[:3, 3] = t_wc
            sub_poses[orig_idx] = pose

        return sub_poses, recon, n_total, sub_indices

    finally:
        shutil.rmtree(work, ignore_errors=True)


def interpolate_poses(sub_poses: dict, n_total: int) -> tuple[list, np.ndarray]:
    """Interpolate subsampled poses to full framerate."""
    reg_indices = sorted(sub_poses.keys())
    key_idx = np.array(reg_indices, dtype=float)
    key_t = np.array([sub_poses[i][:3, 3] for i in reg_indices])
    key_R = Rotation.from_matrix([sub_poses[i][:3, :3] for i in reg_indices])

    first, last = reg_indices[0], reg_indices[-1]
    all_idx = np.arange(first, last + 1, dtype=float)

    interp_x = interp1d(key_idx, key_t[:, 0])
    interp_y = interp1d(key_idx, key_t[:, 1])
    interp_z = interp1d(key_idx, key_t[:, 2])
    slerp = Slerp(key_idx, key_R)

    poses_list = []
    for idx in all_idx:
        i = int(idx)
        t = np.array([interp_x(idx), interp_y(idx), interp_z(idx)])
        R = slerp(idx).as_matrix()
        pose = np.eye(4)
        pose[:3, :3] = R
        pose[:3, 3] = t
        poses_list.append({
            "timestamp": float(i) / 60.0,
            "rgbIndex": i,
            "depth": None,
            "transform": pose.tolist(),
        })

    return poses_list


def process_session(session_dir: Path, force: bool = False, subsample: int = 6):
    """Full pipeline: COLMAP SfM → interpolate → write poses.json."""
    session_dir = Path(session_dir)

    if not force and not needs_processing(session_dir):
        meta = load_metadata(session_dir)
        source = meta.get("captureSource", "iphone_arkit")
        if source == "iphone_arkit":
            print(f"  {session_dir.name}: ARKit session, skipping")
        else:
            print(f"  {session_dir.name}: already processed, skipping")
        return True

    print(f"  {session_dir.name}: processing...", end=" ", flush=True)

    try:
        sub_poses, recon, n_total, sub_indices = run_colmap_sfm(session_dir, subsample)
    except RuntimeError as e:
        print(f"FAILED — {e}")
        return False

    if len(sub_poses) < 3:
        print(f"FAILED — only {len(sub_poses)} images registered")
        return False

    # Interpolate to 60fps
    poses_list = interpolate_poses(sub_poses, n_total)

    # Write poses.json
    with open(session_dir / "poses.json", "w") as f:
        json.dump(poses_list, f, indent=2)

    # Write calib.txt from COLMAP intrinsics
    for cam_id, cam in recon.cameras.items():
        p = cam.params
        (session_dir / "calib.txt").write_text(
            f"{p[0]:.4f} {p[0]:.4f} {p[1]:.4f} {p[2]:.4f}"
        )
        break

    # Update metadata
    meta = load_metadata(session_dir)
    meta["slamProcessed"] = True
    meta["poseCount"] = len(poses_list)
    with open(session_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    n_keyframes = recon.num_images()
    print(f"OK — {n_keyframes}/{len(sub_indices)} keyframes → {len(poses_list)} poses")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Process capture sessions through COLMAP SfM"
    )
    parser.add_argument(
        "--session", type=str, default=None,
        help="Path to a single session directory",
    )
    parser.add_argument(
        "--data-dir", type=str, default=None,
        help="Process all unprocessed sessions in this directory",
    )
    parser.add_argument(
        "--subsample", type=int, default=6,
        help="Frame subsample rate (default: 6, i.e. 60fps→10fps)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-process even if already processed",
    )
    args = parser.parse_args()

    check_colmap()

    if args.session:
        process_session(Path(args.session), force=args.force, subsample=args.subsample)
    elif args.data_dir:
        data_dir = Path(args.data_dir)
        sessions = sorted([
            d for d in data_dir.iterdir()
            if d.is_dir() and d.name.startswith("session_")
        ])
        print(f"Found {len(sessions)} sessions in {data_dir}")
        ok = 0
        fail = 0
        skip = 0
        for s in sessions:
            meta = load_metadata(s)
            if not args.force and not needs_processing(s):
                skip += 1
                source = meta.get("captureSource", "iphone_arkit")
                if source == "iphone_arkit":
                    print(f"  {s.name}: ARKit, skipping")
                else:
                    print(f"  {s.name}: already processed, skipping")
                continue
            if process_session(s, force=args.force, subsample=args.subsample):
                ok += 1
            else:
                fail += 1
        print(f"\nDone: {ok} processed, {fail} failed, {skip} skipped")
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
