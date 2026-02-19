"""
IMU-based automatic per-session scale calibration for COLMAP poses.

COLMAP SfM produces poses in an arbitrary coordinate system with unknown scale.
This script estimates the metric scale factor by comparing IMU-derived velocity
changes (single-integrated accelerometer) with COLMAP velocity changes.

Algorithm:
1. Load IMU accel (gravity-removed, in g's from CoreMotion) and COLMAP poses
2. Convert accel from g's to m/s² (* 9.81)
3. Estimate accelerometer bias from low-motion periods (gyro magnitude < threshold)
4. For overlapping windows (~1.0s, stride 0.3s):
   - Single-integrate debiased IMU accel -> velocity change (m/s)
   - Differentiate COLMAP positions -> velocity change (COLMAP units/s)
   - If both velocity changes above threshold:
     window_scale = ||imu_delta_v|| / ||colmap_delta_v||
5. Scale = trimmed median (drop outliers outside 2x IQR)
6. Apply: multiply poses.json translation components by scale
7. Store scale_factor in metadata.json

Usage:
    python -m stack.scripts.calibrate_scale --session data/raw/session_2026-02-17_220023
    python -m stack.scripts.calibrate_scale --data-dir data/raw
    python -m stack.scripts.calibrate_scale --data-dir data/raw --verbose
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.interpolate import interp1d

# CoreMotion userAcceleration is in g's, not m/s²
G_TO_MS2 = 9.80665


def estimate_scale_factor(
    imu_timestamps: np.ndarray,
    imu_accel_g: np.ndarray,
    imu_gyro: np.ndarray,
    pose_timestamps: np.ndarray,
    pose_positions: np.ndarray,
    window_duration: float = 1.0,
    window_stride: float = 0.3,
    gyro_still_threshold: float = 0.15,
    min_delta_v_imu: float = 0.02,
    min_delta_v_colmap: float = 0.1,
    verbose: bool = False,
) -> float | None:
    """Estimate metric scale factor from IMU and COLMAP data.

    Uses velocity-change matching: integrate accel over windows to get delta-v
    in metric units, compare with COLMAP-derived delta-v in arbitrary units.

    Args:
        imu_timestamps: (N_imu,) timestamps in seconds (relative to session start)
        imu_accel_g: (N_imu, 3) gravity-removed acceleration in g's (CoreMotion)
        imu_gyro: (N_imu, 3) angular velocity in rad/s
        pose_timestamps: (N_pose,) timestamps in seconds (relative to session start)
        pose_positions: (N_pose, 3) COLMAP positions in arbitrary units
        window_duration: Window length in seconds
        window_stride: Stride between windows in seconds
        gyro_still_threshold: Gyro magnitude threshold for "still" periods (rad/s)
        min_delta_v_imu: Minimum IMU velocity change to consider (m/s)
        min_delta_v_colmap: Minimum COLMAP velocity change to consider (units/s)
        verbose: Print debug info

    Returns:
        Scale factor (multiply COLMAP positions by this to get meters), or None.
    """
    if len(imu_timestamps) < 10 or len(pose_timestamps) < 10:
        return None

    # Convert from g's to m/s²
    imu_accel = imu_accel_g * G_TO_MS2

    # Estimate accelerometer bias from still periods
    gyro_mag = np.linalg.norm(imu_gyro, axis=1)
    still_mask = gyro_mag < gyro_still_threshold
    if still_mask.sum() > 10:
        accel_bias = imu_accel[still_mask].mean(axis=0)
    else:
        accel_bias = imu_accel.mean(axis=0)

    if verbose:
        print(f"    Accel bias (m/s²): {accel_bias}")
        print(f"    Accel RMS (m/s²): {np.sqrt((imu_accel**2).mean(axis=0))}")
        print(f"    Still frames: {still_mask.sum()}/{len(still_mask)}")

    debiased_accel = imu_accel - accel_bias

    # Build COLMAP velocity from position differentiation
    if pose_timestamps[-1] <= pose_timestamps[0]:
        return None

    dt_pose = np.diff(pose_timestamps)
    valid_dt = dt_pose > 1e-6
    colmap_vel = np.zeros((len(pose_timestamps), 3))
    colmap_vel[1:][valid_dt] = np.diff(pose_positions, axis=0)[valid_dt] / dt_pose[valid_dt, None]

    # Smooth COLMAP velocity (5-frame running mean to reduce noise)
    kernel = 5
    for dim in range(3):
        colmap_vel[:, dim] = np.convolve(
            colmap_vel[:, dim], np.ones(kernel) / kernel, mode="same"
        )

    vel_interp = interp1d(
        pose_timestamps, colmap_vel, axis=0,
        kind="linear", bounds_error=False, fill_value=0.0,
    )

    # Sliding window velocity-change matching
    overlap_start = max(imu_timestamps[0], pose_timestamps[0]) + 0.1
    overlap_end = min(imu_timestamps[-1], pose_timestamps[-1]) - 0.1

    if overlap_end - overlap_start < window_duration * 2:
        return None

    window_scales = []
    t = overlap_start
    while t + window_duration <= overlap_end:
        t_start = t
        t_end = t + window_duration

        # IMU: integrate accel over window -> velocity change
        imu_mask = (imu_timestamps >= t_start) & (imu_timestamps <= t_end)
        imu_idx = np.where(imu_mask)[0]
        if len(imu_idx) < 10:
            t += window_stride
            continue

        win_times = imu_timestamps[imu_idx]
        win_accel = debiased_accel[imu_idx]
        dt = np.diff(win_times)

        # Cumulative velocity from integration
        velocity = np.zeros_like(win_accel)
        for i in range(1, len(velocity)):
            velocity[i] = velocity[i - 1] + win_accel[i - 1] * dt[i - 1]

        # IMU delta-v = velocity at end of window
        imu_delta_v = np.linalg.norm(velocity[-1])

        # COLMAP: velocity change over same window
        colmap_v_start = vel_interp(t_start)
        colmap_v_end = vel_interp(t_end)
        colmap_delta_v = np.linalg.norm(colmap_v_end - colmap_v_start)

        if imu_delta_v > min_delta_v_imu and colmap_delta_v > min_delta_v_colmap:
            scale = imu_delta_v / colmap_delta_v
            window_scales.append(scale)

        t += window_stride

    if len(window_scales) < 3:
        if verbose:
            print(f"    Too few valid windows: {len(window_scales)}")
        return None

    window_scales = np.array(window_scales)

    # Trimmed median: remove outliers outside 2x IQR
    q1, q3 = np.percentile(window_scales, [25, 75])
    iqr = q3 - q1
    lower = q1 - 2.0 * iqr
    upper = q3 + 2.0 * iqr
    trimmed = window_scales[(window_scales >= lower) & (window_scales <= upper)]

    if len(trimmed) < 3:
        scale = float(np.median(window_scales))
    else:
        scale = float(np.median(trimmed))

    if verbose:
        print(f"    Valid windows: {len(window_scales)} ({len(trimmed)} after trim)")
        print(f"    Scale estimates: median={scale:.6f}, mean={trimmed.mean():.6f}, "
              f"std={trimmed.std():.6f}")
        print(f"    Range: [{trimmed.min():.6f}, {trimmed.max():.6f}]")

    return scale


def calibrate_session(session_dir: Path, verbose: bool = False) -> float | None:
    """Calibrate a single session's COLMAP poses to metric scale.

    Returns the scale factor, or None if calibration failed.
    """
    session_dir = Path(session_dir)

    # Load metadata
    meta_file = session_dir / "metadata.json"
    if not meta_file.exists():
        print(f"  {session_dir.name}: no metadata.json, skipping")
        return None

    with open(meta_file) as f:
        metadata = json.load(f)

    # Check if already calibrated
    if metadata.get("scale_factor") is not None:
        if verbose:
            print(f"  {session_dir.name}: already calibrated (scale={metadata['scale_factor']:.6f})")
        return metadata["scale_factor"]

    # Load IMU data
    imu_file = session_dir / "imu.json"
    if not imu_file.exists():
        print(f"  {session_dir.name}: no imu.json, skipping")
        return None

    with open(imu_file) as f:
        imu_raw = json.load(f)

    if len(imu_raw) < 20:
        print(f"  {session_dir.name}: too few IMU readings ({len(imu_raw)}), skipping")
        return None

    # Load poses
    poses_file = session_dir / "poses.json"
    if not poses_file.exists():
        print(f"  {session_dir.name}: no poses.json, skipping")
        return None

    with open(poses_file) as f:
        poses_raw = json.load(f)

    if len(poses_raw) < 20:
        print(f"  {session_dir.name}: too few poses ({len(poses_raw)}), skipping")
        return None

    # Extract arrays
    imu_timestamps = np.array([r["timestamp"] for r in imu_raw], dtype=np.float64)
    imu_accel = np.array([r["accel"] for r in imu_raw], dtype=np.float64)
    imu_gyro = np.array([r["gyro"] for r in imu_raw], dtype=np.float64)

    pose_timestamps = np.array([p["timestamp"] for p in poses_raw], dtype=np.float64)
    pose_positions = np.array([
        np.array(p["transform"], dtype=np.float64).reshape(4, 4)[:3, 3]
        for p in poses_raw
    ])

    # Normalize timestamps to start from 0
    t0_imu = imu_timestamps[0]
    t0_pose = pose_timestamps[0]
    imu_timestamps_rel = imu_timestamps - t0_imu
    pose_timestamps_rel = pose_timestamps - t0_pose

    if verbose:
        print(f"  {session_dir.name}:")
        print(f"    IMU: {len(imu_raw)} readings, {imu_timestamps_rel[-1]:.1f}s")
        print(f"    Poses: {len(poses_raw)} frames, {pose_timestamps_rel[-1]:.1f}s")
        pos_span = pose_positions.max(axis=0) - pose_positions.min(axis=0)
        print(f"    Position span (COLMAP): {pos_span}")

    scale = estimate_scale_factor(
        imu_timestamps_rel, imu_accel, imu_gyro,
        pose_timestamps_rel, pose_positions,
        verbose=verbose,
    )

    if scale is None:
        print(f"  {session_dir.name}: calibration failed")
        return None

    # Apply scale to poses
    for pose_entry in poses_raw:
        transform = np.array(pose_entry["transform"], dtype=np.float64).reshape(4, 4)
        transform[:3, 3] *= scale
        pose_entry["transform"] = transform.tolist()

    # Write updated poses
    with open(poses_file, "w") as f:
        json.dump(poses_raw, f, indent=2)

    # Update metadata
    metadata["scale_factor"] = scale
    with open(meta_file, "w") as f:
        json.dump(metadata, f, indent=2)

    pos_span_m = (pose_positions.max(axis=0) - pose_positions.min(axis=0)) * scale
    total_travel = np.sum(np.linalg.norm(np.diff(pose_positions, axis=0), axis=1)) * scale
    print(f"  {session_dir.name}: scale={scale:.6f}, "
          f"workspace={np.linalg.norm(pos_span_m):.2f}m, "
          f"travel={total_travel:.2f}m")

    return scale


def main():
    parser = argparse.ArgumentParser(
        description="IMU-based scale calibration for COLMAP poses"
    )
    parser.add_argument(
        "--session", type=str, default=None,
        help="Path to a single session directory",
    )
    parser.add_argument(
        "--data-dir", type=str, default=None,
        help="Calibrate all sessions in this directory",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print detailed calibration info",
    )
    args = parser.parse_args()

    if args.session:
        calibrate_session(Path(args.session), verbose=args.verbose)
    elif args.data_dir:
        data_dir = Path(args.data_dir)
        sessions = sorted([
            d for d in data_dir.iterdir()
            if d.is_dir() and d.name.startswith("session_") and (d / "poses.json").exists()
        ])
        print(f"Found {len(sessions)} sessions in {data_dir}")
        ok = 0
        fail = 0
        skip = 0
        for s in sessions:
            result = calibrate_session(s, verbose=args.verbose)
            if result is not None:
                ok += 1
            else:
                # Check if it was a skip (already calibrated or no IMU)
                meta_file = s / "metadata.json"
                if meta_file.exists():
                    with open(meta_file) as f:
                        meta = json.load(f)
                    if meta.get("scale_factor") is not None:
                        skip += 1
                        continue
                fail += 1
        print(f"\nDone: {ok} calibrated, {fail} failed, {skip} already calibrated")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
