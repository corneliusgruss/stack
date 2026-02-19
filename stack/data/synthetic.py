"""
Generate synthetic demonstration sessions for pipeline testing.

Creates sessions in the same format as StackCapture (iPhoneSession-compatible),
so the full training pipeline can be tested without real hardware data.

Usage:
    python -m stack.data.synthetic --output data/raw/synthetic --num-sessions 10
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation, Slerp


def generate_smooth_trajectory(num_frames: int, rng: np.random.Generator) -> np.ndarray:
    """Generate smooth 7D pose trajectory [x, y, z, qx, qy, qz, qw].

    Uses sinusoidal position and slerp-interpolated rotations.
    """
    t = np.linspace(0, 2 * np.pi, num_frames)

    # Smooth sinusoidal positions with random frequencies and amplitudes
    freqs = rng.uniform(0.5, 2.0, size=3)
    phases = rng.uniform(0, 2 * np.pi, size=3)
    amps = rng.uniform(0.05, 0.3, size=3)
    offsets = rng.uniform(-0.5, 0.5, size=3)

    positions = np.zeros((num_frames, 3), dtype=np.float32)
    for dim in range(3):
        positions[:, dim] = offsets[dim] + amps[dim] * np.sin(freqs[dim] * t + phases[dim])

    # Smooth rotations via slerp between random keypoints
    num_keypoints = 5
    key_times = np.linspace(0, 1, num_keypoints)
    key_rotations = Rotation.random(num_keypoints, random_state=rng.integers(0, 2**31))
    slerp = Slerp(key_times, key_rotations)

    interp_times = np.linspace(0, 1, num_frames)
    rotations = slerp(interp_times)
    quats = rotations.as_quat()  # (N, 4) as [x, y, z, w]

    return np.concatenate([positions, quats.astype(np.float32)], axis=1)


def generate_smooth_joints(num_frames: int, rng: np.random.Generator) -> np.ndarray:
    """Generate smooth 4D joint angle trajectories (degrees).

    Returns (num_frames, 4) array for [index_mcp, index_pip, three_finger_mcp, three_finger_pip].
    """
    t = np.linspace(0, 2 * np.pi, num_frames)
    joints = np.zeros((num_frames, 4), dtype=np.float32)

    for j in range(4):
        freq = rng.uniform(0.3, 1.5)
        phase = rng.uniform(0, 2 * np.pi)
        amp = rng.uniform(10, 45)  # degrees
        offset = rng.uniform(20, 90)  # base angle
        joints[:, j] = offset + amp * np.sin(freq * t + phase)

    return joints


def generate_synthetic_session(
    session_dir: Path,
    num_frames: int = 300,
    seed: int | None = None,
    fps: float = 60.0,
    image_size: tuple[int, int] = (480, 360),
) -> Path:
    """Generate a single synthetic session in StackCapture v2 format.

    Args:
        session_dir: Where to write the session.
        num_frames: Number of frames to generate.
        seed: Random seed for reproducibility.
        fps: Simulated frame rate.
        image_size: (width, height) of generated images.

    Returns:
        Path to the created session directory.
    """
    rng = np.random.default_rng(seed)
    session_dir = Path(session_dir)
    rgb_dir = session_dir / "rgb"
    rgb_dir.mkdir(parents=True, exist_ok=True)

    # Generate trajectories
    poses_7d = generate_smooth_trajectory(num_frames, rng)
    joints = generate_smooth_joints(num_frames, rng)

    # Timestamps (simulated 60 FPS)
    dt = 1.0 / fps
    base_timestamp = 1000000.0  # Arbitrary ARKit-style boot time
    base_epoch = 1739000000.0   # Arbitrary epoch time for encoders
    rgb_timestamps = base_timestamp + np.arange(num_frames) * dt
    enc_timestamps = base_epoch + np.arange(num_frames) * dt

    # Build poses.json
    poses_list = []
    for i in range(num_frames):
        # Build 4x4 transform from pose_7d
        pos = poses_7d[i, :3]
        quat = poses_7d[i, 3:]  # [qx, qy, qz, qw]
        rot = Rotation.from_quat(quat)
        transform = np.eye(4, dtype=np.float32)
        transform[:3, :3] = rot.as_matrix()
        transform[:3, 3] = pos

        poses_list.append({
            "timestamp": float(rgb_timestamps[i]),
            "rgbIndex": i,
            "depth": None,
            "transform": transform.tolist(),
        })

    # Build encoders.json
    encoders_list = []
    for i in range(num_frames):
        encoders_list.append({
            "timestamp": float(enc_timestamps[i]),
            "esp_timestamp_ms": int(i * dt * 1000),
            "index_mcp": float(joints[i, 0]),
            "index_pip": float(joints[i, 1]),
            "three_finger_mcp": float(joints[i, 2]),
            "three_finger_pip": float(joints[i, 3]),
        })

    # Metadata
    width, height = image_size
    duration = num_frames / fps
    metadata = {
        "deviceModel": "synthetic",
        "rgbResolution": [width, height],
        "frameCount": num_frames,
        "poseCount": num_frames,
        "encoderCount": num_frames,
        "durationSeconds": duration,
        "bleConnected": True,
        "synthetic": True,
        "seed": seed,
    }

    # Write JSON files
    with open(session_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    with open(session_dir / "poses.json", "w") as f:
        json.dump(poses_list, f)
    with open(session_dir / "encoders.json", "w") as f:
        json.dump(encoders_list, f)

    # Generate simple colored images (fast — no need for realistic images)
    base_color = rng.integers(50, 200, size=3).astype(np.uint8)
    for i in range(num_frames):
        # Slight color variation per frame for visual debugging
        color = np.clip(base_color.astype(np.int16) + rng.integers(-10, 10, size=3), 0, 255).astype(np.uint8)
        img = np.full((height, width, 3), color, dtype=np.uint8)
        Image.fromarray(img).save(rgb_dir / f"{i:06d}.jpg", quality=80)

    return session_dir


def generate_synthetic_dataset(
    data_dir: Path,
    num_sessions: int = 10,
    frames_per_session: int = 300,
    seed: int = 42,
) -> list[Path]:
    """Generate multiple synthetic sessions for training.

    Args:
        data_dir: Root directory for generated sessions.
        num_sessions: Number of sessions to generate.
        frames_per_session: Frames per session.
        seed: Base random seed.

    Returns:
        List of created session directory paths.
    """
    data_dir = Path(data_dir)
    sessions = []

    for i in range(num_sessions):
        session_name = f"session_synthetic_{i:03d}"
        session_dir = data_dir / session_name
        print(f"Generating {session_name} ({frames_per_session} frames)...")
        generate_synthetic_session(
            session_dir,
            num_frames=frames_per_session,
            seed=seed + i,
        )
        sessions.append(session_dir)

    print(f"\nGenerated {num_sessions} sessions in {data_dir}")
    return sessions


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic demo sessions")
    parser.add_argument("--output", default="data/raw/synthetic", help="Output directory")
    parser.add_argument("--num-sessions", type=int, default=10, help="Number of sessions")
    parser.add_argument("--frames", type=int, default=300, help="Frames per session")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    generate_synthetic_dataset(
        Path(args.output),
        num_sessions=args.num_sessions,
        frames_per_session=args.frames,
        seed=args.seed,
    )
