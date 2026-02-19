"""
Load capture sessions from StackCapture app.

Supports both ARKit sessions (poses from Apple) and raw/ultrawide sessions
(poses from COLMAP SfM offline processing).

Episode format: 11D = pose (7) + joints (4)

Session format (v3 — camera-agnostic):
    session_YYYY-MM-DD_HHMMSS/
    ├── metadata.json       # Device, resolution, frame counts, captureSource
    ├── poses.json          # [{timestamp, rgbIndex, transform_4x4}, ...]
    ├── encoders.json       # [{timestamp, esp_timestamp_ms, index_mcp, ...}, ...]
    ├── imu.json            # [{timestamp, accel:[x,y,z], gyro:[x,y,z]}, ...]
    ├── calib.txt           # "fx fy cx cy" — camera intrinsics (for SLAM)
    ├── video.mov           # Full-resolution HEVC video
    └── rgb/
        ├── 000000.jpg      # 480x360 @ quality 0.8
        └── ...
"""

from pathlib import Path
from typing import Dict, Optional, Tuple

import json

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation


class SessionLoader:
    """Loader for a single capture session (any source)."""

    def __init__(self, session_path: Path):
        self.path = Path(session_path)

        if not self.path.exists():
            raise FileNotFoundError(f"Session not found: {self.path}")

        # Load metadata
        metadata_file = self.path / "metadata.json"
        if metadata_file.exists():
            with open(metadata_file) as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {}

        # Determine capture source
        self._capture_source = self.metadata.get("captureSource", "iphone_arkit")
        self._slam_processed = self.metadata.get("slamProcessed", None)

        # Validate: non-ARKit sessions need SLAM processing for poses
        if self._capture_source != "iphone_arkit" and not self._slam_processed:
            poses_file = self.path / "poses.json"
            if poses_file.exists():
                with open(poses_file) as f:
                    poses_data = json.load(f)
                if len(poses_data) == 0:
                    raise ValueError(
                        f"Session {self.path.name} was captured with {self._capture_source} "
                        f"but has not been processed through SLAM yet.\n"
                        f"Run: stack-slam --session {self.path}\n"
                        f"Or use notebooks/run_slam.ipynb on Colab."
                    )

        # Load poses (sort by rgbIndex in case of out-of-order async writes)
        poses_file = self.path / "poses.json"
        if poses_file.exists():
            with open(poses_file) as f:
                poses_unsorted = json.load(f)
            self._poses_raw = sorted(poses_unsorted, key=lambda p: p["rgbIndex"])
        else:
            self._poses_raw = []

        # Load encoder readings
        encoders_file = self.path / "encoders.json"
        if encoders_file.exists():
            with open(encoders_file) as f:
                self._encoders_raw = json.load(f)
        else:
            self._encoders_raw = []

        # Load IMU readings (if present — ultrawide/raw mode)
        imu_file = self.path / "imu.json"
        if imu_file.exists():
            with open(imu_file) as f:
                self._imu_raw = json.load(f)
        else:
            self._imu_raw = []

        # Load camera intrinsics (if present)
        calib_file = self.path / "calib.txt"
        if calib_file.exists():
            parts = calib_file.read_text().strip().split()
            self._intrinsics = np.array([float(x) for x in parts[:4]], dtype=np.float32)
        else:
            self._intrinsics = None

        self.rgb_dir = self.path / "rgb"

    @property
    def capture_source(self) -> str:
        return self._capture_source

    @property
    def slam_processed(self) -> Optional[bool]:
        return self._slam_processed

    @property
    def num_rgb_frames(self) -> int:
        """Count actual files on disk (metadata count can lag due to async flush)."""
        return len(list(self.rgb_dir.glob("*.jpg")))

    @property
    def num_poses(self) -> int:
        return len(self._poses_raw)

    @property
    def num_encoder_readings(self) -> int:
        return len(self._encoders_raw)

    @property
    def num_imu_readings(self) -> int:
        return len(self._imu_raw)

    @property
    def has_encoders(self) -> bool:
        return len(self._encoders_raw) > 0

    @property
    def has_imu(self) -> bool:
        return len(self._imu_raw) > 0

    @property
    def has_video(self) -> bool:
        return (self.path / "video.mov").exists()

    @property
    def has_intrinsics(self) -> bool:
        return self._intrinsics is not None

    @property
    def intrinsics(self) -> Optional[np.ndarray]:
        """Camera intrinsics [fx, fy, cx, cy] or None."""
        return self._intrinsics

    @property
    def rgb_resolution(self) -> Tuple[int, int]:
        """Returns (width, height)."""
        res = self.metadata.get("rgbResolution", [480, 360])
        return tuple(res)

    @property
    def duration(self) -> Optional[float]:
        """Duration in seconds."""
        return self.metadata.get("durationSeconds")

    def get_timestamps(self) -> np.ndarray:
        """Get all pose timestamps (T,)."""
        return np.array([p["timestamp"] for p in self._poses_raw], dtype=np.float64)

    def get_rgb_frame(self, index: int) -> np.ndarray:
        """Load single RGB frame as (H, W, 3) uint8."""
        filename = f"{index:06d}.jpg"
        filepath = self.rgb_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"RGB frame not found: {filepath}")
        img = Image.open(filepath)
        return np.array(img)

    def get_pose_transform(self, index: int) -> np.ndarray:
        """Get 4x4 transform matrix for frame index."""
        pose = self._poses_raw[index]
        transform = np.array(pose["transform"], dtype=np.float32)
        return transform

    def get_pose_7d(self, index: int) -> np.ndarray:
        """
        Get 7D pose [x, y, z, qx, qy, qz, qw] for frame index.

        Note: Quaternion is in scipy/PyTorch convention (xyzw), not ARKit (wxyz).
        """
        transform = self.get_pose_transform(index)
        position = transform[:3, 3]
        rotation_matrix = transform[:3, :3]
        rotation = Rotation.from_matrix(rotation_matrix)
        quat = rotation.as_quat()  # Returns [x, y, z, w]
        return np.concatenate([position, quat]).astype(np.float32)

    def get_all_poses_7d(self) -> np.ndarray:
        """Get all poses as (T, 7) array [x, y, z, qx, qy, qz, qw]."""
        poses = np.zeros((self.num_poses, 7), dtype=np.float32)
        for i in range(self.num_poses):
            poses[i] = self.get_pose_7d(i)
        return poses

    def get_encoder_timestamps(self) -> np.ndarray:
        """Get encoder timestamps (T_enc,) in seconds since epoch."""
        return np.array([e["timestamp"] for e in self._encoders_raw], dtype=np.float64)

    def get_encoder_reading(self, index: int) -> np.ndarray:
        """Get single encoder reading as [index_mcp, index_pip, three_finger_mcp, three_finger_pip]."""
        e = self._encoders_raw[index]
        return np.array([
            e["index_mcp"], e["index_pip"],
            e["three_finger_mcp"], e["three_finger_pip"]
        ], dtype=np.float32)

    def get_all_encoders(self) -> np.ndarray:
        """Get all encoder readings as (T_enc, 4) array."""
        if not self._encoders_raw:
            return np.zeros((0, 4), dtype=np.float32)
        readings = np.zeros((len(self._encoders_raw), 4), dtype=np.float32)
        for i, e in enumerate(self._encoders_raw):
            readings[i] = [
                e["index_mcp"], e["index_pip"],
                e["three_finger_mcp"], e["three_finger_pip"]
            ]
        return readings

    def get_aligned_encoders(self) -> np.ndarray:
        """
        Align encoder readings to RGB frame timestamps using nearest-neighbor matching.
        Returns (T_rgb, 4) array. Frames without a matching encoder reading get zeros.
        """
        if not self._encoders_raw:
            return np.zeros((self.num_poses, 4), dtype=np.float32)

        rgb_timestamps = self.get_timestamps()  # ARKit timestamps (seconds since boot)
        enc_timestamps = self.get_encoder_timestamps()  # iPhone Date timestamps (epoch)
        enc_values = self.get_all_encoders()

        # ARKit timestamps are time since device boot, encoder timestamps are epoch.
        # Align by matching relative offsets: both start near recording start.
        # Use the offset between first pose timestamp and first encoder timestamp.
        if len(rgb_timestamps) == 0 or len(enc_timestamps) == 0:
            return np.zeros((self.num_poses, 4), dtype=np.float32)

        # Normalize both to start from 0
        rgb_rel = rgb_timestamps - rgb_timestamps[0]
        enc_rel = enc_timestamps - enc_timestamps[0]

        aligned = np.zeros((len(rgb_timestamps), 4), dtype=np.float32)
        for i, t in enumerate(rgb_rel):
            # Find nearest encoder reading
            idx = np.argmin(np.abs(enc_rel - t))
            # Only use if within 50ms
            if abs(enc_rel[idx] - t) < 0.05:
                aligned[i] = enc_values[idx]

        return aligned

    def get_imu_data(self) -> Optional[dict]:
        """Get IMU data as dict with 'timestamps', 'accel', 'gyro' arrays.

        Returns None if no IMU data.
        """
        if not self._imu_raw:
            return None

        n = len(self._imu_raw)
        timestamps = np.zeros(n, dtype=np.float64)
        accel = np.zeros((n, 3), dtype=np.float64)
        gyro = np.zeros((n, 3), dtype=np.float64)

        for i, reading in enumerate(self._imu_raw):
            timestamps[i] = reading["timestamp"]
            accel[i] = reading["accel"]
            gyro[i] = reading["gyro"]

        return {
            "timestamps": timestamps,
            "accel": accel,
            "gyro": gyro,
        }

    def load_all_rgb(self, max_frames: Optional[int] = None) -> np.ndarray:
        """
        Load all RGB frames as (T, H, W, 3) uint8.

        Args:
            max_frames: Maximum number of frames to load (None for all)
        """
        n = min(self.num_rgb_frames, max_frames) if max_frames else self.num_rgb_frames
        width, height = self.rgb_resolution

        images = np.zeros((n, height, width, 3), dtype=np.uint8)
        for i in range(n):
            images[i] = self.get_rgb_frame(i)
        return images

    def get_episode_11d(self) -> np.ndarray:
        """
        Get full 11D proprioception: 7 (pose) + 4 (joints).
        Returns (T, 11) array aligned to RGB frames.
        """
        poses = self.get_all_poses_7d()  # (T, 7)
        encoders = self.get_aligned_encoders()  # (T, 4)

        return np.concatenate([poses, encoders], axis=1).astype(np.float32)

    def get_episode_12d(self) -> np.ndarray:
        """Deprecated: use get_episode_11d(). Depth dimension was always zero."""
        import warnings
        warnings.warn(
            "get_episode_12d() is deprecated, use get_episode_11d(). "
            "Depth dimension has been removed.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.get_episode_11d()

    def summary(self) -> str:
        """Return a summary string of the session."""
        lines = [
            f"Session: {self.path.name}",
            f"  Source: {self._capture_source}",
            f"  RGB frames: {self.num_rgb_frames} ({self.rgb_resolution[0]}x{self.rgb_resolution[1]})",
            f"  Poses: {self.num_poses}",
            f"  Encoder readings: {self.num_encoder_readings}",
        ]
        if self.has_imu:
            lines.append(f"  IMU readings: {self.num_imu_readings}")
        if self.has_intrinsics:
            lines.append(f"  Intrinsics: fx={self._intrinsics[0]:.1f} fy={self._intrinsics[1]:.1f}")
        if self.duration:
            lines.append(f"  Duration: {self.duration:.1f}s")
        if self.metadata.get("deviceModel"):
            lines.append(f"  Device: {self.metadata['deviceModel']}")
        if self.has_video:
            lines.append(f"  Video: video.mov")
        if self.metadata.get("bleConnected"):
            lines.append(f"  Glove: Connected")
        if self._slam_processed is not None:
            lines.append(f"  SLAM processed: {self._slam_processed}")
        return "\n".join(lines)


# Backward compatibility alias
iPhoneSession = SessionLoader


def load_session(path: str | Path) -> SessionLoader:
    """Load a capture session from a directory path."""
    return SessionLoader(Path(path))


def verify_session(path: str | Path) -> bool:
    """
    Verify that a session was captured correctly.

    Returns True if session is valid.
    """
    try:
        session = load_session(path)

        # Check basic counts
        assert session.num_poses > 0, "No poses"
        assert session.num_rgb_frames > 0, "No RGB frames"
        # Async flush can cause slight mismatch — warn but don't fail
        if session.num_poses != session.num_rgb_frames:
            usable = min(session.num_poses, session.num_rgb_frames)
            print(f"  Note: {session.num_poses} poses vs {session.num_rgb_frames} RGB files "
                  f"(using {usable} frames, async flush mismatch)")

        # Load first frame
        img = session.get_rgb_frame(0)
        assert img.shape == (session.rgb_resolution[1], session.rgb_resolution[0], 3), "RGB shape mismatch"

        # Check poses
        pose = session.get_pose_7d(0)
        assert pose.shape == (7,), "Pose shape mismatch"
        assert not np.isnan(pose).any(), "Pose contains NaN"

        # Check 11D episode if encoders present
        if session.has_encoders:
            episode = session.get_episode_11d()
            assert episode.shape == (session.num_poses, 11), f"Episode shape mismatch: {episode.shape}"

        print(f"Session verified: {path}")
        print(session.summary())
        return True

    except Exception as e:
        print(f"Session verification failed: {e}")
        return False


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m stack.data.iphone_loader <session_path>")
        sys.exit(1)

    session_path = Path(sys.argv[1])
    if verify_session(session_path):
        session = load_session(session_path)
        print("\nSample data:")
        print(f"  Source: {session.capture_source}")
        print(f"  First pose: {session.get_pose_7d(0)}")
        print(f"  First RGB shape: {session.get_rgb_frame(0).shape}")
        if session.has_encoders:
            print(f"  Encoder readings: {session.num_encoder_readings}")
            print(f"  First encoder: {session.get_encoder_reading(0)}")
            print(f"  11D episode shape: {session.get_episode_11d().shape}")
        if session.has_imu:
            imu = session.get_imu_data()
            print(f"  IMU readings: {session.num_imu_readings}")
            print(f"  IMU rate: {1.0 / np.mean(np.diff(imu['timestamps'])):.0f} Hz")
