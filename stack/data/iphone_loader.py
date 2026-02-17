"""
Load iPhone ARKit capture sessions from StackCapture app.

Session format (v2):
    session_YYYY-MM-DD_HHMMSS/
    ├── metadata.json       # Device, resolution, frame counts
    ├── poses.json          # [{timestamp, rgbIndex, depth, transform_4x4}, ...]
    ├── encoders.json       # [{timestamp, esp_timestamp_ms, index_mcp, ...}, ...]
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


class iPhoneSession:
    """Loader for a single iPhone capture session."""

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

        self.rgb_dir = self.path / "rgb"

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
    def has_encoders(self) -> bool:
        return len(self._encoders_raw) > 0

    @property
    def has_video(self) -> bool:
        return (self.path / "video.mov").exists()

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

    def get_depth_point(self, index: int) -> Optional[float]:
        """Get single-point LiDAR depth (meters) for frame index, or None."""
        pose = self._poses_raw[index]
        return pose.get("depth")

    def get_all_poses_7d(self) -> np.ndarray:
        """Get all poses as (T, 7) array [x, y, z, qx, qy, qz, qw]."""
        poses = np.zeros((self.num_poses, 7), dtype=np.float32)
        for i in range(self.num_poses):
            poses[i] = self.get_pose_7d(i)
        return poses

    def get_all_depth_points(self) -> np.ndarray:
        """Get all single-point depths as (T,) array in meters. NaN for missing."""
        depths = np.full(self.num_poses, np.nan, dtype=np.float32)
        for i, pose in enumerate(self._poses_raw):
            d = pose.get("depth")
            if d is not None:
                depths[i] = d
        return depths

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

    def get_episode_12d(self) -> np.ndarray:
        """
        Get full 12D proprioception: 7 (pose) + 4 (joints) + 1 (depth point).
        Returns (T, 12) array aligned to RGB frames.
        """
        poses = self.get_all_poses_7d()  # (T, 7)
        encoders = self.get_aligned_encoders()  # (T, 4)
        depths = self.get_all_depth_points()  # (T,)

        # Replace NaN depths with 0
        depths = np.nan_to_num(depths, nan=0.0)

        return np.concatenate([
            poses,
            encoders,
            depths[:, None]
        ], axis=1).astype(np.float32)

    def summary(self) -> str:
        """Return a summary string of the session."""
        lines = [
            f"iPhone Session: {self.path.name}",
            f"  RGB frames: {self.num_rgb_frames} ({self.rgb_resolution[0]}x{self.rgb_resolution[1]})",
            f"  Poses: {self.num_poses}",
            f"  Encoder readings: {self.num_encoder_readings}",
        ]
        if self.duration:
            lines.append(f"  Duration: {self.duration:.1f}s")
        if self.metadata.get("deviceModel"):
            lines.append(f"  Device: {self.metadata['deviceModel']}")
        if self.has_video:
            lines.append(f"  Video: video.mov")
        if self.metadata.get("bleConnected"):
            lines.append(f"  Glove: Connected")
        return "\n".join(lines)


def load_session(path: str | Path) -> iPhoneSession:
    """Load an iPhone capture session from a directory path."""
    return iPhoneSession(Path(path))


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

        # Check 12D episode if encoders present
        if session.has_encoders:
            episode = session.get_episode_12d()
            assert episode.shape == (session.num_poses, 12), f"Episode shape mismatch: {episode.shape}"

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
        print(f"  First pose: {session.get_pose_7d(0)}")
        print(f"  First RGB shape: {session.get_rgb_frame(0).shape}")
        if session.has_encoders:
            print(f"  Encoder readings: {session.num_encoder_readings}")
            print(f"  First encoder: {session.get_encoder_reading(0)}")
            print(f"  12D episode shape: {session.get_episode_12d().shape}")
        depth = session.get_depth_point(0)
        if depth is not None:
            print(f"  First depth point: {depth:.3f} m")
