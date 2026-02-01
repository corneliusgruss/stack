"""
Load iPhone ARKit capture sessions from StackCapture app.

Session format:
    session_YYYY-MM-DD_HHMMSS/
    ├── metadata.json       # Device, resolution, frame counts
    ├── poses.json          # [{timestamp, rgbIndex, transform_4x4}, ...]
    ├── rgb/
    │   ├── 000000.jpg      # 1920x1440 @ quality 0.85
    │   └── ...
    └── depth/
        ├── 000000.bin      # 256x192 Float16 (raw meters)
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

        self.rgb_dir = self.path / "rgb"
        self.depth_dir = self.path / "depth"

    @property
    def num_rgb_frames(self) -> int:
        return self.metadata.get("rgbFrameCount", len(list(self.rgb_dir.glob("*.jpg"))))

    @property
    def num_depth_frames(self) -> int:
        return self.metadata.get("depthFrameCount", len(list(self.depth_dir.glob("*.bin"))))

    @property
    def num_poses(self) -> int:
        return len(self._poses_raw)

    @property
    def rgb_resolution(self) -> Tuple[int, int]:
        """Returns (width, height)."""
        res = self.metadata.get("rgbResolution", [1920, 1440])
        return tuple(res)

    @property
    def depth_resolution(self) -> Tuple[int, int]:
        """Returns (width, height)."""
        res = self.metadata.get("depthResolution", [256, 192])
        return tuple(res)

    @property
    def duration(self) -> Optional[float]:
        """Duration in seconds."""
        return self.metadata.get("durationSeconds")

    def get_timestamps(self) -> np.ndarray:
        """Get all timestamps (T,)."""
        return np.array([p["timestamp"] for p in self._poses_raw], dtype=np.float64)

    def get_rgb_frame(self, index: int) -> np.ndarray:
        """Load single RGB frame as (H, W, 3) uint8."""
        filename = f"{index:06d}.jpg"
        filepath = self.rgb_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"RGB frame not found: {filepath}")
        img = Image.open(filepath)
        return np.array(img)

    def get_depth_frame(self, index: int) -> np.ndarray:
        """Load single depth frame as (H, W) float32 in meters."""
        filename = f"{index:06d}.bin"
        filepath = self.depth_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"Depth frame not found: {filepath}")

        width, height = self.depth_resolution
        # ARKit depth is Float32 (meters)
        depth = np.fromfile(filepath, dtype=np.float32).reshape(height, width)
        return depth

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

    def load_all_depth(self, max_frames: Optional[int] = None) -> np.ndarray:
        """
        Load all depth frames as (T, H, W) float32 in meters.

        Args:
            max_frames: Maximum number of frames to load (None for all)
        """
        n = min(self.num_depth_frames, max_frames) if max_frames else self.num_depth_frames
        width, height = self.depth_resolution

        depth = np.zeros((n, height, width), dtype=np.float32)
        for i in range(n):
            try:
                depth[i] = self.get_depth_frame(i)
            except FileNotFoundError:
                pass  # Missing depth frames are kept as zeros
        return depth

    def get_depth_index_for_rgb(self, rgb_index: int) -> Optional[int]:
        """Get the depth frame index corresponding to an RGB frame."""
        pose = self._poses_raw[rgb_index]
        return pose.get("depthIndex")

    def summary(self) -> str:
        """Return a summary string of the session."""
        lines = [
            f"iPhone Session: {self.path.name}",
            f"  RGB frames: {self.num_rgb_frames} ({self.rgb_resolution[0]}x{self.rgb_resolution[1]})",
            f"  Depth frames: {self.num_depth_frames} ({self.depth_resolution[0]}x{self.depth_resolution[1]})",
            f"  Poses: {self.num_poses}",
        ]
        if self.duration:
            lines.append(f"  Duration: {self.duration:.1f}s")
        if self.metadata.get("deviceModel"):
            lines.append(f"  Device: {self.metadata['deviceModel']}")
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

        # Check basic counts match
        assert session.num_poses > 0, "No poses"
        assert session.num_rgb_frames > 0, "No RGB frames"
        assert session.num_poses == session.num_rgb_frames, "Pose/RGB count mismatch"

        # Load first frame
        img = session.get_rgb_frame(0)
        assert img.shape == (session.rgb_resolution[1], session.rgb_resolution[0], 3), "RGB shape mismatch"

        # Load first depth (if available)
        if session.num_depth_frames > 0:
            depth = session.get_depth_frame(0)
            assert depth.shape == (session.depth_resolution[1], session.depth_resolution[0]), "Depth shape mismatch"

        # Check poses
        pose = session.get_pose_7d(0)
        assert pose.shape == (7,), "Pose shape mismatch"
        assert not np.isnan(pose).any(), "Pose contains NaN"

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
        if session.num_depth_frames > 0:
            print(f"  First depth range: {session.get_depth_frame(0).min():.2f} - {session.get_depth_frame(0).max():.2f} m")
