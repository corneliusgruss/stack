"""
Dataset classes for loading demonstration data.

Supports Zarr format (UMI-FT compatible) and raw session directories.
"""

from pathlib import Path
from typing import Optional, Dict, Any
import json

import numpy as np

try:
    import zarr
    ZARR_AVAILABLE = True
except ImportError:
    ZARR_AVAILABLE = False


class DemoDataset:
    """
    Dataset of manipulation demonstrations.

    Each demonstration contains:
    - RGB images (from iPhone)
    - Depth images (from iPhone LiDAR)
    - Wrist poses (from ARKit)
    - Joint angles (from encoders)
    - Timestamps
    """

    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.episodes: list[Dict[str, Any]] = []
        self._load_episodes()

    def _load_episodes(self):
        """Scan data directory for episodes."""
        if not self.data_dir.exists():
            return

        # Check for Zarr format
        zarr_path = self.data_dir / "dataset.zarr"
        if zarr_path.exists() and ZARR_AVAILABLE:
            self._load_zarr(zarr_path)
            return

        # Fall back to raw session directories
        for session_dir in sorted(self.data_dir.iterdir()):
            if session_dir.is_dir() and not session_dir.name.startswith("."):
                self._load_session(session_dir)

    def _load_zarr(self, zarr_path: Path):
        """Load UMI-FT style Zarr dataset."""
        root = zarr.open(str(zarr_path), mode="r")

        # UMI-FT stores episodes as groups
        for episode_key in root.keys():
            episode = root[episode_key]
            self.episodes.append({
                "id": episode_key,
                "source": "zarr",
                "data": episode,
            })

    def _load_session(self, session_dir: Path):
        """Load raw session directory."""
        metadata_file = session_dir / "metadata.json"
        if metadata_file.exists():
            with open(metadata_file) as f:
                metadata = json.load(f)
        else:
            metadata = {"name": session_dir.name}

        self.episodes.append({
            "id": session_dir.name,
            "source": "raw",
            "path": session_dir,
            "metadata": metadata,
        })

    def __len__(self) -> int:
        return len(self.episodes)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.episodes[idx]

    def get_episode_data(self, idx: int) -> Dict[str, np.ndarray]:
        """
        Load full episode data.

        Returns dict with:
        - images: (T, H, W, 3) RGB
        - depth: (T, H, W) depth
        - poses: (T, 7) wrist pose [x, y, z, qw, qx, qy, qz]
        - joints: (T, 4) encoder angles [idx_mcp, idx_pip, 3f_mcp, 3f_pip]
        - timestamps: (T,) seconds
        """
        episode = self.episodes[idx]

        if episode["source"] == "zarr":
            return self._load_zarr_episode(episode["data"])
        else:
            return self._load_raw_episode(episode["path"])

    def _load_zarr_episode(self, episode_group) -> Dict[str, np.ndarray]:
        """Load episode from Zarr."""
        return {
            "images": np.array(episode_group["rgb"]),
            "depth": np.array(episode_group["depth"]) if "depth" in episode_group else None,
            "poses": np.array(episode_group["poses"]),
            "joints": np.array(episode_group["joints"]) if "joints" in episode_group else None,
            "timestamps": np.array(episode_group["timestamps"]),
        }

    def _load_raw_episode(self, session_dir: Path) -> Dict[str, np.ndarray]:
        """Load episode from raw session directory."""
        # TODO: Implement raw loading
        # This will load from:
        # - session_dir/images/*.jpg
        # - session_dir/poses.csv
        # - session_dir/encoders_*.csv
        raise NotImplementedError("Raw session loading not yet implemented")
