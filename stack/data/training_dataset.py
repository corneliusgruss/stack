"""
PyTorch Dataset for diffusion policy training.

Loads iPhoneSession data and produces (obs_images, obs_proprio, action_chunk)
training samples using sliding window sampling.

Observation: 11D = pose (7) + joints (4)
Action: 11D = pose (7) + joints (4)
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageEnhance
import random

from stack.data.iphone_loader import iPhoneSession


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@dataclass
class NormalizationStats:
    """Per-dimension normalization statistics computed from training data."""
    proprio_mean: np.ndarray   # (11,)
    proprio_std: np.ndarray    # (11,)
    action_min: np.ndarray     # (11,)
    action_max: np.ndarray     # (11,)

    def normalize_proprio(self, proprio: np.ndarray) -> np.ndarray:
        """Zero-mean unit-variance normalization."""
        std = np.where(self.proprio_std < 1e-6, 1.0, self.proprio_std)
        return (proprio - self.proprio_mean) / std

    def normalize_action(self, action: np.ndarray) -> np.ndarray:
        """Normalize to [-1, 1]."""
        range_ = self.action_max - self.action_min
        range_ = np.where(range_ < 1e-6, 1.0, range_)
        return 2.0 * (action - self.action_min) / range_ - 1.0

    def unnormalize_action(self, action: np.ndarray) -> np.ndarray:
        """Inverse of normalize_action."""
        range_ = self.action_max - self.action_min
        range_ = np.where(range_ < 1e-6, 1.0, range_)
        return (action + 1.0) / 2.0 * range_ + self.action_min

    def state_dict(self) -> dict:
        return {
            "proprio_mean": self.proprio_mean,
            "proprio_std": self.proprio_std,
            "action_min": self.action_min,
            "action_max": self.action_max,
        }

    @classmethod
    def from_state_dict(cls, d: dict) -> "NormalizationStats":
        return cls(
            proprio_mean=np.array(d["proprio_mean"], dtype=np.float32),
            proprio_std=np.array(d["proprio_std"], dtype=np.float32),
            action_min=np.array(d["action_min"], dtype=np.float32),
            action_max=np.array(d["action_max"], dtype=np.float32),
        )


def compute_normalization_stats(sessions: list[iPhoneSession]) -> NormalizationStats:
    """Compute normalization statistics across all sessions."""
    all_proprio = []
    all_actions = []

    for session in sessions:
        episode = session.get_episode_11d()  # (T, 11)
        all_proprio.append(episode)
        all_actions.append(episode)  # actions = proprio (both 11D)

    all_proprio = np.concatenate(all_proprio, axis=0)
    all_actions = np.concatenate(all_actions, axis=0)

    return NormalizationStats(
        proprio_mean=all_proprio.mean(axis=0).astype(np.float32),
        proprio_std=all_proprio.std(axis=0).astype(np.float32),
        action_min=all_actions.min(axis=0).astype(np.float32),
        action_max=all_actions.max(axis=0).astype(np.float32),
    )


class StackDiffusionDataset(Dataset):
    """
    PyTorch Dataset for diffusion policy training.

    Each sample contains:
    - obs_images: (obs_horizon, 3, H, W) — normalized RGB images
    - obs_proprio: (obs_horizon, 11) — normalized proprioception
    - action_chunk: (action_horizon, 11) — normalized action targets

    Uses sliding window sampling across episodes. No cross-episode windows.
    """

    def __init__(
        self,
        session_dirs: list[Path],
        obs_horizon: int = 2,
        action_horizon: int = 16,
        image_size: int = 224,
        stats: NormalizationStats | None = None,
        augment: bool = False,
        random_crop: bool = False,
        color_jitter: bool = False,
    ):
        self.obs_horizon = obs_horizon
        self.action_horizon = action_horizon
        self.image_size = image_size
        self.augment = augment
        self.random_crop = random_crop and augment
        self.color_jitter = color_jitter and augment
        self._crop_size = int(image_size * 1.07)

        # Load all sessions
        self.sessions: list[iPhoneSession] = []
        for d in session_dirs:
            try:
                self.sessions.append(iPhoneSession(d))
            except Exception as e:
                print(f"Warning: skipping {d}: {e}")

        # Preload episode data (11D proprio per frame)
        self._episodes: list[np.ndarray] = []
        for s in self.sessions:
            self._episodes.append(s.get_episode_11d())

        # Compute or use provided normalization stats
        if stats is not None:
            self.stats = stats
        else:
            self.stats = compute_normalization_stats(self.sessions)

        # Build index: list of (episode_idx, frame_idx) for valid windows
        self._samples: list[tuple[int, int]] = []
        for ep_idx, episode in enumerate(self._episodes):
            T = len(episode)
            # Need obs_horizon frames before and action_horizon frames after
            # Window starts at frame_idx, obs = [frame_idx - obs_horizon + 1, frame_idx]
            # actions = [frame_idx + 1, frame_idx + action_horizon]
            for t in range(self.obs_horizon - 1, T - self.action_horizon):
                self._samples.append((ep_idx, t))

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ep_idx, t = self._samples[idx]
        session = self.sessions[ep_idx]
        episode = self._episodes[ep_idx]

        # Observation window: [t - obs_horizon + 1, t] inclusive
        obs_start = t - self.obs_horizon + 1

        # Load and preprocess images
        obs_images = np.zeros(
            (self.obs_horizon, 3, self.image_size, self.image_size),
            dtype=np.float32,
        )
        for i in range(self.obs_horizon):
            frame_idx = obs_start + i
            img = session.get_rgb_frame(frame_idx)  # (H, W, 3) uint8
            img = self._preprocess_image(img)
            obs_images[i] = img

        # Observation proprio: (obs_horizon, 11)
        obs_proprio = episode[obs_start:obs_start + self.obs_horizon].copy()
        obs_proprio = self.stats.normalize_proprio(obs_proprio)

        # Action chunk: (action_horizon, 11) — pose + joints
        action_start = t + 1
        action_end = action_start + self.action_horizon
        actions = episode[action_start:action_end].copy()
        actions = self.stats.normalize_action(actions)

        return (
            torch.from_numpy(obs_images),
            torch.from_numpy(obs_proprio),
            torch.from_numpy(actions),
        )

    def _preprocess_image(self, img: np.ndarray) -> np.ndarray:
        """Resize, optionally augment, ImageNet normalize. Returns (3, H, W)."""
        pil_img = Image.fromarray(img)

        if self.random_crop:
            pil_img = pil_img.resize((self._crop_size, self._crop_size), Image.BILINEAR)
            left = random.randint(0, self._crop_size - self.image_size)
            top = random.randint(0, self._crop_size - self.image_size)
            pil_img = pil_img.crop((left, top, left + self.image_size, top + self.image_size))
        else:
            pil_img = pil_img.resize((self.image_size, self.image_size), Image.BILINEAR)

        if self.color_jitter:
            pil_img = ImageEnhance.Brightness(pil_img).enhance(random.uniform(0.7, 1.3))
            pil_img = ImageEnhance.Contrast(pil_img).enhance(random.uniform(0.7, 1.3))
            pil_img = ImageEnhance.Color(pil_img).enhance(random.uniform(0.8, 1.2))

        arr = np.array(pil_img, dtype=np.float32) / 255.0
        arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
        return arr.transpose(2, 0, 1)
