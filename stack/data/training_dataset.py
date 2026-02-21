"""
PyTorch Dataset for diffusion policy training.

Loads iPhoneSession data and produces (obs_images, obs_proprio, action_chunk)
training samples using sliding window sampling.

Supports two action representations:
- "absolute_quat": 11D = pose (7) + joints (4)  [legacy]
- "relative_6d":   13D = rel_pos (3) + rot6d (6) + joints (4)
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageEnhance
import random

from stack.data.iphone_loader import iPhoneSession
from stack.data.transforms import (
    compute_relative_transform,
    relative_transform_to_13d,
    rotmat_to_6d,
)


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@dataclass
class NormalizationStats:
    """Per-dimension normalization statistics computed from training data."""
    proprio_mean: np.ndarray   # (D,)
    proprio_std: np.ndarray    # (D,)
    action_min: np.ndarray     # (D,)
    action_max: np.ndarray     # (D,)

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


def compute_normalization_stats(
    sessions: list[iPhoneSession],
    action_dim: int = 13,
    action_repr: str = "relative_6d",
    obs_horizon: int = 2,
    action_horizon: int = 16,
    num_samples: int = 10000,
) -> NormalizationStats:
    """Compute normalization statistics across all sessions.

    For absolute_quat: stats over raw 11D episodes.
    For relative_6d: sample random windows and compute relative stats.
    """
    if action_repr == "absolute_quat":
        all_data = []
        for session in sessions:
            episode = session.get_episode_11d()[:, :action_dim]
            all_data.append(episode)
        all_data = np.concatenate(all_data, axis=0)
        return NormalizationStats(
            proprio_mean=all_data.mean(axis=0).astype(np.float32),
            proprio_std=all_data.std(axis=0).astype(np.float32),
            action_min=all_data.min(axis=0).astype(np.float32),
            action_max=all_data.max(axis=0).astype(np.float32),
        )

    # relative_6d: sample random windows and compute relative 13D stats
    all_transforms = []
    all_joints = []
    episode_lengths = []
    for session in sessions:
        all_transforms.append(session.get_all_transforms())
        all_joints.append(session.get_aligned_encoders())
        episode_lengths.append(len(all_transforms[-1]))

    rng = np.random.default_rng(42)
    all_relative = []

    total_windows = sum(
        max(0, L - obs_horizon - action_horizon + 1) for L in episode_lengths
    )
    samples_per = max(1, num_samples // len(sessions))

    for ep_idx in range(len(sessions)):
        T = episode_lengths[ep_idx]
        transforms = all_transforms[ep_idx]
        joints = all_joints[ep_idx]
        max_t = T - action_horizon
        if max_t <= obs_horizon - 1:
            continue

        valid_range = range(obs_horizon - 1, max_t)
        n_sample = min(samples_per, len(valid_range))
        chosen_ts = rng.choice(list(valid_range), size=n_sample, replace=False)

        for t in chosen_ts:
            T_ref = transforms[t]  # (4, 4) reference frame

            # Compute relative for obs window and action window
            obs_start = t - obs_horizon + 1
            action_start = t + 1
            action_end = action_start + action_horizon

            # Obs proprio (relative)
            for i in range(obs_start, t + 1):
                T_rel = compute_relative_transform(T_ref, transforms[i])
                vec = relative_transform_to_13d(T_rel, joints[i])
                all_relative.append(vec)

            # Action chunk (relative)
            for i in range(action_start, action_end):
                T_rel = compute_relative_transform(T_ref, transforms[i])
                vec = relative_transform_to_13d(T_rel, joints[i])
                all_relative.append(vec)

    all_relative = np.array(all_relative, dtype=np.float32)

    return NormalizationStats(
        proprio_mean=all_relative.mean(axis=0).astype(np.float32),
        proprio_std=all_relative.std(axis=0).astype(np.float32),
        action_min=all_relative.min(axis=0).astype(np.float32),
        action_max=all_relative.max(axis=0).astype(np.float32),
    )


class StackDiffusionDataset(Dataset):
    """
    PyTorch Dataset for diffusion policy training.

    Each sample contains:
    - obs_images: (obs_horizon, 3, H, W) — normalized RGB images
    - obs_proprio: (obs_horizon, D) — normalized proprioception
    - action_chunk: (action_horizon, D) — normalized action targets

    Uses sliding window sampling across episodes. No cross-episode windows.
    """

    def __init__(
        self,
        session_dirs: list[Path],
        obs_horizon: int = 2,
        action_horizon: int = 16,
        image_size: int = 224,
        action_dim: int = 13,
        action_repr: str = "relative_6d",
        stats: NormalizationStats | None = None,
        augment: bool = False,
        random_crop: bool = False,
        color_jitter: bool = False,
    ):
        self.obs_horizon = obs_horizon
        self.action_horizon = action_horizon
        self.image_size = image_size
        self.action_dim = action_dim
        self.action_repr = action_repr
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

        if action_repr == "relative_6d":
            # Preload 4x4 transforms and joint angles for relative computation
            self._transforms: list[np.ndarray] = []
            self._joints: list[np.ndarray] = []
            self._episode_lengths: list[int] = []
            for s in self.sessions:
                transforms = s.get_all_transforms()
                joints = s.get_aligned_encoders()
                self._transforms.append(transforms)
                self._joints.append(joints)
                self._episode_lengths.append(len(transforms))
            # Also keep _episodes for backward compat (not used in __getitem__)
            self._episodes = [None] * len(self.sessions)
        else:
            # Legacy absolute mode: preload 11D episodes sliced to action_dim
            self._episodes: list[np.ndarray] = []
            for s in self.sessions:
                self._episodes.append(s.get_episode_11d()[:, :action_dim])
            self._transforms = None
            self._joints = None
            self._episode_lengths = [len(ep) for ep in self._episodes]

        # Compute or use provided normalization stats
        if stats is not None:
            self.stats = stats
        else:
            self.stats = compute_normalization_stats(
                self.sessions,
                action_dim=action_dim,
                action_repr=action_repr,
                obs_horizon=obs_horizon,
                action_horizon=action_horizon,
            )

        # Build index: list of (episode_idx, frame_idx) for valid windows
        self._samples: list[tuple[int, int]] = []
        for ep_idx in range(len(self.sessions)):
            T = self._episode_lengths[ep_idx]
            for t in range(self.obs_horizon - 1, T - self.action_horizon):
                self._samples.append((ep_idx, t))

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ep_idx, t = self._samples[idx]
        session = self.sessions[ep_idx]

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

        if self.action_repr == "relative_6d":
            transforms = self._transforms[ep_idx]
            joints = self._joints[ep_idx]
            T_ref = transforms[t]  # reference = last obs timestep

            # Obs proprio: relative to T_ref
            obs_proprio = np.zeros((self.obs_horizon, self.action_dim), dtype=np.float32)
            for i in range(self.obs_horizon):
                fi = obs_start + i
                T_rel = compute_relative_transform(T_ref, transforms[fi])
                obs_proprio[i] = relative_transform_to_13d(T_rel, joints[fi])
            obs_proprio = self.stats.normalize_proprio(obs_proprio)

            # Action chunk: relative to T_ref
            action_start = t + 1
            action_end = action_start + self.action_horizon
            actions = np.zeros((self.action_horizon, self.action_dim), dtype=np.float32)
            for i in range(self.action_horizon):
                fi = action_start + i
                T_rel = compute_relative_transform(T_ref, transforms[fi])
                actions[i] = relative_transform_to_13d(T_rel, joints[fi])
            actions = self.stats.normalize_action(actions)
        else:
            episode = self._episodes[ep_idx]
            obs_proprio = episode[obs_start:obs_start + self.obs_horizon].copy()
            obs_proprio = self.stats.normalize_proprio(obs_proprio)

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
