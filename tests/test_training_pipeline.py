"""
Integration tests for the full training pipeline.

Tests synthetic data generation, dataset creation, training step,
checkpoint save/load, and evaluation — end-to-end smoke tests.
"""

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from stack.data.synthetic import generate_synthetic_session, generate_synthetic_dataset
from stack.data.iphone_loader import iPhoneSession
from stack.data.training_dataset import (
    StackDiffusionDataset,
    NormalizationStats,
    compute_normalization_stats,
)
from stack.policy.diffusion import DiffusionPolicy, PolicyConfig


@pytest.fixture
def tmp_dir():
    """Temporary directory cleaned up after test."""
    d = Path(tempfile.mkdtemp())
    yield d
    shutil.rmtree(d)


@pytest.fixture
def synthetic_sessions(tmp_dir):
    """Generate 2 small synthetic sessions."""
    sessions = []
    for i in range(2):
        session_dir = tmp_dir / f"session_{i:03d}"
        generate_synthetic_session(session_dir, num_frames=60, seed=i)
        sessions.append(session_dir)
    return sessions


@pytest.fixture
def small_config():
    """Small policy config for fast tests."""
    return PolicyConfig(
        obs_dim=11,
        action_dim=11,
        image_size=64,  # Small for speed
        obs_horizon=2,
        action_horizon=16,  # Must be >= 16 for UNet downsampling
        num_diffusion_steps=10,  # Few steps for speed
        hidden_dim=32,  # Small for speed
        num_layers=2,
        batch_size=4,
        learning_rate=1e-4,
        num_epochs=1,
    )


class TestSyntheticData:
    def test_generate_single_session(self, tmp_dir):
        session_dir = tmp_dir / "test_session"
        generate_synthetic_session(session_dir, num_frames=50, seed=0)

        assert (session_dir / "metadata.json").exists()
        assert (session_dir / "poses.json").exists()
        assert (session_dir / "encoders.json").exists()
        assert (session_dir / "rgb").is_dir()
        assert len(list((session_dir / "rgb").glob("*.jpg"))) == 50

    def test_session_loadable_by_iphone_session(self, tmp_dir):
        session_dir = tmp_dir / "test_session"
        generate_synthetic_session(session_dir, num_frames=30, seed=42)

        session = iPhoneSession(session_dir)
        assert session.num_poses == 30
        assert session.num_rgb_frames == 30
        assert session.has_encoders
        assert session.num_encoder_readings == 30

    def test_11d_episode_shape(self, tmp_dir):
        session_dir = tmp_dir / "test_session"
        generate_synthetic_session(session_dir, num_frames=30, seed=42)

        session = iPhoneSession(session_dir)
        episode = session.get_episode_11d()
        assert episode.shape == (30, 11)
        assert episode.dtype == np.float32
        assert not np.isnan(episode).any()

    def test_generate_dataset(self, tmp_dir):
        sessions = generate_synthetic_dataset(
            tmp_dir / "dataset", num_sessions=3, frames_per_session=20, seed=0
        )
        assert len(sessions) == 3
        for s in sessions:
            assert s.exists()
            assert (s / "poses.json").exists()

    def test_reproducibility(self, tmp_dir):
        dir1 = tmp_dir / "s1"
        dir2 = tmp_dir / "s2"
        generate_synthetic_session(dir1, num_frames=20, seed=123)
        generate_synthetic_session(dir2, num_frames=20, seed=123)

        s1 = iPhoneSession(dir1)
        s2 = iPhoneSession(dir2)
        np.testing.assert_array_equal(s1.get_episode_11d(), s2.get_episode_11d())


class TestDataset:
    def test_dataset_creation(self, synthetic_sessions, small_config):
        dataset = StackDiffusionDataset(
            synthetic_sessions,
            obs_horizon=small_config.obs_horizon,
            action_horizon=small_config.action_horizon,
            image_size=small_config.image_size,
        )
        assert len(dataset) > 0

    def test_sample_shapes(self, synthetic_sessions, small_config):
        dataset = StackDiffusionDataset(
            synthetic_sessions,
            obs_horizon=small_config.obs_horizon,
            action_horizon=small_config.action_horizon,
            image_size=small_config.image_size,
        )
        images, proprio, actions = dataset[0]

        assert images.shape == (small_config.obs_horizon, 3, small_config.image_size, small_config.image_size)
        assert proprio.shape == (small_config.obs_horizon, 11)
        assert actions.shape == (small_config.action_horizon, 11)
        assert images.dtype == torch.float32
        assert proprio.dtype == torch.float32
        assert actions.dtype == torch.float32

    def test_no_nan_in_samples(self, synthetic_sessions, small_config):
        dataset = StackDiffusionDataset(
            synthetic_sessions,
            obs_horizon=small_config.obs_horizon,
            action_horizon=small_config.action_horizon,
            image_size=small_config.image_size,
        )
        for i in range(min(5, len(dataset))):
            images, proprio, actions = dataset[i]
            assert not torch.isnan(images).any(), f"NaN in images at sample {i}"
            assert not torch.isnan(proprio).any(), f"NaN in proprio at sample {i}"
            assert not torch.isnan(actions).any(), f"NaN in actions at sample {i}"

    def test_normalization_stats(self, synthetic_sessions):
        sessions = [iPhoneSession(d) for d in synthetic_sessions]
        stats = compute_normalization_stats(sessions)

        assert stats.proprio_mean.shape == (11,)
        assert stats.proprio_std.shape == (11,)
        assert stats.action_min.shape == (11,)
        assert stats.action_max.shape == (11,)

        # Round-trip test
        action = np.random.randn(5, 11).astype(np.float32)
        normalized = stats.normalize_action(action)
        recovered = stats.unnormalize_action(normalized)
        np.testing.assert_allclose(recovered, action, atol=1e-5)

    def test_stats_serialization(self, synthetic_sessions):
        sessions = [iPhoneSession(d) for d in synthetic_sessions]
        stats = compute_normalization_stats(sessions)

        state = stats.state_dict()
        stats2 = NormalizationStats.from_state_dict(state)

        np.testing.assert_array_equal(stats.proprio_mean, stats2.proprio_mean)
        np.testing.assert_array_equal(stats.action_min, stats2.action_min)

    def test_dataloader(self, synthetic_sessions, small_config):
        dataset = StackDiffusionDataset(
            synthetic_sessions,
            obs_horizon=small_config.obs_horizon,
            action_horizon=small_config.action_horizon,
            image_size=small_config.image_size,
        )
        loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)
        batch = next(iter(loader))
        images, proprio, actions = batch

        assert images.shape[0] == 4
        assert proprio.shape[0] == 4
        assert actions.shape[0] == 4


class TestTrainingStep:
    def test_forward_pass(self, synthetic_sessions, small_config):
        dataset = StackDiffusionDataset(
            synthetic_sessions,
            obs_horizon=small_config.obs_horizon,
            action_horizon=small_config.action_horizon,
            image_size=small_config.image_size,
        )
        loader = torch.utils.data.DataLoader(dataset, batch_size=4)
        images, proprio, actions = next(iter(loader))

        policy = DiffusionPolicy(small_config)
        loss_dict = policy(images, proprio, actions)

        assert "loss" in loss_dict
        assert torch.isfinite(loss_dict["loss"])
        assert loss_dict["loss"].item() > 0

    def test_backward_pass(self, synthetic_sessions, small_config):
        dataset = StackDiffusionDataset(
            synthetic_sessions,
            obs_horizon=small_config.obs_horizon,
            action_horizon=small_config.action_horizon,
            image_size=small_config.image_size,
        )
        loader = torch.utils.data.DataLoader(dataset, batch_size=4)
        images, proprio, actions = next(iter(loader))

        policy = DiffusionPolicy(small_config)
        loss_dict = policy(images, proprio, actions)
        loss_dict["loss"].backward()

        # Check gradients exist and are finite
        for name, param in policy.named_parameters():
            if param.grad is not None:
                assert torch.isfinite(param.grad).all(), f"Non-finite gradient in {name}"

    def test_predict_shape(self, synthetic_sessions, small_config):
        dataset = StackDiffusionDataset(
            synthetic_sessions,
            obs_horizon=small_config.obs_horizon,
            action_horizon=small_config.action_horizon,
            image_size=small_config.image_size,
        )
        loader = torch.utils.data.DataLoader(dataset, batch_size=2)
        images, proprio, _ = next(iter(loader))

        policy = DiffusionPolicy(small_config)
        policy.eval()
        pred = policy.predict(images, proprio)

        assert pred.shape == (2, small_config.action_horizon, small_config.action_dim)
        assert torch.isfinite(pred).all()


class TestCheckpoint:
    def test_save_and_load(self, synthetic_sessions, small_config, tmp_dir):
        policy = DiffusionPolicy(small_config)
        optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-4)

        # Compute stats
        sessions = [iPhoneSession(d) for d in synthetic_sessions]
        stats = compute_normalization_stats(sessions)

        # Do one forward pass to get non-zero gradients/state
        dataset = StackDiffusionDataset(
            synthetic_sessions,
            obs_horizon=small_config.obs_horizon,
            action_horizon=small_config.action_horizon,
            image_size=small_config.image_size,
        )
        loader = torch.utils.data.DataLoader(dataset, batch_size=4)
        images, proprio, actions = next(iter(loader))
        loss = policy(images, proprio, actions)["loss"]
        loss.backward()
        optimizer.step()

        # Save
        ckpt_path = tmp_dir / "test_checkpoint.pt"
        torch.save({
            "epoch": 0,
            "model": policy.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": vars(small_config),
            "normalizer": stats.state_dict(),
        }, ckpt_path)

        # Load
        checkpoint = torch.load(ckpt_path, weights_only=False)
        loaded_config = PolicyConfig(**checkpoint["config"])
        loaded_policy = DiffusionPolicy(loaded_config)
        loaded_policy.load_state_dict(checkpoint["model"])

        # Verify weights match
        for (n1, p1), (n2, p2) in zip(
            policy.named_parameters(), loaded_policy.named_parameters()
        ):
            assert n1 == n2
            torch.testing.assert_close(p1.data, p2.data)

        # Verify normalizer round-trips
        loaded_stats = NormalizationStats.from_state_dict(checkpoint["normalizer"])
        np.testing.assert_array_equal(stats.proprio_mean, loaded_stats.proprio_mean)


class TestEvaluation:
    def test_prediction_on_synthetic(self, synthetic_sessions, small_config):
        """Run prediction on synthetic data and verify output shapes."""
        policy = DiffusionPolicy(small_config)
        policy.eval()

        session = iPhoneSession(synthetic_sessions[0])
        episode = session.get_episode_11d()

        sessions = [iPhoneSession(d) for d in synthetic_sessions]
        stats = compute_normalization_stats(sessions)

        # Build one observation window
        from stack.data.training_dataset import IMAGENET_MEAN, IMAGENET_STD
        from PIL import Image as PILImage

        t = small_config.obs_horizon  # First valid timestep
        obs_start = t - small_config.obs_horizon + 1

        obs_images = np.zeros(
            (1, small_config.obs_horizon, 3, small_config.image_size, small_config.image_size),
            dtype=np.float32,
        )
        for i in range(small_config.obs_horizon):
            img = session.get_rgb_frame(obs_start + i)
            pil_img = PILImage.fromarray(img)
            pil_img = pil_img.resize(
                (small_config.image_size, small_config.image_size), PILImage.BILINEAR
            )
            arr = np.array(pil_img, dtype=np.float32) / 255.0
            arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
            obs_images[0, i] = arr.transpose(2, 0, 1)

        obs_proprio = episode[obs_start:obs_start + small_config.obs_horizon].copy()
        obs_proprio = stats.normalize_proprio(obs_proprio)[np.newaxis]

        pred = policy.predict(
            torch.from_numpy(obs_images),
            torch.from_numpy(obs_proprio),
        )

        assert pred.shape == (1, small_config.action_horizon, small_config.action_dim)
        assert torch.isfinite(pred).all()
