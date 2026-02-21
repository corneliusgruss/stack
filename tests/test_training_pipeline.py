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
from stack.data.transforms import (
    rotmat_to_6d,
    rot6d_to_rotmat,
    pose_7d_to_mat,
    mat_to_pose_7d,
    compute_relative_transform,
    relative_transform_to_13d,
    action_13d_to_absolute,
    rot6d_angular_error,
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
    """Small policy config for fast tests (relative_6d)."""
    return PolicyConfig(
        obs_dim=13,
        action_dim=13,
        action_repr="relative_6d",
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


@pytest.fixture
def small_config_absolute():
    """Small policy config for fast tests (absolute_quat, legacy)."""
    return PolicyConfig(
        obs_dim=11,
        action_dim=11,
        action_repr="absolute_quat",
        image_size=64,
        obs_horizon=2,
        action_horizon=16,
        num_diffusion_steps=10,
        hidden_dim=32,
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

    def test_get_all_transforms(self, tmp_dir):
        session_dir = tmp_dir / "test_session"
        generate_synthetic_session(session_dir, num_frames=30, seed=42)

        session = iPhoneSession(session_dir)
        transforms = session.get_all_transforms()
        assert transforms.shape == (30, 4, 4)
        assert transforms.dtype == np.float32
        # Check they're valid transforms (last row = [0, 0, 0, 1])
        np.testing.assert_allclose(transforms[:, 3, :], [[0, 0, 0, 1]] * 30, atol=1e-6)

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


class TestTransforms:
    """Tests for stack/data/transforms.py"""

    def test_rotmat_to_6d_roundtrip(self):
        """rotmat -> 6d -> rotmat should recover original."""
        from scipy.spatial.transform import Rotation
        rng = np.random.default_rng(42)
        for _ in range(10):
            R_orig = Rotation.random(random_state=rng).as_matrix()
            r6d = rotmat_to_6d(R_orig)
            R_recovered = rot6d_to_rotmat(r6d)
            np.testing.assert_allclose(R_recovered, R_orig, atol=1e-6)

    def test_6d_identity(self):
        """Identity rotation should give [1,0,0, 0,1,0]."""
        R = np.eye(3)
        r6d = rotmat_to_6d(R)
        np.testing.assert_allclose(r6d, [1, 0, 0, 0, 1, 0], atol=1e-8)

    def test_pose_7d_mat_roundtrip(self):
        """pose_7d -> mat -> pose_7d should recover original."""
        from scipy.spatial.transform import Rotation
        rng = np.random.default_rng(42)
        for _ in range(10):
            pos = rng.standard_normal(3)
            quat = Rotation.random(random_state=rng).as_quat()  # xyzw
            pose = np.concatenate([pos, quat])
            T = pose_7d_to_mat(pose)
            pose_rec = mat_to_pose_7d(T)
            np.testing.assert_allclose(pose_rec[:3], pos, atol=1e-5)
            # Quaternions may differ by sign
            dot = abs(np.dot(pose_rec[3:], quat))
            assert dot > 0.999, f"Quaternion mismatch: dot={dot}"

    def test_relative_transform_identity(self):
        """Relative transform of a pose with itself should be identity."""
        from scipy.spatial.transform import Rotation
        rng = np.random.default_rng(42)
        R = Rotation.random(random_state=rng).as_matrix()
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = [1.0, 2.0, 3.0]

        T_rel = compute_relative_transform(T, T)
        np.testing.assert_allclose(T_rel[:3, :3], np.eye(3), atol=1e-5)
        np.testing.assert_allclose(T_rel[:3, 3], [0, 0, 0], atol=1e-5)

    def test_relative_transform_to_13d_identity(self):
        """Self-relative transform should give [0,0,0, 1,0,0, 0,1,0, joints]."""
        T_rel = np.eye(4, dtype=np.float32)
        joints = np.array([30.0, 45.0, 60.0, 15.0], dtype=np.float32)
        vec = relative_transform_to_13d(T_rel, joints)
        assert vec.shape == (13,)
        np.testing.assert_allclose(vec[:3], [0, 0, 0], atol=1e-6)
        np.testing.assert_allclose(vec[3:9], [1, 0, 0, 0, 1, 0], atol=1e-6)
        np.testing.assert_allclose(vec[9:], joints, atol=1e-6)

    def test_action_13d_roundtrip(self):
        """13d relative -> absolute -> 13d relative should recover."""
        from scipy.spatial.transform import Rotation
        rng = np.random.default_rng(42)

        # Random reference
        R_ref = Rotation.random(random_state=rng).as_matrix()
        T_ref = np.eye(4, dtype=np.float32)
        T_ref[:3, :3] = R_ref
        T_ref[:3, 3] = rng.standard_normal(3)

        # Random target
        R_t = Rotation.random(random_state=rng).as_matrix()
        T_t = np.eye(4, dtype=np.float32)
        T_t[:3, :3] = R_t
        T_t[:3, 3] = rng.standard_normal(3)

        joints = rng.uniform(0, 90, 4).astype(np.float32)

        # Forward: compute relative
        T_rel = compute_relative_transform(T_ref, T_t)
        a13d = relative_transform_to_13d(T_rel, joints)

        # Backward: recover absolute
        T_abs, joints_rec = action_13d_to_absolute(a13d, T_ref)

        np.testing.assert_allclose(T_abs[:3, 3], T_t[:3, 3], atol=1e-4)
        np.testing.assert_allclose(T_abs[:3, :3], T_t[:3, :3], atol=1e-4)
        np.testing.assert_allclose(joints_rec, joints, atol=1e-6)

    def test_rot6d_angular_error_zero(self):
        """Same rotation should give zero error."""
        r6d = np.array([1, 0, 0, 0, 1, 0], dtype=np.float32)
        err = rot6d_angular_error(r6d, r6d)
        assert err < 0.1  # degrees

    def test_rot6d_angular_error_90deg(self):
        """90-degree rotation should give ~90 degree error."""
        from scipy.spatial.transform import Rotation
        R1 = np.eye(3)
        R2 = Rotation.from_euler("z", 90, degrees=True).as_matrix()
        r6d_1 = rotmat_to_6d(R1)
        r6d_2 = rotmat_to_6d(R2)
        err = rot6d_angular_error(r6d_1, r6d_2)
        np.testing.assert_allclose(err, 90.0, atol=1.0)

    def test_batched_operations(self):
        """Test that transforms work with batched inputs."""
        from scipy.spatial.transform import Rotation
        rng = np.random.default_rng(42)
        n = 5

        # Batch of rotation matrices
        Rs = Rotation.random(n, random_state=rng).as_matrix()
        r6ds = rotmat_to_6d(Rs)
        assert r6ds.shape == (n, 6)

        Rs_rec = rot6d_to_rotmat(r6ds)
        assert Rs_rec.shape == (n, 3, 3)
        np.testing.assert_allclose(Rs_rec, Rs, atol=1e-5)


class TestDataset:
    def test_dataset_creation_relative(self, synthetic_sessions, small_config):
        dataset = StackDiffusionDataset(
            synthetic_sessions,
            obs_horizon=small_config.obs_horizon,
            action_horizon=small_config.action_horizon,
            image_size=small_config.image_size,
            action_dim=small_config.action_dim,
            action_repr="relative_6d",
        )
        assert len(dataset) > 0

    def test_dataset_creation_absolute(self, synthetic_sessions, small_config_absolute):
        dataset = StackDiffusionDataset(
            synthetic_sessions,
            obs_horizon=small_config_absolute.obs_horizon,
            action_horizon=small_config_absolute.action_horizon,
            image_size=small_config_absolute.image_size,
            action_dim=small_config_absolute.action_dim,
            action_repr="absolute_quat",
        )
        assert len(dataset) > 0

    def test_sample_shapes_relative(self, synthetic_sessions, small_config):
        dataset = StackDiffusionDataset(
            synthetic_sessions,
            obs_horizon=small_config.obs_horizon,
            action_horizon=small_config.action_horizon,
            image_size=small_config.image_size,
            action_dim=small_config.action_dim,
            action_repr="relative_6d",
        )
        images, proprio, actions = dataset[0]

        assert images.shape == (small_config.obs_horizon, 3, small_config.image_size, small_config.image_size)
        assert proprio.shape == (small_config.obs_horizon, 13)
        assert actions.shape == (small_config.action_horizon, 13)
        assert images.dtype == torch.float32
        assert proprio.dtype == torch.float32
        assert actions.dtype == torch.float32

    def test_sample_shapes_absolute(self, synthetic_sessions, small_config_absolute):
        dataset = StackDiffusionDataset(
            synthetic_sessions,
            obs_horizon=small_config_absolute.obs_horizon,
            action_horizon=small_config_absolute.action_horizon,
            image_size=small_config_absolute.image_size,
            action_dim=small_config_absolute.action_dim,
            action_repr="absolute_quat",
        )
        images, proprio, actions = dataset[0]

        assert proprio.shape == (small_config_absolute.obs_horizon, 11)
        assert actions.shape == (small_config_absolute.action_horizon, 11)

    def test_last_obs_is_near_identity(self, synthetic_sessions, small_config):
        """In relative_6d mode, last obs timestep should be near identity."""
        dataset = StackDiffusionDataset(
            synthetic_sessions,
            obs_horizon=small_config.obs_horizon,
            action_horizon=small_config.action_horizon,
            image_size=small_config.image_size,
            action_dim=small_config.action_dim,
            action_repr="relative_6d",
        )
        _, proprio, _ = dataset[0]
        # Unnormalize last obs proprio
        last_proprio = dataset.stats.normalize_proprio.__func__  # hacky but let's just check raw
        # Actually let's check the unnormalized version
        raw_proprio = proprio[-1].numpy()  # normalized
        # The last obs in relative mode is identity before normalization:
        # [0,0,0, 1,0,0, 0,1,0, j1,j2,j3,j4]
        # After normalization it won't be exactly identity, but it should be consistent

        # Instead check that all samples have the same last-obs values (since it's always identity + joints)
        _, proprio2, _ = dataset[1]
        # Position part (first 3 dims) should be identical (0 before norm)
        np.testing.assert_allclose(
            proprio[-1, :3].numpy(), proprio2[-1, :3].numpy(), atol=1e-5
        )
        # Rotation part (dims 3:9) should be identical (identity before norm)
        np.testing.assert_allclose(
            proprio[-1, 3:9].numpy(), proprio2[-1, 3:9].numpy(), atol=1e-5
        )

    def test_no_nan_in_samples(self, synthetic_sessions, small_config):
        dataset = StackDiffusionDataset(
            synthetic_sessions,
            obs_horizon=small_config.obs_horizon,
            action_horizon=small_config.action_horizon,
            image_size=small_config.image_size,
            action_dim=small_config.action_dim,
            action_repr="relative_6d",
        )
        for i in range(min(5, len(dataset))):
            images, proprio, actions = dataset[i]
            assert not torch.isnan(images).any(), f"NaN in images at sample {i}"
            assert not torch.isnan(proprio).any(), f"NaN in proprio at sample {i}"
            assert not torch.isnan(actions).any(), f"NaN in actions at sample {i}"

    def test_normalization_stats_relative(self, synthetic_sessions):
        sessions = [iPhoneSession(d) for d in synthetic_sessions]
        stats = compute_normalization_stats(sessions, action_dim=13, action_repr="relative_6d")

        assert stats.proprio_mean.shape == (13,)
        assert stats.proprio_std.shape == (13,)
        assert stats.action_min.shape == (13,)
        assert stats.action_max.shape == (13,)

        # Relative positions should be centered near zero
        assert abs(stats.proprio_mean[0]) < 1.0  # dx
        assert abs(stats.proprio_mean[1]) < 1.0  # dy
        assert abs(stats.proprio_mean[2]) < 1.0  # dz

    def test_normalization_stats_absolute(self, synthetic_sessions):
        sessions = [iPhoneSession(d) for d in synthetic_sessions]
        stats = compute_normalization_stats(sessions, action_dim=11, action_repr="absolute_quat")

        assert stats.proprio_mean.shape == (11,)
        assert stats.action_min.shape == (11,)

        # Round-trip test
        action = np.random.randn(5, 11).astype(np.float32)
        normalized = stats.normalize_action(action)
        recovered = stats.unnormalize_action(normalized)
        np.testing.assert_allclose(recovered, action, atol=1e-5)

    def test_stats_serialization(self, synthetic_sessions):
        sessions = [iPhoneSession(d) for d in synthetic_sessions]
        stats = compute_normalization_stats(sessions, action_dim=13, action_repr="relative_6d")

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
            action_dim=small_config.action_dim,
            action_repr="relative_6d",
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
            action_dim=small_config.action_dim,
            action_repr="relative_6d",
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
            action_dim=small_config.action_dim,
            action_repr="relative_6d",
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
            action_dim=small_config.action_dim,
            action_repr="relative_6d",
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
        stats = compute_normalization_stats(sessions, action_dim=13, action_repr="relative_6d")

        # Do one forward pass to get non-zero gradients/state
        dataset = StackDiffusionDataset(
            synthetic_sessions,
            obs_horizon=small_config.obs_horizon,
            action_horizon=small_config.action_horizon,
            image_size=small_config.image_size,
            action_dim=small_config.action_dim,
            action_repr="relative_6d",
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

    def test_backward_compat_load(self, synthetic_sessions, tmp_dir):
        """Old checkpoints without action_repr should default to absolute_quat."""
        config = PolicyConfig(
            obs_dim=11, action_dim=11, action_repr="absolute_quat",
            image_size=64, hidden_dim=32, num_diffusion_steps=10,
        )
        policy = DiffusionPolicy(config)

        # Save without action_repr (simulate old checkpoint)
        ckpt_path = tmp_dir / "old_checkpoint.pt"
        old_config_dict = vars(config).copy()
        del old_config_dict["action_repr"]
        torch.save({
            "epoch": 0,
            "model": policy.state_dict(),
            "config": old_config_dict,
        }, ckpt_path)

        # Load with backward compat
        checkpoint = torch.load(ckpt_path, weights_only=False)
        ckpt_config = checkpoint["config"].copy()
        if "action_repr" not in ckpt_config:
            ckpt_config["action_repr"] = "absolute_quat"
        loaded_config = PolicyConfig(**ckpt_config)
        assert loaded_config.action_repr == "absolute_quat"
        assert loaded_config.obs_dim == 11


class TestEvaluation:
    def test_prediction_on_synthetic(self, synthetic_sessions, small_config):
        """Run prediction on synthetic data and verify output shapes."""
        policy = DiffusionPolicy(small_config)
        policy.eval()

        session = iPhoneSession(synthetic_sessions[0])

        sessions = [iPhoneSession(d) for d in synthetic_sessions]
        stats = compute_normalization_stats(sessions, action_dim=13, action_repr="relative_6d")

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

        # Build relative proprio
        transforms = session.get_all_transforms()
        joints = session.get_aligned_encoders()
        T_ref = transforms[t]
        obs_proprio = np.zeros((small_config.obs_horizon, 13), dtype=np.float32)
        for i in range(small_config.obs_horizon):
            fi = obs_start + i
            T_rel = compute_relative_transform(T_ref, transforms[fi])
            obs_proprio[i] = relative_transform_to_13d(T_rel, joints[fi])
        obs_proprio = stats.normalize_proprio(obs_proprio)[np.newaxis]

        pred = policy.predict(
            torch.from_numpy(obs_images),
            torch.from_numpy(obs_proprio),
        )

        assert pred.shape == (1, small_config.action_horizon, small_config.action_dim)
        assert torch.isfinite(pred).all()
