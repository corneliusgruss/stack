"""
Tests for evaluation visualization pipeline.

Tests trajectory output from eval, plotting functions, frame overlay,
and train/val split consistency with train.py.
"""

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from stack.data.synthetic import generate_synthetic_session
from stack.data.iphone_loader import iPhoneSession
from stack.data.training_dataset import NormalizationStats, compute_normalization_stats
from stack.policy.diffusion import DiffusionPolicy, PolicyConfig
from stack.scripts.eval import evaluate_episode, find_session_dirs, split_sessions
from stack.viz.eval_viz import (
    plot_trajectory_comparison_3d,
    plot_position_over_time,
    plot_joints_over_time,
    plot_error_distribution,
    plot_per_session_metrics,
    render_prediction_on_frame,
    create_dashboard,
)


@pytest.fixture
def tmp_dir():
    d = Path(tempfile.mkdtemp())
    yield d
    shutil.rmtree(d)


@pytest.fixture
def synthetic_sessions(tmp_dir):
    """Generate 3 small synthetic sessions."""
    sessions = []
    for i in range(3):
        session_dir = tmp_dir / f"session_{i:03d}"
        generate_synthetic_session(session_dir, num_frames=60, seed=i)
        sessions.append(session_dir)
    return sessions


@pytest.fixture
def small_config():
    return PolicyConfig(
        obs_dim=11,
        action_dim=11,
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


class TestEvaluateWithTrajectories:
    def test_return_trajectories_shapes(self, synthetic_sessions, small_config):
        """Verify returned trajectory arrays have correct shapes."""
        policy = DiffusionPolicy(small_config)
        policy.eval()

        sessions = [iPhoneSession(d) for d in synthetic_sessions]
        stats = compute_normalization_stats(sessions)

        session = sessions[0]
        result = evaluate_episode(
            policy, session, stats,
            obs_horizon=small_config.obs_horizon,
            action_horizon=small_config.action_horizon,
            image_size=small_config.image_size,
            device=torch.device("cpu"),
            eval_stride=5,
            return_trajectories=True,
        )

        assert not result["skipped"]
        assert "pred_actions_all" in result
        assert "gt_actions_all" in result
        assert "eval_indices" in result

        n_windows = result["num_windows"]
        assert result["pred_actions_all"].shape == (n_windows, small_config.action_horizon, 11)
        assert result["gt_actions_all"].shape == (n_windows, small_config.action_horizon, 11)
        assert result["eval_indices"].shape == (n_windows,)

    def test_no_trajectories_by_default(self, synthetic_sessions, small_config):
        """When return_trajectories=False, no trajectory arrays returned."""
        policy = DiffusionPolicy(small_config)
        policy.eval()

        sessions = [iPhoneSession(d) for d in synthetic_sessions]
        stats = compute_normalization_stats(sessions)

        result = evaluate_episode(
            policy, sessions[0], stats,
            obs_horizon=small_config.obs_horizon,
            action_horizon=small_config.action_horizon,
            image_size=small_config.image_size,
            device=torch.device("cpu"),
            eval_stride=10,
            return_trajectories=False,
        )

        assert "pred_actions_all" not in result
        assert "gt_actions_all" not in result

    def test_stride_reduces_windows(self, synthetic_sessions, small_config):
        """Larger stride = fewer evaluation windows."""
        policy = DiffusionPolicy(small_config)
        policy.eval()

        sessions = [iPhoneSession(d) for d in synthetic_sessions]
        stats = compute_normalization_stats(sessions)

        r1 = evaluate_episode(
            policy, sessions[0], stats,
            obs_horizon=small_config.obs_horizon,
            action_horizon=small_config.action_horizon,
            image_size=small_config.image_size,
            device=torch.device("cpu"),
            eval_stride=1,
        )
        r5 = evaluate_episode(
            policy, sessions[0], stats,
            obs_horizon=small_config.obs_horizon,
            action_horizon=small_config.action_horizon,
            image_size=small_config.image_size,
            device=torch.device("cpu"),
            eval_stride=5,
        )

        assert r5["num_windows"] < r1["num_windows"]
        # Stride 5 should give roughly 1/5 as many windows
        assert r5["num_windows"] <= r1["num_windows"] // 4


class TestPlotFunctions:
    def test_plot_trajectory_comparison_3d(self):
        pred = np.random.randn(50, 3) * 0.1
        gt = np.random.randn(50, 3) * 0.1

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        result = plot_trajectory_comparison_3d(pred, gt, ax=ax)
        assert result is ax
        plt.close(fig)

    def test_plot_position_over_time(self):
        n = 50
        pred = np.random.randn(n, 3) * 0.1
        gt = np.random.randn(n, 3) * 0.1
        idx = np.arange(n) * 10

        fig = plot_position_over_time(pred, gt, idx)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_joints_over_time(self):
        n = 50
        pred = np.random.randn(n, 4) * 10
        gt = np.random.randn(n, 4) * 10
        idx = np.arange(n) * 10

        fig = plot_joints_over_time(pred, gt, idx)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_error_distribution(self):
        errors = {
            "position": np.random.exponential(0.01, 200),
            "rotation": np.random.exponential(5.0, 200),
            "joints": np.random.exponential(3.0, 200),
        }
        fig = plot_error_distribution(errors)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_per_session_metrics(self):
        results = [
            {"name": f"session_{i}", "position_mse": 0.01 * i,
             "rotation_error_deg": 5.0 * i, "joint_error_deg": 3.0 * i}
            for i in range(1, 4)
        ]
        fig = plot_per_session_metrics(results)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_saves_to_file(self, tmp_dir):
        pred = np.random.randn(20, 3) * 0.1
        gt = np.random.randn(20, 3) * 0.1
        idx = np.arange(20) * 10

        save_path = str(tmp_dir / "test_plot.png")
        fig = plot_position_over_time(pred, gt, idx, save_path=save_path)
        plt.close(fig)
        assert Path(save_path).exists()
        assert Path(save_path).stat().st_size > 0


class TestRenderPredictionOnFrame:
    def test_output_is_valid_rgb(self):
        rgb = np.random.randint(0, 255, (360, 480, 3), dtype=np.uint8)
        pose = np.eye(4, dtype=np.float32)
        # Camera at origin looking along +z. Points need positive z in camera frame.
        # With identity pose: R_cw=I, t_cw=[0,0,0], so p_cam = p_world.
        # Points at z=1..2 are in front of camera.

        pred_pos = np.zeros((16, 3), dtype=np.float32)
        pred_pos[:, 2] = np.linspace(1.0, 2.0, 16)  # Along z-axis, in front
        pred_pos[:, 0] = np.linspace(-0.1, 0.1, 16)  # Slight x spread
        gt_pos = pred_pos + 0.02  # Slight offset
        intrinsics = np.array([183.0, 183.0, 240.0, 180.0])

        result = render_prediction_on_frame(rgb, pose, pred_pos, gt_pos, intrinsics)

        assert result.shape == (360, 480, 3)
        assert result.dtype == np.uint8
        # Should be different from input (overlay was drawn)
        assert not np.array_equal(result, rgb)

    def test_behind_camera_handled(self):
        """Points behind camera should not crash."""
        rgb = np.zeros((360, 480, 3), dtype=np.uint8)
        pose = np.eye(4, dtype=np.float32)

        # All points behind camera (negative Z in camera frame)
        pred_pos = np.array([[0, 0, -1]] * 5, dtype=np.float32)
        gt_pos = pred_pos.copy()
        intrinsics = np.array([183.0, 183.0, 240.0, 180.0])

        result = render_prediction_on_frame(rgb, pose, pred_pos, gt_pos, intrinsics)
        assert result.shape == (360, 480, 3)


class TestTrainValSplitConsistency:
    def test_split_matches_train_logic(self, synthetic_sessions):
        """Verify split_sessions produces same split as train.py logic."""
        session_dirs = sorted(synthetic_sessions)
        val_split = 0.2
        seed = 42

        # Our split function
        val_dirs = split_sessions(session_dirs, val_split, seed, "val")
        train_dirs = split_sessions(session_dirs, val_split, seed, "train")

        # Replicate train.py logic directly
        n_val = max(1, int(len(session_dirs) * val_split))
        n_train = len(session_dirs) - n_val
        rng = np.random.default_rng(seed)
        indices = rng.permutation(len(session_dirs))
        expected_train = [session_dirs[i] for i in indices[:n_train]]
        expected_val = [session_dirs[i] for i in indices[n_train:]]

        assert len(val_dirs) == len(expected_val)
        assert len(train_dirs) == len(expected_train)
        for a, b in zip(val_dirs, expected_val):
            assert a == b
        for a, b in zip(train_dirs, expected_train):
            assert a == b

    def test_all_returns_everything(self, synthetic_sessions):
        session_dirs = sorted(synthetic_sessions)
        all_dirs = split_sessions(session_dirs, 0.2, 42, "all")
        assert len(all_dirs) == len(session_dirs)

    def test_no_overlap(self, synthetic_sessions):
        session_dirs = sorted(synthetic_sessions)
        train = split_sessions(session_dirs, 0.2, 42, "train")
        val = split_sessions(session_dirs, 0.2, 42, "val")

        train_set = set(str(d) for d in train)
        val_set = set(str(d) for d in val)
        assert len(train_set & val_set) == 0
        assert len(train_set | val_set) == len(session_dirs)


class TestDashboard:
    def test_create_dashboard(self, tmp_dir):
        """Dashboard creation with mock session results."""
        n = 20
        session_results = []
        for s in range(2):
            session_results.append({
                "name": f"session_{s:03d}",
                "position_mse": 0.01 * (s + 1),
                "rotation_error_deg": 5.0 * (s + 1),
                "joint_error_deg": 3.0 * (s + 1),
                "num_windows": n,
                "pred_actions_all": np.random.randn(n, 16, 11).astype(np.float32) * 0.1,
                "gt_actions_all": np.random.randn(n, 16, 11).astype(np.float32) * 0.1,
                "eval_indices": np.arange(n) * 10,
                "position_errors": np.random.exponential(0.01, n),
                "rotation_errors": np.random.exponential(5.0, n),
                "joint_errors": np.random.exponential(3.0, n),
            })

        out = str(tmp_dir / "dashboard.png")
        fig = create_dashboard(session_results, out, best_idx=0)
        plt.close(fig)

        assert Path(out).exists()
        assert Path(out).stat().st_size > 0
