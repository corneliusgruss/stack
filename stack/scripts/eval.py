"""
Evaluation script for trained diffusion policies.

Loads a checkpoint, runs prediction on held-out episodes, and computes
per-dimension error metrics (position, rotation, joints).

Usage:
    python -m stack.scripts.eval --checkpoint outputs/checkpoint_best.pt --data-dir data/raw/synthetic
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from stack.policy.diffusion import DiffusionPolicy, PolicyConfig
from stack.data.iphone_loader import iPhoneSession
from stack.data.training_dataset import StackDiffusionDataset, NormalizationStats


def quaternion_angular_error(q_pred: np.ndarray, q_true: np.ndarray) -> np.ndarray:
    """Compute angular error (degrees) between quaternion arrays.

    Args:
        q_pred: (..., 4) predicted quaternions [qx, qy, qz, qw]
        q_true: (..., 4) ground truth quaternions

    Returns:
        (...,) angular errors in degrees
    """
    # Flatten for batch processing
    shape = q_pred.shape[:-1]
    q_pred_flat = q_pred.reshape(-1, 4)
    q_true_flat = q_true.reshape(-1, 4)

    errors = np.zeros(q_pred_flat.shape[0])
    for i in range(len(errors)):
        r_pred = Rotation.from_quat(q_pred_flat[i])
        r_true = Rotation.from_quat(q_true_flat[i])
        r_diff = r_true.inv() * r_pred
        errors[i] = r_diff.magnitude() * 180.0 / np.pi

    return errors.reshape(shape)


def evaluate_episode(
    policy: DiffusionPolicy,
    session: iPhoneSession,
    stats: NormalizationStats,
    obs_horizon: int,
    action_horizon: int,
    image_size: int,
    device: torch.device,
) -> dict:
    """Evaluate policy on a single episode.

    Slides observation window through the episode, predicts action chunks,
    and compares to ground truth.
    """
    from stack.data.training_dataset import IMAGENET_MEAN, IMAGENET_STD
    from PIL import Image as PILImage

    episode = session.get_episode_12d()  # (T, 12)
    T = len(episode)

    if T < obs_horizon + action_horizon:
        return {"skipped": True, "reason": "too short"}

    all_pos_errors = []
    all_rot_errors = []
    all_joint_errors = []

    num_windows = T - obs_horizon - action_horizon + 1

    for t in range(obs_horizon - 1, T - action_horizon):
        obs_start = t - obs_horizon + 1

        # Build observation images
        obs_images = np.zeros((1, obs_horizon, 3, image_size, image_size), dtype=np.float32)
        for i in range(obs_horizon):
            img = session.get_rgb_frame(obs_start + i)
            pil_img = PILImage.fromarray(img)
            pil_img = pil_img.resize((image_size, image_size), PILImage.BILINEAR)
            arr = np.array(pil_img, dtype=np.float32) / 255.0
            arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
            obs_images[0, i] = arr.transpose(2, 0, 1)

        # Build observation proprio
        obs_proprio = episode[obs_start:obs_start + obs_horizon].copy()
        obs_proprio = stats.normalize_proprio(obs_proprio)
        obs_proprio = obs_proprio[np.newaxis]  # (1, obs_horizon, 12)

        # Predict
        images_t = torch.from_numpy(obs_images).to(device)
        proprio_t = torch.from_numpy(obs_proprio).to(device)
        pred_actions_norm = policy.predict(images_t, proprio_t)  # (1, action_horizon, 11)
        pred_actions_norm = pred_actions_norm.cpu().numpy()[0]  # (action_horizon, 11)

        # Unnormalize
        pred_actions = stats.unnormalize_action(pred_actions_norm)

        # Ground truth actions
        gt_actions = episode[t + 1:t + 1 + action_horizon, :11]

        # Position error (first 3 dims)
        pos_error = np.linalg.norm(pred_actions[:, :3] - gt_actions[:, :3], axis=-1)
        all_pos_errors.append(pos_error.mean())

        # Rotation error (dims 3:7, quaternion)
        rot_error = quaternion_angular_error(pred_actions[:, 3:7], gt_actions[:, 3:7])
        all_rot_errors.append(rot_error.mean())

        # Joint error (dims 7:11)
        joint_error = np.abs(pred_actions[:, 7:11] - gt_actions[:, 7:11])
        all_joint_errors.append(joint_error.mean())

    return {
        "skipped": False,
        "num_windows": num_windows,
        "position_mse": float(np.mean(all_pos_errors)),
        "rotation_error_deg": float(np.mean(all_rot_errors)),
        "joint_error_deg": float(np.mean(all_joint_errors)),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained diffusion policy")
    parser.add_argument("--checkpoint", required=True, help="Model checkpoint path")
    parser.add_argument("--data-dir", required=True, help="Session data directory")
    parser.add_argument("--device", default=None, help="Device (cuda/mps/cpu)")
    parser.add_argument("--max-episodes", type=int, default=None, help="Max episodes to eval")
    parser.add_argument("--output", help="Save results to .npz file")
    args = parser.parse_args()

    # Device
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = PolicyConfig(**checkpoint["config"])

    # Load EMA model if available, otherwise use regular model
    policy = DiffusionPolicy(config).to(device)
    if "ema_model" in checkpoint:
        policy.load_state_dict(checkpoint["ema_model"])
        print("Using EMA model weights")
    else:
        policy.load_state_dict(checkpoint["model"])
    policy.eval()

    # Load normalizer
    if "normalizer" in checkpoint:
        stats = NormalizationStats.from_state_dict(checkpoint["normalizer"])
    else:
        print("ERROR: No normalizer stats in checkpoint. Cannot evaluate.")
        return

    print(f"Loaded: {args.checkpoint}")
    if "epoch" in checkpoint:
        print(f"Epoch: {checkpoint['epoch'] + 1}")
    if "val_loss" in checkpoint:
        print(f"Checkpoint val loss: {checkpoint['val_loss']:.4f}")

    # Find sessions
    data_dir = Path(args.data_dir)
    session_dirs = sorted(
        d for d in data_dir.iterdir()
        if d.is_dir() and not d.name.startswith(".") and (d / "poses.json").exists()
    )

    if args.max_episodes:
        session_dirs = session_dirs[:args.max_episodes]

    print(f"\nEvaluating on {len(session_dirs)} episodes...")
    print("=" * 60)

    results = []
    for i, session_dir in enumerate(session_dirs):
        session = iPhoneSession(session_dir)
        result = evaluate_episode(
            policy, session, stats,
            obs_horizon=config.obs_horizon,
            action_horizon=config.action_horizon,
            image_size=config.image_size,
            device=device,
        )

        if result["skipped"]:
            print(f"  [{i+1}/{len(session_dirs)}] {session_dir.name}: SKIPPED ({result['reason']})")
            continue

        results.append(result)
        print(
            f"  [{i+1}/{len(session_dirs)}] {session_dir.name}: "
            f"pos={result['position_mse']:.4f}m  "
            f"rot={result['rotation_error_deg']:.1f}°  "
            f"joints={result['joint_error_deg']:.1f}°"
        )

    if not results:
        print("\nNo episodes evaluated.")
        return

    # Aggregate
    print("\n" + "=" * 60)
    print("SUMMARY")
    print(f"  Episodes evaluated: {len(results)}")
    print(f"  Position MSE:       {np.mean([r['position_mse'] for r in results]):.4f} m")
    print(f"  Rotation error:     {np.mean([r['rotation_error_deg'] for r in results]):.1f}°")
    print(f"  Joint error:        {np.mean([r['joint_error_deg'] for r in results]):.1f}°")

    if args.output:
        np.savez(args.output, results=results)
        print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
