"""
Evaluation script for trained diffusion policies.

Loads a checkpoint, runs prediction on held-out episodes, and computes
per-dimension error metrics (position, rotation, joints).

Usage:
    python -m stack.scripts.eval --checkpoint outputs/checkpoint_best.pt --data-dir data/raw/synthetic
    python -m stack.scripts.eval --checkpoint outputs/checkpoint_best.pt --data-dir data/raw --split val
    python -m stack.scripts.eval --checkpoint outputs/checkpoint_best.pt --data-dir data/raw --wandb
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

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


def find_session_dirs(data_dir: Path) -> list[Path]:
    """Find all session directories under data_dir."""
    return sorted(
        d for d in data_dir.iterdir()
        if d.is_dir() and not d.name.startswith(".") and (d / "poses.json").exists()
    )


def split_sessions(
    session_dirs: list[Path],
    val_split: float = 0.2,
    seed: int = 42,
    split: str = "all",
) -> list[Path]:
    """Split sessions into train/val matching train.py logic.

    Args:
        session_dirs: All session directories
        val_split: Fraction held out for validation
        seed: Random seed for split (must match train.py)
        split: Which split to return: 'train', 'val', or 'all'

    Returns:
        List of session directories for the requested split
    """
    if split == "all":
        return session_dirs

    n_val = max(1, int(len(session_dirs) * val_split))
    n_train = len(session_dirs) - n_val
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(session_dirs))
    train_dirs = [session_dirs[i] for i in indices[:n_train]]
    val_dirs = [session_dirs[i] for i in indices[n_train:]]

    if split == "val":
        return val_dirs
    elif split == "train":
        return train_dirs
    else:
        raise ValueError(f"Unknown split: {split!r}. Use 'train', 'val', or 'all'.")


def evaluate_episode(
    policy: DiffusionPolicy,
    session: iPhoneSession,
    stats: NormalizationStats,
    obs_horizon: int,
    action_horizon: int,
    image_size: int,
    device: torch.device,
    eval_stride: int = 1,
    return_trajectories: bool = False,
) -> dict:
    """Evaluate policy on a single episode.

    Slides observation window through the episode, predicts action chunks,
    and compares to ground truth.

    Args:
        eval_stride: Evaluate every Nth timestep (default 1 = all)
        return_trajectories: If True, also return per-timestep predictions/GT arrays
    """
    from stack.data.training_dataset import IMAGENET_MEAN, IMAGENET_STD
    from PIL import Image as PILImage

    episode = session.get_episode_11d()  # (T, 11)
    T = len(episode)

    if T < obs_horizon + action_horizon:
        return {"skipped": True, "reason": "too short"}

    all_pos_errors = []
    all_rot_errors = []
    all_joint_errors = []

    # Trajectory storage
    pred_actions_list = []
    gt_actions_list = []
    eval_indices_list = []

    for t in range(obs_horizon - 1, T - action_horizon, eval_stride):
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
        obs_proprio = obs_proprio[np.newaxis]  # (1, obs_horizon, 11)

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

        if return_trajectories:
            pred_actions_list.append(pred_actions)
            gt_actions_list.append(gt_actions)
            eval_indices_list.append(t)

    num_windows = len(all_pos_errors)
    result = {
        "skipped": False,
        "num_windows": num_windows,
        "position_mse": float(np.mean(all_pos_errors)),
        "rotation_error_deg": float(np.mean(all_rot_errors)),
        "joint_error_deg": float(np.mean(all_joint_errors)),
        "position_errors": np.array(all_pos_errors),
        "rotation_errors": np.array(all_rot_errors),
        "joint_errors": np.array(all_joint_errors),
    }

    if return_trajectories:
        result["pred_actions_all"] = np.stack(pred_actions_list)  # (num_windows, action_horizon, 11)
        result["gt_actions_all"] = np.stack(gt_actions_list)      # (num_windows, action_horizon, 11)
        result["eval_indices"] = np.array(eval_indices_list)       # (num_windows,)

    return result


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained diffusion policy")
    parser.add_argument("--checkpoint", required=True, help="Model checkpoint path")
    parser.add_argument("--data-dir", required=True, help="Session data directory")
    parser.add_argument("--device", default=None, help="Device (cuda/mps/cpu)")
    parser.add_argument("--max-episodes", type=int, default=None, help="Max episodes to eval")
    parser.add_argument("--output", help="Save results to .npz file")
    parser.add_argument("--eval-stride", type=int, default=1, help="Evaluate every Nth timestep")
    parser.add_argument("--val-split", type=float, default=0.2, help="Validation split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for split")
    parser.add_argument("--split", default="all", choices=["train", "val", "all"],
                        help="Which split to evaluate")
    parser.add_argument("--save-trajectories", action="store_true",
                        help="Save per-timestep prediction data")
    parser.add_argument("--wandb", action="store_true", help="Log eval results to wandb")
    parser.add_argument("--wandb-run-id", default=None, help="Resume existing wandb run")
    args = parser.parse_args()

    use_wandb = args.wandb and WANDB_AVAILABLE

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

    # Initialize wandb
    if use_wandb:
        wandb.init(
            project="stack",
            job_type="eval",
            config={
                "checkpoint": str(args.checkpoint),
                "split": args.split,
                "eval_stride": args.eval_stride,
                **{f"policy/{k}": v for k, v in checkpoint.get("config", {}).items()},
            },
            id=args.wandb_run_id,
            resume="must" if args.wandb_run_id else None,
        )

    # Find and split sessions
    data_dir = Path(args.data_dir)
    all_session_dirs = find_session_dirs(data_dir)
    session_dirs = split_sessions(all_session_dirs, args.val_split, args.seed, args.split)

    if args.max_episodes:
        session_dirs = session_dirs[:args.max_episodes]

    # Check calibration status
    import json as _json
    calibrated_count = 0
    uncalibrated_count = 0
    for sd in session_dirs:
        meta_file = sd / "metadata.json"
        if meta_file.exists():
            with open(meta_file) as f:
                meta = _json.load(f)
            if meta.get("scale_factor") is not None:
                calibrated_count += 1
            else:
                uncalibrated_count += 1

    pos_unit = "m" if uncalibrated_count == 0 and calibrated_count > 0 else "COLMAP units"
    if uncalibrated_count > 0:
        print(f"\nWARNING: {uncalibrated_count}/{len(session_dirs)} sessions not scale-calibrated.")
        print(f"  Position errors are in COLMAP arbitrary units, not meters.")
        print(f"  Run: python -m stack.scripts.calibrate_scale --data-dir {args.data_dir}")

    print(f"\nEvaluating on {len(session_dirs)} episodes (split={args.split}, stride={args.eval_stride})...")
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
            eval_stride=args.eval_stride,
            return_trajectories=args.save_trajectories,
        )

        if result["skipped"]:
            print(f"  [{i+1}/{len(session_dirs)}] {session_dir.name}: SKIPPED ({result['reason']})")
            continue

        results.append({**result, "name": session_dir.name})
        print(
            f"  [{i+1}/{len(session_dirs)}] {session_dir.name}: "
            f"pos={result['position_mse']:.4f} {pos_unit}  "
            f"rot={result['rotation_error_deg']:.1f}deg  "
            f"joints={result['joint_error_deg']:.1f}deg"
        )

    if not results:
        print("\nNo episodes evaluated.")
        if use_wandb:
            wandb.finish()
        return

    # Aggregate
    agg_pos = float(np.mean([r["position_mse"] for r in results]))
    agg_rot = float(np.mean([r["rotation_error_deg"] for r in results]))
    agg_joint = float(np.mean([r["joint_error_deg"] for r in results]))

    print("\n" + "=" * 60)
    print("SUMMARY")
    print(f"  Episodes evaluated: {len(results)}")
    print(f"  Position MSE:       {agg_pos:.4f} {pos_unit}")
    print(f"  Rotation error:     {agg_rot:.1f}deg")
    print(f"  Joint error:        {agg_joint:.1f}deg")

    if use_wandb:
        # Aggregate scalars
        wandb.log({
            "eval/position_mse": agg_pos,
            "eval/rotation_error_deg": agg_rot,
            "eval/joint_error_deg": agg_joint,
            "eval/num_episodes": len(results),
        })

        # Per-session table
        table = wandb.Table(columns=[
            "session", "num_windows", "position_mse", "rotation_error_deg", "joint_error_deg",
        ])
        for r in results:
            table.add_data(
                r["name"], r["num_windows"],
                r["position_mse"], r["rotation_error_deg"], r["joint_error_deg"],
            )
        wandb.log({"eval/per_session": table})

        # Summary
        wandb.run.summary["eval/position_mse"] = agg_pos
        wandb.run.summary["eval/rotation_error_deg"] = agg_rot
        wandb.run.summary["eval/joint_error_deg"] = agg_joint
        wandb.run.summary["eval/num_episodes"] = len(results)
        wandb.run.summary["eval/position_unit"] = pos_unit

        wandb.finish()

    if args.output:
        np.savez(args.output, results=results)
        print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
