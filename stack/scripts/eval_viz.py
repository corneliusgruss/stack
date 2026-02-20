"""
Evaluation + Visualization CLI for diffusion policy.

Orchestrates evaluation on train/val sessions and produces a full set of
visualization artifacts including a dashboard PNG for portfolio use.

Usage:
    python -m stack.scripts.eval_viz \
        --checkpoint outputs/real_v1/checkpoint_best.pt \
        --data-dir data/raw \
        --output-dir outputs/real_v1/eval \
        --device mps \
        --eval-stride 10 \
        --split val \
        --video
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from stack.policy.diffusion import DiffusionPolicy, PolicyConfig
from stack.data.iphone_loader import iPhoneSession
from stack.data.training_dataset import NormalizationStats
from stack.scripts.eval import (
    evaluate_episode,
    find_session_dirs,
    split_sessions,
)
from stack.viz.eval_viz import (
    plot_trajectory_comparison_3d,
    plot_position_over_time,
    plot_joints_over_time,
    plot_error_distribution,
    plot_per_session_metrics,
    render_prediction_video,
    create_dashboard,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description="Evaluate and visualize diffusion policy")
    parser.add_argument("--checkpoint", required=True, help="Model checkpoint path")
    parser.add_argument("--data-dir", required=True, help="Session data directory")
    parser.add_argument("--output-dir", default="outputs/eval", help="Output directory for artifacts")
    parser.add_argument("--device", default=None, help="Device (cuda/mps/cpu)")
    parser.add_argument("--eval-stride", type=int, default=10, help="Evaluate every Nth timestep")
    parser.add_argument("--val-split", type=float, default=0.2, help="Validation split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for split")
    parser.add_argument("--split", default="val", choices=["train", "val", "all"],
                        help="Which split to evaluate")
    parser.add_argument("--video", action="store_true", help="Render overlay videos (slower)")
    parser.add_argument("--max-episodes", type=int, default=None, help="Max episodes to eval")
    parser.add_argument("--wandb", action="store_true", help="Log eval viz results to wandb")
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

    policy = DiffusionPolicy(config).to(device)
    if "ema_model" in checkpoint:
        policy.load_state_dict(checkpoint["ema_model"])
        print("Using EMA model weights")
    else:
        policy.load_state_dict(checkpoint["model"])
    policy.eval()

    if "normalizer" not in checkpoint:
        print("ERROR: No normalizer stats in checkpoint.")
        return
    stats = NormalizationStats.from_state_dict(checkpoint["normalizer"])

    print(f"Loaded: {args.checkpoint}")
    if "epoch" in checkpoint:
        print(f"Epoch: {checkpoint['epoch'] + 1}")
    if "val_loss" in checkpoint:
        print(f"Val loss: {checkpoint['val_loss']:.4f}")

    # Initialize wandb
    if use_wandb:
        wandb.init(
            project="stack",
            job_type="eval_viz",
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

    print(f"\nEvaluating {len(session_dirs)} sessions (split={args.split}, stride={args.eval_stride})...")

    # Output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # === Step 1: Evaluate all sessions ===
    session_results = []
    for i, session_dir in enumerate(tqdm(session_dirs, desc="Evaluating")):
        session = iPhoneSession(session_dir)
        result = evaluate_episode(
            policy, session, stats,
            obs_horizon=config.obs_horizon,
            action_horizon=config.action_horizon,
            image_size=config.image_size,
            device=device,
            eval_stride=args.eval_stride,
            return_trajectories=True,
        )

        if result["skipped"]:
            print(f"  {session_dir.name}: SKIPPED ({result['reason']})")
            continue

        result["name"] = session_dir.name
        result["session_dir"] = str(session_dir)
        session_results.append(result)

        print(
            f"  [{i+1}/{len(session_dirs)}] {session_dir.name}: "
            f"pos={result['position_mse']:.4f}m  "
            f"rot={result['rotation_error_deg']:.1f}deg  "
            f"joints={result['joint_error_deg']:.1f}deg  "
            f"({result['num_windows']} windows)"
        )

    if not session_results:
        print("No sessions evaluated. Check data directory and split settings.")
        if use_wandb:
            wandb.finish()
        return

    # === Step 2: Per-session plots ===
    print("\nGenerating per-session plots...")
    for idx, result in enumerate(session_results):
        name = result["name"]
        session_out = output_dir / f"session_{idx:03d}"
        session_out.mkdir(parents=True, exist_ok=True)

        # 1-step predictions (first action of each chunk)
        pred_pos = result["pred_actions_all"][:, 0, :3]
        gt_pos = result["gt_actions_all"][:, 0, :3]
        pred_joints = result["pred_actions_all"][:, 0, 7:11]
        gt_joints = result["gt_actions_all"][:, 0, 7:11]
        eval_idx = result["eval_indices"]

        # 3D trajectory
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")
        plot_trajectory_comparison_3d(
            pred_pos, gt_pos,
            title=f"Trajectory: {name}",
            save_path=str(session_out / "trajectory_3d.png"),
            ax=ax,
        )
        plt.close(fig)

        # Position over time
        fig = plot_position_over_time(
            pred_pos, gt_pos, eval_idx,
            title=f"Position: {name}",
            save_path=str(session_out / "position_over_time.png"),
        )
        plt.close(fig)

        # Joints over time
        fig = plot_joints_over_time(
            pred_joints, gt_joints, eval_idx,
            title=f"Joints: {name}",
            save_path=str(session_out / "joints_over_time.png"),
        )
        plt.close(fig)

        # Video overlay (optional)
        if args.video:
            print(f"  Rendering video for {name}...")
            session = iPhoneSession(result["session_dir"])
            render_prediction_video(
                session,
                result["pred_actions_all"],
                result["gt_actions_all"],
                result["eval_indices"],
                str(session_out / "prediction_overlay.mp4"),
                fps=10,
            )

    # === Step 3: Aggregate plots ===
    print("Generating aggregate plots...")

    # Error distributions
    all_pos = np.concatenate([r["position_errors"] for r in session_results])
    all_rot = np.concatenate([r["rotation_errors"] for r in session_results])
    all_joints = np.concatenate([r["joint_errors"] for r in session_results])

    fig = plot_error_distribution(
        {"position": all_pos, "rotation": all_rot, "joints": all_joints},
        save_path=str(output_dir / "error_distributions.png"),
    )
    plt.close(fig)

    # Per-session metrics bar chart
    fig = plot_per_session_metrics(
        session_results,
        save_path=str(output_dir / "per_session_metrics.png"),
    )
    plt.close(fig)

    # === Step 4: Dashboard ===
    print("Creating dashboard...")
    # Pick best session (lowest position error)
    best_idx = int(np.argmin([r["position_mse"] for r in session_results]))

    fig = create_dashboard(
        session_results,
        str(output_dir / "dashboard.png"),
        best_idx=best_idx,
    )
    plt.close(fig)

    # === Step 5: Save metrics JSON ===
    metrics = {
        "checkpoint": str(args.checkpoint),
        "split": args.split,
        "eval_stride": args.eval_stride,
        "num_sessions": len(session_results),
        "aggregate": {
            "position_mse_m": float(np.mean([r["position_mse"] for r in session_results])),
            "rotation_error_deg": float(np.mean([r["rotation_error_deg"] for r in session_results])),
            "joint_error_deg": float(np.mean([r["joint_error_deg"] for r in session_results])),
        },
        "per_session": [
            {
                "name": r["name"],
                "num_windows": r["num_windows"],
                "position_mse_m": r["position_mse"],
                "rotation_error_deg": r["rotation_error_deg"],
                "joint_error_deg": r["joint_error_deg"],
            }
            for r in session_results
        ],
    }
    if "epoch" in checkpoint:
        metrics["epoch"] = checkpoint["epoch"] + 1
    if "val_loss" in checkpoint:
        metrics["val_loss"] = float(checkpoint["val_loss"])

    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    # === Step 6: Upload to wandb ===
    if use_wandb:
        wandb_images = {
            "eval_viz/dashboard": wandb.Image(str(output_dir / "dashboard.png")),
            "eval_viz/error_distributions": wandb.Image(str(output_dir / "error_distributions.png")),
            "eval_viz/per_session_metrics": wandb.Image(str(output_dir / "per_session_metrics.png")),
        }

        # Upload best session trajectory
        best_dir = output_dir / f"session_{best_idx:03d}"
        if (best_dir / "trajectory_3d.png").exists():
            wandb_images["eval_viz/best_trajectory_3d"] = wandb.Image(str(best_dir / "trajectory_3d.png"))
        if (best_dir / "position_over_time.png").exists():
            wandb_images["eval_viz/best_position_over_time"] = wandb.Image(str(best_dir / "position_over_time.png"))
        if (best_dir / "joints_over_time.png").exists():
            wandb_images["eval_viz/best_joints_over_time"] = wandb.Image(str(best_dir / "joints_over_time.png"))

        wandb.log(wandb_images)

        # Summary metrics
        wandb.run.summary["eval_viz/position_mse_m"] = metrics["aggregate"]["position_mse_m"]
        wandb.run.summary["eval_viz/rotation_error_deg"] = metrics["aggregate"]["rotation_error_deg"]
        wandb.run.summary["eval_viz/joint_error_deg"] = metrics["aggregate"]["joint_error_deg"]
        wandb.run.summary["eval_viz/num_sessions"] = len(session_results)

        wandb.finish()

    # === Summary ===
    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)
    print(f"  Sessions: {len(session_results)}")
    print(f"  Position error:  {metrics['aggregate']['position_mse_m']:.4f} m")
    print(f"  Rotation error:  {metrics['aggregate']['rotation_error_deg']:.1f} deg")
    print(f"  Joint error:     {metrics['aggregate']['joint_error_deg']:.1f} deg")
    print(f"\nArtifacts saved to: {output_dir}")
    print(f"  dashboard.png              <- portfolio screenshot")
    print(f"  error_distributions.png")
    print(f"  per_session_metrics.png")
    print(f"  metrics.json")
    for idx in range(len(session_results)):
        print(f"  session_{idx:03d}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
