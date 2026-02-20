"""
Training script for diffusion policy.

Usage:
    python -m stack.scripts.train --config configs/default.yaml --data-dir data/raw/synthetic
    python -m stack.scripts.train --config configs/default.yaml --data-dir data/raw/synthetic --wandb
"""

import argparse
import copy
import math
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    from omegaconf import OmegaConf
    OMEGACONF_AVAILABLE = True
except ImportError:
    OMEGACONF_AVAILABLE = False

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from stack.policy.diffusion import DiffusionPolicy, PolicyConfig
from stack.data.training_dataset import StackDiffusionDataset, NormalizationStats


def get_device(requested: str | None = None) -> torch.device:
    """Pick best available device: cuda > mps > cpu."""
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def find_session_dirs(data_dir: Path) -> list[Path]:
    """Find all session directories under data_dir."""
    sessions = []
    for d in sorted(data_dir.iterdir()):
        if d.is_dir() and not d.name.startswith("."):
            # Must have poses.json to be a valid session
            if (d / "poses.json").exists():
                sessions.append(d)
    return sessions


@torch.no_grad()
def ema_update(ema_model: DiffusionPolicy, model: DiffusionPolicy, decay: float):
    """Update EMA model parameters."""
    for ema_p, p in zip(ema_model.parameters(), model.parameters()):
        ema_p.data.mul_(decay).add_(p.data, alpha=1.0 - decay)


@torch.no_grad()
def validate(
    policy: DiffusionPolicy,
    val_loader: DataLoader,
    device: torch.device,
) -> float:
    """Compute mean validation loss."""
    policy.eval()
    total_loss = 0.0
    count = 0
    for images, proprio, actions in val_loader:
        images = images.to(device)
        proprio = proprio.to(device)
        actions = actions.to(device)
        loss_dict = policy(images, proprio, actions)
        total_loss += loss_dict["loss"].item() * images.shape[0]
        count += images.shape[0]
    return total_loss / max(count, 1)


@torch.no_grad()
def log_sample_predictions(
    policy: DiffusionPolicy,
    val_dataset: StackDiffusionDataset,
    stats: NormalizationStats,
    device: torch.device,
    global_step: int,
    num_samples: int = 4,
):
    """Run DDPM inference on random val samples and log predicted vs GT action plots."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    policy.eval()
    rng = np.random.default_rng()
    indices = rng.choice(len(val_dataset), size=min(num_samples, len(val_dataset)), replace=False)

    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 3 * num_samples))
    if num_samples == 1:
        axes = axes[np.newaxis, :]

    dim_labels = ["Position (XYZ)", "Rotation (quat)", "Joints"]
    dim_slices = [slice(0, 3), slice(3, 7), slice(7, 11)]

    for row, idx in enumerate(indices):
        images, proprio, actions = val_dataset[idx]
        images_t = images.unsqueeze(0).to(device)
        proprio_t = proprio.unsqueeze(0).to(device)

        pred_norm = policy.predict(images_t, proprio_t).cpu().numpy()[0]  # (action_horizon, 11)
        pred = stats.unnormalize_action(pred_norm)
        gt = stats.unnormalize_action(actions.numpy())

        for col, (label, sl) in enumerate(zip(dim_labels, dim_slices)):
            ax = axes[row, col]
            gt_vals = gt[:, sl]
            pred_vals = pred[:, sl]
            t_axis = np.arange(gt_vals.shape[0])
            for d in range(gt_vals.shape[1]):
                ax.plot(t_axis, gt_vals[:, d], "C0-", alpha=0.6, linewidth=1)
                ax.plot(t_axis, pred_vals[:, d], "C1--", alpha=0.6, linewidth=1)
            if row == 0:
                ax.set_title(label)
            if col == 0:
                ax.set_ylabel(f"Sample {row}")
            ax.set_xlabel("t")
            ax.tick_params(labelsize=7)

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color="C0", label="Ground Truth"),
        Line2D([0], [0], color="C1", linestyle="--", label="Predicted"),
    ]
    fig.legend(handles=legend_elements, loc="upper right", fontsize=9)
    fig.suptitle("Sample Predictions (val set)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    wandb.log({"val/sample_predictions": wandb.Image(fig)}, step=global_step)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Train diffusion policy")
    parser.add_argument("--config", default="configs/default.yaml", help="Config file")
    parser.add_argument("--data-dir", default="data/raw/synthetic", help="Session data directory")
    parser.add_argument("--output-dir", default="outputs", help="Output directory")
    parser.add_argument("--device", default=None, help="Device (cuda/mps/cpu)")
    parser.add_argument("--wandb", action="store_true", help="Enable W&B logging")
    parser.add_argument("--resume", help="Resume from checkpoint")
    parser.add_argument("--val-split", type=float, default=0.2, help="Validation split ratio")
    args = parser.parse_args()

    # Load config
    if OMEGACONF_AVAILABLE and Path(args.config).exists():
        cfg = OmegaConf.load(args.config)
        # Filter to only PolicyConfig fields
        import dataclasses
        valid_fields = {f.name for f in dataclasses.fields(PolicyConfig)}
        policy_kwargs = {k: v for k, v in cfg.policy.items() if k in valid_fields}
        policy_config = PolicyConfig(**policy_kwargs)
        training_cfg = cfg.get("training", {})
    else:
        print("Using default config")
        policy_config = PolicyConfig()
        training_cfg = {}

    gradient_clip = float(training_cfg.get("gradient_clip", 1.0))
    ema_decay = float(training_cfg.get("ema_decay", 0.995))
    checkpoint_every = int(training_cfg.get("checkpoint_every", 10))
    eval_every = int(training_cfg.get("eval_every", 5))
    num_workers = int(training_cfg.get("num_workers", 0))
    seed = int(training_cfg.get("seed", 42))
    early_stopping_patience = int(training_cfg.get("early_stopping_patience", 0))

    data_cfg = cfg.get("data", {}) if OMEGACONF_AVAILABLE and Path(args.config).exists() else {}
    random_crop = bool(data_cfg.get("random_crop", True))
    color_jitter = bool(data_cfg.get("color_jitter", True))

    # Parse wandb config from YAML
    logging_cfg = cfg.get("logging", {}).get("wandb", {}) if OMEGACONF_AVAILABLE and Path(args.config).exists() else {}
    wandb_cfg_enabled = bool(logging_cfg.get("enabled", False))
    wandb_project = str(logging_cfg.get("project", "stack"))
    wandb_entity = logging_cfg.get("entity", None)
    log_images_every = int(logging_cfg.get("log_images_every", 10))
    log_artifacts = bool(logging_cfg.get("log_artifacts", True))

    # Seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Device
    device = get_device(args.device)
    print(f"Device: {device}")

    # Output
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find session directories
    data_dir = Path(args.data_dir)
    session_dirs = find_session_dirs(data_dir)
    if not session_dirs:
        print(f"ERROR: No sessions found in {data_dir}")
        print("Run: python -m stack.data.synthetic --output data/raw/synthetic")
        return

    print(f"Found {len(session_dirs)} sessions in {data_dir}")

    # Train/val split
    n_val = max(1, int(len(session_dirs) * args.val_split))
    n_train = len(session_dirs) - n_val
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(session_dirs))
    train_dirs = [session_dirs[i] for i in indices[:n_train]]
    val_dirs = [session_dirs[i] for i in indices[n_train:]]
    print(f"Train: {n_train} sessions, Val: {n_val} sessions")

    # Create datasets
    train_dataset = StackDiffusionDataset(
        train_dirs,
        obs_horizon=policy_config.obs_horizon,
        action_horizon=policy_config.action_horizon,
        image_size=policy_config.image_size,
        action_dim=policy_config.action_dim,
        augment=True,
        random_crop=random_crop,
        color_jitter=color_jitter,
    )
    val_dataset = StackDiffusionDataset(
        val_dirs,
        obs_horizon=policy_config.obs_horizon,
        action_horizon=policy_config.action_horizon,
        image_size=policy_config.image_size,
        action_dim=policy_config.action_dim,
        stats=train_dataset.stats,
    )
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    if len(train_dataset) == 0:
        print("ERROR: No valid training samples. Check session data.")
        return

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=policy_config.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=policy_config.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # Model
    policy = DiffusionPolicy(policy_config).to(device)
    ema_policy = copy.deepcopy(policy)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=policy_config.learning_rate)

    # Cosine LR schedule
    total_steps = len(train_loader) * policy_config.num_epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=1e-6
    )

    # Resume
    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        policy.load_state_dict(checkpoint["model"])
        ema_policy.load_state_dict(checkpoint.get("ema_model", checkpoint["model"]))
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"] + 1
        if "scheduler" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler"])
        print(f"Resumed from epoch {start_epoch}")

    # W&B
    use_wandb = (args.wandb or wandb_cfg_enabled) and WANDB_AVAILABLE
    if use_wandb:
        param_count = sum(p.numel() for p in policy.parameters())
        trainable_count = sum(p.numel() for p in policy.parameters() if p.requires_grad)

        # Hardware info
        hw_config = {"hardware/device": str(device)}
        if device.type == "cuda":
            hw_config["hardware/gpu_name"] = torch.cuda.get_device_name(device)
            hw_config["hardware/gpu_memory_gb"] = round(
                torch.cuda.get_device_properties(device).total_mem / 1e9, 1
            )

        wandb.init(
            project=wandb_project,
            entity=wandb_entity if wandb_entity else None,
            config={
                # Policy config
                **{f"policy/{k}": v for k, v in vars(policy_config).items()},
                # Training hyperparams
                "training/gradient_clip": gradient_clip,
                "training/ema_decay": ema_decay,
                "training/eval_every": eval_every,
                "training/seed": seed,
                "training/early_stopping_patience": early_stopping_patience,
                "training/random_crop": random_crop,
                "training/color_jitter": color_jitter,
                # Model info
                "model/param_count_total": param_count,
                "model/param_count_trainable": trainable_count,
                # Dataset stats
                "dataset/num_train_sessions": n_train,
                "dataset/num_val_sessions": n_val,
                "dataset/total_train_samples": len(train_dataset),
                "dataset/total_val_samples": len(val_dataset),
                "dataset/batches_per_epoch": len(train_loader),
                "dataset/total_steps": total_steps,
                # Normalization stats (for reproducibility)
                "dataset/proprio_mean": train_dataset.stats.proprio_mean.tolist(),
                "dataset/proprio_std": train_dataset.stats.proprio_std.tolist(),
                "dataset/action_min": train_dataset.stats.action_min.tolist(),
                "dataset/action_max": train_dataset.stats.action_max.tolist(),
                # Hardware
                **hw_config,
            },
        )

    # Save config and normalizer
    torch.save(train_dataset.stats.state_dict(), output_dir / "normalizer.pt")

    param_count = sum(p.numel() for p in policy.parameters())
    trainable_count = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    print(f"\nModel parameters: {param_count:,} total, {trainable_count:,} trainable")
    print(f"Batches/epoch: {len(train_loader)}")
    print(f"Total steps: {total_steps}")

    # Training loop
    print("\nTraining...")
    print("=" * 60)

    global_step = start_epoch * len(train_loader)
    best_val_loss = float("inf")
    best_val_epoch = -1
    evals_without_improvement = 0
    early_stopped = False

    for epoch in range(start_epoch, policy_config.num_epochs):
        policy.train()
        epoch_loss = 0.0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{policy_config.num_epochs}")
        for images, proprio, actions in pbar:
            images = images.to(device)
            proprio = proprio.to(device)
            actions = actions.to(device)

            loss_dict = policy(images, proprio, actions)
            loss = loss_dict["loss"]

            optimizer.zero_grad()
            loss.backward()
            grad_norm = clip_grad_norm_(policy.parameters(), gradient_clip)
            optimizer.step()
            scheduler.step()

            # EMA update
            ema_update(ema_policy, policy, ema_decay)

            epoch_loss += loss.item()
            num_batches += 1
            global_step += 1

            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                lr=f"{scheduler.get_last_lr()[0]:.2e}",
                grad=f"{grad_norm:.2f}",
            )

            if use_wandb:
                wandb.log({
                    "train/loss": loss.item(),
                    "train/grad_norm": grad_norm,
                    "train/lr": scheduler.get_last_lr()[0],
                }, step=global_step)

        avg_train_loss = epoch_loss / max(num_batches, 1)
        print(f"  Epoch {epoch+1} avg train loss: {avg_train_loss:.4f}")

        if use_wandb:
            wandb.log({"train/epoch_loss": avg_train_loss, "epoch": epoch + 1}, step=global_step)

        # Validation
        if (epoch + 1) % eval_every == 0 and len(val_dataset) > 0:
            val_loss = validate(ema_policy, val_loader, device)
            print(f"  Val loss (EMA): {val_loss:.4f}")

            if use_wandb:
                wandb.log({
                    "val/loss": val_loss,
                    "val/best_loss": min(best_val_loss, val_loss),
                }, step=global_step)

            # Sample prediction images
            if use_wandb and log_images_every > 0 and (epoch + 1) % log_images_every == 0:
                log_sample_predictions(
                    ema_policy, val_dataset, train_dataset.stats,
                    device, global_step,
                )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_epoch = epoch + 1
                evals_without_improvement = 0
                best_path = output_dir / "checkpoint_best.pt"
                torch.save({
                    "epoch": epoch,
                    "model": policy.state_dict(),
                    "ema_model": ema_policy.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "config": vars(policy_config),
                    "normalizer": train_dataset.stats.state_dict(),
                    "val_loss": val_loss,
                }, best_path)
                print(f"  New best! Saved: {best_path}")

                if use_wandb:
                    wandb.run.summary["best_val_loss"] = best_val_loss
                    wandb.run.summary["best_val_epoch"] = best_val_epoch

                    if log_artifacts:
                        artifact = wandb.Artifact(
                            "checkpoint-best", type="model",
                            metadata={"epoch": epoch + 1, "val_loss": val_loss},
                        )
                        artifact.add_file(str(best_path))
                        wandb.log_artifact(artifact)
            else:
                evals_without_improvement += 1
                if early_stopping_patience > 0:
                    print(f"  No improvement ({evals_without_improvement}/{early_stopping_patience})")
                    if evals_without_improvement >= early_stopping_patience:
                        print(f"\nEarly stopping at epoch {epoch + 1}")
                        early_stopped = True
                        break

        # Periodic checkpoint
        if (epoch + 1) % checkpoint_every == 0:
            ckpt_path = output_dir / f"checkpoint_{epoch+1:04d}.pt"
            torch.save({
                "epoch": epoch,
                "model": policy.state_dict(),
                "ema_model": ema_policy.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "config": vars(policy_config),
                "normalizer": train_dataset.stats.state_dict(),
                "train_loss": avg_train_loss,
            }, ckpt_path)
            print(f"  Saved: {ckpt_path}")

    # Save final checkpoint
    final_path = output_dir / "checkpoint_final.pt"
    torch.save({
        "epoch": policy_config.num_epochs - 1,
        "model": policy.state_dict(),
        "ema_model": ema_policy.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "config": vars(policy_config),
        "normalizer": train_dataset.stats.state_dict(),
    }, final_path)

    print("\n" + "=" * 60)
    print(f"Training complete! Final checkpoint: {final_path}")
    if early_stopped:
        print(f"(Stopped early at epoch {epoch + 1} of {policy_config.num_epochs})")
    if best_val_loss < float("inf"):
        print(f"Best val loss: {best_val_loss:.4f}")

    if use_wandb:
        wandb.run.summary["final_train_loss"] = avg_train_loss
        wandb.run.summary["total_epochs"] = epoch + 1
        wandb.run.summary["best_val_loss"] = best_val_loss
        wandb.run.summary["best_val_epoch"] = best_val_epoch
        wandb.run.summary["early_stopped"] = early_stopped
        wandb.finish()


if __name__ == "__main__":
    main()
