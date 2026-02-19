"""
Evaluation visualization for diffusion policy.

Provides trajectory comparisons, time series plots, error distributions,
video overlays, and a composite dashboard for portfolio/course use.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colormaps as mpl_colormaps


# Joint names for labeling
JOINT_NAMES = ["index_mcp", "index_pip", "three_finger_mcp", "three_finger_pip"]


def plot_trajectory_comparison_3d(
    pred_positions: np.ndarray,
    gt_positions: np.ndarray,
    title: str = "Predicted vs Ground Truth Trajectory",
    save_path: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Plot predicted and GT 3D trajectories on same axes.

    Args:
        pred_positions: (N, 3) predicted XYZ positions
        gt_positions: (N, 3) ground truth XYZ positions
        title: Plot title
        save_path: If set, save figure to this path
        ax: Existing 3D axes (creates new if None)

    Returns:
        Matplotlib 3D axes
    """
    if ax is None:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")

    # GT: blue solid
    ax.plot(gt_positions[:, 0], gt_positions[:, 1], gt_positions[:, 2],
            "b-", linewidth=1.5, alpha=0.8, label="Ground Truth")
    # Predicted: red dashed
    ax.plot(pred_positions[:, 0], pred_positions[:, 1], pred_positions[:, 2],
            "r--", linewidth=1.5, alpha=0.8, label="Predicted")

    # Start/end markers
    ax.scatter(*gt_positions[0], c="green", s=100, marker="o", label="Start", zorder=5)
    ax.scatter(*gt_positions[-1], c="black", s=100, marker="x", label="End", zorder=5)

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title(title)
    ax.legend(fontsize=8)

    # Equal aspect ratio
    all_pts = np.vstack([pred_positions, gt_positions])
    max_range = max(all_pts.max(0) - all_pts.min(0)) / 2
    mid = (all_pts.max(0) + all_pts.min(0)) / 2
    ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax.set_zlim(mid[2] - max_range, mid[2] + max_range)

    if save_path:
        ax.figure.savefig(save_path, dpi=150, bbox_inches="tight")

    return ax


def plot_position_over_time(
    pred_positions: np.ndarray,
    gt_positions: np.ndarray,
    eval_indices: np.ndarray,
    title: str = "Position Over Time",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot X, Y, Z position comparison over frame index.

    Args:
        pred_positions: (N, 3) predicted positions (1-step ahead)
        gt_positions: (N, 3) ground truth positions
        eval_indices: (N,) frame indices
        title: Plot title
        save_path: If set, save figure

    Returns:
        Matplotlib Figure
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    labels = ["X (m)", "Y (m)", "Z (m)"]

    for i, (ax, label) in enumerate(zip(axes, labels)):
        ax.plot(eval_indices, gt_positions[:, i], "b-", linewidth=1, alpha=0.8, label="GT")
        ax.plot(eval_indices, pred_positions[:, i], "r.", markersize=3, alpha=0.6, label="Pred")
        ax.set_ylabel(label)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Frame Index")
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_joints_over_time(
    pred_joints: np.ndarray,
    gt_joints: np.ndarray,
    eval_indices: np.ndarray,
    title: str = "Joint Angles Over Time",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot 4 joint angle comparisons over frame index.

    Args:
        pred_joints: (N, 4) predicted joint angles
        gt_joints: (N, 4) ground truth joint angles
        eval_indices: (N,) frame indices
        title: Plot title
        save_path: If set, save figure

    Returns:
        Matplotlib Figure
    """
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

    for i, (ax, name) in enumerate(zip(axes, JOINT_NAMES)):
        ax.plot(eval_indices, gt_joints[:, i], "b-", linewidth=1, alpha=0.8, label="GT")
        ax.plot(eval_indices, pred_joints[:, i], "r.", markersize=3, alpha=0.6, label="Pred")
        ax.set_ylabel(f"{name} (deg)")
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Frame Index")
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_error_distribution(
    errors_dict: dict,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot error histograms for position, rotation, and joints.

    Args:
        errors_dict: Dict with keys 'position', 'rotation', 'joints', each an array of errors
        save_path: If set, save figure

    Returns:
        Matplotlib Figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    configs = [
        ("position", "Position Error (m)", "steelblue"),
        ("rotation", "Rotation Error (deg)", "coral"),
        ("joints", "Joint Error (deg)", "seagreen"),
    ]

    for ax, (key, xlabel, color) in zip(axes, configs):
        data = errors_dict[key]
        ax.hist(data, bins=30, color=color, alpha=0.7, edgecolor="white")
        mean_val = np.mean(data)
        ax.axvline(mean_val, color="red", linestyle="--", linewidth=1.5)
        ax.annotate(f"mean={mean_val:.4f}", xy=(mean_val, ax.get_ylim()[1] * 0.9),
                    fontsize=9, color="red", ha="left")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Count")
        ax.grid(True, alpha=0.3)

    fig.suptitle("Error Distributions", fontsize=13)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_per_session_metrics(
    session_results: list[dict],
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Grouped bar chart of per-session metrics.

    Args:
        session_results: List of dicts with 'name', 'position_mse', 'rotation_error_deg', 'joint_error_deg'
        save_path: If set, save figure

    Returns:
        Matplotlib Figure
    """
    names = [r["name"] for r in session_results]
    pos_errors = [r["position_mse"] for r in session_results]
    rot_errors = [r["rotation_error_deg"] for r in session_results]
    joint_errors = [r["joint_error_deg"] for r in session_results]

    x = np.arange(len(names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(max(10, len(names) * 1.2), 5))
    bars1 = ax.bar(x - width, pos_errors, width, label="Position (m)", color="steelblue")
    bars2 = ax.bar(x, rot_errors, width, label="Rotation (deg)", color="coral")
    bars3 = ax.bar(x + width, joint_errors, width, label="Joints (deg)", color="seagreen")

    # Mean lines
    ax.axhline(np.mean(pos_errors), color="steelblue", linestyle="--", alpha=0.5)
    ax.axhline(np.mean(rot_errors), color="coral", linestyle="--", alpha=0.5)
    ax.axhline(np.mean(joint_errors), color="seagreen", linestyle="--", alpha=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Error")
    ax.set_title("Per-Session Metrics")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def render_prediction_on_frame(
    rgb: np.ndarray,
    current_pose_4x4: np.ndarray,
    pred_positions_3d: np.ndarray,
    gt_positions_3d: np.ndarray,
    intrinsics: np.ndarray,
) -> np.ndarray:
    """Project predicted and GT future trajectory onto current camera frame.

    Args:
        rgb: (H, W, 3) uint8 image
        current_pose_4x4: 4x4 camera-to-world transform at current frame
        pred_positions_3d: (K, 3) predicted future positions in world frame
        gt_positions_3d: (K, 3) GT future positions in world frame
        intrinsics: [fx, fy, cx, cy]

    Returns:
        (H, W, 3) uint8 image with overlay
    """
    output = rgb.copy()
    h, w = output.shape[:2]
    fx, fy, cx, cy = intrinsics

    # World-to-camera transform
    R_wc = current_pose_4x4[:3, :3]
    t_wc = current_pose_4x4[:3, 3]
    R_cw = R_wc.T
    t_cw = -R_cw @ t_wc

    def project(p_world):
        p_cam = R_cw @ p_world + t_cw
        if p_cam[2] <= 0.01:
            return None
        u = int(fx * p_cam[0] / p_cam[2] + cx)
        v = int(fy * p_cam[1] / p_cam[2] + cy)
        if 0 <= u < w and 0 <= v < h:
            return (u, v)
        return None

    def draw_circle(img, cx, cy, r, color):
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                if dx * dx + dy * dy <= r * r:
                    px, py = cx + dx, cy + dy
                    if 0 <= px < img.shape[1] and 0 <= py < img.shape[0]:
                        img[py, px] = color

    def draw_line(img, x0, y0, x1, y1, color):
        steps = max(abs(x1 - x0), abs(y1 - y0), 1)
        for s in range(steps + 1):
            t = s / steps
            x = int(x0 + t * (x1 - x0))
            y = int(y0 + t * (y1 - y0))
            if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
                img[y, x] = color

    # Draw GT trajectory (blue)
    gt_pts = [project(p) for p in gt_positions_3d]
    for i, pt in enumerate(gt_pts):
        if pt is not None:
            draw_circle(output, pt[0], pt[1], 3, (0, 0, 255))
            if i > 0 and gt_pts[i - 1] is not None:
                draw_line(output, gt_pts[i - 1][0], gt_pts[i - 1][1], pt[0], pt[1], (0, 0, 255))

    # Draw predicted trajectory (red)
    pred_pts = [project(p) for p in pred_positions_3d]
    for i, pt in enumerate(pred_pts):
        if pt is not None:
            draw_circle(output, pt[0], pt[1], 3, (255, 0, 0))
            if i > 0 and pred_pts[i - 1] is not None:
                draw_line(output, pred_pts[i - 1][0], pred_pts[i - 1][1], pt[0], pt[1], (255, 0, 0))

    return output


def render_prediction_video(
    session,
    pred_actions: np.ndarray,
    gt_actions: np.ndarray,
    eval_indices: np.ndarray,
    output_path: str,
    fps: int = 10,
) -> bool:
    """Render overlay video of predicted vs GT trajectories on camera frames.

    Args:
        session: iPhoneSession object
        pred_actions: (N, action_horizon, 11) predicted actions
        gt_actions: (N, action_horizon, 11) ground truth actions
        eval_indices: (N,) frame indices
        output_path: Path for output MP4
        fps: Output video frame rate

    Returns:
        True if video was written successfully
    """
    try:
        import cv2
    except ImportError:
        # Fallback: save frames as PNGs
        output_dir = Path(output_path).with_suffix("")
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"  opencv-python not available, saving frames to {output_dir}/")

        intrinsics = session.intrinsics
        if intrinsics is None:
            intrinsics = np.array([183.0, 183.0, 240.0, 180.0])

        from PIL import Image as PILImage
        for i, t in enumerate(eval_indices):
            rgb = session.get_rgb_frame(int(t))
            pose = session.get_pose_transform(int(t))
            frame = render_prediction_on_frame(
                rgb, pose,
                pred_actions[i, :, :3],
                gt_actions[i, :, :3],
                intrinsics,
            )
            PILImage.fromarray(frame).save(output_dir / f"frame_{i:04d}.png")

        print(f"  Saved {len(eval_indices)} frames. Combine with:")
        print(f"  ffmpeg -r {fps} -i {output_dir}/frame_%04d.png -c:v libx264 -pix_fmt yuv420p {output_path}")
        return False

    intrinsics = session.intrinsics
    if intrinsics is None:
        intrinsics = np.array([183.0, 183.0, 240.0, 180.0])

    h, w = session.rgb_resolution[1], session.rgb_resolution[0]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

    for i, t in enumerate(eval_indices):
        rgb = session.get_rgb_frame(int(t))
        pose = session.get_pose_transform(int(t))
        frame = render_prediction_on_frame(
            rgb, pose,
            pred_actions[i, :, :3],
            gt_actions[i, :, :3],
            intrinsics,
        )
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    writer.release()
    return True


def create_dashboard(
    session_results: list[dict],
    output_path: str,
    best_idx: int = 0,
) -> plt.Figure:
    """Create a 2x3 composite dashboard figure.

    Top row: 3D trajectory, position over time, joints over time (best session)
    Bottom row: error distributions (pos, rot+joints), per-session bar chart

    Args:
        session_results: List of result dicts from evaluate_episode with trajectories
        output_path: Path to save the dashboard PNG
        best_idx: Index into session_results for the "best" session to showcase

    Returns:
        Matplotlib Figure
    """
    best = session_results[best_idx]

    # Extract 1-step predictions (first action step of each chunk)
    pred_pos = best["pred_actions_all"][:, 0, :3]
    gt_pos = best["gt_actions_all"][:, 0, :3]
    pred_joints = best["pred_actions_all"][:, 0, 7:11]
    gt_joints = best["gt_actions_all"][:, 0, 7:11]
    eval_idx = best["eval_indices"]

    fig = plt.figure(figsize=(20, 12))

    # Top left: 3D trajectory
    ax1 = fig.add_subplot(2, 3, 1, projection="3d")
    plot_trajectory_comparison_3d(pred_pos, gt_pos, title="3D Trajectory", ax=ax1)

    # Top center: position over time
    ax_pos = [fig.add_subplot(2, 3, 2)]
    # Compact: overlay all 3 dims on one plot
    colors_gt = ["blue", "green", "purple"]
    colors_pred = ["red", "orange", "magenta"]
    dim_labels = ["X", "Y", "Z"]
    for i in range(3):
        ax_pos[0].plot(eval_idx, gt_pos[:, i], color=colors_gt[i], linewidth=1, alpha=0.7,
                       label=f"GT {dim_labels[i]}")
        ax_pos[0].plot(eval_idx, pred_pos[:, i], ".", color=colors_pred[i], markersize=2, alpha=0.5,
                       label=f"Pred {dim_labels[i]}")
    ax_pos[0].set_xlabel("Frame")
    ax_pos[0].set_ylabel("Position (m)")
    ax_pos[0].set_title("Position Over Time")
    ax_pos[0].legend(fontsize=6, ncol=2)
    ax_pos[0].grid(True, alpha=0.3)

    # Top right: joints over time (compact)
    ax_j = fig.add_subplot(2, 3, 3)
    joint_colors_gt = ["blue", "green", "purple", "brown"]
    joint_colors_pred = ["red", "orange", "magenta", "pink"]
    for i in range(4):
        ax_j.plot(eval_idx, gt_joints[:, i], color=joint_colors_gt[i], linewidth=1, alpha=0.7,
                  label=f"GT {JOINT_NAMES[i]}")
        ax_j.plot(eval_idx, pred_joints[:, i], ".", color=joint_colors_pred[i], markersize=2, alpha=0.5)
    ax_j.set_xlabel("Frame")
    ax_j.set_ylabel("Angle (deg)")
    ax_j.set_title("Joint Angles Over Time")
    ax_j.legend(fontsize=5, ncol=2)
    ax_j.grid(True, alpha=0.3)

    # Bottom left: position error distribution
    all_pos_errors = np.concatenate([r["position_errors"] for r in session_results])
    ax_pe = fig.add_subplot(2, 3, 4)
    ax_pe.hist(all_pos_errors, bins=30, color="steelblue", alpha=0.7, edgecolor="white")
    mean_pe = np.mean(all_pos_errors)
    ax_pe.axvline(mean_pe, color="red", linestyle="--")
    ax_pe.set_title(f"Position Error (mean={mean_pe:.4f}m)")
    ax_pe.set_xlabel("Error (m)")
    ax_pe.grid(True, alpha=0.3)

    # Bottom center: rotation + joint error
    all_rot_errors = np.concatenate([r["rotation_errors"] for r in session_results])
    all_joint_errors = np.concatenate([r["joint_errors"] for r in session_results])
    ax_re = fig.add_subplot(2, 3, 5)
    ax_re.hist(all_rot_errors, bins=30, color="coral", alpha=0.7, edgecolor="white", label="Rotation")
    ax_re.hist(all_joint_errors, bins=30, color="seagreen", alpha=0.5, edgecolor="white", label="Joints")
    ax_re.set_title(f"Rot={np.mean(all_rot_errors):.1f}deg, Joints={np.mean(all_joint_errors):.1f}deg")
    ax_re.set_xlabel("Error (deg)")
    ax_re.legend(fontsize=8)
    ax_re.grid(True, alpha=0.3)

    # Bottom right: per-session bar chart
    ax_bar = fig.add_subplot(2, 3, 6)
    names = [r["name"] for r in session_results]
    short_names = [n[-8:] if len(n) > 8 else n for n in names]
    x = np.arange(len(names))
    w = 0.25
    ax_bar.bar(x - w, [r["position_mse"] for r in session_results], w, label="Pos (m)", color="steelblue")
    ax_bar.bar(x, [r["rotation_error_deg"] for r in session_results], w, label="Rot (deg)", color="coral")
    ax_bar.bar(x + w, [r["joint_error_deg"] for r in session_results], w, label="Joints (deg)", color="seagreen")
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(short_names, rotation=45, ha="right", fontsize=7)
    ax_bar.set_title("Per-Session Metrics")
    ax_bar.legend(fontsize=7)
    ax_bar.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Diffusion Policy Evaluation Dashboard", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig
