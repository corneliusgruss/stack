"""
Visualization tools for iPhone ARKit capture sessions.

Provides:
- Depth colorization with turbo colormap
- 3D trajectory plotting for ARKit tracking validation
- Interactive frame browser (RGB + depth slider)
- Pose axes overlay on RGB frames
- Session summary statistics
"""

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib import colormaps as mpl_colormaps

from stack.data.iphone_loader import iPhoneSession, load_session


def colorize_depth(
    depth: np.ndarray,
    vmin: float = 0.1,
    vmax: float = 3.0,
    colormap: str = "turbo",
) -> np.ndarray:
    """
    Convert depth map to colorized RGB image.

    Args:
        depth: (H, W) float32 depth in meters
        vmin: Minimum depth for colormap (meters)
        vmax: Maximum depth for colormap (meters)
        colormap: Matplotlib colormap name

    Returns:
        (H, W, 3) uint8 colorized depth image
    """
    # Normalize depth to [0, 1] range
    depth_normalized = (depth - vmin) / (vmax - vmin)
    depth_normalized = np.clip(depth_normalized, 0, 1)

    # Apply colormap
    cmap = mpl_colormaps[colormap]
    colored = cmap(depth_normalized)[:, :, :3]  # Drop alpha channel

    # Mark invalid depth (zeros) as black
    invalid_mask = depth <= 0
    colored[invalid_mask] = 0

    return (colored * 255).astype(np.uint8)


def plot_trajectory_3d(
    poses_7d: np.ndarray,
    timestamps: Optional[np.ndarray] = None,
    ax: Optional[plt.Axes] = None,
    title: str = "ARKit Trajectory",
) -> plt.Axes:
    """
    Plot 3D trajectory from poses.

    Args:
        poses_7d: (T, 7) array [x, y, z, qx, qy, qz, qw]
        timestamps: (T,) array for coloring by time (optional)
        ax: Existing 3D axes (creates new if None)
        title: Plot title

    Returns:
        Matplotlib 3D axes
    """
    if ax is None:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")

    x, y, z = poses_7d[:, 0], poses_7d[:, 1], poses_7d[:, 2]

    # Color by timestamp if provided
    if timestamps is not None:
        t_norm = (timestamps - timestamps.min()) / (timestamps.max() - timestamps.min() + 1e-6)
        colors = mpl_colormaps["viridis"](t_norm)
        ax.scatter(x, y, z, c=colors, s=2, alpha=0.6)
    else:
        ax.plot(x, y, z, "b-", linewidth=1, alpha=0.7)

    # Mark start (green) and end (red)
    ax.scatter([x[0]], [y[0]], [z[0]], c="green", s=100, marker="o", label="Start")
    ax.scatter([x[-1]], [y[-1]], [z[-1]], c="red", s=100, marker="x", label="End")

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title(title)
    ax.legend()

    # Equal aspect ratio
    max_range = max(x.max() - x.min(), y.max() - y.min(), z.max() - z.min()) / 2
    mid_x, mid_y, mid_z = (x.max() + x.min()) / 2, (y.max() + y.min()) / 2, (z.max() + z.min()) / 2
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    return ax


def overlay_pose_axes(
    rgb: np.ndarray,
    pose_4x4: np.ndarray,
    axis_length: float = 0.05,
    fx: float = 1400.0,
    fy: float = 1400.0,
    cx: Optional[float] = None,
    cy: Optional[float] = None,
) -> np.ndarray:
    """
    Draw XYZ axes on RGB image based on camera pose.

    Args:
        rgb: (H, W, 3) uint8 image
        pose_4x4: 4x4 camera-to-world transform
        axis_length: Length of axes in meters
        fx, fy: Focal lengths in pixels
        cx, cy: Principal point (defaults to image center)

    Returns:
        (H, W, 3) uint8 image with axes drawn
    """
    h, w = rgb.shape[:2]
    if cx is None:
        cx = w / 2
    if cy is None:
        cy = h / 2

    # Create output image (copy to avoid modifying original)
    output = rgb.copy()

    # Origin in world frame
    origin = pose_4x4[:3, 3]

    # Axis endpoints in world frame
    axes_world = np.array([
        origin,  # Origin
        origin + pose_4x4[:3, 0] * axis_length,  # X (red)
        origin + pose_4x4[:3, 1] * axis_length,  # Y (green)
        origin + pose_4x4[:3, 2] * axis_length,  # Z (blue)
    ])

    # Project to image coordinates
    # For a camera-to-world transform, we need world-to-camera for projection
    # Since this is showing the phone's own axes, project relative to identity
    # (the axes are in the camera's local frame)
    def project_point(pt_world: np.ndarray) -> Tuple[int, int]:
        # Transform to camera frame (inverse of pose)
        R = pose_4x4[:3, :3].T
        t = -R @ pose_4x4[:3, 3]
        pt_cam = R @ pt_world + t

        if pt_cam[2] <= 0:
            return None  # Behind camera

        u = int(fx * pt_cam[0] / pt_cam[2] + cx)
        v = int(fy * pt_cam[1] / pt_cam[2] + cy)
        return (u, v)

    # For self-visualization, draw axes in a corner instead
    # (projecting phone's own pose onto its image doesn't make sense)
    # Instead, draw a small orientation indicator in the corner
    corner_x, corner_y = 100, h - 100
    scale = 50  # pixels

    # Extract rotation from pose
    R = pose_4x4[:3, :3]

    # Draw axes (X=red, Y=green, Z=blue)
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    for i, color in enumerate(colors):
        # Project axis direction to 2D (simplified: just use XY components)
        dx = int(R[0, i] * scale)
        dy = int(-R[1, i] * scale)  # Flip Y for image coordinates

        end_x = corner_x + dx
        end_y = corner_y + dy

        # Draw line (simple Bresenham-style)
        _draw_line(output, corner_x, corner_y, end_x, end_y, color, thickness=3)

    # Draw origin circle
    _draw_circle(output, corner_x, corner_y, 5, (255, 255, 255))

    return output


def _draw_line(
    img: np.ndarray,
    x0: int, y0: int,
    x1: int, y1: int,
    color: Tuple[int, int, int],
    thickness: int = 1,
) -> None:
    """Draw a line on an image (simple implementation)."""
    h, w = img.shape[:2]
    steps = max(abs(x1 - x0), abs(y1 - y0), 1)
    for i in range(steps + 1):
        t = i / steps
        x = int(x0 + t * (x1 - x0))
        y = int(y0 + t * (y1 - y0))
        for dx in range(-thickness // 2, thickness // 2 + 1):
            for dy in range(-thickness // 2, thickness // 2 + 1):
                px, py = x + dx, y + dy
                if 0 <= px < w and 0 <= py < h:
                    img[py, px] = color


def _draw_circle(
    img: np.ndarray,
    cx: int, cy: int,
    radius: int,
    color: Tuple[int, int, int],
) -> None:
    """Draw a filled circle on an image."""
    h, w = img.shape[:2]
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx * dx + dy * dy <= radius * radius:
                px, py = cx + dx, cy + dy
                if 0 <= px < w and 0 <= py < h:
                    img[py, px] = color


def create_frame_browser(session: iPhoneSession) -> None:
    """
    Create interactive matplotlib figure to browse RGB and depth frames.

    Args:
        session: Loaded iPhone session
    """
    fig, (ax_rgb, ax_depth) = plt.subplots(1, 2, figsize=(14, 6))
    plt.subplots_adjust(bottom=0.15)

    n_frames = session.num_poses

    # Initial frame
    rgb = session.get_rgb_frame(0)
    pose = session.get_pose_transform(0)
    rgb_with_axes = overlay_pose_axes(rgb, pose)

    depth_idx = session.get_depth_index_for_rgb(0)
    if depth_idx is not None:
        depth = session.get_depth_frame(depth_idx)
        depth_colored = colorize_depth(depth)
    else:
        depth_colored = np.zeros((session.depth_resolution[1], session.depth_resolution[0], 3), dtype=np.uint8)
        depth_colored[:, :] = [64, 64, 64]  # Gray placeholder

    im_rgb = ax_rgb.imshow(rgb_with_axes)
    im_depth = ax_depth.imshow(depth_colored)

    ax_rgb.set_title("RGB (with pose axes)")
    ax_rgb.axis("off")
    ax_depth.set_title("Depth (turbo colormap)")
    ax_depth.axis("off")

    timestamps = session.get_timestamps()

    # Slider
    ax_slider = plt.axes([0.15, 0.05, 0.7, 0.03])
    slider = Slider(ax_slider, "Frame", 0, n_frames - 1, valinit=0, valstep=1)

    def update(val: float) -> None:
        idx = int(val)

        # Update RGB
        rgb = session.get_rgb_frame(idx)
        pose = session.get_pose_transform(idx)
        rgb_with_axes = overlay_pose_axes(rgb, pose)
        im_rgb.set_data(rgb_with_axes)

        # Update depth
        depth_idx = session.get_depth_index_for_rgb(idx)
        if depth_idx is not None:
            depth = session.get_depth_frame(depth_idx)
            depth_colored = colorize_depth(depth)
            ax_depth.set_title(f"Depth #{depth_idx} (turbo colormap)")
        else:
            depth_colored = np.zeros((session.depth_resolution[1], session.depth_resolution[0], 3), dtype=np.uint8)
            depth_colored[:, :] = [64, 64, 64]
            ax_depth.set_title("No depth for this frame")

        im_depth.set_data(depth_colored)

        # Update title with timestamp
        t = timestamps[idx] - timestamps[0]
        fig.suptitle(f"Frame {idx}/{n_frames - 1} | t={t:.3f}s")
        fig.canvas.draw_idle()

    slider.on_changed(update)

    # Initial title
    fig.suptitle(f"Frame 0/{n_frames - 1} | t=0.000s")

    plt.show()


def save_summary_stats(session: iPhoneSession, output_dir: Path) -> None:
    """
    Save session statistics to a text file.

    Args:
        session: Loaded iPhone session
        output_dir: Directory to save stats.txt
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamps = session.get_timestamps()
    duration = timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 0

    # Calculate frame rates
    rgb_fps = session.num_rgb_frames / duration if duration > 0 else 0
    depth_fps = session.num_depth_frames / duration if duration > 0 else 0

    # Depth coverage
    depth_available = 0
    for i in range(session.num_poses):
        if session.get_depth_index_for_rgb(i) is not None:
            depth_available += 1
    depth_coverage = depth_available / session.num_poses * 100 if session.num_poses > 0 else 0

    # Timestamp gaps (detect dropped frames)
    if len(timestamps) > 1:
        dt = np.diff(timestamps)
        median_dt = np.median(dt)
        gaps = dt[dt > median_dt * 2]  # Frames with >2x median interval
        num_gaps = len(gaps)
        max_gap = dt.max()
    else:
        num_gaps = 0
        max_gap = 0

    # Pose statistics
    poses = session.get_all_poses_7d()
    pos = poses[:, :3]
    travel_dist = np.sum(np.linalg.norm(np.diff(pos, axis=0), axis=1))

    lines = [
        f"Session: {session.path.name}",
        f"",
        f"Frame Counts:",
        f"  RGB frames: {session.num_rgb_frames}",
        f"  Depth frames: {session.num_depth_frames}",
        f"  Poses: {session.num_poses}",
        f"",
        f"Timing:",
        f"  Duration: {duration:.2f}s",
        f"  RGB frame rate: {rgb_fps:.1f} Hz",
        f"  Depth frame rate: {depth_fps:.1f} Hz",
        f"  Timestamp gaps (>2x median): {num_gaps}",
        f"  Max timestamp gap: {max_gap * 1000:.1f}ms",
        f"",
        f"Coverage:",
        f"  Depth coverage: {depth_coverage:.1f}% of RGB frames have depth",
        f"",
        f"Motion:",
        f"  Total travel distance: {travel_dist:.3f}m",
        f"  Position range X: [{pos[:, 0].min():.3f}, {pos[:, 0].max():.3f}]m",
        f"  Position range Y: [{pos[:, 1].min():.3f}, {pos[:, 1].max():.3f}]m",
        f"  Position range Z: [{pos[:, 2].min():.3f}, {pos[:, 2].max():.3f}]m",
        f"",
        f"Device: {session.metadata.get('deviceModel', 'Unknown')}",
        f"RGB resolution: {session.rgb_resolution[0]}x{session.rgb_resolution[1]}",
        f"Depth resolution: {session.depth_resolution[0]}x{session.depth_resolution[1]}",
    ]

    stats_file = output_dir / "stats.txt"
    with open(stats_file, "w") as f:
        f.write("\n".join(lines))

    print(f"Saved stats to: {stats_file}")


def visualize_session(session_path: str, save: bool = False) -> None:
    """
    Main visualization entry point.

    Args:
        session_path: Path to session directory
        save: If True, save outputs to session_path/viz/
    """
    session = load_session(session_path)
    print(session.summary())
    print()

    poses = session.get_all_poses_7d()
    timestamps = session.get_timestamps()

    if save:
        output_dir = Path(session_path) / "viz"
        output_dir.mkdir(exist_ok=True)

        # Save trajectory plot
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")
        plot_trajectory_3d(poses, timestamps, ax, title=f"Trajectory: {session.path.name}")
        fig.savefig(output_dir / "trajectory_3d.png", dpi=150, bbox_inches="tight")
        print(f"Saved: {output_dir / 'trajectory_3d.png'}")
        plt.close(fig)

        # Save sample RGB frames (first, middle, last)
        indices = [0, session.num_rgb_frames // 2, session.num_rgb_frames - 1]
        for label, idx in zip(["first", "middle", "last"], indices):
            rgb = session.get_rgb_frame(idx)
            pose = session.get_pose_transform(idx)
            rgb_with_axes = overlay_pose_axes(rgb, pose)

            fig, ax = plt.subplots(figsize=(10, 7.5))
            ax.imshow(rgb_with_axes)
            ax.set_title(f"RGB Frame {idx} ({label})")
            ax.axis("off")
            fig.savefig(output_dir / f"rgb_{label}.png", dpi=100, bbox_inches="tight")
            print(f"Saved: {output_dir / f'rgb_{label}.png'}")
            plt.close(fig)

        # Save sample depth frames
        for label, idx in zip(["first", "middle", "last"], indices):
            depth_idx = session.get_depth_index_for_rgb(idx)
            if depth_idx is not None:
                depth = session.get_depth_frame(depth_idx)
                depth_colored = colorize_depth(depth)

                fig, ax = plt.subplots(figsize=(8, 6))
                ax.imshow(depth_colored)
                ax.set_title(f"Depth Frame {depth_idx} ({label})")
                ax.axis("off")
                fig.savefig(output_dir / f"depth_{label}.png", dpi=100, bbox_inches="tight")
                print(f"Saved: {output_dir / f'depth_{label}.png'}")
                plt.close(fig)

        # Save stats
        save_summary_stats(session, output_dir)

        print(f"\nAll outputs saved to: {output_dir}")

    else:
        # Interactive mode
        print("Opening 3D trajectory plot...")
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")
        plot_trajectory_3d(poses, timestamps, ax, title=f"Trajectory: {session.path.name}")
        plt.show()

        print("Opening frame browser...")
        create_frame_browser(session)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m stack.viz.iphone_viz <session_path> [--save]")
        sys.exit(1)

    path = sys.argv[1]
    save_mode = "--save" in sys.argv

    visualize_session(path, save=save_mode)
