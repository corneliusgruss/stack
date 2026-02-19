"""Visualization tools for Stack data collection."""

from stack.viz.iphone_viz import (
    colorize_depth,
    plot_trajectory_3d,
    create_frame_browser,
    overlay_pose_axes,
    save_summary_stats,
    visualize_session,
)

from stack.viz.eval_viz import (
    plot_trajectory_comparison_3d,
    plot_position_over_time,
    plot_joints_over_time,
    plot_error_distribution,
    plot_per_session_metrics,
    render_prediction_on_frame,
    render_prediction_video,
    create_dashboard,
)

__all__ = [
    "colorize_depth",
    "plot_trajectory_3d",
    "create_frame_browser",
    "overlay_pose_axes",
    "save_summary_stats",
    "visualize_session",
    "plot_trajectory_comparison_3d",
    "plot_position_over_time",
    "plot_joints_over_time",
    "plot_error_distribution",
    "plot_per_session_metrics",
    "render_prediction_on_frame",
    "render_prediction_video",
    "create_dashboard",
]
