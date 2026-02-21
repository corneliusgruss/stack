"""
Open-loop policy replay on MuJoCo gripper model.

Takes a real session, runs the policy on real images to get predicted
action trajectories, then replays those actions on the MuJoCo gripper
for visualization. The policy never sees sim images — only real data.

Usage:
    python -m stack.scripts.sim_replay \
        --checkpoint outputs/real_v2/checkpoint_best.pt \
        --session data/raw/session_2026-02-19_152352 \
        --output outputs/real_v2/sim_replay.mp4 \
        --device mps
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import mujoco
import imageio
from PIL import Image as PILImage
from scipy.spatial.transform import Rotation

from stack.policy.diffusion import DiffusionPolicy, PolicyConfig
from stack.data.training_dataset import NormalizationStats, IMAGENET_MEAN, IMAGENET_STD
from stack.data.iphone_loader import iPhoneSession
from stack.data.transforms import (
    action_13d_to_absolute,
    compute_relative_transform,
    relative_transform_to_13d,
    mat_to_pose_7d,
)


MJCF_PATH = Path(__file__).resolve().parents[2] / "sim" / "gripper.xml"
JOINT_NAMES = ["index_mcp", "index_pip", "three_finger_mcp", "three_finger_pip"]
ACTUATOR_NAMES = ["index_mcp_act", "index_pip_act", "three_finger_mcp_act", "three_finger_pip_act"]


def xyzw_to_wxyz(q):
    return np.array([q[3], q[0], q[1], q[2]], dtype=np.float64)


def ensure_quaternion_continuity(q_new, q_prev):
    if np.dot(q_new, q_prev) < 0:
        return -q_new
    return q_new


def wrap_joint_angle(angle_deg):
    angle_deg = angle_deg % 360.0
    if angle_deg > 180.0:
        angle_deg -= 360.0
    return angle_deg


def clamp_joint_angles(angles_deg):
    limits = [(-30, 120), (0, 90), (-30, 120), (0, 90)]
    clamped = np.zeros_like(angles_deg)
    for i, (lo, hi) in enumerate(limits):
        clamped[i] = np.clip(wrap_joint_angle(angles_deg[i]), lo, hi)
    return clamped


def load_policy(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    ckpt_config = checkpoint["config"].copy()
    if "action_repr" not in ckpt_config:
        ckpt_config["action_repr"] = "absolute_quat"
    config = PolicyConfig(**ckpt_config)
    policy = DiffusionPolicy(config).to(device)
    if "ema_model" in checkpoint:
        policy.load_state_dict(checkpoint["ema_model"])
    else:
        policy.load_state_dict(checkpoint["model"])
    policy.eval()
    if "normalizer" not in checkpoint:
        raise RuntimeError("No normalizer stats in checkpoint")
    stats = NormalizationStats.from_state_dict(checkpoint["normalizer"])
    return policy, config, stats


def estimate_frame_rotation(episode):
    """Estimate rotation from COLMAP frame to MuJoCo Z-up frame.

    Uses the camera orientations to find the average "down" direction
    in COLMAP world. In the real setup, the camera mostly looks downward
    at the table, so the average camera Z-axis ≈ gravity direction.

    Returns:
        Rotation object that transforms COLMAP world → MuJoCo world (Z-up)
    """
    # Sample camera orientations across the trajectory
    n_samples = min(100, len(episode))
    indices = np.linspace(0, len(episode) - 1, n_samples, dtype=int)

    # Camera Z-axis (look direction) in COLMAP world
    cam_z_dirs = []
    cam_y_dirs = []
    for idx in indices:
        q_xyzw = episode[idx, 3:7]
        r = Rotation.from_quat(q_xyzw)
        cam_z_dirs.append(r.apply([0, 0, 1]))  # camera forward
        cam_y_dirs.append(r.apply([0, 1, 0]))  # camera down

    avg_cam_z = np.mean(cam_z_dirs, axis=0)
    avg_cam_z /= np.linalg.norm(avg_cam_z)

    # The camera looks down at the table, so avg camera Z ≈ down direction
    # In MuJoCo, down is -Z. So we want: COLMAP avg_cam_z → MuJoCo -Z
    # i.e., gravity_colmap → [0, 0, -1]
    gravity_colmap = avg_cam_z  # "down" in COLMAP frame

    # Build rotation: align gravity_colmap with MuJoCo -Z
    # Also need a horizontal reference. Use trajectory principal direction.
    positions = episode[:, :3]
    pos_centered = positions - positions.mean(axis=0)

    # PCA to find main horizontal motion direction
    cov = np.cov(pos_centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    # Largest eigenvector = main motion direction
    main_dir = eigvecs[:, -1]  # in COLMAP frame

    # Remove gravity component from main_dir to get horizontal direction
    main_dir_horiz = main_dir - np.dot(main_dir, gravity_colmap) * gravity_colmap
    main_dir_horiz /= np.linalg.norm(main_dir_horiz)

    # Build orthonormal basis for COLMAP frame aligned with gravity:
    # z_colmap = -gravity (up)
    # x_colmap = main horizontal motion direction
    # y_colmap = z × x
    z_col = -gravity_colmap  # "up" in COLMAP
    x_col = main_dir_horiz
    y_col = np.cross(z_col, x_col)
    y_col /= np.linalg.norm(y_col)
    x_col = np.cross(y_col, z_col)  # ensure orthogonal

    # This basis (x_col, y_col, z_col) should map to MuJoCo (x, y, z)
    # R_colmap_to_mujoco: columns are where COLMAP basis vectors go
    R_mat = np.column_stack([x_col, y_col, z_col])
    # R_mat transforms: COLMAP coords → aligned coords
    # We want R such that R @ colmap_vec = mujoco_vec
    # R = I @ inv(R_mat) since R_mat columns are COLMAP basis in COLMAP coords
    # Actually R_mat rows are the new basis expressed in COLMAP coords
    # So R = R_mat.T transforms COLMAP → aligned
    R_frame = np.linalg.inv(R_mat)

    print(f"  Gravity in COLMAP frame: {gravity_colmap.round(3)}")
    print(f"  Main motion dir (horiz): {main_dir_horiz.round(3)}")

    return Rotation.from_matrix(R_frame)


def transform_trajectory(trajectory, R_frame, position_scale, start_pos):
    """Transform trajectory from COLMAP frame to MuJoCo sim frame.

    Applies frame rotation, scaling, and offset to positions and quaternions.

    Returns:
        Transformed trajectory (T, 11) in sim coordinates
    """
    transformed = trajectory.copy()

    # Transform positions: rotate then scale then offset
    positions = trajectory[:, :3]
    pos_centered = positions - positions[0]  # relative to first frame
    pos_rotated = R_frame.apply(pos_centered)
    pos_scaled = pos_rotated * position_scale
    pos_final = pos_scaled + np.array(start_pos)
    transformed[:, :3] = pos_final

    # Transform quaternions: compose frame rotation with each camera orientation
    for i in range(len(trajectory)):
        q_xyzw = trajectory[i, 3:7]
        r_cam = Rotation.from_quat(q_xyzw)
        r_new = R_frame * r_cam
        transformed[i, 3:7] = r_new.as_quat()

    return transformed


def apply_action_to_sim(data, model, action, prev_quat_wxyz):
    """Apply a pre-transformed action to the MuJoCo sim state."""
    mocap_id = model.body("mocap_wrist").mocapid[0]

    # Position (already in sim coordinates)
    data.mocap_pos[mocap_id] = action[:3].copy()

    # Quaternion (xyzw → wxyz, continuity, normalize)
    quat_xyzw = action[3:7].copy()
    quat_wxyz = xyzw_to_wxyz(quat_xyzw)
    quat_wxyz = ensure_quaternion_continuity(quat_wxyz, prev_quat_wxyz)
    quat_norm = np.linalg.norm(quat_wxyz)
    if quat_norm > 1e-6:
        quat_wxyz /= quat_norm
    else:
        quat_wxyz = np.array([1, 0, 0, 0], dtype=np.float64)
    data.mocap_quat[mocap_id] = quat_wxyz

    # Joint angles
    joint_angles = clamp_joint_angles(action[7:11])
    for i, name in enumerate(ACTUATOR_NAMES):
        act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        data.ctrl[act_id] = np.radians(joint_angles[i])

    return quat_wxyz


def predict_trajectory(policy, session, stats, config, device):
    """Run policy on real session images to produce full predicted trajectory.

    Returns trajectory as 11D absolute [x,y,z,qx,qy,qz,qw,j1,j2,j3,j4]
    regardless of action_repr — relative actions are converted back.
    """
    obs_horizon = config.obs_horizon
    action_horizon = config.action_horizon
    action_dim = config.action_dim
    image_size = config.image_size
    action_repr = getattr(config, "action_repr", "absolute_quat")
    is_relative = action_repr == "relative_6d"
    execute_horizon = 8

    if is_relative:
        transforms = session.get_all_transforms()  # (T, 4, 4)
        joints = session.get_aligned_encoders()     # (T, 4)
        T = len(transforms)
    else:
        episode = session.get_episode_11d()[:, :action_dim]
        T = len(episode)

    all_actions = []
    t = obs_horizon - 1

    while t < T - 1:
        obs_start = t - obs_horizon + 1

        obs_images = np.zeros((1, obs_horizon, 3, image_size, image_size), dtype=np.float32)
        for i in range(obs_horizon):
            img = session.get_rgb_frame(obs_start + i)
            pil_img = PILImage.fromarray(img)
            pil_img = pil_img.resize((image_size, image_size), PILImage.BILINEAR)
            arr = np.array(pil_img, dtype=np.float32) / 255.0
            arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
            obs_images[0, i] = arr.transpose(2, 0, 1)

        if is_relative:
            T_ref = transforms[t]
            obs_proprio = np.zeros((obs_horizon, action_dim), dtype=np.float32)
            for i in range(obs_horizon):
                fi = obs_start + i
                T_rel = compute_relative_transform(T_ref, transforms[fi])
                obs_proprio[i] = relative_transform_to_13d(T_rel, joints[fi])
        else:
            obs_proprio = episode[obs_start:obs_start + obs_horizon].copy()

        obs_proprio = stats.normalize_proprio(obs_proprio)
        obs_proprio = obs_proprio[np.newaxis]

        images_t = torch.from_numpy(obs_images).to(device)
        proprio_t = torch.from_numpy(obs_proprio).to(device)
        with torch.no_grad():
            pred_norm = policy.predict(images_t, proprio_t)
        pred_norm = pred_norm.cpu().numpy()[0]
        pred_actions = stats.unnormalize_action(pred_norm)

        steps_to_take = min(execute_horizon, T - 1 - t)

        if is_relative:
            # Convert relative 13D to absolute 11D for sim replay
            abs_chunk = np.zeros((steps_to_take, 11), dtype=np.float32)
            for ai in range(steps_to_take):
                T_abs, joints_abs = action_13d_to_absolute(pred_actions[ai], T_ref)
                pose_7d = mat_to_pose_7d(T_abs)
                abs_chunk[ai] = np.concatenate([pose_7d, joints_abs])
            all_actions.append(abs_chunk)
        else:
            all_actions.append(pred_actions[:steps_to_take])

        t += steps_to_take

    return np.concatenate(all_actions, axis=0)


def main():
    parser = argparse.ArgumentParser(description="Open-loop policy replay on MuJoCo model")
    parser.add_argument("--checkpoint", required=True, help="Trained checkpoint path")
    parser.add_argument("--session", required=True, help="Real session directory")
    parser.add_argument("--output", default="sim_replay.mp4", help="Output video path")
    parser.add_argument("--device", default=None, help="Device (cuda/mps/cpu)")
    parser.add_argument("--position-scale", type=float, default=0.25,
                        help="Scale factor for COLMAP positions → meters")
    parser.add_argument("--start-pos", type=float, nargs=3, default=[0.0, 0.0, 0.15],
                        help="Sim start position for the wrist (meters)")
    parser.add_argument("--video-camera", default="overhead", help="Camera for video rendering")
    parser.add_argument("--render-size", type=int, nargs=2, default=[1280, 720],
                        help="Video resolution (width height)")
    parser.add_argument("--side-by-side", action="store_true",
                        help="Render real image + sim view side by side")
    parser.add_argument("--mjcf", default=None, help="Override MJCF model path")
    parser.add_argument("--ground-truth", action="store_true",
                        help="Replay ground truth actions instead of policy predictions")
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

    # Load policy and session
    policy, config, stats = load_policy(args.checkpoint, device)
    session = iPhoneSession(args.session)
    episode = session.get_episode_11d()[:, :min(config.action_dim, 11)]
    print(f"Session: {args.session}")
    print(f"  Frames: {session.num_rgb_frames}, Episode length: {len(episode)}")

    # Get trajectory (in COLMAP coordinates, always 11D absolute for sim)
    if args.ground_truth:
        print("Using ground truth actions")
        trajectory_raw = episode[config.obs_horizon:]
    else:
        print("Running policy on real images...")
        trajectory_raw = predict_trajectory(policy, session, stats, config, device)
    print(f"  Trajectory: {len(trajectory_raw)} steps")

    # Estimate COLMAP → MuJoCo frame rotation from camera orientations
    print("Estimating frame rotation...")
    R_frame = estimate_frame_rotation(episode)

    # Transform trajectory to sim coordinates
    trajectory = transform_trajectory(
        trajectory_raw, R_frame, args.position_scale, args.start_pos
    )
    print(f"  Sim position range:")
    for i, ax in enumerate("XYZ"):
        vals = trajectory[:, i]
        print(f"    {ax}: [{vals.min():.3f}, {vals.max():.3f}]")

    # Load MuJoCo
    mjcf_path = Path(args.mjcf) if args.mjcf else MJCF_PATH
    model = mujoco.MjModel.from_xml_path(str(mjcf_path))
    data = mujoco.MjData(model)

    # Set initial pose
    mocap_id = model.body("mocap_wrist").mocapid[0]
    data.mocap_pos[mocap_id] = trajectory[0, :3]
    quat_wxyz = xyzw_to_wxyz(trajectory[0, 3:7])
    quat_wxyz /= np.linalg.norm(quat_wxyz)
    data.mocap_quat[mocap_id] = quat_wxyz

    # Settle
    for _ in range(1000):
        mujoco.mj_step(model, data)

    # Renderer
    vid_w, vid_h = args.render_size
    if args.side_by_side:
        panel_w = vid_w // 2
        vid_renderer = mujoco.Renderer(model, height=vid_h, width=panel_w)
    else:
        vid_renderer = mujoco.Renderer(model, height=vid_h, width=vid_w)

    # Replay
    video_frames = []
    prev_quat_wxyz = quat_wxyz.copy()
    substeps = 50
    obs_horizon = config.obs_horizon
    frame_offset = obs_horizon

    print(f"\nReplaying {len(trajectory)} steps...")

    for step_idx in range(len(trajectory)):
        action = trajectory[step_idx]
        prev_quat_wxyz = apply_action_to_sim(data, model, action, prev_quat_wxyz)

        for _ in range(substeps):
            mujoco.mj_step(model, data)

        vid_renderer.update_scene(data, camera=args.video_camera)
        sim_frame = vid_renderer.render()

        if args.side_by_side:
            real_frame_idx = frame_offset + step_idx
            if real_frame_idx < session.num_rgb_frames:
                real_img = session.get_rgb_frame(real_frame_idx)
                real_pil = PILImage.fromarray(real_img).resize((panel_w, vid_h), PILImage.BILINEAR)
                real_frame = np.array(real_pil)
            else:
                real_frame = np.zeros((vid_h, panel_w, 3), dtype=np.uint8)
            combined = np.concatenate([real_frame, sim_frame], axis=1)
            video_frames.append(combined)
        else:
            video_frames.append(sim_frame.copy())

        if step_idx % 100 == 0:
            pos = data.mocap_pos[mocap_id]
            print(f"  Step {step_idx}/{len(trajectory)}: "
                  f"pos=[{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving video: {output_path} ({len(video_frames)} frames)")
    writer = imageio.get_writer(str(output_path), fps=10, codec="libx264",
                                quality=8, pixelformat="yuv420p")
    for frame in video_frames:
        writer.append_data(frame)
    writer.close()
    print(f"Done. Video saved to {output_path}")


if __name__ == "__main__":
    main()
