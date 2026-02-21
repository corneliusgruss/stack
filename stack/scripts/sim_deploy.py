"""
MuJoCo simulation deployment for trained diffusion policy.

Loads a checkpoint, runs the policy in a MuJoCo gripper scene, and renders
a video of the rollout. The gripper wrist is controlled via mocap (direct
pose), and fingers via position actuators.

Usage:
    python -m stack.scripts.sim_deploy \
        --checkpoint outputs/real_v2/checkpoint_best.pt \
        --output outputs/real_v2/sim_rollout.mp4 \
        --max-steps 100 --device mps
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import mujoco
import imageio

from PIL import Image as PILImage

from stack.policy.diffusion import DiffusionPolicy, PolicyConfig
from stack.data.training_dataset import NormalizationStats, IMAGENET_MEAN, IMAGENET_STD
from stack.data.transforms import (
    action_13d_to_absolute,
    compute_relative_transform,
    relative_transform_to_13d,
    mat_to_pose_7d,
    pose_7d_to_mat,
)


# MJCF path relative to project root
MJCF_PATH = Path(__file__).resolve().parents[2] / "sim" / "gripper.xml"

# Joint names in the order matching the policy's 11D action vector dims 7-10
JOINT_NAMES = ["index_mcp", "index_pip", "three_finger_mcp", "three_finger_pip"]
ACTUATOR_NAMES = ["index_mcp_act", "index_pip_act", "three_finger_mcp_act", "three_finger_pip_act"]


def xyzw_to_wxyz(q):
    """Convert quaternion from [x, y, z, w] (scipy/policy) to [w, x, y, z] (MuJoCo)."""
    return np.array([q[3], q[0], q[1], q[2]], dtype=np.float64)


def wxyz_to_xyzw(q):
    """Convert quaternion from [w, x, y, z] (MuJoCo) to [x, y, z, w] (policy)."""
    return np.array([q[1], q[2], q[3], q[0]], dtype=np.float64)


def ensure_quaternion_continuity(q_new, q_prev):
    """Flip quaternion sign if it's closer to -q_prev (avoid discontinuities)."""
    if np.dot(q_new, q_prev) < 0:
        return -q_new
    return q_new


def wrap_joint_angle(angle_deg):
    """Wrap angle to [-180, 180] degrees."""
    angle_deg = angle_deg % 360.0
    if angle_deg > 180.0:
        angle_deg -= 360.0
    return angle_deg


def clamp_joint_angles(angles_deg):
    """Clamp joint angles to physical limits. Order: idx_mcp, idx_pip, tf_mcp, tf_pip."""
    limits = [(-30, 120), (0, 90), (-30, 120), (0, 90)]
    clamped = np.zeros_like(angles_deg)
    for i, (lo, hi) in enumerate(limits):
        clamped[i] = np.clip(wrap_joint_angle(angles_deg[i]), lo, hi)
    return clamped


def render_obs_image(renderer, data, image_size):
    """Render wrist camera view and preprocess for policy input.

    Returns:
        (3, image_size, image_size) float32 array, ImageNet normalized, CHW
    """
    renderer.update_scene(data, camera="wrist_cam")
    rgb = renderer.render()  # (H, W, 3) uint8
    pil_img = PILImage.fromarray(rgb)
    pil_img = pil_img.resize((image_size, image_size), PILImage.BILINEAR)
    arr = np.array(pil_img, dtype=np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    return arr.transpose(2, 0, 1)  # CHW


def read_sim_transform(data, model):
    """Read current 4x4 transform from MuJoCo mocap state.

    Returns:
        T: (4, 4) transform matrix
        joints_deg: (4,) joint angles in degrees
    """
    from scipy.spatial.transform import Rotation as R

    mocap_id = model.body("mocap_wrist").mocapid[0]
    pos = data.mocap_pos[mocap_id].copy()
    quat_wxyz = data.mocap_quat[mocap_id].copy()
    quat_xyzw = wxyz_to_xyzw(quat_wxyz)

    rot = R.from_quat(quat_xyzw).as_matrix()
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = rot
    T[:3, 3] = pos

    joints_deg = np.zeros(4)
    for i, name in enumerate(JOINT_NAMES):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        qpos_addr = model.jnt_qposadr[joint_id]
        joints_deg[i] = np.degrees(data.qpos[qpos_addr])

    return T, joints_deg


def read_proprio(data, model, pos_offset=None, position_scale=1.0):
    """Read current proprioception from MuJoCo state as 11D absolute.

    Returns positions in policy coordinates (sim pos → policy pos via offset).

    Returns:
        (11,) float64 array: [x, y, z, qx, qy, qz, qw, idx_mcp, idx_pip, tf_mcp, tf_pip]
    """
    mocap_id = model.body("mocap_wrist").mocapid[0]
    pos = data.mocap_pos[mocap_id].copy()  # (3,) sim meters
    if pos_offset is not None:
        pos = (pos - pos_offset) / position_scale  # back to policy coordinates
    quat_wxyz = data.mocap_quat[mocap_id].copy()  # (4,) wxyz
    quat_xyzw = wxyz_to_xyzw(quat_wxyz)  # (4,) xyzw

    # Joint angles in degrees
    joints_deg = np.zeros(4)
    for i, name in enumerate(JOINT_NAMES):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        qpos_addr = model.jnt_qposadr[joint_id]
        joints_deg[i] = np.degrees(data.qpos[qpos_addr])

    return np.concatenate([pos, quat_xyzw, joints_deg])


def load_policy(checkpoint_path, device):
    """Load trained diffusion policy from checkpoint (matches eval.py pattern)."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    ckpt_config = checkpoint["config"].copy()
    if "action_repr" not in ckpt_config:
        ckpt_config["action_repr"] = "absolute_quat"
    config = PolicyConfig(**ckpt_config)

    policy = DiffusionPolicy(config).to(device)
    if "ema_model" in checkpoint:
        policy.load_state_dict(checkpoint["ema_model"])
        print("Using EMA model weights")
    else:
        policy.load_state_dict(checkpoint["model"])
    policy.eval()

    if "normalizer" not in checkpoint:
        raise RuntimeError("No normalizer stats in checkpoint")
    stats = NormalizationStats.from_state_dict(checkpoint["normalizer"])

    print(f"Loaded: {checkpoint_path}")
    if "epoch" in checkpoint:
        print(f"  Epoch: {checkpoint['epoch'] + 1}")
    if "val_loss" in checkpoint:
        print(f"  Val loss: {checkpoint['val_loss']:.4f}")
    print(f"  obs_dim={config.obs_dim}, action_dim={config.action_dim}")
    print(f"  obs_horizon={config.obs_horizon}, action_horizon={config.action_horizon}")

    return policy, config, stats


def get_initial_state(stats):
    """Compute initial state from normalization stats midpoint.

    The midpoint of the action min/max range is a reasonable starting pose
    since training data is centered around it.
    """
    midpoint = (stats.action_min + stats.action_max) / 2.0
    # Normalize the quaternion portion (dims 3:7)
    quat = midpoint[3:7]
    quat_norm = np.linalg.norm(quat)
    if quat_norm > 1e-6:
        midpoint[3:7] = quat / quat_norm
    else:
        midpoint[3:7] = [0, 0, 0, 1]  # identity quaternion xyzw
    return midpoint


def main():
    parser = argparse.ArgumentParser(description="Deploy diffusion policy in MuJoCo simulation")
    parser.add_argument("--checkpoint", required=True, help="Trained checkpoint path")
    parser.add_argument("--output", default="sim_rollout.mp4", help="Output video path")
    parser.add_argument("--max-steps", type=int, default=300, help="Max sim steps (10 Hz)")
    parser.add_argument("--execute-horizon", type=int, default=8, help="Steps per action chunk")
    parser.add_argument("--device", default=None, help="Device (cuda/mps/cpu)")
    parser.add_argument("--position-scale", type=float, default=1.0,
                        help="Scale factor for position (COLMAP units → meters)")
    parser.add_argument("--start-pos", type=float, nargs=3, default=[0.0, 0.0, 0.20],
                        help="Starting wrist position in sim (meters)")
    parser.add_argument("--video-camera", default="overhead", help="Camera for video")
    parser.add_argument("--render-size", type=int, nargs=2, default=[1280, 720],
                        help="Video resolution (width height)")
    parser.add_argument("--dummy-images", action="store_true",
                        help="Use zero images (debug: isolate proprio contribution)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--mjcf", default=None, help="Override MJCF model path")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Device selection (matches eval.py)
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # Load policy
    policy, config, stats = load_policy(args.checkpoint, device)

    # Load MuJoCo model
    mjcf_path = Path(args.mjcf) if args.mjcf else MJCF_PATH
    print(f"Loading MJCF: {mjcf_path}")
    model = mujoco.MjModel.from_xml_path(str(mjcf_path))
    data = mujoco.MjData(model)

    # Create renderers
    obs_renderer = mujoco.Renderer(model, height=240, width=320)
    vid_w, vid_h = args.render_size
    vid_renderer = mujoco.Renderer(model, height=vid_h, width=vid_w)

    # Set initial gripper state from normalization stats midpoint
    initial_state = get_initial_state(stats)
    print(f"Policy midpoint: pos={initial_state[:3]}, quat={initial_state[3:7]}, joints={initial_state[7:]}")

    mocap_id = model.body("mocap_wrist").mocapid[0]

    # Compute offset: map policy coordinates to sim coordinates
    # Policy positions are in COLMAP units centered on training workspace
    # Sim positions are in meters centered on the table
    sim_start = np.array(args.start_pos, dtype=np.float64)
    policy_origin = initial_state[:3].copy() * args.position_scale
    pos_offset = sim_start - policy_origin  # add this to every policy position
    print(f"Sim start: {sim_start}, offset: {pos_offset}")

    # Apply initial pose
    data.mocap_pos[mocap_id] = sim_start

    quat_xyzw = initial_state[3:7].copy()
    quat_wxyz = xyzw_to_wxyz(quat_xyzw)
    quat_wxyz /= np.linalg.norm(quat_wxyz)
    data.mocap_quat[mocap_id] = quat_wxyz

    # Set initial joint angles
    joint_angles = clamp_joint_angles(initial_state[7:11])
    for i, name in enumerate(ACTUATOR_NAMES):
        act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        data.ctrl[act_id] = np.radians(joint_angles[i])  # ctrl is in radians

    # Settle simulation
    for _ in range(500):
        mujoco.mj_step(model, data)

    # Initialize observation buffer
    obs_horizon = config.obs_horizon
    image_size = config.image_size
    action_dim = config.action_dim
    action_repr = getattr(config, "action_repr", "absolute_quat")
    is_relative = action_repr == "relative_6d"

    # Fill obs buffer with initial observations
    obs_images = np.zeros((obs_horizon, 3, image_size, image_size), dtype=np.float32)
    obs_proprio = np.zeros((obs_horizon, action_dim), dtype=np.float32)

    for i in range(obs_horizon):
        if args.dummy_images:
            obs_images[i] = np.zeros((3, image_size, image_size), dtype=np.float32)
        else:
            obs_images[i] = render_obs_image(obs_renderer, data, image_size)
        if is_relative:
            # For relative mode, initial obs proprio is identity (relative to self)
            # [0,0,0, 1,0,0, 0,1,0, j1,j2,j3,j4]
            _, joints_deg = read_sim_transform(data, model)
            identity_13d = np.array([0, 0, 0, 1, 0, 0, 0, 1, 0] + list(joints_deg), dtype=np.float32)
            obs_proprio[i] = stats.normalize_proprio(identity_13d)
        else:
            proprio = read_proprio(data, model, pos_offset, args.position_scale)
            obs_proprio[i] = stats.normalize_proprio(proprio[:action_dim].astype(np.float32))

    # Main loop
    video_frames = []
    action_queue = []
    action_queue_abs = []  # absolute (pos, quat, joints) for relative mode
    prev_quat_wxyz = data.mocap_quat[mocap_id].copy()
    T_ref = None  # reference transform for relative mode

    max_pos_delta = 0.05  # meters per step (0.5 m/s at 10 Hz)
    substeps = 50  # 50 substeps × 0.002s = 0.1s per control step (10 Hz)

    print(f"\nRunning simulation: {args.max_steps} steps @ 10 Hz ({args.max_steps * 0.1:.0f}s)")
    print(f"  Execute horizon: {args.execute_horizon}, Action horizon: {config.action_horizon}")
    print(f"  Action repr: {action_repr}")
    print(f"  Position scale: {args.position_scale}")
    print()

    for step in range(args.max_steps):
        # Get new prediction when action queue is empty
        if len(action_queue_abs) == 0 and len(action_queue) == 0:
            # For relative mode: get current sim state as T_ref
            if is_relative:
                T_ref, ref_joints = read_sim_transform(data, model)
                # Build relative obs proprio
                obs_proprio_buf = np.zeros((obs_horizon, action_dim), dtype=np.float32)
                # Last obs timestep = identity, previous = relative to current
                identity_13d = np.array([0, 0, 0, 1, 0, 0, 0, 1, 0] + list(ref_joints), dtype=np.float32)
                obs_proprio_buf[-1] = stats.normalize_proprio(identity_13d)
                # Earlier obs timesteps: approximate as identity too (closed-loop, small dt)
                for i in range(obs_horizon - 1):
                    obs_proprio_buf[i] = obs_proprio_buf[-1]
            else:
                obs_proprio_buf = obs_proprio

            # Build observation tensors
            images_t = torch.from_numpy(obs_images[np.newaxis]).to(device)
            proprio_t = torch.from_numpy(obs_proprio_buf[np.newaxis]).to(device)

            # Predict action chunk
            with torch.no_grad():
                pred_actions_norm = policy.predict(images_t, proprio_t)
            pred_actions_norm = pred_actions_norm.cpu().numpy()[0]

            # Unnormalize
            pred_actions = stats.unnormalize_action(pred_actions_norm)

            if is_relative:
                # Convert relative 13D actions to absolute sim poses
                for ai in range(min(args.execute_horizon, len(pred_actions))):
                    T_abs, joints_abs = action_13d_to_absolute(pred_actions[ai], T_ref)
                    pose_7d = mat_to_pose_7d(T_abs)
                    abs_action = np.concatenate([pose_7d, joints_abs])
                    action_queue_abs.append(abs_action)
            else:
                action_queue = list(pred_actions[:args.execute_horizon])

            if step % 50 == 0:
                print(f"  Step {step}/{args.max_steps}: new prediction, "
                      f"pos=[{pred_actions[0, 0]:.3f}, {pred_actions[0, 1]:.3f}, {pred_actions[0, 2]:.3f}]")

        # Pop next action (absolute 11D in either mode)
        if is_relative:
            action = action_queue_abs.pop(0)
        else:
            action = action_queue.pop(0)

        if is_relative:
            # In relative mode, action is already in sim coordinates (absolute)
            target_pos = action[:3].copy()
        else:
            # Apply position with offset mapping
            target_pos = action[:3].copy() * args.position_scale + pos_offset
        current_pos = data.mocap_pos[mocap_id].copy()
        delta = target_pos - current_pos
        delta_norm = np.linalg.norm(delta)
        if delta_norm > max_pos_delta:
            delta = delta * (max_pos_delta / delta_norm)
        data.mocap_pos[mocap_id] = current_pos + delta

        # Apply quaternion (xyzw → wxyz, continuity, normalize)
        quat_xyzw = action[3:7].copy()
        quat_wxyz = xyzw_to_wxyz(quat_xyzw)
        quat_wxyz = ensure_quaternion_continuity(quat_wxyz, prev_quat_wxyz)
        quat_norm = np.linalg.norm(quat_wxyz)
        if quat_norm > 1e-6:
            quat_wxyz /= quat_norm
        else:
            quat_wxyz = np.array([1, 0, 0, 0], dtype=np.float64)
        data.mocap_quat[mocap_id] = quat_wxyz
        prev_quat_wxyz = quat_wxyz.copy()

        # Apply joint angles (clamp to physical limits)
        joint_angles = clamp_joint_angles(action[7:11])
        for i, name in enumerate(ACTUATOR_NAMES):
            act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            data.ctrl[act_id] = np.radians(joint_angles[i])

        # Step simulation
        for _ in range(substeps):
            mujoco.mj_step(model, data)

        # Update observation buffer (shift left, append new)
        if args.dummy_images:
            new_img = np.zeros((3, image_size, image_size), dtype=np.float32)
        else:
            new_img = render_obs_image(obs_renderer, data, image_size)

        if not is_relative:
            new_proprio = read_proprio(data, model, pos_offset, args.position_scale)
            new_proprio_norm = stats.normalize_proprio(new_proprio[:action_dim].astype(np.float32))
            obs_proprio = np.roll(obs_proprio, -1, axis=0)
            obs_proprio[-1] = new_proprio_norm

        obs_images = np.roll(obs_images, -1, axis=0)
        obs_images[-1] = new_img

        # Render video frame
        vid_renderer.update_scene(data, camera=args.video_camera)
        frame = vid_renderer.render()
        video_frames.append(frame.copy())

    # Save video
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving video: {output_path} ({len(video_frames)} frames)")
    writer = imageio.get_writer(str(output_path), fps=10, codec="libx264",
                                quality=8, pixelformat="yuv420p")
    for frame in video_frames:
        writer.append_data(frame)
    writer.close()
    print(f"Done. Video saved to {output_path}")

    # Read final state
    final_proprio = read_proprio(data, model, pos_offset, args.position_scale)
    print(f"\nFinal state:")
    print(f"  Position: [{final_proprio[0]:.4f}, {final_proprio[1]:.4f}, {final_proprio[2]:.4f}]")
    print(f"  Joints:   [{final_proprio[7]:.1f}, {final_proprio[8]:.1f}, {final_proprio[9]:.1f}, {final_proprio[10]:.1f}] deg")


if __name__ == "__main__":
    main()
