"""
SE(3) and rotation utilities for relative action representation.

Provides conversions between quaternion, rotation matrix, and 6D rotation
representations, plus relative transform computation for frame-invariant
action representation (Zhou et al., CVPR 2019; Chi et al., RSS 2024).

All functions support both single and batched inputs.
"""

import numpy as np
from scipy.spatial.transform import Rotation


def rotmat_to_6d(R: np.ndarray) -> np.ndarray:
    """Extract 6D rotation representation (first two columns of rotation matrix).

    Args:
        R: (..., 3, 3) rotation matrix

    Returns:
        (..., 6) first two columns flattened [r1x, r1y, r1z, r2x, r2y, r2z]
    """
    return np.concatenate([R[..., :, 0], R[..., :, 1]], axis=-1)


def rot6d_to_rotmat(r6d: np.ndarray) -> np.ndarray:
    """Recover rotation matrix from 6D representation via Gram-Schmidt.

    Args:
        r6d: (..., 6) 6D rotation

    Returns:
        (..., 3, 3) proper rotation matrix
    """
    a1 = r6d[..., :3]
    a2 = r6d[..., 3:6]

    # Normalize first column
    b1 = a1 / (np.linalg.norm(a1, axis=-1, keepdims=True) + 1e-8)

    # Gram-Schmidt: make second column orthogonal to first, then normalize
    dot = np.sum(b1 * a2, axis=-1, keepdims=True)
    b2 = a2 - dot * b1
    b2 = b2 / (np.linalg.norm(b2, axis=-1, keepdims=True) + 1e-8)

    # Third column via cross product
    b3 = np.cross(b1, b2)

    # Stack into rotation matrix: columns are b1, b2, b3
    return np.stack([b1, b2, b3], axis=-1)


def pose_7d_to_mat(pose_7d: np.ndarray) -> np.ndarray:
    """Convert 7D pose [x, y, z, qx, qy, qz, qw] to 4x4 transform matrix.

    Args:
        pose_7d: (..., 7) pose vector

    Returns:
        (..., 4, 4) homogeneous transform matrix
    """
    shape = pose_7d.shape[:-1]
    flat = pose_7d.reshape(-1, 7)
    n = flat.shape[0]

    mats = np.zeros((n, 4, 4), dtype=np.float64)
    for i in range(n):
        pos = flat[i, :3]
        quat = flat[i, 3:7]  # xyzw
        R = Rotation.from_quat(quat).as_matrix()
        mats[i, :3, :3] = R
        mats[i, :3, 3] = pos
        mats[i, 3, 3] = 1.0

    return mats.reshape(*shape, 4, 4).astype(np.float32)


def mat_to_pose_7d(T: np.ndarray) -> np.ndarray:
    """Convert 4x4 transform matrix to 7D pose [x, y, z, qx, qy, qz, qw].

    Args:
        T: (..., 4, 4) homogeneous transform

    Returns:
        (..., 7) pose vector
    """
    shape = T.shape[:-2]
    flat = T.reshape(-1, 4, 4)
    n = flat.shape[0]

    poses = np.zeros((n, 7), dtype=np.float32)
    for i in range(n):
        poses[i, :3] = flat[i, :3, 3]
        R = flat[i, :3, :3]
        poses[i, 3:7] = Rotation.from_matrix(R).as_quat()  # xyzw

    return poses.reshape(*shape, 7)


def compute_relative_transform(T_ref: np.ndarray, T_t: np.ndarray) -> np.ndarray:
    """Compute relative transform: T_rel = T_ref^{-1} @ T_t.

    Args:
        T_ref: (..., 4, 4) reference frame transform
        T_t: (..., 4, 4) target frame transform

    Returns:
        (..., 4, 4) relative transform in reference frame's local coordinates
    """
    # Efficient SE(3) inverse: R^T, -R^T @ t
    R_ref = T_ref[..., :3, :3]
    t_ref = T_ref[..., :3, 3:]  # (..., 3, 1)

    R_ref_T = np.swapaxes(R_ref, -2, -1)  # (..., 3, 3)

    R_t = T_t[..., :3, :3]
    t_t = T_t[..., :3, 3:]

    # Relative rotation and translation
    R_rel = R_ref_T @ R_t
    t_rel = R_ref_T @ (t_t - t_ref)

    # Build 4x4
    shape = T_ref.shape[:-2]
    T_rel = np.zeros((*shape, 4, 4), dtype=np.float32)
    T_rel[..., :3, :3] = R_rel
    T_rel[..., :3, 3:] = t_rel
    T_rel[..., 3, 3] = 1.0

    return T_rel


def relative_transform_to_13d(T_rel: np.ndarray, joints: np.ndarray) -> np.ndarray:
    """Convert relative 4x4 transform + joints to 13D action vector.

    Args:
        T_rel: (..., 4, 4) relative transform
        joints: (..., 4) joint angles in degrees

    Returns:
        (..., 13) = [dx, dy, dz, r1..r6, j1, j2, j3, j4]
    """
    pos = T_rel[..., :3, 3]       # (..., 3)
    R = T_rel[..., :3, :3]        # (..., 3, 3)
    r6d = rotmat_to_6d(R)         # (..., 6)

    return np.concatenate([pos, r6d, joints], axis=-1).astype(np.float32)


def action_13d_to_absolute(a13d: np.ndarray, T_ref: np.ndarray):
    """Convert 13D relative action back to absolute 4x4 transform + joints.

    Args:
        a13d: (..., 13) relative action [dx,dy,dz, r1..r6, j1..j4]
        T_ref: (..., 4, 4) reference frame transform

    Returns:
        T_abs: (..., 4, 4) absolute transform
        joints: (..., 4) joint angles in degrees
    """
    pos_rel = a13d[..., :3]
    r6d = a13d[..., 3:9]
    joints = a13d[..., 9:13]

    R_rel = rot6d_to_rotmat(r6d)      # (..., 3, 3)

    # Build relative 4x4
    shape = a13d.shape[:-1]
    T_rel = np.zeros((*shape, 4, 4), dtype=np.float32)
    T_rel[..., :3, :3] = R_rel
    T_rel[..., :3, 3] = pos_rel
    T_rel[..., 3, 3] = 1.0

    # T_abs = T_ref @ T_rel
    T_abs = T_ref @ T_rel

    return T_abs, joints


def rot6d_angular_error(r6d_pred: np.ndarray, r6d_true: np.ndarray) -> np.ndarray:
    """Compute angular error (degrees) between 6D rotation representations.

    Args:
        r6d_pred: (..., 6) predicted 6D rotation
        r6d_true: (..., 6) ground truth 6D rotation

    Returns:
        (...,) angular errors in degrees
    """
    shape = r6d_pred.shape[:-1]
    pred_flat = r6d_pred.reshape(-1, 6)
    true_flat = r6d_true.reshape(-1, 6)

    R_pred = rot6d_to_rotmat(pred_flat)
    R_true = rot6d_to_rotmat(true_flat)

    errors = np.zeros(pred_flat.shape[0])
    for i in range(len(errors)):
        r_pred = Rotation.from_matrix(R_pred[i])
        r_true = Rotation.from_matrix(R_true[i])
        r_diff = r_true.inv() * r_pred
        errors[i] = r_diff.magnitude() * 180.0 / np.pi

    return errors.reshape(shape)
