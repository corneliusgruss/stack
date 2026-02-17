"""
Contact Module - Contact detection and mechanics.

Handles detection of contact between fingertip pad and thumb pad,
and computation of contact forces.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional
from .geometry import FingerGeometry, FingertipPad, ThumbGeometry


@dataclass
class ContactState:
    """Result of contact analysis."""
    in_contact: bool
    distance: float           # Distance between pad centers (mm)
    penetration: float        # Overlap if in contact (mm)
    fingertip_pad: np.ndarray # Fingertip pad center position
    thumb_tip: np.ndarray     # Thumb pad center position
    contact_point: np.ndarray # Point of contact (on fingertip surface)
    normal: np.ndarray        # Contact normal (fingertip → thumb)
    threshold: float          # Contact threshold (sum of radii)


def compute_contact(finger: FingerGeometry, fingertip: FingertipPad,
                    thumb: ThumbGeometry, theta1: float, theta2: float) -> ContactState:
    """
    Compute contact state between fingertip pad and thumb pad.

    Contact occurs when distance < fingertip_radius + thumb_radius.
    """
    # Pad positions
    pad_pos = fingertip.position(finger, theta1, theta2)
    thumb_pos = thumb.tip_position

    # Distance and direction
    to_thumb = thumb_pos - pad_pos
    distance = np.linalg.norm(to_thumb)
    normal = to_thumb / (distance + 1e-10)

    # Contact threshold
    threshold = fingertip.radius + thumb.pad_radius

    # Contact point on fingertip surface
    contact_point = pad_pos + normal * fingertip.radius

    in_contact = distance < threshold
    penetration = max(0, threshold - distance)

    return ContactState(
        in_contact=in_contact,
        distance=distance,
        penetration=penetration,
        fingertip_pad=pad_pos,
        thumb_tip=thumb_pos,
        contact_point=contact_point,
        normal=normal,
        threshold=threshold
    )


def compute_contact_force(finger: FingerGeometry, contact: ContactState,
                          net_torques: np.ndarray, theta1: float, theta2: float) -> float:
    """
    Compute contact force from torque balance.

    At equilibrium with contact:
        τ_net - J^T · n · F = 0
        F = τ_net · (J^T · n) / |J^T · n|²

    Args:
        finger: Finger geometry
        contact: Contact state
        net_torques: Cable torque + spring torque [τ₁, τ₂]
        theta1, theta2: Joint angles

    Returns:
        Contact force magnitude (N), 0 if not in contact
    """
    if not contact.in_contact:
        return 0.0

    # Jacobian
    J = finger.jacobian(theta1, theta2)

    # Contact Jacobian: J^T · n
    J_contact = J.T @ contact.normal

    # Solve: F = τ · J_contact / |J_contact|²
    J_norm_sq = np.dot(J_contact, J_contact)
    if J_norm_sq < 1e-10:
        return 0.0

    force = np.dot(net_torques, J_contact) / J_norm_sq
    return max(0, force)  # Unilateral: can only push
