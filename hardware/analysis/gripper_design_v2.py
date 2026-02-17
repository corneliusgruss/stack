"""
Gripper Design v2 - Parallel Model

Fresh start with a simpler "parallel" model:
- At neutral (MCP=0°, PIP=0°), finger and thumb are parallel
- Both point in the same direction (vertically up in our coordinate system)
- Separated by a gap D between their inner (flexor) surfaces

This simplifies the geometry significantly for CAD.

Author: Cornelius Gruss
Started: January 2026
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple


# =============================================================================
# CORE PARAMETERS
# =============================================================================

@dataclass
class GripperParams:
    """All gripper parameters in one place."""
    # Finger
    L1: float = 45.0      # Proximal link length (mm)
    L2: float = 40.0      # Distal link length (mm)

    # Thumb
    L_thumb: float = 45.0 # Thumb length (mm)

    # Widths (same for both)
    W: float = 22.0       # Width of finger and thumb (mm)

    # Gap between inner surfaces at neutral
    D: float = 50.0       # Grip aperture at neutral (mm)

    # Cable actuation
    r: float = 8.0        # Pulley radius for both joints (mm)

    @property
    def finger_length(self) -> float:
        """Total finger length."""
        return self.L1 + self.L2

    @property
    def pad_offset(self) -> float:
        """Offset from centerline to contact surface."""
        return self.W / 2

    @property
    def centerline_distance(self) -> float:
        """Distance between finger and thumb centerlines."""
        return self.D + self.W


# Default parameters
PARAMS = GripperParams()


# =============================================================================
# COORDINATE SYSTEM
# =============================================================================
#
# At NEUTRAL (MCP=0°, PIP=0°):
#   - Finger and thumb are parallel vertical lines
#   - Finger MCP at origin (0, 0)
#   - Finger points UP (+Y direction)
#   - Thumb is to the RIGHT (+X direction)
#
#                Y (up)
#                ↑
#                │
#    finger tip  ●                    ● thumb tip
#                │                    │
#                │                    │
#                │                    │
#    finger MCP  ●────────────────────● thumb base ──→ X (right)
#              (0,0)
#
#                │←──── D + W ───────→│  (centerline to centerline)
#                   │←─── D ───→│        (inner surface to inner surface)
#
# Joint angles:
#   - MCP (θ1): 0° = neutral, positive = flex toward thumb
#   - PIP (θ2): 0° = straight, positive = flex (curl fingertip toward thumb)
#   - PIP cannot go negative (no hyperextension)
#


# =============================================================================
# GEOMETRY FUNCTIONS
# =============================================================================

def finger_positions(params: GripperParams, theta1: float, theta2: float) -> dict:
    """
    Calculate finger joint positions.

    Args:
        params: Gripper parameters
        theta1: MCP angle (rad), 0 = neutral (vertical), + = flex toward thumb
        theta2: PIP angle (rad), 0 = straight, + = flex

    Returns:
        Dict with 'mcp', 'pip', 'tip' positions
    """
    mcp = np.array([0.0, 0.0])

    # Proximal link direction (from vertical, flexing toward +X)
    pip = mcp + params.L1 * np.array([np.sin(theta1), np.cos(theta1)])

    # Distal link direction
    total_angle = theta1 + theta2
    tip = pip + params.L2 * np.array([np.sin(total_angle), np.cos(total_angle)])

    return {'mcp': mcp, 'pip': pip, 'tip': tip}


def finger_inner_surface(params: GripperParams, theta1: float, theta2: float) -> dict:
    """
    Calculate positions on the finger's inner (flexor) surface.

    The inner surface is offset from the centerline toward +X (toward thumb).
    """
    pos = finger_positions(params, theta1, theta2)

    # Inner surface offset direction (perpendicular to link, toward thumb)
    # For proximal link at angle theta1, inner normal is (cos(theta1), -sin(theta1))
    # For distal link at angle theta1+theta2, inner normal is (cos(theta1+theta2), -sin(theta1+theta2))

    prox_normal = np.array([np.cos(theta1), -np.sin(theta1)])
    dist_normal = np.array([np.cos(theta1 + theta2), -np.sin(theta1 + theta2)])

    offset = params.pad_offset

    return {
        'mcp_inner': pos['mcp'] + offset * np.array([1, 0]),  # At MCP, inner is just +X
        'pip_inner': pos['pip'] + offset * prox_normal,
        'tip_inner': pos['tip'] + offset * dist_normal,
    }


def thumb_positions(params: GripperParams) -> dict:
    """
    Calculate thumb positions (fixed, parallel to finger at neutral).

    Thumb is a straight line, parallel to Y axis, offset by centerline_distance.
    """
    base = np.array([params.centerline_distance, 0.0])
    tip = base + np.array([0.0, params.L_thumb])

    return {'base': base, 'tip': tip}


def thumb_inner_surface(params: GripperParams) -> dict:
    """
    Calculate positions on the thumb's inner (flexor) surface.

    The inner surface faces -X (toward the finger).
    """
    pos = thumb_positions(params)
    offset = params.pad_offset

    return {
        'base_inner': pos['base'] - np.array([offset, 0]),
        'tip_inner': pos['tip'] - np.array([offset, 0]),
    }


def grip_aperture(params: GripperParams, theta1: float, theta2: float) -> float:
    """
    Calculate grip aperture (distance between inner surfaces).

    Specifically, distance from fingertip inner surface to thumb inner surface.
    """
    finger_inner = finger_inner_surface(params, theta1, theta2)
    thumb_inner = thumb_inner_surface(params)

    # Distance from fingertip inner to closest point on thumb inner surface
    # (thumb is a vertical line, so we measure horizontal distance at the fingertip height)
    finger_tip_inner = finger_inner['tip_inner']
    thumb_x_inner = params.centerline_distance - params.pad_offset

    # Simple horizontal distance (approximate, good when finger is mostly vertical)
    aperture = thumb_x_inner - finger_tip_inner[0]

    return aperture


def find_contact_pip_angle(params: GripperParams, theta1: float = 0.0) -> float:
    """
    Find the PIP angle at which fingertip contacts thumb.

    Contact occurs when the inner surfaces touch (aperture ≈ 0).

    Args:
        theta1: MCP angle (default 0 = neutral)

    Returns:
        PIP angle (theta2) at contact
    """
    # Binary search for contact
    theta2_low, theta2_high = 0.0, np.pi/2

    for _ in range(50):
        theta2_mid = (theta2_low + theta2_high) / 2
        aperture = grip_aperture(params, theta1, theta2_mid)

        if aperture > 0:
            theta2_low = theta2_mid
        else:
            theta2_high = theta2_mid

    return theta2_high


# =============================================================================
# CABLE ACTUATION
# =============================================================================

def cable_travel(params: GripperParams, theta1: float, theta2: float) -> float:
    """Cable pull distance from neutral."""
    return params.r * theta1 + params.r * theta2


def cable_torque(params: GripperParams, tension: float) -> Tuple[float, float]:
    """Joint torques from cable tension."""
    return tension * params.r, tension * params.r


# =============================================================================
# VISUALIZATION DATA
# =============================================================================

def get_visualization_data(params: GripperParams, theta1: float, theta2: float) -> dict:
    """Get all data needed for visualization."""
    finger = finger_positions(params, theta1, theta2)
    finger_inner = finger_inner_surface(params, theta1, theta2)
    thumb = thumb_positions(params)
    thumb_inner = thumb_inner_surface(params)
    aperture = grip_aperture(params, theta1, theta2)

    return {
        'finger': finger,
        'finger_inner': finger_inner,
        'thumb': thumb,
        'thumb_inner': thumb_inner,
        'aperture': aperture,
        'theta1_deg': np.degrees(theta1),
        'theta2_deg': np.degrees(theta2),
    }


# =============================================================================
# SUMMARY OUTPUT
# =============================================================================

def print_summary(params: GripperParams = PARAMS):
    """Print design summary."""
    print("=" * 60)
    print("GRIPPER DESIGN v2 - PARALLEL MODEL")
    print("=" * 60)

    print("\n[PARAMETERS]")
    print(f"  Finger L1 (proximal):    {params.L1} mm")
    print(f"  Finger L2 (distal):      {params.L2} mm")
    print(f"  Finger total length:     {params.finger_length} mm")
    print(f"  Thumb length:            {params.L_thumb} mm")
    print(f"  Width (both):            {params.W} mm")
    print(f"  Gap D at neutral:        {params.D} mm")
    print(f"  Centerline distance:     {params.centerline_distance} mm")
    print(f"  Pulley radius r:         {params.r} mm")

    print("\n[NEUTRAL POSITION] (MCP=0°, PIP=0°)")
    print(f"  Finger MCP:              (0, 0)")
    print(f"  Finger tip:              (0, {params.finger_length})")
    print(f"  Thumb base:              ({params.centerline_distance}, 0)")
    print(f"  Thumb tip:               ({params.centerline_distance}, {params.L_thumb})")
    print(f"  Grip aperture:           {params.D} mm")

    # Find contact angle
    theta2_contact = find_contact_pip_angle(params, theta1=0.0)
    print(f"\n[CONTACT] (MCP=0°)")
    print(f"  PIP angle at contact:    {np.degrees(theta2_contact):.1f}°")
    print(f"  Cable travel to contact: {cable_travel(params, 0, theta2_contact):.1f} mm")

    # Trajectory at MCP=0
    print("\n[FINGERTIP TRAJECTORY] (MCP=0°, varying PIP)")
    print("  PIP     Tip X     Tip Y    Inner X   Inner Y   Aperture")
    print("  " + "-" * 55)

    for pip_deg in [0, 15, 30, 45, 60, 75, 90]:
        theta2 = np.radians(pip_deg)
        finger = finger_positions(params, 0, theta2)
        finger_inner = finger_inner_surface(params, 0, theta2)
        aperture = grip_aperture(params, 0, theta2)

        print(f"  {pip_deg:3d}°   {finger['tip'][0]:6.1f}   {finger['tip'][1]:6.1f}"
              f"   {finger_inner['tip_inner'][0]:6.1f}   {finger_inner['tip_inner'][1]:6.1f}"
              f"   {aperture:6.1f}")

    print("\n" + "=" * 60)

    # ASCII visualization at neutral
    print("\n[NEUTRAL POSITION - SIDE VIEW]")
    print("""
                    Y (up)
                    ↑
                    │
    finger tip  ●   │                      ● thumb tip
                │   │                      │
                │   │                      │
         L1+L2  │   │               L_thumb│
                │   │                      │
                │   │                      │
    finger MCP  ●───┼──────────────────────● thumb base ──→ X
              (0,0) │
                    │
                    │←─────── D={:.0f}mm ──────→│ (inner surfaces)
    """.format(params.D))


if __name__ == "__main__":
    print_summary()
