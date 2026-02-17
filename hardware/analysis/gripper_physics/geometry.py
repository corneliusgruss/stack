"""
Geometry Module - Pure kinematics, no physics.

Defines the geometric primitives: finger links, thumb, fingertip pad.
All units: mm, radians.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Tuple


@dataclass
class FingerGeometry:
    """
    Two-link planar finger geometry.

    Coordinate system:
    - Origin at MCP joint
    - Y-axis points UP (toward fingertip when extended)
    - X-axis points RIGHT (toward thumb)
    - Positive angles = flexion (clockwise rotation)
    """
    L1: float  # Proximal link length (mm)
    L2: float  # Distal link length (mm)
    width1: float = 12.0  # Proximal link width for visualization (mm)
    width2: float = 10.0  # Distal link width (mm)
    base: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0]))

    def forward_kinematics(self, theta1: float, theta2: float) -> Dict[str, np.ndarray]:
        """
        Compute joint positions.

        Args:
            theta1: MCP angle (rad), positive = flexion
            theta2: PIP angle (rad), positive = flexion

        Returns:
            Dict with 'mcp', 'pip', 'tip' positions as (x, y) arrays
        """
        mcp = self.base.copy()

        # PIP: rotate theta1 from vertical, translate L1
        pip = mcp + self.L1 * np.array([np.sin(theta1), np.cos(theta1)])

        # Tip: rotate theta1+theta2 from vertical, translate L2
        total = theta1 + theta2
        tip = pip + self.L2 * np.array([np.sin(total), np.cos(total)])

        return {'mcp': mcp, 'pip': pip, 'tip': tip}

    def tip_position(self, theta1: float, theta2: float) -> np.ndarray:
        """Get fingertip position."""
        return self.forward_kinematics(theta1, theta2)['tip']

    def link_directions(self, theta1: float, theta2: float) -> Dict[str, np.ndarray]:
        """Get unit direction vectors for each link."""
        total = theta1 + theta2
        return {
            'proximal': np.array([np.sin(theta1), np.cos(theta1)]),
            'distal': np.array([np.sin(total), np.cos(total)])
        }

    def jacobian(self, theta1: float, theta2: float) -> np.ndarray:
        """
        Analytical Jacobian: ∂tip/∂(θ₁, θ₂)

        J = [L1·cos(θ₁) + L2·cos(θ₁+θ₂)   L2·cos(θ₁+θ₂)]
            [-L1·sin(θ₁) - L2·sin(θ₁+θ₂)  -L2·sin(θ₁+θ₂)]
        """
        c1, s1 = np.cos(theta1), np.sin(theta1)
        c12, s12 = np.cos(theta1 + theta2), np.sin(theta1 + theta2)

        return np.array([
            [self.L1 * c1 + self.L2 * c12, self.L2 * c12],
            [-self.L1 * s1 - self.L2 * s12, -self.L2 * s12]
        ])


@dataclass
class FingertipPad:
    """
    Fingertip contact pad - offset from kinematic tip toward flexor side.
    """
    radius: float = 6.0   # Pad radius (mm)
    offset: float = 3.0   # Offset from kinematic tip (mm)

    def position(self, finger: FingerGeometry, theta1: float, theta2: float) -> np.ndarray:
        """Compute pad center position."""
        fk = finger.forward_kinematics(theta1, theta2)
        tip, pip = fk['tip'], fk['pip']

        # Flexor normal: perpendicular to distal link, toward palm
        distal = tip - pip
        distal = distal / (np.linalg.norm(distal) + 1e-10)
        flexor_normal = np.array([distal[1], -distal[0]])  # 90° clockwise

        return tip + flexor_normal * self.offset

    def flexor_normal(self, finger: FingerGeometry, theta1: float, theta2: float) -> np.ndarray:
        """Get the flexor-side normal direction."""
        dirs = finger.link_directions(theta1, theta2)
        distal = dirs['distal']
        return np.array([distal[1], -distal[0]])


@dataclass
class ThumbGeometry:
    """
    Fixed thumb geometry for pinch grip.

    Anatomical model:
    - Thumb base is at the PALM (below and to the right of MCP)
    - Thumb extends from palm toward where fingertip curls to
    - Thumb tip faces the approaching fingertip

    In our coordinate system:
    - Finger MCP at origin, finger points UP when extended
    - Thumb base is at positive X, negative Y (palm area)
    - Thumb tip is where we want contact to occur
    """
    tip_position: np.ndarray   # Thumb pad center (where contact occurs)
    pad_radius: float = 8.0    # Thumb pad radius (mm)
    base_position: np.ndarray = field(default=None)  # Palm attachment point
    length: float = 40.0       # Thumb link length (mm)

    def __post_init__(self):
        if self.base_position is None:
            # Anatomical: base at palm, below and right of tip
            # Thumb points from palm (lower-right) toward fingertip path (upper-left)
            # Base is at: tip + offset toward lower-right
            self.base_position = self.tip_position + np.array([30.0, -25.0])

    @property
    def direction(self) -> np.ndarray:
        """Unit vector from base to tip."""
        d = self.tip_position - self.base_position
        return d / (np.linalg.norm(d) + 1e-10)

    @property
    def angle(self) -> float:
        """Thumb angle from base to tip (radians from +X axis)."""
        d = self.tip_position - self.base_position
        return np.arctan2(d[1], d[0])

    @property
    def angle_deg(self) -> float:
        """Thumb angle in degrees."""
        return np.degrees(self.angle)
