"""
Actuation Module - Cable and spring physics.

Models the cable-driven actuation and return springs.
Includes cable path computation for visualization.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple
from .geometry import FingerGeometry


@dataclass
class CableActuation:
    """
    Cable-driven actuation with pulleys at each joint.

    Cable path: servo horn → MCP pulley → along proximal → PIP pulley → to tip
    Tension T creates flexion torques: τ = T × r
    """
    r1: float  # MCP moment arm / pulley radius (mm)
    r2: float  # PIP moment arm / pulley radius (mm)

    # Servo horn geometry
    servo_offset: Tuple[float, float] = (0.0, -25.0)  # Relative to MCP
    servo_horn_radius: float = 12.0  # mm

    def joint_torques(self, tension: float) -> np.ndarray:
        """Flexion torques from cable tension: τ = T·r"""
        return tension * np.array([self.r1, self.r2])

    def cable_travel(self, theta1: float, theta2: float) -> float:
        """Cable pull distance from extended: ΔL = r₁·θ₁ + r₂·θ₂"""
        return self.r1 * theta1 + self.r2 * theta2

    def tension_for_travel(self, travel: float, theta1: float, theta2: float) -> float:
        """Inverse: what tension gives this travel (approximate)."""
        current_travel = self.cable_travel(theta1, theta2)
        if current_travel > 0:
            return travel / current_travel
        return 0.0

    def cable_path(self, finger: FingerGeometry, theta1: float, theta2: float) -> Dict:
        """
        Compute cable path points for visualization.

        Returns dict with all points needed to draw the cable.
        """
        fk = finger.forward_kinematics(theta1, theta2)
        dirs = finger.link_directions(theta1, theta2)

        # Servo horn center
        servo_center = fk['mcp'] + np.array(self.servo_offset)

        # Servo horn rotation based on cable travel
        travel = self.cable_travel(theta1, theta2)
        horn_angle = travel / self.servo_horn_radius
        servo_attach = servo_center + self.servo_horn_radius * np.array([
            np.sin(horn_angle), np.cos(horn_angle)
        ])

        # Flexor normals for each link
        prox_normal = np.array([dirs['proximal'][1], -dirs['proximal'][0]])
        dist_normal = np.array([dirs['distal'][1], -dirs['distal'][0]])

        # Cable guide points
        mcp_guide = fk['mcp'] + prox_normal * self.r1
        pip_entry = fk['pip'] + prox_normal * self.r2
        pip_exit = fk['pip'] + dist_normal * self.r2
        tip_anchor = fk['tip'] + dist_normal * (self.r2 * 0.5)

        return {
            'servo_center': servo_center,
            'servo_radius': self.servo_horn_radius,
            'servo_attach': servo_attach,
            'mcp_guide': mcp_guide,
            'pip_entry': pip_entry,
            'pip_exit': pip_exit,
            'tip_anchor': tip_anchor,
            'path': [servo_attach, mcp_guide, pip_entry, pip_exit, tip_anchor],
            'mcp_pulley_center': fk['mcp'],
            'pip_pulley_center': fk['pip'],
        }


@dataclass
class SpringReturn:
    """
    Torsion springs at each joint providing extension torque.

    τ_spring = -k·θ (restoring toward θ=0)
    """
    k1: float  # MCP spring constant (N·mm/rad)
    k2: float  # PIP spring constant (N·mm/rad)

    def joint_torques(self, theta1: float, theta2: float) -> np.ndarray:
        """Extension torques (negative = opposing flexion)."""
        return -np.array([self.k1 * theta1, self.k2 * theta2])

    def elastic_path(self, finger: FingerGeometry, theta1: float, theta2: float) -> Dict:
        """
        Compute elastic path on extensor (back) side for visualization.
        """
        fk = finger.forward_kinematics(theta1, theta2)
        dirs = finger.link_directions(theta1, theta2)

        # Extensor normals (opposite of flexor)
        prox_back = np.array([-dirs['proximal'][1], dirs['proximal'][0]])
        dist_back = np.array([-dirs['distal'][1], dirs['distal'][0]])

        offset = 4.0  # mm from link centerline

        # Elastic anchor points
        palm_anchor = fk['mcp'] + np.array([-5.0, -15.0])
        mcp_back = fk['mcp'] + prox_back * offset
        pip_back_prox = fk['pip'] + prox_back * offset
        pip_back_dist = fk['pip'] + dist_back * offset
        tip_back = fk['tip'] + dist_back * offset

        # Stretch factor for coloring
        total_angle = theta1 + theta2
        stretch = min(1.0, total_angle / np.radians(120))

        return {
            'path': [palm_anchor, mcp_back, pip_back_prox, pip_back_dist, tip_back],
            'stretch': stretch,
            'palm_anchor': palm_anchor,
            'tip_anchor': tip_back,
        }


@dataclass
class ActuationSystem:
    """Combined cable + spring actuation."""
    cable: CableActuation
    spring: SpringReturn

    def net_torques(self, tension: float, theta1: float, theta2: float) -> np.ndarray:
        """Net torques at each joint."""
        return self.cable.joint_torques(tension) + self.spring.joint_torques(theta1, theta2)

    def free_equilibrium(self, tension: float) -> Tuple[float, float]:
        """
        Solve free equilibrium: τ_cable + τ_spring = 0

        T·r = k·θ  →  θ = T·r/k
        """
        theta1 = tension * self.cable.r1 / self.spring.k1
        theta2 = tension * self.cable.r2 / self.spring.k2

        # Joint limits
        theta1 = np.clip(theta1, 0, np.pi / 2)
        theta2 = np.clip(theta2, 0, np.pi / 2)

        return theta1, theta2
