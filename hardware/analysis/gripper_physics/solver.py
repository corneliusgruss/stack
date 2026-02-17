"""
Solver Module - Equilibrium solver.

Solves for finger configuration at given tension,
handling both free motion and contact constraints.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple
from .geometry import FingerGeometry, FingertipPad, ThumbGeometry
from .actuation import ActuationSystem
from .contact import ContactState, compute_contact, compute_contact_force


@dataclass
class EquilibriumResult:
    """Complete result of equilibrium solve."""
    theta1: float              # MCP angle (rad)
    theta2: float              # PIP angle (rad)
    tension: float             # Cable tension (N)
    contact: ContactState      # Contact information
    contact_force: float       # Normal force at contact (N)
    cable_torques: np.ndarray  # [τ₁, τ₂] from cable
    spring_torques: np.ndarray # [τ₁, τ₂] from springs
    net_torques: np.ndarray    # [τ₁, τ₂] net (before contact)

    @property
    def theta1_deg(self) -> float:
        return np.degrees(self.theta1)

    @property
    def theta2_deg(self) -> float:
        return np.degrees(self.theta2)

    @property
    def grip_span(self) -> float:
        """Distance from fingertip pad to thumb."""
        return self.contact.distance


class EquilibriumSolver:
    """
    Solves equilibrium for cable-driven finger with optional contact.

    Free: τ_cable + τ_spring = 0  →  θ = T·r/k
    Contact: finger stops at contact, force builds up
    """

    def __init__(self, finger: FingerGeometry, actuation: ActuationSystem,
                 fingertip: FingertipPad, thumb: ThumbGeometry):
        self.finger = finger
        self.actuation = actuation
        self.fingertip = fingertip
        self.thumb = thumb

    def solve(self, tension: float) -> EquilibriumResult:
        """
        Solve equilibrium at given tension.

        1. Compute free equilibrium
        2. Check for contact
        3. If contact: find contact configuration and force
        """
        # Free equilibrium
        theta1_free, theta2_free = self.actuation.free_equilibrium(tension)

        # Check contact at free position
        contact = compute_contact(
            self.finger, self.fingertip, self.thumb, theta1_free, theta2_free
        )

        if not contact.in_contact:
            # No contact - free solution is valid
            cable_tau = self.actuation.cable.joint_torques(tension)
            spring_tau = self.actuation.spring.joint_torques(theta1_free, theta2_free)

            return EquilibriumResult(
                theta1=theta1_free,
                theta2=theta2_free,
                tension=tension,
                contact=contact,
                contact_force=0.0,
                cable_torques=cable_tau,
                spring_torques=spring_tau,
                net_torques=cable_tau + spring_tau
            )

        # Contact! Find configuration at contact
        theta1_c, theta2_c = self._find_contact_config(tension)

        # Recompute contact at constrained position
        contact = compute_contact(
            self.finger, self.fingertip, self.thumb, theta1_c, theta2_c
        )

        # Compute torques and contact force
        cable_tau = self.actuation.cable.joint_torques(tension)
        spring_tau = self.actuation.spring.joint_torques(theta1_c, theta2_c)
        net_tau = cable_tau + spring_tau

        contact_force = compute_contact_force(
            self.finger, contact, net_tau, theta1_c, theta2_c
        )

        return EquilibriumResult(
            theta1=theta1_c,
            theta2=theta2_c,
            tension=tension,
            contact=contact,
            contact_force=contact_force,
            cable_torques=cable_tau,
            spring_torques=spring_tau,
            net_torques=net_tau
        )

    def _find_contact_config(self, tension: float) -> Tuple[float, float]:
        """
        Binary search for configuration at first contact.
        """
        t_low, t_high = 0.0, tension

        for _ in range(30):
            t_mid = (t_low + t_high) / 2
            theta1, theta2 = self.actuation.free_equilibrium(t_mid)
            contact = compute_contact(
                self.finger, self.fingertip, self.thumb, theta1, theta2
            )

            if contact.distance > contact.threshold:
                t_low = t_mid
            else:
                t_high = t_mid

        return self.actuation.free_equilibrium(t_high)

    def find_contact_tension(self) -> Optional[float]:
        """Find tension at which contact first occurs."""
        for t in np.linspace(0, 50, 500):
            theta1, theta2 = self.actuation.free_equilibrium(t)
            contact = compute_contact(
                self.finger, self.fingertip, self.thumb, theta1, theta2
            )
            if contact.in_contact:
                return t
        return None

    def solve_range(self, t_min: float, t_max: float, n_points: int = 50) -> list:
        """Solve equilibrium for a range of tensions."""
        tensions = np.linspace(t_min, t_max, n_points)
        return [self.solve(t) for t in tensions]
