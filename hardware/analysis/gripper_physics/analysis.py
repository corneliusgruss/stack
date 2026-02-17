"""
Analysis Module - Design and analysis tools.

Provides tools for:
- Trajectory computation
- Force/torque analysis
- Optimal thumb placement
- Design parameter sweeps
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional
from .geometry import FingerGeometry, FingertipPad, ThumbGeometry
from .actuation import CableActuation, SpringReturn, ActuationSystem
from .solver import EquilibriumSolver, EquilibriumResult


@dataclass
class TrajectoryPoint:
    """Single point on fingertip trajectory."""
    tension: float
    theta1: float
    theta2: float
    tip_position: np.ndarray
    pad_position: np.ndarray


def compute_trajectory(finger: FingerGeometry, actuation: ActuationSystem,
                       fingertip: FingertipPad, t_range: Tuple[float, float] = (0, 25),
                       n_points: int = 100) -> List[TrajectoryPoint]:
    """
    Compute fingertip trajectory as tension varies.

    Returns list of TrajectoryPoints from t_min to t_max.
    """
    tensions = np.linspace(t_range[0], t_range[1], n_points)
    trajectory = []

    for t in tensions:
        theta1, theta2 = actuation.free_equilibrium(t)
        fk = finger.forward_kinematics(theta1, theta2)
        pad = fingertip.position(finger, theta1, theta2)

        trajectory.append(TrajectoryPoint(
            tension=t,
            theta1=theta1,
            theta2=theta2,
            tip_position=fk['tip'],
            pad_position=pad
        ))

    return trajectory


def compute_optimal_thumb_position(finger: FingerGeometry, actuation: ActuationSystem,
                                   fingertip: FingertipPad,
                                   contact_tension: float = 15.0) -> np.ndarray:
    """
    Compute optimal thumb position for pinch grip.

    Key insight: Place thumb at the END of fingertip trajectory so the
    grip CLOSES (distance decreases) as tension increases.

    Args:
        contact_tension: Desired tension at which contact occurs

    Returns:
        Optimal thumb tip position (x, y)
    """
    # Position at contact tension
    theta1_c, theta2_c = actuation.free_equilibrium(contact_tension)
    pad_contact = fingertip.position(finger, theta1_c, theta2_c)

    # Position at max tension (90°, 90°)
    pad_max = fingertip.position(finger, np.pi/2, np.pi/2)

    # Direction from contact to max
    to_max = pad_max - pad_contact
    dist = np.linalg.norm(to_max)

    if dist > 1e-6:
        direction = to_max / dist
        # Place thumb slightly past contact position toward max
        threshold = fingertip.radius + 8.0  # Assume 8mm thumb pad
        return pad_contact + direction * (threshold * 0.3)
    else:
        return pad_contact


@dataclass
class AnalysisCurves:
    """Collection of analysis curves."""
    tensions: np.ndarray
    # Angles
    theta1: np.ndarray
    theta2: np.ndarray
    # Torques
    cable_torque_mcp: np.ndarray
    cable_torque_pip: np.ndarray
    spring_torque_mcp: np.ndarray
    spring_torque_pip: np.ndarray
    net_torque_mcp: np.ndarray
    net_torque_pip: np.ndarray
    # Contact
    grip_span: np.ndarray
    contact_force: np.ndarray
    in_contact: np.ndarray
    # Positions
    tip_x: np.ndarray
    tip_y: np.ndarray
    pad_x: np.ndarray
    pad_y: np.ndarray


def compute_analysis_curves(solver: EquilibriumSolver,
                            t_range: Tuple[float, float] = (0, 25),
                            n_points: int = 100) -> AnalysisCurves:
    """
    Compute all analysis curves for a range of tensions.

    Returns AnalysisCurves with arrays for plotting.
    """
    results = solver.solve_range(t_range[0], t_range[1], n_points)

    return AnalysisCurves(
        tensions=np.array([r.tension for r in results]),
        theta1=np.array([r.theta1 for r in results]),
        theta2=np.array([r.theta2 for r in results]),
        cable_torque_mcp=np.array([r.cable_torques[0] for r in results]),
        cable_torque_pip=np.array([r.cable_torques[1] for r in results]),
        spring_torque_mcp=np.array([r.spring_torques[0] for r in results]),
        spring_torque_pip=np.array([r.spring_torques[1] for r in results]),
        net_torque_mcp=np.array([r.net_torques[0] for r in results]),
        net_torque_pip=np.array([r.net_torques[1] for r in results]),
        grip_span=np.array([r.grip_span for r in results]),
        contact_force=np.array([r.contact_force for r in results]),
        in_contact=np.array([r.contact.in_contact for r in results]),
        tip_x=np.array([solver.finger.tip_position(r.theta1, r.theta2)[0] for r in results]),
        tip_y=np.array([solver.finger.tip_position(r.theta1, r.theta2)[1] for r in results]),
        pad_x=np.array([r.contact.fingertip_pad[0] for r in results]),
        pad_y=np.array([r.contact.fingertip_pad[1] for r in results]),
    )


def analyze_design(finger: FingerGeometry, cable: CableActuation, spring: SpringReturn,
                   fingertip: FingertipPad, thumb: ThumbGeometry) -> dict:
    """
    Comprehensive design analysis.

    Returns dict with key metrics:
    - contact_tension: when contact occurs
    - max_grip_span: at T=0
    - min_grip_span: at contact
    - force_at_20N: contact force at 20N tension
    - cable_travel_range: total cable travel
    """
    actuation = ActuationSystem(cable, spring)
    solver = EquilibriumSolver(finger, actuation, fingertip, thumb)

    # Contact tension
    contact_t = solver.find_contact_tension()

    # Grip span range
    result_0 = solver.solve(0)
    result_20 = solver.solve(20)

    # Cable travel
    theta1_max, theta2_max = actuation.free_equilibrium(30)
    max_travel = cable.cable_travel(theta1_max, theta2_max)

    return {
        'contact_tension': contact_t,
        'max_grip_span': result_0.grip_span,
        'grip_span_at_20N': result_20.grip_span,
        'force_at_20N': result_20.contact_force,
        'max_cable_travel': max_travel,
        'theta1_at_20N': result_20.theta1_deg,
        'theta2_at_20N': result_20.theta2_deg,
    }
