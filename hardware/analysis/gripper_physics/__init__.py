"""
Gripper Physics Simulation Package

A physics-based 2D simulation of the cable-driven gripper for ME740 project.

Modules:
- geometry: Finger, thumb, and pad geometry
- actuation: Cable and spring models
- contact: Contact detection and forces
- solver: Equilibrium solver
- analysis: Design analysis tools
- visualization: Interactive plotting

Usage:
    from gripper_physics import create_default_gripper, run_interactive

    gripper = create_default_gripper()
    run_interactive(*gripper)

Or run directly:
    python -m gripper_physics
    python -m gripper_physics --verify
"""

import numpy as np
from .geometry import FingerGeometry, FingertipPad, ThumbGeometry
from .actuation import CableActuation, SpringReturn, ActuationSystem
from .contact import compute_contact, compute_contact_force, ContactState
from .solver import EquilibriumSolver, EquilibriumResult
from .analysis import (compute_trajectory, compute_optimal_thumb_position,
                       compute_analysis_curves, analyze_design)
from .visualization import GripperSimulator, run_interactive


def create_default_gripper(contact_tension: float = 15.0):
    """
    Create default gripper components.

    Returns tuple: (finger, cable, spring, fingertip, thumb)
    """
    finger = FingerGeometry(L1=45.0, L2=40.0)
    cable = CableActuation(r1=10.0, r2=7.0)
    spring = SpringReturn(k1=100.0, k2=80.0)
    fingertip = FingertipPad(radius=6.0, offset=3.0)

    # Compute optimal thumb position
    actuation = ActuationSystem(cable, spring)
    thumb_pos = compute_optimal_thumb_position(finger, actuation, fingertip, contact_tension)
    thumb = ThumbGeometry(tip_position=thumb_pos, pad_radius=8.0)

    return finger, cable, spring, fingertip, thumb


__all__ = [
    # Geometry
    'FingerGeometry', 'FingertipPad', 'ThumbGeometry',
    # Actuation
    'CableActuation', 'SpringReturn', 'ActuationSystem',
    # Contact
    'compute_contact', 'compute_contact_force', 'ContactState',
    # Solver
    'EquilibriumSolver', 'EquilibriumResult',
    # Analysis
    'compute_trajectory', 'compute_optimal_thumb_position',
    'compute_analysis_curves', 'analyze_design',
    # Visualization
    'GripperSimulator', 'run_interactive',
    # Convenience
    'create_default_gripper',
]
