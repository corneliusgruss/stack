"""
Main entry point for gripper physics simulation.

Usage:
    python -m gripper_physics           # Interactive visualization
    python -m gripper_physics --verify  # Run verification tests
"""

import sys
import numpy as np

from .geometry import FingerGeometry, FingertipPad, ThumbGeometry
from .actuation import CableActuation, SpringReturn, ActuationSystem
from .solver import EquilibriumSolver
from .analysis import compute_optimal_thumb_position, analyze_design
from .visualization import run_interactive
from . import create_default_gripper


def run_verification():
    """Comprehensive verification of the physics model."""
    print("=" * 70)
    print("GRIPPER PHYSICS VERIFICATION")
    print("=" * 70)

    # Create default gripper
    finger, cable, spring, fingertip, thumb = create_default_gripper(contact_tension=15.0)
    actuation = ActuationSystem(cable, spring)
    solver = EquilibriumSolver(finger, actuation, fingertip, thumb)

    print(f"\n{'='*70}")
    print("PARAMETERS")
    print("=" * 70)
    print(f"  Finger:    L1={finger.L1}mm, L2={finger.L2}mm")
    print(f"  Cable:     r1={cable.r1}mm (MCP), r2={cable.r2}mm (PIP)")
    print(f"  Springs:   k1={spring.k1} N·mm/rad, k2={spring.k2} N·mm/rad")
    print(f"  Fingertip: radius={fingertip.radius}mm, offset={fingertip.offset}mm")
    print(f"  Thumb:     radius={thumb.pad_radius}mm, position=({thumb.tip_position[0]:.1f}, {thumb.tip_position[1]:.1f})")
    print(f"  Contact threshold: {fingertip.radius + thumb.pad_radius}mm")

    # Test 1: Kinematics
    print(f"\n{'='*70}")
    print("1. KINEMATICS CHECK")
    print("=" * 70)
    fk = finger.forward_kinematics(0, 0)
    print(f"  Extended (θ=0°, 0°): tip at ({fk['tip'][0]:.1f}, {fk['tip'][1]:.1f})")
    print(f"  Expected: (0, {finger.L1 + finger.L2})")

    fk90 = finger.forward_kinematics(np.pi/2, 0)
    print(f"  θ₁=90°, θ₂=0°:       tip at ({fk90['tip'][0]:.1f}, {fk90['tip'][1]:.1f})")
    print(f"  Expected: ({finger.L1 + finger.L2}, 0)")

    fk_full = finger.forward_kinematics(np.pi/2, np.pi/2)
    print(f"  θ₁=90°, θ₂=90°:      tip at ({fk_full['tip'][0]:.1f}, {fk_full['tip'][1]:.1f})")
    print(f"  Expected: ({finger.L1}, {-finger.L2})")

    # Test 2: Equilibrium
    print(f"\n{'='*70}")
    print("2. FREE EQUILIBRIUM (θ = T·r/k)")
    print("=" * 70)
    print("  T(N)    θ₁ (deg)   θ₂ (deg)   Expected θ₁   Expected θ₂")
    print("  " + "-" * 60)
    for T in [5, 10, 15, 20, 25]:
        theta1, theta2 = actuation.free_equilibrium(T)
        exp1 = min(np.degrees(T * cable.r1 / spring.k1), 90)
        exp2 = min(np.degrees(T * cable.r2 / spring.k2), 90)
        print(f"  {T:4d}     {np.degrees(theta1):5.1f}°     {np.degrees(theta2):5.1f}°"
              f"        {exp1:5.1f}°        {exp2:5.1f}°")

    # Test 3: Contact
    print(f"\n{'='*70}")
    print("3. CONTACT ANALYSIS")
    print("=" * 70)
    contact_t = solver.find_contact_tension()
    print(f"  Contact first occurs at T = {contact_t:.2f} N" if contact_t else "  No contact found")

    print("\n  Grip cycle:")
    print("  T(N)   θ₁      θ₂      Grip Span   Status      Force")
    print("  " + "-" * 60)
    for T in range(0, 26, 2):
        r = solver.solve(T)
        status = "CONTACT" if r.contact.in_contact else "open"
        force = f"{r.contact_force:.2f}N" if r.contact_force > 0.01 else "-"
        print(f"  {T:4d}  {r.theta1_deg:5.1f}°  {r.theta2_deg:5.1f}°"
              f"    {r.grip_span:6.1f}mm    {status:8s}  {force}")

    # Test 4: Torques
    print(f"\n{'='*70}")
    print("4. TORQUE ANALYSIS AT T=15N")
    print("=" * 70)
    r15 = solver.solve(15)
    print(f"  Cable torques:  τ₁={r15.cable_torques[0]:.1f} N·mm, τ₂={r15.cable_torques[1]:.1f} N·mm")
    print(f"  Spring torques: τ₁={r15.spring_torques[0]:.1f} N·mm, τ₂={r15.spring_torques[1]:.1f} N·mm")
    print(f"  Net torques:    τ₁={r15.net_torques[0]:.1f} N·mm, τ₂={r15.net_torques[1]:.1f} N·mm")
    print(f"  Contact force:  {r15.contact_force:.2f} N")

    # Test 5: Design metrics
    print(f"\n{'='*70}")
    print("5. DESIGN METRICS")
    print("=" * 70)
    metrics = analyze_design(finger, cable, spring, fingertip, thumb)
    print(f"  Contact tension:    {metrics['contact_tension']:.1f} N" if metrics['contact_tension'] else "  No contact")
    print(f"  Max grip span:      {metrics['max_grip_span']:.1f} mm (at T=0)")
    print(f"  Grip span at 20N:   {metrics['grip_span_at_20N']:.1f} mm")
    print(f"  Force at 20N:       {metrics['force_at_20N']:.2f} N")
    print(f"  Max cable travel:   {metrics['max_cable_travel']:.1f} mm")

    # Test 6: Trajectory summary
    print(f"\n{'='*70}")
    print("6. FINGERTIP TRAJECTORY")
    print("=" * 70)
    print("  T(N)    Pad X     Pad Y     Direction")
    print("  " + "-" * 50)
    prev_pad = None
    for T in [0, 5, 10, 15, 20]:
        theta1, theta2 = actuation.free_equilibrium(T)
        pad = fingertip.position(finger, theta1, theta2)
        if prev_pad is not None:
            dx, dy = pad - prev_pad
            direction = f"({dx:+.0f}, {dy:+.0f})"
        else:
            direction = "-"
        print(f"  {T:4d}    {pad[0]:5.1f}    {pad[1]:6.1f}     {direction}")
        prev_pad = pad

    print("\n" + "=" * 70)
    print("VERIFICATION COMPLETE")
    print("=" * 70)


def main():
    """Main entry point."""
    if '--verify' in sys.argv:
        run_verification()
    else:
        print("Starting Gripper Physics Simulation...")
        print("Use sliders to adjust parameters.")
        print("'Optimal Thumb' places thumb for contact at current tension.")
        print("Run with --verify for verification tests.\n")

        components = create_default_gripper(contact_tension=15.0)
        run_interactive(*components)


if __name__ == '__main__':
    main()
