"""
Finger Kinematics and Force Analysis
ME740 Custom Gripper Project

This script analyzes:
1. Finger kinematics (joint angles → fingertip position)
2. Tendon forces (cable tension → joint torques)
3. Visualization of finger motion

Author: Cornelius Gruss
Date: January 26, 2026
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, FancyBboxPatch
from matplotlib.animation import FuncAnimation

# =============================================================================
# FINGER PARAMETERS (Large hand, 2-joint index finger)
# =============================================================================

# Link lengths (mm)
L1 = 45.0  # Proximal phalanx (base to PIP)
L2 = 45.0  # Distal phalanx (PIP to fingertip) - combined middle + tip

# Moment arms for tendon routing (mm)
r1 = 10.0  # Moment arm at MCP joint
r2 = 8.0   # Moment arm at PIP joint

# Joint angle limits (degrees)
MCP_MIN, MCP_MAX = 0, 90    # MCP joint range
PIP_MIN, PIP_MAX = 0, 100   # PIP joint range

# CoinFT sensor
COINFT_DIAMETER = 20.0  # mm
COINFT_THICKNESS = 3.0  # mm

# =============================================================================
# FORWARD KINEMATICS
# =============================================================================

def forward_kinematics(theta1_deg, theta2_deg, origin=(0, 0)):
    """
    Calculate joint and fingertip positions given joint angles.

    Args:
        theta1_deg: MCP joint angle (degrees, 0 = straight)
        theta2_deg: PIP joint angle (degrees, 0 = straight)
        origin: (x, y) position of MCP joint

    Returns:
        dict with positions of MCP, PIP, and fingertip
    """
    theta1 = np.radians(theta1_deg)
    theta2 = np.radians(theta2_deg)

    # MCP joint position (origin)
    mcp = np.array(origin)

    # PIP joint position
    pip = mcp + L1 * np.array([np.sin(theta1), np.cos(theta1)])

    # Fingertip position (angles accumulate)
    total_angle = theta1 + theta2
    tip = pip + L2 * np.array([np.sin(total_angle), np.cos(total_angle)])

    return {
        'mcp': mcp,
        'pip': pip,
        'tip': tip,
        'theta1': theta1_deg,
        'theta2': theta2_deg
    }


# =============================================================================
# TENDON FORCE ANALYSIS
# =============================================================================

def tendon_torques(cable_tension, r1=r1, r2=r2):
    """
    Calculate joint torques from cable tension.

    For a tendon routed through both joints:
    - Torque at each joint = tension × moment arm

    Args:
        cable_tension: Force in cable (N)
        r1: Moment arm at MCP (mm)
        r2: Moment arm at PIP (mm)

    Returns:
        dict with torques at each joint (N·mm)
    """
    tau1 = cable_tension * r1  # MCP torque
    tau2 = cable_tension * r2  # PIP torque

    return {
        'tau_mcp': tau1,
        'tau_pip': tau2,
        'tension': cable_tension
    }


def required_tension(desired_torque_mcp, r1=r1):
    """
    Calculate required cable tension for a desired MCP torque.
    """
    return desired_torque_mcp / r1


def cable_travel(theta1_deg, theta2_deg, r1=r1, r2=r2):
    """
    Calculate how much cable is pulled for given joint angles.

    Cable travel at each joint ≈ angle × moment arm (for small angles)
    More precisely: arc length = r × theta (radians)

    Returns:
        Total cable travel in mm
    """
    theta1 = np.radians(theta1_deg)
    theta2 = np.radians(theta2_deg)

    travel1 = r1 * theta1
    travel2 = r2 * theta2

    return travel1 + travel2


# =============================================================================
# VISUALIZATION
# =============================================================================

def draw_finger(ax, theta1, theta2, origin=(0, 0), color='steelblue',
                show_tendon=True, show_sensors=True):
    """
    Draw the finger at given joint angles.
    """
    fk = forward_kinematics(theta1, theta2, origin)

    # Draw links
    link_width = 15

    # Proximal link
    ax.plot([fk['mcp'][0], fk['pip'][0]],
            [fk['mcp'][1], fk['pip'][1]],
            color=color, linewidth=link_width, solid_capstyle='round')

    # Distal link
    ax.plot([fk['pip'][0], fk['tip'][0]],
            [fk['pip'][1], fk['tip'][1]],
            color=color, linewidth=link_width, solid_capstyle='round')

    # Draw joints
    ax.plot(*fk['mcp'], 'o', color='darkblue', markersize=12, zorder=5)
    ax.plot(*fk['pip'], 'o', color='darkblue', markersize=10, zorder=5)
    ax.plot(*fk['tip'], 'o', color='darkblue', markersize=8, zorder=5)

    # Draw CoinFT sensor on fingertip
    if show_sensors:
        sensor = Circle(fk['tip'], COINFT_DIAMETER/2,
                       color='gold', alpha=0.7, zorder=6)
        ax.add_patch(sensor)
        ax.annotate('CoinFT', fk['tip'], fontsize=8, ha='center', va='center')

    # Draw tendon path
    if show_tendon:
        theta1_rad = np.radians(theta1)
        theta2_rad = np.radians(theta2)

        # Tendon attachment points (offset from joint centers by moment arm)
        # Simplified: just show the cable path
        cable_start = fk['mcp'] - np.array([r1, 0])

        ax.plot([cable_start[0], fk['mcp'][0] - r1*np.cos(theta1_rad)],
                [cable_start[1], fk['mcp'][1] + r1*np.sin(theta1_rad)],
                'r-', linewidth=1, label='Tendon')

    return fk


def draw_thumb(ax, position, angle=30):
    """
    Draw the fixed thumb.
    """
    thumb_length = 40
    thumb_width = 20

    theta = np.radians(angle)

    # Thumb base
    ax.plot(*position, 's', color='coral', markersize=15, zorder=5)

    # Thumb direction
    tip = position + thumb_length * np.array([-np.cos(theta), np.sin(theta)])
    ax.plot([position[0], tip[0]], [position[1], tip[1]],
            color='coral', linewidth=12, solid_capstyle='round')

    # CoinFT on thumb
    sensor = Circle(tip, COINFT_DIAMETER/2, color='gold', alpha=0.7, zorder=6)
    ax.add_patch(sensor)

    return tip


def visualize_finger_workspace():
    """
    Show the finger's workspace (all reachable positions).
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    # Sample joint angles
    theta1_range = np.linspace(MCP_MIN, MCP_MAX, 20)
    theta2_range = np.linspace(PIP_MIN, PIP_MAX, 20)

    tips = []
    for t1 in theta1_range:
        for t2 in theta2_range:
            fk = forward_kinematics(t1, t2)
            tips.append(fk['tip'])

    tips = np.array(tips)

    # Plot workspace
    ax.scatter(tips[:, 0], tips[:, 1], alpha=0.3, s=10, c='lightblue')

    # Draw finger at a few positions
    for t1 in [0, 30, 60, 90]:
        draw_finger(ax, t1, t1*0.8, show_tendon=False, show_sensors=False,
                   color=plt.cm.viridis(t1/90))

    # Draw thumb
    thumb_pos = np.array([-30, 20])
    draw_thumb(ax, thumb_pos)

    ax.set_xlim(-80, 100)
    ax.set_ylim(-20, 120)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_title('Index Finger Workspace with Fixed Thumb')

    plt.tight_layout()
    return fig


def visualize_gripper_closing():
    """
    Animate the gripper closing motion.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    # Left: Gripper visualization
    ax1 = axes[0]
    # Right: Cable travel and torque plots
    ax2 = axes[1]

    # Thumb position
    thumb_pos = np.array([-25, 25])

    # Animation frames
    n_frames = 50
    theta1_sequence = np.linspace(0, 80, n_frames)
    theta2_sequence = np.linspace(0, 70, n_frames)  # PIP follows MCP

    # Storage for plots
    cable_travels = []
    torques_mcp = []
    torques_pip = []

    cable_tension = 10  # N (example)

    for t1, t2 in zip(theta1_sequence, theta2_sequence):
        cable_travels.append(cable_travel(t1, t2))
        torques = tendon_torques(cable_tension)
        torques_mcp.append(torques['tau_mcp'])
        torques_pip.append(torques['tau_pip'])

    def animate(frame):
        ax1.clear()
        ax2.clear()

        t1 = theta1_sequence[frame]
        t2 = theta2_sequence[frame]

        # Draw gripper
        draw_finger(ax1, t1, t2, show_sensors=True)
        draw_thumb(ax1, thumb_pos)

        # Draw fixed fingers (simplified as a block)
        fixed_block = FancyBboxPatch((15, 0), 30, 80,
                                      boxstyle="round,pad=0.05",
                                      facecolor='lightgray',
                                      edgecolor='gray',
                                      alpha=0.5)
        ax1.add_patch(fixed_block)
        ax1.annotate('Fixed\nFingers', (30, 40), ha='center', fontsize=9)

        ax1.set_xlim(-80, 80)
        ax1.set_ylim(-20, 120)
        ax1.set_aspect('equal')
        ax1.set_title(f'Gripper Position\nMCP: {t1:.0f}°, PIP: {t2:.0f}°')
        ax1.grid(True, alpha=0.3)

        # Plot cable travel
        ax2.plot(theta1_sequence[:frame+1], cable_travels[:frame+1], 'b-',
                label='Cable Travel (mm)')
        ax2.axhline(y=cable_travels[frame], color='b', linestyle='--', alpha=0.5)

        ax2.set_xlim(0, 90)
        ax2.set_ylim(0, max(cable_travels)*1.1)
        ax2.set_xlabel('MCP Angle (degrees)')
        ax2.set_ylabel('Cable Travel (mm)')
        ax2.set_title(f'Cable Travel: {cable_travels[frame]:.1f} mm')
        ax2.legend(loc='upper left')
        ax2.grid(True, alpha=0.3)

        return []

    anim = FuncAnimation(fig, animate, frames=n_frames, interval=50, blit=True)

    plt.tight_layout()
    return fig, anim


def analyze_forces():
    """
    Analyze tendon forces and joint torques.
    """
    print("=" * 60)
    print("TENDON FORCE ANALYSIS")
    print("=" * 60)

    print(f"\nFinger Parameters:")
    print(f"  Proximal link (L1): {L1} mm")
    print(f"  Distal link (L2): {L2} mm")
    print(f"  MCP moment arm (r1): {r1} mm")
    print(f"  PIP moment arm (r2): {r2} mm")

    # Example: gripping an object with 5N force at fingertip
    grip_force = 5.0  # N

    # Rough estimate: to generate grip force, we need cable tension
    # This depends on finger configuration, but roughly:
    # Grip force ≈ cable_tension × (r1 + r2) / (L1 + L2) × some_factor

    print(f"\n--- Scenario: {grip_force} N grip force at fingertip ---")

    # For a rough estimate, assume finger at 45° each joint
    theta1, theta2 = 45, 45

    # Required cable tension (rough approximation)
    # More accurate would need full Jacobian analysis
    estimated_tension = grip_force * (L1 + L2) / (r1 + r2)

    print(f"\n  At MCP={theta1}°, PIP={theta2}°:")
    print(f"  Estimated cable tension: {estimated_tension:.1f} N")

    torques = tendon_torques(estimated_tension)
    print(f"  MCP torque: {torques['tau_mcp']:.1f} N·mm = {torques['tau_mcp']/10:.2f} N·cm")
    print(f"  PIP torque: {torques['tau_pip']:.1f} N·mm = {torques['tau_pip']/10:.2f} N·cm")

    travel = cable_travel(theta1, theta2)
    print(f"  Cable travel: {travel:.1f} mm")

    print(f"\n--- Cable travel for full range of motion ---")
    full_travel = cable_travel(MCP_MAX, PIP_MAX)
    print(f"  MCP: 0→{MCP_MAX}°, PIP: 0→{PIP_MAX}°")
    print(f"  Total cable travel: {full_travel:.1f} mm")

    print(f"\n--- Servo/Motor requirements ---")
    # If using a servo with a pulley
    pulley_radius = 10  # mm
    rotation_needed = full_travel / pulley_radius  # radians
    rotation_degrees = np.degrees(rotation_needed)
    print(f"  With {pulley_radius}mm pulley radius:")
    print(f"  Rotation needed: {rotation_degrees:.0f}° ({rotation_needed:.2f} rad)")

    # Torque at servo
    max_tension = 20  # N (reasonable for finger gripper)
    servo_torque = max_tension * pulley_radius
    print(f"  For {max_tension}N cable tension:")
    print(f"  Servo torque needed: {servo_torque:.0f} N·mm = {servo_torque/10:.0f} N·cm")


def plot_force_analysis():
    """
    Plot force/torque relationships.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. Cable travel vs joint angles
    ax1 = axes[0, 0]
    theta1_range = np.linspace(0, 90, 50)

    for theta2 in [0, 30, 60, 90]:
        travels = [cable_travel(t1, theta2) for t1 in theta1_range]
        ax1.plot(theta1_range, travels, label=f'PIP={theta2}°')

    ax1.set_xlabel('MCP Angle (degrees)')
    ax1.set_ylabel('Cable Travel (mm)')
    ax1.set_title('Cable Travel vs Joint Angles')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Joint torques vs cable tension
    ax2 = axes[0, 1]
    tensions = np.linspace(0, 30, 50)

    tau_mcp = [tendon_torques(t)['tau_mcp'] for t in tensions]
    tau_pip = [tendon_torques(t)['tau_pip'] for t in tensions]

    ax2.plot(tensions, tau_mcp, 'b-', label=f'MCP (r={r1}mm)')
    ax2.plot(tensions, tau_pip, 'r-', label=f'PIP (r={r2}mm)')

    ax2.set_xlabel('Cable Tension (N)')
    ax2.set_ylabel('Joint Torque (N·mm)')
    ax2.set_title('Joint Torques vs Cable Tension')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Moment arm effect
    ax3 = axes[1, 0]
    r_range = np.linspace(4, 15, 50)
    tension = 15  # N

    torques_at_r = [tension * r for r in r_range]
    travel_at_r = [r * np.radians(90) for r in r_range]  # for 90° rotation

    ax3.plot(r_range, torques_at_r, 'b-', label='Torque (N·mm)')
    ax3.set_xlabel('Moment Arm (mm)')
    ax3.set_ylabel('Torque (N·mm)', color='b')
    ax3.tick_params(axis='y', labelcolor='b')

    ax3_twin = ax3.twinx()
    ax3_twin.plot(r_range, travel_at_r, 'r-', label='Cable travel for 90°')
    ax3_twin.set_ylabel('Cable Travel for 90° (mm)', color='r')
    ax3_twin.tick_params(axis='y', labelcolor='r')

    ax3.set_title(f'Moment Arm Trade-off (at {tension}N tension)')
    ax3.grid(True, alpha=0.3)

    # 4. Gripper closing sequence
    ax4 = axes[1, 1]

    # Simulate coordinated closing
    t1_seq = np.linspace(0, 80, 100)
    t2_seq = t1_seq * 0.9  # PIP follows MCP with slight lag

    # Calculate fingertip trajectory
    tips_x = []
    tips_y = []
    for t1, t2 in zip(t1_seq, t2_seq):
        fk = forward_kinematics(t1, t2)
        tips_x.append(fk['tip'][0])
        tips_y.append(fk['tip'][1])

    ax4.plot(tips_x, tips_y, 'b-', linewidth=2)
    ax4.plot(tips_x[0], tips_y[0], 'go', markersize=10, label='Start')
    ax4.plot(tips_x[-1], tips_y[-1], 'ro', markersize=10, label='End')

    # Draw thumb target
    thumb_x, thumb_y = -25, 25
    ax4.plot(thumb_x, thumb_y, 's', color='coral', markersize=15, label='Thumb')

    ax4.set_xlabel('X (mm)')
    ax4.set_ylabel('Y (mm)')
    ax4.set_title('Fingertip Trajectory During Closing')
    ax4.set_aspect('equal')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("FINGER KINEMATICS AND FORCE ANALYSIS")
    print("="*60 + "\n")

    # Run force analysis
    analyze_forces()

    # Generate plots
    print("\n\nGenerating visualizations...")

    fig1 = visualize_finger_workspace()
    fig1.savefig('finger_workspace.png', dpi=150, bbox_inches='tight')
    print("  Saved: finger_workspace.png")

    fig2 = plot_force_analysis()
    fig2.savefig('force_analysis.png', dpi=150, bbox_inches='tight')
    print("  Saved: force_analysis.png")

    print("\nDone! Check the generated PNG files.")

    # Show plots (comment out if running headless)
    plt.show()
