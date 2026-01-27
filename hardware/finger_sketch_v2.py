"""
Finger Sketch v2 - Right Hand, Correct Orientation
Focus on joint placement and actuation options
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle, Wedge, Arc
from matplotlib.transforms import Affine2D

def draw_right_hand_finger():
    """
    Draw right hand index finger - palm facing viewer.
    Thumb on LEFT, finger bends LEFT toward thumb.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 10))

    # ==========================================================
    # PANEL 1: Basic structure - palm view
    # ==========================================================
    ax1 = axes[0]
    ax1.set_xlim(-80, 80)
    ax1.set_ylim(-20, 160)
    ax1.set_aspect('equal')
    ax1.set_title('RIGHT HAND - Palm View\n(Finger extended)', fontsize=12, fontweight='bold')

    # Parameters
    L1 = 45  # proximal
    L2 = 45  # distal
    link_width = 20

    # Coordinate system:
    # - Y = up (toward fingertip)
    # - X = left is toward thumb
    # - Palm facing viewer

    # Draw palm
    palm = FancyBboxPatch((-50, 0), 100, 30,
                          boxstyle="round,pad=0.02",
                          facecolor='peachpuff',
                          edgecolor='black', linewidth=2)
    ax1.add_patch(palm)
    ax1.text(0, 15, 'PALM', ha='center', va='center', fontsize=11)

    # Draw thumb (on LEFT for right hand)
    thumb_base = (-40, 25)
    thumb_angle = 60  # pointing up-left
    thumb_length = 50
    thumb_tip = (thumb_base[0] - thumb_length * np.cos(np.radians(thumb_angle)),
                 thumb_base[1] + thumb_length * np.sin(np.radians(thumb_angle)))

    ax1.plot([thumb_base[0], thumb_tip[0]], [thumb_base[1], thumb_tip[1]],
             color='peachpuff', linewidth=25, solid_capstyle='round')
    ax1.plot([thumb_base[0], thumb_tip[0]], [thumb_base[1], thumb_tip[1]],
             color='black', linewidth=2)

    # Thumb CoinFT
    coinft_thumb = Circle(thumb_tip, 10, facecolor='gold', edgecolor='orange', linewidth=2, zorder=10)
    ax1.add_patch(coinft_thumb)
    ax1.text(-70, 90, 'THUMB\n(fixed)\n+ CoinFT', ha='center', fontsize=9)

    # Draw index finger (center, pointing up)
    finger_x = 10  # slightly right of center
    mcp_y = 35
    pip_y = mcp_y + L1
    tip_y = pip_y + L2

    # Proximal phalanx
    prox = FancyBboxPatch((finger_x - link_width/2, mcp_y + 4), link_width, L1 - 8,
                          boxstyle="round,pad=0.02",
                          facecolor='peachpuff', edgecolor='black', linewidth=2)
    ax1.add_patch(prox)

    # Distal phalanx
    dist = FancyBboxPatch((finger_x - link_width/2, pip_y + 4), link_width, L2 - 8,
                          boxstyle="round,pad=0.02",
                          facecolor='peachpuff', edgecolor='black', linewidth=2)
    ax1.add_patch(dist)

    # Joints
    mcp = Circle((finger_x, mcp_y), 6, facecolor='darkblue', edgecolor='black', linewidth=2, zorder=10)
    pip = Circle((finger_x, pip_y), 5, facecolor='darkblue', edgecolor='black', linewidth=2, zorder=10)
    ax1.add_patch(mcp)
    ax1.add_patch(pip)

    # Labels
    ax1.annotate('MCP\n(Joint 1)', xy=(finger_x, mcp_y), xytext=(45, mcp_y),
                 fontsize=10, ha='left', arrowprops=dict(arrowstyle='->', color='black'))
    ax1.annotate('PIP\n(Joint 2)', xy=(finger_x, pip_y), xytext=(45, pip_y),
                 fontsize=10, ha='left', arrowprops=dict(arrowstyle='->', color='black'))

    # CoinFT at fingertip
    coinft_finger = Circle((finger_x, tip_y - 5), 10, facecolor='gold',
                           edgecolor='orange', linewidth=2, zorder=10)
    ax1.add_patch(coinft_finger)
    ax1.text(finger_x, tip_y + 10, 'INDEX\n+ CoinFT', ha='center', fontsize=9)

    # Fixed fingers (middle, ring, pinky) - simplified
    for fx in [35, 50, 60]:
        fixed = FancyBboxPatch((fx - 7, 35), 14, 70,
                               boxstyle="round,pad=0.02",
                               facecolor='lightgray', edgecolor='gray',
                               linewidth=1, alpha=0.5)
        ax1.add_patch(fixed)
    ax1.text(50, 100, 'Fixed\nfingers', ha='center', fontsize=8, color='gray')

    # Arrow showing bend direction
    ax1.annotate('', xy=(-20, 100), xytext=(finger_x, 100),
                 arrowprops=dict(arrowstyle='->', color='green', lw=3))
    ax1.text(-5, 108, 'Bends toward\nthumb', ha='center', fontsize=9, color='green')

    ax1.set_xlabel('X (mm) ← Thumb side | Pinky side →')
    ax1.set_ylabel('Y (mm)')
    ax1.grid(True, alpha=0.3)

    # ==========================================================
    # PANEL 2: Flexed position - TOP VIEW showing rotation
    # ==========================================================
    ax2 = axes[1]
    ax2.set_xlim(-80, 80)
    ax2.set_ylim(-20, 160)
    ax2.set_aspect('equal')
    ax2.set_title('RIGHT HAND - Flexed 45°/45°\n(Finger curling toward thumb)', fontsize=12, fontweight='bold')

    # Draw palm
    palm2 = FancyBboxPatch((-50, 0), 100, 30,
                           boxstyle="round,pad=0.02",
                           facecolor='peachpuff', edgecolor='black', linewidth=2)
    ax2.add_patch(palm2)
    ax2.text(0, 15, 'PALM', ha='center', va='center', fontsize=11)

    # Draw thumb (same position)
    ax2.plot([thumb_base[0], thumb_tip[0]], [thumb_base[1], thumb_tip[1]],
             color='peachpuff', linewidth=25, solid_capstyle='round')
    ax2.plot([thumb_base[0], thumb_tip[0]], [thumb_base[1], thumb_tip[1]],
             color='black', linewidth=2)
    coinft_thumb2 = Circle(thumb_tip, 10, facecolor='gold', edgecolor='orange', linewidth=2, zorder=10)
    ax2.add_patch(coinft_thumb2)

    # Flexed finger
    theta1 = -45  # negative = bending LEFT (toward thumb)
    theta2 = -45
    theta1_rad = np.radians(theta1)
    theta2_rad = np.radians(theta2)

    mcp_pos = np.array([finger_x, mcp_y])
    pip_pos = mcp_pos + L1 * np.array([np.sin(theta1_rad), np.cos(theta1_rad)])
    tip_pos = pip_pos + L2 * np.array([np.sin(theta1_rad + theta2_rad), np.cos(theta1_rad + theta2_rad)])

    # Draw links
    ax2.plot([mcp_pos[0], pip_pos[0]], [mcp_pos[1], pip_pos[1]],
             color='peachpuff', linewidth=22, solid_capstyle='round')
    ax2.plot([pip_pos[0], tip_pos[0]], [pip_pos[1], tip_pos[1]],
             color='peachpuff', linewidth=22, solid_capstyle='round')

    # Outlines
    ax2.plot([mcp_pos[0], pip_pos[0]], [mcp_pos[1], pip_pos[1]],
             color='black', linewidth=2)
    ax2.plot([pip_pos[0], tip_pos[0]], [pip_pos[1], tip_pos[1]],
             color='black', linewidth=2)

    # Joints
    mcp2 = Circle(mcp_pos, 6, facecolor='darkblue', edgecolor='black', linewidth=2, zorder=10)
    pip2 = Circle(pip_pos, 5, facecolor='darkblue', edgecolor='black', linewidth=2, zorder=10)
    ax2.add_patch(mcp2)
    ax2.add_patch(pip2)

    # CoinFT at tip
    coinft2 = Circle(tip_pos, 10, facecolor='gold', edgecolor='orange', linewidth=2, zorder=10)
    ax2.add_patch(coinft2)

    # Show angles
    arc1 = Arc(mcp_pos, 30, 30, angle=90, theta1=theta1, theta2=0, color='purple', linewidth=2)
    ax2.add_patch(arc1)
    ax2.text(mcp_pos[0] + 20, mcp_pos[1] + 20, f'θ₁={-theta1}°', fontsize=10, color='purple')

    arc2 = Arc(pip_pos, 25, 25, angle=90+theta1, theta1=theta2, theta2=0, color='purple', linewidth=2)
    ax2.add_patch(arc2)
    ax2.text(pip_pos[0] + 5, pip_pos[1] + 18, f'θ₂={-theta2}°', fontsize=10, color='purple')

    # Cable path (on palm/thumb side of finger)
    cable_offset = 12  # toward thumb side (negative X)

    # Cable from palm
    cable_start = np.array([finger_x - cable_offset, 25])
    ax2.plot([cable_start[0], cable_start[0]], [5, 25], 'r-', linewidth=3)

    # Cable to MCP
    cable_mcp = mcp_pos + np.array([-cable_offset, 0])
    ax2.plot([cable_start[0], cable_mcp[0]], [cable_start[1], cable_mcp[1]], 'r-', linewidth=3)

    # Cable along proximal (perpendicular offset from link)
    perp1 = np.array([-np.cos(theta1_rad), np.sin(theta1_rad)])
    cable_pip = pip_pos + 10 * perp1
    ax2.plot([cable_mcp[0], cable_pip[0]], [cable_mcp[1], cable_pip[1]], 'r-', linewidth=3)

    # Cable along distal to anchor
    perp2 = np.array([-np.cos(theta1_rad + theta2_rad), np.sin(theta1_rad + theta2_rad)])
    cable_tip = tip_pos + 8 * perp2
    ax2.plot([cable_pip[0], cable_tip[0]], [cable_pip[1], cable_tip[1]], 'r-', linewidth=3)

    # Anchor
    ax2.plot(cable_tip[0], cable_tip[1], 'ro', markersize=8, zorder=15)

    # Pull arrow
    ax2.annotate('', xy=(cable_start[0], -5), xytext=(cable_start[0], 5),
                 arrowprops=dict(arrowstyle='->', color='red', lw=3))
    ax2.text(cable_start[0] - 5, -12, 'PULL', fontsize=10, color='red', fontweight='bold')

    # Fixed fingers (dimmed)
    for fx in [35, 50, 60]:
        fixed = FancyBboxPatch((fx - 7, 35), 14, 70,
                               boxstyle="round,pad=0.02",
                               facecolor='lightgray', edgecolor='gray',
                               linewidth=1, alpha=0.3)
        ax2.add_patch(fixed)

    ax2.text(-40, 145, 'Cable runs on\nTHUMB SIDE\nof finger', fontsize=9,
             color='red', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax2.set_xlabel('X (mm)')
    ax2.set_ylabel('Y (mm)')
    ax2.grid(True, alpha=0.3)

    # ==========================================================
    # PANEL 3: Joint offset + Motor-at-joint option
    # ==========================================================
    ax3 = axes[2]
    ax3.set_xlim(-60, 60)
    ax3.set_ylim(-20, 140)
    ax3.set_aspect('equal')
    ax3.set_title('JOINT OPTIONS\n(Cross-section at MCP)', fontsize=12, fontweight='bold')

    # Option A: Centered joint (standard)
    y_offset_a = 90
    ax3.text(0, 130, 'OPTION A: Centered Joint', ha='center', fontsize=11, fontweight='bold')

    # Link cross-section
    link_a = Rectangle((-20, y_offset_a), 40, 30, facecolor='peachpuff',
                       edgecolor='black', linewidth=2)
    ax3.add_patch(link_a)

    # Joint in center
    joint_a = Circle((0, y_offset_a + 15), 5, facecolor='darkblue', edgecolor='black', linewidth=2)
    ax3.add_patch(joint_a)
    ax3.text(0, y_offset_a + 15, '●', ha='center', va='center', fontsize=6, color='white')

    # Cable on thumb side
    ax3.plot([-15, -15], [y_offset_a - 10, y_offset_a + 40], 'r-', linewidth=3)
    ax3.annotate('Cable\n(10mm from\njoint)', xy=(-15, y_offset_a + 15),
                 xytext=(-45, y_offset_a + 15), fontsize=8, ha='right',
                 arrowprops=dict(arrowstyle='->', color='red'))

    # Moment arm
    ax3.annotate('', xy=(0, y_offset_a + 15), xytext=(-15, y_offset_a + 15),
                 arrowprops=dict(arrowstyle='<->', color='green', lw=2))
    ax3.text(-7, y_offset_a + 8, 'r=15mm', fontsize=8, color='green')

    # Option B: Offset joint (Sunday style)
    y_offset_b = 40
    ax3.text(0, 80, 'OPTION B: Offset Joint (Sunday style)', ha='center', fontsize=11, fontweight='bold')

    # Link cross-section
    link_b = Rectangle((-20, y_offset_b), 40, 30, facecolor='peachpuff',
                       edgecolor='black', linewidth=2)
    ax3.add_patch(link_b)

    # Joint offset toward thumb side (left)
    joint_b = Circle((-8, y_offset_b + 15), 5, facecolor='darkblue', edgecolor='black', linewidth=2)
    ax3.add_patch(joint_b)
    ax3.text(-8, y_offset_b + 15, '●', ha='center', va='center', fontsize=6, color='white')

    # Cable very close to joint
    ax3.plot([-18, -18], [y_offset_b - 10, y_offset_b + 40], 'r-', linewidth=3)

    # Moment arm (smaller)
    ax3.annotate('', xy=(-8, y_offset_b + 15), xytext=(-18, y_offset_b + 15),
                 arrowprops=dict(arrowstyle='<->', color='green', lw=2))
    ax3.text(-13, y_offset_b + 8, 'r=10mm', fontsize=8, color='green')

    ax3.text(25, y_offset_b + 15, '• More room on\n  pinky side\n• Joint closer to\n  cable', fontsize=8)

    # Option C: Motor at joint
    y_offset_c = -10
    ax3.text(0, 30, 'OPTION C: Motor AT Joint', ha='center', fontsize=11, fontweight='bold')

    # Motor/servo representation
    motor = FancyBboxPatch((-25, y_offset_c), 20, 25, boxstyle="round,pad=0.02",
                           facecolor='gray', edgecolor='black', linewidth=2)
    ax3.add_patch(motor)
    ax3.text(-15, y_offset_c + 12, 'Servo', ha='center', va='center', fontsize=8, color='white')

    # Servo horn = moment arm
    horn_angle = 30
    horn_length = 15
    horn_end = (-5 + horn_length * np.cos(np.radians(horn_angle)),
                y_offset_c + 12 + horn_length * np.sin(np.radians(horn_angle)))
    ax3.plot([-5, horn_end[0]], [y_offset_c + 12, horn_end[1]], 'gray', linewidth=4)
    ax3.plot(horn_end[0], horn_end[1], 'ko', markersize=6)

    # Cable from horn
    ax3.plot([horn_end[0], horn_end[0]], [horn_end[1], horn_end[1] + 20], 'r-', linewidth=3)

    # Link attached
    link_c = Rectangle((5, y_offset_c), 30, 25, facecolor='peachpuff',
                        edgecolor='black', linewidth=2)
    ax3.add_patch(link_c)

    ax3.text(35, y_offset_c + 12, '• No cable to palm\n• Direct drive\n• Servo horn = \n  moment arm', fontsize=8)

    ax3.set_xlabel('← Thumb side | Pinky side →')
    ax3.axhline(y=85, color='gray', linestyle='--', alpha=0.5)
    ax3.axhline(y=35, color='gray', linestyle='--', alpha=0.5)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


if __name__ == "__main__":
    fig = draw_right_hand_finger()
    fig.savefig('finger_sketch_v2.png', dpi=150, bbox_inches='tight')
    print("Saved: finger_sketch_v2.png")
    plt.show()
