"""
Detailed Finger Sketch showing joints and cable routing
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch, FancyArrowPatch, Arc
from matplotlib.lines import Line2D

def draw_detailed_finger():
    """
    Draw a detailed technical sketch of the 2-joint finger
    showing cable routing and joint locations.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 10))

    # ==========================================================
    # LEFT: Side view - EXTENDED position
    # ==========================================================
    ax1 = axes[0]
    ax1.set_xlim(-30, 50)
    ax1.set_ylim(-20, 160)
    ax1.set_aspect('equal')
    ax1.set_title('SIDE VIEW - Extended Position\n(showing cable routing)', fontsize=14, fontweight='bold')

    # Parameters
    L1 = 45  # proximal length
    L2 = 45  # distal length
    r1 = 10  # MCP moment arm
    r2 = 8   # PIP moment arm
    link_width = 18

    # Positions (finger pointing up, extended)
    mcp_y = 20
    pip_y = mcp_y + L1
    tip_y = pip_y + L2
    center_x = 10

    # Draw palm base
    palm = FancyBboxPatch((-15, 0), 50, 25,
                          boxstyle="round,pad=0.02",
                          facecolor='lightgray',
                          edgecolor='black',
                          linewidth=2)
    ax1.add_patch(palm)
    ax1.text(10, 12, 'PALM', ha='center', va='center', fontsize=10)

    # Draw proximal phalanx
    prox = FancyBboxPatch((center_x - link_width/2, mcp_y + 3), link_width, L1 - 6,
                          boxstyle="round,pad=0.02",
                          facecolor='steelblue',
                          edgecolor='black',
                          linewidth=2,
                          alpha=0.7)
    ax1.add_patch(prox)
    ax1.text(center_x, mcp_y + L1/2, 'Proximal\n45mm', ha='center', va='center',
             fontsize=9, color='white', fontweight='bold')

    # Draw distal phalanx
    dist = FancyBboxPatch((center_x - link_width/2, pip_y + 3), link_width, L2 - 6,
                          boxstyle="round,pad=0.02",
                          facecolor='steelblue',
                          edgecolor='black',
                          linewidth=2,
                          alpha=0.7)
    ax1.add_patch(dist)
    ax1.text(center_x, pip_y + L2/2, 'Distal\n45mm', ha='center', va='center',
             fontsize=9, color='white', fontweight='bold')

    # Draw MCP joint
    mcp_joint = Circle((center_x, mcp_y), 6, facecolor='darkblue',
                        edgecolor='black', linewidth=2, zorder=10)
    ax1.add_patch(mcp_joint)
    ax1.annotate('MCP Joint\n(Joint 1)', xy=(center_x, mcp_y),
                 xytext=(-25, mcp_y), fontsize=10,
                 ha='right', va='center',
                 arrowprops=dict(arrowstyle='->', color='black'))

    # Draw PIP joint
    pip_joint = Circle((center_x, pip_y), 5, facecolor='darkblue',
                        edgecolor='black', linewidth=2, zorder=10)
    ax1.add_patch(pip_joint)
    ax1.annotate('PIP Joint\n(Joint 2)', xy=(center_x, pip_y),
                 xytext=(-25, pip_y), fontsize=10,
                 ha='right', va='center',
                 arrowprops=dict(arrowstyle='->', color='black'))

    # Draw CoinFT sensor at fingertip
    coinft = Circle((center_x, tip_y - 5), 10, facecolor='gold',
                    edgecolor='orange', linewidth=2, zorder=10, alpha=0.8)
    ax1.add_patch(coinft)
    ax1.text(center_x, tip_y - 5, 'CoinFT\n20mm', ha='center', va='center', fontsize=7)

    # Draw TPU cover
    tpu = FancyBboxPatch((center_x - 12, tip_y - 18), 24, 5,
                         boxstyle="round,pad=0.01",
                         facecolor='black',
                         edgecolor='black',
                         linewidth=1,
                         alpha=0.7)
    ax1.add_patch(tpu)
    ax1.annotate('TPU cover\n1.5mm', xy=(center_x + 12, tip_y - 15),
                 xytext=(35, tip_y - 10), fontsize=8,
                 arrowprops=dict(arrowstyle='->', color='gray'))

    # ==========================================================
    # CABLE ROUTING
    # ==========================================================
    cable_x = center_x - r1  # cable runs on the palm side (left)

    # Cable from palm to MCP
    ax1.plot([cable_x, cable_x], [5, mcp_y - 3], 'r-', linewidth=3, label='Tendon/Cable')

    # Cable around MCP (moment arm = r1 = 10mm)
    ax1.plot([cable_x, cable_x], [mcp_y - 3, mcp_y + 3], 'r-', linewidth=3)

    # Cable from MCP to PIP (inside finger)
    cable_x_upper = center_x - r2  # slightly different moment arm
    ax1.plot([cable_x, cable_x_upper], [mcp_y + 3, pip_y - 3], 'r-', linewidth=3)

    # Cable around PIP
    ax1.plot([cable_x_upper, cable_x_upper], [pip_y - 3, pip_y + 3], 'r-', linewidth=3)

    # Cable to anchor point (fingertip)
    ax1.plot([cable_x_upper, center_x - 5], [pip_y + 3, tip_y - 15], 'r-', linewidth=3)

    # Anchor point
    ax1.plot(center_x - 5, tip_y - 15, 'ro', markersize=8, zorder=15)
    ax1.annotate('Cable\nanchor', xy=(center_x - 5, tip_y - 15),
                 xytext=(35, tip_y - 30), fontsize=8,
                 arrowprops=dict(arrowstyle='->', color='red'))

    # Moment arm annotations
    # MCP moment arm
    ax1.annotate('', xy=(center_x, mcp_y), xytext=(cable_x, mcp_y),
                 arrowprops=dict(arrowstyle='<->', color='green', lw=2))
    ax1.text((center_x + cable_x)/2, mcp_y - 8, f'r₁={r1}mm', ha='center',
             fontsize=9, color='green', fontweight='bold')

    # PIP moment arm
    ax1.annotate('', xy=(center_x, pip_y), xytext=(cable_x_upper, pip_y),
                 arrowprops=dict(arrowstyle='<->', color='green', lw=2))
    ax1.text((center_x + cable_x_upper)/2, pip_y - 8, f'r₂={r2}mm', ha='center',
             fontsize=9, color='green', fontweight='bold')

    # Pull direction arrow
    ax1.annotate('', xy=(cable_x, -10), xytext=(cable_x, 5),
                 arrowprops=dict(arrowstyle='->', color='red', lw=3))
    ax1.text(cable_x, -15, 'PULL\n(to trigger)', ha='center', fontsize=9, color='red')

    # Legend
    ax1.plot([], [], 'r-', linewidth=3, label='Cable/Tendon')
    ax1.plot([], [], 'o', color='darkblue', markersize=10, label='Joint (hinge)')
    ax1.legend(loc='upper right')

    ax1.set_xlabel('X (mm)')
    ax1.set_ylabel('Y (mm)')
    ax1.grid(True, alpha=0.3)

    # ==========================================================
    # RIGHT: Side view - FLEXED position (45°/45°)
    # ==========================================================
    ax2 = axes[1]
    ax2.set_xlim(-30, 100)
    ax2.set_ylim(-20, 120)
    ax2.set_aspect('equal')
    ax2.set_title('SIDE VIEW - Flexed Position (45°/45°)\n(showing how joints bend)', fontsize=14, fontweight='bold')

    # Joint angles
    theta1 = 45  # MCP angle
    theta2 = 45  # PIP angle

    theta1_rad = np.radians(theta1)
    theta2_rad = np.radians(theta2)

    # Calculate positions
    mcp_pos = np.array([10, 20])
    pip_pos = mcp_pos + L1 * np.array([np.sin(theta1_rad), np.cos(theta1_rad)])
    tip_pos = pip_pos + L2 * np.array([np.sin(theta1_rad + theta2_rad), np.cos(theta1_rad + theta2_rad)])

    # Draw palm
    palm2 = FancyBboxPatch((-15, 0), 50, 25,
                           boxstyle="round,pad=0.02",
                           facecolor='lightgray',
                           edgecolor='black',
                           linewidth=2)
    ax2.add_patch(palm2)
    ax2.text(10, 12, 'PALM', ha='center', va='center', fontsize=10)

    # Draw links as thick lines (easier for angled view)
    ax2.plot([mcp_pos[0], pip_pos[0]], [mcp_pos[1], pip_pos[1]],
             color='steelblue', linewidth=20, solid_capstyle='round', alpha=0.7)
    ax2.plot([pip_pos[0], tip_pos[0]], [pip_pos[1], tip_pos[1]],
             color='steelblue', linewidth=20, solid_capstyle='round', alpha=0.7)

    # Draw joints
    mcp_joint2 = Circle(mcp_pos, 6, facecolor='darkblue',
                        edgecolor='black', linewidth=2, zorder=10)
    ax2.add_patch(mcp_joint2)

    pip_joint2 = Circle(pip_pos, 5, facecolor='darkblue',
                        edgecolor='black', linewidth=2, zorder=10)
    ax2.add_patch(pip_joint2)

    # Draw CoinFT at tip
    coinft2 = Circle(tip_pos, 10, facecolor='gold',
                     edgecolor='orange', linewidth=2, zorder=10, alpha=0.8)
    ax2.add_patch(coinft2)
    ax2.text(tip_pos[0], tip_pos[1], 'CoinFT', ha='center', va='center', fontsize=7)

    # Draw cable path (simplified for angled view)
    cable_offset = 8

    # Cable direction perpendicular to link
    perp1 = np.array([-np.cos(theta1_rad), np.sin(theta1_rad)])
    perp2 = np.array([-np.cos(theta1_rad + theta2_rad), np.sin(theta1_rad + theta2_rad)])

    cable_mcp = mcp_pos + cable_offset * np.array([-1, 0])
    cable_pip = pip_pos + cable_offset * perp1
    cable_tip = tip_pos + cable_offset * perp2

    ax2.plot([cable_mcp[0], cable_mcp[0]], [5, mcp_pos[1]], 'r-', linewidth=3)
    ax2.plot([cable_mcp[0], cable_pip[0]], [mcp_pos[1], cable_pip[1]], 'r-', linewidth=3)
    ax2.plot([cable_pip[0], tip_pos[0] - 5], [cable_pip[1], tip_pos[1] - 5], 'r-', linewidth=3)

    # Draw angle arcs
    # MCP angle
    arc1 = Arc(mcp_pos, 25, 25, angle=90, theta1=-theta1, theta2=0, color='purple', linewidth=2)
    ax2.add_patch(arc1)
    ax2.text(mcp_pos[0] + 18, mcp_pos[1] + 15, f'θ₁={theta1}°', fontsize=10, color='purple')

    # PIP angle (relative to proximal link)
    arc2 = Arc(pip_pos, 20, 20, angle=90-theta1, theta1=-theta2, theta2=0, color='purple', linewidth=2)
    ax2.add_patch(arc2)
    ax2.text(pip_pos[0] + 15, pip_pos[1] + 5, f'θ₂={theta2}°', fontsize=10, color='purple')

    # Draw thumb for reference
    thumb_pos = np.array([-15, 30])
    thumb_tip = thumb_pos + 35 * np.array([np.cos(np.radians(30)), np.sin(np.radians(-30))])
    ax2.plot([thumb_pos[0], thumb_tip[0]], [thumb_pos[1], thumb_tip[1]],
             color='coral', linewidth=15, solid_capstyle='round', alpha=0.7)
    thumb_sensor = Circle(thumb_tip, 10, facecolor='gold', edgecolor='orange',
                          linewidth=2, zorder=10, alpha=0.8)
    ax2.add_patch(thumb_sensor)
    ax2.text(thumb_tip[0], thumb_tip[1], 'CoinFT', ha='center', va='center', fontsize=7)
    ax2.text(thumb_pos[0] - 10, thumb_pos[1], 'THUMB\n(fixed)', ha='center', fontsize=9, color='coral')

    # Annotations
    ax2.annotate(f'Fingertip position:\n({tip_pos[0]:.0f}, {tip_pos[1]:.0f}) mm',
                 xy=tip_pos, xytext=(70, 90), fontsize=9,
                 bbox=dict(boxstyle='round', facecolor='wheat'),
                 arrowprops=dict(arrowstyle='->', color='black'))

    # Cable travel annotation
    cable_travel = r1 * theta1_rad + r2 * theta2_rad
    ax2.text(5, -10, f'Cable pulled: {cable_travel:.1f} mm', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='lightyellow'))

    ax2.set_xlabel('X (mm)')
    ax2.set_ylabel('Y (mm)')
    ax2.grid(True, alpha=0.3)
    ax2.legend(['Cable/Tendon'], loc='upper right')

    plt.tight_layout()
    return fig


def draw_cable_routing_detail():
    """
    Draw detailed cross-section of cable routing at joint.
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.set_xlim(-40, 40)
    ax.set_ylim(-40, 60)
    ax.set_aspect('equal')
    ax.set_title('JOINT DETAIL - Cable Routing at MCP\n(cross-section view)', fontsize=14, fontweight='bold')

    # Draw the two links meeting at joint
    # Proximal (below)
    ax.fill([-15, 15, 15, -15], [-35, -35, -5, -5], color='steelblue', alpha=0.7, edgecolor='black', linewidth=2)
    ax.text(0, -20, 'Proximal\nLink', ha='center', va='center', fontsize=10, color='white', fontweight='bold')

    # Distal (above) - slightly rotated
    angle = 30
    theta = np.radians(angle)
    # Rotated rectangle vertices
    corners = np.array([[-15, 5], [15, 5], [15, 40], [-15, 40]])
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                                 [np.sin(theta), np.cos(theta)]])
    rotated = np.dot(corners, rotation_matrix.T)
    ax.fill(rotated[:, 0], rotated[:, 1], color='steelblue', alpha=0.7, edgecolor='black', linewidth=2)

    # Joint pin
    joint = Circle((0, 0), 5, facecolor='darkgray', edgecolor='black', linewidth=2, zorder=10)
    ax.add_patch(joint)
    ax.text(0, 0, '●', ha='center', va='center', fontsize=8)
    ax.annotate('Joint pin\n(M3 screw\nor steel rod)', xy=(0, 0), xytext=(25, 10),
                fontsize=9, arrowprops=dict(arrowstyle='->', color='black'))

    # Cable channel (on the flexion side)
    channel_x = -12
    # Through proximal
    ax.plot([channel_x, channel_x], [-35, -5], 'r-', linewidth=4, label='Cable')

    # Around joint (this is where moment arm matters)
    # Cable wraps around a pulley or channel at distance r from joint center
    r = 10  # moment arm
    arc_angles = np.linspace(-90, -90 + angle, 20)
    arc_x = -r * np.cos(np.radians(arc_angles))
    arc_y = -r * np.sin(np.radians(arc_angles))
    ax.plot(arc_x, arc_y, 'r-', linewidth=4)

    # Cable continues up through distal (in rotated frame)
    cable_end = np.array([-r, 0]) + 30 * np.array([np.sin(theta), np.cos(theta)])
    ax.plot([-r * np.cos(np.radians(-90 + angle)), cable_end[0]],
            [-r * np.sin(np.radians(-90 + angle)), cable_end[1]], 'r-', linewidth=4)

    # Moment arm annotation
    ax.annotate('', xy=(0, 0), xytext=(-r, 0),
                arrowprops=dict(arrowstyle='<->', color='green', lw=2))
    ax.text(-r/2, -8, 'r = 10mm\n(moment arm)', ha='center', fontsize=9, color='green')

    # Pulley/guide
    pulley = Circle((-r, 0), 3, facecolor='gray', edgecolor='black', linewidth=1, zorder=11)
    ax.add_patch(pulley)
    ax.annotate('Cable guide\nor pulley', xy=(-r, 0), xytext=(-30, 20),
                fontsize=9, arrowprops=dict(arrowstyle='->', color='gray'))

    # Show rotation direction
    ax.annotate('', xy=(15, 25), xytext=(20, 5),
                arrowprops=dict(arrowstyle='->', color='purple', lw=2,
                               connectionstyle='arc3,rad=0.3'))
    ax.text(25, 15, 'Flexion\ndirection', fontsize=9, color='purple')

    # Pull direction
    ax.annotate('', xy=(channel_x, -45), xytext=(channel_x, -35),
                arrowprops=dict(arrowstyle='->', color='red', lw=3))
    ax.text(channel_x, -50, 'PULL', ha='center', fontsize=10, color='red', fontweight='bold')

    # Explanation
    explanation = """
    HOW IT WORKS:
    1. Cable runs through channel in finger
    2. At joint, cable wraps around guide/pulley
    3. Distance from joint center to cable = moment arm (r)
    4. When pulled: Torque = Tension × r
    5. Larger r = more torque, but more cable travel
    """
    ax.text(0, -70, explanation, ha='center', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
            transform=ax.transData, verticalalignment='top')

    ax.set_ylim(-80, 60)
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')

    return fig


if __name__ == "__main__":
    # Generate detailed finger sketch
    fig1 = draw_detailed_finger()
    fig1.savefig('finger_detailed_sketch.png', dpi=150, bbox_inches='tight')
    print("Saved: finger_detailed_sketch.png")

    # Generate cable routing detail
    fig2 = draw_cable_routing_detail()
    fig2.savefig('cable_routing_detail.png', dpi=150, bbox_inches='tight')
    print("Saved: cable_routing_detail.png")

    plt.show()
