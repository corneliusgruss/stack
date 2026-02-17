"""
Visualization Module - Separate windows for different views.

Windows:
1. Hand View - Main gripper schematic with finger, thumb, cable, servo
2. Analysis View - Force, torque, and grip span plots
3. Control Panel - Sliders for parameters
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Circle, Polygon
from typing import Optional, List
from dataclasses import dataclass

from .geometry import FingerGeometry, FingertipPad, ThumbGeometry
from .actuation import CableActuation, SpringReturn, ActuationSystem
from .solver import EquilibriumSolver, EquilibriumResult
from .analysis import compute_analysis_curves, AnalysisCurves, compute_optimal_thumb_position


class HandView:
    """
    Main hand visualization - shows the physical gripper.
    """

    def __init__(self, fig, ax):
        self.fig = fig
        self.ax = ax

    def draw(self, finger: FingerGeometry, thumb: ThumbGeometry,
             fingertip: FingertipPad, cable: CableActuation,
             spring: SpringReturn, result: EquilibriumResult):
        """Draw complete hand visualization."""
        self.ax.clear()
        self.ax.set_aspect('equal')
        self.ax.set_xlabel('X (mm)', fontsize=10)
        self.ax.set_ylabel('Y (mm)', fontsize=10)
        self.ax.set_title(f'Gripper - Tension: {result.tension:.1f} N', fontsize=12, fontweight='bold')
        self.ax.grid(True, alpha=0.3, linestyle='--')

        # Draw in order: back elements first, front elements last
        self._draw_elastic(finger, spring, result)
        self._draw_cable_system(finger, cable, result)
        self._draw_palm(finger)
        self._draw_thumb(thumb, result)
        self._draw_finger(finger, fingertip, result)
        self._draw_contact(result)
        self._draw_info(result)

        # Set view limits
        self.ax.set_xlim(-50, 110)
        self.ax.set_ylim(-80, 100)

    def _draw_palm(self, finger: FingerGeometry):
        """Draw palm base structure."""
        # Palm as a rounded rectangle at the base
        palm = FancyBboxPatch(
            (-20, -30), 40, 30,
            boxstyle="round,pad=0.02,rounding_size=5",
            facecolor='#D4C4B5', edgecolor='#8B7355', linewidth=2,
            alpha=0.7, zorder=1
        )
        self.ax.add_patch(palm)
        self.ax.text(0, -15, 'PALM', ha='center', va='center',
                     fontsize=8, color='#5D4E37', fontweight='bold')

    def _draw_finger(self, finger: FingerGeometry, fingertip: FingertipPad,
                     result: EquilibriumResult):
        """Draw finger links with proper styling."""
        fk = finger.forward_kinematics(result.theta1, result.theta2)

        # Colors
        link_color = '#2E86AB'
        joint_color = '#1A5276'

        # Proximal link (thicker)
        self.ax.plot([fk['mcp'][0], fk['pip'][0]], [fk['mcp'][1], fk['pip'][1]],
                     color=link_color, linewidth=14, solid_capstyle='round', zorder=10)
        # Link outline
        self.ax.plot([fk['mcp'][0], fk['pip'][0]], [fk['mcp'][1], fk['pip'][1]],
                     color=joint_color, linewidth=16, solid_capstyle='round', zorder=9)

        # Distal link (slightly thinner)
        self.ax.plot([fk['pip'][0], fk['tip'][0]], [fk['pip'][1], fk['tip'][1]],
                     color=link_color, linewidth=11, solid_capstyle='round', zorder=10)
        self.ax.plot([fk['pip'][0], fk['tip'][0]], [fk['pip'][1], fk['tip'][1]],
                     color=joint_color, linewidth=13, solid_capstyle='round', zorder=9)

        # Joints as circles
        for name, pos in fk.items():
            if name == 'mcp':
                size, color = 18, joint_color
            elif name == 'pip':
                size, color = 15, joint_color
            else:
                size, color = 10, '#F39C12'

            self.ax.plot(pos[0], pos[1], 'o', markersize=size, color=color,
                         markeredgecolor='white', markeredgewidth=2, zorder=11)

        # Fingertip pad
        pad_pos = fingertip.position(finger, result.theta1, result.theta2)
        pad = Circle(pad_pos, fingertip.radius, facecolor='#F5B041',
                     edgecolor='#D68910', linewidth=2, alpha=0.8, zorder=12)
        self.ax.add_patch(pad)

        # Joint labels with angles
        self.ax.annotate(f'MCP\n{result.theta1_deg:.0f}°',
                         fk['mcp'] + np.array([-25, 5]), fontsize=9,
                         ha='center', fontweight='bold', color=joint_color)
        self.ax.annotate(f'PIP\n{result.theta2_deg:.0f}°',
                         fk['pip'] + np.array([15, 5]), fontsize=9,
                         ha='center', fontweight='bold', color=joint_color)

    def _draw_thumb(self, thumb: ThumbGeometry, result: EquilibriumResult):
        """Draw thumb with proper anatomical orientation."""
        # Thumb color
        thumb_color = '#E74C3C'
        thumb_dark = '#922B21'

        # Draw thumb link from base to tip
        self.ax.plot([thumb.base_position[0], thumb.tip_position[0]],
                     [thumb.base_position[1], thumb.tip_position[1]],
                     color=thumb_color, linewidth=12, solid_capstyle='round', zorder=8)
        self.ax.plot([thumb.base_position[0], thumb.tip_position[0]],
                     [thumb.base_position[1], thumb.tip_position[1]],
                     color=thumb_dark, linewidth=14, solid_capstyle='round', zorder=7)

        # Thumb base (CMC joint equivalent)
        self.ax.plot(thumb.base_position[0], thumb.base_position[1], 'o',
                     markersize=14, color=thumb_dark,
                     markeredgecolor='white', markeredgewidth=2, zorder=9)

        # Thumb tip pad
        thumb_pad = Circle(thumb.tip_position, thumb.pad_radius,
                           facecolor='#F5B041', edgecolor='#D68910',
                           linewidth=2, alpha=0.8, zorder=12)
        self.ax.add_patch(thumb_pad)

        # Label
        self.ax.annotate('THUMB', thumb.base_position + np.array([15, -10]),
                         fontsize=9, color=thumb_dark, fontweight='bold')

    def _draw_cable_system(self, finger: FingerGeometry, cable: CableActuation,
                           result: EquilibriumResult):
        """Draw cable, servo horn, and pulleys."""
        cp = cable.cable_path(finger, result.theta1, result.theta2)

        # Servo motor body
        servo_body = FancyBboxPatch(
            (cp['servo_center'][0] - 15, cp['servo_center'][1] - 20),
            30, 25,
            boxstyle="round,pad=0.02,rounding_size=3",
            facecolor='#2C3E50', edgecolor='#1A252F', linewidth=2,
            zorder=2
        )
        self.ax.add_patch(servo_body)
        self.ax.text(cp['servo_center'][0], cp['servo_center'][1] - 8,
                     'SERVO', ha='center', va='center', fontsize=7,
                     color='white', fontweight='bold')

        # Servo horn (rotating disk)
        horn = Circle(cp['servo_center'], cp['servo_radius'],
                      facecolor='#7F8C8D', edgecolor='#5D6D7E',
                      linewidth=2, zorder=3)
        self.ax.add_patch(horn)

        # Horn center axis
        self.ax.plot(*cp['servo_center'], '+', color='white',
                     markersize=10, markeredgewidth=2, zorder=4)

        # Cable attachment on horn
        self.ax.plot(*cp['servo_attach'], 'o', color='#E74C3C',
                     markersize=8, markeredgecolor='white',
                     markeredgewidth=1, zorder=4)

        # MCP pulley
        mcp_pulley = Circle(cp['mcp_pulley_center'], cable.r1,
                            facecolor='#E8DAEF', edgecolor='#8E44AD',
                            linewidth=1.5, alpha=0.6, zorder=5)
        self.ax.add_patch(mcp_pulley)

        # PIP pulley
        pip_pulley = Circle(cp['pip_pulley_center'], cable.r2,
                            facecolor='#E8DAEF', edgecolor='#8E44AD',
                            linewidth=1.5, alpha=0.6, zorder=5)
        self.ax.add_patch(pip_pulley)

        # Cable path
        path = cp['path']
        xs = [p[0] for p in path]
        ys = [p[1] for p in path]
        self.ax.plot(xs, ys, color='#C0392B', linewidth=2.5,
                     linestyle='-', zorder=6, label='Cable')

        # Cable guides
        self.ax.plot(*cp['mcp_guide'], 's', color='#8E44AD', markersize=5, zorder=6)
        self.ax.plot(*cp['pip_entry'], 's', color='#8E44AD', markersize=4, zorder=6)

    def _draw_elastic(self, finger: FingerGeometry, spring: SpringReturn,
                      result: EquilibriumResult):
        """Draw elastic return on back of finger."""
        ep = spring.elastic_path(finger, result.theta1, result.theta2)

        # Color intensity based on stretch
        green_intensity = 0.6 - 0.3 * ep['stretch']
        color = (0.2, green_intensity, 0.2)

        path = ep['path']
        xs = [p[0] for p in path]
        ys = [p[1] for p in path]

        # Draw as wavy line to indicate elastic
        self.ax.plot(xs, ys, color=color, linewidth=3, linestyle='--',
                     alpha=0.7, zorder=3, label='Elastic')

        # Anchor points
        self.ax.plot(*ep['palm_anchor'], 'o', color='#27AE60',
                     markersize=6, markeredgecolor='white', markeredgewidth=1, zorder=4)
        self.ax.plot(*ep['tip_anchor'], 'o', color='#27AE60',
                     markersize=6, markeredgecolor='white', markeredgewidth=1, zorder=4)

    def _draw_contact(self, result: EquilibriumResult):
        """Draw contact point and force vector."""
        if not result.contact.in_contact:
            return

        cp = result.contact.contact_point

        # Contact star
        self.ax.plot(cp[0], cp[1], '*', markersize=25, color='#F39C12',
                     markeredgecolor='white', markeredgewidth=1.5, zorder=20)

        # Force vector
        if result.contact_force > 0.1:
            scale = 3.0  # mm per Newton
            n = result.contact.normal
            self.ax.arrow(cp[0], cp[1],
                          n[0] * result.contact_force * scale,
                          n[1] * result.contact_force * scale,
                          head_width=4, head_length=3,
                          fc='#E74C3C', ec='#E74C3C', linewidth=2, zorder=21)
            self.ax.text(cp[0] + n[0] * result.contact_force * scale + 5,
                         cp[1] + n[1] * result.contact_force * scale,
                         f'{result.contact_force:.1f}N', fontsize=10,
                         color='#E74C3C', fontweight='bold')

    def _draw_info(self, result: EquilibriumResult):
        """Draw info box."""
        info_text = (
            f"Grip span: {result.grip_span:.1f} mm\n"
            f"Cable travel: {result.cable_torques[0]/10:.1f} mm"  # Approximate
        )
        if result.contact.in_contact:
            info_text += f"\nContact force: {result.contact_force:.2f} N"

        self.ax.text(0.02, 0.98, info_text, transform=self.ax.transAxes,
                     fontsize=9, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))


class AnalysisView:
    """
    Analysis plots - force, torque, grip span vs tension.
    """

    def __init__(self, fig, axes):
        self.fig = fig
        self.axes = axes  # Dict with 'force', 'torque', 'trajectory'

    def draw(self, curves: AnalysisCurves, current_tension: float,
             thumb: ThumbGeometry, fingertip: FingertipPad):
        """Draw all analysis plots."""
        self._draw_force_span(curves, current_tension)
        self._draw_torques(curves, current_tension)
        self._draw_trajectory(curves, current_tension, thumb, fingertip)

    def _draw_force_span(self, curves: AnalysisCurves, tension: float):
        """Force and grip span vs tension."""
        ax = self.axes['force']
        ax.clear()

        # Grip span
        color1 = '#2E86AB'
        ax.plot(curves.tensions, curves.grip_span, color=color1, linewidth=2)
        ax.set_xlabel('Tension (N)')
        ax.set_ylabel('Grip Span (mm)', color=color1)
        ax.tick_params(axis='y', labelcolor=color1)

        # Contact force on secondary axis
        ax2 = ax.twinx()
        color2 = '#E74C3C'
        ax2.plot(curves.tensions, curves.contact_force, color=color2, linewidth=2)
        ax2.set_ylabel('Contact Force (N)', color=color2)
        ax2.tick_params(axis='y', labelcolor=color2)
        ax2.set_ylim(bottom=0)

        # Current tension line
        ax.axvline(x=tension, color='gray', linestyle='--', alpha=0.7)

        # Contact threshold line
        contact_idx = np.where(curves.in_contact)[0]
        if len(contact_idx) > 0:
            t_contact = curves.tensions[contact_idx[0]]
            ax.axvline(x=t_contact, color='#27AE60', linestyle=':', linewidth=2)
            ax.text(t_contact + 0.5, ax.get_ylim()[1] * 0.9, 'Contact',
                    fontsize=9, color='#27AE60')

        ax.set_title('Grip Span & Contact Force', fontweight='bold')
        ax.grid(True, alpha=0.3)

    def _draw_torques(self, curves: AnalysisCurves, tension: float):
        """Torque breakdown."""
        ax = self.axes['torque']
        ax.clear()

        # Cable torques
        ax.plot(curves.tensions, curves.cable_torque_mcp, 'r-', linewidth=2,
                label='Cable τ₁ (MCP)')
        ax.plot(curves.tensions, curves.cable_torque_pip, 'r--', linewidth=2,
                label='Cable τ₂ (PIP)')

        # Spring torques (show absolute value for comparison)
        ax.plot(curves.tensions, -curves.spring_torque_mcp, 'b-', linewidth=2,
                label='Spring |τ₁|')
        ax.plot(curves.tensions, -curves.spring_torque_pip, 'b--', linewidth=2,
                label='Spring |τ₂|')

        # Net torques
        ax.plot(curves.tensions, curves.net_torque_mcp, 'g-', linewidth=1.5,
                alpha=0.7, label='Net τ₁')
        ax.plot(curves.tensions, curves.net_torque_pip, 'g--', linewidth=1.5,
                alpha=0.7, label='Net τ₂')

        ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
        ax.axvline(x=tension, color='gray', linestyle='--', alpha=0.7)

        ax.set_xlabel('Tension (N)')
        ax.set_ylabel('Torque (N·mm)')
        ax.set_title('Joint Torques', fontweight='bold')
        ax.legend(loc='upper left', fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)

    def _draw_trajectory(self, curves: AnalysisCurves, tension: float,
                         thumb: ThumbGeometry, fingertip: FingertipPad):
        """Fingertip trajectory."""
        ax = self.axes['trajectory']
        ax.clear()
        ax.set_aspect('equal')

        # Draw trajectory colored by tension
        for i in range(len(curves.tensions) - 1):
            t = curves.tensions[i]
            color = plt.cm.viridis(t / 25.0)
            ax.plot([curves.pad_x[i], curves.pad_x[i+1]],
                    [curves.pad_y[i], curves.pad_y[i+1]],
                    color=color, linewidth=3)

        # Current position
        idx = np.argmin(np.abs(curves.tensions - tension))
        ax.plot(curves.pad_x[idx], curves.pad_y[idx], 'o',
                markersize=15, color='#2E86AB',
                markeredgecolor='white', markeredgewidth=2)

        # Thumb
        thumb_circle = Circle(thumb.tip_position, thumb.pad_radius,
                              facecolor='#F5B041', edgecolor='#E74C3C',
                              linewidth=2, alpha=0.6)
        ax.add_patch(thumb_circle)
        ax.plot(*thumb.tip_position, 'x', markersize=10, color='#E74C3C',
                markeredgewidth=2)

        # Line from current to thumb
        ax.plot([curves.pad_x[idx], thumb.tip_position[0]],
                [curves.pad_y[idx], thumb.tip_position[1]],
                'k--', alpha=0.5, linewidth=1)

        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_title('Fingertip Trajectory', fontweight='bold')
        ax.grid(True, alpha=0.3)

        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(0, 25))
        sm.set_array([])
        self.fig.colorbar(sm, ax=ax, label='Tension (N)', shrink=0.8)


class GripperSimulator:
    """
    Main simulator with multiple windows.
    """

    def __init__(self, finger: FingerGeometry, cable: CableActuation,
                 spring: SpringReturn, fingertip: FingertipPad, thumb: ThumbGeometry):
        self.finger = finger
        self.cable = cable
        self.spring = spring
        self.fingertip = fingertip
        self.thumb = thumb

        self.actuation = ActuationSystem(cable, spring)
        self.solver = EquilibriumSolver(finger, self.actuation, fingertip, thumb)

        # Windows
        self.hand_fig = None
        self.analysis_fig = None
        self.control_fig = None

        self.sliders = {}
        self.result = None
        self.curves = None

    def setup(self):
        """Create all windows."""
        # Hand view window
        self.hand_fig, self.hand_ax = plt.subplots(figsize=(10, 10))
        self.hand_fig.canvas.manager.set_window_title('Gripper - Hand View')
        self.hand_view = HandView(self.hand_fig, self.hand_ax)

        # Analysis window
        self.analysis_fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        self.analysis_fig.canvas.manager.set_window_title('Gripper - Analysis')
        self.analysis_view = AnalysisView(self.analysis_fig, {
            'force': axes[0, 0],
            'torque': axes[0, 1],
            'trajectory': axes[1, 0],
        })
        axes[1, 1].axis('off')  # Info panel
        self.info_ax = axes[1, 1]

        # Control panel window
        self.control_fig = plt.figure(figsize=(8, 6))
        self.control_fig.canvas.manager.set_window_title('Gripper - Controls')
        self._setup_controls()

        # Adjust layouts
        self.hand_fig.tight_layout()
        self.analysis_fig.tight_layout()

    def _setup_controls(self):
        """Create control sliders."""
        self.control_fig.text(0.5, 0.95, 'Gripper Parameters',
                              ha='center', fontsize=14, fontweight='bold')

        slider_configs = [
            ('tension', 'Tension (N)', 0, 25, 12),
            ('thumb_x', 'Thumb X (mm)', 20, 80, self.thumb.tip_position[0]),
            ('thumb_y', 'Thumb Y (mm)', -60, 20, self.thumb.tip_position[1]),
            ('L1', 'L1 Proximal (mm)', 30, 60, self.finger.L1),
            ('L2', 'L2 Distal (mm)', 30, 60, self.finger.L2),
            ('r1', 'r1 MCP Pulley (mm)', 5, 15, self.cable.r1),
            ('r2', 'r2 PIP Pulley (mm)', 3, 12, self.cable.r2),
            ('k1', 'k1 MCP Spring', 50, 200, self.spring.k1),
            ('k2', 'k2 PIP Spring', 40, 150, self.spring.k2),
        ]

        for i, (name, label, vmin, vmax, vinit) in enumerate(slider_configs):
            ax = self.control_fig.add_axes([0.25, 0.85 - i * 0.08, 0.5, 0.04])
            self.sliders[name] = Slider(ax, label, vmin, vmax, valinit=vinit)
            self.sliders[name].on_changed(self.update)

        # Buttons
        btn_optimal = self.control_fig.add_axes([0.25, 0.05, 0.2, 0.05])
        self.btn_optimal = Button(btn_optimal, 'Optimal Thumb')
        self.btn_optimal.on_clicked(self.set_optimal_thumb)

        btn_reset = self.control_fig.add_axes([0.55, 0.05, 0.2, 0.05])
        self.btn_reset = Button(btn_reset, 'Reset')
        self.btn_reset.on_clicked(self.reset)

    def update(self, val=None):
        """Update simulation and all views."""
        # Update parameters from sliders
        self.finger.L1 = self.sliders['L1'].val
        self.finger.L2 = self.sliders['L2'].val
        self.cable.r1 = self.sliders['r1'].val
        self.cable.r2 = self.sliders['r2'].val
        self.spring.k1 = self.sliders['k1'].val
        self.spring.k2 = self.sliders['k2'].val

        self.thumb.tip_position = np.array([
            self.sliders['thumb_x'].val,
            self.sliders['thumb_y'].val
        ])
        self.thumb.base_position = self.thumb.tip_position + np.array([30, -25])

        # Rebuild solver
        self.actuation = ActuationSystem(self.cable, self.spring)
        self.solver = EquilibriumSolver(
            self.finger, self.actuation, self.fingertip, self.thumb
        )

        # Solve current state
        tension = self.sliders['tension'].val
        self.result = self.solver.solve(tension)
        self.curves = compute_analysis_curves(self.solver, (0, 25), 100)

        # Update all views
        self.hand_view.draw(
            self.finger, self.thumb, self.fingertip,
            self.cable, self.spring, self.result
        )
        self.analysis_view.draw(self.curves, tension, self.thumb, self.fingertip)
        self._update_info()

        # Redraw
        self.hand_fig.canvas.draw_idle()
        self.analysis_fig.canvas.draw_idle()

    def _update_info(self):
        """Update info panel."""
        self.info_ax.clear()
        self.info_ax.axis('off')

        contact_t = self.solver.find_contact_tension()

        info = [
            "═══ CURRENT STATE ═══",
            f"Tension: {self.result.tension:.1f} N",
            f"θ₁ (MCP): {self.result.theta1_deg:.1f}°",
            f"θ₂ (PIP): {self.result.theta2_deg:.1f}°",
            f"Grip span: {self.result.grip_span:.1f} mm",
            "",
            "═══ CONTACT ═══",
            f"Contact at: {contact_t:.1f} N" if contact_t else "No contact",
            f"Force: {self.result.contact_force:.2f} N",
            "",
            "═══ TORQUES ═══",
            f"Cable: [{self.result.cable_torques[0]:.0f}, {self.result.cable_torques[1]:.0f}]",
            f"Spring: [{self.result.spring_torques[0]:.0f}, {self.result.spring_torques[1]:.0f}]",
            f"Net: [{self.result.net_torques[0]:.0f}, {self.result.net_torques[1]:.0f}]",
        ]

        self.info_ax.text(0.1, 0.95, '\n'.join(info),
                          transform=self.info_ax.transAxes,
                          fontfamily='monospace', fontsize=10,
                          verticalalignment='top',
                          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    def set_optimal_thumb(self, event=None):
        """Position thumb optimally."""
        tension = self.sliders['tension'].val
        opt = compute_optimal_thumb_position(
            self.finger, self.actuation, self.fingertip, tension
        )
        self.sliders['thumb_x'].set_val(opt[0])
        self.sliders['thumb_y'].set_val(opt[1])

    def reset(self, event=None):
        """Reset to defaults."""
        for slider in self.sliders.values():
            slider.reset()

    def run(self):
        """Start simulation."""
        self.setup()
        self.update()
        plt.show()


def run_interactive(finger: FingerGeometry, cable: CableActuation,
                    spring: SpringReturn, fingertip: FingertipPad,
                    thumb: ThumbGeometry):
    """Convenience function to run simulation."""
    sim = GripperSimulator(finger, cable, spring, fingertip, thumb)
    sim.run()
