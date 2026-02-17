"""
Design Validation - Physical Analysis
Test the gripper design against real objects and tasks

Author: Cornelius Gruss
Date: January 26, 2026
"""

import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# GRIPPER DESIGN PARAMETERS
# =============================================================================

# Link lengths (mm)
L1 = 45.0  # Proximal phalanx
L2 = 45.0  # Distal phalanx
L_total = L1 + L2  # Total finger length

# Moment arms (mm)
r1 = 10.0  # MCP moment arm
r2 = 8.0   # PIP moment arm

# Thumb position relative to MCP (mm)
thumb_offset_x = -50  # thumb is 50mm to the left (toward thumb side)
thumb_offset_y = 10   # thumb is 10mm forward

# Grip span range (mm) - distance between thumb and index fingertip
grip_min = 20   # fully closed
grip_max = 100  # fully open

# Friction coefficient (rubber/silicone on various surfaces)
mu_rubber_plastic = 0.6
mu_rubber_glass = 0.5
mu_rubber_metal = 0.4
mu_rubber_ceramic = 0.5

# =============================================================================
# OBJECTS TO TEST
# =============================================================================

objects = {
    'french_press_plunger': {
        'name': 'French Press Plunger Handle',
        'diameter': 25,  # mm
        'weight': 50,    # grams (just the plunger part)
        'material': 'plastic',
        'task': 'push_down',
        'push_force_required': 15,  # N (force to push plunger through coffee)
        'notes': 'Need to push down while gripping'
    },
    'french_press_body': {
        'name': 'French Press Body',
        'diameter': 90,  # mm
        'weight': 400,   # grams (glass + frame, empty)
        'weight_full': 800,  # grams (with water)
        'material': 'glass',
        'task': 'hold_steady',
        'notes': 'Hold while other hand plunges'
    },
    'coffee_mug': {
        'name': 'Coffee Mug',
        'diameter': 80,  # mm (body)
        'handle_width': 25,  # mm
        'weight': 300,   # grams empty
        'weight_full': 550,  # grams with coffee
        'material': 'ceramic',
        'task': 'hold_and_pour',
        'pour_angle': 45,  # degrees
        'notes': 'Pick up, hold, pour'
    },
    'small_pitcher': {
        'name': 'Small Water Pitcher',
        'diameter': 70,  # mm
        'handle_width': 20,  # mm
        'weight': 200,   # grams empty
        'weight_full': 700,  # grams with water
        'material': 'plastic',
        'task': 'pour',
        'notes': 'Pour water into French press'
    },
    'espresso_cup': {
        'name': 'Espresso Cup',
        'diameter': 55,  # mm
        'weight': 100,   # grams
        'weight_full': 150,  # grams
        'material': 'ceramic',
        'task': 'hold',
        'notes': 'Small cup, precision grip'
    }
}

# =============================================================================
# PHYSICS CALCULATIONS
# =============================================================================

def get_friction_coef(material):
    """Get friction coefficient for rubber gripper on material."""
    friction_map = {
        'plastic': mu_rubber_plastic,
        'glass': mu_rubber_glass,
        'metal': mu_rubber_metal,
        'ceramic': mu_rubber_ceramic
    }
    return friction_map.get(material, 0.5)


def grip_force_to_hold(weight_g, mu, safety_factor=1.5):
    """
    Calculate normal grip force needed to hold an object without slipping.

    Physics: Friction force = μ × Normal force
             Friction force must > Weight

    For pinch grip (two opposing surfaces):
        2 × μ × F_grip > Weight
        F_grip > Weight / (2 × μ)

    Args:
        weight_g: Object weight in grams
        mu: Friction coefficient
        safety_factor: Multiply required force by this

    Returns:
        Required grip force in Newtons
    """
    weight_n = weight_g / 1000 * 9.81  # Convert to Newtons
    f_grip = (weight_n / (2 * mu)) * safety_factor
    return f_grip


def grip_force_to_pour(weight_g, mu, pour_angle_deg, grip_distance_from_cg):
    """
    Calculate grip force needed while pouring (tilted object).

    When tilted, the center of gravity creates a moment that tries to
    rotate the object out of grip. Need extra grip force.

    Args:
        weight_g: Object weight in grams
        mu: Friction coefficient
        pour_angle_deg: Tilt angle in degrees
        grip_distance_from_cg: Distance from grip point to center of gravity (mm)

    Returns:
        Required grip force in Newtons
    """
    weight_n = weight_g / 1000 * 9.81
    angle_rad = np.radians(pour_angle_deg)

    # Moment from tilted CG
    moment = weight_n * grip_distance_from_cg/1000 * np.sin(angle_rad)

    # Additional grip force to counter moment (rough approximation)
    # Assuming grip width of 50mm
    grip_width = 0.05  # m
    additional_force = moment / grip_width

    # Base grip force to hold
    base_grip = grip_force_to_hold(weight_g, mu, safety_factor=1.0)

    return base_grip + additional_force


def grip_force_for_pushing(push_force, mu):
    """
    Calculate grip force needed when pushing down on something.

    When pushing, friction must resist the push force.

    Args:
        push_force: Force being applied (N)
        mu: Friction coefficient

    Returns:
        Required grip force in Newtons
    """
    # Friction must exceed push force to not slip
    f_grip = push_force / mu * 1.5  # safety factor
    return f_grip


def cable_tension_for_grip(grip_force, finger_config='pinch'):
    """
    Calculate cable tension needed to generate a grip force.

    Simplified model: grip force at fingertip requires torque at joints.

    For pinch grip at fingertip:
        Torque_MCP = grip_force × L_total (moment arm from MCP to fingertip)
        Cable_tension = Torque_MCP / r1

    Actually both joints contribute, but MCP sees the most load.

    Args:
        grip_force: Desired grip force at fingertip (N)

    Returns:
        Required cable tension (N)
    """
    # Torque at MCP (worst case - fingertip contact)
    torque_mcp = grip_force * (L_total / 1000)  # N·m

    # Cable tension
    cable_tension = torque_mcp / (r1 / 1000)  # N

    return cable_tension


def motor_torque_required(cable_tension, pulley_radius_mm=10):
    """
    Calculate motor torque to generate cable tension.

    Args:
        cable_tension: Required cable tension (N)
        pulley_radius_mm: Pulley radius on motor shaft (mm)

    Returns:
        Motor torque in N·mm and kg·cm
    """
    torque_nmm = cable_tension * pulley_radius_mm
    torque_kgcm = torque_nmm / 98.1  # convert N·mm to kg·cm

    return torque_nmm, torque_kgcm


# =============================================================================
# ANALYSIS
# =============================================================================

def analyze_object(obj_key, obj_data):
    """Analyze gripper requirements for one object."""

    print(f"\n{'='*60}")
    print(f"OBJECT: {obj_data['name']}")
    print(f"{'='*60}")

    print(f"\nProperties:")
    print(f"  Diameter: {obj_data['diameter']} mm")
    print(f"  Weight: {obj_data['weight']} g", end='')
    if 'weight_full' in obj_data:
        print(f" (full: {obj_data['weight_full']} g)")
    else:
        print()
    print(f"  Material: {obj_data['material']}")
    print(f"  Task: {obj_data['task']}")

    mu = get_friction_coef(obj_data['material'])
    print(f"  Friction coef (rubber on {obj_data['material']}): {mu}")

    # Check if object fits in gripper
    if obj_data['diameter'] > grip_max:
        print(f"\n  ⚠️  WARNING: Object diameter ({obj_data['diameter']}mm) > max grip ({grip_max}mm)")
        print(f"      May need to grip by handle or different approach")
    elif obj_data['diameter'] < grip_min:
        print(f"\n  ⚠️  WARNING: Object diameter ({obj_data['diameter']}mm) < min grip ({grip_min}mm)")
    else:
        print(f"\n  ✓ Object fits in grip range ({grip_min}-{grip_max}mm)")

    # Calculate grip force based on task
    weight = obj_data.get('weight_full', obj_data['weight'])

    if obj_data['task'] == 'push_down':
        push_force = obj_data.get('push_force_required', 10)
        grip_force = grip_force_for_pushing(push_force, mu)
        print(f"\n  Task: Push down with {push_force}N force")
        print(f"  Required grip force: {grip_force:.1f} N")

    elif obj_data['task'] == 'hold_and_pour' or obj_data['task'] == 'pour':
        pour_angle = obj_data.get('pour_angle', 45)
        grip_distance = obj_data['diameter'] / 2  # rough estimate
        grip_force = grip_force_to_pour(weight, mu, pour_angle, grip_distance)
        print(f"\n  Task: Hold and pour at {pour_angle}° angle")
        print(f"  Weight when full: {weight}g")
        print(f"  Required grip force: {grip_force:.1f} N")

    else:  # hold_steady or hold
        grip_force = grip_force_to_hold(weight, mu)
        print(f"\n  Task: Hold steady")
        print(f"  Weight: {weight}g")
        print(f"  Required grip force: {grip_force:.1f} N")

    # Calculate cable tension and motor requirements
    cable_tension = cable_tension_for_grip(grip_force)
    motor_torque_nmm, motor_torque_kgcm = motor_torque_required(cable_tension)

    print(f"\n  Cable tension needed: {cable_tension:.1f} N")
    print(f"  Motor torque (10mm pulley): {motor_torque_nmm:.0f} N·mm = {motor_torque_kgcm:.1f} kg·cm")

    # Check against common servos
    print(f"\n  Servo options:")
    servos = [
        ('SG90', 1.8),
        ('MG90S', 2.2),
        ('MG996R', 10.0),
        ('STS3215 (SO-101)', 16.5)
    ]
    for servo_name, servo_torque in servos:
        if servo_torque >= motor_torque_kgcm:
            print(f"    ✓ {servo_name} ({servo_torque} kg·cm) - OK")
        else:
            print(f"    ✗ {servo_name} ({servo_torque} kg·cm) - insufficient")

    return {
        'name': obj_data['name'],
        'grip_force': grip_force,
        'cable_tension': cable_tension,
        'motor_torque_kgcm': motor_torque_kgcm,
        'fits_gripper': grip_min <= obj_data['diameter'] <= grip_max
    }


def run_full_analysis():
    """Run analysis on all objects."""

    print("\n" + "="*70)
    print("GRIPPER DESIGN VALIDATION")
    print("="*70)

    print(f"\nGripper Parameters:")
    print(f"  Finger length: {L1} + {L2} = {L_total} mm")
    print(f"  Moment arms: r1={r1}mm (MCP), r2={r2}mm (PIP)")
    print(f"  Grip range: {grip_min} - {grip_max} mm")

    results = []
    for obj_key, obj_data in objects.items():
        result = analyze_object(obj_key, obj_data)
        results.append(result)

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    print(f"\n{'Object':<30} {'Grip(N)':<10} {'Cable(N)':<10} {'Motor(kg·cm)':<12}")
    print("-"*62)
    for r in results:
        print(f"{r['name']:<30} {r['grip_force']:<10.1f} {r['cable_tension']:<10.1f} {r['motor_torque_kgcm']:<12.1f}")

    max_grip = max(r['grip_force'] for r in results)
    max_cable = max(r['cable_tension'] for r in results)
    max_motor = max(r['motor_torque_kgcm'] for r in results)

    print("-"*62)
    print(f"{'MAXIMUM REQUIRED':<30} {max_grip:<10.1f} {max_cable:<10.1f} {max_motor:<12.1f}")

    print(f"\n\nRECOMMENDATION:")
    print(f"  Minimum motor: {max_motor:.1f} kg·cm")
    if max_motor <= 2.2:
        print(f"  → MG90S (2.2 kg·cm) sufficient")
    elif max_motor <= 10:
        print(f"  → MG996R (10 kg·cm) recommended")
    else:
        print(f"  → STS3215 (16.5 kg·cm) or similar recommended")

    print(f"\n  For training hand (human trigger):")
    print(f"  → Maximum cable pull: {max_cable:.1f} N")
    print(f"  → This is {'easily achievable' if max_cable < 50 else 'moderate effort'} with finger pull")

    return results


def plot_analysis(results):
    """Create visualization of analysis results."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    names = [r['name'] for r in results]
    grip_forces = [r['grip_force'] for r in results]
    cable_tensions = [r['cable_tension'] for r in results]
    motor_torques = [r['motor_torque_kgcm'] for r in results]

    # Short names for plotting
    short_names = ['Plunger', 'Press Body', 'Mug', 'Pitcher', 'Espresso']

    # 1. Grip force comparison
    ax1 = axes[0, 0]
    bars1 = ax1.bar(short_names, grip_forces, color='steelblue')
    ax1.set_ylabel('Grip Force (N)')
    ax1.set_title('Required Grip Force by Object')
    ax1.axhline(y=max(grip_forces), color='r', linestyle='--', label=f'Max: {max(grip_forces):.1f}N')
    ax1.legend()
    for bar, val in zip(bars1, grip_forces):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.1f}', ha='center', va='bottom', fontsize=9)

    # 2. Cable tension comparison
    ax2 = axes[0, 1]
    bars2 = ax2.bar(short_names, cable_tensions, color='coral')
    ax2.set_ylabel('Cable Tension (N)')
    ax2.set_title('Required Cable Tension by Object')
    ax2.axhline(y=30, color='g', linestyle='--', label='Easy finger pull (30N)')
    ax2.axhline(y=50, color='orange', linestyle='--', label='Moderate pull (50N)')
    ax2.legend()
    for bar, val in zip(bars2, cable_tensions):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.1f}', ha='center', va='bottom', fontsize=9)

    # 3. Motor torque with servo lines
    ax3 = axes[1, 0]
    bars3 = ax3.bar(short_names, motor_torques, color='green')
    ax3.set_ylabel('Motor Torque (kg·cm)')
    ax3.set_title('Required Motor Torque by Object')
    ax3.axhline(y=1.8, color='gray', linestyle=':', label='SG90 (1.8)')
    ax3.axhline(y=2.2, color='blue', linestyle='--', label='MG90S (2.2)')
    ax3.axhline(y=10, color='orange', linestyle='--', label='MG996R (10)')
    ax3.axhline(y=16.5, color='red', linestyle='--', label='STS3215 (16.5)')
    ax3.legend(loc='upper right')
    ax3.set_ylim(0, 20)
    for bar, val in zip(bars3, motor_torques):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{val:.1f}', ha='center', va='bottom', fontsize=9)

    # 4. Object size vs grip range
    ax4 = axes[1, 1]
    diameters = [objects[k]['diameter'] for k in objects.keys()]
    colors = ['green' if grip_min <= d <= grip_max else 'red' for d in diameters]
    bars4 = ax4.bar(short_names, diameters, color=colors)
    ax4.axhline(y=grip_min, color='blue', linestyle='--', label=f'Min grip ({grip_min}mm)')
    ax4.axhline(y=grip_max, color='blue', linestyle='--', label=f'Max grip ({grip_max}mm)')
    ax4.axhspan(grip_min, grip_max, alpha=0.2, color='blue', label='Grip range')
    ax4.set_ylabel('Object Diameter (mm)')
    ax4.set_title('Object Size vs Gripper Range')
    ax4.legend()
    for bar, val in zip(bars4, diameters):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    return fig


def analyze_french_press_sequence():
    """
    Detailed analysis of the complete French press task sequence.
    """
    print("\n" + "="*70)
    print("FRENCH PRESS TASK SEQUENCE ANALYSIS")
    print("="*70)

    sequence = [
        {
            'step': 1,
            'action': 'Pick up pitcher with water',
            'hand': 'Right (index)',
            'object': 'small_pitcher',
            'weight': 700,  # full
            'grip_type': 'handle',
            'notes': 'Grip handle, lift'
        },
        {
            'step': 2,
            'action': 'Hold French press steady',
            'hand': 'Left (would need second gripper)',
            'object': 'french_press_body',
            'weight': 400,  # empty
            'grip_type': 'wrap',
            'notes': 'Stabilize while pouring'
        },
        {
            'step': 3,
            'action': 'Pour water into French press',
            'hand': 'Right',
            'object': 'small_pitcher',
            'weight': 700,
            'grip_type': 'handle',
            'pour_angle': 60,
            'notes': 'Controlled pour'
        },
        {
            'step': 4,
            'action': 'Set down pitcher',
            'hand': 'Right',
            'object': 'small_pitcher',
            'weight': 200,  # now empty
            'grip_type': 'handle',
            'notes': 'Place gently'
        },
        {
            'step': 5,
            'action': 'Grip plunger handle',
            'hand': 'Right',
            'object': 'french_press_plunger',
            'weight': 50,
            'grip_type': 'pinch',
            'notes': 'Prepare to plunge'
        },
        {
            'step': 6,
            'action': 'Push plunger down',
            'hand': 'Right',
            'object': 'french_press_plunger',
            'weight': 50,
            'grip_type': 'pinch',
            'push_force': 15,  # Newtons
            'notes': 'Slow, controlled push'
        },
        {
            'step': 7,
            'action': 'Pick up French press',
            'hand': 'Right',
            'object': 'french_press_body',
            'weight': 800,  # full with coffee
            'grip_type': 'handle',
            'notes': 'Grip handle'
        },
        {
            'step': 8,
            'action': 'Pour into mug',
            'hand': 'Right',
            'object': 'french_press_body',
            'weight': 800,
            'grip_type': 'handle',
            'pour_angle': 45,
            'notes': 'Controlled pour'
        }
    ]

    print(f"\n{'Step':<5} {'Action':<35} {'Grip Force':<12} {'Cable Tension':<15}")
    print("-"*70)

    max_grip = 0
    max_cable = 0

    for s in sequence:
        obj = objects.get(s['object'], {})
        mu = get_friction_coef(obj.get('material', 'plastic'))
        weight = s.get('weight', obj.get('weight', 100))

        if 'push_force' in s:
            grip_force = grip_force_for_pushing(s['push_force'], mu)
        elif 'pour_angle' in s:
            grip_force = grip_force_to_pour(weight, mu, s['pour_angle'], 50)
        else:
            grip_force = grip_force_to_hold(weight, mu)

        cable_tension = cable_tension_for_grip(grip_force)

        max_grip = max(max_grip, grip_force)
        max_cable = max(max_cable, cable_tension)

        print(f"{s['step']:<5} {s['action']:<35} {grip_force:<12.1f} {cable_tension:<15.1f}")

    print("-"*70)
    print(f"{'MAX':<5} {'':<35} {max_grip:<12.1f} {max_cable:<15.1f}")

    print(f"\n\nKEY FINDINGS:")
    print(f"  • Hardest step: Pouring full French press (weight + angle)")
    print(f"  • Maximum grip force needed: {max_grip:.1f} N")
    print(f"  • Maximum cable tension: {max_cable:.1f} N")

    _, motor_torque = motor_torque_required(max_cable)
    print(f"  • Motor torque required: {motor_torque:.1f} kg·cm")
    print(f"  • Recommended servo: {'MG996R (10 kg·cm)' if motor_torque < 10 else 'STS3215 (16.5 kg·cm)'}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    # Run analysis
    results = run_full_analysis()

    # French press sequence
    analyze_french_press_sequence()

    # Generate plots
    print("\n\nGenerating plots...")
    fig = plot_analysis(results)
    fig.savefig('design_validation.png', dpi=150, bbox_inches='tight')
    print("Saved: design_validation.png")

    plt.show()
