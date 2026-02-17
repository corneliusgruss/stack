"""
Gripper Design Calculations

A working document capturing the design process for the thumb-opposition gripper.
Two variants: glove (data collection) and robot hand (deployment).

Key insight: Both variants share identical external kinematics so learned policies transfer.

Author: Cornelius Gruss
Started: January 2026
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple

# =============================================================================
# DESIGN DECISIONS LOG
# =============================================================================
#
# 1. No torsion springs - using elastic cord/band for return instead
#    - Simpler mechanism
#    - Glove doesn't fight human hand
#    - Elastic just needs to overcome friction, not provide precision force
#
# 2. Two-link finger model (MCP + PIP joints)
#    - Drops DIP joint from human hand (acceptable simplification)
#    - Matches Sunday Robotics approach
#
# 3. Fixed thumb position
#    - Actuated thumb adds complexity
#    - Fixed position at "sweet spot" works for most grips
#
# 4. Contact angle at ~60° per joint
#    - Balanced tradeoff between aperture range and force range
#    - Can grip any size object from 0 to max aperture
#    - Contact angle affects force capability, not grip range
#
# =============================================================================


@dataclass
class FingerParams:
    """Finger geometry parameters."""
    L1: float  # Proximal link length (mm)
    L2: float  # Distal link length (mm)
    width: float = 22.0  # Finger width (mm) - for contact pad offset

    @property
    def total_length(self) -> float:
        return self.L1 + self.L2

    @property
    def pad_offset(self) -> float:
        """Offset from centerline to contact pad (flexor side)."""
        return self.width / 2


@dataclass
class ThumbParams:
    """Thumb tip position parameters."""
    tx: float  # Tip X position (mm) - horizontal distance from index MCP
    ty: float  # Tip Y position (mm) - vertical distance from index MCP


@dataclass
class ThumbGeometry:
    """Full thumb geometry including base (MCP) position."""
    tip_x: float       # Thumb tip X (mm)
    tip_y: float       # Thumb tip Y (mm)
    base_x: float      # Thumb MCP/base X (mm)
    base_y: float      # Thumb MCP/base Y (mm)

    @property
    def length(self) -> float:
        """Thumb length from base to tip."""
        dx = self.tip_x - self.base_x
        dy = self.tip_y - self.base_y
        return np.sqrt(dx**2 + dy**2)

    @property
    def angle_rad(self) -> float:
        """Thumb angle from base to tip (radians from +X axis)."""
        dx = self.tip_x - self.base_x
        dy = self.tip_y - self.base_y
        return np.arctan2(dy, dx)

    @property
    def angle_deg(self) -> float:
        """Thumb angle in degrees."""
        return np.degrees(self.angle_rad)

    @property
    def palm_width(self) -> float:
        """Distance from index MCP (origin) to thumb MCP."""
        return np.sqrt(self.base_x**2 + self.base_y**2)

    @property
    def palm_angle_rad(self) -> float:
        """Angle of palm line (index MCP to thumb MCP) from +X axis."""
        return np.arctan2(self.base_y, self.base_x)

    @property
    def palm_angle_deg(self) -> float:
        """Palm angle in degrees."""
        return np.degrees(self.palm_angle_rad)


@dataclass
class JointLimits:
    """Joint angle limits."""
    theta_min: float  # Minimum angle - fully extended (radians)
    theta_max: float  # Maximum angle - fully flexed (radians)

    @property
    def theta_min_deg(self) -> float:
        return np.degrees(self.theta_min)

    @property
    def theta_max_deg(self) -> float:
        return np.degrees(self.theta_max)


@dataclass
class ContactParams:
    """Contact configuration - where fingertip meets thumb."""
    theta1_c: float  # MCP angle at contact (radians)
    theta2_c: float  # PIP angle at contact (radians)

    @property
    def theta1_c_deg(self) -> float:
        return np.degrees(self.theta1_c)

    @property
    def theta2_c_deg(self) -> float:
        return np.degrees(self.theta2_c)


# =============================================================================
# STEP 1: FINGER LINK LENGTHS
# =============================================================================
#
# Requirements:
#   - Max aperture ~70-80mm for general purpose gripping (cups, bottles, etc)
#   - Must fit human finger inside for glove variant
#
# Human finger reference (adult index):
#   - Proximal phalanx: ~40-50mm
#   - Middle + distal: ~40-50mm combined
#
# Decision: L1=45mm, L2=40mm, width=22mm
#   - Total reach 85mm
#   - Matches human finger proportions
#   - Allows comfortable glove fit
#   - Width of 22mm provides 11mm on each side from centerline
#   - This accommodates 8mm cable offset and 8mm elastic offset
#

FINGER = FingerParams(
    L1=45.0,   # mm - proximal link
    L2=40.0,   # mm - distal link
    width=22.0 # mm - finger width (contact pad offset = width/2 = 11mm)
)

# Joint limits (same for MCP and PIP)
JOINT_LIMITS = JointLimits(
    theta_min=0.0,         # Fully extended
    theta_max=np.pi / 2,   # 90° - fully flexed
)


# =============================================================================
# STEP 2: CONTACT ANGLE SELECTION
# =============================================================================
#
# The contact angle determines WHERE on the fingertip trajectory the thumb sits.
#
# Tradeoff:
#   - Early contact (low θ): Large force range, small aperture range
#   - Late contact (high θ): Small force range, large aperture range
#   - Mid contact (~60°): Balanced - good for general purpose
#
# Note: Contact angle affects FORCE capability at different object sizes,
#       NOT the range of object sizes you can grip.
#       You can grip any size from 0 to max aperture regardless of contact angle.
#
# Observation from Sunday Robotics videos:
#   - MCP joint rarely at 0° during tasks
#   - Typical "home" position around 25° MCP
#   - So theoretical max aperture isn't the practical operating range
#
# Decision: θ1_c = 60°, θ2_c = 60°
#

CONTACT = ContactParams(
    theta1_c=np.radians(60),  # MCP angle at contact
    theta2_c=np.radians(60),  # PIP angle at contact
)


# =============================================================================
# STEP 3: CALCULATE THUMB POSITION
# =============================================================================
#
# Thumb tip = CONTACT PAD position when finger is at contact configuration
#
# Important: The kinematic model gives centerline positions, but actual contact
# happens on the flexor surface, offset by half the finger width (11mm).
#
# Forward kinematics (2-link planar arm) for centerline:
#   x = L1*sin(θ1) + L2*sin(θ1 + θ2)
#   y = L1*cos(θ1) + L2*cos(θ1 + θ2)
#
# Contact pad position (offset toward flexor side):
#   pad_x = tip_x + (width/2) * cos(θ1 + θ2)
#   pad_y = tip_y - (width/2) * sin(θ1 + θ2)
#
# Coordinate system:
#   - Origin at MCP joint
#   - Y points UP (toward fingertip when extended)
#   - X points RIGHT (toward thumb)
#   - Positive angles = flexion (finger curls right and down)
#   - Flexor normal = (cos(θ1+θ2), -sin(θ1+θ2)) perpendicular to distal link
#

def calculate_fingertip_position(finger: FingerParams,
                                  theta1: float,
                                  theta2: float) -> Tuple[float, float]:
    """
    Calculate fingertip (x, y) position for given joint angles.
    This is the KINEMATIC tip (centerline of the link).

    Args:
        finger: Finger geometry
        theta1: MCP joint angle (radians), 0 = extended
        theta2: PIP joint angle (radians), 0 = extended

    Returns:
        (x, y) position in mm
    """
    x = finger.L1 * np.sin(theta1) + finger.L2 * np.sin(theta1 + theta2)
    y = finger.L1 * np.cos(theta1) + finger.L2 * np.cos(theta1 + theta2)
    return x, y


def calculate_flexor_normal(theta1: float, theta2: float) -> Tuple[float, float]:
    """
    Calculate the flexor-side normal direction for the distal link.
    This is perpendicular to the distal link, pointing toward the flexor side
    (where cable runs and contact pad is located).

    The flexor normal is the link direction rotated 90° clockwise.
    """
    total_angle = theta1 + theta2
    nx = np.cos(total_angle)
    ny = -np.sin(total_angle)
    return nx, ny


def calculate_contact_pad_position(finger: FingerParams,
                                   theta1: float,
                                   theta2: float) -> Tuple[float, float]:
    """
    Calculate the contact pad position (where fingertip actually touches objects).

    The contact pad is on the flexor side of the distal link, offset from the
    kinematic centerline by half the finger width.

    This is where the thumb should be positioned to meet the finger.
    """
    # Kinematic tip (centerline)
    tip_x, tip_y = calculate_fingertip_position(finger, theta1, theta2)

    # Flexor normal direction
    nx, ny = calculate_flexor_normal(theta1, theta2)

    # Contact pad is offset toward flexor side
    pad_x = tip_x + finger.pad_offset * nx
    pad_y = tip_y + finger.pad_offset * ny

    return pad_x, pad_y


def calculate_thumb_position(finger: FingerParams,
                             contact: ContactParams) -> ThumbParams:
    """
    Calculate thumb position from contact configuration.

    The thumb is placed where the CONTACT PAD is at the contact angles,
    NOT the kinematic tip. This accounts for finger width - the contact
    surface is on the flexor side, offset from centerline by half the width.
    """
    tx, ty = calculate_contact_pad_position(
        finger, contact.theta1_c, contact.theta2_c
    )
    return ThumbParams(tx=tx, ty=ty)


def calculate_thumb_geometry(thumb_tip: ThumbParams,
                             perpendicular_angle: float,
                             thumb_length: float = 40.0) -> ThumbGeometry:
    """
    Calculate full thumb geometry given the tip position and palm constraint.

    The palm line (index MCP to thumb MCP) is perpendicular to the index
    proximal link when θ1 = perpendicular_angle.

    Args:
        thumb_tip: Thumb tip position (where contact happens)
        perpendicular_angle: Index MCP angle (radians) at which proximal link
                            is perpendicular to palm line (e.g., 25° = 0.436 rad)
        thumb_length: Desired thumb length in mm (default 40mm to match finger)

    Returns:
        ThumbGeometry with base and tip positions
    """
    # Palm line direction: perpendicular to proximal link at given angle
    # Proximal link at θ1 points in direction (sin(θ1), cos(θ1))
    # Palm line is 90° clockwise from this: (cos(θ1), -sin(θ1))
    palm_dir_x = np.cos(perpendicular_angle)
    palm_dir_y = -np.sin(perpendicular_angle)

    # Thumb MCP is somewhere along the palm line from origin
    # We solve for palm_width such that thumb has desired length
    #
    # thumb_tip = thumb_base + thumb_vector
    # |thumb_vector| = thumb_length
    #
    # thumb_base = (W * palm_dir_x, W * palm_dir_y)
    # thumb_vector = (tip_x - W * palm_dir_x, tip_y - W * palm_dir_y)
    #
    # thumb_length² = (tip_x - W*palm_dir_x)² + (tip_y - W*palm_dir_y)²

    # Expand and solve quadratic for W:
    # Let a = palm_dir_x² + palm_dir_y² = 1
    # Let b = -2*(tip_x*palm_dir_x + tip_y*palm_dir_y)
    # Let c = tip_x² + tip_y² - thumb_length²
    # W² + bW + c = 0

    tx, ty = thumb_tip.tx, thumb_tip.ty
    b = -2 * (tx * palm_dir_x + ty * palm_dir_y)
    c = tx**2 + ty**2 - thumb_length**2

    discriminant = b**2 - 4*c
    if discriminant < 0:
        raise ValueError("No valid palm width for given thumb length")

    # Two solutions - take the smaller positive one
    W1 = (-b + np.sqrt(discriminant)) / 2
    W2 = (-b - np.sqrt(discriminant)) / 2

    # Choose the solution that makes anatomical sense (positive, reasonable size)
    if W2 > 0:
        palm_width = W2
    else:
        palm_width = W1

    base_x = palm_width * palm_dir_x
    base_y = palm_width * palm_dir_y

    return ThumbGeometry(
        tip_x=tx,
        tip_y=ty,
        base_x=base_x,
        base_y=base_y
    )


# Calculate thumb tip position (where contact happens)
THUMB = calculate_thumb_position(FINGER, CONTACT)

# =============================================================================
# STEP 3b: CALCULATE THUMB BASE (MCP) POSITION
# =============================================================================
#
# Observation from Sunday Robotics: the palm line (index MCP to thumb MCP)
# is approximately perpendicular to the index proximal link when the hand
# is in a "home" or slightly closed position (~25°).
#
# This gives us a constraint to find the thumb base position:
#   - Thumb tip is at (68.1, -7.0) from step 3
#   - Palm line perpendicular to proximal link at θ1 = 25°
#   - Thumb length ~40mm (similar to finger links)
#
# From these constraints we can solve for palm width and thumb base position.
#

PERPENDICULAR_ANGLE = np.radians(25)  # θ1 at which palm line ⊥ proximal link
THUMB_LENGTH = 40.0  # mm - desired thumb length

THUMB_GEOMETRY = calculate_thumb_geometry(THUMB, PERPENDICULAR_ANGLE, THUMB_LENGTH)


# =============================================================================
# STEP 4: CALCULATE APERTURE RANGE
# =============================================================================
#
# Max aperture = distance from thumb to fingertip when fully extended (θ=0,0)
# Min aperture = 0 (when fingertip touches thumb at contact angles)
#

def calculate_aperture(finger: FingerParams,
                       thumb: ThumbParams,
                       theta1: float,
                       theta2: float) -> float:
    """
    Calculate grip aperture (distance from contact pad to thumb).
    Uses contact pad position, not kinematic tip, for accurate aperture.
    """
    pad_x, pad_y = calculate_contact_pad_position(finger, theta1, theta2)
    distance = np.sqrt((pad_x - thumb.tx)**2 + (pad_y - thumb.ty)**2)
    return distance


def calculate_max_aperture(finger: FingerParams, thumb: ThumbParams) -> float:
    """Max aperture when finger is fully extended."""
    return calculate_aperture(finger, thumb, 0, 0)


MAX_APERTURE = calculate_max_aperture(FINGER, THUMB)

# Result: ~111mm theoretical max
# Practical range is smaller since finger rarely fully extends


# =============================================================================
# STEP 5: CABLE ACTUATION
# =============================================================================
#
# Cable moment arms r1 (MCP) and r2 (PIP) determine:
#   - Torque multiplication: τ = T * r
#   - Cable travel: ΔL = r1*θ1 + r2*θ2
#   - Joint coupling: how θ1 and θ2 relate during motion
#
# Design considerations:
#   - Larger r = more torque but more cable travel needed
#   - r1/r2 ratio affects finger curl shape
#   - r1 ≈ r2 gives uniform curl (both joints flex equally)
#
# Decision: r1 = r2 = 8mm
#   - Uniform curl like Sunday Robotics
#   - Cable travel to contact (60°/60°): 8 * 2.09 = 16.7mm
#   - Cable travel to full flex (90°/90°): 8 * 3.14 = 25.1mm
#   - Well within servo capability (typical horn can pull 50-80mm)
#   - 8mm pulleys fit in reasonably sized joint housings
#
# Joint limits:
#   - Contact angle: 60° per joint (where fingertip meets thumb)
#   - Maximum flexion: 90° per joint (physical limit)
#   - Finger CAN flex beyond contact to increase grip force
#
# Cable routing at joints (must not pinch at 90° max flexion):
#   - Option A: Channel/groove in joint at r=8mm radius
#   - Option B: Small pulley or rod at joint, r=8mm
#   - Cable must maintain 8mm distance from joint axis through 0-90° range
#
# Cable anchor:
#   - Attaches to distal link, ~5-10mm back from fingertip
#   - On flexor side of distal link
#

@dataclass
class CableParams:
    """Cable actuation parameters."""
    r1: float  # MCP moment arm / pulley radius (mm)
    r2: float  # PIP moment arm / pulley radius (mm)

    def cable_travel(self, theta1: float, theta2: float) -> float:
        """Total cable pull for given joint angles."""
        return self.r1 * theta1 + self.r2 * theta2

    def joint_torques(self, tension: float) -> Tuple[float, float]:
        """Joint torques from cable tension."""
        return tension * self.r1, tension * self.r2


CABLE = CableParams(
    r1=8.0,  # mm - MCP pulley radius (uniform curl)
    r2=8.0,  # mm - PIP pulley radius (uniform curl)
)


# =============================================================================
# STEP 6: ELASTIC RETURN
# =============================================================================
#
# Single elastic cord runs from palm anchor to fingertip back (Option A).
# As finger flexes, elastic wraps around joints and stretches.
#
# Geometry insight:
#   - Elastic offset d = distance from joint axis to elastic path
#   - At each joint, flexion adds arc length: d × θ
#   - Total extension = d × (θ1 + θ2)
#
# Design choice: d = r = 8mm (elastic offset = cable moment arm)
#   - This means elastic extension = cable travel (elegant!)
#   - At contact (120° total): extension = 16.7mm
#   - At full flex (180° total): extension = 25.1mm
#
# Force/torque:
#   - F_elastic = k × extension = k × d × (θ1 + θ2)
#   - τ_elastic = F × d = k × d² × (θ1 + θ2) (same at both joints)
#
# Sizing requirements:
#   - Must overcome friction: τ_elastic > τ_friction (~5-20 N·mm)
#   - Must not fight grip: τ_elastic ≈ 10-20% of τ_cable
#   - At T=10N: τ_cable = 80 N·mm, so τ_elastic ≈ 8-16 N·mm
#
# Solving for k at contact (θ1 + θ2 = 2.09 rad):
#   τ_elastic = k × d² × (θ1 + θ2) = k × 64 × 2.09 = 134k
#   For τ_elastic = 10 N·mm: k = 10/134 = 0.075 N/mm = 75 N/m
#

@dataclass
class ElasticParams:
    """Elastic return parameters."""
    d: float      # Offset from joint axes (mm) - where elastic runs
    k: float      # Stiffness (N/mm)
    L_rest: float # Rest length when extended (mm)

    def extension(self, theta1: float, theta2: float) -> float:
        """Elastic extension for given joint angles."""
        return self.d * (theta1 + theta2)

    def force(self, theta1: float, theta2: float) -> float:
        """Elastic force (N)."""
        return self.k * self.extension(theta1, theta2)

    def torque(self, theta1: float, theta2: float) -> float:
        """Elastic torque at each joint (N·mm). Same for both joints."""
        return self.force(theta1, theta2) * self.d


# Rest length = path along back of finger when extended
# Approximately L1 + L2 + anchor offsets
ELASTIC_REST_LENGTH = FINGER.L1 + FINGER.L2 + 15  # ~100mm with anchors

ELASTIC = ElasticParams(
    d=8.0,       # mm - matches cable moment arm for elegance
    k=0.075,     # N/mm = 75 N/m - soft elastic cord
    L_rest=ELASTIC_REST_LENGTH,
)


# =============================================================================
# STEP 7: SERVO FEASIBILITY
# =============================================================================
#
# Can a standard hobby servo drive this mechanism?
#
# Requirements:
#   - Cable travel: 25mm (to full flex at 90°/90°)
#   - Cable tension: ~10N for useful grip force
#
# Servo geometry:
#   - Horn radius R_horn determines torque/travel tradeoff
#   - Cable travel = R_horn × θ_servo
#   - Servo torque = T_cable × R_horn
#
# With R_horn = 15mm:
#   - For 25mm travel: θ_servo = 25/15 = 96° (well within 180° range)
#   - For 10N tension: τ_servo = 10 × 15 = 150 N·mm = 1.5 kg·cm
#
# Common servos:
#   - SG90 (micro):       1.8 kg·cm - borderline
#   - MG90S (micro metal): 2.2 kg·cm - should work ✓
#   - MG996R (standard):  10 kg·cm  - plenty of margin ✓
#
# Grip force estimate:
#   - Cable tension 10N → joint torque 80 N·mm
#   - Minus elastic ~10 N·mm → net 70 N·mm
#   - With ~35mm moment arm at contact → F_grip ≈ 2N
#   - Holding a 25mm cube (~15g) needs ~0.5N → plenty of margin
#
# VERDICT: Standard hobby servo (MG90S or MG996R) works fine.
#

@dataclass
class ServoParams:
    """Servo and horn parameters."""
    horn_radius: float    # mm
    max_rotation: float   # radians
    stall_torque: float   # N·mm

    def cable_travel(self, theta: float) -> float:
        """Cable travel for given servo rotation."""
        return self.horn_radius * theta

    def max_cable_travel(self) -> float:
        """Maximum cable travel at full rotation."""
        return self.cable_travel(self.max_rotation)

    def torque_for_tension(self, tension: float) -> float:
        """Servo torque required for given cable tension."""
        return tension * self.horn_radius

    def max_tension(self) -> float:
        """Maximum cable tension at stall torque."""
        return self.stall_torque / self.horn_radius


# Using MG90S as reference (compact, sufficient torque)
SERVO = ServoParams(
    horn_radius=15.0,           # mm
    max_rotation=np.pi,         # 180°
    stall_torque=220.0,         # N·mm (2.2 kg·cm)
)


# =============================================================================
# SUMMARY OUTPUT
# =============================================================================

def print_design_summary():
    """Print current design parameters."""
    print("=" * 60)
    print("GRIPPER DESIGN SUMMARY")
    print("=" * 60)

    print("\n[FINGER]")
    print(f"  L1 (proximal):  {FINGER.L1} mm")
    print(f"  L2 (distal):    {FINGER.L2} mm")
    print(f"  Total length:   {FINGER.total_length} mm")
    print(f"  Width:          {FINGER.width} mm")
    print(f"  Pad offset:     {FINGER.pad_offset} mm (from centerline to contact surface)")
    print(f"  Joint range:    {JOINT_LIMITS.theta_min_deg:.0f}° - {JOINT_LIMITS.theta_max_deg:.0f}°")

    print("\n[CONTACT CONFIGURATION]")
    print(f"  θ1_c (MCP):     {CONTACT.theta1_c_deg}°")
    print(f"  θ2_c (PIP):     {CONTACT.theta2_c_deg}°")

    print("\n[THUMB TIP] (where contact pad meets thumb at 60°/60°)")
    print(f"  tip_x:          {THUMB.tx:.1f} mm")
    print(f"  tip_y:          {THUMB.ty:.1f} mm")
    # Also show kinematic tip for reference
    kin_x, kin_y = calculate_fingertip_position(FINGER, CONTACT.theta1_c, CONTACT.theta2_c)
    print(f"  (kinematic tip: ({kin_x:.1f}, {kin_y:.1f}) - centerline reference)")

    print("\n[THUMB GEOMETRY] (base/MCP from palm line constraint)")
    print(f"  base_x:         {THUMB_GEOMETRY.base_x:.1f} mm")
    print(f"  base_y:         {THUMB_GEOMETRY.base_y:.1f} mm")
    print(f"  thumb length:   {THUMB_GEOMETRY.length:.1f} mm")
    print(f"  thumb angle:    {THUMB_GEOMETRY.angle_deg:.1f}° (from +X axis)")
    print(f"  palm width:     {THUMB_GEOMETRY.palm_width:.1f} mm (index MCP to thumb MCP)")
    print(f"  palm angle:     {THUMB_GEOMETRY.palm_angle_deg:.1f}° (from +X axis)")
    print(f"  perpendicular at θ1 = {np.degrees(PERPENDICULAR_ANGLE):.0f}°")

    print("\n[APERTURE]")
    print(f"  Max (extended): {MAX_APERTURE:.1f} mm")
    print(f"  Min (contact):  0 mm")

    print("\n[CABLE ACTUATION]")
    print(f"  r1 (MCP):       {CABLE.r1} mm")
    print(f"  r2 (PIP):       {CABLE.r2} mm")
    max_travel = CABLE.cable_travel(np.pi/2, np.pi/2)
    print(f"  Max travel:     {max_travel:.1f} mm (at 90°/90°)")

    print("\n[ELASTIC RETURN]")
    print(f"  Offset d:       {ELASTIC.d} mm")
    print(f"  Stiffness k:    {ELASTIC.k} N/mm = {ELASTIC.k * 1000:.0f} N/m")
    print(f"  Rest length:    {ELASTIC.L_rest} mm")
    ext_contact = ELASTIC.extension(CONTACT.theta1_c, CONTACT.theta2_c)
    force_contact = ELASTIC.force(CONTACT.theta1_c, CONTACT.theta2_c)
    torque_contact = ELASTIC.torque(CONTACT.theta1_c, CONTACT.theta2_c)
    print(f"  At contact:")
    print(f"    Extension:    {ext_contact:.1f} mm")
    print(f"    Force:        {force_contact:.2f} N")
    print(f"    Torque:       {torque_contact:.1f} N·mm per joint")

    print("\n[SERVO]")
    print(f"  Horn radius:    {SERVO.horn_radius} mm")
    print(f"  Max rotation:   {np.degrees(SERVO.max_rotation):.0f}°")
    print(f"  Stall torque:   {SERVO.stall_torque} N·mm ({SERVO.stall_torque/100:.1f} kg·cm)")
    print(f"  Max cable pull: {SERVO.max_cable_travel():.1f} mm")
    print(f"  Max tension:    {SERVO.max_tension():.1f} N")
    rotation_for_contact = CABLE.cable_travel(CONTACT.theta1_c, CONTACT.theta2_c) / SERVO.horn_radius
    print(f"  Rotation to contact: {np.degrees(rotation_for_contact):.0f}°")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    print_design_summary()

    # Also show contact pad trajectory (this is where actual contact happens)
    print("\n[CONTACT PAD TRAJECTORY] (flexor surface, offset from centerline)")
    print("  θ1     θ2     Pad X     Pad Y     Aperture (mm)")
    print("  " + "-" * 50)
    for angle in [0, 15, 30, 45, 60, 75, 90]:
        theta = np.radians(angle)
        pad_x, pad_y = calculate_contact_pad_position(FINGER, theta, theta)
        aperture = calculate_aperture(FINGER, THUMB, theta, theta)
        print(f"  {angle:3d}°   {angle:3d}°   {pad_x:7.1f}   {pad_y:7.1f}   {aperture:7.1f}")
