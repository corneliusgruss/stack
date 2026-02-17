# Custom Thumb-Opposition Gripper Design

**Project:** ME740 Bimanual Diffusion Policy (stack)
**Author:** Cornelius Gruss
**Started:** January 26, 2026
**Inspiration:** Sunday Robotics Skill Capture Glove, UMI-FT

---

## Design Overview

A proprioceptive data collection gripper inspired by Sunday Robotics' Skill Capture Glove. Two moving finger units (index + 3-finger) with joint encoders provide rich state information for imitation learning.

**Key insight:** With good enough proprioception (joint angles + wrist pose), we can train policies without needing robot hardware for validation. Policy outputs can be compared against held-out human demos or tested in simulation.

```
        TOP VIEW (right hand)

        ┌─────────────────────────────────┐
        │                                 │
        │   [3-Finger Unit]               │
        │   (middle+ring+pinky)           │
        │      2 joints ← MOVES           │
        │      2 encoders                 │
        │         ┌───┐                   │
        │         │   │                   │
        │         │   │                   │
        │    ┌────┴───┴────┐              │
        │    │             │              │
        │    │   [Index]   │              │
        │    │  2 joints   │ ← MOVES      │
        │    │  2 encoders │              │
        │    └──────┬──────┘              │
        │           │                     │
        │    ┌──────┴──────┐              │
        │    │    Palm     │              │
        │    │  + Teensy   │              │
        │    └──────┬──────┘              │
        │           │                     │
        │    ┌──────┴──────┐              │
        │    │   [Thumb]   │              │
        │    │   (fixed)   │              │
        │    └─────────────┘              │
        │                                 │
        │    [iPhone Mount - 15° tilt]    │
        └─────────────────────────────────┘

Grip modes:
- PINCH:  Index vs Thumb (precision grip)
- POWER:  All fingers curl around object
- WRAP:   3-finger cradles, index secures top
```

---

## State Representation

```
Full gripper state (per timestep):

┌─────────────────────────────────────────────────────────────┐
│ Wrist pose (from ARKit):     x, y, z, qw, qx, qy, qz  → 7D │
│ Index MCP angle (encoder):   θ₁                        → 1D │
│ Index PIP angle (encoder):   θ₂                        → 1D │
│ 3-Finger MCP angle (encoder): θ₃                       → 1D │
│ 3-Finger PIP angle (encoder): θ₄                       → 1D │
├─────────────────────────────────────────────────────────────┤
│ Total proprioception:                                  → 11D │
└─────────────────────────────────────────────────────────────┘

Plus: RGB image, depth image (observations for policy)
```

**Comparison to simpler approaches:**

| Approach | State Dim | Info Captured |
|----------|-----------|---------------|
| Original UMI (ArUco only) | 7 + 1 = 8D | Wrist pose + gripper width |
| This design | 7 + 4 = 11D | Wrist pose + full finger shape |
| Sunday (estimated) | 7 + 10+ | Wrist pose + all finger joints |

---

## Components

### 1. Thumb (Fixed)
- **Function:** Stationary opposition surface
- **Joints:** None (rigid)
- **Material:** PLA body + TPU pad for grip
- **Position:** Anatomically correct thumb opposition angle

### 2. Index Finger (Moving)
- **Function:** Primary precision grasping
- **Joints:** 2 (MCP + PIP style)
- **Actuation:** Cable/tendon driven (single pull = both joints flex)
- **Sensors:** 2x AS5600 magnetic encoders (one per joint)
- **Return:** Elastic cord

### 3. 3-Finger Unit (Moving) - Middle + Ring + Pinky
- **Function:** Power grasping, object cradling
- **Design:** Same mechanism as index, but wider (3 fingers as one unit)
- **Joints:** 2 (MCP + PIP style)
- **Actuation:** Cable/tendon driven (single pull = both joints flex)
- **Sensors:** 2x AS5600 magnetic encoders (one per joint)
- **Return:** Elastic cord

### 4. Palm
- **Function:** Houses electronics, cable routing, mounting
- **Features:**
  - Cable channels for both tendons
  - Teensy/ESP32 mount (reads 4 encoders)
  - Anchor points for elastic returns
  - iPhone bracket attachment point

### 5. iPhone Mount
- **Angle:** 15° downward tilt (per UMI-FT)
- **Phone:** iPhone 16 Pro (LiDAR + ARKit)
- **Attachment:** Rigid, vibration-free
- **View:** Must see thumb, both finger units, workspace

### 6. Trigger/Handle
- **Function:** User holds here, squeezes to close fingers
- **Mechanism:** Two triggers or linked mechanism
- **Option A:** Single trigger closes both finger units together
- **Option B:** Two independent triggers (more control, more complex)
- **Recommendation:** Start with Option A (simpler)

---

## Mechanism Detail: Finger Units

Both index and 3-finger unit use identical mechanism, just different widths.

### Cable Routing (Tendon Drive)

```
SIDE VIEW - Finger Unit (same for index and 3-finger)

         ╭─────╮  Fingertip
         │     │
         │  ●  │  ← cable anchor point
       ──┼──●──┼──  PIP joint + ENCODER 2
         │  │  │       (AS5600 + magnet)
         │  │  │  ← cable runs through channel
         │  │  │
       ──┼──●──┼──  MCP joint + ENCODER 1
         │  │  │       (AS5600 + magnet)
         │  │  │
         ╰──┼──╯
            │
         cable to trigger
```

### Encoder Mounting (AS5600)

```
CROSS-SECTION: Joint with Encoder

        Fixed side (proximal)          Rotating side (distal)
        ┌─────────────────┐           ┌─────────────────┐
        │                 │           │                 │
        │  ┌───────────┐  │           │  ┌─────────┐    │
        │  │  AS5600   │  │     ●     │  │ Magnet  │    │
        │  │   PCB     │  │ ────┼──── │  │ (6x2mm) │    │
        │  └───────────┘  │   axis    │  └─────────┘    │
        │                 │           │                 │
        └─────────────────┘           └─────────────────┘

        Encoder reads magnet rotation = joint angle

AS5600 specs:
- 12-bit resolution (4096 positions per revolution)
- I2C interface (address 0x36, need address jumper for 2nd encoder)
- 3.3V or 5V operation
- ~$3 each
```

**Encoder wiring:**
- 4 encoders total → I2C bus to Teensy/ESP32
- AS5600 has fixed I2C address (0x36), so need I2C multiplexer (TCA9548A) or use AS5600L variant (programmable address)
- Alternative: Use 2x AS5048A (SPI) for index, 2x AS5600 (I2C) for 3-finger

### Return Mechanism (Elastic)

```
SIDE VIEW - Cable vs Elastic

        PALM SIDE              BACK SIDE
        (flexor cable)         (extensor elastic)
              │                      │
              ▼                      ▼
         ╭────●────────────────●────╮
         │        Fingertip         │
         ╰────────────┬─────────────╯
                      │
            ══════════╪══════════  PIP joint + encoder
                      │
         ╭────●───────┼───────●────╮
         │            │            │
         │  Proximal  │            │
         ╰────────────┼────────────╯
                      │
            ══════════╪══════════  MCP joint + encoder
                      │
         ╭────────────┼────────────╮
         │   Palm     │            │
         │     ↓      ●────────────│← elastic anchor
         │   cable                 │
         └─────────────────────────┘

    CLOSE: Pull cable → finger flexes → elastic stretches
    OPEN:  Release cable → elastic pulls finger extended
```

### Elastic Sizing (unchanged from original analysis)

| Parameter | Value | Notes |
|-----------|-------|-------|
| Type | Elastic cord | 1-2mm diameter |
| Unstretched length | ~70-80mm | Along finger back |
| Stretched length | ~100-110mm | At full close |
| Extension | ~30mm | Matches cable travel |
| Force at full stretch | 1.5-2.5 N | 10-20% of grip force |
| Spring constant | 50-80 N/m | Very soft |

---

## Electronics

### Microcontroller Options

| Option | Pros | Cons | Cost |
|--------|------|------|------|
| **Teensy 4.0** | Fast, good I2C, Arduino compatible | Overkill for 4 encoders | ~$25 |
| **ESP32** | WiFi/BLE built-in, cheap | More complex setup | ~$10 |
| **Arduino Nano** | Simple, cheap | Limited I2C, slower | ~$5 |

**Recommendation:** ESP32 - can stream encoder data over WiFi/BLE to laptop, no cable tether needed.

### Wiring Diagram

```
                    ┌─────────────────────────────────┐
                    │           ESP32                 │
                    │                                 │
  Index MCP ────────┤ GPIO 21 (SDA) ──┬── I2C Bus    │
  (AS5600)          │ GPIO 22 (SCL) ──┘              │
                    │                                 │
  Index PIP ────────┤ (via TCA9548A multiplexer)     │
  (AS5600)          │                                 │
                    │                                 │
  3-Finger MCP ─────┤                                │
  (AS5600)          │                                 │
                    │                                 │
  3-Finger PIP ─────┤                                │
  (AS5600)          │                                 │
                    │                                 │
                    │ USB/WiFi ─────── Laptop        │
                    └─────────────────────────────────┘

Alternative: Use AS5048A (SPI) for 2 encoders, AS5600 (I2C) for other 2
            Avoids multiplexer, but mixed protocols
```

### Data Flow

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   Encoders   │    │    ESP32     │    │   Laptop     │
│  (4x AS5600) │───▶│  (100Hz)     │───▶│  (Python)    │
│              │    │              │    │              │
│  θ₁,θ₂,θ₃,θ₄ │    │  WiFi/USB    │    │  Sync with   │
│              │    │  streaming   │    │  ARKit data  │
└──────────────┘    └──────────────┘    └──────────────┘
                                               │
                                               ▼
                                        ┌──────────────┐
                                        │   iPhone     │
                                        │   (ARKit)    │
                                        │              │
                                        │  Wrist pose  │
                                        │  RGB + Depth │
                                        └──────────────┘
```

---

## Sensors Summary

### V1: Joint Encoders + ARKit (Current Scope)

| Sensor | Qty | Location | Data | Rate |
|--------|-----|----------|------|------|
| AS5600 | 4 | MCP/PIP joints | Joint angles (θ) | 100Hz |
| iPhone ARKit | 1 | Wrist mount | 6DoF pose | 60Hz |
| iPhone RGB | 1 | Wrist mount | Egocentric image | 60Hz |
| iPhone LiDAR | 1 | Wrist mount | Depth | 30Hz |

### V2: Add Force Sensing (Deferred)

| Sensor | Qty | Location | Data |
|--------|-----|----------|------|
| CoinFT | 2 | Thumb + Index tip | 6-axis F/T |
| FSR | 2 | Fingertips | Simple contact force |

---

## iPhone Integration (per UMI-FT)

### Hardware
- iPhone 16 Pro
- Rigid mount, 15° downward tilt
- Must see: thumb, both finger units, workspace

### Software (ARKit)
- 6DoF pose tracking (position + orientation)
- Main RGB: 60Hz
- Ultrawide RGB: 10Hz
- LiDAR depth: 30Hz

### Data Captured
- Wrist pose (from ARKit)
- Joint angles (from encoders via ESP32)
- RGB + depth images
- Timestamp synchronization critical

---

## Materials Summary

| Component | Material | Color | Why |
|-----------|----------|-------|-----|
| Finger bodies | PLA | Any | Rigid, easy to print |
| Finger joints | PLA | Any | Rigid hinges, encoder mount points |
| Fingertip pads | TPU 95A | Black | Compliant, grip |
| Thumb pad | TPU 95A | Black | Same as fingertips |
| Palm / handle | PLA | Any | Structural, electronics housing |
| iPhone mount | PLA | Any | Rigid, no flex |
| Tendon | Braided fishing line | - | Strong, low stretch (Sufix 832) |
| Return elastic | Elastic cord | - | Passive finger extension |

**Print Settings (typical):**
- PLA: 0.2mm layer height, 20% infill, 3 walls
- TPU 95A: 0.2mm layer height, 100% infill (thin pads), slow speed

---

## Bill of Materials

### V1 - Proprioceptive Gripper

| Component | Qty | Est. Cost | Source | Status |
|-----------|-----|-----------|--------|--------|
| PLA filament | - | $20 | EPIC/own | Have |
| TPU 95A filament | - | $30 | Amazon | TODO |
| Braided fishing line (tendon) | 1 | $15 | Amazon | TODO |
| Elastic cord (1-2mm) | 1 | $5 | Amazon | TODO |
| M2 screws + inserts | 1 set | $15 | Amazon | TODO |
| M3 screws + nuts | 1 set | $10 | Amazon | TODO |
| **AS5600 encoders + magnets** | 4 | $8.99 | Amazon | Ordered 2026-01-27 |
| **TCA9548A I2C multiplexer** | 1 | $6 (4-pack) | Amazon | Ordered 2026-01-27 |
| **ESP32 dev board (HiLetgo)** | 1 | $9.99 | Amazon | Ordered 2026-01-27 |
| Dupont wires | 1 set | $7 | Amazon | Ordered 2026-01-27 |
| iPhone 16 Pro | 1 | $0 | Own | Have |
| **V1 Total** | | **~$127** | | |

### V2 - Force Sensing (Deferred)

| Component | Qty | Est. Cost | Source | Status |
|-----------|-----|-----------|--------|--------|
| CoinFT sensors | 2 | $20 | Stanford? | Email sent |
| Additional Teensy (if needed) | 1 | $25 | PJRC | - |

### Optional - Robot Deployment

| Component | Qty | Est. Cost | Source | Status |
|-----------|-----|-----------|--------|--------|
| SO-101 follower kits | 2 | ~$250 | Seeed Studio | Optional |

---

## Design Constraints

### Must Have
- Human hand scale (natural demonstration)
- Comfortable for 30+ min sessions
- All joints visible to iPhone camera (or use encoders)
- Rigid iPhone mount (no wobble)

### Nice to Have
- SO-101 compatible mount (for optional robot deployment)
- Modular finger units (easy to swap/repair)
- Clean cable management

---

## Open Questions

1. ~~**Joint coupling:** Free underactuated vs mechanical linkage?~~ → Free underactuated (simpler)
2. **Trigger mechanism:** Single trigger for both units, or independent?
3. **I2C addressing:** Multiplexer vs mixed SPI/I2C vs AS5600L?
4. **iPhone mount:** Integrated into palm or separate bracket?
5. **Left vs Right:** Build one hand first, mirror for bimanual later?
6. **Encoder calibration:** How to establish zero position consistently?

---

## Validation Strategy (No Robot Required)

Since robot deployment is optional, validation can be done via:

### 1. Held-Out Demo Comparison
- Collect N demos, train on N-k, test on k
- Policy predicts action sequence, compare to ground truth
- Metrics: position error, joint angle error, timing

### 2. Simulation Deployment
- Model gripper in MuJoCo/PyBullet
- Run trained policy in sim
- Measure task success rate

### 3. Action Prediction Visualization
- Play back demo video
- Overlay policy-predicted gripper state
- Visual sanity check

---

## References

- [Sunday Robotics](https://www.sunday.ai/) - Skill Capture Glove design philosophy
- [UMI-FT](https://umi-ft.github.io/) - iPhone + ARKit approach
- [UMI](https://github.com/real-stanford/universal_manipulation_interface) - Original UMI
- [CoinFT](https://coin-ft.github.io/) - Force sensing (V2)
- [Robot Nano Hand](https://robotnanohand.com/) - Tendon routing reference
- [AS5600 Datasheet](https://ams.com/as5600) - Magnetic encoder

---

## Next Steps

### Immediate (This Week)
- [ ] Measure own hand dimensions (thumb, index, palm, 3-finger width)
- [ ] Decide trigger mechanism (single vs dual)
- [x] Order electronics (ESP32, AS5600 x4, magnets, multiplexer) ← Ordered 2026-01-27, ~$32
- [ ] CAD v0.1: Single finger joint with encoder mount
- [ ] Print test joint, verify encoder fits

### After Test Joint Works
- [ ] CAD full index finger (2 joints, 2 encoders)
- [ ] CAD 3-finger unit (same mechanism, wider)
- [ ] CAD palm with electronics bay
- [ ] CAD iPhone mount

### Software (Parallel)
- [ ] ESP32 firmware: read 4 encoders, stream over WiFi
- [ ] Python receiver: sync encoder data with ARKit
- [ ] ARKit app: pose + RGB + depth capture

---

## Revision History

| Date | Change |
|------|--------|
| 2026-01-26 | Initial design document created |
| 2026-01-27 | Major update: Added 4 joint encoders (AS5600), 3-finger unit now moves (same mechanism as index), robot deployment now optional, added electronics section, updated BOM |
| 2026-01-29 | Major redesign: Fully actuated (direct servo per joint), parallel model geometry |

---

## Design Update v2 (2026-01-29)

### Key Design Changes

**From cable-driven underactuated → Fully actuated with direct servos**

| Aspect | Old Design | New Design |
|--------|------------|------------|
| Actuation | Single cable per finger | Direct servo at each joint |
| DOF | Coupled (1 cable, 2 joints) | Independent (2 servos, 2 joints) |
| Return mechanism | Elastic cord | Servo (bidirectional) |
| Complexity | Cables + elastic routing | Just mount servos |

### Why Fully Actuated?

Observation from Sunday Robotics videos:
- Sock folding: MCP rotates, PIP stays straight
- Wine glass: PIP curls, MCP stays fixed
- **Conclusion:** MCP and PIP are independently controlled

This also explains why 2 encoders per finger are needed - if joints were coupled, one encoder would suffice.

### Parallel Model Geometry

At **neutral position** (MCP=0°, PIP=0°):
- Finger and thumb are **parallel vertical lines**
- Separated by gap **D** between inner surfaces
- Thumb tip at approximately **PIP height**

```
NEUTRAL POSITION (MCP=0°, PIP=0°)

    finger tip  ●                      ● thumb tip
                │                      │
                │                      │
    PIP height ─● PIP                  │
                │                      │
                │                      │
    MCP height ─● MCP ─────────────────● thumb base
              (0,0)
                │←─────── D ──────────→│
                   (inner surface gap)
```

### Locked Dimensions

```
┌─────────────────────────────────────────────────────────────┐
│  FINGER                                                     │
│    L1 (proximal):      45 mm                                │
│    L2 (distal):        40 mm                                │
│    Total length:       85 mm                                │
│    Width:              22 mm                                │
│    PIP range:          0° to 90° (cannot hyperextend)       │
│    MCP range:          Can extend and flex                  │
├─────────────────────────────────────────────────────────────┤
│  THUMB                                                      │
│    Length:             45-50 mm                             │
│    Width:              22 mm                                │
│    Position:           Parallel to finger at neutral        │
│    Gap D:              50 mm (between inner surfaces)       │
├─────────────────────────────────────────────────────────────┤
│  ACTUATION                                                  │
│    Type:               Direct servo at each joint           │
│    Servos per finger:  2 (MCP + PIP independent)            │
│    No cables needed                                         │
│    No elastic needed                                        │
├─────────────────────────────────────────────────────────────┤
│  SENSING                                                    │
│    Encoders per finger: 2 (MCP + PIP)                       │
│    Type:               AS5600 magnetic encoder              │
│    Why 2 encoders:     Joints are independent (not coupled) │
└─────────────────────────────────────────────────────────────┘
```

### Glove vs Robot Hand

Both variants share **identical external kinematics** for policy transfer:

| Variant | Actuation | Sensing |
|---------|-----------|---------|
| **Glove** | Human finger inside | 2 encoders per finger |
| **Robot** | 2 servos per finger | 2 encoders per finger |

### 3D Print Tolerances

For joints with 8mm pin:
- **Clearance fit (rotation):** Hole = 8.3mm, Pin = 8.0mm (0.3mm gap)
- **Press fit (fixed):** Hole = 7.9mm, Pin = 8.0mm (0.1mm interference)

### First Print Goals

1. Single finger with two joints
2. Verify servo fits in 22mm width
3. Test joint rotation (smooth 0-90°)
4. Validate encoder mounting

---

## Build Log

### 2026-01-31 (Friday)

**Printer:** Bambu Lab P1S (purchased 2026-01-29)

**TPU Finger Tip Pads - Press Fit Testing**
- Printing sizing samples for TPU pads that press-fit into PLA finger tips
- First batch (3 samples): None fit - interference too tight
- Second batch: Printing now, adjusted tolerances

**Notes:**
- TPU compresses, so need looser fit than expected
- Will document final working tolerance once dialed in

### 2026-02-01 (Saturday)

**Joint Mechanism - Snap-Fit Success**

Switched from press-fit pin to cantilever snap-fit. First test worked.

**Working Parameters:**
- Ridge OD: 8mm
- Tip wall ID: 8.4mm (clearance for rotation)
- Cantilever thickness: 1.5mm
- Catch depth: 1mm

**Notes:**
- Circles aren't perfect (FDM faceting) but good enough
- Snap-fit more forgiving than press-fit for 3D printing
- Will copy this joint design to all 4 joints (Index MCP, Index PIP, 3-Finger MCP, 3-Finger PIP)

### 2026-02-11 (Tuesday)

**Electronics Chain - Fully Validated**

Soldered AS5600 encoder to TCA9548A mux (CH0/SD0), wired mux to ESP32 (3V3, GND, GPIO21=SDA, GPIO22=SCL).

**Results:**
- ESP32-D0WD-V3 confirmed working (dual core, 240 MHz, 4 MB flash)
- TCA9548A detected at 0x70 on I2C bus
- AS5600 detected at 0x36 behind mux channel 0
- Encoder streaming smooth angle data at 100 Hz
- Remaining 3 encoders just need soldering to CH1-CH3 (no firmware changes)

**Issues:**
- Mini breadboard caused I2C failures — unreliable contacts. Fixed by soldering header pins directly.
- Encoder readings jittery when magnet/PCB not physically fixed (expected — will be stable once mounted in joint)

**Next:**
- Print magnet mount test jig: validate air gap (target 1-2mm), centering on rotation axis, reading stability
- Then proceed to full finger assembly with encoder + magnet integrated
