# Stack

**Proprioceptive data collection and diffusion policy for dexterous manipulation.**

Stack enables collecting human demonstrations with joint-level proprioception using a custom instrumented glove, then training diffusion policies that can be deployed on robot hands.

<!-- TODO: Add demo GIF here -->
<!-- ![Demo](docs/assets/demo.gif) -->

## Key Features

- **Rich proprioception**: 4 joint encoders capture full finger articulation (not just gripper width)
- **iPhone-based tracking**: ARKit provides 6DoF wrist pose + RGB + depth
- **UMI-FT compatible**: Data format matches Stanford's UMI ecosystem
- **Diffusion policy**: State-of-the-art imitation learning from demonstrations

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        DATA COLLECTION                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐        │
│   │   iPhone    │    │   Glove     │    │   Laptop    │        │
│   │   ARKit     │    │  Encoders   │    │   Logger    │        │
│   │             │    │             │    │             │        │
│   │  Pose (7D)  │───▶│  Joints(4D) │───▶│  Zarr DB    │        │
│   │  RGB + D    │    │  100 Hz     │    │             │        │
│   └─────────────┘    └─────────────┘    └─────────────┘        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                         TRAINING                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Observations ────▶ Visual Encoder ────┐                      │
│   (RGB, Depth)       (CNN)              │                      │
│                                         ├──▶ Diffusion ──▶ Actions
│   Proprioception ──▶ State Encoder ─────┘    Policy     (11D)  │
│   (Pose + Joints)    (MLP)                   (UNet1D)          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Observation Space

| Source | Dimensions | Description |
|--------|------------|-------------|
| ARKit pose | 7 | Position (3) + Quaternion (4) |
| Index MCP | 1 | Metacarpophalangeal joint angle |
| Index PIP | 1 | Proximal interphalangeal joint angle |
| 3-Finger MCP | 1 | Combined middle/ring/pinky MCP |
| 3-Finger PIP | 1 | Combined middle/ring/pinky PIP |
| **Total** | **11** | Full proprioceptive state |

This is richer than UMI (7D pose + 1D width = 8D) and enables learning more dexterous behaviors.

## Hardware

### Instrumented Glove
- 4× AS5600 magnetic encoders (12-bit, 100Hz)
- TCA9548A I2C multiplexer
- ESP32 microcontroller
- 3D printed finger channels + palm

### Tracking
- iPhone 15 Pro with ARKit
- 15° downward-tilted mount
- RGB (60Hz) + LiDAR depth (30Hz)

See [`hardware/DESIGN.md`](hardware/DESIGN.md) for full design documentation.

## Installation

```bash
# Clone
git clone https://github.com/corneliusgruss/stack.git
cd stack

# Install (with training dependencies)
pip install -e ".[train]"

# Or minimal install for data collection only
pip install -e .
```

### ESP32 Firmware

```bash
# Install Arduino IDE or PlatformIO
# Open firmware/encoder_reader/encoder_reader.ino
# Flash to ESP32
```

## Quick Start

### 1. Collect Demonstrations

```bash
# Start data collection session
stack-collect --session demo_01

# In another terminal, start iPhone recording
# (iOS app documentation coming soon)
```

### 2. Process Data

```bash
# Convert raw sessions to training format
python -m stack.data.process --input data/raw --output data/processed
```

### 3. Train Policy

```bash
# Train diffusion policy
stack-train --config configs/default.yaml

# With W&B logging
stack-train --config configs/default.yaml --wandb
```

### 4. Evaluate

```bash
# Evaluate on held-out demonstrations
stack-eval --checkpoint outputs/checkpoint_0100.pt --data data/processed/val
```

## Project Structure

```
stack/
├── stack/                  # Main Python package
│   ├── data/               # Data loading and processing
│   ├── policy/             # Diffusion policy implementation
│   └── scripts/            # CLI entry points
├── firmware/               # ESP32 encoder reader
├── hardware/               # CAD files and design docs
├── configs/                # Training configurations
├── tests/                  # Unit tests
└── data/                   # Local data storage (gitignored)
    ├── raw/                # Raw demonstration sessions
    ├── processed/          # Zarr datasets for training
    └── models/             # Trained checkpoints
```

## References

- [Diffusion Policy](https://github.com/real-stanford/diffusion_policy) - Chi et al. RSS 2023
- [UMI](https://github.com/real-stanford/universal_manipulation_interface) - Chi et al. RSS 2024
- [UMI-FT](https://github.com/real-stanford/UMI-FT) - Force-aware manipulation

## License

MIT

---

*Built by Cornelius Gruss | BU Robotics | 2026*
