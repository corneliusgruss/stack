# stack - Project Context

## What This Is
Proprioceptive data collection and diffusion policy for dexterous manipulation.

**Goal:** Build a system that rivals UMI-FT but with richer proprioception (4 joint angles vs 1 gripper width). Demonstrate to Sunday Robotics and DeepMind that Cornelius can do this work at a high level.

**Course:** ME740 Vision, Robotics, and Planning (Spring 2026)

## Current Phase
**Phase 1: Hardware** - Designing and 3D printing the instrumented glove

### Status (2026-02-01)
- [x] Gripper mechanical design (DESIGN.md)
- [x] Electronics ordered (ESP32, AS5600 encoders, TCA9548A multiplexer)
- [x] ESP32 firmware written
- [x] Python data collection pipeline written
- [x] Diffusion policy scaffolding written
- [x] ARKit iPhone app implemented & tested (ios/StackCapture)
- [x] Python iPhone session loader (stack/data/iphone_loader.py)
- [x] Visualization tools (stack/viz/)
- [x] iPhone capture verified: 60 FPS, 0% dropped frames
- [x] Orientation confirmed correct for horizontal mount (power button up)
- [ ] **IN PROGRESS:** Printing finger joints, testing fit
- [ ] Printing palm, testing MCP joint
- [ ] Full glove assembly
- [ ] Encoder + iPhone timestamp alignment

## Key Differentiator
UMI-FT captures: **pose (7D) + gripper width (1D) = 8D**
Stack captures: **pose (7D) + 4 joint angles (4D) = 11D**

This richer proprioception could enable learning more dexterous behaviors.

## Project Structure (Updated 2026-02-01)

```
stack/
├── stack/                  # Main Python package
│   ├── data/               # Encoder communication, dataset loading, iPhone loader
│   ├── viz/                # Visualization tools
│   ├── policy/             # Diffusion policy (Chi et al.)
│   └── scripts/            # CLI: collect, train, eval
├── ios/                    # iPhone ARKit app
│   ├── PLAN.md             # App planning doc
│   └── StackCapture/       # Xcode project
├── firmware/               # ESP32 encoder reader
│   └── encoder_reader/     # Arduino sketch
├── hardware/
│   ├── DESIGN.md           # Full hardware design doc
│   ├── analysis/           # Python kinematics/physics scripts
│   ├── cad/exports/        # STL/STEP exports from Fusion
│   └── docs/               # Design diagrams (PNG)
├── configs/                # Hydra/OmegaConf training configs
├── tests/                  # pytest tests
└── data/                   # Local data (gitignored)
    ├── raw/                # Raw demo sessions
    ├── processed/          # Zarr datasets
    └── models/             # Trained checkpoints
```

## CLI Commands

```bash
# Data collection
stack-collect --session demo_01

# Training
stack-train --config configs/default.yaml

# Evaluation
stack-eval --checkpoint outputs/checkpoint_0100.pt
```

## Hardware Summary

| Component | Status |
|-----------|--------|
| AS5600 encoders (×4) | Ordered |
| TCA9548A multiplexer | Ordered |
| ESP32 (HiLetgo) | Ordered |
| iPhone 15 Pro | Have |
| PLA filament | Have |
| Bambu Lab P1S | Have (bought 2026-01-29) |
| Finger prints | In progress |

## Key Files

- `hardware/DESIGN.md` - Full gripper design documentation
- `stack/data/encoder.py` - ESP32 serial communication
- `stack/data/iphone_loader.py` - iPhone session loading
- `stack/viz/iphone_viz.py` - Session visualization tools
- `stack/policy/diffusion.py` - Diffusion policy implementation
- `firmware/encoder_reader/encoder_reader.ino` - ESP32 firmware
- `ios/StackCapture/` - iPhone ARKit data collection app (60 FPS, LiDAR depth)
- `configs/default.yaml` - Training configuration

## References

- [UMI-FT](https://github.com/real-stanford/UMI-FT) - Primary reference
- [Diffusion Policy](https://github.com/real-stanford/diffusion_policy) - Policy architecture
- [Sunday Robotics](https://sunday.ai) - Inspiration (Skill Capture Glove)

## Timeline Pressure

Cornelius graduates May 2026 and needs a job before then. Sunday Robotics rejected his application Jan 19. The UMI-FT author was just hired there. This project needs to demonstrate comparable capability.

## Working Practices

**Always update documentation:** Every conversation could be the last before memory resets. When design decisions are made, hardware changes happen, or new context is learned:
1. Update this CLAUDE.md with status changes
2. Update `hardware/DESIGN.md` with design decisions
3. Capture learnings (print settings, tolerances, what worked/failed)

Don't wait until end of session - update as we go.

## Session Log

### 2026-02-01 (Saturday)
- Implemented iOS ARKit capture app (StackCapture)
- Fixed CVPixelBuffer recycling issue (immediate JPEG encoding)
- Achieved 60 FPS capture with 0% dropped frames
- Depth is Float32 (not Float16 as initially assumed)
- USB file transfer via Finder enabled (much faster than AirDrop)
- Visualization tools created (frame browser, 3D trajectory, depth colorization)
- Confirmed image orientation correct for horizontal phone mount (power button up)
- **Blockers:** Glove assembly (hardware), GPU for training

### 2026-01-31 (Friday evening)
- Cornelius bought a Bambu Lab P1S printer (no longer need EPIC)
- Currently printing TPU finger tip pad sizing samples for press fit
- First batch of 3 samples didn't fit, second batch printing now
- Next steps: finger assembly → palm
