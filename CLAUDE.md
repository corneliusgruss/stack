# stack - Project Context

## What This Is
Proprioceptive data collection and diffusion policy for dexterous manipulation.

**Goal:** Build a system that rivals UMI-FT but with richer proprioception (4 joint angles vs 1 gripper width). Demonstrate to Sunday Robotics and DeepMind that Cornelius can do this work at a high level.

**Course:** ME740 Vision, Robotics, and Planning (Spring 2026)

## Current Phase
**Phase 1: Hardware** - Designing and 3D printing the instrumented glove

### Status (2026-02-17)
- [x] Gripper mechanical design (DESIGN.md)
- [x] Electronics ordered (ESP32, AS5600 encoders, TCA9548A multiplexer)
- [x] ESP32 firmware written (Serial + BLE dual output)
- [x] Python data collection pipeline written
- [x] Diffusion policy fully implemented (ResNet18 + ConditionalUnet1D, matches Chi et al.)
- [x] ARKit iPhone app implemented & tested (ios/StackCapture)
- [x] Python session loader (stack/data/iphone_loader.py) — camera-agnostic SessionLoader
- [x] Visualization tools (stack/viz/)
- [x] iPhone capture verified: 60 FPS, 0% dropped frames
- [x] Orientation confirmed correct for horizontal mount (power button up)
- [x] ESP32 board tested (ESP32-D0WD-V3, dual core, 240 MHz, 4 MB flash)
- [x] TCA9548A multiplexer verified on I2C bus at 0x70
- [x] AS5600 encoder verified on mux CH0 at 0x36 — smooth angle readings at 100 Hz
- [x] Full electronics chain validated: ESP32 → TCA9548A → AS5600 (all 4 encoders soldered + tested)
- [x] Test magnet-to-encoder fit in joint (air gap, centering, stability)
- [x] Printing finger joints, testing fit
- [x] Printing palm, testing MCP joint
- [x] StackCapture app overhauled: BLE, ultrawide, landscape, size reduction
- [x] ESP32 BLE peripheral (50 Hz encoder notifications as "StackGlove")
- [x] iOS BLE manager (auto-scan, auto-reconnect, live encoder display)
- [x] Landscape orientation (volume buttons down for recording grip)
- [x] Ultrawide camera selection (0.5x, wider FoV for hand visibility)
- [x] Image resize to 480x360 JPEG (was 1920x1440 — 9x storage reduction)
- [x] Full-res HEVC video recording (video.mov for papers/demos)
- [x] Depth maps dropped (not used by UMI-FT, biggest storage hog)
- [x] Single-point LiDAR depth (virtual ToF sensor, median of 5x5 center)
- [x] Encoder + iPhone timestamp alignment (nearest-neighbor matching)
- [x] Python loader updated for new session format (12D episodes)
- [x] Training pipeline complete: synthetic data, dataset, train loop, eval, tests (16/16 pass)
- [x] Architecture upgraded: ResNet18 visual encoder + ConditionalUnet1D (matches Diffusion Policy paper)
- [x] Flash BLE firmware to ESP32 and verify with nRF Connect
- [x] Build & deploy updated StackCapture to iPhone
- [x] Camera-agnostic capture: dual mode iOS app (ARKit 1x / Ultrawide SLAM)
- [x] RawCaptureSession: AVFoundation ultrawide capture + CMMotionManager IMU (200Hz)
- [x] Camera intrinsics extraction + calib.txt output
- [x] DROID-SLAM pipeline: CLI script + Colab notebook
- [x] Python SessionLoader: handles ARKit, ultrawide, and future stereo sessions
- [x] All 16 existing tests pass with new SessionLoader (backward compat via iPhoneSession alias)
- [ ] **IN PROGRESS:** Full glove assembly
- [ ] End-to-end test: record session with glove + load in Python
- [ ] First ultrawide capture test + DROID-SLAM processing
- [ ] Scale calibration with known object

## Key Differentiator
UMI-FT captures: **pose (7D) + gripper width (1D) = 8D**
Stack captures: **pose (7D) + 4 joint angles (4D) + 1 depth point (1D) = 12D**

This richer proprioception could enable learning more dexterous behaviors.
The depth point acts as a virtual ToF sensor (like Sunday Robotics' Skill Capture Glove) — for free via iPhone LiDAR.

## Project Structure (Updated 2026-02-17)

```
stack/
├── stack/                  # Main Python package
│   ├── data/               # Session loading, encoder comm, dataset, synthetic data
│   ├── viz/                # Visualization tools
│   ├── policy/             # Diffusion policy (Chi et al.)
│   └── scripts/            # CLI: collect, train, eval, run_slam
├── ios/                    # iPhone capture app (dual mode: ARKit + ultrawide)
│   ├── PLAN.md             # App planning doc
│   └── StackCapture/       # Xcode project
├── firmware/               # ESP32 encoder reader
│   └── encoder_reader/     # Arduino sketch
├── hardware/
│   ├── DESIGN.md           # Full hardware design doc
│   ├── analysis/           # Python kinematics/physics scripts
│   ├── cad/exports/        # STL/STEP exports from Fusion
│   └── docs/               # Design diagrams (PNG)
├── notebooks/              # Jupyter/Colab notebooks
│   ├── train_colab.ipynb   # Training on Colab
│   └── run_slam.ipynb      # DROID-SLAM processing on Colab
├── docs/                   # Project documentation
│   ├── gpu_access.md       # SCC/Colab/local GPU setup guide
│   └── gruss_me740_proposal.tex  # Symlink to approved proposal
├── configs/                # Hydra/OmegaConf training configs
├── tests/                  # pytest tests
└── data/                   # Local data (gitignored)
    ├── raw/                # Raw demo sessions
    ├── processed/          # Zarr datasets
    └── models/             # Trained checkpoints
```

**Proposal:** `docs/gruss_me740_proposal.tex` (symlink → `~/workspace/docs/academic/ms_robotics_bu/me740_vision_robotics/`)

## CLI Commands

```bash
# Data collection
stack-collect --session demo_01

# Training
stack-train --config configs/default.yaml

# Evaluation
stack-eval --checkpoint outputs/checkpoint_0100.pt

# SLAM processing (ultrawide sessions)
python -m stack.scripts.run_slam --session data/raw/session_...
python -m stack.scripts.run_slam --data-dir data/raw  # batch process
```

## Hardware Summary

| Component | Status |
|-----------|--------|
| AS5600 encoders (×4) | Ordered |
| TCA9548A multiplexer | Ordered |
| ESP32 (HiLetgo) | Ordered |
| iPhone 16 Pro | Have |
| PLA filament | Have |
| Bambu Lab P1S | Have (bought 2026-01-29) |
| Finger prints | In progress |

## Key Files

- `docs/gpu_access.md` - GPU access plan (SCC, Colab, local) with draft email + SGE examples
- `docs/gruss_me740_proposal.tex` - Approved ME740 proposal (symlink)
- `hardware/DESIGN.md` - Full gripper design documentation
- `stack/data/encoder.py` - ESP32 serial communication (USB)
- `stack/data/iphone_loader.py` - Camera-agnostic SessionLoader (v3: ARKit + ultrawide + future stereo)
- `stack/viz/iphone_viz.py` - Session visualization tools
- `stack/policy/diffusion.py` - Diffusion policy (ResNet18 + ConditionalUnet1D)
- `stack/data/synthetic.py` - Synthetic data generator for pipeline testing
- `stack/data/training_dataset.py` - PyTorch Dataset with sliding window sampling + normalization
- `stack/scripts/train.py` - Training loop (EMA, gradient clip, cosine LR, MPS support)
- `stack/scripts/eval.py` - Evaluation (position/rotation/joint error metrics)
- `stack/scripts/run_slam.py` - DROID-SLAM processing for raw sessions (CLI)
- `notebooks/run_slam.ipynb` - DROID-SLAM processing (Colab notebook)
- `tests/test_training_pipeline.py` - 16 integration tests (all pass)
- `firmware/encoder_reader/encoder_reader.ino` - ESP32 firmware (Serial 100Hz + BLE 50Hz)
- `ios/StackCapture/` - iPhone capture app (dual mode: ARKit 1x / Ultrawide SLAM)
- `ios/StackCapture/.../Capture/RawCaptureSession.swift` - AVFoundation ultrawide capture + IMU
- `ios/StackCapture/.../Capture/CaptureCoordinator.swift` - Dual-mode capture orchestration
- `ios/StackCapture/.../BLE/BLEManager.swift` - CoreBluetooth manager for StackGlove
- `ios/StackCapture/.../Capture/VideoRecorder.swift` - HEVC video recorder
- `configs/default.yaml` - Training configuration

## Network Architecture

Matches Chi et al. "Diffusion Policy" (RSS 2023). Three components:

**1. Visual Encoder — ResNet18 (ImageNet pretrained)**
- Input: (B, obs_horizon, 3, 224, 224) RGB images
- Per-image: ResNet18 → 512-dim → Linear → hidden_dim
- Fused with proprio across obs_horizon via MLP → hidden_dim conditioning vector

**2. Noise Prediction — ConditionalUnet1D**
- 1D convolutions over the action sequence (temporal axis)
- 3-level UNet: channels (hidden_dim, 2x, 4x), e.g., (256, 512, 1024)
- FiLM conditioning at every residual block from obs + timestep embedding
- Skip connections between encoder/decoder at each resolution
- action_horizon=16 → downsampled to 8 → 4 → mid → upsampled back to 16

**3. Diffusion — DDPM**
- 100 steps, cosine beta schedule
- Training: predict noise ε, MSE loss
- Inference: iterative denoising from pure noise → action chunk

**Data flow:**
```
RGB images ──→ ResNet18 ──→ ┐
                             ├──→ obs_encoder ──→ obs_cond (hidden_dim)
Proprio (12D) ────────────→ ┘                         │
                                                       ↓
Noisy actions (11D × 16) ──→ ConditionalUnet1D ←── FiLM(obs_cond + timestep_emb)
                                    │
                                    ↓
                            ε_pred (11D × 16)
```

**Parameter count:** ~12M (ResNet18: 11M, UNet: ~1M, obs encoder: ~70K)

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

## Session Data Format (v3 — camera-agnostic)

```
session_YYYY-MM-DD_HHMMSS/
├── metadata.json         # Device, captureSource, slamProcessed, cameraIntrinsics
├── poses.json            # [{timestamp, rgbIndex, depth, transform}, ...] — ARKit or SLAM
├── encoders.json         # [{timestamp, esp_timestamp_ms, angles...}, ...]
├── imu.json              # [{timestamp, accel:[x,y,z], gyro:[x,y,z]}, ...] (ultrawide only)
├── calib.txt             # "fx fy cx cy" camera intrinsics (ultrawide only, for SLAM)
├── video.mov             # Full-resolution HEVC video for visualization
└── rgb/
    ├── 000000.jpg        # 480x360 JPEG quality 0.8
    └── ...
```

**metadata.json additions (v3):**
- `captureSource`: `"iphone_arkit"` | `"iphone_ultrawide"` | `"stereo_usb"`
- `slamProcessed`: `false` until DROID-SLAM fills poses.json (null for ARKit)
- `cameraIntrinsics`: `{fx, fy, cx, cy}` (scaled to 480x360)
- `imuCount`: number of IMU readings

**Capture modes:**
- **ARKit 1x**: Wide camera, ARKit provides poses + LiDAR depth. Start here.
- **Ultrawide SLAM**: 120° FOV ultrawide, poses from DROID-SLAM, IMU recorded. Better for wrist mount.

**Storage:** ~69 MB/min (was ~600 MB/min). 50 demos of 30s each = ~1.7 GB.

## Session Log

### 2026-02-17 (Monday)
- Built camera-agnostic capture pipeline (Track B from DROID-SLAM plan)
- iOS app now has dual capture modes: "ARKit 1x" and "Ultrawide SLAM" with toggle
- New `RawCaptureSession.swift`: AVFoundation ultrawide capture (120° FOV, no ARKit)
- CMMotionManager IMU recording at 200 Hz (accel + gyro → imu.json)
- Camera intrinsics extraction from AVCaptureDevice format metadata → calib.txt
- `CaptureCoordinator` updated for dual mode: handles frames from ARKit OR raw session
- `StorageManager` writes imu.json, calib.txt, captureSource, slamProcessed in metadata
- `SessionModels.swift`: added CaptureSource enum, CameraIntrinsics, IMUReading
- Created `stack/scripts/run_slam.py` (CLI) + `notebooks/run_slam.ipynb` (Colab)
- DROID-SLAM pipeline: load frames → track → global BA → scale correction → write poses.json
- Python `SessionLoader` (renamed from iPhoneSession, alias kept for backward compat)
- Loader handles captureSource, validates slamProcessed, loads imu.json + calib.txt
- All 16 existing tests pass — full backward compatibility confirmed
- **Next:** Build & deploy updated app, test ultrawide capture, first SLAM run on Colab

### 2026-02-16 (Sunday)
- Major StackCapture app overhaul: BLE, ultrawide, landscape, size reduction
- ESP32 firmware: added BLE peripheral ("StackGlove"), 50 Hz encoder notifications, calibrate command
- iOS BLEManager: auto-scan, auto-reconnect, live encoder display, recording integration
- Landscape left orientation (volume buttons down for comfortable recording grip)
- Ultrawide camera (0.5x) selection with fallback to wide
- Image resize from 1920x1440 to 480x360 (~9x storage reduction)
- Added full-res HEVC video recording (video.mov, ~12 MB/min)
- Removed full depth map saving (biggest storage hog, not used by UMI-FT)
- Added single-point LiDAR depth sampling (virtual ToF, median of 5x5 center region)
- Updated Python loader for new session format (12D episodes, encoder alignment)
- Updated diffusion policy obs_dim from 11 to 12 (+ depth point)
- **Next:** Flash BLE firmware, build updated app, end-to-end test

### 2026-02-11 (Tuesday)
- Soldered encoder to TCA9548A multiplexer, wired to ESP32
- Mini breadboard caused I2C connection failures — soldered header pins directly instead
- Flashed board test firmware: ESP32-D0WD-V3 confirmed (dual core, 240 MHz, 4 MB flash)
- I2C scan found TCA9548A at 0x70, AS5600 at 0x36 on CH0
- Flashed encoder_reader firmware: smooth 100 Hz angle readings confirmed
- Encoder readings are noisy when board/magnet aren't physically stable (expected)
- **Next:** Print magnet mount test jig to validate air gap + centering before full finger CAD
- **Learning:** Tiny breadboards unreliable for I2C — solder directly or use full-size breadboard

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
