# stack - Project Context

## What This Is
Proprioceptive data collection and diffusion policy for dexterous manipulation.

**Goal:** Build a system that rivals UMI-FT but with richer proprioception (4 joint angles vs 1 gripper width). Demonstrate to Sunday Robotics and DeepMind that Cornelius can do this work at a high level.

**Course:** ME740 Vision, Robotics, and Planning (Spring 2026)

## Current Phase
**Phase 1: Hardware** - Designing and 3D printing the instrumented glove

### Status (2026-02-18)
- [x] Gripper mechanical design (DESIGN.md)
- [x] Electronics ordered (ESP32, AS5600 encoders, TCA9548A multiplexer)
- [x] ESP32 firmware written (Serial + BLE dual output, per-channel encoder invert)
- [x] Python data collection pipeline written
- [x] Diffusion policy fully implemented (ResNet18 + ConditionalUnet1D, matches Chi et al.)
- [x] StackCapture app: ultrawide-only (ARKit removed), BLE, landscape, IMU
- [x] Python session loader (stack/data/iphone_loader.py) — camera-agnostic SessionLoader
- [x] Visualization tools (stack/viz/)
- [x] iPhone capture verified: 60 FPS, 0% dropped frames
- [x] ESP32 board tested (ESP32-D0WD-V3, dual core, 240 MHz, 4 MB flash)
- [x] Full electronics chain validated: ESP32 → TCA9548A → AS5600 (all 4 encoders soldered + tested)
- [x] Glove assembly complete (3 encoders working — 1 AS5600 index_pip had pad rip off)
- [x] 17 demo sessions collected (pick-and-place on 4 surfaces, ~25-30s each)
- [x] COLMAP SfM pipeline: local batch processing on MacBook (replaced DROID-SLAM)
- [x] All 17 sessions processed: poses.json + calib.txt written, slamProcessed=true
- [x] COLMAP intrinsics: f≈183 at 480x360 (~120° FOV ultrawide confirmed)
- [x] Training pipeline complete: synthetic data, dataset, train loop, eval, tests (16/16 pass)
- [x] Architecture upgraded: ResNet18 visual encoder + ConditionalUnet1D (matches Diffusion Policy paper)
- [x] Sessions uploaded to Google Drive (BU school account, bu:stack_sessions/)
- [x] SCC account active (cgruss@scc1.bu.edu, group: trialscc, home: /usr3/graduate/cgruss, 10GB quota)
- [x] SCC training running: 17 sessions, 100 epochs, V100 GPU, ~3.5 min/epoch
- [x] Loss dropping fast: 0.62 → 0.03 by epoch 2
- [x] Removed dead depth dimension (obs 12D→11D, depth was always 0 with ultrawide)
- [x] IMU-based scale calibration script (stack/scripts/calibrate_scale.py)
- [x] Auto-calibration integrated into run_slam.py pipeline
- [x] Eval reports position units (meters vs COLMAP units) based on calibration status
- [ ] **NEXT:** Evaluate training results, download checkpoint
- [ ] Run scale calibration on 17 sessions: `python -m stack.scripts.calibrate_scale --data-dir data/raw`
- [ ] Replace broken AS5600 encoder (Amazon order)

## Key Differentiator
UMI-FT captures: **pose (7D) + gripper width (1D) = 8D**
Stack captures: **pose (7D) + 4 joint angles (4D) = 11D**

This richer proprioception could enable learning more dexterous behaviors.

## Project Structure (Updated 2026-02-17)

```
stack/
├── stack/                  # Main Python package
│   ├── data/               # Session loading, encoder comm, dataset, synthetic data
│   ├── viz/                # Visualization tools
│   ├── policy/             # Diffusion policy (Chi et al.)
│   └── scripts/            # CLI: collect, train, eval, run_slam
├── ios/                    # iPhone capture app (ultrawide-only, no ARKit)
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
│   └── run_slam.ipynb      # COLMAP processing on Colab (backup, prefer local)
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

## GPU Compute

**Primary: BU SCC** (Shared Computing Cluster)
- Account: `cgruss@scc1.bu.edu`, project: `trialscc` (3-month trial, expires ~May 2026)
- Scheduler: **SGE (qsub)**, NOT SLURM
- Best GPUs available: A100-80G (24 total), L40S (118 total), H200 (20 total)
- See `docs/gpu_access.md` for SGE flags, batch script examples, module loads
- Colab notebook: `notebooks/train_colab.ipynb` (backup / quick iteration)
- Local MPS: dev/debug only, not for real training

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
- `stack/scripts/run_slam.py` - COLMAP SfM processing for raw sessions (local CLI)
- `stack/scripts/calibrate_scale.py` - IMU-based scale calibration for COLMAP poses
- `notebooks/run_slam.ipynb` - COLMAP processing (Colab notebook, backup)
- `tests/test_training_pipeline.py` - 16 integration tests (all pass)
- `firmware/encoder_reader/encoder_reader.ino` - ESP32 firmware (Serial 100Hz + BLE 50Hz)
- `ios/StackCapture/` - iPhone capture app (ultrawide-only, BLE, landscape)
- `ios/StackCapture/.../Capture/RawCaptureSession.swift` - AVFoundation ultrawide capture + IMU
- `ios/StackCapture/.../Capture/CaptureCoordinator.swift` - Ultrawide capture orchestration
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
Proprio (11D) ────────────→ ┘                         │
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
- `slamProcessed`: `false` until COLMAP fills poses.json
- `cameraIntrinsics`: `{fx, fy, cx, cy}` (scaled to 480x360)
- `imuCount`: number of IMU readings

**Capture mode:** Ultrawide only (120° FOV, poses from COLMAP SfM, IMU recorded)

**Processing pipeline:**
1. Capture on iPhone (ultrawide + BLE encoders)
2. Transfer sessions to `data/raw/`
3. `python -m stack.scripts.run_slam --data-dir data/raw` (COLMAP, local MacBook)
4. Upload: `scp data/raw/*.zip cgruss@scc1.bu.edu:~/stack/data/raw/`
5. Train on BU SCC: `qsub scripts/scc_train.sh`

**Storage:** ~69 MB/min (was ~600 MB/min). 50 demos of 30s each = ~1.7 GB.

## Session Log

### 2026-02-18 (Tuesday)
- Collected 17 demo sessions with glove (pick-and-place, 5 sequences × 4 surfaces)
- Attempted DROID-SLAM on Colab: lietorch/pytorch_scatter build times out, GPU SIFT crashes
- Switched to COLMAP SfM — pip-installable, no GPU needed, runs locally
- Processed all 17 sessions locally on MacBook (~10 min each): `python -m stack.scripts.run_slam --data-dir data/raw`
- Results: 17/17 OK, 793–2970 poses per session, f≈183 (120° ultrawide confirmed)
- Pipeline: subsample 60fps→10fps, COLMAP SIFT+sequential matching+mapper, Slerp interpolation back to 60fps
- Removed ARKit mode from iOS app entirely (ultrawide-only now)
- Firmware: added per-channel encoder invert (index=false, three-finger=true)
- Google Drive upload issues: incomplete frames (1597 vs 2970), Drive random access slow for training
- Got BU SCC access (trial account, Duo MFA required for SSH)
- SCC setup: `module load academic-ml/spring-2026` has PyTorch 2.8+CUDA 12.8, `diffusers` installed via pip
- `pip install -e .` fails on SCC shared conda env — use `PYTHONPATH=~/stack:$PYTHONPATH` instead
- SGE `#$ -o` directive: `~` doesn't expand, must use full path (`/usr3/graduate/cgruss/...`)
- scp'd 17 session zips directly to SCC (1.1GB, ~2.3GB unzipped) — bypasses Drive entirely
- First training run submitted: V100 GPU, 286 batches/epoch, ~3.5 min/epoch
- Loss: 0.62 → 0.03 by epoch 2, learning rate 1e-4
- **Next:** Evaluate training results, download checkpoint, scale calibration

### 2026-02-17 (Monday)
- Built camera-agnostic capture pipeline
- iOS app: ultrawide capture (RawCaptureSession.swift) + CMMotionManager IMU at 200Hz
- Removed ARKit dependency: simpler, no dual-mode complexity
- Python SessionLoader (renamed from iPhoneSession, backward compat alias kept)
- All 16 existing tests pass
- **Next:** Collect demos, process with SLAM

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
- Updated diffusion policy obs_dim to 11 (pose 7D + joints 4D)
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
