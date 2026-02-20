# Stack — Training Results v2

**Date:** 2026-02-19
**Author:** Cornelius Gruss
**Project:** Proprioceptive Diffusion Policy for Dexterous Manipulation (ME740)

## Summary

Trained a diffusion policy (Chi et al., RSS 2023) on 35 hand-demonstrated pick-and-place sessions captured with a custom instrumented glove (4 AS5600 magnetic encoders) and iPhone ultrawide camera. The policy predicts 16-step action chunks (7D pose + 4 joint angles = 11D) from 2-frame observation windows (RGB images + proprioception).

**Key result:** 3.3 cm position error, 6.6 deg rotation error, and 5.4 deg joint angle error on 7 held-out validation sessions — from only 35 demonstrations and 30 epochs of training.

## What Makes This Different

Most imitation learning systems (e.g., UMI, UMI-FT) capture **pose (7D) + gripper width (1D) = 8D**. Our system captures **pose (7D) + 4 joint angles (4D) = 11D** — richer proprioception that could enable learning more dexterous behaviors. The 4 joint angles come from magnetic encoders mounted on an instrumented glove, transmitted wirelessly via BLE to the iPhone during recording.

## Architecture

| Component | Details |
|-----------|---------|
| Visual encoder | ResNet18 (ImageNet pretrained, frozen) |
| Noise prediction | ConditionalUnet1D (64/128/256 channels) |
| Diffusion | DDPM, 100 steps, cosine beta schedule |
| Observation | 2-frame window: RGB (224x224) + 11D proprio |
| Action | 16-step chunks, 11D (pose + joints) |
| Total params | 17.5M (6.4M trainable, backbone frozen) |

## Training Setup

| Setting | Value |
|---------|-------|
| Dataset | 35 sessions, 28 train / 7 val |
| Training samples | 43,754 windows |
| Batch size | 64 |
| Optimizer | AdamW, cosine LR from 1e-4 |
| Augmentation | Random crop (1.07x) + color jitter |
| Backbone | ResNet18 frozen (only UNet + obs encoder trained) |
| Hardware | NVIDIA V100 on BU SCC |
| Training time | ~70 min (30 epochs x 2.3 min/epoch) |

## Results (Validation Set, Epoch 30)

### Aggregate Metrics

| Metric | Value |
|--------|-------|
| Position error (MAE) | **0.033 m (3.3 cm)** |
| Rotation error | **6.6 deg** |
| Joint angle error | **5.4 deg** |
| Validation loss | 0.0174 |

### Per-Session Breakdown

| Session | Position (m) | Rotation (deg) | Joints (deg) |
|---------|-------------|---------------|-------------|
| session_152453 | 0.031 | 4.0 | 4.0 |
| session_152535 | 0.030 | 4.7 | 2.9 |
| session_152705 | 0.040 | 16.5 | 4.3 |
| session_153001 | 0.033 | 6.1 | 10.0 |
| session_153515 | 0.033 | 4.8 | 2.8 |
| session_155214 | 0.030 | 4.5 | 3.8 |
| session_155253 | 0.035 | 5.6 | 10.4 |

Most sessions achieve 4-6 deg rotation and 3-4 deg joint error. One outlier (session_152705) at 16.5 deg rotation — likely an unusual hand motion in that demo.

### Training Progression

| Version | Demos | Trainable Params | Best Val Loss | Position | Rotation | Joints | Notes |
|---------|-------|-----------------|--------------|----------|----------|--------|-------|
| v1 | 17 (3 encoders) | 101M | 0.079 (epoch 5) | N/A (no scale) | 42.9 deg | 20.7 deg | Overfit, no scale calibration |
| **v2** | **35 (4 encoders)** | **6.4M** | **0.017 (epoch 30)** | **3.3 cm** | **6.6 deg** | **5.4 deg** | Frozen backbone, augmentation |

Key improvements from v1 → v2:
- 2x more demos (35 vs 17), all with 4 working encoders
- Frozen ResNet18 backbone (6.4M vs 101M trainable params)
- Image augmentation (random crop + color jitter)
- IMU-based scale calibration (positions now in meters)
- Right-sized model (hidden_dim 64 vs 256)

## Data Collection

- **Device:** iPhone 16 Pro ultrawide camera (120 deg FOV, 60 FPS, 480x360)
- **Encoders:** 4x AS5600 magnetic encoders on custom 3D-printed glove, ESP32 via BLE at 50 Hz
- **Pose estimation:** COLMAP SfM (offline, ~10 min/session on MacBook)
- **Scale calibration:** IMU-based (double-integrated accelerometer vs COLMAP displacement)
- **Task:** Pick-and-place 3D printed parts into box, varied surfaces and locations
- **Session length:** ~25-30 seconds each

## Artifacts

- Dashboard: `outputs/real_v2/eval/dashboard.png`
- Per-session plots: `outputs/real_v2/eval/session_*/`
- Metrics JSON: `outputs/real_v2/eval/metrics.json`
- Checkpoint: `outputs/real_v2/checkpoint_best.pt`
- Training log: `outputs/real_v2/train_v2.log`

## Next Steps

1. Resume training past epoch 30 (was improving when disk quota crashed it)
2. Collect 15+ more demos to reach 50+ total
3. Retrain on full 50+ dataset
4. Target: <2 cm position, <4 deg rotation, <3 deg joints
