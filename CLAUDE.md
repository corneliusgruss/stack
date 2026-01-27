# stack - Project Context

## What This Is
ME740 course project implementing UMI-style imitation learning for cube stacking.

**Core insight:** Collect demonstrations with a handheld gripper (tracked via iPhone ARKit), train a diffusion policy, deploy on SO-101 robot arms.

## Current Phase
Phase 1: Gripper design + ARKit pipeline

## Key Decisions Made
- **Task:** Cube stacking (not French press - too much rotation complexity)
- **Tracking:** iPhone Pro + ARKit (not ORB-SLAM3 - simpler, real-time)
- **Policy:** Diffusion Policy (proven, well-documented)
- **Hardware:** SO-101 arms (cheap, LeRobot compatible)

## Architecture Overview

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Data Collection │     │  Policy Training │     │   Deployment    │
│  ────────────── │     │  ────────────── │     │  ────────────── │
│  iPhone + ARKit │────▶│  Diffusion Model │────▶│  SO-101 + IK    │
│  + Gripper      │     │  (PyTorch)       │     │  + Controller   │
└─────────────────┘     └─────────────────┘     └─────────────────┘
        │                       │                       │
        ▼                       ▼                       ▼
   data/raw/              models/                 Real robot
   - images               - checkpoints           execution
   - poses
   - gripper_state
```

## Open Questions
- [ ] Gripper-to-iPhone mounting: rigid attachment with known transform?
- [ ] ARKit coordinate frame: how to establish consistent task frame?
- [ ] Action chunking: how many future steps to predict? (UMI uses 16)
- [ ] SO-101 workspace: will it reach standard tabletop setup?

## File Conventions
- Hardware CAD: `hardware/*.step`, `hardware/*.stl`
- Training configs: `src/policy/configs/*.yaml`
- Collected demos: `data/raw/<session_id>/`
- Trained models: `models/<experiment_name>/`

## Dependencies (planned)
- Python 3.10+
- PyTorch 2.x
- diffusers (for DDPM)
- LeRobot (for SO-101 control)
- pyrealsense2 (backup if ARKit doesn't work)

## Links
- [Diffusion Policy repo](https://github.com/real-stanford/diffusion_policy)
- [UMI repo](https://github.com/real-stanford/universal_manipulation_interface)
- [LeRobot docs](https://huggingface.co/docs/lerobot)
