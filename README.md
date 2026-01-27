# stack

Bimanual imitation learning for cube stacking using diffusion policy.

**Course:** ME740 Vision, Robotics, and Planning (Spring 2026)
**Goal:** Demonstrate UMI-style data collection → diffusion policy → real robot deployment

## Approach

1. **Data Collection**: Handheld gripper + iPhone Pro (ARKit for 6DoF tracking)
2. **Policy**: Diffusion Policy (Chi et al. RSS 2023)
3. **Deployment**: SO-101 follower arms (bimanual)

## Task Progression

| Phase | Task | Arms | Status |
|-------|------|------|--------|
| 1 | Pick-and-place | Single | Not started |
| 2 | 3-cube stack | Single | Not started |
| 3 | Bimanual stack (stabilize + place) | Dual | Not started |

## Project Structure

```
stack/
├── src/                    # Python code
│   ├── collection/         # ARKit data collection pipeline
│   ├── policy/             # Diffusion policy training
│   └── deployment/         # Robot control & inference
├── hardware/               # CAD files, gripper design
├── data/                   # Collected demonstrations
├── docs/                   # Design docs, notes
└── scripts/                # Utility scripts
```

## Hardware

| Component | Status | Notes |
|-----------|--------|-------|
| iPhone Pro | Have | ARKit 6DoF tracking |
| Custom gripper | Design needed | Thumb-opposition, iPhone mount |
| SO-101 arms (x2) | Order needed | ~$250 total from Seeed Studio |
| 3D printed cubes | Print needed | 40mm cubes, bright colors |

## References

- [Diffusion Policy](https://github.com/real-stanford/diffusion_policy) - Chi et al. RSS 2023
- [UMI](https://github.com/real-stanford/universal_manipulation_interface) - Chi et al. RSS 2024
- [UMI-FT](https://umi-ft.github.io/) - iPhone + ARKit approach
- [SO-101 / LeRobot](https://huggingface.co/docs/lerobot/so101) - Low-cost robot arms

## Timeline

See `docs/career/tracker.md` for weekly milestones.

---
*ME740 Spring 2026 - Cornelius Gruss*
