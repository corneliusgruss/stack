# ARKit Data Collection App - Planning Doc

**Created:** 2026-01-31
**Status:** Implemented

---

## Goal

Capture synchronized pose, RGB, and depth data from iPhone 15 Pro for demonstration collection. Data will be combined with encoder readings (from ESP32) during post-processing.

---

## Architecture: Record Locally, Transfer After

```
┌─────────────────────────────────────────────────────────────┐
│                     iPhone 15 Pro                           │
│                                                             │
│   ┌─────────────┐    ┌─────────────┐    ┌──────────────┐   │
│   │   ARKit     │    │   Camera    │    │    LiDAR     │   │
│   │  (6DoF pose)│    │   (RGB)     │    │   (depth)    │   │
│   └──────┬──────┘    └──────┬──────┘    └──────┬───────┘   │
│          │                  │                   │           │
│          └──────────────────┼───────────────────┘           │
│                             ▼                               │
│                    ┌────────────────┐                       │
│                    │  Local Storage │                       │
│                    │  (per session) │                       │
│                    └────────────────┘                       │
│                             │                               │
└─────────────────────────────┼───────────────────────────────┘
                              │ AirDrop / USB
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                        Laptop                               │
│                                                             │
│   ┌──────────────┐         ┌──────────────┐                │
│   │ iPhone data  │         │ Encoder data │                │
│   │ (pose, RGB,  │         │ (from ESP32) │                │
│   │  depth, ts)  │         │              │                │
│   └──────┬───────┘         └──────┬───────┘                │
│          │                        │                         │
│          └────────────┬───────────┘                         │
│                       ▼                                     │
│              ┌────────────────┐                             │
│              │ Post-processing│                             │
│              │ (align by ts)  │                             │
│              └───────┬────────┘                             │
│                      ▼                                      │
│              ┌────────────────┐                             │
│              │  Zarr dataset  │                             │
│              └────────────────┘                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Data Capture Specs

| Data | Source | Rate | Format |
|------|--------|------|--------|
| Pose (6DoF) | ARKit world tracking | 60 Hz | 4x4 transform matrix |
| RGB | Main camera | 60 Hz | JPEG @ 0.85 quality |
| Depth | LiDAR (smoothed) | 30 Hz | Float32 (meters) |
| Timestamp | ARKit frame timestamp | - | Double (seconds) |

---

## File Structure (per session)

```
session_YYYY-MM-DD_HHMMSS/
├── metadata.json          # Session info, device, start time
├── poses.json             # Array of {timestamp, rgbIndex, depthIndex, transform}
├── rgb/
│   ├── 000000.jpg
│   ├── 000001.jpg
│   └── ...
└── depth/
    ├── 000000.bin         # Float32 raw depth (meters)
    ├── 000001.bin
    └── ...
```

---

## iOS App Implementation

### App Structure

```
ios/StackCapture/
├── StackCaptureApp.swift           # Entry point
├── Views/
│   ├── CaptureView.swift           # Main recording UI + AR preview
│   ├── SessionListView.swift       # Browse saved sessions
│   └── SessionDetailView.swift     # Session info + export
├── AR/
│   ├── ARSessionManager.swift      # ARKit lifecycle
│   └── ARSessionCoordinator.swift  # Frame capture delegate
├── Capture/
│   ├── CaptureCoordinator.swift    # Orchestrates recording
│   └── StorageManager.swift        # Async file I/O
├── Models/
│   └── SessionModels.swift         # PoseFrame, Metadata, etc.
└── Export/
    └── ShareSheet.swift            # UIActivityViewController wrapper
```

### Features
- [x] Start/stop recording button
- [x] ARKit world tracking session (60 fps)
- [x] Capture RGB frames at 60 Hz
- [x] Capture depth frames at 30 Hz (every other frame)
- [x] Save pose with each RGB frame
- [x] Timestamp everything with ARKit frame time
- [x] Live preview of camera
- [x] Session browser with metadata display
- [x] Export session folder (share sheet → AirDrop)
- [x] Tracking state indicator

---

## Python Loading

Use `stack.data.iphone_loader` to load sessions:

```python
from stack.data import load_session, verify_session

# Verify a session
verify_session("session_2026-01-31_153045")

# Load and use
session = load_session("session_2026-01-31_153045")
print(session.summary())

# Get data
poses = session.get_all_poses_7d()  # (T, 7) [x, y, z, qx, qy, qz, qw]
rgb = session.get_rgb_frame(0)       # (H, W, 3) uint8
depth = session.get_depth_frame(0)   # (H, W) float32 meters
timestamps = session.get_timestamps() # (T,) float64
```

---

## Timestamp Synchronization Strategy

**Problem:** iPhone and ESP32 have different clocks.

**Solution:**
1. At session start, both devices note current time (Unix timestamp)
2. iPhone uses ARKit frame timestamps (relative to session start)
3. ESP32 uses millis() (relative to session start)
4. Python aligns using the session start times + relative offsets

**Alternative (simpler):**
- Start both recordings manually at same moment (good enough for demos)
- Post-processing can fine-tune alignment using motion correlation

---

## Next Steps

1. Build & run on iPhone 15 Pro
2. Record test session, verify data integrity
3. AirDrop to laptop, run `verify_session()`
4. Integrate with encoder data collection for aligned demos

---

## References

- [ARKit Documentation](https://developer.apple.com/documentation/arkit)
- [UMI-FT Paper](https://umi-ft.github.io/)
