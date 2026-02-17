# Glove Build Tracker

**Rule: Only work on one part at a time. Print it, test it, check it off, move on.**

---

## Build Order

### Phase 1: Index Finger
- [x] Encoder cap (AS5600 housing)
- [x] Index proximal phalanx
- [x] Index distal phalanx
- [ ] PIP joint — reprinting, iterating on fit
- [ ] MCP joint — reprinting, iterating on fit
- [ ] Magnet mounting in joint — needs to stay fixed
- [ ] Index fingertip (TPU — print while soldering encoders)

### Phase 2: Palm + Thumb
- [x] Palm base plate (v1 done, iterating on tolerances)
- [ ] Thumb body (fixed)
- [ ] Wrist strap slot (for velcro attachment)

### Phase 3: 3-Finger Unit
- [ ] 3-finger proximal (wider copy of index)
- [ ] 3-finger distal
- [ ] 3-finger PIP + MCP joints
- [ ] 3-finger fingertip (TPU — print while soldering)

### Phase 4: iPhone Mount + Wearability
- [ ] iPhone holder bracket
- [x] Velcro strap ordered (arriving today)

### Electronics
- [x] Solder wires to AS5600 board #1
- [x] Solder headers onto TCA9548A multiplexer
- [x] Wire ESP32 → TCA9548A → AS5600
- [x] Flash firmware, verify encoder reads angles
- [ ] Solder + wire remaining 3 encoders
- [ ] Mount electronics onto glove

---

## Current Status (Feb 13)

> **Reprinting and iterating** on part fits — joints, palm tolerances, magnet mount. Goal: get all PLA parts to assemble without failing. Then TPU pads while soldering remaining encoders.

---

## Current Iteration Issues

| Issue | Fix | Status |
|-------|-----|--------|
| Magnet moves in joint | Tighter press-fit or glue pocket | Iterating |
| Palm bends when fingers inserted | Loosen part tolerances | Iterating |
| PIP joint stiff | 9mm hole, sand pin | Iterating |
| Glove falls off hand (iPhone weight) | Velcro strap + slot in palm | Strap ordered |
| Parts breaking during assembly | Reprinting with better fits | Iterating |

---

## Parts Inventory

| File | Sub-assembly | Status | Notes |
|------|-------------|--------|-------|
| encoder-cap.3mf | Index / 3-finger | Done | Friction fit 12.3×5.2 into 12.2×5.1 hole |
| index-proximal.3mf | Index finger | Iterating | Reprinting for fit |
| index-distal.3mf | Index finger | Iterating | Reprinting for fit |

---

## Learnings

| Date | Part | Result | Lesson |
|------|------|--------|--------|
| 2026-02-01 | Snap-fit joint | Success | Ridge OD 8mm, wall ID 8.4mm, cantilever 1.5mm, catch 1mm |
| 2026-01-31 | TPU press-fit pads | Failed | TPU compresses more than expected, need looser fit |
| 2026-02-07 | Encoder cap cantilever | Failed | PLA flex across layer lines too brittle, switched to friction fit |
| 2026-02-07 | Encoder cap friction fit | Success | 0.1mm interference per side for PLA-on-PLA |
| 2026-02-11 | Electronics | Success | ESP32 + TCA9548A + AS5600 reads angles, firmware works |
| 2026-02-11 | Palm v1 | Bending | Loosen part tolerances rather than reinforce palm |
| 2026-02-11 | Magnet mount | Too loose | Magnet moves in joint, needs tighter fit or pocket |
| 2026-02-13 | iPhone mount | Weight issue | iPhone shifts COM, glove falls off — need wrist strap |

---

## Print Settings Reference

- **PLA:** 0.2mm layers, 20% infill, 3 walls
- **TPU 95A:** 0.2mm layers, 100% infill, slow speed
- **Joint clearance:** Hole = 9mm, Pin = 8.0mm (1.0mm gap) + sand pin
- **Friction fit (PLA-on-PLA):** 0.1mm interference per side
