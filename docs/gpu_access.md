# GPU Access Plan

## Status: Active (SCC)
**Last updated:** 2026-02-19

## Overview

Training the diffusion policy requires GPU compute. The proposal already mentions SCC access (approved by Prof. Baillieul). Three options available, use in parallel.

## Option 1: BU SCC (Primary)

The BU Shared Computing Cluster has serious GPUs available for free:

| GPU | Count | VRAM | Notes |
|-----|-------|------|-------|
| H200 | 16 | 144 GB | Newest, high demand |
| A100-80G | 24 | 80 GB | Ideal for training |
| A6000 | 77 | 48 GB | Good availability |
| A40 | 68 | 48 GB | Good availability |

**Important:** SCC uses **SGE (qsub)**, not SLURM.

### How to Get Access

A faculty member must create a project and add you. Two paths:

1. **Course project** — Prof. Baillieul creates an ME740 SCC project, students get auto-added
2. **Research project** — Any faculty advisor creates a project and adds you manually

Turnaround: same-day to ~3 business days.

### Access (Active as of 2026-02-18)

```bash
# SSH in (requires Duo MFA)
ssh cgruss@scc1.bu.edu

# Home directory: /usr3/graduate/cgruss (10 GB quota!)
# Group: trialscc

# Check GPU availability
qgpus

# Interactive GPU session (debugging)
qrsh -l gpus=1 -l gpu_c=7.0 -pe omp 4

# Load pre-built PyTorch environment
module load miniconda
module load academic-ml/spring-2026
conda activate spring-2026-pyt
# PyTorch 2.8.0+cu128

# Extra deps not in academic-ml:
pip install --user diffusers omegaconf wandb

# Package: don't use pip install -e (fails on shared conda)
export PYTHONPATH=~/stack:$PYTHONPATH
```

### Data Pipeline: Local → SCC

**Problem:** SCC home quota is 10 GB. Raw sessions are ~70 MB each (480x360 JPEGs + video + IMU). 50 sessions = 3.5 GB raw, plus checkpoints at ~183 MB each — blows the quota fast.

**Solution:** Pack data locally with pre-resized 224x224 images, upload only what training needs.

#### Step 1: Process locally (MacBook)

```bash
# Run COLMAP + scale calibration on raw sessions
python -m stack.scripts.run_slam --data-dir data/raw

# Pack for SCC: resize 480x360 → 224x224, strip video.mov + imu.json
# Saves ~72% disk (2.6 GB → 717 MB for 35 sessions)
python -m stack.scripts.pack_for_scc --data-dir data/raw --output-dir data/scc_packed
```

#### Step 2: Upload to SCC

```bash
# Upload packed tars (~717 MB for 35 sessions)
scp data/scc_packed/*.tar cgruss@scc1.bu.edu:~/stack/data/packed/

# On SCC: unpack and remove tars
cd ~/stack/data/packed
for t in *.tar; do tar xf "$t"; done
rm *.tar
```

#### Step 3: Train on SCC

```bash
cd ~/stack && git pull
qsub scripts/scc_train.sh
# Monitors: tail -f ~/stack/outputs/train_v2.log
# Job status: qstat -u cgruss
```

#### Step 4: Retrieve results

```bash
# Download best checkpoint
scp cgruss@scc1.bu.edu:~/stack/outputs/real_v2/checkpoint_best.pt outputs/real_v2/

# Download training log (for loss curves)
scp cgruss@scc1.bu.edu:~/stack/outputs/train_v2.log outputs/real_v2/

# Run eval locally on MacBook (MPS, ~7 min for val split)
python -m stack.scripts.eval_viz \
    --checkpoint outputs/real_v2/checkpoint_best.pt \
    --data-dir data/raw \
    --output-dir outputs/real_v2/eval \
    --device mps --eval-stride 10 --split val
```

### Disk Budget (10 GB quota)

| Item | Size | Notes |
|------|------|-------|
| Code (~/.git + source) | ~50 MB | |
| Packed data (35 sessions) | ~750 MB | 224x224 JPEGs + JSON |
| Packed data (50 sessions) | ~1.1 GB | Target dataset |
| checkpoint_best.pt | ~183 MB | EMA + model + optimizer |
| checkpoint_periodic.pt | ~183 MB | Every 25 epochs |
| normalizer.pt | ~2 KB | |
| pip --user packages | ~200 MB | diffusers, omegaconf, wandb |
| **Total** | **~2.5 GB** | Comfortable under 10 GB |

**Key rules:**
- Never upload video.mov or imu.json to SCC
- Use `pack_for_scc.py` to pre-resize images (72% savings)
- `checkpoint_every: 25` (not 10) to limit checkpoint disk usage
- Batch script auto-deletes old periodic checkpoints on launch
- Auto-resumes from `checkpoint_best.pt` if present

### Batch Script

See `scripts/scc_train.sh`. Key features:
- Auto-resumes from `checkpoint_best.pt`
- Cleans old periodic checkpoints on launch
- 12-hour wall clock, 1 GPU (compute capability >= 7.0)
- Installs missing pip packages with `--user`

### Key SGE Flags

| Flag | Purpose | Example |
|------|---------|---------|
| `-l gpus=N` | Number of GPUs | `-l gpus=1` |
| `-l gpu_type=MODEL` | Specific GPU | `-l gpu_type=A40` |
| `-l gpu_memory=#G` | Min VRAM | `-l gpu_memory=48G` |
| `-l gpu_c=#` | Min compute capability | `-l gpu_c=7.0` |
| `-pe omp N` | CPU cores | `-pe omp 4` |
| `-l h_rt=HH:MM:SS` | Wall-clock limit | `-l h_rt=12:00:00` |

Never set `CUDA_VISIBLE_DEVICES` manually — SGE handles it.

### Gotchas

- `#$ -o` doesn't expand `~` — use full path `/usr3/graduate/cgruss/...`
- `$JOB_ID` doesn't expand in `#$ -o` directives
- `pip install -e .` fails on shared conda env — use `PYTHONPATH` instead
- Duo MFA required on every SSH connect — can't automate from scripts
- `omegaconf` not in academic-ml module — must `pip install --user`

## Option 2: Google Colab Pro (Backup)

- BU education license available
- A100 GPU runtimes
- Session timeout after ~12-24h (even Pro)
- **Not recommended for training** — Google Drive I/O is too slow for thousands of image files
- Use for quick experiments only

## Option 3: MacBook Pro (Local Dev)

- MPS (Apple Silicon) — ~0.5s per diffusion prediction
- Use for: development, debugging, evaluation (eval_viz.py), COLMAP processing
- NOT for real training (too slow)

## Links

- [SCC Getting an Account](https://www.bu.edu/tech/support/research/account-management/getting-an-scc-account/)
- [SCC GPU Computing](https://www.bu.edu/tech/support/research/software-and-programming/gpu-computing/)
- [SCC Batch Scripts](https://www.bu.edu/tech/support/research/system-usage/running-jobs/batch-script-examples/)
- [SCC OnDemand Portal](https://scc-ondemand.bu.edu)
- SCC Help: help@scc.bu.edu
