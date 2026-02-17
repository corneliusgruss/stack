# GPU Access Plan

## Status: Pending
**Last updated:** 2026-02-16

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

### Draft Email to Prof. Baillieul

> **Subject:** SCC GPU Access for ME740 Project
>
> Hi Professor Baillieul,
>
> My project proposal (Proprioceptive Data Collection for Diffusion-Based Manipulation Learning) was approved — thank you! Training the diffusion policy will require GPU compute.
>
> Does ME740 have an SCC course project I can be added to? If not, would you be able to create one? Alternatively, if there's another way you'd recommend I get access, I'm happy to follow that path.
>
> Thanks,
> Cornelius

### Backup: Email help@scc.bu.edu

Request a trial account mentioning ME740 course project + GPU needs. Gets you limited access in 1-3 days.

### Once You Have Access

```bash
# SSH in
ssh cgruss@scc1.bu.edu

# Check GPU availability
qgpus

# Interactive GPU session (debugging)
qrsh -l gpus=1 -l gpu_c=7.0 -pe omp 4

# Load pre-built PyTorch environment
module load miniconda
module load academic-ml/fall-2025
conda activate fall-2025-pyt
```

### Example Batch Script

```bash
#!/bin/bash -l

#$ -P me740_project    # Replace with actual project name
#$ -N diffusion_train
#$ -pe omp 4
#$ -l gpus=1
#$ -l gpu_c=7.0        # Gets A40, A6000, A100, or H200
#$ -l h_rt=12:00:00    # 12 hours (max 48h for GPU jobs)
#$ -j y

module load miniconda
module load academic-ml/fall-2025
conda activate fall-2025-pyt

cd /projectnb/me740_project/cgruss/stack
python -m stack.scripts.train --config configs/default.yaml
```

Submit: `qsub train.sh`

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

## Option 2: Google Colab Pro (Immediate)

- BU education license available
- A100 GPU runtimes
- Good for first training runs while waiting for SCC
- Session timeout after ~12-24h (even Pro)
- Use for validating the pipeline works end-to-end

## Option 3: MacBook Pro (Local Dev)

- MPS (Apple Silicon) — limited PyTorch op support
- Use for development, debugging, tiny test runs (1-2 demos, few epochs)
- NOT for real training

## Links

- [SCC Getting an Account](https://www.bu.edu/tech/support/research/account-management/getting-an-scc-account/)
- [SCC GPU Computing](https://www.bu.edu/tech/support/research/software-and-programming/gpu-computing/)
- [SCC Batch Scripts](https://www.bu.edu/tech/support/research/system-usage/running-jobs/batch-script-examples/)
- [SCC OnDemand Portal](https://scc-ondemand.bu.edu)
- SCC Help: help@scc.bu.edu
