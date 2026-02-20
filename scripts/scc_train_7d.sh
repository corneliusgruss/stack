#!/bin/bash -l

#$ -N stack_ablation_7d
#$ -pe omp 4
#$ -l gpus=1
#$ -l gpu_c=7.0
#$ -l h_rt=12:00:00
#$ -j y
#$ -o /usr3/graduate/cgruss/stack/outputs/ablation_7d.log
#$ -cwd

module load miniconda
module load academic-ml/spring-2026
conda activate spring-2026-pyt
source ~/stack/venv/bin/activate

export PYTHONPATH=~/stack:$PYTHONPATH
export WANDB_DIR=/usr3/graduate/cgruss/stack/outputs
export WANDB_CACHE_DIR=/usr3/graduate/cgruss/stack/outputs/.wandb_cache
cd ~/stack

echo "=== Job $JOB_ID started at $(date) ==="
echo "Host: $(hostname)"
echo "Ablation: 7D (pose only, no joints)"
nvidia-smi
echo "==="

# Clean up old periodic checkpoints
find outputs/ablation_7d/ -name 'checkpoint_0*.pt' -delete 2>/dev/null

# Resume from best checkpoint if available
RESUME_FLAG=""
if [ -f outputs/ablation_7d/checkpoint_best.pt ]; then
    RESUME_FLAG="--resume outputs/ablation_7d/checkpoint_best.pt"
    echo "Resuming from checkpoint_best.pt"
fi

python -m stack.scripts.train \
    --config configs/ablation_7d.yaml \
    --data-dir data/packed \
    --output-dir outputs/ablation_7d \
    --device cuda \
    --wandb \
    $RESUME_FLAG

echo "=== Job finished at $(date) ==="
