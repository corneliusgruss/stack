#!/bin/bash -l

#$ -N stack_train_v2
#$ -pe omp 4
#$ -l gpus=1
#$ -l gpu_c=7.0
#$ -l h_rt=12:00:00
#$ -j y
#$ -o /usr3/graduate/cgruss/stack/outputs/train_v2.log
#$ -cwd

module load miniconda
module load academic-ml/spring-2026
conda activate spring-2026-pyt

pip install --user -q diffusers omegaconf wandb

export PYTHONPATH=~/stack:$PYTHONPATH
export WANDB_DIR=/usr3/graduate/cgruss/stack/outputs
export WANDB_CACHE_DIR=/usr3/graduate/cgruss/stack/outputs/.wandb_cache
cd ~/stack

echo "=== Job $JOB_ID started at $(date) ==="
echo "Host: $(hostname)"
nvidia-smi
echo "==="

# Clean up old periodic checkpoints to save disk quota (keep best + latest)
find outputs/real_v2/ -name 'checkpoint_0*.pt' -delete 2>/dev/null

# Resume from best checkpoint if available
RESUME_FLAG=""
if [ -f outputs/real_v2/checkpoint_best.pt ]; then
    RESUME_FLAG="--resume outputs/real_v2/checkpoint_best.pt"
    echo "Resuming from checkpoint_best.pt"
fi

python -m stack.scripts.train \
    --config configs/default.yaml \
    --data-dir data/packed \
    --output-dir outputs/real_v2 \
    --device cuda \
    --wandb \
    $RESUME_FLAG

echo "=== Job finished at $(date) ==="
