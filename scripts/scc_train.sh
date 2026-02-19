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

pip install --user -q diffusers

export PYTHONPATH=~/stack:$PYTHONPATH
cd ~/stack

echo "=== Job $JOB_ID started at $(date) ==="
echo "Host: $(hostname)"
nvidia-smi
echo "==="

python -m stack.scripts.train \
    --config configs/default.yaml \
    --data-dir data/raw \
    --output-dir outputs/real_v2 \
    --device cuda

echo "=== Job finished at $(date) ==="
