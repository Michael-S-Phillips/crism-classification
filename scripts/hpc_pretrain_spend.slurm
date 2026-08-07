#!/bin/bash
#SBATCH --job-name=spatial_mae_spend
#SBATCH --account=sbyrne
#SBATCH --partition=gpu_standard
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64gb
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/pretrain_spend_%j.log
#SBATCH --error=logs/pretrain_spend_%j.log

# SPEND-style spatial-spectral MAE pre-training (v4).
# Single GPU, 200 epochs, ~32-hour wall time expected. 48-hour budget.
# Designed to run CONCURRENTLY with the v3 denoising job — separate
# job name, separate output log, same data root.
# Spec: docs/superpowers/specs/2026-05-16-spend-spatial-mae-design.md

WORK_DIR=/groups/sbyrne/phillipsm/crism_classification
DATA_ROOT=/xdisk/sbyrne/phillipsm/CRISM_MRDR          # all bulk data (xdisk)
DATA_DIR=${DATA_ROOT}/crism_classification/data
PYTHON=/groups/sbyrne/phillipsm/micromamba/envs/crism/bin/python

cd "$WORK_DIR"

# Make sure config.local.yaml points at the right tile dir
if [ ! -f config.local.yaml ]; then
    cat > config.local.yaml <<EOF
data_root: ${DATA_ROOT}
EOF
fi

mkdir -p logs

echo "=== SPEND MAE pre-training start: $(date) ==="

${PYTHON} -u scripts/pretrain_spatial_mae_spend.py \
    --epochs 200 \
    --warmup 10 \
    --batch_size 1024 \
    --patches_per_epoch 200000 \
    --num_workers 6 \
    --embed_dim 128 \
    --n_heads 4 \
    --n_layers 6 \
    --decoder_dim 64 \
    --decoder_layers 2 \
    --mask_ratio 0.75 \
    --spectral_mask_ratio 0.5 \
    --anneal_start_epoch 161 \
    --anneal_end_epoch 181 \
    --run_name spatial_mae_spend_128d_6l

echo "=== SPEND MAE pre-training end: $(date) ==="
