#!/bin/bash
#SBATCH --job-name=spatial_mae_denoising
#SBATCH --account=sbyrne
#SBATCH --partition=gpu_standard
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64gb
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/pretrain_denoising_%j.log
#SBATCH --error=logs/pretrain_denoising_%j.log

# Denoising spatial-spectral MAE pre-training.
# Single GPU, 200 epochs, ~32-hour wall time expected. 48-hour budget.
# Spec: docs/superpowers/specs/2026-05-16-denoising-mae-design.md

WORK_DIR=/groups/sbyrne/phillipsm/crism_classification
PYTHON=/groups/sbyrne/phillipsm/micromamba/envs/crism/bin/python

cd "$WORK_DIR"

# Make sure config.local.yaml points at the right tile dir
if [ ! -f config.local.yaml ]; then
    cat > config.local.yaml <<EOF
data_root: /xdisk/sbyrne/phillipsm/CRISM_MRDR
checkpoint_dir: ${WORK_DIR}/checkpoints
checkpoints_dir: ${WORK_DIR}/checkpoints
output_dir: ${WORK_DIR}/data
patch_cache_dir: ${WORK_DIR}/data/patch_cache
EOF
fi

mkdir -p logs checkpoints

echo "=== Denoising MAE pre-training start: $(date) ==="

${PYTHON} -u scripts/pretrain_spatial_mae_denoising.py \
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
    --sigma_gauss 0.0087 \
    --sigma_spike 0.0058 \
    --sigma_column 0.0049 \
    --spike_center_band 15 \
    --spike_fwhm_bands 3.0 \
    --run_name spatial_mae_denoising_128d_6l

echo "=== Denoising MAE pre-training end: $(date) ==="
