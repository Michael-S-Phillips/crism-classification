#!/bin/bash
#SBATCH --job-name=spatial_mae_cont
#SBATCH --account=sbyrne
#SBATCH --partition=gpu_standard
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64gb
#SBATCH --time=4:00:00
#SBATCH --output=logs/pretrain_continuation_%j.log
#SBATCH --error=logs/pretrain_continuation_%j.log

# Continuation pre-training: resume the March plain MAE checkpoint and train
# for another ~200 epochs on the new global patch cache. The original March
# checkpoint is preserved at spatial_mae_128d_6l_march_best.pt. This run
# writes to spatial_mae_128d_6l_cont_best.pt so neither original is touched.
#
# Expected wall time ~50 min on the cache; 4 hr budget for safety.
# Spec context: docs/superpowers/specs/2026-05-18-pretrain-patch-cache-design.md

WORK_DIR=/groups/sbyrne/phillipsm/crism_classification
PYTHON=/groups/sbyrne/phillipsm/micromamba/envs/crism/bin/python

cd "$WORK_DIR"

mkdir -p logs checkpoints

echo "=== Continuation pretrain start: $(date) ==="

# LR strategy informed by the original March run's wandb history:
#   - March peaked at lr ≈ 3.0e-4 (batch 512) and decayed to ~0 by epoch 200.
#   - For the continuation we use --lr 1.0e-4 (≈1/3 of the March peak):
#     small enough to not shock the converged weights, large enough to
#     make meaningful additional progress.
#   - --warmup 10 ramps gracefully so the first continuation epoch sees
#     LR ~= 1e-5 (10% of peak), not the full base_lr.
#   - --epochs 400 with --resume at epoch 194 → train epochs 195–400
#     (about 205 more), with cosine decay completing inside the window.
${PYTHON} -u scripts/pretrain_spatial_mae.py \
    --epochs 400 \
    --warmup 10 \
    --lr 1.0e-4 \
    --batch_size 1024 \
    --patches_per_epoch 200000 \
    --num_workers 6 \
    --embed_dim 128 \
    --n_heads 4 \
    --n_layers 6 \
    --decoder_dim 64 \
    --mask_ratio 0.75 \
    --resume "${WORK_DIR}/checkpoints/spatial_mae_128d_6l_march_best.pt" \
    --run_name spatial_mae_128d_6l_cont

echo "=== Continuation pretrain end: $(date) ==="
