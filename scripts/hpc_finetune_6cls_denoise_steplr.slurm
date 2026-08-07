#!/bin/bash
#SBATCH --job-name=crism_6cls_steplr
#SBATCH --account=sbyrne
#SBATCH --partition=gpu_standard
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64gb
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/6cls_denoise_steplr_%j.log
#SBATCH --error=logs/6cls_denoise_steplr_%j.log

# Denoise-backbone 6-class FT, re-run with:
#   - stop_metric = val_mAP (default): patience and best-checkpoint track the
#     PRIMARY-mineral objective, so the saved best.pt is the best primary-
#     mineral epoch (not the best-alteration epoch the mc11val runs saved).
#   - step LR schedule: base lr 0.01 held for 10 epochs, then x0.1 every 10
#     (0.01 -> 0.001 -> 1e-4 -> 1e-5 ...). encoder_lr_scale=0.01 is preserved
#     across the decay (StepLR scales both param groups), so the encoder runs
#     at 1/100 of the head throughout: encoder 1e-4 -> 1e-5 -> 1e-6 -> 1e-7.
#     Aggressive early adaptation (to carve alteration/HCP features), then
#     anneal to consolidate the primary-mineral fit.
#
# NOTE on interpretation: "0.01" here is the BASE learning rate on the step
# schedule, with the encoder differential kept at 0.01. If the intent was to
# schedule encoder_lr_scale itself instead, this is the file to change.
#
# Same clean MC11-val data as ft_6cls_mc11val_denoise so curves are directly
# comparable.

WORK_DIR=/groups/sbyrne/phillipsm/crism_classification
DATA_ROOT=/xdisk/sbyrne/phillipsm/CRISM_MRDR          # all bulk data (xdisk)
DATA_DIR=${DATA_ROOT}/crism_classification/data
CKPT_DIR=${DATA_ROOT}/crism_classification/checkpoints
ENCODER=${CKPT_DIR}/spatial_mae_denoising_128d_6l_best.pt
CACHE_DIR=${DATA_DIR}/patch_cache_mc11val
PARQUET=${DATA_DIR}/mrral_pixels_with_review_mc11val.parquet
MTRDR_PATCHES=${DATA_DIR}/contrastive/extra_plag_roi/patches.npy
MTRDR_ROWS=${DATA_DIR}/mtrdr_plag_rows.parquet

PYTHON=/groups/sbyrne/phillipsm/micromamba/envs/crism/bin/python

cd "$WORK_DIR"
mkdir -p logs

if [ ! -f config.local.yaml ]; then
    cat > config.local.yaml <<EOF
data_root: ${DATA_ROOT}
EOF
fi

for f in "$ENCODER" "$PARQUET" "$MTRDR_PATCHES" "$MTRDR_ROWS"; do
    if [ ! -f "$f" ]; then echo "ERROR: missing $f"; exit 1; fi
done
if [ ! -d "$CACHE_DIR" ]; then
    echo "ERROR: cache dir missing: $CACHE_DIR (build via hpc_build_mc11val_data.slurm)"
    exit 1
fi

echo "=== ft_6cls_denoise_steplr: base_lr 0.01 step(/10 every 10ep), enc_scale 0.01, stop=val_mAP ==="

$PYTHON -u scripts/train.py \
    --model spatial_vit \
    --run_name ft_6cls_denoise_steplr \
    --pretrain_ckpt "$ENCODER" \
    --with_alteration \
    --mrral_parquets "$PARQUET" \
    --patch_cache_dir "$CACHE_DIR" \
    --synth_train_cache "$MTRDR_PATCHES" \
    --synth_train_parquet "$MTRDR_ROWS" \
    --lr 0.01 --encoder_lr_scale 0.01 \
    --lr_schedule step --lr_step_size 10 --lr_gamma 0.1 \
    --stop_metric val_mAP \
    --epochs 100 --patience 30 --min_delta 0.0005 \
    --batch_size 256 \
    --embed_dim 128 --n_heads 4 --n_layers 6 \
    --patch_size 7 \
    --asl_loss

echo "=== ft_6cls_denoise_steplr done ==="
