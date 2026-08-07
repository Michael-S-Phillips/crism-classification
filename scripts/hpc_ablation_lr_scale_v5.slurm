#!/bin/bash
#SBATCH --job-name=spvit_lr_ablation_v5
#SBATCH --account=sbyrne
#SBATCH --partition=gpu_standard
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32gb
#SBATCH --time=0-24:00:00
#SBATCH --array=0-3
#SBATCH --output=logs/lr_ablation_v5_%a_%j.out
#SBATCH --error=logs/lr_ablation_v5_%a_%j.err

# v5: builds on v4_fixed with three additional knobs
#   1. Per-class loss weights — boost HCP (1.5x) and plagioclase (3.0x),
#      the two classes still underperforming after stratification + label fix.
#   2. Longer training budget (epochs 50 -> 100) and looser patience
#      (10 -> 25) with a tolerance band (min_delta=0.001) — v4_fixed runs
#      were stopping in the teens with val_mAP still slowly climbing.
#   3. Same stratified-split mrral_pixels.parquet + cache as v4_fixed.

# ── Paths ─────────────────────────────────────────────────────────────────────
WORK_DIR=/groups/sbyrne/phillipsm/crism_classification
DATA_ROOT=/xdisk/sbyrne/phillipsm/CRISM_MRDR          # all bulk data (xdisk)
DATA_DIR=${DATA_ROOT}/crism_classification/data
CKPT_DIR=${DATA_ROOT}/crism_classification/checkpoints
PRETRAIN_CKPT=${CKPT_DIR}/spatial_mae_128d_6l_best.pt
CACHE_DIR=${DATA_DIR}/patch_cache

# ── Condition arrays (task ID 0-3) ───────────────────────────────────────────
RUN_NAMES=("spvit_frozen_v5" "spvit_lrscale0001_v5" "spvit_lrscale001_v5" "spvit_lrscale01_v5")
LR_SCALE_ARGS=("--freeze_encoder" "--encoder_lr_scale 0.001" "--encoder_lr_scale 0.01" "--encoder_lr_scale 0.1")

# Per-class weights, in LABEL_COLS order: olivine, lcp, hcp, plagioclase, other.
# Boost HCP (1.5x) and plagioclase (3.0x) since they remain the weakest classes.
CLASS_WEIGHTS="1.0,1.0,1.5,3.0,1.0"

# ── Python from conda env (direct path avoids activation issues in SLURM batch) ─
PYTHON=/groups/sbyrne/phillipsm/micromamba/envs/crism/bin/python

cd "$WORK_DIR"

# ── Write config.local.yaml (guard avoids concurrent-task race on array jobs) ─
if [ ! -f config.local.yaml ]; then
    cat > config.local.yaml <<EOF
data_root: ${DATA_ROOT}
EOF
fi

# ── Patch cache guard ────────────────────────────────────────────────────────
if [ ! -f "${CACHE_DIR}/mrral_train_patches_p7.npy" ]; then
    echo "ERROR: mrral_train_patches_p7.npy not found at ${CACHE_DIR}" >&2
    echo "Submit scripts/hpc_build_cache.slurm first and re-submit this with --dependency=afterok:<jobid>." >&2
    exit 1
fi

mkdir -p logs

# ── Training ─────────────────────────────────────────────────────────────────
echo "=== LR scale ablation v5: ${RUN_NAMES[$SLURM_ARRAY_TASK_ID]} (task $SLURM_ARRAY_TASK_ID) ==="
${PYTHON} -u scripts/train.py \
    --model spatial_vit \
    --run_name "${RUN_NAMES[$SLURM_ARRAY_TASK_ID]}" \
    --epochs 100 \
    --patience 25 \
    --min_delta 0.001 \
    --batch_size 256 \
    --lr 5e-4 \
    --lr_t_max 100 \
    --embed_dim 128 \
    --n_heads 4 \
    --n_layers 6 \
    --patch_size 7 \
    --asl_loss \
    --class_weights "${CLASS_WEIGHTS}" \
    --pretrain_ckpt "${PRETRAIN_CKPT}" \
    ${LR_SCALE_ARGS[$SLURM_ARRAY_TASK_ID]}
