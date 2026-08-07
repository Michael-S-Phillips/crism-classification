#!/bin/bash
#SBATCH --job-name=spvit_lr_ablation_v4
#SBATCH --account=sbyrne
#SBATCH --partition=gpu_standard
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32gb
#SBATCH --time=0-12:00:00
#SBATCH --array=0-3
#SBATCH --output=logs/lr_ablation_v4_%a_%j.out
#SBATCH --error=logs/lr_ablation_v4_%a_%j.err

# v4: re-run the lr-scale ablation against the supplementary-data parquet
# (mrral_pixels.parquet at 2.42M pixels, includes 10 sup tiles + full Hellas).
# Same configs as scripts/hpc_ablation_lr_scale.slurm, with _v4 run names so
# the original checkpoints aren't overwritten.

# ── Paths ─────────────────────────────────────────────────────────────────────
WORK_DIR=/groups/sbyrne/phillipsm/crism_classification
DATA_ROOT=/xdisk/sbyrne/phillipsm/CRISM_MRDR          # all bulk data (xdisk)
DATA_DIR=${DATA_ROOT}/crism_classification/data
CKPT_DIR=${DATA_ROOT}/crism_classification/checkpoints
PRETRAIN_CKPT=${CKPT_DIR}/spatial_mae_128d_6l_best.pt
CACHE_DIR=${DATA_DIR}/patch_cache

# ── Condition arrays (task ID 0-3) ───────────────────────────────────────────
RUN_NAMES=("spvit_frozen_v4" "spvit_lrscale0001_v4" "spvit_lrscale001_v4" "spvit_lrscale01_v4")
# Intentionally unquoted on expansion so multi-word values word-split into separate argv tokens.
LR_SCALE_ARGS=("--freeze_encoder" "--encoder_lr_scale 0.001" "--encoder_lr_scale 0.01" "--encoder_lr_scale 0.1")

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
echo "=== LR scale ablation v4: ${RUN_NAMES[$SLURM_ARRAY_TASK_ID]} (task $SLURM_ARRAY_TASK_ID) ==="
${PYTHON} -u scripts/train.py \
    --model spatial_vit \
    --run_name "${RUN_NAMES[$SLURM_ARRAY_TASK_ID]}" \
    --epochs 50 \
    --patience 10 \
    --batch_size 256 \
    --lr 5e-4 \
    --embed_dim 128 \
    --n_heads 4 \
    --n_layers 6 \
    --patch_size 7 \
    --asl_loss \
    --pretrain_ckpt "${PRETRAIN_CKPT}" \
    ${LR_SCALE_ARGS[$SLURM_ARRAY_TASK_ID]}
