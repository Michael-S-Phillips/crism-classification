#!/bin/bash
#SBATCH --job-name=crism_6cls_fresh
#SBATCH --account=sbyrne
#SBATCH --partition=gpu_standard
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64gb
#SBATCH --time=5-00:00:00
#SBATCH --array=0-2
#SBATCH --output=logs/6cls_fresh_%a_%j.log
#SBATCH --error=logs/6cls_fresh_%a_%j.log

# 6-class FT from FRESH (never fine-tuned) MAE encoders, with a long
# training budget.
#
# Rationale: both warm-start sweeps (champion + cont1 backbones) begin
# from encoders already specialized by supervised mafic/plag training —
# their epoch-1 peak + slow-decay pattern suggests the representation
# resists moving toward alteration. A raw pretrain encoder has no such
# specialization; with a long budget (300 epochs / patience 75) it can
# learn the 6-class task from scratch on equal footing per class.
#
# Arms:
#   0: plag_aware_mae @ lr_scale 0.001  (best pretrain encoder; the
#      champion's own starting point, conventional scale)
#   1: plag_aware_mae @ lr_scale 0.01   (more encoder adaptation)
#   2: spend_mae      @ lr_scale 0.01   (v4 SPEND — documented to
#      outperform the denoising MAE on alteration pixels; its smoothing
#      bias hurts sharp mafic absorptions but favors the broad
#      hydration features that define alteration)
#
# These use --pretrain_ckpt (MAE ckpts carry encoder_state), unlike the
# warm-start sweeps which use --init_ckpt. Heads are fresh Linear(128,6).
# Same v2-only data as all other 6-class runs for comparability.

WORK_DIR=/groups/sbyrne/phillipsm/crism_classification
CKPT_DIR=${WORK_DIR}/checkpoints
CACHE_DIR=${WORK_DIR}/data/patch_cache_v2_only
PARQUET=${WORK_DIR}/data/mrral_pixels_with_review_v2.parquet
MTRDR_PATCHES=${WORK_DIR}/data/contrastive/extra_plag_roi/patches.npy
MTRDR_ROWS=${WORK_DIR}/data/mtrdr_plag_rows.parquet

RUN_NAMES=("ft_6cls_fresh_plagmae_ls0001"
           "ft_6cls_fresh_plagmae_ls001"
           "ft_6cls_fresh_spend_ls001")
ENCODERS=("${CKPT_DIR}/plag_aware_mae_128d_6l_best.pt"
          "${CKPT_DIR}/plag_aware_mae_128d_6l_best.pt"
          "${CKPT_DIR}/spatial_mae_spend_128d_6l_best.pt")
LR_SCALES=("0.001" "0.01" "0.01")

RUN_NAME=${RUN_NAMES[$SLURM_ARRAY_TASK_ID]}
ENCODER=${ENCODERS[$SLURM_ARRAY_TASK_ID]}
LR_SCALE=${LR_SCALES[$SLURM_ARRAY_TASK_ID]}

PYTHON=/groups/sbyrne/phillipsm/micromamba/envs/crism/bin/python

cd "$WORK_DIR"
mkdir -p logs checkpoints

if [ ! -f config.local.yaml ]; then
    cat > config.local.yaml <<EOF
data_root: /xdisk/sbyrne/phillipsm/CRISM_MRDR
checkpoint_dir: ${CKPT_DIR}
checkpoints_dir: ${CKPT_DIR}
output_dir: ${WORK_DIR}/data
patch_cache_dir: ${WORK_DIR}/data/patch_cache
EOF
fi

for f in "$ENCODER" "$PARQUET" "$MTRDR_PATCHES" "$MTRDR_ROWS"; do
    if [ ! -f "$f" ]; then echo "ERROR: missing $f"; exit 1; fi
done
if [ ! -d "$CACHE_DIR" ]; then
    echo "ERROR: cache dir missing: $CACHE_DIR (build via hpc_build_cache_v2_only.slurm)"
    exit 1
fi

echo "=== arm $SLURM_ARRAY_TASK_ID: $RUN_NAME  (encoder=$(basename $ENCODER), scale=$LR_SCALE) ==="

$PYTHON -u scripts/train.py \
    --model spatial_vit \
    --run_name "$RUN_NAME" \
    --pretrain_ckpt "$ENCODER" \
    --with_alteration \
    --mrral_parquets "$PARQUET" \
    --patch_cache_dir "$CACHE_DIR" \
    --synth_train_cache "$MTRDR_PATCHES" \
    --synth_train_parquet "$MTRDR_ROWS" \
    --lr 5e-4 --encoder_lr_scale "$LR_SCALE" \
    --epochs 300 --patience 75 --min_delta 0.0005 \
    --batch_size 256 \
    --embed_dim 128 --n_heads 4 --n_layers 6 \
    --patch_size 7 \
    --asl_loss

echo "=== $RUN_NAME done ==="
