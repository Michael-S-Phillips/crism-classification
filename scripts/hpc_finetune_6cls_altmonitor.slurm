#!/bin/bash
#SBATCH --job-name=crism_6cls_altmon
#SBATCH --account=sbyrne
#SBATCH --partition=gpu_standard
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64gb
#SBATCH --time=5-00:00:00
#SBATCH --array=0-1
#SBATCH --output=logs/6cls_altmon_%a_%j.log
#SBATCH --error=logs/6cls_altmon_%a_%j.log

# Alteration-monitored re-run of the two best fresh-encoder arms.
#
# Diagnosis of the fresh-encoder sweep: val_mAP peaks ~epoch 8 and the
# val_mAP patience (75) then kills the run ~epoch 83 — while
# val_AP_alteration is still climbing (denoise 0.049 -> 0.089) and
# train_loss is still falling (0.080 -> 0.016). The early-stop metric
# (val_mAP, dominated by the 5 easy classes) is fighting the actual
# goal (alteration). Fix: monitor and best-checkpoint on
# val_AP_alteration directly, so the run trains as long as ALTERATION
# improves and stops when IT plateaus.
#
# val_AP_alteration generalizes (train alteration = MC11 review pixels,
# val alteration = Argyre/Hellas/Nili) so it rewards real learning, not
# the memorization that a raw train_loss stop criterion would chase.
#
# Two best alteration backbones from the fresh sweep, both lr_scale 0.01
# (the scale that consistently beat 0.001 on alteration — encoder-limited):
#   0: v3 denoising MAE  (fresh-sweep alt max 0.0886 — best)
#   1: SPEND MAE         (fresh-sweep alt max 0.0855 — 2nd; smoothing
#                         bias favors broad hydration bands)
#
# Same v2 data + cache as the fresh sweep so alteration AP is comparable.

WORK_DIR=/groups/sbyrne/phillipsm/crism_classification
CKPT_DIR=${WORK_DIR}/checkpoints
CACHE_DIR=${WORK_DIR}/data/patch_cache_v2_only
PARQUET=${WORK_DIR}/data/mrral_pixels_with_review_v2.parquet
MTRDR_PATCHES=${WORK_DIR}/data/contrastive/extra_plag_roi/patches.npy
MTRDR_ROWS=${WORK_DIR}/data/mtrdr_plag_rows.parquet

RUN_NAMES=("ft_6cls_altmon_denoise" "ft_6cls_altmon_spend")
ENCODERS=("${CKPT_DIR}/spatial_mae_denoising_128d_6l_best.pt"
          "${CKPT_DIR}/spatial_mae_spend_128d_6l_best.pt")

RUN_NAME=${RUN_NAMES[$SLURM_ARRAY_TASK_ID]}
ENCODER=${ENCODERS[$SLURM_ARRAY_TASK_ID]}

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

echo "=== arm $SLURM_ARRAY_TASK_ID: $RUN_NAME (monitor=val_AP_alteration, encoder=$(basename $ENCODER)) ==="

$PYTHON -u scripts/train.py \
    --model spatial_vit \
    --run_name "$RUN_NAME" \
    --pretrain_ckpt "$ENCODER" \
    --with_alteration \
    --mrral_parquets "$PARQUET" \
    --patch_cache_dir "$CACHE_DIR" \
    --synth_train_cache "$MTRDR_PATCHES" \
    --synth_train_parquet "$MTRDR_ROWS" \
    --lr 5e-4 --encoder_lr_scale 0.01 \
    --stop_metric val_AP_alteration \
    --epochs 300 --patience 40 --min_delta 0.0005 \
    --batch_size 256 \
    --embed_dim 128 --n_heads 4 --n_layers 6 \
    --patch_size 7 \
    --asl_loss

echo "=== $RUN_NAME done ==="
