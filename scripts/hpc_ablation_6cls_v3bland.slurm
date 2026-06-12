#!/bin/bash
#SBATCH --job-name=crism_6cls_v3b_abl
#SBATCH --account=sbyrne
#SBATCH --partition=gpu_standard
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64gb
#SBATCH --time=0-24:00:00
#SBATCH --array=0-2
#SBATCH --output=logs/6cls_v3b_abl_%a_%j.log
#SBATCH --error=logs/6cls_v3b_abl_%a_%j.log

# 6-class LR ablation on the v3-bland (cont1) backbone.
#
# Companion to hpc_ablation_6cls_lrscale.slurm, which sweeps the champion
# (ft_review_mtrdr / plag-aware MAE) encoder. This sweep warm-starts from
# ft_bland_v3_lrscale0001_cont1_best.pt (val_mAP 0.7262) instead:
#   v3 denoising MAE -> bland FT -> cont1 continuation.
#
# Why test this backbone: it is a genuinely independent representation —
# no plag-aware pretraining and no relabel/review fine-tuning in its
# lineage — and its MC13 HCP product reviewed cleanest of all deployed
# models (7 confirms / 0 rejects vs the relabeled model's 9/14). If
# alteration separability lives in a representation less specialized
# toward the mafic/plag axes, this encoder may take to the new class
# better than the champion's.
#
# Arms: the standard lrscale triple (no frozen arm per sweep convention).
# Same data as the champion-backbone sweep so all runs are comparable:
# v2 parquet ONLY + v2-only cache, MTRDR plag synth rows.

WORK_DIR=/groups/sbyrne/phillipsm/crism_classification
CKPT_DIR=${WORK_DIR}/checkpoints
ENCODER=${CKPT_DIR}/ft_bland_v3_lrscale0001_cont1_best.pt
CACHE_DIR=${WORK_DIR}/data/patch_cache_v2_only
PARQUET=${WORK_DIR}/data/mrral_pixels_with_review_v2.parquet
MTRDR_PATCHES=${WORK_DIR}/data/contrastive/extra_plag_roi/patches.npy
MTRDR_ROWS=${WORK_DIR}/data/mtrdr_plag_rows.parquet

RUN_NAMES=("ft_6cls_v3b_lrscale0001" "ft_6cls_v3b_lrscale001" "ft_6cls_v3b_lrscale01")
LR_SCALES=("0.001" "0.01" "0.1")

RUN_NAME=${RUN_NAMES[$SLURM_ARRAY_TASK_ID]}
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

echo "=== arm $SLURM_ARRAY_TASK_ID: $RUN_NAME  (encoder_lr_scale=$LR_SCALE) ==="

$PYTHON -u scripts/train.py \
    --model spatial_vit \
    --run_name "$RUN_NAME" \
    --init_ckpt "$ENCODER" \
    --with_alteration \
    --mrral_parquets "$PARQUET" \
    --patch_cache_dir "$CACHE_DIR" \
    --synth_train_cache "$MTRDR_PATCHES" \
    --synth_train_parquet "$MTRDR_ROWS" \
    --lr 5e-4 --encoder_lr_scale "$LR_SCALE" \
    --epochs 150 --patience 40 --min_delta 0.0005 \
    --batch_size 256 \
    --embed_dim 128 --n_heads 4 --n_layers 6 \
    --patch_size 7 \
    --asl_loss

echo "=== $RUN_NAME done ==="
