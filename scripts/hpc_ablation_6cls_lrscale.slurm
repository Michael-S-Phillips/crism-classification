#!/bin/bash
#SBATCH --job-name=crism_6cls_lr_abl
#SBATCH --account=sbyrne
#SBATCH --partition=gpu_standard
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64gb
#SBATCH --time=0-24:00:00
#SBATCH --array=0-2
#SBATCH --output=logs/6cls_lr_abl_%a_%j.log
#SBATCH --error=logs/6cls_lr_abl_%a_%j.log

# LR ablation for the 6-class alteration model.
#
# Motivation (wandb ft_6cls_alteration_v2, epochs 1-19): val_mAP peaked at
# epoch 1 (0.6528) and decays as LCP/HCP slowly erode, while alteration AP
# creeps up at ~+0.0005/epoch from a dismal 0.05. Hypothesis: alteration is
# encoder-limited (like plag) — the warm-started encoder never had to
# represent alteration spectra, and at encoder_lr_scale=0.001 it cannot
# adapt; the drift it does undergo erodes the old classes without buying
# alteration anything. These arms test whether letting the encoder (or the
# head) move faster finds a better optimum.
#
# Arms (the running ft_6cls_alteration_v2 IS the lr=5e-4/scale=0.001 arm;
# no frozen-encoder arm per the established sweep convention — it always
# underperforms):
#   0: encoder_lr_scale 0.01,  head lr 5e-4
#   1: encoder_lr_scale 0.1,   head lr 5e-4
#   2: encoder_lr_scale 0.001, head lr 2e-3  (head-bottleneck control)
#
# Longer budget than the v2 run: patience 40 + min_delta 0.0005, because
# alteration improves slowly and we want the stopper to wait out the
# LCP/HCP-vs-alteration crossover if there is one. best-checkpointing on
# val_mAP keeps the peak regardless.
#
# Same data as ft_6cls_alteration_v2: v2 parquet ONLY + v2-only cache
# (post-audit configuration), MTRDR plag synth rows, champion warm-start.

WORK_DIR=/groups/sbyrne/phillipsm/crism_classification
DATA_ROOT=/xdisk/sbyrne/phillipsm/CRISM_MRDR          # all bulk data (xdisk)
DATA_DIR=${DATA_ROOT}/crism_classification/data
CKPT_DIR=${DATA_ROOT}/crism_classification/checkpoints
ENCODER=${CKPT_DIR}/ft_review_mtrdr_best.pt
CACHE_DIR=${DATA_DIR}/patch_cache_v2_only
PARQUET=${DATA_DIR}/mrral_pixels_with_review_v2.parquet
MTRDR_PATCHES=${DATA_DIR}/contrastive/extra_plag_roi/patches.npy
MTRDR_ROWS=${DATA_DIR}/mtrdr_plag_rows.parquet

RUN_NAMES=("ft_6cls_lrscale001" "ft_6cls_lrscale01" "ft_6cls_headlr2e3")
LR_ARGS=("--lr 5e-4 --encoder_lr_scale 0.01"
         "--lr 5e-4 --encoder_lr_scale 0.1"
         "--lr 2e-3 --encoder_lr_scale 0.001")

RUN_NAME=${RUN_NAMES[$SLURM_ARRAY_TASK_ID]}
THIS_LR_ARGS=${LR_ARGS[$SLURM_ARRAY_TASK_ID]}

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
    echo "ERROR: cache dir missing: $CACHE_DIR (build via hpc_build_cache_v2_only.slurm)"
    exit 1
fi

echo "=== arm $SLURM_ARRAY_TASK_ID: $RUN_NAME  ($THIS_LR_ARGS) ==="

$PYTHON -u scripts/train.py \
    --model spatial_vit \
    --run_name "$RUN_NAME" \
    --init_ckpt "$ENCODER" \
    --with_alteration \
    --mrral_parquets "$PARQUET" \
    --patch_cache_dir "$CACHE_DIR" \
    --synth_train_cache "$MTRDR_PATCHES" \
    --synth_train_parquet "$MTRDR_ROWS" \
    $THIS_LR_ARGS \
    --epochs 150 --patience 40 --min_delta 0.0005 \
    --batch_size 256 \
    --embed_dim 128 --n_heads 4 --n_layers 6 \
    --patch_size 7 \
    --asl_loss

echo "=== $RUN_NAME done ==="
