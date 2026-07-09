#!/bin/bash
#SBATCH --job-name=crism_7cls_v3b_abl
#SBATCH --account=sbyrne
#SBATCH --partition=gpu_standard
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64gb
#SBATCH --time=0-24:00:00
#SBATCH --array=0-2
#SBATCH --output=logs/7cls_v3b_abl_%a_%j.log
#SBATCH --error=logs/7cls_v3b_abl_%a_%j.log

# 7-class LR-scale ablation on the v3-bland encoder.
#
# Encoder: ft_bland_v3_lrscale0001_cont1_best.pt
#   Lineage: spatial_mae_denoising_128d_6l_best.pt
#            → ft_bland_v3_lrscale0001_best.pt
#            → ft_bland_v3_lrscale0001_cont1_best.pt
#   This encoder produced the cleanest MC13 HCP results (7 confirms / 0 rejects
#   vs relabeled model's 9/14), making it the strongest starting representation
#   for a clean 7-class re-build.
#
# Data: mrral_pixels_7cls.parquet + patch_cache_7cls
#   7 classes: olivine | lcp | hcp | plagioclase | bland | alteration | junk
#   Splits: ALL sources unit-aware pixel-balanced 70/15/15 (scripts/split_units.py);
#           whole geographic units held out together (no adjacent-tile leakage).
#   Plag: Argyre/Hellas gpkg (unit-re-split) + MTRDR synth injected at train/val
#   Bland: balanced sources subsampled to 300k, unit-split
#   Alteration: review tags + confirms, unit-balanced holdout
#   Junk: ambiguous hard_neg, per-polygon capped, unit-balanced holdout
#
# Monitored metric: val_mAP_core (mAP EXCLUDING junk) drives early stopping and
#   best-checkpoint selection; full val_mAP (all 7 classes) is still logged.
# EXPECT val_mAP LOWER than prior 7-class runs after this split rebuild — the
#   adjacent-tile leakage that inflated val_AP_plagioclase/olivine is removed.
#   This is metric honesty, NOT a regression.
#
# Arms (standard 3-arm sweep, no frozen-encoder per convention):
#   0: encoder_lr_scale 0.001
#   1: encoder_lr_scale 0.01
#   2: encoder_lr_scale 0.1

WORK_DIR=/groups/sbyrne/phillipsm/crism_classification
CKPT_DIR=${WORK_DIR}/checkpoints
ENCODER=${CKPT_DIR}/ft_bland_v3_lrscale0001_cont1_best.pt
CACHE_DIR=${WORK_DIR}/data/patch_cache_7cls
PARQUET=${WORK_DIR}/data/mrral_pixels_7cls.parquet
MTRDR_PATCHES=${WORK_DIR}/data/contrastive/extra_plag_roi/patches.npy
MTRDR_ROWS=${WORK_DIR}/data/mtrdr_plag_rows.parquet

RUN_NAMES=("ft_7cls_v3b_lrscale0001" "ft_7cls_v3b_lrscale001" "ft_7cls_v3b_lrscale01")
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

# Guard: required inputs
for f in "$ENCODER" "$PARQUET" "$MTRDR_PATCHES" "$MTRDR_ROWS"; do
    if [ ! -f "$f" ]; then echo "ERROR: missing $f"; exit 1; fi
done
if [ ! -d "$CACHE_DIR" ]; then
    echo "ERROR: patch_cache_7cls missing — run hpc_build_7cls_data.slurm first"
    exit 1
fi

echo "=== arm $SLURM_ARRAY_TASK_ID: $RUN_NAME  (encoder_lr_scale=$LR_SCALE) ==="

$PYTHON -u scripts/train.py \
    --model spatial_vit \
    --run_name "$RUN_NAME" \
    --init_ckpt "$ENCODER" \
    --seven_class \
    --mrral_parquets "$PARQUET" \
    --patch_cache_dir "$CACHE_DIR" \
    --synth_train_cache "$MTRDR_PATCHES" \
    --synth_train_parquet "$MTRDR_ROWS" \
    --synth_val_cache "$MTRDR_PATCHES" \
    --synth_val_parquet "$MTRDR_ROWS" \
    --lr 5e-4 --encoder_lr_scale "$LR_SCALE" \
    --epochs 150 --patience 40 --min_delta 0.0005 \
    --batch_size 256 \
    --embed_dim 128 --n_heads 4 --n_layers 6 \
    --patch_size 7 \
    --asl_loss

echo "=== $RUN_NAME done ==="
