#!/bin/bash
#SBATCH --job-name=crism_6cls_mc11val
#SBATCH --account=sbyrne
#SBATCH --partition=gpu_standard
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64gb
#SBATCH --time=5-00:00:00
#SBATCH --array=0-2
#SBATCH --output=logs/6cls_mc11val_%a_%j.log
#SBATCH --error=logs/6cls_mc11val_%a_%j.log

# 6-class FT measured on the CLEAN MC11 held-out alteration validation set.
#
# This is the honest remeasurement: val_AP_alteration now scores held-out
# MC11 review alteration (same distribution as the training alteration
# signal), not the mafic-contaminated gpkg polygons. stop_metric and
# best-checkpoint both follow val_AP_alteration, so each run trains as
# long as ALTERATION generalizes to held-out MC11 polygons.
#
# Three best alteration backbones from the prior sweeps, all lr_scale 0.01
# (the scale that consistently helped alteration):
#   0: fresh v3 denoising MAE   (prior alt max 0.089 on the bad val)
#   1: fresh SPEND MAE          (prior 0.086; smoothing bias favors
#                                broad hydration bands)
#   2: champion ft_review_mtrdr (warm-start; best 5-class backbone)
#
# Data: data/patch_cache_mc11val + mrral_pixels_with_review_mc11val.parquet
# (built by hpc_build_mc11val_data.slurm — chain with afterok). Also carries
# the band-0 (I/F>1.0 -> nodata) cleanup via the rebuilt cache.

WORK_DIR=/groups/sbyrne/phillipsm/crism_classification
DATA_ROOT=/xdisk/sbyrne/phillipsm/CRISM_MRDR          # all bulk data (xdisk)
DATA_DIR=${DATA_ROOT}/crism_classification/data
CKPT_DIR=${DATA_ROOT}/crism_classification/checkpoints
CACHE_DIR=${DATA_DIR}/patch_cache_mc11val
PARQUET=${DATA_DIR}/mrral_pixels_with_review_mc11val.parquet
MTRDR_PATCHES=${DATA_DIR}/contrastive/extra_plag_roi/patches.npy
MTRDR_ROWS=${DATA_DIR}/mtrdr_plag_rows.parquet

RUN_NAMES=("ft_6cls_mc11val_denoise" "ft_6cls_mc11val_spend" "ft_6cls_mc11val_champion")
ENCODERS=("${CKPT_DIR}/spatial_mae_denoising_128d_6l_best.pt"
          "${CKPT_DIR}/spatial_mae_spend_128d_6l_best.pt"
          "${CKPT_DIR}/ft_review_mtrdr_best.pt")
# denoise/spend are MAE pretrains (encoder_state) -> --pretrain_ckpt;
# champion is a fine-tune ckpt (model_state) -> --init_ckpt.
CKPT_FLAGS=("--pretrain_ckpt" "--pretrain_ckpt" "--init_ckpt")

RUN_NAME=${RUN_NAMES[$SLURM_ARRAY_TASK_ID]}
ENCODER=${ENCODERS[$SLURM_ARRAY_TASK_ID]}
CKPT_FLAG=${CKPT_FLAGS[$SLURM_ARRAY_TASK_ID]}

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

# Confirm the clean val alteration set is present (~18k held-out MC11 + pure).
$PYTHON -c "
import pandas as pd
df = pd.read_parquet('${PARQUET}', columns=['split','alteration'])
v = df[df.split=='val']; n = int((v.alteration>0.5).sum())
print(f'  val alteration positives: {n:,} (prevalence {n/len(v):.3f})')
if n < 5000:
    print('ERROR: clean MC11 val alteration too small — did --alt_holdout_frac run?')
    raise SystemExit(1)
" || exit 1

echo "=== arm $SLURM_ARRAY_TASK_ID: $RUN_NAME ($CKPT_FLAG $(basename $ENCODER), monitor=val_AP_alteration) ==="

$PYTHON -u scripts/train.py \
    --model spatial_vit \
    --run_name "$RUN_NAME" \
    $CKPT_FLAG "$ENCODER" \
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
