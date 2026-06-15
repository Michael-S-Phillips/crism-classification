#!/bin/bash
#SBATCH --job-name=crism_6cls_purealt
#SBATCH --account=sbyrne
#SBATCH --partition=gpu_standard
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64gb
#SBATCH --time=1-12:00:00
#SBATCH --array=0-1
#SBATCH --output=logs/6cls_purealt_%a_%j.log
#SBATCH --error=logs/6cls_purealt_%a_%j.log

# 6-class FT on the PURE-ALTERATION training set.
#
# Hypothesis (from the flat ~0.05 alteration AP across all backbones):
# the mixed "mafic + alteration" gpkg rows poison the alteration signal —
# they teach the model that ordinary olivine/HCP spectra are alteration,
# burying the hydration-band signature. In this lineage the model learns
# alteration ONLY from single-label examples: the 103,895 MC13
# review-tagged pixels + 412 pixels from the 8 pure gpkg polygons.
# Mixed train rows keep their mafic labels (alteration=0). Val/test
# stamping is unchanged, so val_AP_alteration is directly comparable
# with every other 6-class run.
#
# Data: data/mrral_pixels_with_review_v2pure.parquet + patch_cache_v2pure
# (built by hpc_build_purealt_data.slurm — chain with afterok).
#
# Arms mirror the baseline + the most promising ablation scale so the
# data effect is isolated at both:
#   0: champion warm-start @ encoder_lr_scale 0.001  (baseline mirror)
#   1: champion warm-start @ encoder_lr_scale 0.01

WORK_DIR=/groups/sbyrne/phillipsm/crism_classification
CKPT_DIR=${WORK_DIR}/checkpoints
ENCODER=${CKPT_DIR}/ft_review_mtrdr_best.pt
CACHE_DIR=${WORK_DIR}/data/patch_cache_v2pure
PARQUET=${WORK_DIR}/data/mrral_pixels_with_review_v2pure.parquet
MTRDR_PATCHES=${WORK_DIR}/data/contrastive/extra_plag_roi/patches.npy
MTRDR_ROWS=${WORK_DIR}/data/mtrdr_plag_rows.parquet

RUN_NAMES=("ft_6cls_purealt_ls0001" "ft_6cls_purealt_ls001")
LR_SCALES=("0.001" "0.01")

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
    echo "ERROR: cache dir missing: $CACHE_DIR (build via hpc_build_purealt_data.slurm)"
    exit 1
fi

# Verify the pure lineage: train alteration positives should be ~104k
# (review-tagged) — far below the ~140k of the mixed build.
$PYTHON -c "
import pandas as pd
df = pd.read_parquet('${PARQUET}', columns=['split', 'alteration'])
tr = df[df.split == 'train']
n = int((tr.alteration > 0.5).sum())
print(f'  train alteration positives: {n:,}')
if n > 130_000:
    print('ERROR: looks like the MIXED build, not pure — check step 1 of '
          'hpc_build_purealt_data.slurm ran with --pure_train_only.')
    raise SystemExit(1)
" || exit 1

echo "=== arm $SLURM_ARRAY_TASK_ID: $RUN_NAME (encoder_lr_scale=$LR_SCALE) ==="

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
