#!/bin/bash
#SBATCH --job-name=crism_handcore_ft
#SBATCH --account=sbyrne
#SBATCH --partition=gpu_standard
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64gb
#SBATCH --time=1-00:00:00
#SBATCH --array=0-1
#SBATCH --output=logs/handcore_ft_%a_%j.log
#SBATCH --error=logs/handcore_ft_%a_%j.log

# Fine-tune on the HAND-CORE parquet, with plagioclase leaning on MTRDR.
#
# Prereq: scripts/hpc_build_handcore_data.slurm has produced
#   ${DATA_DIR}/mrral_pixels_7cls_handcore.parquet, and a patch cache exists
#   for it (ALL splits — an incomplete CR cache now fails fast at startup
#   rather than silently training on raw patches).
#
# ── Why plagioclase is configured the way it is ──────────────────────────────
# A spectral-angle confusion matrix over the hand-core classes (2026-08-08,
# scripts/sam_confusion_matrix.py) found:
#     plag(MTRDR)  recall 0.87 (k=1) -> 0.97 (k=5), self-angle 1.09 deg
#     plag(hand)   recall 0.29 (k=1) -> 0.41 (k=5), confused with alteration
#                  and lcp at 0.22 each
# Allowing 5 endmembers per class rescues `junk` (0.50 -> 0.81), so the method
# does detect multi-modality. Hand plagioclase does not recover, which means it
# is genuinely entangled with other classes rather than several clean modes.
# Under the hand-core policy plag is 100% hand-labelled — there are zero plag
# confirms in either review session — so the class is composed almost entirely
# of its least separable variant.
#
# So: train plagioclase on the LARGER MTRDR pool and validate on a DISJOINT one.
#   train:  data/patch_cache/mtrdr_plag_rows.parquet   8,671 rows, 18 tiles
#   val:    data/mtrdr_plag_rows.parquet                 109 val rows, 31 tiles
# Verified 2026-08-08: the two sets share ZERO pixels and ZERO tiles, so this
# adds no spatial leakage. The 8,671-row pool was already on disk with its own
# patch cache and referenced by nothing.
#
# NOTE the synth-train injection now filters split=='train' (fix e83a827).
# Before that fix it served every row in the parquet, so a synth set carrying
# its own val rows trained on them AND validated on them.
#
# Submit:  sbatch scripts/hpc_finetune_handcore.slurm

WORK_DIR=/groups/sbyrne/phillipsm/crism_classification
DATA_ROOT=/xdisk/sbyrne/phillipsm/CRISM_MRDR
PYTHON=/groups/sbyrne/phillipsm/micromamba/envs/crism/bin/python

CKPT_DIR=${DATA_ROOT}/crism_classification/checkpoints
DATA_DIR=${DATA_ROOT}/crism_classification/data
ENCODER=${CKPT_DIR}/spatial_mae_cr_denoising_256d_6l_best.pt
EMBED_DIM=256

PARQUET=${DATA_DIR}/mrral_pixels_7cls_handcore.parquet
CACHE_DIR=${DATA_DIR}/patch_cache_handcore_cr

# Plagioclase: big disjoint pool for train, small disjoint pool for val.
MTRDR_TRAIN_PATCHES=${DATA_DIR}/patch_cache/mtrdr_plag_patches_p7.npy
MTRDR_TRAIN_ROWS=${DATA_DIR}/patch_cache/mtrdr_plag_rows.parquet
MTRDR_VAL_PATCHES=${DATA_DIR}/contrastive/extra_plag_roi/patches.npy
MTRDR_VAL_ROWS=${DATA_DIR}/mtrdr_plag_rows.parquet

# Arm 0 is the clean baseline: ONE variable at a time. Arm 1 changes only the
# weight scheme, so any difference is attributable to hand-vs-review weighting
# rather than to the data rebuild.
RUN_NAMES=("ft_7cls_handcore_level" "ft_7cls_handcore_reviewup")
SCHEMES=("level" "review_up")
RUN_NAME=${RUN_NAMES[$SLURM_ARRAY_TASK_ID]}
SCHEME=${SCHEMES[$SLURM_ARRAY_TASK_ID]}

cd "$WORK_DIR"
mkdir -p logs
if [ ! -f config.local.yaml ]; then
    printf 'data_root: %s\n' "${DATA_ROOT}" > config.local.yaml
fi

# ── PREREQUISITES ────────────────────────────────────────────────────────────
for f in "$ENCODER" "$PARQUET" "$MTRDR_TRAIN_PATCHES" "$MTRDR_TRAIN_ROWS" \
         "$MTRDR_VAL_PATCHES" "$MTRDR_VAL_ROWS"; do
    [ -f "$f" ] || { echo "ERROR: missing $f"; exit 1; }
done
for split in train val; do
    for f in "${CACHE_DIR}/mrral_${split}_patches_p7.npy" \
             "${CACHE_DIR}/mrral_${split}_patches_p7_brightness.npy"; do
        [ -f "$f" ] || { echo "ERROR: CR patch cache incomplete — missing $f"; \
            echo "Build it: $PYTHON scripts/build_cr_labeled_cache.py --raw_dir <raw> --out_dir ${CACHE_DIR} --splits train val test"; \
            exit 1; }
    done
done

# Refuse to train on spectra we have not checked. Cheap next to a wasted GPU day.
echo "=== spectra audit ==="
$PYTHON scripts/audit_spectra_quality.py "$PARQUET" --fail_tile_over 5 || {
    echo "ERROR: $PARQUET failed the spectra audit. Refusing to train."; exit 1; }

echo "=== arm $SLURM_ARRAY_TASK_ID: $RUN_NAME (weight_scheme=$SCHEME) ==="

$PYTHON -u scripts/train.py \
    --model spatial_vit_aux \
    --run_name "$RUN_NAME" \
    --pretrain_ckpt "$ENCODER" \
    --seven_class \
    --continuum_removed --cache_is_cr --brightness_aux \
    --mrral_parquets "$PARQUET" \
    --patch_cache_dir "$CACHE_DIR" \
    --weight_scheme "$SCHEME" \
    --synth_train_cache "$MTRDR_TRAIN_PATCHES" \
    --synth_train_parquet "$MTRDR_TRAIN_ROWS" \
    --synth_val_cache "$MTRDR_VAL_PATCHES" \
    --synth_val_parquet "$MTRDR_VAL_ROWS" \
    --lr 5e-4 --encoder_lr_scale 0.001 \
    --epochs 150 --patience 40 --min_delta 0.0005 \
    --batch_size 256 \
    --embed_dim ${EMBED_DIM} --n_heads 4 --n_layers 6 \
    --patch_size 7 \
    --asl_loss

echo "=== $RUN_NAME done ==="
echo "Floor-test with:"
echo "  CLASSIFY_EXTRA_ARGS=\"--continuum_removed --brightness_aux --embed_dim ${EMBED_DIM} --seven_class\" \\"
echo "    bash scripts/floor_test.sh ${CKPT_DIR}/${RUN_NAME}_best.pt ${RUN_NAME}"
