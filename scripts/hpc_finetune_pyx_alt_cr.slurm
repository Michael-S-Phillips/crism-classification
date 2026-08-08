#!/bin/bash
#SBATCH --job-name=crism_pyxalt_cr
#SBATCH --account=sbyrne
#SBATCH --partition=gpu_standard
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64gb
#SBATCH --time=1-00:00:00
#SBATCH --array=0-2
#SBATCH --output=logs/pyxalt_cr_%a_%j.log
#SBATCH --error=logs/pyxalt_cr_%a_%j.log

# pyx-merge on the CR representation, trained on HAND-LABELED DATA ONLY (no review).
#   * Vocab: --pyx_alt = 5-class olivine/pyx/plagioclase/other/alteration
#     (lcp+hcp -> pyx). The base parquet's native classes; bland/junk are
#     review-derived and deliberately absent here.
#   * Data:  the BASE parquet mrral_pixels.parquet ALONE — no with_review concat,
#     no relabels, no MTRDR plag patches. This is the single-variable test of
#     "does the pyx+CR recipe hold up on hand-labels only?" vs ft_6cls_pyxcr_*.
#   * Encoder: spatial_mae_cr_denoising_256d_6l (MAE-pretrained on UNLABELED
#     tiles — no labels, so reusing it introduces no review data).
#
# NOTE: unlike ft_6cls_pyxcr_* this CANNOT reuse patch_cache_7cls_cr — that cache
# is row-aligned to the review-augmented 7cls parquet. We need a CR labeled cache
# built from the BASE parquet's raw cache (see PREREQUISITES below).
#
# Storage layout (harmonized): code + conda env in /groups; all bulk data
# (tiles, parquets, caches, checkpoints, outputs) under DATA_ROOT on xdisk.
#
# Submit:  sbatch scripts/hpc_finetune_pyx_alt_cr.slurm
# Floor-test each arm with:
#   CLASSIFY_EXTRA_ARGS="--continuum_removed --brightness_aux --embed_dim 256 --pyx_alt" \
#     bash scripts/floor_test.sh <ckpt> <tag>

WORK_DIR=/groups/sbyrne/phillipsm/crism_classification          # code (git repo)
DATA_ROOT=/xdisk/sbyrne/phillipsm/CRISM_MRDR                   # all bulk data (xdisk)
PYTHON=/groups/sbyrne/phillipsm/micromamba/envs/crism/bin/python

CKPT_DIR=${DATA_ROOT}/crism_classification/checkpoints
DATA_DIR=${DATA_ROOT}/crism_classification/data
ENCODER=${CKPT_DIR}/spatial_mae_cr_denoising_256d_6l_best.pt
EMBED_DIM=256
PARQUET=${DATA_DIR}/mrral_pixels.parquet                       # BASE, hand-labeled only
RAW_CACHE=${DATA_DIR}/patch_cache                              # raw cache aligned to base parquet
CACHE_DIR=${DATA_DIR}/patch_cache_base_cr                      # CR labeled cache (built below)

RUN_NAMES=("ft_5cls_pyxalt_cr_lrscale0001" "ft_5cls_pyxalt_cr_lrscale001" "ft_5cls_pyxalt_cr_lrscale01")
LR_SCALES=("0.001" "0.01" "0.1")
RUN_NAME=${RUN_NAMES[$SLURM_ARRAY_TASK_ID]}
LR_SCALE=${LR_SCALES[$SLURM_ARRAY_TASK_ID]}

cd "$WORK_DIR"
mkdir -p logs

# Only data_root is machine-specific; config_loader derives checkpoints_dir,
# output_dir, patch_cache_dir, gpkg_dir from it (all under DATA_ROOT).
if [ ! -f config.local.yaml ]; then
    cat > config.local.yaml <<EOF
data_root: ${DATA_ROOT}
EOF
fi

# ── PREREQUISITES ────────────────────────────────────────────────────────────
# 1. Base parquet + encoder present.
for f in "$ENCODER" "$PARQUET"; do
    if [ ! -f "$f" ]; then echo "ERROR: missing $f"; exit 1; fi
done
# 2. Base raw patch cache (built by cache_mrral_patches.py from the base parquet).
for split in train val; do
    if [ ! -f "${RAW_CACHE}/mrral_${split}_patches_p7.npy" ]; then
        echo "ERROR: base raw patch cache missing the '$split' split at $RAW_CACHE"
        echo "Build it:  $PYTHON scripts/cache_mrral_patches.py    # uses base parquet + data_root"
        exit 1
    fi
done
# 3. CR labeled cache + brightness sidecar, built FROM the base raw cache.
#    Check EVERY split the run touches, not just train: a train-only cache used
#    to pass this guard and then die in the first validation pass (2026-08-07).
for split in train val; do
    for f in "${CACHE_DIR}/mrral_${split}_patches_p7.npy" \
             "${CACHE_DIR}/mrral_${split}_patches_p7_brightness.npy"; do
        if [ ! -f "$f" ]; then
            echo "ERROR: base CR labeled cache incomplete — missing $f"
            echo "Build it (all splits, and check it exits 0):"
            echo "  $PYTHON scripts/build_cr_labeled_cache.py --raw_dir ${RAW_CACHE} --out_dir ${CACHE_DIR} --splits train val test"
            exit 1
        fi
    done
done

echo "=== arm $SLURM_ARRAY_TASK_ID: $RUN_NAME (encoder_lr_scale=$LR_SCALE, embed_dim=$EMBED_DIM, pyx_alt+CR, hand-labeled only) ==="

$PYTHON -u scripts/train.py \
    --model spatial_vit_aux \
    --run_name "$RUN_NAME" \
    --pretrain_ckpt "$ENCODER" \
    --pyx_alt \
    --continuum_removed --cache_is_cr --brightness_aux \
    --mrral_parquets "$PARQUET" \
    --patch_cache_dir "$CACHE_DIR" \
    --lr 5e-4 --encoder_lr_scale "$LR_SCALE" \
    --epochs 150 --patience 40 --min_delta 0.0005 \
    --batch_size 256 \
    --embed_dim ${EMBED_DIM} --n_heads 4 --n_layers 6 \
    --patch_size 7 \
    --asl_loss

echo "=== $RUN_NAME done ==="
