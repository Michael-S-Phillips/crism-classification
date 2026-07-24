#!/bin/bash
#SBATCH --job-name=build_cr_labeled_cache
#SBATCH --account=sbyrne
#SBATCH --partition=standard
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32gb
#SBATCH --time=0-12:00:00
#SBATCH --output=logs/build_cr_labeled_cache_%j.log
#SBATCH --error=logs/build_cr_labeled_cache_%j.log

# One-time conversion of the raw labeled patch cache -> CR patches + brightness
# sidecar (cache_is_cr format) for the CR fine-tune. Single-process chunked hull
# pass; ~3-4 h. Independent of the /xdisk global tile cache — reads the /groups
# labeled cache built for the honest-splits recovery run.
# NOTE: writes ~3x the raw labeled-cache size (train+val+test). If /groups quota is
# tight, point --out_dir at /xdisk and update CACHE_DIR in hpc_finetune_cr.slurm.

WORK_DIR=/groups/sbyrne/phillipsm/crism_classification
PYTHON=/groups/sbyrne/phillipsm/micromamba/envs/crism/bin/python

cd "$WORK_DIR"
mkdir -p logs

RAW_DIR=${WORK_DIR}/data/patch_cache_7cls
OUT_DIR=${WORK_DIR}/data/patch_cache_7cls_cr

if [ ! -d "$RAW_DIR" ]; then
    echo "ERROR: raw labeled cache missing at $RAW_DIR (run hpc_build_7cls_data.slurm first)"
    exit 1
fi

echo "=== CR labeled-cache build start: $(date) ==="
${PYTHON} -u scripts/build_cr_labeled_cache.py \
    --raw_dir "$RAW_DIR" \
    --out_dir "$OUT_DIR" \
    --splits train val test --patch_size 7
echo "=== CR labeled-cache build end: $(date) ==="
