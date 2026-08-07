#!/bin/bash
#SBATCH --job-name=build_global_patch_cache
#SBATCH --account=sbyrne
#SBATCH --partition=standard
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64gb
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/build_global_cache_%j.log
#SBATCH --error=logs/build_global_cache_%j.log

# Build the global pre-training patch cache.
# CPU-only — patch extraction is I/O-bound (reading BSQ mrral tiles).
# ~5.5 hours expected with 16 workers, 24 hr SLURM budget for safety.
# Spec: docs/superpowers/specs/2026-05-18-pretrain-patch-cache-design.md

WORK_DIR=/groups/sbyrne/phillipsm/crism_classification
DATA_ROOT=/xdisk/sbyrne/phillipsm/CRISM_MRDR          # all bulk data (xdisk)
DATA_DIR=${DATA_ROOT}/crism_classification/data
PYTHON=/groups/sbyrne/phillipsm/micromamba/envs/crism/bin/python
OUTPUT=/xdisk/sbyrne/phillipsm/crism_patch_cache

cd "$WORK_DIR"

# Make sure config.local.yaml points at the data on /xdisk.
if [ ! -f config.local.yaml ]; then
    cat > config.local.yaml <<EOF
data_root: ${DATA_ROOT}
EOF
fi

mkdir -p logs

echo "=== Global patch cache build start: $(date) ==="

${PYTHON} -u scripts/build_global_patch_cache.py \
    --output "$OUTPUT" \
    --workers 16 \
    --seed 42 \
    --patches_per_tile_target 2834 \
    --patches_per_shard 100000

echo "=== Global patch cache build end: $(date) ==="
