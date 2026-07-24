#!/bin/bash
#SBATCH --job-name=build_global_cache_cr
#SBATCH --account=sbyrne
#SBATCH --partition=standard
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64gb
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/build_global_cache_cr_%j.log
#SBATCH --error=logs/build_global_cache_cr_%j.log

# Build the global pre-training patch cache as CONTINUUM-REMOVED patches.
# CPU-only; CR adds a per-pixel convex-hull pass, so budget more than the raw
# build (~5.5 h). 24 hr SLURM budget.
# Spec: docs/superpowers/specs/2026-07-15-cr-mrral-representation-design.md
#
# PREREQUISITE: the /xdisk mrral tiles must be REPAIRED first (the tile-refresh
# rsync of truncated/missing tiles). Building from truncated tiles bakes zero-fill
# corruption into the CR cache and every downstream pretrain.

WORK_DIR=/groups/sbyrne/phillipsm/crism_classification
PYTHON=/groups/sbyrne/phillipsm/micromamba/envs/crism/bin/python
OUTPUT=/xdisk/sbyrne/phillipsm/crism_patch_cache_cr

cd "$WORK_DIR"

cat > config.local.yaml <<EOF
data_root: /xdisk/sbyrne/phillipsm/CRISM_MRDR
checkpoint_dir: ${WORK_DIR}/checkpoints
checkpoints_dir: ${WORK_DIR}/checkpoints
output_dir: ${WORK_DIR}/data
patch_cache_dir: ${WORK_DIR}/data/patch_cache
global_patch_cache_dir: ${OUTPUT}
EOF

mkdir -p logs "$OUTPUT"

echo "=== CR global patch cache build start: $(date) ==="

${PYTHON} -u scripts/build_global_patch_cache.py \
    --continuum_removed \
    --output "$OUTPUT" \
    --workers 16 \
    --seed 42 \
    --patches_per_tile_target 2834 \
    --patches_per_shard 100000

echo "=== CR global patch cache build end: $(date) ==="
