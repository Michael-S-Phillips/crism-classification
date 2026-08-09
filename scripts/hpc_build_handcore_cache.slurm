#!/bin/bash
#SBATCH --job-name=crism_handcore_cache
#SBATCH --account=sbyrne
#SBATCH --partition=standard
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64gb
#SBATCH --time=0-12:00:00
#SBATCH --output=logs/handcore_cache_%j.log
#SBATCH --error=logs/handcore_cache_%j.log

# Patch caches for the hand-core parquet. This is the missing link between
# hpc_build_handcore_data.slurm and hpc_finetune_handcore.slurm.
#
#   STEP 1  raw 7x7x59 patches, row-aligned to the parquet   -> patch_cache_handcore
#   STEP 2  continuum-removed patches + brightness sidecar   -> patch_cache_handcore_cr
#
# Both caches are RAW headerless memmaps whose byte count must match the
# parquet's row count exactly. CRISMSpectralPatchDataset enforces that at load
# time — a mismatch means the parquet was rebuilt without rebuilding the cache,
# which is why each recipe gets its OWN cache dir rather than sharing one.
#
# Step 2 must cover EVERY split. A CR cache missing `val` is what killed the
# pyxalt_cr runs: the dataset fell back to reading tiles on the fly, and
# cache_is_cr suppressed continuum removal, so it would have trained on raw
# patches. That now fails fast, but only if the cache is actually complete.
#
# Submit after the data build succeeds:
#   sbatch --dependency=afterok:<data_job_id> scripts/hpc_build_handcore_cache.slurm

WORK_DIR=/groups/sbyrne/phillipsm/crism_classification
DATA_ROOT=/xdisk/sbyrne/phillipsm/CRISM_MRDR
PYTHON=/groups/sbyrne/phillipsm/micromamba/envs/crism/bin/python

DATA_DIR=${DATA_ROOT}/crism_classification/data
PARQUET=${DATA_DIR}/mrral_pixels_7cls_handcore.parquet
RAW_CACHE=${DATA_DIR}/patch_cache_handcore
CR_CACHE=${DATA_DIR}/patch_cache_handcore_cr

cd "$WORK_DIR"
mkdir -p logs
if [ ! -f config.local.yaml ]; then
    printf 'data_root: %s\n' "${DATA_ROOT}" > config.local.yaml
fi

[ -f "$PARQUET" ] || { echo "ERROR: $PARQUET not found — run hpc_build_handcore_data.slurm first"; exit 1; }

# Don't spend 8 hours caching spectra we haven't checked.
echo "=== auditing the parquet before caching ($(date)) ==="
$PYTHON scripts/audit_spectra_quality.py "$PARQUET" --fail_tile_over 5 || {
    echo "ERROR: parquet failed the spectra audit. Refusing to cache."; exit 1; }

# ── STEP 1: raw patch cache ──────────────────────────────────────────────────
echo "=== STEP 1: raw patch cache -> $RAW_CACHE ($(date)) ==="
$PYTHON -u scripts/cache_mrral_patches.py \
    --parquets "$PARQUET" \
    --cache_dir "$RAW_CACHE" \
    --splits train val test \
    --patch_size 7 || { echo "ERROR: raw cache build failed"; exit 1; }

for split in train val test; do
    f="${RAW_CACHE}/mrral_${split}_patches_p7.npy"
    [ -f "$f" ] || { echo "ERROR: raw cache missing $f"; exit 1; }
    echo "  $(basename $f): $(stat -c%s "$f") bytes"
done

# ── STEP 2: CR patches + brightness sidecar ──────────────────────────────────
echo "=== STEP 2: CR cache -> $CR_CACHE ($(date)) ==="
$PYTHON -u scripts/build_cr_labeled_cache.py \
    --raw_dir "$RAW_CACHE" \
    --out_dir "$CR_CACHE" \
    --splits train val test \
    --patch_size 7 \
    --jobs "${SLURM_CPUS_PER_TASK:-8}" || {
        echo "ERROR: CR cache build failed (it exits non-zero if ANY requested"
        echo "  split had no raw cache — a half-built CR cache is worse than none)"
        exit 1; }

# ── Verify completeness: every split, patches AND sidecar ────────────────────
echo "=== verifying the CR cache is complete ==="
missing=0
for split in train val test; do
    for f in "${CR_CACHE}/mrral_${split}_patches_p7.npy" \
             "${CR_CACHE}/mrral_${split}_patches_p7_brightness.npy"; do
        if [ -f "$f" ]; then
            echo "  OK   $(basename $f)  $(stat -c%s "$f") bytes"
        else
            echo "  MISS $f"; missing=1
        fi
    done
done
[ $missing -eq 0 ] || { echo "ERROR: CR cache incomplete — do not fine-tune on it"; exit 1; }

echo "=== DONE ($(date)) ==="
echo "Next: sbatch scripts/hpc_finetune_handcore.slurm"
