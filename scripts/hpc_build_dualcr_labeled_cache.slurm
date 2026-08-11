#!/bin/bash
#SBATCH --job-name=build_dualcr_labeled_cache
#SBATCH --account=sbyrne
#SBATCH --partition=standard
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64gb
#SBATCH --time=0-12:00:00
#SBATCH --output=logs/build_dualcr_labeled_cache_%j.log
#SBATCH --error=logs/build_dualcr_labeled_cache_%j.log

# JOB 3 of 4 — labeled 118-channel dual-CR patch cache for the hand-core parquet.
#   STEP 1  raw 7x7x59 patches, row-aligned to the parquet  -> patch_cache_handcore
#           (skipped if it already exists and is complete)
#   STEP 2  dual-CR patches + brightness sidecar            -> patch_cache_handcore_dualcr
#
# NO dependency on the pretrain — this and hpc_pretrain_dualcr_mae.slurm are
# independent and should run in parallel.
#
# Step 2 must cover EVERY split. A CR cache missing `val` is what killed the
# pyxalt_cr runs: the dataset fell back to reading tiles on the fly while
# cache_is_cr suppressed continuum removal. CRISMSpectralPatchDataset now fails
# fast on that, and its byte-exact size guard is keyed on the actual channel
# count, so a 59-channel cache handed to --dual_cr is rejected rather than read
# as half of a dual patch. Both only help if the cache is genuinely complete —
# hence the verification block at the end.
#
# ALL output on /xdisk. /groups filled up and killed two CR patch-cache builds
# with "No space left on device"; dual doubles the width.

set -uo pipefail

WORK_DIR=/groups/sbyrne/phillipsm/crism_classification          # code + env only
DATA_ROOT=/xdisk/sbyrne/phillipsm/CRISM_MRDR                    # ALL bulk data
DATA_DIR=${DATA_ROOT}/crism_classification/data
PYTHON=/groups/sbyrne/phillipsm/micromamba/envs/crism/bin/python

PARQUET=${DATA_DIR}/mrral_pixels_7cls_handcore.parquet
RAW_CACHE=${DATA_DIR}/patch_cache_handcore
DUAL_CACHE=${DATA_DIR}/patch_cache_handcore_dualcr
SPLITS="train val test"

cd "$WORK_DIR" || { echo "ERROR: cannot cd $WORK_DIR"; exit 1; }
mkdir -p logs

if [ ! -f config.local.yaml ]; then
    printf 'data_root: %s\n' "${DATA_ROOT}" > config.local.yaml
fi

[ -f "$PARQUET" ] || { echo "ERROR: $PARQUET not found — run hpc_build_handcore_data.slurm first"; exit 1; }

# Tiles are FLAT at data_root on HPC, not under mc*/. Probe, never assume:
# STEP 1 reads them through cache_mrral_patches.py, which globs both layouts.
n_flat=$(find "$DATA_ROOT" -maxdepth 1 -name 't*mrral*.hdr' 2>/dev/null | wc -l)
n_mc=$(find "$DATA_ROOT" -maxdepth 2 -path '*/mc*' -name 't*mrral*.hdr' 2>/dev/null | wc -l)
echo "mrral tiles: flat=$n_flat, mc*/=$n_mc"

# Don't spend hours caching spectra we haven't checked.
echo "=== auditing the parquet before caching ($(date)) ==="
$PYTHON scripts/audit_spectra_quality.py "$PARQUET" --fail_tile_over 5 || {
    echo "ERROR: parquet failed the spectra audit. Refusing to cache."; exit 1; }

# ── STEP 1: raw patch cache (only if incomplete) ─────────────────────────────
raw_complete=1
for split in $SPLITS; do
    [ -f "${RAW_CACHE}/mrral_${split}_patches_p7.npy" ] || raw_complete=0
done
if [ "$raw_complete" -eq 1 ]; then
    echo "=== STEP 1: raw cache already complete at $RAW_CACHE, skipping ==="
else
    [ "$((n_flat + n_mc))" -gt 0 ] || {
        echo "ERROR: raw cache incomplete and no t*mrral*.hdr under $DATA_ROOT"; exit 1; }
    echo "=== STEP 1: raw patch cache -> $RAW_CACHE ($(date)) ==="
    $PYTHON -u scripts/cache_mrral_patches.py \
        --parquets "$PARQUET" \
        --cache_dir "$RAW_CACHE" \
        --splits $SPLITS \
        --patch_size 7 || { echo "ERROR: raw cache build failed"; exit 1; }
fi
for split in $SPLITS; do
    f="${RAW_CACHE}/mrral_${split}_patches_p7.npy"
    [ -f "$f" ] || { echo "ERROR: raw cache missing $f"; exit 1; }
    echo "  $(basename "$f"): $(stat -c%s "$f") bytes"
done

# ── STEP 2: dual-CR patches (118 ch) + brightness sidecar ────────────────────
echo "=== STEP 2: dual-CR cache -> $DUAL_CACHE ($(date)) ==="
$PYTHON -u scripts/build_cr_labeled_cache.py \
    --raw_dir "$RAW_CACHE" \
    --out_dir "$DUAL_CACHE" \
    --dual \
    --splits $SPLITS \
    --patch_size 7 \
    --jobs "${SLURM_CPUS_PER_TASK:-8}" || {
        echo "ERROR: dual-CR cache build failed (it exits non-zero if ANY requested"
        echo "  split had no raw cache — a half-built cache is worse than none)"
        exit 1; }

# ── Verify: every split, patches AND sidecar, and 118 channels not 59 ────────
echo "=== verifying the dual-CR cache is complete and 118-channel ==="
missing=0
for split in $SPLITS; do
    for f in "${DUAL_CACHE}/mrral_${split}_patches_p7.npy" \
             "${DUAL_CACHE}/mrral_${split}_patches_p7_brightness.npy"; do
        if [ -f "$f" ]; then
            echo "  OK   $(basename "$f")  $(stat -c%s "$f") bytes"
        else
            echo "  MISS $f"; missing=1
        fi
    done
done
[ $missing -eq 0 ] || { echo "ERROR: dual-CR cache incomplete — do not fine-tune on it"; exit 1; }

# Width check. These are raw headerless memmaps, so the channel count is not
# recoverable from a file on its own — and a bytes % (7*7*118*4) test would be
# worthless, because 59 * (even n) == 118 * (n/2), so any 59-channel cache with an
# even patch count passes it. Two independent references instead:
#   1. the RAW cache: dual must be EXACTLY 2x its byte count
#   2. the PARQUET row count per split: exactly what CRISMSpectralPatchDataset's
#      byte-exact guard will demand at fine-tune startup
$PYTHON - "$RAW_CACHE" "$DUAL_CACHE" "$PARQUET" $SPLITS <<'EOF' || { echo "ERROR: the dual-CR cache did not verify — was --dual actually passed?"; exit 1; }
import os, sys
import pandas as pd
raw, dual, parquet, splits = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]
df = pd.read_parquet(parquet, columns=['split'])
for s in splits:
    f = f'mrral_{s}_patches_p7.npy'
    nr = os.path.getsize(os.path.join(raw, f))
    nd = os.path.getsize(os.path.join(dual, f))
    assert nd == 2 * nr, f'{s}: dual {nd:,} != 2 x raw {nr:,} (118 vs 59 channels)'
    rows = int((df['split'] == s).sum())
    want = rows * 7 * 7 * 118 * 4
    assert nd == want, (f'{s}: {nd:,} bytes but the parquet has {rows:,} rows, '
                        f'which needs exactly {want:,} at 118 channels')
    print(f'  OK {s}: {rows:,} patches x 7x7x118')
print('dual-CR labeled cache verified at 118 channels against raw cache AND parquet')
EOF

echo "=== DONE ($(date)) ==="
echo "Next: sbatch scripts/hpc_finetune_dualcr.slurm"
