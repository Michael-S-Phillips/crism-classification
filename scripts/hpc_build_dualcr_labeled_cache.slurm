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

# ── ONE BUILDER AT A TIME ────────────────────────────────────────────────────
# This job was submitted twice by accident on 2026-08-11 (jobs 23538728 and
# 23538734, 2.5 min apart) and both wrote the same np.memmap files concurrently.
# The transform is deterministic, so both wrote identical bytes and the final
# content survived -- but 23538728's own verification read the test split while
# 23538734 was still writing it and measured std 6.684 / corr +0.969 against
# ~0.99 / +0.32 for train and val. A training job reading at that moment would
# have consumed it silently. Refuse to run rather than race.
#
# flock -n fails IMMEDIATELY rather than queueing: waiting behind a 12-hour
# build inside a 12-hour allocation would just hit the walltime. The lock is
# released when this shell exits, including on scancel or a crash.
LOCKFILE="${DUAL_CACHE}.lock"
mkdir -p "$(dirname "$LOCKFILE")"
if command -v flock >/dev/null 2>&1; then
    exec 9>"$LOCKFILE" || { echo "ERROR: cannot open $LOCKFILE"; exit 1; }
    flock -n 9 || {
        echo "ERROR: another instance of this job already holds $LOCKFILE."
        echo "  Two concurrent builds write the same memmap files, and any reader"
        echo "  in between sees a partially-written cache. Refusing to run."
        echo "  Check: squeue -u \$USER -o '%.10i %.34j %.2t %.11M'"
        exit 1; }
    echo "holding build lock $LOCKFILE"
else
    echo "WARNING: flock not found — cannot guarantee a single builder."
    echo "  Verify no other build_dualcr_labeled_cache job is running before trusting"
    echo "  this cache: squeue -u \$USER"
fi

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

# Width check. FORMAT MATTERS, and it differs from jobs 1/2 — do not "unify" them.
# The LABELED caches are raw HEADERLESS np.memmaps (build_cr_labeled_cache.py:86,
# cache_mrral_patches.py:43), so np.load cannot report their shape and the byte
# count is all we have. The GLOBAL shards (jobs 1/2) are np.save'd with a 128-byte
# .npy header and are checked with np.load(...).shape instead.
# A bare bytes % (7*7*118*4) test would be worthless here regardless, because
# 59 * (even n) == 118 * (n/2), so any 59-channel cache with an even patch count
# passes it. Two independent references instead:
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

# ── DO THE TWO HALVES ACTUALLY DIFFER? ───────────────────────────────────────
# The byte checks above prove 118 channels; they cannot prove channels 59-117
# hold linear-CR rather than a duplicate of hull-CR. --headerless because this
# cache is a deliberately raw np.memmap (build_cr_labeled_cache.py:86), unlike
# the np.save'd global shards. Gates the fine-tune, which is afterok on this job.
echo "=== dual halves check ==="
for s in $SPLITS; do
    ${PYTHON} scripts/check_dual_cache.py --headerless \
        "${DUAL_CACHE}/mrral_${s}_patches_p7.npy" || {
        echo "ERROR: dual halves check failed for split ${s} — do NOT fine-tune."
        exit 1; }
done

echo "=== DONE ($(date)) ==="
echo "Next: sbatch scripts/hpc_finetune_dualcr.slurm"
