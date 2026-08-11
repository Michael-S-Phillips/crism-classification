#!/bin/bash
#SBATCH --job-name=build_dualcr_global_cache
#SBATCH --account=sbyrne
#SBATCH --partition=standard
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32gb
#SBATCH --time=0-04:00:00
#SBATCH --output=logs/build_dualcr_global_cache_%j.log
#SBATCH --error=logs/build_dualcr_global_cache_%j.log

# JOB 1 of 4 — global MAE pre-training patch cache in the DUAL-CR representation:
# 118 channels, upper-hull CR in 0-58 ⊕ linear CR in 59-117, each block divided by
# its own global std from data/mrral_cr_scales.json.
#
# Why linear CR at all: upper-hull CR divides by the convex hull, and a broad
# convex arch IS approximately the hull, so hull CR destroys alteration's 1-2 µm
# arch (41% retained vs 84% for bland). Per-pixel the arch alone separates
# alteration from every other class at AUC 0.990 under linear CR against 0.856
# under hull CR. This cache exists to test whether that flattening is what makes
# alteration a false attractor.
#
# ALL bulk output goes on /xdisk. /groups filled up and killed two CR patch-cache
# builds with "No space left on device"; the dual caches add ~90 GB, and the
# global dual cache alone is 2x the 59-channel one.
#
# Submit:
#   G=$(sbatch --parsable scripts/hpc_build_dualcr_global_cache.slurm)
#   P=$(sbatch --parsable --dependency=afterok:$G scripts/hpc_pretrain_dualcr_mae.slurm)
#   L=$(sbatch --parsable scripts/hpc_build_dualcr_labeled_cache.slurm)
#   F=$(sbatch --parsable --dependency=afterok:$P:$L scripts/hpc_finetune_dualcr.slurm)
# The labeled cache has NO dependency on the pretrain — they are independent.

set -uo pipefail

WORK_DIR=/groups/sbyrne/phillipsm/crism_classification          # code + env only
DATA_ROOT=/xdisk/sbyrne/phillipsm/CRISM_MRDR                    # ALL bulk data
DATA_DIR=${DATA_ROOT}/crism_classification/data
PYTHON=/groups/sbyrne/phillipsm/micromamba/envs/crism/bin/python

# Top-level xdisk, matching the existing global caches
# (/xdisk/sbyrne/phillipsm/crism_patch_cache{,_cr}).
OUTPUT=/xdisk/sbyrne/phillipsm/crism_patch_cache_dualcr

cd "$WORK_DIR" || { echo "ERROR: cannot cd $WORK_DIR"; exit 1; }
mkdir -p logs

if [ ! -f config.local.yaml ]; then
    printf 'data_root: %s\n' "${DATA_ROOT}" > config.local.yaml
fi

# ── Resolve the mrral tiles by PROBING, never by assuming a layout ────────────
# On HPC the tiles are FLAT at data_root; on the workstation they sit under mc*/.
# build_global_patch_cache.py does its own probing, so this is a prerequisite
# check only — but it must not hard-code either layout.
n_flat=$(find "$DATA_ROOT" -maxdepth 1 -name 't*mrral*.hdr' 2>/dev/null | wc -l)
n_mc=$(find "$DATA_ROOT" -maxdepth 2 -path '*/mc*' -name 't*mrral*.hdr' 2>/dev/null | wc -l)
n_tiles=$(( n_flat + n_mc ))
echo "mrral tiles found under $DATA_ROOT: flat=$n_flat, mc*/=$n_mc, total=$n_tiles"
if [ "$n_tiles" -eq 0 ]; then
    echo "ERROR: no t*mrral*.hdr under $DATA_ROOT (checked flat and mc*/)."
    echo "  The tile-refresh rsync must land before any cache build; building"
    echo "  from truncated/absent tiles bakes zero-fill corruption into the cache"
    echo "  and every downstream pretrain."
    exit 1
fi

# The dual cache is 2x the 59-channel one. Fail now rather than 3 h in.
avail_gb=$(df -BG --output=avail "$(dirname "$OUTPUT")" 2>/dev/null | tail -1 | tr -dc '0-9')
echo "free space on $(dirname "$OUTPUT"): ${avail_gb:-unknown} GB"
if [ -n "${avail_gb:-}" ] && [ "$avail_gb" -lt 250 ]; then
    echo "ERROR: need ~250 GB free for the 118-channel global cache, have ${avail_gb} GB"
    exit 1
fi

echo "=== dual-CR global patch cache build start: $(date) ==="
echo "    output: $OUTPUT  (118 channels)"

${PYTHON} -u scripts/build_global_patch_cache.py \
    --continuum_removed \
    --dual \
    --output "$OUTPUT" \
    --workers "${SLURM_CPUS_PER_TASK:-8}" \
    --seed 42 \
    --patches_per_tile_target 2834 \
    --patches_per_shard 100000 || {
        echo "ERROR: dual-CR global cache build failed"; exit 1; }

# ── Verify the shards are actually 118 channels wide ─────────────────────────
shards=$(ls -1 "$OUTPUT"/global_patches_*.npy 2>/dev/null | wc -l)
[ "$shards" -gt 0 ] || { echo "ERROR: no shards written to $OUTPUT"; exit 1; }
echo "shards written: $shards"

${PYTHON} - "$OUTPUT" <<'EOF' || { echo "ERROR: shard width check failed"; exit 1; }
import json, os, sys
import numpy as np
out = sys.argv[1]
# ── How to check a GLOBAL shard's width (and how NOT to) ─────────────────────
# These shards are written with np.save (build_global_patch_cache.py:302), so
# they carry a 128-byte .npy header and np.load can report the true shape. Use
# that. Do NOT compute the width from the byte count:
#   * getsize() == n_patches*7*7*118*4 FAILS on a correct cache, because of the
#     128-byte header.
#   * getsize() % (7*7*118*4) == 0 PASSES on a wrong one, because
#     59 * (even n) == 118 * (n/2).
# Both mistakes were made and caught during Task 7. np.load(mmap_mode='r').shape
# is header-agnostic and subsumes the width check, so it replaces both.
# NB the LABELED caches (jobs 3/4) are genuinely headerless np.memmaps
# (build_cr_labeled_cache.py:86, cache_mrral_patches.py:43) and must be checked
# by byte count instead. The two formats are different on purpose — do not
# "unify" these checks.
with open(os.path.join(out, 'shard_index.json')) as f:
    idx = json.load(f)
assert idx['dual'] is True, f"shard_index says dual={idx['dual']} — --dual did not take"
assert idx['continuum_removed'] is True, 'shard_index says continuum_removed=False'
n_ch = idx['n_channels']
assert n_ch == 118, f'shard_index says n_channels={n_ch}, expected 118'
P = idx['patch_size']
for rec in idx['shards']:
    path = rec['path'] if os.path.isabs(rec['path']) else os.path.join(out, rec['path'])
    shape = np.load(path, mmap_mode='r').shape
    assert shape == (rec['n_patches'], P, P, n_ch), (
        f'{path}: shape {shape} but shard_index declares '
        f'{(rec["n_patches"], P, P, n_ch)}')
    br = path.replace('.npy', '_brightness.npy')
    assert os.path.exists(br), f'missing brightness sidecar {br}'
    print(f'  OK {os.path.basename(path)}  {shape}')
print(f'{len(idx["shards"])} shards verified at {n_ch} channels (dual=True)')
EOF

# ── DO THE TWO HALVES ACTUALLY DIFFER? ───────────────────────────────────────
# Everything above confirms the cache is 118 channels wide and self-consistent.
# None of it confirms that channels 59-117 hold linear-CR rather than a second
# copy of hull-CR. That failure yields finite floats, the right shape, and a
# shard_index saying dual=True -- it would survive every check above, and the
# experiment would then read as the hypothesis being FALSIFIED rather than as a
# build bug. This MUST gate the pretrain: the chain is afterok, so a downstream
# job starts the moment this one exits 0.
echo "=== dual halves check ==="
${PYTHON} scripts/check_dual_cache.py "$OUTPUT" || {
    echo "ERROR: dual halves check failed on $OUTPUT — do NOT pretrain on this cache."
    exit 1; }

echo "=== dual-CR global patch cache build end: $(date) ==="
echo "Next: sbatch --dependency=afterok:\$SLURM_JOB_ID scripts/hpc_pretrain_dualcr_mae.slurm"
