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
out = sys.argv[1]
# The shards are RAW headerless memmaps, so the channel count is NOT recoverable
# from the file alone. A bytes-divisible-by-7*7*118*4 test is worthless here:
# 59 * (even n) == 118 * (n/2), so every 59-channel shard with an even patch count
# passes it. shard_index.json is the authoritative record — check the declared
# width AND cross-check it against the declared per-shard patch count.
with open(os.path.join(out, 'shard_index.json')) as f:
    idx = json.load(f)
assert idx['dual'] is True, f"shard_index says dual={idx['dual']} — --dual did not take"
assert idx['continuum_removed'] is True, 'shard_index says continuum_removed=False'
n_ch = idx['n_channels']
assert n_ch == 118, f'shard_index says n_channels={n_ch}, expected 118'
P = idx['patch_size']
for rec in idx['shards']:
    path = rec['path'] if os.path.isabs(rec['path']) else os.path.join(out, rec['path'])
    want = rec['n_patches'] * P * P * n_ch * 4
    got = os.path.getsize(path)
    assert got == want, (f'{path}: {got:,} bytes but shard_index declares '
                         f'{rec["n_patches"]:,} patches x {P}x{P}x{n_ch} = {want:,}')
    br = path.replace('.npy', '_brightness.npy')
    assert os.path.exists(br), f'missing brightness sidecar {br}'
    print(f'  OK {os.path.basename(path)}  {rec["n_patches"]:,} x {P}x{P}x{n_ch}')
print(f'{len(idx["shards"])} shards verified at {n_ch} channels (dual=True)')
EOF

echo "=== dual-CR global patch cache build end: $(date) ==="
echo "Next: sbatch --dependency=afterok:\$SLURM_JOB_ID scripts/hpc_pretrain_dualcr_mae.slurm"
