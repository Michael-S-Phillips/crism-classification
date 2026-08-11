#!/bin/bash
#SBATCH --job-name=dualcr_mae_pretrain
#SBATCH --account=sbyrne
#SBATCH --partition=gpu_standard
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64gb
#SBATCH --time=0-04:00:00
#SBATCH --output=logs/dualcr_mae_pretrain_%j.log
#SBATCH --error=logs/dualcr_mae_pretrain_%j.log

# JOB 2 of 4 — denoising spatial-spectral MAE pretrain on the 118-channel dual-CR
# global cache. embed_dim 256 / 6 layers, matching the encoder the 7-class
# hand-core fine-tune warm-starts from, so the dual arm differs from
# ft_7cls_handcore_level in REPRESENTATION ONLY.
#
# Three things here are load-bearing:
#   --n_bands 118          the encoder's band_embed width. A cache/-n_bands
#                          mismatch raises a RuntimeError, so this fails loudly.
#   --continuum_removed    without it the loader keeps normalize=True and
#                          per-patch z-scores the dual patch. The two blocks sit
#                          ~7.5x apart in mean, that offset dominates the patch
#                          std, and the real within-block signal shrinks ~4x. The
#                          run would complete and learn almost nothing. The script
#                          now refuses --n_bands 118 without this flag.
#   --n_channel_blocks 2   logs the per-block reconstruction loss. Diagnostic
#                          only (equal-sized blocks => same loss and gradient as
#                          pooled); it reveals a cache written un-standardised or
#                          a stale data/mrral_cr_scales.json. If the two block
#                          losses differ by more than ~2x on epoch 1, STOP: the
#                          cache is not per-block standardised.
#
# CrismNoiseAugmentation auto-detects the dual representation from n_bands == 118
# and rescales its sigmas by 1/hull_std. No flag needed and none should be added.
#
# SIDE EFFECT: this job REWRITES config.local.yaml unconditionally (the value is
# run-specific; a stale global_patch_cache_dir pointing at the 59-channel cache
# would pretrain the wrong representation). It leaves global_patch_cache_dir
# pointing at the DUAL cache, so a later 59-band job must reset it. Job 3 runs in
# parallel but only reads data_root, which is identical either way.
#
# Submit chained on the global cache:
#   sbatch --dependency=afterok:<global_cache_job> scripts/hpc_pretrain_dualcr_mae.slurm

set -uo pipefail

WORK_DIR=/groups/sbyrne/phillipsm/crism_classification          # code + env only
DATA_ROOT=/xdisk/sbyrne/phillipsm/CRISM_MRDR                    # ALL bulk data
DATA_DIR=${DATA_ROOT}/crism_classification/data
PYTHON=/groups/sbyrne/phillipsm/micromamba/envs/crism/bin/python

DUAL_CACHE=/xdisk/sbyrne/phillipsm/crism_patch_cache_dualcr
CKPT_DIR=${DATA_ROOT}/crism_classification/checkpoints
EMBED_DIM=256
RUN_NAME=spatial_mae_dualcr_denoising_${EMBED_DIM}d_6l

cd "$WORK_DIR" || { echo "ERROR: cannot cd $WORK_DIR"; exit 1; }
mkdir -p logs

# global_patch_cache_dir is NOT derived from data_root by config_loader, so it
# must be written explicitly or pretrain_spatial_mae_denoising.py raises KeyError.
# Written every run (not "if absent") because the value is run-specific: a stale
# config.local.yaml pointing at the 59-channel cache would make --n_bands 118
# raise, or worse, silently pretrain the wrong representation if the widths ever
# coincided. Checkpoints go on xdisk, never /groups.
cat > config.local.yaml <<EOF
data_root: ${DATA_ROOT}
global_patch_cache_dir: ${DUAL_CACHE}
checkpoints_dir: ${CKPT_DIR}
EOF
mkdir -p "$CKPT_DIR"

if [ ! -d "$DUAL_CACHE" ] || [ -z "$(ls -A "$DUAL_CACHE"/global_patches_*.npy 2>/dev/null)" ]; then
    echo "ERROR: dual-CR global cache missing at $DUAL_CACHE"
    echo "Build it first: sbatch scripts/hpc_build_dualcr_global_cache.slurm"
    exit 1
fi

# Cheap independent re-check of the shard width — the pretrain would raise on a
# mismatch anyway, but not before burning GPU allocation on startup.
${PYTHON} - "$DUAL_CACHE" <<'EOF' || { echo "ERROR: cache is not the 118-channel dual cache"; exit 1; }
import json, os, sys
import numpy as np
# Global shards are np.save'd (build_global_patch_cache.py:302) so they carry a
# 128-byte .npy header and np.load reports the true shape — use it. A byte-count
# test is wrong twice over here: == n*7*7*118*4 fails on a CORRECT cache (header),
# and % (7*7*118*4) passes on a WRONG one (59 * even n == 118 * n/2).
# The LABELED caches (jobs 3/4) ARE headerless np.memmaps and are correctly
# checked by byte count there. Different formats on purpose; do not unify.
out = sys.argv[1]
with open(os.path.join(out, 'shard_index.json')) as f:
    idx = json.load(f)
assert idx['dual'] is True, f"shard_index says dual={idx['dual']}"
assert idx['continuum_removed'] is True, 'shard_index says continuum_removed=False'
assert idx['n_channels'] == 118, f"n_channels={idx['n_channels']}, expected 118"
P, n_ch = idx['patch_size'], idx['n_channels']
for rec in idx['shards']:
    path = rec['path'] if os.path.isabs(rec['path']) else os.path.join(out, rec['path'])
    shape = np.load(path, mmap_mode='r').shape
    assert shape == (rec['n_patches'], P, P, n_ch), (
        f'{path}: shape {shape}, shard_index declares {(rec["n_patches"], P, P, n_ch)}')
print(f'{len(idx["shards"])} shards verified at {n_ch} channels (dual=True)')
EOF

echo "=== $RUN_NAME start $(date) ==="

${PYTHON} -u scripts/pretrain_spatial_mae_denoising.py \
    --continuum_removed \
    --n_bands 118 \
    --n_channel_blocks 2 \
    --epochs 200 \
    --warmup 10 \
    --batch_size 1024 \
    --patches_per_epoch 200000 \
    --num_workers "${SLURM_CPUS_PER_TASK:-6}" \
    --embed_dim ${EMBED_DIM} \
    --n_heads 4 \
    --n_layers 6 \
    --decoder_dim 64 \
    --decoder_layers 2 \
    --mask_ratio 0.75 \
    --sigma_gauss 0.0087 \
    --sigma_spike 0.0058 \
    --sigma_column 0.0049 \
    --spike_center_band 15 \
    --spike_fwhm_bands 3.0 \
    --run_name ${RUN_NAME} || { echo "ERROR: pretrain failed"; exit 1; }

BEST=${CKPT_DIR}/${RUN_NAME}_best.pt
[ -f "$BEST" ] || { echo "ERROR: no best checkpoint at $BEST"; exit 1; }
echo "encoder: $BEST"
echo "=== $RUN_NAME end $(date) ==="
echo "Check the logged blocks=[hull, linear] losses: a >2x gap on epoch 1 means"
echo "the cache is not per-block standardised — do NOT fine-tune from it."
