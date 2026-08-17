#!/bin/bash
#SBATCH --job-name=dualcr_ft_hardneg
#SBATCH --account=sbyrne
#SBATCH --partition=gpu_standard
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64gb
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/dualcr_ft_hardneg_%j.log
#SBATCH --error=logs/dualcr_ft_hardneg_%j.log

# JOB 4 of 4 — 7-class hand-core fine-tune on the 118-channel dual-CR
# representation, warm-started from the dual-CR MAE encoder.
#
# The comparator is ft_7cls_handcore_level. Every hyperparameter below is copied
# from scripts/hpc_finetune_handcore.slurm arm 0 so the arms differ in
# REPRESENTATION only:
#   spatial_vit_aux + brightness_aux, 7-class, weight_scheme level, ASL loss,
#   lr 5e-4, encoder_lr_scale 0.001, 150 epochs, patience 40, min_delta 0.0005,
#   batch 256, embed_dim 256, 4 heads, 6 layers, patch 7.
#
# ── PLAGIOCLASE. READ BEFORE INTERPRETING ANY PLAG NUMBER FROM THIS RUN ──────
# Task 7 flagged that omitting --synth_* introduced a SECOND variable for
# plagioclase (this arm trained plag on hand labels only, the comparator did not).
# That is now RESOLVED ON THE DUAL SIDE: this job converts the MTRDR plag caches
# to 118-channel dual-CR (option (a)) and passes them, so plag sees the same
# pixels as the comparator, in this arm's representation.
#
# The EXISTING hull-CR checkpoints have NO MTRDR plag in them at all. Until
# 07547e8 (2026-08-10 09:21) --synth_* was accepted but silently dropped for
# spatial_vit_aux, and every handcore log predating it (newest 09:18) lacks any
# "Concatenating ... synthetic plag patches" line. So ft_7cls_handcore_level
# trained plagioclase on HAND LABELS ONLY — the least-separable variant, SAM
# recall 0.29 at k=1, which on its own explains plag over-firing in the floor
# test. scripts/hpc_finetune_handcore.slurm has since been fixed to inject MTRDR
# plag as hull-CR with a brightness sidecar, but the checkpoints on disk predate
# that. So:
#   * ALTERATION and BLAND — the actual hypothesis under test — never used a synth
#     cache and are comparable against ft_7cls_handcore_level today.
#   * PLAGIOCLASE is comparable only against a RE-RUN of the fixed handcore job.
#     Comparing this arm's plag to the existing checkpoint would compare
#     hand-plus-MTRDR against hand-only, not one representation against another.
#
# ── What is being tested ─────────────────────────────────────────────────────
# Upper-hull CR divides by the convex hull, and a broad convex arch IS
# approximately the hull, so hull CR destroys alteration's 1-2 µm arch (41%
# retained vs 84% for bland). Per-pixel the arch alone separates alteration from
# every other class at AUC 0.990 under linear CR against 0.856 under hull CR. If
# hull CR's flattening is what makes alteration a false attractor, the dual
# representation should reduce alteration over-prediction on the floor tiles.
#
# val_mAP is NOT comparable across data-build eras. The floor test (6 fixed
# Nili/Argyre tiles) is the leakage-immune comparator — see MODELS.md.
#
# Submit chained on the pretrain AND the labeled cache:
#   sbatch --dependency=afterok:<pretrain_job>:<labeled_cache_job> \
#          scripts/hpc_finetune_dualcr.slurm

set -uo pipefail

WORK_DIR=/groups/sbyrne/phillipsm/crism_classification          # code + env only
DATA_ROOT=/xdisk/sbyrne/phillipsm/CRISM_MRDR                    # ALL bulk data
DATA_DIR=${DATA_ROOT}/crism_classification/data
PYTHON=/groups/sbyrne/phillipsm/micromamba/envs/crism/bin/python

CKPT_DIR=${DATA_ROOT}/crism_classification/checkpoints
EMBED_DIM=256
ENCODER=${CKPT_DIR}/spatial_mae_dualcr_denoising_${EMBED_DIM}d_6l_best.pt

PARQUET=${DATA_DIR}/mrral_pixels_7cls_handcore_hardneg.parquet
CACHE_DIR=${DATA_DIR}/patch_cache_handcore_dualcr_hardneg
RUN_NAME=ft_7cls_handcore_dualcr_hardneg

# MTRDR plagioclase — the same pixels ft_7cls_handcore_level uses, in this arm's
# representation. Sources are RAW reflectance; the dual copies are built below.
# EVERY path here is under ${DATA_DIR} (xdisk). Never /groups: it filled up and
# killed two cache builds with Errno 28.
SYNTH_DIR=${DATA_DIR}/patch_cache_dualcr_synth
SYNTH_TRAIN_RAW=${DATA_DIR}/patch_cache/mtrdr_plag_patches_p7.npy
SYNTH_TRAIN_ROWS=${DATA_DIR}/patch_cache/mtrdr_plag_rows.parquet
SYNTH_VAL_RAW=${DATA_DIR}/contrastive/extra_plag_roi/patches.npy
SYNTH_VAL_ROWS=${DATA_DIR}/mtrdr_plag_rows.parquet
SYNTH_TRAIN=${SYNTH_DIR}/mtrdr_plag_patches_p7_dual.npy
SYNTH_VAL=${SYNTH_DIR}/extra_plag_roi_patches_dual.npy

cd "$WORK_DIR" || { echo "ERROR: cannot cd $WORK_DIR"; exit 1; }
mkdir -p logs

# Checkpoints and all outputs on xdisk, never /groups.
if [ ! -f config.local.yaml ]; then
    printf 'data_root: %s\n' "${DATA_ROOT}" > config.local.yaml
fi
mkdir -p "$CKPT_DIR"

# ── PREREQUISITES ────────────────────────────────────────────────────────────
for f in "$ENCODER" "$PARQUET"; do
    [ -f "$f" ] || { echo "ERROR: missing $f"; exit 1; }
done
for split in train val; do
    for f in "${CACHE_DIR}/mrral_${split}_patches_p7.npy" \
             "${CACHE_DIR}/mrral_${split}_patches_p7_brightness.npy"; do
        [ -f "$f" ] || {
            echo "ERROR: dual-CR patch cache incomplete — missing $f"
            echo "Build it: sbatch scripts/hpc_build_dualcr_labeled_cache.slurm"
            exit 1; }
    done
done

# The dataset's byte-exact size guard would catch a 59-channel cache, and a
# 59-wide encoder warm-start would raise on the band_embed copy_ — but both only
# after SLURM has handed over a GPU. Check the two widths up front instead.
$PYTHON - "$ENCODER" "$CACHE_DIR" "$PARQUET" <<'EOF' || { echo "ERROR: encoder/cache width preflight failed"; exit 1; }
import os, sys, torch
import pandas as pd
enc_path, cache_dir, parquet = sys.argv[1], sys.argv[2], sys.argv[3]
ck = torch.load(enc_path, map_location='cpu', weights_only=False)
state = ck.get('encoder_state') or ck.get('mae_state') or ck
KEY = 'band_embed.weight'
hits = [k for k in state if k.endswith(KEY)]
assert hits, f'no *{KEY} in {enc_path}; keys: {sorted(state)[:6]}'
# band_embed is nn.Linear(n_bands, embed_dim) -> weight (embed_dim, n_bands), so
# shape[-1] IS the channel count. A 59-wide encoder would raise on the copy_ into
# a 118-wide param, but only after SLURM has handed over the GPU.
w = state[hits[0]]
assert w.shape[-1] == 118, f'encoder is {w.shape[-1]}-channel, not 118 (dual-CR)'
print(f'  encoder OK: {hits[0]} {tuple(w.shape)}')
# The LABELED cache is a raw HEADERLESS np.memmap (build_cr_labeled_cache.py:86),
# so np.load cannot report its shape and byte count is all we have — unlike the
# np.save'd GLOBAL shards in jobs 1/2, which are checked via np.load(...).shape.
# Different formats on purpose; do not unify the two checks.
# A bare bytes % (7*7*118*4) test proves nothing either, because
# 59 * (even n) == 118 * (n/2). The parquet row count is the real reference, and
# it is exactly what CRISMSpectralPatchDataset will demand at startup.
df = pd.read_parquet(parquet, columns=['split'])
for s in ('train', 'val'):
    n = os.path.getsize(os.path.join(cache_dir, f'mrral_{s}_patches_p7.npy'))
    rows = int((df['split'] == s).sum())
    want = rows * 7 * 7 * 118 * 4
    assert n == want, (f'{s} cache is {n:,} bytes but the parquet has {rows:,} '
                       f'rows, needing exactly {want:,} at 118 channels')
    print(f'  cache OK: {s} {rows:,} patches x 7x7x118')
EOF

# ── MTRDR PLAGIOCLASE AS DUAL-CR ─────────────────────────────────────────────
# The source caches are RAW reflectance. SyntheticPatchDataset serves its .npy
# verbatim, so they must be converted, not merely passed. Built once and reused;
# output lives on xdisk under ${SYNTH_DIR}.
for f in "$SYNTH_TRAIN_RAW" "$SYNTH_TRAIN_ROWS" "$SYNTH_VAL_RAW" "$SYNTH_VAL_ROWS"; do
    [ -f "$f" ] || { echo "ERROR: missing MTRDR plag source $f"; exit 1; }
done
#
# The existence check requires the BRIGHTNESS SIDECAR as well as the patches.
# This job runs --brightness_aux, so SyntheticPatchDataset needs the sidecar to
# return the same 4-tuple as the labeled dataset. Converted caches built before
# 2026-08-11 are already on the HPC WITHOUT a sidecar, so a patches-only check
# would report "present", skip the rebuild, and the sidecar would never appear.
#
# The build is ATOMIC: convert into a mktemp -d INSIDE ${SYNTH_DIR} (same
# filesystem, so the rename is a true atomic rename rather than a copy), then
# move the finished files into place. This job is not an array job today, but it
# shares ${SYNTH_DIR} semantics with the --array=0-1 handcore job, where two
# concurrent tasks could both miss an "if absent" check and write one path.
mkdir -p "$SYNTH_DIR"
for pair in "${SYNTH_TRAIN_RAW}|${SYNTH_TRAIN}" "${SYNTH_VAL_RAW}|${SYNTH_VAL}"; do
    raw=${pair%%|*}
    dual=${pair##*|}
    dual_bright="${dual%.npy}_brightness.npy"
    if [ -f "$dual" ] && [ -f "$dual_bright" ]; then
        echo "  dual plag cache + brightness sidecar present: $dual"
        continue
    fi
    echo "  building dual plag cache + brightness sidecar: $dual"
    tmpdir=$(mktemp -d "${SYNTH_DIR}/.build.XXXXXXXX") \
        || { echo "ERROR: mktemp -d failed under $SYNTH_DIR"; exit 1; }
    $PYTHON scripts/convert_synth_cache_representation.py \
        --input "$raw" --output "${tmpdir}/$(basename "$dual")" --mode dual \
        || { echo "ERROR: dual plag conversion failed for $raw"; rm -rf "$tmpdir"; exit 1; }
    # Sidecar first, patches last: nothing keys on the sidecar alone, so a crash
    # between the two renames leaves the pair detectably incomplete and the next
    # run rebuilds it.
    mv -f "${tmpdir}/$(basename "$dual_bright")" "$dual_bright" \
        && mv -f "${tmpdir}/$(basename "$dual")" "$dual" \
        || { echo "ERROR: could not install $dual"; rm -rf "$tmpdir"; exit 1; }
    rm -rf "$tmpdir"
done

# Preflight the plag caches on their TRUE array shape.
$PYTHON - "$SYNTH_TRAIN" "$SYNTH_TRAIN_ROWS" "$SYNTH_VAL" "$SYNTH_VAL_ROWS" <<'EOF' || { echo "ERROR: dual plag cache preflight failed"; exit 1; }
import os
import sys
import numpy as np
import pandas as pd
from data.dataset import synth_brightness_path
# np.load(..., mmap_mode='r').shape, never a byte-size computation. These caches
# ARE np.save'd/open_memmap'd (128-byte header), so getsize() arithmetic is off by
# 128 — and byte counts cannot distinguish 118 channels from 59 anyway, because
# 59 * (2k) == 118 * k. Two vacuous byte-arithmetic preflights were already caught
# in this task; do not reintroduce one.
for cache, parquet in zip(sys.argv[1::2], sys.argv[2::2]):
    shape = np.load(cache, mmap_mode='r').shape
    assert len(shape) == 4 and shape[1:3] == (7, 7), f'{cache}: bad shape {shape}'
    assert shape[-1] == 118, (
        f'{cache} is {shape[-1]}-channel, not 118 (dual-CR): shape {shape}. '
        f'Rebuild with scripts/convert_synth_cache_representation.py --mode dual')
    rows = len(pd.read_parquet(parquet, columns=['split']))
    assert shape[0] == rows, (
        f'{cache} has {shape[0]:,} patches but {parquet} has {rows:,} rows; '
        f'SyntheticPatchDataset asserts they match row-for-row')
    # This job runs --brightness_aux, so the labeled dataset yields
    # (patch, brightness, label, weight). Without this sidecar the synth half
    # yields a 3-tuple and default_collate kills the job on the first mixed
    # batch with "each element in list of batch should be of equal size".
    # The sidecar is 59-band brightness of the RAW source and is 3-D (n,7,7)
    # regardless of the 118-channel patch width.
    bpath = synth_brightness_path(cache)
    assert os.path.exists(bpath), (
        f'{cache} has no brightness sidecar {bpath}. This job runs '
        f'--brightness_aux, so SyntheticPatchDataset cannot serve it. Delete '
        f'{cache} and re-run: the converter writes the sidecar alongside.')
    b = np.load(bpath, mmap_mode='r')
    assert b.shape == (shape[0], 7, 7), (
        f'{bpath} has shape {b.shape}, expected {(shape[0], 7, 7)} — one (7,7) '
        f'brightness map per patch, aligned row-for-row with {cache}')
    print(f'  plag cache OK: {cache} {shape} vs {rows:,} parquet rows '
          f'(+ brightness sidecar {b.shape})')
EOF

# Refuse to train on spectra we have not checked. Cheap next to a wasted GPU day.
echo "=== spectra audit ==="
$PYTHON scripts/audit_spectra_quality.py "$PARQUET" --fail_tile_over 5 || {
    echo "ERROR: $PARQUET failed the spectra audit. Refusing to train."; exit 1; }

# ── RESUME (OPT-IN) ──────────────────────────────────────────────────────────
# --checkpoint_every 5 writes ${CKPT_DIR}/${RUN_NAME}_resume.pt every 5 epochs:
# weights + optimizer + LR scheduler + best/patience bookkeeping, i.e. everything
# needed to CONTINUE this run in a second job. 150 epochs does not fit the 24h
# walltime (runs reach ~117-123), and ft_7cls_handcore_reviewup was CANCELLED DUE
# TO TIME LIMIT at 123/150.
#
# Resume is OPT-IN — `RESUME=1 sbatch scripts/hpc_finetune_dualcr.slurm`. Plain
# sbatch ALWAYS starts fresh. Auto-resuming whenever the file happens to exist
# would silently continue a stale experiment when a new one was intended; and
# RESUME=1 with no file present fails here rather than quietly starting over,
# which would burn another 24h producing a run nobody asked for.
#
# --epochs is TOTAL across jobs, not additional: 150 stays 150, so a job resumed
# at 123 runs 27 more epochs.
RESUME=${RESUME:-0}
RESUME_CKPT=${CKPT_DIR}/${RUN_NAME}_resume.pt
INIT_ARGS=(--pretrain_ckpt "$ENCODER")
if [ "$RESUME" = "1" ]; then
    [ -f "$RESUME_CKPT" ] || {
        echo "ERROR: RESUME=1 but $RESUME_CKPT does not exist."
        echo "       Submit without RESUME=1 to start this run from scratch."
        exit 1; }
    # --resume and --pretrain_ckpt are mutually exclusive (train.py refuses the
    # pair): the resume file carries its own encoder weights, further trained.
    INIT_ARGS=(--resume "$RESUME_CKPT")
    echo "RESUMING $RUN_NAME from $RESUME_CKPT (encoder comes from the resume file)"
else
    echo "Fresh start for $RUN_NAME (use RESUME=1 sbatch ... to continue a killed run)"
fi

echo "=== $RUN_NAME start $(date) ==="

$PYTHON -u scripts/train.py \
    --model spatial_vit_aux \
    --run_name "$RUN_NAME" \
    "${INIT_ARGS[@]}" \
    --seven_class \
    --continuum_removed --cache_is_cr --dual_cr --brightness_aux \
    --mrral_parquets "$PARQUET" \
    --patch_cache_dir "$CACHE_DIR" \
    --weight_scheme level \
    --synth_train_cache "$SYNTH_TRAIN" \
    --synth_train_parquet "$SYNTH_TRAIN_ROWS" \
    --synth_val_cache "$SYNTH_VAL" \
    --synth_val_parquet "$SYNTH_VAL_ROWS" \
    --lr 5e-4 --encoder_lr_scale 0.001 \
    --epochs 150 --patience 40 --min_delta 0.0005 \
    --checkpoint_every 5 \
    --batch_size 256 \
    --embed_dim ${EMBED_DIM} --n_heads 4 --n_layers 6 \
    --patch_size 7 \
    --asl_loss || { echo "ERROR: fine-tune failed"; exit 1; }

echo "=== $RUN_NAME end $(date) ==="
echo "Floor-test with (--dual_cr is REQUIRED; classify_tile_supervised verifies the"
echo "checkpoint's own first-layer width against it and aborts on a mismatch):"
# No vocab flag: classify_tile_supervised._set_n_classes reads head.weight and a
# 7-wide head auto-selects the 7-class vocab. --seven_class does not exist there.
echo "  CLASSIFY_EXTRA_ARGS=\"--continuum_removed --dual_cr --brightness_aux --embed_dim ${EMBED_DIM}\" \\"
echo "    bash scripts/floor_test.sh ${CKPT_DIR}/${RUN_NAME}_best.pt ${RUN_NAME}"
