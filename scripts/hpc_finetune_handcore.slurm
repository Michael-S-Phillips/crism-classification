#!/bin/bash
#SBATCH --job-name=crism_handcore_ft
#SBATCH --account=sbyrne
#SBATCH --partition=gpu_standard
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64gb
#SBATCH --time=1-00:00:00
#SBATCH --array=0-1
#SBATCH --output=logs/handcore_ft_%a_%j.log
#SBATCH --error=logs/handcore_ft_%a_%j.log

# Fine-tune on the HAND-CORE parquet, with plagioclase leaning on MTRDR.
#
# Prereq: scripts/hpc_build_handcore_data.slurm has produced
#   ${DATA_DIR}/mrral_pixels_7cls_handcore.parquet, and a patch cache exists
#   for it (ALL splits — an incomplete CR cache now fails fast at startup
#   rather than silently training on raw patches).
#
# ── Why plagioclase is configured the way it is ──────────────────────────────
# A spectral-angle confusion matrix over the hand-core classes (2026-08-08,
# scripts/sam_confusion_matrix.py) found:
#     plag(MTRDR)  recall 0.87 (k=1) -> 0.97 (k=5), self-angle 1.09 deg
#     plag(hand)   recall 0.29 (k=1) -> 0.41 (k=5), confused with alteration
#                  and lcp at 0.22 each
# Allowing 5 endmembers per class rescues `junk` (0.50 -> 0.81), so the method
# does detect multi-modality. Hand plagioclase does not recover, which means it
# is genuinely entangled with other classes rather than several clean modes.
# Under the hand-core policy plag is 100% hand-labelled — there are zero plag
# confirms in either review session — so the class is composed almost entirely
# of its least separable variant.
#
# So: train plagioclase on the LARGER MTRDR pool and validate on a DISJOINT one.
#   train:  data/patch_cache/mtrdr_plag_rows.parquet   8,671 rows, 18 tiles
#   val:    data/mtrdr_plag_rows.parquet                 109 val rows, 31 tiles
# Verified 2026-08-08: the two sets share ZERO pixels and ZERO tiles, so this
# adds no spatial leakage. The 8,671-row pool was already on disk with its own
# patch cache and referenced by nothing.
#
# NOTE the synth-train injection now filters split=='train' (fix e83a827).
# Before that fix it served every row in the parquet, so a synth set carrying
# its own val rows trained on them AND validated on them.
#
# ── 2026-08-11: no earlier run of this job used MTRDR plag AT ALL ────────────
# Until 07547e8 (2026-08-10 09:21) --synth_* was accepted but silently DROPPED
# for spatial_vit_aux. Every handcore log before that — the newest is 09:18 —
# lacks any "Concatenating ... synthetic plag patches" line. So
# ft_7cls_handcore_level and _reviewup trained plagioclase on HAND LABELS ONLY,
# i.e. exactly the least-separable variant described above (SAM recall 0.29 at
# k=1, and unlike junk it does not recover at k=5). That alone explains plag
# over-firing everywhere in the floor test; no further mechanism is needed.
#
# A SECOND, latent bug sat behind it: the two source caches are raw reflectance
# (p50 ~0.19) while hull-CR is bounded [0,1] and centres at 0.934, and
# SyntheticPatchDataset serves its .npy VERBATIM. Once 07547e8 made the
# injection real, that would have put plagioclase — and only plagioclase — at
# ~4x the level of every other class. It never reached a trained model, because
# the first run after the forwarding fix crashed first (see below).
#
# Both are fixed here: the caches are converted to hull-CR once into
# ${SYNTH_DIR} on xdisk (row order and count preserved, so both *_ROWS parquets
# are unchanged), and SyntheticPatchDataset now REFUSES a representation
# mismatch rather than training through it.
#
# THIRD, exposed by the first real injection (2026-08-11): this job runs
# --brightness_aux, so the labeled dataset yields (patch, brightness, label,
# weight) while SyntheticPatchDataset yielded a 3-tuple — default_collate
# rejects the mixed batch with "each element in list of batch should be of equal
# size". Hence the brightness sidecar built alongside the patches below.
#
# ft_7cls_handcore_level / _reviewup are therefore not plagioclase comparators
# for any run that injects MTRDR plag. Re-run this job to get one.
#
# Submit:  sbatch scripts/hpc_finetune_handcore.slurm

WORK_DIR=/groups/sbyrne/phillipsm/crism_classification
DATA_ROOT=/xdisk/sbyrne/phillipsm/CRISM_MRDR
PYTHON=/groups/sbyrne/phillipsm/micromamba/envs/crism/bin/python

CKPT_DIR=${DATA_ROOT}/crism_classification/checkpoints
DATA_DIR=${DATA_ROOT}/crism_classification/data
ENCODER=${CKPT_DIR}/spatial_mae_cr_denoising_256d_6l_best.pt
EMBED_DIM=256

PARQUET=${DATA_DIR}/mrral_pixels_7cls_handcore.parquet
CACHE_DIR=${DATA_DIR}/patch_cache_handcore_cr

# Plagioclase: big disjoint pool for train, small disjoint pool for val.
# The sources are RAW; the hull-CR copies are what this job actually trains on.
MTRDR_TRAIN_RAW=${DATA_DIR}/patch_cache/mtrdr_plag_patches_p7.npy
MTRDR_VAL_RAW=${DATA_DIR}/contrastive/extra_plag_roi/patches.npy
MTRDR_TRAIN_ROWS=${DATA_DIR}/patch_cache/mtrdr_plag_rows.parquet
MTRDR_VAL_ROWS=${DATA_DIR}/mtrdr_plag_rows.parquet

SYNTH_DIR=${DATA_DIR}/patch_cache_hullcr_synth
MTRDR_TRAIN_PATCHES=${SYNTH_DIR}/mtrdr_plag_patches_p7_hull.npy
MTRDR_VAL_PATCHES=${SYNTH_DIR}/extra_plag_roi_patches_hull.npy

# Arm 0 is the clean baseline: ONE variable at a time. Arm 1 changes only the
# weight scheme, so any difference is attributable to hand-vs-review weighting
# rather than to the data rebuild.
RUN_NAMES=("ft_7cls_handcore_level" "ft_7cls_handcore_reviewup")
SCHEMES=("level" "review_up")
RUN_NAME=${RUN_NAMES[$SLURM_ARRAY_TASK_ID]}
SCHEME=${SCHEMES[$SLURM_ARRAY_TASK_ID]}

cd "$WORK_DIR"
mkdir -p logs
if [ ! -f config.local.yaml ]; then
    printf 'data_root: %s\n' "${DATA_ROOT}" > config.local.yaml
fi

# ── PREREQUISITES ────────────────────────────────────────────────────────────
for f in "$ENCODER" "$PARQUET" "$MTRDR_TRAIN_RAW" "$MTRDR_TRAIN_ROWS" \
         "$MTRDR_VAL_RAW" "$MTRDR_VAL_ROWS"; do
    [ -f "$f" ] || { echo "ERROR: missing $f"; exit 1; }
done

# ── MTRDR PLAGIOCLASE AS HULL-CR ─────────────────────────────────────────────
# Built once and reused. Output on xdisk under ${SYNTH_DIR}, never /groups.
#
# The existence check requires the BRIGHTNESS SIDECAR as well as the patches.
# This job runs --brightness_aux, so SyntheticPatchDataset needs the sidecar to
# return the same 4-tuple as the labeled dataset. Converted caches built before
# 2026-08-11 are already on the HPC WITHOUT a sidecar, so a patches-only check
# would report "present", skip the rebuild, and the sidecar would never appear.
#
# The build is ATOMIC. This job is --array=0-1, so two tasks reach this loop at
# once; the previous "if absent, convert straight to $hull" let both miss the
# check and write the same path concurrently, interleaving two writers into one
# memmap. Each task now converts into its own mktemp -d INSIDE ${SYNTH_DIR}
# (same filesystem, so the rename is a true atomic rename, not a copy) and moves
# the finished files into place. The transform is deterministic, so whichever
# task wins the race every reader still sees one complete, correct pair.
mkdir -p "$SYNTH_DIR"
for pair in "${MTRDR_TRAIN_RAW}|${MTRDR_TRAIN_PATCHES}" \
            "${MTRDR_VAL_RAW}|${MTRDR_VAL_PATCHES}"; do
    raw=${pair%%|*}
    hull=${pair##*|}
    hull_bright="${hull%.npy}_brightness.npy"
    if [ -f "$hull" ] && [ -f "$hull_bright" ]; then
        echo "  hull-CR plag cache + brightness sidecar present: $hull"
        continue
    fi
    echo "  building hull-CR plag cache + brightness sidecar: $hull"
    tmpdir=$(mktemp -d "${SYNTH_DIR}/.build.XXXXXXXX") \
        || { echo "ERROR: mktemp -d failed under $SYNTH_DIR"; exit 1; }
    $PYTHON scripts/convert_synth_cache_representation.py \
        --input "$raw" --output "${tmpdir}/$(basename "$hull")" --mode hull \
        || { echo "ERROR: hull-CR plag conversion failed for $raw"; rm -rf "$tmpdir"; exit 1; }
    # Sidecar first, patches last: nothing keys on the sidecar alone, so a crash
    # between the two renames leaves the pair detectably incomplete and the next
    # run rebuilds it.
    mv -f "${tmpdir}/$(basename "$hull_bright")" "$hull_bright" \
        && mv -f "${tmpdir}/$(basename "$hull")" "$hull" \
        || { echo "ERROR: could not install $hull"; rm -rf "$tmpdir"; exit 1; }
    rm -rf "$tmpdir"
done

# Preflight the plag caches on their TRUE array shape, never a byte count.
# These are np.save'd (128-byte header), so getsize() arithmetic is off by 128 —
# and byte counts cannot distinguish 59 channels from 118 anyway, since
# 59 * (2k) == 118 * k. Two vacuous byte-arithmetic preflights were already
# caught in this work; do not reintroduce one.
$PYTHON - "$MTRDR_TRAIN_PATCHES" "$MTRDR_TRAIN_ROWS" \
         "$MTRDR_VAL_PATCHES" "$MTRDR_VAL_ROWS" <<'EOF' || { echo "ERROR: hull-CR plag cache preflight failed"; exit 1; }
import os
import sys
import numpy as np
import pandas as pd
from data.continuum_removal import detect_representation
from data.dataset import synth_brightness_path

for cache, parquet in zip(sys.argv[1::2], sys.argv[2::2]):
    arr = np.load(cache, mmap_mode='r')
    shape = arr.shape
    assert len(shape) == 4 and shape[1:3] == (7, 7), f'{cache}: bad shape {shape}'
    assert shape[-1] == 59, (
        f'{cache} is {shape[-1]}-channel, not 59: shape {shape}')
    repr_ = detect_representation(arr)
    assert repr_ == 'hull', (
        f'{cache} looks like {repr_!r}, not hull-CR. This job trains with '
        f'--continuum_removed, so a raw cache here would put plagioclase on a '
        f'different representation from every other class. Rebuild with '
        f'scripts/convert_synth_cache_representation.py --mode hull')
    rows = len(pd.read_parquet(parquet, columns=['split']))
    assert shape[0] == rows, (
        f'{cache} has {shape[0]:,} patches but {parquet} has {rows:,} rows; '
        f'SyntheticPatchDataset asserts they match row-for-row')
    # This job runs --brightness_aux, so the labeled dataset yields
    # (patch, brightness, label, weight). Without this sidecar the synth half
    # yields a 3-tuple and default_collate kills the job on the first mixed
    # batch with "each element in list of batch should be of equal size".
    bpath = synth_brightness_path(cache)
    assert os.path.exists(bpath), (
        f'{cache} has no brightness sidecar {bpath}. This job runs '
        f'--brightness_aux, so SyntheticPatchDataset cannot serve it. Delete '
        f'{cache} and re-run: the converter writes the sidecar alongside.')
    b = np.load(bpath, mmap_mode='r')
    assert b.shape == (shape[0], 7, 7), (
        f'{bpath} has shape {b.shape}, expected {(shape[0], 7, 7)} — one (7,7) '
        f'brightness map per patch, aligned row-for-row with {cache}')
    print(f'  plag cache OK: {cache} {shape} {repr_} vs {rows:,} parquet rows '
          f'(+ brightness sidecar {b.shape})')
EOF
for split in train val; do
    for f in "${CACHE_DIR}/mrral_${split}_patches_p7.npy" \
             "${CACHE_DIR}/mrral_${split}_patches_p7_brightness.npy"; do
        [ -f "$f" ] || { echo "ERROR: CR patch cache incomplete — missing $f"; \
            echo "Build it: $PYTHON scripts/build_cr_labeled_cache.py --raw_dir <raw> --out_dir ${CACHE_DIR} --splits train val test"; \
            exit 1; }
    done
done

# Refuse to train on spectra we have not checked. Cheap next to a wasted GPU day.
echo "=== spectra audit ==="
$PYTHON scripts/audit_spectra_quality.py "$PARQUET" --fail_tile_over 5 || {
    echo "ERROR: $PARQUET failed the spectra audit. Refusing to train."; exit 1; }

echo "=== arm $SLURM_ARRAY_TASK_ID: $RUN_NAME (weight_scheme=$SCHEME) ==="

$PYTHON -u scripts/train.py \
    --model spatial_vit_aux \
    --run_name "$RUN_NAME" \
    --pretrain_ckpt "$ENCODER" \
    --seven_class \
    --continuum_removed --cache_is_cr --brightness_aux \
    --mrral_parquets "$PARQUET" \
    --patch_cache_dir "$CACHE_DIR" \
    --weight_scheme "$SCHEME" \
    --synth_train_cache "$MTRDR_TRAIN_PATCHES" \
    --synth_train_parquet "$MTRDR_TRAIN_ROWS" \
    --synth_val_cache "$MTRDR_VAL_PATCHES" \
    --synth_val_parquet "$MTRDR_VAL_ROWS" \
    --lr 5e-4 --encoder_lr_scale 0.001 \
    --epochs 150 --patience 40 --min_delta 0.0005 \
    --batch_size 256 \
    --embed_dim ${EMBED_DIM} --n_heads 4 --n_layers 6 \
    --patch_size 7 \
    --asl_loss

echo "=== $RUN_NAME done ==="
echo "Floor-test with:"
# NOTE: no vocab flag. classify_tile_supervised._set_n_classes reads
# head.weight and a 7-wide head auto-selects the 7-class vocab. --seven_class
# does not exist there (only --pyx / --pyx_alt do, to disambiguate 5/6-wide
# heads), so passing it would abort with an unrecognised-argument error.
echo "  CLASSIFY_EXTRA_ARGS=\"--continuum_removed --brightness_aux --embed_dim ${EMBED_DIM}\" \\"
echo "    bash scripts/floor_test.sh ${CKPT_DIR}/${RUN_NAME}_best.pt ${RUN_NAME}"
