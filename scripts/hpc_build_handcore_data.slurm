#!/bin/bash
#SBATCH --job-name=crism_handcore_data
#SBATCH --account=sbyrne
#SBATCH --partition=standard
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=180gb
#SBATCH --time=0-08:00:00
#SBATCH --output=logs/handcore_data_%j.log
#SBATCH --error=logs/handcore_data_%j.log

# Build the HAND-CORE 7-class training parquet.
#
# Recipe (spec: docs/superpowers/specs/2026-08-08-hand-core-dataset-design.md):
#   hand-labeled data is the CORE; review data augments it. Three classes are
#   deliberate exceptions because hand labels cannot carry them:
#     bland       review-only (the base parquet's `other` rows are DROPPED)
#     alteration  review is the source of truth; hand alteration rides along
#     junk        review-only (absent from hand labels entirely)
#   v3 review is admitted at High+Moderate. The UNGRADED legacy session is
#   admitted only for alteration/lcp/hcp, with a 5k/polygon cap on its confirms.
#
# WHY THIS NEEDS A BIG-MEMORY NODE: the legacy hard-negatives directory is
# 8.26M rows x 59 float32 bands. This build was OOM-killed on a 64 GB
# workstation. --mem=180gb is deliberate, not padding.
#
# EVERY POLICY FLAG IS PASSED EXPLICITLY. The builder's defaults are inert by
# design, so a bare run reproduces the PRE-hand-core dataset. Dropping a flag
# here silently changes the recipe without failing.
#
# Submit:  sbatch scripts/hpc_build_handcore_data.slurm

WORK_DIR=/groups/sbyrne/phillipsm/crism_classification
DATA_ROOT=/xdisk/sbyrne/phillipsm/CRISM_MRDR
PYTHON=/groups/sbyrne/phillipsm/micromamba/envs/crism/bin/python

DATA_DIR=${DATA_ROOT}/crism_classification/data
BASE_PARQUET=${DATA_DIR}/mrral_pixels.parquet
V3_HN=${DATA_DIR}/mc13_review_7cls_v3/hard_negatives
V3_CF=${DATA_DIR}/mc13_review_7cls_v3/confirmed_pixels
OUT=${DATA_DIR}/mrral_pixels_7cls_handcore.parquet

cd "$WORK_DIR"
mkdir -p logs

if [ ! -f config.local.yaml ]; then
    printf 'data_root: %s\n' "${DATA_ROOT}" > config.local.yaml
fi

# ── STEP 0: spectra-quality gate ─────────────────────────────────────────────
# The 2026-08-08 t1444 incident: a review session extracted a tile mid-download
# and froze 537,525 rows at reflectance 0.0 across 2251-2457 nm. Nothing
# crashed. Because bland is review-only here, ~72% of that class would have
# carried a zero tail no other class had — a shortcut the model learns instead
# of mineralogy. Internal checks alone cannot see it; --verify_against_tiles
# re-reads the source .img, which is the only thing that catches a stale
# extraction. Gate the build on it rather than discovering it in a floor test.
echo "=== STEP 0: auditing review spectra ($(date)) ==="
for d in "$V3_HN" "$V3_CF"; do
    [ -d "$d" ] || { echo "ERROR: missing review dir $d"; exit 1; }
    $PYTHON scripts/audit_spectra_quality.py "$d" --verify_against_tiles --sample 25
    rc=$?
    if [ $rc -ne 0 ]; then
        echo "ERROR: spectra audit FAILED for $d (exit $rc)."
        echo "  A named tile is stale or corrupt. Repair it with:"
        echo "    $PYTHON scripts/reextract_stale_review_tile.py --tile <tile> \\"
        echo "        --session ${DATA_DIR}/mc13_review_7cls_v3"
        echo "  then re-submit. Refusing to build on bad spectra."
        exit 1
    fi
done

echo "=== STEP 0b: auditing the base parquet ==="
$PYTHON scripts/audit_spectra_quality.py "$BASE_PARQUET" || {
    echo "ERROR: base parquet failed the spectra audit. Refusing to build."; exit 1; }

# ── STEP 1: build ────────────────────────────────────────────────────────────
echo "=== STEP 1: hand-core build ($(date)) ==="
$PYTHON -u scripts/build_7cls_dataset.py \
    --base_parquet   "$BASE_PARQUET" \
    --bland_sources  review \
    --review_grades  High Moderate \
    --legacy_classes alteration lcp hcp \
    --legacy_confirm_cap 5000 \
    --ndviz_dir      '' \
    --out            "$OUT" || { echo "ERROR: build failed"; exit 1; }

# ── STEP 2: audit the product ────────────────────────────────────────────────
echo "=== STEP 2: auditing the built parquet ($(date)) ==="
$PYTHON scripts/audit_spectra_quality.py "$OUT" || {
    echo "ERROR: the BUILT parquet failed the spectra audit. Do not train on it."
    exit 1; }

# ── STEP 3: composition report ───────────────────────────────────────────────
# The spec's volumes (bland ~743.8k, junk ~165.4k, alteration ~259k) are
# PROJECTIONS and have never been validated at full volume. Print the truth.
echo "=== STEP 3: composition vs spec projections ==="
$PYTHON - "$OUT" <<'PYEOF'
import sys
import pandas as pd
df = pd.read_parquet(sys.argv[1], columns=[
    'olivine_t1', 'olivine_t2', 'lcp', 'hcp', 'plagioclase', 'bland',
    'alteration', 'junk', 'confidence_tier', 'split'])
print(f'total rows: {len(df):,}\n')
proj = {'bland': 743_808, 'junk': 165_431, 'alteration': 259_334}
print(f'{"class":<14}{"train":>10}{"val":>9}{"test":>9}{"total":>11}{"projected":>11}')
print('-' * 64)
for c in ['olivine_t1', 'olivine_t2', 'lcp', 'hcp', 'plagioclase', 'bland',
          'alteration', 'junk']:
    pos = df[df[c] > 0]['split'].value_counts()
    tot = int((df[c] > 0).sum())
    p = f'{proj[c]:,}' if c in proj else '-'
    print(f'{c:<14}{pos.get("train",0):>10,}{pos.get("val",0):>9,}'
          f'{pos.get("test",0):>9,}{tot:>11,}{p:>11}')
print('\nconfidence tiers:')
print(df['confidence_tier'].value_counts().to_string())
print('\nsplit fractions per class (target 0.70 / 0.15 / 0.15):')
for c in ['bland', 'alteration', 'junk']:
    v = df[df[c] > 0]['split'].value_counts(normalize=True)
    flag = '  <-- SKEWED' if v.get('val', 0) < 0.10 else ''
    print(f'  {c:<12} train {v.get("train",0):.3f}  val {v.get("val",0):.3f}  '
          f'test {v.get("test",0):.3f}{flag}')
PYEOF

echo "=== DONE ($(date)) — $OUT ==="
echo "Next: build the patch cache for this parquet (all splits) BEFORE fine-tuning."
echo "An incomplete CR cache now fails fast at startup instead of training on raw patches."
