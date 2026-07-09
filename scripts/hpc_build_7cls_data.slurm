#!/bin/bash
#SBATCH --job-name=crism_7cls_data
#SBATCH --partition=standard
#SBATCH --account=sbyrne
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64gb
#SBATCH --time=12:00:00
#SBATCH --output=logs/7cls_data_%j.log
#SBATCH --error=logs/7cls_data_%j.log

# Build the 7-class training parquet + patch cache.
#
# Data design (see scripts/build_7cls_dataset.py for full rationale):
#   - Splits: ALL labeled sources use the unit-aware, pixel-balanced splitter
#     (scripts/split_units.py). Polygons within 0.25 deg cluster into geographic
#     units; whole units go to one split (70/15/15 on per-class pixels, >=5%
#     val/test guard). Kills adjacent-tile unit leakage; base-parquet inherited
#     splits are OVERRIDDEN. Each source split independently (own seed offset).
#   - Review sessions are ADDITIVE: original MC13 session (weight 1.0) + the
#     confidence-graded v3 session (MC13+MC11, per-polygon 0.5/0.75/1.0)
#   - Plagioclase: Argyre/Hellas gpkg (unit-re-split) + MTRDR synth
#   - Alteration: review tags (negative_of='alteration') + alteration confirms,
#     per-polygon confidence weights preserved, unit-balanced holdout
#   - Bland (was "other"): sources subsampled to 300k, each unit-split on 'other':
#       1. Bland tiles  2. MC13 review blands  3. MC11 review blands (tile-id range)
#   - Reassigned minerals (reject→mineral, incl. multi-label): weighted positives
#   - Junk: ambiguous hard_negatives, now PER-POLYGON CAPPED (was uncapped),
#     unit-split on 'junk', per-polygon weights preserved
#   - ndviz relabel session (--ndviz_dir): interactive N-D visualizer relabels
#     ingested with pixel-level supersede (every ndviz pixel replaces lower
#     sources, then ndviz positives appended); absent dir = no-op
#
# Outputs:
#   data/mrral_pixels_7cls.parquet
#   data/patch_cache_7cls/

WORKDIR="/groups/sbyrne/${USER}/crism_classification"

export MAMBA_EXE='/opt/ohpc/pub/apps/micromamba/2.0.2-2/bin/micromamba'
export MAMBA_ROOT_PREFIX='/groups/sbyrne/phillipsm/micromamba'
eval "$($MAMBA_EXE shell hook --shell bash --root-prefix $MAMBA_ROOT_PREFIX)"
micromamba activate crism

cd "$WORKDIR"
mkdir -p logs data/patch_cache_7cls

PARQUET="data/mrral_pixels_7cls.parquet"
CACHE_DIR="data/patch_cache_7cls"

# Guard: the v3 review session data must be rsynced from the workstation first
for d in data/mc13_review_7cls_v3/confirmed_pixels data/mc13_review_7cls_v3/hard_negatives; do
    if [ ! -d "$d" ]; then
        echo "ERROR: $d missing — rsync data/mc13_review_7cls_v3 from workstation first"
        exit 1
    fi
done

echo "=== step 1: build 7-class parquet ==="
python -u scripts/build_7cls_dataset.py \
    --base_parquet  data/mrral_pixels.parquet \
    --confirmed_dir data/mc13_review/confirmed_pixels \
                    data/mc13_review_7cls_v3/confirmed_pixels \
    --hn_dir        data/mc13_review/hard_negatives \
                    data/mc13_review_7cls_v3/hard_negatives \
    --ndviz_dir     data/ndviz_relabels/hard_negatives \
    --out           "$PARQUET" || exit 1
    # ^ ndviz relabel session: absent dir = no-op (HPC won't have it until
    #   data/ndviz_relabels is rsynced from the workstation)

echo "=== step 2: build patch cache ==="
python -u scripts/cache_mrral_patches.py \
    --parquets "$PARQUET" \
    --cache_dir "$CACHE_DIR" || exit 1

echo "=== step 3: verify cache byte counts + label summary ==="
python -c "
import os
import pandas as pd
df = pd.read_parquet('$PARQUET',
                     columns=['split', 'bland', 'junk', 'alteration', 'plagioclase'])
ok = True
for split in ('train', 'val', 'test'):
    sub = df[df['split'] == split]
    n = len(sub)
    f = 'data/patch_cache_7cls/mrral_{split}_patches_p7.npy'.replace('{split}', split)
    expected = n * 7 * 7 * 59 * 4
    actual = os.path.getsize(f) if os.path.exists(f) else -1
    status = 'OK' if actual == expected else 'MISMATCH'
    if actual != expected: ok = False
    n_bland = int((sub['bland'] > 0.5).sum())
    n_junk  = int((sub['junk']  > 0.5).sum())
    n_alt   = int((sub['alteration'] > 0.5).sum())
    n_plag  = int((sub['plagioclase'] > 0.5).sum())
    print(f'  {split}: {n:,} rows | bland={n_bland:,} junk={n_junk:,} '
          f'alt={n_alt:,} plag={n_plag:,} | cache {status}')
if not ok:
    raise SystemExit(1)
" || { echo 'ERROR: cache verification failed'; exit 1; }

echo "=== 7cls data build complete ==="
