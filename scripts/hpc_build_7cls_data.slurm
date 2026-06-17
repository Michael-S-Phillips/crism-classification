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
#   - Plagioclase: zeroed in base parquet; MTRDR synth rows are sole plag source
#   - Alteration: MC11 review only (103.9k pixels, 65 polygons, polygon-level split)
#   - Bland (was "other"): 3 sources × 300k rows, each 70/15/15 independently:
#       1. Bland tiles (8 tiles, subsampled from ~900k)
#       2. MC13 review blands (rejected polygons from MC13 review)
#       3. MC11 review blands (rejected polygons from MC11 review)
#   - Junk (new): ambiguous hard_negatives (34k), 70/15/15 tile split, weight=2.0
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

echo "=== step 1: build 7-class parquet ==="
python -u scripts/build_7cls_dataset.py \
    --base_parquet  data/mrral_pixels.parquet \
    --confirmed_dir data/mc13_review/confirmed_pixels \
    --hn_dir        data/mc13_review/hard_negatives \
    --out           "$PARQUET" || exit 1

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
