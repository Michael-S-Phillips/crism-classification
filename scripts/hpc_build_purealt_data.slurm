#!/bin/bash
#SBATCH --job-name=crism_purealt_data
#SBATCH --partition=standard
#SBATCH --account=sbyrne
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32gb
#SBATCH --time=12:00:00
#SBATCH --output=logs/purealt_data_%j.log
#SBATCH --error=logs/purealt_data_%j.log

# Build the PURE-ALTERATION training lineage end to end:
#   1. patch (pure_train_only): mixed "mafic + alteration" gpkg polygons
#      keep their mafic labels but get alteration=0 on TRAIN rows; the 8
#      pure-alteration polygons stamp everywhere; val/test stamping is
#      IDENTICAL to the standard build so alteration AP stays comparable
#      across every 6-class run.
#      → data/mrral_pixels_purealt.parquet   (base parquet untouched)
#   2. build_review_augmented_train on that base
#      → data/mrral_pixels_with_review_v2pure.parquet
#      Train alteration positives = the 103,895 single-label review-tagged
#      pixels + the 8 pure polygons' train pixels (capped).
#   3. patch cache → data/patch_cache_v2pure/  + byte-level verification.
#
# Nothing here touches the v2 parquet or v2_only cache the running
# ablations depend on.

WORKDIR="/groups/sbyrne/${USER}/crism_classification"

export MAMBA_EXE='/opt/ohpc/pub/apps/micromamba/2.0.2-2/bin/micromamba'
export MAMBA_ROOT_PREFIX='/groups/sbyrne/phillipsm/micromamba'
eval "$($MAMBA_EXE shell hook --shell bash --root-prefix $MAMBA_ROOT_PREFIX)"
micromamba activate crism

cd "$WORKDIR"
mkdir -p logs data/patch_cache_v2pure

echo "=== step 1: pure-alteration patch ==="
python -u scripts/patch_mrral_pixels_with_alteration.py \
    --pure_train_only \
    --out data/mrral_pixels_purealt.parquet || exit 1

echo "=== step 2: review-augmented build (pure base) ==="
python -u scripts/build_review_augmented_train.py \
    --existing data/mrral_pixels_purealt.parquet \
    --max_pixels_per_polygon 20000 \
    --include_ambiguous \
    --with_alteration_column \
    --out data/mrral_pixels_with_review_v2pure.parquet || exit 1

echo "=== step 3: patch cache ==="
python -u scripts/cache_mrral_patches.py \
    --parquets data/mrral_pixels_with_review_v2pure.parquet \
    --cache_dir data/patch_cache_v2pure || exit 1

echo "=== step 4: verify cache byte counts ==="
python -c "
import os
import pandas as pd
df = pd.read_parquet('data/mrral_pixels_with_review_v2pure.parquet',
                     columns=['split', 'alteration'])
ok = True
for split in ('train', 'val', 'test'):
    sub = df[df['split'] == split]
    n = len(sub)
    n_alt = int((sub['alteration'] > 0.5).sum())
    f = f'data/patch_cache_v2pure/mrral_{split}_patches_p7.npy'
    expected = n * 7 * 7 * 59 * 4
    actual = os.path.getsize(f) if os.path.exists(f) else -1
    status = 'OK' if actual == expected else 'MISMATCH'
    if actual != expected: ok = False
    print(f'  {split}: {n:,} rows ({n_alt:,} alt-pos), cache {actual:,} bytes {status}')
raise SystemExit(0 if ok else 1)
" || { echo 'ERROR: cache verification failed'; exit 1; }

echo "=== purealt data lineage complete ==="
