#!/bin/bash
#SBATCH --job-name=crism_mc11val_data
#SBATCH --partition=standard
#SBATCH --account=sbyrne
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32gb
#SBATCH --time=12:00:00
#SBATCH --output=logs/mc11val_data_%j.log
#SBATCH --error=logs/mc11val_data_%j.log

# Build the CLEAN-MC11-VALIDATION alteration lineage.
#
# Motivation (2026-06-15 spectra audit): the alteration val/test ground
# truth was 100% mafic-contaminated Argyre/Hellas/Nili gpkg polygons, in a
# different region from the clean MC11 review training signal — so
# val_AP_alteration (~0.09 everywhere) could not measure real alteration
# learning. This lineage instead:
#   - patch --pure_only: alteration=1 ONLY on the 8 pure-alteration gpkg
#     polygons (clean); mixed gpkg alteration never stamped (mafic labels
#     intact, alteration=0).
#   - build --alt_holdout_frac 0.15: polygon-level split of the 65 MC11
#     review alteration polygons -> ~45 train / 10 val / 10 test, so
#     val_AP_alteration is measured on clean, same-distribution held-out
#     MC11 pixels (~17.6k val).
# Also picks up the band-0 (I/F>1.0 -> nodata) fix automatically via the
# rebuilt cache.
#
# Outputs (parallel to the v2 lineage; nothing existing is touched):
#   data/mrral_pixels_purealt_only.parquet
#   data/mrral_pixels_with_review_mc11val.parquet
#   data/patch_cache_mc11val/

WORKDIR="/groups/sbyrne/${USER}/crism_classification"

export MAMBA_EXE='/opt/ohpc/pub/apps/micromamba/2.0.2-2/bin/micromamba'
export MAMBA_ROOT_PREFIX='/groups/sbyrne/phillipsm/micromamba'
eval "$($MAMBA_EXE shell hook --shell bash --root-prefix $MAMBA_ROOT_PREFIX)"
micromamba activate crism

cd "$WORKDIR"
mkdir -p logs data/patch_cache_mc11val

echo "=== step 1: pure_only patch (clean pure-gpkg alteration only) ==="
python -u scripts/patch_mrral_pixels_with_alteration.py \
    --pure_only \
    --out data/mrral_pixels_purealt_only.parquet || exit 1

echo "=== step 2: review-augmented build with MC11 polygon holdout ==="
python -u scripts/build_review_augmented_train.py \
    --existing data/mrral_pixels_purealt_only.parquet \
    --max_pixels_per_polygon 20000 \
    --include_ambiguous \
    --with_alteration_column \
    --alt_holdout_frac 0.15 \
    --out data/mrral_pixels_with_review_mc11val.parquet || exit 1

echo "=== step 3: patch cache (picks up band-0 nodata fix) ==="
python -u scripts/cache_mrral_patches.py \
    --parquets data/mrral_pixels_with_review_mc11val.parquet \
    --cache_dir data/patch_cache_mc11val || exit 1

echo "=== step 4: verify cache byte counts + report alteration split ==="
python -c "
import os
import pandas as pd
df = pd.read_parquet('data/mrral_pixels_with_review_mc11val.parquet',
                     columns=['split', 'alteration'])
ok = True
for split in ('train', 'val', 'test'):
    sub = df[df['split'] == split]
    n = len(sub)
    n_alt = int((sub['alteration'] > 0.5).sum())
    f = f'data/patch_cache_mc11val/mrral_{split}_patches_p7.npy'
    expected = n * 7 * 7 * 59 * 4
    actual = os.path.getsize(f) if os.path.exists(f) else -1
    status = 'OK' if actual == expected else 'MISMATCH'
    if actual != expected: ok = False
    prev = n_alt / max(n, 1)
    print(f'  {split}: {n:,} rows, {n_alt:,} alt-pos (prevalence {prev:.3f}), '
          f'cache {actual:,} bytes {status}')
if not ok:
    raise SystemExit(1)
" || { echo 'ERROR: cache verification failed'; exit 1; }

echo "=== mc11val data lineage complete ==="
