#!/bin/bash
# Pre-bland-relabel backup. Run ONCE before scripts/build_mrral_dataset.py
# rewrites the parquet, and before scripts/cache_mrral_patches.py rewrites
# the patch cache.
#
# Spec: docs/superpowers/specs/2026-05-20-relabel-other-bland-tiles-design.md
set -e

PROJ=/mnt/mrdr/crism_classification

PARQUET="${PROJ}/data/mrral_pixels.parquet"
BACKUP_PARQUET="${PROJ}/data/mrral_pixels.pre-bland.parquet"

if [ -f "$BACKUP_PARQUET" ]; then
    echo "Backup already exists: ${BACKUP_PARQUET} (refusing to overwrite)"
else
    cp "$PARQUET" "$BACKUP_PARQUET"
    echo "Backed up parquet -> ${BACKUP_PARQUET}"
fi

CACHE_DIR="${PROJ}/data/patch_cache"
for split in train val test; do
    SRC="${CACHE_DIR}/mrral_${split}_patches_p7.npy"
    DST="${CACHE_DIR}/mrral_${split}_patches_p7.pre-bland.npy"
    if [ ! -f "$SRC" ]; then
        echo "  SKIP $split: source cache not found at $SRC"
        continue
    fi
    if [ -f "$DST" ]; then
        echo "  $split backup already exists (refusing to overwrite)"
        continue
    fi
    cp "$SRC" "$DST"
    echo "  Backed up $split cache -> $DST"
done
echo "Done."
