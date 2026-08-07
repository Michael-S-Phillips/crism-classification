#!/usr/bin/env bash
# One-time migration: consolidate all bulk data under the canonical xdisk layout
# the harmonized slurm files + config_loader expect:
#
#   /xdisk/sbyrne/phillipsm/CRISM_MRDR/            <- data_root
#   ├── mc02 … mc30/  categorized_mineral_units/   (already here)
#   └── crism_classification/
#       ├── data/          <- parquets + all patch caches
#       └── checkpoints/   <- model checkpoints
#
# SAFE BY DESIGN: `mv -n` never overwrites, nothing is deleted, every source is
# existence-guarded, and unresolved items are printed at the end for you to
# handle by hand. Re-runnable. Run on HPC.
set -uo pipefail

DATA_ROOT=/xdisk/sbyrne/phillipsm/CRISM_MRDR
DATA_DIR=$DATA_ROOT/crism_classification/data
CKPT_DIR=$DATA_ROOT/crism_classification/checkpoints
GROUPS=/groups/sbyrne/phillipsm/crism_classification
XD=/xdisk/sbyrne/phillipsm

mkdir -p "$DATA_DIR" "$CKPT_DIR"

move() {  # move SRC into DEST dir — logged, guarded, never clobbers
    local src="$1" dest="$2"
    if [ -e "$src" ]; then
        if [ -e "$dest/$(basename "$src")" ]; then
            echo "  KEEP (exists at dest, not overwritten): $src"
        else
            echo "  mv   $src  ->  $dest/"
            mv -n "$src" "$dest"/
        fi
    fi
}

echo "== 1. checkpoints: /groups -> $CKPT_DIR =="
shopt -s nullglob
for f in "$GROUPS"/checkpoints/*; do move "$f" "$CKPT_DIR"; done

echo "== 2. parquets: xdisk sibling /data + /groups/data -> $DATA_DIR =="
for p in "$XD"/data/*.parquet "$GROUPS"/data/*.parquet; do move "$p" "$DATA_DIR"; done

echo "== 3. labeled patch caches: xdisk /data + /groups/data -> $DATA_DIR =="
for c in "$XD"/data/patch_cache* "$GROUPS"/data/patch_cache*; do move "$c" "$DATA_DIR"; done

echo "== 4. contrastive extras (if present in /groups) -> $DATA_DIR =="
move "$GROUPS/data/contrastive" "$DATA_DIR"
shopt -u nullglob

echo "== 5. refresh config.local.yaml to just data_root (config_loader derives the rest) =="
if [ -d "$GROUPS" ]; then
    printf 'data_root: %s\n' "$DATA_ROOT" > "$GROUPS/config.local.yaml"
    echo "  wrote $GROUPS/config.local.yaml"
fi

echo
echo "############ REVIEW BY HAND (not moved automatically) ############"
echo "[base parquet] pyx_alt + many slurm need $DATA_DIR/mrral_pixels.parquet"
if [ -e "$DATA_DIR/mrral_pixels.parquet" ]; then
    echo "    OK present."
else
    echo "    !! MISSING. It is not on xdisk. Copy the hand-labeled base parquet here,"
    echo "       e.g. from your Mac:  scp <mac>:/Volumes/Mars_GIS/CRISM/MRDR/crism_classification/data/mrral_pixels.parquet $DATA_DIR/"
fi
echo "[global caches] left in place — the pretraining caches use a different name"
echo "    ($XD/crism_patch_cache, $XD/crism_patch_cache_cr). Encoders already exist,"
echo "    so low priority; reconcile only if you re-run MAE pretraining."
echo "[strays] inspect & remove if redundant (NOT touched):"
for d in "$XD/patch_cache_7cls" "$XD/patch_cache_7cl_cr" "$XD/vrdr_patch_cache" "$XD/mars_js"; do
    [ -e "$d" ] && echo "    $d"
done
echo "[review inputs] hpc_build_7cls_data*.slurm still read review data via relative"
echo "    data/ paths under /groups — those two builders aren't harmonized yet."
echo
echo "== verify paths resolve (run from $GROUPS with the crism env) =="
echo "  python -c \"from config_loader import load_config; import os; c=load_config(); [print(k, c[k], os.path.exists(c[k])) for k in ('data_root','checkpoints_dir','output_dir','patch_cache_dir','gpkg_dir')]\""
