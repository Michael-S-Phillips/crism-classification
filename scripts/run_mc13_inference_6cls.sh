#!/usr/bin/env bash
# Batch inference: 6-class champion on all 54 MC13 tiles.
# Saves per-tile probs to /tmp/6cls_mc13/{tid}_probs.npz
# Run from /mnt/mrdr/crism_classification/
set -e

CKPT="checkpoints/ft_6cls_mc11val_denoise_best.pt"
PROBS_DIR="/tmp/6cls_mc13"
MC13_DIR="/mnt/mrdr/mc13"
PYTHON="conda run -n crism python"

mkdir -p "$PROBS_DIR"

TILES=(
t1028 t1029 t1030 t1031 t1032 t1033 t1034 t1035 t1036
t1100 t1101 t1102 t1103 t1104 t1105 t1106 t1107 t1108
t1172 t1173 t1174 t1175 t1176 t1177 t1178 t1179 t1180
t1244 t1245 t1246 t1247 t1248 t1249 t1250 t1251 t1252
t1316 t1317 t1318 t1319 t1320 t1321 t1322 t1323 t1324
t1388 t1389 t1390 t1391 t1392 t1393 t1394 t1395 t1396
)

for tid in "${TILES[@]}"; do
    npz="$PROBS_DIR/${tid}_probs.npz"
    if [ -f "$npz" ]; then
        echo "SKIP $tid (already done)"
        continue
    fi

    img=$(ls "$MC13_DIR"/${tid}_mrral_*_0327_4.img 2>/dev/null | head -1)
    if [ -z "$img" ]; then
        echo "WARN: no mrral img for $tid"
        continue
    fi

    # Use gpkg if available
    gpkg_path="/mnt/mrdr/categorized_mineral_units/$(echo $tid | tr '[:lower:]' '[:upper:]').gpkg"
    gpkg_arg=""
    if [ -f "$gpkg_path" ]; then
        gpkg_arg="--gpkg $gpkg_path"
    fi

    echo "=== $tid ==="
    $PYTHON scripts/classify_tile_supervised.py \
        --tile "$img" \
        --ckpt "$CKPT" \
        --save_probs "$npz" \
        --batch_size 4096 \
        --no_plot \
        $gpkg_arg
done

echo "Done. $(ls $PROBS_DIR/*.npz | wc -l) / ${#TILES[@]} tiles complete."
