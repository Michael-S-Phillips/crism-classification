#!/usr/bin/env bash
# Inference: all 3 ft_6cls_v3b arms on the 4 Nili Fossae tiles.
# Saves per-arm probs to /tmp/v3b_nili_{arm}/t*_probs.npz
# Run from /mnt/mrdr/crism_classification/
set -e

TILES=(t1249 t1250 t1321 t1322)
MC_DIR="/mnt/mrdr/mc13"
PYTHON="conda run -n crism python"

ARMS=(
    "ft_6cls_v3b_lrscale0001_best.pt:/tmp/v3b_nili_lrscale0001"
    "ft_6cls_v3b_lrscale001_best.pt:/tmp/v3b_nili_lrscale001"
    "ft_6cls_v3b_lrscale01_best.pt:/tmp/v3b_nili_lrscale01"
)

for arm in "${ARMS[@]}"; do
    ckpt="checkpoints/${arm%%:*}"
    probs_dir="${arm##*:}"
    mkdir -p "$probs_dir"

    echo "=== ARM: $ckpt → $probs_dir ==="
    for tid in "${TILES[@]}"; do
        npz="${probs_dir}/${tid}_probs.npz"
        if [ -f "$npz" ]; then
            echo "  SKIP $tid"
            continue
        fi
        img=$(ls "$MC_DIR"/${tid}_mrral_*_0327_4.img 2>/dev/null | head -1)
        if [ -z "$img" ]; then echo "  WARN: no mrral for $tid"; continue; fi
        echo "  $tid"
        $PYTHON scripts/classify_tile_supervised.py \
            --tile "$img" \
            --ckpt "$ckpt" \
            --save_probs "$npz" \
            --batch_size 4096 \
            --no_plot
    done
done

echo "Done."
