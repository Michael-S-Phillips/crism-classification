#!/usr/bin/env bash
# Batch inference: new 7-class v3 (per-polygon-capped) lrscale001 model on all
# MC11 + MC13 tiles. Saves per-tile probs to /tmp/7cls_v3_{mc11,mc13}/{tid}_probs.npz
# then vectorizes per-mineral threshold polygons for the review app.
# Run from /mnt/mrdr/crism_classification/
set -e

CKPT="checkpoints/ft_7cls_v3b_lrscale001_best.pt"
PYTHON="conda run -n crism python"

run_region () {
    local region="$1" mc_dir="$2" probs_dir="$3" out_dir="$4"
    mkdir -p "$probs_dir"
    echo "##### REGION $region : $(ls $mc_dir/*_mrral_*_0327_4.img 2>/dev/null | wc -l) tiles #####"
    for img in "$mc_dir"/*_mrral_*_0327_4.img; do
        [ -e "$img" ] || continue
        local tid
        tid=$(basename "$img" | cut -d_ -f1)
        local npz="$probs_dir/${tid}_probs.npz"
        if [ -f "$npz" ]; then echo "SKIP $tid (done)"; continue; fi
        echo "=== $region $tid ==="
        $PYTHON scripts/classify_tile_supervised.py \
            --tile "$img" --ckpt "$CKPT" \
            --save_probs "$npz" --batch_size 4096 --no_plot
    done
    echo "##### $region inference done: $(ls $probs_dir/*.npz 2>/dev/null | wc -l) npz #####"
    # vectorize per-mineral threshold polygons for review
    local tiles
    tiles=$(ls "$probs_dir"/*_probs.npz | xargs -n1 basename | sed 's/_probs.npz//' | tr '\n' ' ')
    $PYTHON scripts/vectorize_per_mineral_thresholds_nili_6cls.py \
        --probs_dir "$probs_dir" --out_dir "$out_dir" \
        --tile_dir "$mc_dir" --tiles $tiles
    echo "##### $region vectorize done -> $out_dir #####"
}

run_region mc11 /mnt/mrdr/mc11 /tmp/7cls_v3_mc11 data/vector_mc11_7cls_v3_lrscale001
run_region mc13 /mnt/mrdr/mc13 /tmp/7cls_v3_mc13 data/vector_mc13_7cls_v3_lrscale001

echo "ALL DONE."
