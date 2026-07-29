#!/usr/bin/env bash
# Full MC11 classification with the CR (continuum-removed) 7-class model.
# Classifies all 54 MC11 mrral tiles → per-tile probs → vectorizes per-mineral
# threshold polygons. Resumable (skip-if-probs-exist). CR inference is slower than
# raw (per-pixel upper-hull), so budget ~10-20 min/tile (~a several-hour run).
# Run from /mnt/mrdr/crism_classification/.
set -e

CKPT="checkpoints/ft_7cls_cr_lrscale0001_best.pt"     # CR 256d, --continuum_removed
PYTHON="conda run -n crism python"
MC_DIR="/mnt/mrdr/mc11"
PROBS_DIR="/tmp/cr_mc11"
OUT_DIR="data/vector_mc11_cr_lrscale0001"

mkdir -p "$PROBS_DIR"
echo "##### CR MC11: $(ls $MC_DIR/*_mrral_*_0327_4.img 2>/dev/null | wc -l) tiles, ckpt $CKPT #####"
for img in "$MC_DIR"/*_mrral_*_0327_4.img; do
    [ -e "$img" ] || continue
    tid=$(basename "$img" | cut -d_ -f1)
    npz="$PROBS_DIR/${tid}_probs.npz"
    if [ -f "$npz" ]; then echo "SKIP $tid (done)"; continue; fi
    echo "=== $tid  $(date +%H:%M:%S) ==="
    $PYTHON scripts/classify_tile_supervised.py \
        --tile "$img" --ckpt "$CKPT" \
        --continuum_removed --brightness_aux --embed_dim 256 \
        --save_probs "$npz" --batch_size 4096 --no_plot
done
echo "##### CR MC11 inference done: $(ls $PROBS_DIR/*.npz 2>/dev/null | wc -l) npz #####"

tiles=$(ls "$PROBS_DIR"/*_probs.npz | xargs -n1 basename | sed 's/_probs.npz//' | tr '\n' ' ')
$PYTHON scripts/vectorize_per_mineral_thresholds_nili_6cls.py \
    --probs_dir "$PROBS_DIR" --out_dir "$OUT_DIR" \
    --tile_dir "$MC_DIR" --tiles $tiles
echo "##### CR MC11 vectorize done -> $OUT_DIR #####"
echo "ALL DONE."
