#!/usr/bin/env bash
# Re-run the MC deployment for tiles affected by the load_tile PHYS_MAX bug.
#
# Fixed in 82afe80. Before it, blue-edge values (~3900 I/F at 410 nm) were
# CLIPPED to 0.5 instead of masked, so the model was deployed on pixels it was
# never trained on. On t1389 this manufactured 84,371 px of >=0.99 olivine that
# vanish entirely once the pixels are masked.
#
# Two things this script is careful about:
#
#  1. It does NOT overwrite the original deployment. Corrected probs go to
#     $NEW_PROBS and corrected vectors to $NEW_VEC, so the old and new products
#     can be diffed. The old ones are the evidence for how large the error was.
#
#  2. Each MC's GeoPackages merge across ALL of that MC's tiles, so vectorizing
#     from a directory holding only the re-run tiles would silently produce a
#     partial map. $NEW_PROBS is therefore populated with symlinks to every
#     unaffected tile's existing probs, and real files only where re-run.
#
# Usage:
#   bash scripts/rerun_phys_max_tiles.sh [tiles.txt]
# Default tile list: reports/phys_max_contamination_tiles.txt (from
# scripts/scan_phys_max_contamination.py).

set -euo pipefail
cd "$(dirname "$0")/.."
REPO="$PWD"

TILE_LIST="${1:-reports/phys_max_contamination_tiles.txt}"
CKPT="${CKPT:-checkpoints/ft_6cls_handcore_dualcr_pyx_level_best.pt}"
OLD_PROBS="${OLD_PROBS:-data/mc_deploy_pyx/probs}"
NEW_PROBS="${NEW_PROBS:-data/mc_deploy_pyx_physmax/probs}"
NEW_VEC="${NEW_VEC:-reports/mc_deploy_pyx_physmax}"
DATA_ROOT="${DATA_ROOT:-/mnt/mars-gis/CRISM/MRDR}"
THRESHOLDS="${THRESHOLDS:-0.5 0.85 0.97 0.99 0.995 0.999 0.9995 0.9999}"

[ -f "$TILE_LIST" ] || { echo "no tile list at $TILE_LIST -- run scripts/scan_phys_max_contamination.py first" >&2; exit 1; }
[ -f "$CKPT" ] || { echo "no checkpoint at $CKPT" >&2; exit 1; }

mapfile -t AFFECTED < "$TILE_LIST"
if [ "${#AFFECTED[@]}" -eq 0 ]; then
  # Exit rather than fall through: stage 3 would happily vectorize a directory
  # of pure symlinks and produce a "corrected" product identical to the old one,
  # which reads as confirmation that the bug did not matter.
  echo "tile list is empty -- nothing was affected, nothing to re-run" >&2
  exit 0
fi
echo "re-running ${#AFFECTED[@]} affected tiles with $CKPT"

# --- stage 1: seed $NEW_PROBS with symlinks to every unaffected tile ---------
# Symlink, not copy: these files are ~48 MB each and byte-identical to the
# originals, which remain the authoritative copy.
declare -A IS_AFFECTED=()
for p in "${AFFECTED[@]}"; do
  t="$(basename "$p")"; t="${t%%_mrral*}"
  IS_AFFECTED["$t"]=1
done

n_link=0
for src in "$OLD_PROBS"/*/*_probs.npz; do
  mc="$(basename "$(dirname "$src")")"
  t="$(basename "$src" _probs.npz)"
  mkdir -p "$NEW_PROBS/$mc"
  dst="$NEW_PROBS/$mc/${t}_probs.npz"
  if [ -n "${IS_AFFECTED[$t]:-}" ]; then
    # An affected tile must be regenerated, never linked. If a stale link or
    # file is sitting here from an interrupted run, drop it -- otherwise stage 2
    # would skip the tile and the "corrected" product would carry the old probs.
    rm -f "$dst"
  elif [ ! -e "$dst" ]; then
    ln -s "$(cd "$(dirname "$src")" && pwd)/$(basename "$src")" "$dst"
    n_link=$((n_link + 1))
  fi
done
echo "linked $n_link unaffected tiles into $NEW_PROBS"

# --- stage 2: classify each affected tile -----------------------------------
i=0
for img in "${AFFECTED[@]}"; do
  i=$((i + 1))
  t="$(basename "$img")"; t="${t%%_mrral*}"
  mc="$(basename "$(dirname "$img")")"
  out="$NEW_PROBS/$mc/${t}_probs.npz"
  mkdir -p "$NEW_PROBS/$mc"
  if [ -s "$out" ] && [ ! -L "$out" ]; then
    echo "[$i/${#AFFECTED[@]}] $t already regenerated, skipping"
    continue
  fi
  echo "[$i/${#AFFECTED[@]}] classifying $t ($mc)"
  conda run -n crism python scripts/classify_tile_supervised.py \
    --tile "$img" --ckpt "$CKPT" --save_probs "$out" \
    --no_plot --continuum_removed --dual_cr --brightness_aux --pyx \
    --embed_dim 256
done

# --- stage 3: re-vectorize every MC on the corrected probs -------------------
for mcdir in "$NEW_PROBS"/*/; do
  mc="$(basename "$mcdir")"
  echo "vectorizing $mc"
  # shellcheck disable=SC2086  # THRESHOLDS is a deliberate word-split list
  conda run -n crism python scripts/vectorize_per_mineral_thresholds_nili_6cls.py \
    --probs_dir "$mcdir" \
    --tile_dir "$DATA_ROOT/$mc" \
    --out_dir "$NEW_VEC/$mc" \
    --tiles $(for f in "$mcdir"*_probs.npz; do basename "$f" _probs.npz; done) \
    --thresholds $THRESHOLDS
done

echo "done. corrected probs: $NEW_PROBS   corrected vectors: $NEW_VEC"
echo "originals left untouched at $OLD_PROBS and reports/mc_deploy_pyx"
