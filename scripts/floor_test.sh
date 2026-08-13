#!/usr/bin/env bash
# Checkpoint floor test: classify the 8 standard tiles (4 Nili, 2 Argyre, 2 MC11),
# vectorize per-mineral threshold polygons, and write a summary report.
#
# Usage:
#   bash scripts/floor_test.sh <checkpoint.pt> [tag]
#
# tag defaults to the checkpoint basename (sans .pt). Outputs:
#   /tmp/floor_test_<tag>/{nili,argyre,mc11}/       per-tile probs (resumable)
#   reports/floor_tests/<tag>/{nili,argyre,mc11}/   threshold polygon gpkgs
#   reports/floor_tests/<tag>/summary.md            tables + gpkg sizes
#
# Inference is skip-if-exists per tile, so a killed run resumes where it left
# off. Expect ~35 min for a cold run (8 tiles x ~4 min). MC11 (t1086/t1087) is
# the altered/dusty OOD probe — no established good/bad signature yet; read it
# qualitatively (false minerals on altered ground = the classic MC11 failure).
set -euo pipefail

CKPT="${1:?usage: floor_test.sh <checkpoint.pt> [tag]}"
TAG="${2:-$(basename "${CKPT%.pt}")}"
PROJ="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="conda run -n crism python"
PROBS_ROOT="/tmp/floor_test_${TAG}"
REPORT_DIR="${PROJ}/reports/floor_tests/${TAG}"

cd "$PROJ"
# A baseline run (CLASSIFY_CMD set) has no checkpoint: $CKPT is a placeholder
# that the scorer accepts and ignores. Without CLASSIFY_CMD the check is
# unchanged, so existing callers still fail loudly on a bad checkpoint path.
if [ -z "${CLASSIFY_CMD:-}" ]; then
    [ -f "$CKPT" ] || { echo "ERROR: checkpoint not found: $CKPT" >&2; exit 1; }
fi
mkdir -p "$REPORT_DIR"

# Resolve the machine-local data root from config (portable: Mac /Volumes, HPC /xdisk).
DATA_ROOT=$($PYTHON -c "from config_loader import load_config; print(load_config()['data_root'])")
[ -n "$DATA_ROOT" ] || { echo "ERROR: could not resolve data_root from config" >&2; exit 1; }

# region <name> <tile_dir> <tile...>
run_region () {
    local region="$1" tile_dir="$2"; shift 2
    local probs_dir="${PROBS_ROOT}/${region}"
    local out_dir="${REPORT_DIR}/${region}"
    mkdir -p "$probs_dir"
    for tid in "$@"; do
        local npz="${probs_dir}/${tid}_probs.npz"
        if [ -f "$npz" ]; then echo "SKIP $tid (probs exist)"; continue; fi
        local img
        img=$(ls "${tile_dir}/${tid}"_mrral_*_0327_4.img 2>/dev/null | head -1)
        [ -n "$img" ] || { echo "ERROR: no mrral img for $tid in $tile_dir" >&2; exit 1; }
        echo "=== ${region} ${tid} ==="
        # CLASSIFY_CMD lets a BASELINE produce the same probs npz and run
        # through this identical vectorization. Defaulting to the supervised
        # classifier keeps every existing caller byte-identical. A forked copy
        # of this script would drift, and a drifted vectorization silently
        # stops being the same comparison. --ckpt is always passed, so a
        # baseline scorer must accept and ignore it.
        ${CLASSIFY_CMD:-$PYTHON scripts/classify_tile_supervised.py} \
            --tile "$img" --ckpt "$CKPT" \
            --save_probs "$npz" --no_plot ${CLASSIFY_EXTRA_ARGS:-}
    done
    $PYTHON scripts/vectorize_per_mineral_thresholds_nili_6cls.py \
        --probs_dir "$probs_dir" --out_dir "$out_dir" \
        --tile_dir "$tile_dir" --tiles "$@" \
        | tee "${REPORT_DIR}/${region}_vectorize.log"
}

run_region nili   "${DATA_ROOT}/mc13" t1249 t1250 t1321 t1322
run_region argyre "${DATA_ROOT}/mc26" t0434 t0435
run_region mc11   "${DATA_ROOT}/mc11" t1086 t1087

# ── Summary report ────────────────────────────────────────────────────────────
SUMMARY="${REPORT_DIR}/summary.md"
{
    echo "# Floor test: ${TAG}"
    echo
    echo "- checkpoint: \`${CKPT}\`"
    echo "- date: $(date -u +%Y-%m-%dT%H:%MZ)"
    echo "- tiles: nili t1249 t1250 t1321 t1322 | argyre t0434 t0435 | mc11 t1086 t1087"
    for region in nili argyre mc11; do
        echo
        echo "## ${region} — per-mineral × threshold polygon counts"
        echo '```'
        # the table is the last block of the vectorize log
        sed -n '/Per-mineral × threshold polygon counts:/,$p' \
            "${REPORT_DIR}/${region}_vectorize.log"
        echo '```'
        echo
        echo "gpkg sizes:"
        echo '```'
        ls -la "${REPORT_DIR}/${region}"/*.gpkg 2>/dev/null | awk '{printf "%10d  %s\n", $5, $9}'
        echo '```'
    done
    echo
    prev=$(ls -1dt "${PROJ}"/reports/floor_tests/*/ 2>/dev/null \
           | grep -v "/${TAG}/" | head -1 || true)
    if [ -n "$prev" ]; then
        echo "Previous floor test for comparison: \`${prev}summary.md\`"
    else
        echo "No previous floor test found — this is the baseline."
    fi
} > "$SUMMARY"

echo
echo "Floor test complete → ${SUMMARY}"
