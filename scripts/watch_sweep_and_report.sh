#!/usr/bin/env bash
# Polls for sweep completion then auto-generates the report.
# Runs independently of Claude — launch with:
#   nohup bash scripts/watch_sweep_and_report.sh > logs/watch_sweep.log 2>&1 &
#
# Completion criteria:
#   - All 11 sweep checkpoints exist in checkpoints/
#   - The sweep.py process is no longer running
#
# After completion, runs generate_report.py and writes to reports/.

set -euo pipefail

PROJ="$(cd "$(dirname "$0")/.." && pwd)"
CKPT_DIR="$PROJ/checkpoints"
LOG_DIR="$PROJ/logs"
mkdir -p "$LOG_DIR"

EXPECTED_CHECKPOINTS=(
    mlp_sw1_best.pt
    mlp_sw2_best.pt
    mlp_sw3_best.pt
    mlp_sw4_best.pt
    cnn_sw1_best.pt
    cnn_sw2_best.pt
    cnn_sw3_best.pt
    cnn_sw4_best.pt
    vit_sw1_best.pt
    vit_sw2_best.pt
    vit_sw3_best.pt
)

POLL_INTERVAL=120  # seconds between checks

echo "[$(date)] Watcher started. Polling every ${POLL_INTERVAL}s for sweep completion."
echo "[$(date)] Expecting checkpoints in: $CKPT_DIR"

while true; do
    # Check all checkpoints exist
    all_exist=true
    missing=()
    for ckpt in "${EXPECTED_CHECKPOINTS[@]}"; do
        if [[ ! -f "$CKPT_DIR/$ckpt" ]]; then
            all_exist=false
            missing+=("$ckpt")
        fi
    done

    if [[ "$all_exist" == "true" ]]; then
        # Also confirm sweep.py is not running
        if ! pgrep -f "sweep\.py" > /dev/null 2>&1; then
            echo "[$(date)] All checkpoints present and sweep.py not running — sweep complete!"
            break
        else
            echo "[$(date)] All checkpoints present but sweep.py still running — waiting..."
        fi
    else
        echo "[$(date)] Still waiting. Missing: ${missing[*]}"
    fi

    sleep "$POLL_INTERVAL"
done

echo "[$(date)] Generating report..."
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
REPORT_LOG="$LOG_DIR/generate_report_${TIMESTAMP}.log"

conda run -n crism python "$PROJ/scripts/generate_report.py" \
    --output "$PROJ/reports/sweep_report_${TIMESTAMP}.md" \
    > "$REPORT_LOG" 2>&1

if [[ $? -eq 0 ]]; then
    echo "[$(date)] Report generated successfully: reports/sweep_report_${TIMESTAMP}.md"
    echo "[$(date)] Full log: $REPORT_LOG"
else
    echo "[$(date)] ERROR: Report generation failed. See $REPORT_LOG"
    exit 1
fi

echo "[$(date)] Watcher done."
