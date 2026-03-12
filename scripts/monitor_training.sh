#!/usr/bin/env bash
# Hourly monitor for the full training run.
# Checks if the run is alive; restarts from first missing checkpoint if not.
# Cron entry:
#   0 * * * * bash /mnt/crism/MRDR/crism_classification/scripts/monitor_training.sh >> /mnt/crism/MRDR/crism_classification/logs/monitor.log 2>&1

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJ_DIR="$(dirname "$SCRIPT_DIR")"
LOG_DIR="$PROJ_DIR/logs"
CKPT_DIR="$PROJ_DIR/checkpoints"
PID_FILE="$LOG_DIR/training.pid"
NOHUP_LOG="$LOG_DIR/nohup.out"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [monitor] $*"
}

# Expected checkpoints in completion order
SKLEARN_CKPTS=(
    "logreg:${CKPT_DIR}/logreg_model.pkl"
    "svc:${CKPT_DIR}/svc_model.pkl"
    "rf:${CKPT_DIR}/rf_model.pkl"
    "xgb:${CKPT_DIR}/xgb_model.pkl"
    "lgbm:${CKPT_DIR}/lgbm_model.pkl"
)
TORCH_CKPTS=(
    "mlp:${CKPT_DIR}/mlp_best.pt"
    "cnn:${CKPT_DIR}/cnn_best.pt"
    "vit:${CKPT_DIR}/vit_best.pt"
)
ALL_CKPTS=("${SKLEARN_CKPTS[@]}" "${TORCH_CKPTS[@]}")

log "=== Monitor check start ==="

# List which models are done
done_count=0
first_missing=""
for entry in "${ALL_CKPTS[@]}"; do
    model="${entry%%:*}"
    ckpt="${entry##*:}"
    if [[ -f "$ckpt" ]]; then
        log "  [done]    $model  ($ckpt)"
        (( done_count++ ))
    else
        log "  [missing] $model  ($ckpt)"
        if [[ -z "$first_missing" ]]; then
            first_missing="$model"
        fi
    fi
done

if [[ $done_count -eq ${#ALL_CKPTS[@]} ]]; then
    log "All ${#ALL_CKPTS[@]} models complete. Nothing to do."
    exit 0
fi

log "$done_count / ${#ALL_CKPTS[@]} models complete. Next: $first_missing"

# Check if a training process is currently running
is_alive=false
if [[ -f "$PID_FILE" ]]; then
    pid="$(cat "$PID_FILE")"
    if kill -0 "$pid" 2>/dev/null; then
        is_alive=true
        log "Training process PID $pid is alive."
    else
        log "PID file exists ($pid) but process is dead."
        rm -f "$PID_FILE"
    fi
else
    log "No PID file found."
fi

# Also check for any lingering conda/python training processes
if ! $is_alive; then
    if pgrep -f "train\.py" > /dev/null 2>&1; then
        log "Found orphaned train.py process — treating as alive, skipping restart."
        is_alive=true
    fi
fi

if $is_alive; then
    log "Training is running. Tail of $NOHUP_LOG:"
    tail -5 "$NOHUP_LOG" 2>/dev/null | sed 's/^/    /'
    log "=== Monitor check end (no action needed) ==="
    exit 0
fi

# Training is not running — restart it
log "Training is NOT running. Restarting from first missing checkpoint: $first_missing"
log "run_all_models.sh will skip completed models automatically."

mkdir -p "$LOG_DIR"
nohup bash "$SCRIPT_DIR/run_all_models.sh" >> "$NOHUP_LOG" 2>&1 &
new_pid=$!
echo $new_pid > "$PID_FILE"
log "Relaunched run_all_models.sh as PID $new_pid"
log "=== Monitor check end (restarted) ==="
