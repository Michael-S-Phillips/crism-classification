#!/usr/bin/env bash
# Run all 8 model families sequentially, skipping already-completed checkpoints.
# Usage:
#   bash scripts/run_all_models.sh              # with W&B
#   bash scripts/run_all_models.sh --no_wandb   # without W&B
#
# Run in background that survives logout:
#   nohup bash scripts/run_all_models.sh > logs/nohup.out 2>&1 &

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJ_DIR="$(dirname "$SCRIPT_DIR")"
WANDB_FLAG="${1:-}"
LOG_DIR="$PROJ_DIR/logs"
CKPT_DIR="$PROJ_DIR/checkpoints"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_DIR/run_all_${TIMESTAMP}.log"
PID_FILE="$LOG_DIR/training.pid"

mkdir -p "$LOG_DIR"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

# Write PID file so monitor_training.sh can track this process
echo $$ > "$PID_FILE"
log "PID $$ written to $PID_FILE"

run_model() {
    local model="$1"
    shift

    # Determine checkpoint path to skip already-finished models
    if [[ "$model" == logreg || "$model" == svc || "$model" == rf || "$model" == xgb || "$model" == lgbm ]]; then
        local ckpt="$CKPT_DIR/${model}_model.pkl"
    else
        local ckpt="$CKPT_DIR/${model}_best.pt"
    fi

    if [[ -f "$ckpt" ]]; then
        log "===== Skipping $model (checkpoint exists: $ckpt) ====="
        return 0
    fi

    log "===== Starting: $model ====="
    conda run -n crism python "$PROJ_DIR/scripts/train.py" \
        --model "$model" $WANDB_FLAG "$@" \
        2>&1 | tee -a "$LOG_FILE"
    log "===== Finished: $model ====="
}

log "Full model run starting. Log: $LOG_FILE"
log "W&B flag: '${WANDB_FLAG}'"

# --- Fast sklearn models ---
run_model logreg
run_model svc
run_model rf   --n_estimators 300
run_model xgb  --n_estimators 300 --max_depth 6 --learning_rate 0.05
run_model lgbm --n_estimators 300 --learning_rate 0.05

# --- Pre-cache spatial patches (skip if all splits already written) ---
CACHE_DIR="$PROJ_DIR/data/patch_cache"
CACHE_NEEDED=0
for split in train val test; do
    [[ ! -f "$CACHE_DIR/${split}_patches_p7.npy" ]] && CACHE_NEEDED=1 && break
done
if [[ "$CACHE_NEEDED" -eq 1 ]]; then
    log "===== Caching spatial patches ====="
    conda run -n crism python "$PROJ_DIR/scripts/cache_patches.py" \
        2>&1 | tee -a "$LOG_FILE"
    log "===== Patch cache complete ====="
else
    log "===== Patch cache already exists, skipping ====="
fi

# --- Neural models ---
run_model mlp  --epochs 200 --patience 15 --lr 1e-3 --batch_size 512
run_model cnn  --epochs 200 --patience 15 --lr 5e-4 --batch_size 256 --patch_size 7
run_model vit  --epochs 200 --patience 15 --lr 5e-4 --batch_size 256 --patch_size 7 \
               --embed_dim 128 --n_heads 4 --n_layers 4

log "All models complete."

# Clean up PID file on clean exit
rm -f "$PID_FILE"
