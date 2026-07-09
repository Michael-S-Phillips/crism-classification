#!/usr/bin/env bash
# Launch (or restart) the N-D Visualizer app, detached from the calling shell
# so it survives terminal/agent exit. Usage:
#   bash scripts/label_quant/run_nd_visualizer.sh          # start on port 8502
#   bash scripts/label_quant/run_nd_visualizer.sh 8600     # custom port
set -e
PORT="${1:-8502}"
PROJ="$(cd "$(dirname "$0")/../.." && pwd)"
LOG="/tmp/nd_visualizer_${PORT}.log"

cd "$PROJ"

# Stop any existing instance on this port (best effort).
existing=$(pgrep -f "streamlit run scripts/label_quant/nd_visualizer_app.py.*--server.port $PORT" || true)
if [ -n "$existing" ]; then
    echo "stopping existing instance (pid $existing)"
    kill $existing 2>/dev/null || true
    sleep 2
fi

setsid nohup conda run -n crism streamlit run scripts/label_quant/nd_visualizer_app.py \
    --server.port "$PORT" --server.headless true \
    > "$LOG" 2>&1 < /dev/null &

# Wait for bind
for i in $(seq 1 20); do
    code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 1 "http://localhost:$PORT" 2>/dev/null || true)
    if [ "$code" = "200" ]; then
        echo "N-D visualizer UP: http://localhost:$PORT  (log: $LOG)"
        exit 0
    fi
    sleep 1
done
echo "ERROR: app did not come up within 20s — check $LOG" >&2
tail -20 "$LOG" >&2
exit 1
