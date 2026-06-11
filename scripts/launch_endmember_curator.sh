#!/usr/bin/env bash
# Launch the endmember curator Streamlit app.
set -euo pipefail
cd "$(dirname "$0")/.."

# Build the polygon-spectra cache if missing
if [ ! -f data/endmember_curator/polygon_spectra.parquet ]; then
    echo "polygon cache not found — running precompute…"
    PYTHONPATH=. conda run -n crism --no-capture-output python -m endmember_curator.precompute
fi

PORT="${PORT:-8602}"
echo "starting endmember curator on http://localhost:${PORT}"
PYTHONPATH=. conda run -n crism --no-capture-output streamlit run \
    endmember_curator/app.py \
    --server.port "${PORT}" \
    --browser.gatherUsageStats false
