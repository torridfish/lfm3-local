#!/usr/bin/env bash
# Concurrency soak test for a running llama-server.
#
#   ./scripts/soak.sh [port] [requests] [workers] [max_tokens]
#
# Checks that concurrent slots do not crash LFM3 (short-conv layers, llama.cpp #14670),
# and measures how much decode throughput actually parallelises. Delegates the load
# generation to scripts/soak.py (threads, not xargs, so failures are counted not fatal).
# The server must already be running with --parallel >= workers.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

PORT="${1:-8130}"; N="${2:-120}"; WORKERS="${3:-4}"; MAXTOK="${4:-32}"
LOG="${SOAK_LOG:-}"

echo "soaking :$PORT with $N requests, $WORKERS workers, max_tokens=$MAXTOK"
.venv/bin/python "$LFM_ROOT/scripts/soak.py" "$PORT" "$N" "$WORKERS" "$MAXTOK"

echo
if [ -n "$LOG" ] && [ -f "$LOG" ]; then
  bad=$(grep -icE "GGML_ASSERT|abort|segmentation|terminate called|core dumped" "$LOG" || true)
  echo "server log fatal lines: $bad"
else
  echo "server log not scanned (set SOAK_LOG=...)"
fi
alive=$(curl -s -o /dev/null -w '%{http_code}' -m 5 "http://127.0.0.1:$PORT/health" || true)
echo "server health after soak: ${alive:-down}"
