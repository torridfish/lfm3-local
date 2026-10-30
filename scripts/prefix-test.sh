#!/usr/bin/env bash
# Does prefix caching actually save prefill work on this model/machine?
#
# LFM3 is a hybrid short-conv + sliding-window-attention model, so "the prefix is cached"
# is not a given: SWA layers have their own cache and llama-server may have to drop or
# rebuild it. This measures real prompt tokens evaluated vs reused on a growing
# conversation over an ~11.7K-token document.
#
#   ./scripts/prefix-test.sh              # default cache reuse vs --cache-reuse 0
#   ./scripts/prefix-test.sh all          # + 8K context and SWA-full variants
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

STAGE="${1:-basic}"
DOC="${DOC:-$LFM_ROOT/testdata/longprompt.txt}"
CTX="${CTX:-32768}"
STAMP="$(date +%Y%m%d-%H%M%S)"
OUT="$LFM_ROOT/logs/prefix-$STAMP.md"
PORT="${PORT:-8102}"

waitok() {
  for _ in $(seq 1 120); do
    [ "$(curl -s -o /dev/null -w '%{http_code}' -m 2 "http://127.0.0.1:$1/health" 2>/dev/null)" = "200" ] && return 0
    sleep 1
  done
  return 1
}

port_busy() { ss -ltn 2>/dev/null | grep -q ":$1 "; }

SERVER_PID=""
cleanup() {
  if [ -n "$SERVER_PID" ]; then kill "$SERVER_PID" 2>/dev/null || true; wait "$SERVER_PID" 2>/dev/null || true; fi
}
trap cleanup EXIT INT TERM

# run_one <label> [extra server args...]
run_one() {
  local label="$1"; shift
  local log="$LFM_ROOT/logs/prefix-$STAMP-$label.log"
  if port_busy "$PORT"; then
    echo "port $PORT is already in use — refusing to test against a stale server" >&2
    echo "| $label | abort | - | - | - | - | port $PORT busy |" | tee -a "$OUT"
    return 0
  fi
  "$RT_VULKAN/llama-server" -m "$M17_QAD" -ngl 99 -t "$LFM_THREADS" -fa off -ub 1024 \
    -c "$CTX" --jinja --reasoning off --host 127.0.0.1 --port "$PORT" "$@" > "$log" 2>&1 &
  SERVER_PID=$!
  local pid=$SERVER_PID
  if ! waitok "$PORT"; then
    echo "| $label | server failed to start | - | - | - | - | see $(basename "$log") |" | tee -a "$OUT"
    kill "$pid" 2>/dev/null || true
    return 0
  fi
  .venv/bin/python "$LFM_ROOT/scripts/prefix_probe.py" "$PORT" "$label" "$DOC" | tee -a "$OUT" || true
  kill "$pid" 2>/dev/null || true
  wait "$pid" 2>/dev/null || true
  SERVER_PID=""
  sleep 2
}

{
  echo "# prefix-cache test $STAMP"
  echo
  echo "- doc: $(basename "$DOC") ($(wc -c < "$DOC") bytes)  ·  ctx $CTX  ·  Vulkan, QAD-Q4_0"
  echo "- \`prompt_n\` = tokens actually evaluated, \`cache_n\` = tokens reused from cache"
  echo
  echo "| setup | request | prompt_n | cache_n | pp (t/s) | prefill s | answer |"
  echo "|---|---|---:|---:|---:|---:|---|"
} > "$OUT"

run_one "default"
run_one "cache-reuse-0" --cache-reuse 0

# The extra variants probe *why* ~1k tokens get re-evaluated each turn:
# LFM3 mixes sliding-window attention with short convolutions, so the SWA cache has to be
# rebuilt around the reuse point unless it is kept at full size.
if [ "$STAGE" = "all" ]; then
  run_one "swa-full"        --swa-full
  run_one "ctxcp-0"         --ctx-checkpoints 0
  run_one "ctxcp-4"         --ctx-checkpoints 4
fi

echo
echo "saved → $OUT"
