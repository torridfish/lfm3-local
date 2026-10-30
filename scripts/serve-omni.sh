#!/usr/bin/env bash
# Serve LFM3-Omni-2B (vision + audio in, text + speech out) — OpenAI-compatible API + web UI.
#
#   ./scripts/serve-omni.sh [port] [vulkan|cpu]
#
# Notes:
#   * Omni is a non-reasoning model: thinking is disabled here.
#   * Vision/audio projectors are loaded from mmproj-F16 next to the weights.
#   * `--mmproj-offload` keeps the vision encoder on the iGPU (big pp win).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

PORT="${1:-8111}"
BACKEND="${2:-vulkan}"
CTX="${CTX:-8192}"   # omni + media chunks are memory hungry; raise only if needed
SWA_FULL="${SWA_FULL:-1}"   # keeps prefix caching usable across turns
FA="${FA:-off}"
UB="${UB:-1024}"

case "$BACKEND" in
  vulkan) BIN="$RT_VULKAN/llama-server"; EXTRA=(-ngl 99 -fa "$FA" -ub "$UB" --mmproj-offload) ;;
  cpu)    BIN="$RT_CPU/llama-server";    EXTRA=(-ngl 0) ;;
  *) echo "unknown backend '$BACKEND' (vulkan|cpu)" >&2; exit 1 ;;
esac

SWA_ARGS=()
if [ "$SWA_FULL" = "1" ]; then SWA_ARGS+=(--swa-full); fi

echo "omni backend=$BACKEND ctx=$CTX port=$PORT fa=$FA swa_full=$SWA_FULL  ($BIN)"
exec "$BIN" \
  -m "$MOMNI" --mmproj "$MMPROJ" \
  --jinja -c "$CTX" -t "$LFM_THREADS" \
  --temp 0 --top-k 50 \
  --reasoning off \
  --host 127.0.0.1 --port "$PORT" --alias lfm3-omni \
  "${EXTRA[@]}" "${SWA_ARGS[@]}"
