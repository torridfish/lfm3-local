#!/usr/bin/env bash
# Benchmark matrix for LFM3 on this machine.
#
#   ./scripts/bench.sh          # default: 1.7B quant x backend sweep
#   ./scripts/bench.sh omni     # omni-2b sweep
#
# Writes markdown tables to logs/bench-<stamp>.md and prints them.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

WHAT="${1:-text}"
STAMP="$(date +%Y%m%d-%H%M%S)"
OUT="$LFM_ROOT/logs/bench-$STAMP.md"
P="${P:-256}"; N="${N:-64}"; R="${R:-2}"

run() { # backend-bin model description
  "$1" -m "$2" -p "$P" -n "$N" -r "$R" "${@:4}" 2>&1 \
    | grep -E "pp$P|tg$N" \
    | awk -F'|' -v d="$3" 'NF>6 {gsub(/ /,"",$7); printf "| %s | %s | %s |\n", d, $7, $8}' \
    | sed 's/ *±[^|]*//'
}

{
  echo "# bench $STAMP"
  echo
  echo "| setup | test | t/s |"
  echo "|---|---|---|"
  if [ "$WHAT" = "omni" ]; then
    run "$RT_VULKAN/llama-bench" "$MOMNI" "omni Q4_K_M vulkan" -ngl 99
    run "$RT_CPU/llama-bench"    "$MOMNI" "omni Q4_K_M cpu t=$LFM_THREADS" -ngl 0 -t "$LFM_THREADS"
  else
    for q in "QAD-Q4_0:$M17_QAD" "Q4_K_M:${M17_Q4KM:-}" "Q8_0:${M17_Q8:-}"; do
      name="${q%%:*}"; path="${q#*:}"
      [ -n "$path" ] && [ -f "$path" ] || { echo "| $name | (not downloaded, skipped) | - |" | tee -a "$OUT"; continue; }
      run "$RT_VULKAN/llama-bench" "$path" "$name vulkan" -ngl 99
      run "$RT_CPU/llama-bench"    "$path" "$name cpu t=$LFM_THREADS" -ngl 0 -t "$LFM_THREADS"
      run "$RT_OPENVINO/llama-bench" "$path" "$name openvino(CPU)" -ngl 99
    done
  fi
} | tee "$OUT"

echo
echo "saved → $OUT"
