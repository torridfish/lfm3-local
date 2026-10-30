#!/usr/bin/env bash
# Hardware tuning sweep for LFM3-1.7B on this machine.
# Measures pp/tg for thread counts, GPU-layer splits, ubatch sizes, KV cache types,
# and flash-attention settings. Prints one markdown table and saves it to logs/.
#
#   ./scripts/hw-sweep.sh              # everything (a few minutes)
#   ./scripts/hw-sweep.sh threads      # only CPU thread counts
#   ./scripts/hw-sweep.sh ngl          # only GPU-layer split (mixed CPU+iGPU)
#   ./scripts/hw-sweep.sh ub           # only ubatch
#   ./scripts/hw-sweep.sh kv           # only KV cache quantisation
#   ./scripts/hw-sweep.sh fa           # only flash attention
#
# Knobs: P=256 N=64 R=2 MODEL=$M17_QAD
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

STAGE="${1:-all}"
P="${P:-256}"; N="${N:-64}"; R="${R:-2}"
MODEL="${MODEL:-$M17_QAD}"
STAMP="$(date +%Y%m%d-%H%M%S)"
OUT="$LFM_ROOT/logs/hw-sweep-$STAMP.md"

PROFILE="$(powerprofilesctl get 2>/dev/null || echo n/a)"
LOG="$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | sed 's/^ //')"

row() { # label bin bench-args...
  local label="$1" bin="$2"; shift 2
  local out pp tg
  out="$("$bin" -m "$MODEL" -p "$P" -n "$N" -r "$R" "$@" 2>&1 || true)"
  # llama-bench adds/removes columns (threads, ngl, ...) so locate the test cell
  # by name and take the cell right after it, instead of using fixed positions.
  pp="$(awk -F'|' '{for(i=1;i<=NF;i++){c=$i; gsub(/^ +| +$/,"",c); if(c ~ /^pp[0-9]+$/){v=$(i+1); gsub(/ /,"",v); print v}}}' <<<"$out" | head -1 | cut -d'±' -f1)"
  tg="$(awk -F'|' '{for(i=1;i<=NF;i++){c=$i; gsub(/^ +| +$/,"",c); if(c ~ /^tg[0-9]+$/){v=$(i+1); gsub(/ /,"",v); print v}}}' <<<"$out" | head -1 | cut -d'±' -f1)"
  printf '| %-28s | %8s | %8s |\n' "$label" "${pp:-FAIL}" "${tg:-FAIL}" | tee -a "$OUT"
}

{
  echo "# hw-sweep $STAMP"
  echo
  echo "- cpu: $LOG"
  echo "- power profile: **$PROFILE**  (switch with \`powerprofilesctl set performance\`)"
  echo "- model: $(basename "$MODEL")  ·  -p $P -n $N -r $R"
  echo
  echo "| setup | pp$P (t/s) | tg$N (t/s) |"
  echo "|---|---:|---:|"
} > "$OUT"

do_threads() {
  for t in 2 4 6 8; do row "cpu t=$t" "$RT_CPU/llama-bench" -ngl 0 -t "$t"; done
  for t in 4 6; do row "vulkan+ngl99 t=$t" "$RT_VULKAN/llama-bench" -ngl 99 -t "$t"; done
}

do_ngl() {
  # LFM3-1.7B has 24 layers. tg favours CPU (bandwidth), pp favours iGPU (compute) —
  # this sweep looks for a split that gets both.
  for ngl in 0 8 12 16 20 24 99; do
    row "vulkan ngl=$ngl (t=$LFM_THREADS)" "$RT_VULKAN/llama-bench" -ngl "$ngl" -t "$LFM_THREADS"
  done
}

do_ub() {
  for ub in 128 256 512 1024 2048; do
    row "vulkan ngl=99 ub=$ub" "$RT_VULKAN/llama-bench" -ngl 99 -ub "$ub"
  done
  row "vulkan ngl=99 b=4096 ub=2048" "$RT_VULKAN/llama-bench" -ngl 99 -b 4096 -ub 2048
}

do_kv() {
  row "vulkan ngl=99 kv=f16" "$RT_VULKAN/llama-bench" -ngl 99
  row "vulkan ngl=99 kv=q8_0" "$RT_VULKAN/llama-bench" -ngl 99 -ctk q8_0 -ctv q8_0
  row "vulkan ngl=99 kv=q4_0" "$RT_VULKAN/llama-bench" -ngl 99 -ctk q4_0 -ctv q4_0
  row "cpu t=$LFM_THREADS kv=q8_0" "$RT_CPU/llama-bench" -ngl 0 -t "$LFM_THREADS" -ctk q8_0 -ctv q8_0
}

do_fa() {
  for fa in on off auto; do
    row "vulkan ngl=99 fa=$fa" "$RT_VULKAN/llama-bench" -ngl 99 -fa "$fa"
  done
}

case "$STAGE" in
  all)     do_threads; do_ngl; do_ub; do_kv; do_fa ;;
  threads) do_threads ;;
  ngl)     do_ngl ;;
  ub)      do_ub ;;
  kv)      do_kv ;;
  fa)      do_fa ;;
  *) echo "stages: all | threads | ngl | ub | kv | fa" >&2; exit 1 ;;
esac

echo
echo "saved → $OUT"
