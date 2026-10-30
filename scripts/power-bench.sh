#!/usr/bin/env bash
# Power-efficiency sweep: tokens per joule, not just tokens per second.
#
#   ./scripts/power-bench.sh configs     # current power profile, backend/thread matrix
#   ./scripts/power-bench.sh profiles    # power-saver / balanced / performance, fixed config
#   ./scripts/power-bench.sh all         # both
#
# Power measurement, in order of preference:
#   1. Battery discharge (`power_now` on BAT*)  → whole-machine watts, what battery life feels
#      like. **Unplug the charger** or the number is charging power, not draw.
#   2. Intel RAPL (`/sys/class/powercap/intel-rapl:0/energy_uj`) → SoC watts. Needs root.
#   3. Nothing readable → the script still reports t/s and refuses to print tok/J.
#
# Workloads are measured separately so the energy is attributed correctly:
#   prefill  -p 2048 -n 0    (prompt processing)
#   decode   -p 0    -n 256  (token generation — what a chat turn is dominated by)
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

STAGE="${1:-all}"
MODEL="${MODEL:-$M17_QAD}"
R="${R:-3}"
PP_N="${PP_N:-2048}"    # prefill tokens
TG_N="${TG_N:-512}"     # decode tokens (longer window = less noise)
STAMP="$(date +%Y%m%d-%H%M%S)"
OUT="$LFM_ROOT/logs/power-$STAMP.md"
ORIG_PROFILE="$(powerprofilesctl get 2>/dev/null || echo "")"

# ---------------- power source detection ----------------
POWER_SRC="none"; BAT=""; RAPL="/sys/class/powercap/intel-rapl:0/energy_uj"
for b in /sys/class/power_supply/BAT*; do
  [ -r "$b/power_now" ] || continue
  if [ "$(cat "$b/status" 2>/dev/null)" = "Discharging" ]; then
    POWER_SRC="battery"; BAT="$b"; break
  fi
  [ -z "$BAT" ] && BAT="$b"
done
if [ "$POWER_SRC" = "none" ] && [ -r "$RAPL" ]; then POWER_SRC="rapl"; fi

WARN=""
case "$POWER_SRC" in
  battery) echo "power source: battery ($BAT) — whole-machine draw" ;;
  rapl)    echo "power source: RAPL package-0 — SoC draw only (excludes panel/SSD)" ;;
  *)
    POWER_SRC="none"
    if [ -n "$BAT" ]; then
      WARN="battery present but status=$(cat "$BAT/status" 2>/dev/null) — unplug the charger, then re-run for real numbers"
    else
      WARN="no readable power source (no battery, RAPL needs root)"
    fi
    echo "power source: NONE — $WARN" >&2
    ;;
esac

emitter() { # writes watts (float) lines until killed
  case "$POWER_SRC" in
    battery)
      while :; do cat "$BAT/power_now" 2>/dev/null | awk '{printf "%.4f\n", $1/1e6; fflush()}'; sleep 0.5; done ;;
    rapl)
      local prev cur
      prev=$(cat "$RAPL" 2>/dev/null || echo 0)
      while :; do
        sleep 0.5
        cur=$(cat "$RAPL" 2>/dev/null || echo "$prev")
        awk -v d=$(( cur > prev ? cur - prev : 0 )) 'BEGIN{printf "%.4f\n", d/500000; fflush()}'
        prev=$cur
      done ;;
    *) while :; do sleep 1; done ;;
  esac
}

SAMPLE_FILE=""
SAMPLE_PID=""

start_sampling() {
  SAMPLE_FILE="$(mktemp)"
  emitter > "$SAMPLE_FILE" 2>/dev/null &
  SAMPLE_PID=$!
  sleep 2                                  # discard the ramp-up window
  : > "$SAMPLE_FILE"
}

stop_sampling() { # → sets AVG_W
  kill "$SAMPLE_PID" 2>/dev/null || true
  wait "$SAMPLE_PID" 2>/dev/null || true
  AVG_W="$(awk 'NF {print $1}' "$SAMPLE_FILE" 2>/dev/null | sort -n \
             | awk '{a[NR]=$1} END {if (NR==0) print "0"; else if (NR%2) printf "%.2f", a[(NR+1)/2]; else printf "%.2f", (a[NR/2]+a[NR/2+1])/2}' \
             || echo 0)"
  AVG_W="${AVG_W:-0}"
  rm -f "$SAMPLE_FILE"
  if [ -z "${AVG_W:-}" ]; then AVG_W=0; fi
}

tkps() { # extract pp/tg t/s from llama-bench output on stdin
  awk -F'|' '{for(i=1;i<=NF;i++){c=$i; gsub(/^ +| +$/,"",c);
    if (c ~ /^(pp|tg)[0-9]+( @ d[0-9]+)?$/) {t=c; gsub(/ /,"",t); v=$(i+1); gsub(/ /,"",v); sub(/±.*/,"",v); printf "%s %s\n", t, v}}}'
}

measure() { # label bin args...   → appends a row
  local label="$1" bin="$2"; shift 2
  local pout tout pp tg
  pout="$("$bin" -m "$MODEL" -p "$PP_N" -n 0 -r "$R" "$@" 2>/dev/null | tkps || true)"
  tout="$("$bin" -m "$MODEL" -p 0 -n "$TG_N" -r "$R" "$@" 2>/dev/null | tkps || true)"
  pp="$(echo "$pout" | awk '$1 ~ /^pp/ {print $2; exit}')"
  tg="$(echo "$tout" | awk '$1 ~ /^tg/ {print $2; exit}')"

  start_sampling
  "$bin" -m "$MODEL" -p "$PP_N" -n "$TG_N" -r "$R" "$@" >/dev/null 2>&1 || true
  stop_sampling

  local tpj="n/a"
  if [ "$POWER_SRC" != "none" ] && [ -n "$tg" ] && awk -v a="$AVG_W" 'BEGIN{exit !(a>0)}'; then
    tpj="$(awk -v t="$tg" -v w="$AVG_W" 'BEGIN{printf "%.2f", t/w}')"
  fi
  printf '| %-30s | %8s | %8s | %9s | %7s |\n' \
    "$label" "${pp:-FAIL}" "${tg:-FAIL}" "${AVG_W}" "$tpj" | tee -a "$OUT"
}

set_profile() {
  if [ "$1" = "$(powerprofilesctl get 2>/dev/null)" ]; then return 0; fi
  powerprofilesctl set "$1" >/dev/null 2>&1
}

configs() {
  # Always record which power profile the numbers were taken under - comparing two
  # sweeps taken under different profiles is meaningless.
  local prof; prof="$(powerprofilesctl get 2>/dev/null || echo unknown)"
  measure "$prof · vulkan ngl=99 t=$LFM_THREADS" "$RT_VULKAN/llama-bench" -ngl 99 -t "$LFM_THREADS" -fa off -ub 1024
  measure "$prof · vulkan ngl=99 t=4"            "$RT_VULKAN/llama-bench" -ngl 99 -t 4 -fa off -ub 1024
  measure "$prof · cpu t=6"                      "$RT_CPU/llama-bench"    -ngl 0 -t 6
  measure "$prof · cpu t=4"                      "$RT_CPU/llama-bench"    -ngl 0 -t 4
  measure "$prof · cpu t=2"                      "$RT_CPU/llama-bench"    -ngl 0 -t 2
  measure "$prof · openvino CPU t=$LFM_THREADS"  "$RT_OPENVINO/llama-bench" -ngl 99 -t "$LFM_THREADS"
}

profiles() {
  for p in power-saver balanced performance; do
    if ! set_profile "$p"; then
      echo "| $p (could not switch — needs polkit auth) | - | - | - | - |" | tee -a "$OUT"
      continue
    fi
    sleep 3
    measure "$p · vulkan ngl=99" "$RT_VULKAN/llama-bench" -ngl 99 -t "$LFM_THREADS" -fa off -ub 1024
    measure "$p · cpu t=6"       "$RT_CPU/llama-bench"    -ngl 0 -t 6
  done
}

{
  echo "# power $STAMP"
  echo
  echo "- model: $(basename "$MODEL")  ·  prefill -p $PP_N, decode -n $TG_N, r=$R"
  echo "- power source: **$POWER_SRC**${BAT:+ ($BAT)}${WARN:+ — $WARN}"
  echo '- tok/J is decode tokens per joule (tg_tps / median_watts), higher is better'
  echo "- watts are the MEDIAN of 0.5 s samples; system-side readings still move ~±30% with" \
       "screen brightness and background load, so only compare rows from the same sweep"
  echo
  echo "| setup | pp (t/s) | tg (t/s) | W (median) | tok/J |"
  echo "|---|---:|---:|---:|---:|"
} > "$OUT"

case "$STAGE" in
  configs)  configs ;;
  profiles) profiles ;;
  all)      configs; profiles ;;
  *) echo "stages: configs | profiles | all" >&2; exit 1 ;;
esac

[ -n "$ORIG_PROFILE" ] && set_profile "$ORIG_PROFILE"

echo
echo "saved → $OUT"
if [ -n "$WARN" ]; then echo "⚠️  $WARN"; fi
