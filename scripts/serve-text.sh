#!/usr/bin/env bash
# Serve LFM3-1.7B (OpenAI-compatible API + web UI).
#
#   ./scripts/serve-text.sh [qad|q4km|q8] [auto|vulkan|cpu|openvino-gpu|openvino-npu] [port]
#
# Backend "auto" (the default) picks from CTX, because the crossover is measured:
#   CTX <= AUTO_SPLIT (4096) -> openvino-gpu : 2x prefill, +14% decode, +26% tok/J
#   CTX >  AUTO_SPLIT        -> vulkan       : 1.5x decode, and prefix cache re-evaluates
#                                              4 tokens per turn instead of 516
# At ~8K the cold prefill gap is 1176 vs 390 t/s, so long context must stay on Vulkan.
#
# Env overrides: NO_THINK=0 (keep thinking on), CTX, NP (slots), FA, UB, SWA_FULL=0,
#                AUTO_SPLIT, BACKEND, GGML_OPENVINO_DEVICE
# Defaults: qad + auto + :8110 (the port ~/.pi/agent/models.json points at), thinking off, fa off, ub 1024, 1 slot, swa-full on.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

QUANT="${1:-qad}"
BACKEND="${2:-${BACKEND:-auto}}"
PORT="${3:-8110}"
CTX="${CTX:-32768}"
# Context ceiling at which OpenVINO GPU stops winning. Measured: it wins on first-token
# latency up to ~4K (2.9 s vs 4.0 s at 2K), ties at 4K, and loses hard at 8K.
AUTO_SPLIT="${AUTO_SPLIT:-4096}"
NO_THINK="${NO_THINK:-1}"
# On this iGPU, flash attention is ~1.8x SLOWER on prompt processing (see README).
FA="${FA:-off}"
UB="${UB:-1024}"
# Slots: 1 = requests queue (safe). >1 =真並行, but LFM2's short-conv layers crashed
# with -np>1 (llama.cpp #14670) and LFM3 shares the architecture - test before relying on it.
NP="${NP:-1}"
# Keep the sliding-window cache at full size so a cached prefix survives the next turn.
# Measured: re-evaluated tokens per turn drop from ~1070 to ~22, multi-turn prefill
# 2.2-2.8 s -> 1.4-1.8 s. Costs device memory; fine up to 128K ctx on this box.
SWA_FULL="${SWA_FULL:-1}"

case "$QUANT" in
  qad|q4_0) MODEL="$M17_QAD" ;;
  q4km|q4_k_m) MODEL="$M17_Q4KM" ;;
  q8|q8_0) MODEL="$M17_Q8" ;;
  *) echo "unknown quant '$QUANT' (qad|q4km|q8)" >&2; exit 1 ;;
esac

if [ -z "${MODEL:-}" ] || [ ! -f "$MODEL" ]; then
  echo "quant '$QUANT' is not on disk (Q4_K_M and Q8_0 were deleted; QAD-Q4_0 is the default)." >&2
  echo "re-download with:" >&2
  echo "  hf download LiquidAI/LFM3-1.7B-GGUF LFM3-1.7B-$QUANT.gguf --local-dir ~/models/lfm3/LFM3-1.7B-GGUF" >&2
  exit 1
fi

WHY=""
case "$BACKEND" in
  auto)
    if [ ! -x "$RT_OPENVINO/llama-server" ]; then
      BACKEND=vulkan; WHY="openvino runtime not present"
    elif [ "$((CTX * NP))" -le "$AUTO_SPLIT" ]; then
      BACKEND=openvino-gpu; WHY="ctx $((CTX * NP)) <= $AUTO_SPLIT: GPU has 2x prefill here"
    else
      BACKEND=vulkan; WHY="ctx $((CTX * NP)) > $AUTO_SPLIT: Vulkan keeps long-context prefill and cheap prefix reuse"
    fi
    ;;
esac

case "$BACKEND" in
  vulkan)  BIN="$RT_VULKAN/llama-server"; EXTRA=(-ngl 99 -fa "$FA" -ub "$UB") ;;
  cpu)     BIN="$RT_CPU/llama-server";    EXTRA=(-ngl 0) ;;
  openvino|openvino-gpu)
    BIN="$RT_OPENVINO/llama-server"; EXTRA=(-ngl 99)
    export GGML_OPENVINO_DEVICE=GPU ;;
  openvino-npu)
    # Works, but measured 26 t/s / 2.41 tok/J and dies above ~768 ctx (PLAN.md 1c).
    BIN="$RT_OPENVINO/llama-server"; EXTRA=(-ngl 99 -b 512 -ub 512)
    export GGML_OPENVINO_DEVICE=NPU
    if [ "$CTX" -gt 512 ]; then
      echo "openvino-npu: forcing ctx 512 (the NPU fails with Compute error at 1024)" >&2
      CTX=512
    fi ;;
  *) echo "unknown backend '$BACKEND' (auto|vulkan|cpu|openvino-gpu|openvino-npu)" >&2; exit 1 ;;
esac

ARGS=(
  -m "$MODEL"
  --jinja
  -c "$CTX"
  --parallel "$NP"
  -t "$LFM_THREADS"
  --temp 0.1 --top-k 50 --repeat-penalty 1.1
  --host 127.0.0.1 --port "$PORT"
  --alias lfm3
  "${EXTRA[@]}"
)

if [ "$NO_THINK" = "1" ]; then
  ARGS+=(--reasoning off)          # modern flag; --chat-template-kwargs is deprecated
fi
if [ "$SWA_FULL" = "1" ] && [ "$BACKEND" = "vulkan" ]; then
  ARGS+=(--swa-full)   # only Vulkan has the SWA rebuild problem this fixes
fi

PROF="$(powerprofilesctl get 2>/dev/null || echo unknown)"
echo "backend=$BACKEND quant=$QUANT ctx=$CTX port=$PORT fa=$FA slots=$NP swa_full=$SWA_FULL profile=$PROF  ($BIN)"
if [ -n "$WHY" ]; then echo "auto: $WHY"; fi
case "$PROF:$BACKEND" in
  performance:*)   echo "hint: performance buys ~1% speed for ~30% more watts - 'powerprofilesctl set power-saver' is strictly better" ;;
  balanced:vulkan) echo "hint: 'powerprofilesctl set power-saver' keeps the same tg on Vulkan at ~5% less draw" ;;
esac

exec "$BIN" "${ARGS[@]}"
