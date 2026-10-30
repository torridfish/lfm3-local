#!/usr/bin/env bash
# One-shot Omni-2B calls (no server needed).
#
#   ./scripts/omni.sh image <file>  "<prompt>"
#   ./scripts/omni.sh video <file>  "<prompt>"
#   ./scripts/omni.sh audio <file>  "<prompt>"      # for ASR use: "Perform ASR in English."
#   ./scripts/omni.sh tts   "<text>" [out.wav]
#   ./scripts/omni.sh ask   "<prompt>"
#
# Backend defaults to the iGPU; pass BACKEND=cpu to compare.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

BACKEND="${BACKEND:-vulkan}"
case "$BACKEND" in
  vulkan) RT="$RT_VULKAN"; NGL=(-ngl 99) ;;
  cpu)    RT="$RT_CPU";    NGL=(-ngl 0) ;;
  *) echo "BACKEND must be vulkan|cpu" >&2; exit 1 ;;
esac

MODE="${1:-}"; shift || true

case "$MODE" in
  image)
    FILE="$1"; PROMPT="$2"
    exec "$RT/llama-mtmd-cli" -m "$MOMNI" --mmproj "$MMPROJ" "${NGL[@]}" -t "$LFM_THREADS" \
      --image "$FILE" -p "$PROMPT" --temp 0 --top-k 50 ;;
  video)
    FILE="$1"; PROMPT="$2"
    exec "$RT/llama-mtmd-cli" -m "$MOMNI" --mmproj "$MMPROJ" "${NGL[@]}" -t "$LFM_THREADS" \
      --video "$FILE" -sys "Answer questions about the video using what you see and hear." \
      -p "$PROMPT" --temp 0 ;;
  audio)
    FILE="$1"; PROMPT="${2:-Perform ASR in English.}"
    exec "$RT/llama-mtmd-cli" -m "$MOMNI" --mmproj "$MMPROJ" "${NGL[@]}" -t "$LFM_THREADS" \
      --audio "$FILE" -sys "$PROMPT" -p "<__media__>" --temp 0 ;;
  tts)
    TEXT="$1"; OUT="${2:-$LFM_ROOT/testdata/tts-$(date +%H%M%S).wav}"
    exec "$RT/llama-tts" -m "$MOMNI" -mm "$MMPROJ" "${NGL[@]}" -t "$LFM_THREADS" \
      -p "$TEXT" -o "$OUT" ;;
  ask)
    exec "$RT/llama-mtmd-cli" -m "$MOMNI" --mmproj "$MMPROJ" "${NGL[@]}" -t "$LFM_THREADS" \
      -p "$1" --temp 0 --top-k 50 ;;
  *)
    sed -n '2,12p' "${BASH_SOURCE[0]}" >&2; exit 1 ;;
esac
