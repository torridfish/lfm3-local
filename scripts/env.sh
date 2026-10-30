#!/usr/bin/env bash
# Shared paths + env for the lfm lab. Source this, don't execute it.
#
# Works when sourced from bash *or* zsh: bash sets BASH_SOURCE, zsh sets $0 to
# the sourced file. Resolving it wrong silently produced paths like
# ~/Code/runtimes/... , so the location is verified below.

_lfm_src="${BASH_SOURCE[0]:-$0}"
LFM_ROOT="$(cd "$(dirname "$_lfm_src")/.." && pwd)"
unset _lfm_src

if [ ! -d "$LFM_ROOT/runtimes" ]; then
  echo "env.sh: LFM_ROOT resolved to '$LFM_ROOT' which has no runtimes/ dir." >&2
  echo "env.sh: source this file by path (source ~/Code/lfm/scripts/env.sh), not via a relative cd." >&2
  return 1 2>/dev/null || exit 1
fi
LFM_MODELS="${LFM_MODELS:-$HOME/models/lfm3}"

RT_VULKAN="$LFM_ROOT/runtimes/ubuntu-vulkan-x64"
RT_CPU="$LFM_ROOT/runtimes/ubuntu-x64"
RT_OPENVINO="$LFM_ROOT/runtimes/ubuntu-openvino-2026.4.1-x64"

M17_QAD="$LFM_MODELS/LFM3-1.7B-GGUF/LFM3-1.7B-QAD-Q4_0.gguf"
# Q4_K_M and Q8_0 were deleted after the quantisation matrix showed QAD-Q4_0 is both
# smaller and faster (see PLAN.md section 2). Re-download if a comparison needs them.
_d="$LFM_MODELS/LFM3-1.7B-GGUF"
M17_Q4KM=""; M17_Q8=""
[ -f "$_d/LFM3-1.7B-Q4_K_M.gguf" ] && M17_Q4KM="$_d/LFM3-1.7B-Q4_K_M.gguf"
[ -f "$_d/LFM3-1.7B-Q8_0.gguf" ] && M17_Q8="$_d/LFM3-1.7B-Q8_0.gguf"
unset _d

MOMNI="$LFM_MODELS/LFM3-Omni-2B-GGUF/LFM3-Omni-2B-Q4_K_M.gguf"
# mmproj quant: F16 (906 MB), Q8_0 (484 MB), Q4_K_M (279 MB). Vision encoding measured at
# 222-445 ms for all three (noise), i.e. quantising the projector buys memory, not speed.
# Default to Q4_K_M when present; falls back to F16.
_MMDIR="$LFM_MODELS/LFM3-Omni-2B-GGUF"
if [ -f "$_MMDIR/mmproj-LFM3-Omni-2B-Q4_K_M.gguf" ]; then
  MMPROJ="${MMPROJ:-$_MMDIR/mmproj-LFM3-Omni-2B-Q4_K_M.gguf}"
else
  MMPROJ="${MMPROJ:-$_MMDIR/mmproj-LFM3-Omni-2B-F16.gguf}"
fi
unset _MMDIR

# 6 physical cores is the measured sweet spot; 8 regresses.
LFM_THREADS="${LFM_THREADS:-6}"

# OpenVINO plugins in the prebuilt tarball need their own dir on the loader path.
# Device selection: CPU (only one that works today) / GPU / NPU - needs the matching
# Intel driver installed system-wide. Set GGML_OPENVINO_DEVICE=GPU to try.
if [ -d "$RT_OPENVINO" ]; then
  export LD_LIBRARY_PATH="$RT_OPENVINO:${LD_LIBRARY_PATH:-}"
  export GGML_OPENVINO_DEVICE="${GGML_OPENVINO_DEVICE:-CPU}"
fi

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
