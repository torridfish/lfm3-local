#!/usr/bin/env bash
# Fetch and unpack the llama.cpp SYCL build (oneAPI path to the Arc iGPU) so it can be
# compared against the Vulkan build. Nothing outside runtimes/ and .venv/ is touched.
#
#   bash ./scripts/get-sycl.sh
#
# Runtime libs the SYCL build needs:
#   libiomp5.so            <- pip intel-openmp          (installed into .venv)
#   libsvml.so / libimf.so <- pip intel-cmplr-lib-rt    (installed into .venv)
#   libze_loader.so.1      <- Ubuntu package libze1     (needs apt, blocks the run)
#
# The last one has no pip equivalent, so without `sudo apt install libze1` this stops
# after unpacking and tells you why.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

TAG="${TAG:-b11985}"
NAME="llama-$TAG-bin-ubuntu-sycl-fp16-x64"
URL="https://github.com/ggml-org/llama.cpp/releases/download/$TAG/$NAME.tar.gz"
DEST="$LFM_ROOT/runtimes/ubuntu-sycl-x64"
RTLIB="$LFM_ROOT/.venv/lib"

mkdir -p "$LFM_ROOT/runtimes"
cd "$LFM_ROOT/runtimes"

if [ ! -f "$NAME.tar.gz" ]; then
  echo "downloading $URL"
  curl -fsSL --retry 3 -O "$URL"
else
  echo "archive already present: $NAME.tar.gz"
fi

rm -rf "$DEST"
mkdir -p "$DEST"
tar xzf "$NAME.tar.gz" -C "$DEST" --strip-components=1
echo "unpacked -> $DEST"

echo
echo "=== Intel runtime libs via pip (no root) ==="
have_omp=$(find "$RTLIB" -maxdepth 1 -name libiomp5.so 2>/dev/null | wc -l)
have_svml=$(find "$RTLIB" -maxdepth 1 -name libsvml.so 2>/dev/null | wc -l)
if [ "$have_omp" -eq 0 ] || [ "$have_svml" -eq 0 ]; then
  uv pip install --python "$LFM_ROOT/.venv/bin/python" -q intel-openmp intel-cmplr-lib-rt
fi
ls "$RTLIB"/libiomp5.so "$RTLIB"/libsvml.so 2>/dev/null || true

ze_count=$(ldconfig -p 2>/dev/null | grep -c "libze_loader.so.1" || true)
if [ "${ze_count:-0}" -eq 0 ]; then
  echo
  echo "BLOCKED: libze_loader.so.1 (Intel Level Zero loader) is missing."
  echo "  sudo apt install -y libze1"
  echo "  (libze1 belongs to the same package family as intel-opencl-icd, which the"
  echo "   OpenVINO GPU path needs - one apt run unblocks both experiments.)"
  exit 1
fi

M="${M17_QAD:-$LFM_MODELS/LFM3-1.7B-GGUF/LFM3-1.7B-QAD-Q4_0.gguf}"
echo
echo "=== SYCL benchmark (QAD-Q4_0) ==="
LD_LIBRARY_PATH="$RTLIB:${LD_LIBRARY_PATH:-}" "$DEST/llama-bench" \
  -m "$M" -p 256 -n 64 -r 2 -ngl 99
