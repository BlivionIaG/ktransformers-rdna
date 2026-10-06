#!/usr/bin/env bash
# Compile every HIP translation unit for gfx1030 and, separately, gfx1100.
# No GPU is required. A gfx1100 WMMA file is not part of this module; the
# dot_arch_probe.cu device pass fails the build if the packed-DOT bodies
# were not enabled for the offload arch.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
HIPCC="${HIPCC:-$ROCM_PATH/bin/hipcc}"
if [[ ! -x "$HIPCC" ]]; then
  echo "hipcc not found at $HIPCC" >&2
  exit 1
fi

PYTHON="${PYTHON:-python3}"
TORCH_INC="$("$PYTHON" - <<'PY'
import os
import torch
print(os.path.join(os.path.dirname(torch.__file__), "include"))
PY
)"
PY_INC="$("$PYTHON" -c 'import sysconfig; print(sysconfig.get_path("include"))')"

OUT="${OUT:-/tmp/ktransformers-rdna-compile}"
mkdir -p "$OUT"
# hipcc adds $ROCM_PATH/include as -idirafter, after /usr/include. Ubuntu's
# HIP 5.7 headers in /usr/include/hip then win over ROCm 6.4. A distinct
# directory that only contains a symlink to the 6.4 hip/ tree is searched
# as a real -I and is not deduped against that idirafter.
HIP_FRONT="$OUT/hip-include"
mkdir -p "$HIP_FRONT"
ln -sfn "$ROCM_PATH/include/hip" "$HIP_FRONT/hip"
INCLUDE=(
  -I"$HIP_FRONT"
  -I"$ROOT/csrc"
  -I"$TORCH_INC"
  -I"$TORCH_INC/torch/csrc/api/include"
  -I"$PY_INC"
  -I"$ROCM_PATH/include"
)
# ROCm 6.4 hip-dev no longer ships cuda_runtime_api.h. PyTorch's ATen/cuda
# headers still include it. Prefer a real header when the install has one.
if [[ ! -f "$ROCM_PATH/include/cuda_runtime_api.h" ]]; then
  INCLUDE=(-I"$ROOT/csrc/rocm_cuda_compat" "${INCLUDE[@]}")
fi
# ROCm's clang looks for libstdc++ next to a non-multiarch path. Ubuntu
# keeps those headers under /usr/include/c++/<ver> and
# /usr/include/<triple>/c++/<ver>. Append whichever exist.
STD_FLAGS=()
ver="$(ls -d /usr/include/c++/[0-9]* 2>/dev/null | sort -V | tail -1 || true)"
if [[ -n "$ver" && -d "$ver" ]]; then
  STD_FLAGS+=(-idirafter "$ver")
  triple="$(gcc -dumpmachine 2>/dev/null || true)"
  multi="/usr/include/${triple}/c++/$(basename "$ver")"
  if [[ -n "$triple" && -d "$multi" ]]; then
    STD_FLAGS+=(-idirafter "$multi")
  fi
fi

FLAGS=(
  -std=c++17
  -fPIC
  -D__HIP_PLATFORM_AMD__=1
  -DUSE_ROCM=1
  -DC10_CUDA_NO_CMAKE_CONFIGURE_FILE
  -DTORCH_EXTENSION_NAME=ktransformers_rdna_compile_check
  -include "$ROOT/csrc/rocm_cuda_compat/torch_api_macros.h"
  -include "$ROOT/csrc/rdna_dot_arch.h"
  "${STD_FLAGS[@]}"
  "${INCLUDE[@]}"
  -O1
  -Wno-deprecated-pragma
)
SOURCES=(
  dot_arch_probe.cu
  gemv_f16_rdna2.cu
  q_gemm_rdna2.cu
  q_gemm_rdna2_prefill.cu
  moe_q_gemm_rdna2.cu
  fa_rdna2.cu
  bindings.cpp
)

for arch in gfx1030 gfx1100; do
  echo "== $arch =="
  for src in "${SOURCES[@]}"; do
    obj="$OUT/${arch}-$(basename "$src").o"
    echo "  $src"
    "$HIPCC" "${FLAGS[@]}" --offload-arch="$arch" -c "$ROOT/csrc/$src" -o "$obj"
  done
done

echo "compiled gfx1030 and gfx1100 objects in $OUT"
