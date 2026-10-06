#!/usr/bin/env bash
# Phase 0 smoke for a human with a gfx1030 or gfx1100 GPU.
# CI does not run this. The small MoE is Qwen3-30B-A3B (not Qwen3.5 / DSv4-Flash).
#
# Requires a ROCm kt-kernel build (CPUINFER_USE_ROCM=1) and the SGLang fork
# https://github.com/BlivionIaG/sglang-kt-rdna on PYTHONPATH, with SGLANG_USE_AITER=0.
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-30B-A3B}"
KT_METHOD="${KT_METHOD:-BF16}"
KT_WEIGHT_PATH="${KT_WEIGHT_PATH:-${MODEL_PATH}}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-}"
PORT="${PORT:-30000}"

if ! command -v rocminfo >/dev/null 2>&1; then
  echo "rocminfo is not on PATH. Install ROCm and run this on the GPU host." >&2
  exit 1
fi

ARCH="$(rocminfo | sed -n 's/.*Name:[[:space:]]*\(gfx[0-9a-z]*\).*/\1/p' | sort -u | head -n 1)"
if [[ -z "${ARCH}" ]]; then
  echo "rocminfo did not report a gfx arch." >&2
  exit 1
fi
echo "GPU arch: ${ARCH}"

case "${ARCH}" in
  gfx1030|gfx1031|gfx1032|gfx1033|gfx1034|gfx1035|gfx1036|gfx1037)
    : "${ATTENTION_BACKEND:=torch_native}"
    ;;
  gfx1100|gfx1101|gfx1102|gfx1103|gfx1150|gfx1151)
    : "${ATTENTION_BACKEND:=triton}"
    ;;
  *)
    echo "This smoke script expects gfx1030-class or gfx1100-class. Found ${ARCH}." >&2
    exit 1
    ;;
esac

python3 - <<'PY'
import sys
try:
    import kt_kernel
except Exception as exc:
    sys.exit(f"import kt_kernel failed: {exc}\nBuild with CPUINFER_USE_ROCM=1 --no-build-isolation --no-deps")
ext = getattr(kt_kernel, "kt_kernel_ext", None)
if ext is None:
    sys.exit("kt_kernel.kt_kernel_ext is missing. The ROCm extension did not import.")
for name in ("submit_with_cuda_stream", "rdna_wmma_allowed", "rdna_compiled_archs"):
    if not hasattr(ext, name):
        sys.exit(f"kt_kernel_ext.{name} is missing. Rebuild with CPUINFER_USE_ROCM=1.")
print("compiled archs:", ext.rdna_compiled_archs())
print("cpu variant:", getattr(ext, "__cpu_variant__", "unknown"))
PY

python3 - "${ARCH}" <<'PY'
import sys
import kt_kernel
ext = kt_kernel.kt_kernel_ext
arch = sys.argv[1]
allowed = ext.rdna_wmma_allowed(arch)
print(f"rdna_wmma_allowed({arch}) = {allowed}")
if arch.startswith("gfx103"):
    if allowed:
        sys.exit("gfx1030 reported WMMA allowed; the gate is wrong")
    try:
        ext.rdna_load_wmma(arch, "/does/not/exist.so")
    except RuntimeError as exc:
        text = str(exc)
        if "refusing to load" not in text:
            sys.exit(f"unexpected refusal: {text}")
        print("gfx1030 WMMA load refused without opening a library")
    else:
        sys.exit("gfx1030 WMMA load did not raise")
elif arch.startswith("gfx11"):
    if not allowed:
        sys.exit(f"{arch} should be allowed to load WMMA")
    print("gfx1100-class WMMA gate is open; not launching the kernel in this check")
PY

cat <<EOF

Host checks passed. Start the server from a checkout of
https://github.com/BlivionIaG/sglang-kt-rdna (sgl-kernel built for ${ARCH}):

  export SGLANG_USE_AITER=0
  python -m sglang.launch_server \\
    --model ${MODEL_PATH} \\
    --kt-weight-path ${KT_WEIGHT_PATH} \\
    --kt-method ${KT_METHOD} \\
    --kt-cpuinfer \${KT_CPUINFER:-\$(nproc)} \\
    --kt-num-gpu-experts 0 \\
    --attention-backend ${ATTENTION_BACKEND} \\
    --disable-cuda-graph \\
    --tp 1 \\
    --port ${PORT}

Then, from another shell, greedy-decode a fixed prompt:

  python - <<'PY'
  import json, urllib.request
  body = json.dumps({
      "model": "${MODEL_PATH}",
      "prompt": "The capital of France is",
      "max_tokens": 16,
      "temperature": 0,
  }).encode()
  req = urllib.request.Request(
      "http://127.0.0.1:${PORT}/v1/completions",
      data=body,
      headers={"Content-Type": "application/json"},
  )
  print(urllib.request.urlopen(req, timeout=600).read().decode())
  PY

Phase 0 smoke model is Qwen3-30B-A3B. If GPTQ_INT4 fails to load, rerun with
KT_METHOD=BF16 and MODEL_PATH=Qwen/Qwen3-30B-A3B and record the error.
Qwen3.5 (BF16/FP8/GPTQ_INT4) and DeepSeek-V4-Flash (MXFP4) are later targets;
their CPU kernels are already in this build. See docs/rdna/PHASE0.md.
EOF
