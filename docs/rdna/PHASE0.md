# Phase 0: ROCm build for gfx1030 and gfx1100

Repository: [BlivionIaG/ktransformers-rdna](https://github.com/BlivionIaG/ktransformers-rdna). SGLang GPU ops: [BlivionIaG/sglang-kt-rdna](https://github.com/BlivionIaG/sglang-kt-rdna).

Phase 0 makes the kt-kernel ROCm build real, splits gfx1030 from gfx1100 WMMA, and records every CUDA-only dependency. It does not run a model. There is no AMD GPU in CI.

## Status

| Area | State |
|---|---|
| kt-kernel host module on ROCm | `CPUINFER_USE_ROCM=1` resolves `hipcc`, `amdhip64`, and `KT_ROCM_ARCHS` (default `gfx1030;gfx1100`, or `PYTORCH_ROCM_ARCH`). Links the HIP runtime. CPU experts stay host C++ |
| Per-arch device objects | `libkt_rdna_gfx1030.so` and `libkt_rdna_gfx1100.so` each get one `--offload-arch`. `libkt_rdna_wmma_gfx1100.so` is the only WMMA object. `rdna_load_wmma("gfx1030", ...)` returns before `dlopen` |
| FP8 layerwise transport | Host `hipMemcpyAsync` path in `fp8_layerwise_transport.cpp` when `KTRANSFORMERS_USE_ROCM` is set. The FP8 payload layout is unchanged |
| `KTransformersOps` (Marlin, GGUF dequant, `topk_softmax`) | Not built on ROCm. `cuda/setup.py` exits with the reason |
| SGLang / `sgl-kernel` | Handled in sglang-kt-rdna. Not compiled here |
| Legacy `local_chat.py` | Dead. Not modified except a pointer in `doc/en/ROCm.md` |
| Runtime tokens | Not run. Smoke script is `kt-kernel/scripts/rdna_smoke.sh` |

## What compiles

CI (`.github/workflows/rocm-rdna.yml`) uses the public image `rocm/dev-ubuntu-24.04:7.14.0-full`. Override the tag with `workflow_dispatch`. The lab pin is ROCm 7.14.0; `hipcc --offload-arch=gfx1030 -O3` is the device command.

1. **Contracts job** (no ROCm): Python source checks, `rdna_runtime.cpp`, and the CPU-only translation of `fp8_layerwise_transport.cpp`.
2. **hip-arch job**: `libkt_rdna_gfx1030.so`, `libkt_rdna_gfx1100.so`, `libkt_rdna_wmma_gfx1100.so`. Then `rdna_verify_arch_split.sh`, which disassembles the gfx1030 code object and fails on any `v_wmma_*` or `v_mfma_*` instruction. A successful link is not enough. `REQUIRE_DOT=1` also requires a packed DOT opcode (`v_dot2_f32_f16` or `v_dot2c_f32_f16`, `v_dot4_i32_i8` or `v_dot4c_i32_i8`, or an sdot form); that check is off by default. The script also checks that `wmma_gfx11.hip` fails for `--offload-arch=gfx1030` and for a mixed gfx1030+gfx1100 command. Then `fp8_layerwise_transport.cpp` against the HIP runtime headers.
3. **kt-kernel job**: `python3 setup.py build_ext` with `CPUINFER_USE_ROCM=1`, `CPUINFER_CPU_INSTRUCT=AVX2`, `CPUINFER_ENABLE_AVX512=OFF`, `CPUINFER_ENABLE_AMX=OFF`. That is the AVX2 CPU expert tier plus the HIP host bridge plus the same device libraries. llama.cpp is forced off HIP so ggml does not absorb a second GPU backend.

NVIDIA builds do not take the `KTRANSFORMERS_USE_ROCM` branch. `fp8_layerwise_transport.cpp` still includes `cuda_runtime_api.h` when `KTRANSFORMERS_USE_CUDA` is set.

## What is gated

- **Marlin** (`gptq_marlin_gemm`): Leave. Clear error from `kt-kernel/cuda/setup.py`.
- **GGUF GPU dequant**: Leave. CPU `LLAMAFILE` remains.
- **`topk_softmax` in this tree**: Leave. sgl-kernel in sglang-kt-rdna owns it.
- **WMMA on gfx1030**: compile error in the WMMA file, CMake `FATAL_ERROR` if a WMMA library is requested for a non-gfx11 arch, and a runtime refusal in `rdna_wmma_allowed` / `rdna_load_wmma`.
- **INT4/W4A16 layerwise payload**: not in this phase. The transport copies the existing FP8 buffers. Generalizing the payload is a later optimization.
- **hipGraph capture of `hipLaunchHostFunc`**: not tested. Smoke keeps `--disable-cuda-graph`.

## CPU paths under the ROCm config

Qwen3.5 and DeepSeek-V4-Flash experts stay on these kernels. Details and file paths are in [INVENTORY.md](INVENTORY.md).

| Model | Method | In the CI ROCm build (`AVX2`, AVX512 umbrella off) | When `CPUINFER_CPU_INSTRUCT=AVX512` or `FANCY` |
|---|---|---|---|
| Qwen3.5 | `BF16` | `AVX2BF16_MOE` | also `AMXBF16_MOE` if `__AVX512F__` |
| Qwen3.5 | `FP8` | `AVX2FP8_MOE` | also `AMXFP8_MOE` |
| Qwen3.5 | `GPTQ_INT4` | `AVX2GPTQInt4_MOE` and the AVX-VNNI-256 sources (used only if the CPU has `avx_vnni`) | no extra class. There is no AMX GPTQ_INT4 |
| DeepSeek-V4-Flash | `MXFP4` | `AVX2MXFP4_MOE` | also `AMXFP4_KGroup_MOE` (AVX512-BF16, not AMX tiles) |

`CPUINFER_ENABLE_AMX=ON` adds AMX-tile INT4/INT8 (`AMXInt4_MOE`, `AMXInt8_MOE`). Those are not the Qwen3.5 GPTQ or DSv4-Flash MXFP4 layouts.

## Phase 0 smoke

Model: **Qwen/Qwen3-30B-A3B** (or the GPTQ-Int4 repo with `--kt-method GPTQ_INT4`). Not Qwen3.5 and not DeepSeek-V4-Flash.

On a machine with the GPU, ROCm, a ROCm kt-kernel install, and the sglang-kt-rdna checkout:

```bash
bash kt-kernel/scripts/rdna_smoke.sh
```

The script checks `rocminfo`, `submit_with_cuda_stream`, and the WMMA gate, then prints the server command: `--kt-num-gpu-experts 0`, `--disable-cuda-graph`, `--attention-backend torch_native` on gfx1030 and `triton` on gfx1100, `SGLANG_USE_AITER=0`. Send one greedy completion to `/v1/completions`. If `GPTQ_INT4` fails to load, rerun with `KT_METHOD=BF16`.

## Phase 1, in order

1. **sgl-kernel for gfx1030 and gfx1100** (elementwise, top-k, moe align, RoPE). Handled in sglang-kt-rdna. Required before the smoke server imports.
2. **Attention.** gfx1030: `fa_rdna2` GQA (D=128/256; D=64 stays torch-native). gfx1100: Triton first; WMMA attention only inside `libkt_rdna_wmma_gfx1100.so`.
3. **RMSNorm.** `forward_native` until a small HIP kernel exists. AITER stays off.
4. **Dense GEMM / GEMV.** gfx1030: `q_gemm_rdna2` (W4A16) and `gemv_f16_rdna2`. gfx1100: rocBLAS for fp16, WMMA GEMM only in the WMMA library.
5. **Hot experts.** `moe_q_gemm_rdna2` on gfx1030. gfx1100 WMMA MoE GEMM only in the WMMA library. `--kt-num-gpu-experts 0` does not need this for the first smoke.
6. **Not phase 1:** MLA, FP8 GEMM, Marlin, FlashInfer, the INT4 layerwise payload, custom all-reduce, the legacy server.

Kernel sources from the parallel port drop in as `kt-kernel/rdna/gfx1030/*.hip`, `kt-kernel/rdna/gfx1100/*.hip`, and `kt-kernel/rdna/gfx1100/wmma/*.hip`. Those directories are not created here. CMake picks them up without putting WMMA objects into the gfx1030 link.

## Build

```bash
export CPUINFER_USE_ROCM=1
export CPUINFER_ENABLE_AMX=OFF
export CPUINFER_CPU_INSTRUCT=AVX2   # or AVX512 / FANCY on hosts that have it
export KT_ROCM_ARCHS=gfx1030;gfx1100
export ROCM_PATH=/opt/rocm          # optional
cd kt-kernel
pip install . --no-build-isolation --no-deps
```

Device libraries alone:

```bash
cmake -S kt-kernel/rocm -B build/rocm -DKT_ROCM_ARCHS='gfx1030;gfx1100'
cmake --build build/rocm
bash kt-kernel/scripts/rdna_verify_arch_split.sh build/rocm
```
