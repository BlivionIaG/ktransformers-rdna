# RDNA dependency inventory

Repository: [BlivionIaG/ktransformers-rdna](https://github.com/BlivionIaG/ktransformers-rdna).

The serving host is SGLang + kt-kernel. SGLang's GPU stack (`sgl-kernel`, attention, Marlin, FlashInfer, Triton, CUDA graphs) is **handled in [BlivionIaG/sglang-kt-rdna](https://github.com/BlivionIaG/sglang-kt-rdna)**. This file inventories what lives in this repository, plus a cross-reference so those SGLang ops are not forgotten.

Silicon used for the decisions:

- **gfx1030** (RDNA2, e.g. Radeon Pro V620): Wave32, packed DOT only (`v_dot2_f32_f16`, `sdot4`). No WMMA, no MFMA, no FP8.
- **gfx1100** (RDNA3, e.g. 7900 XTX / W7800): WMMA 16x16x16, still no FP8. WMMA is a separate object and is not loaded on gfx1030.

Target models for the CPU expert tier: **Qwen3.5** (and successors) via `BF16`, `FP8`, and `GPTQ_INT4`; **DeepSeek-V4-Flash** via `MXFP4`. Phase 0 smoke stays **Qwen3-30B-A3B**.

## CPU expert paths

These are host C++. `CPUINFER_USE_ROCM=1` does not wrap them in a CUDA or HIP ifdef, and it does not port them to the GPU. They compile in a ROCm build under the same CPU flags as a CUDA or CPU-only build (`CPUINFER_CPU_INSTRUCT`, `CPUINFER_ENABLE_AVX512`, `CPUINFER_ENABLE_AMX`).

| Method | Model | AVX2 (every x86_64 build) | AVX-VNNI-256 (compiled on x86_64, used if the host has `avx_vnni`) | AVX512-BF16 class | AMX tiles | ROCm build |
|---|---|---|---|---|---|---|
| `BF16` | Qwen3.5 | `AVX2BF16_MOE` in `operators/avx2/bf16-moe.hpp`, bound in `ext_bindings.cpp` | none | `AMXBF16_MOE` when `USE_AMX_AVX_KERNEL` and `__AVX512F__`. Compute is `_mm512_dpbf16_ps` (`operators/amx/la/avx_kernels.hpp`) | not this class. `HAVE_AMX` enables `AMXInt4_MOE` / `AMXInt8_MOE`, a different layout | AVX2 object is in every x86_64 ROCm build. The AVX512 class is in the same build when the AVX512 umbrella is on |
| `FP8` | Qwen3.5 block FP8 | `AVX2FP8_MOE` in `operators/avx2/fp8-moe.hpp` | none | `AMXFP8_MOE` (`operators/amx/fp8-moe.hpp`), same AVX512 umbrella | no FP8 AMX-tile class | same as BF16 |
| `GPTQ_INT4` | Qwen3.5 GPTQ-Int4 | `AVX2GPTQInt4_MOE` in `operators/avx2/gptq_int4-moe.hpp` | `AVXVNNI256GPTQInt4_MOE`, `AVXVNNI256GPTQInt4Packed_MOE`. Selector: `python/utils/amx.py` `_select_gptq_int4_backend` | no separate class | none | sources are in the x86_64 ROCm binary. Runtime picks VNNI only if the CPU has it |
| `MXFP4` | DeepSeek-V4-Flash routed experts | `AVX2MXFP4_MOE` in `operators/avx2/mxfp4-moe.hpp` (E2M1 nibble, group-32, BF16 activations) | none | `AMXFP4_KGroup_MOE` in `operators/amx/fp4-moe.hpp` (`_mm512_dpbf16_ps` after PSHUFB). Selector: `_select_mxfp4_backend` | no AMX-tile MXFP4 | AVX2 always on x86_64 ROCm. AVX512 class when the umbrella is on |

Python entry points: `python/experts.py` `INFERENCE_METHODS`, dispatch in `python/utils/amx.py`. GGUF MXFP4 type id 39 is patched into llama.cpp by `third_party_patches/llama.cpp/0001-ggml-mxfp4-type.patch` for the `LLAMAFILE` loader; native safetensors MXFP4 uses `MXFP4SafeTensorLoader` in `python/utils/loader.py`.

The CI ROCm job forces `CPUINFER_CPU_INSTRUCT=AVX2` and `CPUINFER_ENABLE_AVX512=OFF`, so it compiles the AVX2 column and not the AVX512 column. An AVX512 or AMX host build is the same ROCm CMake switch with `CPUINFER_CPU_INSTRUCT=AVX512` or `FANCY` and `CPUINFER_ENABLE_AMX=ON`. That binary SIGILLs if it is later run on a CPU without the ISA it was compiled for.

`LLAMAFILE` (GGUF experts, `third_party/llamafile/`) stays the CPU GGUF path. It is not a Qwen3.5 or DSv4-Flash serving method for this port, and it is not hipified.

## kt-kernel GPU surface

| Piece | Where | gfx1030 | gfx1100 |
|---|---|---|---|
| `submit_with_cuda_stream` / `sync_with_cuda_stream` → `hipLaunchHostFunc` | `cpu_backend/cpuinfer.h`, `cpu_backend/vendors/hip.h` | Keep. `CPUINFER_USE_ROCM=1` already selects `hip.h`. hipBLAS is not included unless `KT_HIP_ENABLE_BLAS` is set. `__CUDA_ARCH__` is device-only | same |
| FP8 layerwise H2D (`cudaMemcpyAsync` / `hipMemcpyAsync`, copy stream, events) | `fp8_layerwise_transport.cpp` | **Take, done.** Host HIP runtime, same FP8 payload. Builds when `KTRANSFORMERS_USE_ROCM` is set. INT4/W4A16 payload is later | same |
| `gptq_marlin_gemm` | `cuda/gptq_marlin/`, `cuda/setup.py` | **Leave.** `CPUINFER_USE_ROCM=1` refuses to build `KTransformersOps`. The file stubs itself on `__HIP_PLATFORM_AMD__`. Replacement is the phase-1 W4A16 GEMM import, not Marlin | Leave. gfx1100 WMMA GEMM is a separate object, not Marlin |
| GGUF dequant `q2_k`…`q8_0`, `iq4_xs` | `cuda/custom_gguf/dequant.cu` | **Leave.** GPU GGUF dequant is not on the serving path. CPU `LLAMAFILE` keeps GGUF | Leave |
| `topk_softmax` | `cuda/moe/moe_topk_softmax_kernels.cu` | **Leave here.** Handled in sglang-kt-rdna (`sgl-kernel`) | same |
| Per-arch device libraries | `rocm/arch_probe.hip`, `rocm/wmma_gfx11.hip`, `rocm/CMakeLists.txt` | `libkt_rdna_gfx1030.so`, one `--offload-arch=gfx1030`, `v_dot2_f32_f16`, no WMMA symbol | `libkt_rdna_gfx1100.so` (no WMMA) and `libkt_rdna_wmma_gfx1100.so`. `rdna_load_wmma` refuses gfx1030 before `dlopen` |
| Drop-in kernels | not created by this change. Optional `kt-kernel/rdna/gfx1030/*.hip`, `kt-kernel/rdna/gfx1100/*.hip`, `kt-kernel/rdna/gfx1100/wmma/*.hip` | non-WMMA sources only | WMMA sources only under `gfx1100/wmma/` |

`cuda/binding.cpp` already hides Marlin and `topk_softmax` behind `KTRANSFORMERS_USE_CUDA`. The ROCm build does not define that macro.

## Handled in sglang-kt-rdna

Not built by this repository. Listed so the serving path is complete. Status there is owned by [BlivionIaG/sglang-kt-rdna](https://github.com/BlivionIaG/sglang-kt-rdna).

| Op | gfx1030 | gfx1100 |
|---|---|---|
| `sgl-kernel` `silu_and_mul` / `gelu_*`, `topk`, `topk_sigmoid`, `moe_align_block_size`, RoPE, KV transfer, grammar bitmask | Build for gfx1030. Drop the gfx942/gfx950 gate, FP8 defines off, dynamic LDS ≤ 48 KiB, wave32 audit | Build for gfx1100 |
| `custom_all_reduce` / `quick_all_reduce` | Leave. RCCL | Leave. RCCL |
| `rms_norm` / fused add (AITER or vLLM custom ops on the HIP path) | `forward_native` first, then a small HIP RMSNorm | same |
| FlashInfer attention, FlashMLA, `custom_flashinfer` | Leave. Import `fa_rdna2` (GQA, D=128/256). D=64 stays torch-native | Triton attention first. WMMA FA later, gfx1100-only object |
| Triton attention / fused MoE / MXFP4 Triton | Not a product path (Triton dropped RDNA2) | Fallback, including Triton fused MoE |
| Marlin MoE, DeepGEMM, `fp8_blockwise_scaled_mm` | Leave. INT4 experts: `moe_q_gemm_rdna2`. FP8 GEMM stays out (no FP8 unit) | Triton fused MoE or `moe_q_gemm` in the gfx1100 WMMA object. No FP8 GEMM |
| Dense cuBLAS / hipBLASLt / Marlin linear | `q_gemm_rdna2` and `gemv_f16_rdna2`. rocBLAS for fp16 prefill | rocBLAS, plus WMMA GEMM in the gfx1100 object |
| `gptq_marlin_repack` / `marlin_permute_scales` | Leave. RDNA pack is later | later |
| `moe_fused_gate` | torch grouped top-k | same |
| CUDA graph runner capturing kt host callbacks | Start with graphs off (`--disable-cuda-graph`). hipGraph + `hipLaunchHostFunc` is unverified | same |
| `cudaHostRegister` | Verify on ROCm torch, else `hipHostRegister` | same |
| AITER | Leave (`SGLANG_USE_AITER=0`) | Leave |

## Legacy standalone server (dead)

`archive/ktransformers`, `archive/csrc`, and `doc/en/ROCm.md` describe `local_chat.py`, `KLinearMarlin` / `KLinearTorch`, FlashInfer MLA, and the old Q8 linear. That path was validated only as a beta on gfx1100 (ROCm 6.2.4, 7900 XTX). Marlin was already stubbed. Two-GPU hipGraph instantiation segfaulted (upstream PR #178). This fork does not fix it.

| Legacy piece | Decision |
|---|---|
| `archive/csrc/.../cuda/gptq_marlin` | Leave |
| `archive/csrc/.../custom_gguf/dequant.cu` | Leave |
| FlashInfer / `third_party/custom_flashinfer` | Leave |
| balance_serve `kvc2` CUDA streams | Leave |
| `doc/en/ROCm.md` install steps | Historical. The banner points here |

## What phase 1 still has to add

Ordered list is in [PHASE0.md](PHASE0.md). The four GPU kernels (attention GQA, W4A16 GEMM, f16 GEMV, MoE quant GEMM) land as per-arch HIP sources, with any WMMA variant only under the gfx1100 WMMA library.
