# gfx1030 packed-DOT kernels

Native RDNA paths for the GPU side of serving: GQA attention, dense W4A16
linear, fp16 skinny GEMV, and GPTQ-Int4 GEMM for experts that stay on the
GPU. Cold routed experts stay on kt-kernel. SGLang
([BlivionIaG/sglang-kt-rdna](https://github.com/BlivionIaG/sglang-kt-rdna))
imports this package. The legacy standalone ktransformers server is not a
caller.

The HIP sources are the proven kernels from
[opengfx1030/vllm-rdna](https://github.com/opengfx1030/vllm-rdna) branch
`rdna_extras` (`csrc/rocm`, commit `c6d99ca`), with the upstream Apache-2.0
headers kept. `hippihx` stubs in that tree are not used. No gfx1100 WMMA
object is built. gfx1100 runs the same `fdot2` kernels. Wave64 parts
(CDNA) do not.

Import path: `ktransformers_rdna`, built from `rdna/`.

```bash
cd rdna
PYTORCH_ROCM_ARCH=gfx1030;gfx1100 python setup.py build_ext --inplace
```

`PYTORCH_ROCM_ARCH` lists both chips so one fatbin carries both device
images. Runtime dispatch still refuses any other gfx name.

## What SGLang calls

```python
import ktransformers_rdna as rdna

backend = rdna.select_attention_backend(
    arch=rdna.gcn_arch(),
    model_type="qwen3_moe",   # Qwen3-235B-A22B is GQA, head dim 128
    head_dim=128,
    num_heads=64,
    num_kv_heads=4,
)
# "rdna_fa" or "fallback"
if backend == "rdna_fa":
    out = rdna.paged_attention(
        q, key_cache, value_cache, block_table, seq_lens, cu_query_lens,
        num_heads=64, num_kv_heads=4, block_size=16,
    )
else:
    raise rdna.RdnaFallback(backend)  # or just keep FlashInfer / Triton

linear = rdna.select_linear_backend(
    arch=rdna.gcn_arch(),
    weight_format="gptq_int4",  # or "float16"
    act_dtype="float16",
    kind="dense",               # or "expert"
    m=x.shape[0],               # only consulted for fp16 GEMV
)
# "rdna_q_gemm" | "rdna_gemv" | "rdna_moe_q_gemm" | "fallback"
```

`fallback` means keep the caller's existing op (Marlin, FlashInfer, Triton,
or the ROCm beta path). Do not invent a result. MLA models
(`deepseek_v2` / `v3` / `v4`, or `attn_type="mla"`) always return
`fallback`. That is a phase 1 gap: `fa_rdna2` has no DeepSeek-style
latent attention.

`KT_USE_RDNA2_FA=0` forces the attention fallback. The decode kernel still
reads `VLLM_FA_RDNA2_GQA_DECODE=1` for its head-dim-256 GQA decode tile
(same switch as vllm-rdna). `VLLM_FA_RDNA2_GQA_MODE=subgroup` (the default)
selects the register-softmax prefill kernel.

## Attention — `fa_rdna2`

| | |
|---|---|
| Op | `ktransformers_rdna.paged_attention` |
| Coverage | fp16 GQA/MHA, head dim **128 or 256**, paged KV |
| Not covered | MLA, other head dims, bf16, fp8 KV, non-paged KV |
| MVP | Qwen3-235B-A22B-GPTQ-Int4: 64 query heads, 4 KV heads, head dim 128 |

Q: `[num_tokens, num_heads, head_dim]` fp16, contiguous.

K and V cache, both fp16:

```text
[num_blocks, num_kv_heads, head_dim // x, block_size, x]
```

`x` is the packing factor (8 in the vLLM cache). Element `d` of a token
lives at `[block, kv_head, d // x, slot, d % x]`.

`block_table`: `[num_seqs, max_blocks]` int32.

`seq_lens`: `[num_seqs]` int32, KV length including any prefix.

`cu_query_lens`: `[num_seqs + 1]` int32. Queries of sequence `s` are
`q[cu[s] : cu[s+1]]` and are the **last** positions of that KV sequence
(causal). A decode step is one query token per sequence.

`sliding_window=0` disables the window. `scale` defaults to `head_dim ** -0.5`.

Prefill picks a kernel the same way vllm-rdna does: GQA subgroup kernel,
else head-dim-128 with KV under 4096 (`prefill_paged_varlen_short`), else
split-K when the batch is small, else `prefill_paged_varlen`. Decode uses
split-K (`kv_splits` in 1..16, default 8).

## W4A16 GEMM — `q_gemm_rdna2`

| | |
|---|---|
| Op | `prepare_gptq_int4` then `apply_gptq_int4` |
| Format | GPTQ-Int4, uint4b8, fp16 activations |
| Shapes | `K % 32 == 0`, `N % 8 == 0`, `group_size >= 32` and divides `K` |
| Dispatch | M ≤ 16, or M ≤ 32 and K ≥ 4096: decode kernel. Otherwise prefill. |
| Not covered | AWQ as a separate zero convention is accepted on the dense kernel only when `use_v2_format=True` (literal zeros, no +1). Marlin, GGUF, FP8, MXFP4, bf16, channel-wise (group ≤ 0) return `fallback`. |

`apply_gptq_int4` returns fp16 `[..., N]`.

The C++ ops write a **persistent** buffer. The Python wrapper clones it
unless `graph.set_capturing(True)` is set, in which case the address must
stay stable for HIP graph replay. Do not hold an uncloned result across
another `apply_gptq_int4` call: the next call zeroes that buffer.

### Weight layout

Raw checkpoint nibbles are `[K, N]` integers in `0..15`, scales `[K/group, N]`
fp16. `prepare_gptq_int4` produces:

| Tensor | Shape | Notes |
|---|---|---|
| `qweight` | `[K/8, N]` int32 | 8 K-nibbles per word, then exllama `shuffle_4bit_8` |
| `scales` | `[groups, N]` fp16 | |
| `qzeros` | `[groups, N/8]` int32 | 8 N-nibbles per word, little-endian. Symmetric uint4b8 stores **7** |
| `g_idx` | `[K]` int32 or empty | empty = identity. Otherwise `argsort` of the checkpoint `g_idx` |
| `use_v2_format` | bool | `False` for GPTQv1. The kernel adds 1 to the stored zero, so the effective symmetric zero is 8 |

Shuffle of one packed word (nibbles `n0..n7`, low nibble = `n0`):

```text
result nibbles, low to high: n0, n2, n4, n6, n1, n3, n5, n7
```

That is the exllamav2 pattern `77775555 33331111 66664444 22220000`. The
dequant is the usual bit trick: `(q - (stored_zero + 1)) * scale` for
GPTQv1, in fp16, accumulated with `v_dot2_f32_f16`.

Qwen3-235B-A22B-GPTQ-Int4 dense shapes that pass the checks: hidden 4096,
group 128, attention output width a multiple of 8.

## fp16 GEMV — `gemv_f16_rdna2`

| | |
|---|---|
| Op | `ktransformers_rdna.gemv_f16(x, w, bias=None)` |
| Math | `y[M, N] = x[M, K] @ w[N, K].T (+ bias[N])` |
| Coverage | fp16, M in **1..8**, `K % 8 == 0` |
| Not covered | M > 8 (returns `fallback`), quantized weights, bf16 |

`w` is a plain fp16 matrix, not a GPTQ pack. This is the skinny decode
path for projections that are not W4A16.

## Quantized MoE — `moe_q_gemm_rdna2`

| | |
|---|---|
| Op | `apply_gptq_int4_experts` |
| Format | GPTQ-Int4 only. The kernel hard-codes zero offset +1 (uint4b8). AWQ experts return `fallback` |
| Tile | `block_size_m` ∈ {1, 2, 4, 8}. Default 1 when M ≤ 4, else 4 |
| Activation | fp16, SwiGLU (`silu(gate) * up`) |
| Not covered | expert counts that do not fit the resident table, block sizes above 8, bf16, FP8, MXFP4, W4A8 |

Hot experts are the ones SGLang keeps for `--kt-num-gpu-experts`. Pass
only that resident table. Remap global ids into `0 .. E-1`. Ids that are
not resident are `-1` and are skipped; kt-kernel still owns those cold
experts, and the caller adds the two partial sums.

```python
w13 = rdna.moe.stack_experts([rdna.moe.prepare_expert(q, s) for q, s in gate_up])
w2 = rdna.moe.stack_experts([rdna.moe.prepare_expert(q, s) for q, s in down])
y = rdna.moe.apply_gptq_int4_experts(hidden, w13, w2, local_ids, topk_weights)
```

Per-expert raw nibbles:

| Role | `q` | `scales` |
|---|---|---|
| gate+up (`w13`) | `[K, 2I]` | `[K/group, 2I]` |
| down (`w2`) | `[I, K]` | `[I/group, K]` |

After `prepare_expert` / `stack_experts`:

| Tensor | Shape |
|---|---|
| `w13.qweight` | `[E, K/8, 2I]` int32 shuffled |
| `w13.scales` | `[E, K/group, 2I]` fp16 |
| `w13.qzeros` | `[E, K/group, (2I)/8]` int32 |
| `w2.qweight` | `[E, I/8, K]` int32 shuffled |
| `w2.scales` | `[E, I/group, K]` fp16 |
| `w2.qzeros` | `[E, I/group, K/8]` int32 |

`topk_ids` is `[M, topk]` int32. `align_blocks` groups them into
`sorted_token_ids` (flattened `m * topk + slot`), `expert_ids` per block,
and `num_tokens_post_padded`. Padding ids are `M * topk`. The down GEMM
reuses those ids with `top_k=1` so each id is a row of the expanded
SwiGLU activation, and `output_topk=topk` atomically reduces expert slots
back to `[M, K]`. `fp32_accum=False` (upstream default) uses packed fp16
atomics. `fp32_accum=True` accumulates in a cached fp32 scratch and casts
once; the hardware tests use that.

## HIP graph capture

From `rdna2_graph_keepalive` in vllm-rdna. gfx1030's
`hipStreamIsCapturing` is true during eager mixed prefill and during
replay, so the kernels do not trust it.

```python
rdna.set_capturing(True)          # immediately before FULL graph capture
# ... capture the forward; q_gemm returns the persist alias, not a clone
rdna.freeze_capture_persist()     # after a successful FULL capture
```

`freeze_capture_persist` clears the capturing flag and freezes the capture
slots. Later eager work grows a different buffer, so a long prefill cannot
recycle a device pointer baked into the graph. Persist allocations are
never returned to the caching allocator.

`set_capturing(False)` is the abort path.

## Not ported

Kept on the existing fallback, or out of scope for this package:

- MLA / DeepSeek latent attention (phase 1 gap)
- Marlin (`gptq_marlin_gemm`) and any weight format other than GPTQ-Int4 uint4b8 for experts, plus fp16 GEMV for skinny dense decode
- FlashInfer and Triton attention outside GQA head dim 128/256
- GGUF dequant, `topk_softmax` (the caller still does router softmax)
- FP8, MXFP4, EXL3, W4A8, AWQ MoE, bf16
- gfx1100 WMMA (`q_gemm_rdna3_wmma` and `moe_q_gemm_rdna3` are not in this tree)
- MoE `block_size_m` above 8
- A full graph runtime. Only the capture/freeze rules and the persist buffers inside the kernels

## Tests on hardware

There is no AMD GPU in the build VM. `tests/test_kernels.py` skips there.
CPU tests cover dispatch, the exllama nibble shuffle, and the reference
itself:

```bash
cd rdna
PYTHONPATH=. python -m pytest tests/test_dispatch.py tests/test_reference.py -q
```

On a gfx1030 or gfx1100 machine, after the build above:

```bash
cd rdna
PYTHONPATH=. python -m pytest tests -q
```

`tests/test_kernels.py` compares `gemv_f16`, `q_gemm` (M=1, 4, 17),
GPTQ-Int4 SwiGLU experts, and paged attention (head dim 128 and 256,
causal and non-causal) to `ktransformers_rdna.reference`.

Compile check, no GPU, both offload archs (needs ROCm `hipcc`, a ROCm
PyTorch, and Python development headers such as `python3-dev`):

```bash
ROCM_PATH=/opt/rocm-6.4.1 bash rdna/scripts/compile_check.sh
```

The script compiles each translation unit twice, `--offload-arch=gfx1030`
and `--offload-arch=gfx1100`. `dot_arch_probe.cu` errors out if a device
pass did not enable the `fdot2` bodies.

ROCm 6.4 no longer installs `cuda_runtime_api.h`. PyTorch's `ATen/cuda`
headers still include that name, plus `cublas_v2.h`, `cublasLt.h`,
`cusparse.h`, and `cusolverDn.h`. When those files are absent,
`compile_check.sh` and `setup.py` add `rdna/csrc/rocm_cuda_compat/`, which
forwards the few types those headers name (`cudaStream_t`, the BLAS
handles) onto HIP. The same flags define
`C10_CUDA_NO_CMAKE_CONFIGURE_FILE`, which is how this ROCm PyTorch skips
the generated `c10/cuda/impl/cuda_cmake_macros.h` it does not ship, and
define `TORCH_CUDA_CPP_API` when the wheel left that macro undefined.
On Ubuntu, the script also adds the multiarch libstdc++ directories ROCm
clang does not search on its own, and a `-I` directory that points at
ROCm's `hip/` headers so an older `/usr/include/hip` does not win.
`hipcc` records `$ROCM_PATH/include` as `-idirafter`, after `/usr/include`.
A ROCm install that still ships `cuda_runtime_api.h` keeps that header.

The scalar fp16 tail atomic uses `__hip_atomic_compare_exchange_strong`
on `unsigned short`. ROCm 6.4's `atomicCAS` has no 16-bit overload. The
packed 64-bit tail still uses `atomicCAS`.
