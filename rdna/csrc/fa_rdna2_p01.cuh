// Ported into https://github.com/BlivionIaG/ktransformers-rdna
// from https://github.com/opengfx1030/vllm-rdna branch rdna_extras
// (csrc/rocm/fa_rdna2.cu, commit c6d99ca30580f0ce6a97a168fa53bf9c6ee6b2af).
// Original license and copyright headers below are preserved.
// This module builds the packed-DOT (fdot2) bodies for gfx1030 and gfx1100.
// No gfx1100 WMMA translation unit is compiled or loaded.
//
// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Flash Attention v2 kernels for RDNA2 (gfx1030)
//
// Two kernels are provided:
//   - Decode: Br=1 per CTA, split-K across multiple CTAs per head.
//   - Prefill: Br=16 per CTA, no split-K (one CTA per (b, h_q, q_block)).
//
// Decode layout: Q[B, H_q, D], K[N, H_kv, D], V[N, H_kv, D], O[B, H_q, D].
// Prefill layout: Q[B, H_q, N_q, D], K[N, H_kv, D], V[N, H_kv, D], O[B, H_q, N_q, D].
//
// GQA: H_q may be a multiple of H_kv (kv_group_num = H_q / H_kv).
// Each query head H_q[h] attends to H_kv[h / kv_group_num].
//
// Decode algorithm (FA2 split-K, Br=1):
//   Stage 1 (per CTA, per (batch, head, kv_split)):
//     Initialize m_i = -inf, l_i = 0, O_i = 0
//     For each K/V block in this split:
//       Load K_j[BC][D], V_j[BC][D] into shared memory
//       S[BC] = Q . K_j^T * scale
//       m_new = max(m_i, max(S))
//       P[BC] = exp(S - m_new)   (write to shared memory)
//       l_new = exp(m_i - m_new) * l_i + sum(P)
//       O_i = exp(m_i - m_new) * O_i + P . V_j
//       m_i = m_new, l_i = l_new
//     Write (O_i, m_i, l_i) to global
//
//   Stage 2 (per (batch, head)):
//     Load all splits' (O_k, m_k, l_k)
//     m_global = max(m_k)
//     O_final = sum(exp(m_k - m_global) * O_k) / sum(exp(m_k - m_global) * l_k)
//
// Prefill algorithm (FA2, Br=16, no split-K):
//   Per CTA (per (batch, head, q_block)):
//     Load Q[Br x D] into shared memory
//     Initialize m[Br] = -inf, l[Br] = 0, O[Br x D] = 0
//     For each K/V block:
//       Load K[BC x D], V[BC x D] into shared memory
//       S[Br x BC] = Q . K^T * scale
//       For each row br:
//         m_new[br] = max(m[br], max_k S[br, k])
//         P[br, k] = exp(S[br, k] - m_new[br])
//         l_new[br] = exp(m[br] - m_new[br]) * l[br] + sum_k P[br, k]
//         O[br, :] = exp(m[br] - m_new[br]) * O[br, :] + sum_k P[br, k] * V[k, :]
//         m[br] = m_new[br], l[br] = l_new[br]
//     O[br, :] /= l[br]
//
// Tile sizes for gfx1030 (V620, 72 CUs, 4MB L2):
//   Decode:  Br = 1,  Bc = 64, head_dim = 128, THREADS = 128 (~33 KB smem)
//   Prefill: Br = 16, Bc = 64, head_dim = 128, THREADS = 128 (~48 KB smem)
//
// Each thread owns one O element (D=128, THREADS=128) and one P[k] element
// for the online softmax. P[k] is written to shared memory so all threads
// can use it for the O update (PV dot product).
//
// V_DOT2_F32_F16 intrinsic: 2 fp16 multiply-adds per instruction.
// fp32 accumulators for m, l, o_acc (numerical stability).

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <torch/all.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include "rdna2_graph_keepalive.cuh"

// ---- Tile constants for gfx1030 -----------------------------------------
// Defined at global scope (not inside namespace vllm::fa_rdna2) because
// __global__ kernel functions in HIP do not reliably resolve namespace-
// scoped constexpr values at template instantiation time. Keeping these
// at global scope lets every kernel reference them unqualified.
constexpr int BC = 64;
constexpr int BC_256 = 32;
constexpr int MAX_SPLITS = 16;
constexpr int BR_PREFILL = 16;
constexpr int THREADS_PREFILL = 128;
constexpr int HEAD_DIM_PAGED_128 = 128;
__device__ __forceinline__ float fdot2(half2 q, half2 k, float acc) {
  return __builtin_amdgcn_fdot2(q, k, acc, false);
}

__device__ __forceinline__ float warp_reduce_max(float v) {
  #pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    v = fmaxf(v, __shfl_xor(v, offset));
  }
  return v;
}

__device__ __forceinline__ float warp_reduce_sum(float v) {
  #pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    v += __shfl_xor(v, offset);
  }
  return v;
}

// Block reductions across 4 warps (THREADS=128). shared[4] is scratch.
// Result is broadcast to all threads in the block via shared memory.
// Caller must size shared as 5 floats: shared[0..3] for warp partials,
// shared[4] for the broadcast result. The caller is responsible for
// not clobbering shared[4] between the reduction and the broadcast read.
__device__ __forceinline__ float block_reduce_max(float v, float* shared) {
  const int lane = threadIdx.x & 31;
  const int wid = threadIdx.x >> 5;
  v = warp_reduce_max(v);
  if (lane == 0) shared[wid] = v;
  __syncthreads();
  v = (threadIdx.x < 4) ? shared[lane] : -INFINITY;
  if (wid == 0) v = warp_reduce_max(v);
  if (wid == 0 && lane == 0) shared[4] = v;
  __syncthreads();
  return shared[4];
}

__device__ __forceinline__ float block_reduce_sum(float v, float* shared) {
  const int lane = threadIdx.x & 31;
  const int wid = threadIdx.x >> 5;
  v = warp_reduce_sum(v);
  if (lane == 0) shared[wid] = v;
  __syncthreads();
  v = (threadIdx.x < 4) ? shared[lane] : 0.0f;
  if (wid == 0) v = warp_reduce_sum(v);
  if (wid == 0 && lane == 0) shared[4] = v;
  __syncthreads();
  return shared[4];
}

// XOR swizzle for sK/sV shared-memory indexing to reduce LDS bank conflicts.
// Pattern: d_swizzled = d ^ ((k & 7) << 4). For HEAD_DIM=128 / RDNA2 64-bank
// LDS, this shifts access patterns so 32-thread lanes reading different rows
// (k) but the same column (d) hit distinct banks rather than colliding on one.
// Smem cost: zero (same storage layout, remapped indexing on read & write).
__device__ __forceinline__ int fa_swz_d(int d, int k) {
  return d ^ ((k & 7) << 4);
}

// vLLM sliding-window semantics (FlashAttention window (w - 1, 0)): query
// position q attends key position k iff k <= q and q - k < w.
__device__ __forceinline__ bool fa_masked(int q, int k, int causal,
                                          int sliding_window) {
  return (causal && k > q) || (sliding_window > 0 && q - k >= sliding_window);
}

// Clips the K/V tile walk [lo, hi) (tiles at lo + i * tile) of a CTA whose
// query rows sit at positions [q_first, q_first + rows) to the tiles that are
// not masked for every row: past the causal diagonal of the last row, or left
// of the sliding window of the first row. Kept tiles keep their boundaries,
// so the result is bit-identical to walking and masking every tile.
__device__ __forceinline__ void fa_clip_kv_walk(int& lo, int& hi, int q_first,
                                                int rows, int causal,
                                                int sliding_window, int tile) {
  if (causal) hi = min(hi, q_first + rows);
  if (sliding_window > 0) {
    const int first = q_first - sliding_window + 1;
    if (first > lo) lo += (first - lo) / tile * tile;
  }
}

// Sequence of decode query token `tok` and the number of keys it attends to.
// Without cu_query_lens every query token is its own sequence. With it, a
// sequence's queries are its last q_len positions (spec-decode verify or a
// short extend), so token i of a q_len-token query sees seq_len - (q_len -
// 1 - i) keys: the causal mask becomes a per-token KV length. Tokens past
// the last query (graph padding) see none.
__device__ __forceinline__ int fa_decode_token_seq(const int* cu_query_lens,
                                                   int num_seqs, int tok,
                                                   const int* seq_lens,
                                                   int& kv_len) {
  if (cu_query_lens == nullptr) {
    kv_len = seq_lens[tok];
    return tok;
  }
  // Last sequence starting at or before tok (zero-length ones are skipped).
  int lo = 0, hi = num_seqs;
  while (hi - lo > 1) {
    const int mid = (lo + hi) >> 1;
    if (cu_query_lens[mid] <= tok) {
      lo = mid;
    } else {
      hi = mid;
    }
  }
  const int end = cu_query_lens[lo + 1];
  kv_len = tok < end ? seq_lens[lo] - (end - 1 - tok) : 0;
  return lo;
}

// fp8 e4m3fn -> fp16 conversion (software, no fp8 hardware needed on gfx1030)
// e4m3: [S(1) | E(4) | M(3)], bias=7
// fp16:  [S(1) | E(5) | M(10)], bias=15
// Normal values: fp16_exp = e4m3_exp + 8, fp16_mant = e4m3_mant << 7
__device__ __forceinline__ half fp8_e4m3_to_half(uint8_t val) {
    uint32_t s = (val >> 7) & 1;
    uint32_t e = (val >> 3) & 0xF;
    uint32_t m = val & 0x7;
    union { uint16_t bits; half h; } u;
    if (e == 0 && m == 0) {
        u.bits = s << 15;  // +/-0
    } else if (e == 0) {
        // Subnormal: 2^(1-7) * m/8 = m * 2^(-9)
        float v = (float)m * 0.001953125f;
        u.h = __float2half_rn(s ? -v : v);
    } else {
        // Normal (exp 1..15, all finite in e4m3fn): direct bit conversion
        u.bits = (uint16_t)((s << 15) | ((e + 8) << 10) | (m << 7));
    }
    return u.h;
}

// Per-element KV load from the paged cache with optional inline fp8 dequant.
// For IS_FP8 the cache element is a raw uint8 e4m3 byte; it is converted to
// an fp16 half and scaled by the per-tensor kv scale inside the kernel --
// the shared-memory tile stays fp16 so the fdot2 (V_DOT2_F32_F16) QK dot
// product and the fp32 PV accumulation are unchanged.
template <typename KV_T, bool IS_FP8>
__device__ __forceinline__ half fa_kv_load(const KV_T* ptr, float scale) {
    if constexpr (IS_FP8) {
        return __hmul(fp8_e4m3_to_half(static_cast<uint8_t>(*ptr)),
                      __float2half(scale));
    } else {
        return *ptr;
    }
}


// =====================================================================
// DECODE STAGE 1 (PAGED): per-CTA partials, K/V read from paged blocks
// =====================================================================
//
// Reads K/V from vLLM's paged cache ([num_blocks, H_kv, D/x, block_size, x]).
// Each query token is a separate sequence in the batch with its own
// block_table and seq_len.
//
// Grid: (num_tokens, H_q, kv_splits)
//   blockIdx.x = token_idx (== seq_idx for decode — one query per seq)
//   blockIdx.y = h_q
//   blockIdx.z = split
//
// Strides for the 5D paged cache (element units):
//   stride_kc0/vc0 = one block
//   stride_kc1/vc1 = one head within a block
//   stride_kc2/vc2 = one D/x sub-dim
//   stride_kc3/vc3 = one slot within a block
//   stride_kc4/vc4 = one x-element
// K and V have SEPARATE stride sets because reshape_and_cache writes K
// packed ([.., D/x, bs, x], x-innermost) but V unpacked ([.., D, bs],
// slot-innermost). The Python side re-views V via
// view(nb, h, D/x, x, bs).permute(0, 1, 2, 4, 3) so its strides describe
// the real physical layout (vc3 = 1, vc4 = bs).
//   x_dim = packing factor (typically 8 for fp16)
//
// HEAD_DIM = 128 specialization (the original kernel). A HEAD_DIM = 256
// variant `fa_decode_paged_splitk_kernel_256` follows below for Qwen3.5.

// HEAD_DIM = 128 specialization of the paged decode kernel.
//
// Template parameters:
//   KV_T        : storage dtype (half, uint8_t, int8_t)
//   IS_FP8      : KV_T=uint8_t fp8 (e4m3) — per-tensor scalar scale
//   IS_INT8     : KV_T=int8_t signed int8 — per-(token, head) scale tensor
//
// For IS_INT8 the per-(token, head) scale tensor is `k_scale_per_tok` /
// `v_scale_per_tok` shaped [N_kv_total, H_kv]. We look up the scale inline
// at the KV load point — no smem scale table, no full-seq fp16 workspace.
// The smem tile layout is identical to the FP8 path (per-block tile of
// dequantized fp16 K/V) so the same online-softmax + fdot2 hot loop is
// reused without code duplication. Per the kv-int8.md wiki contract, this
// is the structural fix for OPT-D: replace the scalar-__hmul _pth stub with
// the occupancy-fixed fp16 decode tile shape, fused i8 KV load via fdot2.
template <typename KV_T, bool IS_FP8, bool IS_INT8 = false>
__global__ __launch_bounds__(128)
    __attribute__((amdgpu_waves_per_eu(4, 8))) void fa_decode_paged_splitk_kernel(
    const half* __restrict__ Q,
    const KV_T* __restrict__ key_cache,
    const KV_T* __restrict__ value_cache,
    const int* __restrict__ block_table,
    const int* __restrict__ seq_lens,
    const int stride_kc0,
    const int stride_kc1,
    const int stride_kc2,
    const int stride_kc3,
    const int stride_kc4,
    const int stride_vc0,
    const int stride_vc1,
    const int stride_vc2,
    const int stride_vc3,
    const int stride_vc4,
    const int max_blocks,
    const int block_size,
    const int x_dim,
    float* __restrict__ O_partial,
    float* __restrict__ M_partial,
    float* __restrict__ L_partial,
    const int num_tokens,
    const int H_q,
    const int H_kv,
    const int kv_splits,
    const int kv_group_num,
    const float scale,
    const int sliding_window,
    const float k_scale,
    const float v_scale,
    const float* __restrict__ k_scale_per_tok,
    const float* __restrict__ v_scale_per_tok,
    const int* __restrict__ cu_query_lens,
    const int num_seqs) {

  const int token_idx = blockIdx.x;
  const int h_q = blockIdx.y;
  const int split = blockIdx.z;
  const int t = threadIdx.x;
  const int h_kv = h_q / kv_group_num;
  if (token_idx >= num_tokens || h_q >= H_q || split >= kv_splits) return;

  // Per-query sequence length and block table base.
  int seq_len;
  const int seq = fa_decode_token_seq(cu_query_lens, num_seqs, token_idx,
                                      seq_lens, seq_len);
  const int* my_block_table = block_table + seq * max_blocks;
  if (seq_len <= 0) {
    // Empty sequence: write zeros and skip.
    if (t < HEAD_DIM_PAGED_128) {
      O_partial[((token_idx * H_q + h_q) * kv_splits + split) * HEAD_DIM_PAGED_128 + t] = 0.0f;
    }
    if (t == 0) {
      M_partial[(token_idx * H_q + h_q) * kv_splits + split] = -INFINITY;
      L_partial[(token_idx * H_q + h_q) * kv_splits + split] = 0.0f;
    }
    return;
  }

  // sK/sV rows padded so the vectorized K store (one 16B write per lane
  // across consecutive n_local) does not collapse onto one smem bank quad.
  constexpr int DSK = HEAD_DIM_PAGED_128 + 8;
  extern __shared__ unsigned char smem_raw[];
  half*  sQ   = reinterpret_cast<half*>(smem_raw);
  half*  sK   = sQ + HEAD_DIM_PAGED_128;
  half*  sV   = sK + BC * DSK;
  float* sP   = reinterpret_cast<float*>(sV + BC * DSK);
  float* sRed = sP + BC;
  __shared__ int s_blk[BC];
  __shared__ int s_slot[BC];

  // Load Q for this query token.
  sQ[t] = Q[(token_idx * H_q + h_q) * HEAD_DIM_PAGED_128 + t];
  __syncthreads();

  float m_i = -INFINITY;
  float l_i = 0.0f;
  float o_acc = 0.0f;

  // Split this sequence's KV range across kv_splits CTAs. Keys left of the
  // sliding window are never visited, and every split is a whole number of
  // tiles so each one (not only split 0) starts 8-slot aligned and takes the
  // vectorized V load.
  const int kv_lo = sliding_window > 0
                        ? max(0, seq_len - sliding_window) / BC * BC : 0;
  const int tokens_per_split =
      ((seq_len - kv_lo + kv_splits - 1) / kv_splits + BC - 1) / BC * BC;
  const int blk_start = kv_lo + split * tokens_per_split;
  const int blk_end   = min(blk_start + tokens_per_split, seq_len);

  for (int n = blk_start; n < blk_end; n += BC) {
    const int blk_size = min(BC, blk_end - n);

    // Page mapping once per KV block: kills per-element div/mod and
    // block_table re-reads.
    if (t < BC) {
      const int n_global = n + t;
      const bool ok = (t < blk_size);
      s_blk[t] = ok ? my_block_table[n_global / block_size] : 0;
      s_slot[t] = ok ? (n_global % block_size) : 0;
    }
    __syncthreads();

    const bool kv_vec_ok =
        (sizeof(KV_T) == 2) && (!IS_FP8) && (!IS_INT8)
        && stride_kc4 == 1 && stride_vc3 == 1
        && x_dim == 8 && ((block_size & 7) == 0);
    if (kv_vec_ok) {
      // K is x-packed: 8 consecutive d contiguous per (slot, d_sub).
      constexpr int NX = HEAD_DIM_PAGED_128 / 8;
      for (int i = t; i < BC * NX; i += 128) {
        const int n_local = i % BC;
