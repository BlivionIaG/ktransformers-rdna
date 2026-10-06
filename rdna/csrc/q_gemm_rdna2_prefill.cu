// Ported into https://github.com/BlivionIaG/ktransformers-rdna
// from https://github.com/opengfx1030/vllm-rdna branch rdna_extras
// (csrc/rocm/q_gemm_rdna2_prefill.cu, commit c6d99ca30580f0ce6a97a168fa53bf9c6ee6b2af).
// Original license and copyright headers below are preserved.
// This module builds the packed-DOT (fdot2) bodies for gfx1030 and gfx1100.
// No gfx1100 WMMA translation unit is compiled or loaded.
//
// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// W4A16 GPTQ prefill kernel for AMD RDNA2 (gfx1030).
// The host launcher continues in q_gemm_rdna2_prefill_launch.inc so the
// preprocessed translation unit matches the upstream file.
//
// Design (single algorithm, multi-tile):
//   * 3-D grid: each block owns one M_TILE x N_TILE output tile over a K-split.
//   * Activations (A) are staged in LDS; weights (B, int4) are read directly
//     from global and dequantized on the fly.
//   * Inner dot uses __builtin_amdgcn_fdot2 (RDNA2's native fp16 dot).
//   * Epilogue uses a 64-bit packed CAS atomic-add on fp16.
//   * K_PER_SPLIT is templated so the LDS array is a true 2-D static shared
//     array (lets the backend emit ds_read_b128 for A).
//   * Dynamic fallback for odd K-per-split values.
//
// Tile configurations (compile-time `Config` template):
//   ConfigV1 (THREADS=512, N=2048, M=8, LDS=8)   — original v1 tile, wins for
//                                                   small M (M <= 64).
//   ConfigA  (THREADS=256, N=1024, M=16, LDS=0)  — general-purpose prefill
//                                                   tile, wins M >= 96 large N.
//   ConfigC  (THREADS=128, N= 512, M=16, LDS=0)  — small-N tile, wins
//                                                   N <= 1024.

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include <torch/all.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include "qdq_4_rdna2.cuh"

#include "q_gemm_rdna2_common.cuh"
#include "rdna2_graph_keepalive.cuh"

#if defined(__HIPCC__) && defined(__gfx1030__)
  #define __HIP__RDNA2__
#endif

namespace vllm {
namespace gptq_rdna2_prefill {

// The shared W4A16 helpers (refresh_group, epilogue, dot22_8_f, etc.) live
// in the sibling gptq_rdna2 namespace via q_gemm_rdna2_common.cuh. Pull
// them in so the call sites below can stay unqualified.
using namespace vllm::gptq_rdna2;

// ---------------------------------------------------------------------------
// Tile configuration template.
//
// All tile dimensions are compile-time constants.  N_PER_THREAD is fixed at 4
// so the pk4 fp16 atomic epilogue is shared across every config.  LDS_PAD=0
// for the wider-M configs keeps the static A tile within the 64 KiB local
// memory limit of gfx1030.
// ---------------------------------------------------------------------------
template <int Threads_, int NPerThread_, int KStep_, int MTile_, int LdsPad_>
struct Config {
  static constexpr int THREADS      = Threads_;
  static constexpr int N_PER_THREAD = NPerThread_;
  static constexpr int N_TILE       = THREADS * N_PER_THREAD;
  static constexpr int K_STEP       = KStep_;
  static constexpr int M_TILE       = MTile_;
  static constexpr int LDS_PAD      = LdsPad_;
};

// Configs exposed to the dispatcher.
using ConfigV1 = Config<512, 4, 32,  8,  8>;   // v1 tile (small M)
using ConfigA  = Config<256, 4, 32, 16,  0>;   // general prefill (large N)
using ConfigC  = Config<128, 4, 32, 16,  0>;   // small N (N_TILE=512)

#if defined(__HIP__RDNA2__) || !defined(__HIP_DEVICE_COMPILE__)

// ---------------------------------------------------------------------------
// Device-side helpers (dot22_8_f, atomic_add_pk4_f16, load4_zeros,
// load4_scales, refresh_group, epilogue) live in q_gemm_rdna2_common.cuh.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Static K-per-split variant: LDS is a 2-D static shared array with a
// constant row stride, so the backend can emit ds_read_b128 for A.
// ---------------------------------------------------------------------------
template <typename Config, int K_PER_SPLIT>
__global__ __launch_bounds__(Config::THREADS) void gemm_static_kernel(
    const half* __restrict__ a, const uint32_t* __restrict__ b_q_weight,
    const uint32_t* __restrict__ b_qzeros, const half* __restrict__ b_scales,
    half* __restrict__ c, const int size_m, const int size_n,
    const int size_k, const int groups, const int zero_offset,
    const int* __restrict__ b_q_perm, const int split_k) {
  constexpr int k_per_split = K_PER_SPLIT;
  constexpr int THREADS = Config::THREADS;
  constexpr int N_PER_THREAD = Config::N_PER_THREAD;
  constexpr int N_TILE = Config::N_TILE;
  constexpr int K_STEP = Config::K_STEP;
  constexpr int M_TILE = Config::M_TILE;
  constexpr int LDS_PAD = Config::LDS_PAD;

  const int t = threadIdx.x;
  const int n = blockIdx.x * N_TILE + t * N_PER_THREAD;
  const int m_tile = blockIdx.y * M_TILE;
  const bool active = (n < size_n);

  const int k_split = blockIdx.z;
  const int k_start = k_split * k_per_split;
  const int k_end = (k_split == split_k - 1) ? size_k : (k_start + k_per_split);

  const int groupsize = size_k / groups;
  int group = k_start / groupsize;
  int nextgroup = (group + 1) * groupsize;

  half2 z1z16_h[N_PER_THREAD][2];
  half2 y1y16_h[N_PER_THREAD][2];

  if (active) {
    refresh_group<Config::N_PER_THREAD>(group, n, b_qzeros, b_scales, size_n, zero_offset,
                          z1z16_h, y1y16_h);
  }

  float block_c[M_TILE][N_PER_THREAD];
  #pragma unroll
  for (int m = 0; m < M_TILE; ++m) {
    #pragma unroll
    for (int j = 0; j < N_PER_THREAD; ++j) block_c[m][j] = 0.0f;
  }

  __shared__ half block_a[M_TILE][k_per_split + LDS_PAD];
  if (b_q_perm) {
    #pragma unroll 1
    for (int idx = t; idx < M_TILE * k_per_split; idx += THREADS) {
      const int m = idx / k_per_split;
      const int kk = idx % k_per_split;
      const int m_row = m_tile + m;
      const int k = k_start + kk;
      if (m_row < size_m)
        block_a[m][kk] = (a + m_row * size_k)[b_q_perm[k]];
    }
  } else {
    #pragma unroll 1
    for (int idx = t; idx < (M_TILE * k_per_split) / 4; idx += THREADS) {
      const int quad = idx % (k_per_split / 4);
      const int m = idx / (k_per_split / 4);
      const int m_row = m_tile + m;
      const int k = k_start + quad * 4;
      if (m_row < size_m) {
        const float2 a4 = *(const float2*)(a + m_row * size_k + k);
        *(float2*)(&block_a[m][quad * 4]) = a4;
      }
    }
  }
  __syncthreads();

  const uint32_t* b_ptr = b_q_weight + (k_start / 8) * size_n + n;

  int k = k_start;
  if (active) {
    while (k < k_end) {
      int4 b_prefetch[K_STEP / 8];
      #pragma unroll
      for (int j = 0; j < K_STEP / 8; ++j) {
        b_prefetch[j] = *(const int4*)(b_ptr + j * size_n);
      }
      b_ptr += (K_STEP / 8) * size_n;

      #pragma unroll
      for (int j = 0; j < K_STEP / 8; ++j) {
        // Per-8-element group boundary check. The block-level K_STEP may
        // span multiple quantization groups (e.g. K_STEP=64 with
        // groupsize=32), so the refresh must fire mid-block, not just at
        // the block start. k and nextgroup are block-uniform so this
        // branch is warp-uniform on RDNA2.
        if (k + 8 * j == nextgroup) {
          group++;
          nextgroup += groupsize;
          refresh_group<Config::N_PER_THREAD>(group, n, b_qzeros, b_scales,
                                              size_n, zero_offset, z1z16_h,
                                              y1y16_h);
        }
        const int a_off = 8 * j;
        half2 dq[N_PER_THREAD][4];
        uint32_t w[N_PER_THREAD];
        w[0] = static_cast<uint32_t>(b_prefetch[j].x);
        w[1] = static_cast<uint32_t>(b_prefetch[j].y);
        w[2] = static_cast<uint32_t>(b_prefetch[j].z);
        w[3] = static_cast<uint32_t>(b_prefetch[j].w);
        #pragma unroll
        for (int col = 0; col < N_PER_THREAD; ++col) {
          vllm::gptq_rdna2::dequant_4bit_8_fp16(
              w[col], dq[col], z1z16_h[col], y1y16_h[col]);
        }

        #pragma unroll
        for (int m = 0; m < M_TILE; ++m) {
          const int m_row = m_tile + m;
          if (m_row >= size_m) continue;
          const half* a_window = &block_a[m][(k - k_start) + a_off];
          #pragma unroll
          for (int col = 0; col < N_PER_THREAD; ++col) {
            block_c[m][col] += dot22_8_f(dq[col], a_window);
          }
        }
      }

      k += K_STEP;
    }
    epilogue<Config::M_TILE>(block_c, m_tile, size_m, size_n, n, c);
  }
}

// ---------------------------------------------------------------------------
// Dynamic K-per-split fallback: LDS size and row stride are runtime values.
// Always works regardless of k_per_split, but pays a small address-compute
// cost vs the static variant.
// ---------------------------------------------------------------------------
template <typename Config>
__global__ __launch_bounds__(Config::THREADS) void gemm_dynamic_kernel(
    const half* __restrict__ a, const uint32_t* __restrict__ b_q_weight,
    const uint32_t* __restrict__ b_qzeros, const half* __restrict__ b_scales,
    half* __restrict__ c, const int size_m, const int size_n,
    const int size_k, const int groups, const int zero_offset,
    const int* __restrict__ b_q_perm, const int split_k) {
  constexpr int THREADS = Config::THREADS;
  constexpr int N_PER_THREAD = Config::N_PER_THREAD;
  constexpr int N_TILE = Config::N_TILE;
  constexpr int K_STEP = Config::K_STEP;
  constexpr int M_TILE = Config::M_TILE;
  constexpr int LDS_PAD = Config::LDS_PAD;

  const int t = threadIdx.x;
  const int n = blockIdx.x * N_TILE + t * N_PER_THREAD;
  const int m_tile = blockIdx.y * M_TILE;
  const bool active = (n < size_n);

  const int k_split = blockIdx.z;
  const int k_per_split = size_k / split_k;
  const int k_start = k_split * k_per_split;
  const int k_end = (k_split == split_k - 1) ? size_k : (k_start + k_per_split);
  // Round the LDS row stride up to a 16-byte (8 half) boundary so 128-bit
  // LDS loads are always aligned.
  const int row_stride = ((k_per_split + LDS_PAD + 7) / 8) * 8;

  const int groupsize = size_k / groups;
  int group = k_start / groupsize;
  int nextgroup = (group + 1) * groupsize;

  half2 z1z16_h[N_PER_THREAD][2];
  half2 y1y16_h[N_PER_THREAD][2];

  if (active) {
    refresh_group<Config::N_PER_THREAD>(group, n, b_qzeros, b_scales, size_n, zero_offset,
                          z1z16_h, y1y16_h);
  }

  float block_c[M_TILE][N_PER_THREAD];
  #pragma unroll
  for (int m = 0; m < M_TILE; ++m) {
    #pragma unroll
    for (int j = 0; j < N_PER_THREAD; ++j) block_c[m][j] = 0.0f;
  }

  extern __shared__ half block_a[];
  if (b_q_perm) {
    #pragma unroll 1
    for (int idx = t; idx < M_TILE * k_per_split; idx += THREADS) {
      const int m = idx / k_per_split;
      const int kk = idx % k_per_split;
      const int m_row = m_tile + m;
      const int k = k_start + kk;
      if (m_row < size_m)
        block_a[m * row_stride + kk] = (a + m_row * size_k)[b_q_perm[k]];
    }
  } else {
    if (k_per_split % 4 == 0) {
      const int half4_count = (M_TILE * k_per_split) / 4;
      #pragma unroll 1
      for (int idx = t; idx < half4_count; idx += THREADS) {
        const int quad = idx % (k_per_split / 4);
        const int m = idx / (k_per_split / 4);
        const int m_row = m_tile + m;
        const int k = k_start + quad * 4;
        if (m_row < size_m) {
          const float2 a4 = *(const float2*)(a + m_row * size_k + k);
          *(float2*)(block_a + m * row_stride + quad * 4) = a4;
        }
      }
    } else {
      const int half2_count = (M_TILE * k_per_split) / 2;
      #pragma unroll 1
      for (int idx = t; idx < half2_count; idx += THREADS) {
        const int pair = idx % (k_per_split / 2);
        const int m = idx / (k_per_split / 2);
        const int m_row = m_tile + m;
        const int k = k_start + pair * 2;
        if (m_row < size_m) {
          const half2 a2 = *(const half2*)(a + m_row * size_k + k);
          *(half2*)(block_a + m * row_stride + pair * 2) = a2;
        }
      }
    }
  }
  __syncthreads();

  const uint32_t* b_ptr = b_q_weight + (k_start / 8) * size_n + n;

  int k = k_start;
  if (active) {
    while (k < k_end) {
      int4 b_prefetch[K_STEP / 8];
      #pragma unroll
      for (int j = 0; j < K_STEP / 8; ++j) {
        b_prefetch[j] = *(const int4*)(b_ptr + j * size_n);
      }
      b_ptr += (K_STEP / 8) * size_n;

      #pragma unroll
      for (int j = 0; j < K_STEP / 8; ++j) {
        // Per-8-element group boundary check. The block-level K_STEP may
        // span multiple quantization groups (e.g. K_STEP=64 with
        // groupsize=32), so the refresh must fire mid-block, not just at
        // the block start. k and nextgroup are block-uniform so this
        // branch is warp-uniform on RDNA2.
        if (k + 8 * j == nextgroup) {
          group++;
          nextgroup += groupsize;
          refresh_group<Config::N_PER_THREAD>(group, n, b_qzeros, b_scales,
                                              size_n, zero_offset, z1z16_h,
                                              y1y16_h);
        }
        const int a_off = 8 * j;
        half2 dq[N_PER_THREAD][4];
        uint32_t w[N_PER_THREAD];
        w[0] = static_cast<uint32_t>(b_prefetch[j].x);
        w[1] = static_cast<uint32_t>(b_prefetch[j].y);
        w[2] = static_cast<uint32_t>(b_prefetch[j].z);
        w[3] = static_cast<uint32_t>(b_prefetch[j].w);
        #pragma unroll
        for (int col = 0; col < N_PER_THREAD; ++col) {
          vllm::gptq_rdna2::dequant_4bit_8_fp16(
              w[col], dq[col], z1z16_h[col], y1y16_h[col]);
        }

        #pragma unroll
        for (int m = 0; m < M_TILE; ++m) {
          const int m_row = m_tile + m;
          if (m_row >= size_m) continue;
          // Force a 128-bit LDS load; the dynamic row stride is always a
          // multiple of 8 halfs, so the address is 16-byte aligned.
          const int a_base = m * row_stride + (k - k_start) + a_off;
          float4 a8 = *(const float4*)(block_a + a_base);
          const half* a_ptr = reinterpret_cast<const half*>(&a8);
          #pragma unroll
          for (int col = 0; col < N_PER_THREAD; ++col) {
            block_c[m][col] += dot22_8_f(dq[col], a_ptr);
          }
        }
      }

      k += K_STEP;
    }
    epilogue<Config::M_TILE>(block_c, m_tile, size_m, size_n, n, c);
  }
}

#else  // non-RDNA2 device pass

// Stub kernels: same signatures so the launcher compiles on any gfx target.
// Unused at runtime (the dispatch path is gated by on_gfx10x() in Python).
template <typename Config, int K_PER_SPLIT>
__global__ __launch_bounds__(Config::THREADS) void gemm_static_kernel(
    const half*, const uint32_t*, const uint32_t*, const half*, half*, const int,
    const int, const int, const int, const int, const int*, const int) {}

template <typename Config>
__global__ __launch_bounds__(Config::THREADS) void gemm_dynamic_kernel(
    const half*, const uint32_t*, const uint32_t*, const half*, half*, const int,
    const int, const int, const int, const int, const int*, const int) {}

#endif  // __HIP__RDNA2__ || !__HIP_DEVICE_COMPILE__

#include "q_gemm_rdna2_prefill_launch.inc"
