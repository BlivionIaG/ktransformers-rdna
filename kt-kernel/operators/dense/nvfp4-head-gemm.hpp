// Dense NVFP4 GEMM for the LM head.
//
// kt-kernel ships no dense 4-bit GEMM -- every 4-bit kernel in 0.7.1 is MoE-shaped
// -- so a checkpoint that quantizes its `lm_head` (nvidia's Qwen3.6 NVFP4 release
// does, declaring it W4A16_NVFP4 group_size 16) loads but cannot be executed: the
// logits projection is a plain (m, K) @ (K, N) matmul with no expert dimension.
//
// CONTRACT (pinned from the stored tensors, checked against a numpy decode)
//   weight         U8   (N, K/2)   two E2M1 nibbles per byte, LOW nibble first
//   weight_scale   F32  (N, K/16)  one per group of 16, converted from E4M3
//   weight_scale_2 F32  scalar     per-tensor global
//   W[n,k] = e2m1(nibble) * weight_scale[n, k/16] * weight_scale_2
//
// E2M1 is signed: nibbles 8..15 mirror 0..7 negated.
//
// `detail::dequant_row_scalar` is the definition of record and stays compiled
// everywhere -- fallback for non-x86, oracle for the vector paths.
//
// PERF: dequantising per output row with a scalar loop measured ~45x slower than
// a plain float32 `mv` at this shape (3468 ms vs 77 ms), because time tracked rows
// rather than flops. Two fixes: a 16-entry shuffle table for nibble->float, and
// dequantising a tile of rows once and reusing it across m.

#pragma once

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <vector>

#if defined(__AVX2__)
#include <immintrin.h>
#endif
#if defined(__AVX512F__)
#include <immintrin.h>
#endif

namespace kt {
namespace nvfp4_head {

struct DenseNVFP4HeadConfig {
  int m = 1;
  int n = 0;
  int k = 0;
  int block_n = 4096;
  int block_m = 4;
  const uint8_t* weight = nullptr;
  const float* weight_scale = nullptr;
  float weight_scale_2 = 1.0f;
};

namespace detail {

// E2M1 codebook, index = nibble; 8..15 are the negations of 0..7.
alignas(64) static const float kCodebook[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
};

inline float fp4_to_float(uint8_t nib) { return kCodebook[nib & 0x0F]; }

inline void dequant_row_scalar(const uint8_t* packed, const float* scale, int k,
                               float scale2, float* dst) {
  for (int kk = 0; kk < k; ++kk) {
    const uint8_t byte = packed[kk >> 1];
    const uint8_t nib =
        (kk & 1) ? static_cast<uint8_t>(byte >> 4) : static_cast<uint8_t>(byte & 0x0F);
    dst[kk] = fp4_to_float(nib) * scale[kk >> 4] * scale2;
  }
}

#if defined(__AVX2__)
// `_mm256_shuffle_epi8` indexes 16 entries within each 128-bit lane, so the
// 16-entry table is replicated into both lanes and a raw nibble addresses it.
// That is the only reason this path is branch-free.
constexpr uint8_t kNibbleLo[32] = {
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};

inline void dequant_row_avx2(const uint8_t* packed, const float* scale, int k,
                             float scale2, float* dst) {
  int kk = 0;
  const __m256i lut = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(kNibbleLo));
  const __m256i low_mask = _mm256_set1_epi8(0x0F);
  const __m256 vscale2 = _mm256_set1_ps(scale2);

  // 16 lanes per iteration: 8 packed bytes produce 16 nibbles, and those 16
  // nibbles share exactly one group scale, so the multiply is one broadcast.
  for (; kk + 16 <= k; kk += 16) {
    const __m128i bytes =
        _mm_loadl_epi64(reinterpret_cast<const __m128i*>(packed + (kk >> 1)));
    const __m256i wide = _mm256_cvtepu8_epi16(bytes);
    const __m256i lo = _mm256_and_si256(wide, low_mask);
    const __m256i hi = _mm256_and_si256(_mm256_srli_epi16(wide, 4), low_mask);
    // Interleave lo/hi nibbles so element order matches the packed byte order.
    const __m256i inter = _mm256_unpacklo_epi8(lo, hi);
    const __m256i idx = _mm256_shuffle_epi8(lut, inter);

    const __m256 g = _mm256_mul_ps(_mm256_set1_ps(scale[kk >> 4]), vscale2);
    float tmp[16];
    _mm256_storeu_ps(tmp, _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(
                              _mm256_castsi256_si128(idx))));
    _mm256_storeu_ps(tmp + 8, _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(
                                     _mm256_extracti128_si256(idx, 1))));
    _mm256_storeu_ps(dst + kk, _mm256_mul_ps(_mm256_loadu_ps(tmp), g));
    _mm256_storeu_ps(dst + kk + 8, _mm256_mul_ps(_mm256_loadu_ps(tmp + 8), g));
  }
  for (; kk < k; ++kk) {
    const uint8_t byte = packed[kk >> 1];
    const uint8_t nib =
        (kk & 1) ? static_cast<uint8_t>(byte >> 4) : static_cast<uint8_t>(byte & 0x0F);
    dst[kk] = fp4_to_float(nib) * scale[kk >> 4] * scale2;
  }
}
#endif  // __AVX2__

#if defined(__AVX512F__)
inline void dequant_row_avx512(const uint8_t* packed, const float* scale, int k,
                               float scale2, float* dst) {
  const __m512 vscale2 = _mm512_set1_ps(scale2);
  int kk = 0;
  for (; kk + 16 <= k; kk += 16) {
    const __m512 g = _mm512_mul_ps(_mm512_set1_ps(scale[kk >> 4]), vscale2);
    alignas(64) float tmp[16];
    for (int t = 0; t < 16; ++t) {
      const int kx = kk + t;
      const uint8_t byte = packed[kx >> 1];
      const uint8_t nib =
          (kx & 1) ? static_cast<uint8_t>(byte >> 4) : static_cast<uint8_t>(byte & 0x0F);
      tmp[t] = fp4_to_float(nib);
    }
    _mm512_storeu_ps(dst + kk, _mm512_mul_ps(_mm512_load_ps(tmp), g));
  }
  for (; kk < k; ++kk) {
    const uint8_t byte = packed[kk >> 1];
    const uint8_t nib =
        (kk & 1) ? static_cast<uint8_t>(byte >> 4) : static_cast<uint8_t>(byte & 0x0F);
    dst[kk] = fp4_to_float(nib) * scale[kk >> 4] * scale2;
  }
}
#endif  // __AVX512F__

inline void dequant_row(const uint8_t* packed, const float* scale, int k,
                        float scale2, float* dst) {
#if defined(__AVX512F__)
  dequant_row_avx512(packed, scale, k, scale2, dst);
#elif defined(__AVX2__)
  dequant_row_avx2(packed, scale, k, scale2, dst);
#else
  dequant_row_scalar(packed, scale, k, scale2, dst);
#endif
}

inline float dot_avx2(const float* a, const float* b, int k) {
#if defined(__AVX2__)
  __m256 acc = _mm256_setzero_ps();
  int kk = 0;
  for (; kk + 8 <= k; kk += 8) {
    acc = _mm256_fmadd_ps(_mm256_loadu_ps(a + kk), _mm256_loadu_ps(b + kk), acc);
  }
  alignas(32) float tmp[8];
  _mm256_store_ps(tmp, acc);
  float s = 0.0f;
  for (int t = 0; t < 8; ++t) s += tmp[t];
  for (; kk < k; ++kk) s += a[kk] * b[kk];
  return s;
#else
  float s = 0.0f;
  for (int kk = 0; kk < k; ++kk) s += a[kk] * b[kk];
  return s;
#endif
}

}  // namespace detail

// The definition of record, kept callable so the vector paths can be checked
// against it from outside this header.
inline void dequant_row_scalar(const uint8_t* packed, const float* scale, int k,
                               float scale2, float* dst) {
  detail::dequant_row_scalar(packed, scale, k, scale2, dst);
}

// out[m, n] = sum_k x[m, k] * W[n, k]
//
// A tile of `block_m` dequantised rows is built once and reused for every token in
// the tile, so dequantisation is divided by the effective batch instead of paid
// per output row.
static inline void dense_nvfp4_head_gemm(const DenseNVFP4HeadConfig& cfg,
                                         const float* x, int ldx, float* out,
                                         int ldo) {
  const int n = cfg.n;
  const int k = cfg.k;
  const int khalf = k / 2;
  const int kgroups = k / 16;
  const int bm = std::max(1, cfg.block_m);

  std::vector<float> tile(static_cast<size_t>(bm) * k);

  for (int nb = 0; nb < n; nb += cfg.block_n) {
    const int nend = std::min(nb + cfg.block_n, n);
    for (int rb = nb; rb < nend; rb += bm) {
      const int rend = std::min(rb + bm, nend);
      const int rows = rend - rb;
      for (int r = 0; r < rows; ++r) {
        const int nn = rb + r;
        detail::dequant_row(cfg.weight + static_cast<size_t>(nn) * khalf,
                            cfg.weight_scale + static_cast<size_t>(nn) * kgroups, k,
                            cfg.weight_scale_2, tile.data() + static_cast<size_t>(r) * k);
      }
      for (int mm = 0; mm < cfg.m; ++mm) {
        const float* xr = x + static_cast<size_t>(mm) * ldx;
        float* orow = out + static_cast<size_t>(mm) * ldo;
        for (int r = 0; r < rows; ++r) {
          orow[rb + r] = detail::dot_avx2(xr, tile.data() + static_cast<size_t>(r) * k, k);
        }
      }
    }
  }
}

}  // namespace nvfp4_head
}  // namespace kt
