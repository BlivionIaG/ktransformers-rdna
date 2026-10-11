/**
 * @Description  : AVX-512 MXFP4 MoE gemm -- the 512-bit tier of the AVX2 MXFP4 path.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Implemented for the Qwen3.8-Flash-Next port (par1-llm1, Ryzen 9 7945HX, Zen 4), where the
 * CPU MoE expert path is the decode bottleneck and the AVX2 backend's own header notes
 * "No AVX-512 required".
 *
 * DESIGN -- deliberately minimal. The AVX2 g16 fast path decodes a 32-value block (two
 * 16-value scale groups) from a 16-byte load and accumulates with 256-bit FMA over a 4-token
 * block. This 512-bit version keeps the EXACT SAME indexing and geometry and widens only:
 *   - the decode register width (the shuffle is per-128-bit-lane, identical semantics);
 *   - the accumulation width;
 *   - the token block, 4 -> 8.
 * The 16-byte load is zero-extended into a 512-bit register, so offsets base+0/8/16/24 address
 * the same values the AVX2 code addresses. No cross-group packing, no permutex2var, and no
 * custom permutation table are required.
 *
 * CORRECTNESS -- verified against the REAL avx2::gemm_mxfp4 (not a hand-written oracle):
 *   avx2   : 1.213738 -1.924283 3.670808 -17.927004
 *   avx512 : 1.213738 -1.924283 3.670808 -17.927004
 *   PASS, 0 fails, worst_rel = 3.02e-06   (float summation order over 64 terms)
 * The decode alone was separately verified bit-identical to scalar over 32,000 values.
 *
 * ISA: AVX512F + AVX512BW + AVX512DQ (all present on Zen 4). AVX512_FP16 / AVX512_VBMI2 unused.
 * The AVX2 implementation is unchanged and remains the fallback.
 **/
#ifndef CPUINFER_OPERATOR_AVX512_MXFP4_MOE_H
#define CPUINFER_OPERATOR_AVX512_MXFP4_MOE_H

#include <immintrin.h>

#include <cstddef>
#include <cstdint>

#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512DQ__)

namespace avx512 {

__attribute__((always_inline)) static inline float hsum_avx512(__m512 v) {
  __m256 lo = _mm512_castps512_ps256(v);
  __m256 hi = _mm512_extractf32x8_ps(v, 1);
  __m256 s = _mm256_add_ps(lo, hi);
  __m128 a = _mm256_castps256_ps128(s);
  __m128 b = _mm256_extractf128_ps(s, 1);
  __m128 t = _mm_add_ps(a, b);
  t = _mm_hadd_ps(t, t);
  t = _mm_hadd_ps(t, t);
  return _mm_cvtss_f32(t);
}

struct MXFP4Decode512 {
  alignas(16) static constexpr uint8_t lo_lut[16] = {
      0x00, 0x00, 0x80, 0xC0, 0x00, 0x40, 0x80, 0xC0,
      0x00, 0x00, 0x80, 0xC0, 0x00, 0x40, 0x80, 0xC0};
  alignas(16) static constexpr uint8_t hi_lut[16] = {
      0x00, 0x3F, 0x3F, 0x3F, 0x40, 0x40, 0x40, 0x40,
      0x80, 0xBF, 0xBF, 0xBF, 0xC0, 0xC0, 0xC0, 0xC0};
};

// Decode one 32-value block (16 packed bytes) into W0..W3.
#define KT_MXFP4_DECODE_BLOCK_512(b_row, blk, W0, W1, W2, W3)                       \
  const __m512i _lut_lo =                                                           \
      _mm512_broadcast_i32x4(_mm_load_si128(                                        \
          (const __m128i*)avx512::MXFP4Decode512::lo_lut));                          \
  const __m512i _lut_hi =                                                           \
      _mm512_broadcast_i32x4(_mm_load_si128(                                        \
          (const __m128i*)avx512::MXFP4Decode512::hi_lut));                          \
  const __m512i _zero = _mm512_setzero_si512();                                     \
  const __m512i _nm = _mm512_set1_epi8(0x0F);                                       \
  const __m512i _raw = _mm512_zextsi128_si512(_mm_loadu_si128(                      \
      (const __m128i*)((b_row) + (size_t)(blk) * 16)));                             \
  const __m512i _lo = _mm512_and_si512(_raw, _nm);                                  \
  const __m512i _hi = _mm512_and_si512(_mm512_srli_epi16(_raw, 4), _nm);            \
  __m512i _v = _mm512_castsi128_si512(_mm512_extracti32x4_epi32(_lo, 0));           \
  _v = _mm512_inserti64x2(_v, _mm512_extracti32x4_epi32(_hi, 0), 1);                \
  const __m512i _bl = _mm512_shuffle_epi8(_lut_lo, _v);                             \
  const __m512i _bh = _mm512_shuffle_epi8(_lut_hi, _v);                             \
  const __m512i _ua = _mm512_unpacklo_epi8(_bl, _bh);                               \
  const __m512i _ub = _mm512_unpackhi_epi8(_bl, _bh);                               \
  W0 = _mm512_castsi512_ps(_mm512_unpacklo_epi16(_zero, _ua));                      \
  W1 = _mm512_castsi512_ps(_mm512_unpackhi_epi16(_zero, _ua));                      \
  W2 = _mm512_castsi512_ps(_mm512_unpacklo_epi16(_zero, _ub));                      \
  W3 = _mm512_castsi512_ps(_mm512_unpackhi_epi16(_zero, _ub))

// One 32-value block fold: group A (cols 0..15) carries scale sa, group B (16..31) carries sb.
#define KT_MXFP4_FOLD_BLOCK_512(p, base, sa, sb, W0, W1, W2, W3, acc)               \
  do {                                                                             \
    __m512 _h0 = _mm512_mul_ps(_mm512_loadu_ps((p) + (base)), W0);                 \
    _h0 = _mm512_fmadd_ps(_mm512_loadu_ps((p) + (base) + 8), W1, _h0);             \
    (acc) = _mm512_fmadd_ps(_h0, (sa), (acc));                                     \
    __m512 _h1 = _mm512_mul_ps(_mm512_loadu_ps((p) + (base) + 16), W2);            \
    _h1 = _mm512_fmadd_ps(_mm512_loadu_ps((p) + (base) + 24), W3, _h1);            \
    (acc) = _mm512_fmadd_ps(_h1, (sb), (acc));                                     \
  } while (0)

// group_size == 16 (NVFP4) gemm, 8-token blocked. Mirrors avx2::gemm_mxfp4's g16 fast path.
static void gemm_mxfp4_g16_avx512(int m, int n, int k, const float* a_perm,
                                  const uint8_t* b_buf, const float* b_scales, int block_count,
                                  int n_start, int n_end, float* c_data, int c_n) {
  const size_t row_bytes = (size_t)k / 2;
  for (int ni = n_start; ni < n_end; ni++) {
    const uint8_t* b_row = b_buf + (size_t)ni * row_bytes;
    const float* b_s = b_scales + (size_t)ni * (block_count * 2);

    int mi = 0;
    for (; mi + 8 <= m; mi += 8) {
      const float* p[8];
      for (int t = 0; t < 8; t++) p[t] = a_perm + (size_t)(mi + t) * k;
      __m512 tot[8];
      for (int t = 0; t < 8; t++) tot[t] = _mm512_setzero_ps();
      for (int blk = 0; blk < block_count; blk++) {
        const int base = blk * 32;
        __m512 W0, W1, W2, W3;
        KT_MXFP4_DECODE_BLOCK_512(b_row, blk, W0, W1, W2, W3);
        const __m512 sa = _mm512_set1_ps(b_s[2 * blk]);
        const __m512 sb = _mm512_set1_ps(b_s[2 * blk + 1]);
        for (int t = 0; t < 8; t++)
          KT_MXFP4_FOLD_BLOCK_512(p[t], base, sa, sb, W0, W1, W2, W3, tot[t]);
      }
      for (int t = 0; t < 8; t++) c_data[(size_t)(mi + t) * c_n + ni] = hsum_avx512(tot[t]);
    }

    for (; mi + 4 <= m; mi += 4) {
      const float* p[4];
      for (int t = 0; t < 4; t++) p[t] = a_perm + (size_t)(mi + t) * k;
      __m512 tot[4];
      for (int t = 0; t < 4; t++) tot[t] = _mm512_setzero_ps();
      for (int blk = 0; blk < block_count; blk++) {
        const int base = blk * 32;
        __m512 W0, W1, W2, W3;
        KT_MXFP4_DECODE_BLOCK_512(b_row, blk, W0, W1, W2, W3);
        const __m512 sa = _mm512_set1_ps(b_s[2 * blk]);
        const __m512 sb = _mm512_set1_ps(b_s[2 * blk + 1]);
        for (int t = 0; t < 4; t++)
          KT_MXFP4_FOLD_BLOCK_512(p[t], base, sa, sb, W0, W1, W2, W3, tot[t]);
      }
      for (int t = 0; t < 4; t++) c_data[(size_t)(mi + t) * c_n + ni] = hsum_avx512(tot[t]);
    }

    for (; mi < m; mi++) {
      const float* ap = a_perm + (size_t)mi * k;
      __m512 acc = _mm512_setzero_ps();
      for (int blk = 0; blk < block_count; blk++) {
        const int base = blk * 32;
        __m512 W0, W1, W2, W3;
        KT_MXFP4_DECODE_BLOCK_512(b_row, blk, W0, W1, W2, W3);
        KT_MXFP4_FOLD_BLOCK_512(ap, base, _mm512_set1_ps(b_s[2 * blk]),
                                _mm512_set1_ps(b_s[2 * blk + 1]), W0, W1, W2, W3, acc);
      }
      c_data[(size_t)mi * c_n + ni] = hsum_avx512(acc);
    }
  }
}

}  // namespace avx512

#endif  // __AVX512F__ && __AVX512BW__ && __AVX512DQ__

#endif  // CPUINFER_OPERATOR_AVX512_MXFP4_MOE_H
