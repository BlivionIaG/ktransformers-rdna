    } else {
      for (int i = t; i < BC_256 * 256; i += 256) {
        const int n_local = i / 256;
        const int d = i % 256;
        if (n_local < blk_size) {
          const int d_sub = d / x_dim;
          const int x_idx = d % x_dim;
          const half* k_ptr = key_cache
              + s_blk[n_local] * stride_kc0 + h_kv * stride_kc1
              + d_sub * stride_kc2 + s_slot[n_local] * stride_kc3
              + x_idx * stride_kc4;
          const half* v_ptr = value_cache
              + s_blk[n_local] * stride_vc0 + h_kv * stride_vc1
              + d_sub * stride_vc2 + s_slot[n_local] * stride_vc3
              + x_idx * stride_vc4;
          sK[n_local * DSK + d] = *k_ptr;
          sV[n_local * DSK + d] = *v_ptr;
        }
      }
    }
    __syncthreads();

    for (int idx = t; idx < BR_PREFILL * BC_256; idx += 256) {
      const int br = idx / BC_256;
      const int k = idx % BC_256;
      float acc = 0.0f;
      if (br < br_size && k < blk_size) {
        if (!fa_masked(q_first + br, n + k, causal, sliding_window)) {
          const half* sQ_row = sQ + br * 256;
          const half* sK_row = sK + k * DSK;
          #pragma unroll
          for (int d = 0; d < 256; d += 2) {
            half2 q2 = *reinterpret_cast<const half2*>(&sQ_row[d]);
            half2 k2 = *reinterpret_cast<const half2*>(&sK_row[d]);
            acc = fdot2(q2, k2, acc);
          }
          sP[br * BC_256 + k] = acc * scale;
        } else {
          sP[br * BC_256 + k] = -INFINITY;
        }
      } else {
        sP[br * BC_256 + k] = 0.0f;
      }
    }
    __syncthreads();

    if (t < BR_PREFILL && t < br_size) {
      float row_max = -INFINITY;
      for (int k = 0; k < blk_size; ++k) {
        row_max = fmaxf(row_max, sP[t * BC_256 + k]);
      }
      // Skip update if all KV positions in this block are masked (causal).
      // Without this guard, exp(-INFINITY - (-INFINITY)) = exp(NaN) = NaN,
      // which corrupts sL and propagates to L_partial -> NaN output.
      if (row_max > -INFINITY) {
        float new_m = fmaxf(sM[t], row_max);
        float exp_diff = expf(sM[t] - new_m);

        float sum_p = 0.0f;
        for (int k = 0; k < blk_size; ++k) {
          sum_p += expf(sP[t * BC_256 + k] - new_m);
        }
        sL[t] = exp_diff * sL[t] + sum_p;

        for (int d = 0; d < 256; ++d) {
          sO[t * 256 + d] *= exp_diff;
        }
        sM[t] = new_m;

        for (int k = 0; k < blk_size; ++k) {
          sP[t * BC_256 + k] = expf(sP[t * BC_256 + k] - new_m);
        }
      } else {
        // All-masked block: zero sP so PV loop contributes nothing.
        for (int k = 0; k < blk_size; ++k) {
          sP[t * BC_256 + k] = 0.0f;
        }
      }
    }
    __syncthreads();

    for (int idx = t; idx < BR_PREFILL * 256; idx += 256) {
      const int br = idx / 256;
      const int d = idx % 256;
      if (br < br_size) {
        float pv = 0.0f;
        #pragma unroll
        for (int k = 0; k < BC_256; ++k) {
          if (k < blk_size) {
            float p_val = sP[br * BC_256 + k];
            float v_val = __half2float(sV[k * DSK + d]);
            pv = fmaf(p_val, v_val, pv);
          }
        }
        sO[br * 256 + d] += pv;
      }
    }
    __syncthreads();
  }

  const int partial_base = ((q_start_global * H_q + h_q) * kv_splits) + split_idx;
  // Only real rows (br < br_size) are written: the partial buffers are
  // sized [N, H_q, kv_splits, *], so writing padding rows of the last
  // q_block would index past the allocation when N % BR_PREFILL != 0.
  if (t < br_size) {
    const int br = t;
    const int64_t slot = partial_base + br * (H_q * kv_splits);
    M_partial[slot] = sM[br];
    L_partial[slot] = sL[br];
  }
  for (int br = 0; br < br_size; ++br) {
    const int64_t slot = partial_base + br * (H_q * kv_splits);
    for (int d = t; d < 256; d += 256) {
      O_partial[slot * 256 + d] = sO[br * 256 + d];
    }
  }
}

// HEAD_DIM = 256 reduction kernel.
// Partial layout: [N, H_q, BR_PREFILL, kv_splits, D].
// Grid: (max_q_blocks, H_q, 1). One CTA per (q_block, h_q); loops over BR_PREFILL
// query rows within the block.
//
// IMPORTANT: The partial write address uses q_start_global (= q_block*BR_PREFILL),
// NOT the per-br token. All br rows in a q_block share the same q_start_global.
__global__ void fa_prefill_paged_varlen_splitk_reduce_kernel_256(
    const float* __restrict__ O_partial,
    const float* __restrict__ M_partial,
    const float* __restrict__ L_partial,
    half* __restrict__ O,
    const int max_q_blocks,
    const int H_q,
    const int kv_splits,
    const int stride_qo_tok,
    const int stride_qo_h,
    const int num_tokens) {
  const int q_block = blockIdx.x;
  const int h_q = blockIdx.y;
  const int d = threadIdx.x;
  if (q_block >= max_q_blocks || h_q >= H_q || d >= 256) return;

  const int q_start_global = q_block * BR_PREFILL;

  for (int br = 0; br < BR_PREFILL; ++br) {
    const int token = q_start_global + br;
    // Padding rows of the last q_block have no partials written for them;
    // skip so we never read/write past the [N, ...] buffers.
    if (token >= num_tokens) break;
    // Partial slot base: uses q_start_global (shared across all br in this q_block)
    // + br offset within the BR_PREFILL dimension. NOT token.
    const int64_t slot_base =
        ((int64_t)((q_start_global + br) * H_q + h_q)) * (int64_t)kv_splits;

    float m_global = -INFINITY;
    for (int s = 0; s < kv_splits; ++s) {
      m_global = fmaxf(m_global, M_partial[slot_base + s]);
    }
    if (m_global == -INFINITY) {
      O[token * stride_qo_tok + h_q * stride_qo_h + d] = __float2half(0.0f);
      continue;
    }
    float l_combined = 0.0f;
    float o_combined = 0.0f;
    for (int s = 0; s < kv_splits; ++s) {
      const float w = expf(M_partial[slot_base + s] - m_global);
      l_combined += w * L_partial[slot_base + s];
      o_combined += w * O_partial[(slot_base + s) * 256 + d];
    }
    O[token * stride_qo_tok + h_q * stride_qo_h + d] =
        __float2half_rn(o_combined / l_combined);
  }
}

// =====================================================================
// SPLIT-K PREFILL (PAGED, VARLEN) — INT8 PER-TOKEN-HEAD
// =====================================================================
//
// Native int8 prefill kernel. Mirrors the fp16 splitk structure but uses
// the v3 wiki "live contract" pattern for the QK dot product (packed
// dword reads from sK, fused i8->fp32->scale->fp16->V_DOT2). The K/V
// cache is int8 with per-(token,head) scales; sK/sV are stored as int8
// in shared memory (½ the smem cost of the fp16 splitk kernel).
//
// Smem layout (D=128, BR=16, BC=64):
//   sQ         [BR][D] fp16    = 16 * 128 * 2  = 4 KB
//   sK, sV     [BC][D] int8    = 64 * 128 * 1  = 8 KB each (vs 16 KB fp16)
//   sP         [BC][BR] fp32   = 64 * 16 * 4   = 4 KB
//   sM, sL     [BR] fp32       = 16 * 4 * 2    = 128 B
//   sO         [BR][D] fp32    = 16 * 128 * 4  = 8 KB
//   sKscales   [BC] fp32       = 64 * 4        = 256 B
//   sVscales   [BC] fp32       = 64 * 4        = 256 B
//   sRed       [5] fp32        = 20 B
//   Total                      ≈ 32.7 KB (vs ~48 KB fp16 splitk)
//
// block_size is hardcoded to 16 (matches the int8 cache layout used by
// vLLM for INT8_PER_TOKEN_HEAD).
//
// Grid: (max_q_blocks_per_seq, H_q, num_seqs * kv_splits). z encodes
// (seq_idx, split_idx) — same convention as the fp16 splitk kernel.
// =====================================================================

__global__ __launch_bounds__(128, 1) void fa_prefill_paged_varlen_splitk_kernel_int8_128(
    const half* __restrict__ Q,
    const int8_t* __restrict__ key_cache,
    const int8_t* __restrict__ value_cache,
    const int* __restrict__ block_table,
    const int* __restrict__ cu_query_lens,
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
    const int num_seqs,
    const int kv_splits,
    float* __restrict__ O_partial,
    float* __restrict__ M_partial,
    float* __restrict__ L_partial,
    const int H_q,
    const int H_kv,
    const int kv_group_num,
    const float scale,
    const int causal,
    const int sliding_window,
    const float* __restrict__ k_scale_ptr,
    const float* __restrict__ v_scale_ptr) {

  constexpr int HEAD_DIM = 128;
  constexpr int THREADS_PREFILL_LOC = 128;
  constexpr int BC_LOC = 64;
  constexpr int BR_PREFILL_LOC = 16;
  constexpr int NDWORDS = HEAD_DIM / 4;

  const int seq_idx = blockIdx.z / kv_splits;
  const int split_idx = blockIdx.z % kv_splits;
  const int q_block = blockIdx.x;
  const int h_q = blockIdx.y;
  const int t = threadIdx.x;
  const int h_kv = h_q / kv_group_num;
  if (seq_idx >= num_seqs || h_q >= H_q) return;

  const int q_start_in_seq = q_block * BR_PREFILL_LOC;
  const int seq_query_len = cu_query_lens[seq_idx + 1] - cu_query_lens[seq_idx];
  if (q_start_in_seq >= seq_query_len) return;
  const int q_start_global = cu_query_lens[seq_idx] + q_start_in_seq;
  const int seq_len = seq_lens[seq_idx];
  const int br_size = min(BR_PREFILL_LOC, seq_query_len - q_start_in_seq);
  const int* seq_block_table = block_table + seq_idx * max_blocks;

  // Split the KV range [0, seq_len) into kv_splits chunks, then drop the
  // tiles this q block cannot see.
  const int kv_per_split = (seq_len + kv_splits - 1) / kv_splits;
  const int q_first = (seq_len - seq_query_len) + q_start_in_seq;
  int kv_start = split_idx * kv_per_split;
  int kv_end = min(kv_start + kv_per_split, seq_len);
  fa_clip_kv_walk(kv_start, kv_end, q_first, br_size, causal, sliding_window,
                  BC_LOC);
  if (kv_start >= kv_end) {
    // Empty split: write zero partials.
    const int partial_base_empty = ((q_start_global * H_q + h_q) * kv_splits) + split_idx;
    for (int br = 0; br < br_size; ++br) {
      M_partial[partial_base_empty + br * (H_q * kv_splits)] = -INFINITY;
      L_partial[partial_base_empty + br * (H_q * kv_splits)] = 0.0f;
      for (int d = t; d < HEAD_DIM; d += THREADS_PREFILL_LOC) {
        O_partial[(partial_base_empty + br * (H_q * kv_splits)) * HEAD_DIM + d] = 0.0f;
      }
    }
    return;
  }

  const int stride_qo_tok = H_q * HEAD_DIM;
  const int stride_qo_h = HEAD_DIM;

  extern __shared__ unsigned char smem_raw[];
  half*   sQ  = reinterpret_cast<half*>(smem_raw);
  int8_t* sK  = reinterpret_cast<int8_t*>(sQ + BR_PREFILL_LOC * HEAD_DIM);
  int8_t* sV  = sK + BC_LOC * HEAD_DIM;
  float*  sP  = reinterpret_cast<float*>(sV + BC_LOC * HEAD_DIM);
  float*  sM  = sP + BC_LOC * BR_PREFILL_LOC;
  float*  sL  = sM + BR_PREFILL_LOC;
  float*  sO  = sL + BR_PREFILL_LOC;
  float*  sKscales = sO + BR_PREFILL_LOC * HEAD_DIM;
  float*  sVscales = sKscales + BC_LOC;

  // Load Q[Br x D] into shared memory (fp16).
  {
    const half* Q_row = Q + (q_start_global * stride_qo_tok + h_q * stride_qo_h);
    for (int i = t; i < BR_PREFILL_LOC * HEAD_DIM; i += THREADS_PREFILL_LOC) {
      const int br = i / HEAD_DIM;
      const int d = i % HEAD_DIM;
      sQ[i] = (br < br_size) ? Q_row[br * stride_qo_tok + d] : __float2half(0.0f);
    }
  }
  __syncthreads();

  if (t < BR_PREFILL_LOC) {
    sM[t] = -INFINITY;
    sL[t] = 0.0f;
  }
  for (int i = t; i < BR_PREFILL_LOC * HEAD_DIM; i += THREADS_PREFILL_LOC) {
    sO[i] = 0.0f;
  }
  __syncthreads();

  // Stream over this split's KV range.
  for (int n = kv_start; n < kv_end; n += BC_LOC) {
    const int blk_size = min(BC_LOC, kv_end - n);

    // Cooperative load sK[BC][D] and sV[BC][D] from paged int8 cache.
    // Each byte read goes through the paged address translation; the
    // resulting sK/sV are laid out [BC][D] int8 (contiguous in d).
    for (int i = t; i < BC_LOC * HEAD_DIM; i += THREADS_PREFILL_LOC) {
      const int n_local = i / HEAD_DIM;
      const int d = i % HEAD_DIM;
      if (n_local < blk_size) {
        const int n_global = n + n_local;
        const int block_idx = seq_block_table[n_global / block_size];
        const int slot = n_global % block_size;
        const int d_sub = d / x_dim;
        const int x_idx = d % x_dim;
        const int8_t* k_ptr = key_cache
            + block_idx * stride_kc0
            + h_kv * stride_kc1
            + d_sub * stride_kc2
            + slot * stride_kc3
            + x_idx * stride_kc4;
        const int8_t* v_ptr = value_cache
            + block_idx * stride_vc0
            + h_kv * stride_vc1
            + d_sub * stride_vc2
            + slot * stride_vc3
            + x_idx * stride_vc4;
        sK[i] = *k_ptr;
        sV[i] = *v_ptr;
      }
    }
    // Pre-load per-(token,head) K and V scales for this KV block.
    for (int k = t; k < BC_LOC; k += THREADS_PREFILL_LOC) {
      if (k < blk_size) {
        const int n_global = n + k;
        sKscales[k] = k_scale_ptr[n_global * H_kv + h_kv];
        sVscales[k] = v_scale_ptr[n_global * H_kv + h_kv];
      } else {
        sKscales[k] = 1.0f;
        sVscales[k] = 1.0f;
      }
    }
    __syncthreads();

    // ---- QK: v3 wiki pattern — packed int reads from sK, fused
    //      i8->fp32 with per-(token,head) scale, promote to half2 pairs,
    //      V_DOT2_F32_F16 against pre-loaded fp16 Q. ----
    for (int idx = t; idx < BR_PREFILL_LOC * BC_LOC; idx += THREADS_PREFILL_LOC) {
      const int br = idx / BC_LOC;
      const int k = idx % BC_LOC;
      float acc = 0.0f;
      if (br < br_size && k < blk_size) {
        if (!fa_masked(q_first + br, n + k, causal, sliding_window)) {
          const float k_s = sKscales[k];
          const half* sQ_row = sQ + br * HEAD_DIM;
          const int8_t* sK_row = sK + k * HEAD_DIM;
          #pragma unroll
          for (int w = 0; w < NDWORDS; ++w) {
            // Packed dword read from sK (contiguous within row).
            const int32_t k_packed = *reinterpret_cast<const int*>(&sK_row[w * 4]);
            const float kf0 = (float)(int8_t)(k_packed & 0xFF) * k_s;
            const float kf1 = (float)(int8_t)((k_packed >> 8) & 0xFF) * k_s;
            const float kf2 = (float)(int8_t)((k_packed >> 16) & 0xFF) * k_s;
            const float kf3 = (float)(int8_t)((k_packed >> 24) & 0xFF) * k_s;
            const half2 k01 = __halves2half2(__float2half_rn(kf0), __float2half_rn(kf1));
            const half2 k23 = __halves2half2(__float2half_rn(kf2), __float2half_rn(kf3));
            const half2 q01 = *reinterpret_cast<const half2*>(&sQ_row[w * 4 + 0]);
            const half2 q23 = *reinterpret_cast<const half2*>(&sQ_row[w * 4 + 2]);
            acc = fdot2(q01, k01, acc);
            acc = fdot2(q23, k23, acc);
          }
          sP[br * BC_LOC + k] = acc * scale;
        } else {
          sP[br * BC_LOC + k] = -INFINITY;
        }
      } else {
        sP[br * BC_LOC + k] = 0.0f;
      }
    }
    __syncthreads();

    // ---- Online softmax update. Same as fp16 splitk. ----
    if (t < BR_PREFILL_LOC && t < br_size) {
      float row_max = -INFINITY;
      for (int k = 0; k < blk_size; ++k) {
        row_max = fmaxf(row_max, sP[t * BC_LOC + k]);
      }
