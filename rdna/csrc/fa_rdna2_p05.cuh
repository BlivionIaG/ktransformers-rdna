
  // Write output. Vectorized half2 writes.
  for (int idx = t; idx < BR_PREFILL * (HEAD_DIM / 2); idx += THREADS_PREFILL) {
    const int br = idx / (HEAD_DIM / 2);
    const int d_pair = idx % (HEAD_DIM / 2);
    const int d = d_pair * 2;
    if (br < br_size) {
      const float inv_l = 1.0f / sL[br];
      half* O_row = O + (q_start_global + br) * stride_qo_tok + h_q * stride_qo_h;
      float final0 = sO[br * HEAD_DIM + d] * inv_l;
      float final1 = sO[br * HEAD_DIM + d + 1] * inv_l;
      *reinterpret_cast<half2*>(O_row + d) = __floats2half2_rn(final0, final1);
    }
  }
}

// HEAD_DIM = 256 varlen variant.
template <typename KV_T, bool IS_FP8>
__global__ __launch_bounds__(256, 1) void fa_prefill_paged_varlen_kernel_256(
    const half* __restrict__ Q,
    const KV_T* __restrict__ key_cache,
    const KV_T* __restrict__ value_cache,
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
    half* __restrict__ O,
    const int H_q,
    const int H_kv,
    const int kv_group_num,
    const float scale,
    const int causal,
    const int sliding_window,
    const float k_scale,
    const float v_scale) {

  // Grid: (max_q_blocks_per_seq, H_q, num_seqs). Each CTA handles one
  // sequence's query block — blockIdx.z = seq_idx, blockIdx.x = q_block
  // within that sequence. This guarantees every token is covered by
  // exactly one CTA (no boundary gaps when q_block spans two sequences).
  const int seq_idx = blockIdx.z;
  const int q_block = blockIdx.x;
  const int h_q = blockIdx.y;
  const int t = threadIdx.x;
  const int h_kv = h_q / kv_group_num;
  if (seq_idx >= num_seqs || h_q >= H_q) return;

  const int q_start_in_seq = q_block * BR_PREFILL;
  const int seq_query_len = cu_query_lens[seq_idx + 1] - cu_query_lens[seq_idx];
  if (q_start_in_seq >= seq_query_len) return;
  const int q_start_global = cu_query_lens[seq_idx] + q_start_in_seq;
  const int seq_len = seq_lens[seq_idx];
  const int br_size = min(BR_PREFILL, seq_query_len - q_start_in_seq);
  const int* seq_block_table = block_table + seq_idx * max_blocks;

  const int stride_qo_tok = H_q * 256;
  const int stride_qo_h = 256;

  if (seq_len <= 0) {
    for (int idx = t; idx < BR_PREFILL * 256; idx += 256) {
      const int br = idx / 256;
      const int d = idx % 256;
      if (br < br_size) {
        O[(q_start_global + br) * stride_qo_tok + h_q * stride_qo_h + d] = __float2half(0.0f);
      }
    }
    return;
  }

  // sK/sV rows padded to 264 halves so the vectorized K store (one 16B
  // write per lane across consecutive n_local) does not collapse onto a
  // single smem bank quad.
  constexpr int DSK = 256 + 8;
  extern __shared__ unsigned char smem_raw[];
  half*  sQ  = reinterpret_cast<half*>(smem_raw);
  half*  sK  = sQ + BR_PREFILL * 256;
  half*  sV  = sK + BC_256 * DSK;
  float* sP  = reinterpret_cast<float*>(sV + BC_256 * DSK);
  float* sM  = sP + BC_256 * BR_PREFILL;
  float* sL  = sM + BR_PREFILL;
  float* sO  = sL + BR_PREFILL;
  __shared__ int s_blk[BC_256];
  __shared__ int s_slot[BC_256];

  {
    const half* Q_row = Q + (q_start_global * stride_qo_tok + h_q * stride_qo_h);
    for (int g = t; g < BR_PREFILL * 32; g += 256) {
      const int br = g / 32;
      const int dx = g % 32;
      uint4 q4 = make_uint4(0, 0, 0, 0);
      if (br < br_size) {
        q4 = *reinterpret_cast<const uint4*>(Q_row + br * stride_qo_tok
                                             + dx * 8);
      }
      *reinterpret_cast<uint4*>(&sQ[br * 256 + dx * 8]) = q4;
    }
  }
  __syncthreads();

  if (t < BR_PREFILL) {
    sM[t] = -INFINITY;
    sL[t] = 0.0f;
  }
  for (int i = t; i < BR_PREFILL * 256; i += 256) {
    sO[i] = 0.0f;
  }
  __syncthreads();

  // Stream over the KV tiles this q block can see.
  const int q_first = (seq_len - seq_query_len) + q_start_in_seq;
  int kv_lo = 0, kv_hi = seq_len;
  fa_clip_kv_walk(kv_lo, kv_hi, q_first, br_size, causal, sliding_window,
                  BC_256);
  for (int n = kv_lo; n < kv_hi; n += BC_256) {
    const int blk_size = min(BC_256, kv_hi - n);

    // Page mapping once per KV block: kills the per-element div/mod and
    // block_table re-reads.
    if (t < BC_256) {
      const int n_global = n + t;
      const bool ok = (t < blk_size);
      s_blk[t] = ok ? seq_block_table[n_global / block_size] : 0;
      s_slot[t] = ok ? (n_global % block_size) : 0;
    }
    __syncthreads();

    const bool kv_vec_ok =
        (sizeof(KV_T) == 2) && (!IS_FP8)
        && stride_kc4 == 1 && stride_vc3 == 1
        && x_dim == 8 && ((block_size & 7) == 0);
    if (kv_vec_ok) {
      // K is x-packed: 8 consecutive d are contiguous per (slot, d_sub).
      // n_local-fastest mapping: each warp covers 32 consecutive slots
      // (512B contiguous global) for one d_sub; padded rows keep the 16B
      // smem stores at <=4-way bank conflicts.
      constexpr int NX = 256 / 8;
      for (int i = t; i < BC_256 * NX; i += 256) {
        const int n_local = i % BC_256;
        const int d_sub = i / BC_256;
        if (n_local < blk_size) {
          const half* kp = reinterpret_cast<const half*>(key_cache)
              + s_blk[n_local] * stride_kc0
              + h_kv * stride_kc1 + d_sub * stride_kc2
              + s_slot[n_local] * stride_kc3;
          *reinterpret_cast<uint4*>(&sK[n_local * DSK + d_sub * 8]) =
              *reinterpret_cast<const uint4*>(kp);
        }
      }
      // V is slot-innermost: 8 consecutive slots contiguous per d.
      // Slot-group-fastest mapping: 64B contiguous global runs per warp.
      constexpr int NSG = BC_256 / 8;
      for (int i = t; i < 256 * NSG; i += 256) {
        const int sg = i % NSG;
        const int d = i / NSG;
        const int n_local = sg * 8;
        if (n_local < blk_size) {
          const half* vp = reinterpret_cast<const half*>(value_cache)
              + s_blk[n_local] * stride_vc0
              + h_kv * stride_vc1 + (d / 8) * stride_vc2
              + (d % 8) * stride_vc4 + s_slot[n_local] * stride_vc3;
          if ((s_slot[n_local] & 7) == 0) {
            const uint4 v4 = *reinterpret_cast<const uint4*>(vp);
            const half* vv = reinterpret_cast<const half*>(&v4);
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
              sV[(n_local + j) * DSK + d] = vv[j];
            }
          } else {
            // Misaligned groups can straddle a block boundary — per-element
            // page lookup (aligned groups never straddle: block_size%8==0).
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
              const int nl = n_local + j;
              if (nl < blk_size) {
                sV[nl * DSK + d] = *(reinterpret_cast<const half*>(value_cache)
                    + s_blk[nl] * stride_vc0 + h_kv * stride_vc1
                    + (d / 8) * stride_vc2 + (d % 8) * stride_vc4
                    + s_slot[nl] * stride_vc3);
              }
            }
          }
        }
      }
    } else {
      for (int i = t; i < BC_256 * 256; i += 256) {
        const int n_local = i / 256;
        const int d = i % 256;
        if (n_local < blk_size) {
          const int d_sub = d / x_dim;
          const int x_idx = d % x_dim;
          const KV_T* k_ptr = key_cache
              + s_blk[n_local] * stride_kc0 + h_kv * stride_kc1
              + d_sub * stride_kc2 + s_slot[n_local] * stride_kc3
              + x_idx * stride_kc4;
          const KV_T* v_ptr = value_cache
              + s_blk[n_local] * stride_vc0 + h_kv * stride_vc1
              + d_sub * stride_vc2 + s_slot[n_local] * stride_vc3
              + x_idx * stride_vc4;
          sK[n_local * DSK + d] = fa_kv_load<KV_T, IS_FP8>(k_ptr, k_scale);
          sV[n_local * DSK + d] = fa_kv_load<KV_T, IS_FP8>(v_ptr, v_scale);
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

  for (int idx = t; idx < BR_PREFILL * 256; idx += 256) {
    const int br = idx / 256;
    const int d = idx % 256;
    if (br < br_size) {
      const float inv_l = 1.0f / sL[br];
      half* O_row = O + (q_start_global + br) * stride_qo_tok + h_q * stride_qo_h;
      float final_val = sO[br * 256 + d] * inv_l;
      O_row[d] = __float2half_rn(final_val);
    }
  }
}

// =====================================================================
// SPLIT-K PREFILL (PAGED, VARLEN): Multiple CTAs per (seq, q_block, h_q)
// =====================================================================
//
// Partitions the KV sequence dimension across multiple CTAs when
// seq_len is large. Each split CTA processes a different KV range and
// outputs partial O (unnormalized), M (row max), L (row sum).
// A separate reduction kernel combines the partials using the standard
// online softmax merge formula.
//
// Grid:
//   x: max_q_blocks_per_seq
//   y: H_q
//   z: num_seqs * kv_splits
// where the z dim encodes both seq_idx and split_idx.
//
// This increases grid utilization from H_q to H_q * kv_splits, which is
// important for large seq_len where the non-split kernel only launches
// H_q CTAs per q_block (40 for Qwen3.5, underutilizing 72 CUs).
// =====================================================================

__global__ __launch_bounds__(128, 1) void fa_prefill_paged_varlen_splitk_kernel_128(
    const half* __restrict__ Q,
    const half* __restrict__ key_cache,
    const half* __restrict__ value_cache,
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
    const int sliding_window) {

  const int seq_idx = blockIdx.z / kv_splits;
  const int split_idx = blockIdx.z % kv_splits;
  const int q_block = blockIdx.x;
  const int h_q = blockIdx.y;
  const int t = threadIdx.x;
  const int h_kv = h_q / kv_group_num;
  if (seq_idx >= num_seqs || h_q >= H_q) return;

  const int q_start_in_seq = q_block * BR_PREFILL;
  const int seq_query_len = cu_query_lens[seq_idx + 1] - cu_query_lens[seq_idx];
  if (q_start_in_seq >= seq_query_len) return;
  const int q_start_global = cu_query_lens[seq_idx] + q_start_in_seq;
  const int seq_len = seq_lens[seq_idx];
  const int br_size = min(BR_PREFILL, seq_query_len - q_start_in_seq);
  const int* seq_block_table = block_table + seq_idx * max_blocks;

  // Split the KV range [0, seq_len) into kv_splits chunks, then drop the
  // tiles this q block cannot see.
  const int kv_per_split = (seq_len + kv_splits - 1) / kv_splits;
  const int q_first = (seq_len - seq_query_len) + q_start_in_seq;
  int kv_start = split_idx * kv_per_split;
  int kv_end = min(kv_start + kv_per_split, seq_len);
  fa_clip_kv_walk(kv_start, kv_end, q_first, br_size, causal, sliding_window,
                  BC);
  if (kv_start >= kv_end) {
    // Empty split: write zero partials.
    const int partial_base_empty = ((q_start_global * H_q + h_q) * kv_splits) + split_idx;
    for (int br = 0; br < br_size; ++br) {
      M_partial[partial_base_empty + br * (H_q * kv_splits)] = -INFINITY;
      L_partial[partial_base_empty + br * (H_q * kv_splits)] = 0.0f;
      for (int d = t; d < 128; d += 128) {
        O_partial[(partial_base_empty + br * (H_q * kv_splits)) * 128 + d] = 0.0f;
      }
    }
    return;
  }

  // O and Q have shape [N_q, H_q, D]. Stride: token dim = H_q*D, head dim = D.
  const int stride_qo_tok = H_q * 128;
