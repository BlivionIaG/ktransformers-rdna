// DECODE STAGE 1b: GQA-aware split-K kernel, head_dim = 256, fp16
// =====================================================================
// One CTA per (token, kv-head, split) handles the whole GQA group
// (G = H_q/H_kv query heads) so each KV tile is loaded once per group
// instead of once per query head: cuts DRAM KV traffic by G when L2
// does not catch the per-head reuse. 256 threads = 8 waves; wave w
// computes the S row for head g=w (32 k-lanes), the PV phase has every
// thread own one output column for all G heads.
//
// Fast-path preconditions (checked host-side; otherwise the per-head
// kernel runs): fp16, x_dim == 8, block_size % 8 == 0, K x-packed
// (stride_kc4 == 1), V slot-innermost (stride_vc3 == 1).
//
#define GQA_MAX_G 8
#define GQA_BC 32
#define GQA_DSK (256 + 8)

__global__ __launch_bounds__(256)
void fa_decode_paged_splitk_gqa_kernel_256(
    const half* __restrict__ Q,
    const half* __restrict__ key_cache,
    const half* __restrict__ value_cache,
    const int* __restrict__ block_table,
    const int* __restrict__ seq_lens,
    const int stride_kc0, const int stride_kc1, const int stride_kc2,
    const int stride_kc3,
    const int stride_vc0, const int stride_vc1, const int stride_vc2,
    const int stride_vc3, const int stride_vc4,
    const int max_blocks, const int block_size,
    float* __restrict__ O_partial,
    float* __restrict__ M_partial,
    float* __restrict__ L_partial,
    const int num_tokens, const int H_q, const int H_kv,
    const int kv_splits, const float scale, const int sliding_window,
    const int* __restrict__ cu_query_lens, const int num_seqs) {

  const int token_idx = blockIdx.x;
  const int h_kv = blockIdx.y;
  const int split = blockIdx.z;
  const int t = threadIdx.x;
  const int G = H_q / H_kv;
  if (token_idx >= num_tokens || h_kv >= H_kv || split >= kv_splits) return;
  int seq_len;
  const int seq = fa_decode_token_seq(cu_query_lens, num_seqs, token_idx,
                                      seq_lens, seq_len);
  if (seq_len <= 0) {
    if (t < 256) {
      for (int g = 0; g < G; ++g) {
        const int h_q = h_kv * G + g;
        O_partial[(((int64_t)token_idx * H_q + h_q) * kv_splits + split) * 256 + t] =
            0.0f;
      }
    }
    if (t == 0) {
      for (int g = 0; g < G; ++g) {
        const int h_q = h_kv * G + g;
        M_partial[(token_idx * H_q + h_q) * kv_splits + split] = -INFINITY;
        L_partial[(token_idx * H_q + h_q) * kv_splits + split] = 0.0f;
      }
    }
    return;
  }

  extern __shared__ unsigned char smem_raw[];
  half*  sQ = reinterpret_cast<half*>(smem_raw);             // [G][256]
  half*  sK = sQ + GQA_MAX_G * 256;                          // [BC][DSK]
  half*  sV = sK + GQA_BC * GQA_DSK;                         // [BC][DSK]
  float* sP = reinterpret_cast<float*>(sV + GQA_BC * GQA_DSK);  // [G][BC]
  float* sMnew = sP + GQA_MAX_G * GQA_BC;                    // [G]
  float* sLnew = sMnew + GQA_MAX_G;                          // [G]
  __shared__ int s_blk[GQA_BC];
  __shared__ int s_slot[GQA_BC];

  if (t < 256) {
    for (int g = 0; g < G; ++g) {
      sQ[g * 256 + t] = Q[((int64_t)token_idx * H_q + h_kv * G + g) * 256 + t];
    }
  }
  __syncthreads();

  const int* my_bt = block_table + (int64_t)seq * max_blocks;
  float m_i[GQA_MAX_G], l_i[GQA_MAX_G], o_acc[GQA_MAX_G];
#pragma unroll
  for (int g = 0; g < GQA_MAX_G; ++g) {
    m_i[g] = -INFINITY; l_i[g] = 0.0f; o_acc[g] = 0.0f;
  }

  // Same split layout as fa_decode_paged_splitk_kernel_256.
  const int kv_lo = sliding_window > 0
                        ? max(0, seq_len - sliding_window) / GQA_BC * GQA_BC
                        : 0;
  const int tokens_per_split =
      ((seq_len - kv_lo + kv_splits - 1) / kv_splits + GQA_BC - 1) / GQA_BC *
      GQA_BC;
  const int tok_start = kv_lo + split * tokens_per_split;
  const int tok_end = min(tok_start + tokens_per_split, seq_len);
  constexpr int NX = 256 / 8;
  constexpr int NSG = GQA_BC / 8;

  for (int n = tok_start; n < tok_end; n += GQA_BC) {
    const int blk_size = min(GQA_BC, tok_end - n);
    if (t < GQA_BC) {
      const int n_global = n + t;
      const bool ok = (t < blk_size);
      s_blk[t] = ok ? my_bt[n_global / block_size] : 0;
      s_slot[t] = ok ? (n_global % block_size) : 0;
    }
    __syncthreads();

    // K: x-packed rows (8 contiguous d per (slot, d_sub)).
    for (int i = t; i < GQA_BC * NX; i += 256) {
      const int n_local = i % GQA_BC;
      const int d_sub = i / GQA_BC;
      if (n_local < blk_size) {
        const half* kp = key_cache + (int64_t)s_blk[n_local] * stride_kc0
            + h_kv * stride_kc1 + d_sub * stride_kc2
            + s_slot[n_local] * stride_kc3;
        *reinterpret_cast<uint4*>(&sK[n_local * GQA_DSK + d_sub * 8]) =
            *reinterpret_cast<const uint4*>(kp);
      }
    }
    // V: slot-innermost (8 contiguous slots per d).
    for (int i = t; i < 256 * NSG; i += 256) {
      const int sg = i % NSG;
      const int d = i / NSG;
      const int n_local = sg * 8;
      if (n_local < blk_size) {
        const half* vp = value_cache + (int64_t)s_blk[n_local] * stride_vc0
            + h_kv * stride_vc1 + (d / 8) * stride_vc2
            + (d % 8) * stride_vc4 + s_slot[n_local] * stride_vc3;
        if ((s_slot[n_local] & 7) == 0) {
          const uint4 v4 = *reinterpret_cast<const uint4*>(vp);
          const half* vv = reinterpret_cast<const half*>(&v4);
#pragma unroll
          for (int j = 0; j < 8; ++j) sV[(n_local + j) * GQA_DSK + d] = vv[j];
        } else {
#pragma unroll
          for (int j = 0; j < 8; ++j) {
            const int nl = n_local + j;
            if (nl < blk_size) {
              sV[nl * GQA_DSK + d] = *(value_cache
                  + (int64_t)s_blk[nl] * stride_vc0 + h_kv * stride_vc1
                  + (d / 8) * stride_vc2 + (d % 8) * stride_vc4
                  + s_slot[nl] * stride_vc3);
            }
          }
        }
      }
    }
    __syncthreads();

    // S phase: wave w computes head g=w for its 32 k-lanes.
    // Idle waves (g >= G) still participate in the warp shuffle so the
    // DPP is well-defined; they do not write sP/sMnew.
    const int wlane = t & 31;
    const int g = t >> 5;
    float s_k = -INFINITY;
    if (g < G && wlane < blk_size) {
      const int kv_idx = n + wlane;
      const bool in_window =
          (sliding_window <= 0) || (kv_idx >= seq_len - sliding_window);
      if (in_window) {
        float acc = 0.0f;
        const half* sK_row = sK + wlane * GQA_DSK;
        const half* sQ_row = sQ + g * 256;
#pragma unroll
        for (int d = 0; d < 256; d += 2) {
          half2 q2 = *reinterpret_cast<const half2*>(&sQ_row[d]);
          half2 k2 = *reinterpret_cast<const half2*>(&sK_row[d]);
          acc = fdot2(q2, k2, acc);
        }
        s_k = acc * scale;
      }
    }
    float mx = s_k;
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
      mx = fmaxf(mx, __shfl_xor(mx, off));
    const float m_new_g = (g < G) ? fmaxf(m_i[g], mx) : -INFINITY;
    const float p_k =
        (g < G && wlane < blk_size && s_k > -INFINITY)
            ? expf(s_k - m_new_g) : 0.0f;
    float sm = p_k;
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) sm += __shfl_xor(sm, off);
    if (wlane == 0 && g < G) {
      sMnew[g] = m_new_g;
      sLnew[g] = sm;
    }
    if (g < G) sP[g * GQA_BC + wlane] = p_k;
    __syncthreads();

    // PV phase: thread t owns output column t for all G heads.
    // Unroll over GQA_MAX_G so o_acc/m_i/l_i stay in registers (dynamic
    // G as a loop bound spills to scratch on gfx1030 and races sP).
    // Skip the online-softmax update when the tile is all-masked:
    // exp(-inf - -inf) is NaN and poisons L_partial (same guard as the
    // per-head kernel).
    if (t < 256) {
#pragma unroll
      for (int gg = 0; gg < GQA_MAX_G; ++gg) {
        if (gg < G) {
          const float m_new = sMnew[gg];
          if (m_new > -INFINITY) {
            const float en = expf(m_i[gg] - m_new);
            float pv = 0.0f;
            for (int k = 0; k < blk_size; ++k)
              pv += sP[gg * GQA_BC + k] * __half2float(sV[k * GQA_DSK + t]);
            o_acc[gg] = en * o_acc[gg] + pv;
            l_i[gg] = en * l_i[gg] + sLnew[gg];
            m_i[gg] = m_new;
          }
        }
      }
    }
    __syncthreads();
  }

  if (t < 256) {
    for (int g = 0; g < G; ++g) {
      const int h_q = h_kv * G + g;
      O_partial[(((int64_t)token_idx * H_q + h_q) * kv_splits + split) * 256 + t]
          = o_acc[g];
    }
  }
  if (t == 0) {
    for (int g = 0; g < G; ++g) {
      const int h_q = h_kv * G + g;
      M_partial[((int64_t)token_idx * H_q + h_q) * kv_splits + split] = m_i[g];
      L_partial[((int64_t)token_idx * H_q + h_q) * kv_splits + split] = l_i[g];
    }
  }
}

// =====================================================================
// DECODE STAGE 2: combine partials across splits
// =====================================================================

__global__ void fa_decode_combine_kernel(
    const float* __restrict__ O_partial,
    const float* __restrict__ M_partial,
    const float* __restrict__ L_partial,
    half* __restrict__ O,
    const int B,
    const int H_q,
    const int kv_splits,
    const int D) {

  const int b = blockIdx.x;
  const int h_q = blockIdx.y;
  const int t = threadIdx.x;
  if (b >= B || h_q >= H_q) return;

  extern __shared__ unsigned char smem_raw[];
  float* sM  = reinterpret_cast<float*>(smem_raw);
  float* sL  = sM + kv_splits;
  float* sWg = sL + kv_splits;

  if (t < kv_splits) {
    sM[t] = M_partial[(b * H_q + h_q) * kv_splits + t];
    sL[t] = L_partial[(b * H_q + h_q) * kv_splits + t];
  }
  __syncthreads();

  float m_global = -INFINITY;
  if (t == 0) {
    for (int s = 0; s < kv_splits; ++s) {
      m_global = fmaxf(m_global, sM[s]);
    }
    float den = 0.0f;
    for (int s = 0; s < kv_splits; ++s) {
      // Empty splits carry m = -inf. When every split is empty (seq_len 0,
      // e.g. CUDA-graph padding rows) expf(-inf - -inf) would be NaN.
      float w = sM[s] == -INFINITY ? 0.0f : expf(sM[s] - m_global);
      sWg[s] = w;
      den += w * sL[s];
    }
    sWg[kv_splits] = den;
  }
  __syncthreads();

  if (t < D) {
    float num = 0.0f;
    for (int s = 0; s < kv_splits; ++s) {
      num += sWg[s] * O_partial[((b * H_q + h_q) * kv_splits + s) * D + t];
    }
    float den = sWg[kv_splits];
    O[(b * H_q + h_q) * D + t] = __float2half_rn(den > 0.0f ? num / den : 0.0f);
  }
}

// =====================================================================
// PREFILL (PAGED, VARLEN): multiple sequences per launch
// =====================================================================
//
// Grid: (max_q_blocks, H_q, num_seqs). CTA (q_block, h_q, seq) handles
// BR_PREFILL consecutive query rows of one sequence. The queries are the
// last seq_query_len positions of the sequence (chunked prefill / prefix
// cache), so row br sits at position
//   (seq_len - seq_query_len) + q_block * BR_PREFILL + br.
// K/V tiles stream from the paged cache through the sequence's block_table;
// tiles no row of the CTA can see are skipped (fa_clip_kv_walk).
// =====================================================================

template <typename KV_T, bool IS_FP8>
__global__ __launch_bounds__(128, 1) void fa_prefill_paged_varlen_kernel_128(
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

  // block_table is [num_seqs, max_blocks] — slice to this sequence.
  const int* seq_block_table = block_table + seq_idx * max_blocks;

  // O and Q have shape [N_q, H_q, D]. Stride: token dim = H_q*D, head dim = D.
  const int stride_qo_tok = H_q * 128;
  const int stride_qo_h = 128;

  if (seq_len <= 0) {
    for (int idx = t; idx < BR_PREFILL * 128; idx += THREADS_PREFILL) {
      const int br = idx / 128;
      const int d = idx % 128;
      if (br < br_size) {
        O[(q_start_global + br) * stride_qo_tok + h_q * stride_qo_h + d] = __float2half(0.0f);
      }
    }
    return;
  }

  extern __shared__ unsigned char smem_raw[];
  half*  sQ  = reinterpret_cast<half*>(smem_raw);
  half*  sK  = sQ + BR_PREFILL * 128;
  half*  sV  = sK + BC * 128;
  float* sP  = reinterpret_cast<float*>(sV + BC * 128);
  float* sM  = sP + BC * BR_PREFILL;
  float* sL  = sM + BR_PREFILL;
  float* sO  = sL + BR_PREFILL;

  // Load Q[Br x D] into shared memory.
  {
    const half* Q_row = Q + (q_start_global * stride_qo_tok + h_q * stride_qo_h);
    for (int i = t; i < BR_PREFILL * 128; i += THREADS_PREFILL) {
      const int br = i / 128;
      const int d = i % 128;
      sQ[i] = (br < br_size) ? Q_row[br * stride_qo_tok + d] : __float2half(0.0f);
    }
  }
  __syncthreads();

  if (t < BR_PREFILL) {
    sM[t] = -INFINITY;
    sL[t] = 0.0f;
  }
  for (int i = t; i < BR_PREFILL * 128; i += THREADS_PREFILL) {
    sO[i] = 0.0f;
  }
  __syncthreads();
