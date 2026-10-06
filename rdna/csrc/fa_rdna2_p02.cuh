        const int d_sub = i / BC;
        if (n_local < blk_size) {
          const half* kp = reinterpret_cast<const half*>(key_cache)
              + s_blk[n_local] * stride_kc0 + h_kv * stride_kc1
              + d_sub * stride_kc2 + s_slot[n_local] * stride_kc3;
          *reinterpret_cast<uint4*>(&sK[n_local * DSK + d_sub * 8]) =
              *reinterpret_cast<const uint4*>(kp);
        }
      }
      // V is slot-innermost: 8 consecutive slots contiguous per d.
      constexpr int NSG = BC / 8;
      for (int i = t; i < HEAD_DIM_PAGED_128 * NSG; i += 128) {
        const int sg = i % NSG;
        const int d = i / NSG;
        const int n_local = sg * 8;
        if (n_local < blk_size) {
          const half* vp = reinterpret_cast<const half*>(value_cache)
              + s_blk[n_local] * stride_vc0 + h_kv * stride_vc1
              + (d / 8) * stride_vc2 + (d % 8) * stride_vc4
              + s_slot[n_local] * stride_vc3;
          if ((s_slot[n_local] & 7) == 0) {
            const uint4 v4 = *reinterpret_cast<const uint4*>(vp);
            const half* vv = reinterpret_cast<const half*>(&v4);
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
              sV[(n_local + j) * DSK + d] = vv[j];
            }
          } else {
            // Misaligned groups can straddle a block boundary.
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
      for (int i = t; i < BC * HEAD_DIM_PAGED_128; i += 128) {
        const int n_local = i / HEAD_DIM_PAGED_128;
        const int d = i % HEAD_DIM_PAGED_128;
        if (n_local < blk_size) {
          const int n_global = n + n_local;
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
          if constexpr (IS_INT8) {
            const float k_s = k_scale_per_tok[n_global * H_kv + h_kv];
            const float v_s = v_scale_per_tok[n_global * H_kv + h_kv];
            const float kf = (float)*k_ptr * k_s;
            const float vf = (float)*v_ptr * v_s;
            sK[n_local * DSK + d] = __float2half_rn(kf);
            sV[n_local * DSK + d] = __float2half_rn(vf);
          } else {
            sK[n_local * DSK + d] = fa_kv_load<KV_T, IS_FP8>(k_ptr, k_scale);
            sV[n_local * DSK + d] = fa_kv_load<KV_T, IS_FP8>(v_ptr, v_scale);
          }
        }
      }
    }
    __syncthreads();

    // Compute S[k] = Q . K[k]^T * scale for k in [0, blk_size).
    // For paged decode, q_idx is always at the END of the sequence (the
    // current token). So sliding_window mask is:
    // if (seq_len - 1 - kv_idx) >= sliding_window: mask. I.e. kv_idx < seq_len - sliding_window.
    float s_k = -INFINITY;
    if (t < blk_size) {
      const int kv_idx = n + t;
      const bool in_window = (sliding_window <= 0) || (kv_idx >= seq_len - sliding_window);
      if (in_window) {
        float acc = 0.0f;
        const half* sK_row = sK + t * DSK;
        #pragma unroll
        for (int d = 0; d < HEAD_DIM_PAGED_128; d += 2) {
          half2 q2 = *reinterpret_cast<const half2*>(&sQ[d]);
          half2 k2 = *reinterpret_cast<const half2*>(&sK_row[d]);
          acc = fdot2(q2, k2, acc);
        }
        s_k = acc * scale;
      }
    }
    if (t < BC) sP[t] = s_k;
    __syncthreads();

    // Online softmax: block reduce max, then exp + accumulate.
    float s_for_max = (t < BC) ? sP[t] : -INFINITY;
    float m_new = block_reduce_max(s_for_max, sRed);
    m_new = fmaxf(m_i, m_new);  // also fold in previous m_i

    // Skip the online-softmax update when the entire block is masked
    // (causal or sliding window). Without this guard, exp(-INFINITY -
    // (-INFINITY)) = exp(NaN) = NaN corrupts sL and propagates to output.
    // m_new is broadcast across the block so this branch is uniform —
    // one predicated instruction, no extra syncs.
    if (m_new > -INFINITY) {
      // PV dot product: o_acc *= exp(m_i - m_new)
      float alpha = expf(m_i - m_new);
      o_acc *= alpha;

      // l_i update
      float p_k = 0.0f;
      if (t < BC) {
        p_k = (t < blk_size) ? expf(sP[t] - m_new) : 0.0f;
        sP[t] = p_k;
      }
      __syncthreads();
      float l_new = alpha * l_i + block_reduce_sum(p_k, sRed);

      // PV accumulation: o_acc += sum_k sP[k] * sV[k, t]
      // Each thread t owns output dim t.
      float pv = 0.0f;
      if (t < HEAD_DIM_PAGED_128) {
        for (int k = 0; k < blk_size; k++) {
          // sV layout: [blk_size][HEAD_DIM]; thread t accumulates V[k][t].
          pv += sP[k] * __half2float(sV[k * DSK + t]);
        }
      }
      o_acc += pv;

      m_i = m_new;
      l_i = l_new;
    }
    __syncthreads();
  }

  // Write partial outputs.
  if (t < HEAD_DIM_PAGED_128) {
    O_partial[((token_idx * H_q + h_q) * kv_splits + split) * HEAD_DIM_PAGED_128 + t] = o_acc;
  }
  if (t == 0) {
    M_partial[(token_idx * H_q + h_q) * kv_splits + split] = m_i;
    L_partial[(token_idx * H_q + h_q) * kv_splits + split] = l_i;
  }
}

// HEAD_DIM = 256 specialization of the paged decode kernel for Qwen3.5 /
// GDN hybrid models. Same algorithm as the 128 variant but with 256-element
// query/key/value vectors and 256 threads per block. The IS_INT8 path
// mirrors the 128 variant — fused i8->fp16 dequant in the load loop, no
// smem scale table.
template <typename KV_T, bool IS_FP8, bool IS_INT8 = false>
__global__ __launch_bounds__(256)
    __attribute__((amdgpu_waves_per_eu(4, 8))) void fa_decode_paged_splitk_kernel_256(
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

  int seq_len;
  const int seq = fa_decode_token_seq(cu_query_lens, num_seqs, token_idx,
                                      seq_lens, seq_len);
  const int* my_block_table = block_table + seq * max_blocks;
  if (seq_len <= 0) {
    if (t < 256) {
      O_partial[((token_idx * H_q + h_q) * kv_splits + split) * 256 + t] = 0.0f;
    }
    if (t == 0) {
      M_partial[(token_idx * H_q + h_q) * kv_splits + split] = -INFINITY;
      L_partial[(token_idx * H_q + h_q) * kv_splits + split] = 0.0f;
    }
    return;
  }

  // sK/sV rows padded so the vectorized K store (one 16B write per lane
  // across consecutive n_local) does not collapse onto one smem bank quad.
  constexpr int DSK = 256 + 8;
  extern __shared__ unsigned char smem_raw[];
  half*  sQ   = reinterpret_cast<half*>(smem_raw);
  half*  sK   = sQ + 256;
  half*  sV   = sK + BC_256 * DSK;
  float* sP   = reinterpret_cast<float*>(sV + BC_256 * DSK);
  float* sRed = sP + BC_256;
  __shared__ int s_blk[BC_256];
  __shared__ int s_slot[BC_256];

  sQ[t] = Q[(token_idx * H_q + h_q) * 256 + t];
  __syncthreads();

  float m_i = -INFINITY;
  float l_i = 0.0f;
  float o_acc = 0.0f;

  // Same split layout as the HEAD_DIM = 128 kernel, in BC_256 tiles.
  const int kv_lo = sliding_window > 0
                        ? max(0, seq_len - sliding_window) / BC_256 * BC_256
                        : 0;
  const int tokens_per_split =
      ((seq_len - kv_lo + kv_splits - 1) / kv_splits + BC_256 - 1) / BC_256 *
      BC_256;
  const int blk_start = kv_lo + split * tokens_per_split;
  const int blk_end   = min(blk_start + tokens_per_split, seq_len);

  for (int n = blk_start; n < blk_end; n += BC_256) {
    const int blk_size = min(BC_256, blk_end - n);

    // Page mapping once per KV block: kills per-element div/mod and
    // block_table re-reads.
    if (t < BC_256) {
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
      constexpr int NX = 256 / 8;
      for (int i = t; i < BC_256 * NX; i += 256) {
        const int n_local = i % BC_256;
        const int d_sub = i / BC_256;
        if (n_local < blk_size) {
          const half* kp = reinterpret_cast<const half*>(key_cache)
              + s_blk[n_local] * stride_kc0 + h_kv * stride_kc1
              + d_sub * stride_kc2 + s_slot[n_local] * stride_kc3;
          *reinterpret_cast<uint4*>(&sK[n_local * DSK + d_sub * 8]) =
              *reinterpret_cast<const uint4*>(kp);
        }
      }
      // V is slot-innermost: 8 consecutive slots contiguous per d.
      constexpr int NSG = BC_256 / 8;
      for (int i = t; i < 256 * NSG; i += 256) {
        const int sg = i % NSG;
        const int d = i / NSG;
        const int n_local = sg * 8;
        if (n_local < blk_size) {
          const half* vp = reinterpret_cast<const half*>(value_cache)
              + s_blk[n_local] * stride_vc0 + h_kv * stride_vc1
              + (d / 8) * stride_vc2 + (d % 8) * stride_vc4
              + s_slot[n_local] * stride_vc3;
          if ((s_slot[n_local] & 7) == 0) {
            const uint4 v4 = *reinterpret_cast<const uint4*>(vp);
            const half* vv = reinterpret_cast<const half*>(&v4);
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
              sV[(n_local + j) * DSK + d] = vv[j];
            }
          } else {
            // Misaligned groups can straddle a block boundary.
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
          const int n_global = n + n_local;
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
          if constexpr (IS_INT8) {
            const float k_s = k_scale_per_tok[n_global * H_kv + h_kv];
            const float v_s = v_scale_per_tok[n_global * H_kv + h_kv];
            const float kf = (float)*k_ptr * k_s;
            const float vf = (float)*v_ptr * v_s;
            sK[n_local * DSK + d] = __float2half_rn(kf);
            sV[n_local * DSK + d] = __float2half_rn(vf);
          } else {
            sK[n_local * DSK + d] = fa_kv_load<KV_T, IS_FP8>(k_ptr, k_scale);
            sV[n_local * DSK + d] = fa_kv_load<KV_T, IS_FP8>(v_ptr, v_scale);
          }
        }
      }
    }
    __syncthreads();

    // Compute S[k] = Q . K[k]^T * scale for k in [0, blk_size).
    // For paged decode, q_idx is at the END of the sequence.
    // Sliding window mask: kv_idx >= seq_len - sliding_window.
    float s_k = -INFINITY;
    if (t < blk_size) {
      const int kv_idx = n + t;
      const bool in_window = (sliding_window <= 0) || (kv_idx >= seq_len - sliding_window);
      if (in_window) {
        float acc = 0.0f;
        const half* sK_row = sK + t * DSK;
        #pragma unroll
        for (int d = 0; d < 256; d += 2) {
          half2 q2 = *reinterpret_cast<const half2*>(&sQ[d]);
          half2 k2 = *reinterpret_cast<const half2*>(&sK_row[d]);
          acc = fdot2(q2, k2, acc);
        }
        s_k = acc * scale;
      }
    }

    float s_for_max = (t < blk_size) ? s_k : -INFINITY;
    float m_new = block_reduce_max(s_for_max, sRed);
    m_new = fmaxf(m_i, m_new);

    // Skip the online-softmax update when the entire block is masked
    // (causal or sliding window). exp(-INFINITY - (-INFINITY)) = exp(NaN)
    // would otherwise corrupt sL and propagate to output. Uniform branch.
    if (m_new > -INFINITY) {
      float exp_diff = expf(m_i - m_new);

      float p_k = (t < blk_size) ? expf(s_k - m_new) : 0.0f;
      if (t < BC_256) sP[t] = p_k;
      __syncthreads();

      float sum_p = block_reduce_sum(p_k, sRed);
      float l_new = exp_diff * l_i + sum_p;

      if (t < 256) {
        float pv = 0.0f;
        for (int k = 0; k < blk_size; k++) {
            pv += sP[k] * __half2float(sV[k * DSK + t]);
        }
        o_acc = exp_diff * o_acc + pv;
      }

      m_i = m_new;
      l_i = l_new;
    }
    __syncthreads();
  }

  if (t < 256) {
    O_partial[((token_idx * H_q + h_q) * kv_splits + split) * 256 + t] = o_acc;
  }
  if (t == 0) {
    M_partial[(token_idx * H_q + h_q) * kv_splits + split] = m_i;
    L_partial[(token_idx * H_q + h_q) * kv_splits + split] = l_i;
  }
}


// =====================================================================
