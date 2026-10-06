
  // Stream over the KV tiles this q block can see.
  const int q_first = (seq_len - seq_query_len) + q_start_in_seq;
  int kv_lo = 0, kv_hi = seq_len;
  fa_clip_kv_walk(kv_lo, kv_hi, q_first, br_size, causal, sliding_window, BC);
  for (int n = kv_lo; n < kv_hi; n += BC) {
    const int blk_size = min(BC, kv_hi - n);

    // Cooperative load sK[BC][D] and sV[BC][D] from paged cache.
    // STORAGE: write at swizzled offset to match the QK^T / PV reads below.
    for (int i = t; i < BC * 128; i += THREADS_PREFILL) {
      const int n_local = i / 128;
      const int d = i % 128;
      if (n_local < blk_size) {
        const int n_global = n + n_local;
        const int block_idx = seq_block_table[n_global / block_size];
        const int slot = n_global % block_size;
        const int d_sub = d / x_dim;
        const int x_idx = d % x_dim;
        const KV_T* k_ptr = key_cache
            + block_idx * stride_kc0
            + h_kv * stride_kc1
            + d_sub * stride_kc2
            + slot * stride_kc3
            + x_idx * stride_kc4;
        const KV_T* v_ptr = value_cache
            + block_idx * stride_vc0
            + h_kv * stride_vc1
            + d_sub * stride_vc2
            + slot * stride_vc3
            + x_idx * stride_vc4;
        const int d_swz = fa_swz_d(d, n_local);
        sK[n_local * 128 + d_swz] = fa_kv_load<KV_T, IS_FP8>(k_ptr, k_scale);
        sV[n_local * 128 + d_swz] = fa_kv_load<KV_T, IS_FP8>(v_ptr, v_scale);
      }
    }
    __syncthreads();

    // Compute sP[br, k] = sum_d sQ[br, d] * sK[k, d] * scale.
    for (int idx = t; idx < BR_PREFILL * BC; idx += THREADS_PREFILL) {
      const int br = idx / BC;
      const int k = idx % BC;
      float acc = 0.0f;
      if (br < br_size && k < blk_size) {
        if (!fa_masked(q_first + br, n + k, causal, sliding_window)) {
          const half* sQ_row = sQ + br * 128;
          const half* sK_row = sK + k * 128;
          #pragma unroll
          for (int d = 0; d < 128; d += 2) {
            // Read sK at the SAME swizzled offset it was stored at.
            half2 q2 = *reinterpret_cast<const half2*>(&sQ_row[d]);
            half2 k2 = *reinterpret_cast<const half2*>(&sK_row[fa_swz_d(d, k)]);
            acc = fdot2(q2, k2, acc);
          }
          sP[br * BC + k] = acc * scale;
        } else {
          sP[br * BC + k] = -INFINITY;
        }
      } else {
        sP[br * BC + k] = 0.0f;
      }
    }
    __syncthreads();

    // Online softmax update.
    if (t < BR_PREFILL && t < br_size) {
      float row_max = -INFINITY;
      for (int k = 0; k < blk_size; ++k) {
        row_max = fmaxf(row_max, sP[t * BC + k]);
      }
      // Skip update if all positions in this block are masked (causal or
      // sliding window). Without this guard, exp(-INFINITY - (-INFINITY)) =
      // exp(NaN) = NaN, which corrupts sL and propagates to output.
      if (row_max > -INFINITY) {
        float new_m = fmaxf(sM[t], row_max);
        float exp_diff = expf(sM[t] - new_m);

        float sum_p = 0.0f;
        for (int k = 0; k < blk_size; ++k) {
          sum_p += expf(sP[t * BC + k] - new_m);
        }
        sL[t] = exp_diff * sL[t] + sum_p;

        for (int d = 0; d < 128; ++d) {
          sO[t * 128 + d] *= exp_diff;
        }
        sM[t] = new_m;

        for (int k = 0; k < blk_size; ++k) {
          sP[t * BC + k] = expf(sP[t * BC + k] - new_m);
        }
      } else {
        // All-masked block: zero sP so PV loop contributes nothing.
        for (int k = 0; k < blk_size; ++k) {
          sP[t * BC + k] = 0.0f;
        }
      }
    }
    __syncthreads();

    // PV dot product. Scalar half V reads at swizzled offset to match the
    // K/V storage layout and avoid bank conflicts when lanes read different
    // rows (k) at the same column (d).
    for (int idx = t; idx < BR_PREFILL * 128; idx += THREADS_PREFILL) {
      const int br = idx / 128;
      const int d = idx % 128;
      if (br < br_size) {
        float pv = 0.0f;
        #pragma unroll
        for (int k = 0; k < BC; ++k) {
          if (k < blk_size) {
            float p_val = sP[br * BC + k];
            float v_val = __half2float(sV[k * 128 + fa_swz_d(d, k)]);
            pv = fmaf(p_val, v_val, pv);
          }
        }
        sO[br * 128 + d] += pv;
      }
    }
    __syncthreads();
  }

  // Write output.
  for (int idx = t; idx < BR_PREFILL * 128; idx += THREADS_PREFILL) {
    const int br = idx / 128;
    const int d = idx % 128;
    if (br < br_size) {
      const float inv_l = 1.0f / sL[br];
      half* O_row = O + (q_start_global + br) * stride_qo_tok + h_q * stride_qo_h;
      float final_val = sO[br * 128 + d] * inv_l;
      O_row[d] = __float2half_rn(final_val);
    }
  }
}

// Sub-4096 optimized variant: BR_PREFILL=32, THREADS_PREFILL=256.
// Larger BR (32 vs 16) processes more query tokens per CTA, reducing
// grid overhead. More threads (256 vs 128) improve warp utilization
// for the larger BR. Shared memory usage: ~48 KB (fits 64 KB limit).
// Trade-off: fewer CTAs (1024/32=32 q_blocks for N=1024, so
// 32*H_q=32*16=512 CTAs total for H_q=16). Triton autotunes this
// shape and finds similar configs, so this variant is designed to
// match or beat Triton at N<4096 where the default BR=16 kernel
// underutilizes the 72 CUs of V620.
__global__ __launch_bounds__(256, 1) void fa_prefill_paged_varlen_kernel_128_short(
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
    half* __restrict__ O,
    const int H_q,
    const int H_kv,
    const int kv_group_num,
    const float scale,
    const int causal,
    const int sliding_window) {

  constexpr int BR_PREFILL = 32;
  constexpr int THREADS_PREFILL = 256;
  constexpr int BC = 32;
  constexpr int HEAD_DIM = 128;

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

  const int stride_qo_tok = H_q * HEAD_DIM;
  const int stride_qo_h = HEAD_DIM;

  if (seq_len <= 0) {
    for (int idx = t; idx < BR_PREFILL * HEAD_DIM; idx += THREADS_PREFILL) {
      const int br = idx / HEAD_DIM;
      const int d = idx % HEAD_DIM;
      if (br < br_size) {
        O[(q_start_global + br) * stride_qo_tok + h_q * stride_qo_h + d] = __float2half(0.0f);
      }
    }
    return;
  }

  extern __shared__ unsigned char smem_raw[];
  half*  sQ  = reinterpret_cast<half*>(smem_raw);
  half*  sK  = sQ + BR_PREFILL * HEAD_DIM;
  half*  sV  = sK + BC * HEAD_DIM;
  float* sP  = reinterpret_cast<float*>(sV + BC * HEAD_DIM);
  float* sM  = sP + BC * BR_PREFILL;
  float* sL  = sM + BR_PREFILL;
  float* sO  = sL + BR_PREFILL;

  // Load Q[Br x D] into shared memory. Vectorized half2 loads.
  {
    const half* Q_row = Q + (q_start_global * stride_qo_tok + h_q * stride_qo_h);
    const int N_HALVES = BR_PREFILL * HEAD_DIM;
    for (int i = t; i < N_HALVES / 2; i += THREADS_PREFILL) {
      const int br = (i * 2) / HEAD_DIM;
      const int d = (i * 2) % HEAD_DIM;
      half2 q2;
      if (br < br_size) {
        q2 = *reinterpret_cast<const half2*>(Q_row + br * stride_qo_tok + d);
      } else {
        q2 = __halves2half2(__float2half(0.0f), __float2half(0.0f));
      }
      *reinterpret_cast<half2*>(sQ + i * 2) = q2;
    }
  }
  __syncthreads();

  if (t < BR_PREFILL) {
    sM[t] = -INFINITY;
    sL[t] = 0.0f;
  }
  // Initialize sO (vectorized).
  for (int i = t; i < BR_PREFILL * HEAD_DIM / 2; i += THREADS_PREFILL) {
    *reinterpret_cast<float2*>(sO + i * 2) = make_float2(0.0f, 0.0f);
  }
  __syncthreads();

  // Stream over the KV tiles this q block can see.
  const int q_first = (seq_len - seq_query_len) + q_start_in_seq;
  int kv_lo = 0, kv_hi = seq_len;
  fa_clip_kv_walk(kv_lo, kv_hi, q_first, br_size, causal, sliding_window, BC);
  for (int n = kv_lo; n < kv_hi; n += BC) {
    const int blk_size = min(BC, kv_hi - n);

    // Vectorized half2 K/V loads. Two consecutive d values share the same
    // d_sub when x_dim >= 2, so we can load 4 contiguous bytes per thread.
    // x_dim stride_kc4 = 1 (packed), so d and d+1 are adjacent in memory.
    // STORAGE: write to sK[n_local * HEAD_DIM + fa_swz_d(d, n_local)] so the
    // swizzled index puts each thread's half2 on a different bank.
    for (int i = t; i < (BC * HEAD_DIM) / 2; i += THREADS_PREFILL) {
      const int n_local = (i * 2) / HEAD_DIM;
      const int d = (i * 2) % HEAD_DIM;
      if (n_local < blk_size) {
        const int n_global = n + n_local;
        const int block_idx = seq_block_table[n_global / block_size];
        const int slot = n_global % block_size;
        const int d_sub = d / x_dim;
        const int x_idx = d % x_dim;
        const half* k_ptr = key_cache
            + block_idx * stride_kc0
            + h_kv * stride_kc1
            + d_sub * stride_kc2
            + slot * stride_kc3
            + x_idx * stride_kc4;
        const half* v_ptr = value_cache
            + block_idx * stride_vc0
            + h_kv * stride_vc1
            + d_sub * stride_vc2
            + slot * stride_vc3
            + x_idx * stride_vc4;
        const int d_swz = fa_swz_d(d, n_local);
        *reinterpret_cast<half2*>(sK + n_local * HEAD_DIM + d_swz) =
            *reinterpret_cast<const half2*>(k_ptr);
        // V is stored unpacked (slot-innermost), so consecutive d are NOT
        // contiguous (stride_vc4 = block_size). Scalar loads.
        sV[n_local * HEAD_DIM + d_swz] = *v_ptr;
        sV[n_local * HEAD_DIM + d_swz + 1] = *(v_ptr + stride_vc4);
      }
    }
    __syncthreads();

    // Compute sP[br, k] = sum_d sQ[br, d] * sK[k, d] * scale.
    for (int idx = t; idx < BR_PREFILL * BC; idx += THREADS_PREFILL) {
      const int br = idx / BC;
      const int k = idx % BC;
      float acc = 0.0f;
      if (br < br_size && k < blk_size) {
        if (!fa_masked(q_first + br, n + k, causal, sliding_window)) {
          const half* sQ_row = sQ + br * HEAD_DIM;
          const half* sK_row = sK + k * HEAD_DIM;
          #pragma unroll
          for (int d = 0; d < HEAD_DIM; d += 2) {
            // Read sK at the SAME swizzled offset it was stored at.
            half2 q2 = *reinterpret_cast<const half2*>(&sQ_row[d]);
            half2 k2 = *reinterpret_cast<const half2*>(&sK_row[fa_swz_d(d, k)]);
            acc = fdot2(q2, k2, acc);
          }
          sP[br * BC + k] = acc * scale;
        } else {
          sP[br * BC + k] = -INFINITY;
        }
      } else {
        sP[br * BC + k] = 0.0f;
      }
    }
    __syncthreads();

    // Online softmax update. FA2-style: process BR_PREFILL rows using
    // one wavefront at a time. Each row is reduced via 32-lane butterfly
    // (__shfl_xor, which compiles to DPP v_mov_b32_dpp on RDNA2).
    // Rows are processed sequentially: each iteration uses one wave to
    // reduce one row. This keeps the butterfly correct (all 32 lanes on
    // the same row) while still using only 5 shfl instructions per reduce.
    for (int br = 0; br < BR_PREFILL; ++br) {
      if (t < 32 && br < br_size) {
        const int lane = t;  // 0..31
        float p_val = (lane < blk_size) ? sP[br * BC + lane] : -INFINITY;

        // 32-lane butterfly max reduction (5 steps, all 32 lanes participate).
        float row_max = p_val;
        row_max = fmaxf(row_max, __shfl_xor(row_max, 1));
        row_max = fmaxf(row_max, __shfl_xor(row_max, 2));
        row_max = fmaxf(row_max, __shfl_xor(row_max, 4));
        row_max = fmaxf(row_max, __shfl_xor(row_max, 8));
        row_max = fmaxf(row_max, __shfl_xor(row_max, 16));
        // All 32 lanes now hold row_max.

        if (row_max > -INFINITY) {
          float old_m = sM[br];
          float new_m = fmaxf(old_m, row_max);
          float exp_diff = expf(old_m - new_m);

          float exp_p = (lane < blk_size) ? expf(p_val - new_m) : 0.0f;

          // 32-lane butterfly sum reduction.
          float sum_p = exp_p;
          sum_p += __shfl_xor(sum_p, 1);
          sum_p += __shfl_xor(sum_p, 2);
          sum_p += __shfl_xor(sum_p, 4);
          sum_p += __shfl_xor(sum_p, 8);
          sum_p += __shfl_xor(sum_p, 16);

          // Lane 0 writes sM/sL.
          if (lane == 0) {
            sL[br] = exp_diff * sL[br] + sum_p;
            sM[br] = new_m;
          }
          // Scale sO by exp_diff. All 32 lanes stride through HEAD_DIM.
          // With HEAD_DIM=128 and 32 lanes, each lane handles 4 d-values.
          for (int d = lane; d < HEAD_DIM; d += 32) {
            sO[br * HEAD_DIM + d] *= exp_diff;
          }
          // Write exp_p back to sP for PV loop.
          if (lane < blk_size) {
            sP[br * BC + lane] = exp_p;
          }
        } else {
          // All-masked block: zero sP so PV loop contributes nothing.
          if (lane < blk_size) {
            sP[br * BC + lane] = 0.0f;
          }
        }
      }
    }
    __syncthreads();

    // PV dot product. Vectorized half2 V reads with XOR swizzle to match
    // the K/V storage layout and eliminate bank conflicts when lanes read
    // different rows (k) at the same column (d).
    for (int idx = t; idx < BR_PREFILL * (HEAD_DIM / 2); idx += THREADS_PREFILL) {
      const int br = idx / (HEAD_DIM / 2);
      const int d_pair = idx % (HEAD_DIM / 2);
      const int d = d_pair * 2;
      if (br < br_size) {
        float pv0 = 0.0f;
        float pv1 = 0.0f;
        #pragma unroll
        for (int k = 0; k < BC; ++k) {
          if (k < blk_size) {
            float p_val = sP[br * BC + k];
            half2 v2 = *reinterpret_cast<const half2*>(sV + k * HEAD_DIM + fa_swz_d(d, k));
            float2 v_f = __half22float2(v2);
            pv0 = fmaf(p_val, v_f.x, pv0);
            pv1 = fmaf(p_val, v_f.y, pv1);
          }
        }
        sO[br * HEAD_DIM + d] += pv0;
        sO[br * HEAD_DIM + d + 1] += pv1;
      }
    }
    __syncthreads();
  }
