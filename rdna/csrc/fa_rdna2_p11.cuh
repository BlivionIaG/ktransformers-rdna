  TORCH_CHECK(num_seqs > 0, "num_seqs must be > 0");
  TORCH_CHECK(cu_query_lens.size(0) >= (int64_t)num_seqs + 1,
              "cu_query_lens must be at least [num_seqs+1]");

  auto float_opts = torch::TensorOptions().dtype(torch::kFloat32).device(Q.device());
  // Partial layout: [N, H_q, kv_splits, D] — each query token owns one
  // slot per (head, kv_split). The splitk kernel indexes
  //   ((token_idx * H_q + h_q) * kv_splits + split) * D + t
  // which matches this layout. (The earlier [N, H_q, BR_PREFILL, kv_splits,
  // D] shape allocated BR_PREFILL extra rows per token, wasting
  // BR_PREFILL x memory and OOM-ing at 16k prefill with cudagraphs.)
  auto O_partial = rdna2_persist_empty(
      g_pref_Op, {num_tokens, H_q, (int)kv_splits, D}, float_opts);
  auto M_partial = rdna2_persist_empty(
      g_pref_Mp, {num_tokens, H_q, (int)kv_splits}, float_opts);
  auto L_partial = rdna2_persist_empty(
      g_pref_Lp, {num_tokens, H_q, (int)kv_splits}, float_opts);

  const int max_q_blocks = (num_tokens + BR_PREFILL - 1)
                           / BR_PREFILL;

  if (D == 128) {
    constexpr int HEAD_DIM = 128;
    constexpr int THREADS = 128;
    dim3 grid(max_q_blocks, H_q, num_seqs * (int)kv_splits);
    dim3 block(THREADS);
    size_t smem = BR_PREFILL * HEAD_DIM * sizeof(half)
                + BC * HEAD_DIM * sizeof(half) * 2
                + BC * BR_PREFILL * sizeof(float)
                + BR_PREFILL * sizeof(float) * 3
                + BR_PREFILL * HEAD_DIM * sizeof(float)
                + (THREADS / 32 + 1) * sizeof(float);
    hipFuncSetAttribute(
        reinterpret_cast<const void*>(fa_prefill_paged_varlen_splitk_kernel_128),
        hipFuncAttributeMaxDynamicSharedMemorySize, smem);
    fa_prefill_paged_varlen_splitk_kernel_128<<<grid, block, smem, stream.stream()>>>(
        (const half*)Q.data_ptr(),
        (const half*)key_cache.data_ptr(),
        (const half*)value_cache.data_ptr(),
        (const int*)block_table.data_ptr(),
        (const int*)cu_query_lens.data_ptr(),
        (const int*)seq_lens.data_ptr(),
        (int)key_cache.stride(0),
        (int)key_cache.stride(1),
        (int)key_cache.stride(2),
        (int)key_cache.stride(3),
        (int)key_cache.stride(4),
        (int)value_cache.stride(0),
        (int)value_cache.stride(1),
        (int)value_cache.stride(2),
        (int)value_cache.stride(3),
        (int)value_cache.stride(4),
        max_blocks,
        (int)block_size,
        x_dim,
        num_seqs,
        (int)kv_splits,
        (float*)O_partial.data_ptr(),
        (float*)M_partial.data_ptr(),
        (float*)L_partial.data_ptr(),
        H_q, H_kv, kv_group_num, scale, (int)causal, (int)sliding_window);
    // Reduction: one CTA per (q_block, h_q), 128 threads; loops over BR_PREFILL rows.
    dim3 reduce_grid(max_q_blocks, H_q, 1);
    dim3 reduce_block(HEAD_DIM);
    fa_prefill_paged_varlen_splitk_reduce_kernel_128<<<reduce_grid, reduce_block, 0, stream.stream()>>>(
        (const float*)O_partial.data_ptr(),
        (const float*)M_partial.data_ptr(),
        (const float*)L_partial.data_ptr(),
        (half*)out.data_ptr(),
        max_q_blocks, H_q, (int)kv_splits,
        H_q * HEAD_DIM, HEAD_DIM, num_tokens);
  } else {
    constexpr int HEAD_DIM = 256;
    constexpr int THREADS = 256;
    constexpr int BC_LOC = BC_256;
    dim3 grid(max_q_blocks, H_q, num_seqs * (int)kv_splits);
    dim3 block(THREADS);
    size_t smem = BR_PREFILL * HEAD_DIM * sizeof(half)
                + BC_LOC * (HEAD_DIM + 8) * sizeof(half) * 2
                + BC_LOC * BR_PREFILL * sizeof(float)
                + BR_PREFILL * sizeof(float) * 3
                + BR_PREFILL * HEAD_DIM * sizeof(float)
                + (THREADS / 32 + 1) * sizeof(float);
    hipFuncSetAttribute(
        reinterpret_cast<const void*>(fa_prefill_paged_varlen_splitk_kernel_256),
        hipFuncAttributeMaxDynamicSharedMemorySize, smem);
    fa_prefill_paged_varlen_splitk_kernel_256<<<grid, block, smem, stream.stream()>>>(
        (const half*)Q.data_ptr(),
        (const half*)key_cache.data_ptr(),
        (const half*)value_cache.data_ptr(),
        (const int*)block_table.data_ptr(),
        (const int*)cu_query_lens.data_ptr(),
        (const int*)seq_lens.data_ptr(),
        (int)key_cache.stride(0),
        (int)key_cache.stride(1),
        (int)key_cache.stride(2),
        (int)key_cache.stride(3),
        (int)key_cache.stride(4),
        (int)value_cache.stride(0),
        (int)value_cache.stride(1),
        (int)value_cache.stride(2),
        (int)value_cache.stride(3),
        (int)value_cache.stride(4),
        max_blocks,
        (int)block_size,
        x_dim,
        num_seqs,
        (int)kv_splits,
        (float*)O_partial.data_ptr(),
        (float*)M_partial.data_ptr(),
        (float*)L_partial.data_ptr(),
        H_q, H_kv, kv_group_num, scale, (int)causal, (int)sliding_window);
    dim3 reduce_grid(max_q_blocks, H_q, 1);
    dim3 reduce_block(HEAD_DIM);
    fa_prefill_paged_varlen_splitk_reduce_kernel_256<<<reduce_grid, reduce_block, 0, stream.stream()>>>(
        (const float*)O_partial.data_ptr(),
        (const float*)M_partial.data_ptr(),
        (const float*)L_partial.data_ptr(),
        (half*)out.data_ptr(),
        max_q_blocks, H_q, (int)kv_splits,
        H_q * HEAD_DIM, HEAD_DIM, num_tokens);
  }
  hipError_t err = hipGetLastError();
  TORCH_CHECK(err == hipSuccess, "fa_rdna2 paged prefill varlen splitk launch failed: ",
              hipGetErrorString(err));
}


// =====================================================================
// PAGED PREFILL KERNEL — GQA MULTI-HEAD-PER-CTA (D=128/256)
// =====================================================================
//
// Instantiated with HEADS_PER_CTA=2, BR_STEP=8 for even GQA groups: each CTA
// processes 2 q-heads sharing the same h_kv (at D=256, measured faster than
// 6-heads/CTA and than the per-head varlen kernel at every tested shape).
// Odd groups (and MHA) use HEADS_PER_CTA=1, BR_STEP=16, so a CTA always
// holds 16 query rows.
//
// Each CTA processes HEADS_PER_CTA q-heads that share the same h_kv.
// Grid: (ceil(num_tokens/BR_STEP), H_kv * (kv_group_num/HEADS_PER_CTA), num_seqs)
//
// Per-row flash-attention: per-row m/l/O accumulators, per-row causal
// and sliding-window masks, matching fa_prefill_paged_varlen_kernel_256.
//
template <int HEAD_DIM, int HEADS_PER_CTA, int BR_STEP, typename KV_T,
          bool IS_FP8, bool IS_INT8 = false>
__global__ __launch_bounds__(256, 1)
void fa_prefill_paged_varlen_gqa_kernel(
    const half* __restrict__ Q,
    const KV_T* __restrict__ key_cache,
    const KV_T* __restrict__ value_cache,
    const int* __restrict__ block_table,
    const int* __restrict__ cu_query_lens,
    const int* __restrict__ seq_lens,
    const int stride_kc0, const int stride_kc1,
    const int stride_kc2, const int stride_kc3, const int stride_kc4,
    const int stride_vc0, const int stride_vc1,
    const int stride_vc2, const int stride_vc3, const int stride_vc4,
    const int max_blocks, const int block_size, const int x_dim,
    const int num_seqs, half* __restrict__ O,
    const int H_q, const int H_kv, const int kv_group_num,
    const float scale, const int causal, const int sliding_window,
    const float k_scale, const float v_scale,
    const float* __restrict__ k_scale_per_tok,
    const float* __restrict__ v_scale_per_tok) {

  constexpr int BR = BR_STEP;
  constexpr int BC = 16;
  constexpr int HEADS = HEADS_PER_CTA;
  constexpr int THREADS = 256;
  constexpr int DSK = HEAD_DIM + 8;
  constexpr int NQ8 = HEAD_DIM / 8;

  const int q_block = blockIdx.x;
  const int head_group = blockIdx.y;
  const int seq_idx = blockIdx.z;
  const int t = threadIdx.x;

  const int NUM_GROUPS = kv_group_num / HEADS;
  const int h_kv = head_group / NUM_GROUPS;
  const int group_idx = head_group % NUM_GROUPS;
  const int q_head_start = h_kv * kv_group_num + group_idx * HEADS;

  const int seq_query_start = cu_query_lens[seq_idx];
  const int seq_query_len = cu_query_lens[seq_idx + 1] - seq_query_start;
  const int q_start_in_seq = q_block * BR;
  const int seq_len = seq_lens[seq_idx];

  if (q_start_in_seq >= seq_query_len) return;
  const int br_size = min(BR, seq_query_len - q_start_in_seq);
  if (seq_len <= 0) {
    for (int i = t; i < HEADS * br_size * HEAD_DIM; i += THREADS) {
      const int qh_i = i / (br_size * HEAD_DIM);
      const int qr = (i / HEAD_DIM) % br_size;
      const int gt = seq_query_start + q_start_in_seq + qr;
      O[(gt * H_q + q_head_start + qh_i) * HEAD_DIM + i % HEAD_DIM] =
          __float2half(0.0f);
    }
    return;
  }
  const int q_base_local = (seq_len - seq_query_len) + q_start_in_seq;
  const int* seq_block_table = block_table + seq_idx * max_blocks;

  extern __shared__ unsigned char smem_raw[];
  half*  sQ = reinterpret_cast<half*>(smem_raw);
  half*  sK = sQ + HEADS * BR * HEAD_DIM;
  half*  sV = sK + BC * DSK;
  float* sP = reinterpret_cast<float*>(sV + BC * DSK);

  // The output accumulator lives in registers, not shared memory: thread t
  // owns row t/RP and the RDS consecutive output dims at (t % RP) * RDS.
  // Keeping O out of smem removes the per-K-tile rescale sweep and the
  // read-modify-write accumulate that dominated this kernel's LDS traffic.
  constexpr int ROWS = HEADS * BR;
  constexpr int RP = THREADS / ROWS;
  constexpr int RDS = HEAD_DIM / RP;
  static_assert(RP * ROWS == THREADS,
                "register-O mapping needs exactly one thread per (row, strip)");
  static_assert(RDS % 8 == 0, "register-O strip must be 16-byte loadable");
  const int o_row = t / RP;
  const int o_d0 = (t % RP) * RDS;

  // Thread t also scores key t % BC of row o_row, so the RP lanes that own a
  // row's O strips hold that row's BC scores: the online softmax runs in
  // registers with in-group shuffles, and the running max/sum are replicated
  // across the group (the xor butterfly gives every lane the same value).
  static_assert(ROWS * BC == THREADS && RP == BC && (BC & (BC - 1)) == 0,
                "softmax mapping needs one score per thread, BC lanes per row");
  const int s_k = t % BC;
  const int s_qr = o_row % BR;

  for (int i = t; i < HEADS * BR * NQ8; i += THREADS) {
    const int qh_i = i / (BR * NQ8);
    const int rem = i % (BR * NQ8);
    const int qr = rem / NQ8;
    const int d8 = rem % NQ8;
    half* dst = &sQ[(qh_i * BR + qr) * HEAD_DIM + d8 * 8];
    if (qr < br_size) {
      const int gt = seq_query_start + q_start_in_seq + qr;
      const half* src =
          Q + (gt * H_q + q_head_start + qh_i) * HEAD_DIM + d8 * 8;
      *reinterpret_cast<uint4*>(dst) = *reinterpret_cast<const uint4*>(src);
    } else {
      *reinterpret_cast<uint4*>(dst) = make_uint4(0u, 0u, 0u, 0u);
    }
  }
  __syncthreads();

  float m_run = -INFINITY;
  float l_run = 0.0f;
  float o_acc[RDS];
  #pragma unroll
  for (int j = 0; j < RDS; ++j) {
    o_acc[j] = 0.0f;
  }

  // Stream over the KV tiles this q block can see.
  int kv_lo = 0, kv_hi = seq_len;
  fa_clip_kv_walk(kv_lo, kv_hi, q_base_local, br_size, causal, sliding_window,
                  BC);
  for (int n = kv_lo; n < kv_hi; n += BC) {
    const int blk_size = min(BC, kv_hi - n);

    __shared__ int s_blk[BC];
    __shared__ int s_slot[BC];
    if (t < BC) {
      const int n_global = n + t;
      const bool ok = (t < blk_size);
      s_blk[t] = ok ? seq_block_table[n_global / block_size] : 0;
      s_slot[t] = ok ? (n_global % block_size) : 0;
    }
    __syncthreads();

    const bool kv_vec_ok =
        (sizeof(KV_T) == 2) && (!IS_FP8) && (!IS_INT8)
        && stride_kc4 == 1 && stride_vc3 == 1
        && x_dim == 8 && ((block_size & 7) == 0);
    if (kv_vec_ok) {
      constexpr int NX = HEAD_DIM / 8;
      for (int i = t; i < BC * NX; i += THREADS) {
        const int n_local = i % BC;
        const int d_sub = i / BC;
        if (n_local < blk_size) {
          const half* kp = reinterpret_cast<const half*>(key_cache)
              + s_blk[n_local] * stride_kc0 + h_kv * stride_kc1
              + d_sub * stride_kc2 + s_slot[n_local] * stride_kc3;
          *reinterpret_cast<uint4*>(&sK[n_local * DSK + d_sub * 8]) =
              *reinterpret_cast<const uint4*>(kp);
        }
      }
      constexpr int NSG = BC / 8;
      for (int i = t; i < HEAD_DIM * NSG; i += THREADS) {
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
      for (int i = t; i < BC * HEAD_DIM; i += THREADS) {
        const int n_local = i / HEAD_DIM;
        const int d = i % HEAD_DIM;
        if (n_local < blk_size) {
          const int d_sub = d / x_dim;
          const int x_idx = d % x_dim;
          const KV_T* k_ptr = key_cache
              + s_blk[n_local] * stride_kc0
              + h_kv * stride_kc1
              + d_sub * stride_kc2
              + s_slot[n_local] * stride_kc3
              + x_idx * stride_kc4;
          const KV_T* v_ptr = value_cache
              + s_blk[n_local] * stride_vc0
              + h_kv * stride_vc1
              + d_sub * stride_vc2
              + s_slot[n_local] * stride_vc3
              + x_idx * stride_vc4;
          if constexpr (IS_INT8) {
            const int n_global = n + n_local;
            const float ks = k_scale_per_tok[n_global * H_kv + h_kv];
            const float vs = v_scale_per_tok[n_global * H_kv + h_kv];
            sK[n_local * DSK + d] = __float2half_rn((float)*k_ptr * ks);
            sV[n_local * DSK + d] = __float2half_rn((float)*v_ptr * vs);
          } else {
            sK[n_local * DSK + d] = fa_kv_load<KV_T, IS_FP8>(k_ptr, k_scale);
            sV[n_local * DSK + d] = fa_kv_load<KV_T, IS_FP8>(v_ptr, v_scale);
          }
        }
      }
    }
    __syncthreads();

    float score = -INFINITY;
    if (s_qr < br_size && s_k < blk_size &&
        !fa_masked(q_base_local + s_qr, n + s_k, causal, sliding_window)) {
      const half* sQ_row = sQ + o_row * HEAD_DIM;
      const half* sK_row = sK + s_k * DSK;
      float acc0 = 0.0f;
      float acc1 = 0.0f;
      #pragma unroll
      for (int d = 0; d < HEAD_DIM; d += 8) {
        const uint4 qv = *reinterpret_cast<const uint4*>(&sQ_row[d]);
        const uint4 kv = *reinterpret_cast<const uint4*>(&sK_row[d]);
        const half2* qh = reinterpret_cast<const half2*>(&qv);
        const half2* kh = reinterpret_cast<const half2*>(&kv);
        acc0 = fdot2(qh[0], kh[0], acc0);
        acc1 = fdot2(qh[1], kh[1], acc1);
        acc0 = fdot2(qh[2], kh[2], acc0);
        acc1 = fdot2(qh[3], kh[3], acc1);
      }
      score = (acc0 + acc1) * scale;
    }

    float tile_max = score;
    #pragma unroll
    for (int off = BC / 2; off > 0; off >>= 1) {
      tile_max = fmaxf(tile_max, __shfl_xor(tile_max, off));
    }
    // A row whose keys are all masked in this tile keeps its state (and
    // avoids expf(-inf - -inf) = NaN).
    float p = 0.0f;
    float o_scale = 1.0f;
    if (tile_max > -INFINITY) {
      const float m_new = fmaxf(m_run, tile_max);
      o_scale = (m_run == -INFINITY) ? 0.0f : expf(m_run - m_new);
      p = __expf(score - m_new);
      m_run = m_new;
    }
    float p_sum = p;
    #pragma unroll
    for (int off = BC / 2; off > 0; off >>= 1) {
      p_sum += __shfl_xor(p_sum, off);
    }
    l_run = o_scale * l_run + p_sum;
    sP[o_row * BC + s_k] = p;
    #pragma unroll
    for (int j = 0; j < RDS; ++j) {
