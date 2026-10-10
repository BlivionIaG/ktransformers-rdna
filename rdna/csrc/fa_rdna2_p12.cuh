      o_acc[j] *= o_scale;
    }
    __syncthreads();

    // P·V: the softmax weight sP[row*BC+k] is hoisted out of the dim loop and
    // reused RDS times, and the V strip is fetched with 16-byte loads.
    if (o_row % BR < br_size) {
      #pragma unroll
      for (int k = 0; k < BC; ++k) {
        if (k < blk_size) {
          const float p = sP[o_row * BC + k];
          #pragma unroll
          for (int seg = 0; seg < RDS / 8; ++seg) {
            const uint4 vseg = *reinterpret_cast<const uint4*>(
                &sV[k * DSK + o_d0 + seg * 8]);
            const half2* hs = reinterpret_cast<const half2*>(&vseg);
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
              const float2 vf = __half22float2(hs[j]);
              o_acc[seg * 8 + 2 * j] = fmaf(p, vf.x, o_acc[seg * 8 + 2 * j]);
              o_acc[seg * 8 + 2 * j + 1] =
                  fmaf(p, vf.y, o_acc[seg * 8 + 2 * j + 1]);
            }
          }
        }
      }
    }
    __syncthreads();
  }

  if (o_row % BR < br_size) {
    const int o_qh = o_row / BR;
    const int o_qr = o_row % BR;
    const float inv_l = 1.0f / l_run;
    const int gt = seq_query_start + q_start_in_seq + o_qr;
    half* o_dst = O + (gt * H_q + q_head_start + o_qh) * HEAD_DIM + o_d0;
    #pragma unroll
    for (int j = 0; j < RDS / 2; ++j) {
      *reinterpret_cast<half2*>(o_dst + 2 * j) =
          __floats2half2_rn(o_acc[2 * j] * inv_l, o_acc[2 * j + 1] * inv_l);
    }
  }
}


template <int HEAD_DIM, int HEADS_PER_CTA, int BR_STEP>
void fa_rdna2_prefill_paged_varlen_gqa_impl(
    torch::Tensor Q,
    torch::Tensor key_cache,
    torch::Tensor value_cache,
    torch::Tensor block_table,
    torch::Tensor cu_query_lens,
    torch::Tensor seq_lens,
    int64_t block_size,
    int64_t causal,
    int64_t sliding_window,
    double scale,
    torch::Tensor out) {
  TORCH_CHECK(Q.is_cuda() && key_cache.is_cuda() && value_cache.is_cuda(),
              "Q/key_cache/value_cache must be on HIP device");
  TORCH_CHECK(block_table.is_cuda() && cu_query_lens.is_cuda() && seq_lens.is_cuda(),
              "block_table/cu_query_lens/seq_lens must be on HIP device");
  TORCH_CHECK(Q.scalar_type() == torch::kHalf, "Q must be fp16");
  TORCH_CHECK(key_cache.scalar_type() == torch::kHalf, "key_cache must be fp16");
  TORCH_CHECK(value_cache.scalar_type() == torch::kHalf, "value_cache must be fp16");
  TORCH_CHECK(block_table.scalar_type() == torch::kInt32, "block_table must be int32");
  TORCH_CHECK(cu_query_lens.scalar_type() == torch::kInt32, "cu_query_lens must be int32");
  TORCH_CHECK(seq_lens.scalar_type() == torch::kInt32, "seq_lens must be int32");
  TORCH_CHECK(Q.dim() == 3, "Q must be [num_tokens, H_q, D]");
  TORCH_CHECK(key_cache.dim() == 5, "key_cache must be 5D");
  TORCH_CHECK(value_cache.dim() == 5, "value_cache must be 5D");
  TORCH_CHECK(Q.size(2) == HEAD_DIM, "GQA prefill: head size mismatch");
  TORCH_CHECK(block_table.dim() == 2, "block_table must be [num_seqs, max_blocks]");
  fa_check_io(Q, out);

  const c10::cuda::OptionalCUDAGuard device_guard(device_of(Q));
  auto stream = c10::cuda::getCurrentCUDAStream();

  const int num_tokens = Q.size(0);
  const int H_q = Q.size(1);
  const int H_kv = key_cache.size(1);
  const int max_blocks = block_table.size(0) > 1
                             ? (int)block_table.stride(0)
                             : (int)block_table.size(1);
  const int x_dim = key_cache.size(4);
  const int num_seqs = seq_lens.size(0);
  TORCH_CHECK(num_tokens > 0 && num_seqs > 0, "num_tokens/num_seqs must be > 0");
  TORCH_CHECK(H_q % H_kv == 0, "H_q must be divisible by H_kv");
  const int kv_group_num = H_q / H_kv;
  TORCH_CHECK(kv_group_num % HEADS_PER_CTA == 0,
              "kv_group_num must be divisible by HEADS_PER_CTA");
  TORCH_CHECK(cu_query_lens.size(0) >= (int64_t)num_seqs + 1,
              "cu_query_lens must be at least [num_seqs+1]");

  constexpr int BC = 16;
  constexpr int THREADS = 256;
  constexpr int DSK = HEAD_DIM + 8;
  const int num_groups = kv_group_num / HEADS_PER_CTA;
  const int max_q_blocks = (num_tokens + BR_STEP - 1) / BR_STEP;
  dim3 grid(max_q_blocks, H_kv * num_groups, num_seqs);
  dim3 block(THREADS);
  // sQ + sK + sV + sP; O and the softmax state live in registers.
  size_t smem = HEADS_PER_CTA * BR_STEP * HEAD_DIM * sizeof(half)
              + BC * DSK * sizeof(half) * 2
              + HEADS_PER_CTA * BR_STEP * BC * sizeof(float);
  hipFuncSetAttribute(
      reinterpret_cast<const void*>(
          fa_prefill_paged_varlen_gqa_kernel<HEAD_DIM, HEADS_PER_CTA,
                                             BR_STEP, half, false>),
      hipFuncAttributeMaxDynamicSharedMemorySize, smem);
  fa_prefill_paged_varlen_gqa_kernel<HEAD_DIM, HEADS_PER_CTA, BR_STEP,
                                     half, false>
      <<<grid, block, smem, stream.stream()>>>(
          (const half*)Q.data_ptr(),
          (const half*)key_cache.data_ptr(),
          (const half*)value_cache.data_ptr(),
          (const int*)block_table.data_ptr(),
          (const int*)cu_query_lens.data_ptr(),
          (const int*)seq_lens.data_ptr(),
          (int)key_cache.stride(0), (int)key_cache.stride(1),
          (int)key_cache.stride(2), (int)key_cache.stride(3), (int)key_cache.stride(4),
          (int)value_cache.stride(0), (int)value_cache.stride(1),
          (int)value_cache.stride(2), (int)value_cache.stride(3),
          (int)value_cache.stride(4),
          max_blocks, (int)block_size, x_dim, num_seqs, (half*)out.data_ptr(),
          H_q, H_kv, kv_group_num, scale, (int)causal, (int)sliding_window,
          0.0f, 0.0f, nullptr, nullptr);
  hipError_t err = hipGetLastError();
  TORCH_CHECK(err == hipSuccess, "fa_rdna2 GQA prefill launch failed: ",
              hipGetErrorString(err));
}


void fa_rdna2_prefill_paged_varlen_gqa(
    torch::Tensor Q,
    torch::Tensor key_cache,
    torch::Tensor value_cache,
    torch::Tensor block_table,
    torch::Tensor cu_query_lens,
    torch::Tensor seq_lens,
    int64_t block_size,
    int64_t causal,
    int64_t sliding_window,
    double scale,
    torch::Tensor out) {
  TORCH_CHECK(Q.dim() == 3 && key_cache.dim() == 5 && key_cache.size(1) > 0,
              "Q must be [num_tokens, H_q, D] and key_cache 5D");
  TORCH_CHECK(Q.size(2) == 128 || Q.size(2) == 256,
              "GQA prefill supports D=128 and D=256");
  // Two q-heads per CTA for even GQA groups, one (16 query rows) otherwise.
  const bool even_group = (Q.size(1) / key_cache.size(1)) % 2 == 0;
  auto run = [&](auto kernel_impl) {
    kernel_impl(Q, key_cache, value_cache, block_table, cu_query_lens,
                seq_lens, block_size, causal, sliding_window, scale, out);
  };
  if (Q.size(2) == 128) {
    even_group ? run(fa_rdna2_prefill_paged_varlen_gqa_impl<128, 2, 8>)
               : run(fa_rdna2_prefill_paged_varlen_gqa_impl<128, 1, 16>);
  } else {
    even_group ? run(fa_rdna2_prefill_paged_varlen_gqa_impl<256, 2, 8>)
               : run(fa_rdna2_prefill_paged_varlen_gqa_impl<256, 1, 16>);
  }
}


// =====================================================================
// PAGED PREFILL HOST WRAPPER (INT8 PER-TOKEN-HEAD KV CACHE)
// =====================================================================
//
// Native int8 prefill kernel (replaces the Python-side dequant in
// rdna_attn.py:492-507). Reads K/V from an int8 per-token-head KV cache,
// dequantizes inline using the per-(token,head) fp32 scale tables, and
// runs the v3 wiki "live contract" pattern in the inner QK loop.
//
// Inputs mirror the fp8 prefill wrapper, but k_scale/v_scale are
// torch::Tensor [num_tokens, H_kv] fp32 (per-token-head layout) instead
// of doubles.
//
torch::Tensor fa_rdna2_prefill_paged_varlen_int8(
    torch::Tensor Q,
    torch::Tensor key_cache,
    torch::Tensor value_cache,
    torch::Tensor block_table,
    torch::Tensor cu_query_lens,
    torch::Tensor seq_lens,
    int64_t block_size,
    int64_t causal,
    int64_t sliding_window,
    int64_t kv_splits,
    torch::Tensor k_scale,
    torch::Tensor v_scale) {
  TORCH_CHECK(Q.is_cuda() && key_cache.is_cuda() && value_cache.is_cuda(),
              "Q/key_cache/value_cache must be on HIP device");
  TORCH_CHECK(block_table.is_cuda() && cu_query_lens.is_cuda() && seq_lens.is_cuda(),
              "block_table/cu_query_lens/seq_lens must be on HIP device");
  TORCH_CHECK(k_scale.is_cuda() && v_scale.is_cuda(),
              "k_scale/v_scale must be on HIP device");
  TORCH_CHECK(Q.scalar_type() == torch::kHalf, "Q must be fp16");
  TORCH_CHECK(key_cache.scalar_type() == torch::kInt8,
              "int8 key_cache must be int8");
  TORCH_CHECK(value_cache.scalar_type() == torch::kInt8,
              "int8 value_cache must be int8");
  TORCH_CHECK(block_table.scalar_type() == torch::kInt32, "block_table must be int32");
  TORCH_CHECK(cu_query_lens.scalar_type() == torch::kInt32, "cu_query_lens must be int32");
  TORCH_CHECK(seq_lens.scalar_type() == torch::kInt32, "seq_lens must be int32");
  TORCH_CHECK(k_scale.scalar_type() == torch::kFloat32 && v_scale.scalar_type() == torch::kFloat32,
              "k_scale/v_scale must be float32");
  TORCH_CHECK(Q.dim() == 3, "Q must be [num_tokens, H_q, D]");
  TORCH_CHECK(key_cache.dim() == 5, "key_cache must be 5D");
  TORCH_CHECK(value_cache.dim() == 5, "value_cache must be 5D");
  TORCH_CHECK(Q.size(2) == 128 || Q.size(2) == 256, "D must be 128 or 256");
  TORCH_CHECK(key_cache.size(4) == value_cache.size(4), "x packing must match");
  TORCH_CHECK(key_cache.size(2) * key_cache.size(4) == (int64_t)Q.size(2),
              "D/x * x must equal D");
  TORCH_CHECK(block_table.dim() == 2, "block_table must be [num_seqs, max_blocks]");
  TORCH_CHECK(kv_splits >= 1 && kv_splits <= MAX_SPLITS,
              "kv_splits must be in [1, 16]");

  const c10::cuda::OptionalCUDAGuard device_guard(device_of(Q));
  auto stream = c10::cuda::getCurrentCUDAStream();

  const int num_tokens = Q.size(0);
  const int H_q = Q.size(1);
  const int D = (int)Q.size(2);
  const int H_kv = key_cache.size(1);
  const int max_blocks = block_table.size(0) > 1
                             ? (int)block_table.stride(0)
                             : (int)block_table.size(1);
  const int x_dim = key_cache.size(4);
  const int num_seqs = seq_lens.size(0);
  TORCH_CHECK(H_q % H_kv == 0, "H_q must be divisible by H_kv");
  const int kv_group_num = H_q / H_kv;
  const float scale = 1.0f / sqrtf((float)D);
  TORCH_CHECK(num_tokens > 0, "num_tokens must be > 0");
  TORCH_CHECK(num_seqs > 0, "num_seqs must be > 0");
  TORCH_CHECK(cu_query_lens.size(0) >= (int64_t)num_seqs + 1,
              "cu_query_lens must be at least [num_seqs+1]");

  auto half_opts = torch::TensorOptions().dtype(torch::kHalf).device(Q.device());
  auto float_opts = torch::TensorOptions().dtype(torch::kFloat32).device(Q.device());
  auto O = rdna2_persist_zeros(g_pref_O, {num_tokens, H_q, D}, half_opts);
  // Partial layout: [N, H_q, BR_PREFILL, kv_splits, D] — the splitk
  // kernel indexes ((q_start_global * H_q + h_q) * BR_PREFILL + br) *
  // kv_splits + split, and the existing reduce kernel reads the same
  // layout. The fp16 splitk wrapper uses [N, H_q, kv_splits, D]
  // (smaller) which corrupts memory; we use the correct larger layout
  // here so the int8 path is safe. Distinct persist slots from fp16
  // split-K so growing int8 cannot reshape the live fp16 buffers.
  auto O_partial = rdna2_persist_zeros(
      g_pref_Op, {num_tokens, H_q, BR_PREFILL, (int)kv_splits, D},
      float_opts);
  auto M_partial = rdna2_persist_zeros(
      g_pref_Mp, {num_tokens, H_q, BR_PREFILL, (int)kv_splits},
      float_opts);
  auto L_partial = rdna2_persist_zeros(
      g_pref_Lp, {num_tokens, H_q, BR_PREFILL, (int)kv_splits},
      float_opts);

  const int max_q_blocks = (num_tokens + BR_PREFILL - 1) / BR_PREFILL;

  if (D == 128) {
    constexpr int HEAD_DIM = 128;
    constexpr int THREADS = 128;
    constexpr int BC_LOC = 64;
    constexpr int BR_PREFILL_LOC = BR_PREFILL;
    dim3 grid(max_q_blocks, H_q, num_seqs * (int)kv_splits);
    dim3 block(THREADS);
    size_t smem = BR_PREFILL_LOC * HEAD_DIM * sizeof(half)
                + BC_LOC * HEAD_DIM * sizeof(int8_t) * 2
                + BC_LOC * BR_PREFILL_LOC * sizeof(float)
                + BR_PREFILL_LOC * sizeof(float) * 3
                + BR_PREFILL_LOC * HEAD_DIM * sizeof(float)
                + BC_LOC * sizeof(float) * 2
                + (THREADS / 32 + 1) * sizeof(float);
    hipFuncSetAttribute(
        reinterpret_cast<const void*>(fa_prefill_paged_varlen_splitk_kernel_int8_128),
        hipFuncAttributeMaxDynamicSharedMemorySize, smem);
    fa_prefill_paged_varlen_splitk_kernel_int8_128<<<grid, block, smem, stream.stream()>>>(
        (const half*)Q.data_ptr(),
        (const int8_t*)key_cache.data_ptr(),
        (const int8_t*)value_cache.data_ptr(),
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
        H_q, H_kv, kv_group_num, scale, (int)causal, (int)sliding_window,
        (const float*)k_scale.data_ptr(),
        (const float*)v_scale.data_ptr());
    dim3 reduce_grid(max_q_blocks, H_q, 1);
    dim3 reduce_block(HEAD_DIM);
    fa_prefill_paged_varlen_splitk_reduce_kernel_128<<<reduce_grid, reduce_block, 0, stream.stream()>>>(
        (const float*)O_partial.data_ptr(),
        (const float*)M_partial.data_ptr(),
        (const float*)L_partial.data_ptr(),
        (half*)O.data_ptr(),
        max_q_blocks, H_q, (int)kv_splits,
        H_q * HEAD_DIM, HEAD_DIM, num_tokens);
  } else {
    constexpr int HEAD_DIM = 256;
    constexpr int THREADS = 256;
    constexpr int BC_LOC = 32;
    constexpr int BR_PREFILL_LOC = BR_PREFILL;
    dim3 grid(max_q_blocks, H_q, num_seqs * (int)kv_splits);
    dim3 block(THREADS);
    size_t smem = BR_PREFILL_LOC * HEAD_DIM * sizeof(half)
                + BC_LOC * HEAD_DIM * sizeof(int8_t) * 2
                + BC_LOC * BR_PREFILL_LOC * sizeof(float)
                + BR_PREFILL_LOC * sizeof(float) * 3
                + BR_PREFILL_LOC * HEAD_DIM * sizeof(float)
                + BC_LOC * sizeof(float) * 2
                + (THREADS / 32 + 1) * sizeof(float);
    hipFuncSetAttribute(
        reinterpret_cast<const void*>(fa_prefill_paged_varlen_splitk_kernel_int8_256),
        hipFuncAttributeMaxDynamicSharedMemorySize, smem);
    fa_prefill_paged_varlen_splitk_kernel_int8_256<<<grid, block, smem, stream.stream()>>>(
        (const half*)Q.data_ptr(),
        (const int8_t*)key_cache.data_ptr(),
        (const int8_t*)value_cache.data_ptr(),
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
        H_q, H_kv, kv_group_num, scale, (int)causal, (int)sliding_window,
        (const float*)k_scale.data_ptr(),
        (const float*)v_scale.data_ptr());
    dim3 reduce_grid(max_q_blocks, H_q, 1);
    dim3 reduce_block(HEAD_DIM);
    fa_prefill_paged_varlen_splitk_reduce_kernel_256<<<reduce_grid, reduce_block, 0, stream.stream()>>>(
        (const float*)O_partial.data_ptr(),
        (const float*)M_partial.data_ptr(),
        (const float*)L_partial.data_ptr(),
        (half*)O.data_ptr(),
        max_q_blocks, H_q, (int)kv_splits,
        H_q * HEAD_DIM, HEAD_DIM, num_tokens);
  }
  hipError_t err = hipGetLastError();
  TORCH_CHECK(err == hipSuccess, "fa_rdna2 paged prefill varlen int8 launch failed: ",
              hipGetErrorString(err));
  return O;
}


// =====================================================================
// INT8 PER-TOKEN-HEAD DECODE HOST WRAPPER
// =====================================================================
//
// Same shape as fa_rdna2_decode_paged_fp8 but reads int8 K/V cache and
// dequantizes inline using per-(token, head) fp32 scales. Dispatches the
// new templated `fa_decode_paged_splitk_kernel<int8_t, false, true>` (D=128)
// and `fa_decode_paged_splitk_kernel_256<int8_t, false, true>` (D=256) —
// same shape as the FP8 template (`__launch_bounds__(128/256, 1)`, BC=64/32
// smem tiles of dequantized fp16 K/V), so the FP8 occupancy-fixed kernel
// is reused as-is for int8 KV.
//
// Layout expected (matches triton_reshape_and_cache_flash_per_token_head_quant):
//   key_cache   : [num_blocks, H_kv, D, block_size, 1] int8 (x_dim=1)
//   value_cache : [num_blocks, H_kv, D, block_size, 1] int8
//   k_scale     : [num_tokens, H_kv] fp32 — per-(token, head) K scale
//   v_scale     : [num_tokens, H_kv] fp32 — per-(token, head) V scale
//
torch::Tensor fa_rdna2_decode_paged_int8(
    torch::Tensor Q,
    torch::Tensor key_cache,
    torch::Tensor value_cache,
    torch::Tensor block_table,
    torch::Tensor seq_lens,
