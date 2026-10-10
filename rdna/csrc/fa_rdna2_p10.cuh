  const int kv_group_num = H_q / H_kv;
  TORCH_CHECK(num_tokens > 0, "num_tokens must be > 0");
  TORCH_CHECK(num_seqs > 0, "num_seqs must be > 0");
  TORCH_CHECK(cu_query_lens.size(0) >= (int64_t)num_seqs + 1,
              "cu_query_lens must be at least [num_seqs+1]");

  // Grid: (max_q_blocks_per_seq, H_q, num_seqs). max_q_blocks_per_seq must
  // be large enough for the longest sequence's query blocks. We compute it
  // from the maximum per-sequence query length derived from cu_query_lens
  // and seq_lens (the max is stored implicitly in cu_query_lens[num_seqs]).
  // For simplicity we use ceil(num_tokens / BR_PREFILL) which is an upper
  // bound — some CTAs will early-exit when q_block >= seq_query_len.
  const int max_q_blocks = (num_tokens + BR_PREFILL - 1)
                           / BR_PREFILL;

  if (D == 128) {
    constexpr int HEAD_DIM = 128;
    constexpr int THREADS = 128;
    dim3 grid(max_q_blocks, H_q, num_seqs);
    dim3 block(THREADS);
    size_t smem = BR_PREFILL * HEAD_DIM * sizeof(half)
                + BC * HEAD_DIM * sizeof(half) * 2
                + BC * BR_PREFILL * sizeof(float)
                + BR_PREFILL * sizeof(float) * 3
                + BR_PREFILL * HEAD_DIM * sizeof(float)
                + (THREADS / 32 + 1) * sizeof(float);
    hipFuncSetAttribute(
        reinterpret_cast<const void*>(fa_prefill_paged_varlen_kernel_128<half, false>),
        hipFuncAttributeMaxDynamicSharedMemorySize, smem);
    fa_prefill_paged_varlen_kernel_128<half, false><<<grid, block, smem, stream.stream()>>>(
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
        (half*)out.data_ptr(),
        H_q, H_kv, kv_group_num, scale, (int)causal, (int)sliding_window,
        0.0f, 0.0f);
  } else {
    constexpr int HEAD_DIM = 256;
    constexpr int THREADS = 256;
    constexpr int BC_LOC = BC_256;
    dim3 grid(max_q_blocks, H_q, num_seqs);
    dim3 block(THREADS);
    size_t smem = BR_PREFILL * HEAD_DIM * sizeof(half)
                + BC_LOC * (HEAD_DIM + 8) * sizeof(half) * 2
                + BC_LOC * BR_PREFILL * sizeof(float)
                + BR_PREFILL * sizeof(float) * 3
                + BR_PREFILL * HEAD_DIM * sizeof(float)
                + (THREADS / 32 + 1) * sizeof(float);
    hipFuncSetAttribute(
        reinterpret_cast<const void*>(fa_prefill_paged_varlen_kernel_256<half, false>),
        hipFuncAttributeMaxDynamicSharedMemorySize, smem);
    fa_prefill_paged_varlen_kernel_256<half, false><<<grid, block, smem, stream.stream()>>>(
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
        (half*)out.data_ptr(),
        H_q, H_kv, kv_group_num, scale, (int)causal, (int)sliding_window,
        0.0f, 0.0f);
  }
  hipError_t err = hipGetLastError();
  TORCH_CHECK(err == hipSuccess, "fa_rdna2 paged prefill varlen launch failed: ",
              hipGetErrorString(err));
}

// =====================================================================
// PAGED PREFILL HOST WRAPPER (FP8 KV CACHE)
// =====================================================================
//
// Same prefill kernel as fa_rdna2_prefill_paged_varlen but reads K/V from
// an fp8-e4m3 KV cache (uint8 storage) with inline fp8->fp16 dequant at
// the KV load point. Per-tensor k_scale/v_scale applied in-kernel.
// Only HEAD_DIM 128/256 via the primary varlen kernel; the short and
// split-k variants are fp16-only.
//
torch::Tensor fa_rdna2_prefill_paged_varlen_fp8(
    torch::Tensor Q,
    torch::Tensor key_cache,
    torch::Tensor value_cache,
    torch::Tensor block_table,
    torch::Tensor cu_query_lens,
    torch::Tensor seq_lens,
    int64_t block_size,
    int64_t causal,
    int64_t sliding_window,
    double k_scale,
    double v_scale) {
  TORCH_CHECK(Q.is_cuda() && key_cache.is_cuda() && value_cache.is_cuda(),
              "Q/key_cache/value_cache must be on HIP device");
  TORCH_CHECK(block_table.is_cuda() && cu_query_lens.is_cuda() && seq_lens.is_cuda(),
              "block_table/cu_query_lens/seq_lens must be on HIP device");
  TORCH_CHECK(Q.scalar_type() == torch::kHalf, "Q must be fp16");
  TORCH_CHECK(key_cache.scalar_type() == torch::kUInt8,
              "fp8 key_cache must be uint8");
  TORCH_CHECK(value_cache.scalar_type() == torch::kUInt8,
              "fp8 value_cache must be uint8");
  TORCH_CHECK(block_table.scalar_type() == torch::kInt32, "block_table must be int32");
  TORCH_CHECK(cu_query_lens.scalar_type() == torch::kInt32, "cu_query_lens must be int32");
  TORCH_CHECK(seq_lens.scalar_type() == torch::kInt32, "seq_lens must be int32");
  TORCH_CHECK(Q.dim() == 3, "Q must be [num_tokens, H_q, D]");
  TORCH_CHECK(key_cache.dim() == 5, "key_cache must be 5D");
  TORCH_CHECK(value_cache.dim() == 5, "value_cache must be 5D");
  TORCH_CHECK(Q.size(2) == 128 || Q.size(2) == 256, "D must be 128 or 256");
  TORCH_CHECK(key_cache.size(4) == value_cache.size(4), "x packing must match");
  TORCH_CHECK(key_cache.size(2) * key_cache.size(4) == (int64_t)Q.size(2),
              "D/x * x must equal D");
  TORCH_CHECK(block_table.dim() == 2, "block_table must be [num_seqs, max_blocks]");

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
  auto O = rdna2_persist_zeros(g_pref_O, {num_tokens, H_q, D}, half_opts);

  const int max_q_blocks = (num_tokens + BR_PREFILL - 1)
                           / BR_PREFILL;

  if (D == 128) {
    constexpr int HEAD_DIM = 128;
    constexpr int THREADS = 128;
    dim3 grid(max_q_blocks, H_q, num_seqs);
    dim3 block(THREADS);
    size_t smem = BR_PREFILL * HEAD_DIM * sizeof(half)
                + BC * HEAD_DIM * sizeof(half) * 2
                + BC * BR_PREFILL * sizeof(float)
                + BR_PREFILL * sizeof(float) * 3
                + BR_PREFILL * HEAD_DIM * sizeof(float)
                + (THREADS / 32 + 1) * sizeof(float);
    hipFuncSetAttribute(
        reinterpret_cast<const void*>(fa_prefill_paged_varlen_kernel_128<uint8_t, true>),
        hipFuncAttributeMaxDynamicSharedMemorySize, smem);
    fa_prefill_paged_varlen_kernel_128<uint8_t, true><<<grid, block, smem, stream.stream()>>>(
        (const half*)Q.data_ptr(),
        (const uint8_t*)key_cache.data_ptr(),
        (const uint8_t*)value_cache.data_ptr(),
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
        (half*)O.data_ptr(),
        H_q, H_kv, kv_group_num, scale, (int)causal, (int)sliding_window,
        (float)k_scale, (float)v_scale);
  } else {
    constexpr int HEAD_DIM = 256;
    constexpr int THREADS = 256;
    constexpr int BC_LOC = BC_256;
    dim3 grid(max_q_blocks, H_q, num_seqs);
    dim3 block(THREADS);
    size_t smem = BR_PREFILL * HEAD_DIM * sizeof(half)
                + BC_LOC * (HEAD_DIM + 8) * sizeof(half) * 2
                + BC_LOC * BR_PREFILL * sizeof(float)
                + BR_PREFILL * sizeof(float) * 3
                + BR_PREFILL * HEAD_DIM * sizeof(float)
                + (THREADS / 32 + 1) * sizeof(float);
    hipFuncSetAttribute(
        reinterpret_cast<const void*>(fa_prefill_paged_varlen_kernel_256<uint8_t, true>),
        hipFuncAttributeMaxDynamicSharedMemorySize, smem);
    fa_prefill_paged_varlen_kernel_256<uint8_t, true><<<grid, block, smem, stream.stream()>>>(
        (const half*)Q.data_ptr(),
        (const uint8_t*)key_cache.data_ptr(),
        (const uint8_t*)value_cache.data_ptr(),
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
        (half*)O.data_ptr(),
        H_q, H_kv, kv_group_num, scale, (int)causal, (int)sliding_window,
        (float)k_scale, (float)v_scale);
  }
  hipError_t err = hipGetLastError();
  TORCH_CHECK(err == hipSuccess, "fa_rdna2 paged prefill varlen (fp8) launch failed: ",
              hipGetErrorString(err));
  return O;
}

// Sub-4096 optimized varlen prefill host wrapper (HEAD_DIM=128 only).
// Uses BR_PREFILL=32, THREADS_PREFILL=256 for better grid utilization
// at short sequence lengths. Only valid for D=128; for D=256 the caller
// should use fa_rdna2_prefill_paged_varlen with the >=4096 path.
void fa_rdna2_prefill_paged_varlen_short(
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
  TORCH_CHECK(Q.size(2) == 128, "D must be 128 for short variant");
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
  TORCH_CHECK(H_q % H_kv == 0, "H_q must be divisible by H_kv");
  const int kv_group_num = H_q / H_kv;
  TORCH_CHECK(num_tokens > 0, "num_tokens must be > 0");
  TORCH_CHECK(num_seqs > 0, "num_seqs must be > 0");
  TORCH_CHECK(cu_query_lens.size(0) >= (int64_t)num_seqs + 1,
              "cu_query_lens must be at least [num_seqs+1]");

  constexpr int BR_PREFILL = 32;
  constexpr int HEAD_DIM = 128;
  constexpr int BC = 32;
  constexpr int THREADS = 256;

  const int max_q_blocks = (num_tokens + BR_PREFILL - 1) / BR_PREFILL;
  dim3 grid(max_q_blocks, H_q, num_seqs);
  dim3 block(THREADS);
  size_t smem = BR_PREFILL * HEAD_DIM * sizeof(half)
              + BC * HEAD_DIM * sizeof(half) * 2
              + BC * BR_PREFILL * sizeof(float)
              + BR_PREFILL * sizeof(float) * 3
              + BR_PREFILL * HEAD_DIM * sizeof(float)
              + (THREADS / 32 + 1) * sizeof(float);
  hipFuncSetAttribute(
      reinterpret_cast<const void*>(fa_prefill_paged_varlen_kernel_128_short),
      hipFuncAttributeMaxDynamicSharedMemorySize, smem);
  fa_prefill_paged_varlen_kernel_128_short<<<grid, block, smem, stream.stream()>>>(
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
      (half*)out.data_ptr(),
      H_q, H_kv, kv_group_num, scale, (int)causal, (int)sliding_window);

  hipError_t err = hipGetLastError();
  TORCH_CHECK(err == hipSuccess, "fa_rdna2 paged prefill varlen short launch failed: ",
              hipGetErrorString(err));
}

// Split-K paged prefill varlen host wrapper.
// Partitions the KV sequence across kv_splits CTAs, each producing
// partial O/M/L. A reduction kernel combines them into the final O.
void fa_rdna2_prefill_paged_varlen_splitk(
    torch::Tensor Q,
    torch::Tensor key_cache,
    torch::Tensor value_cache,
    torch::Tensor block_table,
    torch::Tensor cu_query_lens,
    torch::Tensor seq_lens,
    int64_t block_size,
    int64_t causal,
    int64_t kv_splits,
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
  TORCH_CHECK(Q.size(2) == 128 || Q.size(2) == 256, "D must be 128 or 256");
  TORCH_CHECK(key_cache.size(4) == value_cache.size(4), "x packing must match");
  TORCH_CHECK(key_cache.size(2) * key_cache.size(4) == (int64_t)Q.size(2),
              "D/x * x must equal D");
  TORCH_CHECK(block_table.dim() == 2, "block_table must be [num_seqs, max_blocks]");
  TORCH_CHECK(kv_splits >= 1 && kv_splits <= MAX_SPLITS,
              "kv_splits must be in [1, 16]");
  fa_check_io(Q, out);

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
  TORCH_CHECK(num_tokens > 0, "num_tokens must be > 0");
