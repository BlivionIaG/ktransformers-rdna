
  const int num_tokens = Q.size(0);
  const int H_q = Q.size(1);
  const int D = (int)Q.size(2);
  const int H_kv = key_cache.size(1);
  const int max_blocks = block_table.size(0) > 1
                             ? (int)block_table.stride(0)
                             : (int)block_table.size(1);
  const int x_dim = key_cache.size(4);
  TORCH_CHECK(H_q % H_kv == 0, "H_q must be divisible by H_kv");
  const int kv_group_num = H_q / H_kv;

  auto float_opts = torch::TensorOptions().dtype(torch::kFloat32).device(Q.device());

  auto O_partial = rdna2_persist_empty(
      g_dec_Op, {num_tokens, H_q, (int)kv_splits, D}, float_opts);
  auto M_partial = rdna2_persist_empty(
      g_dec_Mp, {num_tokens, H_q, (int)kv_splits}, float_opts);
  auto L_partial = rdna2_persist_empty(
      g_dec_Lp, {num_tokens, H_q, (int)kv_splits}, float_opts);

  dim3 grid1(num_tokens, H_q, (int)kv_splits);

  if (D == 128) {
    constexpr int HEAD_DIM = 128;
    constexpr int THREADS = 128;
    dim3 block1(THREADS);
    size_t smem1 = HEAD_DIM * sizeof(half)
                 + BC * (HEAD_DIM + 8) * sizeof(half) * 2
                 + BC * sizeof(float)
                 + (THREADS / 32 + 1) * sizeof(float);
    hipFuncSetAttribute(
        reinterpret_cast<const void*>(fa_decode_paged_splitk_kernel<half, false, false>),
        hipFuncAttributeMaxDynamicSharedMemorySize, smem1);
    fa_decode_paged_splitk_kernel<half, false, false><<<grid1, block1, smem1, stream.stream()>>>(
        (const half*)Q.data_ptr(),
        (const half*)key_cache.data_ptr(),
        (const half*)value_cache.data_ptr(),
        (const int*)block_table.data_ptr(),
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
        (float*)O_partial.data_ptr(),
        (float*)M_partial.data_ptr(),
        (float*)L_partial.data_ptr(),
        num_tokens, H_q, H_kv,
        (int)kv_splits, kv_group_num, scale, (int)sliding_window,
        0.0f, 0.0f,
        nullptr, nullptr,
        cu_ptr, num_seqs);
  } else if (fa_gqa_decode_enabled() && (int64_t)num_tokens * H_kv >= 4
             && x_dim == 8
             && kv_group_num > 1 && kv_group_num <= GQA_MAX_G
             && (block_size & 7) == 0
             && key_cache.stride(4) == 1 && value_cache.stride(3) == 1) {
    // Off by default since 2026-09-09: with this gate, mixed 16k n_dec=1
    // used the per-head kernel_256 (coherent) while n_dec>=2 used this GQA
    // kernel and produced garbage from the first decode token. Opt back in
    // with VLLM_FA_RDNA2_GQA_DECODE=1 to re-validate it.
    // GQA decode: one CTA per (token, kv-head, split) for the whole group.
    // Requires G = H_q/H_kv in (1, GQA_MAX_G]; G==1 is the per-head kernel.
    dim3 grid_gqa(num_tokens, H_kv, (int)kv_splits);
    dim3 block_gqa(256);
    size_t smem_gqa = GQA_MAX_G * 256 * sizeof(half)
                    + 2 * (size_t)GQA_BC * GQA_DSK * sizeof(half)
                    + GQA_MAX_G * GQA_BC * sizeof(float)
                    + 2 * GQA_MAX_G * sizeof(float);
    hipFuncSetAttribute(
        reinterpret_cast<const void*>(fa_decode_paged_splitk_gqa_kernel_256),
        hipFuncAttributeMaxDynamicSharedMemorySize, smem_gqa);
    fa_decode_paged_splitk_gqa_kernel_256<<<grid_gqa, block_gqa, smem_gqa, stream.stream()>>>(
        (const half*)Q.data_ptr(),
        (const half*)key_cache.data_ptr(),
        (const half*)value_cache.data_ptr(),
        (const int*)block_table.data_ptr(),
        (const int*)seq_lens.data_ptr(),
        (int)key_cache.stride(0),
        (int)key_cache.stride(1),
        (int)key_cache.stride(2),
        (int)key_cache.stride(3),
        (int)value_cache.stride(0),
        (int)value_cache.stride(1),
        (int)value_cache.stride(2),
        (int)value_cache.stride(3),
        (int)value_cache.stride(4),
        max_blocks,
        (int)block_size,
        (float*)O_partial.data_ptr(),
        (float*)M_partial.data_ptr(),
        (float*)L_partial.data_ptr(),
        num_tokens, H_q, H_kv,
        (int)kv_splits, scale, (int)sliding_window,
        cu_ptr, num_seqs);
  } else {
    constexpr int HEAD_DIM = 256;
    constexpr int THREADS = 256;
    constexpr int BC_LOC = BC_256;
    dim3 block1(THREADS);
    size_t smem1 = HEAD_DIM * sizeof(half)
                 + BC_LOC * (HEAD_DIM + 8) * sizeof(half) * 2
                 + BC_LOC * sizeof(float)
                 + (THREADS / 32 + 1) * sizeof(float);
    hipFuncSetAttribute(
        reinterpret_cast<const void*>(fa_decode_paged_splitk_kernel_256<half, false, false>),
        hipFuncAttributeMaxDynamicSharedMemorySize, smem1);
    fa_decode_paged_splitk_kernel_256<half, false, false><<<grid1, block1, smem1, stream.stream()>>>(
        (const half*)Q.data_ptr(),
        (const half*)key_cache.data_ptr(),
        (const half*)value_cache.data_ptr(),
        (const int*)block_table.data_ptr(),
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
        (float*)O_partial.data_ptr(),
        (float*)M_partial.data_ptr(),
        (float*)L_partial.data_ptr(),
        num_tokens, H_q, H_kv,
        (int)kv_splits, kv_group_num, scale, (int)sliding_window,
        0.0f, 0.0f,
        nullptr, nullptr,
        cu_ptr, num_seqs);
  }
  hipError_t err1 = hipGetLastError();
  TORCH_CHECK(err1 == hipSuccess,
              "fa_rdna2 paged splitk launch failed: ",
              hipGetErrorString(err1),
              " (grid=", grid1.x, ",", grid1.y, ",", grid1.z, ")");

  // Combine kernel — same as contiguous case, just `B` → `num_tokens`.
  dim3 grid2(num_tokens, H_q);
  dim3 block2(D);  // one thread per output dim element
  size_t smem2 = (3 * (int)kv_splits + 1) * sizeof(float);
  fa_decode_combine_kernel<<<grid2, block2, smem2, stream.stream()>>>(
      (const float*)O_partial.data_ptr(),
      (const float*)M_partial.data_ptr(),
      (const float*)L_partial.data_ptr(),
      (half*)out.data_ptr(),
      num_tokens, H_q, (int)kv_splits, D);
  hipError_t err2 = hipGetLastError();
  TORCH_CHECK(err2 == hipSuccess, "fa_rdna2 paged combine launch failed: ",
              hipGetErrorString(err2));
}


// =====================================================================
// PAGED DECODE HOST WRAPPER (FP8 KV CACHE)
// =====================================================================
//
// Same decode kernel as fa_rdna2_decode_paged but reads K/V from an
// fp8-e4m3 KV cache (uint8 storage) and dequantizes fp8->fp16 inline at
// the KV load point inside the kernel, applying the per-tensor scales.
// No Python/Triton dequant pass and no memory copy: the fp8 cache is read
// directly and dequantized into the fp16 shared-memory tile.
//
torch::Tensor fa_rdna2_decode_paged_fp8(
    torch::Tensor Q,
    torch::Tensor key_cache,
    torch::Tensor value_cache,
    torch::Tensor block_table,
    torch::Tensor seq_lens,
    int64_t block_size,
    int64_t kv_splits,
    int64_t sliding_window,
    double k_scale,
    double v_scale) {
  TORCH_CHECK(Q.is_cuda() && key_cache.is_cuda() && value_cache.is_cuda(),
              "Q/key_cache/value_cache must be on HIP device");
  TORCH_CHECK(block_table.is_cuda() && seq_lens.is_cuda(),
              "block_table and seq_lens must be on HIP device");
  TORCH_CHECK(Q.scalar_type() == torch::kHalf, "Q must be fp16");
  TORCH_CHECK(key_cache.scalar_type() == torch::kUInt8,
              "fp8 key_cache must be uint8");
  TORCH_CHECK(value_cache.scalar_type() == torch::kUInt8,
              "fp8 value_cache must be uint8");
  TORCH_CHECK(block_table.scalar_type() == torch::kInt32, "block_table must be int32");
  TORCH_CHECK(seq_lens.scalar_type() == torch::kInt32, "seq_lens must be int32");
  TORCH_CHECK(Q.dim() == 3, "Q must be [num_tokens, H_q, D]");
  TORCH_CHECK(key_cache.dim() == 5, "key_cache must be 5D [num_blocks, H_kv, D/x, block_size, x]");
  TORCH_CHECK(value_cache.dim() == 5, "value_cache must be 5D");
  TORCH_CHECK(Q.size(2) == 128 || Q.size(2) == 256,
              "D must be 128 or 256");
  TORCH_CHECK(key_cache.size(4) == value_cache.size(4), "x packing must match");
  TORCH_CHECK(key_cache.size(2) * key_cache.size(4) == (int64_t)Q.size(2),
              "D/x * x must equal D");
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
  TORCH_CHECK(H_q % H_kv == 0, "H_q must be divisible by H_kv");
  const int kv_group_num = H_q / H_kv;
  const float scale = 1.0f / sqrtf((float)D);

  auto float_opts = torch::TensorOptions().dtype(torch::kFloat32).device(Q.device());
  auto half_opts = torch::TensorOptions().dtype(torch::kHalf).device(Q.device());

  auto O_partial = rdna2_persist_zeros(
      g_dec_Op, {num_tokens, H_q, (int)kv_splits, D}, float_opts);
  auto M_partial = rdna2_persist_zeros(
      g_dec_Mp, {num_tokens, H_q, (int)kv_splits}, float_opts);
  auto L_partial = rdna2_persist_zeros(
      g_dec_Lp, {num_tokens, H_q, (int)kv_splits}, float_opts);
  auto O = rdna2_persist_zeros(g_dec_O, {num_tokens, H_q, D}, half_opts);

  dim3 grid1(num_tokens, H_q, (int)kv_splits);
  const float reduction_bytes = (float)((D + D + D) * sizeof(float) + D * sizeof(float) * 2 + D * sizeof(float));
  (void)reduction_bytes;

  if (D == 128) {
    constexpr int HEAD_DIM = 128;
    constexpr int THREADS = 128;
    dim3 block1(THREADS);
    size_t smem1 = HEAD_DIM * sizeof(half)
                 + BC * (HEAD_DIM + 8) * sizeof(half) * 2
                 + BC * sizeof(float)
                 + (THREADS / 32 + 1) * sizeof(float);
    hipFuncSetAttribute(
        reinterpret_cast<const void*>(fa_decode_paged_splitk_kernel<uint8_t, true, false>),
        hipFuncAttributeMaxDynamicSharedMemorySize, smem1);
    fa_decode_paged_splitk_kernel<uint8_t, true, false><<<grid1, block1, smem1, stream.stream()>>>(
        (const half*)Q.data_ptr(),
        (const uint8_t*)key_cache.data_ptr(),
        (const uint8_t*)value_cache.data_ptr(),
        (const int*)block_table.data_ptr(),
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
        (float*)O_partial.data_ptr(),
        (float*)M_partial.data_ptr(),
        (float*)L_partial.data_ptr(),
        num_tokens, H_q, H_kv,
        (int)kv_splits, kv_group_num, scale, (int)sliding_window,
        (float)k_scale, (float)v_scale,
        nullptr, nullptr,
        nullptr, 0);
  } else {
    constexpr int HEAD_DIM = 256;
    constexpr int THREADS = 256;
    constexpr int BC_LOC = BC_256;
    dim3 block1(THREADS);
    size_t smem1 = HEAD_DIM * sizeof(half)
                 + BC_LOC * (HEAD_DIM + 8) * sizeof(half) * 2
                 + BC_LOC * sizeof(float)
                 + (THREADS / 32 + 1) * sizeof(float);
    hipFuncSetAttribute(
        reinterpret_cast<const void*>(fa_decode_paged_splitk_kernel_256<uint8_t, true, false>),
        hipFuncAttributeMaxDynamicSharedMemorySize, smem1);
    fa_decode_paged_splitk_kernel_256<uint8_t, true, false><<<grid1, block1, smem1, stream.stream()>>>(
        (const half*)Q.data_ptr(),
        (const uint8_t*)key_cache.data_ptr(),
        (const uint8_t*)value_cache.data_ptr(),
        (const int*)block_table.data_ptr(),
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
        (float*)O_partial.data_ptr(),
        (float*)M_partial.data_ptr(),
        (float*)L_partial.data_ptr(),
        num_tokens, H_q, H_kv,
        (int)kv_splits, kv_group_num, scale, (int)sliding_window,
        (float)k_scale, (float)v_scale,
        nullptr, nullptr,
        nullptr, 0);
  }
  hipError_t err1 = hipGetLastError();
  TORCH_CHECK(err1 == hipSuccess,
              "fa_rdna2 paged splitk (fp8) launch failed: ",
              hipGetErrorString(err1),
              " (grid=", grid1.x, ",", grid1.y, ",", grid1.z, ")");

  dim3 grid2(num_tokens, H_q);
  dim3 block2(D);  // one thread per output dim element
  size_t smem2 = (3 * (int)kv_splits + 1) * sizeof(float);
  fa_decode_combine_kernel<<<grid2, block2, smem2, stream.stream()>>>(
      (const float*)O_partial.data_ptr(),
      (const float*)M_partial.data_ptr(),
      (const float*)L_partial.data_ptr(),
      (half*)O.data_ptr(),
      num_tokens, H_q, (int)kv_splits, D);
  hipError_t err2 = hipGetLastError();
  TORCH_CHECK(err2 == hipSuccess, "fa_rdna2 paged combine (fp8) launch failed: ",
              hipGetErrorString(err2));

  return O;
}


// =====================================================================
// PAGED PREFILL HOST WRAPPER
// =====================================================================
//
// Reads K/V from vLLM's paged KV cache (5D layout):
//   key_cache:   [num_blocks, H_kv, D/x, block_size, x]
//   value_cache: [num_blocks, H_kv, D/x, block_size, x]  (after reinterp)
//   block_table: [num_tokens, max_blocks] (int32) — per-token block indices
//   seq_lens:    [num_tokens] (int32) — KV length per query token
//   Q:           [num_tokens, H_q, D] fp16 query tensor (Br > 1 for prefill)
//
// Output: O [num_tokens, H_q, D] fp16 attention output
//
// Supports HEAD_DIM=128 and HEAD_DIM=256.
//
void fa_rdna2_prefill_paged_varlen(
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
  TORCH_CHECK(Q.size(2) == 128 || Q.size(2) == 256, "D must be 128 or 256");
  TORCH_CHECK(key_cache.size(4) == value_cache.size(4), "x packing must match");
  TORCH_CHECK(key_cache.size(2) * key_cache.size(4) == (int64_t)Q.size(2),
              "D/x * x must equal D");
  TORCH_CHECK(block_table.dim() == 2, "block_table must be [num_seqs, max_blocks]");
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
