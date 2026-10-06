    int64_t block_size,
    int64_t kv_splits,
    int64_t sliding_window,
    torch::Tensor k_scale,
    torch::Tensor v_scale) {
  TORCH_CHECK(Q.is_cuda() && key_cache.is_cuda() && value_cache.is_cuda(),
              "Q/key_cache/value_cache must be on HIP device");
  TORCH_CHECK(Q.scalar_type() == torch::kHalf, "Q must be fp16");
  TORCH_CHECK(key_cache.scalar_type() == torch::kInt8, "int8 key_cache must be int8");
  TORCH_CHECK(value_cache.scalar_type() == torch::kInt8, "int8 value_cache must be int8");
  TORCH_CHECK(k_scale.scalar_type() == torch::kFloat32 && v_scale.scalar_type() == torch::kFloat32,
              "k_scale/v_scale must be float32");
  TORCH_CHECK(k_scale.dim() == 2 && v_scale.dim() == 2,
              "k_scale/v_scale must be 2D [num_tokens, H_kv]");

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

  if (D == 128) {
    constexpr int HEAD_DIM = 128;
    constexpr int THREADS = 128;
    dim3 block1(THREADS);
    size_t smem1 = HEAD_DIM * sizeof(half)
                 + BC * (HEAD_DIM + 8) * sizeof(half) * 2
                 + BC * sizeof(float)
                 + (THREADS / 32 + 1) * sizeof(float);
    hipFuncSetAttribute(
        reinterpret_cast<const void*>(fa_decode_paged_splitk_kernel<int8_t, false, true>),
        hipFuncAttributeMaxDynamicSharedMemorySize, smem1);
    fa_decode_paged_splitk_kernel<int8_t, false, true><<<grid1, block1, smem1, stream.stream()>>>(
        (const half*)Q.data_ptr(),
        (const int8_t*)key_cache.data_ptr(),
        (const int8_t*)value_cache.data_ptr(),
        (const int*)block_table.data_ptr(),
        (const int*)seq_lens.data_ptr(),
        (int)key_cache.stride(0), (int)key_cache.stride(1),
        (int)key_cache.stride(2), (int)key_cache.stride(3), (int)key_cache.stride(4),
        (int)value_cache.stride(0), (int)value_cache.stride(1),
        (int)value_cache.stride(2), (int)value_cache.stride(3), (int)value_cache.stride(4),
        max_blocks, (int)block_size, x_dim,
        (float*)O_partial.data_ptr(),
        (float*)M_partial.data_ptr(),
        (float*)L_partial.data_ptr(),
        num_tokens, H_q, H_kv,
        (int)kv_splits, kv_group_num, scale, (int)sliding_window,
        0.0f, 0.0f,  // scalar scales unused; IS_INT8 path uses per-tok ptrs
        (const float*)k_scale.data_ptr(),
        (const float*)v_scale.data_ptr(),
        nullptr, 0);
  } else if (D == 256) {
    constexpr int HEAD_DIM = 256;
    constexpr int THREADS = 256;
    constexpr int BC_LOC = BC_256;
    dim3 block1(THREADS);
    size_t smem1 = HEAD_DIM * sizeof(half)
                 + BC_LOC * (HEAD_DIM + 8) * sizeof(half) * 2
                 + BC_LOC * sizeof(float)
                 + (THREADS / 32 + 1) * sizeof(float);
    hipFuncSetAttribute(
        reinterpret_cast<const void*>(fa_decode_paged_splitk_kernel_256<int8_t, false, true>),
        hipFuncAttributeMaxDynamicSharedMemorySize, smem1);
    fa_decode_paged_splitk_kernel_256<int8_t, false, true><<<grid1, block1, smem1, stream.stream()>>>(
        (const half*)Q.data_ptr(),
        (const int8_t*)key_cache.data_ptr(),
        (const int8_t*)value_cache.data_ptr(),
        (const int*)block_table.data_ptr(),
        (const int*)seq_lens.data_ptr(),
        (int)key_cache.stride(0), (int)key_cache.stride(1),
        (int)key_cache.stride(2), (int)key_cache.stride(3), (int)key_cache.stride(4),
        (int)value_cache.stride(0), (int)value_cache.stride(1),
        (int)value_cache.stride(2), (int)value_cache.stride(3), (int)value_cache.stride(4),
        max_blocks, (int)block_size, x_dim,
        (float*)O_partial.data_ptr(),
        (float*)M_partial.data_ptr(),
        (float*)L_partial.data_ptr(),
        num_tokens, H_q, H_kv,
        (int)kv_splits, kv_group_num, scale, (int)sliding_window,
        0.0f, 0.0f,
        (const float*)k_scale.data_ptr(),
        (const float*)v_scale.data_ptr(),
        nullptr, 0);
  } else {
    TORCH_CHECK(false, "int8 decode: only HEAD_DIM=128 or 256 supported");
  }

  // Stage 2: combine partials across splits (kernel writes raw o_acc +
  // m_i + l_i, this kernel does the per-split online-softmax rescale and
  // divides by l_total). Must run even for kv_splits=1 because the
  // decode kernel writes unnormalized o_acc. Mirrors the FP8/fp16 path.
  dim3 grid2(num_tokens, H_q);
  dim3 block2(D);
  size_t smem2 = (3 * (int)kv_splits + 1) * sizeof(float);
  fa_decode_combine_kernel<<<grid2, block2, smem2, stream.stream()>>>(
      (const float*)O_partial.data_ptr(),
      (const float*)M_partial.data_ptr(),
      (const float*)L_partial.data_ptr(),
      (half*)O.data_ptr(),
      num_tokens, H_q, (int)kv_splits, D);
  hipError_t err = hipGetLastError();
  TORCH_CHECK(err == hipSuccess, "fa_rdna2 int8 decode combine launch failed: ",
              hipGetErrorString(err));
  return O;
}


// =====================================================================
// INT8 PER-(TOKEN, HEAD) KV-CACHE WRITER (reshape_and_cache_int8_rdna2)
// =====================================================================
//
// Symmetric signed int8 quantize + write. Layout (matches RDNA_ATTN
// `get_kv_cache_shape` for cache_dtype_str == "int8_per_token_head"):
//   kv_cache: [2, num_blocks, H_kv, D + 4, block_size] int8
//     - kv_cache[0, b, h, 0..D, s]      : K int8 bytes (packed)
//     - kv_cache[0, b, h, D..D + 4, s]  : K fp32 scale bytes (raw LE)
//     - kv_cache[1, ...]                : V same
//
// One CTA per (token, head). Block size = HEAD_DIM threads (matches the
// decode kernel's warp32 layout). Per-(token, head) absmax reduction via
// warp_reduce_max + block_reduce_max (same helpers as the decode kernel).
//
// Scale = max(absmax / 127, 1e-6) per the kv-int8.md wiki contract.
// Quantize: q[d] = round(K[d] / scale), clamp [-127, 127].
// Store one int8 per D-element per slot. Scale stored as 4 LE bytes
// (raw fp32 bits) at position D per slot.
//
// No atomic add needed — each (token, head) pair is written by exactly
// one CTA so writes are race-free.
//
template <int HEAD_DIM>
__global__ __launch_bounds__(HEAD_DIM, 4) void reshape_and_cache_int8_rdna2_kernel(
    const half* __restrict__ key,         // [num_tokens, H_kv, D]
    const half* __restrict__ value,       // [num_tokens, H_kv, D]
    int8_t* __restrict__ key_cache,       // base of K cache [num_blocks, H_kv, D+4, block_size]
    int8_t* __restrict__ value_cache,     // base of V cache [num_blocks, H_kv, D+4, block_size]
    const int* __restrict__ slot_mapping, // [num_tokens]
    const int num_tokens,
    const int H_kv,
    const int block_size) {
  const int token_idx = blockIdx.x;
  const int h_kv = blockIdx.y;
  const int d = threadIdx.x;

  // Slot for this token (-1 sentinel = skip).
  const int slot = slot_mapping[token_idx];
  if (slot < 0) return;
  if (token_idx >= num_tokens || h_kv >= H_kv) return;

  const int block_idx = slot / block_size;
  const int slot_in_block = slot % block_size;

  // Per-(token, head) absmax via warp+block reduce.
  __shared__ float shared[10];
  __shared__ float s_k_scale_sh;
  __shared__ float s_v_scale_sh;

  const half* k_row = key + (token_idx * H_kv + h_kv) * HEAD_DIM;
  const half* v_row = value + (token_idx * H_kv + h_kv) * HEAD_DIM;

  float k_x = (d < HEAD_DIM) ? __half2float(k_row[d]) : 0.0f;
  float v_x = (d < HEAD_DIM) ? __half2float(v_row[d]) : 0.0f;
  float k_amax_t = fabsf(k_x);
  float v_amax_t = fabsf(v_x);

  k_amax_t = warp_reduce_max(k_amax_t);
  v_amax_t = warp_reduce_max(v_amax_t);
  k_amax_t = block_reduce_max(k_amax_t, shared);
  v_amax_t = block_reduce_max(v_amax_t, shared);

  if (d == 0) {
    s_k_scale_sh = fmaxf(k_amax_t / 127.0f, 1e-6f);
    s_v_scale_sh = fmaxf(v_amax_t / 127.0f, 1e-6f);
  }
  __syncthreads();

  const float k_inv = 1.0f / s_k_scale_sh;
  const float v_inv = 1.0f / s_v_scale_sh;

  // Contiguous-layout strides for [num_blocks, H_kv, D+4, block_size] int8.
  //   stride_block = H_kv * (D+4) * block_size
  //   stride_h     = (D+4) * block_size
  //   stride_d     = block_size  (one slot at a time)
  //   stride_s     = 1           (innermost)
  const int row_bytes = (HEAD_DIM + 4) * block_size;
  int8_t* k_dst = key_cache + block_idx * (H_kv * row_bytes) + h_kv * row_bytes;
  int8_t* v_dst = value_cache + block_idx * (H_kv * row_bytes) + h_kv * row_bytes;

  // Quantize + write one int8 per thread (D bytes per slot).
  if (d < HEAD_DIM) {
    const float kq = roundf(k_x * k_inv);
    const float vq = roundf(v_x * v_inv);
    const int8_t kb = (int8_t)max(-127, min(127, (int)kq));
    const int8_t vb = (int8_t)max(-127, min(127, (int)vq));
    k_dst[d * block_size + slot_in_block] = kb;
    v_dst[d * block_size + slot_in_block] = vb;
  }

  // Scale bytes (raw fp32 LE) — 4 threads (d=0..3) each write 1 byte.
  // Per the wiki, "Pack store as int (4×i8)" — equivalent to storing the
  // 4 raw fp32 bytes at consecutive int8 positions.
  if (d < 4) {
    const int32_t k_bits = __float_as_int(s_k_scale_sh);
    const int32_t v_bits = __float_as_int(s_v_scale_sh);
    k_dst[HEAD_DIM * block_size + slot_in_block + d] =
        (int8_t)((k_bits >> (d * 8)) & 0xFF);
    v_dst[HEAD_DIM * block_size + slot_in_block + d] =
        (int8_t)((v_bits >> (d * 8)) & 0xFF);
  }
}

// Host wrapper: quantizes fp16 K/V to int8 with per-(token, head) scales
// and writes them into the interleaved kv_cache (D bytes data + 4 bytes
// scale per slot, per the kv-int8.md wiki contract).
//
// Inputs:
//   key            : [num_tokens, H_kv, D] fp16
//   value          : [num_tokens, H_kv, D] fp16
//   kv_cache       : [2, num_blocks, H_kv, D + 4, block_size] int8
//                    (in-place write — K cache at slice 0, V at slice 1)
//   slot_mapping   : [num_tokens] int32 — global slot per token (-1 = skip)
//
void reshape_and_cache_int8_rdna2(
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor kv_cache,
    torch::Tensor slot_mapping) {
  TORCH_CHECK(key.is_cuda() && value.is_cuda() && kv_cache.is_cuda() && slot_mapping.is_cuda(),
              "key/value/kv_cache/slot_mapping must be on HIP device");
  TORCH_CHECK(key.scalar_type() == torch::kHalf && value.scalar_type() == torch::kHalf,
              "key/value must be fp16");
  TORCH_CHECK(kv_cache.scalar_type() == torch::kInt8,
              "kv_cache must be int8");
  TORCH_CHECK(slot_mapping.scalar_type() == torch::kInt32,
              "slot_mapping must be int32");
  TORCH_CHECK(key.dim() == 3 && value.dim() == 3,
              "key/value must be [num_tokens, H_kv, D]");
  TORCH_CHECK(kv_cache.dim() == 5 && kv_cache.size(0) == 2,
              "kv_cache must be [2, num_blocks, H_kv, D + 4, block_size]");
  TORCH_CHECK(slot_mapping.dim() == 1, "slot_mapping must be [num_tokens]");

  const int num_tokens = (int)key.size(0);
  const int H_kv = (int)key.size(1);
  const int D = (int)key.size(2);
  TORCH_CHECK(value.size(0) == num_tokens && value.size(1) == H_kv &&
              value.size(2) == D, "key/value shape mismatch");
  TORCH_CHECK(kv_cache.size(2) == H_kv, "H_kv mismatch");
  TORCH_CHECK(kv_cache.size(3) == D + 4,
              "kv_cache D dim must be D + 4 (interleaved scale bytes)");
  TORCH_CHECK(kv_cache.size(1) > 0, "num_blocks must be > 0");
  TORCH_CHECK(D == 128 || D == 256,
              "reshape_and_cache_int8_rdna2: only HEAD_DIM=128 or 256 supported");

  const int block_size = (int)kv_cache.size(4);
  TORCH_CHECK(block_size % 4 == 0,
              "block_size must be a multiple of 4 (4-byte scale alignment)");

  const c10::cuda::OptionalCUDAGuard device_guard(device_of(key));
  auto stream = c10::cuda::getCurrentCUDAStream();

  int8_t* k_cache_ptr = (int8_t*)kv_cache.data_ptr();  // slice 0 base
  // Slice 1 base = base + 1 * stride(0). The slice 0/1 separation is the
  // outermost kv dim; stride(0) is num_blocks * H_kv * (D+4) * block_size.
  int8_t* v_cache_ptr = k_cache_ptr + kv_cache.stride(0);

  dim3 grid(num_tokens, H_kv);
  if (D == 128) {
    reshape_and_cache_int8_rdna2_kernel<128><<<grid, 128, 0, stream.stream()>>>(
        (const half*)key.data_ptr(),
        (const half*)value.data_ptr(),
        k_cache_ptr, v_cache_ptr,
        (const int*)slot_mapping.data_ptr(),
        num_tokens, H_kv, block_size);
  } else {
    reshape_and_cache_int8_rdna2_kernel<256><<<grid, 256, 0, stream.stream()>>>(
        (const half*)key.data_ptr(),
        (const half*)value.data_ptr(),
        k_cache_ptr, v_cache_ptr,
        (const int*)slot_mapping.data_ptr(),
        num_tokens, H_kv, block_size);
  }
  hipError_t err = hipGetLastError();
  TORCH_CHECK(err == hipSuccess, "reshape_and_cache_int8_rdna2 launch failed: ",
              hipGetErrorString(err));
}

// =====================================================================
// FP16 FLASH KV-CACHE WRITER (reshape_and_cache_flash_rdna2)
// =====================================================================
//
// Stride-aware port of triton_reshape_and_cache_flash for FA-RDNA2:
//   K 5D: [num_blocks, H_kv, D/x, block_size, x]  (x-innermost packed)
//   V 4D: [num_blocks, H_kv, D, block_size]       (slot-innermost)
// Hybrid GDN pages pad stride(0) past packed numel; using the tensor's
// real strides (not packed H*D*bs) is what keeps writes in-page.
//
// One CTA per token, 128 threads walk H*D. slot_mapping[t] < 0 = skip.

template <typename SlotT>
__global__ __launch_bounds__(128, 4) void reshape_and_cache_flash_rdna2_kernel(
    const half* __restrict__ key,
    const half* __restrict__ value,
    half* __restrict__ key_cache,
    half* __restrict__ value_cache,
    const SlotT* __restrict__ slot_mapping,
    int num_tokens,
    int H,
    int D,
    int block_size,
    int x,
    int64_t key_stride,
    int64_t value_stride,
    int64_t k_s0, int64_t k_s1, int64_t k_s2, int64_t k_s3, int64_t k_s4,
    int64_t v_s0, int64_t v_s1, int64_t v_s2, int64_t v_s3,
    int64_t num_blocks) {
  const int token = blockIdx.x;
  if (token >= num_tokens) {
    return;
  }
  const int64_t slot = static_cast<int64_t>(slot_mapping[token]);
  if (slot < 0) {
    return;
  }
  const int64_t block_idx = slot / block_size;
  const int64_t block_off = slot % block_size;
  if (block_idx < 0 || block_idx >= num_blocks) {
    return;
  }
  const int n = H * D;
  const half* ksrc = key + static_cast<int64_t>(token) * key_stride;
  const half* vsrc = value + static_cast<int64_t>(token) * value_stride;
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    const int h = i / D;
    const int d = i % D;
    const int64_t k_idx = block_idx * k_s0 + static_cast<int64_t>(h) * k_s1 +
                          static_cast<int64_t>(d / x) * k_s2 +
                          block_off * k_s3 + static_cast<int64_t>(d % x) * k_s4;
    const int64_t v_idx = block_idx * v_s0 + static_cast<int64_t>(h) * v_s1 +
                          static_cast<int64_t>(d) * v_s2 + block_off * v_s3;
    key_cache[k_idx] = ksrc[i];
    value_cache[v_idx] = vsrc[i];
  }
}

void reshape_and_cache_flash_rdna2(
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor key_cache,
    torch::Tensor value_cache,
    torch::Tensor slot_mapping) {
  TORCH_CHECK(key.is_cuda() && value.is_cuda() && key_cache.is_cuda() &&
                  value_cache.is_cuda() && slot_mapping.is_cuda(),
              "reshape_and_cache_flash_rdna2: all tensors must be on HIP");
  TORCH_CHECK(key.scalar_type() == torch::kHalf &&
                  value.scalar_type() == torch::kHalf &&
                  key_cache.scalar_type() == torch::kHalf &&
                  value_cache.scalar_type() == torch::kHalf,
              "reshape_and_cache_flash_rdna2: fp16 only");
  TORCH_CHECK(key.dim() == 3 && value.dim() == 3,
              "key/value must be [num_tokens, H_kv, D]");
  TORCH_CHECK(key_cache.dim() == 5,
              "key_cache must be 5D [nb, H, D/x, bs, x]");
  TORCH_CHECK(value_cache.dim() == 4,
              "value_cache must be 4D [nb, H, D, bs]");
  TORCH_CHECK(slot_mapping.dim() == 1, "slot_mapping must be 1D");
  TORCH_CHECK(slot_mapping.scalar_type() == torch::kInt ||
                  slot_mapping.scalar_type() == torch::kLong,
              "slot_mapping must be int32 or int64");

  const int num_tokens = static_cast<int>(
      std::min(slot_mapping.size(0), key.size(0)));
  const int H = static_cast<int>(key.size(1));
  const int D = static_cast<int>(key.size(2));
  const int x = static_cast<int>(key_cache.size(4));
  const int block_size = static_cast<int>(key_cache.size(3));
  TORCH_CHECK(x > 0 && D % x == 0, "head_size must be divisible by x");
  TORCH_CHECK(key_cache.size(1) == H && key_cache.size(2) == D / x,
