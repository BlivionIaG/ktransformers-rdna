      if (row_max > -INFINITY) {
        float new_m = fmaxf(sM[t], row_max);
        float exp_diff = expf(sM[t] - new_m);
        float sum_p = 0.0f;
        for (int k = 0; k < blk_size; ++k) {
          sum_p += expf(sP[t * BC_LOC + k] - new_m);
        }
        sL[t] = exp_diff * sL[t] + sum_p;
        for (int d = 0; d < HEAD_DIM; ++d) {
          sO[t * HEAD_DIM + d] *= exp_diff;
        }
        sM[t] = new_m;
        for (int k = 0; k < blk_size; ++k) {
          sP[t * BC_LOC + k] = expf(sP[t * BC_LOC + k] - new_m);
        }
      } else {
        for (int k = 0; k < blk_size; ++k) {
          sP[t * BC_LOC + k] = 0.0f;
        }
      }
    }
    __syncthreads();

    // ---- PV: scalar int8 reads from sV with per-(token,head) scale. ----
    for (int idx = t; idx < BR_PREFILL_LOC * HEAD_DIM; idx += THREADS_PREFILL_LOC) {
      const int br = idx / HEAD_DIM;
      const int d = idx % HEAD_DIM;
      if (br < br_size) {
        float pv = 0.0f;
        #pragma unroll
        for (int k = 0; k < BC_LOC; ++k) {
          if (k < blk_size) {
            float p_val = sP[br * BC_LOC + k];
            float v_val = (float)sV[k * HEAD_DIM + d] * sVscales[k];
            pv = fmaf(p_val, v_val, pv);
          }
        }
        sO[br * HEAD_DIM + d] += pv;
      }
    }
    __syncthreads();
  }

  // Write partial O (unnormalized), M, L (same layout as fp16 splitk).
  const int partial_base = ((q_start_global * H_q + h_q) * kv_splits) + split_idx;
  // Only real rows (br < br_size) are written (padding rows of the last
  // q_block have no partial-buffer slots allocated for them).
  if (t < br_size) {
    const int br = t;
    const int64_t slot = partial_base + br * (H_q * kv_splits);
    M_partial[slot] = sM[br];
    L_partial[slot] = sL[br];
  }
  for (int br = 0; br < br_size; ++br) {
    const int64_t slot = partial_base + br * (H_q * kv_splits);
    for (int d = t; d < HEAD_DIM; d += THREADS_PREFILL_LOC) {
      O_partial[slot * HEAD_DIM + d] = sO[br * HEAD_DIM + d];
    }
  }
}

// HEAD_DIM = 256 split-K prefill kernel (int8 per-token-head).
__global__ __launch_bounds__(256, 1) void fa_prefill_paged_varlen_splitk_kernel_int8_256(
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

  constexpr int HEAD_DIM = 256;
  constexpr int THREADS_PREFILL_LOC = 256;
  constexpr int BC_LOC = 32;
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

  for (int n = kv_start; n < kv_end; n += BC_LOC) {
    const int blk_size = min(BC_LOC, kv_end - n);

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

    if (t < BR_PREFILL_LOC && t < br_size) {
      float row_max = -INFINITY;
      for (int k = 0; k < blk_size; ++k) {
        row_max = fmaxf(row_max, sP[t * BC_LOC + k]);
      }
      if (row_max > -INFINITY) {
        float new_m = fmaxf(sM[t], row_max);
        float exp_diff = expf(sM[t] - new_m);
        float sum_p = 0.0f;
        for (int k = 0; k < blk_size; ++k) {
          sum_p += expf(sP[t * BC_LOC + k] - new_m);
        }
        sL[t] = exp_diff * sL[t] + sum_p;
        for (int d = 0; d < HEAD_DIM; ++d) {
          sO[t * HEAD_DIM + d] *= exp_diff;
        }
        sM[t] = new_m;
        for (int k = 0; k < blk_size; ++k) {
          sP[t * BC_LOC + k] = expf(sP[t * BC_LOC + k] - new_m);
        }
      } else {
        for (int k = 0; k < blk_size; ++k) {
          sP[t * BC_LOC + k] = 0.0f;
        }
      }
    }
    __syncthreads();

    for (int idx = t; idx < BR_PREFILL_LOC * HEAD_DIM; idx += THREADS_PREFILL_LOC) {
      const int br = idx / HEAD_DIM;
      const int d = idx % HEAD_DIM;
      if (br < br_size) {
        float pv = 0.0f;
        #pragma unroll
        for (int k = 0; k < BC_LOC; ++k) {
          if (k < blk_size) {
            float p_val = sP[br * BC_LOC + k];
            float v_val = (float)sV[k * HEAD_DIM + d] * sVscales[k];
            pv = fmaf(p_val, v_val, pv);
          }
        }
        sO[br * HEAD_DIM + d] += pv;
      }
    }
    __syncthreads();
  }

  const int partial_base = ((q_start_global * H_q + h_q) * kv_splits) + split_idx;
  // Only real rows (br < br_size) are written (padding rows of the last
  // q_block have no partial-buffer slots allocated for them).
  if (t < br_size) {
    const int br = t;
    const int64_t slot = partial_base + br * (H_q * kv_splits);
    M_partial[slot] = sM[br];
    L_partial[slot] = sL[br];
  }
  for (int br = 0; br < br_size; ++br) {
    const int64_t slot = partial_base + br * (H_q * kv_splits);
    for (int d = t; d < HEAD_DIM; d += THREADS_PREFILL_LOC) {
      O_partial[slot * HEAD_DIM + d] = sO[br * HEAD_DIM + d];
    }
  }
}

// =====================================================================
// Public entry point
// =====================================================================

// Grow-only decode workspaces. FULL HIP graphs capture these data_ptrs;
// a per-call torch::zeros would free them when the wrapper returns, and
// eager 16k split-K prefill would recycle the pages → first-token-ok then
// duct on replay (16k c=4 mixed prefill + FULL decode).
namespace {
Rdna2PersistBuf g_dec_O, g_dec_Op, g_dec_Mp, g_dec_Lp;
// Prefill workspaces MUST be distinct from decode persist.
Rdna2PersistBuf g_pref_O, g_pref_Op, g_pref_Mp, g_pref_Lp;
}  // namespace

// Opt-in switch for fa_decode_paged_splitk_gqa_kernel_256 (read once).
static bool fa_gqa_decode_enabled() {
  static const bool enabled = [] {
    const char* v = std::getenv("VLLM_FA_RDNA2_GQA_DECODE");
    return v != nullptr && v[0] == '1';
  }();
  return enabled;
}

// The fp16 entry points write the attention output straight into the
// caller's buffer: no staging tensor and no copy kernel afterwards.
static void fa_check_io(const torch::Tensor& Q, const torch::Tensor& out) {
  TORCH_CHECK(Q.is_contiguous(), "Q must be contiguous [num_tokens, H_q, D]");
  TORCH_CHECK(out.is_cuda() && out.scalar_type() == torch::kHalf &&
                  out.is_contiguous() && out.sizes() == Q.sizes(),
              "out must be a contiguous fp16 tensor shaped like Q");
}

// =====================================================================
// PAGED DECODE HOST WRAPPER
// =====================================================================
//
// Reads K/V from vLLM's paged KV cache (5D layout):
//   key_cache:   [num_blocks, H_kv, D/x, block_size, x]
//   value_cache: [num_blocks, H_kv, D/x, block_size, x]
//   block_table: [num_tokens, max_blocks] (int32)
//   seq_lens:    [num_tokens] (int32) — KV length per query token
//
// Output: O [num_tokens, H_q, D] fp16
void fa_rdna2_decode_paged(
    torch::Tensor Q,
    torch::Tensor key_cache,
    torch::Tensor value_cache,
    torch::Tensor block_table,
    torch::Tensor seq_lens,
    int64_t block_size,
    int64_t kv_splits,
    int64_t sliding_window,
    double scale,
    torch::Tensor out,
    c10::optional<torch::Tensor> cu_query_lens) {
  TORCH_CHECK(Q.is_cuda() && key_cache.is_cuda() && value_cache.is_cuda(),
              "Q/key_cache/value_cache must be on HIP device");
  TORCH_CHECK(block_table.is_cuda() && seq_lens.is_cuda(),
              "block_table and seq_lens must be on HIP device");
  TORCH_CHECK(Q.scalar_type() == torch::kHalf, "Q must be fp16");
  TORCH_CHECK(key_cache.scalar_type() == torch::kHalf, "key_cache must be fp16");
  TORCH_CHECK(value_cache.scalar_type() == torch::kHalf, "value_cache must be fp16");
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
  fa_check_io(Q, out);
  // With cu_query_lens, seq_lens/block_table rows are per sequence and each
  // sequence contributes its last q_len positions as queries.
  const int* cu_ptr = nullptr;
  int num_seqs = 0;
  if (cu_query_lens.has_value() && cu_query_lens->defined()) {
    TORCH_CHECK(cu_query_lens->is_cuda() &&
                    cu_query_lens->scalar_type() == torch::kInt32 &&
                    cu_query_lens->dim() == 1 && cu_query_lens->size(0) >= 2,
                "cu_query_lens must be a [num_seqs + 1] int32 tensor");
    num_seqs = (int)cu_query_lens->size(0) - 1;
    TORCH_CHECK(seq_lens.size(0) >= num_seqs && block_table.size(0) >= num_seqs,
                "seq_lens/block_table need a row per sequence");
    cu_ptr = cu_query_lens->data_ptr<int>();
  }

  const c10::cuda::OptionalCUDAGuard device_guard(device_of(Q));
  auto stream = c10::cuda::getCurrentCUDAStream();
