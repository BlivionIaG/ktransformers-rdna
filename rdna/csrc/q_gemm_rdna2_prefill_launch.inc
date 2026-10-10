
// ---------------------------------------------------------------------------
// Config dispatch and launcher.
//
// select_config(M, N):
//   - M < 64    -> ConfigV1 (small-M corner where v1 wins by a wide margin).
//   - N <= 1024 -> ConfigC  (small-N tile, keeps multiple N blocks per CU).
//   - otherwise -> ConfigA  (general prefill tile, wins M >= 96 large N).
// ---------------------------------------------------------------------------
enum ConfigId : int { ConfigId_V1 = 0, ConfigId_A = 1, ConfigId_C = 3 };

template <typename Config>
inline void launch_for_config(
    const half* a, const uint32_t* b_q_weight, const uint32_t* b_qzeros,
    const half* b_scales, const int* b_q_perm, half* c, int size_m,
    int size_n, int size_k, int groups, int split_k, bool use_v2_format,
    cudaStream_t stream) {
  constexpr int N_TILE = Config::N_TILE;
  constexpr int M_TILE = Config::M_TILE;
  constexpr int LDS_PAD = Config::LDS_PAD;

  const int zero_offset = use_v2_format ? 0 : 1;
  dim3 block(Config::THREADS);
  dim3 grid((size_n + N_TILE - 1) / N_TILE, (size_m + M_TILE - 1) / M_TILE,
            split_k);
  const int k_per_split = size_k / split_k;
  const int lds_k_stride = ((k_per_split + LDS_PAD + 7) / 8) * 8;
  const size_t shmem = M_TILE * lds_k_stride * sizeof(half);

  switch (k_per_split) {
    case 256:
      gemm_static_kernel<Config, 256>
          <<<grid, block, shmem, stream>>>(a, b_q_weight, b_qzeros, b_scales, c,
                                           size_m, size_n, size_k, groups,
                                           zero_offset, b_q_perm, split_k);
      break;
    case 512:
      gemm_static_kernel<Config, 512>
          <<<grid, block, shmem, stream>>>(a, b_q_weight, b_qzeros, b_scales, c,
                                           size_m, size_n, size_k, groups,
                                           zero_offset, b_q_perm, split_k);
      break;
    case 1024:
      gemm_static_kernel<Config, 1024>
          <<<grid, block, shmem, stream>>>(a, b_q_weight, b_qzeros, b_scales, c,
                                           size_m, size_n, size_k, groups,
                                           zero_offset, b_q_perm, split_k);
      break;
    default:
      gemm_dynamic_kernel<Config>
          <<<grid, block, shmem, stream>>>(a, b_q_weight, b_qzeros, b_scales, c,
                                           size_m, size_n, size_k, groups,
                                           zero_offset, b_q_perm, split_k);
      break;
  }
}

// Split-K search.  Mirrors v1's LDS-budget heuristic and adds a third tier
// (64 KiB) when the problem is medium-sized, matching gfx1030's full local
// memory capacity.
template <typename Config>
int compute_split_k(int size_m, int size_n, int size_k) {
  constexpr int N_TILE = Config::N_TILE;
  constexpr int M_TILE = Config::M_TILE;
  constexpr int LDS_PAD = Config::LDS_PAD;
  constexpr int K_STEP = Config::K_STEP;

  const int blocks_per_k_split =
      ((size_m + M_TILE - 1) / M_TILE) *
      ((size_n + N_TILE - 1) / N_TILE);

  auto lds_bytes = [&](int split) {
    const int k_per_split = size_k / split;
    const int row_stride = ((k_per_split + LDS_PAD + 7) / 8) * 8;
    return M_TILE * row_stride * sizeof(half);
  };
  // 16 KiB when the grid is large (avoid HSA invalid-allocation crashes),
  // 64 KiB at medium scale (use full gfx1030 LDS capacity to cut atomic
  // contention), 32 KiB otherwise.
  const size_t lds_budget =
      (blocks_per_k_split > 1024) ? (16 * 1024)
      : (blocks_per_k_split > 256) ? (64 * 1024)
                                   : (32 * 1024);

  // Repair, don't re-tune: keep the pre-W4A8 powers-of-two search whenever its
  // choice is usable, so shapes that never hit the NaN bug keep their exact
  // split and timing. The legacy choice is unusable only when its
  // k_per_split = size_k/split is not a K_STEP multiple: the kernel advances K
  // in K_STEP-wide chunks and never clamps the final chunk to k_per_split, so
  // the tail walks past the split's LDS row and global K range (observed NaN /
  // garbage at e.g. k=4352 -> split=16 -> k_per_split=272 at ConfigV1).
  const int max_split_k = size_k / K_STEP;
  int legacy = 1;
  while (legacy < max_split_k && lds_bytes(legacy) > lds_budget) {
    legacy *= 2;
  }
  while (legacy < 16 && max_split_k >= legacy * 2 &&
         (blocks_per_k_split * legacy < 2048 ||
          (size_k / legacy) > 2048)) {
    const int candidate = legacy * 2;
    if (lds_bytes(candidate) > lds_budget) break;
    legacy = candidate;
  }
  int result;
  if (size_k % legacy == 0 && (size_k / legacy) % K_STEP == 0) {
    result = legacy;
  } else {
    // Fallback for the misaligned legacy choice only: enumerate every usable
    // split <= 16 rather than only powers of two, so a k with few factors of
    // two (e.g. 4160) still gets a valid split. `size_k % K_STEP == 0` (checked
    // at entry) makes split=1 usable, so the list is never empty.
    int splits[16];
    int count = 0;
    for (int s = 1; s <= 16; ++s) {
      if (size_k % s == 0 && (size_k / s) % K_STEP == 0) {
        splits[count++] = s;
      }
    }

    int i = 0;
    while (i + 1 < count && lds_bytes(splits[i]) > lds_budget) {
      ++i;
    }
    while (i + 1 < count &&
           (blocks_per_k_split * splits[i] < 2048 ||
            (size_k / splits[i]) > 2048)) {
      if (lds_bytes(splits[i + 1]) > lds_budget) {
        break;
      }
      ++i;
    }
    result = splits[i];
  }

  static const bool debug_split =
      std::getenv("VLLM_RDNA2_PREFILL_DEBUG") != nullptr;
  if (debug_split) {
    printf("[rdna2_prefill_split] m=%d n=%d k=%d split=%d\n", size_m, size_n,
           size_k, result);
  }
  return result;
}

// Public dispatcher entry: pick a Config, compute split_k, launch.
// K-aware config selection. Mirrors the empirical 4-branch rule from the
// 3264-cell microbench (M=1..32, K<4096, N=128..16384): ConfigV1 is the safe
// default; ConfigC wins at M=4..8 with high N and K, and at M=12 with mid N
// and any K>=512; ConfigA wins at small M (M<4) with large N. The envelope
// guard (M<=32 AND K<4096) is enforced by the outer dispatcher
// (_rdna2_w4a16_select_kernel) so this function never sees out-of-envelope
// cells; the V1 default covers anything outside the explicit K-gated
// branches.
// K-aware config selection. Three tiers:
//
//   - Small M (M <= 32): original 3264-cell microbench. ConfigV1 is the
//     safe default; ConfigC wins at M=4..8 with high N and K, and at
//     M=12/32 in narrow N windows; ConfigA wins at small M (M<4) with
//     large N.
//
//   - Large M (M > 256): empirical data from 2026-09-10 profile run on
//     Qwen3.8-27B-AWQ-INT4 (gfx1030, TP=2). ConfigV1 (M_TILE=8) is at
//     53% of peak fp16 for M=1856-2048. ConfigA (M_TILE=16) halves the
//     M-blocks and reduces atomic contention per output tile from 16
//     blocks/tile (split_k=16 with M_TILE=8) to 8 blocks/tile (split_k=8
//     with M_TILE=16). ConfigC (N_TILE=512) is for small-N tiles where
//     ConfigA's N_TILE=1024 would over-shard the grid.
//
//   - Envelope guard: the outer dispatcher (rdna2_w4a16.py) routes
//     AWQ/GPTQ prefill to this function only when M > 32. ConfigA is
//     the right choice for Qwen3.8-27B-AWQ high-N shapes (intermediate
//     projection per-rank N=8704, down-projection per-rank N=2560).
inline int select_config(int size_m, int size_n, int size_k) {
  // Debug override: force a specific config for kernel-level bisection.
  static const int force_config = []() {
    const char* e = std::getenv("VLLM_RDNA2_PREFILL_FORCE_CONFIG");
    return e ? std::atoi(e) : -1;
  }();
  if (force_config >= 0) return force_config;

  // Large-M prefill: prefer ConfigA (M_TILE=16, N_TILE=1024) for high N.
  // ConfigC (M_TILE=16, N_TILE=512) for small N. ConfigV1 (M_TILE=8)
  // would need 2x more M-blocks per output tile and 2x more atomic
  // contention under the existing split_k heuristic.
  if (size_m > 256) {
    if (size_n >= 4096) return ConfigId_A;
    return ConfigId_C;
  }
  // Existing small-M logic (unchanged).
  if (size_m < 4 && size_n > 4096) return ConfigId_A;
  if (4 <= size_m && size_m <= 8 && size_n >= 2048 && size_k >= 1024)
    return ConfigId_C;
  if (size_m == 12 && size_n >= 2560 && size_n <= 8192 && size_k >= 512)
    return ConfigId_C;
  if (size_m == 32 && size_n >= 5120 && size_n <= 6144 && size_k >= 1536)
    return ConfigId_C;
  return ConfigId_V1;
}

// Debug override: force split_k for isolating the atomic-epilogue cost.
// split_k divides size_k; the last split absorbs any remainder. The caller
// must ensure the forced split_k keeps k_per_split within the LDS budget —
// this knob is for experiments, not production.
inline int maybe_force_split_k(int computed) {
  static const int force = []() {
    const char* e = std::getenv("VLLM_RDNA2_PREFILL_FORCE_SPLIT_K");
    return e ? std::atoi(e) : -1;
  }();
  return force > 0 ? force : computed;
}

void launch_dispatch(
    const half* a, const uint32_t* b_q_weight, const uint32_t* b_qzeros,
    const half* b_scales, const int* b_q_perm, half* c, int size_m,
    int size_n, int size_k, int groups, bool use_v2_format,
    cudaStream_t stream) {
  switch (select_config(size_m, size_n, size_k)) {
    case ConfigId_V1: {
      const int split_k = maybe_force_split_k(compute_split_k<ConfigV1>(size_m, size_n, size_k));
      launch_for_config<ConfigV1>(a, b_q_weight, b_qzeros, b_scales, b_q_perm, c,
                                  size_m, size_n, size_k, groups, split_k,
                                  use_v2_format, stream);
      break;
    }
    case ConfigId_C: {
      const int split_k = maybe_force_split_k(compute_split_k<ConfigC>(size_m, size_n, size_k));
      launch_for_config<ConfigC>(a, b_q_weight, b_qzeros, b_scales, b_q_perm, c,
                                 size_m, size_n, size_k, groups, split_k,
                                 use_v2_format, stream);
      break;
    }
    case ConfigId_A:
    default: {
      const int split_k = maybe_force_split_k(compute_split_k<ConfigA>(size_m, size_n, size_k));
      launch_for_config<ConfigA>(a, b_q_weight, b_qzeros, b_scales, b_q_perm, c,
                                 size_m, size_n, size_k, groups, split_k,
                                 use_v2_format, stream);
      break;
    }
  }
}

}  // namespace gptq_rdna2_prefill
}  // namespace vllm

// ---------------------------------------------------------------------------
// torch::Tensor public entry point.
//
// `gptq_gemm_rdna2_prefill` is the canonical name going forward.  It calls
// the K-aware `select_config(M, K, N)` to pick V1 / A / C internally; the
// Python dispatcher does not need to know about configs.  An ABI shim with
// the old `..._prefill_direct` name is kept at the bottom of the file so
// existing Python callers continue to link without modification.
// ---------------------------------------------------------------------------
torch::Tensor gptq_gemm_rdna2_prefill(
    torch::Tensor a, torch::Tensor b_q_weight, torch::Tensor b_qzeros,
    torch::Tensor b_scales, torch::Tensor b_g_idx, bool use_v2_format) {
  TORCH_CHECK(a.is_cuda(), "a must be a CUDA/HIP tensor");
  TORCH_CHECK(b_q_weight.is_cuda(), "b_q_weight must be a CUDA/HIP tensor");
  TORCH_CHECK(b_qzeros.is_cuda(), "b_qzeros must be a CUDA/HIP tensor");
  TORCH_CHECK(b_scales.is_cuda(), "b_scales must be a CUDA/HIP tensor");
  TORCH_CHECK(a.dim() == 2, "a must be 2D [M, K]");
  TORCH_CHECK(b_q_weight.dim() == 2, "b_q_weight must be 2D [K/8, N]");
  TORCH_CHECK(a.scalar_type() == torch::kHalf,
              "gptq_gemm_rdna2_prefill only supports fp16");
  TORCH_CHECK(a.scalar_type() == b_scales.scalar_type(),
              "b_scales dtype must match a");

  const int size_m = a.size(0);
  const int size_k = a.size(1);
  const int size_n = b_q_weight.size(1);
  const int groups = b_qzeros.size(0);
  const int groupsize = size_k / groups;

  TORCH_CHECK(size_k % 32 == 0, "K must be divisible by 32");
  // N divisibility: the smallest config (ConfigC, N_PER_THREAD=4) requires
  // N % 8 == 0 for aligned int4 loads.
  TORCH_CHECK(size_n % 8 == 0, "N must be divisible by 8");
  TORCH_CHECK(size_k % groups == 0, "K must be divisible by groups");
  TORCH_CHECK(groupsize >= 32, "group_size must be >= 32");

  static Rdna2PersistBuf g_prefill_c;
  auto c = rdna2_persist_zeros(g_prefill_c, {size_m, size_n}, a.options());
  const c10::cuda::OptionalCUDAGuard device_guard(device_of(a));
  auto stream = c10::cuda::getCurrentCUDAStream();

  vllm::gptq_rdna2_prefill::launch_dispatch(
      reinterpret_cast<const half*>(a.data_ptr()),
      reinterpret_cast<const uint32_t*>(b_q_weight.data_ptr()),
      reinterpret_cast<const uint32_t*>(b_qzeros.data_ptr()),
      reinterpret_cast<const half*>(b_scales.data_ptr()),
      b_g_idx.numel() ? b_g_idx.data_ptr<int>() : nullptr,
      reinterpret_cast<half*>(c.data_ptr()), size_m, size_n, size_k, groups,
      use_v2_format, stream.stream());

  return c;
}

// ---------------------------------------------------------------------------
// ABI-compat shim.
//
// Same signature and behaviour as the original v1 public entry.  Kept so
// existing Python callers (RDNA2W4A16LinearKernel.apply_weights in
// rdna2_w4a16.py) that call `ops.gptq_gemm_rdna2_prefill_direct(...)` keep
// working without a Python change.  New code should use
// `ops.gptq_gemm_rdna2_prefill` directly.
// ---------------------------------------------------------------------------
torch::Tensor gptq_gemm_rdna2_prefill_direct(
    torch::Tensor a, torch::Tensor b_q_weight, torch::Tensor b_qzeros,
    torch::Tensor b_scales, torch::Tensor b_g_idx, bool use_v2_format) {
  return gptq_gemm_rdna2_prefill(a, b_q_weight, b_qzeros, b_scales, b_g_idx,
                                 use_v2_format);
}
