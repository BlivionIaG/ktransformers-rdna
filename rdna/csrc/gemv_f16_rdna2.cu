// Ported into https://github.com/BlivionIaG/ktransformers-rdna
// from https://github.com/opengfx1030/vllm-rdna branch rdna_extras
// csrc/rocm/skinny_gemms_int4.cu (gemv_f16_rdna2 only),
// commit c6d99ca30580f0ce6a97a168fa53bf9c6ee6b2af.
//
// The rest of skinny_gemms_int4.cu (int4 MoE GEMV, int8 GEMV) is not part
// of this module. The kernel below is the fp16 skinny decode GEMV, kept
// behavior-identical: y[M, N] = x[M, K] @ w[N, K]^T (+ optional bias),
// M in 1..8, K % 8 == 0, fp16, wave32, v_dot2_f32_f16.
//
// gfx1100 uses this same DOT kernel. No WMMA variant is built.

#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// ---------------------------------------------------------------------------
// fp16 skinny GEMM for gfx1030 decode (M <= 8 tokens): y[M,N] = x[M,K].w[N,K]^T
// (+ bias). Wave-per-output-row, lanes stride K with 16-byte loads, 4-deep
// unroll for memory-level parallelism, v_dot2_f32_f16 accumulate, wave32
// shuffle reduce.
// ---------------------------------------------------------------------------
template <int WAVES, int MT>
__global__ void __launch_bounds__(WAVES * 32)
gemv_f16_rdna2_(const half* __restrict__ x, const half* __restrict__ w,
                const half* __restrict__ bias, half* __restrict__ y,
                const int N, const int K) {
  const int wave = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int n = blockIdx.x * WAVES + wave;
  if (n >= N) return;
  const int K8 = K / 8;
  const uint4* __restrict__ wrow =
      reinterpret_cast<const uint4*>(w + (size_t)n * K);
  const uint4* __restrict__ xr = reinterpret_cast<const uint4*>(x);
  float acc[MT];
#pragma unroll
  for (int m = 0; m < MT; m++) acc[m] = 0.f;
  for (int i = lane; i < K8; i += 32 * 4) {
    uint4 wq[4];
#pragma unroll
    for (int u = 0; u < 4; u++) {
      const int idx = i + 32 * u;
      wq[u] = (idx < K8) ? wrow[idx] : make_uint4(0, 0, 0, 0);
    }
#pragma unroll
    for (int u = 0; u < 4; u++) {
      const int idx = i + 32 * u;
      if (idx < K8) {
        const half2* wh = reinterpret_cast<const half2*>(&wq[u]);
#pragma unroll
        for (int m = 0; m < MT; m++) {
          const uint4 xq = xr[(size_t)m * K8 + idx];
          const half2* xh = reinterpret_cast<const half2*>(&xq);
          float a = acc[m];
          a = __builtin_amdgcn_fdot2(wh[0], xh[0], a, false);
          a = __builtin_amdgcn_fdot2(wh[1], xh[1], a, false);
          a = __builtin_amdgcn_fdot2(wh[2], xh[2], a, false);
          a = __builtin_amdgcn_fdot2(wh[3], xh[3], a, false);
          acc[m] = a;
        }
      }
    }
  }
#pragma unroll
  for (int m = 0; m < MT; m++) {
#pragma unroll
    for (int off = 16; off >= 1; off >>= 1) acc[m] += __shfl_xor(acc[m], off);
  }
  if (lane == 0) {
    const float b = bias ? __half2float(bias[n]) : 0.f;
#pragma unroll
    for (int m = 0; m < MT; m++)
      y[(size_t)m * N + n] = __float2half(acc[m] + b);
  }
}

template <int MT>
static void gemv_f16_rdna2_launch(const half* x, const half* w,
                                  const half* bias, half* y, int N, int K,
                                  hipStream_t s) {
  // Fewer waves per block for small N so the grid still covers all CUs.
  if (N >= 1152)
    gemv_f16_rdna2_<8, MT><<<(N + 7) / 8, 256, 0, s>>>(x, w, bias, y, N, K);
  else if (N >= 576)
    gemv_f16_rdna2_<4, MT><<<(N + 3) / 4, 128, 0, s>>>(x, w, bias, y, N, K);
  else if (N >= 288)
    gemv_f16_rdna2_<2, MT><<<(N + 1) / 2, 64, 0, s>>>(x, w, bias, y, N, K);
  else
    gemv_f16_rdna2_<1, MT><<<N, 32, 0, s>>>(x, w, bias, y, N, K);
}

at::Tensor gemv_f16_rdna2(const at::Tensor& x, const at::Tensor& w,
                          const std::optional<at::Tensor>& bias) {
  const int M = x.size(0), K = x.size(1), N = w.size(0);
  TORCH_CHECK(M >= 1 && M <= 8, "gemv_f16_rdna2: M must be 1..8");
  TORCH_CHECK(w.size(1) == K, "gemv_f16_rdna2: K mismatch");
  TORCH_CHECK(K % 8 == 0, "gemv_f16_rdna2: K % 8 == 0");
  TORCH_CHECK(x.scalar_type() == at::kHalf && w.scalar_type() == at::kHalf,
              "gemv_f16_rdna2: fp16 only");
  TORCH_CHECK(x.is_contiguous() && w.is_contiguous(),
              "gemv_f16_rdna2: contiguous operands");
  const half* bp = nullptr;
  if (bias.has_value()) {
    TORCH_CHECK(bias->scalar_type() == at::kHalf && bias->is_contiguous() &&
                    bias->numel() == N,
                "gemv_f16_rdna2: bias must be contiguous fp16 [N]");
    bp = reinterpret_cast<const half*>(bias->const_data_ptr());
  }
  const c10::cuda::OptionalCUDAGuard guard(x.device());
  auto y = at::empty({M, N}, x.options());
  const hipStream_t s = c10::cuda::getCurrentCUDAStream();
  const half* xp = reinterpret_cast<const half*>(x.const_data_ptr());
  const half* wp = reinterpret_cast<const half*>(w.const_data_ptr());
  half* yp = reinterpret_cast<half*>(y.mutable_data_ptr());
  switch (M) {
    case 1: gemv_f16_rdna2_launch<1>(xp, wp, bp, yp, N, K, s); break;
    case 2: gemv_f16_rdna2_launch<2>(xp, wp, bp, yp, N, K, s); break;
    case 3: gemv_f16_rdna2_launch<3>(xp, wp, bp, yp, N, K, s); break;
    case 4: gemv_f16_rdna2_launch<4>(xp, wp, bp, yp, N, K, s); break;
    case 5: gemv_f16_rdna2_launch<5>(xp, wp, bp, yp, N, K, s); break;
    case 6: gemv_f16_rdna2_launch<6>(xp, wp, bp, yp, N, K, s); break;
    case 7: gemv_f16_rdna2_launch<7>(xp, wp, bp, yp, N, K, s); break;
    default: gemv_f16_rdna2_launch<8>(xp, wp, bp, yp, N, K, s); break;
  }
  return y;
}
