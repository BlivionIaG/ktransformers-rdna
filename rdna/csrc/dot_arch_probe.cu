// Compile probe. The device pass for every offload arch in this module
// must enable the packed-DOT bodies and accept fdot2. A gfx1100 WMMA
// object is intentionally absent.

#include "rdna_dot_arch.h"

#if defined(__HIP_DEVICE_COMPILE__)
#if !defined(__HIP__RDNA2__)
#error "packed DOT bodies were not enabled for this offload arch"
#endif

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

__global__ void rdna_dot_arch_probe(float* out) {
  float acc = 0.f;
  const half2 a = __halves2half2(__float2half(1.f), __float2half(1.f));
  const half2 b = a;
  acc = __builtin_amdgcn_fdot2(a, b, acc, false);
  if (out && threadIdx.x == 0) {
    out[0] = acc;
  }
}
#endif
