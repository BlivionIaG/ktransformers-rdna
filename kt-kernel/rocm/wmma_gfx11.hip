// gfx1100-class WMMA only. CMake compiles this with one --offload-arch from
// {gfx1100, gfx1101, gfx1102, gfx1103, gfx1150, gfx1151}. A gfx1030 device
// pass hits the #error below, including a mixed --offload-arch=gfx1030,gfx1100
// command, because hipcc device-compiles each arch separately.
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#ifndef KT_RDNA_ARCH_TAG
#error "KT_RDNA_ARCH_TAG must be set to one gfx11 architecture"
#endif

#define KT_RDNA_STR_IMPL(x) #x
#define KT_RDNA_STR(x) KT_RDNA_STR_IMPL(x)

extern "C" {
__attribute__((used, visibility("default")))
const char kt_rdna_wmma_object_arch[] = "KT_RDNA_WMMA_TU=" KT_RDNA_STR(KT_RDNA_ARCH_TAG);
__attribute__((used, visibility("default")))
int kt_rdna_wmma_entry() { return 1100; }
}

#if defined(__HIP_DEVICE_COMPILE__)
#if !(defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1103__) || \
      defined(__gfx1150__) || defined(__gfx1151__))
#error "kt_rdna WMMA object compiled for a non-gfx11 device"
#endif

// hipcc rejects ext_vector_type of __half: __half is a class, not a scalar.
// The WMMA builtin takes a vector of _Float16. Bit-convert only where a
// __half value meets that vector.
__device__ inline _Float16 kt_rdna_f16_from_half(__half value) {
  // __half_raw::data is the _Float16 member of the HIP half union.
  return static_cast<__half_raw>(value).data;
}

__device__ float kt_rdna_wmma_f32_16x16x16_f16(float* out) {
  using half16 = __attribute__((__vector_size__(16 * sizeof(_Float16)))) _Float16;
  using float8 = __attribute__((__vector_size__(8 * sizeof(float)))) float;
  __half zero{};
  half16 a = {};
  half16 b = {};
  a[0] = kt_rdna_f16_from_half(zero);
  b[0] = kt_rdna_f16_from_half(zero);
  float8 c = {};
  float8 d = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
  if (out != nullptr && threadIdx.x == 0) out[0] = d[0];
  return d[0];
}
#endif

extern "C" __global__ void kt_rdna_wmma_probe(float* out) {
#if defined(__HIP_DEVICE_COMPILE__)
  kt_rdna_wmma_f32_16x16x16_f16(out);
#else
  (void)out;
#endif
}
