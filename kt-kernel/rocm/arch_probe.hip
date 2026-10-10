// Packed-dot probe compiled once per offload arch.
// hipcc is invoked with a single --offload-arch. Matrix-core builtins do not
// belong in this translation unit.
#include <hip/hip_runtime.h>

#include <cstdint>

#ifndef KT_RDNA_ARCH_TAG
#error "KT_RDNA_ARCH_TAG must be set to one architecture by kt-kernel/rocm/CMakeLists.txt"
#endif

#define KT_RDNA_STR_IMPL(x) #x
#define KT_RDNA_STR(x) KT_RDNA_STR_IMPL(x)

extern "C" {
// Default visibility keeps this in the loadable image. hipcc -shared can
// otherwise drop the unreferenced host string, and `strings` then misses it.
__attribute__((used, visibility("default")))
const char kt_rdna_object_arch[] = "KT_RDNA_OBJECT_ARCH=" KT_RDNA_STR(KT_RDNA_ARCH_TAG);
__attribute__((used, visibility("default")))
int kt_rdna_object_has_matrix_core() { return 0; }
}

__device__ float kt_rdna_fdot2(unsigned a, unsigned b) {
  float acc = 0.f;
#if defined(__HIP_DEVICE_COMPILE__)
  // Packed f16 dot. Legal on gfx1030 and gfx1100.
  asm volatile("v_dot2_f32_f16 %0, %1, %2, %0" : "+v"(acc) : "v"(a), "v"(b));
#endif
  return acc;
}

extern "C" __global__ void kt_rdna_fdot2_probe(const unsigned* a, const unsigned* b, float* out, unsigned n) {
  const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = kt_rdna_fdot2(a[i], b[i]);
}
