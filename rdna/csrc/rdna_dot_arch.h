// Packed-DOT device-body switch for the gfx1030 kernels in this module.
//
// gfx1030 is Wave32 and has packed DOT (v_dot2_f32_f16 / fdot2) only:
// no WMMA, MFMA, or FP8. gfx1100 has WMMA, but this module does not
// compile a WMMA translation unit. Both offload passes use the same
// fdot2 bodies. Included with hipcc -include so the upstream kernel
// sources keep their own __gfx1030__ guard and still see __HIP__RDNA2__
// on the gfx1100 device pass (that guard does not undefine a macro that
// is already set).
//
// A wave64 device pass (CDNA) must not take this path: the kernels reduce
// with wave32 shuffles.

#pragma once

#if defined(__HIP_DEVICE_COMPILE__)
#if defined(__gfx1030__) || defined(__gfx1031__) || defined(__gfx1032__) || \
    defined(__gfx1034__) || defined(__gfx1035__) || defined(__gfx1036__) || \
    defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || \
    defined(__gfx1103__)
#define __HIP__RDNA2__
#endif
#else
#define __HIP__RDNA2__
#endif
