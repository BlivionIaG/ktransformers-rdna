# Resolve a ROCm toolchain for the kt-kernel host module and the per-arch
# device libraries under kt-kernel/rocm/. Included from the parent
# CMakeLists.txt and from the standalone rocm project. Does not enable HIP
# language support: device code is compiled by invoking hipcc once per arch
# so a gfx1030 object never shares a translation unit with WMMA.

if(KT_ROCM_RESOLVED)
    return()
endif()
set(KT_ROCM_RESOLVED ON)

if(DEFINED ENV{ROCM_PATH} AND NOT "$ENV{ROCM_PATH}" STREQUAL "")
    set(_KT_ROCM_PATH_DEFAULT "$ENV{ROCM_PATH}")
elseif(EXISTS "/opt/rocm")
    set(_KT_ROCM_PATH_DEFAULT "/opt/rocm")
else()
    set(_KT_ROCM_PATH_DEFAULT "/usr")
endif()
set(ROCM_PATH "${_KT_ROCM_PATH_DEFAULT}" CACHE PATH "ROCm installation root")

if(NOT KT_ROCM_ARCHS)
    if(DEFINED ENV{KT_ROCM_ARCHS} AND NOT "$ENV{KT_ROCM_ARCHS}" STREQUAL "")
        set(KT_ROCM_ARCHS "$ENV{KT_ROCM_ARCHS}")
    elseif(DEFINED ENV{PYTORCH_ROCM_ARCH} AND NOT "$ENV{PYTORCH_ROCM_ARCH}" STREQUAL "")
        set(KT_ROCM_ARCHS "$ENV{PYTORCH_ROCM_ARCH}")
    else()
        set(KT_ROCM_ARCHS "gfx1030;gfx1100")
    endif()
endif()
set(KT_ROCM_ARCHS "${KT_ROCM_ARCHS}" CACHE STRING
    "Semicolon-separated HIP offload architectures. Default gfx1030;gfx1100. Each arch is compiled as its own object; WMMA is emitted only for gfx1100-class archs.")
string(REPLACE "," ";" KT_ROCM_ARCHS "${KT_ROCM_ARCHS}")
string(REPLACE " " ";" KT_ROCM_ARCHS "${KT_ROCM_ARCHS}")
list(REMOVE_DUPLICATES KT_ROCM_ARCHS)
if(NOT KT_ROCM_ARCHS)
    message(FATAL_ERROR "KT_ROCM_ARCHS is empty. Pass gfx1030, gfx1100, or both.")
endif()

find_program(KT_ROCM_HIPCC NAMES hipcc
    HINTS "${ROCM_PATH}/bin" "${ROCM_PATH}/llvm/bin"
    REQUIRED)
find_path(KT_ROCM_INCLUDE_DIR hip/hip_runtime_api.h
    HINTS "${ROCM_PATH}/include" "${ROCM_PATH}/include/hip"
    REQUIRED)
find_library(KT_ROCM_HIP_LIBRARY NAMES amdhip64
    HINTS "${ROCM_PATH}/lib" "${ROCM_PATH}/lib64"
    REQUIRED)

# RDNA3 WMMA (16x16x16). gfx1030 is Wave32 packed-DOT only and must not see these.
set(KT_ROCM_WMMA_ARCHS gfx1100 gfx1101 gfx1102 gfx1103 gfx1150 gfx1151)

message(STATUS "ROCm path: ${ROCM_PATH}")
message(STATUS "hipcc: ${KT_ROCM_HIPCC}")
message(STATUS "HIP include: ${KT_ROCM_INCLUDE_DIR}")
message(STATUS "amdhip64: ${KT_ROCM_HIP_LIBRARY}")
message(STATUS "ROCm archs: ${KT_ROCM_ARCHS}")
message(STATUS "WMMA archs (separate objects): ${KT_ROCM_WMMA_ARCHS}")
