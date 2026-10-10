# Per-arch HIP libraries. Standalone:
#   cmake -S kt-kernel/rocm -B build/rocm -DKT_ROCM_ARCHS=gfx1030;gfx1100
#   cmake --build build/rocm
# From the kt-kernel build this directory is add_subdirectory'd when
# KTRANSFORMERS_USE_ROCM=ON.
#
# Drop-in sources from the kernel port (directories are not created here):
#   kt-kernel/rdna/gfx1030/*.hip              -> libkt_rdna_gfx1030.so only
#   kt-kernel/rdna/gfx1100/*.hip              -> libkt_rdna_gfx1100.so only
#   kt-kernel/rdna/gfx1100/wmma/*.hip         -> libkt_rdna_wmma_gfx1100.so only
# Do not put WMMA in the gfx1030 or non-wmma gfx1100 lists.

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    cmake_minimum_required(VERSION 3.16)
    project(kt_rdna_arch LANGUAGES NONE)
endif()

include("${CMAKE_CURRENT_SOURCE_DIR}/../cmake/ROCm.cmake")

set(KT_RDNA_DROPIN "${CMAKE_CURRENT_SOURCE_DIR}/../rdna")
if(CMAKE_LIBRARY_OUTPUT_DIRECTORY)
    set(KT_RDNA_OUTDIR "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}")
else()
    set(KT_RDNA_OUTDIR "${CMAKE_CURRENT_BINARY_DIR}")
endif()
file(MAKE_DIRECTORY "${KT_RDNA_OUTDIR}")

function(kt_rocm_arch_has_wmma ARCH OUT_VAR)
    if(ARCH IN_LIST KT_ROCM_WMMA_ARCHS)
        set(${OUT_VAR} ON PARENT_SCOPE)
    else()
        set(${OUT_VAR} OFF PARENT_SCOPE)
    endif()
endfunction()

function(kt_rocm_compile_hip SRC ARCH OBJ_OUT)
    get_filename_component(_base "${SRC}" NAME_WE)
    set(_obj "${CMAKE_CURRENT_BINARY_DIR}/${_base}.${ARCH}.o")
    add_custom_command(
        OUTPUT "${_obj}"
        COMMAND "${KT_ROCM_HIPCC}"
                -c "${SRC}"
                --offload-arch=${ARCH}
                -O3
                -fPIC
                -D__HIP_PLATFORM_AMD__
                -DKT_RDNA_ARCH_TAG=${ARCH}
                -I"${KT_ROCM_INCLUDE_DIR}"
                -o "${_obj}"
        DEPENDS "${SRC}"
        COMMENT "hipcc --offload-arch=${ARCH} ${_base}"
        VERBATIM
    )
    set(${OBJ_OUT} "${_obj}" PARENT_SCOPE)
endfunction()

# Link one shared library for exactly one arch. SOURCES is a CMake list of HIP files.
function(kt_rocm_add_arch_library LIB_NAME ARCH)
    set(_extra ${ARGN})
    kt_rocm_compile_hip("${CMAKE_CURRENT_SOURCE_DIR}/arch_probe.hip" "${ARCH}" _probe)
    set(_objs "${_probe}")
    foreach(_src IN LISTS _extra)
        kt_rocm_compile_hip("${_src}" "${ARCH}" _obj)
        list(APPEND _objs "${_obj}")
    endforeach()
    set(_so "${KT_RDNA_OUTDIR}/${LIB_NAME}.so")
    add_custom_command(
        OUTPUT "${_so}"
        COMMAND "${KT_ROCM_HIPCC}"
                --offload-arch=${ARCH}
                -shared
                -fPIC
                ${_objs}
                -o "${_so}"
        DEPENDS ${_objs}
        COMMENT "link ${LIB_NAME}.so (${ARCH} only)"
        VERBATIM
    )
    add_custom_target(${LIB_NAME} ALL DEPENDS "${_so}")
endfunction()

function(kt_rocm_add_wmma_library LIB_NAME ARCH)
    kt_rocm_arch_has_wmma("${ARCH}" _ok)
    if(NOT _ok)
        message(FATAL_ERROR
            "Refusing to compile WMMA library ${LIB_NAME} for ${ARCH}. "
            "WMMA objects are gfx1100, gfx1101, gfx1102, gfx1103, gfx1150, gfx1151 only.")
    endif()
    set(_extra ${ARGN})
    set(_sources "${CMAKE_CURRENT_SOURCE_DIR}/wmma_gfx11.hip")
    foreach(_src IN LISTS _extra)
        list(APPEND _sources "${_src}")
    endforeach()
    set(_objs "")
    foreach(_src IN LISTS _sources)
        kt_rocm_compile_hip("${_src}" "${ARCH}" _obj)
        list(APPEND _objs "${_obj}")
    endforeach()
    set(_so "${KT_RDNA_OUTDIR}/${LIB_NAME}.so")
    add_custom_command(
        OUTPUT "${_so}"
        COMMAND "${KT_ROCM_HIPCC}"
                --offload-arch=${ARCH}
                -shared
                -fPIC
                ${_objs}
                -o "${_so}"
        DEPENDS ${_objs}
        COMMENT "link ${LIB_NAME}.so (WMMA, ${ARCH} only)"
        VERBATIM
    )
    add_custom_target(${LIB_NAME} ALL DEPENDS "${_so}")
endfunction()

set(_rdna_targets "")
foreach(_arch IN LISTS KT_ROCM_ARCHS)
    set(_extra "")
    if(EXISTS "${KT_RDNA_DROPIN}/${_arch}")
        file(GLOB _extra "${KT_RDNA_DROPIN}/${_arch}/*.hip" "${KT_RDNA_DROPIN}/${_arch}/*.cu")
    endif()
    kt_rocm_add_arch_library("libkt_rdna_${_arch}" "${_arch}" ${_extra})
    list(APPEND _rdna_targets "libkt_rdna_${_arch}")

    kt_rocm_arch_has_wmma("${_arch}" _wmma)
    if(_wmma)
        set(_wmma_extra "")
        if(EXISTS "${KT_RDNA_DROPIN}/${_arch}/wmma")
            file(GLOB _wmma_extra "${KT_RDNA_DROPIN}/${_arch}/wmma/*.hip" "${KT_RDNA_DROPIN}/${_arch}/wmma/*.cu")
        endif()
        kt_rocm_add_wmma_library("libkt_rdna_wmma_${_arch}" "${_arch}" ${_wmma_extra})
        list(APPEND _rdna_targets "libkt_rdna_wmma_${_arch}")
    endif()
endforeach()

add_custom_target(kt_rdna_arch_libs)
foreach(_tgt IN LISTS _rdna_targets)
    add_dependencies(kt_rdna_arch_libs "${_tgt}")
endforeach()

message(STATUS "RDNA device libraries: ${_rdna_targets}")
message(STATUS "RDNA library directory: ${KT_RDNA_OUTDIR}")
