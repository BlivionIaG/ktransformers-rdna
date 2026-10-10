"""Self-contained HIP extension for the gfx1030 packed-DOT kernels.

Build (ROCm PyTorch, no GPU required to compile):

    cd rdna
    PYTORCH_ROCM_ARCH=gfx1030;gfx1100 python setup.py build_ext --inplace

The fatbin contains fdot2 code for gfx1030 and gfx1100. It does not contain
a gfx1100 WMMA object. Import path after install: ``ktransformers_rdna``.
"""

import glob
import os
import subprocess
from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ROOT = Path(__file__).resolve().parent
CSRC = ROOT / "csrc"
ARCH_HEADER = CSRC / "rdna_dot_arch.h"

os.environ.setdefault("PYTORCH_ROCM_ARCH", "gfx1030;gfx1100")


def _libstdcxx_flags():
    """ROCm clang on Ubuntu does not search the multiarch libstdc++ dirs."""
    vers = sorted(glob.glob("/usr/include/c++/[0-9]*"))
    if not vers:
        return []
    flags = ["-idirafter", vers[-1]]
    try:
        triple = subprocess.check_output(
            ["gcc", "-dumpmachine"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return flags
    multi = f"/usr/include/{triple}/c++/{Path(vers[-1]).name}"
    if Path(multi).is_dir():
        flags += ["-idirafter", multi]
    return flags


def _hip_include_front():
    """Prefer ROCm's hip/ over an older copy in /usr/include/hip.

    hipcc records $ROCM_PATH/include as -idirafter, so /usr/include wins.
    A distinct directory holding only a symlink is a real -I and is not
    deduped against that idirafter entry.
    """
    rocm = Path(os.environ.get("ROCM_PATH", "/opt/rocm"))
    hip = rocm / "include" / "hip"
    if not hip.is_dir():
        return []
    dest = Path("/tmp/ktransformers-rdna-hipinc")
    dest.mkdir(parents=True, exist_ok=True)
    link = dest / "hip"
    target = hip.resolve()
    if not link.is_symlink() or Path(os.path.realpath(link)) != target:
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(hip)
    return [str(dest)]


def _compat_include():
    rocm = Path(os.environ.get("ROCM_PATH", "/opt/rocm"))
    dirs = _hip_include_front()
    if (rocm / "include" / "cuda_runtime_api.h").is_file():
        return dirs
    return dirs + [str(CSRC / "rocm_cuda_compat")]

sources = [
    "csrc/bindings.cpp",
    "csrc/dot_arch_probe.cu",
    "csrc/fa_rdna2.cu",
    "csrc/q_gemm_rdna2.cu",
    "csrc/q_gemm_rdna2_prefill.cu",
    "csrc/moe_q_gemm_rdna2.cu",
    "csrc/gemv_f16_rdna2.cu",
]

hip_flags = [
    "-std=c++17",
    f"-I{CSRC}",
    "-include",
    str(CSRC / "rocm_cuda_compat" / "torch_api_macros.h"),
    "-include",
    str(ARCH_HEADER),
    "-U__HIPCC_RTC__",
    "-DC10_CUDA_NO_CMAKE_CONFIGURE_FILE",
    *_libstdcxx_flags(),
]

setup(
    name="ktransformers-rdna-kernels",
    version="0.1.0",
    description="gfx1030 packed-DOT HIP kernels (attention, W4A16 GEMM, GEMV, MoE)",
    packages=["ktransformers_rdna"],
    package_dir={"ktransformers_rdna": "ktransformers_rdna"},
    ext_modules=[
        CUDAExtension(
            name="ktransformers_rdna._C",
            sources=sources,
            include_dirs=_compat_include(),
            extra_compile_args={"cxx": hip_flags, "nvcc": hip_flags},
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    python_requires=">=3.10",
)
