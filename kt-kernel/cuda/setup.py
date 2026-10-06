
import os
import sys

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


def _rocm_requested() -> bool:
    value = os.environ.get("CPUINFER_USE_ROCM", "").strip().lower()
    return value in {"1", "on", "true", "yes", "y"}


if _rocm_requested():
    sys.exit(
        "KTransformersOps is not built on ROCm.\n"
        "  gptq_marlin_gemm: Leave (no ROCm Marlin; phase 1 uses the gfx1030/gfx1100 GEMM import).\n"
        "  GGUF GPU dequant (q2_k..q8_0, iq4_xs): Leave (GGUF experts stay on CPU LLAMAFILE).\n"
        "  topk_softmax: handled in https://github.com/BlivionIaG/sglang-kt-rdna (sgl-kernel).\n"
        "Unset CPUINFER_USE_ROCM to build this CUDA extension for NVIDIA."
    )

setup(
    name='KTransformersOps',
    ext_modules=[
        CUDAExtension(
            'KTransformersOps', [
                'custom_gguf/dequant.cu',
                'binding.cpp',
                'gptq_marlin/gptq_marlin.cu',
                'moe/moe_topk_softmax_kernels.cu',
                # 'gptq_marlin_repack.cu',
            ],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': [
                    '-O3',
                    '--use_fast_math',
                    '-Xcompiler', '-fPIC',
                ]
            },
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)