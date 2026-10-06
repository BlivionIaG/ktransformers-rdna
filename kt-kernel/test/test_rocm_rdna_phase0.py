"""Source contracts for the ROCm gfx1030/gfx1100 phase-0 build.

These checks do not need a GPU. They pin the arch split, the HIP layerwise
transport, the CUDA-extension gate, and the CPU expert paths used by Qwen3.5
and DeepSeek-V4-Flash.
"""

from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
KT = ROOT / "kt-kernel"


class TestRdnaPhase0(unittest.TestCase):
    def test_docs_name_the_fork_and_the_target_models(self):
        phase0 = (ROOT / "docs/rdna/PHASE0.md").read_text(encoding="utf-8")
        inventory = (ROOT / "docs/rdna/INVENTORY.md").read_text(encoding="utf-8")
        for text in (phase0, inventory):
            self.assertIn("https://github.com/BlivionIaG/ktransformers-rdna", text)
            self.assertIn("https://github.com/BlivionIaG/sglang-kt-rdna", text)
            self.assertIn("Qwen3-30B-A3B", text)
            self.assertIn("Qwen3.5", text)
            self.assertIn("DeepSeek-V4-Flash", text)
            self.assertIn("MXFP4", text)
            self.assertIn("GPTQ_INT4", text)
        self.assertIn("gfx1030", phase0)
        self.assertIn("gfx1100", phase0)

    def test_wmma_tu_rejects_non_gfx11_device_passes(self):
        src = (KT / "rocm/wmma_gfx11.hip").read_text(encoding="utf-8")
        self.assertIn("non-gfx11", src)
        self.assertIn("__builtin_amdgcn_wmma_f32_16x16x16_f16_w32", src)
        self.assertIn("__HIP_DEVICE_COMPILE__", src)
        probe = (KT / "rocm/arch_probe.hip").read_text(encoding="utf-8")
        self.assertNotIn("wmma", probe.lower())
        self.assertIn("v_dot2_f32_f16", probe)

    def test_cmake_compiles_one_arch_per_object(self):
        cmake = (KT / "rocm/CMakeLists.txt").read_text(encoding="utf-8")
        self.assertIn("--offload-arch=${ARCH}", cmake)
        self.assertIn("Refusing to compile WMMA", cmake)
        self.assertNotIn("--offload-arch=gfx1030;gfx1100", cmake)
        header = (KT / "rocm/rdna_arch.h").read_text(encoding="utf-8")
        self.assertIn('"gfx1100"', header)
        self.assertNotIn('"gfx1030"', header)

    def test_cuda_extension_refuses_rocm(self):
        setup = (KT / "cuda/setup.py").read_text(encoding="utf-8")
        self.assertIn("CPUINFER_USE_ROCM", setup)
        self.assertIn("gptq_marlin_gemm", setup)
        self.assertIn("sglang-kt-rdna", setup)

    def test_fp8_transport_accepts_hip_and_keeps_cuda(self):
        src = (KT / "fp8_layerwise_transport.cpp").read_text(encoding="utf-8")
        self.assertIn("#include <cuda_runtime_api.h>", src)
        self.assertIn("#include <hip/hip_runtime_api.h>", src)
        self.assertIn("KTRANSFORMERS_USE_ROCM", src)
        self.assertIn("CUDA- or ROCm-enabled", src)
        self.assertIn("INT4/W4A16 payload is a later change", src)

    def test_cpu_expert_paths_are_not_cuda_gated(self):
        bindings = (KT / "ext_bindings.cpp").read_text(encoding="utf-8")
        avx2 = bindings.split("#if defined(__x86_64__)", 1)[1]
        for symbol in (
            "AVX2BF16_MOE",
            "AVX2FP8_MOE",
            "AVX2GPTQInt4_MOE",
            "AVX2MXFP4_MOE",
        ):
            self.assertIn(symbol, avx2)
        amx = bindings.split("#if defined(__x86_64__) && defined(USE_AMX_AVX_KERNEL)", 1)[1]
        for symbol in ("AMXBF16_MOE", "AMXFP8_MOE", "AMXFP4_KGroup_MOE"):
            self.assertIn(symbol, amx)
        before_avx2 = bindings.split("AVX2BF16_MOE", 1)[0]
        self.assertNotIn("#if defined(KTRANSFORMERS_USE_CUDA)", before_avx2[-800:])

    def test_ci_image_is_public_rocm_7_14(self):
        workflow = (ROOT / ".github/workflows/rocm-rdna.yml").read_text(encoding="utf-8")
        self.assertIn("rocm/dev-ubuntu-24.04:7.14.0-full", workflow)
        self.assertIn("gfx1030;gfx1100", workflow)
        smoke = (KT / "scripts/rdna_smoke.sh").read_text(encoding="utf-8")
        self.assertIn("Qwen/Qwen3-30B-A3B", smoke)
        self.assertIn("sglang-kt-rdna", smoke)


if __name__ == "__main__":
    unittest.main()
