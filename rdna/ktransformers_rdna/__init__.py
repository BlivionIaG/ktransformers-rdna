"""gfx1030 packed-DOT kernels for the SGLang serving path.

SGLang (https://github.com/BlivionIaG/sglang-kt-rdna) imports this package
as an attention backend and as the GPU hot-expert / quantized-linear method.
The legacy standalone ktransformers server is not a caller.

Kernels come from https://github.com/opengfx1030/vllm-rdna (``rdna_extras``).
See ``docs/rdna/KERNELS.md``.
"""

from .arch import (
    attention_kind,
    gcn_arch,
    select_attention_backend,
    select_linear_backend,
    uses_dot_kernels,
)
from .attention import RdnaFallback, paged_attention
from .gemv import gemv_f16
from .graph import freeze_capture_persist, is_capturing, set_capturing
from .linear import apply_gptq_int4, prepare_gptq_int4
from .moe import apply_gptq_int4_experts

__all__ = [
    "RdnaFallback",
    "apply_gptq_int4",
    "apply_gptq_int4_experts",
    "attention_kind",
    "freeze_capture_persist",
    "gcn_arch",
    "gemv_f16",
    "is_capturing",
    "paged_attention",
    "prepare_gptq_int4",
    "select_attention_backend",
    "select_linear_backend",
    "set_capturing",
    "uses_dot_kernels",
]
