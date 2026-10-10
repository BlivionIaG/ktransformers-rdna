"""Runtime arch dispatch.

gfx1030 (RDNA2) and gfx1100 (RDNA3) both run the packed-DOT kernels in this
module. gfx1100 WMMA is a separate object that this package does not build
and does not load. Anything else, including MLA attention, stays on the
caller's existing fallback (FlashInfer / Triton / Marlin on CUDA, or the
ROCm beta path).
"""

from __future__ import annotations

import os
from typing import Optional

# Wave32 parts whose packed DOT (fdot2) matches the kernels we compile.
_DOT_PREFIXES = ("gfx103", "gfx110")

# DeepSeek-style MLA. fa_rdna2 does not implement it.
_MLA_MODEL_TYPES = {
    "deepseek_v2",
    "deepseek_v3",
    "deepseek_v32",
    "deepseek_v4",
    "deepseek_v4_flash",
    "kimi_k2",
    "kimi_vl",
    "glm4_moe",
    "glm4_moe_lite",
    "minimax_m2",
    "minimax_text_01",
}

_MLA_ATTN = {"mla", "deepseek_mla", "flashmla", "trtllm_mla"}


def gcn_arch(device: Optional[int] = None) -> Optional[str]:
    """Return the live GPU's gfx name, or None when no HIP device is visible.

    ``KT_RDNA_ARCH`` is consulted only when no device is present, so unit
    tests can exercise the selector. A real device always wins.
    """
    try:
        import torch
    except ImportError:
        return os.environ.get("KT_RDNA_ARCH") or None
    if torch.cuda.is_available():
        index = 0 if device is None else device
        name = torch.cuda.get_device_properties(index).gcnArchName
        return name.split(":")[0]
    override = os.environ.get("KT_RDNA_ARCH")
    return override or None


def uses_dot_kernels(arch: Optional[str] = None) -> bool:
    """True when this module's fdot2 kernels are the native path."""
    if arch is None:
        arch = gcn_arch()
    if not arch:
        return False
    return arch.startswith(_DOT_PREFIXES)


def attention_kind(model_type: Optional[str] = None, attn_type: Optional[str] = None) -> str:
    """``gqa`` for the fa_rdna2 path, ``mla`` when the caller must fall back.

    Phase 1 does not implement MLA. DeepSeek-style model types and an
    explicit ``mla`` attention type both report ``mla``.
    """
    kind = (attn_type or "").lower()
    model = (model_type or "").lower()
    if kind in _MLA_ATTN or model in _MLA_MODEL_TYPES:
        return "mla"
    if "mla" in kind or "mla" in model:
        return "mla"
    return "gqa"


def select_attention_backend(
    *,
    arch: Optional[str] = None,
    model_type: Optional[str] = None,
    attn_type: Optional[str] = None,
    head_dim: Optional[int] = None,
    num_heads: Optional[int] = None,
    num_kv_heads: Optional[int] = None,
) -> str:
    """``rdna_fa`` or ``fallback``.

    ``rdna_fa`` is GQA (or MHA) fp16 attention with head dim 128 or 256 on
    a DOT arch. MLA and every other shape stay on the existing backend.
    """
    if os.environ.get("KT_USE_RDNA2_FA", "1") not in ("1", "true", "True"):
        return "fallback"
    if attention_kind(model_type, attn_type) == "mla":
        return "fallback"
    if not uses_dot_kernels(arch):
        return "fallback"
    if head_dim not in (None, 128, 256):
        return "fallback"
    if num_heads is not None and num_kv_heads is not None:
        if num_kv_heads <= 0 or num_heads % num_kv_heads != 0:
            return "fallback"
    return "rdna_fa"


def select_linear_backend(
    *,
    arch: Optional[str] = None,
    weight_format: str,
    act_dtype: str = "float16",
    kind: str = "dense",
    m: Optional[int] = None,
) -> str:
    """Choose the GPU linear / expert kernel, or ``fallback``.

    ``kind`` is ``dense`` or ``expert``. Supported quantized format is
    GPTQ-Int4 (uint4b8). fp16 dense skinny decode (M <= 8) uses
    ``rdna_gemv``. Anything else, including Marlin-only layouts, returns
    ``fallback`` so the caller keeps its existing op.
    """
    if not uses_dot_kernels(arch):
        return "fallback"
    fmt = weight_format.lower().replace("-", "_")
    act = act_dtype.lower()
    if kind == "expert":
        if fmt in ("gptq_int4", "gptq_int4_uint4b8") and act in ("float16", "fp16", "torch.float16"):
            return "rdna_moe_q_gemm"
        return "fallback"
    if fmt in ("float16", "fp16", "torch.float16") and act in (
        "float16",
        "fp16",
        "torch.float16",
    ):
        if m is not None and 1 <= m <= 8:
            return "rdna_gemv"
        return "fallback"
    if fmt in ("gptq_int4", "gptq_int4_uint4b8") and act in ("float16", "fp16", "torch.float16"):
        return "rdna_q_gemm"
    return "fallback"
