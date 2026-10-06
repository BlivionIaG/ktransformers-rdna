"""Quantized linear backend: q_gemm instead of Marlin on RDNA.

SGLang calls ``select_linear_backend`` (re-exported from ``arch``) and, when
the result is ``rdna_q_gemm``, ``prepare_gptq_int4`` once at load and
``apply_gptq_int4`` in the forward. Unsupported formats return ``fallback``
and these functions are not called.
"""

from __future__ import annotations

from typing import Optional

import torch

from . import graph
from .arch import select_linear_backend
from .layout import prepare_gptq_int4_tensors


def can_implement(
    *,
    weight_format: str,
    act_dtype: str = "float16",
    k: int,
    n: int,
    group_size: int,
    arch: Optional[str] = None,
) -> tuple[bool, str]:
    backend = select_linear_backend(
        arch=arch, weight_format=weight_format, act_dtype=act_dtype, kind="dense"
    )
    if backend != "rdna_q_gemm":
        return False, f"backend={backend}"
    if group_size < 32 or k % group_size != 0:
        return False, "group_size must be >= 32 and divide K"
    if k % 32 != 0 or n % 8 != 0:
        return False, "need K % 32 == 0 and N % 8 == 0"
    return True, ""


def prepare_gptq_int4(
    q_kn: torch.Tensor,
    scales_gn: torch.Tensor,
    *,
    zeros_gn: torch.Tensor | None = None,
    g_idx: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Pack raw GPTQ nibbles into the kernel layout. See ``layout.py``."""
    return prepare_gptq_int4_tensors(
        q_kn, scales_gn, zeros_gn=zeros_gn, g_idx=g_idx
    )


def _maybe_clone(y: torch.Tensor, clone: Optional[bool]) -> torch.Tensor:
    if clone is None:
        clone = not graph.is_capturing()
    return y.clone() if clone else y


def apply_gptq_int4(
    x: torch.Tensor,
    prepared: dict[str, torch.Tensor],
    *,
    bias: torch.Tensor | None = None,
    clone: Optional[bool] = None,
) -> torch.Tensor:
    """y = x @ W_dequant, fp16.

    Decode (M <= 16, or M <= 32 and K >= 4096) uses ``gptq_gemm_rdna2``.
    Larger M uses ``gptq_gemm_rdna2_prefill``. Both outputs alias a persist
    buffer; the default clone keeps eager results stable. Pass ``clone=False``
    only inside ``graph.set_capturing(True)``.
    """
    from ._ops import load

    if x.dtype != torch.float16:
        raise TypeError("q_gemm activations are fp16")
    x2 = x.reshape(-1, x.shape[-1]).contiguous()
    m, k = x2.shape
    qweight = prepared["qweight"]
    if qweight.shape[0] * 8 != k:
        raise ValueError("qweight K does not match activation K")
    use_prefill = not (m <= 16 or (m <= 32 and k >= 4096))
    op = load()
    args = (
        x2,
        qweight,
        prepared["qzeros"],
        prepared["scales"],
        prepared["g_idx"],
        bool(prepared.get("use_v2_format", False)),
    )
    if use_prefill:
        y = op.gptq_gemm_rdna2_prefill(*args)
    else:
        y = op.gptq_gemm_rdna2(*args)
    y = _maybe_clone(y, clone)
    if bias is not None:
        y = y + bias
    return y.reshape(x.shape[:-1] + (qweight.shape[1],))
