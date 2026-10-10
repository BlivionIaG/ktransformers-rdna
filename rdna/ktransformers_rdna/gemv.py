"""fp16 skinny decode GEMV. y[M, N] = x[M, K] @ w[N, K].T (+ bias)."""

from __future__ import annotations

from typing import Optional

import torch

from .arch import select_linear_backend


def can_implement(
    *,
    m: int,
    k: int,
    n: int,
    arch: Optional[str] = None,
) -> tuple[bool, str]:
    backend = select_linear_backend(
        arch=arch, weight_format="float16", act_dtype="float16", kind="dense", m=m
    )
    if backend != "rdna_gemv":
        return False, f"backend={backend}"
    if k % 8 != 0:
        return False, "K must be a multiple of 8"
    if n <= 0:
        return False, "N must be positive"
    return True, ""


def gemv_f16(
    x: torch.Tensor,
    w: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """``w`` is [N, K] fp16, ``x`` is [M, K] fp16, M in 1..8."""
    from ._ops import load

    if x.dtype != torch.float16 or w.dtype != torch.float16:
        raise TypeError("gemv_f16 is fp16 only")
    x2 = x.reshape(-1, x.shape[-1]).contiguous()
    return load().gemv_f16_rdna2(x2, w.contiguous(), bias)
