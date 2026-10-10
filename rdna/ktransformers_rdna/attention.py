"""GQA paged attention on fa_rdna2.

SGLang calls ``select_attention_backend``. ``rdna_fa`` means ``paged_attention``
below. ``fallback`` means the existing FlashInfer / Triton backend, which is
also the path for MLA (DeepSeek-style). This module does not implement MLA.

Q is [num_tokens, num_heads, head_dim] fp16.
K and V caches are the vLLM/SGLang paged layout
[num_blocks, num_kv_heads, head_dim // x, block_size, x] fp16.
``block_table`` is [num_seqs, max_blocks] int32.
``seq_lens`` is [num_seqs] int32 (KV length, prefix + query).
``cu_query_lens`` is [num_seqs + 1] int32.

The kernel reads ``VLLM_FA_RDNA2_GQA_DECODE`` (set to ``1`` to take the
head-dim-256 GQA decode kernel). ``KT_USE_RDNA2_FA=0`` forces fallback.
"""

from __future__ import annotations

import os
from typing import Optional

import torch

from .arch import select_attention_backend


def can_use(
    *,
    head_dim: int,
    num_heads: int,
    num_kv_heads: int,
    model_type: Optional[str] = None,
    attn_type: Optional[str] = None,
    arch: Optional[str] = None,
) -> tuple[bool, str]:
    backend = select_attention_backend(
        arch=arch,
        model_type=model_type,
        attn_type=attn_type,
        head_dim=head_dim,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
    )
    if backend != "rdna_fa":
        return False, backend
    return True, ""


def _scale(scale: Optional[float], head_dim: int) -> float:
    if scale is None:
        return head_dim ** -0.5
    return float(scale)


def _out(q: torch.Tensor, out: Optional[torch.Tensor]) -> torch.Tensor:
    if out is None:
        return torch.empty_like(q)
    return out


def decode_paged(
    q: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    *,
    block_size: int = 16,
    kv_splits: int = 8,
    sliding_window: int = 0,
    scale: Optional[float] = None,
    out: Optional[torch.Tensor] = None,
    cu_query_lens: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    from ._ops import load

    out = _out(q, out)
    load().fa_rdna2_decode_paged(
        q,
        key_cache,
        value_cache,
        block_table,
        seq_lens,
        int(block_size),
        int(kv_splits),
        int(sliding_window),
        _scale(scale, q.shape[-1]),
        out,
        cu_query_lens,
    )
    return out


def prefill_paged(
    q: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
    cu_query_lens: torch.Tensor,
    seq_lens: torch.Tensor,
    *,
    num_heads: int,
    num_kv_heads: int,
    block_size: int = 16,
    causal: bool = True,
    sliding_window: int = 0,
    scale: Optional[float] = None,
    out: Optional[torch.Tensor] = None,
    max_seqlen_k: Optional[int] = None,
) -> torch.Tensor:
    """Dispatch the same way vllm-rdna's RDNA attention backend does."""
    from ._ops import load

    out = _out(q, out)
    ext = load()
    if max_seqlen_k is None:
        max_seqlen_k = int(seq_lens.max().item()) if seq_lens.numel() else 0
    num_seqs = int(seq_lens.shape[0])
    kv_splits = min(8, (max_seqlen_k + 1023) // 1024)
    sc = _scale(scale, q.shape[-1])
    common = (
        q,
        key_cache,
        value_cache,
        block_table,
        cu_query_lens,
        seq_lens,
        int(block_size),
        int(causal),
    )
    gqa_mode = os.environ.get("VLLM_FA_RDNA2_GQA_MODE", "subgroup")
    if num_kv_heads and num_heads % num_kv_heads == 0 and gqa_mode == "subgroup":
        ext.fa_rdna2_prefill_paged_varlen_gqa(*common, int(sliding_window), sc, out)
    elif max_seqlen_k < 4096 and q.shape[-1] == 128:
        ext.fa_rdna2_prefill_paged_varlen_short(*common, int(sliding_window), sc, out)
    elif kv_splits >= 2 and num_seqs <= 4 and num_heads * kv_splits >= 64:
        ext.fa_rdna2_prefill_paged_varlen_splitk(
            *common, int(kv_splits), int(sliding_window), sc, out
        )
    else:
        ext.fa_rdna2_prefill_paged_varlen(*common, int(sliding_window), sc, out)
    return out


def paged_attention(
    q: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    cu_query_lens: torch.Tensor,
    *,
    num_heads: int,
    num_kv_heads: int,
    block_size: int = 16,
    causal: bool = True,
    sliding_window: int = 0,
    scale: Optional[float] = None,
    out: Optional[torch.Tensor] = None,
    kv_splits: int = 8,
    model_type: Optional[str] = None,
) -> torch.Tensor:
    """Prefill when any sequence has more than one query token, else decode.

    Raises ``RdnaFallback`` for MLA and for shapes this kernel does not cover.
    SGLang should catch that and use its existing attention backend.
    """
    ok, reason = can_use(
        head_dim=q.shape[-1],
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        model_type=model_type,
    )
    if not ok:
        raise RdnaFallback(reason)
    query_lens = cu_query_lens[1:] - cu_query_lens[:-1]
    if int(query_lens.max().item()) <= 1 and q.shape[0] == seq_lens.shape[0]:
        return decode_paged(
            q,
            key_cache,
            value_cache,
            block_table,
            seq_lens,
            block_size=block_size,
            kv_splits=kv_splits,
            sliding_window=sliding_window,
            scale=scale,
            out=out,
            cu_query_lens=None,
        )
    return prefill_paged(
        q,
        key_cache,
        value_cache,
        block_table,
        cu_query_lens,
        seq_lens,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        block_size=block_size,
        causal=causal,
        sliding_window=sliding_window,
        scale=scale,
        out=out,
    )


class RdnaFallback(RuntimeError):
    """The caller should use its existing attention or linear backend."""
