"""GPTQ-Int4 expert GEMM for GPU-resident (hot) experts.

Cold experts stay on kt-kernel. The caller remaps global expert ids onto
the resident table and sets every other id to -1; this kernel skips -1.
Router softmax itself is not provided here.

Weight layout (one resident table, not the full MoE):
    w13  [E, K/8, 2I] int32 shuffled gate+up
    w13_scales [E, K/group, 2I] fp16
    w13_zeros  [E, K/group, 2I/8] int32
    w2   [E, I/8, K] int32 shuffled down
    w2_scales [E, I/group, K] fp16
    w2_zeros  [E, I/group, K/8] int32

``block_size_m`` is 1, 2, 4, or 8. The kernel's zero offset is the GPTQv1
+1 (uint4b8). AWQ expert weights are ``fallback``.
"""

from __future__ import annotations

from typing import Optional

import torch

from .arch import select_linear_backend
from .layout import prepare_gptq_int4_tensors


def can_implement(
    *,
    weight_format: str,
    act_dtype: str = "float16",
    group_size: int,
    k: int,
    n: int,
    arch: Optional[str] = None,
) -> tuple[bool, str]:
    backend = select_linear_backend(
        arch=arch, weight_format=weight_format, act_dtype=act_dtype, kind="expert"
    )
    if backend != "rdna_moe_q_gemm":
        return False, f"backend={backend}"
    if group_size < 32 or k % group_size != 0 or k % 32 != 0 or n % 8 != 0:
        return False, "need group>=32 dividing K, K%32==0, N%8==0"
    return True, ""


def prepare_expert(q_kn: torch.Tensor, scales_gn: torch.Tensor) -> dict[str, torch.Tensor]:
    """Pack one expert. Stack the results on dim 0 for the resident table."""
    return prepare_gptq_int4_tensors(q_kn, scales_gn)


def align_blocks(topk_ids: torch.Tensor, block_size_m: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Group [M, topk] expert ids into kernel blocks.

    Returns ``sorted_token_ids`` [num_blocks * block], ``expert_ids``
    [num_blocks], and ``num_tokens_post_padded`` [1], all int32.
    Token id ``m * topk + slot`` indexes the routing pair. Padding ids are
    ``M * topk``, which the kernel treats as out of range. Expert id -1
    is skipped.
    """
    if block_size_m not in (1, 2, 4, 8):
        raise ValueError("block_size_m must be 1, 2, 4, or 8")
    if topk_ids.dim() != 2:
        raise ValueError("topk_ids must be [M, topk]")
    m, topk = topk_ids.shape
    device = topk_ids.device
    ids = topk_ids.to(torch.int64).cpu()
    valid = ids >= 0
    experts = sorted({int(e) for e in ids[valid].tolist()})
    flat = []
    expert_ids = []
    pad = m * topk
    for expert in experts:
        slots = (ids == expert).nonzero(as_tuple=False)
        tokens = [int(r) * topk + int(c) for r, c in slots.tolist()]
        while len(tokens) % block_size_m != 0:
            tokens.append(pad)
        for base in range(0, len(tokens), block_size_m):
            flat.extend(tokens[base : base + block_size_m])
            expert_ids.append(expert)
    if not flat:
        flat = [pad] * block_size_m
        expert_ids = [-1]
    sorted_ids = torch.tensor(flat, dtype=torch.int32, device=device)
    eids = torch.tensor(expert_ids, dtype=torch.int32, device=device)
    npost = torch.tensor([sorted_ids.numel()], dtype=torch.int32, device=device)
    return sorted_ids, eids, npost


def expert_gemm(
    a: torch.Tensor,
    c: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    qzeros: torch.Tensor,
    topk_weights: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    top_k: int,
    block_size_m: int,
    *,
    mul_topk_weight: bool,
    output_topk: int,
    fp32_accum: bool = False,
) -> None:
    """One packed expert GEMM. ``c`` is overwritten (fp32 path) or accumulated (fp16 path)."""
    from ._ops import load

    weights = topk_weights
    if weights.numel() and weights.dtype != torch.float32:
        weights = weights.float()
    load().moe_gptq_gemm_rdna2(
        a.contiguous(),
        c,
        qweight,
        scales,
        qzeros,
        weights.contiguous(),
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        int(top_k),
        int(block_size_m),
        bool(mul_topk_weight),
        int(output_topk),
        bool(fp32_accum),
    )


def apply_gptq_int4_experts(
    hidden: torch.Tensor,
    w13: dict[str, torch.Tensor],
    w2: dict[str, torch.Tensor],
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    *,
    block_size_m: Optional[int] = None,
    fp32_accum: bool = False,
) -> torch.Tensor:
    """SwiGLU MoE over the resident experts: gate+up, silu, down-reduce.

    ``hidden`` is [M, K] fp16. ``topk_ids`` is [M, topk] int, already mapped
    into the resident table (-1 = cold expert, computed elsewhere and added
    by the caller). ``w13`` / ``w2`` are stacked ``prepare_expert`` outputs
    with a leading expert dimension.
    """
    if hidden.dtype != torch.float16:
        raise TypeError("MoE activations are fp16")
    m, _k = hidden.shape
    topk = topk_ids.shape[1]
    if block_size_m is None:
        block_size_m = 1 if m <= 4 else 4
    sorted_ids, expert_ids, npost = align_blocks(topk_ids, block_size_m)
    n_up = w13["qweight"].shape[-1]
    gate_up = torch.zeros(m * topk, n_up, dtype=torch.float16, device=hidden.device)
    empty = torch.empty(0, dtype=torch.float32, device=hidden.device)
    expert_gemm(
        hidden,
        gate_up,
        w13["qweight"],
        w13["scales"],
        w13["qzeros"],
        empty,
        sorted_ids,
        expert_ids,
        npost,
        topk,
        block_size_m,
        mul_topk_weight=False,
        output_topk=0,
        fp32_accum=fp32_accum,
    )
    gate, up = gate_up.chunk(2, dim=-1)
    activated = torch.nn.functional.silu(gate) * up
    out = torch.zeros(m, hidden.shape[-1], dtype=torch.float16, device=hidden.device)
    # Same routing ids as the gate+up launch. top_k=1 makes token_id the
    # expanded activation row; output_topk folds expert slots back onto M.
    expert_gemm(
        activated,
        out,
        w2["qweight"],
        w2["scales"],
        w2["qzeros"],
        topk_weights.reshape(-1),
        sorted_ids,
        expert_ids,
        npost,
        1,
        block_size_m,
        mul_topk_weight=True,
        output_topk=topk,
        fp32_accum=fp32_accum,
    )
    return out


def stack_experts(parts: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Stack per-expert ``prepare_expert`` dicts into a resident table."""
    return {
        "qweight": torch.stack([p["qweight"] for p in parts], 0).contiguous(),
        "scales": torch.stack([p["scales"] for p in parts], 0).contiguous(),
        "qzeros": torch.stack([p["qzeros"] for p in parts], 0).contiguous(),
    }
