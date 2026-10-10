"""PyTorch references for the hardware numerical tests.

These run on CPU or GPU. They follow the kernel math, not the bit-trick,
so fp16 kernel results are compared with a relative tolerance.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .layout import dequant_gptq_int4


def gptq_linear(
    x: torch.Tensor,
    q_kn: torch.Tensor,
    scales_gn: torch.Tensor,
    zeros_gn: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    w = dequant_gptq_int4(q_kn, scales_gn, zeros_gn)
    y = x.to(torch.float32) @ w
    if bias is not None:
        y = y + bias.to(torch.float32)
    return y.to(x.dtype)


def gemv_f16(x: torch.Tensor, w: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
    y = x.to(torch.float32) @ w.to(torch.float32).t()
    if bias is not None:
        y = y + bias.to(torch.float32)
    return y.to(torch.float16)


def moe_gptq_int4(
    hidden: torch.Tensor,
    w13_kn: torch.Tensor,
    w13_scales: torch.Tensor,
    w2_kn: torch.Tensor,
    w2_scales: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
) -> torch.Tensor:
    """``w13_kn`` is [E, K, 2I] raw nibbles, ``w2_kn`` is [E, I, K]."""
    m, _k = hidden.shape
    out = torch.zeros(m, hidden.shape[-1], dtype=torch.float32, device=hidden.device)
    for row in range(m):
        for slot in range(topk_ids.shape[1]):
            expert = int(topk_ids[row, slot])
            if expert < 0:
                continue
            gate_up = gptq_linear(
                hidden[row : row + 1], w13_kn[expert], w13_scales[expert]
            )
            gate, up = gate_up.chunk(2, dim=-1)
            mid = F.silu(gate) * up
            down = gptq_linear(mid, w2_kn[expert], w2_scales[expert])
            out[row] += down[0].to(torch.float32) * float(topk_weights[row, slot])
    return out.to(torch.float16)


def paged_attention(
    q: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    cu_query_lens: torch.Tensor,
    *,
    causal: bool = False,
    sliding_window: int = 0,
    scale: float | None = None,
) -> torch.Tensor:
    """Streaming fp32 reference. Cache layout matches fa_rdna2."""
    num_heads = q.shape[1]
    head_dim = q.shape[2]
    num_seqs = block_table.shape[0]
    num_kv = key_cache.shape[1]
    block_size = key_cache.shape[3]
    group = num_heads // num_kv
    if scale is None:
        scale = head_dim ** -0.5
    out = torch.zeros_like(q, dtype=torch.float32)
    for s in range(num_seqs):
        length = int(seq_lens[s])
        blocks = block_table[s].tolist()
        keys = []
        values = []
        for pos in range(length):
            blk = blocks[pos // block_size]
            off = pos % block_size
            # [H_kv, D/x, x] -> [H_kv, D]
            # [H_kv, D/x, x] in row-major order is element d at [d // x, d % x].
            k_row = key_cache[blk, :, :, off, :].reshape(num_kv, head_dim)
            v_row = value_cache[blk, :, :, off, :].reshape(num_kv, head_dim)
            keys.append(k_row)
            values.append(v_row)
        k = torch.stack(keys, 0).repeat_interleave(group, dim=1).float()
        v = torch.stack(values, 0).repeat_interleave(group, dim=1).float()
        q0 = int(cu_query_lens[s])
        q1 = int(cu_query_lens[s + 1])
        qf = q[q0:q1].float()
        scores = torch.einsum("qhd,khd->qhk", qf, k) * scale
        q_pos = torch.arange(q0, q1, device=q.device)
        k_pos = torch.arange(length, device=q.device)
        # Queries are the last (q1-q0) positions of the KV sequence.
        q_abs = k_pos[-1] - (q1 - q0 - 1) + (q_pos - q0)
        if causal:
            mask = k_pos.unsqueeze(0) <= q_abs.unsqueeze(1)
            scores = scores.masked_fill(~mask.unsqueeze(1), -1e9)
        if sliding_window > 0:
            window = k_pos.unsqueeze(0) > (q_abs.unsqueeze(1) - sliding_window)
            if causal:
                window = window & (k_pos.unsqueeze(0) <= q_abs.unsqueeze(1))
            scores = scores.masked_fill(~window.unsqueeze(1), -1e9)
        probs = torch.softmax(scores, dim=-1)
        out[q0:q1] = torch.einsum("qhk,khd->qhd", probs, v)
    return out.to(torch.float16)
