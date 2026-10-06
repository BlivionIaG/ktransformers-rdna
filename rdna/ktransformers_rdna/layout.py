"""GPTQ-Int4 packing for the RDNA DOT kernels.

The kernels consume the exllama shuffle of a K-packed uint32 matrix, the
same layout as ``gptq_shuffle`` in opengfx1030/vllm-rdna
(``shuffle_4bit_8`` from exllamav2, comment ``77775555 33331111 66664444 22220000``).

Dense tensors
    qweight  [K/8, N] int32, shuffled
    scales   [K/group, N] fp16
    qzeros   [K/group, N/8] int32, 8 nibbles along N, little-endian
    g_idx    [K] int32 inverse permutation, or a 0-length tensor

Expert tensors add a leading E dimension. The MoE kernel hard-codes the
GPTQv1 +1 zero offset, so stored zeros are ``bias - 1`` (7 for uint4b8).
"""

from __future__ import annotations

import torch


def shuffle_4bit_u32(qa: torch.Tensor) -> torch.Tensor:
    """Vectorized ``shuffle_4bit_8`` on a uint32 tensor of any shape."""
    qa = qa.to(torch.int64) & 0xFFFFFFFF
    qb = torch.zeros_like(qa)
    for i in range(4):
        qa0 = qa & 0x0F
        qa1 = (qa >> 4) & 0x0F
        qa = qa >> 8
        qb = qb | (qa1 << (i * 4 + 16)) | (qa0 << (i * 4))
    return (qb & 0xFFFFFFFF).to(torch.int32)


def pack_k_nibbles(q_kn: torch.Tensor) -> torch.Tensor:
    """Pack [K, N] nibbles in 0..15 into [K/8, N] int32, low nibble first."""
    if q_kn.dim() != 2:
        raise ValueError("q_kn must be [K, N]")
    k, n = q_kn.shape
    if k % 8 != 0:
        raise ValueError(f"K={k} is not a multiple of 8")
    grouped = q_kn.reshape(k // 8, 8, n).to(torch.int32)
    packed = torch.zeros(k // 8, n, dtype=torch.int32, device=q_kn.device)
    for i in range(8):
        packed = packed | ((grouped[:, i, :] & 0xF) << (4 * i))
    return packed


def pack_n_nibbles(values_gn: torch.Tensor) -> torch.Tensor:
    """Pack [G, N] nibbles into [G, N/8] int32, low nibble = column 0."""
    if values_gn.dim() != 2:
        raise ValueError("values must be [G, N]")
    g, n = values_gn.shape
    if n % 8 != 0:
        raise ValueError(f"N={n} is not a multiple of 8")
    grouped = values_gn.reshape(g, n // 8, 8).to(torch.int32)
    packed = torch.zeros(g, n // 8, dtype=torch.int32, device=values_gn.device)
    for i in range(8):
        packed = packed | ((grouped[:, :, i] & 0xF) << (4 * i))
    return packed


def unpack_k_nibbles(packed: torch.Tensor) -> torch.Tensor:
    """Inverse of ``pack_k_nibbles`` before the shuffle."""
    k8, n = packed.shape
    words = packed.to(torch.int32)
    cols = []
    for i in range(8):
        cols.append((words >> (4 * i)) & 0xF)
    return torch.stack(cols, dim=1).reshape(k8 * 8, n)


def prepare_gptq_int4_tensors(
    q_kn: torch.Tensor,
    scales_gn: torch.Tensor,
    *,
    zeros_gn: torch.Tensor | None = None,
    g_idx: torch.Tensor | None = None,
    sym_bias: int = 8,
) -> dict[str, torch.Tensor]:
    """Build the dense kernel tensors from raw nibbles.

    ``q_kn`` is [K, N] stored integers in 0..15 (checkpoint order, not shuffled).
    ``scales_gn`` is [K/group, N]. Symmetric GPTQ (uint4b8) omits zeros; the
    stored zero is ``sym_bias - 1`` because the kernel adds one when
    ``use_v2_format`` is false.
    """
    if q_kn.dim() != 2 or scales_gn.dim() != 2:
        raise ValueError("q_kn is [K, N] and scales_gn is [groups, N]")
    k, n = q_kn.shape
    groups = scales_gn.shape[0]
    if scales_gn.shape[1] != n:
        raise ValueError("scales N does not match qweight N")
    if k % groups != 0:
        raise ValueError("group size does not divide K")
    if n % 8 != 0 or k % 32 != 0:
        raise ValueError("need K % 32 == 0 and N % 8 == 0")
    if zeros_gn is None:
        zeros_gn = torch.full(
            (groups, n),
            sym_bias - 1,
            dtype=torch.int32,
            device=q_kn.device,
        )
    elif zeros_gn.shape != (groups, n):
        raise ValueError("zeros_gn must be [groups, N] stored zeros")
    qweight = shuffle_4bit_u32(pack_k_nibbles(q_kn.contiguous()))
    qzeros = pack_n_nibbles(zeros_gn.contiguous())
    scales = scales_gn.contiguous().to(torch.float16)
    if g_idx is None:
        perm = torch.empty(0, dtype=torch.int32, device=q_kn.device)
    else:
        perm = torch.argsort(g_idx.to(torch.int64)).to(torch.int32)
    return {
        "qweight": qweight.contiguous(),
        "qzeros": qzeros.contiguous(),
        "scales": scales,
        "g_idx": perm.contiguous(),
        "use_v2_format": False,
        "group_size": k // groups,
    }


def dequant_gptq_int4(
    q_kn: torch.Tensor,
    scales_gn: torch.Tensor,
    zeros_gn: torch.Tensor | None = None,
    *,
    sym_bias: int = 8,
    zero_offset: int = 1,
) -> torch.Tensor:
    """fp32 weight [K, N] matching the kernel's (q - (z + offset)) * scale."""
    k, n = q_kn.shape
    groups = scales_gn.shape[0]
    group = k // groups
    if zeros_gn is None:
        stored = torch.full((groups, n), sym_bias - 1, dtype=torch.float32, device=q_kn.device)
    else:
        stored = zeros_gn.to(torch.float32)
    z = (stored + zero_offset).repeat_interleave(group, dim=0)
    s = scales_gn.to(torch.float32).repeat_interleave(group, dim=0)
    return (q_kn.to(torch.float32) - z) * s
