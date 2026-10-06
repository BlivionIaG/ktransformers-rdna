"""CPU checks of the packing layout and the PyTorch reference."""

import torch

from ktransformers_rdna.layout import (
    dequant_gptq_int4,
    pack_k_nibbles,
    prepare_gptq_int4_tensors,
    shuffle_4bit_u32,
    unpack_k_nibbles,
)
from ktransformers_rdna.reference import gemv_f16, gptq_linear, moe_gptq_int4, paged_attention


def test_shuffle_matches_exllama_bit_pattern():
    # nibbles 0..7 in little-endian order inside one uint32.
    qa = 0
    for i, nibble in enumerate(range(8)):
        qa |= nibble << (4 * i)
    out = int(shuffle_4bit_u32(torch.tensor([qa], dtype=torch.int32))[0]) & 0xFFFFFFFF
    # shuffle_4bit_8: even nibbles in the low half, odd nibbles in the high half.
    # bits: n0, n2, n4, n6, n1, n3, n5, n7
    expected = 0
    order = [0, 2, 4, 6, 1, 3, 5, 7]
    for i, nibble in enumerate(order):
        expected |= nibble << (4 * i)
    assert out == expected


def test_pack_roundtrip():
    q = torch.randint(0, 16, (32, 16), dtype=torch.int32)
    assert torch.equal(unpack_k_nibbles(pack_k_nibbles(q)), q)


def test_prepare_shapes_and_symmetric_zero():
    k, n, group = 64, 16, 32
    q = torch.randint(0, 16, (k, n), dtype=torch.int32)
    scales = torch.rand(k // group, n)
    prepared = prepare_gptq_int4_tensors(q, scales)
    assert prepared["qweight"].shape == (k // 8, n)
    assert prepared["qzeros"].shape == (k // group, n // 8)
    assert prepared["scales"].dtype == torch.float16
    assert prepared["g_idx"].numel() == 0
    assert prepared["use_v2_format"] is False
    # uint4b8 stored zero is 7 (kernel adds 1).
    z0 = int(prepared["qzeros"][0, 0]) & 0xF
    assert z0 == 7


def test_linear_reference_matches_matmul():
    k, n, group, m = 64, 16, 32, 3
    q = torch.randint(0, 16, (k, n), dtype=torch.int32)
    scales = torch.rand(k // group, n)
    x = torch.randn(m, k)
    y = gptq_linear(x, q, scales)
    w = dequant_gptq_int4(q, scales)
    ref = (x.float() @ w).to(y.dtype)
    torch.testing.assert_close(y, ref)


def test_gemv_reference():
    x = torch.randn(2, 32, dtype=torch.float16)
    w = torch.randn(8, 32, dtype=torch.float16)
    bias = torch.randn(8, dtype=torch.float16)
    y = gemv_f16(x, w, bias)
    ref = (x.float() @ w.float().t() + bias.float()).half()
    torch.testing.assert_close(y, ref)


def test_moe_reference_skips_cold_experts():
    e, m, k, inter, group, topk = 2, 1, 64, 32, 32, 2
    hidden = torch.randn(m, k, dtype=torch.float16)
    w13 = torch.randint(0, 16, (e, k, 2 * inter), dtype=torch.int32)
    w13_s = torch.rand(e, k // group, 2 * inter)
    w2 = torch.randint(0, 16, (e, inter, k), dtype=torch.int32)
    w2_s = torch.rand(e, inter // group, k)
    ids = torch.tensor([[0, -1]], dtype=torch.int32)
    weights = torch.tensor([[0.5, 0.5]], dtype=torch.float32)
    y = moe_gptq_int4(hidden, w13, w13_s, w2, w2_s, ids, weights)
    only = moe_gptq_int4(
        hidden, w13, w13_s, w2, w2_s, torch.tensor([[0]], dtype=torch.int32), torch.ones(1, 1)
    )
    torch.testing.assert_close(y, (only.float() * 0.5).half())


def test_paged_reference_matches_sdpa():
    heads, kv_heads, dim, x, block, length = 4, 2, 16, 8, 8, 8
    q = torch.randn(length, heads, dim, dtype=torch.float16)
    blocks = 1
    key_cache = torch.randn(blocks, kv_heads, dim // x, block, x, dtype=torch.float16)
    value_cache = torch.randn_like(key_cache)
    block_table = torch.zeros(1, 1, dtype=torch.int32)
    seq_lens = torch.tensor([length], dtype=torch.int32)
    cu = torch.tensor([0, length], dtype=torch.int32)
    got = paged_attention(q, key_cache, value_cache, block_table, seq_lens, cu, causal=True)
    # [H, D/x, block, x] -> [block, H, D] with d at [d // x, d % x].
    k = key_cache[0].permute(2, 0, 1, 3).reshape(block, kv_heads, dim)
    v = value_cache[0].permute(2, 0, 1, 3).reshape(block, kv_heads, dim)
    k = k.repeat_interleave(heads // kv_heads, dim=1).float()
    v = v.repeat_interleave(heads // kv_heads, dim=1).float()
    scores = torch.einsum("qhd,khd->qhk", q.float(), k) * (dim ** -0.5)
    causal = torch.arange(length)[:, None] >= torch.arange(length)[None, :]
    scores = scores.masked_fill(~causal[:, None, :], -1e9)
    ref = torch.einsum("qhk,khd->qhd", torch.softmax(scores, dim=-1), v).half()
    torch.testing.assert_close(got, ref, rtol=1e-4, atol=1e-4)
