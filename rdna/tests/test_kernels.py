"""Numerical checks against the PyTorch reference.

These launch the HIP kernels and need a gfx1030 or gfx1100 GPU. The cloud
build VM has no AMD GPU, so pytest skips them there. On hardware:

    cd rdna
    PYTORCH_ROCM_ARCH=gfx1030;gfx1100 python setup.py build_ext --inplace
    PYTHONPATH=. python -m pytest tests/test_kernels.py -q
"""

import pytest
import torch

from ktransformers_rdna import _ops
from ktransformers_rdna.arch import gcn_arch, uses_dot_kernels
from ktransformers_rdna.layout import prepare_gptq_int4_tensors
from ktransformers_rdna.moe import apply_gptq_int4_experts, stack_experts
from ktransformers_rdna.reference import gemv_f16, gptq_linear, moe_gptq_int4, paged_attention


def _hip_ready():
    if not torch.cuda.is_available():
        return False
    if not uses_dot_kernels(gcn_arch()):
        return False
    return _ops.available()


pytestmark = pytest.mark.skipif(
    not _hip_ready(),
    reason="needs a built ktransformers_rdna._C and a gfx1030/gfx1100 GPU",
)


def _close(got, ref):
    torch.testing.assert_close(got.float(), ref.float(), rtol=5e-2, atol=2e-2)


def test_gemv_matches_reference():
    from ktransformers_rdna.gemv import gemv_f16 as kernel

    torch.manual_seed(0)
    x = torch.randn(3, 128, device="cuda", dtype=torch.float16)
    w = torch.randn(32, 128, device="cuda", dtype=torch.float16)
    bias = torch.randn(32, device="cuda", dtype=torch.float16)
    _close(kernel(x, w, bias), gemv_f16(x, w, bias))


@pytest.mark.parametrize("m", [1, 4, 17])
def test_qgemm_matches_reference(m):
    from ktransformers_rdna.linear import apply_gptq_int4

    torch.manual_seed(1)
    k, n, group = 128, 32, 32
    q = torch.randint(0, 16, (k, n), device="cuda")
    scales = torch.rand(k // group, n, device="cuda")
    x = torch.randn(m, k, device="cuda", dtype=torch.float16)
    prepared = prepare_gptq_int4_tensors(q, scales)
    got = apply_gptq_int4(x, prepared)
    _close(got, gptq_linear(x, q, scales))


def test_moe_matches_reference():
    torch.manual_seed(2)
    e, m, k, inter, group, topk = 2, 2, 128, 32, 32, 2
    hidden = torch.randn(m, k, device="cuda", dtype=torch.float16)
    w13 = torch.randint(0, 16, (e, k, 2 * inter), device="cuda")
    w13_s = torch.rand(e, k // group, 2 * inter, device="cuda")
    w2 = torch.randint(0, 16, (e, inter, k), device="cuda")
    w2_s = torch.rand(e, inter // group, k, device="cuda")
    ids = torch.tensor([[0, 1], [1, -1]], device="cuda", dtype=torch.int32)
    weights = torch.tensor([[0.6, 0.4], [1.0, 0.0]], device="cuda")
    parts13 = [prepare_gptq_int4_tensors(w13[i], w13_s[i]) for i in range(e)]
    parts2 = [prepare_gptq_int4_tensors(w2[i], w2_s[i]) for i in range(e)]
    got = apply_gptq_int4_experts(
        hidden,
        stack_experts(parts13),
        stack_experts(parts2),
        ids,
        weights,
        fp32_accum=True,
    )
    ref = moe_gptq_int4(hidden, w13, w13_s, w2, w2_s, ids, weights)
    _close(got, ref)


@pytest.mark.parametrize("head_dim,causal,length", [(128, False, 32), (128, True, 48), (256, True, 32)])
def test_attention_matches_reference(head_dim, causal, length):
    from ktransformers_rdna.attention import paged_attention as kernel

    torch.manual_seed(3)
    heads, kv_heads, x, block = 8, 2, 8, 16
    blocks = (length + block - 1) // block
    q = torch.randn(length, heads, head_dim, device="cuda", dtype=torch.float16)
    key_cache = torch.randn(
        blocks, kv_heads, head_dim // x, block, x, device="cuda", dtype=torch.float16
    )
    value_cache = torch.randn_like(key_cache)
    block_table = torch.arange(blocks, device="cuda", dtype=torch.int32).view(1, blocks)
    seq_lens = torch.tensor([length], device="cuda", dtype=torch.int32)
    cu = torch.tensor([0, length], device="cuda", dtype=torch.int32)
    got = kernel(
        q,
        key_cache,
        value_cache,
        block_table,
        seq_lens,
        cu,
        num_heads=heads,
        num_kv_heads=kv_heads,
        block_size=block,
        causal=causal,
    )
    ref = paged_attention(
        q, key_cache, value_cache, block_table, seq_lens, cu, causal=causal
    )
    _close(got, ref)
