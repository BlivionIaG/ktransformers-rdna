"""Selector tests. No GPU and no compiled extension required."""

import os

import pytest

from ktransformers_rdna.arch import (
    attention_kind,
    select_attention_backend,
    select_linear_backend,
    uses_dot_kernels,
)
from ktransformers_rdna.attention import can_use
from ktransformers_rdna.gemv import can_implement as gemv_can
from ktransformers_rdna.linear import can_implement as linear_can
from ktransformers_rdna.moe import align_blocks, can_implement as moe_can


@pytest.mark.parametrize("arch", ["gfx1030", "gfx1031", "gfx1100", "gfx1101"])
def test_dot_archs(arch):
    assert uses_dot_kernels(arch)


@pytest.mark.parametrize("arch", ["gfx942", "gfx90a", "gfx1150", "gfx1200", None])
def test_other_archs_fall_back(arch):
    assert not uses_dot_kernels(arch)
    assert (
        select_attention_backend(arch=arch, head_dim=128, num_heads=64, num_kv_heads=4)
        == "fallback"
    )
    assert (
        select_linear_backend(arch=arch, weight_format="gptq_int4", kind="dense")
        == "fallback"
    )


def test_mla_is_a_phase1_gap():
    assert attention_kind(model_type="deepseek_v3") == "mla"
    assert attention_kind(attn_type="mla") == "mla"
    assert (
        select_attention_backend(
            arch="gfx1030",
            model_type="deepseek_v3",
            head_dim=128,
            num_heads=16,
            num_kv_heads=16,
        )
        == "fallback"
    )
    ok, reason = can_use(
        head_dim=128,
        num_heads=16,
        num_kv_heads=16,
        model_type="deepseek_v4",
        arch="gfx1030",
    )
    assert not ok and reason == "fallback"


def test_qwen3_gqa_selects_fa():
    assert (
        select_attention_backend(
            arch="gfx1030",
            model_type="qwen3_moe",
            head_dim=128,
            num_heads=64,
            num_kv_heads=4,
        )
        == "rdna_fa"
    )
    assert (
        select_attention_backend(
            arch="gfx1100",
            head_dim=256,
            num_heads=24,
            num_kv_heads=4,
        )
        == "rdna_fa"
    )


def test_attention_shape_limits():
    assert (
        select_attention_backend(arch="gfx1030", head_dim=64, num_heads=8, num_kv_heads=2)
        == "fallback"
    )
    assert (
        select_attention_backend(arch="gfx1030", head_dim=128, num_heads=6, num_kv_heads=4)
        == "fallback"
    )


def test_fa_env_disables(monkeypatch):
    monkeypatch.setenv("KT_USE_RDNA2_FA", "0")
    assert (
        select_attention_backend(arch="gfx1030", head_dim=128, num_heads=8, num_kv_heads=2)
        == "fallback"
    )


def test_quant_backend_choice():
    assert (
        select_linear_backend(arch="gfx1030", weight_format="gptq_int4", kind="dense")
        == "rdna_q_gemm"
    )
    assert (
        select_linear_backend(arch="gfx1100", weight_format="gptq-int4", kind="expert")
        == "rdna_moe_q_gemm"
    )
    assert (
        select_linear_backend(
            arch="gfx1030", weight_format="float16", kind="dense", m=4
        )
        == "rdna_gemv"
    )
    assert (
        select_linear_backend(
            arch="gfx1030", weight_format="float16", kind="dense", m=32
        )
        == "fallback"
    )
    for fmt in ("marlin", "awq_int4", "gguf_q4", "fp8", "mxfp4"):
        assert (
            select_linear_backend(arch="gfx1030", weight_format=fmt, kind="dense")
            == "fallback"
        )
        assert (
            select_linear_backend(arch="gfx1030", weight_format=fmt, kind="expert")
            == "fallback"
        )


def test_can_implement_shapes():
    ok, _ = linear_can(
        weight_format="gptq_int4", k=4096, n=9216, group_size=128, arch="gfx1030"
    )
    assert ok
    ok, _ = linear_can(
        weight_format="gptq_int4", k=4096, n=7, group_size=128, arch="gfx1030"
    )
    assert not ok
    ok, _ = moe_can(
        weight_format="gptq_int4", k=1536, n=4096, group_size=128, arch="gfx1100"
    )
    assert ok
    ok, _ = gemv_can(m=1, k=4096, n=4096, arch="gfx1030")
    assert ok
    ok, _ = gemv_can(m=9, k=4096, n=32, arch="gfx1030")
    assert not ok


def test_align_blocks_padding():
    import torch

    ids = torch.tensor([[0, -1], [0, 1]], dtype=torch.int32)
    sorted_ids, expert_ids, npost = align_blocks(ids, block_size_m=2)
    assert int(npost) % 2 == 0
    assert -1 not in expert_ids.tolist()
    assert set(expert_ids.tolist()) <= {0, 1}
    # Token 0 slot 1 is cold (-1) and must not appear as a real row.
    real = [t for t in sorted_ids.tolist() if t < ids.numel()]
    assert 1 not in real


def test_no_wmma_sources():
    root = os.path.join(os.path.dirname(__file__), "..", "csrc")
    for dirpath, _, files in os.walk(root):
        for name in files:
            text = open(os.path.join(dirpath, name), encoding="utf-8").read().lower()
            assert "amdgcn_wmma" not in text
            assert "q_gemm_rdna3_wmma" not in text
            assert "mfma" not in text or "no wmma, mfma" in text or "no wmma, no mfma" in text
