"""ExLlamaV3-compatible n-gram PLE dequant for the EXL3 Qwen3.8-Flash-Next pack.

Drop-in port of turboderp-org/exllamav3/exllamav3/modules/quant/exl3_lib/ngram_codec.py
to plain torch. The pack stores the n-gram table as 128 shards of (2500012, 51) int16
under `model.language_model.layers.{L}.ple.ple_embedding.ngram_embedding.shard_{S}.trellis`,
plus head_bias (16, 160) fp16, head_offsets (16,) int64, head_vocab_sizes (16,) int64,
and layer_multipliers (3,) int64. The codec decomposes per row:
  (N, dim) int16 stream
  -> states (N, dim) int64, scales (N,) float16 via unpack_rows
  -> out = codebook[states] * scale
  -> optionally: hadamard(out) * signs
  -> + bias
"""
import torch
from safetensors import safe_open
from safetensors.torch import load_file

# ExLlamaV3 constants (verified against turboderp-org/exllamav3 qwen4_exp.py + ngram_codec.py)
MUL1 = 0x83DCD12D
ROW_DIM = 160        # n-gram row
NGRAM_VOCAB = 2500012
NUM_SHARDS = 128
SHARD_COLS = 51      # int16 words per row, i.e. 1 + 160 * 5 // 16 = 51 for K=5


def _mul1_codebook(device) -> torch.Tensor:
    s = torch.arange(65536, dtype=torch.int64, device=device)
    prod = (s * MUL1) & 0xFFFFFFFF
    bsum = (prod & 255) + ((prod >> 8) & 255) + ((prod >> 16) & 255) + ((prod >> 24) & 255)
    h = (1024 + bsum).float()
    k_inv = torch.tensor([0x1eee], dtype=torch.uint16).view(torch.float16).float().item()
    k_bias = torch.tensor([0xc931], dtype=torch.uint16).view(torch.float16).float().item()
    return (h * k_inv + k_bias).to(torch.float16)


_CODEBOOK_CACHE: dict[str, torch.Tensor] = {}


def _get_codebook(device) -> torch.Tensor:
    key = str(device)
    if key not in _CODEBOOK_CACHE:
        _CODEBOOK_CACHE[key] = _mul1_codebook(device)
    return _CODEBOOK_CACHE[key]


def _words_per_row(K: int, dim: int = ROW_DIM) -> int:
    return 1 + dim * K // 16


def _ring_dim(words: int, K: int) -> int:
    return (words - 1) * 16 // K


def _unpack_rows(packed: torch.Tensor, K: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Inverse of pack_rows. (N, words) int16 -> (states (N, dim) int64, scales (N,) float16)."""
    dev = packed.device
    words = packed.shape[1] - 1
    dim = _ring_dim(packed.shape[1], K)
    scales = packed[:, 0].contiguous().view(torch.float16)
    stream = packed[:, 1:].contiguous().view(torch.uint16).to(torch.int64)
    bit = torch.arange(dim, device=dev) * K
    window = stream[:, bit >> 4] | (stream[:, ((bit >> 4) + 1) % words] << 16)
    symbols = (window >> (bit & 15)) & ((1 << K) - 1)
    states = torch.zeros_like(symbols)
    for j in range((15 + K) // K):
        states |= torch.roll(symbols, j, dims=1) << (j * K)
    return states & 0xFFFF, scales


def _hadamard_rotate(x: torch.Tensor) -> torch.Tensor:
    shape = x.shape
    n = shape[-1]
    xx = x.reshape(-1, n)
    h = 1
    while h < n:
        xx = xx.view(-1, n // (2 * h), 2, h)
        xx = torch.stack((xx[:, :, 0] + xx[:, :, 1], xx[:, :, 0] - xx[:, :, 1]), dim=2)
        h *= 2
    return (xx.reshape(shape) * (n ** -0.5))


def dequant_packed_rows(
    packed: torch.Tensor,
    K: int = 5,
    bias: torch.Tensor | None = None,
    signs: torch.Tensor | None = None,
) -> torch.Tensor:
    """Dequantize the EXL3 PLE n-gram rows. (N, 51) int16 -> (N, 160) bf16 (or float16)."""
    states, scales = _unpack_rows(packed, K)
    codebook = _get_codebook(packed.device)
    out = codebook[states].float() * scales.float().unsqueeze(1)
    if signs is not None:
        out = _hadamard_rotate(out) * signs.float()
    if bias is not None:
        out = out + bias.float()
    return out.to(torch.bfloat16)


def _load_exl3_ple_pack(pack_path: str) -> dict:
    """Load the EXL3 n-gram PLE pack: 128 shards stacked to (2500012, 51) + metadata.

    Returns dict with: shards (V, 51) int16, head_bias (H, 160) fp16, head_offsets (H,) i64,
    head_vocab_sizes (H,) i64, layer_multipliers (3,) i64.
    """
    path = f"{pack_path}/ngram_embedding.safetensors"
    base = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding"
    with safe_open(path, framework="pt", device="cpu") as f:
        head_bias = f.get_tensor(f"{base}.head_bias").to(torch.float16)
        head_offsets = f.get_tensor(f"{base}.head_offsets").to(torch.int64)
        head_vocab_sizes = f.get_tensor(f"{base}.head_vocab_sizes").to(torch.int64)
        layer_multipliers = f.get_tensor(f"{base}.layer_multipliers").to(torch.int64)
        shards = [f.get_tensor(f"{base}.shard_{s}.trellis").to(torch.int16) for s in range(NUM_SHARDS)]
    return {
        "shards": torch.cat(shards, dim=0),                    # (2500012, 51) int16
        "head_bias": head_bias,                                # (H, 160) fp16
        "head_offsets": head_offsets,                          # (H,) i64
        "head_vocab_sizes": head_vocab_sizes,                  # (H,) i64
        "layer_multipliers": layer_multipliers,                # (3,) i64
    }


def ple_lookup_token_ids(
    pack: dict,
    token_ids: torch.Tensor,            # (B, T) int64 of token IDs (need n-gram prefix)
    ngram_size: int = 4,                 # pack default; check config later
    K: int = 5,
) -> torch.Tensor:
    """Given a token sequence, return the PLE n-gram embedding per head per position.

    Output: (B, T, H * 160) bf16 (the H heads concatenated).
    For now, returns the dequantized rows for the (T - ngram_size + 1) positions
    starting at index ngram_size - 1, zero-padded before that. The n-gram hash function
    matches ExLlamaV3's compute_ngram_ids (default: xxhash with 64-bit state, but
    to be read from the sglang model_args.n_gram once ported).
    """
    raise NotImplementedError(
        "ple_lookup_token_ids requires the n-gram hash function from "
        "sglang/srt/layers/rotary_embedding/.../ngram.py or equivalent. "
        "Read ngram.compute_ngram_ids in turboderp-org/exllamav3/modules/ngram_embedding.py "
        "(line 250+) and port its xxhash-style token-history hash to torch."
    )


if __name__ == "__main__":
    import sys, time
    pack_path = sys.argv[1] if len(sys.argv) > 1 else "/home/kletorch/models/qwen38-exl3"
    print(f"loading pack from {pack_path}...")
    t0 = time.time()
    pack = _load_exl3_ple_pack(pack_path)
    print(f"  loaded in {time.time()-t0:.1f}s")
    print(f"  shards: {tuple(pack['shards'].shape)} {pack['shards'].dtype}")
    print(f"  head_bias: {tuple(pack['head_bias'].shape)} {pack['head_bias'].dtype}")
    print(f"  head_offsets: {pack['head_offsets'].tolist()}")
    print(f"  head_vocab_sizes: {pack['head_vocab_sizes'].tolist()}")
    print(f"  layer_multipliers: {pack['layer_multipliers'].tolist()}")

    # dequant a slice
    print("\ndequantizing first 5 rows (K=5)...")
    t0 = time.time()
    sample = pack["shards"][:5].cuda()
    out = dequant_packed_rows(sample, K=5)
    print(f"  dequant in {time.time()-t0:.2f}s")
    print(f"  out shape: {tuple(out.shape)} {out.dtype}")
    print(f"  out sample: {out[0, :8].tolist()}")
    print(f"  |out|max: {out.abs().max().item():.4f}")

    # verify K computation matches SHARD_COLS
    assert _words_per_row(5) == SHARD_COLS, f"K=5 should give {SHARD_COLS} words, got {_words_per_row(5)}"
    assert _ring_dim(SHARD_COLS, 5) == ROW_DIM, f"ring_dim mismatch"
    print(f"\n  K=5: words_per_row={_words_per_row(5)} (matches SHARD_COLS={SHARD_COLS}), ring_dim={_ring_dim(SHARD_COLS, 5)} (matches ROW_DIM={ROW_DIM})")
    print("OK")
