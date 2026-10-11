"""ExLlamaV3-compatible n-gram PLE for the EXL3 Qwen3.8-Flash-Next pack.

Pure-torch port of turboderp-org/exllamav3/exllamav3/exllamav3_ext/ngram.cu
(hash + gather + dequant) and modules/quant/exl3_lib/ngram_codec.py (codec +
mul1_codebook). Algorithm:

  Hash:    mixed = id_p * mult[0]; for s in 1..ngram_size: mixed ^= id_{p-s} * mult[s]
           (with EOS-segmented backoff: positions before seg_start use eos_token)
  Head:    head = (s-1) * heads_per_ngram + h, h in [0, heads_per_ngram)
  Row:     row = (mixed % sizes[head]) + offsets[head]      # 0..320M
  Dequant: unpack (51 uint16 words) -> states (160,) int16, scale (1,) fp16
           out = codebook[states].float() * scale            # mul1 codebook
           if signs: out = hadamard(out) * signs
           if bias: out = out + bias

On-pack layout: 128 shards of (2500012, 51) int16 stacked into (320001536, 51).
"""
import torch
from safetensors import safe_open

MUL1 = 0x83DCD12D
ROW_DIM = 160
NGRAM_VOCAB = 2500012
NUM_SHARDS = 128
SHARD_COLS = 51
NGRAM_VOCAB_TOTAL = NGRAM_VOCAB * NUM_SHARDS
HEADS_PER_NGRAM_DEFAULT = 16
NGRAM_SIZE_DEFAULT = 4
EOS_TOKEN_DEFAULT = 248044


def _mul1_codebook(device) -> torch.Tensor:
    s = torch.arange(65536, dtype=torch.int64, device=device)
    prod = (s * MUL1) & 0xFFFFFFFF
    bsum = (prod & 255) + ((prod >> 8) & 255) + ((prod >> 16) & 255) + ((prod >> 24) & 255)
    h = (1024 + bsum).float()
    k_inv = torch.tensor([0x1eee], dtype=torch.uint16).view(torch.float16).float().item()
    k_bias = torch.tensor([0xc931], dtype=torch.uint16).view(torch.float16).float().item()
    return (h * k_inv + k_bias).to(torch.float16)


_CODEBOOK: dict[str, torch.Tensor] = {}


def _codebook(device) -> torch.Tensor:
    key = str(device)
    if key not in _CODEBOOK:
        _CODEBOOK[key] = _mul1_codebook(device)
    return _CODEBOOK[key]


def _words_per_row(K: int, dim: int = ROW_DIM) -> int:
    return 1 + dim * K // 16


def _ring_dim(words: int, K: int) -> int:
    return (words - 1) * 16 // K


def _unpack(packed: torch.Tensor, K: int) -> tuple[torch.Tensor, torch.Tensor]:
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


def _hadamard(x: torch.Tensor) -> torch.Tensor:
    shape = x.shape
    n = shape[-1]
    if n & (n - 1):
        block = n & -n
        xx = x.reshape(*shape[:-1], n // block, block)
        h = 1
        while h < block:
            xx = xx.view(*shape[:-1], n // block, block // (2 * h), 2, h)
            a, b = xx[..., 0, :], xx[..., 1, :]
            xx = torch.stack((a + b, a - b), dim=-2)
            h *= 2
        return xx.reshape(shape) * (block ** -0.5)
    xx = x.reshape(-1, n)
    h = 1
    while h < n:
        xx = xx.view(-1, n // (2 * h), 2, h)
        a, b = xx[:, :, 0], xx[:, :, 1]
        xx = torch.stack((a + b, a - b), dim=2)
        h *= 2
    return (xx.reshape(shape) * (n ** -0.5))


def dequant(packed: torch.Tensor, K: int = 5, signs: torch.Tensor | None = None,
            bias: torch.Tensor | None = None) -> torch.Tensor:
    states, scales = _unpack(packed, K)
    out = _codebook(packed.device)[states].float() * scales.float().unsqueeze(1)
    if signs is not None:
        out = _hadamard(out) * signs.float()
    if bias is not None:
        out = out + bias.float()
    return out.to(torch.bfloat16)


def ngram_hash(ids: torch.Tensor, ctx: int, seq_len: int,
               multipliers: torch.Tensor, offsets: torch.Tensor, sizes: torch.Tensor,
               heads_per_ngram: int = HEADS_PER_NGRAM_DEFAULT,
               eos_token: int = EOS_TOKEN_DEFAULT) -> torch.Tensor:
    """Compute per-head row indices for the n-gram table.

    Per turboderp-org/exllamav3/exllamav3_ext/ngram.cu: for each n-gram size s in 0..ngram_size,
    build the (s+1)-token prefix ending at position p, mix it with multipliers[0..s], and emit
    `heads_per_ngram` row indices under that s-block of the head table.

    ids: (bsz, T) int64, multipliers: (ngram_size,) int64, offsets/sizes: (H,) int64.
    Returns: (bsz, seq_len, H) int64 row indices in [0, total_table_rows).
    """
    bsz, T = ids.shape
    H = offsets.shape[0]
    ngram_size = multipliers.shape[0]
    heads_per_ngram = H // ngram_size
    mult = multipliers.tolist()
    offs = offsets.tolist()
    szs = sizes.tolist()
    out = torch.zeros(bsz, seq_len, H, dtype=torch.int64)
    MASK = 0xFFFFFFFFFFFFFFFF
    for b in range(bsz):
        row = ids[b].tolist()
        prev_eos = -1
        for p in range(T):
            if p > 0 and row[p - 1] == eos_token:
                prev_eos = p - 1
            if p < ctx:
                continue
            seg_start = prev_eos + 1
            for s in range(ngram_size):
                mixed = 0
                for inner_s in range(s + 1):
                    sp = p - inner_s
                    src = row[sp] if (sp >= seg_start and sp >= 0) else eos_token
                    mixed = (mixed ^ (src * mult[inner_s])) & MASK
                for hh in range(heads_per_ngram):
                    h = s * heads_per_ngram + hh
                    out[b, p - ctx, h] = (mixed % szs[h]) + offs[h]
    return out


def load_pack(pack_path: str) -> dict:
    """Load the EXL3 PLE pack. Returns dict with shards, head_bias, head_offsets,
    head_vocab_sizes, layer_multipliers, ngram_size, eos_token, heads_per_ngram.
    """
    with safe_open(f"{pack_path}/ngram_embedding.safetensors", framework="pt", device="cpu") as f:
        base = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding"
        head_bias = f.get_tensor(f"{base}.head_bias")
        head_offsets = f.get_tensor(f"{base}.head_offsets")
        head_vocab_sizes = f.get_tensor(f"{base}.head_vocab_sizes")
        layer_multipliers = f.get_tensor(f"{base}.layer_multipliers")
        shards = [f.get_tensor(f"{base}.shard_{s}.trellis") for s in range(NUM_SHARDS)]
    return {
        "shards": torch.cat(shards, dim=0).pin_memory() if torch.cuda.is_available() else torch.cat(shards, dim=0),
        "head_bias": head_bias,
        "head_offsets": head_offsets.to(torch.int64),
        "head_vocab_sizes": head_vocab_sizes.to(torch.int64),
        "layer_multipliers": layer_multipliers.to(torch.int64),
        "ngram_size": NGRAM_SIZE_DEFAULT,
        "eos_token": EOS_TOKEN_DEFAULT,
        "heads_per_ngram": HEADS_PER_NGRAM_DEFAULT,
    }


def ple_lookup(pack: dict, ids: torch.Tensor, ctx: int, signs: torch.Tensor | None = None) -> torch.Tensor:
    """Top-level PLE lookup: token ids -> (bsz, seq_len, H, 160) bf16.

    ids: (bsz, ctx + seq_len) int64 (must include context prefix of ngram_size - 1 tokens
    for each non-first position, OR pass ctx+seq_len >= ngram_size and the first
    ngram_size - 1 positions will use eos_token).
    """
    bsz, T = ids.shape
    seq_len = T - ctx
    H = pack["heads_per_ngram"]
    ngram_size = pack["ngram_size"]
    assert ctx >= ngram_size - 1, f"ctx={ctx} must be >= ngram_size-1={ngram_size-1}"
    row_idx = ngram_hash(
        ids, ctx, seq_len,
        pack["layer_multipliers"],
        pack["head_offsets"],
        pack["head_vocab_sizes"],
        heads_per_ngram=H,
        eos_token=pack["eos_token"],
    )                                                                                # (bsz, seq_len, H)
    flat_idx = row_idx.reshape(-1)                                                    # (bsz * seq_len * H,)
    rows = pack["shards"][flat_idx].to(torch.int16)                                    # (N, 51) int16
    bias = pack["head_bias"]                                                         # (H, 160) fp16
    H_d = bias.shape[0]
    if signs is None:
        signs = None  # hadamard skipped; signs tensor is a codec constant, not in the pack
    bias_p = bias.view(1, H_d, ROW_DIM).expand(flat_idx.numel() // H_d, H_d, ROW_DIM).contiguous().view(-1, ROW_DIM)
    out = dequant(rows, K=5, signs=signs, bias=bias_p)                                # (N, 160) bf16
    out = out.view(bsz, seq_len, H, ROW_DIM)
    return out


if __name__ == "__main__":
    import sys, time
    pack_path = sys.argv[1] if len(sys.argv) > 1 else "/home/kletorch/models/qwen38-exl3"
    print(f"loading pack from {pack_path}...")
    t0 = time.time()
    pack = load_pack(pack_path)
    print(f"  loaded in {time.time()-t0:.1f}s")
    print(f"  shards: {tuple(pack['shards'].shape)} {pack['shards'].dtype}")
    print(f"  layer_multipliers: {pack['layer_multipliers'].tolist()}")
    if torch.cuda.is_available():
        pass  # shards (32 GB) don't fit on the 15 GB GPU; keep on CPU and gather per batch

    print("\ndequantizing +5 random rows for sanity check...")
    t0 = time.time()
    sample = pack["shards"][:5]
    out = dequant(sample, K=5)
    print(f"  dequant in {time.time()-t0:.2f}s, shape {tuple(out.shape)}, |out|max {out.abs().max().item():.4f}")

    print("\nfull hash + lookup on (1, 16) random ids...")
    torch.manual_seed(42)
    bsz, T = 1, 16
    ctx = pack["ngram_size"] - 1
    ids = torch.randint(0, 250000, (bsz, ctx + T), dtype=torch.int64)
    t0 = time.time()
    ple = ple_lookup(pack, ids, ctx)
    print(f"  lookup in {time.time()-t0:.2f}s")
    print(f"  ple shape: {tuple(ple.shape)} {ple.dtype}")
    print(f"  |ple|max: {ple.abs().max().item():.4f}")
    print(f"  row idx first 5: {ple.cpu().view(bsz*T, -1)[0, :5].tolist()}")
    print("OK")
