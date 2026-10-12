"""sglang patch: replace Qwen4ExpNGramEmbedding's VocabParallelEmbedding with an
EXL3-packed table lookup when the on-disk format is the EXL3 int16-trellis shards.

Apply this patch to <venv>/lib/python3.12/site-packages/sglang/srt/models/qwen4_exp.py
by replacing the `Qwen4ExpNGramEmbedding` class. The packed table is loaded
once at init time (one-time 24s for 32 GB on first model load), then each forward
pass hashes the n-gram IDs, gathers the packed rows, and dequantizes to bf16.
"""
import torch
from safetensors import safe_open

MUL1 = 0x83DCD12D
ROW_DIM = 160
NUM_SHARDS = 128
SHARD_COLS = 51
NGRAM_VOCAB_PER_SHARD = 2500012
EOS_TOKEN_DEFAULT = 248044
K_DEFAULT = 5


def _mul1_codebook(device):
    s = torch.arange(65536, dtype=torch.int64, device=device)
    prod = (s * MUL1) & 0xFFFFFFFF
    bsum = (prod & 255) + ((prod >> 8) & 255) + ((prod >> 16) & 255) + ((prod >> 24) & 255)
    h = (1024 + bsum).float()
    k_inv = torch.tensor([0x1eee], dtype=torch.uint16).view(torch.float16).float().item()
    k_bias = torch.tensor([0xc931], dtype=torch.uint16).view(torch.float16).float().item()
    return (h * k_inv + k_bias).to(torch.float16)


def _ring_dim(words, K):
    return (words - 1) * 16 // K


def _unpack(packed, K):
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


def _hadamard(x, block):
    shape = x.shape
    n = shape[-1]
    if n != block:
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


def _dequant(packed, K=5):
    states, scales = _unpack(packed, K)
    out = _mul1_codebook(packed.device)[states].float() * scales.float().unsqueeze(1)
    return _hadamard(out, 16).to(torch.bfloat16)


class Exl3PackedNGramEmbedding(torch.nn.Module):
    """Drop-in replacement for VocabParallelEmbedding when the PLE table is in
    the EXL3 int16-trellis packed format. The forward(path) takes a (bsz, total_heads)
    int64 of row indices (computed by compute_ngram_ids) and returns (bsz, total_heads, dim) bf16.
    """

    def __init__(self, pack_path: str, head_bias: torch.Tensor, layer_idx: int, K: int = K_DEFAULT):
        super().__init__()
        self.K = K
        self.register_buffer("head_bias", head_bias)              # (H, dim) fp16
        with safe_open(f"{pack_path}/ngram_embedding.safetensors", framework="pt", device="cpu") as f:
            base = f"model.language_model.layers.{layer_idx}.ple.ple_embedding.ngram_embedding"
            shards = [f.get_tensor(f"{base}.shard_{s}.trellis").pin_memory() if torch.cuda.is_available()
                      else f.get_tensor(f"{base}.shard_{s}.trellis") for s in range(NUM_SHARDS)]
        stacked = torch.cat(shards, dim=0)                      # (320_001_536, 51) int16
        self.register_buffer("packed", stacked, persistent=False)
        self.dim = _ring_dim(SHARD_COLS, K)

    def allocate_output(self, shape, device):
        return torch.empty(shape, dtype=torch.bfloat16, device=device)

    def reduce(self, output):
        # No TP reduce for a single-process model
        return output

    def gather(self, row_idx: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        """PLE prefetch path. Mimics Qwen4ExpPinnedHostEmbedding.gather(lookup_ids, out=out_view).
        The model calls this on a non-default stream; we must avoid non-pinned CPU<->GPU copies."""
        N = row_idx.numel()
        H = self.head_bias.shape[0]
        if N == 0:
            return out
        if not hasattr(self, "_row_buf") or self._row_buf.numel() < N:
            self._row_buf = torch.empty(N, dtype=torch.int64, pin_memory=True)
            self._deq_buf = torch.empty(N, self.dim, dtype=torch.bfloat16, pin_memory=True)
        row_buf = self._row_buf[:N]
        deq_buf = self._deq_buf[:N]
        row_buf.copy_(row_idx.reshape(-1), non_blocking=True)
        rows = self.packed[row_buf]
        deq = _dequant(rows, K=self.K)
        head_idx = torch.arange(N) % H
        torch.add(deq, self.head_bias[head_idx], out=deq_buf)
        # H2D from pinned is capture-safe
        out.copy_(deq_buf.view(out.shape), non_blocking=True)
        return out

    def forward(self, row_idx: torch.Tensor) -> torch.Tensor:
