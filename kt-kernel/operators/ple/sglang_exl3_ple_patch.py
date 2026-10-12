import importlib.util
import json
import os
import sys
import torch
from safetensors import safe_open

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

_EXL3_PATH = os.path.join(_HERE, "exl3_ple_sglang.py")
if "exl3_ple_sglang" not in sys.modules:
    _spec = importlib.util.spec_from_file_location("exl3_ple_sglang", _EXL3_PATH)
    _mod = importlib.util.module_from_spec(_spec)
    sys.modules["exl3_ple_sglang"] = _mod
    _spec.loader.exec_module(_mod)
from exl3_ple_sglang import Exl3PackedNGramEmbedding

from sglang.srt.models.qwen4_exp import Qwen4ExpNGramEmbedding


def _is_exl3_pack(model_path: str) -> bool:
    idx_path = f"{model_path}/model.safetensors.index.json"
    if not os.path.exists(idx_path):
        return False
    with open(idx_path) as f:
        idx = json.load(f)
    wm = idx.get("weight_map", {})
    has_shards = any("shard_0.trellis" in k for k in wm)
    has_weight = any(k.endswith("ngram_embedding.weight") for k in wm)
    return has_shards and not has_weight


def _load_exl3_metadata(model_path: str) -> dict:
    with open(f"{model_path}/model.safetensors.index.json") as f:
        idx = json.load(f)
    sample_key = next(k for k in idx["weight_map"] if "shard_0.trellis" in k)
    sf_path = f"{model_path}/{idx['weight_map'][sample_key]}"
    layer_idx = int(sample_key.split(".")[3])
    with safe_open(sf_path, framework="pt", device="cpu") as f:
        base = f"model.language_model.layers.{layer_idx}.ple.ple_embedding.ngram_embedding"
        head_bias = f.get_tensor(f"{base}.head_bias")
        head_offsets = f.get_tensor(f"{base}.head_offsets").to(torch.int64)
        head_vocab_sizes = f.get_tensor(f"{base}.head_vocab_sizes").to(torch.int64)
        layer_multipliers = f.get_tensor(f"{base}.layer_multipliers").to(torch.int64)
    ngram_size = layer_multipliers.shape[0]
    H = head_offsets.shape[0]
    return {
        "head_bias": head_bias,
        "head_offsets": head_offsets,
        "head_vocab_sizes": head_vocab_sizes,
        "layer_multipliers": layer_multipliers,
        "ngram_size": ngram_size,
        "ngram_heads": H,
        "layer_idx": layer_idx,
    }


def _exl3_embed_ngram_ids(self, ngram_ids: torch.Tensor, forward_batch, physical_tokens: int) -> torch.Tensor:
    """EXL3 path: hash + gather + dequant via Exl3PackedNGramEmbedding.
    sglang's compute_ngram_ids is kept and produces per-position ngram_ids; this
    layer just routes the lookup to the packed table.
    """
    lookup_ids, semantic_tokens = self._prepare_embedding_lookup(
        ngram_ids, forward_batch, physical_tokens
    )
    embeddings = self.exl3_embedding(lookup_ids)
    return self._finish_embedding_lookup(
        embeddings, semantic_tokens, forward_batch, physical_tokens
    )


def patch_qwen4_exp_ple_to_exl3(model_path: str) -> None:
    if not _is_exl3_pack(model_path):
        for cand in (model_path, "/home/kletorch/models/qwen38-exl3", os.environ.get("PACK", "")):
            if cand and _is_exl3_pack(cand):
                model_path = cand
                break
        else:
            print(f"[exl3_ple] not EXL3 pack at {model_path} (no shard_0.trellis); no-op")
            return
    meta = _load_exl3_metadata(model_path)

    _orig_init = Qwen4ExpNGramEmbedding.__init__

    def _patched_init(self, *args, **kwargs):
        _orig_init(self, *args, **kwargs)
        dev = self.ngram_heads_vocab_sizes.device
        self.exl3_embedding = Exl3PackedNGramEmbedding(
            model_path, meta["head_bias"], meta["layer_idx"]
        ).to(dev)
        # Overwrite the synthesized prime-based heads with the EXL3 pack's actual
        # head layout. .data.copy_() avoids the register_buffer "already exists" guard.
        with torch.no_grad():
            self.ngram_heads_offsets.data.copy_(meta["head_offsets"].to(dev))
            self.ngram_heads_vocab_sizes.data.copy_(meta["head_vocab_sizes"].to(dev))
            self.layer_multipliers.data.copy_(meta["layer_multipliers"].to(dev))
        self.ngram_size = meta["ngram_size"]
        self.ngram_heads = meta["ngram_heads"]

    Qwen4ExpNGramEmbedding.__init__ = _patched_init
    Qwen4ExpNGramEmbedding._embed_ngram_ids = _exl3_embed_ngram_ids
    Qwen4ExpNGramEmbedding.is_exl3 = True
    print(f"[exl3_ple] patched Qwen4ExpNGramEmbedding for {model_path} (ngram_size={meta['ngram_size']}, heads={meta['ngram_heads']}, layer={meta['layer_idx']})")


if __name__ == "__main__":
    import sys
    pack = sys.argv[1] if len(sys.argv) > 1 else "/home/kletorch/models/qwen38-exl3"
    patch_qwen4_exp_ple_to_exl3(pack)
    print("patch applied (or no-op if not EXL3)")
