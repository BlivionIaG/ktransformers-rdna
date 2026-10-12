"""build a 1-layer Qwen3.5+MTP-head checkpoint for sglang's NEXTN.

Takes layer 0 from the main bf16 overlay and the MTP head from the EXL3 pack.
Layer 0's safetensors keys are reused as-is (the sglang MTP class is Qwen3_5ForCausalLM
with num_hidden_layers=1, so it expects the same layer.0.* tensor names as the main model).
"""
import json
import os
import shutil
import subprocess
from pathlib import Path
import torch
from safetensors import safe_open
from safetensors.torch import save_file

MAIN_BF16 = Path("/home/kletorch/models/qwen38-bf16")
EXL3_PACK = Path("/home/kletorch/models/qwen38-exl3")
OUT = Path("/home/kletorch/models/qwen38-mtp-shim-v2")

shutil.rmtree(OUT, ignore_errors=True)
OUT.mkdir(parents=True, exist_ok=True)

print("reading main bf16 model...")
tensors = {}
for sf in sorted(MAIN_BF16.glob("model-*.safetensors")):
    with safe_open(str(sf), framework="pt", device="cpu") as f:
        for k in f.keys():
            if k.startswith("model.language_model.layers.0.") or k.startswith("model.language_model.norm") or k.startswith("model.language_model.embed_tokens") or k.startswith("model.language_model.lm_head") or k.startswith("model.language_model.mtp_hyper_connection_mixer"):
                new_k = "model.mtp.model." + k[len("model."):]
                tensors[new_k] = f.get_tensor(k)
            elif k.startswith("mtp.layers.0."):
                new_k = "model.mtp.model.language_model.layers.0." + k[len("mtp.layers.0."):]
                tensors[new_k] = f.get_tensor(k)
print(f"loaded {len(tensors)} tensors from main bf16")
print("sample keys:", list(tensors.keys())[:5])

print("reading MTP head from EXL3 pack...")
with safe_open(str(EXL3_PACK / "mtp_hyper_connection_mixer_patch.safetensors"), framework="pt", device="cpu") as f:
    for k in f.keys():
        v = f.get_tensor(k)
        new_k = k
        if k == "mtp.hyper_connection_mixer.hc_norm.weight":
            new_k = "model.mtp.pre_fc_norm_embedding.weight"
        elif k == "mtp.hyper_connection_mixer.input_mix_weight_down.weight":
            new_k = "model.mtp.fc.weight"
        elif k == "mtp.hyper_connection_mixer.input_mix_weight_up.weight":
            new_k = "model.mtp.pre_fc_norm_hidden.weight"
        if new_k == "model.mtp.fc.weight":
            tensors[new_k] = v.t().contiguous()
        else:
            tensors[new_k] = v
        print(f"  {k} -> {new_k}: {tuple(v.shape)}")

if "mtp.fc.weight" not in tensors:
    print("WARNING: mtp.fc.weight not set")

config = {
    "architectures": ["Qwen4ExpForCausalLMMTP"],
    "model_type": "qwen4_exp",
    "hidden_size": 4096,
    "intermediate_size": 12288,
    "num_attention_heads": 32,
    "num_hidden_layers": 1,
    "num_key_value_heads": 4,
    "max_position_embeddings": 262144,
    "vocab_size": 2500016,
    "torch_dtype": "bfloat16",
    "transformers_version": "4.45.0",
    "tie_word_embeddings": False,
    "mtp_num_tokens": 1,
    "mtp_hyper_connection_enabled": True,
    "mtp_input_mix_dim": 320,
    "speculative_eagle_topk": 1,
    "speculative_num_draft_tokens": 3,
    "num_experts": 512,
    "num_experts_per_tok": 8,
    "moe_intermediate_size": 2048,
    "shared_expert_intermediate_size": 0,
    "num_fused_shared_experts": 0,
    "full_attention_interval": 1,
    "use_moe": True,
    "rotary_dim": 128,
    "rope_theta": 1000000.0,
}
(OUT / "config.json").write_text(json.dumps(config, indent=2))

for name in ["tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "special_tokens_map.json"]:
    src = EXL3_PACK / name
    if src.exists():
        shutil.copy2(src, OUT / name)

out_file = OUT / "model.safetensors"
print(f"writing {out_file} with {len(tensors)} tensors...")
save_file(tensors, str(out_file), metadata={"format": "pt"})
print(f"wrote {out_file} ({out_file.stat().st_size // 1024 // 1024} MB)")

print()
print("=== shim ready ===")
for p in sorted(OUT.iterdir()):
    print(f"  {p.name}  {p.stat().st_size // 1024}KB")
