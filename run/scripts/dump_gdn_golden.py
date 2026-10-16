"""Extract layer 0's real GatedDeltaNet weights from the downloaded
safetensors, run the REAL transformers forward() on a small deterministic
input, and dump everything as raw f32 binary for the Rust side to compare
against (`run/tests/gated_delta_golden.rs`). No full-model instantiation
- just this one nn.Module, real weights, real math.

Needs: a venv with torch + transformers (5.17.0+, for
transformers.models.qwen3_5) — e.g.:
  python3 -m venv /tmp/glia-verify
  /tmp/glia-verify/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
  /tmp/glia-verify/bin/pip install transformers safetensors
  /tmp/glia-verify/bin/python run/scripts/dump_gdn_golden.py
And the real heretic-org/Qwen3.8-27B-heretic-ara snapshot downloaded
locally (DIR below) — `mi download heretic-org/Qwen3.8-27B-heretic-ara`.
"""
import json, struct, sys
import torch
from safetensors.torch import load_file

DIR = "/Users/master/.cache/huggingface/hub/models--heretic-org--Qwen3.8-27B-heretic-ara/snapshots/2dc9b364104881cbb85e390f00195ba6b9d745e9"
OUT = "/tmp/gdn_golden"
LAYER = 0  # layer_types[0] == "linear_attention"

torch.manual_seed(1234)

with open(f"{DIR}/model.safetensors.index.json") as f:
    index = json.load(f)
weight_map = index["weight_map"]

prefix = f"model.language_model.layers.{LAYER}.linear_attn."
needed = {k: v for k, v in weight_map.items() if k.startswith(prefix)}
print("tensors needed:", sorted(needed.keys()))

by_shard = {}
for name, shard in needed.items():
    by_shard.setdefault(shard, []).append(name)

weights = {}
for shard, names in by_shard.items():
    shard_data = load_file(f"{DIR}/{shard}")
    for n in names:
        weights[n] = shard_data[n].float()
        print(n, tuple(weights[n].shape), weights[n].dtype)

with open(f"{DIR}/config.json") as f:
    config = json.load(f)
tc = config.get("text_config", config)

from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5GatedDeltaNet
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

cfg = Qwen3_5TextConfig(
    hidden_size=tc["hidden_size"],
    num_hidden_layers=tc["num_hidden_layers"],
    linear_num_value_heads=tc["linear_num_value_heads"],
    linear_num_key_heads=tc["linear_num_key_heads"],
    linear_key_head_dim=tc["linear_key_head_dim"],
    linear_value_head_dim=tc["linear_value_head_dim"],
    linear_conv_kernel_dim=tc["linear_conv_kernel_dim"],
    hidden_act=tc["hidden_act"],
    rms_norm_eps=tc["rms_norm_eps"],
    layer_types=tc["layer_types"],
)

layer = Qwen3_5GatedDeltaNet(cfg, layer_idx=LAYER)
sd = {k[len(prefix):]: v for k, v in weights.items()}
missing, unexpected = layer.load_state_dict(sd, strict=False)
print("missing:", missing, "unexpected:", unexpected)
layer.eval()

T = 6
hidden = torch.randn(1, T, tc["hidden_size"], dtype=torch.float32) * 0.02

with torch.no_grad():
    out = layer(hidden)

def dump(path, t):
    arr = t.detach().contiguous().to(torch.float32).numpy()
    with open(path, "wb") as f:
        for d in arr.shape:
            f.write(struct.pack("<Q", d))
        f.write(struct.pack("<Q", 0))  # shape terminator
        f.write(arr.tobytes())

dump(f"{OUT}/input.bin", hidden[0])
dump(f"{OUT}/output.bin", out[0])
for name, t in weights.items():
    short = name[len(prefix):]
    dump(f"{OUT}/w_{short}.bin", t)

print("OK, dumped to", OUT)
print("output stats: mean", out.mean().item(), "std", out.std().item(), "abs_max", out.abs().max().item())
