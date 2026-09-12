"""Extract real Qwen3.8-27B vision-tower weights (patch_embed, pos_embed,
the first 2 blocks, merger) and run the REAL transformers
Qwen3_5VisionModel forward on a small synthetic-but-correctly-shaped
single image, dumping input+weights+output for the Rust side
(`run/tests/vision_golden.rs`) to compare against.

Only 2 of the real 27 blocks are used (`depth=2` truncation) — enough to
exercise the block/attention/mlp/residual wiring without a 27-block
dump. patch_embed/pos_embed/merger are the real, full-size tensors
(they're shared/config-sized regardless of depth).

Needs: same venv as dump_gdn_golden.py (torch + transformers 5.17.0+).
"""
import json, struct
import torch
from safetensors.torch import load_file

DIR = "/Users/master/.cache/huggingface/hub/models--heretic-org--Qwen3.8-27B-heretic-ara/snapshots/2dc9b364104881cbb85e390f00195ba6b9d745e9"
OUT = "/tmp/vision_golden"
DEPTH = 2

torch.manual_seed(4321)

with open(f"{DIR}/model.safetensors.index.json") as f:
    index = json.load(f)
weight_map = index["weight_map"]

prefix = "model.visual."
wanted_prefixes = ["patch_embed.", "pos_embed.", "merger."] + [f"blocks.{i}." for i in range(DEPTH)]
needed = {
    k: v for k, v in weight_map.items()
    if k.startswith(prefix) and any(k[len(prefix):].startswith(p) for p in wanted_prefixes)
}
print("tensors needed:", len(needed))

by_shard = {}
for name, shard in needed.items():
    by_shard.setdefault(shard, []).append(name)

weights = {}
for shard, names in by_shard.items():
    shard_data = load_file(f"{DIR}/{shard}")
    for n in names:
        weights[n] = shard_data[n].float()

with open(f"{DIR}/config.json") as f:
    config = json.load(f)
vc = config["vision_config"]

from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5VisionConfig

cfg = Qwen3_5VisionConfig(
    depth=DEPTH,
    hidden_size=vc["hidden_size"],
    hidden_act=vc["hidden_act"],
    intermediate_size=vc["intermediate_size"],
    num_heads=vc["num_heads"],
    in_channels=vc["in_channels"],
    patch_size=vc["patch_size"],
    spatial_merge_size=vc["spatial_merge_size"],
    temporal_patch_size=vc["temporal_patch_size"],
    out_hidden_size=vc["out_hidden_size"],
    num_position_embeddings=vc["num_position_embeddings"],
)

model = Qwen3_5VisionModel(cfg)
sd = {k[len(prefix):]: v for k, v in weights.items()}
missing, unexpected = model.load_state_dict(sd, strict=False)
print("missing:", missing)
print("unexpected:", unexpected)
model.eval()

# One 4x4-patch image, single frame — merge_size=2 gives 4 merged output tokens.
grid_h, grid_w, grid_t = 4, 4, 1
num_patches = grid_t * grid_h * grid_w
patch_dim = vc["in_channels"] * vc["temporal_patch_size"] * vc["patch_size"] * vc["patch_size"]
pixel_values = torch.randn(num_patches, patch_dim, dtype=torch.float32) * 0.5
grid_thw = torch.tensor([[grid_t, grid_h, grid_w]], dtype=torch.long)

with torch.no_grad():
    out = model(pixel_values, grid_thw)

merged = out.pooler_output
last_hidden = out.last_hidden_state


def dump(path, t):
    arr = t.detach().contiguous().to(torch.float32).numpy()
    with open(path, "wb") as f:
        for d in arr.shape:
            f.write(struct.pack("<Q", d))
        f.write(struct.pack("<Q", 0))
        f.write(arr.tobytes())


dump(f"{OUT}/pixel_values.bin", pixel_values)
dump(f"{OUT}/merged_output.bin", merged)
dump(f"{OUT}/last_hidden_output.bin", last_hidden)
for name, t in weights.items():
    short = name[len(prefix):]
    dump(f"{OUT}/w_{short}.bin", t)

with open(f"{OUT}/grid_thw.json", "w") as f:
    json.dump({"grid_t": grid_t, "grid_h": grid_h, "grid_w": grid_w, "depth": DEPTH}, f)

print("OK, dumped to", OUT)
print("merged stats: mean", merged.mean().item(), "std", merged.std().item(), "abs_max", merged.abs().max().item())
