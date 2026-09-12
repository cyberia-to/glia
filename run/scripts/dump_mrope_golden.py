"""Dump real transformers `Qwen3_5Model.get_rope_index` output (the 3D
mRoPE position-id builder) and real `Qwen3_5TextRotaryEmbedding.forward`
output (the interleaved mrope cos/sin recomposition), for a synthetic
token sequence with one image span — for
`run/tests/mrope_golden.rs` to compare against a pure-Rust
reimplementation.

Both are pure index/config arithmetic — no real weights needed, so
this instantiates a TINY Qwen3_5Model (num_hidden_layers=1, depth=1,
small hidden sizes) purely to get a real bound instance to call the
real methods on. Only `config.vision_config.spatial_merge_size` and
the text config's real `rope_parameters` (rope_theta,
partial_rotary_factor, mrope_section, mrope_interleaved) matter for
correctness — pulled from the real 27B config, not guessed.

Needs: same venv as dump_gdn_golden.py (torch + transformers 5.17.0+).
"""
import json, struct
import torch

OUT = "/tmp/mrope_golden"
DIR = "/Users/master/.cache/huggingface/hub/models--heretic-org--Qwen3.8-27B-heretic-ara/snapshots/2dc9b364104881cbb85e390f00195ba6b9d745e9"

with open(f"{DIR}/config.json") as f:
    real_config = json.load(f)
real_tc = real_config["text_config"]
real_vc = real_config["vision_config"]

from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config, Qwen3_5TextConfig, Qwen3_5VisionConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Model

text_config = Qwen3_5TextConfig(
    vocab_size=64,
    hidden_size=32,
    intermediate_size=64,
    num_hidden_layers=1,
    num_attention_heads=real_tc["num_attention_heads"],
    num_key_value_heads=real_tc["num_key_value_heads"],
    head_dim=real_tc["head_dim"],
    rope_parameters=real_tc["rope_parameters"],
    layer_types=["full_attention"],
)
vision_config = Qwen3_5VisionConfig(
    depth=1,
    hidden_size=32,
    num_heads=4,
    intermediate_size=64,
    patch_size=real_vc["patch_size"],
    spatial_merge_size=real_vc["spatial_merge_size"],
    temporal_patch_size=real_vc["temporal_patch_size"],
    out_hidden_size=32,
    num_position_embeddings=real_vc["num_position_embeddings"],
)
config = Qwen3_5Config(text_config=text_config, vision_config=vision_config)
model = Qwen3_5Model(config)
model.eval()

# Sequence: 5 text tokens, one 4x4-patch image (merge=2 -> 4 placeholder
# tokens), 3 more text tokens.
grid_thw = torch.tensor([[1, 4, 4]], dtype=torch.long)
n_image_tokens = 4
input_ids = torch.arange(1, 5 + n_image_tokens + 3 + 1).unsqueeze(0)  # values unused by get_rope_index
mm_token_type_ids = torch.tensor([[0] * 5 + [1] * n_image_tokens + [0] * 3], dtype=torch.int32)

position_ids, mrope_deltas = model.get_rope_index(
    input_ids, mm_token_type_ids, image_grid_thw=grid_thw, attention_mask=None
)
print("position_ids shape", position_ids.shape)
print(position_ids)
print("mrope_deltas", mrope_deltas)

# Interleaved mrope cos/sin recomposition, real Qwen3_5TextRotaryEmbedding.
rotary = model.language_model.rotary_emb
dummy_x = torch.zeros(1, dtype=torch.float32)
cos, sin = rotary(dummy_x, position_ids)
print("cos shape", cos.shape)


def dump(path, t):
    arr = t.detach().contiguous().to(torch.float32).numpy()
    with open(path, "wb") as f:
        for d in arr.shape:
            f.write(struct.pack("<Q", d))
        f.write(struct.pack("<Q", 0))
        f.write(arr.tobytes())


import os
os.makedirs(OUT, exist_ok=True)
dump(f"{OUT}/mm_token_type_ids.bin", mm_token_type_ids.float())
dump(f"{OUT}/grid_thw.bin", grid_thw.float())
dump(f"{OUT}/position_ids.bin", position_ids.float())
dump(f"{OUT}/cos.bin", cos)
dump(f"{OUT}/sin.bin", sin)

with open(f"{OUT}/meta.json", "w") as f:
    json.dump(
        {
            "spatial_merge_size": real_vc["spatial_merge_size"],
            "head_dim": real_tc["head_dim"],
            "rope_theta": real_tc["rope_parameters"]["rope_theta"],
            "partial_rotary_factor": real_tc["rope_parameters"]["partial_rotary_factor"],
            "mrope_section": real_tc["rope_parameters"]["mrope_section"],
            "seq_len": input_ids.shape[1],
        },
        f,
    )

print("OK, dumped to", OUT)
