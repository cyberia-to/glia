"""Build a TINY but real Qwen3.5-family multimodal model (real
transformers classes, real random-but-seeded weights, real safetensors
+ config.json on disk via save_pretrained), run a real HF forward pass
on a synthetic multimodal prompt (text + one image), and dump
per-position hidden states/logits — the reference for
`run/tests/e2e_multimodal.rs` to compare the Rust runtime's actual
`mi import` + `LlamaModel::forward_ex` sequential-decode output
against.

Why: every piece of the Qwen3.5/3.8 architecture (GatedDeltaNet,
VisionTower, mRoPE, the embedding splice) has been golden-tested in
ISOLATION against real weights, but the actual WIRING through
forward.rs's live decode loop had never run end-to-end on anything —
the real 27B model can't even be loaded on this machine (memory
ceiling). A tiny model with the SAME architecture (layer_types mix,
rope_parameters shape, vision config shape) sidesteps that entirely:
it's a few MB, loads and runs in seconds, and exercises the identical
code paths (import/pipeline.rs's flat-rope_parameters extraction,
config.rs's LayerKind parsing, forward.rs's GatedDeltaNet/mRoPE/
TokenOverride branches) as the real model would.

Needs: same venv as the other dump scripts (torch + transformers 5.17.0+).
"""
import json, os, struct
import torch

OUT_DIR = "/tmp/e2e_tiny_model"
DUMP_DIR = "/tmp/e2e_tiny_golden"

torch.manual_seed(2026)

from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config, Qwen3_5TextConfig, Qwen3_5VisionConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration

HIDDEN = 64
HEAD_DIM = 32
NUM_LAYERS = 4
VOCAB = 96
IMAGE_TOKEN_ID = 90
VIDEO_TOKEN_ID = 91
VISION_START_ID = 92
VISION_END_ID = 93

text_config = Qwen3_5TextConfig(
    vocab_size=VOCAB,
    hidden_size=HIDDEN,
    intermediate_size=128,
    num_hidden_layers=NUM_LAYERS,
    num_attention_heads=4,  # must be an exact multiple of num_key_value_heads (GQA)
    num_key_value_heads=2,
    head_dim=HEAD_DIM,
    layer_types=["linear_attention", "linear_attention", "full_attention", "linear_attention"],
    # linear_num_value_heads must be an exact multiple of linear_num_key_heads —
    # HF's own repeat_interleave guard (`num_v_heads // num_k_heads > 1`) skips
    # expansion entirely otherwise and crashes downstream on a real shape
    # mismatch inside torch_chunk_gated_delta_rule (found via this exact test).
    linear_num_key_heads=2,
    linear_num_value_heads=4,
    linear_key_head_dim=16,
    linear_value_head_dim=16,
    linear_conv_kernel_dim=4,
    rope_parameters={
        "rope_theta": 10000000,
        "partial_rotary_factor": 0.25,
        "mrope_section": [2, 1, 1],
        "mrope_interleaved": True,
        "rope_type": "default",
    },
    tie_word_embeddings=False,
)
vision_config = Qwen3_5VisionConfig(
    depth=1,
    hidden_size=32,
    num_heads=4,
    intermediate_size=64,
    patch_size=16,
    spatial_merge_size=2,
    temporal_patch_size=2,
    out_hidden_size=HIDDEN,  # must match text hidden_size for the splice
    num_position_embeddings=16,
)
config = Qwen3_5Config(
    text_config=text_config,
    vision_config=vision_config,
    image_token_id=IMAGE_TOKEN_ID,
    video_token_id=VIDEO_TOKEN_ID,
    vision_start_token_id=VISION_START_ID,
    vision_end_token_id=VISION_END_ID,
    tie_word_embeddings=False,
)

model = Qwen3_5ForConditionalGeneration(config)

# NOTE, found via this exact test: Qwen3_5RMSNorm applies
# `x_normed * (1.0 + weight)` (a Gemma-shaped zero-centered gain), and
# its `weight` parameter correctly inits to `torch.zeros(dim)` — so
# the model's default random init ALREADY gives identity norm gain
# (1+0=1) with no fixup needed here. An earlier version of this script
# force-filled every "*.weight" matching "norm" to 1.0, which actually
# BROKE things (turned identity gain 1.0 into a wrong 2.0 gain via
# 1+1.0) — that fixup is deliberately absent now. GatedDeltaNet's own
# `Qwen3_5RMSNormGated` (a different class, plain `weight * x`, inits
# to `torch.ones`) is unaffected either way.
model.eval()

print("layer_types:", config.text_config.layer_types)
print("rope_parameters:", config.text_config.rope_parameters)

os.makedirs(OUT_DIR, exist_ok=True)
model.save_pretrained(OUT_DIR, safe_serialization=True)
with open(f"{OUT_DIR}/config.json") as f:
    saved = json.load(f)
print("saved config.json text_config.rope_parameters:", saved["text_config"].get("rope_parameters"))

# Synthetic prompt: 3 text, vision_start, 4 image placeholders (grid 4x4,
# merge=2 -> 4 merged tokens), vision_end, 2 text.
grid_t, grid_h, grid_w = 1, 4, 4
n_image_tokens = (grid_h // 2) * (grid_w // 2)
input_ids = [5, 6, 7, VISION_START_ID] + [IMAGE_TOKEN_ID] * n_image_tokens + [VISION_END_ID, 8, 9]
mm_token_type_ids = [0, 0, 0, 0] + [1] * n_image_tokens + [0, 0, 0]
input_ids_t = torch.tensor([input_ids], dtype=torch.long)
mm_token_type_ids_t = torch.tensor([mm_token_type_ids], dtype=torch.int32)
grid_thw = torch.tensor([[grid_t, grid_h, grid_w]], dtype=torch.long)

patch_dim = 3 * vision_config.temporal_patch_size * vision_config.patch_size * vision_config.patch_size
num_patches = grid_t * grid_h * grid_w
pixel_values = torch.randn(num_patches, patch_dim, dtype=torch.float32) * 0.5

with torch.no_grad():
    out = model(
        input_ids=input_ids_t,
        pixel_values=pixel_values,
        image_grid_thw=grid_thw,
        mm_token_type_ids=mm_token_type_ids_t,
        output_hidden_states=True,
    )

logits = out.logits[0]  # [seq_len, vocab]
hidden_states = out.hidden_states[-1][0]  # final layer hidden states [seq_len, hidden]
print("logits shape", logits.shape, "hidden shape", hidden_states.shape)
print("logits stats: mean", logits.mean().item(), "abs_max", logits.abs().max().item())
print("num hidden_states entries (embed + each layer):", len(out.hidden_states))


def dump(path, t):
    arr = t.detach().contiguous().to(torch.float32).numpy()
    with open(path, "wb") as f:
        for d in arr.shape:
            f.write(struct.pack("<Q", d))
        f.write(struct.pack("<Q", 0))
        f.write(arr.tobytes())


os.makedirs(DUMP_DIR, exist_ok=True)
dump(f"{DUMP_DIR}/pixel_values.bin", pixel_values)
dump(f"{DUMP_DIR}/logits.bin", logits)
dump(f"{DUMP_DIR}/hidden_states.bin", hidden_states)
# Per-layer hidden states (index 0 = post-embed, index i = post-layer(i-1)),
# so the Rust test can localize a divergence to a specific layer instead
# of only seeing it at the final logits.
for i, hs in enumerate(out.hidden_states):
    dump(f"{DUMP_DIR}/hidden_layer_{i}.bin", hs[0])

# Vision tower weights, same names/layout as dump_vision_golden.py, so
# the Rust test can reuse vision_golden.rs's loader code verbatim.
sd = model.state_dict()
vprefix = "model.visual."
for name, t in sd.items():
    if not name.startswith(vprefix):
        continue
    short = name[len(vprefix):]
    dump(f"{DUMP_DIR}/w_{short}.bin", t)

with open(f"{DUMP_DIR}/meta.json", "w") as f:
    json.dump(
        {
            "input_ids": input_ids,
            "mm_token_type_ids": mm_token_type_ids,
            "grid_thw": [grid_t, grid_h, grid_w],
            "image_token_id": IMAGE_TOKEN_ID,
            "vocab_size": VOCAB,
            "hidden_size": HIDDEN,
            "spatial_merge_size": vision_config.spatial_merge_size,
            "vision_hidden_size": vision_config.hidden_size,
            "vision_num_heads": vision_config.num_heads,
            "vision_intermediate_size": vision_config.intermediate_size,
            "vision_num_grid_per_side": int(vision_config.num_position_embeddings ** 0.5),
            "vision_rope_theta": vision_config.rope_parameters["rope_theta"],
            "vision_out_hidden_size": vision_config.out_hidden_size,
        },
        f,
    )

print("OK — model dir:", OUT_DIR, "golden dir:", DUMP_DIR)
