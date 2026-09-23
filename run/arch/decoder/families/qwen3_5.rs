//! Qwen3.5 / Qwen3.8 (`model_type = "qwen3_5"`).
//!
//! Two real divergences from LlamaStyle, both confirmed against real
//! source (not guessed) — found via `run/tests/e2e_multimodal.rs`,
//! the first end-to-end wiring test for this family:
//!
//! - `rmsnorm_plus_one`: `Qwen3_5RMSNorm.forward` applies
//!   `x_normed * (1.0 + weight)` — a Gemma-shaped ZERO-CENTERED gain
//!   (`weight` inits to `torch.zeros(dim)`, so untrained/identity is
//!   `weight=0` → gain `1.0`), not the plain `weight * x_normed` most
//!   of this codebase's other curated families use. Applies to
//!   `input_layernorm`, `post_attention_layernorm`, `q_norm`, `k_norm`,
//!   and the model's final `norm` — i.e. every norm EXCEPT one:
//!   GatedDeltaNet's own `Qwen3_5RMSNormGated` (`ops.md`
//!   §"GatedDeltaNet") is a genuinely different class (`weight` inits
//!   to `torch.ones`, plain multiply, no `+1`) — already handled
//!   correctly in `backend::cpu::gated_delta`'s hand-written norm step,
//!   which does not go through this family-profile flag at all.
//!   Getting this wrong compounds silently through every layer (two
//!   `Qwen3_5RMSNorm` applications per layer) — small-looking at layer
//!   0, large by the final layer, exactly the growing-with-depth
//!   divergence pattern that exposed this bug.
//! - `has_attn_output_gate` (see its own doc comment on
//!   `FamilyProfile`), confirmed against `Qwen3_5Attention.__init__`
//!   and the real 27B model's actual q_proj tensor shape (`[12288,
//!   5120]` = `24 heads * 256 head_dim * 2`, not `[6144, 5120]`).
//!
//! GatedDeltaNet ("linear_attention" layers) and mRoPE
//! ("mrope_section") are separate mechanisms, not family-profile flags
//! — they're driven by `LlamaConfig` fields already (`layer_types`,
//! `mrope_section`) since they're config-shaped, not a fixed per-model
//! architectural constant the way these two are.
//!
//! Spec: specs/ops.md §"GatedDeltaNet", §"mRoPE, interleaved".

use super::{AttnScale, FamilyProfile};

pub fn profile() -> FamilyProfile {
    FamilyProfile {
        name: "Qwen3.5/3.8",
        rmsnorm_plus_one: true,
        scaled_embeddings: false,
        v_norm_per_head: false,
        attn_scale: AttnScale::PerHeadDim,
        has_attn_output_gate: true,
    }
}
