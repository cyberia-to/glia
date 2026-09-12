//! Multimodal (image) generation — real image file → vision tower →
//! spliced prompt → sequential decode, using the same
//! `LlamaModel::forward_ex`/`TokenOverride` path `run/tests/
//! e2e_multimodal.rs` verified against real HF output.
//!
//! **Scope note**: this builds the placeholder block as
//! `[vision_start_token_id] + [image_token_id] * n ++ [vision_end_token_id]`
//! and inserts it right before the tokenized prompt text (after any
//! BOS the chat template added). This is a reasonable, functional
//! approximation of how a real `Qwen3VLProcessor` interleaves image
//! placeholders into a chat-templated conversation, but it has NOT
//! been verified against the exact token layout the real processor
//! would produce for the SAME prompt (that layout depends on the
//! model's own chat_template.jinja, not just token IDs) — the
//! splice/mRoPE/vision-tower MATH downstream of this point is what's
//! golden-tested, not this exact placement convention.
//!
//! Spec: ops.md §"VisionTower", §"mRoPE, interleaved";
//! gated-delta-vl-plan.md's fusion progress entries.

use crate::arch::decoder::{LlamaModel, TokenOverride};
use crate::backend::cpu::mrope::{mrope_position_ids, ModalityRun};
use crate::backend::cpu::vision::{vision_tower_forward, VisionDims};
use crate::backend::Backend;
use crate::generate::{sample, SampleConfig};
use crate::tokenizer::Tokenizer;
use crate::vision_preprocess::PreprocessConfig;

/// Run the vision tower on a real image file's bytes, using the
/// model's own loaded vision weights/config. Returns `(image_embeds
/// [n_tokens, hidden], grid_t, grid_h, grid_w)`.
pub fn embed_image(model: &LlamaModel, image_bytes: &[u8]) -> Result<(Vec<f32>, usize, usize, usize, usize), String> {
    let vc = model.config.vision.ok_or("model has no vision tower (not a VL checkpoint)")?;
    let vw = model.weights.vision.as_ref().ok_or("model config has vision but weights.vision is None")?;

    let pp_cfg = PreprocessConfig {
        patch_size: vc.patch_size,
        temporal_patch_size: vc.temporal_patch_size,
        merge_size: vc.spatial_merge_size,
        in_channels: vc.in_channels,
        ..PreprocessConfig::default()
    };
    let (pixel_values, grid_t, grid_h, grid_w) =
        crate::vision_preprocess::preprocess_image_bytes(image_bytes, &pp_cfg)?;
    if grid_t != 1 {
        return Err("video input (grid_t != 1) is not supported yet".into());
    }

    let vdims = VisionDims {
        hidden_size: vc.hidden_size,
        num_heads: vc.num_heads,
        intermediate_size: vc.intermediate_size,
        spatial_merge_size: vc.spatial_merge_size,
        num_grid_per_side: (vc.num_position_embeddings as f64).sqrt() as usize,
        rope_theta: vc.rope_theta,
        out_hidden_size: vc.out_hidden_size,
    };
    let blocks: Vec<_> = vw.blocks.iter().map(|b| b.as_ref()).collect();
    let merger = vw.merger.as_ref();
    let embeds = vision_tower_forward(
        &pixel_values, grid_h, grid_w,
        &vw.patch_embed_weight, &vw.patch_embed_bias, &vw.pos_embed_table,
        &blocks, &merger, vdims,
    )
    .map_err(|e| format!("vision tower forward: {e}"))?;
    let n_tokens = embeds.shape[0];
    Ok((embeds.as_f32().to_vec(), n_tokens, grid_t, grid_h, grid_w))
}

/// Full multimodal generation: real image bytes + a chat-templated
/// text prompt → generated text. See this module's doc comment for
/// the placeholder-placement scope note.
#[allow(clippy::too_many_arguments)]
pub fn generate_multimodal(
    model: &mut LlamaModel,
    tok: &Tokenizer,
    backend: &dyn Backend,
    prompt: &str,
    image_bytes: &[u8],
    max_tokens: usize,
    sample_cfg: SampleConfig,
) -> Result<(String, usize), String> {
    model.reset_kv_cache();

    let (image_embeds, n_image_tokens, grid_t, grid_h, grid_w) = embed_image(model, image_bytes)?;
    let spatial_merge_size = model.config.vision.unwrap().spatial_merge_size;
    let image_token_id = model.config.image_token_id.ok_or("model has no image_token_id")?;
    let vision_start_token_id = model.config.vision_start_token_id.ok_or("model has no vision_start_token_id")?;
    let vision_end_token_id = model.config.vision_end_token_id.ok_or("model has no vision_end_token_id")?;

    let mut prompt_ids = tok.encode(prompt);
    if let Some(bos) = tok.bos_token_id {
        if prompt_ids.first() != Some(&bos) {
            prompt_ids.insert(0, bos);
        }
    }

    // Splice: [prompt up to and incl. BOS] ++ [vision_start, image_token*n,
    // vision_end] ++ [rest of prompt] — see module doc's scope note.
    let split_at = if tok.bos_token_id.is_some() { 1 } else { 0 };
    let mut input_ids: Vec<u32> = prompt_ids[..split_at].to_vec();
    let mut mm_types: Vec<u32> = vec![0; split_at];
    input_ids.push(vision_start_token_id);
    mm_types.push(0);
    for _ in 0..n_image_tokens {
        input_ids.push(image_token_id);
        mm_types.push(1);
    }
    input_ids.push(vision_end_token_id);
    mm_types.push(0);
    input_ids.extend_from_slice(&prompt_ids[split_at..]);
    mm_types.extend(std::iter::repeat_n(0u32, prompt_ids.len() - split_at));

    let runs = [
        ModalityRun::Text { len: split_at + 1 }, // BOS (if any) + vision_start
        ModalityRun::Vision { t: grid_t, h: grid_h, w: grid_w },
        ModalityRun::Text { len: 1 + (prompt_ids.len() - split_at) }, // vision_end + rest
    ];
    let (positions, _next_pos) = mrope_position_ids(&runs, spatial_merge_size);

    let mut image_idx = 0usize;
    let mut logits = Vec::new();
    for (pos, &token_id) in input_ids.iter().enumerate() {
        let triple = [positions[0][pos], positions[1][pos], positions[2][pos]];
        let embed = if mm_types[pos] == 1 {
            let hidden = model.config.hidden_size;
            let row = image_embeds[image_idx * hidden..(image_idx + 1) * hidden].to_vec();
            image_idx += 1;
            Some(row)
        } else {
            None
        };
        let override_ = TokenOverride { embed, position: Some(triple) };
        logits = model
            .forward_ex(token_id, backend, Some(&override_))
            .map_err(|e| format!("forward_ex: {e}"))?;
    }

    let mut generated = Vec::with_capacity(max_tokens.min(1024));
    for _ in 0..max_tokens {
        let next = sample(&logits, sample_cfg);
        if tok.is_eos(next) {
            break;
        }
        generated.push(next);
        logits = model.forward(next, backend).map_err(|e| format!("forward: {e}"))?;
    }
    let text = tok.decode(&generated, false);
    Ok((text, generated.len()))
}
