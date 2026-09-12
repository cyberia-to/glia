//! Full multimodal end-to-end test: real `mi import`, real
//! `LlamaModel::load`, real sequential `forward_ex` decode — against a
//! REAL (tiny) Qwen3.5-family HF model's REAL forward pass.
//!
//! Why this exists: every piece of the Qwen3.5/3.8 architecture
//! (GatedDeltaNet, VisionTower, mRoPE, the embedding-splice wiring)
//! has been golden-tested in ISOLATION against real weights, but the
//! actual WIRING through `forward.rs`'s live decode loop had never
//! run end-to-end on anything — the real 27B model can't even be
//! loaded on this machine (memory ceiling). This test sidesteps that:
//! a tiny model with the SAME architecture shape (mixed
//! linear_attention/full_attention layers, the same flat
//! rope_parameters, a real vision tower) is a few hundred KB, loads
//! and runs in milliseconds, and exercises the IDENTICAL code paths
//! (import/pipeline.rs's config extraction, config.rs's LayerKind
//! parsing, forward.rs's GatedDeltaNet/mRoPE/TokenOverride branches)
//! the real model would.
//!
//! Regenerate the golden data + tiny model with:
//!   python3 run/scripts/dump_e2e_tiny_model.py
//!   cargo run --release -p import --bin mi -- import /tmp/e2e_tiny_model
//! (needs a venv with torch+transformers — see the script's header)
//!
//! Spec: specs/gated-delta-vl-plan.md's wiring progress entries.

use run::arch::decoder::{LlamaModel, TokenOverride};
use run::backend::cpu::vision::{vision_tower_forward, VisionBlockWeights, VisionDims, VisionMergerWeights};
use run::backend::cpu::mrope::{mrope_position_ids, ModalityRun};
use run::backend::cpu::CpuBackend;
use run::core::tensor::Tensor;
use std::path::{Path, PathBuf};

const GOLDEN_DIR: &str = "/tmp/e2e_tiny_golden";
const MODEL_PATH: &str = "/Users/master/llm/e2e_tiny_model.canonical.model";

fn read_dump(path: &Path) -> (Vec<usize>, Vec<f32>) {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    let mut shape = Vec::new();
    let mut off = 0;
    loop {
        let d = u64::from_le_bytes(bytes[off..off + 8].try_into().unwrap());
        off += 8;
        if d == 0 {
            break;
        }
        shape.push(d as usize);
    }
    let data_bytes = &bytes[off..];
    let n = data_bytes.len() / 4;
    let mut data = Vec::with_capacity(n);
    for i in 0..n {
        data.push(f32::from_le_bytes(data_bytes[i * 4..i * 4 + 4].try_into().unwrap()));
    }
    (shape, data)
}

fn load(name: &str) -> Tensor {
    let (shape, data) = read_dump(&PathBuf::from(GOLDEN_DIR).join(format!("{name}.bin")));
    Tensor::from_f32(shape, data)
}

fn load_patch_embed_weight() -> Tensor {
    let (shape, data) = read_dump(&PathBuf::from(GOLDEN_DIR).join("w_patch_embed.proj.weight.bin"));
    assert_eq!(shape.len(), 5, "expected Conv3d [hidden, C, T, P, P]");
    let hidden = shape[0];
    let k: usize = shape[1..].iter().product();
    Tensor::from_f32(vec![hidden, k], data)
}

#[test]
fn e2e_multimodal_sequential_decode_matches_real_hf_forward() {
    if !PathBuf::from(GOLDEN_DIR).join("logits.bin").exists() || !PathBuf::from(MODEL_PATH).exists() {
        eprintln!(
            "skip: no golden dump / tiny model — run:\n\
             python3 run/scripts/dump_e2e_tiny_model.py\n\
             cargo run --release -p import --bin mi -- import /tmp/e2e_tiny_model"
        );
        return;
    }

    let meta: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(PathBuf::from(GOLDEN_DIR).join("meta.json")).unwrap()).unwrap();
    let input_ids: Vec<u32> = meta["input_ids"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect();
    let mm_types: Vec<u32> = meta["mm_token_type_ids"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect();
    let grid_thw: Vec<usize> = meta["grid_thw"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as usize).collect();
    let image_token_id = meta["image_token_id"].as_u64().unwrap() as u32;
    let spatial_merge_size = meta["spatial_merge_size"].as_u64().unwrap() as usize;
    let seq_len = input_ids.len();
    assert_eq!(mm_types.len(), seq_len);

    // ── 1. Vision tower: real weights, real forward, same as vision_golden.rs ──
    let pixel_values = load("pixel_values");
    let patch_embed_weight = load_patch_embed_weight();
    let patch_embed_bias = load("w_patch_embed.proj.bias");
    let pos_embed_table = load("w_pos_embed.weight");
    let norm1_w = load("w_blocks.0.norm1.weight");
    let norm1_b = load("w_blocks.0.norm1.bias");
    let norm2_w = load("w_blocks.0.norm2.weight");
    let norm2_b = load("w_blocks.0.norm2.bias");
    let qkv_w = load("w_blocks.0.attn.qkv.weight");
    let qkv_b = load("w_blocks.0.attn.qkv.bias");
    let proj_w = load("w_blocks.0.attn.proj.weight");
    let proj_b = load("w_blocks.0.attn.proj.bias");
    let fc1_w = load("w_blocks.0.mlp.linear_fc1.weight");
    let fc1_b = load("w_blocks.0.mlp.linear_fc1.bias");
    let fc2_w = load("w_blocks.0.mlp.linear_fc2.weight");
    let fc2_b = load("w_blocks.0.mlp.linear_fc2.bias");
    let blocks = vec![VisionBlockWeights {
        norm1_weight: &norm1_w, norm1_bias: &norm1_b,
        norm2_weight: &norm2_w, norm2_bias: &norm2_b,
        qkv_weight: &qkv_w, qkv_bias: &qkv_b,
        proj_weight: &proj_w, proj_bias: &proj_b,
        fc1_weight: &fc1_w, fc1_bias: &fc1_b,
        fc2_weight: &fc2_w, fc2_bias: &fc2_b,
    }];
    let merger_norm_w = load("w_merger.norm.weight");
    let merger_norm_b = load("w_merger.norm.bias");
    let merger_fc1_w = load("w_merger.linear_fc1.weight");
    let merger_fc1_b = load("w_merger.linear_fc1.bias");
    let merger_fc2_w = load("w_merger.linear_fc2.weight");
    let merger_fc2_b = load("w_merger.linear_fc2.bias");
    let merger = VisionMergerWeights {
        norm_weight: &merger_norm_w, norm_bias: &merger_norm_b,
        fc1_weight: &merger_fc1_w, fc1_bias: &merger_fc1_b,
        fc2_weight: &merger_fc2_w, fc2_bias: &merger_fc2_b,
    };
    let vdims = VisionDims {
        hidden_size: meta["vision_hidden_size"].as_u64().unwrap() as usize,
        num_heads: meta["vision_num_heads"].as_u64().unwrap() as usize,
        intermediate_size: meta["vision_intermediate_size"].as_u64().unwrap() as usize,
        spatial_merge_size,
        num_grid_per_side: meta["vision_num_grid_per_side"].as_u64().unwrap() as usize,
        rope_theta: meta["vision_rope_theta"].as_f64().unwrap() as f32,
        out_hidden_size: meta["vision_out_hidden_size"].as_u64().unwrap() as usize,
    };
    let (grid_t, grid_h, grid_w) = (grid_thw[0], grid_thw[1], grid_thw[2]);
    assert_eq!(grid_t, 1, "test only handles a single image frame");
    let image_embeds = vision_tower_forward(
        &pixel_values, grid_h, grid_w,
        &patch_embed_weight, &patch_embed_bias, &pos_embed_table,
        &blocks, &merger, vdims,
    )
    .expect("vision tower forward");
    let n_image_tokens = (grid_h / spatial_merge_size) * (grid_w / spatial_merge_size);
    let hidden_size = meta["hidden_size"].as_u64().unwrap() as usize;
    assert_eq!(image_embeds.shape, vec![n_image_tokens, hidden_size]);
    let image_embeds_data = image_embeds.as_f32();

    // ── 2. mRoPE position ids, derived generically from mm_token_type_ids ──
    let mut runs = Vec::new();
    let mut i = 0;
    while i < seq_len {
        let ty = mm_types[i];
        let start = i;
        while i < seq_len && mm_types[i] == ty {
            i += 1;
        }
        let len = i - start;
        if ty == 0 {
            runs.push(ModalityRun::Text { len });
        } else if ty == 1 {
            assert_eq!(len, n_image_tokens, "image run length must match the vision tower's token count");
            runs.push(ModalityRun::Vision { t: grid_t, h: grid_h, w: grid_w });
        } else {
            panic!("unsupported mm_token_type_id {ty} in this test");
        }
    }
    let (positions, _next_pos) = mrope_position_ids(&runs, spatial_merge_size);

    // ── 3. Sequential decode through the REAL runtime, one token at a time ──
    let mut model = LlamaModel::load(Path::new(MODEL_PATH)).expect("load tiny model");
    let backend = CpuBackend::new();
    let mut image_idx = 0usize;
    let mut ours_logits = vec![0f32; seq_len * (meta["vocab_size"].as_u64().unwrap() as usize)];
    let vocab_size = meta["vocab_size"].as_u64().unwrap() as usize;

    if std::env::var("E2E_DEBUG_LAYERS").is_ok() {
        std::env::set_var("RUN_DEBUG_LAYERS", "1");
    }
    for pos in 0..seq_len {
        let token_id = input_ids[pos];
        let triple = [positions[0][pos], positions[1][pos], positions[2][pos]];
        let embed = if mm_types[pos] == 1 {
            assert_eq!(token_id, image_token_id, "image-typed position must carry the image placeholder token id");
            let row = image_embeds_data[image_idx * hidden_size..(image_idx + 1) * hidden_size].to_vec();
            image_idx += 1;
            Some(row)
        } else {
            None
        };
        let override_ = TokenOverride { embed, position: Some(triple) };
        let logits = model.forward_ex(token_id, &backend, Some(&override_)).expect("forward_ex");
        ours_logits[pos * vocab_size..(pos + 1) * vocab_size].copy_from_slice(&logits);
    }
    assert_eq!(image_idx, n_image_tokens, "must have consumed every image embedding row");

    // ── 4. Compare against the real HF forward pass ──
    let (hf_shape, hf_logits) = read_dump(&PathBuf::from(GOLDEN_DIR).join("logits.bin"));
    assert_eq!(hf_shape, vec![seq_len, vocab_size], "logits shape mismatch");

    let mut worst_abs = 0f32;
    let mut worst_pos = 0usize;
    let mut max_ref = 0f32;
    for pos in 0..seq_len {
        for v in 0..vocab_size {
            let idx = pos * vocab_size + v;
            let diff = (ours_logits[idx] - hf_logits[idx]).abs();
            if diff > worst_abs {
                worst_abs = diff;
                worst_pos = pos;
            }
            max_ref = max_ref.max(hf_logits[idx].abs());
        }
    }
    eprintln!("worst logits abs diff {worst_abs} at seq pos {worst_pos}, max|hf|={max_ref}");
    {
        let n = ours_logits.len() as f32;
        let mean_abs_err: f32 = ours_logits.iter().zip(hf_logits.iter()).map(|(o, h)| (o - h).abs()).sum::<f32>() / n;
        let mean_abs_ref: f32 = hf_logits.iter().map(|h| h.abs()).sum::<f32>() / n;
        // Pearson correlation — near 1.0 means "same shape, scaled/noisy",
        // near 0 means "structurally different" (a real wiring bug).
        let mean_o: f32 = ours_logits.iter().sum::<f32>() / n;
        let mean_h: f32 = hf_logits.iter().sum::<f32>() / n;
        let mut cov = 0f32;
        let mut var_o = 0f32;
        let mut var_h = 0f32;
        for (&o, &h) in ours_logits.iter().zip(hf_logits.iter()) {
            cov += (o - mean_o) * (h - mean_h);
            var_o += (o - mean_o).powi(2);
            var_h += (h - mean_h).powi(2);
        }
        let corr = cov / (var_o.sqrt() * var_h.sqrt());
        eprintln!("mean abs err {mean_abs_err} (mean|hf|={mean_abs_ref}), pearson correlation={corr}");
        assert!(
            corr > 0.98,
            "logits correlation {corr} too low for quantization noise alone (real bugs measured <0.75 \
             here while building this test) — suspect a structural wiring bug, not just Q8 noise"
        );
    }
    // This tolerance is deliberately looser than the pure-f32 golden
    // tests (gated_delta_golden.rs, vision_golden.rs, mrope_golden.rs
    // all use ~1e-4 relative): those compare a single op against raw
    // f32 weights, this compares the WHOLE pipeline — real Q8
    // quantization on every projection, compounded through 4 layers
    // (2 RMSNorm applications each) plus lm_head. 10% relative was
    // picked empirically: a real wiring bug (found and fixed twice
    // building this test — the attention-gate width/layout, and
    // Qwen3_5RMSNorm's `(1+weight)` zero-centered gain missing
    // entirely) each produced 60-80% worst-case divergence and
    // Pearson correlation under 0.75; correctly-wired real Q8 noise
    // measures ~4% worst-case with correlation >0.999. This gap is
    // wide enough that 10% won't mask a real regression.
    let tol = (max_ref * 0.10).max(1e-4);
    assert!(
        worst_abs < tol,
        "sequential forward_ex decode diverges from real HF batched forward: worst abs diff {worst_abs} > tol {tol} \
         (at sequence position {worst_pos}) — the live decode wiring (GatedDeltaNet/mRoPE/TokenOverride splice) has a bug"
    );
}
