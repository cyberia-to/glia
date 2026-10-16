//! VisionTower vs. the real HF reference — real weights, real math.
//!
//! Golden data comes from `/tmp/vision_golden/`, regenerate with:
//!   python3 run/scripts/dump_vision_golden.py
//! (needs a venv with torch+transformers and the real
//! heretic-org/Qwen3.8-27B-heretic-ara snapshot on disk — see the
//! script's own header. Not committed: real-weight dumps for
//! patch_embed/pos_embed/2 blocks/merger, regenerable from the HF
//! download any time.)
//!
//! Loads the real `model.visual.*` patch_embed/pos_embed/merger weights
//! plus the first 2 real blocks (`depth=2` truncation — enough to
//! exercise the block/attention/mlp/residual wiring), runs the real
//! `transformers.models.qwen3_5.Qwen3_5VisionModel.forward()` on a
//! single synthetic 4x4-patch image, and compares against
//! `vision_tower_forward` here.
//!
//! Skipped if the golden dump is missing (same pattern as
//! `gated_delta_golden.rs`).
//!
//! Spec: specs/ops.md "VisionTower".

use run::backend::cpu::vision::{
    vision_tower_forward, VisionBlockWeights, VisionDims, VisionMergerWeights,
};
use run::core::tensor::Tensor;
use std::path::{Path, PathBuf};

const DIR: &str = "/tmp/vision_golden";
const DEPTH: usize = 2;

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
    let (shape, data) = read_dump(&PathBuf::from(DIR).join(format!("{name}.bin")));
    Tensor::from_f32(shape, data)
}

/// Loads `w_patch_embed.proj.weight` (Conv3d layout `[hidden, C, T, P, P]`)
/// and flattens the trailing 4 dims into one `K` axis for `matmul_f32`
/// (the degenerate-Conv3d-as-matmul equivalence — see ops.md).
fn load_patch_embed_weight() -> Tensor {
    let (shape, data) = read_dump(&PathBuf::from(DIR).join("w_patch_embed.proj.weight.bin"));
    assert_eq!(shape.len(), 5, "expected Conv3d [hidden, C, T, P, P]");
    let hidden = shape[0];
    let k: usize = shape[1..].iter().product();
    Tensor::from_f32(vec![hidden, k], data)
}

#[test]
fn vision_tower_matches_hf_reference_on_real_weights() {
    if !PathBuf::from(DIR).join("merged_output.bin").exists() {
        eprintln!(
            "skip: no golden dump at {DIR} — run:\n\
             python3 run/scripts/dump_vision_golden.py\n\
             (needs a venv with torch+transformers and the real \
             heretic-org/Qwen3.8-27B-heretic-ara snapshot on disk — \
             see the script's own header)"
        );
        return;
    }

    let grid_json = std::fs::read_to_string(PathBuf::from(DIR).join("grid_thw.json")).unwrap();
    let grid_h = grid_json
        .split("\"grid_h\": ")
        .nth(1)
        .unwrap()
        .split(',')
        .next()
        .unwrap()
        .trim()
        .parse::<usize>()
        .unwrap();
    let grid_w = grid_json
        .split("\"grid_w\": ")
        .nth(1)
        .unwrap()
        .split([',', '}'])
        .next()
        .unwrap()
        .trim()
        .parse::<usize>()
        .unwrap();

    let pixel_values = load("pixel_values");
    let patch_embed_weight = load_patch_embed_weight();
    let patch_embed_bias = load("w_patch_embed.proj.bias");
    let pos_embed_table = load("w_pos_embed.weight");

    let norm1_weights: Vec<Tensor> = (0..DEPTH).map(|i| load(&format!("w_blocks.{i}.norm1.weight"))).collect();
    let norm1_biases: Vec<Tensor> = (0..DEPTH).map(|i| load(&format!("w_blocks.{i}.norm1.bias"))).collect();
    let norm2_weights: Vec<Tensor> = (0..DEPTH).map(|i| load(&format!("w_blocks.{i}.norm2.weight"))).collect();
    let norm2_biases: Vec<Tensor> = (0..DEPTH).map(|i| load(&format!("w_blocks.{i}.norm2.bias"))).collect();
    let qkv_weights: Vec<Tensor> = (0..DEPTH).map(|i| load(&format!("w_blocks.{i}.attn.qkv.weight"))).collect();
    let qkv_biases: Vec<Tensor> = (0..DEPTH).map(|i| load(&format!("w_blocks.{i}.attn.qkv.bias"))).collect();
    let proj_weights: Vec<Tensor> = (0..DEPTH).map(|i| load(&format!("w_blocks.{i}.attn.proj.weight"))).collect();
    let proj_biases: Vec<Tensor> = (0..DEPTH).map(|i| load(&format!("w_blocks.{i}.attn.proj.bias"))).collect();
    let fc1_weights: Vec<Tensor> = (0..DEPTH).map(|i| load(&format!("w_blocks.{i}.mlp.linear_fc1.weight"))).collect();
    let fc1_biases: Vec<Tensor> = (0..DEPTH).map(|i| load(&format!("w_blocks.{i}.mlp.linear_fc1.bias"))).collect();
    let fc2_weights: Vec<Tensor> = (0..DEPTH).map(|i| load(&format!("w_blocks.{i}.mlp.linear_fc2.weight"))).collect();
    let fc2_biases: Vec<Tensor> = (0..DEPTH).map(|i| load(&format!("w_blocks.{i}.mlp.linear_fc2.bias"))).collect();

    let blocks: Vec<VisionBlockWeights> = (0..DEPTH)
        .map(|i| VisionBlockWeights {
            norm1_weight: &norm1_weights[i],
            norm1_bias: &norm1_biases[i],
            norm2_weight: &norm2_weights[i],
            norm2_bias: &norm2_biases[i],
            qkv_weight: &qkv_weights[i],
            qkv_bias: &qkv_biases[i],
            proj_weight: &proj_weights[i],
            proj_bias: &proj_biases[i],
            fc1_weight: &fc1_weights[i],
            fc1_bias: &fc1_biases[i],
            fc2_weight: &fc2_weights[i],
            fc2_bias: &fc2_biases[i],
        })
        .collect();

    let merger_norm_weight = load("w_merger.norm.weight");
    let merger_norm_bias = load("w_merger.norm.bias");
    let merger_fc1_weight = load("w_merger.linear_fc1.weight");
    let merger_fc1_bias = load("w_merger.linear_fc1.bias");
    let merger_fc2_weight = load("w_merger.linear_fc2.weight");
    let merger_fc2_bias = load("w_merger.linear_fc2.bias");
    let merger = VisionMergerWeights {
        norm_weight: &merger_norm_weight,
        norm_bias: &merger_norm_bias,
        fc1_weight: &merger_fc1_weight,
        fc1_bias: &merger_fc1_bias,
        fc2_weight: &merger_fc2_weight,
        fc2_bias: &merger_fc2_bias,
    };

    let dims = VisionDims {
        hidden_size: 1152,
        num_heads: 16,
        intermediate_size: 4304,
        spatial_merge_size: 2,
        num_grid_per_side: 48,
        rope_theta: 10000.0,
        out_hidden_size: 5120,
    };

    let ours = vision_tower_forward(
        &pixel_values,
        grid_h,
        grid_w,
        &patch_embed_weight,
        &patch_embed_bias,
        &pos_embed_table,
        &blocks,
        &merger,
        dims,
    )
    .expect("forward");

    let (hf_shape, hf_data) = read_dump(&PathBuf::from(DIR).join("merged_output.bin"));
    assert_eq!(ours.shape, hf_shape, "output shape mismatch");

    let ours_data = ours.as_f32();
    let mut worst_abs = 0f32;
    let mut worst_i = 0usize;
    let mut max_ref = 0f32;
    for (i, (&o, &h)) in ours_data.iter().zip(hf_data.iter()).enumerate() {
        let diff = (o - h).abs();
        if diff > worst_abs {
            worst_abs = diff;
            worst_i = i;
        }
        max_ref = max_ref.max(h.abs());
    }
    eprintln!(
        "worst abs diff {worst_abs} at idx {worst_i} (ours={} hf={}), max|hf|={max_ref}",
        ours_data[worst_i], hf_data[worst_i]
    );
    let tol = (max_ref * 1e-4).max(1e-6);
    assert!(
        worst_abs < tol,
        "vision_tower_forward diverges from the real HF reference: worst abs diff {worst_abs} > tol {tol}"
    );
}
