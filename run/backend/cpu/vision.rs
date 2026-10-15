//! VisionTower — the Qwen3.5/3.8 native vision encoder.
//!
//! Spec: specs/ops.md §"VisionTower". Patch embed through merger, one
//! image/frame at a time (packed multi-image batching is the caller's
//! concern — call this once per image and concatenate `cu_seqlens`
//! segment boundaries if batching several). f32 throughout: the whole
//! tower is ~460M params (~1.8 GB f32) versus the text decoder's tens
//! of billions — no memory pressure to trade away correctness-first
//! simplicity for here, unlike GatedDeltaNet's big projections.

use crate::backend::BackendError;
use crate::backend::cpu::{gelu_erf_f32, gelu_tanh_f32, matmul_f32};
use crate::core::tensor::Tensor;

#[derive(Clone, Copy)]
pub struct VisionDims {
    pub hidden_size: usize,
    pub num_heads: usize,
    pub intermediate_size: usize,
    pub spatial_merge_size: usize,
    pub num_grid_per_side: usize,
    pub rope_theta: f32,
    pub out_hidden_size: usize,
}

impl VisionDims {
    fn head_dim(&self) -> usize {
        self.hidden_size / self.num_heads
    }
}

pub struct VisionBlockWeights<'a> {
    pub norm1_weight: &'a Tensor,
    pub norm1_bias: &'a Tensor,
    pub norm2_weight: &'a Tensor,
    pub norm2_bias: &'a Tensor,
    pub qkv_weight: &'a Tensor,
    pub qkv_bias: &'a Tensor,
    pub proj_weight: &'a Tensor,
    pub proj_bias: &'a Tensor,
    pub fc1_weight: &'a Tensor,
    pub fc1_bias: &'a Tensor,
    pub fc2_weight: &'a Tensor,
    pub fc2_bias: &'a Tensor,
}

pub struct VisionMergerWeights<'a> {
    pub norm_weight: &'a Tensor,
    pub norm_bias: &'a Tensor,
    pub fc1_weight: &'a Tensor,
    pub fc1_bias: &'a Tensor,
    pub fc2_weight: &'a Tensor,
    pub fc2_bias: &'a Tensor,
}

/// One patch's block-major `(row, col)` position — shared by position-
/// embedding interpolation and vision RoPE (ops.md's PatchOrder note).
fn patch_order_positions(grid_h: usize, grid_w: usize, merge: usize) -> Vec<(usize, usize)> {
    let blocks_w = grid_w / merge;
    let n = grid_h * grid_w;
    let mut out = Vec::with_capacity(n);
    for k in 0..n {
        let in_col = k % merge;
        let in_row = (k / merge) % merge;
        let block_col = (k / (merge * merge)) % blocks_w;
        let block_row = k / (merge * merge * blocks_w);
        out.push((block_row * merge + in_row, block_col * merge + in_col));
    }
    out
}

/// `x`: `[num_patches, patch_dim]` (already flattened `[C,T,ph,pw]` per
/// row by the — not-yet-implemented — image preprocessor). Returns
/// `[num_patches, hidden_size]`.
pub fn patch_embed(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor, BackendError> {
    let y = matmul_f32(x, weight)?;
    add_bias(&y, bias)
}

fn add_bias(x: &Tensor, bias: &Tensor) -> Result<Tensor, BackendError> {
    let cols = *x.shape.last().unwrap_or(&0);
    let b = bias.as_f32();
    let mut out = x.as_f32().to_vec();
    for row in out.chunks_mut(cols) {
        for (v, bi) in row.iter_mut().zip(b.iter()) {
            *v += bi;
        }
    }
    Ok(Tensor::from_f32(x.shape.clone(), out))
}

/// One axis's 2-tap bilinear gather indices + weights for `align_corners
/// = true` (ops.md's exact weight rule — weight computed from the
/// UNCLAMPED tap offset, gather index clamped separately).
fn bilinear_axis(src_pos: usize, axis_len: usize, side: usize) -> [(usize, f32); 2] {
    let side_f = side as f32;
    let denom = (axis_len.max(2) - 1) as f32; // max(len-1, 1)
    let src = src_pos as f32 * (side_f - 1.0) / denom;
    let floor = src.floor();
    let mut out = [(0usize, 0f32); 2];
    for (i, offset) in [0.0f32, 1.0f32].into_iter().enumerate() {
        let unclamped = floor + offset;
        let tap = (unclamped.max(0.0).min(side_f - 1.0)) as usize;
        let weight = (1.0 - (src - unclamped).abs()).max(0.0);
        out[i] = (tap, weight);
    }
    out
}

/// Resample `table[num_grid_per_side^2, hidden]` to one row per patch,
/// in block-major `(row, col)` order (`patch_order_positions`).
pub fn gather_pos_embed(
    table: &Tensor,
    grid_h: usize,
    grid_w: usize,
    merge: usize,
    side: usize,
) -> Tensor {
    let hidden = table.shape[1];
    let t = table.as_f32();
    let positions = patch_order_positions(grid_h, grid_w, merge);
    let mut out = vec![0f32; positions.len() * hidden];
    for (i, &(row, col)) in positions.iter().enumerate() {
        let h_taps = bilinear_axis(row, grid_h, side);
        let w_taps = bilinear_axis(col, grid_w, side);
        let dst = &mut out[i * hidden..(i + 1) * hidden];
        for &(hi, hw) in &h_taps {
            for &(wi, ww) in &w_taps {
                let weight = hw * ww;
                if weight == 0.0 {
                    continue;
                }
                let src_row = &t[(hi * side + wi) * hidden..(hi * side + wi + 1) * hidden];
                for (d, s) in dst.iter_mut().zip(src_row.iter()) {
                    *d += s * weight;
                }
            }
        }
    }
    Tensor::from_f32(vec![positions.len(), hidden], out)
}

/// Axial 2D RoPE cos/sin for every patch — `head_dim`-length each,
/// shared across all heads (ops.md's exact frequency construction).
fn vision_rope_cos_sin(
    positions: &[(usize, usize)],
    head_dim: usize,
    theta: f32,
) -> (Vec<f32>, Vec<f32>) {
    let spatial_dim = head_dim / 2;
    let n_freq = spatial_dim / 2;
    let inv_freq: Vec<f32> = (0..n_freq)
        .map(|i| 1.0 / theta.powf((2 * i) as f32 / spatial_dim as f32))
        .collect();
    let mut cos = vec![0f32; positions.len() * head_dim];
    let mut sin = vec![0f32; positions.len() * head_dim];
    for (p, &(row, col)) in positions.iter().enumerate() {
        // freq_hw = concat(row*inv_freq, col*inv_freq); full = concat(freq_hw, freq_hw).
        let mut full = vec![0f32; head_dim];
        for i in 0..n_freq {
            full[i] = row as f32 * inv_freq[i];
            full[n_freq + i] = col as f32 * inv_freq[i];
        }
        for i in 0..spatial_dim {
            full[spatial_dim + i] = full[i];
        }
        for d in 0..head_dim {
            cos[p * head_dim + d] = full[d].cos();
            sin[p * head_dim + d] = full[d].sin();
        }
    }
    (cos, sin)
}

/// `x*cos + rotate_half(x)*sin`, per head, in place style (returns new).
fn apply_rope(x: &[f32], cos: &[f32], sin: &[f32], num_heads: usize, head_dim: usize) -> Vec<f32> {
    let half = head_dim / 2;
    let seq = x.len() / (num_heads * head_dim);
    let mut out = vec![0f32; x.len()];
    for t in 0..seq {
        let c = &cos[t * head_dim..(t + 1) * head_dim];
        let s = &sin[t * head_dim..(t + 1) * head_dim];
        for h in 0..num_heads {
            let base = (t * num_heads + h) * head_dim;
            let row = &x[base..base + head_dim];
            let dst = &mut out[base..base + head_dim];
            for d in 0..head_dim {
                let rotated = if d < half { -row[d + half] } else { row[d - half] };
                dst[d] = row[d] * c[d] + rotated * s[d];
            }
        }
    }
    out
}

fn layer_norm(x: &Tensor, weight: &Tensor, bias: &Tensor, eps: f32) -> Tensor {
    let cols = *x.shape.last().unwrap();
    let w = weight.as_f32();
    let b = bias.as_f32();
    let xs = x.as_f32();
    let mut out = vec![0f32; xs.len()];
    for (row_in, row_out) in xs.chunks(cols).zip(out.chunks_mut(cols)) {
        let mean = row_in.iter().sum::<f32>() / cols as f32;
        let var = row_in.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / cols as f32;
        let inv_std = 1.0 / (var + eps).sqrt();
        for (i, v) in row_in.iter().enumerate() {
            row_out[i] = (v - mean) * inv_std * w[i] + b[i];
        }
    }
    Tensor::from_f32(x.shape.clone(), out)
}

/// One vision block: `x = x + attn(norm1(x)); x = x + mlp(norm2(x))`.
/// `segments`: patch-count boundaries for packed variable-length
/// attention (ops.md's "packed" note) — one bidirectional Sdpa per
/// segment, zero cross-segment attention. A single image is one
/// segment: `&[0, num_patches]`.
pub fn vision_block(
    x: &Tensor,
    w: &VisionBlockWeights,
    dims: VisionDims,
    cos: &[f32],
    sin: &[f32],
    segments: &[usize],
) -> Result<Tensor, BackendError> {
    let hidden = dims.hidden_size;
    let n = x.shape[0];
    let normed1 = layer_norm(x, w.norm1_weight, w.norm1_bias, 1e-6);
    let qkv = add_bias(&matmul_f32(&normed1, w.qkv_weight)?, w.qkv_bias)?;
    let qkv_data = qkv.as_f32();
    let num_heads = dims.num_heads;
    let head_dim = dims.head_dim();
    // qkv row layout: [q(hidden), k(hidden), v(hidden)] each split into
    // num_heads*head_dim — matches `.reshape(seq,3,num_heads,-1)`.
    let mut q = vec![0f32; n * hidden];
    let mut k = vec![0f32; n * hidden];
    let mut v = vec![0f32; n * hidden];
    for t in 0..n {
        let row = &qkv_data[t * 3 * hidden..(t + 1) * 3 * hidden];
        q[t * hidden..(t + 1) * hidden].copy_from_slice(&row[..hidden]);
        k[t * hidden..(t + 1) * hidden].copy_from_slice(&row[hidden..2 * hidden]);
        v[t * hidden..(t + 1) * hidden].copy_from_slice(&row[2 * hidden..]);
    }
    let q = apply_rope(&q, cos, sin, num_heads, head_dim);
    let k = apply_rope(&k, cos, sin, num_heads, head_dim);

    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut attn_out = vec![0f32; n * hidden];
    // One full (non-causal) Sdpa per segment — no attention crosses a
    // segment boundary.
    for win in segments.windows(2) {
        let (s, e) = (win[0], win[1]);
        for h in 0..num_heads {
            for ti in s..e {
                let qt = &q[(ti * num_heads + h) * head_dim..(ti * num_heads + h + 1) * head_dim];
                let mut scores = vec![0f32; e - s];
                for tj in s..e {
                    let kt = &k[(tj * num_heads + h) * head_dim..(tj * num_heads + h + 1) * head_dim];
                    scores[tj - s] = qt.iter().zip(kt.iter()).map(|(a, b)| a * b).sum::<f32>() * scale;
                }
                let max_s = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0f32;
                for sc in scores.iter_mut() {
                    *sc = (*sc - max_s).exp();
                    sum += *sc;
                }
                for sc in scores.iter_mut() {
                    *sc /= sum;
                }
                let out = &mut attn_out[(ti * num_heads + h) * head_dim..(ti * num_heads + h + 1) * head_dim];
                for tj in s..e {
                    let vt = &v[(tj * num_heads + h) * head_dim..(tj * num_heads + h + 1) * head_dim];
                    let p = scores[tj - s];
                    for (o, vv) in out.iter_mut().zip(vt.iter()) {
                        *o += p * vv;
                    }
                }
            }
        }
    }
    let attn_out = Tensor::from_f32(vec![n, hidden], attn_out);
    let proj = add_bias(&matmul_f32(&attn_out, w.proj_weight)?, w.proj_bias)?;
    let x1 = add(x, &proj);

    let normed2 = layer_norm(&x1, w.norm2_weight, w.norm2_bias, 1e-6);
    let mid = add_bias(&matmul_f32(&normed2, w.fc1_weight)?, w.fc1_bias)?;
    let mid = gelu_tanh_f32(&mid)?;
    let mlp_out = add_bias(&matmul_f32(&mid, w.fc2_weight)?, w.fc2_bias)?;
    Ok(add(&x1, &mlp_out))
}

fn add(a: &Tensor, b: &Tensor) -> Tensor {
    let sum: Vec<f32> = a.as_f32().iter().zip(b.as_f32().iter()).map(|(x, y)| x + y).collect();
    Tensor::from_f32(a.shape.clone(), sum)
}

/// `x`: `[num_patches, hidden]`, ALREADY in block-major merge-group
/// order (PatchOrder). Returns `[num_patches / merge^2, out_hidden]`.
///
/// NOTE: LayerNorm runs PER-PATCH over `hidden` features (not over the
/// merged `hidden*merge^2` width) — `use_postshuffle_norm=False` in the
/// reference normalizes before the merge-group reshape, only the two
/// Linears operate on the merged width.
pub fn patch_merger(x: &Tensor, w: &VisionMergerWeights, merge: usize) -> Result<Tensor, BackendError> {
    let hidden = x.shape[1];
    let n = x.shape[0];
    let normed = layer_norm(x, w.norm_weight, w.norm_bias, 1e-6);
    let grouped = Tensor::from_f32(vec![n / (merge * merge), hidden * merge * merge], normed.as_f32().to_vec());
    let h = add_bias(&matmul_f32(&grouped, w.fc1_weight)?, w.fc1_bias)?;
    let h = gelu_erf_f32(&h)?;
    add_bias(&matmul_f32(&h, w.fc2_weight)?, w.fc2_bias)
}

/// Full tower: patch embed → +pos_embed → RoPE angles → blocks → merger.
/// `pixel_values`: `[num_patches, patch_dim]` for ONE image (call once
/// per image for a packed multi-image batch — see module doc).
pub fn vision_tower_forward(
    pixel_values: &Tensor,
    grid_h: usize,
    grid_w: usize,
    patch_embed_weight: &Tensor,
    patch_embed_bias: &Tensor,
    pos_embed_table: &Tensor,
    blocks: &[VisionBlockWeights],
    merger: &VisionMergerWeights,
    dims: VisionDims,
) -> Result<Tensor, BackendError> {
    let positions = patch_order_positions(grid_h, grid_w, dims.spatial_merge_size);
    let mut x = patch_embed(pixel_values, patch_embed_weight, patch_embed_bias)?;
    let pos = gather_pos_embed(
        pos_embed_table,
        grid_h,
        grid_w,
        dims.spatial_merge_size,
        dims.num_grid_per_side,
    );
    x = add(&x, &pos);
    let (cos, sin) = vision_rope_cos_sin(&positions, dims.head_dim(), dims.rope_theta);
    let segments = [0usize, grid_h * grid_w];
    for blk in blocks {
        x = vision_block(&x, blk, dims, &cos, &sin, &segments)?;
    }
    patch_merger(&x, merger, dims.spatial_merge_size)
}
