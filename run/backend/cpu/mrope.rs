//! 3D interleaved mRoPE — Qwen3.5/3.8's multimodal position-id scheme.
//!
//! Two independent pieces, both pure index/config arithmetic (no
//! weights): `mrope_position_ids` (HF's `Qwen3_5Model.get_rope_index`)
//! builds a `(t,h,w)` position triple per token; `mrope_cos_sin` (HF's
//! `Qwen3_5TextRotaryEmbedding.forward` + `recomposition_frequencies`)
//! turns those triples into per-token cos/sin for the rotated portion
//! of the head dim. Verified against real transformers output —
//! `run/tests/mrope_golden.rs`.
//!
//! Only the `full_attention` (Sdpa) layers ever call this —
//! `linear_attention` (GatedDeltaNet) layers never rotate at all, and
//! text-only sequences (`t==h==w` for every token) make this
//! numerically identical to plain 1D RoPE — see ops.md.
//!
//! Spec: specs/ops.md §3 "Rope", gated-delta-vl-plan.md's fusion trace.

/// One contiguous run of the token sequence, all the same modality.
pub enum ModalityRun {
    /// `len` consecutive text tokens.
    Text { len: usize },
    /// One image or one video frame: `(t, h, w)` = the RAW (pre-merge)
    /// patch grid from `grid_thw`, as delivered by the vision
    /// preprocessor / vision tower caller.
    Vision { t: usize, h: usize, w: usize },
}

/// HF's `Qwen3_5Model.get_rope_index`, single-sequence (no padding/
/// batch — the batched/masked case is a caller concern this runtime's
/// one-token-at-a-time decode loop doesn't need yet).
///
/// Returns `[pos_t, pos_h, pos_w]`, one `f32` per token per axis, and
/// `next_pos` — the running counter's value after the whole sequence
/// (feed as `start_position` to continue into a later call, e.g. once
/// decode moves past the prefixed prompt).
pub fn mrope_position_ids(runs: &[ModalityRun], spatial_merge_size: usize) -> ([Vec<f32>; 3], usize) {
    let mut t_pos = Vec::new();
    let mut h_pos = Vec::new();
    let mut w_pos = Vec::new();
    let mut current_pos = 0usize;

    for run in runs {
        match *run {
            ModalityRun::Text { len } => {
                for i in 0..len {
                    let p = (current_pos + i) as f32;
                    t_pos.push(p);
                    h_pos.push(p);
                    w_pos.push(p);
                }
                current_pos += len;
            }
            ModalityRun::Vision { t, h, w } => {
                let llm_t = t; // temp_merge_size is always 1 for images/videos here
                let llm_h = h / spatial_merge_size;
                let llm_w = w / spatial_merge_size;
                // meshgrid(T,H,W, indexing="ij") flattened: T outer, H middle, W inner.
                for ti in 0..llm_t {
                    for hi in 0..llm_h {
                        for wi in 0..llm_w {
                            t_pos.push((ti + current_pos) as f32);
                            h_pos.push((hi + current_pos) as f32);
                            w_pos.push((wi + current_pos) as f32);
                        }
                    }
                }
                current_pos += h.max(w) / spatial_merge_size;
            }
        }
    }
    ([t_pos, h_pos, w_pos], current_pos)
}

/// `rope_theta`/`partial_rotary_factor` give `rope_dim` (even, ≤
/// `head_dim`) the same way `LlamaConfig::layer_rope_dim` does for the
/// plain partial-rotary case. `mrope_section` (e.g. `[11,11,10]`, HF
/// config field) must sum to `rope_dim/2`.
///
/// Returns `(cos, sin)`, each `seq_len * rope_dim` f32, row-major per
/// token — same layout `rope.rs::rope_f32` expects for its rotated
/// slice.
pub fn mrope_cos_sin(
    positions: &[[f32; 3]],
    rope_dim: usize,
    rope_theta: f32,
    mrope_section: [usize; 3],
) -> (Vec<f32>, Vec<f32>) {
    let n_freq = rope_dim / 2;
    debug_assert_eq!(mrope_section.iter().sum::<usize>(), n_freq);
    let inv_freq: Vec<f32> = (0..n_freq)
        .map(|i| 1.0 / rope_theta.powf((2 * i) as f32 / rope_dim as f32))
        .collect();

    let seq_len = positions.len();
    let mut cos = vec![0f32; seq_len * rope_dim];
    let mut sin = vec![0f32; seq_len * rope_dim];
    for (t, pos) in positions.iter().enumerate() {
        let mut half = vec![0f32; n_freq];
        for j in 0..n_freq {
            // Interleaved ownership: frequency index j belongs to axis
            // (j % 3) — 0=T, 1=H, 2=W (the `slice(offset, mrope_section[dim]*3, 3)`
            // formula in the reference reduces to exactly this pattern;
            // verified against mrope_section=[11,11,10] summing to n_freq).
            let axis = j % 3;
            half[j] = pos[axis] * inv_freq[j];
        }
        let base = t * rope_dim;
        for j in 0..n_freq {
            let c = half[j].cos();
            let s = half[j].sin();
            cos[base + j] = c;
            cos[base + n_freq + j] = c;
            sin[base + j] = s;
            sin[base + n_freq + j] = s;
        }
    }
    (cos, sin)
}

/// Rotates `x` (`[..., head_dim]`) using precomputed per-token
/// `cos`/`sin` (`[seq_len, rope_dim]`, from `mrope_cos_sin`).
///
/// **Different index convention from `rope.rs::rope_f32`** — that
/// function (built for Gemma-4's "proportional" rope_type) pairs
/// dim `j` with dim `j + head_dim/2` across the FULL head_dim, with
/// the unrotated portion interleaved as zero-frequency (identity)
/// pairs. Qwen3.5/3.8 (`rope_type="default"`, HF's
/// `Qwen3_5Attention`/generic `apply_rotary_pos_emb`) instead treats
/// the rotated dims as a plain CONTIGUOUS PREFIX: `x_rot = x[..
/// rope_dim]` gets `rotate_half` applied treating that prefix as its
/// own self-contained vector (splitting it at `rope_dim/2`, not
/// `head_dim/2`), and `x_pass = x[rope_dim..head_dim]` is copied
/// through UNCHANGED as a contiguous tail. Verified against real
/// `apply_rotary_pos_emb` output in `run/tests/mrope_golden.rs` — do
/// not "simplify" this to reuse `rope_f32`, the two are only
/// equivalent when `rope_dim == head_dim`.
pub fn apply_rope_cos_sin_f32(x: &[f32], cos: &[f32], sin: &[f32], head_dim: usize, rope_dim: usize) -> Vec<f32> {
    let half = rope_dim / 2;
    let n = x.len() / head_dim;
    let mut out = vec![0f32; x.len()];
    for row in 0..n {
        let x_row = &x[row * head_dim..(row + 1) * head_dim];
        let c = &cos[row * rope_dim..(row + 1) * rope_dim];
        let s = &sin[row * rope_dim..(row + 1) * rope_dim];
        let out_row = &mut out[row * head_dim..(row + 1) * head_dim];
        for j in 0..half {
            let x1 = x_row[j];
            let x2 = x_row[j + half];
            out_row[j] = x1 * c[j] - x2 * s[j];
            out_row[j + half] = x2 * c[j + half] + x1 * s[j + half];
        }
        out_row[rope_dim..head_dim].copy_from_slice(&x_row[rope_dim..head_dim]);
    }
    out
}
