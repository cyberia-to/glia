//! Gated DeltaNet — the "linear_attention" layer (Qwen3.5/3.8/3-Next).
//!
//! Spec: specs/ops.md §5 "GatedDeltaNet". Sequential (non-chunked) form,
//! verified against `transformers.models.qwen3_5.modeling_qwen3_5`
//! (`torch_recurrent_gated_delta_rule`) — see `run/tests/gated_delta.rs`
//! for the numeric comparison against real downloaded weights.
//!
//! Single-sequence only (batch=1, the shape every other op in this crate
//! already assumes for decode/prefill): tensors here are `[T, ...]`, not
//! `[B, T, ...]`. `T=1` is a decode step, `T>1` is prefill — the loop
//! below is the same either way; there is no separate "decode path"
//! because there is no KV cache to special-case (see the state note at
//! the bottom of the spec section).

use crate::backend::BackendError;
use crate::backend::cpu::matmul_f32;
use crate::core::tensor::Tensor;

/// Everything one GatedDeltaNet layer owns, host-resident f32.
pub struct GatedDeltaWeights<'a> {
    pub in_proj_qkv: &'a Tensor, // [key_dim*2 + value_dim, hidden]
    pub in_proj_z: &'a Tensor,   // [value_dim, hidden]
    pub in_proj_b: &'a Tensor,   // [num_v_heads, hidden]
    pub in_proj_a: &'a Tensor,   // [num_v_heads, hidden]
    pub conv1d_weight: &'a Tensor, // [conv_dim, kernel_size] (depthwise, bias-free)
    pub a_log: &'a Tensor,       // [num_v_heads]
    pub dt_bias: &'a Tensor,     // [num_v_heads]
    pub norm_weight: &'a Tensor, // [head_v_dim] (RmsNormGated gain)
    pub out_proj: &'a Tensor,    // [hidden, value_dim]
}

#[derive(Clone, Copy)]
pub struct GatedDeltaDims {
    pub num_v_heads: usize,
    pub num_k_heads: usize,
    pub head_k_dim: usize,
    pub head_v_dim: usize,
    pub conv_kernel_size: usize,
}

impl GatedDeltaDims {
    fn key_dim(&self) -> usize {
        self.num_k_heads * self.head_k_dim
    }
    fn value_dim(&self) -> usize {
        self.num_v_heads * self.head_v_dim
    }
    fn conv_dim(&self) -> usize {
        self.key_dim() * 2 + self.value_dim()
    }
}

/// `x`: `[T, hidden]`. Returns `[T, hidden]`.
///
/// `state`: the `[num_v_heads, head_k_dim, head_v_dim]` recurrent state,
/// owned by the caller and mutated in place — same pattern as this
/// crate's own `kv: &mut (Vec<f32>, Vec<f32>)` for Sdpa layers, except
/// fixed-size rather than append-only (ops.md's "KV-cache analogue"
/// note). Zero it once per fresh conversation (same moment
/// `reset_kv_cache` zeroes the Sdpa cache); every call after that reads
/// and updates it — there is no separate "prefill" mode, a run of T>1
/// calls with the same buffer is mathematically identical to T separate
/// calls of length 1 each (the recurrence has no lookahead).
pub fn gated_delta_forward(
    x: &Tensor,
    w: &GatedDeltaWeights,
    dims: GatedDeltaDims,
    eps: f32,
    state: &mut [f32],
) -> Result<Tensor, BackendError> {
    if x.rank() != 2 {
        return Err(BackendError::ShapeMismatch {
            op: "GatedDeltaNet",
            expected: vec![0, 0],
            got: x.shape.clone(),
        });
    }
    let t = x.shape[0];
    let hidden = x.shape[1];
    let (kd, vd, cd) = (dims.key_dim(), dims.value_dim(), dims.conv_dim());

    // 1. Projections — bias-free linear, same convention as every other
    //    Linear in this crate (y = x @ W^T).
    let mixed_qkv = matmul_f32(x, w.in_proj_qkv)?; // [T, cd]
    let z = matmul_f32(x, w.in_proj_z)?; // [T, vd]
    let b = matmul_f32(x, w.in_proj_b)?; // [T, num_v_heads]
    let a = matmul_f32(x, w.in_proj_a)?; // [T, num_v_heads]

    // 2. Causal depthwise conv along T, then SiLU. Left-padded by
    //    kernel_size-1 zeros so position i only ever sees i-k+1..=i.
    let mixed_qkv = causal_depthwise_conv1d_silu(&mixed_qkv, w.conv1d_weight, cd, dims.conv_kernel_size)?;

    // 3. Split into query/key/value and reshape to per-head.
    let mq = mixed_qkv.as_f32();
    let mut query = vec![0f32; t * kd];
    let mut key = vec![0f32; t * kd];
    let mut value = vec![0f32; t * vd];
    for ti in 0..t {
        let row = &mq[ti * cd..(ti + 1) * cd];
        query[ti * kd..(ti + 1) * kd].copy_from_slice(&row[..kd]);
        key[ti * kd..(ti + 1) * kd].copy_from_slice(&row[kd..2 * kd]);
        value[ti * vd..(ti + 1) * vd].copy_from_slice(&row[2 * kd..]);
    }

    // 4. Gates: beta = sigmoid(b); g = -exp(A_log) * softplus(a + dt_bias).
    //    Both [T, num_v_heads].
    let a_log = w.a_log.as_f32();
    let dt_bias = w.dt_bias.as_f32();
    let b_data = b.as_f32();
    let a_data = a.as_f32();
    let h = dims.num_v_heads;
    let mut beta = vec![0f32; t * h];
    let mut g = vec![0f32; t * h];
    for ti in 0..t {
        for hi in 0..h {
            let idx = ti * h + hi;
            beta[idx] = sigmoid(b_data[idx]);
            g[idx] = -a_log[hi].exp() * softplus(a_data[idx] + dt_bias[hi]);
        }
    }

    // 5. K/Q head expansion (num_v_heads / num_k_heads, repeat_interleave —
    //    same rule as GQA, ops.md §5) then per-head L2-norm, then query
    //    scale. Expansion happens BEFORE norm so every expanded copy
    //    normalizes identically (norm has no memory of the source head).
    let ratio = dims.num_v_heads / dims.num_k_heads;
    if dims.num_v_heads % dims.num_k_heads != 0 {
        return Err(BackendError::InvalidInput {
            op: "GatedDeltaNet",
            reason: format!(
                "num_v_heads {} not a multiple of num_k_heads {}",
                dims.num_v_heads, dims.num_k_heads
            ),
        });
    }
    let hk = dims.head_k_dim;
    let hv = dims.head_v_dim;
    let mut query_h = vec![0f32; t * h * hk]; // expanded to num_v_heads
    let mut key_h = vec![0f32; t * h * hk];
    for ti in 0..t {
        for hi in 0..h {
            let src = hi / ratio; // which of the num_k_heads this v-head reads
            let q_src = &query[ti * kd + src * hk..ti * kd + (src + 1) * hk];
            let k_src = &key[ti * kd + src * hk..ti * kd + (src + 1) * hk];
            let q_dst = &mut query_h[(ti * h + hi) * hk..(ti * h + hi + 1) * hk];
            let k_dst = &mut key_h[(ti * h + hi) * hk..(ti * h + hi + 1) * hk];
            l2_normalize_into(q_src, q_dst, 1e-6);
            l2_normalize_into(k_src, k_dst, 1e-6);
            let scale = 1.0 / (hk as f32).sqrt();
            for v in q_dst.iter_mut() {
                *v *= scale;
            }
        }
    }

    // 6. Sequential recurrence — the delta rule. One [hk, hv] state
    //    matrix per head, decayed and rank-1-updated every token, carried
    //    in the caller-owned `state` buffer across calls.
    if state.len() != h * hk * hv {
        return Err(BackendError::ShapeMismatch {
            op: "GatedDeltaNet",
            expected: vec![h, hk, hv],
            got: vec![state.len()],
        });
    }
    let mut out = vec![0f32; t * h * hv]; // [T, num_v_heads, head_v_dim], pre out_proj
    for ti in 0..t {
        for hi in 0..h {
            let st = &mut state[hi * hk * hv..(hi + 1) * hk * hv];
            let q_t = &query_h[(ti * h + hi) * hk..(ti * h + hi + 1) * hk];
            let k_t = &key_h[(ti * h + hi) * hk..(ti * h + hi + 1) * hk];
            let v_t = &value[ti * vd + hi * hv..ti * vd + (hi + 1) * hv];
            let decay_t = g[ti * h + hi].exp();
            let beta_t = beta[ti * h + hi];

            for s in st.iter_mut() {
                *s *= decay_t;
            }
            // kv_mem[j] = sum_i state[i,j] * k_t[i]  (k_t @ state)
            let mut kv_mem = vec![0f32; hv];
            for i in 0..hk {
                let row = &st[i * hv..(i + 1) * hv];
                let ki = k_t[i];
                for j in 0..hv {
                    kv_mem[j] += row[j] * ki;
                }
            }
            // delta = (v_t - kv_mem) * beta_t; state += outer(k_t, delta)
            let mut delta = vec![0f32; hv];
            for j in 0..hv {
                delta[j] = (v_t[j] - kv_mem[j]) * beta_t;
            }
            for i in 0..hk {
                let ki = k_t[i];
                let row = &mut st[i * hv..(i + 1) * hv];
                for j in 0..hv {
                    row[j] += ki * delta[j];
                }
            }
            // out[j] = sum_i state[i,j] * q_t[i]  (q_t @ state, post-update)
            let out_row = &mut out[(ti * h + hi) * hv..(ti * h + hi + 1) * hv];
            for i in 0..hk {
                let row = &st[i * hv..(i + 1) * hv];
                let qi = q_t[i];
                for j in 0..hv {
                    out_row[j] += row[j] * qi;
                }
            }
        }
    }

    // 7. RmsNormGated: normalize (reduction over head_v_dim only), THEN
    //    multiply by the learned gain, THEN by Silu(z) — norm before
    //    gate, not the other way around (ops.md's explicit callout).
    let norm_w = w.norm_weight.as_f32();
    let z_data = z.as_f32();
    let mut gated = vec![0f32; t * h * hv];
    for ti in 0..t {
        for hi in 0..h {
            let row = &out[(ti * h + hi) * hv..(ti * h + hi + 1) * hv];
            let z_row = &z_data[ti * vd + hi * hv..ti * vd + (hi + 1) * hv];
            let dst = &mut gated[(ti * h + hi) * hv..(ti * h + hi + 1) * hv];
            let ms: f32 = row.iter().map(|v| v * v).sum::<f32>() / hv as f32;
            let inv = 1.0 / (ms + eps).sqrt();
            for j in 0..hv {
                dst[j] = row[j] * inv * norm_w[j] * silu(z_row[j]);
            }
        }
    }

    // 8. out_proj back to hidden.
    let gated_t = Tensor::from_f32(vec![t, vd], gated);
    let result = matmul_f32(&gated_t, w.out_proj)?;
    debug_assert_eq!(result.shape, vec![t, hidden]);
    Ok(result)
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn silu(x: f32) -> f32 {
    x * sigmoid(x)
}

/// `softplus(x) = log(1 + exp(x))`, computed stably for large |x| the way
/// every real implementation does (PyTorch's `F.softplus` included) —
/// the naive form overflows `exp` for x gtr ~88 in f32.
fn softplus(x: f32) -> f32 {
    if x > 20.0 {
        x
    } else {
        (1.0 + x.exp()).ln()
    }
}

/// `x * rsqrt(sum(x^2) + eps)` — eps goes INSIDE the sqrt (RMSNorm-style),
/// matching `transformers.models.qwen3_5.modeling_qwen3_5.l2norm` exactly
/// ("intended to align with the l2norm implementation in the FLA
/// library"). `1/max(norm, eps)` looks equivalent for well-scaled inputs
/// but is a different function — it was the first (wrong) guess here and
/// cost ~1.4% relative error at the golden test's worst element before
/// being checked against the real formula.
fn l2_normalize_into(src: &[f32], dst: &mut [f32], eps: f32) {
    let sum_sq: f32 = src.iter().map(|v| v * v).sum();
    let inv = (sum_sq + eps).powf(-0.5);
    for (d, s) in dst.iter_mut().zip(src.iter()) {
        *d = s * inv;
    }
}

/// Depthwise causal 1D conv (groups = channels, no bias) + SiLU, fused —
/// exactly `causal_conv1d_fn(..., activation="silu")` in the reference.
/// `x`: `[T, C]`. `weight`: `[C, K]` (one length-K kernel per channel).
fn causal_depthwise_conv1d_silu(
    x: &Tensor,
    weight: &Tensor,
    channels: usize,
    kernel_size: usize,
) -> Result<Tensor, BackendError> {
    if x.shape.last() != Some(&channels) {
        return Err(BackendError::ShapeMismatch {
            op: "CausalConv1d",
            expected: vec![0, channels],
            got: x.shape.clone(),
        });
    }
    let t = x.shape[0];
    let xs = x.as_f32();
    let ws = weight.as_f32();
    let mut out = vec![0f32; t * channels];
    for ti in 0..t {
        for c in 0..channels {
            let mut acc = 0f32;
            // Kernel tap k reads input position (ti - (kernel_size-1) + k);
            // positions before 0 are the implicit left zero-padding.
            for k in 0..kernel_size {
                let src_t = ti as isize - (kernel_size as isize - 1) + k as isize;
                if src_t < 0 {
                    continue;
                }
                acc += xs[src_t as usize * channels + c] * ws[c * kernel_size + k];
            }
            out[ti * channels + c] = silu(acc);
        }
    }
    Ok(Tensor::from_f32(vec![t, channels], out))
}
