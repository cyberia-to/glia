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
use crate::core::dtype::DType;
use crate::core::tensor::Tensor;

/// Dispatches to the quantized fused dequant+matmul on `backend` when `w`
/// is a real quant weight (the live model path — `w` is GPU-resident after
/// `to_backend()`), or the plain CPU matmul when `w` is already f32 (the
/// golden tests, which load real HF weights straight off disk with no
/// quantization involved). Same output either way, just different input
/// representations of the same math.
fn proj_matmul(x: &Tensor, w: &Tensor, backend: &dyn crate::backend::Backend) -> Result<Tensor, BackendError> {
    if w.dtype == DType::F32 {
        matmul_f32(x, w)
    } else {
        backend.quant_matmul(x, w)
    }
}

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
/// calls of length 1 each (the recurrence has no lookahead), PROVIDED
/// `conv_state` is also carried across calls (see its own doc below) —
/// without it, that "T calls of length 1 == one call of length T" claim
/// is false for anything but the conv1d step's very first token.
///
/// `conv_state`: `[conv_dim, kernel_size-1]`, the causal conv1d's
/// left-context cache — HF's `cache_params.layers[i].conv_states[0]`
/// (`causal_conv1d_update`). Oldest-to-newest per channel. Zero it
/// alongside `state` on a fresh conversation; every call updates it in
/// place with its trailing `kernel_size-1` (history ++ this call's `x`)
/// values, so the NEXT call sees genuine left-context instead of
/// implicit zero-padding. Missing this (as this function did before
/// 2026-09-12) makes every decode step after the first treat itself as
/// the start of a brand new sequence for the conv1d step specifically —
/// wrong output, no crash, no shape mismatch to catch it.
/// `backend`: dispatches the per-token recurrence step through
/// `Backend::gated_delta_recurrence_step` — CPU backends run it
/// in-process, honeycrisp runs the real Metal kernel (`run/backend/
/// honeycrisp/kernels/gated_delta.rs`, verified in isolation against
/// this same CPU math in `run/tests/gated_delta_honeycrisp.rs`). The
/// five big projections (`in_proj_qkv/z/b/a`, `out_proj`) also dispatch
/// through `backend.quant_matmul` (see `proj_matmul` below) whenever `w`
/// is a real quant weight, so on honeycrisp they run as on-device fused
/// dequant+matmul against GPU-resident weights — these are the dominant
/// FLOP cost at real model scale (27B: re-dequantizing them from scratch
/// on the host every token was the actual bottleneck, not the recurrence
/// step). Only conv1d, the gates, and RmsNormGated remain plain host f32
/// elementwise code — small relative to the projections' cost.
pub fn gated_delta_forward(
    x: &Tensor,
    w: &GatedDeltaWeights,
    dims: GatedDeltaDims,
    eps: f32,
    state: &mut [f32],
    conv_state: &mut [f32],
    backend: &dyn crate::backend::Backend,
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

    static STAGE_DEBUG_CALLS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let stage_debug = std::env::var("GDN_STAGE_DEBUG").is_ok() && {
        // Skip the first several hundred calls: HcPipeline lazily compiles
        // each distinct Metal kernel geometry on first use, and that
        // one-time JIT cost (many ms) would otherwise swamp steady-state
        // per-token timing. Sample once everything is warm.
        let n = STAGE_DEBUG_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        (500..508).contains(&n)
    };
    let t0 = std::time::Instant::now();

    // 1. Projections — bias-free linear, same convention as every other
    //    Linear in this crate (y = x @ W^T).
    // All four share the same `x` — one fused GPU dispatch instead of four.
    let mixed_qkv = proj_matmul(x, w.in_proj_qkv, backend)?;
    let z = proj_matmul(x, w.in_proj_z, backend)?;
    let b = proj_matmul(x, w.in_proj_b, backend)?;
    let a = proj_matmul(x, w.in_proj_a, backend)?;
    let t1 = std::time::Instant::now();

    // 2. Causal depthwise conv along T, then SiLU. Left-padded by real
    //    cross-call history (`conv_state`), not implicit zeros.
    let mixed_qkv = causal_depthwise_conv1d_silu(
        &mixed_qkv, w.conv1d_weight, cd, dims.conv_kernel_size, conv_state,
    )?;
    let t2 = std::time::Instant::now();

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
    let t3 = std::time::Instant::now();
    let mut out = vec![0f32; t * h * hv]; // [T, num_v_heads, head_v_dim], pre out_proj
    for ti in 0..t {
        // One call per TOKEN, batched over all `h` heads — matches
        // `Backend::gated_delta_recurrence_step`'s contract exactly
        // (it's the per-head loop lifted into the trait method so a
        // GPU backend can parallelize across heads in one dispatch).
        let q_t = &query_h[ti * h * hk..(ti + 1) * h * hk];
        let k_t = &key_h[ti * h * hk..(ti + 1) * h * hk];
        let v_t = &value[ti * vd..(ti + 1) * vd];
        let decay_t: Vec<f32> = g[ti * h..(ti + 1) * h].iter().map(|v| v.exp()).collect();
        let beta_t = &beta[ti * h..(ti + 1) * h];
        let out_t = backend.gated_delta_recurrence_step(state, q_t, k_t, v_t, &decay_t, beta_t, h, hk, hv)?;
        out[ti * h * hv..(ti + 1) * h * hv].copy_from_slice(&out_t);
    }

    let t4 = std::time::Instant::now();
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

    let t5 = std::time::Instant::now();
    // 8. out_proj back to hidden.
    let gated_t = Tensor::from_f32(vec![t, vd], gated);
    let result = proj_matmul(&gated_t, w.out_proj, backend)?;
    let t6 = std::time::Instant::now();
    if stage_debug {
        eprintln!(
            "GDN stage us: proj={:.1} conv={:.1} gates+l2norm={:.1} recurrence={:.1} rmsnormgated={:.1} out_proj={:.1}",
            (t1 - t0).as_secs_f64() * 1e6,
            (t2 - t1).as_secs_f64() * 1e6,
            (t3 - t2).as_secs_f64() * 1e6,
            (t4 - t3).as_secs_f64() * 1e6,
            (t5 - t4).as_secs_f64() * 1e6,
            (t6 - t5).as_secs_f64() * 1e6,
        );
    }
    debug_assert_eq!(result.shape, vec![t, hidden]);
    Ok(result)
}

/// One head's delta-rule step: decay `st` in place, read `kv_mem = st^T
/// @ k_t` (using the DECAYED state — decay happens before this read,
/// not after), rank-1-update `st += outer(k_t, (v_t - kv_mem) * beta_t)`,
/// then write `out = st^T @ q_t` (using the just-updated state). `st`:
/// `[hk, hv]` row-major, mutated in place. `out`: `[hv]`, overwritten.
///
/// Extracted as its own function so the honeycrisp GPU kernel test can
/// compare a single head's GPU output against this SAME reference call
/// on identical synthetic input, independent of the surrounding
/// projections/conv/gates — see `run/tests/gated_delta_honeycrisp.rs`.
pub fn recurrence_step(
    st: &mut [f32],
    q_t: &[f32],
    k_t: &[f32],
    v_t: &[f32],
    decay_t: f32,
    beta_t: f32,
    hk: usize,
    hv: usize,
    out: &mut [f32],
) {
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
    for o in out.iter_mut() {
        *o = 0.0;
    }
    for i in 0..hk {
        let row = &st[i * hv..(i + 1) * hv];
        let qi = q_t[i];
        for j in 0..hv {
            out[j] += row[j] * qi;
        }
    }
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
/// exactly `causal_conv1d_fn`/`causal_conv1d_update` (`activation="silu"`)
/// in the reference, generalized to work identically for both (T>1,
/// `conv_state` all-zero — first call of a conversation) and (T=1,
/// `conv_state` populated — every decode step after that).
///
/// `x`: `[T, C]`. `weight`: `[C, K]` (one length-K kernel per channel).
/// `conv_state`: `[C, K-1]`, oldest-to-newest, read as this call's left
/// context and OVERWRITTEN in place with the trailing `K-1` values of
/// `history ++ x` for the next call — see `gated_delta_forward`'s doc.
fn causal_depthwise_conv1d_silu(
    x: &Tensor,
    weight: &Tensor,
    channels: usize,
    kernel_size: usize,
    conv_state: &mut [f32],
) -> Result<Tensor, BackendError> {
    if x.shape.last() != Some(&channels) {
        return Err(BackendError::ShapeMismatch {
            op: "CausalConv1d",
            expected: vec![0, channels],
            got: x.shape.clone(),
        });
    }
    let state_len = kernel_size - 1;
    if conv_state.len() != channels * state_len {
        return Err(BackendError::ShapeMismatch {
            op: "CausalConv1d",
            expected: vec![channels, state_len],
            got: vec![conv_state.len()],
        });
    }
    let t = x.shape[0];
    let xs = x.as_f32();
    let ws = weight.as_f32();
    let mut out = vec![0f32; t * channels];
    // Extended per-channel timeline: [history (K-1) ++ this call's x (T)],
    // length K-1+T. Tap k of position `ti` (0-indexed within this call)
    // reads extended index `ti + k` (so the newest tap, k=K-1, lands on
    // extended index `ti+K-1` — `x[ti]` once `ti >= 0`, history before that).
    for c in 0..channels {
        let hist = &conv_state[c * state_len..(c + 1) * state_len];
        for ti in 0..t {
            let mut acc = 0f32;
            for k in 0..kernel_size {
                let ext_idx = ti + k; // index into the conceptual [hist ++ x] timeline
                let v = if ext_idx < state_len {
                    hist[ext_idx]
                } else {
                    xs[(ext_idx - state_len) * channels + c]
                };
                acc += v * ws[c * kernel_size + k];
            }
            out[ti * channels + c] = silu(acc);
        }
    }
    // Update conv_state to the trailing `state_len` values of the full
    // extended timeline (length state_len + t) — identical to HF's
    // `conv_state.copy_(hidden_states_new[:, :, -state_len:])`.
    for c in 0..channels {
        let hist = &conv_state[c * state_len..(c + 1) * state_len];
        // Build the new history by walking the same extended-index space
        // one more time; state_len is tiny (kernel_size-1, e.g. 3) so a
        // temporary buffer isn't worth avoiding for clarity.
        let mut new_hist = vec![0f32; state_len];
        for (i, slot) in new_hist.iter_mut().enumerate() {
            let ext_idx = t + i; // trailing state_len of [hist ++ x] (length state_len+t)
            *slot = if ext_idx < state_len {
                hist[ext_idx]
            } else {
                xs[(ext_idx - state_len) * channels + c]
            };
        }
        conv_state[c * state_len..(c + 1) * state_len].copy_from_slice(&new_hist);
    }
    Ok(Tensor::from_f32(vec![t, channels], out))
}
