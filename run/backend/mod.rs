//! Execution backends.
//!
//! Trait + implementations: cpu (portable reference), wgpu (GPU),
//! honeycrisp (Apple Silicon turbo). Adding a backend = new submodule here.
//!
//! Spec: specs/execution.md, specs/architecture.md

pub mod cpu;
pub mod wgpu;

#[cfg(target_os = "macos")]
pub mod honeycrisp;

use crate::core::dtype::DType;
use crate::core::op::Op;
use crate::core::tensor::Tensor;
use thiserror::Error;

/// Per-layer inputs for `forward_decode_fused_layers`.
/// All quant weight tensors must be GPU-resident (after `to_backend`).
pub struct LayerFusedInput<'a> {
    pub input_norm:   &'a Tensor,
    pub q_proj:       &'a Tensor,
    pub k_proj:       &'a Tensor,
    pub v_proj:       &'a Tensor,
    pub q_bias:       Option<&'a Tensor>,
    pub k_bias:       Option<&'a Tensor>,
    pub v_bias:       Option<&'a Tensor>,
    pub q_norm:       Option<&'a Tensor>,
    pub k_norm:       Option<&'a Tensor>,
    pub o_proj:       &'a Tensor,
    pub post_norm:    &'a Tensor,
    pub gate_proj:    &'a Tensor,
    pub up_proj:      &'a Tensor,
    pub down_proj:    &'a Tensor,
    pub num_heads:      u32,
    pub kv_heads:       u32,
    pub head_dim:       u32,
    pub rope_dim:       u32,
    pub rope_theta:     f32,
    pub attn_scale:     f32,
    pub window:         u32,
    pub layer_idx:      usize,
    /// Gemma-2/3/4: RmsNorm applied to attention output before residual add.
    pub post_attn_norm: Option<&'a Tensor>,
    /// Gemma-2/3/4: RmsNorm applied to FFN output before residual add.
    pub post_ffw_norm:  Option<&'a Tensor>,
    /// Gemma-4: use gelu_pytorch_tanh instead of SiLU for the FFN gate.
    pub use_gelu_tanh:  bool,
    /// Gemma-4: per-layer scalar applied to layer output (1.0 = identity).
    pub layer_output_scale: f32,
}

/// Three backends + cpu reference library.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BackendKind {
    /// Pure-Rust CPU reference. Always correct, slow.
    /// Not user-facing; used internally by wgpu+rs for op fallback.
    Cpu,
    /// Portable wgpu GPU + CPU fallback.
    WgpuRs,
    /// Apple Silicon turbo (Metal + ANE + AMX + NEON + unimem).
    Honeycrisp,
    /// Future: trident-compiled bytecode on deterministic VM.
    Nox,
}

impl BackendKind {
    pub fn as_str(self) -> &'static str {
        match self {
            BackendKind::Cpu => "cpu",
            BackendKind::WgpuRs => "wgpu+rs",
            BackendKind::Honeycrisp => "honeycrisp",
            BackendKind::Nox => "nox",
        }
    }
}

/// Errors surfaced by backends. Structured, actionable.
#[derive(Error, Debug)]
pub enum BackendError {
    #[error("backend {backend} does not support op {op} with dtype {input_dtype:?}")]
    UnsupportedOp {
        backend: &'static str,
        op: &'static str,
        input_dtype: DType,
    },

    #[error("shape mismatch in {op}: expected {expected:?}, got {got:?}")]
    ShapeMismatch {
        op: &'static str,
        expected: Vec<usize>,
        got: Vec<usize>,
    },

    #[error("invalid input to {op}: {reason}")]
    InvalidInput { op: &'static str, reason: String },

    #[error("out of memory on {backend}: requested {requested_bytes} bytes")]
    OutOfMemory {
        backend: &'static str,
        requested_bytes: usize,
    },

    #[error("backend {backend} init failed: {reason}")]
    BackendInit {
        backend: &'static str,
        reason: String,
    },

    #[error("dtype {dtype:?} not implemented in {backend} (blocker: {blocker})")]
    UnsupportedDtype {
        backend: &'static str,
        dtype: DType,
        blocker: &'static str,
    },

    #[error("context overflow: requested position {pos}, max {max}")]
    ContextOverflow { pos: usize, max: usize },

    #[error("NaN/Inf detected in {op} output (layer {layer}, position {pos})")]
    NonFiniteOutput {
        op: &'static str,
        layer: usize,
        pos: usize,
    },

    #[error("tensor: {0}")]
    Tensor(#[from] crate::core::tensor::TensorError),

    #[error("internal: {0}")]
    Internal(String),
}

/// What every backend must implement.
///
/// The CPU reference library implements ALL ops in f32.
/// GPU backends implement what they support; missing ops route to CPU.
pub trait Backend: Send + Sync {
    /// Identity for logging and dispatch.
    fn kind(&self) -> BackendKind;

    /// True if this backend has a native implementation of `op` for the
    /// given input tensor shapes/dtypes.
    ///
    /// If false, the dispatcher falls back to CPU reference.
    fn supports(&self, op: &Op, inputs: &[&Tensor]) -> bool;

    /// Execute one op. Inputs are guaranteed to be on this backend's memory
    /// (caller uploads first if needed).
    ///
    /// Returns output tensor(s), or a structured error.
    fn execute(&self, op: &Op, inputs: &[&Tensor]) -> Result<Vec<Tensor>, BackendError>;

    /// Upload host bytes to backend memory.
    fn upload(
        &self,
        bytes: &[u8],
        shape: Vec<usize>,
        dtype: DType,
    ) -> Result<Tensor, BackendError>;

    /// Move a host tensor to backend memory. Default implementation
    /// uploads raw bytes; backends may override for zero-copy paths.
    fn to_backend(&self, t: &Tensor) -> Result<Tensor, BackendError> {
        match &t.data {
            crate::core::tensor::TensorData::Host(bytes) => {
                self.upload(bytes.as_slice(), t.shape.clone(), t.dtype)
            }
            crate::core::tensor::TensorData::Backend(_) => Ok(t.clone()),
        }
    }

    /// Download tensor to host memory as F32.
    /// For quantized inputs, dequantizes to F32 on download.
    fn download_f32(&self, t: &Tensor) -> Result<Vec<f32>, BackendError>;

    /// True if this backend pre-uploads quant weight bytes during to_backend().
    /// Backends that override quant_matmul to use GPU-resident tensors return true.
    /// Default false: quant weight tensors stay host-resident.
    fn uploads_quant_weights(&self) -> bool { false }

    /// Number of untimed warmup forward passes the bench should run after the
    /// first-forward measurement and before the timed decode loop.
    /// GPU backends that need DVFS ramp-up time return a non-zero value here
    /// so the budget-capped timed steps all land in steady state.
    fn decode_warmup_steps(&self) -> usize { 0 }

    /// Fused dequantize+matmul for quantized weight matrices.
    ///
    /// `x` is f32 [..., K]. `w` is a weight Tensor with dtype Q4_K/Q6_K/etc
    /// and shape [N, K]. After `to_backend()`, `w` may be GPU-resident.
    ///
    /// Default: extracts host bytes from `w` and delegates to CPU kernel.
    /// GPU backends override to run on-device kernels using `w` directly.
    fn quant_matmul(&self, x: &Tensor, w: &Tensor) -> Result<Tensor, BackendError> {
        let bytes = w.as_host_bytes().ok_or_else(|| BackendError::Internal(
            "quant_matmul default: w is not host-resident (backend missing override)".into(),
        ))?;
        let n = w.shape[0];
        let k = w.shape[1];
        crate::backend::cpu::matmul_quant_f32(x, bytes, w.dtype, n, k)
    }

    /// Batched fused dequant+matmul: y_i = x @ w_i^T for each w in `ws`.
    ///
    /// All matmuls share the same `x`. GPU backends should encode every
    /// dispatch into a single command buffer with one wait — saves the
    /// fixed per-op submit overhead. Default: iterate `quant_matmul`.
    fn quant_matmul_multi(
        &self,
        x: &Tensor,
        ws: &[&Tensor],
    ) -> Result<Vec<Tensor>, BackendError> {
        ws.iter().map(|w| self.quant_matmul(x, w)).collect()
    }

    /// Batched RmsNorm: encode several independent norms into one command buffer.
    /// Default: iterate.
    fn rms_norm_multi(
        &self,
        pairs: &[(&Tensor, &Tensor)],
        eps: f32,
    ) -> Result<Vec<Tensor>, BackendError> {
        fallback_rms_norm_multi(self, pairs, eps)
    }

    /// Fused RmsNorm followed by N quant matmuls against the normalized output.
    /// GPU backends should encode all (norm + N matmuls) into one command buffer
    /// with a single wait. Default: norm op + iterate quant_matmul.
    fn fused_norm_quant_matmul_multi(
        &self,
        x: &Tensor,
        gamma: &Tensor,
        eps: f32,
        ws: &[&Tensor],
    ) -> Result<Vec<Tensor>, BackendError> {
        fallback_fused_norm_quant_matmul_multi(self, x, gamma, eps, ws)
    }

    /// Fused: RmsNorm(hidden) → q_proj/k_proj/v_proj → qk_norm(Q)/qk_norm(K).
    /// Returns (q_norm, k_norm, v) in that order — all in ONE GPU command buffer.
    /// Default: separate per-op call chain.
    fn fused_norm_qkv_qknorm(
        &self,
        hidden: &Tensor,
        input_norm_gamma: &Tensor,
        q_proj_w: &Tensor,
        k_proj_w: &Tensor,
        v_proj_w: &Tensor,
        q_norm_gamma: &Tensor,
        k_norm_gamma: &Tensor,
        eps: f32,
        num_q_heads: usize,
        num_k_heads: usize,
        head_dim: usize,
    ) -> Result<(Tensor, Tensor, Tensor), BackendError> {
        fallback_fused_norm_qkv_qknorm(
            self, hidden, input_norm_gamma, q_proj_w, k_proj_w, v_proj_w,
            q_norm_gamma, k_norm_gamma, eps, num_q_heads, num_k_heads, head_dim,
        )
    }

    /// True if backend can compute scaled-dot-product attention with its own
    /// GPU-resident KV cache. When true, callers should use `gpu_attention`
    /// instead of running attention manually on the CPU.
    fn supports_gpu_attention(&self) -> bool { false }

    /// Reset the backend's GPU KV cache (e.g. on `reset_kv_cache`).
    fn reset_gpu_kv_cache(&self) {}

    /// GPU-side decode-mode SDPA. Backend owns the KV cache (per layer_idx),
    /// appends `k`,`v` at `position`, runs attention against the full cached
    /// sequence, and returns the output. Default returns "unsupported".
    fn gpu_attention(
        &self,
        q: &Tensor,           // [num_heads, head_dim] f32 (post-RoPE)
        k: &Tensor,           // [kv_heads, head_dim]   f32 (post-RoPE)
        v: &Tensor,           // [kv_heads, head_dim]   f32 (post-V-norm if any)
        layer_idx: usize,
        position: usize,      // = past_seq_len, write-offset in cache
        num_heads: u32,
        kv_heads: u32,
        head_dim: u32,
        max_seq: u32,
        scale: f32,
        window: u32,          // 0 = no sliding window
    ) -> Result<Tensor, BackendError> {
        let _ = (q, k, v, layer_idx, position, num_heads, kv_heads, head_dim, max_seq, scale, window);
        Err(BackendError::Internal("gpu_attention not supported on this backend".into()))
    }

    /// Fused attention block: kv_append + attention + o_proj + residual_add.
    /// Returns hidden_in + o_proj(attention(q, k_cache, v_cache)).
    /// Default: compose individual ops.
    fn fused_attn_oproj_residual(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        hidden_in: &Tensor,
        o_proj_w: &Tensor,
        layer_idx: usize,
        position: usize,
        num_heads: u32,
        kv_heads: u32,
        head_dim: u32,
        max_seq: u32,
        scale: f32,
        window: u32,
    ) -> Result<Tensor, BackendError> {
        fallback_fused_attn_oproj_residual(
            self, q, k, v, hidden_in, o_proj_w, layer_idx, position,
            num_heads, kv_heads, head_dim, max_seq, scale, window,
        )
    }

    /// Fused FFN block: post_norm + gate + up + silu*up + down + residual_add.
    /// Returns hidden_in + down_proj(silu(gate(norm(hidden_in))) * up(...)).
    /// Default = composition of per-op calls.
    fn fused_ffn_residual(
        &self,
        hidden_in: &Tensor,
        post_norm_gamma: &Tensor,
        gate_w: &Tensor,
        up_w: &Tensor,
        down_w: &Tensor,
        eps: f32,
    ) -> Result<Tensor, BackendError> {
        fallback_fused_ffn_residual(self, hidden_in, post_norm_gamma, gate_w, up_w, down_w, eps)
    }

    /// Fused FFN: post_norm(hidden) → gate, up → silu(gate)*up → down_proj.
    /// Single GPU command buffer, single wait. Default: per-op on CPU.
    fn fused_norm_swiglu_down(
        &self,
        hidden: &Tensor,
        post_norm_gamma: &Tensor,
        gate_w: &Tensor,
        up_w: &Tensor,
        down_w: &Tensor,
        eps: f32,
    ) -> Result<Tensor, BackendError> {
        fallback_fused_norm_swiglu_down(self, hidden, post_norm_gamma, gate_w, up_w, down_w, eps)
    }

    /// Run all transformer layers in a single GPU command buffer (decode mode).
    /// Returns `Ok(Some(hidden))` on success, `Ok(None)` if not supported by
    /// this backend or for this model configuration — caller falls back to
    /// the per-layer loop.
    fn forward_decode_fused_layers(
        &self,
        hidden: &Tensor,
        layers: &[LayerFusedInput<'_>],
        past_seq_len: usize,
        max_seq: u32,
        eps: f32,
    ) -> Result<Option<Tensor>, BackendError> {
        let _ = (hidden, layers, past_seq_len, max_seq, eps);
        Ok(None)
    }

    /// Fused SiLU(gate) * up. Default: split into two ops on CPU.
    /// GPU backends override with a single kernel — half the memory bandwidth
    /// and a single dispatch instead of two.
    /// Qwen3.5/3.8 interleaved-mRoPE Q/K rotation — contiguous `rope_dim`
    /// prefix rotated (own `rotate_half`, split at `rope_dim/2`), the
    /// `head_dim - rope_dim` tail passed through unchanged. NOT the same
    /// index convention as `Op::Rope` (built for Gemma-4's interleaved-
    /// pairs-across-full-head_dim scheme) — see
    /// `backend::cpu::mrope::apply_rope_cos_sin_f32`'s doc comment and
    /// ops.md §"mRoPE, interleaved". `x`: `[n_rows, head_dim]`. `cos`/
    /// `sin`: `[rope_dim]`, one token's precomputed frequencies
    /// (duplicated across the two halves — `mrope_cos_sin`'s output).
    /// Default: CPU reference. GPU backends override with a kernel.
    fn apply_mrope_cos_sin(
        &self,
        x: &Tensor,
        cos: &[f32],
        sin: &[f32],
        head_dim: usize,
        rope_dim: usize,
    ) -> Result<Tensor, BackendError> {
        let x_data = if let Some(b) = x.as_host_bytes() {
            bytemuck::cast_slice::<u8, f32>(b).to_vec()
        } else {
            self.download_f32(x)?
        };
        let out = crate::backend::cpu::mrope::apply_rope_cos_sin_f32(&x_data, cos, sin, head_dim, rope_dim);
        Ok(Tensor::from_f32(x.shape.clone(), out))
    }

    fn silu_mul(&self, gate: &Tensor, up: &Tensor) -> Result<Tensor, BackendError> {
        // Default falls back to host f32 path.
        let g = if let Some(b) = gate.as_host_bytes() {
            bytemuck::cast_slice::<u8, f32>(b).to_vec()
        } else {
            self.download_f32(gate)?
        };
        let u = if let Some(b) = up.as_host_bytes() {
            bytemuck::cast_slice::<u8, f32>(b).to_vec()
        } else {
            self.download_f32(up)?
        };
        let out: Vec<f32> = g.iter().zip(u.iter())
            .map(|(g, u)| (g / (1.0 + (-g).exp())) * u)
            .collect();
        Ok(Tensor::from_f32(gate.shape.clone(), out))
    }
}

// ── unfused fallbacks ────────────────────────────────────────────────────────
//
// The composed-of-smaller-ops bodies of the fused trait methods, as free
// functions, so an *implementation* whose fast path is ineligible can reach
// them. They used to live only as trait default bodies, and a backend that
// wrote `return Backend::method(self, ...)` to "call the default" was in fact
// calling its own override — infinite tail recursion, which LLVM dutifully
// turned into a branch-to-self. soma's mind spun exactly there for as long as
// anyone cared to wait, at 99% CPU, three frames deep into the first layer of
// the first token of any model whose weights had not been uploaded yet.

pub fn fallback_rms_norm_multi<B: Backend + ?Sized>(
    b: &B,
    pairs: &[(&Tensor, &Tensor)],
    eps: f32,
) -> Result<Vec<Tensor>, BackendError> {
    pairs.iter()
        .map(|(x, g)| b.execute(&crate::core::op::Op::RmsNorm { eps }, &[x, g]).map(|mut v| v.remove(0)))
        .collect()
}

pub fn fallback_fused_norm_quant_matmul_multi<B: Backend + ?Sized>(
    b: &B,
    x: &Tensor,
    gamma: &Tensor,
    eps: f32,
    ws: &[&Tensor],
) -> Result<Vec<Tensor>, BackendError> {
    let normed = b.execute(&crate::core::op::Op::RmsNorm { eps }, &[x, gamma])?
        .remove(0);
    b.quant_matmul_multi(&normed, ws)
}

#[allow(clippy::too_many_arguments)]
pub fn fallback_fused_norm_qkv_qknorm<B: Backend + ?Sized>(
    b: &B,
    hidden: &Tensor,
    input_norm_gamma: &Tensor,
    q_proj_w: &Tensor,
    k_proj_w: &Tensor,
    v_proj_w: &Tensor,
    q_norm_gamma: &Tensor,
    k_norm_gamma: &Tensor,
    eps: f32,
    num_q_heads: usize,
    num_k_heads: usize,
    head_dim: usize,
) -> Result<(Tensor, Tensor, Tensor), BackendError> {
    let qkv = b.fused_norm_quant_matmul_multi(
        hidden, input_norm_gamma, eps, &[q_proj_w, k_proj_w, v_proj_w],
    )?;
    let mut it = qkv.into_iter();
    let q = it.next().unwrap();
    let k = it.next().unwrap();
    let v = it.next().unwrap();
    // Reshape q/k for per-head norm. For host tensors this is metadata-only.
    let q_reshaped = Tensor { shape: vec![num_q_heads, head_dim], dtype: q.dtype, data: q.data };
    let k_reshaped = Tensor { shape: vec![num_k_heads, head_dim], dtype: k.dtype, data: k.data };
    let normed = b.rms_norm_multi(
        &[(&q_reshaped, q_norm_gamma), (&k_reshaped, k_norm_gamma)],
        eps,
    )?;
    let mut nit = normed.into_iter();
    let q_n = nit.next().unwrap();
    let k_n = nit.next().unwrap();
    // Reshape back
    let q_flat = Tensor { shape: vec![1, num_q_heads * head_dim], dtype: q_n.dtype, data: q_n.data };
    let k_flat = Tensor { shape: vec![1, num_k_heads * head_dim], dtype: k_n.dtype, data: k_n.data };
    Ok((q_flat, k_flat, v))
}

#[allow(clippy::too_many_arguments)]
pub fn fallback_fused_attn_oproj_residual<B: Backend + ?Sized>(
    b: &B,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    hidden_in: &Tensor,
    o_proj_w: &Tensor,
    layer_idx: usize,
    position: usize,
    num_heads: u32,
    kv_heads: u32,
    head_dim: u32,
    max_seq: u32,
    scale: f32,
    window: u32,
) -> Result<Tensor, BackendError> {
    let attn = b.gpu_attention(
        q, k, v, layer_idx, position,
        num_heads, kv_heads, head_dim, max_seq, scale, window,
    )?;
    let attn_proj = b.quant_matmul(&attn, o_proj_w)?;
    b.execute(&crate::core::op::Op::Add, &[hidden_in, &attn_proj])
        .map(|mut v| v.remove(0))
}

pub fn fallback_fused_ffn_residual<B: Backend + ?Sized>(
    b: &B,
    hidden_in: &Tensor,
    post_norm_gamma: &Tensor,
    gate_w: &Tensor,
    up_w: &Tensor,
    down_w: &Tensor,
    eps: f32,
) -> Result<Tensor, BackendError> {
    let ffn_out = b.fused_norm_swiglu_down(
        hidden_in, post_norm_gamma, gate_w, up_w, down_w, eps,
    )?;
    b.execute(&crate::core::op::Op::Add, &[hidden_in, &ffn_out])
        .map(|mut v| v.remove(0))
}

pub fn fallback_fused_norm_swiglu_down<B: Backend + ?Sized>(
    b: &B,
    hidden: &Tensor,
    post_norm_gamma: &Tensor,
    gate_w: &Tensor,
    up_w: &Tensor,
    down_w: &Tensor,
    eps: f32,
) -> Result<Tensor, BackendError> {
    let gate_up = b.fused_norm_quant_matmul_multi(
        hidden, post_norm_gamma, eps, &[gate_w, up_w],
    )?;
    let mut it = gate_up.into_iter();
    let gate = it.next().unwrap();
    let up = it.next().unwrap();
    let mid = b.silu_mul(&gate, &up)?;
    b.quant_matmul(&mid, down_w)
}
