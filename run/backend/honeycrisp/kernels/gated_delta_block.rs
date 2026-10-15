//! The three small elementwise/reduction kernels that, together with the
//! Q8 NRM/RES matmul kernels and `gated_delta.rs`'s recurrence kernel, let
//! a whole GatedDeltaNet attention block for one decode token run inside a
//! single command buffer (see `HoneycrispBackend::gated_delta_block_fused`).
//! Each is a direct transliteration of the corresponding stage in
//! `backend::cpu::gated_delta::gated_delta_forward` (T=1 only):
//!
//!   conv_silu   — step 2: causal depthwise conv1d over [conv_state ++ x],
//!                 SiLU, then shift-and-append `x` into conv_state in place.
//!   prep        — steps 3-5: split q/k out of the conv output, expand
//!                 num_k_heads → num_v_heads (repeat_interleave), L2-normalize
//!                 per head, scale q by 1/sqrt(head_k_dim); plus the gates
//!                 beta = sigmoid(b), decay = exp(-exp(A_log) * softplus(a + dt_bias)).
//!   gated_norm  — step 7: RmsNormGated — rms over head_v_dim, × gain, × SiLU(z).
//!
//! Reductions use simd_sum + a threadgroup partial per simdgroup; the
//! reduced dimension (head_k_dim / head_v_dim) must be a multiple of 32 and
//! at most 1024 (one threadgroup) — the caller checks and falls back to the
//! per-op path otherwise.

/// `x`: this token's projected mixed_qkv `[channels]`; `w`: `[channels, KSZ]`;
/// `hist`: `[channels, KSZ-1]` oldest→newest, updated in place; `y`: `[channels]`.
const CONV_SILU_TEMPLATE: &str = r#"
#include <metal_stdlib>
using namespace metal;

constant constexpr uint KSZ = __KSZ__u;

struct Dims { uint channels; uint pad0; uint pad1; uint pad2; };

kernel void kmain(
    device const float *x    [[buffer(0)]],
    device const float *w    [[buffer(1)]],
    device       float *hist [[buffer(2)]],
    device       float *y    [[buffer(3)]],
    constant     Dims  &dims [[buffer(4)]],
    uint c [[thread_position_in_grid]]
) {
    if (c >= dims.channels) return;
    device       float *h  = hist + c * (KSZ - 1u);
    device const float *wc = w + c * KSZ;
    float xc = x[c];
    // Extended timeline [hist(KSZ-1) ++ x]: tap k < KSZ-1 reads hist[k],
    // the newest tap reads x.
    float acc = wc[KSZ - 1u] * xc;
    for (uint k = 0; k + 1u < KSZ; k++) acc += wc[k] * h[k];
    y[c] = acc / (1.0f + exp(-acc));
    // New history = trailing KSZ-1 of the extended timeline.
    for (uint k = 0; k + 2u < KSZ; k++) h[k] = h[k + 1u];
    h[KSZ - 2u] = xc;
}
"#;

/// One threadgroup per v-head (HK threads). `y` is the conv output laid out
/// `[q(kd) | k(kd) | v(vd)]`; q/k for v-head `hi` come from k-head `hi / RATIO`.
const PREP_TEMPLATE: &str = r#"
#include <metal_stdlib>
using namespace metal;

constant constexpr uint HK    = __HK__u;
constant constexpr uint NSG   = __NSG__u;   // HK / 32 simdgroups per threadgroup
constant constexpr uint RATIO = __RATIO__u; // num_v_heads / num_k_heads

struct Dims { uint kd; uint num_heads; float q_scale; float l2_eps; };

kernel void kmain(
    device const float *y       [[buffer(0)]],
    device const float *b       [[buffer(1)]],
    device const float *a       [[buffer(2)]],
    device const float *a_log   [[buffer(3)]],
    device const float *dt_bias [[buffer(4)]],
    device       float *q_h     [[buffer(5)]],
    device       float *k_h     [[buffer(6)]],
    device       float *decay   [[buffer(7)]],
    device       float *beta    [[buffer(8)]],
    constant     Dims  &dims    [[buffer(9)]],
    uint hi   [[threadgroup_position_in_grid]],
    uint i    [[thread_position_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg   [[simdgroup_index_in_threadgroup]]
) {
    threadgroup float pq[NSG];
    threadgroup float pk[NSG];
    uint src = hi / RATIO;
    float qv = y[src * HK + i];
    float kv = y[dims.kd + src * HK + i];
    float sq = simd_sum(qv * qv);
    float sk = simd_sum(kv * kv);
    if (lane == 0) { pq[sg] = sq; pk[sg] = sk; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tq = 0.0f, tk = 0.0f;
    for (uint s = 0; s < NSG; s++) { tq += pq[s]; tk += pk[s]; }
    float inv_q = rsqrt(tq + dims.l2_eps);
    float inv_k = rsqrt(tk + dims.l2_eps);
    q_h[hi * HK + i] = qv * inv_q * dims.q_scale;
    k_h[hi * HK + i] = kv * inv_k;
    if (i == 0) {
        float bb = b[hi];
        beta[hi] = 1.0f / (1.0f + exp(-bb));
        float aa = a[hi] + dt_bias[hi];
        float sp = (aa > 20.0f) ? aa : log(1.0f + exp(aa));
        decay[hi] = exp(-exp(a_log[hi]) * sp);
    }
}
"#;

/// One threadgroup per v-head (HV threads). `out`: recurrence output
/// `[num_heads, HV]`; `z`: `[num_heads * HV]`; `norm_w`: `[HV]`.
const GATED_NORM_TEMPLATE: &str = r#"
#include <metal_stdlib>
using namespace metal;

constant constexpr uint HV  = __HV__u;
constant constexpr uint NSG = __NSG__u;   // HV / 32

struct Dims { uint num_heads; float eps; uint pad0; uint pad1; };

kernel void kmain(
    device const float *out    [[buffer(0)]],
    device const float *z      [[buffer(1)]],
    device const float *norm_w [[buffer(2)]],
    device       float *gated  [[buffer(3)]],
    constant     Dims  &dims   [[buffer(4)]],
    uint hi   [[threadgroup_position_in_grid]],
    uint j    [[thread_position_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg   [[simdgroup_index_in_threadgroup]]
) {
    threadgroup float p[NSG];
    uint idx = hi * HV + j;
    float r = out[idx];
    float s = simd_sum(r * r);
    if (lane == 0) p[sg] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float t = 0.0f;
    for (uint k = 0; k < NSG; k++) t += p[k];
    float inv = rsqrt(t / float(HV) + dims.eps);
    float zz = z[idx];
    gated[idx] = r * inv * norm_w[j] * (zz / (1.0f + exp(-zz)));
}
"#;

pub fn msl_conv_silu(kernel_size: usize) -> String {
    CONV_SILU_TEMPLATE.replace("__KSZ__", &kernel_size.to_string())
}

pub fn msl_prep(head_k_dim: usize, ratio: usize) -> String {
    PREP_TEMPLATE
        .replace("__HK__", &head_k_dim.to_string())
        .replace("__NSG__", &(head_k_dim / 32).to_string())
        .replace("__RATIO__", &ratio.to_string())
}

pub fn msl_gated_norm(head_v_dim: usize) -> String {
    GATED_NORM_TEMPLATE
        .replace("__HV__", &head_v_dim.to_string())
        .replace("__NSG__", &(head_v_dim / 32).to_string())
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct ConvDims { pub channels: u32, pub pad0: u32, pub pad1: u32, pub pad2: u32 }

#[repr(C)]
#[derive(Clone, Copy)]
pub struct PrepDims { pub kd: u32, pub num_heads: u32, pub q_scale: f32, pub l2_eps: f32 }

#[repr(C)]
#[derive(Clone, Copy)]
pub struct GatedNormDims { pub num_heads: u32, pub eps: f32, pub pad0: u32, pub pad1: u32 }
