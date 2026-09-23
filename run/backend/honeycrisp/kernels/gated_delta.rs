//! Metal kernel for GatedDeltaNet's per-head delta-rule recurrence step
//! — the persistent-state update at the heart of the "linear_attention"
//! layer (Qwen3.5/3.8). Same math as
//! `backend::cpu::gated_delta::recurrence_step`, one head per
//! threadgroup, one thread per `head_v_dim` column.
//!
//! Each thread `j` owns column `j` of the `[head_k_dim, head_v_dim]`
//! state matrix entirely (state is row-major, column stride = HV) —
//! no cross-thread synchronization needed at all within a head: decay,
//! the rank-1 update, and the output read are all purely
//! column-local. `HK` iterations per thread, 3 passes (decay+kv_mem,
//! rank-1 update, output) — cheap (e.g. 3*128=384 FMA-ish ops for the
//! 27B model's dims) and fully parallel across `num_v_heads *
//! head_v_dim` threads (6144 for the 27B model).
//!
//! `state` is READ-MODIFY-WRITE in place — the first genuinely
//! persistent-mutable-state GPU kernel in this codebase (KV-cache
//! append is the closest precedent, but it's a pure write). Caller
//! owns the buffer's lifecycle (alloc once per layer, zero on a fresh
//! conversation) — this kernel only ever reads and updates it.

use crate::backend::BackendError;
use crate::backend::honeycrisp::device::HoneycrispDevice;

const MSL_TEMPLATE: &str = r#"
#include <metal_stdlib>
using namespace metal;

constant constexpr uint HK = __HK__u;
constant constexpr uint HV = __HV__u;

struct Dims { uint num_heads; uint pad0; uint pad1; uint pad2; };

kernel void kmain(
    device       float  *state  [[buffer(0)]],  // [num_heads, HK, HV], read-write
    device const float  *q_t    [[buffer(1)]],  // [num_heads, HK]
    device const float  *k_t    [[buffer(2)]],  // [num_heads, HK]
    device const float  *v_t    [[buffer(3)]],  // [num_heads, HV]
    device const float  *decay  [[buffer(4)]],  // [num_heads]
    device const float  *beta   [[buffer(5)]],  // [num_heads]
    device       float  *out    [[buffer(6)]],  // [num_heads, HV]
    constant     Dims   &dims   [[buffer(7)]],
    uint2                gid    [[thread_position_in_grid]]
) {
    uint hi = gid.y;
    uint j  = gid.x;
    if (hi >= dims.num_heads || j >= HV) return;

    device float *st = state + hi * HK * HV;
    device const float *q = q_t + hi * HK;
    device const float *k = k_t + hi * HK;
    float v_j = v_t[hi * HV + j];
    float decay_t = decay[hi];
    float beta_t = beta[hi];

    // Pass 1: decay column j in place, accumulate kv_mem from the
    // DECAYED values (decay must land before this read — matches
    // `recurrence_step`'s `for s in st.iter_mut() { *s *= decay_t }`
    // running before the kv_mem loop).
    float kv_mem = 0.0f;
    for (uint i = 0; i < HK; i++) {
        float decayed = st[i * HV + j] * decay_t;
        st[i * HV + j] = decayed;
        kv_mem += decayed * k[i];
    }
    float delta_j = (v_j - kv_mem) * beta_t;
    // Pass 2: rank-1 update.
    for (uint i = 0; i < HK; i++) {
        st[i * HV + j] += k[i] * delta_j;
    }
    // Pass 3: output from the just-updated state.
    float out_j = 0.0f;
    for (uint i = 0; i < HK; i++) {
        out_j += st[i * HV + j] * q[i];
    }
    out[hi * HV + j] = out_j;
}
"#;

pub fn msl_for(head_k_dim: usize, head_v_dim: usize) -> String {
    MSL_TEMPLATE
        .replace("__HK__", &head_k_dim.to_string())
        .replace("__HV__", &head_v_dim.to_string())
}

/// Runs the recurrence step for every head in one dispatch. `state` is
/// mutated in place (read-modify-write); returns the new `out`
/// `[num_heads, head_v_dim]` buffer.
pub fn dispatch(
    dev: &HoneycrispDevice,
    pipeline: &aruminium::Pipeline,
    state: &aruminium::Buffer,
    q_t: &aruminium::Buffer,
    k_t: &aruminium::Buffer,
    v_t: &aruminium::Buffer,
    decay: &aruminium::Buffer,
    beta: &aruminium::Buffer,
    num_heads: u32,
    head_v_dim: u32,
) -> Result<aruminium::Buffer, BackendError> {
    let out = dev.alloc((num_heads as usize * head_v_dim as usize * 4).max(4))?;

    #[repr(C)]
    #[derive(Clone, Copy)]
    struct Dims { num_heads: u32, pad0: u32, pad1: u32, pad2: u32 }
    let dims = Dims { num_heads, pad0: 0, pad1: 0, pad2: 0 };

    unsafe {
        aruminium::autorelease_pool(|| {
            dev.dispatch.batch_raw(|enc| {
                enc.bind(pipeline);
                enc.bind_buffer(state, 0, 0);
                enc.bind_buffer(q_t, 0, 1);
                enc.bind_buffer(k_t, 0, 2);
                enc.bind_buffer(v_t, 0, 3);
                enc.bind_buffer(decay, 0, 4);
                enc.bind_buffer(beta, 0, 5);
                enc.bind_buffer(&out, 0, 6);
                let bytes = std::slice::from_raw_parts(
                    &dims as *const Dims as *const u8,
                    std::mem::size_of::<Dims>(),
                );
                enc.push(bytes, 7);
                enc.launch_groups((1, num_heads as usize, 1), (head_v_dim as usize, 1, 1));
            });
        });
    }
    Ok(out)
}
