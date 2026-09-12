//! Metal kernel for Qwen3.5/3.8's interleaved-mRoPE Q/K rotation.
//!
//! Different index convention from `kernels::rope`'s NeoX kernels (built
//! for Gemma-4's "proportional" scheme, interleaved pairs across the
//! FULL head_dim): here `rope_dim` is a CONTIGUOUS PREFIX — `rotate_half`
//! applied within `x[..rope_dim]` (split at `rope_dim/2`, not
//! `head_dim/2`), `x[rope_dim..head_dim]` passed through unchanged as a
//! tail. Same math as `backend::cpu::mrope::apply_rope_cos_sin_f32` —
//! see that function's doc comment and ops.md §"mRoPE, interleaved" for
//! why the two conventions are not interchangeable.
//!
//! `cos`/`sin` are `[rope_dim]` — ONE token's precomputed frequencies
//! (`mrope_cos_sin`), duplicated across the two halves
//! (`cos[j] == cos[j+rope_dim/2]`). Decode is always one token per
//! call, so no batching over positions is needed here.

use crate::backend::BackendError;
use crate::backend::honeycrisp::device::HoneycrispDevice;

const MSL_TEMPLATE: &str = r#"
#include <metal_stdlib>
using namespace metal;

constant constexpr uint HEAD_DIM = __HEAD_DIM__u;
constant constexpr uint ROPE_DIM = __ROPE_DIM__u;
constant constexpr uint ROPE_HALF = __ROPE_HALF__u;

struct Dims { uint n_rows; uint pad0; uint pad1; uint pad2; };

kernel void kmain(
    device const float  *x      [[buffer(0)]],
    device const float  *cos_t  [[buffer(1)]],
    device const float  *sin_t  [[buffer(2)]],
    device       float  *y      [[buffer(3)]],
    constant     Dims   &dims   [[buffer(4)]],
    uint2                gid    [[thread_position_in_grid]]
) {
    uint row = gid.y;
    uint d   = gid.x;
    if (row >= dims.n_rows || d >= HEAD_DIM) return;
    uint base = row * HEAD_DIM;

    if (d >= ROPE_DIM) {
        y[base + d] = x[base + d];
        return;
    }
    if (d < ROPE_HALF) {
        float x1 = x[base + d];
        float x2 = x[base + d + ROPE_HALF];
        float c = cos_t[d];
        float s = sin_t[d];
        y[base + d] = x1 * c - x2 * s;
    } else {
        uint j = d - ROPE_HALF;
        float x1 = x[base + j];
        float x2 = x[base + j + ROPE_HALF];
        float c = cos_t[d];
        float s = sin_t[d];
        y[base + d] = x2 * c + x1 * s;
    }
}
"#;

pub fn msl_for(head_dim: usize, rope_dim: usize) -> String {
    MSL_TEMPLATE
        .replace("__HEAD_DIM__", &head_dim.to_string())
        .replace("__ROPE_DIM__", &rope_dim.to_string())
        .replace("__ROPE_HALF__", &(rope_dim / 2).to_string())
}

/// Rotates `x` (`[n_rows, head_dim]`) in place per `apply_rope_cos_sin_f32`'s
/// contiguous-prefix convention. `cos`/`sin` are `[rope_dim]` buffers.
pub fn dispatch(
    dev: &HoneycrispDevice,
    pipeline: &aruminium::Pipeline,
    x: &aruminium::Buffer,
    cos_t: &aruminium::Buffer,
    sin_t: &aruminium::Buffer,
    n_rows: u32,
    head_dim: u32,
) -> Result<aruminium::Buffer, BackendError> {
    let out = dev.alloc((n_rows as usize * head_dim as usize * 4).max(4))?;

    #[repr(C)]
    #[derive(Clone, Copy)]
    struct Dims { n_rows: u32, pad0: u32, pad1: u32, pad2: u32 }
    let dims = Dims { n_rows, pad0: 0, pad1: 0, pad2: 0 };

    unsafe {
        aruminium::autorelease_pool(|| {
            dev.dispatch.batch_raw(|enc| {
                enc.bind(pipeline);
                enc.bind_buffer(x, 0, 0);
                enc.bind_buffer(cos_t, 0, 1);
                enc.bind_buffer(sin_t, 0, 2);
                enc.bind_buffer(&out, 0, 3);
                let bytes = std::slice::from_raw_parts(
                    &dims as *const Dims as *const u8,
                    std::mem::size_of::<Dims>(),
                );
                enc.push(bytes, 4);
                enc.launch_groups((1, n_rows as usize, 1), (head_dim as usize, 1, 1));
            });
        });
    }
    Ok(out)
}
