//! Metal Q4_K fused dequant+matmul.
//!
//! Q4_K block layout (144 bytes):
//!   f16 d, f16 dmin, scales[12] (6-bit packed), qs[128] (4-bit nibbles)
//! Dequant: x = d * s[j] * nibble - dmin * m[j]
//! Output: y[b, row] = sum_k x[b, k] * W[row, k]
//!
//! SIMD-parallel kernels: 1 SIMD (32 lanes) per output row.
//! Lane `l` handles nibble-position `l` within each 32-wide sub-group.
//! Full lane utilization: all 32 lanes active for every superblock.
//! simd_sum() reduces partial sums within the SIMD.

use crate::backend::BackendError;
use crate::backend::honeycrisp::device::HoneycrispDevice;

pub const SIMDS_PER_GROUP: u32 = 16;
pub const LANES: u32 = 32;

/// Simple 1-thread-per-row kernel (used by the generic execute path).
pub const MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct Dims { uint batch; uint n_rows; uint n_blocks; uint pad; };

kernel void kmain(
    device const float   *x    [[buffer(0)]],
    device const uint8_t *w    [[buffer(1)]],
    device       float   *y    [[buffer(2)]],
    constant     Dims    &dims [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= dims.batch * dims.n_rows) return;
    uint b   = gid / dims.n_rows;
    uint row = gid % dims.n_rows;

    device const uint8_t *w_row = w + row * dims.n_blocks * 144u;
    device const float   *x_b   = x + b * dims.n_blocks * 256u;

    float sum = 0.0f;
    for (uint blk = 0; blk < dims.n_blocks; blk++) {
        device const uint8_t *block = w_row + blk * 144u;
        float d    = float(as_type<half>((ushort)(uint(block[0]) | (uint(block[1]) << 8))));
        float dmin = float(as_type<half>((ushort)(uint(block[2]) | (uint(block[3]) << 8))));
        device const uint8_t *sc = block + 4u;
        device const uint8_t *qs = block + 16u;
        device const float   *xb = x_b + blk * 256u;

        for (uint j = 0; j < 8u; j++) {
            uint s, m;
            if (j < 4u) {
                s = uint(sc[j])     & 0x3Fu;
                m = uint(sc[j + 4]) & 0x3Fu;
            } else {
                s = (uint(sc[j + 4]) & 0x0Fu) | ((uint(sc[j - 4]) >> 6u) << 4u);
                m = (uint(sc[j + 4]) >> 4u)   | ((uint(sc[j])     >> 6u) << 4u);
            }
            float ds = d * float(s);
            float dm = dmin * float(m);

            uint qs_off = (j / 2u) * 32u;
            uint shift  = (j & 1u) * 4u;
            device const float *xj = xb + j * 32u;

            float nibble_dot = 0.0f, x_sum = 0.0f;
            for (uint l = 0; l < 32u; l++) {
                float nib = float((uint(qs[qs_off + l]) >> shift) & 0x0Fu);
                nibble_dot += nib * xj[l];
                x_sum      += xj[l];
            }
            sum += ds * nibble_dot - dm * x_sum;
        }
    }
    y[gid] = sum;
}
"#;

// Shared helper injected into every SIMD-parallel kernel.
// Extracts 6-bit (scale, min) for sub-group j from the 12-byte scales field.
const SCALE_MIN_HELPER: &str = r#"
inline uint2 q4k_sm(device const uint8_t* sc, uint j) {
    uint s, m;
    if (j < 4u) {
        s = uint(sc[j])     & 0x3Fu;
        m = uint(sc[j+4u])  & 0x3Fu;
    } else {
        s = (uint(sc[j+4u]) & 0x0Fu) | ((uint(sc[j-4u]) >> 6u) << 4u);
        m = (uint(sc[j+4u]) >> 4u)   | ((uint(sc[j])    >> 6u) << 4u);
    }
    return uint2(s, m);
}
"#;

/// SIMD-parallel NRM+Q4K matmul.
/// Buffers: 0=x, 1=norm_w, 2=w, 3=y, 4=Dims
/// One SIMD (32 lanes) per output row. All 32 lanes active per superblock.
pub const MSL_NRM: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct Dims { uint batch; uint n_rows; uint n_blocks; float eps; };

constant constexpr uint SPG  = 16u;  // SIMDs per threadgroup
constant constexpr uint LSIM = 32u;  // lanes per SIMD

inline uint2 q4k_sm(device const uint8_t* sc, uint j) {
    uint s, m;
    if (j < 4u) {
        s = uint(sc[j])     & 0x3Fu;
        m = uint(sc[j+4u])  & 0x3Fu;
    } else {
        s = (uint(sc[j+4u]) & 0x0Fu) | ((uint(sc[j-4u]) >> 6u) << 4u);
        m = (uint(sc[j+4u]) >> 4u)   | ((uint(sc[j])    >> 6u) << 4u);
    }
    return uint2(s, m);
}

kernel void kmain(
    device const float   *x      [[buffer(0)]],
    device const float   *norm_w [[buffer(1)]],
    device const uint8_t *w      [[buffer(2)]],
    device       float   *y      [[buffer(3)]],
    constant     Dims    &dims   [[buffer(4)]],
    uint tgpig_x [[threadgroup_position_in_grid]],
    uint lane    [[thread_index_in_simdgroup]],
    uint sgitg   [[simdgroup_index_in_threadgroup]]
) {
    uint row_global = tgpig_x * SPG + sgitg;
    uint b   = row_global / dims.n_rows;
    uint row = row_global - b * dims.n_rows;
    if (b >= dims.batch || row >= dims.n_rows) return;

    uint k_floats = dims.n_blocks * 256u;
    device const float *x_b = x + b * k_floats;

    // Pass 1: RMSNorm — stride-32 over k_floats, reduce within SIMD
    float sq = 0.0f;
    for (uint i = lane; i < k_floats; i += LSIM) {
        float v = x_b[i]; sq += v * v;
    }
    float inv_rms = rsqrt(simd_sum(sq) / float(k_floats) + dims.eps);

    // Pass 2: Q4K matmul with inline norm
    device const uint8_t *w_row = w + row * dims.n_blocks * 144u;
    float acc = 0.0f;
    for (uint sb = 0u; sb < dims.n_blocks; sb++) {
        device const uint8_t *blk = w_row + sb * 144u;
        float d    = float(as_type<half>((ushort)(uint(blk[0]) | (uint(blk[1]) << 8u))));
        float dmin = float(as_type<half>((ushort)(uint(blk[2]) | (uint(blk[3]) << 8u))));
        device const uint8_t *sc = blk + 4u;
        device const uint8_t *qs = blk + 16u;
        device const float *x_sb = x_b    + sb * 256u;
        device const float *n_sb = norm_w + sb * 256u;
        for (uint grp = 0u; grp < 4u; grp++) {
            uint2 sm0 = q4k_sm(sc, grp*2u);
            uint2 sm1 = q4k_sm(sc, grp*2u+1u);
            float ds0 = d*float(sm0.x), dm0 = dmin*float(sm0.y);
            float ds1 = d*float(sm1.x), dm1 = dmin*float(sm1.y);
            uint qb   = uint(qs[grp*32u + lane]);
            float n0  = float(qb & 0xFu), n1 = float(qb >> 4u);
            uint  xb  = grp * 64u + lane;
            float x0  = x_sb[xb]      * inv_rms * n_sb[xb];
            float x1  = x_sb[xb+32u]  * inv_rms * n_sb[xb+32u];
            acc += (ds0*n0 - dm0)*x0 + (ds1*n1 - dm1)*x1;
        }
    }
    float r = simd_sum(acc);
    if (lane == 0u) y[b * dims.n_rows + row] = r;
}
"#;

/// SIMD-parallel dual NRM+Q4K matmul (K and V in one dispatch).
/// One SIMD handles row `r` of BOTH wa (k_proj) and wb (v_proj).
/// Buffers: 0=x, 1=norm_w, 2=wa, 3=wb, 4=ya, 5=yb, 6=Dims
/// dims.n_rows = kv_dim (same for K and V).
pub const MSL_DUAL_NRM: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct Dims { uint batch; uint n_rows; uint n_blocks; float eps; };

constant constexpr uint SPG  = 16u;
constant constexpr uint LSIM = 32u;

inline uint2 q4k_sm(device const uint8_t* sc, uint j) {
    uint s, m;
    if (j < 4u) {
        s = uint(sc[j])     & 0x3Fu;
        m = uint(sc[j+4u])  & 0x3Fu;
    } else {
        s = (uint(sc[j+4u]) & 0x0Fu) | ((uint(sc[j-4u]) >> 6u) << 4u);
        m = (uint(sc[j+4u]) >> 4u)   | ((uint(sc[j])    >> 6u) << 4u);
    }
    return uint2(s, m);
}

kernel void kmain(
    device const float   *x      [[buffer(0)]],
    device const float   *norm_w [[buffer(1)]],
    device const uint8_t *wa     [[buffer(2)]],
    device const uint8_t *wb     [[buffer(3)]],
    device       float   *ya     [[buffer(4)]],
    device       float   *yb     [[buffer(5)]],
    constant     Dims    &dims   [[buffer(6)]],
    uint tgpig_x [[threadgroup_position_in_grid]],
    uint lane    [[thread_index_in_simdgroup]],
    uint sgitg   [[simdgroup_index_in_threadgroup]]
) {
    uint row_global = tgpig_x * SPG + sgitg;
    uint b   = row_global / dims.n_rows;
    uint row = row_global - b * dims.n_rows;
    if (b >= dims.batch || row >= dims.n_rows) return;

    uint k_floats = dims.n_blocks * 256u;
    device const float *x_b = x + b * k_floats;

    // RMSNorm
    float sq = 0.0f;
    for (uint i = lane; i < k_floats; i += LSIM) {
        float v = x_b[i]; sq += v * v;
    }
    float inv_rms = rsqrt(simd_sum(sq) / float(k_floats) + dims.eps);

    device const uint8_t *wa_row = wa + row * dims.n_blocks * 144u;
    device const uint8_t *wb_row = wb + row * dims.n_blocks * 144u;
    float acc_a = 0.0f, acc_b = 0.0f;
    for (uint sb = 0u; sb < dims.n_blocks; sb++) {
        device const uint8_t *ba = wa_row + sb * 144u;
        device const uint8_t *bb = wb_row + sb * 144u;
        float da    = float(as_type<half>((ushort)(uint(ba[0]) | (uint(ba[1]) << 8u))));
        float dmina = float(as_type<half>((ushort)(uint(ba[2]) | (uint(ba[3]) << 8u))));
        float db    = float(as_type<half>((ushort)(uint(bb[0]) | (uint(bb[1]) << 8u))));
        float dminb = float(as_type<half>((ushort)(uint(bb[2]) | (uint(bb[3]) << 8u))));
        device const uint8_t *sca = ba + 4u, *qsa = ba + 16u;
        device const uint8_t *scb = bb + 4u, *qsb = bb + 16u;
        device const float *xs = x_b    + sb * 256u;
        device const float *ns = norm_w + sb * 256u;
        for (uint grp = 0u; grp < 4u; grp++) {
            uint2 sma0 = q4k_sm(sca, grp*2u), sma1 = q4k_sm(sca, grp*2u+1u);
            uint2 smb0 = q4k_sm(scb, grp*2u), smb1 = q4k_sm(scb, grp*2u+1u);
            uint qba = uint(qsa[grp*32u + lane]);
            uint qbb = uint(qsb[grp*32u + lane]);
            float na0 = float(qba & 0xFu), na1 = float(qba >> 4u);
            float nb0 = float(qbb & 0xFu), nb1 = float(qbb >> 4u);
            uint  xb  = grp * 64u + lane;
            float x0  = xs[xb]     * inv_rms * ns[xb];
            float x1  = xs[xb+32u] * inv_rms * ns[xb+32u];
            acc_a += (da*float(sma0.x)*na0 - dmina*float(sma0.y))*x0
                   + (da*float(sma1.x)*na1 - dmina*float(sma1.y))*x1;
            acc_b += (db*float(smb0.x)*nb0 - dminb*float(smb0.y))*x0
                   + (db*float(smb1.x)*nb1 - dminb*float(smb1.y))*x1;
        }
    }
    float ta = simd_sum(acc_a), tb = simd_sum(acc_b);
    if (lane == 0u) {
        uint oi = b * dims.n_rows + row;
        ya[oi] = ta; yb[oi] = tb;
    }
}
"#;

/// SIMD-parallel NRM + Q4K gate+up matmul + SwiGLU fusion.
/// Buffers: 0=x, 1=norm_w, 2=gate_w, 3=up_w, 4=y, 5=Dims
pub const MSL_GUS_NRM: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct Dims { uint batch; uint n_rows; uint n_blocks; float eps; };

constant constexpr uint SPG  = 16u;
constant constexpr uint LSIM = 32u;

inline uint2 q4k_sm(device const uint8_t* sc, uint j) {
    uint s, m;
    if (j < 4u) {
        s = uint(sc[j])     & 0x3Fu;
        m = uint(sc[j+4u])  & 0x3Fu;
    } else {
        s = (uint(sc[j+4u]) & 0x0Fu) | ((uint(sc[j-4u]) >> 6u) << 4u);
        m = (uint(sc[j+4u]) >> 4u)   | ((uint(sc[j])    >> 6u) << 4u);
    }
    return uint2(s, m);
}

kernel void kmain(
    device const float   *x      [[buffer(0)]],
    device const float   *norm_w [[buffer(1)]],
    device const uint8_t *wa     [[buffer(2)]],  // gate
    device const uint8_t *wb     [[buffer(3)]],  // up
    device       float   *y      [[buffer(4)]],
    constant     Dims    &dims   [[buffer(5)]],
    uint tgpig_x [[threadgroup_position_in_grid]],
    uint lane    [[thread_index_in_simdgroup]],
    uint sgitg   [[simdgroup_index_in_threadgroup]]
) {
    uint row_global = tgpig_x * SPG + sgitg;
    uint b   = row_global / dims.n_rows;
    uint row = row_global - b * dims.n_rows;
    if (b >= dims.batch || row >= dims.n_rows) return;

    uint k_floats = dims.n_blocks * 256u;
    device const float *x_b = x + b * k_floats;

    // RMSNorm
    float sq = 0.0f;
    for (uint i = lane; i < k_floats; i += LSIM) {
        float v = x_b[i]; sq += v * v;
    }
    float inv_rms = rsqrt(simd_sum(sq) / float(k_floats) + dims.eps);

    device const uint8_t *wa_row = wa + row * dims.n_blocks * 144u;
    device const uint8_t *wb_row = wb + row * dims.n_blocks * 144u;
    float acc_g = 0.0f, acc_u = 0.0f;
    for (uint sb = 0u; sb < dims.n_blocks; sb++) {
        device const uint8_t *bg = wa_row + sb * 144u;
        device const uint8_t *bu = wb_row + sb * 144u;
        float dg    = float(as_type<half>((ushort)(uint(bg[0]) | (uint(bg[1]) << 8u))));
        float dming = float(as_type<half>((ushort)(uint(bg[2]) | (uint(bg[3]) << 8u))));
        float du    = float(as_type<half>((ushort)(uint(bu[0]) | (uint(bu[1]) << 8u))));
        float dminu = float(as_type<half>((ushort)(uint(bu[2]) | (uint(bu[3]) << 8u))));
        device const uint8_t *scg = bg + 4u, *qsg = bg + 16u;
        device const uint8_t *scu = bu + 4u, *qsu = bu + 16u;
        device const float *xs = x_b    + sb * 256u;
        device const float *ns = norm_w + sb * 256u;
        for (uint grp = 0u; grp < 4u; grp++) {
            uint2 sg0 = q4k_sm(scg, grp*2u),   sg1 = q4k_sm(scg, grp*2u+1u);
            uint2 su0 = q4k_sm(scu, grp*2u),   su1 = q4k_sm(scu, grp*2u+1u);
            uint qbg  = uint(qsg[grp*32u + lane]);
            uint qbu  = uint(qsu[grp*32u + lane]);
            float ng0 = float(qbg & 0xFu), ng1 = float(qbg >> 4u);
            float nu0 = float(qbu & 0xFu), nu1 = float(qbu >> 4u);
            uint  xb  = grp * 64u + lane;
            float x0  = xs[xb]     * inv_rms * ns[xb];
            float x1  = xs[xb+32u] * inv_rms * ns[xb+32u];
            acc_g += (dg*float(sg0.x)*ng0 - dming*float(sg0.y))*x0
                   + (dg*float(sg1.x)*ng1 - dming*float(sg1.y))*x1;
            acc_u += (du*float(su0.x)*nu0 - dminu*float(su0.y))*x0
                   + (du*float(su1.x)*nu1 - dminu*float(su1.y))*x1;
        }
    }
    float gate = simd_sum(acc_g), up = simd_sum(acc_u);
    if (lane == 0u) {
        float s = gate / (1.0f + exp(-gate));  // silu
        y[b * dims.n_rows + row] = s * up;
    }
}
"#;

/// SIMD-parallel Q4K matmul (no NRM). For o_proj and down_proj.
/// Buffers: 0=x, 1=w, 2=y, 3=Dims
pub const MSL_LARGE: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct Dims { uint batch; uint n_rows; uint n_blocks; uint pad; };

constant constexpr uint SPG  = 16u;
constant constexpr uint LSIM = 32u;

inline uint2 q4k_sm(device const uint8_t* sc, uint j) {
    uint s, m;
    if (j < 4u) {
        s = uint(sc[j])     & 0x3Fu;
        m = uint(sc[j+4u])  & 0x3Fu;
    } else {
        s = (uint(sc[j+4u]) & 0x0Fu) | ((uint(sc[j-4u]) >> 6u) << 4u);
        m = (uint(sc[j+4u]) >> 4u)   | ((uint(sc[j])    >> 6u) << 4u);
    }
    return uint2(s, m);
}

kernel void kmain(
    device const float   *x    [[buffer(0)]],
    device const uint8_t *w    [[buffer(1)]],
    device       float   *y    [[buffer(2)]],
    constant     Dims    &dims [[buffer(3)]],
    uint tgpig_x [[threadgroup_position_in_grid]],
    uint lane    [[thread_index_in_simdgroup]],
    uint sgitg   [[simdgroup_index_in_threadgroup]]
) {
    uint row_global = tgpig_x * SPG + sgitg;
    uint b   = row_global / dims.n_rows;
    uint row = row_global - b * dims.n_rows;
    if (b >= dims.batch || row >= dims.n_rows) return;

    uint k_floats = dims.n_blocks * 256u;
    device const float *x_b = x + b * k_floats;
    device const uint8_t *w_row = w + row * dims.n_blocks * 144u;

    float acc = 0.0f;
    for (uint sb = 0u; sb < dims.n_blocks; sb++) {
        device const uint8_t *blk = w_row + sb * 144u;
        float d    = float(as_type<half>((ushort)(uint(blk[0]) | (uint(blk[1]) << 8u))));
        float dmin = float(as_type<half>((ushort)(uint(blk[2]) | (uint(blk[3]) << 8u))));
        device const uint8_t *sc = blk + 4u;
        device const uint8_t *qs = blk + 16u;
        device const float *xs = x_b + sb * 256u;
        for (uint grp = 0u; grp < 4u; grp++) {
            uint2 sm0 = q4k_sm(sc, grp*2u);
            uint2 sm1 = q4k_sm(sc, grp*2u+1u);
            float ds0 = d*float(sm0.x), dm0 = dmin*float(sm0.y);
            float ds1 = d*float(sm1.x), dm1 = dmin*float(sm1.y);
            uint qb  = uint(qs[grp*32u + lane]);
            float n0 = float(qb & 0xFu), n1 = float(qb >> 4u);
            uint xb  = grp * 64u + lane;
            float x0 = xs[xb], x1 = xs[xb+32u];
            acc += (ds0*n0 - dm0)*x0 + (ds1*n1 - dm1)*x1;
        }
    }
    float r = simd_sum(acc);
    if (lane == 0u) y[b * dims.n_rows + row] = r;
}
"#;

pub fn dispatch(
    dev: &HoneycrispDevice,
    pipeline: &aruminium::Pipeline,
    x: &aruminium::Buffer,
    w: &aruminium::Buffer,
    batch: u32,
    n_rows: u32,
    n_blocks: u32,
) -> Result<aruminium::Buffer, BackendError> {
    let total = batch * n_rows;
    let out = dev.alloc((total * 4) as usize)?;

    #[repr(C)]
    #[derive(Clone, Copy)]
    struct Dims { batch: u32, n_rows: u32, n_blocks: u32, pad: u32 }
    let dims = Dims { batch, n_rows, n_blocks, pad: 0 };

    unsafe {
        aruminium::autorelease_pool(|| {
            dev.dispatch.batch_raw(|enc| {
                enc.bind(pipeline);
                enc.bind_buffer(x, 0, 0);
                enc.bind_buffer(w, 0, 1);
                enc.bind_buffer(&out, 0, 2);
                let bytes = std::slice::from_raw_parts(
                    &dims as *const Dims as *const u8,
                    std::mem::size_of::<Dims>(),
                );
                enc.push(bytes, 3);
                let n = total as usize;
                enc.launch_groups(((n + 63) / 64, 1, 1), (64, 1, 1));
            });
        });
    }
    Ok(out)
}

/// Dispatch for `MSL_LARGE`: SIMD-parallel, `SIMDS_PER_GROUP` (16) output
/// rows per threadgroup, `LANES` (32) lanes cooperating on each row via
/// `simd_sum`. `MSL_LARGE` was written but never given a matching launch —
/// the plain `dispatch()` above launches `(64,1,1)`-thread groups, which is
/// only correct for the scalar 1-thread-per-row `MSL` kernel; calling
/// `MSL_LARGE` through it starves 14 of every 16 simdgroups the kernel
/// assumes exist per threadgroup (`row_global = tgpig_x*16 + sgitg` reads
/// stale/wrong `row` for any `sgitg >= 2`, since only `64/32 = 2` simdgroups
/// actually launch). Mirrors `q8_matmul::dispatch`'s launch geometry, which
/// gets this right for the same threadgroup shape.
pub fn dispatch_large(
    dev: &HoneycrispDevice,
    pipeline: &aruminium::Pipeline,
    x: &aruminium::Buffer,
    w: &aruminium::Buffer,
    batch: u32,
    n_rows: u32,
    n_blocks: u32,
) -> Result<aruminium::Buffer, BackendError> {
    let total_rows = batch * n_rows;
    let out = dev.alloc((total_rows * 4) as usize)?;

    #[repr(C)]
    #[derive(Clone, Copy)]
    struct Dims { batch: u32, n_rows: u32, n_blocks: u32, pad: u32 }
    let dims = Dims { batch, n_rows, n_blocks, pad: 0 };

    let groups_x = (total_rows + SIMDS_PER_GROUP - 1) / SIMDS_PER_GROUP;
    let threads_per_group = SIMDS_PER_GROUP * LANES;

    unsafe {
        aruminium::autorelease_pool(|| {
            dev.dispatch.batch_raw(|enc| {
                enc.bind(pipeline);
                enc.bind_buffer(x, 0, 0);
                enc.bind_buffer(w, 0, 1);
                enc.bind_buffer(&out, 0, 2);
                let bytes = std::slice::from_raw_parts(
                    &dims as *const Dims as *const u8,
                    std::mem::size_of::<Dims>(),
                );
                enc.push(bytes, 3);
                enc.launch_groups((groups_x as usize, 1, 1), (threads_per_group as usize, 1, 1));
            });
        });
    }
    Ok(out)
}

/// Compute threadgroup count for SIMD-parallel kernels (SPG=16, LANES=32).
pub fn tg_count(n_rows: u32, batch: u32) -> usize {
    let total = (n_rows * batch) as usize;
    (total + SIMDS_PER_GROUP as usize - 1) / SIMDS_PER_GROUP as usize
}

pub fn tg_size() -> usize { (SIMDS_PER_GROUP * LANES) as usize }
