//! Honeycrisp's GatedDeltaNet recurrence-step Metal kernel vs. the
//! already-HF-verified CPU reference (`recurrence_step`) — on
//! synthetic data at the 27B model's real dims (48 heads,
//! head_k_dim=head_v_dim=128), run on THIS machine's real Metal
//! device. Doesn't need the 27B model itself (still can't load —
//! deferred memory ceiling): the recurrence math has no dependency on
//! actual model weights, only on shapes, so a synthetic
//! deterministic-seeded input is a legitimate correctness check of
//! the kernel itself (the WEIGHTS/projections/gates around it are
//! separately verified in gated_delta_golden.rs against real HF
//! output; this test is purely about the state-update kernel).
//!
//! Runs several consecutive steps (not just one) to also exercise the
//! read-modify-write persistence contract — state must carry
//! correctly from one dispatch to the next, the same way it must
//! carry across decode calls in the live loop.
//!
//! Spec: specs/ops.md §"GatedDeltaNet".

#![cfg(target_os = "macos")]

use run::backend::Backend;
use run::backend::cpu::gated_delta::recurrence_step;

/// Deterministic pseudo-random f32 in [-1, 1), no external RNG crate needed.
fn lcg(seed: &mut u64) -> f32 {
    *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    let bits = (*seed >> 40) as u32;
    (bits as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
}

#[test]
fn honeycrisp_gated_delta_recurrence_matches_cpu_reference() {
    let backend = match run::backend::honeycrisp::HoneycrispBackend::new() {
        Ok(b) => b,
        Err(e) => {
            eprintln!("skip: honeycrisp unavailable on this machine: {e}");
            return;
        }
    };

    const NUM_HEADS: usize = 48;
    const HK: usize = 128;
    const HV: usize = 128;
    const STEPS: usize = 4;

    let mut seed = 0xC0FFEE_u64;
    let mut cpu_state = vec![0f32; NUM_HEADS * HK * HV];
    let mut gpu_state = vec![0f32; NUM_HEADS * HK * HV];

    for step in 0..STEPS {
        let q: Vec<f32> = (0..NUM_HEADS * HK).map(|_| lcg(&mut seed) * 0.1).collect();
        let k: Vec<f32> = (0..NUM_HEADS * HK).map(|_| lcg(&mut seed) * 0.1).collect();
        let v: Vec<f32> = (0..NUM_HEADS * HV).map(|_| lcg(&mut seed) * 0.1).collect();
        // decay in (0,1] (real values are exp(negative) = decay in (0,1]),
        // beta in [0,1] (real values are sigmoid outputs).
        let decay: Vec<f32> = (0..NUM_HEADS).map(|_| 0.5 + 0.5 * (lcg(&mut seed) * 0.5 + 0.5)).collect();
        let beta: Vec<f32> = (0..NUM_HEADS).map(|_| lcg(&mut seed) * 0.5 + 0.5).collect();

        // CPU reference: one recurrence_step call per head.
        let mut cpu_out = vec![0f32; NUM_HEADS * HV];
        for hi in 0..NUM_HEADS {
            let st = &mut cpu_state[hi * HK * HV..(hi + 1) * HK * HV];
            let out_row = &mut cpu_out[hi * HV..(hi + 1) * HV];
            recurrence_step(
                st,
                &q[hi * HK..(hi + 1) * HK],
                &k[hi * HK..(hi + 1) * HK],
                &v[hi * HV..(hi + 1) * HV],
                decay[hi],
                beta[hi],
                HK,
                HV,
                out_row,
            );
        }

        // GPU: one batched dispatch for all heads.
        let gpu_out = backend
            .gated_delta_recurrence_step(&mut gpu_state, &q, &k, &v, &decay, &beta, NUM_HEADS, HK, HV)
            .expect("honeycrisp gated_delta_recurrence_step");

        let mut worst_out = 0f32;
        let mut max_ref = 0f32;
        for (&g, &c) in gpu_out.iter().zip(cpu_out.iter()) {
            worst_out = worst_out.max((g - c).abs());
            max_ref = max_ref.max(c.abs());
        }
        let mut worst_state = 0f32;
        for (&g, &c) in gpu_state.iter().zip(cpu_state.iter()) {
            worst_state = worst_state.max((g - c).abs());
        }
        eprintln!(
            "step {step}: worst out diff {worst_out} (max|cpu|={max_ref}), worst state diff {worst_state}"
        );
        let tol = (max_ref * 1e-4).max(1e-5);
        assert!(worst_out < tol, "step {step}: output diverges from CPU reference: {worst_out} > {tol}");
        assert!(worst_state < 1e-4, "step {step}: state diverges from CPU reference: {worst_state}");
    }
}
