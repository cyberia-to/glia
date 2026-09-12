//! Proves `conv_state` actually does its job: N sequential single-token
//! (`T=1`) calls, chaining `state` AND `conv_state` across calls, must
//! produce the exact same output as one `T=N` call — the property
//! `gated_delta_forward`'s own doc comment claims and the real decode
//! loop (`forward.rs`, always `T=1` per call) depends on.
//!
//! Reuses `gated_delta_golden.rs`'s dump (`/tmp/gdn_golden/`, T=6 real
//! input/output) — no separate Python extraction needed. Before
//! `conv_state` existed, this exact test would have FAILED (every
//! `T=1` call implicitly zero-padded its conv1d window instead of
//! seeing the previous calls' tokens) even though the T=6-in-one-call
//! golden test passed — that's precisely the bug `conv_state` fixes,
//! and precisely why "the T=6 test passes" was never sufficient
//! evidence that the live decode loop (which never calls with T=6,
//! always T=1) was producing correct output.
//!
//! Spec: specs/ops.md §5 "GatedDeltaNet".

use run::backend::cpu::gated_delta::{gated_delta_forward, GatedDeltaDims, GatedDeltaWeights};
use run::core::tensor::Tensor;
use std::path::{Path, PathBuf};

const DIR: &str = "/tmp/gdn_golden";

fn read_dump(path: &Path) -> (Vec<usize>, Vec<f32>) {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    let mut shape = Vec::new();
    let mut off = 0;
    loop {
        let d = u64::from_le_bytes(bytes[off..off + 8].try_into().unwrap());
        off += 8;
        if d == 0 {
            break;
        }
        shape.push(d as usize);
    }
    let data_bytes = &bytes[off..];
    let n = data_bytes.len() / 4;
    let mut data = Vec::with_capacity(n);
    for i in 0..n {
        data.push(f32::from_le_bytes(data_bytes[i * 4..i * 4 + 4].try_into().unwrap()));
    }
    (shape, data)
}

fn load_tensor(name: &str) -> Tensor {
    let (shape, data) = read_dump(&PathBuf::from(DIR).join(format!("{name}.bin")));
    Tensor::from_f32(shape, data)
}

#[test]
fn sequential_t1_calls_match_one_t6_call() {
    if !PathBuf::from(DIR).join("output.bin").exists() {
        eprintln!(
            "skip: no golden dump at {DIR} — run:\n\
             python3 run/scripts/dump_gdn_golden.py"
        );
        return;
    }

    let (in_shape, in_data) = read_dump(&PathBuf::from(DIR).join("input.bin"));
    let hidden = in_shape[1];
    let t = in_shape[0];

    let in_proj_qkv = load_tensor("w_in_proj_qkv.weight");
    let in_proj_z = load_tensor("w_in_proj_z.weight");
    let in_proj_b = load_tensor("w_in_proj_b.weight");
    let in_proj_a = load_tensor("w_in_proj_a.weight");
    let (conv_shape, conv_data) = read_dump(&PathBuf::from(DIR).join("w_conv1d.weight.bin"));
    let conv1d_weight = Tensor::from_f32(vec![conv_shape[0], conv_shape[2]], conv_data);
    let a_log = load_tensor("w_A_log");
    let dt_bias = load_tensor("w_dt_bias");
    let norm_weight = load_tensor("w_norm.weight");
    let out_proj = load_tensor("w_out_proj.weight");

    let dims = GatedDeltaDims {
        num_v_heads: 48,
        num_k_heads: 16,
        head_k_dim: 128,
        head_v_dim: 128,
        conv_kernel_size: conv_shape[2],
    };
    let weights = GatedDeltaWeights {
        in_proj_qkv: &in_proj_qkv,
        in_proj_z: &in_proj_z,
        in_proj_b: &in_proj_b,
        in_proj_a: &in_proj_a,
        conv1d_weight: &conv1d_weight,
        a_log: &a_log,
        dt_bias: &dt_bias,
        norm_weight: &norm_weight,
        out_proj: &out_proj,
    };

    let mut state = vec![0f32; dims.num_v_heads * dims.head_k_dim * dims.head_v_dim];
    let mut conv_state = vec![0f32; conv_shape[0] * (dims.conv_kernel_size - 1)];

    // T separate single-token calls, chaining state+conv_state — exactly
    // how forward.rs's decode loop calls this (one token per forward()).
    let mut sequential_out = vec![0f32; t * hidden];
    for ti in 0..t {
        let row = Tensor::from_f32(vec![1, hidden], in_data[ti * hidden..(ti + 1) * hidden].to_vec());
        let out = gated_delta_forward(&row, &weights, dims, 1e-6, &mut state, &mut conv_state).expect("forward");
        sequential_out[ti * hidden..(ti + 1) * hidden].copy_from_slice(out.as_f32());
    }

    let (hf_shape, hf_data) = read_dump(&PathBuf::from(DIR).join("output.bin"));
    assert_eq!(vec![t, hidden], hf_shape, "shape mismatch vs. golden dump");

    let mut worst_abs = 0f32;
    let mut max_ref = 0f32;
    for (&o, &h) in sequential_out.iter().zip(hf_data.iter()) {
        worst_abs = worst_abs.max((o - h).abs());
        max_ref = max_ref.max(h.abs());
    }
    eprintln!("sequential-T1 vs one-T{t}-call: worst abs diff {worst_abs}, max|hf|={max_ref}");
    let tol = (max_ref * 1e-4).max(1e-6);
    assert!(
        worst_abs < tol,
        "T={t} sequential single-token calls diverge from the real HF T={t}-in-one-call reference: \
         worst abs diff {worst_abs} > tol {tol} — conv_state cross-call history is broken"
    );
}
