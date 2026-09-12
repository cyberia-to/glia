//! Gated DeltaNet vs. the real HF reference — real weights, real math.
//!
//! Golden data comes from `/tmp/gdn_golden/`, regenerate with:
//!   python3 run/scripts/dump_gdn_golden.py
//! (needs a venv with torch+transformers and the real
//! heretic-org/Qwen3.8-27B-heretic-ara snapshot on disk — see the
//! script's own header. Not committed: ~440 MB of real-layer weight
//! dumps, regenerable from the HF download any time.)
//! Loads layer 0's real `linear_attn.*` weights (Qwen3.8-27B-heretic-ara,
//! `heretic-org/Qwen3.8-27B-heretic-ara`), runs the real
//! `transformers.models.qwen3_5.Qwen3_5GatedDeltaNet.forward()` on a
//! small random input, and compares against `gated_delta_forward` here.
//!
//! Skipped if the golden dump is missing (not everyone has run the
//! Python extraction — same pattern as `tier3_goldens.rs`).
//!
//! Spec: specs/ops.md §5 "GatedDeltaNet".

use run::backend::cpu::gated_delta::{gated_delta_forward, GatedDeltaDims, GatedDeltaWeights};
use run::backend::cpu::CpuBackend;
use run::core::tensor::Tensor;
use std::path::{Path, PathBuf};

const DIR: &str = "/tmp/gdn_golden";

/// Reads the dump format `extract_and_run.py` writes: a sequence of u64
/// dims terminated by a 0, then raw f32 data — deliberately not npy/safetensors
/// so this test has zero new parsing dependencies.
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
fn gated_delta_matches_hf_reference_on_real_weights() {
    if !PathBuf::from(DIR).join("output.bin").exists() {
        eprintln!(
            "skip: no golden dump at {DIR} — run:\n\
             python3 run/scripts/dump_gdn_golden.py\n\
             (needs a venv with torch+transformers and the real \
             heretic-org/Qwen3.8-27B-heretic-ara snapshot on disk — \
             see the script's own header)"
        );
        return;
    }

    let (in_shape, in_data) = read_dump(&PathBuf::from(DIR).join("input.bin"));
    let x = Tensor::from_f32(in_shape.clone(), in_data);

    let in_proj_qkv = load_tensor("w_in_proj_qkv.weight");
    let in_proj_z = load_tensor("w_in_proj_z.weight");
    let in_proj_b = load_tensor("w_in_proj_b.weight");
    let in_proj_a = load_tensor("w_in_proj_a.weight");
    // Dumped as [conv_dim, 1, kernel_size] (PyTorch Conv1d's own layout,
    // in_channels/groups=1) — same bytes as [conv_dim, kernel_size],
    // squeeze the size-1 dim the way the reference's forward() does
    // (`self.conv1d.weight.squeeze(1)`) before handing it to our conv.
    let (conv_shape, conv_data) = read_dump(&PathBuf::from(DIR).join("w_conv1d.weight.bin"));
    assert_eq!(conv_shape.len(), 3, "expected [conv_dim, 1, kernel_size]");
    assert_eq!(conv_shape[1], 1, "conv1d is depthwise: in_channels/groups must be 1");
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
    // Zero conv_state == this call is the whole conversation's first —
    // identical to the old implicit-zero-padding behavior this golden
    // test already verified, so the T=6 comparison below is unaffected
    // by conv_state's addition (see run/tests/gated_delta_conv_state.rs
    // for the cross-call-history behavior conv_state actually exists for).
    let mut conv_state = vec![0f32; conv_shape[0] * (dims.conv_kernel_size - 1)];
    let cpu_backend = CpuBackend::new();
    let ours = gated_delta_forward(&x, &weights, dims, 1e-6, &mut state, &mut conv_state, &cpu_backend).expect("forward");
    let (hf_shape, hf_data) = read_dump(&PathBuf::from(DIR).join("output.bin"));
    assert_eq!(ours.shape, hf_shape, "output shape mismatch");

    let ours_data = ours.as_f32();
    let mut worst_abs = 0f32;
    let mut worst_i = 0usize;
    let mut max_ref = 0f32;
    for (i, (&o, &h)) in ours_data.iter().zip(hf_data.iter()).enumerate() {
        let diff = (o - h).abs();
        if diff > worst_abs {
            worst_abs = diff;
            worst_i = i;
        }
        max_ref = max_ref.max(h.abs());
    }
    eprintln!(
        "worst abs diff {worst_abs} at idx {worst_i} (ours={} hf={}), max|hf|={max_ref}",
        ours_data[worst_i], hf_data[worst_i]
    );
    // Pure f32-vs-f32, no quantization on either side — should agree to
    // near machine precision modulo summation-order differences between
    // this crate's loop order and PyTorch's BLAS calls. 1e-4 relative to
    // the reference's own magnitude is generous for that, tight enough
    // to catch a real formula error (those show up as O(1) divergence,
    // not O(1e-5)).
    let tol = (max_ref * 1e-4).max(1e-6);
    assert!(
        worst_abs < tol,
        "gated_delta_forward diverges from the real HF reference: worst abs diff {worst_abs} > tol {tol}"
    );
}
