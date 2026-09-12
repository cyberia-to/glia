//! Honeycrisp's `apply_mrope_cos_sin` Metal kernel vs. the real HF
//! reference — same golden data as `mrope_golden.rs`, run through the
//! ACTUAL GPU kernel this time (not just the CPU reference function).
//!
//! This is the one piece of the Qwen3.5/3.8 GPU work that's testable
//! on this machine right now without the full 27B model: it only
//! needs a working Metal device and the small mRoPE golden dump, not
//! multi-gigabyte weights. Skipped (not failed) if either is missing.
//!
//! Spec: specs/ops.md §"mRoPE, interleaved".

#![cfg(target_os = "macos")]

use run::backend::Backend;
use run::backend::cpu::mrope::mrope_cos_sin;
use run::core::tensor::Tensor;
use std::path::{Path, PathBuf};

const DIR: &str = "/tmp/mrope_golden";

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

#[test]
fn honeycrisp_mrope_matches_hf_reference() {
    if !PathBuf::from(DIR).join("q_rotated.bin").exists() {
        eprintln!(
            "skip: no golden dump at {DIR} — run:\n\
             python3 run/scripts/dump_mrope_golden.py"
        );
        return;
    }
    let backend = match run::backend::honeycrisp::HoneycrispBackend::new() {
        Ok(b) => b,
        Err(e) => {
            eprintln!("skip: honeycrisp unavailable on this machine: {e}");
            return;
        }
    };

    let meta: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(PathBuf::from(DIR).join("meta.json")).unwrap()).unwrap();
    let head_dim = meta["head_dim"].as_u64().unwrap() as usize;
    let rope_theta = meta["rope_theta"].as_f64().unwrap() as f32;
    let partial_rotary_factor = meta["partial_rotary_factor"].as_f64().unwrap() as f32;
    let mrope_section: Vec<usize> = meta["mrope_section"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect();
    let rope_dim = ((head_dim as f32 * partial_rotary_factor) as usize) & !1;

    let (pos_shape, hf_pos) = read_dump(&PathBuf::from(DIR).join("position_ids.bin"));
    let seq_len = pos_shape[2];
    // Use token index 6 — inside the image span (see mrope_golden.rs's
    // comment on the 5-text/4-image/3-text layout), where t/h/w genuinely
    // differ, the strongest test of the interleaved axis-ownership rule.
    let t = 6usize;
    let triple = [hf_pos[0 * seq_len + t], hf_pos[1 * seq_len + t], hf_pos[2 * seq_len + t]];
    let (cos, sin) = mrope_cos_sin(&[triple], rope_dim, rope_theta, [mrope_section[0], mrope_section[1], mrope_section[2]]);

    let (q_shape, q_all) = read_dump(&PathBuf::from(DIR).join("q_input.bin"));
    assert_eq!(q_shape, vec![seq_len, head_dim]);
    let q_row = q_all[t * head_dim..(t + 1) * head_dim].to_vec();
    let x = Tensor::from_f32(vec![1, head_dim], q_row);

    let gpu_out = backend
        .apply_mrope_cos_sin(&x, &cos, &sin, head_dim, rope_dim)
        .expect("honeycrisp apply_mrope_cos_sin");
    let gpu_data = backend.download_f32(&gpu_out).expect("download");

    let (_, hf_q_rotated) = read_dump(&PathBuf::from(DIR).join("q_rotated.bin"));
    let hf_row = &hf_q_rotated[t * head_dim..(t + 1) * head_dim];

    let mut worst = 0f32;
    let mut max_ref = 0f32;
    for (&g, &h) in gpu_data.iter().zip(hf_row.iter()) {
        worst = worst.max((g - h).abs());
        max_ref = max_ref.max(h.abs());
    }
    eprintln!("honeycrisp mrope: worst diff {worst} (max|hf|={max_ref})");
    assert!(worst < 1e-4, "honeycrisp apply_mrope_cos_sin diverges from HF reference: {worst}");
}
