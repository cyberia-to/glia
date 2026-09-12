//! 3D interleaved mRoPE vs. the real HF reference.
//!
//! Golden data comes from `/tmp/mrope_golden/`, regenerate with:
//!   python3 run/scripts/dump_mrope_golden.py
//! (needs a venv with torch+transformers; only reads the real
//! `config.json` for its `rope_parameters`/vision config values — no
//! big weight files needed, this is pure index/config arithmetic.)
//!
//! Sequence under test: 5 text tokens, one 4x4-patch single-frame
//! image (spatial_merge_size=2 -> 4 placeholder tokens), 3 more text
//! tokens — exercises the text/vision/text run transitions and the
//! `current_pos` advance rule.
//!
//! Spec: specs/ops.md §3 "Rope"; gated-delta-vl-plan.md's fusion trace.

use run::backend::cpu::mrope::{mrope_cos_sin, mrope_position_ids, ModalityRun};
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

fn worst_diff(ours: &[f32], hf: &[f32]) -> (f32, f32) {
    let mut worst = 0f32;
    let mut max_ref = 0f32;
    for (&o, &h) in ours.iter().zip(hf.iter()) {
        worst = worst.max((o - h).abs());
        max_ref = max_ref.max(h.abs());
    }
    (worst, max_ref)
}

#[test]
fn mrope_matches_hf_reference() {
    if !PathBuf::from(DIR).join("position_ids.bin").exists() {
        eprintln!(
            "skip: no golden dump at {DIR} — run:\n\
             python3 run/scripts/dump_mrope_golden.py"
        );
        return;
    }

    let meta: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(PathBuf::from(DIR).join("meta.json")).unwrap()).unwrap();
    let spatial_merge_size = meta["spatial_merge_size"].as_u64().unwrap() as usize;
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

    // Same sequence the dump script built: 5 text, one (1,4,4) image, 3 text.
    let runs = [
        ModalityRun::Text { len: 5 },
        ModalityRun::Vision { t: 1, h: 4, w: 4 },
        ModalityRun::Text { len: 3 },
    ];
    let (positions, _next_pos) = mrope_position_ids(&runs, spatial_merge_size);

    let (hf_shape, hf_pos) = read_dump(&PathBuf::from(DIR).join("position_ids.bin"));
    assert_eq!(hf_shape.len(), 3, "expected (3, batch, seq_len)");
    let seq_len = hf_shape[2];
    assert_eq!(positions[0].len(), seq_len);

    // hf dump layout: (3, 1, seq_len) row-major -> axis a, token t at hf_pos[a*seq_len + t].
    for (axis, ours_axis) in positions.iter().enumerate() {
        for t in 0..seq_len {
            let hf_v = hf_pos[axis * seq_len + t];
            assert_eq!(
                ours_axis[t], hf_v,
                "position mismatch axis {axis} token {t}: ours={} hf={hf_v}",
                ours_axis[t]
            );
        }
    }

    // Pack per-token (t,h,w) triples for mrope_cos_sin.
    let triples: Vec<[f32; 3]> = (0..seq_len)
        .map(|t| [positions[0][t], positions[1][t], positions[2][t]])
        .collect();
    let (ours_cos, ours_sin) = mrope_cos_sin(
        &triples,
        rope_dim,
        rope_theta,
        [mrope_section[0], mrope_section[1], mrope_section[2]],
    );

    let (cos_shape, hf_cos) = read_dump(&PathBuf::from(DIR).join("cos.bin"));
    let (_, hf_sin) = read_dump(&PathBuf::from(DIR).join("sin.bin"));
    assert_eq!(cos_shape, vec![1, seq_len, rope_dim], "cos shape mismatch");

    let (worst_cos, max_cos) = worst_diff(&ours_cos, &hf_cos);
    let (worst_sin, max_sin) = worst_diff(&ours_sin, &hf_sin);
    eprintln!("worst cos diff {worst_cos} (max|hf|={max_cos}), worst sin diff {worst_sin} (max|hf|={max_sin})");
    assert!(worst_cos < 1e-5, "cos diverges from HF reference: {worst_cos}");
    assert!(worst_sin < 1e-5, "sin diverges from HF reference: {worst_sin}");
}
