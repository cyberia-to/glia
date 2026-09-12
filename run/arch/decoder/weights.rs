//! LlamaStyle weight loading.
//!
//! Matmul weights are kept QUANTIZED on host (not dequantized to f32).
//! This reduces memory 6-8× and enables fused dequant+matmul kernels
//! that read quant bytes directly.
//!
//! Non-matmul weights (norms, position caches, embed-for-lookup) are
//! dequantized at load.

use crate::backend::cpu::quant::try_dequantize_to_f32;
use crate::core::dtype::DType;
use crate::core::tensor::{Tensor, TensorData};
use crate::format::{FormatError, LoadedModel};
use crate::arch::decoder::config::LlamaConfig;
use std::sync::Arc;

/// Quantized matmul weight. Bytes stay in their native quant format;
/// `tensor` mirrors them as a Tensor so backends can upload to GPU once
/// during `to_backend()` and skip per-call uploads during inference.
#[derive(Clone)]
pub struct QuantWeight {
    pub shape: Vec<usize>,
    pub dtype: DType,
    /// Raw quant bytes on host — always valid as the CPU fallback.
    pub bytes: Arc<Vec<u8>>,
    /// Same bytes wrapped as a Tensor. After `to_backend()` this is
    /// GPU-resident; before it's a host Tensor backed by `bytes`.
    pub tensor: Tensor,
}

impl QuantWeight {
    pub fn numel(&self) -> usize { self.shape.iter().product() }
    pub fn n(&self) -> usize { self.shape[0] }
    pub fn k(&self) -> usize { self.shape[1] }
}

pub struct LayerWeights {
    pub input_norm: Tensor,
    pub q_proj: QuantWeight,
    pub k_proj: QuantWeight,
    pub v_proj: QuantWeight,
    pub o_proj: QuantWeight,
    pub q_proj_bias: Option<Tensor>,
    pub k_proj_bias: Option<Tensor>,
    pub v_proj_bias: Option<Tensor>,
    pub q_norm: Option<Tensor>,
    pub k_norm: Option<Tensor>,
    /// Pre-FFN norm. (HF "post_attention_layernorm" / GGUF "ffn_norm".)
    pub post_norm: Tensor,
    pub gate_proj: QuantWeight,
    pub up_proj: QuantWeight,
    pub down_proj: QuantWeight,
    /// Gemma 2/3/4: norm applied to attention output before residual.
    pub post_attn_norm: Option<Tensor>,
    /// Gemma 2/3/4: norm applied to FFN output before residual.
    pub post_ffw_norm: Option<Tensor>,
    /// Gemma-4: per-channel scale applied to the residual layer output.
    pub layer_output_scale: Option<Tensor>,
    /// `Some` exactly when `config.layer_types[i] == LinearAttn` — this
    /// layer's `self_attn.*` fields above are unset placeholders (the
    /// tensors don't exist in the source at all) and `forward_layer`
    /// must branch to `backend::cpu::gated_delta` before touching them.
    pub linear_attn: Option<GatedDeltaLayerWeights>,
}

/// GatedDeltaNet layer weights (`linear_attn.*`, spec: ops.md
/// §"GatedDeltaNet"). The five big projections stay quantized, same as
/// every other matmul weight in this file; the four small ones (two
/// [num_v_heads] gate params, the conv kernel, the gate norm) are
/// dequantized at load like the norm weights are.
pub struct GatedDeltaLayerWeights {
    pub in_proj_qkv: QuantWeight,
    pub in_proj_z: QuantWeight,
    pub in_proj_b: QuantWeight,
    pub in_proj_a: QuantWeight,
    pub out_proj: QuantWeight,
    /// `[conv_dim, kernel_size]` — declared `[conv_dim, 1, kernel_size]`
    /// in the source (PyTorch depthwise Conv1d's own layout); reshaped
    /// at load, same bytes.
    pub conv1d_weight: Tensor,
    pub a_log: Tensor,
    pub dt_bias: Tensor,
    pub norm_weight: Tensor,
}

pub struct Weights {
    pub layers: Vec<LayerWeights>,
    pub final_norm: Tensor,
    /// LM head stays quantized. None = tied to embed_tokens (dequanted).
    pub lm_head: Option<QuantWeight>,
    /// Stays quantized on host — used both for the tied-weight lm_head
    /// matmul AND for the per-token embed lookup, which dequantizes ONLY
    /// the one row it needs (`Weights::embed_row`) rather than ever
    /// materializing the whole table. A large-vocab model's embed table
    /// dequantized whole was ~5 GB of dead weight held for single-row
    /// reads and contributed to `run/specs/gated-delta-vl-plan.md`'s
    /// "still doesn't run end-to-end" finding.
    pub embed_tokens_quant: QuantWeight,
    /// Native VL (Qwen3.5/3.8) vision tower — `None` for text-only
    /// models. Spec: ops.md §"VisionTower".
    pub vision: Option<VisionWeights>,
}

/// One vision block's weights, owned (vs. `vision::VisionBlockWeights`,
/// which borrows — `as_ref()` builds that borrowed view for a call).
pub struct VisionBlockOwned {
    pub norm1_weight: Tensor,
    pub norm1_bias: Tensor,
    pub norm2_weight: Tensor,
    pub norm2_bias: Tensor,
    pub qkv_weight: Tensor,
    pub qkv_bias: Tensor,
    pub proj_weight: Tensor,
    pub proj_bias: Tensor,
    pub fc1_weight: Tensor,
    pub fc1_bias: Tensor,
    pub fc2_weight: Tensor,
    pub fc2_bias: Tensor,
}

impl VisionBlockOwned {
    pub fn as_ref(&self) -> crate::backend::cpu::vision::VisionBlockWeights<'_> {
        crate::backend::cpu::vision::VisionBlockWeights {
            norm1_weight: &self.norm1_weight,
            norm1_bias: &self.norm1_bias,
            norm2_weight: &self.norm2_weight,
            norm2_bias: &self.norm2_bias,
            qkv_weight: &self.qkv_weight,
            qkv_bias: &self.qkv_bias,
            proj_weight: &self.proj_weight,
            proj_bias: &self.proj_bias,
            fc1_weight: &self.fc1_weight,
            fc1_bias: &self.fc1_bias,
            fc2_weight: &self.fc2_weight,
            fc2_bias: &self.fc2_bias,
        }
    }
}

pub struct VisionMergerOwned {
    pub norm_weight: Tensor,
    pub norm_bias: Tensor,
    pub fc1_weight: Tensor,
    pub fc1_bias: Tensor,
    pub fc2_weight: Tensor,
    pub fc2_bias: Tensor,
}

impl VisionMergerOwned {
    pub fn as_ref(&self) -> crate::backend::cpu::vision::VisionMergerWeights<'_> {
        crate::backend::cpu::vision::VisionMergerWeights {
            norm_weight: &self.norm_weight,
            norm_bias: &self.norm_bias,
            fc1_weight: &self.fc1_weight,
            fc1_bias: &self.fc1_bias,
            fc2_weight: &self.fc2_weight,
            fc2_bias: &self.fc2_bias,
        }
    }
}

pub struct VisionWeights {
    pub patch_embed_weight: Tensor,
    pub patch_embed_bias: Tensor,
    pub pos_embed_table: Tensor,
    pub blocks: Vec<VisionBlockOwned>,
    pub merger: VisionMergerOwned,
}

impl Weights {
    /// Load all weights using per-layer attention dims from `config`.
    /// LlamaStyle has uniform dims; LlamaStyle+ (Gemma-4) varies by layer.
    ///
    /// Norm-offset families (Gemma 2 / 3) store RMSNorm weights as `w - 1`
    /// because the math is `(1 + w) * x / rms`. We add 1.0 at load so the
    /// runtime stays on one `Op::RmsNorm` codepath. The flag lives on the
    /// family profile.
    pub fn load(lm: &LoadedModel, config: &LlamaConfig) -> Result<Self, FormatError> {
        let hidden_size = config.hidden_size;
        let vocab_size = config.vocab_size;
        let intermediate_size = config.intermediate_size;
        let norm_offset = config.family.rmsnorm_plus_one;

        // Embed: some imports have [vocab, hidden] (HF-style), others
        // [hidden, vocab] (GGUF-native metadata). Physical byte layout is
        // always [vocab × hidden] values row-major, so we just force the
        // shape to [vocab, hidden] regardless of what the metadata says.
        // Stays quantized — see the field doc on `embed_tokens_quant`.
        let embed_tokens_quant = load_quant_weight_reshaped(
            lm,
            "model.embed_tokens.weight",
            vec![vocab_size, hidden_size],
        )?;

        let mut final_norm = load_tensor_f32(lm, "model.norm.weight")?;
        if norm_offset {
            offset_norm_by_one(&mut final_norm);
        }

        let lm_head = if config.tie_word_embeddings {
            None
        } else {
            Some(load_quant_weight_reshaped(
                lm,
                "lm_head.weight",
                vec![vocab_size, hidden_size],
            )?)
        };

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            let q_dim = config.num_attention_heads * config.layer_head_dim(i);
            let kv_dim = config.layer_kv_heads(i) * config.layer_head_dim(i);
            let kind = config.layer_types.get(i).copied().unwrap_or(super::config::LayerKind::Sliding);
            let mut layer = load_layer(
                lm,
                i,
                hidden_size,
                q_dim,
                kv_dim,
                intermediate_size,
                kind,
                config,
            )?;
            if norm_offset {
                offset_norm_by_one(&mut layer.input_norm);
                offset_norm_by_one(&mut layer.post_norm);
                if let Some(ref mut n) = layer.q_norm {
                    offset_norm_by_one(n);
                }
                if let Some(ref mut n) = layer.k_norm {
                    offset_norm_by_one(n);
                }
                if let Some(ref mut n) = layer.post_attn_norm {
                    offset_norm_by_one(n);
                }
                if let Some(ref mut n) = layer.post_ffw_norm {
                    offset_norm_by_one(n);
                }
            }
            layers.push(layer);
        }

        let vision = match &config.vision {
            Some(vc) => Some(load_vision_weights(lm, vc)?),
            None => None,
        };

        Ok(Self {
            embed_tokens_quant,
            layers,
            final_norm,
            lm_head,
            vision,
        })
    }

    /// Dequantize exactly one row of the embed table — the `hidden_size`
    /// values for `token_id`, nothing else. Works for any canonical
    /// encoding: row byte length is `total_bytes / vocab_size`, not a
    /// hardcoded block size, so it stays correct whichever encoding
    /// `canonical_encoding_for` chose for this tensor at import time.
    /// Requires `hidden_size` elements to end on a byte boundary for the
    /// encoding in use (true for every block size in this codebase — 32,
    /// 256 — as long as `hidden_size` is itself a multiple of the block
    /// size, which every model here satisfies; a model that didn't would
    /// already have failed import's own block-alignment check).
    pub fn embed_row(&self, token_id: usize, vocab_size: usize) -> Result<Vec<f32>, FormatError> {
        let qw = &self.embed_tokens_quant;
        if qw.bytes.is_empty() {
            return Err(FormatError::Invalid(
                "embed_tokens_quant bytes are gone (host bytes were freed after a GPU upload) \
                 — embed lookup needs them; this backend's uploads_quant_weights() path doesn't \
                 support host-side row lookup yet".into(),
            ));
        }
        let row_bytes = qw.bytes.len() / vocab_size;
        let start = token_id * row_bytes;
        let end = start + row_bytes;
        let bytes = qw.bytes.get(start..end).ok_or_else(|| {
            FormatError::Invalid(format!(
                "token_id {token_id} out of range for embed table ({vocab_size} rows)"
            ))
        })?;
        try_dequantize_to_f32(bytes, qw.dtype)
            .map_err(|e| FormatError::Invalid(format!("dequant embed row {token_id}: {e}")))
    }
}

/// Add 1.0 to every element of a norm weight tensor in place.
/// Gemma family RMSNorm applies `(1 + w) * x / rms`; storing weights with
/// the +1 baked in lets the runtime use a single `Op::RmsNorm` codepath.
fn offset_norm_by_one(t: &mut Tensor) {
    let mut data = t.to_f32_vec();
    for v in data.iter_mut() {
        *v += 1.0;
    }
    *t = Tensor::from_f32(t.shape.clone(), data);
}

fn load_tensor_f32(lm: &LoadedModel, name: &str) -> Result<Tensor, FormatError> {
    let meta = lm
        .tensors
        .iter()
        .find(|t| t.name == name)
        .ok_or_else(|| FormatError::Invalid(format!("missing tensor {name}")))?;
    let bytes = lm
        .tensor_bytes_owned(name)
        .ok_or_else(|| FormatError::Invalid(format!("bytes missing for {name}")))?;
    let f32s = try_dequantize_to_f32(&bytes, meta.dtype)
        .map_err(|e| FormatError::Invalid(format!("dequant {name}: {e}")))?;
    Tensor::try_from_f32(meta.shape.clone(), f32s)
        .map_err(|e| FormatError::Invalid(format!("tensor {name}: {e}")))
}

fn load_quant_weight(lm: &LoadedModel, name: &str) -> Result<QuantWeight, FormatError> {
    let meta = lm
        .tensors
        .iter()
        .find(|t| t.name == name)
        .ok_or_else(|| FormatError::Invalid(format!("missing tensor {name}")))?;
    let bytes = lm
        .tensor_bytes_owned(name)
        .ok_or_else(|| FormatError::Invalid(format!("bytes missing for {name}")))?;
    let raw: Arc<Vec<u8>> = Arc::new(bytes);
    let tensor = Tensor {
        shape: meta.shape.clone(),
        dtype: meta.dtype,
        data: TensorData::Host(raw.clone()),
    };
    Ok(QuantWeight {
        shape: meta.shape.clone(),
        dtype: meta.dtype,
        bytes: raw,
        tensor,
    })
}

/// Load a tensor and override its shape to the expected canonical form.
/// Used for embed/lm_head where import may have reported GGUF-native [K, N].
/// Panics if element count would mismatch.
fn load_tensor_f32_reshaped(
    lm: &LoadedModel,
    name: &str,
    shape: Vec<usize>,
) -> Result<Tensor, FormatError> {
    let t = load_tensor_f32(lm, name)?;
    let declared: usize = t.shape.iter().product();
    let expected: usize = shape.iter().product();
    if declared != expected {
        return Err(FormatError::Invalid(format!(
            "{name}: declared shape {:?} ({} elems) != expected {:?} ({} elems)",
            t.shape, declared, shape, expected
        )));
    }
    Ok(Tensor::try_from_f32(shape, t.to_f32_vec())
        .map_err(|e| FormatError::Invalid(format!("{name}: {e}")))?)
}

fn load_quant_weight_reshaped(
    lm: &LoadedModel,
    name: &str,
    shape: Vec<usize>,
) -> Result<QuantWeight, FormatError> {
    let mut qw = load_quant_weight(lm, name)?;
    let declared: usize = qw.shape.iter().product();
    let expected: usize = shape.iter().product();
    if declared != expected {
        return Err(FormatError::Invalid(format!(
            "{name}: declared shape {:?} ({} elems) != expected {:?} ({} elems)",
            qw.shape, declared, shape, expected
        )));
    }
    qw.shape = shape.clone();
    qw.tensor = Tensor {
        shape,
        dtype: qw.dtype,
        data: TensorData::Host(qw.bytes.clone()),
    };
    Ok(qw)
}

/// Load the native VL vision tower — `model.visual.*` tensors, plain
/// f32 throughout (the whole tower is ~460M params / ~1.8GB f32, no
/// memory pressure to trade correctness-first simplicity away for —
/// see `backend::cpu::vision`'s module doc). Spec: ops.md
/// §"VisionTower".
fn load_vision_weights(
    lm: &LoadedModel,
    vc: &super::config::VisionConfig,
) -> Result<VisionWeights, FormatError> {
    let patch_dim = vc.in_channels * vc.temporal_patch_size * vc.patch_size * vc.patch_size;
    // Declared as Conv3d [hidden, C, T, P, P] in the source; flatten the
    // trailing 4 dims into one K axis for matmul (the degenerate-
    // Conv3d-as-matmul equivalence, ops.md §"VisionTower" step 1).
    let patch_embed_weight =
        load_tensor_f32_reshaped(lm, "model.visual.patch_embed.proj.weight", vec![vc.hidden_size, patch_dim])?;
    let patch_embed_bias = load_tensor_f32(lm, "model.visual.patch_embed.proj.bias")?;
    let pos_embed_table = load_tensor_f32_reshaped(
        lm,
        "model.visual.pos_embed.weight",
        vec![vc.num_position_embeddings, vc.hidden_size],
    )?;

    let mut blocks = Vec::with_capacity(vc.depth);
    for i in 0..vc.depth {
        let p = format!("model.visual.blocks.{i}");
        blocks.push(VisionBlockOwned {
            norm1_weight: load_tensor_f32(lm, &format!("{p}.norm1.weight"))?,
            norm1_bias: load_tensor_f32(lm, &format!("{p}.norm1.bias"))?,
            norm2_weight: load_tensor_f32(lm, &format!("{p}.norm2.weight"))?,
            norm2_bias: load_tensor_f32(lm, &format!("{p}.norm2.bias"))?,
            qkv_weight: load_tensor_f32(lm, &format!("{p}.attn.qkv.weight"))?,
            qkv_bias: load_tensor_f32(lm, &format!("{p}.attn.qkv.bias"))?,
            proj_weight: load_tensor_f32(lm, &format!("{p}.attn.proj.weight"))?,
            proj_bias: load_tensor_f32(lm, &format!("{p}.attn.proj.bias"))?,
            fc1_weight: load_tensor_f32(lm, &format!("{p}.mlp.linear_fc1.weight"))?,
            fc1_bias: load_tensor_f32(lm, &format!("{p}.mlp.linear_fc1.bias"))?,
            fc2_weight: load_tensor_f32(lm, &format!("{p}.mlp.linear_fc2.weight"))?,
            fc2_bias: load_tensor_f32(lm, &format!("{p}.mlp.linear_fc2.bias"))?,
        });
    }

    let merger = VisionMergerOwned {
        norm_weight: load_tensor_f32(lm, "model.visual.merger.norm.weight")?,
        norm_bias: load_tensor_f32(lm, "model.visual.merger.norm.bias")?,
        fc1_weight: load_tensor_f32(lm, "model.visual.merger.linear_fc1.weight")?,
        fc1_bias: load_tensor_f32(lm, "model.visual.merger.linear_fc1.bias")?,
        fc2_weight: load_tensor_f32(lm, "model.visual.merger.linear_fc2.weight")?,
        fc2_bias: load_tensor_f32(lm, "model.visual.merger.linear_fc2.bias")?,
    };

    Ok(VisionWeights { patch_embed_weight, patch_embed_bias, pos_embed_table, blocks, merger })
}

fn load_layer(
    lm: &LoadedModel,
    i: usize,
    hidden: usize,
    q_dim: usize,
    kv_dim: usize,
    intermediate: usize,
    kind: super::config::LayerKind,
    config: &LlamaConfig,
) -> Result<LayerWeights, FormatError> {
    let prefix = format!("model.layers.{i}");
    let try_load_f32 = |name: &str| -> Option<Tensor> {
        let full = format!("{prefix}.{name}");
        lm.tensors.iter().find(|t| t.name == full)?;
        load_tensor_f32(lm, &full).ok()
    };
    let must_f32 = |name: &str| -> Result<Tensor, FormatError> {
        load_tensor_f32(lm, &format!("{prefix}.{name}"))
    };
    // Matmul weight with expected [N, K]. If metadata stores [K, N],
    // we force the canonical shape (physical byte layout is the same —
    // the whole array is just N*K values row-major).
    let quant_nk = |name: &str, n: usize, k: usize| -> Result<QuantWeight, FormatError> {
        load_quant_weight_reshaped(lm, &format!("{prefix}.{name}"), vec![n, k])
    };

    // GatedDeltaNet layers (spec: ops.md §"GatedDeltaNet") carry
    // `linear_attn.*` instead of `self_attn.*` — those tensors do not
    // exist in the source at all for this layer. Load that branch
    // entirely, and fill q/k/v/o_proj with zero-size placeholders that
    // `forward_layer` guarantees are never read (it branches to
    // `backend::cpu::gated_delta` before reaching any Sdpa code for a
    // `LinearAttn` layer — see that dispatch's own comment).
    if kind == super::config::LayerKind::LinearAttn {
        let placeholder = || QuantWeight {
            shape: vec![0, 0],
            dtype: crate::core::dtype::DType::F32,
            bytes: Arc::new(Vec::new()),
            tensor: Tensor::from_f32(vec![0, 0], Vec::new()),
        };
        let need = |name: &str, v: Option<usize>| -> Result<usize, FormatError> {
            v.ok_or_else(|| {
                FormatError::Invalid(format!(
                    "layer {i} is linear_attention but config has no {name} \
                     (import didn't carry the GatedDeltaNet dims — re-import \
                     with a build that writes them, see ops.md §GatedDeltaNet)"
                ))
            })
        };
        let num_v_heads = need("linear_num_value_heads", config.linear_num_value_heads)?;
        let num_k_heads = need("linear_num_key_heads", config.linear_num_key_heads)?;
        let head_k_dim = need("linear_key_head_dim", config.linear_key_head_dim)?;
        let head_v_dim = need("linear_value_head_dim", config.linear_value_head_dim)?;
        let conv_kernel_dim = need("linear_conv_kernel_dim", config.linear_conv_kernel_dim)?;
        let key_dim = num_k_heads * head_k_dim;
        let value_dim = num_v_heads * head_v_dim;
        return Ok(LayerWeights {
            input_norm: must_f32("input_layernorm.weight")?,
            q_proj: placeholder(),
            k_proj: placeholder(),
            v_proj: placeholder(),
            o_proj: placeholder(),
            q_proj_bias: None,
            k_proj_bias: None,
            v_proj_bias: None,
            q_norm: None,
            k_norm: None,
            post_norm: must_f32("post_attention_layernorm.weight")?,
            gate_proj: quant_nk("mlp.gate_proj.weight", intermediate, hidden)?,
            up_proj: quant_nk("mlp.up_proj.weight", intermediate, hidden)?,
            down_proj: quant_nk("mlp.down_proj.weight", hidden, intermediate)?,
            post_attn_norm: try_load_f32("post_attention_norm.weight"),
            post_ffw_norm: try_load_f32("post_ffw_norm.weight"),
            layer_output_scale: try_load_f32("layer_output_scale.weight"),
            linear_attn: Some(GatedDeltaLayerWeights {
                in_proj_qkv: quant_nk("linear_attn.in_proj_qkv.weight", key_dim * 2 + value_dim, hidden)?,
                in_proj_z: quant_nk("linear_attn.in_proj_z.weight", value_dim, hidden)?,
                in_proj_b: quant_nk("linear_attn.in_proj_b.weight", num_v_heads, hidden)?,
                in_proj_a: quant_nk("linear_attn.in_proj_a.weight", num_v_heads, hidden)?,
                out_proj: quant_nk("linear_attn.out_proj.weight", hidden, value_dim)?,
                conv1d_weight: load_tensor_f32_reshaped(
                    lm,
                    &format!("{prefix}.linear_attn.conv1d.weight"),
                    vec![key_dim * 2 + value_dim, conv_kernel_dim],
                )?,
                a_log: must_f32("linear_attn.A_log")?,
                dt_bias: must_f32("linear_attn.dt_bias")?,
                norm_weight: must_f32("linear_attn.norm.weight")?,
            }),
        });
    }

    // Qwen3.5/3.8: q_proj is TWICE q_dim wide (Q + a per-element sigmoid
    // gate applied to the attention output later — see FamilyProfile::
    // has_attn_output_gate's doc comment). o_proj below still takes the
    // ungated q_dim width: the gate never reaches it.
    let q_proj_dim = if config.family.has_attn_output_gate { q_dim * 2 } else { q_dim };
    let q_proj  = quant_nk("self_attn.q_proj.weight", q_proj_dim, hidden)?;
    let k_proj  = quant_nk("self_attn.k_proj.weight", kv_dim, hidden)?;
    let v_proj  = quant_nk("self_attn.v_proj.weight", kv_dim, hidden)?;
    if i == 0 && std::env::var("RUN_DEBUG_WEIGHTS").is_ok() {
        for (name, qw) in [("q_proj", &q_proj), ("k_proj", &k_proj), ("v_proj", &v_proj)] {
            let b = &qw.bytes[..18.min(qw.bytes.len())];
            let i16_scale = i16::from_le_bytes([b[0], b[1]]) as f32 / 2048.0;
            let f16_scale = half::f16::from_bits(u16::from_le_bytes([b[0], b[1]])).to_f32();
            eprintln!("  layer0 {name} dtype={:?} bytes[0..2]={:02X}{:02X} as_i16_scale={:.6} as_f16_scale={:.6}",
                qw.dtype, b[0], b[1], i16_scale, f16_scale);
        }
        let qn_name = format!("{prefix}.self_attn.q_norm.weight");
        if let Some(qn) = lm.tensors.iter().find(|t| t.name == qn_name) {
            let qn_bytes = lm.tensor_bytes(&qn_name).unwrap_or(&[]);
            eprintln!("  layer0 q_norm dtype={:?} size={} shape={:?}", qn.dtype, qn_bytes.len(), qn.shape);
            if qn_bytes.len() >= 4 {
                let v = f32::from_le_bytes([qn_bytes[0], qn_bytes[1], qn_bytes[2], qn_bytes[3]]);
                eprintln!("  layer0 q_norm bytes[0..4]={:02X}{:02X}{:02X}{:02X} as_f32={:.6}",
                    qn_bytes[0], qn_bytes[1], qn_bytes[2], qn_bytes[3], v);
            }
        }
        let input_norm = must_f32("input_layernorm.weight")?;
        let in_vals = input_norm.try_as_f32().unwrap_or(&[]);
        let in_m = in_vals.iter().map(|v| v.abs()).fold(0f32, f32::max);
        let in_rms = (in_vals.iter().map(|v|v*v).sum::<f32>() / in_vals.len() as f32).sqrt();
        eprintln!("  layer0 input_norm abs_max={in_m:.4} rms={in_rms:.4} len={}", in_vals.len());
    }
    Ok(LayerWeights {
        input_norm: must_f32("input_layernorm.weight")?,
        q_proj,
        k_proj,
        v_proj,
        o_proj: quant_nk("self_attn.o_proj.weight", hidden, q_dim)?,
        q_proj_bias: try_load_f32("self_attn.q_proj.bias"),
        k_proj_bias: try_load_f32("self_attn.k_proj.bias"),
        v_proj_bias: try_load_f32("self_attn.v_proj.bias"),
        q_norm: try_load_f32("self_attn.q_norm.weight"),
        k_norm: try_load_f32("self_attn.k_norm.weight"),
        post_norm: must_f32("post_attention_layernorm.weight")?,
        gate_proj: quant_nk("mlp.gate_proj.weight", intermediate, hidden)?,
        up_proj: quant_nk("mlp.up_proj.weight", intermediate, hidden)?,
        down_proj: quant_nk("mlp.down_proj.weight", hidden, intermediate)?,
        post_attn_norm: try_load_f32("post_attention_norm.weight"),
        post_ffw_norm: try_load_f32("post_ffw_norm.weight"),
        layer_output_scale: try_load_f32("layer_output_scale.weight"),
        linear_attn: None,
    })
}
