//! LlamaStyle configuration parsed from .model config section.

use crate::arch::decoder::families::{AttnScale, FamilyProfile};
use crate::format::FormatError;

/// Per-layer attention kind. LlamaStyle has all `Sliding` (single shape).
/// LlamaStyle+ (Gemma 3/4) interleaves `Sliding` and `Full`; full layers
/// in Gemma 4 use `global_head_dim` / `num_global_key_value_heads`.
/// Qwen3.5/3.8/3-Next interleave `Sliding`/`Full` (relabeled "full" vs
/// everything else by the source `layer_types`) with `LinearAttn` — a
/// GatedDeltaNet layer that replaces Sdpa entirely (spec: ops.md
/// §"GatedDeltaNet"). `LinearAttn` deliberately has NO defaults here:
/// every `layer_head_dim`/`layer_kv_heads`/`layer_window`/`layer_rope_*`
/// method below is Sliding/Full-shaped math that does not apply to it,
/// and `forward_layer` must branch to `backend::cpu::gated_delta` before
/// calling any of them for a `LinearAttn` layer — see that dispatch's own
/// comment for why silently falling through here used to be a bug, not
/// a feature.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LayerKind {
    Sliding,
    Full,
    LinearAttn,
}

/// Activation function for the FFN gate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HiddenActivation {
    Silu,
    GeluTanh,
    GeluErf,
}

#[derive(Clone, Debug)]
pub struct LlamaConfig {
    pub model_type: String,
    pub hidden_size: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub num_hidden_layers: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    pub rope_theta: f32,
    pub rms_norm_eps: f32,
    pub tie_word_embeddings: bool,
    pub head_dim: usize,
    /// Detected from tensor presence.
    pub has_qk_norm: bool,
    pub has_attn_bias: bool,
    pub eos_token_ids: Vec<u32>,

    // ── LlamaStyle+ (Gemma 3/4) optional fields ──
    /// One entry per layer. LlamaStyle defaults to all `Sliding`.
    pub layer_types: Vec<LayerKind>,
    /// Window size for `Sliding` layers (Gemma 3/4 typical: 1024).
    pub sliding_window: Option<usize>,
    /// FFN activation. Default `Silu`.
    pub hidden_activation: HiddenActivation,
    /// Logit softcapping value applied after lm_head. None or 0.0 = skip.
    pub final_logit_softcapping: Option<f32>,
    /// K and V projections share weights (the importer materialises both names).
    /// Informational; the runtime always sees both tensors.
    pub attention_k_eq_v: bool,
    /// Gemma-4: head_dim used by `Full` layers (per arch.md §LlamaStyle+).
    /// None = `Full` layers use the regular `head_dim`.
    pub global_head_dim: Option<usize>,
    /// Gemma-4: kv_heads used by `Full` layers.
    /// None = `Full` layers use the regular `num_key_value_heads`.
    pub num_global_key_value_heads: Option<usize>,
    /// Gemma-4: per-layer-kind rope_theta. `Full` layers use `rope_theta_full`
    /// (typically 1e6), `Sliding` layers use the regular `rope_theta` (1e4).
    /// None = both kinds use the same `rope_theta`.
    pub rope_theta_full: Option<f32>,
    /// Gemma-4: fraction of head_dim that gets rotated for `Full` layers
    /// (`partial_rotary_factor`, e.g. 0.25). Sliding layers always rotate
    /// the full head_dim. None = no partial rotary.
    pub partial_rotary_factor_full: Option<f32>,
    /// Qwen3.5/3.8: interleaved 3D mRoPE frequency-axis split (e.g.
    /// `[11, 11, 10]`, sums to `rope_dim/2`). `None` for every other
    /// family — plain 1D RoPE. Spec: ops.md §"mRoPE, interleaved".
    pub mrope_section: Option<[usize; 3]>,
    /// Gemma family: divisor for attention scaling. Default per HF Gemma 3
    /// is 256 regardless of head_dim. LlamaStyle defaults to head_dim
    /// (standard 1/sqrt(head_dim)). Affects full layers most because their
    /// head_dim differs from sliding's.
    pub query_pre_attn_scalar: Option<usize>,

    /// Per-family quirks derived from `model_type` at parse time.
    /// Runtime code reads `family.*` fields instead of matching on the
    /// string — see `families/` for the per-family profiles.
    pub family: FamilyProfile,

    // ── GatedDeltaNet dims (Qwen3.5/3.8/3-Next) ──
    // Spec: run/specs/ops.md §"GatedDeltaNet". `None` unless
    // `layer_types` contains at least one `LinearAttn` — plain
    // LlamaStyle/LlamaStyle+ models never populate these.
    pub linear_num_value_heads: Option<usize>,
    pub linear_num_key_heads: Option<usize>,
    pub linear_key_head_dim: Option<usize>,
    pub linear_value_head_dim: Option<usize>,
    pub linear_conv_kernel_dim: Option<usize>,

    // ── Native VL (Qwen3.5/3.8) — vision tower + fusion ──
    // Spec: ops.md §"VisionTower". `None` for text-only models.
    pub vision: Option<VisionConfig>,
    pub image_token_id: Option<u32>,
    pub video_token_id: Option<u32>,
    pub vision_start_token_id: Option<u32>,
    pub vision_end_token_id: Option<u32>,
}

/// `[architecture.vision]` — the native VL vision tower's own config,
/// separate from the text decoder's `hidden_size`/etc. Spec: ops.md
/// §"VisionTower".
#[derive(Clone, Copy, Debug)]
pub struct VisionConfig {
    pub hidden_size: usize,
    pub num_heads: usize,
    pub intermediate_size: usize,
    pub depth: usize,
    pub patch_size: usize,
    pub in_channels: usize,
    pub spatial_merge_size: usize,
    pub temporal_patch_size: usize,
    pub num_position_embeddings: usize,
    pub out_hidden_size: usize,
    pub rope_theta: f32,
}

impl LlamaConfig {
    /// Per-layer head_dim. Gemma-4 full layers use `global_head_dim`.
    pub fn layer_head_dim(&self, layer: usize) -> usize {
        match self.layer_types.get(layer).copied() {
            Some(LayerKind::Full) => self.global_head_dim.unwrap_or(self.head_dim),
            _ => self.head_dim,
        }
    }

    /// Per-layer kv_heads. Gemma-4 full layers use `num_global_key_value_heads`.
    pub fn layer_kv_heads(&self, layer: usize) -> usize {
        match self.layer_types.get(layer).copied() {
            Some(LayerKind::Full) => self
                .num_global_key_value_heads
                .unwrap_or(self.num_key_value_heads),
            _ => self.num_key_value_heads,
        }
    }

    /// Returns the sliding-window mask boundary for a layer, or None for full attention.
    pub fn layer_window(&self, layer: usize) -> Option<usize> {
        match self.layer_types.get(layer).copied() {
            Some(LayerKind::Sliding) => self.sliding_window,
            _ => None,
        }
    }

    /// Per-layer RoPE base. Gemma-4 full layers use `rope_theta_full` (1e6);
    /// sliding layers and LlamaStyle use `rope_theta` (1e4 default).
    pub fn layer_rope_theta(&self, layer: usize) -> f32 {
        match self.layer_types.get(layer).copied() {
            Some(LayerKind::Full) => self.rope_theta_full.unwrap_or(self.rope_theta),
            _ => self.rope_theta,
        }
    }

    /// Per-layer rotated dim. Gemma-4 full layers use partial_rotary_factor.
    /// Returns even count ≤ head_dim.
    pub fn layer_rope_dim(&self, layer: usize) -> usize {
        let head_dim = self.layer_head_dim(layer);
        match self.layer_types.get(layer).copied() {
            Some(LayerKind::Full) => match self.partial_rotary_factor_full {
                Some(f) if f > 0.0 && f < 1.0 => {
                    let d = (head_dim as f32 * f) as usize;
                    // round down to even
                    d & !1
                }
                _ => head_dim,
            },
            _ => head_dim,
        }
    }

    /// Per-layer KV cache capacity (in tokens). Caps sliding-attention layers
    /// to their window size so huge max_position_embeddings models don't blow
    /// RAM with mostly-zero KV entries that will never be attended to.
    pub fn layer_kv_cache_seq(&self, layer: usize, global_max_seq: usize) -> usize {
        match self.layer_window(layer) {
            Some(w) if w < global_max_seq => w,
            _ => global_max_seq,
        }
    }

    /// Attention scaling per the family profile:
    ///   - `Unity` (Gemma 4): 1.0 — Q and K are pre-normalised.
    ///   - `FixedDivisor(n)` (Gemma 2/3): 1/sqrt(n), independent of head_dim.
    ///   - `PerHeadDim` (LlamaStyle): 1/sqrt(layer_head_dim).
    pub fn layer_attn_scale(&self, layer: usize) -> f32 {
        match self.family.attn_scale {
            AttnScale::Unity => 1.0,
            AttnScale::FixedDivisor(n) => 1.0 / (n as f32).sqrt(),
            AttnScale::PerHeadDim => 1.0 / (self.layer_head_dim(layer) as f32).sqrt(),
        }
    }
}

impl LlamaConfig {
    pub fn parse(config_toml: &str, tensors: &[crate::format::TensorMeta]) -> Result<Self, FormatError> {
        let value: toml::Value = toml::from_str(config_toml)
            .map_err(|e| FormatError::Invalid(format!("config.toml: {e}")))?;

        let model_type = value
            .get("model_type")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown")
            .to_string();

        let arch = value.get("architecture").ok_or_else(|| {
            FormatError::Invalid("config.toml: missing [architecture]".into())
        })?;

        let get_usize = |key: &str| -> Result<usize, FormatError> {
            arch.get(key)
                .and_then(|v| v.as_integer())
                .map(|i| i as usize)
                .ok_or_else(|| FormatError::Invalid(format!("missing/invalid {key}")))
        };
        let get_usize_default = |key: &str, default: usize| -> usize {
            arch.get(key)
                .and_then(|v| v.as_integer())
                .map(|i| i as usize)
                .unwrap_or(default)
        };

        let hidden_size = get_usize("hidden_size")?;
        let num_attention_heads = get_usize("num_attention_heads")?;
        let num_key_value_heads =
            get_usize_default("num_key_value_heads", num_attention_heads);
        let num_hidden_layers = get_usize("num_hidden_layers")?;
        let intermediate_size = get_usize("intermediate_size")?;
        let vocab_size = get_usize("vocab_size")?;
        let max_position_embeddings = get_usize_default("max_position_embeddings", 2048);

        let rope_theta = arch
            .get("rope_theta")
            .and_then(|v| v.as_float().or_else(|| v.as_integer().map(|i| i as f64)))
            .unwrap_or(10000.0) as f32;

        // rms_norm_eps is stored as direct (0.000001) or inverse (1000000) — canonicalize.
        let eps_raw = arch
            .get("rms_norm_eps")
            .and_then(|v| v.as_float().or_else(|| v.as_integer().map(|i| i as f64)))
            .unwrap_or(1e-6);
        let rms_norm_eps = if eps_raw >= 1.0 {
            (1.0 / eps_raw) as f32
        } else {
            eps_raw as f32
        };

        let tie_word_embeddings = arch
            .get("tie_word_embeddings")
            .and_then(|v| v.as_bool())
            .or_else(|| value.get("tie_word_embeddings").and_then(|v| v.as_bool()))
            .unwrap_or(true);

        // head_dim: config if present, else derive from q_proj shape.
        // Qwen3 uses head_dim=128 independent of hidden_size/num_heads.
        let head_dim = arch
            .get("head_dim")
            .and_then(|v| v.as_integer())
            .map(|i| i as usize)
            .or_else(|| {
                let inferred = tensors
                    .iter()
                    .find(|t| t.name == "model.layers.0.self_attn.q_proj.weight")
                    .map(|t| t.shape[0] / num_attention_heads);
                if inferred.is_some() {
                    log::warn!("head_dim not in config — inferred {} from q_proj shape; add head_dim to config for reliability", inferred.unwrap());
                }
                inferred
            })
            .unwrap_or_else(|| {
                let fallback = hidden_size / num_attention_heads;
                log::warn!("head_dim not in config and q_proj tensor missing — falling back to hidden_size/num_heads = {fallback}");
                fallback
            });

        // Spec validation per arch.md LlamaStyle.
        if head_dim == 0 || head_dim % 2 != 0 {
            return Err(FormatError::Invalid(format!(
                "head_dim must be positive and even, got {head_dim}"
            )));
        }
        if num_attention_heads == 0 {
            return Err(FormatError::Invalid("num_attention_heads must be > 0".into()));
        }
        if num_key_value_heads == 0 || num_attention_heads % num_key_value_heads != 0 {
            return Err(FormatError::Invalid(format!(
                "GQA requires num_heads ({num_attention_heads}) divisible by kv_heads ({num_key_value_heads})"
            )));
        }
        if num_hidden_layers == 0 {
            return Err(FormatError::Invalid("num_hidden_layers must be > 0".into()));
        }
        if vocab_size == 0 {
            return Err(FormatError::Invalid("vocab_size must be > 0".into()));
        }
        if rope_theta <= 0.0 {
            return Err(FormatError::Invalid(format!(
                "rope_theta must be positive, got {rope_theta}"
            )));
        }
        if !(rms_norm_eps > 0.0 && rms_norm_eps < 1.0) {
            return Err(FormatError::Invalid(format!(
                "rms_norm_eps outside sane range (0, 1): {rms_norm_eps}"
            )));
        }

        // Detect variants by tensor presence
        let has_qk_norm = tensors
            .iter()
            .any(|t| t.name == "model.layers.0.self_attn.q_norm.weight");
        let has_attn_bias = tensors
            .iter()
            .any(|t| t.name == "model.layers.0.self_attn.q_proj.bias");

        // EOS tokens from [tokenizer].eos_token_ids
        let eos_token_ids = value
            .get("tokenizer")
            .and_then(|t| t.get("eos_token_ids"))
            .and_then(|v| v.as_array())
            .map(|a| {
                a.iter()
                    .filter_map(|v| v.as_integer().map(|i| i as u32))
                    .collect()
            })
            .unwrap_or_default();

        // ── LlamaStyle+ (Gemma 3/4) parsing ──
        // Spec: specs/format.md §"LlamaStyle+ extra fields"
        let layer_types: Vec<LayerKind> = arch
            .get("layer_types")
            .and_then(|v| v.as_array())
            .map(|a| {
                a.iter()
                    .filter_map(|v| v.as_str())
                    .map(|s| match s {
                        "full_attention" | "full" => LayerKind::Full,
                        // Was silently landing in the `_ => Sliding` arm
                        // below (Qwen3.8-27B: 48 of 64 layers) — Sliding
                        // means "run Sdpa against self_attn tensors that
                        // do not exist on this layer" (it has linear_attn.*
                        // instead). See LayerKind's doc comment.
                        "linear_attention" => LayerKind::LinearAttn,
                        _ => LayerKind::Sliding,
                    })
                    .collect()
            })
            .unwrap_or_else(|| vec![LayerKind::Sliding; num_hidden_layers]);
        if layer_types.len() != num_hidden_layers {
            return Err(FormatError::Invalid(format!(
                "layer_types length {} != num_hidden_layers {}",
                layer_types.len(),
                num_hidden_layers
            )));
        }
        let sliding_window = arch
            .get("sliding_window")
            .and_then(|v| v.as_integer())
            .map(|i| i as usize);
        let hidden_activation = arch
            .get("hidden_activation")
            .and_then(|v| v.as_str())
            .map(|s| match s {
                "gelu_pytorch_tanh" | "gelu_tanh" => HiddenActivation::GeluTanh,
                "gelu" | "gelu_erf" => HiddenActivation::GeluErf,
                _ => HiddenActivation::Silu,
            })
            .unwrap_or(HiddenActivation::Silu);
        let final_logit_softcapping = arch
            .get("final_logit_softcapping")
            .and_then(|v| v.as_float().or_else(|| v.as_integer().map(|i| i as f64)))
            .map(|f| f as f32)
            .filter(|&f| f > 0.0);
        let attention_k_eq_v = arch
            .get("attention_k_eq_v")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        let global_head_dim = arch
            .get("global_head_dim")
            .and_then(|v| v.as_integer())
            .map(|i| i as usize);
        let num_global_key_value_heads = arch
            .get("num_global_key_value_heads")
            .and_then(|v| v.as_integer())
            .map(|i| i as usize);
        let rope_theta_full = arch
            .get("rope_theta_full")
            .and_then(|v| v.as_float().or_else(|| v.as_integer().map(|i| i as f64)))
            .map(|f| f as f32);
        let partial_rotary_factor_full = arch
            .get("partial_rotary_factor_full")
            .and_then(|v| v.as_float().or_else(|| v.as_integer().map(|i| i as f64)))
            .map(|f| f as f32);
        let mrope_section: Option<[usize; 3]> = arch
            .get("mrope_section")
            .and_then(|v| v.as_array())
            .map(|a| a.iter().filter_map(|v| v.as_integer()).map(|i| i as usize).collect::<Vec<_>>())
            .filter(|v| v.len() == 3)
            .map(|v| [v[0], v[1], v[2]]);
        // Explicit config value wins; family profile supplies defaults for
        // families that need a non-head_dim scalar.
        let query_pre_attn_scalar = arch
            .get("query_pre_attn_scalar")
            .and_then(|v| v.as_integer())
            .map(|i| i as usize);
        let family = FamilyProfile::for_model_type(&model_type, query_pre_attn_scalar);
        let linear_num_value_heads = arch
            .get("linear_num_value_heads")
            .and_then(|v| v.as_integer())
            .map(|i| i as usize);
        let linear_num_key_heads = arch
            .get("linear_num_key_heads")
            .and_then(|v| v.as_integer())
            .map(|i| i as usize);
        let linear_key_head_dim = arch
            .get("linear_key_head_dim")
            .and_then(|v| v.as_integer())
            .map(|i| i as usize);
        let linear_value_head_dim = arch
            .get("linear_value_head_dim")
            .and_then(|v| v.as_integer())
            .map(|i| i as usize);
        let linear_conv_kernel_dim = arch
            .get("linear_conv_kernel_dim")
            .and_then(|v| v.as_integer())
            .map(|i| i as usize);

        let image_token_id = arch.get("image_token_id").and_then(|v| v.as_integer()).map(|i| i as u32);
        let video_token_id = arch.get("video_token_id").and_then(|v| v.as_integer()).map(|i| i as u32);
        let vision_start_token_id =
            arch.get("vision_start_token_id").and_then(|v| v.as_integer()).map(|i| i as u32);
        let vision_end_token_id =
            arch.get("vision_end_token_id").and_then(|v| v.as_integer()).map(|i| i as u32);
        let vision = arch.get("vision").map(|vc| {
            let vget = |key: &str, default: usize| -> usize {
                vc.get(key).and_then(|v| v.as_integer()).map(|i| i as usize).unwrap_or(default)
            };
            VisionConfig {
                hidden_size: vget("hidden_size", 1152),
                num_heads: vget("num_heads", 16),
                intermediate_size: vget("intermediate_size", 4304),
                depth: vget("depth", 27),
                patch_size: vget("patch_size", 16),
                in_channels: vget("in_channels", 3),
                spatial_merge_size: vget("spatial_merge_size", 2),
                temporal_patch_size: vget("temporal_patch_size", 2),
                num_position_embeddings: vget("num_position_embeddings", 2304),
                out_hidden_size: vget("out_hidden_size", hidden_size),
                rope_theta: vc.get("rope_theta").and_then(|v| v.as_integer()).map(|i| i as f32).unwrap_or(10000.0),
            }
        });

        Ok(Self {
            model_type,
            hidden_size,
            num_attention_heads,
            num_key_value_heads,
            num_hidden_layers,
            intermediate_size,
            vocab_size,
            max_position_embeddings,
            rope_theta,
            rms_norm_eps,
            tie_word_embeddings,
            head_dim,
            has_qk_norm,
            has_attn_bias,
            eos_token_ids,
            layer_types,
            sliding_window,
            hidden_activation,
            final_logit_softcapping,
            attention_k_eq_v,
            global_head_dim,
            num_global_key_value_heads,
            rope_theta_full,
            partial_rotary_factor_full,
            mrope_section,
            query_pre_attn_scalar,
            family,
            linear_num_value_heads,
            linear_num_key_heads,
            linear_key_head_dim,
            linear_value_head_dim,
            linear_conv_kernel_dim,
            vision,
            image_token_id,
            video_token_id,
            vision_start_token_id,
            vision_end_token_id,
        })
    }
}
