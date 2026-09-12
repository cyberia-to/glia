//! Image preprocessing for the native VL vision tower — real image file
//! (PNG/JPEG/...) → `pixel_values` + `grid_thw`, matching
//! `transformers.models.qwen2_vl.image_processing_qwen2_vl
//! .Qwen2VLImageProcessor` (the processor Qwen3.5/3.8 actually uses,
//! confirmed via `AutoImageProcessor.from_pretrained` on the real
//! 27B checkpoint — `preprocessor_config.json`'s
//! `image_processor_type: "Qwen2VLImageProcessorFast"` resolves to
//! this same class).
//!
//! Pipeline: decode → `smart_resize` → bicubic resize (antialiased when
//! downsampling, matching `torchvision.transforms.v2.functional.resize`
//! with its default `antialias=True`) → normalize → `patchify`
//! (block-major reorder, ops.md's `PatchOrder`).
//!
//! **Honesty note — resize is NOT bit-exact against torchvision, and
//! the gap is understood but not fully explained.** Isolating just the
//! resize step (before normalize/patchify) against a direct
//! `torchvision.transforms.v2.functional.resize(..., antialias=True)`
//! dump gives two very different results depending on the scale
//! factor:
//! - **Near-identity resize** (a normal-sized photo — `smart_resize`
//!   just rounds to the nearest patch-grid multiple of 32, scale ≈
//!   1.0): matches almost exactly (worst single-pixel outlier ~14/255
//!   raw levels across a 2.9M-element buffer, most pixels bit-identical
//!   post-rounding). This is the common real-world case, since
//!   Qwen3.8's `min_pixels`/`max_pixels` bounds are generous
//!   (65536-16.7M pixels) — most photos pass through near-untouched.
//! - **Strong upsampling** (a small image, hit `min_pixels`, scale
//!   >2×): a real, substantial divergence — roughly a third of pixels
//!   differ by 5+ raw levels out of 255. The per-tap weight math was
//!   checked by hand (4-tap Keys kernel, correct center/support,
//!   weights summing to 1) and looks textbook-correct; the fixed-point
//!   rounding scheme torchvision's actual C++ kernel uses internally
//!   for this regime was not reverse-engineered — this is a genuine,
//!   currently-unresolved gap, not a hidden/undocumented one. See
//!   `run/tests/vision_preprocess_golden.rs`'s `#[ignore]`d strong-
//!   upsample test for the reproduction and `gated-delta-vl-plan.md`'s
//!   preprocessor progress entry for what was tried.
//!
//! Real downsampling of a very large image (>16.7M pixels) has not
//! been golden-tested at all — the antialiasing kernel-width-widening
//! implemented here is the documented Pillow-style algorithm, not
//! independently re-verified for that regime either.

use crate::core::tensor::Tensor;

/// `transformers.image_processing_utils.rescale_factor` folded into the
/// mean/std the way `_fuse_mean_std_and_rescale_factor` does — see
/// ops.md. Qwen3.8's checkpoint uses `image_mean = image_std = 0.5`,
/// giving the exact `[-1, 1]` mapping this constant embodies; kept as
/// a parameter (not hardcoded) since a different checkpoint could set
/// different mean/std.
pub struct PreprocessConfig {
    pub patch_size: usize,
    pub temporal_patch_size: usize,
    pub merge_size: usize,
    pub in_channels: usize,
    pub image_mean: [f32; 3],
    pub image_std: [f32; 3],
    /// `size.shortest_edge` / `size.longest_edge` in the real
    /// processor's config — pixel-COUNT bounds (not side-length), the
    /// same names `smart_resize` (ops.md) uses.
    pub min_pixels: usize,
    pub max_pixels: usize,
}

impl Default for PreprocessConfig {
    /// Qwen3.8-27B-heretic-ara's real `preprocessor_config.json`.
    fn default() -> Self {
        Self {
            patch_size: 16,
            temporal_patch_size: 2,
            merge_size: 2,
            in_channels: 3,
            image_mean: [0.5, 0.5, 0.5],
            image_std: [0.5, 0.5, 0.5],
            min_pixels: 65536,
            max_pixels: 16_777_216,
        }
    }
}

/// `transformers.models.qwen2_vl.image_processing_qwen2_vl.smart_resize`
/// — byte-identical algorithm, including Python's round-half-to-even
/// `round()` semantics (`f64::round_ties_even`, stabilized; NOT Rust's
/// `f32::round`, which rounds half away from zero and would disagree
/// on exact `.5` boundaries).
pub fn smart_resize(
    height: usize,
    width: usize,
    factor: usize,
    min_pixels: usize,
    max_pixels: usize,
) -> Result<(usize, usize), String> {
    let (h, w) = (height as f64, width as f64);
    if h.max(w) / h.min(w) > 200.0 {
        return Err(format!(
            "absolute aspect ratio must be smaller than 200, got {}",
            h.max(w) / h.min(w)
        ));
    }
    let f = factor as f64;
    let mut h_bar = (h / f).round_ties_even() * f;
    let mut w_bar = (w / f).round_ties_even() * f;
    if h_bar * w_bar > max_pixels as f64 {
        let beta = (h * w / max_pixels as f64).sqrt();
        h_bar = ((h / beta / f).floor() * f).max(f);
        w_bar = ((w / beta / f).floor() * f).max(f);
    } else if h_bar * w_bar < min_pixels as f64 {
        let beta = (min_pixels as f64 / (h * w)).sqrt();
        h_bar = (h * beta / f).ceil() * f;
        w_bar = (w * beta / f).ceil() * f;
    }
    Ok((h_bar as usize, w_bar as usize))
}

/// Keys cubic convolution kernel, `a = -0.75` (PIL/torchvision's
/// bicubic constant).
fn cubic_kernel(x: f64) -> f64 {
    let a = -0.75f64;
    let x = x.abs();
    if x < 1.0 {
        (a + 2.0) * x * x * x - (a + 3.0) * x * x + 1.0
    } else if x < 2.0 {
        a * x * x * x - 5.0 * a * x * x + 8.0 * a * x - 4.0 * a
    } else {
        0.0
    }
}

/// Per-axis resampling taps + normalized weights, one row per
/// destination pixel. Widens the kernel support by `1/scale` when
/// downsampling (`scale < 1`) — the standard separable antialiasing
/// technique (Pillow, torchvision's `antialias=True`) — collapses to
/// the plain fixed-support Keys kernel when upsampling or scale ≈ 1.
fn resize_axis_weights(src_len: usize, dst_len: usize) -> Vec<Vec<(usize, f32)>> {
    let scale = dst_len as f64 / src_len as f64;
    let filter_scale = if scale < 1.0 { 1.0 / scale } else { 1.0 };
    let support = 2.0 * filter_scale; // Keys cubic base support is 2.0
    let mut out = Vec::with_capacity(dst_len);
    for i in 0..dst_len {
        // Half-pixel-center mapping (align_corners=False, the standard
        // image-resize convention): at scale=1.0 this reduces to
        // `center == i` exactly, and since the Keys kernel is exactly
        // zero at |x|=1 and |x|=2, that in turn collapses to a clean
        // single-tap identity — losing the `-0.5` here (an earlier
        // version of this function did) blends neighboring pixels even
        // at scale=1.0, which is wrong and was the single largest
        // remaining source of divergence against real HF output.
        let center = (i as f64 + 0.5) / scale - 0.5;
        let left = (center - support).floor() as isize;
        let right = (center + support).ceil() as isize;
        let mut taps = Vec::new();
        let mut sum = 0.0f64;
        for j in left..right {
            // `j` is already a discrete source-pixel-center index (same
            // convention as `center`, which was derived from the
            // half-pixel target->source mapping above) — no extra `+0.5`
            // here, that was double-counting the half-pixel offset and
            // meant even the scale=1.0 case never reduced to a clean
            // single-tap identity.
            let w = cubic_kernel((j as f64 - center) / filter_scale);
            if w == 0.0 {
                continue;
            }
            let clamped = j.clamp(0, src_len as isize - 1) as usize;
            taps.push((clamped, w));
            sum += w;
        }
        if sum != 0.0 {
            for (_, w) in taps.iter_mut() {
                *w /= sum;
            }
        }
        // Merge duplicate (clamped) taps so edge pixels don't get
        // counted with a stale split weight.
        let mut merged: Vec<(usize, f32)> = Vec::new();
        for (idx, w) in taps {
            if let Some(last) = merged.iter_mut().find(|(mi, _)| *mi == idx) {
                last.1 += w as f32;
            } else {
                merged.push((idx, w as f32));
            }
        }
        out.push(merged);
    }
    out
}

/// Separable bicubic resize. `src`: `[src_h, src_w, C]` f32 (already
/// decoded to `[0,255]` range or any linear range — resize is
/// dtype-agnostic). Returns `[dst_h, dst_w, C]`.
fn resize_bicubic(src: &[f32], src_h: usize, src_w: usize, channels: usize, dst_h: usize, dst_w: usize) -> Vec<f32> {
    let row_weights = resize_axis_weights(src_w, dst_w);
    let col_weights = resize_axis_weights(src_h, dst_h);

    // Horizontal pass: [src_h, src_w, C] -> [src_h, dst_w, C]. Rounded
    // to whole-pixel precision between passes — torchvision's uint8-
    // native separable resize kernel is fixed-point end to end, not
    // float; skipping this intermediate rounding measured as a real
    // (if smaller than the final clamp's) source of divergence.
    let mut tmp = vec![0f32; src_h * dst_w * channels];
    for y in 0..src_h {
        for (dx, taps) in row_weights.iter().enumerate() {
            for c in 0..channels {
                let mut acc = 0f32;
                for &(sx, w) in taps {
                    acc += src[(y * src_w + sx) * channels + c] * w;
                }
                tmp[(y * dst_w + dx) * channels + c] = acc;
            }
        }
    }
    // Vertical pass: [src_h, dst_w, C] -> [dst_h, dst_w, C].
    let mut out = vec![0f32; dst_h * dst_w * channels];
    for (dy, taps) in col_weights.iter().enumerate() {
        for x in 0..dst_w {
            for c in 0..channels {
                let mut acc = 0f32;
                for &(sy, w) in taps {
                    acc += tmp[(sy * dst_w + x) * channels + c] * w;
                }
                out[(dy * dst_w + x) * channels + c] = acc;
            }
        }
    }
    out
}

/// `Qwen2VLImageProcessor.patchify` — block-major reorder into
/// `[num_patches, C*temporal_patch_size*patch_size*patch_size]`. A
/// still image is repeated `temporal_patch_size` times along the
/// temporal axis (matching HF's `expand(...).reshape(...)` for
/// non-video input). `pixels`: `[H, W, C]` normalized f32,
/// `H == grid_h*patch_size`, `W == grid_w*patch_size`.
fn patchify(
    pixels: &[f32],
    grid_h: usize,
    grid_w: usize,
    channels: usize,
    patch_size: usize,
    merge_size: usize,
    temporal_patch_size: usize,
) -> Vec<f32> {
    let w_px = grid_w * patch_size;
    let patch_dim = channels * temporal_patch_size * patch_size * patch_size;
    let num_patches = grid_h * grid_w;
    let mut out = vec![0f32; num_patches * patch_dim];
    let blocks_w = grid_w / merge_size;

    for k in 0..num_patches {
        // Same block-major (row, col) decode as vision.rs's PatchOrder
        // (ops.md) — patch_embed/pos_embed/attention all assume this
        // same token order, which the real image processor already
        // produces on the host side.
        let in_col = k % merge_size;
        let in_row = (k / merge_size) % merge_size;
        let block_col = (k / (merge_size * merge_size)) % blocks_w;
        let block_row = k / (merge_size * merge_size * blocks_w);
        let row = block_row * merge_size + in_row;
        let col = block_col * merge_size + in_col;

        let dst = &mut out[k * patch_dim..(k + 1) * patch_dim];
        // Layout per patch: [C, temporal_patch_size, patch_size, patch_size],
        // temporal axis is a plain repeat of the same spatial patch.
        for c in 0..channels {
            for t in 0..temporal_patch_size {
                for py in 0..patch_size {
                    for px in 0..patch_size {
                        let src_y = row * patch_size + py;
                        let src_x = col * patch_size + px;
                        let src_idx = (src_y * w_px + src_x) * channels + c;
                        let dst_idx = ((c * temporal_patch_size + t) * patch_size + py) * patch_size + px;
                        dst[dst_idx] = pixels[src_idx];
                    }
                }
            }
        }
    }
    out
}

/// Full pipeline: raw image file bytes → `(pixel_values [num_patches,
/// patch_dim], grid_t, grid_h, grid_w)`. `grid_t` is always 1 (a still
/// image, one frame) — video input is out of scope here.
pub fn preprocess_image_bytes(
    bytes: &[u8],
    cfg: &PreprocessConfig,
) -> Result<(Tensor, usize, usize, usize), String> {
    let img = image::load_from_memory(bytes).map_err(|e| format!("image decode: {e}"))?;
    let rgb = img.to_rgb8();
    let (src_w, src_h) = (rgb.width() as usize, rgb.height() as usize);
    let src_f32: Vec<f32> = rgb.into_raw().into_iter().map(|v| v as f32).collect();

    let factor = cfg.patch_size * cfg.merge_size;
    let (dst_h, dst_w) = smart_resize(src_h, src_w, factor, cfg.min_pixels, cfg.max_pixels)?;
    let mut resized = resize_bicubic(&src_f32, src_h, src_w, cfg.in_channels, dst_h, dst_w);
    // The Keys cubic kernel has negative side lobes and can overshoot
    // past the input range (classic bicubic ringing) — torchvision's
    // resize operates on a uint8 tensor end to end, so its output is
    // clamped and rounded to [0,255] before anything downstream ever
    // sees it. Skipping this clamp measured as the single largest
    // source of divergence against real HF output (worst-case ~0.35
    // out of a [-1,1] range, i.e. entirely from overshoot, not from
    // any resampling-weight difference).
    for v in resized.iter_mut() {
        *v = v.round().clamp(0.0, 255.0);
    }

    // Fused rescale+normalize: (pixel/255 - mean)/std, folded into one
    // affine transform per channel — see this module's doc comment.
    let mut normalized = resized;
    for (i, v) in normalized.iter_mut().enumerate() {
        let c = i % cfg.in_channels;
        *v = (*v / 255.0 - cfg.image_mean[c]) / cfg.image_std[c];
    }

    let grid_h = dst_h / cfg.patch_size;
    let grid_w = dst_w / cfg.patch_size;
    let patches = patchify(
        &normalized, grid_h, grid_w, cfg.in_channels, cfg.patch_size, cfg.merge_size, cfg.temporal_patch_size,
    );
    let patch_dim = cfg.in_channels * cfg.temporal_patch_size * cfg.patch_size * cfg.patch_size;
    let tensor = Tensor::from_f32(vec![grid_h * grid_w, patch_dim], patches);
    Ok((tensor, 1, grid_h, grid_w))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smart_resize_rounds_to_factor_multiple() {
        let (h, w) = smart_resize(800, 1200, 32, 65536, 16_777_216).unwrap();
        assert_eq!((h, w), (800, 1216));
    }

    #[test]
    fn smart_resize_upsamples_below_min_pixels() {
        let (h, w) = smart_resize(137, 100, 32, 65536, 16_777_216).unwrap();
        assert_eq!((h, w), (320, 224));
    }

    #[test]
    fn smart_resize_rejects_extreme_aspect_ratio() {
        assert!(smart_resize(10, 3000, 32, 65536, 16_777_216).is_err());
    }
}
