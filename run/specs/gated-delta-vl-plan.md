# Plan: Gated DeltaNet + native VL (Qwen3.5/3.8 family)

Status: draft, 2026-09-11. Target models: `p-e-w/Qwen3-8B-heretic`
(already in-family, LlamaStyle, no new work) and
`heretic-org/Qwen3.8-27B-heretic-ara` (new family — this plan).

## Why this model needs new work

Qwen3.8-27B (`Qwen3_5ForConditionalGeneration`, model_type `qwen3_5`) is
neither LlamaStyle nor an existing VL family. Reverse-engineered from
`model.safetensors.index.json` (no weights needed — index-only fetch):

- 64 layers, `layer_types`: 48 `linear_attention` + 16 `full_attention`,
  every 4th layer full (`full_attention_interval: 4`). Precedent for
  per-layer-type dispatch within one curated family already exists
  (Gemma 3/4 sliding/full alternation is a LlamaStyle variant flag,
  not a new family) — this extends that pattern, doesn't invent it.
- `linear_attn.{A_log, dt_bias, conv1d.weight, in_proj_a/b/qkv/z, norm,
  out_proj}` — this is Gated DeltaNet (Qwen3-Next's linear-attention
  layer): causal depthwise conv1d pre-mix, per-channel decay (A_log)
  and gate (dt_bias) forming an SSM-style chunked recurrence, output
  gated by `in_proj_z` before `out_proj`. Open reference exists
  (flash-linear-attention `fla` package, HF `modeling_qwen3_next.py`)
  — not undocumented math, just unimplemented here.
- `model.visual.*` — plain ViT (patch_embed Conv2d, transformer blocks,
  a merger MLP into the LM's embedding space). This is `ViTStyle`
  fusion, already scoped (scope.md: "Multimodal (VL) ... Curated:
  hybrid (ViT + LLM)") but [runtime report](../../audit/run/reality.md) shows the simpler qwen2_vl
  already erroring ("nested config + VL arch") — this is a
  prerequisite fix, not new scope.
- `mtp.*` — a DeepSeek-V3-style multi-token-prediction head. Optional:
  correct greedy/sampled generation needs only `language_model.*` +
  `lm_head`. Skip for v1; speculative decoding via MTP is future work
  ([gap audit](../../audit/run/gaps.md) #31, already logged as unacknowledged).

## Work items, in dependency order

1. **Import tensor extraction** (`import/pipeline.rs`): bigger than
   the console-log undercounting first assumed. `layer_types` itself
   already passes through to the manifest untouched (it's just quoted
   verbatim into `layer_types = [...]`) — the real gap is that weight
   extraction assumes ONE tensor-naming scheme for every layer
   (`self_attn.q_proj/k_proj/v_proj...`). 48 of this model's 64 layers
   carry `linear_attn.{A_log,conv1d,in_proj_*,dt_bias,norm,out_proj}`
   instead — a full second extraction branch, selected per layer by
   its `layer_types` entry, quantized and written alongside the
   existing self_attn branch. Only the 16 full-attention layers fit
   the current code as-is.
2. **Spec: GatedDeltaNet op family** — new `ops.md` section (§5.5 or
   §8, alongside Attention): CausalConv1d (exists) → per-channel
   decay/gate recurrence (new primitive, chunk-parallel scan) → output
   gate (Sigmoid/Silu, exists) → out_proj. Reference: `fla`'s
   `chunk_gated_delta_rule` for the exact recurrence math + a
   sequential (non-chunked) reference form for the CPU/correctness
   path first — matches this project's own "correct first, fast
   later" ordering (see [runtime report](../../audit/run/reality.md)'s CPU-first verification pattern).
3. **Spec: mixed-family layer stack** — architecture.md's dispatcher
   assumes one family per model; extend to a per-layer family list
   (already implicit in Gemma's variant-flag precedent, needs to be
   explicit for a family boundary, not just a variant).
4. **VL fusion fix** — the qwen2.5-vl `nested config + VL arch` error
   is a prerequisite: image tower must load correctly and merge into
   the token stream via `image_token_id` placeholder replacement
   before this model's language stack matters at all.
5. **Backend kernel**: wgpu+rs (portable) first — CPU reference
   implementation of the recurrence, matching the project's own
   "CPU is reference library inside wgpu+rs" principle. honeycrisp
   (Metal/ANE) is a speed follow-up, not a correctness blocker.
6. **Golden test**: HF `transformers` reference forward pass on a
   short prompt, per-op activation comparison — same bar as
   qwen3-0.6b-abl's existing golden pass ([runtime report](../../audit/run/reality.md)).

## Sequencing note

Steps 1-3 are architecture/spec work, no GPU needed, gate everything
else. Step 4 (VL fix) and steps 2/5 (GatedDeltaNet) are independent
of each other and can proceed in parallel once 1-3 land. Step 6 gates
calling this "done" — a model that loads without crashing but was
never checked against a reference is exactly the "silent corruption"
scope.md's acceptance criterion #5 forbids.

## New finding (2026-09-11): import itself OOMs on this machine

Ran the actual import against the real 27B weights (heretic-org's repo,
all 6 shards downloaded + verified sha256). Two smaller, unrelated bugs
surfaced and were fixed by hand first: the failed `mi download` runs
never fetched the small metadata siblings (config.json, tokenizer.json,
...) before dying on a large shard - refetched directly, trivial. And
`linear_attn.A_log`/`dt_bias` (48 elements/layer, one scalar per head)
aren't a multiple of the quantizer's 32-wide block - falls back to u32
per-tensor without crashing (`../../audit/run/gaps.md` #7, K-quant block boundaries -
already a known gap, now with a concrete trigger case).

The real blocker: `import_as` peaked at **54 GB resident (48 GB
compressed)** during the packing phase and wedged (`stuck`, 0% CPU) on
this 51.5 GB Mac. The safetensors loader mmaps source shards (cheap,
evictable), but packing apparently accumulates a large fraction of the
model in heap memory rather than streaming tensor-by-tensor to the
output file - fine at 8B (8.7 GB packed, worked cleanly), not at 27B.

This is independent of the linear-attention/VL runtime gap above and
blocks even producing a `.model` file to test against, on this
hardware. Two ways through, not mutually exclusive:

- **Fix the packer to stream** (write each tensor's packed bytes to
  the output file as it's produced, drop it, move to the next) -
  general improvement, benefits every future large-model import, not
  specific to this one.
- **Import on a bigger machine** and copy the resulting `.model` over
  - sidesteps the packer's memory shape without touching it, but only
    helps future imports, not future *runtime* memory (the *runtime*
    concern in the "RAM feasibility" section above was already
    evaluated separately and looks fine - the quantized model is
    ~15-20 GB, comfortably under 48 GB. This is an import-time-only
    problem).

## Progress (2026-09-11, same session as the OOM finding above)

Landed:
- **Import OOM fixed** (the finding above) — `LazySafetensors` (lazy
  mmap-backed reader) + `write_model_file_streaming` (temp-file weights
  instead of one giant `Vec<u8>`). 8B re-import byte-identical to the
  old eager path (regression check, not just "still works"). 27B now
  imports in ~200s, RSS never left ~1 GB (was 54 GB resident, wedged).
- **Missing shard found**: `model-auxiliary.safetensors` (0.85 GB, the
  vision tower) was never fetched by any of the earlier `mi download`
  attempts — refetched directly, import now sees all 1199 tensors.
- **Step 1 (import tensor extraction)**: turned out simpler than
  scoped — the existing per-tensor quant-or-u32-fallback path already
  handles `linear_attn.*` tensors without a dedicated branch (they just
  hit the "not a multiple of 32" fallback and store as u32/lossless).
  No separate extraction branch was needed after all.
- **Step 2 (GatedDeltaNet op spec)**: written into `ops.md` §5, verified
  against the real `transformers.models.qwen3_5` source (installed in
  `/tmp/glia-verify`, a throwaway venv — torch 2.14.0, transformers
  5.17.0) rather than from memory. Exact tensor names, dims, and the
  sequential (`torch_recurrent_gated_delta_rule`) reference algorithm
  are all in the spec now, not just "known technique, GatedDeltaNet."
- **CPU reference implementation**: `run/backend/cpu/gated_delta.rs`,
  the full per-layer forward (projections, causal depthwise conv+SiLU,
  gates, GQA-style K/Q expansion, L2Norm, the delta-rule recurrence,
  RmsNormGated, out_proj) as a standalone function on host f32 — NOT
  yet wired into `Backend`/`Op` multi-backend dispatch (still real
  follow-up work, see below).
- **Verified against real weights, not synthetic ones**: extracted
  layer 0's actual `linear_attn.*` weights from the downloaded
  `heretic-org/Qwen3.8-27B-heretic-ara` snapshot, ran the REAL
  `Qwen3_5GatedDeltaNet.forward()` (transformers' own reference path,
  `missing: [] unexpected: []` on `load_state_dict`) on a 6-token
  random input, dumped input/weights/output to `/tmp/gdn_golden/`.
  `run/tests/gated_delta_golden.rs` compares `gated_delta_forward`
  against that dump: **worst abs diff 1.8e-8 against a 7.3e-3-scale
  reference (~2.5e-6 relative)** — first attempt was 1.4% off (an
  `l2norm` eps-placement bug, `rsqrt(sum(x²)+eps)` not
  `1/max(norm,eps)` — caught BECAUSE the tolerance was tight, not
  loosened to pass). This is the real thing agreeing with the real
  reference on real weights, not two guesses agreeing with each other.
- **Silent misrouting fixed regardless of full integration status**:
  `LayerKind` gained a third variant (`LinearAttn`); the parser no
  longer folds `"linear_attention"` into the `_ => Sliding` arm;
  `forward_layer` now refuses loudly (`BackendError::UnsupportedOp`)
  the instant it sees a `LinearAttn` layer instead of silently running
  Sdpa against `self_attn.*` tensors that don't exist on that layer.
  Full test suite (`cargo test -p run`, minus two pre-existing failures
  unrelated to this work — `TransformerConfig` missing a field, present
  before any of today's changes too, confirmed via `git stash`) is
  green, including the new golden test.

Not done — genuinely still open:
- **Wiring `gated_delta_forward` into `forward_layer`**'s actual
  backend-dispatched, KV-cache-integrated pipeline (today it refuses
  instead of running). The math is verified; the plumbing (per-layer
  state storage analogous to `kv: &mut (Vec<f32>, Vec<f32>)` but O(1)
  not O(T), `Backend`/`Op` integration for a GPU path later) is not
  started.
- **VL fusion** (vision tower forward + merger + `image_token_id`
  placeholder replacement) — entirely untouched this session. Still
  blocked behind the qwen2.5-vl `nested config + VL arch` error this
  plan already named as a prerequisite.
- **honeycrisp/wgpu+rs kernels** for the recurrence — CPU reference
  only. The per-token sequential loop is the correctness baseline by
  design (ops.md's own "CPU reference first" convention); a
  chunked/parallel kernel for speed is unstarted.

## Progress (2026-09-11, continued — wired into forward_layer)

Landed on top of the Progress section above:

- **Two more prerequisite bugs found and fixed, both pre-existing and
  general** (not GatedDeltaNet-specific, but blocked it from loading at
  all): (1) `import/naming.rs`'s `gguf_to_hf` never stripped the
  `model.language_model.` prefix VL checkpoints use — every tensor
  lookup in `run/arch/decoder/weights.rs` hardcodes the shorter
  `model.layers.N.*` LlamaStyle convention, so even this model's plain
  Full-attention layers would have failed to load. (2) The importer
  never carried `linear_num_value_heads`/`linear_num_key_heads`/
  `linear_key_head_dim`/`linear_value_head_dim`/`linear_conv_kernel_dim`
  into the packed config at all — added to both `import/pipeline.rs`
  (write) and `run/arch/decoder/config.rs` (read). Re-imported 27B
  after each fix; both confirmed in the actual packed `.model` file.
- **`gated_delta_forward` wired into `forward_layer`**: `LayerWeights`
  gained a `linear_attn: Option<GatedDeltaLayerWeights>` field (the
  five big projections stay quantized — `QuantWeight`, same as every
  other matmul weight; the four small ones dequant at load like norms
  do); `load_layer` branches on `LayerKind` and fills `self_attn.*`
  fields with inert zero-size placeholders for `LinearAttn` layers
  (never read — `forward_layer` branches before touching them).
  `forward_layer` itself now computes `hidden1` via one of two paths
  (GatedDeltaNet or the original Sdpa block) that both feed the SAME
  shared FFN epilogue unchanged — norm→attention→residual differs,
  FFN doesn't. `LlamaModel` gained `gdn_state: Vec<Option<Vec<f32>>>`,
  the fixed-size recurrent state per `LinearAttn` layer (the KV-cache
  analogue — allocated at load, zeroed in `reset_kv_cache`, threaded
  through as `&mut [f32]` at every `forward_layer` call site).
  `gated_delta_forward` gained a `state: &mut [f32]` parameter so it
  persists across calls (`forward()` processes one token per call,
  prefill included — confirmed from `tier3_goldens.rs`'s own usage —
  so there is no separate "prefill" mode to special-case).
- **The GPU-fused batch-decode path would have silently bypassed all
  of this**: `forward_decode_fused_layers` groups adjacent layers by
  Sdpa geometry (head_dim/kv_heads/window) and hands them to a fused
  kernel that has never heard of `linear_attn.*`; `LinearAttn` layers'
  fallback geometry values (meaningless `_ => Sliding` defaults) would
  have looked like an ordinary uniform group. Added an explicit
  `is_linear_attn` guard that forces those layers through
  `forward_layer` (this dispatch) one at a time instead, never through
  the fused path.
- **`cargo test -p run` still green** (all 7 runnable suites, including
  `gated_delta_golden`) after every change in this batch — checked
  after each of the two prerequisite-bug fixes and after the
  `forward_layer` wiring itself, not just once at the end.
- **A second, independent OOM — found, root-caused, and fixed at the
  RUNTIME's own loader**, not the importer this time:
  `format.rs::read_model_file`'s own doc comment already said "large
  files are mmap'd" but then did `mmap[weights_start..weights_end]
  .to_vec()` anyway — copying the ENTIRE weights section (30 GB for
  this model) into an owned `Vec<u8>` immediately, on top of every
  individual tensor's own copy into its `QuantWeight`/`Tensor` a
  moment later. Same failure class as the import-side fix, same
  playbook: `ModelFile.weights` is now a `WeightBytes` enum
  (`Owned(Vec<u8>)` for small files, `Mapped { mmap, weights_start }`
  for large ones) — `tensor_bytes()` slices directly out of the live
  mmap for the large-file case, no intermediate full copy. This was
  pre-existing (unrelated to GatedDeltaNet) and is exactly the
  documented cause of `gemma-4-31b`'s honeycrisp OOM in `reality.md` —
  same bug, different model hitting it first.
- **Still doesn't run end-to-end on THIS machine, right now**: even
  after the loader fix, `mr run` on the 27B model gets SIGKILLed
  (exit 137) partway through loading — system free memory collapses
  to 0 within ~15s of starting and the process dies within ~80-130s.
  Not a code bug this time as far as traced: the machine has ~12 GB
  already committed to other running apps (optica, Zed, Telegram,
  browsers — this user's own session, not mine to close), and this
  specific model has a large-vocab-specific inefficiency on top of its
  own honest footprint: `Weights::load` unconditionally dequantizes
  the FULL `embed_tokens` table to f32 (5.08 GB for 248,320 × 5120)
  just to serve single-row lookups, and ALSO keeps a quantized mirror
  (`embed_tokens_quant`, ~1.27 GB) that this specific model never uses
  since `tie_word_embeddings = false` here — real weight ≈29.5 GB +
  ~6.3 GB of avoidable embed overhead + ~12 GB other apps gets close
  enough to 51.5 GB physical that it doesn't fit today. Not touched
  this session — real, separate, well-scoped fix (row-wise dequant
  lookup instead of whole-table), but its own piece of work with its
  own ripple effects (a debug env var reads the full f32 table too).

## Progress (2026-09-12 — embed fix + memory ceiling conclusion)

- **Fixed a third, real, separate memory waste**: `Weights::load`
  unconditionally dequantized the ENTIRE `embed_tokens` table to f32
  (~5.08 GB for this model's 248,320-row vocab) just to serve single-
  row lookups, plus kept an unused quantized mirror (this model has
  `tie_word_embeddings = false`, so the mirror served no lm_head
  purpose either). Replaced with `Weights::embed_row(token_id,
  vocab_size)` — dequantizes exactly one row (`total_bytes / vocab_size`
  gives the row's byte length correctly for ANY canonical encoding,
  not a hardcoded block size). Verified via `tier3_goldens.rs`: argmax
  still matches HF's top-5 on the small model after the change — real
  regression coverage, not just "still compiles."
- **Tried to fix the mmap-doubling with `MADV_DONTNEED`, confirmed it
  does not fully work on this Darwin setup**: added
  `WeightBytes::drop_range` (calls `Mmap::unchecked_advise_range` right
  after each tensor's bytes are copied out, per `memmap2`'s own safety
  contract — nothing borrows the range afterward). Measured with
  `footprint` (Apple's own per-process memory tool) before and after:
  footprint still climbed to **58 GB — essentially 2x the model's own
  29.5 GB packed size** — before SIGKILL, only marginally slower than
  without the advise calls. Conclusion: Darwin's accounting for
  repeatedly-touched-then-advised file-backed mmap pages does not
  behave like the Linux semantics the crate's doc comments describe
  ("RSS immediately reduced") — or something else is pinning those
  pages that `unchecked_advise_range` doesn't reach. Not resolved this
  session; the `drop_range` call is a real, harmless, kept improvement
  (best-effort, costs nothing if it doesn't help) but is not sufficient
  by itself.
- **Conclusion: this is now a resource-availability question, not an
  architecture-support gap.** The GatedDeltaNet math is proven correct
  (golden test, 1.8e-8 abs diff against real HF weights) and IS wired
  into the real inference dispatch (`forward_layer`, with persistent
  state, with the GPU-fused-batch bypass closed). Whether a specific
  29.5 GB model's full weights fit in RAM alongside ~12 GB of this
  user's other running apps on a 51.5 GB Mac, on the CPU backend
  specifically, is a deployment constraint — not a defect in what
  "supporting the qwen3_5 architecture" means. Options for whoever
  picks this up next, not mutually exclusive: (a) free up the other
  ~12 GB before running this specific model, (b) run on a machine with
  more headroom, (c) investigate Darwin's mmap/footprint accounting
  further (this needs Instruments or a kernel-level trace, not just
  `footprint`/`vm_stat`, to find what's actually pinning pages), (d) a
  properly lazy `QuantWeight` that reads matmul weight bytes on demand
  per forward call instead of holding all of them resident at once —
  a bigger architectural change than this session's scope, trading
  memory for reading the file from disk/mmap on every token.

## VL / vision tower — scoped from source, not started (2026-09-12)

Traced the real implementation (transformers 5.17.0, same venv as
GatedDeltaNet's verification) rather than guessing. This is bigger than
"a plain ViT" — it's the same "naive dynamic resolution" family as
Qwen2-VL/Qwen2.5-VL, not a fixed-grid encoder. Five real primitives,
none of which exist in this runtime today:

1. **3D-conv patch embed** (`Qwen3_5VisionPatchEmbed`, modeling_qwen3_5.py
   ~961): `nn.Conv3d(in_channels, hidden, kernel=[temporal_patch,
   patch, patch], stride=same)` — treats a still image as a 1-frame
   clip. Tensor names: `model.visual.patch_embed.proj.{weight,bias}`.
2. **Learned position embeddings, bilinearly resampled per image**
   (`Qwen3_5VisionModel.forward`, ~1195; the resampling itself is
   `get_vision_interpolation_indices_and_weights` in the SHARED
   `transformers/vision_utils.py:231` — not in the model file). A
   square `num_position_embeddings`-entry table gets bilinear-
   interpolated to each image's actual (h, w) grid, weighted-summed
   (`(pos_embed(idx) * weight[:,:,None]).sum(1)`), then added to the
   patch embeddings. Tensor: `model.visual.pos_embed.weight`.
3. **Vision RoPE** (`Qwen3_5VisionRotaryEmbedding`, ~77; position ids
   from `get_vision_position_ids`, `vision_utils.py:81`) — axial 2D:
   same freq table for H and W, head_dim//4 each, concatenated;
   position INDICES are block-major over `spatial_merge_size ×
   spatial_merge_size` blocks (not row-major over the raw grid) — get
   this wrong and every downstream number is subtly off, not a crash.
4. **Packed variable-length attention** (`Qwen3_5VisionAttention`,
   ~1011; boundaries from `get_vision_attention_seqlens`,
   `vision_utils.py:68`) — every image/frame in the batch is its own
   attention segment via `cu_seqlens` (no causal mask, bidirectional
   within a segment, zero cross-segment attention). Tensor names:
   `model.visual.blocks.N.attn.{qkv,proj}.{weight,bias}`,
   `model.visual.blocks.N.norm{1,2}.{weight,bias}` (LayerNorm, not
   RmsNorm — a first for this codebase's vision path), `.mlp.linear_fc{1,2}`.
5. **Patch merger** (`Qwen3_5VisionPatchMerger`, ~981): LayerNorm →
   Linear → GELU → Linear, projecting `spatial_merge_size²`-grouped
   patches into the LM's `out_hidden_size` — this is what actually
   produces the image tokens that replace `image_token_id` placeholders
   in the text stream. Tensor names: `model.visual.merger.norm.*`,
   `.linear_fc{1,2}.*`.

**Fusion into the text stream** (not yet traced at all): where exactly
`image_token_id` placeholders in the tokenized prompt get replaced by
`merger` output, and how multiple images/videos of different grid
sizes interleave — this is a SIXTH piece, in `Qwen3_5Model.forward`
(the top-level multimodal wrapper), not yet read.

**Why this wasn't attempted this session**: every one of the 5+1 pieces
is genuinely novel to this runtime (no existing CausalConv3d, no
bilinear interpolation op, no block-major position indexing, no
packed/segmented attention, no LayerNorm-based vision block — every
existing block in this codebase is RmsNorm) and needs its own golden
test against real preprocessed image tensors — which requires tracing
the image PREPROCESSOR too (`preprocessor_config.json`,
`video_preprocessor_config.json` — read but not analyzed) to produce
a correct `(hidden_states, grid_thw)` pair to test against, not just
the model weights. This is comparable in size to the entire
GatedDeltaNet effort above, or larger, and deserves its own dedicated
pass with the same rigor (spec first, CPU reference, golden test
against real weights) rather than a rushed partial implementation in
the tail of an already-long session.

## Effort

Real, multi-session work — not a config tweak. Steps 1+3 are small
(hours). Step 2 (new op + kernel) and step 4 (VL fix) are each their
own multi-session effort. Do not schedule this as "the next release."

## Progress — 2026-09-12, VisionTower implemented + golden-tested

All 5 primitives scoped above (patch embed, position interpolation,
vision RoPE, packed attention, patch merger) are now implemented in
`run/backend/cpu/vision.rs` and verified against the real
`transformers.models.qwen3_5.Qwen3_5VisionModel` forward pass —
`run/tests/vision_golden.rs` extracts real `model.visual.*` weights
(patch_embed, pos_embed, merger, and the first 2 of the model's real
27 blocks — `run/scripts/dump_vision_golden.py`, `depth=2`
truncation) and runs a synthetic single 4×4-patch image through both;
worst abs diff 0.0013 against `max|hf|=511.35` (2.6e-6 relative) —
pure f32, no quantization on either side, comparable precision to the
GatedDeltaNet golden test above.

One real bug caught by the golden test (would NOT have been caught by
a shape check alone): `Qwen3_5VisionPatchMerger.forward` runs LayerNorm
PER-PATCH (over `hidden`=1152 features) BEFORE reshaping 4 patches into
one merge group — I initially implemented it the other way (reshape
then normalize over 4608), which is shape-INCOMPATIBLE with the real
`merger.norm.weight`'s actual width (1152), so it failed loudly
(index-out-of-bounds) rather than silently producing wrong numbers.
Fixed in both `vision.rs` and the `ops.md` §5 pseudocode (was wrong in
the spec too — written before this was checked against source this
carefully).

Still open, per ops.md's "Not yet done" note:
- **Fusion into the text stream**: `image_token_id` placeholder
  replacement + multi-image/video position-id bookkeeping in
  `Qwen3_5Model.forward` — IDs and the 1:1 replacement contract are
  recorded, exact splice mechanics are not traced.
- **Image preprocessor**: turning a real image file into
  `pixel_values` + `grid_thw` (resize, normalize, patchify) — the
  golden test above uses synthetic-but-correctly-shaped patches, not
  a real decoded image.
- **GPU kernels**: CPU reference only, matching GatedDeltaNet's
  current state — deferred by design, not started.
- **Memory ceiling for the full 27B model**: EXPLICITLY DEFERRED by
  the user to a future dedicated "night session when nothing else is
  running on the machine" — not attempted again this session.
