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

## Effort

Real, multi-session work — not a config tweak. Steps 1+3 are small
(hours). Step 2 (new op + kernel) and step 4 (VL fix) are each their
own multi-session effort. Do not schedule this as "the next release."
