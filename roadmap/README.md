# glia roadmap

## done

### format & import
- canonical five encodings implemented: `u32` (16.16 fixed-point), `u16` (8.8), `q8` (32-val blocks), `q4` (32-val blocks), `ternary` (2-bit)
- import pipeline: all weights dequant→f32→canonical at pack time; config stored as integers (eps as 1/ε, sampling per-mille)
- Q4K pass-through: GGUF Q4_K sources written to .model as-is (zero-copy, 0.9 GB vs 1.56 GB Q8 for 1.5b)
- K-quant re-encoding: Q6_K/Q5_K/Q3_K/Q2_K sources re-quantized to Q4K for dtype uniformity
- import normalization: Q4_0→Q4_K, Q4_1→Q4_K at import
- all K-quant dequant: Q2_K, Q3_K, Q4_K, Q5_K, Q6_K in both CPU and importer
- mmap for files >1 GB (Gemma-4 needs 32 MB header scan)
- GGUF loader: dim reversal (GGUF [K,N] → canonical [N,K])

### manifest models (all 4 correct on cpu)
- qwen3-0.6b-abl: correct output, passes HF per-op activation golden (tier3_goldens)
- qwen2.5-coder-1.5b-abl: correct output, verified vs Ollama
- qwen2.5-coder-14b-abl: correct output, verified vs Ollama
- gemma-4-31b: correct output on cpu; sliding-window attention, logit softcapping, K=V tying, per-layer head_dim/kv_heads, GELU activation all implemented
- BOS auto-prepend (gemma-4 was broken without it; fixed by name-lookup in vocab)

### honeycrisp backend (Metal GPU)
- SIMD-parallel Q4 kernels: 16 SIMDs × 32 lanes per threadgroup, 1 row per SIMD
- SIMD-parallel Q8 kernels: same geometry, wider mb32/mb64 variants for deep models
- fused NRM+matmul (Q, K, V with shared RMSNorm), dual NRM (K+V one kernel), GUS NRM (gate+up+SwiGLU fused)
- LARGE4 / LARGE8 variants for wide weight matrices (k_dim > 2048)
- Q4K SIMD kernels: MSL_NRM, MSL_DUAL_NRM, MSL_GUS_NRM, MSL_LARGE; wired into fused decode path
- batch_raw single Metal command buffer for all layers per forward step (minimal CPU overhead)
- qk_norm + RoPE fused path (Qwen3-style per-head Q/K norms)
- QKV bias fusion (Qwen2-style, fused into matmul for mb64 path)
- post_attn_norm / post_ffw_norm path (Gemma-style)
- KV cache: per-layer, GQA-aware, inplace append

### infrastructure
- `mr bench` / `mr status` / `mr profile` / `mr run` — full CLI
- soma manifest (model catalog with tiers)
- three backends: honeycrisp, wgpu+rs, cpu — all functional
- cpu backend: rayon-parallel Q4_K matmul, all K-quant decoders

---

## open proposals

Each proposal is self-contained and independently actionable.

---

### P-1 — Q4K benchmark + quality verification

**Scope:** confirm that Q4K models produce correct output and measure tok/s.

- run `mr bench qwen2.5-coder-1.5b-q4k --steps 50` and compare vs Q8 baseline and Ollama Q4_K_M
- run `mr run qwen2.5-coder-1.5b-q4k --prompt "write fibonacci in rust"` and verify output coherence
- import qwen2.5-coder-14b via Q4K pass-through; benchmark on honeycrisp
- acceptance: tok/s ≥ Q8 (fewer bytes → faster DRAM), output quality visually correct

**Blocks:** everything below that depends on Q4K throughput.

---

### P-2 — honeycrisp parity with Ollama on all 4 manifest models

**Scope:** close the tok/s gap to Ollama Q4_K_M on M-series.

Current state: honeycrisp is faster than cpu but behind Ollama on small models (kernel launch overhead dominates when k_dim is small).

- profile each model with `mr profile` to find the bottleneck op
- for qwen3-0.6b (k_dim=1024, 28 layers): kernel launch dominates — batch multiple layers per Metal command buffer, or reduce barrier count
- for qwen2.5-coder-1.5b: Q4K path (P-1 prerequisite) closes half the gap
- for qwen2.5-coder-14b: Q4K + LARGE4 already fast; verify within 10% of Ollama
- for gemma-4-31b: OOM on honeycrisp — add offload policy (keep only active layers on GPU)
- acceptance: each manifest model ≥ Ollama tok/s on honeycrisp

---

### P-3 — prefill speed (batched forward pass)

**Scope:** accelerate prompt processing (currently token-by-token).

Decode is memory-bandwidth-bound (one token at a time). Prefill with batch > 1 is compute-bound and can be 10-50× faster per token.

- add `prefill(tokens: &[u32])` path to curated decoder: batch KV fills, single causal-masked SDPA
- on honeycrisp: extend matmul kernels to handle arbitrary batch (currently batch=1 only)
- on cpu: add tiled matrix multiply for prefill (not just matvec)
- acceptance: `mr bench --prefill 512` shows > 5× speedup over decode-loop prefill

---

### P-4 — wgpu+rs decode performance

**Scope:** bring wgpu+rs from ~3 tok/s to competitive on non-Apple hardware.

Current state: wgpu+rs is 10-100× behind honeycrisp. Root cause: per-dispatch overhead — every matmul is a separate compute pass.

- port the batch_raw pattern to wgpu: accumulate all layer dispatches into one submit
- implement fused NRM+Q4 WGSL shader (analog of honeycrisp MSL_NRM)
- implement fused GUS NRM WGSL shader
- acceptance: wgpu+rs > 50 tok/s on qwen3-0.6b on any GPU

---

### P-5 — tiled / chunked attention (long context)

**Scope:** remove the 2048-token sequence limit imposed by fixed-size score arrays.

- replace `array<f32, 2048>` in attention WGSL with storage buffer (dynamic allocation)
- implement chunked attention: process keys in 256-token tiles, accumulate softmax numerically stable
- on honeycrisp: extend SDPA kernel to tile over total_seq, not score-array limited
- acceptance: `mr run model --prompt <4096-token context>` completes correctly

---

### P-6 — speculative decoding (draft + verify)

**Scope:** lossless 2-3× throughput improvement using a tiny draft model.

- add `speculate(draft_model, target_model, k=4)` path to mr: draft generates k tokens, target verifies in one forward pass
- qwen3-0.6b as draft for qwen2.5-coder-1.5b (same tokenizer family)
- acceptance: 1.5b model shows ≥ 2× tok/s improvement with 0.6b draft, output identical to greedy

---

### P-7 — BertStyle architecture (encoder-only)

**Scope:** unblock 5 soma models: deberta-zeroshot, modernbert, jina-v5-nano, granite-hap-125m, granite-hap-38m.

- add `arch/encoder/` curated path: bidirectional attention (no causal mask), [CLS] pooling
- BertStyle config parsing: `position_embedding_type`, `type_vocab_size`, `hidden_act`
- bidirectional SDPA kernel on honeycrisp and wgpu+rs (no causal mask, no KV cache)
- import normalization for BERT naming conventions (query → q_proj, etc.)
- acceptance: `mr run jina-v5-nano --prompt "hello world"` returns embedding vector

---

### P-8 — WhisperStyle architecture (encoder-decoder)

**Scope:** unblock whisper-small and all ASR models.

- add `arch/encoder_decoder/` curated path
- Whisper-specific: mel-spectrogram input, cross-attention, forced decoder prefix
- import: handle `model.encoder.*` / `model.decoder.*` tensor naming
- acceptance: `mr run whisper-small --audio path.wav` returns transcript

---

### P-9 — Vision-Language (qwen2.5-vl)

**Scope:** unblock qwen2.5-vl-7b-abl (multimodal).

- add vision encoder path: patch embedding → ViT-style encoder → cross-attention projection
- config parsing: `text_config` / `vision_config` nesting; `image_token_id`; `spatial_merge_size`
- import: normalize qwen2_vl tensor namespace
- acceptance: `mr run qwen2.5-vl-7b --image path.jpg --prompt "describe this"` returns text

---

### P-10 — ANE integration (neural engine)

**Scope:** use Apple Neural Engine for small-head inference (norm, attention head computations).

- `rane` crate: ANE compute path using ANECompilerService or CoreML model conversion
- dispatch NRMNorm, small-head matmul to ANE; keep large matmul on GPU
- benchmark: ANE latency for single-head ops vs Metal
- acceptance: qwen3-0.6b decode uses ANE for norms, shows measurable improvement in overall tok/s

---

### P-11 — AMX integration (Apple Matrix Extensions)

**Scope:** use AMX coprocessor for CPU-side matmul to close gap with llama.cpp.

- `acpu` crate: AMX-accelerated matmul using LLVM AMX intrinsics or assembly
- replace rayon Q4_K matmul with AMX-backed path for cpu backend
- target: cpu backend within 2× of llama.cpp on M4 (llama.cpp uses hand-tuned NEON + AMX)
- acceptance: `mr bench qwen2.5-coder-1.5b --backend cpu` ≥ 50 tok/s on M4

---

### P-12 — safetensors import fully wired

**Scope:** allow import directly from HuggingFace safetensors snapshots (no GGUF intermediary).

- safetensors loader exists but is not wired to config/tokenizer parsing
- wire: `mi import <HF_snapshot_dir>` with safetensors weights → same pipeline as GGUF
- acceptance: `mi import ~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/...` produces correct .model

---

### P-13 — import reverse (transformer → cybergraph)

**Scope:** extract a cybergraph projection from any .model file.

- weight tensors → particles (CID per tensor block)
- layer connectivity → cyberlinks
- attention patterns → dialect candidates
- tokenizer vocab → name particles
- config → root-particle frontmatter
- output: `.graph` file renderable in cyb browser, recompilable via mc
- acceptance: `mr reverse qwen3-0.6b.model → qwen3.graph`, `mr compile qwen3.graph → qwen3-rt.model`, output ε-equivalent on 10 prompts

---

### P-14 — OpenAI-compatible serve endpoint

**Scope:** `mr serve` exposes `/v1/chat/completions` (OpenAI API).

- axum HTTP server in `run/cli/`
- streaming (`stream: true`) via SSE
- multi-model routing: classify with qwen3-0.6b, route to coder-1.5b / coder-14b based on task
- model hot-swap: load/unload within RAM budget
- acceptance: `curl http://localhost:11434/v1/chat/completions -d '{"model":"qwen2.5-coder-1.5b","messages":[...]}'` returns tokens

---

*archive: original plans at `roadmap/canonical-format-alignment.md`, `roadmap/format-support.md`, `roadmap/gemma-4-support.md`, `roadmap/inference-optimization.md`, `roadmap/runtime-universality.md`*
