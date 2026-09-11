# Operations

Math definitions for every op in the runtime. Every backend
implements the math below within its dtype tolerance. The CPU
reference library in wgpu+rs implements all of them in f32 — that's
the correctness authority.

Notation: `x`, `y`, tensors. `x[i]`, element access. `⊙`, element-wise
multiply. `@`, matrix multiply. `W`, weight.

## 1. Linear algebra

### Matmul

```
Matmul(x, W) := x @ W^T
  x: shape [..., K]
  W: shape [N, K]
  y: shape [..., N]
  y[..., i] = sum over k=0..K of x[..., k] * W[i, k]
```

All higher-level ops (attention, FFN) reduce to Matmul + elementwise.

### Add, Mul, Sub, Div

Elementwise with broadcasting ([tensor.md](tensor.md#broadcasting)).

### Transpose, Permute, Reshape

Logical rearrangement. Output is a view (same data, new shape/stride)
when possible, otherwise allocates and copies.

### Concat, Split, Chunk

`Concat` glues along one axis. `Split` with explicit sizes, `Chunk`
with equal parts. Shape must match on all other axes.

### Clamp, NanToNum

Numerical stability.
```
Clamp(x, lo, hi): y[i] = min(max(x[i], lo), hi)
NanToNum(x, nan=0, posinf=F32_MAX, neginf=F32_MIN):
  y[i] = x[i] if finite else replacement
```

### Argmax

Index of maximum along axis. Used in greedy decoding.

## 2. Normalization

### RmsNorm

Root-Mean-Square norm, Llama-family.

```
RmsNorm(x, g, ε):
  # x: [..., D], g: [D] (learned gain), ε: small scalar
  rms = sqrt(mean(x^2) + ε)             # mean over last dim only
  y = (x / rms) ⊙ g
```

**Critical:** ε is added to the mean of squares, **before** the square
root. Common bug: adding ε after sqrt (different numerical behavior
for small x).

Tolerance: F32 1e-6, F16 1e-3.

### LayerNorm

Standard layer normalization.

```
LayerNorm(x, g, b, ε):
  # x: [..., D], g: [D] gain, b: [D] bias
  μ = mean(x)
  σ² = mean((x - μ)²)
  y = (x - μ) / sqrt(σ² + ε) ⊙ g + b
```

### BatchNorm, GroupNorm, InstanceNorm

Same structure as LayerNorm, different reduction axis set:

- **BatchNorm**: reduce over [B, ..., spatial], per-channel
- **GroupNorm**: reduce over group of channels + spatial
- **InstanceNorm**: reduce over spatial only, per [B, C]

### AdaLN

Adaptive layer norm, DiT family. Scale and shift are modulated by
an external conditioning signal:

```
AdaLN(x, scale, shift, ε):
  # Scale and shift are produced by a separate MLP from timestep/text
  y_norm = LayerNorm(x, 1, 0, ε)         # no learned g/b
  y = y_norm ⊙ (1 + scale) + shift
```

Variant `AdaLN-Zero`: the conditioning is initialized to produce
zero output, i.e. y = x + residual · gate.

## 3. Position encoding

### Rope (Rotary Position Embedding)

Standard NeoX-style pairing (first half of head_dim with second half):

```
Rope(x, pos, head_dim, base):
  # x: [..., num_heads, head_dim]
  # pos: current sequence position(s)
  # base: rope_theta (typically 10000 or 1000000)
  # head_dim must be even; if odd, validation fails at load time
  half = head_dim / 2
  for j in 0..half:
    θ = pos / base^(2*j / head_dim)
    c, s = cos(θ), sin(θ)
    x1, x2 = x[..., j], x[..., j + half]
    y[..., j]       = x1 * c - x2 * s
    y[..., j+half]  = x1 * s + x2 * c
```

Alternative pairing (standard, not NeoX): consecutive pairs
`(x[2j], x[2j+1])`. Choice is per-model; Qwen/Llama use NeoX. Set by
the architecture template ([arch.md](arch.md)). Families document
which pairing they use.

Cos/sin cache: precompute `cos[pos, j]` and `sin[pos, j]` for all
positions up to max_seq. Per-model `base` (rope_theta) parameter.

**Edge cases:**
- `head_dim` must be even — validated at load, error if odd.
- `pos=0` produces `θ=0 → cos=1, sin=0` → identity rotation. Correct.
- `pos > max_position_embeddings` is an implementation choice:
  extrapolate cos/sin formula (may produce wrong results) OR error.
  Spec: error with `ContextOverflowError`.
- `base < 1.0` or `base > 1e9` → warn but permit; extreme values
  may produce numerical issues.

**Multi-axis RoPE (3D for DiT video):**

For video/image DiT, position is a vector `[t, h, w]`. head_dim is
split into sub-ranges per axis:

```
# dim_per_axis: e.g. [t_dim, h_dim, w_dim] summing to head_dim
# All must be even.
for axis, (pos_axis, dim_axis) in enumerate(zip(pos_vec, dims_per_axis)):
    offset = sum(dims_per_axis[0..axis])
    Rope_on_slice(x[..., offset : offset + dim_axis], pos_axis, dim_axis, base)
```

Each axis gets an independent RoPE over its own sub-range. `base`
may differ per axis (configured per-model).

### SinusoidalEmbed

Diffusion timestep embedding.

```
SinusoidalEmbed(t, dim):
  # t: scalar timestep, dim: embedding dimension
  half = dim / 2
  for j in 0..half:
    freq = exp(-j * log(10000) / half)
    y[2j]     = sin(t * freq)
    y[2j + 1] = cos(t * freq)
```

### RelativePosEmbedding

T5-style learned relative position bias. Adds to attention scores.

### PosEmbed, TokenEmbed

Lookup from a learned embedding table. `y = W[id]`.

## 4. Activation

### Silu (Swish-1)

```
Silu(x) := x * sigmoid(x) = x / (1 + exp(-x))
```

### Gelu

Two variants. Models specify which.

```
Gelu_erf(x)  := x * 0.5 * (1 + erf(x / sqrt(2)))       # exact
Gelu_tanh(x) := 0.5 * x * (1 + tanh(sqrt(2/φ*) * (x + 0.044715 * x^3)))
```

BERT-family uses `Gelu_erf`. GPT-2, Gemma use `Gelu_tanh`. Spec per-model.

### Relu, LeakyRelu, PRelu, Sigmoid, Tanh

Standard element-wise.

### Softmax

Numerically stable (subtract max):
```
Softmax(x, dim):
  m = max(x, dim)
  e = exp(x - m)
  y = e / sum(e, dim)
```

Without the max subtraction, large x produces Inf/NaN.

### SwiGlu

Gated feed-forward. Llama/Qwen/Mistral FFN.

```
SwiGlu(x, W_gate, W_up, W_down):
  gate = x @ W_gate^T
  up   = x @ W_up^T
  y    = (Silu(gate) ⊙ up) @ W_down^T
```

### GeGlu

GELU-gated variant (some encoder-decoder models).

```
GeGlu(x, W_gate, W_up, W_down):
  gate = x @ W_gate^T
  up   = x @ W_up^T
  y    = (Gelu(gate) ⊙ up) @ W_down^T
```

### Glu

Sigmoid-gated (Stable Audio Conformer and similar).

## 5. Attention

### Sdpa (Scaled Dot-Product Attention)

Standard causal or non-causal attention, possibly with Grouped Query
Attention (GQA). Optional additive mask input.

```
Sdpa(Q, K, V, num_heads, kv_heads, head_dim, causal, mask=None):
  # Q: [B, num_heads, Sq, head_dim]
  # K, V: [B, kv_heads, Sk, head_dim]
  # mask (optional): [B, 1, Sq, Sk] or [Sq, Sk] additive f32 mask
  # if kv_heads < num_heads, expand K, V (see GQA below)
  scale = 1 / sqrt(head_dim)
  scores = Q @ K^T * scale              # [B, num_heads, Sq, Sk]
  if causal:
    scores[..., i, j] += causal_mask[i, j]
  if mask is not None:
    scores += broadcast(mask, [B, num_heads, Sq, Sk])
  probs = Softmax(scores, dim=-1)
  y = probs @ V                         # [B, num_heads, Sq, head_dim]
```

**Scale is divided, not multiplied.** Some implementations bake it
into Q; equivalent but spec here uses explicit scale.

**Causal mask value:** use `-1e4` (F16-safe large negative),
NOT `-inf`. Reasons:
- `-inf` in F16 softmax can produce NaN if a whole row is masked
  (all rows should have at least one unmasked entry, but defensively
  `-1e4` + at least one `0` survives)
- F16 overflow during intermediate accumulation is avoided
- After softmax, `exp(-1e4) ≈ 0` to F16 precision — effectively the
  same as `-inf`

**Mask shapes:** attention supports three mask patterns:
- **Causal** (`causal=true`): implicit lower-triangular mask added
  inside the kernel, no input needed
- **Padding mask**: `[B, 1, 1, Sk]` additive, -1e4 for padded positions
- **Generic**: `[B, num_heads, Sq, Sk]` additive

Shape `[Sq, Sk]` or `[1, Sq, Sk]` etc. broadcast along missing dims.

### GQA expansion (when num_heads > kv_heads)

When `num_heads > kv_heads`, each KV head is shared by
`repeat = num_heads / kv_heads` Q heads. Expansion is logical
(no copy):

```
# K has shape [B, kv_heads, Sk, head_dim]
# Expand to [B, num_heads, Sk, head_dim] via repeat_interleave:
K_expanded[b, h, s, d] = K[b, h / repeat, s, d]
# Where h / repeat is integer division.
```

**Must be `repeat_interleave` (groups of `repeat` consecutive Q
heads share one KV head), NOT `tile` (strided sharing).**

Backends may implement expansion as a virtual view (no memory copy)
or as a physical expand. Output must match.

`num_heads % kv_heads == 0` is required; non-integer ratio is invalid.

### SdpaCross

Cross attention: Q from decoder, K/V from encoder.

```
SdpaCross(Q_dec, K_enc, V_enc, num_heads, head_dim):
  # Q: [B, num_heads, Sq, head_dim]
  # K,V: [B, num_heads, Se, head_dim]     (encoder output)
  # No causal mask.
  scale = 1 / sqrt(head_dim)
  probs = Softmax(Q @ K^T * scale, dim=-1)
  y = probs @ V
```

### SdpaWindow

Windowed attention (Swin, Mamba-2 attention step).

Each query attends only to keys within a local window. Implementation
reshapes [Sq] into [num_windows, window_size] and runs attention
inside each window.

### FlashAttention

Same math as Sdpa, different memory access pattern (tiled, avoids
materializing full [Sq, Sk] score matrix). Output must match Sdpa
within ε (verification requirement). For decode (Sq=1), FlashAttention
is equivalent to Sdpa.

### KvCache

Stateful append. One cache per (conversation, layer).

Data structure:
```rust
pub struct KvCache {
    /// [num_layers] — one entry per transformer layer
    pub layers: Vec<LayerKvCache>,
    /// Current position (next write offset). Shared across layers.
    pub past_seq_len: usize,
    /// Maximum sequence length this cache supports.
    pub max_seq: usize,
}

pub struct LayerKvCache {
    /// Shape [kv_heads, max_seq, head_dim] — row-major, contiguous.
    /// Positions 0..past_seq_len are valid; past_seq_len..max_seq are
    /// uninitialized (not read by attention).
    pub k: Tensor,
    pub v: Tensor,
}
```

Append op:
```
KvCache.append(layer_idx, K_new, V_new):
    # K_new, V_new: [kv_heads, s, head_dim], s = seq_len of this step
    L = self.layers[layer_idx]
    p = self.past_seq_len
    L.k[kv, p:p+s, :] = K_new[kv, :, :]    # write per head
    L.v[kv, p:p+s, :] = V_new[kv, :, :]
    # past_seq_len updated only after ALL layers have appended
    # (single update per forward pass, at end)
```

Read for attention (layer `i`, current step):
```
K_full = self.layers[i].k[:, 0:p+s, :]     # slice, view
V_full = self.layers[i].v[:, 0:p+s, :]
# passed to Sdpa
```

**Lifecycle rules:**

1. Cache is allocated once at first forward call, sized to
   `max_seq = config.max_position_embeddings` (capped to a practical
   limit like 32K to control memory).
2. Each decode step appends `s=1`; prefill step may append any `s`.
3. `past_seq_len` advances by `s` at the **end** of a forward call,
   after all layers appended successfully.
4. `reset_kv_cache()` sets `past_seq_len = 0`. Does NOT zero memory
   (uninitialized positions aren't read).
5. If `past_seq_len + s > max_seq` at the start of a forward,
   `ContextOverflowError` is returned. No silent truncation.

**Memory:** for `hidden=4096, num_layers=32, kv_heads=8,
head_dim=128, max_seq=8192, dtype=F16`, one cache is
`32 × 2 × 8 × 8192 × 128 × 2 bytes ≈ 1 GiB`. Budget accordingly.

### QK-norm (Qwen3, DeepSeek-V3)

Applied BEFORE Rope, inside the attention forward:

```
Q = x @ W_q^T                # [B, Sq, num_heads, head_dim]
K = x @ W_k^T                # [B, Sq, kv_heads, head_dim]
Q = RmsNorm(Q, g_q, ε)       # per-head — gain shape [head_dim]
K = RmsNorm(K, g_k, ε)       # per-head
Q = Rope(Q, pos, head_dim, rope_theta)
K = Rope(K, pos, head_dim, rope_theta)
# then Sdpa(Q, K, V, ...)
```

**Critical:** the RmsNorm is applied **per head** — the reduction is
over head_dim, not over (num_heads × head_dim). Each head's vector
of length head_dim gets normalized independently, then multiplied
element-wise by the gain vector of shape [head_dim].

Tolerance: same as RmsNorm.

### GatedDeltaNet (Qwen3.5/3.8/3-Next "linear_attention" layers)

A layer that replaces Sdpa entirely (no scores matrix, no Sq×Sk
softmax) with a per-token recurrent state update — the "delta rule"
(DeltaNet, gated per Qwen3-Next). `layer_types` marks which of a
model's layers use this instead of Sdpa; the two coexist within one
model (e.g. Qwen3.8-27B: 48 GatedDeltaNet layers, 16 Sdpa, 3:1
interleave, never mixed within a layer).

Verified against `transformers.models.qwen3_5.modeling_qwen3_5`
(`Qwen3_5GatedDeltaNet`, `torch_recurrent_gated_delta_rule`) —
2026-09, transformers 5.17.0. The sequential (non-chunked) form below
matches `torch_recurrent_gated_delta_rule`'s reference math exactly;
`torch_chunk_gated_delta_rule` is the same computation reassociated
for parallelism and MUST produce identical output — the sequential
form is the correctness baseline (same "CPU reference first" order
as every other family here), a chunked/parallel kernel is a backend
optimization, not a different answer.

**Per-layer weights** (HF tensor names, `linear_attn.*` under each
decoder layer):
```
in_proj_qkv  [key_dim*2 + value_dim, hidden]   in_proj_z  [value_dim, hidden]
in_proj_b    [num_v_heads, hidden]             in_proj_a  [num_v_heads, hidden]
conv1d.weight [conv_dim, 1, kernel_size]  (depthwise, groups=conv_dim, no bias)
A_log        [num_v_heads]                     dt_bias    [num_v_heads]
norm.weight  [head_v_dim]        (RmsNormGated gain)
out_proj     [hidden, value_dim]
```
Dims (config keys, `[architecture]` in `.model`'s config.toml):
`linear_num_value_heads` (num_v_heads), `linear_num_key_heads`
(num_k_heads), `linear_key_head_dim`, `linear_value_head_dim`,
`linear_conv_kernel_dim`. `key_dim = num_k_heads * key_head_dim`,
`value_dim = num_v_heads * value_head_dim`,
`conv_dim = key_dim*2 + value_dim`. Qwen3.8-27B: 48/16/128/128/4.
`A_log`/`dt_bias` are `num_v_heads`-length — NOT a multiple of any
32/256 quant block, so they fall back to `u32` (unquantized) at
import; see `import/quant.rs`'s block-alignment fallback (`gaps.md`
#7). Small enough (192 bytes at 48 heads) that this costs nothing.

**Forward** (one layer, hidden_states `[B, T, H]`):
```
GatedDeltaNet(x, weights, T):
  # 1. Projections (all bias-free linear)
  mixed_qkv = x @ in_proj_qkv^T          # [B, T, conv_dim]
  z         = x @ in_proj_z^T  -> [B, T, num_v_heads, head_v_dim]
  b         = x @ in_proj_b^T            # [B, T, num_v_heads]
  a         = x @ in_proj_a^T            # [B, T, num_v_heads]

  # 2. Causal depthwise conv along T, THEN activation (silu).
  #    groups=conv_dim: channel c only sees its own kernel row.
  #    "Causal": pad kernel_size-1 zeros on the LEFT, no lookahead.
  mixed_qkv = Silu(CausalConv1d(mixed_qkv, conv1d.weight, kernel_size))

  query, key, value = split(mixed_qkv, [key_dim, key_dim, value_dim], dim=-1)
  query -> [B, T, num_k_heads, head_k_dim]
  key   -> [B, T, num_k_heads, head_k_dim]
  value -> [B, T, num_v_heads, head_v_dim]

  # 3. Gates
  beta = Sigmoid(b)                                  # [B, T, num_v_heads]
  g    = -Exp(A_log) * Softplus(a + dt_bias)          # [B, T, num_v_heads], log-space decay (<=0)

  # 4. K/Q head expansion to num_v_heads (GQA-style, repeat_interleave —
  #    see "GQA expansion" above; same rule, num_v_heads/num_k_heads ratio)
  if num_v_heads > num_k_heads:
    query, key = repeat_interleave(query, key, ratio=num_v_heads/num_k_heads, dim=2)

  # 5. L2-normalize query and key per head (eps=1e-6), THEN scale query
  query = L2Norm(query, dim=-1) / sqrt(head_k_dim)
  key   = L2Norm(key, dim=-1)

  # 6. Sequential recurrence — the actual "delta rule". One state
  #    matrix per (batch, head): [head_k_dim, head_v_dim]. head_k_dim
  #    and head_v_dim need not match (128/128 here, but not assumed).
  state = zeros([B, num_v_heads, head_k_dim, head_v_dim])
  for t in 0..T:
    q_t, k_t, v_t = query[:,:,t], key[:,:,t], value[:,:,t]
    decay_t = Exp(g[:,:,t])                    # [B, num_v_heads], scalar per head
    state = state * decay_t[...,None,None]     # decay the WHOLE state matrix
    kv_mem = sum_k( state[...,k,:] * k_t[...,k] )   # = k_t @ state, [B,H,head_v_dim]
    delta  = (v_t - kv_mem) * beta[:,:,t][...,None]
    state  = state + outer(k_t, delta)         # rank-1 update: state[h,i,j] += k_t[h,i]*delta[h,j]
    out[:,:,t] = sum_k( state[...,k,:] * q_t[...,k] )  # = q_t @ state, [B,H,head_v_dim]

  # 7. Gated RMSNorm (norm BEFORE gate multiply, not after), then out_proj
  out = RmsNorm(out, eps=1e-6) * norm.weight * Silu(z)   # per head_v_dim, reduction over head_v_dim only
  return out.reshape(B, T, value_dim) @ out_proj^T       # [B, T, hidden]
```

**Numerically load-bearing details** (each one differs from the
"obvious" reading and silently breaks output if missed):
- Recurrence runs in **f32** regardless of the model's storage dtype
  (the reference casts explicitly before the loop) — bf16 accumulation
  over hundreds of steps compounds error the decay/delta terms are
  sized against.
- `g` (decay) is **log-space and non-positive**: `state *= exp(g_t)`,
  not `state *= g_t`. `A_log` is stored as `log(A)`, `A` itself never
  materializes.
- L2Norm on Q/K happens **after** conv+split, **before** the
  recurrence loop — not fused into the projections.
- Query scaling (`/ sqrt(head_k_dim)`) happens **after** L2Norm, not
  before — order matters, L2Norm removes magnitude so a pre-scale
  would be silently discarded.
- RmsNormGated normalizes, multiplies by the learned gain, **then**
  multiplies by `Silu(z)` — z is a gate on the normalized-and-scaled
  output, not a pre-norm input.
- No causal mask needed anywhere — the sequential loop **is** the
  causality (token t only ever reads state built from tokens < t).

**KV-cache analogue:** GatedDeltaNet layers carry no growing KV
cache — `state` is one fixed-size `[head_k_dim, head_v_dim]` matrix
per head, updated in place every token. A decode step is one loop
iteration, not an append to a growing tensor (see `KvCache` above,
which this layer type does not use at all). This is the practical
payoff of the 3:1 interleave: 48 of 64 layers carry O(1) memory
across arbitrarily long context, only 16 carry the usual O(T) KvCache.

Tolerance: same as Sdpa (this is attention's replacement, not a new
numeric regime) — but verify in f32 given the recurrence's own
internal f32 requirement above; testing the bf16-storage round-trip
at the tensor boundary is a separate, additional check, not a
substitute for it.

## 6. Convolution

### Conv1d, Conv2d, Conv3d

Standard convolution with stride, padding, dilation, groups. For Conv2d:

```
Conv2d(x, W, bias, kernel, stride, padding, dilation, groups):
  # x: [B, C_in, H, W]
  # W: [C_out, C_in/groups, kH, kW]
  # bias: [C_out] or None
  # output: [B, C_out, H', W']
  H' = (H + 2*pH - dH*(kH-1) - 1) / sH + 1
  W' = (W + 2*pW - dW*(kW-1) - 1) / sW + 1
  for c_out in 0..C_out:
    g = c_out / (C_out / groups)              # which group
    for cin in 0..C_in/groups:
      for ki in 0..kH:
        for kj in 0..kW:
          y[b, c_out, i, j] += x[b, g*(C_in/groups) + cin,
                                 i*sH + ki*dH - pH,
                                 j*sW + kj*dW - pW] * W[c_out, cin, ki, kj]
    y[b, c_out, i, j] += bias[c_out]
```

Out-of-bounds reads (from padding): zero (zero-padding). Other
padding modes (replicate, reflect) are separate ops or flags.

`groups` cases:
- `groups = 1`: standard convolution
- `groups = C_in = C_out`: depthwise (each channel independent)
- `groups = k` (intermediate): grouped convolution (ResNeXt-style)

Constraint: `C_in % groups == 0` and `C_out % groups == 0`.

### ConvTranspose2d

Learned upsampling (inverse strides).

### CausalConv1d

Replicate-pad left so output index `t` depends only on input ≤ t.
Video models (Wan, Hunyuan) use this.

### DepthwiseConv

`groups = C_in`. Each input channel has its own filter.

### Pool

Max or average pooling over spatial window.

```
Pool(x, mode, kernel, stride, padding):
  # mode: max | avg
  # x: [B, C, H, W], kernel: (kH, kW), stride: (sH, sW), padding: (pH, pW)
  H' = (H + 2*pH - kH) / sH + 1
  W' = (W + 2*pW - kW) / sW + 1
  y[b, c, i, j] = REDUCE over (ki in 0..kH, kj in 0..kW):
      x[b, c, i*sH + ki - pH, j*sW + kj - pW]
    where REDUCE = max for mode=max, sum/(kH*kW) for mode=avg
  # Out-of-bounds reads (from padding): -inf for max, 0 for avg
```

Defaults: stride = kernel (non-overlapping), padding = 0.

## 7. Spatial

### Interpolate

Nearest/bilinear/area resizing.

### PixelShuffle / PixelUnshuffle

```
PixelShuffle(x, r):
  # [B, C*r^2, H, W] → [B, C, H*r, W*r]
PixelUnshuffle(x, r):
  # [B, C, H*r, W*r] → [B, C*r^2, H, W]
```

### PatchEmbed

Conv2d with kernel=stride=patch_size, mapping image to sequence of
patch embeddings. Used by ViT, DiT.

### Unpatchify

Inverse of PatchEmbed.

## 8. Sampling

### Sample

Unified sampling op accepting method config.

```
Sample(logits, method):
  match method:
    Greedy:
      return argmax(logits)
    Temperature(t):
      probs = softmax(logits / t)
      return sample_categorical(probs)
    TopK(k, t):
      top_values, top_indices = topk(logits, k)
      probs = softmax(top_values / t)
      return top_indices[sample_categorical(probs)]
    TopP(p, t):
      sorted_logits, sorted_idx = sort_desc(logits)
      sorted_probs = softmax(sorted_logits / t)
      cumsum = cumulative_sum(sorted_probs)
      keep_mask = cumsum <= p   # include exactly up to p
      # always keep first token (handles case where top token alone > p)
      keep_mask[0] = true
      filtered_probs = sorted_probs * keep_mask
      filtered_probs /= sum(filtered_probs)   # renormalize
      return sorted_idx[sample_categorical(filtered_probs)]
    MinP(p, t):
      # keep tokens with post-softmax prob >= p * max(probs)
      probs = softmax(logits / t)
      threshold = p * max(probs)
      keep_mask = probs >= threshold
      probs = probs * keep_mask
      probs /= sum(probs)
      return sample_categorical(probs)
```

Config schema in `[sampling]` TOML ([format.md](format.md)):

```toml
[sampling]
method = "top_p"              # greedy | temperature | top_k | top_p | min_p
temperature = 0.6
top_p = 0.95                  # for top_p
top_k = 40                    # for top_k
min_p = 0.05                  # for min_p
seed = 0                      # 0 = non-deterministic; nonzero = reproducible
```

Defaults: `method = "greedy"`, `temperature = 1.0` if not specified.

`sample_categorical` draws one index from a probability distribution
using a seeded RNG (Xorshift or PCG). Same seed + same distribution
= same output (determinism for reproducibility).

## 9. Quantize / Dequantize

Convert between dtypes. See [quant.md](quant.md) for exact formats.

```
Quantize(x, dtype):    # e.g. F32 → Q4_K
Dequantize(x, source): # e.g. Q4_K → F32
```

Quantized matmul is conceptually `Dequantize(W) ⊙ x` but implemented
as a fused kernel that reads quantized bytes and produces f32/f16 output
without materializing the dequantized weight.

## 10. Fused ops

All fused ops are performance optimizations. Their output must match
the corresponding unfused composition within ε.

- **FusedNormMatmul(x, norm_w, W) = Matmul(RmsNorm(x, norm_w, ε), W)**
- **FusedSkipNorm(x, skip, norm_w) = RmsNorm(x + skip, norm_w, ε), x + skip** (returns both)
- **FusedSwiGlu(x, W_gate, W_up) = Silu(x @ W_gate^T) ⊙ (x @ W_up^T)**

These are NOT new semantics — they must numerically match the
unfused equivalent to within 1e-4 (F32) or 1e-2 (F16). Any backend
may implement them as fused or unfused; choice is a performance
decision.

## Tolerance summary

| Context | F32 | F16 | Q4 |
|---|---|---|---|
| Single op output | 1e-6 | 1e-3 | 1e-2 |
| Layer composition | 1e-5 | 1e-3 | 1e-2 |
| Full forward (hundreds of ops) | 1e-4 | 1e-2 | 5e-2 |

See [test.md](test.md) for how these are verified.
