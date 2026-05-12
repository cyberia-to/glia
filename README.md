---
tags: cyber, core
crystal-type: entity
crystal-domain: cyber
alias: glia, model runtime, universal runtime
---
# glia

universal `.model` runtime. graph-agnostic.

glia does not know about [[cybergraph]], [[particles]], or [[cyberlinks]]. it receives a `.model` and runs it. the same runtime that executes a graph-compiled model executes any other model.

```
tru  →  .model artifact  →  glia  →  inference outputs (tensors, features, neural activations)
                                          ↓
                                         mir  (neural features for rendering)
```

## two crates

| crate | binary | what |
|---|---|---|
| `import/` | `mi` | source format → canonical `.model`. boundary where format variance ends and `.model` invariance begins |
| `run/` | `mr` | universal `.model` runtime. backends: honeycrisp (ANE+AMX), wgpu (cross-platform), cpu |

`import` depends on `run`. `run` knows nothing about import.

## import — source format → .model

`mi import <DIR>` converts external model formats to the canonical `.model` file that `run/` operates on.

| format | source | status |
|---|---|---|
| GGUF (single or sharded) | llama.cpp, ollama | wired |
| safetensors (single or sharded) | HuggingFace | loader present, not wired |
| ONNX | generic | loader present, not wired |
| MLX | Apple MLX | not implemented |

after import, the runtime never sees BF16, GGUF block ordering, fused KV tensors, or nested VL configs. format variance ends at the import boundary. workarounds in runtime are bugs.

see `import/specs/` for the full import contract.

## run — universal .model runtime

three execution paths:

| path | what | when |
|---|---|---|
| curated | known-arch fast path (LlamaStyle, BertStyle, DiTDiffusion, ...) | production inference |
| graph IR | generic graph walk over `.model` IR nodes | unknown/novel architectures |
| nox | trident-compiled programs running on nox VM | proved computation |

three backends:

| backend | hardware | status |
|---|---|---|
| honeycrisp | ANE (rane) + AMX (acpu) + Metal GPU (aruminium) | primary |
| wgpu | Vulkan / Metal / DX12 / WebGPU | cross-platform fallback |
| cpu | single-threaded, deterministic | reference |

hardware dispatch on macOS: large matrix ops → `acpu` (AMX); NRF head inference → `rane` (ANE).

see `run/specs/` for the full runtime spec (architecture, ops, IR, quantization, tokenizer, execution contract).

## what glia is not

glia does not compile `.graph` → `.model` — that is [[tru]]. glia does not render — that is [[mir]]. glia receives a `.model` and produces outputs.

## in the stack

```
tru (compile model)   →  .model + graph field (φ*, eigenvectors)
glia (run model)      →  inference outputs (features, activations)
mir (render)          →  R-1.0 world
```

see [[stack]]
