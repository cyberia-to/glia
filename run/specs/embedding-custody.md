# Quantized embedding custody across backend upload

`Weights::embed_row` performs bounded host-side row lookup. Its quantized bytes
must remain available after `LlamaModel::to_backend`, even when the backend uploads
quantized matrix weights. This preserves the row-only memory profile; it does not
restore a full dequantized vocabulary table.

When embeddings also supply the tied language-model head, upload the tensor for
the backend matmul and retain the quantized host bytes for token lookup. With a
separate language-model head, the embedding tensor has no GPU matmul consumer and
need not be uploaded. Other layer/head weights may release their host bytes under
the existing backend contract. Both prefill and decode use the same rule.
