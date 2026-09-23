//! Autoregressive text generation — prefill + decode + sampling.
//!
//! Spec: specs/execution.md, specs/ops.md §8

use crate::backend::{Backend, BackendError};
use crate::arch::decoder::LlamaModel;
use crate::tokenizer::{ChatMessage, Tokenizer};

/// Uniform interface for any model that can run one forward step and return logits.
///
/// Implemented by [`LlamaModel`] (curated path) and graph executor wrappers
/// (graph path). The caller drives the generation loop; the runner owns state.
pub trait ModelRunner {
    /// Run one forward step for a single token. Returns logits [vocab].
    fn step(&mut self, token: u32, backend: &dyn Backend) -> Result<Vec<f32>, BackendError>;
    /// Reset KV state (start a new conversation).
    fn reset(&mut self);
}

impl ModelRunner for LlamaModel {
    fn step(&mut self, token: u32, backend: &dyn Backend) -> Result<Vec<f32>, BackendError> {
        self.forward(token, backend)
    }
    fn reset(&mut self) {
        self.reset_kv_cache();
    }
}

#[derive(Clone, Copy, Debug)]
pub struct SampleConfig {
    pub method: SampleKind,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SampleKind {
    Greedy,
    TopP,
    TopK,
}

impl Default for SampleConfig {
    fn default() -> Self {
        Self {
            method: SampleKind::Greedy,
            temperature: 1.0,
            top_p: 0.95,
            top_k: 40,
        }
    }
}

/// Sample a token id from logits.
pub fn sample(logits: &[f32], config: SampleConfig) -> u32 {
    match config.method {
        SampleKind::Greedy => argmax(logits) as u32,
        SampleKind::TopP => sample_top_p(logits, config.temperature, config.top_p),
        SampleKind::TopK => {
            let mut logits = logits.to_vec();
            if config.temperature > 0.0 && config.temperature != 1.0 {
                for l in &mut logits {
                    *l /= config.temperature;
                }
            }
            // Select the top-k in O(n), sort only those k.
            let mut pairs: Vec<(usize, f32)> = logits.into_iter().enumerate().collect();
            let k = config.top_k.min(pairs.len()).max(1);
            if k < pairs.len() {
                pairs.select_nth_unstable_by(k - 1, |a, b| b.1.partial_cmp(&a.1).unwrap());
                pairs.truncate(k);
            }
            pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            let kept = &pairs[..];
            let max = kept.iter().map(|(_, v)| *v).fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = kept.iter().map(|(_, v)| (v - max).exp()).collect();
            let total: f32 = exps.iter().sum();
            let r = rand_f32() * total;
            let mut acc = 0f32;
            for (i, e) in exps.iter().enumerate() {
                acc += e;
                if acc >= r {
                    return kept[i].0 as u32;
                }
            }
            kept[0].0 as u32
        }
    }
}

/// Nucleus sampling without sorting the vocabulary.
///
/// The naive form softmaxes and *sorts* all |V| logits per token — ~152k
/// pairs for the qwen family — and that sort was worth more wall time than
/// the entire 28-layer forward pass: measured on an M4 Max, temperature 0.7
/// halved decode from ~250 tok/s to ~105 against greedy. The distribution
/// this samples from is exactly the old one; only the work changed:
///
/// 1. One O(|V|) pass finds the max (for a stable softmax).
/// 2. One O(|V|) pass computes the full softmax denominator — the
///    probabilities are true softmax over the whole vocabulary, not over a
///    truncation.
/// 3. An O(|V|) select pulls the top CANDIDATES logits; only those are
///    sorted. The nucleus is found among them, cutting at top_p of *total*
///    mass, same as before.
/// 4. In the astronomically rare case the candidate set does not hold top_p
///    of the mass (a near-uniform distribution), fall back to the full sort
///    rather than sample from the wrong set.
fn sample_top_p(logits: &[f32], temperature: f32, top_p: f32) -> u32 {
    /// Plenty for any top_p anyone runs: at 0.95 the nucleus of a trained
    /// model is a handful of tokens; 512 leaves two orders of margin.
    const CANDIDATES: usize = 512;

    let inv_t = if temperature > 0.0 { 1.0 / temperature } else { 1.0 };
    let max = logits.iter().fold(f32::NEG_INFINITY, |m, &l| m.max(l)) * inv_t;
    let denom: f32 = logits.iter().map(|&l| (l * inv_t - max).exp()).sum();

    let mut pairs: Vec<(usize, f32)> = logits.iter().map(|&l| l * inv_t).enumerate().collect();
    let k = CANDIDATES.min(pairs.len()).max(1);
    if k < pairs.len() {
        pairs.select_nth_unstable_by(k - 1, |a, b| b.1.partial_cmp(&a.1).unwrap());
        pairs.truncate(k);
    }
    pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    // Probabilities against the FULL softmax denominator.
    let mut cum = 0f32;
    let mut keep = 0;
    for (i, (_, l)) in pairs.iter().enumerate() {
        cum += (l - max).exp() / denom;
        keep = i + 1;
        if cum >= top_p {
            break;
        }
    }
    if cum < top_p && k < logits.len() {
        // The nucleus is wider than the candidate set: do it the slow,
        // certain way.
        let mut all: Vec<(usize, f32)> = logits.iter().map(|&l| l * inv_t).enumerate().collect();
        all.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        pairs = all;
        cum = 0.0;
        keep = 0;
        for (i, (_, l)) in pairs.iter().enumerate() {
            cum += (l - max).exp() / denom;
            keep = i + 1;
            if cum >= top_p {
                break;
            }
        }
    }

    let kept = &pairs[..keep.max(1)];
    let total: f32 = kept.iter().map(|(_, l)| (l - max).exp() / denom).sum();
    let r = rand_f32() * total;
    let mut acc = 0f32;
    for (id, l) in kept {
        acc += (l - max).exp() / denom;
        if acc >= r {
            return *id as u32;
        }
    }
    kept[0].0 as u32
}

fn argmax(logits: &[f32]) -> usize {
    logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0)
}

/// Suppress (or amplify) logits for recently generated tokens before sampling.
///
/// `penalty > 1.0` makes recent tokens *less* likely — the usual case for
/// reducing n-gram loops. `penalty < 1.0` amplifies them. `penalty == 1.0`
/// is a no-op.
///
/// Sign-aware: positive logits are divided, negative logits are multiplied,
/// so the penalty always pushes the score toward zero (following HF's
/// `transformers` convention).
pub fn apply_repetition_penalty(logits: &mut [f32], recent_tokens: &[u32], penalty: f32) {
    if penalty == 1.0 {
        return;
    }
    for &token in recent_tokens {
        let idx = token as usize;
        if idx < logits.len() {
            if logits[idx] > 0.0 {
                logits[idx] /= penalty;
            } else {
                logits[idx] *= penalty;
            }
        }
    }
}

fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut exps: Vec<f32> = logits.iter().map(|v| (v - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    for e in &mut exps {
        *e /= sum;
    }
    exps
}

/// Non-cryptographic RNG — xorshift64 seeded from system time, one global state.
/// Fine for sampling; deterministic seeding comes via SampleConfig in the future.
fn rand_f32() -> f32 {
    use std::sync::Mutex;
    static STATE: Mutex<u64> = Mutex::new(0);
    let mut s = STATE.lock().unwrap();
    if *s == 0 {
        *s = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0x1234567890ABCDEF)
            | 1;
    }
    let mut x = *s;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *s = x;
    (x as f32) / (u64::MAX as f32)
}

/// Generate text autoregressively. Works with any [`ModelRunner`].
///
/// Returns (generated_text, generated_token_count).
pub fn generate(
    model: &mut dyn ModelRunner,
    tok: &Tokenizer,
    backend: &dyn Backend,
    prompt: &str,
    max_tokens: usize,
    sample_cfg: SampleConfig,
) -> Result<(String, usize), BackendError> {
    model.reset();

    // Tokenize prompt. Auto-prepend BOS if the model expects one (gemma
    // family was trained with `<bos>` at start of every input; without it
    // the model produces nonsense). Skip if the prompt already starts with
    // the BOS token text (chat templates may include it explicitly).
    let mut prompt_ids = tok.encode(prompt);
    if let Some(bos) = tok.bos_token_id {
        if prompt_ids.first() != Some(&bos) {
            prompt_ids.insert(0, bos);
        }
    }
    log::debug!("prompt tokens: {}", prompt_ids.len());
    if std::env::var("RUN_DEBUG_TOKENS").is_ok() {
        let pieces: Vec<String> = prompt_ids
            .iter()
            .map(|&t| {
                format!(
                    "{t}=\"{}\"",
                    tok.decode(&[t], false).replace('\n', "\\n")
                )
            })
            .collect();
        eprintln!("prompt encoded: [{}]", pieces.join(" "));
    }

    // Prefill: run forward for each prompt token, keep logits from last.
    let mut logits: Vec<f32> = Vec::new();
    for &tid in &prompt_ids {
        logits = model.step(tid, backend)?;
    }

    // Decode: sample next token, feed back in, repeat.
    let debug_tokens = std::env::var("RUN_DEBUG_TOKENS").is_ok();
    // Cap initial capacity — callers may pass usize::MAX for "unlimited".
    let mut generated = Vec::with_capacity(max_tokens.min(1024));
    for step in 0..max_tokens {
        let next = sample(&logits, sample_cfg);
        if debug_tokens {
            let mut idx: Vec<usize> = (0..logits.len()).collect();
            idx.sort_unstable_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap());
            let top10: Vec<String> = idx
                .iter()
                .take(10)
                .map(|&i| format!(
                    "{i}={:.2}\"{}\"",
                    logits[i],
                    tok.decode(&[i as u32], false).replace('\n', "\\n").replace('"', "\\\"")
                ))
                .collect();
            // Show rank+logit of a probe token if RUN_DEBUG_PROBE is set (e.g. 9079 for ▁Paris)
            let probe = std::env::var("RUN_DEBUG_PROBE")
                .ok()
                .and_then(|s| s.parse::<usize>().ok());
            let probe_str = probe
                .map(|p| {
                    let rank = idx.iter().position(|&i| i == p).unwrap_or(usize::MAX);
                    format!(" probe[{p}]=#{rank}@{:.2}", logits.get(p).copied().unwrap_or(0.0))
                })
                .unwrap_or_default();
            eprintln!(
                "step {step:3}: sampled={next}\"{}\" top10=[{}]{probe_str}",
                tok.decode(&[next], false).replace('\n', "\\n"),
                top10.join(" "),
            );
        }
        if tok.is_eos(next) {
            break;
        }
        generated.push(next);
        logits = model.step(next, backend)?;
    }

    let text = tok.decode(&generated, false);
    Ok((text, generated.len()))
}

/// Build a prompt from a chat message list using the model's template.
pub fn build_chat_prompt(tok: &Tokenizer, messages: &[ChatMessage]) -> String {
    tok.apply_chat_template(messages, true)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn repetition_penalty_no_op_at_one() {
        let mut logits = [1.0, 2.0, -0.5];
        let before = logits;
        apply_repetition_penalty(&mut logits, &[0, 1, 2], 1.0);
        assert_eq!(logits, before);
    }

    #[test]
    fn repetition_penalty_divides_positive_multiplies_negative() {
        let mut logits = [2.0f32, -1.0, 3.0, -4.0];
        apply_repetition_penalty(&mut logits, &[0, 1], 2.0);
        assert!((logits[0] - 1.0).abs() < 1e-6, "positive → divided by 2");
        assert!((logits[1] - (-2.0)).abs() < 1e-6, "negative → multiplied by 2");
        // Indices not in recent list untouched
        assert_eq!(logits[2], 3.0);
        assert_eq!(logits[3], -4.0);
    }

    #[test]
    fn repetition_penalty_skips_out_of_range() {
        let mut logits = [1.0f32, 2.0];
        apply_repetition_penalty(&mut logits, &[0, 5, 10], 2.0); // 5, 10 out of range
        assert_eq!(logits[0], 0.5);
        assert_eq!(logits[1], 2.0); // untouched
    }
}
