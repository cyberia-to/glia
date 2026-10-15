//! HuggingFace model download.
//!
//! Contract: see [`import/specs/hf.md`]. Selection priority is
//! safetensors > GGUF > ONNX, with the sibling metadata files
//! (config.json, tokenizer.json, etc.) downloaded alongside.

use hf_hub::api::sync::Api;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

/// Live download state, written by the fetch thread, read by whoever draws
/// a progress bar. Shared as atomics because the reader is a UI frame loop
/// and the writer is deep inside an HTTP read loop — neither should wait
/// for the other.
#[derive(Default)]
pub struct DownloadStatus {
    /// The file currently transferring.
    pub file: Mutex<String>,
    /// Bytes done / total for that file.
    pub file_done: AtomicU64,
    pub file_total: AtomicU64,
    /// Files finished / expected overall (weights + metadata siblings).
    pub files_done: AtomicU32,
    pub files_total: AtomicU32,
}

impl DownloadStatus {
    /// (bytes done, bytes total, files done, files total, current file).
    pub fn snapshot(&self) -> (u64, u64, u32, u32, String) {
        (
            self.file_done.load(Ordering::Relaxed),
            self.file_total.load(Ordering::Relaxed),
            self.files_done.load(Ordering::Relaxed),
            self.files_total.load(Ordering::Relaxed),
            self.file.lock().map(|f| f.clone()).unwrap_or_default(),
        )
    }
}

/// Adapter from hf-hub's [`hf_hub::api::Progress`] callbacks to the shared
/// status block.
struct Observed(Arc<DownloadStatus>);

impl hf_hub::api::Progress for Observed {
    fn init(&mut self, size: usize, filename: &str) {
        if let Ok(mut f) = self.0.file.lock() {
            *f = filename.to_string();
        }
        self.0.file_total.store(size as u64, Ordering::Relaxed);
        self.0.file_done.store(0, Ordering::Relaxed);
    }
    fn update(&mut self, size: usize) {
        self.0.file_done.fetch_add(size as u64, Ordering::Relaxed);
    }
    fn finish(&mut self) {
        self.0.files_done.fetch_add(1, Ordering::Relaxed);
    }
}

/// Files downloaded for an HF model: the artifact + its metadata. The
/// `artifact` is the principal weights file; `siblings` are everything else
/// fetched (sharded safetensors files, ONNX external data, tokenizer/config
/// JSONs). All paths share the same `hf-hub` snapshot directory; calling
/// `snapshot_dir()` returns it.
#[derive(Debug)]
pub struct DownloadedModel {
    pub artifact: PathBuf,
    pub kind: ArtifactKind,
    pub siblings: Vec<PathBuf>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArtifactKind {
    Safetensors,
    Gguf,
    Onnx,
}

impl DownloadedModel {
    /// Snapshot directory containing the artifact and siblings.
    pub fn snapshot_dir(&self) -> Option<&std::path::Path> {
        self.artifact.parent()
    }
}

/// Files we always try to fetch alongside a model artifact. Missing entries
/// are not errors — different model families ship different metadata sets.
const SIBLING_CANDIDATES: &[&str] = &[
    "config.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "generation_config.json",
];

/// Download a model from HuggingFace Hub. Selects the canonical artifact
/// per the priority list (safetensors > GGUF > ONNX), then fetches metadata
/// siblings when present.
///
/// Returns the artifact path, kind, and the set of metadata files actually
/// downloaded. The caller can then point `mi import` at
/// `result.snapshot_dir()`.
pub fn download_model(model_id: &str) -> Result<DownloadedModel, String> {
    download_model_observed(model_id, None)
}

/// [`download_model`], with a live [`DownloadStatus`] the caller can watch.
/// Cached files return instantly (and still count toward `files_done`), so
/// the bar only moves for bytes that actually cross the network.
pub fn download_model_observed(
    model_id: &str,
    status: Option<Arc<DownloadStatus>>,
) -> Result<DownloadedModel, String> {
    let api = Api::new().map_err(|e| format!("HF API init failed: {e}"))?;
    let repo = api.model(model_id.to_string());
    // Api::new() caches under Cache::default(); mirror that so the cache
    // check below sees the same directory get() would hit.
    let cache = hf_hub::Cache::default().repo(hf_hub::Repo::model(model_id.to_string()));

    let info = repo
        .info()
        .map_err(|e| format!("repo info for {model_id} failed: {e}"))?;
    let filenames: Vec<String> = info.siblings.iter().map(|s| s.rfilename.clone()).collect();

    let (artifact_name, kind) = pick_artifact(&filenames).ok_or_else(|| {
        format!("no recognized artifact in {model_id}; tried safetensors / gguf / onnx")
    })?;

    // Plan the full fetch list up front so files_total is honest from the
    // first byte: artifact, shards, then metadata siblings.
    let mut planned: Vec<String> = vec![artifact_name.clone()];
    if artifact_name.ends_with(".safetensors.index.json") {
        planned.extend(
            filenames
                .iter()
                .filter(|f| f.ends_with(".safetensors") && f.contains("of"))
                .cloned(),
        );
    } else if artifact_name.ends_with(".onnx") {
        let data_name = format!("{artifact_name}_data");
        if filenames.iter().any(|f| f == &data_name) {
            planned.push(data_name);
        }
    }
    for name in SIBLING_CANDIDATES {
        if filenames.iter().any(|f| f == *name) {
            planned.push(name.to_string());
        }
    }
    if let Some(st) = &status {
        st.files_total.store(planned.len() as u32, Ordering::Relaxed);
    }

    // Cache hits count as done files but never move the byte bar — the bar
    // only reports bytes actually crossing the network.
    let fetch = |name: &str| -> Result<PathBuf, String> {
        if let Some(st) = &status {
            if let Some(p) = cache.get(name) {
                st.files_done.fetch_add(1, Ordering::Relaxed);
                return Ok(p);
            }
            log::info!("Downloading: {name}");
            return repo
                .download_with_progress(name, Observed(st.clone()))
                .map_err(|e| format!("download {name} from {model_id} failed: {e}"));
        }
        repo.get(name)
            .map_err(|e| format!("download {name} from {model_id} failed: {e}"))
    };

    let artifact = fetch(&artifact_name)?;

    let mut siblings: Vec<PathBuf> = Vec::new();
    for name in planned.iter().skip(1) {
        match fetch(name) {
            Ok(p) => siblings.push(p),
            // Weights are load-bearing, metadata is best-effort — same
            // split the sequential version had.
            Err(e) if name.ends_with(".safetensors") || name.ends_with("_data") => {
                return Err(e)
            }
            Err(e) => log::warn!("sibling {name} listed but download failed: {e}"),
        }
    }

    Ok(DownloadedModel {
        artifact,
        kind,
        siblings,
    })
}

/// Walk the canonical priority list against the listing, return the first
/// match. Returned String is the path-within-repo, owned (the caller used
/// it directly with `repo.get()`).
fn pick_artifact(filenames: &[String]) -> Option<(String, ArtifactKind)> {
    // Priority 1: single-file safetensors at repo root.
    if filenames.iter().any(|f| f == "model.safetensors") {
        return Some(("model.safetensors".to_string(), ArtifactKind::Safetensors));
    }
    // Priority 1b: sharded safetensors (download via index).
    if filenames
        .iter()
        .any(|f| f == "model.safetensors.index.json")
    {
        return Some((
            "model.safetensors.index.json".to_string(),
            ArtifactKind::Safetensors,
        ));
    }
    // Priority 2: GGUF — pick the first .gguf file. Quantization tag varies
    // by repo, so don't filter on that.
    if let Some(name) = filenames.iter().find(|f| f.ends_with(".gguf")) {
        return Some((name.clone(), ArtifactKind::Gguf));
    }
    // Priority 3: ONNX — first .onnx file in the repo.
    if let Some(name) = filenames.iter().find(|f| f.ends_with(".onnx")) {
        return Some((name.clone(), ArtifactKind::Onnx));
    }
    None
}
