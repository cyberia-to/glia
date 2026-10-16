//! Safetensors loader — parse JSON header + mmap tensor data
//!
//! Format:
//! [8 bytes: header_size as u64 LE]
//! [header_size bytes: JSON metadata]
//! [rest: raw tensor data]
//!
//! JSON metadata: { "tensor_name": { "dtype": "F16", "shape": [4096, 4096], "data_offsets": [start, end] }, ... }

use std::collections::HashMap;
use std::path::Path;

use crate::types::{DType, Weight, Weights};

/// Safetensors tensor descriptor (from JSON header)
#[derive(Debug, serde::Deserialize)]
struct TensorDescriptor {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: [u64; 2],
}

/// Load a safetensors file (or multi-shard) into Graph IR
pub fn load_safetensors(path: &Path) -> Result<Weights, String> {
    // (preserved signature)
    // Check for multi-shard: if model.safetensors.index.json exists, load all shards
    if let Some(dir) = path.parent() {
        let index_path = dir.join("model.safetensors.index.json");
        if index_path.exists() {
            return load_safetensors_sharded(dir, &index_path);
        }
    }

    load_safetensors_single(path)
}

/// Load all shards from a safetensors index
fn load_safetensors_sharded(dir: &Path, index_path: &Path) -> Result<Weights, String> {
    let index_str = std::fs::read_to_string(index_path)
        .map_err(|e| format!("Cannot read index: {e}"))?;
    let index: serde_json::Value = serde_json::from_str(&index_str)
        .map_err(|e| format!("Invalid index JSON: {e}"))?;

    let weight_map = index.get("weight_map")
        .and_then(|v| v.as_object())
        .ok_or("No weight_map in index")?;

    // Build tensor→shard mapping
    let mut tensor_to_shard: HashMap<String, String> = HashMap::new();
    for (tensor_name, shard) in weight_map {
        if let Some(s) = shard.as_str() {
            tensor_to_shard.insert(tensor_name.clone(), s.to_string());
        }
    }

    // Collect unique shard files (sorted)
    let mut shard_files: Vec<String> = weight_map.values()
        .filter_map(|v| v.as_str().map(String::from))
        .collect::<std::collections::HashSet<_>>()
        .into_iter().collect();
    shard_files.sort();

    // mmap all shards and parse headers
    struct ShardInfo {
        mmap: memmap2::Mmap,
        data_start: usize,
        descriptors: serde_json::Map<String, serde_json::Value>,
    }
    let mut shards: HashMap<String, ShardInfo> = HashMap::new();

    for shard_file in &shard_files {
        let shard_path = dir.join(shard_file);
        if !shard_path.exists() {
            log::warn!("Shard {} not found, skipping", shard_file);
            continue;
        }
        let file = std::fs::File::open(&shard_path)
            .map_err(|e| format!("Cannot open {}: {e}", shard_path.display()))?;
        let mmap = unsafe { memmap2::Mmap::map(&file)
            .map_err(|e| format!("Cannot mmap {}: {e}", shard_path.display()))? };

        if mmap.len() < 8 { continue; }
        let header_size = u64::from_le_bytes([
            mmap[0], mmap[1], mmap[2], mmap[3], mmap[4], mmap[5], mmap[6], mmap[7],
        ]) as usize;
        let data_start = 8 + header_size;
        if data_start > mmap.len() { continue; }

        let header_str = std::str::from_utf8(&mmap[8..8 + header_size])
            .map_err(|e| format!("Invalid header UTF-8 in {}: {e}", shard_file))?;
        let descriptors: serde_json::Map<String, serde_json::Value> = serde_json::from_str(header_str)
            .map_err(|e| format!("Invalid header JSON in {}: {e}", shard_file))?;

        shards.insert(shard_file.clone(), ShardInfo { mmap, data_start, descriptors });
    }

    // For each shard: build global→local offset map, then load tensors
    let mut weights = Weights::new();
    for shard_file in &shard_files {
        let shard = match shards.get(shard_file) {
            Some(s) => s,
            None => continue,
        };

        // Get tensors belonging to this shard
        let shard_tensor_names: Vec<&String> = tensor_to_shard.iter()
            .filter(|(_, sf)| sf.as_str() == shard_file.as_str())
            .map(|(tn, _)| tn)
            .collect();

        // Sort by global offset to compute local positions
        let mut tensor_offsets: Vec<(&String, u64, u64)> = shard_tensor_names.iter()
            .filter_map(|tn| {
                let desc = shard.descriptors.get(tn.as_str())?;
                let d: TensorDescriptor = serde_json::from_value(desc.clone()).ok()?;
                Some((*tn, d.data_offsets[0], d.data_offsets[1] - d.data_offsets[0]))
            })
            .collect();
        tensor_offsets.sort_by_key(|(_, offset, _)| *offset);

        // Build global→local map
        let mut global_to_local: HashMap<u64, u64> = HashMap::new();
        let mut local_pos: u64 = 0;
        for &(_, global_start, size) in &tensor_offsets {
            global_to_local.insert(global_start, local_pos);
            local_pos += size;
        }

        // Load each tensor using local offset
        let mut count = 0;
        for tensor_name in &shard_tensor_names {
            let desc_value = match shard.descriptors.get(tensor_name.as_str()) {
                Some(v) => v,
                None => continue,
            };
            let desc: TensorDescriptor = match serde_json::from_value(desc_value.clone()) {
                Ok(d) => d,
                Err(_) => continue,
            };

            let dtype = safetensors_dtype(&desc.dtype);
            let size = (desc.data_offsets[1] - desc.data_offsets[0]) as usize;
            let local = *global_to_local.get(&desc.data_offsets[0]).unwrap_or(&desc.data_offsets[0]) as usize;
            let byte_start = shard.data_start + local;
            let byte_end = byte_start + size;

            if byte_end > shard.mmap.len() {
                log::warn!("Tensor {} OOB in {}: local={} size={} end={} > {}",
                    tensor_name, shard_file, local, size, byte_end, shard.mmap.len());
                continue;
            }

            let raw_data = shard.mmap[byte_start..byte_end].to_vec();
            weights.insert(
                tensor_name.to_string(),
                Weight {
                    data: raw_data,
                    shape: desc.shape,
                    dtype,
                    needs_transpose: false,
                },
            );
            count += 1;
        }
        log::info!(
            "Shard {}: {}/{} tensors loaded",
            shard_file,
            count,
            shard_tensor_names.len()
        );
    }

    log::info!(
        "Safetensors sharded: {} weights from {} shards in {}",
        weights.len(),
        shards.len(),
        dir.display()
    );
    Ok(weights)
}

/// Load a single safetensors file into Graph IR
fn load_safetensors_single(path: &Path) -> Result<Weights, String> {
    let file = std::fs::File::open(path)
        .map_err(|e| format!("Cannot open {}: {e}", path.display()))?;
    let mmap = unsafe {
        memmap2::Mmap::map(&file)
            .map_err(|e| format!("Cannot mmap {}: {e}", path.display()))?
    };

    if mmap.len() < 8 {
        return Err("File too small for safetensors".to_string());
    }

    // Read header size (first 8 bytes, u64 LE)
    let header_size = u64::from_le_bytes([
        mmap[0], mmap[1], mmap[2], mmap[3],
        mmap[4], mmap[5], mmap[6], mmap[7],
    ]) as usize;

    if 8 + header_size > mmap.len() {
        return Err(format!(
            "Header size {header_size} exceeds file size {}",
            mmap.len()
        ));
    }

    // Parse JSON header
    let header_bytes = &mmap[8..8 + header_size];
    let header_str = std::str::from_utf8(header_bytes)
        .map_err(|e| format!("Invalid UTF-8 in safetensors header: {e}"))?;

    let descriptors: HashMap<String, serde_json::Value> = serde_json::from_str(header_str)
        .map_err(|e| format!("Invalid JSON in safetensors header: {e}"))?;

    let data_start = 8 + header_size;


    let mut weights = Weights::new();
    let mut loaded = 0;

    for (name, value) in &descriptors {
        if name == "__metadata__" {
            continue;
        }

        let desc: TensorDescriptor = serde_json::from_value(value.clone())
            .map_err(|e| format!("Invalid tensor descriptor for {name}: {e}"))?;

        let dtype = safetensors_dtype(&desc.dtype);
        let [offset_start, offset_end] = desc.data_offsets;
        let byte_start = data_start + offset_start as usize;
        let byte_end = data_start + offset_end as usize;

        if byte_end > mmap.len() {
            log::warn!("Tensor {name} OOB: {byte_end} > {}, skipping", mmap.len());
            continue;
        }

        weights.insert(
            name.clone(),
            Weight {
                data: mmap[byte_start..byte_end].to_vec(),
                shape: desc.shape,
                dtype,
                needs_transpose: false,
            },
        );
        loaded += 1;
    }

    log::info!(
        "Safetensors loaded: {} tensors from {}",
        loaded,
        path.display()
    );

    // Try to load config.json from same directory to detect architecture
    if let Some(dir) = path.parent() {
        let config_path = dir.join("config.json");
        if config_path.exists() {
            if let Ok(config_str) = std::fs::read_to_string(&config_path) {
                if let Ok(config) = serde_json::from_str::<serde_json::Value>(&config_str) {
                    if let Some(archs) = config.get("architectures").and_then(|v| v.as_array()) {
                        let arch_names: Vec<String> = archs
                            .iter()
                            .filter_map(|v| v.as_str().map(String::from))
                            .collect();
                        log::info!("Detected architecture: {:?}", arch_names);
                    }
                    if let Some(hidden) = config.get("hidden_size").and_then(|v| v.as_u64()) {
                        log::info!("Hidden size: {hidden}");
                    }
                    if let Some(layers) = config
                        .get("num_hidden_layers")
                        .and_then(|v| v.as_u64())
                    {
                        log::info!("Num layers: {layers}");
                    }
                }
            }
        }
    }

    Ok(weights)
}

// ── lazy reader (streaming import path) ─────────────────────────────────────
//
// The eager loaders above build one `Weights` holding every tensor's bytes
// copied out of the mmap — fine at 8B (a few GB), but a 27B model's ~55 GB
// of source bytes plus the packer's own growing output made `import_as`
// peak at 54 GB resident and wedge on a 48 GB machine. The mmaps themselves
// cost nothing until touched; only the eager `.to_vec()` per tensor forces
// the whole model into the heap at once. This type keeps the mmaps open and
// defers that copy to one tensor at a time, so the pipeline can pack-and-
// write-and-drop each tensor before touching the next.

/// One shard's mmap plus where its tensor data begins.
struct LazyShard {
    mmap: memmap2::Mmap,
}

/// Where one tensor lives: which shard, what it is, and its byte range
/// within that shard's mmap (already resolved to local offsets for the
/// multi-shard case, so `read` never has to know about global vs. local).
struct TensorLocation {
    shard: String,
    dtype: DType,
    shape: Vec<usize>,
    byte_start: usize,
    byte_end: usize,
}

/// A safetensors model (single-file or sharded), indexed but not loaded.
pub struct LazySafetensors {
    shards: HashMap<String, LazyShard>,
    index: HashMap<String, TensorLocation>,
}

impl LazySafetensors {
    /// Every tensor name this model has — cheap, no data touched.
    pub fn names(&self) -> Vec<String> {
        self.index.keys().cloned().collect()
    }

    /// dtype + shape without copying the tensor's bytes.
    pub fn dtype_shape(&self, name: &str) -> Option<(DType, Vec<usize>)> {
        self.index.get(name).map(|t| (t.dtype, t.shape.clone()))
    }

    /// Copy this one tensor's bytes out of its shard's mmap. The caller
    /// owns the result and should let it drop once packed — that's the
    /// whole point.
    pub fn read(&self, name: &str) -> Option<Weight> {
        let loc = self.index.get(name)?;
        let shard = self.shards.get(&loc.shard)?;
        if loc.byte_end > shard.mmap.len() {
            log::warn!(
                "Tensor {name} OOB in {}: {} > {}",
                loc.shard, loc.byte_end, shard.mmap.len()
            );
            return None;
        }
        Some(Weight {
            data: shard.mmap[loc.byte_start..loc.byte_end].to_vec(),
            shape: loc.shape.clone(),
            dtype: loc.dtype,
            needs_transpose: false,
        })
    }
}

/// Index a safetensors model (single file or sharded index) without
/// reading any tensor data. Mirrors [`load_safetensors`]'s dispatch.
pub fn open_lazy(path: &Path) -> Result<LazySafetensors, String> {
    if let Some(dir) = path.parent() {
        let index_path = dir.join("model.safetensors.index.json");
        if index_path.exists() {
            return open_lazy_sharded(dir, &index_path);
        }
    }
    open_lazy_single(path)
}

fn parse_shard_header(path: &Path) -> Result<(memmap2::Mmap, usize, HashMap<String, TensorDescriptor>), String> {
    let file = std::fs::File::open(path)
        .map_err(|e| format!("Cannot open {}: {e}", path.display()))?;
    let mmap = unsafe {
        memmap2::Mmap::map(&file).map_err(|e| format!("Cannot mmap {}: {e}", path.display()))?
    };
    if mmap.len() < 8 {
        return Err(format!("{} too small for safetensors", path.display()));
    }
    let header_size = u64::from_le_bytes([
        mmap[0], mmap[1], mmap[2], mmap[3], mmap[4], mmap[5], mmap[6], mmap[7],
    ]) as usize;
    if 8 + header_size > mmap.len() {
        return Err(format!(
            "Header size {header_size} exceeds file size {} in {}",
            mmap.len(), path.display()
        ));
    }
    let header_str = std::str::from_utf8(&mmap[8..8 + header_size])
        .map_err(|e| format!("Invalid UTF-8 in {}'s header: {e}", path.display()))?;
    let raw: HashMap<String, serde_json::Value> = serde_json::from_str(header_str)
        .map_err(|e| format!("Invalid JSON in {}'s header: {e}", path.display()))?;
    let mut descriptors = HashMap::new();
    for (name, value) in raw {
        if name == "__metadata__" {
            continue;
        }
        let desc: TensorDescriptor = serde_json::from_value(value)
            .map_err(|e| format!("Invalid tensor descriptor for {name}: {e}"))?;
        descriptors.insert(name, desc);
    }
    let data_start = 8 + header_size;
    Ok((mmap, data_start, descriptors))
}

fn open_lazy_single(path: &Path) -> Result<LazySafetensors, String> {
    let (mmap, data_start, descriptors) = parse_shard_header(path)?;
    let shard_key = path.display().to_string();
    let mut index = HashMap::new();
    for (name, desc) in descriptors {
        let [start, end] = desc.data_offsets;
        index.insert(
            name,
            TensorLocation {
                shard: shard_key.clone(),
                dtype: safetensors_dtype(&desc.dtype),
                shape: desc.shape,
                byte_start: data_start + start as usize,
                byte_end: data_start + end as usize,
            },
        );
    }
    let mut shards = HashMap::new();
    shards.insert(shard_key, LazyShard { mmap });
    log::info!("Safetensors indexed (lazy): {} tensors from {}", index.len(), path.display());
    Ok(LazySafetensors { shards, index })
}

fn open_lazy_sharded(dir: &Path, index_path: &Path) -> Result<LazySafetensors, String> {
    let index_str = std::fs::read_to_string(index_path)
        .map_err(|e| format!("Cannot read index: {e}"))?;
    let index_json: serde_json::Value = serde_json::from_str(&index_str)
        .map_err(|e| format!("Invalid index JSON: {e}"))?;
    let weight_map = index_json
        .get("weight_map")
        .and_then(|v| v.as_object())
        .ok_or("No weight_map in index")?;

    let mut tensor_to_shard: HashMap<String, String> = HashMap::new();
    for (tensor_name, shard) in weight_map {
        if let Some(s) = shard.as_str() {
            tensor_to_shard.insert(tensor_name.clone(), s.to_string());
        }
    }
    let mut shard_files: Vec<String> = tensor_to_shard.values().cloned().collect::<std::collections::HashSet<_>>().into_iter().collect();
    shard_files.sort();

    let mut shards = HashMap::new();
    let mut index = HashMap::new();
    for shard_file in &shard_files {
        let (mmap, data_start, descriptors) = parse_shard_header(&dir.join(shard_file))?;

        // Same local-offset resolution as the eager sharded loader: sort
        // this shard's tensors by their (shard-local) source offset, then
        // lay them out back to back from data_start.
        let mut ordered: Vec<(&String, &TensorDescriptor)> = descriptors
            .iter()
            .filter(|(n, _)| tensor_to_shard.get(*n).map(|s| s.as_str()) == Some(shard_file.as_str()))
            .collect();
        ordered.sort_by_key(|(_, d)| d.data_offsets[0]);

        let mut local_pos = 0usize;
        for (name, desc) in ordered {
            let size = (desc.data_offsets[1] - desc.data_offsets[0]) as usize;
            let byte_start = data_start + local_pos;
            let byte_end = byte_start + size;
            index.insert(
                name.clone(),
                TensorLocation {
                    shard: shard_file.clone(),
                    dtype: safetensors_dtype(&desc.dtype),
                    shape: desc.shape.clone(),
                    byte_start,
                    byte_end,
                },
            );
            local_pos += size;
        }
        shards.insert(shard_file.clone(), LazyShard { mmap });
    }
    log::info!(
        "Safetensors indexed (lazy): {} tensors from {} shards in {}",
        index.len(), shards.len(), dir.display()
    );
    Ok(LazySafetensors { shards, index })
}

/// Convert safetensors dtype string to DType
fn safetensors_dtype(s: &str) -> DType {
    match s {
        "F32" => DType::F32,
        "F16" => DType::F16,
        "BF16" => DType::BF16,
        "I8" => DType::I8,
        "U8" => DType::U8,
        _ => {
            log::warn!("Unknown safetensors dtype: {s}, defaulting to F32");
            DType::F32
        }
    }
}
