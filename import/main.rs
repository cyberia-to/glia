//! mi — model importer CLI (binary entry point of the `import` crate).
//!
//! HuggingFace directory (safetensors + tokenizer.json + config.json)
//! or GGUF → cyb `.model`. Runtime is in `run/`.

use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "mi")]
#[command(about = "Model importer — HF / GGUF / safetensors → cyb .model")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Import a directory containing GGUF + tokenizer.json + config.json
    /// into a `.model` file under `~/llm/`.
    Import {
        /// Path to the source directory.
        dir: String,
    },

    /// List models cached under ~/.cache/huggingface/hub.
    List,

    /// Download a model from HuggingFace (best-effort ONNX/safetensors discovery).
    Download {
        /// HF repo id (e.g. "Qwen/Qwen3-0.6B").
        model: String,
    },
}

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    let cli = Cli::parse();
    match cli.command {
        Commands::Import { dir } => run_import(&dir),
        Commands::List => run_list(),
        Commands::Download { model } => run_download(&model),
    }
}

fn run_list() {
    println!("Cached HF models:");
    let cache_dir = std::env::var("HOME")
        .map(|h| std::path::PathBuf::from(h).join(".cache/huggingface/hub"))
        .unwrap_or_default();
    if !cache_dir.exists() {
        println!("  (none)");
        return;
    }
    if let Ok(entries) = std::fs::read_dir(&cache_dir) {
        for entry in entries.flatten() {
            if entry.file_type().map(|t| t.is_dir()).unwrap_or(false) {
                if let Some(s) = entry.file_name().to_str() {
                    if s.starts_with("models--") {
                        println!("  {}", s.replace("models--", "").replace("--", "/"));
                    }
                }
            }
        }
    }
}

fn run_download(model_id: &str) {
    println!("Downloading {model_id}...");
    match import::hf::download_model(model_id) {
        Ok(downloaded) => {
            println!(
                "Artifact ({:?}): {}",
                downloaded.kind,
                downloaded.artifact.display()
            );
            for s in &downloaded.siblings {
                println!("  sibling: {}", s.display());
            }
            if let Some(dir) = downloaded.snapshot_dir() {
                println!("\nSnapshot dir: {}", dir.display());
                println!("Run: mi import {}", dir.display());
            }
        }
        Err(e) => eprintln!("Error: {e}"),
    }
}

/// Import: source dir with weights artifact + tokenizer.json + config.json → `.model`.
fn run_import(dir_path: &str) {
    let dir = std::path::Path::new(dir_path);
    let name = dir.file_name().and_then(|n| n.to_str()).unwrap_or("model");
    // Keep the historical .canonical suffix for CLI imports, so nothing that
    // greps for it breaks; the library takes whatever name it is given.
    let output_name = format!("{}.canonical", name.strip_suffix("-import").unwrap_or(name));
    match import::pipeline::import_snapshot(dir_path, &output_name) {
        Ok(p) => println!("Imported: {}", p.display()),
        Err(e) => {
            eprintln!("FAIL: {e}");
            std::process::exit(1);
        }
    }
}
