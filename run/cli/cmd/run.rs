use crate::util::{pick_backend, resolve_model_path};
use run::arch::decoder::{config::LlamaConfig, LlamaModel};
use run::backend::Backend;
use run::format::LoadedModel;
use run::generate::{generate, ModelRunner, SampleConfig, SampleKind};
use run::ir::GraphRunner;
use run::tokenizer::{build_tokenizer, ChatMessage};
use std::time::Instant;

pub fn run(args: Vec<String>) {
    if args.is_empty() {
        eprintln!("usage: mr run <model> [--prompt TEXT] [--max-tokens N] ...");
        std::process::exit(2);
    }
    let model_arg = &args[0];
    let mut prompt = String::new();
    let mut max_tokens: usize = usize::MAX;
    let mut temperature: f32 = 0.0;
    let mut backend_name: String = "auto".into();
    let mut use_chat = true;
    let mut force_graph = false;
    let mut image_path: Option<String> = None;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--prompt"      => { i += 1; prompt = args[i].clone(); }
            "--max-tokens"  => { i += 1; max_tokens = args[i].parse().unwrap_or(usize::MAX); }
            "--temperature" => { i += 1; temperature = args[i].parse().unwrap_or(0.0); }
            "--backend"     => { i += 1; backend_name = args[i].clone(); }
            "--no-chat"     => use_chat = false,
            "--path=graph"  => force_graph = true,
            "--image"       => { i += 1; image_path = Some(args[i].clone()); }
            other => { eprintln!("unknown flag: {other}"); std::process::exit(2); }
        }
        i += 1;
    }

    let path = resolve_model_path(model_arg);
    if !path.exists() {
        eprintln!("model not found: {}", path.display());
        std::process::exit(1);
    }

    println!("Loading: {}", path.display());
    let t_load = Instant::now();
    let lm = LoadedModel::load(&path).unwrap_or_else(|e| { eprintln!("load failed: {e}"); std::process::exit(1); });
    let tok = build_tokenizer(&lm).unwrap_or_else(|e| { eprintln!("tokenizer build failed: {e}"); std::process::exit(1); });

    let backend: Box<dyn Backend> = pick_backend(&backend_name);

    // Multimodal path: needs the concrete LlamaModel (TokenOverride /
    // forward_ex aren't part of the generic ModelRunner trait), so this
    // bypasses the curated/graph dispatch below entirely and returns
    // early. See run/multimodal.rs's module doc for the placeholder-
    // placement scope note.
    if let Some(img_path) = &image_path {
        let mut m = LlamaModel::from_loaded(&lm).unwrap_or_else(|e| {
            eprintln!("--image requires the curated model path, which failed to load: {e}");
            std::process::exit(1);
        });
        if m.config.vision.is_none() {
            eprintln!("model '{}' has no vision tower (not a VL checkpoint)", path.display());
            std::process::exit(1);
        }
        if let Err(e) = m.to_backend(&*backend) {
            eprintln!("weight upload failed: {e}");
        }
        println!(
            "Loaded in {:.1}s  [{}, vision tower present]",
            t_load.elapsed().as_secs_f64(), m.config.model_type,
        );
        println!("Backend: {}", backend.kind().as_str());

        let image_bytes = std::fs::read(img_path).unwrap_or_else(|e| {
            eprintln!("failed to read image {img_path}: {e}"); std::process::exit(1);
        });
        let final_prompt = if use_chat {
            tok.apply_chat_template(&[ChatMessage { role: "user".into(), content: prompt.clone() }], true)
        } else {
            prompt.clone()
        };
        let cfg = SampleConfig {
            method: if temperature <= 0.0 { SampleKind::Greedy } else { SampleKind::TopP },
            temperature: if temperature <= 0.0 { 1.0 } else { temperature },
            top_p: 0.95,
            top_k: 40,
        };
        println!("---\n{final_prompt}\n---");
        let t_gen = Instant::now();
        match run::multimodal::generate_multimodal(&mut m, &tok, &*backend, &final_prompt, &image_bytes, max_tokens, cfg) {
            Ok((text, count)) => {
                let dt = t_gen.elapsed().as_secs_f64();
                println!("{text}\n---");
                println!("Generated {count} tokens in {dt:.1}s ({:.1} tok/s)", count as f64 / dt);
            }
            Err(e) => { eprintln!("generate_multimodal failed: {e}"); std::process::exit(1); }
        }
        return;
    }

    // Dispatch: curated (LlamaModel) → graph (GraphRunner) → error.
    // --path=graph was already parsed above.
    let (mut runner, label): (Box<dyn ModelRunner>, String) = if !force_graph {
        match LlamaModel::from_loaded(&lm) {
            Ok(mut m) => {
                let lbl = format!(
                    "{}, {} layers, hidden={}, heads={}/{}, head_dim={}, qk_norm={}",
                    m.config.model_type, m.config.num_hidden_layers,
                    m.config.hidden_size, m.config.num_attention_heads,
                    m.config.num_key_value_heads, m.config.head_dim,
                    m.config.has_qk_norm,
                );
                if let Err(e) = m.to_backend(&*backend) {
                    eprintln!("weight upload failed: {e}");
                }
                (Box::new(m), lbl)
            }
            Err(curated_err) => {
                // Curated path failed — try graph path.
                let llama_cfg = LlamaConfig::parse(&lm.file.config, &lm.tensors)
                    .unwrap_or_else(|e| { eprintln!("config parse failed: {e}"); std::process::exit(1); });
                // Use a CPU backend for the graph executor (portable, correct).
                let graph_backend = run::backend::cpu::CpuBackend::new();
                match GraphRunner::from_loaded(&lm, &llama_cfg, Box::new(graph_backend)) {
                    Some(Ok(gr)) => {
                        let lbl = format!("[graph path] {} (curated unavailable: {curated_err})", llama_cfg.model_type);
                        (Box::new(gr), lbl)
                    }
                    Some(Err(e)) => {
                        eprintln!("graph executor init failed: {e}");
                        std::process::exit(1);
                    }
                    None => {
                        eprintln!("model_type '{}' has no curated path and no graph template", llama_cfg.model_type);
                        eprintln!("  curated error: {curated_err}");
                        eprintln!("  hint: run `mr graph {}` to embed a graph section", path.display());
                        std::process::exit(1);
                    }
                }
            }
        }
    } else {
        // Explicit --path=graph: force graph executor.
        let llama_cfg = LlamaConfig::parse(&lm.file.config, &lm.tensors)
            .unwrap_or_else(|e| { eprintln!("config parse failed: {e}"); std::process::exit(1); });
        let graph_backend = run::backend::cpu::CpuBackend::new();
        match GraphRunner::from_loaded(&lm, &llama_cfg, Box::new(graph_backend)) {
            Some(Ok(gr)) => {
                let lbl = format!("[graph path forced] {}", llama_cfg.model_type);
                (Box::new(gr), lbl)
            }
            Some(Err(e)) => { eprintln!("graph executor init failed: {e}"); std::process::exit(1); }
            None => { eprintln!("no graph template for model_type '{}'", llama_cfg.model_type); std::process::exit(1); }
        }
    };

    println!(
        "Loaded in {:.1}s  [{}]",
        t_load.elapsed().as_secs_f64(),
        label,
    );
    println!("Backend: {}", backend.kind().as_str());

    let final_prompt = if use_chat {
        tok.apply_chat_template(&[ChatMessage { role: "user".into(), content: prompt.clone() }], true)
    } else {
        prompt.clone()
    };

    let cfg = SampleConfig {
        method: if temperature <= 0.0 { SampleKind::Greedy } else { SampleKind::TopP },
        temperature: if temperature <= 0.0 { 1.0 } else { temperature },
        top_p: 0.95,
        top_k: 40,
    };

    println!("---");
    println!("{final_prompt}");
    println!("---");

    let t_gen = Instant::now();
    match generate(runner.as_mut(), &tok, &*backend, &final_prompt, max_tokens, cfg) {
        Ok((text, count)) => {
            let dt = t_gen.elapsed().as_secs_f64();
            println!("{text}");
            println!("---");
            println!("Generated {count} tokens in {dt:.1}s ({:.1} tok/s)", count as f64 / dt);
        }
        Err(e) => { eprintln!("generate failed: {e}"); std::process::exit(1); }
    }
}
