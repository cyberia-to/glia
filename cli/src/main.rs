//! glia — one door to the model layer.
//!
//!   glia import …   import an ML model (HF directory → .model)
//!   glia run    …   run a model (portable, spec-driven runtime)
//!
//! `glia <cmd> [args…]` forwards to the command's binary, built alongside this
//! one (`import` → `mi`, `run` → `mr`).

use std::io::IsTerminal;
use std::process::{exit, Command};

/// (subcommand, built binary name, description)
const SUBS: &[(&str, &str, &str)] = &[
    ("import", "mi", "import an ML model (HF directory → .model)"),
    ("run", "mr", "run a model — portable, spec-driven runtime"),
];

fn tty() -> bool {
    std::io::stdout().is_terminal()
}
fn paint(code: &str, s: &str) -> String {
    if tty() { format!("\x1b[{code}m{s}\x1b[0m") } else { s.to_string() }
}
fn dim(s: &str) -> String {
    paint("90", s)
}
fn bold(s: &str) -> String {
    paint("1", s)
}

const LOGO: &str = "\
\x1b[31m ██████╗ ██╗     ██╗ █████╗ \x1b[0m
\x1b[33m██╔════╝ ██║     ██║██╔══██╗\x1b[0m
\x1b[32m██║  ███╗██║     ██║███████║\x1b[0m
\x1b[36m██║   ██║██║     ██║██╔══██║\x1b[0m
\x1b[34m╚██████╔╝███████╗██║██║  ██║\x1b[0m
\x1b[35m ╚═════╝ ╚══════╝╚═╝╚═╝  ╚═╝\x1b[0m";

fn help() {
    if tty() {
        println!("{LOGO}");
        println!("{}\n", paint("37", "    the model layer"));
    }
    println!("{}", dim("usage: glia <command> [args…]"));
    for (name, _, desc) in SUBS {
        println!("  {}  {}", bold(&format!("{name:<7}")), dim(desc));
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let sub = args.first().map(String::as_str).unwrap_or("");

    if matches!(sub, "" | "help" | "--help" | "-h") {
        help();
        return;
    }
    let Some((_, binary, _)) = SUBS.iter().find(|(name, _, _)| *name == sub) else {
        eprintln!("glia: unknown command '{sub}' — try `glia help`");
        exit(2);
    };

    // The command's binary sits next to us. Canonicalize first so a
    // `~/.cargo/bin` symlink resolves back to the real build directory.
    let dir = std::env::current_exe()
        .ok()
        .and_then(|e| std::fs::canonicalize(&e).ok().or(Some(e)))
        .and_then(|e| e.parent().map(|p| p.to_path_buf()))
        .unwrap_or_default();
    let target = dir.join(binary);
    if !target.exists() {
        eprintln!("glia: {sub} is not built — run `cy install glia`");
        exit(1);
    }
    match Command::new(&target).args(&args[1..]).status() {
        Ok(status) => exit(status.code().unwrap_or(0)),
        Err(e) => {
            eprintln!("glia: cannot run {sub}: {e}");
            exit(1);
        }
    }
}
