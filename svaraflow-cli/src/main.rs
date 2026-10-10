//! Command-line front end for SvaraFlow.
//!
//! Reuses the same crates as the Tauri app and the JSON-RPC server.
//! Argument parsing is hand-rolled for the scaffold; `clap` will be
//! introduced via `cargo add clap --features derive` once the first
//! real subcommand lands.

use anyhow::{bail, Result};

/// Binary name as declared by Cargo.
const BIN_NAME: &str = env!("CARGO_PKG_NAME");

/// Binary version as declared by Cargo.
const BIN_VERSION: &str = env!("CARGO_PKG_VERSION");

/// Usage text printed by `--help`, `-h`, and when no argument is given.
const USAGE: &str = "\
SvaraFlow CLI

USAGE:
    svaraflow-cli <OPTION>

OPTIONS:
    --version, -V    Print the version and exit.
    --help, -h       Print this help and exit.
";

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    run(&args)
}

/// Runs the CLI against the given arguments (excluding `argv[0]`).
///
/// Split out from `main` so it can be exercised directly by tests
/// without spawning a process.
fn run(args: &[String]) -> Result<()> {
    match args.first().map(String::as_str) {
        Some("--version" | "-V") => {
            println!("{BIN_NAME} {BIN_VERSION}");
            Ok(())
        }
        Some("--help" | "-h") | None => {
            println!("{USAGE}");
            Ok(())
        }
        Some(other) => bail!("unknown argument: {other}\n\n{USAGE}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_flag_is_accepted() {
        assert!(run(&["--version".to_string()]).is_ok());
    }

    #[test]
    fn short_version_flag_is_accepted() {
        assert!(run(&["-V".to_string()]).is_ok());
    }

    #[test]
    fn help_flag_is_accepted() {
        assert!(run(&["--help".to_string()]).is_ok());
    }

    #[test]
    fn no_args_prints_help() {
        assert!(run(&[]).is_ok());
    }

    #[test]
    fn unknown_flag_is_rejected() {
        assert!(run(&["--wat".to_string()]).is_err());
    }
}
