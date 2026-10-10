//! SvaraFlow command-line interface.

use anyhow::Result;
use clap::Parser;

mod cli;
mod presets;
mod silence_cmd;
mod timeline_cmd;
mod transcribe_cmd;

use cli::{Cli, Command};

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Timeline(args) => timeline_cmd::run(args),
        Command::Silence(args) => silence_cmd::run(args),
        Command::Transcribe(args) => transcribe_cmd::run_transcribe(args),
        Command::Models(args) => transcribe_cmd::run_models(args),
    }
}