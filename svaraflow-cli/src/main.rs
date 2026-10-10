//! SvaraFlow command-line interface.
//!
//! Parses arguments with `clap` and dispatches to one of the subcommand
//! handlers. All business logic lives in `svaraflow-core`; this binary only
//! wires arguments to core calls and formats output.

use anyhow::Result;
use clap::Parser;

mod cli;
mod silence_cmd;
mod timeline_cmd;

use cli::{Cli, Command};

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Timeline(args) => timeline_cmd::run(args),
        Command::Silence(args) => silence_cmd::run(args),
    }
}
