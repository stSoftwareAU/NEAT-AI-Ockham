//! Opt-in, manifest-pinned connection-budget research. See docs/connection-budgets.md.

#[path = "connection_budget/allocation.rs"]
mod allocation;
#[path = "connection_budget/manifest.rs"]
mod manifest;
#[path = "connection_budget/measurement.rs"]
mod measurement;
#[path = "connection_budget/pilot.rs"]
mod pilot;
#[path = "connection_budget/runner.rs"]
mod runner;
#[cfg(test)]
#[path = "connection_budget/tests.rs"]
mod tests;

use std::path::PathBuf;

use clap::{Parser, Subcommand};

#[derive(Parser)]
struct Args {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Write application-agnostic synthetic pilot inputs and a study specification.
    Pilot { directory: PathBuf, scorer: PathBuf },
    /// Write the layered multi-cut follow-up with shared per-removal screening/promotion.
    MatchedPilot { directory: PathBuf, scorer: PathBuf },
    /// Validate and pin inputs and the executable before any validation scoring.
    Freeze { spec: PathBuf, manifest: PathBuf },
    /// Execute a frozen manifest into a new output directory.
    Run { manifest: PathBuf, out: PathBuf },
}

fn main() {
    let result = match Args::parse().command {
        Command::Pilot { directory, scorer } => pilot::write(&directory, &scorer),
        Command::MatchedPilot { directory, scorer } => pilot::write_matched(&directory, &scorer),
        Command::Freeze { spec, manifest } => manifest::freeze(&spec, &manifest),
        Command::Run { manifest, out } => runner::run(&manifest, &out),
    };
    if let Err(error) = result {
        eprintln!("connection-budget experiment failed: {error}");
        std::process::exit(1);
    }
}
