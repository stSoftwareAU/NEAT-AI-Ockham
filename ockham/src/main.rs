//! `neat_ai_ockham` command-line interface.
//!
//! Issue #1: start, parse flags, and report configuration. Pruning lands in
//! later issues; this binary must not attempt optimisation yet.

use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Duration;

use clap::Parser;
use neat_ai_ockham::config::{
    DEFAULT_CANDIDATE_COUNT, DEFAULT_MAX_CONSECUTIVE_SCORER_FAILURES, DEFAULT_MIN_IMPROVEMENT,
    DEFAULT_SCREEN_SAMPLE_RATE, DEFAULT_SCREEN_THRESHOLD, DEFAULT_TIMEOUT_SECONDS, OckhamConfig,
};
use neat_ai_ockham::{crate_version, log};

#[derive(Parser, Debug)]
#[command(name = "neat_ai_ockham")]
#[command(about = "Experimental pruning optimiser for already-fit NEAT-AI creatures")]
#[command(version)]
struct Cli {
    /// Source creature JSON (never modified).
    creature: Option<PathBuf>,
    /// Training corpus directory of `.bin` files.
    training_data: Option<PathBuf>,

    /// Output directory for `best.json`, `experiments.jsonl`, `winners/`, `workspace/`.
    #[arg(long, default_value = ".")]
    output_dir: PathBuf,
    /// NEAT-AI-scorer binary.
    #[arg(long, default_value = "rust_scorer")]
    scorer: PathBuf,
    /// Extra argument passed verbatim to the scorer (repeatable).
    #[arg(long = "scorer-arg")]
    scorer_args: Vec<String>,
    /// Wall-clock budget in seconds.
    #[arg(long, default_value_t = DEFAULT_TIMEOUT_SECONDS)]
    timeout_seconds: u64,
    /// Stop after this many experiments (in addition to the wall-clock budget).
    #[arg(long)]
    max_experiments: Option<u64>,
    /// RNG seed (drawn from OS entropy when absent; printed for replay).
    #[arg(long)]
    seed: Option<u64>,
    /// Candidate creatures generated per sampled sweep batch.
    #[arg(long, default_value_t = DEFAULT_CANDIDATE_COUNT)]
    candidates: usize,
    /// Screen sample rate in (0,1); 0 disables the screen.
    #[arg(long, default_value_t = DEFAULT_SCREEN_SAMPLE_RATE)]
    screen_sample_rate: f64,
    /// Sampled Δscore a candidate must exceed to be promoted.
    #[arg(long, default_value_t = DEFAULT_SCREEN_THRESHOLD)]
    screen_threshold: f64,
    /// Strict minimum authoritative score improvement.
    #[arg(long, default_value_t = DEFAULT_MIN_IMPROVEMENT)]
    min_improvement: f64,
    /// Consecutive scorer failures tolerated before stopping.
    #[arg(long, default_value_t = DEFAULT_MAX_CONSECUTIVE_SCORER_FAILURES)]
    max_consecutive_scorer_failures: u32,
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    let (Some(creature), Some(training_data)) = (cli.creature.clone(), cli.training_data.clone())
    else {
        eprintln!(
            "usage: neat_ai_ockham <creature.json> <training-data-dir> [OPTIONS]\n       neat_ai_ockham --help"
        );
        return ExitCode::FAILURE;
    };

    let config = OckhamConfig {
        creature,
        training_data,
        output_dir: cli.output_dir,
        scorer_path: cli.scorer,
        scorer_args: cli.scorer_args,
        timeout: Duration::from_secs(cli.timeout_seconds),
        max_experiments: cli.max_experiments,
        seed: cli.seed,
        candidates: cli.candidates,
        screen_sample_rate: if cli.screen_sample_rate == 0.0 {
            None
        } else {
            Some(cli.screen_sample_rate)
        },
        screen_threshold: cli.screen_threshold,
        min_improvement: cli.min_improvement,
        max_consecutive_scorer_failures: cli.max_consecutive_scorer_failures,
    };
    if let Err(e) = config.validate() {
        eprintln!("{e}");
        return ExitCode::from(2);
    }

    log::info(&format!(
        "NEAT-AI-Ockham {} — configuration only; pruning is not attempted yet",
        crate_version()
    ));
    log::detail(&format!(
        "timeout {}s, candidates {}, screen {:?}",
        config.timeout.as_secs(),
        config.candidates,
        config.screen_sample_rate
    ));
    match serde_json::to_string_pretty(&config.report()) {
        Ok(json) => {
            println!("{json}");
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("cannot serialise configuration: {e}");
            ExitCode::FAILURE
        }
    }
}
