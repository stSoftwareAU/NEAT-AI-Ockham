//! `neat_ai_ockham` command-line interface.
//!
//! Issue #2: load the immutable incumbent, copy it, and score a full-corpus
//! baseline. Pruning lands in later issues; this binary must not prune yet.

use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Duration;

use clap::{Parser, Subcommand};
use neat_ai_ockham::config::{
    DEFAULT_CANDIDATE_COUNT, DEFAULT_MAX_CONSECUTIVE_SCORER_FAILURES, DEFAULT_MIN_IMPROVEMENT,
    DEFAULT_ORDERING, DEFAULT_ORDERING_RANDOM_QUOTA, DEFAULT_SCREEN_SAMPLE_RATE,
    DEFAULT_SCREEN_THRESHOLD, DEFAULT_TIMEOUT_SECONDS, OckhamConfig,
};
use neat_ai_ockham::stats::DEFAULT_SAMPLE_RECORDS;
use neat_ai_ockham::{ExternalScorer, Ordering, establish_run, log};

#[derive(Parser, Debug)]
#[command(name = "neat_ai_ockham")]
#[command(about = "Experimental pruning optimiser for already-fit NEAT-AI creatures")]
#[command(version)]
#[command(subcommand_negates_reqs = true)]
struct Cli {
    #[command(subcommand)]
    command: Option<Command>,
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
    /// Latest global champion JSON for the population re-entry comparison.
    #[arg(long)]
    global_champion: Option<PathBuf>,
    /// Cap sampled winners sent to full scoring (highest sample Δ first). Omit to full-score every sampled winner.
    #[arg(long)]
    max_full: Option<usize>,
    /// Stop after this many **new** authoritative local accepts so a win can be checked in quickly.
    /// Replay of known wins from `--learnings-dir` is not counted against this cap.
    #[arg(long)]
    max_accepts: Option<u64>,
    /// Shared full-corpus prune-verdict cache. Omitted: do not read or write learnings.
    #[arg(long)]
    learnings_dir: Option<PathBuf>,
    /// Host name for the per-host jsonl file (default: unqualified hostname).
    #[arg(long)]
    learnings_host: Option<String>,
    /// Max known-win UUIDs to replay on the incumbent before the random sweep; 0 = all still present.
    #[arg(long, default_value_t = 0)]
    learnings_replay: usize,
    /// Candidate ordering strategy. Ranking only changes which neuron is tested
    /// sooner; every candidate still faces the sampled screen and full scorer.
    #[arg(long, default_value_t = DEFAULT_ORDERING, value_parser = Ordering::parse)]
    ordering: Ordering,
    /// Fraction of sweep slots reserved for the random control, in [0, 1).
    #[arg(long, default_value_t = DEFAULT_ORDERING_RANDOM_QUOTA)]
    ordering_random_quota: f64,
    /// Records sampled for hidden-neuron activation statistics; 0 scans the whole corpus.
    #[arg(long, default_value_t = DEFAULT_SAMPLE_RECORDS)]
    stats_sample_records: u64,
    /// Screen never-checked neurons before re-screening the stalest ones.
    /// Defaults to on with `--learnings-dir` and off without it.
    #[arg(
        long,
        num_args = 0..=1,
        require_equals = true,
        default_missing_value = "true"
    )]
    unchecked_first: Option<bool>,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Summarise one or more `experiments.jsonl` journals.
    Report {
        /// Journal files.
        journals: Vec<PathBuf>,
    },
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    if let Some(Command::Report { journals }) = cli.command {
        if journals.is_empty() {
            eprintln!("usage: neat_ai_ockham report <experiments.jsonl> [...]");
            return ExitCode::FAILURE;
        }
        return match neat_ai_ockham::summarise(&journals) {
            Ok(report) => match serde_json::to_string_pretty(&report) {
                Ok(json) => {
                    println!("{json}");
                    ExitCode::SUCCESS
                }
                Err(e) => {
                    eprintln!("cannot serialise report: {e}");
                    ExitCode::FAILURE
                }
            },
            Err(e) => {
                eprintln!("{e}");
                ExitCode::FAILURE
            }
        };
    }
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
        global_champion: cli.global_champion,
        max_full: cli.max_full,
        max_accepts: cli.max_accepts,
        learnings_dir: cli.learnings_dir,
        learnings_host: cli.learnings_host,
        learnings_replay: cli.learnings_replay,
        ordering: cli.ordering,
        ordering_random_quota: cli.ordering_random_quota,
        unchecked_first: cli.unchecked_first,
        stats_sample_records: cli.stats_sample_records,
    };
    if let Err(e) = config.validate() {
        eprintln!("{e}");
        return ExitCode::from(2);
    }

    let scorer = ExternalScorer {
        binary: config.scorer_path.clone(),
        extra_args: config.scorer_args.clone(),
    };
    match establish_run(&config, &scorer) {
        Ok(run) => {
            match serde_json::to_string_pretty(&run) {
                Ok(json) => println!("{json}"),
                Err(e) => {
                    eprintln!("cannot serialise baseline: {e}");
                    return ExitCode::FAILURE;
                }
            }
            ExitCode::SUCCESS
        }
        Err(e) => {
            log::warn(&format!("run aborted: {e}"));
            ExitCode::FAILURE
        }
    }
}
