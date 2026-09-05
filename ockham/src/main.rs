//! `neat_ai_ockham` command-line interface.
//!
//! Issue #2: load the immutable incumbent, copy it, and score a full-corpus
//! baseline. Pruning lands in later issues; this binary must not prune yet.

use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Duration;

use clap::{Parser, Subcommand};
use neat_ai_ockham::config::{
    DEFAULT_CANDIDATE_COUNT, DEFAULT_MAX_CONSECUTIVE_SCORER_FAILURES, DEFAULT_MIN_IMPROVEMENT,
    DEFAULT_ORDERING, DEFAULT_SCREEN_SAMPLE_RATE, DEFAULT_SCREEN_THRESHOLD,
    DEFAULT_TIMEOUT_SECONDS, OckhamConfig,
};
use neat_ai_ockham::model::{
    DEFAULT_EPOCHS, DEFAULT_L2, DEFAULT_LEARNING_RATE, PriorityModel, TrainingConfig,
};
use neat_ai_ockham::neighbourhood::{DEFAULT_NEIGHBOURHOOD_PROPOSALS, DEFAULT_NEIGHBOURHOOD_SIZE};
use neat_ai_ockham::screening::{DEFAULT_SCREEN_REJECT_MARGIN, ScreenLadder};
use neat_ai_ockham::stats::DEFAULT_SAMPLE_RECORDS;
use neat_ai_ockham::telemetry;
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
    /// Progressive screening ladder: ascending `rate[:margin]` stages, e.g.
    /// `0.0025:0.02,0.01,0.05`. Omitted: one stage at `--screen-sample-rate`.
    #[arg(long)]
    screen_stages: Option<String>,
    /// Early-rejection margin for a ladder stage that names none: a sampled Δ
    /// at or below its negation is rejected there instead of re-tested.
    /// Default 0.01; only meaningful with `--screen-stages`.
    #[arg(long)]
    screen_reject_margin: Option<f64>,
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
    /// Cap winners scored individually (highest sample Δ first). Never restricts bundle membership.
    /// Omit to score every sampled winner individually.
    #[arg(long)]
    max_full: Option<usize>,
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
    /// Omitted: 0, or 0.1 for `--ordering learned`, so a fitted model cannot
    /// permanently starve the candidates it ranks last.
    #[arg(long)]
    ordering_random_quota: Option<f64>,
    /// Fitted ranking model for `--ordering learned`, from `train-ordering`.
    /// Ranking only: the scorer still decides what survives.
    #[arg(long)]
    ordering_model: Option<PathBuf>,
    /// Append one candidate feature/outcome row per scored candidate here,
    /// as offline training data for `train-ordering`. Omitted: write nothing.
    #[arg(long)]
    candidate_log: Option<PathBuf>,
    /// Also propose bounded structural neighbourhood group cuts — chains and
    /// low-fan-out branches removed as one candidate (issue #108). Experimental
    /// and off by default; a group still faces the screen and the full scorer.
    #[arg(long)]
    group_cuts: bool,
    /// Hidden neurons in one group proposal. Refused outside 2-8, whether or
    /// not `--group-cuts` is given: a size the razor would clamp is a typo
    /// worth stopping for.
    #[arg(long, default_value_t = DEFAULT_NEIGHBOURHOOD_SIZE)]
    group_max_size: usize,
    /// Group proposals offered per sweep batch. Only with `--group-cuts`.
    #[arg(long, default_value_t = DEFAULT_NEIGHBOURHOOD_PROPOSALS)]
    group_proposals: usize,
    /// Skip the exact structural cleanup pre-pass. The pre-pass removes only
    /// structure proven redundant — dead wood exposed by exactly-zero weights,
    /// constant folds and cost-reducing IDENTITY collapses — before the first
    /// sampled screen, and spends no scorer budget doing it. Skip it to measure
    /// what it buys.
    #[arg(long)]
    no_exact_cleanup: bool,
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
    /// Check hidden neurons an older corpus once removed before the rest.
    /// A hint only: every one still faces the screen and the full corpus.
    /// Defaults to on with `--learnings-dir` and off without it.
    #[arg(
        long,
        num_args = 0..=1,
        require_equals = true,
        default_missing_value = "true"
    )]
    old_corpus_first: Option<bool>,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Summarise one or more `experiments.jsonl` journals.
    Report {
        /// Journal files.
        journals: Vec<PathBuf>,
    },
    /// Fit the learned candidate ranker from `--candidate-log` files (#107).
    ///
    /// Offline and reproducible: the same rows and hyper-parameters produce the
    /// same model. The model only ever ranks the sweep — it can never accept a
    /// cut, which stays the full-corpus scorer's decision alone.
    TrainOrdering {
        /// Candidate logs written by `--candidate-log`.
        logs: Vec<PathBuf>,
        /// Where to write the fitted model JSON.
        #[arg(long)]
        out: PathBuf,
        /// Gradient-descent passes.
        #[arg(long, default_value_t = DEFAULT_EPOCHS)]
        epochs: usize,
        /// Learning rate.
        #[arg(long, default_value_t = DEFAULT_LEARNING_RATE)]
        learning_rate: f64,
        /// L2 penalty.
        #[arg(long, default_value_t = DEFAULT_L2)]
        l2: f64,
        /// Hold every Nth row out of training to evaluate the fit; 0 evaluates
        /// on the training rows and says so. 1 is refused — it would hold out
        /// every row and leave nothing to fit.
        #[arg(long, default_value_t = 5)]
        holdout_every: usize,
        /// Confirmed-win threshold on the full-corpus delta.
        #[arg(long, default_value_t = DEFAULT_MIN_IMPROVEMENT)]
        min_improvement: f64,
    },
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    if let Some(Command::TrainOrdering {
        logs,
        out,
        epochs,
        learning_rate,
        l2,
        holdout_every,
        min_improvement,
    }) = &cli.command
    {
        return match train_ordering(
            logs,
            out,
            TrainingConfig {
                epochs: *epochs,
                learning_rate: *learning_rate,
                l2: *l2,
                corpora: Vec::new(),
            },
            *holdout_every,
            *min_improvement,
        ) {
            Ok(report) => {
                println!("{report}");
                ExitCode::SUCCESS
            }
            Err(e) => {
                eprintln!("{e}");
                ExitCode::FAILURE
            }
        };
    }
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

    // A malformed ladder is a configuration fault, not a quietly ignored flag —
    // and neither is a margin given with no ladder to apply it to.
    let screen_stages = match cli.screen_stages.as_deref() {
        Some(spec) => {
            let margin = cli
                .screen_reject_margin
                .unwrap_or(DEFAULT_SCREEN_REJECT_MARGIN);
            match ScreenLadder::parse(spec, margin) {
                Ok(ladder) => Some(ladder),
                Err(e) => {
                    eprintln!("{e}");
                    return ExitCode::from(2);
                }
            }
        }
        None => {
            if cli.screen_reject_margin.is_some() {
                eprintln!("--screen-reject-margin has no effect without --screen-stages");
                return ExitCode::from(2);
            }
            None
        }
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
        screen_stages,
        screen_threshold: cli.screen_threshold,
        min_improvement: cli.min_improvement,
        max_consecutive_scorer_failures: cli.max_consecutive_scorer_failures,
        global_champion: cli.global_champion,
        max_full: cli.max_full,
        learnings_dir: cli.learnings_dir,
        learnings_host: cli.learnings_host,
        learnings_replay: cli.learnings_replay,
        ordering: cli.ordering,
        ordering_random_quota: OckhamConfig::resolve_random_quota(
            cli.ordering,
            cli.ordering_random_quota,
        ),
        ordering_model: cli.ordering_model,
        candidate_log: cli.candidate_log,
        unchecked_first: cli.unchecked_first,
        old_corpus_first: cli.old_corpus_first,
        stats_sample_records: cli.stats_sample_records,
        group_cuts: cli.group_cuts,
        group_max_size: cli.group_max_size,
        group_proposals: cli.group_proposals,
        exact_cleanup: !cli.no_exact_cleanup,
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

/// Fit and evaluate the learned ranker, returning the JSON report to print.
///
/// The holdout is every `holdout_every`-th row by position, so the split is
/// reproducible from the inputs alone — no RNG, no shuffle, no seed to carry.
/// `0` trains and evaluates on the same rows and labels the report `train`, so
/// an optimistic number is never presented as a held-out one.
fn train_ordering(
    logs: &[PathBuf],
    out: &Path,
    mut config: TrainingConfig,
    holdout_every: usize,
    min_improvement: f64,
) -> Result<String, String> {
    if logs.is_empty() {
        return Err(
            "usage: neat_ai_ockham train-ordering <candidates.jsonl> [...] --out <model.json>"
                .into(),
        );
    }
    // Holding out every row leaves nothing to fit. Refused by name rather than
    // quietly reinterpreted as "no holdout", which would report a training-set
    // number under the holdout heading.
    if holdout_every == 1 {
        return Err(
            "--holdout-every 1 would hold out every row; use 0 to evaluate on the training rows"
                .into(),
        );
    }
    let mut records = Vec::new();
    for log in logs {
        records.extend(telemetry::load(log)?);
    }
    let (rows, skipped) = telemetry::training_rows(&records, min_improvement);
    if skipped > 0 {
        eprintln!("train-ordering: {skipped} row(s) do not carry the current feature schema");
    }
    config.corpora = telemetry::corpora(&records);
    let (train, holdout): (Vec<_>, Vec<_>) = if holdout_every > 0 {
        let (a, b): (Vec<_>, Vec<_>) = rows
            .iter()
            .cloned()
            .enumerate()
            .partition(|(i, _)| i % holdout_every != 0);
        (
            a.into_iter().map(|(_, r)| r).collect(),
            b.into_iter().map(|(_, r)| r).collect(),
        )
    } else {
        (rows.clone(), Vec::new())
    };
    let model = PriorityModel::fit(&train, config)?;
    model.save(out)?;
    let evaluated_on = if holdout.is_empty() {
        "train"
    } else {
        "holdout"
    };
    let evaluation = model.evaluate(if holdout.is_empty() { &train } else { &holdout });
    let report = serde_json::json!({
        "model": out,
        "records": records.len(),
        "skippedRecords": skipped,
        "trainingRows": train.len(),
        "holdoutRows": holdout.len(),
        "evaluatedOn": evaluated_on,
        "evaluation": evaluation,
        "corpora": telemetry::corpora(&records),
        "coefficients": model
            .coefficients()
            .into_iter()
            .collect::<std::collections::BTreeMap<_, _>>(),
        "bias": model.bias(),
    });
    serde_json::to_string_pretty(&report).map_err(|e| e.to_string())
}
