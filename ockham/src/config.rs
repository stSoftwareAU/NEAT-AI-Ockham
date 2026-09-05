//! Run configuration — every knob the CLI exposes, with defaults.

use std::path::PathBuf;
use std::time::Duration;

use serde::Serialize;

use crate::ordering::{Ordering, OrderingConfig};
use crate::screening::{ScreenLadder, ScreenStage};
use crate::stats::{DEFAULT_SAMPLE_RECORDS, SampleSpec};

/// Default wall-clock budget (45 minutes).
pub const DEFAULT_TIMEOUT_SECONDS: u64 = 45 * 60;
/// Default candidate batch size for the sampled sweep (issue #6).
pub const DEFAULT_CANDIDATE_COUNT: usize = 100;
/// Default scorer subsample rate for screening (issue #6).
pub const DEFAULT_SCREEN_SAMPLE_RATE: f64 = 0.05;
/// Default sampled Δscore a candidate must exceed to be promoted (issue #6).
pub const DEFAULT_SCREEN_THRESHOLD: f64 = 0.0;
/// Default strict minimum authoritative improvement (NEAT-AI family convention).
pub const DEFAULT_MIN_IMPROVEMENT: f64 = 1e-6;
/// Abort after this many consecutive scorer failures.
pub const DEFAULT_MAX_CONSECUTIVE_SCORER_FAILURES: u32 = 3;
/// Default candidate ordering — the random control stays the default until
/// benchmark evidence shows a ranking earns better economics (issue #11).
pub const DEFAULT_ORDERING: Ordering = Ordering::Random;
/// Default fraction of sweep slots reserved for random exploration (issue #11).
pub const DEFAULT_ORDERING_RANDOM_QUOTA: f64 = 0.0;
/// Random exploration a `learned` run reserves when the flag is omitted (#107).
///
/// A hand-written ranking is a fixed function of the creature, so a neuron it
/// buries is buried for a reason a human can read. A fitted model is not: it
/// learns from the outcomes of the candidates it chose, so a uuid it ranks last
/// is never tried, never logged, and never gets to change its own mind. One
/// visit in ten drawn from the random control is what stops that loop closing —
/// `--ordering-random-quota` still overrides it, `0` included.
pub const DEFAULT_LEARNED_RANDOM_QUOTA: f64 = 0.1;

/// Complete configuration.
#[derive(Debug, Clone, PartialEq)]
pub struct OckhamConfig {
    /// Source creature path (read-only).
    pub creature: PathBuf,
    /// Training corpus directory.
    pub training_data: PathBuf,
    /// Output directory (`best.json`, `experiments.jsonl`, `winners/`, `workspace/`).
    pub output_dir: PathBuf,
    /// Scorer binary.
    pub scorer_path: PathBuf,
    /// Extra scorer arguments passed verbatim.
    pub scorer_args: Vec<String>,
    /// Optional latest global champion JSON to compare at re-entry (Issue #9).
    pub global_champion: Option<PathBuf>,
    /// Wall-clock budget.
    pub timeout: Duration,
    /// Maximum experiments (`None` = until timeout).
    pub max_experiments: Option<u64>,
    /// RNG seed (`None` = drawn when optimisation starts).
    pub seed: Option<u64>,
    /// Candidate batch size.
    pub candidates: usize,
    /// Screen sample rate (`None` = no screen).
    pub screen_sample_rate: Option<f64>,
    /// Progressive screening ladder (`None` = the fixed-rate control, #104).
    pub screen_stages: Option<ScreenLadder>,
    /// Sampled Δscore a candidate must exceed to be promoted.
    pub screen_threshold: f64,
    /// Strict minimum authoritative improvement.
    pub min_improvement: f64,
    /// Consecutive scorer failures tolerated before aborting.
    pub max_consecutive_scorer_failures: u32,
    /// Cap on sampled winners sent to full scoring (`None` = every sampled winner).
    pub max_full: Option<usize>,
    /// Shared full-corpus prune-verdict cache (`None` = do not read or write).
    pub learnings_dir: Option<PathBuf>,
    /// Host label for the per-host jsonl file (`None` = [`crate::learnings::default_host`]).
    pub learnings_host: Option<String>,
    /// Max known-win UUIDs to replay before the random sweep (`0` = all still present).
    pub learnings_replay: usize,
    /// Named candidate ordering strategy (issue #11).
    pub ordering: Ordering,
    /// Fraction of sweep slots reserved for the random control, in `[0, 1)`.
    pub ordering_random_quota: f64,
    /// Fitted ranking model for `--ordering learned` (issue #107).
    ///
    /// Required by that ordering; only that ordering ranks with it. Supplying
    /// one to another ordering is not an error, but the file is still read and
    /// an unreadable model still stops the run — a path that was typed and
    /// silently ignored is how a `learned` run comes to rank by something else.
    /// A model only ever reorders the sweep; the scorer settles every candidate.
    pub ordering_model: Option<PathBuf>,
    /// Candidate feature/outcome log for offline training (issue #107).
    ///
    /// `None` writes nothing, which is what a control run does.
    pub candidate_log: Option<PathBuf>,
    /// Screen never-checked neurons before re-screening stale ones (issue #38).
    ///
    /// `None` takes the default: on when [`Self::learnings_dir`] is set, off
    /// without it — with no screen store there is no coverage state to prefer.
    pub unchecked_first: Option<bool>,
    /// Check neurons an older corpus once removed before the rest (issue #88).
    ///
    /// `None` takes the default: on when [`Self::learnings_dir`] is set, off
    /// without it — with no cache there are no old-corpus verdicts to read.
    pub old_corpus_first: Option<bool>,
    /// Cap on records visited by the activation scan; `0` = full corpus (#44).
    pub stats_sample_records: u64,
}

impl Default for OckhamConfig {
    fn default() -> Self {
        Self {
            creature: "creature.json".into(),
            training_data: "training".into(),
            output_dir: ".".into(),
            scorer_path: "rust_scorer".into(),
            scorer_args: Vec::new(),
            global_champion: None,
            timeout: Duration::from_secs(DEFAULT_TIMEOUT_SECONDS),
            max_experiments: None,
            seed: None,
            candidates: DEFAULT_CANDIDATE_COUNT,
            screen_sample_rate: Some(DEFAULT_SCREEN_SAMPLE_RATE),
            screen_stages: None,
            screen_threshold: DEFAULT_SCREEN_THRESHOLD,
            min_improvement: DEFAULT_MIN_IMPROVEMENT,
            max_consecutive_scorer_failures: DEFAULT_MAX_CONSECUTIVE_SCORER_FAILURES,
            max_full: None,
            learnings_dir: None,
            learnings_host: None,
            learnings_replay: 0,
            ordering: DEFAULT_ORDERING,
            ordering_random_quota: DEFAULT_ORDERING_RANDOM_QUOTA,
            ordering_model: None,
            candidate_log: None,
            unchecked_first: None,
            old_corpus_first: None,
            stats_sample_records: DEFAULT_SAMPLE_RECORDS,
        }
    }
}

impl OckhamConfig {
    /// Validate cross-field constraints; error messages name the flag.
    pub fn validate(&self) -> Result<(), String> {
        if self.candidates == 0 {
            return Err("--candidates must be > 0".into());
        }
        if let Some(rate) = self.screen_sample_rate
            && !(rate > 0.0 && rate < 1.0)
        {
            return Err("--screen-sample-rate must be in (0, 1); use 0 to disable".into());
        }
        // A ladder and a disabled screen are contradictory instructions, and
        // silently honouring one of them would hide which (#104).
        if self.screen_stages.is_some() && self.screen_sample_rate.is_none() {
            return Err(
                "--screen-stages cannot be combined with --screen-sample-rate 0; \
                 a disabled screen has no stages"
                    .into(),
            );
        }
        if !self.screen_threshold.is_finite() {
            return Err("--screen-threshold must be finite".into());
        }
        if self.min_improvement <= 0.0 || self.min_improvement.is_nan() {
            return Err("--min-improvement must be > 0".into());
        }
        if self.timeout.is_zero() {
            return Err("--timeout-seconds must be > 0".into());
        }
        if self.max_consecutive_scorer_failures == 0 {
            return Err("--max-consecutive-scorer-failures must be > 0".into());
        }
        if let Some(n) = self.max_full
            && n == 0
        {
            return Err("--max-full must be > 0".into());
        }
        self.ordering_config().validate()?;
        // The model is a file rather than a flag value, so the path is checked
        // here and the loaded model is checked again at `--ordering learned`
        // run start: a run that asked to be ranked by a model must stop when it
        // is missing, never rank by something else and say nothing.
        if self.ordering == Ordering::Learned && self.ordering_model.is_none() {
            return Err("--ordering learned requires --ordering-model <model.json> \
                 (build one with `neat_ai_ockham train-ordering`)"
                .into());
        }
        Ok(())
    }

    /// The screening ladder this run will actually climb (issue #104).
    ///
    /// `Ok(None)` when screening is disabled. `--screen-stages` wins when
    /// given; otherwise the fixed-rate control — one stage at
    /// `--screen-sample-rate`, which is exactly the screen Ockham has always
    /// run.
    ///
    /// A rate outside `(0, 1)` is an error rather than a `None`: a caller that
    /// skipped [`Self::validate`] would otherwise get a silently un-screened
    /// run — every candidate straight to the full corpus — in place of the
    /// cheap screen it asked for.
    pub fn screen_ladder(&self) -> Result<Option<ScreenLadder>, String> {
        let Some(rate) = self.screen_sample_rate else {
            return Ok(None);
        };
        match &self.screen_stages {
            Some(ladder) => Ok(Some(ladder.clone())),
            None => ScreenLadder::single(rate).map(Some),
        }
    }

    /// Reserved random quota for `ordering`, honouring `flag` when given (#107).
    ///
    /// A `learned` run that names no quota reserves
    /// [`DEFAULT_LEARNED_RANDOM_QUOTA`]; every other ordering keeps
    /// [`DEFAULT_ORDERING_RANDOM_QUOTA`]. An explicit flag always wins, so a
    /// control comparison can still pin a learned run to no exploration at all.
    pub fn resolve_random_quota(ordering: Ordering, flag: Option<f64>) -> f64 {
        match (flag, ordering) {
            (Some(quota), _) => quota,
            (None, Ordering::Learned) => DEFAULT_LEARNED_RANDOM_QUOTA,
            (None, _) => DEFAULT_ORDERING_RANDOM_QUOTA,
        }
    }

    /// Strategy plus reserved random quota for the sweep (issue #11).
    pub fn ordering_config(&self) -> OrderingConfig<'static> {
        OrderingConfig {
            strategy: self.ordering,
            random_quota: self.ordering_random_quota,
            priority: None,
        }
    }

    /// Sampling policy for the hidden-neuron activation scan (issue #44).
    pub fn stats_sample_spec(&self) -> SampleSpec {
        SampleSpec::with_max_records(self.stats_sample_records)
    }

    /// Whether the sweep prefers never-screened neurons (issue #38).
    ///
    /// `--unchecked-first` wins when given; otherwise it follows
    /// `--learnings-dir`, because coverage state only exists with a store.
    pub fn unchecked_first_enabled(&self) -> bool {
        self.unchecked_first
            .unwrap_or_else(|| self.learnings_dir.is_some())
    }

    /// Whether the sweep checks old-corpus wins first (issue #88).
    ///
    /// `--old-corpus-first` wins when given; otherwise it follows
    /// `--learnings-dir`, because sibling `corpus-*` caches only exist there.
    pub fn old_corpus_first_enabled(&self) -> bool {
        self.old_corpus_first
            .unwrap_or_else(|| self.learnings_dir.is_some())
    }

    /// Machine-readable configuration dump (CLI stdout).
    pub fn report(&self) -> ConfigReport {
        ConfigReport {
            crate_version: crate::crate_version().to_string(),
            creature: self.creature.clone(),
            training_data: self.training_data.clone(),
            output_dir: self.output_dir.clone(),
            scorer: self.scorer_path.clone(),
            scorer_args: self.scorer_args.clone(),
            global_champion: self.global_champion.clone(),
            timeout_seconds: self.timeout.as_secs(),
            max_experiments: self.max_experiments,
            seed: self.seed,
            candidates: self.candidates,
            screen_sample_rate: self.screen_sample_rate,
            // An unresolvable ladder reports as no stages; `validate` is what
            // refuses the run, and a report is not the place to panic.
            screen_stages: self
                .screen_ladder()
                .ok()
                .flatten()
                .map(|l| l.stages().to_vec())
                .unwrap_or_default(),
            screen_threshold: self.screen_threshold,
            min_improvement: self.min_improvement,
            max_consecutive_scorer_failures: self.max_consecutive_scorer_failures,
            max_full: self.max_full,
            learnings_dir: self.learnings_dir.clone(),
            learnings_host: self.learnings_host.clone(),
            learnings_replay: self.learnings_replay,
            ordering: self.ordering,
            ordering_random_quota: self.ordering_random_quota,
            ordering_model: self.ordering_model.clone(),
            candidate_log: self.candidate_log.clone(),
            unchecked_first: self.unchecked_first_enabled(),
            old_corpus_first: self.old_corpus_first_enabled(),
            stats_sample_records: self.stats_sample_records,
            optimisation: "loop",
        }
    }
}

/// JSON report printed when the CLI starts without yet running optimisation.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ConfigReport {
    /// `neat_ai_ockham` crate version.
    pub crate_version: String,
    /// Source creature path (never modified).
    pub creature: PathBuf,
    /// Training corpus directory.
    pub training_data: PathBuf,
    /// Output directory.
    pub output_dir: PathBuf,
    /// Scorer binary.
    pub scorer: PathBuf,
    /// Extra scorer arguments.
    pub scorer_args: Vec<String>,
    /// Optional latest global champion path.
    pub global_champion: Option<PathBuf>,
    /// Wall-clock budget in seconds.
    pub timeout_seconds: u64,
    /// Optional experiment cap.
    pub max_experiments: Option<u64>,
    /// Optional RNG seed.
    pub seed: Option<u64>,
    /// Candidate batch size.
    pub candidates: usize,
    /// Screen sample rate (`None` = screening disabled).
    pub screen_sample_rate: Option<f64>,
    /// Resolved screening ladder — the control is one stage (#104).
    ///
    /// Empty when screening is disabled. Reported resolved rather than as the
    /// flag was typed, so the run says which samples it will actually take.
    pub screen_stages: Vec<ScreenStage>,
    /// Sample promotion threshold.
    pub screen_threshold: f64,
    /// Authoritative acceptance threshold.
    pub min_improvement: f64,
    /// Consecutive scorer failures tolerated.
    pub max_consecutive_scorer_failures: u32,
    /// Optional cap on sampled winners sent to full scoring.
    pub max_full: Option<usize>,
    /// Shared learnings directory (`None` = cache disabled).
    pub learnings_dir: Option<PathBuf>,
    /// Optional host override for the per-host jsonl file.
    pub learnings_host: Option<String>,
    /// Known-win replay cap (`0` = every still-present known win).
    pub learnings_replay: usize,
    /// Named candidate ordering strategy.
    pub ordering: Ordering,
    /// Fraction of sweep slots reserved for the random control.
    pub ordering_random_quota: f64,
    /// Fitted ranking model for `--ordering learned` (issue #107).
    pub ordering_model: Option<PathBuf>,
    /// Candidate feature/outcome log for offline training (issue #107).
    pub candidate_log: Option<PathBuf>,
    /// Resolved unchecked-first selection (defaults to `learnings_dir.is_some()`).
    pub unchecked_first: bool,
    /// Resolved old-corpus-first priority (defaults to `learnings_dir.is_some()`).
    pub old_corpus_first: bool,
    /// Cap on records visited by the activation scan (`0` = full corpus).
    pub stats_sample_records: u64,
    /// Optimisation status for this bootstrap issue (`deferred`).
    pub optimisation: &'static str,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_match_the_charter() {
        let c = OckhamConfig::default();
        c.validate().unwrap();
        assert_eq!(c.timeout, Duration::from_secs(2700));
        assert_eq!(c.candidates, 100);
        assert_eq!(c.screen_sample_rate, Some(0.05));
        assert_eq!(c.screen_threshold, 0.0);
        assert_eq!(c.min_improvement, 1e-6);
        assert_eq!(c.report().optimisation, "loop");
        assert_eq!(c.report().timeout_seconds, DEFAULT_TIMEOUT_SECONDS);
        assert!(c.learnings_dir.is_none());
        assert_eq!(c.learnings_replay, 0);
        assert_eq!(c.ordering, Ordering::Random, "random stays the control");
        assert_eq!(c.ordering_random_quota, 0.0);
        assert_eq!(c.report().ordering, Ordering::Random);
        // Telemetry and the learned ranker are opt-in: a control run writes no
        // candidate log and loads no model (issue #107).
        assert!(c.candidate_log.is_none());
        assert!(c.ordering_model.is_none());
        assert!(c.report().candidate_log.is_none());
    }

    #[test]
    fn a_learned_run_reserves_exploration_unless_the_flag_says_otherwise() {
        // A fitted model learns only from the candidates it chose, so a run
        // that names no quota still draws one visit in ten from the control.
        assert_eq!(
            OckhamConfig::resolve_random_quota(Ordering::Learned, None),
            DEFAULT_LEARNED_RANDOM_QUOTA
        );
        // Hand-written rankings are unchanged: the control stays the control.
        for ordering in [Ordering::Random, Ordering::Composite, Ordering::LowVariance] {
            assert_eq!(
                OckhamConfig::resolve_random_quota(ordering, None),
                DEFAULT_ORDERING_RANDOM_QUOTA,
                "{ordering} must keep its default"
            );
        }
        // An explicit flag always wins, `0` included, so a control comparison
        // can pin a learned run to no exploration at all.
        assert_eq!(
            OckhamConfig::resolve_random_quota(Ordering::Learned, Some(0.0)),
            0.0
        );
        assert_eq!(
            OckhamConfig::resolve_random_quota(Ordering::Composite, Some(0.5)),
            0.5
        );
    }

    #[test]
    fn a_learned_ordering_without_a_model_names_the_flag() {
        let bad = OckhamConfig {
            ordering: Ordering::Learned,
            ..OckhamConfig::default()
        };
        let err = bad.validate().unwrap_err();
        assert!(err.contains("--ordering-model"), "{err}");
        let good = OckhamConfig {
            ordering: Ordering::Learned,
            ordering_model: Some("model.json".into()),
            ..OckhamConfig::default()
        };
        // The path is accepted here and the file is read at run start, where a
        // model that will not load stops the run.
        good.validate().unwrap();
        assert_eq!(good.report().ordering_model, Some("model.json".into()));
        // Every other ordering ignores the model entirely.
        OckhamConfig {
            ordering: Ordering::Composite,
            ..OckhamConfig::default()
        }
        .validate()
        .unwrap();
    }

    #[test]
    fn bad_values_name_the_flag() {
        let c = OckhamConfig::default();
        let bad = OckhamConfig {
            candidates: 0,
            ..c.clone()
        };
        assert!(bad.validate().unwrap_err().contains("--candidates"));
        let bad = OckhamConfig {
            screen_sample_rate: Some(1.5),
            ..c.clone()
        };
        assert!(bad.validate().unwrap_err().contains("--screen-sample-rate"));
        let bad = OckhamConfig {
            min_improvement: 0.0,
            ..c.clone()
        };
        assert!(bad.validate().unwrap_err().contains("--min-improvement"));
        let bad = OckhamConfig {
            timeout: Duration::ZERO,
            ..c
        };
        assert!(bad.validate().unwrap_err().contains("--timeout-seconds"));
        let bad = OckhamConfig {
            max_full: Some(0),
            ..OckhamConfig::default()
        };
        assert!(bad.validate().unwrap_err().contains("--max-full"));
        let bad = OckhamConfig {
            ordering_random_quota: 1.0,
            ..OckhamConfig::default()
        };
        assert!(
            bad.validate()
                .unwrap_err()
                .contains("--ordering-random-quota")
        );
    }

    #[test]
    fn unchecked_first_follows_the_learnings_dir_by_default() {
        let without = OckhamConfig::default();
        assert!(
            !without.unchecked_first_enabled(),
            "no cache means no coverage state to prefer"
        );
        assert!(!without.report().unchecked_first);

        let with = OckhamConfig {
            learnings_dir: Some("/tmp/learnings".into()),
            ..OckhamConfig::default()
        };
        assert!(with.unchecked_first_enabled());
        assert!(with.report().unchecked_first);
    }

    #[test]
    fn an_explicit_unchecked_first_flag_overrides_the_default() {
        let off = OckhamConfig {
            learnings_dir: Some("/tmp/learnings".into()),
            unchecked_first: Some(false),
            ..OckhamConfig::default()
        };
        off.validate().unwrap();
        assert!(!off.unchecked_first_enabled());
        assert!(!off.report().unchecked_first);

        let on = OckhamConfig {
            unchecked_first: Some(true),
            ..OckhamConfig::default()
        };
        on.validate().unwrap();
        assert!(on.unchecked_first_enabled());
        assert!(on.report().unchecked_first);
    }

    /// Issue #88: the old-corpus priority is on wherever there is a cache to
    /// read it from, and off wherever there is not.
    #[test]
    fn old_corpus_first_follows_the_learnings_dir_by_default() {
        let without = OckhamConfig::default();
        assert!(
            !without.old_corpus_first_enabled(),
            "no cache means no sibling corpus directories to read"
        );
        assert!(!without.report().old_corpus_first);

        let with = OckhamConfig {
            learnings_dir: Some("/tmp/learnings".into()),
            ..OckhamConfig::default()
        };
        assert!(with.old_corpus_first_enabled());
        assert!(with.report().old_corpus_first);
    }

    #[test]
    fn an_explicit_old_corpus_first_flag_overrides_the_default() {
        let off = OckhamConfig {
            learnings_dir: Some("/tmp/learnings".into()),
            old_corpus_first: Some(false),
            ..OckhamConfig::default()
        };
        off.validate().unwrap();
        assert!(!off.old_corpus_first_enabled());
        assert!(!off.report().old_corpus_first);

        let on = OckhamConfig {
            old_corpus_first: Some(true),
            ..OckhamConfig::default()
        };
        on.validate().unwrap();
        assert!(on.old_corpus_first_enabled());
        assert!(on.report().old_corpus_first);
    }

    #[test]
    fn the_activation_scan_samples_by_default_and_zero_restores_the_full_scan() {
        let c = OckhamConfig::default();
        assert_eq!(c.stats_sample_records, DEFAULT_SAMPLE_RECORDS);
        assert_eq!(c.stats_sample_spec().max_records, DEFAULT_SAMPLE_RECORDS);
        assert!(c.stats_sample_spec().target_rel_se > 0.0);
        assert_eq!(c.report().stats_sample_records, DEFAULT_SAMPLE_RECORDS);

        let full = OckhamConfig {
            stats_sample_records: 0,
            ..OckhamConfig::default()
        };
        full.validate().unwrap();
        assert_eq!(full.stats_sample_spec(), SampleSpec::full());
        assert_eq!(full.stats_sample_spec().target_rel_se, 0.0);
    }

    #[test]
    fn zero_screen_sample_rate_disables_screening() {
        let c = OckhamConfig {
            screen_sample_rate: None,
            ..OckhamConfig::default()
        };
        c.validate().unwrap();
        assert!(c.report().screen_sample_rate.is_none());
        assert!(c.screen_ladder().unwrap().is_none());
        assert!(c.report().screen_stages.is_empty());
    }

    /// Issue #104: the default ladder is the fixed 5% control, unchanged.
    #[test]
    fn the_default_ladder_is_the_fixed_rate_control() {
        let c = OckhamConfig::default();
        let ladder = c
            .screen_ladder()
            .unwrap()
            .expect("screening is on by default");
        assert!(!ladder.is_progressive());
        assert_eq!(ladder.stages().len(), 1);
        assert_eq!(ladder.stages()[0].rate, DEFAULT_SCREEN_SAMPLE_RATE);
        assert_eq!(c.report().screen_stages, ladder.stages().to_vec());
    }

    #[test]
    fn explicit_stages_replace_the_control_and_report_resolved() {
        let c = OckhamConfig {
            screen_stages: Some(ScreenLadder::parse("0.0025,0.01,0.05", 0.01).unwrap()),
            ..OckhamConfig::default()
        };
        c.validate().unwrap();
        let ladder = c.screen_ladder().unwrap().unwrap();
        assert!(ladder.is_progressive());
        assert_eq!(ladder.promotion_rate(), 0.05);
        assert_eq!(c.report().screen_stages.len(), 3);
        assert_eq!(c.report().screen_stages[0].reject_margin, 0.01);
    }

    #[test]
    fn stages_with_screening_disabled_is_a_contradiction() {
        let c = OckhamConfig {
            screen_sample_rate: None,
            screen_stages: Some(ScreenLadder::parse("0.0025,0.05", 0.01).unwrap()),
            ..OckhamConfig::default()
        };
        assert!(c.validate().unwrap_err().contains("--screen-stages"));
    }
}
