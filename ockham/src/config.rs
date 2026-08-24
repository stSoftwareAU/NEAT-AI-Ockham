//! Run configuration — every knob the CLI exposes, with defaults.

use std::path::PathBuf;
use std::time::Duration;

use serde::Serialize;

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
    /// Sampled Δscore a candidate must exceed to be promoted.
    pub screen_threshold: f64,
    /// Strict minimum authoritative improvement.
    pub min_improvement: f64,
    /// Consecutive scorer failures tolerated before aborting.
    pub max_consecutive_scorer_failures: u32,
}

impl Default for OckhamConfig {
    fn default() -> Self {
        Self {
            creature: "creature.json".into(),
            training_data: "training".into(),
            output_dir: ".".into(),
            scorer_path: "rust_scorer".into(),
            scorer_args: Vec::new(),
            timeout: Duration::from_secs(DEFAULT_TIMEOUT_SECONDS),
            max_experiments: None,
            seed: None,
            candidates: DEFAULT_CANDIDATE_COUNT,
            screen_sample_rate: Some(DEFAULT_SCREEN_SAMPLE_RATE),
            screen_threshold: DEFAULT_SCREEN_THRESHOLD,
            min_improvement: DEFAULT_MIN_IMPROVEMENT,
            max_consecutive_scorer_failures: DEFAULT_MAX_CONSECUTIVE_SCORER_FAILURES,
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
        Ok(())
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
            timeout_seconds: self.timeout.as_secs(),
            max_experiments: self.max_experiments,
            seed: self.seed,
            candidates: self.candidates,
            screen_sample_rate: self.screen_sample_rate,
            screen_threshold: self.screen_threshold,
            min_improvement: self.min_improvement,
            max_consecutive_scorer_failures: self.max_consecutive_scorer_failures,
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
    /// Sample promotion threshold.
    pub screen_threshold: f64,
    /// Authoritative acceptance threshold.
    pub min_improvement: f64,
    /// Consecutive scorer failures tolerated.
    pub max_consecutive_scorer_failures: u32,
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
    }

    #[test]
    fn zero_screen_sample_rate_disables_screening() {
        let c = OckhamConfig {
            screen_sample_rate: None,
            ..OckhamConfig::default()
        };
        c.validate().unwrap();
        assert!(c.report().screen_sample_rate.is_none());
    }
}
