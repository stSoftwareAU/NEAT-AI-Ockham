//! The `bench-strata` report: its JSON types, distributions, paired
//! bootstrap intervals and the predeclared criterion (issue #263).
//!
//! Everything here is data. The `evidence` field says so, and the criterion is
//! only as good as the corpora it was run on.

use std::collections::BTreeMap;

use serde::Serialize;

use crate::ordering::SplitMix64;
use crate::strata::manifest::Manifest;

/// Schema tag written to every report.
pub const REPORT_SCHEMA: &str = "ockham-strata-bench/1";

/// The evidence caveat written to every report.
pub const EVIDENCE: &str = "fixture-or-unreviewed: this report is data, not a claim; an optimisation benefit needs representative private corpora";

/// Paired bootstrap resamples.
pub const BOOTSTRAP_RESAMPLES: usize = 2000;

/// The predeclared rule, written verbatim into every criterion.
pub const CRITERION_RULE: &str = "met iff regressionReduction >= minRegressionReduction AND meanFitnessDifference >= -maxFitnessLoss, where regressionReduction = aggregate minority regressions - arm minority regressions and meanFitnessDifference = arm mean validationDelta - aggregate mean validationDelta";

/// Descriptive statistics over a set of values.
///
/// `std` is the sample standard deviation (divisor `n - 1`), and `0.0` when
/// there is exactly one value. Every field is `null` when there are none.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Dist {
    /// Number of values.
    pub n: usize,
    /// Arithmetic mean.
    pub mean: Option<f64>,
    /// Median; the mean of the two middle values for an even count.
    pub median: Option<f64>,
    /// Smallest value.
    pub min: Option<f64>,
    /// Largest value.
    pub max: Option<f64>,
    /// Sample standard deviation.
    pub std: Option<f64>,
}

impl Dist {
    /// Distribution of `values`.
    pub fn of(values: &[f64]) -> Self {
        let n = values.len();
        if n == 0 {
            return Self {
                n,
                mean: None,
                median: None,
                min: None,
                max: None,
                std: None,
            };
        }
        let mean = values.iter().sum::<f64>() / n as f64;
        let mut sorted = values.to_vec();
        sorted.sort_by(f64::total_cmp);
        let median = if n % 2 == 1 {
            sorted[n / 2]
        } else {
            (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
        };
        let std = if n == 1 {
            0.0
        } else {
            let ss: f64 = values.iter().map(|v| (v - mean).powi(2)).sum();
            (ss / (n - 1) as f64).sqrt()
        };
        Self {
            n,
            mean: Some(mean),
            median: Some(median),
            min: Some(sorted[0]),
            max: Some(sorted[n - 1]),
            std: Some(std),
        }
    }
}

/// Seeded percentile bootstrap 95% interval for the mean of `values`.
///
/// `None` with fewer than two values. Otherwise [`BOOTSTRAP_RESAMPLES`]
/// resamples with replacement are drawn from `seed`, and the 2.5th and 97.5th
/// percentiles of their means are returned.
pub fn bootstrap_ci95(values: &[f64], seed: u64) -> Option<[f64; 2]> {
    let n = values.len();
    if n < 2 {
        return None;
    }
    let mut rng = SplitMix64(seed);
    let mut means: Vec<f64> = (0..BOOTSTRAP_RESAMPLES)
        .map(|_| {
            let mut sum = 0.0;
            for _ in 0..n {
                let u = (rng.next() >> 11) as f64 / 9_007_199_254_740_992.0;
                let idx = ((u * n as f64) as usize).min(n - 1);
                sum += values[idx];
            }
            sum / n as f64
        })
        .collect();
    means.sort_by(f64::total_cmp);
    Some([
        means[percentile_index(0.025)],
        means[percentile_index(0.975)],
    ])
}

fn percentile_index(q: f64) -> usize {
    let last = BOOTSTRAP_RESAMPLES - 1;
    ((q * last as f64).round() as usize).min(last)
}

/// Per-seed paired differences of one metric between two arms.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Paired {
    /// The arm being judged.
    pub treatment: String,
    /// The arm it is compared with.
    pub control: String,
    /// Metric name.
    pub metric: String,
    /// `treatment - control`, one per seed, in seed order.
    pub differences: Vec<f64>,
    /// Mean of `differences`; `null` when there are none.
    pub mean: Option<f64>,
    /// Bootstrap 95% interval; `null` with fewer than two seeds.
    pub ci95: Option<[f64; 2]>,
}

impl Paired {
    /// Build a paired row, bootstrapping its interval from `seed`.
    pub fn new(
        treatment: &str,
        control: &str,
        metric: &str,
        differences: Vec<f64>,
        seed: u64,
    ) -> Self {
        let mean = if differences.is_empty() {
            None
        } else {
            Some(differences.iter().sum::<f64>() / differences.len() as f64)
        };
        let ci95 = bootstrap_ci95(&differences, seed);
        Self {
            treatment: treatment.to_string(),
            control: control.to_string(),
            metric: metric.to_string(),
            differences,
            mean,
            ci95,
        }
    }
}

/// The predeclared success criterion for one treatment arm against
/// `aggregate`.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Criterion {
    /// The treatment arm.
    pub arm: String,
    /// Aggregate minority regressions minus the arm's. Positive is better.
    pub regression_reduction: i64,
    /// Arm mean validation delta minus aggregate's; `null` when either is absent.
    pub mean_fitness_difference: Option<f64>,
    /// Whether the rule holds.
    pub met: bool,
    /// The predeclared rule, verbatim.
    pub rule: String,
}

/// Inputs to [`criterion`], gathered so the call stays readable.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CriterionInput<'a> {
    /// The treatment arm's name.
    pub arm: &'a str,
    /// Minority regressions of `aggregate` over all seeds.
    pub aggregate_regressions: u64,
    /// Minority regressions of the treatment arm over all seeds.
    pub arm_regressions: u64,
    /// `aggregate`'s mean validation delta, if any seed produced one.
    pub aggregate_fitness: Option<f64>,
    /// The treatment arm's mean validation delta.
    pub arm_fitness: Option<f64>,
    /// The manifest's `success.minRegressionReduction`.
    pub min_regression_reduction: u64,
    /// The manifest's `success.maxFitnessLoss`.
    pub max_fitness_loss: f64,
}

/// Judge one treatment arm against `aggregate` under the predeclared rule.
pub fn criterion(input: CriterionInput<'_>) -> Criterion {
    let regression_reduction = input.aggregate_regressions as i64 - input.arm_regressions as i64;
    let mean_fitness_difference = match (input.arm_fitness, input.aggregate_fitness) {
        (Some(arm), Some(agg)) => Some(arm - agg),
        _ => None,
    };
    let reduction_ok =
        regression_reduction >= 0 && regression_reduction as u64 >= input.min_regression_reduction;
    let fitness_ok = mean_fitness_difference.is_some_and(|d| d >= -input.max_fitness_loss);
    Criterion {
        arm: input.arm.to_string(),
        regression_reduction,
        mean_fitness_difference,
        met: reduction_ok && fitness_ok,
        rule: CRITERION_RULE.to_string(),
    }
}

/// Identity and volume of one corpus stratum, as the report records it.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct StratumVersion {
    /// Stratum name.
    pub name: String,
    /// Corpus identity from [`crate::corpus::corpus_info`].
    pub identity: String,
    /// Records in the stratum.
    pub records: u64,
}

/// Software, scorer and input identities.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Versions {
    /// Crate version that produced the report.
    pub ockham_version: String,
    /// Scorer identity from [`crate::scorer::DirectoryScorer::identity`].
    pub scorer_identity: String,
    /// The `source` line of the `neat-core` package in the workspace
    /// `Cargo.lock`, which pins the core that compiled and scored the run.
    pub neat_core_source: String,
    /// SHA-256 of the parent creature's bytes.
    pub parent_checksum: String,
    /// SHA-256 of the manifest's bytes.
    pub manifest_sha256: String,
    /// Calibration strata, in manifest order.
    pub calibration: Vec<StratumVersion>,
    /// Validation strata, in manifest order.
    pub validation: Vec<StratumVersion>,
}

/// One fit cell: an informed arm at one candidate temperature.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct FitCriterion {
    /// `aggregate` or `worst-stratum`.
    pub arm: String,
    /// The candidate temperature.
    pub temperature: f64,
    /// Mean calibration full score over the built bundles; `null` if none built.
    pub criterion: Option<f64>,
    /// Bundles that built.
    pub built: usize,
    /// Bundles refused by core.
    pub refused: usize,
}

/// The temperatures the fit selected.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Temperatures {
    /// Fitted temperature for `aggregate`.
    pub aggregate: f64,
    /// Fitted temperature for `worst-stratum`.
    pub worst_stratum: f64,
}

/// What the calibration fit produced.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Fitted {
    /// Scale per calibration stratum.
    pub scales: BTreeMap<String, f64>,
    /// Selected temperatures.
    pub temperatures: Temperatures,
    /// Every (arm, temperature) cell considered.
    pub fit_criteria: Vec<FitCriterion>,
}

/// How much of the parent the statistics reached.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Coverage {
    /// Hidden neurons in the parent.
    pub hidden_neurons: usize,
    /// Neurons whose single cut was built and scored.
    pub measured: usize,
    /// Neurons whose single cut core refused; excluded from every arm.
    pub refused_singles: usize,
    /// Records sampled per calibration stratum for the statistics.
    pub stats_records: BTreeMap<String, u64>,
}

/// Costs of the run. Wall-clock figures are indicative only.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Costs {
    /// Wall time of preparation (steps 1 to 6, including the parent latency
    /// probe), in milliseconds.
    pub prep_ms: f64,
    /// Scorer calls made during preparation.
    pub prep_scorer_calls: u64,
    /// Total wall time spent inside the latency probe, in milliseconds.
    pub latency_probe_ms: f64,
    /// Latency of the parent creature, in milliseconds per record.
    pub parent_latency_ms: Option<f64>,
    /// Scorer calls made scoring the validation strata.
    pub validation_scorer_calls: u64,
}

/// Short, fixed descriptions of each method step, for readers of the report.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Method {
    /// Where the formula comes from.
    pub source: String,
    /// How importance is measured.
    pub importance: String,
    /// How scales are fitted.
    pub scales: String,
    /// How the statistics are formed.
    pub statistics: String,
    /// How bundles are sampled.
    pub sampler: String,
    /// How temperatures are chosen.
    pub temperature_fit: String,
    /// How each arm is evaluated.
    pub evaluation: String,
    /// How validation is run and what it may influence.
    pub validation: String,
    /// How latency is measured.
    pub latency: String,
}

/// One member of the Pareto archive, with its measured objectives.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ArchiveEntry {
    /// Member neuron UUIDs, sorted.
    pub members: Vec<String>,
    /// Worst-stratum loss on calibration, `max_t (parent − bundle)`; lower is
    /// better.
    pub worst_stratum_loss: f64,
    /// Synapse count of the bundle's creature after the cut.
    pub synapses: usize,
    /// Measured latency of the bundle, in milliseconds per record.
    pub latency_ms: f64,
    /// Full calibration delta against the parent.
    pub calibration_delta: f64,
    /// Whether the calibration delta beat `minImprovement`.
    pub win: bool,
}

/// One arm's result for one seed.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RunRecord {
    /// Evaluation seed.
    pub seed: u64,
    /// Arm name.
    pub arm: String,
    /// Bundles scored by the full calibration scorer.
    pub bundles_scored: usize,
    /// Bundles core refused to build.
    pub refused: usize,
    /// Scored bundles whose calibration delta beat `minImprovement`.
    pub wins: usize,
    /// The selected bundle's neuron UUIDs; `null` when nothing won.
    pub selected: Option<Vec<String>>,
    /// Calibration delta of the selected bundle; `null` when nothing won.
    pub calibration_delta: Option<f64>,
    /// Weighted validation delta; `0` when nothing won.
    pub validation_delta: f64,
    /// Per-validation-stratum delta; all `0` when nothing won.
    pub validation_strata: BTreeMap<String, f64>,
    /// The validation stratum with the fewest records.
    pub minority_stratum: String,
    /// Whether the minority stratum regressed beyond `regressionTolerance`.
    pub minority_regression: bool,
    /// Validation strata that regressed beyond `regressionTolerance`.
    pub regressed_strata: usize,
    /// Hidden neurons the selected cut removed, cascade included.
    pub hidden_removed: usize,
    /// Synapses the selected cut removed.
    pub synapses_removed: usize,
    /// Latency of the selected creature, in milliseconds per record.
    pub latency_ms: Option<f64>,
    /// Wall time of the arm for this seed, in milliseconds (indicative).
    pub arm_ms: f64,
    /// Calibration scorer calls made for this arm's own bundles.
    pub scorer_calls: u64,
    /// Wins per hour of arm and preparation time; `null` when no time elapsed.
    pub wins_per_hour: Option<f64>,
    /// Pareto archive size after the cap; `null` for other arms.
    pub archive_size: Option<usize>,
    /// The Pareto archive with its measured objectives, in archive order;
    /// `null` for other arms.
    pub archive: Option<Vec<ArchiveEntry>>,
}

/// One arm's distributions over all seeds.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ArmSummary {
    /// Arm name.
    pub name: String,
    /// Validation delta over seeds.
    pub validation_delta: Dist,
    /// Calibration delta over seeds that selected a bundle.
    pub calibration_delta: Dist,
    /// Seeds whose minority stratum regressed.
    pub minority_regressions: usize,
    /// Hidden neurons removed over seeds.
    pub hidden_removed: Dist,
    /// Synapses removed over seeds.
    pub synapses_removed: Dist,
    /// Wins over seeds.
    pub wins: Dist,
    /// Wins per hour over seeds that produced a value.
    pub wins_per_hour: Dist,
}

/// A failure recorded during the run and kept in the report.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Failure {
    /// Stage name: `importance`, `fit`, `evaluate`, `validation` or `latency`.
    pub stage: String,
    /// Seed, when the failure belongs to one.
    pub seed: Option<u64>,
    /// Arm, when the failure belongs to one.
    pub arm: Option<String>,
    /// Neuron, when one neuron caused it.
    pub uuid: Option<String>,
    /// Core's or the scorer's message.
    pub detail: String,
}

/// The whole report, written as camelCase JSON.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct StrataReport {
    /// [`REPORT_SCHEMA`].
    pub schema: String,
    /// [`EVIDENCE`].
    pub evidence: String,
    /// Short method descriptions.
    pub method: Method,
    /// Software and input identities.
    pub versions: Versions,
    /// The effective manifest, defaults applied.
    pub manifest: Manifest,
    /// Calibration fit results.
    pub fitted: Fitted,
    /// Statistics coverage.
    pub coverage: Coverage,
    /// Run costs.
    pub costs: Costs,
    /// One entry per arm per seed, in seed then arm order.
    pub runs: Vec<RunRecord>,
    /// Per-arm distributions, in arm order.
    pub arms: Vec<ArmSummary>,
    /// Paired differences against the control arms.
    pub paired: Vec<Paired>,
    /// Predeclared criteria, one per treatment arm.
    pub criteria: Vec<Criterion>,
    /// Failures recorded during the run.
    pub failures: Vec<Failure>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dist_of_literal_numbers() {
        let d = Dist::of(&[4.0, 1.0, 3.0, 2.0]);
        assert_eq!(d.n, 4);
        assert_eq!(d.mean, Some(2.5));
        assert_eq!(d.median, Some(2.5));
        assert_eq!(d.min, Some(1.0));
        assert_eq!(d.max, Some(4.0));
        // Sample variance: (2.25 + 0.25 + 0.25 + 2.25) / 3 = 5/3.
        let std = d.std.unwrap();
        assert!((std - (5.0f64 / 3.0).sqrt()).abs() < 1e-12, "std {std}");
    }

    #[test]
    fn dist_of_one_value_has_zero_spread_and_empty_is_all_null() {
        let one = Dist::of(&[7.0]);
        assert_eq!(one.n, 1);
        assert_eq!(one.median, Some(7.0));
        assert_eq!(one.std, Some(0.0));
        let none = Dist::of(&[]);
        assert_eq!(none.n, 0);
        assert_eq!(none.mean, None);
        assert_eq!(none.median, None);
        assert_eq!(none.min, None);
        assert_eq!(none.max, None);
        assert_eq!(none.std, None);
    }

    #[test]
    fn bootstrap_is_deterministic_for_a_seed_and_brackets_the_mean() {
        let values = [0.1, -0.2, 0.3, 0.05, 0.12];
        let a = bootstrap_ci95(&values, 9001).unwrap();
        let b = bootstrap_ci95(&values, 9001).unwrap();
        assert_eq!(a, b);
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        assert!(a[0] <= mean && mean <= a[1], "{a:?} should bracket {mean}");
        assert!(a[0] < a[1]);
    }

    #[test]
    fn bootstrap_is_null_with_fewer_than_two_values() {
        assert_eq!(bootstrap_ci95(&[], 1), None);
        assert_eq!(bootstrap_ci95(&[0.5], 1), None);
    }

    #[test]
    fn paired_row_takes_differences_mean_and_interval() {
        let p = Paired::new(
            "worst-stratum",
            "aggregate",
            "validationDelta",
            vec![1.0, 3.0],
            9001,
        );
        assert_eq!(p.mean, Some(2.0));
        assert!(p.ci95.is_some());
        let single = Paired::new("a", "b", "m", vec![0.5], 9001);
        assert_eq!(single.mean, Some(0.5));
        assert_eq!(single.ci95, None);
    }

    fn input(
        aggregate_regressions: u64,
        arm_regressions: u64,
        aggregate_fitness: Option<f64>,
        arm_fitness: Option<f64>,
    ) -> CriterionInput<'static> {
        CriterionInput {
            arm: "worst-stratum",
            aggregate_regressions,
            arm_regressions,
            aggregate_fitness,
            arm_fitness,
            min_regression_reduction: 1,
            max_fitness_loss: 0.0,
        }
    }

    #[test]
    fn criterion_is_met_when_regressions_fall_and_fitness_holds() {
        let c = criterion(input(3, 0, Some(0.01), Some(0.02)));
        assert_eq!(c.regression_reduction, 3);
        assert_eq!(c.mean_fitness_difference, Some(0.01));
        assert!(c.met);
        assert_eq!(c.rule, CRITERION_RULE);
    }

    #[test]
    fn criterion_is_not_met_without_a_regression_reduction() {
        let c = criterion(input(2, 2, Some(0.01), Some(0.02)));
        assert_eq!(c.regression_reduction, 0);
        assert!(!c.met);
    }

    #[test]
    fn criterion_is_not_met_when_fitness_falls_below_the_allowed_loss() {
        let c = criterion(input(3, 0, Some(0.01), Some(-0.001)));
        assert_eq!(c.regression_reduction, 3);
        assert!(!c.met);
        // Fitness difference is -0.001 - 0.01 = -0.011, inside a 0.02 allowance.
        let allowed = criterion(CriterionInput {
            max_fitness_loss: 0.02,
            ..input(3, 0, Some(0.01), Some(-0.001))
        });
        assert!(allowed.met);
    }

    #[test]
    fn criterion_is_not_met_when_fitness_is_unknown() {
        let c = criterion(input(3, 0, None, Some(0.02)));
        assert_eq!(c.mean_fitness_difference, None);
        assert!(!c.met);
    }
}
