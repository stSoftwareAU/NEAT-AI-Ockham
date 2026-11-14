use std::collections::BTreeMap;

use serde::Serialize;
use sha2::{Digest, Sha256};

use super::RunRecord;
use crate::allocation::Arm;
use crate::manifest::{Spec, StudyKind};

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Comparison {
    pub comparator: Arm,
    pub requested_connections: usize,
    pub expected_pairs: usize,
    pub matched_pairs: usize,
    pub independent_units: usize,
    pub exclusions: BTreeMap<String, usize>,
    pub pairs: Vec<PairedDifference>,
    pub full_score_gain: Option<Interval>,
    pub held_out_score_gain: Option<Interval>,
    pub global_worst_output_mse_increase: Option<f64>,
    pub passes_predeclared_criteria: bool,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PairedDifference {
    pub subject: String,
    pub independent_unit: String,
    pub seed: u64,
    pub achieved_connections: usize,
    pub full_score_gain: f64,
    pub held_out_score_gain: f64,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Interval {
    pub mean: f64,
    pub lower: f64,
    pub upper: f64,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Summary {
    pub format_version: u32,
    pub conclusion: &'static str,
    pub study_kind: StudyKind,
    pub planned_runs: usize,
    pub completed_runs: usize,
    pub failed_runs: usize,
    pub timeouts: u64,
    pub target_misses: usize,
    pub representative_independent_units: usize,
    pub uncertainty: &'static str,
    pub timing_environment: &'static str,
    pub comparisons: Vec<Comparison>,
    pub freeze_ms: f64,
    pub verification_ms: f64,
    pub total_run_ms: f64,
    pub freeze_records_hashed: u64,
    pub verification_records_hashed: u64,
    pub search_statistics_records: u64,
    pub search_proposal_attempts: u64,
    pub scorer_reported_record_evaluations: u64,
    pub scorer_reserved_record_evaluations: u64,
    pub diagnostic_record_evaluations: u64,
    pub latency_record_evaluations: u64,
}

pub fn interval(values: &[f64], repeats: usize, tail: f64) -> Option<Interval> {
    if values.len() < 2 {
        return None;
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let mut estimates = Vec::with_capacity(repeats);
    for replicate in 0..repeats {
        let mut sum = 0.0;
        for draw in 0..values.len() {
            let mut hasher = Sha256::new();
            hasher.update(b"connection-budget-bootstrap-v1");
            hasher.update((replicate as u64).to_le_bytes());
            hasher.update((draw as u64).to_le_bytes());
            let digest = hasher.finalize();
            let index = u64::from_le_bytes(digest[..8].try_into().unwrap()) as usize % values.len();
            sum += values[index];
        }
        estimates.push(sum / values.len() as f64);
    }
    estimates.sort_by(f64::total_cmp);
    Some(Interval {
        mean,
        lower: estimates[((repeats - 1) as f64 * tail).floor() as usize],
        upper: estimates[((repeats - 1) as f64 * (1.0 - tail)).ceil() as usize],
    })
}

fn compute_matches(a: &RunRecord, b: &RunRecord, tolerance: f64) -> bool {
    [
        (a.search.proposal_attempts, b.search.proposal_attempts),
        (a.search.statistics_records, b.search.statistics_records),
        (
            a.search.reserved_record_evaluations,
            b.search.reserved_record_evaluations,
        ),
        (
            a.search.reported_record_evaluations,
            b.search.reported_record_evaluations,
        ),
        (a.search.scorer_calls, b.search.scorer_calls),
    ]
    .into_iter()
    .all(|(a, b)| a.abs_diff(b) as f64 <= a.max(b) as f64 * tolerance)
}

pub fn summarise(spec: &Spec, records: &[RunRecord]) -> Summary {
    let mut comparisons = Vec::new();
    let expected = spec.creatures.len() * spec.seeds.len();
    let tail = 0.05 / (2 * 3 * spec.connection_targets.len()) as f64;
    for target in &spec.connection_targets {
        for comparator in [Arm::PerNeuron, Arm::PerDepth, Arm::Ockham] {
            let mut comparison = Comparison {
                comparator,
                requested_connections: *target,
                expected_pairs: expected,
                matched_pairs: 0,
                independent_units: 0,
                exclusions: BTreeMap::new(),
                pairs: Vec::new(),
                full_score_gain: None,
                held_out_score_gain: None,
                global_worst_output_mse_increase: None,
                passes_predeclared_criteria: false,
            };
            let mut clusters: BTreeMap<String, Vec<(f64, f64)>> = BTreeMap::new();
            for (subject, seed) in spec
                .creatures
                .iter()
                .flat_map(|s| spec.seeds.iter().map(move |seed| (s, seed)))
            {
                let global = records.iter().find(|r| {
                    r.arm == Arm::Global
                        && r.requested_connections == *target
                        && r.subject == subject.id
                        && r.seed == *seed
                });
                let Some(global) = global else {
                    *comparison
                        .exclusions
                        .entry("missing global arm".into())
                        .or_insert(0) += 1;
                    continue;
                };
                let other = records.iter().find(|r| {
                    r.arm == comparator
                        && r.subject == global.subject
                        && r.seed == global.seed
                        && r.requested_connections == *target
                });
                let reason = match other {
                    None => Some("missing arm"),
                    Some(other)
                        if global.error.is_some()
                            || other.error.is_some()
                            || global.search.failed_calls > 0
                            || other.search.failed_calls > 0
                            || global.search.timeout_exits > 0
                            || other.search.timeout_exits > 0 =>
                    {
                        Some("failed or timed out")
                    }
                    Some(other)
                        if global.full.is_none()
                            || other.full.is_none()
                            || global.held_out.is_none()
                            || other.held_out.is_none()
                            || global.diagnostics.is_none()
                            || other.diagnostics.is_none() =>
                    {
                        Some("missing evaluation")
                    }
                    Some(other) if global.achieved_connections != other.achieved_connections => {
                        Some("unmatched achieved connections")
                    }
                    Some(_) if global.achieved_connections >= global.opening_connections => {
                        Some("no connection saving")
                    }
                    Some(_) if global.accepted_connections != global.achieved_connections => {
                        Some("global proposal not accepted")
                    }
                    Some(other) if other.accepted_connections != other.achieved_connections => {
                        Some("comparator proposal not accepted")
                    }
                    Some(other)
                        if !compute_matches(
                            global,
                            other,
                            spec.success.compute_relative_tolerance,
                        ) =>
                    {
                        Some("unmatched compute")
                    }
                    Some(_) => None,
                };
                if let Some(reason) = reason {
                    *comparison.exclusions.entry(reason.into()).or_insert(0) += 1;
                    continue;
                }
                let other = other.unwrap();
                let full = global.full.as_ref().unwrap().proposal.score
                    - other.full.as_ref().unwrap().proposal.score;
                let held = global.held_out.as_ref().unwrap().proposal.score
                    - other.held_out.as_ref().unwrap().proposal.score;
                clusters
                    .entry(global.independent_unit.clone())
                    .or_default()
                    .push((full, held));
                comparison.pairs.push(PairedDifference {
                    subject: global.subject.clone(),
                    independent_unit: global.independent_unit.clone(),
                    seed: global.seed,
                    achieved_connections: global.achieved_connections,
                    full_score_gain: full,
                    held_out_score_gain: held,
                });
                comparison.matched_pairs += 1;
                let damage = global
                    .diagnostics
                    .as_ref()
                    .unwrap()
                    .worst_output_mse_increase;
                comparison.global_worst_output_mse_increase = Some(
                    comparison
                        .global_worst_output_mse_increase
                        .map_or(damage, |d| d.max(damage)),
                );
            }
            comparison.independent_units = clusters.len();
            let means: Vec<_> = clusters
                .values()
                .map(|rows| {
                    (
                        rows.iter().map(|r| r.0).sum::<f64>() / rows.len() as f64,
                        rows.iter().map(|r| r.1).sum::<f64>() / rows.len() as f64,
                    )
                })
                .collect();
            comparison.full_score_gain = interval(
                &means.iter().map(|r| r.0).collect::<Vec<_>>(),
                spec.success.bootstrap_replicates,
                tail,
            );
            comparison.held_out_score_gain = interval(
                &means.iter().map(|r| r.1).collect::<Vec<_>>(),
                spec.success.bootstrap_replicates,
                tail,
            );
            comparison.passes_predeclared_criteria = clusters.len()
                >= spec.success.minimum_independent_units
                && comparison.matched_pairs as f64 / expected as f64
                    >= spec.success.minimum_matched_fraction
                && comparison
                    .full_score_gain
                    .as_ref()
                    .is_some_and(|i| i.lower > spec.success.minimum_score_gain)
                && comparison
                    .held_out_score_gain
                    .as_ref()
                    .is_some_and(|i| i.lower > spec.success.minimum_score_gain)
                && comparison
                    .global_worst_output_mse_increase
                    .is_some_and(|d| d <= spec.success.maximum_worst_output_mse_increase);
            comparisons.push(comparison);
        }
    }
    let representative = if spec.study_kind == StudyKind::Representative {
        records
            .iter()
            .filter(|r| {
                r.error.is_none()
                    && r.full.is_some()
                    && r.held_out.is_some()
                    && r.diagnostics.is_some()
            })
            .map(|r| &r.independent_unit)
            .collect::<std::collections::BTreeSet<_>>()
            .len()
    } else {
        0
    };
    let supported = representative >= spec.success.minimum_independent_units
        && comparisons.iter().all(|c| c.passes_predeclared_criteria);
    let negative = representative >= spec.success.minimum_independent_units
        && comparisons.iter().any(|c| {
            c.independent_units >= spec.success.minimum_independent_units
                && c.matched_pairs as f64 / c.expected_pairs as f64
                    >= spec.success.minimum_matched_fraction
                && (c
                    .full_score_gain
                    .as_ref()
                    .is_some_and(|i| i.upper <= spec.success.minimum_score_gain)
                    || c.held_out_score_gain
                        .as_ref()
                        .is_some_and(|i| i.upper <= spec.success.minimum_score_gain)
                    || c.global_worst_output_mse_increase
                        .is_some_and(|d| d > spec.success.maximum_worst_output_mse_increase))
        });
    Summary {
        format_version: 1,
        conclusion: if supported {
            "supported-within-declared-study"
        } else if negative {
            "not-supported-within-declared-study"
        } else {
            "inconclusive"
        },
        study_kind: spec.study_kind,
        planned_runs: expected * spec.connection_targets.len() * 4,
        completed_runs: records.len(),
        failed_runs: records
            .iter()
            .filter(|r| {
                r.error.is_some() || r.search.timeout_exits > 0 || r.evaluation.timeout_exits > 0
            })
            .count(),
        timeouts: records
            .iter()
            .map(|r| r.search.timeout_exits + r.evaluation.timeout_exits)
            .sum(),
        target_misses: records.iter().filter(|r| !r.target_reached).count(),
        representative_independent_units: representative,
        uncertainty: "paired seed/model means clustered by independent unit; deterministic percentile bootstrap; Bonferroni familywise 95% across arms and targets; null for fewer than two units",
        timing_environment: "indicative measurements on a shared, parallel host; latency excludes scorer execution, IO and compilation",
        comparisons,
        freeze_ms: 0.0,
        verification_ms: 0.0,
        total_run_ms: 0.0,
        freeze_records_hashed: 0,
        verification_records_hashed: 0,
        search_statistics_records: records.iter().map(|r| r.search.statistics_records).sum(),
        search_proposal_attempts: records.iter().map(|r| r.search.proposal_attempts).sum(),
        scorer_reported_record_evaluations: records
            .iter()
            .map(|r| {
                r.search.reported_record_evaluations + r.evaluation.reported_record_evaluations
            })
            .sum(),
        scorer_reserved_record_evaluations: records
            .iter()
            .map(|r| {
                r.search.reserved_record_evaluations + r.evaluation.reserved_record_evaluations
            })
            .sum(),
        diagnostic_record_evaluations: records
            .iter()
            .filter_map(|r| r.diagnostics.as_ref())
            .map(|d| d.records * 2)
            .sum(),
        latency_record_evaluations: records
            .iter()
            .filter_map(|r| r.diagnostics.as_ref())
            .map(|d| {
                d.records.min(spec.latency.records as u64)
                    * (spec.latency.warmup_passes + spec.latency.measured_passes) as u64
                    * 2
            })
            .sum(),
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::*;
    use crate::manifest::Subject;
    use crate::measurement::{Diagnostics, Usage};
    use crate::pilot::specification;
    use crate::runner::Scores;
    use neat_ai_ockham::scorer::ScoreResult;

    fn spec() -> Spec {
        let subjects = ["a", "b"]
            .into_iter()
            .map(|id| Subject {
                id: id.into(),
                independent_unit: id.into(),
                provenance: "independent analysis fixture".into(),
                creature: "parent.json".into(),
                training: "training".into(),
                calibration: "calibration".into(),
                validation: "validation".into(),
            })
            .collect();
        let mut spec = specification(subjects, Path::new("scorer"));
        spec.seeds = vec![7, 19];
        spec.connection_targets = vec![6];
        spec.success.minimum_independent_units = 2;
        spec
    }

    fn rows(spec: &Spec, global_score: f64) -> Vec<RunRecord> {
        let score = |value| ScoreResult {
            score: value,
            error: 1.0 - value,
            complexity_penalty: 0.0,
            record_count: 4,
            sample_rate: None,
            gpu_backend: None,
            cost_name: Some("MSE".into()),
            time_taken: 0.0,
        };
        let mut rows = Vec::new();
        for subject in &spec.creatures {
            for seed in &spec.seeds {
                for arm in Arm::ALL {
                    let scores = Scores {
                        parent: score(0.5),
                        proposal: score(if arm == Arm::Global {
                            global_score
                        } else {
                            0.6
                        }),
                    };
                    rows.push(RunRecord {
                        format_version: 1,
                        manifest_sha256: "analysis-fixture".into(),
                        subject: subject.id.clone(),
                        independent_unit: subject.independent_unit.clone(),
                        seed: *seed,
                        arm,
                        requested_connections: 6,
                        opening_connections: 7,
                        achieved_connections: 6,
                        accepted_connections: 6,
                        target_reached: true,
                        stop_reason: "target-reached".into(),
                        error: None,
                        search: Usage {
                            proposal_attempts: 2,
                            statistics_records: 8,
                            scorer_calls: 2,
                            reserved_record_evaluations: 12,
                            reported_record_evaluations: 12,
                            ..Default::default()
                        },
                        evaluation: Usage::default(),
                        quotas: BTreeMap::new(),
                        region_connections: BTreeMap::new(),
                        opening_depth_connections: BTreeMap::new(),
                        depth_connections: BTreeMap::new(),
                        opening_neuron_connections: BTreeMap::new(),
                        neuron_connections: BTreeMap::new(),
                        opening_output_support: BTreeMap::new(),
                        output_support: BTreeMap::new(),
                        full: Some(scores.clone()),
                        held_out: Some(scores),
                        diagnostics: Some(Diagnostics {
                            records: 4,
                            per_output_mse_increase: vec![0.0],
                            worst_output_mse_increase: 0.0,
                            stratum_mse_increase: vec![Some(0.0); 4],
                            stratum_records: vec![1; 4],
                            parent_latency_ns_per_record: Vec::new(),
                            proposal_latency_ns_per_record: Vec::new(),
                            compilation_ms: 0.0,
                        }),
                        directory: "unused-analysis-fixture".into(),
                    });
                }
            }
        }
        rows
    }

    #[test]
    fn bootstrap_is_repeatable_and_requires_independent_units() {
        assert!(interval(&[1.0], 200, 0.025).is_none());
        let interval = interval(&[1.0, 3.0], 200, 0.025).unwrap();
        assert_eq!(
            (interval.mean, interval.lower, interval.upper),
            (2.0, 1.0, 3.0)
        );
    }

    #[test]
    fn repeated_seeds_are_clustered_and_synthetic_evidence_cannot_support_the_hypothesis() {
        let mut spec = spec();
        let mut rows = rows(&spec, 0.8);
        let report = summarise(&spec, &rows);
        assert_eq!(report.conclusion, "inconclusive");
        assert_eq!(report.comparisons[0].matched_pairs, 4);
        assert_eq!(report.comparisons[0].independent_units, 2);
        assert!(
            report
                .comparisons
                .iter()
                .all(|c| c.passes_predeclared_criteria)
        );
        for row in &mut rows {
            row.independent_unit = "same-unit".into();
        }
        let clustered = summarise(&spec, &rows);
        assert_eq!(clustered.comparisons[0].independent_units, 1);
        assert!(clustered.comparisons[0].full_score_gain.is_none());
        spec.study_kind = StudyKind::Representative;
        let report = summarise(&spec, &self::rows(&spec, 0.8));
        assert_eq!(report.conclusion, "supported-within-declared-study");
        let negative = summarise(&spec, &self::rows(&spec, 0.55));
        assert_eq!(negative.conclusion, "not-supported-within-declared-study");
    }

    #[test]
    fn comparisons_exclude_mismatches_failures_rejected_global_plans_and_missing_coverage() {
        let spec = spec();
        let base = rows(&spec, 0.8);
        for (case, expected) in [
            (0, "unmatched achieved connections"),
            (1, "unmatched compute"),
            (2, "failed or timed out"),
            (3, "missing evaluation"),
            (4, "no connection saving"),
            (5, "global proposal not accepted"),
            (6, "comparator proposal not accepted"),
        ] {
            let mut rows = base.clone();
            for row in rows.iter_mut().filter(|r| r.arm == Arm::Global) {
                match case {
                    0 => row.achieved_connections = 5,
                    1 => row.search.reported_record_evaluations = 100,
                    2 => row.error = Some("connection-budget: scorer failed".into()),
                    3 => row.held_out = None,
                    4 => row.achieved_connections = 7,
                    5 => row.accepted_connections = 7,
                    _ => {}
                }
            }
            if case == 6 {
                for row in rows.iter_mut().filter(|r| r.arm == Arm::PerNeuron) {
                    row.accepted_connections = 7;
                }
            }
            if case == 4 {
                for row in &mut rows {
                    row.achieved_connections = 7;
                }
            }
            let report = summarise(&spec, &rows);
            assert_eq!(report.comparisons[0].matched_pairs, 0, "{case}");
            assert_eq!(report.comparisons[0].exclusions[expected], 4, "{case}");
        }
        let missing = summarise(&spec, &[]);
        assert_eq!(missing.comparisons[0].exclusions["missing global arm"], 4);
        let without_control: Vec<_> = base.into_iter().filter(|r| r.arm == Arm::Global).collect();
        let missing = summarise(&spec, &without_control);
        assert_eq!(missing.comparisons[0].exclusions["missing arm"], 4);
    }
}

#[cfg(test)]
mod matching_tests {
    use super::*;
    use crate::measurement::Usage;

    #[test]
    fn each_compute_dimension_has_an_independent_ten_percent_boundary() {
        // Reuse real serialized records so this also checks the archived data contract.
        let row: RunRecord = serde_json::from_str(
            include_str!("../../../docs/research/connection-budget-v1/results.jsonl")
                .lines()
                .next()
                .unwrap(),
        )
        .unwrap();
        for dimension in 0..5 {
            let mut a = row.clone();
            a.search = Usage {
                proposal_attempts: 100,
                statistics_records: 100,
                reserved_record_evaluations: 100,
                reported_record_evaluations: 100,
                scorer_calls: 100,
                ..Default::default()
            };
            let mut b = a.clone();
            let counter = match dimension {
                0 => &mut b.search.proposal_attempts,
                1 => &mut b.search.statistics_records,
                2 => &mut b.search.reserved_record_evaluations,
                3 => &mut b.search.reported_record_evaluations,
                _ => &mut b.search.scorer_calls,
            };
            *counter = 90;
            assert!(compute_matches(&a, &b, 0.1));
            match dimension {
                0 => b.search.proposal_attempts = 89,
                1 => b.search.statistics_records = 89,
                2 => b.search.reserved_record_evaluations = 89,
                3 => b.search.reported_record_evaluations = 89,
                _ => b.search.scorer_calls = 89,
            }
            assert!(!compute_matches(&a, &b, 0.1));
        }
    }
}
