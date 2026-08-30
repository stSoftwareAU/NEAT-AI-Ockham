//! Summarise `experiments.jsonl` journals (Issue #10, extended by #11).
//!
//! Distinguishes **local cumulative Ockham improvement** from **population
//! headroom** when re-entry records exist. Tiny accepted steps are kept in the
//! cumulative trajectory rather than dismissed.
//!
//! Issue #11 adds the measures needed to compare one named ordering against
//! the random control on equal terms: time to the first authoritative local
//! winner, candidates screened before it, accepted-win size distribution,
//! scorer calls consumed and growth-cost reduction. Cumulative gain — not the
//! single largest cut — remains the headline.

use std::path::Path;

use serde::Serialize;

use crate::ablation::growth_units;
use crate::coverage::Coverage;
use crate::journal::Event;
use crate::ordering::Ordering;

/// Aggregate view of one or more journals.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Report {
    /// Journals consumed.
    pub journals: Vec<String>,
    /// Named ordering from the first start record.
    pub ordering: Option<Ordering>,
    /// Random exploration quota from the first start record.
    pub ordering_random_quota: Option<f64>,
    /// Opening authoritative score (first start record).
    pub opening_score: Option<f64>,
    /// Final authoritative score (last stop record).
    pub final_score: Option<f64>,
    /// `final - opening`.
    pub cumulative_delta: Option<f64>,
    /// Authoritative local accepts.
    pub accepts: u64,
    /// Sweep batches.
    pub experiments: u64,
    /// Full-cohort accepts (from `full` records).
    pub full_accepts: u64,
    /// Full-cohort rejects.
    pub full_rejects: u64,
    /// Sampled screen scorer calls consumed.
    pub screen_calls: u64,
    /// Full-corpus scorer cohort calls consumed.
    pub full_calls: u64,
    /// Screen-coverage records filed across the run (Issue #36).
    pub screened: u64,
    /// Hidden neurons on the incumbent at the last coverage record (Issue #37).
    pub hidden: Option<usize>,
    /// Hidden neurons carrying GRQ-provenance tags, at that same record (Issue #40).
    pub tagged: Option<usize>,
    /// `hidden - tagged`: the coverage denominator, at that same record (Issue #40).
    pub checkable: Option<usize>,
    /// Checkable UUIDs screened at least once, at that same record.
    pub checked: Option<usize>,
    /// Checkable UUIDs still never screened, at that same record (Issue #40).
    pub unchecked: Option<usize>,
    /// Hidden neurons cut by the run that wrote that record (Issue #40).
    pub cut: Option<usize>,
    /// `checked / checkable * 100` at that same record.
    pub coverage_percent: Option<f64>,
    /// Milliseconds from the loop start to the first authoritative local win.
    pub first_win_ms: Option<u64>,
    /// Candidates screened before the first authoritative local win.
    pub candidates_before_first_win: Option<u64>,
    /// Neurons cut by each accepted winner, in acceptance order.
    pub accepted_cut_sizes: Vec<usize>,
    /// Total neurons cut across every accepted winner.
    pub accepted_cuts: usize,
    /// Wall-clock milliseconds spent in the optimisation loop.
    pub elapsed_ms: Option<u64>,
    /// Authoritative local accepts per hour of loop wall-clock.
    pub accepts_per_hour: Option<f64>,
    /// Opening `hidden + synapses / 10` growth units.
    pub opening_growth_units: Option<f64>,
    /// Final `hidden + synapses / 10` growth units.
    pub final_growth_units: Option<f64>,
    /// `opening - final` growth units; positive means structure was removed.
    pub growth_units_saved: Option<f64>,
    /// Last stop reason.
    pub stop_reason: Option<String>,
    /// Effective seed from the first start record.
    pub seed: Option<u64>,
}

/// Read JSONL journals and fold them into a [`Report`].
pub fn summarise(paths: &[impl AsRef<Path>]) -> Result<Report, String> {
    let mut report = Report {
        journals: Vec::new(),
        ordering: None,
        ordering_random_quota: None,
        opening_score: None,
        final_score: None,
        cumulative_delta: None,
        accepts: 0,
        experiments: 0,
        full_accepts: 0,
        full_rejects: 0,
        screen_calls: 0,
        full_calls: 0,
        screened: 0,
        hidden: None,
        tagged: None,
        checkable: None,
        checked: None,
        unchecked: None,
        cut: None,
        coverage_percent: None,
        first_win_ms: None,
        candidates_before_first_win: None,
        accepted_cut_sizes: Vec::new(),
        accepted_cuts: 0,
        elapsed_ms: None,
        accepts_per_hour: None,
        opening_growth_units: None,
        final_growth_units: None,
        growth_units_saved: None,
        stop_reason: None,
        seed: None,
    };
    let mut candidates_seen = 0u64;
    for path in paths {
        let path = path.as_ref();
        report.journals.push(path.display().to_string());
        let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
        for (i, line) in text.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let event: Event = serde_json::from_str(line)
                .map_err(|e| format!("{}:{}: {e}", path.display(), i + 1))?;
            match event {
                Event::Start {
                    seed,
                    ordering,
                    ordering_random_quota,
                    hidden,
                    synapses,
                    opening_score,
                    ..
                } => {
                    if report.seed.is_none() {
                        report.seed = Some(seed);
                        report.ordering = Some(ordering);
                        report.ordering_random_quota = Some(ordering_random_quota);
                    }
                    if report.opening_score.is_none() {
                        report.opening_score = Some(opening_score);
                    }
                    if report.opening_growth_units.is_none() {
                        report.opening_growth_units = Some(growth_units(hidden, synapses));
                    }
                }
                Event::Batch { candidates, .. } => {
                    report.experiments += 1;
                    candidates_seen += candidates as u64;
                }
                Event::Screen { .. } => report.screen_calls += 1,
                Event::Screened { screened, .. } => report.screened += screened as u64,
                Event::Coverage {
                    hidden,
                    tagged,
                    checkable,
                    checked,
                    cut,
                } => {
                    // Coverage is a snapshot of one incumbent, not a total:
                    // the last record read is the current state. The percent
                    // comes from `Coverage` so the report can never disagree
                    // with the tag or the commit description.
                    let cov = Coverage {
                        hidden,
                        tagged,
                        checkable,
                        checked,
                        cut,
                    };
                    report.hidden = Some(cov.hidden);
                    report.tagged = Some(cov.tagged);
                    report.checkable = Some(cov.checkable);
                    report.checked = Some(cov.checked);
                    report.unchecked = Some(cov.unchecked());
                    report.cut = Some(cov.cut);
                    report.coverage_percent = Some(cov.percent());
                }
                Event::Full {
                    accepted,
                    cuts,
                    elapsed_ms,
                    ..
                } => {
                    report.full_calls += 1;
                    if accepted {
                        report.full_accepts += 1;
                        report.accepted_cut_sizes.push(cuts);
                        report.accepted_cuts += cuts;
                        if report.first_win_ms.is_none() {
                            report.first_win_ms = Some(elapsed_ms);
                            report.candidates_before_first_win = Some(candidates_seen);
                        }
                    } else {
                        report.full_rejects += 1;
                    }
                }
                Event::Stop {
                    reason,
                    accepts,
                    experiments,
                    final_score,
                    cumulative_delta,
                    final_hidden,
                    final_synapses,
                    elapsed_ms,
                } => {
                    report.stop_reason = Some(reason);
                    report.accepts = accepts;
                    if experiments > report.experiments {
                        report.experiments = experiments;
                    }
                    report.final_score = Some(final_score);
                    report.cumulative_delta = Some(cumulative_delta);
                    report.final_growth_units = Some(growth_units(final_hidden, final_synapses));
                    report.elapsed_ms = Some(elapsed_ms);
                }
            }
        }
    }
    if report.cumulative_delta.is_none()
        && let (Some(open), Some(final_score)) = (report.opening_score, report.final_score)
    {
        report.cumulative_delta = Some(final_score - open);
    }
    if let (Some(open), Some(fin)) = (report.opening_growth_units, report.final_growth_units) {
        report.growth_units_saved = Some(open - fin);
    }
    if let Some(ms) = report.elapsed_ms.filter(|ms| *ms > 0) {
        report.accepts_per_hour = Some(report.accepts as f64 * 3_600_000.0 / ms as f64);
    }
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::journal::{self, Event};

    fn start(ordering: Ordering) -> Event {
        Event::Start {
            seed: 7,
            ordering,
            ordering_random_quota: 0.25,
            permutation_identity: "x".into(),
            unchecked_first: false,
            hidden: 3,
            synapses: 10,
            opening_score: 0.50,
        }
    }

    fn full(accepted: bool, score: f64, cuts: usize, elapsed_ms: u64) -> Event {
        Event::Full {
            individuals: 1,
            bundles: 0,
            accepted,
            score: accepted.then_some(score),
            delta: accepted.then_some(0.000002),
            cuts,
            elapsed_ms,
        }
    }

    #[test]
    fn report_compounds_tiny_accepts_rather_than_only_the_final_score() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("experiments.jsonl");
        journal::append(&path, &start(Ordering::Random)).unwrap();
        journal::append(&path, &full(true, 0.500002, 1, 1_000)).unwrap();
        journal::append(&path, &full(true, 0.500004, 1, 2_000)).unwrap();
        journal::append(
            &path,
            &Event::Stop {
                reason: "timeout".into(),
                accepts: 2,
                experiments: 4,
                final_score: 0.500004,
                cumulative_delta: 0.000004,
                final_hidden: 1,
                final_synapses: 8,
                elapsed_ms: 3_600_000,
            },
        )
        .unwrap();
        let report = summarise(&[&path]).unwrap();
        assert_eq!(report.accepts, 2);
        assert_eq!(report.full_accepts, 2);
        assert!(report.cumulative_delta.unwrap() > 0.0);
        assert_eq!(report.opening_score, Some(0.50));
        assert_eq!(report.final_score, Some(0.500004));
        assert_eq!(report.seed, Some(7));
    }

    #[test]
    fn report_names_the_ordering_and_its_random_quota() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("experiments.jsonl");
        journal::append(&path, &start(Ordering::LowVariance)).unwrap();
        let report = summarise(&[&path]).unwrap();
        assert_eq!(report.ordering, Some(Ordering::LowVariance));
        assert_eq!(report.ordering_random_quota, Some(0.25));
        let json = serde_json::to_string(&report).unwrap();
        assert!(json.contains("\"ordering\":\"low-variance\""), "{json}");
    }

    #[test]
    fn report_measures_discovery_economics_not_just_the_largest_cut() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("experiments.jsonl");
        journal::append(&path, &start(Ordering::LowMeanAbs)).unwrap();
        for batch in 0..2u64 {
            journal::append(
                &path,
                &Event::Batch {
                    batch,
                    candidates: 40,
                    skipped: 2,
                    remaining: 100,
                },
            )
            .unwrap();
            journal::append(
                &path,
                &Event::Screen {
                    winners: 1,
                    losers: 39,
                    ms: 500,
                },
            )
            .unwrap();
            journal::append(&path, &full(false, 0.0, 0, 1_000 * (batch + 1))).unwrap();
        }
        // The third batch finally lands a tiny two-neuron bundle.
        journal::append(
            &path,
            &Event::Batch {
                batch: 2,
                candidates: 40,
                skipped: 0,
                remaining: 20,
            },
        )
        .unwrap();
        journal::append(
            &path,
            &Event::Screen {
                winners: 2,
                losers: 38,
                ms: 500,
            },
        )
        .unwrap();
        journal::append(&path, &full(true, 0.5001, 2, 9_000)).unwrap();
        journal::append(&path, &full(true, 0.5002, 1, 12_000)).unwrap();
        journal::append(
            &path,
            &Event::Stop {
                reason: "max-accepts".into(),
                accepts: 2,
                experiments: 3,
                final_score: 0.5002,
                cumulative_delta: 0.0002,
                final_hidden: 0,
                final_synapses: 5,
                elapsed_ms: 1_800_000,
            },
        )
        .unwrap();

        let report = summarise(&[&path]).unwrap();
        assert_eq!(report.screen_calls, 3);
        assert_eq!(report.full_calls, 4);
        assert_eq!(report.first_win_ms, Some(9_000));
        // 80 candidates screened across the two fruitless batches, then 40 more
        // in the batch that produced the win.
        assert_eq!(report.candidates_before_first_win, Some(120));
        assert_eq!(report.accepted_cut_sizes, vec![2, 1]);
        assert_eq!(report.accepted_cuts, 3);
        assert_eq!(report.elapsed_ms, Some(1_800_000));
        assert_eq!(report.accepts_per_hour, Some(4.0));
        assert_eq!(report.opening_growth_units, Some(growth_units(3, 10)));
        assert_eq!(report.final_growth_units, Some(growth_units(0, 5)));
        assert_eq!(report.growth_units_saved, Some(3.5));
    }

    #[test]
    fn screen_coverage_records_are_totalled_without_inflating_scorer_calls() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("experiments.jsonl");
        journal::append(&path, &start(Ordering::Random)).unwrap();
        journal::append(
            &path,
            &Event::Screen {
                winners: 1,
                losers: 3,
                ms: 100,
            },
        )
        .unwrap();
        journal::append(
            &path,
            &Event::Screened {
                batch: 0,
                screened: 4,
            },
        )
        .unwrap();
        journal::append(
            &path,
            &Event::Screened {
                batch: 1,
                screened: 2,
            },
        )
        .unwrap();
        let report = summarise(&[&path]).unwrap();
        assert_eq!(report.screened, 6);
        assert_eq!(report.screen_calls, 1, "coverage is not a scorer call");
    }

    #[test]
    fn the_last_coverage_record_becomes_the_reported_progress() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("experiments.jsonl");
        journal::append(&path, &start(Ordering::Random)).unwrap();
        journal::append(
            &path,
            &Event::Coverage {
                hidden: 12,
                tagged: 2,
                checkable: 10,
                checked: 2,
                cut: 0,
            },
        )
        .unwrap();
        // A later run screened more of the same creature.
        journal::append(
            &path,
            &Event::Coverage {
                hidden: 10,
                tagged: 2,
                checkable: 8,
                checked: 3,
                cut: 2,
            },
        )
        .unwrap();
        let report = summarise(&[&path]).unwrap();
        assert_eq!(report.hidden, Some(10));
        assert_eq!(report.checked, Some(3));
        assert_eq!(report.coverage_percent, Some(37.5));
        let json = serde_json::to_string(&report).unwrap();
        assert!(json.contains("\"coveragePercent\":37.5"), "{json}");
    }

    /// Every figure of the commit-description block reaches `report` (#40).
    #[test]
    fn the_report_carries_the_whole_coverage_block_not_just_the_percentage() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("experiments.jsonl");
        journal::append(&path, &start(Ordering::Random)).unwrap();
        journal::append(
            &path,
            &Event::Coverage {
                hidden: 5013,
                tagged: 42,
                checkable: 4971,
                checked: 1204,
                cut: 7,
            },
        )
        .unwrap();
        let report = summarise(&[&path]).unwrap();
        assert_eq!(report.tagged, Some(42));
        assert_eq!(report.checkable, Some(4971));
        assert_eq!(report.unchecked, Some(3767));
        assert_eq!(report.cut, Some(7));
        let json = serde_json::to_string(&report).unwrap();
        assert!(json.contains("\"unchecked\":3767"), "{json}");
    }

    #[test]
    fn a_journal_with_no_coverage_record_reports_no_block_figures() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("experiments.jsonl");
        journal::append(&path, &start(Ordering::Random)).unwrap();
        let report = summarise(&[&path]).unwrap();
        assert_eq!(report.tagged, None);
        assert_eq!(report.checkable, None);
        assert_eq!(report.unchecked, None, "no coverage state is not zero left");
        assert_eq!(report.cut, None);
    }

    #[test]
    fn a_journal_with_no_coverage_record_reports_no_coverage() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("experiments.jsonl");
        journal::append(&path, &start(Ordering::Random)).unwrap();
        let report = summarise(&[&path]).unwrap();
        assert_eq!(report.hidden, None);
        assert_eq!(report.checked, None);
        assert_eq!(report.coverage_percent, None, "no coverage state is not 0%");
    }

    #[test]
    fn a_journal_written_before_issue_36_parses_with_no_screen_coverage() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("experiments.jsonl");
        std::fs::write(
            &path,
            concat!(
                r#"{"record":"start","seed":3,"permutation_identity":"x","hidden":2,"opening_score":0.5}"#,
                "\n",
                r#"{"record":"batch","batch":0,"candidates":2,"skipped":0,"remaining":0}"#,
                "\n",
                r#"{"record":"screen","winners":1,"losers":1,"ms":50}"#,
                "\n",
                r#"{"record":"full","individuals":1,"bundles":0,"accepted":false}"#,
                "\n",
                r#"{"record":"stop","reason":"timeout","accepts":0,"experiments":1,"final_score":0.5,"cumulative_delta":0.0}"#,
                "\n",
            ),
        )
        .unwrap();
        let report = summarise(&[&path]).unwrap();
        assert_eq!(report.screen_calls, 1);
        assert_eq!(report.screened, 0, "old journals carry no coverage records");
        assert_eq!(report.hidden, None, "coverage is absent, not zero");
        assert_eq!(report.coverage_percent, None);
        assert_eq!(report.stop_reason.as_deref(), Some("timeout"));
    }

    #[test]
    fn a_journal_written_before_issue_11_still_parses_as_the_random_control() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("experiments.jsonl");
        std::fs::write(
            &path,
            concat!(
                r#"{"record":"start","seed":3,"permutation_identity":"x","hidden":2,"opening_score":0.5}"#,
                "\n",
                r#"{"record":"full","individuals":1,"bundles":0,"accepted":true,"score":0.6,"delta":0.1}"#,
                "\n",
                r#"{"record":"stop","reason":"timeout","accepts":1,"experiments":1,"final_score":0.6,"cumulative_delta":0.1}"#,
                "\n",
            ),
        )
        .unwrap();
        let report = summarise(&[&path]).unwrap();
        assert_eq!(report.ordering, Some(Ordering::Random));
        assert_eq!(report.accepts, 1);
        assert_eq!(report.first_win_ms, Some(0));
        assert_eq!(report.accepted_cut_sizes, vec![0]);
    }
}
