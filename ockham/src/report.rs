//! Summarise `experiments.jsonl` journals (Issue #10).
//!
//! Distinguishes **local cumulative Ockham improvement** from **population
//! headroom** when re-entry records exist. Tiny accepted steps are kept in the
//! cumulative trajectory rather than dismissed.

use std::path::Path;

use serde::Serialize;

use crate::journal::Event;

/// Aggregate view of one or more journals.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Report {
    /// Journals consumed.
    pub journals: Vec<String>,
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
    /// Last stop reason.
    pub stop_reason: Option<String>,
    /// Effective seed from the first start record.
    pub seed: Option<u64>,
}

/// Read JSONL journals and fold them into a [`Report`].
pub fn summarise(paths: &[impl AsRef<Path>]) -> Result<Report, String> {
    let mut report = Report {
        journals: Vec::new(),
        opening_score: None,
        final_score: None,
        cumulative_delta: None,
        accepts: 0,
        experiments: 0,
        full_accepts: 0,
        full_rejects: 0,
        stop_reason: None,
        seed: None,
    };
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
                    opening_score,
                    ..
                } => {
                    if report.seed.is_none() {
                        report.seed = Some(seed);
                    }
                    if report.opening_score.is_none() {
                        report.opening_score = Some(opening_score);
                    }
                }
                Event::Batch { .. } => report.experiments += 1,
                Event::Full { accepted, .. } => {
                    if accepted {
                        report.full_accepts += 1;
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
                } => {
                    report.stop_reason = Some(reason);
                    report.accepts = accepts;
                    if experiments > report.experiments {
                        report.experiments = experiments;
                    }
                    report.final_score = Some(final_score);
                    report.cumulative_delta = Some(cumulative_delta);
                }
                Event::Screen { .. } => {}
            }
        }
    }
    if report.cumulative_delta.is_none()
        && let (Some(open), Some(final_score)) = (report.opening_score, report.final_score)
    {
        report.cumulative_delta = Some(final_score - open);
    }
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::journal::{self, Event};

    #[test]
    fn report_compounds_tiny_accepts_rather_than_only_the_final_score() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("experiments.jsonl");
        journal::append(
            &path,
            &Event::Start {
                seed: 7,
                permutation_identity: "x".into(),
                hidden: 3,
                opening_score: 0.50,
            },
        )
        .unwrap();
        journal::append(
            &path,
            &Event::Full {
                individuals: 1,
                bundles: 0,
                accepted: true,
                score: Some(0.500002),
                delta: Some(0.000002),
            },
        )
        .unwrap();
        journal::append(
            &path,
            &Event::Full {
                individuals: 1,
                bundles: 0,
                accepted: true,
                score: Some(0.500004),
                delta: Some(0.000002),
            },
        )
        .unwrap();
        journal::append(
            &path,
            &Event::Stop {
                reason: "timeout".into(),
                accepts: 2,
                experiments: 4,
                final_score: 0.500004,
                cumulative_delta: 0.000004,
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
}
