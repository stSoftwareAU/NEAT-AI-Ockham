//! Candidate feature/outcome telemetry — the learned model's training set (#107).
//!
//! One append-only JSON line per candidate the run actually spent scorer time
//! on: the feature vector the ranking saw, the sampled Δ, the full-corpus Δ
//! when one was measured, what the scorer decided, the structure the accept
//! removed and the scorer milliseconds it cost. That is everything
//! [`crate::model`] needs to fit a ranker offline, and nothing a run needs to
//! make a decision — the log is written after the verdict, never read during
//! one.
//!
//! Opt-in (`--candidate-log`), so a control run keeps its exact behaviour and
//! pays nothing for the feature extraction.
//!
//! Records are **self-describing**: the feature values are stored by name, and
//! each line carries its format version, the corpus identity, the incumbent
//! checksum and the ordering that produced the visit. A row whose features a
//! later schema no longer knows is skipped by [`training_rows`] with a count,
//! rather than being read against the wrong columns.

use std::collections::BTreeMap;
use std::io::{BufRead, BufReader, Write};
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::features::{CandidateFeatures, FEATURE_NAMES};
use crate::incumbent::now_unix;
use crate::model::TrainingRow;

/// Current candidate-log format version.
pub const CANDIDATE_LOG_FORMAT_VERSION: u32 = 1;

/// How far one candidate got.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum CandidateOutcome {
    /// The sampled screen did not promote it — no full-corpus verdict exists.
    ScreenedOut,
    /// Fully scored and not applied.
    Rejected,
    /// Fully scored and applied to the incumbent.
    Accepted,
}

/// Per-run stamp shared by every record the run writes.
#[derive(Debug, Clone, PartialEq)]
pub struct RunStamp {
    /// Host that produced the rows.
    pub host: String,
    /// Corpus identity the outcomes were measured against.
    pub corpus_identity: String,
    /// Incumbent checksum the features were read from.
    pub creature_checksum: String,
    /// Ordering that produced the visitation order.
    pub ordering: String,
    /// Run seed, so a row set can be traced back to the run that made it.
    pub seed: u64,
}

/// One candidate's features beside what the scorer made of it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CandidateRecord {
    /// [`CANDIDATE_LOG_FORMAT_VERSION`].
    pub version: u32,
    /// Unix seconds when the row was written.
    pub unix_secs: u64,
    /// Host that produced it.
    pub host: String,
    /// Corpus identity the outcome was measured against.
    pub corpus_identity: String,
    /// Incumbent checksum the features were read from.
    pub creature_checksum: String,
    /// Ordering the run used.
    pub ordering: String,
    /// Run seed.
    pub seed: u64,
    /// Hidden neuron the candidate cut.
    pub uuid: String,
    /// `identity` or `ablation`.
    pub kind: String,
    /// Feature values by name — the schema is carried, not assumed.
    pub features: BTreeMap<String, f64>,
    /// Sampled Δ against the incumbent scored in the same call.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sample_delta: Option<f64>,
    /// Full-corpus Δ when the candidate was scored individually.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub full_delta: Option<f64>,
    /// How far it got.
    pub outcome: CandidateOutcome,
    /// Growth units the accepted transform actually removed; `0` when nothing
    /// was applied, because nothing was removed.
    pub growth_units_removed: f64,
    /// Scorer wall time attributed to the stage that judged it (ms).
    pub scorer_ms: u64,
}

impl CandidateRecord {
    /// Build a record from `features` and the verdict the scorer returned.
    pub fn new(
        stamp: &RunStamp,
        uuid: &str,
        kind: &str,
        features: &CandidateFeatures,
        outcome: CandidateOutcome,
    ) -> Self {
        Self {
            version: CANDIDATE_LOG_FORMAT_VERSION,
            unix_secs: now_unix(),
            host: stamp.host.clone(),
            corpus_identity: stamp.corpus_identity.clone(),
            creature_checksum: stamp.creature_checksum.clone(),
            ordering: stamp.ordering.clone(),
            seed: stamp.seed,
            uuid: uuid.to_string(),
            kind: kind.to_string(),
            features: features
                .named()
                .into_iter()
                .map(|(name, value)| (name.to_string(), value))
                .collect(),
            sample_delta: None,
            full_delta: None,
            outcome,
            growth_units_removed: 0.0,
            scorer_ms: 0,
        }
    }

    /// Whether this row is a scorer-confirmed pruning win.
    ///
    /// An accept is one, and so is a candidate whose own full-corpus Δ cleared
    /// `min_improvement` while a better cut won its cohort — *confirmed but not
    /// applied* is a win the ranking should learn from, not a failure
    /// (Issue #52).
    pub fn is_win(&self, min_improvement: f64) -> bool {
        self.outcome == CandidateOutcome::Accepted
            || self.full_delta.is_some_and(|d| d > min_improvement)
    }

    /// The feature vector in [`FEATURE_NAMES`] order, or `None` when this row
    /// does not carry every feature the current schema ranks on.
    pub fn vector(&self) -> Option<Vec<f64>> {
        FEATURE_NAMES
            .iter()
            .map(|name| self.features.get(*name).copied())
            .collect()
    }
}

/// Append `records` to `path` as one JSON line each.
///
/// Each line is written with a single `write_all`, so an interrupted run leaves
/// a valid prefix — the same contract [`crate::journal`] keeps.
pub fn append(path: &Path, records: &[CandidateRecord]) -> Result<(), String> {
    if records.is_empty() {
        return Ok(());
    }
    if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent).map_err(|e| format!("{}: {e}", parent.display()))?;
    }
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .map_err(|e| format!("{}: {e}", path.display()))?;
    for record in records {
        let mut line = serde_json::to_string(record).map_err(|e| e.to_string())?;
        line.push('\n');
        file.write_all(line.as_bytes())
            .map_err(|e| format!("{}: {e}", path.display()))?;
    }
    Ok(())
}

/// Read every record in `path`.
///
/// A line that does not parse is a corrupt training set, not a row to skip
/// quietly: it errors, naming the file and the line.
pub fn load(path: &Path) -> Result<Vec<CandidateRecord>, String> {
    let file = std::fs::File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut records = Vec::new();
    for (i, line) in BufReader::new(file).lines().enumerate() {
        let line = line.map_err(|e| format!("{}:{}: {e}", path.display(), i + 1))?;
        if line.trim().is_empty() {
            continue;
        }
        let record: CandidateRecord = serde_json::from_str(&line)
            .map_err(|e| format!("{}:{}: {e}", path.display(), i + 1))?;
        if record.version != CANDIDATE_LOG_FORMAT_VERSION {
            return Err(format!(
                "{}:{}: candidate-log format version {} is not {CANDIDATE_LOG_FORMAT_VERSION}",
                path.display(),
                i + 1,
                record.version
            ));
        }
        records.push(record);
    }
    Ok(records)
}

/// Training rows, and how many records the current schema could not read.
///
/// The skipped count is returned rather than logged away: a training set that
/// silently shrank to a handful of rows would fit a model nobody could account
/// for.
pub fn training_rows(
    records: &[CandidateRecord],
    min_improvement: f64,
) -> (Vec<TrainingRow>, usize) {
    let mut rows = Vec::with_capacity(records.len());
    let mut skipped = 0;
    for record in records {
        match record.vector() {
            Some(features) if features.iter().all(|v| v.is_finite()) => rows.push(TrainingRow {
                features,
                win: record.is_win(min_improvement),
            }),
            _ => skipped += 1,
        }
    }
    (rows, skipped)
}

/// Corpus identities present in `records`, sorted — training-set provenance.
pub fn corpora(records: &[CandidateRecord]) -> Vec<String> {
    let mut seen: Vec<String> = records
        .iter()
        .map(|r| r.corpus_identity.clone())
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect();
    seen.sort();
    seen
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stamp() -> RunStamp {
        RunStamp {
            host: "GRQ-1".into(),
            corpus_identity: "corpus-a".into(),
            creature_checksum: "abc".into(),
            ordering: "composite".into(),
            seed: 42,
        }
    }

    fn features() -> CandidateFeatures {
        CandidateFeatures {
            measured: true,
            variance: 0.5,
            mean_abs: 0.25,
            range: 1.0,
            outgoing_weight: 2.0,
            fan_in: 1,
            fan_out: 2,
            direct_growth_units: 1.3,
            cascade_growth_units: 3.4,
            identity: true,
            blocked: false,
            depth_fraction: 0.5,
            prior_wins: 1,
            prior_failures: 0,
        }
    }

    fn record(outcome: CandidateOutcome) -> CandidateRecord {
        CandidateRecord::new(&stamp(), "h1", "ablation", &features(), outcome)
    }

    fn temp_dir(name: &str) -> std::path::PathBuf {
        let dir =
            std::env::temp_dir().join(format!("ockham-telemetry-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        dir
    }

    #[test]
    fn a_record_carries_every_feature_by_name() {
        let record = record(CandidateOutcome::Rejected);
        assert_eq!(record.features.len(), FEATURE_NAMES.len());
        for name in FEATURE_NAMES {
            assert!(record.features.contains_key(*name), "missing {name}");
        }
        assert_eq!(record.vector().unwrap(), features().vector());
        assert_eq!(record.version, CANDIDATE_LOG_FORMAT_VERSION);
    }

    #[test]
    fn records_round_trip_through_the_log() {
        let dir = temp_dir("round-trip");
        let path = dir.join("candidates.jsonl");
        let mut screened = record(CandidateOutcome::ScreenedOut);
        screened.sample_delta = Some(-0.2);
        screened.scorer_ms = 40;
        let mut accepted = record(CandidateOutcome::Accepted);
        accepted.sample_delta = Some(0.3);
        accepted.full_delta = Some(0.02);
        accepted.growth_units_removed = 3.4;
        accepted.scorer_ms = 900;
        append(&path, &[screened.clone(), accepted.clone()]).unwrap();
        append(&path, &[]).unwrap();
        let read = load(&path).unwrap();
        assert_eq!(read, vec![screened, accepted]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_corrupt_line_fails_loud_rather_than_being_skipped() {
        let dir = temp_dir("corrupt");
        let path = dir.join("candidates.jsonl");
        append(&path, &[record(CandidateOutcome::Rejected)]).unwrap();
        let mut file = std::fs::OpenOptions::new()
            .append(true)
            .open(&path)
            .unwrap();
        file.write_all(b"{not json}\n").unwrap();
        let err = load(&path).unwrap_err();
        assert!(err.contains("candidates.jsonl:2"), "{err}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_record_from_another_format_version_is_refused() {
        let dir = temp_dir("version");
        let path = dir.join("candidates.jsonl");
        let mut old = record(CandidateOutcome::Rejected);
        old.version = CANDIDATE_LOG_FORMAT_VERSION + 1;
        append(&path, &[old]).unwrap();
        let err = load(&path).unwrap_err();
        assert!(err.contains("format version"), "{err}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn confirmed_but_not_applied_counts_as_a_win() {
        let mut confirmed = record(CandidateOutcome::Rejected);
        confirmed.full_delta = Some(0.01);
        assert!(confirmed.is_win(1e-6));
        let mut loser = record(CandidateOutcome::Rejected);
        loser.full_delta = Some(-0.01);
        assert!(!loser.is_win(1e-6));
        assert!(record(CandidateOutcome::Accepted).is_win(1e-6));
        assert!(!record(CandidateOutcome::ScreenedOut).is_win(1e-6));
    }

    #[test]
    fn training_rows_count_what_the_schema_cannot_read() {
        let good = record(CandidateOutcome::Accepted);
        let mut missing = record(CandidateOutcome::Rejected);
        missing.features.remove(FEATURE_NAMES[1]);
        let mut infinite = record(CandidateOutcome::Rejected);
        infinite
            .features
            .insert(FEATURE_NAMES[1].to_string(), f64::INFINITY);
        let (rows, skipped) = training_rows(&[good, missing, infinite], 1e-6);
        assert_eq!(rows.len(), 1);
        assert_eq!(skipped, 2);
        assert!(rows[0].win);
    }

    #[test]
    fn provenance_lists_every_corpus_the_rows_came_from() {
        let mut other = record(CandidateOutcome::Rejected);
        other.corpus_identity = "corpus-b".into();
        let rows = vec![record(CandidateOutcome::Accepted), other];
        assert_eq!(corpora(&rows), ["corpus-a", "corpus-b"]);
    }
}
