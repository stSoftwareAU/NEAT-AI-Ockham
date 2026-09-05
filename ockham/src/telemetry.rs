//! Candidate feature/outcome telemetry — the learned model's training set (#107).
//!
//! One append-only JSON line per **sweep** candidate the scorer judged: the
//! feature vector the ranking saw, the sampled Δ, the full-corpus Δ when one was
//! measured, what the scorer decided, the structure the accept removed and the
//! scorer milliseconds it cost. That is everything [`crate::model`] needs to fit
//! a ranker offline, and nothing a run needs to make a decision — the log is
//! written after the verdict, never read during one.
//!
//! The **replay** stages are deliberately not logged. A replayed candidate is a
//! uuid the learnings cache already called a winner (Issues #52, #101), so its
//! outcomes are drawn from a population the ranking did not choose and would
//! teach a ranker that its own past wins predict future ones. The rows here are
//! exactly the candidates an ordering picked out of the sweep, which is the
//! decision the model is being fitted to make.
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

use neat_core::CreatureExport;

use crate::features::{CandidateFeatures, FEATURE_NAMES};
use crate::incumbent::now_unix;
use crate::model::TrainingRow;
use crate::stats::ActivationStats;

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
    /// Fully scored and applied to the incumbent, alone or inside a winning
    /// bundle.
    ///
    /// A bundle member was applied without its own delta deciding it, so
    /// [`CandidateRecord::full_delta`] — what the scorer measured for the neuron
    /// alone — is what [`CandidateRecord::is_win`] reads when it is present.
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
    /// How the sweep built the candidate: `identity`, `ablation`, `constant`
    /// or `merge`.
    pub kind: String,
    /// Survivor that absorbed the neuron, for a `merge` candidate (#109).
    ///
    /// The pair is the provenance a merge needs and no other kind has: the same
    /// uuid removed against a different survivor is a different cut. Additive
    /// and optional, so a row written by an older host still loads.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub merged_with: Option<String>,
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
            merged_with: None,
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
    /// The scorer's own measurement of *this* neuron decides whenever it exists:
    /// a candidate whose full-corpus Δ cleared `min_improvement` while a better
    /// cut won its cohort is a win the ranking should learn from — *confirmed
    /// but not applied* is not a failure (Issue #52) — and a bundle member that
    /// rode a winning bundle to acceptance on a Δ of its own below the threshold
    /// is not a win, whatever the bundle did.
    ///
    /// With no individual Δ measured, an accept is the only evidence there is.
    pub fn is_win(&self, min_improvement: f64) -> bool {
        match self.full_delta {
            Some(delta) => delta > min_improvement,
            None => self.outcome == CandidateOutcome::Accepted,
        }
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

/// Opt-in candidate feature/outcome telemetry (Issue #107).
///
/// Written **after** a verdict and never read during one: this is the training
/// set the learned ranker is fitted from offline, and nothing here can promote,
/// suppress or accept a cut. A store fault warns rather than ending the run —
/// the log is evidence about the search, not part of it — but it warns loudly,
/// because a training set that silently stopped growing is worse than no log.
pub struct CandidateLog<'a> {
    /// File the rows are appended to.
    pub path: &'a Path,
    /// Per-run stamp every row carries.
    pub stamp: RunStamp,
    /// Historical evidence the ranking read, so the logged features are the
    /// ones the ordering actually saw.
    pub evidence: &'a crate::features::PriorEvidence,
}

impl CandidateLog<'_> {
    /// Rows for the candidates the sampled screen did not promote.
    ///
    /// `checksum` is the incumbent the features are read from, not the one the
    /// run opened with: an accept moves the incumbent, and a row stamped with a
    /// checksum for a creature it was never extracted from cannot be traced
    /// back to the topology that produced it.
    ///
    /// The screen time is the whole cohort's — winners, losers and the
    /// incumbent scored in one call — so it is shared across `cohort` creatures
    /// rather than charged to the losers alone, which would inflate the column
    /// every time the screen promoted anything.
    pub fn screened_out(
        &self,
        creature: &CreatureExport,
        stats: &ActivationStats,
        checksum: &str,
        losers: &[crate::sweep::ScreenedLoser],
        screen_ms: u64,
        cohort: usize,
    ) {
        if losers.is_empty() {
            return;
        }
        let features = crate::features::extract(creature, stats, self.evidence);
        let each_ms = screen_ms / cohort.max(losers.len()).max(1) as u64;
        let mut records = Vec::with_capacity(losers.len());
        let mut unknown = 0usize;
        for loser in losers {
            let Some(f) = features.get(&loser.uuid) else {
                unknown += 1;
                continue;
            };
            let mut record = CandidateRecord::new(
                &self.stamp,
                &loser.uuid,
                crate::learnings::kind_label(loser.kind),
                f,
                CandidateOutcome::ScreenedOut,
            );
            record.creature_checksum = checksum.to_string();
            record.merged_with = loser.merged_with.clone();
            record.sample_delta = Some(loser.delta);
            record.scorer_ms = each_ms;
            records.push(record);
        }
        self.write(&records, unknown);
    }

    /// Rows for the candidates the full corpus judged individually.
    ///
    /// Only individually scored uuids the sweep proposed are logged: a uuid that
    /// appeared solely inside a bundle had no contribution of its own measured,
    /// and a row carrying the bundle's delta as if it were the neuron's would
    /// teach the ranker something the scorer never said (the reasoning
    /// `file_full_outcome` applies to `full_delta`).
    pub fn judged(
        &self,
        creature: &CreatureExport,
        stats: &ActivationStats,
        checksum: &str,
        sampled: &[crate::sweep::SampledWinner],
        full: &crate::promote::FullOutcome,
    ) {
        if full.individuals.is_empty() {
            return;
        }
        let features = crate::features::extract(creature, stats, self.evidence);
        let before = crate::ablation::StructureSnapshot::of(creature);
        // The uuids of an accepted **individual** — a bundle's saving is shared
        // structure no member removed on its own, so it is attributed to none of
        // them rather than to each of them.
        let solo_win: Option<&crate::promote::FullCandidate> = full
            .winner
            .as_ref()
            .map(|w| &w.candidate)
            .filter(|c| c.uuids.len() == 1);
        let winner: std::collections::HashSet<&str> = full
            .winner
            .as_ref()
            .map(|w| w.candidate.uuids.iter().map(String::as_str).collect())
            .unwrap_or_default();
        let each_ms = full.full_ms / full.entries().max(1) as u64;
        let mut records = Vec::with_capacity(full.individuals.len());
        let mut unknown = 0usize;
        for scored in &full.individuals {
            // A cohort entry naming no uuid describes no neuron; counted, not
            // dropped in silence.
            let Some(uuid) = scored.uuids.first() else {
                unknown += 1;
                continue;
            };
            // The kind is the sweep's — `identity`, `ablation`, `constant` or
            // `merge` — never the cohort's `individual`, so one column carries
            // one vocabulary. A uuid this run's sweep did not propose (a carried
            // winner from an earlier batch) has no candidate kind to record and
            // is left for the batch that did propose it.
            // The group candidate keyed on this uuid is not the row's
            // candidate (#108): a group's kind and sampled delta describe the
            // whole neighbourhood, and this row is about one neuron the scorer
            // judged alone.
            let (Some(f), Some(candidate)) = (
                features.get(uuid),
                sampled
                    .iter()
                    .find(|w| !w.candidate.is_group() && &w.candidate.uuid == uuid),
            ) else {
                unknown += 1;
                continue;
            };
            let accepted = winner.contains(uuid.as_str());
            let mut record = CandidateRecord::new(
                &self.stamp,
                uuid,
                crate::learnings::kind_label(candidate.candidate.kind),
                f,
                if accepted {
                    CandidateOutcome::Accepted
                } else {
                    CandidateOutcome::Rejected
                },
            );
            record.creature_checksum = checksum.to_string();
            record.merged_with = candidate.candidate.merged_with.clone();
            record.sample_delta = Some(candidate.delta);
            record.full_delta = Some(scored.delta);
            // Nothing is removed by a candidate that was not applied, so a
            // rejected row records the zero it actually saved; the saving it
            // *would* have made is already in its features.
            record.growth_units_removed = match solo_win {
                Some(win) if win.uuids.first() == Some(uuid) => {
                    before.growth_units - win.after.growth_units
                }
                _ => 0.0,
            };
            record.scorer_ms = each_ms;
            records.push(record);
        }
        self.write(&records, unknown);
    }

    /// Append `records`, saying what was written and what could not be.
    ///
    /// `unknown` counts candidates with no feature vector or no sweep candidate
    /// to name their kind. It is reported rather than dropped quietly: a
    /// training set that silently shrank fits a model nobody can account for.
    fn write(&self, records: &[CandidateRecord], unknown: usize) {
        if unknown > 0 {
            crate::log::warn(&format!(
                "candidate log: {unknown} judged candidate(s) carried no feature vector; \
                 their outcomes are not in the training set"
            ));
        }
        match append(self.path, records) {
            Ok(()) if !records.is_empty() => crate::log::detail(&format!(
                "candidate log: {} row(s) appended to {}",
                records.len(),
                self.path.display()
            )),
            Ok(()) => {}
            Err(e) => crate::log::warn(&format!(
                "candidate log unwritable ({e}); {} training row(s) lost",
                records.len()
            )),
        }
    }
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

    /// Issue #109: the pair is the provenance a merge needs, so it has to reach
    /// the candidate log — a survivor recorded nowhere leaves the audit trail
    /// naming which neuron went but not what it was merged into.
    #[test]
    fn a_screened_out_merge_writes_its_survivor_to_the_log() {
        use crate::sweep::{CandidateKind, ScreenRejection, ScreenedLoser};

        let dir = temp_dir("merge-provenance");
        let path = dir.join("candidates.jsonl");
        let creature = crate::fixtures::wide_creature(1, 2, "TANH");
        let stats = crate::stats::ActivationStats {
            neurons: creature
                .neurons
                .iter()
                .enumerate()
                .filter(|(_, n)| n.neuron_type == "hidden")
                .map(|(i, n)| crate::stats::NeuronStats {
                    uuid: n.uuid.clone(),
                    neuron_index: i,
                    count: 10,
                    mean: 0.0,
                    variance: 1.0,
                    std_dev: 1.0,
                    mean_abs: 0.5,
                    min: -1.0,
                    max: 1.0,
                })
                .collect(),
            ..crate::stats::ActivationStats::empty()
        };
        let evidence = crate::features::PriorEvidence::new();
        let log = CandidateLog {
            path: &path,
            stamp: stamp(),
            evidence: &evidence,
        };
        log.screened_out(
            &creature,
            &stats,
            "checksum-1",
            &[
                ScreenedLoser {
                    uuid: "h0".into(),
                    kind: CandidateKind::Merge,
                    merged_with: Some("h1".into()),
                    delta: -0.4,
                    stage: 1,
                    reason: ScreenRejection::BelowThreshold,
                },
                ScreenedLoser {
                    uuid: "h1".into(),
                    kind: CandidateKind::Ablation,
                    merged_with: None,
                    delta: -0.5,
                    stage: 1,
                    reason: ScreenRejection::BelowThreshold,
                },
            ],
            100,
            2,
        );

        let rows = load(&path).unwrap();
        let merged = rows.iter().find(|r| r.uuid == "h0").expect("merge row");
        assert_eq!(merged.kind, "merge");
        assert_eq!(
            merged.merged_with.as_deref(),
            Some("h1"),
            "the survivor must survive the round trip: {merged:?}"
        );
        let ablated = rows.iter().find(|r| r.uuid == "h1").expect("ablation row");
        assert_eq!(ablated.kind, "ablation");
        assert!(
            ablated.merged_with.is_none(),
            "only a merge names a survivor"
        );
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
    fn an_unwritable_log_path_fails_loud_rather_than_dropping_rows() {
        let dir = temp_dir("unwritable");
        std::fs::create_dir_all(&dir).unwrap();
        // A file where the parent directory should be: appending cannot work,
        // and the error must name the path rather than lose the rows quietly.
        let blocker = dir.join("blocked");
        std::fs::write(&blocker, "not a directory").unwrap();
        let err = append(
            &blocker.join("candidates.jsonl"),
            &[record(CandidateOutcome::Rejected)],
        )
        .unwrap_err();
        assert!(err.contains("blocked"), "{err}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_missing_log_names_the_file_it_could_not_read() {
        let err = load(Path::new("/nonexistent/ockham-candidates.jsonl")).unwrap_err();
        assert!(err.contains("ockham-candidates.jsonl"), "{err}");
    }

    #[test]
    fn provenance_lists_every_corpus_the_rows_came_from() {
        let mut other = record(CandidateOutcome::Rejected);
        other.corpus_identity = "corpus-b".into();
        let rows = vec![record(CandidateOutcome::Accepted), other];
        assert_eq!(corpora(&rows), ["corpus-a", "corpus-b"]);
    }
}
