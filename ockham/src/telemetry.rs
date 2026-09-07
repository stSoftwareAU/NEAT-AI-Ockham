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

/// What one candidate kind was worth to a run (Issue #138).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct KindCounts {
    /// Candidates of this kind the run judged, whatever became of them.
    pub proposals: usize,
    /// Of those, the ones the full corpus accepted.
    pub accepts: usize,
    /// Of those, the ones that reached the training set as a row.
    pub logged: usize,
}

/// Candidate kind that carries no training row **by construction** (#138).
///
/// Every feature in the vector is a hidden neuron's — fan-in, outgoing weight,
/// depth, cascade estimate — and a synapse candidate names an edge, so there is
/// nothing to key a row on. Its absence from the training set is the design,
/// not a fault, which is why it is named here and excluded from the anomaly
/// warning rather than inflating it on every batch.
const ROWLESS_KIND: &str = "synapse";

/// Proposals, accepts and logged rows per candidate kind (Issue #138).
///
/// The candidate log is keyed by a **hidden-neuron** feature vector, and a
/// synapse candidate names an edge rather than a neuron — so it carries no
/// vector and no training row. Counting what was offered beside what was
/// written is what stops an edge cut vanishing from the telemetry altogether:
/// every synapse proposal and every synapse accept is reported here, whether or
/// not the training set could hold it.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct KindTally {
    counts: BTreeMap<String, KindCounts>,
    anomalies: usize,
}

impl KindTally {
    /// Record one judged candidate of `kind`.
    pub fn offer(&mut self, kind: &str, accepted: bool, logged: bool) {
        let counts = self.counts.entry(kind.to_string()).or_default();
        counts.proposals += 1;
        counts.accepts += usize::from(accepted);
        counts.logged += usize::from(logged);
    }

    /// Record a cohort entry whose **kind could not be named** at all.
    ///
    /// An entry naming no uuid, or a uuid this run's sweep did not propose.
    /// Kept out of the kind map on purpose: `unnamed` and `unproposed` are not
    /// candidate kinds, and rendering them beside `ablation` and `synapse`
    /// would put two vocabularies in one column. They are still counted, and
    /// they are what the anomaly warning is really for.
    pub fn anomaly(&mut self) {
        self.anomalies += 1;
    }

    /// What `kind` was worth; all-zero for a kind the run never proposed.
    pub fn counts(&self, kind: &str) -> KindCounts {
        self.counts.get(kind).copied().unwrap_or_default()
    }

    /// Cohort entries whose kind could not be named.
    pub fn anomalies(&self) -> usize {
        self.anomalies
    }

    /// Whether nothing at all was tallied.
    pub fn is_empty(&self) -> bool {
        self.counts.is_empty() && self.anomalies == 0
    }

    /// Candidates of every kind that carried no training row.
    pub fn unlogged(&self) -> usize {
        self.counts
            .values()
            .map(|c| c.proposals - c.logged)
            .sum::<usize>()
            + self.anomalies
    }

    /// Candidates that carried no row **where one was expected**.
    ///
    /// [`Self::unlogged`] minus the kinds that can never hold one. This is the
    /// figure the warning fires on: a training set that silently stopped
    /// growing is the fault worth shouting about, and warning on an edge cut —
    /// which by design holds no row — would drown that signal in noise.
    pub fn unexpected_unlogged(&self) -> usize {
        self.counts
            .iter()
            .filter(|(kind, _)| kind.as_str() != ROWLESS_KIND)
            .map(|(_, c)| c.proposals - c.logged)
            .sum::<usize>()
            + self.anomalies
    }

    /// `kind proposed/accepted/logged` per kind, in kind order.
    pub fn summary(&self) -> String {
        let mut parts: Vec<String> = self
            .counts
            .iter()
            .map(|(kind, c)| {
                format!(
                    "{kind} {} proposed, {} accepted, {} logged",
                    c.proposals, c.accepts, c.logged
                )
            })
            .collect();
        if self.anomalies > 0 {
            parts.push(format!("{} of no nameable kind", self.anomalies));
        }
        parts.join(" · ")
    }
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
        let mut tally = KindTally::default();
        for loser in losers {
            let kind = crate::learnings::kind_label(loser.kind);
            let Some(f) = features.get(&loser.uuid) else {
                tally.offer(kind, false, false);
                continue;
            };
            let mut record = CandidateRecord::new(
                &self.stamp,
                &loser.uuid,
                kind,
                f,
                CandidateOutcome::ScreenedOut,
            );
            record.creature_checksum = checksum.to_string();
            record.merged_with = loser.merged_with.clone();
            record.sample_delta = Some(loser.delta);
            record.scorer_ms = each_ms;
            records.push(record);
            tally.offer(kind, false, true);
        }
        self.write(&records, &tally);
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
        let mut tally = KindTally::default();
        for scored in &full.individuals {
            // A cohort entry naming no uuid describes no visit; counted, not
            // dropped in silence.
            let Some(uuid) = scored.uuids.first() else {
                tally.anomaly();
                continue;
            };
            let accepted = winner.contains(uuid.as_str());
            // The kind is the sweep's — `identity`, `ablation`, `constant`,
            // `merge` or `synapse` — never the cohort's `individual`, so one
            // column carries one vocabulary. A uuid this run's sweep did not
            // propose (a carried winner from an earlier batch) has no candidate
            // kind to record and is left for the batch that did propose it.
            // The group candidate keyed on this uuid is not the row's
            // candidate (#108): a group's kind and sampled delta describe the
            // whole neighbourhood, and this row is about one visit the scorer
            // judged alone.
            let Some(candidate) = sampled
                .iter()
                .find(|w| !w.candidate.is_group() && &w.candidate.uuid == uuid)
            else {
                // A carried winner from an earlier batch: this run's sweep did
                // not propose it, so there is no kind to record it under.
                tally.anomaly();
                continue;
            };
            let kind = crate::learnings::kind_label(candidate.candidate.kind);
            // A synapse candidate names an **edge**, so the hidden-neuron
            // feature vectors never carry one and it can hold no training row
            // (#138). The tally above still reports the proposal and the
            // accept, which is what keeps an edge cut visible in the telemetry.
            let Some(f) = features.get(uuid) else {
                tally.offer(kind, accepted, false);
                continue;
            };
            let mut record = CandidateRecord::new(
                &self.stamp,
                uuid,
                kind,
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
            tally.offer(kind, accepted, true);
        }
        self.write(&records, &tally);
    }

    /// Append `records`, saying what was written and what could not be.
    ///
    /// `tally` is every candidate the caller judged, by kind. It is reported
    /// rather than dropped quietly: a training set that silently shrank fits a
    /// model nobody can account for, and an edge cut that held no row would
    /// otherwise leave no trace at all (#138).
    fn write(&self, records: &[CandidateRecord], tally: &KindTally) {
        // Every kind the run judged is reported, logged or not (#138): a
        // synapse candidate names an edge, so it carries no hidden-neuron
        // feature vector and no training row, and a count is the only place its
        // proposals and accepts are said out loud.
        if !tally.is_empty() {
            crate::log::detail(&format!("candidate log: {}", tally.summary()));
        }
        // Only where a row was **expected**: an edge cut holds none by design
        // and is reported by the tally above, so warning about it every batch
        // would bury the fault this warning exists for — a training set that
        // silently stopped growing.
        let unknown = tally.unexpected_unlogged();
        if unknown > 0 {
            crate::log::warn(&format!(
                "candidate log: {unknown} judged candidate(s) carried no feature vector where one \
                 was expected; their outcomes are not in the training set"
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

    /// Issue #138: a synapse candidate carries no hidden-neuron feature vector,
    /// so it can never hold a training row — and without a per-kind tally its
    /// proposals and its accepts would leave no trace in the telemetry at all.
    #[test]
    fn the_tally_reports_synapse_proposals_and_accepts_that_carry_no_row() {
        let mut tally = KindTally::default();
        tally.offer("ablation", false, true);
        tally.offer("ablation", true, true);
        tally.offer("synapse", false, false);
        tally.offer("synapse", true, false);
        tally.offer("synapse", false, false);

        assert_eq!(
            tally.counts("synapse"),
            KindCounts {
                proposals: 3,
                accepts: 1,
                logged: 0,
            }
        );
        assert_eq!(
            tally.counts("ablation"),
            KindCounts {
                proposals: 2,
                accepts: 1,
                logged: 2,
            }
        );
        assert_eq!(
            tally.counts("merge"),
            KindCounts::default(),
            "a kind the run never proposed reads as zero, not as absent"
        );
        assert_eq!(tally.unlogged(), 3, "every synapse proposal is unlogged");
        assert_eq!(
            tally.unexpected_unlogged(),
            0,
            "an edge cut holds no row by design, so it is not an anomaly"
        );
        assert_eq!(
            tally.summary(),
            "ablation 2 proposed, 1 accepted, 2 logged · synapse 3 proposed, 1 accepted, 0 logged"
        );
    }

    /// A **neuron** candidate with no feature vector is the fault the warning
    /// exists for, and it must survive the synapse exclusion (Issue #138).
    #[test]
    fn a_neuron_candidate_with_no_row_is_still_an_anomaly() {
        let mut tally = KindTally::default();
        tally.offer("synapse", false, false);
        tally.offer("ablation", false, false);
        tally.anomaly();

        assert_eq!(tally.unlogged(), 3);
        assert_eq!(
            tally.unexpected_unlogged(),
            2,
            "the ablation row and the unnameable entry, never the synapse"
        );
        assert_eq!(tally.anomalies(), 1);
        assert!(
            tally.summary().ends_with("1 of no nameable kind"),
            "{}",
            tally.summary()
        );
    }

    #[test]
    fn an_empty_tally_reports_nothing_rather_than_an_empty_line() {
        let tally = KindTally::default();
        assert!(tally.is_empty());
        assert_eq!(tally.unlogged(), 0);
        assert_eq!(tally.unexpected_unlogged(), 0);
        assert_eq!(tally.anomalies(), 0);
        assert_eq!(tally.summary(), "");
    }

    /// An anomaly alone is still something to report, so the tally is not
    /// "empty" just because no kind was nameable.
    #[test]
    fn a_tally_of_anomalies_alone_is_not_empty() {
        let mut tally = KindTally::default();
        tally.anomaly();
        assert!(!tally.is_empty());
        assert_eq!(tally.summary(), "1 of no nameable kind");
    }
}
