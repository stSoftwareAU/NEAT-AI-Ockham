//! Append-only `experiments.jsonl` journal (Issue #8).
//!
//! Every line is one JSON object with a `record` discriminator. Lines are
//! written with a single `write_all` of the complete line so an interrupted
//! run leaves a valid prefix.

use std::io::Write;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::incumbent::now_unix;
use crate::ordering::Ordering;

/// One journal event.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "record", rename_all = "camelCase")]
pub enum Event {
    /// Run started after the baseline gate.
    Start {
        /// Effective RNG seed.
        seed: u64,
        /// Named candidate ordering (Issue #11). Older journals read `random`.
        #[serde(default)]
        ordering: Ordering,
        /// Fraction of sweep slots reserved for the random control.
        #[serde(default)]
        ordering_random_quota: f64,
        /// Permutation identity, hashed **before** any coverage reorder.
        permutation_identity: String,
        /// Whether unchecked-first selection reordered the sweep (Issue #38).
        #[serde(default)]
        unchecked_first: bool,
        /// How many neurons old-corpus verdicts moved to the front (Issue #88).
        ///
        /// Beside `unchecked_first` for the same reason: both reorder the sweep
        /// after `permutation_identity` is hashed, so a run whose order they
        /// changed is only reconstructable if the journal says so. `0` on a run
        /// with the priority off, no cache, or nothing left to prioritise.
        #[serde(default)]
        old_corpus_first: usize,
        /// Hidden neurons on the opening incumbent.
        hidden: usize,
        /// Synapses on the opening incumbent.
        #[serde(default)]
        synapses: usize,
        /// Opening authoritative score.
        opening_score: f64,
    },
    /// One sweep batch was generated.
    Batch {
        /// Batch index (0-based).
        batch: u64,
        /// Valid candidates emitted.
        candidates: usize,
        /// Visits skipped.
        skipped: usize,
        /// Remaining unvisited hidden neurons.
        remaining: usize,
    },
    /// The sweep ran out of neurons and was rebuilt (Issue #77).
    ///
    /// A run that has visited every hidden neuron restarts its sweep rather
    /// than idling to the deadline, so this is a normal, meaningful step in the
    /// fleet history: it says the creature was screened end to end and the run
    /// rolled into re-screening the stalest neurons.
    SweepRestart {
        /// How many times the sweep has been rebuilt this run (1-based).
        restarts: u64,
        /// Hidden neurons the fresh sweep will visit.
        hidden: usize,
        /// Distinct UUIDs this run had newly screened when it restarted.
        newly_screened: usize,
    },
    /// Sampled screen result.
    Screen {
        /// Sampled winners promoted to full scoring.
        winners: usize,
        /// Sampled losers.
        losers: usize,
        /// Screen wall time (ms).
        ms: u64,
    },
    /// One rung of the progressive screening ladder (Issue #104).
    ///
    /// Written per stage, and only when the ladder is progressive: the
    /// fixed-rate control is fully described by [`Self::Screen`], and a second
    /// record per batch saying the same thing would inflate every existing
    /// journal for nothing. What the ladder costs and what it decided cannot be
    /// read off `screen` alone — the whole claim is that a rejected candidate
    /// stopped after a small fraction of the corpus, and only `recordsScored`
    /// can settle that.
    ScreenStage {
        /// Batch index (0-based).
        batch: u64,
        /// Position in the ladder (0-based).
        stage: usize,
        /// Sample rate this stage scored at.
        rate: f64,
        /// Deterministic sample phase.
        phase: u64,
        /// Candidates that entered the stage.
        entered: usize,
        /// Candidates the stage rejected.
        rejected: usize,
        /// Candidates carried on, or promoted by the final stage.
        promoted: usize,
        /// Records read across the cohort, incumbent included.
        records_scored: u64,
        /// Mean sampled Δ over the candidates that entered.
        mean_delta: f64,
        /// Scorer wall time (ms).
        ms: u64,
        /// What survivors did: `carried` or `promoted`.
        outcome: String,
    },
    /// Screen-coverage records filed for one batch (Issue #36).
    ///
    /// A sibling of [`Self::Screen`] rather than a field on it: coverage is
    /// also filed when screening is disabled, and `screen` must keep counting
    /// only real sampled scorer calls.
    Screened {
        /// Batch index (0-based).
        batch: u64,
        /// Screen-coverage records filed for this batch.
        screened: usize,
    },
    /// The screening a run did after a replay accept ended its search (#91).
    ///
    /// A search accept opens no tail — since Issue #96 it restarts the sweep
    /// and the search runs on. The run's `stop` reason names the replay accept,
    /// because that is what ended the search — so without this record the journal could not tell a tail that
    /// ran out of wall clock from one that ran out of experiments or found a
    /// whole pass proposing nothing. A stop reason that answers three questions
    /// with one word is how a plateau hides (#77).
    CoverageTail {
        /// Screening batches the tail completed.
        batches: u64,
        /// Candidates the tail put through the sampled screen.
        screened: usize,
        /// Distinct UUIDs newly checked by the end of the tail.
        newly_checked: usize,
        /// What ended the tail: `timeout`, `max-experiments`, `no-candidates`,
        /// `cancelled`, `scorer-failures`, or `no-hidden`.
        ended: String,
    },
    /// Screening coverage over the incumbent at the end of a run (Issue #37).
    ///
    /// Written only when a learnings dir is configured: without the screen
    /// store there is no coverage state, and `checked: 0` would be a lie
    /// rather than a measurement.
    Coverage {
        /// Hidden neurons on the final incumbent.
        hidden: usize,
        /// Hidden neurons carrying tags, screened like any other (#87).
        #[serde(default)]
        tagged: usize,
        /// Hidden neurons Ockham may try — all of them, tagged included (#74).
        checkable: usize,
        /// Hidden UUIDs with at least one screen record.
        checked: usize,
        /// Checked UUIDs the razor could never propose a cut for (#93).
        #[serde(default)]
        blocked: usize,
        /// That total split by reason code (Issue #103).
        ///
        /// Journalled per run and per epoch, so the blocked population can be
        /// inspected across epochs from the append-only journal alone rather
        /// than only in the run that happened to be watched. `#[serde(default)]`
        /// so a journal written before #103 still reads, as no reasons.
        #[serde(default)]
        blocked_by_reason: crate::blocked::BlockedBreakdown,
        /// Hidden neurons removed this run.
        #[serde(default)]
        cut: usize,
        /// Corpus identity these figures were measured against (Issue #100).
        ///
        /// The screening epoch: coverage is only authoritative for the corpus
        /// it was measured against, so a reader of two runs' journals can tell
        /// a fresh epoch from a collapse in coverage. `report` needs no change
        /// to carry it — it reports the latest coverage snapshot, which is the
        /// current epoch's by construction. `None` on records written before
        /// this field existed.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        corpus_identity: Option<String>,
    },
    /// Full-corpus cohort result.
    Full {
        /// Individuals scored.
        individuals: usize,
        /// Bundles scored.
        bundles: usize,
        /// Whether a local winner was selected.
        accepted: bool,
        /// Winner score when accepted.
        #[serde(skip_serializing_if = "Option::is_none")]
        score: Option<f64>,
        /// Winner Δ vs the same-call incumbent.
        #[serde(skip_serializing_if = "Option::is_none")]
        delta: Option<f64>,
        /// Neurons cut by the accepted winner (`0` when nothing was accepted).
        #[serde(default)]
        cuts: usize,
        /// Milliseconds from the first sweep batch to this cohort result.
        #[serde(default)]
        elapsed_ms: u64,
    },
    /// Estimated versus actual structural saving of an accepted cut (#106).
    ///
    /// The cascade dry-run predicts, from topology alone, what an ablation of
    /// these neurons would remove; this is what the accepted creature actually
    /// shed. Written on every accept, whatever ordering the run used, so the
    /// prediction is audited against the outcome instead of being trusted — a
    /// signal that drifts from what the razor really removes is a signal to
    /// stop paying for.
    Cascade {
        /// Cohort kind of the accepted winner: `individual` or `bundle`.
        ///
        /// A bundle removes structure several cuts share, so its prediction and
        /// its outcome compose differently from a single cut's; without the
        /// kind the two could not be told apart in the same series.
        kind: String,
        /// Hidden neurons the accepted winner was asked to cut.
        cuts: usize,
        /// Hidden neurons the dry-run predicted, requested plus cascade.
        estimated_hidden: usize,
        /// Synapses the dry-run predicted.
        estimated_synapses: usize,
        /// Growth units the dry-run predicted.
        estimated_growth_units: f64,
        /// Hidden neurons the accepted creature actually removed.
        ///
        /// Signed: an accepted transform may add structure on one axis while
        /// the growth units it is judged on still fall, and clamping that to
        /// zero would report the addition as "removed nothing".
        actual_hidden: i64,
        /// Synapses the accepted creature actually removed; signed, as above.
        actual_synapses: i64,
        /// Growth units the accepted creature actually removed.
        actual_growth_units: f64,
    },
    /// Cohort sizing against the wall clock (Issue #58).
    ///
    /// Written after every full cohort so `--report` can show whether runs are
    /// budget-starved: a persistent trim says the budget, not the algorithm, is
    /// the binding constraint.
    Budget {
        /// Rolling full-corpus cost estimate, milliseconds per creature.
        #[serde(default)]
        est_ms_per_creature: f64,
        /// Wall clock left when the cohort was sized.
        #[serde(default)]
        remaining_secs: u64,
        /// Entries scored, excluding the incumbent baseline.
        #[serde(default)]
        entries: usize,
        /// Individual entries dropped to fit the budget.
        #[serde(default)]
        dropped_individuals: usize,
        /// Bundle entries dropped to fit the budget.
        #[serde(default)]
        dropped_bundles: usize,
    },
    /// What the run tried, kept and rejected (Issue #59).
    Winners {
        /// Sampled winners promoted to full scoring.
        #[serde(default)]
        screened: usize,
        /// Distinct UUIDs with a full-corpus positive of their own.
        #[serde(default)]
        confirmed: usize,
        /// Hidden neurons removed by accepted winners.
        #[serde(default)]
        applied: usize,
        /// Confirmed winners still standing at the end of the run.
        #[serde(default)]
        carried: usize,
        /// Bundle plans scored.
        #[serde(default)]
        plans: usize,
        /// Plans skipped because a cut no longer proposed.
        #[serde(default)]
        skipped: usize,
        /// Cuts in the largest accepted winner.
        #[serde(default)]
        best_cuts: usize,
        /// Full-corpus delta of that winner.
        #[serde(default)]
        best_delta: f64,
        /// Cohort entries dropped over budget.
        #[serde(default)]
        dropped: usize,
        /// Rolling cost estimate, milliseconds per creature.
        #[serde(default)]
        est_ms_per_creature: u64,
    },
    /// Loop stopped.
    Stop {
        /// Why the loop ended.
        reason: String,
        /// Authoritative local accepts.
        accepts: u64,
        /// Batches attempted.
        experiments: u64,
        /// Final authoritative score.
        final_score: f64,
        /// Cumulative score gain from the opening parent.
        cumulative_delta: f64,
        /// Distinct hidden UUIDs this run moved from unscreened to screened
        /// (Issue #77).
        ///
        /// Zero while unchecked neurons remain is the plateau signature: the
        /// run reported well-formed coverage without advancing it.
        #[serde(default)]
        newly_screened: usize,
        /// Hidden neurons left on the final incumbent.
        #[serde(default)]
        final_hidden: usize,
        /// Synapses left on the final incumbent.
        #[serde(default)]
        final_synapses: usize,
        /// Wall-clock milliseconds spent in the optimisation loop.
        #[serde(default)]
        elapsed_ms: u64,
    },
}

/// Append `event` as one JSON line.
pub fn append(path: &Path, event: &Event) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| format!("{}: {e}", parent.display()))?;
    }
    let mut line = serde_json::to_string(event).map_err(|e| e.to_string())?;
    line.push('\n');
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .map_err(|e| format!("{}: {e}", path.display()))?;
    file.write_all(line.as_bytes())
        .map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(())
}

/// Unix seconds for journal timestamps (attached by callers when useful).
pub fn unix() -> u64 {
    now_unix()
}
