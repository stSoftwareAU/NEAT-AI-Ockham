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
        /// Permutation identity.
        permutation_identity: String,
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
    /// Sampled screen result.
    Screen {
        /// Sampled winners promoted to full scoring.
        winners: usize,
        /// Sampled losers.
        losers: usize,
        /// Screen wall time (ms).
        ms: u64,
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
    /// Screening coverage over the incumbent at the end of a run (Issue #37).
    ///
    /// Written only when a learnings dir is configured: without the screen
    /// store there is no coverage state, and `checked: 0` would be a lie
    /// rather than a measurement.
    Coverage {
        /// Hidden neurons on the final incumbent.
        hidden: usize,
        /// Hidden neurons carrying GRQ-provenance tags, skipped as candidates.
        #[serde(default)]
        tagged: usize,
        /// `hidden - tagged`: the coverage denominator.
        checkable: usize,
        /// Checkable UUIDs with at least one screen record.
        checked: usize,
        /// Hidden neurons removed this run.
        #[serde(default)]
        cut: usize,
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
