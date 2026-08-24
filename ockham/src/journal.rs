//! Append-only `experiments.jsonl` journal (Issue #8).
//!
//! Every line is one JSON object with a `record` discriminator. Lines are
//! written with a single `write_all` of the complete line so an interrupted
//! run leaves a valid prefix.

use std::io::Write;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::incumbent::now_unix;

/// One journal event.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "record", rename_all = "camelCase")]
pub enum Event {
    /// Run started after the baseline gate.
    Start {
        /// Effective RNG seed.
        seed: u64,
        /// Permutation identity.
        permutation_identity: String,
        /// Hidden neurons on the opening incumbent.
        hidden: usize,
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
