//! Seeded random sweep and sampled scorer screening (Issue #6).
//!
//! Hidden-neuron UUIDs are shuffled once with a recorded seed and visited
//! **without replacement**. Each visit tries an exact IDENTITY collapse, then
//! a mean-activation ablation. Unsupported or invalid attempts are skipped and
//! the batch is refilled while unvisited neurons remain.
//!
//! The incumbent and every valid candidate in a batch are scored together in
//! one sampled scorer call. Sampled winners are returned for later
//! authoritative promotion; they never become `best.json` here.

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::time::Instant;

use neat_core::{CreatureExport, SquashType, creature_to_json, parse_squash_name};
use serde::Serialize;

use crate::ablation::ablate_mean;
use crate::collapse::{CollapseOptions, collapse_identity};
use crate::incumbent::sha256_hex;
use crate::scorer::{DirectoryScorer, ScorerMode};
use crate::stats::ActivationStats;

/// Draw a seed from the clock and process id when the user omitted `--seed`.
pub fn draw_seed() -> u64 {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0);
    nanos ^ u64::from(std::process::id()).wrapping_shl(32) ^ 0xA5A5_A5A5_A5A5_A5A5
}

/// SplitMix64 — enough for a reproducible Fisher–Yates shuffle.
struct SplitMix64(u64);

impl SplitMix64 {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn shuffle<T>(&mut self, items: &mut [T]) {
        for i in (1..items.len()).rev() {
            let j = (self.next() as usize) % (i + 1);
            items.swap(i, j);
        }
    }
}

/// Kind of pruning proposal.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub enum CandidateKind {
    /// Exact IDENTITY collapse (#5).
    Identity,
    /// Mean-activation ablation (#4).
    Ablation,
}

/// One valid pruning candidate produced by the sweep.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SweepCandidate {
    /// Hidden neuron that was visited.
    pub uuid: String,
    /// Index in the seeded permutation.
    pub permutation_index: usize,
    /// How the candidate was built.
    pub kind: CandidateKind,
    /// Cohort file stem (`c000`, …).
    pub stem: String,
    /// Candidate creature.
    #[serde(skip)]
    pub creature: CreatureExport,
}

/// A visitation that did not emit a candidate.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SweepSkip {
    /// Hidden neuron UUID.
    pub uuid: String,
    /// Index in the permutation.
    pub permutation_index: usize,
    /// Why it was skipped.
    pub reason: String,
}

/// Seeded without-replacement walk over hidden neurons.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Sweep {
    /// Seed that produced [`Self::order`].
    pub seed: u64,
    /// SHA-256 of `seed` plus the ordered UUID list.
    pub permutation_identity: String,
    /// Hidden UUIDs in visitation order.
    pub order: Vec<String>,
    /// Next index to visit.
    pub next: usize,
}

impl Sweep {
    /// Shuffle the incumbent's hidden UUIDs with `seed`.
    pub fn new(creature: &CreatureExport, seed: u64) -> Self {
        let mut order: Vec<String> = creature
            .neurons
            .iter()
            .filter(|n| n.neuron_type == "hidden")
            .map(|n| n.uuid.clone())
            .collect();
        SplitMix64(seed).shuffle(&mut order);
        let mut ident = format!("seed={seed}\n");
        for uuid in &order {
            ident.push_str(uuid);
            ident.push('\n');
        }
        Self {
            seed,
            permutation_identity: sha256_hex(ident.as_bytes()),
            order,
            next: 0,
        }
    }

    /// Remaining unvisited neurons.
    pub fn remaining(&self) -> usize {
        self.order.len().saturating_sub(self.next)
    }

    /// True when every hidden UUID has been visited.
    pub fn exhausted(&self) -> bool {
        self.next >= self.order.len()
    }

    /// Move still-unvisited `uuids` to the front of the remaining order.
    ///
    /// Prefer-list order is preserved. Unknown or already-visited UUIDs are ignored.
    pub fn prefer(&mut self, uuids: &[String]) {
        if self.next >= self.order.len() || uuids.is_empty() {
            return;
        }
        let mut remaining: Vec<String> = self.order.split_off(self.next);
        let mut front = Vec::new();
        for u in uuids {
            if let Some(i) = remaining.iter().position(|x| x == u) {
                front.push(remaining.remove(i));
            }
        }
        self.order.extend(front);
        self.order.extend(remaining);
    }

    /// Build up to `size` valid candidates, refilling past skips.
    pub fn fill_batch(
        &mut self,
        incumbent: &CreatureExport,
        stats: &ActivationStats,
        size: usize,
    ) -> (Vec<SweepCandidate>, Vec<SweepSkip>) {
        self.fill_batch_avoiding(incumbent, stats, size, &HashSet::new())
    }

    /// [`Self::fill_batch`] that skips UUIDs in `avoid` (fresh known failures).
    pub fn fill_batch_avoiding(
        &mut self,
        incumbent: &CreatureExport,
        stats: &ActivationStats,
        size: usize,
        avoid: &HashSet<String>,
    ) -> (Vec<SweepCandidate>, Vec<SweepSkip>) {
        self.fill_batch_skipping(incumbent, stats, size, avoid, &HashSet::new())
    }

    /// [`Self::fill_batch`] skipping known failures and tagged provenance neurons (#26).
    pub fn fill_batch_skipping(
        &mut self,
        incumbent: &CreatureExport,
        stats: &ActivationStats,
        size: usize,
        avoid: &HashSet<String>,
        tagged: &HashSet<String>,
    ) -> (Vec<SweepCandidate>, Vec<SweepSkip>) {
        let mut candidates = Vec::new();
        let mut skips = Vec::new();
        while candidates.len() < size && !self.exhausted() {
            let permutation_index = self.next;
            let uuid = self.order[permutation_index].clone();
            self.next += 1;
            if tagged.contains(&uuid) {
                skips.push(SweepSkip {
                    uuid,
                    permutation_index,
                    reason: "tagged".into(),
                });
                continue;
            }
            if avoid.contains(&uuid) {
                skips.push(SweepSkip {
                    uuid,
                    permutation_index,
                    reason: "known-failure".into(),
                });
                continue;
            }
            match propose(incumbent, stats, &uuid) {
                Ok((kind, creature)) => {
                    let stem = format!("c{:03}", candidates.len());
                    candidates.push(SweepCandidate {
                        uuid,
                        permutation_index,
                        kind,
                        stem,
                        creature,
                    });
                }
                Err(reason) => skips.push(SweepSkip {
                    uuid,
                    permutation_index,
                    reason,
                }),
            }
        }
        (candidates, skips)
    }
}

fn is_identity(creature: &CreatureExport, uuid: &str) -> bool {
    creature.neurons.iter().any(|n| {
        n.uuid == uuid
            && parse_squash_name(n.squash.as_deref().unwrap_or("IDENTITY"))
                .is_ok_and(|s| s == SquashType::Identity)
    })
}

pub(crate) fn propose(
    incumbent: &CreatureExport,
    stats: &ActivationStats,
    uuid: &str,
) -> Result<(CandidateKind, CreatureExport), String> {
    if is_identity(incumbent, uuid) {
        match collapse_identity(incumbent, uuid, CollapseOptions::default()) {
            Ok(c) => return Ok((CandidateKind::Identity, c.creature)),
            Err(e) => {
                // Cost-increasing IDENTITY still has an approximate ablation path.
                if stats.by_uuid(uuid).is_none() {
                    return Err(e.to_string());
                }
            }
        }
    }
    let mean = stats
        .by_uuid(uuid)
        .map(|s| s.mean)
        .ok_or_else(|| format!("no activation stats for `{uuid}`"))?;
    match ablate_mean(incumbent, uuid, mean, stats.by_uuid(uuid)) {
        Ok(a) => Ok((CandidateKind::Ablation, a.creature)),
        Err(e) => Err(e.to_string()),
    }
}

/// One sampled winner. Not an acceptance.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SampledWinner {
    /// Candidate that beat the sampled incumbent.
    pub candidate: SweepCandidate,
    /// Sampled candidate score.
    pub score: f64,
    /// Sampled incumbent score from the same call.
    pub baseline_score: f64,
    /// `score - baseline_score`.
    pub delta: f64,
}

/// Outcome of one sampled screen. Never writes `best.json`.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ScreenOutcome {
    /// Sample rate.
    pub sample_rate: f64,
    /// Sample phase.
    pub sample_phase: u64,
    /// Sampled incumbent score.
    pub baseline_score: f64,
    /// Candidates that beat the sampled incumbent by `threshold`.
    pub winners: Vec<SampledWinner>,
    /// Candidates that did not.
    pub losers: Vec<String>,
    /// Wall time of the scorer call (ms).
    pub screen_ms: u64,
    /// Candidates scored per second.
    pub candidates_per_sec: f64,
    /// Extrapolated ms to finish the remaining permutation at this rate.
    pub estimated_full_sweep_ms: u64,
}

/// Parameters for [`screen_batch`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScreenConfig<'a> {
    /// Sample rate in `(0, 1)`.
    pub sample_rate: f64,
    /// Sample phase.
    pub sample_phase: u64,
    /// Sampled Δscore required to promote (`delta > threshold`).
    pub threshold: f64,
    /// Unvisited neurons remaining after this batch (for sweep ETA).
    pub remaining_after: usize,
    /// Directory that receives `baseline.json` and candidate files.
    pub dir: &'a Path,
}

/// Score the incumbent and `candidates` in one sampled scorer cohort.
///
/// Writes into [`ScreenConfig::dir`] and does **not** touch `best.json`.
pub fn screen_batch(
    scorer: &dyn DirectoryScorer,
    training_dir: &Path,
    incumbent: &CreatureExport,
    candidates: Vec<SweepCandidate>,
    cfg: ScreenConfig<'_>,
) -> Result<ScreenOutcome, String> {
    std::fs::create_dir_all(cfg.dir).map_err(|e| format!("{}: {e}", cfg.dir.display()))?;
    let baseline_json =
        creature_to_json(incumbent).map_err(|e| format!("serialise incumbent: {e}"))?;
    std::fs::write(cfg.dir.join("baseline.json"), baseline_json)
        .map_err(|e| format!("baseline.json: {e}"))?;
    for c in &candidates {
        let json = creature_to_json(&c.creature).map_err(|e| format!("{}: {e}", c.uuid))?;
        std::fs::write(cfg.dir.join(format!("{}.json", c.stem)), json)
            .map_err(|e| format!("{}: {e}", c.stem))?;
    }
    let mode = ScorerMode::Sample {
        rate: cfg.sample_rate,
        phase: cfg.sample_phase,
    };
    let started = Instant::now();
    let results = scorer
        .score_directory(cfg.dir, training_dir, mode)
        .map_err(|e| e.to_string())?;
    let screen_ms = started.elapsed().as_millis() as u64;
    let baseline = results
        .get("baseline")
        .ok_or_else(|| "screen: scorer returned no `baseline` entry".to_string())?;
    let n = candidates.len();
    let candidates_per_sec = if screen_ms == 0 {
        n as f64
    } else {
        n as f64 * 1000.0 / screen_ms as f64
    };
    let batches_left = if n == 0 {
        0
    } else {
        cfg.remaining_after.div_ceil(n)
    };
    let estimated_full_sweep_ms = screen_ms.saturating_mul(batches_left as u64);

    let mut winners = Vec::new();
    let mut losers = Vec::new();
    for c in candidates {
        let result = results.get(&c.stem).ok_or_else(|| {
            format!(
                "screen: scorer returned no entry for candidate stem `{}`",
                c.stem
            )
        })?;
        let delta = result.score - baseline.score;
        if delta > cfg.threshold {
            winners.push(SampledWinner {
                candidate: c,
                score: result.score,
                baseline_score: baseline.score,
                delta,
            });
        } else {
            losers.push(c.uuid);
        }
    }
    Ok(ScreenOutcome {
        sample_rate: cfg.sample_rate,
        sample_phase: cfg.sample_phase,
        baseline_score: baseline.score,
        winners,
        losers,
        screen_ms,
        candidates_per_sec,
        estimated_full_sweep_ms,
    })
}

/// Directory used for one screen cohort.
pub fn screen_dir(workspace: &Path, batch: u64) -> PathBuf {
    workspace.join(format!("screen-{batch}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::baseline::fake::ScriptedScorer;
    use crate::fixtures::{creature, neuron, synapse};
    use crate::incumbent::validate_creature;
    use crate::stats::{ActivationStats, NeuronStats, STATS_FORMAT_VERSION};

    fn two_hidden() -> CreatureExport {
        creature(
            1,
            1,
            vec![
                neuron("hidden", "h_a", 0.0, Some("IDENTITY")),
                neuron("hidden", "h_b", 0.0, Some("IDENTITY")),
                neuron("output", "output-0", 0.0, Some("IDENTITY")),
            ],
            vec![
                synapse("input-0", "h_a", 1.0),
                synapse("input-0", "h_b", 1.0),
                synapse("h_a", "output-0", 1.0),
                synapse("h_b", "output-0", 1.0),
            ],
        )
    }

    fn stats_for(creature: &CreatureExport) -> ActivationStats {
        ActivationStats {
            format_version: STATS_FORMAT_VERSION,
            creature_checksum: "t".into(),
            corpus_identity: "c".into(),
            record_count: 1,
            scan_ms: 0,
            from_cache: false,
            neurons: creature
                .neurons
                .iter()
                .enumerate()
                .filter(|(_, n)| n.neuron_type == "hidden")
                .map(|(i, n)| NeuronStats {
                    uuid: n.uuid.clone(),
                    neuron_index: i,
                    count: 1,
                    mean: 0.0,
                    variance: 0.0,
                    std_dev: 0.0,
                    mean_abs: 0.0,
                    min: 0.0,
                    max: 0.0,
                })
                .collect(),
        }
    }

    #[test]
    fn fixed_seed_reproduces_visitation_order() {
        let creature = two_hidden();
        validate_creature(&creature).unwrap();
        let a = Sweep::new(&creature, 42);
        let b = Sweep::new(&creature, 42);
        assert_eq!(a.order, b.order);
        assert_eq!(a.permutation_identity, b.permutation_identity);
        let c = Sweep::new(&creature, 43);
        assert_ne!(a.order, c.order);
    }

    #[test]
    fn no_neuron_is_visited_twice_before_exhaustion() {
        let creature = two_hidden();
        let stats = stats_for(&creature);
        let mut sweep = Sweep::new(&creature, 7);
        let n = sweep.order.len();
        let mut seen = Vec::new();
        while !sweep.exhausted() {
            let (batch, skips) = sweep.fill_batch(&creature, &stats, 1);
            for s in skips {
                seen.push(s.uuid);
            }
            for c in batch {
                seen.push(c.uuid);
            }
        }
        assert_eq!(seen.len(), n);
        let mut uniq = seen.clone();
        uniq.sort();
        uniq.dedup();
        assert_eq!(uniq.len(), n);
        let (more, _) = sweep.fill_batch(&creature, &stats, 10);
        assert!(more.is_empty());
    }

    #[test]
    fn screen_scores_incumbent_and_candidates_in_one_cohort() {
        let creature = two_hidden();
        let stats = stats_for(&creature);
        let mut sweep = Sweep::new(&creature, 1);
        let (batch, skips) = sweep.fill_batch(&creature, &stats, 8);
        assert!(skips.is_empty(), "{skips:?}");
        assert_eq!(batch.len(), 2);

        let tmp = tempfile::tempdir().unwrap();
        let scorer = ScriptedScorer {
            baseline_score: 0.50,
            candidate_score: Some(0.51),
            ..ScriptedScorer::ok(0.50, 0.50)
        };
        let outcome = screen_batch(
            &scorer,
            tmp.path(),
            &creature,
            batch,
            ScreenConfig {
                sample_rate: 0.05,
                sample_phase: 3,
                threshold: 0.0,
                remaining_after: sweep.remaining(),
                dir: &tmp.path().join("screen"),
            },
        )
        .unwrap();
        assert_eq!(
            scorer.last_mode.get(),
            Some(ScorerMode::Sample {
                rate: 0.05,
                phase: 3
            })
        );
        let stems = scorer.last_stems.borrow().clone();
        assert!(stems.contains(&"baseline".into()));
        assert!(stems.iter().any(|s| s.starts_with('c')));
        assert_eq!(outcome.winners.len(), 2);
        assert!(outcome.losers.is_empty());
        assert!(!tmp.path().join("best.json").exists());
    }

    #[test]
    fn sample_losers_are_not_returned_as_winners() {
        let creature = two_hidden();
        let stats = stats_for(&creature);
        let mut sweep = Sweep::new(&creature, 9);
        let (batch, _) = sweep.fill_batch(&creature, &stats, 8);
        let tmp = tempfile::tempdir().unwrap();
        let scorer = ScriptedScorer {
            baseline_score: 0.80,
            candidate_score: Some(0.10),
            ..ScriptedScorer::ok(0.80, 0.20)
        };
        let outcome = screen_batch(
            &scorer,
            tmp.path(),
            &creature,
            batch,
            ScreenConfig {
                sample_rate: 0.05,
                sample_phase: 0,
                threshold: 0.0,
                remaining_after: 0,
                dir: &tmp.path().join("screen"),
            },
        )
        .unwrap();
        assert!(outcome.winners.is_empty());
        assert_eq!(outcome.losers.len(), 2);
        assert!(!tmp.path().join("best.json").exists());
    }

    #[test]
    fn prefer_moves_still_unvisited_uuids_to_the_front() {
        let creature = two_hidden();
        let mut sweep = Sweep::new(&creature, 1);
        let last = sweep.order.last().cloned().unwrap();
        sweep.prefer(std::slice::from_ref(&last));
        assert_eq!(sweep.order[sweep.next], last);
    }

    #[test]
    fn fill_batch_skips_known_failures() {
        let creature = two_hidden();
        let stats = stats_for(&creature);
        let mut sweep = Sweep::new(&creature, 9);
        let blocked = sweep.order[0].clone();
        let avoid = HashSet::from([blocked.clone()]);
        let (batch, skips) = sweep.fill_batch_avoiding(&creature, &stats, 8, &avoid);
        assert!(batch.iter().all(|c| c.uuid != blocked));
        assert!(
            skips
                .iter()
                .any(|s| s.uuid == blocked && s.reason == "known-failure"),
            "{skips:?}"
        );
    }

    #[test]
    fn fill_batch_skips_tagged_neurons_as_tagged() {
        let creature = two_hidden();
        let stats = stats_for(&creature);
        let mut sweep = Sweep::new(&creature, 9);
        let blocked = sweep.order[0].clone();
        let tagged = HashSet::from([blocked.clone()]);
        let (batch, skips) =
            sweep.fill_batch_skipping(&creature, &stats, 8, &HashSet::new(), &tagged);
        assert!(batch.iter().all(|c| c.uuid != blocked));
        assert!(
            skips
                .iter()
                .any(|s| s.uuid == blocked && s.reason == "tagged"),
            "{skips:?}"
        );
    }
}
