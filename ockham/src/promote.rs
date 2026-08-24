//! Full-corpus scoring of sampled winners and grouped bundles (Issue #7).
//!
//! Every sampled winner is scored on the full corpus together with the current
//! Ockham incumbent. Ranked prefixes (`2`, `4`, `8`, `16`, all) are also built
//! as independent bundles from the same incumbent snapshot. Individual score
//! deltas are never assumed additive.
//!
//! The highest full-corpus score strictly above the incumbent by
//! `min_improvement` wins. A sampled win is never enough to update `best.json`.

use std::collections::HashSet;
use std::path::Path;

use neat_core::{CreatureExport, creature_to_json};
use serde::Serialize;

use crate::ablation::StructureSnapshot;
use crate::incumbent::{sha256_hex, validate_creature};
use crate::scorer::{DirectoryScorer, ScoreResult, ScorerMode};
use crate::stats::ActivationStats;
use crate::sweep::{SampledWinner, SweepCandidate, propose};

/// Bundle prefix lengths from the charter.
const BUNDLE_PREFIXES: &[usize] = &[2, 4, 8, 16];

/// One fully-scored candidate (individual or bundle).
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct FullCandidate {
    /// Cohort file stem.
    pub stem: String,
    /// `individual` or `bundle`.
    pub kind: &'static str,
    /// Hidden UUIDs applied, in order.
    pub uuids: Vec<String>,
    /// Full-corpus score.
    pub score: f64,
    /// Full-corpus error.
    pub error: f64,
    /// Complexity penalty when the scorer reported it.
    pub complexity_penalty: f64,
    /// Structure after the transform.
    pub after: StructureSnapshot,
    /// `score - incumbent_score`.
    pub delta: f64,
}

/// Authoritative local winner. Tiny deltas are retained.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct LocalWinner {
    /// Winning candidate.
    pub candidate: FullCandidate,
    /// Checksum of the exported JSON that won.
    pub checksum: String,
    /// Winning creature (not journalled as nested JSON).
    #[serde(skip)]
    pub creature: CreatureExport,
}

/// Outcome of one full-score cohort. Sample results never accept.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct FullOutcome {
    /// Same-call full incumbent score.
    pub incumbent_score: f64,
    /// Same-call full incumbent error.
    pub incumbent_error: f64,
    /// Individual sampled-winner full scores.
    pub individuals: Vec<FullCandidate>,
    /// Bundle full scores (skipped bundles are omitted).
    pub bundles: Vec<FullCandidate>,
    /// Sampled winners whose full score did not beat the incumbent.
    pub sample_false_positives: Vec<String>,
    /// Authoritative local winner, if any.
    pub winner: Option<LocalWinner>,
    /// Full scorer wall time (ms).
    pub full_ms: u64,
}

/// Configuration for [`evaluate_full`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FullConfig<'a> {
    /// Strict minimum `score - incumbent` to accept.
    pub min_improvement: f64,
    /// Directory for the full-score cohort.
    pub dir: &'a Path,
    /// When set, a winner is written here as `best.json`.
    pub best_path: Option<&'a Path>,
}

/// Rank sampled winners by sample delta and emit unique bundle UUID prefixes.
pub fn bundle_plans(winners: &[SampledWinner]) -> Vec<Vec<String>> {
    let mut ranked = winners.to_vec();
    ranked.sort_by(|a, b| {
        b.delta
            .partial_cmp(&a.delta)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let ids: Vec<String> = ranked.iter().map(|w| w.candidate.uuid.clone()).collect();
    let mut plans = Vec::new();
    let mut seen = HashSet::new();
    let mut push = |plan: Vec<String>| {
        if plan.len() < 2 {
            return;
        }
        let key = plan.join("\0");
        if seen.insert(key) {
            plans.push(plan);
        }
    };
    for n in BUNDLE_PREFIXES {
        if *n <= ids.len() {
            push(ids[..*n].to_vec());
        }
    }
    if ids.len() > 1 {
        push(ids);
    }
    plans
}

/// Apply `uuids` in order on a clone of `incumbent`, with cleanup after each.
pub fn apply_bundle(
    incumbent: &CreatureExport,
    stats: &ActivationStats,
    uuids: &[String],
) -> Result<CreatureExport, String> {
    let mut current = incumbent.clone();
    for uuid in uuids {
        if current.neurons.iter().all(|n| n.uuid != *uuid) {
            return Err(format!("bundle: `{uuid}` already gone after a prior step"));
        }
        let (_, next) = propose(&current, stats, uuid)?;
        current = next;
    }
    validate_creature(&current).map_err(|e| e.to_string())?;
    Ok(current)
}

/// Full-score sampled winners and their bundle prefixes in one scorer call.
///
/// Scorer failure means no winner. A sampled win cannot update `best.json`.
pub fn evaluate_full(
    scorer: &dyn DirectoryScorer,
    training_dir: &Path,
    incumbent: &CreatureExport,
    stats: &ActivationStats,
    sampled: &[SampledWinner],
    cfg: FullConfig<'_>,
) -> Result<FullOutcome, String> {
    std::fs::create_dir_all(cfg.dir).map_err(|e| format!("{}: {e}", cfg.dir.display()))?;
    let baseline_json =
        creature_to_json(incumbent).map_err(|e| format!("serialise incumbent: {e}"))?;
    std::fs::write(cfg.dir.join("baseline.json"), baseline_json)
        .map_err(|e| format!("baseline.json: {e}"))?;

    let mut pending: Vec<(String, &'static str, Vec<String>, CreatureExport)> = Vec::new();
    for (i, w) in sampled.iter().enumerate() {
        let stem = format!("i{i:03}");
        write_creature(cfg.dir, &stem, &w.candidate.creature)?;
        pending.push((
            stem,
            "individual",
            vec![w.candidate.uuid.clone()],
            w.candidate.creature.clone(),
        ));
    }
    let mut skipped_bundles = Vec::new();
    for (i, plan) in bundle_plans(sampled).into_iter().enumerate() {
        match apply_bundle(incumbent, stats, &plan) {
            Ok(creature) => {
                let stem = format!("b{i:03}");
                write_creature(cfg.dir, &stem, &creature)?;
                pending.push((stem, "bundle", plan, creature));
            }
            Err(reason) => skipped_bundles.push(reason),
        }
    }
    let _ = skipped_bundles;

    let started = std::time::Instant::now();
    let results = scorer
        .score_directory(cfg.dir, training_dir, ScorerMode::Full)
        .map_err(|e| e.to_string())?;
    let full_ms = started.elapsed().as_millis() as u64;
    let baseline = results
        .get("baseline")
        .ok_or_else(|| "full: scorer returned no `baseline` entry".to_string())?;

    let mut individuals = Vec::new();
    let mut bundles = Vec::new();
    let mut sample_false_positives = Vec::new();
    let mut best: Option<(f64, LocalWinner, CreatureExport)> = None;

    for (stem, kind, uuids, creature) in pending {
        let result = results
            .get(&stem)
            .ok_or_else(|| format!("full: scorer returned no entry for `{stem}`"))?;
        let cand = full_candidate(stem, kind, uuids, &creature, result, baseline.score);
        if kind == "individual" && cand.delta <= cfg.min_improvement {
            sample_false_positives.push(cand.uuids[0].clone());
        }
        if cand.delta > cfg.min_improvement {
            let json = creature_to_json(&creature).map_err(|e| e.to_string())?;
            let winner = LocalWinner {
                checksum: sha256_hex(json.as_bytes()),
                candidate: cand.clone(),
                creature: creature.clone(),
            };
            let take = match &best {
                None => true,
                Some((score, _, _)) => cand.score > *score,
            };
            if take {
                best = Some((cand.score, winner, creature));
            }
        }
        if kind == "individual" {
            individuals.push(cand);
        } else {
            bundles.push(cand);
        }
    }

    let winner = if let Some((_, winner, creature)) = best {
        if let Some(path) = cfg.best_path {
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent)
                    .map_err(|e| format!("{}: {e}", parent.display()))?;
            }
            let json = creature_to_json(&creature).map_err(|e| e.to_string())?;
            std::fs::write(path, json).map_err(|e| format!("best.json: {e}"))?;
        }
        Some(winner)
    } else {
        None
    };

    Ok(FullOutcome {
        incumbent_score: baseline.score,
        incumbent_error: baseline.error,
        individuals,
        bundles,
        sample_false_positives,
        winner,
        full_ms,
    })
}

fn write_creature(dir: &Path, stem: &str, creature: &CreatureExport) -> Result<(), String> {
    let json = creature_to_json(creature).map_err(|e| format!("{stem}: {e}"))?;
    std::fs::write(dir.join(format!("{stem}.json")), json).map_err(|e| format!("{stem}: {e}"))
}

fn full_candidate(
    stem: String,
    kind: &'static str,
    uuids: Vec<String>,
    creature: &CreatureExport,
    result: &ScoreResult,
    incumbent_score: f64,
) -> FullCandidate {
    FullCandidate {
        stem,
        kind,
        uuids,
        score: result.score,
        error: result.error,
        complexity_penalty: result.complexity_penalty,
        after: StructureSnapshot::of(creature),
        delta: result.score - incumbent_score,
    }
}

/// Build a [`SampledWinner`] for tests from an already-valid candidate.
pub fn sampled(
    candidate: SweepCandidate,
    sample_score: f64,
    sample_baseline: f64,
) -> SampledWinner {
    SampledWinner {
        delta: sample_score - sample_baseline,
        score: sample_score,
        baseline_score: sample_baseline,
        candidate,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::baseline::fake::ScriptedScorer;
    use crate::fixtures::{creature, neuron, synapse};
    use crate::incumbent::validate_creature;
    use crate::stats::{ActivationStats, NeuronStats, STATS_FORMAT_VERSION};
    use crate::sweep::Sweep;
    use std::collections::BTreeMap;

    fn three_hidden() -> CreatureExport {
        creature(
            1,
            1,
            vec![
                neuron("hidden", "h1", 0.0, Some("IDENTITY")),
                neuron("hidden", "h2", 0.0, Some("IDENTITY")),
                neuron("hidden", "h3", 0.0, Some("IDENTITY")),
                neuron("output", "output-0", 0.0, Some("IDENTITY")),
            ],
            vec![
                synapse("input-0", "h1", 1.0),
                synapse("input-0", "h2", 1.0),
                synapse("input-0", "h3", 1.0),
                synapse("h1", "output-0", 1.0),
                synapse("h2", "output-0", 1.0),
                synapse("h3", "output-0", 1.0),
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

    fn candidates(incumbent: &CreatureExport, stats: &ActivationStats) -> Vec<SweepCandidate> {
        let mut sweep = Sweep::new(incumbent, 1);
        let (batch, skips) = sweep.fill_batch(incumbent, stats, 8);
        assert!(skips.is_empty(), "{skips:?}");
        batch
    }

    fn winners_from(batch: Vec<SweepCandidate>, sample_scores: &[f64]) -> Vec<SampledWinner> {
        batch
            .into_iter()
            .zip(sample_scores)
            .map(|(c, s)| sampled(c, *s, 0.50))
            .collect()
    }

    #[test]
    fn sample_false_positive_is_rejected_by_full_scoring() {
        let incumbent = three_hidden();
        validate_creature(&incumbent).unwrap();
        let stats = stats_for(&incumbent);
        let batch = candidates(&incumbent, &stats);
        let one = vec![sampled(batch.into_iter().next().unwrap(), 0.90, 0.50)];
        let tmp = tempfile::tempdir().unwrap();
        let best = tmp.path().join("best.json");
        let mut stem_scores = BTreeMap::new();
        stem_scores.insert("baseline".into(), 0.50);
        stem_scores.insert("i000".into(), 0.40);
        let scorer = ScriptedScorer {
            stem_scores,
            ..ScriptedScorer::ok(0.50, 0.50)
        };
        let out = evaluate_full(
            &scorer,
            tmp.path(),
            &incumbent,
            &stats,
            &one,
            FullConfig {
                min_improvement: 1e-6,
                dir: &tmp.path().join("full"),
                best_path: Some(&best),
            },
        )
        .unwrap();
        assert!(out.winner.is_none());
        assert_eq!(out.sample_false_positives.len(), 1);
        assert!(!best.exists(), "sample win must not write best.json");
    }

    #[test]
    fn interacting_bundle_is_not_assumed_additive() {
        let incumbent = three_hidden();
        let stats = stats_for(&incumbent);
        let batch = candidates(&incumbent, &stats);
        assert!(batch.len() >= 2);
        let sampled = winners_from(batch.into_iter().take(2).collect(), &[0.70, 0.60]);
        let tmp = tempfile::tempdir().unwrap();
        let mut stem_scores = BTreeMap::new();
        stem_scores.insert("baseline".into(), 0.50);
        stem_scores.insert("i000".into(), 0.70);
        stem_scores.insert("i001".into(), 0.60);
        stem_scores.insert("b000".into(), 0.40);
        let scorer = ScriptedScorer {
            stem_scores,
            ..ScriptedScorer::ok(0.50, 0.50)
        };
        let out = evaluate_full(
            &scorer,
            tmp.path(),
            &incumbent,
            &stats,
            &sampled,
            FullConfig {
                min_improvement: 1e-6,
                dir: &tmp.path().join("full"),
                best_path: None,
            },
        )
        .unwrap();
        assert_eq!(out.bundles.len(), 1);
        assert!(out.bundles[0].delta < 0.0);
        let win = out.winner.expect("an individual should win");
        assert_eq!(win.candidate.kind, "individual");
        assert!(win.candidate.delta > 0.0);
    }

    #[test]
    fn bundle_can_outperform_every_individual() {
        let incumbent = three_hidden();
        let stats = stats_for(&incumbent);
        let batch = candidates(&incumbent, &stats);
        let sampled = winners_from(batch.into_iter().take(2).collect(), &[0.52, 0.51]);
        let tmp = tempfile::tempdir().unwrap();
        let mut stem_scores = BTreeMap::new();
        stem_scores.insert("baseline".into(), 0.50);
        stem_scores.insert("i000".into(), 0.52);
        stem_scores.insert("i001".into(), 0.51);
        stem_scores.insert("b000".into(), 0.80);
        let scorer = ScriptedScorer {
            stem_scores,
            ..ScriptedScorer::ok(0.50, 0.50)
        };
        let out = evaluate_full(
            &scorer,
            tmp.path(),
            &incumbent,
            &stats,
            &sampled,
            FullConfig {
                min_improvement: 1e-6,
                dir: &tmp.path().join("full"),
                best_path: None,
            },
        )
        .unwrap();
        let win = out.winner.expect("bundle should win");
        assert_eq!(win.candidate.kind, "bundle");
        assert_eq!(win.candidate.score, 0.80);
    }

    #[test]
    fn tiny_positive_full_delta_is_accepted_as_next_parent() {
        let incumbent = three_hidden();
        let stats = stats_for(&incumbent);
        let batch = candidates(&incumbent, &stats);
        let one = vec![sampled(batch.into_iter().next().unwrap(), 0.51, 0.50)];
        let tmp = tempfile::tempdir().unwrap();
        let best = tmp.path().join("out").join("best.json");
        let mut stem_scores = BTreeMap::new();
        stem_scores.insert("baseline".into(), 0.50);
        stem_scores.insert("i000".into(), 0.50 + 2e-6);
        let scorer = ScriptedScorer {
            stem_scores,
            ..ScriptedScorer::ok(0.50, 0.50)
        };
        let out = evaluate_full(
            &scorer,
            tmp.path(),
            &incumbent,
            &stats,
            &one,
            FullConfig {
                min_improvement: 1e-6,
                dir: &tmp.path().join("full"),
                best_path: Some(&best),
            },
        )
        .unwrap();
        let win = out.winner.expect("tiny win");
        assert!(win.candidate.delta > 1e-6);
        assert!(best.exists());
        assert_eq!(win.candidate.kind, "individual");
    }
}
