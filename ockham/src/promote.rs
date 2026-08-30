//! Full-corpus scoring of sampled winners and grouped bundles (Issue #7).
//!
//! Every sampled winner is scored on the full corpus together with the current
//! Ockham incumbent, and combination bundles are built from **every** winner —
//! this batch's and the confirmed ones carried forward (Issues #45, #54, #55).
//! Individual score deltas are never assumed additive: [`apply_bundle`]
//! re-proposes each cut against the creature the previous cut produced.
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
use crate::sweep::{CandidateKind, SampledWinner, SweepCandidate, propose};

/// Most bundle plans one cohort may carry (Issue #55).
///
/// A batch can screen hundreds of winners; without a cap the plan generator
/// would eat the wall clock it is meant to spend on cuts.
pub const MAX_BUNDLE_PLANS: usize = 12;

/// Winner count at which complementary disjoint plans start being emitted.
///
/// Below this a nested prefix chain already covers the list, and the halves
/// would duplicate plans that exist anyway.
pub const DISJOINT_HALVES_MIN: usize = 8;

/// Smallest number of members a kind-grouped plan is worth emitting for.
const KIND_GROUP_MIN: usize = 2;

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
    /// Plans dropped because a cut in them no longer proposed (Issue #55).
    pub skipped_bundles: usize,
    /// Individual entries dropped to fit the wall-clock budget (Issue #58).
    pub dropped_individuals: usize,
    /// Bundle entries dropped to fit the wall-clock budget (Issue #58).
    pub dropped_bundles: usize,
    /// Plans the generator's own cap refused to build (Issue #55).
    pub capped_plans: usize,
}

impl FullOutcome {
    /// Creatures the scorer was asked for, excluding the incumbent baseline.
    pub fn entries(&self) -> usize {
        self.individuals.len() + self.bundles.len()
    }

    /// Entries dropped to fit the budget.
    pub fn dropped(&self) -> usize {
        self.dropped_individuals + self.dropped_bundles
    }
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
    /// Extra UUID plans scored as bundles (largest-first replay, etc.).
    pub extra_plans: &'a [Vec<String>],
    /// Cap on winners written as **individual** cohort entries (Issue #54).
    ///
    /// Never restricts bundle membership: an operator capping a short run's
    /// individual scoring must not silently shrink the combinations tried.
    pub max_individuals: Option<usize>,
    /// Confirmed winners carried from earlier batches (Issue #56).
    ///
    /// Bundle membership only — they already have a full-corpus verdict, so
    /// re-scoring them individually would buy nothing.
    pub pool: &'a [BundleMember],
    /// Cap on entries actually scored, to fit the wall clock (Issue #58).
    pub max_entries: Option<usize>,
}

impl<'a> FullConfig<'a> {
    /// Full-score config with no extra plans, no caps and no carried pool.
    pub fn new(min_improvement: f64, dir: &'a Path, best_path: Option<&'a Path>) -> Self {
        Self {
            min_improvement,
            dir,
            best_path,
            extra_plans: &[],
            max_individuals: None,
            pool: &[],
            max_entries: None,
        }
    }
}

/// One winner offered to bundle construction.
///
/// Bundles are UUID plans re-proposed from the incumbent, so a member needs
/// only its uuid, the kind of cut it represents and the delta it is ranked by
/// — never a candidate creature. That is what lets a confirmed winner from an
/// earlier batch join a bundle without being scored again (Issue #56).
#[derive(Debug, Clone, PartialEq)]
pub struct BundleMember {
    /// Hidden neuron UUID.
    pub uuid: String,
    /// How the cut was proposed.
    pub kind: CandidateKind,
    /// Ranking delta: the sample delta for a fresh winner, the measured
    /// full-corpus delta for a carried one.
    pub delta: f64,
}

/// Bundle members for this batch's sampled winners.
pub fn members_of(winners: &[SampledWinner]) -> Vec<BundleMember> {
    winners
        .iter()
        .map(|w| BundleMember {
            uuid: w.candidate.uuid.clone(),
            kind: w.candidate.kind,
            delta: w.delta,
        })
        .collect()
}

/// Bundle plans for one cohort, in **keep-priority order** (Issue #55).
#[derive(Debug, Clone, PartialEq, Default)]
pub struct BundlePlans {
    /// Plans to score, most valuable first.
    pub plans: Vec<Vec<String>>,
    /// How many leading plans are structurally distinct rather than nested
    /// prefixes: the all-winners plan, the disjoint halves and the kind groups.
    pub primary: usize,
    /// Plans the cap refused to emit.
    pub dropped: usize,
}

/// Rank members by delta and build combination plans over **all** of them.
///
/// A single ranked prefix chain cannot localise a winner that poisons every
/// bundle it joins: if member three is the culprit, every prefix from four up
/// is spoiled and the run learns nothing about the other thirty-five. So the
/// generator emits structurally different plans and orders them by how much
/// they are worth keeping when the budget bites (Issue #58):
///
/// 1. the all-winners plan;
/// 2. the two disjoint halves, which between them isolate a poisoning member
///    to one half in a single cohort;
/// 3. all-identity and all-ablation groups, which interact with cleanup
///    differently;
/// 4. nested power-of-two prefixes, largest first.
///
/// Plans shorter than two are skipped, duplicates are de-duplicated on the
/// joined uuid key, and the order is deterministic for a given member list —
/// ties in delta are broken by uuid.
pub fn bundle_plans(members: &[BundleMember]) -> BundlePlans {
    let mut ranked = members.to_vec();
    ranked.sort_by(|a, b| {
        b.delta
            .partial_cmp(&a.delta)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.uuid.cmp(&b.uuid))
    });
    let ids: Vec<String> = ranked.iter().map(|m| m.uuid.clone()).collect();
    let n = ids.len();

    let mut out = BundlePlans::default();
    let mut seen = HashSet::new();
    let mut push = |out: &mut BundlePlans, plan: Vec<String>| {
        if plan.len() < 2 {
            return;
        }
        if !seen.insert(plan.join("\0")) {
            return;
        }
        if out.plans.len() >= MAX_BUNDLE_PLANS {
            out.dropped += 1;
            return;
        }
        out.plans.push(plan);
    };

    if n >= 2 {
        push(&mut out, ids.clone());
    }
    if n >= DISJOINT_HALVES_MIN {
        let half = n / 2;
        push(&mut out, ids[..half].to_vec());
        push(&mut out, ids[half..].to_vec());
    }
    let by_kind = |kind: CandidateKind| -> Vec<String> {
        ranked
            .iter()
            .filter(|m| m.kind == kind)
            .map(|m| m.uuid.clone())
            .collect()
    };
    let identity = by_kind(CandidateKind::Identity);
    let ablation = by_kind(CandidateKind::Ablation);
    if identity.len() >= KIND_GROUP_MIN && ablation.len() >= KIND_GROUP_MIN {
        push(&mut out, identity);
        push(&mut out, ablation);
    }
    out.primary = out.plans.len();

    let mut prefixes: Vec<usize> = Vec::new();
    let mut len = 2usize;
    while len < n {
        prefixes.push(len);
        len *= 2;
    }
    for len in prefixes.into_iter().rev() {
        push(&mut out, ids[..len].to_vec());
    }
    out
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

/// Apply every UUID that still proposes, skipping the rest.
///
/// Used by known-win replay: a stale cut must not abort the whole bundle.
pub fn apply_available(
    incumbent: &CreatureExport,
    stats: &ActivationStats,
    uuids: &[String],
) -> (Vec<String>, CreatureExport) {
    let mut current = incumbent.clone();
    let mut applied = Vec::new();
    for uuid in uuids {
        if current.neurons.iter().all(|n| n.uuid != *uuid) {
            continue;
        }
        match propose(&current, stats, uuid) {
            Ok((_, next)) => {
                current = next;
                applied.push(uuid.clone());
            }
            Err(_) => continue,
        }
    }
    (applied, current)
}

/// Most plans one replay cohort may carry (Issue #57).
pub const MAX_REPLAY_PLANS: usize = 6;

/// Known wins to probe individually when every replay plan misses (Issue #57).
///
/// Replaces the old `--max-full` fallback: an individual-scoring cap for the
/// search loop was never a sensible size for a replay probe.
pub const REPLAY_PROBE_LIMIT: usize = 8;

/// Combined replay plan plus bounded greedy shrink steps (Issue #57).
///
/// `applied` must already be ranked best-measured-delta first: the plans are
/// largest-first prefixes, so shrinking drops the **weakest** members rather
/// than the most recently filed ones. Every step is a plan in the same cohort,
/// so "does the bundle work without its two worst members?" is answered by the
/// scorer call that asked the combined question, not a run later.
///
/// Lengths are `n`, `n - 1`, `n / 2`, `16`, `8`, `4`, de-duplicated, clamped to
/// `2..=n` and capped at [`MAX_REPLAY_PLANS`].
pub fn replay_plans(applied: &[String]) -> Vec<Vec<String>> {
    let n = applied.len();
    if n < 2 {
        return Vec::new();
    }
    let mut lengths: Vec<usize> = [n, n.saturating_sub(1), n / 2, 16, 8, 4]
        .into_iter()
        .filter(|len| (2..=n).contains(len))
        .collect();
    lengths.sort_unstable_by(|a, b| b.cmp(a));
    lengths.dedup();
    lengths.truncate(MAX_REPLAY_PLANS);
    lengths
        .into_iter()
        .map(|len| applied[..len].to_vec())
        .collect()
}

/// Full-score sampled winners and their combination bundles in one scorer call.
///
/// Individuals come from `sampled`, ranked by sample delta and capped by
/// [`FullConfig::max_individuals`]. Bundles are built over every winner —
/// `sampled` plus [`FullConfig::pool`] — because a cap on individual scoring
/// must never decide which combinations are tried (Issue #54). When
/// [`FullConfig::max_entries`] is set, the cohort is trimmed to fit the run's
/// wall clock, keeping the structurally distinct plans and the strongest
/// individuals and dropping nested prefixes first (Issue #58).
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

    // Rank here rather than trusting the caller: the cap must keep the best
    // sample deltas, and equal deltas must break the same way on every host.
    let mut ranked: Vec<&SampledWinner> = sampled.iter().collect();
    ranked.sort_by(|a, b| {
        b.delta
            .partial_cmp(&a.delta)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.candidate.uuid.cmp(&b.candidate.uuid))
    });
    if let Some(cap) = cfg.max_individuals {
        ranked.truncate(cap);
    }

    // Bundle membership is the union of this batch's winners and the confirmed
    // winners carried forward; a fresh sample delta wins a uuid collision.
    let mut members = members_of(sampled);
    let fresh: HashSet<&str> = members.iter().map(|m| m.uuid.as_str()).collect();
    let carried: Vec<BundleMember> = cfg
        .pool
        .iter()
        .filter(|m| !fresh.contains(m.uuid.as_str()))
        .cloned()
        .collect();
    members.extend(carried);
    let plans = bundle_plans(&members);

    // Trim priority: structurally distinct plans, then the strongest
    // individuals, then the nested prefixes that add the least.
    let mut ordered: Vec<Entry<'_>> = Vec::new();
    for plan in cfg
        .extra_plans
        .iter()
        .chain(plans.plans[..plans.primary].iter())
    {
        ordered.push(Entry::Bundle(plan.clone()));
    }
    ordered.extend(ranked.into_iter().map(Entry::Individual));
    for plan in &plans.plans[plans.primary..] {
        ordered.push(Entry::Bundle(plan.clone()));
    }
    let (dropped_individuals, dropped_bundles) = match cfg.max_entries {
        Some(max) => {
            let keep = max.max(1);
            let dropped: (usize, usize) =
                ordered.iter().skip(keep).fold((0, 0), |acc, e| match e {
                    Entry::Individual(_) => (acc.0 + 1, acc.1),
                    Entry::Bundle(_) => (acc.0, acc.1 + 1),
                });
            ordered.truncate(keep);
            dropped
        }
        None => (0, 0),
    };

    // Trim priority decided which entries survive; the cohort is still written
    // individuals-first, in the order each group was ranked, so stems and the
    // first-strictly-better winner rule keep the meaning they always had.
    let mut pending: Vec<(String, &'static str, Vec<String>, CreatureExport)> = Vec::new();
    let mut skipped_bundles = 0usize;
    for (i, w) in ordered
        .iter()
        .filter_map(|e| match e {
            Entry::Individual(w) => Some(*w),
            Entry::Bundle(_) => None,
        })
        .enumerate()
    {
        let stem = format!("i{i:03}");
        write_creature(cfg.dir, &stem, &w.candidate.creature)?;
        pending.push((
            stem,
            "individual",
            vec![w.candidate.uuid.clone()],
            w.candidate.creature.clone(),
        ));
    }
    let mut bundle_i = 0usize;
    for plan in ordered.into_iter().filter_map(|e| match e {
        Entry::Bundle(plan) => Some(plan),
        Entry::Individual(_) => None,
    }) {
        match apply_bundle(incumbent, stats, &plan) {
            Ok(creature) => {
                let stem = format!("b{bundle_i:03}");
                bundle_i += 1;
                write_creature(cfg.dir, &stem, &creature)?;
                pending.push((stem, "bundle", plan, creature));
            }
            Err(_) => skipped_bundles += 1,
        }
    }

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
        skipped_bundles,
        dropped_individuals,
        dropped_bundles,
        capped_plans: plans.dropped,
    })
}

/// One cohort entry before it is written and scored.
enum Entry<'a> {
    /// A sampled winner scored on its own.
    Individual(&'a SampledWinner),
    /// A UUID plan applied in order from the incumbent.
    Bundle(Vec<String>),
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
            FullConfig::new(1e-6, &tmp.path().join("full"), Some(&best)),
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
            FullConfig::new(1e-6, &tmp.path().join("full"), None),
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
            FullConfig::new(1e-6, &tmp.path().join("full"), None),
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
            FullConfig::new(1e-6, &tmp.path().join("full"), Some(&best)),
        )
        .unwrap();
        let win = out.winner.expect("tiny win");
        assert!(win.candidate.delta > 1e-6);
        assert!(best.exists());
        assert_eq!(win.candidate.kind, "individual");
    }

    #[test]
    fn apply_available_skips_unproposable_uuids() {
        let incumbent = three_hidden();
        let stats = stats_for(&incumbent);
        let before = incumbent
            .neurons
            .iter()
            .filter(|n| n.neuron_type == "hidden")
            .count();
        let (applied, creature) =
            apply_available(&incumbent, &stats, &["nope".into(), "h1".into()]);
        assert_eq!(applied, vec!["h1".to_string()]);
        let after = creature
            .neurons
            .iter()
            .filter(|n| n.neuron_type == "hidden")
            .count();
        assert!(after < before);
    }

    #[test]
    fn replay_plans_shrink_from_the_weakest_end() {
        let ids: Vec<String> = (0..10).map(|i| format!("h{i}")).collect();
        let plans = replay_plans(&ids);
        let lengths: Vec<usize> = plans.iter().map(Vec::len).collect();
        assert_eq!(lengths, vec![10, 9, 8, 5, 4]);
        for plan in &plans {
            assert_eq!(plan[0], "h0", "every plan keeps the strongest member");
        }
        assert!(
            plans[1].iter().all(|u| u != "h9"),
            "the first shrink drops the weakest: {:?}",
            plans[1]
        );
        assert!(replay_plans(&ids[..1]).is_empty());
    }

    #[test]
    fn a_replay_cohort_is_bounded() {
        let ids: Vec<String> = (0..40).map(|i| format!("h{i:02}")).collect();
        let plans = replay_plans(&ids);
        assert!(plans.len() <= MAX_REPLAY_PLANS, "{}", plans.len());
        assert_eq!(plans[0].len(), 40, "the combined plan is tried first");
        assert!(plans.len() > 1, "a combined miss must have somewhere to go");
    }

    /// Bundle members with alternating kinds, deltas descending by index.
    fn members(n: usize) -> Vec<BundleMember> {
        (0..n)
            .map(|i| BundleMember {
                uuid: format!("h{i:02}"),
                kind: if i % 2 == 0 {
                    CandidateKind::Identity
                } else {
                    CandidateKind::Ablation
                },
                delta: 1.0 - i as f64 / 100.0,
            })
            .collect()
    }

    #[test]
    fn thirty_eight_winners_produce_every_power_of_two_prefix_and_the_whole_set() {
        let plans = bundle_plans(&members(38));
        let lengths: HashSet<usize> = plans.plans.iter().map(Vec::len).collect();
        for n in [2usize, 4, 8, 16, 32, 38] {
            assert!(lengths.contains(&n), "missing a {n}-cut plan: {lengths:?}");
        }
        assert_eq!(plans.dropped, 0);
        assert!(plans.plans.len() <= MAX_BUNDLE_PLANS);
    }

    #[test]
    fn thirty_eight_winners_produce_two_complementary_disjoint_plans() {
        let plans = bundle_plans(&members(38));
        // Documented order: all-winners, then the two halves.
        assert_eq!(plans.plans[0].len(), 38);
        let a: HashSet<&String> = plans.plans[1].iter().collect();
        let b: HashSet<&String> = plans.plans[2].iter().collect();
        assert_eq!(a.len(), 19);
        assert_eq!(b.len(), 19);
        assert!(a.is_disjoint(&b), "the halves must not overlap");
        assert_eq!(a.union(&b).count(), 38, "their union must be every winner");
    }

    #[test]
    fn both_kinds_present_emit_an_identity_and_an_ablation_plan() {
        let plans = bundle_plans(&members(38));
        let identity: Vec<String> = members(38)
            .iter()
            .filter(|m| m.kind == CandidateKind::Identity)
            .map(|m| m.uuid.clone())
            .collect();
        assert!(
            plans.plans.iter().any(|p| {
                p.len() == identity.len()
                    && p.iter().collect::<HashSet<_>>() == identity.iter().collect::<HashSet<_>>()
            }),
            "no all-identity plan: {:?}",
            plans.plans
        );
    }

    #[test]
    fn one_kind_only_emits_no_duplicate_of_the_all_winners_plan() {
        let single: Vec<BundleMember> = members(10)
            .into_iter()
            .map(|m| BundleMember {
                kind: CandidateKind::Ablation,
                ..m
            })
            .collect();
        let plans = bundle_plans(&single);
        let all = plans.plans.iter().filter(|p| p.len() == 10).count();
        assert_eq!(
            all, 1,
            "the all-winners plan appears once: {:?}",
            plans.plans
        );
    }

    #[test]
    fn two_winners_still_produce_exactly_one_plan() {
        let plans = bundle_plans(&members(2));
        assert_eq!(plans.plans.len(), 1);
        assert_eq!(plans.plans[0].len(), 2);
        assert_eq!(plans.primary, 1);
        assert_eq!(bundle_plans(&members(1)).plans.len(), 0);
        assert_eq!(bundle_plans(&[]).plans.len(), 0);
    }

    #[test]
    fn the_plan_cap_bounds_the_cohort_and_says_what_it_dropped() {
        let plans = bundle_plans(&members(4096));
        assert_eq!(plans.plans.len(), MAX_BUNDLE_PLANS);
        assert!(plans.dropped > 0, "a dropped count is the audit trail");
    }

    #[test]
    fn plans_are_deduplicated_and_deterministic() {
        let first = bundle_plans(&members(20));
        let mut shuffled = members(20);
        shuffled.reverse();
        assert_eq!(first, bundle_plans(&shuffled), "ranking, not input order");
        let keys: HashSet<String> = first.plans.iter().map(|p| p.join("\0")).collect();
        assert_eq!(keys.len(), first.plans.len(), "no duplicate plans");
    }

    /// Winners whose kinds and deltas are known, for cohort-shape assertions.
    fn many_winners(n: usize) -> (CreatureExport, ActivationStats, Vec<SampledWinner>) {
        let uuids: Vec<String> = (0..n).map(|i| format!("h{i:02}")).collect();
        let mut neurons: Vec<_> = uuids
            .iter()
            .map(|u| neuron("hidden", u, 0.0, Some("IDENTITY")))
            .collect();
        neurons.push(neuron("output", "output-0", 0.0, Some("IDENTITY")));
        let mut synapses = Vec::new();
        for u in &uuids {
            synapses.push(synapse("input-0", u, 1.0));
            synapses.push(synapse(u, "output-0", 1.0));
        }
        let incumbent = creature(1, 1, neurons, synapses);
        validate_creature(&incumbent).unwrap();
        let stats = stats_for(&incumbent);
        let mut sweep = Sweep::new(&incumbent, 1);
        let (batch, _) = sweep.fill_batch(&incumbent, &stats, n);
        let winners = batch
            .into_iter()
            .enumerate()
            .map(|(i, c)| sampled(c, 0.60 - i as f64 / 1000.0, 0.50))
            .collect();
        (incumbent, stats, winners)
    }

    /// The exact shape of the fleet log in Issue #45: 38 screened winners and
    /// the `--max-full 8` GRQ used to pass.
    #[test]
    fn a_cap_on_individual_scoring_never_shrinks_bundle_membership() {
        let (incumbent, stats, winners) = many_winners(38);
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join("full");
        let scorer = ScriptedScorer::ok(0.50, 0.50);
        let out = evaluate_full(
            &scorer,
            tmp.path(),
            &incumbent,
            &stats,
            &winners,
            FullConfig {
                max_individuals: Some(8),
                ..FullConfig::new(1e-6, &dir, None)
            },
        )
        .unwrap();
        assert_eq!(out.individuals.len(), 8, "the cap applies to individuals");
        assert!(
            out.bundles.iter().any(|b| b.uuids.len() == 38),
            "bundles must still see every winner: {:?}",
            out.bundles
                .iter()
                .map(|b| b.uuids.len())
                .collect::<Vec<_>>()
        );
        // The kept individuals are the strongest sample deltas.
        let kept: HashSet<&String> = out.individuals.iter().flat_map(|c| &c.uuids).collect();
        let mut ranked = winners.clone();
        ranked.sort_by(|a, b| {
            b.delta
                .partial_cmp(&a.delta)
                .unwrap()
                .then_with(|| a.candidate.uuid.cmp(&b.candidate.uuid))
        });
        for w in ranked.iter().take(8) {
            assert!(kept.contains(&w.candidate.uuid), "{kept:?}");
        }
    }

    /// Equal sample deltas must break the same way on every host, or two hosts
    /// under one cap score different individuals from the same screen.
    #[test]
    fn a_cap_on_equal_deltas_is_deterministic() {
        let (incumbent, stats, winners) = many_winners(8);
        let flat: Vec<SampledWinner> = winners
            .into_iter()
            .map(|w| SampledWinner { delta: 0.1, ..w })
            .collect();
        let tmp = tempfile::tempdir().unwrap();
        let kept = |dir: &str| -> Vec<String> {
            let out = evaluate_full(
                &ScriptedScorer::ok(0.50, 0.50),
                tmp.path(),
                &incumbent,
                &stats,
                &flat,
                FullConfig {
                    max_individuals: Some(3),
                    ..FullConfig::new(1e-6, &tmp.path().join(dir), None)
                },
            )
            .unwrap();
            out.individuals
                .iter()
                .filter_map(|c| c.uuids.first().cloned())
                .collect()
        };
        let mut expected: Vec<String> = flat.iter().map(|w| w.candidate.uuid.clone()).collect();
        expected.sort();
        expected.truncate(3);
        assert_eq!(kept("a"), expected, "ties break on uuid");
        assert_eq!(kept("b"), kept("c"), "and repeat");
    }

    #[test]
    fn no_cap_full_scores_every_winner_individually() {
        let (incumbent, stats, winners) = many_winners(12);
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join("full");
        let out = evaluate_full(
            &ScriptedScorer::ok(0.50, 0.50),
            tmp.path(),
            &incumbent,
            &stats,
            &winners,
            FullConfig::new(1e-6, &dir, None),
        )
        .unwrap();
        assert_eq!(out.individuals.len(), 12);
        assert_eq!(out.dropped(), 0);
    }

    #[test]
    fn carried_winners_join_bundles_without_being_scored_individually() {
        let (incumbent, stats, winners) = many_winners(12);
        let (fresh, carried) = winners.split_at(4);
        let pool: Vec<BundleMember> = members_of(carried);
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join("full");
        let out = evaluate_full(
            &ScriptedScorer::ok(0.50, 0.50),
            tmp.path(),
            &incumbent,
            &stats,
            fresh,
            FullConfig {
                pool: &pool,
                ..FullConfig::new(1e-6, &dir, None)
            },
        )
        .unwrap();
        assert_eq!(out.individuals.len(), 4, "only this batch's winners");
        assert!(
            out.bundles.iter().any(|b| b.uuids.len() == 12),
            "the pool joins bundle membership"
        );
    }

    #[test]
    fn a_tight_budget_keeps_the_distinct_plans_and_drops_nested_prefixes() {
        let (incumbent, stats, winners) = many_winners(12);
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join("full");
        let untrimmed = evaluate_full(
            &ScriptedScorer::ok(0.50, 0.50),
            tmp.path(),
            &incumbent,
            &stats,
            &winners,
            FullConfig::new(1e-6, &dir, None),
        )
        .unwrap();
        assert!(untrimmed.entries() > 6);

        let dir = tmp.path().join("trimmed");
        let out = evaluate_full(
            &ScriptedScorer::ok(0.50, 0.50),
            tmp.path(),
            &incumbent,
            &stats,
            &winners,
            FullConfig {
                max_entries: Some(6),
                ..FullConfig::new(1e-6, &dir, None)
            },
        )
        .unwrap();
        assert!(out.entries() <= 6, "{}", out.entries());
        assert_eq!(
            out.entries() + out.dropped(),
            untrimmed.entries(),
            "every dropped entry is counted"
        );
        assert!(out.dropped_individuals > 0);
        assert!(
            out.bundles.iter().any(|b| b.uuids.len() == 12),
            "the all-winners plan survives a trim: {:?}",
            out.bundles
                .iter()
                .map(|b| b.uuids.len())
                .collect::<Vec<_>>()
        );
        assert_eq!(
            out.bundles.iter().filter(|b| b.uuids.len() == 6).count(),
            2,
            "so do the disjoint halves"
        );
    }

    #[test]
    fn a_stale_plan_is_counted_rather_than_discarded() {
        let incumbent = three_hidden();
        let stats = stats_for(&incumbent);
        let batch = candidates(&incumbent, &stats);
        let one = vec![sampled(batch.into_iter().next().unwrap(), 0.60, 0.50)];
        let tmp = tempfile::tempdir().unwrap();
        let gone = vec!["h1".to_string(), "not-a-neuron".to_string()];
        let out = evaluate_full(
            &ScriptedScorer::ok(0.50, 0.50),
            tmp.path(),
            &incumbent,
            &stats,
            &one,
            FullConfig {
                extra_plans: std::slice::from_ref(&gone),
                ..FullConfig::new(1e-6, &tmp.path().join("full"), None)
            },
        )
        .unwrap();
        assert_eq!(out.skipped_bundles, 1);
        assert!(out.bundles.is_empty());
    }
}
