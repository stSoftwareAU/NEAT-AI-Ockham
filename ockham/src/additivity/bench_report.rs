//! Report types and aggregation for the additivity benchmark (Issue #260).
//!
//! Everything here is serialised as camelCase JSON by the `bench-additivity`
//! command. Nothing in this module decides anything: it only summarises rows
//! that [`super::bench`] has already scored with the full-corpus scorer.

use std::collections::BTreeMap;

use serde::Serialize;

use super::manifest::{BenchManifest, REPRESENTATIVE_PARENTS};
use super::rank::{self, ARMS, Arm, AuditMetrics, Distribution, PairedComparison};

/// Layout version of the report JSON. Version 2 split the search pool from the
/// audit pool and added the search-side fields.
pub const REPORT_FORMAT_VERSION: u32 = 2;

/// Statement written into every report about who may accept a creature.
pub const AUTHORITY: &str = "predictions only rank candidates for full evaluation; the full-corpus \
     scorer remains the only acceptance authority; nothing is accepted, published or written to \
     any population, journal or learnings store";

/// Full scores closer than this are ties, matching [`rank::paired`].
const TIE_TOLERANCE: f64 = 1e-12;

/// The predeclared protocol, echoed verbatim so a report stands on its own.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ManifestEcho {
    /// Pruning budgets as fractions of each parent's hidden neurons.
    pub budgets: Vec<f64>,
    /// Seeds, one independent run per (parent, budget, seed).
    pub seeds: Vec<u64>,
    /// Fraction of the training corpus used as the audit core.
    pub core_fraction: f64,
    /// Cost function name.
    pub cost: String,
    /// Bundles audited per (parent, budget, seed).
    pub audit_pool: usize,
    /// Per-arm full-scorer budget: at most this many search-pool picks are scored.
    pub full_evaluations: usize,
    /// Top-ranked bundles counted by the top-k recall.
    pub top_k: usize,
    /// Promotion threshold used by the audit.
    pub min_improvement: f64,
    /// Predeclared per-arm search budget in milliseconds, or `None` when the
    /// search is not wall-clock limited.
    pub search_wall_clock_ms: Option<u64>,
    /// Predeclared adoption rule, verbatim.
    pub decision_rule: String,
}

impl ManifestEcho {
    /// Echo of `manifest`'s predeclared protocol.
    #[must_use]
    pub fn of(manifest: &BenchManifest) -> Self {
        Self {
            budgets: manifest.budgets.clone(),
            seeds: manifest.seeds.clone(),
            core_fraction: manifest.core_fraction,
            cost: manifest.cost.clone(),
            audit_pool: manifest.audit_pool,
            full_evaluations: manifest.full_evaluations,
            top_k: manifest.top_k,
            min_improvement: manifest.min_improvement,
            search_wall_clock_ms: manifest.search_wall_clock_ms,
            decision_rule: manifest.decision_rule.clone(),
        }
    }
}

/// A single-neuron cut the core engine refused. It is never scored.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct FailedProposal {
    /// Hidden neuron UUID.
    pub uuid: String,
    /// Core's refusal message.
    pub reason: String,
}

/// Output-cache counters for one parent.
#[derive(Debug, Clone, Default, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CacheStats {
    /// Entries served from disk without rebuilding.
    pub hits: u64,
    /// Entries built by streaming the core records.
    pub builds: u64,
    /// Rejections counted by kind (`nonFinite`, `mismatched`, `overLimit`, …).
    pub rejections: BTreeMap<String, u64>,
    /// Bytes held by finished entries at the end of the parent's run.
    pub bytes_used: u64,
    /// Largest byte count observed during the parent's run.
    pub peak_bytes: u64,
    /// Configured byte limit.
    pub max_cache_bytes: u64,
    /// Computed bound on the bench's own RAM, in bytes, excluding the parent and
    /// the bundle creatures: one cache entry's accumulator plus fixed chunk
    /// buffers. The bench holds at most `auditPool` materialised bundle creatures
    /// at once, the audit pool then the search pool, never both. The measured
    /// peak is `measuredPeakRssBytes` on the report. It never holds neurons ×
    /// records × outputs.
    pub ram_bound_bytes: u64,
    /// Wall time spent building entries, in milliseconds.
    pub build_ms: u64,
    /// Wall time spent reading entries for losses and predictions, in milliseconds.
    pub read_ms: u64,
}

/// Per-parent setup: baseline, singles and cache counters.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ParentReport {
    /// Parent file name.
    pub parent: String,
    /// Parent creature checksum.
    pub parent_checksum: String,
    /// Authoritative parent score (larger is better).
    pub parent_score: f64,
    /// Authoritative parent error.
    pub parent_error: f64,
    /// Authoritative parent complexity penalty.
    pub parent_complexity_penalty: f64,
    /// Hidden neurons in the parent.
    pub hidden_neurons: usize,
    /// Records in the deterministic audit core.
    pub core_records: u64,
    /// Single cuts the core engine proposed.
    pub proposable: usize,
    /// Proposable singles with a cached loss.
    pub estimated: usize,
    /// Single cuts refused by the core engine.
    pub failed_proposals: Vec<FailedProposal>,
    /// Why estimates are unavailable for the whole parent, when they are.
    pub estimate_blocked: Option<String>,
    /// Wall time proposing singles, in milliseconds.
    pub singles_ms: u64,
    /// Output-cache counters.
    pub cache: CacheStats,
}

/// A bundle that was materialised, deduplicated and scored by the audit.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct BundleRow {
    /// Position in the audit pool, also the file name `b{index}.json`.
    pub index: usize,
    /// Requested member UUIDs, in rank order.
    pub members: Vec<String>,
    /// Members absent from the core's requested list (taken by an earlier
    /// member's cascade, constant fold or splice).
    pub overlaps: Vec<String>,
    /// Members that are not hidden neurons of the parent.
    pub missing: Vec<String>,
    /// Neurons removed on top of the request: the cleanup cascade plus neurons
    /// the core folded into constant support (`PruneDetail::cascade_uuids`).
    pub cascade_neurons: Vec<String>,
    /// Hidden pass-throughs the core cleanup spliced out
    /// (`PruneDetail::spliced_neurons`): their sources now wire straight into
    /// their targets. Kept apart from `cascade_neurons`, as core keeps them.
    pub spliced_neurons: Vec<String>,
    /// Hidden neurons removed, from the materialised structure.
    pub hidden_removed: usize,
    /// Synapses removed, from the materialised structure.
    pub synapses_removed: usize,
    /// Growth units removed, from the materialised structure.
    pub growth_units_removed: f64,
    /// Whether `hidden_removed` reaches the requested removals.
    pub meets_budget: bool,
    /// SHA-256 of the materialised creature's canonical JSON.
    pub creature_digest: String,
    /// Full-scorer score (the truth for the audit).
    pub score: f64,
    /// Full-scorer error.
    pub error: f64,
    /// Full-scorer structural penalty, recorded separately from the score.
    pub complexity_penalty: f64,
    /// Full score minus the parent's score.
    pub truth_delta: f64,
    /// Diagnostic additive loss improvement, `-Σ(Li - L0)`.
    pub additive_loss: Option<f64>,
    /// Cached-output loss improvement, `L0 - predicted loss`.
    pub cached_output: Option<f64>,
}

/// A pool bundle whose creature digest matched an earlier bundle.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct DuplicateBundle {
    /// Requested member UUIDs.
    pub members: Vec<String>,
    /// Index of the scored bundle with the same creature.
    pub duplicate_of: usize,
    /// SHA-256 of the shared creature; equals `bundles[duplicate_of].creature_digest`.
    pub creature_digest: String,
}

/// A pool bundle the core engine refused to materialise.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Refusal {
    /// Requested member UUIDs.
    pub members: Vec<String>,
    /// Core's refusal message.
    pub reason: String,
}

/// One pick an arm sent to the full scorer.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Pick {
    /// Index into the row's `search.bundles`.
    pub bundle: usize,
    /// Full score from the arm's own search call.
    pub score: f64,
}

/// A unique bundle of the search pool. It is estimated but never scored as a
/// whole: only the picks an arm sends to the scorer get a full score.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SearchBundle {
    /// Position in the search pool, also the scorer file name `b{index}.json`
    /// in each search call.
    pub index: usize,
    /// Requested member UUIDs, in rank order.
    pub members: Vec<String>,
    /// Hidden neurons removed, from the materialised structure.
    pub hidden_removed: usize,
    /// Synapses removed, from the materialised structure.
    pub synapses_removed: usize,
    /// Growth units removed, from the materialised structure.
    pub growth_units_removed: f64,
    /// Whether `hidden_removed` reaches the requested removals.
    pub meets_budget: bool,
    /// SHA-256 of the materialised creature's canonical JSON.
    pub creature_digest: String,
    /// Diagnostic additive loss improvement, `-Σ(Li - L0)`, when estimated.
    pub additive_loss: Option<f64>,
    /// Cached-output loss improvement, `L0 - predicted loss`, when estimated.
    pub cached_output: Option<f64>,
}

/// The search pool of one row: drawn with its own seed, materialised, and
/// estimated, separately from the audit pool.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SearchPool {
    /// Distinct member sets the search pool drew.
    pub requested: usize,
    /// Search-pool bundles the core materialised, including duplicates.
    pub size: usize,
    /// Search-pool bundles whose creature repeated an earlier one.
    pub duplicates: Vec<DuplicateBundle>,
    /// Search-pool bundles the core refused.
    pub refused: Vec<Refusal>,
    /// Unique search-pool bundles, in the order they were materialised.
    pub bundles: Vec<SearchBundle>,
    /// Wall time materialising the search pool, in milliseconds.
    pub materialisation_ms: u64,
    /// Wall time computing the search pool's estimates, in milliseconds.
    pub estimate_ms: u64,
}

/// Milliseconds attributed to one arm's economics.
///
/// Single proposals and cache build or read are charged to arms that rank by
/// singles (`current`, `additiveLoss`, `cachedOutput`). `random` is charged
/// search-pool materialisation, its own ranking and its own scorer calls; it is
/// not charged single proposals, cache build or cache reads. The proposability
/// screen is shared and reported at parent level. The audit pool's costs are not
/// charged to any arm.
///
/// `total` is the sum of the components. Under a `searchWallClockMs` budget the
/// first five components are what the budget is charged with, and `scorer` is
/// what the arm's own calls took.
#[derive(Debug, Clone, Copy, Default, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ArmCost {
    /// Single cut proposals for the parent.
    pub single_proposals: u64,
    /// Cache entry construction for the parent.
    pub cache_build: u64,
    /// Cache reads for singles, plus, for `cachedOutput`, the search pool's
    /// cached-output estimate reads.
    pub cache_read_inference: u64,
    /// Materialising the search pool.
    pub pool_materialisation: u64,
    /// Ordering the search pool.
    pub ranking: u64,
    /// The arm's own full-scorer search calls.
    pub scorer: u64,
    /// Sum of the components above.
    pub total: u64,
}

/// Why an arm's search stopped scoring picks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub enum SearchStop {
    /// The arm scored its `fullEvaluations` picks.
    FullEvaluations,
    /// The search pool had fewer than `fullEvaluations` unique bundles, and all
    /// of them were scored.
    PoolExhausted,
    /// The wall-clock budget ran out before the arm's picks were all scored.
    WallClock,
}

/// One arm's result for one row.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ArmRow {
    /// Arm name.
    pub arm: Arm,
    /// Audit pool ordering the arm produced, over `bundles`.
    pub ranking: Vec<usize>,
    /// Ranking audit against the full-scorer truth of the audit pool.
    pub audit: AuditMetrics,
    /// Search pool ordering the arm produced, over `search.bundles`.
    pub search_ranking: Vec<usize>,
    /// The search-pool picks the arm scored, in ranking order, each with its
    /// search-call score.
    pub picks: Vec<Pick>,
    /// Why the search stopped: all `fullEvaluations` picks scored, the pool ran
    /// out first, or the wall-clock budget did. Always `Some` on a feasible row.
    pub stopped_by: Option<SearchStop>,
    /// Index into `search.bundles` of the best full score among the picks, or
    /// `None` when there were no picks.
    pub selected_bundle: Option<usize>,
    /// Selected bundle's full score.
    pub selected_score: Option<f64>,
    /// Selected bundle's full error.
    pub selected_error: Option<f64>,
    /// Selected bundle's full structural penalty.
    pub selected_complexity_penalty: Option<f64>,
    /// Selected score minus the parent's score.
    pub selected_delta: Option<f64>,
    /// Best selected score among this row's arms minus this arm's selected
    /// score. At least zero; paired within the shared search pool. `None` when
    /// this arm selected nothing. The audit's own regret is `audit.regret`.
    pub selected_regret: Option<f64>,
    /// Selected bundle's hidden neurons removed, from `search.bundles`.
    pub selected_hidden_removed: Option<usize>,
    /// Selected bundle's synapses removed, from `search.bundles`.
    pub selected_synapses_removed: Option<usize>,
    /// Held-out score of the selected creature, when validation data is set.
    pub validation_score: Option<f64>,
    /// Held-out error of the selected creature, when validation data is set.
    pub validation_error: Option<f64>,
    /// Full-scorer calls made for the arm's search: one batched call over its
    /// picks without a wall-clock budget, else one call per pick scored.
    pub full_scorer_calls: u64,
    /// Record evaluations: picks scored times corpus records, plus core records for singles.
    pub record_evaluations: u64,
    /// Cost breakdown in milliseconds.
    pub cost_ms: ArmCost,
    /// Measured prediction time per estimated search-pool bundle, in milliseconds,
    /// for predictor arms: the time spent forming their search-pool predictions,
    /// divided by the search bundles that had an estimate. `None` for
    /// non-predictor arms, and when no search bundle had an estimate.
    pub inference_latency_ms: Option<f64>,
}

/// Audit call accounting, reported apart from the economics.
#[derive(Debug, Clone, Copy, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AuditCost {
    /// Full-scorer calls (one per row).
    pub full_calls: u64,
    /// Scored bundles times corpus records.
    pub record_evaluations: u64,
    /// Wall time of the audit call, in milliseconds.
    pub ms: u64,
}

/// One (parent, budget, seed) run.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct BenchRow {
    /// Parent file name.
    pub parent: String,
    /// Parent index in the manifest.
    pub parent_index: usize,
    /// Budget fraction.
    pub budget: f64,
    /// Budget index in the manifest.
    pub budget_index: usize,
    /// Seed.
    pub seed: u64,
    /// Requested removals, `ceil(budget × hidden)`.
    pub requested_removals: usize,
    /// False when no ranking was possible; the row then carries no bundles.
    pub feasible: bool,
    /// Why the row is infeasible.
    pub reason: Option<String>,
    /// Proposable singles available for removal.
    pub eligible: usize,
    /// Distinct member sets the audit pool drew.
    pub pool_requested: usize,
    /// Audit-pool bundles the core materialised, including duplicates.
    pub pool_size: usize,
    /// Audit-pool bundles whose creature repeated an earlier one.
    pub duplicates: Vec<DuplicateBundle>,
    /// Audit-pool bundles the core refused.
    pub refused: Vec<Refusal>,
    /// Unique audit-pool bundles, scored by the audit.
    pub bundles: Vec<BundleRow>,
    /// Audit call accounting.
    pub audit: AuditCost,
    /// Wall time materialising the audit pool, in milliseconds.
    pub pool_materialisation_ms: u64,
    /// Wall time computing the audit pool's estimates, in milliseconds.
    pub estimate_ms: u64,
    /// Parent's held-out score, when validation data is set.
    pub validation_parent_score: Option<f64>,
    /// The separately drawn search pool the arms ranked and searched. `None`
    /// for infeasible rows, and for rows that stopped before the search.
    pub search: Option<SearchPool>,
    /// One result per arm, in report order. Empty for infeasible rows.
    pub arms: Vec<ArmRow>,
}

/// Distribution summary for one arm across the rows of one budget.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ArmAggregate {
    /// Arm name.
    pub arm: Arm,
    /// Spearman correlation of the arm's ordering with the truth.
    pub spearman: Option<Distribution>,
    /// Top-k recall of the arm's ordering.
    pub top_k_recall: Option<Distribution>,
    /// Best selected score in the row minus this arm's selected score.
    pub regret: Option<Distribution>,
    /// Selected full score minus the parent's score.
    pub selected_delta: Option<Distribution>,
    /// Arm total cost in milliseconds.
    pub total_ms: Option<Distribution>,
}

/// Aggregate over every feasible row of one budget.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct BudgetAggregate {
    /// Budget fraction.
    pub budget: f64,
    /// Rows with a ranking.
    pub feasible_rows: usize,
    /// Rows without a ranking.
    pub infeasible_rows: usize,
    /// Per-arm distributions, in report order.
    pub arms: Vec<ArmAggregate>,
    /// Selected full score: cachedOutput against current, over (parent, seed).
    pub cached_output_vs_current: Option<PairedComparison>,
    /// Selected full score: cachedOutput against random, over (parent, seed).
    pub cached_output_vs_random: Option<PairedComparison>,
}

/// A case where an arm selected a strictly better full score than cachedOutput.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct BeatenCase {
    /// Parent file name.
    pub parent: String,
    /// Budget fraction.
    pub budget: f64,
    /// Seed.
    pub seed: u64,
    /// The arm that did better (`current` or `random`).
    pub arm: Arm,
    /// That arm's selected full score.
    pub arm_score: f64,
    /// cachedOutput's selected full score.
    pub cached_output_score: f64,
}

/// The complete benchmark report written by the CLI.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct BenchReport {
    /// Layout version, see [`REPORT_FORMAT_VERSION`].
    pub format_version: u32,
    /// Predeclared protocol.
    pub manifest: ManifestEcho,
    /// True only for a representative parent count.
    pub representative: bool,
    /// Evidence grade of this run.
    pub evidence: String,
    /// Who may accept a creature.
    pub authority: String,
    /// Per-parent setup and cache counters.
    pub parents: Vec<ParentReport>,
    /// Every (parent, budget, seed) row.
    pub rows: Vec<BenchRow>,
    /// Per-budget aggregates.
    pub aggregate: Vec<BudgetAggregate>,
    /// Every case where current or random beat cachedOutput on selected score.
    pub cases_beating_cached_output: Vec<BeatenCase>,
    /// The bench process's own peak resident set in bytes (`VmHWM`), read at the
    /// end of the run. The external scorer child processes are not included.
    /// `None` where `/proc/self/status` is unavailable. `ramBoundBytes` stays
    /// the computed bound.
    pub measured_peak_rss_bytes: Option<u64>,
}

/// Evidence grade for a run over `parents` parents.
#[must_use]
pub fn evidence_for(parents: usize, representative: bool) -> String {
    if representative {
        "representative: apply the predeclared decision rule".to_string()
    } else {
        format!(
            "fixture-only: {parents} parent(s) < {REPRESENTATIVE_PARENTS} — not experimental evidence"
        )
    }
}

fn arm_of(row: &BenchRow, arm: Arm) -> Option<&ArmRow> {
    row.arms.iter().find(|a| a.arm == arm)
}

fn distribution_of(values: impl Iterator<Item = Option<f64>>) -> Option<Distribution> {
    let finite: Vec<f64> = values.flatten().filter(|v| v.is_finite()).collect();
    rank::distribution(&finite)
}

/// Summarise the feasible rows per budget and list every beating case.
///
/// # Errors
///
/// Propagates [`rank::paired`] refusals, which should not occur for finite scores.
pub fn aggregate(
    rows: &[BenchRow],
    budgets: &[f64],
) -> Result<(Vec<BudgetAggregate>, Vec<BeatenCase>), String> {
    let mut out = Vec::with_capacity(budgets.len());
    let mut beaten = Vec::new();
    for (budget_index, &budget) in budgets.iter().enumerate() {
        let mine: Vec<&BenchRow> = rows
            .iter()
            .filter(|row| row.budget_index == budget_index)
            .collect();
        let feasible: Vec<&BenchRow> = mine.iter().copied().filter(|r| r.feasible).collect();
        let arms = ARMS
            .iter()
            .map(|&arm| {
                let picked = |f: &dyn Fn(&ArmRow) -> Option<f64>| {
                    distribution_of(feasible.iter().map(|row| arm_of(row, arm).and_then(f)))
                };
                ArmAggregate {
                    arm,
                    spearman: picked(&|a| a.audit.spearman),
                    top_k_recall: picked(&|a| a.audit.top_k_recall),
                    regret: picked(&|a| a.selected_regret),
                    selected_delta: picked(&|a| a.selected_delta),
                    total_ms: picked(&|a| Some(a.cost_ms.total as f64)),
                }
            })
            .collect();

        let mut cached = Vec::new();
        let mut current = Vec::new();
        let mut random = Vec::new();
        for row in &feasible {
            let score = |arm: Arm| arm_of(row, arm).and_then(|a| a.selected_score);
            let (Some(c), Some(cur), Some(rnd)) = (
                score(Arm::CachedOutput),
                score(Arm::Current),
                score(Arm::Random),
            ) else {
                continue;
            };
            cached.push(c);
            current.push(cur);
            random.push(rnd);
            for (arm, value) in [(Arm::Current, cur), (Arm::Random, rnd)] {
                if value - c > TIE_TOLERANCE {
                    beaten.push(BeatenCase {
                        parent: row.parent.clone(),
                        budget,
                        seed: row.seed,
                        arm,
                        arm_score: value,
                        cached_output_score: c,
                    });
                }
            }
        }
        let pair = |other: &[f64]| -> Result<Option<PairedComparison>, String> {
            if cached.is_empty() {
                Ok(None)
            } else {
                rank::paired(&cached, other).map(Some)
            }
        };
        out.push(BudgetAggregate {
            budget,
            feasible_rows: feasible.len(),
            infeasible_rows: mine.len() - feasible.len(),
            arms,
            cached_output_vs_current: pair(&current)?,
            cached_output_vs_random: pair(&random)?,
        });
    }
    Ok((out, beaten))
}
