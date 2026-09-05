//! Seeded random sweep and sampled scorer screening (Issue #6).
//!
//! Hidden-neuron UUIDs are ordered once from a recorded seed and named
//! [`crate::ordering::Ordering`] and visited **without replacement**. Each visit
//! tries an exact IDENTITY collapse, then a correlated-neuron merge
//! ([`crate::merge`], Issue #109) when discovery proposed a partner, then a
//! mean-activation ablation, then a constant substitution
//! ([`crate::substitute`], Issue #103) for the structure the ablation fails
//! closed on. Attempts that produce nothing are skipped with a
//! [`crate::blocked::BlockedReason`] and the batch is refilled while unvisited
//! neurons remain.
//!
//! An ordering only changes *when* a neuron is tested (Issue #11). Every
//! candidate still passes `creature.validate()`, the sampled screen and full
//! authoritative scoring.
//!
//! The incumbent and every valid candidate in a batch are scored together in
//! one sampled scorer call. Sampled winners are returned for later
//! authoritative promotion; they never become `best.json` here.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::time::Instant;

use neat_core::{CreatureExport, SquashType, creature_to_json, parse_squash_name};
use serde::Serialize;

use crate::ablation::{GroupMember, ablate_group, ablate_mean};
use crate::blocked::BlockedReason;
use crate::collapse::{CollapseOptions, CollapseSkip, collapse_identity};
use crate::incumbent::sha256_hex;
use crate::merge::{MergeSkip, merge_correlated};
use crate::ordering::{Ordering, OrderingConfig, hidden_order};
use crate::scorer::{DirectoryScorer, ScorerMode};
use crate::signature::MergeIndex;
use crate::stats::ActivationStats;
use crate::substitute::substitute_constant;

/// Draw a seed from the clock and process id when the user omitted `--seed`.
pub fn draw_seed() -> u64 {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0);
    nanos ^ u64::from(std::process::id()).wrapping_shl(32) ^ 0xA5A5_A5A5_A5A5_A5A5
}

/// Kind of pruning proposal.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub enum CandidateKind {
    /// Exact IDENTITY collapse (#5).
    Identity,
    /// Mean-activation ablation (#4).
    Ablation,
    /// Mean-valued constant substitution (#103).
    ///
    /// The path for structure the ablation fails closed on: the neuron becomes
    /// a `constant` and its outgoing edge — role and weight — is preserved.
    Constant,
    /// Structural neighbourhood group ablation (#108).
    ///
    /// A whole chain or low-fan-out branch cut as one proposal, because some
    /// structure is only removable as a group. Screened and scored exactly like
    /// any other candidate.
    Group,
    /// Correlated-neuron merge (#109).
    ///
    /// Two hidden neurons behaved almost identically, so one of them goes and
    /// the other absorbs its downstream contribution.
    Merge,
}

/// One valid pruning candidate produced by the sweep.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SweepCandidate {
    /// Hidden neuron that was visited; the first member of a group (#108).
    pub uuid: String,
    /// Every hidden neuron this candidate was asked to cut (Issue #108).
    ///
    /// One entry — [`Self::uuid`] — for a single-neuron candidate, and the
    /// whole neighbourhood, upstream-first, for a group proposal. Always
    /// non-empty and always headed by [`Self::uuid`], so a reader that only
    /// knows about single cuts still names a real neuron.
    pub members: Vec<String>,
    /// Index in the seeded permutation.
    pub permutation_index: usize,
    /// How the candidate was built.
    pub kind: CandidateKind,
    /// Survivor that absorbed this neuron, for a [`CandidateKind::Merge`] (#109).
    ///
    /// Provenance, not decoration: a merge is the only candidate whose meaning
    /// depends on a *second* neuron, so the pair is what a learnings entry, a
    /// replay or a Rebase check-in has to carry to say what was tried.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub merged_with: Option<String>,
    /// Cohort file stem (`c000`, …).
    pub stem: String,
    /// Candidate creature.
    #[serde(skip)]
    pub creature: CreatureExport,
}

impl SweepCandidate {
    /// Whether this candidate cuts a whole neighbourhood at once (#108).
    ///
    /// The kind, not the member count: one test for "is this a group?" across
    /// the run, so screen coverage, the bundle pool, the candidate log and the
    /// cohort can never disagree about a candidate.
    pub fn is_group(&self) -> bool {
        self.kind == CandidateKind::Group
    }

    /// Hidden neurons this candidate cuts, upstream-first.
    ///
    /// Never empty: every construction site seeds [`Self::members`] with at
    /// least [`Self::uuid`].
    pub fn cuts(&self) -> &[String] {
        &self.members
    }
}

/// Why one visit produced no candidate, in words and as a code (Issue #103).
///
/// The message names the neuron and the structure, which is what an audit trail
/// needs; the code is what the tally, the screen record and every report count
/// by, which is what a work list needs. Deriving the second from the first by
/// parsing was how a non-finite mean used to hide among the aggregates.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Blocked {
    /// Stable reason code.
    pub reason: BlockedReason,
    /// Full message, naming the neuron and the structure that blocked it.
    pub detail: String,
}

impl Blocked {
    /// A blocked visit with `reason` and the message `detail`.
    pub fn new(reason: BlockedReason, detail: impl Into<String>) -> Self {
        Self {
            reason,
            detail: detail.into(),
        }
    }
}

impl std::fmt::Display for Blocked {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.detail)
    }
}

/// [`SweepSkip::reason`] of a visit a standing full-corpus verdict suppressed.
///
/// Named rather than spelled twice: the run classifies a skip by this exact
/// reason when filing screen coverage (Issue #93).
pub const KNOWN_FAILURE_REASON: &str = "known-failure";

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
    /// Reason code, or `None` for a standing full-corpus verdict (Issue #103).
    ///
    /// A known failure is not blocked: the cut was proposed, scored and judged,
    /// so it carries no blocked reason and is not counted in the breakdown.
    pub blocked: Option<BlockedReason>,
}

/// Seeded without-replacement walk over hidden neurons.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Sweep {
    /// Seed that produced [`Self::order`].
    pub seed: u64,
    /// Named ordering strategy that produced [`Self::order`].
    pub ordering: Ordering,
    /// SHA-256 of `seed`, the ordering name and the ordered UUID list.
    pub permutation_identity: String,
    /// Hidden UUIDs in visitation order.
    pub order: Vec<String>,
    /// Next index to visit.
    pub next: usize,
    /// Whether coverage-driven unchecked-first selection reordered the tail (#38).
    ///
    /// Recorded so a run is reconstructable: [`Self::permutation_identity`]
    /// covers the pre-reorder order only.
    pub unchecked_first: bool,
    /// Neurons old-corpus verdicts moved to the front of the tail (#88).
    ///
    /// Recorded for the same reason as [`Self::unchecked_first`]: it reorders
    /// the sweep after the identity above is fixed. `0` when the priority is
    /// off, there is no cache, or nothing qualified.
    pub old_corpus_first: usize,
}

impl Sweep {
    /// Shuffle the incumbent's hidden UUIDs with `seed` (the random control).
    pub fn new(creature: &CreatureExport, seed: u64) -> Self {
        Self::with_ordering(
            creature,
            &ActivationStats::empty(),
            seed,
            OrderingConfig::default(),
        )
    }

    /// Order the incumbent's hidden UUIDs with `seed` and `cfg` (Issue #11).
    ///
    /// The order is always a permutation of every hidden UUID: an ordering
    /// reprioritises the sweep, it never shrinks it.
    pub fn with_ordering(
        creature: &CreatureExport,
        stats: &ActivationStats,
        seed: u64,
        cfg: OrderingConfig<'_>,
    ) -> Self {
        let order = hidden_order(creature, stats, cfg, seed);
        // The strategy that actually ranked, so a permutation identity always
        // names the ranking behind it (#107).
        let strategy = cfg.effective_strategy();
        let mut ident = format!(
            "seed={seed}\nordering={}\nrandomQuota={}\n",
            strategy.name(),
            cfg.random_quota
        );
        for uuid in &order {
            ident.push_str(uuid);
            ident.push('\n');
        }
        Self {
            seed,
            ordering: strategy,
            permutation_identity: sha256_hex(ident.as_bytes()),
            order,
            next: 0,
            unchecked_first: false,
            old_corpus_first: 0,
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
    /// Prefer-list order is preserved. Unknown or already-visited UUIDs are
    /// ignored. Returns how many were actually moved, so a caller reporting the
    /// reordering counts the neurons it moved rather than the ones it asked for.
    pub fn prefer(&mut self, uuids: &[String]) -> usize {
        if self.next >= self.order.len() || uuids.is_empty() {
            return 0;
        }
        let mut remaining: Vec<String> = self.order.split_off(self.next);
        let mut front = Vec::new();
        for u in uuids {
            if let Some(i) = remaining.iter().position(|x| x == u) {
                front.push(remaining.remove(i));
            }
        }
        let moved = front.len();
        self.order.extend(front);
        self.order.extend(remaining);
        moved
    }

    /// Partition the unvisited tail into unchecked-first, then stalest-first (#38).
    ///
    /// The tail becomes two blocks, and every UUID stays in exactly one of them:
    ///
    /// - **A** — UUIDs with no screen record, in ordering-strategy order.
    /// - **B** — UUIDs in `screened`, ordered by `oldest_first`
    ///   ([`crate::learnings::oldest_screened_first`]); any not named there keep
    ///   their ordering-strategy order behind the ones that are.
    ///
    /// This reprioritises the sweep, it never shrinks it: the result is a
    /// permutation of the same tail, so a run that exhausts block A rolls
    /// straight into re-screening the stalest neurons instead of stopping.
    /// Already-visited entries and [`Self::permutation_identity`] are untouched.
    pub fn prefer_unchecked(&mut self, screened: &HashSet<String>, oldest_first: &[String]) {
        self.unchecked_first = true;
        if self.next >= self.order.len() {
            return;
        }
        let tail = self.order.split_off(self.next);
        let (mut unchecked, mut deferred): (Vec<String>, Vec<String>) =
            tail.into_iter().partition(|uuid| !screened.contains(uuid));
        let staleness: HashMap<&str, usize> = oldest_first
            .iter()
            .enumerate()
            .map(|(i, uuid)| (uuid.as_str(), i))
            .collect();
        // Stable, so an unranked screened uuid keeps its strategy order last.
        deferred.sort_by_key(|uuid| staleness.get(uuid.as_str()).copied().unwrap_or(usize::MAX));
        self.order.append(&mut unchecked);
        self.order.append(&mut deferred);
    }

    /// Build up to `size` valid candidates, refilling past skips.
    ///
    /// The merge-free control: correlated-neuron merging (#109) proposes
    /// nothing here. Use [`Self::fill_batch_avoiding`] with a populated
    /// [`MergeIndex`] to include it.
    pub fn fill_batch(
        &mut self,
        incumbent: &CreatureExport,
        stats: &ActivationStats,
        size: usize,
    ) -> (Vec<SweepCandidate>, Vec<SweepSkip>) {
        self.fill_batch_avoiding(incumbent, stats, MergeIndex::empty(), size, &HashSet::new())
    }

    /// [`Self::fill_batch`] with merge proposals, skipping UUIDs in `avoid`.
    ///
    /// Tags confer no exemption (#63): every hidden neuron is a candidate,
    /// tagged or not, and the only skips are known failures plus the reasons
    /// proposing a candidate reports for itself.
    pub fn fill_batch_avoiding(
        &mut self,
        incumbent: &CreatureExport,
        stats: &ActivationStats,
        merges: &MergeIndex,
        size: usize,
        avoid: &HashSet<String>,
    ) -> (Vec<SweepCandidate>, Vec<SweepSkip>) {
        let mut candidates = Vec::new();
        let mut skips = Vec::new();
        while candidates.len() < size && !self.exhausted() {
            let permutation_index = self.next;
            let uuid = self.order[permutation_index].clone();
            self.next += 1;
            if avoid.contains(&uuid) {
                skips.push(SweepSkip {
                    uuid,
                    permutation_index,
                    reason: KNOWN_FAILURE_REASON.into(),
                    blocked: None,
                });
                continue;
            }
            match propose(incumbent, stats, merges, &uuid) {
                Ok(proposed) => {
                    let stem = format!("c{:03}", candidates.len());
                    candidates.push(SweepCandidate {
                        members: vec![uuid.clone()],
                        uuid,
                        permutation_index,
                        kind: proposed.kind,
                        merged_with: proposed.merged_with,
                        stem,
                        creature: proposed.creature,
                    });
                }
                Err(blocked) => skips.push(SweepSkip {
                    uuid,
                    permutation_index,
                    reason: blocked.detail,
                    blocked: Some(blocked.reason),
                }),
            }
        }
        (candidates, skips)
    }
}

/// Whether `uuid` carries an `IDENTITY` squash — an exact-fold opportunity.
///
/// Shared with the orderings (#107) so one place decides what counts as an
/// identity neuron, including that an unparsable squash name does not.
pub(crate) fn is_identity(creature: &CreatureExport, uuid: &str) -> bool {
    creature.neurons.iter().any(|n| {
        n.uuid == uuid
            && parse_squash_name(n.squash.as_deref().unwrap_or("IDENTITY"))
                .is_ok_and(|s| s == SquashType::Identity)
    })
}

/// One candidate the sweep built, and what built it.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Proposed {
    /// How the candidate was built.
    pub kind: CandidateKind,
    /// Survivor for a [`CandidateKind::Merge`]; `None` for every other kind.
    pub merged_with: Option<String>,
    /// Validated candidate creature.
    pub creature: CreatureExport,
}

impl Proposed {
    fn of(kind: CandidateKind, creature: CreatureExport) -> Self {
        Self {
            kind,
            merged_with: None,
            creature,
        }
    }
}

/// The first merge proposal for `uuid` that builds a valid candidate (#109).
///
/// Proposals arrive strongest-correlation first, and most of a forest-heavy
/// creature's pairs are structurally unmergeable, so the strongest partner is
/// tried first and the rest are fallbacks rather than a search.
fn propose_merge(
    incumbent: &CreatureExport,
    merges: &MergeIndex,
    uuid: &str,
) -> Result<Proposed, Option<MergeSkip>> {
    let mut first: Option<MergeSkip> = None;
    for proposal in merges.for_removed(uuid) {
        match merge_correlated(incumbent, &proposal.survivor_uuid, uuid, proposal.relation) {
            Ok(merge) => {
                return Ok(Proposed {
                    kind: CandidateKind::Merge,
                    merged_with: Some(proposal.survivor_uuid.clone()),
                    creature: merge.creature,
                });
            }
            // Kept, not dropped: a run where every merge proposal fails on the
            // same structure must be able to say so rather than reporting only
            // whatever the fallback path went on to complain about.
            Err(skip) => first.get_or_insert(skip),
        };
    }
    Err(first)
}

/// Append the merge skip to `blocked`, when discovery proposed one at all.
///
/// The fallback path's own reason still classifies the visit — it is the one
/// that actually stopped the razor — but the merge attempt is named beside it
/// so a merge-enabled run can see why its proposals went nowhere.
fn with_merge_detail(mut blocked: Blocked, merge: Option<MergeSkip>) -> Blocked {
    if let Some(skip) = merge {
        blocked.detail = format!("{}; merge: {skip}", blocked.detail);
    }
    blocked
}

pub(crate) fn propose(
    incumbent: &CreatureExport,
    stats: &ActivationStats,
    merges: &MergeIndex,
    uuid: &str,
) -> Result<Proposed, Blocked> {
    if is_identity(incumbent, uuid) {
        match collapse_identity(incumbent, uuid, CollapseOptions::default()) {
            Ok(c) => return Ok(Proposed::of(CandidateKind::Identity, c.creature)),
            Err(e) => {
                // Cost-increasing IDENTITY still has an approximate ablation path.
                if stats.by_uuid(uuid).is_none() {
                    // A near-duplicate partner is a path that needs no mean of
                    // this neuron at all, so it is tried before the statistic
                    // this branch is about to report missing.
                    let merge = match propose_merge(incumbent, merges, uuid) {
                        Ok(merged) => return Ok(merged),
                        Err(skip) => skip,
                    };
                    // Without a measured mean there is no fallback to take, so
                    // a collapse the razor could otherwise have retried is
                    // blocked on the missing statistic rather than on itself.
                    let reason = match e {
                        CollapseSkip::CostIncrease { .. } => BlockedReason::MissingActivation,
                        ref skip => skip.blocked_reason(),
                    };
                    return Err(with_merge_detail(
                        Blocked::new(reason, e.to_string()),
                        merge,
                    ));
                }
            }
        }
    }
    // A merge removes a whole neuron and compensates through a partner that is
    // already carrying the same behaviour, so it is tried ahead of folding this
    // neuron's activation down to its own mean (#109).
    let merge = match propose_merge(incumbent, merges, uuid) {
        Ok(merged) => return Ok(merged),
        Err(skip) => skip,
    };
    let Some(mean) = stats.by_uuid(uuid).map(|s| s.mean) else {
        return Err(with_merge_detail(
            Blocked::new(
                BlockedReason::MissingActivation,
                format!("no activation stats for `{uuid}`"),
            ),
            merge,
        ));
    };
    let ablation = match ablate_mean(incumbent, uuid, mean, stats.by_uuid(uuid)) {
        Ok(a) => return Ok(Proposed::of(CandidateKind::Ablation, a.creature)),
        Err(e) => e,
    };
    // The bias fold cannot express an aggregate target or a role-carrying edge,
    // and that is most of a forest-heavy creature. Keeping the edge and
    // constant-folding the source can (Issue #103) — and when it cannot either,
    // the reason reported is the one that actually stopped the razor.
    if !ablation.substitution_may_help() {
        return Err(with_merge_detail(
            Blocked::new(ablation.blocked_reason(), ablation.to_string()),
            merge,
        ));
    }
    match substitute_constant(incumbent, uuid, mean) {
        Ok(s) => Ok(Proposed::of(CandidateKind::Constant, s.creature)),
        Err(substitution) => Err(with_merge_detail(
            Blocked::new(
                substitution.blocked_reason(),
                format!("{ablation}; constant substitution: {substitution}"),
            ),
            merge,
        )),
    }
}

/// Build the group candidate that cuts every neuron of `members` (Issue #108).
///
/// The whole [`crate::ablation::GroupAblation`] comes back, not just the
/// creature: it names every neuron the transform removed and says which were
/// the requested group cuts and which the cleanup cascade stranded, which is
/// what a run has to record about a group it proposes.
///
/// The same substitution [`propose`] applies to one neuron, applied to the
/// whole neighbourhood on one clone before the exact cleanup runs. A member
/// without a measured mean blocks the group rather than being guessed at, and
/// every other refusal is the one [`crate::ablation::ablate_group`] reports.
///
/// Building a group is not accepting one: the candidate goes through the same
/// sampled screen and the same full-corpus scoring as every other proposal.
pub(crate) fn propose_group(
    incumbent: &CreatureExport,
    stats: &ActivationStats,
    members: &[String],
) -> Result<crate::ablation::GroupAblation, Blocked> {
    let mut cuts = Vec::with_capacity(members.len());
    for uuid in members {
        let mean = stats.by_uuid(uuid).map(|s| s.mean).ok_or_else(|| {
            Blocked::new(
                BlockedReason::MissingActivation,
                format!("group: no activation stats for `{uuid}`"),
            )
        })?;
        cuts.push(GroupMember {
            uuid: uuid.clone(),
            mean,
        });
    }
    ablate_group(incumbent, &cuts)
        .map_err(|e| Blocked::new(e.blocked_reason(), format!("group: {e}")))
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

/// One candidate the sampled screen did not promote.
///
/// Carries the [`CandidateKind`] so screen-coverage records match the kind the
/// verdict cache stores (Issue #36); the losing candidate creature itself is
/// dropped because nothing downstream scores it again.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ScreenedLoser {
    /// Hidden neuron that was screened.
    pub uuid: String,
    /// How the candidate was built.
    pub kind: CandidateKind,
    /// Survivor that absorbed it, for a [`CandidateKind::Merge`] (#109).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub merged_with: Option<String>,
    /// Sampled Δ against the incumbent scored in the same call.
    pub delta: f64,
    /// Ladder stage that ended it; `0` for the fixed-rate control (#104).
    pub stage: usize,
    /// Why it ended (#104).
    pub reason: ScreenRejection,
}

/// Why a screened candidate went no further (Issue #104).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub enum ScreenRejection {
    /// The sampled Δ was at or below `-reject_margin` — clearly worse.
    ClearlyWorse,
    /// The promotion stage's sampled Δ did not clear `--screen-threshold`.
    BelowThreshold,
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
    pub losers: Vec<ScreenedLoser>,
    /// Records the scorer read, summed over the cohort including the incumbent.
    ///
    /// The scorer reports records per creature; the cohort cost is that figure
    /// across every creature it scored, which is what a ladder stage's price
    /// has to be measured in (#104).
    pub records_scored: u64,
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
    // Summed from what each creature's result actually reports, never
    // extrapolated from the incumbent's: the economics of a ladder rung rest on
    // this figure, and an assumed record count would be a guess wearing the
    // costume of a measurement (#104).
    let mut records_scored = baseline.record_count;
    for c in candidates {
        let result = results.get(&c.stem).ok_or_else(|| {
            format!(
                "screen: scorer returned no entry for candidate stem `{}`",
                c.stem
            )
        })?;
        records_scored = records_scored.saturating_add(result.record_count);
        let delta = result.score - baseline.score;
        if delta > cfg.threshold {
            winners.push(SampledWinner {
                candidate: c,
                score: result.score,
                baseline_score: baseline.score,
                delta,
            });
        } else {
            losers.push(ScreenedLoser {
                uuid: c.uuid,
                kind: c.kind,
                merged_with: c.merged_with,
                delta,
                stage: 0,
                reason: ScreenRejection::BelowThreshold,
            });
        }
    }
    Ok(ScreenOutcome {
        sample_rate: cfg.sample_rate,
        sample_phase: cfg.sample_phase,
        baseline_score: baseline.score,
        winners,
        losers,
        records_scored,
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
    use crate::stats::{ActivationStats, NeuronStats, STATS_FORMAT_VERSION, SampleSpec};

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
            corpus_record_count: 1,
            sample: SampleSpec::full(),
            stopped_early: false,
            scan_ms: 0,
            from_cache: false,
            probes: Vec::new(),
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
        // Losers carry their kind so a screen record matches the verdict cache.
        let mut lost: Vec<&str> = outcome.losers.iter().map(|l| l.uuid.as_str()).collect();
        lost.sort_unstable();
        assert_eq!(lost, vec!["h_a", "h_b"]);
        assert!(
            outcome
                .losers
                .iter()
                .all(|l| l.kind == CandidateKind::Identity)
        );
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

    /// Stats that make `h_b` the flattest and quietest hidden neuron.
    fn skewed_stats(creature: &CreatureExport) -> ActivationStats {
        let mut stats = stats_for(creature);
        for n in &mut stats.neurons {
            let loud = n.uuid == "h_a";
            n.variance = if loud { 9.0 } else { 0.001 };
            n.mean_abs = if loud { 3.0 } else { 0.01 };
            n.max = if loud { 5.0 } else { 0.05 };
            n.min = -n.max;
        }
        stats
    }

    #[test]
    fn a_named_ordering_reprioritises_the_sweep_without_changing_what_is_tested() {
        let creature = two_hidden();
        let stats = skewed_stats(&creature);
        let seed = 5;
        let control = Sweep::with_ordering(&creature, &stats, seed, OrderingConfig::default());
        let ranked = Sweep::with_ordering(
            &creature,
            &stats,
            seed,
            OrderingConfig::new(Ordering::LowVariance),
        );
        assert_eq!(ranked.ordering, Ordering::LowVariance);
        assert_eq!(
            ranked.order[0], "h_b",
            "flattest neuron must be tested first"
        );
        assert_ne!(
            control.permutation_identity, ranked.permutation_identity,
            "each ordering must have its own reproducible identity"
        );

        // Same neurons, same gates — only the visitation order moved.
        let mut control_set: Vec<String> = control.order.clone();
        let mut ranked_set: Vec<String> = ranked.order.clone();
        control_set.sort();
        ranked_set.sort();
        assert_eq!(control_set, ranked_set);

        let mut sweep = ranked;
        let (batch, skips) = sweep.fill_batch(&creature, &stats, 8);
        assert!(skips.is_empty(), "{skips:?}");
        assert_eq!(batch.len(), 2);
        assert_eq!(batch[0].uuid, "h_b");
        for c in &batch {
            validate_creature(&c.creature).expect("ordering must not bypass creature.validate()");
        }
    }

    #[test]
    fn a_named_ordering_is_reproducible_for_a_fixed_seed() {
        let creature = two_hidden();
        let stats = skewed_stats(&creature);
        let cfg = OrderingConfig {
            strategy: Ordering::LowMeanAbs,
            random_quota: 0.25,
            priority: None,
        };
        let a = Sweep::with_ordering(&creature, &stats, 17, cfg);
        let b = Sweep::with_ordering(&creature, &stats, 17, cfg);
        assert_eq!(a.order, b.order);
        assert_eq!(a.permutation_identity, b.permutation_identity);
    }

    /// Six hidden neurons with deliberately different signals, so every
    /// [`Ordering`] strategy produces a distinct tail to partition.
    fn six_hidden() -> CreatureExport {
        let mut neurons: Vec<_> = (0..6)
            .map(|i| {
                let squash = if i % 2 == 0 { "IDENTITY" } else { "TANH" };
                neuron("hidden", &format!("h{i}"), 0.0, Some(squash))
            })
            .collect();
        neurons.push(neuron("output", "output-0", 0.0, Some("IDENTITY")));
        neurons.push(neuron("output", "output-1", 0.0, Some("IDENTITY")));
        let mut synapses = Vec::new();
        for i in 0..6 {
            let uuid = format!("h{i}");
            synapses.push(synapse("input-0", &uuid, 1.0));
            synapses.push(synapse(&uuid, "output-0", 0.5 + i as f64));
            if i % 3 == 0 {
                synapses.push(synapse(&uuid, "output-1", 1.0));
            }
        }
        creature(1, 2, neurons, synapses)
    }

    /// Statistics that rank the six hidden neurons differently on every signal.
    fn varied_stats(creature: &CreatureExport) -> ActivationStats {
        let mut stats = stats_for(creature);
        for (i, n) in stats.neurons.iter_mut().enumerate() {
            let k = (i + 1) as f64;
            n.variance = k * 0.5;
            n.std_dev = n.variance.sqrt();
            n.mean_abs = k * 0.1;
            n.max = k;
            n.min = -k;
        }
        stats
    }

    fn uuid_set(uuids: &[&str]) -> HashSet<String> {
        uuids.iter().map(|u| (*u).to_string()).collect()
    }

    fn uuid_list(uuids: &[&str]) -> Vec<String> {
        uuids.iter().map(|u| (*u).to_string()).collect()
    }

    fn every_ordering() -> Vec<OrderingConfig<'static>> {
        let mut cfgs = Vec::new();
        for strategy in Ordering::ALL {
            for random_quota in [0.0, 0.25, 0.5, 0.9] {
                cfgs.push(OrderingConfig {
                    strategy: *strategy,
                    random_quota,
                    priority: None,
                });
            }
        }
        cfgs
    }

    #[test]
    fn prefer_unchecked_is_a_permutation_under_every_ordering_and_quota() {
        let creature = six_hidden();
        let stats = varied_stats(&creature);
        let screened = uuid_set(&["h1", "h3", "h5"]);
        let oldest = uuid_list(&["h5", "h1", "h3"]);
        let expected = {
            let mut u: Vec<String> = (0..6).map(|i| format!("h{i}")).collect();
            u.sort();
            u
        };
        for cfg in every_ordering() {
            let mut sweep = Sweep::with_ordering(&creature, &stats, 3, cfg);
            assert!(!sweep.unchecked_first);
            sweep.prefer_unchecked(&screened, &oldest);
            assert!(sweep.unchecked_first);
            let mut got = sweep.order.clone();
            got.sort();
            assert_eq!(
                got, expected,
                "{} quota={} lost or duplicated a neuron",
                cfg.strategy, cfg.random_quota
            );
        }
    }

    #[test]
    fn unchecked_keep_strategy_order_and_screened_recycle_oldest_first() {
        let creature = six_hidden();
        let stats = varied_stats(&creature);
        let screened = uuid_set(&["h1", "h3", "h5"]);
        let oldest = uuid_list(&["h5", "h1", "h3"]);
        for cfg in every_ordering() {
            let mut sweep = Sweep::with_ordering(&creature, &stats, 8, cfg);
            let before = sweep.order.clone();
            sweep.prefer_unchecked(&screened, &oldest);
            let block_a: Vec<String> = before
                .iter()
                .filter(|u| !screened.contains(*u))
                .cloned()
                .collect();
            assert_eq!(
                &sweep.order[..block_a.len()],
                block_a.as_slice(),
                "block A must keep {} order",
                cfg.strategy
            );
            assert_eq!(
                &sweep.order[block_a.len()..],
                oldest.as_slice(),
                "block B must be oldest-screened first"
            );
        }
    }

    #[test]
    fn an_empty_screen_set_leaves_the_order_unchanged() {
        let creature = six_hidden();
        let stats = varied_stats(&creature);
        for cfg in every_ordering() {
            let mut sweep = Sweep::with_ordering(&creature, &stats, 21, cfg);
            let before = sweep.order.clone();
            sweep.prefer_unchecked(&HashSet::new(), &[]);
            assert_eq!(
                sweep.order, before,
                "a cold cache must not change {} quota={}",
                cfg.strategy, cfg.random_quota
            );
        }
    }

    #[test]
    fn the_same_inputs_reproduce_the_same_coverage_driven_order() {
        let creature = six_hidden();
        let stats = varied_stats(&creature);
        let screened = uuid_set(&["h0", "h4"]);
        let oldest = uuid_list(&["h4", "h0"]);
        let cfg = OrderingConfig {
            strategy: Ordering::LowMeanAbs,
            random_quota: 0.3,
            priority: None,
        };
        let mut a = Sweep::with_ordering(&creature, &stats, 17, cfg);
        let mut b = Sweep::with_ordering(&creature, &stats, 17, cfg);
        a.prefer_unchecked(&screened, &oldest);
        b.prefer_unchecked(&screened, &oldest);
        assert_eq!(a.order, b.order);
        assert_eq!(a.permutation_identity, b.permutation_identity);
    }

    #[test]
    fn the_permutation_identity_predates_the_coverage_reorder() {
        let creature = six_hidden();
        let stats = varied_stats(&creature);
        let cfg = OrderingConfig::new(Ordering::LowVariance);
        let mut sweep = Sweep::with_ordering(&creature, &stats, 4, cfg);
        let identity = sweep.permutation_identity.clone();
        sweep.prefer_unchecked(&uuid_set(&["h0", "h1"]), &uuid_list(&["h1", "h0"]));
        assert_ne!(
            sweep.order[0], "h0",
            "the reorder must actually move the tail"
        );
        assert_eq!(
            sweep.permutation_identity, identity,
            "#11 strategy comparisons hash the pre-reorder order"
        );
        assert_eq!(
            identity,
            Sweep::with_ordering(&creature, &stats, 4, cfg).permutation_identity
        );
    }

    #[test]
    fn a_fully_screened_creature_still_visits_every_neuron() {
        let creature = six_hidden();
        let stats = varied_stats(&creature);
        let all: Vec<&str> = ["h0", "h1", "h2", "h3", "h4", "h5"].into();
        let mut sweep = Sweep::new(&creature, 5);
        sweep.prefer_unchecked(&uuid_set(&all), &uuid_list(&all));
        assert_eq!(
            sweep.order,
            uuid_list(&all),
            "block A is empty, so the stalest-first recycle order is the sweep"
        );
        let mut visited = 0;
        while !sweep.exhausted() {
            let (batch, skips) = sweep.fill_batch(&creature, &stats, 2);
            assert!(
                !batch.is_empty() || !skips.is_empty(),
                "recycling must never starve a run"
            );
            visited += batch.len() + skips.len();
        }
        assert_eq!(visited, 6);
    }

    #[test]
    fn already_visited_uuids_are_left_alone() {
        let creature = six_hidden();
        let stats = varied_stats(&creature);
        let mut sweep = Sweep::new(&creature, 12);
        let (batch, skips) = sweep.fill_batch(&creature, &stats, 2);
        assert_eq!(batch.len() + skips.len(), sweep.next);
        let visited: Vec<String> = sweep.order[..sweep.next].to_vec();
        let screened = uuid_set(&visited.iter().map(String::as_str).collect::<Vec<_>>());
        sweep.prefer_unchecked(&screened, &visited);
        assert_eq!(
            sweep.order[..sweep.next],
            visited[..],
            "the visited prefix must not move"
        );
        assert!(
            sweep.order[sweep.next..]
                .iter()
                .all(|u| !screened.contains(u)),
            "already-visited UUIDs must not be re-queued into the tail"
        );
    }

    #[test]
    fn fill_batch_skips_known_failures() {
        let creature = two_hidden();
        let stats = stats_for(&creature);
        let mut sweep = Sweep::new(&creature, 9);
        let blocked = sweep.order[0].clone();
        let avoid = HashSet::from([blocked.clone()]);
        let (batch, skips) =
            sweep.fill_batch_avoiding(&creature, &stats, MergeIndex::empty(), 8, &avoid);
        assert!(batch.iter().all(|c| c.uuid != blocked));
        assert!(
            skips
                .iter()
                .any(|s| s.uuid == blocked && s.reason == "known-failure"),
            "{skips:?}"
        );
    }

    /// Two hidden TANH neurons computing exactly the same function, plus one
    /// that does not. Only the twins are a merge candidate.
    fn twins_and_a_stranger() -> CreatureExport {
        creature(
            1,
            1,
            vec![
                neuron("hidden", "h_a", 0.25, Some("TANH")),
                neuron("hidden", "h_b", 0.25, Some("TANH")),
                neuron("hidden", "h_odd", -1.0, Some("TANH")),
                neuron("output", "output-0", 0.0, Some("IDENTITY")),
            ],
            vec![
                synapse("input-0", "h_a", 1.5),
                synapse("input-0", "h_b", 1.5),
                synapse("input-0", "h_odd", -0.2),
                synapse("h_a", "output-0", 2.0),
                synapse("h_b", "output-0", -0.75),
                synapse("h_odd", "output-0", 1.0),
            ],
        )
    }

    /// Statistics whose probe vectors make `h_a` and `h_b` exact duplicates.
    fn twin_stats(creature: &CreatureExport) -> ActivationStats {
        let twin: Vec<f32> = (0..32).map(|i| ((i % 11) as f32) - 5.0).collect();
        let odd: Vec<f32> = (0..32).map(|i| ((i % 7) as f32) - 3.0).collect();
        let mut stats = stats_for(creature);
        stats.probes = vec![
            crate::stats::NeuronProbes {
                uuid: "h_a".into(),
                values: twin.clone(),
            },
            crate::stats::NeuronProbes {
                uuid: "h_b".into(),
                values: twin,
            },
            crate::stats::NeuronProbes {
                uuid: "h_odd".into(),
                values: odd,
            },
        ];
        stats
    }

    /// Issue #109: a duplicated neuron the activation rankings cannot tell
    /// apart is proposed as a merge, and the candidate names its survivor.
    #[test]
    fn a_correlated_pair_is_proposed_as_a_merge_naming_its_survivor() {
        let creature = twins_and_a_stranger();
        validate_creature(&creature).unwrap();
        let stats = twin_stats(&creature);
        let merges =
            crate::signature::discover(&stats, crate::signature::DiscoveryConfig::default());
        assert!(!merges.is_empty(), "{:?}", merges.report());

        let mut sweep = Sweep::new(&creature, 3);
        let (batch, skips) =
            sweep.fill_batch_avoiding(&creature, &stats, &merges, 8, &HashSet::new());
        assert!(skips.is_empty(), "{skips:?}");
        let merged: Vec<&SweepCandidate> = batch
            .iter()
            .filter(|c| c.kind == CandidateKind::Merge)
            .collect();
        assert!(!merged.is_empty(), "no merge proposed: {batch:?}");
        for c in &merged {
            let survivor = c
                .merged_with
                .as_deref()
                .expect("a merge names its survivor");
            assert_ne!(survivor, c.uuid);
            assert!(["h_a", "h_b"].contains(&survivor), "{survivor}");
            assert!(c.creature.neurons.iter().all(|n| n.uuid != c.uuid));
            validate_creature(&c.creature).expect("a merge must not bypass validate()");
        }
        // The uncorrelated neuron takes the ordinary path and carries no pair.
        let odd = batch.iter().find(|c| c.uuid == "h_odd").expect("h_odd");
        assert_ne!(odd.kind, CandidateKind::Merge);
        assert!(odd.merged_with.is_none());
    }

    /// The same sweep with no merge index is the control it always was.
    #[test]
    fn without_a_merge_index_no_candidate_is_a_merge() {
        let creature = twins_and_a_stranger();
        let stats = twin_stats(&creature);
        let mut sweep = Sweep::new(&creature, 3);
        let (batch, _) = sweep.fill_batch(&creature, &stats, 8);
        assert!(batch.iter().all(|c| c.kind != CandidateKind::Merge));
        assert!(batch.iter().all(|c| c.merged_with.is_none()));
    }

    /// `creature` serialised with a GRQ-style tag on every hidden neuron.
    fn tagged_json(creature: &CreatureExport) -> String {
        let mut value: serde_json::Value =
            serde_json::from_str(&creature_to_json(creature).unwrap()).unwrap();
        for n in value["neurons"].as_array_mut().unwrap() {
            if n["type"] == "hidden" {
                n.as_object_mut().unwrap().insert(
                    "tags".into(),
                    serde_json::json!([{"name": "discovered", "value": "ReLU6"}]),
                );
            }
        }
        serde_json::to_string(&value).unwrap()
    }

    /// The inverse of the removed `fill_batch_skips_tagged_neurons_as_tagged`:
    /// a tag records where a neuron came from, it does not exempt it (#63).
    #[test]
    fn every_hidden_neuron_is_a_candidate_even_when_all_are_tagged() {
        let json = tagged_json(&two_hidden());
        let meta = crate::tags::CreatureMeta::from_json(&json);
        assert_eq!(
            meta.neuron_tags.len(),
            2,
            "fixture must tag every hidden neuron: {:?}",
            meta.neuron_tags
        );
        let creature = neat_core::parse_creature_json(&json).unwrap();
        let stats = stats_for(&creature);
        let mut sweep = Sweep::new(&creature, 9);
        let (batch, skips) =
            sweep.fill_batch_avoiding(&creature, &stats, MergeIndex::empty(), 8, &HashSet::new());
        assert!(skips.is_empty(), "a tag must not skip a neuron: {skips:?}");
        let mut proposed: Vec<&str> = batch.iter().map(|c| c.uuid.as_str()).collect();
        proposed.sort_unstable();
        assert_eq!(proposed, vec!["h_a", "h_b"]);
    }

    #[test]
    fn output_neurons_are_never_proposed() {
        let creature = six_hidden();
        let stats = varied_stats(&creature);
        let outputs: Vec<&str> = creature
            .neurons
            .iter()
            .filter(|n| n.neuron_type == "output")
            .map(|n| n.uuid.as_str())
            .collect();
        assert_eq!(outputs, vec!["output-0", "output-1"]);
        let mut sweep = Sweep::new(&creature, 4);
        let mut visited = Vec::new();
        while !sweep.exhausted() {
            let (batch, skips) = sweep.fill_batch(&creature, &stats, 2);
            visited.extend(batch.into_iter().map(|c| c.uuid));
            visited.extend(skips.into_iter().map(|s| s.uuid));
        }
        assert_eq!(visited.len(), 6, "only the hidden neurons are visited");
        assert!(
            !visited.iter().any(|u| outputs.contains(&u.as_str())),
            "an output neuron must never be proposed: {visited:?}"
        );
    }
}
