//! Seeded random sweep and sampled scorer screening (Issue #6).
//!
//! **Visits** are ordered once from a recorded seed and named
//! [`crate::ordering::Ordering`] and made **without replacement**. A visit is a
//! hidden-neuron UUID, or a synapse visit key naming one edge (Issue #135).
//!
//! A neuron visit tries an exact IDENTITY collapse, then a correlated-neuron
//! merge ([`crate::merge`], Issue #109) when discovery proposed a partner, then
//! a mean-activation ablation, then a constant substitution
//! ([`crate::substitute`], Issue #103) for the structure the ablation fails
//! closed on. A synapse visit resolves its source's fold value
//! ([`crate::stats::source_value`], Issue #134) and calls
//! [`crate::ablation::ablate_synapse`] (Issue #133). Attempts that produce
//! nothing are skipped with a [`crate::blocked::BlockedReason`] and the batch is
//! refilled while unvisited visits remain.
//!
//! An ordering only changes *when* something is tested (Issue #11). Every
//! candidate still passes `creature.validate()`, the sampled screen and full
//! authoritative scoring.
//!
//! The incumbent and every valid candidate in a batch are scored together in
//! one sampled scorer call. Sampled winners are returned for later
//! authoritative promotion; they never become `best.json` here.

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::path::{Path, PathBuf};
use std::time::Instant;

use neat_core::{CreatureExport, SquashType, creature_to_json, parse_squash_name};
use serde::Serialize;

use crate::ablation::{GroupMember, ablate_group, ablate_mean, ablate_synapse};
use crate::blocked::BlockedReason;
use crate::collapse::{CollapseOptions, CollapseSkip, collapse_identity};
use crate::incumbent::sha256_hex;
use crate::merge::{MergeSkip, merge_correlated};
use crate::ordering::{Ordering, OrderingConfig, hidden_order, synapse_order};
use crate::scorer::{DirectoryScorer, ScorerMode};
use crate::signature::MergeIndex;
use crate::stats::{ActivationStats, source_value};
use crate::substitute::substitute_constant;

/// Draw a seed from the clock and process id when the user omitted `--seed`.
pub fn draw_seed() -> u64 {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0);
    nanos ^ u64::from(std::process::id()).wrapping_shl(32) ^ 0xA5A5_A5A5_A5A5_A5A5
}

/// Separator inside a synapse visit key — ASCII UNIT SEPARATOR, `U+001F`.
///
/// A control character precisely because a neuron UUID is a label: NEAT-AI-core
/// creatures name neurons `input-N`, `output-N`, `h1` or a generated id, and
/// none of those forms can carry an unprintable byte. So a synapse visit key
/// can never be mistaken for — or collide with — a neuron UUID in any keyed
/// store the sweep, the screen coverage or the verdict cache shares (#135).
pub const SYNAPSE_KEY_SEPARATOR: char = '\u{1F}';

/// Leading segment that marks a visit key as an edge rather than a neuron.
pub const SYNAPSE_KEY_TAG: &str = "synapse";

/// The canonical visit key for the edge `from_uuid`→`to_uuid` (Issue #135).
///
/// One key per ordered pair, because NEAT-AI-core rule 26 lets a pair repeat
/// only with distinct roles and [`crate::ablation::ablate_synapse`] judges the
/// whole pair at once. Round-trips through [`parse_synapse_key`].
pub fn synapse_key(from_uuid: &str, to_uuid: &str) -> String {
    format!("{SYNAPSE_KEY_TAG}{SYNAPSE_KEY_SEPARATOR}{from_uuid}{SYNAPSE_KEY_SEPARATOR}{to_uuid}")
}

/// The endpoints of a synapse visit key, or `None` when `key` is not one.
///
/// Fails closed on anything that is not exactly three separated segments headed
/// by [`SYNAPSE_KEY_TAG`] with two non-empty endpoints, so a neuron UUID — or a
/// truncated key — never parses as an edge.
pub fn parse_synapse_key(key: &str) -> Option<(&str, &str)> {
    let rest = key.strip_prefix(SYNAPSE_KEY_TAG)?;
    let rest = rest.strip_prefix(SYNAPSE_KEY_SEPARATOR)?;
    let (from_uuid, to_uuid) = rest.split_once(SYNAPSE_KEY_SEPARATOR)?;
    if from_uuid.is_empty() || to_uuid.is_empty() || to_uuid.contains(SYNAPSE_KEY_SEPARATOR) {
        return None;
    }
    Some((from_uuid, to_uuid))
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
    /// Single-synapse ablation with bias compensation (#135).
    ///
    /// The finest unit the razor cuts: one **edge** goes and the target's bias
    /// absorbs what the source contributed on average, so a candidate can leave
    /// every neuron in place.
    Synapse,
}

/// One valid pruning candidate produced by the sweep.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SweepCandidate {
    /// Hidden neuron that was visited; the first member of a group (#108), or
    /// the synapse visit key for a [`CandidateKind::Synapse`] (#135).
    pub uuid: String,
    /// Every hidden neuron this candidate was asked to cut (Issue #108).
    ///
    /// One entry — [`Self::uuid`] — for a single-neuron candidate, and the
    /// whole neighbourhood, upstream-first, for a group proposal. Always
    /// non-empty and always headed by [`Self::uuid`], so a reader that only
    /// knows about single cuts still names a real visit.
    ///
    /// A [`CandidateKind::Synapse`] holds its **visit key**, not a neuron
    /// (Issue #135): an edge cut names no hidden neuron of its own, and what it
    /// cascades away is the transform's to report. A consumer that needs
    /// neuron uuids — bundle membership in [`crate::promote`], say — must read
    /// the kind, which is why the run does not yet walk edge visits.
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
    /// Source endpoint of the cut edge, for a [`CandidateKind::Synapse`] (#135).
    ///
    /// The same provenance contract [`Self::merged_with`] carries for a merge:
    /// [`Self::uuid`] is a visit **key**, not a neuron, so the edge it names has
    /// to travel with the candidate for a replay or a check-in to say what was
    /// cut. `None` — and unserialised — for every other kind.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub from_uuid: Option<String>,
    /// Destination endpoint of the cut edge, for a [`CandidateKind::Synapse`].
    #[serde(skip_serializing_if = "Option::is_none")]
    pub to_uuid: Option<String>,
    /// Weight the cut edge carried, for a [`CandidateKind::Synapse`].
    #[serde(skip_serializing_if = "Option::is_none")]
    pub weight: Option<f64>,
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
    /// Hidden neuron UUID, or the synapse visit key (Issue #135).
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

/// Seeded without-replacement walk over every visit (Issue #135).
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Sweep {
    /// Seed that produced [`Self::order`].
    pub seed: u64,
    /// Named ordering strategy that produced [`Self::order`].
    pub ordering: Ordering,
    /// SHA-256 of `seed`, the ordering name and the ordered visit list.
    pub permutation_identity: String,
    /// Visits in visitation order (Issue #135).
    ///
    /// A permutation of every hidden-neuron UUID **and** every synapse visit
    /// key ([`synapse_key`]) the incumbent carries, so a run is reconstructable
    /// from the identity above: an ordering reprioritises the walk, it never
    /// shrinks it.
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
    /// Synapse visits [`Self::retain_neuron_visits`] dropped (Issue #135).
    ///
    /// Recorded for the same reason as the two above: it changes the walk after
    /// the identity is fixed, so a run is only reconstructable if what it put
    /// aside is stated. `0` on a sweep that kept its edge half.
    pub synapse_visits_deferred: usize,
}

impl Sweep {
    /// Shuffle the incumbent's visits with `seed` (the random control).
    pub fn new(creature: &CreatureExport, seed: u64) -> Self {
        Self::with_ordering(
            creature,
            &ActivationStats::empty(),
            seed,
            OrderingConfig::default(),
        )
    }

    /// Order the incumbent's visits with `seed` and `cfg` (Issues #11, #135).
    ///
    /// The order is always a permutation of every hidden UUID **and** every
    /// synapse visit key: an ordering reprioritises the sweep, it never shrinks
    /// it. The strategies rank the hidden neurons ([`hidden_order`]); the
    /// synapse keys are shuffled from the same `seed` ([`synapse_order`]) and
    /// interleaved deterministically, so no strategy needs a per-synapse
    /// feature vector to mix edges into the walk.
    pub fn with_ordering(
        creature: &CreatureExport,
        stats: &ActivationStats,
        seed: u64,
        cfg: OrderingConfig<'_>,
    ) -> Self {
        let neurons = hidden_order(creature, stats, cfg, seed);
        let synapses = synapse_order(creature, seed);
        let order = interleave_visits(neurons, synapses, interleave_phase(seed));
        // The strategy that actually ranked, so a permutation identity always
        // names the ranking behind it (#107).
        let strategy = cfg.effective_strategy();
        let mut ident = format!(
            "seed={seed}\nordering={}\nrandomQuota={}\n",
            strategy.name(),
            cfg.random_quota
        );
        for visit in &order {
            ident.push_str(visit);
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
            synapse_visits_deferred: 0,
        }
    }

    /// Drop every synapse visit from the walk, returning how many went (#135).
    ///
    /// The one place the edge half of the pool is removed. It exists because
    /// the pool is built before the run can record, count or report an edge
    /// cut — screen-record and learnings parity is Issue #136, epoch coverage
    /// Issue #137, and accepting a pure synapse win Issue #138 — and a visit a
    /// run cannot record is a visit it would make again every batch forever.
    ///
    /// The count is returned rather than discarded: this reorders the walk
    /// after [`Self::permutation_identity`] is hashed, exactly as
    /// [`Self::prefer_unchecked`] and [`Self::prefer`] do, so the caller has to
    /// be able to say what it dropped for the run to stay reconstructable.
    pub fn retain_neuron_visits(&mut self) -> usize {
        let before = self.order.len();
        self.order
            .retain(|visit| parse_synapse_key(visit).is_none());
        self.next = self.next.min(self.order.len());
        self.synapse_visits_deferred = before - self.order.len();
        self.synapse_visits_deferred
    }

    /// Remaining unvisited visits — neurons and synapses alike (#135).
    pub fn remaining(&self) -> usize {
        self.order.len().saturating_sub(self.next)
    }

    /// True when every visit — hidden UUID and synapse key — has been made.
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
                        from_uuid: proposed.from_uuid,
                        to_uuid: proposed.to_uuid,
                        weight: proposed.weight,
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

/// The seed's offset into the interleave pattern, a fraction in `[0, 1)` (#135).
///
/// Drawn from the seed alone, so the mix is reproducible from the recorded
/// permutation identity and moves when the seed does.
fn interleave_phase(seed: u64) -> f64 {
    // The same SplitMix64 finaliser the orderings use, folded to a fraction:
    // the raw seed's low bits are not evenly spread, and a `seed % 100` phase
    // would give consecutive seeds neighbouring mixes.
    let mut z = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64
}

/// Mix `synapses` into `neurons` at an even rate, preserving both orders (#135).
///
/// Both input orders survive intact — the strategy's neuron ranking is not
/// reshuffled by the presence of edges, and neither list is truncated — so the
/// result is a permutation of every visit the incumbent has.
///
/// `phase` is the seed's say in *where* the edges land, and it is what stops
/// the mix having a fixed shape: without it the very first slot would go to a
/// synapse for every seed and every creature, so the top-ranked hidden neuron
/// (#107) could never be visited first. It is a fraction in `[0, 1)`, so the
/// even rate itself is unchanged — only the offset the pattern starts from.
fn interleave_visits(neurons: Vec<String>, synapses: Vec<String>, phase: f64) -> Vec<String> {
    if neurons.is_empty() || synapses.is_empty() {
        let mut out = neurons;
        out.extend(synapses);
        return out;
    }
    let total = neurons.len() + synapses.len();
    let quota = synapses.len() as f64 / total as f64;
    let mut out = Vec::with_capacity(total);
    let (mut at_neuron, mut at_synapse) = (0usize, 0usize);
    while out.len() < total {
        let take_synapse = at_synapse < synapses.len()
            && (at_neuron >= neurons.len()
                || (at_synapse as f64 + phase) < quota * (out.len() + 1) as f64);
        if take_synapse {
            out.push(synapses[at_synapse].clone());
            at_synapse += 1;
        } else {
            out.push(neurons[at_neuron].clone());
            at_neuron += 1;
        }
    }
    out
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
    /// Source endpoint for a [`CandidateKind::Synapse`]; `None` otherwise (#135).
    pub from_uuid: Option<String>,
    /// Destination endpoint for a [`CandidateKind::Synapse`]; `None` otherwise.
    pub to_uuid: Option<String>,
    /// Weight the cut edge carried, for a [`CandidateKind::Synapse`].
    pub weight: Option<f64>,
    /// Validated candidate creature.
    pub creature: CreatureExport,
}

impl Proposed {
    fn of(kind: CandidateKind, creature: CreatureExport) -> Self {
        Self {
            kind,
            merged_with: None,
            from_uuid: None,
            to_uuid: None,
            weight: None,
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
) -> Result<Proposed, Option<MergeRefusal>> {
    let mut refusal: Option<MergeRefusal> = None;
    for proposal in merges.for_removed(uuid) {
        match merge_correlated(incumbent, &proposal.survivor_uuid, uuid, proposal.relation) {
            Ok(merge) => {
                return Ok(Proposed {
                    kind: CandidateKind::Merge,
                    merged_with: Some(proposal.survivor_uuid.clone()),
                    from_uuid: None,
                    to_uuid: None,
                    weight: None,
                    creature: merge.creature,
                });
            }
            // Kept, not dropped: a run where every merge proposal fails on the
            // same structure must be able to say so rather than reporting only
            // whatever the fallback path went on to complain about. The
            // strongest partner's reason is the one named; the weaker partners
            // are counted, so a reader can tell one refusal from twenty.
            Err(skip) => match &mut refusal {
                None => {
                    refusal = Some(MergeRefusal {
                        first: skip,
                        others: 0,
                    });
                }
                Some(seen) => seen.others += 1,
            },
        };
    }
    Err(refusal)
}

/// Why every merge proposal for one visited neuron was refused (#109).
///
/// `first` is the strongest-correlation partner's reason — the one an operator
/// would act on — and `others` counts the weaker partners refused behind it, so
/// the report distinguishes a single unmergeable pair from a neuron nothing
/// will merge with.
struct MergeRefusal {
    first: MergeSkip,
    others: usize,
}

impl fmt::Display for MergeRefusal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.others {
            0 => write!(f, "{}", self.first),
            n => write!(f, "{} (and {n} weaker partner(s) refused)", self.first),
        }
    }
}

/// Append the merge refusal to `blocked`, when discovery proposed one at all.
///
/// The fallback path's own reason still classifies the visit — it is the one
/// that actually stopped the razor — but the merge attempt is named beside it
/// so a merge-enabled run can see why its proposals went nowhere.
fn with_merge_detail(mut blocked: Blocked, merge: Option<MergeRefusal>) -> Blocked {
    if let Some(refusal) = merge {
        blocked.detail = format!("{}; merge: {refusal}", blocked.detail);
    }
    blocked
}

/// The candidate that cuts the edge `from_uuid`→`to_uuid`, or why not (#135).
///
/// The source's fold value comes from the one resolver
/// ([`crate::stats::source_value`]), so a hidden, `constant` or input source is
/// asked for its value the same way. A source that resolves to nothing is a
/// fold this run cannot justify — [`BlockedReason::MissingActivation`] — and
/// every other refusal is the one [`ablate_synapse`] reports, under its own
/// reason code. The value is resolved **first**, so an edge that is both
/// unmeasured and structurally unsafe is filed under the missing value.
///
/// Resolving a value is not the same as being cuttable: an `input-N` source
/// resolves a mean and is then refused by [`ablate_synapse`], because an input
/// is not a listed neuron the transform can fold through (Issue #133). Those
/// edges are visited and blocked, never filtered out of the pool.
///
/// Nothing here weighs the edge: no weight, magnitude or contribution threshold
/// decides eligibility, because only the full-corpus scorer accepts.
fn propose_synapse(
    incumbent: &CreatureExport,
    stats: &ActivationStats,
    from_uuid: &str,
    to_uuid: &str,
) -> Result<Proposed, Blocked> {
    let Some(source) = source_value(incumbent, stats, from_uuid) else {
        return Err(Blocked::new(
            BlockedReason::MissingActivation,
            format!("no source value for `{from_uuid}` (synapse `{from_uuid}`→`{to_uuid}`)"),
        ));
    };
    match ablate_synapse(incumbent, from_uuid, to_uuid, source.value) {
        Ok(a) => Ok(Proposed {
            kind: CandidateKind::Synapse,
            merged_with: None,
            from_uuid: Some(a.from_uuid),
            to_uuid: Some(a.to_uuid),
            weight: Some(a.weight),
            creature: a.creature,
        }),
        Err(e) => Err(Blocked::new(e.blocked_reason(), e.to_string())),
    }
}

pub(crate) fn propose(
    incumbent: &CreatureExport,
    stats: &ActivationStats,
    merges: &MergeIndex,
    uuid: &str,
) -> Result<Proposed, Blocked> {
    // An edge visit is its own ladder: there is no identity collapse, merge or
    // constant substitution for a synapse, so it never enters the neuron one.
    //
    // A listed neuron wins the tie. Nothing in NEAT-AI-core forbids a creature
    // naming a neuron with the separator this key format uses, and a neuron
    // routed into the edge ladder would be a cut nothing asked for — so the
    // incumbent's own neuron list decides, not the shape of the string.
    if let Some((from_uuid, to_uuid)) = parse_synapse_key(uuid)
        && !incumbent.neurons.iter().any(|n| n.uuid == uuid)
    {
        return propose_synapse(incumbent, stats, from_uuid, to_uuid);
    }
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
    /// Hidden neuron that was screened, or the synapse visit key (#135).
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
            inputs: Vec::new(),
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

    /// Restrict `sweep` to its hidden-neuron visits.
    ///
    /// Issue #135 mixed synapse visits into the same seeded pool. The tests
    /// that use this are about the **neuron** ladder — identity → merge →
    /// ablation → constant substitution — so they walk the neuron half of the
    /// permutation and leave the edge half to the synapse tests below. Nothing
    /// about the neuron visits themselves changed, which is what these
    /// unchanged assertions go on demonstrating.
    fn neuron_visits_only(sweep: &mut Sweep) {
        sweep.retain_neuron_visits();
    }

    /// Every distinct synapse pair on `creature`, as visit keys.
    fn expected_synapse_keys(creature: &CreatureExport) -> Vec<String> {
        let mut keys: Vec<String> = creature
            .synapses
            .iter()
            .map(|s| synapse_key(&s.from_uuid, &s.to_uuid))
            .collect();
        keys.sort();
        keys.dedup();
        keys
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
        neuron_visits_only(&mut sweep);
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
        neuron_visits_only(&mut sweep);
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
        let mut control = Sweep::with_ordering(&creature, &stats, seed, OrderingConfig::default());
        let mut ranked = Sweep::with_ordering(
            &creature,
            &stats,
            seed,
            OrderingConfig::new(Ordering::LowVariance),
        );
        // The strategies rank hidden neurons and only hidden neurons (#135), so
        // the ranking claim below is made over the neuron half of the walk.
        neuron_visits_only(&mut control);
        neuron_visits_only(&mut ranked);
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
        // Every visit, not just every neuron: `prefer_unchecked` treats a
        // synapse key as an ordinary member of the walk (#135).
        let expected = {
            let mut u: Vec<String> = (0..6).map(|i| format!("h{i}")).collect();
            u.extend(expected_synapse_keys(&creature));
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
                "{} quota={} lost or duplicated a visit",
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
        // Over the neuron half, so "the reorder moved the tail" stays a claim
        // about the ranking rather than about whichever visit the mix put
        // first (#135).
        neuron_visits_only(&mut sweep);
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
        neuron_visits_only(&mut sweep);
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
        neuron_visits_only(&mut sweep);
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

    /// Issue #109: replay rebuilds a recorded verdict as the transform it was
    /// judged as. An index restricted to the uuids the cache recorded as merges
    /// re-derives a merge for those and the ordinary transform for the rest,
    /// so an accepted ablation cannot come back as a merge.
    #[test]
    fn a_restricted_index_re_derives_a_merge_only_for_the_uuids_it_names() {
        let creature = twins_and_a_stranger();
        let stats = twin_stats(&creature);
        let merges =
            crate::signature::discover(&stats, crate::signature::DiscoveryConfig::default());
        assert!(
            !merges.for_removed("h_a").is_empty() && !merges.for_removed("h_b").is_empty(),
            "both directions are proposed before the restriction"
        );

        let only_b = merges.restricted_to(&HashSet::from(["h_b".to_string()]));
        assert!(only_b.for_removed("h_a").is_empty());
        assert_eq!(
            only_b.report(),
            merges.report(),
            "the restriction re-runs no discovery, so its report is unchanged"
        );

        let as_merge = propose(&creature, &stats, &only_b, "h_b").unwrap();
        assert_eq!(as_merge.kind, CandidateKind::Merge);
        assert_eq!(as_merge.merged_with.as_deref(), Some("h_a"));

        let as_before = propose(&creature, &stats, &only_b, "h_a").unwrap();
        assert_ne!(
            as_before.kind,
            CandidateKind::Merge,
            "a uuid the cache did not record as a merge must not re-derive one"
        );
        assert!(as_before.merged_with.is_none());
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
        neuron_visits_only(&mut sweep);
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
        neuron_visits_only(&mut sweep);
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

    // ---------------------------------------------------------------------
    // Synapse visits (Issue #135)
    // ---------------------------------------------------------------------

    /// `stats_for`, plus a measured mean for every input wire.
    ///
    /// Without it an `input-N` source resolves to nothing and every edge out of
    /// an input is a blocked visit — correct, and tested below, but not what
    /// the mixed-pool tests are about.
    fn stats_with_inputs(creature: &CreatureExport) -> ActivationStats {
        let mut stats = stats_for(creature);
        stats.inputs = (0..creature.input)
            .map(|i| NeuronStats {
                uuid: format!("input-{i}"),
                neuron_index: i,
                count: 1,
                mean: 0.5,
                variance: 0.0,
                std_dev: 0.0,
                mean_abs: 0.5,
                min: 0.5,
                max: 0.5,
            })
            .collect();
        stats
    }

    /// A `condition` edge into an `IF` neuron — a typed pair that must fail closed.
    fn typed_edge_creature() -> CreatureExport {
        creature(
            1,
            1,
            vec![
                neuron("hidden", "h_cond", 0.0, Some("IDENTITY")),
                neuron("hidden", "h_if", 0.0, Some("IF")),
                neuron("output", "output-0", 0.0, Some("IDENTITY")),
            ],
            vec![
                synapse("input-0", "h_cond", 1.0),
                crate::fixtures::typed_synapse("h_cond", "h_if", 1.0, "condition"),
                crate::fixtures::typed_synapse("input-0", "h_if", 1.0, "positive"),
                crate::fixtures::typed_synapse("input-0", "h_if", -1.0, "negative"),
                synapse("h_if", "output-0", 1.0),
            ],
        )
    }

    /// An ordinary edge into a `MEAN` neuron — no bias can stand in for it.
    fn aggregate_target_creature() -> CreatureExport {
        creature(
            1,
            1,
            vec![
                neuron("hidden", "h_src", 0.0, Some("IDENTITY")),
                neuron("hidden", "h_mean", 0.0, Some("MEAN")),
                neuron("output", "output-0", 0.0, Some("IDENTITY")),
            ],
            vec![
                synapse("input-0", "h_src", 1.0),
                synapse("input-0", "h_mean", 1.0),
                synapse("h_src", "h_mean", 1.0),
                synapse("h_mean", "output-0", 1.0),
            ],
        )
    }

    /// A visit key round-trips, and nothing that is not one parses as one.
    #[test]
    fn a_synapse_visit_key_round_trips_and_never_collides_with_a_neuron_uuid() {
        for (from, to) in [
            ("h_a", "output-0"),
            ("input-0", "h_b"),
            ("6f709b1c-3180-f8b0-0000-000000000001", "output-11"),
        ] {
            let key = synapse_key(from, to);
            assert_eq!(parse_synapse_key(&key), Some((from, to)));
            assert_ne!(key, from, "a key is never one of its endpoints");
            assert_ne!(key, to);
        }
        // Distinct pairs never share a key, in either direction.
        assert_ne!(synapse_key("h_a", "h_b"), synapse_key("h_b", "h_a"));
        assert_ne!(synapse_key("a", "bc"), synapse_key("ab", "c"));
        // Nothing a neuron UUID can be parses as an edge.
        for not_a_key in [
            "h_a",
            "input-0",
            "output-0",
            SYNAPSE_KEY_TAG,
            "synapse:h_a:output-0",
            &format!("{SYNAPSE_KEY_TAG}{SYNAPSE_KEY_SEPARATOR}h_a"),
            &format!("{SYNAPSE_KEY_TAG}{SYNAPSE_KEY_SEPARATOR}{SYNAPSE_KEY_SEPARATOR}output-0"),
            &synapse_key("h_a", &synapse_key("h_b", "h_c")),
        ] {
            assert_eq!(parse_synapse_key(not_a_key), None, "{not_a_key:?}");
        }
    }

    /// Issue #135: the walk is every hidden neuron **and** every synapse, once.
    #[test]
    fn every_hidden_neuron_and_every_synapse_is_visited_exactly_once() {
        let creature = six_hidden();
        let stats = varied_stats(&creature);
        for cfg in every_ordering() {
            let sweep = Sweep::with_ordering(&creature, &stats, 11, cfg);
            let mut got = sweep.order.clone();
            let unique = {
                let mut u = got.clone();
                u.sort();
                u.dedup();
                u.len()
            };
            assert_eq!(unique, got.len(), "a visit must not repeat");
            got.retain(|v| parse_synapse_key(v).is_some());
            got.sort();
            assert_eq!(
                got,
                expected_synapse_keys(&creature),
                "{} quota={} lost a synapse visit",
                cfg.strategy,
                cfg.random_quota
            );
            let neurons = sweep.order.len() - got.len();
            assert_eq!(neurons, 6, "every hidden neuron is still visited");
        }
    }

    /// A typed pair is a visit, not a filtered-out edge: it is walked and it
    /// fails closed, because a blocked visit is still coverage.
    #[test]
    fn a_typed_synapse_enters_the_pool_rather_than_being_filtered_out() {
        let creature = typed_edge_creature();
        validate_creature(&creature).unwrap();
        let sweep = Sweep::new(&creature, 2);
        assert!(
            sweep.order.contains(&synapse_key("h_cond", "h_if")),
            "{:?}",
            sweep.order
        );
        // One key per pair, even though `input-0`→`h_if` carries two roles.
        assert_eq!(
            sweep
                .order
                .iter()
                .filter(|v| **v == synapse_key("input-0", "h_if"))
                .count(),
            1
        );
    }

    /// Issue #135: same creature, stats and seed — same mixed order and
    /// identity; a different seed changes both.
    #[test]
    fn the_seed_alone_decides_the_mixed_order_and_its_identity() {
        let creature = six_hidden();
        let stats = varied_stats(&creature);
        let cfg = OrderingConfig {
            strategy: Ordering::LowMeanAbs,
            random_quota: 0.25,
            priority: None,
        };
        let a = Sweep::with_ordering(&creature, &stats, 17, cfg);
        let b = Sweep::with_ordering(&creature, &stats, 17, cfg);
        assert_eq!(a.order, b.order);
        assert_eq!(a.permutation_identity, b.permutation_identity);
        assert!(a.order.iter().any(|v| parse_synapse_key(v).is_some()));

        let other = Sweep::with_ordering(&creature, &stats, 18, cfg);
        assert_ne!(a.order, other.order);
        assert_ne!(a.permutation_identity, other.permutation_identity);
    }

    /// The identity covers the **whole** ordered visit list: moving one visit
    /// changes it, so a run stays reconstructable from the hash it recorded.
    #[test]
    fn the_permutation_identity_covers_every_visit_including_the_synapses() {
        let creature = two_hidden();
        let stats = stats_with_inputs(&creature);
        let sweep = Sweep::with_ordering(&creature, &stats, 6, OrderingConfig::default());
        let ident = sweep.permutation_identity.clone();

        let mut fewer = creature.clone();
        fewer
            .synapses
            .retain(|s| s.to_uuid != "output-0" || s.from_uuid != "h_b");
        let dropped = Sweep::with_ordering(&fewer, &stats, 6, OrderingConfig::default());
        assert_ne!(
            ident, dropped.permutation_identity,
            "dropping a synapse visit must change the identity"
        );
    }

    /// Issue #135: one batch holds neuron **and** synapse candidates, and every
    /// one of them is a creature `validate()` accepts.
    #[test]
    fn fill_batch_returns_a_mixed_batch_of_valid_candidates() {
        let creature = two_hidden();
        validate_creature(&creature).unwrap();
        let stats = stats_with_inputs(&creature);
        let mut sweep = Sweep::new(&creature, 4);
        let visits = sweep.order.len();
        let (batch, skips) = sweep.fill_batch_avoiding(
            &creature,
            &stats,
            MergeIndex::empty(),
            visits,
            &HashSet::new(),
        );
        assert_eq!(batch.len() + skips.len(), visits, "every visit advances");
        assert!(
            batch.iter().any(|c| c.kind == CandidateKind::Synapse),
            "no synapse candidate: {batch:?}"
        );
        assert!(
            batch.iter().any(|c| c.kind != CandidateKind::Synapse),
            "no neuron candidate: {batch:?}"
        );
        for c in &batch {
            validate_creature(&c.creature)
                .expect("a synapse candidate must not bypass creature.validate()");
            assert_eq!(c.members.first(), Some(&c.uuid), "members head the uuid");
            assert!(!c.members.is_empty());
        }
    }

    /// Edge provenance travels with the candidate, and with nothing else.
    #[test]
    fn only_a_synapse_candidate_carries_and_serialises_its_edge() {
        let creature = two_hidden();
        let stats = stats_with_inputs(&creature);
        let mut sweep = Sweep::new(&creature, 4);
        let visits = sweep.order.len();
        let (batch, _) = sweep.fill_batch_avoiding(
            &creature,
            &stats,
            MergeIndex::empty(),
            visits,
            &HashSet::new(),
        );
        let mut saw_synapse = false;
        for c in &batch {
            let json: serde_json::Value = serde_json::to_value(c).unwrap();
            if c.kind == CandidateKind::Synapse {
                saw_synapse = true;
                let (from, to) = parse_synapse_key(&c.uuid).expect("a synapse names its edge");
                assert_eq!(c.from_uuid.as_deref(), Some(from));
                assert_eq!(c.to_uuid.as_deref(), Some(to));
                let weight = c.weight.expect("a synapse carries its weight");
                let carried = creature
                    .synapses
                    .iter()
                    .find(|s| s.from_uuid == from && s.to_uuid == to)
                    .expect("the edge is on the incumbent")
                    .weight;
                assert_eq!(weight, carried);
                assert_eq!(json["fromUuid"], from);
                assert_eq!(json["toUuid"], to);
                assert_eq!(json["weight"], serde_json::json!(weight));
            } else {
                assert!(c.from_uuid.is_none() && c.to_uuid.is_none() && c.weight.is_none());
                assert!(json.get("fromUuid").is_none(), "{json}");
                assert!(json.get("toUuid").is_none(), "{json}");
                assert!(json.get("weight").is_none(), "{json}");
            }
        }
        assert!(saw_synapse, "{batch:?}");
    }

    /// Issue #135: a visit that proposes nothing still advances the walk, with
    /// the reason code the refusal maps to.
    #[test]
    fn a_synapse_visit_that_proposes_nothing_is_skipped_with_its_reason() {
        // No input statistics, so an `input-N` source resolves to no fold value.
        let creature = two_hidden();
        let stats = stats_for(&creature);
        let blocked = propose(
            &creature,
            &stats,
            MergeIndex::empty(),
            &synapse_key("input-0", "h_a"),
        )
        .unwrap_err();
        assert_eq!(
            blocked.reason,
            BlockedReason::MissingActivation,
            "{blocked}"
        );

        let typed = typed_edge_creature();
        let typed_stats = stats_with_inputs(&typed);
        let blocked = propose(
            &typed,
            &typed_stats,
            MergeIndex::empty(),
            &synapse_key("h_cond", "h_if"),
        )
        .unwrap_err();
        assert_eq!(blocked.reason, BlockedReason::UnsafeTopology, "{blocked}");

        let aggregate = aggregate_target_creature();
        validate_creature(&aggregate).unwrap();
        let aggregate_stats = stats_with_inputs(&aggregate);
        let blocked = propose(
            &aggregate,
            &aggregate_stats,
            MergeIndex::empty(),
            &synapse_key("h_src", "h_mean"),
        )
        .unwrap_err();
        assert_eq!(blocked.reason, BlockedReason::AggregateSquash, "{blocked}");
    }

    /// Every refused visit is filed as a skip, so the walk always advances.
    #[test]
    fn a_creature_whose_edges_all_refuse_still_advances_visit_by_visit() {
        let creature = two_hidden();
        // No input means: every `input-N`→hidden edge is a blocked visit.
        let stats = stats_for(&creature);
        let mut sweep = Sweep::new(&creature, 8);
        let visits = sweep.order.len();
        let mut seen = Vec::new();
        while !sweep.exhausted() {
            let (batch, skips) = sweep.fill_batch(&creature, &stats, 1);
            assert!(!batch.is_empty() || !skips.is_empty());
            seen.extend(batch.into_iter().map(|c| c.uuid));
            seen.extend(skips.into_iter().map(|s| s.uuid));
        }
        assert_eq!(seen.len(), visits);
        let mut unique = seen.clone();
        unique.sort();
        unique.dedup();
        assert_eq!(unique.len(), visits, "no visit is made twice");
    }

    /// Selection treats a synapse key as an ordinary member of the walk.
    #[test]
    fn prefer_and_prefer_unchecked_move_synapse_visits_like_any_other() {
        let creature = two_hidden();
        let stats = stats_with_inputs(&creature);
        let mut sweep = Sweep::with_ordering(&creature, &stats, 3, OrderingConfig::default());
        let key = sweep
            .order
            .iter()
            .rev()
            .find(|v| parse_synapse_key(v).is_some())
            .cloned()
            .expect("the walk holds a synapse visit");
        assert_eq!(sweep.prefer(std::slice::from_ref(&key)), 1);
        assert_eq!(sweep.order[sweep.next], key);

        let before = sweep.order.clone();
        sweep.prefer_unchecked(&HashSet::from([key.clone()]), std::slice::from_ref(&key));
        assert_eq!(
            sweep.order.last(),
            Some(&key),
            "a screened synapse visit defers like a screened neuron"
        );
        let (mut got, mut expected) = (sweep.order.clone(), before);
        got.sort();
        expected.sort();
        assert_eq!(got, expected, "the reorder must stay a permutation");
    }

    /// The mix has no fixed shape: which stream leads, and where the edges
    /// land, moves with the seed. Without the phase the first slot went to a
    /// synapse for every seed, so the top-ranked hidden neuron (#107) could
    /// never be visited first.
    #[test]
    fn the_seed_moves_the_shape_of_the_mix_not_just_the_order_within_it() {
        let creature = six_hidden();
        let stats = varied_stats(&creature);
        let cfg = OrderingConfig::new(Ordering::LowVariance);
        let shapes: HashSet<Vec<bool>> = (0..64u64)
            .map(|seed| {
                Sweep::with_ordering(&creature, &stats, seed, cfg)
                    .order
                    .iter()
                    .map(|v| parse_synapse_key(v).is_some())
                    .collect()
            })
            .collect();
        assert!(
            shapes.len() > 1,
            "the neuron/synapse pattern must depend on the seed"
        );
        assert!(
            (0..64u64).any(|seed| {
                let sweep = Sweep::with_ordering(&creature, &stats, seed, cfg);
                parse_synapse_key(&sweep.order[0]).is_none()
            }),
            "a hidden neuron must be able to be the first visit"
        );
        assert!(
            (0..64u64).any(|seed| {
                let sweep = Sweep::with_ordering(&creature, &stats, seed, cfg);
                parse_synapse_key(&sweep.order[0]).is_some()
            }),
            "a synapse must be able to be the first visit"
        );
        // Still a permutation, whatever shape the seed picked.
        for seed in 0..64u64 {
            let mut got = Sweep::with_ordering(&creature, &stats, seed, cfg).order;
            let len = got.len();
            got.sort();
            got.dedup();
            assert_eq!(got.len(), len, "seed {seed} lost or duplicated a visit");
        }
    }

    /// One empty stream is the whole walk — the case a creature with no
    /// synapses (or none but synapses) takes.
    #[test]
    fn an_empty_stream_leaves_the_other_one_untouched() {
        let neurons = uuid_list(&["h0", "h1", "h2"]);
        let synapses = vec![synapse_key("h0", "output-0")];
        assert_eq!(
            interleave_visits(neurons.clone(), Vec::new(), 0.4),
            neurons,
            "no synapses means the neuron ranking is the walk"
        );
        assert_eq!(
            interleave_visits(Vec::new(), synapses.clone(), 0.4),
            synapses,
            "no hidden neurons means the edges are the walk"
        );
        assert!(interleave_visits(Vec::new(), Vec::new(), 0.4).is_empty());
        // And with both, nothing is lost whatever the phase.
        for phase in [0.0, 0.25, 0.5, 0.99] {
            let mut got = interleave_visits(neurons.clone(), synapses.clone(), phase);
            assert_eq!(got.len(), 4, "phase {phase}");
            got.sort();
            let mut expected: Vec<String> = neurons.iter().chain(&synapses).cloned().collect();
            expected.sort();
            assert_eq!(got, expected, "phase {phase}");
        }
    }

    /// Issue #133: an `input-N` source resolves a fold value and is then refused
    /// by the transform, because an input is not a listed neuron to fold
    /// through. The commonest refusal on a real creature, so it is pinned: the
    /// edge is visited and blocked, never quietly filtered out of the pool.
    #[test]
    fn an_input_sourced_edge_resolves_a_value_and_still_fails_closed() {
        let creature = two_hidden();
        let stats = stats_with_inputs(&creature);
        assert!(
            crate::stats::source_value(&creature, &stats, "input-0").is_some(),
            "the fixture must measure the input, or this tests the wrong refusal"
        );
        let blocked = propose(
            &creature,
            &stats,
            MergeIndex::empty(),
            &synapse_key("input-0", "h_a"),
        )
        .unwrap_err();
        assert_eq!(blocked.reason, BlockedReason::UnsafeTopology, "{blocked}");
        // And it is a *visit*, filed as a skip rather than dropped from the walk.
        let mut sweep = Sweep::new(&creature, 4);
        let visits = sweep.order.len();
        let (batch, skips) = sweep.fill_batch(&creature, &stats, visits);
        assert_eq!(batch.len() + skips.len(), visits);
        assert!(
            skips.iter().any(|s| s.uuid == synapse_key("input-0", "h_a")
                && s.blocked == Some(BlockedReason::UnsafeTopology)),
            "{skips:?}"
        );
    }

    /// A neuron whose uuid happens to look like a visit key is still a neuron:
    /// the incumbent's own neuron list decides, not the shape of the string.
    #[test]
    fn a_neuron_named_like_a_visit_key_takes_the_neuron_ladder() {
        let impostor = synapse_key("h_a", "output-0");
        let creature = creature(
            1,
            1,
            vec![
                neuron("hidden", &impostor, 0.0, Some("IDENTITY")),
                neuron("output", "output-0", 0.0, Some("IDENTITY")),
            ],
            vec![
                synapse("input-0", &impostor, 1.0),
                synapse(&impostor, "output-0", 1.0),
            ],
        );
        validate_creature(&creature).unwrap();
        let stats = stats_with_inputs(&creature);
        let proposed = propose(&creature, &stats, MergeIndex::empty(), &impostor)
            .expect("the impostor takes the neuron ladder");
        assert_ne!(proposed.kind, CandidateKind::Synapse);
        assert!(proposed.from_uuid.is_none() && proposed.to_uuid.is_none());
    }

    /// Issue #135: the label the verdict cache stores for an edge cut.
    #[test]
    fn a_synapse_candidate_is_labelled_synapse() {
        assert_eq!(
            crate::learnings::kind_label(CandidateKind::Synapse),
            "synapse"
        );
    }
}
