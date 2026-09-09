//! Ockham's request path into the canonical NEAT-AI-core pruning engine (#182).
//!
//! Ockham does not own what a structural rewrite does. `NEAT-AI-core`'s
//! [`prune_neuron`] and [`prune_synapse`] own the removal, the compensation,
//! the `IF`/typed-role rewrites, the cleanup fixed point and the validation;
//! this module is the request Ockham makes of them and the record it keeps of
//! what came back. `docs/pruning-ownership.md` is the checked-in contract.
//!
//! ```mermaid
//! flowchart LR
//!     V[sweep visits a neuron or an edge] --> H["hints: mean, variance"]
//!     H --> C["core prune_neuron / prune_synapse"]
//!     C -- PruneError --> B["Blocked, under an Ockham reason code"]
//!     C -- PruneResult --> G["validate_creature guard"]
//!     G -- rejected --> B
//!     G -- accepted --> P["PrunedCandidate + PruneDetail"]
//!     P --> S[screen, score, report]
//! ```
//!
//! There is **no second rewrite** here and no fallback to one: a candidate core
//! will not build is not a candidate. The [`validate_creature`] guard is not a
//! repair path — it fails the visit loudly rather than letting a creature core
//! promised was canonical reach the scorer unchecked.

use neat_core::{
    CreatureExport, PruneError, PruneResult, PruneStats, SynapseKey, SynapseType,
    TransformClass as CoreTransformClass, UncompensatedReason, parse_synapse_type, prune_neuron,
    prune_synapse,
};
use serde::{Deserialize, Serialize};

use crate::ablation::{StructureSnapshot, TransformClass};
use crate::blocked::BlockedReason;
use crate::incumbent::validate_creature;
use crate::stats::NeuronStats;

/// One `(from, to, role)` edge, in the form telemetry carries it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SynapseRef {
    /// Source endpoint.
    pub from_uuid: String,
    /// Destination endpoint.
    pub to_uuid: String,
    /// Role the edge played at its target (`standard`, `positive`, …).
    pub role: String,
}

impl SynapseRef {
    fn of(key: &SynapseKey) -> Self {
        Self {
            from_uuid: key.from_uuid.clone(),
            to_uuid: key.to_uuid.clone(),
            role: role_label(key.role).to_string(),
        }
    }
}

/// One compensating bias fold core applied.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BiasFoldRecord {
    /// Target whose bias moved.
    pub target_uuid: String,
    /// Total weight the removal took out of that target.
    pub weight_sum: f64,
    /// What was added to the bias.
    pub delta: f64,
    /// `true` when the folded value is what the creature computed on every
    /// record, so the fold changes nothing.
    pub exact: bool,
    /// Variance the compensation could not carry, when Ockham measured one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub residual_variance: Option<f64>,
}

/// Weight core moved onto a correlated survivor's edge.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct WeightShareRecord {
    /// Survivor carrying the correlated part.
    pub from_uuid: String,
    /// Target it feeds.
    pub to_uuid: String,
    /// What was added to that edge's weight.
    pub delta: f64,
}

/// A target that read the removed structure and got nothing back.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct UncompensatedRecord {
    /// Target that lost the term.
    pub target_uuid: String,
    /// Role the removed edges played there.
    pub role: String,
    /// Total weight that role lost.
    pub weight_sum: f64,
    /// The target's squash, which is what makes an aggregate uncompensable.
    pub squash: String,
    /// `no-statistics` or `aggregate-target`.
    pub reason: String,
}

/// An `IF` the removal left with a condition the creature itself decides.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct StaticIfRecord {
    /// The `IF` neuron.
    pub uuid: String,
    /// Branch the condition always takes.
    pub branch: String,
}

/// What the core engine did, in the form Ockham telemetry records (#182).
///
/// Every field is core's own report of the rewrite — nothing here is Ockham
/// re-deriving what the graph became — so a run's evidence says whether a cut
/// was exact or approximate, what the cascade took, which `IF` structure was
/// rewritten, and which targets were left carrying the removal uncompensated.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PruneDetail {
    /// Whether the whole rewrite preserves the creature's output exactly.
    pub transform_class: TransformClass,
    /// Cleanup passes the fixed point took, summed over the requests made.
    pub passes: usize,
    /// Neurons the request itself removed, in request order.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub requested_neurons: Vec<String>,
    /// Edges the request itself removed.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub removed_synapses: Vec<SynapseRef>,
    /// Neurons the cleanup cascade removed on top of the request.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub cascade_neurons: Vec<String>,
    /// Edges the cascade removed alongside them.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub cascade_synapses: Vec<SynapseRef>,
    /// Hidden neurons the cascade folded into constant support.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub folded_neurons: Vec<String>,
    /// `IF` neurons downgraded to `IDENTITY` — the one inexact cleanup rewrite.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub downgraded_if_neurons: Vec<String>,
    /// `IF` neurons flattened to the branch their condition always takes.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub static_if_neurons: Vec<StaticIfRecord>,
    /// Zero-weight support edges added to give an `IF` back an emptied role.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub restored_if_roles: Vec<SynapseRef>,
    /// The compensating folds applied.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub bias_folds: Vec<BiasFoldRecord>,
    /// The correlated-survivor shares applied.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub weight_shares: Vec<WeightShareRecord>,
    /// Targets left carrying the removal without compensation.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub uncompensated: Vec<UncompensatedRecord>,
}

impl PruneDetail {
    /// Absorb one core [`PruneResult`], so a group of requests reads as one
    /// transform: the classes combine (any approximate step makes the whole
    /// candidate approximate) and the passes add up.
    fn absorb(&mut self, result: &PruneResult) {
        if result.transform == CoreTransformClass::Approximate {
            self.transform_class = TransformClass::Approximate;
        }
        self.passes += result.passes;
        if let Some(uuid) = &result.removed_neuron {
            self.requested_neurons.push(uuid.clone());
        }
        self.removed_synapses
            .extend(result.removed_synapses.iter().map(SynapseRef::of));
        self.cascade_neurons
            .extend(result.cascade_neurons.iter().cloned());
        self.cascade_synapses
            .extend(result.cascade_synapses.iter().map(SynapseRef::of));
        self.folded_neurons
            .extend(result.folded_neurons.iter().cloned());
        self.downgraded_if_neurons
            .extend(result.downgraded_if_neurons.iter().cloned());
        self.static_if_neurons
            .extend(result.static_if_neurons.iter().map(|r| StaticIfRecord {
                uuid: r.uuid.clone(),
                branch: role_label(r.branch).to_string(),
            }));
        self.restored_if_roles
            .extend(result.restored_if_roles.iter().map(SynapseRef::of));
        self.bias_folds
            .extend(result.bias_folds.iter().map(|f| BiasFoldRecord {
                target_uuid: f.target_uuid.clone(),
                weight_sum: f.weight_sum,
                delta: f.delta,
                exact: f.exact,
                residual_variance: f.residual_variance,
            }));
        self.weight_shares
            .extend(result.weight_shares.iter().map(|s| WeightShareRecord {
                from_uuid: s.from_uuid.clone(),
                to_uuid: s.to_uuid.clone(),
                delta: s.delta,
            }));
        self.uncompensated
            .extend(result.uncompensated.iter().map(|u| UncompensatedRecord {
                target_uuid: u.target_uuid.clone(),
                role: role_label(u.role).to_string(),
                weight_sum: u.weight_sum,
                squash: u.squash.to_string(),
                reason: match u.reason {
                    UncompensatedReason::NoStatistics => "no-statistics".to_string(),
                    UncompensatedReason::AggregateTarget => "aggregate-target".to_string(),
                },
            }));
    }

    /// Neurons the transform took out of the hidden set on top of the
    /// requested ones: what the cascade removed, and what it folded into
    /// constant support.
    ///
    /// The two are recorded apart from the request because they answer
    /// different questions: the request is what the razor chose to cut, this is
    /// what that choice stranded. A learning that conflated them could not
    /// reconstruct the proposal it came from.
    pub fn cascade_uuids(&self) -> Vec<String> {
        let mut seen = std::collections::HashSet::new();
        self.cascade_neurons
            .iter()
            .chain(self.folded_neurons.iter())
            // A group cut can fold a neuron into constant support on one
            // request and strand it on a later one, so the same uuid reaches
            // here twice; it names one neuron either way.
            .filter(|uuid| seen.insert(uuid.as_str()))
            .cloned()
            .collect()
    }

    /// Every neuron the transform removed: the requested ones, then the cascade.
    pub fn removed_uuids(&self) -> Vec<String> {
        self.requested_neurons
            .iter()
            .cloned()
            .chain(self.cascade_uuids())
            .collect()
    }
}

/// A candidate the core engine built, and what it cost.
#[derive(Debug, Clone, PartialEq)]
pub struct PrunedCandidate {
    /// Canonical, validated candidate creature.
    pub creature: CreatureExport,
    /// Structure before the transform.
    pub before: StructureSnapshot,
    /// Structure after it.
    pub after: StructureSnapshot,
    /// Core's own report of the rewrite.
    pub detail: PruneDetail,
}

impl PrunedCandidate {
    /// Whether every target that read the removed structure got something back.
    ///
    /// `false` names the case Ockham's own constant substitution (#103) was
    /// built for: core still returns a valid creature, but a target that
    /// aggregates its inward terms cannot absorb a bias fold, so the candidate
    /// simply loses the term.
    pub fn fully_compensated(&self) -> bool {
        self.detail.uncompensated.is_empty()
    }
}

/// Why the core engine produced no candidate.
///
/// Every refusal means **no creature**: nothing partially rewritten reaches the
/// screen or the scorer.
#[derive(Debug, Clone, PartialEq)]
pub struct PruneRefusal {
    reason: BlockedReason,
    detail: String,
}

impl PruneRefusal {
    /// The reason code the blocked visit is counted under (#103).
    pub fn blocked_reason(&self) -> BlockedReason {
        self.reason
    }

    /// Core's own message for the refusal.
    fn of(error: &PruneError) -> Self {
        Self {
            reason: reason_for(error),
            detail: error.to_string(),
        }
    }
}

impl std::fmt::Display for PruneRefusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.detail)
    }
}

/// The Ockham reason code one core refusal is counted under.
///
/// Structured on the error, never parsed out of its message: the blocked tally
/// has to be deterministic (`docs/blocked-reasons.md`).
fn reason_for(error: &PruneError) -> BlockedReason {
    match error {
        // A statistic Ockham supplied is not a number core can use, so the
        // visit had no usable activation after all.
        PruneError::NonFiniteStatistic { .. }
        | PruneError::NegativeVariance { .. }
        | PruneError::DegenerateProxy { .. }
        | PruneError::InconsistentCovariance { .. }
        | PruneError::UnknownProxy { .. }
        | PruneError::MissingProxyEdge { .. } => BlockedReason::MissingActivation,
        // The request named structure that is not a caller's to remove, or is
        // not there at all.
        PruneError::UnknownNeuron { .. }
        | PruneError::UnknownSynapse { .. }
        | PruneError::Protected { .. }
        | PruneError::UnknownNeuronType { .. } => BlockedReason::UnsafeTopology,
        // Core promised a canonical validated creature or none at all, so a
        // cleanup failure is a rewrite-engine defect, not a normal outcome
        // (`docs/pruning-ownership.md`).
        PruneError::Cleanup(_) => BlockedReason::ValidationFailed,
    }
}

/// The wire name of a synapse role.
fn role_label(role: SynapseType) -> &'static str {
    match role {
        SynapseType::Standard => "standard",
        SynapseType::Positive => "positive",
        SynapseType::Negative => "negative",
        SynapseType::Condition => "condition",
    }
}

/// Whether `creature` carries `uuid` as a hidden neuron.
fn is_hidden(creature: &CreatureExport, uuid: &str) -> bool {
    creature
        .neurons
        .iter()
        .any(|n| n.uuid == uuid && n.neuron_type == "hidden")
}

/// One hidden neuron and the mean that stands in for it (#108).
#[derive(Debug, Clone, PartialEq)]
pub struct GroupMember {
    /// Hidden neuron UUID.
    pub uuid: String,
    /// Full-corpus mean post-activation of that neuron.
    pub mean: f64,
}

/// The statistical hint Ockham has about one neuron, in core's form.
///
/// The mean is what stands in for the removed activation; the variance is
/// passed so core can report the residual variance the fold could not carry.
/// A variance is only offered once two records have been accumulated — a
/// single-record population variance is zero by construction, not a
/// measurement — and never when it is not a finite, non-negative number, which
/// core would refuse the whole request for.
fn hint(mean: f64, stats: Option<&NeuronStats>) -> PruneStats {
    PruneStats {
        mean_activation: mean,
        variance: stats
            .filter(|s| s.count >= 2)
            .map(|s| s.variance)
            .filter(|v| v.is_finite() && *v >= 0.0),
        proxy: None,
    }
}

/// Build the candidate, once core has returned one.
///
/// The [`validate_creature`] guard is Ockham's fail-loud boundary: core
/// promises a canonical validated creature, and a creature that is not one is
/// reported as [`BlockedReason::ValidationFailed`] rather than screened.
fn candidate(before: StructureSnapshot, detail: PruneDetail, creature: CreatureExport) -> Result<PrunedCandidate, PruneRefusal> {
    validate_creature(&creature).map_err(|e| PruneRefusal {
        reason: BlockedReason::ValidationFailed,
        detail: format!("core prune returned a creature validation rejects: {e}"),
    })?;
    let after = StructureSnapshot::of(&creature);
    Ok(PrunedCandidate {
        creature,
        before,
        after,
        detail,
    })
}

/// Prune hidden neuron `uuid` from `incumbent` through core (#182).
///
/// `mean` is the full-corpus mean post-activation Ockham measured, and `stats`
/// the rest of what it measured about the same neuron. Core decides what the
/// graph becomes: the compensation, the cascade, the `IF` rewrites, the
/// canonical fixed point and the validation are all its own.
///
/// # Errors
///
/// [`PruneRefusal`] when core refuses the request, or when the creature it
/// returned does not validate here — no candidate in either case.
pub fn prune_hidden_neuron(
    incumbent: &CreatureExport,
    uuid: &str,
    mean: f64,
    stats: Option<&NeuronStats>,
) -> Result<PrunedCandidate, PruneRefusal> {
    let before = StructureSnapshot::of(incumbent);
    let result =
        prune_neuron(incumbent, uuid, Some(&hint(mean, stats))).map_err(|e| PruneRefusal::of(&e))?;
    let mut detail = PruneDetail::default();
    detail.absorb(&result);
    candidate(before, detail, result.creature)
}

/// Prune a whole neighbourhood of hidden neurons through core (#108, #182).
///
/// Members are requested one at a time, upstream-first, each against the
/// creature the last request returned — so core's cleanup fixed point runs
/// between them and the group is a sequence of canonical creatures rather than
/// one hand-rolled multi-removal. A member an earlier member's cascade already
/// took is not requested again: it is already gone, and core would refuse to
/// remove a neuron the creature no longer carries.
///
/// # Errors
///
/// [`PruneRefusal`] when the group is empty, when core refuses any member, or
/// when the creature it returned does not validate here.
pub fn prune_hidden_group(
    incumbent: &CreatureExport,
    members: &[GroupMember],
) -> Result<PrunedCandidate, PruneRefusal> {
    let before = StructureSnapshot::of(incumbent);
    let mut requested: Vec<&GroupMember> = Vec::with_capacity(members.len());
    let mut seen = std::collections::HashSet::new();
    for member in members {
        if seen.insert(member.uuid.as_str()) {
            requested.push(member);
        }
    }
    // Named rather than silently returning the incumbent: a candidate that
    // removes nothing would be scored, tie the baseline and be reported as a
    // proposal the razor tried.
    if requested.is_empty() {
        return Err(PruneRefusal {
            reason: BlockedReason::UnsafeTopology,
            detail: "group cut requested with no members".to_string(),
        });
    }

    let mut working = incumbent.clone();
    let mut detail = PruneDetail::default();
    let mut cut_any = false;
    for member in requested {
        // A member an earlier request already dealt with — removed by its
        // cascade, or folded into constant support — is not asked for again:
        // core would refuse to remove a neuron the creature no longer carries,
        // or a constant, and the cascade has already recorded what became of
        // it. Every other member is requested, so a uuid the incumbent never
        // carried as a hidden neuron gets core's own refusal rather than being
        // stepped over.
        let was_hidden = is_hidden(incumbent, &member.uuid);
        if was_hidden && !is_hidden(&working, &member.uuid) {
            continue;
        }
        let result = prune_neuron(&working, &member.uuid, Some(&hint(member.mean, None)))
            .map_err(|e| PruneRefusal::of(&e))?;
        detail.absorb(&result);
        working = result.creature;
        cut_any = true;
    }
    if !cut_any {
        return Err(PruneRefusal {
            reason: BlockedReason::UnsafeTopology,
            detail: "group cut requested with no members the incumbent carries".to_string(),
        });
    }
    candidate(before, detail, working)
}

/// Prune the edge `from_uuid`→`to_uuid` from `incumbent` through core (#182).
///
/// `source_value` is what the source contributed on average — resolved by the
/// one resolver, [`crate::stats::source_value`] — and `source_stats` is the
/// rest of what Ockham measured about that source. A source the creature
/// itself fixes (a constant, a neuron with nothing to sum) is core's to value,
/// and the hint is ignored for it.
///
/// The visit key names a **pair**, and a pair may repeat with distinct roles at
/// an `IF` (NEAT-AI-core rule 26), so the role of the first edge the incumbent
/// lists for the pair is what the request names — one edge, deterministically
/// chosen, reported on [`PruneDetail::removed_synapses`].
///
/// # Errors
///
/// [`PruneRefusal`] when the incumbent carries no such pair, when core refuses
/// the request, or when the creature it returned does not validate here.
pub fn prune_edge(
    incumbent: &CreatureExport,
    from_uuid: &str,
    to_uuid: &str,
    source_value: f64,
    source_stats: Option<&NeuronStats>,
) -> Result<PrunedCandidate, PruneRefusal> {
    let Some(role) = incumbent
        .synapses
        .iter()
        .find(|s| s.from_uuid == from_uuid && s.to_uuid == to_uuid)
        .map(|s| parse_synapse_type(s.synapse_type.as_deref()))
    else {
        return Err(PruneRefusal {
            reason: BlockedReason::UnsafeTopology,
            detail: format!("no synapse `{from_uuid}`→`{to_uuid}`"),
        });
    };
    let key = SynapseKey {
        from_uuid: from_uuid.to_string(),
        to_uuid: to_uuid.to_string(),
        role,
    };
    let before = StructureSnapshot::of(incumbent);
    let result = prune_synapse(incumbent, &key, Some(&hint(source_value, source_stats)))
        .map_err(|e| PruneRefusal::of(&e))?;
    let mut detail = PruneDetail::default();
    detail.absorb(&result);
    candidate(before, detail, result.creature)
}

/// Total weight the edges named by `keys` carry on `creature`.
///
/// What the removal actually took out of the target, read from the creature the
/// request was made against rather than recomputed from the result.
pub fn removed_weight(creature: &CreatureExport, keys: &[SynapseRef]) -> f64 {
    keys.iter()
        .flat_map(|key| {
            creature.synapses.iter().filter(move |s| {
                s.from_uuid == key.from_uuid
                    && s.to_uuid == key.to_uuid
                    && role_label(parse_synapse_type(s.synapse_type.as_deref())) == key.role
            })
        })
        .map(|s| s.weight)
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixtures::{creature, hidden_identity_creature, neuron, synapse, typed_synapse};
    use neat_core::{SquashType, apply_squash, compile_creature};

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() <= 1e-12 * a.abs().max(b.abs()).max(1.0)
    }

    fn bias_of(creature: &CreatureExport, uuid: &str) -> f64 {
        creature
            .neurons
            .iter()
            .find(|n| n.uuid == uuid)
            .unwrap_or_else(|| panic!("no neuron {uuid}"))
            .bias
    }

    fn uuids(creature: &CreatureExport) -> Vec<&str> {
        creature.neurons.iter().map(|n| n.uuid.as_str()).collect()
    }

    /// Seconds to propose the same prune `ROUNDS` times on a `hidden`-wide creature.
    fn prune_seconds(hidden: usize) -> f64 {
        const ROUNDS: usize = 3;
        let wide = crate::fixtures::wide_creature(8, hidden, "TANH");
        let started = std::time::Instant::now();
        for _ in 0..ROUNDS {
            prune_hidden_neuron(&wide, "h0", 0.25, None).expect("h0 feeds the output");
        }
        started.elapsed().as_secs_f64()
    }

    /// Issue #91, carried across the migration: one prune must cost the
    /// creature, not its square. A GRQ forest is thousands of neurons wide, so
    /// a quadratic rewrite is the difference between filling a batch and
    /// screening two an hour.
    ///
    /// A ratio, never a wall-clock budget (the standards forbid those): the
    /// same work is timed at one size and four times that size on the same
    /// machine, so a slower machine slows both readings and the test still
    /// holds. Growing with the creature is ~4x; growing with its square ~16x.
    ///
    /// The small reading is taken twice, on either side of the large one, and
    /// the larger of the two is used: load arriving *during* the test would
    /// otherwise inflate only the second reading and fail a correct engine.
    #[test]
    fn one_prune_costs_the_creature_not_its_square() {
        let before = prune_seconds(400);
        let large = prune_seconds(1_600);
        let after = prune_seconds(400);
        let small = before.max(after).max(1e-9);
        let growth = large / small;
        assert!(
            growth < 8.0,
            "four times the creature must not cost sixteen times the work: {growth:.1}x \
             ({small:.4}s → {large:.4}s)"
        );
    }

    fn two_hidden() -> CreatureExport {
        // input → h_a → output (weight 3)
        // input → h_b → output (weight 1)
        creature(
            1,
            1,
            vec![
                neuron("hidden", "h_a", 0.0, Some("IDENTITY")),
                neuron("hidden", "h_b", 0.0, Some("IDENTITY")),
                neuron("output", "output-0", 0.25, Some("IDENTITY")),
            ],
            vec![
                synapse("input-0", "h_a", 1.0),
                synapse("input-0", "h_b", 1.0),
                synapse("h_a", "output-0", 3.0),
                synapse("h_b", "output-0", 1.0),
            ],
        )
    }

    fn chain_plus_keep() -> CreatureExport {
        // input → h_up → h_leaf → output
        // input → h_keep → output
        creature(
            1,
            1,
            vec![
                neuron("hidden", "h_up", 0.1, Some("IDENTITY")),
                neuron("hidden", "h_leaf", 0.0, Some("IDENTITY")),
                neuron("hidden", "h_keep", 0.0, Some("IDENTITY")),
                neuron("output", "output-0", 0.0, Some("IDENTITY")),
            ],
            vec![
                synapse("input-0", "h_up", 1.0),
                synapse("h_up", "h_leaf", 1.0),
                synapse("h_leaf", "output-0", 2.0),
                synapse("input-0", "h_keep", 1.0),
                synapse("h_keep", "output-0", 1.0),
            ],
        )
    }

    fn constant_fold_fixture() -> CreatureExport {
        // h_src = IDENTITY(1.5 + 0 * x) is constant 1.5.
        // h_mid = TANH(h_src); output = 2 * h_mid + h_keep.
        creature(
            1,
            1,
            vec![
                neuron("hidden", "h_src", 1.5, Some("IDENTITY")),
                neuron("hidden", "h_mid", 0.0, Some("TANH")),
                neuron("hidden", "h_keep", 0.0, Some("IDENTITY")),
                neuron("output", "output-0", 0.0, Some("IDENTITY")),
            ],
            vec![
                synapse("input-0", "h_src", 0.0),
                synapse("h_src", "h_mid", 1.0),
                synapse("h_mid", "output-0", 2.0),
                synapse("input-0", "h_keep", 1.0),
                synapse("h_keep", "output-0", 1.0),
            ],
        )
    }

    fn typed_if_fixture() -> CreatureExport {
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
                typed_synapse("h_cond", "h_if", 1.0, "condition"),
                typed_synapse("input-0", "h_if", 1.0, "positive"),
                typed_synapse("input-0", "h_if", -1.0, "negative"),
                synapse("h_if", "output-0", 1.0),
            ],
        )
    }

    /// `h_a` feeds both outputs, `h_b` only the first, so cutting
    /// `h_a`→`output-0` strands nothing: a pure edge removal.
    fn shared_output_fan_out() -> CreatureExport {
        creature(
            1,
            2,
            vec![
                neuron("hidden", "h_a", 0.0, Some("IDENTITY")),
                neuron("hidden", "h_b", 0.0, Some("IDENTITY")),
                neuron("output", "output-0", 0.25, Some("IDENTITY")),
                neuron("output", "output-1", 0.0, Some("IDENTITY")),
            ],
            vec![
                synapse("input-0", "h_a", 2.0),
                synapse("input-0", "h_b", 1.0),
                synapse("h_a", "output-0", 3.0),
                synapse("h_a", "output-1", 1.0),
                synapse("h_b", "output-0", 1.0),
            ],
        )
    }

    fn constant_leaf() -> CreatureExport {
        // `c0` emits 0.5 into the output and feeds nothing else.
        creature(
            1,
            1,
            vec![
                neuron("constant", "c0", 0.5, None),
                neuron("hidden", "h_keep", 0.0, Some("IDENTITY")),
                neuron("output", "output-0", 0.0, Some("IDENTITY")),
            ],
            vec![
                synapse("c0", "output-0", 2.0),
                synapse("input-0", "h_keep", 1.0),
                synapse("h_keep", "output-0", 1.0),
            ],
        )
    }

    fn ordinary_edge_into_aggregate() -> CreatureExport {
        // `h_mean` averages its inward synapses, so no bias can stand in for one.
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

    fn outputs(creature: &CreatureExport, xs: &[f32]) -> Vec<f32> {
        let mut net = compile_creature(creature).unwrap();
        xs.iter().map(|&x| net.activate(&[x], 1)[0]).collect()
    }

    /// Mean of output `index` over `xs`, in f64.
    fn mean_output(creature: &CreatureExport, xs: &[f32], index: usize) -> f64 {
        let mut net = compile_creature(creature).unwrap();
        let sum: f64 = xs
            .iter()
            .map(|&x| f64::from(net.activate(&[x], creature.output)[index]))
            .sum();
        sum / xs.len() as f64
    }

    /// Inputs whose mean is exactly 0.5, all exactly representable in f32.
    const XS: [f32; 4] = [1.0, 2.0, -1.0, 0.0];

    #[test]
    fn the_measured_mean_is_folded_into_what_read_the_neuron() {
        let incumbent = two_hidden();
        validate_creature(&incumbent).unwrap();
        let original = incumbent.clone();
        let result = prune_hidden_neuron(&incumbent, "h_a", 2.0, None).unwrap();
        assert_eq!(incumbent, original, "incumbent must be untouched");
        assert_eq!(result.detail.transform_class, TransformClass::Approximate);
        let bias = bias_of(&result.creature, "output-0");
        assert!(close(bias, 0.25 + 2.0 * 3.0), "bias {bias}");
        assert_eq!(result.detail.bias_folds.len(), 1);
        assert_eq!(result.detail.bias_folds[0].target_uuid, "output-0");
        assert!(close(result.detail.bias_folds[0].weight_sum, 3.0));
        assert_eq!(result.detail.requested_neurons, vec!["h_a"]);
        assert!(result.fully_compensated());
        assert!(result.creature.neurons.iter().all(|n| n.uuid != "h_a"));
        assert!(result.creature.neurons.iter().any(|n| n.uuid == "h_b"));
        validate_creature(&result.creature).unwrap();
    }

    #[test]
    fn the_cascade_removes_a_newly_unreachable_chain() {
        let incumbent = chain_plus_keep();
        let result = prune_hidden_neuron(&incumbent, "h_leaf", 1.0, None).unwrap();
        let left = uuids(&result.creature);
        assert!(!left.contains(&"h_leaf"));
        assert!(
            !left.contains(&"h_up"),
            "h_up lost its only outgoing and must cascade: {left:?}"
        );
        assert!(left.contains(&"h_keep"));
        assert_eq!(result.detail.cascade_neurons, vec!["h_up"]);
        assert_eq!(result.detail.removed_uuids(), vec!["h_leaf", "h_up"]);
        let bias = bias_of(&result.creature, "output-0");
        assert!(close(bias, 1.0 * 2.0), "bias {bias}");
        validate_creature(&result.creature).unwrap();
    }

    #[test]
    fn a_neuron_whose_removal_fixes_its_reader_keeps_the_outputs() {
        let incumbent = constant_fold_fixture();
        let xs = [0.0f32, 1.0, -2.5, 3.25];
        let before = outputs(&incumbent, &xs);
        // `h_src` is IDENTITY(1.5 + 0 * x), so its measured mean is exactly
        // 1.5. Removing it leaves `h_mid` with nothing to sum, and the cleanup
        // resolves that to constant support rather than leaving a hidden
        // neuron whose activation never moves.
        let result = prune_hidden_neuron(&incumbent, "h_src", 1.5, None).unwrap();
        assert_eq!(result.detail.folded_neurons, vec!["h_mid"]);
        let after = outputs(&result.creature, &xs);
        for (x, (a, b)) in xs.iter().zip(before.iter().zip(after.iter())) {
            assert!(
                (a - b).abs() <= 1e-5 * a.abs().max(b.abs()).max(1.0),
                "x={x}: {a} vs {b}"
            );
        }
        // Whatever shape the cleanup chose, `h_mid` is worth tanh(1.5) on
        // every record and the creature still computes 2 * that.
        let expected_const = f64::from(apply_squash(SquashType::Tanh, 1.5));
        let folded = match result.creature.neurons.iter().find(|n| n.uuid == "h_mid") {
            // Kept as constant support: it emits its bias, and the edge into
            // the output carries whatever weight makes the sum right.
            Some(support) => {
                let weight = result
                    .creature
                    .synapses
                    .iter()
                    .find(|s| s.from_uuid == "h_mid" && s.to_uuid == "output-0")
                    .expect("the support constant still feeds the output")
                    .weight;
                support.bias * weight
            }
            // Folded away entirely: the output carries it as bias.
            None => bias_of(&result.creature, "output-0"),
        };
        assert!(
            close(folded, expected_const * 2.0),
            "{folded} vs {}",
            expected_const * 2.0
        );
        validate_creature(&result.creature).unwrap();
    }

    /// The migration's headline: what Ockham's own rewrite failed closed on,
    /// the shared engine builds. A typed `IF` source and the `IF` itself were
    /// `AblationSkip::TypedSynapse` / `AggregateTarget` before Issue #182.
    #[test]
    fn typed_and_aggregate_neurons_are_candidates_the_core_engine_builds() {
        let typed = typed_if_fixture();
        validate_creature(&typed).unwrap();

        let condition = prune_hidden_neuron(&typed, "h_cond", 0.5, None).unwrap();
        validate_creature(&condition.creature).unwrap();
        assert!(
            condition.creature.neurons.iter().all(|n| n.uuid != "h_cond"),
            "{:?}",
            uuids(&condition.creature)
        );

        let aggregate = prune_hidden_neuron(&typed, "h_if", 0.5, None).unwrap();
        validate_creature(&aggregate.creature).unwrap();
        assert!(aggregate.creature.neurons.iter().all(|n| n.uuid != "h_if"));
        assert_eq!(typed, typed_if_fixture(), "the source must not move");
    }

    #[test]
    fn an_aggregate_target_is_named_rather_than_folded() {
        let aggregate = ordinary_edge_into_aggregate();
        let result = prune_hidden_neuron(&aggregate, "h_src", 0.5, None).unwrap();
        assert!(
            !result.fully_compensated(),
            "a MEAN target cannot absorb a bias fold: {:?}",
            result.detail
        );
        let named = &result.detail.uncompensated[0];
        assert_eq!(named.target_uuid, "h_mean");
        assert_eq!(named.reason, "aggregate-target");
        assert_eq!(named.squash, "MEAN");
        validate_creature(&result.creature).unwrap();
    }

    #[test]
    fn a_protected_or_unknown_target_is_refused_with_no_creature() {
        let incumbent = hidden_identity_creature(0.0, 1.0);
        let unknown = prune_hidden_neuron(&incumbent, "nope", 0.0, None).unwrap_err();
        assert_eq!(unknown.blocked_reason(), BlockedReason::UnsafeTopology);
        let output = prune_hidden_neuron(&incumbent, "output-0", 0.0, None).unwrap_err();
        assert_eq!(output.blocked_reason(), BlockedReason::UnsafeTopology);
        assert!(output.to_string().contains("protected"), "{output}");
        let nan = prune_hidden_neuron(&incumbent, "h1", f64::NAN, None).unwrap_err();
        assert_eq!(nan.blocked_reason(), BlockedReason::MissingActivation);
        let infinite = prune_hidden_neuron(&incumbent, "h1", f64::INFINITY, None).unwrap_err();
        assert_eq!(infinite.blocked_reason(), BlockedReason::MissingActivation);
    }

    #[test]
    fn a_measured_variance_is_reported_as_the_residual_it_leaves() {
        let incumbent = two_hidden();
        let stats = NeuronStats {
            uuid: "h_a".into(),
            neuron_index: 0,
            count: 64,
            mean: 2.0,
            variance: 0.25,
            std_dev: 0.5,
            mean_abs: 2.0,
            min: 1.0,
            max: 3.0,
        };
        let result = prune_hidden_neuron(&incumbent, "h_a", 2.0, Some(&stats)).unwrap();
        let fold = &result.detail.bias_folds[0];
        assert_eq!(
            fold.residual_variance,
            Some(3.0 * 3.0 * 0.25),
            "the variance the fold could not carry is w^2 * the measured variance"
        );
        // A single record is no measurement of spread, so nothing is offered.
        let one_record = NeuronStats { count: 1, ..stats };
        let result = prune_hidden_neuron(&incumbent, "h_a", 2.0, Some(&one_record)).unwrap();
        assert_eq!(result.detail.bias_folds[0].residual_variance, None);
    }

    fn group(uuids: &[&str], mean: f64) -> Vec<GroupMember> {
        uuids
            .iter()
            .map(|u| GroupMember {
                uuid: (*u).to_string(),
                mean,
            })
            .collect()
    }

    #[test]
    fn a_group_cut_removes_every_member_and_its_cascade() {
        let incumbent = chain_plus_keep();
        let original = incumbent.clone();
        let result = prune_hidden_group(&incumbent, &group(&["h_up", "h_leaf"], 1.0)).unwrap();
        assert_eq!(incumbent, original, "incumbent must be untouched");
        // Removing `h_up` leaves `h_leaf` with nothing to sum, so the cleanup
        // resolves it to constant support before the group reaches it: the
        // member is accounted for as a fold, not requested from an engine that
        // protects constants from direct removal.
        assert_eq!(result.detail.requested_neurons, vec!["h_up"]);
        assert_eq!(result.detail.folded_neurons, vec!["h_leaf"]);
        assert_eq!(result.after.hidden_neurons, 1, "only `h_keep` is left");
        assert!(result.after.growth_units < result.before.growth_units);
        validate_creature(&result.creature).unwrap();
    }

    #[test]
    fn a_group_cut_distinguishes_primary_cuts_from_cleanup_cascade() {
        // Cutting the leaf alone already strands `h_up`; asking for the leaf
        // and the keeper leaves `h_up` to the cascade, so the record must say
        // which two the razor chose and which one that choice stranded.
        let incumbent = chain_plus_keep();
        let result = prune_hidden_group(&incumbent, &group(&["h_leaf", "h_keep"], 0.5)).unwrap();
        assert_eq!(result.detail.requested_neurons, vec!["h_leaf", "h_keep"]);
        assert_eq!(result.detail.cascade_neurons, vec!["h_up"]);
        assert_eq!(result.after.hidden_neurons, 0);
        validate_creature(&result.creature).unwrap();
    }

    #[test]
    fn a_group_cut_folds_each_member_mean_into_what_survives_it() {
        // Cutting both hidden neurons folds 2.0 * 3.0 and 0.5 * 1.0 into the
        // output bias, which starts at 0.25.
        let incumbent = two_hidden();
        let members = vec![
            GroupMember {
                uuid: "h_a".into(),
                mean: 2.0,
            },
            GroupMember {
                uuid: "h_b".into(),
                mean: 0.5,
            },
        ];
        let result = prune_hidden_group(&incumbent, &members).unwrap();
        assert_eq!(result.detail.transform_class, TransformClass::Approximate);
        let bias = bias_of(&result.creature, "output-0");
        assert!(close(bias, 0.25 + 2.0 * 3.0 + 0.5 * 1.0), "bias {bias}");
        assert_eq!(result.detail.bias_folds.len(), 2);
        validate_creature(&result.creature).unwrap();
    }

    #[test]
    fn a_repeated_member_is_folded_once() {
        let incumbent = two_hidden();
        let once = prune_hidden_group(&incumbent, &group(&["h_a"], 2.0)).unwrap();
        let twice = prune_hidden_group(&incumbent, &group(&["h_a", "h_a"], 2.0)).unwrap();
        assert_eq!(twice.detail.requested_neurons, vec!["h_a"]);
        assert_eq!(twice.detail.bias_folds, once.detail.bias_folds);
        assert_eq!(twice.creature, once.creature);
    }

    #[test]
    fn a_member_an_earlier_cascade_took_is_not_requested_twice() {
        // Cutting `h_leaf` strands `h_up`, so by the time the group reaches
        // `h_up` it is already gone — recorded as the cascade neuron it was,
        // never re-requested from an engine that would refuse it.
        let incumbent = chain_plus_keep();
        let result = prune_hidden_group(&incumbent, &group(&["h_leaf", "h_up"], 1.0)).unwrap();
        assert_eq!(result.detail.requested_neurons, vec!["h_leaf"]);
        assert_eq!(result.detail.cascade_neurons, vec!["h_up"]);
        assert_eq!(uuids(&result.creature), vec!["h_keep", "output-0"]);
        validate_creature(&result.creature).unwrap();
    }

    #[test]
    fn a_group_cut_that_disconnects_every_output_folds_it_to_a_constant() {
        // `h_a` and `h_b` are the only paths to the output. Cutting both is
        // still a *buildable* candidate — the output keeps both folded means as
        // its bias and stops depending on the input — and it is emitted rather
        // than second-guessed. A creature that ignores its input scores badly,
        // and it is the scorer that says so, never the razor.
        let incumbent = two_hidden();
        let result = prune_hidden_group(&incumbent, &group(&["h_a", "h_b"], 2.0)).unwrap();
        assert_eq!(result.after.hidden_neurons, 0);
        assert_eq!(result.after.synapses, 0);
        let bias = bias_of(&result.creature, "output-0");
        assert!(close(bias, 0.25 + 2.0 * 3.0 + 2.0 * 1.0), "bias {bias}");
        validate_creature(&result.creature).unwrap();
    }

    #[test]
    fn an_unbuildable_member_blocks_the_whole_group() {
        let incumbent = chain_plus_keep();
        for members in [
            group(&["h_up", "nope"], 1.0),
            group(&["h_up", "output-0"], 1.0),
            group(&["h_up"], f64::NAN),
        ] {
            let err = prune_hidden_group(&incumbent, &members).unwrap_err();
            assert!(
                matches!(
                    err.blocked_reason(),
                    BlockedReason::UnsafeTopology | BlockedReason::MissingActivation
                ),
                "{err}"
            );
        }
        let err = prune_hidden_group(&incumbent, &[]).unwrap_err();
        assert_eq!(err.blocked_reason(), BlockedReason::UnsafeTopology);
        assert_eq!(
            incumbent,
            chain_plus_keep(),
            "a refusal must not mutate the source"
        );
    }

    #[test]
    fn a_single_member_group_matches_the_single_neuron_prune() {
        let incumbent = chain_plus_keep();
        let single = prune_hidden_neuron(&incumbent, "h_leaf", 1.0, None).unwrap();
        let grouped = prune_hidden_group(&incumbent, &group(&["h_leaf"], 1.0)).unwrap();
        assert_eq!(grouped.creature, single.creature);
        assert_eq!(grouped.detail, single.detail);
    }

    #[test]
    fn folding_one_edge_leaves_the_mean_output_unchanged() {
        let incumbent = shared_output_fan_out();
        validate_creature(&incumbent).unwrap();
        let original = incumbent.clone();
        // `h_a` is IDENTITY(2 * x), so its mean over XS is exactly 1.0.
        let result = prune_edge(&incumbent, "h_a", "output-0", 1.0, None).unwrap();
        assert_eq!(incumbent, original, "incumbent must be untouched");
        assert_eq!(result.detail.transform_class, TransformClass::Approximate);

        let before = mean_output(&incumbent, &XS, 0);
        let after = mean_output(&result.creature, &XS, 0);
        assert!(
            (before - after).abs() <= 1e-9,
            "mean output moved: {before} vs {after}"
        );
        let bias = bias_of(&result.creature, "output-0");
        assert!(close(bias, 0.25 + 1.0 * 3.0), "bias {bias}");
        assert_eq!(result.detail.bias_folds.len(), 1);
        assert!(close(result.detail.bias_folds[0].weight_sum, 3.0));
        assert!(close(
            removed_weight(&incumbent, &result.detail.removed_synapses),
            3.0
        ));
        // `output-1` never saw the transform.
        assert!(close(
            mean_output(&incumbent, &XS, 1),
            mean_output(&result.creature, &XS, 1)
        ));
        validate_creature(&result.creature).unwrap();
    }

    #[test]
    fn a_pure_edge_cut_costs_one_tenth_of_a_growth_unit() {
        let incumbent = shared_output_fan_out();
        let result = prune_edge(&incumbent, "h_a", "output-0", 1.0, None).unwrap();
        assert!(
            result.detail.cascade_neurons.is_empty(),
            "nothing was stranded: {:?}",
            result.detail
        );
        assert_eq!(result.after.hidden_neurons, result.before.hidden_neurons);
        assert_eq!(result.after.synapses, result.before.synapses - 1);
        assert!(
            close(result.after.growth_units, result.before.growth_units - 0.1),
            "{} → {}",
            result.before.growth_units,
            result.after.growth_units
        );
        assert!(
            !result
                .creature
                .synapses
                .iter()
                .any(|s| s.from_uuid == "h_a" && s.to_uuid == "output-0")
        );
    }

    #[test]
    fn cutting_the_last_outgoing_edge_cascades_the_chain() {
        // input → h_up → h_leaf → output; cutting the leaf's only outgoing edge
        // strands the leaf, and stranding the leaf strands `h_up` behind it.
        let incumbent = chain_plus_keep();
        let result = prune_edge(&incumbent, "h_leaf", "output-0", 1.0, None).unwrap();
        assert_eq!(result.detail.cascade_neurons, vec!["h_leaf", "h_up"]);
        assert_eq!(uuids(&result.creature), vec!["h_keep", "output-0"]);
        let bias = bias_of(&result.creature, "output-0");
        assert!(close(bias, 1.0 * 2.0), "bias {bias}");
        validate_creature(&result.creature).unwrap();
    }

    #[test]
    fn cutting_a_constants_only_edge_cascades_the_constant() {
        let incumbent = constant_leaf();
        validate_creature(&incumbent).unwrap();
        // The source is a constant, so its value is the creature's to prove:
        // the hint is ignored and the fold is exact.
        let result = prune_edge(&incumbent, "c0", "output-0", 99.0, None).unwrap();
        assert_eq!(result.detail.transform_class, TransformClass::Exact);
        assert_eq!(result.detail.cascade_neurons, vec!["c0"]);
        assert_eq!(result.after.constant_neurons, 0);
        let bias = bias_of(&result.creature, "output-0");
        assert!(close(bias, 0.5 * 2.0), "bias {bias}");
        validate_creature(&result.creature).unwrap();
    }

    /// The second half of the migration's headline, for edges: a typed `IF`
    /// role and an edge out of an observation were both refused outright
    /// before Issue #182, and both are ordinary candidates in core.
    #[test]
    fn typed_roles_and_input_edges_are_candidates_the_core_engine_builds() {
        let typed = typed_if_fixture();
        let condition = prune_edge(&typed, "h_cond", "h_if", 0.5, None).unwrap();
        validate_creature(&condition.creature).unwrap();
        assert_eq!(condition.detail.removed_synapses[0].role, "condition");
        assert!(
            !condition.detail.static_if_neurons.is_empty(),
            "losing the condition makes the IF statically decidable: {:?}",
            condition.detail
        );

        let incumbent = shared_output_fan_out();
        let from_input = prune_edge(&incumbent, "input-0", "h_a", 0.5, None).unwrap();
        validate_creature(&from_input.creature).unwrap();
        assert_eq!(from_input.detail.removed_synapses[0].from_uuid, "input-0");
        assert_eq!(typed, typed_if_fixture(), "the source must not move");
    }

    #[test]
    fn an_edge_into_an_aggregate_target_is_named_rather_than_folded() {
        let aggregate = ordinary_edge_into_aggregate();
        let result = prune_edge(&aggregate, "h_src", "h_mean", 0.5, None).unwrap();
        assert!(!result.fully_compensated(), "{:?}", result.detail);
        assert_eq!(result.detail.uncompensated[0].target_uuid, "h_mean");
        validate_creature(&result.creature).unwrap();
    }

    #[test]
    fn unknown_endpoints_edges_and_source_values_are_refused() {
        let incumbent = shared_output_fan_out();
        for (from, to, value) in [
            ("h_a", "output-0", f64::NAN),
            ("h_a", "output-0", f64::INFINITY),
        ] {
            let err = prune_edge(&incumbent, from, to, value, None).unwrap_err();
            assert_eq!(err.blocked_reason(), BlockedReason::MissingActivation);
        }
        for (from, to) in [("nope", "output-0"), ("h_a", "nope"), ("h_a", "h_b")] {
            let err = prune_edge(&incumbent, from, to, 1.0, None).unwrap_err();
            assert_eq!(err.blocked_reason(), BlockedReason::UnsafeTopology, "{err}");
        }
        assert_eq!(
            incumbent,
            shared_output_fan_out(),
            "a refusal must not mutate the source"
        );
    }

    #[test]
    fn an_output_left_with_no_incoming_edge_is_still_a_candidate() {
        // NEAT-AI-core has no rule that an output must be fed: the cut leaves
        // `output-0` on its folded bias alone and validation accepts it, so the
        // candidate is emitted and the full-corpus scorer — never the razor —
        // decides whether an input-blind output is worse.
        let incumbent = hidden_identity_creature(0.0, 1.0);
        let result = prune_edge(&incumbent, "h1", "output-0", 0.75, None).unwrap();
        assert_eq!(result.after.synapses, 0);
        assert_eq!(result.after.hidden_neurons, 0);
        assert_eq!(result.detail.cascade_neurons, vec!["h1"]);
        let bias = bias_of(&result.creature, "output-0");
        assert!(close(bias, 0.75), "bias {bias}");
        validate_creature(&result.creature).unwrap();
    }

    #[test]
    fn an_invalid_incumbent_yields_no_candidate_at_all() {
        // `c0` is a constant with an inward edge, which NEAT-AI-core rule 14
        // forbids. The cleanup cannot repair it into a valid canonical form, so
        // the request fails and nothing reaches the scorer.
        let invalid = creature(
            1,
            1,
            vec![
                neuron("constant", "c0", 0.5, None),
                neuron("hidden", "h_keep", 0.0, Some("IDENTITY")),
                neuron("output", "output-0", 0.0, Some("IDENTITY")),
            ],
            vec![
                synapse("input-0", "c0", 1.0),
                synapse("c0", "output-0", 2.0),
                synapse("input-0", "h_keep", 1.0),
                synapse("h_keep", "output-0", 1.0),
            ],
        );
        assert!(
            validate_creature(&invalid).is_err(),
            "fixture must be invalid"
        );
        let err = prune_edge(&invalid, "h_keep", "output-0", 1.0, None).unwrap_err();
        assert_eq!(err.blocked_reason(), BlockedReason::ValidationFailed);
    }

    #[test]
    fn the_detail_round_trips_through_json() {
        let incumbent = chain_plus_keep();
        let result = prune_hidden_neuron(&incumbent, "h_leaf", 1.0, None).unwrap();
        let json = serde_json::to_string(&result.detail).unwrap();
        let back: PruneDetail = serde_json::from_str(&json).unwrap();
        assert_eq!(back, result.detail);
        assert!(json.contains("\"cascadeNeurons\""), "{json}");
        assert!(
            !json.contains("\"weightShares\""),
            "an empty list is not written: {json}"
        );
    }
}
