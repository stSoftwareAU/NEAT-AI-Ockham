//! Mean-activation neuron ablation with recursive exact cleanup (Issue #4).
//!
//! All work happens on a **clone** of the incumbent. The requested removal
//! replaces a hidden neuron's downstream contribution with its measured mean:
//!
//! ```text
//! bias_j' = bias_j + mean_i * w_ij
//! ```
//!
//! That step is deliberately approximate. The recursive cleanup that follows
//! (dead hidden/constant neurons, constant folding of known squashes) is exact.
//! Neither distinction grants acceptance: the full-corpus scorer still decides.
//!
//! Unsupported aggregate/typed-synapse cases are skipped, never guessed. The
//! final candidate must pass NEAT-AI-core `creature.validate()`.

use std::collections::HashSet;
use std::fmt;

use neat_core::{
    CreatureExport, NeuronExport, SquashType, SynapseExport, apply_squash, parse_squash_name,
};
use serde::Serialize;

use crate::blocked::BlockedReason;
use crate::fixtures::sort_synapses_canonically;
use crate::incumbent::validate_creature;
use crate::stats::NeuronStats;

/// NEAT structural growth units: `hidden + synapses / 10`.
///
/// The scorer multiplies this by its `growthCost`; Ockham records the unitless
/// quantity so deltas are comparable without copying the scorer's cost knob.
pub fn growth_units(hidden_neurons: usize, synapses: usize) -> f64 {
    hidden_neurons as f64 + synapses as f64 / 10.0
}

/// Neuron/synapse counts and the unitless growth-cost proxy.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct StructureSnapshot {
    /// Hidden neuron count.
    pub hidden_neurons: usize,
    /// Constant neuron count.
    pub constant_neurons: usize,
    /// Synapse count.
    pub synapses: usize,
    /// [`growth_units`].
    pub growth_units: f64,
}

impl StructureSnapshot {
    /// Snapshot of `creature`.
    pub fn of(creature: &CreatureExport) -> Self {
        let hidden_neurons = creature
            .neurons
            .iter()
            .filter(|n| n.neuron_type == "hidden")
            .count();
        let constant_neurons = creature
            .neurons
            .iter()
            .filter(|n| n.neuron_type == "constant")
            .count();
        Self {
            hidden_neurons,
            constant_neurons,
            synapses: creature.synapses.len(),
            growth_units: growth_units(hidden_neurons, creature.synapses.len()),
        }
    }
}

/// Whether the candidate used the approximate mean substitution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub enum TransformClass {
    /// At least one downstream bias used `mean_i * w_ij`.
    Approximate,
    /// Only exact dead-structure / constant-fold cleanup ran.
    Exact,
}

/// One downstream bias update.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct BiasCompensation {
    /// Destination neuron UUID.
    pub to_uuid: String,
    /// Synapse weight used.
    pub weight: f64,
    /// Scalar multiplied by `weight` (`mean_i` or a folded constant).
    pub source_value: f64,
    /// Bias before the update.
    pub bias_before: f64,
    /// Bias after the update.
    pub bias_after: f64,
    /// `mean` for the requested ablation; `constant` for cleanup folds.
    pub kind: &'static str,
}

/// A neuron removed during the atomic transform.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RemovedNeuron {
    /// Neuron UUID.
    pub uuid: String,
    /// `hidden` or `constant`.
    pub neuron_type: String,
    /// Why it was removed.
    pub reason: &'static str,
}

/// Why a requested ablation was not emitted.
#[derive(Debug, Clone, PartialEq)]
pub enum AblationSkip {
    /// No listed neuron has this UUID.
    UnknownNeuron(String),
    /// Only hidden neurons are ablation targets.
    NotHidden {
        /// Requested UUID.
        uuid: String,
        /// Declared type.
        neuron_type: String,
    },
    /// Mean was NaN/Inf.
    NonFiniteMean(f64),
    /// Requested neuron uses an aggregate squash.
    AggregateNeuron {
        /// Neuron UUID.
        uuid: String,
        /// Squash name.
        squash: String,
    },
    /// A synapse incident to the transform is typed.
    TypedSynapse {
        /// Source UUID.
        from_uuid: String,
        /// Destination UUID.
        to_uuid: String,
        /// Role name.
        synapse_type: String,
    },
    /// Downstream target is an aggregate squash (bias fold is not a sum).
    AggregateTarget {
        /// Target UUID.
        uuid: String,
        /// Squash name.
        squash: String,
    },
    /// Hidden with no incoming synapses has an unknown squash.
    UnknownSquash {
        /// Neuron UUID.
        uuid: String,
        /// Declared squash (or empty).
        squash: String,
    },
    /// Final candidate failed `creature.validate()`.
    Invalid(String),
    /// A group cut was requested with no members (Issue #108).
    ///
    /// Named rather than silently returning the incumbent: a candidate that
    /// removes nothing would be scored, tie the baseline and be reported as a
    /// proposal the razor tried.
    EmptyGroup,
}

impl AblationSkip {
    /// The reason code this skip is counted under (Issue #103).
    ///
    /// Structured, never parsed back out of the message: the tally has to be
    /// deterministic, and a reason names the neuron it is about.
    pub fn blocked_reason(&self) -> BlockedReason {
        match self {
            Self::AggregateNeuron { .. }
            | Self::AggregateTarget { .. }
            | Self::UnknownSquash { .. } => BlockedReason::AggregateSquash,
            Self::NonFiniteMean(_) => BlockedReason::MissingActivation,
            Self::UnknownNeuron(_)
            | Self::NotHidden { .. }
            | Self::TypedSynapse { .. }
            | Self::EmptyGroup => BlockedReason::UnsafeTopology,
            Self::Invalid(_) => BlockedReason::ValidationFailed,
        }
    }

    /// Whether a constant substitution is worth trying instead (Issue #103).
    ///
    /// True for the structural blocks — an aggregate target, an aggregate
    /// source, a typed edge — where the fold is impossible but the *edge* can
    /// be preserved. False where there was nothing to substitute in the first
    /// place (no neuron, no finite mean) or where a candidate was built and
    /// rejected.
    pub fn substitution_may_help(&self) -> bool {
        matches!(
            self,
            Self::AggregateNeuron { .. }
                | Self::AggregateTarget { .. }
                | Self::UnknownSquash { .. }
                | Self::TypedSynapse { .. }
        )
    }
}

impl fmt::Display for AblationSkip {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownNeuron(u) => write!(f, "no neuron `{u}`"),
            Self::NotHidden { uuid, neuron_type } => {
                write!(f, "`{uuid}` is {neuron_type}, not hidden")
            }
            Self::NonFiniteMean(m) => write!(f, "non-finite mean {m}"),
            Self::AggregateNeuron { uuid, squash } => {
                write!(f, "`{uuid}` squash `{squash}` is aggregate; skipped")
            }
            Self::TypedSynapse {
                from_uuid,
                to_uuid,
                synapse_type,
            } => write!(
                f,
                "typed synapse `{from_uuid}`→`{to_uuid}` ({synapse_type}); skipped"
            ),
            Self::AggregateTarget { uuid, squash } => {
                write!(f, "aggregate target `{uuid}` (`{squash}`); skipped")
            }
            Self::UnknownSquash { uuid, squash } => {
                write!(f, "`{uuid}` squash `{squash}` is unknown; skipped")
            }
            Self::Invalid(m) => write!(f, "candidate failed creature.validate(): {m}"),
            Self::EmptyGroup => write!(f, "group cut requested with no members"),
        }
    }
}

/// Record of one emitted ablation candidate.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Ablation {
    /// Requested hidden-neuron UUID.
    pub requested_uuid: String,
    /// Requested neuron index in the incumbent.
    pub requested_index: usize,
    /// Mean used for the approximate substitution.
    pub mean: f64,
    /// Copy of the activation stats that proposed this removal, when supplied.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub activation: Option<NeuronStats>,
    /// Approximate vs exact.
    pub transform_class: TransformClass,
    /// Downstream bias updates, in application order.
    pub compensations: Vec<BiasCompensation>,
    /// Neurons removed (requested first, then cascade).
    pub removed_neurons: Vec<RemovedNeuron>,
    /// Structure before the transform.
    pub before: StructureSnapshot,
    /// Structure after the transform.
    pub after: StructureSnapshot,
    /// Validated candidate creature.
    #[serde(skip)]
    pub creature: CreatureExport,
}

/// Ablate hidden neuron `uuid` from a clone of `incumbent`.
///
/// `mean` is the full-corpus mean post-activation of that neuron. `stats` is
/// recorded on the candidate when present; it is not consulted for the arithmetic.
pub fn ablate_mean(
    incumbent: &CreatureExport,
    uuid: &str,
    mean: f64,
    stats: Option<&NeuronStats>,
) -> Result<Ablation, AblationSkip> {
    if !mean.is_finite() {
        return Err(AblationSkip::NonFiniteMean(mean));
    }
    let requested_index = require_ablatable_hidden(incumbent, uuid)?;

    // Rejection is decided on the incumbent, before anything is copied
    // (Issue #91). Most hidden neurons of a GRQ forest feed an aggregate
    // squash, so most visits end here: cloning a 7,000-neuron creature first
    // and throwing it away was the sweep paying full price for every neuron it
    // could never prune.
    reject_unfoldable_edges(incumbent, uuid)?;

    let mut working = incumbent.clone();
    working.memetic = None;
    let before = StructureSnapshot::of(&working);

    let mut compensations = Vec::new();
    let mut removed_neurons = Vec::new();
    let used_mean = fold_and_remove(
        &mut working,
        uuid,
        mean,
        &mut compensations,
        &mut removed_neurons,
    )?;

    cleanup_cascade(&mut working, &mut compensations, &mut removed_neurons)?;
    sort_synapses_canonically(&mut working);

    validate_creature(&working).map_err(|e| AblationSkip::Invalid(e.to_string()))?;

    let after = StructureSnapshot::of(&working);
    Ok(Ablation {
        requested_uuid: uuid.to_string(),
        requested_index,
        mean,
        activation: stats.cloned(),
        transform_class: if used_mean {
            TransformClass::Approximate
        } else {
            TransformClass::Exact
        },
        compensations,
        removed_neurons,
        before,
        after,
        creature: working,
    })
}

/// One member of a group cut: a hidden neuron and the mean that replaces it.
#[derive(Debug, Clone, PartialEq)]
pub struct GroupMember {
    /// Hidden neuron UUID.
    pub uuid: String,
    /// Full-corpus mean post-activation of that neuron.
    pub mean: f64,
}

/// Record of one emitted group ablation candidate (Issue #108).
///
/// The same transform as [`ablate_mean`], applied to several hidden neurons on
/// one clone of the incumbent before the exact cleanup runs once over the
/// result. Every neuron the transform removed is listed in
/// [`Self::removed_neurons`], and each entry says whether it was a requested
/// group cut or structure the cleanup cascade stranded.
#[derive(Debug, Clone, PartialEq)]
pub struct GroupAblation {
    /// Requested hidden-neuron UUIDs, in the order they were removed.
    pub requested_uuids: Vec<String>,
    /// Approximate vs exact.
    pub transform_class: TransformClass,
    /// Downstream bias updates, in application order.
    pub compensations: Vec<BiasCompensation>,
    /// Neurons removed (the requested group first, then the cascade).
    pub removed_neurons: Vec<RemovedNeuron>,
    /// Structure before the transform.
    pub before: StructureSnapshot,
    /// Structure after the transform.
    pub after: StructureSnapshot,
    /// Validated candidate creature.
    pub creature: CreatureExport,
}

impl GroupAblation {
    /// UUIDs the cleanup cascade removed on top of the requested group.
    ///
    /// The two are recorded apart because they answer different questions: the
    /// group is what the razor chose to cut, the cascade is what that choice
    /// stranded. A learning that conflated them could not reconstruct the
    /// proposal it came from.
    pub fn cascade_uuids(&self) -> Vec<String> {
        self.removed_neurons
            .iter()
            .filter(|n| n.reason != "requested")
            .map(|n| n.uuid.clone())
            .collect()
    }
}

/// Ablate a whole group of hidden neurons from one clone of `incumbent` (#108).
///
/// Some structure is only removable as a group: a chain or a low-fan-out branch
/// can be collectively redundant while each neuron on its own is a poor
/// approximation. Members are folded and removed in the order given — each with
/// the same `bias_j += mean_i * w_ij` substitution [`ablate_mean`] applies — and
/// the exact cleanup cascade then runs once over the result.
///
/// Fails closed exactly where the single-neuron transform does: an unknown or
/// non-hidden member, a non-finite mean, an aggregate squash, an aggregate fold
/// target, a typed edge, or a candidate `creature.validate()` rejects. Repeated
/// UUIDs are folded once. An empty group is [`AblationSkip::EmptyGroup`] rather
/// than a candidate identical to the incumbent.
///
/// Being buildable is not being good: a group candidate still faces the sampled
/// screen and the full-corpus scorer, which alone accepts a cut.
pub fn ablate_group(
    incumbent: &CreatureExport,
    members: &[GroupMember],
) -> Result<GroupAblation, AblationSkip> {
    // Deduplicated first so a repeated uuid cannot fold the same mean twice.
    let mut requested: Vec<&GroupMember> = Vec::with_capacity(members.len());
    let mut seen = HashSet::new();
    for member in members {
        if seen.insert(member.uuid.as_str()) {
            requested.push(member);
        }
    }
    if requested.is_empty() {
        return Err(AblationSkip::EmptyGroup);
    }
    // Every member is rejected on the incumbent before anything is copied, so a
    // group the razor could never build costs no clone (Issue #91).
    for member in &requested {
        if !member.mean.is_finite() {
            return Err(AblationSkip::NonFiniteMean(member.mean));
        }
        require_ablatable_hidden(incumbent, &member.uuid)?;
        reject_unfoldable_edges(incumbent, &member.uuid)?;
    }

    let mut working = incumbent.clone();
    working.memetic = None;
    let before = StructureSnapshot::of(&working);

    let mut compensations = Vec::new();
    let mut removed_neurons = Vec::new();
    let mut used_mean = false;
    for member in &requested {
        used_mean |= fold_and_remove(
            &mut working,
            &member.uuid,
            member.mean,
            &mut compensations,
            &mut removed_neurons,
        )?;
    }

    cleanup_cascade(&mut working, &mut compensations, &mut removed_neurons)?;
    sort_synapses_canonically(&mut working);

    validate_creature(&working).map_err(|e| AblationSkip::Invalid(e.to_string()))?;

    let after = StructureSnapshot::of(&working);
    Ok(GroupAblation {
        requested_uuids: requested.iter().map(|m| m.uuid.clone()).collect(),
        transform_class: if used_mean {
            TransformClass::Approximate
        } else {
            TransformClass::Exact
        },
        compensations,
        removed_neurons,
        before,
        after,
        creature: working,
    })
}

/// Index of `uuid` in `creature`, once it is a hidden neuron the razor may fold.
fn require_ablatable_hidden(creature: &CreatureExport, uuid: &str) -> Result<usize, AblationSkip> {
    let index = creature
        .neurons
        .iter()
        .position(|n| n.uuid == uuid)
        .ok_or_else(|| AblationSkip::UnknownNeuron(uuid.to_string()))?;
    let requested = &creature.neurons[index];
    if requested.neuron_type != "hidden" {
        return Err(AblationSkip::NotHidden {
            uuid: uuid.to_string(),
            neuron_type: requested.neuron_type.clone(),
        });
    }
    if squash_of(requested)?.is_aggregate() {
        return Err(AblationSkip::AggregateNeuron {
            uuid: uuid.to_string(),
            squash: requested
                .squash
                .clone()
                .unwrap_or_else(|| "IDENTITY".into()),
        });
    }
    Ok(index)
}

/// Reject the edges around `uuid` a bias fold cannot express.
///
/// A typed edge carries a role and an aggregate target is not a sum of its
/// inputs, so neither can absorb a mean. Checked on the creature the candidate
/// would be built from, before it is cloned.
fn reject_unfoldable_edges(creature: &CreatureExport, uuid: &str) -> Result<(), AblationSkip> {
    for syn in &synapses_from(creature, uuid) {
        require_ordinary(syn)?;
        let target = neuron_by_uuid(creature, &syn.to_uuid)
            .ok_or_else(|| AblationSkip::UnknownNeuron(syn.to_uuid.clone()))?;
        reject_aggregate_neuron(target)?;
    }
    for syn in &synapses_to(creature, uuid) {
        require_ordinary(syn)?;
    }
    Ok(())
}

/// Fold `mean` into every downstream bias of `uuid` on `working`, then remove it.
///
/// Returns whether a mean was actually folded — a neuron whose outgoing edges
/// have already gone with an earlier member of the same group folds nothing,
/// and a transform that folded no mean is exact.
fn fold_and_remove(
    working: &mut CreatureExport,
    uuid: &str,
    mean: f64,
    compensations: &mut Vec<BiasCompensation>,
    removed: &mut Vec<RemovedNeuron>,
) -> Result<bool, AblationSkip> {
    let mut used_mean = false;
    for syn in synapses_from(working, uuid) {
        apply_bias_fold(working, &syn, mean, "mean", compensations)?;
        used_mean = true;
    }
    removed.push(RemovedNeuron {
        uuid: uuid.to_string(),
        neuron_type: "hidden".into(),
        reason: "requested",
    });
    remove_neuron(working, uuid);
    Ok(used_mean)
}

pub(crate) fn cleanup_cascade(
    working: &mut CreatureExport,
    compensations: &mut Vec<BiasCompensation>,
    removed: &mut Vec<RemovedNeuron>,
) -> Result<(), AblationSkip> {
    loop {
        if let Some(uuid) = first_dead_non_output(working) {
            let neuron_type = neuron_by_uuid(working, &uuid)
                .map(|n| n.neuron_type.clone())
                .unwrap_or_else(|| "hidden".into());
            removed.push(RemovedNeuron {
                uuid: uuid.clone(),
                neuron_type,
                reason: "no-outgoing",
            });
            remove_neuron(working, &uuid);
            continue;
        }
        if let Some(uuid) = first_hidden_without_incoming(working) {
            let neuron = neuron_by_uuid(working, &uuid)
                .cloned()
                .ok_or_else(|| AblationSkip::UnknownNeuron(uuid.clone()))?;
            let squash = squash_of(&neuron)?;
            if squash.is_aggregate() {
                return Err(AblationSkip::AggregateNeuron {
                    uuid: uuid.clone(),
                    squash: neuron.squash.clone().unwrap_or_default(),
                });
            }
            let constant = f64::from(apply_squash(squash, neuron.bias as f32));
            if !constant.is_finite() {
                return Err(AblationSkip::UnknownSquash {
                    uuid: uuid.clone(),
                    squash: neuron.squash.clone().unwrap_or_default(),
                });
            }
            let outgoing = synapses_from(working, &uuid);
            for syn in &outgoing {
                require_ordinary(syn)?;
                let target = neuron_by_uuid(working, &syn.to_uuid)
                    .ok_or_else(|| AblationSkip::UnknownNeuron(syn.to_uuid.clone()))?;
                reject_aggregate_neuron(target)?;
            }
            for syn in outgoing {
                apply_bias_fold(working, &syn, constant, "constant", compensations)?;
            }
            removed.push(RemovedNeuron {
                uuid: uuid.clone(),
                neuron_type: "hidden".into(),
                reason: "no-incoming",
            });
            remove_neuron(working, &uuid);
            continue;
        }
        break;
    }
    Ok(())
}

/// UUIDs that are the source of at least one synapse.
///
/// One pass over the synapses, so the caller answers "has this neuron any
/// outgoing?" for every neuron at once. Asking per neuron instead costs a scan
/// of every synapse per neuron, and the cleanup loop asks on every iteration:
/// on a 7,000-neuron GRQ forest that quadratic shape was the sweep's dominant
/// cost, ~300ms per visited neuron (Issue #91).
fn synapse_sources(working: &CreatureExport) -> HashSet<&str> {
    working
        .synapses
        .iter()
        .map(|s| s.from_uuid.as_str())
        .collect()
}

/// UUIDs that are the destination of at least one synapse. See [`synapse_sources`].
fn synapse_targets(working: &CreatureExport) -> HashSet<&str> {
    working
        .synapses
        .iter()
        .map(|s| s.to_uuid.as_str())
        .collect()
}

/// The first non-output neuron feeding nothing, or `None`.
///
/// Shared with [`crate::substitute`]: a neuron with no outgoing synapse reaches
/// no output, so removing it changes no output value — and NEAT-AI-core rejects
/// one that stays (rules 16 and 18).
pub(crate) fn first_dead_non_output(working: &CreatureExport) -> Option<String> {
    let sources = synapse_sources(working);
    working
        .neurons
        .iter()
        .find(|n| n.neuron_type != "output" && !sources.contains(n.uuid.as_str()))
        .map(|n| n.uuid.clone())
}

fn first_hidden_without_incoming(working: &CreatureExport) -> Option<String> {
    let targets = synapse_targets(working);
    working
        .neurons
        .iter()
        .find(|n| n.neuron_type == "hidden" && !targets.contains(n.uuid.as_str()))
        .map(|n| n.uuid.clone())
}

fn apply_bias_fold(
    working: &mut CreatureExport,
    syn: &SynapseExport,
    source_value: f64,
    kind: &'static str,
    compensations: &mut Vec<BiasCompensation>,
) -> Result<(), AblationSkip> {
    let target = working
        .neurons
        .iter_mut()
        .find(|n| n.uuid == syn.to_uuid)
        .ok_or_else(|| AblationSkip::UnknownNeuron(syn.to_uuid.clone()))?;
    let bias_before = target.bias;
    target.bias += source_value * syn.weight;
    compensations.push(BiasCompensation {
        to_uuid: syn.to_uuid.clone(),
        weight: syn.weight,
        source_value,
        bias_before,
        bias_after: target.bias,
        kind,
    });
    Ok(())
}

/// Remove `uuid` and every synapse incident to it.
pub(crate) fn remove_neuron(working: &mut CreatureExport, uuid: &str) {
    working.neurons.retain(|n| n.uuid != uuid);
    working
        .synapses
        .retain(|s| s.from_uuid != uuid && s.to_uuid != uuid);
}

fn synapses_from(working: &CreatureExport, uuid: &str) -> Vec<SynapseExport> {
    working
        .synapses
        .iter()
        .filter(|s| s.from_uuid == uuid)
        .cloned()
        .collect()
}

fn synapses_to(working: &CreatureExport, uuid: &str) -> Vec<SynapseExport> {
    working
        .synapses
        .iter()
        .filter(|s| s.to_uuid == uuid)
        .cloned()
        .collect()
}

fn neuron_by_uuid<'a>(working: &'a CreatureExport, uuid: &str) -> Option<&'a NeuronExport> {
    working.neurons.iter().find(|n| n.uuid == uuid)
}

fn require_ordinary(syn: &SynapseExport) -> Result<(), AblationSkip> {
    match &syn.synapse_type {
        Some(ty) => Err(AblationSkip::TypedSynapse {
            from_uuid: syn.from_uuid.clone(),
            to_uuid: syn.to_uuid.clone(),
            synapse_type: ty.clone(),
        }),
        None => Ok(()),
    }
}

fn reject_aggregate_neuron(neuron: &NeuronExport) -> Result<(), AblationSkip> {
    let squash = squash_of(neuron)?;
    if squash.is_aggregate() {
        Err(AblationSkip::AggregateTarget {
            uuid: neuron.uuid.clone(),
            squash: neuron.squash.clone().unwrap_or_else(|| "IDENTITY".into()),
        })
    } else {
        Ok(())
    }
}

fn squash_of(neuron: &NeuronExport) -> Result<SquashType, AblationSkip> {
    let name = neuron.squash.as_deref().unwrap_or("IDENTITY");
    parse_squash_name(name).map_err(|_| AblationSkip::UnknownSquash {
        uuid: neuron.uuid.clone(),
        squash: name.to_string(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixtures::{creature, hidden_identity_creature, neuron, synapse, typed_synapse};
    use crate::incumbent::validate_creature;
    use neat_core::compile_creature;

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() <= 1e-12 * a.abs().max(b.abs()).max(1.0)
    }

    /// Seconds to propose the same ablation `ROUNDS` times on a `hidden`-wide creature.
    fn ablation_seconds(hidden: usize) -> f64 {
        const ROUNDS: usize = 3;
        let wide = crate::fixtures::wide_creature(8, hidden, "TANH");
        let started = std::time::Instant::now();
        for _ in 0..ROUNDS {
            ablate_mean(&wide, "h0", 0.25, None).expect("h0 feeds the output and is prunable");
        }
        started.elapsed().as_secs_f64()
    }

    /// Issue #91: the cleanup scans counted a neuron's synapses by walking
    /// every synapse, once per neuron, on every cleanup iteration — so one
    /// ablation cost the square of the creature's size. On a 7,000-neuron GRQ
    /// forest that was ~300ms per visited neuron, and a run screened two or
    /// three batches an hour instead of filling batch after batch.
    ///
    /// A ratio, never a wall-clock budget (the standards forbid those): the
    /// same work is timed at one size and four times that size on the same
    /// machine, so a slower machine slows both readings and the test still
    /// holds. Growing with the creature is ~4x; growing with its square is
    /// ~16x, and the unfixed scans measured 9.4x.
    ///
    /// The small reading is taken twice, on either side of the large one, and
    /// the larger of the two is used: load that arrives *during* the test would
    /// otherwise inflate only the second reading and fail a correct tree.
    #[test]
    fn one_ablation_costs_the_creature_not_its_square() {
        let before = ablation_seconds(400);
        let large = ablation_seconds(1_600);
        let after = ablation_seconds(400);
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

    fn outputs(creature: &CreatureExport, xs: &[f32]) -> Vec<f32> {
        let mut net = compile_creature(creature).unwrap();
        xs.iter().map(|&x| net.activate(&[x], 1)[0]).collect()
    }

    #[test]
    fn bias_compensation_matches_hand_arithmetic() {
        let incumbent = two_hidden();
        validate_creature(&incumbent).unwrap();
        let original = incumbent.clone();
        let result = ablate_mean(&incumbent, "h_a", 2.0, None).unwrap();
        assert_eq!(incumbent, original, "incumbent must be untouched");
        assert_eq!(result.transform_class, TransformClass::Approximate);
        let out = result
            .creature
            .neurons
            .iter()
            .find(|n| n.uuid == "output-0")
            .unwrap();
        assert!(close(out.bias, 0.25 + 2.0 * 3.0), "bias {}", out.bias);
        assert_eq!(result.compensations.len(), 1);
        assert!(close(result.compensations[0].bias_after, out.bias));
        assert!(result.creature.neurons.iter().all(|n| n.uuid != "h_a"));
        assert!(result.creature.neurons.iter().any(|n| n.uuid == "h_b"));
        validate_creature(&result.creature).unwrap();
    }

    #[test]
    fn dead_output_cascade_removes_newly_unreachable_chain() {
        let incumbent = chain_plus_keep();
        validate_creature(&incumbent).unwrap();
        let result = ablate_mean(&incumbent, "h_leaf", 1.0, None).unwrap();
        let uuids: Vec<&str> = result
            .creature
            .neurons
            .iter()
            .map(|n| n.uuid.as_str())
            .collect();
        assert!(!uuids.contains(&"h_leaf"));
        assert!(
            !uuids.contains(&"h_up"),
            "h_up lost its only outgoing and must cascade: {uuids:?}"
        );
        assert!(uuids.contains(&"h_keep"));
        assert!(
            result
                .removed_neurons
                .iter()
                .any(|n| n.uuid == "h_up" && n.reason == "no-outgoing")
        );
        let out = result
            .creature
            .neurons
            .iter()
            .find(|n| n.uuid == "output-0")
            .unwrap();
        assert!(close(out.bias, 1.0 * 2.0), "bias {}", out.bias);
        validate_creature(&result.creature).unwrap();
    }

    #[test]
    fn constant_neuron_fixture_folds_to_matching_outputs() {
        let incumbent = constant_fold_fixture();
        validate_creature(&incumbent).unwrap();
        let xs = [0.0f32, 1.0, -2.5, 3.25];
        let before = outputs(&incumbent, &xs);
        let result = ablate_mean(&incumbent, "h_src", 1.5, None).unwrap();
        assert!(
            result
                .removed_neurons
                .iter()
                .any(|n| n.uuid == "h_mid" && n.reason == "no-incoming"),
            "{:?}",
            result.removed_neurons
        );
        let after = outputs(&result.creature, &xs);
        for (x, (a, b)) in xs.iter().zip(before.iter().zip(after.iter())) {
            assert!(
                (a - b).abs() <= 1e-5 * a.abs().max(b.abs()).max(1.0),
                "x={x}: {a} vs {b}"
            );
        }
        let expected_const = f64::from(apply_squash(SquashType::Tanh, 1.5));
        let out = result
            .creature
            .neurons
            .iter()
            .find(|n| n.uuid == "output-0")
            .unwrap();
        assert!(
            close(out.bias, expected_const * 2.0),
            "bias {} vs {}",
            out.bias,
            expected_const * 2.0
        );
        validate_creature(&result.creature).unwrap();
    }

    #[test]
    fn typed_and_aggregate_cases_fail_closed() {
        let typed = typed_if_fixture();
        let err = ablate_mean(&typed, "h_cond", 0.0, None).unwrap_err();
        assert!(matches!(err, AblationSkip::TypedSynapse { .. }), "{err}");
        let err = ablate_mean(&typed, "h_if", 0.0, None).unwrap_err();
        assert!(
            matches!(
                err,
                AblationSkip::AggregateTarget { .. } | AblationSkip::AggregateNeuron { .. }
            ),
            "{err}"
        );
        assert_eq!(typed, typed_if_fixture(), "skip must not mutate the source");
    }

    #[test]
    fn emitted_candidate_validates_and_incumbent_is_unchanged() {
        let incumbent = hidden_identity_creature(0.5, 2.0);
        let original = incumbent.clone();
        let result = ablate_mean(&incumbent, "h1", 0.5, None).unwrap();
        validate_creature(&result.creature).unwrap();
        assert_eq!(incumbent, original);
        assert!(result.creature.neurons.iter().all(|n| n.uuid != "h1"));
        assert_eq!(result.after.hidden_neurons, 0);
        assert!(result.after.growth_units < result.before.growth_units);
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
        let result = ablate_group(&incumbent, &group(&["h_up", "h_leaf"], 1.0)).unwrap();
        assert_eq!(incumbent, original, "incumbent must be untouched");
        assert_eq!(result.requested_uuids, vec!["h_up", "h_leaf"]);
        let left: Vec<&str> = result
            .creature
            .neurons
            .iter()
            .map(|n| n.uuid.as_str())
            .collect();
        assert_eq!(left, vec!["h_keep", "output-0"], "{left:?}");
        // Both members are primary cuts; nothing else was strandable here.
        let requested: Vec<&str> = result
            .removed_neurons
            .iter()
            .filter(|n| n.reason == "requested")
            .map(|n| n.uuid.as_str())
            .collect();
        assert_eq!(requested, vec!["h_up", "h_leaf"]);
        assert!(
            result.cascade_uuids().is_empty(),
            "{:?}",
            result.removed_neurons
        );
        assert!(result.after.growth_units < result.before.growth_units);
        validate_creature(&result.creature).unwrap();
    }

    #[test]
    fn a_group_cut_distinguishes_primary_cuts_from_cleanup_cascade() {
        // Cutting the leaf alone already strands `h_up`; asking for the leaf
        // and the keeper leaves `h_up` to the cascade, so the record must say
        // which two the razor chose and which one that choice stranded.
        let incumbent = chain_plus_keep();
        let result = ablate_group(&incumbent, &group(&["h_leaf", "h_keep"], 0.5)).unwrap();
        assert_eq!(result.requested_uuids, vec!["h_leaf", "h_keep"]);
        assert_eq!(result.cascade_uuids(), vec!["h_up"]);
        assert!(
            result
                .removed_neurons
                .iter()
                .any(|n| n.uuid == "h_up" && n.reason == "no-outgoing")
        );
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
        let result = ablate_group(&incumbent, &members).unwrap();
        assert_eq!(result.transform_class, TransformClass::Approximate);
        let out = result
            .creature
            .neurons
            .iter()
            .find(|n| n.uuid == "output-0")
            .unwrap();
        assert!(
            close(out.bias, 0.25 + 2.0 * 3.0 + 0.5 * 1.0),
            "bias {}",
            out.bias
        );
        assert_eq!(result.compensations.len(), 2);
        validate_creature(&result.creature).unwrap();
    }

    #[test]
    fn a_repeated_member_is_folded_once() {
        let incumbent = two_hidden();
        let once = ablate_group(&incumbent, &group(&["h_a"], 2.0)).unwrap();
        let twice = ablate_group(&incumbent, &group(&["h_a", "h_a"], 2.0)).unwrap();
        assert_eq!(twice.requested_uuids, vec!["h_a"]);
        assert_eq!(twice.compensations, once.compensations);
        assert_eq!(twice.creature, once.creature);
    }

    #[test]
    fn a_group_cut_that_disconnects_every_output_folds_it_to_a_constant() {
        // `h_a` and `h_b` are the only paths to the output. Cutting both is
        // still a *buildable* candidate — the output keeps both folded means as
        // its bias and stops depending on the input — and it is emitted rather
        // than second-guessed. A creature that ignores its input scores badly,
        // and it is the scorer that says so, never the razor.
        let incumbent = two_hidden();
        let result = ablate_group(&incumbent, &group(&["h_a", "h_b"], 2.0)).unwrap();
        assert_eq!(result.after.hidden_neurons, 0);
        assert_eq!(result.after.synapses, 0);
        let out = result
            .creature
            .neurons
            .iter()
            .find(|n| n.uuid == "output-0")
            .unwrap();
        assert!(
            close(out.bias, 0.25 + 2.0 * 3.0 + 2.0 * 1.0),
            "bias {}",
            out.bias
        );
        validate_creature(&result.creature).unwrap();
    }

    #[test]
    fn an_unbuildable_member_blocks_the_whole_group() {
        let incumbent = chain_plus_keep();
        for (members, expect_unknown) in [
            (group(&["h_up", "nope"], 1.0), true),
            (group(&["h_up", "output-0"], 1.0), false),
        ] {
            let err = ablate_group(&incumbent, &members).unwrap_err();
            if expect_unknown {
                assert!(matches!(err, AblationSkip::UnknownNeuron(_)), "{err}");
            } else {
                assert!(matches!(err, AblationSkip::NotHidden { .. }), "{err}");
            }
        }
        let typed = typed_if_fixture();
        let err = ablate_group(&typed, &group(&["h_cond"], 0.0)).unwrap_err();
        assert!(matches!(err, AblationSkip::TypedSynapse { .. }), "{err}");
        let err = ablate_group(&incumbent, &group(&["h_up"], f64::NAN)).unwrap_err();
        assert!(matches!(err, AblationSkip::NonFiniteMean(_)), "{err}");
        let err = ablate_group(&incumbent, &[]).unwrap_err();
        assert!(matches!(err, AblationSkip::EmptyGroup), "{err}");
        assert_eq!(
            incumbent,
            chain_plus_keep(),
            "a skip must not mutate the source"
        );
    }

    #[test]
    fn a_single_member_group_matches_the_single_neuron_ablation() {
        let incumbent = chain_plus_keep();
        let single = ablate_mean(&incumbent, "h_leaf", 1.0, None).unwrap();
        let grouped = ablate_group(&incumbent, &group(&["h_leaf"], 1.0)).unwrap();
        assert_eq!(grouped.creature, single.creature);
        assert_eq!(grouped.removed_neurons, single.removed_neurons);
        assert_eq!(grouped.compensations, single.compensations);
        assert_eq!(grouped.transform_class, single.transform_class);
    }

    #[test]
    fn unknown_neuron_and_non_finite_mean_are_skipped() {
        let incumbent = hidden_identity_creature(0.0, 1.0);
        assert!(matches!(
            ablate_mean(&incumbent, "nope", 0.0, None),
            Err(AblationSkip::UnknownNeuron(_))
        ));
        assert!(matches!(
            ablate_mean(&incumbent, "h1", f64::NAN, None),
            Err(AblationSkip::NonFiniteMean(_))
        ));
        assert!(matches!(
            ablate_mean(&incumbent, "output-0", 0.0, None),
            Err(AblationSkip::NotHidden { .. })
        ));
    }
}
