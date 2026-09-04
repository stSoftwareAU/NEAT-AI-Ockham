//! Constant substitution — a candidate for the blocked majority (Issue #103).
//!
//! [`crate::ablation::ablate_mean`] removes a hidden neuron and folds its mean
//! activation into every downstream **bias**. That only works where the target
//! sums its inputs, so it fails closed on the structure a forest-heavy creature
//! is mostly made of: an aggregate target (`IF`, `MEAN`, `MINIMUM`, …) does not
//! sum, and a typed synapse carries a role a bias cannot stand in for. Those
//! neurons were visited, counted as `blocked`, and never tested.
//!
//! This module tests them. The substitution keeps the **edge** and replaces the
//! **source**: the hidden neuron becomes a `constant` neuron emitting its
//! measured mean, its incoming synapses go, and its outgoing synapses — weights
//! and roles untouched — stay exactly where they were.
//!
//! ```text
//! before:  x → h(TANH) --condition--> IF → output
//! after:        c=mean(h) --condition--> IF → output      (x's branch cascades away)
//! ```
//!
//! Two things follow, and they are why this is safe where the bias fold is not:
//!
//! - the aggregate target still reads a value on the same synapse, so `MEAN`
//!   still averages the same number of inputs and an `IF` still has one edge of
//!   each role (NEAT-AI-core rule 12);
//! - nothing downstream is rewritten, so the approximation is exactly the one
//!   the razor already makes — *this neuron's output is close enough to its
//!   mean* — and no worse.
//!
//! It is still only a **proposal**. The candidate faces `creature.validate()`,
//! the sampled screen and the authoritative full scorer like any other, and the
//! scorer alone decides whether the structure it removes was earning its keep.

use std::collections::HashSet;
use std::fmt;

use neat_core::{CreatureExport, NeuronExport};

use crate::ablation::{RemovedNeuron, StructureSnapshot};
use crate::blocked::BlockedReason;
use crate::fixtures::sort_synapses_canonically;
use crate::incumbent::validate_creature;

/// Neuron type of a substituted source, and of nothing else Ockham writes.
const CONSTANT_TYPE: &str = "constant";

/// Why a requested constant substitution was not emitted.
#[derive(Debug, Clone, PartialEq)]
pub enum SubstitutionSkip {
    /// No listed neuron has this UUID.
    UnknownNeuron(String),
    /// Only hidden neurons are substitution targets.
    NotHidden {
        /// Requested UUID.
        uuid: String,
        /// Declared type.
        neuron_type: String,
    },
    /// Mean was NaN/Inf, so there is no value to emit.
    NonFiniteMean(f64),
    /// The neuron feeds nothing, so a constant in its place would be dead
    /// weight NEAT-AI-core rejects (rule 16).
    NoOutgoing(String),
    /// Final candidate failed `creature.validate()`.
    Invalid(String),
}

impl SubstitutionSkip {
    /// The reason code this skip is counted under (Issue #103).
    pub fn blocked_reason(&self) -> BlockedReason {
        match self {
            Self::UnknownNeuron(_) | Self::NotHidden { .. } => BlockedReason::UnsafeTopology,
            Self::NonFiniteMean(_) => BlockedReason::MissingActivation,
            Self::NoOutgoing(_) => BlockedReason::NoOutputPath,
            Self::Invalid(_) => BlockedReason::ValidationFailed,
        }
    }
}

impl fmt::Display for SubstitutionSkip {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownNeuron(u) => write!(f, "no neuron `{u}`"),
            Self::NotHidden { uuid, neuron_type } => {
                write!(f, "`{uuid}` is {neuron_type}, not hidden")
            }
            Self::NonFiniteMean(m) => write!(f, "non-finite mean {m}"),
            Self::NoOutgoing(u) => write!(f, "`{u}` feeds nothing; a constant there is dead weight"),
            Self::Invalid(m) => write!(f, "candidate failed creature.validate(): {m}"),
        }
    }
}

/// One emitted constant-substitution candidate.
#[derive(Debug, Clone, PartialEq)]
pub struct ConstantSubstitution {
    /// Hidden neuron that became a constant.
    pub uuid: String,
    /// Mean the constant emits.
    pub mean: f64,
    /// Neurons the cascade removed once they fed nothing.
    pub removed_neurons: Vec<RemovedNeuron>,
    /// Structure before the transform.
    pub before: StructureSnapshot,
    /// Structure after the transform.
    pub after: StructureSnapshot,
    /// Validated candidate creature.
    pub creature: CreatureExport,
}

/// Replace hidden neuron `uuid` with a constant emitting `mean`.
///
/// The incoming half of the neuron is deleted and whatever upstream structure
/// that leaves feeding nothing cascades away with it; the outgoing half is
/// preserved byte for byte, roles included. The incumbent is never mutated.
pub fn substitute_constant(
    incumbent: &CreatureExport,
    uuid: &str,
    mean: f64,
) -> Result<ConstantSubstitution, SubstitutionSkip> {
    if !mean.is_finite() {
        return Err(SubstitutionSkip::NonFiniteMean(mean));
    }
    let requested = incumbent
        .neurons
        .iter()
        .position(|n| n.uuid == uuid)
        .ok_or_else(|| SubstitutionSkip::UnknownNeuron(uuid.to_string()))?;
    if incumbent.neurons[requested].neuron_type != "hidden" {
        return Err(SubstitutionSkip::NotHidden {
            uuid: uuid.to_string(),
            neuron_type: incumbent.neurons[requested].neuron_type.clone(),
        });
    }
    if !incumbent.synapses.iter().any(|s| s.from_uuid == uuid) {
        return Err(SubstitutionSkip::NoOutgoing(uuid.to_string()));
    }

    let mut working = incumbent.clone();
    // Memetic state indexes neurons and synapses that are about to move
    // (NEAT-AI-core rule 31), exactly as the ablation path drops it.
    working.memetic = None;
    let before = StructureSnapshot::of(&working);

    let mut constant = working.neurons.remove(requested);
    constant.neuron_type = CONSTANT_TYPE.into();
    // Rule 15: a constant emits its bias, so it carries no squash — the squash
    // that produced the mean is already inside the measured value.
    constant.squash = None;
    constant.bias = mean;
    working.synapses.retain(|s| s.to_uuid != uuid);
    // Rule 11: inside the computational slice every constant precedes every
    // hidden neuron. Moving it to the head of that slice preserves the relative
    // order of everything else, so every surviving synapse still runs forwards.
    working.neurons.insert(constant_slot(&working.neurons), constant);

    let removed_neurons = cascade_dead_sources(&mut working);
    sort_synapses_canonically(&mut working);
    validate_creature(&working).map_err(|e| SubstitutionSkip::Invalid(e.to_string()))?;

    let after = StructureSnapshot::of(&working);
    Ok(ConstantSubstitution {
        uuid: uuid.to_string(),
        mean,
        removed_neurons,
        before,
        after,
        creature: working,
    })
}

/// Where a new constant belongs: ahead of the first non-constant neuron.
fn constant_slot(neurons: &[NeuronExport]) -> usize {
    neurons
        .iter()
        .position(|n| n.neuron_type != CONSTANT_TYPE)
        .unwrap_or(neurons.len())
}

/// Remove every non-output neuron left feeding nothing, until none remain.
///
/// A neuron with no outgoing synapse cannot reach an output, so removing it
/// changes no output value — and NEAT-AI-core rejects one that stays (rules 16
/// and 18). Only *incoming* edges are deleted with it, so no surviving neuron
/// loses an input: an `IF` target keeps one edge of each role, and an aggregate
/// keeps the same arity.
fn cascade_dead_sources(working: &mut CreatureExport) -> Vec<RemovedNeuron> {
    let mut removed = Vec::new();
    loop {
        let sources: HashSet<&str> = working
            .synapses
            .iter()
            .map(|s| s.from_uuid.as_str())
            .collect();
        let dead = working
            .neurons
            .iter()
            .find(|n| n.neuron_type != "output" && !sources.contains(n.uuid.as_str()))
            .map(|n| (n.uuid.clone(), n.neuron_type.clone()));
        let Some((uuid, neuron_type)) = dead else {
            return removed;
        };
        removed.push(RemovedNeuron {
            uuid: uuid.clone(),
            neuron_type,
            reason: "no-outgoing",
        });
        working.neurons.retain(|n| n.uuid != uuid);
        working.synapses.retain(|s| s.to_uuid != uuid);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ablation::{AblationSkip, ablate_mean};
    use crate::fixtures::{creature, neuron, synapse, typed_synapse};
    use neat_core::compile_creature;

    /// `h_cond` feeds an `IF` neuron through a typed `condition` synapse, and
    /// `h_if` is the aggregate itself: the two shapes `ablate_mean` fails
    /// closed on, and the bulk of a forest-heavy creature.
    fn typed_if_fixture() -> CreatureExport {
        creature(
            1,
            1,
            vec![
                neuron("hidden", "h_cond", 0.25, Some("IDENTITY")),
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

    /// `h_src` is a genuine constant: `IDENTITY(1.5 + 0 * x)`.
    fn constant_source_fixture() -> CreatureExport {
        creature(
            1,
            1,
            vec![
                neuron("hidden", "h_src", 1.5, Some("IDENTITY")),
                neuron("hidden", "h_mean", 0.0, Some("MEAN")),
                neuron("output", "output-0", 0.0, Some("IDENTITY")),
            ],
            vec![
                synapse("input-0", "h_src", 0.0),
                synapse("h_src", "h_mean", 1.0),
                synapse("input-0", "h_mean", 1.0),
                synapse("h_mean", "output-0", 1.0),
            ],
        )
    }

    fn outputs(creature: &CreatureExport, xs: &[f32]) -> Vec<f32> {
        let mut net = compile_creature(creature).unwrap();
        xs.iter().map(|&x| net.activate(&[x], 1)[0]).collect()
    }

    /// The point of the whole exercise: a neuron the ablation path blocks is
    /// proposable here, and the candidate is one NEAT-AI-core accepts.
    #[test]
    fn a_typed_edge_the_ablation_path_blocks_substitutes_a_constant() {
        let incumbent = typed_if_fixture();
        validate_creature(&incumbent).expect("fixture is a valid incumbent");
        assert!(
            matches!(
                ablate_mean(&incumbent, "h_cond", 0.5, None),
                Err(AblationSkip::TypedSynapse { .. })
            ),
            "the fixture must be blocked for the ablation path"
        );

        let result = substitute_constant(&incumbent, "h_cond", 0.5).expect("substitution");
        validate_creature(&result.creature).expect("candidate must validate");
        assert_eq!(incumbent, typed_if_fixture(), "incumbent must be untouched");

        let substituted = result
            .creature
            .neurons
            .iter()
            .find(|n| n.uuid == "h_cond")
            .expect("the neuron stays, as a constant");
        assert_eq!(substituted.neuron_type, "constant");
        assert_eq!(substituted.squash, None, "rule 15: a constant has no squash");
        assert!((substituted.bias - 0.5).abs() < f64::EPSILON);
        assert!(
            result
                .creature
                .synapses
                .iter()
                .any(|s| s.from_uuid == "h_cond"
                    && s.to_uuid == "h_if"
                    && s.synapse_type.as_deref() == Some("condition")),
            "the role-carrying edge must survive untouched: {:?}",
            result.creature.synapses
        );
        assert!(
            result.creature.synapses.iter().all(|s| s.to_uuid != "h_cond"),
            "the incoming half is what the substitution removes"
        );
        assert!(result.after.hidden_neurons < result.before.hidden_neurons);
    }

    /// The aggregate neuron itself — the dominant blocked category — is
    /// proposable too, and its now-dead upstream cascades away.
    #[test]
    fn the_aggregate_neuron_itself_substitutes_and_its_upstream_cascades() {
        let incumbent = typed_if_fixture();
        assert!(
            matches!(
                ablate_mean(&incumbent, "h_if", 0.5, None),
                Err(AblationSkip::AggregateNeuron { .. } | AblationSkip::AggregateTarget { .. })
            ),
            "the aggregate neuron must be blocked for the ablation path"
        );

        let result = substitute_constant(&incumbent, "h_if", -0.75).expect("substitution");
        validate_creature(&result.creature).expect("candidate must validate");
        assert!(
            result
                .removed_neurons
                .iter()
                .any(|n| n.uuid == "h_cond" && n.reason == "no-outgoing"),
            "h_cond fed only the IF neuron: {:?}",
            result.removed_neurons
        );
        assert_eq!(
            result.after.hidden_neurons, 0,
            "both hidden neurons are gone; one constant is left"
        );
    }

    /// Exactness where it is checkable: a neuron whose activation really is
    /// constant substitutes with no change to any output.
    #[test]
    fn substituting_a_genuinely_constant_neuron_preserves_every_output() {
        let incumbent = constant_source_fixture();
        validate_creature(&incumbent).unwrap();
        let xs = [0.0f32, 1.0, -2.5, 3.25];
        let before = outputs(&incumbent, &xs);
        let result = substitute_constant(&incumbent, "h_src", 1.5).expect("substitution");
        let after = outputs(&result.creature, &xs);
        for (x, (a, b)) in xs.iter().zip(before.iter().zip(after.iter())) {
            assert!(
                (a - b).abs() <= 1e-5 * a.abs().max(b.abs()).max(1.0),
                "x={x}: {a} vs {b}"
            );
        }
    }

    /// Rule 11 — every constant precedes every hidden neuron — must hold after
    /// the move, whatever the neuron's position was before it.
    #[test]
    fn the_substituted_constant_is_ordered_ahead_of_every_hidden_neuron() {
        let incumbent = constant_source_fixture();
        let result = substitute_constant(&incumbent, "h_mean", 0.5).expect("substitution");
        let types: Vec<&str> = result
            .creature
            .neurons
            .iter()
            .map(|n| n.neuron_type.as_str())
            .collect();
        assert_eq!(
            types,
            vec!["constant", "output"],
            "h_src fed only h_mean, so it cascades away: {types:?}"
        );
        validate_creature(&result.creature).unwrap();
    }

    #[test]
    fn a_non_finite_mean_is_a_missing_activation_skip() {
        let incumbent = typed_if_fixture();
        let err = substitute_constant(&incumbent, "h_cond", f64::NAN).unwrap_err();
        assert!(matches!(err, SubstitutionSkip::NonFiniteMean(_)), "{err}");
        assert_eq!(err.blocked_reason(), BlockedReason::MissingActivation);
    }

    #[test]
    fn an_unknown_or_non_hidden_neuron_is_an_unsafe_topology_skip() {
        let incumbent = typed_if_fixture();
        let err = substitute_constant(&incumbent, "nope", 0.5).unwrap_err();
        assert!(matches!(err, SubstitutionSkip::UnknownNeuron(_)), "{err}");
        assert_eq!(err.blocked_reason(), BlockedReason::UnsafeTopology);
        let err = substitute_constant(&incumbent, "output-0", 0.5).unwrap_err();
        assert!(matches!(err, SubstitutionSkip::NotHidden { .. }), "{err}");
        assert_eq!(err.blocked_reason(), BlockedReason::UnsafeTopology);
    }

    /// A constant that feeds nothing is dead weight NEAT-AI-core rejects, so
    /// the substitution refuses to build one rather than emitting a candidate
    /// that cannot validate.
    #[test]
    fn a_neuron_that_feeds_nothing_is_refused_rather_than_emitted() {
        let mut incumbent = typed_if_fixture();
        incumbent.synapses.retain(|s| s.from_uuid != "h_cond");
        let err = substitute_constant(&incumbent, "h_cond", 0.5).unwrap_err();
        assert!(matches!(err, SubstitutionSkip::NoOutgoing(_)), "{err}");
        assert_eq!(err.blocked_reason(), BlockedReason::NoOutputPath);
    }

    /// Fail closed, loudly: a candidate NEAT-AI-core rejects is reported as a
    /// validation failure, never emitted as something to score.
    #[test]
    fn a_candidate_that_fails_validation_is_reported_not_emitted() {
        // Dropping the `positive` edge leaves the IF neuron short of a role, so
        // the candidate breaks rule 12 however the substitution is built.
        let mut incumbent = typed_if_fixture();
        incumbent
            .synapses
            .retain(|s| s.synapse_type.as_deref() != Some("positive"));
        let err = substitute_constant(&incumbent, "h_cond", 0.5).unwrap_err();
        assert!(matches!(err, SubstitutionSkip::Invalid(_)), "{err}");
        assert_eq!(err.blocked_reason(), BlockedReason::ValidationFailed);
    }
}
