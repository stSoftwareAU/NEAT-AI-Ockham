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
//! [`ablate_synapse`] is the same family one step finer (Issue #133): the unit
//! removed is the **edge**, not the neuron. The removed synapse's contribution
//! is folded into the target's bias,
//!
//! ```text
//! bias_j' = bias_j + source_value * w_ij
//! ```
//!
//! which is the same approximate step, and the same exact cleanup cascade then
//! removes whatever that edge stranded. No weight or contribution threshold
//! decides which edges are eligible — every ordinary synapse is, and the
//! full-corpus scorer remains the only acceptance authority.
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

/// NEAT structural growth units: `hidden + synapses / 10`.
///
/// The scorer multiplies this by its `growthCost`; Ockham records the unitless
/// quantity so deltas are comparable without copying the scorer's cost knob.
pub fn growth_units(hidden_neurons: usize, synapses: usize) -> f64 {
    hidden_neurons as f64 + synapses as f64 / 10.0
}

/// Neuron/synapse counts and the unitless growth-cost proxy.
// `Deserialize` so a filed cleanup report can be read back (Issue #110).
#[derive(Debug, Clone, PartialEq, Serialize, serde::Deserialize)]
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

/// How faithful a rewrite was to the creature it started from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum TransformClass {
    /// A statistic stood in for something the creature computed per record —
    /// a mean-activation fold, a fitted relation.
    Approximate,
    /// The candidate computes what the incumbent computed: only exact
    /// structural repair ran.
    #[default]
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
    /// No synapse joins this ordered pair on the incumbent (Issue #133).
    UnknownSynapse {
        /// Source UUID.
        from_uuid: String,
        /// Destination UUID.
        to_uuid: String,
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
            | Self::UnknownSynapse { .. }
            | Self::EmptyGroup => BlockedReason::UnsafeTopology,
            Self::Invalid(_) => BlockedReason::ValidationFailed,
        }
    }

    /// Whether a constant substitution is worth trying instead (Issue #103).
    ///
    /// True for the structural blocks — an aggregate target, an aggregate
    /// source, a typed edge — where the fold is impossible but the *edge* can
    /// be preserved. False where there was nothing to substitute in the first
    /// place (no neuron, no finite mean, no such edge) or where a candidate was
    /// built and rejected.
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
            Self::UnknownSynapse { from_uuid, to_uuid } => {
                write!(f, "no synapse `{from_uuid}`→`{to_uuid}`")
            }
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

/// Every synapse leaving `uuid`.
fn synapses_from(working: &CreatureExport, uuid: &str) -> Vec<SynapseExport> {
    working
        .synapses
        .iter()
        .filter(|s| s.from_uuid == uuid)
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
    use crate::fixtures::{creature, neuron, synapse, typed_synapse};
    use crate::incumbent::validate_creature;

    /// The folded constants come back through an `f32` activation, so the
    /// tolerance is `f32`-sized rather than the `f64` one exact arithmetic
    /// would use.
    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() <= 1e-6 * a.abs().max(b.abs()).max(1.0)
    }

    /// input → h_up → h_leaf → output, plus input → h_keep → output.
    fn chain_plus_keep() -> CreatureExport {
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

    #[test]
    fn growth_units_count_hidden_neurons_plus_a_tenth_of_the_synapses() {
        assert!(close(growth_units(3, 5), 3.5));
        let snapshot = StructureSnapshot::of(&chain_plus_keep());
        assert_eq!(snapshot.hidden_neurons, 3);
        assert_eq!(snapshot.constant_neurons, 0);
        assert_eq!(snapshot.synapses, 5);
        assert!(close(snapshot.growth_units, 3.5));
    }

    #[test]
    fn the_cascade_strips_a_chain_that_reaches_no_output() {
        // Dropping the leaf's only outgoing edge strands the leaf, and
        // stranding the leaf strands `h_up` behind it.
        let mut working = chain_plus_keep();
        working
            .synapses
            .retain(|s| !(s.from_uuid == "h_leaf" && s.to_uuid == "output-0"));
        let mut compensations = Vec::new();
        let mut removed = Vec::new();
        cleanup_cascade(&mut working, &mut compensations, &mut removed).unwrap();
        let gone: Vec<(&str, &str)> = removed
            .iter()
            .map(|n| (n.uuid.as_str(), n.reason))
            .collect();
        assert_eq!(
            gone,
            vec![("h_leaf", "no-outgoing"), ("h_up", "no-outgoing")],
            "{gone:?}"
        );
        assert_eq!(
            working
                .neurons
                .iter()
                .map(|n| n.uuid.as_str())
                .collect::<Vec<_>>(),
            vec!["h_keep", "output-0"]
        );
        assert!(compensations.is_empty(), "nothing was worth folding");
        validate_creature(&working).unwrap();
    }

    #[test]
    fn the_cascade_folds_a_hidden_neuron_left_with_nothing_to_sum() {
        // `h_up` has no incoming edge, so it activates to IDENTITY(0.1) on
        // every record and that constant belongs in what reads it.
        let mut working = chain_plus_keep();
        working.synapses.retain(|s| s.to_uuid != "h_up");
        let mut compensations = Vec::new();
        let mut removed = Vec::new();
        cleanup_cascade(&mut working, &mut compensations, &mut removed).unwrap();
        assert!(
            removed
                .iter()
                .any(|n| n.uuid == "h_up" && n.reason == "no-incoming"),
            "{removed:?}"
        );
        let folded = compensations
            .iter()
            .find(|c| c.to_uuid == "h_leaf")
            .expect("the constant lands in what read it");
        assert_eq!(folded.kind, "constant");
        assert!(close(folded.bias_after, 0.1), "{}", folded.bias_after);
        validate_creature(&working).unwrap();
    }

    #[test]
    fn the_cascade_fails_closed_on_structure_a_bias_cannot_absorb() {
        // `h_src` is left with nothing to sum, but what reads it is an `IF`
        // arm: no bias stands in for a role, so the cleanup refuses rather
        // than guessing.
        let mut working = creature(
            1,
            1,
            vec![
                neuron("hidden", "h_src", 0.5, Some("IDENTITY")),
                neuron("hidden", "h_if", 0.0, Some("IF")),
                neuron("output", "output-0", 0.0, Some("IDENTITY")),
            ],
            vec![
                synapse("input-0", "h_src", 1.0),
                typed_synapse("h_src", "h_if", 1.0, "condition"),
                typed_synapse("input-0", "h_if", 1.0, "positive"),
                typed_synapse("input-0", "h_if", -1.0, "negative"),
                synapse("h_if", "output-0", 1.0),
            ],
        );
        working.synapses.retain(|s| s.to_uuid != "h_src");
        let err = cleanup_cascade(&mut working, &mut Vec::new(), &mut Vec::new()).unwrap_err();
        assert!(matches!(err, AblationSkip::TypedSynapse { .. }), "{err}");
        assert_eq!(err.blocked_reason(), BlockedReason::UnsafeTopology);
    }

    #[test]
    fn every_skip_names_the_structure_it_is_about() {
        let skip = AblationSkip::AggregateTarget {
            uuid: "h_mean".into(),
            squash: "MEAN".into(),
        };
        assert_eq!(skip.blocked_reason(), BlockedReason::AggregateSquash);
        assert_eq!(
            skip.to_string(),
            "aggregate target `h_mean` (`MEAN`); skipped"
        );
        let skip = AblationSkip::UnknownNeuron("nope".into());
        assert_eq!(skip.blocked_reason(), BlockedReason::UnsafeTopology);
        assert_eq!(skip.to_string(), "no neuron `nope`");
    }
}
