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

use std::fmt;

use neat_core::{
    CreatureExport, NeuronExport, SquashType, SynapseExport, apply_squash, parse_squash_name,
};
use serde::Serialize;

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
        }
    }
}

/// Provenance of one emitted ablation candidate.
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
/// recorded in provenance when present; it is not consulted for the arithmetic.
pub fn ablate_mean(
    incumbent: &CreatureExport,
    uuid: &str,
    mean: f64,
    stats: Option<&NeuronStats>,
) -> Result<Ablation, AblationSkip> {
    if !mean.is_finite() {
        return Err(AblationSkip::NonFiniteMean(mean));
    }
    let requested_index = incumbent
        .neurons
        .iter()
        .position(|n| n.uuid == uuid)
        .ok_or_else(|| AblationSkip::UnknownNeuron(uuid.to_string()))?;
    let requested = &incumbent.neurons[requested_index];
    if requested.neuron_type != "hidden" {
        return Err(AblationSkip::NotHidden {
            uuid: uuid.to_string(),
            neuron_type: requested.neuron_type.clone(),
        });
    }
    let requested_squash = squash_of(requested)?;
    if requested_squash.is_aggregate() {
        return Err(AblationSkip::AggregateNeuron {
            uuid: uuid.to_string(),
            squash: requested
                .squash
                .clone()
                .unwrap_or_else(|| "IDENTITY".into()),
        });
    }

    let mut working = incumbent.clone();
    working.memetic = None;
    let before = StructureSnapshot::of(&working);

    let outgoing = synapses_from(&working, uuid);
    for syn in &outgoing {
        require_ordinary(syn)?;
        let target = neuron_by_uuid(&working, &syn.to_uuid)
            .ok_or_else(|| AblationSkip::UnknownNeuron(syn.to_uuid.clone()))?;
        reject_aggregate_neuron(target)?;
    }
    for syn in &synapses_to(&working, uuid) {
        require_ordinary(syn)?;
    }

    let mut compensations = Vec::new();
    let mut used_mean = false;
    for syn in outgoing {
        apply_bias_fold(&mut working, &syn, mean, "mean", &mut compensations)?;
        used_mean = true;
    }

    let mut removed_neurons = vec![RemovedNeuron {
        uuid: uuid.to_string(),
        neuron_type: "hidden".into(),
        reason: "requested",
    }];
    remove_neuron(&mut working, uuid);

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

fn first_dead_non_output(working: &CreatureExport) -> Option<String> {
    working.neurons.iter().find_map(|n| {
        if n.neuron_type == "output" {
            return None;
        }
        let out = working
            .synapses
            .iter()
            .filter(|s| s.from_uuid == n.uuid)
            .count();
        (out == 0).then(|| n.uuid.clone())
    })
}

fn first_hidden_without_incoming(working: &CreatureExport) -> Option<String> {
    working.neurons.iter().find_map(|n| {
        if n.neuron_type != "hidden" {
            return None;
        }
        let incoming = working
            .synapses
            .iter()
            .filter(|s| s.to_uuid == n.uuid)
            .count();
        (incoming == 0).then(|| n.uuid.clone())
    })
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

fn remove_neuron(working: &mut CreatureExport, uuid: &str) {
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
