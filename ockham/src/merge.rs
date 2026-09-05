//! Correlated-neuron merging — removing one of two near-duplicates (#109).
//!
//! [`crate::signature`] finds pairs of hidden neurons that rise and fall
//! together and fits
//!
//! ```text
//! removed(t) ≈ scale * survivor(t) + offset
//! ```
//!
//! over the probe records. This module spends that relation. For every ordinary
//! outgoing synapse `removed → z` carrying weight `w`, the contribution
//! `w * removed(t)` is rewritten in terms of the survivor:
//!
//! ```text
//! bias_z      += w * offset
//! survivor → z  weight += w * scale
//! ```
//!
//! and `removed` — with everything left feeding nothing behind it — goes.
//! Parallel synapses merge by adding weights, exactly as an IDENTITY collapse
//! does.
//!
//! **Two busy neurons can still be one neuron too many.** 🪒
//!
//! The arithmetic is exact wherever the relation is: an exactly duplicated
//! linear neuron has `scale = 1`, `offset = 0`, and the merged creature
//! computes the same outputs to the last bit the forward pass can carry. It is
//! still reported as [`TransformClass::Approximate`], because a relation fitted
//! from sampled probes is evidence, never proof — the outputs are only known to
//! match on records the razor actually measured.
//!
//! Unsupported topology is skipped, never guessed:
//!
//! - a **typed** outgoing synapse carries a role a plain weighted edge cannot
//!   stand in for;
//! - an **aggregate** target (`IF`, `MEAN`, `MINIMUM`, …) does not sum its
//!   inputs, so folding two edges into one changes what it computes;
//! - a survivor that does not already precede the target would make the
//!   rewritten edge run backwards through a forward-only creature.
//!
//! Unlike an IDENTITY collapse, a merge cannot cost more than it saves: it
//! deletes one hidden neuron and every synapse incident to it, and writes back
//! at most one edge per *outgoing* synapse it deleted. NEAT growth units
//! therefore always fall. [`MergeSkip::CostIncrease`] guards that invariant
//! rather than enforcing a policy — a candidate that somehow grew the creature
//! is reported instead of being scored.
//!
//! Every emitted candidate faces `creature.validate()`, the sampled screen and
//! the authoritative full scorer like any other proposal.

use std::fmt;

use neat_core::{CreatureExport, SynapseExport, parse_squash_name};
use serde::Serialize;

use crate::ablation::{
    RemovedNeuron, StructureSnapshot, TransformClass, cleanup_cascade, remove_neuron,
};
use crate::blocked::BlockedReason;
use crate::fixtures::sort_synapses_canonically;
use crate::incumbent::validate_creature;

/// Fitted `removed ≈ scale * survivor + offset`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct LinearRelation {
    /// Multiplier applied to the survivor's contribution.
    pub scale: f64,
    /// Constant part, folded into downstream biases.
    pub offset: f64,
}

impl LinearRelation {
    /// The identity relation — an exactly duplicated neuron.
    pub const IDENTICAL: Self = Self {
        scale: 1.0,
        offset: 0.0,
    };
}

/// Why a requested merge was not emitted.
#[derive(Debug, Clone, PartialEq)]
pub enum MergeSkip {
    /// No listed neuron has this UUID.
    UnknownNeuron(String),
    /// Only hidden neurons are merge partners.
    NotHidden {
        /// Requested UUID.
        uuid: String,
        /// Declared type.
        neuron_type: String,
    },
    /// Survivor and removed are the same neuron.
    SameNeuron(String),
    /// The fitted relation was NaN/Inf, so there is nothing to fold.
    NonFiniteRelation(LinearRelation),
    /// The removed neuron feeds nothing, so there is nothing to absorb.
    NoOutgoing(String),
    /// A synapse the transform would rewrite is typed.
    TypedSynapse {
        /// Source UUID.
        from_uuid: String,
        /// Destination UUID.
        to_uuid: String,
        /// Role name.
        synapse_type: String,
    },
    /// A downstream target is an aggregate squash.
    AggregateTarget {
        /// Target UUID.
        uuid: String,
        /// Squash name.
        squash: String,
    },
    /// The rewritten edge would connect the survivor to itself.
    SelfLoop {
        /// The UUID that would connect to itself.
        uuid: String,
    },
    /// The survivor does not precede the target, so the edge would run backwards.
    NotForward {
        /// Survivor UUID.
        survivor_uuid: String,
        /// Target the survivor would have to feed.
        to_uuid: String,
    },
    /// The merge grew the creature — the structural invariant did not hold.
    CostIncrease {
        /// Growth units before.
        before: f64,
        /// Growth units after.
        after: f64,
    },
    /// Final candidate failed `creature.validate()`.
    Invalid(String),
}

impl MergeSkip {
    /// The reason code this skip is counted under (Issue #103).
    pub fn blocked_reason(&self) -> BlockedReason {
        match self {
            Self::AggregateTarget { .. } => BlockedReason::AggregateSquash,
            Self::NonFiniteRelation(_) => BlockedReason::MissingActivation,
            Self::NoOutgoing(_) => BlockedReason::NoOutputPath,
            Self::UnknownNeuron(_)
            | Self::NotHidden { .. }
            | Self::SameNeuron(_)
            | Self::TypedSynapse { .. }
            | Self::SelfLoop { .. }
            | Self::NotForward { .. } => BlockedReason::UnsafeTopology,
            Self::Invalid(_) => BlockedReason::ValidationFailed,
            // A merge that grew the creature broke the structural invariant
            // above; nothing the razor could have built a path for.
            Self::CostIncrease { .. } => BlockedReason::Other,
        }
    }
}

impl fmt::Display for MergeSkip {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownNeuron(u) => write!(f, "no neuron `{u}`"),
            Self::NotHidden { uuid, neuron_type } => {
                write!(f, "`{uuid}` is {neuron_type}, not hidden")
            }
            Self::SameNeuron(u) => write!(f, "`{u}` cannot merge into itself"),
            Self::NonFiniteRelation(r) => {
                write!(
                    f,
                    "non-finite relation scale {} offset {}",
                    r.scale, r.offset
                )
            }
            Self::NoOutgoing(u) => write!(f, "`{u}` feeds nothing; there is nothing to absorb"),
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
            Self::SelfLoop { uuid } => write!(f, "merge would connect `{uuid}` to itself"),
            Self::NotForward {
                survivor_uuid,
                to_uuid,
            } => write!(
                f,
                "survivor `{survivor_uuid}` does not precede `{to_uuid}`; skipped"
            ),
            Self::CostIncrease { before, after } => {
                write!(f, "merge raises growth units {before} → {after}; skipped")
            }
            Self::Invalid(m) => write!(f, "candidate failed creature.validate(): {m}"),
        }
    }
}

/// One `survivor → z` synapse written or merged during a merge.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AbsorbedSynapse {
    /// Destination UUID (`z`).
    pub to_uuid: String,
    /// Weight the removed neuron carried into `z`.
    pub removed_weight: f64,
    /// `removed_weight * scale` added to the survivor's edge.
    pub added_weight: f64,
    /// Survivor edge weight after the merge.
    pub weight_after: f64,
    /// True when an existing parallel survivor edge absorbed the weight.
    pub merged: bool,
}

/// Record of one emitted correlated-neuron merge.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct NeuronMerge {
    /// Hidden neuron that absorbed the contribution.
    pub survivor_uuid: String,
    /// Hidden neuron the candidate removed.
    pub removed_uuid: String,
    /// Relation the compensation spent.
    pub relation: LinearRelation,
    /// Always [`TransformClass::Approximate`] — a fitted relation is evidence.
    pub transform_class: TransformClass,
    /// Downstream bias updates `bias_z += w * offset`, in application order.
    pub bias_updates: Vec<(String, f64, f64)>,
    /// Survivor edges written or merged.
    pub absorbed: Vec<AbsorbedSynapse>,
    /// Neurons removed (the requested one first, then the cascade).
    pub removed_neurons: Vec<RemovedNeuron>,
    /// Structure before the transform.
    pub before: StructureSnapshot,
    /// Structure after the transform.
    pub after: StructureSnapshot,
    /// Validated candidate creature.
    #[serde(skip)]
    pub creature: CreatureExport,
}

/// Remove `removed_uuid`, folding its contribution into `survivor_uuid`.
///
/// The incumbent is never mutated. `relation` comes from
/// [`crate::signature::correlate`]; nothing here re-derives or second-guesses
/// it, and nothing here accepts the candidate it builds.
pub fn merge_correlated(
    incumbent: &CreatureExport,
    survivor_uuid: &str,
    removed_uuid: &str,
    relation: LinearRelation,
) -> Result<NeuronMerge, MergeSkip> {
    if !relation.scale.is_finite() || !relation.offset.is_finite() {
        return Err(MergeSkip::NonFiniteRelation(relation));
    }
    if survivor_uuid == removed_uuid {
        return Err(MergeSkip::SameNeuron(survivor_uuid.to_string()));
    }
    let survivor_index = hidden_index(incumbent, survivor_uuid)?;
    hidden_index(incumbent, removed_uuid)?;

    // Every rejection is decided on the incumbent, before anything is cloned:
    // a wide creature is expensive to copy and most pairs never get past here.
    let outgoing: Vec<SynapseExport> = incumbent
        .synapses
        .iter()
        .filter(|s| s.from_uuid == removed_uuid)
        .cloned()
        .collect();
    if outgoing.is_empty() {
        return Err(MergeSkip::NoOutgoing(removed_uuid.to_string()));
    }
    for syn in &outgoing {
        if let Some(ty) = &syn.synapse_type {
            return Err(MergeSkip::TypedSynapse {
                from_uuid: syn.from_uuid.clone(),
                to_uuid: syn.to_uuid.clone(),
                synapse_type: ty.clone(),
            });
        }
        if syn.to_uuid == survivor_uuid {
            return Err(MergeSkip::SelfLoop {
                uuid: survivor_uuid.to_string(),
            });
        }
        let target_index = incumbent
            .neurons
            .iter()
            .position(|n| n.uuid == syn.to_uuid)
            .ok_or_else(|| MergeSkip::UnknownNeuron(syn.to_uuid.clone()))?;
        let target = &incumbent.neurons[target_index];
        let squash = target.squash.as_deref().unwrap_or("IDENTITY");
        if parse_squash_name(squash).is_ok_and(|s| s.is_aggregate()) {
            return Err(MergeSkip::AggregateTarget {
                uuid: target.uuid.clone(),
                squash: squash.to_string(),
            });
        }
        // Removing the neuron does not reorder what is left, so the incumbent's
        // own indices decide whether the rewritten edge still runs forwards.
        if survivor_index >= target_index {
            return Err(MergeSkip::NotForward {
                survivor_uuid: survivor_uuid.to_string(),
                to_uuid: syn.to_uuid.clone(),
            });
        }
        // An existing typed parallel edge cannot take the absorbed weight, and
        // adding a second edge beside it would be a duplicate connection.
        if incumbent.synapses.iter().any(|s| {
            s.from_uuid == survivor_uuid && s.to_uuid == syn.to_uuid && s.synapse_type.is_some()
        }) {
            return Err(MergeSkip::TypedSynapse {
                from_uuid: survivor_uuid.to_string(),
                to_uuid: syn.to_uuid.clone(),
                synapse_type: "existing-typed".into(),
            });
        }
    }

    let mut working = incumbent.clone();
    // Memetic state indexes neurons and synapses that are about to move
    // (NEAT-AI-core rule 31), exactly as every other transform drops it.
    working.memetic = None;
    let before = StructureSnapshot::of(&working);

    let mut bias_updates = Vec::new();
    let mut absorbed = Vec::new();
    for out in &outgoing {
        if relation.offset != 0.0 {
            let target = working
                .neurons
                .iter_mut()
                .find(|n| n.uuid == out.to_uuid)
                .ok_or_else(|| MergeSkip::UnknownNeuron(out.to_uuid.clone()))?;
            let bias_before = target.bias;
            target.bias += relation.offset * out.weight;
            bias_updates.push((out.to_uuid.clone(), bias_before, target.bias));
        }
        absorb(
            &mut working,
            survivor_uuid,
            &out.to_uuid,
            out.weight,
            relation.scale,
            &mut absorbed,
        );
    }

    let mut removed_neurons = vec![RemovedNeuron {
        uuid: removed_uuid.to_string(),
        neuron_type: "hidden".into(),
        reason: "merged",
    }];
    remove_neuron(&mut working, removed_uuid);
    cleanup_cascade(&mut working, &mut Vec::new(), &mut removed_neurons)
        .map_err(|e| MergeSkip::Invalid(e.to_string()))?;
    sort_synapses_canonically(&mut working);

    let after = StructureSnapshot::of(&working);
    // The invariant, checked rather than assumed: a transform that grew the
    // creature is a fault to report, not a candidate to score.
    if after.growth_units >= before.growth_units {
        return Err(MergeSkip::CostIncrease {
            before: before.growth_units,
            after: after.growth_units,
        });
    }
    validate_creature(&working).map_err(|e| MergeSkip::Invalid(e.to_string()))?;

    Ok(NeuronMerge {
        survivor_uuid: survivor_uuid.to_string(),
        removed_uuid: removed_uuid.to_string(),
        relation,
        transform_class: TransformClass::Approximate,
        bias_updates,
        absorbed,
        removed_neurons,
        before,
        after,
        creature: working,
    })
}

/// Index of `uuid` in `creature.neurons`, if it is a hidden neuron.
fn hidden_index(creature: &CreatureExport, uuid: &str) -> Result<usize, MergeSkip> {
    let index = creature
        .neurons
        .iter()
        .position(|n| n.uuid == uuid)
        .ok_or_else(|| MergeSkip::UnknownNeuron(uuid.to_string()))?;
    if creature.neurons[index].neuron_type != "hidden" {
        return Err(MergeSkip::NotHidden {
            uuid: uuid.to_string(),
            neuron_type: creature.neurons[index].neuron_type.clone(),
        });
    }
    Ok(index)
}

/// Add `removed_weight * scale` to the survivor's edge into `to_uuid`.
///
/// A parallel ordinary edge absorbs the weight; otherwise a new one is written.
/// A typed parallel edge is impossible here — [`merge_correlated`] refuses the
/// merge before anything is cloned.
fn absorb(
    working: &mut CreatureExport,
    survivor_uuid: &str,
    to_uuid: &str,
    removed_weight: f64,
    scale: f64,
    absorbed: &mut Vec<AbsorbedSynapse>,
) {
    let added = removed_weight * scale;
    if let Some(existing) = working
        .synapses
        .iter_mut()
        .find(|s| s.from_uuid == survivor_uuid && s.to_uuid == to_uuid && s.synapse_type.is_none())
    {
        existing.weight += added;
        absorbed.push(AbsorbedSynapse {
            to_uuid: to_uuid.to_string(),
            removed_weight,
            added_weight: added,
            weight_after: existing.weight,
            merged: true,
        });
        return;
    }
    working.synapses.push(SynapseExport {
        from_uuid: survivor_uuid.to_string(),
        to_uuid: to_uuid.to_string(),
        weight: added,
        synapse_type: None,
    });
    absorbed.push(AbsorbedSynapse {
        to_uuid: to_uuid.to_string(),
        removed_weight,
        added_weight: added,
        weight_after: added,
        merged: false,
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixtures::{creature, neuron, synapse, typed_synapse};
    use neat_core::compile_creature;

    /// `h_a` and `h_b` compute exactly the same function of `input-0`, and both
    /// feed the one output with different weights. `h_keep` is there so the
    /// creature still has a hidden neuron after the merge.
    fn twin_creature(squash: &str) -> CreatureExport {
        creature(
            1,
            1,
            vec![
                neuron("hidden", "h_a", 0.25, Some(squash)),
                neuron("hidden", "h_b", 0.25, Some(squash)),
                neuron("hidden", "h_keep", 0.0, Some("IDENTITY")),
                neuron("output", "output-0", 0.5, Some("IDENTITY")),
            ],
            vec![
                synapse("input-0", "h_a", 1.5),
                synapse("input-0", "h_b", 1.5),
                synapse("input-0", "h_keep", 0.25),
                synapse("h_a", "output-0", 2.0),
                synapse("h_b", "output-0", -0.75),
                synapse("h_keep", "output-0", 1.0),
            ],
        )
    }

    fn outputs(creature: &CreatureExport, xs: &[f32]) -> Vec<f32> {
        let mut net = compile_creature(creature).unwrap();
        xs.iter().map(|&x| net.activate(&[x], 1)[0]).collect()
    }

    fn weight(creature: &CreatureExport, from: &str, to: &str) -> f64 {
        creature
            .synapses
            .iter()
            .find(|s| s.from_uuid == from && s.to_uuid == to)
            .map(|s| s.weight)
            .unwrap_or(0.0)
    }

    /// The acceptance criterion: an exactly duplicated linear neuron merges with
    /// no change to any output.
    #[test]
    fn an_exactly_duplicated_identity_neuron_merges_with_identical_outputs() {
        let incumbent = twin_creature("IDENTITY");
        validate_creature(&incumbent).expect("fixture must be a valid incumbent");
        let xs = [0.0f32, 1.0, -2.5, 3.25, 17.0];
        let before = outputs(&incumbent, &xs);

        let result = merge_correlated(&incumbent, "h_a", "h_b", LinearRelation::IDENTICAL)
            .expect("an exact duplicate must merge");
        validate_creature(&result.creature).expect("candidate must validate");
        assert_eq!(incumbent, twin_creature("IDENTITY"), "incumbent untouched");

        assert!(result.creature.neurons.iter().all(|n| n.uuid != "h_b"));
        assert!((weight(&result.creature, "h_a", "output-0") - 1.25).abs() < 1e-12);
        assert!(result.absorbed.iter().all(|a| a.merged));
        assert!(result.bias_updates.is_empty(), "offset 0 folds no bias");
        assert!(result.after.hidden_neurons < result.before.hidden_neurons);

        let after = outputs(&result.creature, &xs);
        for (x, (a, b)) in xs.iter().zip(before.iter().zip(&after)) {
            assert!(
                (a - b).abs() <= 1e-5 * a.abs().max(b.abs()).max(1.0),
                "x={x}: {a} vs {b}"
            );
        }
    }

    /// The same holds for a squashed duplicate: the relation is between the
    /// *post*-activation values, so the squash never enters the arithmetic.
    #[test]
    fn an_exactly_duplicated_tanh_neuron_merges_with_identical_outputs() {
        let incumbent = twin_creature("TANH");
        let xs = [0.0f32, 0.5, -1.5, 4.0];
        let before = outputs(&incumbent, &xs);
        let result =
            merge_correlated(&incumbent, "h_a", "h_b", LinearRelation::IDENTICAL).expect("merge");
        let after = outputs(&result.creature, &xs);
        for (a, b) in before.iter().zip(&after) {
            assert!(
                (a - b).abs() <= 1e-5 * a.abs().max(b.abs()).max(1.0),
                "{a} vs {b}"
            );
        }
    }

    /// A scaled and shifted duplicate: the scale rides the weight and the
    /// offset lands in the downstream bias.
    #[test]
    fn a_scaled_relation_folds_the_offset_into_the_downstream_bias() {
        let incumbent = twin_creature("IDENTITY");
        let relation = LinearRelation {
            scale: 2.0,
            offset: 0.5,
        };
        let result = merge_correlated(&incumbent, "h_a", "h_b", relation).unwrap();
        // h_b carried -0.75 into the output; h_a absorbs 2.0 * -0.75 on top of
        // its own 2.0, and the output bias takes 0.5 * -0.75.
        assert!((weight(&result.creature, "h_a", "output-0") - 0.5).abs() < 1e-12);
        let out = result
            .creature
            .neurons
            .iter()
            .find(|n| n.uuid == "output-0")
            .unwrap();
        assert!(
            (out.bias - (0.5 + 0.5 * -0.75)).abs() < 1e-12,
            "{}",
            out.bias
        );
        assert_eq!(result.bias_updates.len(), 1);
        assert_eq!(result.transform_class, TransformClass::Approximate);
        validate_creature(&result.creature).unwrap();
    }

    /// A merge writes a new survivor edge where none existed, and the removed
    /// neuron's now-dead upstream cascades away with it.
    #[test]
    fn a_new_survivor_edge_is_written_and_dead_upstream_cascades() {
        let incumbent = creature(
            1,
            2,
            vec![
                neuron("hidden", "h_a", 0.0, Some("IDENTITY")),
                neuron("hidden", "h_up", 0.0, Some("IDENTITY")),
                neuron("hidden", "h_b", 0.0, Some("IDENTITY")),
                neuron("output", "output-0", 0.0, Some("IDENTITY")),
                neuron("output", "output-1", 0.0, Some("IDENTITY")),
            ],
            vec![
                synapse("input-0", "h_a", 1.0),
                synapse("input-0", "h_up", 1.0),
                synapse("h_up", "h_b", 1.0),
                synapse("h_a", "output-0", 1.0),
                synapse("h_b", "output-1", 3.0),
            ],
        );
        validate_creature(&incumbent).unwrap();
        let result =
            merge_correlated(&incumbent, "h_a", "h_b", LinearRelation::IDENTICAL).expect("merge");
        assert!((weight(&result.creature, "h_a", "output-1") - 3.0).abs() < 1e-12);
        assert!(result.absorbed.iter().any(|a| !a.merged), "a new edge");
        assert!(
            result
                .removed_neurons
                .iter()
                .any(|n| n.uuid == "h_up" && n.reason == "no-outgoing"),
            "{:?}",
            result.removed_neurons
        );
        validate_creature(&result.creature).unwrap();
    }

    #[test]
    fn a_typed_outgoing_edge_or_aggregate_target_fails_closed() {
        let incumbent = creature(
            1,
            1,
            vec![
                neuron("hidden", "h_a", 0.0, Some("IDENTITY")),
                neuron("hidden", "h_b", 0.0, Some("IDENTITY")),
                neuron("hidden", "h_if", 0.0, Some("IF")),
                neuron("output", "output-0", 0.0, Some("IDENTITY")),
            ],
            vec![
                synapse("input-0", "h_a", 1.0),
                synapse("input-0", "h_b", 1.0),
                typed_synapse("h_a", "h_if", 1.0, "condition"),
                typed_synapse("input-0", "h_if", 1.0, "positive"),
                typed_synapse("input-0", "h_if", -1.0, "negative"),
                synapse("h_b", "h_if", 0.0),
                synapse("h_if", "output-0", 1.0),
            ],
        );
        // `h_a` carries a role into the IF neuron.
        let err =
            merge_correlated(&incumbent, "h_b", "h_a", LinearRelation::IDENTICAL).unwrap_err();
        assert!(matches!(err, MergeSkip::TypedSynapse { .. }), "{err}");
        assert_eq!(err.blocked_reason(), BlockedReason::UnsafeTopology);
        // `h_b`'s ordinary edge still lands on an aggregate that does not sum.
        let err =
            merge_correlated(&incumbent, "h_a", "h_b", LinearRelation::IDENTICAL).unwrap_err();
        assert!(matches!(err, MergeSkip::AggregateTarget { .. }), "{err}");
        assert_eq!(err.blocked_reason(), BlockedReason::AggregateSquash);
    }

    /// A survivor that sits after the target would make the rewritten edge run
    /// backwards through a forward-only creature.
    #[test]
    fn a_survivor_that_does_not_precede_the_target_is_refused() {
        let incumbent = creature(
            1,
            1,
            vec![
                neuron("hidden", "h_b", 0.0, Some("IDENTITY")),
                neuron("hidden", "h_mid", 0.0, Some("IDENTITY")),
                neuron("hidden", "h_a", 0.0, Some("IDENTITY")),
                neuron("output", "output-0", 0.0, Some("IDENTITY")),
            ],
            vec![
                synapse("input-0", "h_b", 1.0),
                synapse("input-0", "h_a", 1.0),
                synapse("h_b", "h_mid", 1.0),
                synapse("h_mid", "output-0", 1.0),
                synapse("h_a", "output-0", 1.0),
            ],
        );
        validate_creature(&incumbent).unwrap();
        let err =
            merge_correlated(&incumbent, "h_a", "h_b", LinearRelation::IDENTICAL).unwrap_err();
        assert!(matches!(err, MergeSkip::NotForward { .. }), "{err}");
        // The other direction is structurally fine, which is exactly why both
        // survivor directions are proposed.
        merge_correlated(&incumbent, "h_b", "h_a", LinearRelation::IDENTICAL)
            .expect("the reverse direction runs forwards");
    }

    /// The structural invariant: a merge deletes a whole neuron and writes
    /// back at most one edge per outgoing synapse it deleted, so however wide
    /// the fan-out, growth units fall.
    #[test]
    fn a_wide_fan_out_merge_still_shrinks_the_creature() {
        let outs = 12usize;
        let mut neurons = vec![
            neuron("hidden", "h_a", 0.0, Some("IDENTITY")),
            neuron("hidden", "h_b", 0.0, Some("IDENTITY")),
        ];
        let mut synapses = vec![
            synapse("input-0", "h_a", 1.0),
            synapse("input-0", "h_b", 1.0),
            synapse("h_a", "output-0", 1.0),
        ];
        for i in 0..outs {
            neurons.push(neuron(
                "output",
                &format!("output-{i}"),
                0.0,
                Some("IDENTITY"),
            ));
            synapses.push(synapse("h_b", &format!("output-{i}"), 1.0));
        }
        let incumbent = creature(1, outs, neurons, synapses);
        validate_creature(&incumbent).unwrap();
        let result =
            merge_correlated(&incumbent, "h_a", "h_b", LinearRelation::IDENTICAL).expect("merge");
        assert!(
            result.after.growth_units < result.before.growth_units,
            "{:?} → {:?}",
            result.before,
            result.after
        );
        assert_eq!(result.after.hidden_neurons, 1);
        validate_creature(&result.creature).unwrap();
    }

    /// Two correlated neurons in a chain: the removed one already feeds the
    /// survivor, so absorbing its edge would connect the survivor to itself.
    #[test]
    fn a_removed_neuron_that_feeds_the_survivor_is_refused() {
        let incumbent = creature(
            1,
            1,
            vec![
                neuron("hidden", "h_b", 0.0, Some("IDENTITY")),
                neuron("hidden", "h_a", 0.0, Some("IDENTITY")),
                neuron("output", "output-0", 0.0, Some("IDENTITY")),
            ],
            vec![
                synapse("input-0", "h_b", 1.0),
                synapse("h_b", "h_a", 1.0),
                synapse("h_a", "output-0", 1.0),
            ],
        );
        validate_creature(&incumbent).unwrap();
        let err =
            merge_correlated(&incumbent, "h_a", "h_b", LinearRelation::IDENTICAL).unwrap_err();
        assert!(matches!(err, MergeSkip::SelfLoop { .. }), "{err}");
        assert_eq!(err.blocked_reason(), BlockedReason::UnsafeTopology);
    }

    #[test]
    fn unusable_requests_are_named_rather_than_guessed() {
        let incumbent = twin_creature("IDENTITY");
        for (survivor, removed, relation) in [
            ("h_a", "h_a", LinearRelation::IDENTICAL),
            ("nope", "h_b", LinearRelation::IDENTICAL),
            ("h_a", "nope", LinearRelation::IDENTICAL),
            ("output-0", "h_b", LinearRelation::IDENTICAL),
            (
                "h_a",
                "h_b",
                LinearRelation {
                    scale: f64::NAN,
                    offset: 0.0,
                },
            ),
        ] {
            let err = merge_correlated(&incumbent, survivor, removed, relation).unwrap_err();
            assert!(
                !matches!(err, MergeSkip::Invalid(_)),
                "{survivor}/{removed} must be named, not guessed at: {err}"
            );
        }
        // A neuron that feeds nothing cannot be merged into anything.
        let mut orphan = twin_creature("IDENTITY");
        orphan.synapses.retain(|s| s.from_uuid != "h_b");
        let err = merge_correlated(&orphan, "h_a", "h_b", LinearRelation::IDENTICAL).unwrap_err();
        assert!(matches!(err, MergeSkip::NoOutgoing(_)), "{err}");
        assert_eq!(err.blocked_reason(), BlockedReason::NoOutputPath);
    }
}
