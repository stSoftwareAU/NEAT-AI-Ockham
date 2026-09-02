//! Exact cost-aware IDENTITY neuron collapse (Issue #5).
//!
//! For a hidden IDENTITY neuron
//!
//! ```text
//! y = bias_y + Σ(x_k * a_k)
//! ```
//!
//! and an ordinary outgoing synapse `y → z` with weight `b`, eliminate `y` by
//!
//! ```text
//! bias_z += bias_y * b
//! x_k → z  weight += a_k * b
//! ```
//!
//! Parallel synapses merge by adding weights. The automatic transform is
//! emitted only when NEAT growth units fall (`hidden + synapses/10`), unless
//! an explicit experimental override requests otherwise. Typed/aggregate
//! synapses are skipped, never guessed.

use std::fmt;

use neat_core::{CreatureExport, SquashType, SynapseExport, parse_squash_name};
use serde::Serialize;

use crate::ablation::{StructureSnapshot, TransformClass, cleanup_cascade};
use crate::fixtures::sort_synapses_canonically;
use crate::incumbent::validate_creature;

/// Options for [`collapse_identity`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct CollapseOptions {
    /// When true, emit even if growth units rise. Default is false.
    pub allow_cost_increase: bool,
}

/// Why an IDENTITY collapse was not emitted.
#[derive(Debug, Clone, PartialEq)]
pub enum CollapseSkip {
    /// No listed neuron has this UUID.
    UnknownNeuron(String),
    /// Only hidden neurons collapse.
    NotHidden {
        /// Requested UUID.
        uuid: String,
        /// Declared type.
        neuron_type: String,
    },
    /// Squash is missing or not IDENTITY.
    NotIdentity {
        /// Neuron UUID.
        uuid: String,
        /// Declared squash.
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
    /// Downstream target is an aggregate squash.
    AggregateTarget {
        /// Target UUID.
        uuid: String,
        /// Squash name.
        squash: String,
    },
    /// Bypass would create a self-connection.
    SelfLoop {
        /// The UUID that would connect to itself.
        uuid: String,
    },
    /// Automatic collapse would raise growth units.
    CostIncrease {
        /// Growth units before.
        before: f64,
        /// Growth units after.
        after: f64,
    },
    /// Final candidate failed `creature.validate()`.
    Invalid(String),
}

impl fmt::Display for CollapseSkip {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownNeuron(u) => write!(f, "no neuron `{u}`"),
            Self::NotHidden { uuid, neuron_type } => {
                write!(f, "`{uuid}` is {neuron_type}, not hidden")
            }
            Self::NotIdentity { uuid, squash } => {
                write!(f, "`{uuid}` squash `{squash}` is not IDENTITY")
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
            Self::SelfLoop { uuid } => {
                write!(f, "collapse of `{uuid}` would create a self-connection")
            }
            Self::CostIncrease { before, after } => write!(
                f,
                "IDENTITY collapse raises growth units {before} → {after}; skipped"
            ),
            Self::Invalid(m) => write!(f, "candidate failed creature.validate(): {m}"),
        }
    }
}

/// One `x_k → z` synapse written or merged during collapse.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct BypassedSynapse {
    /// Source UUID (`x_k`).
    pub from_uuid: String,
    /// Destination UUID (`z`).
    pub to_uuid: String,
    /// `a_k * b` added to the destination weight.
    pub added_weight: f64,
    /// Weight after merge.
    pub weight_after: f64,
    /// True when an existing parallel synapse absorbed the weight.
    pub merged: bool,
}

/// Record of one emitted IDENTITY collapse.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct IdentityCollapse {
    /// Collapsed hidden-neuron UUID.
    pub uuid: String,
    /// Always [`TransformClass::Exact`].
    pub transform_class: TransformClass,
    /// Bias of the removed IDENTITY neuron.
    pub bias: f64,
    /// Downstream bias updates `bias_z += bias_y * b`.
    pub bias_updates: Vec<(String, f64, f64)>,
    /// Bypassed / merged synapses.
    pub bypasses: Vec<BypassedSynapse>,
    /// Structure before.
    pub before: StructureSnapshot,
    /// Structure after.
    pub after: StructureSnapshot,
    /// Validated candidate.
    #[serde(skip)]
    pub creature: CreatureExport,
}

/// Collapse hidden IDENTITY neuron `uuid` on a clone of `incumbent`.
pub fn collapse_identity(
    incumbent: &CreatureExport,
    uuid: &str,
    options: CollapseOptions,
) -> Result<IdentityCollapse, CollapseSkip> {
    let neuron = incumbent
        .neurons
        .iter()
        .find(|n| n.uuid == uuid)
        .ok_or_else(|| CollapseSkip::UnknownNeuron(uuid.to_string()))?;
    if neuron.neuron_type != "hidden" {
        return Err(CollapseSkip::NotHidden {
            uuid: uuid.to_string(),
            neuron_type: neuron.neuron_type.clone(),
        });
    }
    let squash_name = neuron.squash.as_deref().unwrap_or("IDENTITY");
    match parse_squash_name(squash_name) {
        Ok(SquashType::Identity) => {}
        _ => {
            return Err(CollapseSkip::NotIdentity {
                uuid: uuid.to_string(),
                squash: squash_name.to_string(),
            });
        }
    }
    let bias_y = neuron.bias;

    let incoming: Vec<SynapseExport> = incumbent
        .synapses
        .iter()
        .filter(|s| s.to_uuid == uuid)
        .cloned()
        .collect();
    let outgoing: Vec<SynapseExport> = incumbent
        .synapses
        .iter()
        .filter(|s| s.from_uuid == uuid)
        .cloned()
        .collect();
    for syn in incoming.iter().chain(&outgoing) {
        if let Some(ty) = &syn.synapse_type {
            return Err(CollapseSkip::TypedSynapse {
                from_uuid: syn.from_uuid.clone(),
                to_uuid: syn.to_uuid.clone(),
                synapse_type: ty.clone(),
            });
        }
    }
    for syn in &outgoing {
        let target = incumbent
            .neurons
            .iter()
            .find(|n| n.uuid == syn.to_uuid)
            .ok_or_else(|| CollapseSkip::UnknownNeuron(syn.to_uuid.clone()))?;
        let t_name = target.squash.as_deref().unwrap_or("IDENTITY");
        if parse_squash_name(t_name).is_ok_and(|s| s.is_aggregate()) {
            return Err(CollapseSkip::AggregateTarget {
                uuid: target.uuid.clone(),
                squash: t_name.to_string(),
            });
        }
        for src in &incoming {
            if src.from_uuid == syn.to_uuid {
                return Err(CollapseSkip::SelfLoop {
                    uuid: src.from_uuid.clone(),
                });
            }
        }
    }

    let mut working = incumbent.clone();
    working.memetic = None;
    let before = StructureSnapshot::of(&working);

    let mut bias_updates = Vec::new();
    let mut bypasses = Vec::new();
    for out in &outgoing {
        {
            let target = working
                .neurons
                .iter_mut()
                .find(|n| n.uuid == out.to_uuid)
                .ok_or_else(|| CollapseSkip::UnknownNeuron(out.to_uuid.clone()))?;
            let before_bias = target.bias;
            target.bias += bias_y * out.weight;
            bias_updates.push((out.to_uuid.clone(), before_bias, target.bias));
        }
        for src in &incoming {
            add_or_merge(
                &mut working,
                &src.from_uuid,
                &out.to_uuid,
                src.weight * out.weight,
                &mut bypasses,
            )?;
        }
    }

    working.neurons.retain(|n| n.uuid != uuid);
    working
        .synapses
        .retain(|s| s.from_uuid != uuid && s.to_uuid != uuid);

    cleanup_cascade(&mut working, &mut Vec::new(), &mut Vec::new())
        .map_err(|e| CollapseSkip::Invalid(e.to_string()))?;
    sort_synapses_canonically(&mut working);

    let after = StructureSnapshot::of(&working);
    if !options.allow_cost_increase && after.growth_units >= before.growth_units {
        return Err(CollapseSkip::CostIncrease {
            before: before.growth_units,
            after: after.growth_units,
        });
    }

    validate_creature(&working).map_err(|e| CollapseSkip::Invalid(e.to_string()))?;

    Ok(IdentityCollapse {
        uuid: uuid.to_string(),
        transform_class: TransformClass::Exact,
        bias: bias_y,
        bias_updates,
        bypasses,
        before,
        after,
        creature: working,
    })
}

fn add_or_merge(
    working: &mut CreatureExport,
    from_uuid: &str,
    to_uuid: &str,
    added: f64,
    bypasses: &mut Vec<BypassedSynapse>,
) -> Result<(), CollapseSkip> {
    if let Some(existing) = working
        .synapses
        .iter_mut()
        .find(|s| s.from_uuid == from_uuid && s.to_uuid == to_uuid && s.synapse_type.is_none())
    {
        existing.weight += added;
        bypasses.push(BypassedSynapse {
            from_uuid: from_uuid.to_string(),
            to_uuid: to_uuid.to_string(),
            added_weight: added,
            weight_after: existing.weight,
            merged: true,
        });
        return Ok(());
    }
    if working
        .synapses
        .iter()
        .any(|s| s.from_uuid == from_uuid && s.to_uuid == to_uuid && s.synapse_type.is_some())
    {
        return Err(CollapseSkip::TypedSynapse {
            from_uuid: from_uuid.to_string(),
            to_uuid: to_uuid.to_string(),
            synapse_type: "existing-typed".into(),
        });
    }
    working.synapses.push(SynapseExport {
        from_uuid: from_uuid.to_string(),
        to_uuid: to_uuid.to_string(),
        weight: added,
        synapse_type: None,
    });
    bypasses.push(BypassedSynapse {
        from_uuid: from_uuid.to_string(),
        to_uuid: to_uuid.to_string(),
        added_weight: added,
        weight_after: added,
        merged: false,
    });
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixtures::{creature, neuron, synapse};
    use crate::incumbent::validate_creature;
    use neat_core::compile_creature;

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() <= 1e-12 * a.abs().max(b.abs()).max(1.0)
    }

    fn one_in_one_out(bias: f64, w_in: f64, w_out: f64) -> CreatureExport {
        creature(
            1,
            1,
            vec![
                neuron("hidden", "h1", bias, Some("IDENTITY")),
                neuron("output", "output-0", 0.0, Some("IDENTITY")),
            ],
            vec![
                synapse("input-0", "h1", w_in),
                synapse("h1", "output-0", w_out),
            ],
        )
    }

    fn weight(creature: &CreatureExport, from: &str, to: &str) -> f64 {
        creature
            .synapses
            .iter()
            .find(|s| s.from_uuid == from && s.to_uuid == to)
            .map(|s| s.weight)
            .unwrap_or(0.0)
    }

    fn bias(creature: &CreatureExport, uuid: &str) -> f64 {
        creature
            .neurons
            .iter()
            .find(|n| n.uuid == uuid)
            .map(|n| n.bias)
            .unwrap()
    }

    fn outputs_1(creature: &CreatureExport, xs: &[f32]) -> Vec<f32> {
        let mut net = compile_creature(creature).unwrap();
        xs.iter().map(|&x| net.activate(&[x], 1)[0]).collect()
    }

    #[test]
    fn zero_bias_one_in_one_out_collapses_to_the_product_synapse() {
        let incumbent = one_in_one_out(0.0, 2.0, 1.0);
        validate_creature(&incumbent).unwrap();
        let result = collapse_identity(&incumbent, "h1", CollapseOptions::default()).unwrap();
        assert_eq!(result.transform_class, TransformClass::Exact);
        assert!(result.creature.neurons.iter().all(|n| n.uuid != "h1"));
        assert!(close(weight(&result.creature, "input-0", "output-0"), 2.0));
        assert!(close(bias(&result.creature, "output-0"), 0.0));
        validate_creature(&result.creature).unwrap();
        let xs = [0.0f32, 1.0, -3.0];
        let before = outputs_1(&incumbent, &xs);
        let after = outputs_1(&result.creature, &xs);
        for (a, b) in before.iter().zip(&after) {
            assert!((a - b).abs() <= 1e-5);
        }
        assert_eq!(incumbent.neurons.len(), 2, "incumbent unchanged");
    }

    #[test]
    fn non_zero_bias_updates_downstream_exactly() {
        let incumbent = one_in_one_out(0.5, 2.0, 3.0);
        let result = collapse_identity(&incumbent, "h1", CollapseOptions::default()).unwrap();
        assert!(close(bias(&result.creature, "output-0"), 0.5 * 3.0));
        assert!(close(weight(&result.creature, "input-0", "output-0"), 6.0));
        let xs = [-1.0f32, 0.0, 2.0];
        let before = outputs_1(&incumbent, &xs);
        let after = outputs_1(&result.creature, &xs);
        for (a, b) in before.iter().zip(&after) {
            assert!((a - b).abs() <= 1e-5, "{a} vs {b}");
        }
    }

    #[test]
    fn multiple_in_multiple_out_preserves_outputs() {
        let incumbent = creature(
            2,
            2,
            vec![
                neuron("hidden", "h1", 0.5, Some("IDENTITY")),
                neuron("output", "output-0", 0.0, Some("IDENTITY")),
                neuron("output", "output-1", 0.1, Some("IDENTITY")),
            ],
            vec![
                synapse("input-0", "h1", 1.0),
                synapse("input-1", "h1", 3.0),
                synapse("h1", "output-0", 2.0),
                synapse("h1", "output-1", 4.0),
            ],
        );
        validate_creature(&incumbent).unwrap();
        let result = collapse_identity(&incumbent, "h1", CollapseOptions::default()).unwrap();
        assert!(close(bias(&result.creature, "output-0"), 0.5 * 2.0));
        assert!(close(bias(&result.creature, "output-1"), 0.1 + 0.5 * 4.0));
        assert!(close(weight(&result.creature, "input-0", "output-0"), 2.0));
        assert!(close(weight(&result.creature, "input-0", "output-1"), 4.0));
        assert!(close(weight(&result.creature, "input-1", "output-0"), 6.0));
        assert!(close(weight(&result.creature, "input-1", "output-1"), 12.0));
        let mut net_a = compile_creature(&incumbent).unwrap();
        let mut net_b = compile_creature(&result.creature).unwrap();
        for x0 in [0.0f32, 1.0, -0.5] {
            for x1 in [0.0f32, 2.0, -1.0] {
                let a = net_a.activate(&[x0, x1], 2);
                let b = net_b.activate(&[x0, x1], 2);
                for (u, v) in a.iter().zip(&b) {
                    assert!((u - v).abs() <= 1e-5, "{x0},{x1}: {a:?} vs {b:?}");
                }
            }
        }
        validate_creature(&result.creature).unwrap();
    }

    #[test]
    fn parallel_synapses_merge_by_adding_weights() {
        let incumbent = creature(
            1,
            1,
            vec![
                neuron("hidden", "h1", 0.0, Some("IDENTITY")),
                neuron("output", "output-0", 0.0, Some("IDENTITY")),
            ],
            vec![
                synapse("input-0", "h1", 1.0),
                synapse("h1", "output-0", 1.0),
                synapse("input-0", "output-0", 0.5),
            ],
        );
        validate_creature(&incumbent).unwrap();
        let result = collapse_identity(&incumbent, "h1", CollapseOptions::default()).unwrap();
        assert_eq!(result.creature.synapses.len(), 1);
        assert!(close(weight(&result.creature, "input-0", "output-0"), 1.5));
        assert!(result.bypasses.iter().any(|b| b.merged));
        validate_creature(&result.creature).unwrap();
    }

    #[test]
    fn cost_increasing_automatic_collapse_is_rejected() {
        let n = 5usize;
        let mut neurons = vec![neuron("hidden", "h1", 0.0, Some("IDENTITY"))];
        let mut synapses = Vec::new();
        for i in 0..n {
            neurons.push(neuron(
                "output",
                &format!("output-{i}"),
                0.0,
                Some("IDENTITY"),
            ));
            synapses.push(synapse(&format!("input-{i}"), "h1", 1.0));
            synapses.push(synapse("h1", &format!("output-{i}"), 1.0));
        }
        let incumbent = creature(n, n, neurons, synapses);
        validate_creature(&incumbent).unwrap();
        let before = StructureSnapshot::of(&incumbent);
        let err = collapse_identity(&incumbent, "h1", CollapseOptions::default()).unwrap_err();
        match err {
            CollapseSkip::CostIncrease {
                before: b,
                after: a,
            } => {
                assert!(a > b, "{a} should exceed {b}");
                assert!(close(b, before.growth_units));
            }
            other => panic!("expected CostIncrease, got {other}"),
        }
        let forced = collapse_identity(
            &incumbent,
            "h1",
            CollapseOptions {
                allow_cost_increase: true,
            },
        )
        .unwrap();
        assert!(forced.after.growth_units > forced.before.growth_units);
        validate_creature(&forced.creature).unwrap();
    }

    #[test]
    fn non_identity_is_skipped() {
        let incumbent = creature(
            1,
            1,
            vec![
                neuron("hidden", "h1", 0.0, Some("TANH")),
                neuron("output", "output-0", 0.0, Some("IDENTITY")),
            ],
            vec![
                synapse("input-0", "h1", 1.0),
                synapse("h1", "output-0", 1.0),
            ],
        );
        assert!(matches!(
            collapse_identity(&incumbent, "h1", CollapseOptions::default()),
            Err(CollapseSkip::NotIdentity { .. })
        ));
    }
}
