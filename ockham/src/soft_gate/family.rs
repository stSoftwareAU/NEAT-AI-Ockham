use std::collections::BTreeSet;

use neat_core::CreatureExport;
use serde::Serialize;

use crate::ablation::StructureSnapshot;
use crate::prune::{PruneDetail, prune_hidden_neuron};
use crate::stats::ActivationStats;

use super::Budget;

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub(super) struct Candidate {
    #[serde(skip)]
    pub creature: CreatureExport,
    pub target: f64,
    pub target_hidden: usize,
    pub size: StructureSnapshot,
    pub retained_original_hidden: BTreeSet<String>,
    pub support_nodes: BTreeSet<String>,
    pub reachable: bool,
    pub target_growth_units: f64,
    pub overshoot_growth_units: f64,
    pub overshoot_hidden: usize,
    pub achieved_capacity: f64,
    pub capacity_overshoot: f64,
    pub rewrites: Vec<PruneDetail>,
    pub refusals: Vec<String>,
}

pub(super) fn materialise(
    parent: &CreatureExport,
    removal_order: &[String],
    targets: &[f64],
    stats: &ActivationStats,
    budget: &mut Budget,
) -> Result<Vec<Candidate>, String> {
    let original: BTreeSet<_> = parent
        .neurons
        .iter()
        .filter(|n| n.neuron_type == "hidden")
        .map(|n| n.uuid.clone())
        .collect();
    let outputs: BTreeSet<_> = parent
        .neurons
        .iter()
        .filter(|n| n.neuron_type == "output")
        .map(|n| n.uuid.clone())
        .collect();
    if removal_order.iter().any(|id| !original.contains(id)) {
        return Err("removal sequence contains a protected or unknown neuron".into());
    }
    let mut working = parent.clone();
    let mut sequence = removal_order.iter();
    let mut rewrites = Vec::new();
    let mut refusals = Vec::new();
    let full_size = StructureSnapshot::of(parent);
    let mut result = Vec::new();
    for &target in targets {
        let target_hidden = (target * original.len() as f64).floor() as usize;
        while retained_hidden(&working, &original).len() > target_hidden {
            let Some(uuid) = sequence.next() else {
                break;
            };
            if !working
                .neurons
                .iter()
                .any(|n| n.neuron_type == "hidden" && n.uuid == *uuid)
            {
                continue;
            }
            budget.rewrite()?;
            let stat = stats.by_uuid(uuid);
            match prune_hidden_neuron(&working, uuid, stat.map(|s| s.mean), stat) {
                Ok(candidate) => {
                    working = candidate.creature;
                    rewrites.push(candidate.detail);
                }
                Err(e) => refusals.push(format!("{uuid}: {e}")),
            }
        }
        neat_core::creature_validate(&working, &crate::incumbent::OCKHAM_VALIDATE_OPTIONS)
            .map_err(|e| format!("invalid experiment candidate: {e}"))?;
        if working.input != parent.input
            || working.output != parent.output
            || working
                .neurons
                .iter()
                .filter(|n| n.neuron_type == "output")
                .map(|n| n.uuid.clone())
                .collect::<BTreeSet<_>>()
                != outputs
        {
            return Err("core rewrite changed protected inputs or outputs".into());
        }
        let size = StructureSnapshot::of(&working);
        let retained_original_hidden = retained_hidden(&working, &original);
        let retained_count = retained_original_hidden.len();
        let achieved_capacity = retained_count as f64 / original.len() as f64;
        let support_nodes = working
            .neurons
            .iter()
            .filter(|n| {
                n.neuron_type != "output"
                    && !retained_original_hidden.contains(&n.uuid)
                    && !parent
                        .neurons
                        .iter()
                        .any(|p| p.neuron_type == "constant" && p.uuid == n.uuid)
            })
            .map(|n| n.uuid.clone())
            .collect();
        result.push(Candidate {
            creature: working.clone(),
            target,
            target_hidden,
            reachable: retained_count <= target_hidden,
            target_growth_units: target * full_size.growth_units,
            overshoot_growth_units: (target * full_size.growth_units - size.growth_units).max(0.0),
            overshoot_hidden: target_hidden.saturating_sub(retained_count),
            size,
            achieved_capacity,
            capacity_overshoot: (target - achieved_capacity).max(0.0),
            retained_original_hidden,
            support_nodes,
            rewrites: rewrites.clone(),
            refusals: refusals.clone(),
        });
    }
    Ok(result)
}

fn retained_hidden(creature: &CreatureExport, original: &BTreeSet<String>) -> BTreeSet<String> {
    creature
        .neurons
        .iter()
        .filter(|n| n.neuron_type == "hidden" && original.contains(&n.uuid))
        .map(|n| n.uuid.clone())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::super::{Cost, manifest::Limits};
    use super::*;
    use crate::fixtures::{creature, neuron, synapse, wide_creature};
    use crate::stats::ActivationStats;

    fn budget() -> Budget {
        Budget::new(
            &Limits {
                record_evaluations: 100_000,
                compilations: 10_000,
                rewrites: 1000,
                io_bytes: 1_000_000,
                seconds: 60.0,
            },
            &Cost::default(),
        )
        .unwrap()
    }

    #[test]
    fn family_is_nested_by_original_identity_and_preserves_the_full_parent() {
        let parent = wide_creature(1, 20, "TANH");
        let before = neat_core::creature_to_json(&parent).unwrap();
        let order = crate::ordering::random_order(&parent, 7);
        let family = materialise(
            &parent,
            &order,
            &[1.0, 0.9, 0.75, 0.5],
            &ActivationStats::empty(),
            &mut budget(),
        )
        .unwrap();
        assert_eq!(
            neat_core::creature_to_json(&family[0].creature).unwrap(),
            before
        );
        for pair in family.windows(2) {
            assert!(
                pair[1]
                    .retained_original_hidden
                    .is_subset(&pair[0].retained_original_hidden)
            );
            assert!(pair[1].size.growth_units <= pair[1].target_growth_units);
        }
        for item in &family {
            neat_core::creature_validate(
                &item.creature,
                &crate::incumbent::OCKHAM_VALIDATE_OPTIONS,
            )
            .unwrap();
            assert_eq!(item.creature.input, 1);
            assert_eq!(item.creature.output, 1);
        }
        assert_eq!(neat_core::creature_to_json(&parent).unwrap(), before);
    }

    #[test]
    fn cascade_overshoot_and_unreachable_budgets_are_reported() {
        let parent = creature(
            1,
            1,
            vec![
                neuron("hidden", "a", 0.0, Some("TANH")),
                neuron("hidden", "b", 0.0, Some("TANH")),
                neuron("output", "output-0", 0.0, Some("IDENTITY")),
            ],
            vec![
                synapse("input-0", "a", 1.0),
                synapse("a", "b", 1.0),
                synapse("b", "output-0", 1.0),
            ],
        );
        let items = materialise(
            &parent,
            &["a".into(), "b".into()],
            &[0.9, 0.5],
            &ActivationStats::empty(),
            &mut budget(),
        )
        .unwrap();
        assert!(items[0].overshoot_growth_units > 0.5);
        assert!(!items[0].rewrites.is_empty());
        assert_eq!(items[0].retained_original_hidden.len(), 0);
        let blocked = materialise(
            &parent,
            &[],
            &[0.5],
            &ActivationStats::empty(),
            &mut budget(),
        )
        .unwrap();
        assert!(!blocked[0].reachable);
        assert_eq!(blocked[0].size.hidden_neurons, 2);
    }
}
