use std::collections::BTreeMap;

use neat_ai_ockham::fixtures::{creature, neuron, synapse};
use neat_ai_ockham::stats::{ActivationStats, NeuronStats};

use super::allocation::{Arm, Constraints, Plan, quotas, ranked_edges};

#[test]
fn quotas_waterfill_caps_and_round_in_stable_region_order() {
    let capacities = BTreeMap::from([("a".into(), 1), ("b".into(), 4), ("c".into(), 4)]);
    assert_eq!(quotas(&capacities, 5).unwrap(), BTreeMap::from([
        ("a".into(), 1), ("b".into(), 2), ("c".into(), 2),
    ]));
    assert_eq!(quotas(&capacities, 4).unwrap()["b"], 2);
    assert_eq!(quotas(&capacities, 0).unwrap().values().sum::<usize>(), 0);
    assert_eq!(quotas(&capacities, 9).unwrap(), capacities);
    assert_eq!(quotas(&capacities, 10).unwrap_err(), "target exceeds capacity");
}

fn parent() -> neat_core::CreatureExport {
    creature(2, 1, vec![
        neuron("hidden", "a", 0.0, Some("TANH")),
        neuron("hidden", "b", 0.0, Some("TANH")),
        neuron("output", "output-0", 0.0, Some("IDENTITY")),
    ], vec![
        synapse("input-0", "a", 0.1), synapse("input-1", "a", 0.2),
        synapse("input-0", "b", 1.0), synapse("input-1", "b", 2.0),
        synapse("a", "output-0", 1.0), synapse("b", "output-0", 1.0),
        synapse("input-0", "output-0", 3.0),
    ])
}

fn stats() -> ActivationStats {
    let mut stats = ActivationStats::empty();
    for uuid in ["input-0", "input-1", "a", "b"] {
        let value = NeuronStats { uuid: uuid.into(), neuron_index: 0, count: 4,
            mean: 0.0, variance: 1.0, std_dev: 1.0, mean_abs: 1.0, min: -1.0, max: 1.0 };
        if uuid.starts_with("input") { stats.inputs.push(value); }
        else { stats.neurons.push(value); }
    }
    stats
}

#[test]
fn allocation_arms_use_the_same_importance_but_different_eligible_regions() {
    let parent = parent();
    let ranked = ranked_edges(&parent, &stats(), 7).unwrap();
    assert_eq!(ranked[0].key.from_uuid, "input-0");
    assert_eq!(ranked[0].key.to_uuid, "a");
    let global = Plan::new(&parent, Arm::Global, 4).unwrap();
    let neuron = Plan::new(&parent, Arm::PerNeuron, 4).unwrap();
    let depth = Plan::new(&parent, Arm::PerDepth, 4).unwrap();
    // Region a has two edges and receives quota two; global may still cut it.
    assert!(global.eligible(&parent, &ranked[0].key));
    assert!(!neuron.eligible(&parent, &ranked[0].key));
    assert!(depth.eligible(&parent, &ranked[0].key));
    assert_eq!(depth.depths["output-0"], 2);
}

#[test]
fn stable_ties_are_reproducible_and_missing_measurements_fail_loudly() {
    let mut parent = parent();
    for edge in &mut parent.synapses { edge.weight = 1.0; }
    let a = ranked_edges(&parent, &stats(), 42).unwrap();
    let b = ranked_edges(&parent, &stats(), 42).unwrap();
    assert_eq!(a.iter().map(|e| &e.key).collect::<Vec<_>>(),
        b.iter().map(|e| &e.key).collect::<Vec<_>>());
    assert_eq!(a.len(), parent.synapses.len());
    let error = ranked_edges(&parent, &ActivationStats::empty(), 42).unwrap_err();
    assert!(error.contains("unmeasured source"), "{error}");
}

#[test]
fn usefulness_refuses_losing_an_input_and_does_not_count_zero_weight_paths() {
    let parent = parent();
    let constraints = Constraints::new(&parent).unwrap();
    constraints.check(&parent).unwrap();
    let mut disconnected = parent.clone();
    disconnected.synapses.retain(|e| e.from_uuid != "input-1");
    assert_eq!(constraints.check(&disconnected).unwrap_err(), "lost useful input input-1");
    for edge in &mut disconnected.synapses { edge.weight = 0.0; }
    assert!(constraints.check(&disconnected).unwrap_err().starts_with("output lacks input path"));
}
