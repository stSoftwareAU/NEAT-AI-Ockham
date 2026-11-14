use neat_ai_ockham::fixtures::{creature, neuron, synapse, typed_synapse};
use neat_ai_ockham::prune::{prune_edge, prune_edge_by_key};
use neat_core::{SynapseKey, SynapseType};

#[test]
fn explicit_role_cuts_the_requested_edge_and_leaves_the_parent_immutable() {
    let parent = creature(
        2,
        1,
        vec![neuron("output", "output-0", 0.0, Some("IF"))],
        vec![
            typed_synapse("input-0", "output-0", 1.0, "condition"),
            typed_synapse("input-0", "output-0", 2.0, "positive"),
            typed_synapse("input-1", "output-0", 3.0, "negative"),
        ],
    );
    let before = parent.clone();
    let key = SynapseKey {
        from_uuid: "input-0".into(),
        to_uuid: "output-0".into(),
        role: SynapseType::Positive,
    };
    let cut = prune_edge_by_key(&parent, &key, Some(0.25), None).unwrap();
    assert_eq!(cut.detail.removed_synapses[0].role, "positive");
    neat_ai_ockham::incumbent::validate_creature(&cut.creature).unwrap();
    assert_eq!(parent, before);

    // Pair-based callers keep the established first-role contract.
    let first = prune_edge(&parent, "input-0", "output-0", Some(0.25), None).unwrap();
    assert_ne!(first.detail.removed_synapses[0].role, "positive");
}

#[test]
fn an_absent_role_is_not_substituted_with_an_existing_role() {
    let parent = creature(
        1,
        1,
        vec![neuron("output", "output-0", 0.0, Some("IDENTITY"))],
        vec![synapse("input-0", "output-0", 1.0)],
    );
    let mut key = SynapseKey {
        from_uuid: "input-0".into(),
        to_uuid: "output-0".into(),
        role: SynapseType::Positive,
    };
    let error = prune_edge_by_key(&parent, &key, None, None).unwrap_err();
    assert!(error.to_string().contains("positive"), "{error}");
    key.role = SynapseType::Standard;
    let cut = prune_edge_by_key(&parent, &key, None, None).unwrap();
    assert_eq!(cut.detail.removed_synapses[0].role, "standard");
}
