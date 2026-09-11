//! Shared creature fixtures.

use neat_core::{CreatureExport, NeuronExport, SynapseExport};

/// Listed neuron constructor used by fixtures and structural tests.
pub fn neuron(neuron_type: &str, uuid: &str, bias: f64, squash: Option<&str>) -> NeuronExport {
    NeuronExport {
        id: None,
        neuron_type: neuron_type.into(),
        uuid: uuid.into(),
        bias,
        squash: squash.map(str::to_string),
    }
}

/// Ordinary (untyped) synapse constructor.
pub fn synapse(from_uuid: &str, to_uuid: &str, weight: f64) -> SynapseExport {
    SynapseExport {
        from_uuid: from_uuid.into(),
        to_uuid: to_uuid.into(),
        weight,
        synapse_type: None,
    }
}

/// Typed synapse constructor (`positive` / `negative` / `condition`).
pub fn typed_synapse(
    from_uuid: &str,
    to_uuid: &str,
    weight: f64,
    synapse_type: &str,
) -> SynapseExport {
    SynapseExport {
        from_uuid: from_uuid.into(),
        to_uuid: to_uuid.into(),
        weight,
        synapse_type: Some(synapse_type.into()),
    }
}

/// Forward-only creature wrapping the supplied neurons and synapses.
///
/// Synapses are sorted by `(from index, to index, type)` so the fixture
/// satisfies NEAT-AI-core `creature.validate()` rule 25.
pub fn creature(
    input: usize,
    output: usize,
    neurons: Vec<NeuronExport>,
    synapses: Vec<SynapseExport>,
) -> CreatureExport {
    let mut creature = CreatureExport {
        input,
        output,
        neurons,
        synapses,
        semantic_version: Some("4.0.0".into()),
        forward_only: true,
        memetic: None,
    };
    sort_synapses_canonically(&mut creature);
    creature
}

/// Sort synapses by `(from index, to index, type)` (validate rule 25).
pub fn sort_synapses_canonically(creature: &mut CreatureExport) {
    let mut index =
        std::collections::HashMap::with_capacity(creature.input + creature.neurons.len());
    for i in 0..creature.input {
        index.insert(format!("input-{i}"), i);
    }
    for (j, neuron) in creature.neurons.iter().enumerate() {
        index.insert(neuron.uuid.clone(), creature.input + j);
    }
    let resolve = |uuid: &str| index.get(uuid).copied().unwrap_or(usize::MAX);
    creature.synapses.sort_by_key(|s| {
        (
            resolve(&s.from_uuid),
            resolve(&s.to_uuid),
            neat_core::parse_synapse_type(s.synapse_type.as_deref()) as u8,
        )
    });
}

/// `hidden` hidden neurons, each fed by every input and feeding the one output.
///
/// The shape a cost measurement needs: every hidden neuron is prunable, so the
/// razor's work per neuron is the whole of what is timed (Issue #91).
pub fn wide_creature(inputs: usize, hidden: usize, squash: &str) -> CreatureExport {
    let mut neurons = Vec::with_capacity(hidden + 1);
    let mut synapses = Vec::with_capacity(hidden * (inputs + 1));
    for h in 0..hidden {
        let uuid = format!("h{h}");
        neurons.push(neuron("hidden", &uuid, 0.01, Some(squash)));
        for i in 0..inputs {
            synapses.push(synapse(&format!("input-{i}"), &uuid, 0.1));
        }
        synapses.push(synapse(&uuid, "output-0", 1.0 / hidden as f64));
    }
    neurons.push(neuron("output", "output-0", 0.0, Some("IDENTITY")));
    creature(inputs, 1, neurons, synapses)
}

/// Minimal forward-only creature: each output is the identity of `input-j`
/// (or `input-0` when there are fewer inputs than outputs).
pub fn identity_creature(inputs: usize, outputs: usize) -> CreatureExport {
    assert!(inputs >= 1 && outputs >= 1);
    let neurons = (0..outputs)
        .map(|j| NeuronExport {
            id: None,
            neuron_type: "output".into(),
            uuid: format!("output-{j}"),
            bias: 0.0,
            squash: Some("IDENTITY".into()),
        })
        .collect();
    let synapses = (0..outputs)
        .map(|j| SynapseExport {
            from_uuid: format!("input-{}", j.min(inputs - 1)),
            to_uuid: format!("output-{j}"),
            weight: 1.0,
            synapse_type: None,
        })
        .collect();
    CreatureExport {
        input: inputs,
        output: outputs,
        neurons,
        synapses,
        semantic_version: Some("4.0.0".into()),
        forward_only: true,
        memetic: None,
    }
}

/// JSON text of [`identity_creature`].
pub fn identity_creature_json(inputs: usize, outputs: usize) -> String {
    neat_core::creature_to_json_pretty(&identity_creature(inputs, outputs)).unwrap()
}

/// Same topology as [`identity_creature`] but with `forwardOnly: false`.
pub fn recurrent_flagged_creature_json(inputs: usize, outputs: usize) -> String {
    let mut creature = identity_creature(inputs, outputs);
    creature.forward_only = false;
    neat_core::creature_to_json_pretty(&creature).unwrap()
}

/// Hidden IDENTITY neuron `h1` between one input and one output.
///
/// `h1` computes `IDENTITY(bias + weight * input-0)` and feeds the output
/// with weight 1. Used to pin activation-statistics arithmetic.
pub fn hidden_identity_creature(bias: f64, weight: f64) -> CreatureExport {
    creature(
        1,
        1,
        vec![
            neuron("hidden", "h1", bias, Some("IDENTITY")),
            neuron("output", "output-0", 0.0, Some("IDENTITY")),
        ],
        vec![
            synapse("input-0", "h1", weight),
            synapse("h1", "output-0", 1.0),
        ],
    )
}

/// `input-0 → a → b → output-0` with a redundant `a → output-0` shortcut.
///
/// The pure-synapse-win shape (Issue #138): cutting the shortcut leaves both
/// hidden neurons in place — `a` still feeds `b`, and `b` still feeds the
/// output — so the candidate removes one synapse, no neuron, and 0.1 growth
/// units. Every other visit takes a hidden neuron with it: cutting `a → b`
/// leaves `b` with no incoming synapse so the cleanup folds it away, cutting
/// `b → output-0` leaves `b` feeding nothing, and either neuron cut removes
/// itself. The two edges out of `input-0` are refused outright — the razor cuts
/// an edge only where the source is a listed neuron.
pub fn shortcut_edge_creature() -> CreatureExport {
    creature(
        1,
        1,
        vec![
            neuron("hidden", "a", 0.1, Some("TANH")),
            neuron("hidden", "b", 0.2, Some("TANH")),
            neuron("output", "output-0", 0.0, Some("IDENTITY")),
        ],
        vec![
            synapse("input-0", "a", 0.7),
            synapse("a", "b", 0.5),
            synapse("a", "output-0", 0.3),
            synapse("b", "output-0", 1.0),
        ],
    )
}

/// UUIDs on `creature` whose squash is one of NEAT-AI-core's six aggregates.
///
/// In declaration order. One home for the membership rule so the fixture pin
/// below and the Issue #202 run gate ask the same question of a creature rather
/// than restating `SquashType::is_aggregate` twice.
pub fn aggregate_uuids(creature: &CreatureExport) -> Vec<&str> {
    creature
        .neurons
        .iter()
        .filter(|n| {
            n.squash
                .as_deref()
                .and_then(|s| neat_core::parse_squash_name(s).ok())
                .is_some_and(|s| s.is_aggregate())
        })
        .map(|n| n.uuid.as_str())
        .collect()
}

/// Aggregate-heavy creature: an `IF` output over a `HYPOT` hidden neuron.
///
/// Every aggregate path the razor has to walk is present here, so a sweep over
/// this fixture actually visits the shapes the other fixtures never reach
/// (Issue #202):
///
/// * `h_hyp` is a **hidden aggregate** — `HYPOT` reduces its whole inward
///   range, so it absorbs no bias fold, and it carries two inward edges so a
///   cut leaves one behind for core's one-edge conversion (#197).
/// * `output-0` is an **`IF`**, so three of its inward edges carry roles
///   (`condition`, `positive`, `negative`) and cutting any of them leaves the
///   `IF` short a role for core's rewrite (#198).
///
/// Shape:
///
/// ```text
/// input-0 ──► h_cond ─condition─► output-0 (IF)
///        └──► h_hyp  ─positive──►
/// input-1 ──► h_hyp
///        └──► h_arm  ─negative──►
/// ```
pub fn if_hypot_creature() -> CreatureExport {
    creature(
        2,
        1,
        vec![
            neuron("hidden", "h_cond", 0.1, Some("LOGISTIC")),
            neuron("hidden", "h_hyp", 0.0, Some("HYPOT")),
            neuron("hidden", "h_arm", 0.2, Some("TANH")),
            neuron("output", "output-0", 0.0, Some("IF")),
        ],
        vec![
            synapse("input-0", "h_cond", 1.0),
            synapse("input-0", "h_hyp", 0.5),
            synapse("input-1", "h_hyp", 0.25),
            synapse("input-1", "h_arm", 1.0),
            typed_synapse("h_cond", "output-0", 1.0, "condition"),
            typed_synapse("h_hyp", "output-0", 2.0, "positive"),
            typed_synapse("h_arm", "output-0", -1.0, "negative"),
        ],
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use neat_core::parse_creature_json;

    /// The Issue #138 fixture is only a pure-synapse-win fixture while cutting
    /// `a → output-0` leaves both hidden neurons standing. Pinned here, beside
    /// the fixture, so a future edit to it fails loudly rather than quietly
    /// turning three tests into something else.
    #[test]
    fn the_shortcut_fixture_offers_a_cut_that_removes_no_neuron() {
        let before = shortcut_edge_creature();
        let cut = crate::prune::prune_edge(&before, "a", "output-0", Some(0.0), None)
            .expect("the shortcut must be cuttable");
        assert!(
            cut.detail.cascade_uuids().is_empty(),
            "no neuron may go with it: {:?}",
            cut.detail
        );
        assert_eq!(cut.before.hidden_neurons, 2);
        assert_eq!(cut.after.hidden_neurons, 2);
        assert_eq!(cut.after.synapses, cut.before.synapses - 1);
        assert!((cut.before.growth_units - cut.after.growth_units - 0.1).abs() < 1e-9);
    }

    /// The Issue #202 fixture is only an aggregate fixture while it validates
    /// *and* carries both aggregates. Pinned here so an edit that drops the
    /// `HYPOT` or downgrades the `IF` fails loudly rather than quietly turning
    /// the run-level gate into a sweep of point-wise structure.
    #[test]
    fn the_if_hypot_fixture_validates_and_carries_both_aggregates() {
        let c = if_hypot_creature();
        crate::incumbent::validate_creature(&c).expect("the fixture must validate");
        assert_eq!(aggregate_uuids(&c), vec!["h_hyp", "output-0"]);
        assert_eq!(
            c.synapses.iter().filter(|s| s.to_uuid == "h_hyp").count(),
            2,
            "the HYPOT keeps two inward edges, so a cut leaves one behind (#197)"
        );
        let mut roles: Vec<&str> = c
            .synapses
            .iter()
            .filter(|s| s.to_uuid == "output-0")
            .filter_map(|s| s.synapse_type.as_deref())
            .collect();
        roles.sort_unstable();
        assert_eq!(roles, vec!["condition", "negative", "positive"]);
    }

    #[test]
    fn identity_fixture_is_forward_only() {
        let creature = parse_creature_json(&identity_creature_json(2, 1)).unwrap();
        assert!(creature.forward_only);
        assert_eq!(creature.input, 2);
        assert_eq!(creature.output, 1);
        assert_eq!(creature.synapses.len(), 1);
    }
}
