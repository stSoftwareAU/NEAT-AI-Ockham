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

#[cfg(test)]
mod tests {
    use super::*;
    use neat_core::parse_creature_json;

    #[test]
    fn identity_fixture_is_forward_only() {
        let creature = parse_creature_json(&identity_creature_json(2, 1)).unwrap();
        assert!(creature.forward_only);
        assert_eq!(creature.input, 2);
        assert_eq!(creature.output, 1);
        assert_eq!(creature.synapses.len(), 1);
    }
}
