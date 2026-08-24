//! Shared creature fixtures.

use neat_core::{CreatureExport, NeuronExport, SynapseExport};

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
    CreatureExport {
        input: 1,
        output: 1,
        neurons: vec![
            NeuronExport {
                id: None,
                neuron_type: "hidden".into(),
                uuid: "h1".into(),
                bias,
                squash: Some("IDENTITY".into()),
            },
            NeuronExport {
                id: None,
                neuron_type: "output".into(),
                uuid: "output-0".into(),
                bias: 0.0,
                squash: Some("IDENTITY".into()),
            },
        ],
        synapses: vec![
            SynapseExport {
                from_uuid: "input-0".into(),
                to_uuid: "h1".into(),
                weight,
                synapse_type: None,
            },
            SynapseExport {
                from_uuid: "h1".into(),
                to_uuid: "output-0".into(),
                weight: 1.0,
                synapse_type: None,
            },
        ],
        semantic_version: Some("4.0.0".into()),
        forward_only: true,
        memetic: None,
    }
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
