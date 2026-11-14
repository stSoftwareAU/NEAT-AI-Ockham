#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixtures::{creature, neuron, synapse, wide_creature};
    use super::super::{Cost, manifest::Limits};

    fn budget() -> Budget {
        Budget::new(&Limits { record_evaluations: 1_000_000, compilations: 100_000,
            rewrites: 1000, io_bytes: 1_000_000, seconds: 60.0 }, &Cost::default()).unwrap()
    }

    #[test]
    fn gate_perturbations_use_core_inference_and_freeze_the_parent() {
        let parent = creature(1, 1, vec![
            neuron("hidden", "h", 0.0, Some("IDENTITY")),
            neuron("output", "output-0", 0.1, Some("IDENTITY")),
        ], vec![synapse("input-0", "h", 0.5), synapse("h", "output-0", 0.8)]);
        let before = neat_core::creature_to_json(&parent).unwrap();
        let mut canonical = neat_core::compile_creature(&parent).unwrap();
        let mut dense = gated(&parent, &["h".into()], &[1.0], &mut budget()).unwrap();
        assert_eq!(dense.activate(&[0.5], 1), canonical.activate(&[0.5], 1));
        let mut half = gated(&parent, &["h".into()], &[0.5], &mut budget()).unwrap();
        assert!((half.activate(&[0.5], 1)[0] - 0.2).abs() < 1e-6);
        let rows = vec![Row { inputs: vec![0.5], targets: vec![0.0] }];
        let derivative = mask_derivatives(&parent, &["h".into()], &[1.0], &rows, 0.001, &mut budget()).unwrap()[0];
        // d((0.2*m + 0.1)^2)/dm at m=1 is 0.12.
        assert!((derivative - 0.12).abs() < 1e-4, "{derivative}");
        assert_eq!(neat_core::creature_to_json(&parent).unwrap(), before);
    }

    #[test]
    fn supported_activations_and_fits_are_reproducible() {
        for squash in ["IDENTITY", "LOGISTIC", "TANH"] {
            let parent = wide_creature(1, 4, squash);
            supported(&parent).unwrap();
            let before = neat_core::creature_to_json(&parent).unwrap();
            let rows = (0..8).map(|i| Row { inputs: vec![i as f32 / 8.0], targets: vec![0.1] }).collect::<Vec<_>>();
            let config = Fit { steps: 2, learning_rate: 0.1, temperature: 0.2, difference_step: 0.001 };
            let a = fit(&parent, &rows, &config, 7, &mut budget()).unwrap();
            let b = fit(&parent, &rows, &config, 7, &mut budget()).unwrap();
            assert_eq!(a.log_scores, b.log_scores);
            assert_eq!(a.removal_order, b.removal_order);
            assert!(a.log_scores.iter().any(|s| s.abs() > 0.01));
            assert_eq!(neat_core::creature_to_json(&parent).unwrap(), before);
        }
    }

    #[test]
    fn discontinuous_aggregate_unknown_and_typed_semantics_are_refused() {
        for squash in ["STEP", "IF", "MINIMUM", "MAXIMUM", "ABSOLUTE", "invented"] {
            let parent = wide_creature(1, 4, squash);
            assert!(supported(&parent).unwrap_err().contains("unsupported activation"));
        }
        let mut parent = wide_creature(1, 4, "TANH");
        parent.synapses[0].synapse_type = Some("condition".into());
        assert!(supported(&parent).unwrap_err().contains("typed synapses"));
        parent.synapses[0].synapse_type = None;
        supported(&parent).unwrap();
    }

    #[test]
    fn budget_exhaustion_refuses_fitting_before_inference() {
        let parent = wide_creature(1, 4, "TANH");
        let mut b = budget();
        b.limits.record_evaluations = 1;
        let rows = vec![Row { inputs: vec![0.2], targets: vec![0.0] }; 2];
        let err = loss(&parent, &["h0".into()], &[1.0], &rows, &mut b).unwrap_err();
        assert!(err.contains("budget exhausted"));
    }
}
