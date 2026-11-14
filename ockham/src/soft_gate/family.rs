#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixtures::{creature, neuron, synapse, wide_creature};
    use crate::stats::ActivationStats;
    use super::super::{Cost, manifest::Limits};

    fn budget() -> Budget {
        Budget::new(&Limits { record_evaluations: 100_000, compilations: 10_000,
            rewrites: 1000, io_bytes: 1_000_000, seconds: 60.0 }, &Cost::default()).unwrap()
    }

    #[test]
    fn family_is_nested_by_original_identity_and_preserves_the_full_parent() {
        let parent = wide_creature(1, 20, "TANH");
        let before = neat_core::creature_to_json(&parent).unwrap();
        let order = crate::ordering::random_order(&parent, 7);
        let family = materialise(&parent, &order, &[1.0, 0.9, 0.75, 0.5], &ActivationStats::empty(), &mut budget()).unwrap();
        assert_eq!(neat_core::creature_to_json(&family[0].creature).unwrap(), before);
        for pair in family.windows(2) {
            assert!(pair[1].retained_original_hidden.is_subset(&pair[0].retained_original_hidden));
            assert!(pair[1].size.growth_units <= pair[1].target_growth_units);
        }
        for item in &family {
            neat_core::creature_validate(&item.creature, &crate::incumbent::OCKHAM_VALIDATE_OPTIONS).unwrap();
            assert_eq!(item.creature.input, 1);
            assert_eq!(item.creature.output, 1);
        }
        assert_eq!(neat_core::creature_to_json(&parent).unwrap(), before);
    }

    #[test]
    fn cascade_overshoot_and_unreachable_budgets_are_reported() {
        let parent = creature(1, 1, vec![
            neuron("hidden", "a", 0.0, Some("TANH")),
            neuron("hidden", "b", 0.0, Some("TANH")),
            neuron("output", "output-0", 0.0, Some("IDENTITY")),
        ], vec![synapse("input-0", "a", 1.0), synapse("a", "b", 1.0), synapse("b", "output-0", 1.0)]);
        let items = materialise(&parent, &["a".into(), "b".into()], &[0.9, 0.5], &ActivationStats::empty(), &mut budget()).unwrap();
        assert!(items[0].overshoot_growth_units > 0.5);
        assert!(!items[0].rewrites.is_empty());
        assert_eq!(items[0].retained_original_hidden.len(), 0);
        let blocked = materialise(&parent, &[], &[0.5], &ActivationStats::empty(), &mut budget()).unwrap();
        assert!(!blocked[0].reachable);
        assert_eq!(blocked[0].size.hidden_neurons, 2);
    }
}
