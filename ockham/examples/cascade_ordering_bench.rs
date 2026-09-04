//! Benchmark cascade-aware ordering against `high-growth-saving` and random (#106).
//!
//! Every candidate the sweep visits costs scorer time, so what an ordering is
//! worth is the structure the razor gets to remove per visit. This measures
//! exactly that, on a creature carrying both lone neurons and chains that only
//! come out together.
//!
//! The score is **not** the cascade estimate — scoring a ranking by its own
//! ranking key proves nothing. Each visited neuron is put through the real
//! [`neat_ai_ockham::ablate_mean`], recursive cleanup and all, and what that
//! transform actually removes is what is summed.
//!
//! Building the order is timed too, because a ranking nobody can afford is not
//! a ranking: the dry run indexes the creature once and estimates every hidden
//! neuron from that index.

use std::time::Instant;

use neat_ai_ockham::ablate_mean;
use neat_ai_ockham::fixtures::{creature, neuron, synapse};
use neat_ai_ockham::ordering::{Ordering, OrderingConfig, hidden_order};
use neat_ai_ockham::stats::{ActivationStats, NeuronStats};
use neat_core::CreatureExport;

/// Lone neurons plus chains whose members only come out together.
///
/// A lone neuron is `input-i → l → output-0`: cutting it saves its own
/// structure and nothing else. A chain is `input-i → c0 → … → cN → output-0`:
/// cutting any member strands the rest, so the whole chain leaves at once.
fn mixed_creature(inputs: usize, lone: usize, chains: usize, length: usize) -> CreatureExport {
    let mut neurons = Vec::with_capacity(lone + chains * length + 1);
    let mut synapses = Vec::with_capacity(2 * lone + chains * (length + 1));
    for l in 0..lone {
        let uuid = format!("l{l}");
        neurons.push(neuron("hidden", &uuid, 0.0, Some("TANH")));
        synapses.push(synapse(&format!("input-{}", l % inputs), &uuid, 0.5));
        // Two outgoing edges each, so the edge-count ranking prefers a lone
        // neuron to a chain member — which is the mistake being measured.
        synapses.push(synapse(&uuid, "output-0", 0.1));
        synapses.push(synapse(&uuid, "output-1", 0.1));
    }
    for c in 0..chains {
        for step in 0..length {
            let uuid = format!("c{c}_{step}");
            neurons.push(neuron("hidden", &uuid, 0.0, Some("TANH")));
            let source = if step == 0 {
                format!("input-{}", c % inputs)
            } else {
                format!("c{c}_{}", step - 1)
            };
            synapses.push(synapse(&source, &uuid, 0.5));
        }
        synapses.push(synapse(&format!("c{c}_{}", length - 1), "output-0", 0.1));
    }
    neurons.push(neuron("output", "output-0", 0.0, Some("IDENTITY")));
    neurons.push(neuron("output", "output-1", 0.0, Some("IDENTITY")));
    creature(inputs, 2, neurons, synapses)
}

/// Statistics for every hidden neuron; chain members are the quieter ones.
fn stats_for(creature: &CreatureExport) -> ActivationStats {
    let mut stats = ActivationStats::empty();
    stats.neurons = creature
        .neurons
        .iter()
        .enumerate()
        .filter(|(_, n)| n.neuron_type == "hidden")
        .map(|(i, n)| {
            let mean_abs = if n.uuid.starts_with('c') { 0.05 } else { 0.4 };
            NeuronStats {
                uuid: n.uuid.clone(),
                neuron_index: i,
                count: 1_000,
                mean: mean_abs,
                variance: mean_abs,
                std_dev: mean_abs.sqrt(),
                mean_abs,
                min: -mean_abs,
                max: mean_abs,
            }
        })
        .collect();
    stats
}

fn main() {
    const INPUTS: usize = 6;
    const LONE: usize = 2_000;
    const CHAINS: usize = 200;
    const LENGTH: usize = 5;
    const VISITS: usize = 200;

    let creature = mixed_creature(INPUTS, LONE, CHAINS, LENGTH);
    let stats = stats_for(&creature);
    println!(
        "creature: {} neurons, {} synapses ({LONE} lone, {CHAINS} chains of {LENGTH})",
        creature.neurons.len(),
        creature.synapses.len()
    );
    println!("first {VISITS} visits, scored by what ablate_mean really removes:");

    for strategy in [
        Ordering::Random,
        Ordering::HighGrowthSaving,
        Ordering::CascadeSaving,
        Ordering::CascadeRiskRatio,
    ] {
        let started = Instant::now();
        let order = hidden_order(&creature, &stats, OrderingConfig::new(strategy), 42);
        let build_ms = started.elapsed().as_secs_f64() * 1000.0;
        let (mut saving, mut hidden, mut blocked) = (0.0f64, 0usize, 0usize);
        for uuid in order.iter().take(VISITS) {
            match ablate_mean(&creature, uuid, 0.1, None) {
                Ok(ablation) => {
                    saving += ablation.before.growth_units - ablation.after.growth_units;
                    hidden += ablation.before.hidden_neurons - ablation.after.hidden_neurons;
                }
                // A visit the razor cannot propose for buys nothing, which is
                // exactly what a ranking is supposed to avoid spending on.
                Err(_) => blocked += 1,
            }
        }
        println!(
            "  {:<20} {saving:8.1} growth units, {hidden:5} hidden neurons, \
             {blocked:3} blocked ({:.2} units per visit, order built in {build_ms:.1}ms)",
            strategy.name(),
            saving / VISITS as f64,
        );
    }
}
