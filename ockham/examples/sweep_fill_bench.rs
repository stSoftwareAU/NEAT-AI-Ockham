//! Benchmark one screening batch on a forest-shaped creature (Issue #91).
//!
//! Most hidden neurons of a GRQ forest feed an aggregate squash, so the razor
//! can propose nothing for them: the sweep visits them, is rejected, and moves
//! on. This measures how long filling one batch of candidates costs when that
//! rejected majority dominates the walk — the wall clock that decides how many
//! batches, and so how much screen coverage, fit in a run's budget.

use std::time::Instant;

use neat_ai_ockham::fixtures::{creature, neuron, synapse};
use neat_ai_ockham::stats::{ActivationStats, NeuronStats};
use neat_ai_ockham::sweep::Sweep;
use neat_core::CreatureExport;

/// Share of hidden neurons whose only outgoing synapse feeds an aggregate.
///
/// Measured at 78.8% on a real GRQ creature in Issue #93; rounded here.
const BLOCKED_IN_TEN: usize = 8;

/// Forest-shaped creature: `hidden` hidden neurons, `inputs` inputs, one output.
fn forest_creature(inputs: usize, hidden: usize, hubs: usize) -> CreatureExport {
    let mut neurons = Vec::with_capacity(hidden + hubs + 1);
    let mut synapses = Vec::with_capacity(hidden * (inputs + 1) + hubs);
    for h in 0..hidden {
        let uuid = format!("h{h}");
        neurons.push(neuron("hidden", &uuid, (h % 7) as f64 * 0.01, Some("TANH")));
        for i in 0..inputs {
            let weight = ((h * inputs + i) % 13) as f64 * 0.05 - 0.3;
            synapses.push(synapse(&format!("input-{}", i % inputs), &uuid, weight));
        }
        if h % 10 < BLOCKED_IN_TEN {
            synapses.push(synapse(&uuid, &format!("agg{}", (h / 10) % hubs), 0.5));
        } else {
            synapses.push(synapse(&uuid, "output-0", 1.0 / hidden as f64));
        }
    }
    for hub in 0..hubs {
        neurons.push(neuron("hidden", &format!("agg{hub}"), 0.0, Some("MEAN")));
        synapses.push(synapse(&format!("agg{hub}"), "output-0", 0.1));
    }
    neurons.push(neuron("output", "output-0", 0.0, Some("IDENTITY")));
    creature(inputs, 1, neurons, synapses)
}

/// Non-zero mean for every hidden neuron, so no visit is skipped for want of one.
fn stats_for(creature: &CreatureExport) -> ActivationStats {
    let mut stats = ActivationStats::empty();
    stats.neurons = creature
        .neurons
        .iter()
        .enumerate()
        .filter(|(_, n)| n.neuron_type == "hidden")
        .map(|(i, n)| NeuronStats {
            uuid: n.uuid.clone(),
            neuron_index: i,
            count: 1_000,
            mean: 0.25,
            variance: 0.1,
            std_dev: 0.316,
            mean_abs: 0.25,
            min: -1.0,
            max: 1.0,
        })
        .collect();
    stats
}

fn main() {
    const INPUTS: usize = 6;
    const HIDDEN: usize = 7_000;
    const HUBS: usize = 40;
    const VISITS: usize = 400;

    let creature = forest_creature(INPUTS, HIDDEN, HUBS);
    let stats = stats_for(&creature);
    println!(
        "forest creature: {} neurons, {} synapses, {BLOCKED_IN_TEN}/10 hidden feed an aggregate",
        creature.neurons.len(),
        creature.synapses.len()
    );

    // The sweep's cost is per *visit*, and on a forest most visits are
    // rejections, so time a fixed number of visits: how long the walk takes is
    // what decides whether a run screens anything at all.
    let order: Vec<String> = creature
        .neurons
        .iter()
        .filter(|n| n.neuron_type == "hidden")
        .map(|n| n.uuid.clone())
        .take(VISITS)
        .collect();
    let mut sweep = Sweep::new(&creature, 42);
    sweep.order.retain(|uuid| order.contains(uuid));
    let started = Instant::now();
    let (candidates, skips) = sweep.fill_batch(&creature, &stats, VISITS);
    let elapsed = started.elapsed();
    let proposed = candidates.len();
    let mut reasons: Vec<(String, usize)> = Vec::new();
    for skip in &skips {
        let class = skip.reason.split('`').next().unwrap_or("").trim().to_string();
        match reasons.iter_mut().find(|(seen, _)| *seen == class) {
            Some((_, n)) => *n += 1,
            None => reasons.push((class, 1)),
        }
    }
    reasons.sort_by(|a, b| b.1.cmp(&a.1));
    println!(
        "{VISITS} visits: {proposed} candidates in {:.3}s ({:.2}ms per visit)",
        elapsed.as_secs_f64(),
        elapsed.as_secs_f64() * 1000.0 / order.len() as f64,
    );
    for (class, n) in reasons.iter().take(5) {
        println!("  skip: {class} × {n}");
    }
    for skip in skips.iter().rev().take(2) {
        println!("  example skip: {}", skip.reason);
    }

    let clone_started = Instant::now();
    let mut sink = 0usize;
    for _ in 0..20 {
        let copy = creature.clone();
        sink += copy.synapses.len();
    }
    println!(
        "creature.clone(): {:.2}ms each ({sink})",
        clone_started.elapsed().as_secs_f64() * 1000.0 / 20.0
    );
}
