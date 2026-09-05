//! Benchmark downstream-sensitivity ordering against the existing controls (#105).
//!
//! Every candidate the sweep visits costs scorer time, so what an ordering is
//! worth is how quickly it reaches cuts a judge confirms. This measures exactly
//! that on a creature carrying the failure mode the issue names: neurons that
//! fire hard and change nothing, because the weights between them and the
//! outputs attenuate everything they produce.
//!
//! The score is **not** the ranking key — scoring a ranking by its own key
//! proves nothing. Each visited neuron goes through the real
//! [`neat_ai_ockham::ablate_mean`], recursive cleanup and all, and the candidate
//! is then judged by a **proxy scorer**: the NEAT-AI-core compiled forward pass,
//! run over a fixed probe set, comparing the candidate's outputs against the
//! incumbent's. A cut is *confirmed* when the outputs are unchanged within
//! tolerance. That stands in for the full-corpus scorer, which needs a real
//! corpus and a real judge; the run-level economics come from `report` on a
//! real run, and this harness reports the same four measures so the two can be
//! read side by side.
//!
//! Building the order is timed too, because a ranking nobody can afford is not
//! a ranking: the backward propagation indexes the creature once and reads
//! every hidden neuron from that index.

use std::time::Instant;

use neat_ai_ockham::fixtures::{creature, neuron, synapse};
use neat_ai_ockham::ordering::{Ordering, OrderingConfig, hidden_order};
use neat_ai_ockham::stats::{ActivationStats, NeuronStats};
use neat_ai_ockham::{SensitivityIndex, ablate_mean};
use neat_core::{CreatureExport, compile_creature};

/// Outputs stay within this of the incumbent for the proxy judge to confirm.
const TOLERANCE: f32 = 1e-6;
/// Probe records the proxy judge compares the outputs over.
const PROBES: usize = 64;

/// A creature mixing three populations the rankings have to tell apart.
///
/// - `mute{i}_{step}` — loud chains of `depth` neurons whose last edge into the
///   output carries weight zero. Every edge inside the chain is heavy, so a
///   one-layer signal sees only the tail; nothing downstream can see any of
///   them, and they are all dead wood.
/// - `quiet{i}` — barely-audible neurons wired straight into an output with a
///   heavy weight. The activation rankings screen them first and the judge
///   rejects them, which is the cost this ordering is trying to avoid.
/// - `live{i}` — ordinary contributing neurons, loud and heavily wired.
fn mixed_creature(
    inputs: usize,
    chains: usize,
    depth: usize,
    quiet: usize,
    live: usize,
) -> CreatureExport {
    let mut neurons = Vec::with_capacity(chains * depth + quiet + live + 2);
    let mut synapses = Vec::with_capacity(chains * (depth + 1) + 2 * quiet + 2 * live);
    for i in 0..chains {
        for step in 0..depth {
            let uuid = format!("mute{i}_{step}");
            neurons.push(neuron("hidden", &uuid, 0.0, Some("TANH")));
            let source = if step == 0 {
                format!("input-{}", i % inputs)
            } else {
                format!("mute{i}_{}", step - 1)
            };
            synapses.push(synapse(&source, &uuid, 4.0));
        }
        // The dead weight: everything behind it is invisible to the outputs,
        // however heavily the chain is wired inside itself.
        synapses.push(synapse(&format!("mute{i}_{}", depth - 1), "output-0", 0.0));
    }
    for i in 0..quiet {
        let uuid = format!("quiet{i}");
        neurons.push(neuron("hidden", &uuid, 0.0, Some("TANH")));
        synapses.push(synapse(&format!("input-{}", i % inputs), &uuid, 0.05));
        synapses.push(synapse(&uuid, "output-1", 3.0));
    }
    for i in 0..live {
        let uuid = format!("live{i}");
        neurons.push(neuron("hidden", &uuid, 0.0, Some("TANH")));
        synapses.push(synapse(&format!("input-{}", i % inputs), &uuid, 0.8));
        synapses.push(synapse(&uuid, "output-0", 1.5));
    }
    neurons.push(neuron("output", "output-0", 0.0, Some("IDENTITY")));
    neurons.push(neuron("output", "output-1", 0.0, Some("IDENTITY")));
    creature(inputs, 2, neurons, synapses)
}

/// Statistics for every hidden neuron: the muted ones are the loudest.
fn stats_for(creature: &CreatureExport) -> ActivationStats {
    let mut stats = ActivationStats::empty();
    stats.neurons = creature
        .neurons
        .iter()
        .enumerate()
        .filter(|(_, n)| n.neuron_type == "hidden")
        .map(|(i, n)| {
            let mean_abs = match () {
                _ if n.uuid.starts_with("quiet") => 0.02,
                _ if n.uuid.starts_with("live") => 0.45,
                // The muted chains fire hardest of all — and change nothing.
                _ => 0.90,
            };
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

/// Fixed probe inputs — the same records for every strategy and candidate.
fn probes(inputs: usize) -> Vec<Vec<f32>> {
    (0..PROBES)
        .map(|record| {
            (0..inputs)
                .map(|i| ((record * 7 + i * 13) % 21) as f32 / 10.0 - 1.0)
                .collect()
        })
        .collect()
}

/// Outputs of `creature` over every probe, or `None` when it will not compile.
fn outputs(creature: &CreatureExport, probes: &[Vec<f32>]) -> Option<Vec<Vec<f32>>> {
    let mut net = compile_creature(creature).ok()?;
    Some(
        probes
            .iter()
            .map(|input| net.activate(input, creature.output))
            .collect(),
    )
}

/// Whether the candidate leaves every probe output within [`TOLERANCE`].
fn confirmed(before: &[Vec<f32>], after: &[Vec<f32>]) -> bool {
    before.len() == after.len()
        && before.iter().zip(after).all(|(b, a)| {
            b.len() == a.len() && b.iter().zip(a).all(|(x, y)| (x - y).abs() <= TOLERANCE)
        })
}

fn main() {
    const INPUTS: usize = 6;
    const CHAINS: usize = 60;
    const DEPTH: usize = 4;
    const QUIET: usize = 600;
    const LIVE: usize = 600;
    const VISITS: usize = 150;

    let creature = mixed_creature(INPUTS, CHAINS, DEPTH, QUIET, LIVE);
    let stats = stats_for(&creature);
    let probe_set = probes(INPUTS);
    let baseline = outputs(&creature, &probe_set).expect("the incumbent must compile");
    println!(
        "creature: {} neurons, {} synapses ({CHAINS} muted chains of {DEPTH}, {QUIET} quiet, \
         {LIVE} live)",
        creature.neurons.len(),
        creature.synapses.len()
    );
    let index = SensitivityIndex::new(&creature);
    println!(
        "sensitivity: mute0_0 {:?}, quiet0 {:?}, live0 {:?} (importance, outputs anchored at 1)",
        index.importance("mute0_0"),
        index.importance("quiet0"),
        index.importance("live0"),
    );
    println!(
        "first {VISITS} visits, judged by a compiled forward pass over {PROBES} probes \
         (not by the ranking key):"
    );
    println!(
        "  {:<26} {:>9} {:>9} {:>10} {:>12} {:>10}",
        "ordering", "first(ms)", "cuts/h", "units/h", "calls/cut", "build(ms)"
    );

    for strategy in [
        Ordering::Random,
        Ordering::LowVariance,
        Ordering::LowMeanAbs,
        Ordering::LowOutgoingContribution,
        Ordering::LowFanOut,
        Ordering::HighGrowthSaving,
        Ordering::LowOutputSensitivity,
        Ordering::LowEstimatedEffect,
    ] {
        let started = Instant::now();
        let order = hidden_order(&creature, &stats, OrderingConfig::new(strategy), 42);
        let build_ms = started.elapsed().as_secs_f64() * 1000.0;

        let judging = Instant::now();
        let (mut calls, mut cuts, mut units) = (0u64, 0u64, 0.0f64);
        let mut first_ms: Option<f64> = None;
        for uuid in order.iter().take(VISITS) {
            let Ok(ablation) = ablate_mean(&creature, uuid, 0.1, None) else {
                // A visit the razor cannot propose for never reaches a judge.
                continue;
            };
            calls += 1;
            let Some(after) = outputs(&ablation.creature, &probe_set) else {
                continue;
            };
            if !confirmed(&baseline, &after) {
                continue;
            }
            cuts += 1;
            units += ablation.before.growth_units - ablation.after.growth_units;
            first_ms.get_or_insert_with(|| judging.elapsed().as_secs_f64() * 1000.0);
        }
        let hours = judging.elapsed().as_secs_f64() / 3_600.0;
        let per_hour = |n: f64| if hours > 0.0 { n / hours } else { 0.0 };
        let calls_per_cut = if cuts > 0 {
            format!("{:.1}", calls as f64 / cuts as f64)
        } else {
            // No confirmed cut means every call bought nothing, which is the
            // outcome to report rather than a division by zero.
            "none".to_string()
        };
        println!(
            "  {:<26} {:>9} {:>9.0} {:>10.0} {:>12} {:>10.1}",
            strategy.name(),
            first_ms.map_or("none".to_string(), |ms| format!("{ms:.1}")),
            per_hour(cuts as f64),
            per_hour(units),
            calls_per_cut,
            build_ms,
        );
    }
    println!(
        "first(ms) is time to the first confirmed cut; cuts/h and units/h are confirmed cuts \
         and growth units per hour of this harness; calls/cut is proxy scorer calls per \
         confirmed cut. Only a full-corpus scorer accepts a cut."
    );
}
