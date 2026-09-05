//! Benchmark bounded group cuts against the equivalent individual cuts (#108).
//!
//! The claim behind structural neighbourhood pruning is that some dead wood
//! only comes out as a group. This measures what a group proposal is worth
//! against the cuts a single-neuron sweep would make instead, on a creature
//! carrying chains, single-output tributaries and lone neurons.
//!
//! Nothing here is scored by the ranking key it was chosen with. Every proposal
//! is put through the real [`neat_ai_ockham::ablate_group`] and every
//! comparison cut through the real [`neat_ai_ockham::ablate_mean`], recursive
//! cleanup and all, and what those transforms actually remove is what is
//! reported: neurons and synapses removed per accepted proposal, and the
//! wall-clock cost of proposing them.
//!
//! This is a **structural** benchmark, not a quality one. It says what a group
//! removes and what it costs to propose; whether the creature is any good
//! afterwards is the full-corpus scorer's verdict, and only a real run against
//! a real scorer can report that.

use std::collections::HashSet;
use std::time::Instant;

use neat_ai_ockham::fixtures::{creature, neuron, synapse};
use neat_ai_ockham::neighbourhood::{
    NeighbourhoodConfig, NeighbourhoodKind, group_batch, propose_neighbourhoods,
};
use neat_ai_ockham::stats::{ActivationStats, NeuronStats};
use neat_ai_ockham::{GroupMember, ablate_group, ablate_mean};
use neat_core::{CreatureExport, compile_creature};

/// Lone neurons, linear chains and single-output tributaries in one creature.
///
/// - lone: `input-i → l → output-0`, nothing behind it.
/// - chain: `input-i → c0 → … → cN → output-0`, each link the only way on.
/// - tributary: two feeders into one exit neuron, which alone reaches the
///   output.
fn mixed_creature(
    inputs: usize,
    lone: usize,
    chains: usize,
    length: usize,
    tributaries: usize,
) -> CreatureExport {
    let mut neurons = Vec::new();
    let mut synapses = Vec::new();
    for l in 0..lone {
        let uuid = format!("l{l}");
        neurons.push(neuron("hidden", &uuid, 0.0, Some("TANH")));
        synapses.push(synapse(&format!("input-{}", l % inputs), &uuid, 0.5));
        synapses.push(synapse(&uuid, "output-0", 0.1));
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
        synapses.push(synapse(&format!("c{c}_{}", length - 1), "output-0", 0.01));
    }
    for t in 0..tributaries {
        // Feeders are listed before the exit they feed: NEAT-AI-core reads a
        // synapse into an earlier-listed neuron as recursive, and this creature
        // is forward-only.
        let exit = format!("t{t}_exit");
        for feeder in 0..2 {
            let uuid = format!("t{t}_{feeder}");
            neurons.push(neuron("hidden", &uuid, 0.0, Some("TANH")));
            synapses.push(synapse(&format!("input-{}", t % inputs), &uuid, 0.5));
            synapses.push(synapse(&uuid, &exit, 0.5));
        }
        neurons.push(neuron("hidden", &exit, 0.0, Some("TANH")));
        synapses.push(synapse(&exit, "output-0", 0.01));
    }
    neurons.push(neuron("output", "output-0", 0.0, Some("IDENTITY")));
    creature(inputs, 1, neurons, synapses)
}

/// Statistics for every hidden neuron; chain and tributary members are quiet.
fn stats_for(creature: &CreatureExport) -> ActivationStats {
    let mut stats = ActivationStats::empty();
    stats.neurons = creature
        .neurons
        .iter()
        .enumerate()
        .filter(|(_, n)| n.neuron_type == "hidden")
        .map(|(i, n)| {
            let mean_abs = if n.uuid.starts_with('l') { 0.4 } else { 0.05 };
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

/// Structure one transform removed.
#[derive(Default)]
struct Removed {
    proposals: usize,
    hidden: usize,
    synapses: usize,
    growth_units: f64,
}

impl Removed {
    fn per_proposal(&self, value: f64) -> f64 {
        if self.proposals == 0 {
            0.0
        } else {
            value / self.proposals as f64
        }
    }

    fn line(&self, label: &str, ms: f64) -> String {
        format!(
            "  {label:<26} {:3} proposals, {:5} hidden, {:5} synapses, {:7.1} growth units \
             ({:.2} hidden and {:.2} units per proposal, {ms:.1}ms)",
            self.proposals,
            self.hidden,
            self.synapses,
            self.growth_units,
            self.per_proposal(self.hidden as f64),
            self.per_proposal(self.growth_units),
        )
    }
}

fn main() {
    const INPUTS: usize = 6;
    const LONE: usize = 500;
    const CHAINS: usize = 60;
    const LENGTH: usize = 4;
    const TRIBUTARIES: usize = 60;

    let creature = mixed_creature(INPUTS, LONE, CHAINS, LENGTH, TRIBUTARIES);
    let stats = stats_for(&creature);
    let cfg = NeighbourhoodConfig {
        max_size: 4,
        max_proposals: 32,
    };
    println!(
        "creature: {} neurons, {} synapses ({LONE} lone, {CHAINS} chains of {LENGTH}, \
         {TRIBUTARIES} tributaries)",
        creature.neurons.len(),
        creature.synapses.len()
    );

    let started = Instant::now();
    let proposals = propose_neighbourhoods(&creature, &stats, cfg);
    let propose_ms = started.elapsed().as_secs_f64() * 1000.0;
    let chains = proposals
        .iter()
        .filter(|p| p.kind == NeighbourhoodKind::Chain)
        .count();
    println!(
        "proposals: {} bounded neighbourhoods ({chains} chain, {} branch) ranked in \
         {propose_ms:.1}ms",
        proposals.len(),
        proposals.len() - chains,
    );

    let started = Instant::now();
    let batch = group_batch(&creature, &stats, cfg, &HashSet::new());
    let build_ms = started.elapsed().as_secs_f64() * 1000.0;
    if !batch.blocked.is_empty() {
        println!(
            "  {} proposal(s) the transform refused: {}",
            batch.blocked.len(),
            batch.blocked[0]
        );
    }

    // What the group cuts actually remove.
    let mut group = Removed::default();
    for proposal in &proposals {
        let members: Vec<GroupMember> = proposal
            .members
            .iter()
            .filter_map(|uuid| {
                stats.by_uuid(uuid).map(|s| GroupMember {
                    uuid: uuid.clone(),
                    mean: s.mean,
                })
            })
            .collect();
        let Ok(built) = ablate_group(&creature, &members) else {
            continue;
        };
        group.proposals += 1;
        group.hidden += built.before.hidden_neurons - built.after.hidden_neurons;
        group.synapses += built.before.synapses - built.after.synapses;
        group.growth_units += built.before.growth_units - built.after.growth_units;
    }

    // The control: the same neurons cut one at a time, each from the untouched
    // incumbent — which is exactly what the single-neuron sweep would do.
    let started_single = Instant::now();
    let mut single = Removed::default();
    for proposal in &proposals {
        let mut best: Option<(usize, usize, f64)> = None;
        for uuid in &proposal.members {
            let Some(mean) = stats.by_uuid(uuid).map(|s| s.mean) else {
                continue;
            };
            let Ok(built) = ablate_mean(&creature, uuid, mean, None) else {
                continue;
            };
            let removed = (
                built.before.hidden_neurons - built.after.hidden_neurons,
                built.before.synapses - built.after.synapses,
                built.before.growth_units - built.after.growth_units,
            );
            // The best single cut the sweep could have made in this
            // neighbourhood — the fairest control, not the weakest.
            if best.is_none_or(|b| removed.2 > b.2) {
                best = Some(removed);
            }
        }
        if let Some((hidden, synapses, growth_units)) = best {
            single.proposals += 1;
            single.hidden += hidden;
            single.synapses += synapses;
            single.growth_units += growth_units;
        }
    }
    let single_ms = started_single.elapsed().as_secs_f64() * 1000.0;

    println!("structure removed, measured by the real transforms:");
    println!("{}", group.line("group cut", build_ms));
    println!("{}", single.line("best single cut", single_ms));
    let ratio = if single.growth_units > 0.0 {
        group.growth_units / single.growth_units
    } else {
        f64::INFINITY
    };
    println!(
        "  group cuts remove {ratio:.2}x the growth units of the best single cut in the same \
         neighbourhood"
    );
    println!(
        "  candidates built for one batch: {} (stems g000…)",
        batch.candidates.len()
    );
    fidelity();

    println!(
        "\nStructure and fidelity only. Whether any of it should go is the full-corpus\n\
         scorer's call — a run reports accepted group cuts as `groupAccepts`,\n\
         `groupHiddenRemoved` and `groupGrowthUnitsRemovedPerHour` in `neat_ai_ockham report`."
    );
}

/// How far each transform moves the creature's outputs (Issue #108).
///
/// The exact cleanup usually strands a whole chain behind a single cut, so the
/// two transforms remove the *same* structure — what differs is the arithmetic
/// left behind. Cutting the chain head folds its mean into the next neuron and
/// then folds `squash(bias + mean × w)` onward as a constant: the activation at
/// the mean input. The group cut folds each member's **own** measured mean: the
/// mean of the activations. Those are not the same number for any curved
/// squash, and this measures which one lands closer to the creature it
/// replaced.
fn fidelity() {
    const SAMPLES: usize = 401;
    let creature = curved_chain();
    let stats = stats_measured(&creature, SAMPLES);
    let proposals = propose_neighbourhoods(
        &creature,
        &stats,
        NeighbourhoodConfig {
            max_size: 4,
            max_proposals: 1,
        },
    );
    let Some(proposal) = proposals.first() else {
        println!("\nfidelity: nothing proposed on the fidelity fixture");
        return;
    };
    let members: Vec<GroupMember> = proposal
        .members
        .iter()
        .map(|uuid| GroupMember {
            uuid: uuid.clone(),
            mean: stats.by_uuid(uuid).expect("measured").mean,
        })
        .collect();
    let grouped = ablate_group(&creature, &members).expect("the chain must build");
    println!(
        "\nfidelity on one {}-neuron chain, {SAMPLES} inputs in [-2, 2]:",
        proposal.members.len()
    );
    println!(
        "  {:<26} {:.6} mean |Δoutput|, {} hidden removed",
        "group cut",
        mean_abs_deviation(&creature, &grouped.creature, SAMPLES),
        grouped.before.hidden_neurons - grouped.after.hidden_neurons,
    );
    for uuid in &proposal.members {
        let mean = stats.by_uuid(uuid).expect("measured").mean;
        let single = ablate_mean(&creature, uuid, mean, None).expect("member must build");
        println!(
            "  {:<26} {:.6} mean |Δoutput|, {} hidden removed",
            format!("single cut of {uuid}"),
            mean_abs_deviation(&creature, &single.creature, SAMPLES),
            single.before.hidden_neurons - single.after.hidden_neurons,
        );
    }
}

/// `input-0 → f0 → f1 → f2 → output-0`, biased off centre.
///
/// The biases matter: with a zero bias and inputs symmetric about zero,
/// `E[tanh(x)]` and `tanh(E[x])` are both zero and every transform agrees by
/// symmetry. Real creatures are not centred, so the fixture is not either.
fn curved_chain() -> CreatureExport {
    let mut neurons = Vec::new();
    let mut synapses = Vec::new();
    for step in 0..3 {
        let uuid = format!("f{step}");
        neurons.push(neuron("hidden", &uuid, 0.6, Some("TANH")));
        let source = if step == 0 {
            "input-0".to_string()
        } else {
            format!("f{}", step - 1)
        };
        synapses.push(synapse(&source, &uuid, 1.4));
    }
    neurons.push(neuron("output", "output-0", 0.0, Some("IDENTITY")));
    synapses.push(synapse("f2", "output-0", 1.0));
    creature(1, 1, neurons, synapses)
}

/// Inputs the fidelity fixtures are measured over: `SAMPLES` points in [-2, 2].
fn inputs(samples: usize) -> Vec<f32> {
    (0..samples)
        .map(|i| -2.0 + 4.0 * i as f32 / (samples.max(2) - 1) as f32)
        .collect()
}

/// Real measured statistics over [`inputs`], not invented ones.
fn stats_measured(creature: &CreatureExport, samples: usize) -> ActivationStats {
    let mut net = compile_creature(creature).expect("fixture compiles");
    let hidden: Vec<(usize, String)> = creature
        .neurons
        .iter()
        .enumerate()
        .filter(|(_, n)| n.neuron_type == "hidden")
        .map(|(i, n)| (i, n.uuid.clone()))
        .collect();
    let mut sums = vec![(0.0f64, 0.0f64); hidden.len()];
    for x in inputs(samples) {
        net.activate(&[x], 1);
        for (slot, (index, _)) in hidden.iter().enumerate() {
            let value = f64::from(net.activations[net.num_inputs + index]);
            sums[slot].0 += value;
            sums[slot].1 += value.abs();
        }
    }
    let n = samples as f64;
    let mut stats = ActivationStats::empty();
    stats.neurons = hidden
        .iter()
        .enumerate()
        .map(|(slot, (index, uuid))| NeuronStats {
            uuid: uuid.clone(),
            neuron_index: *index,
            count: samples as u64,
            mean: sums[slot].0 / n,
            variance: 0.0,
            std_dev: 0.0,
            mean_abs: sums[slot].1 / n,
            min: -1.0,
            max: 1.0,
        })
        .collect();
    stats
}

/// Mean `|candidate(x) - original(x)|` over [`inputs`].
fn mean_abs_deviation(
    original: &CreatureExport,
    candidate: &CreatureExport,
    samples: usize,
) -> f64 {
    let mut before = compile_creature(original).expect("original compiles");
    let mut after = compile_creature(candidate).expect("candidate compiles");
    let mut total = 0.0;
    for x in inputs(samples) {
        let a = before.activate(&[x], 1)[0];
        let b = after.activate(&[x], 1)[0];
        total += f64::from((a - b).abs());
    }
    total / samples as f64
}
