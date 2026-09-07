//! Benchmark the synapse half of the seeded sweep pool (Issue #135).
//!
//! Every ordinary synapse is a visit, with no weight, magnitude or contribution
//! threshold deciding which edges enter the pool. That makes the pool an order
//! of magnitude larger than the neuron-only one, so the two questions this
//! answers are what a synapse visit *yields* — proposals built, refusals by
//! reason — and what it *costs* per visit on a forest-heavy creature.
//!
//! Nothing here is scored. Every proposal goes through the real
//! [`neat_ai_ockham::Sweep`], the real source-value resolver and the real
//! [`neat_ai_ockham::ablation::ablate_synapse`], recursive cleanup and
//! `creature.validate()` and all, and what those transforms actually removed is
//! what is reported. Whether a candidate is any *good* is the full-corpus
//! scorer's verdict, and only a real run can report that.

use std::collections::HashSet;
use std::time::Instant;

use neat_ai_ockham::blocked::BlockedBreakdown;
use neat_ai_ockham::fixtures::{creature, neuron, synapse, typed_synapse};
use neat_ai_ockham::stats::{ActivationStats, NeuronStats};
use neat_ai_ockham::{CandidateKind, MergeIndex, Sweep, parse_synapse_key};
use neat_core::CreatureExport;

/// A forest-heavy creature: many shallow trees into a shared output layer.
///
/// `trees` roots, each feeding `fan` leaves, each leaf feeding the output —
/// plus one typed `condition` edge per `typed_every`th tree and one aggregate
/// `MEAN` collector per `aggregate_every`th, because a real creature's edges
/// are not all cuttable and the refusal breakdown is half the measurement.
fn forest(
    inputs: usize,
    trees: usize,
    fan: usize,
    typed_every: usize,
    aggregate_every: usize,
) -> CreatureExport {
    let mut neurons = Vec::new();
    let mut synapses = Vec::new();
    for t in 0..trees {
        let root = format!("r{t}");
        neurons.push(neuron("hidden", &root, 0.0, Some("TANH")));
        synapses.push(synapse(&format!("input-{}", t % inputs), &root, 0.5));
        let aggregate = aggregate_every > 0 && t % aggregate_every == 0;
        for f in 0..fan {
            let leaf = format!("l{t}_{f}");
            neurons.push(neuron("hidden", &leaf, 0.0, Some("TANH")));
            synapses.push(synapse(&root, &leaf, 0.3));
            synapses.push(synapse(&leaf, "output-0", 0.1));
        }
        // Declared after the leaves that feed it: a forward-only creature needs
        // every edge to run from a lower index to a higher one.
        if aggregate {
            let collector = format!("m{t}");
            neurons.push(neuron("hidden", &collector, 0.0, Some("MEAN")));
            synapses.push(synapse(&root, &collector, 0.4));
            for f in 0..fan {
                synapses.push(synapse(&format!("l{t}_{f}"), &collector, 0.15));
            }
            synapses.push(synapse(&collector, "output-0", 0.2));
        }
        if typed_every > 0 && t % typed_every == 0 {
            let gate = format!("g{t}");
            neurons.push(neuron("hidden", &gate, 0.0, Some("IF")));
            synapses.push(typed_synapse(&root, &gate, 1.0, "condition"));
            synapses.push(typed_synapse(
                &format!("input-{}", t % inputs),
                &gate,
                1.0,
                "positive",
            ));
            synapses.push(typed_synapse(
                &format!("input-{}", t % inputs),
                &gate,
                -1.0,
                "negative",
            ));
            synapses.push(synapse(&gate, "output-0", 0.1));
        }
    }
    neurons.push(neuron("output", "output-0", 0.0, Some("IDENTITY")));
    creature(inputs, 1, neurons, synapses)
}

/// A measured mean for every hidden neuron and every input wire.
///
/// Flat, deliberately: this benchmark measures the pool's yield and cost, not
/// how well any one fold value happens to approximate a source.
fn stats_for(creature: &CreatureExport) -> ActivationStats {
    let measured = |uuid: &str, index: usize| NeuronStats {
        uuid: uuid.to_string(),
        neuron_index: index,
        count: 1_000,
        mean: 0.25,
        variance: 0.1,
        std_dev: 0.1_f64.sqrt(),
        mean_abs: 0.25,
        min: -1.0,
        max: 1.0,
    };
    let mut stats = ActivationStats::empty();
    stats.neurons = creature
        .neurons
        .iter()
        .enumerate()
        .filter(|(_, n)| n.neuron_type == "hidden")
        .map(|(i, n)| measured(&n.uuid, i))
        .collect();
    stats.inputs = (0..creature.input)
        .map(|i| measured(&format!("input-{i}"), i))
        .collect();
    stats
}

fn main() {
    const INPUTS: usize = 6;
    const TREES: usize = 150;
    const FAN: usize = 4;
    const TYPED_EVERY: usize = 8;
    const AGGREGATE_EVERY: usize = 5;

    let creature = forest(INPUTS, TREES, FAN, TYPED_EVERY, AGGREGATE_EVERY);
    // Fail loud on a fixture the razor could never work on: a benchmark over an
    // invalid incumbent would report refusals that say nothing about the sweep.
    neat_ai_ockham::incumbent::validate_creature(&creature)
        .expect("the benchmark creature must be valid before anything is cut");
    let stats = stats_for(&creature);
    let hidden = creature
        .neurons
        .iter()
        .filter(|n| n.neuron_type == "hidden")
        .count();
    println!(
        "creature: {hidden} hidden neurons, {} synapses ({TREES} trees of {FAN}, \
         a typed gate every {TYPED_EVERY}, an aggregate collector every {AGGREGATE_EVERY})",
        creature.synapses.len()
    );

    let started = Instant::now();
    let mut sweep = Sweep::new(&creature, 20_260_907);
    let order_ms = started.elapsed().as_secs_f64() * 1000.0;
    let synapse_visits = sweep
        .order
        .iter()
        .filter(|v| parse_synapse_key(v).is_some())
        .count();
    println!(
        "pool: {} visits ({} neuron, {synapse_visits} synapse) ordered in {order_ms:.1}ms",
        sweep.order.len(),
        sweep.order.len() - synapse_visits,
    );

    // One pass over the whole pool, exactly as a run fills batches: the cost
    // below is the real cost of proposing, cleanup cascade and validate() and
    // all, not of a dry run that stops at the refusal check.
    let started = Instant::now();
    let mut proposals = 0usize;
    let mut refusals = BlockedBreakdown::default();
    let mut uncoded = 0usize;
    let mut synapses_removed = 0usize;
    let mut neurons_cascaded = 0usize;
    while !sweep.exhausted() {
        let (batch, skips) =
            sweep.fill_batch_avoiding(&creature, &stats, MergeIndex::empty(), 64, &HashSet::new());
        for c in batch.iter().filter(|c| c.kind == CandidateKind::Synapse) {
            proposals += 1;
            synapses_removed += creature.synapses.len() - c.creature.synapses.len();
            neurons_cascaded += creature.neurons.len() - c.creature.neurons.len();
        }
        for skip in skips
            .iter()
            .filter(|s| parse_synapse_key(&s.uuid).is_some())
        {
            match skip.blocked {
                Some(reason) => refusals.add(reason),
                // Counted, never swallowed: a refusal with no reason code is
                // not "no refusal", and the printed total has to match the
                // visits that produced no candidate.
                None => uncoded += 1,
            }
        }
    }
    let visit_ms = started.elapsed().as_secs_f64() * 1000.0;

    println!(
        "synapse proposals: {proposals} built and validated, {} refused",
        refusals.total() + uncoded
    );
    // "Built", never "accepted": nothing here is scored, and only the
    // full-corpus scorer accepts.
    println!(
        "  removed per proposal built: {:.2} synapse(s), {:.2} neuron(s) cascaded",
        synapses_removed as f64 / proposals.max(1) as f64,
        neurons_cascaded as f64 / proposals.max(1) as f64,
    );
    println!("refusals by reason:");
    for (reason, count) in refusals.entries() {
        println!("  {:<20} {count:>6}  {}", reason.code(), reason.describe());
    }
    if uncoded > 0 {
        println!(
            "  {:<20} {uncoded:>6}  refused with no reason code",
            "(uncoded)"
        );
    }
    println!(
        "cost: {visit_ms:.1}ms over {synapse_visits} synapse visit(s) — \
         {:.3}ms per visit, {:.3}ms per proposal built",
        visit_ms / synapse_visits.max(1) as f64,
        visit_ms / proposals.max(1) as f64,
    );
}
