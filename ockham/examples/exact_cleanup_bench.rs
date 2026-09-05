//! What the exact cleanup pre-pass costs and what it saves (Issue #110).
//!
//! The pre-pass removes structure that is *provably* redundant, so the honest
//! comparison is against what discovering the same structure statistically
//! would have cost: a sampled screen per candidate, then a full-corpus score
//! per candidate that survives it.
//!
//! Two halves, kept apart on purpose:
//!
//! * **Measured** — the pre-pass itself. Real [`canonicalise`] over real
//!   creatures, timed with the wall clock, including its validations.
//! * **Modelled** — the scorer. Cost is linear in records read
//!   (`creatures × rate × corpus`), which is what a record-streaming scorer
//!   actually costs. A corpus large enough to make the comparison meaningful
//!   cannot live in a benchmark fixture, so it is modelled rather than faked.
//!
//! The modelled arm is deliberately generous to the statistical route: it
//! assumes every dead neuron is proposed once, screened once, and confirmed by
//! exactly one full score — no re-screens, no rejected candidates, no bundles.
//!
//! Run it with:
//!
//! ```bash
//! cargo run --release --example exact_cleanup_bench
//! ```

use std::time::Instant;

use neat_ai_ockham::ablation::StructureSnapshot;
use neat_ai_ockham::canonical::canonicalise;
use neat_ai_ockham::fixtures::{creature, neuron, synapse};
use neat_core::CreatureExport;

/// Records in the modelled training corpus.
const CORPUS_RECORDS: f64 = 2_000_000.0;
/// Records the modelled scorer reads per millisecond, per creature.
const RECORDS_PER_MS: f64 = 20_000.0;
/// Sampled screen rate, mirroring `--screen-sample-rate`.
const SCREEN_RATE: f64 = 0.05;

/// Modelled scorer milliseconds for `creatures` scored at `rate` of the corpus.
fn scorer_ms(creatures: f64, rate: f64) -> f64 {
    creatures * rate * CORPUS_RECORDS / RECORDS_PER_MS
}

/// A creature carrying `live` working hidden neurons plus provable dead wood.
///
/// * `passthrough` hidden `IDENTITY` neurons on the path input → output, each
///   collapsible exactly;
/// * `zero` branches whose only outgoing synapse carries weight exactly `0.0`,
///   each stranding its own neuron the moment that synapse goes.
///
/// The creature is valid as built — NEAT-AI-core refuses a hidden neuron with
/// no inward or outward connection, so dead wood is what an exact rewrite
/// exposes, never what a valid incumbent already carries.
fn seeded_creature(live: usize, passthrough: usize, zero: usize) -> CreatureExport {
    let inputs = 8;
    let mut neurons = Vec::new();
    let mut synapses = Vec::new();
    for h in 0..live {
        let uuid = format!("live{h}");
        neurons.push(neuron("hidden", &uuid, 0.01, Some("TANH")));
        for i in 0..inputs {
            synapses.push(synapse(&format!("input-{i}"), &uuid, 0.1));
        }
        synapses.push(synapse(&uuid, "output-0", 1.0 / live.max(1) as f64));
    }
    for h in 0..passthrough {
        let uuid = format!("pass{h}");
        neurons.push(neuron("hidden", &uuid, 0.0, Some("IDENTITY")));
        synapses.push(synapse("input-0", &uuid, 0.5));
        synapses.push(synapse(&uuid, "output-0", 0.25));
    }
    for h in 0..zero {
        let uuid = format!("zero{h}");
        neurons.push(neuron("hidden", &uuid, 0.2, Some("TANH")));
        synapses.push(synapse("input-1", &uuid, 0.7));
        synapses.push(synapse(&uuid, "output-0", 0.0));
    }
    neurons.push(neuron("output", "output-0", 0.0, Some("IDENTITY")));
    creature(inputs, 1, neurons, synapses)
}

fn main() {
    println!("Exact cleanup pre-pass — cost measured, scorer work modelled (Issue #110)");
    println!(
        "modelled corpus {:.0} records at {:.0} records/ms, screen rate {SCREEN_RATE}",
        CORPUS_RECORDS, RECORDS_PER_MS
    );
    println!();
    println!(
        "{:>7} {:>7} {:>7} | {:>9} {:>10} {:>8} | {:>12} {:>12}",
        "live", "ident", "zero", "hidden↓", "growth↓", "pass ms", "screen+full", "saved"
    );
    println!("{}", "-".repeat(96));

    let mut total_measured_ms = 0.0;
    let mut total_modelled_ms = 0.0;
    for &(live, passthrough, zero) in &[
        (50usize, 25usize, 25usize),
        (200, 100, 100),
        (1_000, 250, 250),
        (2_000, 500, 500),
    ] {
        let incumbent = seeded_creature(live, passthrough, zero);
        let before = StructureSnapshot::of(&incumbent);
        let started = Instant::now();
        let done = canonicalise(&incumbent).expect("the pre-pass must not fail");
        let measured_ms = started.elapsed().as_secs_f64() * 1_000.0;
        let after = &done.report.after;
        let hidden_removed = before.hidden_neurons - after.hidden_neurons;

        // What the statistical route would have spent to find the same
        // neurons: one sampled screen each, then one full score each.
        let modelled_ms =
            scorer_ms(hidden_removed as f64, SCREEN_RATE) + scorer_ms(hidden_removed as f64, 1.0);
        total_measured_ms += measured_ms;
        total_modelled_ms += modelled_ms;

        println!(
            "{live:>7} {passthrough:>7} {zero:>7} | {hidden_removed:>9} {:>10.1} {measured_ms:>8.1} | \
             {:>12.0} {:>11.0}×",
            done.report.growth_units_saved,
            modelled_ms,
            if measured_ms > 0.0 {
                modelled_ms / measured_ms
            } else {
                f64::INFINITY
            },
        );
        assert!(
            done.report.rejected.is_empty(),
            "no rewrite should have been rolled back: {:?}",
            done.report.rejected
        );
    }

    println!();
    println!(
        "total: {total_measured_ms:.1}ms measured pre-pass work replaced \
         {:.0}ms ({:.1} hours) of modelled scorer work",
        total_modelled_ms,
        total_modelled_ms / 3_600_000.0
    );
    println!(
        "the pre-pass spends no candidate or full score of its own: the run's \
         authoritative baseline is the single scorer pass over the result"
    );
}
