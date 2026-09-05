//! Benchmark composite and learned ordering against the controls (Issue #107).
//!
//! What an ordering is worth is the scorer-verified pruning it buys per hour of
//! scorer time, so this simulates one Ockham budget per strategy and reports
//! the four economics the issue asks for:
//!
//! - **time to first cut** — scorer seconds spent before the first confirmed win;
//! - **confirmed cuts per hour**;
//! - **growth units removed per hour**;
//! - **missed-winner rate** — confirmable neurons the budget never reached.
//!
//! The scorer is **simulated**, and deliberately so: a real one needs a corpus
//! and a binary, and the question here is only which order reaches the winners
//! first. The simulation declares its ground truth up front — a quiet neuron
//! (`mean_abs < QUIET`) is confirmable, except that one in ten is not, and one
//! loud neuron in twenty is confirmable anyway. The disagreement matters: a
//! ground truth that *is* one of the ranking signals would score that signal
//! against itself, and a real scorer is never that obliging.
//!
//! Every strategy is scored against the same truth, the same costs and the same
//! creature. What this can show is a ranking's discovery rate; what it cannot
//! show is whether a real scorer agrees, which is why `random` stays the
//! default until real runs say otherwise.
//!
//! The growth units are **not** the ranking key: every visited neuron goes
//! through the real [`ablate_mean`], recursive cleanup and all, and what that
//! transform actually removes is what is summed.

use std::time::Instant;

use neat_ai_ockham::ablate_mean;
use neat_ai_ockham::features::{CandidateFeatures, PriorEvidence};
use neat_ai_ockham::fixtures::{creature, neuron, synapse};
use neat_ai_ockham::model::{PriorityModel, TrainingConfig, TrainingRow};
use neat_ai_ockham::ordering::{Ordering, OrderingConfig, hidden_order};
use neat_ai_ockham::priority::PriorityContext;
use neat_ai_ockham::stats::{ActivationStats, NeuronStats};
use neat_core::CreatureExport;

/// Mean absolute activation below which the simulated scorer confirms a cut.
const QUIET: f64 = 0.1;
/// Simulated sampled-screen cost per candidate, in scorer milliseconds.
const SCREEN_MS: f64 = 40.0;
/// Simulated full-corpus cost per promoted candidate.
const FULL_MS: f64 = 800.0;
/// Simulated scorer budget, in milliseconds.
///
/// Deliberately smaller than the whole sweep: an ordering only matters when the
/// budget cannot reach every neuron, which is the condition the fleet actually
/// runs in — thousands of hidden neurons against one wall clock. Give a run
/// enough time to visit everything and every ordering scores the same, because
/// order stops mattering once nothing is left out.
const BUDGET_MS: f64 = 5.0 * 60.0 * 1000.0;

/// Lone neurons plus chains whose members only come out together.
///
/// Quiet and loud neurons are mixed through both shapes, and a loud one carries
/// the heavy outgoing weight it earned, so no single signal — quietness,
/// fan-out or cascade size — separates the winners on its own.
fn mixed_creature(inputs: usize, lone: usize, chains: usize, length: usize) -> CreatureExport {
    let mut neurons = Vec::new();
    let mut synapses = Vec::new();
    let hidden = |neurons: &mut Vec<_>, uuid: &str| {
        neurons.push(neuron("hidden", uuid, 0.0, Some("TANH")));
    };
    for l in 0..lone {
        let uuid = format!("l{l}");
        hidden(&mut neurons, &uuid);
        synapses.push(synapse(&format!("input-{}", l % inputs), &uuid, 0.5));
        synapses.push(synapse(&uuid, "output-0", outgoing_weight(&uuid)));
        synapses.push(synapse(&uuid, "output-1", outgoing_weight(&uuid)));
    }
    for c in 0..chains {
        for step in 0..length {
            let uuid = format!("c{c}_{step}");
            hidden(&mut neurons, &uuid);
            let source = if step == 0 {
                format!("input-{}", c % inputs)
            } else {
                format!("c{c}_{}", step - 1)
            };
            synapses.push(synapse(&source, &uuid, outgoing_weight(&source)));
        }
        let last = format!("c{c}_{}", length - 1);
        synapses.push(synapse(&last, "output-0", outgoing_weight(&last)));
    }
    neurons.push(neuron("output", "output-0", 0.0, Some("IDENTITY")));
    neurons.push(neuron("output", "output-1", 0.0, Some("IDENTITY")));
    creature(inputs, 2, neurons, synapses)
}

/// Whether this neuron is one the simulated scorer would let go.
///
/// Keyed by uuid rather than by position, so the creature's weights and the
/// activation statistics agree without either having to be built first.
fn quiet(uuid: &str) -> bool {
    unit(fnv(uuid)) < 0.2
}

/// FNV-1a over the uuid — a stable seed, not a cryptographic hash.
fn fnv(uuid: &str) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in uuid.as_bytes() {
        h ^= *byte as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01B3);
    }
    h
}

/// Mean absolute activation the simulated corpus would measure.
fn mean_abs(uuid: &str) -> f64 {
    if quiet(uuid) {
        0.01 + unit(fnv(uuid) ^ 7) * 0.05
    } else {
        0.4 + unit(fnv(uuid) ^ 13)
    }
}

/// Outgoing weight: a neuron the network leans on carries a heavy edge.
fn outgoing_weight(uuid: &str) -> f64 {
    if uuid.starts_with("input") {
        0.5
    } else if quiet(uuid) {
        0.05
    } else {
        1.0 + unit(fnv(uuid) ^ 29) * 2.0
    }
}

/// Deterministic pseudo-random in `[0, 1)` — SplitMix64, no crate needed.
fn unit(seed: u64) -> f64 {
    let mut z = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64
}

/// Activation statistics: a fifth of the neurons are quiet, spread evenly.
fn stats_for(creature: &CreatureExport) -> ActivationStats {
    let mut stats = ActivationStats::empty();
    stats.neurons = creature
        .neurons
        .iter()
        .enumerate()
        .filter(|(_, n)| n.neuron_type == "hidden")
        .map(|(i, n)| {
            let mean_abs = mean_abs(&n.uuid);
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

/// The simulated ground truth: a quiet neuron is usually a confirmable cut.
///
/// One quiet neuron in ten is not, and one loud neuron in twenty is — no signal
/// the ranking reads separates the winners perfectly, which is the only
/// condition under which a ranking benchmark says anything.
fn confirmable(stats: &ActivationStats, uuid: &str) -> bool {
    let Some(measured) = stats.by_uuid(uuid) else {
        return false;
    };
    let noise = unit(fnv(uuid) ^ 97);
    if measured.mean_abs < QUIET {
        noise >= 0.1
    } else {
        noise < 0.05
    }
}

/// Whether the sampled screen promotes this candidate to full scoring.
///
/// Every true winner survives, and one loud candidate in ten is a sampled false
/// positive — the cost a ranking pays for spending a visit on the wrong neuron.
fn promoted(stats: &ActivationStats, uuid: &str, index: usize) -> bool {
    confirmable(stats, uuid) || unit(index as u64 + 101) < 0.1
}

/// What one strategy earned inside the simulated budget.
struct Economics {
    order_ms: f64,
    visits: usize,
    cuts: usize,
    growth_units: f64,
    first_cut_ms: Option<f64>,
    /// Confirmable neurons the budget never turned into a cut.
    missed: usize,
    /// Visits the razor could propose nothing for.
    refused: usize,
}

impl Economics {
    fn per_hour(&self, total: f64) -> f64 {
        total * 3_600_000.0 / BUDGET_MS
    }
}

/// Spend `BUDGET_MS` of simulated scorer time following `order`.
fn simulate(creature: &CreatureExport, stats: &ActivationStats, order: &[String]) -> Economics {
    let mut spent = 0.0;
    let mut economics = Economics {
        order_ms: 0.0,
        visits: 0,
        cuts: 0,
        growth_units: 0.0,
        first_cut_ms: None,
        missed: 0,
        refused: 0,
    };
    for (index, uuid) in order.iter().enumerate() {
        if spent + SCREEN_MS > BUDGET_MS {
            break;
        }
        spent += SCREEN_MS;
        economics.visits += 1;
        if !promoted(stats, uuid, index) {
            continue;
        }
        if spent + FULL_MS > BUDGET_MS {
            break;
        }
        spent += FULL_MS;
        if !confirmable(stats, uuid) {
            continue;
        }
        // A confirmed cut: what the razor really removes is what it is worth.
        // A refusal is counted, never swallowed: a visit the razor can propose
        // nothing for is a cost the ranking paid and bought nothing with.
        match ablate_mean(creature, uuid, 0.1, None) {
            Ok(ablation) => {
                economics.cuts += 1;
                economics.growth_units +=
                    ablation.before.growth_units - ablation.after.growth_units;
                economics.first_cut_ms.get_or_insert(spent);
            }
            Err(_) => economics.refused += 1,
        }
    }
    // Every confirmable neuron the budget did not turn into a cut is missed,
    // whether it was never reached or reached too late to be scored.
    let winners = order.iter().filter(|uuid| confirmable(stats, uuid)).count();
    economics.missed = winners.saturating_sub(economics.cuts);
    economics
}

/// Training and held-out rows from a previous random-ordering run.
///
/// This is the offline pipeline exactly as an operator would run it: an earlier
/// sweep wrote its candidate log, and the model is fitted from those outcomes
/// before it ranks anything. The two blocks are disjoint prefixes of that run's
/// order, so the reported quality is measured on rows the fit never saw —
/// evaluating on the training rows would flatter the model with a number the
/// `train-ordering` command deliberately refuses to present as held out.
fn history(
    creature: &CreatureExport,
    stats: &ActivationStats,
    rows: usize,
    holdout: usize,
) -> (Vec<TrainingRow>, Vec<TrainingRow>) {
    let order = hidden_order(creature, stats, OrderingConfig::default(), 7);
    let features = neat_ai_ockham::features::extract(creature, stats, &PriorEvidence::new());
    let mut all: Vec<TrainingRow> = order
        .iter()
        .take(rows + holdout)
        .filter_map(|uuid| {
            let f: &CandidateFeatures = features.get(uuid)?;
            Some(TrainingRow {
                features: f.vector(),
                win: confirmable(stats, uuid),
            })
        })
        .collect();
    let held = all.split_off(all.len().min(rows));
    (all, held)
}

fn main() {
    const INPUTS: usize = 6;
    const LONE: usize = 1_500;
    const CHAINS: usize = 150;
    const LENGTH: usize = 5;

    let creature = mixed_creature(INPUTS, LONE, CHAINS, LENGTH);
    let stats = stats_for(&creature);
    let hidden = stats.neurons.len();
    let winners = stats.neurons.iter().filter(|n| n.mean_abs < QUIET).count();
    println!(
        "creature: {} neurons, {} synapses ({LONE} lone, {CHAINS} chains of {LENGTH})",
        creature.neurons.len(),
        creature.synapses.len()
    );
    println!(
        "simulated scorer: {hidden} hidden, {winners} confirmable; screen {SCREEN_MS}ms, \
         full {FULL_MS}ms, budget {:.0} minutes",
        BUDGET_MS / 60_000.0
    );

    let (training, holdout) = history(&creature, &stats, 400, 200);
    let model = PriorityModel::fit(&training, TrainingConfig::default())
        .expect("a previous run's outcomes fit a model");
    let evaluation = model.evaluate(&holdout);
    println!(
        "learned model: {} training row(s), {} held-out row(s), {} held-out win(s), \
         held-out AUC {:.3}",
        training.len(),
        holdout.len(),
        evaluation.wins,
        evaluation.auc
    );
    let learned = PriorityContext::with(PriorEvidence::new(), Some(model));

    println!(
        "\n{:<26} {:>10} {:>12} {:>14} {:>9} {:>8} {:>9}",
        "--ordering", "first cut", "cuts/hour", "units/hour", "missed", "refused", "order ms"
    );
    // Every named ordering, so the comparison is against each existing strategy
    // individually rather than a chosen few.
    for strategy in Ordering::ALL.iter().copied() {
        let cfg = OrderingConfig::new(strategy).with_priority(&learned);
        let started = Instant::now();
        let order = hidden_order(&creature, &stats, cfg, 42);
        let order_ms = started.elapsed().as_secs_f64() * 1000.0;
        let mut economics = simulate(&creature, &stats, &order);
        economics.order_ms = order_ms;
        let first = economics
            .first_cut_ms
            .map(|ms| format!("{:.1}s", ms / 1000.0))
            .unwrap_or_else(|| "none".into());
        println!(
            "{:<26} {first:>10} {:>12.1} {:>14.1} {:>8.1}% {:>8} {:>9.1}",
            strategy.name(),
            economics.per_hour(economics.cuts as f64),
            economics.per_hour(economics.growth_units),
            100.0 * economics.missed as f64 / winners.max(1) as f64,
            economics.refused,
            economics.order_ms,
        );
    }
    println!(
        "\nEvery strategy visits every neuron eventually, and every cut above was \
         confirmed by the simulated scorer — a ranking only chooses what to test first."
    );
}
