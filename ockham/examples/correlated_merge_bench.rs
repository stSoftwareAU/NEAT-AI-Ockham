//! Benchmark correlated-neuron merging against the ablation control (#109).
//!
//! Two questions, and the harness answers both on the same creature:
//!
//! 1. **Does the discovery find the duplicates?** The creature carries planted
//!    exact twins, planted near-twins and a crowd of unrelated neurons. Every
//!    probe vector is measured with the real NEAT-AI-core forward pass, so the
//!    signatures are the ones a run would build, not ones written to be found.
//! 2. **Do the proposals survive judging?** Each proposal is turned into a real
//!    candidate by [`merge_correlated`], screened on a *subset* of the probes
//!    and then judged on all of them — the sampled-screen-then-full-scorer shape
//!    the razor actually runs, standing in for a corpus and a scorer this
//!    harness has neither of.
//!
//! The proxy judge is deliberately strict: a cut is *confirmed* only when every
//! probe output is unchanged within [`TOLERANCE`]. That is the right bar for a
//! merge, whose whole claim is that the survivor already carries the removed
//! neuron's behaviour. The run-level economics come from `report` on a real
//! run; the same four measures are printed here so the two read side by side.
//!
//! A third table times discovery alone as the creature grows, which is the
//! scaling claim Issue #109 asks for.
//!
//! ```text
//! cargo run --release --example correlated_merge_bench
//! ```

use std::time::Instant;

use neat_ai_ockham::fixtures::{creature, neuron, synapse};
use neat_ai_ockham::signature::{DiscoveryConfig, discover};
use neat_ai_ockham::stats::{ActivationStats, NeuronProbes};
use neat_ai_ockham::{ablate_mean, merge_correlated};
use neat_core::{CreatureExport, compile_creature};

/// Outputs must stay within this of the incumbent for the judge to confirm.
const TOLERANCE: f32 = 1e-5;
/// Probe records the signatures and the full judge both use.
const PROBES: usize = 64;
/// Leading probes the cheap screen looks at — the sampled screen's stand-in.
const SCREEN_PROBES: usize = 16;

/// A creature mixing three populations the discovery has to tell apart.
///
/// - `twin{i}_a` / `twin{i}_b` — identical bias, identical incoming weights,
///   identical squash. One of the two is dead weight, exactly.
/// - `near{i}_a` / `near{i}_b` — the same shape with every incoming weight
///   nudged by 2%: strongly correlated, not identical, and the case the scorer
///   rather than the razor has to settle.
/// - `solo{i}` — ordinary neurons on their own weight vectors.
///
/// Every hidden neuron reads **every** input on its own weights. That matters:
/// a fixture where each neuron reads one input makes every neuron a monotone
/// function of that input, so the whole creature correlates and the benchmark
/// would measure the fixture rather than the discovery.
fn duplicated_creature(inputs: usize, twins: usize, near: usize, solo: usize) -> CreatureExport {
    let mut neurons = Vec::with_capacity(2 * twins + 2 * near + solo + 1);
    let mut synapses = Vec::new();
    let push =
        |uuid: &str, seed: usize, jitter: f64, w_out: f64, ns: &mut Vec<_>, ss: &mut Vec<_>| {
            let weights = pseudo_random(seed);
            ns.push(neuron("hidden", uuid, f64::from(weights[0]), Some("TANH")));
            for i in 0..inputs {
                let w = f64::from(weights[1 + i % (PROBES - 1)]) * 4.0 * jitter;
                ss.push(synapse(&format!("input-{i}"), uuid, w));
            }
            ss.push(synapse(uuid, "output-0", w_out));
        };
    for i in 0..twins {
        push(
            &format!("twin{i}_a"),
            i,
            1.0,
            0.9,
            &mut neurons,
            &mut synapses,
        );
        push(
            &format!("twin{i}_b"),
            i,
            1.0,
            -0.4,
            &mut neurons,
            &mut synapses,
        );
    }
    for i in 0..near {
        let seed = twins + i;
        push(
            &format!("near{i}_a"),
            seed,
            1.0,
            1.2,
            &mut neurons,
            &mut synapses,
        );
        push(
            &format!("near{i}_b"),
            seed,
            1.02,
            0.5,
            &mut neurons,
            &mut synapses,
        );
    }
    for i in 0..solo {
        let uuid = format!("solo{i}");
        push(
            &uuid,
            twins + near + i,
            1.0,
            0.25,
            &mut neurons,
            &mut synapses,
        );
    }
    neurons.push(neuron("output", "output-0", 0.0, Some("IDENTITY")));
    creature(inputs, 1, neurons, synapses)
}

/// Fixed probe inputs — the same records for every measurement here.
fn probe_inputs(inputs: usize) -> Vec<Vec<f32>> {
    (0..PROBES)
        .map(|record| {
            (0..inputs)
                .map(|i| ((record * 11 + i * 17) % 29) as f32 / 7.0 - 2.0)
                .collect()
        })
        .collect()
}

/// Outputs of `creature` over every probe; a compile fault is never swallowed.
///
/// `merge_correlated` only returns a candidate it has validated, so one that
/// will not compile is a fault to surface — not a quiet "not confirmed" that
/// reads in the table exactly like a judge rejecting the cut on behaviour.
fn outputs(creature: &CreatureExport, probes: &[Vec<f32>]) -> Vec<Vec<f32>> {
    let mut net =
        compile_creature(creature).unwrap_or_else(|e| panic!("candidate must compile: {e}"));
    probes
        .iter()
        .map(|input| net.activate(input, creature.output))
        .collect()
}

/// Whether every one of the first `records` probe outputs is within tolerance.
fn agrees(before: &[Vec<f32>], after: &[Vec<f32>], records: usize) -> bool {
    before.len() >= records
        && after.len() >= records
        && before[..records]
            .iter()
            .zip(&after[..records])
            .all(|(b, a)| {
                b.len() == a.len() && b.iter().zip(a).all(|(x, y)| (x - y).abs() <= TOLERANCE)
            })
}

/// Activation statistics whose probe vectors are the real forward pass.
fn measured_stats(creature: &CreatureExport, probes: &[Vec<f32>]) -> ActivationStats {
    let mut net = compile_creature(creature).expect("incumbent must compile");
    let hidden: Vec<(String, usize)> = creature
        .neurons
        .iter()
        .enumerate()
        .filter(|(_, n)| n.neuron_type == "hidden")
        .map(|(i, n)| (n.uuid.clone(), net.num_inputs + i))
        .collect();
    let mut values: Vec<Vec<f32>> = vec![Vec::with_capacity(probes.len()); hidden.len()];
    for input in probes {
        let _ = net.activate(input, creature.output);
        for (slot, (_, index)) in values.iter_mut().zip(&hidden) {
            slot.push(net.activations[*index]);
        }
    }
    ActivationStats {
        probes: hidden
            .into_iter()
            .zip(values)
            .map(|((uuid, _), values)| NeuronProbes { uuid, values })
            .collect(),
        ..ActivationStats::empty()
    }
}

/// One judged proposal set.
///
/// `neurons_removed` / `synapses_removed` count each **unordered pair** once.
/// Both survivor directions of one pair are proposed and both can confirm, but
/// only one of the two neurons is actually redundant — summing per proposal
/// would report twice the saving the creature could ever make.
#[derive(Default)]
struct Tally {
    proposals: usize,
    candidates: usize,
    /// Proposals the transform refused, by reason — never dropped in silence.
    refused: std::collections::BTreeMap<String, usize>,
    screened: usize,
    confirmed: usize,
    neurons_removed: usize,
    synapses_removed: usize,
    judge_ms: f64,
    counted: std::collections::HashSet<(String, String)>,
}

/// The unordered key for a confirmed cut, so a pair is counted once.
fn pair_key(a: &str, b: &str) -> (String, String) {
    if a <= b {
        (a.into(), b.into())
    } else {
        (b.into(), a.into())
    }
}

impl Tally {
    fn rate(part: usize, whole: usize) -> String {
        if whole == 0 {
            // Nothing offered means nothing survived; a percentage of zero
            // would be a division by zero wearing a measurement's costume.
            return "none".into();
        }
        format!("{:.0}%", 100.0 * part as f64 / whole as f64)
    }

    fn per_hour(&self, n: usize) -> f64 {
        let hours = self.judge_ms / 3_600_000.0;
        if hours > 0.0 { n as f64 / hours } else { 0.0 }
    }

    fn print(&self, label: &str) {
        println!(
            "  {:<14} {:>10} {:>11} {:>9} {:>10} {:>11} {:>10} {:>10}",
            label,
            self.proposals,
            self.candidates,
            Self::rate(self.screened, self.candidates),
            Self::rate(self.confirmed, self.candidates),
            format!("{:.0}", self.per_hour(self.confirmed)),
            self.neurons_removed,
            self.synapses_removed,
        );
        // A proposal the transform refused is why `candidates` is below
        // `proposals`; printing the reasons is the difference between a
        // measured gap and an unexplained one.
        if !self.refused.is_empty() {
            let by_reason: Vec<String> = self
                .refused
                .iter()
                .map(|(reason, n)| format!("{reason} {n}"))
                .collect();
            println!("    {label} refused {}", by_reason.join(", "));
        }
    }
}

fn main() {
    const INPUTS: usize = 6;
    const TWINS: usize = 40;
    const NEAR: usize = 40;
    const SOLO: usize = 400;

    let incumbent = duplicated_creature(INPUTS, TWINS, NEAR, SOLO);
    let probes = probe_inputs(INPUTS);
    let baseline = outputs(&incumbent, &probes);
    let hidden = incumbent
        .neurons
        .iter()
        .filter(|n| n.neuron_type == "hidden")
        .count();
    println!(
        "creature: {hidden} hidden neurons, {} synapses ({TWINS} exact twin pairs, {NEAR} near \
         pairs, {SOLO} unrelated)",
        incumbent.synapses.len()
    );

    let stats = measured_stats(&incumbent, &probes);
    let cfg = DiscoveryConfig::default();
    let index = discover(&stats, cfg);
    let report = index.report();
    println!(
        "discovery: {} proposal(s) in {}ms — {} signed, {} pair(s) compared, {} above |r|>={} \
         (bands of {} bits)",
        report.proposals,
        report.discovery_ms,
        report.signed,
        report.compared_pairs,
        report.correlated_pairs,
        cfg.min_correlation,
        report.band_bits,
    );
    if report.truncated_buckets > 0 {
        println!(
            "discovery: {} bucket(s) truncated, {} member(s) not compared — the table below is \
             that much of a lower bound",
            report.truncated_buckets, report.dropped_members
        );
    }

    // Merge proposals, judged.
    let mut merge = Tally {
        proposals: report.proposals,
        ..Tally::default()
    };
    let mut ablation = Tally::default();
    let judging = Instant::now();
    for proposal in index.proposals() {
        let built = match merge_correlated(
            &incumbent,
            &proposal.survivor_uuid,
            &proposal.removed_uuid,
            proposal.relation,
        ) {
            Ok(built) => built,
            Err(skip) => {
                *merge
                    .refused
                    .entry(skip.blocked_reason().code().into())
                    .or_default() += 1;
                continue;
            }
        };
        merge.candidates += 1;
        let after = outputs(&built.creature, &probes);
        if agrees(&baseline, &after, SCREEN_PROBES) {
            merge.screened += 1;
        }
        if agrees(&baseline, &after, PROBES) {
            merge.confirmed += 1;
            if merge
                .counted
                .insert(pair_key(&proposal.survivor_uuid, &proposal.removed_uuid))
            {
                merge.neurons_removed += built.before.hidden_neurons - built.after.hidden_neurons;
                merge.synapses_removed += built.before.synapses - built.after.synapses;
            }
        }
    }
    merge.judge_ms = judging.elapsed().as_secs_f64() * 1000.0;

    // The control: the same neurons, cut the way the razor cuts them today.
    let control_uuids: Vec<String> = index
        .proposals()
        .iter()
        .map(|p| p.removed_uuid.clone())
        .collect();
    ablation.proposals = control_uuids.len();
    let judging = Instant::now();
    for uuid in &control_uuids {
        let mean = stats
            .probes_of(uuid)
            .map(|v| f64::from(v.iter().sum::<f32>()) / v.len() as f64)
            .unwrap_or(0.0);
        let built = match ablate_mean(&incumbent, uuid, mean, None) {
            Ok(built) => built,
            Err(blocked) => {
                *ablation
                    .refused
                    .entry(blocked.blocked_reason().code().into())
                    .or_default() += 1;
                continue;
            }
        };
        ablation.candidates += 1;
        let after = outputs(&built.creature, &probes);
        if agrees(&baseline, &after, SCREEN_PROBES) {
            ablation.screened += 1;
        }
        if agrees(&baseline, &after, PROBES) {
            ablation.confirmed += 1;
            if ablation.counted.insert(pair_key(uuid, uuid)) {
                ablation.neurons_removed +=
                    built.before.hidden_neurons - built.after.hidden_neurons;
                ablation.synapses_removed += built.before.synapses - built.after.synapses;
            }
        }
    }
    ablation.judge_ms = judging.elapsed().as_secs_f64() * 1000.0;

    println!("\nthe same neurons, cut two ways, judged by a compiled forward pass:");
    println!(
        "  {:<14} {:>10} {:>11} {:>9} {:>10} {:>11} {:>10} {:>10}",
        "transform",
        "proposals",
        "candidates",
        "screened",
        "confirmed",
        "confirmed/h",
        "neurons",
        "synapses"
    );
    merge.print("merge");
    ablation.print("ablation");
    println!(
        "screened is the share of candidates the {SCREEN_PROBES}-probe screen kept; confirmed is \
         the share the full {PROBES}-probe judge confirmed; neurons and synapses are what the \
         confirmed cuts removed, counting each pair once — both survivor directions confirm, but \
         only one neuron of the two is redundant."
    );

    println!("\ndiscovery cost as the creature grows (signatures only, no scoring):");
    println!(
        "  {:>9} {:>9} {:>10} {:>14} {:>11} {:>10}",
        "hidden", "bandBits", "buckets", "pairsCompared", "proposals", "ms"
    );
    for scale in [1_000usize, 2_000, 4_000, 8_000] {
        let stats = synthetic_stats(scale);
        let started = Instant::now();
        let index = discover(&stats, cfg);
        let ms = started.elapsed().as_secs_f64() * 1000.0;
        let r = index.report();
        println!(
            "  {scale:>9} {:>9} {:>10} {:>14} {:>11} {ms:>10.1}",
            r.band_bits, r.buckets, r.compared_pairs, r.proposals
        );
    }

    // The same measurement on a **real** compiled creature of several thousand
    // hidden neurons, so the scaling claim covers the probe capture and the
    // forward pass behind it rather than the signature pass alone.
    println!("\nthe same on real compiled creatures, probe capture included:");
    println!(
        "  {:>9} {:>11} {:>10} {:>14} {:>11} {:>12}",
        "hidden", "synapses", "probe(ms)", "pairsCompared", "proposals", "discover(ms)"
    );
    for (twins, solo) in [(100usize, 900usize), (250, 2_250), (500, 4_500)] {
        let big = duplicated_creature(INPUTS, twins, 0, solo);
        let hidden = big
            .neurons
            .iter()
            .filter(|n| n.neuron_type == "hidden")
            .count();
        let started = Instant::now();
        let big_stats = measured_stats(&big, &probes);
        let probe_ms = started.elapsed().as_secs_f64() * 1000.0;
        let started = Instant::now();
        let big_index = discover(&big_stats, cfg);
        let discover_ms = started.elapsed().as_secs_f64() * 1000.0;
        let r = big_index.report();
        println!(
            "  {hidden:>9} {:>11} {probe_ms:>10.1} {:>14} {:>11} {discover_ms:>12.1}",
            big.synapses.len(),
            r.compared_pairs,
            r.proposals
        );
    }
    println!(
        "pairsCompared is the only quadratic term, and the band widens with the creature to hold \
         it near linear; every neuron beyond a bucket's cap is reported, never dropped quietly."
    );
}

/// `n` neurons with distinct pseudo-random probe vectors, plus planted twins.
///
/// The scaling table measures the discovery method rather than a creature, so
/// the vectors are generated instead of measured — but one pair in every
/// hundred is an exact duplicate, so a run that found nothing would show it.
fn synthetic_stats(n: usize) -> ActivationStats {
    let mut probes: Vec<NeuronProbes> = (0..n)
        .map(|i| NeuronProbes {
            uuid: format!("h{i:06}"),
            values: pseudo_random(i),
        })
        .collect();
    for i in (0..n).step_by(100) {
        probes.push(NeuronProbes {
            uuid: format!("twin{i:06}"),
            values: probes[i].values.clone(),
        });
    }
    ActivationStats {
        probes,
        ..ActivationStats::empty()
    }
}

/// A distinct probe vector for `index`, from a fixed seed.
fn pseudo_random(index: usize) -> Vec<f32> {
    let mut state = (index as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
    (0..PROBES)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((state >> 40) as f32) / 16_777_216.0 - 0.5
        })
        .collect()
}
