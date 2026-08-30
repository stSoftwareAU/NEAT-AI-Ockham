//! Benchmark hidden-neuron activation statistics (Issues #3, #44).
//!
//! Two measurements:
//!
//! 1. accumulator overhead — a compiled forward pass with and without the
//!    statistics accumulators (#3);
//! 2. full-corpus scan vs the sampled scan on a wider creature, with the
//!    largest hidden-mean error the sample introduces (#44).

use std::time::Instant;

use neat_ai_ockham::corpus::{corpus_info, for_each_chunk, write_bin_file};
use neat_ai_ockham::fixtures::{creature, hidden_identity_creature, neuron, synapse};
use neat_ai_ockham::stats::{DEFAULT_CHUNK_RECORDS, SampleSpec, compute_activation_stats};
use neat_core::training_data::TrainingDataConfig;
use neat_core::{CreatureExport, compile_creature};

/// `inputs`-in, one-out creature with `hidden` fully-connected hidden neurons.
fn wide_creature(inputs: usize, hidden: usize) -> CreatureExport {
    let mut neurons = Vec::with_capacity(hidden + 1);
    let mut synapses = Vec::with_capacity(hidden * (inputs + 1));
    for h in 0..hidden {
        let uuid = format!("h{h}");
        neurons.push(neuron("hidden", &uuid, (h % 7) as f64 * 0.01, Some("TANH")));
        for i in 0..inputs {
            let weight = ((h * inputs + i) % 13) as f64 * 0.05 - 0.3;
            synapses.push(synapse(&format!("input-{i}"), &uuid, weight));
        }
        synapses.push(synapse(&uuid, "output-0", 1.0 / hidden as f64));
    }
    neurons.push(neuron("output", "output-0", 0.0, Some("IDENTITY")));
    creature(inputs, 1, neurons, synapses)
}

fn accumulator_overhead() {
    let tmp = tempfile::tempdir().expect("tempdir");
    const N: u64 = 50_000;
    let recs: Vec<(Vec<f32>, Vec<f32>)> = (0..N)
        .map(|i| {
            let x = (i % 97) as f32 / 97.0;
            (vec![x], vec![x])
        })
        .collect();
    write_bin_file(&tmp.path().join("0.bin"), &recs).unwrap();
    let creature = hidden_identity_creature(0.1, 0.5);
    let cfg = TrainingDataConfig::new(1, 1);
    let corpus = corpus_info(tmp.path(), &cfg).unwrap();

    let mut net = compile_creature(&creature).unwrap();
    let started = Instant::now();
    for_each_chunk(tmp.path(), &cfg, DEFAULT_CHUNK_RECORDS, |chunk| {
        for r in 0..chunk.records {
            let _ = net.activate(&chunk.inputs[r..r + 1], 1);
        }
        Ok(())
    })
    .unwrap();
    let infer_s = started.elapsed().as_secs_f64().max(1e-9);

    let started = Instant::now();
    let stats = compute_activation_stats(
        &creature,
        "bench",
        tmp.path(),
        &corpus,
        DEFAULT_CHUNK_RECORDS,
        &SampleSpec::full(),
    )
    .unwrap();
    let stats_s = started.elapsed().as_secs_f64().max(1e-9);

    println!(
        "accumulator overhead: records={N}  inference={:.0} rec/s ({infer_s:.3}s)  with-stats={:.0} rec/s ({stats_s:.3}s)  overhead={:.1}%  hidden-mean={:.6}",
        N as f64 / infer_s,
        N as f64 / stats_s,
        (stats_s / infer_s - 1.0) * 100.0,
        stats.by_uuid("h1").map(|h| h.mean).unwrap_or(0.0)
    );
}

fn full_vs_sampled() {
    const RECORDS: u64 = 500_000;
    const INPUTS: usize = 16;
    const HIDDEN: usize = 256;
    let tmp = tempfile::tempdir().expect("tempdir");
    let cfg = TrainingDataConfig::new(INPUTS, 1);
    // Five files, so the sampled plan also exercises seeking across files.
    let per_file = RECORDS / 5;
    for f in 0..5u64 {
        let recs: Vec<(Vec<f32>, Vec<f32>)> = (0..per_file)
            .map(|i| {
                let n = f * per_file + i;
                let inputs = (0..INPUTS)
                    .map(|k| (((n as usize * 31 + k * 17) % 1009) as f32 / 1009.0) - 0.5)
                    .collect();
                (inputs, vec![0.0])
            })
            .collect();
        write_bin_file(&tmp.path().join(format!("{f}.bin")), &recs).unwrap();
    }
    let corpus = corpus_info(tmp.path(), &cfg).unwrap();
    let creature = wide_creature(INPUTS, HIDDEN);

    let started = Instant::now();
    let full = compute_activation_stats(
        &creature,
        "bench",
        tmp.path(),
        &corpus,
        DEFAULT_CHUNK_RECORDS,
        &SampleSpec::full(),
    )
    .unwrap();
    let full_s = started.elapsed().as_secs_f64().max(1e-9);

    let spec = SampleSpec::default();
    let started = Instant::now();
    let sampled = compute_activation_stats(
        &creature,
        "bench",
        tmp.path(),
        &corpus,
        DEFAULT_CHUNK_RECORDS,
        &spec,
    )
    .unwrap();
    let sampled_s = started.elapsed().as_secs_f64().max(1e-9);

    let worst_mean_error = full
        .neurons
        .iter()
        .zip(&sampled.neurons)
        .map(|(f, s)| (f.mean - s.mean).abs())
        .fold(0.0f64, f64::max);
    let worst_rank_signal = full
        .neurons
        .iter()
        .zip(&sampled.neurons)
        .map(|(f, s)| (f.mean_abs - s.mean_abs).abs())
        .fold(0.0f64, f64::max);

    println!(
        "full vs sampled: corpus={RECORDS} records, hidden={HIDDEN}\n  \
         full   : {:>9} records  {full_s:.3}s  ({:.0} rec/s)\n  \
         sampled: {:>9} records  {sampled_s:.3}s  ({:.0} rec/s)  stoppedEarly={}\n  \
         speed-up={:.1}x  worst |Δmean|={worst_mean_error:.3e}  worst |Δmean_abs|={worst_rank_signal:.3e}",
        full.record_count,
        full.record_count as f64 / full_s,
        sampled.record_count,
        sampled.record_count as f64 / sampled_s,
        sampled.stopped_early,
        full_s / sampled_s,
    );
}

fn main() {
    accumulator_overhead();
    full_vs_sampled();
}
