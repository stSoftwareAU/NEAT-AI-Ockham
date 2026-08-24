//! Benchmark hidden-neuron activation statistics vs inference alone (Issue #3).
//!
//! Reports records/sec for a compiled forward pass with and without the
//! statistics accumulators, on a synthetic stream of 50_000 records.

use std::time::Instant;

use neat_ai_ockham::corpus::{corpus_info, for_each_chunk, write_bin_file};
use neat_ai_ockham::fixtures::hidden_identity_creature;
use neat_ai_ockham::stats::{DEFAULT_CHUNK_RECORDS, compute_activation_stats};
use neat_core::compile_creature;
use neat_core::training_data::TrainingDataConfig;

fn main() {
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
    )
    .unwrap();
    let stats_s = started.elapsed().as_secs_f64().max(1e-9);

    println!(
        "records={N}  inference={:.0} rec/s ({infer_s:.3}s)  with-stats={:.0} rec/s ({stats_s:.3}s)  overhead={:.1}%  hidden-mean={:.6}",
        N as f64 / infer_s,
        N as f64 / stats_s,
        (stats_s / infer_s - 1.0) * 100.0,
        stats.by_uuid("h1").map(|h| h.mean).unwrap_or(0.0)
    );
}
