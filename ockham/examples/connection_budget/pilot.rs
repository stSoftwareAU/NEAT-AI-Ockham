use std::path::Path;

use neat_ai_ockham::corpus::write_bin_file;
use neat_ai_ockham::fixtures::{creature, neuron, synapse};
use neat_core::compile_creature;
use sha2::{Digest, Sha256};

use super::manifest::{Latency, Limits, Spec, StudyKind, Subject, Success, write_new};

fn draw(seed: usize, row: usize, column: usize) -> f64 {
    let mut hasher = Sha256::new();
    hasher.update(b"connection-budget-synthetic-v1");
    for value in [seed, row, column] {
        hasher.update((value as u64).to_le_bytes());
    }
    let digest = hasher.finalize();
    f64::from(u32::from_le_bytes(digest[..4].try_into().unwrap())) / f64::from(u32::MAX) * 2.0 - 1.0
}

pub fn specification(subjects: Vec<Subject>, scorer: &Path) -> Spec {
    Spec {
        format_version: 1, study_kind: StudyKind::SyntheticPilot,
        independence_statement: "Independent generated teacher networks, not evolved creatures or representative task data; seeds are repeated measures within a teacher.".into(),
        expected_independent_units: subjects.len(), creatures: subjects, seeds: vec![7, 19], connection_targets: vec![30, 34],
        scorer: scorer.into(), scorer_args: vec!["--gpu".into(), "off".into(), "--cost".into(), "MSE".into()],
        limits: Limits { proposal_attempts: 200, statistics_records: 10_000, scorer_calls: 24,
            scorer_record_evaluations: 100_000, search_seconds: 30, call_timeout_seconds: 10, max_corpus_records: 10_000 },
        success: Success { minimum_independent_units: 8, minimum_matched_fraction: 0.8, minimum_score_gain: 1e-6,
            maximum_worst_output_mse_increase: 1e-4, compute_relative_tolerance: 0.1, bootstrap_replicates: 2000 },
        latency: Latency { records: 128, warmup_passes: 2, measured_passes: 10 }, screen_rate: 0.5, min_improvement: 1e-12,
    }
}

pub fn write(directory: &Path, scorer: &Path) -> Result<(), String> {
    std::fs::create_dir(directory).map_err(|e| format!("new pilot directory: {e}"))?;
    let mut subjects = Vec::new();
    for seed in 0..8 {
        let dir = directory.join(format!("teacher-{seed}"));
        std::fs::create_dir(&dir).map_err(|e| e.to_string())?;
        let mut neurons = Vec::new();
        let mut edges = Vec::new();
        for h in 0..4 {
            neurons.push(neuron(
                "hidden",
                &format!("h{h}"),
                draw(seed, h, 11) * 0.1,
                Some("TANH"),
            ));
            for input in 0..4 {
                edges.push(synapse(
                    &format!("input-{input}"),
                    &format!("h{h}"),
                    draw(seed, h, input),
                ));
            }
        }
        for h in 4..6 {
            neurons.push(neuron("hidden", &format!("h{h}"), 0.0, Some("TANH")));
            for source in 0..4 {
                edges.push(synapse(
                    &format!("h{source}"),
                    &format!("h{h}"),
                    draw(seed, h, source) * if source >= 2 { 1e-6 } else { 1.0 },
                ));
            }
        }
        for output in 0..2 {
            let uuid = format!("output-{output}");
            neurons.push(neuron("output", &uuid, 0.0, Some("IDENTITY")));
            for source in 4..6 {
                edges.push(synapse(
                    &format!("h{source}"),
                    &uuid,
                    draw(seed, output + 10, source),
                ));
            }
            for input in 0..4 {
                edges.push(synapse(
                    &format!("input-{input}"),
                    &uuid,
                    draw(seed, output + 20, input) * 1e-6,
                ));
            }
        }
        let parent = creature(4, 2, neurons, edges);
        let mut teacher = parent.clone();
        for edge in &mut teacher.synapses {
            if edge.weight.abs() < 1e-4 {
                edge.weight = 0.0;
            }
        }
        let mut net = compile_creature(&teacher).map_err(|e| e.to_string())?;
        let rows: Vec<_> = (0..384)
            .map(|row| {
                let input: Vec<_> = (0..4).map(|i| draw(seed + 100, row, i) as f32).collect();
                let target = net.activate(&input, 2);
                (input, target)
            })
            .collect();
        let creature_path = dir.join("parent.json");
        write_new(&creature_path, &parent)?;
        let training = dir.join("training");
        let calibration = dir.join("calibration");
        let validation = dir.join("validation");
        for path in [&training, &calibration, &validation] {
            std::fs::create_dir(path).map_err(|e| e.to_string())?;
        }
        for (path, data) in [
            (&training, &rows[..256]),
            (&calibration, &rows[..64]),
            (&validation, &rows[256..]),
        ] {
            write_bin_file(&path.join("records.bin"), data).map_err(|e| e.to_string())?;
        }
        subjects.push(Subject { id: format!("teacher-{seed}"), independent_unit: format!("teacher-{seed}"),
            provenance: "Generated by connection_budget pilot v1; a teacher plus deliberately weak extra connections, without evolution or fitting.".into(),
            creature: creature_path, training, calibration, validation });
    }
    write_new(
        &directory.join("spec-v1.json"),
        &specification(subjects, scorer),
    )
}
