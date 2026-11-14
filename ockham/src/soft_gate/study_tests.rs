use super::super::manifest::Limits;
use super::*;
use crate::ablation::StructureSnapshot;
use crate::corpus::for_each_chunk;
use crate::fixtures::wide_creature;
use crate::scorer::{ScoreResult, ScorerError};

struct CoreScorer;
impl DirectoryScorer for CoreScorer {
    fn score_directory(
        &self,
        creatures: &Path,
        data: &Path,
        mode: ScorerMode,
    ) -> Result<BTreeMap<String, ScoreResult>, ScorerError> {
        assert_eq!(mode, ScorerMode::Full);
        let mut scores = BTreeMap::new();
        for entry in std::fs::read_dir(creatures).unwrap() {
            let path = entry.unwrap().path();
            let parent = load_incumbent(&path).unwrap();
            let cfg = TrainingDataConfig::new(parent.creature.input, parent.creature.output);
            let mut net = compile_creature(&parent.creature).unwrap();
            let mut sum = 0.0;
            let n = for_each_chunk(data, &cfg, 256, |chunk| {
                for i in 0..chunk.records {
                    let predicted = net.activate(
                        &chunk.inputs[i * cfg.num_inputs..(i + 1) * cfg.num_inputs],
                        cfg.num_outputs,
                    );
                    sum += predicted
                        .iter()
                        .zip(&chunk.targets[i * cfg.num_outputs..(i + 1) * cfg.num_outputs])
                        .map(|(a, b)| (f64::from(*a) - f64::from(*b)).powi(2))
                        .sum::<f64>()
                        / cfg.num_outputs as f64;
                }
                Ok(())
            })
            .unwrap();
            let error = sum / n as f64;
            let penalty = StructureSnapshot::of(&parent.creature).growth_units * 0.0001;
            scores.insert(
                path.file_stem().unwrap().to_str().unwrap().into(),
                ScoreResult {
                    score: 1.0 - error - penalty,
                    error,
                    complexity_penalty: penalty,
                    record_count: n,
                    sample_rate: None,
                    gpu_backend: None,
                    cost_name: Some("MSE".into()),
                    time_taken: 0.0,
                },
            );
        }
        Ok(scores)
    }
    fn identity(&self) -> String {
        "test-core-MSE".into()
    }
}

#[test]
fn sampled_or_partial_results_cannot_authorise_a_prune() {
    let full = CoreScorer;
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().join("creatures");
    let data = tmp.path().join("data");
    std::fs::create_dir(&dir).unwrap();
    std::fs::create_dir(&data).unwrap();
    std::fs::write(
        dir.join("c0.json"),
        neat_core::creature_to_json(&wide_creature(1, 4, "TANH")).unwrap(),
    )
    .unwrap();
    write_bin_file(&data.join("0.bin"), &vec![(vec![0.2], vec![0.1]); 4]).unwrap();
    let mut scores = full.score_directory(&dir, &data, ScorerMode::Full).unwrap();
    validate_full(&scores, ScorerMode::Full, 4, 1).unwrap();
    assert!(
        validate_full(
            &scores,
            ScorerMode::Sample {
                rate: 0.5,
                phase: 0
            },
            4,
            1
        )
        .unwrap_err()
        .contains("sampled")
    );
    scores.get_mut("c0").unwrap().record_count = 2;
    assert!(
        validate_full(&scores, ScorerMode::Full, 4, 1)
            .unwrap_err()
            .contains("record count")
    );
    scores.get_mut("c0").unwrap().record_count = 4;
    scores.get_mut("c0").unwrap().sample_rate = Some(0.5);
    assert!(
        validate_full(&scores, ScorerMode::Full, 4, 1)
            .unwrap_err()
            .contains("sampled")
    );
    scores.clear();
    assert!(
        validate_full(&scores, ScorerMode::Full, 4, 1)
            .unwrap_err()
            .contains("missing c0")
    );
}

#[test]
fn calibration_selection_is_isolated_from_validation_targets() {
    let root = tempfile::tempdir().unwrap();
    let data = root.path().join("data");
    std::fs::create_dir(&data).unwrap();
    let rows = (0..16)
        .map(|i| (vec![i as f32 / 16.0], vec![0.2]))
        .collect::<Vec<_>>();
    write_bin_file(&data.join("0.bin"), &rows).unwrap();
    let cfg = TrainingDataConfig::new(1, 1);
    let ranges = [RecordRange { start: 0, len: 8 }];
    let a = load_rows(&data, &cfg, &ranges).unwrap();
    let mut changed = rows;
    for row in &mut changed[12..] {
        row.1[0] = 999.0;
    }
    write_bin_file(&data.join("0.bin"), &changed).unwrap();
    let b = load_rows(&data, &cfg, &ranges).unwrap();
    let parent = wide_creature(1, 4, "TANH");
    let limits = Limits {
        record_evaluations: 1_000_000,
        compilations: 100_000,
        rewrites: 1000,
        io_bytes: 1_000_000,
        seconds: 60.0,
    };
    let config = super::super::manifest::Fit {
        steps: 2,
        learning_rate: 0.1,
        temperature: 0.2,
        difference_step: 0.001,
    };
    let fit_a = fit::fit(
        &parent,
        &a,
        &config,
        7,
        &mut Budget::new(&limits, &Cost::default()).unwrap(),
    )
    .unwrap();
    let fit_b = fit::fit(
        &parent,
        &b,
        &config,
        7,
        &mut Budget::new(&limits, &Cost::default()).unwrap(),
    )
    .unwrap();
    assert_eq!(fit_a.log_scores, fit_b.log_scores);
    assert_eq!(fit_a.removal_order, fit_b.removal_order);
    let held = load_rows(&data, &cfg, &[RecordRange { start: 12, len: 4 }]).unwrap();
    assert_eq!(held[0].targets, [999.0]);
}

#[test]
fn real_core_rewrites_and_full_judgement_complete_both_search_paths() {
    let root = tempfile::tempdir().unwrap();
    let data = root.path().join("data");
    std::fs::create_dir(&data).unwrap();
    let parent_path = root.path().join("parent.json");
    let text = neat_core::creature_to_json(&wide_creature(1, 20, "TANH")).unwrap();
    std::fs::write(&parent_path, &text).unwrap();
    write_bin_file(
        &data.join("0.bin"),
        &(0..32)
            .map(|i| (vec![i as f32 / 32.0], vec![0.1]))
            .collect::<Vec<_>>(),
    )
    .unwrap();
    let path = root.path().join("manifest.json");
    super::super::prepare_manifest(
        &parent_path,
        &data,
        &std::env::current_exe().unwrap(),
        &path,
        16,
        8,
    )
    .unwrap();
    let mut manifest: Manifest = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
    manifest.fit.steps = 2;
    manifest.independent_trials_per_budget = 2;
    manifest.latency_records = 2;
    manifest.latency_repetitions = 3;
    let cfg = TrainingDataConfig::new(1, 1);
    let corpus = corpus_info(&data, &cfg).unwrap();
    let parent = load_incumbent(&parent_path).unwrap();
    let rows = load_rows(&data, &cfg, &manifest.calibration).unwrap();
    let mut arms = Vec::new();
    for name in ["gates", "independent"] {
        let dir = root.path().join(name);
        std::fs::create_dir(&dir).unwrap();
        let mut budget = Budget::new(&manifest.limits_per_family, &Cost::default()).unwrap();
        let arm = run_arm(
            &manifest,
            &parent,
            &corpus,
            &rows,
            &ActivationStats::empty(),
            &CoreScorer,
            name,
            7,
            &dir,
            None,
            &mut budget,
        )
        .unwrap();
        assert_eq!(
            arm.measurements
                .iter()
                .map(|m| m.candidate.size.hidden_neurons)
                .collect::<Vec<_>>(),
            [20, 18, 15, 10]
        );
        assert_eq!(
            std::fs::read_to_string(dir.join("candidates/c0.json")).unwrap(),
            text
        );
        assert_eq!(arm.measurements[0].full.record_count, 32);
        for m in &arm.measurements[1..] {
            assert_eq!(
                m.authoritative_improvement,
                m.full.score > arm.measurements[0].full.score
            );
        }
        if name == "gates" {
            for pair in arm.measurements.windows(2) {
                assert!(
                    pair[1]
                        .candidate
                        .retained_original_hidden
                        .is_subset(&pair[0].candidate.retained_original_hidden)
                );
            }
        } else {
            assert_eq!(arm.search_trials, [2, 2, 2]);
        }
        arms.push(arm);
    }
    assert_eq!(std::fs::read_to_string(&parent_path).unwrap(), text);
    assert_eq!(
        paired_summary(&arms, &manifest, true)["verdict"],
        "inconclusive"
    );

    // Statistical decisions use supplied measurements, never wall-clock
    // assertions in tests. The production measurements above exercise wiring.
    manifest.representative = true;
    manifest.seeds = (0..10).collect();
    let mut paired = Vec::new();
    for &seed in &manifest.seeds {
        for template in &arms {
            let gate = template.name == "gates";
            let measurements = template.measurements.iter().map(|m| {
                let mut full = m.full.clone(); full.score = if gate { 0.91 } else { 0.90 };
                Measurement {
                    candidate: m.candidate.clone(), calibration_mse: 0.1, validation_mse: 0.1,
                    full, authoritative_improvement: false, latency_ns_per_record: Vec::new(),
                    median_latency_ns: if gate { 9.0 } else { 10.0 },
                    pareto_error_growth: true, pareto_error_latency: true,
                }
            }).collect();
            paired.push(Arm { name: template.name.clone(), seed, ranking: None,
                search_trials: Vec::new(), measurements, cost: Cost { elapsed_seconds: 100.0, ..Cost::default() },
                construction_seconds: 1.0 });
        }
    }
    assert_eq!(paired_summary(&paired, &manifest, true)["verdict"], "retain");
    paired[0].cost.elapsed_seconds = 50.0;
    assert_eq!(paired_summary(&paired, &manifest, true)["criteria"]["matchedCost"], false);
    paired[0].cost.elapsed_seconds = 100.0;
    paired[0].measurements[2].validation_mse = 0.2;
    assert_eq!(paired_summary(&paired, &manifest, true)["criteria"]["heldOutError"], false);
    assert_eq!(paired_summary(&paired, &manifest, true)["verdict"], "does-not-meet-retention-criteria");
    paired[0].measurements[2].validation_mse = 0.1;
    paired[1].measurements[1].candidate.capacity_overshoot = 0.1;
    assert_eq!(paired_summary(&paired, &manifest, true)["criteria"]["capacityOvershoot"], false);
    paired[1].measurements[1].candidate.capacity_overshoot = 0.0;
    paired[0].measurements[3].candidate.reachable = false;
    assert_eq!(paired_summary(&paired, &manifest, true)["criteria"]["reachableBudgets"], false);
    paired[0].measurements[3].candidate.reachable = true;
    paired[0].measurements[2].median_latency_ns = 10.0;
    assert_eq!(paired_summary(&paired, &manifest, true)["criteria"]["latencyReduction"], false);
    paired[0].measurements[2].median_latency_ns = 9.0;
    assert_eq!(paired_summary(&paired, &manifest, false)["criteria"]["completePairs"], false);
    assert!(paired_summary(&paired[..18], &manifest, true)["conservative95Interval"].is_null());
    assert_eq!(paired_summary(&paired, &manifest, true)["verdict"], "retain");

    let points = &mut paired[0].measurements;
    for (i, m) in points.iter_mut().enumerate() {
        m.full.error = [0.1, 0.2, 0.1, 0.1][i];
        m.candidate.size.growth_units = [20.0, 18.0, 15.0, 10.0][i];
        m.median_latency_ns = [20.0, 18.0, 15.0, 10.0][i];
    }
    mark_frontiers(points);
    assert_eq!(points.iter().map(|m| m.pareto_error_growth).collect::<Vec<_>>(), [false, false, false, true]);
    assert_eq!(points.iter().map(|m| m.pareto_error_latency).collect::<Vec<_>>(), [false, false, false, true]);
}
