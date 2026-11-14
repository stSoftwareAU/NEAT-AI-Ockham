use std::collections::BTreeMap;
use std::io::Read;
use std::ops::ControlFlow;
use std::path::Path;
use std::time::Instant;

use neat_core::{compile_creature, training_data::TrainingDataConfig};
use serde::Serialize;
use serde_json::{Value, json};

use crate::corpus::{CorpusInfo, RecordRange, corpus_info, for_each_selected_chunk, write_bin_file};
use crate::incumbent::{Incumbent, load_incumbent, sha256_hex};
use crate::scorer::{DirectoryScorer, ExternalScorer, ScorerMode};
use crate::stats::{ActivationStats, SampleSpec, compute_activation_stats};

use super::family::{Candidate, materialise};
use super::fit::{self, Ranking, Row};
use super::manifest::{Manifest, corpus_digest, file_digest, write_new};
use super::{Budget, Cost};

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct Measurement {
    candidate: Candidate,
    calibration_mse: f64,
    validation_mse: f64,
    full: crate::scorer::ScoreResult,
    authoritative_improvement: bool,
    latency_ns_per_record: Vec<f64>,
    median_latency_ns: f64,
    pareto_error_growth: bool,
    pareto_error_latency: bool,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct Arm {
    name: String,
    seed: u64,
    ranking: Option<Ranking>,
    search_trials: Vec<usize>,
    measurements: Vec<Measurement>,
    cost: Cost,
    construction_seconds: f64,
}

/// Execute an explicitly hashed predeclaration into a new private directory.
/// No candidate is installed into the ordinary optimiser or population.
///
/// # Errors
/// Invalid or changed inputs are refused before output creation. Runtime faults
/// write `results.json` with failures, then return an error (non-zero at the CLI).
pub fn run_manifest(path: &Path, expected_digest: &str, out: &Path) -> Result<Value, String> {
    let started = Instant::now();
    let mut bytes = Vec::new();
    std::fs::File::open(path).map_err(|e| e.to_string())?.take(1_048_577).read_to_end(&mut bytes).map_err(|e| e.to_string())?;
    if bytes.len() > 1_048_576 { return Err("manifest exceeds 1 MiB".into()); }
    if sha256_hex(&bytes) != expected_digest { return Err("manifest checksum mismatch".into()); }
    let manifest: Manifest = serde_json::from_slice(&bytes).map_err(|e| format!("invalid manifest: {e}"))?;
    let parent = load_incumbent(&manifest.creature.path).map_err(|e| e.to_string())?;
    fit::supported(&parent.creature)?;
    let cfg = TrainingDataConfig::new(parent.creature.input, parent.creature.output);
    let corpus = corpus_info(&manifest.corpus.path, &cfg)?;
    manifest.validate(&corpus)?;
    let checked_bytes = verify_inputs(&manifest)?;
    std::fs::create_dir(out).map_err(|e| format!("create new experiment directory {}: {e}", out.display()))?;
    write_new(&out.join("manifest.json"), &bytes)?;
    let mut report = json!({
        "formatVersion": 1, "status": "failed", "manifestSha256": expected_digest,
        "method": manifest.method, "hardware": manifest.hardware,
        "representative": manifest.representative, "dataProvenance": manifest.data_provenance,
        "corpusRecords": corpus.record_count, "pairedSeedsRequested": manifest.seeds,
        "arms": [], "failedArms": [], "failures": [],
        "exclusions": if manifest.representative { Vec::<String>::new() } else { vec!["No representative-data claim; smoke study only".into()] },
        "retentionDecision": "undecided-no-completed-representative-comparison",
        "acceptance": "diagnostic candidates only; no best.json or population entry",
        "limitsPerFourBudgetFamily": manifest.limits_per_family,
    });
    let common = Cost {
        io_bytes: manifest.preparation_bytes.saturating_add(checked_bytes).saturating_add(bytes.len() as u64),
        preparation_seconds: manifest.preparation_seconds + started.elapsed().as_secs_f64(),
        ..Cost::default()
    };
    let result = execute(&manifest, &parent, &corpus, out, common, &mut report);
    if let Err(ref error) = result {
        report["failures"].as_array_mut().ok_or("internal failures array missing")?.push(json!(error));
    }
    write_new(&out.join("results.json"), &serde_json::to_vec_pretty(&report).map_err(|e| e.to_string())?)?;
    result.map(|()| report)
}

fn verify_inputs(manifest: &Manifest) -> Result<u64, String> {
    let mut bytes = 0;
    for (label, input) in [("creature", &manifest.creature), ("scorer", &manifest.scorer)] {
        let (digest, n) = file_digest(&input.path)?;
        if digest != input.sha256 { return Err(format!("{label} checksum mismatch")); }
        bytes += n;
    }
    let (digest, n) = corpus_digest(&manifest.corpus.path)?;
    if digest != manifest.corpus.sha256 { return Err("corpus checksum mismatch".into()); }
    bytes += n;
    let (digest, n) = file_digest(&std::env::current_exe().map_err(|e| e.to_string())?)?;
    if digest != manifest.executable_sha256 { return Err("experiment executable checksum mismatch: prepare with this binary".into()); }
    Ok(bytes + n)
}

fn load_rows(data: &Path, cfg: &TrainingDataConfig, ranges: &[RecordRange]) -> Result<Vec<Row>, String> {
    let mut rows = Vec::new();
    let expected: u64 = ranges.iter().map(|r| r.len).sum();
    let read = for_each_selected_chunk(data, cfg, ranges, 4096, |chunk| {
        for i in 0..chunk.records {
            let inputs = chunk.inputs[i * cfg.num_inputs..(i + 1) * cfg.num_inputs].to_vec();
            let targets = chunk.targets[i * cfg.num_outputs..(i + 1) * cfg.num_outputs].to_vec();
            if inputs.iter().chain(&targets).any(|x| !x.is_finite()) {
                return Err(format!("non-finite data at record {}", chunk.first_index + i as u64));
            }
            rows.push(Row { inputs, targets });
        }
        Ok(ControlFlow::Continue(()))
    })?;
    if read != expected { return Err(format!("split read {read} records, expected {expected}")); }
    Ok(rows)
}

fn execute(manifest: &Manifest, parent: &Incumbent, corpus: &CorpusInfo, out: &Path, mut common: Cost, report: &mut Value) -> Result<(), String> {
    let shared_started = Instant::now();
    let cfg = TrainingDataConfig::new(parent.creature.input, parent.creature.output);
    let rows = load_rows(&manifest.corpus.path, &cfg, &manifest.calibration)?;
    let calibration_dir = out.join("calibration");
    std::fs::create_dir(&calibration_dir).map_err(|e| e.to_string())?;
    write_bin_file(&calibration_dir.join("0.bin"), &rows.iter().map(|r| (r.inputs.clone(), r.targets.clone())).collect::<Vec<_>>()).map_err(|e| e.to_string())?;
    let calibration_info = corpus_info(&calibration_dir, &cfg)?;
    common.record_evaluations = rows.len() as u64;
    common.compilations = 1;
    common.io_bytes += (rows.len() * cfg.bytes_per_record() * 3) as u64;
    Budget::new(&manifest.limits_per_family, &common)?;
    let stats = compute_activation_stats(&parent.creature, &parent.checksum, &calibration_dir, &calibration_info, 4096, &SampleSpec::full())?;
    common.preparation_seconds += shared_started.elapsed().as_secs_f64();
    let scorer = ExternalScorer { binary: manifest.scorer.path.clone(), extra_args: vec!["--gpu".into(), "off".into(), "--cost".into(), "MSE".into()] };
    let mut arms = Vec::new();
    let mut failures = Vec::new();
    for &seed in &manifest.seeds {
        // The control's construction allowance is determined by the paired
        // gate family's measured construction cost, as predeclared in v1.
        let names = ["gates", "independent"];
        let mut matched_construction_seconds = None;
        for name in names {
            let mut budget = Budget::new(&manifest.limits_per_family, &common)?;
            let dir = out.join(format!("seed-{seed}-{name}"));
            std::fs::create_dir(&dir).map_err(|e| e.to_string())?;
            match run_arm(manifest, parent, corpus, &rows, &stats, &scorer, name, seed, &dir, matched_construction_seconds, &mut budget) {
                Ok(mut arm) => {
                    if name == "gates" { matched_construction_seconds = Some(arm.construction_seconds); }
                    match budget.finish() {
                        Ok(cost) => {
                            arm.cost = cost;
                            arms.push(arm);
                            report["arms"] = serde_json::to_value(&arms).map_err(|e| e.to_string())?;
                        }
                        Err(e) => failures.push(format!("seed {seed} {name}: {e}")),
                    }
                }
                Err(e) => {
                    report["failedArms"].as_array_mut().ok_or("internal failed arms array missing")?.push(json!({ "seed": seed, "arm": name, "cost": budget.cost,
                        "activeSeconds": budget.started.elapsed().as_secs_f64(), "cause": e }));
                    failures.push(format!("seed {seed} {name}: {e}"));
                }
            }
        }
    }
    // Any mid-run mutation invalidates the comparison, including held-out data.
    let post_started = Instant::now();
    let post_bytes = verify_inputs(manifest)?;
    let post_seconds = post_started.elapsed().as_secs_f64();
    for arm in &mut arms {
        arm.cost.io_bytes = arm.cost.io_bytes.saturating_add(post_bytes);
        arm.cost.elapsed_seconds += post_seconds;
        if arm.cost.io_bytes > manifest.limits_per_family.io_bytes || arm.cost.elapsed_seconds > manifest.limits_per_family.seconds {
            failures.push(format!("seed {} {}: family budget exhausted by final integrity check", arm.seed, arm.name));
        }
    }
    report["arms"] = serde_json::to_value(&arms).map_err(|e| e.to_string())?;
    report["pairedSummary"] = paired_summary(&arms, manifest, failures.is_empty());
    report["coverage"] = json!({ "completedFamilies": arms.len(), "requestedFamilies": 2 * manifest.seeds.len(),
        "calibrationRecords": rows.len(), "validationRecords": manifest.validation.iter().map(|r| r.len).sum::<u64>(),
        "representativeCreatures": usize::from(manifest.representative), "supportedActivations": ["IDENTITY", "LOGISTIC", "TANH"],
        "excluded": "other activations, typed edges, recurrence; untested data is not evidence" });
    report["failures"] = json!(failures);
    if !failures.is_empty() { return Err("one or more paired families failed; see results.json".into()); }
    report["status"] = json!("completed");
    report["retentionDecision"] = if report["pairedSummary"]["verdict"] == "retain" {
        json!("retain-opt-in-experiment")
    } else if manifest.representative {
        json!("remove-optional-implementation-after-completed-negative-study")
    } else { json!("undecided-no-representative-data") };
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn run_arm(manifest: &Manifest, parent: &Incumbent, corpus: &CorpusInfo, rows: &[Row], stats: &ActivationStats, scorer: &dyn DirectoryScorer, name: &str, seed: u64, dir: &Path, matched_construction_seconds: Option<f64>, budget: &mut Budget) -> Result<Arm, String> {
    let construction_started = Instant::now();
    let (candidates, ranking, trials) = if name == "gates" {
        let ranking = fit::fit(&parent.creature, rows, &manifest.fit, seed, budget)?;
        let candidates = materialise(&parent.creature, &ranking.removal_order, &manifest.targets, stats, budget)?;
        (candidates, Some(ranking), Vec::new())
    } else {
        let mut candidates = materialise(&parent.creature, &[], &[1.0], stats, budget)?;
        let mut trials = Vec::new();
        let final_records = final_record_allowance(manifest, corpus.record_count)?;
        let available = budget.limits.record_evaluations.saturating_sub(budget.cost.record_evaluations).saturating_sub(final_records);
        let per_budget = available / 3;
        let per_rewrite = budget.limits.rewrites.saturating_sub(budget.cost.rewrites) / 3;
        let per_compile = budget.limits.compilations.saturating_sub(budget.cost.compilations).saturating_sub(16) / 3;
        for (i, &target) in manifest.targets[1..].iter().enumerate() {
            let before = budget.cost.clone();
            let target_started = Instant::now();
            let mut best: Option<(f64, Candidate)> = None;
            let mut count = 0;
            for trial in 0..manifest.independent_trials_per_budget {
                if count > 0 && matched_construction_seconds.is_some_and(|seconds| target_started.elapsed().as_secs_f64() >= seconds / 3.0) { break; }
                // Reserve final scoring and diagnostics before starting another trial.
                if budget.cost.record_evaluations - before.record_evaluations + rows.len() as u64 > per_budget
                    || budget.cost.rewrites - before.rewrites + stats.neurons.len() as u64 > per_rewrite
                    || budget.cost.compilations - before.compilations + 1 > per_compile
                { break; }
                let trial_seed = seed.wrapping_add((i as u64 + 1).wrapping_mul(0x9e37_79b9)).wrapping_add(trial as u64);
                let order = crate::ordering::random_order(&parent.creature, trial_seed);
                let mut proposed = materialise(&parent.creature, &order, &[target], stats, budget)?;
                let candidate = proposed.pop().ok_or("independent search produced no candidate")?;
                let error = fit::loss(&candidate.creature, &[], &[], rows, budget)?;
                count += 1;
                if best.as_ref().is_none_or(|(prior, _)| error < *prior) { best = Some((error, candidate)); }
            }
            let (_, candidate) = best.ok_or("family budget exhausted before independent search reached a candidate")?;
            candidates.push(candidate);
            trials.push(count);
        }
        (candidates, None, trials)
    };
    let construction_seconds = construction_started.elapsed().as_secs_f64();
    let creature_dir = dir.join("candidates");
    std::fs::create_dir(&creature_dir).map_err(|e| e.to_string())?;
    for (i, candidate) in candidates.iter().enumerate() {
        let text = if i == 0 { parent.text.clone() } else { neat_core::creature_to_json(&candidate.creature).map_err(|e| e.to_string())? };
        budget.io(text.len() as u64)?;
        write_new(&creature_dir.join(format!("c{i}.json")), text.as_bytes())?;
    }
    budget.records(usize::try_from(corpus.record_count.checked_mul(candidates.len() as u64).ok_or("scorer work overflow")?).map_err(|e| e.to_string())?)?;
    budget.io(corpus.record_count.saturating_mul(((parent.creature.input + parent.creature.output) * 4) as u64))?;
    let scores = scorer.score_directory(&creature_dir, &manifest.corpus.path, ScorerMode::Full).map_err(|e| e.to_string())?;
    validate_full(&scores, ScorerMode::Full, corpus.record_count, candidates.len())?;
    budget.check()?;
    let baseline = scores["c0"].score;
    let mut measurements = Vec::new();
    for (i, candidate) in candidates.into_iter().enumerate() {
        let calibration_mse = fit::loss(&candidate.creature, &[], &[], rows, budget)?;
        let validation_mse = validation_error(manifest, &candidate, budget)?;
        let latency = latency(manifest, &candidate, budget)?;
        let mut sorted = latency.clone(); sorted.sort_by(f64::total_cmp);
        let full = scores[&format!("c{i}")].clone();
        measurements.push(Measurement { candidate, calibration_mse, validation_mse,
            authoritative_improvement: i > 0 && full.score > baseline, full,
            median_latency_ns: sorted[sorted.len() / 2], latency_ns_per_record: latency,
            pareto_error_growth: true, pareto_error_latency: true });
    }
    mark_frontiers(&mut measurements);
    Ok(Arm { name: name.into(), seed, ranking, search_trials: trials, measurements, cost: Cost::default(), construction_seconds })
}

fn final_record_allowance(manifest: &Manifest, corpus_records: u64) -> Result<u64, String> {
    let calibration = manifest.calibration.iter().map(|r| r.len).sum::<u64>();
    let validation = manifest.validation.iter().map(|r| r.len).sum::<u64>();
    let latency = validation.min(manifest.latency_records as u64) * (manifest.latency_repetitions as u64 + 10);
    (corpus_records + calibration + validation + latency).checked_mul(4).ok_or_else(|| "evaluation budget overflow".into())
}

fn validate_full(scores: &BTreeMap<String, crate::scorer::ScoreResult>, mode: ScorerMode, records: u64, candidates: usize) -> Result<(), String> {
    if !mode.is_authoritative() { return Err("sampled-only acceptance refused".into()); }
    for i in 0..candidates {
        let key = format!("c{i}");
        let score = scores.get(&key).ok_or_else(|| format!("full scorer missing {key}"))?;
        if score.sample_rate.is_some_and(|rate| rate != 1.0) { return Err(format!("sampled scorer result for {key}")); }
        if score.record_count != records { return Err(format!("full scorer record count for {key}: {} != {records}", score.record_count)); }
        if score.cost_name.as_deref() != Some("MSE") { return Err(format!("full scorer must report MSE for {key}")); }
        if !score.score.is_finite() || !score.error.is_finite() || score.error < 0.0 { return Err(format!("non-finite or negative full result for {key}")); }
    }
    Ok(())
}

fn validation_error(manifest: &Manifest, candidate: &Candidate, budget: &mut Budget) -> Result<f64, String> {
    let cfg = TrainingDataConfig::new(candidate.creature.input, candidate.creature.output);
    budget.compile()?;
    let mut net = compile_creature(&candidate.creature).map_err(|e| e.to_string())?;
    let mut sum = 0.0;
    let read = for_each_selected_chunk(&manifest.corpus.path, &cfg, &manifest.validation, 4096, |chunk| {
        budget.records(chunk.records)?;
        budget.io((chunk.records * cfg.bytes_per_record()) as u64)?;
        for i in 0..chunk.records {
            let inputs = &chunk.inputs[i * cfg.num_inputs..(i + 1) * cfg.num_inputs];
            let targets = &chunk.targets[i * cfg.num_outputs..(i + 1) * cfg.num_outputs];
            if inputs.iter().chain(targets).any(|x| !x.is_finite()) { return Err("non-finite validation record".into()); }
            let prediction = net.activate(inputs, cfg.num_outputs);
            sum += prediction.iter().zip(targets).map(|(a, b)| (f64::from(*a) - f64::from(*b)).powi(2)).sum::<f64>() / cfg.num_outputs as f64;
        }
        Ok(ControlFlow::Continue(()))
    })?;
    let expected: u64 = manifest.validation.iter().map(|r| r.len).sum();
    if read != expected || !sum.is_finite() { return Err("incomplete or non-finite validation measurement".into()); }
    Ok(sum / read as f64)
}

fn latency(manifest: &Manifest, candidate: &Candidate, budget: &mut Budget) -> Result<Vec<f64>, String> {
    let mut remaining = manifest.latency_records as u64;
    let ranges = manifest.validation.iter().filter_map(|r| {
        let len = r.len.min(remaining); remaining -= len;
        (len > 0).then_some(RecordRange { start: r.start, len })
    }).collect::<Vec<_>>();
    let cfg = TrainingDataConfig::new(candidate.creature.input, candidate.creature.output);
    let rows = load_rows(&manifest.corpus.path, &cfg, &ranges)?;
    budget.io((rows.len() * cfg.bytes_per_record()) as u64)?;
    budget.compile()?;
    let mut net = compile_creature(&candidate.creature).map_err(|e| e.to_string())?;
    let mut samples = Vec::new();
    for repetition in 0..manifest.latency_repetitions + 10 {
        budget.records(rows.len())?;
        let start = Instant::now();
        for row in &rows { std::hint::black_box(net.activate(std::hint::black_box(&row.inputs), cfg.num_outputs)); }
        let ns = start.elapsed().as_nanos() as f64 / rows.len() as f64;
        if repetition >= 10 { samples.push(ns); }
    }
    Ok(samples)
}

fn mark_frontiers(measurements: &mut [Measurement]) {
    for i in 0..measurements.len() {
        for j in 0..measurements.len() {
            if i == j { continue; }
            let a = &measurements[i]; let b = &measurements[j];
            let better_error = b.full.error <= a.full.error;
            let growth = better_error && b.candidate.size.growth_units <= a.candidate.size.growth_units
                && (b.full.error < a.full.error || b.candidate.size.growth_units < a.candidate.size.growth_units);
            let latency = better_error && b.median_latency_ns <= a.median_latency_ns
                && (b.full.error < a.full.error || b.median_latency_ns < a.median_latency_ns);
            if growth { measurements[i].pareto_error_growth = false; }
            if latency { measurements[i].pareto_error_latency = false; }
        }
    }
}

fn paired_summary(arms: &[Arm], manifest: &Manifest, no_failures: bool) -> Value {
    let mut deltas = Vec::new();
    let mut diagnostics_ok = true;
    let mut cost_pairs = Vec::new();
    for seed in &manifest.seeds {
        let Some(gates) = arms.iter().find(|a| a.seed == *seed && a.name == "gates") else { continue; };
        let Some(control) = arms.iter().find(|a| a.seed == *seed && a.name == "independent") else { continue; };
        let max = gates.cost.elapsed_seconds.max(control.cost.elapsed_seconds);
        let imbalance = if max > 0.0 { (gates.cost.elapsed_seconds - control.cost.elapsed_seconds).abs() / max } else { 0.0 };
        diagnostics_ok &= imbalance <= manifest.criteria.max_cost_imbalance;
        cost_pairs.push(json!({ "seed": seed, "gatesSeconds": gates.cost.elapsed_seconds,
            "independentSeconds": control.cost.elapsed_seconds, "fractionalImbalance": imbalance,
            "matched": imbalance <= manifest.criteria.max_cost_imbalance }));
        let mut delta = 0.0;
        for i in 1..4 {
            let g = &gates.measurements[i]; let c = &control.measurements[i];
            delta += (g.full.score - c.full.score) / 3.0;
            diagnostics_ok &= g.candidate.reachable && c.candidate.reachable
                && g.candidate.overshoot_hidden as f64 / gates.measurements[0].candidate.size.hidden_neurons as f64 <= manifest.criteria.max_capacity_overshoot
                && g.validation_mse <= c.validation_mse * (1.0 + manifest.criteria.max_validation_error_increase) + 1e-12;
            if i >= 2 { diagnostics_ok &= g.median_latency_ns <= c.median_latency_ns * (1.0 - manifest.criteria.min_latency_reduction); }
        }
        deltas.push(delta);
    }
    let n = deltas.len();
    let mean = if n == 0 { 0.0 } else { deltas.iter().sum::<f64>() / n as f64 };
    let se = if n > 1 { (deltas.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1) as f64 / n as f64).sqrt() } else { 0.0 };
    let complete = no_failures && n == manifest.seeds.len();
    // Conservative t critical value for n>=10 (df=9); descriptive below 10.
    let interval = if n >= 10 { Some([mean - 2.262 * se, mean + 2.262 * se]) } else { None };
    let retained = manifest.representative && complete && diagnostics_ok && interval.is_some_and(|ci| ci[0] > manifest.criteria.min_paired_score_gain);
    json!({ "pairedScoreDeltas": deltas, "meanScoreDelta": (n > 0).then_some(mean), "standardError": (n > 1).then_some(se),
        "conservative95Interval": interval, "completePairs": n,
        "totalFamilyCostPairs": cost_pairs,
        "uncertaintyUnit": "seed; conditional on this creature, corpus, scorer and host; not across-creature evidence",
        "verdict": if retained { "retain" } else { "inconclusive-or-does-not-meet-retention-criteria" },
        "diagnosticsPass": diagnostics_ok, "complete": complete })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ablation::StructureSnapshot;
    use crate::corpus::for_each_chunk;
    use crate::fixtures::wide_creature;
    use crate::scorer::{ScoreResult, ScorerError};
    use super::super::manifest::Limits;

    struct CoreScorer;
    impl DirectoryScorer for CoreScorer {
        fn score_directory(&self, creatures: &Path, data: &Path, mode: ScorerMode) -> Result<BTreeMap<String, ScoreResult>, ScorerError> {
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
                        let predicted = net.activate(&chunk.inputs[i * cfg.num_inputs..(i + 1) * cfg.num_inputs], cfg.num_outputs);
                        sum += predicted.iter().zip(&chunk.targets[i * cfg.num_outputs..(i + 1) * cfg.num_outputs])
                            .map(|(a, b)| (f64::from(*a) - f64::from(*b)).powi(2)).sum::<f64>() / cfg.num_outputs as f64;
                    }
                    Ok(())
                }).unwrap();
                let error = sum / n as f64;
                let penalty = StructureSnapshot::of(&parent.creature).growth_units * 0.0001;
                scores.insert(path.file_stem().unwrap().to_str().unwrap().into(), ScoreResult {
                    score: 1.0 - error - penalty, error, complexity_penalty: penalty,
                    record_count: n, sample_rate: None, gpu_backend: None, cost_name: Some("MSE".into()), time_taken: 0.0,
                });
            }
            Ok(scores)
        }
        fn identity(&self) -> String { "test-core-MSE".into() }
    }

    #[test]
    fn sampled_or_partial_results_cannot_authorise_a_prune() {
        let full = CoreScorer;
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join("creatures");
        let data = tmp.path().join("data");
        std::fs::create_dir(&dir).unwrap(); std::fs::create_dir(&data).unwrap();
        std::fs::write(dir.join("c0.json"), neat_core::creature_to_json(&wide_creature(1, 4, "TANH")).unwrap()).unwrap();
        write_bin_file(&data.join("0.bin"), &vec![(vec![0.2], vec![0.1]); 4]).unwrap();
        let mut scores = full.score_directory(&dir, &data, ScorerMode::Full).unwrap();
        validate_full(&scores, ScorerMode::Full, 4, 1).unwrap();
        assert!(validate_full(&scores, ScorerMode::Sample { rate: 0.5, phase: 0 }, 4, 1).unwrap_err().contains("sampled"));
        scores.get_mut("c0").unwrap().record_count = 2;
        assert!(validate_full(&scores, ScorerMode::Full, 4, 1).unwrap_err().contains("record count"));
        scores.get_mut("c0").unwrap().record_count = 4;
        scores.get_mut("c0").unwrap().sample_rate = Some(0.5);
        assert!(validate_full(&scores, ScorerMode::Full, 4, 1).unwrap_err().contains("sampled"));
        scores.clear();
        assert!(validate_full(&scores, ScorerMode::Full, 4, 1).unwrap_err().contains("missing c0"));
    }

    #[test]
    fn calibration_selection_is_isolated_from_validation_targets() {
        let root = tempfile::tempdir().unwrap();
        let data = root.path().join("data"); std::fs::create_dir(&data).unwrap();
        let rows = (0..16).map(|i| (vec![i as f32 / 16.0], vec![0.2])).collect::<Vec<_>>();
        write_bin_file(&data.join("0.bin"), &rows).unwrap();
        let cfg = TrainingDataConfig::new(1, 1);
        let ranges = [RecordRange { start: 0, len: 8 }];
        let a = load_rows(&data, &cfg, &ranges).unwrap();
        let mut changed = rows;
        for row in &mut changed[12..] { row.1[0] = 999.0; }
        write_bin_file(&data.join("0.bin"), &changed).unwrap();
        let b = load_rows(&data, &cfg, &ranges).unwrap();
        let parent = wide_creature(1, 4, "TANH");
        let limits = Limits { record_evaluations: 1_000_000, compilations: 100_000, rewrites: 1000, io_bytes: 1_000_000, seconds: 60.0 };
        let config = super::super::manifest::Fit { steps: 2, learning_rate: 0.1, temperature: 0.2, difference_step: 0.001 };
        let fit_a = fit::fit(&parent, &a, &config, 7, &mut Budget::new(&limits, &Cost::default()).unwrap()).unwrap();
        let fit_b = fit::fit(&parent, &b, &config, 7, &mut Budget::new(&limits, &Cost::default()).unwrap()).unwrap();
        assert_eq!(fit_a.log_scores, fit_b.log_scores);
        assert_eq!(fit_a.removal_order, fit_b.removal_order);
        let held = load_rows(&data, &cfg, &[RecordRange { start: 12, len: 4 }]).unwrap();
        assert_eq!(held[0].targets, [999.0]);
    }
}
