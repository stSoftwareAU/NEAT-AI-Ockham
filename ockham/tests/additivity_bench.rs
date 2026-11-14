//! Integration tests for the opt-in additivity benchmark (Issue #260).
//!
//! The tests drive the real bench code (pruning, corpus streaming, output
//! cache, ranking and reporting) against a faithful in-test full scorer: it
//! compiles each creature, streams the training corpus, and computes the mean
//! `neat_core::loss::mse_record` and the documented score `1 - error - penalty`.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicUsize, Ordering};

use neat_ai_ockham::ablation::growth_units;
use neat_ai_ockham::additivity::bench::run_bench;
use neat_ai_ockham::additivity::bench_report::BenchReport;
use neat_ai_ockham::additivity::manifest::{BenchManifest, ResourceLimits, ScorerSpec};
use neat_ai_ockham::additivity::rank::{ARMS, Arm};
use neat_ai_ockham::clock::SystemClock;
use neat_ai_ockham::corpus::{corpus_info, for_each_chunk, write_bin_file};
use neat_ai_ockham::fixtures::{creature, identity_creature, neuron, synapse};
use neat_ai_ockham::scorer::{DirectoryScorer, ScoreResult, ScorerError, ScorerMode};
use neat_ai_ockham::stats::SampleSpec;
use neat_core::loss::mse_record;
use neat_core::training_data::TrainingDataConfig;
use neat_core::{CreatureExport, compile_creature, creature_to_json, parse_creature_json};

const RECORDS: usize = 300;
const INPUTS: usize = 2;
const OUTPUTS: usize = 1;
const HIDDEN: usize = 6;

/// Faithful in-test full scorer; counts its calls.
struct FaithfulScorer {
    calls: AtomicUsize,
}

impl FaithfulScorer {
    fn new() -> Self {
        Self {
            calls: AtomicUsize::new(0),
        }
    }
}

impl DirectoryScorer for FaithfulScorer {
    fn score_directory(
        &self,
        creature_dir: &Path,
        training_dir: &Path,
        mode: ScorerMode,
    ) -> Result<BTreeMap<String, ScoreResult>, ScorerError> {
        assert_eq!(
            mode,
            ScorerMode::Full,
            "the bench must only request full scores"
        );
        self.calls.fetch_add(1, Ordering::SeqCst);
        let mut results = BTreeMap::new();
        let entries =
            fs::read_dir(creature_dir).map_err(|e| ScorerError::Malformed(e.to_string()))?;
        for entry in entries {
            let path = entry
                .map_err(|e| ScorerError::Malformed(e.to_string()))?
                .path();
            if path.extension().and_then(|e| e.to_str()) != Some("json") {
                continue;
            }
            let stem = path.file_stem().unwrap().to_string_lossy().into_owned();
            let text =
                fs::read_to_string(&path).map_err(|e| ScorerError::Malformed(e.to_string()))?;
            let parsed =
                parse_creature_json(&text).map_err(|e| ScorerError::Malformed(e.to_string()))?;
            let hidden = parsed
                .neurons
                .iter()
                .filter(|n| n.neuron_type == "hidden")
                .count();
            let penalty = 1e-4 * growth_units(hidden, parsed.synapses.len());
            let mut net =
                compile_creature(&parsed).map_err(|e| ScorerError::Malformed(e.to_string()))?;
            let cfg = TrainingDataConfig::new(parsed.input, parsed.output);
            let mut total = 0.0_f64;
            let records = for_each_chunk(training_dir, &cfg, 64, |chunk| {
                for r in 0..chunk.records {
                    let input = &chunk.inputs[r * parsed.input..(r + 1) * parsed.input];
                    let target = &chunk.targets[r * parsed.output..(r + 1) * parsed.output];
                    let output = net.activate(input, parsed.output);
                    total += mse_record(target, &output);
                }
                Ok(())
            })
            .map_err(ScorerError::Malformed)?;
            let error = total / records as f64;
            results.insert(
                stem,
                ScoreResult {
                    score: 1.0 - error - penalty,
                    error,
                    complexity_penalty: penalty,
                    record_count: records,
                    sample_rate: None,
                    gpu_backend: None,
                    cost_name: Some("MSE".to_string()),
                    time_taken: 0.0,
                },
            );
        }
        Ok(results)
    }

    fn identity(&self) -> String {
        "faithful-test-scorer".to_string()
    }
}

/// Delegates to the faithful scorer, then reports a different cost function.
struct MisreportingScorer {
    cost: &'static str,
}

impl DirectoryScorer for MisreportingScorer {
    fn score_directory(
        &self,
        creature_dir: &Path,
        training_dir: &Path,
        mode: ScorerMode,
    ) -> Result<BTreeMap<String, ScoreResult>, ScorerError> {
        let mut results =
            FaithfulScorer::new().score_directory(creature_dir, training_dir, mode)?;
        for result in results.values_mut() {
            result.cost_name = Some(self.cost.to_string());
        }
        Ok(results)
    }

    fn identity(&self) -> String {
        "misreporting-test-scorer".to_string()
    }
}

/// Six hidden neurons with differing output weights, so their usefulness differs.
fn fixture_parent() -> CreatureExport {
    let mut neurons = Vec::new();
    let mut synapses = Vec::new();
    for h in 0..HIDDEN {
        let uuid = format!("h{h}");
        neurons.push(neuron(
            "hidden",
            &uuid,
            0.05 * h as f64 - 0.1,
            Some("LOGISTIC"),
        ));
        synapses.push(synapse("input-0", &uuid, 0.4 + 0.1 * h as f64));
        synapses.push(synapse("input-1", &uuid, -0.3 + 0.12 * h as f64));
        synapses.push(synapse(
            &uuid,
            "output-0",
            (HIDDEN - h) as f64 / HIDDEN as f64,
        ));
    }
    neurons.push(neuron("output", "output-0", 0.0, Some("IDENTITY")));
    creature(INPUTS, OUTPUTS, neurons, synapses)
}

/// Deterministic corpus; `tweak` changes one record without changing the length.
fn corpus_records(tweak: Option<usize>) -> Vec<(Vec<f32>, Vec<f32>)> {
    let mut state = 0x2602_u64;
    let mut next = move || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 33) as f32 / (1u64 << 31) as f32) * 2.0 - 1.0
    };
    (0..RECORDS)
        .map(|i| {
            let x0 = next();
            let x1 = next();
            let mut y = (x0 - 0.5 * x1).tanh();
            if tweak == Some(i) {
                y = -y;
            }
            (vec![x0, x1], vec![y])
        })
        .collect()
}

struct Fixture {
    root: tempfile::TempDir,
}

impl Fixture {
    fn new() -> Self {
        let root = tempfile::tempdir().expect("tempdir");
        fs::create_dir_all(root.path().join("parents")).unwrap();
        fs::create_dir_all(root.path().join("train")).unwrap();
        let parent = fixture_parent();
        fs::write(
            root.path().join("parents/elite.json"),
            creature_to_json(&parent).unwrap(),
        )
        .unwrap();
        write_bin_file(&root.path().join("train/0.bin"), &corpus_records(None)).unwrap();
        Self { root }
    }

    fn root(&self) -> &Path {
        self.root.path()
    }

    fn path(&self, rel: &str) -> PathBuf {
        self.root.path().join(rel)
    }

    fn rewrite_training(&self, tweak: Option<usize>) {
        write_bin_file(&self.path("train/0.bin"), &corpus_records(tweak)).unwrap();
    }

    fn manifest(&self, budgets: Vec<f64>, seeds: Vec<u64>, max_cache_bytes: u64) -> BenchManifest {
        BenchManifest {
            format_version: 1,
            parents: vec![self.path("parents/elite.json")],
            training_data: self.path("train"),
            validation_data: None,
            scorer: ScorerSpec {
                binary: PathBuf::from("unused-in-tests"),
                args: Vec::new(),
            },
            cost: "MSE".to_string(),
            seeds,
            core_fraction: 0.1,
            budgets,
            audit_pool: 6,
            full_evaluations: 3,
            top_k: 2,
            min_improvement: 0.0,
            decision_rule: "fixture-only test rule".to_string(),
            limits: ResourceLimits {
                max_cache_bytes,
                chunk_records: 16,
            },
            cache_dir: self.path("cache"),
        }
    }
}

fn run(manifest: &BenchManifest) -> BenchReport {
    let scorer = FaithfulScorer::new();
    let workspace = manifest.cache_dir.join("workspace");
    run_bench(manifest, &scorer, &SystemClock, &workspace).expect("bench runs")
}

/// Every path under `root`, relative to it, sorted.
fn all_paths(root: &Path) -> Vec<String> {
    let mut found = Vec::new();
    let mut pending = vec![root.to_path_buf()];
    while let Some(dir) = pending.pop() {
        for entry in fs::read_dir(&dir).unwrap() {
            let path = entry.unwrap().path();
            found.push(path.strip_prefix(root).unwrap().display().to_string());
            if path.is_dir() {
                pending.push(path);
            }
        }
    }
    found.sort();
    found
}

fn listing(dir: &Path) -> Vec<String> {
    let mut names: Vec<String> = fs::read_dir(dir)
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    names.sort();
    names
}

#[test]
fn end_to_end_reports_every_arm_and_leaves_the_parent_alone() {
    let fixture = Fixture::new();
    let manifest = fixture.manifest(vec![0.2, 0.4], vec![1, 2], 1 << 30);
    let parent_bytes = fs::read(fixture.path("parents/elite.json")).unwrap();
    let parents_before = listing(&fixture.path("parents"));

    let report = run(&manifest);

    assert_eq!(report.rows.len(), 4, "one row per (budget, seed)");
    assert!(!report.representative);
    assert!(
        report.evidence.starts_with("fixture-only:"),
        "evidence was {:?}",
        report.evidence
    );
    assert!(report.authority.contains("full-corpus scorer"));
    assert_eq!(report.aggregate.len(), 2);
    for row in &report.rows {
        assert_eq!(row.arms.len(), ARMS.len());
        for (got, want) in row.arms.iter().zip(ARMS) {
            assert_eq!(got.arm, want);
        }
        if !row.feasible {
            panic!("fixture rows should be feasible: {:?}", row.reason);
        }
        assert!(row.pool_size <= manifest.audit_pool);
        assert_eq!(
            row.bundles.len() + row.duplicates.len(),
            row.pool_size,
            "every materialised pool bundle is scored or a duplicate"
        );
        assert_eq!(
            row.pool_size + row.refused.len(),
            row.pool_requested,
            "every requested pool bundle is materialised or refused"
        );
        for arm_row in &row.arms {
            if !arm_row.arm.predicts() {
                assert!(
                    arm_row.inference_latency_ms.is_none(),
                    "only predictor arms carry an inference latency"
                );
            }
            if let Some(ms) = arm_row.inference_latency_ms {
                assert!(ms.is_finite() && ms >= 0.0, "latency was {ms}");
            }
        }
        assert_eq!(row.audit.full_calls, 1);
        assert_eq!(
            row.audit.record_evaluations,
            row.bundles.len() as u64 * RECORDS as u64
        );
    }
    assert_eq!(
        fs::read(fixture.path("parents/elite.json")).unwrap(),
        parent_bytes,
        "the parent file must be unchanged"
    );
    assert_eq!(listing(&fixture.path("parents")), parents_before);
    let workspace = manifest.cache_dir.join("workspace");
    assert!(
        listing(&workspace).is_empty(),
        "scoring directories must be removed after use: {:?}",
        listing(&workspace)
    );
    // The bench leaves nothing an ordinary run would read: no experiment log,
    // no learnings, no stats cache. Only the output cache and the workspace.
    let paths = all_paths(fixture.root());
    assert!(
        !paths.iter().any(|p| p.ends_with("experiments.jsonl")),
        "an experiment log was written: {paths:?}"
    );
    assert!(
        !paths.iter().any(|p| p.contains("learnings")),
        "a learnings file was written: {paths:?}"
    );
    assert!(
        !fixture.path("cache/stats").exists(),
        "the bench must not write the stats cache"
    );
    assert_eq!(
        listing(&manifest.cache_dir),
        vec!["outputs".to_string(), "workspace".to_string()],
        "cache dir holds only the output cache and the workspace"
    );
}

#[test]
fn a_scorer_reporting_another_cost_fails_the_run_naming_it() {
    let fixture = Fixture::new();
    let manifest = fixture.manifest(vec![0.2], vec![1], 1 << 30);
    let scorer = MisreportingScorer { cost: "MAE" };
    let workspace = manifest.cache_dir.join("workspace");

    let err = run_bench(&manifest, &scorer, &SystemClock, &workspace).unwrap_err();

    assert!(
        err.contains("MAE"),
        "error must name the reported cost: {err}"
    );
    assert!(
        err.contains("MSE"),
        "error must name the declared cost: {err}"
    );
}

#[test]
fn predictions_only_order_and_the_best_full_score_among_picks_selects() {
    let fixture = Fixture::new();
    let manifest = fixture.manifest(vec![0.2, 0.4], vec![1, 2], 1 << 30);
    let train_before = listing(&fixture.path("train"));
    let parents_before = listing(&fixture.path("parents"));

    let report = run(&manifest);

    for row in report.rows.iter().filter(|r| r.feasible) {
        for arm_row in &row.arms {
            assert!(
                !arm_row.picks.is_empty(),
                "every arm sends picks to the scorer"
            );
            let top = arm_row
                .picks
                .iter()
                .map(|p| p.score)
                .fold(f64::NEG_INFINITY, f64::max);
            let selected = arm_row.selected_bundle.expect("a selection exists");
            let selected_pick = arm_row
                .picks
                .iter()
                .find(|p| p.bundle == selected)
                .expect("the selection is one of the arm's picks");
            assert!(
                (selected_pick.score - top).abs() < 1e-12,
                "{:?} selected {} at {} but its best pick scored {}",
                arm_row.arm,
                selected,
                selected_pick.score,
                top
            );
            assert_eq!(arm_row.selected_score, Some(selected_pick.score));
            for pick in &arm_row.picks {
                let audited = row.bundles[pick.bundle].score;
                assert!(
                    (pick.score - audited).abs() < 1e-9,
                    "search and audit scores must agree for the same creature"
                );
            }
            let expected_picks: Vec<usize> = arm_row.picks.iter().map(|p| p.bundle).collect();
            assert_eq!(
                expected_picks,
                arm_row.ranking[..expected_picks.len()].to_vec()
            );
        }
    }
    assert_eq!(
        listing(&fixture.path("train")),
        train_before,
        "training data untouched"
    );
    assert_eq!(
        listing(&fixture.path("parents")),
        parents_before,
        "no creature written beside the parent"
    );
}

#[test]
fn cache_is_reused_on_a_second_run_and_invalidated_by_a_changed_record() {
    let fixture = Fixture::new();
    let manifest = fixture.manifest(vec![0.2], vec![1], 1 << 30);

    let first = run(&manifest);
    let first_cache = &first.parents[0].cache;
    assert!(first_cache.builds > 0, "a cold cache must build entries");

    let second = run(&manifest);
    let second_cache = &second.parents[0].cache;
    assert_eq!(second_cache.builds, 0, "a warm cache builds nothing");
    assert!(second_cache.hits > 0, "a warm cache serves hits");

    // Change a record inside the planned audit core, not merely a middle record of
    // the corpus: a record outside the core cannot change the cache key.
    let corpus = corpus_info(
        &fixture.path("train"),
        &TrainingDataConfig::new(INPUTS, OUTPUTS),
    )
    .expect("corpus info");
    let core = SampleSpec::with_max_records((0.1 * RECORDS as f64).ceil() as u64).plan(&corpus);
    let middle = (core[0].start + core[0].len / 2) as usize;
    fixture.rewrite_training(Some(middle));
    let third = run(&manifest);
    let third_cache = &third.parents[0].cache;
    assert!(
        third_cache.builds > 0,
        "a changed middle record changes the dataset digest, so the cache rebuilds"
    );
}

#[test]
fn a_cache_limit_only_fits_the_base_entries_and_the_run_still_completes() {
    let fixture = Fixture::new();
    // Probe the audit core size with an unlimited run first.
    let probe = run(&fixture.manifest(vec![0.4], vec![1], 1 << 30));
    let core = probe.parents[0].core_records;
    assert!(
        core >= 8,
        "the audit core needs at least 8 records, got {core}"
    );

    // Room for targets, y0 and exactly one single, nothing more.
    let entry_bytes = core * OUTPUTS as u64 * 4;
    let mut manifest = fixture.manifest(vec![0.4], vec![1], 3 * entry_bytes);
    manifest.cache_dir = fixture.path("tight-cache");
    let report = run(&manifest);

    let rejections = &report.parents[0].cache.rejections;
    assert!(
        rejections.get("overLimit").copied().unwrap_or(0) > 0,
        "over-limit writes must be reported, got {rejections:?}"
    );
    let unestimated: usize = report
        .rows
        .iter()
        .flat_map(|r| r.arms.iter())
        .filter(|a| a.arm == Arm::CachedOutput)
        .map(|a| a.audit.unestimated)
        .sum();
    assert!(
        unestimated > 0,
        "cachedOutput must have unestimated bundles"
    );
    for row in report.rows.iter().filter(|r| r.feasible) {
        for arm_row in &row.arms {
            assert!(
                arm_row.selected_score.is_some(),
                "every arm still gets a full-scored selection"
            );
        }
    }
    assert!(
        listing(&manifest.cache_dir.join("workspace")).is_empty(),
        "scoring directories must not linger"
    );
}

#[test]
fn a_parent_without_hidden_neurons_is_reported_infeasible_with_no_bundles() {
    let fixture = Fixture::new();
    // Nothing to remove: k = 0 for every budget, so no ranking is possible.
    let bare = identity_creature(INPUTS, OUTPUTS);
    fs::write(
        fixture.path("parents/elite.json"),
        creature_to_json(&bare).unwrap(),
    )
    .unwrap();
    let manifest = fixture.manifest(vec![0.2, 0.4], vec![1, 2], 1 << 30);

    let report = run(&manifest);

    assert_eq!(report.rows.len(), 4);
    for row in &report.rows {
        assert!(
            !row.feasible,
            "a parent with no hidden neurons cannot be ranked"
        );
        assert!(
            row.reason.as_deref().unwrap_or("").contains("no hidden"),
            "reason was {:?}",
            row.reason
        );
        assert!(row.bundles.is_empty(), "no bundle may be pretended");
        assert!(row.arms.is_empty(), "no arm may select anything");
        assert_eq!(row.pool_size, 0);
    }
    assert_eq!(report.parents[0].proposable, 0);
    assert_eq!(report.aggregate.len(), 2);
    for budget in &report.aggregate {
        assert_eq!(budget.feasible_rows, 0);
        assert_eq!(budget.infeasible_rows, 2);
    }
    assert!(report.cases_beating_cached_output.is_empty());
}

#[test]
fn bench_additivity_refuses_a_non_mse_cost_and_lists_itself_in_help() {
    let fixture = Fixture::new();
    let manifest_path = fixture.path("bad-cost.json");
    let manifest = serde_json::json!({
        "formatVersion": 1,
        "parents": ["parents/elite.json"],
        "trainingData": "train",
        "scorer": { "binary": "unused" },
        "cost": "MAE",
        "seeds": [1],
        "auditPool": 4,
        "fullEvaluations": 2,
        "minImprovement": 0.0,
        "decisionRule": "rule",
        "limits": { "maxCacheBytes": 100000, "chunkRecords": 8 },
        "cacheDir": "cache"
    });
    fs::write(&manifest_path, manifest.to_string()).unwrap();
    let out = fixture.path("report.json");

    let failed = Command::new(env!("CARGO_BIN_EXE_neat_ai_ockham"))
        .arg("bench-additivity")
        .arg(&manifest_path)
        .arg("--out")
        .arg(&out)
        .stdin(std::process::Stdio::null())
        .output()
        .expect("binary runs");
    assert!(
        !failed.status.success(),
        "a non-MSE cost must exit non-zero"
    );
    let stderr = String::from_utf8_lossy(&failed.stderr);
    assert!(
        stderr.contains("cost"),
        "stderr must name the cost field: {stderr}"
    );
    assert!(!out.exists(), "no report is written on refusal");

    let help = Command::new(env!("CARGO_BIN_EXE_neat_ai_ockham"))
        .arg("--help")
        .stdin(std::process::Stdio::null())
        .output()
        .expect("binary runs");
    let text = String::from_utf8_lossy(&help.stdout);
    assert!(
        text.contains("bench-additivity"),
        "--help must list the command: {text}"
    );
}

/// Hidden neurons in the wide parent: at a 40 % budget that removes five.
const WIDE_HIDDEN: usize = 12;

/// Twelve hidden neurons, each wired from both inputs and into the output.
fn wide_parent() -> CreatureExport {
    let mut neurons = Vec::new();
    let mut synapses = Vec::new();
    for h in 0..WIDE_HIDDEN {
        let uuid = format!("h{h}");
        neurons.push(neuron(
            "hidden",
            &uuid,
            0.02 * h as f64 - 0.1,
            Some("LOGISTIC"),
        ));
        synapses.push(synapse("input-0", &uuid, 0.3 + 0.05 * h as f64));
        synapses.push(synapse("input-1", &uuid, -0.2 + 0.04 * h as f64));
        synapses.push(synapse(
            &uuid,
            "output-0",
            (WIDE_HIDDEN - h) as f64 / WIDE_HIDDEN as f64,
        ));
    }
    neurons.push(neuron("output", "output-0", 0.0, Some("IDENTITY")));
    creature(INPUTS, OUTPUTS, neurons, synapses)
}

#[test]
fn every_many_member_bundle_gets_a_cached_output_estimate_with_ample_limits() {
    let fixture = Fixture::new();
    fs::write(
        fixture.path("parents/elite.json"),
        creature_to_json(&wide_parent()).unwrap(),
    )
    .unwrap();
    let manifest = fixture.manifest(vec![0.4], vec![7], 1 << 30);

    let report = run(&manifest);

    let parent = &report.parents[0];
    assert_eq!(parent.hidden_neurons, WIDE_HIDDEN);
    assert_eq!(
        parent.estimated, parent.proposable,
        "every single must have a loss"
    );
    assert!(
        parent.cache.rejections.is_empty(),
        "ample limits must not reject anything, got {:?}",
        parent.cache.rejections
    );
    assert!(
        parent.cache.ram_bound_bytes >= parent.core_records * OUTPUTS as u64 * 4,
        "the RAM bound must cover one cache entry"
    );
    let mut many_member_bundles = 0;
    for row in report.rows.iter().filter(|r| r.feasible) {
        for bundle in row.bundles.iter().filter(|b| b.members.len() >= 5) {
            many_member_bundles += 1;
            assert!(
                bundle.cached_output.is_some(),
                "bundle {} with {} members must be estimated",
                bundle.index,
                bundle.members.len()
            );
        }
        let unestimated = row
            .arms
            .iter()
            .filter(|a| a.arm == Arm::CachedOutput)
            .map(|a| a.audit.unestimated)
            .sum::<usize>();
        assert_eq!(unestimated, 0, "cachedOutput must estimate every bundle");
    }
    assert!(
        many_member_bundles > 0,
        "the fixture must produce bundles of five or more members"
    );
}

/// Four flat hidden neurons plus a chain `input -> c_a -> c_b -> output`.
/// `c_b` is fed only by `c_a`, so cutting `c_a` strands it, and cutting `c_b`
/// leaves `c_a` with nowhere to go. Every activation stays in `HARD_TANH`'s
/// linear band, because a single-input `IDENTITY` neuron is spliced into a
/// direct edge by the core cleanup and would never leave the bundle.
fn chain_parent() -> CreatureExport {
    let mut neurons = Vec::new();
    let mut synapses = Vec::new();
    for h in 0..4 {
        let uuid = format!("f{h}");
        neurons.push(neuron("hidden", &uuid, 0.01 * h as f64, Some("HARD_TANH")));
        synapses.push(synapse("input-0", &uuid, 0.2 + 0.05 * h as f64));
        synapses.push(synapse("input-1", &uuid, -0.1 + 0.03 * h as f64));
        synapses.push(synapse(&uuid, "output-0", 0.3 + 0.1 * h as f64));
    }
    neurons.push(neuron("hidden", "c_a", 0.02, Some("HARD_TANH")));
    neurons.push(neuron("hidden", "c_b", 0.03, Some("HARD_TANH")));
    neurons.push(neuron("hidden", "c_c", 0.04, Some("HARD_TANH")));
    neurons.push(neuron("output", "output-0", 0.0, Some("IDENTITY")));
    synapses.push(synapse("input-0", "c_a", 0.4));
    synapses.push(synapse("input-1", "c_a", 0.2));
    synapses.push(synapse("c_a", "c_b", 0.5));
    synapses.push(synapse("c_b", "c_c", 0.7));
    synapses.push(synapse("c_c", "output-0", 0.6));
    creature(INPUTS, OUTPUTS, neurons, synapses)
}

#[test]
fn the_bench_records_cascades_overlaps_and_duplicate_creatures() {
    let fixture = Fixture::new();
    fs::write(
        fixture.path("parents/elite.json"),
        creature_to_json(&chain_parent()).unwrap(),
    )
    .unwrap();
    let mut manifest = fixture.manifest(vec![0.3], vec![1, 2, 3], 1 << 30);
    // 7 proposable cuts, 3 removals each: draw all 35 sets. The pool is then
    // the same whatever order the ranker puts the cuts in, so the chain's
    // alternative cut sets that collapse to one creature are always present.
    manifest.audit_pool = 35;

    let report = run(&manifest);

    let mut with_overlap = 0;
    let mut with_removal_beyond_members = 0;
    let mut with_cascade = 0;
    let mut duplicates_seen = 0;
    for row in report.rows.iter().filter(|r| r.feasible) {
        assert_eq!(row.requested_removals, 3);
        assert_eq!(row.eligible, 7);
        assert!(row.refused.is_empty(), "refused: {:?}", row.refused);
        // Each unique creature is scored once; a duplicate adds no scorer work.
        assert_eq!(row.bundles.len() + row.duplicates.len(), row.pool_size);
        assert_eq!(
            row.audit.record_evaluations,
            row.bundles.len() as u64 * RECORDS as u64,
            "duplicate creatures must not be scored again"
        );
        let digests: std::collections::BTreeSet<&str> = row
            .bundles
            .iter()
            .map(|b| b.creature_digest.as_str())
            .collect();
        assert_eq!(
            digests.len(),
            row.bundles.len(),
            "two scored bundles share a creature"
        );
        for duplicate in &row.duplicates {
            duplicates_seen += 1;
            let first = &row.bundles[duplicate.duplicate_of];
            let mut got = duplicate.members.clone();
            let mut want = first.members.clone();
            got.sort();
            want.sort();
            assert_ne!(got, want, "a duplicate is a different member set");
        }
        for bundle in &row.bundles {
            // Every named member is gone from the materialised creature, and
            // `meetsBudget` reads the materialised structure.
            assert!(
                bundle.hidden_removed >= bundle.members.len(),
                "bundle {} removed {} hidden neurons for {:?}",
                bundle.index,
                bundle.hidden_removed,
                bundle.members
            );
            assert_eq!(
                bundle.meets_budget,
                bundle.hidden_removed >= row.requested_removals
            );
            assert!(bundle.missing.is_empty());
            // An overlap is a named member the removal did not request itself:
            // another member's cut took it, or its inputs were folded away.
            for uuid in &bundle.overlaps {
                assert!(bundle.members.contains(uuid), "{uuid} is not a member");
            }
            if !bundle.overlaps.is_empty() {
                with_overlap += 1;
            }
            if bundle.hidden_removed > bundle.members.len() {
                with_removal_beyond_members += 1;
            }
            if !bundle.cascade_neurons.is_empty() {
                with_cascade += 1;
            }
        }
        // The ranked-prefix bundle names the whole chain, so it must overlap.
        let chain = &row.bundles[0];
        for uuid in ["c_a", "c_b", "c_c"] {
            assert!(
                chain.members.iter().any(|m| m == uuid),
                "{:?}",
                chain.members
            );
        }
        assert!(
            !chain.overlaps.is_empty(),
            "the bundle naming the whole chain must record an overlap"
        );
    }
    assert!(with_overlap > 0, "no bundle recorded an overlap");
    assert!(with_cascade > 0, "no bundle recorded a cascade neuron");
    assert!(
        with_removal_beyond_members > 0,
        "no bundle removed more hidden neurons than it named"
    );
    assert!(duplicates_seen > 0, "no duplicate creature was recorded");
}

/// Delegates to the faithful scorer and keeps every call's results.
struct RecordingScorer {
    inner: FaithfulScorer,
    calls: std::sync::Mutex<Vec<(PathBuf, BTreeMap<String, ScoreResult>)>>,
}

impl DirectoryScorer for RecordingScorer {
    fn score_directory(
        &self,
        creature_dir: &Path,
        training_dir: &Path,
        mode: ScorerMode,
    ) -> Result<BTreeMap<String, ScoreResult>, ScorerError> {
        let results = self
            .inner
            .score_directory(creature_dir, training_dir, mode)?;
        self.calls
            .lock()
            .unwrap()
            .push((training_dir.to_path_buf(), results.clone()));
        Ok(results)
    }

    fn identity(&self) -> String {
        self.inner.identity()
    }
}

#[test]
fn a_held_out_corpus_of_another_size_is_scored_and_reported() {
    const HELD_OUT: usize = 120;
    let fixture = Fixture::new();
    fs::create_dir_all(fixture.path("held")).unwrap();
    // A different length and different targets, so training and held-out
    // results cannot be confused.
    let held: Vec<(Vec<f32>, Vec<f32>)> = corpus_records(None)
        .into_iter()
        .take(HELD_OUT)
        .map(|(x, y)| (x, y.iter().map(|v| 0.5 * v).collect()))
        .collect();
    write_bin_file(&fixture.path("held/0.bin"), &held).unwrap();
    let mut manifest = fixture.manifest(vec![0.4], vec![1], 1 << 30);
    manifest.validation_data = Some(fixture.path("held"));
    let scorer = RecordingScorer {
        inner: FaithfulScorer::new(),
        calls: std::sync::Mutex::new(Vec::new()),
    };
    let workspace = manifest.cache_dir.join("workspace");

    let report = run_bench(&manifest, &scorer, &SystemClock, &workspace)
        .expect("a held-out corpus of another size must not fail the run");

    let calls = scorer.calls.lock().unwrap();
    let held_calls: Vec<_> = calls
        .iter()
        .filter(|(dir, _)| dir == &fixture.path("held"))
        .collect();
    assert_eq!(held_calls.len(), 1, "one held-out scoring call per row");
    let held_results = &held_calls[0].1;
    assert!(
        held_results
            .values()
            .all(|r| r.record_count == HELD_OUT as u64),
        "the scorer must have seen the held-out records"
    );
    assert_ne!(HELD_OUT, RECORDS);

    let row = &report.rows[0];
    assert!(row.feasible, "{:?}", row.reason);
    let parent = &held_results["baseline"];
    assert_eq!(row.validation_parent_score, Some(parent.score));
    assert_ne!(
        row.validation_parent_score,
        Some(report.parents[0].parent_score),
        "the held-out parent score must not be the training score"
    );
    for arm in &row.arms {
        let selected = arm.selected_bundle.expect("a selection exists");
        let result = &held_results[&format!("b{selected}")];
        assert_eq!(arm.validation_score, Some(result.score), "{:?}", arm.arm);
        assert_eq!(arm.validation_error, Some(result.error), "{:?}", arm.arm);
    }
}
