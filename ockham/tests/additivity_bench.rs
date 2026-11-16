//! Integration tests for the opt-in additivity benchmark (Issue #260).
//!
//! The tests drive the real bench code (pruning, corpus streaming, output
//! cache, ranking and reporting) against a faithful in-test full scorer: it
//! compiles each creature, streams the training corpus, and computes the mean
//! `neat_core::loss::mse_record` and the documented score `1 - error - penalty`.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use neat_ai_ockham::ablation::growth_units;
use neat_ai_ockham::additivity::bench::run_bench;
use neat_ai_ockham::additivity::bench_report::{BenchReport, SearchStop, aggregate};
use neat_ai_ockham::additivity::manifest::{BenchManifest, ResourceLimits, ScorerSpec};
use neat_ai_ockham::additivity::rank::{ARMS, Arm};
use neat_ai_ockham::clock::{Clock, ManualClock, SystemClock};
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
            search_wall_clock_ms: None,
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
    if cfg!(target_os = "linux") {
        assert!(
            report
                .measured_peak_rss_bytes
                .is_some_and(|bytes| bytes > 0),
            "a Linux run must measure its own peak resident set"
        );
        for (index, row) in report.rows.iter().enumerate() {
            let Some(bytes) = row.measured_peak_rss_bytes else {
                panic!("row {index} has no measured peak resident set");
            };
            assert!(bytes > 0, "row {index} measured {bytes} bytes");
        }
    }
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

    // Search-vs-audit score comparisons made, and those whose search index differs
    // from the audit index of the same creature.
    let mut comparisons = 0;
    let mut index_differs = 0;
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
            let search = row
                .search
                .as_ref()
                .expect("feasible rows carry a search pool");
            for pick in &arm_row.picks {
                // A pick is in the search pool; where its creature is also in the
                // audit pool, the two full scores must agree.
                let digest = &search.bundles[pick.bundle].creature_digest;
                if let Some(audited) = row.bundles.iter().find(|b| &b.creature_digest == digest) {
                    comparisons += 1;
                    if audited.index != pick.bundle {
                        index_differs += 1;
                    }
                    assert!(
                        (pick.score - audited.score).abs() < 1e-9,
                        "search and audit scores must agree for the same creature"
                    );
                }
            }
            let expected_picks: Vec<usize> = arm_row.picks.iter().map(|p| p.bundle).collect();
            assert_eq!(
                expected_picks,
                arm_row.search_ranking[..expected_picks.len()].to_vec()
            );
        }
    }
    assert!(
        comparisons > 0,
        "at least one pick must share its creature with the audit pool, so the score agreement is checked"
    );
    assert!(
        index_differs > 0,
        "at least one compared pick must have a search index different from its audit index"
    );
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
///
/// The chain weights are large enough that each cut costs a clearly different
/// loss, so the ranker orders the three chain neurons by that gap rather than
/// by float noise or the uuid tie-break. Whether two cut sets collapse to one
/// creature depends on that order (`a`, `b`, `c` in rank order yields none), so
/// the fixture must pin it.
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
    synapses.push(synapse("c_a", "c_b", 3.0));
    synapses.push(synapse("c_b", "c_c", 3.0));
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
    // the same whatever order the ranker puts the cuts in; the chain weights
    // pin the rank order so the sets that collapse to one creature are present.
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
            assert_eq!(
                duplicate.creature_digest, first.creature_digest,
                "duplicateOf must point at the bundle with the same creature"
            );
            let mut got = duplicate.members.clone();
            let mut want = first.members.clone();
            got.sort();
            want.sort();
            assert_ne!(got, want, "a duplicate is a different member set");
        }
        for bundle in &row.bundles {
            // Every hidden neuron a bundle removed is named: a member the cut
            // took itself, or a neuron it stranded (cascade or constant fold).
            assert_names_every_removal(bundle);
            assert!(
                bundle.spliced_neurons.is_empty(),
                "HARD_TANH keeps the chain splice-free: {:?}",
                bundle.spliced_neurons
            );
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

#[test]
fn search_bundles_record_cascades_and_overlaps_including_those_the_audit_pool_lacks() {
    let fixture = Fixture::new();
    fs::write(
        fixture.path("parents/elite.json"),
        creature_to_json(&chain_parent()).unwrap(),
    )
    .unwrap();
    let mut manifest = fixture.manifest(vec![0.3], vec![1, 2, 3], 1 << 30);
    // 35 member sets exist; a pool of 12 is drawn separately for audit and
    // search, so the search pool holds creatures the audit pool does not.
    manifest.audit_pool = 12;

    let report = run(&manifest);

    let mut with_overlap = 0;
    let mut with_cascade = 0;
    let mut only_in_search = 0;
    let mut only_in_search_with_removal_beyond_request = 0;
    for row in report.rows.iter().filter(|r| r.feasible) {
        let search = row
            .search
            .as_ref()
            .expect("feasible rows carry a search pool");
        let audit_digests: BTreeSet<&str> = row
            .bundles
            .iter()
            .map(|b| b.creature_digest.as_str())
            .collect();
        for bundle in &search.bundles {
            // Every removed hidden neuron is named, as for an audit bundle.
            assert_eq!(
                bundle.hidden_removed,
                bundle.members.len() - bundle.overlaps.len()
                    + bundle.cascade_neurons.len()
                    + bundle.spliced_neurons.len(),
                "search bundle {} removed neurons it does not name: members {:?}, overlaps {:?}, cascade {:?}, spliced {:?}",
                bundle.index,
                bundle.members,
                bundle.overlaps,
                bundle.cascade_neurons,
                bundle.spliced_neurons
            );
            assert!(bundle.missing.is_empty());
            assert!(bundle.spliced_neurons.is_empty());
            for uuid in &bundle.overlaps {
                assert!(bundle.members.contains(uuid), "{uuid} is not a member");
            }
            // Removed-neuron identities come from the chain, never elsewhere.
            for uuid in bundle.cascade_neurons.iter().chain(&bundle.overlaps) {
                assert!(
                    ["c_a", "c_b", "c_c"].contains(&uuid.as_str()),
                    "unexpected removed neuron {uuid}"
                );
            }
            with_overlap += usize::from(!bundle.overlaps.is_empty());
            with_cascade += usize::from(!bundle.cascade_neurons.is_empty());
            if !audit_digests.contains(bundle.creature_digest.as_str()) {
                only_in_search += 1;
                if !bundle.overlaps.is_empty() || !bundle.cascade_neurons.is_empty() {
                    only_in_search_with_removal_beyond_request += 1;
                }
            }
        }
    }
    assert!(with_overlap > 0, "no search bundle recorded an overlap");
    assert!(
        with_cascade > 0,
        "no search bundle recorded a cascade neuron"
    );
    assert!(
        only_in_search > 0,
        "every search bundle was in the audit pool"
    );
    assert!(
        only_in_search_with_removal_beyond_request > 0,
        "no search-only bundle recorded an overlap or cascade"
    );
}

/// Every hidden neuron a bundle removed is named exactly once: a member the
/// cut took itself, or a neuron it stranded (cascade or constant fold), or a
/// pass-through the cleanup spliced out.
fn assert_names_every_removal(bundle: &neat_ai_ockham::additivity::bench_report::BundleRow) {
    assert_eq!(
        bundle.hidden_removed,
        bundle.members.len() - bundle.overlaps.len()
            + bundle.cascade_neurons.len()
            + bundle.spliced_neurons.len(),
        "bundle {} removed neurons it does not name: members {:?}, overlaps {:?}, cascade {:?}, spliced {:?}",
        bundle.index,
        bundle.members,
        bundle.overlaps,
        bundle.cascade_neurons,
        bundle.spliced_neurons
    );
}

/// Four flat hidden neurons plus two splice shapes: `s_x -> m_x` where each
/// `m_x` is a `MINIMUM` over `input-0` and `s_x`. Cutting `s_x` leaves `m_x`
/// with one inward edge; core converts it to `IDENTITY` and splices it out,
/// naming it only on `PruneDetail::spliced_neurons` (neat-core 0.22.0).
fn splice_parent() -> CreatureExport {
    let mut neurons = Vec::new();
    let mut synapses = Vec::new();
    for h in 0..4 {
        let uuid = format!("f{h}");
        neurons.push(neuron("hidden", &uuid, 0.01 * h as f64, Some("HARD_TANH")));
        synapses.push(synapse("input-0", &uuid, 0.2 + 0.05 * h as f64));
        synapses.push(synapse("input-1", &uuid, -0.1 + 0.03 * h as f64));
        synapses.push(synapse(&uuid, "output-0", 0.3 + 0.1 * h as f64));
    }
    for (x, w) in [("a", 2.0), ("b", 3.0)] {
        let s = format!("s_{x}");
        let m = format!("m_{x}");
        neurons.push(neuron("hidden", &s, 0.0, Some("HARD_TANH")));
        neurons.push(neuron("hidden", &m, 0.0, Some("MINIMUM")));
        synapses.push(synapse("input-1", &s, 0.5));
        synapses.push(synapse("input-0", &m, 1.0));
        synapses.push(synapse(&s, &m, w));
        synapses.push(synapse(&m, "output-0", 0.4));
    }
    neurons.push(neuron("output", "output-0", 0.0, Some("IDENTITY")));
    creature(INPUTS, OUTPUTS, neurons, synapses)
}

#[test]
fn the_bench_names_neurons_the_core_splices_out() {
    let fixture = Fixture::new();
    fs::write(
        fixture.path("parents/elite.json"),
        creature_to_json(&splice_parent()).unwrap(),
    )
    .unwrap();
    // 8 hidden neurons, 2 removals each: draw all 28 pairs.
    let mut manifest = fixture.manifest(vec![0.25], vec![1, 2], 1 << 30);
    manifest.audit_pool = 28;

    let report = run(&manifest);

    let mut with_splice = 0;
    let mut spliced_member = 0;
    for row in report.rows.iter().filter(|r| r.feasible) {
        assert_eq!(row.requested_removals, 2);
        for bundle in &row.bundles {
            assert_names_every_removal(bundle);
            for uuid in &bundle.spliced_neurons {
                assert!(
                    uuid == "m_a" || uuid == "m_b",
                    "only a MINIMUM left with one edge is spliced: {uuid}"
                );
                if bundle.members.contains(uuid) {
                    spliced_member += 1;
                    assert!(
                        bundle.overlaps.contains(uuid),
                        "a member an earlier splice took is an overlap: {bundle:?}"
                    );
                }
            }
            if !bundle.spliced_neurons.is_empty() {
                with_splice += 1;
            }
        }
    }
    assert!(with_splice > 0, "no bundle recorded a spliced neuron");
    assert!(
        spliced_member > 0,
        "no bundle had a member taken by an earlier splice"
    );
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

#[test]
fn bench_additivity_refuses_a_report_path_that_would_overwrite_an_input() {
    let fixture = Fixture::new();
    let manifest_path = fixture.path("manifest.json");
    let manifest = serde_json::json!({
        "formatVersion": 1,
        "parents": ["parents/elite.json"],
        "trainingData": "train",
        "scorer": { "binary": "unused" },
        "cost": "MSE",
        "seeds": [1],
        "auditPool": 4,
        "fullEvaluations": 2,
        "minImprovement": 0.0,
        "decisionRule": "rule",
        "limits": { "maxCacheBytes": 100000, "chunkRecords": 8 },
        "cacheDir": "cache"
    });
    fs::write(&manifest_path, manifest.to_string()).unwrap();
    let parent = fixture.path("parents/elite.json");
    let parent_before = fs::read(&parent).unwrap();
    let corpus_before = listing(&fixture.path("train"));

    for (out, expected) in [
        (parent.clone(), "would overwrite the input"),
        (manifest_path.clone(), "would overwrite the input"),
        (
            fixture.path("train/report.json"),
            "is inside the input corpus",
        ),
    ] {
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
            "--out {} must exit non-zero",
            out.display()
        );
        let stderr = String::from_utf8_lossy(&failed.stderr);
        assert!(stderr.contains(expected), "{}: {stderr}", out.display());
    }
    assert_eq!(fs::read(&parent).unwrap(), parent_before, "parent changed");
    assert_eq!(listing(&fixture.path("train")), corpus_before);
    assert!(
        !fixture.path("cache").exists(),
        "no scorer work was started"
    );
}

/// Delegates to the faithful scorer, and spends `step` of the manual clock on
/// every call, so a wall-clock budget is decided by how many calls were made.
struct ClockedScorer {
    inner: FaithfulScorer,
    clock: ManualClock,
    step: Duration,
}

impl DirectoryScorer for ClockedScorer {
    fn score_directory(
        &self,
        creature_dir: &Path,
        training_dir: &Path,
        mode: ScorerMode,
    ) -> Result<BTreeMap<String, ScoreResult>, ScorerError> {
        self.clock.advance(self.step);
        self.inner.score_directory(creature_dir, training_dir, mode)
    }

    fn identity(&self) -> String {
        self.inner.identity()
    }
}

/// The fixture with `fullEvaluations` equal to the audit pool, so the arms can
/// send more picks than a short budget allows.
fn search_manifest(fixture: &Fixture, search_wall_clock_ms: Option<u64>) -> BenchManifest {
    let mut manifest = fixture.manifest(vec![0.2], vec![1], 1 << 30);
    manifest.full_evaluations = manifest.audit_pool;
    manifest.search_wall_clock_ms = search_wall_clock_ms;
    manifest
}

#[test]
fn a_search_wall_clock_budget_stops_each_arm_after_three_scored_picks() {
    let fixture = Fixture::new();
    let manifest = search_manifest(&fixture, Some(25));
    let clock = ManualClock::new();
    let scorer = ClockedScorer {
        inner: FaithfulScorer::new(),
        clock: clock.clone(),
        step: Duration::from_millis(10),
    };
    let workspace = manifest.cache_dir.join("workspace");

    let report = run_bench(&manifest, &scorer, &clock, &workspace).expect("bench runs");

    assert_eq!(report.manifest.search_wall_clock_ms, Some(25));
    let row = &report.rows[0];
    assert!(row.feasible, "{:?}", row.reason);
    let search = row
        .search
        .as_ref()
        .expect("feasible rows carry a search pool");
    assert!(
        search.bundles.len() > 3,
        "the search pool must outlast three picks, has {}",
        search.bundles.len()
    );
    for arm in &row.arms {
        // Nothing but the scorer advances the manual clock, so every charged
        // phase costs zero: 0 < 25 calls (10 ms), 10 < 25 calls (20 ms),
        // 20 < 25 calls (30 ms), and 30 is not below 25, so the search stops.
        // The third call ends at 30, past the 25 ms budget, so its pick is kept
        // and charged but is not within budget.
        assert_eq!(arm.picks.len(), 3, "{:?}", arm.arm);
        assert_eq!(arm.full_scorer_calls, 3, "{:?}", arm.arm);
        assert_eq!(arm.stopped_by, Some(SearchStop::WallClock), "{:?}", arm.arm);
        assert_eq!(arm.cost_ms.scorer, 30, "{:?}", arm.arm);
        assert_eq!(
            arm.cost_ms.total, 30,
            "{:?}: only the arm's own scorer calls may take time",
            arm.arm
        );
        let within: Vec<bool> = arm.picks.iter().map(|p| p.within_budget).collect();
        assert_eq!(within, vec![true, true, false], "{:?}", arm.arm);
        // The selection is the best full score among the first two picks, the
        // only within-budget ones, ties to the lower bundle index.
        let mut expected = &arm.picks[0];
        for pick in &arm.picks[1..2] {
            if pick.score > expected.score
                || (pick.score == expected.score && pick.bundle < expected.bundle)
            {
                expected = pick;
            }
        }
        assert_eq!(arm.selected_bundle, Some(expected.bundle), "{:?}", arm.arm);
        assert_eq!(arm.selected_score, Some(expected.score), "{:?}", arm.arm);
    }
}

#[test]
fn a_search_wall_clock_budget_shorter_than_one_call_makes_one_pick_and_selects_nothing() {
    let fixture = Fixture::new();
    let manifest = search_manifest(&fixture, Some(5));
    let clock = ManualClock::new();
    let scorer = ClockedScorer {
        inner: FaithfulScorer::new(),
        clock: clock.clone(),
        step: Duration::from_millis(10),
    };
    let workspace = manifest.cache_dir.join("workspace");

    let report = run_bench(&manifest, &scorer, &clock, &workspace).expect("bench runs");

    // Positive control: the three-pick test above uses this same fixture and
    // step, and under its 25 ms budget the arms do select a bundle.
    let row = &report.rows[0];
    assert!(row.feasible, "{:?}", row.reason);
    assert!(row.search.is_some(), "feasible rows carry a search pool");
    for arm in &row.arms {
        // Each arm starts its first call (0 < 5), and that call ends at 10, past
        // the 5 ms budget. The pick is kept but not within budget, so nothing is
        // selectable and the search stops after one pick.
        assert_eq!(arm.picks.len(), 1, "{:?}", arm.arm);
        assert!(!arm.picks[0].within_budget, "{:?}", arm.arm);
        assert_eq!(arm.full_scorer_calls, 1, "{:?}", arm.arm);
        assert_eq!(arm.stopped_by, Some(SearchStop::WallClock), "{:?}", arm.arm);
        assert_eq!(arm.cost_ms.scorer, 10, "{:?}", arm.arm);
        assert_eq!(arm.selected_bundle, None, "{:?}", arm.arm);
        assert_eq!(arm.selected_score, None, "{:?}", arm.arm);
        assert_eq!(arm.selected_regret, None, "{:?}", arm.arm);
    }
}

#[test]
fn without_a_search_budget_each_arm_makes_one_batched_call_over_its_picks() {
    let fixture = Fixture::new();
    let manifest = search_manifest(&fixture, None);
    let clock = ManualClock::new();
    let scorer = ClockedScorer {
        inner: FaithfulScorer::new(),
        clock: clock.clone(),
        step: Duration::from_millis(10),
    };
    let workspace = manifest.cache_dir.join("workspace");

    let report = run_bench(&manifest, &scorer, &clock, &workspace).expect("bench runs");

    assert_eq!(report.manifest.search_wall_clock_ms, None);
    let row = &report.rows[0];
    assert!(row.feasible, "{:?}", row.reason);
    for arm in &row.arms {
        assert_eq!(arm.picks.len(), 6, "{:?}", arm.arm);
        assert!(
            arm.picks.iter().all(|p| p.within_budget),
            "{:?}: without a budget every pick is within budget",
            arm.arm
        );
        assert_eq!(arm.full_scorer_calls, 1, "{:?}", arm.arm);
        assert_eq!(
            arm.stopped_by,
            Some(SearchStop::FullEvaluations),
            "{:?}",
            arm.arm
        );
        assert_eq!(arm.cost_ms.scorer, 10, "{:?}", arm.arm);
    }
}

#[test]
fn arms_rank_a_separate_search_pool_and_regret_is_paired_within_it() {
    let fixture = Fixture::new();
    let manifest = fixture.manifest(vec![0.2, 0.4], vec![1, 2], 1 << 30);

    let report = run(&manifest);

    let mut checked_rows = 0;
    for row in report.rows.iter().filter(|r| r.feasible) {
        checked_rows += 1;
        let search = row
            .search
            .as_ref()
            .expect("feasible rows carry a search pool");
        let m = search.bundles.len();
        assert!(m > 0, "a feasible row has search bundles");
        let best = row
            .arms
            .iter()
            .filter_map(|a| a.selected_score)
            .fold(f64::NEG_INFINITY, f64::max);
        for arm in &row.arms {
            let mut ordering = arm.search_ranking.clone();
            ordering.sort_unstable();
            assert_eq!(
                ordering,
                (0..m).collect::<Vec<usize>>(),
                "{:?} must rank every search bundle exactly once",
                arm.arm
            );
            assert!(arm.stopped_by.is_some(), "{:?}", arm.arm);
            for pick in &arm.picks {
                assert!(pick.bundle < m, "{:?} picked outside the pool", arm.arm);
            }
            if let Some(i) = arm.selected_bundle {
                assert!(i < m, "{:?} selected outside the pool", arm.arm);
                assert_eq!(
                    arm.selected_hidden_removed,
                    Some(search.bundles[i].hidden_removed)
                );
            }
            assert_eq!(
                arm.selected_regret,
                arm.selected_score.map(|score| best - score),
                "{:?}: regret is the row's best selection minus this arm's",
                arm.arm
            );
            if let Some(regret) = arm.selected_regret {
                assert!(regret >= 0.0, "{:?} regret was {regret}", arm.arm);
            }
        }
    }
    assert!(
        checked_rows > 0,
        "at least one feasible row must be checked"
    );
}

#[test]
fn selected_reductions_come_from_the_search_pool_where_bundles_differ() {
    let fixture = Fixture::new();
    fs::write(
        fixture.path("parents/elite.json"),
        creature_to_json(&chain_parent()).unwrap(),
    )
    .unwrap();
    let mut manifest = fixture.manifest(vec![0.3], vec![1, 2, 3], 1 << 30);
    // The chain's bundles remove different numbers of hidden neurons and
    // synapses, so an index into the wrong pool reports other reductions.
    manifest.audit_pool = 35;

    let report = run(&manifest);

    let mut checked = 0;
    let mut differing_from_audit = 0;
    for row in report.rows.iter().filter(|r| r.feasible) {
        let search = row
            .search
            .as_ref()
            .expect("feasible rows carry a search pool");
        for arm in &row.arms {
            let Some(i) = arm.selected_bundle else {
                continue;
            };
            checked += 1;
            assert_eq!(
                arm.selected_hidden_removed,
                Some(search.bundles[i].hidden_removed),
                "{:?}",
                arm.arm
            );
            assert_eq!(
                arm.selected_synapses_removed,
                Some(search.bundles[i].synapses_removed),
                "{:?}",
                arm.arm
            );
            if let Some(audit) = row.bundles.get(i)
                && (audit.hidden_removed, audit.synapses_removed)
                    != (
                        search.bundles[i].hidden_removed,
                        search.bundles[i].synapses_removed,
                    )
            {
                differing_from_audit += 1;
            }
        }
    }
    assert!(checked > 0, "at least one selection must be checked");
    assert!(
        differing_from_audit > 0,
        "at least one selection must differ from the audit bundle at its index"
    );
}

#[test]
fn the_search_pool_draws_member_sets_the_audit_pool_does_not_hold() {
    let fixture = Fixture::new();
    let manifest = fixture.manifest(vec![0.2, 0.4], vec![1, 2], 1 << 30);

    let report = run(&manifest);

    let member_sets = |uuids: Vec<&Vec<String>>| -> BTreeSet<Vec<String>> {
        uuids
            .into_iter()
            .map(|members| {
                let mut sorted = members.clone();
                sorted.sort();
                sorted
            })
            .collect()
    };
    let mut differing_rows = 0;
    for row in report.rows.iter().filter(|r| r.feasible) {
        let search = row
            .search
            .as_ref()
            .expect("feasible rows carry a search pool");
        let audit_sets = member_sets(row.bundles.iter().map(|b| &b.members).collect());
        let search_sets = member_sets(search.bundles.iter().map(|b| &b.members).collect());
        if search_sets != audit_sets {
            differing_rows += 1;
        }
    }
    assert!(
        differing_rows > 0,
        "at least one row's search pool must hold member sets the audit pool does not"
    );
}

/// A clock that moves by `tick_ms` on every reading, so each timed phase costs
/// a deterministic, non-zero amount that grows with the work it does: the
/// charged (non-scorer) phases, the scorer calls and everything between.
struct TickingClock {
    base: Instant,
    reads: AtomicU64,
    tick_ms: u64,
}

impl TickingClock {
    fn new(tick_ms: u64) -> Self {
        Self {
            base: Instant::now(),
            reads: AtomicU64::new(0),
            tick_ms,
        }
    }
}

impl Clock for TickingClock {
    fn now(&self) -> Instant {
        let n = self.reads.fetch_add(1, Ordering::SeqCst);
        self.base + Duration::from_millis(n * self.tick_ms)
    }
}

const TICK_MS: u64 = 10;

fn run_ticking(manifest: &BenchManifest) -> BenchReport {
    let scorer = FaithfulScorer::new();
    let workspace = manifest.cache_dir.join("workspace");
    run_bench(manifest, &scorer, &TickingClock::new(TICK_MS), &workspace).expect("bench runs")
}

fn arm_row(report: &BenchReport, arm: Arm) -> &neat_ai_ockham::additivity::bench_report::ArmRow {
    report.rows[0]
        .arms
        .iter()
        .find(|a| a.arm == arm)
        .expect("arm present")
}

/// What an arm is charged before its own scorer calls.
fn charged_ms(report: &BenchReport, arm: Arm) -> u64 {
    let cost = &arm_row(report, arm).cost_ms;
    cost.total - cost.scorer
}

#[test]
fn charged_costs_shorten_the_search_of_the_arms_that_pay_them() {
    let fixture = Fixture::new();
    // Reference run without a budget: reads each arm's charged cost.
    let reference = run_ticking(&search_manifest(&fixture, None));
    assert!(reference.rows[0].feasible, "{:?}", reference.rows[0].reason);
    let random_charged = charged_ms(&reference, Arm::Random);
    assert!(random_charged > 0, "random still pays materialisation");
    for arm in [Arm::Current, Arm::AdditiveLoss, Arm::CachedOutput] {
        let cost = &arm_row(&reference, arm).cost_ms;
        assert!(
            cost.total > cost.scorer,
            "{arm:?}: charged phases must cost time"
        );
        assert!(cost.single_proposals > 0 && cost.cache_build > 0, "{arm:?}");
        assert!(
            charged_ms(&reference, arm) >= random_charged + 2 * TICK_MS,
            "{arm:?} pays singles and cache costs that random does not"
        );
    }
    let random_cost = &arm_row(&reference, Arm::Random).cost_ms;
    assert_eq!(random_cost.single_proposals, 0);
    assert_eq!(random_cost.cache_build, 0);
    // The search pool's estimate time is charged to the predictor arms that
    // form it, and to no other: current only reads single losses.
    assert_eq!(random_cost.cache_read_inference, 0);
    let current_read = arm_row(&reference, Arm::Current)
        .cost_ms
        .cache_read_inference;
    for arm in [Arm::AdditiveLoss, Arm::CachedOutput] {
        assert!(
            arm_row(&reference, arm).cost_ms.cache_read_inference > current_read,
            "{arm:?} is charged its search-pool inference on top of the single reads"
        );
    }

    // A budget that leaves random room for exactly three calls.
    let budget = random_charged + 2 * TICK_MS + 1;
    // A fresh fixture: a reused cache directory would make the build cost differ.
    let report = run_ticking(&search_manifest(&Fixture::new(), Some(budget)));
    let row = &report.rows[0];
    assert!(row.feasible, "{:?}", row.reason);
    for arm in ARMS {
        let charged = charged_ms(&report, arm);
        assert_eq!(
            charged,
            charged_ms(&reference, arm),
            "{arm:?}: charged phases do not depend on the budget"
        );
        // A call starts only while charged + scorer time is below the budget.
        let expected = if budget > charged {
            (budget - charged).div_ceil(TICK_MS)
        } else {
            0
        };
        let a = arm_row(&report, arm);
        assert_eq!(a.picks.len() as u64, expected, "{arm:?}");
        assert_eq!(a.full_scorer_calls, expected, "{arm:?}");
        assert_eq!(a.cost_ms.scorer, expected * TICK_MS, "{arm:?}");
        assert_eq!(a.cost_ms.total, charged + expected * TICK_MS, "{arm:?}");
        assert_eq!(a.stopped_by, Some(SearchStop::WallClock), "{arm:?}");
    }
    let random_picks = arm_row(&report, Arm::Random).picks.len();
    assert_eq!(random_picks, 3);
    for arm in [Arm::Current, Arm::AdditiveLoss, Arm::CachedOutput] {
        assert!(
            arm_row(&report, arm).picks.len() < random_picks,
            "{arm:?} is charged singles and cache costs, so it scores fewer picks"
        );
    }
}

#[test]
fn an_arm_with_no_selection_stays_visible_in_the_aggregate() {
    let fixture = Fixture::new();
    let reference = run_ticking(&search_manifest(&fixture, None));
    assert!(reference.rows[0].feasible, "{:?}", reference.rows[0].reason);
    let random_charged = charged_ms(&reference, Arm::Random);
    let cached_charged = charged_ms(&reference, Arm::CachedOutput);
    assert!(
        cached_charged > random_charged + TICK_MS,
        "cachedOutput pays costs random does not"
    );

    // A budget at cachedOutput's charged cost: it cannot start a first call,
    // while random (charged less) still scores and selects.
    // A fresh fixture: a reused cache directory would make the build cost differ.
    let report = run_ticking(&search_manifest(&Fixture::new(), Some(cached_charged)));
    let row = &report.rows[0];
    assert!(row.feasible, "{:?}", row.reason);
    let cached = arm_row(&report, Arm::CachedOutput);
    assert!(cached.picks.is_empty());
    assert_eq!(cached.full_scorer_calls, 0);
    assert_eq!(cached.stopped_by, Some(SearchStop::WallClock));
    assert_eq!(cached.selected_bundle, None);
    assert_eq!(cached.selected_score, None);
    assert_eq!(cached.cost_ms.scorer, 0);
    assert_eq!(cached.cost_ms.total, cached_charged);
    let random = arm_row(&report, Arm::Random);
    let random_score = random.selected_score.expect("random still selects");

    let aggregate = &report.aggregate[0];
    assert_eq!(aggregate.feasible_rows, 1);
    let cached_aggregate = aggregate
        .arms
        .iter()
        .find(|a| a.arm == Arm::CachedOutput)
        .unwrap();
    assert_eq!(cached_aggregate.rows_without_selection, 1);
    assert!(cached_aggregate.selected_delta.is_none());
    let random_aggregate = aggregate
        .arms
        .iter()
        .find(|a| a.arm == Arm::Random)
        .unwrap();
    assert_eq!(random_aggregate.rows_without_selection, 0);
    // The row is excluded from the paired comparison, and said to be.
    assert_eq!(aggregate.paired_rows_excluded_vs_current, 1);
    assert_eq!(aggregate.paired_rows_excluded_vs_random, 1);
    assert!(aggregate.cached_output_vs_current.is_none());
    assert!(aggregate.cached_output_vs_random.is_none());
    // And it is listed as a case random won, with no cachedOutput score.
    let case = report
        .cases_beating_cached_output
        .iter()
        .find(|c| c.arm == Arm::Random)
        .expect("random selected where cachedOutput selected nothing");
    assert_eq!(case.cached_output_score, None);
    assert_eq!(case.arm_score, random_score);
    assert_eq!(case.seed, row.seed);
    let json = serde_json::to_value(case).unwrap();
    assert!(json["cachedOutputScore"].is_null(), "{json}");
}

#[test]
fn a_pairwise_comparison_survives_the_other_arm_selecting_nothing() {
    let fixture = Fixture::new();
    let mut report = run_ticking(&search_manifest(&fixture, None));
    assert!(report.rows[0].feasible, "{:?}", report.rows[0].reason);
    let set = |report: &mut BenchReport, arm: Arm, score: Option<f64>| {
        let a = report.rows[0]
            .arms
            .iter_mut()
            .find(|a| a.arm == arm)
            .expect("arm present");
        a.selected_score = score;
    };
    // Current selects nothing; cachedOutput and random both select, random better.
    set(&mut report, Arm::CachedOutput, Some(1.0));
    set(&mut report, Arm::Current, None);
    set(&mut report, Arm::Random, Some(2.0));
    let budgets = [report.rows[0].budget];
    let (aggregates, beaten) = aggregate(&report.rows, &budgets).expect("aggregates");
    let a = &aggregates[0];
    assert_eq!(a.paired_rows_excluded_vs_current, 1);
    assert_eq!(a.paired_rows_excluded_vs_random, 0);
    assert!(a.cached_output_vs_current.is_none());
    assert!(
        a.cached_output_vs_random.is_some(),
        "random and cachedOutput both selected, so they are compared"
    );
    let case = beaten
        .iter()
        .find(|c| c.arm == Arm::Random)
        .expect("random beat cachedOutput and is listed");
    assert_eq!(case.arm_score, 2.0);
    assert_eq!(case.cached_output_score, Some(1.0));
    assert!(
        beaten.iter().all(|c| c.arm != Arm::Current),
        "current selected nothing, so it beat nothing"
    );

    // And the mirror: random selects nothing, current beats cachedOutput.
    set(&mut report, Arm::Current, Some(3.0));
    set(&mut report, Arm::Random, None);
    let (aggregates, beaten) = aggregate(&report.rows, &budgets).expect("aggregates");
    let a = &aggregates[0];
    assert_eq!(a.paired_rows_excluded_vs_current, 0);
    assert_eq!(a.paired_rows_excluded_vs_random, 1);
    assert!(a.cached_output_vs_current.is_some());
    assert!(a.cached_output_vs_random.is_none());
    let case = beaten
        .iter()
        .find(|c| c.arm == Arm::Current)
        .expect("current beat cachedOutput and is listed");
    assert_eq!(case.cached_output_score, Some(1.0));
    assert!(beaten.iter().all(|c| c.arm != Arm::Random));
}
