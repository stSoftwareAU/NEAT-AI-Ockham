//! `bench-strata` end to end (issue #263).
//!
//! The scorer here is an in-test fixture, not NEAT-AI's: it prices each
//! removed neuron per stratum from a table, so every score is a literal
//! function of the cut. The corpora are real `.bin` directories, so the
//! activation statistics, the manifest and the scratch directory all run for
//! real. Nothing here depends on wall-clock time.

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;

use neat_ai_ockham::corpus::write_bin_file;
use neat_ai_ockham::fixtures::{creature, neuron, synapse};
use neat_ai_ockham::incumbent::sha256_hex;
use neat_ai_ockham::scorer::{DirectoryScorer, ScoreResult, ScorerError, ScorerMode};
use neat_ai_ockham::strata::bench::run_bench;
use neat_ai_ockham::strata::report::{ArmSummary, RunRecord, StrataReport};
use neat_core::{CreatureExport, creature_to_json_pretty, parse_creature_json};
use serde_json::{Value, json};

/// Removing a neuron changes the score by `-cost` plus this per neuron.
const COMPLEXITY_BONUS: f64 = 0.002;

/// Per-(stratum, neuron) removal cost table. Missing entries cost nothing.
type Costs = BTreeMap<(String, String), f64>;

/// Prices every absent neuron per stratum. The stratum is the training
/// directory's final path component. Every call is recorded for assertions.
struct StratumScorer {
    costs: Costs,
    records: BTreeMap<String, u64>,
    calls: RefCell<Vec<(PathBuf, Vec<String>)>>,
}

impl StratumScorer {
    fn new(costs: Costs) -> Self {
        let records = [("north", 1000u64), ("south", 1000), ("rare", 100)]
            .into_iter()
            .map(|(n, c)| (n.to_string(), c))
            .collect();
        Self {
            costs,
            records,
            calls: RefCell::new(Vec::new()),
        }
    }
}

fn read_creature(path: &Path) -> Result<CreatureExport, ScorerError> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| ScorerError::Malformed(format!("{}: {e}", path.display())))?;
    parse_creature_json(&text)
        .map_err(|e| ScorerError::Malformed(format!("{}: {e}", path.display())))
}

impl DirectoryScorer for StratumScorer {
    fn score_directory(
        &self,
        creature_dir: &Path,
        training_dir: &Path,
        mode: ScorerMode,
    ) -> Result<BTreeMap<String, ScoreResult>, ScorerError> {
        assert_eq!(
            mode,
            ScorerMode::Full,
            "the bench must score authoritatively"
        );
        let stratum = training_dir
            .file_name()
            .expect("training dir has a name")
            .to_string_lossy()
            .into_owned();
        let parent = read_creature(&creature_dir.join("baseline.json"))?;
        let hidden: Vec<String> = parent
            .neurons
            .iter()
            .filter(|n| n.neuron_type == "hidden")
            .map(|n| n.uuid.clone())
            .collect();
        let mut out = BTreeMap::new();
        let mut stems = Vec::new();
        let entries = std::fs::read_dir(creature_dir)
            .map_err(|e| ScorerError::Spawn(format!("{}: {e}", creature_dir.display())))?;
        for entry in entries {
            let path = entry.map_err(|e| ScorerError::Spawn(e.to_string()))?.path();
            if path.extension().and_then(|e| e.to_str()) != Some("json") {
                continue;
            }
            let stem = path
                .file_stem()
                .expect("json file has a stem")
                .to_string_lossy()
                .into_owned();
            let candidate = read_creature(&path)?;
            let present: Vec<&str> = candidate.neurons.iter().map(|n| n.uuid.as_str()).collect();
            let absent: Vec<&String> = hidden
                .iter()
                .filter(|h| !present.contains(&h.as_str()))
                .collect();
            let cost: f64 = absent
                .iter()
                .map(|u| {
                    self.costs
                        .get(&(stratum.clone(), (*u).clone()))
                        .copied()
                        .unwrap_or(0.0)
                })
                .sum();
            let score = 0.5 - cost + COMPLEXITY_BONUS * absent.len() as f64;
            out.insert(
                stem.clone(),
                ScoreResult {
                    score,
                    error: 1.0 - score,
                    complexity_penalty: 0.0,
                    record_count: self.records.get(&stratum).copied().unwrap_or(0),
                    sample_rate: None,
                    gpu_backend: None,
                    cost_name: Some("MSE".to_string()),
                    time_taken: 0.0,
                },
            );
            stems.push(stem);
        }
        stems.sort();
        self.calls
            .borrow_mut()
            .push((training_dir.to_path_buf(), stems));
        Ok(out)
    }

    fn identity(&self) -> String {
        "stratum-scorer-fixture".to_string()
    }
}

/// A scorer that always fails, as a crashed binary would.
struct FailingScorer;

impl DirectoryScorer for FailingScorer {
    fn score_directory(
        &self,
        _creature_dir: &Path,
        _training_dir: &Path,
        _mode: ScorerMode,
    ) -> Result<BTreeMap<String, ScoreResult>, ScorerError> {
        Err(ScorerError::Failed {
            status: "exit status: 3".to_string(),
            stderr: "boom".to_string(),
        })
    }

    fn identity(&self) -> String {
        "failing-scorer-fixture".to_string()
    }
}

/// Latency probe for tests: the synapse count, as `f64`.
fn synapse_probe() -> impl Fn(&CreatureExport) -> Result<f64, String> {
    |c: &CreatureExport| Ok(c.synapses.len() as f64)
}

/// Parent: `h0` is the trap (a large fan-in, so it removes more synapses),
/// `h1`–`h5` are neutral neurons with two inputs each. Non-identity squashes
/// keep core from splicing them away.
fn parent_creature() -> CreatureExport {
    let mut neurons = vec![neuron("hidden", "h0", 0.0, Some("HARD_TANH"))];
    let mut synapses = Vec::new();
    for i in 0..4 {
        synapses.push(synapse(&format!("input-{i}"), "h0", 0.5));
    }
    synapses.push(synapse("h0", "output-0", 1.0));
    for h in 1..=5 {
        let uuid = format!("h{h}");
        neurons.push(neuron("hidden", &uuid, 0.0, Some("HARD_TANH")));
        synapses.push(synapse("input-0", &uuid, 0.3));
        synapses.push(synapse("input-1", &uuid, 0.3));
        synapses.push(synapse(&uuid, "output-0", 0.2));
    }
    neurons.push(neuron("output", "output-0", 0.0, Some("IDENTITY")));
    creature(4, 1, neurons, synapses)
}

fn write_stratum(dir: &Path, records: usize, seed: u32) {
    std::fs::create_dir_all(dir).unwrap();
    let rows: Vec<(Vec<f32>, Vec<f32>)> = (0..records as u32)
        .map(|i| {
            let i = i + seed * 7919;
            (
                vec![
                    ((i * 7) % 11) as f32 / 10.0,
                    ((i * 3) % 13) as f32 / 13.0,
                    ((i * 5) % 7) as f32 / 7.0,
                    ((i * 11) % 17) as f32 / 17.0,
                ],
                vec![((i * 2) % 5) as f32 / 5.0],
            )
        })
        .collect();
    write_bin_file(&dir.join("part.bin"), &rows).unwrap();
}

/// A temp tree holding the parent, three calibration strata (two large, one
/// small "rare") and three validation strata, plus a manifest writer.
struct Fixture {
    dir: tempfile::TempDir,
}

impl Fixture {
    fn new() -> Self {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        std::fs::write(
            root.join("parent.json"),
            creature_to_json_pretty(&parent_creature()).unwrap(),
        )
        .unwrap();
        for (side, counts) in [("cal", [1000, 1000, 100]), ("val", [800, 800, 100])] {
            for (name, (count, seed)) in ["north", "south", "rare"]
                .into_iter()
                .zip(counts.into_iter().zip([1u32, 2, 3]))
            {
                write_stratum(&root.join(side).join(name), count, seed);
            }
        }
        Self { dir }
    }

    fn root(&self) -> &Path {
        self.dir.path()
    }

    fn parent(&self) -> PathBuf {
        self.root().join("parent.json")
    }

    fn report(&self) -> PathBuf {
        self.root().join("report.json")
    }

    fn validation_dirs(&self) -> Vec<PathBuf> {
        ["north", "south", "rare"]
            .iter()
            .map(|n| self.root().join("val").join(n))
            .collect()
    }

    /// Write `manifest.json` from the default manifest after `edit`.
    fn manifest(&self, edit: impl FnOnce(&mut Value)) -> PathBuf {
        let mut value = json!({
            "parent": "parent.json",
            "calibration": [
                {"name": "north", "dir": "cal/north"},
                {"name": "south", "dir": "cal/south"},
                {"name": "rare", "dir": "cal/rare"}
            ],
            "validation": [
                {"name": "north", "dir": "val/north"},
                {"name": "south", "dir": "val/south"},
                {"name": "rare", "dir": "val/rare"}
            ],
            "seeds": [1, 2, 3, 4, 5],
            "fitSeed": 9001,
            "bundleSizes": [4, 2],
            "bundlesPerArm": 2,
            "paretoCandidates": 4,
            "archiveCap": 8,
            "latencyRecords": 16,
            "success": {"maxFitnessLoss": 0.0, "minRegressionReduction": 1}
        });
        edit(&mut value);
        let path = self.root().join("manifest.json");
        std::fs::write(&path, serde_json::to_string_pretty(&value).unwrap()).unwrap();
        path
    }
}

/// The trap costs a lot only on the rare stratum, and removing it slightly
/// helps the two large strata.
fn trap_costs() -> Costs {
    let mut c = Costs::new();
    c.insert(("north".into(), "h0".into()), -0.03);
    c.insert(("south".into(), "h0".into()), -0.03);
    c.insert(("rare".into(), "h0".into()), 0.5);
    c
}

/// Every neuron in every stratum costs the same: no stratum-specific neuron.
fn uniform_costs(cost: f64) -> Costs {
    let mut c = Costs::new();
    for stratum in ["north", "south", "rare"] {
        for h in ["h0", "h1", "h2", "h3", "h4", "h5"] {
            c.insert((stratum.into(), h.into()), cost);
        }
    }
    c
}

fn run_of<'a>(report: &'a StrataReport, seed: u64, arm: &str) -> &'a RunRecord {
    report
        .runs
        .iter()
        .find(|r| r.seed == seed && r.arm == arm)
        .unwrap_or_else(|| panic!("no run for seed {seed} arm {arm}"))
}

fn arm_of<'a>(report: &'a StrataReport, arm: &str) -> &'a ArmSummary {
    report
        .arms
        .iter()
        .find(|a| a.name == arm)
        .unwrap_or_else(|| panic!("no arm summary for {arm}"))
}

fn bin() -> Command {
    Command::new(env!("CARGO_BIN_EXE_neat_ai_ockham"))
}

#[test]
fn end_to_end_worst_stratum_avoids_the_minority_regression_that_aggregate_walks_into() {
    let fx = Fixture::new();
    let manifest = fx.manifest(|_| {});
    let scorer = StratumScorer::new(trap_costs());
    let probe = synapse_probe();
    let parent_before = std::fs::read(fx.parent()).unwrap();

    let report = run_bench(&manifest, &fx.report(), &scorer, &probe).expect("the bench runs");

    // The report is on disk, under the documented schema.
    let on_disk: Value = serde_json::from_slice(&std::fs::read(fx.report()).unwrap()).unwrap();
    assert_eq!(on_disk["schema"], "ockham-strata-bench/1");

    // Every arm on every seed, in report order.
    let arms: Vec<&str> = report.runs.iter().map(|r| r.arm.as_str()).collect();
    assert_eq!(report.runs.len(), 4 * 5, "{arms:?}");
    for seed in [1, 2, 3, 4, 5] {
        for arm in ["aggregate", "worst-stratum", "pareto", "random"] {
            run_of(&report, seed, arm);
        }
        // Every arm scored exactly bundlesPerArm bundles on this seed.
        for arm in ["aggregate", "worst-stratum", "pareto", "random"] {
            assert_eq!(
                run_of(&report, seed, arm).bundles_scored,
                2,
                "seed {seed} {arm}"
            );
        }
    }

    // The aggregate arm walks into the trap; the worst-stratum arm avoids it.
    let aggregate = arm_of(&report, "aggregate");
    let worst = arm_of(&report, "worst-stratum");
    assert_eq!(worst.minority_regressions, 0);
    assert!(aggregate.minority_regressions > 0, "{aggregate:?}");
    for seed in [1, 2, 3, 4, 5] {
        let selected = run_of(&report, seed, "aggregate")
            .selected
            .clone()
            .unwrap_or_default();
        assert!(
            selected.contains(&"h0".to_string()),
            "seed {seed}: {selected:?}"
        );
    }

    // The predeclared criterion holds for worst-stratum.
    let criterion = report
        .criteria
        .iter()
        .find(|c| c.arm == "worst-stratum")
        .expect("criterion for worst-stratum");
    assert!(criterion.met, "{criterion:?}");
    assert!(criterion.regression_reduction >= 1, "{criterion:?}");

    // The parent is untouched and the scratch directory is gone.
    assert_eq!(std::fs::read(fx.parent()).unwrap(), parent_before);
    assert!(!fx.root().join("report.json.work").exists());

    // Single cuts and fits never reach a validation stratum.
    let validation = fx.validation_dirs();
    for (dir, stems) in scorer.calls.borrow().iter() {
        if validation.contains(dir) {
            for stem in stems {
                assert!(
                    !stem.starts_with("single-") && !stem.starts_with("fit-"),
                    "{stem} was scored against {}",
                    dir.display()
                );
            }
        }
    }

    // The parent's checksum is recorded as the SHA-256 of its bytes.
    assert_eq!(report.versions.parent_checksum, sha256_hex(&parent_before));
    assert_eq!(report.versions.scorer_identity, "stratum-scorer-fixture");
}

#[test]
fn uniform_costs_do_not_meet_the_worst_stratum_criterion() {
    let fx = Fixture::new();
    let manifest = fx.manifest(|_| {});
    let scorer = StratumScorer::new(uniform_costs(-0.005));
    let report =
        run_bench(&manifest, &fx.report(), &scorer, &synapse_probe()).expect("the bench runs");

    let criterion = report
        .criteria
        .iter()
        .find(|c| c.arm == "worst-stratum")
        .expect("criterion for worst-stratum");
    assert_eq!(criterion.regression_reduction, 0, "{criterion:?}");
    assert!(
        criterion.regression_reduction < 1,
        "minRegressionReduction is 1, so 0 must not meet it"
    );
    assert!(!criterion.met, "{criterion:?}");
}

#[test]
fn existing_scratch_directory_is_refused_and_left_untouched() {
    let fx = Fixture::new();
    let manifest = fx.manifest(|_| {});
    let work = fx.root().join("report.json.work");
    std::fs::create_dir(&work).unwrap();
    std::fs::write(work.join("marker.txt"), "keep me").unwrap();
    let scorer = StratumScorer::new(trap_costs());

    let err = run_bench(&manifest, &fx.report(), &scorer, &synapse_probe())
        .expect_err("an existing scratch directory must be refused");

    assert!(err.contains("already exists"), "{err}");
    assert_eq!(
        std::fs::read_to_string(work.join("marker.txt")).unwrap(),
        "keep me"
    );
    assert!(!fx.report().exists(), "no report may be written");
    assert!(scorer.calls.borrow().is_empty(), "nothing may be scored");
}

#[test]
fn scorer_failure_names_its_stage_and_removes_the_scratch_directory() {
    let fx = Fixture::new();
    let manifest = fx.manifest(|_| {});

    let err = run_bench(&manifest, &fx.report(), &FailingScorer, &synapse_probe())
        .expect_err("a failing scorer must fail the run");

    assert!(err.contains("stage 'importance'"), "{err}");
    assert!(err.contains("boom"), "{err}");
    assert!(
        !fx.root().join("report.json.work").exists(),
        "scratch removed"
    );
    assert!(!fx.report().exists(), "no report written on failure");
}

#[test]
fn bench_strata_help_lists_its_flags() {
    let out = bin().args(["bench-strata", "--help"]).output().unwrap();
    assert!(out.status.success());
    let text = String::from_utf8_lossy(&out.stdout).into_owned();
    for flag in ["--out", "--scorer", "--scorer-arg"] {
        assert!(text.contains(flag), "--help lacks {flag}: {text}");
    }
}

#[test]
fn bench_strata_names_a_missing_manifest_and_exits_non_zero() {
    let out = bin()
        .args(["bench-strata", "/nonexistent.json", "--out", "x"])
        .output()
        .unwrap();
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr).into_owned();
    assert!(stderr.contains("/nonexistent.json"), "{stderr}");
}
