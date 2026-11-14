//! The `bench-strata` driver (issue #263).
//!
//! [`run_bench`] loads the manifest, measures the calibration strata, scores
//! every single-neuron cut once per calibration stratum, fits the scales and
//! temperatures, then runs each arm on each seed and validates the winners.
//! Every creature it writes lives under `<report>.work/`, which is removed at
//! the end of the run whether it succeeded or not. The parent is only read.

use std::cell::{Cell, RefCell};
use std::collections::{BTreeMap, HashMap};
use std::ops::ControlFlow;
use std::path::{Path, PathBuf};
use std::time::Instant;

use neat_core::training_data::TrainingDataConfig;
use neat_core::{CreatureExport, compile_creature, creature_to_json};

use crate::corpus::{CorpusInfo, RecordRange, corpus_info, for_each_selected_chunk};
use crate::incumbent::{Incumbent, load_incumbent};
use crate::ordering::SplitMix64;
use crate::prune::{GroupMember, PrunedCandidate, prune_hidden_group, prune_hidden_neuron};
use crate::scorer::{DirectoryScorer, ScoreResult, ScorerMode};
use crate::stats::{DEFAULT_CHUNK_RECORDS, SampleSpec, compute_activation_stats};
use crate::strata::ARMS;
use crate::strata::manifest::{self, Loaded, Manifest, StratumDir};
use crate::strata::pareto::{self, Point};
use crate::strata::report::{
    ArmSummary, Costs, Coverage, Criterion, CriterionInput, Dist, EVIDENCE, Failure, FitCriterion,
    Fitted, Method, Paired, REPORT_SCHEMA, RunRecord, StrataReport, StratumVersion, Temperatures,
    Versions, criterion,
};
use crate::strata::sampler::{
    aggregate_statistic, draw_bundle, fit_scales, normalise, pick_temperature, pooled_mean,
    predicted_cost, select_winner, stratum_weights, worst_statistic,
};

const AGGREGATE: &str = ARMS[0];
const WORST: &str = ARMS[1];
const PARETO: &str = ARMS[2];
const RANDOM: &str = ARMS[3];

/// Milliseconds in an hour, for the wins-per-hour figure.
const MS_PER_HOUR: f64 = 3.6e6;

/// Latency measurements per creature. The median is reported.
const LATENCY_REPETITIONS: usize = 3;

/// The scratch directory for a report: a sibling named `<report>.work`.
pub fn work_dir_for(out: &Path) -> PathBuf {
    let mut name = out.file_name().unwrap_or_default().to_os_string();
    name.push(".work");
    out.with_file_name(name)
}

/// Run the experiment and write its report to `out_path`.
///
/// The scratch directory is `<out_path>.work`. It must not already exist: an
/// existing one is refused and left untouched. It is removed before this
/// returns, on success and on failure alike. A report is written only when the
/// run succeeds and the scratch directory is gone.
///
/// # Errors
///
/// A message naming the stage that failed, the manifest problem, an existing
/// scratch directory, a scorer failure, or a failed cleanup.
pub fn run_bench(
    manifest_path: &Path,
    out_path: &Path,
    scorer: &dyn DirectoryScorer,
    latency: &dyn Fn(&CreatureExport) -> Result<f64, String>,
) -> Result<StrataReport, String> {
    let loaded = manifest::load(manifest_path)?;
    let work = work_dir_for(out_path);
    if work.exists() {
        return Err(format!(
            "scratch directory '{}' already exists; remove it or choose another --out",
            work.display()
        ));
    }
    std::fs::create_dir(&work)
        .map_err(|e| format!("cannot create scratch directory '{}': {e}", work.display()))?;
    let result = execute(&loaded, &work, scorer, latency);
    let removal = std::fs::remove_dir_all(&work);
    let report = match (result, removal) {
        (Ok(report), Ok(())) => report,
        (Ok(_), Err(e)) => {
            return Err(format!(
                "cleanup of scratch directory '{}' failed: {e}",
                work.display()
            ));
        }
        (Err(e), Ok(())) => return Err(e),
        (Err(e), Err(r)) => {
            return Err(format!(
                "{e}; cleanup of scratch directory '{}' also failed: {r}",
                work.display()
            ));
        }
    };
    write_report_atomically(out_path, &report)?;
    Ok(report)
}

/// A latency probe that compiles the creature and times its activations over
/// the first `records` records of `dir`. Three passes; the median ms per
/// record is returned. Indicative only: wall-clock figures are never asserted.
pub fn wall_clock_probe(
    dir: PathBuf,
    records: u64,
) -> impl Fn(&CreatureExport) -> Result<f64, String> {
    move |creature: &CreatureExport| measure_ms_per_record(creature, &dir, records)
}

fn measure_ms_per_record(
    creature: &CreatureExport,
    dir: &Path,
    records: u64,
) -> Result<f64, String> {
    let cfg = TrainingDataConfig::new(creature.input, creature.output);
    let ranges = [RecordRange {
        start: 0,
        len: records,
    }];
    let mut samples = Vec::with_capacity(LATENCY_REPETITIONS);
    for _ in 0..LATENCY_REPETITIONS {
        let mut net = compile_creature(creature)
            .map_err(|e| format!("cannot compile creature for the latency probe: {e}"))?;
        let mut visited = 0u64;
        let started = Instant::now();
        for_each_selected_chunk(dir, &cfg, &ranges, DEFAULT_CHUNK_RECORDS, |chunk| {
            for r in 0..chunk.records {
                let inputs = &chunk.inputs[r * creature.input..(r + 1) * creature.input];
                let _ = net.activate(inputs, creature.output);
            }
            visited += chunk.records as u64;
            Ok(ControlFlow::Continue(()))
        })?;
        let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
        if visited == 0 {
            return Err(format!(
                "latency probe found no records in '{}'",
                dir.display()
            ));
        }
        samples.push(elapsed_ms / visited as f64);
    }
    samples.sort_by(f64::total_cmp);
    Ok(samples[LATENCY_REPETITIONS / 2])
}

fn write_report_atomically(out: &Path, report: &StrataReport) -> Result<(), String> {
    let mut json = serde_json::to_string_pretty(report)
        .map_err(|e| format!("cannot serialise report: {e}"))?;
    json.push('\n');
    let mut tmp_name = out.file_name().unwrap_or_default().to_os_string();
    tmp_name.push(".tmp");
    let tmp = out.with_file_name(tmp_name);
    std::fs::write(&tmp, json)
        .map_err(|e| format!("cannot write report '{}': {e}", tmp.display()))?;
    std::fs::rename(&tmp, out)
        .map_err(|e| format!("cannot move report into place at '{}': {e}", out.display()))
}

fn execute(
    loaded: &Loaded,
    work: &Path,
    scorer: &dyn DirectoryScorer,
    latency: &dyn Fn(&CreatureExport) -> Result<f64, String>,
) -> Result<StrataReport, String> {
    let started = Instant::now();
    let m = &loaded.manifest;
    let incumbent = load_incumbent(&m.parent).map_err(|e| format!("stage 'parent': {e}"))?;
    let cfg = TrainingDataConfig::new(incumbent.creature.input, incumbent.creature.output);
    let cal = open_strata(&m.calibration, &cfg, "calibration")?;
    let val = open_strata(&m.validation, &cfg, "validation")?;
    let cal_weights = weights_of(&cal, "calibration")?;
    let val_weights = weights_of(&val, "validation")?;
    let mut run = Run {
        manifest: m,
        manifest_sha256: loaded.sha256.clone(),
        incumbent,
        work: work.to_path_buf(),
        scorer,
        latency,
        cal,
        val,
        cal_weights,
        val_weights,
        base_cal: Vec::new(),
        base_total: 0.0,
        norm: Vec::new(),
        measured: Vec::new(),
        calls: Cell::new(0),
        probe_ms: Cell::new(0.0),
        failures: RefCell::new(Vec::new()),
    };
    let prep = run.prepare(started)?;
    let parent_latency = run.probe(&run.incumbent.creature)?;
    let (runs, validation_calls) = run.run_seeds(&prep)?;
    Ok(run.assemble(&prep, runs, validation_calls, parent_latency))
}

fn open_strata(
    dirs: &[StratumDir],
    cfg: &TrainingDataConfig,
    side: &str,
) -> Result<Vec<Stratum>, String> {
    dirs.iter()
        .map(|s| -> Result<Stratum, String> {
            let info = corpus_info(&s.dir, cfg)
                .map_err(|e| format!("stage 'corpus' ({side} stratum '{}'): {e}", s.name))?;
            Ok(Stratum {
                name: s.name.clone(),
                dir: s.dir.clone(),
                info,
            })
        })
        .collect()
}

fn weights_of(strata: &[Stratum], side: &str) -> Result<Vec<f64>, String> {
    let counts: Vec<u64> = strata.iter().map(|s| s.info.record_count).collect();
    if counts.iter().sum::<u64>() == 0 {
        return Err(format!("stage 'corpus': the {side} strata hold no records"));
    }
    Ok(stratum_weights(&counts))
}

/// One named corpus stratum with its identity and record count.
struct Stratum {
    name: String,
    dir: PathBuf,
    info: CorpusInfo,
}

/// A hidden neuron whose single cut core accepted, with its step-2 mean.
struct Measured {
    /// Position among the parent's hidden neurons, for file names.
    index: usize,
    uuid: String,
    mean: Option<f64>,
}

/// A bundle core built, with the positional bookkeeping the arms need.
struct Built {
    /// Bundle index within its arm; orders ties and names the file.
    j: usize,
    /// Member UUIDs in draw order.
    members: Vec<String>,
    pruned: PrunedCandidate,
    /// Measured latency, when the arm already probed it (Pareto).
    latency: Option<f64>,
}

/// The selected bundle of one arm for one seed.
struct Winner {
    members: Vec<String>,
    creature: CreatureExport,
    hidden_removed: usize,
    synapses_removed: usize,
    latency_ms: Option<f64>,
}

/// One arm's outcome for one seed, before validation.
struct ArmRun {
    arm: &'static str,
    bundles_scored: usize,
    refused: usize,
    wins: usize,
    winner: Option<Winner>,
    calibration_delta: Option<f64>,
    arm_ms: f64,
    scorer_calls: u64,
    archive_size: Option<usize>,
}

/// Validation deltas of one arm for one seed.
struct ValidationDelta {
    strata: BTreeMap<String, f64>,
    weighted: f64,
}

/// Everything the seed loop needs from preparation.
struct Prep {
    prep_ms: f64,
    prep_calls: u64,
    scales: BTreeMap<String, f64>,
    tau_agg: f64,
    tau_worst: f64,
    fit: Vec<FitCriterion>,
    coverage: Coverage,
    agg: Vec<f64>,
    worst: Vec<f64>,
}

/// Start time and scorer-call count of one arm, for its own cost figures.
struct ArmTimer {
    started: Instant,
    calls: u64,
}

/// The state of one run: the parent, the strata and the scorer.
struct Run<'a> {
    manifest: &'a Manifest,
    manifest_sha256: String,
    incumbent: Incumbent,
    work: PathBuf,
    scorer: &'a dyn DirectoryScorer,
    latency: &'a dyn Fn(&CreatureExport) -> Result<f64, String>,
    cal: Vec<Stratum>,
    val: Vec<Stratum>,
    cal_weights: Vec<f64>,
    val_weights: Vec<f64>,
    /// Parent's calibration score per stratum.
    base_cal: Vec<f64>,
    /// Calibration-weighted parent score.
    base_total: f64,
    /// Normalised importance `[stratum][measured neuron]`.
    norm: Vec<Vec<f64>>,
    measured: Vec<Measured>,
    calls: Cell<u64>,
    probe_ms: Cell<f64>,
    failures: RefCell<Vec<Failure>>,
}

impl Run<'_> {
    fn bundle_size(&self, j: usize) -> usize {
        self.manifest.bundle_sizes[j % self.manifest.bundle_sizes.len()]
    }

    fn members_of(&self, positions: &[usize]) -> Vec<String> {
        positions
            .iter()
            .map(|&k| self.measured[k].uuid.clone())
            .collect()
    }

    fn build(&self, positions: &[usize]) -> Result<PrunedCandidate, String> {
        let group: Vec<GroupMember> = positions
            .iter()
            .map(|&k| GroupMember {
                uuid: self.measured[k].uuid.clone(),
                mean: self.measured[k].mean,
            })
            .collect();
        prune_hidden_group(&self.incumbent.creature, &group).map_err(|e| e.to_string())
    }

    fn fail(
        &self,
        stage: &str,
        seed: Option<u64>,
        arm: Option<&str>,
        uuid: Option<&str>,
        detail: String,
    ) {
        self.failures.borrow_mut().push(Failure {
            stage: stage.to_string(),
            seed,
            arm: arm.map(str::to_string),
            uuid: uuid.map(str::to_string),
            detail,
        });
    }

    fn score(
        &self,
        dir: &Path,
        training: &Path,
        stage: &str,
        stratum: &str,
    ) -> Result<BTreeMap<String, ScoreResult>, String> {
        self.calls.set(self.calls.get() + 1);
        self.scorer
            .score_directory(dir, training, ScorerMode::Full)
            .map_err(|e| format!("stage '{stage}' (stratum '{stratum}'): scorer failed: {e}"))
    }

    fn probe(&self, creature: &CreatureExport) -> Result<f64, String> {
        let started = Instant::now();
        let result = (self.latency)(creature).map_err(|e| format!("stage 'latency': {e}"));
        self.probe_ms
            .set(self.probe_ms.get() + started.elapsed().as_secs_f64() * 1000.0);
        result
    }

    /// Write `baseline.json` and each named creature into `dir`.
    fn write_dir(&self, dir: &Path, creatures: &[(String, &CreatureExport)]) -> Result<(), String> {
        std::fs::create_dir_all(dir)
            .map_err(|e| format!("cannot create '{}': {e}", dir.display()))?;
        write_file(&dir.join("baseline.json"), &self.incumbent.text)?;
        for (name, creature) in creatures {
            let text = creature_to_json(creature)
                .map_err(|e| format!("cannot serialise '{name}': {e}"))?;
            write_file(&dir.join(format!("{name}.json")), &text)?;
        }
        Ok(())
    }

    /// Steps 1 to 6: means, importance, scales and temperatures.
    fn prepare(&mut self, started: Instant) -> Result<Prep, String> {
        let m = self.manifest;
        let creature = self.incumbent.creature.clone();
        let checksum = self.incumbent.checksum.clone();
        let sample = SampleSpec::with_max_records(m.stats_records);

        // Step 2: record-weighted means over the calibration strata only.
        let mut parts: HashMap<String, Vec<(u64, f64)>> = HashMap::new();
        let mut stats_records = BTreeMap::new();
        for s in &self.cal {
            let stats = compute_activation_stats(
                &creature,
                &checksum,
                &s.dir,
                &s.info,
                DEFAULT_CHUNK_RECORDS,
                &sample,
            )
            .map_err(|e| format!("stage 'stats' (stratum '{}'): {e}", s.name))?;
            stats_records.insert(s.name.clone(), stats.record_count);
            for n in &stats.neurons {
                parts
                    .entry(n.uuid.clone())
                    .or_default()
                    .push((n.count, n.mean));
            }
        }
        let hidden: Vec<(usize, String, Option<f64>)> = creature
            .neurons
            .iter()
            .filter(|n| n.neuron_type == "hidden")
            .enumerate()
            .map(|(h, n)| {
                let mean = parts.get(&n.uuid).and_then(|p| pooled_mean(p));
                (h, n.uuid.clone(), mean)
            })
            .collect();

        // Step 3: importance from one single cut per hidden neuron.
        let imp_dir = self.work.join("importance");
        self.write_dir(&imp_dir, &[])?;
        let mut measured = Vec::new();
        let mut refused = 0usize;
        for (h, uuid, mean) in &hidden {
            match prune_hidden_neuron(&creature, uuid, *mean, None) {
                Ok(pruned) => {
                    let text = creature_to_json(&pruned.creature)
                        .map_err(|e| format!("cannot serialise single cut '{uuid}': {e}"))?;
                    write_file(&imp_dir.join(format!("single-{h}.json")), &text)?;
                    measured.push(Measured {
                        index: *h,
                        uuid: uuid.clone(),
                        mean: *mean,
                    });
                }
                Err(e) => {
                    refused += 1;
                    self.fail("importance", None, None, Some(uuid.as_str()), e.to_string());
                }
            }
        }
        let mut importance: Vec<Vec<f64>> = Vec::with_capacity(self.cal.len());
        let mut base_cal = Vec::with_capacity(self.cal.len());
        for s in &self.cal {
            let results = self.score(&imp_dir, &s.dir, "importance", &s.name)?;
            let base = require(&results, "baseline", "importance", &s.name)?;
            let mut row = Vec::with_capacity(measured.len());
            for mm in &measured {
                let single = require(
                    &results,
                    &format!("single-{}", mm.index),
                    "importance",
                    &s.name,
                )?;
                row.push(base - single);
            }
            importance.push(row);
            base_cal.push(base);
        }
        self.base_total = base_cal
            .iter()
            .zip(&self.cal_weights)
            .map(|(b, w)| b * w)
            .sum();
        self.base_cal = base_cal;

        // Step 4: scales from calibration only; statistics on the normalised matrix.
        let scales_vec = fit_scales(&importance);
        let norm = normalise(&importance, &scales_vec);
        let agg = aggregate_statistic(&norm, &self.cal_weights);
        let worst = worst_statistic(&norm);
        self.norm = norm;
        self.measured = measured;

        // Step 6: temperature fit, calibration only, from fitSeed.
        let fit_dir = self.work.join("fit");
        let mut fits: Vec<(String, CreatureExport)> = Vec::new();
        let mut cells: Vec<FitCell> = Vec::new();
        for (arm, x) in [(AGGREGATE, &agg), (WORST, &worst)] {
            for (ti, tau) in m.temperatures.iter().enumerate() {
                let mut rng = SplitMix64(m.fit_seed);
                let mut stems = Vec::new();
                let mut refused = 0usize;
                for j in 0..m.bundles_per_arm {
                    let positions = draw_bundle(&mut rng, x, self.bundle_size(j), Some(*tau));
                    match self.build(&positions) {
                        Ok(pruned) => {
                            let stem = format!("fit-{arm}-{ti}-{j}");
                            fits.push((stem.clone(), pruned.creature));
                            stems.push(stem);
                        }
                        Err(e) => {
                            refused += 1;
                            self.fail("fit", None, Some(arm), None, e);
                        }
                    }
                }
                cells.push(FitCell {
                    arm,
                    tau: *tau,
                    stems,
                    refused,
                });
            }
        }
        let fit_refs: Vec<(String, &CreatureExport)> =
            fits.iter().map(|(n, c)| (n.clone(), c)).collect();
        self.write_dir(&fit_dir, &fit_refs)?;
        let mut weighted: BTreeMap<String, f64> = BTreeMap::new();
        for (t, s) in self.cal.iter().enumerate() {
            let results = self.score(&fit_dir, &s.dir, "fit", &s.name)?;
            for cell in &cells {
                for stem in &cell.stems {
                    let score = require(&results, stem, "fit", &s.name)?;
                    *weighted.entry(stem.clone()).or_insert(0.0) += self.cal_weights[t] * score;
                }
            }
        }
        let mut fit_rows = Vec::with_capacity(cells.len());
        let mut values_agg = Vec::new();
        let mut values_worst = Vec::new();
        for cell in &cells {
            let mean_score = if cell.stems.is_empty() {
                None
            } else {
                let sum: f64 = cell.stems.iter().map(|s| weighted[s]).sum();
                Some(sum / cell.stems.len() as f64)
            };
            let value = mean_score.unwrap_or(f64::NEG_INFINITY);
            if cell.arm == AGGREGATE {
                values_agg.push(value);
            } else {
                values_worst.push(value);
            }
            fit_rows.push(FitCriterion {
                arm: cell.arm.to_string(),
                temperature: cell.tau,
                criterion: mean_score,
                built: cell.stems.len(),
                refused: cell.refused,
            });
        }
        let tau_agg = m.temperatures[pick_temperature(&values_agg)];
        let tau_worst = m.temperatures[pick_temperature(&values_worst)];

        let scales: BTreeMap<String, f64> = self
            .cal
            .iter()
            .zip(&scales_vec)
            .map(|(s, v)| (s.name.clone(), *v))
            .collect();
        let coverage = Coverage {
            hidden_neurons: hidden.len(),
            measured: self.measured.len(),
            refused_singles: refused,
            stats_records,
        };
        Ok(Prep {
            prep_ms: started.elapsed().as_secs_f64() * 1000.0,
            prep_calls: self.calls.get(),
            scales,
            tau_agg,
            tau_worst,
            fit: fit_rows,
            coverage,
            agg,
            worst,
        })
    }

    fn timer(&self) -> ArmTimer {
        ArmTimer {
            started: Instant::now(),
            calls: self.calls.get(),
        }
    }

    /// Score `bundles` against every calibration stratum in `dir` and return
    /// each bundle's calibration delta, in input order.
    fn score_bundles(
        &self,
        dir: &Path,
        stage: &str,
        bundles: &[&Built],
    ) -> Result<Vec<f64>, String> {
        let names: Vec<String> = bundles.iter().map(|b| format!("bundle-{}", b.j)).collect();
        let refs: Vec<(String, &CreatureExport)> = names
            .iter()
            .zip(bundles)
            .map(|(n, b)| (n.clone(), &b.pruned.creature))
            .collect();
        self.write_dir(dir, &refs)?;
        let mut totals = vec![0.0; bundles.len()];
        for (t, s) in self.cal.iter().enumerate() {
            let results = self.score(dir, &s.dir, stage, &s.name)?;
            for (b, name) in names.iter().enumerate() {
                totals[b] += self.cal_weights[t] * require(&results, name, stage, &s.name)?;
            }
        }
        Ok(totals.into_iter().map(|v| v - self.base_total).collect())
    }

    /// Select the winner among scored bundles and assemble the arm's outcome.
    fn finish_arm(
        &self,
        timer: ArmTimer,
        seed: u64,
        arm: &'static str,
        refused: usize,
        scored: &[&Built],
        archive_size: Option<usize>,
    ) -> Result<ArmRun, String> {
        let m = self.manifest;
        let dir = self
            .work
            .join("eval")
            .join(format!("seed-{seed}"))
            .join(arm);
        let deltas = self.score_bundles(&dir, "evaluate", scored)?;
        let wins = deltas.iter().filter(|d| **d > m.min_improvement).count();
        let (winner, calibration_delta) = match select_winner(&deltas, m.min_improvement) {
            None => (None, None),
            Some(b) => {
                let chosen = scored[b];
                let before = &chosen.pruned.before;
                let after = &chosen.pruned.after;
                let latency_ms = match chosen.latency {
                    Some(l) => Some(l),
                    None => Some(self.probe(&chosen.pruned.creature)?),
                };
                (
                    Some(Winner {
                        members: chosen.members.clone(),
                        creature: chosen.pruned.creature.clone(),
                        hidden_removed: before.hidden_neurons.saturating_sub(after.hidden_neurons),
                        synapses_removed: before.synapses.saturating_sub(after.synapses),
                        latency_ms,
                    }),
                    Some(deltas[b]),
                )
            }
        };
        Ok(ArmRun {
            arm,
            bundles_scored: scored.len(),
            refused,
            wins,
            winner,
            calibration_delta,
            arm_ms: timer.started.elapsed().as_secs_f64() * 1000.0,
            scorer_calls: self.calls.get() - timer.calls,
            archive_size,
        })
    }

    /// The single-sampler arms: `bundlesPerArm` draws, each built canonically.
    fn simple_arm(
        &self,
        seed: u64,
        arm: &'static str,
        x: &[f64],
        tau: Option<f64>,
    ) -> Result<ArmRun, String> {
        let timer = self.timer();
        let mut rng = SplitMix64(seed);
        let mut built = Vec::new();
        let mut refused = 0usize;
        for j in 0..self.manifest.bundles_per_arm {
            let positions = draw_bundle(&mut rng, x, self.bundle_size(j), tau);
            match self.build(&positions) {
                Ok(pruned) => built.push(Built {
                    j,
                    members: self.members_of(&positions),
                    pruned,
                    latency: None,
                }),
                Err(e) => {
                    refused += 1;
                    self.fail("evaluate", Some(seed), Some(arm), None, e);
                }
            }
        }
        let scored: Vec<&Built> = built.iter().collect();
        self.finish_arm(timer, seed, arm, refused, &scored, None)
    }

    /// The Pareto arm: alternate samplers, keep the non-dominated archive,
    /// full-score its lowest members.
    fn pareto_arm(&self, seed: u64, prep: &Prep) -> Result<ArmRun, String> {
        let m = self.manifest;
        let timer = self.timer();
        let mut rng = SplitMix64(seed);
        let mut built: Vec<Built> = Vec::new();
        let mut points: Vec<Point> = Vec::new();
        let mut refused = 0usize;
        for j in 0..m.pareto_candidates {
            let (x, tau) = if j % 2 == 0 {
                (&prep.agg, prep.tau_agg)
            } else {
                (&prep.worst, prep.tau_worst)
            };
            let positions = draw_bundle(&mut rng, x, self.bundle_size(j), Some(tau));
            match self.build(&positions) {
                Ok(pruned) => {
                    let latency = self.probe(&pruned.creature)?;
                    let members = self.members_of(&positions);
                    points.push(Point::new(
                        predicted_cost(&self.norm, &positions),
                        pruned.after.synapses,
                        latency,
                        members.clone(),
                    ));
                    built.push(Built {
                        j,
                        members,
                        pruned,
                        latency: Some(latency),
                    });
                }
                Err(e) => {
                    refused += 1;
                    self.fail("evaluate", Some(seed), Some(PARETO), None, e);
                }
            }
        }
        let arch = pareto::archive(&points, m.archive_cap);
        let take = arch.len().min(m.bundles_per_arm);
        let mut chosen: Vec<&Built> = arch[..take].iter().map(|&c| &built[c]).collect();
        chosen.sort_by_key(|b| b.j);
        self.finish_arm(timer, seed, PARETO, refused, &chosen, Some(arch.len()))
    }

    /// All four arms for one seed, in report order.
    fn seed_arms(&self, seed: u64, prep: &Prep) -> Result<Vec<ArmRun>, String> {
        let random_x = vec![0.0; self.measured.len()];
        Ok(vec![
            self.simple_arm(seed, AGGREGATE, &prep.agg, Some(prep.tau_agg))?,
            self.simple_arm(seed, WORST, &prep.worst, Some(prep.tau_worst))?,
            self.pareto_arm(seed, prep)?,
            self.simple_arm(seed, RANDOM, &random_x, None)?,
        ])
    }

    /// Score every arm's winner once per validation stratum. Runs only after
    /// every arm of the seed has selected; its scores never feed back.
    fn validate_seed(
        &self,
        seed: u64,
        arms: &[ArmRun],
    ) -> Result<(Vec<ValidationDelta>, u64), String> {
        let calls_before = self.calls.get();
        let dir = self.work.join("validation").join(format!("seed-{seed}"));
        let winners: Vec<(String, &CreatureExport)> = arms
            .iter()
            .filter_map(|a| a.winner.as_ref().map(|w| (a.arm.to_string(), &w.creature)))
            .collect();
        self.write_dir(&dir, &winners)?;
        let mut per_arm: Vec<BTreeMap<String, f64>> = vec![BTreeMap::new(); arms.len()];
        for s in &self.val {
            let results = self.score(&dir, &s.dir, "validation", &s.name)?;
            let base = require(&results, "baseline", "validation", &s.name)?;
            for (a, arm) in arms.iter().enumerate() {
                if arm.winner.is_some() {
                    let score = require(&results, arm.arm, "validation", &s.name)?;
                    per_arm[a].insert(s.name.clone(), score - base);
                }
            }
        }
        let deltas = arms
            .iter()
            .zip(per_arm)
            .map(|(arm, mut strata)| {
                if arm.winner.is_none() {
                    // noChange: nothing removed, nothing to regress.
                    strata = self.val.iter().map(|s| (s.name.clone(), 0.0)).collect();
                }
                let weighted = self
                    .val
                    .iter()
                    .zip(&self.val_weights)
                    .map(|(s, w)| w * strata[&s.name])
                    .sum();
                ValidationDelta { strata, weighted }
            })
            .collect();
        Ok((deltas, self.calls.get() - calls_before))
    }

    /// Steps 7: every seed, every arm, then validation.
    fn run_seeds(&self, prep: &Prep) -> Result<(Vec<RunRecord>, u64), String> {
        let minority = minority_index(&self.val);
        let mut runs = Vec::new();
        let mut validation_calls = 0u64;
        for &seed in &self.manifest.seeds {
            let arms = self.seed_arms(seed, prep)?;
            let (validation, calls) = self.validate_seed(seed, &arms)?;
            validation_calls += calls;
            for (arm, val) in arms.into_iter().zip(validation) {
                runs.push(self.record(seed, arm, val, minority));
            }
        }
        Ok((runs, validation_calls))
    }

    fn record(&self, seed: u64, arm: ArmRun, val: ValidationDelta, minority: usize) -> RunRecord {
        let tol = self.manifest.regression_tolerance;
        let minority_name = self.val[minority].name.clone();
        let minority_regression = val.strata[&minority_name] < -tol;
        let regressed_strata = val.strata.values().filter(|d| **d < -tol).count();
        let (selected, hidden_removed, synapses_removed, latency_ms) = match arm.winner {
            Some(w) => (
                Some(w.members),
                w.hidden_removed,
                w.synapses_removed,
                w.latency_ms,
            ),
            None => (None, 0, 0, None),
        };
        RunRecord {
            seed,
            arm: arm.arm.to_string(),
            bundles_scored: arm.bundles_scored,
            refused: arm.refused,
            wins: arm.wins,
            selected,
            calibration_delta: arm.calibration_delta,
            validation_delta: val.weighted,
            validation_strata: val.strata,
            minority_stratum: minority_name,
            minority_regression,
            regressed_strata,
            hidden_removed,
            synapses_removed,
            latency_ms,
            arm_ms: arm.arm_ms,
            scorer_calls: arm.scorer_calls,
            wins_per_hour: None,
            archive_size: arm.archive_size,
        }
    }

    fn assemble(
        &self,
        prep: &Prep,
        mut runs: Vec<RunRecord>,
        validation_calls: u64,
        parent_latency: f64,
    ) -> StrataReport {
        let m = self.manifest;
        let arm_count = ARMS.len() as f64;
        let seed_count = m.seeds.len() as f64;
        for r in &mut runs {
            let hours = (r.arm_ms + prep.prep_ms / (arm_count * seed_count)) / MS_PER_HOUR;
            r.wins_per_hour = if hours == 0.0 {
                None
            } else {
                Some(r.wins as f64 / hours)
            };
        }
        let arms: Vec<ArmSummary> = ARMS.iter().map(|name| arm_summary(name, &runs)).collect();
        let paired = paired_rows(&runs, &m.seeds, m.fit_seed);
        let criteria = criteria_rows(&arms, m);
        StrataReport {
            schema: REPORT_SCHEMA.to_string(),
            evidence: EVIDENCE.to_string(),
            method: method(),
            versions: Versions {
                ockham_version: crate::crate_version().to_string(),
                scorer_identity: self.scorer.identity(),
                parent_checksum: self.incumbent.checksum.clone(),
                manifest_sha256: self.manifest_sha256.clone(),
                calibration: versions_of(&self.cal),
                validation: versions_of(&self.val),
            },
            manifest: m.clone(),
            fitted: Fitted {
                scales: prep.scales.clone(),
                temperatures: Temperatures {
                    aggregate: prep.tau_agg,
                    worst_stratum: prep.tau_worst,
                },
                fit_criteria: prep.fit.clone(),
            },
            coverage: prep.coverage.clone(),
            costs: Costs {
                prep_ms: prep.prep_ms,
                prep_scorer_calls: prep.prep_calls,
                latency_probe_ms: self.probe_ms.get(),
                parent_latency_ms: Some(parent_latency),
                validation_scorer_calls: validation_calls,
            },
            runs,
            arms,
            paired,
            criteria,
            failures: self.failures.borrow().clone(),
        }
    }
}

/// One fit cell: an informed arm at one candidate temperature.
struct FitCell {
    arm: &'static str,
    tau: f64,
    stems: Vec<String>,
    refused: usize,
}

fn require(
    results: &BTreeMap<String, ScoreResult>,
    stem: &str,
    stage: &str,
    stratum: &str,
) -> Result<f64, String> {
    results.get(stem).map(|r| r.score).ok_or_else(|| {
        format!("stage '{stage}' (stratum '{stratum}'): scorer output has no entry for '{stem}'")
    })
}

fn write_file(path: &Path, text: &str) -> Result<(), String> {
    std::fs::write(path, text).map_err(|e| format!("cannot write '{}': {e}", path.display()))
}

/// The validation stratum with the fewest records; ties go to name order.
fn minority_index(val: &[Stratum]) -> usize {
    (0..val.len())
        .min_by(|&a, &b| {
            val[a]
                .info
                .record_count
                .cmp(&val[b].info.record_count)
                .then_with(|| val[a].name.cmp(&val[b].name))
        })
        .unwrap_or(0)
}

fn versions_of(strata: &[Stratum]) -> Vec<StratumVersion> {
    strata
        .iter()
        .map(|s| StratumVersion {
            name: s.name.clone(),
            identity: s.info.identity.clone(),
            records: s.info.record_count,
        })
        .collect()
}

fn arm_summary(name: &str, runs: &[RunRecord]) -> ArmSummary {
    let of_arm: Vec<&RunRecord> = runs.iter().filter(|r| r.arm == name).collect();
    let dist = |f: fn(&RunRecord) -> Option<f64>| {
        let values: Vec<f64> = of_arm.iter().filter_map(|r| f(r)).collect();
        Dist::of(&values)
    };
    ArmSummary {
        name: name.to_string(),
        validation_delta: dist(|r| Some(r.validation_delta)),
        calibration_delta: dist(|r| r.calibration_delta),
        minority_regressions: of_arm.iter().filter(|r| r.minority_regression).count(),
        hidden_removed: dist(|r| Some(r.hidden_removed as f64)),
        synapses_removed: dist(|r| Some(r.synapses_removed as f64)),
        wins: dist(|r| Some(r.wins as f64)),
        wins_per_hour: dist(|r| r.wins_per_hour),
    }
}

/// A per-run metric, read as a number for paired differences.
type Metric = fn(&RunRecord) -> f64;

fn paired_rows(runs: &[RunRecord], seeds: &[u64], fit_seed: u64) -> Vec<Paired> {
    const PAIRS: [(&str, &str); 5] = [
        (WORST, AGGREGATE),
        (PARETO, AGGREGATE),
        (WORST, RANDOM),
        (PARETO, RANDOM),
        (AGGREGATE, RANDOM),
    ];
    let metrics: [(&str, Metric); 3] = [
        ("validationDelta", |r| r.validation_delta),
        ("minorityRegression", |r| {
            f64::from(u8::from(r.minority_regression))
        }),
        ("hiddenRemoved", |r| r.hidden_removed as f64),
    ];
    let mut rows = Vec::new();
    for (treatment, control) in PAIRS {
        for (metric, value) in metrics {
            let mut differences = Vec::new();
            for &seed in seeds {
                let t = runs.iter().find(|r| r.seed == seed && r.arm == treatment);
                let c = runs.iter().find(|r| r.seed == seed && r.arm == control);
                if let (Some(t), Some(c)) = (t, c) {
                    differences.push(value(t) - value(c));
                }
            }
            rows.push(Paired::new(
                treatment,
                control,
                metric,
                differences,
                fit_seed,
            ));
        }
    }
    rows
}

fn criteria_rows(arms: &[ArmSummary], m: &Manifest) -> Vec<Criterion> {
    let find = |name: &str| arms.iter().find(|a| a.name == name);
    let Some(agg) = find(AGGREGATE) else {
        return Vec::new();
    };
    [WORST, PARETO]
        .into_iter()
        .filter_map(|treatment| {
            let arm = find(treatment)?;
            Some(criterion(CriterionInput {
                arm: treatment,
                aggregate_regressions: agg.minority_regressions as u64,
                arm_regressions: arm.minority_regressions as u64,
                aggregate_fitness: agg.validation_delta.mean,
                arm_fitness: arm.validation_delta.mean,
                min_regression_reduction: m.success.min_regression_reduction,
                max_fitness_loss: m.success.max_fitness_loss,
            }))
        })
        .collect()
}

fn method() -> Method {
    Method {
        source: "arXiv:2610.27812 (abstract only; full method not retrieved — this reproduces the issue's formula)".into(),
        importance: "I[t][i] = baseline score minus the score with neuron i removed alone (core prune, step-2 mean), per calibration stratum".into(),
        scales: "s_t = median over measured neurons of |I[t][i]|; 1.0 when that median is not above 1e-12; N = I / s".into(),
        statistics: "aggregate a_i = sum_t w_t N[t][i] with calibration record weights; worst m_i = max_t N[t][i]".into(),
        sampler: "k distinct neurons drawn without replacement, P(i) proportional to exp(-(x_i - min x) / tau); the random arm draws uniformly".into(),
        temperature_fit: "per informed arm and candidate tau, bundlesPerArm bundles from fitSeed scored on calibration; highest mean calibration score wins, ties to the earliest tau".into(),
        evaluation: "per seed and arm: bundlesPerArm bundles built canonically; pareto keeps the non-dominated archive of paretoCandidates and full-scores its lowest members; the winner is the largest calibration delta above minImprovement".into(),
        validation: "after every arm of a seed has selected, the winners are scored once per validation stratum; validation never feeds back into a choice".into(),
        latency: "median of three wall-clock passes over the first latencyRecords calibration records, ms per record (indicative only)".into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn work_dir_is_a_sibling_named_after_the_report() {
        assert_eq!(
            work_dir_for(Path::new("/tmp/x/report.json")),
            PathBuf::from("/tmp/x/report.json.work")
        );
    }

    #[test]
    fn wall_clock_probe_runs_over_a_real_corpus_and_refuses_an_empty_one() {
        let corpus = tempfile::tempdir().unwrap();
        let rows: Vec<(Vec<f32>, Vec<f32>)> =
            (0..10).map(|i| (vec![i as f32, 0.5], vec![0.0])).collect();
        crate::corpus::write_bin_file(&corpus.path().join("part.bin"), &rows).unwrap();
        let creature = crate::fixtures::wide_creature(2, 3, "HARD_TANH");
        let probe = wall_clock_probe(corpus.path().to_path_buf(), 4);
        let ms = probe(&creature).expect("the probe runs over real records");
        // Only the sign and finiteness are asserted; the figure is indicative.
        assert!(ms.is_finite() && ms >= 0.0, "{ms}");

        let empty = tempfile::tempdir().unwrap();
        let empty_probe = wall_clock_probe(empty.path().to_path_buf(), 4);
        assert!(empty_probe(&creature).is_err());
    }

    #[test]
    fn minority_is_the_fewest_records_with_name_order_on_ties() {
        let stratum = |name: &str, records: u64| Stratum {
            name: name.to_string(),
            dir: PathBuf::from(name),
            info: CorpusInfo {
                identity: String::new(),
                record_count: records,
                file_count: 1,
                input_count: 1,
                output_count: 1,
            },
        };
        let strata = [
            stratum("south", 100),
            stratum("north", 800),
            stratum("east", 100),
        ];
        // Tie on 100 between south and east: name order picks east.
        assert_eq!(minority_index(&strata), 2);
    }
}
