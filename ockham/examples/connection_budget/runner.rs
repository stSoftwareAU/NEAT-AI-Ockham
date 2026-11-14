use std::cell::RefCell;
use std::collections::BTreeMap;
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::Path;
use std::time::Instant;

use neat_ai_ockham::clock::{Clock, SystemClock};
use neat_ai_ockham::corpus::corpus_info;
use neat_ai_ockham::incumbent::{load_incumbent, sha256_hex};
use neat_ai_ockham::promote::{FullConfig, evaluate_full};
use neat_ai_ockham::prune::prune_edge_by_key;
use neat_ai_ockham::scorer::{DirectoryScorer, ScoreResult, ScorerMode};
use neat_ai_ockham::stats::{ActivationStats, SampleSpec, compute_activation_stats, source_value};
use neat_ai_ockham::sweep::{CandidateKind, ScreenConfig, Sweep, SweepCandidate, screen_batch};
use neat_core::{CreatureExport, creature_to_json};
use neat_core::training_data::TrainingDataConfig;
use serde::{Deserialize, Serialize};

use super::allocation::{Arm, Constraints, Plan, output_support, ranked_edges};
use super::manifest::{Frozen, Spec, Subject, file_sha256, read_json, write_new};
use super::measurement::{Diagnostics, Metered, Usage, diagnostics, external};

#[path = "summary.rs"]
mod summary;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Scores {
    pub parent: ScoreResult,
    pub proposal: ScoreResult,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RunRecord {
    pub format_version: u32,
    pub manifest_sha256: String,
    pub subject: String,
    pub independent_unit: String,
    pub seed: u64,
    pub arm: Arm,
    pub requested_connections: usize,
    pub opening_connections: usize,
    pub achieved_connections: usize,
    pub accepted_connections: usize,
    pub target_reached: bool,
    pub stop_reason: String,
    pub error: Option<String>,
    pub search: Usage,
    pub evaluation: Usage,
    pub quotas: BTreeMap<String, usize>,
    pub region_connections: BTreeMap<String, usize>,
    pub output_support: BTreeMap<String, std::collections::BTreeSet<String>>,
    pub full: Option<Scores>,
    pub held_out: Option<Scores>,
    pub diagnostics: Option<Diagnostics>,
    pub directory: String,
}

fn write_creature(path: &Path, creature: &CreatureExport) -> Result<(), String> {
    let json = creature_to_json(creature).map_err(|e| e.to_string())?;
    let mut file = OpenOptions::new().write(true).create_new(true).open(path).map_err(|e| e.to_string())?;
    file.write_all(json.as_bytes()).map_err(|e| e.to_string())
}

fn calibrate(creature: &CreatureExport, subject: &Subject, meter: &Metered<'_>) -> Result<ActivationStats, String> {
    let start = Instant::now();
    let corpus = corpus_info(&subject.calibration, &TrainingDataConfig::new(creature.input, creature.output))?;
    meter.statistics(corpus.record_count)?;
    let json = creature_to_json(creature).map_err(|e| e.to_string())?;
    let result = compute_activation_stats(creature, &sha256_hex(json.as_bytes()), &subject.calibration, &corpus, 4096, &SampleSpec::full());
    meter.usage.borrow_mut().preparation_ms += start.elapsed().as_secs_f64() * 1000.0;
    result
}

// Use the existing screen and authoritative promotion, with no bundle or acceptance rule added.
fn judge(parent: &CreatureExport, stats: &ActivationStats, candidate: SweepCandidate, subject: &Subject,
    spec: &Spec, meter: &Metered<'_>, dir: &Path, force_measurement: bool) -> Result<Option<CreatureExport>, String> {
    let screen_dir = dir.join("screen");
    let screened = screen_batch(meter.clock, meter, &subject.calibration, parent, vec![candidate], ScreenConfig {
        sample_rate: spec.screen_rate, sample_phase: 0,
        // Budget proposals are all measured; a calibration loss is recorded, not used to hide a full result.
        threshold: if force_measurement { -f64::MAX } else { 0.0 },
        remaining_after: 0, dir: &screen_dir,
    })?;
    if screened.winners.is_empty() { return Ok(None); }
    let full_dir = dir.join("full");
    let mut cfg = FullConfig::new(spec.min_improvement, &full_dir, None);
    cfg.max_entries = Some(1);
    cfg.max_individuals = Some(1);
    let outcome = evaluate_full(meter.clock, meter, &subject.training, parent, stats, &screened.winners, cfg)?;
    Ok(outcome.winner.map(|w| w.creature))
}

#[allow(clippy::too_many_arguments)]
pub fn search_one(spec: &Spec, subject: &Subject, parent: &CreatureExport, arm: Arm, target: usize, seed: u64,
    scorer: &dyn DirectoryScorer, clock: &dyn Clock, dir: &Path) -> Result<RunRecord, String> {
    std::fs::create_dir(dir).map_err(|e| e.to_string())?;
    let plan = Plan::new(parent, arm, target)?;
    let constraints = Constraints::new(parent)?;
    let usage = RefCell::new(Usage::default());
    let started = clock.now();
    let meter = Metered { inner: scorer, limits: &spec.limits, usage: &usage, clock, started, widths: (parent.input, parent.output) };
    let mut proposal = parent.clone();
    let mut accepted = parent.clone();
    let mut stop = "target-reached".to_string();
    let mut step = 0;
    let result = (|| -> Result<(), String> {
        let mut stats = calibrate(&proposal, subject, &meter)?;
        if arm == Arm::Ockham {
            let mut sweep = Sweep::new(&proposal, seed);
            while proposal.synapses.len() > target {
                if meter.expired() { stop = "search-timeout".into(); break; }
                let left = spec.limits.proposal_attempts.saturating_sub(usage.borrow().proposal_attempts as usize);
                if left == 0 { stop = "proposal-limit".into(); break; }
                sweep.order.truncate(sweep.next.saturating_add(left));
                let before = sweep.next;
                let (candidates, skipped) = sweep.fill_batch(&proposal, &stats, 1);
                {
                    let mut usage = usage.borrow_mut();
                    usage.proposal_attempts += (sweep.next - before) as u64;
                    usage.visited_keys.extend(sweep.order[before..sweep.next].iter().cloned());
                    for skip in skipped { usage.refuse(&skip.reason); }
                }
                let Some(candidate) = candidates.into_iter().next() else { stop = "sweep-exhausted".into(); break; };
                if let Err(reason) = plan.check(&proposal, &candidate.creature).and_then(|()| constraints.check(&candidate.creature)) {
                    usage.borrow_mut().refuse(&reason); continue;
                }
                let step_dir = dir.join(format!("step-{step}"));
                step += 1;
                std::fs::create_dir(&step_dir).map_err(|e| e.to_string())?;
                if let Some(winner) = judge(&proposal, &stats, candidate, subject, spec, &meter, &step_dir, false)? {
                    proposal = winner;
                    accepted = proposal.clone();
                    if proposal.synapses.len() > target {
                        stats = calibrate(&proposal, subject, &meter)?;
                        sweep = Sweep::new(&proposal, seed.wrapping_add(step));
                    }
                }
            }
        } else {
            while proposal.synapses.len() > target {
                if meter.expired() { stop = "search-timeout".into(); break; }
                let ranked = ranked_edges(&proposal, &stats, seed)?;
                let mut chosen = None;
                for edge in ranked {
                    if usage.borrow().proposal_attempts >= spec.limits.proposal_attempts as u64 { stop = "proposal-limit".into(); break; }
                    if meter.expired() { stop = "search-timeout".into(); break; }
                    {
                        let mut usage = usage.borrow_mut();
                        usage.proposal_attempts += 1;
                        usage.visited_keys.insert(format!("{}→{}:{:?}", edge.key.from_uuid, edge.key.to_uuid, edge.key.role));
                    }
                    if !plan.eligible(&proposal, &edge.key) { usage.borrow_mut().refuse("region at quota"); continue; }
                    let source = source_value(&proposal, &stats, &edge.key.from_uuid).map(|s| s.value);
                    let source_stats = stats.by_uuid(&edge.key.from_uuid).or_else(|| stats.input_by_uuid(&edge.key.from_uuid));
                    let cut = match prune_edge_by_key(&proposal, &edge.key, source, source_stats) {
                        Ok(cut) => cut,
                        Err(reason) => { usage.borrow_mut().refuse(&format!("core: {reason}")); continue; }
                    };
                    if let Err(reason) = plan.check(&proposal, &cut.creature).and_then(|()| constraints.check(&cut.creature)) {
                        usage.borrow_mut().refuse(&reason); continue;
                    }
                    write_new(&dir.join(format!("cut-{step}.json")), &serde_json::json!({
                        "from": edge.key.from_uuid, "to": edge.key.to_uuid, "role": format!("{:?}", edge.key.role),
                        "importance": edge.importance, "beforeConnections": proposal.synapses.len(),
                        "afterConnections": cut.creature.synapses.len(), "core": cut.detail,
                    }))?;
                    chosen = Some(cut.creature);
                    step += 1;
                    break;
                }
                let Some(next) = chosen else {
                    if stop == "target-reached" { stop = "infeasible-target".into(); }
                    break;
                };
                proposal = next;
                if proposal.synapses.len() > target { stats = calibrate(&proposal, subject, &meter)?; }
            }
            if proposal != *parent && !meter.expired() {
                let candidate = SweepCandidate {
                    uuid: "budget-proposal".into(), members: vec!["budget-proposal".into()], permutation_index: 0,
                    kind: CandidateKind::Group, merged_with: None, from_uuid: None, to_uuid: None, weight: None,
                    prune: None, stem: "proposal".into(), creature: proposal.clone(),
                };
                if let Some(winner) = judge(parent, &stats, candidate, subject, spec, &meter, dir, true)? { accepted = winner; }
            }
        }
        Ok(())
    })();
    let mut usage = usage.into_inner();
    usage.search_ms = clock.since(started).as_secs_f64() * 1000.0;
    let error = match result {
        Ok(()) => None,
        Err(reason) if usage.refused_calls > 0 || reason == "statistics-record-limit" => { stop = "compute-limit".into(); None },
        Err(reason) => { stop = "failed".into(); Some(format!("connection-budget: {reason}")) },
    };
    if stop == "search-timeout" {
        usage.timeout_exits += 1;
    }
    write_creature(&dir.join("proposal.json"), &proposal)?;
    write_creature(&dir.join("best.json"), &accepted)?;
    Ok(RunRecord {
        format_version: 1, manifest_sha256: String::new(), subject: subject.id.clone(), independent_unit: subject.independent_unit.clone(),
        seed, arm, requested_connections: target, opening_connections: parent.synapses.len(),
        achieved_connections: proposal.synapses.len(), accepted_connections: accepted.synapses.len(),
        target_reached: proposal.synapses.len() == target, stop_reason: stop, error, search: usage, evaluation: Usage::default(),
        quotas: plan.quotas.clone(), region_connections: plan.counts(&proposal)?, output_support: output_support(&proposal)?,
        full: None, held_out: None, diagnostics: None, directory: dir.file_name().unwrap().to_string_lossy().into_owned(),
    })
}

fn append(file: &mut File, record: &RunRecord) -> Result<(), String> {
    let mut json = serde_json::to_vec(record).map_err(|e| e.to_string())?;
    json.push(b'\n');
    file.write_all(&json).map_err(|e| e.to_string())?;
    file.sync_data().map_err(|e| e.to_string())
}

pub fn run(manifest_path: &Path, out: &Path) -> Result<(), String> {
    let frozen: Frozen = read_json(manifest_path)?;
    frozen.verify()?;
    let scorer = external(&frozen.spec)?;
    execute(&frozen, manifest_path, out, &scorer)
}

pub fn execute(frozen: &Frozen, manifest_path: &Path, out: &Path, scorer: &dyn DirectoryScorer) -> Result<(), String> {
    let spec = &frozen.spec;
    std::fs::create_dir(out).map_err(|e| format!("new output directory {}: {e}", out.display()))?;
    write_new(&out.join("manifest.json"), frozen)?;
    let manifest_sha256 = file_sha256(manifest_path)?;
    let mut search_log = OpenOptions::new().write(true).create_new(true).open(out.join("search.jsonl")).map_err(|e| e.to_string())?;
    let mut records = Vec::new();
    for subject in &spec.creatures {
        let parent = load_incumbent(&subject.creature).map_err(|e| e.to_string())?;
        for target in &spec.connection_targets {
            for seed in &spec.seeds {
                // Rotate execution order across seeds; allocations and ties still use the declared seed.
                for offset in 0..4 {
                    let arm = Arm::ALL[(*seed as usize % 4 + offset) % 4];
                    let dir = out.join(format!("r{:05}", records.len()));
                    let mut record = search_one(spec, subject, &parent.creature, arm, *target, *seed, scorer, &SystemClock, &dir)?;
                    record.manifest_sha256 = manifest_sha256.clone();
                    append(&mut search_log, &record)?;
                    records.push(record);
                }
            }
        }
    }
    let mut results = OpenOptions::new().write(true).create_new(true).open(out.join("results.jsonl")).map_err(|e| e.to_string())?;
    // Search is over for every arm before any validation score can be observed.
    for record in &mut records {
        let subject = spec.creatures.iter().find(|s| s.id == record.subject).unwrap();
        let parent = load_incumbent(&subject.creature).map_err(|e| e.to_string())?;
        let dir = out.join(&record.directory);
        let proposal = load_incumbent(&dir.join("proposal.json")).map_err(|e| e.to_string())?;
        let cohort = dir.join("evaluation");
        std::fs::create_dir(&cohort).map_err(|e| e.to_string())?;
        write_creature(&cohort.join("baseline.json"), &parent.creature)?;
        write_creature(&cohort.join("proposal.json"), &proposal.creature)?;
        let mut limits = spec.limits.clone();
        limits.scorer_calls = 2;
        limits.scorer_record_evaluations = 2 * (frozen.inputs[spec.creatures.iter().position(|s| s.id == subject.id).unwrap()].training.records
            + frozen.inputs[spec.creatures.iter().position(|s| s.id == subject.id).unwrap()].validation.records);
        limits.search_seconds = 2 * (limits.call_timeout_seconds + 2);
        let usage = RefCell::new(Usage::default());
        let meter = Metered { inner: scorer, limits: &limits, usage: &usage, clock: &SystemClock, started: Instant::now(), widths: (parent.creature.input, parent.creature.output) };
        let evaluated = (|| -> Result<(), String> {
            for (corpus, dest) in [(&subject.training, &mut record.full), (&subject.validation, &mut record.held_out)] {
                let scores = meter.score_directory(&cohort, corpus, ScorerMode::Full).map_err(|e| e.to_string())?;
                *dest = Some(Scores { parent: scores["baseline"].clone(), proposal: scores["proposal"].clone() });
            }
            record.diagnostics = Some(diagnostics(&parent.creature, &proposal.creature, &subject.validation, &spec.latency)?);
            Ok(())
        })();
        if let Err(error) = evaluated {
            record.error = Some(format!("connection-budget evaluation: {error}"));
        }
        record.evaluation = usage.into_inner();
        append(&mut results, record)?;
    }
    frozen.verify()?;
    write_new(&out.join("summary.json"), &summary::summarise(spec, &records))?;
    if records.iter().any(|r| r.error.is_some() || r.search.timeout_exits > 0 || r.evaluation.failed_calls > 0) {
        return Err("one or more trials failed or timed out; inspect results.jsonl and summary.json".into());
    }
    Ok(())
}
