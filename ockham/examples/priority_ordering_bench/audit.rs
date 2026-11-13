use std::collections::BTreeMap;
use std::io::Read;
use std::path::Path;
use std::time::Instant;

use neat_ai_ockham::features::{self, CandidateFeatures, PriorEvidence};
use neat_ai_ockham::incumbent::{load_incumbent, sha256_hex};
use neat_ai_ockham::scorer::{DirectoryScorer, ScorerMode};
use neat_ai_ockham::stats::{SampleSpec, compute_activation_stats};
use neat_ai_ockham::sweep::{Sweep, present_visits};
use neat_ai_ockham::telemetry::{CandidateOutcome, CandidateRecord, RunStamp};
use neat_core::training_data::TrainingDataConfig;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::manifest::{Manifest, SAMPLE_RATE, Split, Task, seeded_order};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct AuditRow {
    pub visit: String,
    pub kind: Option<String>,
    pub candidate_sha256: Option<String>,
    pub features: Option<CandidateFeatures>,
    pub sample_delta: Option<f64>,
    pub full_delta: Option<f64>,
    pub sample_ms: u64,
    pub full_ms: u64,
    pub telemetry: Option<CandidateRecord>,
    pub failure: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct Pool {
    pub task: String,
    pub seed: u64,
    pub population: usize,
    pub selected: Vec<String>,
    pub rows: Vec<AuditRow>,
    pub preparation_ms: u64,
    pub features_ms: u64,
    pub generation_ms: u64,
    pub elapsed_ms: u64,
}

impl Pool {
    pub fn complete(&self) -> bool {
        self.rows.iter().filter(|r| r.kind.is_some()).all(|r| r.sample_delta.is_some() && r.full_delta.is_some() && r.failure.is_none())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct Audit {
    pub format_version: u32,
    pub manifest_sha256: String,
    pub freeze_sha256: Option<String>,
    pub phase: String,
    pub pools: Vec<Pool>,
    pub failures: Vec<String>,
    pub elapsed_ms: u64,
}

pub fn ms(start: Instant) -> u64 { start.elapsed().as_millis().try_into().unwrap_or(u64::MAX) }

// Hash every corpus byte; the ordinary fast head/tail cache identity is not a
// sufficient content boundary for this experiment's split checks.
pub fn corpus_digest(dir: &Path, max_bytes: u64) -> Result<String, String> {
    let mut files = neat_core::training_data::find_bin_files(dir).map_err(|e| e.to_string())?;
    files.sort();
    if files.is_empty() { return Err("no corpus .bin files".into()); }
    let mut hash = Sha256::new(); let mut total = 0u64;
    let mut buffer = [0u8; 65536];
    for path in files {
        let mut file = std::fs::File::open(path).map_err(|e| e.to_string())?;
        loop {
            let n = file.read(&mut buffer).map_err(|e| e.to_string())?;
            if n == 0 { break; }
            total = total.checked_add(n as u64).ok_or("corpus size overflow")?;
            if total > max_bytes { return Err("corpus exceeds maxCorpusBytes".into()); }
            hash.update(&buffer[..n]);
        }
    }
    Ok(format!("{:x}", hash.finalize()))
}

fn check_features(f: &CandidateFeatures) -> Result<(), String> {
    if [f.variance, f.mean_abs, f.range, f.outgoing_weight, f.direct_growth_units,
        f.cascade_growth_units, f.depth_fraction, f.downstream_sensitivity()]
        .iter().any(|x| !x.is_finite())
    { return Err("non-finite raw candidate feature".into()); }
    Ok(())
}

fn score_delta(scorer: &impl DirectoryScorer, dir: &Path, task: &Task, mode: ScorerMode, records: u64) -> Result<f64, String> {
    let result = scorer.score_directory(dir, &task.corpus, mode).map_err(|e| e.to_string())?;
    let baseline = result.get("baseline").ok_or("missing baseline score")?;
    let candidate = result.get("candidate").ok_or("missing candidate score")?;
    for value in [baseline, candidate] {
        if !value.score.is_finite() || !value.error.is_finite() { return Err("non-finite scorer result".into()); }
        if value.record_count == 0 || (mode.is_authoritative() && value.record_count != records) || value.record_count > records {
            return Err("incomplete scoring: record count differs from requested corpus".into());
        }
    }
    if baseline.record_count != candidate.record_count { return Err("incomplete scoring: baseline/candidate counts differ".into()); }
    let delta = candidate.score-baseline.score;
    if !delta.is_finite() { return Err("non-finite score delta".into()); }
    Ok(delta)
}

pub fn collect_pool(m: &Manifest, task: &Task, seed: u64, visits: usize, scorer: &impl DirectoryScorer, dir: &Path) -> Result<Pool, String> {
    let start = Instant::now();
    let parent = load_incumbent(&task.parent).map_err(|e| e.to_string())?;
    if parent.checksum != task.parent_sha256 { return Err("parent digest mismatch".into()); }
    if corpus_digest(&task.corpus,m.max_corpus_bytes)? != task.corpus_sha256 { return Err("corpus digest mismatch".into()); }
    let config = TrainingDataConfig::new(parent.creature.input, parent.creature.output);
    let corpus = neat_ai_ockham::corpus::corpus_info(&task.corpus,&config)?;
    // Select from the unfiltered visit population before measuring or proposing.
    let population: Vec<_> = present_visits(&parent.creature).into_iter().collect();
    let mut selected = seeded_order(&population,seed); selected.truncate(visits);
    let stats = compute_activation_stats(&parent.creature,&parent.checksum,&task.corpus,&corpus,1024,&SampleSpec::with_max_records(m.stats_records))?;
    let preparation_ms = ms(start);
    let feature_start = Instant::now();
    // No learnings cache is read: temporal evidence cannot leak through priors.
    let features = features::extract(&parent.creature,&stats,&PriorEvidence::new());
    for f in features.values() { check_features(f)?; }
    let features_ms = ms(feature_start);
    let generation_start = Instant::now();
    let mut sweep = Sweep::new(&parent.creature,seed);
    sweep.order.clone_from(&selected);
    let (candidates,skips) = sweep.fill_batch(&parent.creature,&stats,selected.len());
    let generation_ms = ms(generation_start);
    std::fs::create_dir(dir).map_err(|e| e.to_string())?;
    let mut rows = BTreeMap::new();
    for skip in skips {
        rows.insert(skip.uuid.clone(), AuditRow { visit:skip.uuid,kind:None,candidate_sha256:None,features:None,
            sample_delta:None,full_delta:None,sample_ms:0,full_ms:0,telemetry:None,failure:Some(format!("generation: {}",skip.reason)) });
    }
    let stamp = RunStamp { host:"ordering-experiment".into(), corpus_identity:task.corpus_sha256.clone(),
        creature_checksum:parent.checksum.clone(), ordering:"random-audit".into(), seed };
    for (i,candidate) in candidates.into_iter().enumerate() {
        let kind = serde_json::to_value(candidate.kind).map_err(|e| e.to_string())?.as_str().ok_or("candidate kind is not a string")?.to_string();
        let text = neat_core::creature_to_json(&candidate.creature).map_err(|e| e.to_string())?;
        let mut row = AuditRow { visit:candidate.uuid.clone(), kind:Some(kind.clone()), candidate_sha256:Some(sha256_hex(text.as_bytes())),
            features:features.get(&candidate.uuid).copied(),sample_delta:None,full_delta:None,sample_ms:0,full_ms:0,telemetry:None,failure:None };
        if ms(start) >= m.max_task_ms {
            row.failure = Some("compute limit: maxTaskMs exhausted before scoring".into());
        } else {
            let cohort = dir.join(format!("candidate-{i}")); std::fs::create_dir(&cohort).map_err(|e| e.to_string())?;
            std::fs::write(cohort.join("baseline.json"), &parent.text).map_err(|e| e.to_string())?;
            std::fs::write(cohort.join("candidate.json"), text).map_err(|e| e.to_string())?;
            let tick = Instant::now();
            match score_delta(scorer,&cohort,task,ScorerMode::Sample {rate:SAMPLE_RATE,phase:seed},corpus.record_count) {
                Ok(value) => row.sample_delta=Some(value), Err(error) => row.failure=Some(format!("partial scorer: {error}")),
            }
            row.sample_ms=ms(tick);
            let tick = Instant::now();
            // Even sampled losers and partial scorer failures get an individual
            // full score. No ranking or sampled outcome gates this call.
            if ms(start) < m.max_task_ms {
                match score_delta(scorer,&cohort,task,ScorerMode::Full,corpus.record_count) {
                    Ok(value) => row.full_delta=Some(value),
                    Err(error) => row.failure=Some(format!("{}full scorer: {error}",row.failure.map_or(String::new(),|s|format!("{s}; ")))),
                }
            } else { row.failure=Some(format!("{}compute limit: full score not run",row.failure.map_or(String::new(),|s|format!("{s}; ")))); }
            row.full_ms=ms(tick);
        }
        if let Some(f) = &row.features {
            let mut record = CandidateRecord::new(&stamp,&candidate.uuid,&kind,f,if row.full_delta.is_some(){CandidateOutcome::Rejected}else{CandidateOutcome::ScreenedOut});
            record.unix_secs=task.observed_unix; record.sample_delta=row.sample_delta; record.full_delta=row.full_delta;
            record.scorer_ms=row.sample_ms+row.full_ms; record.prune=candidate.prune; record.merged_with=candidate.merged_with;
            row.telemetry=Some(record);
        }
        rows.insert(candidate.uuid,row);
    }
    let rows = selected.iter().map(|id| rows.remove(id).ok_or_else(|| format!("selected visit {id} has no audit outcome"))).collect::<Result<Vec<_>,_>>()?;
    Ok(Pool {task:task.id.clone(),seed,population:population.len(),selected,rows,preparation_ms,features_ms,generation_ms,elapsed_ms:ms(start)})
}

pub fn collect(m: &Manifest, pilot: bool, visits: usize, freeze_sha256: Option<String>, scorer: &impl DirectoryScorer, dir: &Path) -> Audit {
    let start=Instant::now();
    let mut audit=Audit {format_version:1,manifest_sha256:m.identity().expect("serialisable manifest"),freeze_sha256,
        phase:if pilot {"pilot"} else {"audit"}.into(),pools:Vec::new(),failures:Vec::new(),elapsed_ms:0};
    for (i,task) in m.tasks.iter().enumerate() {
        if pilot && task.split != Split::Calibration { continue; }
        for seed in &m.seeds {
            if ms(start) >= m.max_run_ms { audit.failures.push(format!("{} seed {seed}: maxRunMs exhausted",task.id)); continue; }
            match collect_pool(m,task,*seed,visits,scorer,&dir.join(format!("task-{i}-seed-{seed}"))) {
                Ok(pool) => { if !pool.complete() { audit.failures.push(format!("{} seed {seed}: incomplete scoring",task.id)); } audit.pools.push(pool); }
                Err(error) => audit.failures.push(format!("{} seed {seed}: {error}",task.id)),
            }
        }
    }
    audit.elapsed_ms=ms(start);
    audit
}
