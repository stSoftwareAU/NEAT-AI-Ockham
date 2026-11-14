use std::collections::{BTreeMap, BTreeSet};
use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use neat_core::training_data::{TrainingDataConfig, find_bin_files};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::corpus::{CorpusInfo, RecordRange, corpus_info};
use crate::incumbent::{load_incumbent, sha256_hex};
use crate::stats::SampleSpec;

pub(super) const METHOD: &str = "neat-soft-gate-fd-v1";
pub(super) const REFERENCE: &str = "3cc29a2b32045c0ee697b673e2c3be8f92b70e57";
pub(super) const TARGETS: [f64; 4] = [1.0, 0.9, 0.75, 0.5];

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(super) struct Input {
    pub path: PathBuf,
    pub sha256: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(super) struct Fit {
    pub steps: usize,
    pub learning_rate: f64,
    pub temperature: f64,
    pub difference_step: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(super) struct Limits {
    pub record_evaluations: u64,
    pub compilations: u64,
    pub rewrites: u64,
    pub io_bytes: u64,
    pub seconds: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(super) struct Criteria {
    pub min_paired_score_gain: f64,
    pub max_validation_error_increase: f64,
    pub min_latency_reduction: f64,
    pub max_capacity_overshoot: f64,
    pub max_cost_imbalance: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(super) struct Manifest {
    pub format_version: u32,
    pub method: String,
    pub core_commit: String,
    pub reference_commit: String,
    pub paper_version: String,
    pub reference_source_sha256: BTreeMap<String, String>,
    pub capacity_metric: String,
    pub targets: Vec<f64>,
    pub creature: Input,
    pub corpus: Input,
    pub scorer: Input,
    pub executable_sha256: String,
    pub hardware: String,
    pub scorer_environment: BTreeMap<String, Option<String>>,
    pub representative: bool,
    pub data_provenance: String,
    pub seeds: Vec<u64>,
    pub calibration: Vec<RecordRange>,
    pub validation: Vec<RecordRange>,
    pub fit: Fit,
    pub limits_per_family: Limits,
    pub independent_trials_per_budget: usize,
    pub latency_records: usize,
    pub latency_repetitions: usize,
    pub criteria: Criteria,
    pub preparation_seconds: f64,
    pub preparation_bytes: u64,
}

fn reference_sources() -> BTreeMap<String, String> {
    [
        ("src/models/prunable/refine/core.py", "497ea32b356f85e1a5bfcc1804fff8b6fa20ccfc53fec17a683ef4e7e88e41d3"),
        ("src/models/prunable/refine/masking.py", "24ba6d00ec5cd4f178e5067fd66a69d3a61f6ab59f76f806ff3f2928eaafca7d"),
        ("src/models/prunable/refine/budget.py", "1be8638223a105b41406ab2f6f76f2974f71f5df91c3634f570642c709312f49"),
    ].into_iter().map(|(path, sha)| (path.into(), sha.into())).collect()
}

// Embedded at build time, so a family-pin refresh cannot leave a manifest
// claiming the previously compiled core revision.
fn core_commit() -> Result<&'static str, String> {
    let package = include_str!("../../../Cargo.lock").split("[[package]]")
        .find(|package| package.lines().any(|line| line == "name = \"neat-core\""))
        .ok_or("build lockfile has no neat-core package")?;
    let source = package.lines().find(|line| line.starts_with("source = "))
        .ok_or("build lockfile has no neat-core source")?;
    let commit = source.rsplit('#').next().ok_or("neat-core source has no revision")?.trim_end_matches('"');
    if commit.len() != 40 || !commit.bytes().all(|c| c.is_ascii_hexdigit()) {
        return Err("neat-core source is not a pinned git revision".into());
    }
    Ok(commit)
}

pub(super) fn scorer_environment() -> BTreeMap<String, Option<String>> {
    ["NEAT_SCORER_ACTIVATION_THREADS", "NEAT_SCORER_FILE_THREADS", "NEAT_SCORER_GPU_SCRATCH_BYTES",
        "NEAT_SCORER_READ_BYTES", "NEAT_SCORER_WORKER_SPLIT"]
        .into_iter().map(|name| (name.into(), std::env::var(name).ok())).collect()
}

pub(super) fn hardware() -> String {
    let cpu = std::fs::read_to_string("/proc/cpuinfo")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|line| line.starts_with("model name") || line.starts_with("CPU implementer"))
                .map(str::to_owned)
        })
        .unwrap_or_else(|| "CPU model unrecorded".into());
    format!(
        "{}-{}; logical CPUs={}; {cpu}",
        std::env::consts::ARCH,
        std::env::consts::OS,
        std::thread::available_parallelism().map_or(1, usize::from),
    )
}

pub(super) fn file_digest(path: &Path) -> Result<(String, u64), String> {
    let mut file = File::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
    let mut hash = Sha256::new();
    let mut bytes = 0;
    let mut buf = [0; 65536];
    loop {
        let n = file.read(&mut buf).map_err(|e| e.to_string())?;
        if n == 0 { break; }
        hash.update(&buf[..n]);
        bytes += n as u64;
    }
    Ok((format!("{:x}", hash.finalize()), bytes))
}

// Include complete file contents: corpus_info's fast identity is a cache key,
// not the content checksum a predeclared research comparison requires.
pub(super) fn corpus_digest(path: &Path) -> Result<(String, u64), String> {
    let mut files = find_bin_files(path).map_err(|e| e.to_string())?;
    files.sort();
    let mut hash = Sha256::new();
    let mut bytes = 0;
    for file in files {
        let name = file.file_name().ok_or("corpus file has no name")?.as_encoded_bytes();
        let (digest, len) = file_digest(&file)?;
        hash.update((name.len() as u64).to_le_bytes());
        hash.update(name);
        hash.update(len.to_le_bytes());
        hash.update(digest.as_bytes());
        bytes += len;
    }
    Ok((format!("{:x}", hash.finalize()), bytes))
}

fn input(path: &Path) -> Result<(Input, u64), String> {
    let path = path.canonicalize().map_err(|e| e.to_string())?;
    let (sha256, bytes) = file_digest(&path)?;
    Ok((Input { path, sha256 }, bytes))
}

/// Write a new, editable predeclaration. Review and hash it before running.
/// This performs no fitting, pruning or validation-set evaluation.
///
/// # Errors
/// Refuses invalid inputs, split sizes and an existing destination.
pub fn prepare_manifest(
    creature: &Path,
    corpus: &Path,
    scorer: &Path,
    out: &Path,
    calibration_records: u64,
    validation_records: u64,
) -> Result<String, String> {
    let started = Instant::now();
    let parent = load_incumbent(creature).map_err(|e| e.to_string())?;
    super::fit::supported(&parent.creature)?;
    let config = TrainingDataConfig::new(parent.creature.input, parent.creature.output);
    let info = corpus_info(corpus, &config)?;
    if calibration_records == 0 || validation_records == 0
        || calibration_records.checked_add(validation_records).is_none_or(|n| n > info.record_count)
    {
        return Err("calibration and validation sizes must be positive and fit the corpus".into());
    }
    let (creature, creature_bytes) = input(creature)?;
    let (scorer, scorer_bytes) = input(scorer)?;
    let (executable_sha256, executable_bytes) = file_digest(&std::env::current_exe().map_err(|e| e.to_string())?)?;
    let (sha256, corpus_bytes) = corpus_digest(corpus)?;
    let training_info = CorpusInfo { record_count: info.record_count - validation_records, ..info.clone() };
    let calibration = SampleSpec::with_max_records(calibration_records).plan(&training_info);
    let manifest = Manifest {
        format_version: 1,
        method: METHOD.into(),
        core_commit: core_commit()?.into(),
        reference_commit: REFERENCE.into(),
        paper_version: "2610.20588v1".into(),
        reference_source_sha256: reference_sources(),
        capacity_metric: "retained-original-hidden-neurons".into(),
        targets: TARGETS.to_vec(),
        creature,
        corpus: Input { path: corpus.canonicalize().map_err(|e| e.to_string())?, sha256 },
        scorer,
        executable_sha256,
        hardware: hardware(),
        scorer_environment: scorer_environment(),
        representative: false,
        data_provenance: "Unclassified input: smoke study until provenance is reviewed".into(),
        seeds: vec![7, 19, 31, 43, 59, 71, 89, 101, 127, 149],
        calibration,
        validation: vec![RecordRange { start: info.record_count - validation_records, len: validation_records }],
        fit: Fit { steps: 300, learning_rate: 0.1, temperature: 0.2, difference_step: 0.001 },
        limits_per_family: Limits {
            record_evaluations: 100_000_000,
            compilations: 1_000_000,
            rewrites: 100_000,
            io_bytes: 100_000_000_000,
            seconds: 900.0,
        },
        independent_trials_per_budget: 100,
        latency_records: 32,
        latency_repetitions: 31,
        criteria: Criteria {
            min_paired_score_gain: 0.001,
            max_validation_error_increase: 0.01,
            min_latency_reduction: 0.05,
            max_capacity_overshoot: 0.025,
            max_cost_imbalance: 0.1,
        },
        preparation_seconds: started.elapsed().as_secs_f64(),
        preparation_bytes: creature_bytes + scorer_bytes + executable_bytes + corpus_bytes,
    };
    manifest.validate(&info)?;
    let bytes = serde_json::to_vec_pretty(&manifest).map_err(|e| e.to_string())?;
    write_new(out, &bytes)?;
    Ok(sha256_hex(&bytes))
}

pub(super) fn write_new(path: &Path, bytes: &[u8]) -> Result<(), String> {
    let mut file = OpenOptions::new().write(true).create_new(true).open(path)
        .map_err(|e| format!("create {}: {e}", path.display()))?;
    file.write_all(bytes).map_err(|e| e.to_string())
}

impl Manifest {
    pub fn validate(&self, info: &CorpusInfo) -> Result<(), String> {
        if self.format_version != 1 || self.method != METHOD || self.core_commit != core_commit()?
            || self.reference_commit != REFERENCE || self.targets != TARGETS
            || self.paper_version != "2610.20588v1" || self.reference_source_sha256 != reference_sources()
            || self.capacity_metric != "retained-original-hidden-neurons"
        {
            return Err("unsupported manifest version, method, core, reference or targets".into());
        }
        if self.seeds.is_empty() || self.seeds.len() > 100
            || self.seeds.iter().collect::<BTreeSet<_>>().len() != self.seeds.len()
        { return Err("seeds must contain 1..100 distinct paired seeds".into()); }
        for input in [&self.creature, &self.corpus, &self.scorer] {
            if !input.path.is_absolute() || !digest_valid(&input.sha256) {
                return Err("input paths must be absolute with SHA-256 checksums".into());
            }
        }
        if !digest_valid(&self.executable_sha256) || self.hardware != hardware()
            || self.scorer_environment != scorer_environment()
            || self.data_provenance.trim().is_empty() || self.data_provenance.len() > 4096
        { return Err("invalid executable checksum, hardware or data provenance".into()); }
        validate_ranges(&self.calibration, info.record_count, "calibration")?;
        validate_ranges(&self.validation, info.record_count, "validation")?;
        for a in &self.calibration {
            for b in &self.validation {
                if a.start < b.end() && b.start < a.end() {
                    return Err("calibration and validation overlap".into());
                }
            }
        }
        let calibration: u64 = self.calibration.iter().map(|r| r.len).sum();
        let f = &self.fit;
        let l = &self.limits_per_family;
        if calibration > 1_000_000 || !(1..=10_000).contains(&f.steps)
            || !positive(f.learning_rate) || f.learning_rate > 1.0
            || !positive(f.temperature) || f.temperature > 10.0
            || !positive(f.difference_step) || !(0.0001..=0.01).contains(&f.difference_step)
            || !positive(l.seconds) || l.seconds > 86_400.0
            || l.record_evaluations == 0 || l.rewrites == 0 || l.compilations == 0 || l.io_bytes == 0
            || !(1..=100_000).contains(&self.independent_trials_per_budget)
            || !(1..=10_000).contains(&self.latency_records)
            || !(3..=1001).contains(&self.latency_repetitions)
            || !self.preparation_seconds.is_finite() || self.preparation_seconds < 0.0
        { return Err("invalid fit, compute or preparation limits".into()); }
        let c = &self.criteria;
        if !positive(c.min_paired_score_gain)
            || !c.max_validation_error_increase.is_finite() || !(0.0..=1.0).contains(&c.max_validation_error_increase)
            || !c.min_latency_reduction.is_finite() || !(0.0..1.0).contains(&c.min_latency_reduction)
            || !c.max_capacity_overshoot.is_finite() || !(0.0..=0.5).contains(&c.max_capacity_overshoot)
            || !c.max_cost_imbalance.is_finite() || !(0.0..=0.5).contains(&c.max_cost_imbalance)
        { return Err("invalid numerical success criteria".into()); }
        Ok(())
    }
}

fn positive(x: f64) -> bool { x.is_finite() && x > 0.0 }
fn digest_valid(s: &str) -> bool { s.len() == 64 && s.bytes().all(|c| c.is_ascii_hexdigit()) }

fn validate_ranges(ranges: &[RecordRange], total: u64, label: &str) -> Result<(), String> {
    if ranges.is_empty() || ranges.len() > 10_000
        || ranges.iter().any(|r| r.len == 0 || r.start >= total || r.len > total - r.start)
        || ranges.windows(2).any(|p| p[0].end() > p[1].start)
    { return Err(format!("invalid {label} ranges: require sorted, non-empty, non-overlapping records within corpus")); }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::corpus::write_bin_file;
    use crate::fixtures::wide_creature;

    #[test]
    fn invalid_manifests_refuse_the_specific_contract_and_valid_controls_pass() {
        let tmp = tempfile::tempdir().unwrap();
        let parent = tmp.path().join("parent.json");
        std::fs::write(&parent, neat_core::creature_to_json(&wide_creature(1, 4, "TANH")).unwrap()).unwrap();
        let data = tmp.path().join("data"); std::fs::create_dir(&data).unwrap();
        write_bin_file(&data.join("0.bin"), &vec![(vec![0.2], vec![0.1]); 16]).unwrap();
        let path = tmp.path().join("manifest.json");
        prepare_manifest(&parent, &data, &std::env::current_exe().unwrap(), &path, 8, 4).unwrap();
        let manifest: Manifest = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        let info = corpus_info(&data, &TrainingDataConfig::new(1, 1)).unwrap();
        manifest.validate(&info).unwrap();
        let mut changed = manifest.clone(); changed.validation = changed.calibration.clone();
        assert!(changed.validate(&info).unwrap_err().contains("overlap"));
        changed = manifest.clone(); changed.calibration[0].len = 0;
        assert!(changed.validate(&info).unwrap_err().contains("invalid calibration ranges"));
        changed = manifest.clone(); changed.validation[0] = RecordRange { start: 15, len: u64::MAX };
        assert!(changed.validate(&info).unwrap_err().contains("invalid validation ranges"));
        changed = manifest.clone(); changed.seeds = vec![7, 7];
        assert!(changed.validate(&info).unwrap_err().contains("distinct paired seeds"));
        changed = manifest.clone(); changed.format_version = 2;
        assert!(changed.validate(&info).unwrap_err().contains("unsupported manifest version"));
        changed = manifest.clone(); changed.fit.temperature = f64::NAN;
        assert!(changed.validate(&info).unwrap_err().contains("invalid fit"));
        changed = manifest.clone(); changed.criteria.max_cost_imbalance = -0.1;
        assert!(changed.validate(&info).unwrap_err().contains("success criteria"));
        changed = manifest.clone(); changed.core_commit = "f".repeat(40);
        assert!(changed.validate(&info).unwrap_err().contains("unsupported manifest"));
        let mut value = serde_json::to_value(&manifest).unwrap(); value["unrecognisedSwitch"] = true.into();
        assert!(serde_json::from_value::<Manifest>(value).unwrap_err().to_string().contains("unknown field"));
        manifest.validate(&info).unwrap();
    }
}
