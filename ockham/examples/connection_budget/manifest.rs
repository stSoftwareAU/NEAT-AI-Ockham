use std::collections::{BTreeSet, HashSet};
use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use neat_ai_ockham::corpus::{corpus_info, for_each_chunk};
use neat_ai_ockham::incumbent::{load_incumbent, sha256_hex};
use neat_core::training_data::TrainingDataConfig;
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use sha2::{Digest, Sha256};

pub const PAPER_TEXT_SHA256: &str =
    "786335336bdeaf76ce456aabf353177039ce76310a5f9963b580cdb540c71da3";

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
pub struct Spec {
    pub format_version: u32,
    pub study_kind: StudyKind,
    pub independence_statement: String,
    pub expected_independent_units: usize,
    pub creatures: Vec<Subject>,
    pub seeds: Vec<u64>,
    pub connection_targets: Vec<usize>,
    pub scorer: PathBuf,
    pub scorer_args: Vec<String>,
    pub limits: Limits,
    pub success: Success,
    pub latency: Latency,
    pub screen_rate: f64,
    pub min_improvement: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum StudyKind {
    SyntheticPilot,
    Representative,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
pub struct Subject {
    pub id: String,
    pub independent_unit: String,
    pub provenance: String,
    pub creature: PathBuf,
    pub training: PathBuf,
    pub calibration: PathBuf,
    pub validation: PathBuf,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
pub struct Limits {
    pub proposal_attempts: usize,
    pub statistics_records: u64,
    pub scorer_calls: u64,
    pub scorer_record_evaluations: u64,
    pub search_seconds: u64,
    pub call_timeout_seconds: u64,
    pub max_corpus_records: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
pub struct Success {
    pub minimum_independent_units: usize,
    pub minimum_matched_fraction: f64,
    pub minimum_score_gain: f64,
    pub maximum_worst_output_mse_increase: f64,
    pub compute_relative_tolerance: f64,
    pub bootstrap_replicates: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
pub struct Latency {
    pub records: usize,
    pub warmup_passes: usize,
    pub measured_passes: usize,
}

impl Spec {
    pub fn validate(&self) -> Result<(), String> {
        if self.format_version != 1 {
            return Err("unsupported specification formatVersion".into());
        }
        if self.creatures.is_empty()
            || self.creatures.len() > 1000
            || self.seeds.is_empty()
            || self.seeds.len() > 100
            || self.connection_targets.is_empty()
            || self.connection_targets.len() > 100
        {
            return Err(
                "study size is empty or exceeds 1000 subjects / 100 seeds / 100 targets".into(),
            );
        }
        if self.creatures.len() * self.seeds.len() * self.connection_targets.len() * 4 > 10_000 {
            return Err("study exceeds 10000 arm runs".into());
        }
        if self.independence_statement.trim().is_empty() {
            return Err("independenceStatement is required".into());
        }
        let mut ids = BTreeSet::new();
        for subject in &self.creatures {
            for label in [&subject.id, &subject.independent_unit] {
                if label.is_empty()
                    || label.len() > 64
                    || !label
                        .bytes()
                        .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_')
                {
                    return Err("subject and independent unit IDs require 1..64 ASCII letters, digits, hyphens or underscores".into());
                }
            }
            if !ids.insert(&subject.id) {
                return Err("duplicate subject ID".into());
            }
            if subject.provenance.trim().is_empty() {
                return Err("subject provenance is required".into());
            }
        }
        if self
            .creatures
            .iter()
            .map(|s| &s.independent_unit)
            .collect::<BTreeSet<_>>()
            .len()
            != self.expected_independent_units
        {
            return Err("expectedIndependentUnits disagrees with subject clustering".into());
        }
        if self.seeds.iter().collect::<BTreeSet<_>>().len() != self.seeds.len()
            || self
                .connection_targets
                .iter()
                .collect::<BTreeSet<_>>()
                .len()
                != self.connection_targets.len()
        {
            return Err("duplicate seed or connection target".into());
        }
        let limits = &self.limits;
        if limits.proposal_attempts == 0
            || limits.statistics_records == 0
            || limits.scorer_calls < 3
            || limits.scorer_record_evaluations == 0
            || limits.search_seconds == 0
            || limits.call_timeout_seconds == 0
            || limits.max_corpus_records < 2
            || limits.max_corpus_records > 10_000_000
        {
            return Err("invalid compute limits (at least three scorer calls and two corpus records required)".into());
        }
        if limits.proposal_attempts > 1_000_000
            || limits.statistics_records > 100_000_000
            || limits.scorer_calls > 1_000_000
            || limits.scorer_record_evaluations > 1_000_000_000_000
            || limits.search_seconds > 86_400
            || limits.call_timeout_seconds > 3600
        {
            return Err("compute limits exceed supported research bounds".into());
        }
        if !(0.0..1.0).contains(&self.screen_rate)
            || self.screen_rate == 0.0
            || !self.min_improvement.is_finite()
            || self.min_improvement <= 0.0
        {
            return Err(
                "screenRate must be in (0,1) and minImprovement finite and positive".into(),
            );
        }
        let success = &self.success;
        if success.minimum_independent_units < 2
            || !(0.0..=1.0).contains(&success.minimum_matched_fraction)
            || success.minimum_matched_fraction == 0.0
            || !success.minimum_score_gain.is_finite()
            || success.minimum_score_gain < 0.0
            || !success.maximum_worst_output_mse_increase.is_finite()
            || success.maximum_worst_output_mse_increase < 0.0
            || !(0.0..=1.0).contains(&success.compute_relative_tolerance)
            || !(100..=100_000).contains(&success.bootstrap_replicates)
        {
            return Err("invalid success criteria or bootstrap limits".into());
        }
        if self.latency.records == 0
            || self.latency.records > 100_000
            || self.latency.measured_passes == 0
            || self.latency.measured_passes > 1000
            || self.latency.warmup_passes > 1000
        {
            return Err("invalid latency sample or pass count".into());
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
pub struct CorpusPin {
    pub sha256: String,
    pub records: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
pub struct InputPin {
    pub creature_sha256: String,
    pub training: CorpusPin,
    pub calibration: CorpusPin,
    pub validation: CorpusPin,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
pub struct Frozen {
    pub format_version: u32,
    pub spec: Spec,
    pub inputs: Vec<InputPin>,
    pub source_sha256: String,
    pub executable_sha256: String,
    pub scorer_sha256: String,
    pub code_revision: String,
    pub crate_version: String,
    pub paper_version: String,
    pub paper_text_sha256: String,
    pub freeze_ms: f64,
}

pub fn read_json<T: DeserializeOwned>(path: &Path) -> Result<T, String> {
    serde_json::from_reader(File::open(path).map_err(|e| format!("{}: {e}", path.display()))?)
        .map_err(|e| format!("{}: {e}", path.display()))
}

pub fn write_new<T: Serialize>(path: &Path, value: &T) -> Result<(), String> {
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|e| format!("{}: {e}", path.display()))?;
    serde_json::to_writer_pretty(&mut file, value).map_err(|e| e.to_string())?;
    file.write_all(b"\n").map_err(|e| e.to_string())?;
    file.sync_all().map_err(|e| e.to_string())
}

pub fn file_sha256(path: &Path) -> Result<String, String> {
    let mut file = File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buf = [0u8; 65536];
    loop {
        let n = file.read(&mut buf).map_err(|e| e.to_string())?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

struct CorpusRows {
    pin: CorpusPin,
    rows: HashSet<[u8; 32]>,
    inputs: HashSet<[u8; 32]>,
}

fn pin_corpus(path: &Path, config: &TrainingDataConfig, cap: u64) -> Result<CorpusRows, String> {
    let info = corpus_info(path, config)?;
    if info.record_count < 2 || info.record_count > cap {
        return Err("corpus outside declared record limits".into());
    }
    let mut digest = Sha256::new();
    digest.update((config.num_inputs as u64).to_le_bytes());
    digest.update((config.num_outputs as u64).to_le_bytes());
    let mut rows = HashSet::new();
    let mut inputs = HashSet::new();
    let records = for_each_chunk(path, config, 4096, |chunk| {
        for record in 0..chunk.records {
            let input = &chunk.inputs[record * config.num_inputs..(record + 1) * config.num_inputs];
            let target =
                &chunk.targets[record * config.num_outputs..(record + 1) * config.num_outputs];
            let mut bytes = Vec::with_capacity((input.len() + target.len()) * 4);
            for value in input.iter().chain(target) {
                if !value.is_finite() {
                    return Err("non-finite corpus value".into());
                }
                // Normalise signed zero for overlap detection as well as identity.
                bytes.extend_from_slice(&if *value == 0.0 { 0.0f32 } else { *value }.to_le_bytes());
            }
            digest.update(&bytes);
            inputs.insert(Sha256::digest(&bytes[..input.len() * 4]).into());
            rows.insert(Sha256::digest(&bytes).into());
        }
        Ok(())
    })?;
    Ok(CorpusRows {
        pin: CorpusPin {
            sha256: format!("{:x}", digest.finalize()),
            records,
        },
        rows,
        inputs,
    })
}

pub fn pin_inputs(spec: &Spec) -> Result<Vec<InputPin>, String> {
    let mut pins = Vec::new();
    let mut creatures = HashSet::new();
    for subject in &spec.creatures {
        let parent = load_incumbent(&subject.creature).map_err(|e| e.to_string())?;
        if spec
            .connection_targets
            .iter()
            .any(|target| *target > parent.creature.synapses.len())
        {
            return Err(
                "connection target exceeds parent; a pruning budget cannot request growth".into(),
            );
        }
        let canonical = neat_core::creature_to_json(&parent.creature).map_err(|e| e.to_string())?;
        if !creatures.insert(sha256_hex(canonical.as_bytes())) {
            return Err("duplicate creature; independent evidence cannot be cloned".into());
        }
        super::allocation::Constraints::new(&parent.creature)?;
        let config = TrainingDataConfig::new(parent.creature.input, parent.creature.output);
        let training = pin_corpus(&subject.training, &config, spec.limits.max_corpus_records)?;
        let calibration = pin_corpus(
            &subject.calibration,
            &config,
            spec.limits.max_corpus_records,
        )?;
        let validation = pin_corpus(&subject.validation, &config, spec.limits.max_corpus_records)?;
        if !calibration.rows.is_subset(&training.rows) {
            return Err("calibration records are not a subset of full training data".into());
        }
        if !validation.inputs.is_disjoint(&training.inputs) {
            return Err("validation inputs overlap full training data".into());
        }
        pins.push(InputPin {
            creature_sha256: parent.checksum,
            training: training.pin,
            calibration: calibration.pin,
            validation: validation.pin,
        });
    }
    Ok(pins)
}

pub fn source_sha256() -> Result<String, String> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();
    let mut paths = vec![
        root.join("Cargo.toml"),
        root.join("Cargo.lock"),
        root.join("rust-toolchain.toml"),
        root.join("ockham/Cargo.toml"),
    ];
    fn walk(dir: &Path, paths: &mut Vec<PathBuf>) -> Result<(), String> {
        for entry in std::fs::read_dir(dir).map_err(|e| e.to_string())? {
            let path = entry.map_err(|e| e.to_string())?.path();
            if path.is_dir() {
                walk(&path, paths)?;
            } else if path.extension().is_some_and(|e| e == "rs") {
                paths.push(path);
            }
        }
        Ok(())
    }
    walk(&root.join("ockham/src"), &mut paths)?;
    walk(&root.join("ockham/examples"), &mut paths)?;
    paths.sort();
    let mut hasher = Sha256::new();
    for path in paths {
        hasher.update(
            path.strip_prefix(root)
                .unwrap()
                .to_string_lossy()
                .as_bytes(),
        );
        hasher.update([0]);
        hasher.update(file_sha256(&path)?.as_bytes());
    }
    Ok(format!("{:x}", hasher.finalize()))
}

pub fn freeze(spec_path: &Path, manifest_path: &Path) -> Result<(), String> {
    let started = std::time::Instant::now();
    let spec: Spec = read_json(spec_path)?;
    spec.validate()?;
    let inputs = pin_inputs(&spec)?;
    let revision = std::process::Command::new("git")
        .args(["rev-parse", "HEAD"])
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()
        .map_err(|e| e.to_string())?;
    if !revision.status.success() {
        return Err("cannot identify code revision".into());
    }
    let frozen = Frozen {
        format_version: 1,
        inputs,
        source_sha256: source_sha256()?,
        executable_sha256: file_sha256(&std::env::current_exe().map_err(|e| e.to_string())?)?,
        scorer_sha256: file_sha256(&spec.scorer)?,
        code_revision: String::from_utf8(revision.stdout)
            .map_err(|e| e.to_string())?
            .trim()
            .into(),
        crate_version: env!("CARGO_PKG_VERSION").into(),
        paper_version: "2611.09233v1".into(),
        paper_text_sha256: PAPER_TEXT_SHA256.into(),
        spec,
        freeze_ms: started.elapsed().as_secs_f64() * 1000.0,
    };
    write_new(manifest_path, &frozen)
}

impl Frozen {
    pub fn verify(&self) -> Result<(), String> {
        self.spec.validate()?;
        if self.format_version != 1
            || self.paper_version != "2611.09233v1"
            || self.paper_text_sha256 != PAPER_TEXT_SHA256
        {
            return Err("unsupported manifest or paper pin".into());
        }
        if source_sha256()? != self.source_sha256
            || file_sha256(&std::env::current_exe().map_err(|e| e.to_string())?)?
                != self.executable_sha256
            || file_sha256(&self.spec.scorer)? != self.scorer_sha256
        {
            return Err("code, executable or scorer changed after freeze".into());
        }
        if pin_inputs(&self.spec)? != self.inputs {
            return Err("input changed after freeze".into());
        }
        Ok(())
    }
}
