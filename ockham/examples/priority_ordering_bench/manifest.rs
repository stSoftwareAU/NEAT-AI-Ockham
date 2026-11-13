use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

use neat_ai_ockham::incumbent::sha256_hex;
use neat_ai_ockham::model::pairwise::FitConfig;
use serde::{Deserialize, Serialize};

pub const PROTOCOL: &str = "ockham-ranking-v1";
pub const SAMPLE_RATE: f64 = 0.05;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Split {
    Source,
    Calibration,
    Validation,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct HistorySpec {
    pub path: PathBuf,
    pub sha256: String,
    pub acquisition_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct Task {
    pub id: String,
    pub family: String,
    pub lineage: String,
    pub corpus_version: String,
    pub split: Split,
    pub observed_unix: u64,
    pub parent: PathBuf,
    pub parent_sha256: String,
    pub corpus: PathBuf,
    pub corpus_sha256: String,
    pub corpus_identity: String,
    pub history: Option<HistorySpec>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct Manifest {
    pub format_version: u32,
    pub protocol: String,
    pub code_revision: String,
    pub scorer: PathBuf,
    pub scorer_sha256: String,
    pub scorer_args: Vec<String>,
    pub historical_cutoff_unix: u64,
    pub seeds: Vec<u64>,
    pub pilot_visits: usize,
    pub audit_visits_ceiling: usize,
    pub minimum_complete_pool: usize,
    pub minimum_independent_groups: usize,
    pub max_task_ms: u64,
    pub max_run_ms: u64,
    pub scorer_timeout_secs: u64,
    pub stats_records: u64,
    pub max_corpus_bytes: u64,
    pub evaluation_budgets: Vec<usize>,
    pub calibration_budgets: Vec<usize>,
    pub history_fractions: Vec<f64>,
    pub fit: FitConfig,
    pub bootstrap_replicates: usize,
    pub minimum_recall_gain: f64,
    pub maximum_false_rejection: f64,
    pub maximum_cost_ratio: f64,
    pub minimum_improvement: f64,
    pub tasks: Vec<Task>,
}

fn digest(value: &str, length: usize) -> bool {
    value.len() == length && value.bytes().all(|c| c.is_ascii_hexdigit())
}

fn unique<T: Ord>(values: &[T]) -> bool {
    values.iter().collect::<BTreeSet<_>>().len() == values.len()
}

impl Manifest {
    pub fn validate(&self) -> Result<(), String> {
        if self.format_version != 1 || self.protocol != PROTOCOL {
            return Err("unsupported experiment protocol/version".into());
        }
        if !digest(&self.code_revision, 40) || !digest(&self.scorer_sha256, 64) {
            return Err("code/scorer identities must be full hex digests".into());
        }
        self.fit.validate()?;
        if self.tasks.is_empty()
            || self.tasks.len() > 128
            || self.seeds.is_empty()
            || self.seeds.len() > 32
            || !unique(&self.seeds)
            || !(1..=16).contains(&self.pilot_visits)
            || !(20..=512).contains(&self.audit_visits_ceiling)
            || self.minimum_complete_pool < 20
            || self.minimum_complete_pool > self.audit_visits_ceiling
            || self.minimum_independent_groups < 3
            || self.max_task_ms == 0
            || self.max_run_ms < self.max_task_ms
            || self.scorer_timeout_secs == 0
            || self.scorer_timeout_secs > 600
            || self.stats_records == 0
            || self.stats_records > 100_000
            || self.max_corpus_bytes == 0
            || !(200..=10_000).contains(&self.bootstrap_replicates)
        {
            return Err("invalid experiment compute/sample limits".into());
        }
        if self.evaluation_budgets.is_empty()
            || self.evaluation_budgets.len() > 8
            || !unique(&self.evaluation_budgets)
            || self
                .evaluation_budgets
                .iter()
                .any(|b| *b == 0 || *b > self.audit_visits_ceiling)
            || self.calibration_budgets.is_empty()
            || self.calibration_budgets.len() > 8
            || !unique(&self.calibration_budgets)
            || !self.calibration_budgets.contains(&0)
            || self.calibration_budgets.iter().any(|b| *b > 512)
            || self.history_fractions.len() > 8
            || !self.history_fractions.contains(&0.0)
            || !self.history_fractions.contains(&1.0)
            || self
                .history_fractions
                .iter()
                .any(|x| !x.is_finite() || *x < 0.0 || *x > 1.0)
        {
            return Err(
                "invalid budget/history grid (must include cold starts and full history)".into(),
            );
        }
        if !self.minimum_recall_gain.is_finite()
            || self.minimum_recall_gain < 0.0
            || self.minimum_recall_gain > 1.0
            || !self.maximum_false_rejection.is_finite()
            || !(0.0..=1.0).contains(&self.maximum_false_rejection)
            || !self.maximum_cost_ratio.is_finite()
            || self.maximum_cost_ratio <= 0.0
            || !self.minimum_improvement.is_finite()
            || self.minimum_improvement < 0.0
        {
            return Err("invalid success thresholds".into());
        }
        let mut ids = BTreeSet::new();
        let mut splits = BTreeMap::new();
        let mut present = BTreeSet::new();
        for t in &self.tasks {
            if t.id.trim().is_empty()
                || t.id.len() > 128
                || !t
                    .id
                    .bytes()
                    .all(|b| b.is_ascii_alphanumeric() || b"-_.".contains(&b))
                || !ids.insert(&t.id)
                || t.family.trim().is_empty()
                || t.lineage.trim().is_empty()
                || t.corpus_version.trim().is_empty()
                || !digest(&t.parent_sha256, 64)
                || !digest(&t.corpus_sha256, 64)
            {
                return Err("invalid or duplicate task identity".into());
            }
            if t.corpus_identity.is_empty() {
                return Err("missing corpus cache identity".into());
            }
            match (&t.history,t.split) {
                (Some(h),Split::Source) if digest(&h.sha256,64) => {},
                (None,Split::Calibration|Split::Validation) => {},
                _ => return Err("source tasks require a digested historical log; target tasks cannot supply one".into()),
            }
            present.insert(t.split);
            if t.split == Split::Source && t.observed_unix >= self.historical_cutoff_unix {
                return Err(format!("{} violates historical cutoff", t.id));
            }
            // Content identities also prevent renamed copies bypassing the split.
            for (kind, value) in [
                ("lineage", &t.lineage),
                ("version", &t.corpus_version),
                ("parent", &t.parent_sha256),
                ("corpus", &t.corpus_sha256),
            ] {
                if let Some(old) = splits.insert((kind, value.to_lowercase()), t.split)
                    && old != t.split
                {
                    return Err(format!(
                        "split leakage: {} shares {kind} across partitions",
                        t.id
                    ));
                }
            }
        }
        if present.len() != 3 {
            return Err("source, calibration and validation tasks are required".into());
        }
        for t in self.tasks.iter().filter(|t| t.split == Split::Validation) {
            if !self
                .tasks
                .iter()
                .any(|c| c.split == Split::Calibration && c.family == t.family)
            {
                return Err(format!("{} has no target calibration family", t.id));
            }
        }
        Ok(())
    }

    pub fn identity(&self) -> Result<String, String> {
        Ok(sha256_hex(
            &serde_json::to_vec(self).map_err(|e| e.to_string())?,
        ))
    }
}

// Hash priorities give a deterministic random permutation across *all* visits,
// rather than the ordinary sweep's
// neuron/edge interleaving.
pub fn seeded_order(items: &[String], seed: u64) -> Vec<String> {
    let mut keyed: Vec<_> = items
        .iter()
        .map(|id| {
            let mut key = seed.to_le_bytes().to_vec();
            key.extend_from_slice(id.as_bytes());
            (sha256_hex(&key), id.clone())
        })
        .collect();
    keyed.sort();
    keyed.into_iter().map(|(_, id)| id).collect()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct Freeze {
    pub format_version: u32,
    pub manifest_sha256: String,
    pub pilot_sha256: String,
    pub audit_visits: usize,
}
