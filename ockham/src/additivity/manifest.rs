//! Benchmark manifest for the additivity pruning benchmark (Issue #260).
//!
//! The manifest is the single predeclared description of an offline run: the
//! immutable parent creatures, the corpora, the scorer, the budgets and the
//! adoption rule. Relative paths resolve against the manifest's own directory,
//! and unknown fields are refused so a typo cannot silently change the protocol.

use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

/// The only manifest layout this build understands.
pub const MANIFEST_FORMAT_VERSION: u32 = 1;

/// Default fraction of the training corpus used as the audit core.
pub const DEFAULT_CORE_FRACTION: f64 = 0.02;

/// Default pruning budgets, as fractions of each parent's prunable mass.
pub const DEFAULT_BUDGETS: [f64; 4] = [0.1, 0.2, 0.3, 0.4];

/// Default number of top-ranked bundles promoted to full scoring.
pub const DEFAULT_TOP_K: usize = 3;

/// Parent count below which results are fixture-only, not evidence.
pub const REPRESENTATIVE_PARENTS: usize = 20;

/// Largest manifest file accepted, checked before the file is read.
pub const MAX_MANIFEST_BYTES: u64 = 1 << 20;

/// The external scorer the benchmark shells out to.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ScorerSpec {
    /// Scorer executable. A value containing a path separator resolves against
    /// the manifest directory; a bare name is a `$PATH` lookup and stays as is.
    pub binary: PathBuf,
    /// Extra arguments passed to the scorer.
    #[serde(default)]
    pub args: Vec<String>,
}

/// Hard caps on the resources one benchmark run may consume.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ResourceLimits {
    /// Total bytes the output cache may hold on disk.
    pub max_cache_bytes: u64,
    /// Records streamed per chunk, bounding memory per reader or writer.
    pub chunk_records: usize,
}

/// A predeclared offline benchmark: parents, corpora, scorer and decision rule.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct BenchManifest {
    /// Manifest layout version; must equal [`MANIFEST_FORMAT_VERSION`].
    pub format_version: u32,
    /// Immutable elite creature JSON files, one per parent.
    pub parents: Vec<PathBuf>,
    /// Training corpus directory.
    pub training_data: PathBuf,
    /// Untouched held-out corpus directory, if any.
    #[serde(default)]
    pub validation_data: Option<PathBuf>,
    /// External scorer used for full evaluations.
    pub scorer: ScorerSpec,
    /// Cost function name. Only `"MSE"` is accepted (case-sensitive).
    pub cost: String,
    /// Random seeds; each (parent, budget, seed) triple is an independent run.
    pub seeds: Vec<u64>,
    /// Fraction of the training corpus used as the audit core.
    #[serde(default = "default_core_fraction")]
    pub core_fraction: f64,
    /// Pruning budgets, strictly ascending fractions in (0, 1].
    #[serde(default = "default_budgets")]
    pub budgets: Vec<f64>,
    /// Bundles audited per (parent, budget, seed).
    pub audit_pool: usize,
    /// Predeclared per-arm budget of full-scorer evaluations.
    pub full_evaluations: usize,
    /// Number of top-ranked bundles promoted to full scoring.
    #[serde(default = "default_top_k")]
    pub top_k: usize,
    /// Promotion threshold on the full-score delta.
    pub min_improvement: f64,
    /// Predeclared adoption rule, echoed verbatim in the report.
    pub decision_rule: String,
    /// Resource caps for the run.
    pub limits: ResourceLimits,
    /// Root directory of the output cache.
    pub cache_dir: PathBuf,
}

fn default_core_fraction() -> f64 {
    DEFAULT_CORE_FRACTION
}

fn default_budgets() -> Vec<f64> {
    DEFAULT_BUDGETS.to_vec()
}

fn default_top_k() -> usize {
    DEFAULT_TOP_K
}

impl BenchManifest {
    /// Read, resolve and validate a manifest file.
    ///
    /// The file size is checked before any read, and relative paths resolve
    /// against the manifest's parent directory. Every error names `path`.
    pub fn load(path: &Path) -> Result<Self, String> {
        let size = fs::metadata(path)
            .map_err(|e| format!("manifest '{}': cannot read: {e}", path.display()))?
            .len();
        if size > MAX_MANIFEST_BYTES {
            return Err(format!(
                "manifest '{}' is {size} bytes, over the {MAX_MANIFEST_BYTES}-byte limit",
                path.display()
            ));
        }
        let text = fs::read_to_string(path)
            .map_err(|e| format!("manifest '{}': cannot read: {e}", path.display()))?;
        let mut manifest: Self = serde_json::from_str(&text)
            .map_err(|e| format!("manifest '{}': {e}", path.display()))?;
        let base = path.parent().unwrap_or(Path::new("."));
        manifest.parents = manifest.parents.iter().map(|p| base.join(p)).collect();
        manifest.training_data = base.join(&manifest.training_data);
        manifest.validation_data = manifest.validation_data.map(|p| base.join(p));
        if manifest
            .scorer
            .binary
            .to_string_lossy()
            .contains(std::path::MAIN_SEPARATOR)
        {
            manifest.scorer.binary = base.join(&manifest.scorer.binary);
        }
        manifest.cache_dir = base.join(&manifest.cache_dir);
        manifest
            .validate()
            .map_err(|e| format!("manifest '{}': {e}", path.display()))?;
        Ok(manifest)
    }

    /// Check every protocol bound, refusing with a message that names the field.
    pub fn validate(&self) -> Result<(), String> {
        if self.format_version != MANIFEST_FORMAT_VERSION {
            return Err(format!(
                "formatVersion must be {MANIFEST_FORMAT_VERSION}, got {}",
                self.format_version
            ));
        }
        if !(1..=64).contains(&self.parents.len()) {
            return Err(format!(
                "parents must list 1 to 64 files, got {}",
                self.parents.len()
            ));
        }
        let mut seen_parents = HashSet::new();
        for parent in &self.parents {
            if !seen_parents.insert(parent) {
                return Err(format!(
                    "parents must be distinct, but '{}' is listed more than once",
                    parent.display()
                ));
            }
        }
        if !(1..=32).contains(&self.seeds.len()) {
            return Err(format!(
                "seeds must list 1 to 32 values, got {}",
                self.seeds.len()
            ));
        }
        let mut seen_seeds = HashSet::new();
        for seed in &self.seeds {
            if !seen_seeds.insert(seed) {
                return Err(format!("seeds must be distinct, but {seed} appears twice"));
            }
        }
        if self.cost != "MSE" {
            return Err(format!(
                "cost must be \"MSE\" (case-sensitive), got \"{}\": other costs need a per-record \
                 cost API in NEAT-AI-core, which only exports neat_core::loss::mse_record today",
                self.cost
            ));
        }
        let core = self.core_fraction;
        if !(core.is_finite() && core > 0.0 && core <= 1.0) {
            return Err(format!(
                "coreFraction must be finite and in (0, 1], got {core}"
            ));
        }
        if !(1..=8).contains(&self.budgets.len()) {
            return Err(format!(
                "budgets must list 1 to 8 fractions, got {}",
                self.budgets.len()
            ));
        }
        for budget in &self.budgets {
            if !(budget.is_finite() && *budget > 0.0 && *budget <= 1.0) {
                return Err(format!(
                    "budgets values must be finite and in (0, 1], got {budget}"
                ));
            }
        }
        if self.budgets.windows(2).any(|w| w[1] <= w[0]) {
            return Err("budgets must be strictly ascending".to_string());
        }
        if !(2..=64).contains(&self.audit_pool) {
            return Err(format!(
                "auditPool must be between 2 and 64, got {}",
                self.audit_pool
            ));
        }
        if !(1..=self.audit_pool).contains(&self.full_evaluations) {
            return Err(format!(
                "fullEvaluations must be between 1 and auditPool ({}), got {}",
                self.audit_pool, self.full_evaluations
            ));
        }
        if !(1..=self.audit_pool).contains(&self.top_k) {
            return Err(format!(
                "topK must be between 1 and auditPool ({}), got {}",
                self.audit_pool, self.top_k
            ));
        }
        if !(self.min_improvement.is_finite() && self.min_improvement >= 0.0) {
            return Err(format!(
                "minImprovement must be finite and non-negative, got {}",
                self.min_improvement
            ));
        }
        if self.decision_rule.trim().is_empty() {
            return Err("decisionRule must not be empty".to_string());
        }
        if self.decision_rule.chars().count() > 2000 {
            return Err("decisionRule must be at most 2000 characters".to_string());
        }
        if self.limits.max_cache_bytes == 0 {
            return Err("limits.maxCacheBytes must be greater than zero".to_string());
        }
        if !(1..=1_000_000).contains(&self.limits.chunk_records) {
            return Err(format!(
                "limits.chunkRecords must be between 1 and 1000000, got {}",
                self.limits.chunk_records
            ));
        }
        Ok(())
    }

    /// True when the parent count meets [`REPRESENTATIVE_PARENTS`].
    ///
    /// Results from fewer parents are fixture-only and are not evidence.
    pub fn representative(&self) -> bool {
        self.parents.len() >= REPRESENTATIVE_PARENTS
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const VALID_JSON: &str = r#"{
        "formatVersion": 1,
        "parents": ["parents/a.json", "parents/b.json"],
        "trainingData": "data/train",
        "validationData": "data/validate",
        "scorer": { "binary": "bin/scorer", "args": ["--fast"] },
        "cost": "MSE",
        "seeds": [1, 2, 3],
        "auditPool": 8,
        "fullEvaluations": 4,
        "minImprovement": 0.0,
        "decisionRule": "adopt when the full-score delta reaches minImprovement",
        "limits": { "maxCacheBytes": 1048576, "chunkRecords": 1000 },
        "cacheDir": "cache"
    }"#;

    fn valid() -> BenchManifest {
        BenchManifest {
            format_version: MANIFEST_FORMAT_VERSION,
            parents: vec![PathBuf::from("a.json"), PathBuf::from("b.json")],
            training_data: PathBuf::from("train"),
            validation_data: None,
            scorer: ScorerSpec {
                binary: PathBuf::from("scorer"),
                args: Vec::new(),
            },
            cost: "MSE".to_string(),
            seeds: vec![1, 2, 3],
            core_fraction: DEFAULT_CORE_FRACTION,
            budgets: DEFAULT_BUDGETS.to_vec(),
            audit_pool: 8,
            full_evaluations: 4,
            top_k: DEFAULT_TOP_K,
            min_improvement: 0.0,
            decision_rule: "adopt when the full-score delta reaches minImprovement".to_string(),
            limits: ResourceLimits {
                max_cache_bytes: 1 << 20,
                chunk_records: 1000,
            },
            cache_dir: PathBuf::from("cache"),
        }
    }

    fn write_manifest(dir: &Path, json: &str) -> PathBuf {
        let path = dir.join("manifest.json");
        fs::write(&path, json).unwrap();
        path
    }

    #[test]
    fn valid_manifest_loads_with_paths_resolved_and_defaults_applied() {
        let tmp = tempfile::tempdir().unwrap();
        let path = write_manifest(tmp.path(), VALID_JSON);
        let m = BenchManifest::load(&path).unwrap();
        let base = tmp.path();
        assert_eq!(
            m.parents,
            vec![base.join("parents/a.json"), base.join("parents/b.json")]
        );
        assert_eq!(m.training_data, base.join("data/train"));
        assert_eq!(m.validation_data, Some(base.join("data/validate")));
        assert_eq!(m.scorer.binary, base.join("bin/scorer"));
        assert_eq!(m.scorer.args, vec!["--fast".to_string()]);
        assert_eq!(m.cache_dir, base.join("cache"));
        assert_eq!(m.core_fraction, 0.02);
        assert_eq!(m.budgets, vec![0.1, 0.2, 0.3, 0.4]);
        assert_eq!(m.top_k, 3);
        assert!(!m.representative());
    }

    #[test]
    fn bare_scorer_name_stays_unresolved() {
        let tmp = tempfile::tempdir().unwrap();
        let json = VALID_JSON.replace(r#""binary": "bin/scorer""#, r#""binary": "scorer""#);
        let path = write_manifest(tmp.path(), &json);
        let m = BenchManifest::load(&path).unwrap();
        assert_eq!(m.scorer.binary, PathBuf::from("scorer"));
    }

    #[test]
    fn unknown_field_is_refused_and_named() {
        let tmp = tempfile::tempdir().unwrap();
        let json = VALID_JSON.replace(
            r#""cacheDir": "cache""#,
            r#""cacheDir": "cache", "bogusField": 1"#,
        );
        let path = write_manifest(tmp.path(), &json);
        let err = BenchManifest::load(&path).unwrap_err();
        assert!(err.contains("bogusField"), "error was: {err}");
        assert!(err.contains("manifest.json"), "error was: {err}");
    }

    #[test]
    fn oversize_file_is_refused_before_parse() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("manifest.json");
        let size = MAX_MANIFEST_BYTES + 1;
        fs::write(&path, vec![b' '; size as usize]).unwrap();
        let err = BenchManifest::load(&path).unwrap_err();
        assert!(err.contains(&size.to_string()), "error was: {err}");
        assert!(
            err.contains(&MAX_MANIFEST_BYTES.to_string()),
            "error was: {err}"
        );
    }

    #[test]
    fn each_invalid_field_is_refused_naming_that_field() {
        type Mutate = fn(&mut BenchManifest);
        let cases: [(&str, Mutate); 7] = [
            ("cost", |m| m.cost = "MAE".to_string()),
            ("fullEvaluations", |m| m.full_evaluations = m.audit_pool + 1),
            ("budgets", |m| m.budgets = vec![0.2, 0.1]),
            ("decisionRule", |m| m.decision_rule = "   ".to_string()),
            ("parents", |m| {
                m.parents = vec![PathBuf::from("a.json"), PathBuf::from("a.json")]
            }),
            ("coreFraction", |m| m.core_fraction = 0.0),
            ("chunkRecords", |m| m.limits.chunk_records = 0),
        ];
        assert!(valid().validate().is_ok());
        for (field, mutate) in cases {
            let mut m = valid();
            mutate(&mut m);
            let err = m.validate().unwrap_err();
            assert!(
                err.contains(field),
                "expected '{field}' in error, got: {err}"
            );
        }
    }

    #[test]
    fn representative_needs_twenty_parents() {
        let mut m = valid();
        m.parents = vec![PathBuf::from("only.json")];
        assert!(!m.representative());
        m.parents = (0..20)
            .map(|i| PathBuf::from(format!("p{i}.json")))
            .collect();
        assert!(m.representative());
    }
}
