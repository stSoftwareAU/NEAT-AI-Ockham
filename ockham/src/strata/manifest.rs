//! The `bench-strata` manifest: parsing, defaults and refusals (issue #263).
//!
//! The manifest is JSON with camelCase keys and no unknown keys. Its size is
//! checked from file metadata before the file is read. Relative paths resolve
//! against the manifest's own directory. Every refusal names the field at
//! fault, so a typo is stopped loudly rather than quietly defaulted.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::incumbent::sha256_hex;

/// Largest manifest accepted, checked before the file is read.
pub const MAX_MANIFEST_BYTES: u64 = 1024 * 1024;

/// Fewest calibration strata accepted.
pub const MIN_STRATA: usize = 3;

/// Most calibration strata accepted.
pub const MAX_STRATA: usize = 5;

fn default_temperatures() -> Vec<f64> {
    vec![0.25, 0.5, 1.0, 2.0, 4.0]
}

fn default_bundle_sizes() -> Vec<usize> {
    vec![2, 4]
}

fn default_bundles_per_arm() -> usize {
    6
}

fn default_pareto_candidates() -> usize {
    24
}

fn default_archive_cap() -> usize {
    8
}

fn default_latency_records() -> u64 {
    256
}

fn default_stats_records() -> u64 {
    4096
}

fn default_regression_tolerance() -> f64 {
    1e-6
}

fn default_latency_tolerance() -> f64 {
    0.05
}

/// One named stratum: a training directory of `.bin` files.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct StratumDir {
    /// Stratum name; validation names must equal the calibration names.
    pub name: String,
    /// Directory of `.bin` files. Relative paths resolve against the manifest.
    pub dir: PathBuf,
}

/// The predeclared success rule the report's criteria are judged against.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct Success {
    /// Largest mean validation-fitness loss against `aggregate` that is still
    /// accepted. Finite and non-negative.
    pub max_fitness_loss: f64,
    /// Fewest minority-stratum regressions an arm must avoid relative to
    /// `aggregate` for the regression criterion to hold.
    pub min_regression_reduction: u64,
}

/// The experiment description, with defaults applied and paths resolved.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct Manifest {
    /// Parent creature JSON. Never modified.
    pub parent: PathBuf,
    /// Calibration strata: the only data the search may look at.
    pub calibration: Vec<StratumDir>,
    /// Validation strata: scored once per seed, after selection.
    pub validation: Vec<StratumDir>,
    /// Evaluation seeds.
    pub seeds: Vec<u64>,
    /// Seed for the temperature fit, which is never one of `seeds`.
    pub fit_seed: u64,
    /// Candidate temperatures for the two informed samplers.
    #[serde(default = "default_temperatures")]
    pub temperatures: Vec<f64>,
    /// Bundle sizes, used cyclically by bundle index.
    #[serde(default = "default_bundle_sizes")]
    pub bundle_sizes: Vec<usize>,
    /// Bundles each arm scores per seed.
    #[serde(default = "default_bundles_per_arm")]
    pub bundles_per_arm: usize,
    /// Candidate bundles the Pareto arm draws and builds per seed; the
    /// cheapest by predicted cost are full-scored.
    #[serde(default = "default_pareto_candidates")]
    pub pareto_candidates: usize,
    /// Largest Pareto archive kept over the scored bundles.
    #[serde(default = "default_archive_cap")]
    pub archive_cap: usize,
    /// Relative latency band within which two latencies do not decide Pareto
    /// dominance. Finite, at least `0` and below `1`.
    #[serde(default = "default_latency_tolerance")]
    pub latency_tolerance: f64,
    /// Records per latency measurement.
    #[serde(default = "default_latency_records")]
    pub latency_records: u64,
    /// Records sampled per calibration stratum for the means; `0` scans all.
    #[serde(default = "default_stats_records")]
    pub stats_records: u64,
    /// Strict calibration improvement a bundle must beat to win. May be zero.
    #[serde(default)]
    pub min_improvement: f64,
    /// Changes smaller than this (negative) count as no regression.
    #[serde(default = "default_regression_tolerance")]
    pub regression_tolerance: f64,
    /// The predeclared success rule.
    pub success: Success,
}

/// A loaded manifest and the checksum of its exact bytes.
#[derive(Debug, Clone, PartialEq)]
pub struct Loaded {
    /// The manifest, with relative paths resolved.
    pub manifest: Manifest,
    /// SHA-256 of the manifest file's bytes.
    pub sha256: String,
}

/// Read, parse and validate the manifest at `path`.
///
/// # Errors
///
/// A message naming the file or the field at fault: an unreadable or oversized
/// file, unknown or mistyped fields, or any refused value.
pub fn load(path: &Path) -> Result<Loaded, String> {
    let meta = std::fs::metadata(path)
        .map_err(|e| format!("cannot read manifest '{}': {e}", path.display()))?;
    if meta.len() > MAX_MANIFEST_BYTES {
        return Err(format!(
            "manifest '{}' is {} bytes; the limit is {MAX_MANIFEST_BYTES} bytes (1 MiB)",
            path.display(),
            meta.len()
        ));
    }
    let bytes = std::fs::read(path)
        .map_err(|e| format!("cannot read manifest '{}': {e}", path.display()))?;
    let text = std::str::from_utf8(&bytes)
        .map_err(|e| format!("manifest '{}' is not UTF-8: {e}", path.display()))?;
    refuse_bad_success_reduction(text, path)?;
    let mut manifest: Manifest =
        serde_json::from_str(text).map_err(|e| format!("manifest '{}': {e}", path.display()))?;
    let base = path.parent().unwrap_or_else(|| Path::new("."));
    manifest.parent = resolve(base, &manifest.parent);
    for stratum in manifest
        .calibration
        .iter_mut()
        .chain(manifest.validation.iter_mut())
    {
        stratum.dir = resolve(base, &stratum.dir);
    }
    validate(&manifest)?;
    Ok(Loaded {
        manifest,
        sha256: sha256_hex(&bytes),
    })
}

/// `minRegressionReduction` is a `u64`; serde's own message would not name the
/// field, so the refusal is made here with the field in it.
fn refuse_bad_success_reduction(text: &str, path: &Path) -> Result<(), String> {
    let raw: serde_json::Value =
        serde_json::from_str(text).map_err(|e| format!("manifest '{}': {e}", path.display()))?;
    if let Some(value) = raw
        .get("success")
        .and_then(|s| s.get("minRegressionReduction"))
        && value.as_u64().is_none()
    {
        return Err(format!(
            "manifest field `success.minRegressionReduction` must be a non-negative integer, got {value}"
        ));
    }
    Ok(())
}

fn resolve(base: &Path, path: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        base.join(path)
    }
}

/// Canonical form for comparing directories; falls back to the path as given
/// when it does not exist, so the later corpus scan reports the real fault.
fn canonical(path: &Path) -> PathBuf {
    std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf())
}

fn refuse(field: &str, detail: impl std::fmt::Display) -> String {
    format!("manifest field `{field}`: {detail}")
}

fn validate(m: &Manifest) -> Result<(), String> {
    validate_strata(m)?;
    if m.seeds.is_empty() {
        return Err(refuse("seeds", "must name at least one seed"));
    }
    if has_duplicates(&m.seeds) {
        return Err(refuse("seeds", "must not repeat a seed"));
    }
    if m.seeds.contains(&m.fit_seed) {
        return Err(refuse(
            "fitSeed",
            format!("{} is one of seeds; the fit needs its own seed", m.fit_seed),
        ));
    }
    if m.temperatures.is_empty() || m.temperatures.iter().any(|t| !t.is_finite() || *t <= 0.0) {
        return Err(refuse(
            "temperatures",
            "must be non-empty, with every value finite and greater than 0",
        ));
    }
    if m.bundle_sizes.is_empty() || m.bundle_sizes.contains(&0) {
        return Err(refuse(
            "bundleSizes",
            "must be non-empty, with every size at least 1",
        ));
    }
    if m.bundles_per_arm == 0 {
        return Err(refuse("bundlesPerArm", "must be at least 1"));
    }
    if m.pareto_candidates < m.bundles_per_arm {
        return Err(refuse(
            "paretoCandidates",
            format!(
                "{} is below bundlesPerArm {}; the Pareto arm needs at least as many candidates",
                m.pareto_candidates, m.bundles_per_arm
            ),
        ));
    }
    if m.archive_cap == 0 {
        return Err(refuse("archiveCap", "must be at least 1"));
    }
    if m.latency_records == 0 {
        return Err(refuse("latencyRecords", "must be at least 1"));
    }
    if !m.min_improvement.is_finite() || m.min_improvement < 0.0 {
        return Err(refuse("minImprovement", "must be finite and not negative"));
    }
    if !m.regression_tolerance.is_finite() || m.regression_tolerance < 0.0 {
        return Err(refuse(
            "regressionTolerance",
            "must be finite and not negative",
        ));
    }
    if !m.latency_tolerance.is_finite() || m.latency_tolerance < 0.0 || m.latency_tolerance >= 1.0 {
        return Err(refuse(
            "latencyTolerance",
            "must be finite, at least 0 and below 1",
        ));
    }
    if !m.success.max_fitness_loss.is_finite() || m.success.max_fitness_loss < 0.0 {
        return Err(refuse("maxFitnessLoss", "must be finite and not negative"));
    }
    Ok(())
}

fn validate_strata(m: &Manifest) -> Result<(), String> {
    let n = m.calibration.len();
    if !(MIN_STRATA..=MAX_STRATA).contains(&n) {
        return Err(refuse(
            "calibration",
            format!("needs {MIN_STRATA} to {MAX_STRATA} strata, got {n}"),
        ));
    }
    let cal_names: Vec<&str> = m.calibration.iter().map(|s| s.name.as_str()).collect();
    if cal_names.iter().any(|name| name.trim().is_empty()) {
        return Err(refuse("calibration", "stratum names must not be empty"));
    }
    if has_duplicates(&cal_names) {
        return Err(refuse("calibration", "stratum names must be unique"));
    }
    let val_names: Vec<&str> = m.validation.iter().map(|s| s.name.as_str()).collect();
    if has_duplicates(&val_names) || val_names.len() != cal_names.len() {
        return Err(refuse(
            "validation",
            "names must be exactly the calibration names, each once",
        ));
    }
    if val_names.iter().any(|name| !cal_names.contains(name)) {
        return Err(refuse(
            "validation",
            "names must be exactly the calibration names, each once",
        ));
    }
    let cal_dirs: Vec<PathBuf> = m.calibration.iter().map(|s| canonical(&s.dir)).collect();
    for val in &m.validation {
        let dir = canonical(&val.dir);
        if cal_dirs.contains(&dir) {
            return Err(refuse(
                "validation",
                format!(
                    "stratum '{}' uses '{}', which is also a calibration directory",
                    val.name,
                    val.dir.display()
                ),
            ));
        }
    }
    Ok(())
}

fn has_duplicates<T: PartialEq>(items: &[T]) -> bool {
    items
        .iter()
        .enumerate()
        .any(|(i, a)| items[i + 1..].contains(a))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{Value, json};

    /// Build a manifest file under a fresh temp dir with every stratum
    /// directory present, apply `edit` to the JSON, and load it.
    fn load_edited(edit: impl FnOnce(&mut Value)) -> Result<Loaded, String> {
        let tmp = tempfile::tempdir().unwrap();
        for d in [
            "cal/north",
            "cal/south",
            "cal/rare",
            "val/north",
            "val/south",
            "val/rare",
        ] {
            std::fs::create_dir_all(tmp.path().join(d)).unwrap();
        }
        let mut value = base_json();
        edit(&mut value);
        let path = tmp.path().join("manifest.json");
        std::fs::write(&path, serde_json::to_string_pretty(&value).unwrap()).unwrap();
        load(&path)
    }

    fn base_json() -> Value {
        json!({
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
            "success": {"maxFitnessLoss": 0.0, "minRegressionReduction": 1}
        })
    }

    /// The refused value is named; the same manifest with only the value made
    /// legal loads.
    fn refusal(field: &str, edit_bad: impl FnOnce(&mut Value), edit_good: impl FnOnce(&mut Value)) {
        let err = load_edited(edit_bad).expect_err("refused value must not load");
        assert!(err.contains(field), "error should name `{field}`: {err}");
        load_edited(edit_good).unwrap_or_else(|e| panic!("positive control for {field}: {e}"));
    }

    #[test]
    fn defaults_are_applied_when_optional_fields_are_absent() {
        let loaded = load_edited(|_| {}).unwrap();
        let m = loaded.manifest;
        assert_eq!(m.temperatures, vec![0.25, 0.5, 1.0, 2.0, 4.0]);
        assert_eq!(m.bundle_sizes, vec![2, 4]);
        assert_eq!(m.bundles_per_arm, 6);
        assert_eq!(m.pareto_candidates, 24);
        assert_eq!(m.archive_cap, 8);
        assert_eq!(m.latency_tolerance, 0.05);
        assert_eq!(m.latency_records, 256);
        assert_eq!(m.stats_records, 4096);
        assert_eq!(m.min_improvement, 0.0);
        assert_eq!(m.regression_tolerance, 1e-6);
        assert_eq!(m.fit_seed, 9001);
        assert_eq!(loaded.sha256.len(), 64);
    }

    #[test]
    fn relative_paths_resolve_against_the_manifest_directory() {
        let tmp = tempfile::tempdir().unwrap();
        for d in [
            "cal/north",
            "cal/south",
            "cal/rare",
            "val/north",
            "val/south",
            "val/rare",
        ] {
            std::fs::create_dir_all(tmp.path().join(d)).unwrap();
        }
        let path = tmp.path().join("manifest.json");
        std::fs::write(&path, base_json().to_string()).unwrap();
        let m = load(&path).unwrap().manifest;
        assert_eq!(m.parent, tmp.path().join("parent.json"));
        assert_eq!(m.calibration[0].dir, tmp.path().join("cal/north"));
        assert_eq!(m.validation[2].dir, tmp.path().join("val/rare"));
    }

    #[test]
    fn calibration_count_outside_three_to_five_is_refused() {
        refusal(
            "calibration",
            |v| {
                v["calibration"] = json!([
                    {"name": "north", "dir": "cal/north"},
                    {"name": "south", "dir": "cal/south"}
                ]);
            },
            |_| {},
        );
    }

    #[test]
    fn empty_or_duplicated_stratum_names_are_refused() {
        refusal(
            "calibration",
            |v| v["calibration"][1]["name"] = json!("north"),
            |v| v["calibration"][1]["name"] = json!("south"),
        );
        refusal(
            "calibration",
            |v| v["calibration"][0]["name"] = json!(""),
            |v| v["calibration"][0]["name"] = json!("north"),
        );
    }

    #[test]
    fn validation_names_must_match_calibration_names() {
        refusal(
            "validation",
            |v| v["validation"][2]["name"] = json!("other"),
            |v| v["validation"][2]["name"] = json!("rare"),
        );
    }

    #[test]
    fn validation_dir_equal_to_a_calibration_dir_is_refused() {
        refusal(
            "validation",
            |v| v["validation"][0]["dir"] = json!("cal/north"),
            |v| v["validation"][0]["dir"] = json!("val/north"),
        );
    }

    #[test]
    fn empty_or_duplicated_seeds_are_refused() {
        refusal(
            "seeds",
            |v| v["seeds"] = json!([]),
            |v| v["seeds"] = json!([1]),
        );
        refusal(
            "seeds",
            |v| v["seeds"] = json!([1, 1]),
            |v| v["seeds"] = json!([1, 2]),
        );
    }

    #[test]
    fn fit_seed_inside_seeds_is_refused() {
        refusal(
            "fitSeed",
            |v| v["fitSeed"] = json!(3),
            |v| v["fitSeed"] = json!(9002),
        );
    }

    #[test]
    fn temperatures_must_be_non_empty_and_positive() {
        refusal(
            "temperatures",
            |v| v["temperatures"] = json!([]),
            |v| v["temperatures"] = json!([0.5]),
        );
        refusal(
            "temperatures",
            |v| v["temperatures"] = json!([0.5, 0.0]),
            |v| v["temperatures"] = json!([0.5, 1.0]),
        );
        refusal(
            "temperatures",
            |v| v["temperatures"] = json!([-1.0]),
            |v| v["temperatures"] = json!([1.0]),
        );
    }

    #[test]
    fn bundle_sizes_must_be_non_empty_and_non_zero() {
        refusal(
            "bundleSizes",
            |v| v["bundleSizes"] = json!([]),
            |v| v["bundleSizes"] = json!([2]),
        );
        refusal(
            "bundleSizes",
            |v| v["bundleSizes"] = json!([0, 2]),
            |v| v["bundleSizes"] = json!([1, 2]),
        );
    }

    #[test]
    fn bundles_per_arm_zero_is_refused() {
        refusal(
            "bundlesPerArm",
            |v| v["bundlesPerArm"] = json!(0),
            |v| v["bundlesPerArm"] = json!(1),
        );
    }

    #[test]
    fn pareto_candidates_below_bundles_per_arm_is_refused() {
        refusal(
            "paretoCandidates",
            |v| {
                v["bundlesPerArm"] = json!(6);
                v["paretoCandidates"] = json!(5);
            },
            |v| {
                v["bundlesPerArm"] = json!(6);
                v["paretoCandidates"] = json!(6);
            },
        );
    }

    #[test]
    fn archive_cap_zero_is_refused() {
        refusal(
            "archiveCap",
            |v| v["archiveCap"] = json!(0),
            |v| v["archiveCap"] = json!(1),
        );
    }

    #[test]
    fn latency_records_zero_is_refused() {
        refusal(
            "latencyRecords",
            |v| v["latencyRecords"] = json!(0),
            |v| v["latencyRecords"] = json!(1),
        );
    }

    #[test]
    fn negative_min_improvement_is_refused_and_zero_is_legal() {
        refusal(
            "minImprovement",
            |v| v["minImprovement"] = json!(-0.1),
            |v| v["minImprovement"] = json!(0.0),
        );
    }

    #[test]
    fn negative_regression_tolerance_is_refused() {
        refusal(
            "regressionTolerance",
            |v| v["regressionTolerance"] = json!(-1e-6),
            |v| v["regressionTolerance"] = json!(0.0),
        );
    }

    #[test]
    fn latency_tolerance_outside_zero_to_one_is_refused_by_name() {
        refusal(
            "latencyTolerance",
            |v| v["latencyTolerance"] = json!(1.0),
            |v| v["latencyTolerance"] = json!(0.0),
        );
        refusal(
            "latencyTolerance",
            |v| v["latencyTolerance"] = json!(-0.01),
            |v| v["latencyTolerance"] = json!(0.99),
        );
    }

    #[test]
    fn negative_max_fitness_loss_is_refused() {
        refusal(
            "maxFitnessLoss",
            |v| v["success"]["maxFitnessLoss"] = json!(-0.5),
            |v| v["success"]["maxFitnessLoss"] = json!(0.0),
        );
    }

    #[test]
    fn negative_min_regression_reduction_is_refused() {
        refusal(
            "minRegressionReduction",
            |v| v["success"]["minRegressionReduction"] = json!(-1),
            |v| v["success"]["minRegressionReduction"] = json!(0),
        );
    }

    #[test]
    fn unknown_field_is_refused_by_name() {
        let err = load_edited(|v| v["stratumWeights"] = json!("equal")).unwrap_err();
        assert!(err.contains("stratumWeights"), "{err}");
    }

    #[test]
    fn oversized_manifest_is_refused_before_it_is_read() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("manifest.json");
        // Not JSON at all: if the size check did not come first, the error
        // would be a parse error rather than the size limit.
        let body = "x".repeat(usize::try_from(MAX_MANIFEST_BYTES).unwrap() + 1);
        std::fs::write(&path, body).unwrap();
        let err = load(&path).unwrap_err();
        assert!(err.contains("1 MiB"), "{err}");
        assert!(
            !err.contains("expected"),
            "must not reach the parser: {err}"
        );
    }

    #[test]
    fn missing_manifest_names_its_path() {
        let err = load(Path::new("/nonexistent-issue-263/manifest.json")).unwrap_err();
        assert!(
            err.contains("/nonexistent-issue-263/manifest.json"),
            "{err}"
        );
    }
}
