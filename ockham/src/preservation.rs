//! Report-only preservation audit (Issue #264).
//!
//! Opt-in and report-only: the `audit-preservation` subcommand is the only
//! caller. It reads creatures and corpora, writes one JSON report, and never
//! writes a creature or changes acceptance. Nothing on the ordinary run path
//! reads it.
//!
//! Parent similarity never substitutes for authoritative fitness. A child is
//! "genuinely better" only when its `fullDelta` (from the full scorer) is
//! positive. The sample gate and the preservation score are compared against
//! that label; neither decides acceptance.
//!
//! Formulas, per split, over the first `R = min(maxRecords, records)` records
//! with `K` outputs (`p` parent, `c` child, `t` target, `p̄` parent mean):
//!
//! - `E_child = Σ(c − p)² / (RK)` and `E_mean = Σ(p̄ − p)² / (RK)`
//! - `E_shuf(s) = Σ(p[π_s(r)] − p)² / (RK)` for seeded permutations `π_s`
//! - `preservationVsMean = 1 − E_child / E_mean`
//! - `preservationVsShuffle = 1 − E_child / mean_s E_shuf(s)`
//! - `taskLoss = Σ(c − t)² / (RK)`, reported but never used to decide
//!
//! Degenerate handling: a control denominator below `1e-24` (including exactly
//! zero) gives a `null` score and a reason, `zeroVarianceParent` for the mean
//! control or `zeroShuffleError` for the shuffle control. No NaN or infinity is
//! emitted. `collapsedChild` is true when the parent varies and every child
//! output column is constant on the split. A non-finite child output excludes
//! that child from the comparison and is listed in `failures`.
//!
//! The threshold is fitted on calibration only, maximising balanced accuracy
//! with the smallest-θ tie-break. Validation is untouched until θ is fixed.

use std::collections::HashSet;
use std::ops::ControlFlow;
use std::path::{Path, PathBuf};
use std::time::Instant;

use neat_core::training_data::TrainingDataConfig;
use neat_core::{CreatureExport, compile_creature};
use serde::{Deserialize, Serialize};

use crate::corpus::{CorpusInfo, RecordRange, corpus_info, for_each_selected_chunk};
use crate::incumbent::load_incumbent;
use crate::ordering::SplitMix64;

/// Identifier written into every report.
pub const SCHEMA: &str = "ockham-preservation-audit/1";

const MAX_MANIFEST_BYTES: u64 = 1024 * 1024;
const MAX_CHILDREN: usize = 256;
const CHUNK_RECORDS: usize = 1024;
const DEGENERATE_EPS: f64 = 1e-24;
const ZERO_VARIANCE_PARENT: &str = "zeroVarianceParent";
const ZERO_SHUFFLE_ERROR: &str = "zeroShuffleError";
const EVIDENCE: &str = "fixture-or-unverified";
const REPORT_ONLY: &str = "report-only; not a demonstration of benefit";
const FORMULAS: &str = "E_child=sum((c-p)^2)/(RK); E_mean=sum((pbar-p)^2)/(RK); \
E_shuf(s)=sum((p[pi_s(r)]-p)^2)/(RK); preservationVsMean=1-E_child/E_mean; \
preservationVsShuffle=1-E_child/mean_s(E_shuf); taskLoss=sum((c-t)^2)/(RK); \
threshold fitted on calibration (balanced accuracy, smallest-theta tie-break); \
gates and AUC evaluated on validation; aucGain bootstrap over children; \
denominators below 1e-24 give null";
const SOURCE: &str = "arXiv:2611.13681 (abstract only; full method not retrieved)";

/// One child creature with its authoritative and sampled fitness changes.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct ChildSpec {
    name: String,
    creature: PathBuf,
    full_delta: f64,
    sample_delta: f64,
}

/// The audit manifest, validated by [`validate_manifest`].
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct Manifest {
    parent: PathBuf,
    children: Vec<ChildSpec>,
    calibration: PathBuf,
    validation: PathBuf,
    shuffle_seeds: Vec<u64>,
    #[serde(default)]
    bootstrap_seed: u64,
    #[serde(default = "default_resamples")]
    bootstrap_resamples: u64,
    #[serde(default = "default_max_records")]
    max_records: u64,
    #[serde(default)]
    sample_tolerance: f64,
    min_auc_gain: f64,
}

fn default_resamples() -> u64 {
    1000
}

fn default_max_records() -> u64 {
    4096
}

/// Parse a manifest from JSON text, rejecting unknown fields.
fn parse_manifest(text: &str) -> Result<Manifest, String> {
    serde_json::from_str(text).map_err(|e| format!("manifest: {e}"))
}

/// Check every refusal rule. Each message names the offending field.
fn validate_manifest(m: &Manifest, base: &Path) -> Result<(), String> {
    if !(1..=MAX_CHILDREN).contains(&m.children.len()) {
        return Err(format!(
            "children: expected 1 to {MAX_CHILDREN} entries, got {}",
            m.children.len()
        ));
    }
    let mut names = HashSet::new();
    for (i, c) in m.children.iter().enumerate() {
        if c.name.trim().is_empty() {
            return Err(format!("children[{i}].name must be non-empty"));
        }
        if !names.insert(c.name.as_str()) {
            return Err(format!("children: duplicate name '{}'", c.name));
        }
        if !c.full_delta.is_finite() {
            return Err(format!("children[{i}].fullDelta must be finite"));
        }
        if !c.sample_delta.is_finite() {
            return Err(format!("children[{i}].sampleDelta must be finite"));
        }
    }
    if m.shuffle_seeds.is_empty() {
        return Err("shuffleSeeds must be non-empty".into());
    }
    let mut seeds = HashSet::new();
    for s in &m.shuffle_seeds {
        if !seeds.insert(*s) {
            return Err(format!("shuffleSeeds: duplicate seed {s}"));
        }
    }
    if m.shuffle_seeds.contains(&m.bootstrap_seed) {
        return Err(format!(
            "bootstrapSeed {} must not appear in shuffleSeeds",
            m.bootstrap_seed
        ));
    }
    if !(100..=100_000).contains(&m.bootstrap_resamples) {
        return Err(format!(
            "bootstrapResamples must be within 100..=100000, got {}",
            m.bootstrap_resamples
        ));
    }
    if !(1..=1_000_000).contains(&m.max_records) {
        return Err(format!(
            "maxRecords must be within 1..=1000000, got {}",
            m.max_records
        ));
    }
    if !m.sample_tolerance.is_finite() || m.sample_tolerance < 0.0 {
        return Err("sampleTolerance must be finite and >= 0".into());
    }
    if !m.min_auc_gain.is_finite() {
        return Err("minAucGain must be finite".into());
    }
    let cal = resolve(base, &m.calibration);
    let val = resolve(base, &m.validation);
    if same_location(&cal, &val) {
        return Err(format!(
            "validation must differ from calibration: both resolve to '{}'",
            val.display()
        ));
    }
    Ok(())
}

/// Resolve a manifest path against the manifest's directory.
fn resolve(base: &Path, p: &Path) -> PathBuf {
    if p.is_absolute() {
        p.to_path_buf()
    } else {
        base.join(p)
    }
}

/// Whether two corpus paths name the same directory (canonical when possible).
fn same_location(a: &Path, b: &Path) -> bool {
    match (std::fs::canonicalize(a), std::fs::canonicalize(b)) {
        (Ok(x), Ok(y)) => x == y,
        _ => a == b,
    }
}

/// Directory that relative manifest paths are resolved against.
fn base_dir(manifest: &Path) -> PathBuf {
    match manifest.parent() {
        Some(p) if !p.as_os_str().is_empty() => p.to_path_buf(),
        _ => PathBuf::from("."),
    }
}

/// Read the manifest, refusing anything over 1 MiB before reading it.
fn read_manifest(path: &Path) -> Result<String, String> {
    let meta =
        std::fs::metadata(path).map_err(|e| format!("manifest '{}': {e}", path.display()))?;
    if meta.len() > MAX_MANIFEST_BYTES {
        return Err(format!(
            "manifest '{}' is {} bytes; the limit is {MAX_MANIFEST_BYTES}",
            path.display(),
            meta.len()
        ));
    }
    std::fs::read_to_string(path).map_err(|e| format!("manifest '{}': {e}", path.display()))
}

/// Squared Euclidean distance between two equal-length rows.
fn row_sq(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| {
            let d = x - y;
            d * d
        })
        .sum()
}

/// Row `r` of a row-major matrix that is `k` columns wide.
fn row(rows: &[f64], k: usize, r: usize) -> &[f64] {
    &rows[r * k..(r + 1) * k]
}

/// Mean squared error from per-record sums: `Σ per_record / (records × k)`.
fn mean_error(per_record: &[f64], k: usize) -> f64 {
    let n = (per_record.len() * k).max(1) as f64;
    per_record.iter().sum::<f64>() / n
}

/// `1 − num / den`, or `None` when `den` is below the degenerate epsilon.
fn ratio_score(num: f64, den: f64) -> Option<f64> {
    if den.is_nan() || den < DEGENERATE_EPS {
        return None;
    }
    let score = 1.0 - num / den;
    score.is_finite().then_some(score)
}

/// Squared error between outputs and targets for one record (`f64` accumulate).
fn task_sq(outputs: &[f32], targets: &[f32]) -> f64 {
    outputs
        .iter()
        .zip(targets)
        .map(|(&c, &t)| {
            let d = f64::from(c) - f64::from(t);
            d * d
        })
        .sum()
}

/// Per-column min and max, so "zero variance" is an exact test.
#[derive(Debug, Clone)]
struct ColumnRange {
    min: Vec<f64>,
    max: Vec<f64>,
}

impl ColumnRange {
    fn new(k: usize) -> Self {
        Self {
            min: vec![f64::INFINITY; k],
            max: vec![f64::NEG_INFINITY; k],
        }
    }

    fn push(&mut self, values: &[f64]) {
        for (j, &v) in values.iter().enumerate() {
            self.min[j] = self.min[j].min(v);
            self.max[j] = self.max[j].max(v);
        }
    }

    /// True when at least one column takes two different values.
    fn varies(&self) -> bool {
        self.min.iter().zip(&self.max).any(|(lo, hi)| lo != hi)
    }
}

/// Timing and volume of one streamed activation pass.
#[derive(Debug, Clone, Copy)]
struct PassStats {
    activated: u64,
    elapsed_ms: f64,
}

/// Milliseconds per 1000 activated records, indicative only.
fn per_1k(stats: PassStats) -> f64 {
    if stats.activated == 0 {
        0.0
    } else {
        stats.elapsed_ms * 1000.0 / stats.activated as f64
    }
}

/// Activate `creature` over records `0..records` of `dir`.
///
/// `visit` receives the global index, outputs and targets of each record and
/// may return `Break` to stop early (a non-finite output).
fn stream_outputs<F>(
    creature: &CreatureExport,
    dir: &Path,
    records: u64,
    mut visit: F,
) -> Result<PassStats, String>
where
    F: FnMut(u64, &[f32], &[f32]) -> ControlFlow<()>,
{
    let mut net =
        compile_creature(creature).map_err(|e| format!("cannot compile creature: {e}"))?;
    let cfg = TrainingDataConfig::new(creature.input, creature.output);
    let (width, k) = (creature.input, creature.output);
    let ranges = [RecordRange {
        start: 0,
        len: records,
    }];
    let started = Instant::now();
    let mut activated = 0u64;
    let mut broke = false;
    for_each_selected_chunk(dir, &cfg, &ranges, CHUNK_RECORDS, |chunk| {
        for r in 0..chunk.records {
            let inputs = &chunk.inputs[r * width..(r + 1) * width];
            let targets = &chunk.targets[r * k..(r + 1) * k];
            let outputs = net.activate(inputs, k);
            activated += 1;
            if visit(chunk.first_index + r as u64, &outputs, targets).is_break() {
                broke = true;
                return Ok(ControlFlow::Break(()));
            }
        }
        Ok(ControlFlow::Continue(()))
    })?;
    if !broke && activated != records {
        return Err(format!(
            "corpus '{}' yielded {activated} of {records} records",
            dir.display()
        ));
    }
    Ok(PassStats {
        activated,
        elapsed_ms: started.elapsed().as_secs_f64() * 1000.0,
    })
}

/// The parent's outputs over one split.
struct ParentPass {
    rows: Vec<f64>,
    columns: ColumnRange,
    task_sum: f64,
    stats: PassStats,
}

fn parent_pass(
    parent: &CreatureExport,
    dir: &Path,
    records: u64,
    split: &str,
) -> Result<ParentPass, String> {
    let k = parent.output;
    let mut rows = Vec::with_capacity(records as usize * k);
    let mut columns = ColumnRange::new(k);
    let mut task_sum = 0.0;
    let mut bad: Option<u64> = None;
    let stats = stream_outputs(parent, dir, records, |index, out, targets| {
        if out.iter().any(|v| !v.is_finite()) {
            bad = Some(index);
            return ControlFlow::Break(());
        }
        let start = rows.len();
        rows.extend(out.iter().map(|&v| f64::from(v)));
        columns.push(&rows[start..]);
        task_sum += task_sq(out, targets);
        ControlFlow::Continue(())
    })?;
    if let Some(index) = bad {
        return Err(format!(
            "parent creature: nonFiniteOutput at record {index} ({split} split)"
        ));
    }
    Ok(ParentPass {
        rows,
        columns,
        task_sum,
        stats,
    })
}

/// Parent-only controls for one split.
struct Controls {
    mean_rec: Vec<f64>,
    mean_error: f64,
    shuffle_errors: Vec<f64>,
    shuffle_mean: f64,
}

/// Column means of a row-major matrix over `records` rows.
fn column_mean(rows: &[f64], records: usize, k: usize) -> Vec<f64> {
    let mut mean = vec![0.0; k];
    for p in rows.chunks_exact(k) {
        for (m, v) in mean.iter_mut().zip(p) {
            *m += v;
        }
    }
    for m in &mut mean {
        *m /= records as f64;
    }
    mean
}

/// Mean-control and shuffle-control errors for the parent on one split.
fn controls(parent_rows: &[f64], records: usize, k: usize, seeds: &[u64]) -> Controls {
    let mean = column_mean(parent_rows, records, k);
    let mean_rec: Vec<f64> = parent_rows
        .chunks_exact(k)
        .map(|p| row_sq(&mean, p))
        .collect();
    let shuffle_errors: Vec<f64> = seeds
        .iter()
        .map(|&seed| {
            let mut perm: Vec<usize> = (0..records).collect();
            SplitMix64(seed).shuffle(&mut perm);
            let rec: Vec<f64> = (0..records)
                .map(|r| row_sq(row(parent_rows, k, perm[r]), row(parent_rows, k, r)))
                .collect();
            mean_error(&rec, k)
        })
        .collect();
    let shuffle_mean = shuffle_errors.iter().sum::<f64>() / shuffle_errors.len() as f64;
    Controls {
        mean_error: mean_error(&mean_rec, k),
        mean_rec,
        shuffle_errors,
        shuffle_mean,
    }
}

/// One child's outputs over one split, streamed against the parent rows.
struct ChildPass {
    child_rec: Vec<f64>,
    columns: ColumnRange,
    task_sum: f64,
    failure: Option<String>,
    stats: PassStats,
}

fn child_pass(
    child: &CreatureExport,
    dir: &Path,
    records: u64,
    k: usize,
    parent_rows: &[f64],
) -> Result<ChildPass, String> {
    let mut child_rec = Vec::with_capacity(records as usize);
    let mut columns = ColumnRange::new(k);
    let mut task_sum = 0.0;
    let mut failure: Option<String> = None;
    let mut buf: Vec<f64> = Vec::with_capacity(k);
    let stats = stream_outputs(child, dir, records, |index, out, targets| {
        if out.iter().any(|v| !v.is_finite()) {
            failure = Some(format!("nonFiniteOutput at record {index}"));
            return ControlFlow::Break(());
        }
        buf.clear();
        buf.extend(out.iter().map(|&v| f64::from(v)));
        child_rec.push(row_sq(&buf, row(parent_rows, k, index as usize)));
        columns.push(&buf);
        task_sum += task_sq(out, targets);
        ControlFlow::Continue(())
    })?;
    Ok(ChildPass {
        child_rec,
        columns,
        task_sum,
        failure,
        stats,
    })
}

/// Reasons a child's preservation scores are `null`.
fn degenerate_reasons(vs_mean: Option<f64>, vs_shuffle: Option<f64>) -> Vec<&'static str> {
    let mut reasons = Vec::new();
    if vs_mean.is_none() {
        reasons.push(ZERO_VARIANCE_PARENT);
    }
    if vs_shuffle.is_none() {
        reasons.push(ZERO_SHUFFLE_ERROR);
    }
    reasons
}

/// Percentile 2.5 and 97.5 of `values` (nearest rank), or `None` if empty.
fn percentile_interval(mut values: Vec<f64>) -> Option<[f64; 2]> {
    if values.is_empty() {
        return None;
    }
    values.sort_by(f64::total_cmp);
    let at = |q: f64| values[((values.len() - 1) as f64 * q).round() as usize];
    Some([at(0.025), at(0.975)])
}

/// Bootstrap 95% CI of `preservationVsMean`, resampling records.
///
/// `child` and `mean` are per-record sums of squares over the same records.
fn bootstrap_ci(
    child: &[f64],
    mean: &[f64],
    k: usize,
    resamples: usize,
    seed: u64,
) -> Option<[f64; 2]> {
    let r = child.len();
    if r == 0 {
        return None;
    }
    let denom = (r * k) as f64;
    let mut rng = SplitMix64(seed);
    let mut values = Vec::with_capacity(resamples);
    for _ in 0..resamples {
        let (mut c, mut m) = (0.0, 0.0);
        for _ in 0..r {
            let i = (rng.next() % r as u64) as usize;
            c += child[i];
            m += mean[i];
        }
        if let Some(v) = ratio_score(c / denom, m / denom) {
            values.push(v);
        }
    }
    percentile_interval(values)
}

/// Mann–Whitney AUC for predicting `positives` from `scores` (ties count 0.5).
///
/// `None` when there are no positives or no negatives.
fn auc(scores: &[f64], positives: &[bool]) -> Option<f64> {
    let n_pos = positives.iter().filter(|&&p| p).count();
    let n_neg = positives.len() - n_pos;
    if n_pos == 0 || n_neg == 0 {
        return None;
    }
    let mut order: Vec<usize> = (0..scores.len()).collect();
    order.sort_by(|&a, &b| scores[a].total_cmp(&scores[b]));
    let mut rank_sum = 0.0;
    let mut start = 0;
    while start < order.len() {
        let value = scores[order[start]];
        let mut end = start;
        while end + 1 < order.len() && scores[order[end + 1]] == value {
            end += 1;
        }
        let mid_rank = (start + end) as f64 / 2.0 + 1.0;
        let positives_here = order[start..=end].iter().filter(|&&i| positives[i]).count();
        rank_sum += positives_here as f64 * mid_rank;
        start = end + 1;
    }
    let (p, n) = (n_pos as f64, n_neg as f64);
    Some((rank_sum - p * (p + 1.0) / 2.0) / (p * n))
}

/// Bootstrap 95% CI of `AUC_pres − AUC_sample`, resampling children.
fn bootstrap_auc_gain(
    pres: &[f64],
    sample: &[f64],
    better: &[bool],
    resamples: usize,
    seed: u64,
) -> Option<[f64; 2]> {
    let n = pres.len();
    if n == 0 {
        return None;
    }
    let mut rng = SplitMix64(seed);
    let mut gains = Vec::with_capacity(resamples);
    let (mut p, mut s, mut b) = (
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
    );
    for _ in 0..resamples {
        p.clear();
        s.clear();
        b.clear();
        for _ in 0..n {
            let i = (rng.next() % n as u64) as usize;
            p.push(pres[i]);
            s.push(sample[i]);
            b.push(better[i]);
        }
        if let (Some(a_p), Some(a_s)) = (auc(&p, &b), auc(&s, &b)) {
            gains.push(a_p - a_s);
        }
    }
    percentile_interval(gains)
}

/// Threshold maximising balanced accuracy; accept iff `score >= θ`.
///
/// Positives are `better` children. Ties go to the smallest θ. `None` when
/// either class is absent.
fn fit_threshold(scores: &[f64], better: &[bool]) -> Option<f64> {
    let n_pos = better.iter().filter(|&&b| b).count() as u128;
    let n_neg = better.len() as u128 - n_pos;
    if n_pos == 0 || n_neg == 0 {
        return None;
    }
    let mut candidates = scores.to_vec();
    candidates.sort_by(f64::total_cmp);
    candidates.dedup();
    let mut best: Option<(u128, f64)> = None;
    for &theta in &candidates {
        let (mut tp, mut tn) = (0u128, 0u128);
        for (&s, &b) in scores.iter().zip(better) {
            if b && s >= theta {
                tp += 1;
            } else if !b && s < theta {
                tn += 1;
            }
        }
        // Twice the balanced accuracy times P·N, kept integral for exact ties.
        let balanced = tp * n_neg + tn * n_pos;
        if best.is_none_or(|(b, _)| balanced > b) {
            best = Some((balanced, theta));
        }
    }
    best.map(|(_, theta)| theta)
}

/// Fit θ on calibration scores, explaining any refusal.
fn fit_threshold_checked(scores: &[Option<f64>], better: &[bool]) -> Result<f64, String> {
    let (s, b): (Vec<f64>, Vec<bool>) = scores
        .iter()
        .zip(better)
        .filter_map(|(score, &is_better)| score.map(|v| (v, is_better)))
        .unzip();
    if s.is_empty() {
        return Err(
            "threshold not fittable: no calibration child has a non-null preservationVsMean".into(),
        );
    }
    if !b.iter().any(|&x| x) {
        return Err("threshold not fittable: calibration has no better child".into());
    }
    if b.iter().all(|&x| x) {
        return Err("threshold not fittable: calibration has no worse child".into());
    }
    fit_threshold(&s, &b).ok_or_else(|| "threshold not fittable: no candidate threshold".into())
}

/// A child creature loaded and width-checked against the parent.
struct Child {
    name: String,
    sha256: String,
    creature: CreatureExport,
    full_delta: f64,
    sample_delta: f64,
}

fn load_children(
    specs: &[ChildSpec],
    base: &Path,
    parent: &CreatureExport,
) -> Result<Vec<Child>, String> {
    specs
        .iter()
        .map(|spec| {
            let path = resolve(base, &spec.creature);
            let inc = load_incumbent(&path)
                .map_err(|e| format!("child '{}' ('{}'): {e}", spec.name, path.display()))?;
            if inc.creature.input != parent.input || inc.creature.output != parent.output {
                return Err(format!(
                    "child '{}': widths {}x{} (input x output) differ from parent {}x{}",
                    spec.name, inc.creature.input, inc.creature.output, parent.input, parent.output
                ));
            }
            Ok(Child {
                name: spec.name.clone(),
                sha256: inc.checksum,
                creature: inc.creature,
                full_delta: spec.full_delta,
                sample_delta: spec.sample_delta,
            })
        })
        .collect()
}

/// Settings shared by every split.
struct Settings {
    shuffle_seeds: Vec<u64>,
    bootstrap_seed: u64,
    resamples: usize,
}

/// One split to score.
struct Split<'a> {
    name: &'static str,
    dir: &'a Path,
    records: u64,
    corpus_records: u64,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct ParentReport {
    task_loss: f64,
    latency_ms_per_1k_records: f64,
    mean_control_error: f64,
    shuffle_control_errors: Vec<f64>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct ChildReport {
    name: String,
    failure: Option<String>,
    output_error: Option<f64>,
    preservation_vs_mean: Option<f64>,
    preservation_vs_shuffle: Option<f64>,
    ci95: Option<[f64; 2]>,
    collapsed_child: bool,
    degenerate: Vec<&'static str>,
    task_loss: Option<f64>,
    latency_ms_per_1k_records: Option<f64>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct SplitReport {
    records: u64,
    corpus_records: u64,
    parent: ParentReport,
    children: Vec<ChildReport>,
}

/// Everything one split produced.
struct SplitRun {
    report: SplitReport,
    pres: Vec<Option<f64>>,
    failures: Vec<Option<String>>,
    activated: u64,
    elapsed_ms: f64,
}

/// Child report for a failed child: every score is absent.
fn failed_child(name: &str, failure: String) -> ChildReport {
    ChildReport {
        name: name.into(),
        failure: Some(failure),
        output_error: None,
        preservation_vs_mean: None,
        preservation_vs_shuffle: None,
        ci95: None,
        collapsed_child: false,
        degenerate: Vec::new(),
        task_loss: None,
        latency_ms_per_1k_records: None,
    }
}

/// Score one child's pass against the parent controls.
fn score_child(
    name: &str,
    cp: ChildPass,
    ctrl: &Controls,
    parent_varies: bool,
    shape: (usize, usize),
    settings: &Settings,
) -> (ChildReport, Option<f64>) {
    let (records, k) = shape;
    if let Some(failure) = cp.failure {
        return (failed_child(name, failure), None);
    }
    let error = mean_error(&cp.child_rec, k);
    let vs_mean = ratio_score(error, ctrl.mean_error);
    let vs_shuffle = ratio_score(error, ctrl.shuffle_mean);
    let ci95 = vs_mean.and_then(|_| {
        bootstrap_ci(
            &cp.child_rec,
            &ctrl.mean_rec,
            k,
            settings.resamples,
            settings.bootstrap_seed,
        )
    });
    let report = ChildReport {
        name: name.into(),
        failure: None,
        output_error: Some(error),
        preservation_vs_mean: vs_mean,
        preservation_vs_shuffle: vs_shuffle,
        ci95,
        collapsed_child: parent_varies && !cp.columns.varies(),
        degenerate: degenerate_reasons(vs_mean, vs_shuffle),
        task_loss: Some(cp.task_sum / (records * k) as f64),
        latency_ms_per_1k_records: Some(per_1k(cp.stats)),
    };
    (report, vs_mean)
}

/// Score the parent and every child on one split.
fn run_split(
    split: Split<'_>,
    parent: &CreatureExport,
    children: &[Child],
    settings: &Settings,
) -> Result<SplitRun, String> {
    let k = parent.output;
    let records = split.records as usize;
    let pp = parent_pass(parent, split.dir, split.records, split.name)?;
    let ctrl = controls(&pp.rows, records, k, &settings.shuffle_seeds);
    let parent_varies = pp.columns.varies();
    let mut activated = pp.stats.activated;
    let mut elapsed_ms = pp.stats.elapsed_ms;
    let mut child_reports = Vec::with_capacity(children.len());
    let mut pres = Vec::with_capacity(children.len());
    let mut failures = Vec::with_capacity(children.len());
    for child in children {
        let cp = child_pass(&child.creature, split.dir, split.records, k, &pp.rows)?;
        activated += cp.stats.activated;
        elapsed_ms += cp.stats.elapsed_ms;
        failures.push(cp.failure.clone());
        let (report, score) = score_child(
            &child.name,
            cp,
            &ctrl,
            parent_varies,
            (records, k),
            settings,
        );
        child_reports.push(report);
        pres.push(score);
    }
    let cells = (records * k) as f64;
    let report = SplitReport {
        records: split.records,
        corpus_records: split.corpus_records,
        parent: ParentReport {
            task_loss: pp.task_sum / cells,
            latency_ms_per_1k_records: per_1k(pp.stats),
            mean_control_error: ctrl.mean_error,
            shuffle_control_errors: ctrl.shuffle_errors,
        },
        children: child_reports,
    };
    Ok(SplitRun {
        report,
        pres,
        failures,
        activated,
        elapsed_ms,
    })
}

/// Method description written into the report.
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct Method {
    formulas: &'static str,
    source: &'static str,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct NamedDigest {
    name: String,
    sha256: String,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct Versions {
    ockham: &'static str,
    parent_sha256: String,
    children: Vec<NamedDigest>,
    calibration_corpus: String,
    validation_corpus: String,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct GateReport {
    rejected: Option<u64>,
    false_rejections: Option<u64>,
    false_rejection_rate: Option<f64>,
    auc: Option<f64>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct Gates {
    sample: GateReport,
    preservation: GateReport,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct PairedRow {
    name: String,
    better: bool,
    sample_rejects: bool,
    preservation_rejects: Option<bool>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct AucGain {
    value: Option<f64>,
    ci95: Option<[f64; 2]>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct Criterion {
    status: &'static str,
    rule: String,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct Coverage {
    children_supplied: usize,
    children_scored: usize,
    validation_records: u64,
    validation_corpus_records: u64,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct Costs {
    calibration_ms: u64,
    validation_ms: u64,
    records_activated: u64,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct Splits {
    calibration: SplitReport,
    validation: SplitReport,
}

/// The full report written to disk.
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct Report {
    schema: &'static str,
    evidence: &'static str,
    report_only: &'static str,
    method: Method,
    versions: Versions,
    manifest: serde_json::Value,
    threshold: Option<f64>,
    splits: Splits,
    gates: Gates,
    paired: Vec<PairedRow>,
    auc_gain: AucGain,
    criterion: Criterion,
    coverage: Coverage,
    costs: Costs,
    failures: Vec<String>,
}

/// Gate counts and AUC over the comparison children.
///
/// `fitted` is false when θ is unavailable: counts are then `null` but the
/// AUC, which does not depend on θ, is still reported.
fn gate_report(
    rejects: &[Option<bool>],
    fitted: bool,
    better: &[bool],
    scores: &[Option<f64>],
) -> GateReport {
    let rejected =
        fitted.then(|| rejects.iter().filter(|r| matches!(r, Some(true))).count() as u64);
    let false_rejections = fitted.then(|| {
        rejects
            .iter()
            .zip(better)
            .filter(|&(r, &b)| matches!(r, Some(true)) && b)
            .count() as u64
    });
    let betters = better.iter().filter(|&&b| b).count();
    let false_rejection_rate =
        false_rejections.and_then(|f| (betters > 0).then(|| f as f64 / betters as f64));
    let all_scores: Option<Vec<f64>> = scores.iter().copied().collect();
    GateReport {
        rejected,
        false_rejections,
        false_rejection_rate,
        auc: all_scores.and_then(|s| auc(&s, better)),
    }
}

/// Threshold, gates, pairs and criterion, computed from both splits.
struct Evaluation {
    threshold: Option<f64>,
    threshold_failure: Option<String>,
    gates: Gates,
    paired: Vec<PairedRow>,
    auc_gain: AucGain,
    criterion: Criterion,
    scored: usize,
}

fn evaluate(children: &[Child], cal: &SplitRun, val: &SplitRun, m: &Manifest) -> Evaluation {
    // A child takes part only when it scored on both splits.
    let comparison: Vec<usize> = (0..children.len())
        .filter(|&i| cal.failures[i].is_none() && val.failures[i].is_none())
        .collect();
    let better: Vec<bool> = comparison
        .iter()
        .map(|&i| children[i].full_delta > 0.0)
        .collect();
    let cal_pres: Vec<Option<f64>> = comparison.iter().map(|&i| cal.pres[i]).collect();
    let (threshold, threshold_failure) = match fit_threshold_checked(&cal_pres, &better) {
        Ok(t) => (Some(t), None),
        Err(why) => (None, Some(why)),
    };

    let sample_rej: Vec<bool> = comparison
        .iter()
        .map(|&i| children[i].sample_delta < -m.sample_tolerance)
        .collect();
    let sample_rejects: Vec<Option<bool>> = sample_rej.iter().map(|&r| Some(r)).collect();
    let sample_scores: Vec<Option<f64>> = comparison
        .iter()
        .map(|&i| Some(children[i].sample_delta))
        .collect();
    let val_pres: Vec<Option<f64>> = comparison.iter().map(|&i| val.pres[i]).collect();
    let pres_rejects: Vec<Option<bool>> = val_pres
        .iter()
        .map(|s| threshold.and_then(|t| s.map(|v| v < t)))
        .collect();

    let sample_gate = gate_report(&sample_rejects, true, &better, &sample_scores);
    let pres_gate = gate_report(&pres_rejects, threshold.is_some(), &better, &val_pres);

    let paired: Vec<PairedRow> = comparison
        .iter()
        .enumerate()
        .map(|(j, &i)| PairedRow {
            name: children[i].name.clone(),
            better: better[j],
            sample_rejects: sample_rej[j],
            preservation_rejects: pres_rejects[j],
        })
        .collect();

    let pres_all: Option<Vec<f64>> = val_pres.iter().copied().collect();
    let sample_all: Vec<f64> = comparison
        .iter()
        .map(|&i| children[i].sample_delta)
        .collect();
    let value = match (pres_gate.auc, sample_gate.auc) {
        (Some(a), Some(b)) => Some(a - b),
        _ => None,
    };
    let ci95 = match (&pres_all, value) {
        (Some(p), Some(_)) => bootstrap_auc_gain(
            p,
            &sample_all,
            &better,
            m.bootstrap_resamples as usize,
            m.bootstrap_seed,
        ),
        _ => None,
    };

    let status = match (
        value,
        pres_gate.false_rejection_rate,
        sample_gate.false_rejection_rate,
    ) {
        (Some(gain), Some(pres_rate), Some(sample_rate)) => {
            if gain >= m.min_auc_gain && pres_rate <= sample_rate {
                "met"
            } else {
                "notMet"
            }
        }
        _ => "notEvaluable",
    };
    let rule = format!(
        "met iff aucGain.value >= minAucGain ({}) AND preservation.falseRejectionRate <= \
         sample.falseRejectionRate (predeclared); notEvaluable when any input is null; \
         report-only: similarity never substitutes for authoritative fitness and success \
         needs representative runs",
        m.min_auc_gain
    );

    Evaluation {
        threshold,
        threshold_failure,
        gates: Gates {
            sample: sample_gate,
            preservation: pres_gate,
        },
        paired,
        auc_gain: AucGain { value, ci95 },
        criterion: Criterion { status, rule },
        scored: comparison.len(),
    }
}

/// Records to use for a split: `min(maxRecords, corpus records)`, never zero.
fn split_records(name: &str, max: u64, info: &CorpusInfo) -> Result<u64, String> {
    let records = max.min(info.record_count);
    if records == 0 {
        return Err(format!("{name} corpus has no records"));
    }
    Ok(records)
}

/// Write the report to `<out>.tmp`, then rename it over `out`.
fn write_atomically(out: &Path, value: &serde_json::Value) -> Result<(), String> {
    let text =
        serde_json::to_string_pretty(value).map_err(|e| format!("cannot encode report: {e}"))?;
    let mut tmp_name = out.as_os_str().to_owned();
    tmp_name.push(".tmp");
    let tmp = PathBuf::from(tmp_name);
    std::fs::write(&tmp, format!("{text}\n"))
        .map_err(|e| format!("cannot write '{}': {e}", tmp.display()))?;
    std::fs::rename(&tmp, out).map_err(|e| {
        format!(
            "cannot rename '{}' to '{}': {e}",
            tmp.display(),
            out.display()
        )
    })
}

/// Run the preservation audit described by `manifest_path` and write the
/// report JSON to `out_path`.
///
/// Relative manifest paths resolve against the manifest's directory. The
/// report is written atomically and the same value is returned. Report-only:
/// this never writes a creature and never changes acceptance.
pub fn run_audit(manifest_path: &Path, out_path: &Path) -> Result<serde_json::Value, String> {
    let text = read_manifest(manifest_path)?;
    let manifest = parse_manifest(&text)?;
    let echo: serde_json::Value =
        serde_json::from_str(&text).map_err(|e| format!("manifest: {e}"))?;
    let base = base_dir(manifest_path);
    validate_manifest(&manifest, &base)?;

    let parent_path = resolve(&base, &manifest.parent);
    let parent = load_incumbent(&parent_path)
        .map_err(|e| format!("parent '{}': {e}", parent_path.display()))?;
    let children = load_children(&manifest.children, &base, &parent.creature)?;

    let cfg = TrainingDataConfig::new(parent.creature.input, parent.creature.output);
    let cal_dir = resolve(&base, &manifest.calibration);
    let val_dir = resolve(&base, &manifest.validation);
    let cal_info = corpus_info(&cal_dir, &cfg).map_err(|e| format!("calibration corpus: {e}"))?;
    let val_info = corpus_info(&val_dir, &cfg).map_err(|e| format!("validation corpus: {e}"))?;
    let cal_records = split_records("calibration", manifest.max_records, &cal_info)?;
    let val_records = split_records("validation", manifest.max_records, &val_info)?;

    let settings = Settings {
        shuffle_seeds: manifest.shuffle_seeds.clone(),
        bootstrap_seed: manifest.bootstrap_seed,
        resamples: manifest.bootstrap_resamples as usize,
    };
    let cal = run_split(
        Split {
            name: "calibration",
            dir: &cal_dir,
            records: cal_records,
            corpus_records: cal_info.record_count,
        },
        &parent.creature,
        &children,
        &settings,
    )?;
    let val = run_split(
        Split {
            name: "validation",
            dir: &val_dir,
            records: val_records,
            corpus_records: val_info.record_count,
        },
        &parent.creature,
        &children,
        &settings,
    )?;

    let eval = evaluate(&children, &cal, &val, &manifest);
    let mut failures = Vec::new();
    for (split, run) in [("calibration", &cal), ("validation", &val)] {
        for (child, failure) in children.iter().zip(&run.failures) {
            if let Some(f) = failure {
                failures.push(format!("{split} child '{}': {f}", child.name));
            }
        }
    }
    if let Some(why) = eval.threshold_failure.clone() {
        failures.push(why);
    }

    let report = Report {
        schema: SCHEMA,
        evidence: EVIDENCE,
        report_only: REPORT_ONLY,
        method: Method {
            formulas: FORMULAS,
            source: SOURCE,
        },
        versions: Versions {
            ockham: env!("CARGO_PKG_VERSION"),
            parent_sha256: parent.checksum.clone(),
            children: children
                .iter()
                .map(|c| NamedDigest {
                    name: c.name.clone(),
                    sha256: c.sha256.clone(),
                })
                .collect(),
            calibration_corpus: cal_info.identity.clone(),
            validation_corpus: val_info.identity.clone(),
        },
        manifest: echo,
        threshold: eval.threshold,
        splits: Splits {
            calibration: cal.report,
            validation: val.report,
        },
        gates: eval.gates,
        paired: eval.paired,
        auc_gain: eval.auc_gain,
        criterion: eval.criterion,
        coverage: Coverage {
            children_supplied: children.len(),
            children_scored: eval.scored,
            validation_records: val_records,
            validation_corpus_records: val_info.record_count,
        },
        costs: Costs {
            calibration_ms: cal.elapsed_ms.round() as u64,
            validation_ms: val.elapsed_ms.round() as u64,
            records_activated: cal.activated + val.activated,
        },
        failures,
    };
    let value = serde_json::to_value(&report).map_err(|e| format!("cannot encode report: {e}"))?;
    write_atomically(out_path, &value)?;
    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::corpus::write_bin_file;
    use crate::fixtures::identity_creature;
    use neat_core::creature_to_json_pretty;
    use serde_json::json;

    fn base_manifest() -> serde_json::Value {
        json!({
            "parent": "parent.json",
            "children": [
                {"name": "c1", "creature": "c1.json", "fullDelta": 0.01, "sampleDelta": -0.002},
                {"name": "c2", "creature": "c2.json", "fullDelta": -0.02, "sampleDelta": 0.001}
            ],
            "calibration": "calib",
            "validation": "valid",
            "shuffleSeeds": [1, 2, 3],
            "bootstrapSeed": 7,
            "minAucGain": 0.05
        })
    }

    fn check(v: &serde_json::Value) -> Result<(), String> {
        let text = v.to_string();
        let m = parse_manifest(&text)?;
        validate_manifest(&m, Path::new("."))
    }

    #[test]
    fn ratio_score_identical_and_degenerate() {
        assert_eq!(ratio_score(0.0, 2.0), Some(1.0));
        assert_eq!(ratio_score(0.5, 2.0), Some(0.75));
        assert_eq!(ratio_score(1.0, 0.0), None);
        assert_eq!(ratio_score(1.0, 1e-25), None);
        assert_eq!(ratio_score(1.0, f64::NAN), None);
    }

    #[test]
    fn constant_child_at_parent_mean_scores_zero_and_constant_parent_is_degenerate() {
        let k = 2;
        let parent = vec![1.0, 2.0, 3.0, 0.5, 2.0, 4.0, 0.0, 1.0, 5.0, 3.0];
        let ctrl = controls(&parent, 5, k, &[1]);
        // A child that outputs the parent mean on every record.
        let mean = column_mean(&parent, 5, k);
        let child_rec: Vec<f64> = parent.chunks_exact(k).map(|p| row_sq(&mean, p)).collect();
        let e_child = mean_error(&child_rec, k);
        assert_eq!(ratio_score(e_child, ctrl.mean_error), Some(0.0));

        let mut cols = ColumnRange::new(k);
        for _ in 0..5 {
            cols.push(&mean);
        }
        assert!(!cols.varies());
        let mut parent_cols = ColumnRange::new(k);
        for p in parent.chunks_exact(k) {
            parent_cols.push(p);
        }
        assert!(parent_cols.varies());

        let flat = vec![2.0; 10];
        let flat_ctrl = controls(&flat, 5, k, &[1]);
        assert_eq!(flat_ctrl.mean_error, 0.0);
        let vs_mean = ratio_score(0.3, flat_ctrl.mean_error);
        let vs_shuffle = ratio_score(0.3, flat_ctrl.shuffle_mean);
        assert_eq!(vs_mean, None);
        assert_eq!(
            degenerate_reasons(vs_mean, vs_shuffle),
            vec!["zeroVarianceParent", "zeroShuffleError"]
        );
    }

    #[test]
    fn auc_separation_reversal_ties_and_no_positives() {
        let pos = [false, false, true, true];
        assert_eq!(auc(&[0.1, 0.2, 0.8, 0.9], &pos), Some(1.0));
        assert_eq!(auc(&[0.9, 0.8, 0.2, 0.1], &pos), Some(0.0));
        assert_eq!(auc(&[0.5; 4], &[true, false, true, false]), Some(0.5));
        assert_eq!(auc(&[0.1, 0.2, 0.3], &[false, false, false]), None);
    }

    #[test]
    fn fit_threshold_maximises_balanced_accuracy_with_smallest_tie() {
        // Hand fixture: θ = 0.4 gives TPR 1.0 and TNR 0.5 (BA 0.75), the best.
        let scores = [0.9, 0.7, 0.6, 0.2, 0.4];
        let better = [true, false, true, false, true];
        assert_eq!(fit_threshold(&scores, &better), Some(0.4));
        // θ = 0.1 and θ = 0.2 both give BA 0.5; the smaller wins.
        assert_eq!(fit_threshold(&[0.1, 0.2], &[true, false]), Some(0.1));
        assert_eq!(fit_threshold(&[0.1, 0.2], &[true, true]), None);
        assert_eq!(fit_threshold(&[0.1, 0.2], &[false, false]), None);
    }

    #[test]
    fn bootstrap_ci_is_reproducible_and_brackets_the_point() {
        let k = 1;
        let mean = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let child = [0.5, 3.0, 2.0, 4.5, 1.0, 6.5, 6.0, 7.5];
        let first = bootstrap_ci(&child, &mean, k, 1000, 7);
        assert!(first.is_some());
        assert_eq!(first, bootstrap_ci(&child, &mean, k, 1000, 7));
        let point = ratio_score(mean_error(&child, k), mean_error(&mean, k)).unwrap();
        let [lo, hi] = first.unwrap();
        assert!(lo <= point && point <= hi, "{lo} <= {point} <= {hi}");
    }

    #[test]
    fn manifest_refusals_name_their_field() {
        let mut v = base_manifest();
        v["shuffleSeeds"] = json!([]);
        assert!(check(&v).unwrap_err().contains("shuffleSeeds"));

        let mut v = base_manifest();
        v["children"][1]["name"] = json!("c1");
        assert!(check(&v).unwrap_err().contains("children"));

        let mut v = base_manifest();
        v["validation"] = json!("calib");
        assert!(check(&v).unwrap_err().contains("validation"));

        let mut v = base_manifest();
        v["bootstrapSeed"] = json!(2);
        assert!(check(&v).unwrap_err().contains("bootstrapSeed"));

        let mut v = base_manifest();
        v["bootstrapResamples"] = json!(50);
        assert!(check(&v).unwrap_err().contains("bootstrapResamples"));

        let mut v = base_manifest();
        v["bogus"] = json!(1);
        assert!(check(&v).unwrap_err().contains("bogus"));
    }

    #[test]
    fn valid_manifest_parses_and_defaults_apply() {
        let v = base_manifest();
        assert_eq!(check(&v), Ok(()));
        let m = parse_manifest(&v.to_string()).unwrap();
        assert_eq!(m.bootstrap_resamples, 1000);
        assert_eq!(m.max_records, 4096);
        assert_eq!(m.sample_tolerance, 0.0);
    }

    fn write_creature(path: &Path, creature: &CreatureExport) {
        std::fs::write(path, creature_to_json_pretty(creature).unwrap()).unwrap();
    }

    /// Records with varied inputs; the target is `0.5 * x0 + 0.1`.
    fn write_corpus(dir: &Path, n: usize, phase: usize) {
        std::fs::create_dir_all(dir).unwrap();
        let records: Vec<(Vec<f32>, Vec<f32>)> = (0..n)
            .map(|i| {
                let j = i + phase;
                let x0 = ((j * 7 + 3) % 13) as f32 / 13.0;
                let x1 = ((j * 5 + 1) % 17) as f32 / 17.0;
                (vec![x0, x1], vec![0.5 * x0 + 0.1])
            })
            .collect();
        write_bin_file(&dir.join("part-000.bin"), &records).unwrap();
    }

    /// Fixture tree: parent, four children, two corpora and a manifest.
    fn fixture(root: &Path, wide_child: bool) -> PathBuf {
        let parent = identity_creature(2, 1);
        write_creature(&root.join("parent.json"), &parent);
        write_creature(&root.join("same.json"), &parent);
        write_creature(&root.join("same-copy.json"), &parent);

        let mut scaled = identity_creature(2, 1);
        scaled.synapses[0].weight = 2.0;
        write_creature(&root.join("scaled.json"), &scaled);

        let mut collapsed = identity_creature(2, 1);
        collapsed.synapses[0].weight = 0.0;
        collapsed.neurons[0].bias = 0.5;
        write_creature(&root.join("collapsed.json"), &collapsed);

        let mut children = vec![
            json!({"name": "same", "creature": "same.json", "fullDelta": 0.02, "sampleDelta": 0.01}),
            json!({"name": "same-copy", "creature": "same-copy.json", "fullDelta": 0.01, "sampleDelta": 0.02}),
            json!({"name": "scaled", "creature": "scaled.json", "fullDelta": -0.03, "sampleDelta": -0.01}),
            json!({"name": "collapsed", "creature": "collapsed.json", "fullDelta": -0.05, "sampleDelta": 0.0}),
        ];
        if wide_child {
            write_creature(&root.join("wide.json"), &identity_creature(3, 1));
            children.push(
                json!({"name": "wide", "creature": "wide.json", "fullDelta": 0.0, "sampleDelta": 0.0}),
            );
        }
        write_corpus(&root.join("calibration"), 40, 0);
        write_corpus(&root.join("validation"), 30, 5);

        let manifest = json!({
            "parent": "parent.json",
            "children": children,
            "calibration": "calibration",
            "validation": "validation",
            "shuffleSeeds": [1, 2, 3],
            "minAucGain": 0.05
        });
        let path = root.join("manifest.json");
        std::fs::write(&path, manifest.to_string()).unwrap();
        path
    }

    fn child_entry<'a>(split: &'a serde_json::Value, name: &str) -> &'a serde_json::Value {
        split["children"]
            .as_array()
            .unwrap()
            .iter()
            .find(|c| c["name"] == name)
            .unwrap()
    }

    #[test]
    fn end_to_end_fixture_audit_writes_and_returns_the_same_report() {
        let dir = tempfile::tempdir().unwrap();
        let manifest = fixture(dir.path(), false);
        let out = dir.path().join("report.json");

        let value = run_audit(&manifest, &out).unwrap();

        assert_eq!(value["schema"], SCHEMA);
        assert_eq!(value["evidence"], EVIDENCE);
        let cal = &value["splits"]["calibration"];
        let val = &value["splits"]["validation"];
        assert_eq!(child_entry(cal, "same")["preservationVsMean"], json!(1.0));
        assert_eq!(child_entry(val, "same")["preservationVsMean"], json!(1.0));

        let collapsed = child_entry(cal, "collapsed");
        assert_eq!(collapsed["collapsedChild"], json!(true));
        assert!(collapsed["preservationVsMean"].as_f64().unwrap() <= 0.0);

        assert!(value["threshold"].is_number());
        assert_eq!(value["coverage"]["childrenSupplied"], json!(4));
        assert_eq!(value["coverage"]["childrenScored"], json!(4));
        assert_eq!(value["coverage"]["validationRecords"], json!(30));
        assert_eq!(value["coverage"]["validationCorpusRecords"], json!(30));
        assert_eq!(value["criterion"]["status"], json!("notMet"));

        let on_disk: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&out).unwrap()).unwrap();
        assert_eq!(on_disk, value);
        assert!(!value.to_string().contains("NaN"));
    }

    #[test]
    fn width_mismatched_child_is_refused_by_name() {
        let dir = tempfile::tempdir().unwrap();
        let manifest = fixture(dir.path(), true);
        let out = dir.path().join("report.json");
        let err = run_audit(&manifest, &out).unwrap_err();
        assert!(err.contains("wide"), "{err}");
        assert!(!out.exists());
    }
}
