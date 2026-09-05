//! Sampled hidden-neuron activation statistics (Issues #3, #44).
//!
//! Statistics **propose** candidates only. They are not a proxy acceptance
//! score and must not be presented as proof that a neuron is unimportant.
//!
//! Accumulation uses the same NEAT-AI-core compiled forward pass as scoring
//! (`CompiledNetwork::activate`), with `f64` running moments so a long corpus
//! does not lose the mean to `f32` rounding. Per-record activations are not
//! retained: memory is one compiled network plus one accumulator per hidden
//! neuron.
//!
//! Because the statistics only *propose*, they do not need full-corpus
//! precision: a multi-million-record corpus costs minutes of the run budget
//! while the extra precision it buys is far below the score movement the loop
//! chases. [`SampleSpec`] therefore visits evenly-spread blocks of records
//! (deterministically placed from the corpus identity) and stops once every
//! neuron mean's standard error is small against that neuron's activation
//! scale. Set `max_records = 0` to restore the full-corpus scan.
//!
//! Set [`SampleSpec::probes`] to retain each neuron's post-activation at a
//! handful of deterministically-placed probe records (Issue #109). Those short
//! vectors are the behavioural signature [`crate::signature`] buckets and
//! correlates to find near-duplicate neurons; they cost `probes` floats per
//! hidden neuron and nothing at all when the count is zero.

use std::ops::ControlFlow;
use std::path::{Path, PathBuf};
use std::time::Instant;

use neat_core::training_data::TrainingDataConfig;
use neat_core::{CreatureExport, compile_creature};
use serde::{Deserialize, Serialize};

use crate::corpus::{CorpusInfo, RecordRange, for_each_selected_chunk};
use crate::incumbent::Incumbent;

/// Cache / on-disk format version. Bump when the JSON shape changes.
pub const STATS_FORMAT_VERSION: u32 = 2;
/// Default records per streaming chunk.
pub const DEFAULT_CHUNK_RECORDS: usize = 4096;
/// Default cap on records visited by the activation scan (issue #44).
pub const DEFAULT_SAMPLE_RECORDS: u64 = 100_000;
/// Default contiguous records per sampled block — one sequential read each.
pub const DEFAULT_SAMPLE_BLOCK_RECORDS: usize = 4096;
/// Default records visited before adaptive stopping may trigger.
pub const DEFAULT_MIN_SAMPLE_RECORDS: u64 = 20_000;
/// Default standard error of a neuron mean, relative to that neuron's
/// activation scale, at which the scan may stop early.
pub const DEFAULT_TARGET_REL_SE: f64 = 0.01;
/// Probe records retained per neuron when correlated-neuron merging is on (#109).
///
/// One `u64` sign signature is built from the first 64, so retaining more than
/// that buys correlation precision, not bucket precision.
pub const DEFAULT_PROBE_RECORDS: usize = 64;
/// Most probe records a [`SampleSpec`] may retain per neuron (#109).
///
/// The signature is a `u64`, so a longer probe vector would leave bits of the
/// behaviour outside the bucket key that selects pairs to correlate.
pub const MAX_PROBE_RECORDS: usize = 64;

/// How much of the corpus the activation scan visits (issue #44).
///
/// The plan is a function of the spec and the corpus alone, so a cached scan
/// is reproducible for a given `(incumbent, corpus, spec)`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SampleSpec {
    /// Cap on records visited; `0` scans the whole corpus.
    pub max_records: u64,
    /// Contiguous records per sampled block.
    pub block_records: usize,
    /// Records that must be visited before adaptive stopping may trigger.
    pub min_records: u64,
    /// Relative standard-error target for adaptive stopping; `0` disables it.
    pub target_rel_se: f64,
    /// Probe activations retained per neuron; `0` retains none (#109).
    #[serde(default)]
    pub probes: usize,
}

impl Default for SampleSpec {
    fn default() -> Self {
        Self {
            max_records: DEFAULT_SAMPLE_RECORDS,
            block_records: DEFAULT_SAMPLE_BLOCK_RECORDS,
            min_records: DEFAULT_MIN_SAMPLE_RECORDS,
            target_rel_se: DEFAULT_TARGET_REL_SE,
            probes: 0,
        }
    }
}

impl SampleSpec {
    /// Visit every record, with no adaptive stopping (pre-#44 behaviour).
    pub fn full() -> Self {
        Self {
            max_records: 0,
            target_rel_se: 0.0,
            ..Self::default()
        }
    }

    /// Default sampling capped at `max_records`; `0` means the full corpus.
    ///
    /// The block size and the adaptive-stopping floor are clamped to the cap,
    /// so a small cap stays a cap instead of being overrun by one large block
    /// or holding a scan open past its own limit.
    pub fn with_max_records(max_records: u64) -> Self {
        if max_records == 0 {
            return Self::full();
        }
        let defaults = Self::default();
        Self {
            max_records,
            block_records: defaults
                .block_records
                .min(usize::try_from(max_records).unwrap_or(usize::MAX)),
            min_records: defaults.min_records.min(max_records),
            ..defaults
        }
    }

    /// This spec with `probes` activations retained per neuron (Issue #109).
    ///
    /// Not clamped: a count above [`MAX_PROBE_RECORDS`] is refused by
    /// [`crate::config::OckhamConfig::validate`] under the flag's own name,
    /// because a silently shortened probe set would give a run weaker
    /// signatures than the ones it asked for and say nothing.
    pub fn with_probes(self, probes: usize) -> Self {
        Self { probes, ..self }
    }

    /// Filename-safe tag identifying this spec in a cache key.
    ///
    /// The probe count is part of the tag, so a scan that retained no probes is
    /// never served to a run that asked for them — a merge-enabled run reading
    /// a probe-free cache would find no signatures and silently propose nothing.
    pub fn tag(&self) -> String {
        let base = if self.max_records == 0 {
            "full".to_string()
        } else {
            format!(
                "n{}b{}m{}e{}",
                self.max_records,
                self.block_records.max(1),
                self.min_records,
                (self.target_rel_se.max(0.0) * 1e6).round() as u64
            )
        };
        match self.probes {
            0 => base,
            p => format!("{base}p{p}"),
        }
    }

    /// Sampled-record indices at which a probe activation is retained (#109).
    ///
    /// Ascending, distinct, and a pure function of the spec and the plan, so a
    /// cached scan is reproducible. The slots are spread over the **whole**
    /// sampled plan, not a prefix of it: a signature is a claim about how a
    /// neuron behaves on the corpus, and probes crowded into the first blocks
    /// would call two neurons duplicates on the evidence of the corpus's
    /// opening records alone. The scan pays for that by holding its
    /// adaptive stop until the last probe slot has been captured — a run that
    /// asked for signatures asked for the records they are built from.
    ///
    /// Two structural bounds shorten the result, and neither is silent: a count
    /// above [`MAX_PROBE_RECORDS`] is refused by the flag's own validation
    /// before it reaches here, and a plan too short to hold one record per
    /// probe is reported by the scan before it starts.
    pub fn probe_slots(&self, planned: u64) -> Vec<u64> {
        let probes = (self.probes.min(MAX_PROBE_RECORDS) as u64).min(planned);
        (0..probes)
            .map(|p| p * planned / probes)
            .collect::<Vec<u64>>()
    }

    /// Ascending, non-overlapping record ranges to visit for `corpus`.
    ///
    /// One block per stratum of `record_count / blocks` records, placed inside
    /// its stratum by a generator seeded from the corpus identity — evenly
    /// spread like systematic sampling, without its fixed-phase aliasing.
    pub fn plan(&self, corpus: &CorpusInfo) -> Vec<RecordRange> {
        let total = corpus.record_count;
        if total == 0 || self.max_records == 0 || self.max_records >= total {
            return vec![RecordRange {
                start: 0,
                len: total,
            }];
        }
        // A block never exceeds the cap, so the plan cannot overrun it.
        let block = (self.block_records.max(1) as u64).min(self.max_records);
        let blocks = self.max_records.div_ceil(block).max(1).min(total);
        let mut state =
            u64::from_str_radix(&corpus.identity, 16).unwrap_or(0) ^ 0x9e37_79b9_7f4a_7c15;
        let mut ranges = Vec::with_capacity(blocks as usize);
        for i in 0..blocks {
            let start = (u128::from(i) * u128::from(total) / u128::from(blocks)) as u64;
            let end = (u128::from(i + 1) * u128::from(total) / u128::from(blocks)) as u64;
            let span = end - start;
            let len = block.min(span);
            state = split_mix64(state);
            let offset = match span - len {
                0 => 0,
                slack => state % (slack + 1),
            };
            ranges.push(RecordRange {
                start: start + offset,
                len,
            });
        }
        ranges
    }
}

/// SplitMix64 — deterministic block placement inside each stratum.
fn split_mix64(state: u64) -> u64 {
    let mut z = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

/// One hidden neuron's post-activation summary over the scanned records.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct NeuronStats {
    /// Neuron UUID.
    pub uuid: String,
    /// Index into `creature.neurons`.
    pub neuron_index: usize,
    /// Records accumulated.
    pub count: u64,
    /// Mean post-activation (f64).
    pub mean: f64,
    /// Population variance (`m2 / n`). Zero when `count < 2`.
    pub variance: f64,
    /// `sqrt(variance)`.
    pub std_dev: f64,
    /// Mean of `|activation|`.
    pub mean_abs: f64,
    /// Minimum activation.
    pub min: f64,
    /// Maximum activation.
    pub max: f64,
}

/// One hidden neuron's retained probe activations (Issue #109).
///
/// The behavioural signature [`crate::signature`] works from: the neuron's
/// post-activation at each probe record, in ascending record order. Every
/// neuron in one [`ActivationStats`] is probed at the same records, so two
/// vectors are directly comparable element by element.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct NeuronProbes {
    /// Neuron UUID.
    pub uuid: String,
    /// Post-activation at each probe record.
    pub values: Vec<f32>,
}

/// Sampled hidden-neuron statistics for one incumbent + corpus.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ActivationStats {
    /// [`STATS_FORMAT_VERSION`].
    pub format_version: u32,
    /// Incumbent checksum the stats were measured against.
    pub creature_checksum: String,
    /// Corpus identity the stats were measured against.
    pub corpus_identity: String,
    /// Records streamed — the sample, not the corpus.
    pub record_count: u64,
    /// Records in the whole corpus.
    #[serde(default)]
    pub corpus_record_count: u64,
    /// Sampling policy the scan followed.
    #[serde(default = "SampleSpec::full")]
    pub sample: SampleSpec,
    /// `true` when adaptive stopping ended the scan before the plan ran out.
    #[serde(default)]
    pub stopped_early: bool,
    /// Wall time of the scan (ms), excluding cache hits.
    pub scan_ms: u64,
    /// Per-hidden-neuron summaries.
    pub neurons: Vec<NeuronStats>,
    /// Retained probe activations, one entry per measured neuron (Issue #109).
    ///
    /// Empty unless [`SampleSpec::probes`] asked for them.
    #[serde(default)]
    pub probes: Vec<NeuronProbes>,
    /// `true` when this object was loaded from the keyed cache.
    #[serde(default)]
    pub from_cache: bool,
}

#[derive(Clone)]
struct Accumulator {
    uuid: String,
    neuron_index: usize,
    activation_index: usize,
    count: u64,
    mean: f64,
    m2: f64,
    abs_sum: f64,
    min: f64,
    max: f64,
    probes: Vec<f32>,
}

impl Accumulator {
    fn new(uuid: String, neuron_index: usize, activation_index: usize, probes: usize) -> Self {
        Self {
            uuid,
            neuron_index,
            activation_index,
            count: 0,
            mean: 0.0,
            m2: 0.0,
            abs_sum: 0.0,
            min: f64::INFINITY,
            max: f64::NEG_INFINITY,
            probes: Vec::with_capacity(probes),
        }
    }

    fn push(&mut self, x: f32) {
        let x = f64::from(x);
        self.count += 1;
        let delta = x - self.mean;
        self.mean += delta / self.count as f64;
        self.m2 += delta * (x - self.mean);
        self.abs_sum += x.abs();
        if x < self.min {
            self.min = x;
        }
        if x > self.max {
            self.max = x;
        }
    }

    /// Standard error of the mean relative to this neuron's activation scale.
    ///
    /// A neuron that barely moves converges immediately; a noisy one keeps the
    /// scan running until its mean is pinned down to `target_rel_se`.
    fn relative_standard_error(&self) -> f64 {
        if self.count < 2 {
            return f64::INFINITY;
        }
        let n = self.count as f64;
        let std_dev = (self.m2 / n).max(0.0).sqrt();
        if std_dev == 0.0 {
            return 0.0;
        }
        let scale = std_dev.max(self.abs_sum / n);
        std_dev / (n.sqrt() * scale)
    }

    fn finish(&self) -> NeuronStats {
        let (variance, std_dev, min, max, mean_abs) = if self.count == 0 {
            (0.0, 0.0, 0.0, 0.0, 0.0)
        } else {
            let variance = self.m2 / self.count as f64;
            (
                variance,
                variance.max(0.0).sqrt(),
                self.min,
                self.max,
                self.abs_sum / self.count as f64,
            )
        };
        NeuronStats {
            uuid: self.uuid.clone(),
            neuron_index: self.neuron_index,
            count: self.count,
            mean: self.mean,
            variance,
            std_dev,
            mean_abs,
            min,
            max,
        }
    }
}

/// Cache path keyed by format version, creature checksum, corpus and sampling.
pub fn cache_path(
    dir: &Path,
    checksum: &str,
    corpus_identity: &str,
    sample: &SampleSpec,
) -> PathBuf {
    dir.join(format!(
        "activation-stats.v{STATS_FORMAT_VERSION}.{checksum}.{corpus_identity}.{}.json",
        sample.tag()
    ))
}

/// Load a cache file only when every key matches.
pub fn load_cached_stats(
    path: &Path,
    checksum: &str,
    corpus_identity: &str,
    sample: &SampleSpec,
) -> Result<Option<ActivationStats>, String> {
    if !path.is_file() {
        return Ok(None);
    }
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut stats: ActivationStats =
        serde_json::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))?;
    if stats.format_version != STATS_FORMAT_VERSION
        || stats.creature_checksum != checksum
        || stats.corpus_identity != corpus_identity
        || stats.sample != *sample
    {
        return Ok(None);
    }
    stats.from_cache = true;
    Ok(Some(stats))
}

/// Write stats to `path`.
pub fn store_cached_stats(path: &Path, stats: &ActivationStats) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| format!("{}: {e}", parent.display()))?;
    }
    let json = serde_json::to_string_pretty(stats).map_err(|e| e.to_string())?;
    std::fs::write(path, json).map_err(|e| format!("{}: {e}", path.display()))
}

/// Stream the sampled corpus through NEAT-AI-core inference and accumulate
/// hidden-neuron statistics.
///
/// `sample` decides how much of the corpus is visited; [`SampleSpec::full`]
/// keeps the exhaustive scan.
pub fn compute_activation_stats(
    creature: &CreatureExport,
    creature_checksum: &str,
    training_dir: &Path,
    corpus: &CorpusInfo,
    chunk_records: usize,
    sample: &SampleSpec,
) -> Result<ActivationStats, String> {
    let mut net = compile_creature(creature).map_err(|e| e.to_string())?;
    let mut acc: Vec<Accumulator> = creature
        .neurons
        .iter()
        .enumerate()
        .filter(|(_, n)| n.neuron_type == "hidden")
        .map(|(i, n)| Accumulator::new(n.uuid.clone(), i, net.num_inputs + i, sample.probes))
        .collect();
    if acc.is_empty() {
        // Nothing to measure: streaming the corpus could only produce an empty
        // measurement more slowly.
        return Ok(ActivationStats {
            creature_checksum: creature_checksum.to_string(),
            corpus_identity: corpus.identity.clone(),
            corpus_record_count: corpus.record_count,
            sample: *sample,
            ..ActivationStats::empty()
        });
    }
    let cfg = TrainingDataConfig::new(creature.input, creature.output);
    let plan = sample.plan(corpus);
    let planned: u64 = plan.iter().map(|r| r.len.min(corpus.record_count)).sum();
    let probe_slots = sample.probe_slots(planned);
    // A probe set the *plan* cannot hold is reported before the scan starts:
    // one probe needs one guaranteed record, so a short corpus caps the
    // signature length, and a run whose signatures are shorter than it asked
    // for must be told rather than left to infer it (#109).
    if probe_slots.len() < sample.probes {
        crate::log::detail(&format!(
            "activation plan holds {} of {} probe record(s) asked for; \
             behavioural signatures are that much shorter",
            probe_slots.len(),
            sample.probes
        ));
    }
    let started = Instant::now();
    let mut seen = 0u64;
    let mut next_probe = 0usize;
    let mut stopped_early = false;
    let mut last_log = Instant::now();
    let streamed =
        for_each_selected_chunk(training_dir, &cfg, &plan, chunk_records.max(1), |chunk| {
            for r in 0..chunk.records {
                let inputs = &chunk.inputs[r * creature.input..(r + 1) * creature.input];
                let _ = net.activate(inputs, creature.output);
                let probe = probe_slots.get(next_probe) == Some(&(seen + r as u64));
                for a in &mut acc {
                    let x = net.activations[a.activation_index];
                    a.push(x);
                    if probe {
                        a.probes.push(x);
                    }
                }
                if probe {
                    next_probe += 1;
                }
            }
            seen += chunk.records as u64;
            if last_log.elapsed().as_secs() >= 15 {
                let rate = seen as f64 / started.elapsed().as_secs_f64().max(1e-9);
                crate::log::detail(&format!(
                    "activation scan {seen}/{planned} sampled records of {} ({rate:.0} rec/s)",
                    corpus.record_count
                ));
                last_log = Instant::now();
            }
            // Converged means the *means* have converged; the probe set is a
            // separate promise. Stopping with slots still ahead would leave
            // every signature describing a prefix of the corpus, so a probing
            // scan runs on until the last slot is captured (#109).
            if sample.target_rel_se > 0.0
                && seen >= sample.min_records
                && next_probe >= probe_slots.len()
                && acc
                    .iter()
                    .all(|a| a.relative_standard_error() <= sample.target_rel_se)
            {
                stopped_early = true;
                return Ok(ControlFlow::Break(()));
            }
            Ok(ControlFlow::Continue(()))
        })?;
    let scan_ms = started.elapsed().as_millis() as u64;
    if !stopped_early && streamed != planned {
        return Err(format!(
            "activation scan saw {streamed} records but planned {planned} of the \
             {} the corpus identity has",
            corpus.record_count
        ));
    }
    if streamed == 0 && corpus.record_count > 0 {
        return Err(format!(
            "activation scan visited no records of the {} the corpus identity has",
            corpus.record_count
        ));
    }
    // A probe set the scan never filled is reported, never quietly shortened:
    // a signature built from fewer records than asked for is a weaker signature
    // and the run that reads it has to be able to say so (#109).
    if next_probe < probe_slots.len() {
        crate::log::detail(&format!(
            "activation scan captured {next_probe} of {} probe record(s); \
             behavioural signatures are that much shorter",
            probe_slots.len()
        ));
    }
    let neurons: Vec<NeuronStats> = acc.iter().map(Accumulator::finish).collect();
    let probes = if sample.probes == 0 {
        Vec::new()
    } else {
        acc.into_iter()
            .map(|a| NeuronProbes {
                uuid: a.uuid,
                values: a.probes,
            })
            .collect()
    };
    Ok(ActivationStats {
        format_version: STATS_FORMAT_VERSION,
        creature_checksum: creature_checksum.to_string(),
        corpus_identity: corpus.identity.clone(),
        record_count: streamed,
        corpus_record_count: corpus.record_count,
        sample: *sample,
        stopped_early,
        scan_ms,
        neurons,
        probes,
        from_cache: false,
    })
}

/// Load the keyed cache or compute and store.
pub fn ensure_activation_stats(
    incumbent: &Incumbent,
    training_dir: &Path,
    corpus: &CorpusInfo,
    cache_dir: &Path,
    chunk_records: usize,
    sample: &SampleSpec,
) -> Result<ActivationStats, String> {
    let path = cache_path(cache_dir, &incumbent.checksum, &corpus.identity, sample);
    if let Some(cached) = load_cached_stats(&path, &incumbent.checksum, &corpus.identity, sample)? {
        return Ok(cached);
    }
    let stats = compute_activation_stats(
        &incumbent.creature,
        &incumbent.checksum,
        training_dir,
        corpus,
        chunk_records,
        sample,
    )?;
    store_cached_stats(&path, &stats)?;
    Ok(stats)
}

/// Look up stats for a hidden neuron UUID.
impl ActivationStats {
    /// Stats for `uuid`, if that hidden neuron was measured.
    pub fn by_uuid(&self, uuid: &str) -> Option<&NeuronStats> {
        self.neurons.iter().find(|n| n.uuid == uuid)
    }

    /// Measurement-free placeholder for callers that need no activation signal.
    ///
    /// Every `by_uuid` lookup misses, so a statistics-driven ordering degrades
    /// to the random control rather than silently inventing a signal.
    pub fn empty() -> Self {
        Self {
            format_version: STATS_FORMAT_VERSION,
            creature_checksum: String::new(),
            corpus_identity: String::new(),
            record_count: 0,
            corpus_record_count: 0,
            sample: SampleSpec::full(),
            stopped_early: false,
            scan_ms: 0,
            neurons: Vec::new(),
            probes: Vec::new(),
            from_cache: false,
        }
    }

    /// `true` when the scan visited fewer records than the corpus holds.
    pub fn is_sampled(&self) -> bool {
        self.record_count < self.corpus_record_count
    }

    /// Retained probe activations for `uuid`, if any were captured (#109).
    pub fn probes_of(&self, uuid: &str) -> Option<&[f32]> {
        self.probes
            .iter()
            .find(|p| p.uuid == uuid)
            .map(|p| p.values.as_slice())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::corpus::{corpus_info, write_bin_file};
    use crate::fixtures::hidden_identity_creature;
    use crate::incumbent::Incumbent;
    use neat_core::training_data::TrainingDataConfig;

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() <= 1e-12 * a.abs().max(b.abs()).max(1.0)
    }

    fn setup(values: &[f32], bias: f64, weight: f64) -> (tempfile::TempDir, Incumbent, CorpusInfo) {
        let tmp = tempfile::tempdir().unwrap();
        let recs: Vec<(Vec<f32>, Vec<f32>)> = values.iter().map(|&x| (vec![x], vec![x])).collect();
        write_bin_file(&tmp.path().join("0.bin"), &recs).unwrap();
        let inc = Incumbent::from_creature(hidden_identity_creature(bias, weight), "t").unwrap();
        let corpus = corpus_info(tmp.path(), &TrainingDataConfig::new(1, 1)).unwrap();
        (tmp, inc, corpus)
    }

    #[test]
    fn means_min_max_variance_match_hand_calculation() {
        // h = 0.5 + 2 * x  for x in {1, 0, -1} → {2.5, 0.5, -1.5}
        let (tmp, inc, corpus) = setup(&[1.0, 0.0, -1.0], 0.5, 2.0);
        let stats = compute_activation_stats(
            &inc.creature,
            &inc.checksum,
            tmp.path(),
            &corpus,
            2,
            &SampleSpec::full(),
        )
        .unwrap();
        let h = stats.by_uuid("h1").expect("h1");
        assert_eq!(h.count, 3);
        assert!(close(h.mean, 0.5), "mean {}", h.mean);
        assert!(close(h.mean_abs, 1.5), "mean_abs {}", h.mean_abs);
        assert!(close(h.min, -1.5), "min {}", h.min);
        assert!(close(h.max, 2.5), "max {}", h.max);
        assert!(close(h.variance, 8.0 / 3.0), "var {}", h.variance);
        assert!(
            close(h.std_dev, (8.0 / 3.0_f64).sqrt()),
            "std {}",
            h.std_dev
        );
        let again = compute_activation_stats(
            &inc.creature,
            &inc.checksum,
            tmp.path(),
            &corpus,
            1,
            &SampleSpec::full(),
        )
        .unwrap();
        let h2 = again.by_uuid("h1").unwrap();
        assert!(close(h.mean, h2.mean) && close(h.variance, h2.variance));
        assert_eq!(stats.record_count, corpus.record_count);
        assert!(!stats.is_sampled());
    }

    #[test]
    fn cache_is_not_reused_for_a_changed_creature_or_corpus() {
        let (tmp, inc, corpus) = setup(&[1.0, 2.0, 3.0], 0.0, 1.0);
        let cache = tmp.path().join("cache");
        let full = SampleSpec::full();
        let a = ensure_activation_stats(&inc, tmp.path(), &corpus, &cache, 8, &full).unwrap();
        assert!(!a.from_cache);
        let b = ensure_activation_stats(&inc, tmp.path(), &corpus, &cache, 8, &full).unwrap();
        assert!(b.from_cache);
        assert_eq!(a.neurons, b.neurons);

        let other = Incumbent::from_creature(hidden_identity_creature(1.0, 1.0), "u").unwrap();
        let path = cache_path(&cache, &other.checksum, &corpus.identity, &full);
        // Plant a cache file with the wrong checksum inside a matching filename
        // so a naive loader that ignored the JSON keys would still succeed.
        store_cached_stats(
            &path,
            &ActivationStats {
                creature_checksum: "not-the-creature".into(),
                ..a.clone()
            },
        )
        .unwrap();
        assert!(
            load_cached_stats(&path, &other.checksum, &corpus.identity, &full)
                .unwrap()
                .is_none()
        );

        write_bin_file(&tmp.path().join("0.bin"), &[(vec![9.0f32], vec![9.0f32])]).unwrap();
        let corpus2 = corpus_info(tmp.path(), &TrainingDataConfig::new(1, 1)).unwrap();
        assert_ne!(corpus.identity, corpus2.identity);
        let c = ensure_activation_stats(&inc, tmp.path(), &corpus2, &cache, 8, &full).unwrap();
        assert!(!c.from_cache);
        assert_eq!(c.record_count, 1);
        assert_ne!(c.neurons[0].mean, a.neurons[0].mean);
    }

    #[test]
    fn a_full_scan_cache_entry_is_never_served_to_a_sampled_scan() {
        let (tmp, inc, corpus) = setup(&(0..2_000).map(|i| i as f32).collect::<Vec<_>>(), 0.0, 1.0);
        let cache = tmp.path().join("cache");
        let full = SampleSpec::full();
        let sampled = SampleSpec {
            max_records: 400,
            block_records: 100,
            min_records: u64::MAX, // adaptive stopping cannot fire
            target_rel_se: 0.0,
            probes: 0,
        };
        let a = ensure_activation_stats(&inc, tmp.path(), &corpus, &cache, 64, &full).unwrap();
        assert_eq!(a.record_count, 2_000);

        let b = ensure_activation_stats(&inc, tmp.path(), &corpus, &cache, 64, &sampled).unwrap();
        assert!(!b.from_cache, "a different sample spec is a different key");
        assert_eq!(b.record_count, 400);
        assert!(b.is_sampled());
        assert_ne!(
            cache_path(&cache, &inc.checksum, &corpus.identity, &full),
            cache_path(&cache, &inc.checksum, &corpus.identity, &sampled)
        );

        let again =
            ensure_activation_stats(&inc, tmp.path(), &corpus, &cache, 64, &sampled).unwrap();
        assert!(again.from_cache);
        assert_eq!(again.neurons, b.neurons);
    }

    #[test]
    fn sampling_visits_a_fraction_of_the_corpus_and_still_tracks_the_full_mean() {
        // h = x for x drawn from a sawtooth over 20_000 records.
        let values: Vec<f32> = (0..20_000).map(|i| (i % 101) as f32).collect();
        let (tmp, inc, corpus) = setup(&values, 0.0, 1.0);
        let spec = SampleSpec {
            max_records: 4_000,
            block_records: 200,
            min_records: u64::MAX,
            target_rel_se: 0.0,
            probes: 0,
        };
        let sampled =
            compute_activation_stats(&inc.creature, &inc.checksum, tmp.path(), &corpus, 64, &spec)
                .unwrap();
        assert_eq!(sampled.record_count, 4_000, "one fifth of the corpus");
        assert_eq!(sampled.corpus_record_count, 20_000);
        assert!(sampled.is_sampled());

        let exact = compute_activation_stats(
            &inc.creature,
            &inc.checksum,
            tmp.path(),
            &corpus,
            64,
            &SampleSpec::full(),
        )
        .unwrap();
        let (s, f) = (sampled.by_uuid("h1").unwrap(), exact.by_uuid("h1").unwrap());
        assert_eq!(s.count, 4_000);
        assert!(
            (s.mean - f.mean).abs() < 0.05 * f.std_dev,
            "sampled mean {} is far from the full-corpus mean {}",
            s.mean,
            f.mean
        );
    }

    #[test]
    fn the_sample_plan_is_deterministic_ascending_and_capped() {
        let (_tmp, _inc, corpus) =
            setup(&(0..5_000).map(|i| i as f32).collect::<Vec<_>>(), 0.0, 1.0);
        let spec = SampleSpec {
            max_records: 500,
            block_records: 100,
            ..SampleSpec::default()
        };
        let plan = spec.plan(&corpus);
        assert_eq!(plan, spec.plan(&corpus), "same corpus, same plan");
        assert_eq!(plan.len(), 5);
        assert_eq!(plan.iter().map(|r| r.len).sum::<u64>(), 500);
        for pair in plan.windows(2) {
            assert!(pair[0].end() <= pair[1].start, "{plan:?} must not overlap");
        }
        assert!(plan.last().unwrap().end() <= corpus.record_count);
        assert!(
            plan.windows(2).any(|p| p[1].start - p[0].end() > 0),
            "a sampled plan must skip records: {plan:?}"
        );

        // A cap at or above the corpus size degrades to the exhaustive scan.
        let whole = SampleSpec::with_max_records(5_000).plan(&corpus);
        assert_eq!(
            whole,
            vec![RecordRange {
                start: 0,
                len: 5_000
            }]
        );
        assert_eq!(SampleSpec::full().plan(&corpus), whole);
    }

    #[test]
    fn a_cap_below_the_block_size_still_caps_the_scan() {
        let (tmp, inc, corpus) = setup(&(0..5_000).map(|i| i as f32).collect::<Vec<_>>(), 0.0, 1.0);
        let spec = SampleSpec::with_max_records(300);
        assert!(spec.block_records <= 300, "block clamped to the cap");
        assert!(spec.min_records <= 300, "stopping floor clamped to the cap");
        assert_eq!(spec.plan(&corpus).iter().map(|r| r.len).sum::<u64>(), 300);
        let stats =
            compute_activation_stats(&inc.creature, &inc.checksum, tmp.path(), &corpus, 64, &spec)
                .unwrap();
        assert!(stats.record_count <= 300, "{}", stats.record_count);
        assert!(stats.record_count > 0);
    }

    #[test]
    fn a_creature_without_hidden_neurons_is_not_scanned_at_all() {
        let (tmp, _inc, corpus) = setup(&[1.0, 2.0, 3.0], 0.0, 1.0);
        let flat = crate::fixtures::identity_creature(1, 1);
        let stats = compute_activation_stats(
            &flat,
            "t",
            tmp.path(),
            &corpus,
            8,
            &SampleSpec::with_max_records(2),
        )
        .unwrap();
        assert!(stats.neurons.is_empty());
        assert_eq!(stats.record_count, 0, "nothing to measure, nothing to read");
        assert_eq!(stats.corpus_record_count, 3);
    }

    #[test]
    fn adaptive_stopping_ends_a_constant_neuron_scan_early() {
        // A constant activation has zero standard error, so the scan may stop
        // as soon as the minimum sample is in.
        let (tmp, inc, corpus) = setup(&vec![1.0f32; 20_000], 0.0, 1.0);
        let spec = SampleSpec {
            max_records: 10_000,
            block_records: 500,
            min_records: 1_000,
            target_rel_se: 0.01,
            probes: 0,
        };
        let stats = compute_activation_stats(
            &inc.creature,
            &inc.checksum,
            tmp.path(),
            &corpus,
            500,
            &spec,
        )
        .unwrap();
        assert!(stats.stopped_early, "constant neuron must converge early");
        assert_eq!(stats.record_count, 1_000);
        assert!(close(stats.by_uuid("h1").unwrap().mean, 1.0));

        // A moving activation is not declared converged at the same point.
        let values: Vec<f32> = (0..20_000).map(|i| (i % 101) as f32).collect();
        let (tmp, inc, corpus) = setup(&values, 0.0, 1.0);
        let stats = compute_activation_stats(
            &inc.creature,
            &inc.checksum,
            tmp.path(),
            &corpus,
            500,
            &spec,
        )
        .unwrap();
        assert!(
            stats.record_count > 1_000,
            "noisy neuron needs more records"
        );
    }

    /// Issue #109: probe records are retained only when asked for, land at the
    /// same records for every neuron, and are reproducible for a fixed spec.
    #[test]
    fn probe_records_are_retained_only_when_asked_for_and_are_reproducible() {
        let values: Vec<f32> = (0..2_000).map(|i| ((i % 37) as f32) - 18.0).collect();
        let (tmp, inc, corpus) = setup(&values, 0.0, 1.0);
        let bare = SampleSpec {
            max_records: 1_000,
            block_records: 100,
            min_records: u64::MAX,
            target_rel_se: 0.0,
            probes: 0,
        };
        let probed = bare.with_probes(16);
        assert_eq!(probed.probes, 16);
        assert_ne!(bare.tag(), probed.tag(), "the probe count keys the cache");

        let plain =
            compute_activation_stats(&inc.creature, &inc.checksum, tmp.path(), &corpus, 64, &bare)
                .unwrap();
        assert!(plain.probes.is_empty(), "a control run retains nothing");
        assert!(plain.probes_of("h1").is_none());

        let a = compute_activation_stats(
            &inc.creature,
            &inc.checksum,
            tmp.path(),
            &corpus,
            64,
            &probed,
        )
        .unwrap();
        let captured = a.probes_of("h1").expect("h1 must be probed");
        assert_eq!(captured.len(), 16);
        // The summary statistics are untouched by the probing.
        assert_eq!(a.record_count, plain.record_count);
        assert_eq!(a.neurons, plain.neurons);

        // A different chunk size must not move the probe records: the slots are
        // a function of the plan, not of how the reader happened to batch it.
        let b = compute_activation_stats(
            &inc.creature,
            &inc.checksum,
            tmp.path(),
            &corpus,
            7,
            &probed,
        )
        .unwrap();
        assert_eq!(a.probes, b.probes);
        assert_eq!(probed.probe_slots(1_000).len(), 16);
        assert!(
            probed.probe_slots(1_000).windows(2).all(|w| w[0] < w[1]),
            "probe slots must be ascending and distinct"
        );
    }

    /// A cache written without probes must never be served to a run that needs
    /// them: silently signature-free statistics would propose no merge at all
    /// and read as "there were no near-duplicates".
    #[test]
    fn a_probe_free_cache_entry_is_never_served_to_a_probing_scan() {
        let values: Vec<f32> = (0..600).map(|i| ((i % 13) as f32) - 6.0).collect();
        let (tmp, inc, corpus) = setup(&values, 0.0, 1.0);
        let cache = tmp.path().join("cache");
        let bare = SampleSpec::full();
        let probed = bare.with_probes(16);
        let a = ensure_activation_stats(&inc, tmp.path(), &corpus, &cache, 64, &bare).unwrap();
        assert!(a.probes.is_empty());
        let b = ensure_activation_stats(&inc, tmp.path(), &corpus, &cache, 64, &probed).unwrap();
        assert!(!b.from_cache, "a different probe count is a different key");
        assert_eq!(b.probes_of("h1").map(<[f32]>::len), Some(16));
        let again =
            ensure_activation_stats(&inc, tmp.path(), &corpus, &cache, 64, &probed).unwrap();
        assert!(again.from_cache);
        assert_eq!(
            again.probes, b.probes,
            "probes survive the cache round trip"
        );
    }

    /// A converged scan waits for its probes: the slots are spread over the
    /// whole sampled plan, so stopping at the adaptive floor would leave the
    /// signatures describing the corpus's opening records only.
    #[test]
    fn adaptive_stopping_waits_for_the_whole_probe_set() {
        let (tmp, inc, corpus) = setup(&vec![1.0f32; 20_000], 0.0, 1.0);
        let spec = SampleSpec {
            max_records: 10_000,
            block_records: 500,
            min_records: 1_000,
            target_rel_se: 0.01,
            probes: 32,
        };
        let probing = compute_activation_stats(
            &inc.creature,
            &inc.checksum,
            tmp.path(),
            &corpus,
            500,
            &spec,
        )
        .unwrap();
        // The last slot sits near the end of the plan, so the whole plan is
        // scanned and the probe set is complete rather than truncated.
        assert_eq!(probing.probes_of("h1").map(<[f32]>::len), Some(32));
        assert_eq!(probing.record_count, 10_000);

        // A control run keeps the early stop it always had: no probes are
        // asked for, so nothing holds the scan open.
        let control = compute_activation_stats(
            &inc.creature,
            &inc.checksum,
            tmp.path(),
            &corpus,
            500,
            &SampleSpec { probes: 0, ..spec },
        )
        .unwrap();
        assert!(control.stopped_early);
        assert_eq!(control.record_count, 1_000);
    }

    #[test]
    fn memory_is_bounded_by_hidden_neuron_count() {
        let (tmp, inc, corpus) =
            setup(&(0..10_000).map(|i| i as f32).collect::<Vec<_>>(), 0.0, 1.0);
        let stats = compute_activation_stats(
            &inc.creature,
            &inc.checksum,
            tmp.path(),
            &corpus,
            512,
            &SampleSpec::full(),
        )
        .unwrap();
        assert_eq!(stats.neurons.len(), 1);
        assert_eq!(stats.record_count, 10_000);
        assert_eq!(stats.neurons[0].count, 10_000);
    }
}
