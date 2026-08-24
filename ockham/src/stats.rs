//! Full-corpus hidden-neuron activation statistics (Issue #3).
//!
//! Statistics **propose** candidates only. They are not a proxy acceptance
//! score and must not be presented as proof that a neuron is unimportant.
//!
//! Accumulation uses the same NEAT-AI-core compiled forward pass as scoring
//! (`CompiledNetwork::activate`), with `f64` running moments so a long corpus
//! does not lose the mean to `f32` rounding. Per-record activations are not
//! retained: memory is one compiled network plus one accumulator per hidden
//! neuron.

use std::path::{Path, PathBuf};
use std::time::Instant;

use neat_core::training_data::TrainingDataConfig;
use neat_core::{CreatureExport, compile_creature};
use serde::{Deserialize, Serialize};

use crate::corpus::{CorpusInfo, for_each_chunk};
use crate::incumbent::Incumbent;

/// Cache / on-disk format version. Bump when the JSON shape changes.
pub const STATS_FORMAT_VERSION: u32 = 1;
/// Default records per streaming chunk.
pub const DEFAULT_CHUNK_RECORDS: usize = 4096;

/// One hidden neuron's full-corpus post-activation summary.
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

/// Full-corpus hidden-neuron statistics for one incumbent + corpus.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ActivationStats {
    /// [`STATS_FORMAT_VERSION`].
    pub format_version: u32,
    /// Incumbent checksum the stats were measured against.
    pub creature_checksum: String,
    /// Corpus identity the stats were measured against.
    pub corpus_identity: String,
    /// Records streamed.
    pub record_count: u64,
    /// Wall time of the scan (ms), excluding cache hits.
    pub scan_ms: u64,
    /// Per-hidden-neuron summaries.
    pub neurons: Vec<NeuronStats>,
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
}

impl Accumulator {
    fn new(uuid: String, neuron_index: usize, activation_index: usize) -> Self {
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

    fn finish(self) -> NeuronStats {
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
            uuid: self.uuid,
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

/// Cache path keyed by format version, creature checksum and corpus identity.
pub fn cache_path(dir: &Path, checksum: &str, corpus_identity: &str) -> PathBuf {
    dir.join(format!(
        "activation-stats.v{STATS_FORMAT_VERSION}.{checksum}.{corpus_identity}.json"
    ))
}

/// Load a cache file only when every key matches.
pub fn load_cached_stats(
    path: &Path,
    checksum: &str,
    corpus_identity: &str,
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

/// Stream the corpus through NEAT-AI-core inference and accumulate hidden stats.
pub fn compute_activation_stats(
    creature: &CreatureExport,
    creature_checksum: &str,
    training_dir: &Path,
    corpus: &CorpusInfo,
    chunk_records: usize,
) -> Result<ActivationStats, String> {
    let mut net = compile_creature(creature).map_err(|e| e.to_string())?;
    let mut acc: Vec<Accumulator> = creature
        .neurons
        .iter()
        .enumerate()
        .filter(|(_, n)| n.neuron_type == "hidden")
        .map(|(i, n)| Accumulator::new(n.uuid.clone(), i, net.num_inputs + i))
        .collect();
    let cfg = TrainingDataConfig::new(creature.input, creature.output);
    let started = Instant::now();
    let streamed = for_each_chunk(training_dir, &cfg, chunk_records.max(1), |chunk| {
        for r in 0..chunk.records {
            let inputs = &chunk.inputs[r * creature.input..(r + 1) * creature.input];
            let _ = net.activate(inputs, creature.output);
            for a in &mut acc {
                a.push(net.activations[a.activation_index]);
            }
        }
        Ok(())
    })?;
    let scan_ms = started.elapsed().as_millis() as u64;
    if streamed != corpus.record_count {
        return Err(format!(
            "activation scan saw {streamed} records but corpus identity has {}",
            corpus.record_count
        ));
    }
    Ok(ActivationStats {
        format_version: STATS_FORMAT_VERSION,
        creature_checksum: creature_checksum.to_string(),
        corpus_identity: corpus.identity.clone(),
        record_count: streamed,
        scan_ms,
        neurons: acc.into_iter().map(Accumulator::finish).collect(),
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
) -> Result<ActivationStats, String> {
    let path = cache_path(cache_dir, &incumbent.checksum, &corpus.identity);
    if let Some(cached) = load_cached_stats(&path, &incumbent.checksum, &corpus.identity)? {
        return Ok(cached);
    }
    let stats = compute_activation_stats(
        &incumbent.creature,
        &incumbent.checksum,
        training_dir,
        corpus,
        chunk_records,
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
        let stats =
            compute_activation_stats(&inc.creature, &inc.checksum, tmp.path(), &corpus, 2).unwrap();
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
        let again =
            compute_activation_stats(&inc.creature, &inc.checksum, tmp.path(), &corpus, 1).unwrap();
        let h2 = again.by_uuid("h1").unwrap();
        assert!(close(h.mean, h2.mean) && close(h.variance, h2.variance));
    }

    #[test]
    fn cache_is_not_reused_for_a_changed_creature_or_corpus() {
        let (tmp, inc, corpus) = setup(&[1.0, 2.0, 3.0], 0.0, 1.0);
        let cache = tmp.path().join("cache");
        let a = ensure_activation_stats(&inc, tmp.path(), &corpus, &cache, 8).unwrap();
        assert!(!a.from_cache);
        let b = ensure_activation_stats(&inc, tmp.path(), &corpus, &cache, 8).unwrap();
        assert!(b.from_cache);
        assert_eq!(a.neurons, b.neurons);

        let other = Incumbent::from_creature(hidden_identity_creature(1.0, 1.0), "u").unwrap();
        let path = cache_path(&cache, &other.checksum, &corpus.identity);
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
            load_cached_stats(&path, &other.checksum, &corpus.identity)
                .unwrap()
                .is_none()
        );

        write_bin_file(&tmp.path().join("0.bin"), &[(vec![9.0f32], vec![9.0f32])]).unwrap();
        let corpus2 = corpus_info(tmp.path(), &TrainingDataConfig::new(1, 1)).unwrap();
        assert_ne!(corpus.identity, corpus2.identity);
        let c = ensure_activation_stats(&inc, tmp.path(), &corpus2, &cache, 8).unwrap();
        assert!(!c.from_cache);
        assert_eq!(c.record_count, 1);
        assert_ne!(c.neurons[0].mean, a.neurons[0].mean);
    }

    #[test]
    fn memory_is_bounded_by_hidden_neuron_count() {
        let (tmp, inc, corpus) =
            setup(&(0..10_000).map(|i| i as f32).collect::<Vec<_>>(), 0.0, 1.0);
        let stats =
            compute_activation_stats(&inc.creature, &inc.checksum, tmp.path(), &corpus, 512)
                .unwrap();
        assert_eq!(stats.neurons.len(), 1);
        assert_eq!(stats.record_count, 10_000);
        assert_eq!(stats.neurons[0].count, 10_000);
    }
}
