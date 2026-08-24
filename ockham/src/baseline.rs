//! Authoritative baseline (Issue #2).
//!
//! Before any pruning, the incumbent is scored by NEAT-AI-scorer on the full
//! corpus. The resulting `score` is the number every later candidate must beat.
//! Larger is better. Scorer failure, malformed output or checksum/record drift
//! aborts the run (fail closed). Ockham-local inference is never an acceptance
//! authority.

use std::path::Path;
use std::time::Instant;

use serde::{Deserialize, Serialize};

use crate::corpus::CorpusInfo;
use crate::incumbent::Incumbent;
use crate::scorer::{DirectoryScorer, ScorerMode};

/// Scorer-verified baseline for one incumbent.
///
/// `score` is the acceptance metric: **larger is better**.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AuthoritativeBaseline {
    /// Incumbent checksum.
    pub incumbent_checksum: String,
    /// Authoritative score (larger is better).
    pub score: f64,
    /// Authoritative error.
    pub error: f64,
    /// Complexity / growth-cost penalty the scorer applied.
    pub complexity_penalty: f64,
    /// Listed (non-input) neuron count.
    pub neurons: usize,
    /// Hidden neuron count.
    pub hidden_neurons: usize,
    /// Synapse count.
    pub synapses: usize,
    /// Records the scorer saw.
    pub record_count: u64,
    /// Scorer identity (binary digest).
    pub scorer_identity: String,
    /// Extra scorer arguments used for this baseline.
    pub scorer_args: Vec<String>,
    /// Cost function the scorer reported.
    pub cost_name: Option<String>,
    /// Scorer backend label.
    pub scorer_backend: Option<String>,
    /// Corpus identity.
    pub corpus_identity: String,
    /// Corpus record count (must equal `record_count` when the scorer reports one).
    pub corpus_record_count: u64,
    /// Scorer wall time (ms).
    pub scorer_ms: u64,
    /// Unix seconds.
    pub created_at_unix: u64,
    /// Version of Ockham that recorded the baseline.
    pub ockham_version: String,
}

/// Score the incumbent alone and record the authoritative baseline.
///
/// Writes `workspace/baseline-score/baseline.json` (the creature) for the
/// scorer, then `workspace/baseline.json` (this record).
pub fn establish_baseline(
    incumbent: &Incumbent,
    training_dir: &Path,
    corpus: &CorpusInfo,
    scorer: &dyn DirectoryScorer,
    scorer_args: &[String],
    workspace: &Path,
) -> Result<AuthoritativeBaseline, String> {
    let dir = workspace.join("baseline-score");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    let compact = neat_core::creature_to_json(&incumbent.creature).map_err(|e| e.to_string())?;
    std::fs::write(dir.join("baseline.json"), compact).map_err(|e| e.to_string())?;
    let started = Instant::now();
    let results = scorer
        .score_directory(&dir, training_dir, ScorerMode::Full)
        .map_err(|e| format!("baseline: {e}"))?;
    let scorer_ms = started.elapsed().as_millis() as u64;
    let _ = std::fs::remove_dir_all(&dir);
    let result = results
        .get("baseline")
        .ok_or("baseline: scorer returned no `baseline` entry")?;
    if results.len() != 1 {
        return Err(format!(
            "baseline: scorer returned {} entries for a single creature",
            results.len()
        ));
    }
    if !result.score.is_finite() || !result.error.is_finite() {
        return Err("baseline: scorer returned a non-finite score or error".into());
    }
    if result.record_count != 0 && result.record_count != corpus.record_count {
        return Err(format!(
            "baseline: scorer saw {} records but the corpus has {} — refusing to continue",
            result.record_count, corpus.record_count
        ));
    }
    let baseline = AuthoritativeBaseline {
        incumbent_checksum: incumbent.checksum.clone(),
        score: result.score,
        error: result.error,
        complexity_penalty: result.complexity_penalty,
        neurons: incumbent.creature.neurons.len(),
        hidden_neurons: incumbent.hidden_neurons(),
        synapses: incumbent.creature.synapses.len(),
        record_count: if result.record_count == 0 {
            corpus.record_count
        } else {
            result.record_count
        },
        scorer_identity: scorer.identity(),
        scorer_args: scorer_args.to_vec(),
        cost_name: result.cost_name.clone(),
        scorer_backend: result.gpu_backend.clone(),
        corpus_identity: corpus.identity.clone(),
        corpus_record_count: corpus.record_count,
        scorer_ms,
        created_at_unix: crate::incumbent::now_unix(),
        ockham_version: env!("CARGO_PKG_VERSION").to_string(),
    };
    let json = serde_json::to_string_pretty(&baseline).map_err(|e| e.to_string())?;
    std::fs::write(workspace.join("baseline.json"), json).map_err(|e| e.to_string())?;
    Ok(baseline)
}

/// In-process fake scorers for tests.
pub mod fake {
    use std::cell::{Cell, RefCell};
    use std::collections::BTreeMap;
    use std::path::Path;

    use crate::scorer::{DirectoryScorer, ScoreResult, ScorerError, ScorerMode};

    /// Scripted directory scorer: fixed result, spawn-style failure, or raw output.
    #[derive(Default)]
    pub struct ScriptedScorer {
        /// When set, every call fails with this message.
        pub fail_with: Option<String>,
        /// When set, the output is this raw string (to simulate malformed output).
        pub raw_output: Option<String>,
        /// Score returned for `baseline` when neither failure mode is set.
        pub baseline_score: f64,
        /// Error returned for `baseline` when neither failure mode is set.
        pub baseline_error: f64,
        /// Record count claimed for `baseline`.
        pub record_count: u64,
        /// Score returned for every non-`baseline` stem when set.
        pub candidate_score: Option<f64>,
        /// Per-stem score overrides (including `baseline`).
        pub stem_scores: BTreeMap<String, f64>,
        /// Last scorer mode observed (tests).
        pub last_mode: Cell<Option<ScorerMode>>,
        /// Last file stems scored in one call (tests).
        pub last_stems: RefCell<Vec<String>>,
    }

    impl ScriptedScorer {
        /// A successful fake that reports `score` / `error`.
        pub fn ok(score: f64, error: f64) -> Self {
            Self {
                baseline_score: score,
                baseline_error: error,
                record_count: 0,
                ..Self::default()
            }
        }
    }

    impl DirectoryScorer for ScriptedScorer {
        fn score_directory(
            &self,
            creature_dir: &Path,
            _training_dir: &Path,
            mode: ScorerMode,
        ) -> Result<BTreeMap<String, ScoreResult>, ScorerError> {
            self.last_mode.set(Some(mode));
            if let Some(m) = &self.fail_with {
                return Err(ScorerError::Failed {
                    status: "exit 1".into(),
                    stderr: m.clone(),
                });
            }
            if let Some(raw) = &self.raw_output {
                return crate::scorer::parse_scorer_output(raw);
            }
            let mut stems = vec!["baseline".to_string()];
            if creature_dir.is_dir() {
                let mut extra = Vec::new();
                if let Ok(entries) = std::fs::read_dir(creature_dir) {
                    for entry in entries.flatten() {
                        let path = entry.path();
                        if path.extension().and_then(|e| e.to_str()) != Some("json") {
                            continue;
                        }
                        if let Some(stem) = path.file_stem().and_then(|s| s.to_str())
                            && stem != "baseline"
                        {
                            extra.push(stem.to_string());
                        }
                    }
                }
                extra.sort();
                stems.extend(extra);
            }
            self.last_stems.replace(stems.clone());
            let mut out = BTreeMap::new();
            for stem in stems {
                let score = self
                    .stem_scores
                    .get(&stem)
                    .copied()
                    .or_else(|| {
                        if stem == "baseline" {
                            Some(self.baseline_score)
                        } else {
                            self.candidate_score
                        }
                    })
                    .unwrap_or(self.baseline_score);
                out.insert(
                    stem,
                    ScoreResult {
                        score,
                        error: self.baseline_error,
                        complexity_penalty: 1e-8,
                        record_count: self.record_count,
                        sample_rate: match mode {
                            ScorerMode::Sample { rate, .. } => Some(rate),
                            ScorerMode::Full => None,
                        },
                        gpu_backend: Some("fake".into()),
                        cost_name: Some("MSE".into()),
                        time_taken: 0.0,
                    },
                );
            }
            Ok(out)
        }

        fn identity(&self) -> String {
            "fake:scripted".into()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::fake::ScriptedScorer;
    use super::*;
    use crate::corpus::{corpus_info, write_bin_file};
    use crate::fixtures::identity_creature;
    use crate::incumbent::Incumbent;
    use neat_core::training_data::TrainingDataConfig;

    fn setup() -> (tempfile::TempDir, Incumbent, CorpusInfo) {
        let tmp = tempfile::tempdir().unwrap();
        let recs: Vec<(Vec<f32>, Vec<f32>)> = (0..8)
            .map(|i| (vec![i as f32], vec![i as f32 + 0.5]))
            .collect();
        write_bin_file(&tmp.path().join("0.bin"), &recs).unwrap();
        let inc = Incumbent::from_creature(identity_creature(1, 1), "t").unwrap();
        let corpus = corpus_info(tmp.path(), &TrainingDataConfig::new(1, 1)).unwrap();
        (tmp, inc, corpus)
    }

    #[test]
    fn baseline_is_recorded_and_score_is_larger_is_better() {
        let (tmp, inc, corpus) = setup();
        let scorer = ScriptedScorer::ok(0.87, 0.13);
        let ws = tmp.path().join("ws");
        std::fs::create_dir_all(&ws).unwrap();
        let b = establish_baseline(&inc, tmp.path(), &corpus, &scorer, &[], &ws).unwrap();
        assert_eq!(b.score, 0.87);
        assert_eq!(b.error, 0.13);
        assert_eq!(b.record_count, 8);
        assert_eq!(b.synapses, 1);
        assert!(ws.join("baseline.json").exists());
        assert!(!ws.join("baseline-score").exists());
        let back: AuthoritativeBaseline =
            serde_json::from_str(&std::fs::read_to_string(ws.join("baseline.json")).unwrap())
                .unwrap();
        assert_eq!(back, b);
        assert!(b.score > b.error, "score is larger-is-better, not a loss");
    }

    #[test]
    fn failure_and_malformed_abort() {
        let (tmp, inc, corpus) = setup();
        let ws = tmp.path().join("ws");
        std::fs::create_dir_all(&ws).unwrap();
        let failing = ScriptedScorer {
            fail_with: Some("boom".into()),
            ..Default::default()
        };
        assert!(
            establish_baseline(&inc, tmp.path(), &corpus, &failing, &[], &ws)
                .unwrap_err()
                .contains("boom")
        );
        let malformed = ScriptedScorer {
            raw_output: Some("{not json".into()),
            ..Default::default()
        };
        assert!(
            establish_baseline(&inc, tmp.path(), &corpus, &malformed, &[], &ws)
                .unwrap_err()
                .contains("malformed")
        );
        let wrong_count = ScriptedScorer {
            raw_output: Some(r#"{"baseline":{"score":0.5,"error":0.5,"recordCount":3}}"#.into()),
            ..Default::default()
        };
        assert!(
            establish_baseline(&inc, tmp.path(), &corpus, &wrong_count, &[], &ws)
                .unwrap_err()
                .contains("records")
        );
    }
}
