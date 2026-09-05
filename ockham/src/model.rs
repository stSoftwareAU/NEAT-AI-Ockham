//! Learned candidate ranker — a tiny, transparent logistic model (#107).
//!
//! Ockham's own telemetry ([`crate::telemetry`]) records one feature vector and
//! one outcome per candidate the scorer judged. This fits a **logistic
//! regression** over those rows, offline, and predicts
//! `P(the full scorer confirms this cut)` for the next run's sweep.
//!
//! Small and transparent on purpose: fifteen named coefficients and a bias,
//! written to JSON, readable by a human. There is no hidden state, no RNG and
//! no ordering dependence — full-batch gradient descent from a zero start, so
//! the same rows and the same hyper-parameters produce the same model on every
//! host and every rerun.
//!
//! **The model only ranks.** It decides which neuron is tested sooner and
//! nothing else: every candidate it promotes still passes `creature.validate()`,
//! the sampled screen and full-corpus scoring, and only that scorer accepts a
//! cut. Nothing in this module is consulted when a prune is applied.
//!
//! The model is versioned against the feature schema it was fitted on. A model
//! whose feature names differ from [`crate::features::FEATURE_NAMES`] is
//! **refused at load**, rather than read column-by-column against a schema that
//! has since moved.

use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::features::{CandidateFeatures, FEATURE_NAMES};
use crate::priority::logistic;

/// Current model format version.
pub const PRIORITY_MODEL_FORMAT_VERSION: u32 = 1;

/// Default gradient-descent passes over the training rows.
pub const DEFAULT_EPOCHS: usize = 2_000;
/// Default learning rate.
pub const DEFAULT_LEARNING_RATE: f64 = 0.1;
/// Default L2 penalty — keeps a coefficient from running away on thin data.
pub const DEFAULT_L2: f64 = 1e-3;

/// One training example: a feature vector and what the scorer decided.
#[derive(Debug, Clone, PartialEq)]
pub struct TrainingRow {
    /// Feature values in [`FEATURE_NAMES`] order.
    pub features: Vec<f64>,
    /// Whether the full corpus confirmed the cut.
    pub win: bool,
}

/// Hyper-parameters and provenance of one fit.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TrainingConfig {
    /// Gradient-descent passes.
    pub epochs: usize,
    /// Learning rate.
    pub learning_rate: f64,
    /// L2 penalty.
    pub l2: f64,
    /// Corpus identities the training rows came from, for reproducibility.
    pub corpora: Vec<String>,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            epochs: DEFAULT_EPOCHS,
            learning_rate: DEFAULT_LEARNING_RATE,
            l2: DEFAULT_L2,
            corpora: Vec::new(),
        }
    }
}

/// What a fit was built from — enough to reproduce it exactly.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TrainingMeta {
    /// Crate version that fitted the model.
    pub crate_version: String,
    /// Training rows.
    pub rows: usize,
    /// Rows the full corpus confirmed.
    pub wins: usize,
    /// Hyper-parameters and source corpora.
    pub config: TrainingConfig,
}

/// Fitted logistic ranker over [`FEATURE_NAMES`].
///
/// Fields are private so a model can only reach [`Self::probability`] through
/// [`Self::fit`] or [`Self::load`], both of which validate it against the
/// current feature schema — the version, the feature names, the widths and the
/// finiteness of every coefficient. A silently mis-shaped model would rank by
/// whatever the columns happened to line up with.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PriorityModel {
    format_version: u32,
    features: Vec<String>,
    mean: Vec<f64>,
    scale: Vec<f64>,
    weights: Vec<f64>,
    bias: f64,
    training: TrainingMeta,
}

/// Held-out quality of a model, reported by the trainer.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Evaluation {
    /// Rows evaluated.
    pub rows: usize,
    /// Rows the full corpus confirmed.
    pub wins: usize,
    /// Mean negative log-likelihood — lower is better.
    pub log_loss: f64,
    /// Fraction classified correctly at `p > 0.5`.
    pub accuracy: f64,
    /// Probability a random win outranks a random loss; `0.5` is a coin toss.
    ///
    /// This is the number that matters: the model is used to **order**
    /// candidates, so its ranking quality is what decides whether it earns its
    /// place against the random control.
    pub auc: f64,
}

impl PriorityModel {
    /// Fit a model to `rows`. Deterministic: no RNG, no ordering dependence.
    ///
    /// Errors rather than returning a model that cannot rank: no rows, a row
    /// whose width does not match the feature schema, a non-finite value, or
    /// training data with only one outcome in it — a fit on one class predicts
    /// the same number for every candidate, which is the random control wearing
    /// a model's name.
    pub fn fit(rows: &[TrainingRow], config: TrainingConfig) -> Result<Self, String> {
        let width = FEATURE_NAMES.len();
        if rows.is_empty() {
            return Err("cannot fit a priority model: no training rows".into());
        }
        for (i, row) in rows.iter().enumerate() {
            if row.features.len() != width {
                return Err(format!(
                    "training row {i} has {} features; the schema has {width}",
                    row.features.len()
                ));
            }
            if let Some(bad) = row.features.iter().position(|v| !v.is_finite()) {
                return Err(format!(
                    "training row {i} feature `{}` is not finite",
                    FEATURE_NAMES[bad]
                ));
            }
        }
        let wins = rows.iter().filter(|r| r.win).count();
        if wins == 0 || wins == rows.len() {
            return Err(format!(
                "cannot fit a priority model: {wins} win(s) in {} row(s) — both outcomes are \
                 needed to learn a ranking",
                rows.len()
            ));
        }
        if config.epochs == 0 {
            return Err("cannot fit a priority model: --epochs must be > 0".into());
        }
        if !(config.learning_rate.is_finite() && config.learning_rate > 0.0) {
            return Err("cannot fit a priority model: --learning-rate must be > 0".into());
        }
        if !(config.l2.is_finite() && config.l2 >= 0.0) {
            return Err("cannot fit a priority model: --l2 must be >= 0".into());
        }

        let n = rows.len() as f64;
        let mut mean = vec![0.0; width];
        for row in rows {
            for (m, v) in mean.iter_mut().zip(&row.features) {
                *m += v / n;
            }
        }
        let mut scale = vec![0.0; width];
        for row in rows {
            for ((s, v), m) in scale.iter_mut().zip(&row.features).zip(&mean) {
                *s += (v - m).powi(2) / n;
            }
        }
        // A constant column carries no information; a scale of 1 keeps it at
        // zero after standardisation instead of dividing by nothing.
        for s in &mut scale {
            *s = s.sqrt();
            if *s < 1e-12 {
                *s = 1.0;
            }
        }
        let standardised: Vec<Vec<f64>> = rows
            .iter()
            .map(|row| standardise(&row.features, &mean, &scale))
            .collect();
        let labels: Vec<f64> = rows.iter().map(|r| if r.win { 1.0 } else { 0.0 }).collect();

        let mut weights = vec![0.0; width];
        let mut bias = 0.0;
        for _ in 0..config.epochs {
            let mut grad = vec![0.0; width];
            let mut grad_bias = 0.0;
            for (x, y) in standardised.iter().zip(&labels) {
                let error = logistic(dot(&weights, x) + bias) - y;
                for (g, v) in grad.iter_mut().zip(x) {
                    *g += error * v / n;
                }
                grad_bias += error / n;
            }
            for (w, g) in weights.iter_mut().zip(&grad) {
                *w -= config.learning_rate * (g + config.l2 * *w);
            }
            bias -= config.learning_rate * grad_bias;
        }

        let model = Self {
            format_version: PRIORITY_MODEL_FORMAT_VERSION,
            features: FEATURE_NAMES.iter().map(|s| (*s).to_string()).collect(),
            mean,
            scale,
            weights,
            bias,
            training: TrainingMeta {
                crate_version: crate::crate_version().to_string(),
                rows: rows.len(),
                wins,
                config,
            },
        };
        model.validate()?;
        Ok(model)
    }

    /// `P(the full scorer confirms this cut)` — a ranking key, not a permission.
    pub fn probability(&self, f: &CandidateFeatures) -> f64 {
        self.probability_of(&f.vector())
    }

    /// [`Self::probability`] for a raw vector in [`FEATURE_NAMES`] order.
    ///
    /// The width is fixed by the schema and checked when the model is built or
    /// loaded, so a mis-shaped model can never reach here.
    pub fn probability_of(&self, features: &[f64]) -> f64 {
        let z = dot(
            &self.weights,
            &standardise(features, &self.mean, &self.scale),
        ) + self.bias;
        logistic(z)
    }

    /// Ranking and calibration quality of this model on `rows`.
    pub fn evaluate(&self, rows: &[TrainingRow]) -> Evaluation {
        let mut log_loss = 0.0;
        let mut correct = 0usize;
        let mut wins: Vec<f64> = Vec::new();
        let mut losses: Vec<f64> = Vec::new();
        for row in rows {
            let p = self.probability_of(&row.features).clamp(1e-12, 1.0 - 1e-12);
            log_loss -= if row.win { p.ln() } else { (1.0 - p).ln() };
            if (p > 0.5) == row.win {
                correct += 1;
            }
            if row.win { &mut wins } else { &mut losses }.push(p);
        }
        let n = rows.len().max(1) as f64;
        Evaluation {
            rows: rows.len(),
            wins: wins.len(),
            log_loss: log_loss / n,
            accuracy: correct as f64 / n,
            auc: auc(&wins, &losses),
        }
    }

    /// Coefficients by feature name — the whole model, readable by a human.
    pub fn coefficients(&self) -> Vec<(String, f64)> {
        self.features
            .iter()
            .cloned()
            .zip(self.weights.iter().copied())
            .collect()
    }

    /// Intercept of the fitted model.
    pub fn bias(&self) -> f64 {
        self.bias
    }

    /// What this model was fitted from.
    pub fn training(&self) -> &TrainingMeta {
        &self.training
    }

    /// Format version the model was written with.
    pub fn format_version(&self) -> u32 {
        self.format_version
    }

    /// Refuse a model that does not match the current schema.
    fn validate(&self) -> Result<(), String> {
        if self.format_version != PRIORITY_MODEL_FORMAT_VERSION {
            return Err(format!(
                "priority model format version {} is not {PRIORITY_MODEL_FORMAT_VERSION}; \
                 refit it with `train-ordering`",
                self.format_version
            ));
        }
        if self.features.len() != FEATURE_NAMES.len()
            || self.features.iter().zip(FEATURE_NAMES).any(|(a, b)| a != b)
        {
            return Err(format!(
                "priority model was fitted on features {:?}; this build ranks on {FEATURE_NAMES:?} \
                 — refit it with `train-ordering`",
                self.features
            ));
        }
        if self.mean.len() != self.features.len()
            || self.scale.len() != self.features.len()
            || self.weights.len() != self.features.len()
        {
            return Err(
                "priority model is mis-shaped: mean, scale and weights must each carry one \
                 entry per feature"
                    .into(),
            );
        }
        if !self.bias.is_finite()
            || !self.weights.iter().all(|w| w.is_finite())
            || !self.mean.iter().all(|m| m.is_finite())
            || !self.scale.iter().all(|s| s.is_finite() && *s != 0.0)
        {
            return Err("priority model carries a non-finite coefficient".into());
        }
        Ok(())
    }

    /// Read and validate a model from `path`.
    pub fn load(path: &Path) -> Result<Self, String> {
        let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let model: Self =
            serde_json::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))?;
        model
            .validate()
            .map_err(|e| format!("{}: {e}", path.display()))?;
        Ok(model)
    }

    /// Write the model to `path` as pretty JSON.
    pub fn save(&self, path: &Path) -> Result<(), String> {
        if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
            std::fs::create_dir_all(parent).map_err(|e| format!("{}: {e}", parent.display()))?;
        }
        let json = serde_json::to_string_pretty(self).map_err(|e| e.to_string())?;
        std::fs::write(path, json).map_err(|e| format!("{}: {e}", path.display()))
    }
}

/// `(x - mean) / scale`, column by column.
fn standardise(features: &[f64], mean: &[f64], scale: &[f64]) -> Vec<f64> {
    features
        .iter()
        .zip(mean)
        .zip(scale)
        .map(|((x, m), s)| (x - m) / s)
        .collect()
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Mann–Whitney AUC: `P(win scores above loss)`, ties counting a half.
///
/// `0.5` when either class is empty — no evidence of ranking skill either way,
/// which is exactly what a one-class holdout carries.
fn auc(wins: &[f64], losses: &[f64]) -> f64 {
    if wins.is_empty() || losses.is_empty() {
        return 0.5;
    }
    let mut better = 0.0;
    for w in wins {
        for l in losses {
            better += match w.partial_cmp(l) {
                Some(std::cmp::Ordering::Greater) => 1.0,
                Some(std::cmp::Ordering::Equal) => 0.5,
                _ => 0.0,
            };
        }
    }
    better / (wins.len() * losses.len()) as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Rows where quietness (`logMeanAbs`, index 2) decides the outcome.
    fn separable_rows() -> Vec<TrainingRow> {
        let mut rows = Vec::new();
        for i in 0..40 {
            let quiet = i % 2 == 0;
            let mut features = vec![0.0; FEATURE_NAMES.len()];
            features[0] = 1.0; // measured
            features[2] = if quiet { 0.01 } else { 2.0 } + i as f64 * 1e-4;
            features[9] = 2.0; // cascadeGrowthUnits
            rows.push(TrainingRow {
                features,
                win: quiet,
            });
        }
        rows
    }

    fn features_with_mean_abs(mean_abs: f64) -> CandidateFeatures {
        CandidateFeatures {
            measured: true,
            mean_abs,
            outgoing_weight: 0.0,
            cascade_growth_units: 2.0,
            ..CandidateFeatures::default()
        }
    }

    #[test]
    fn a_fitted_model_ranks_the_signal_it_was_trained_on() {
        let model = PriorityModel::fit(&separable_rows(), TrainingConfig::default()).unwrap();
        let quiet = model.probability(&features_with_mean_abs(0.01_f64.exp_m1()));
        let loud = model.probability(&features_with_mean_abs(2.0_f64.exp_m1()));
        assert!(quiet > loud, "quiet {quiet} loud {loud}");
        assert!((0.0..=1.0).contains(&quiet), "{quiet}");
    }

    #[test]
    fn the_same_rows_and_hyper_parameters_reproduce_the_same_model() {
        let a = PriorityModel::fit(&separable_rows(), TrainingConfig::default()).unwrap();
        let b = PriorityModel::fit(&separable_rows(), TrainingConfig::default()).unwrap();
        assert_eq!(a, b, "fitting must be deterministic");
    }

    #[test]
    fn evaluation_reports_ranking_quality() {
        let rows = separable_rows();
        let model = PriorityModel::fit(&rows, TrainingConfig::default()).unwrap();
        let eval = model.evaluate(&rows);
        assert_eq!(eval.rows, rows.len());
        assert_eq!(eval.wins, rows.len() / 2);
        assert!(eval.auc > 0.9, "{eval:?}");
        assert!(eval.accuracy > 0.9, "{eval:?}");
        assert!(eval.log_loss < 0.7, "{eval:?}");
    }

    #[test]
    fn one_class_training_data_is_refused_rather_than_fitted() {
        let mut rows = separable_rows();
        for row in &mut rows {
            row.win = true;
        }
        let err = PriorityModel::fit(&rows, TrainingConfig::default()).unwrap_err();
        assert!(err.contains("both outcomes"), "{err}");
        let empty = PriorityModel::fit(&[], TrainingConfig::default()).unwrap_err();
        assert!(empty.contains("no training rows"), "{empty}");
    }

    #[test]
    fn a_mis_shaped_or_non_finite_row_is_refused() {
        let mut rows = separable_rows();
        rows[0].features.push(1.0);
        let err = PriorityModel::fit(&rows, TrainingConfig::default()).unwrap_err();
        assert!(err.contains("features"), "{err}");
        let mut nan = separable_rows();
        nan[3].features[2] = f64::NAN;
        let err = PriorityModel::fit(&nan, TrainingConfig::default()).unwrap_err();
        assert!(err.contains("not finite"), "{err}");
    }

    #[test]
    fn bad_hyper_parameters_name_the_flag() {
        for (config, needle) in [
            (
                TrainingConfig {
                    epochs: 0,
                    ..Default::default()
                },
                "--epochs",
            ),
            (
                TrainingConfig {
                    learning_rate: 0.0,
                    ..Default::default()
                },
                "--learning-rate",
            ),
            (
                TrainingConfig {
                    l2: -1.0,
                    ..Default::default()
                },
                "--l2",
            ),
        ] {
            let err = PriorityModel::fit(&separable_rows(), config).unwrap_err();
            assert!(err.contains(needle), "{err}");
        }
    }

    #[test]
    fn a_model_round_trips_through_a_file() {
        let dir = std::env::temp_dir().join(format!("ockham-model-{}", std::process::id()));
        let path = dir.join("model.json");
        let model = PriorityModel::fit(&separable_rows(), TrainingConfig::default()).unwrap();
        model.save(&path).unwrap();
        let read = PriorityModel::load(&path).unwrap();
        assert_eq!(read, model);
        assert_eq!(read.format_version(), PRIORITY_MODEL_FORMAT_VERSION);
        assert_eq!(read.coefficients().len(), FEATURE_NAMES.len());
        assert_eq!(read.training().rows, 40);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_model_fitted_on_another_schema_is_refused_at_load() {
        let dir = std::env::temp_dir().join(format!("ockham-stale-{}", std::process::id()));
        let path = dir.join("stale.json");
        let model = PriorityModel::fit(&separable_rows(), TrainingConfig::default()).unwrap();
        let mut json: serde_json::Value =
            serde_json::from_str(&serde_json::to_string(&model).unwrap()).unwrap();
        json["features"][1] = serde_json::json!("logSomethingElse");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(&path, json.to_string()).unwrap();
        let err = PriorityModel::load(&path).unwrap_err();
        assert!(err.contains("refit it"), "{err}");

        json["features"][1] = serde_json::json!("logVariance");
        json["formatVersion"] = serde_json::json!(PRIORITY_MODEL_FORMAT_VERSION + 1);
        std::fs::write(&path, json.to_string()).unwrap();
        let err = PriorityModel::load(&path).unwrap_err();
        assert!(err.contains("format version"), "{err}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn an_unwritable_model_path_names_the_file() {
        let dir = std::env::temp_dir().join(format!("ockham-save-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let blocker = dir.join("blocked");
        std::fs::write(&blocker, "not a directory").unwrap();
        let model = PriorityModel::fit(&separable_rows(), TrainingConfig::default()).unwrap();
        let err = model.save(&blocker.join("model.json")).unwrap_err();
        assert!(err.contains("blocked"), "{err}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn an_unreadable_model_path_names_the_file() {
        let err = PriorityModel::load(Path::new("/nonexistent/ockham-model.json")).unwrap_err();
        assert!(err.contains("ockham-model.json"), "{err}");
    }

    #[test]
    fn auc_is_a_half_when_a_class_is_missing() {
        assert_eq!(auc(&[], &[0.1]), 0.5);
        assert_eq!(auc(&[0.9], &[]), 0.5);
        assert_eq!(auc(&[0.9], &[0.1]), 1.0);
        assert_eq!(auc(&[0.5], &[0.5]), 0.5);
    }
}
