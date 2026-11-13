//! Pairwise logistic ranking; never consulted by pruning acceptance.

use serde::{Deserialize, Serialize};
use super::{dot, standardise};
use crate::priority::logistic;

/// One individually scored candidate. Larger scores rank first.
#[derive(Debug, Clone)]
pub struct RankedRow {
    /// Named by the experiment manifest's ordered feature schema.
    pub features: Vec<f64>,
    /// Observed score; an absent score must never become a negative label.
    pub score: f64,
}

/// Bounded, deterministic full-batch RankNet optimisation.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct FitConfig {
    /// Optimisation passes.
    pub epochs: usize,
    /// Positive step size.
    pub learning_rate: f64,
    /// Non-negative weight penalty.
    pub l2: f64,
    /// Maximum unequal pairs per task, in supplied row order.
    pub max_pairs: usize,
}

impl FitConfig {
    /// Reject invalid or unbounded research fits.
    pub fn validate(self) -> Result<(), String> {
        if self.epochs == 0 || self.epochs > 2000 || self.max_pairs == 0 || self.max_pairs > 4096
            || !self.learning_rate.is_finite() || self.learning_rate <= 0.0 || self.learning_rate > 1.0
            || !self.l2.is_finite() || self.l2 < 0.0
        {
            return Err("invalid pairwise fit limits".into());
        }
        Ok(())
    }
}

/// Linear pairwise fit using the existing model's logistic and vector operations.
///
/// Independent of the deployed pointwise model format. Serialised only inside
/// experiment results; cannot be loaded by ordinary learned ordering.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PairwiseModel {
    mean: Vec<f64>,
    scale: Vec<f64>,
    weights: Vec<f64>,
}

impl PairwiseModel {
    /// Fit normalisation on explicitly permitted rows, then learn within-task
    /// pairs. Empty or tied supervision leaves a zero-score cold start.
    pub fn fit(tasks: &[Vec<RankedRow>], normalisation: &[RankedRow], width: usize, config: FitConfig) -> Result<Self, String> {
        config.validate()?;
        if width == 0 || width > 64 { return Err("invalid pairwise feature width".into()); }
        let mut model = Self { mean: vec![0.0; width], scale: vec![1.0; width], weights: vec![0.0; width] };
        let n = normalisation.len().max(1) as f64;
        for row in normalisation {
            model.check(&row.features)?;
            if !row.score.is_finite() { return Err("pairwise label is not finite".into()); }
            for (mean, value) in model.mean.iter_mut().zip(&row.features) { *mean += value / n; }
        }
        if !normalisation.is_empty() {
            model.scale.fill(0.0);
            for row in normalisation {
                for ((scale, value), mean) in model.scale.iter_mut().zip(&row.features).zip(&model.mean) { *scale += (value - mean).powi(2) / n; }
            }
            for scale in &mut model.scale { *scale = scale.sqrt(); if *scale < 1e-12 { *scale = 1.0; } }
        }
        model.adapt(tasks, config)?;
        Ok(model)
    }

    /// Continue from pretrained weights without refitting normalisation.
    /// Tasks contribute equal total weight, regardless of pair count.
    pub fn adapt(&mut self, tasks: &[Vec<RankedRow>], config: FitConfig) -> Result<(), String> {
        config.validate()?;
        let mut pairs = Vec::new();
        for task in tasks {
            let mut differences = Vec::new();
            for row in task {
                self.check(&row.features)?;
                if !row.score.is_finite() { return Err("pairwise label is not finite".into()); }
            }
            'pairs: for (i, a) in task.iter().enumerate() {
                for b in &task[i + 1..] {
                    if a.score == b.score { continue; }
                    let sign = if a.score > b.score { 1.0 } else { -1.0 };
                    let diff: Vec<f64> = a.features.iter().zip(&b.features).zip(&self.scale)
                        .map(|((a,b),s)| sign * (a-b) / s).collect();
                    self.check(&diff)?;
                    differences.push(diff);
                    if differences.len() == config.max_pairs { break 'pairs; }
                }
            }
            if !differences.is_empty() { pairs.push(differences); }
        }
        if pairs.is_empty() { return Ok(()); }
        for _ in 0..config.epochs {
            let mut grad = vec![0.0; self.weights.len()];
            for task in &pairs {
                let n = (pairs.len() * task.len()) as f64;
                for difference in task {
                    let error = logistic(dot(&self.weights, difference)) - 1.0;
                    for (g,x) in grad.iter_mut().zip(difference) { *g += error * x / n; }
                }
            }
            for (w,g) in self.weights.iter_mut().zip(grad) { *w -= config.learning_rate * (g + config.l2 * *w); }
        }
        if self.weights.iter().chain(&self.mean).chain(&self.scale).any(|x| !x.is_finite()) {
            return Err("pairwise fit produced a non-finite coefficient".into());
        }
        Ok(())
    }

    fn check(&self, features: &[f64]) -> Result<(), String> {
        if features.len() != self.weights.len() { return Err("pairwise feature width mismatch".into()); }
        if features.iter().any(|x| !x.is_finite()) { return Err("pairwise feature is not finite".into()); }
        Ok(())
    }

    /// Uncalibrated ordering score; never an acceptance probability.
    pub fn score(&self, features: &[f64]) -> Result<f64, String> {
        self.check(features)?;
        let score = dot(&self.weights, &standardise(features, &self.mean, &self.scale));
        if !score.is_finite() { return Err("pairwise prediction is not finite".into()); }
        Ok(score)
    }
}
#[cfg(test)]
mod tests {
    use super::*;

    fn rows(reverse: bool) -> Vec<RankedRow> {
        (0..6).map(|i| RankedRow { features: vec![i as f64, 1.0], score: if reverse { -(i as f64) } else { i as f64 } }).collect()
    }

    #[test]
    fn learns_relative_order_and_target_adaptation_can_reverse_it() {
        let source = rows(false);
        let target = rows(true);
        let config = FitConfig { epochs: 200, learning_rate: 0.1, l2: 0.001, max_pairs: 100 };
        let mut model = PairwiseModel::fit(&[source.clone()], &source, 2, config).unwrap();
        assert!(model.score(&[5.0, 1.0]).unwrap() > model.score(&[0.0, 1.0]).unwrap());
        model.adapt(&[target], config).unwrap();
        assert!(model.score(&[5.0, 1.0]).unwrap() < model.score(&[0.0, 1.0]).unwrap());
    }

    #[test]
    fn cold_start_ties_missing_and_non_finite_inputs_are_explicit() {
        let config = FitConfig { epochs: 20, learning_rate: 0.1, l2: 0.001, max_pairs: 100 };
        let model = PairwiseModel::fit(&[], &[], 2, config).unwrap();
        assert_eq!(model.score(&[5.0, 1.0]).unwrap(), 0.0);
        assert!(model.score(&[1.0]).unwrap_err().contains("width"));
        assert!(model.score(&[f64::NAN, 1.0]).unwrap_err().contains("finite"));
        let invalid = vec![RankedRow { features: vec![f64::INFINITY, 1.0], score: 1.0 }];
        assert!(PairwiseModel::fit(&[invalid.clone()], &invalid, 2, config).unwrap_err().contains("finite"));
        let tied = vec![RankedRow { features: vec![2.0, 1.0], score: 1.0 }; 3];
        let model = PairwiseModel::fit(&[tied.clone()], &tied, 2, config).unwrap();
        assert_eq!(model.score(&[5.0, 1.0]).unwrap(), 0.0);
    }

    #[test]
    fn repeated_fits_are_identical_and_labels_do_not_set_normalisation() {
        let source = rows(false);
        let config = FitConfig { epochs: 20, learning_rate: 0.1, l2: 0.001, max_pairs: 10 };
        let a = PairwiseModel::fit(&[source.clone()], &source, 2, config).unwrap();
        let b = PairwiseModel::fit(&[source.clone()], &source, 2, config).unwrap();
        assert_eq!(a, b);
        assert!(a.score(&[5.0, 1.0]).unwrap() > a.score(&[0.0, 1.0]).unwrap());
    }
}
