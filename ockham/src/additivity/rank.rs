//! Bundle ranking and audit metrics for the additivity experiment (Issue #260).
//!
//! An arm orders a pool of pruning bundles; [`audit`] scores that ordering
//! against full-corpus truth. Nothing here accepts a creature: a ranking only
//! decides which bundles the scorer sees first, and the scorer alone decides
//! what is kept.

use crate::ordering::SplitMix64;
use serde::Serialize;
use std::cmp::Ordering;

/// One way of ordering the bundle pool for evaluation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "camelCase")]
pub enum Arm {
    /// Ockham's ranked prefix first (from `promote::bundle_plans`), then the
    /// remaining pool bundles in emission order.
    Current,
    /// Ordered by the additive loss prediction.
    AdditiveLoss,
    /// Ordered by the cached-output prediction.
    CachedOutput,
    /// A seeded random permutation, the control.
    Random,
}

/// Every arm, in report order.
pub const ARMS: [Arm; 4] = [
    Arm::Current,
    Arm::AdditiveLoss,
    Arm::CachedOutput,
    Arm::Random,
];

impl Arm {
    /// Stable lower-camel name used in reports.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::Current => "current",
            Self::AdditiveLoss => "additiveLoss",
            Self::CachedOutput => "cachedOutput",
            Self::Random => "random",
        }
    }

    /// Whether this arm ranks by a prediction, and so carries prediction audits.
    #[must_use]
    pub fn predicts(self) -> bool {
        matches!(self, Self::AdditiveLoss | Self::CachedOutput)
    }
}

/// The pool in emission order, `0..n`.
///
/// The caller emits Ockham's ranked prefix first, so this order is that prefix
/// followed by the remaining pool bundles in the order they were emitted.
#[must_use]
pub fn rank_current(n: usize) -> Vec<usize> {
    (0..n).collect()
}

/// A seeded Fisher–Yates permutation of `0..n`, reproducible for a given seed.
#[must_use]
pub fn rank_random(n: usize, seed: u64) -> Vec<usize> {
    let mut order: Vec<usize> = (0..n).collect();
    SplitMix64(seed).shuffle(&mut order);
    order
}

/// Orders bundles by predicted improvement, higher first.
///
/// `estimates[i]` is bundle `i`'s predicted improvement, or `None`. A bundle
/// with no estimate, or a non-finite one, keeps its current-order slot `i`:
/// a missing estimate falls back to ordinary evaluation order and is never
/// treated as a loss. The estimated bundles fill the remaining slots in
/// descending estimate, ties broken by lower index.
#[must_use]
pub fn rank_by_estimate(estimates: &[Option<f64>]) -> Vec<usize> {
    let mut slots: Vec<Option<usize>> = vec![None; estimates.len()];
    let mut estimated: Vec<(usize, f64)> = Vec::new();
    for (index, estimate) in estimates.iter().enumerate() {
        match estimate {
            Some(value) if value.is_finite() => estimated.push((index, *value)),
            _ => slots[index] = Some(index),
        }
    }
    estimated.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(Ordering::Equal)
            .then(a.0.cmp(&b.0))
    });
    let mut ranked = estimated.into_iter().map(|(index, _)| index);
    for slot in slots.iter_mut().filter(|slot| slot.is_none()) {
        *slot = ranked.next();
    }
    slots.into_iter().flatten().collect()
}

/// A count out of a denominator, kept whole so a report can show both.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct Rate {
    /// Cases that met the condition.
    pub count: usize,
    /// Cases the condition was counted over.
    pub of: usize,
}

/// How well one arm's ordering agreed with full-corpus truth.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AuditMetrics {
    /// Bundles in the pool.
    pub bundles: usize,
    /// Spearman rank correlation between the ordering and truth.
    pub spearman: Option<f64>,
    /// The `k` used for [`AuditMetrics::top_k_recall`].
    pub top_k: usize,
    /// Share of the true top-`k` that the arm's first `k` contain.
    pub top_k_recall: Option<f64>,
    /// Best truth minus the truth of the arm's first bundle.
    pub regret: Option<f64>,
    /// Predicted wins that were not actual wins, over actual non-wins.
    pub false_promotions: Option<Rate>,
    /// Actual wins the prediction rejected, over actual wins.
    pub false_rejections: Option<Rate>,
    /// Bundles with no usable estimate, excluded from the prediction audits.
    pub unestimated: usize,
}

/// Audits one arm's `ranking` against `truth`.
///
/// `truth[i]` is the full-corpus score of pool bundle `i`, higher better. It
/// drives every ranking metric and nothing else. `ranking` must be a permutation
/// of `0..truth.len()`, and `truth` must be finite. `predicted`, when given,
/// holds one estimate per bundle, in the same units as `actual_gain`.
/// `actual_gain[i]` is bundle `i`'s full-corpus error reduction (parent full
/// error minus bundle full error), the actual side of the prediction audits.
///
/// - `spearman` is the Pearson correlation of `n - 1 - position` against the
///   average-tie ranks of `truth`. It is `None` when `n < 2` or either side
///   has zero variance.
/// - `top_k_recall` is `|arm top-k ∩ true top-k| / min(k, n)`, with the true
///   top-k taken by score descending, ties to the lower index. It is `None`
///   when `n == 0` or `min(k, n) == 0`.
/// - `regret` is the best truth minus `truth[ranking[0]]`, and `None` when
///   `n == 0`.
/// - With `predicted`, a bundle is a predicted win when its estimate is at
///   least `min_improvement`, and an actual win when `actual_gain` is. A
///   `None` or non-finite estimate is excluded and counted in `unestimated`.
///   Non-finite estimates are treated as absent, so a NaN never reads as a
///   rejection.
///
/// # Errors
///
/// When `ranking` is not a permutation of the pool, `truth` holds a
/// non-finite value, `predicted` does not match the pool's length, or, with
/// `predicted` given, `actual_gain` does not match the pool's length or holds
/// a non-finite value.
pub fn audit(
    ranking: &[usize],
    truth: &[f64],
    predicted: Option<&[Option<f64>]>,
    actual_gain: &[f64],
    min_improvement: f64,
    top_k: usize,
) -> Result<AuditMetrics, String> {
    let n = truth.len();
    check_permutation(ranking, n)?;
    if let Some(index) = truth.iter().position(|value| !value.is_finite()) {
        return Err(format!("truth[{index}] is not finite"));
    }
    if let Some(estimates) = predicted.filter(|estimates| estimates.len() != n) {
        return Err(format!(
            "predicted has {} estimates for {n} bundles",
            estimates.len()
        ));
    }
    if predicted.is_some() {
        if actual_gain.len() != n {
            return Err(format!(
                "actualGain has {} values for {n} bundles",
                actual_gain.len()
            ));
        }
        if let Some(index) = actual_gain.iter().position(|value| !value.is_finite()) {
            return Err(format!("actual_gain[{index}] is not finite"));
        }
    }

    let spearman = if n < 2 {
        None
    } else {
        let mut position_value = vec![0.0_f64; n];
        for (position, &bundle) in ranking.iter().enumerate() {
            position_value[bundle] = (n - 1 - position) as f64;
        }
        pearson(&position_value, &average_ranks(truth))
    };

    let mut by_truth: Vec<usize> = (0..n).collect();
    by_truth.sort_by(|&a, &b| {
        truth[b]
            .partial_cmp(&truth[a])
            .unwrap_or(Ordering::Equal)
            .then(a.cmp(&b))
    });
    let window = top_k.min(n);
    let top_k_recall = if window == 0 {
        None
    } else {
        let arm_top: Vec<usize> = ranking[..window].to_vec();
        let hits = by_truth[..window]
            .iter()
            .filter(|bundle| arm_top.contains(bundle))
            .count();
        Some(hits as f64 / window as f64)
    };

    let regret = truth
        .iter()
        .copied()
        .fold(None, |best: Option<f64>, value| {
            Some(best.map_or(value, |b| b.max(value)))
        })
        .and_then(|best| ranking.first().map(|&first| best - truth[first]));

    let (false_promotions, false_rejections, unestimated) = match predicted {
        None => (None, None, 0),
        Some(estimates) => {
            let mut promotions = 0_usize;
            let mut promotion_base = 0_usize;
            let mut rejections = 0_usize;
            let mut rejection_base = 0_usize;
            let mut unestimated = 0_usize;
            for (estimate, &gain) in estimates.iter().zip(actual_gain) {
                let Some(value) = estimate.filter(|value| value.is_finite()) else {
                    unestimated += 1;
                    continue;
                };
                let predicted_win = value >= min_improvement;
                let actual_win = gain >= min_improvement;
                if actual_win {
                    rejection_base += 1;
                    if !predicted_win {
                        rejections += 1;
                    }
                } else {
                    promotion_base += 1;
                    if predicted_win {
                        promotions += 1;
                    }
                }
            }
            (
                Some(Rate {
                    count: promotions,
                    of: promotion_base,
                }),
                Some(Rate {
                    count: rejections,
                    of: rejection_base,
                }),
                unestimated,
            )
        }
    };

    Ok(AuditMetrics {
        bundles: n,
        spearman,
        top_k,
        top_k_recall,
        regret,
        false_promotions,
        false_rejections,
        unestimated,
    })
}

/// Refuses any `ranking` that is not a permutation of `0..n`.
fn check_permutation(ranking: &[usize], n: usize) -> Result<(), String> {
    if ranking.len() != n {
        return Err(format!(
            "ranking has {} entries for {n} bundles",
            ranking.len()
        ));
    }
    let mut seen = vec![false; n];
    for &bundle in ranking {
        if bundle >= n {
            return Err(format!("ranking names bundle {bundle}, outside 0..{n}"));
        }
        if seen[bundle] {
            return Err(format!("ranking names bundle {bundle} twice"));
        }
        seen[bundle] = true;
    }
    Ok(())
}

/// Average-tie ranks, 1-based: tied values share the mean of their positions.
fn average_ranks(values: &[f64]) -> Vec<f64> {
    let mut order: Vec<usize> = (0..values.len()).collect();
    order.sort_by(|&a, &b| values[a].partial_cmp(&values[b]).unwrap_or(Ordering::Equal));
    let mut ranks = vec![0.0_f64; values.len()];
    let mut start = 0;
    while start < order.len() {
        let mut end = start + 1;
        while end < order.len() && values[order[end]] == values[order[start]] {
            end += 1;
        }
        let average = (start + 1 + end) as f64 / 2.0;
        for &index in &order[start..end] {
            ranks[index] = average;
        }
        start = end;
    }
    ranks
}

/// Pearson correlation, or `None` when either side has zero variance.
fn pearson(x: &[f64], y: &[f64]) -> Option<f64> {
    let count = x.len() as f64;
    let mean_x = x.iter().sum::<f64>() / count;
    let mean_y = y.iter().sum::<f64>() / count;
    let (mut cross, mut var_x, mut var_y) = (0.0_f64, 0.0_f64, 0.0_f64);
    for (a, b) in x.iter().zip(y) {
        let dx = a - mean_x;
        let dy = b - mean_y;
        cross += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }
    if var_x == 0.0 || var_y == 0.0 {
        None
    } else {
        Some(cross / (var_x * var_y).sqrt())
    }
}

/// Five-number summary with linear-interpolated quartiles.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Distribution {
    /// Finite values summarised.
    pub n: usize,
    /// Smallest value.
    pub min: f64,
    /// First quartile.
    pub q1: f64,
    /// Median.
    pub median: f64,
    /// Third quartile.
    pub q3: f64,
    /// Largest value.
    pub max: f64,
}

/// Summary of the finite values in `values`, or `None` when there are none.
///
/// Quartiles are linearly interpolated between the two nearest order
/// statistics.
#[must_use]
pub fn distribution(values: &[f64]) -> Option<Distribution> {
    let mut sorted: Vec<f64> = values.iter().copied().filter(|v| v.is_finite()).collect();
    if sorted.is_empty() {
        return None;
    }
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    let quantile = |p: f64| -> f64 {
        let position = p * (sorted.len() - 1) as f64;
        let low = position.floor() as usize;
        let high = position.ceil() as usize;
        let fraction = position - low as f64;
        sorted[low] + (sorted[high] - sorted[low]) * fraction
    };
    Some(Distribution {
        n: sorted.len(),
        min: sorted[0],
        q1: quantile(0.25),
        median: quantile(0.5),
        q3: quantile(0.75),
        max: sorted[sorted.len() - 1],
    })
}

/// Paired comparison of arm `a` against arm `b`, higher better.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PairedComparison {
    /// Pairs where `a` beat `b`.
    pub better: usize,
    /// Pairs where `b` beat `a`.
    pub worse: usize,
    /// Pairs whose difference is within `1e-12` of zero.
    pub ties: usize,
    /// Distribution of the differences `a − b`, over every pair.
    pub differences: Option<Distribution>,
}

/// Compares `a` against `b` pair by pair, higher better.
///
/// A pair is a tie when `|a − b| ≤ 1e-12`.
///
/// # Errors
///
/// When the slices differ in length or hold a non-finite value.
pub fn paired(a: &[f64], b: &[f64]) -> Result<PairedComparison, String> {
    if a.len() != b.len() {
        return Err(format!(
            "paired slices differ in length: {} vs {}",
            a.len(),
            b.len()
        ));
    }
    if let Some(index) = a.iter().position(|v| !v.is_finite()) {
        return Err(format!("a[{index}] is not finite"));
    }
    if let Some(index) = b.iter().position(|v| !v.is_finite()) {
        return Err(format!("b[{index}] is not finite"));
    }
    let differences: Vec<f64> = a.iter().zip(b).map(|(x, y)| x - y).collect();
    let (mut better, mut worse, mut ties) = (0_usize, 0_usize, 0_usize);
    for &difference in &differences {
        if difference.abs() <= 1e-12 {
            ties += 1;
        } else if difference > 0.0 {
            better += 1;
        } else {
            worse += 1;
        }
    }
    Ok(PairedComparison {
        better,
        worse,
        ties,
        differences: distribution(&differences),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Truth for a four-bundle pool: bundle 0 best, then 2, 3, 1.
    const TRUTH: [f64; 4] = [0.9, 0.1, 0.5, 0.3];
    /// Gains for audits that pass no predictions, where they are never read.
    const NO_GAIN: [f64; 4] = [0.0; 4];

    #[test]
    fn arm_names_and_prediction_flags() {
        let labels: Vec<&str> = ARMS.iter().map(|arm| arm.label()).collect();
        assert_eq!(
            labels,
            ["current", "additiveLoss", "cachedOutput", "random"]
        );
        let predicts: Vec<bool> = ARMS.iter().map(|arm| arm.predicts()).collect();
        assert_eq!(predicts, [false, true, true, false]);
    }

    #[test]
    fn rank_by_estimate_keeps_unestimated_slots_and_sorts_the_rest() {
        assert_eq!(
            rank_by_estimate(&[Some(0.1), None, Some(0.5), Some(f64::NAN)]),
            vec![2, 1, 0, 3]
        );
    }

    #[test]
    fn rank_by_estimate_breaks_ties_by_lower_index() {
        assert_eq!(
            rank_by_estimate(&[Some(0.2), Some(0.2), Some(0.9)]),
            vec![2, 0, 1]
        );
    }

    #[test]
    fn rank_current_is_the_emission_order() {
        assert_eq!(rank_current(4), vec![0, 1, 2, 3]);
    }

    #[test]
    fn rank_random_is_a_reproducible_permutation() {
        let order = rank_random(10, 7);
        let mut sorted = order.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..10).collect::<Vec<usize>>());
        assert_eq!(order, rank_random(10, 7));
        assert_ne!(rank_random(10, 1), rank_random(10, 2));
    }

    #[test]
    fn a_perfect_ranking_audits_perfectly() {
        let metrics = audit(&[0, 2, 3, 1], &TRUTH, None, &NO_GAIN, 0.0, 2).unwrap();
        assert!((metrics.spearman.unwrap() - 1.0).abs() < 1e-12);
        assert!((metrics.top_k_recall.unwrap() - 1.0).abs() < 1e-12);
        assert!(metrics.regret.unwrap().abs() < 1e-12);
        assert_eq!(metrics.bundles, 4);
        assert_eq!(metrics.top_k, 2);
        assert_eq!(metrics.false_promotions, None);
        assert_eq!(metrics.unestimated, 0);
    }

    #[test]
    fn a_reversed_ranking_is_anti_correlated_and_regretful() {
        // Ranks the worst bundle (1, truth 0.1) first: regret is 0.9 - 0.1.
        let metrics = audit(&[1, 3, 2, 0], &TRUTH, None, &NO_GAIN, 0.0, 2).unwrap();
        assert!((metrics.spearman.unwrap() + 1.0).abs() < 1e-12);
        assert!((metrics.regret.unwrap() - 0.8).abs() < 1e-12);
        assert_eq!(metrics.top_k_recall, Some(0.0));
    }

    #[test]
    fn prediction_audits_judge_actual_gain_not_truth() {
        // Truth drives only the ranking metrics. Here truth and actual gain
        // disagree: bundle 3 scores 0.55 (a score delta of 0.05 from a parent
        // at 0.5), yet its full error reduction is 0.12, which is a win.
        let truth = [0.9, 0.55, 0.8, 0.55];
        // Gain is parent error minus bundle error, in the prediction's units.
        let gain = [0.4, 0.05, 0.3, 0.12];
        // Bundle 1 is a false promotion (predicted win, actual non-win), bundle
        // 3 a false rejection (predicted non-win, actual win), bundle 2 is
        // unestimated and so excluded from both audits.
        let estimates = [Some(0.2), Some(0.15), None, Some(-0.05)];
        let metrics = audit(&[0, 2, 3, 1], &truth, Some(&estimates), &gain, 0.1, 2).unwrap();
        assert_eq!(metrics.unestimated, 1);
        assert_eq!(metrics.false_promotions, Some(Rate { count: 1, of: 1 }));
        assert_eq!(metrics.false_rejections, Some(Rate { count: 1, of: 2 }));
    }

    #[test]
    fn prediction_audits_need_one_finite_gain_per_bundle() {
        let estimates = [Some(0.2), None, None, None];
        assert!(audit(&[0, 1, 2, 3], &TRUTH, Some(&estimates), &[0.2; 3], 0.1, 2).is_err());
        assert!(
            audit(
                &[0, 1, 2, 3],
                &TRUTH,
                Some(&estimates),
                &[0.2, f64::NAN, 0.0, 0.0],
                0.1,
                2
            )
            .is_err()
        );
    }

    #[test]
    fn a_non_permutation_ranking_is_refused() {
        assert!(audit(&[0, 0, 1, 2], &TRUTH, None, &NO_GAIN, 0.0, 2).is_err());
        assert!(audit(&[0, 1, 2], &TRUTH, None, &NO_GAIN, 0.0, 2).is_err());
        assert!(audit(&[0, 1, 2, 4], &TRUTH, None, &NO_GAIN, 0.0, 2).is_err());
        assert!(
            audit(
                &[0, 1, 2, 3],
                &[0.1, f64::NAN, 0.2, 0.3],
                None,
                &NO_GAIN,
                0.0,
                2
            )
            .is_err()
        );
    }

    #[test]
    fn distribution_interpolates_quartiles() {
        let summary = distribution(&[1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        assert_eq!(
            summary,
            Distribution {
                n: 5,
                min: 1.0,
                q1: 2.0,
                median: 3.0,
                q3: 4.0,
                max: 5.0,
            }
        );
        assert_eq!(distribution(&[f64::NAN]), None);
        assert_eq!(distribution(&[1.0, f64::NAN, 3.0]).unwrap().n, 2);
    }

    #[test]
    fn paired_counts_wins_losses_and_ties() {
        let result = paired(&[1.0, 2.0, 3.0, 4.0], &[0.5, 2.0, 4.0, 3.5]).unwrap();
        assert_eq!((result.better, result.worse, result.ties), (2, 1, 1));
        assert_eq!(result.differences.unwrap().n, 4);
    }

    #[test]
    fn paired_refuses_mismatch_and_non_finite_values() {
        assert!(paired(&[1.0, 2.0], &[1.0]).is_err());
        assert!(paired(&[1.0, f64::INFINITY], &[1.0, 2.0]).is_err());
    }
}
