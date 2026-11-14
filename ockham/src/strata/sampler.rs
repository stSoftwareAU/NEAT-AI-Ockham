//! Scales, statistics, the seeded bundle sampler and the temperature choice
//! (issue #263).
//!
//! Everything here is a *prediction* used to order candidates. None of it
//! decides a selection: that is the full calibration scorer's job alone.

use crate::ordering::SplitMix64;

/// A scale at or below this is treated as absent: the stratum's median
/// importance is too small to divide by, so the stratum is left unscaled.
pub const SCALE_FLOOR: f64 = 1e-12;

/// 2^-53, so `(next() >> 11) as f64 * INV_2_53` is uniform on `[0, 1)`.
const INV_2_53: f64 = 1.0 / 9_007_199_254_740_992.0;

/// Median of `values`, averaging the two middle values for an even count.
/// `None` for an empty slice.
pub fn median(values: &[f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let n = sorted.len();
    Some(if n % 2 == 1 {
        sorted[n / 2]
    } else {
        (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
    })
}

/// Per-stratum scale: the median absolute importance over the measured
/// neurons, or `1.0` when that median is not above [`SCALE_FLOOR`].
///
/// `importance[t]` holds stratum `t`'s importance for each measured neuron.
pub fn fit_scales(importance: &[Vec<f64>]) -> Vec<f64> {
    importance
        .iter()
        .map(|row| {
            let magnitudes: Vec<f64> = row.iter().map(|v| v.abs()).collect();
            match median(&magnitudes) {
                Some(m) if m > SCALE_FLOOR => m,
                _ => 1.0,
            }
        })
        .collect()
}

/// `importance[t][i] / scales[t]` for every stratum and neuron.
pub fn normalise(importance: &[Vec<f64>], scales: &[f64]) -> Vec<Vec<f64>> {
    importance
        .iter()
        .zip(scales)
        .map(|(row, scale)| row.iter().map(|v| v / scale).collect())
        .collect()
}

/// Aggregate statistic per neuron: `Σ_t weights[t] * normalised[t][i]`.
pub fn aggregate_statistic(normalised: &[Vec<f64>], weights: &[f64]) -> Vec<f64> {
    let Some(first) = normalised.first() else {
        return Vec::new();
    };
    (0..first.len())
        .map(|i| {
            normalised
                .iter()
                .zip(weights)
                .map(|(row, w)| w * row[i])
                .sum()
        })
        .collect()
}

/// Worst-stratum statistic per neuron: `max_t normalised[t][i]`.
pub fn worst_statistic(normalised: &[Vec<f64>]) -> Vec<f64> {
    let Some(first) = normalised.first() else {
        return Vec::new();
    };
    (0..first.len())
        .map(|i| {
            normalised
                .iter()
                .map(|row| row[i])
                .fold(f64::NEG_INFINITY, f64::max)
        })
        .collect()
}

/// Predicted worst-stratum cost of removing `positions` together:
/// `max_t Σ_{i ∈ positions} normalised[t][i]`. Zero with no strata.
pub fn predicted_cost(normalised: &[Vec<f64>], positions: &[usize]) -> f64 {
    if normalised.is_empty() {
        return 0.0;
    }
    normalised
        .iter()
        .map(|row| positions.iter().map(|&i| row[i]).sum::<f64>())
        .fold(f64::NEG_INFINITY, f64::max)
}

/// Record-weighted pooled mean from `(record count, mean)` parts.
/// `None` when no part has any records.
pub fn pooled_mean(parts: &[(u64, f64)]) -> Option<f64> {
    let total: u64 = parts.iter().map(|(n, _)| n).sum();
    if total == 0 {
        return None;
    }
    let weighted: f64 = parts.iter().map(|(n, m)| *n as f64 * m).sum();
    Some(weighted / total as f64)
}

/// Stratum weights proportional to record counts. Callers must ensure the
/// total is non-zero.
pub fn stratum_weights(counts: &[u64]) -> Vec<f64> {
    let total: u64 = counts.iter().sum();
    counts.iter().map(|&n| n as f64 / total as f64).collect()
}

/// Draw one bundle of `k` distinct candidates without replacement and return
/// their positions in ascending order.
///
/// With `tau = Some(t)` each draw picks candidate `i` with probability
/// proportional to `exp(-(x_i - min_remaining x) / t)`, so a small `t` almost
/// always takes the lowest remaining statistic. With `tau = None` every
/// remaining candidate is equally likely. `k` is clamped to the number of
/// candidates. Uniform variates are `(next() >> 11) * 2^-53`.
pub(crate) fn draw_bundle(
    rng: &mut SplitMix64,
    x: &[f64],
    k: usize,
    tau: Option<f64>,
) -> Vec<usize> {
    let k = k.min(x.len());
    let mut remaining: Vec<usize> = (0..x.len()).collect();
    let mut picked = Vec::with_capacity(k);
    for _ in 0..k {
        let weights: Vec<f64> = match tau {
            None => vec![1.0; remaining.len()],
            Some(t) => {
                let min = remaining
                    .iter()
                    .map(|&i| x[i])
                    .fold(f64::INFINITY, f64::min);
                remaining
                    .iter()
                    .map(|&i| (-(x[i] - min) / t).exp())
                    .collect()
            }
        };
        let total: f64 = weights.iter().sum();
        let target = uniform(rng) * total;
        let mut acc = 0.0;
        let mut chosen = remaining.len() - 1;
        for (p, w) in weights.iter().enumerate() {
            acc += w;
            if target < acc {
                chosen = p;
                break;
            }
        }
        picked.push(remaining.remove(chosen));
    }
    picked.sort_unstable();
    picked
}

fn uniform(rng: &mut SplitMix64) -> f64 {
    (rng.next() >> 11) as f64 * INV_2_53
}

/// Index of the temperature with the largest criterion. Ties go to the
/// earliest index. A criterion of negative infinity means "nothing built" and
/// never wins unless every criterion is negative infinity, in which case the
/// first temperature is returned.
pub fn pick_temperature(criteria: &[f64]) -> usize {
    let mut best = 0;
    for (i, c) in criteria.iter().enumerate().skip(1) {
        if *c > criteria[best] {
            best = i;
        }
    }
    best
}

/// Index of the winning bundle: the largest delta strictly above
/// `min_improvement`, ties to the earliest index. `None` when nothing wins.
pub fn select_winner(deltas: &[f64], min_improvement: f64) -> Option<usize> {
    let mut best: Option<usize> = None;
    for (i, &d) in deltas.iter().enumerate() {
        if d > min_improvement && best.is_none_or(|b| d > deltas[b]) {
            best = Some(i);
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rng(seed: u64) -> SplitMix64 {
        SplitMix64(seed)
    }

    #[test]
    fn tiny_temperature_takes_the_k_lowest_statistic_neurons() {
        // Statistics 0.9, 0.1, 0.5, 0.3, 0.7: the two lowest are indices 1 and 3.
        let x = [0.9, 0.1, 0.5, 0.3, 0.7];
        for seed in 1..=20 {
            let bundle = draw_bundle(&mut rng(seed), &x, 2, Some(1e-6));
            assert_eq!(bundle, vec![1, 3], "seed {seed}");
        }
    }

    #[test]
    fn same_seed_gives_the_same_bundle_and_different_seeds_can_differ() {
        let x = [0.0; 12];
        let a = draw_bundle(&mut rng(7), &x, 4, None);
        let b = draw_bundle(&mut rng(7), &x, 4, None);
        assert_eq!(a, b);
        let differs = (1..=50u64).any(|s| draw_bundle(&mut rng(s), &x, 4, None) != a);
        assert!(differs, "fifty seeds should not all draw the same bundle");
    }

    #[test]
    fn bundle_has_distinct_members_and_k_clamps_to_candidate_count() {
        let x = [0.3, 0.1, 0.2];
        let bundle = draw_bundle(&mut rng(3), &x, 10, Some(1.0));
        assert_eq!(bundle.len(), 3);
        assert_eq!(bundle, vec![0, 1, 2]);
        let empty = draw_bundle(&mut rng(3), &[], 4, None);
        assert!(empty.is_empty());
    }

    #[test]
    fn scales_are_median_absolute_importance_per_stratum() {
        // Stratum 0: |values| 1, 2, 4 → median 2. Stratum 1: 0.5, 3 → median 1.75.
        let importance = vec![vec![-1.0, 2.0, -4.0], vec![0.5, -3.0]];
        assert_eq!(fit_scales(&importance), vec![2.0, 1.75]);
    }

    #[test]
    fn scale_falls_back_to_one_when_the_median_is_zero() {
        let importance = vec![vec![0.0, 0.0, 5.0], vec![0.0, 0.0]];
        assert_eq!(fit_scales(&importance), vec![1.0, 1.0]);
        assert_eq!(fit_scales(&[vec![]]), vec![1.0]);
    }

    #[test]
    fn aggregate_is_weighted_mean_and_worst_is_max_on_a_hand_built_matrix() {
        // Three strata, three neurons, already normalised.
        let n = vec![
            vec![1.0, -2.0, 0.5],
            vec![3.0, 0.0, 0.5],
            vec![-1.0, 4.0, 0.5],
        ];
        let weights = [0.5, 0.25, 0.25];
        // 0.5*1 + 0.25*3 + 0.25*(-1) = 1.0; 0.5*(-2) + 0 + 0.25*4 = 0.0; 0.5.
        assert_eq!(aggregate_statistic(&n, &weights), vec![1.0, 0.0, 0.5]);
        assert_eq!(worst_statistic(&n), vec![3.0, 4.0, 0.5]);
    }

    #[test]
    fn predicted_cost_is_the_worst_stratum_sum_over_the_bundle() {
        let n = vec![vec![1.0, -2.0, 0.5], vec![3.0, 0.0, 0.5]];
        assert_eq!(predicted_cost(&n, &[0, 1]), 3.0);
        assert_eq!(predicted_cost(&n, &[1]), 0.0);
        assert_eq!(predicted_cost(&[], &[0]), 0.0);
    }

    #[test]
    fn pooled_mean_weights_by_record_count() {
        assert_eq!(pooled_mean(&[(3, 1.0), (1, 5.0)]), Some(2.0));
        assert_eq!(pooled_mean(&[(0, 9.0)]), None);
        assert_eq!(pooled_mean(&[]), None);
    }

    #[test]
    fn stratum_weights_are_record_shares() {
        assert_eq!(
            stratum_weights(&[1000, 1000, 100]),
            vec![1000.0 / 2100.0, 1000.0 / 2100.0, 100.0 / 2100.0]
        );
    }

    #[test]
    fn temperature_ties_go_to_the_earliest_and_all_absent_picks_first() {
        assert_eq!(pick_temperature(&[1.0, 3.0, 3.0, 2.0]), 1);
        assert_eq!(pick_temperature(&[f64::NEG_INFINITY, 0.5]), 1);
        assert_eq!(pick_temperature(&[f64::NEG_INFINITY, f64::NEG_INFINITY]), 0);
    }

    #[test]
    fn winner_is_the_largest_strict_win_with_ties_to_the_lowest_index() {
        assert_eq!(select_winner(&[0.1, 0.3, 0.3, -0.2], 0.0), Some(1));
        assert_eq!(select_winner(&[0.0, -1.0], 0.0), None);
        assert_eq!(select_winner(&[0.2], 0.2), None);
        assert_eq!(select_winner(&[], 0.0), None);
    }

    #[test]
    fn median_of_an_even_count_averages_the_middle_pair() {
        assert_eq!(median(&[4.0, 1.0, 3.0, 2.0]), Some(2.5));
        assert_eq!(median(&[]), None);
    }
}
