//! Composite dead-wood priority — expected pruning value (Issue #107).
//!
//! The named orderings each read one signal. This reads them together and ranks
//! by the economics the sweep is actually paid in:
//!
//! ```text
//! expected_pruning_value = P(the full scorer confirms the cut)
//!                        × expected growth-unit saving
//!                        ÷ expected evaluation cost
//! ```
//!
//! `P` is a transparent logistic of the quietness, structural and historical
//! features in [`crate::features`]; the saving is the cascade dry-run's (#106);
//! and the cost is the screen every candidate pays plus the full-corpus score
//! only a survivor triggers — so a candidate that is likelier to survive is
//! worth more *and* costs more, which is exactly the trade the sweep makes.
//!
//! The weights are a hand-built starting point, deliberately separate from the
//! learned model in [`crate::model`]: the composite ordering must be usable,
//! and benchmarkable, with no training data at all.
//!
//! This ranks. It never removes: every candidate the composite promotes still
//! passes `creature.validate()`, the sampled screen and full-corpus scoring.

use crate::features::{CandidateFeatures, PriorEvidence};
use crate::model::PriorityModel;

/// Coefficients of the hand-built survival estimate.
///
/// Signs are baked into the defaults so the field names read as the signal
/// rather than as the direction: a *lower* downstream sensitivity raises `P`
/// because [`CompositeWeights::sensitivity`] is negative.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CompositeWeights {
    /// Intercept — the log-odds of a candidate with no signal at all.
    pub bias: f64,
    /// Coefficient on `ln(1 + mean_abs × Σ abs(outgoing weight))`.
    pub sensitivity: f64,
    /// Coefficient on `ln(1 + activation variance)`.
    pub variance: f64,
    /// Bonus for an `IDENTITY` squash — an exact fold rather than an estimate.
    pub identity: f64,
    /// Coefficient on `ln(1 + fan_out)` — the structural blast radius.
    pub fan_out: f64,
    /// Coefficient on the normalised depth from the inputs.
    pub depth: f64,
    /// Bonus per `ln(1 + wins)` an earlier epoch recorded for the uuid.
    pub prior_win: f64,
    /// Penalty per `ln(1 + failures)` an earlier epoch recorded for the uuid.
    pub prior_failure: f64,
    /// Full-corpus score cost, in units of one screened candidate.
    ///
    /// Only a candidate that survives the sampled screen is scored on the full
    /// corpus. The default is the reciprocal of the default
    /// `--screen-sample-rate`: a full score reads twenty times the records a
    /// `0.05` screen does.
    pub full_score_cost: f64,
    /// Rate at which a screened candidate reaches full scoring.
    ///
    /// The screen is not the scorer, and the ranking cannot predict which
    /// candidates it will promote: sampled false positives are exactly the ones
    /// no signal saw coming. The expected cost of a visit is therefore
    /// `1 + screen_survival × full_score_cost` — one screen, plus a full score
    /// at the fleet's measured promotion rate.
    ///
    /// Charging `P × full_score_cost` instead — the candidate's own confirmation
    /// odds — makes a hopeless candidate look *cheap*, and dividing by that tiny
    /// cost promotes precisely the cuts the scorer will not confirm.
    /// `priority_ordering_bench` measures that form as both fewer confirmed cuts
    /// and fewer growth units per hour, so the cost is charged at a rate rather
    /// than per candidate: it sets the scale of the value, and the ordering is
    /// decided by `P` and the saving.
    pub screen_survival: f64,
}

impl Default for CompositeWeights {
    /// The benchmarked starting point (`priority_ordering_bench`).
    ///
    /// Quietness dominates, because a neuron the network barely uses is the one
    /// the scorer is likeliest to let go; the structural terms break ties among
    /// equally quiet neurons; and the historical terms are deliberately the
    /// smallest, because evidence from an older corpus is a prior and not the
    /// current truth.
    fn default() -> Self {
        Self {
            bias: -1.0,
            sensitivity: -1.2,
            variance: -0.6,
            identity: 1.0,
            fan_out: -0.35,
            depth: 0.2,
            prior_win: 0.8,
            prior_failure: -0.6,
            full_score_cost: 20.0,
            screen_survival: 0.1,
        }
    }
}

impl CompositeWeights {
    /// Hand-built `P(the full scorer confirms this cut)`, in `(0, 1)`.
    ///
    /// A probability, not a permission: nothing downstream may skip the scorer
    /// because this returned a high number.
    pub fn survival_probability(&self, f: &CandidateFeatures) -> f64 {
        let saturating = |n: usize| (1.0 + n as f64).ln();
        let z = self.bias
            + self.sensitivity * (1.0 + f.downstream_sensitivity().max(0.0)).ln()
            + self.variance * (1.0 + f.variance.max(0.0)).ln()
            + self.identity * if f.identity { 1.0 } else { 0.0 }
            + self.fan_out * (1.0 + f.fan_out as f64).ln()
            + self.depth * f.depth_fraction
            + self.prior_win * saturating(f.prior_wins)
            + self.prior_failure * saturating(f.prior_failures);
        logistic(z)
    }
}

/// Numerically stable logistic.
pub fn logistic(z: f64) -> f64 {
    if z >= 0.0 {
        1.0 / (1.0 + (-z).exp())
    } else {
        let e = z.exp();
        e / (1.0 + e)
    }
}

/// Expected pruning value of one candidate at survival probability `p`.
///
/// The saving enters as `ln(1 + growth units)`. A cut twice the size is not
/// twice as likely to compound into the run's cumulative improvement — whether
/// the scorer confirms it at all dominates — so the cascade breaks ties between
/// candidates of similar odds rather than overruling them. Ranking on the raw
/// saving spends the budget on the largest cascades whatever their chances,
/// which `priority_ordering_bench` measures as fewer confirmed cuts per hour
/// *and* fewer growth units per hour than this form.
///
/// `f64::NEG_INFINITY` for a neuron the activation scan never covered: its
/// statistics are absent rather than quiet, so it is ranked last instead of
/// being read as the quietest neuron on the creature. It stays in the sweep —
/// an ordering never shrinks the permutation.
///
/// A cut the transform would refuse saves nothing, so it scores `0.0` and falls
/// behind every candidate the razor can actually build.
pub fn expected_pruning_value(p: f64, f: &CandidateFeatures, weights: &CompositeWeights) -> f64 {
    if !f.measured {
        return f64::NEG_INFINITY;
    }
    let saving = (1.0 + f.cascade_growth_units.max(0.0)).ln();
    let cost = 1.0 + weights.screen_survival.clamp(0.0, 1.0) * weights.full_score_cost.max(0.0);
    p * saving / cost
}

/// Everything the composite and learned rankings read beyond the creature.
///
/// Held for the length of a run and borrowed by
/// [`crate::ordering::OrderingConfig`]: the evidence is fleet history, and the
/// model is a file, so neither belongs in a per-sweep computation.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct PriorityContext {
    /// Hand-built composite coefficients.
    pub weights: CompositeWeights,
    /// What earlier epochs learnt — a prior on the ranking, never a verdict.
    pub evidence: PriorEvidence,
    /// Learned ranker, when `--ordering-model` supplied one.
    pub model: Option<PriorityModel>,
}

impl PriorityContext {
    /// Context with the default weights, no evidence and no model.
    pub fn new() -> Self {
        Self::default()
    }

    /// Context reading `evidence`, with `model` when one was configured.
    pub fn with(evidence: PriorEvidence, model: Option<PriorityModel>) -> Self {
        Self {
            weights: CompositeWeights::default(),
            evidence,
            model,
        }
    }

    /// Hand-built expected pruning value of `f`.
    pub fn composite_value(&self, f: &CandidateFeatures) -> f64 {
        expected_pruning_value(self.weights.survival_probability(f), f, &self.weights)
    }

    /// Learned expected pruning value of `f`, or `None` with no model loaded.
    pub fn learned_value(&self, f: &CandidateFeatures) -> Option<f64> {
        let model = self.model.as_ref()?;
        Some(expected_pruning_value(
            model.probability(f),
            f,
            &self.weights,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base() -> CandidateFeatures {
        CandidateFeatures {
            measured: true,
            variance: 0.5,
            mean_abs: 0.5,
            range: 1.0,
            outgoing_weight: 1.0,
            fan_in: 1,
            fan_out: 1,
            direct_growth_units: 1.2,
            cascade_growth_units: 1.2,
            identity: false,
            blocked: false,
            depth_fraction: 0.5,
            prior_wins: 0,
            prior_failures: 0,
        }
    }

    #[test]
    fn a_quiet_neuron_is_likelier_to_survive_than_a_loud_one() {
        let w = CompositeWeights::default();
        let quiet = CandidateFeatures {
            mean_abs: 0.01,
            outgoing_weight: 0.5,
            variance: 0.001,
            ..base()
        };
        let loud = CandidateFeatures {
            mean_abs: 3.0,
            outgoing_weight: 6.0,
            variance: 9.0,
            ..base()
        };
        assert!(
            w.survival_probability(&quiet) > w.survival_probability(&loud),
            "quiet {} loud {}",
            w.survival_probability(&quiet),
            w.survival_probability(&loud)
        );
    }

    #[test]
    fn every_probability_stays_inside_the_unit_interval() {
        let w = CompositeWeights::default();
        for (mean_abs, weight, wins) in [(0.0, 0.0, 0), (1e9, 1e9, 0), (0.0, 0.0, 1_000)] {
            let f = CandidateFeatures {
                mean_abs,
                outgoing_weight: weight,
                prior_wins: wins,
                ..base()
            };
            let p = w.survival_probability(&f);
            assert!(p > 0.0 && p < 1.0, "{p} for {f:?}");
        }
        assert_eq!(logistic(f64::NEG_INFINITY), 0.0);
        assert_eq!(logistic(f64::INFINITY), 1.0);
    }

    #[test]
    fn history_moves_the_estimate_without_deciding_it() {
        let w = CompositeWeights::default();
        let win = CandidateFeatures {
            prior_wins: 3,
            ..base()
        };
        let fail = CandidateFeatures {
            prior_failures: 3,
            ..base()
        };
        let p_win = w.survival_probability(&win);
        let p_fail = w.survival_probability(&fail);
        assert!(p_win > w.survival_probability(&base()));
        assert!(p_fail < w.survival_probability(&base()));
        // A prior, not a verdict: even a uuid every epoch removed is far from
        // certain, and one every epoch rejected is never ruled out.
        assert!(p_win < 0.99, "{p_win}");
        assert!(p_fail > 0.0, "{p_fail}");
    }

    #[test]
    fn a_bigger_cascade_saving_is_worth_more_at_the_same_probability() {
        let w = CompositeWeights::default();
        let small = base();
        let large = CandidateFeatures {
            cascade_growth_units: 12.0,
            ..base()
        };
        assert!(expected_pruning_value(0.5, &large, &w) > expected_pruning_value(0.5, &small, &w));
    }

    #[test]
    fn a_likelier_cut_outranks_a_bigger_one_it_is_unlikely_to_win() {
        let w = CompositeWeights::default();
        let likely_small = base();
        let unlikely_large = CandidateFeatures {
            cascade_growth_units: 24.0,
            ..base()
        };
        // Twenty times the structure at a twentieth of the odds is the bet the
        // benchmark says not to take: whether the scorer confirms the cut at all
        // dominates how much structure it would have removed.
        assert!(
            expected_pruning_value(0.4, &likely_small, &w)
                > expected_pruning_value(0.02, &unlikely_large, &w)
        );
    }

    #[test]
    fn the_full_score_cost_discounts_a_likely_survivor() {
        let f = base();
        let charged = CompositeWeights::default();
        let free = CompositeWeights {
            full_score_cost: 0.0,
            ..charged
        };
        // Same saving, same probability: charging for the full score it will
        // trigger must lower the value, never raise it.
        assert!(expected_pruning_value(0.9, &f, &charged) < expected_pruning_value(0.9, &f, &free));
    }

    #[test]
    fn the_evaluation_cost_scales_the_value_without_reordering_the_sweep() {
        let cheap = CompositeWeights {
            full_score_cost: 1.0,
            ..CompositeWeights::default()
        };
        let dear = CompositeWeights {
            full_score_cost: 100.0,
            ..CompositeWeights::default()
        };
        let big = CandidateFeatures {
            cascade_growth_units: 24.0,
            ..base()
        };
        let order = |w: &CompositeWeights| {
            expected_pruning_value(0.4, &base(), w) > expected_pruning_value(0.05, &big, w)
        };
        // The cost is charged at a rate rather than per candidate, so a dearer
        // full score lowers every value and swaps none of them: what a
        // candidate is tried before is decided by its odds and its saving.
        assert_eq!(order(&cheap), order(&dear));
        assert!(order(&cheap), "the likelier cut leads either way");
        assert!(
            expected_pruning_value(0.4, &base(), &dear)
                < expected_pruning_value(0.4, &base(), &cheap)
        );
    }

    #[test]
    fn a_refused_cut_is_worth_nothing_and_an_unmeasured_one_ranks_last() {
        let w = CompositeWeights::default();
        let blocked = CandidateFeatures {
            blocked: true,
            cascade_growth_units: 0.0,
            ..base()
        };
        assert_eq!(expected_pruning_value(0.9, &blocked, &w), 0.0);
        let unmeasured = CandidateFeatures {
            measured: false,
            ..base()
        };
        assert_eq!(
            expected_pruning_value(0.9, &unmeasured, &w),
            f64::NEG_INFINITY
        );
    }

    #[test]
    fn a_context_without_a_model_offers_no_learned_value() {
        let ctx = PriorityContext::new();
        assert!(ctx.learned_value(&base()).is_none());
        assert!(ctx.composite_value(&base()) > 0.0);
    }
}
