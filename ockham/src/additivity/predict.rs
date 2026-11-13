//! Additive output prediction for pruning bundles (Issue #260).
//!
//! A bundle's outputs are predicted as `y0 + Σ(yi − y0)` from cached single-cut
//! output deltas `yi − y0`. Everything here is pure: no file IO, and the
//! corpus is folded chunk by chunk through [`LossSum`] so it never has to sit
//! in memory whole. Per-record cost is always `neat_core::loss::mse_record`.

use neat_core::loss::mse_record;
use serde::Serialize;
use std::fmt;

/// Why a prediction could not be formed.
///
/// `index` names the position of the offending value in the slice it came
/// from. For [`additive_loss_change`] a non-finite base loss is reported one
/// past the last member, so it never shares an index with a member.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "camelCase")]
pub enum PredictError {
    /// Two slices that must agree in length do not.
    Mismatched {
        /// Length the governing operand fixes.
        expected: usize,
        /// Length actually supplied.
        actual: usize,
    },
    /// A value is NaN or infinite, or a derived value overflowed.
    NonFinite {
        /// Position of the offending value.
        index: usize,
    },
}

impl fmt::Display for PredictError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Mismatched { expected, actual } => {
                write!(f, "length mismatch: expected {expected}, got {actual}")
            }
            Self::NonFinite { index } => write!(f, "non-finite value at index {index}"),
        }
    }
}

/// Adds one cached single-cut delta into a running prediction:
/// `acc[j] += yi[j] - y0[j]`.
///
/// All three slices must share a length, else [`PredictError::Mismatched`].
/// Every value is validated before anything is written, so any error leaves
/// `acc` exactly as it was. A non-finite `y0` or `yi` value, or a non-finite
/// accumulated value, is [`PredictError::NonFinite`] at its index.
///
/// # Errors
///
/// [`PredictError::Mismatched`] on length disagreement and
/// [`PredictError::NonFinite`] on any non-finite value.
pub fn add_delta(acc: &mut [f32], y0: &[f32], yi: &[f32]) -> Result<(), PredictError> {
    if y0.len() != acc.len() {
        return Err(PredictError::Mismatched {
            expected: acc.len(),
            actual: y0.len(),
        });
    }
    if yi.len() != acc.len() {
        return Err(PredictError::Mismatched {
            expected: acc.len(),
            actual: yi.len(),
        });
    }
    for (index, ((a, base), other)) in acc.iter().zip(y0).zip(yi).enumerate() {
        if !base.is_finite() || !other.is_finite() || !(a + (other - base)).is_finite() {
            return Err(PredictError::NonFinite { index });
        }
    }
    for ((a, base), other) in acc.iter_mut().zip(y0).zip(yi) {
        *a += other - base;
    }
    Ok(())
}

/// Sum of `mse_record(target_row, output_row)` over every row of a chunk.
///
/// `targets` and `outputs` are row-major with `output_count` values per
/// record. The two lengths must match and be a multiple of `output_count`
/// (which must itself be positive), else [`PredictError::Mismatched`]. An
/// empty chunk sums to `0.0`. A record whose loss is non-finite is
/// [`PredictError::NonFinite`] at that record's index. A sum that overflows
/// only after every record is finite reports the record count.
///
/// # Errors
///
/// [`PredictError::Mismatched`] on shape disagreement and
/// [`PredictError::NonFinite`] on a non-finite loss.
pub fn chunk_loss_sum(
    targets: &[f32],
    outputs: &[f32],
    output_count: usize,
) -> Result<f64, PredictError> {
    if output_count == 0 {
        return Err(PredictError::Mismatched {
            expected: 1,
            actual: 0,
        });
    }
    if targets.len() != outputs.len() {
        return Err(PredictError::Mismatched {
            expected: targets.len(),
            actual: outputs.len(),
        });
    }
    if !targets.len().is_multiple_of(output_count) {
        return Err(PredictError::Mismatched {
            expected: output_count,
            actual: targets.len(),
        });
    }
    let records = targets.len() / output_count;
    let mut sum = 0.0_f64;
    for (index, (target, output)) in targets
        .chunks_exact(output_count)
        .zip(outputs.chunks_exact(output_count))
        .enumerate()
    {
        let loss = mse_record(target, output);
        if !loss.is_finite() {
            return Err(PredictError::NonFinite { index });
        }
        sum += loss;
    }
    if !sum.is_finite() {
        return Err(PredictError::NonFinite { index: records });
    }
    Ok(sum)
}

/// A running total of per-record loss, folded chunk by chunk.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct LossSum {
    /// Sum of per-record losses so far.
    pub sum: f64,
    /// Records folded in so far.
    pub records: u64,
}

impl LossSum {
    /// Folds in one chunk's summed loss and the number of records it covers.
    pub fn add(&mut self, chunk_sum: f64, records: u64) {
        self.sum += chunk_sum;
        self.records += records;
    }

    /// Mean loss per record, or `None` when no record has been folded in.
    #[must_use]
    pub fn mean(&self) -> Option<f64> {
        if self.records == 0 {
            None
        } else {
            Some(self.sum / self.records as f64)
        }
    }
}

/// Diagnostic baseline: the additive change `Σ (L_i − L_0)` over the member
/// losses, against the unpruned base loss `L_0`.
///
/// Every input must be finite. A non-finite base is reported one past the
/// last member; a non-finite member, or a sum that overflows, at its index.
///
/// # Errors
///
/// [`PredictError::NonFinite`] when any input or derived term is not finite.
pub fn additive_loss_change(base_loss: f64, member_losses: &[f64]) -> Result<f64, PredictError> {
    if !base_loss.is_finite() {
        return Err(PredictError::NonFinite {
            index: member_losses.len(),
        });
    }
    let mut total = 0.0_f64;
    for (index, loss) in member_losses.iter().enumerate() {
        let change = loss - base_loss;
        if !change.is_finite() {
            return Err(PredictError::NonFinite { index });
        }
        total += change;
    }
    if !total.is_finite() {
        return Err(PredictError::NonFinite {
            index: member_losses.len(),
        });
    }
    Ok(total)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixtures::{creature, neuron, synapse};
    use crate::prune::{GroupMember, PrunedCandidate, prune_hidden_group};
    use neat_core::{CreatureExport, compile_creature};

    /// Inputs and targets are dyadic rationals of small magnitude, so every
    /// product and sum in the networks below is exact in `f32` and the
    /// predicted and actual outputs can be compared bit for bit.
    const XS: [f32; 5] = [-2.0, -0.5, 0.0, 1.0, 3.5];
    const TARGETS: [f32; 5] = [0.5, -1.0, 0.25, 2.0, 0.0];
    /// Inputs inside `[-1, 1]`, where `HARD_TANH` is still linear, so the
    /// hidden neurons of `linear_pair` never leave their linear band.
    const LINEAR_XS: [f32; 5] = [-1.0, -0.5, 0.0, 0.5, 1.0];

    fn member(uuid: &str) -> GroupMember {
        GroupMember {
            uuid: uuid.into(),
            mean: None,
        }
    }

    /// Output of a one-input, one-output creature on each record in turn.
    fn outputs_of(creature: &CreatureExport, xs: &[f32]) -> Vec<f32> {
        let mut net = compile_creature(creature).unwrap();
        xs.iter().flat_map(|&x| net.activate(&[x], 1)).collect()
    }

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() <= 1e-9 * a.abs().max(b.abs()).max(1.0)
    }

    /// `input → h_a (×0.75) → output (×2.0)` and `input → h_b (×−0.5) →
    /// output (×1.0)`, both hidden neurons `HARD_TANH`, output `output_squash`.
    fn linear_pair(output_squash: &str) -> CreatureExport {
        // IDENTITY pass-throughs are avoided: NEAT-AI-core splices them into direct edges.
        creature(
            1,
            1,
            vec![
                neuron("hidden", "h_a", 0.0, Some("HARD_TANH")),
                neuron("hidden", "h_b", 0.0, Some("HARD_TANH")),
                neuron("output", "output-0", 0.25, Some(output_squash)),
            ],
            vec![
                synapse("input-0", "h_a", 0.75),
                synapse("input-0", "h_b", -0.5),
                synapse("h_a", "output-0", 2.0),
                synapse("h_b", "output-0", 1.0),
            ],
        )
    }

    /// `h_a` feeds `h_b` and the output; `h_b` is `TANH`, so pruning `h_a`
    /// strands it. Pruning `h_b` alone leaves `h_a` reaching the output.
    fn cascading_pair() -> CreatureExport {
        creature(
            1,
            1,
            vec![
                neuron("hidden", "h_a", 0.0, Some("IDENTITY")),
                neuron("hidden", "h_b", 0.0, Some("TANH")),
                neuron("output", "output-0", 0.25, Some("IDENTITY")),
            ],
            vec![
                synapse("input-0", "h_a", 1.0),
                synapse("h_a", "h_b", 2.0),
                synapse("h_a", "output-0", 3.0),
                synapse("h_b", "output-0", 1.0),
            ],
        )
    }

    /// Predicted bundle outputs from the parent and its single-cut outputs.
    fn predicted_outputs(y0: &[f32], members: &[&[f32]]) -> Vec<f32> {
        let mut acc = y0.to_vec();
        for yi in members {
            add_delta(&mut acc, y0, yi).unwrap();
        }
        acc
    }

    /// Positive control: the cut requested `uuid`, and the child no longer
    /// carries it as a hidden neuron, so the cut really removed its member.
    fn assert_requested(cut: &PrunedCandidate, uuid: &str) {
        assert!(
            cut.detail
                .requested_neurons
                .iter()
                .any(|n| n.as_str() == uuid),
            "{uuid} was not requested: {:?}",
            cut.detail.requested_neurons
        );
        assert!(
            !cut.creature
                .neurons
                .iter()
                .any(|n| n.neuron_type == "hidden" && n.uuid == uuid),
            "{uuid} is still carried as a hidden neuron"
        );
    }

    #[test]
    fn exact_linear_deltas_predict_the_bundle_exactly() {
        let parent = linear_pair("IDENTITY");
        let y0 = outputs_of(&parent, &LINEAR_XS);
        let child_a = prune_hidden_group(&parent, &[member("h_a")]).unwrap();
        let child_b = prune_hidden_group(&parent, &[member("h_b")]).unwrap();
        let bundle = prune_hidden_group(&parent, &[member("h_a"), member("h_b")]).unwrap();
        assert_requested(&child_a, "h_a");
        assert_requested(&child_b, "h_b");
        assert_requested(&bundle, "h_a");
        assert_requested(&bundle, "h_b");
        let ya = outputs_of(&child_a.creature, &LINEAR_XS);
        let yb = outputs_of(&child_b.creature, &LINEAR_XS);
        let actual = outputs_of(&bundle.creature, &LINEAR_XS);
        assert_ne!(ya, y0, "removing h_a must move the output");
        assert_ne!(yb, y0, "removing h_b must move the output");

        let predicted = predicted_outputs(&y0, &[&ya, &yb]);
        for (index, (p, a)) in predicted.iter().zip(&actual).enumerate() {
            assert!(
                (p - a).abs() <= 1e-6,
                "record {index}: predicted {p} vs actual {a}"
            );
        }

        let mut predicted_total = LossSum::default();
        predicted_total.add(
            chunk_loss_sum(&TARGETS, &predicted, 1).unwrap(),
            LINEAR_XS.len() as u64,
        );
        let predicted_loss = predicted_total.mean().unwrap();
        let mut actual_sum = LossSum::default();
        actual_sum.add(
            chunk_loss_sum(&TARGETS, &actual, 1).unwrap(),
            LINEAR_XS.len() as u64,
        );
        let actual_loss = actual_sum.mean().unwrap();
        assert!(
            close(predicted_loss, actual_loss),
            "predicted {predicted_loss} vs actual {actual_loss}"
        );
    }

    #[test]
    fn a_nonlinear_output_breaks_the_additive_prediction() {
        // The hidden neurons stay inside their linear band, so the only
        // nonlinearity is the TANH output: the bundle's true output is
        // tanh(0.25) at every input, while the additive prediction sums
        // squashed single cuts and so does not match it.
        const TOLERANCE: f32 = 1e-2;
        let parent = linear_pair("TANH");
        let y0 = outputs_of(&parent, &LINEAR_XS);
        let child_a = prune_hidden_group(&parent, &[member("h_a")]).unwrap();
        let child_b = prune_hidden_group(&parent, &[member("h_b")]).unwrap();
        let bundle = prune_hidden_group(&parent, &[member("h_a"), member("h_b")]).unwrap();
        assert_requested(&child_a, "h_a");
        assert_requested(&child_b, "h_b");
        assert_requested(&bundle, "h_a");
        assert_requested(&bundle, "h_b");
        let ya = outputs_of(&child_a.creature, &LINEAR_XS);
        let yb = outputs_of(&child_b.creature, &LINEAR_XS);
        let actual = outputs_of(&bundle.creature, &LINEAR_XS);
        let predicted = predicted_outputs(&y0, &[&ya, &yb]);
        let gap = predicted
            .iter()
            .zip(&actual)
            .map(|(p, a)| (p - a).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            gap > TOLERANCE,
            "a squashed output should break additivity by more than {TOLERANCE}, gap {gap}"
        );
    }

    #[test]
    fn a_cascade_shared_by_the_bundle_is_double_counted() {
        let parent = cascading_pair();
        let y0 = outputs_of(&parent, &XS);
        let child_a = prune_hidden_group(&parent, &[member("h_a")]).unwrap();
        let child_b = prune_hidden_group(&parent, &[member("h_b")]).unwrap();
        let bundle = prune_hidden_group(&parent, &[member("h_a"), member("h_b")]).unwrap();

        // Positive control: each single cut really removed its own member.
        assert_requested(&child_a, "h_a");
        assert_requested(&child_b, "h_b");
        // Removing h_a strands h_b, so the cascade takes it.
        assert!(
            child_a.detail.cascade_uuids().contains(&"h_b".to_string()),
            "h_a's removal must strand h_b: {:?}",
            child_a.detail.cascade_uuids()
        );
        // The bundle names h_b too, but removes it once.
        let removed = bundle.detail.removed_uuids();
        assert_eq!(
            removed.iter().filter(|uuid| uuid.as_str() == "h_b").count(),
            1,
            "h_b removed more than once: {removed:?}"
        );

        let ya = outputs_of(&child_a.creature, &XS);
        let yb = outputs_of(&child_b.creature, &XS);
        let actual = outputs_of(&bundle.creature, &XS);
        // Both single cuts already lost h_b's contribution, so the sum of
        // their deltas removes it twice.
        let predicted = predicted_outputs(&y0, &[&ya, &yb]);
        let gap = predicted
            .iter()
            .zip(&actual)
            .map(|(p, a)| (p - a).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            gap > 1e-2,
            "the double-counted cascade must show, gap {gap}"
        );
    }

    #[test]
    fn non_finite_values_are_refused_with_their_index() {
        let mut acc = vec![0.0_f32; 3];
        assert_eq!(
            add_delta(&mut acc, &[0.0, f32::NAN, 0.0], &[0.0; 3]),
            Err(PredictError::NonFinite { index: 1 })
        );
        assert_eq!(acc, vec![0.0_f32; 3], "a refused delta must not touch acc");
        assert_eq!(
            add_delta(&mut acc, &[0.0; 3], &[0.0, 0.0, f32::INFINITY]),
            Err(PredictError::NonFinite { index: 2 })
        );
        let mut near_max = vec![f32::MAX];
        assert_eq!(
            add_delta(&mut near_max, &[-f32::MAX], &[f32::MAX]),
            Err(PredictError::NonFinite { index: 0 }),
            "an overflowing delta is non-finite"
        );

        assert_eq!(
            chunk_loss_sum(&[0.0, 1.0], &[0.0, f32::NAN], 1),
            Err(PredictError::NonFinite { index: 1 })
        );
        assert_eq!(
            chunk_loss_sum(&[0.0, 1.0], &[f32::INFINITY, 0.0], 1),
            Err(PredictError::NonFinite { index: 0 })
        );

        assert_eq!(
            additive_loss_change(f64::NAN, &[1.0]),
            Err(PredictError::NonFinite { index: 1 })
        );
        assert_eq!(
            additive_loss_change(1.0, &[2.0, f64::NAN]),
            Err(PredictError::NonFinite { index: 1 })
        );
        assert_eq!(additive_loss_change(1.0, &[2.0, 0.5]), Ok(0.5));
    }

    #[test]
    fn mismatched_lengths_are_refused() {
        let mut acc = vec![0.0_f32; 2];
        assert_eq!(
            add_delta(&mut acc, &[0.0], &[0.0, 0.0]),
            Err(PredictError::Mismatched {
                expected: 2,
                actual: 1
            })
        );
        assert_eq!(
            add_delta(&mut acc, &[0.0, 0.0], &[0.0]),
            Err(PredictError::Mismatched {
                expected: 2,
                actual: 1
            })
        );
        assert_eq!(acc, vec![0.0_f32; 2], "a refused delta must not touch acc");
        assert_eq!(
            chunk_loss_sum(&[0.0, 1.0, 2.0], &[0.0, 1.0], 2),
            Err(PredictError::Mismatched {
                expected: 3,
                actual: 2
            })
        );
        assert_eq!(
            chunk_loss_sum(&[0.0, 1.0, 2.0], &[0.0, 1.0, 2.0], 2),
            Err(PredictError::Mismatched {
                expected: 2,
                actual: 3
            })
        );
        assert!(chunk_loss_sum(&[0.0], &[0.0], 0).is_err());
    }

    #[test]
    fn loss_sum_mean_is_none_before_any_record() {
        assert_eq!(LossSum::default().mean(), None);
        let mut total = LossSum::default();
        total.add(3.0, 2);
        assert_eq!(total.mean(), Some(1.5));
    }
}
