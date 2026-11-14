//! The Pareto arm's ordering, archive and selection (issue #263).
//!
//! [`prescreen`] only chooses which built candidates get the full scorer:
//! it orders them by predicted worst-stratum cost. [`archive`] keeps the
//! non-dominated scored bundles over their *measured* objectives: worst-stratum
//! loss, synapses after the cut, and latency, all minimised. [`select_win`]
//! picks the arm's winner from the archive's wins. Only the full calibration
//! scorer decides what counts as a win.

use std::collections::HashSet;

/// A built candidate before scoring: its prediction and identity.
#[derive(Debug, Clone, PartialEq)]
pub struct Candidate {
    /// Predicted worst-stratum cost, `max_t Σ N[t][i]` (lower is better).
    pub predicted: f64,
    /// Synapse count after the cut (lower is better).
    pub synapses: usize,
    /// Member neuron UUIDs, sorted.
    pub members: Vec<String>,
}

impl Candidate {
    /// A candidate with its members sorted, so the tiebreak key is stable.
    pub fn new(predicted: f64, synapses: usize, mut members: Vec<String>) -> Self {
        members.sort();
        Self {
            predicted,
            synapses,
            members,
        }
    }

    fn key(&self) -> String {
        self.members.join(",")
    }
}

/// Indices of the candidates to full-score.
///
/// A candidate whose member set repeats an earlier one is dropped, so the first
/// draw stays. The rest are ordered by `(predicted, synapses, members joined)`
/// ascending and the first `keep` are returned. Fewer come back when fewer
/// distinct candidates exist.
pub fn prescreen(candidates: &[Candidate], keep: usize) -> Vec<usize> {
    let mut seen = HashSet::new();
    let mut unique: Vec<usize> = (0..candidates.len())
        .filter(|&i| seen.insert(candidates[i].key()))
        .collect();
    unique.sort_by(|&a, &b| {
        let (p, q) = (&candidates[a], &candidates[b]);
        p.predicted
            .total_cmp(&q.predicted)
            .then(p.synapses.cmp(&q.synapses))
            .then_with(|| p.key().cmp(&q.key()))
    });
    unique.truncate(keep);
    unique
}

/// One scored bundle's measured objectives and members.
#[derive(Debug, Clone, PartialEq)]
pub struct Point {
    /// Worst-stratum loss on calibration, `max_t (parent − bundle)` (lower is
    /// better).
    pub loss: f64,
    /// Synapse count after the cut (lower is better).
    pub synapses: usize,
    /// Measured latency in milliseconds per record (lower is better).
    pub latency: f64,
    /// Member neuron UUIDs, sorted.
    pub members: Vec<String>,
}

impl Point {
    /// A point with its members sorted, so the tiebreak key is stable.
    pub fn new(loss: f64, synapses: usize, latency: f64, mut members: Vec<String>) -> Self {
        members.sort();
        Self {
            loss,
            synapses,
            latency,
            members,
        }
    }

    fn key(&self) -> String {
        self.members.join(",")
    }

    /// `true` when the three measured objectives are exactly equal.
    fn same_objectives(&self, other: &Self) -> bool {
        self.loss == other.loss && self.synapses == other.synapses && self.latency == other.latency
    }
}

/// `true` when `a` dominates `b`.
///
/// Dominance needs `a` no worse than `b` on loss and synapses, and no worse on
/// latency within the tolerance (`a.latency <= b.latency * (1 + tol)`). It also
/// needs `a` better on at least one objective, where latency counts as better
/// only by more than the tolerance (`a.latency < b.latency * (1 - tol)`). Two
/// latencies within the tolerance of each other therefore never decide
/// dominance.
pub fn dominates(a: &Point, b: &Point, latency_tolerance: f64) -> bool {
    let no_worse = a.loss <= b.loss
        && a.synapses <= b.synapses
        && a.latency <= b.latency * (1.0 + latency_tolerance);
    let better = a.loss < b.loss
        || a.synapses < b.synapses
        || a.latency < b.latency * (1.0 - latency_tolerance);
    no_worse && better
}

/// Indices of the non-dominated points, in input order.
///
/// A point whose objectives exactly equal an earlier kept point is dropped,
/// so the first of any exact duplicates is the one that stays.
pub fn non_dominated(points: &[Point], latency_tolerance: f64) -> Vec<usize> {
    let mut kept: Vec<usize> = Vec::new();
    for (j, p) in points.iter().enumerate() {
        let dominated = points.iter().any(|q| dominates(q, p, latency_tolerance));
        let duplicate = kept.iter().any(|&i| points[i].same_objectives(p));
        if !dominated && !duplicate {
            kept.push(j);
        }
    }
    kept
}

/// The archive: the non-dominated points sorted by
/// `(loss, synapses, latency, sorted members joined)` ascending, cut to the
/// `cap` lowest. Returns indices into `points`.
pub fn archive(points: &[Point], cap: usize, latency_tolerance: f64) -> Vec<usize> {
    let mut kept = non_dominated(points, latency_tolerance);
    kept.sort_by(|&a, &b| {
        let (p, q) = (&points[a], &points[b]);
        p.loss
            .total_cmp(&q.loss)
            .then(p.synapses.cmp(&q.synapses))
            .then(p.latency.total_cmp(&q.latency))
            .then_with(|| p.key().cmp(&q.key()))
    });
    kept.truncate(cap);
    kept
}

/// One archived bundle's inputs to the selection.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Choice {
    /// Worst-stratum loss on calibration (lower is better).
    pub loss: f64,
    /// Full calibration delta against the parent.
    pub delta: f64,
    /// Bundle index within the arm.
    pub j: usize,
}

/// Index of the winner among `choices`.
///
/// Only choices whose delta is strictly above `min_improvement` are wins. Of
/// those the lowest loss is chosen. Ties go to the larger delta, then to the
/// lowest `j`. `None` when no choice is a win.
pub fn select_win(choices: &[Choice], min_improvement: f64) -> Option<usize> {
    choices
        .iter()
        .enumerate()
        .filter(|(_, c)| c.delta > min_improvement)
        .min_by(|(_, a), (_, b)| {
            a.loss
                .total_cmp(&b.loss)
                .then(b.delta.total_cmp(&a.delta))
                .then(a.j.cmp(&b.j))
        })
        .map(|(i, _)| i)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pt(loss: f64, synapses: usize, latency: f64, members: &[&str]) -> Point {
        Point::new(
            loss,
            synapses,
            latency,
            members.iter().map(|m| m.to_string()).collect(),
        )
    }

    fn cand(predicted: f64, synapses: usize, members: &[&str]) -> Candidate {
        Candidate::new(
            predicted,
            synapses,
            members.iter().map(|m| m.to_string()).collect(),
        )
    }

    #[test]
    fn a_dominated_point_is_dropped_and_a_non_dominated_one_is_kept() {
        let points = vec![
            pt(1.0, 10, 2.0, &["a"]),
            // Worse than point 0 on every objective: dominated.
            pt(2.0, 11, 3.0, &["b"]),
            // Better on loss, worse on synapses: a trade-off, kept.
            pt(0.5, 12, 2.0, &["c"]),
        ];
        assert!(dominates(&points[0], &points[1], 0.0));
        assert!(!dominates(&points[0], &points[2], 0.0));
        assert_eq!(non_dominated(&points, 0.0), vec![0, 2]);
    }

    #[test]
    fn latency_within_the_tolerance_does_not_decide_dominance() {
        // Same loss and synapses. Latency 0.98 against 1.00 is a 2% gain, which
        // is inside a 5% tolerance, so neither point dominates the other.
        let a = pt(1.0, 5, 0.98, &["a"]);
        let b = pt(1.0, 5, 1.00, &["b"]);
        assert!(
            dominates(&a, &b, 0.0),
            "exact latency comparison sees a gain"
        );
        assert!(!dominates(&a, &b, 0.05));
        assert!(!dominates(&b, &a, 0.05));
        assert_eq!(non_dominated(&[a, b], 0.05), vec![0, 1]);
    }

    #[test]
    fn exact_duplicates_keep_the_first() {
        let points = vec![pt(1.0, 10, 2.0, &["first"]), pt(1.0, 10, 2.0, &["second"])];
        assert_eq!(non_dominated(&points, 0.0), vec![0]);
        assert_eq!(archive(&points, 8, 0.0), vec![0]);
    }

    #[test]
    fn cap_keeps_the_lowest_by_loss_then_synapses_then_latency_then_members() {
        // All mutually non-dominated: the second objective rises as the first falls.
        let points = vec![
            pt(3.0, 1, 1.0, &["d"]),
            pt(1.0, 3, 1.0, &["b"]),
            pt(2.0, 2, 1.0, &["c"]),
            pt(0.0, 4, 1.0, &["a"]),
        ];
        assert_eq!(non_dominated(&points, 0.0).len(), 4);
        // Sorted by loss: 0.0 (idx 3), 1.0 (idx 1), 2.0 (idx 2), 3.0 (idx 0).
        assert_eq!(archive(&points, 2, 0.0), vec![3, 1]);
        assert_eq!(archive(&points, 8, 0.0), vec![3, 1, 2, 0]);
    }

    #[test]
    fn equal_loss_is_ordered_by_synapses_first() {
        // Same loss, a trade-off between synapses and latency: both are
        // non-dominated, and the one with fewer synapses leads.
        let points = vec![pt(1.0, 5, 1.0, &["x"]), pt(1.0, 3, 2.0, &["y"])];
        assert_eq!(non_dominated(&points, 0.0), vec![0, 1]);
        assert_eq!(archive(&points, 8, 0.0), vec![1, 0]);
    }

    #[test]
    fn members_are_sorted_on_construction() {
        let p = pt(0.0, 0, 0.0, &["b", "a"]);
        assert_eq!(p.members, vec!["a".to_string(), "b".to_string()]);
    }

    #[test]
    fn prescreen_drops_repeated_member_sets_and_keeps_the_lowest_predicted() {
        let candidates = [
            cand(3.0, 1, &["a", "b"]),
            cand(1.0, 2, &["c"]),
            cand(2.0, 3, &["d"]),
            // The same member set as candidate 0, drawn later: dropped.
            cand(0.5, 4, &["b", "a"]),
        ];
        assert_eq!(prescreen(&candidates, 2), vec![1, 2]);
        // Three distinct candidates, so a larger budget returns exactly three.
        assert_eq!(prescreen(&candidates, 8), vec![1, 2, 0]);
    }

    #[test]
    fn selection_takes_the_lowest_loss_win_not_the_largest_delta() {
        let choices = [
            Choice {
                loss: 0.10,
                delta: 0.20,
                j: 0,
            },
            // The largest delta among the wins, but not the lowest loss.
            Choice {
                loss: 0.30,
                delta: 0.90,
                j: 1,
            },
            // The lowest loss among the wins.
            Choice {
                loss: 0.05,
                delta: 0.01,
                j: 2,
            },
            // Lowest loss of all, but not a win.
            Choice {
                loss: -1.0,
                delta: -0.5,
                j: 3,
            },
        ];
        assert_eq!(select_win(&choices, 0.0), Some(2));
    }

    #[test]
    fn selection_breaks_loss_ties_by_larger_delta_then_lowest_j() {
        let tied = [
            Choice {
                loss: 0.1,
                delta: 0.2,
                j: 4,
            },
            Choice {
                loss: 0.1,
                delta: 0.3,
                j: 5,
            },
            Choice {
                loss: 0.1,
                delta: 0.3,
                j: 2,
            },
        ];
        assert_eq!(select_win(&tied, 0.0), Some(2));
        let no_win = [Choice {
            loss: 0.1,
            delta: 0.0,
            j: 0,
        }];
        assert_eq!(select_win(&no_win, 0.0), None);
    }
}
