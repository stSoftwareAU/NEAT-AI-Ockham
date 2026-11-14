//! The Pareto archive for the `pareto` arm (issue #263).
//!
//! Three objectives, all minimised: predicted worst-stratum cost, synapse
//! count after the cut, and measured latency. The archive only chooses which
//! candidates get the full scorer; it never selects anything itself.

/// One candidate bundle's objectives and members.
#[derive(Debug, Clone, PartialEq)]
pub struct Point {
    /// Predicted worst-stratum cost (lower is better).
    pub predicted: f64,
    /// Synapse count after the cut (lower is better).
    pub synapses: usize,
    /// Measured latency in milliseconds per record (lower is better).
    pub latency: f64,
    /// Member neuron UUIDs, sorted.
    pub members: Vec<String>,
}

impl Point {
    /// A point with its members sorted, so the tiebreak key is stable.
    pub fn new(predicted: f64, synapses: usize, latency: f64, mut members: Vec<String>) -> Self {
        members.sort();
        Self {
            predicted,
            synapses,
            latency,
            members,
        }
    }

    fn objectives(&self) -> [f64; 3] {
        [self.predicted, self.synapses as f64, self.latency]
    }

    fn key(&self) -> String {
        self.members.join(",")
    }
}

/// `true` when `a` is no worse than `b` on every objective and strictly better
/// on at least one.
pub fn dominates(a: &Point, b: &Point) -> bool {
    let (x, y) = (a.objectives(), b.objectives());
    let no_worse = x.iter().zip(&y).all(|(p, q)| p <= q);
    let better = x.iter().zip(&y).any(|(p, q)| p < q);
    no_worse && better
}

/// Indices of the non-dominated points, in input order.
///
/// A point whose objectives exactly equal an earlier kept point is dropped,
/// so the first of any exact duplicates is the one that stays.
pub fn non_dominated(points: &[Point]) -> Vec<usize> {
    let mut kept: Vec<usize> = Vec::new();
    for (j, p) in points.iter().enumerate() {
        let dominated = points.iter().any(|q| dominates(q, p));
        let duplicate = kept
            .iter()
            .any(|&i| points[i].objectives() == p.objectives());
        if !dominated && !duplicate {
            kept.push(j);
        }
    }
    kept
}

/// The archive: the non-dominated points sorted by
/// `(predicted, synapses, latency, sorted members joined)` ascending, cut to
/// the `cap` lowest. Returns indices into `points`.
pub fn archive(points: &[Point], cap: usize) -> Vec<usize> {
    let mut kept = non_dominated(points);
    kept.sort_by(|&a, &b| {
        let (p, q) = (&points[a], &points[b]);
        p.predicted
            .total_cmp(&q.predicted)
            .then(p.synapses.cmp(&q.synapses))
            .then(p.latency.total_cmp(&q.latency))
            .then_with(|| p.key().cmp(&q.key()))
    });
    kept.truncate(cap);
    kept
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pt(predicted: f64, synapses: usize, latency: f64, members: &[&str]) -> Point {
        Point::new(
            predicted,
            synapses,
            latency,
            members.iter().map(|m| m.to_string()).collect(),
        )
    }

    #[test]
    fn a_dominated_point_is_dropped_and_a_non_dominated_one_is_kept() {
        let points = vec![
            pt(1.0, 10, 2.0, &["a"]),
            // Worse than point 0 on every objective: dominated.
            pt(2.0, 11, 3.0, &["b"]),
            // Better on predicted, worse on synapses: a trade-off, kept.
            pt(0.5, 12, 2.0, &["c"]),
        ];
        assert!(dominates(&points[0], &points[1]));
        assert!(!dominates(&points[0], &points[2]));
        assert_eq!(non_dominated(&points), vec![0, 2]);
    }

    #[test]
    fn exact_duplicates_keep_the_first() {
        let points = vec![pt(1.0, 10, 2.0, &["first"]), pt(1.0, 10, 2.0, &["second"])];
        assert_eq!(non_dominated(&points), vec![0]);
        assert_eq!(archive(&points, 8), vec![0]);
    }

    #[test]
    fn cap_keeps_the_lowest_by_predicted_then_synapses_then_latency_then_members() {
        // All mutually non-dominated: the second objective rises as the first falls.
        let points = vec![
            pt(3.0, 1, 1.0, &["d"]),
            pt(1.0, 3, 1.0, &["b"]),
            pt(2.0, 2, 1.0, &["c"]),
            pt(0.0, 4, 1.0, &["a"]),
        ];
        assert_eq!(non_dominated(&points).len(), 4);
        // Sorted by predicted: 0.0 (idx 3), 1.0 (idx 1), 2.0 (idx 2), 3.0 (idx 0).
        assert_eq!(archive(&points, 2), vec![3, 1]);
        assert_eq!(archive(&points, 8), vec![3, 1, 2, 0]);
    }

    #[test]
    fn equal_predicted_cost_is_ordered_by_synapses_first() {
        // Same predicted cost, a trade-off between synapses and latency: both
        // are non-dominated, and the one with fewer synapses leads.
        let points = vec![pt(1.0, 5, 1.0, &["x"]), pt(1.0, 3, 2.0, &["y"])];
        assert_eq!(non_dominated(&points), vec![0, 1]);
        assert_eq!(archive(&points, 8), vec![1, 0]);
    }

    #[test]
    fn members_are_sorted_on_construction() {
        let p = pt(0.0, 0, 0.0, &["b", "a"]);
        assert_eq!(p.members, vec!["a".to_string(), "b".to_string()]);
    }
}
