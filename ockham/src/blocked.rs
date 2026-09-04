//! Why the razor could propose nothing for a visit (Issue #103).
//!
//! `blocked` used to be one number: "the sweep has been to N neurons and could
//! not test any of them". One number cannot be attacked. A forest-heavy
//! creature blocks four hidden neurons in five, and until the population is
//! broken down by *reason* there is no way to tell the category worth building
//! a new proposal path for from the category that is genuinely untestable.
//!
//! So every blocked visit now carries a [`BlockedReason`] — a small, stable,
//! deterministic set of codes — the record on disk keeps it, and every
//! reporting surface counts by it.
//!
//! `blocked` still never means *not pruneable forever*. It means the current
//! proposal mechanism does not know how to test this neuron safely, and the
//! code says which mechanism is missing.
//!
//! The codes are **strings on the record**, never a serialised enum: the fleet
//! runs mixed versions against one shared store, and an unknown code read by an
//! older binary must degrade to [`BlockedReason::Other`] rather than fail the
//! load of every record beside it.

use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// Why one visited neuron produced no candidate to score.
///
/// Serialised as its [`BlockedReason::code`] string, never as an enum variant:
/// serde rejects an unknown variant and would fail the load of every record
/// beside it, and the fleet runs mixed versions against one shared store.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BlockedReason {
    /// The neuron, or something the fold would touch, uses an aggregate squash.
    ///
    /// `IF`, `MEAN`, `MINIMUM`, … do not sum their inputs, so a mean-activation
    /// substitution cannot be folded into a downstream bias. The dominant
    /// category on a forest-heavy creature.
    AggregateSquash,
    /// No usable activation statistic: none sampled, or a non-finite mean.
    MissingActivation,
    /// The topology cannot be compensated safely — a typed synapse, or a
    /// neuron the transform cannot treat as an ordinary hidden unit.
    UnsafeTopology,
    /// A candidate was built and `creature.validate()` rejected it.
    ValidationFailed,
    /// The neuron feeds nothing, so no candidate could be built around it.
    ///
    /// NEAT-AI-core rejects a hidden neuron with no outgoing edge (rule 18), so
    /// a validated incumbent holds none: this is what
    /// [`crate::substitute::substitute_constant`] reports when a transform in
    /// progress leaves one, rather than emitting a candidate that cannot
    /// validate.
    NoOutputPath,
    /// An explicit reason outside the codes above, including a code written by
    /// a newer binary than the one reading it.
    Other,
    /// The record predates Issue #103 and carries no reason at all.
    ///
    /// Distinct from [`Self::Other`] on purpose: "we did not record why" is not
    /// the same finding as "we recorded a reason none of the codes covers".
    Unrecorded,
}

impl BlockedReason {
    /// Every code, in the fixed order reporting falls back to.
    pub const ALL: [BlockedReason; 7] = [
        Self::AggregateSquash,
        Self::MissingActivation,
        Self::UnsafeTopology,
        Self::ValidationFailed,
        Self::NoOutputPath,
        Self::Other,
        Self::Unrecorded,
    ];

    /// Stable wire code — what the screen record and every report carry.
    pub fn code(self) -> &'static str {
        match self {
            Self::AggregateSquash => "aggregate-squash",
            Self::MissingActivation => "missing-activation",
            Self::UnsafeTopology => "unsafe-topology",
            Self::ValidationFailed => "validation-failed",
            Self::NoOutputPath => "no-output-path",
            Self::Other => "other",
            Self::Unrecorded => "unrecorded",
        }
    }

    /// One clause naming what the razor would need to test this neuron.
    pub fn describe(self) -> &'static str {
        match self {
            Self::AggregateSquash => {
                "aggregate squash semantics — a mean substitution cannot fold into a non-sum input"
            }
            Self::MissingActivation => "no finite sampled activation statistic to substitute",
            Self::UnsafeTopology => "topology cannot be compensated safely",
            Self::ValidationFailed => "the candidate failed creature.validate()",
            Self::NoOutputPath => "no outgoing contribution path, and no valid exact removal",
            Self::Other => "an explicit reason outside the known codes",
            Self::Unrecorded => "filed before blocked reasons were recorded (#103)",
        }
    }

    /// Read a code back off a record.
    ///
    /// An unknown code is [`Self::Other`] rather than an error: a newer host may
    /// have written a code this binary has never heard of, and dropping the
    /// record would understate the blocked population it is counted in.
    pub fn from_code(code: &str) -> Self {
        Self::ALL
            .into_iter()
            .find(|r| r.code() == code)
            .unwrap_or(Self::Other)
    }
}

impl Serialize for BlockedReason {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(self.code())
    }
}

impl<'de> Deserialize<'de> for BlockedReason {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let code = String::deserialize(deserializer)?;
        Ok(Self::from_code(&code))
    }
}

/// The blocked population of one epoch, split by reason (Issue #103).
///
/// `Copy`, and one `usize` per code rather than a map, so [`crate::Coverage`]
/// stays `Copy` and the JSON shape is fixed: a consumer never has to guess
/// which keys a run might emit.
///
/// The counts are over **UUIDs**, not records, so they sum to exactly the
/// `blocked` total beside them — that invariant is what makes the breakdown
/// usable as a work list.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct BlockedBreakdown {
    /// Blocked by aggregate/unsupported squash semantics.
    pub aggregate_squash: usize,
    /// Blocked by a missing or non-finite activation statistic.
    pub missing_activation: usize,
    /// Blocked because the topology cannot be compensated safely.
    pub unsafe_topology: usize,
    /// Blocked by a candidate that failed validation.
    pub validation_failed: usize,
    /// Blocked with no outgoing contribution path and no valid exact removal.
    pub no_output_path: usize,
    /// Blocked for an explicit reason outside the known codes.
    pub other: usize,
    /// Blocked by a record filed before reasons were recorded.
    pub unrecorded: usize,
}

impl BlockedBreakdown {
    /// Count one blocked uuid under `reason`.
    pub fn add(&mut self, reason: BlockedReason) {
        *self.slot(reason) += 1;
    }

    /// How many blocked UUIDs carry `reason`.
    pub fn count(&self, reason: BlockedReason) -> usize {
        match reason {
            BlockedReason::AggregateSquash => self.aggregate_squash,
            BlockedReason::MissingActivation => self.missing_activation,
            BlockedReason::UnsafeTopology => self.unsafe_topology,
            BlockedReason::ValidationFailed => self.validation_failed,
            BlockedReason::NoOutputPath => self.no_output_path,
            BlockedReason::Other => self.other,
            BlockedReason::Unrecorded => self.unrecorded,
        }
    }

    /// Every counted reason, commonest first, ties broken by code.
    ///
    /// Deterministic for a given set of records however they were read: the
    /// same run on the same store renders the same line every time. Reasons
    /// with no blocked neurons are omitted — a work list should not be padded
    /// with categories that are not there.
    pub fn entries(&self) -> Vec<(BlockedReason, usize)> {
        let mut out: Vec<(BlockedReason, usize)> = BlockedReason::ALL
            .into_iter()
            .map(|r| (r, self.count(r)))
            .filter(|(_, n)| *n > 0)
            .collect();
        out.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.code().cmp(b.0.code())));
        out
    }

    /// Blocked UUIDs across every reason — equal to `Coverage::blocked`.
    pub fn total(&self) -> usize {
        BlockedReason::ALL
            .into_iter()
            .map(|r| self.count(r))
            .sum::<usize>()
    }

    /// The largest blocked category, or `None` when nothing is blocked.
    ///
    /// The one figure the razor's next proposal path should be aimed at.
    pub fn dominant(&self) -> Option<(BlockedReason, usize)> {
        self.entries().first().copied()
    }

    /// `aggregate-squash 380 (92.2%) · unsafe-topology 20 (4.9%)`, or `None`.
    ///
    /// Percentages are of the blocked total, not of the creature: this line
    /// answers "what is blocking the sweep?", and the `blocked:` line beside it
    /// already says how much of the creature that is.
    pub fn summary(&self) -> Option<String> {
        let total = self.total();
        if total == 0 {
            return None;
        }
        Some(
            self.entries()
                .into_iter()
                .map(|(reason, n)| {
                    format!(
                        "{} {n} ({:.1}%)",
                        reason.code(),
                        n as f64 / total as f64 * 100.0
                    )
                })
                .collect::<Vec<_>>()
                .join(" · "),
        )
    }

    fn slot(&mut self, reason: BlockedReason) -> &mut usize {
        match reason {
            BlockedReason::AggregateSquash => &mut self.aggregate_squash,
            BlockedReason::MissingActivation => &mut self.missing_activation,
            BlockedReason::UnsafeTopology => &mut self.unsafe_topology,
            BlockedReason::ValidationFailed => &mut self.validation_failed,
            BlockedReason::NoOutputPath => &mut self.no_output_path,
            BlockedReason::Other => &mut self.other,
            BlockedReason::Unrecorded => &mut self.unrecorded,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn breakdown(pairs: &[(BlockedReason, usize)]) -> BlockedBreakdown {
        let mut out = BlockedBreakdown::default();
        for (reason, n) in pairs {
            for _ in 0..*n {
                out.add(*reason);
            }
        }
        out
    }

    #[test]
    fn every_code_round_trips_through_its_wire_form() {
        for reason in BlockedReason::ALL {
            assert_eq!(
                BlockedReason::from_code(reason.code()),
                reason,
                "{} must survive a write and a read",
                reason.code()
            );
        }
    }

    /// A code from a newer binary is counted, not dropped: the neuron really is
    /// blocked, and the reading host must not understate the population.
    #[test]
    fn an_unknown_code_reads_as_other_rather_than_failing() {
        assert_eq!(
            BlockedReason::from_code("quantum-entanglement"),
            BlockedReason::Other
        );
    }

    /// The acceptance criterion: the reason counts sum to the blocked total.
    #[test]
    fn the_reason_counts_sum_to_the_blocked_total() {
        let b = breakdown(&[
            (BlockedReason::AggregateSquash, 380),
            (BlockedReason::UnsafeTopology, 20),
            (BlockedReason::MissingActivation, 12),
            (BlockedReason::Unrecorded, 5),
        ]);
        assert_eq!(b.total(), 417);
        assert_eq!(
            b.entries().iter().map(|(_, n)| n).sum::<usize>(),
            b.total(),
            "the rendered entries must account for every blocked uuid"
        );
    }

    #[test]
    fn the_breakdown_is_ordered_commonest_first_and_ties_break_on_the_code() {
        let b = breakdown(&[
            (BlockedReason::MissingActivation, 7),
            (BlockedReason::AggregateSquash, 7),
            (BlockedReason::UnsafeTopology, 9),
        ]);
        let codes: Vec<&str> = b.entries().iter().map(|(r, _)| r.code()).collect();
        assert_eq!(
            codes,
            vec!["unsafe-topology", "aggregate-squash", "missing-activation"]
        );
    }

    #[test]
    fn a_reason_with_no_blocked_neurons_is_not_rendered() {
        let b = breakdown(&[(BlockedReason::AggregateSquash, 3)]);
        assert_eq!(b.entries().len(), 1);
        assert_eq!(b.summary().as_deref(), Some("aggregate-squash 3 (100.0%)"));
    }

    #[test]
    fn nothing_blocked_renders_no_line_and_names_no_dominant_category() {
        let b = BlockedBreakdown::default();
        assert_eq!(b.total(), 0);
        assert_eq!(b.summary(), None);
        assert_eq!(b.dominant(), None);
    }

    #[test]
    fn the_dominant_category_is_the_largest_one() {
        let b = breakdown(&[
            (BlockedReason::AggregateSquash, 380),
            (BlockedReason::UnsafeTopology, 20),
        ]);
        assert_eq!(b.dominant(), Some((BlockedReason::AggregateSquash, 380)));
        assert!(
            b.summary()
                .expect("blocked neurons render a line")
                .starts_with("aggregate-squash 380 (95.0%)"),
            "{:?}",
            b.summary()
        );
    }

    /// The JSON keys are the reporting contract: a consumer reads fixed keys
    /// rather than guessing which reasons a run happened to hit.
    #[test]
    fn the_breakdown_serialises_every_code_as_a_fixed_camel_case_key() {
        let json = serde_json::to_string(&breakdown(&[(BlockedReason::NoOutputPath, 2)])).unwrap();
        for key in [
            "aggregateSquash",
            "missingActivation",
            "unsafeTopology",
            "validationFailed",
            "noOutputPath",
            "other",
            "unrecorded",
        ] {
            assert!(json.contains(key), "{key} missing from {json}");
        }
        assert!(json.contains("\"noOutputPath\":2"), "{json}");
        let back: BlockedBreakdown = serde_json::from_str(&json).unwrap();
        assert_eq!(back.no_output_path, 2);
    }

    /// A reason travels as its code, and a code from a newer binary is read
    /// rather than failing the record it arrived on.
    #[test]
    fn a_reason_serialises_as_its_code_string() {
        let json = serde_json::to_string(&BlockedReason::AggregateSquash).unwrap();
        assert_eq!(json, "\"aggregate-squash\"");
        let back: BlockedReason = serde_json::from_str(&json).unwrap();
        assert_eq!(back, BlockedReason::AggregateSquash);
        let unknown: BlockedReason = serde_json::from_str("\"from-the-future\"").unwrap();
        assert_eq!(unknown, BlockedReason::Other);
    }

    /// An artefact written before #103 still deserialises, as nothing blocked.
    #[test]
    fn a_pre_103_breakdown_reads_as_no_reasons_at_all() {
        let back: BlockedBreakdown = serde_json::from_str("{}").unwrap();
        assert_eq!(back, BlockedBreakdown::default());
    }
}
