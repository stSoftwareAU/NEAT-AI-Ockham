//! Exact zero-risk structural canonicalisation pre-pass (Issue #110).
//!
//! Statistical screening spends scorer budget to *discover* whether a neuron
//! matters. Some structure can be proven not to matter without spending
//! anything, so this pass runs first and removes it — if we can prove the wood
//! is dead, we do not buy an experiment to find out. 🪒
//!
//! Every rule here is **algebraically exact**: the canonicalised creature
//! computes the same function as the incumbent, term for term, rather than an
//! approximation of it. What it does not promise is bit-identical `f32`
//! arithmetic — folding `bias_z += bias_y * b` and composing weights re-order
//! floating-point operations, so outputs agree to rounding, not to the last
//! bit. Nothing in this module reads an activation statistic, a sampled score,
//! or a threshold — an approximate cut belongs in [`crate::ablation`], where
//! the scorer decides.
//!
//! # The rules and their invariants
//!
//! | Rule | Transformation | Why it is exact |
//! |---|---|---|
//! | `dead-structure` | drop a non-output neuron with no outgoing synapse | its value reaches no output, so no output can depend on it |
//! | `constant-fold` | fold a hidden neuron with no incoming synapse into its targets' biases | its activation is the constant `squash(bias)`, and `bias_z += c * w` reproduces the term it contributed |
//! | `zero-weight-synapse` | drop an ordinary synapse of weight exactly `0.0` | it contributes `0.0 * x = 0.0` to a weighted sum for every finite `x` |
//! | `identity-collapse` | eliminate a hidden `IDENTITY` neuron into its downstream biases and weights | `y = bias_y + Σ x_k a_k` substituted into `z`, see [`crate::collapse`] |
//!
//! Duplicate/parallel synapse consolidation needs no rule of its own:
//! NEAT-AI-core `validate_no_duplicate_synapses` already refuses a creature
//! carrying two ordinary synapses over one `(from, to)` pair, and the one
//! transform here that can create such a pair — `identity-collapse` — merges
//! them by adding weights as it writes them ([`crate::collapse`]'s
//! `add_or_merge`). Where duplicates *are* legal (an `IF` target, which reads
//! its synapses by role) merging them is not algebraically equivalent, so this
//! pass leaves them alone.
//!
//! # Safety
//!
//! The rules skip rather than guess: typed synapses and aggregate-squash
//! targets (`MIN`, `MAX`, `IF`, `HYPOT`, `MEAN`) are never rewritten, because
//! an aggregate reduces its whole synapse range at once and dropping a member
//! changes the reduction. Every rewrite is validated through
//! [`validate_creature`] before it is kept; a rewrite that fails is rolled back
//! and recorded in the report rather than dropped silently.
//!
//! # Determinism
//!
//! Rules fire in a fixed order and visit their targets in creature declaration
//! order, so the same incumbent always canonicalises to the same creature and
//! the same report. Every applied rewrite strictly lowers
//! [`crate::ablation::growth_units`], which is what bounds the loop: the pass
//! runs to a fixed point, and a run that somehow exceeds the structural bound
//! fails loudly rather than looping.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt;

use neat_core::{CreatureExport, parse_squash_name};
use serde::{Deserialize, Serialize};

use crate::ablation::{StructureSnapshot, cleanup_cascade, growth_units};
use crate::collapse::{CollapseOptions, CollapseSkip, collapse_identity};
use crate::fixtures::sort_synapses_canonically;
use crate::incumbent::validate_creature;

/// One exact rewrite rule.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ExactRule {
    /// Non-output neuron with no outgoing synapse: no path to any output.
    DeadStructure,
    /// Hidden neuron with no incoming synapse: folded to `squash(bias)`.
    ConstantFold,
    /// Ordinary synapse whose weight is exactly `0.0`.
    ZeroWeightSynapse,
    /// Hidden `IDENTITY` neuron eliminated into its downstream neurons.
    IdentityCollapse,
}

impl ExactRule {
    /// Stable kebab-case rule name used in logs, journals and reports.
    pub fn name(self) -> &'static str {
        match self {
            Self::DeadStructure => "dead-structure",
            Self::ConstantFold => "constant-fold",
            Self::ZeroWeightSynapse => "zero-weight-synapse",
            Self::IdentityCollapse => "identity-collapse",
        }
    }
}

impl fmt::Display for ExactRule {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// Why the pre-pass could not produce a canonicalised creature.
#[derive(Debug, Clone, PartialEq)]
pub enum CleanupError {
    /// The supplied creature failed `creature.validate()` before any rewrite.
    InvalidIncumbent(String),
    /// The canonicalised creature failed `creature.validate()`.
    Invalid(String),
    /// The rule loop did not reach a fixed point within the structural bound.
    ///
    /// Every applied rewrite lowers growth units, so this cannot happen unless
    /// a rule is wrong; it is reported rather than looped on.
    NoFixedPoint {
        /// Passes completed before the bound was hit.
        passes: usize,
    },
}

impl fmt::Display for CleanupError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidIncumbent(m) => {
                write!(
                    f,
                    "incumbent failed creature.validate() before cleanup: {m}"
                )
            }
            Self::Invalid(m) => write!(f, "canonicalised creature failed validation: {m}"),
            Self::NoFixedPoint { passes } => write!(
                f,
                "exact cleanup did not reach a fixed point after {passes} pass(es)"
            ),
        }
    }
}

impl std::error::Error for CleanupError {}

/// A neuron the pre-pass removed.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RemovedNeuron {
    /// Neuron UUID.
    pub uuid: String,
    /// `hidden` or `constant`.
    pub neuron_type: String,
    /// Rule that removed it.
    pub rule: ExactRule,
}

/// A synapse the pre-pass removed.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RemovedSynapse {
    /// Source UUID.
    pub from_uuid: String,
    /// Destination UUID.
    pub to_uuid: String,
    /// Weight it carried.
    pub weight: f64,
    /// Rule that removed it.
    pub rule: ExactRule,
}

/// What one rule achieved over the whole pass.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RuleTally {
    /// The rule.
    pub rule: ExactRule,
    /// Rewrites applied.
    pub applications: usize,
    /// Neurons removed by it.
    pub neurons_removed: usize,
    /// Synapses removed by it. Signed: an `identity-collapse` can rewire a
    /// fan-in × fan-out neuron into more synapses than it removed while its
    /// growth units still fall.
    pub synapses_removed: i64,
    /// Growth units it saved.
    pub growth_units_saved: f64,
}

/// Deterministic record of one canonicalisation pre-pass.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CleanupReport {
    /// True when the creature changed.
    pub changed: bool,
    /// Rule-loop passes, the last of which fired nothing.
    pub passes: usize,
    /// Structure before.
    pub before: StructureSnapshot,
    /// Structure after.
    pub after: StructureSnapshot,
    /// `before.growth_units - after.growth_units`.
    pub growth_units_saved: f64,
    /// Per-rule totals, in rule order; rules that never fired are omitted.
    pub rules: Vec<RuleTally>,
    /// Every removed neuron, in removal order.
    pub removed_neurons: Vec<RemovedNeuron>,
    /// Every removed synapse, in removal order.
    pub removed_synapses: Vec<RemovedSynapse>,
    /// IDENTITY collapses that were offered and declined, by reason code.
    ///
    /// `cost-increase` is the ordinary case — collapsing a wide neuron costs
    /// more structure than it saves. The rest name a topology the pass refused
    /// to guess at.
    pub collapse_skips: BTreeMap<String, usize>,
    /// Rewrites rolled back because the result failed validation.
    ///
    /// Empty on every well-formed creature. A non-empty list is a finding, not
    /// a silence: the pass kept the pre-rewrite creature and said so.
    pub rejected: Vec<String>,
}

impl CleanupReport {
    /// One-line summary for the run log.
    pub fn summary(&self) -> String {
        if !self.changed {
            return "exact cleanup: nothing to remove (already canonical)".to_string();
        }
        let rules = self
            .rules
            .iter()
            .map(|r| format!("{} ×{}", r.rule, r.applications))
            .collect::<Vec<_>>()
            .join(" · ");
        format!(
            "exact cleanup: hidden {} → {}, synapses {} → {}, growth units saved {:.1} \
             in {} pass(es) [{rules}]",
            self.before.hidden_neurons,
            self.after.hidden_neurons,
            self.before.synapses,
            self.after.synapses,
            self.growth_units_saved,
            self.passes,
        )
    }

    /// Hidden neurons removed by the pre-pass.
    pub fn hidden_removed(&self) -> usize {
        self.before
            .hidden_neurons
            .saturating_sub(self.after.hidden_neurons)
    }

    /// Synapses removed by the pre-pass, signed like [`RuleTally::synapses_removed`].
    pub fn synapses_removed(&self) -> i64 {
        self.before.synapses as i64 - self.after.synapses as i64
    }

    /// Fold one rule's contribution into the per-rule totals.
    ///
    /// Growth units are computed from the attributed items rather than from a
    /// before/after difference, because [`crate::ablation::growth_units`] is
    /// linear in `(hidden, synapses)`: attributing each removed neuron and
    /// synapse to the rule that removed it therefore makes the per-rule
    /// figures sum exactly to the pass total.
    fn tally(
        &mut self,
        rule: ExactRule,
        applications: usize,
        neurons_removed: usize,
        hidden_removed: usize,
        synapses_removed: i64,
    ) {
        if let Some(t) = self.rules.iter_mut().find(|t| t.rule == rule) {
            t.applications += applications;
            t.neurons_removed += neurons_removed;
            t.synapses_removed += synapses_removed;
            t.growth_units_saved += rule_growth_units(hidden_removed, synapses_removed);
            return;
        }
        self.rules.push(RuleTally {
            rule,
            applications,
            neurons_removed,
            synapses_removed,
            growth_units_saved: rule_growth_units(hidden_removed, synapses_removed),
        });
        self.rules.sort_by_key(|t| t.rule);
    }
}

/// Growth units saved by removing `hidden` hidden neurons and `synapses`
/// synapses, signed so an added synapse subtracts.
///
/// Composed from [`crate::ablation::growth_units`] rather than restating its
/// formula: it is linear in both terms, so a signed synapse delta is the
/// difference of two calls and the cost model stays in one place.
fn rule_growth_units(hidden: usize, synapses: i64) -> f64 {
    growth_units(hidden, synapses.max(0) as usize)
        - growth_units(0, synapses.min(0).unsigned_abs() as usize)
}

/// The canonicalised creature and the report that explains it.
#[derive(Debug, Clone)]
pub struct Canonicalisation {
    /// Validated, canonicalised creature. Identical to the input when the
    /// report says nothing changed.
    pub creature: CreatureExport,
    /// What was removed and why.
    pub report: CleanupReport,
}

/// Run every exact rewrite rule over `incumbent` to a fixed point.
///
/// Returns the canonicalised creature — behaviourally identical to the
/// incumbent for every finite input — and a deterministic report of what was
/// removed. `incumbent` itself is never modified.
///
/// # Errors
///
/// Fails when the supplied creature is already invalid, when the canonicalised
/// creature fails validation, or when the rule loop does not settle within the
/// structural bound.
pub fn canonicalise(incumbent: &CreatureExport) -> Result<Canonicalisation, CleanupError> {
    validate_creature(incumbent).map_err(|e| CleanupError::InvalidIncumbent(e.to_string()))?;
    let mut working = incumbent.clone();
    let before = StructureSnapshot::of(&working);
    let mut report = CleanupReport {
        changed: false,
        passes: 0,
        before: before.clone(),
        after: before.clone(),
        growth_units_saved: 0.0,
        rules: Vec::new(),
        removed_neurons: Vec::new(),
        removed_synapses: Vec::new(),
        collapse_skips: BTreeMap::new(),
        rejected: Vec::new(),
    };

    // Every applied rewrite removes at least one neuron or one synapse, so the
    // loop cannot run longer than the structure it started with.
    let bound = working.neurons.len() + working.synapses.len() + 2;
    loop {
        if report.passes > bound {
            return Err(CleanupError::NoFixedPoint {
                passes: report.passes,
            });
        }
        report.passes += 1;
        if drop_zero_weight_synapses(&mut working, &mut report) {
            continue;
        }
        if collapse_identities(&mut working, &mut report) {
            continue;
        }
        break;
    }

    report.changed = !report.removed_neurons.is_empty()
        || !report.removed_synapses.is_empty()
        || !report.rules.is_empty();
    if report.changed {
        // The structure changed, so the parent's memetic identity would be a
        // lie — the same reason `collapse` drops it.
        working.memetic = None;
        sort_synapses_canonically(&mut working);
        validate_creature(&working).map_err(|e| CleanupError::Invalid(e.to_string()))?;
    }
    report.after = StructureSnapshot::of(&working);
    report.growth_units_saved = report.before.growth_units - report.after.growth_units;
    Ok(Canonicalisation {
        creature: working,
        report,
    })
}

/// Drop ordinary synapses whose weight is exactly `0.0`, then clean up.
///
/// Exactly zero, never "near zero": `0.0 * x` is `0.0` for every finite `x`, so
/// the term vanishes from the weighted sum. A weight of `1e-18` is small, not
/// zero, and belongs to the scorer-verified proposal path.
///
/// Skipped, never guessed: typed synapses, aggregate-squash targets (an
/// aggregate reduces its whole synapse range, so dropping a member changes the
/// reduction), and the last incoming synapse of an output neuron.
///
/// The drop and the [`cleanup_cascade`] it exposes are one atomic step. A
/// creature mid-drop can be transiently invalid — NEAT-AI-core refuses a hidden
/// neuron with no inward or no outward connection — so validation happens after
/// the cascade has folded what the drop stranded. A batch that fails is retried
/// one synapse at a time, so one refused drop cannot cost the rest.
fn drop_zero_weight_synapses(working: &mut CreatureExport, report: &mut CleanupReport) -> bool {
    let keys = zero_weight_keys(working);
    if keys.is_empty() {
        return false;
    }
    // The batch attempt records nothing: a refusal here is only a finding if
    // the per-synapse retry below refuses too, and a note for a drop that then
    // succeeded would be a failure marker over a healthy run.
    if let Some(step) = try_drop(working, &keys, &mut Vec::new()) {
        commit_drop(working, step, report);
        return true;
    }
    // The batch was refused as a whole; every drop still gets its own hearing.
    let mut applied = false;
    for key in keys {
        let mut refusal = Vec::new();
        if let Some(step) = try_drop(working, std::slice::from_ref(&key), &mut refusal) {
            commit_drop(working, step, report);
            applied = true;
        } else {
            // Deduplicated: the same undroppable synapse is re-offered on every
            // pass, and one finding must not become a page of them.
            for note in refusal {
                if !report.rejected.contains(&note) {
                    report.rejected.push(note);
                }
            }
        }
    }
    applied
}

/// `(from, to)` of every ordinary synapse the zero-weight rule may drop.
fn zero_weight_keys(working: &CreatureExport) -> Vec<(String, String)> {
    let aggregate: HashSet<&str> = working
        .neurons
        .iter()
        .filter(|n| {
            parse_squash_name(n.squash.as_deref().unwrap_or("IDENTITY"))
                .is_ok_and(|s| s.is_aggregate())
        })
        .map(|n| n.uuid.as_str())
        .collect();
    let outputs: HashSet<&str> = working
        .neurons
        .iter()
        .filter(|n| n.neuron_type == "output")
        .map(|n| n.uuid.as_str())
        .collect();
    // Counted down as the batch is selected, not read off the creature: two
    // zero-weight synapses into the same output are each "not the last one"
    // against the untouched count, and dropping both would leave that output
    // with no feed at all.
    let mut remaining: HashMap<&str, usize> = HashMap::new();
    for syn in &working.synapses {
        *remaining.entry(syn.to_uuid.as_str()).or_default() += 1;
    }
    let mut keys = Vec::new();
    for syn in &working.synapses {
        if syn.weight != 0.0
            || syn.synapse_type.is_some()
            || aggregate.contains(syn.to_uuid.as_str())
        {
            continue;
        }
        let left = remaining.get(syn.to_uuid.as_str()).copied().unwrap_or(0);
        // An output left with no feed at all is structure this pass has no
        // exact rewrite for; the scorer-verified path may still cut it.
        if outputs.contains(syn.to_uuid.as_str()) && left <= 1 {
            continue;
        }
        remaining.insert(syn.to_uuid.as_str(), left.saturating_sub(1));
        keys.push((syn.from_uuid.clone(), syn.to_uuid.clone()));
    }
    keys
}

/// One accepted zero-weight step: the creature it produced and what it removed.
struct DropStep {
    creature: CreatureExport,
    dropped: Vec<neat_core::SynapseExport>,
    cascade: Vec<crate::ablation::RemovedNeuron>,
}

/// Build the creature that drops `keys`, cleans up after them, and validates.
///
/// `None` — with the reason pushed to `refusals` — when the cascade refuses the
/// topology or the result fails validation. The caller keeps the creature it
/// had, and decides whether that refusal is a finding worth reporting.
fn try_drop(
    working: &CreatureExport,
    keys: &[(String, String)],
    refusals: &mut Vec<String>,
) -> Option<DropStep> {
    let wanted: HashSet<(&str, &str)> =
        keys.iter().map(|(f, t)| (f.as_str(), t.as_str())).collect();
    let is_dropped = |syn: &neat_core::SynapseExport| {
        syn.synapse_type.is_none()
            && wanted.contains(&(syn.from_uuid.as_str(), syn.to_uuid.as_str()))
    };
    let dropped: Vec<_> = working
        .synapses
        .iter()
        .filter(|syn| is_dropped(syn))
        .cloned()
        .collect();
    if dropped.is_empty() {
        return None;
    }
    let mut candidate = working.clone();
    candidate.synapses.retain(|syn| !is_dropped(syn));
    let mut cascade = Vec::new();
    if let Err(skip) = cleanup_cascade(&mut candidate, &mut Vec::new(), &mut cascade) {
        refusals.push(format!(
            "zero-weight removal of {} synapse(s) not applied: {skip}",
            dropped.len()
        ));
        return None;
    }
    if let Err(e) = validate_creature(&candidate) {
        refusals.push(format!(
            "zero-weight removal of {} synapse(s) rolled back: {e}",
            dropped.len()
        ));
        return None;
    }
    Some(DropStep {
        creature: candidate,
        dropped,
        cascade,
    })
}

/// Record an accepted [`DropStep`] and adopt its creature.
fn commit_drop(working: &mut CreatureExport, step: DropStep, report: &mut CleanupReport) {
    let rule_of: HashMap<&str, ExactRule> = step
        .cascade
        .iter()
        .map(|r| (r.uuid.as_str(), rule_for_reason(r.reason)))
        .collect();
    let dropped_keys: HashSet<(&str, &str)> = step
        .dropped
        .iter()
        .map(|s| (s.from_uuid.as_str(), s.to_uuid.as_str()))
        .collect();

    // Attributed per item, so the per-rule figures sum to the pass totals.
    let mut totals: BTreeMap<ExactRule, (usize, usize, usize, i64)> = BTreeMap::new();
    totals.entry(ExactRule::ZeroWeightSynapse).or_default().0 += step.dropped.len();
    for r in &step.cascade {
        let rule = rule_for_reason(r.reason);
        let entry = totals.entry(rule).or_default();
        entry.0 += 1;
        entry.1 += 1;
        if r.neuron_type == "hidden" {
            entry.2 += 1;
        }
        report.removed_neurons.push(RemovedNeuron {
            uuid: r.uuid.clone(),
            neuron_type: r.neuron_type.clone(),
            rule,
        });
    }
    for syn in removed_synapses(working, &step.creature) {
        let key = (syn.from_uuid.as_str(), syn.to_uuid.as_str());
        let rule = if syn.synapse_type.is_none() && dropped_keys.contains(&key) {
            ExactRule::ZeroWeightSynapse
        } else {
            rule_of
                .get(syn.from_uuid.as_str())
                .or_else(|| rule_of.get(syn.to_uuid.as_str()))
                .copied()
                .unwrap_or(ExactRule::DeadStructure)
        };
        totals.entry(rule).or_default().3 += 1;
        report.removed_synapses.push(RemovedSynapse {
            from_uuid: syn.from_uuid,
            to_uuid: syn.to_uuid,
            weight: syn.weight,
            rule,
        });
    }
    for (rule, (applications, neurons, hidden, synapses)) in totals {
        report.tally(rule, applications, neurons, hidden, synapses);
    }
    *working = step.creature;
}

/// The rule a [`crate::ablation::RemovedNeuron`] reason belongs to.
fn rule_for_reason(reason: &str) -> ExactRule {
    match reason {
        "no-incoming" => ExactRule::ConstantFold,
        _ => ExactRule::DeadStructure,
    }
}

/// Collapse every hidden `IDENTITY` neuron whose removal lowers growth units.
///
/// Reuses [`collapse_identity`] outright rather than re-deriving the algebra:
/// the transform, its skips and its cost gate are the ones the accepted-cut
/// path already uses. Visits neurons in declaration order.
fn collapse_identities(working: &mut CreatureExport, report: &mut CleanupReport) -> bool {
    let candidates: Vec<String> = working
        .neurons
        .iter()
        .filter(|n| {
            n.neuron_type == "hidden"
                && parse_squash_name(n.squash.as_deref().unwrap_or("IDENTITY"))
                    .is_ok_and(|s| s == neat_core::SquashType::Identity)
        })
        .map(|n| n.uuid.clone())
        .collect();
    let mut applied = false;
    for uuid in candidates {
        if !working.neurons.iter().any(|n| n.uuid == uuid) {
            continue;
        }
        let before = StructureSnapshot::of(working);
        match collapse_identity(working, &uuid, CollapseOptions::default()) {
            Ok(collapse) => {
                let after = StructureSnapshot::of(&collapse.creature);
                let gone = removed_neuron_uuids(working, &collapse.creature);
                let hidden = gone.iter().filter(|(_, t)| t == "hidden").count();
                report.tally(
                    ExactRule::IdentityCollapse,
                    1,
                    gone.len(),
                    hidden,
                    before.synapses as i64 - after.synapses as i64,
                );
                for (uuid, neuron_type) in gone {
                    report.removed_neurons.push(RemovedNeuron {
                        uuid,
                        neuron_type,
                        rule: ExactRule::IdentityCollapse,
                    });
                }
                for syn in removed_synapses(working, &collapse.creature) {
                    report.removed_synapses.push(RemovedSynapse {
                        from_uuid: syn.from_uuid,
                        to_uuid: syn.to_uuid,
                        weight: syn.weight,
                        rule: ExactRule::IdentityCollapse,
                    });
                }
                *working = collapse.creature;
                applied = true;
            }
            Err(skip) => {
                *report
                    .collapse_skips
                    .entry(skip_code(&skip).to_string())
                    .or_default() += 1;
            }
        }
    }
    applied
}

/// Stable reason code for a declined collapse.
fn skip_code(skip: &CollapseSkip) -> &'static str {
    match skip {
        CollapseSkip::UnknownNeuron(_) => "unknown-neuron",
        CollapseSkip::NotHidden { .. } => "not-hidden",
        CollapseSkip::NotIdentity { .. } => "not-identity",
        CollapseSkip::TypedSynapse { .. } => "typed-synapse",
        CollapseSkip::AggregateTarget { .. } => "aggregate-target",
        CollapseSkip::SelfLoop { .. } => "self-loop",
        CollapseSkip::CostIncrease { .. } => "cost-increase",
        CollapseSkip::Invalid(_) => "invalid",
    }
}

/// `(uuid, type)` of neurons present in `before` and absent from `after`.
fn removed_neuron_uuids(before: &CreatureExport, after: &CreatureExport) -> Vec<(String, String)> {
    let kept: HashSet<&str> = after.neurons.iter().map(|n| n.uuid.as_str()).collect();
    before
        .neurons
        .iter()
        .filter(|n| !kept.contains(n.uuid.as_str()))
        .map(|n| (n.uuid.clone(), n.neuron_type.clone()))
        .collect()
}

/// Synapses present in `before` and absent from `after`, in `before` order.
fn removed_synapses(
    before: &CreatureExport,
    after: &CreatureExport,
) -> Vec<neat_core::SynapseExport> {
    let kept: HashSet<(&str, &str, Option<&str>)> = after
        .synapses
        .iter()
        .map(|s| {
            (
                s.from_uuid.as_str(),
                s.to_uuid.as_str(),
                s.synapse_type.as_deref(),
            )
        })
        .collect();
    before
        .synapses
        .iter()
        .filter(|s| {
            !kept.contains(&(
                s.from_uuid.as_str(),
                s.to_uuid.as_str(),
                s.synapse_type.as_deref(),
            ))
        })
        .cloned()
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixtures::{creature, neuron, synapse, typed_synapse};
    use neat_core::compile_creature;

    /// Outputs of `creature` over `inputs`, one activation per record.
    fn outputs(creature: &CreatureExport, inputs: &[Vec<f32>]) -> Vec<Vec<f32>> {
        let mut net = compile_creature(creature).expect("compiles");
        inputs
            .iter()
            .map(|x| net.activate(x, creature.output))
            .collect()
    }

    /// Assert the canonicalised creature matches the incumbent output for output.
    fn assert_same_outputs(before: &CreatureExport, after: &CreatureExport, inputs: &[Vec<f32>]) {
        let a = outputs(before, inputs);
        let b = outputs(after, inputs);
        for (x, (u, v)) in inputs.iter().zip(a.iter().zip(&b)) {
            for (p, q) in u.iter().zip(v) {
                assert!((p - q).abs() <= 1e-5, "input {x:?}: {u:?} vs {v:?}");
            }
        }
    }

    fn grid(width: usize) -> Vec<Vec<f32>> {
        [0.0f32, 1.0, -2.5, 3.25]
            .iter()
            .map(|&x| (0..width).map(|i| x + i as f32 * 0.5).collect())
            .collect()
    }

    fn has_neuron(creature: &CreatureExport, uuid: &str) -> bool {
        creature.neurons.iter().any(|n| n.uuid == uuid)
    }

    fn fired(report: &CleanupReport, rule: ExactRule) -> bool {
        report.rules.iter().any(|t| t.rule == rule)
    }

    /// `h_dead`'s only outgoing synapse carries weight exactly zero, so
    /// dropping it strands the neuron; `h_live` feeds the output for real.
    ///
    /// NEAT-AI-core refuses a hidden neuron with no inward or outward
    /// connection, so a *valid* incumbent never carries dead structure
    /// outright: dead wood is what an exact rewrite exposes.
    fn creature_with_dead_branch() -> CreatureExport {
        creature(
            2,
            1,
            vec![
                neuron("hidden", "h_live", 0.1, Some("TANH")),
                neuron("hidden", "h_dead", 0.2, Some("TANH")),
                neuron("output", "output-0", 0.0, Some("IDENTITY")),
            ],
            vec![
                synapse("input-0", "h_live", 0.7),
                synapse("input-1", "h_dead", 0.9),
                synapse("h_live", "output-0", 1.3),
                synapse("h_dead", "output-0", 0.0),
            ],
        )
    }

    #[test]
    fn dead_branch_is_removed_and_outputs_are_identical() {
        let incumbent = creature_with_dead_branch();
        let done = canonicalise(&incumbent).unwrap();
        assert!(done.report.changed);
        assert!(fired(&done.report, ExactRule::DeadStructure));
        assert!(!has_neuron(&done.creature, "h_dead"));
        assert!(has_neuron(&done.creature, "h_live"));
        assert_eq!(done.report.hidden_removed(), 1);
        assert!(done.report.growth_units_saved > 0.0);
        assert!(
            done.report
                .removed_neurons
                .iter()
                .any(|n| n.uuid == "h_dead" && n.rule == ExactRule::DeadStructure)
        );
        assert!(
            done.report
                .removed_synapses
                .iter()
                .any(|s| s.from_uuid == "input-1" && s.to_uuid == "h_dead")
        );
        assert_same_outputs(&incumbent, &done.creature, &grid(2));
        assert_eq!(incumbent.neurons.len(), 3, "incumbent untouched");
    }

    #[test]
    fn a_chain_with_no_path_to_an_output_is_removed_recursively() {
        // `h2 → output-0` carries weight zero, so the whole `h1 → h2` chain
        // loses its path to the output one neuron at a time.
        let incumbent = creature(
            1,
            1,
            vec![
                neuron("hidden", "h1", 0.0, Some("TANH")),
                neuron("hidden", "h2", 0.0, Some("TANH")),
                neuron("hidden", "h3", 0.0, Some("TANH")),
                neuron("output", "output-0", 0.0, Some("IDENTITY")),
            ],
            vec![
                synapse("input-0", "h1", 1.0),
                synapse("input-0", "h3", 1.0),
                synapse("h1", "h2", 1.0),
                synapse("h2", "output-0", 0.0),
                synapse("h3", "output-0", 1.0),
            ],
        );
        let done = canonicalise(&incumbent).unwrap();
        assert!(!has_neuron(&done.creature, "h1"));
        assert!(!has_neuron(&done.creature, "h2"));
        assert!(has_neuron(&done.creature, "h3"));
        assert_same_outputs(&incumbent, &done.creature, &grid(1));
    }

    #[test]
    fn a_hidden_neuron_with_no_incoming_is_constant_folded_exactly() {
        // `input-0 → h_const` carries weight zero, so `h_const` becomes the
        // constant `TANH(bias)` the moment that synapse goes.
        let incumbent = creature(
            1,
            1,
            vec![
                neuron("hidden", "h_const", 0.5, Some("TANH")),
                neuron("output", "output-0", 0.25, Some("IDENTITY")),
            ],
            vec![
                synapse("input-0", "h_const", 0.0),
                synapse("input-0", "output-0", 1.0),
                synapse("h_const", "output-0", 2.0),
            ],
        );
        let done = canonicalise(&incumbent).unwrap();
        assert!(fired(&done.report, ExactRule::ConstantFold));
        assert!(!has_neuron(&done.creature, "h_const"));
        let bias = done
            .creature
            .neurons
            .iter()
            .find(|n| n.uuid == "output-0")
            .unwrap()
            .bias;
        let folded =
            0.25 + f64::from(neat_core::apply_squash(neat_core::SquashType::Tanh, 0.5)) * 2.0;
        assert!((bias - folded).abs() <= 1e-12, "{bias} vs {folded}");
        assert_same_outputs(&incumbent, &done.creature, &grid(1));
    }

    #[test]
    fn exactly_zero_weight_synapses_are_removed() {
        let incumbent = creature(
            2,
            1,
            vec![
                neuron("hidden", "h1", 0.1, Some("TANH")),
                neuron("output", "output-0", 0.0, Some("IDENTITY")),
            ],
            vec![
                synapse("input-0", "h1", 0.8),
                synapse("input-1", "h1", 0.0),
                synapse("h1", "output-0", 1.0),
            ],
        );
        let done = canonicalise(&incumbent).unwrap();
        assert!(fired(&done.report, ExactRule::ZeroWeightSynapse));
        assert!(
            !done
                .creature
                .synapses
                .iter()
                .any(|s| s.from_uuid == "input-1")
        );
        assert!(has_neuron(&done.creature, "h1"));
        assert_same_outputs(&incumbent, &done.creature, &grid(2));
    }

    #[test]
    fn a_tiny_non_zero_weight_is_kept() {
        let incumbent = creature(
            2,
            1,
            vec![
                neuron("hidden", "h1", 0.1, Some("TANH")),
                neuron("output", "output-0", 0.0, Some("IDENTITY")),
            ],
            vec![
                synapse("input-0", "h1", 0.8),
                synapse("input-1", "h1", 1e-18),
                synapse("h1", "output-0", 1.0),
            ],
        );
        let done = canonicalise(&incumbent).unwrap();
        assert!(!done.report.changed, "{}", done.report.summary());
        assert!(
            done.creature
                .synapses
                .iter()
                .any(|s| s.from_uuid == "input-1" && s.weight == 1e-18),
            "a near-zero weight is not a zero weight; only the scorer may cut it"
        );
    }

    #[test]
    fn a_zero_weight_into_an_aggregate_target_is_kept() {
        let incumbent = creature(
            2,
            1,
            vec![
                neuron("hidden", "h_agg", 0.0, Some("MINIMUM")),
                neuron("output", "output-0", 0.0, Some("IDENTITY")),
            ],
            vec![
                synapse("input-0", "h_agg", 1.0),
                synapse("input-1", "h_agg", 0.0),
                synapse("h_agg", "output-0", 1.0),
            ],
        );
        let done = canonicalise(&incumbent).unwrap();
        assert!(
            done.creature
                .synapses
                .iter()
                .any(|s| s.to_uuid == "h_agg" && s.weight == 0.0),
            "an aggregate reduces its whole range; a member may not be dropped"
        );
        assert_same_outputs(&incumbent, &done.creature, &grid(2));
    }

    #[test]
    fn a_zero_weight_typed_synapse_is_kept() {
        let incumbent = creature(
            2,
            1,
            vec![
                neuron("hidden", "h1", 0.0, Some("IF")),
                neuron("output", "output-0", 0.0, Some("IDENTITY")),
            ],
            vec![
                typed_synapse("input-0", "h1", 1.0, "condition"),
                typed_synapse("input-1", "h1", 0.0, "positive"),
                typed_synapse("input-1", "h1", 1.0, "negative"),
                synapse("h1", "output-0", 1.0),
            ],
        );
        let done = canonicalise(&incumbent).unwrap();
        assert_eq!(done.creature.synapses.len(), incumbent.synapses.len());
        assert_same_outputs(&incumbent, &done.creature, &grid(2));
    }

    #[test]
    fn a_hidden_identity_neuron_is_collapsed_exactly() {
        let incumbent = creature(
            1,
            1,
            vec![
                neuron("hidden", "h1", 0.5, Some("IDENTITY")),
                neuron("output", "output-0", 0.25, Some("TANH")),
            ],
            vec![
                synapse("input-0", "h1", 2.0),
                synapse("h1", "output-0", 3.0),
            ],
        );
        let done = canonicalise(&incumbent).unwrap();
        assert!(fired(&done.report, ExactRule::IdentityCollapse));
        assert!(!has_neuron(&done.creature, "h1"));
        assert_same_outputs(&incumbent, &done.creature, &grid(1));
    }

    #[test]
    fn a_cost_increasing_identity_collapse_is_declined_and_counted() {
        let mut neurons = vec![neuron("hidden", "h1", 0.0, Some("IDENTITY"))];
        let mut synapses = Vec::new();
        for i in 0..5 {
            neurons.push(neuron(
                "output",
                &format!("output-{i}"),
                0.0,
                Some("IDENTITY"),
            ));
            synapses.push(synapse(&format!("input-{i}"), "h1", 1.0));
            synapses.push(synapse("h1", &format!("output-{i}"), 1.0));
        }
        let incumbent = creature(5, 5, neurons, synapses);
        let done = canonicalise(&incumbent).unwrap();
        assert!(!done.report.changed);
        assert_eq!(done.report.collapse_skips.get("cost-increase"), Some(&1));
    }

    #[test]
    fn rules_compose_recursively_to_a_fixed_point() {
        // Zero weight kills h_zero's only feed; folding it exposes h_chain,
        // whose IDENTITY collapses into the output.
        let incumbent = creature(
            2,
            1,
            vec![
                neuron("hidden", "h_zero", 0.0, Some("TANH")),
                neuron("hidden", "h_chain", 0.0, Some("IDENTITY")),
                neuron("output", "output-0", 0.0, Some("TANH")),
            ],
            vec![
                synapse("input-0", "h_zero", 0.0),
                synapse("input-1", "h_chain", 1.5),
                synapse("h_zero", "output-0", 0.4),
                synapse("h_chain", "output-0", 2.0),
            ],
        );
        let done = canonicalise(&incumbent).unwrap();
        assert!(done.report.changed, "{}", done.report.summary());
        assert!(
            done.creature
                .neurons
                .iter()
                .all(|n| n.neuron_type != "hidden")
        );
        assert_same_outputs(&incumbent, &done.creature, &grid(2));

        // Fixed point: canonicalising the result again changes nothing.
        let again = canonicalise(&done.creature).unwrap();
        assert!(!again.report.changed, "{}", again.report.summary());
        assert_eq!(
            neat_core::creature_to_json(&again.creature).unwrap(),
            neat_core::creature_to_json(&done.creature).unwrap()
        );
    }

    #[test]
    fn canonicalisation_is_deterministic() {
        let incumbent = creature_with_dead_branch();
        let a = canonicalise(&incumbent).unwrap();
        let b = canonicalise(&incumbent).unwrap();
        assert_eq!(
            neat_core::creature_to_json(&a.creature).unwrap(),
            neat_core::creature_to_json(&b.creature).unwrap()
        );
        assert_eq!(
            serde_json::to_string(&a.report).unwrap(),
            serde_json::to_string(&b.report).unwrap()
        );
    }

    #[test]
    fn an_already_canonical_creature_is_returned_unchanged() {
        let incumbent = creature(
            1,
            1,
            vec![
                neuron("hidden", "h1", 0.1, Some("TANH")),
                neuron("output", "output-0", 0.0, Some("IDENTITY")),
            ],
            vec![
                synapse("input-0", "h1", 0.5),
                synapse("h1", "output-0", 1.0),
            ],
        );
        let done = canonicalise(&incumbent).unwrap();
        assert!(!done.report.changed);
        assert_eq!(done.report.growth_units_saved, 0.0);
        assert!(done.report.removed_neurons.is_empty());
        assert!(done.report.rejected.is_empty());
        assert_eq!(
            neat_core::creature_to_json(&done.creature).unwrap(),
            neat_core::creature_to_json(&incumbent).unwrap()
        );
        assert!(done.report.summary().contains("already canonical"));
    }

    #[test]
    fn the_canonicalised_creature_carries_no_duplicate_synapses() {
        // The IDENTITY collapse writes `input-0 → output-0` where one already
        // exists; `add_or_merge` consolidates them by adding weights.
        let incumbent = creature(
            1,
            1,
            vec![
                neuron("hidden", "h1", 0.0, Some("IDENTITY")),
                neuron("output", "output-0", 0.0, Some("IDENTITY")),
            ],
            vec![
                synapse("input-0", "h1", 1.0),
                synapse("input-0", "output-0", 0.5),
                synapse("h1", "output-0", 1.0),
            ],
        );
        let done = canonicalise(&incumbent).unwrap();
        assert_eq!(done.creature.synapses.len(), 1);
        assert!((done.creature.synapses[0].weight - 1.5).abs() <= 1e-12);
        neat_core::validate_no_duplicate_synapses(&done.creature).unwrap();
        assert_same_outputs(&incumbent, &done.creature, &grid(1));
    }

    #[test]
    fn an_invalid_incumbent_fails_loudly() {
        let mut incumbent = creature_with_dead_branch();
        incumbent.forward_only = false;
        let err = canonicalise(&incumbent).unwrap_err();
        assert!(matches!(err, CleanupError::InvalidIncumbent(_)), "{err}");
        assert!(err.to_string().contains("before cleanup"));
    }

    #[test]
    fn the_report_totals_agree_with_the_snapshots() {
        let incumbent = creature_with_dead_branch();
        let done = canonicalise(&incumbent).unwrap();
        let report = &done.report;
        assert_eq!(report.after, StructureSnapshot::of(&done.creature));
        assert!(
            (report.growth_units_saved - (report.before.growth_units - report.after.growth_units))
                .abs()
                <= 1e-12
        );
        let by_rule: f64 = report.rules.iter().map(|t| t.growth_units_saved).sum();
        assert!(
            (by_rule - report.growth_units_saved).abs() <= 1e-9,
            "per-rule {by_rule} vs total {}",
            report.growth_units_saved
        );
        assert_eq!(
            report
                .rules
                .iter()
                .map(|t| t.neurons_removed)
                .sum::<usize>(),
            report.removed_neurons.len()
        );
    }

    #[test]
    fn two_zero_weight_feeds_of_one_output_never_both_go() {
        // Against the untouched incoming count neither synapse is "the last
        // one", so a batch that does not count down as it selects would strand
        // the output with no feed at all.
        let incumbent = creature(
            2,
            1,
            vec![neuron("output", "output-0", 0.4, Some("IDENTITY"))],
            vec![
                synapse("input-0", "output-0", 0.0),
                synapse("input-1", "output-0", 0.0),
            ],
        );
        let done = canonicalise(&incumbent).unwrap();
        assert_eq!(
            done.creature.synapses.len(),
            1,
            "one feed must survive: {:?}",
            done.creature.synapses
        );
        assert_same_outputs(&incumbent, &done.creature, &grid(2));
    }

    #[test]
    fn the_last_incoming_synapse_of_an_output_is_kept() {
        let incumbent = creature(
            1,
            1,
            vec![neuron("output", "output-0", 0.4, Some("IDENTITY"))],
            vec![synapse("input-0", "output-0", 0.0)],
        );
        let done = canonicalise(&incumbent).unwrap();
        assert!(!done.report.changed);
        assert_eq!(done.creature.synapses.len(), 1);
    }
}
