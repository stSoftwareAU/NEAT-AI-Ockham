//! Cascade-aware structural saving estimates (Issue #106).
//!
//! Cutting one hidden neuron strands structure on both sides of it: a neuron
//! that fed only the cut neuron now feeds nothing, and a neuron the cut neuron
//! was the only source for now folds to a constant. [`crate::ablation`] already
//! removes that structure recursively *after* a candidate is built — this
//! module answers, from topology alone and before any scorer time is spent, how
//! much it would remove.
//!
//! The dry run never touches the incumbent. It walks an index of the creature
//! and counts, applying the same two exact rules the cleanup cascade of
//! [`crate::ablation::ablate_mean`] applies, in the same priority order:
//!
//! 1. a listed non-output neuron with no outgoing synapse is removed;
//! 2. a hidden neuron with no incoming synapse folds to a constant and is
//!    removed with its outgoing synapses.
//!
//! Both rules only ever *remove* structure and rule 1 is drained before rule 2
//! is considered, so the estimate for a given creature and cut is the same on
//! every run and under any listing order.
//!
//! Structure the transform refuses is predicted too: an aggregate or unknown
//! squash, an aggregate fold target and a typed edge each make the ablation
//! fail closed, and a cut the razor could never build is reported as saving
//! nothing rather than as the largest cascade on the creature.
//!
//! The estimate is still a prioritisation signal only. It reasons about
//! topology and knows nothing of behaviour, so a candidate it ranks first can
//! still lose. Only the full-corpus scorer accepts a cut.

use std::collections::HashMap;

use neat_core::{CreatureExport, NeuronExport, parse_squash_name};

use crate::ablation::growth_units;

/// Structure a cut is estimated to remove once recursive cleanup has run.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct CascadeEstimate {
    /// Whether the transform would refuse this cut outright.
    ///
    /// An aggregate or unknown squash, an aggregate fold target or a typed edge
    /// makes the ablation fail closed, so nothing is removed and the estimate
    /// is zero. The candidate keeps its place in the sweep — the constant
    /// substitution may still propose one — but it ranks last rather than
    /// first, which is the whole point of predicting the refusal.
    pub blocked: bool,
    /// Requested hidden neurons that are actually on the creature.
    pub requested_neurons: usize,
    /// Further hidden neurons the cleanup would strand and remove.
    pub cascade_hidden: usize,
    /// Cascade hidden neurons left with no incoming synapse — the
    /// constant/foldable structure the cut exposes.
    pub folded_hidden: usize,
    /// Listed non-hidden, non-output neurons (`constant`) the cleanup removes.
    pub cascade_constants: usize,
    /// Synapses removed with all of it.
    pub synapses: usize,
    /// [`growth_units`] the whole removal would save.
    pub growth_units: f64,
}

impl CascadeEstimate {
    /// Hidden neurons removed in total — requested plus cascade.
    pub fn hidden_neurons(&self) -> usize {
        self.requested_neurons + self.cascade_hidden
    }
}

/// What a listed endpoint may have done to it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kind {
    /// Hidden neuron: both cleanup rules apply.
    Hidden,
    /// Output neuron: never removed.
    Output,
    /// Another listed neuron (`constant`): the dead-structure rule applies.
    Listed,
    /// An implicit `input-N` or any endpoint the neuron list does not carry.
    /// The cleanup walks the neuron list, so these are never removed.
    External,
}

/// Topology index of one creature, reused across every candidate cut.
///
/// Built once per creature, `O(neurons + synapses)`. Each estimate then resets
/// scratch state proportional to the creature and walks the structure the cut
/// reaches, so ranking every hidden neuron costs one index and no clone of the
/// creature — where estimating through a real ablation would cost one clone per
/// candidate.
#[derive(Debug, Clone)]
pub struct CascadeIndex<'a> {
    /// Endpoint name → slot; covers listed neurons and implicit inputs.
    slots: HashMap<&'a str, usize>,
    kind: Vec<Kind>,
    uuid: Vec<&'a str>,
    /// Whether the neuron's squash has no scalar the cleanup can fold.
    aggregate: Vec<bool>,
    out_syn: Vec<Vec<usize>>,
    in_syn: Vec<Vec<usize>>,
    syn_from: Vec<usize>,
    syn_to: Vec<usize>,
    /// Whether the synapse carries a role a bias cannot absorb.
    typed: Vec<bool>,
    out_deg: Vec<usize>,
    in_deg: Vec<usize>,
}

/// Whether `neuron`'s squash has no scalar value the cleanup could fold.
///
/// Aggregate squashes are not a sum of their inputs, and a squash NEAT-AI-core
/// cannot parse has no known semantics at all: the transform refuses both
/// rather than guessing, so the dry run counts both as unfoldable.
fn unfoldable(neuron: &NeuronExport) -> bool {
    parse_squash_name(neuron.squash.as_deref().unwrap_or("IDENTITY"))
        .map_or(true, |squash| squash.is_aggregate())
}

impl<'a> CascadeIndex<'a> {
    /// Index `creature`. The creature is read, never written.
    pub fn new(creature: &'a CreatureExport) -> Self {
        let mut index = Self {
            slots: HashMap::with_capacity(creature.neurons.len() + creature.input),
            kind: Vec::with_capacity(creature.neurons.len() + creature.input),
            uuid: Vec::with_capacity(creature.neurons.len() + creature.input),
            aggregate: Vec::with_capacity(creature.neurons.len() + creature.input),
            out_syn: Vec::new(),
            in_syn: Vec::new(),
            syn_from: Vec::with_capacity(creature.synapses.len()),
            syn_to: Vec::with_capacity(creature.synapses.len()),
            typed: Vec::with_capacity(creature.synapses.len()),
            out_deg: Vec::new(),
            in_deg: Vec::new(),
        };
        for neuron in &creature.neurons {
            let kind = match neuron.neuron_type.as_str() {
                "hidden" => Kind::Hidden,
                "output" => Kind::Output,
                _ => Kind::Listed,
            };
            let slot = index.slot_of(neuron.uuid.as_str(), kind);
            index.aggregate[slot] = unfoldable(neuron);
        }
        index.out_syn = vec![Vec::new(); index.kind.len()];
        index.in_syn = vec![Vec::new(); index.kind.len()];
        for (i, synapse) in creature.synapses.iter().enumerate() {
            // An endpoint the neuron list does not carry is an input, or
            // structure Ockham may not touch; either way the cleanup never
            // removes it, so it is indexed as external rather than dropped.
            let from = index.slot_of(synapse.from_uuid.as_str(), Kind::External);
            let to = index.slot_of(synapse.to_uuid.as_str(), Kind::External);
            index.out_syn[from].push(i);
            index.in_syn[to].push(i);
            index.syn_from.push(from);
            index.syn_to.push(to);
            index.typed.push(synapse.synapse_type.is_some());
        }
        index.out_deg = index.out_syn.iter().map(Vec::len).collect();
        index.in_deg = index.in_syn.iter().map(Vec::len).collect();
        index
    }

    /// Slot for `uuid`, allocating it with `kind` when new.
    fn slot_of(&mut self, uuid: &'a str, kind: Kind) -> usize {
        if let Some(&slot) = self.slots.get(uuid) {
            return slot;
        }
        let slot = self.kind.len();
        self.slots.insert(uuid, slot);
        self.kind.push(kind);
        self.uuid.push(uuid);
        self.aggregate.push(false);
        self.out_syn.push(Vec::new());
        self.in_syn.push(Vec::new());
        slot
    }

    /// Structure cutting every hidden neuron in `uuids` would remove.
    ///
    /// UUIDs the creature does not carry as hidden neurons contribute nothing:
    /// only hidden neurons are cut targets. Cutting several at once counts the
    /// structure they share once, so a bundle is not over-credited.
    ///
    /// The cleanup sweeps the whole creature, not only the neighbourhood of the
    /// cut, so structure already stranded on the incumbent is counted too —
    /// that is what the transform really removes, and the estimate is what the
    /// accept is audited against. It is the same addition for every candidate,
    /// so it does not disturb the ranking.
    pub fn estimate(&self, uuids: &[&str]) -> CascadeEstimate {
        let mut state = Sweeper {
            index: self,
            out_deg: self.out_deg.clone(),
            in_deg: self.in_deg.clone(),
            removed: vec![false; self.kind.len()],
            cut_synapse: vec![false; self.syn_from.len()],
            dead_queue: (0..self.kind.len()).collect(),
            fold_queue: (0..self.kind.len()).collect(),
            estimate: CascadeEstimate::default(),
        };
        for uuid in uuids {
            let Some(&slot) = self.slots.get(uuid) else {
                continue;
            };
            if self.kind[slot] != Kind::Hidden || state.removed[slot] {
                continue;
            }
            if state.cut_blocked(slot) {
                return CascadeEstimate {
                    blocked: true,
                    ..CascadeEstimate::default()
                };
            }
            state.estimate.requested_neurons += 1;
            state.strike(slot);
        }
        state.drain();
        let mut estimate = state.estimate;
        if estimate.blocked {
            return CascadeEstimate {
                blocked: true,
                ..CascadeEstimate::default()
            };
        }
        estimate.growth_units = growth_units(estimate.hidden_neurons(), estimate.synapses);
        estimate
    }

    /// One estimate per hidden neuron of the indexed creature.
    ///
    /// The per-creature cache the sweep ranks from: topology does not change
    /// between candidates, so every hidden neuron is estimated once.
    pub fn hidden_estimates(&self) -> HashMap<&'a str, CascadeEstimate> {
        self.kind
            .iter()
            .enumerate()
            .filter(|(_, kind)| **kind == Kind::Hidden)
            .map(|(slot, _)| {
                let uuid = self.uuid[slot];
                (uuid, self.estimate(&[uuid]))
            })
            .collect()
    }
}

/// One dry run over a [`CascadeIndex`]; owns the scratch state it mutates.
struct Sweeper<'a, 'c> {
    index: &'a CascadeIndex<'c>,
    out_deg: Vec<usize>,
    in_deg: Vec<usize>,
    removed: Vec<bool>,
    cut_synapse: Vec<bool>,
    /// Slots to re-test against rule 1 (dead structure).
    dead_queue: Vec<usize>,
    /// Slots to re-test against rule 2 (constant fold).
    fold_queue: Vec<usize>,
    estimate: CascadeEstimate,
}

impl Sweeper<'_, '_> {
    /// Remove `slot` and every synapse incident to it, queueing its neighbours.
    fn strike(&mut self, slot: usize) {
        self.removed[slot] = true;
        for i in 0..self.index.out_syn[slot].len() {
            let syn = self.index.out_syn[slot][i];
            if self.cut_synapse[syn] {
                continue;
            }
            self.cut_synapse[syn] = true;
            self.estimate.synapses += 1;
            let to = self.index.syn_to[syn];
            self.in_deg[to] -= 1;
            self.revisit(to);
        }
        for i in 0..self.index.in_syn[slot].len() {
            let syn = self.index.in_syn[slot][i];
            if self.cut_synapse[syn] {
                continue;
            }
            self.cut_synapse[syn] = true;
            self.estimate.synapses += 1;
            let from = self.index.syn_from[syn];
            self.out_deg[from] -= 1;
            self.revisit(from);
        }
    }

    /// Re-test `slot` against both rules once the graph around it changed.
    fn revisit(&mut self, slot: usize) {
        self.dead_queue.push(slot);
        self.fold_queue.push(slot);
    }

    /// Whether the real transform would refuse to fold `slot` away.
    ///
    /// The cleanup folds a neuron with no incoming synapse into its targets'
    /// biases, and fails the whole candidate when it cannot: an aggregate or
    /// unknown squash has no scalar to fold, an aggregate target is not a sum,
    /// and a typed edge carries a role a bias cannot. Mirroring the refusal is
    /// what keeps the estimate from promising structure the razor can never
    /// take.
    fn fold_blocked(&self, slot: usize) -> bool {
        if self.index.aggregate[slot] {
            return true;
        }
        self.index.out_syn[slot].iter().any(|&syn| {
            !self.cut_synapse[syn]
                && (self.index.typed[syn] || self.index.aggregate[self.index.syn_to[syn]])
        })
    }

    /// Whether the transform would refuse the requested cut of `slot` itself.
    fn cut_blocked(&self, slot: usize) -> bool {
        self.fold_blocked(slot)
            || self.index.in_syn[slot]
                .iter()
                .any(|&syn| !self.cut_synapse[syn] && self.index.typed[syn])
    }

    /// Apply both cleanup rules until nothing more is strandable.
    ///
    /// Rule 1 has strict priority, as it does in the cleanup: dead structure is
    /// drained completely before any constant fold is considered, so a neuron
    /// whose targets die first is counted as dead rather than folded — and
    /// never blocks the candidate for a fold that would not have happened.
    fn drain(&mut self) {
        loop {
            while let Some(slot) = self.dead_queue.pop() {
                if self.removed[slot] || self.out_deg[slot] > 0 {
                    continue;
                }
                match self.index.kind[slot] {
                    Kind::External | Kind::Output => continue,
                    Kind::Hidden => self.estimate.cascade_hidden += 1,
                    Kind::Listed => self.estimate.cascade_constants += 1,
                }
                self.strike(slot);
            }
            let Some(slot) = self.fold_queue.pop() else {
                return;
            };
            if self.removed[slot] || self.index.kind[slot] != Kind::Hidden || self.in_deg[slot] > 0
            {
                continue;
            }
            if self.fold_blocked(slot) {
                self.estimate.blocked = true;
                return;
            }
            self.estimate.cascade_hidden += 1;
            self.estimate.folded_hidden += 1;
            self.strike(slot);
        }
    }
}

/// Structure cutting `uuids` from `creature` would remove ([`CascadeIndex::estimate`]).
///
/// Indexes the creature for this one estimate; rank a whole sweep through a
/// [`CascadeIndex`] instead, which indexes once.
pub fn estimate_cut(creature: &CreatureExport, uuids: &[String]) -> CascadeEstimate {
    let refs: Vec<&str> = uuids.iter().map(String::as_str).collect();
    CascadeIndex::new(creature).estimate(&refs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ablation::{StructureSnapshot, ablate_mean};
    use crate::fixtures::{creature, neuron, synapse};

    /// `input-0 → f1 → f2 → hub → output-0`, plus a lone `keep → output-0`.
    ///
    /// Cutting `hub` strands the whole `f1 → f2` chain behind it: three hidden
    /// neurons and four synapses go, against `keep`'s one neuron and two.
    fn chain() -> CreatureExport {
        creature(
            1,
            1,
            vec![
                neuron("hidden", "f1", 0.0, Some("TANH")),
                neuron("hidden", "f2", 0.0, Some("TANH")),
                neuron("hidden", "hub", 0.0, Some("TANH")),
                neuron("hidden", "keep", 0.0, Some("TANH")),
                neuron("output", "output-0", 0.0, Some("IDENTITY")),
            ],
            vec![
                synapse("input-0", "f1", 1.0),
                synapse("f1", "f2", 1.0),
                synapse("f2", "hub", 1.0),
                synapse("hub", "output-0", 1.0),
                synapse("input-0", "keep", 1.0),
                synapse("keep", "output-0", 1.0),
            ],
        )
    }

    /// `input-0 → src → tail → output-0`, plus `input-0 → other → output-0`.
    ///
    /// Cutting `src` leaves `tail` with no incoming synapse: it still feeds the
    /// output, so it is exposed constant structure rather than dead structure.
    fn fold() -> CreatureExport {
        creature(
            1,
            1,
            vec![
                neuron("hidden", "src", 0.0, Some("TANH")),
                neuron("hidden", "tail", 0.25, Some("TANH")),
                neuron("hidden", "other", 0.0, Some("TANH")),
                neuron("output", "output-0", 0.0, Some("IDENTITY")),
            ],
            vec![
                synapse("input-0", "src", 1.0),
                synapse("src", "tail", 1.0),
                synapse("tail", "output-0", 1.0),
                synapse("input-0", "other", 1.0),
                synapse("other", "output-0", 1.0),
            ],
        )
    }

    fn estimate(creature: &CreatureExport, uuid: &str) -> CascadeEstimate {
        CascadeIndex::new(creature).estimate(&[uuid])
    }

    #[test]
    fn a_cut_that_strands_a_chain_counts_every_neuron_and_synapse_behind_it() {
        let got = estimate(&chain(), "hub");
        assert_eq!(got.requested_neurons, 1, "{got:?}");
        assert_eq!(got.cascade_hidden, 2, "f1 and f2 are stranded: {got:?}");
        assert_eq!(got.folded_hidden, 0, "both are dead, not folded: {got:?}");
        assert_eq!(got.synapses, 4, "{got:?}");
        assert_eq!(got.growth_units, growth_units(3, 4));
    }

    #[test]
    fn a_cut_with_no_cascade_saves_only_its_own_structure() {
        let got = estimate(&chain(), "keep");
        assert_eq!(got.cascade_hidden, 0, "{got:?}");
        assert_eq!(got.synapses, 2, "{got:?}");
        assert_eq!(got.growth_units, growth_units(1, 2));
        assert!(
            got.growth_units < estimate(&chain(), "hub").growth_units,
            "the chain head must outrank the lone neuron"
        );
    }

    #[test]
    fn a_cut_that_leaves_a_neuron_without_incoming_structure_counts_it_as_folded() {
        let got = estimate(&fold(), "src");
        assert_eq!(got.cascade_hidden, 1, "{got:?}");
        assert_eq!(got.folded_hidden, 1, "tail still feeds the output: {got:?}");
        assert_eq!(got.synapses, 3, "{got:?}");
    }

    #[test]
    fn the_estimate_matches_the_structure_the_ablation_actually_removes() {
        let mut compared = 0;
        for fixture in [chain(), fold(), already_stranded()] {
            let hidden: Vec<String> = fixture
                .neurons
                .iter()
                .filter(|n| n.neuron_type == "hidden")
                .map(|n| n.uuid.clone())
                .collect();
            for uuid in &hidden {
                let ablation = ablate_mean(&fixture, uuid, 0.1, None)
                    .unwrap_or_else(|e| panic!("{uuid} must be ablatable on this fixture: {e}"));
                compared += 1;
                let got = estimate(&fixture, uuid);
                let removed_hidden = ablation.before.hidden_neurons - ablation.after.hidden_neurons;
                let removed_synapses = ablation.before.synapses - ablation.after.synapses;
                assert_eq!(got.hidden_neurons(), removed_hidden, "{uuid}: {got:?}");
                assert_eq!(got.synapses, removed_synapses, "{uuid}: {got:?}");
                let actual = ablation.before.growth_units - ablation.after.growth_units;
                assert!(
                    (got.growth_units - actual).abs() < 1e-9,
                    "{uuid}: estimated {} vs actual {actual}: {got:?}",
                    got.growth_units
                );
            }
        }
        // A parity test that silently compared nothing would pass green while
        // the estimate drifted from the cleanup it claims to mirror.
        assert!(compared >= 10, "only {compared} comparisons ran");
    }

    /// The `chain()` topology plus structure the incumbent already stranded.
    ///
    /// `orphan` has no incoming synapse and `dead_end` feeds nothing, so the
    /// cleanup removes both on *any* accepted cut. The estimate has to count
    /// them, or the estimated-versus-actual audit reads as drift.
    fn already_stranded() -> CreatureExport {
        let mut creature = chain();
        creature
            .neurons
            .push(neuron("hidden", "orphan", 0.5, Some("TANH")));
        creature
            .neurons
            .push(neuron("hidden", "dead_end", 0.0, Some("TANH")));
        creature.synapses.push(synapse("orphan", "output-0", 0.25));
        creature.synapses.push(synapse("input-0", "dead_end", 1.0));
        crate::fixtures::sort_synapses_canonically(&mut creature);
        creature
    }

    #[test]
    fn structure_the_incumbent_already_stranded_is_counted_like_the_cleanup_counts_it() {
        let got = estimate(&already_stranded(), "keep");
        // `keep` and its two synapses, plus the folded `orphan` (one synapse)
        // and the dead `dead_end` (one synapse) any cut takes with it.
        assert_eq!(got.hidden_neurons(), 3, "{got:?}");
        assert_eq!(got.folded_hidden, 1, "orphan folds to a constant: {got:?}");
        assert_eq!(got.synapses, 4, "{got:?}");
    }

    /// `input-0 → src → sink → output-0` where `sink` uses `squash`.
    ///
    /// Cutting `src` leaves `sink` with no incoming synapse, so the cleanup has
    /// to fold it — which an aggregate squash refuses.
    fn fold_into(squash: &str, typed: bool) -> CreatureExport {
        let edge = if typed {
            crate::fixtures::typed_synapse("sink", "output-0", 1.0, "condition")
        } else {
            synapse("sink", "output-0", 1.0)
        };
        creature(
            1,
            1,
            vec![
                neuron("hidden", "src", 0.0, Some("TANH")),
                neuron("hidden", "sink", 0.0, Some(squash)),
                neuron("output", "output-0", 0.0, Some("IDENTITY")),
            ],
            vec![
                synapse("input-0", "src", 1.0),
                synapse("src", "sink", 1.0),
                edge,
            ],
        )
    }

    #[test]
    fn a_cut_the_transform_would_refuse_is_estimated_as_saving_nothing() {
        for (squash, typed) in [("MEAN", false), ("NOT_A_SQUASH", false), ("TANH", true)] {
            let fixture = fold_into(squash, typed);
            let got = estimate(&fixture, "src");
            assert!(got.blocked, "{squash} typed={typed}: {got:?}");
            assert_eq!(got.growth_units, 0.0, "{squash} typed={typed}: {got:?}");
            // The transform really does refuse it, which is what the estimate
            // is predicting rather than guessing at.
            assert!(
                ablate_mean(&fixture, "src", 0.1, None).is_err(),
                "{squash} typed={typed} must block the real ablation too"
            );
        }
    }

    #[test]
    fn a_cut_whose_own_structure_is_aggregate_is_estimated_as_saving_nothing() {
        let fixture = creature(
            1,
            1,
            vec![
                neuron("hidden", "agg", 0.0, Some("MEAN")),
                neuron("output", "output-0", 0.0, Some("IDENTITY")),
            ],
            vec![
                synapse("input-0", "agg", 1.0),
                synapse("agg", "output-0", 1.0),
            ],
        );
        let got = estimate(&fixture, "agg");
        assert!(got.blocked, "{got:?}");
        assert_eq!(got.growth_units, 0.0, "{got:?}");
        assert!(ablate_mean(&fixture, "agg", 0.1, None).is_err());
    }

    #[test]
    fn an_empty_cut_and_a_creature_without_hidden_neurons_estimate_nothing() {
        let creature = chain();
        assert_eq!(
            CascadeIndex::new(&creature).estimate(&[]),
            CascadeEstimate::default()
        );
        let bare = crate::fixtures::identity_creature(1, 1);
        let index = CascadeIndex::new(&bare);
        assert!(index.hidden_estimates().is_empty());
        assert_eq!(index.estimate(&["output-0"]), CascadeEstimate::default());
    }

    #[test]
    fn the_estimate_is_deterministic_and_independent_of_listing_order() {
        let creature = chain();
        let first = estimate(&creature, "hub");
        for _ in 0..8 {
            assert_eq!(estimate(&creature, "hub"), first);
        }
        let mut reversed = chain();
        reversed.neurons.reverse();
        reversed.synapses.reverse();
        assert_eq!(estimate(&reversed, "hub"), first);
    }

    #[test]
    fn estimating_never_writes_to_the_creature() {
        let creature = chain();
        let untouched = creature.clone();
        let index = CascadeIndex::new(&creature);
        index.estimate(&["hub"]);
        index.hidden_estimates();
        assert_eq!(creature, untouched);
    }

    #[test]
    fn a_bundle_counts_shared_cascade_structure_once() {
        let creature = chain();
        let both = CascadeIndex::new(&creature).estimate(&["hub", "f2"]);
        assert_eq!(both.requested_neurons, 2, "{both:?}");
        // hub alone already strands f1 and f2: cutting both removes the same
        // three neurons and four synapses, not seven.
        assert_eq!(both.hidden_neurons(), 3, "{both:?}");
        assert_eq!(both.synapses, 4, "{both:?}");
        let separate =
            estimate(&creature, "hub").growth_units + estimate(&creature, "f2").growth_units;
        assert!(both.growth_units < separate, "{both:?}");
    }

    #[test]
    fn unknown_and_non_hidden_uuids_are_estimated_as_no_cuts() {
        let creature = chain();
        let index = CascadeIndex::new(&creature);
        for uuid in ["output-0", "input-0", "nonesuch"] {
            let got = index.estimate(&[uuid]);
            assert_eq!(got, CascadeEstimate::default(), "{uuid}: {got:?}");
        }
    }

    #[test]
    fn every_hidden_neuron_is_estimated_once_per_creature() {
        let creature = chain();
        let index = CascadeIndex::new(&creature);
        let all = index.hidden_estimates();
        assert_eq!(all.len(), 4, "{all:?}");
        for uuid in ["f1", "f2", "hub", "keep"] {
            assert_eq!(all[uuid], index.estimate(&[uuid]), "{uuid}");
        }
    }

    #[test]
    fn the_free_function_agrees_with_the_index() {
        let creature = chain();
        let cut = vec!["hub".to_string()];
        assert_eq!(estimate_cut(&creature, &cut), estimate(&creature, "hub"));
        let snapshot = StructureSnapshot::of(&creature);
        assert!(estimate_cut(&creature, &cut).growth_units < snapshot.growth_units);
    }
}
