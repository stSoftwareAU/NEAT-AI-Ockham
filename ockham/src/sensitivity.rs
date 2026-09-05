//! Downstream output sensitivity of each hidden neuron (Issue #105).
//!
//! A neuron can fire hard and still change nothing: the weights and topology
//! between it and the outputs may attenuate everything it produces. Activation
//! statistics alone cannot see that — they measure how loud a neuron is, not
//! whether anything downstream listens. This module answers the second
//! question, from topology alone and before any scorer time is spent.
//!
//! The estimate is the first-order backward propagation the issue describes:
//!
//! ```text
//! importance(output) = 1
//! importance(N)      = Σ abs(weight(N → child)) × importance(child)
//! ```
//!
//! It is built once per creature in `O(neurons + synapses)` work: the graph is
//! condensed into its strongly connected components, which are emitted in
//! reverse topological order, and each component is resolved from components
//! already resolved behind it.
//!
//! Recurrent and otherwise cyclic topology is handled **conservatively**. A
//! cycle has no first-order fixpoint that a single backward pass can read — the
//! series diverges as soon as the loop gain reaches one — so the component is
//! relaxed once against the largest importance any member sends out of it.
//! Every member keeps at least that exit, and a member that amplifies into it
//! is credited for that hop. The loop is therefore never ranked as dead wood
//! merely because it could not be resolved, and the answer does not depend on
//! which member the walk happened to enter by.
//!
//! Outputs anchor the recursion at `1` and are never propagated through: an
//! output's own outgoing structure, if the creature carries any, cannot make it
//! matter more or less than the thing the creature is scored on.
//!
//! Like every other ranking signal in Ockham this is a **prioritisation
//! heuristic only**. It is first-order and knows nothing of squash saturation
//! or behaviour, so a neuron it ranks first still faces `creature.validate()`,
//! the sampled screen and full authoritative scoring before anything is
//! removed.

use std::collections::HashMap;

use neat_core::CreatureExport;

/// What a listed endpoint is, for the backward walk.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kind {
    /// Hidden neuron: ranked, and propagated through.
    Hidden,
    /// Output neuron: anchors the recursion at `1`.
    Output,
    /// Any other endpoint — a listed constant, or an implicit `input-N`.
    Other,
}

/// Per-neuron downstream output sensitivity for one creature.
///
/// Built once per creature and read for every candidate: the topology does not
/// change while a visitation order is built, so no candidate pays to rebuild
/// it. The creature is read, never written.
#[derive(Debug, Clone)]
pub struct SensitivityIndex<'a> {
    /// Endpoint name → slot; covers listed neurons and implicit inputs.
    slots: HashMap<&'a str, usize>,
    uuid: Vec<&'a str>,
    kind: Vec<Kind>,
    importance: Vec<f64>,
}

impl<'a> SensitivityIndex<'a> {
    /// Index `creature` and propagate importance backwards from its outputs.
    pub fn new(creature: &'a CreatureExport) -> Self {
        let mut builder = Builder::new(creature);
        builder.wire(creature);
        builder.solve()
    }

    /// Downstream output sensitivity of `uuid`, when the creature carries it.
    ///
    /// `None` for an endpoint the creature does not have. Callers rank a
    /// missing value last rather than dropping the neuron: an ordering may
    /// never remove a candidate from the sweep.
    pub fn importance(&self, uuid: &str) -> Option<f64> {
        self.slots.get(uuid).map(|&slot| self.importance[slot])
    }

    /// One importance per hidden neuron — the per-creature ranking cache.
    pub fn hidden_importance(&self) -> HashMap<&'a str, f64> {
        self.kind
            .iter()
            .enumerate()
            .filter(|(_, kind)| **kind == Kind::Hidden)
            .map(|(slot, _)| (self.uuid[slot], self.importance[slot]))
            .collect()
    }
}

/// Scratch graph the index is solved from.
struct Builder<'a> {
    slots: HashMap<&'a str, usize>,
    uuid: Vec<&'a str>,
    kind: Vec<Kind>,
    /// Outgoing `(target slot, abs weight)` per slot. Outputs carry none.
    out: Vec<Vec<(usize, f64)>>,
}

impl<'a> Builder<'a> {
    /// Slot every listed neuron of `creature`. The creature is read, never written.
    fn new(creature: &'a CreatureExport) -> Self {
        let capacity = creature.neurons.len() + creature.input;
        let mut builder = Self {
            slots: HashMap::with_capacity(capacity),
            uuid: Vec::with_capacity(capacity),
            kind: Vec::with_capacity(capacity),
            out: Vec::with_capacity(capacity),
        };
        for neuron in &creature.neurons {
            let kind = match neuron.neuron_type.as_str() {
                "hidden" => Kind::Hidden,
                "output" => Kind::Output,
                _ => Kind::Other,
            };
            let slot = builder.slot_of(neuron.uuid.as_str());
            builder.kind[slot] = kind;
        }
        builder
    }

    /// Slot for `uuid`, allocating it as [`Kind::Other`] when new.
    ///
    /// An endpoint the neuron list does not carry is an implicit `input-N` or
    /// structure Ockham may not touch; either way it is a source the walk
    /// passes through, never a ranked candidate.
    fn slot_of(&mut self, uuid: &'a str) -> usize {
        if let Some(&slot) = self.slots.get(uuid) {
            return slot;
        }
        let slot = self.uuid.len();
        self.slots.insert(uuid, slot);
        self.uuid.push(uuid);
        self.kind.push(Kind::Other);
        self.out.push(Vec::new());
        slot
    }

    /// Add every synapse as a weighted backward-propagation edge.
    ///
    /// Edges leaving an output are dropped: an output anchors the recursion, so
    /// propagating through it would let structure behind the score decide how
    /// much the score matters.
    fn wire(&mut self, creature: &'a CreatureExport) {
        for synapse in &creature.synapses {
            let from = self.slot_of(synapse.from_uuid.as_str());
            let to = self.slot_of(synapse.to_uuid.as_str());
            if self.kind[from] == Kind::Output {
                continue;
            }
            self.out[from].push((to, synapse.weight.abs()));
        }
    }

    /// `Σ abs(weight) × value(child)` over the edges of `slot` that `keep` takes.
    ///
    /// A zero weight contributes nothing even when the child's importance
    /// overflowed: a muted edge passes nothing downstream, and `0 × ∞` must not
    /// turn the muted neuron this ranking exists to find into an undefined one.
    fn contribution<K, V>(&self, slot: usize, keep: K, value: V) -> f64
    where
        K: Fn(usize) -> bool,
        V: Fn(usize) -> f64,
    {
        self.out[slot]
            .iter()
            .filter(|(to, _)| keep(*to))
            .map(|&(to, weight)| {
                if weight == 0.0 {
                    0.0
                } else {
                    weight * value(to)
                }
            })
            .sum()
    }

    /// Resolve every slot's importance and freeze the index.
    fn solve(self) -> SensitivityIndex<'a> {
        let components = strongly_connected_components(&self.out);
        let mut component_of = vec![usize::MAX; self.out.len()];
        for (id, component) in components.iter().enumerate() {
            for &slot in component {
                component_of[slot] = id;
            }
        }
        let mut importance = vec![0.0f64; self.out.len()];
        // Tarjan emits a component only once everything it reaches has been
        // emitted, so each component below is resolved from resolved values.
        for (id, component) in components.iter().enumerate() {
            let cyclic = component.len() > 1
                || self.out[component[0]]
                    .iter()
                    .any(|&(to, _)| to == component[0]);
            let mut largest_exit = 0.0f64;
            for &slot in component {
                let value = if self.kind[slot] == Kind::Output {
                    1.0
                } else {
                    self.contribution(slot, |to| component_of[to] != id, |to| importance[to])
                };
                let value = finite(value);
                importance[slot] = value;
                if value > largest_exit {
                    largest_exit = value;
                }
            }
            if cyclic {
                // No first-order fixpoint to read, so the loop is relaxed once
                // against the largest importance it sends out of itself: every
                // member keeps at least that exit — a recurrent neuron is never
                // mistaken for dead wood — and a member that amplifies into the
                // exit is credited for that hop rather than rounded down to it.
                let relaxed: Vec<f64> = component
                    .iter()
                    .map(|&slot| {
                        let inside =
                            self.contribution(slot, |to| component_of[to] == id, |_| largest_exit);
                        finite(importance[slot] + inside).max(largest_exit)
                    })
                    .collect();
                for (&slot, value) in component.iter().zip(relaxed) {
                    importance[slot] = value;
                }
            }
        }
        SensitivityIndex {
            slots: self.slots,
            uuid: self.uuid,
            kind: self.kind,
            importance,
        }
    }
}

/// A weight product that overflowed or went undefined ranks last, not first.
fn finite(value: f64) -> f64 {
    if value.is_finite() {
        value
    } else {
        f64::INFINITY
    }
}

/// Strongly connected components of `out`, in reverse topological order.
///
/// Tarjan's algorithm, iterative so a creature with a chain thousands of
/// neurons deep cannot overflow the stack. A component is emitted only after
/// every component it can reach, which is exactly the order the backward
/// propagation needs.
fn strongly_connected_components(out: &[Vec<(usize, f64)>]) -> Vec<Vec<usize>> {
    let n = out.len();
    let mut index = vec![usize::MAX; n];
    let mut low = vec![0usize; n];
    let mut on_stack = vec![false; n];
    let mut stack: Vec<usize> = Vec::new();
    let mut call: Vec<(usize, usize)> = Vec::new();
    let mut next = 0usize;
    let mut components: Vec<Vec<usize>> = Vec::new();
    for root in 0..n {
        if index[root] != usize::MAX {
            continue;
        }
        index[root] = next;
        low[root] = next;
        next += 1;
        stack.push(root);
        on_stack[root] = true;
        call.push((root, 0));
        while let Some(&(node, cursor)) = call.last() {
            if cursor < out[node].len() {
                call.last_mut().expect("frame just read").1 += 1;
                let child = out[node][cursor].0;
                if index[child] == usize::MAX {
                    index[child] = next;
                    low[child] = next;
                    next += 1;
                    stack.push(child);
                    on_stack[child] = true;
                    call.push((child, 0));
                } else if on_stack[child] {
                    low[node] = low[node].min(index[child]);
                }
                continue;
            }
            call.pop();
            if let Some(&(parent, _)) = call.last() {
                low[parent] = low[parent].min(low[node]);
            }
            if low[node] == index[node] {
                let mut component = Vec::new();
                while let Some(member) = stack.pop() {
                    on_stack[member] = false;
                    component.push(member);
                    if member == node {
                        break;
                    }
                }
                components.push(component);
            }
        }
    }
    components
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixtures::{creature, neuron, synapse};
    use neat_core::NeuronExport;

    fn hidden(uuid: &str) -> NeuronExport {
        neuron("hidden", uuid, 0.0, Some("TANH"))
    }

    fn output(uuid: &str) -> NeuronExport {
        neuron("output", uuid, 0.0, Some("IDENTITY"))
    }

    /// `input-0 → a →(2) b →(3) output-0`.
    fn chain() -> CreatureExport {
        creature(
            1,
            1,
            vec![hidden("a"), hidden("b"), output("output-0")],
            vec![
                synapse("input-0", "a", 1.0),
                synapse("a", "b", 2.0),
                synapse("b", "output-0", 3.0),
            ],
        )
    }

    #[test]
    fn a_chain_multiplies_the_weights_between_the_neuron_and_the_output() {
        let creature = chain();
        let index = SensitivityIndex::new(&creature);
        assert_eq!(index.importance("output-0"), Some(1.0));
        assert_eq!(index.importance("b"), Some(3.0));
        assert_eq!(index.importance("a"), Some(6.0), "2 × 3");
    }

    #[test]
    fn a_branch_sums_every_downstream_path() {
        // a → output-0 (0.5), and a → b → output-1 (2 × 1).
        let fixture = creature(
            1,
            2,
            vec![
                hidden("a"),
                hidden("b"),
                output("output-0"),
                output("output-1"),
            ],
            vec![
                synapse("input-0", "a", 1.0),
                synapse("a", "output-0", 0.5),
                synapse("a", "b", 2.0),
                synapse("b", "output-1", 1.0),
            ],
        );
        let index = SensitivityIndex::new(&fixture);
        assert_eq!(index.importance("b"), Some(1.0));
        assert_eq!(index.importance("a"), Some(2.5), "0.5 + 2 × 1");
    }

    #[test]
    fn a_zero_weight_downstream_path_leaves_the_neuron_invisible_to_the_outputs() {
        let fixture = creature(
            1,
            1,
            vec![hidden("muted"), hidden("tail"), output("output-0")],
            vec![
                synapse("input-0", "muted", 1.0),
                synapse("muted", "tail", 0.0),
                synapse("tail", "output-0", 4.0),
            ],
        );
        let index = SensitivityIndex::new(&fixture);
        assert_eq!(index.importance("tail"), Some(4.0));
        assert_eq!(
            index.importance("muted"),
            Some(0.0),
            "a zero weight passes nothing downstream, however loud the neuron is"
        );
    }

    #[test]
    fn a_neuron_that_reaches_no_output_has_no_importance() {
        let mut fixture = chain();
        fixture.neurons.push(hidden("dead_end"));
        fixture.synapses.push(synapse("input-0", "dead_end", 5.0));
        crate::fixtures::sort_synapses_canonically(&mut fixture);
        let index = SensitivityIndex::new(&fixture);
        assert_eq!(index.importance("dead_end"), Some(0.0));
    }

    #[test]
    fn a_cycle_keeps_the_exit_it_sends_out_and_credits_one_hop_into_it() {
        // r1 ⇄ r2, and r2 → output-0 with weight 7. The loop gain is 2 × 0.5,
        // so the first-order series diverges and no fixpoint can be read: the
        // component is relaxed once against its largest exit instead.
        let fixture = creature(
            1,
            1,
            vec![hidden("r1"), hidden("r2"), output("output-0")],
            vec![
                synapse("input-0", "r1", 1.0),
                synapse("r1", "r2", 2.0),
                synapse("r2", "r1", 0.5),
                synapse("r2", "output-0", 7.0),
            ],
        );
        let index = SensitivityIndex::new(&fixture);
        assert_eq!(index.importance("r2"), Some(10.5), "7 + 0.5 × 7");
        assert_eq!(
            index.importance("r1"),
            Some(14.0),
            "2 × 7: a member that amplifies into the exit is not rounded down to it"
        );
    }

    #[test]
    fn a_loop_that_reaches_no_output_is_still_dead_wood() {
        // r1 ⇄ r2 with nothing downstream: conservative handling must not
        // invent importance for a loop the outputs cannot see.
        let fixture = creature(
            1,
            1,
            vec![
                hidden("r1"),
                hidden("r2"),
                hidden("live"),
                output("output-0"),
            ],
            vec![
                synapse("input-0", "r1", 1.0),
                synapse("r1", "r2", 2.0),
                synapse("r2", "r1", 0.5),
                synapse("input-0", "live", 1.0),
                synapse("live", "output-0", 1.0),
            ],
        );
        let index = SensitivityIndex::new(&fixture);
        assert_eq!(index.importance("r1"), Some(0.0));
        assert_eq!(index.importance("r2"), Some(0.0));
        assert_eq!(index.importance("live"), Some(1.0));
    }

    #[test]
    fn a_self_loop_is_relaxed_as_a_cycle_rather_than_summed_into_itself() {
        let fixture = creature(
            1,
            1,
            vec![hidden("s"), output("output-0")],
            vec![
                synapse("input-0", "s", 1.0),
                synapse("s", "s", 3.0),
                synapse("s", "output-0", 2.0),
            ],
        );
        let index = SensitivityIndex::new(&fixture);
        // Exit 2, plus one relaxed hop of the self edge: 2 + 3 × 2. The
        // recursion terminates instead of summing the neuron into itself.
        assert_eq!(index.importance("s"), Some(8.0));
    }

    #[test]
    fn a_muted_edge_behind_an_overflowed_path_is_still_dead_wood() {
        // The tail overflows to infinity; `muted` reaches it through a zero
        // weight, so it passes nothing on. Reading that as `0 × ∞ = NaN` would
        // rank the muted neuron last — the exact opposite of the truth.
        let mut neurons = vec![output("output-0"), hidden("muted")];
        let mut synapses = vec![synapse("input-0", "n0", 1.0)];
        const DEPTH: usize = 400;
        for step in 0..DEPTH {
            neurons.push(hidden(&format!("n{step}")));
            let target = if step + 1 == DEPTH {
                "output-0".to_string()
            } else {
                format!("n{}", step + 1)
            };
            synapses.push(synapse(&format!("n{step}"), &target, 1e6));
        }
        synapses.push(synapse("input-0", "muted", 1.0));
        synapses.push(synapse("muted", "n0", 0.0));
        let fixture = creature(1, 1, neurons, synapses);
        let index = SensitivityIndex::new(&fixture);
        assert_eq!(index.importance("n0"), Some(f64::INFINITY));
        assert_eq!(index.importance("muted"), Some(0.0));
    }

    #[test]
    fn the_index_is_deterministic_and_independent_of_listing_order() {
        let fixture = chain();
        let first = SensitivityIndex::new(&fixture).hidden_importance();
        for _ in 0..8 {
            assert_eq!(SensitivityIndex::new(&fixture).hidden_importance(), first);
        }
        let mut reversed = chain();
        reversed.neurons.reverse();
        reversed.synapses.reverse();
        assert_eq!(SensitivityIndex::new(&reversed).hidden_importance(), first);
    }

    #[test]
    fn indexing_never_writes_to_the_creature() {
        let fixture = chain();
        let untouched = fixture.clone();
        let index = SensitivityIndex::new(&fixture);
        index.hidden_importance();
        assert_eq!(fixture, untouched);
    }

    #[test]
    fn every_hidden_neuron_is_covered_and_unknown_endpoints_are_not() {
        let fixture = chain();
        let index = SensitivityIndex::new(&fixture);
        let hidden = index.hidden_importance();
        assert_eq!(hidden.len(), 2, "{hidden:?}");
        assert!(hidden.contains_key("a") && hidden.contains_key("b"));
        assert_eq!(index.importance("nonesuch"), None);
        assert_eq!(index.importance("input-0"), Some(6.0), "inputs are walked");
    }

    #[test]
    fn a_creature_without_hidden_neurons_has_nothing_to_rank() {
        let bare = crate::fixtures::identity_creature(1, 1);
        assert!(SensitivityIndex::new(&bare).hidden_importance().is_empty());
    }

    #[test]
    fn a_chain_thousands_of_neurons_deep_does_not_overflow_the_stack() {
        const DEPTH: usize = 20_000;
        let mut neurons = Vec::with_capacity(DEPTH + 1);
        let mut synapses = Vec::with_capacity(DEPTH + 1);
        for step in 0..DEPTH {
            neurons.push(hidden(&format!("n{step}")));
            let source = if step == 0 {
                "input-0".to_string()
            } else {
                format!("n{}", step - 1)
            };
            synapses.push(synapse(&source, &format!("n{step}"), 1.0));
        }
        neurons.push(output("output-0"));
        synapses.push(synapse(&format!("n{}", DEPTH - 1), "output-0", 1.0));
        let fixture = creature(1, 1, neurons, synapses);
        let index = SensitivityIndex::new(&fixture);
        assert_eq!(
            index.importance("n0"),
            Some(1.0),
            "unit weights all the way"
        );
    }

    #[test]
    fn an_overflowing_weight_product_ranks_last_rather_than_first() {
        let mut neurons = vec![output("output-0")];
        let mut synapses = vec![synapse("input-0", "n0", 1.0)];
        const DEPTH: usize = 400;
        for step in 0..DEPTH {
            neurons.push(hidden(&format!("n{step}")));
            let target = if step + 1 == DEPTH {
                "output-0".to_string()
            } else {
                format!("n{}", step + 1)
            };
            synapses.push(synapse(&format!("n{step}"), &target, 1e6));
        }
        let fixture = creature(1, 1, neurons, synapses);
        let index = SensitivityIndex::new(&fixture);
        assert_eq!(
            index.importance("n0"),
            Some(f64::INFINITY),
            "an unrepresentable product must never read as dead wood"
        );
    }
}
