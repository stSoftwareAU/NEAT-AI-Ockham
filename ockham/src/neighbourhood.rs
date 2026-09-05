//! Bounded structural neighbourhood proposals for group cuts (Issue #108).
//!
//! Some dead wood is a branch, not a twig 🪒. A chain, a leaf branch or a small
//! single-output tributary can be collectively redundant while each neuron in it
//! is, on its own, a poor approximation: cut the middle of a chain and the mean
//! substitution has to carry the whole chain's behaviour through one bias, and
//! the sampled screen quite reasonably says no. Cut the chain as a unit and
//! there is nothing left to approximate.
//!
//! This module proposes those groups, from topology and the ranking signals
//! Ockham already builds, and never from a brute-force search over neuron
//! subsets. Two shapes are generated:
//!
//! - **Chains** — `a → b → c`, where each link is the only way out of `a` and
//!   the only way into `b`. Nothing else reads the intermediate values.
//! - **Branches** — a single-output tributary: a neuron with one outgoing
//!   synapse, grown upstream through predecessors that feed nothing but the
//!   group. The whole tributary reaches the rest of the creature through one
//!   edge.
//!
//! Both are deliberately **bounded** ([`NeighbourhoodConfig::max_size`], 2–8
//! neurons) and **capped** ([`NeighbourhoodConfig::max_proposals`]), because the
//! number of connected subgraphs grows combinatorially and a razor that spends
//! its budget enumerating them prunes nothing.
//!
//! Generation is **deterministic**: the walk visits neurons in the creature's
//! own listing order, and proposals are ranked by
//! `max(mean_abs × downstream importance) ÷ estimated growth units saved` with
//! ties broken on the member UUIDs. The same creature and statistics produce the
//! same proposals in the same order on every host and every run.
//!
//! Ranking is all this module does. A proposal is screened by the ordinary
//! sampled pipeline and can only be accepted by the full-corpus scorer, exactly
//! like a single-neuron candidate — the proposal may be clever, the scorer is
//! still the judge.

use std::collections::{BTreeSet, HashMap, HashSet};

use neat_core::CreatureExport;

use crate::cascade::{CascadeEstimate, CascadeIndex};
use crate::sensitivity::SensitivityIndex;
use crate::stats::ActivationStats;
use crate::sweep::{CandidateKind, SweepCandidate};

/// Smallest group worth proposing — one neuron is an ordinary candidate.
pub const MIN_NEIGHBOURHOOD_SIZE: usize = 2;
/// Largest group this generator will build, whatever is configured.
///
/// A hard ceiling rather than advice: the whole point of a bounded
/// neighbourhood is that the razor cannot be talked into enumerating a large
/// subgraph, and a mis-typed flag must not be the thing that decides it.
pub const MAX_NEIGHBOURHOOD_SIZE: usize = 8;
/// Default group size — the 2–4 neuron chains and branches of the first
/// experiment, at its upper end.
pub const DEFAULT_NEIGHBOURHOOD_SIZE: usize = 4;
/// Default group proposals offered per sweep batch.
pub const DEFAULT_NEIGHBOURHOOD_PROPOSALS: usize = 8;

/// What structural shape a proposal came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NeighbourhoodKind {
    /// A linear chain of hidden neurons, each the only source of the next.
    Chain,
    /// A single-output tributary grown upstream from a one-edge exit.
    Branch,
    /// A small connected subgraph of neurons no louder than the one it grew
    /// from — the shape that is **not** reducible to a single cut.
    ///
    /// A chain or a tributary leaves the creature through one edge, so cutting
    /// its exit alone already strands the rest and the group is only a
    /// different way of writing the same removal. A cluster may leave through
    /// several, and then no single cut removes what it removes.
    Cluster,
}

impl NeighbourhoodKind {
    /// Kebab-case name used in logs and journals.
    pub fn name(self) -> &'static str {
        match self {
            Self::Chain => "chain",
            Self::Branch => "branch",
            Self::Cluster => "cluster",
        }
    }
}

/// How large and how many neighbourhoods a sweep may propose.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NeighbourhoodConfig {
    /// Hidden neurons in one group, clamped to
    /// `MIN_NEIGHBOURHOOD_SIZE..=MAX_NEIGHBOURHOOD_SIZE`.
    pub max_size: usize,
    /// Proposals returned per call, best-ranked first.
    pub max_proposals: usize,
}

impl Default for NeighbourhoodConfig {
    fn default() -> Self {
        Self {
            max_size: DEFAULT_NEIGHBOURHOOD_SIZE,
            max_proposals: DEFAULT_NEIGHBOURHOOD_PROPOSALS,
        }
    }
}

impl NeighbourhoodConfig {
    /// [`Self::max_size`] clamped to the range the generator will honour.
    ///
    /// The CLI refuses an out-of-range `--group-max-size` by name rather than
    /// clamping it ([`crate::config::OckhamConfig::validate`]), so an operator
    /// is never told a size was honoured when it was not. This clamp is the
    /// library's own floor for a caller that never passed through that check —
    /// a bound the generator can rely on, not a second policy for the flag.
    pub fn effective_size(&self) -> usize {
        self.max_size
            .clamp(MIN_NEIGHBOURHOOD_SIZE, MAX_NEIGHBOURHOOD_SIZE)
    }
}

/// One bounded group proposal.
#[derive(Debug, Clone, PartialEq)]
pub struct Neighbourhood {
    /// Shape the proposal came from.
    pub kind: NeighbourhoodKind,
    /// Hidden neurons to cut, upstream-first.
    pub members: Vec<String>,
    /// Largest `mean_abs × downstream importance` in the group.
    ///
    /// The **loudest** member, not the mean: a group is only dead wood when
    /// every neuron in it is, and averaging would let one live neuron hide
    /// behind three quiet ones.
    pub effect: f64,
    /// What the cascade dry run says cutting the whole group would remove.
    pub estimate: CascadeEstimate,
    /// `effect ÷ estimated growth units saved`; smallest is tried first.
    pub rank: f64,
}

/// Rank bounded chain and branch neighbourhoods of `creature` (Issue #108).
///
/// Returns at most [`NeighbourhoodConfig::max_proposals`] groups, best first,
/// deterministically. A group is offered only when every member carries a
/// measured mean — without one the razor could not build the candidate — and
/// only when the cascade dry run says the cut is buildable and removes
/// structure. Nothing here decides that a group is good; the sampled screen and
/// the full-corpus scorer do that.
pub fn propose_neighbourhoods(
    creature: &CreatureExport,
    stats: &ActivationStats,
    cfg: NeighbourhoodConfig,
) -> Vec<Neighbourhood> {
    if cfg.max_proposals == 0 {
        return Vec::new();
    }
    let topology = Topology::new(creature);
    let max_size = cfg.effective_size();

    let sensitivity = SensitivityIndex::new(creature);
    let cascade = CascadeIndex::new(creature);
    // One estimated effect per hidden neuron, measured once: the cluster walk
    // ranks with it and every proposal is scored by it.
    let effects: HashMap<&str, f64> = topology
        .hidden
        .iter()
        .filter_map(|uuid| effect_of(uuid, stats, &sensitivity).map(|e| (*uuid, e)))
        .collect();

    let mut groups: Vec<(NeighbourhoodKind, Vec<&str>)> = Vec::new();
    groups.extend(
        topology
            .chains(max_size)
            .into_iter()
            .map(|m| (NeighbourhoodKind::Chain, m)),
    );
    groups.extend(
        topology
            .branches(max_size)
            .into_iter()
            .map(|m| (NeighbourhoodKind::Branch, m)),
    );
    groups.extend(
        topology
            .clusters(max_size, &effects)
            .into_iter()
            .map(|m| (NeighbourhoodKind::Cluster, m)),
    );

    let mut seen: HashSet<BTreeSet<&str>> = HashSet::new();
    let mut out: Vec<Neighbourhood> = Vec::new();
    for (kind, members) in groups {
        // The same neuron set can be both a chain and a tributary; the shape it
        // was found by does not change what cutting it removes, so it is
        // proposed once, under the shape that found it first.
        if !seen.insert(members.iter().copied().collect()) {
            continue;
        }
        // Every member needs a measured mean: the group ablation substitutes
        // one per member, and a group missing one could never be built.
        let Some(effect) = members
            .iter()
            .map(|uuid| effects.get(uuid).copied())
            .try_fold(0.0f64, |worst, e| e.map(|e| worst.max(e)))
        else {
            continue;
        };
        let estimate = cascade.estimate(&members);
        if estimate.blocked || estimate.growth_units <= 0.0 {
            continue;
        }
        out.push(Neighbourhood {
            kind,
            members: members.iter().map(|m| (*m).to_string()).collect(),
            effect,
            rank: effect / estimate.growth_units,
            estimate,
        });
    }
    drop_subsumed(&mut out);
    // Total order: the ranking key, then the members themselves, so two groups
    // that estimate identically still come back in the same order everywhere.
    out.sort_by(|a, b| {
        a.rank
            .partial_cmp(&b.rank)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.members.cmp(&b.members))
    });
    out.truncate(cfg.max_proposals);
    out
}

/// A ranked proposal the transform refused, and why.
///
/// Reported rather than dropped: a generator whose proposals are all refused
/// looks exactly like a creature with no neighbourhoods in it, and those are
/// very different facts about a run. The membership comes back with the reason
/// so a caller can remember what it has already been refused — the generator is
/// deterministic, and would otherwise offer the same unbuildable group on every
/// batch until the deadline.
#[derive(Debug, Clone, PartialEq)]
pub struct RefusedGroup {
    /// The neighbourhood that could not be built.
    pub members: Vec<String>,
    /// What the transform said, naming the structure that stopped it.
    pub reason: String,
}

/// One built group candidate and the structure its cleanup stranded.
#[derive(Debug, Clone, PartialEq)]
pub struct BuiltGroup {
    /// The candidate the screen and the scorer will see.
    pub candidate: SweepCandidate,
    /// Neurons the exact cleanup removed on top of the requested group.
    ///
    /// Recorded apart from [`SweepCandidate::members`] because they answer
    /// different questions: the members are what the razor chose to cut, the
    /// cascade is what that choice stranded. A run that reported only a count
    /// could not say which neurons a group actually took with it.
    pub cascade: Vec<String>,
}

/// Group candidates built for one batch, and what stopped the rest.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct GroupBatch {
    /// Buildable, validated group candidates, best-ranked first.
    pub candidates: Vec<BuiltGroup>,
    /// Ranked proposals the transform refused.
    pub blocked: Vec<RefusedGroup>,
}

impl GroupBatch {
    /// Proposals considered — built plus refused.
    pub fn considered(&self) -> usize {
        self.candidates.len() + self.blocked.len()
    }
}

/// Key a membership is remembered by once a run has screened it.
pub fn group_key(members: &[String]) -> String {
    members.join(",")
}

/// Rank neighbourhoods of `incumbent` and build a candidate for each (#108).
///
/// The batch companion of [`propose_neighbourhoods`]: every proposal is put
/// through [`crate::ablation::ablate_group`], so what comes back is already
/// validated by `creature.validate()` and ready for the ordinary sampled
/// screen. Stems are `g000`, `g001`, … so a group candidate never collides with
/// the sweep's own `c000` cohort files.
///
/// Memberships in `tried` — by [`group_key`] — are passed over, and the search
/// reaches that much further down the ranked list to refill the batch. Without
/// it a deterministic generator would re-propose its best groups every batch
/// and the run would pay to screen the same candidates until the deadline.
pub fn group_batch(
    incumbent: &CreatureExport,
    stats: &ActivationStats,
    cfg: NeighbourhoodConfig,
    tried: &HashSet<String>,
) -> GroupBatch {
    let deeper = NeighbourhoodConfig {
        max_proposals: cfg.max_proposals.saturating_add(tried.len()),
        ..cfg
    };
    let mut batch = GroupBatch::default();
    for group in propose_neighbourhoods(incumbent, stats, deeper) {
        if batch.candidates.len() >= cfg.max_proposals {
            break;
        }
        if tried.contains(&group_key(&group.members)) {
            continue;
        }
        match crate::sweep::propose_group(incumbent, stats, &group.members) {
            Ok(built) => {
                let stem = format!("g{:03}", batch.candidates.len());
                batch.candidates.push(BuiltGroup {
                    cascade: built.cascade_uuids(),
                    candidate: SweepCandidate {
                        uuid: group.members[0].clone(),
                        members: group.members,
                        permutation_index: 0,
                        kind: CandidateKind::Group,
                        merged_with: None,
                        stem,
                        creature: built.creature,
                    },
                });
            }
            Err(blocked) => batch.blocked.push(RefusedGroup {
                members: group.members,
                reason: format!("{blocked} [{}]", group.kind.name()),
            }),
        }
    }
    batch
}

/// Drop a proposal a larger one already removes the whole of (Issue #108).
///
/// Every upstream sub-cut of a chain strands the same tail, so it carries the
/// same estimated saving with a strictly smaller numerator and would always
/// outrank the chain it is part of. Left alone, the cap fills with two-neuron
/// prefixes and the whole chain — the shape this experiment exists to test — is
/// never proposed at all. Where two proposals remove exactly the same structure
/// and one's members are a subset of the other's, the superset is the proposal:
/// it names what actually goes, and it substitutes each member's own measured
/// mean rather than folding a constant through the rest.
///
/// Comparison is confined to proposals whose dry runs agree, so this costs a
/// bucket walk rather than a pass over every pair.
fn drop_subsumed(groups: &mut Vec<Neighbourhood>) {
    let sets: Vec<BTreeSet<String>> = groups
        .iter()
        .map(|g| g.members.iter().cloned().collect())
        .collect();
    // Bucket by what the cut is estimated to remove; only proposals that agree
    // there can subsume one another.
    let mut buckets: HashMap<(usize, usize), Vec<usize>> = HashMap::new();
    for (i, group) in groups.iter().enumerate() {
        buckets
            .entry((group.estimate.hidden_neurons(), group.estimate.synapses))
            .or_default()
            .push(i);
    }
    let mut subsumed = vec![false; groups.len()];
    for bucket in buckets.values() {
        for &i in bucket {
            for &j in bucket {
                if i != j
                    && !subsumed[j]
                    && sets[i].len() < sets[j].len()
                    && sets[i].is_subset(&sets[j])
                {
                    subsumed[i] = true;
                    break;
                }
            }
        }
    }
    let mut keep = subsumed.into_iter().map(|s| !s);
    groups.retain(|_| keep.next().unwrap_or(true));
}

/// `mean_abs × downstream importance` for one neuron — how much of what it
/// produces still reaches the outputs.
///
/// `None` when the neuron carries no usable measurement: no activation
/// statistic, a non-finite mean, an importance that overflowed, or an endpoint
/// the sensitivity index does not carry — that last one is scored as infinite
/// importance, which is non-finite and so declines the neuron rather than
/// ranking it on a number nothing measured. A neuron with no measured mean
/// could not be group-ablated at all, so a group holding one is never
/// proposed.
fn effect_of(
    uuid: &str,
    stats: &ActivationStats,
    sensitivity: &SensitivityIndex<'_>,
) -> Option<f64> {
    let neuron = stats.by_uuid(uuid)?;
    if !neuron.mean.is_finite() {
        return None;
    }
    let importance = sensitivity.importance(uuid).unwrap_or(f64::INFINITY);
    let effect = neuron.mean_abs * importance;
    effect.is_finite().then_some(effect)
}

/// Hidden-neuron adjacency of one creature, in its own listing order.
struct Topology<'a> {
    /// Hidden UUIDs, in the creature's listing order.
    hidden: Vec<&'a str>,
    hidden_set: HashSet<&'a str>,
    /// Outgoing targets per endpoint, in synapse order.
    out: HashMap<&'a str, Vec<&'a str>>,
    /// Incoming sources per endpoint, in synapse order.
    incoming: HashMap<&'a str, Vec<&'a str>>,
}

impl<'a> Topology<'a> {
    fn new(creature: &'a CreatureExport) -> Self {
        let hidden: Vec<&str> = creature
            .neurons
            .iter()
            .filter(|n| n.neuron_type == "hidden")
            .map(|n| n.uuid.as_str())
            .collect();
        let mut out: HashMap<&str, Vec<&str>> = HashMap::new();
        let mut incoming: HashMap<&str, Vec<&str>> = HashMap::new();
        for synapse in &creature.synapses {
            out.entry(synapse.from_uuid.as_str())
                .or_default()
                .push(synapse.to_uuid.as_str());
            incoming
                .entry(synapse.to_uuid.as_str())
                .or_default()
                .push(synapse.from_uuid.as_str());
        }
        Self {
            hidden_set: hidden.iter().copied().collect(),
            hidden,
            out,
            incoming,
        }
    }

    fn targets(&self, uuid: &str) -> &[&'a str] {
        self.out.get(uuid).map_or(&[][..], Vec::as_slice)
    }

    fn sources(&self, uuid: &str) -> &[&'a str] {
        self.incoming.get(uuid).map_or(&[][..], Vec::as_slice)
    }

    /// The hidden neuron `uuid` feeds, when that is the only thing either of
    /// them is connected to on that side.
    ///
    /// `a → b` is a chain link when `a` feeds nothing but `b` and `b` is fed by
    /// nothing but `a`: no other neuron reads `a`'s value, and none contributes
    /// to `b`'s, so the pair is only ever used as one path.
    fn chain_link(&self, uuid: &str) -> Option<&'a str> {
        let [next] = self.targets(uuid) else {
            return None;
        };
        if !self.hidden_set.contains(next) {
            return None;
        }
        match self.sources(next) {
            [only] if *only == uuid => Some(next),
            _ => None,
        }
    }

    /// Every maximal chain, truncated to `max_size` from its head.
    fn chains(&self, max_size: usize) -> Vec<Vec<&'a str>> {
        let linked_into: HashSet<&str> = self
            .hidden
            .iter()
            .filter_map(|uuid| self.chain_link(uuid))
            .collect();
        let mut out = Vec::new();
        for head in &self.hidden {
            // Start only at heads, so one chain yields one proposal rather than
            // one per suffix of itself.
            if linked_into.contains(head) {
                continue;
            }
            let mut members = vec![*head];
            let mut walked: HashSet<&str> = HashSet::from([*head]);
            while members.len() < max_size {
                let Some(next) = self.chain_link(members[members.len() - 1]) else {
                    break;
                };
                // A cyclic creature has no chain head; this guards the walk
                // anyway, because a ranking pass may never fail to terminate.
                if !walked.insert(next) {
                    break;
                }
                members.push(next);
            }
            if members.len() >= MIN_NEIGHBOURHOOD_SIZE {
                out.push(members);
            }
        }
        out
    }

    /// Small connected subgraphs grown from each hidden neuron (Issue #108).
    ///
    /// From every hidden neuron with a measured effect, the walk admits hidden
    /// neighbours — either direction — that are **no louder than the neuron it
    /// started from**, in the creature's listing order, until the group is
    /// full. The seed is therefore the loudest member and the group's ranking
    /// key is the seed's own effect.
    ///
    /// That is what keeps this "low-importance neighbourhoods" without
    /// inventing a threshold for "quiet": a group seeded on a loud neuron is
    /// ranked on that neuron's volume and sorts behind every genuinely quiet
    /// group, so the cap never spends a slot on it.
    ///
    /// This is the shape a single cut cannot stand in for: a chain or a
    /// tributary reaches the rest of the creature through one edge, so cutting
    /// that edge's neuron already strands the rest, but a cluster may leave
    /// through several. It is still bounded, still deterministic and still
    /// derived from topology and sensitivity — never a search over arbitrary
    /// subsets.
    fn clusters(&self, max_size: usize, effects: &HashMap<&str, f64>) -> Vec<Vec<&'a str>> {
        let mut out = Vec::new();
        for seed in &self.hidden {
            let Some(&loudest) = effects.get(seed) else {
                continue;
            };
            let mut members: Vec<&str> = vec![*seed];
            let mut set: HashSet<&str> = HashSet::from([*seed]);
            let mut frontier = 0usize;
            while frontier < members.len() && members.len() < max_size {
                let current = members[frontier];
                frontier += 1;
                let neighbours = self
                    .targets(current)
                    .iter()
                    .chain(self.sources(current).iter());
                for neighbour in neighbours {
                    if members.len() >= max_size {
                        break;
                    }
                    if !self.hidden_set.contains(neighbour) || set.contains(neighbour) {
                        continue;
                    }
                    // Quieter than the seed, or the group would be ranked on a
                    // neuron the outputs still depend on.
                    if effects.get(neighbour).is_none_or(|e| *e > loudest) {
                        continue;
                    }
                    set.insert(neighbour);
                    members.push(neighbour);
                }
            }
            if members.len() >= MIN_NEIGHBOURHOOD_SIZE {
                out.push(members);
            }
        }
        out
    }

    /// Single-output tributaries: a one-edge exit, grown upstream.
    ///
    /// From each hidden neuron with exactly one outgoing synapse, predecessors
    /// that feed nothing outside the group are pulled in — in listing order,
    /// breadth-first from the exit — until the group is full. The result
    /// reaches the rest of the creature through that one edge, so cutting it
    /// touches nothing else.
    fn branches(&self, max_size: usize) -> Vec<Vec<&'a str>> {
        let mut out = Vec::new();
        for root in &self.hidden {
            if self.targets(root).len() != 1 {
                continue;
            }
            let mut members: Vec<&str> = vec![*root];
            let mut set: HashSet<&str> = HashSet::from([*root]);
            let mut frontier = 0usize;
            while frontier < members.len() && members.len() < max_size {
                let current = members[frontier];
                frontier += 1;
                for source in self.sources(current) {
                    if members.len() >= max_size {
                        break;
                    }
                    if !self.hidden_set.contains(source) || set.contains(source) {
                        continue;
                    }
                    // Only a neuron that feeds the group and nothing else: any
                    // other reader would lose an input the group cannot
                    // compensate for.
                    if self.targets(source).iter().any(|t| !set.contains(t)) {
                        continue;
                    }
                    set.insert(source);
                    members.push(source);
                }
            }
            if members.len() >= MIN_NEIGHBOURHOOD_SIZE {
                // Upstream-first, matching the chain order: the deepest
                // tributary neuron is folded before what it fed.
                members.reverse();
                out.push(members);
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ablation::{GroupMember, ablate_group};
    use crate::fixtures::{creature, neuron, synapse, typed_synapse};
    use crate::stats::{NeuronStats, STATS_FORMAT_VERSION, SampleSpec};
    use neat_core::CreatureExport;

    /// `input-0 → c1 → c2 → c3 → output-0`, plus a lone `keep → output-0`.
    fn chain_creature() -> CreatureExport {
        creature(
            1,
            1,
            vec![
                neuron("hidden", "c1", 0.0, Some("TANH")),
                neuron("hidden", "c2", 0.0, Some("TANH")),
                neuron("hidden", "c3", 0.0, Some("TANH")),
                neuron("hidden", "keep", 0.0, Some("TANH")),
                neuron("output", "output-0", 0.0, Some("IDENTITY")),
            ],
            vec![
                synapse("input-0", "c1", 1.0),
                synapse("c1", "c2", 1.0),
                synapse("c2", "c3", 1.0),
                synapse("c3", "output-0", 0.01),
                synapse("input-0", "keep", 1.0),
                synapse("keep", "output-0", 2.0),
            ],
        )
    }

    /// `b1` and `b2` both feed `b3`, which is the branch's only exit.
    fn branch_creature() -> CreatureExport {
        creature(
            1,
            1,
            vec![
                neuron("hidden", "b1", 0.0, Some("TANH")),
                neuron("hidden", "b2", 0.0, Some("TANH")),
                neuron("hidden", "b3", 0.0, Some("TANH")),
                neuron("hidden", "keep", 0.0, Some("TANH")),
                neuron("output", "output-0", 0.0, Some("IDENTITY")),
            ],
            vec![
                synapse("input-0", "b1", 1.0),
                synapse("input-0", "b2", 1.0),
                synapse("b1", "b3", 1.0),
                synapse("b2", "b3", 1.0),
                synapse("b3", "output-0", 0.01),
                synapse("input-0", "keep", 1.0),
                synapse("keep", "output-0", 2.0),
            ],
        )
    }

    /// Statistics carrying `mean_abs` for `pick`ed hidden neurons.
    fn stats_with(creature: &CreatureExport, pick: impl Fn(&str) -> f64) -> ActivationStats {
        ActivationStats {
            format_version: STATS_FORMAT_VERSION,
            creature_checksum: "t".into(),
            corpus_identity: "c".into(),
            record_count: 1,
            corpus_record_count: 1,
            sample: SampleSpec::full(),
            stopped_early: false,
            scan_ms: 0,
            from_cache: false,
            probes: Vec::new(),
            neurons: creature
                .neurons
                .iter()
                .enumerate()
                .filter(|(_, n)| n.neuron_type == "hidden")
                .map(|(i, n)| {
                    let mean_abs = pick(&n.uuid);
                    NeuronStats {
                        uuid: n.uuid.clone(),
                        neuron_index: i,
                        count: 10,
                        mean: mean_abs,
                        variance: 0.0,
                        std_dev: 0.0,
                        mean_abs,
                        min: 0.0,
                        max: mean_abs,
                    }
                })
                .collect(),
        }
    }

    /// Uniform statistics, so ranking differences come from topology alone.
    fn stats_for(creature: &CreatureExport, mean_abs: f64) -> ActivationStats {
        stats_with(creature, |_| mean_abs)
    }

    fn names(groups: &[Neighbourhood]) -> Vec<Vec<String>> {
        groups.iter().map(|g| g.members.clone()).collect()
    }

    #[test]
    fn a_linear_chain_is_proposed_head_to_tail_as_one_group() {
        let creature = chain_creature();
        let stats = stats_for(&creature, 0.1);
        let groups = propose_neighbourhoods(&creature, &stats, NeighbourhoodConfig::default());
        let chain = groups
            .iter()
            .find(|g| g.kind == NeighbourhoodKind::Chain)
            .unwrap_or_else(|| panic!("no chain proposed: {:?}", names(&groups)));
        assert_eq!(chain.members, vec!["c1", "c2", "c3"]);
        assert_eq!(chain.estimate.hidden_neurons(), 3);
        assert!(chain.estimate.growth_units > 0.0);
        assert!(
            groups.iter().all(|g| !g.members.contains(&"keep".into())),
            "`keep` has no chain or tributary: {:?}",
            names(&groups)
        );
    }

    #[test]
    fn a_single_output_tributary_is_proposed_as_a_branch() {
        let creature = branch_creature();
        let stats = stats_for(&creature, 0.1);
        let groups = propose_neighbourhoods(&creature, &stats, NeighbourhoodConfig::default());
        let branch = groups
            .iter()
            .find(|g| g.members.contains(&"b3".into()))
            .unwrap_or_else(|| panic!("no branch proposed: {:?}", names(&groups)));
        assert_eq!(branch.kind, NeighbourhoodKind::Branch);
        assert_eq!(branch.members, vec!["b2", "b1", "b3"]);
        assert_eq!(branch.estimate.hidden_neurons(), 3);
        assert!(
            groups.iter().all(|g| !g.members.contains(&"keep".into())),
            "`keep` feeds the output directly: {:?}",
            names(&groups)
        );
    }

    #[test]
    fn a_group_batch_builds_a_validated_candidate_per_proposal() {
        let creature = chain_creature();
        let stats = stats_for(&creature, 0.1);
        let batch = group_batch(
            &creature,
            &stats,
            NeighbourhoodConfig::default(),
            &HashSet::new(),
        );
        assert!(!batch.candidates.is_empty());
        assert!(batch.blocked.is_empty(), "{:?}", batch.blocked);
        assert_eq!(batch.considered(), batch.candidates.len());
        let built = &batch.candidates[0];
        let first = &built.candidate;
        assert_eq!(first.kind, CandidateKind::Group);
        assert!(first.is_group());
        assert_eq!(first.uuid, first.members[0]);
        assert_eq!(first.cuts(), first.members);
        assert_eq!(first.stem, "g000");
        crate::incumbent::validate_creature(&first.creature).unwrap();
        for member in &first.members {
            assert!(
                first.creature.neurons.iter().all(|n| &n.uuid != member),
                "{member} must be gone from the candidate"
            );
        }
        // The cleanup cascade is recorded apart from the requested cuts, by
        // name, and never repeats one of them.
        assert!(
            built.cascade.iter().all(|u| !first.members.contains(u)),
            "{:?} vs {:?}",
            built.cascade,
            first.members
        );
        // Stems must not collide with the sweep's own `c000` cohort files.
        assert!(
            batch
                .candidates
                .iter()
                .all(|c| c.candidate.stem.starts_with('g'))
        );
    }

    #[test]
    fn a_batch_without_statistics_proposes_nothing_to_build() {
        let creature = chain_creature();
        let batch = group_batch(
            &creature,
            &ActivationStats::empty(),
            NeighbourhoodConfig::default(),
            &HashSet::new(),
        );
        assert_eq!(batch, GroupBatch::default());
        assert_eq!(batch.considered(), 0);
    }

    #[test]
    fn a_proposal_is_buildable_by_the_group_ablation() {
        for creature in [chain_creature(), branch_creature()] {
            let stats = stats_for(&creature, 0.1);
            let groups = propose_neighbourhoods(&creature, &stats, NeighbourhoodConfig::default());
            assert!(!groups.is_empty());
            for group in &groups {
                let members: Vec<GroupMember> = group
                    .members
                    .iter()
                    .map(|uuid| GroupMember {
                        uuid: uuid.clone(),
                        mean: stats.by_uuid(uuid).unwrap().mean,
                    })
                    .collect();
                let built = ablate_group(&creature, &members)
                    .unwrap_or_else(|e| panic!("{:?} must build: {e}", group.members));
                // What the dry run predicted is what the transform removed.
                assert_eq!(
                    built.before.hidden_neurons - built.after.hidden_neurons,
                    group.estimate.hidden_neurons(),
                    "{:?}",
                    group.members
                );
                assert_eq!(
                    built.before.synapses - built.after.synapses,
                    group.estimate.synapses,
                    "{:?}",
                    group.members
                );
            }
        }
    }

    /// `a` and `b` are quiet and feed each other, but leave through two
    /// different survivors, each of which keeps its own input.
    fn web_creature() -> CreatureExport {
        creature(
            1,
            1,
            vec![
                neuron("hidden", "a", 0.0, Some("TANH")),
                neuron("hidden", "b", 0.0, Some("TANH")),
                neuron("hidden", "x", 0.0, Some("TANH")),
                neuron("hidden", "y", 0.0, Some("TANH")),
                neuron("output", "output-0", 0.0, Some("IDENTITY")),
            ],
            vec![
                synapse("input-0", "a", 1.0),
                synapse("input-0", "b", 1.0),
                synapse("input-0", "x", 1.0),
                synapse("input-0", "y", 1.0),
                synapse("a", "b", 0.5),
                synapse("a", "x", 0.01),
                synapse("b", "y", 0.01),
                synapse("x", "output-0", 1.0),
                synapse("y", "output-0", 1.0),
            ],
        )
    }

    #[test]
    fn a_two_exit_cluster_is_proposed_where_no_single_cut_removes_it() {
        let creature = web_creature();
        let stats = stats_with(&creature, |uuid| match uuid {
            "a" | "b" => 0.005,
            _ => 0.4,
        });
        let groups = propose_neighbourhoods(&creature, &stats, NeighbourhoodConfig::default());
        let cluster = groups
            .iter()
            .find(|g| g.kind == NeighbourhoodKind::Cluster)
            .unwrap_or_else(|| panic!("no cluster proposed: {:?}", names(&groups)));
        assert_eq!(cluster.members, vec!["a", "b"]);
        assert_eq!(
            groups[0].members,
            vec!["a", "b"],
            "the quiet pair must rank first: {:?}",
            names(&groups)
        );
        // A group seeded on a loud survivor is ranked on that survivor's
        // volume, so it sorts behind the quiet pair rather than spending the
        // cap's first slot.
        let loud = groups
            .iter()
            .position(|g| g.members.iter().any(|m| m == "x" || m == "y"));
        assert!(loud.is_none_or(|i| i > 0), "{:?}", names(&groups));
        // Two exits, so no single cut in the neighbourhood removes what the
        // group does — the case chains and tributaries cannot show.
        let members: Vec<GroupMember> = cluster
            .members
            .iter()
            .map(|uuid| GroupMember {
                uuid: uuid.clone(),
                mean: stats.by_uuid(uuid).unwrap().mean,
            })
            .collect();
        let grouped = ablate_group(&creature, &members).unwrap();
        let group_saving = grouped.before.growth_units - grouped.after.growth_units;
        for uuid in &cluster.members {
            let mean = stats.by_uuid(uuid).unwrap().mean;
            let single = crate::ablation::ablate_mean(&creature, uuid, mean, None).unwrap();
            let single_saving = single.before.growth_units - single.after.growth_units;
            assert!(
                group_saving > single_saving,
                "the group must remove more than cutting {uuid} alone: \
                 {group_saving} vs {single_saving}"
            );
        }
    }

    #[test]
    fn a_proposal_a_larger_one_already_removes_is_not_offered_twice() {
        // Every upstream sub-cut of a chain strands the same tail, so without
        // subsumption the cap fills with two-neuron prefixes and the chain this
        // experiment is about is never proposed.
        let creature = chain_creature();
        let stats = stats_for(&creature, 0.1);
        let groups = propose_neighbourhoods(&creature, &stats, NeighbourhoodConfig::default());
        assert!(
            groups.iter().any(|g| g.members == ["c1", "c2", "c3"]),
            "the whole chain must survive ranking: {:?}",
            names(&groups)
        );
        assert!(
            groups.iter().all(|g| g.members != ["c1", "c2"]),
            "a sub-cut removing exactly the same structure is not a second \
             proposal: {:?}",
            names(&groups)
        );
    }

    #[test]
    fn a_membership_already_tried_is_passed_over_for_the_next_one() {
        let creature = web_creature();
        let stats = stats_with(&creature, |uuid| match uuid {
            "a" | "b" => 0.005,
            _ => 0.4,
        });
        let cfg = NeighbourhoodConfig {
            max_size: 2,
            max_proposals: 1,
        };
        let first = group_batch(&creature, &stats, cfg, &HashSet::new());
        let key = group_key(&first.candidates[0].candidate.members);
        let second = group_batch(&creature, &stats, cfg, &HashSet::from([key.clone()]));
        assert!(
            second
                .candidates
                .iter()
                .all(|c| group_key(&c.candidate.members) != key),
            "a membership already screened must not come back this run"
        );
        assert_ne!(
            second
                .candidates
                .first()
                .map(|c| c.candidate.members.clone()),
            first
                .candidates
                .first()
                .map(|c| c.candidate.members.clone()),
            "the search must reach further down the ranked list"
        );
    }

    #[test]
    fn a_group_key_names_its_membership_in_order() {
        assert_eq!(group_key(&["h_a".into(), "h_b".into()]), "h_a,h_b");
        assert_eq!(group_key(&["h_a".into()]), "h_a");
        assert_eq!(group_key(&[]), "");
        // Order is membership order, so two orders of one set are two keys —
        // the transform folds means in that order and the cut differs.
        assert_ne!(
            group_key(&["h_b".into(), "h_a".into()]),
            group_key(&["h_a".into(), "h_b".into()])
        );
    }

    #[test]
    fn every_shape_has_a_kebab_case_name() {
        assert_eq!(NeighbourhoodKind::Chain.name(), "chain");
        assert_eq!(NeighbourhoodKind::Branch.name(), "branch");
        assert_eq!(NeighbourhoodKind::Cluster.name(), "cluster");
    }

    #[test]
    fn generation_is_deterministic_and_bounded() {
        let creature = chain_creature();
        let stats = stats_for(&creature, 0.1);
        let cfg = NeighbourhoodConfig {
            max_size: 2,
            max_proposals: 1,
        };
        let first = propose_neighbourhoods(&creature, &stats, cfg);
        let second = propose_neighbourhoods(&creature, &stats, cfg);
        assert_eq!(
            first, second,
            "the same inputs must propose the same groups"
        );
        assert_eq!(first.len(), 1, "max_proposals must cap the list");
        assert!(
            first.iter().all(|g| g.members.len() <= 2),
            "max_size must cap each group: {:?}",
            names(&first)
        );
        assert!(
            propose_neighbourhoods(
                &creature,
                &stats,
                NeighbourhoodConfig {
                    max_proposals: 0,
                    ..cfg
                }
            )
            .is_empty()
        );
    }

    #[test]
    fn a_size_outside_the_bounds_is_clamped_rather_than_obeyed() {
        assert_eq!(
            NeighbourhoodConfig {
                max_size: 0,
                max_proposals: 4
            }
            .effective_size(),
            MIN_NEIGHBOURHOOD_SIZE
        );
        assert_eq!(
            NeighbourhoodConfig {
                max_size: 4_000,
                max_proposals: 4
            }
            .effective_size(),
            MAX_NEIGHBOURHOOD_SIZE
        );
    }

    #[test]
    fn the_quietest_group_with_the_largest_saving_ranks_first() {
        // Two chains: `q1 → q2` is quiet, `l1 → l2` is loud. Same topology,
        // same saving — only the measured activation differs.
        let creature = creature(
            1,
            1,
            vec![
                neuron("hidden", "q1", 0.0, Some("TANH")),
                neuron("hidden", "q2", 0.0, Some("TANH")),
                neuron("hidden", "l1", 0.0, Some("TANH")),
                neuron("hidden", "l2", 0.0, Some("TANH")),
                neuron("hidden", "keep", 0.0, Some("TANH")),
                neuron("output", "output-0", 0.0, Some("IDENTITY")),
            ],
            vec![
                synapse("input-0", "q1", 1.0),
                synapse("q1", "q2", 1.0),
                synapse("q2", "output-0", 1.0),
                synapse("input-0", "l1", 1.0),
                synapse("l1", "l2", 1.0),
                synapse("l2", "output-0", 1.0),
                synapse("input-0", "keep", 1.0),
                synapse("keep", "output-0", 1.0),
            ],
        );
        let stats = stats_with(
            &creature,
            |uuid| if uuid.starts_with('l') { 4.0 } else { 0.01 },
        );
        let groups = propose_neighbourhoods(&creature, &stats, NeighbourhoodConfig::default());
        assert_eq!(groups[0].members, vec!["q1", "q2"], "{:?}", names(&groups));
        assert!(groups[0].rank < groups[1].rank);
    }

    #[test]
    fn structure_the_razor_could_never_cut_is_not_proposed() {
        // `t1 → t2` looks like a chain, but the edge into the aggregate `t2`
        // carries a role a bias cannot absorb, so the ablation fails closed and
        // the group must never be offered.
        let creature = creature(
            1,
            1,
            vec![
                neuron("hidden", "t1", 0.0, Some("IDENTITY")),
                neuron("hidden", "t2", 0.0, Some("IF")),
                neuron("hidden", "keep", 0.0, Some("TANH")),
                neuron("output", "output-0", 0.0, Some("IDENTITY")),
            ],
            vec![
                synapse("input-0", "t1", 1.0),
                typed_synapse("t1", "t2", 1.0, "condition"),
                typed_synapse("input-0", "t2", 1.0, "positive"),
                typed_synapse("input-0", "t2", -1.0, "negative"),
                synapse("t2", "output-0", 1.0),
                synapse("input-0", "keep", 1.0),
                synapse("keep", "output-0", 1.0),
            ],
        );
        let stats = stats_for(&creature, 0.1);
        let groups = propose_neighbourhoods(&creature, &stats, NeighbourhoodConfig::default());
        assert!(groups.is_empty(), "{:?}", names(&groups));
    }

    #[test]
    fn a_creature_with_no_measured_statistics_proposes_nothing() {
        let creature = chain_creature();
        let groups = propose_neighbourhoods(
            &creature,
            &ActivationStats::empty(),
            NeighbourhoodConfig::default(),
        );
        assert!(
            groups.is_empty(),
            "a group needs one measured mean per member: {:?}",
            names(&groups)
        );
    }

    #[test]
    fn a_creature_with_no_chain_or_tributary_proposes_nothing() {
        // Two lone hidden neurons, each fed by the input and feeding the
        // output: no chain link, and no single-output tributary of two.
        let creature = creature(
            1,
            1,
            vec![
                neuron("hidden", "a", 0.0, Some("TANH")),
                neuron("hidden", "b", 0.0, Some("TANH")),
                neuron("output", "output-0", 0.0, Some("IDENTITY")),
            ],
            vec![
                synapse("input-0", "a", 1.0),
                synapse("input-0", "b", 1.0),
                synapse("a", "output-0", 1.0),
                synapse("b", "output-0", 1.0),
            ],
        );
        let stats = stats_for(&creature, 0.1);
        let groups = propose_neighbourhoods(&creature, &stats, NeighbourhoodConfig::default());
        assert!(groups.is_empty(), "{:?}", names(&groups));
    }

    #[test]
    fn a_disconnected_hidden_neuron_is_not_grown_into_a_group() {
        // `orphan` feeds nothing: the exact cleanup removes it on its own, and
        // it is not a tributary of anything.
        let mut creature = chain_creature();
        creature
            .neurons
            .insert(0, neuron("hidden", "orphan", 0.0, Some("TANH")));
        let stats = stats_for(&creature, 0.1);
        let groups = propose_neighbourhoods(&creature, &stats, NeighbourhoodConfig::default());
        assert!(
            groups.iter().all(|g| !g.members.contains(&"orphan".into())),
            "{:?}",
            names(&groups)
        );
    }
}
