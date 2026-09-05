//! Per-candidate feature vectors for composite and learned ordering (#107).
//!
//! Every signal the sweep already exposes one at a time — activation variance,
//! mean absolute activation, range, outgoing weight, fan-in, fan-out, direct
//! and cascade growth saving, squash type, topology depth and what earlier
//! epochs learnt about the uuid — is read **once per creature** into one
//! [`CandidateFeatures`] per hidden neuron.
//!
//! Nothing here decides anything. A feature vector orders the sweep, and the
//! sampled screen and the full-corpus scorer still settle every candidate
//! exactly as they do under the random control.
//!
//! The vector is **named**: [`FEATURE_NAMES`] fixes the order, and a model
//! records the names it was fitted on so a stale model is refused rather than
//! silently read against the wrong columns.

use std::collections::{HashMap, HashSet, VecDeque};

use neat_core::CreatureExport;
use serde::{Deserialize, Serialize};

use crate::ablation::growth_units;
use crate::cascade::CascadeIndex;
use crate::learnings::{HistoricalLearning, Learning, Outcome};
use crate::stats::ActivationStats;

/// Feature names in vector order — the schema a model is fitted against.
///
/// Heavy-tailed magnitudes are stored `ln(1 + x)` so one loud neuron cannot
/// dominate a linear model, and the flags are plain `0.0` / `1.0`.
pub const FEATURE_NAMES: &[&str] = &[
    "measured",
    "logVariance",
    "logMeanAbs",
    "logRange",
    "logOutgoingWeight",
    "logDownstreamSensitivity",
    "logFanIn",
    "logFanOut",
    "directGrowthUnits",
    "cascadeGrowthUnits",
    "identity",
    "blocked",
    "depthFraction",
    "priorWins",
    "priorFailures",
];

/// Every signal one hidden neuron carries into the ranking.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct CandidateFeatures {
    /// Whether the activation scan covered this neuron.
    ///
    /// An unmeasured neuron is not a quiet one: its statistics are zero because
    /// nothing was measured, so a ranking that read them as quiet would visit
    /// the neurons it knows least about first. It is carried as a feature, and
    /// the rankings put it last rather than dropping it from the sweep.
    pub measured: bool,
    /// Population variance of the activation.
    pub variance: f64,
    /// Mean absolute activation.
    pub mean_abs: f64,
    /// Activation range (`max - min`).
    pub range: f64,
    /// `Σ abs(weight)` over outgoing synapses.
    pub outgoing_weight: f64,
    /// Incoming synapse count.
    pub fan_in: usize,
    /// Outgoing synapse count.
    pub fan_out: usize,
    /// Growth units of the neuron and the synapses touching it.
    pub direct_growth_units: f64,
    /// Growth units the cascade dry-run predicts the cut would remove (#106).
    pub cascade_growth_units: f64,
    /// Whether the squash is `IDENTITY` — an exact-fold opportunity.
    pub identity: bool,
    /// Whether the cascade dry-run says the transform would refuse the cut.
    pub blocked: bool,
    /// Position from the inputs, `0.0` at the first layer, `1.0` deepest.
    pub depth_fraction: f64,
    /// Verdicts from earlier epochs that spoke well of cutting this uuid.
    pub prior_wins: usize,
    /// Verdicts from earlier epochs that did not.
    pub prior_failures: usize,
}

impl CandidateFeatures {
    /// `mean_abs_activation × Σ abs(outgoing weight)` — the downstream reach.
    pub fn downstream_sensitivity(&self) -> f64 {
        self.mean_abs * self.outgoing_weight
    }

    /// The feature vector in [`FEATURE_NAMES`] order.
    pub fn vector(&self) -> Vec<f64> {
        let flag = |b: bool| if b { 1.0 } else { 0.0 };
        vec![
            flag(self.measured),
            (1.0 + self.variance.max(0.0)).ln(),
            (1.0 + self.mean_abs.max(0.0)).ln(),
            (1.0 + self.range.max(0.0)).ln(),
            (1.0 + self.outgoing_weight.max(0.0)).ln(),
            (1.0 + self.downstream_sensitivity().max(0.0)).ln(),
            (1.0 + self.fan_in as f64).ln(),
            (1.0 + self.fan_out as f64).ln(),
            self.direct_growth_units,
            self.cascade_growth_units,
            flag(self.identity),
            flag(self.blocked),
            self.depth_fraction,
            self.prior_wins as f64,
            self.prior_failures as f64,
        ]
    }

    /// The named vector, for telemetry a later feature set can still read.
    pub fn named(&self) -> Vec<(&'static str, f64)> {
        FEATURE_NAMES.iter().copied().zip(self.vector()).collect()
    }
}

/// What earlier epochs learnt about each uuid — a prior, never a verdict.
///
/// Historical evidence is read exactly as [`crate::learnings`] reads it: a
/// record from an older corpus is evidence about a uuid, not truth about the
/// corpus in hand. It may move a candidate earlier or later in the sweep and
/// nothing else — every candidate still faces the screen and the full scorer.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct PriorEvidence {
    counts: HashMap<String, (usize, usize)>,
}

impl PriorEvidence {
    /// Empty evidence — the state of a run with no learnings cache.
    pub fn new() -> Self {
        Self::default()
    }

    /// Fold one verdict in: a confirmed positive is a win, anything else is not.
    ///
    /// `min_improvement` is the run's own acceptance threshold, so "confirmed
    /// but not applied" (Issue #52) counts as the win it is rather than as the
    /// cohort loss it was filed as.
    pub fn add(&mut self, learning: &Learning, min_improvement: f64) {
        let win = learning.outcome == Outcome::Accepted
            || crate::learnings::confirmed_positive(learning, min_improvement);
        let entry = self.counts.entry(learning.uuid.clone()).or_insert((0, 0));
        if win {
            entry.0 += 1;
        } else {
            entry.1 += 1;
        }
    }

    /// Evidence from every historical epoch the cache carries (Issues #88, #101).
    pub fn from_history(prior: &[HistoricalLearning], min_improvement: f64) -> Self {
        let mut evidence = Self::new();
        for h in prior {
            evidence.add(&h.learning, min_improvement);
        }
        evidence
    }

    /// `(wins, failures)` recorded for `uuid`; `(0, 0)` when nothing is known.
    pub fn counts(&self, uuid: &str) -> (usize, usize) {
        self.counts.get(uuid).copied().unwrap_or((0, 0))
    }

    /// How many uuids the evidence covers.
    pub fn len(&self) -> usize {
        self.counts.len()
    }

    /// Whether nothing is known at all.
    pub fn is_empty(&self) -> bool {
        self.counts.is_empty()
    }
}

/// Feature vector for every hidden neuron on `creature`.
///
/// One pass: the cascade dry-run indexes the creature once (#106), the fan
/// counts and outgoing weights are tallied in one walk of the synapses, and the
/// depth is one breadth-first walk from the inputs. Ranking a whole sweep
/// therefore costs one index, not one per comparison.
pub fn extract(
    creature: &CreatureExport,
    stats: &ActivationStats,
    evidence: &PriorEvidence,
) -> HashMap<String, CandidateFeatures> {
    let index = CascadeIndex::new(creature);
    let cascade = index.hidden_estimates();
    let depth = depth_fractions(creature);
    let mut fan_in: HashMap<&str, usize> = HashMap::new();
    let mut fan_out: HashMap<&str, usize> = HashMap::new();
    let mut outgoing_weight: HashMap<&str, f64> = HashMap::new();
    for synapse in &creature.synapses {
        *fan_out.entry(synapse.from_uuid.as_str()).or_default() += 1;
        *fan_in.entry(synapse.to_uuid.as_str()).or_default() += 1;
        *outgoing_weight
            .entry(synapse.from_uuid.as_str())
            .or_default() += synapse.weight.abs();
    }
    creature
        .neurons
        .iter()
        .filter(|n| n.neuron_type == "hidden")
        .map(|n| {
            let uuid = n.uuid.as_str();
            let measured = stats.by_uuid(uuid);
            let estimate = cascade.get(uuid);
            let (fan_in, fan_out) = (
                fan_in.get(uuid).copied().unwrap_or(0),
                fan_out.get(uuid).copied().unwrap_or(0),
            );
            let (prior_wins, prior_failures) = evidence.counts(uuid);
            let features = CandidateFeatures {
                measured: measured.is_some(),
                variance: measured.map_or(0.0, |s| s.variance),
                mean_abs: measured.map_or(0.0, |s| s.mean_abs),
                range: measured.map_or(0.0, |s| s.max - s.min),
                outgoing_weight: outgoing_weight.get(uuid).copied().unwrap_or(0.0),
                fan_in,
                fan_out,
                direct_growth_units: growth_units(1, fan_in + fan_out),
                cascade_growth_units: estimate.map_or(0.0, |e| e.growth_units),
                identity: crate::sweep::is_identity(creature, uuid),
                blocked: estimate.is_none_or(|e| e.blocked),
                depth_fraction: depth.get(uuid).copied().unwrap_or(1.0),
                prior_wins,
                prior_failures,
            };
            (n.uuid.clone(), features)
        })
        .collect()
}

/// Normalised distance from the inputs for every listed neuron.
///
/// Breadth-first from every endpoint the neuron list does not carry — the
/// implicit `input-N` sources — so the depth is the shortest path from an
/// input. A neuron nothing reaches is deepest (`1.0`) rather than absent: an
/// unreachable neuron is exactly the dead wood the sweep is looking for, and a
/// missing feature would rank it as though it sat on the input layer.
fn depth_fractions(creature: &CreatureExport) -> HashMap<String, f64> {
    let listed: HashSet<&str> = creature.neurons.iter().map(|n| n.uuid.as_str()).collect();
    let mut forward: HashMap<&str, Vec<&str>> = HashMap::new();
    let mut sources: Vec<&str> = Vec::new();
    for synapse in &creature.synapses {
        forward
            .entry(synapse.from_uuid.as_str())
            .or_default()
            .push(synapse.to_uuid.as_str());
        if !listed.contains(synapse.from_uuid.as_str()) {
            sources.push(synapse.from_uuid.as_str());
        }
    }
    let mut depth: HashMap<&str, usize> = HashMap::new();
    let mut queue: VecDeque<(&str, usize)> = VecDeque::new();
    for source in sources {
        if depth.insert(source, 0).is_none() {
            queue.push_back((source, 0));
        }
    }
    while let Some((node, at)) = queue.pop_front() {
        let Some(next) = forward.get(node) else {
            continue;
        };
        for target in next {
            if depth.contains_key(target) {
                continue;
            }
            depth.insert(target, at + 1);
            queue.push_back((target, at + 1));
        }
    }
    let deepest = depth.values().copied().max().unwrap_or(0).max(1) as f64;
    creature
        .neurons
        .iter()
        .map(|n| {
            let fraction = depth
                .get(n.uuid.as_str())
                .map_or(1.0, |d| (*d as f64 / deepest).min(1.0));
            (n.uuid.clone(), fraction)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixtures::{creature, neuron, synapse};
    use crate::stats::{NeuronStats, STATS_FORMAT_VERSION, SampleSpec};

    /// `input-0 → chain → hub → output-0`, plus an unmeasured lone neuron.
    fn wired() -> CreatureExport {
        creature(
            1,
            1,
            vec![
                neuron("hidden", "chain", 0.0, Some("IDENTITY")),
                neuron("hidden", "hub", 0.0, Some("TANH")),
                neuron("hidden", "lone", 0.0, Some("TANH")),
                neuron("output", "output-0", 0.0, Some("IDENTITY")),
            ],
            vec![
                synapse("input-0", "chain", 1.0),
                synapse("chain", "hub", 2.0),
                synapse("hub", "output-0", -3.0),
                synapse("input-0", "lone", 1.0),
                synapse("lone", "output-0", 0.5),
            ],
        )
    }

    fn stats() -> ActivationStats {
        ActivationStats {
            format_version: STATS_FORMAT_VERSION,
            creature_checksum: "t".into(),
            corpus_identity: "c".into(),
            record_count: 10,
            corpus_record_count: 10,
            sample: SampleSpec::full(),
            stopped_early: false,
            scan_ms: 0,
            from_cache: false,
            neurons: vec![
                NeuronStats {
                    uuid: "chain".into(),
                    neuron_index: 0,
                    count: 10,
                    mean: 0.0,
                    variance: 0.25,
                    std_dev: 0.5,
                    mean_abs: 0.4,
                    min: -1.0,
                    max: 1.0,
                },
                NeuronStats {
                    uuid: "hub".into(),
                    neuron_index: 1,
                    count: 10,
                    mean: 0.0,
                    variance: 4.0,
                    std_dev: 2.0,
                    mean_abs: 2.0,
                    min: -6.0,
                    max: 6.0,
                },
            ],
        }
    }

    fn features() -> HashMap<String, CandidateFeatures> {
        extract(&wired(), &stats(), &PriorEvidence::new())
    }

    #[test]
    fn every_hidden_neuron_gets_a_feature_vector() {
        let got = features();
        let mut uuids: Vec<&String> = got.keys().collect();
        uuids.sort();
        assert_eq!(uuids, ["chain", "hub", "lone"]);
        assert!(
            !got.contains_key("output-0"),
            "outputs are never candidates"
        );
    }

    #[test]
    fn structural_signals_come_from_the_topology() {
        let got = features();
        let hub = got["hub"];
        assert_eq!(hub.fan_in, 1);
        assert_eq!(hub.fan_out, 1);
        assert_eq!(hub.outgoing_weight, 3.0, "abs weight, not signed");
        assert_eq!(hub.direct_growth_units, growth_units(1, 2));
        assert!(!hub.identity);
        assert!(got["chain"].identity, "IDENTITY squash is an exact fold");
    }

    #[test]
    fn cutting_a_chain_head_predicts_more_saving_than_a_lone_neuron() {
        let got = features();
        assert!(
            got["chain"].cascade_growth_units > got["lone"].cascade_growth_units,
            "{got:?}"
        );
    }

    #[test]
    fn an_unmeasured_neuron_is_flagged_rather_than_read_as_quiet() {
        let got = features();
        let lone = got["lone"];
        assert!(!lone.measured, "the scan never covered it");
        assert_eq!(lone.mean_abs, 0.0);
        assert!(got["hub"].measured);
        assert_eq!(got["hub"].downstream_sensitivity(), 6.0);
    }

    #[test]
    fn depth_rises_away_from_the_inputs() {
        let got = features();
        assert!(
            got["chain"].depth_fraction < got["hub"].depth_fraction,
            "{got:?}"
        );
        assert_eq!(got["chain"].depth_fraction, 0.5);
        assert_eq!(got["hub"].depth_fraction, 1.0);
    }

    #[test]
    fn a_neuron_no_input_reaches_is_deepest_not_missing() {
        let orphan = creature(
            1,
            1,
            vec![
                neuron("hidden", "reachable", 0.0, Some("TANH")),
                neuron("hidden", "orphan", 0.0, Some("TANH")),
                neuron("output", "output-0", 0.0, Some("IDENTITY")),
            ],
            vec![
                synapse("input-0", "reachable", 1.0),
                synapse("reachable", "output-0", 1.0),
                synapse("orphan", "output-0", 1.0),
            ],
        );
        let got = extract(&orphan, &ActivationStats::empty(), &PriorEvidence::new());
        assert_eq!(got["orphan"].depth_fraction, 1.0);
    }

    #[test]
    fn the_vector_matches_the_named_schema() {
        let got = features();
        let vector = got["hub"].vector();
        assert_eq!(vector.len(), FEATURE_NAMES.len());
        let named = got["hub"].named();
        assert_eq!(named[0], ("measured", 1.0));
        assert_eq!(
            named.iter().find(|(n, _)| *n == "logFanOut").unwrap().1,
            2.0_f64.ln()
        );
        assert!(vector.iter().all(|v| v.is_finite()), "{vector:?}");
    }

    fn learning(uuid: &str, outcome: Outcome, full_delta: Option<f64>) -> Learning {
        Learning {
            version: crate::learnings::LEARNINGS_FORMAT_VERSION,
            uuid: uuid.into(),
            kind: "ablation".into(),
            outcome,
            unix_secs: 1,
            host: "h".into(),
            full_delta,
            group: None,
        }
    }

    #[test]
    fn historical_evidence_counts_confirmed_wins_and_failures_apart() {
        let mut evidence = PriorEvidence::new();
        evidence.add(&learning("hub", Outcome::Accepted, None), 1e-6);
        // Confirmed but not applied is a win, not a failure (Issue #52).
        evidence.add(&learning("hub", Outcome::Rejected, Some(0.5)), 1e-6);
        evidence.add(&learning("hub", Outcome::Rejected, Some(-0.5)), 1e-6);
        evidence.add(&learning("chain", Outcome::Rejected, None), 1e-6);
        assert_eq!(evidence.counts("hub"), (2, 1));
        assert_eq!(evidence.counts("chain"), (0, 1));
        assert_eq!(evidence.counts("unknown"), (0, 0));
        assert_eq!(evidence.len(), 2);
        assert!(!evidence.is_empty());
    }

    #[test]
    fn evidence_reaches_the_feature_vector() {
        let mut evidence = PriorEvidence::new();
        evidence.add(&learning("hub", Outcome::Accepted, Some(0.2)), 1e-6);
        let got = extract(&wired(), &stats(), &evidence);
        assert_eq!(got["hub"].prior_wins, 1);
        assert_eq!(got["hub"].prior_failures, 0);
        assert_eq!(got["lone"].prior_wins, 0);
    }

    #[test]
    fn history_from_prior_epochs_is_folded_in_by_uuid() {
        let prior = vec![
            HistoricalLearning {
                corpus_identity: "old".into(),
                learning: learning("hub", Outcome::Accepted, None),
            },
            HistoricalLearning {
                corpus_identity: "older".into(),
                learning: learning("hub", Outcome::Rejected, None),
            },
        ];
        let evidence = PriorEvidence::from_history(&prior, 1e-6);
        assert_eq!(evidence.counts("hub"), (1, 1));
    }
}
