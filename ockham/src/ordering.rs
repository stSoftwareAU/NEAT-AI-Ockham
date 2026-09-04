//! Named, reproducible candidate ordering strategies (Issue #11).
//!
//! An ordering only decides **which hidden neuron is tested sooner**. It never
//! declares a neuron safe to remove: every candidate still passes
//! `creature.validate()`, the sampled screen and full authoritative scoring
//! exactly as it does under the random control.
//!
//! Every strategy starts from the seeded random permutation and then applies a
//! **stable** sort by its ranking key, so ties keep an unbiased random order
//! and the whole visitation order is reproducible from
//! `(seed, ordering, random quota)`.
//!
//! [`Ordering::Random`] is the control and remains the default. A strategy is
//! only promoted to the default when benchmark evidence shows better
//! scorer-verified improvement economics.

use std::collections::{HashMap, HashSet};
use std::fmt;

use neat_core::{CreatureExport, SquashType, parse_squash_name};
use serde::{Deserialize, Serialize};

use crate::ablation::growth_units;
use crate::cascade::{CascadeEstimate, CascadeIndex};
use crate::stats::ActivationStats;

/// Ranking strategy for the pruning sweep.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "kebab-case")]
pub enum Ordering {
    /// Seeded random without replacement — the control, and the default.
    #[default]
    Random,
    /// Lowest activation variance (nearly constant neurons) first.
    LowVariance,
    /// Lowest mean absolute activation (quietest neurons) first.
    LowMeanAbs,
    /// Smallest activation range (`max - min`) first.
    NarrowRange,
    /// Smallest `mean_abs_activation × Σ|outgoing weight|` first.
    LowOutgoingContribution,
    /// Fewest outgoing synapses (smallest structural blast radius) first.
    LowFanOut,
    /// Largest growth-unit saving per removed structure first.
    HighGrowthSaving,
    /// `IDENTITY` neurons (exact-fold opportunities) first.
    IdentityFirst,
    /// Largest cascade-aware growth-unit saving first (Issue #106).
    ///
    /// Unlike [`Ordering::HighGrowthSaving`], which counts only the neuron and
    /// the synapses touching it, this counts the structure recursive cleanup
    /// would strand behind the cut as well.
    CascadeSaving,
    /// Least estimated behavioural damage per cascade growth unit first (#106).
    ///
    /// `mean_abs_activation × Σ abs(outgoing weight)` — the downstream
    /// sensitivity [`Ordering::LowOutgoingContribution`] ranks by — divided by
    /// the cascade saving, so a quiet neuron that takes a lot of structure with
    /// it is tried before a loud one that takes little.
    CascadeRiskRatio,
}

impl Ordering {
    /// Every strategy, in documentation order.
    pub const ALL: &'static [Ordering] = &[
        Ordering::Random,
        Ordering::LowVariance,
        Ordering::LowMeanAbs,
        Ordering::NarrowRange,
        Ordering::LowOutgoingContribution,
        Ordering::LowFanOut,
        Ordering::HighGrowthSaving,
        Ordering::IdentityFirst,
        Ordering::CascadeSaving,
        Ordering::CascadeRiskRatio,
    ];

    /// Kebab-case name used on the CLI and in the journal.
    pub fn name(self) -> &'static str {
        match self {
            Ordering::Random => "random",
            Ordering::LowVariance => "low-variance",
            Ordering::LowMeanAbs => "low-mean-abs",
            Ordering::NarrowRange => "narrow-range",
            Ordering::LowOutgoingContribution => "low-outgoing-contribution",
            Ordering::LowFanOut => "low-fan-out",
            Ordering::HighGrowthSaving => "high-growth-saving",
            Ordering::IdentityFirst => "identity-first",
            Ordering::CascadeSaving => "cascade-saving",
            Ordering::CascadeRiskRatio => "cascade-risk-ratio",
        }
    }

    /// Whether the strategy ranks by a cascade dry-run (Issue #106).
    ///
    /// The estimates cost one index of the creature, so they are built only for
    /// the strategies that read them.
    fn needs_cascade(self) -> bool {
        matches!(self, Ordering::CascadeSaving | Ordering::CascadeRiskRatio)
    }

    /// Comma-separated list of every accepted name.
    pub fn names() -> String {
        Self::ALL
            .iter()
            .map(|o| o.name())
            .collect::<Vec<_>>()
            .join(", ")
    }

    /// Parse a kebab-case name; unknown names name the valid set.
    pub fn parse(name: &str) -> Result<Self, String> {
        Self::ALL
            .iter()
            .copied()
            .find(|o| o.name() == name)
            .ok_or_else(|| {
                format!(
                    "unknown ordering `{name}`; expected one of: {}",
                    Self::names()
                )
            })
    }
}

impl fmt::Display for Ordering {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

impl std::str::FromStr for Ordering {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::parse(s)
    }
}

/// Strategy plus the fraction of slots reserved for random exploration.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct OrderingConfig {
    /// Ranking strategy.
    pub strategy: Ordering,
    /// Fraction of visitation slots drawn from the random control, in `[0, 1)`.
    pub random_quota: f64,
}

impl OrderingConfig {
    /// Config for `strategy` with no reserved random quota.
    pub fn new(strategy: Ordering) -> Self {
        Self {
            strategy,
            random_quota: 0.0,
        }
    }

    /// Reject a quota outside `[0, 1)`; the message names the flag.
    pub fn validate(&self) -> Result<(), String> {
        if !(self.random_quota.is_finite() && (0.0..1.0).contains(&self.random_quota)) {
            return Err("--ordering-random-quota must be in [0, 1)".into());
        }
        Ok(())
    }
}

/// SplitMix64 — enough for a reproducible Fisher–Yates shuffle.
struct SplitMix64(u64);

impl SplitMix64 {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn shuffle<T>(&mut self, items: &mut [T]) {
        for i in (1..items.len()).rev() {
            let j = (self.next() as usize) % (i + 1);
            items.swap(i, j);
        }
    }
}

/// Hidden-neuron UUIDs shuffled with `seed` — the random control order.
pub fn random_order(creature: &CreatureExport, seed: u64) -> Vec<String> {
    let mut order: Vec<String> = creature
        .neurons
        .iter()
        .filter(|n| n.neuron_type == "hidden")
        .map(|n| n.uuid.clone())
        .collect();
    SplitMix64(seed).shuffle(&mut order);
    order
}

/// Ranking key for `uuid` under `strategy`. Lower sorts earlier.
///
/// Neurons the statistics do not cover sort last rather than being dropped:
/// an ordering may never remove a neuron from the sweep.
fn rank_key(
    creature: &CreatureExport,
    stats: &ActivationStats,
    strategy: Ordering,
    cascade: Option<&HashMap<&str, CascadeEstimate>>,
    uuid: &str,
) -> f64 {
    let neuron_stats = stats.by_uuid(uuid);
    match strategy {
        Ordering::Random => 0.0,
        Ordering::LowVariance => neuron_stats.map_or(f64::INFINITY, |s| s.variance),
        Ordering::LowMeanAbs => neuron_stats.map_or(f64::INFINITY, |s| s.mean_abs),
        Ordering::NarrowRange => neuron_stats.map_or(f64::INFINITY, |s| s.max - s.min),
        Ordering::LowOutgoingContribution => neuron_stats.map_or(f64::INFINITY, |s| {
            s.mean_abs * outgoing_weight_sum(creature, uuid)
        }),
        Ordering::LowFanOut => creature
            .synapses
            .iter()
            .filter(|s| s.from_uuid == uuid)
            .count() as f64,
        // Negated so the largest structural saving sorts first.
        Ordering::HighGrowthSaving => {
            let touching = creature
                .synapses
                .iter()
                .filter(|s| s.from_uuid == uuid || s.to_uuid == uuid)
                .count();
            -growth_units(1, touching)
        }
        Ordering::IdentityFirst => {
            let identity = creature.neurons.iter().any(|n| {
                n.uuid == uuid
                    && parse_squash_name(n.squash.as_deref().unwrap_or("IDENTITY"))
                        .is_ok_and(|s| s == SquashType::Identity)
            });
            if identity { 0.0 } else { 1.0 }
        }
        // Negated so the largest cascade saving sorts first. A neuron the
        // estimate does not cover sorts last rather than being dropped.
        Ordering::CascadeSaving => -cascade_saving(cascade, uuid).unwrap_or(0.0),
        Ordering::CascadeRiskRatio => match (neuron_stats, cascade_saving(cascade, uuid)) {
            // The saving is at least one growth unit — the candidate itself —
            // so the ratio is never a division by zero.
            (Some(s), Some(saving)) => s.mean_abs * outgoing_weight_sum(creature, uuid) / saving,
            _ => f64::INFINITY,
        },
    }
}

/// `Σ abs(weight)` over the synapses leaving `uuid` — the downstream reach.
fn outgoing_weight_sum(creature: &CreatureExport, uuid: &str) -> f64 {
    creature
        .synapses
        .iter()
        .filter(|s| s.from_uuid == uuid)
        .map(|s| s.weight.abs())
        .sum()
}

/// Estimated cascade growth-unit saving for `uuid`, when one was computed.
fn cascade_saving(cascade: Option<&HashMap<&str, CascadeEstimate>>, uuid: &str) -> Option<f64> {
    cascade?.get(uuid).map(|e| e.growth_units)
}

/// Interleave `ranked` with the random control, reserving `quota` of the slots.
///
/// Slot `i` is filled from the random order while the random share so far is
/// below `quota`; otherwise it takes the next unused ranked neuron. The result
/// is always a permutation of the same UUIDs.
fn blend(ranked: Vec<String>, random: &[String], quota: f64) -> Vec<String> {
    if quota <= 0.0 {
        return ranked;
    }
    let total = ranked.len();
    let mut used: HashSet<String> = HashSet::with_capacity(total);
    let mut out: Vec<String> = Vec::with_capacity(total);
    let (mut ranked_at, mut random_at, mut random_taken) = (0usize, 0usize, 0usize);
    while out.len() < total {
        let take_random = (random_taken as f64) < quota * (out.len() + 1) as f64;
        let (source, cursor) = if take_random {
            (random, &mut random_at)
        } else {
            (&ranked[..], &mut ranked_at)
        };
        while *cursor < source.len() && used.contains(&source[*cursor]) {
            *cursor += 1;
        }
        if *cursor >= source.len() {
            // That source is exhausted; drain whatever the other still holds.
            let other: &[String] = if take_random { &ranked } else { random };
            for uuid in other {
                if used.insert(uuid.clone()) {
                    out.push(uuid.clone());
                }
            }
            break;
        }
        let uuid = source[*cursor].clone();
        used.insert(uuid.clone());
        out.push(uuid);
        if take_random {
            random_taken += 1;
        }
    }
    out
}

/// Hidden-neuron visitation order for `cfg`, reproducible from `seed`.
///
/// The result is always a permutation of every hidden UUID: an ordering
/// reprioritises the sweep, it never shrinks it.
pub fn hidden_order(
    creature: &CreatureExport,
    stats: &ActivationStats,
    cfg: OrderingConfig,
    seed: u64,
) -> Vec<String> {
    let random = random_order(creature, seed);
    if cfg.strategy == Ordering::Random {
        return random;
    }
    // Topology does not change while the order is built, so the cascade
    // dry-runs are the creature's own index, walked once per hidden neuron
    // (Issue #106) rather than once per comparison the sort makes.
    let index = cfg
        .strategy
        .needs_cascade()
        .then(|| CascadeIndex::new(creature));
    let cascade = index.as_ref().map(CascadeIndex::hidden_estimates);
    let mut keyed: Vec<(f64, String)> = random
        .iter()
        .map(|uuid| {
            (
                rank_key(creature, stats, cfg.strategy, cascade.as_ref(), uuid),
                uuid.clone(),
            )
        })
        .collect();
    // Stable, so ties keep the unbiased random order behind them.
    keyed.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    let ranked: Vec<String> = keyed.into_iter().map(|(_, uuid)| uuid).collect();
    blend(ranked, &random, cfg.random_quota)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixtures::{creature, neuron, synapse};
    use crate::stats::{NeuronStats, STATS_FORMAT_VERSION, SampleSpec};

    /// Three hidden neurons with deliberately different signals.
    ///
    /// `h_flat` is nearly constant and quiet, `h_loud` is noisy with a heavy
    /// outgoing weight, `h_hub` is mid-range but has the largest fan-out.
    fn wired() -> CreatureExport {
        creature(
            1,
            2,
            vec![
                neuron("hidden", "h_flat", 0.0, Some("IDENTITY")),
                neuron("hidden", "h_loud", 0.0, Some("TANH")),
                neuron("hidden", "h_hub", 0.0, Some("TANH")),
                neuron("output", "output-0", 0.0, Some("IDENTITY")),
                neuron("output", "output-1", 0.0, Some("IDENTITY")),
            ],
            vec![
                synapse("input-0", "h_flat", 1.0),
                synapse("input-0", "h_loud", 1.0),
                synapse("input-0", "h_hub", 1.0),
                synapse("h_flat", "output-0", 0.5),
                synapse("h_loud", "output-0", 4.0),
                synapse("h_hub", "output-0", 1.0),
                synapse("h_hub", "output-1", 1.0),
            ],
        )
    }

    fn stats() -> ActivationStats {
        let rows = [
            // uuid, variance, mean_abs, min, max
            ("h_flat", 0.001, 0.01, -0.05, 0.05),
            ("h_loud", 4.000, 2.00, -6.00, 6.00),
            ("h_hub", 0.500, 0.40, -1.00, 1.00),
        ];
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
            neurons: rows
                .iter()
                .enumerate()
                .map(|(i, (uuid, variance, mean_abs, min, max))| NeuronStats {
                    uuid: (*uuid).into(),
                    neuron_index: i,
                    count: 10,
                    mean: 0.0,
                    variance: *variance,
                    std_dev: variance.sqrt(),
                    mean_abs: *mean_abs,
                    min: *min,
                    max: *max,
                })
                .collect(),
        }
    }

    fn order(strategy: Ordering) -> Vec<String> {
        hidden_order(&wired(), &stats(), OrderingConfig::new(strategy), 11)
    }

    #[test]
    fn every_ordering_is_a_permutation_of_the_hidden_neurons() {
        let expected = {
            let mut u: Vec<String> = vec!["h_flat".into(), "h_loud".into(), "h_hub".into()];
            u.sort();
            u
        };
        for strategy in Ordering::ALL {
            for quota in [0.0, 0.25, 0.5, 0.9] {
                let got = hidden_order(
                    &wired(),
                    &stats(),
                    OrderingConfig {
                        strategy: *strategy,
                        random_quota: quota,
                    },
                    3,
                );
                let mut sorted = got.clone();
                sorted.sort();
                assert_eq!(sorted, expected, "{strategy} quota={quota} lost a neuron");
            }
        }
    }

    #[test]
    fn a_fixed_seed_and_ordering_reproduce_the_same_visitation_order() {
        for strategy in Ordering::ALL {
            let cfg = OrderingConfig {
                strategy: *strategy,
                random_quota: 0.3,
            };
            let a = hidden_order(&wired(), &stats(), cfg, 99);
            let b = hidden_order(&wired(), &stats(), cfg, 99);
            assert_eq!(a, b, "{strategy} is not reproducible");
        }
    }

    #[test]
    fn random_control_changes_with_the_seed() {
        let a = hidden_order(&wired(), &stats(), OrderingConfig::default(), 1);
        let b = hidden_order(&wired(), &stats(), OrderingConfig::default(), 2);
        assert_ne!(a, b);
    }

    #[test]
    fn low_variance_visits_the_flattest_neuron_first() {
        assert_eq!(order(Ordering::LowVariance), ["h_flat", "h_hub", "h_loud"]);
    }

    #[test]
    fn low_mean_abs_visits_the_quietest_neuron_first() {
        assert_eq!(order(Ordering::LowMeanAbs), ["h_flat", "h_hub", "h_loud"]);
    }

    #[test]
    fn narrow_range_visits_the_tightest_neuron_first() {
        assert_eq!(order(Ordering::NarrowRange), ["h_flat", "h_hub", "h_loud"]);
    }

    #[test]
    fn low_outgoing_contribution_multiplies_mean_abs_by_outgoing_weight() {
        // h_flat 0.01*0.5 = 0.005; h_hub 0.40*2.0 = 0.8; h_loud 2.00*4.0 = 8.0.
        assert_eq!(
            order(Ordering::LowOutgoingContribution),
            ["h_flat", "h_hub", "h_loud"]
        );
    }

    #[test]
    fn low_fan_out_visits_the_smallest_blast_radius_first() {
        // h_hub has two outgoing synapses; the others have one each, and the
        // random tie-break keeps them ahead of it.
        let got = order(Ordering::LowFanOut);
        assert_eq!(got[2], "h_hub", "{got:?}");
    }

    #[test]
    fn high_growth_saving_visits_the_biggest_structural_saving_first() {
        // h_hub touches three synapses; h_flat and h_loud touch two each.
        let got = order(Ordering::HighGrowthSaving);
        assert_eq!(got[0], "h_hub", "{got:?}");
    }

    /// `input-0 → f1 → f2 → hub → output-0`, beside a two-output `loud`.
    ///
    /// Every chain member touches two synapses and `loud` touches three, so
    /// `high-growth-saving` ranks `loud` first — yet cutting any chain member
    /// strands the other two, which only a cascade dry-run sees. `loud` is also
    /// the noisiest and most heavily weighted, so the risk ratio must leave it
    /// behind the quiet chain.
    fn cascading() -> CreatureExport {
        creature(
            1,
            2,
            vec![
                neuron("hidden", "f1", 0.0, Some("TANH")),
                neuron("hidden", "f2", 0.0, Some("TANH")),
                neuron("hidden", "hub", 0.0, Some("TANH")),
                neuron("hidden", "loud", 0.0, Some("TANH")),
                neuron("output", "output-0", 0.0, Some("IDENTITY")),
                neuron("output", "output-1", 0.0, Some("IDENTITY")),
            ],
            vec![
                synapse("input-0", "f1", 1.0),
                synapse("f1", "f2", 1.0),
                synapse("f2", "hub", 1.0),
                synapse("hub", "output-0", 1.0),
                synapse("input-0", "loud", 1.0),
                synapse("loud", "output-0", 4.0),
                synapse("loud", "output-1", 3.0),
            ],
        )
    }

    /// Statistics for [`cascading`]: the chain is quiet, `loud` is not.
    fn cascading_stats() -> ActivationStats {
        let mut stats = stats();
        stats.neurons.clear();
        for (i, (uuid, mean_abs)) in [("f1", 0.02), ("f2", 0.03), ("hub", 0.01), ("loud", 3.0)]
            .iter()
            .enumerate()
        {
            stats.neurons.push(NeuronStats {
                uuid: (*uuid).into(),
                neuron_index: i,
                count: 10,
                mean: 0.0,
                variance: *mean_abs,
                std_dev: mean_abs.sqrt(),
                mean_abs: *mean_abs,
                min: -*mean_abs,
                max: *mean_abs,
            });
        }
        stats
    }

    fn cascading_order(strategy: Ordering) -> Vec<String> {
        hidden_order(
            &cascading(),
            &cascading_stats(),
            OrderingConfig::new(strategy),
            7,
        )
    }

    #[test]
    fn cascade_saving_visits_the_cuts_that_strand_the_most_structure_first() {
        // Cutting any chain member removes all three of them and four synapses
        // (3.4 units); `loud` reaches no further than its own edges (1.3).
        let got = cascading_order(Ordering::CascadeSaving);
        assert_eq!(got[3], "loud", "{got:?}");
        let mut chain = got[..3].to_vec();
        chain.sort();
        assert_eq!(chain, ["f1", "f2", "hub"], "{got:?}");
    }

    #[test]
    fn cascade_saving_outranks_the_edge_count_high_growth_saving_reads() {
        // `loud` touches three synapses to every chain member's two, so the
        // edge-count ranking tries it first and the cascade ranking tries it
        // last: the chain it cannot see is worth two and a half times more.
        let touching = cascading_order(Ordering::HighGrowthSaving);
        assert_eq!(touching[0], "loud", "{touching:?}");
        let cascade = cascading_order(Ordering::CascadeSaving);
        assert_eq!(cascade[3], "loud", "{cascade:?}");
    }

    #[test]
    fn cascade_risk_ratio_prefers_quiet_cuts_that_save_the_most_structure() {
        let got = cascading_order(Ordering::CascadeRiskRatio);
        // hub: 0.01 × 1.0 / 3.4. loud: 3.0 × 6.0 / 1.3 — the loudest neuron
        // with the smallest cascade is tried last however many edges it has.
        assert_eq!(got[0], "hub", "{got:?}");
        assert_eq!(got[3], "loud", "{got:?}");
    }

    #[test]
    fn a_neuron_without_statistics_ranks_last_under_the_risk_ratio() {
        let creature = cascading();
        let mut partial = cascading_stats();
        partial.neurons.retain(|n| n.uuid != "hub");
        let got = hidden_order(
            &creature,
            &partial,
            OrderingConfig::new(Ordering::CascadeRiskRatio),
            7,
        );
        assert_eq!(got.len(), 4, "{got:?}");
        assert_eq!(got[3], "hub", "{got:?}");
    }

    #[test]
    fn identity_first_visits_identity_neurons_ahead_of_the_rest() {
        let got = order(Ordering::IdentityFirst);
        assert_eq!(got[0], "h_flat", "{got:?}");
    }

    #[test]
    fn a_random_quota_reserves_exploration_slots_for_the_control() {
        let cfg = OrderingConfig {
            strategy: Ordering::LowVariance,
            random_quota: 0.9,
        };
        let creature = wired();
        let stats = stats();
        // A heavy quota must reproduce the control order, not the ranking.
        let blended = hidden_order(&creature, &stats, cfg, 11);
        assert_eq!(blended, random_order(&creature, 11));
        // No quota must reproduce the pure ranking.
        let ranked = hidden_order(&creature, &stats, OrderingConfig::new(cfg.strategy), 11);
        assert_eq!(ranked[0], "h_flat");
    }

    #[test]
    fn a_neuron_without_statistics_is_visited_last_not_dropped() {
        let creature = wired();
        let mut partial = stats();
        partial.neurons.retain(|n| n.uuid != "h_flat");
        let got = hidden_order(
            &creature,
            &partial,
            OrderingConfig::new(Ordering::LowVariance),
            5,
        );
        assert_eq!(got.len(), 3, "{got:?}");
        assert_eq!(got[2], "h_flat", "{got:?}");
    }

    #[test]
    fn names_round_trip_and_unknown_names_list_the_valid_set() {
        for strategy in Ordering::ALL {
            assert_eq!(Ordering::parse(strategy.name()).unwrap(), *strategy);
            assert_eq!(strategy.to_string(), strategy.name());
        }
        let err = Ordering::parse("cleverest").unwrap_err();
        assert!(err.contains("cleverest"), "{err}");
        assert!(err.contains("low-variance"), "{err}");
        assert_eq!(Ordering::default(), Ordering::Random);
    }

    #[test]
    fn an_out_of_range_random_quota_names_the_flag() {
        let bad = OrderingConfig {
            strategy: Ordering::LowVariance,
            random_quota: 1.0,
        };
        assert!(
            bad.validate()
                .unwrap_err()
                .contains("--ordering-random-quota")
        );
        assert!(
            OrderingConfig::new(Ordering::LowVariance)
                .validate()
                .is_ok()
        );
    }
}
