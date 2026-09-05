//! Behavioural signatures and correlated-neuron discovery (Issue #109).
//!
//! A mature evolved creature accumulates hidden neurons that behave almost
//! identically. Neither is quiet, so mean-activation ablation never nominates
//! them — yet between the two there is one neuron too many.
//!
//! Finding them cannot cost a full `N × N` correlation matrix: a GRQ forest
//! carries thousands of hidden neurons, and the razor has forty-five minutes.
//! The discovery here is three cheap stages instead:
//!
//! ```text
//! probe vectors  →  64-bit sign signature  →  LSH bands  →  correlate the buckets
//! ```
//!
//! 1. [`crate::stats`] retains each neuron's post-activation at a handful of
//!    deterministically-placed probe records — `probes` floats per neuron, and
//!    nothing at all when the feature is off.
//! 2. [`signature`] reduces that vector to one `u64`: bit `i` is set when the
//!    neuron sat at or above its own probe mean at probe `i`. Two neurons that
//!    rise and fall together have a signature that differs in few bits.
//! 3. Signatures are split into bands of [`DiscoveryConfig::band_bits`] and
//!    bucketed by band value, so only neurons that already agree on a whole
//!    band are ever compared. An anti-correlated pair agrees on the
//!    *complement* of every band, so each key is canonicalised to the smaller
//!    of the band and its complement and the two land in one bucket.
//! 4. Pearson correlation — the expensive, exact part — runs on the probe
//!    vectors of bucket members only.
//!
//! The correlation threshold **proposes**; it never accepts. Every pair that
//! clears it becomes a [`MergeProposal`] in both survivor directions, and each
//! one still has to build a valid candidate through [`crate::merge`], survive
//! `creature.validate()`, the sampled screen and the authoritative full scorer.
//!
//! Nothing here is capped silently: an over-full bucket is truncated and
//! counted in [`DiscoveryReport::truncated_buckets`], so a run reports the
//! comparisons it declined to make instead of presenting a partial sweep as a
//! complete one.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::time::Instant;

use crate::merge::LinearRelation;
use crate::stats::ActivationStats;

/// Bits in a behavioural signature — one `u64`.
pub const SIGNATURE_BITS: u32 = 64;
/// Default `|Pearson r|` a pair must reach to be proposed.
pub const DEFAULT_MIN_CORRELATION: f64 = 0.98;
/// Default *minimum* bits per locality-sensitive band.
///
/// Eight bits give 256 buckets per band and eight bands over a 64-bit
/// signature: a pair has to agree (or disagree) on all eight bits of at least
/// one band to be compared at all. [`discover`] widens the band above this
/// floor on a large creature — see [`effective_band_bits`].
pub const DEFAULT_BAND_BITS: u32 = 8;
/// Default cap on neurons compared pairwise inside one bucket.
///
/// A bucket is quadratic in its own size, so a degenerate creature — thousands
/// of neurons sharing one signature — would otherwise reinstate the `N × N`
/// cost the banding exists to avoid.
pub const DEFAULT_MAX_BUCKET: usize = 48;
/// Default cap on proposals kept per removable neuron.
pub const DEFAULT_MAX_PARTNERS: usize = 3;
/// Fewest probe records a signature may be built from.
///
/// Below this the sign vector is noise: two unrelated neurons agree on four
/// coin flips often enough to fill the sweep with proposals nothing confirms.
pub const MIN_PROBE_RECORDS: usize = 8;

/// Knobs for [`discover`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DiscoveryConfig {
    /// `|Pearson r|` a pair must reach to be proposed.
    pub min_correlation: f64,
    /// Minimum bits per locality-sensitive band, in `1..=SIGNATURE_BITS`.
    ///
    /// A floor, not a fixed width: see [`effective_band_bits`].
    pub band_bits: u32,
    /// Neurons compared pairwise inside one bucket.
    pub max_bucket: usize,
    /// Proposals kept per removable neuron.
    pub max_partners: usize,
}

impl Default for DiscoveryConfig {
    fn default() -> Self {
        Self {
            min_correlation: DEFAULT_MIN_CORRELATION,
            band_bits: DEFAULT_BAND_BITS,
            max_bucket: DEFAULT_MAX_BUCKET,
            max_partners: DEFAULT_MAX_PARTNERS,
        }
    }
}

impl DiscoveryConfig {
    /// Default discovery at `min_correlation`.
    pub fn with_min_correlation(min_correlation: f64) -> Self {
        Self {
            min_correlation,
            ..Self::default()
        }
    }

    /// Reject a configuration that could not discover anything, by name.
    pub fn validate(&self) -> Result<(), String> {
        if !(self.min_correlation > 0.0 && self.min_correlation <= 1.0) {
            return Err("--merge-correlation must be in (0, 1]".into());
        }
        if self.band_bits == 0 || self.band_bits > SIGNATURE_BITS {
            return Err(format!("--merge-band-bits must be in 1..={SIGNATURE_BITS}"));
        }
        if self.max_bucket < 2 {
            return Err("--merge-max-bucket must be >= 2".into());
        }
        if self.max_partners == 0 {
            return Err("--merge-max-partners must be > 0".into());
        }
        Ok(())
    }
}

/// One proposed merge, in one survivor direction.
///
/// `removed ≈ scale * survivor + offset` over the probe records, which is the
/// relation [`crate::merge::merge_correlated`] folds into the survivor's
/// outgoing weights. A proposal is evidence for trying a candidate and nothing
/// more; the scorer decides.
#[derive(Debug, Clone, PartialEq)]
pub struct MergeProposal {
    /// Hidden neuron the candidate would remove.
    pub removed_uuid: String,
    /// Hidden neuron that would absorb its contribution.
    pub survivor_uuid: String,
    /// Pearson correlation of the two probe vectors, sign included.
    pub correlation: f64,
    /// Least-squares fit of `removed` against `survivor`.
    pub relation: LinearRelation,
}

/// What one discovery pass cost and found.
///
/// Every number a benchmark or a run report needs, and — deliberately —
/// [`Self::truncated_buckets`], so a bounded sweep never reads as a complete
/// one.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct DiscoveryReport {
    /// Neurons with a usable probe vector.
    pub signed: usize,
    /// Neurons skipped for a short, missing or flat probe vector.
    pub unsignable: usize,
    /// Non-empty buckets across every band.
    pub buckets: usize,
    /// Distinct pairs whose correlation was actually computed.
    pub compared_pairs: usize,
    /// Pairs that cleared the correlation threshold.
    pub correlated_pairs: usize,
    /// Proposals kept, counting both survivor directions.
    pub proposals: usize,
    /// Bits per band the pass actually used ([`effective_band_bits`]).
    pub band_bits: u32,
    /// Buckets whose members were truncated to `max_bucket`.
    pub truncated_buckets: usize,
    /// Members dropped by that truncation.
    pub dropped_members: usize,
    /// Wall time of the discovery pass (ms).
    pub discovery_ms: u64,
}

/// Merge proposals, looked up by the neuron a candidate would remove.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct MergeIndex {
    by_removed: HashMap<String, Vec<MergeProposal>>,
    report: DiscoveryReport,
}

impl MergeIndex {
    /// The shared empty index — a run with correlated-neuron merging off.
    ///
    /// Borrowed rather than constructed so callers holding a `&MergeIndex` for
    /// the length of a run can default to "no proposals" without an allocation
    /// each time.
    pub fn empty() -> &'static Self {
        static EMPTY: std::sync::OnceLock<MergeIndex> = std::sync::OnceLock::new();
        EMPTY.get_or_init(MergeIndex::default)
    }

    /// Proposals that would remove `uuid`, strongest correlation first.
    pub fn for_removed(&self, uuid: &str) -> &[MergeProposal] {
        self.by_removed.get(uuid).map_or(&[], Vec::as_slice)
    }

    /// This index with proposals for `keep` only (#109).
    ///
    /// Replay uses it so a recorded verdict is rebuilt as the transform it was
    /// judged as: a uuid the fleet accepted as an `ablation` must not come back
    /// as a merge because the current signatures happen to offer a partner —
    /// that is a different cut wearing the winner's uuid. The report is carried
    /// through unchanged: it describes the discovery pass, which the
    /// restriction does not re-run.
    pub fn restricted_to(&self, keep: &HashSet<String>) -> Self {
        Self {
            by_removed: self
                .by_removed
                .iter()
                .filter(|(uuid, _)| keep.contains(*uuid))
                .map(|(uuid, proposals)| (uuid.clone(), proposals.clone()))
                .collect(),
            report: self.report,
        }
    }

    /// `true` when nothing was proposed — the merge path is then inert.
    pub fn is_empty(&self) -> bool {
        self.by_removed.is_empty()
    }

    /// What the discovery pass cost and found.
    pub fn report(&self) -> DiscoveryReport {
        self.report
    }

    /// Every proposal, ordered by removed UUID then strongest correlation.
    pub fn proposals(&self) -> Vec<&MergeProposal> {
        let mut keys: Vec<&String> = self.by_removed.keys().collect();
        keys.sort_unstable();
        keys.into_iter()
            .flat_map(|k| self.by_removed[k].iter())
            .collect()
    }
}

/// Sign signature of a probe vector: bit `i` set when probe `i` is at or above
/// the vector's own mean.
///
/// Centring on the neuron's own mean is what makes the bit comparable across
/// neurons on wildly different scales — a neuron oscillating around 40 and one
/// oscillating around 0.004 produce the same signature when they move together.
pub fn signature(values: &[f32]) -> u64 {
    // The mean is taken over exactly the probes the bits are read from: a
    // longer vector centred on records outside the signature would set bits
    // against a threshold no bit describes.
    let used = &values[..values.len().min(SIGNATURE_BITS as usize)];
    if used.is_empty() {
        return 0;
    }
    let mean = used.iter().map(|v| f64::from(*v)).sum::<f64>() / used.len() as f64;
    let mut bits = 0u64;
    for (i, v) in used.iter().enumerate() {
        if f64::from(*v) >= mean {
            bits |= 1u64 << i;
        }
    }
    bits
}

/// Pearson correlation and the least-squares fit of `removed` on `survivor`.
///
/// `None` when either vector is too short or does not move: a flat neuron has
/// no correlation to measure, and folding it into a survivor would be the
/// constant substitution ([`crate::substitute`]) wearing a merge's costume.
pub fn correlate(survivor: &[f32], removed: &[f32]) -> Option<(f64, LinearRelation)> {
    let n = survivor.len().min(removed.len());
    if n < MIN_PROBE_RECORDS {
        return None;
    }
    let scale = 1.0 / n as f64;
    let mean_s = survivor[..n].iter().map(|v| f64::from(*v)).sum::<f64>() * scale;
    let mean_r = removed[..n].iter().map(|v| f64::from(*v)).sum::<f64>() * scale;
    let (mut cov, mut var_s, mut var_r) = (0.0, 0.0, 0.0);
    for i in 0..n {
        let ds = f64::from(survivor[i]) - mean_s;
        let dr = f64::from(removed[i]) - mean_r;
        cov += ds * dr;
        var_s += ds * ds;
        var_r += dr * dr;
    }
    if var_s <= 0.0 || var_r <= 0.0 {
        return None;
    }
    let r = cov / (var_s * var_r).sqrt();
    let slope = cov / var_s;
    if !r.is_finite() || !slope.is_finite() {
        return None;
    }
    Some((
        r,
        LinearRelation {
            scale: slope,
            offset: mean_r - slope * mean_s,
        },
    ))
}

/// Band width for `neurons` signatures, at or above the configured floor.
///
/// Widening the band with the creature is what keeps discovery affordable. A
/// fixed width gives every unrelated pair a `2^-band_bits` chance of sharing a
/// bucket, so the comparisons a fixed width buys grow with the *square* of the
/// neuron count. Sizing the band so there are about as many buckets as neurons
/// holds the expected occupancy near one and the comparison count near linear.
///
/// Never more than half the signature, so there are always at least two bands
/// to give a near-duplicate pair more than one chance to meet.
pub fn effective_band_bits(configured: u32, neurons: usize) -> u32 {
    let needed = usize::BITS - neurons.saturating_sub(1).leading_zeros();
    configured.max(needed).clamp(1, SIGNATURE_BITS / 2)
}

/// One neuron's signature and the probe vector behind it.
struct Signed<'a> {
    uuid: &'a str,
    bits: u64,
    values: &'a [f32],
}

/// Discover likely near-duplicate hidden neurons in `stats` (Issue #109).
///
/// Sub-quadratic by construction: the signature pass is linear in the neurons,
/// the banding is linear in neurons times bands, and only bucket members —
/// bounded by [`DiscoveryConfig::max_bucket`] — are correlated pairwise.
pub fn discover(stats: &ActivationStats, cfg: DiscoveryConfig) -> MergeIndex {
    let started = Instant::now();
    let mut report = DiscoveryReport::default();
    let mut signed: Vec<Signed<'_>> = Vec::with_capacity(stats.probes.len());
    for probe in &stats.probes {
        let values = probe.values.as_slice();
        if values.len() < MIN_PROBE_RECORDS {
            report.unsignable += 1;
            continue;
        }
        let bits = signature(values);
        // An all-ones or all-zeros signature says the neuron never crossed its
        // own mean, which is a flat neuron rather than a behaviour to match.
        let full = if values.len() >= SIGNATURE_BITS as usize {
            u64::MAX
        } else {
            (1u64 << values.len()) - 1
        };
        if bits == 0 || bits == full {
            report.unsignable += 1;
            continue;
        }
        signed.push(Signed {
            uuid: probe.uuid.as_str(),
            bits,
            values,
        });
    }
    report.signed = signed.len();

    // Only bits a probe record actually filled may be banded. A shorter probe
    // vector leaves the high bits of every signature at zero, and banding over
    // that padding would put the whole creature in one bucket — the exact
    // quadratic sweep the banding exists to avoid.
    let usable_bits = signed
        .iter()
        .map(|s| s.values.len())
        .min()
        .unwrap_or(0)
        .min(SIGNATURE_BITS as usize) as u32;
    if usable_bits < MIN_PROBE_RECORDS as u32 {
        report.discovery_ms = started.elapsed().as_millis() as u64;
        return MergeIndex {
            by_removed: HashMap::new(),
            report,
        };
    }
    let band_bits = effective_band_bits(cfg.band_bits, signed.len()).min((usable_bits / 2).max(1));
    report.band_bits = band_bits;
    let bands = (usable_bits / band_bits).max(1);
    let mask = if band_bits >= SIGNATURE_BITS {
        u64::MAX
    } else {
        (1u64 << band_bits) - 1
    };
    // BTreeMap, not HashMap: the bucket walk below decides which pairs are
    // compared at all, so its order has to be the same on every run.
    let mut buckets: BTreeMap<(u32, u64), Vec<usize>> = BTreeMap::new();
    for (i, s) in signed.iter().enumerate() {
        for band in 0..bands {
            let value = (s.bits >> (band * band_bits)) & mask;
            // An anti-correlated neuron matches the complement of every band,
            // so both land under the same canonical key and one bucket walk
            // finds both directions of the relationship.
            let key = value.min(!value & mask);
            buckets.entry((band, key)).or_default().push(i);
        }
    }
    report.buckets = buckets.len();

    let mut pairs: BTreeSet<(usize, usize)> = BTreeSet::new();
    for members in buckets.values() {
        if members.len() < 2 {
            continue;
        }
        let kept = members.len().min(cfg.max_bucket);
        if kept < members.len() {
            report.truncated_buckets += 1;
            report.dropped_members += members.len() - kept;
        }
        for a in 0..kept {
            for b in (a + 1)..kept {
                pairs.insert((members[a].min(members[b]), members[a].max(members[b])));
            }
        }
    }
    report.compared_pairs = pairs.len();

    let mut by_removed: HashMap<String, Vec<MergeProposal>> = HashMap::new();
    for (a, b) in pairs {
        let Some((r, forward)) = correlate(signed[a].values, signed[b].values) else {
            continue;
        };
        if r.abs() < cfg.min_correlation {
            continue;
        }
        // Both survivor directions: which of two near-duplicates is the cheaper
        // one to lose is a structural question this pass cannot answer, so it
        // proposes each and lets the transform and the scorer settle it. The
        // pair is counted only once both fits exist, so a counted pair is one
        // that really did produce proposals.
        let Some((_, backward)) = correlate(signed[b].values, signed[a].values) else {
            continue;
        };
        report.correlated_pairs += 1;
        for (survivor, removed, relation) in [
            (signed[a].uuid, signed[b].uuid, forward),
            (signed[b].uuid, signed[a].uuid, backward),
        ] {
            by_removed
                .entry(removed.to_string())
                .or_default()
                .push(MergeProposal {
                    removed_uuid: removed.to_string(),
                    survivor_uuid: survivor.to_string(),
                    correlation: r,
                    relation,
                });
        }
    }
    for list in by_removed.values_mut() {
        // Strongest correlation first, UUID as the tie-break so a run is
        // reproducible whatever order the buckets filled in.
        list.sort_by(|x, y| {
            y.correlation
                .abs()
                .partial_cmp(&x.correlation.abs())
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| x.survivor_uuid.cmp(&y.survivor_uuid))
        });
        list.truncate(cfg.max_partners);
    }
    report.proposals = by_removed.values().map(Vec::len).sum();
    report.discovery_ms = started.elapsed().as_millis() as u64;
    MergeIndex { by_removed, report }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stats::{ActivationStats, DEFAULT_PROBE_RECORDS, NeuronProbes};

    fn stats_with(probes: Vec<(&str, Vec<f32>)>) -> ActivationStats {
        ActivationStats {
            probes: probes
                .into_iter()
                .map(|(uuid, values)| NeuronProbes {
                    uuid: uuid.into(),
                    values,
                })
                .collect(),
            ..ActivationStats::empty()
        }
    }

    /// A deterministic sawtooth: the probe vector every fixture is built from.
    fn wave(n: usize, phase: usize) -> Vec<f32> {
        (0..n)
            .map(|i| (((i + phase) % 17) as f32) - 8.0)
            .collect::<Vec<f32>>()
    }

    #[test]
    fn a_signature_is_the_sign_of_each_probe_against_its_own_mean() {
        assert_eq!(signature(&[]), 0, "nothing probed sets no bit");
        // mean 1.0: below, at, above.
        assert_eq!(signature(&[0.0, 1.0, 2.0]), 0b110);
        // A flat neuron never crosses its own mean, so every bit is set — the
        // all-ones case `discover` refuses to sign.
        assert_eq!(signature(&[3.0; 5]), 0b11111);
        // A vector longer than the signature is thresholded on the mean of the
        // records the bits actually come from, not on records outside them.
        let mut long = vec![0.0f32; SIGNATURE_BITS as usize];
        long.push(1_000.0);
        long[0] = 1.0;
        assert_eq!(signature(&long), 1, "the tail must not move the threshold");
    }

    #[test]
    fn correlation_needs_two_moving_vectors_of_usable_length() {
        let a = wave(32, 0);
        assert!(correlate(&a[..4], &a[..4]).is_none(), "too short to sign");
        assert!(
            correlate(&a, &[2.0f32; 32]).is_none(),
            "a flat partner has no correlation to measure"
        );
        let (r, rel) = correlate(&a, &a).expect("a vector correlates with itself");
        assert_eq!(r, 1.0);
        assert_eq!(rel.scale, 1.0);
        assert_eq!(rel.offset, 0.0);
        // The fit is directional: removed on survivor, not the other way round.
        let scaled: Vec<f32> = a.iter().map(|v| 4.0 * v - 1.0).collect();
        let (_, forward) = correlate(&a, &scaled).unwrap();
        let (_, backward) = correlate(&scaled, &a).unwrap();
        assert!((forward.scale - 4.0).abs() < 1e-5, "{forward:?}");
        assert!((backward.scale - 0.25).abs() < 1e-5, "{backward:?}");
    }

    #[test]
    fn a_deliberately_duplicated_neuron_is_discovered_with_an_exact_relation() {
        let base = wave(32, 0);
        let stats = stats_with(vec![
            ("h_a", base.clone()),
            ("h_b", base.clone()),
            ("h_far", wave(32, 5)),
        ]);
        let index = discover(&stats, DiscoveryConfig::default());
        let proposal = index
            .for_removed("h_b")
            .iter()
            .find(|p| p.survivor_uuid == "h_a")
            .expect("an exact duplicate must be proposed");
        assert!((proposal.correlation - 1.0).abs() < 1e-12);
        // Exactly 1 and exactly 0: the compensation for a true duplicate must
        // be the survivor's own weight, not a nearly-right multiple of it.
        assert_eq!(proposal.relation.scale, 1.0);
        assert_eq!(proposal.relation.offset, 0.0);
        // Both survivor directions are offered.
        assert!(
            index
                .for_removed("h_a")
                .iter()
                .any(|p| p.survivor_uuid == "h_b")
        );
        assert!(index.report().proposals >= 2);
    }

    #[test]
    fn a_scaled_and_shifted_duplicate_recovers_its_scale_and_offset() {
        let base = wave(32, 0);
        let scaled: Vec<f32> = base.iter().map(|v| 2.5 * v + 1.25).collect();
        let stats = stats_with(vec![("h_a", base), ("h_b", scaled)]);
        let index = discover(&stats, DiscoveryConfig::default());
        let p = index
            .for_removed("h_b")
            .iter()
            .find(|p| p.survivor_uuid == "h_a")
            .expect("a scaled duplicate is still a duplicate");
        assert!((p.correlation - 1.0).abs() < 1e-6, "{}", p.correlation);
        assert!((p.relation.scale - 2.5).abs() < 1e-5, "{:?}", p.relation);
        assert!((p.relation.offset - 1.25).abs() < 1e-4, "{:?}", p.relation);
    }

    #[test]
    fn an_anti_correlated_pair_shares_a_bucket_and_is_proposed() {
        let base = wave(32, 0);
        let flipped: Vec<f32> = base.iter().map(|v| -v).collect();
        let stats = stats_with(vec![("h_a", base), ("h_b", flipped)]);
        let index = discover(&stats, DiscoveryConfig::default());
        let p = index
            .for_removed("h_b")
            .iter()
            .find(|p| p.survivor_uuid == "h_a")
            .expect("the complement bucket key must catch an inverted neuron");
        assert!((p.correlation + 1.0).abs() < 1e-9, "{}", p.correlation);
        assert!((p.relation.scale + 1.0).abs() < 1e-9, "{:?}", p.relation);
    }

    #[test]
    fn unrelated_neurons_are_not_proposed() {
        // Two coprime periods over the same probe records: they cross their
        // means at genuinely different times.
        let a: Vec<f32> = (0..48).map(|i| ((i % 7) as f32) - 3.0).collect();
        let b: Vec<f32> = (0..48).map(|i| ((i % 11) as f32) - 5.0).collect();
        let stats = stats_with(vec![("h_a", a), ("h_b", b)]);
        let index = discover(&stats, DiscoveryConfig::default());
        assert!(index.is_empty(), "{:?}", index.proposals());
        assert_eq!(index.report().correlated_pairs, 0);
    }

    #[test]
    fn a_flat_or_short_probe_vector_is_never_signed() {
        let stats = stats_with(vec![
            ("h_flat", vec![1.0f32; 32]),
            ("h_short", wave(4, 0)),
            ("h_none", Vec::new()),
        ]);
        let index = discover(&stats, DiscoveryConfig::default());
        assert!(index.is_empty());
        assert_eq!(index.report().signed, 0);
        assert_eq!(index.report().unsignable, 3);
    }

    #[test]
    fn the_threshold_only_widens_or_narrows_what_is_proposed() {
        let base = wave(32, 0);
        let noisy: Vec<f32> = base
            .iter()
            .enumerate()
            .map(|(i, v)| v + if i % 3 == 0 { 2.0 } else { -1.0 })
            .collect();
        let stats = stats_with(vec![("h_a", base), ("h_b", noisy)]);
        let strict = discover(&stats, DiscoveryConfig::with_min_correlation(0.999));
        let loose = discover(&stats, DiscoveryConfig::with_min_correlation(0.5));
        assert!(strict.is_empty(), "{:?}", strict.proposals());
        assert!(!loose.is_empty(), "a loose threshold must still propose");
    }

    #[test]
    fn discovery_is_deterministic_for_the_same_statistics() {
        let mut probes = Vec::new();
        for i in 0..60 {
            probes.push((i, wave(32, i % 6)));
        }
        let owned: Vec<(String, Vec<f32>)> = probes
            .into_iter()
            .map(|(i, v)| (format!("h{i}"), v))
            .collect();
        let stats = stats_with(owned.iter().map(|(u, v)| (u.as_str(), v.clone())).collect());
        let a = discover(&stats, DiscoveryConfig::default());
        let b = discover(&stats, DiscoveryConfig::default());
        assert_eq!(a.proposals(), b.proposals());
        assert_eq!(a.report().proposals, b.report().proposals);
        assert!(a.report().proposals > 0);
    }

    /// No silent caps: a bucket the sweep declined to finish is counted.
    #[test]
    fn an_over_full_bucket_is_truncated_and_reported() {
        let base = wave(32, 0);
        let owned: Vec<(String, Vec<f32>)> = (0..40)
            .map(|i| (format!("h{i:02}"), base.clone()))
            .collect();
        let stats = stats_with(owned.iter().map(|(u, v)| (u.as_str(), v.clone())).collect());
        let cfg = DiscoveryConfig {
            max_bucket: 4,
            ..DiscoveryConfig::default()
        };
        let index = discover(&stats, cfg);
        let report = index.report();
        assert!(report.truncated_buckets > 0, "{report:?}");
        assert!(report.dropped_members > 0, "{report:?}");
        // 4 members per bucket is 6 pairs, however many bands agree.
        assert!(report.compared_pairs <= 6, "{report:?}");
    }

    #[test]
    fn a_bad_configuration_names_the_flag() {
        for (cfg, flag) in [
            (
                DiscoveryConfig::with_min_correlation(1.5),
                "--merge-correlation",
            ),
            (
                DiscoveryConfig {
                    band_bits: 0,
                    ..DiscoveryConfig::default()
                },
                "--merge-band-bits",
            ),
            (
                DiscoveryConfig {
                    max_bucket: 1,
                    ..DiscoveryConfig::default()
                },
                "--merge-max-bucket",
            ),
            (
                DiscoveryConfig {
                    max_partners: 0,
                    ..DiscoveryConfig::default()
                },
                "--merge-max-partners",
            ),
        ] {
            let err = cfg.validate().unwrap_err();
            assert!(err.contains(flag), "{err}");
        }
        DiscoveryConfig::default().validate().unwrap();
    }

    /// A distinct pseudo-random probe vector per neuron, from a fixed seed.
    fn distinct(index: usize, len: usize) -> Vec<f32> {
        let mut state = (index as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
        (0..len)
            .map(|_| {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                ((state >> 40) as f32) / 16_777_216.0 - 0.5
            })
            .collect()
    }

    fn distinct_stats(n: usize) -> ActivationStats {
        let owned: Vec<(String, Vec<f32>)> = (0..n)
            .map(|i| (format!("h{i:05}"), distinct(i, DEFAULT_PROBE_RECORDS)))
            .collect();
        stats_with(owned.iter().map(|(u, v)| (u.as_str(), v.clone())).collect())
    }

    /// Probing fewer records than the signature holds must not put the whole
    /// creature in one bucket: banding over the unfilled high bits would make
    /// every neuron agree on them and reinstate the quadratic sweep.
    #[test]
    fn a_short_probe_vector_is_banded_over_the_bits_it_filled() {
        let owned: Vec<(String, Vec<f32>)> = (0..600)
            .map(|i| (format!("h{i:04}"), distinct(i, 16)))
            .collect();
        let stats = stats_with(owned.iter().map(|(u, v)| (u.as_str(), v.clone())).collect());
        let report = discover(&stats, DiscoveryConfig::default()).report();
        assert!(report.band_bits <= 8, "{report:?}");
        assert_eq!(
            report.truncated_buckets, 0,
            "no bucket may hold them all: {report:?}"
        );
    }

    /// The band widens with the creature, so a bucket holds about one neuron
    /// however many there are — that is what keeps the pair count linear.
    #[test]
    fn the_band_widens_with_the_creature() {
        assert_eq!(effective_band_bits(8, 16), 8, "the floor holds when small");
        assert_eq!(effective_band_bits(8, 4_000), 12);
        assert_eq!(effective_band_bits(8, 60_000), 16);
        assert_eq!(
            effective_band_bits(8, usize::MAX),
            SIGNATURE_BITS / 2,
            "there must always be at least two bands"
        );
    }

    /// Issue #109 requires discovery that scales to several thousand hidden
    /// neurons. A ratio, never a wall-clock budget — the same work is timed at
    /// one size and four times that size on the same machine, so a loaded
    /// runner slows both readings and the test still holds.
    #[test]
    fn discovery_costs_the_creature_not_its_square() {
        fn seconds(n: usize) -> f64 {
            let stats = distinct_stats(n);
            let started = Instant::now();
            let index = discover(&stats, DiscoveryConfig::default());
            assert_eq!(index.report().signed, n);
            started.elapsed().as_secs_f64()
        }
        let before = seconds(1_500);
        let large = seconds(6_000);
        let after = seconds(1_500);
        let small = before.max(after).max(1e-9);
        let growth = large / small;
        assert!(
            growth < 8.0,
            "four times the creature must not cost sixteen times the discovery: \
             {growth:.1}x ({small:.4}s → {large:.4}s)"
        );
    }

    /// The needle stays findable in the haystack: one deliberately duplicated
    /// pair hidden among six thousand unrelated neurons is still proposed.
    #[test]
    fn a_duplicate_is_found_among_several_thousand_unrelated_neurons() {
        let mut owned: Vec<(String, Vec<f32>)> = (0..6_000)
            .map(|i| (format!("h{i:05}"), distinct(i, DEFAULT_PROBE_RECORDS)))
            .collect();
        let twin = owned[1_234].1.clone();
        owned.push(("h_twin".into(), twin));
        let stats = stats_with(owned.iter().map(|(u, v)| (u.as_str(), v.clone())).collect());
        let index = discover(&stats, DiscoveryConfig::default());
        let p = index
            .for_removed("h_twin")
            .iter()
            .find(|p| p.survivor_uuid == "h01234")
            .unwrap_or_else(|| panic!("duplicate lost: {:?}", index.report()));
        assert!((p.correlation - 1.0).abs() < 1e-12);
    }
}
