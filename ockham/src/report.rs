//! Summarise `experiments.jsonl` journals (Issue #10, extended by #11).
//!
//! Distinguishes **local cumulative Ockham improvement** from **population
//! headroom** when re-entry records exist. Tiny accepted steps are kept in the
//! cumulative trajectory rather than dismissed.
//!
//! Issue #11 adds the measures needed to compare one named ordering against
//! the random control on equal terms: time to the first authoritative local
//! winner, candidates screened before it, accepted-win size distribution,
//! scorer calls consumed and growth-cost reduction. Cumulative gain — not the
//! single largest cut — remains the headline.

use std::path::Path;

use serde::Serialize;

use crate::ablation::growth_units;
use crate::blocked::{BlockedBreakdown, BlockedReason};
use crate::coverage::{Coverage, Winners};
use crate::journal::Event;
use crate::ordering::Ordering;

/// Aggregate view of one or more journals.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Report {
    /// Journals consumed.
    pub journals: Vec<String>,
    /// Named ordering from the first start record.
    pub ordering: Option<Ordering>,
    /// Random exploration quota from the first start record.
    pub ordering_random_quota: Option<f64>,
    /// Opening authoritative score (first start record).
    pub opening_score: Option<f64>,
    /// Final authoritative score (last stop record).
    pub final_score: Option<f64>,
    /// `final - opening`.
    pub cumulative_delta: Option<f64>,
    /// Authoritative local accepts.
    pub accepts: u64,
    /// Sweep batches.
    pub experiments: u64,
    /// Full-cohort accepts (from `full` records).
    pub full_accepts: u64,
    /// Full-cohort rejects.
    pub full_rejects: u64,
    /// Sampled screen scorer calls consumed.
    pub screen_calls: u64,
    /// Full-corpus scorer cohort calls consumed.
    pub full_calls: u64,
    /// Screen-coverage records filed across the run (Issue #36).
    ///
    /// Since #93 this counts every neuron the sweep **visited** and filed a
    /// record for, not only the candidates the scorer screened — so a series
    /// spanning that release steps up where the definition widened.
    pub screened: u64,
    /// Sweeps rebuilt after visiting every hidden neuron (Issue #77).
    ///
    /// A restart says a run screened the creature end to end and rolled into
    /// re-screening the stalest neurons — the opposite of the idle spin it
    /// replaced.
    pub sweep_restarts: u64,
    /// Screening batches run after a replay accept ended the search (#91).
    ///
    /// The runs that carry these are the `replay-accepts` runs:
    /// without the figure, a run that screened four hundred neurons after its
    /// cut is indistinguishable in the report from one that screened none.
    pub coverage_tail_batches: u64,
    /// Hidden neurons on the incumbent at the last coverage record (Issue #37).
    pub hidden: Option<usize>,
    /// Hidden neurons carrying tags, at that same record (Issue #40).
    pub tagged: Option<usize>,
    /// The coverage denominator — every hidden neuron, tagged included (#74).
    ///
    /// Derived from `hidden`, so a journal written before #74 is reported on
    /// the new denominator rather than replaying the old overstatement.
    pub checkable: Option<usize>,
    /// Hidden UUIDs screened at least once, at that same record.
    pub checked: Option<usize>,
    /// Checked UUIDs the razor could propose no cut for (Issue #93).
    ///
    /// A subset of `checked`: the sweep visited them and the structure — an
    /// aggregate squash downstream, a typed synapse — left nothing to score.
    pub blocked: Option<usize>,
    /// `blocked` split by reason code, at that same record (Issue #103).
    ///
    /// The work list: the counts sum to `blocked`, and the largest of them is
    /// the category a new proposal path would pay for. `None` on a journal with
    /// no coverage record.
    pub blocked_by_reason: Option<BlockedBreakdown>,
    /// The largest blocked category's code, at that same record (Issue #103).
    ///
    /// Derived from `blocked_by_reason` so the report can never disagree with
    /// it; `None` when nothing is blocked.
    pub dominant_blocked_reason: Option<String>,
    /// Blocked reasons per screening epoch, in the order the journals name them
    /// (Issue #103).
    ///
    /// The coverage figures above are the **latest** snapshot, so on their own
    /// they cannot answer "was this category always this large?". Every epoch
    /// the journals carry keeps its freshest breakdown here, which is what makes
    /// historical blocked reasons inspectable across epochs.
    pub blocked_epochs: Vec<EpochBlocked>,
    /// Hidden UUIDs still never screened, at that same record (Issue #40).
    pub unchecked: Option<usize>,
    /// Hidden neurons cut by the run that wrote that record (Issue #40).
    pub cut: Option<usize>,
    /// `checked / checkable * 100` at that same record, for that epoch alone.
    pub coverage_percent: Option<f64>,
    /// Corpus identity the coverage figures were measured against (Issue #102).
    ///
    /// Every coverage figure above is a statement about this corpus and no
    /// other: `coverage_percent: 100.0` means the sweep finished this epoch,
    /// not that Ockham is done. `None` on a journal written before #100, whose
    /// coverage records name no epoch.
    pub corpus_identity: Option<String>,
    /// Whether the sweep had reached every hidden neuron of that epoch.
    ///
    /// Derived from the same record as the percentage, so `report` can never
    /// disagree with `coverage.txt` about what a finished sweep is. `None` when
    /// the journal carries no coverage record.
    pub sweep_complete: Option<bool>,
    /// What the run tried, kept and rejected (Issue #59); `None` on older
    /// journals, which carry no winners record.
    pub winners: Option<Winners>,
    /// Rolling full-corpus cost estimate from the last budget record (Issue #58).
    pub est_ms_per_creature: Option<f64>,
    /// Cohort entries dropped over budget across the journals.
    pub budget_dropped: usize,
    /// Full cohorts that had to be trimmed to fit the wall clock.
    pub budget_trims: u64,
    /// Milliseconds from the loop start to the first authoritative local win.
    pub first_win_ms: Option<u64>,
    /// Candidates screened before the first authoritative local win.
    pub candidates_before_first_win: Option<u64>,
    /// Neurons cut by each accepted winner, in acceptance order.
    pub accepted_cut_sizes: Vec<usize>,
    /// Total neurons cut across every accepted winner.
    pub accepted_cuts: usize,
    /// Wall-clock milliseconds spent in the optimisation loop.
    pub elapsed_ms: Option<u64>,
    /// Authoritative local accepts per hour of loop wall-clock.
    pub accepts_per_hour: Option<f64>,
    /// Opening `hidden + synapses / 10` growth units.
    pub opening_growth_units: Option<f64>,
    /// Final `hidden + synapses / 10` growth units.
    pub final_growth_units: Option<f64>,
    /// `opening - final` growth units; positive means structure was removed.
    pub growth_units_saved: Option<f64>,
    /// Hidden neurons cut per hour of loop wall-clock (Issue #106).
    pub cuts_per_hour: Option<f64>,
    /// Growth units removed per hour of loop wall-clock (Issue #106).
    pub growth_units_saved_per_hour: Option<f64>,
    /// Accepted winners carrying an estimated-versus-actual record (#106).
    ///
    /// Beside the two totals so a missing ratio is never ambiguous: `0` says no
    /// accept was recorded, and a non-zero count with no ratio says the
    /// accepted cuts were predicted to save nothing.
    pub cascade_accepts: u64,
    /// Growth units the cascade dry-run predicted across accepted cuts (#106).
    pub cascade_estimated_growth_units: Option<f64>,
    /// Growth units those accepted cuts actually removed (Issue #106).
    pub cascade_actual_growth_units: Option<f64>,
    /// `actual / estimated` across accepted cuts; `1.0` is a perfect estimate.
    ///
    /// Below `1.0` the dry-run over-promised — an accepted winner kept
    /// structure the topology said would go, which a constant substitution or a
    /// bundle legitimately does. Above `1.0` it under-promised.
    pub cascade_estimate_ratio: Option<f64>,
    /// Last stop reason.
    pub stop_reason: Option<String>,
    /// Effective seed from the first start record.
    pub seed: Option<u64>,
}

/// One screening epoch's blocked population (Issue #103).
///
/// The freshest coverage snapshot the journals carry for that corpus identity:
/// a run reports the epoch in hand, and the series of these reports how the
/// blocked categories moved as the corpus was extended.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct EpochBlocked {
    /// Corpus identity the figures were measured against; `None` pre-#100.
    pub corpus_identity: Option<String>,
    /// Blocked UUIDs in that epoch.
    pub blocked: usize,
    /// That total split by reason code.
    pub blocked_by_reason: BlockedBreakdown,
    /// The same split rendered with each category's share of the total.
    ///
    /// `aggregate-squash 380 (92.2%) · missing-activation 32 (7.8%)`, commonest
    /// first; empty when the epoch blocked nothing. Counts *and* percentages
    /// per epoch is what the issue asks a reader for, and rendering it here
    /// keeps every surface agreeing on one calculation.
    pub reasons: String,
}

/// Read JSONL journals and fold them into a [`Report`].
pub fn summarise(paths: &[impl AsRef<Path>]) -> Result<Report, String> {
    let mut report = Report {
        journals: Vec::new(),
        ordering: None,
        ordering_random_quota: None,
        opening_score: None,
        final_score: None,
        cumulative_delta: None,
        accepts: 0,
        experiments: 0,
        full_accepts: 0,
        full_rejects: 0,
        screen_calls: 0,
        full_calls: 0,
        screened: 0,
        sweep_restarts: 0,
        coverage_tail_batches: 0,
        hidden: None,
        tagged: None,
        checkable: None,
        checked: None,
        blocked: None,
        blocked_by_reason: None,
        dominant_blocked_reason: None,
        blocked_epochs: Vec::new(),
        unchecked: None,
        cut: None,
        coverage_percent: None,
        corpus_identity: None,
        sweep_complete: None,
        winners: None,
        est_ms_per_creature: None,
        budget_dropped: 0,
        budget_trims: 0,
        first_win_ms: None,
        candidates_before_first_win: None,
        accepted_cut_sizes: Vec::new(),
        accepted_cuts: 0,
        elapsed_ms: None,
        accepts_per_hour: None,
        opening_growth_units: None,
        final_growth_units: None,
        growth_units_saved: None,
        cuts_per_hour: None,
        growth_units_saved_per_hour: None,
        cascade_accepts: 0,
        cascade_estimated_growth_units: None,
        cascade_actual_growth_units: None,
        cascade_estimate_ratio: None,
        stop_reason: None,
        seed: None,
    };
    let mut cascade_estimated = 0.0f64;
    let mut cascade_actual = 0.0f64;
    let mut candidates_seen = 0u64;
    for path in paths {
        let path = path.as_ref();
        report.journals.push(path.display().to_string());
        let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
        for (i, line) in text.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let event: Event = serde_json::from_str(line)
                .map_err(|e| format!("{}:{}: {e}", path.display(), i + 1))?;
            match event {
                Event::Start {
                    seed,
                    ordering,
                    ordering_random_quota,
                    hidden,
                    synapses,
                    opening_score,
                    ..
                } => {
                    if report.seed.is_none() {
                        report.seed = Some(seed);
                        report.ordering = Some(ordering);
                        report.ordering_random_quota = Some(ordering_random_quota);
                    }
                    if report.opening_score.is_none() {
                        report.opening_score = Some(opening_score);
                    }
                    if report.opening_growth_units.is_none() {
                        report.opening_growth_units = Some(growth_units(hidden, synapses));
                    }
                }
                Event::Batch { candidates, .. } => {
                    report.experiments += 1;
                    candidates_seen += candidates as u64;
                }
                Event::SweepRestart { .. } => report.sweep_restarts += 1,
                Event::Cascade {
                    estimated_growth_units,
                    actual_growth_units,
                    ..
                } => {
                    cascade_estimated += estimated_growth_units;
                    cascade_actual += actual_growth_units;
                    report.cascade_accepts += 1;
                }
                Event::CoverageTail { batches, .. } => report.coverage_tail_batches += batches,
                Event::Screen { .. } => report.screen_calls += 1,
                Event::Screened { screened, .. } => report.screened += screened as u64,
                Event::Coverage {
                    hidden,
                    tagged,
                    checked,
                    blocked,
                    blocked_by_reason,
                    cut,
                    corpus_identity,
                    ..
                } => {
                    // Coverage is a snapshot of one incumbent, not a total:
                    // the last record read is the current state. The percent
                    // comes from `Coverage` so the report can never disagree
                    // with the tag or the commit description.
                    //
                    // `checkable` is derived from `hidden` rather than read
                    // back (#74): a journal written before #74 carries the old
                    // `hidden - tagged` denominator, and replaying it would
                    // report the very overstatement #74 removed. Deriving is
                    // exact, not a guess — the two are the same number now.
                    let cov = Coverage {
                        hidden,
                        tagged,
                        checkable: hidden,
                        checked,
                        blocked,
                        // A journal written before #103 carries the total and
                        // no reasons, so the difference is filed as
                        // `unrecorded` rather than left as a breakdown that
                        // silently fails to account for the neurons beside it.
                        blocked_by_reason: account_for_every_blocked(blocked, blocked_by_reason),
                        cut,
                    };
                    report.hidden = Some(cov.hidden);
                    report.tagged = Some(cov.tagged);
                    report.checkable = Some(cov.checkable);
                    report.checked = Some(cov.checked);
                    report.blocked = Some(cov.blocked);
                    report.blocked_by_reason = Some(cov.blocked_by_reason);
                    report.dominant_blocked_reason = cov
                        .blocked_by_reason
                        .dominant()
                        .map(|(reason, _)| reason.code().to_string());
                    // One entry per epoch, holding its freshest snapshot: a
                    // later run under the same corpus replaces the figures
                    // rather than appending a second row for the same epoch.
                    record_epoch_blocked(&mut report.blocked_epochs, &corpus_identity, &cov);
                    report.unchecked = Some(cov.unchecked());
                    report.cut = Some(cov.cut);
                    report.coverage_percent = Some(cov.percent());
                    // The epoch belongs to the snapshot, so it moves with it
                    // (#102): a later record from a fresh corpus replaces both
                    // the figures and the identity they were measured against.
                    report.corpus_identity = corpus_identity;
                    report.sweep_complete = Some(cov.sweep_complete());
                }
                Event::Budget {
                    est_ms_per_creature,
                    dropped_individuals,
                    dropped_bundles,
                    ..
                } => {
                    // The estimate is a running state, so the last record read
                    // is the current one; the drops are a total.
                    if est_ms_per_creature > 0.0 {
                        report.est_ms_per_creature = Some(est_ms_per_creature);
                    }
                    let dropped = dropped_individuals + dropped_bundles;
                    report.budget_dropped += dropped;
                    if dropped > 0 {
                        report.budget_trims += 1;
                    }
                }
                Event::Winners {
                    screened,
                    confirmed,
                    applied,
                    carried,
                    plans,
                    skipped,
                    best_cuts,
                    best_delta,
                    dropped,
                    est_ms_per_creature,
                } => {
                    report.winners = Some(Winners {
                        screened,
                        confirmed,
                        applied,
                        carried,
                        plans,
                        skipped,
                        best_cuts,
                        best_delta,
                        dropped,
                        est_ms_per_creature,
                    });
                }
                Event::Full {
                    accepted,
                    cuts,
                    elapsed_ms,
                    ..
                } => {
                    report.full_calls += 1;
                    if accepted {
                        report.full_accepts += 1;
                        report.accepted_cut_sizes.push(cuts);
                        report.accepted_cuts += cuts;
                        if report.first_win_ms.is_none() {
                            report.first_win_ms = Some(elapsed_ms);
                            report.candidates_before_first_win = Some(candidates_seen);
                        }
                    } else {
                        report.full_rejects += 1;
                    }
                }
                Event::Stop {
                    reason,
                    accepts,
                    experiments,
                    final_score,
                    cumulative_delta,
                    final_hidden,
                    final_synapses,
                    elapsed_ms,
                    // Per-run progress (Issue #77) is a property of one run
                    // and this report sums many, so a total would be
                    // meaningless: the same uuid is "newly screened" in at
                    // most one of the journals being folded together, and the
                    // figure belongs beside that run's own coverage block.
                    newly_screened: _,
                } => {
                    report.stop_reason = Some(reason);
                    report.accepts = accepts;
                    if experiments > report.experiments {
                        report.experiments = experiments;
                    }
                    report.final_score = Some(final_score);
                    report.cumulative_delta = Some(cumulative_delta);
                    report.final_growth_units = Some(growth_units(final_hidden, final_synapses));
                    report.elapsed_ms = Some(elapsed_ms);
                }
            }
        }
    }
    if report.cumulative_delta.is_none()
        && let (Some(open), Some(final_score)) = (report.opening_score, report.final_score)
    {
        report.cumulative_delta = Some(final_score - open);
    }
    if let (Some(open), Some(fin)) = (report.opening_growth_units, report.final_growth_units) {
        report.growth_units_saved = Some(open - fin);
    }
    if let Some(ms) = report.elapsed_ms.filter(|ms| *ms > 0) {
        let per_hour = 3_600_000.0 / ms as f64;
        report.accepts_per_hour = Some(report.accepts as f64 * per_hour);
        // The two economics an ordering is judged on (Issue #106): confirmed
        // cuts bought per hour, and how much structure they took with them.
        report.cuts_per_hour = Some(report.accepted_cuts as f64 * per_hour);
        report.growth_units_saved_per_hour = report.growth_units_saved.map(|g| g * per_hour);
    }
    if report.cascade_accepts > 0 {
        report.cascade_estimated_growth_units = Some(cascade_estimated);
        report.cascade_actual_growth_units = Some(cascade_actual);
        if cascade_estimated > 0.0 {
            report.cascade_estimate_ratio = Some(cascade_actual / cascade_estimated);
        }
    }
    Ok(report)
}

/// Make the breakdown account for every blocked uuid in the record.
///
/// The invariant every surface states is that the reason counts sum to
/// `blocked`. A pre-#103 journal carries the total with no reasons at all, and
/// a mixed-version fleet can file a total whose reasons this binary read short;
/// the shortfall is [`BlockedReason::Unrecorded`], which is exactly what that
/// category exists to say. A breakdown that already exceeds the total is left
/// alone — inventing a smaller number would hide the disagreement.
fn account_for_every_blocked(blocked: usize, mut reasons: BlockedBreakdown) -> BlockedBreakdown {
    for _ in 0..blocked.saturating_sub(reasons.total()) {
        reasons.add(BlockedReason::Unrecorded);
    }
    reasons
}

/// Fold one coverage snapshot into the per-epoch blocked history.
///
/// Keyed on the corpus identity and ordered by first appearance, so the series
/// reads oldest epoch first however many journals were folded together.
fn record_epoch_blocked(
    epochs: &mut Vec<EpochBlocked>,
    corpus_identity: &Option<String>,
    cov: &Coverage,
) {
    let entry = EpochBlocked {
        corpus_identity: corpus_identity.clone(),
        blocked: cov.blocked,
        blocked_by_reason: cov.blocked_by_reason,
        reasons: cov.blocked_by_reason.summary().unwrap_or_default(),
    };
    match epochs
        .iter_mut()
        .find(|e| e.corpus_identity == *corpus_identity)
    {
        Some(held) => *held = entry,
        None => epochs.push(entry),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::journal::{self, Event};

    fn start(ordering: Ordering) -> Event {
        Event::Start {
            seed: 7,
            ordering,
            ordering_random_quota: 0.25,
            permutation_identity: "x".into(),
            unchecked_first: false,
            old_corpus_first: 0,
            hidden: 3,
            synapses: 10,
            opening_score: 0.50,
        }
    }

    fn full(accepted: bool, score: f64, cuts: usize, elapsed_ms: u64) -> Event {
        Event::Full {
            individuals: 1,
            bundles: 0,
            accepted,
            score: accepted.then_some(score),
            delta: accepted.then_some(0.000002),
            cuts,
            elapsed_ms,
        }
    }

    #[test]
    fn report_compounds_tiny_accepts_rather_than_only_the_final_score() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("experiments.jsonl");
        journal::append(&path, &start(Ordering::Random)).unwrap();
        journal::append(&path, &full(true, 0.500002, 1, 1_000)).unwrap();
        journal::append(&path, &full(true, 0.500004, 1, 2_000)).unwrap();
        journal::append(
            &path,
            &Event::Stop {
                reason: "timeout".into(),
                accepts: 2,
                experiments: 4,
                final_score: 0.500004,
                cumulative_delta: 0.000004,
                final_hidden: 1,
                final_synapses: 8,
                elapsed_ms: 3_600_000,
                newly_screened: 40,
            },
        )
        .unwrap();
        let report = summarise(&[&path]).unwrap();
        assert_eq!(report.accepts, 2);
        assert_eq!(report.full_accepts, 2);
        assert!(report.cumulative_delta.unwrap() > 0.0);
        assert_eq!(report.opening_score, Some(0.50));
        assert_eq!(report.final_score, Some(0.500004));
        assert_eq!(report.seed, Some(7));
    }

    #[test]
    fn report_names_the_ordering_and_its_random_quota() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("experiments.jsonl");
        journal::append(&path, &start(Ordering::LowVariance)).unwrap();
        let report = summarise(&[&path]).unwrap();
        assert_eq!(report.ordering, Some(Ordering::LowVariance));
        assert_eq!(report.ordering_random_quota, Some(0.25));
        let json = serde_json::to_string(&report).unwrap();
        assert!(json.contains("\"ordering\":\"low-variance\""), "{json}");
    }

    /// Issue #77: a sweep restart is fleet news — a creature screened end to
    /// end and recycled — so the report counts it rather than dropping it.
    #[test]
    fn report_counts_the_sweeps_a_run_rebuilt() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("experiments.jsonl");
        journal::append(&path, &start(Ordering::Random)).unwrap();
        assert_eq!(summarise(&[&path]).unwrap().sweep_restarts, 0);
        for restarts in 1..=2u64 {
            journal::append(
                &path,
                &Event::SweepRestart {
                    restarts,
                    hidden: 40,
                    newly_screened: 100,
                },
            )
            .unwrap();
        }
        let report = summarise(&[&path]).unwrap();
        assert_eq!(report.sweep_restarts, 2);
        let json = serde_json::to_string(&report).unwrap();
        assert!(json.contains("\"sweepRestarts\":2"), "{json}");
    }

    #[test]
    fn report_measures_discovery_economics_not_just_the_largest_cut() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("experiments.jsonl");
        journal::append(&path, &start(Ordering::LowMeanAbs)).unwrap();
        for batch in 0..2u64 {
            journal::append(
                &path,
                &Event::Batch {
                    batch,
                    candidates: 40,
                    skipped: 2,
                    remaining: 100,
                },
            )
            .unwrap();
            journal::append(
                &path,
                &Event::Screen {
                    winners: 1,
                    losers: 39,
                    ms: 500,
                },
            )
            .unwrap();
            journal::append(&path, &full(false, 0.0, 0, 1_000 * (batch + 1))).unwrap();
        }
        // The third batch finally lands a tiny two-neuron bundle.
        journal::append(
            &path,
            &Event::Batch {
                batch: 2,
                candidates: 40,
                skipped: 0,
                remaining: 20,
            },
        )
        .unwrap();
        journal::append(
            &path,
            &Event::Screen {
                winners: 2,
                losers: 38,
                ms: 500,
            },
        )
        .unwrap();
        journal::append(&path, &full(true, 0.5001, 2, 9_000)).unwrap();
        journal::append(&path, &full(true, 0.5002, 1, 12_000)).unwrap();
        journal::append(
            &path,
            &Event::Stop {
                reason: "timeout".into(),
                accepts: 2,
                experiments: 3,
                final_score: 0.5002,
                cumulative_delta: 0.0002,
                final_hidden: 0,
                final_synapses: 5,
                elapsed_ms: 1_800_000,
                newly_screened: 120,
            },
        )
        .unwrap();

        let report = summarise(&[&path]).unwrap();
        assert_eq!(report.screen_calls, 3);
        assert_eq!(report.full_calls, 4);
        assert_eq!(report.first_win_ms, Some(9_000));
        // 80 candidates screened across the two fruitless batches, then 40 more
        // in the batch that produced the win.
        assert_eq!(report.candidates_before_first_win, Some(120));
        assert_eq!(report.accepted_cut_sizes, vec![2, 1]);
        assert_eq!(report.accepted_cuts, 3);
        assert_eq!(report.elapsed_ms, Some(1_800_000));
        assert_eq!(report.accepts_per_hour, Some(4.0));
        assert_eq!(report.opening_growth_units, Some(growth_units(3, 10)));
        assert_eq!(report.final_growth_units, Some(growth_units(0, 5)));
        assert_eq!(report.growth_units_saved, Some(3.5));
        // Half an hour of loop: three cuts and 3.5 growth units become six
        // cuts and seven growth units an hour (Issue #106).
        assert_eq!(report.cuts_per_hour, Some(6.0));
        assert_eq!(report.growth_units_saved_per_hour, Some(7.0));
    }

    #[test]
    fn accepted_cuts_report_the_estimated_cascade_beside_the_actual_saving() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("experiments.jsonl");
        journal::append(&path, &start(Ordering::CascadeSaving)).unwrap();
        journal::append(
            &path,
            &Event::Cascade {
                kind: "individual".into(),
                cuts: 1,
                estimated_hidden: 3,
                estimated_synapses: 4,
                estimated_growth_units: 3.4,
                actual_hidden: 3,
                actual_synapses: 4,
                actual_growth_units: 3.4,
            },
        )
        .unwrap();
        journal::append(
            &path,
            &Event::Cascade {
                kind: "individual".into(),
                cuts: 1,
                estimated_hidden: 2,
                estimated_synapses: 6,
                estimated_growth_units: 2.6,
                // A constant substitution kept the edge the topology said would
                // go, so the accept removed less than the dry-run promised.
                actual_hidden: 1,
                actual_synapses: 4,
                actual_growth_units: 1.4,
            },
        )
        .unwrap();
        let report = summarise(&[&path]).unwrap();
        assert_eq!(report.cascade_accepts, 2);
        assert_eq!(report.cascade_estimated_growth_units, Some(6.0));
        assert_eq!(report.cascade_actual_growth_units, Some(4.8));
        let ratio = report.cascade_estimate_ratio.expect("two accepted cuts");
        assert!((ratio - 0.8).abs() < 1e-9, "{ratio}");
    }

    #[test]
    fn a_journal_with_no_accepted_cuts_reports_no_cascade_comparison() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("experiments.jsonl");
        journal::append(&path, &start(Ordering::Random)).unwrap();
        let report = summarise(&[&path]).unwrap();
        assert_eq!(report.cascade_accepts, 0);
        assert_eq!(report.cascade_estimated_growth_units, None);
        assert_eq!(report.cascade_actual_growth_units, None);
        assert_eq!(report.cascade_estimate_ratio, None);
    }

    /// A cut the transform refuses is predicted to save nothing, so an accept
    /// that came from another path — a collapse, a substitution — carries a
    /// zero estimate. The count still says an accept was recorded; only the
    /// ratio is absent, and the two together say which case this is.
    #[test]
    fn an_accept_predicted_to_save_nothing_is_counted_without_a_ratio() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("experiments.jsonl");
        journal::append(&path, &start(Ordering::CascadeSaving)).unwrap();
        journal::append(
            &path,
            &Event::Cascade {
                kind: "individual".into(),
                cuts: 1,
                estimated_hidden: 0,
                estimated_synapses: 0,
                estimated_growth_units: 0.0,
                actual_hidden: 1,
                actual_synapses: 0,
                actual_growth_units: 1.0,
            },
        )
        .unwrap();
        let report = summarise(&[&path]).unwrap();
        assert_eq!(report.cascade_accepts, 1);
        assert_eq!(report.cascade_estimated_growth_units, Some(0.0));
        assert_eq!(report.cascade_actual_growth_units, Some(1.0));
        assert_eq!(report.cascade_estimate_ratio, None);
    }

    #[test]
    fn screen_coverage_records_are_totalled_without_inflating_scorer_calls() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("experiments.jsonl");
        journal::append(&path, &start(Ordering::Random)).unwrap();
        journal::append(
            &path,
            &Event::Screen {
                winners: 1,
                losers: 3,
                ms: 100,
            },
        )
        .unwrap();
        journal::append(
            &path,
            &Event::Screened {
                batch: 0,
                screened: 4,
            },
        )
        .unwrap();
        journal::append(
            &path,
            &Event::Screened {
                batch: 1,
                screened: 2,
            },
        )
        .unwrap();
        let report = summarise(&[&path]).unwrap();
        assert_eq!(report.screened, 6);
        assert_eq!(report.screen_calls, 1, "coverage is not a scorer call");
    }

    #[test]
    fn the_last_coverage_record_becomes_the_reported_progress() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("experiments.jsonl");
        journal::append(&path, &start(Ordering::Random)).unwrap();
        journal::append(
            &path,
            &Event::Coverage {
                blocked_by_reason: Default::default(),
                hidden: 12,
                tagged: 2,
                checkable: 12,
                checked: 2,
                blocked: 0,
                cut: 0,
                corpus_identity: None,
            },
        )
        .unwrap();
        // A later run cut two of them and screened more of what was left.
        journal::append(
            &path,
            &Event::Coverage {
                blocked_by_reason: Default::default(),
                hidden: 10,
                tagged: 2,
                checkable: 10,
                checked: 3,
                blocked: 0,
                cut: 2,
                corpus_identity: None,
            },
        )
        .unwrap();
        let report = summarise(&[&path]).unwrap();
        assert_eq!(report.hidden, Some(10));
        assert_eq!(report.checked, Some(3));
        assert_eq!(report.coverage_percent, Some(30.0));
        let json = serde_json::to_string(&report).unwrap();
        assert!(json.contains("\"coveragePercent\":30.0"), "{json}");
    }

    /// Every figure of the commit-description block reaches `report` (#40).
    #[test]
    fn the_report_carries_the_whole_coverage_block_not_just_the_percentage() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("experiments.jsonl");
        journal::append(&path, &start(Ordering::Random)).unwrap();
        journal::append(
            &path,
            &Event::Coverage {
                blocked_by_reason: Default::default(),
                hidden: 5013,
                tagged: 42,
                checkable: 5013,
                checked: 1204,
                blocked: 0,
                cut: 7,
                corpus_identity: None,
            },
        )
        .unwrap();
        let report = summarise(&[&path]).unwrap();
        assert_eq!(report.tagged, Some(42));
        assert_eq!(
            report.checkable,
            Some(5013),
            "tagged stay in the denominator"
        );
        assert_eq!(report.unchecked, Some(3809));
        assert_eq!(report.cut, Some(7));
        let json = serde_json::to_string(&report).unwrap();
        assert!(json.contains("\"unchecked\":3809"), "{json}");
    }

    /// Issue #93: the tag, the commit description and `report` must agree, so
    /// the blocked figure reaches the report too — and a journal written
    /// before it existed reports no blocked neurons rather than failing.
    #[test]
    fn the_report_carries_the_blocked_figure_and_reads_a_pre_93_journal() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("experiments.jsonl");
        journal::append(&path, &start(Ordering::Random)).unwrap();
        journal::append(
            &path,
            &Event::Coverage {
                blocked_by_reason: Default::default(),
                hidden: 5013,
                tagged: 42,
                checkable: 5013,
                checked: 4200,
                blocked: 3000,
                cut: 7,
                corpus_identity: None,
            },
        )
        .unwrap();
        let report = summarise(&[&path]).unwrap();
        assert_eq!(report.blocked, Some(3000));
        let json = serde_json::to_string(&report).unwrap();
        assert!(json.contains("\"blocked\":3000"), "{json}");

        assert_eq!(
            report.blocked_by_reason.map(|b| b.total()),
            Some(3000),
            "a journal that names no reasons still accounts for its blocked \
             total, as `unrecorded` (#103)"
        );

        let older = tmp.path().join("pre-93.jsonl");
        std::fs::write(
            &older,
            format!(
                "{}\n",
                r#"{"record":"coverage","hidden":10,"tagged":0,"checkable":10,"checked":4,"cut":0}"#
            ),
        )
        .unwrap();
        let old = summarise(&[&older]).unwrap();
        assert_eq!(old.checked, Some(4));
        assert_eq!(old.blocked, Some(0), "absent means none, not a failed read");
    }

    /// Issue #103: the breakdown reaches `report` too, and the dominant
    /// category is named — that is the figure the next proposal path is aimed
    /// at, and reading it out of the journal must not need a live run.
    #[test]
    fn the_report_names_the_blocked_breakdown_and_its_dominant_category() {
        use crate::blocked::{BlockedBreakdown, BlockedReason};
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("experiments.jsonl");
        journal::append(&path, &start(Ordering::Random)).unwrap();
        let mut reasons = BlockedBreakdown::default();
        for _ in 0..380 {
            reasons.add(BlockedReason::AggregateSquash);
        }
        for _ in 0..32 {
            reasons.add(BlockedReason::MissingActivation);
        }
        journal::append(
            &path,
            &Event::Coverage {
                blocked_by_reason: reasons,
                hidden: 5013,
                tagged: 0,
                checkable: 5013,
                checked: 4200,
                blocked: 412,
                cut: 0,
                corpus_identity: Some("corp-aaaa1111".into()),
            },
        )
        .unwrap();
        let report = summarise(&[&path]).unwrap();
        assert_eq!(report.blocked, Some(412));
        assert_eq!(
            report.blocked_by_reason.map(|b| b.total()),
            Some(412),
            "the breakdown accounts for every blocked uuid"
        );
        assert_eq!(
            report.dominant_blocked_reason.as_deref(),
            Some("aggregate-squash")
        );
        let json = serde_json::to_string(&report).unwrap();
        assert!(json.contains("\"aggregateSquash\":380"), "{json}");
    }

    /// The historical half of Issue #103: one entry per screening epoch, each
    /// holding that epoch's freshest breakdown, so how the blocked categories
    /// moved across corpus changes is readable from the journals alone.
    #[test]
    fn blocked_reasons_are_reported_per_epoch_across_corpus_changes() {
        use crate::blocked::{BlockedBreakdown, BlockedReason};
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("experiments.jsonl");
        journal::append(&path, &start(Ordering::Random)).unwrap();
        let breakdown = |reason: BlockedReason, n: usize| {
            let mut out = BlockedBreakdown::default();
            for _ in 0..n {
                out.add(reason);
            }
            out
        };
        let coverage =
            |identity: &str, blocked: usize, reasons: BlockedBreakdown| Event::Coverage {
                blocked_by_reason: reasons,
                hidden: 100,
                tagged: 0,
                checkable: 100,
                checked: 90,
                blocked,
                cut: 0,
                corpus_identity: Some(identity.into()),
            };
        // Two runs under the old corpus, then one under the new: the first
        // epoch keeps its freshest figures rather than a second row.
        journal::append(
            &path,
            &coverage(
                "corp-old",
                40,
                breakdown(BlockedReason::AggregateSquash, 40),
            ),
        )
        .unwrap();
        journal::append(
            &path,
            &coverage(
                "corp-old",
                30,
                breakdown(BlockedReason::AggregateSquash, 30),
            ),
        )
        .unwrap();
        journal::append(
            &path,
            &coverage(
                "corp-new",
                5,
                breakdown(BlockedReason::MissingActivation, 5),
            ),
        )
        .unwrap();

        let report = summarise(&[&path]).unwrap();
        assert_eq!(
            report.blocked_epochs.len(),
            2,
            "{:?}",
            report.blocked_epochs
        );
        assert_eq!(
            report.blocked_epochs[0].corpus_identity.as_deref(),
            Some("corp-old")
        );
        assert_eq!(
            report.blocked_epochs[0].blocked, 30,
            "the freshest snapshot"
        );
        assert_eq!(
            report.blocked_epochs[0].blocked_by_reason.aggregate_squash,
            30
        );
        assert_eq!(
            report.blocked_epochs[1].corpus_identity.as_deref(),
            Some("corp-new")
        );
        assert_eq!(
            report.blocked_epochs[1]
                .blocked_by_reason
                .missing_activation,
            5,
            "the new epoch is its own row, never folded into the old one"
        );
        // The current-epoch figures stay the latest snapshot, as before.
        assert_eq!(report.blocked, Some(5));
        assert_eq!(report.corpus_identity.as_deref(), Some("corp-new"));
    }

    /// Issue #102: `report` names the epoch its coverage figures belong to, and
    /// says whether that sweep finished — so a `100%` read out of a journal is
    /// readable as "100% of that corpus", and a later epoch replaces both.
    #[test]
    fn the_report_names_the_epoch_and_whether_that_sweep_finished() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("experiments.jsonl");
        journal::append(&path, &start(Ordering::Random)).unwrap();
        journal::append(
            &path,
            &Event::Coverage {
                blocked_by_reason: Default::default(),
                hidden: 4,
                tagged: 0,
                checkable: 4,
                checked: 4,
                blocked: 0,
                cut: 1,
                corpus_identity: Some("corp-aaaa1111".into()),
            },
        )
        .unwrap();
        let finished = summarise(&[&path]).unwrap();
        assert_eq!(finished.coverage_percent, Some(100.0));
        assert_eq!(finished.corpus_identity.as_deref(), Some("corp-aaaa1111"));
        assert_eq!(finished.sweep_complete, Some(true));

        // The corpus is extended: the fresh epoch replaces the figures and the
        // identity together, so no `100%` survives the change.
        journal::append(
            &path,
            &Event::Coverage {
                blocked_by_reason: Default::default(),
                hidden: 4,
                tagged: 0,
                checkable: 4,
                checked: 1,
                blocked: 0,
                cut: 0,
                corpus_identity: Some("corp-bbbb2222".into()),
            },
        )
        .unwrap();
        let fresh = summarise(&[&path]).unwrap();
        assert_eq!(fresh.coverage_percent, Some(25.0));
        assert_eq!(fresh.corpus_identity.as_deref(), Some("corp-bbbb2222"));
        assert_eq!(fresh.sweep_complete, Some(false));

        let json = serde_json::to_string(&fresh).unwrap();
        assert!(
            json.contains("\"corpusIdentity\":\"corp-bbbb2222\""),
            "{json}"
        );
        assert!(json.contains("\"sweepComplete\":false"), "{json}");

        // A journal written before the epoch was recorded names none, rather
        // than claiming one.
        let older = tmp.path().join("pre-100.jsonl");
        std::fs::write(
            &older,
            format!(
                "{}\n",
                r#"{"record":"coverage","hidden":10,"tagged":0,"checkable":10,"checked":10,"cut":0}"#
            ),
        )
        .unwrap();
        let old = summarise(&[&older]).unwrap();
        assert_eq!(old.corpus_identity, None);
        assert_eq!(old.sweep_complete, Some(true));
    }

    #[test]
    fn a_journal_with_no_coverage_record_names_no_epoch() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("experiments.jsonl");
        journal::append(&path, &start(Ordering::Random)).unwrap();
        let report = summarise(&[&path]).unwrap();
        assert_eq!(report.corpus_identity, None);
        assert_eq!(
            report.sweep_complete, None,
            "no coverage state is not an unfinished sweep"
        );
    }

    /// A journal written before Issue #74 carries the old `hidden - tagged`
    /// denominator. Replaying it verbatim would report the overstatement #74
    /// removed, so the report derives the denominator from `hidden`.
    #[test]
    fn a_pre_issue_74_journal_is_reported_on_the_full_hidden_denominator() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("experiments.jsonl");
        journal::append(&path, &start(Ordering::Random)).unwrap();
        std::fs::write(
            &path,
            format!(
                "{}{}\n",
                std::fs::read_to_string(&path).unwrap(),
                r#"{"record":"coverage","hidden":5013,"tagged":42,"checkable":4971,"checked":1204,"cut":7}"#
            ),
        )
        .unwrap();
        let report = summarise(&[&path]).unwrap();
        assert_eq!(
            report.checkable,
            Some(5013),
            "the old denominator is not replayed"
        );
        assert_eq!(report.unchecked, Some(3809));
        let percent = report.coverage_percent.expect("coverage percent");
        assert!(
            (percent - 1204.0 / 5013.0 * 100.0).abs() < f64::EPSILON,
            "{percent}"
        );
    }

    #[test]
    fn a_journal_with_no_coverage_record_reports_no_block_figures() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("experiments.jsonl");
        journal::append(&path, &start(Ordering::Random)).unwrap();
        let report = summarise(&[&path]).unwrap();
        assert_eq!(report.tagged, None);
        assert_eq!(report.checkable, None);
        assert_eq!(report.unchecked, None, "no coverage state is not zero left");
        assert_eq!(report.cut, None);
    }

    #[test]
    fn a_journal_with_no_coverage_record_reports_no_coverage() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("experiments.jsonl");
        journal::append(&path, &start(Ordering::Random)).unwrap();
        let report = summarise(&[&path]).unwrap();
        assert_eq!(report.hidden, None);
        assert_eq!(report.checked, None);
        assert_eq!(report.coverage_percent, None, "no coverage state is not 0%");
    }

    #[test]
    fn a_journal_written_before_issue_36_parses_with_no_screen_coverage() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("experiments.jsonl");
        std::fs::write(
            &path,
            concat!(
                r#"{"record":"start","seed":3,"permutation_identity":"x","hidden":2,"opening_score":0.5}"#,
                "\n",
                r#"{"record":"batch","batch":0,"candidates":2,"skipped":0,"remaining":0}"#,
                "\n",
                r#"{"record":"screen","winners":1,"losers":1,"ms":50}"#,
                "\n",
                r#"{"record":"full","individuals":1,"bundles":0,"accepted":false}"#,
                "\n",
                r#"{"record":"stop","reason":"timeout","accepts":0,"experiments":1,"final_score":0.5,"cumulative_delta":0.0}"#,
                "\n",
            ),
        )
        .unwrap();
        let report = summarise(&[&path]).unwrap();
        assert_eq!(report.screen_calls, 1);
        assert_eq!(report.screened, 0, "old journals carry no coverage records");
        assert_eq!(report.hidden, None, "coverage is absent, not zero");
        assert_eq!(report.coverage_percent, None);
        assert_eq!(report.stop_reason.as_deref(), Some("timeout"));
    }

    #[test]
    fn the_report_carries_the_winner_figures_and_the_budget_trim() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("experiments.jsonl");
        journal::append(&path, &start(Ordering::Random)).unwrap();
        journal::append(
            &path,
            &Event::Budget {
                est_ms_per_creature: 18_000.0,
                remaining_secs: 240,
                entries: 13,
                dropped_individuals: 24,
                dropped_bundles: 7,
            },
        )
        .unwrap();
        journal::append(
            &path,
            &Event::Winners {
                screened: 38,
                confirmed: 22,
                applied: 1,
                carried: 21,
                plans: 9,
                skipped: 3,
                best_cuts: 14,
                best_delta: 1.2e-4,
                dropped: 31,
                est_ms_per_creature: 18_000,
            },
        )
        .unwrap();
        let report = summarise(&[&path]).unwrap();
        let winners = report.winners.expect("winner figures");
        assert_eq!(winners.confirmed, 22);
        assert_eq!(winners.carried, 21);
        assert_eq!(report.est_ms_per_creature, Some(18_000.0));
        assert_eq!(report.budget_dropped, 31);
        assert_eq!(report.budget_trims, 1);
        let json = serde_json::to_string(&report).unwrap();
        assert!(json.contains("\"confirmed\":22"), "{json}");
        assert!(json.contains("\"budgetTrims\":1"), "{json}");
    }

    #[test]
    fn a_journal_with_no_winners_record_reports_none_rather_than_zeroes() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("experiments.jsonl");
        journal::append(&path, &start(Ordering::Random)).unwrap();
        let report = summarise(&[&path]).unwrap();
        assert_eq!(report.winners, None);
        assert_eq!(report.est_ms_per_creature, None);
        assert_eq!(report.budget_trims, 0);
    }

    /// An untrimmed cohort still journals its estimate, and must not be counted
    /// as budget-starved.
    #[test]
    fn a_cohort_that_fitted_the_budget_is_not_counted_as_a_trim() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("experiments.jsonl");
        journal::append(&path, &start(Ordering::Random)).unwrap();
        journal::append(
            &path,
            &Event::Budget {
                est_ms_per_creature: 900.0,
                remaining_secs: 2400,
                entries: 44,
                dropped_individuals: 0,
                dropped_bundles: 0,
            },
        )
        .unwrap();
        let report = summarise(&[&path]).unwrap();
        assert_eq!(report.budget_trims, 0);
        assert_eq!(report.budget_dropped, 0);
        assert_eq!(report.est_ms_per_creature, Some(900.0));
    }

    #[test]
    fn a_journal_written_before_issue_11_still_parses_as_the_random_control() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("experiments.jsonl");
        std::fs::write(
            &path,
            concat!(
                r#"{"record":"start","seed":3,"permutation_identity":"x","hidden":2,"opening_score":0.5}"#,
                "\n",
                r#"{"record":"full","individuals":1,"bundles":0,"accepted":true,"score":0.6,"delta":0.1}"#,
                "\n",
                r#"{"record":"stop","reason":"timeout","accepts":1,"experiments":1,"final_score":0.6,"cumulative_delta":0.1}"#,
                "\n",
            ),
        )
        .unwrap();
        let report = summarise(&[&path]).unwrap();
        assert_eq!(report.ordering, Some(Ordering::Random));
        assert_eq!(report.accepts, 1);
        assert_eq!(report.first_win_ms, Some(0));
        assert_eq!(report.accepted_cut_sizes, vec![0]);
    }
}
