//! Screening throughput and full-rescan ETA (Issue #162).
//!
//! Coverage says how much of the creature has been *looked at*; the pass
//! counters say how many times the razor has been round it. Neither says how
//! fast scorer-verified removable structure is being found, and the two are not
//! the same thing: a sweep can walk seven thousand entries in an hour while
//! only a few hundred of them ever become a candidate the scorer judges.
//!
//! So the work is measured as a **funnel**, one stage at a time, split by the
//! kind of thing the razor was working on:
//!
//! ```text
//! visits → blocked (nothing proposable) | judged (already decided)
//!        → proposed → sample-screened → sample winners → full-scored
//!        → confirmed → applied
//! ```
//!
//! Every stage is counted separately and none of them is a synonym for another.
//! `visited/hour` is the walk rate; `screened/hour` is the rate at which the
//! razor actually constructs a cut and pays a scorer to judge it. Conflating
//! the two is exactly what this module exists to prevent.
//!
//! Rates are per hour of **measured wall clock** — the loop's own elapsed time,
//! never the configured `--timeout-seconds`, so a run that stopped early, spent its
//! budget in replay or lost a cohort to the deadline still reports the honest
//! whole-run rate it achieved.
//!
//! Two rescan ETAs are reported, because "checked" costs two very different
//! amounts of work (Issue #93 files a record for a visit nothing could be
//! proposed for):
//!
//! - [`Throughput::visit_rescan_hours`] — walking every eligible visit once at
//!   the measured visit rate.
//! - [`Throughput::scored_rescan_hours`] — constructing and sample-scoring
//!   every currently *proposable* candidate once at the measured screen rate.
//!
//! The two agree when every proposed candidate was screened, and that agreement
//! is a finding rather than a defect: finding the proposable candidates means
//! walking the blocked visits too, so the blocked population is paid for by
//! both. They diverge as soon as proposals go unscreened — a coverage tail, a
//! run with screening off, a cohort the budget stopped — which is the case the
//! two numbers exist to separate.

use serde::{Deserialize, Serialize};

use crate::coverage::VisitCounts;
use crate::promote::FullOutcome;

/// Milliseconds in one hour, the unit every rate here is quoted in.
const MS_PER_HOUR: f64 = 3_600_000.0;

/// What the razor was working on when a funnel stage counted it.
///
/// Deliberately two values and not the whole [`crate::sweep::CandidateKind`]:
/// the population a rescan has to get through is hidden neurons and synapse
/// visits, and every other distinction — identity, ablation, constant, merge —
/// is a *way* of cutting one of those two, not a third thing to re-screen.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VisitKind {
    /// A hidden neuron.
    Neuron,
    /// A single synapse — one ordered edge.
    Synapse,
}

impl VisitKind {
    /// Classify a visit key: a [`crate::sweep::synapse_key`] is an edge, and
    /// anything else names a neuron.
    pub fn of_visit(key: &str) -> Self {
        match crate::sweep::parse_synapse_key(key) {
            Some(_) => Self::Synapse,
            None => Self::Neuron,
        }
    }

    /// Classify a full-cohort entry by the kind label the cohort gave it.
    ///
    /// `synapse` is the one label that removes no hidden neuron (Issue #138);
    /// an individual, a bundle and a group all cut neurons, so they count as
    /// neuron work here rather than as a kind of their own.
    pub fn of_cohort(kind: &str) -> Self {
        match kind {
            "synapse" => Self::Synapse,
            _ => Self::Neuron,
        }
    }
}

/// One stage of the screening funnel, per hour of measured wall clock.
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct Rate {
    /// Neuron entries per hour.
    pub neurons: f64,
    /// Synapse entries per hour.
    pub synapses: f64,
    /// Both kinds together.
    pub total: f64,
}

impl Rate {
    /// `counts` spread over `elapsed_ms`; all zero when no time was measured.
    ///
    /// A rate over zero elapsed time is not "infinitely fast", it is unknown,
    /// and zero is what the rendered line says nothing about — which is the
    /// honest reading for a run that never ran.
    fn per_hour(counts: VisitCounts, elapsed_ms: u64) -> Self {
        if elapsed_ms == 0 {
            return Self::default();
        }
        let hours = elapsed_ms as f64 / MS_PER_HOUR;
        Self {
            neurons: counts.neurons as f64 / hours,
            synapses: counts.synapses as f64 / hours,
            total: counts.total as f64 / hours,
        }
    }
}

/// How far the run's visits got, stage by stage (Issue #162).
///
/// Counts are **attempts**, exactly as [`VisitCounts`] is elsewhere: the same
/// neuron reached twice after an accepted cut rebuilt the sweep is two visits,
/// because it cost the run two visits' worth of wall clock.
///
/// Group proposals are excluded throughout. A neighbourhood is screened as one
/// extra candidate riding the batch (Issue #108), not as a sweep visit, and
/// counting it would credit the funnel with a visit the permutation never made.
/// Replayed known winners are excluded for the same reason: their candidates
/// came from the learnings cache, not from this run's screening pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct Funnel {
    /// Every visit attempt the sweep made.
    pub visits: VisitCounts,
    /// Visits over keys the fleet had already checked when the run opened.
    pub revisits: VisitCounts,
    /// Visits the razor could **structurally** propose no cut for.
    ///
    /// The cheap half of the walk: no candidate was built and no scorer was
    /// paid, so a rescan that is mostly these is fast and finds nothing. Why
    /// each was refused is [`crate::coverage::Coverage::blocked_by_reason`]'s
    /// job; this is how many, and of what kind. The same refusal the `blocked:`
    /// line counts (#93, #103) — that line counts distinct keys over the epoch,
    /// this one counts this run's attempts.
    pub blocked: VisitCounts,
    /// Visits skipped because the fleet's learnings had already judged them.
    ///
    /// Counted apart from [`Self::blocked`] (Issue #162) because the structure
    /// is *proposable*: the cut was built, scored and rejected on an earlier
    /// run, and only the known-failure cache stops it being offered again. The
    /// sweep files these as visited rather than blocked, so folding them into
    /// the blocked bucket would shrink the proposable estimate — and the scored
    /// rescan ETA with it — as the cache grows, which is exactly backwards on
    /// the mature creatures the ETA exists for.
    pub judged: VisitCounts,
    /// Valid pruning candidates the sweep constructed.
    pub proposed: VisitCounts,
    /// Candidates that entered the sampled screen.
    ///
    /// Below [`Self::proposed`] whenever a batch was proposed but never
    /// screened, and zero for a run with screening off — where candidates go
    /// straight to full scoring and [`Self::full_scored`] is what moves.
    pub sample_screened: VisitCounts,
    /// Candidates the sampled screen promoted.
    pub sample_winners: VisitCounts,
    /// Candidates the full corpus scored individually.
    pub full_scored: VisitCounts,
    /// Full-scored candidates whose own delta beat `--min-improvement`.
    pub confirmed: VisitCounts,
    /// Candidates an accepted winner applied to the incumbent.
    pub applied: VisitCounts,
}

impl Funnel {
    /// Count one visit the razor could structurally propose nothing for.
    pub(crate) fn record_blocked(&mut self, kind: VisitKind) {
        self.blocked.add(kind);
    }

    /// Count one visit skipped because the learnings cache had judged it.
    pub(crate) fn record_judged(&mut self, kind: VisitKind) {
        self.judged.add(kind);
    }

    /// Count one constructed pruning candidate.
    pub(crate) fn record_proposed(&mut self, kind: VisitKind) {
        self.proposed.add(kind);
    }

    /// Count one candidate put through the sampled screen.
    pub(crate) fn record_sample_screened(&mut self, kind: VisitKind) {
        self.sample_screened.add(kind);
    }

    /// Count one candidate the sampled screen promoted.
    pub(crate) fn record_sample_winner(&mut self, kind: VisitKind) {
        self.sample_winners.add(kind);
    }

    /// Fold one full-corpus cohort into the scored, confirmed and applied
    /// stages.
    ///
    /// Individuals only: a bundle or a group is one plan over several cuts, so
    /// counting its members here would report full scores of candidates the
    /// cohort never judged one at a time. What an accepted plan *removed* is
    /// [`crate::coverage::Winners`]' business, and stays there.
    pub(crate) fn observe_full(&mut self, full: &FullOutcome, min_improvement: f64) {
        for cand in &full.individuals {
            let kind = VisitKind::of_cohort(cand.kind);
            self.full_scored.add(kind);
            if cand.delta > min_improvement {
                self.confirmed.add(kind);
            }
        }
        // Individuals only here too. A bundle or group winner is a plan the
        // scored stage never counted, so counting it applied would put a cut
        // into the funnel that never entered it — and `applied` would exceed
        // `fullScored`. What those plans removed is `winners:`' business.
        if let Some(win) = &full.winner
            && matches!(win.candidate.kind, "individual" | "synapse")
        {
            self.applied.add(VisitKind::of_cohort(win.candidate.kind));
        }
    }
}

/// The funnel, the rates it implies and the rescan ETAs (Issue #162).
///
/// Derived fields are computed once by [`Self::measured`] and stored, the same
/// way [`crate::coverage::Passes::measured`] stores its equivalent-pass figure:
/// one constructor owns the arithmetic, so `coverage.json`, `coverage.txt` and
/// `ockham report` cannot disagree about a rate.
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct Throughput {
    /// Measured wall clock the funnel was counted over.
    ///
    /// The loop's own elapsed time, never `--timeout-seconds`: a run that
    /// stopped on its experiment cap after four minutes of a one-hour budget
    /// must report the rate it actually achieved.
    pub elapsed_ms: u64,
    /// How far the run's visits got, stage by stage.
    pub funnel: Funnel,
    /// Hidden neurons on the final incumbent — the neuron rescan population.
    pub hidden: usize,
    /// Synapse visits on the final incumbent — the edge rescan population.
    pub synapses: usize,
    /// Visit attempts per hour: the walk rate, cheap visits included.
    pub visits_per_hour: Rate,
    /// Constructed candidates per hour.
    pub proposed_per_hour: Rate,
    /// Sample-screened candidates per hour — the scorer-verified work rate.
    pub screened_per_hour: Rate,
    /// Individually full-scored candidates per hour.
    pub full_scored_per_hour: Rate,
    /// Estimated proposable candidates in the current population.
    ///
    /// The eligible population scaled by the proposable fraction this run
    /// measured, per kind: a creature whose edges are mostly typed proposes few
    /// of them, and a rescan of it has correspondingly little to score.
    ///
    /// `None` — not `0.0` — when a kind with a live population was never
    /// visited: its proposable share was never measured, and a total that
    /// quietly dropped it would read as a measurement of both kinds.
    pub proposable_estimate: Option<f64>,
    /// Hours to walk every eligible visit once at [`Self::visits_per_hour`].
    ///
    /// `None` when no time or no visit was measured — an ETA nothing was
    /// measured for is unknown, and reporting `0.0` would read as "already
    /// done".
    pub visit_rescan_hours: Option<f64>,
    /// Hours to construct and sample-score every proposable candidate once.
    ///
    /// `None` when a kind with a live population was never visited, so its
    /// proposable share is unmeasured: the total would silently omit it.
    pub scored_rescan_hours: Option<f64>,
}

impl Throughput {
    /// Assemble the measured funnel, its rates and its ETAs.
    ///
    /// `visits` and `revisits` come from the run's screening progress, which is
    /// the single place visit attempts are counted (Issue #140), so the funnel
    /// head can never disagree with the `visits:` line beside it.
    pub fn measured(
        mut funnel: Funnel,
        visits: VisitCounts,
        revisits: VisitCounts,
        elapsed_ms: u64,
        hidden: usize,
        synapses: usize,
    ) -> Self {
        funnel.visits = visits;
        funnel.revisits = revisits;
        let visits_per_hour = Rate::per_hour(funnel.visits, elapsed_ms);
        let screened_per_hour = Rate::per_hour(funnel.sample_screened, elapsed_ms);
        let visit_rescan_hours = rescan_hours(
            (Some(hidden as f64), visits_per_hour.neurons),
            (Some(synapses as f64), visits_per_hour.synapses),
        );
        // Measured once and used twice: the estimate published beside the ETA
        // and the ETA itself must be the same arithmetic, or a reader could
        // divide one by the rate and not get the other.
        let neurons = proposable(
            hidden,
            funnel.proposed.neurons + funnel.judged.neurons,
            funnel.visits.neurons,
        );
        let synapse_share = proposable(
            synapses,
            funnel.proposed.synapses + funnel.judged.synapses,
            funnel.visits.synapses,
        );
        let proposable_estimate = match (neurons, synapse_share) {
            (Some(n), Some(s)) => Some(n + s),
            _ => None,
        };
        let scored_rescan_hours = rescan_hours(
            (neurons, screened_per_hour.neurons),
            (synapse_share, screened_per_hour.synapses),
        );
        Self {
            elapsed_ms,
            funnel,
            hidden,
            synapses,
            visits_per_hour,
            proposed_per_hour: Rate::per_hour(funnel.proposed, elapsed_ms),
            screened_per_hour,
            full_scored_per_hour: Rate::per_hour(funnel.full_scored, elapsed_ms),
            proposable_estimate,
            visit_rescan_hours,
            scored_rescan_hours,
        }
    }

    /// Whether this run measured anything worth rendering.
    pub fn has_any(&self) -> bool {
        self.elapsed_ms > 0 && self.funnel.visits.total > 0
    }

    /// The description lines: the funnel per kind, the rates, then the ETAs.
    ///
    /// ```text
    /// funnel:    neurons 9100 visits · 8680 blocked · 0 judged · 420 proposed · 312 screened · 24 scored
    /// funnel:    synapses 31400 visits · 29295 blocked · 0 judged · 2105 proposed · 1840 screened · 60 scored
    /// rate:      neurons 312 screened/h · synapses 1840 screened/h · full rescan ~0.8h
    /// eta:       visit rescan ~0.6h · scored rescan ~0.8h · 5013 neurons + 2000 edges eligible
    /// ```
    ///
    /// `full rescan` on the `rate:` line and `scored rescan` on the `eta:` line
    /// are the same figure, deliberately: the compact line GRQ pastes into a
    /// commit subject carries it alone, and the `eta:` line puts it beside the
    /// walk-only estimate it must not be confused with.
    ///
    /// The synapse funnel line is omitted when the run reached no edge, exactly
    /// as the `visits:` line beside it is, so a neuron-only run renders neither
    /// a zero nor a claim about edges it never walked.
    pub(crate) fn lines(&self) -> Vec<String> {
        if !self.has_any() {
            return Vec::new();
        }
        let mut out = vec![format!(
            "{:<11}neurons {}",
            "funnel:",
            self.funnel_clause(VisitKind::Neuron)
        )];
        if self.funnel.visits.synapses > 0 {
            out.push(format!(
                "{:<11}synapses {}",
                "funnel:",
                self.funnel_clause(VisitKind::Synapse)
            ));
        }
        out.push(format!(
            "{:<11}neurons {:.0} screened/h · synapses {:.0} screened/h · full rescan {}",
            "rate:",
            self.screened_per_hour.neurons,
            self.screened_per_hour.synapses,
            hours_clause(self.scored_rescan_hours)
        ));
        out.push(format!(
            "{:<11}visit rescan {} · scored rescan {} · {} + {} eligible",
            "eta:",
            hours_clause(self.visit_rescan_hours),
            hours_clause(self.scored_rescan_hours),
            plural(self.hidden, "neuron"),
            plural(self.synapses, "edge")
        ));
        out
    }

    /// One kind's funnel, stage by stage, in one clause.
    fn funnel_clause(&self, kind: VisitKind) -> String {
        let of = |counts: VisitCounts| match kind {
            VisitKind::Neuron => counts.neurons,
            VisitKind::Synapse => counts.synapses,
        };
        format!(
            "{} visits · {} blocked · {} judged · {} proposed · {} screened · {} scored",
            of(self.funnel.visits),
            of(self.funnel.blocked),
            of(self.funnel.judged),
            of(self.funnel.proposed),
            of(self.funnel.sample_screened),
            of(self.funnel.full_scored)
        )
    }
}

/// `population` scaled by the proposable fraction measured over `visits`.
///
/// `None` when the kind has a live population but was never visited: nothing
/// was measured, so any figure would be invented. An empty population is `0.0`
/// — there is genuinely nothing to propose — rather than unknown.
fn proposable(population: usize, proposed: usize, visits: usize) -> Option<f64> {
    if population == 0 {
        return Some(0.0);
    }
    (visits > 0).then(|| population as f64 * (proposed as f64 / visits as f64))
}

/// Hours for one pass over both kinds: each kind's population at its own rate.
///
/// Per kind and summed, never a blended rate over a blended population: a run
/// that walked only neurons has measured nothing about edges, and projecting
/// the neuron rate onto the edge population would invent the number. Unknown —
/// `None` — when a kind with a live population was never measured or never
/// worked at, because the total would otherwise report the other kind's
/// estimate as though it covered both.
///
/// `Some(0.0)` is a real answer and not an unknown one: there is nothing left
/// of either population to get through.
fn rescan_hours(neurons: (Option<f64>, f64), synapses: (Option<f64>, f64)) -> Option<f64> {
    let mut total = 0.0;
    for (population, rate) in [neurons, synapses] {
        let population = population?;
        if population <= 0.0 {
            continue;
        }
        if rate <= 0.0 {
            return None;
        }
        total += population / rate;
    }
    Some(total)
}

/// `~18.7h`, or `unknown` when nothing was measured to estimate from.
///
/// An estimate that rounds to `~0.0h` is rendered `~<0.1h` instead: a rescan
/// that takes six minutes is fast, and `0.0` reads as though it were already
/// done.
fn hours_clause(hours: Option<f64>) -> String {
    match hours {
        Some(h) if h <= 0.0 => "none left".to_string(),
        Some(h) if h < 0.05 => "~<0.1h".to_string(),
        Some(h) => format!("~{h:.1}h"),
        None => "unknown".to_string(),
    }
}

/// `1 neuron`, `3 edges` — the count with its unit, pluralised.
fn plural(count: usize, unit: &str) -> String {
    if count == 1 {
        format!("{count} {unit}")
    } else {
        format!("{count} {unit}s")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn counts(neurons: usize, synapses: usize) -> VisitCounts {
        let mut out = VisitCounts::default();
        for _ in 0..neurons {
            out.add(VisitKind::Neuron);
        }
        for _ in 0..synapses {
            out.add(VisitKind::Synapse);
        }
        out
    }

    /// One hour of wall clock, so a count reads straight off as a rate.
    const ONE_HOUR_MS: u64 = 3_600_000;

    fn funnel(visits: (usize, usize), blocked: (usize, usize), proposed: (usize, usize)) -> Funnel {
        Funnel {
            visits: counts(visits.0, visits.1),
            blocked: counts(blocked.0, blocked.1),
            proposed: counts(proposed.0, proposed.1),
            sample_screened: counts(proposed.0, proposed.1),
            ..Funnel::default()
        }
    }

    /// A visit key classifies as an edge; a neuron uuid does not (#162).
    #[test]
    fn a_synapse_key_is_a_synapse_visit() {
        let key = crate::sweep::synapse_key("h_a", "h_b");
        assert_eq!(VisitKind::of_visit(&key), VisitKind::Synapse);
        assert_eq!(VisitKind::of_visit("h_a"), VisitKind::Neuron);
        assert_eq!(VisitKind::of_cohort("synapse"), VisitKind::Synapse);
        for kind in ["individual", "bundle", "group"] {
            assert_eq!(VisitKind::of_cohort(kind), VisitKind::Neuron);
        }
    }

    /// The rate is per measured hour, and the two kinds never merge (#162).
    #[test]
    fn rates_are_per_measured_hour_and_split_by_kind() {
        let t = Throughput::measured(
            funnel((900, 1800), (600, 1500), (300, 300)),
            counts(900, 1800),
            counts(0, 0),
            ONE_HOUR_MS / 2,
            900,
            1800,
        );
        // Half an hour of wall clock, so every count doubles into its rate.
        assert_eq!(t.visits_per_hour.neurons, 1800.0);
        assert_eq!(t.visits_per_hour.synapses, 3600.0);
        assert_eq!(t.visits_per_hour.total, 5400.0);
        assert_eq!(t.screened_per_hour.neurons, 600.0);
        assert_eq!(t.screened_per_hour.synapses, 600.0);
        assert_eq!(t.proposed_per_hour.total, 1200.0);
    }

    /// The stages are counted apart: a visit is not a proposal, a proposal is
    /// not a screen, and a screen is not a full score (#162).
    #[test]
    fn the_funnel_stages_cannot_be_conflated() {
        let mut f = funnel((100, 10), (90, 8), (10, 2));
        f.sample_screened = counts(6, 1);
        f.sample_winners = counts(3, 1);
        f.full_scored = counts(2, 1);
        f.confirmed = counts(1, 0);
        f.applied = counts(1, 0);
        let t = Throughput::measured(f, counts(100, 10), counts(20, 0), ONE_HOUR_MS, 100, 10);
        assert_eq!(t.funnel.visits.total, 110);
        assert_eq!(t.funnel.revisits.total, 20);
        assert_eq!(t.funnel.blocked.total, 98);
        assert_eq!(t.funnel.proposed.total, 12);
        assert_eq!(t.funnel.sample_screened.total, 7);
        assert_eq!(t.funnel.sample_winners.total, 4);
        assert_eq!(t.funnel.full_scored.total, 3);
        assert_eq!(t.funnel.confirmed.total, 1);
        assert_eq!(t.funnel.applied.total, 1);
        let lines = t.lines().join("\n");
        assert!(
            lines.contains(
                "funnel:    neurons 100 visits · 90 blocked · 0 judged · 10 proposed · 6 screened · 2 scored"
            ),
            "{lines}"
        );
        assert!(
            lines.contains(
                "funnel:    synapses 10 visits · 8 blocked · 0 judged · 2 proposed · 1 screened · 1 scored"
            ),
            "{lines}"
        );
    }

    /// Blocked visits are cheap, so they raise the walk rate without raising
    /// the screen rate — and the gap between the two ETAs is where they show
    /// (Issue #162).
    #[test]
    fn blocked_visits_separate_the_two_rescan_etas() {
        // 1000 neuron visits in an hour, but only 100 of them proposed, and
        // only 50 of those were screened: the walk is ten times the rate the
        // scorer works at.
        let mut f = funnel((1000, 0), (900, 0), (100, 0));
        f.sample_screened = counts(50, 0);
        let t = Throughput::measured(f, counts(1000, 0), counts(0, 0), ONE_HOUR_MS, 2000, 0);
        // 2000 eligible visits at 1000 visits/h.
        assert_eq!(t.visit_rescan_hours, Some(2.0));
        // 10% of 2000 are proposable, screened at 50/h.
        assert_eq!(t.proposable_estimate, Some(200.0));
        assert_eq!(t.scored_rescan_hours, Some(4.0));
        let lines = t.lines().join("\n");
        assert!(
            lines.contains("eta:       visit rescan ~2.0h · scored rescan ~4.0h"),
            "{lines}"
        );
    }

    /// An unmeasured kind makes **both** ETAs unknown, never a silent zero and
    /// never the other kind's rate projected onto it (Issue #162).
    #[test]
    fn an_unvisited_population_leaves_both_etas_unknown() {
        let f = funnel((100, 0), (50, 0), (50, 0));
        let t = Throughput::measured(f, counts(100, 0), counts(0, 0), ONE_HOUR_MS, 100, 400);
        assert_eq!(
            t.scored_rescan_hours, None,
            "400 edges were never screened, so their share is unmeasured"
        );
        assert_eq!(
            t.visit_rescan_hours, None,
            "400 edges were never walked either — the neuron rate says nothing about them"
        );
        assert_eq!(
            t.proposable_estimate, None,
            "an estimate that omits a live population is not an estimate"
        );
        let lines = t.lines().join("\n");
        assert!(lines.contains("scored rescan unknown"), "{lines}");
        assert!(lines.contains("visit rescan unknown"), "{lines}");
        assert!(
            !lines.contains("funnel:    synapses"),
            "no edge was walked, so no edge funnel is claimed: {lines}"
        );
    }

    /// A visit the learnings cache already judged is proposable structure, so
    /// it is counted apart from a structural refusal and keeps the scored
    /// rescan honest as the cache grows (Issue #162).
    #[test]
    fn a_judged_visit_is_not_a_blocked_one() {
        let mut f = funnel((100, 0), (60, 0), (10, 0));
        f.judged = counts(30, 0);
        let t = Throughput::measured(f, counts(100, 0), counts(0, 0), ONE_HOUR_MS, 100, 0);
        assert_eq!(t.funnel.blocked.neurons, 60);
        assert_eq!(t.funnel.judged.neurons, 30);
        assert_eq!(
            t.funnel.blocked.total + t.funnel.judged.total + t.funnel.proposed.total,
            t.funnel.visits.total,
            "every visit is blocked, judged or proposed"
        );
        // 40 of 100 visits were proposable — the 10 proposed plus the 30 the
        // cache skipped — so 40 of the 100 eligible neurons are, screened at
        // 10/h.
        assert_eq!(t.proposable_estimate, Some(40.0));
        assert_eq!(t.scored_rescan_hours, Some(4.0));
    }

    /// A creature with nothing proposable left is answered, not shrugged at:
    /// `none left` is a measurement, `unknown` is the absence of one (#162).
    #[test]
    fn a_fully_blocked_creature_reports_nothing_left_to_score() {
        let mut f = funnel((100, 0), (100, 0), (0, 0));
        f.sample_screened = counts(0, 0);
        let t = Throughput::measured(f, counts(100, 0), counts(0, 0), ONE_HOUR_MS, 100, 0);
        assert_eq!(t.proposable_estimate, Some(0.0));
        assert_eq!(t.scored_rescan_hours, Some(0.0));
        let lines = t.lines().join("\n");
        assert!(lines.contains("scored rescan none left"), "{lines}");
        assert!(lines.contains("visit rescan ~1.0h"), "{lines}");
    }

    /// The cohort end of the funnel: individuals are scored and confirmed on
    /// their own delta, and only an individual or synapse winner is counted
    /// applied — a bundle plan the scored stage never saw is not (Issue #162).
    #[test]
    fn observe_full_counts_individuals_and_leaves_plans_to_the_winners_block() {
        let candidate = |kind: &'static str, delta: f64| crate::promote::FullCandidate {
            stem: kind.into(),
            kind,
            uuids: vec!["h_a".into(), "h_b".into()],
            score: 0.5 + delta,
            error: 0.5,
            complexity_penalty: 0.0,
            after: crate::ablation::StructureSnapshot {
                hidden_neurons: 1,
                constant_neurons: 0,
                synapses: 2,
                growth_units: 1.2,
            },
            delta,
        };
        let outcome =
            |individuals: Vec<crate::promote::FullCandidate>,
             winner: Option<crate::promote::FullCandidate>| FullOutcome {
                incumbent_score: 0.5,
                incumbent_error: 0.5,
                individuals,
                bundles: Vec::new(),
                groups: Vec::new(),
                sample_false_positives: Vec::new(),
                winner: winner.map(|candidate| crate::promote::LocalWinner {
                    candidate,
                    checksum: "c".into(),
                    creature: crate::fixtures::creature(1, 1, Vec::new(), Vec::new()),
                }),
                full_ms: 1,
                skipped_bundles: 0,
                dropped_individuals: 0,
                dropped_bundles: 0,
                capped_plans: 0,
            };

        // A delta exactly at the threshold is not confirmed: the rule is
        // strictly better, exactly as the cohort's own accept rule is.
        let mut f = Funnel::default();
        f.observe_full(
            &outcome(
                vec![
                    candidate("individual", 0.2),
                    candidate("individual", 0.1),
                    candidate("synapse", 0.3),
                ],
                Some(candidate("individual", 0.2)),
            ),
            0.1,
        );
        assert_eq!(f.full_scored.neurons, 2);
        assert_eq!(f.full_scored.synapses, 1);
        assert_eq!(f.confirmed.neurons, 1, "0.1 is not strictly above 0.1");
        assert_eq!(f.confirmed.synapses, 1);
        assert_eq!(f.applied.neurons, 1);

        // A bundle winner cut neurons the scored stage never judged one at a
        // time, so the funnel counts none of them applied.
        let mut bundled = Funnel::default();
        bundled.observe_full(&outcome(Vec::new(), Some(candidate("bundle", 0.9))), 0.1);
        assert_eq!(bundled.applied.total, 0);
        assert_eq!(bundled.full_scored.total, 0);

        // An empty cohort moves nothing.
        let mut empty = Funnel::default();
        empty.observe_full(&outcome(Vec::new(), None), 0.1);
        assert_eq!(empty, Funnel::default());
    }

    /// A run that measured no wall clock reports no rate at all, rather than
    /// dividing by zero (Issue #162).
    #[test]
    fn no_measured_time_reports_nothing() {
        let t = Throughput::measured(
            funnel((5, 5), (0, 0), (5, 5)),
            counts(5, 5),
            counts(0, 0),
            0,
            5,
            5,
        );
        assert_eq!(t.visits_per_hour, Rate::default());
        assert_eq!(t.visit_rescan_hours, None);
        assert_eq!(t.scored_rescan_hours, None);
        assert!(!t.has_any());
        assert!(t.lines().is_empty());
    }

    /// The whole-run rate is measured over the whole run, replay and full
    /// scoring included — the honest denominator (Issue #162).
    #[test]
    fn the_rate_is_over_the_whole_run_not_the_screening_share() {
        let mut f = funnel((360, 0), (0, 0), (360, 0));
        f.full_scored = counts(36, 0);
        // Two hours of wall clock, most of it in full scoring.
        let t = Throughput::measured(f, counts(360, 0), counts(0, 0), 2 * ONE_HOUR_MS, 360, 0);
        assert_eq!(t.screened_per_hour.neurons, 180.0);
        assert_eq!(t.full_scored_per_hour.neurons, 18.0);
        assert_eq!(t.visit_rescan_hours, Some(2.0));
    }

    /// A rescan of minutes is not a rescan of nothing, and one eligible neuron
    /// is not `1 neurons` (Issue #162).
    #[test]
    fn a_sub_six_minute_eta_and_a_single_neuron_render_honestly() {
        let f = funnel((10, 0), (0, 0), (10, 0));
        let t = Throughput::measured(f, counts(10, 0), counts(0, 0), ONE_HOUR_MS, 1, 0);
        assert_eq!(t.visit_rescan_hours, Some(0.1));
        let lines = t.lines().join("\n");
        assert!(lines.contains("· 1 neuron + 0 edges eligible"), "{lines}");
        let quick = Throughput::measured(
            funnel((1000, 0), (0, 0), (1000, 0)),
            counts(1000, 0),
            counts(0, 0),
            ONE_HOUR_MS,
            10,
            0,
        );
        assert!(
            quick.lines().join("\n").contains("visit rescan ~<0.1h"),
            "{:?}",
            quick.lines()
        );
    }

    /// The artefact round-trips, so `coverage.json` and `report` read back
    /// exactly what was written (Issue #162).
    #[test]
    fn throughput_round_trips_through_json() {
        let t = Throughput::measured(
            funnel((10, 20), (5, 15), (5, 5)),
            counts(10, 20),
            counts(2, 3),
            ONE_HOUR_MS,
            10,
            20,
        );
        let json = serde_json::to_string(&t).unwrap();
        assert_eq!(serde_json::from_str::<Throughput>(&json).unwrap(), t);
        assert!(json.contains("\"screenedPerHour\""), "{json}");
        assert!(json.contains("\"scoredRescanHours\""), "{json}");
    }
}
