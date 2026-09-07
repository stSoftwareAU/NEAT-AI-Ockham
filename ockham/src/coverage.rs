//! Screening coverage over the **current** incumbent (Issue #37).
//!
//! One place answers "how far has Ockham got through this creature?", so the
//! `ockham` tag, the commit description and `--report` can never disagree.
//!
//! Four rules make the denominator honest:
//!
//! - **The current incumbent is the whole world.** A screen record for a uuid
//!   that is no longer on the creature is ignored entirely — it neither raises
//!   `checked` nor `hidden`.
//! - **Every hidden neuron is checkable, tagged ones included** (Issue #74).
//!   Since #63 Ockham proposes tagged neurons like any other, so they sit in
//!   the denominator and a screened tagged uuid raises `checked`. `tagged`
//!   survives as an informational count reported *beside* the percentage,
//!   never deducted from it — deducting overstated progress by half.
//! - **A visit counts, even when there was nothing to try** (Issue #93). Most
//!   hidden neurons of a forest-heavy creature can never be ablated — they feed
//!   an aggregate squash or carry a typed synapse — and while those visits left
//!   no record the numerator was pinned to the prunable minority and could only
//!   *fall*, one neuron per accepted cut. The sweep files coverage for them
//!   too, and [`Coverage::blocked`] reports how many of the checked were
//!   blocked, so a rising percentage never claims a screen that never happened.
//! - **A synapse is a visit too** (Issue #137). The razor cuts single edges as
//!   well as neurons, so the population is every hidden neuron **and** every
//!   synapse the incumbent carries, keyed by [`crate::sweep::synapse_key`].
//!   Typed edges are in it beside ordinary ones: the sweep visits one, the
//!   visit is blocked, and that blocked record is what makes it checked — so a
//!   creature full of typed edges reaches a complete sweep instead of standing
//!   permanently short of one. `checkable` stays the whole visit population, so
//!   [`Coverage::percent`] and [`Coverage::sweep_complete`] keep their meaning
//!   and no consumer changes its arithmetic.
//!
//! Evolution keeps adding hidden neurons, and each new one starts unchecked and
//! *lowers* the percentage. That is intended: coverage is a statement about the
//! creature in front of us, not a monotonic score.
//!
//! It is also a statement about **one corpus** (Issue #100). A sweep can finish;
//! Ockham never does. `sweep X/X checked (100.0% of epoch)` says the sweep is
//! complete for the training data in hand, and the corpus is extended every few
//! days — so the records the count is built from are the ones filed under the
//! current corpus identity ([`crate::learnings::current_epoch_screens`]), and
//! [`CoverageReport::corpus_identity`] names the epoch the figures belong to.
//!
//! Issue #102 puts that scope in the **wording** rather than only in a trailing
//! line: every percentage Ockham renders says `of epoch`, a finished sweep is
//! reported as `sweep complete` rather than as Ockham finishing, and the epoch
//! is named beside the figures — in [`short_epoch`] form for humans, in full in
//! `coverage.json`. Cumulative coverage across every epoch the store has ever
//! seen is kept as its own [`History`] line, never folded into the current
//! percentage.

use std::collections::{HashMap, HashSet};
use std::path::Path;

use neat_core::CreatureExport;
use serde::{Deserialize, Serialize};

use crate::blocked::{BlockedBreakdown, BlockedReason};
use crate::learnings::Screened;

/// Rendered commit-description block, written beside `best.json` (Issue #40).
pub const COVERAGE_TEXT_FILE: &str = "coverage.txt";
/// Serialised [`Coverage`], written beside `best.json` (Issue #40).
pub const COVERAGE_JSON_FILE: &str = "coverage.json";

/// Characters of a corpus identity a human-readable epoch clause carries.
pub const EPOCH_SHORT_LEN: usize = 8;

/// Compact epoch id for a commit subject or a log line (Issue #102).
///
/// The first [`EPOCH_SHORT_LEN`] characters of the corpus identity — enough to
/// see that the epoch changed, short enough for a commit subject. The full
/// identity is never dropped: `coverage.json` and the journal `coverage` record
/// both carry it, so a reset stays diagnosable exactly.
///
/// Truncation is on a character boundary, so a non-hex identity cannot panic.
pub fn short_epoch(identity: &str) -> &str {
    match identity.char_indices().nth(EPOCH_SHORT_LEN) {
        Some((end, _)) => &identity[..end],
        None => identity,
    }
}

/// Screening coverage of one incumbent at one moment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Coverage {
    /// Hidden neurons on the current incumbent.
    pub hidden: usize,
    /// Hidden neurons carrying tags, screened like any other.
    ///
    /// Informational only (Issue #87): the count reports what is there, it
    /// never changes what Ockham may prune.
    pub tagged: usize,
    /// Visits Ockham may try — every hidden neuron **and** every synapse.
    ///
    /// The key keeps its name so `coverage.json` stays deserialisable by
    /// anything already reading it; what changed is the definition. It widened
    /// once for #74 (tagged neurons stopped being deducted) and again for #137:
    /// the population is now [`Self::hidden`] + [`Self::synapses`], because the
    /// razor cuts single edges as well as neurons and a denominator that
    /// counted only neurons would call an epoch swept while every edge on the
    /// creature had never been tried.
    ///
    /// [`Self::percent`] and [`Self::sweep_complete`] keep their arithmetic
    /// unchanged, so no consumer has to change to read the wider figure.
    pub checkable: usize,
    /// Visit keys with at least one screen record — neurons and synapses.
    pub checked: usize,
    /// Synapse visits on the current incumbent: one per ordered edge pair.
    ///
    /// The edge half of [`Self::checkable`] (Issue #137). An **ordinary**
    /// (untyped) edge is what [`crate::ablation::ablate_synapse`] can actually
    /// cut, and a **typed** edge is counted here beside it: the sweep visits
    /// one and files the blocked record that stops it being asked again, so
    /// leaving typed edges out would strand a creature full of them on a sweep
    /// that could never complete.
    ///
    /// `#[serde(default)]` so a pre-#137 `coverage.json` still deserialises,
    /// reading as no synapse visits rather than as a failed parse.
    #[serde(default)]
    pub synapses: usize,
    /// Synapse visit keys with at least one screen record.
    ///
    /// The synapse share of [`Self::checked`], reported beside the total rather
    /// than carved out of it: `checked` stays the whole visit numerator so
    /// [`Self::percent`] keeps its meaning.
    ///
    /// `#[serde(default)]` so a pre-#137 `coverage.json` still deserialises.
    #[serde(default)]
    pub synapses_checked: usize,
    /// Checked UUIDs the razor could never propose a cut for (Issue #93).
    ///
    /// A subset of [`Self::checked`]: every record for the uuid is a skipped
    /// visit, so it has been looked at but never scored. Reported beside the
    /// percentage, never deducted from it — the neuron is on the creature and
    /// the sweep has been to it.
    ///
    /// `#[serde(default)]` so a pre-#93 `coverage.json` still deserialises.
    #[serde(default)]
    pub blocked: usize,
    /// [`Self::blocked`] split by reason code (Issue #103).
    ///
    /// One number cannot be attacked. This says *why* the razor could propose
    /// nothing, so the dominant category is visible and can be worked on: the
    /// counts are over UUIDs and sum to exactly [`Self::blocked`].
    ///
    /// `#[serde(default)]` so a pre-#103 `coverage.json` still deserialises,
    /// reading as no reasons rather than as a failed parse.
    #[serde(default)]
    pub blocked_by_reason: BlockedBreakdown,
    /// Hidden neurons removed this run.
    pub cut: usize,
}

impl Coverage {
    /// `checked / checkable * 100`; `0.0` when there are no hidden neurons.
    ///
    /// Never divides by zero and never exceeds 100.
    pub fn percent(&self) -> f64 {
        if self.checkable == 0 {
            return 0.0;
        }
        (self.checked as f64 / self.checkable as f64 * 100.0).min(100.0)
    }

    /// Whether the sweep has reached every visit **this epoch** (#137).
    ///
    /// Every hidden neuron *and* every synapse: one unchecked edge is enough to
    /// keep the sweep open, because an epoch declared finished with thousands
    /// of edge visits never tried is the failure this counts against.
    ///
    /// A creature with nothing to visit is not a finished sweep: there was
    /// nothing to sweep, and `0/0` must never render as an achievement.
    pub fn sweep_complete(&self) -> bool {
        self.checkable > 0 && self.checked >= self.checkable
    }

    /// One-line progress summary — a log line beside the run's other figures.
    ///
    /// `sweep 1204/5013 checked (24.0% of epoch), 7 cut, 330/2000 synapses,
    /// 42 tagged`. The `X/Y` denominator is [`Self::checkable`], so it always
    /// agrees with the percentage, and `of epoch` says what the percentage is a
    /// percentage *of* (Issue #102) — the corpus in hand, not Ockham's whole
    /// task. The synapse clause names the edge half of that denominator (#137)
    /// and the tagged clause counts neurons inside it (#74); each is omitted
    /// when it has nothing to report, so a creature with no synapse visits
    /// renders exactly the line it did before.
    pub fn summary(&self) -> String {
        let mut out = format!(
            "sweep {}/{} checked ({:.1}% of epoch), {} cut",
            self.checked,
            self.checkable,
            self.percent(),
            self.cut
        );
        if self.synapses > 0 {
            out.push_str(&format!(
                ", {}/{} synapses",
                self.synapses_checked, self.synapses
            ));
        }
        if self.tagged > 0 {
            out.push_str(&format!(", {} tagged", self.tagged));
        }
        out
    }

    /// Visits with no screen record yet — neurons and synapses alike (#137).
    ///
    /// Saturating: a stale record set can report more `checked` than there are
    /// checkable visits, and "minus three unchecked" is not a measurement.
    pub fn unchecked(&self) -> usize {
        self.checkable.saturating_sub(self.checked)
    }

    /// Multi-line commit-description block — the GRQ-facing artefact (Issue #40).
    ///
    /// ```text
    /// 🪒 Ockham neuron screening coverage
    /// sweep:     1204 of 5013 visits (24.0% of epoch)
    /// synapses:  330 of 2000 edges checked this epoch
    /// epoch:     corpus 6fc028da — coverage counts this corpus only
    /// cut:       7 this run
    /// unchecked: 3809 remaining this epoch (~39 runs at 100/run)
    /// blocked:   412 checked with no cut proposed
    /// tagged:    42 carry tags, screened like any other
    /// ```
    ///
    /// Line-oriented and stable: GRQ pastes it into a `git commit` description.
    /// `candidates` is the configured `--candidates` batch size; the
    /// runs-remaining clause is **omitted** rather than rendering `inf` when
    /// the batch size is zero, and a finished sweep reads
    /// `0 remaining — sweep complete for this epoch` instead (Issue #102): the
    /// sweep finishes, Ockham does not. `epoch` is the corpus identity the
    /// figures were measured against, rendered in [`short_epoch`] form and
    /// omitted when the caller has no screen store to name one.
    ///
    /// The `synapses:` line is the additive half of Issue #137: it says how
    /// much of the edge population inside the `sweep:` denominator has been
    /// visited. It carries **no percentage of its own** — the only percentage
    /// in the block is `sweep:`, over the whole visit population, so two
    /// identically-suffixed percentages with different denominators can never
    /// sit one above the other. It is omitted entirely when the creature
    /// carries no synapse visits — so an older `coverage.json`, which deserialises with no synapse
    /// figures at all, still renders the block byte for byte as it did. The
    /// `sweep:` noun follows the same rule: `hidden` while the population is
    /// hidden neurons alone, `visits` once edges are in it, because a widened
    /// denominator labelled `hidden` would be a wrong count, not a stable one.
    ///
    /// The `tagged:` line is omitted when nothing is tagged, and says only how
    /// many neurons carry tags — cutting one needs no declaration (Issue #87).
    /// The `blocked:` line is likewise omitted when nothing is blocked, and
    /// says how much of `checked` was reached by a visit the razor could
    /// propose nothing for (Issue #93). It is followed by a `reasons:` line
    /// breaking that total down by code (Issue #103) — commonest first, each
    /// with its share of the blocked total — so the largest category to attack
    /// is legible from the commit description itself. No trailing newline —
    /// [`write_files`] adds one.
    pub fn description(&self, candidates: usize, epoch: Option<&str>) -> String {
        let unchecked = self.unchecked();
        let remaining = if self.sweep_complete() {
            String::from("0 remaining — sweep complete for this epoch")
        } else if unchecked == 0 {
            // No hidden neurons at all: nothing was swept, so nothing finished.
            String::from("0 remaining — no hidden neurons to sweep")
        } else {
            let runs = if candidates > 0 {
                let n = unchecked.div_ceil(candidates);
                let unit = if n == 1 { "run" } else { "runs" };
                format!(" (~{n} {unit} at {candidates}/run)")
            } else {
                String::new()
            };
            format!("{unchecked} remaining this epoch{runs}")
        };
        let mut out = String::from("🪒 Ockham neuron screening coverage\n");
        let population = if self.synapses > 0 {
            "visits"
        } else {
            "hidden"
        };
        out.push_str(&format!(
            "{:<11}{} of {} {population} ({:.1}% of epoch)\n",
            "sweep:",
            self.checked,
            self.checkable,
            self.percent()
        ));
        if self.synapses > 0 {
            out.push_str(&format!(
                "{:<11}{} of {} edges checked this epoch\n",
                "synapses:", self.synapses_checked, self.synapses
            ));
        }
        if let Some(identity) = epoch {
            out.push_str(&format!(
                "{:<11}corpus {} — coverage counts this corpus only\n",
                "epoch:",
                short_epoch(identity)
            ));
        }
        out.push_str(&format!("{:<11}{} this run\n", "cut:", self.cut));
        out.push_str(&format!("{:<11}{remaining}", "unchecked:"));
        if self.blocked > 0 {
            out.push_str(&format!(
                "\n{:<11}{} checked with no cut proposed",
                "blocked:", self.blocked
            ));
            if let Some(reasons) = self.blocked_by_reason.summary() {
                out.push_str(&format!("\n{:<11}{reasons}", "reasons:"));
            }
        }
        if self.tagged > 0 {
            out.push_str(&format!(
                "\n{:<11}{} carry tags, screened like any other",
                "tagged:", self.tagged
            ));
        }
        out
    }
}

/// Distinct hidden UUIDs one run moved from unchecked to checked (Issue #77).
///
/// Coverage is re-derived from the fleet store on every run, so a run that
/// advanced nothing renders the same well-formed block as one that advanced a
/// full batch — that is what let the #63 plateau run for eight runs unnoticed.
/// This counts only the UUIDs that had **no** screen record when the run
/// opened, so re-visiting the stalest neurons of an already-complete creature
/// is honestly reported as zero new coverage.
///
/// Since #93 a visit the razor could propose nothing for files a record too, so
/// this counts every uuid the run reached — which is why the rendered line says
/// *newly checked* and [`Coverage::blocked`] says how many were never scored.
/// Run-level visit counts, split by the kind of thing the sweep reached.
///
/// Unlike coverage, these are **attempts**, not distinct keys. Reaching the same
/// neuron twice after an accepted cut rebuilt the sweep counts twice: that is
/// precisely the topology-tolerant work #161 needs to keep visible.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct VisitCounts {
    /// Every visit attempt.
    pub total: usize,
    /// Hidden-neuron visit attempts.
    pub neurons: usize,
    /// Synapse visit attempts.
    pub synapses: usize,
}

impl VisitCounts {
    fn observe(&mut self, visit: &str) {
        self.total += 1;
        if crate::sweep::parse_synapse_key(visit).is_some() {
            self.synapses += 1;
        } else {
            self.neurons += 1;
        }
    }
}

#[derive(Debug, Default)]
pub(crate) struct ScreenProgress {
    opening: HashSet<String>,
    added: HashSet<String>,
    visits: VisitCounts,
    revisits: VisitCounts,
}

impl ScreenProgress {
    /// Snapshot what the fleet had already screened when the run opened.
    pub(crate) fn new(screens: &[Screened]) -> Self {
        Self {
            opening: screens.iter().map(|s| s.uuid.clone()).collect(),
            added: HashSet::new(),
            visits: VisitCounts::default(),
            revisits: VisitCounts::default(),
        }
    }

    /// Record that the sweep reached this uuid, whatever came of the visit.
    ///
    /// Wider than [`Self::observe`] on purpose (Issue #140): a revisit files no
    /// screen record — a blocked visit files one per epoch by design (#93) —
    /// so without this the run's own re-screening work is invisible to every
    /// reporting surface, and a fully covered creature reads as idle.
    pub(crate) fn visit(&mut self, uuid: &str) {
        self.visits.observe(uuid);
        if self.opening.contains(uuid) {
            self.revisits.observe(uuid);
        }
    }

    /// Every visit attempt this run, including repeats after sweep rebuilds.
    pub(crate) fn visit_counts(&self) -> VisitCounts {
        self.visits
    }

    /// Visit attempts over keys already checked when this run opened.
    pub(crate) fn revisit_counts(&self) -> VisitCounts {
        self.revisits
    }

    /// Record one filed screen record; only a first-ever record counts.
    pub(crate) fn observe(&mut self, uuid: &str) {
        if !self.opening.contains(uuid) {
            self.added.insert(uuid.to_string());
        }
    }

    /// Whether this uuid already carries a screen record the run can see.
    ///
    /// The fleet's history at open, plus whatever this run has filed since.
    /// A visit with nothing to score files a record only when this is `false`
    /// (Issue #93): the record says "the sweep has been here", and repeating it
    /// every pass would grow the shared append-only log without adding a fact.
    pub(crate) fn seen(&self, uuid: &str) -> bool {
        self.opening.contains(uuid) || self.added.contains(uuid)
    }

    /// Distinct UUIDs newly checked so far this run.
    pub(crate) fn count(&self) -> usize {
        self.added.len()
    }
}

/// The zero-progress warning, or `None` when the run advanced coverage.
///
/// A run that adds nothing to the screened set while unchecked visits remain
/// has not done the job it exists to do, whatever else it reported. The line
/// names both figures so a plateau is legible from one run's log rather than
/// only by diffing two commits.
///
/// Both figures count **visits** since Issue #137 — hidden neurons and synapse
/// keys alike — so a run that screened only synapses is progress and says
/// nothing, and a run that screened nothing is measured against every visit
/// still outstanding rather than against the neurons alone.
pub(crate) fn zero_progress_warning(newly_screened: usize, unchecked: usize) -> Option<String> {
    (newly_screened == 0 && unchecked > 0).then(|| {
        format!(
            "no progress: 0 newly checked visit(s) this run while {unchecked} visit(s) \
             remain unchecked"
        )
    })
}

/// What one run tried, kept and rejected (Issue #59).
///
/// Coverage says how much of the creature has been looked at; this says whether
/// looking paid. A reader of the fleet history needs both to tell whether the
/// exploitation strategy of #45 is working.
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Winners {
    /// Sampled winners promoted to full scoring across the run.
    pub screened: usize,
    /// Distinct UUIDs whose own full-corpus delta beat `--min-improvement`.
    pub confirmed: usize,
    /// Hidden neurons removed by accepted winners.
    pub applied: usize,
    /// Confirmed winners still standing in the in-run pool at the end.
    pub carried: usize,
    /// Bundle plans actually scored.
    pub plans: usize,
    /// Plans dropped because a cut in them no longer proposed.
    pub skipped: usize,
    /// Cuts in the largest accepted winner.
    pub best_cuts: usize,
    /// Full-corpus delta of that winner.
    pub best_delta: f64,
    /// Cohort entries dropped to fit the wall-clock budget.
    pub dropped: usize,
    /// Rolling full-corpus scorer cost estimate, milliseconds per creature.
    pub est_ms_per_creature: u64,
}

impl Winners {
    /// Whether this run has anything to report.
    ///
    /// A run that screened nothing renders exactly as it did before Issue #59.
    pub fn has_any(&self) -> bool {
        self.screened > 0 || self.confirmed > 0 || self.applied > 0 || self.plans > 0
    }

    /// One-line log summary.
    pub fn summary(&self) -> String {
        format!(
            "winners {} screened, {} confirmed, {} applied, {} carried",
            self.screened, self.confirmed, self.applied, self.carried
        )
    }

    /// The description lines, each omitted when it has nothing to report.
    fn lines(&self) -> Vec<String> {
        let mut out = Vec::new();
        if self.screened > 0 || self.confirmed > 0 {
            out.push(format!(
                "{:<11}{} screened · {} confirmed · {} applied · {} carried",
                "winners:", self.screened, self.confirmed, self.applied, self.carried
            ));
        }
        if self.plans > 0 {
            let mut line = format!(
                "{:<11}{} plans · best {} cuts (Δ {:+.1e})",
                "bundles:", self.plans, self.best_cuts, self.best_delta
            );
            if self.skipped > 0 {
                line.push_str(&format!(" · {} skipped", self.skipped));
            }
            out.push(line);
        }
        if self.dropped > 0 {
            let est = if self.est_ms_per_creature > 0 {
                format!(
                    " (est {:.0}s/creature)",
                    self.est_ms_per_creature as f64 / 1000.0
                )
            } else {
                String::new()
            };
            out.push(format!(
                "{:<11}{} entries over budget{est}",
                "dropped:", self.dropped
            ));
        }
        out
    }
}

/// Cumulative screening across **every** corpus epoch (Issue #102).
///
/// Current-epoch coverage answers "how far has this sweep got?"; this answers
/// "how much of this creature has the fleet ever looked at?". The two are
/// reported side by side and never merged: folding a screen taken under last
/// week's training data into today's percentage is exactly the misleading
/// `100%` this issue removes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct History {
    /// Hidden neurons on the current incumbent checked in *any* epoch.
    pub checked_ever: usize,
    /// Distinct corpus epochs those checks were spread across.
    ///
    /// Scoped to the same creature as [`Self::checked_ever`], so the two halves
    /// of the rendered line describe one thing: an epoch that reached nothing
    /// still on the creature contributed no check, and is not counted.
    pub epochs: usize,
}

impl History {
    /// Whether there is any history to report.
    pub fn has_any(&self) -> bool {
        self.epochs > 0
    }

    /// The description line, or `None` when nothing was ever checked.
    fn line(&self, checkable: usize) -> Option<String> {
        self.has_any().then(|| {
            let unit = if self.epochs == 1 { "epoch" } else { "epochs" };
            format!(
                "{:<11}{} of {checkable} ever checked across {} corpus {unit}",
                "history:", self.checked_ever, self.epochs
            )
        })
    }
}

/// How many times the razor has been all the way round the creature (#140).
///
/// `100% ever visited` is not `finished`. [`Coverage`] counts **unique** hidden
/// UUIDs visited at least once this epoch, and once that reaches the whole
/// creature it stops moving however hard the fleet is working: a re-screen of
/// an already-visited neuron is real work, but it is not a new uuid, and
/// counting it as one would break the meaning every existing consumer relies
/// on. So repeated sweeps are reported here instead, beside the unique
/// percentage and never inside it.
///
/// A **pass** is one complete sweep: the run visited every hidden neuron on the
/// incumbent, the sweep was exhausted, and it was rebuilt to re-screen the
/// stalest neurons first (Issue #77). Each rebuild files a marker in the
/// learnings store, so the count survives the run that made it.
///
/// A topology change or an accepted cut resets nothing here: the counters move
/// only when a sweep is exhausted, and the epoch total is read from persisted
/// markers rather than from the creature in hand. A **corpus change** does open
/// a new epoch — [`Self::sweeps_completed_epoch`] counts markers filed under the
/// corpus in hand — for the same reason coverage does (Issue #100).
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Passes {
    /// Exhausted-sweep restarts during **this** Ockham invocation (Issue #77).
    ///
    /// The same event `report` counts as `sweepRestarts`, so the two surfaces
    /// are the same figure read from the same journal.
    pub sweep_restarts_run: u64,
    /// Complete passes the fleet has recorded over the current epoch.
    ///
    /// Counted from the pass markers in the learnings store, filed under the
    /// corpus in hand, and re-read at the end of the run rather than counted
    /// forward from its start: several hosts sweep the same creature at once,
    /// so this is the **fleet's** total, and a marker a store fault lost is not
    /// reported as though it had landed. Exact for an epoch that opened after
    /// the markers existed; a **floor** for one already running when this
    /// shipped, because a pass that finished before markers were persisted left
    /// no record to count and is never guessed at (Issue #140).
    pub sweeps_completed_epoch: u64,
    /// 1-based pass currently being worked: `sweeps_completed_epoch + 1`.
    ///
    /// Derived rather than stored independently, so the rendered line can never
    /// claim a pass number the completed count does not support.
    pub current_pass: u64,
    /// Distinct hidden UUIDs this run's sweep reached, revisits included.
    ///
    /// Deliberately **not** `newlyScreened`: that counts first-ever visits, and
    /// on a fully screened creature it is zero however much re-screening the
    /// run did. This is the work figure that keeps moving.
    pub visited_run: usize,
    /// How many of [`Self::visited_run`] the fleet had **already** checked when
    /// this run opened.
    ///
    /// A revisit is useful work and is **not** new unique coverage, so it is
    /// reported as its own number rather than folded into `progress:`. Measured
    /// against the store as the run opened, not against this run's own earlier
    /// batches: what it answers is how much of the run's work went over ground
    /// the fleet had already covered.
    pub revisited_run: usize,
    /// Every visit attempt this run, including repeat visits caused by accepted
    /// cuts rebuilding the sweep (#161).
    #[serde(default)]
    pub visits_run: usize,
    /// Hidden-neuron share of [`Self::visits_run`] (#163).
    #[serde(default)]
    pub neuron_visits_run: usize,
    /// Synapse share of [`Self::visits_run`] (#163).
    #[serde(default)]
    pub synapse_visits_run: usize,
    /// Revisit attempts over keys that were already checked when the run opened.
    #[serde(default)]
    pub revisit_attempts_run: usize,
    /// Hidden-neuron share of [`Self::revisit_attempts_run`].
    #[serde(default)]
    pub neuron_revisits_run: usize,
    /// Synapse share of [`Self::revisit_attempts_run`].
    #[serde(default)]
    pub synapse_revisits_run: usize,
    /// Creature-equivalent work this invocation performed: total visit attempts
    /// divided by the final current-incumbent visit population (#161).
    #[serde(default)]
    pub equivalent_passes_run: f64,
}

impl Passes {
    /// Passes with `current_pass` derived from the completed count.
    pub fn new(
        sweep_restarts_run: u64,
        sweeps_completed_epoch: u64,
        visited_run: usize,
        revisited_run: usize,
    ) -> Self {
        Self {
            sweep_restarts_run,
            sweeps_completed_epoch,
            current_pass: sweeps_completed_epoch + 1,
            visited_run,
            revisited_run,
            visits_run: 0,
            neuron_visits_run: 0,
            synapse_visits_run: 0,
            revisit_attempts_run: 0,
            neuron_revisits_run: 0,
            synapse_revisits_run: 0,
            equivalent_passes_run: 0.0,
        }
    }

    /// Build pass telemetry from attempt counters rather than distinct keys.
    ///
    /// This is the topology-tolerant measurement path (#161): an accepted cut
    /// may rebuild a permutation before `Sweep::exhausted()` ever fires, but it
    /// cannot erase visit attempts already counted here. The strict restart
    /// counters remain beside it, explicitly separate.
    pub fn measured(
        sweep_restarts_run: u64,
        sweeps_completed_epoch: u64,
        visits: VisitCounts,
        revisits: VisitCounts,
        population: usize,
    ) -> Self {
        Self {
            sweep_restarts_run,
            sweeps_completed_epoch,
            current_pass: sweeps_completed_epoch + 1,
            visited_run: visits.total,
            revisited_run: revisits.total,
            visits_run: visits.total,
            neuron_visits_run: visits.neurons,
            synapse_visits_run: visits.synapses,
            revisit_attempts_run: revisits.total,
            neuron_revisits_run: revisits.neurons,
            synapse_revisits_run: revisits.synapses,
            equivalent_passes_run: if population == 0 {
                0.0
            } else {
                visits.total as f64 / population as f64
            },
        }
    }

    /// The description lines: strict sweep counters, topology-tolerant rescan
    /// work, then the neuron/synapse visit split.
    ///
    /// ```text
    /// passes:    3 complete this epoch · 1 this run · pass 4 in progress
    /// visits:    120 hidden neurons visited this run · 40 revisited
    /// ```
    ///
    /// The `passes:` line is always rendered — `0 complete this epoch · pass 1
    /// in progress` is the honest opening state, and a line that appears only
    /// once there is something to boast about cannot be read as a series. The
    /// `visits:` line is omitted when the run reached nothing.
    fn lines(&self) -> Vec<String> {
        let mut out = vec![format!(
            "{:<11}{} strict complete this epoch · {} strict this run · pass {} in progress",
            "passes:", self.sweeps_completed_epoch, self.sweep_restarts_run, self.current_pass
        )];
        if self.visits_run > 0 {
            out.push(format!(
                "{:<11}{:.2} creature-equivalent this run · {} visit attempts",
                "rescan:", self.equivalent_passes_run, self.visits_run
            ));
            if self.synapse_visits_run > 0 {
                out.push(format!(
                    "{:<11}neurons {} ({} revisits) · synapses {} ({} revisits)",
                    "visits:",
                    self.neuron_visits_run,
                    self.neuron_revisits_run,
                    self.synapse_visits_run,
                    self.synapse_revisits_run
                ));
            } else {
                out.push(format!(
                    "{:<11}neurons {} ({} revisits)",
                    "visits:", self.neuron_visits_run, self.neuron_revisits_run
                ));
            }
        } else if self.visited_run > 0 {
            out.push(format!(
                "{:<11}{} hidden neurons visited this run · {} revisited",
                "visits:", self.visited_run, self.revisited_run
            ));
        }
        out
    }
}

/// Every screen record the store holds, indexed across all epochs (Issue #102).
///
/// Built from the **unfiltered** load, before
/// [`crate::learnings::current_epoch_screens`] narrows it to the corpus in
/// hand, so the cumulative figures survive an epoch change that resets the
/// current-epoch percentage to zero.
#[derive(Debug, Clone, Default)]
pub struct ScreenHistory {
    /// UUIDs reached, indexed by the epoch that reached them.
    ///
    /// Keyed by epoch rather than flattened, so [`Self::over`] can report a
    /// count of epochs that is about the same creature as the count of checks
    /// beside it.
    by_epoch: HashMap<Option<String>, HashSet<String>>,
}

impl ScreenHistory {
    /// Index every record: which UUIDs were reached, under which epoch.
    ///
    /// A record written before the corpus identity was recorded (#76) carries
    /// `None`, and counts as one epoch of its own — "an epoch we cannot name"
    /// is still an epoch, and dropping it would understate the history.
    pub fn new(screens: &[crate::learnings::Screened]) -> Self {
        let mut history = Self::default();
        history.merge(screens);
        history
    }

    /// Fold in more records — the ones this run filed after the index was built.
    ///
    /// Without this the cumulative figures would be frozen at the moment the
    /// store was read, and a run's own epoch would be missing from its own
    /// history line.
    pub fn merge(&mut self, screens: &[crate::learnings::Screened]) {
        for s in screens {
            self.by_epoch
                .entry(s.corpus_identity.clone())
                .or_default()
                .insert(s.uuid.clone());
        }
    }

    /// Cumulative coverage of `creature` — visits reached in any epoch.
    ///
    /// The same population current coverage counts (#137): every hidden neuron
    /// and every synapse visit key the incumbent still carries. A synapse
    /// missing from it would leave that edge's history invisible to the
    /// cumulative line and to unchecked-first selection, which reads the same
    /// still-present set.
    ///
    /// Scoped to the creature in hand for the same reason current coverage is
    /// (#37): a record for a visit that has been pruned away describes a
    /// creature that no longer exists. The epoch count is scoped the same way,
    /// so the rendered line cannot say "0 ever checked across 7 epochs" — an
    /// epoch that reached nothing still on the creature reached nothing to
    /// report.
    pub fn over(&self, creature: &CreatureExport) -> History {
        let population = visit_population(creature);
        let mut checked: HashSet<&str> = HashSet::new();
        let mut epochs = 0;
        for uuids in self.by_epoch.values() {
            let reached = uuids
                .iter()
                .filter(|u| population.contains(u.as_str()))
                .map(String::as_str)
                .collect::<Vec<&str>>();
            if reached.is_empty() {
                continue;
            }
            epochs += 1;
            checked.extend(reached);
        }
        History {
            checked_ever: checked.len(),
            epochs,
        }
    }
}

/// The commit-description artefact: coverage, plus the run's winner economics.
///
/// [`Coverage`] is flattened, so a consumer that deserialises `coverage.json`
/// straight into [`Coverage`] keeps working — serde ignores the extra keys.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CoverageReport {
    /// Screening coverage of the final incumbent.
    #[serde(flatten)]
    pub coverage: Coverage,
    /// Distinct hidden UUIDs **this run** checked for the first time (#77).
    ///
    /// Coverage is cumulative fleet state, so a run that advanced nothing
    /// renders exactly like one that advanced a full batch. This is the
    /// per-run figure beside it: two consecutive commits now show whether the
    /// fleet is moving. `#[serde(default)]` so a pre-#77 `coverage.json` still
    /// deserialises, and the key keeps its name so anything already reading it
    /// keeps working — what widened in #93 is what counts as reaching a uuid.
    #[serde(default)]
    pub newly_screened: usize,
    /// What the run tried and kept; absent when nothing was screened.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub winners: Option<Winners>,
    /// Corpus identity these figures were measured against (Issue #100).
    ///
    /// The screening epoch the percentage belongs to: `100%` means the sweep is
    /// complete for *this* corpus, and the next extension of the training data
    /// opens a new epoch at zero. Every run that writes `coverage.json` names
    /// its corpus, because those files are written only where there is a screen
    /// store to have measured coverage against. `None` on a report built by
    /// [`Self::new`] and on any `coverage.json` written before this field
    /// existed — so an older artefact still deserialises and an older consumer
    /// still ignores it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub corpus_identity: Option<String>,
    /// Cumulative coverage across every epoch (Issue #102); absent with no store.
    ///
    /// Kept beside the current-epoch figures rather than inside them: the
    /// percentage above is this corpus only, and this says what the fleet has
    /// ever reached. `#[serde(default)]` so an artefact written before this
    /// field existed still deserialises.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub history: Option<History>,
    /// Repeated complete sweeps over the creature (Issue #140); absent with no
    /// store.
    ///
    /// The counter that keeps moving once the unique percentage above reaches
    /// 100%: without it a run re-screening the stalest half of a finished
    /// creature is indistinguishable, from the commit description, from one
    /// stuck at the tail of its first pass. `#[serde(default)]` so an artefact
    /// written before this field existed still deserialises.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub passes: Option<Passes>,
}

impl CoverageReport {
    /// Coverage with no per-run progress, no winner figures and no epoch.
    pub fn new(coverage: Coverage) -> Self {
        Self {
            coverage,
            newly_screened: 0,
            winners: None,
            corpus_identity: None,
            history: None,
            passes: None,
        }
    }

    /// The full commit-description block: coverage, epoch, progress, winners.
    ///
    /// The `epoch:` line is rendered inside the coverage block itself (#102),
    /// directly under the percentage it qualifies, and is omitted when the
    /// report names no epoch — so a report built by [`Self::new`] renders as it
    /// did before #100. The `progress:` line is always rendered, zero included
    /// — a plateau is only visible by reading two consecutive commits if the
    /// figure is there in both. The `passes:` line follows it whenever the run
    /// had a screen store to count markers from (Issue #140), saying how many
    /// complete sweeps this epoch has had and which pass is in progress. The
    /// `history:` line is omitted when the store holds no records, and never
    /// contributes to the percentage above it.
    pub fn description(&self, candidates: usize) -> String {
        let mut out = self
            .coverage
            .description(candidates, self.corpus_identity.as_deref());
        out.push_str(&format!(
            "\n{:<11}{} newly checked this run",
            "progress:", self.newly_screened
        ));
        for line in self.passes.iter().flat_map(Passes::lines) {
            out.push('\n');
            out.push_str(&line);
        }
        if let Some(line) = self
            .history
            .as_ref()
            .and_then(|h| h.line(self.coverage.checkable))
        {
            out.push('\n');
            out.push_str(&line);
        }
        for line in self.winners.iter().flat_map(Winners::lines) {
            out.push('\n');
            out.push_str(&line);
        }
        out
    }
}

/// Write `coverage.txt` and `coverage.json` into `dir` (Issues #40, #59).
///
/// The stable contract GRQ consumes: prose for the commit description, and the
/// same figures as JSON for anything that would otherwise parse the prose.
/// Both are written or the error names the file that failed — the caller warns
/// rather than failing the run, matching the learnings-cache rule.
pub fn write_files(dir: &Path, report: &CoverageReport, candidates: usize) -> Result<(), String> {
    std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    let text = dir.join(COVERAGE_TEXT_FILE);
    std::fs::write(&text, format!("{}\n", report.description(candidates)))
        .map_err(|e| format!("{}: {e}", text.display()))?;
    let json =
        serde_json::to_string_pretty(report).map_err(|e| format!("{COVERAGE_JSON_FILE}: {e}"))?;
    let path = dir.join(COVERAGE_JSON_FILE);
    std::fs::write(&path, format!("{json}\n")).map_err(|e| format!("{}: {e}", path.display()))
}

/// Hidden neuron UUIDs of `creature`, in listed order.
///
/// The neuron half of the coverage population. Deliberately narrower than the
/// neuron list [`crate::sweep::present_visits`] builds, which also names
/// inputs, outputs and constants: those are not swept, so a stray record for
/// one must not raise coverage.
fn hidden_uuids(creature: &CreatureExport) -> Vec<&str> {
    creature
        .neurons
        .iter()
        .filter(|n| n.neuron_type == "hidden")
        .map(|n| n.uuid.as_str())
        .collect()
}

/// One [`crate::sweep::synapse_key`] per ordered endpoint pair (Issue #137).
///
/// The edge half of the coverage population, deduplicated exactly as the sweep
/// pool builds it (#135): a repeated pair is one visit, never two.
fn synapse_visits(creature: &CreatureExport) -> HashSet<String> {
    creature
        .synapses
        .iter()
        .map(|s| crate::sweep::synapse_key(&s.from_uuid, &s.to_uuid))
        .collect()
}

/// Every visit `creature` puts in the coverage denominator (Issue #137).
///
/// The single definition of that population: [`coverage`] and
/// [`ScreenHistory::over`] both count against it, so the current-epoch figures
/// and the cumulative ones can never disagree about what is on the creature.
fn visit_population(creature: &CreatureExport) -> HashSet<String> {
    let mut population = synapse_visits(creature);
    population.extend(hidden_uuids(creature).into_iter().map(str::to_string));
    population
}

/// Count coverage of `creature` from `screens`; `tagged` is counted, not excluded.
///
/// Every hidden neuron is in the denominator, tagged ones included (#74) —
/// `tagged` only says how many of them carry tags — and since Issue #137 every
/// synapse the incumbent carries is in it beside them, keyed by
/// [`crate::sweep::synapse_key`]. A typed edge counts too: the sweep visits it,
/// the visit is blocked, and the blocked record is what makes it *checked*, so
/// a creature full of typed edges reaches a complete sweep rather than sitting
/// permanently short of one.
///
/// `cut` is the hidden neurons removed this run, carried through rather than
/// derived, because the creature in hand no longer holds them.
pub fn coverage(
    creature: &CreatureExport,
    tagged: &HashSet<String>,
    screens: &[Screened],
    cut: usize,
) -> Coverage {
    let hidden_uuids = hidden_uuids(creature);
    let hidden = hidden_uuids.len();
    let tagged_hidden = hidden_uuids
        .iter()
        .filter(|uuid| tagged.contains(**uuid))
        .count();
    let hidden_uuids: HashSet<&str> = hidden_uuids.into_iter().collect();
    let synapse_visits = synapse_visits(creature);
    let synapses = synapse_visits.len();
    // A map of the screened visit keys, so a visit screened many times — by
    // this host or another — still counts once. The value is "every record so
    // far was a skipped visit": one real screen anywhere in the fleet's history
    // clears it permanently, so `blocked` never over-reports (Issue #93).
    let mut checked: HashMap<&str, CheckedUuid<'_>> = HashMap::new();
    for s in screens.iter().filter(|s| {
        hidden_uuids.contains(s.uuid.as_str()) || synapse_visits.contains(s.uuid.as_str())
    }) {
        checked.entry(s.uuid.as_str()).or_default().observe(s);
    }
    let mut blocked_by_reason = BlockedBreakdown::default();
    for reason in checked.values().filter_map(CheckedUuid::blocked_reason) {
        blocked_by_reason.add(reason);
    }
    let synapses_checked = checked
        .keys()
        .filter(|key| synapse_visits.contains(**key))
        .count();
    Coverage {
        hidden,
        tagged: tagged_hidden,
        checkable: hidden + synapses,
        checked: checked.len(),
        synapses,
        synapses_checked,
        blocked: blocked_by_reason.total(),
        blocked_by_reason,
        cut,
    }
}

/// What one hidden uuid's screen records add up to.
///
/// A uuid is blocked only when **every** record for it is a skipped visit: one
/// real screen anywhere in the fleet's history clears it permanently, so
/// `blocked` never over-reports (Issue #93). The reason reported is the one on
/// the **latest** such record — the razor's current answer for this neuron, not
/// an answer from before the structure around it changed.
#[derive(Debug)]
struct CheckedUuid<'a> {
    only_skips: bool,
    latest: Option<&'a Screened>,
}

impl Default for CheckedUuid<'_> {
    fn default() -> Self {
        Self {
            only_skips: true,
            latest: None,
        }
    }
}

impl<'a> CheckedUuid<'a> {
    fn observe(&mut self, record: &'a Screened) {
        self.only_skips &= record.is_skipped();
        if record.is_skipped()
            && self
                .latest
                .is_none_or(|held| Self::supersedes(record, held))
        {
            self.latest = Some(record);
        }
    }

    /// Whether `candidate` is the fresher record of the two.
    ///
    /// Ordered on the record's own fields — time, then host, then reason — so
    /// two hosts filing in the same second still produce the same breakdown on
    /// every machine that reads them, whatever order the files were read in.
    fn supersedes(candidate: &Screened, held: &Screened) -> bool {
        let key = |s: &Screened| {
            (
                s.unix_secs,
                s.host.clone(),
                s.blocked_category().map(BlockedReason::code).unwrap_or(""),
            )
        };
        key(candidate) > key(held)
    }

    fn blocked_reason(&self) -> Option<BlockedReason> {
        if !self.only_skips {
            return None;
        }
        Some(
            self.latest
                .and_then(Screened::blocked_category)
                .unwrap_or(BlockedReason::Unrecorded),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixtures::{creature, neuron, synapse, typed_synapse};
    use crate::learnings::{SCREENS_FORMAT_VERSION, ScreenOutcomeKind};

    /// The neuron half of the visit population alone: `n` parallel hidden
    /// IDENTITY neurons `h0..h{n-1}`, and no edges at all.
    ///
    /// Edge-free on purpose since Issue #137 widened the population: a test
    /// about hidden-neuron arithmetic — a tagged neuron staying in the
    /// denominator, a percentage, the rendered block for a creature with no
    /// synapse visits — must measure that half and nothing else.
    /// [`hidden_creature`] is the wired form, and the synapse tests use it.
    fn neurons_only(n: usize) -> CreatureExport {
        let mut neurons: Vec<_> = (0..n)
            .map(|i| neuron("hidden", &format!("h{i}"), 0.0, Some("IDENTITY")))
            .collect();
        neurons.push(neuron("output", "output-0", 0.0, Some("IDENTITY")));
        creature(1, 1, neurons, Vec::new())
    }

    /// [`neurons_only`] wired up: `input-0 → h{i} → output-0` for every neuron.
    ///
    /// `2n` ordered endpoint pairs, so the visit population is `n + 2n` (#137).
    fn hidden_creature(n: usize) -> CreatureExport {
        let mut c = neurons_only(n);
        for i in 0..n {
            c.synapses.push(synapse("input-0", &format!("h{i}"), 1.0));
            c.synapses.push(synapse(&format!("h{i}"), "output-0", 1.0));
        }
        c
    }

    /// The visit key of the edge `from`→`to`, as the sweep pool builds it.
    fn edge(from: &str, to: &str) -> String {
        crate::sweep::synapse_key(from, to)
    }

    fn screen(uuid: &str, unix_secs: u64) -> Screened {
        Screened {
            blocked_reason: Default::default(),
            version: SCREENS_FORMAT_VERSION,
            uuid: uuid.into(),
            kind: "ablation".into(),
            outcome: ScreenOutcomeKind::Loser,
            unix_secs,
            host: "GRQ-1".into(),
            corpus_identity: Some("corp".into()),
        }
    }

    /// A visit the razor could propose nothing for (Issue #93).
    fn visit(uuid: &str, unix_secs: u64) -> Screened {
        Screened {
            kind: crate::learnings::SCREEN_KIND_SKIPPED.into(),
            ..screen(uuid, unix_secs)
        }
    }

    /// A blocked visit carrying the reason that stopped it (Issue #103).
    fn blocked(uuid: &str, unix_secs: u64, reason: BlockedReason) -> Screened {
        Screened {
            blocked_reason: Some(reason),
            ..visit(uuid, unix_secs)
        }
    }

    fn tags(uuids: &[&str]) -> HashSet<String> {
        uuids.iter().map(|u| (*u).to_string()).collect()
    }

    /// Rewritten for Issue #74: tagged neurons stay in the denominator and are
    /// still reported separately.
    #[test]
    fn tagged_neurons_stay_in_the_denominator_and_are_reported_separately() {
        let creature = neurons_only(10);
        let screens = [screen("h2", 1), screen("h3", 2), screen("h4", 3)];
        let cov = coverage(&creature, &tags(&["h0", "h1"]), &screens, 0);
        assert_eq!(cov.hidden, 10);
        assert_eq!(cov.tagged, 2);
        assert_eq!(cov.checkable, 10, "every hidden neuron is checkable");
        assert_eq!(cov.checked, 3);
        assert_eq!(cov.percent(), 30.0);
    }

    /// The denominator is the whole hidden count for every input, so the
    /// percentage can never be inflated by deducting tagged neurons.
    #[test]
    fn checkable_equals_hidden_however_many_neurons_are_tagged() {
        let creature = neurons_only(6);
        for tagged in [
            vec![],
            vec!["h0"],
            vec!["h0", "h1", "h2"],
            vec!["h0", "h1", "h2", "h3", "h4", "h5"],
        ] {
            let cov = coverage(&creature, &tags(&tagged), &[], 0);
            assert_eq!(cov.checkable, cov.hidden, "tagged: {tagged:?}");
            assert_eq!(cov.checkable, 6, "tagged: {tagged:?}");
            assert_eq!(cov.tagged, tagged.len(), "tagged: {tagged:?}");
        }
    }

    /// The detector named in Issue #74: it fails loudly if the denominator
    /// grows without `checked` also counting tagged UUIDs, or vice versa.
    #[test]
    fn an_all_tagged_fully_screened_creature_reports_one_hundred_percent() {
        let creature = neurons_only(4);
        let screens = [
            screen("h0", 1),
            screen("h1", 2),
            screen("h2", 3),
            screen("h3", 4),
        ];
        let cov = coverage(&creature, &tags(&["h0", "h1", "h2", "h3"]), &screens, 0);
        assert_eq!(cov.hidden, 4);
        assert_eq!(cov.tagged, 4);
        assert_eq!(cov.checkable, 4);
        assert_eq!(cov.checked, 4);
        assert_eq!(cov.percent(), 100.0);
    }

    #[test]
    fn screens_for_departed_uuids_raise_neither_checked_nor_hidden() {
        let creature = neurons_only(4);
        let screens = [screen("h0", 1), screen("gone-a", 2), screen("gone-b", 3)];
        let cov = coverage(&creature, &HashSet::new(), &screens, 2);
        assert_eq!(cov.hidden, 4, "pruned neurons are not on the incumbent");
        assert_eq!(cov.checked, 1, "only still-present UUIDs are checked");
        assert_eq!(cov.cut, 2);
    }

    #[test]
    fn a_uuid_screened_many_times_counts_once() {
        let creature = neurons_only(4);
        let screens = [screen("h0", 1), screen("h0", 2), screen("h0", 3)];
        let cov = coverage(&creature, &HashSet::new(), &screens, 0);
        assert_eq!(cov.checked, 1);
        assert_eq!(cov.percent(), 25.0);
    }

    /// Rewritten for Issue #74: a screened tagged uuid raises `checked`.
    #[test]
    fn a_screened_tagged_uuid_counts_as_checked() {
        let creature = neurons_only(4);
        let screens = [screen("h0", 1), screen("h1", 2)];
        let cov = coverage(&creature, &tags(&["h1"]), &screens, 0);
        assert_eq!(cov.checkable, 4);
        assert_eq!(cov.checked, 2, "a tagged uuid is inside the coverage set");
        assert_eq!(cov.percent(), 50.0);
    }

    /// Issue #93: a neuron the razor cannot propose a cut for is checked — the
    /// sweep has been there — and reported as blocked rather than as a screen.
    #[test]
    fn a_visit_with_no_candidate_is_checked_and_reported_as_blocked() {
        let creature = neurons_only(4);
        let screens = [screen("h0", 1), visit("h1", 2), visit("h2", 3)];
        let cov = coverage(&creature, &HashSet::new(), &screens, 0);
        assert_eq!(cov.checked, 3, "every visited uuid counts as checked");
        assert_eq!(cov.blocked, 2, "two of them were never scored");
        assert_eq!(cov.percent(), 75.0);
        assert_eq!(cov.unchecked(), 1, "only h3 was never reached");
    }

    /// Issue #103: the acceptance criterion. Whatever the records say, the
    /// reason counts add up to the blocked total — the breakdown is a partition
    /// of the blocked population, never a sample of it.
    #[test]
    fn the_reason_counts_sum_to_the_blocked_total() {
        let creature = neurons_only(6);
        let screens = [
            screen("h0", 1),
            blocked("h1", 2, BlockedReason::AggregateSquash),
            blocked("h2", 3, BlockedReason::AggregateSquash),
            blocked("h3", 4, BlockedReason::MissingActivation),
            visit("h4", 5),
        ];
        let cov = coverage(&creature, &HashSet::new(), &screens, 0);
        assert_eq!(cov.blocked, 4);
        assert_eq!(cov.blocked_by_reason.total(), cov.blocked);
        assert_eq!(cov.blocked_by_reason.aggregate_squash, 2);
        assert_eq!(cov.blocked_by_reason.missing_activation, 1);
        assert_eq!(
            cov.blocked_by_reason.unrecorded, 1,
            "a visit filed before #103 is its own category, never dropped"
        );
        assert_eq!(
            cov.blocked_by_reason.dominant(),
            Some((BlockedReason::AggregateSquash, 2))
        );
    }

    /// The reason reported for a uuid is the razor's *latest* answer for it:
    /// the structure around a neuron changes, and last week's reason is not a
    /// statement about the creature in hand.
    #[test]
    fn the_freshest_record_decides_the_reason_whatever_order_it_was_read_in() {
        let creature = neurons_only(1);
        let old = blocked("h0", 1, BlockedReason::MissingActivation);
        let new = blocked("h0", 9, BlockedReason::AggregateSquash);
        for screens in [
            vec![old.clone(), new.clone()],
            vec![new.clone(), old.clone()],
        ] {
            let cov = coverage(&creature, &HashSet::new(), &screens, 0);
            assert_eq!(cov.blocked, 1);
            assert_eq!(cov.blocked_by_reason.aggregate_squash, 1, "{screens:?}");
            assert_eq!(cov.blocked_by_reason.missing_activation, 0, "{screens:?}");
        }
    }

    /// A screened uuid is not blocked, so its records contribute no reason —
    /// the breakdown can never exceed the blocked total it splits.
    #[test]
    fn a_uuid_with_one_real_screen_contributes_no_reason() {
        let creature = neurons_only(2);
        let screens = [
            blocked("h0", 1, BlockedReason::AggregateSquash),
            screen("h0", 2),
            blocked("h1", 3, BlockedReason::AggregateSquash),
        ];
        let cov = coverage(&creature, &HashSet::new(), &screens, 0);
        assert_eq!(cov.blocked, 1);
        assert_eq!(cov.blocked_by_reason.aggregate_squash, 1);
    }

    /// The commit description carries the breakdown, so the largest category to
    /// attack is legible without opening the JSON.
    #[test]
    fn the_description_breaks_the_blocked_line_down_by_reason() {
        let creature = neurons_only(4);
        let screens = [
            blocked("h0", 1, BlockedReason::AggregateSquash),
            blocked("h1", 2, BlockedReason::AggregateSquash),
            blocked("h2", 3, BlockedReason::UnsafeTopology),
        ];
        let cov = coverage(&creature, &HashSet::new(), &screens, 0);
        let text = cov.description(100, None);
        assert!(
            text.contains("blocked:   3 checked with no cut proposed"),
            "{text}"
        );
        assert!(
            text.contains("reasons:   aggregate-squash 2 (66.7%) · unsafe-topology 1 (33.3%)"),
            "{text}"
        );
    }

    #[test]
    fn the_description_omits_the_reasons_line_when_nothing_is_blocked() {
        let cov = coverage(&neurons_only(4), &HashSet::new(), &[screen("h0", 1)], 0);
        assert!(!cov.description(100, None).contains("reasons:"));
    }

    /// `blocked` is about the uuid, not the record: one real screen anywhere in
    /// the fleet's history says the razor could propose something for it.
    #[test]
    fn one_real_screen_clears_blocked_however_many_visits_surround_it() {
        let creature = neurons_only(2);
        for screens in [
            vec![visit("h0", 1), screen("h0", 2)],
            vec![screen("h0", 1), visit("h0", 2)],
            vec![visit("h0", 1), screen("h0", 2), visit("h0", 3)],
        ] {
            let cov = coverage(&creature, &HashSet::new(), &screens, 0);
            assert_eq!(cov.checked, 1, "{screens:?}");
            assert_eq!(cov.blocked, 0, "{screens:?}");
        }
    }

    #[test]
    fn a_blocked_uuid_no_longer_on_the_creature_counts_for_nothing() {
        let creature = neurons_only(2);
        let screens = [visit("gone", 1), visit("h0", 2)];
        let cov = coverage(&creature, &HashSet::new(), &screens, 1);
        assert_eq!(cov.checked, 1);
        assert_eq!(
            cov.blocked, 1,
            "h0 is blocked; the departed uuid counts for neither figure"
        );
    }

    /// Issue #100: coverage counts the epoch of the corpus in hand. A creature
    /// screened to 100% under the previous corpus opens the new epoch at
    /// `0 / hidden`, blocked and known-failure visits included — every neuron is
    /// eligible to be visited again — while the records themselves survive.
    #[test]
    fn a_corpus_change_opens_a_new_epoch_at_zero_coverage() {
        let creature = neurons_only(4);
        let old_epoch = vec![
            screen("h0", 1),
            screen("h1", 2),
            visit("h2", 3),
            Screened {
                kind: crate::learnings::SCREEN_KIND_KNOWN_FAILURE.into(),
                ..screen("h3", 4)
            },
        ];
        let before = coverage(&creature, &HashSet::new(), &old_epoch, 0);
        assert_eq!(before.percent(), 100.0, "the sweep finished that corpus");

        let new_epoch = crate::learnings::current_epoch_screens(old_epoch.clone(), "corp-next");
        let after = coverage(&creature, &HashSet::new(), &new_epoch, 0);
        assert_eq!(after.checkable, 4, "every hidden neuron is checkable again");
        assert_eq!(after.checked, 0);
        assert_eq!(
            after.blocked, 0,
            "a blocked visit was blocked under that corpus"
        );
        assert_eq!(after.percent(), 0.0);
        assert_eq!(after.unchecked(), 4);

        // Same corpus, same epoch: a restart against unchanged training data
        // keeps everything it had.
        let same = crate::learnings::current_epoch_screens(old_epoch, "corp");
        assert_eq!(coverage(&creature, &HashSet::new(), &same, 0), before);
    }

    /// Intended behaviour, not a bug: evolution adds hidden neurons, they start
    /// unchecked, and the percentage falls. Coverage describes the creature in
    /// front of us — do not "fix" this by remembering departed UUIDs.
    #[test]
    fn newly_evolved_neurons_lower_the_percentage() {
        let screens = [screen("h0", 1), screen("h1", 2)];
        let before = coverage(&neurons_only(2), &HashSet::new(), &screens, 0);
        assert_eq!(before.percent(), 100.0);
        let after = coverage(&neurons_only(4), &HashSet::new(), &screens, 0);
        assert_eq!(after.percent(), 50.0, "two new neurons start unchecked");
        assert!(after.percent() < before.percent());
    }

    /// Rewritten for Issue #74: an all-tagged creature has a real denominator
    /// now, so the zero-denominator guard is the no-hidden-neurons case below.
    #[test]
    fn nothing_checked_yields_zero_percent_without_panicking() {
        let creature = neurons_only(2);
        let cov = coverage(&creature, &tags(&["h0", "h1"]), &[], 0);
        assert_eq!(cov.checkable, 2);
        assert_eq!(cov.checked, 0);
        assert_eq!(cov.percent(), 0.0);
        assert_eq!(
            cov.summary(),
            "sweep 0/2 checked (0.0% of epoch), 0 cut, 2 tagged"
        );
    }

    /// The only zero denominator left: no hidden neurons at all.
    #[test]
    fn an_empty_denominator_yields_zero_percent_without_panicking() {
        let cov = coverage(&neurons_only(0), &HashSet::new(), &[], 0);
        assert_eq!(cov.checkable, 0);
        assert_eq!(cov.percent(), 0.0);
        assert_eq!(cov.summary(), "sweep 0/0 checked (0.0% of epoch), 0 cut");
    }

    #[test]
    fn percent_never_exceeds_one_hundred() {
        let cov = Coverage {
            blocked_by_reason: Default::default(),
            hidden: 3,
            tagged: 0,
            checkable: 3,
            checked: 9,
            synapses: 0,
            synapses_checked: 0,
            blocked: 0,
            cut: 0,
        };
        assert_eq!(cov.percent(), 100.0);
    }

    #[test]
    fn summary_omits_the_tagged_clause_when_nothing_is_tagged() {
        let creature = neurons_only(4);
        let screens = [screen("h0", 1)];
        let cov = coverage(&creature, &HashSet::new(), &screens, 3);
        assert_eq!(cov.summary(), "sweep 1/4 checked (25.0% of epoch), 3 cut");
        assert!(!cov.summary().contains("tagged"));
    }

    #[test]
    fn summary_appends_the_tagged_clause_when_neurons_carry_tags() {
        let creature = neurons_only(6);
        let screens = [screen("h2", 1), screen("h3", 2)];
        let cov = coverage(&creature, &tags(&["h0"]), &screens, 1);
        assert_eq!(
            cov.summary(),
            "sweep 2/6 checked (33.3% of epoch), 1 cut, 1 tagged"
        );
    }

    /// The fleet-scale example from Issue #40, rendered exactly.
    fn fleet_coverage() -> Coverage {
        Coverage {
            blocked_by_reason: Default::default(),
            hidden: 5013,
            tagged: 42,
            checkable: 5013,
            checked: 1204,
            synapses: 0,
            synapses_checked: 0,
            blocked: 0,
            cut: 7,
        }
    }

    #[test]
    fn the_description_block_renders_exactly_as_grq_will_paste_it() {
        assert_eq!(
            fleet_coverage().description(100, None),
            concat!(
                "🪒 Ockham neuron screening coverage\n",
                "sweep:     1204 of 5013 hidden (24.0% of epoch)\n",
                "cut:       7 this run\n",
                "unchecked: 3809 remaining this epoch (~39 runs at 100/run)\n",
                "tagged:    42 carry tags, screened like any other"
            )
        );
    }

    /// Thousands of fleet commits carry this block: it must never say the
    /// tagged neurons were skipped or are never pruned (Issue #74), and since
    /// Issue #87 it must not declare a tagged cut either — tags are
    /// informational, so cutting one is reported by `cut:` like any other.
    #[test]
    fn the_rendered_artefacts_never_call_tagged_neurons_skipped_or_declared() {
        let cov = fleet_coverage();
        for text in [cov.description(100, None), cov.summary()] {
            assert!(!text.contains("skipped"), "{text}");
            assert!(!text.contains("never pruned"), "{text}");
            assert!(!text.contains("outside the denominator"), "{text}");
            assert!(!text.contains("declared:"), "{text}");
            assert!(!text.contains("provenance"), "{text}");
        }
    }

    /// Issue #93: the block says how much of `checked` was reached by a visit
    /// with nothing to score, so a rising percentage never claims a screen that
    /// never happened.
    #[test]
    fn the_description_reports_the_blocked_share_of_the_checked() {
        let cov = Coverage {
            blocked: 412,
            ..fleet_coverage()
        };
        assert_eq!(
            cov.description(100, None),
            concat!(
                "🪒 Ockham neuron screening coverage\n",
                "sweep:     1204 of 5013 hidden (24.0% of epoch)\n",
                "cut:       7 this run\n",
                "unchecked: 3809 remaining this epoch (~39 runs at 100/run)\n",
                "blocked:   412 checked with no cut proposed\n",
                "tagged:    42 carry tags, screened like any other"
            )
        );
        assert_eq!(
            cov.percent(),
            fleet_coverage().percent(),
            "blocked neurons are reported beside the percentage, never deducted"
        );
    }

    #[test]
    fn the_description_omits_the_blocked_line_when_nothing_is_blocked() {
        let block = fleet_coverage().description(100, None);
        assert!(!block.contains("blocked:"), "{block}");
    }

    #[test]
    fn the_description_omits_the_tagged_line_when_nothing_is_tagged() {
        let cov = Coverage {
            tagged: 0,
            ..fleet_coverage()
        };
        let block = cov.description(100, None);
        assert!(!block.contains("tagged"), "{block}");
        assert!(
            block.ends_with("unchecked: 3809 remaining this epoch (~39 runs at 100/run)"),
            "{block}"
        );
    }

    /// `--candidates 0` must not produce a division by zero — the clause goes.
    /// A single remaining run reads `~1 run`, not `~1 runs`.
    #[test]
    fn the_last_remaining_run_is_singular() {
        let cov = Coverage {
            blocked_by_reason: Default::default(),
            hidden: 4,
            tagged: 0,
            checkable: 4,
            checked: 2,
            synapses: 0,
            synapses_checked: 0,
            blocked: 0,
            cut: 0,
        };
        assert!(
            cov.description(100, None)
                .ends_with("unchecked: 2 remaining this epoch (~1 run at 100/run)"),
            "{}",
            cov.description(100, None)
        );
    }

    #[test]
    fn a_zero_batch_size_drops_the_runs_clause_rather_than_rendering_inf() {
        let block = fleet_coverage().description(0, None);
        assert!(
            block.contains("unchecked: 3809 remaining this epoch\n"),
            "{block}"
        );
        assert!(!block.contains("runs at"), "{block}");
        assert!(!block.contains("inf") && !block.contains("NaN"), "{block}");
    }

    #[test]
    fn complete_coverage_drops_the_runs_clause_because_nothing_is_left() {
        let cov = Coverage {
            blocked_by_reason: Default::default(),
            hidden: 4,
            tagged: 0,
            checkable: 4,
            checked: 4,
            synapses: 0,
            synapses_checked: 0,
            blocked: 0,
            cut: 1,
        };
        assert_eq!(
            cov.description(100, None),
            concat!(
                "🪒 Ockham neuron screening coverage\n",
                "sweep:     4 of 4 hidden (100.0% of epoch)\n",
                "cut:       1 this run\n",
                "unchecked: 0 remaining — sweep complete for this epoch"
            )
        );
    }

    #[test]
    fn more_records_than_checkable_neurons_never_renders_a_negative_remainder() {
        let cov = Coverage {
            blocked_by_reason: Default::default(),
            hidden: 3,
            tagged: 0,
            checkable: 3,
            checked: 9,
            synapses: 0,
            synapses_checked: 0,
            blocked: 0,
            cut: 0,
        };
        assert_eq!(cov.unchecked(), 0);
        assert!(
            cov.description(10, None)
                .contains("unchecked: 0 remaining — sweep complete for this epoch")
        );
    }

    #[test]
    fn both_files_are_written_and_the_json_deserialises_back_into_coverage() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join("out");
        let cov = fleet_coverage();
        write_files(&dir, &CoverageReport::new(cov), 100).unwrap();

        let text = std::fs::read_to_string(dir.join(COVERAGE_TEXT_FILE)).unwrap();
        assert_eq!(
            text,
            format!("{}\n", CoverageReport::new(cov).description(100))
        );

        let json = std::fs::read_to_string(dir.join(COVERAGE_JSON_FILE)).unwrap();
        let back: Coverage = serde_json::from_str(&json).unwrap();
        assert_eq!(back, cov, "the machine-readable contract must round-trip");
        assert!(json.contains("\"checkable\": 5013"), "{json}");
    }

    /// Issue #87: the declaration key is gone from what Ockham writes, and a
    /// `coverage.json` written while it existed still deserialises — the key is
    /// simply ignored rather than failing the read.
    #[test]
    fn the_declaration_key_is_neither_written_nor_required_when_reading() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join("out");
        write_files(&dir, &CoverageReport::new(fleet_coverage()), 100).unwrap();

        let json = std::fs::read_to_string(dir.join(COVERAGE_JSON_FILE)).unwrap();
        assert!(!json.contains("taggedCut"), "{json}");

        let with_key = r#"{"hidden":5013,"tagged":42,"checkable":5013,"checked":1204,"cut":7,
            "taggedCut":3}"#;
        let old: Coverage = serde_json::from_str(with_key).unwrap();
        assert_eq!(
            old,
            fleet_coverage(),
            "an artefact from the declaring era still reads as ordinary coverage"
        );
    }

    /// Issue #93: the fleet runs mixed versions against one shared cache, so a
    /// `coverage.json` written before `blocked` existed must still read.
    /// A `coverage.json` written before #103 still reads, as no reasons — an
    /// older artefact must never fail the parse of a newer reader.
    #[test]
    fn a_pre_103_coverage_json_reads_as_no_blocked_reasons() {
        let pre_103 =
            r#"{"hidden":5013,"tagged":42,"checkable":5013,"checked":1204,"blocked":412,"cut":7}"#;
        let old: Coverage = serde_json::from_str(pre_103).unwrap();
        assert_eq!(old.blocked, 412);
        assert_eq!(
            old.blocked_by_reason,
            crate::blocked::BlockedBreakdown::default()
        );
        assert_eq!(
            old.blocked_by_reason.total(),
            0,
            "an unrecorded breakdown reports nothing rather than guessing"
        );
    }

    #[test]
    fn a_pre_93_coverage_json_reads_as_nothing_blocked() {
        let pre_93 = r#"{"hidden":5013,"tagged":42,"checkable":5013,"checked":1204,"cut":7}"#;
        let old: Coverage = serde_json::from_str(pre_93).unwrap();
        assert_eq!(old, fleet_coverage());
        assert_eq!(old.blocked, 0);
    }

    #[test]
    fn the_cut_count_is_carried_through_rather_than_derived() {
        let cov = coverage(&neurons_only(4), &tags(&["h0"]), &[], 3);
        assert_eq!(
            cov.cut, 3,
            "the cut neurons are no longer on the creature to count"
        );
        assert_eq!(cov.tagged, 1, "the tagged count is informational only");
    }

    /// A blocked write must name the file it could not write, so the caller's
    /// warning is actionable rather than a silent missing artefact.
    #[test]
    fn a_blocked_write_returns_an_error_naming_the_file() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join("out");
        // A directory where coverage.txt belongs: the write cannot succeed.
        std::fs::create_dir_all(dir.join(COVERAGE_TEXT_FILE)).unwrap();
        let err = write_files(&dir, &CoverageReport::new(fleet_coverage()), 100).unwrap_err();
        assert!(err.contains(COVERAGE_TEXT_FILE), "{err}");
    }

    /// The fleet-scale winner figures from Issue #59, rendered exactly.
    fn fleet_winners() -> Winners {
        Winners {
            screened: 38,
            confirmed: 22,
            applied: 1,
            carried: 21,
            plans: 9,
            skipped: 3,
            best_cuts: 14,
            best_delta: 1.2e-4,
            dropped: 12,
            est_ms_per_creature: 18_000,
        }
    }

    #[test]
    fn the_winners_block_renders_exactly_as_grq_will_paste_it() {
        let report = CoverageReport {
            coverage: fleet_coverage(),
            newly_screened: 100,
            winners: Some(fleet_winners()),
            corpus_identity: None,
            history: None,
            passes: None,
        };
        assert_eq!(
            report.description(100),
            concat!(
                "🪒 Ockham neuron screening coverage\n",
                "sweep:     1204 of 5013 hidden (24.0% of epoch)\n",
                "cut:       7 this run\n",
                "unchecked: 3809 remaining this epoch (~39 runs at 100/run)\n",
                "tagged:    42 carry tags, screened like any other\n",
                "progress:  100 newly checked this run\n",
                "winners:   38 screened · 22 confirmed · 1 applied · 21 carried\n",
                "bundles:   9 plans · best 14 cuts (Δ +1.2e-4) · 3 skipped\n",
                "dropped:   12 entries over budget (est 18s/creature)"
            )
        );
    }

    /// Issue #77: the per-run figure is rendered on every run, zero included —
    /// a plateau is only readable across two commits if both carry the line.
    #[test]
    fn a_run_that_advanced_nothing_still_renders_its_zero_progress() {
        let report = CoverageReport::new(fleet_coverage());
        let block = report.description(100);
        assert!(
            block.contains("\nprogress:  0 newly checked this run"),
            "{block}"
        );
        assert!(
            block.ends_with("progress:  0 newly checked this run"),
            "{block}"
        );
    }

    /// The block is pasted into every fleet host's check-in commit, so a run
    /// with nothing to say must add no empty lines and no `0 of 0` filler.
    ///
    /// Since #77 it carries one line the coverage block itself does not: the
    /// UUIDs this run newly screened, which is a per-run fact rather than a
    /// property of the incumbent.
    #[test]
    fn a_run_with_no_winners_renders_exactly_todays_block() {
        let cov = fleet_coverage();
        let report = CoverageReport::new(cov);
        assert_eq!(
            report.description(100),
            format!(
                "{}\nprogress:  0 newly checked this run",
                cov.description(100, None)
            )
        );
        assert!(!report.description(100).contains("winners:"));
        assert!(!report.description(100).contains("bundles:"));
        assert!(!report.description(100).contains("dropped:"));
        assert!(!Winners::default().has_any());
    }

    #[test]
    fn each_winner_line_is_omitted_when_it_has_nothing_to_report() {
        let report = CoverageReport {
            coverage: fleet_coverage(),
            newly_screened: 4,
            winners: Some(Winners {
                screened: 4,
                confirmed: 0,
                applied: 0,
                carried: 0,
                ..Winners::default()
            }),
            corpus_identity: None,
            history: None,
            passes: None,
        };
        let block = report.description(100);
        assert!(block.ends_with("winners:   4 screened · 0 confirmed · 0 applied · 0 carried"));
        assert!(!block.contains("bundles:"), "{block}");
        assert!(!block.contains("dropped:"), "{block}");
    }

    #[test]
    fn a_bundle_line_without_skips_omits_the_skipped_clause() {
        let report = CoverageReport {
            coverage: fleet_coverage(),
            newly_screened: 38,
            winners: Some(Winners {
                skipped: 0,
                dropped: 0,
                ..fleet_winners()
            }),
            corpus_identity: None,
            history: None,
            passes: None,
        };
        let block = report.description(100);
        assert!(
            block.ends_with("bundles:   9 plans · best 14 cuts (Δ +1.2e-4)"),
            "{block}"
        );
    }

    #[test]
    fn the_json_stays_readable_by_a_consumer_that_only_knows_coverage() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join("out");
        let report = CoverageReport {
            coverage: fleet_coverage(),
            newly_screened: 100,
            winners: Some(fleet_winners()),
            corpus_identity: None,
            history: None,
            passes: None,
        };
        write_files(&dir, &report, 100).unwrap();

        let json = std::fs::read_to_string(dir.join(COVERAGE_JSON_FILE)).unwrap();
        let old: Coverage = serde_json::from_str(&json).unwrap();
        assert_eq!(old, fleet_coverage(), "existing fields must not move");
        let back: CoverageReport = serde_json::from_str(&json).unwrap();
        assert_eq!(back, report, "the new key must round-trip");
        assert!(json.contains("\"checkable\": 5013"), "{json}");
        assert!(json.contains("\"newlyScreened\": 100"), "{json}");
        assert!(json.contains("\"winners\": {"), "{json}");

        // A pre-#77 artefact declares no per-run progress, not a missing field.
        let pre_77 = r#"{"hidden":5013,"tagged":42,"checkable":5013,"checked":1204,"cut":7}"#;
        let older: CoverageReport = serde_json::from_str(pre_77).unwrap();
        assert_eq!(older, CoverageReport::new(fleet_coverage()));

        let text = std::fs::read_to_string(dir.join(COVERAGE_TEXT_FILE)).unwrap();
        assert_eq!(text, format!("{}\n", report.description(100)));
    }

    /// Issue #100: the artefact names the corpus its figures were measured
    /// against, so `100%` is readable as "100% of this corpus epoch". Issue
    /// #102 moved the line under the percentage it qualifies and compacted the
    /// identity for humans — the full identity stays in the JSON.
    #[test]
    fn the_artefacts_name_the_corpus_epoch_the_figures_belong_to() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join("out");
        let report = CoverageReport {
            corpus_identity: Some("6fc028da266d6c51".into()),
            ..CoverageReport::new(fleet_coverage())
        };
        write_files(&dir, &report, 100).unwrap();

        let block = report.description(100);
        assert!(
            block.contains(concat!(
                "sweep:     1204 of 5013 hidden (24.0% of epoch)\n",
                "epoch:     corpus 6fc028da — coverage counts this corpus only\n"
            )),
            "{block}"
        );
        let json = std::fs::read_to_string(dir.join(COVERAGE_JSON_FILE)).unwrap();
        assert!(
            json.contains("\"corpusIdentity\": \"6fc028da266d6c51\""),
            "{json}"
        );
        let back: CoverageReport = serde_json::from_str(&json).unwrap();
        assert_eq!(back, report, "the epoch must round-trip");
        let old: Coverage = serde_json::from_str(&json).unwrap();
        assert_eq!(
            old,
            fleet_coverage(),
            "a consumer that only knows Coverage still reads it"
        );
    }

    /// A run with no screen store has no epoch to name, and must render and
    /// serialise exactly as it did before #100.
    #[test]
    fn a_report_with_no_epoch_renders_and_writes_exactly_as_before() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join("out");
        let report = CoverageReport::new(fleet_coverage());
        write_files(&dir, &report, 100).unwrap();
        let block = report.description(100);
        assert!(!block.contains("epoch:"), "{block}");
        let json = std::fs::read_to_string(dir.join(COVERAGE_JSON_FILE)).unwrap();
        assert!(!json.contains("corpusIdentity"), "{json}");

        // A pre-#100 artefact names no epoch, rather than failing to read.
        let pre_100 = r#"{"hidden":5013,"tagged":42,"checkable":5013,"checked":1204,"cut":7,
            "newlyScreened":0}"#;
        let older: CoverageReport = serde_json::from_str(pre_100).unwrap();
        assert_eq!(older.corpus_identity, None);
        assert_eq!(older, report);
    }

    /// Issue #102: a commit subject cannot carry a 16-character hash, so the
    /// human-facing clause is truncated — on a character boundary, whatever the
    /// identity is made of, and never lengthened.
    #[test]
    fn the_epoch_short_id_is_compact_and_never_splits_a_character() {
        assert_eq!(short_epoch("6fc028da266d6c51"), "6fc028da");
        assert_eq!(short_epoch("6fc028da"), "6fc028da", "already short enough");
        assert_eq!(short_epoch("abc"), "abc", "shorter than the cap");
        assert_eq!(short_epoch(""), "");
        assert_eq!(
            short_epoch("π🪒éαβγδεζη"),
            "π🪒éαβγδε",
            "eight characters, not eight bytes"
        );
    }

    /// The principle in one assertion: the **sweep** finishes, Ockham does not.
    #[test]
    fn a_finished_sweep_is_reported_as_a_complete_sweep_never_a_finished_ockham() {
        let creature = neurons_only(4);
        let screens = [
            screen("h0", 1),
            screen("h1", 2),
            screen("h2", 3),
            screen("h3", 4),
        ];
        let cov = coverage(&creature, &HashSet::new(), &screens, 0);
        assert!(cov.sweep_complete());
        let report = CoverageReport {
            corpus_identity: Some("6fc028da266d6c51".into()),
            ..CoverageReport::new(cov)
        };
        let block = report.description(100);
        assert!(
            block.contains("unchecked: 0 remaining — sweep complete for this epoch"),
            "{block}"
        );
        for text in [block, cov.summary()] {
            assert!(text.contains("of epoch"), "{text}");
            assert!(
                !text.to_lowercase().contains("ockham complete")
                    && !text.to_lowercase().contains("ockham finished"),
                "the razor never finishes: {text}"
            );
        }
    }

    /// A creature with nothing to sweep has not completed a sweep.
    #[test]
    fn an_empty_denominator_never_claims_a_complete_sweep() {
        let cov = coverage(&neurons_only(0), &HashSet::new(), &[], 0);
        assert!(!cov.sweep_complete());
        let block = cov.description(100, Some("6fc028da266d6c51"));
        assert!(
            block.ends_with("unchecked: 0 remaining — no hidden neurons to sweep"),
            "{block}"
        );
    }

    /// The headline case of Issue #102, end to end through the artefacts: a
    /// sweep that finished one epoch, a corpus change, and the fresh partial
    /// coverage that must replace the `100%` rather than carry it forward.
    #[test]
    fn a_complete_epoch_then_a_corpus_change_reports_fresh_partial_coverage() {
        let creature = neurons_only(4);
        let old_epoch: Vec<Screened> = ["h0", "h1", "h2", "h3"]
            .iter()
            .enumerate()
            .map(|(i, uuid)| Screened {
                corpus_identity: Some("corp-aaaa1111".into()),
                ..screen(uuid, i as u64 + 1)
            })
            .collect();
        let complete = CoverageReport {
            corpus_identity: Some("corp-aaaa1111".into()),
            history: Some(ScreenHistory::new(&old_epoch).over(&creature)),
            ..CoverageReport::new(coverage(&creature, &HashSet::new(), &old_epoch, 0))
        };
        let block = complete.description(100);
        assert!(
            block.contains("sweep:     4 of 4 hidden (100.0% of epoch)"),
            "{block}"
        );
        assert!(block.contains("epoch:     corpus corp-aaa"), "{block}");
        assert!(
            block.contains("unchecked: 0 remaining — sweep complete for this epoch"),
            "{block}"
        );

        // GRQ pulls extended training data: same creature, new corpus, and two
        // neurons screened against it so far. The store keeps every record.
        let mut every_record = old_epoch.clone();
        for (uuid, at) in [("h0", 5), ("h1", 6)] {
            every_record.push(Screened {
                corpus_identity: Some("corp-bbbb2222".into()),
                ..screen(uuid, at)
            });
        }
        let all = crate::learnings::current_epoch_screens(every_record.clone(), "corp-bbbb2222");
        assert_eq!(
            all.len(),
            2,
            "only the records filed under the new corpus are coverage"
        );

        let fresh = CoverageReport {
            corpus_identity: Some("corp-bbbb2222".into()),
            history: Some(ScreenHistory::new(&every_record).over(&creature)),
            newly_screened: 2,
            ..CoverageReport::new(coverage(&creature, &HashSet::new(), &all, 0))
        };
        let block = fresh.description(100);
        assert!(
            block.contains("sweep:     2 of 4 hidden (50.0% of epoch)"),
            "{block}"
        );
        assert!(block.contains("epoch:     corpus corp-bbb"), "{block}");
        assert!(
            block.contains("unchecked: 2 remaining this epoch"),
            "{block}"
        );
        assert!(
            !block.contains("100.0%") && !block.contains("sweep complete"),
            "a corpus change must never leave a 100% standing: {block}"
        );
        // The previous epoch's work is preserved — beside the percentage, never
        // inside it.
        assert!(
            block.contains("history:   4 of 4 ever checked across 2 corpus epochs"),
            "{block}"
        );
        assert_eq!(fresh.coverage.percent(), 50.0);
    }

    /// The cumulative figures are the creature's own: a record for a uuid that
    /// has been pruned away describes a creature that no longer exists.
    #[test]
    fn history_counts_current_hidden_uuids_across_every_epoch() {
        let records = vec![
            screen("h0", 1),
            Screened {
                corpus_identity: Some("other".into()),
                ..screen("h1", 2)
            },
            Screened {
                corpus_identity: None,
                ..screen("h2", 3)
            },
            screen("gone", 4),
        ];
        let history = ScreenHistory::new(&records).over(&neurons_only(4));
        assert_eq!(
            history.checked_ever, 3,
            "the departed uuid counts for nothing"
        );
        assert_eq!(
            history.epochs, 3,
            "an unnamed pre-#76 epoch is still an epoch"
        );
        assert!(history.has_any());
        assert!(!ScreenHistory::default().over(&neurons_only(4)).has_any());

        // Records filed after the index was built are history too.
        let mut index = ScreenHistory::new(&records);
        index.merge(&[Screened {
            corpus_identity: Some("later".into()),
            ..screen("h3", 5)
        }]);
        let after = index.over(&neurons_only(4));
        assert_eq!(after.checked_ever, 4);
        assert_eq!(after.epochs, 4);
    }

    /// Both halves of the line are about the same creature: an epoch whose only
    /// records are for uuids this creature no longer carries reached nothing to
    /// report, so `0 ever checked across 7 corpus epochs` is unwriteable.
    #[test]
    fn the_history_epoch_count_is_scoped_to_the_creature_it_counts() {
        let records = vec![
            screen("h0", 1),
            Screened {
                corpus_identity: Some("elsewhere".into()),
                ..screen("another-creature", 2)
            },
            Screened {
                corpus_identity: Some("elsewhere-too".into()),
                ..screen("also-departed", 3)
            },
        ];
        let history = ScreenHistory::new(&records).over(&neurons_only(4));
        assert_eq!(history.checked_ever, 1);
        assert_eq!(
            history.epochs, 1,
            "only the epoch that reached a neuron still on the creature counts"
        );

        // Nothing on this creature was ever reached: the line is omitted rather
        // than reporting epochs with no checks behind them.
        let foreign = ScreenHistory::new(&records[1..]).over(&neurons_only(4));
        assert_eq!(foreign.checked_ever, 0);
        assert_eq!(foreign.epochs, 0);
        assert!(!foreign.has_any());
    }

    /// The history line is additive: omitted with no records, and never folded
    /// into the current-epoch percentage above it.
    #[test]
    fn the_history_line_is_omitted_when_the_store_holds_no_records() {
        let report = CoverageReport {
            history: Some(History::default()),
            ..CoverageReport::new(fleet_coverage())
        };
        let block = report.description(100);
        assert!(!block.contains("history:"), "{block}");
        assert!(
            !CoverageReport::new(fleet_coverage())
                .description(100)
                .contains("history:")
        );
    }

    /// `coverage.json` carries the cumulative figures machine-readably, and a
    /// consumer written before #102 still reads the file.
    #[test]
    fn the_history_round_trips_through_the_json_artefact() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join("out");
        let report = CoverageReport {
            corpus_identity: Some("6fc028da266d6c51".into()),
            history: Some(History {
                checked_ever: 4802,
                epochs: 3,
            }),
            ..CoverageReport::new(fleet_coverage())
        };
        write_files(&dir, &report, 100).unwrap();

        let json = std::fs::read_to_string(dir.join(COVERAGE_JSON_FILE)).unwrap();
        assert!(json.contains("\"checkedEver\": 4802"), "{json}");
        assert!(json.contains("\"epochs\": 3"), "{json}");
        let back: CoverageReport = serde_json::from_str(&json).unwrap();
        assert_eq!(back, report, "the cumulative figures must round-trip");
        let old: Coverage = serde_json::from_str(&json).unwrap();
        assert_eq!(
            old,
            fleet_coverage(),
            "a consumer that only knows Coverage still reads it"
        );
        assert!(
            report
                .description(100)
                .contains("history:   4802 of 5013 ever checked across 3 corpus epochs"),
            "{}",
            report.description(100)
        );

        // A pre-#102 artefact carries no history, rather than failing to read.
        let pre_102 = r#"{"hidden":5013,"tagged":42,"checkable":5013,"checked":1204,"cut":7,
            "newlyScreened":0}"#;
        let older: CoverageReport = serde_json::from_str(pre_102).unwrap();
        assert_eq!(older.history, None);
    }

    /// One epoch reads `epoch`, not `epochs` — the block is read by humans.
    #[test]
    fn a_single_epoch_history_line_is_singular() {
        let report = CoverageReport {
            history: Some(History {
                checked_ever: 12,
                epochs: 1,
            }),
            ..CoverageReport::new(fleet_coverage())
        };
        assert!(
            report
                .description(100)
                .contains("history:   12 of 5013 ever checked across 1 corpus epoch"),
            "{}",
            report.description(100)
        );
    }

    /// Issue #140: the artefact says how many complete passes the razor has
    /// made and which one is in progress, so a run beyond its first sweep is
    /// legible from the commit description alone.
    #[test]
    fn the_description_reports_the_completed_passes_and_the_one_in_progress() {
        let report = CoverageReport {
            passes: Some(Passes::new(1, 7, 120, 118)),
            ..CoverageReport::new(fleet_coverage())
        };
        let block = report.description(100);
        assert!(
            block.contains(
                "passes:    7 strict complete this epoch · 1 strict this run · pass 8 in progress\n\
                 visits:    120 hidden neurons visited this run · 118 revisited"
            ),
            "{block}"
        );
    }

    /// The headline case: 100% unique coverage with re-screening under way must
    /// not read as a run stuck at the end of its first pass.
    #[test]
    fn a_fully_covered_epoch_still_reports_the_pass_it_is_working() {
        let cov = Coverage {
            checked: 5013,
            ..fleet_coverage()
        };
        let report = CoverageReport {
            passes: Some(Passes::new(2, 4, 300, 300)),
            corpus_identity: Some("6fc028da266d6c51".into()),
            ..CoverageReport::new(cov)
        };
        let block = report.description(100);
        assert!(cov.sweep_complete(), "{block}");
        assert!(
            block.contains("unchecked: 0 remaining — sweep complete for this epoch"),
            "{block}"
        );
        assert!(block.contains("pass 5 in progress"), "{block}");
        assert!(
            block.contains("visits:    300 hidden neurons visited this run · 300 revisited"),
            "the re-screening a complete sweep is doing must be visible: {block}"
        );
    }

    /// The pass in progress is derived from the completed count, so the two can
    /// never disagree — and a fresh epoch opens honestly at pass 1.
    #[test]
    fn the_pass_in_progress_is_always_one_past_the_completed_count() {
        for completed in [0u64, 1, 9, 743] {
            let passes = Passes::new(0, completed, 0, 0);
            assert_eq!(passes.current_pass, completed + 1);
        }
        let opening = CoverageReport {
            passes: Some(Passes::new(0, 0, 0, 0)),
            ..CoverageReport::new(fleet_coverage())
        };
        let block = opening.description(100);
        assert!(
            block.contains(
                "passes:    0 strict complete this epoch · 0 strict this run · pass 1 in progress"
            ),
            "{block}"
        );
        assert!(
            !block.contains("visits:"),
            "a run that reached nothing has no visits to report: {block}"
        );
    }

    /// A report with no screen store behind it renders exactly as it did before
    /// #140, and the new key round-trips for one that has.
    #[test]
    fn the_passes_object_round_trips_and_is_absent_without_a_store() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join("out");
        let bare = CoverageReport::new(fleet_coverage());
        write_files(&dir, &bare, 100).unwrap();
        let json = std::fs::read_to_string(dir.join(COVERAGE_JSON_FILE)).unwrap();
        assert!(!json.contains("passes"), "{json}");
        assert!(!bare.description(100).contains("passes:"));

        let report = CoverageReport {
            passes: Some(Passes::new(1, 7, 120, 118)),
            ..bare
        };
        write_files(&dir, &report, 100).unwrap();
        let json = std::fs::read_to_string(dir.join(COVERAGE_JSON_FILE)).unwrap();
        assert!(json.contains("\"sweepRestartsRun\": 1"), "{json}");
        assert!(json.contains("\"sweepsCompletedEpoch\": 7"), "{json}");
        assert!(json.contains("\"currentPass\": 8"), "{json}");
        assert!(json.contains("\"visitedRun\": 120"), "{json}");
        assert!(json.contains("\"revisitedRun\": 118"), "{json}");
        let back: CoverageReport = serde_json::from_str(&json).unwrap();
        assert_eq!(back, report, "the pass counters must round-trip");
        let old: Coverage = serde_json::from_str(&json).unwrap();
        assert_eq!(
            old,
            fleet_coverage(),
            "a consumer that only knows Coverage still reads it"
        );

        // A pre-#140 artefact carries no passes, rather than failing to read.
        let pre_140 = r#"{"hidden":5013,"tagged":42,"checkable":5013,"checked":1204,"cut":7,
            "newlyScreened":0}"#;
        let older: CoverageReport = serde_json::from_str(pre_140).unwrap();
        assert_eq!(older.passes, None);
    }

    /// The existing `checked` figure keeps its meaning: repeated passes are
    /// reported beside it, never folded into it (Issue #140).
    #[test]
    fn repeated_passes_never_move_the_unique_coverage_figures() {
        let cov = fleet_coverage();
        let with_passes = CoverageReport {
            passes: Some(Passes::new(3, 9, 400, 400)),
            ..CoverageReport::new(cov)
        };
        assert_eq!(with_passes.coverage, cov);
        assert_eq!(with_passes.coverage.percent(), cov.percent());
        assert_eq!(
            with_passes.newly_screened, 0,
            "a revisit is not a newly checked uuid"
        );
        let block = with_passes.description(100);
        assert!(
            block.contains("sweep:     1204 of 5013 hidden (24.0% of epoch)"),
            "{block}"
        );
        assert!(
            block.contains("progress:  0 newly checked this run"),
            "{block}"
        );
    }

    /// The plateau signature itself: nothing newly screened while unchecked
    /// neurons remain. Eight silent runs become eight warnings.
    #[test]
    fn a_run_that_advanced_nothing_warns_naming_both_figures() {
        let warning = zero_progress_warning(0, 190).expect("a plateau must warn");
        assert!(warning.contains('0'), "{warning}");
        assert!(warning.contains("190"), "{warning}");
        assert!(warning.contains("unchecked"), "{warning}");
        assert_eq!(
            zero_progress_warning(1, 190),
            None,
            "a run that advanced coverage is not a plateau"
        );
        assert_eq!(
            zero_progress_warning(0, 0),
            None,
            "a fully screened creature has nothing left to advance"
        );
    }

    /// Only a uuid the fleet had never screened counts as progress (#77):
    /// recycling the stalest neurons of a complete creature advances nothing.
    #[test]
    fn only_a_first_ever_screen_record_counts_as_progress() {
        let existing = vec![screen("h_a", 10)];
        let mut progress = ScreenProgress::new(&existing);
        assert_eq!(progress.count(), 0);
        progress.observe("h_a");
        assert_eq!(progress.count(), 0, "re-screening is not new coverage");
        progress.observe("h_b");
        progress.observe("h_b");
        assert_eq!(
            progress.count(),
            1,
            "a uuid counts once, however often it is screened"
        );
        progress.observe("h_c");
        assert_eq!(progress.count(), 2);
    }

    /// Issue #140: a revisit is work, not new coverage. Both figures are kept,
    /// and neither is allowed to stand in for the other.
    #[test]
    fn visit_attempts_count_repeat_work_while_progress_counts_only_new_keys() {
        let existing = vec![screen("h_a", 10), screen("h_b", 11)];
        let mut progress = ScreenProgress::new(&existing);
        assert_eq!(progress.visit_counts().total, 0);

        progress.visit("h_a");
        progress.visit("h_a");
        progress.visit("h_b");
        progress.visit("h_new");
        assert_eq!(
            progress.visit_counts().total,
            4,
            "repeat visits are real work and stay counted"
        );
        assert_eq!(
            progress.revisit_counts().total,
            3,
            "two h_a visits plus h_b were revisits"
        );
        assert_eq!(
            progress.count(),
            0,
            "visiting is not filing: only a filed first-ever record is progress"
        );

        progress.observe("h_new");
        assert_eq!(progress.count(), 1);
        assert_eq!(
            progress.revisit_counts().total,
            3,
            "the new uuid is not a revisit"
        );
    }

    #[test]
    fn a_creature_with_no_hidden_neurons_is_complete_and_empty() {
        let cov = coverage(&neurons_only(0), &HashSet::new(), &[], 7);
        assert_eq!(cov.hidden, 0);
        assert_eq!(cov.checkable, 0);
        assert_eq!(cov.percent(), 0.0);
        assert_eq!(cov.cut, 7);
    }

    // ---- Issue #137: synapse visits in the epoch coverage denominator ----

    /// The acceptance criterion: `checkable` is the whole visit population.
    #[test]
    fn synapse_visits_join_the_hidden_neurons_in_the_denominator() {
        let creature = hidden_creature(4);
        let cov = coverage(&creature, &HashSet::new(), &[], 0);
        assert_eq!(cov.hidden, 4);
        assert_eq!(cov.synapses, 8, "input-0 → h{{i}} → output-0 for each of 4");
        assert_eq!(cov.checkable, 12, "N hidden + M synapses");
        assert_eq!(cov.checked, 0);
        assert_eq!(cov.synapses_checked, 0);
    }

    /// A screened edge raises `checked` and the synapse figure beside it, and
    /// the two halves always add up to the whole numerator.
    #[test]
    fn a_screened_edge_raises_both_the_total_and_the_synapse_count() {
        let creature = hidden_creature(2);
        let screens = [
            screen("h0", 1),
            screen(&edge("input-0", "h0"), 2),
            screen(&edge("h1", "output-0"), 3),
        ];
        let cov = coverage(&creature, &HashSet::new(), &screens, 0);
        assert_eq!(cov.checkable, 6, "2 hidden + 4 edges");
        assert_eq!(cov.checked, 3);
        assert_eq!(cov.synapses_checked, 2);
        assert_eq!(cov.percent(), 50.0);
        assert_eq!(cov.unchecked(), 3);
    }

    /// Issue #137: a typed edge is visited, blocked, and therefore checked. A
    /// creature whose every edge is typed must still be able to finish a sweep.
    #[test]
    fn a_typed_edge_is_counted_and_a_blocked_visit_checks_it() {
        let creature = creature(
            1,
            1,
            vec![
                neuron("hidden", "h0", 0.0, Some("IDENTITY")),
                neuron("output", "output-0", 0.0, Some("IDENTITY")),
            ],
            vec![
                typed_synapse("input-0", "h0", 1.0, "condition"),
                typed_synapse("h0", "output-0", 1.0, "positive"),
            ],
        );
        let bare = coverage(&creature, &HashSet::new(), &[], 0);
        assert_eq!(bare.synapses, 2, "a typed edge is a visit like any other");
        assert_eq!(bare.checkable, 3);
        assert!(!bare.sweep_complete(), "nothing has been visited yet");

        let screens = [
            blocked("h0", 1, BlockedReason::UnsafeTopology),
            blocked(&edge("input-0", "h0"), 2, BlockedReason::UnsafeTopology),
            blocked(&edge("h0", "output-0"), 3, BlockedReason::UnsafeTopology),
        ];
        let cov = coverage(&creature, &HashSet::new(), &screens, 0);
        assert_eq!(cov.checked, 3, "every visit was reached");
        assert_eq!(cov.synapses_checked, 2);
        assert_eq!(cov.blocked, 3, "and none of them could be cut");
        assert_eq!(cov.blocked_by_reason.unsafe_topology, 3);
        assert!(
            cov.sweep_complete(),
            "a typed creature reaches a complete sweep, it is not stranded"
        );
    }

    /// Issue #137: one unchecked edge keeps the epoch open. This is the whole
    /// point — an epoch declared finished with edges never tried is the failure.
    #[test]
    fn the_sweep_is_incomplete_while_a_single_synapse_visit_is_unchecked() {
        let creature = hidden_creature(2);
        let mut screens = vec![screen("h0", 1), screen("h1", 2)];
        assert!(
            !coverage(&creature, &HashSet::new(), &screens, 0).sweep_complete(),
            "every hidden neuron checked is not a swept epoch any more"
        );
        for (from, to) in [
            ("input-0", "h0"),
            ("input-0", "h1"),
            ("h0", "output-0"),
            ("h1", "output-0"),
        ] {
            screens.push(screen(&edge(from, to), 3));
        }
        let cov = coverage(&creature, &HashSet::new(), &screens, 0);
        assert!(cov.sweep_complete(), "{cov:?}");
        assert_eq!(cov.percent(), 100.0);
        assert_eq!(cov.synapses_checked, cov.synapses);
    }

    /// Issue #137: the blocked breakdown is still a partition of `blocked`
    /// when the blocked population mixes neuron and synapse visits.
    #[test]
    fn the_reason_counts_sum_to_blocked_across_neuron_and_synapse_visits() {
        let creature = hidden_creature(3);
        let screens = [
            screen("h0", 1),
            blocked("h1", 2, BlockedReason::AggregateSquash),
            visit("h2", 3),
            blocked(&edge("input-0", "h0"), 4, BlockedReason::UnsafeTopology),
            blocked(&edge("h0", "output-0"), 5, BlockedReason::UnsafeTopology),
            blocked(&edge("input-0", "h1"), 6, BlockedReason::MissingActivation),
            screen(&edge("h1", "output-0"), 7),
        ];
        let cov = coverage(&creature, &HashSet::new(), &screens, 0);
        assert_eq!(cov.checked, 7);
        assert_eq!(cov.synapses_checked, 4);
        assert_eq!(cov.blocked, 5, "two neurons and three edges");
        assert_eq!(
            cov.blocked_by_reason.total(),
            cov.blocked,
            "the breakdown is a partition of the blocked population"
        );
        assert_eq!(cov.blocked_by_reason.unsafe_topology, 2);
        assert_eq!(cov.blocked_by_reason.aggregate_squash, 1);
        assert_eq!(cov.blocked_by_reason.missing_activation, 1);
        assert_eq!(cov.blocked_by_reason.unrecorded, 1);
    }

    /// One visit key per **ordered pair**, exactly as the sweep pool builds it:
    /// a repeated pair is one visit, never two entries in the denominator.
    #[test]
    fn a_repeated_endpoint_pair_is_one_synapse_visit() {
        let mut creature = hidden_creature(1);
        creature
            .synapses
            .push(typed_synapse("input-0", "h0", 0.5, "condition"));
        let cov = coverage(&creature, &HashSet::new(), &[], 0);
        assert_eq!(cov.synapses, 2, "three edges, two ordered pairs");
        assert_eq!(cov.checkable, 3);
    }

    /// The #37 rule holds for edges too: a record for an edge the creature no
    /// longer carries describes a creature that no longer exists.
    #[test]
    fn a_screen_record_for_a_departed_edge_raises_nothing() {
        let creature = hidden_creature(1);
        let screens = [
            screen(&edge("h0", "gone"), 1),
            screen(&edge("input-0", "h0"), 2),
        ];
        let cov = coverage(&creature, &HashSet::new(), &screens, 0);
        assert_eq!(cov.synapses, 2);
        assert_eq!(cov.synapses_checked, 1, "only the still-present edge");
        assert_eq!(cov.checked, 1);
    }

    /// Issue #137: `coverage.txt` gains a line, it does not restructure one.
    /// The no-synapse block — which is what a pre-#137 `coverage.json` renders
    /// as — is byte-for-byte what it was.
    #[test]
    fn the_description_gains_a_synapse_line_and_leaves_the_rest_alone() {
        let without = coverage(&neurons_only(4), &HashSet::new(), &[screen("h0", 1)], 1);
        assert_eq!(
            without.description(100, None),
            concat!(
                "🪒 Ockham neuron screening coverage\n",
                "sweep:     1 of 4 hidden (25.0% of epoch)\n",
                "cut:       1 this run\n",
                "unchecked: 3 remaining this epoch (~1 run at 100/run)"
            ),
            "a creature with no synapse visits renders exactly as before"
        );

        let with = coverage(
            &hidden_creature(4),
            &HashSet::new(),
            &[screen("h0", 1), screen(&edge("input-0", "h0"), 2)],
            1,
        );
        assert_eq!(
            with.description(100, None),
            concat!(
                "🪒 Ockham neuron screening coverage\n",
                "sweep:     2 of 12 visits (16.7% of epoch)\n",
                "synapses:  1 of 8 edges checked this epoch\n",
                "cut:       1 this run\n",
                "unchecked: 10 remaining this epoch (~1 run at 100/run)"
            )
        );
    }

    /// The one-line log summary names both populations too, and is unchanged
    /// for a creature with no synapse visits.
    #[test]
    fn the_summary_names_both_populations() {
        let without = coverage(&neurons_only(4), &HashSet::new(), &[screen("h0", 1)], 3);
        assert_eq!(
            without.summary(),
            "sweep 1/4 checked (25.0% of epoch), 3 cut"
        );
        let with = coverage(
            &hidden_creature(4),
            &tags(&["h1"]),
            &[screen("h0", 1), screen(&edge("input-0", "h0"), 2)],
            3,
        );
        assert_eq!(
            with.summary(),
            "sweep 2/12 checked (16.7% of epoch), 3 cut, 1/8 synapses, 1 tagged"
        );
    }

    /// Issue #137: an artefact written before the synapse fields existed still
    /// deserialises, reading as zero rather than failing the parse.
    #[test]
    fn a_pre_137_coverage_json_reads_as_no_synapse_visits() {
        let json = r#"{
            "hidden": 40,
            "tagged": 2,
            "checkable": 40,
            "checked": 12,
            "blocked": 3,
            "cut": 1
        }"#;
        let cov: Coverage =
            serde_json::from_str(json).expect("a pre-#137 artefact must still read");
        assert_eq!(cov.synapses, 0);
        assert_eq!(cov.synapses_checked, 0);
        assert_eq!(
            cov.checkable, 40,
            "the stored denominator is not re-derived"
        );
        assert!(!cov.description(100, None).contains("synapses:"));
    }

    /// A synapse's history must be visible to the cumulative line — and so to
    /// unchecked-first selection, which reads the same still-present set.
    #[test]
    fn the_history_counts_synapse_visits_beside_hidden_neurons() {
        let creature = hidden_creature(2);
        let old = edge("input-0", "h0");
        let history = ScreenHistory::new(&[
            Screened {
                corpus_identity: Some("corp-old".into()),
                ..screen("h0", 1)
            },
            Screened {
                corpus_identity: Some("corp-old".into()),
                ..screen(&old, 2)
            },
            Screened {
                corpus_identity: Some("corp-old".into()),
                ..screen(&edge("h0", "gone"), 3)
            },
        ]);
        let over = history.over(&creature);
        assert_eq!(
            over.checked_ever, 2,
            "the neuron and the still-present edge, never the departed one"
        );
        assert_eq!(over.epochs, 1);
    }

    /// Issue #137: a run that only reached synapse visits has advanced
    /// coverage, so it must not be reported as a plateau — and the figure the
    /// warning names counts visits, not hidden neurons.
    #[test]
    fn a_run_that_only_checked_synapses_is_progress_not_a_plateau() {
        let existing = vec![screen("h0", 1)];
        let mut progress = ScreenProgress::new(&existing);
        progress.observe("h0");
        assert_eq!(progress.count(), 0, "re-screening the neuron is not new");
        progress.observe(&edge("input-0", "h0"));
        assert_eq!(progress.count(), 1, "an edge is coverage like any other");
        assert_eq!(
            zero_progress_warning(progress.count(), 7),
            None,
            "a synapse-only run advanced coverage"
        );
        let warning = zero_progress_warning(0, 7).expect("a plateau must still warn");
        assert!(warning.contains("visit(s)"), "{warning}");
        assert!(!warning.contains("hidden neuron"), "{warning}");
    }
}
