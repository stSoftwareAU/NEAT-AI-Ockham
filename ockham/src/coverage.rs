//! Screening coverage over the **current** incumbent (Issue #37).
//!
//! One place answers "how far has Ockham got through this creature?", so the
//! `ockham` tag, the commit description and `--report` can never disagree.
//!
//! Three rules make the denominator honest:
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
//!
//! Evolution keeps adding hidden neurons, and each new one starts unchecked and
//! *lowers* the percentage. That is intended: coverage is a statement about the
//! creature in front of us, not a monotonic score.

use std::collections::HashSet;
use std::path::Path;

use neat_core::CreatureExport;
use serde::{Deserialize, Serialize};

use crate::learnings::Screened;

/// Rendered commit-description block, written beside `best.json` (Issue #40).
pub const COVERAGE_TEXT_FILE: &str = "coverage.txt";
/// Serialised [`Coverage`], written beside `best.json` (Issue #40).
pub const COVERAGE_JSON_FILE: &str = "coverage.json";

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
    /// Hidden neurons Ockham may try — all of them, tagged included (#74).
    ///
    /// The key keeps its name so `coverage.json` stays deserialisable by
    /// anything already reading it; what changed is the definition.
    pub checkable: usize,
    /// Hidden UUIDs with at least one screen record.
    pub checked: usize,
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

    /// One-line progress summary — it goes in a commit subject.
    ///
    /// `checked 1204 of 5013 hidden (24.0%), 7 cut, 42 tagged`. The `X of Y`
    /// denominator is [`Self::checkable`], so it always agrees with the
    /// percentage. The tagged clause counts neurons *inside* that denominator
    /// (#74) and is omitted when nothing is tagged.
    pub fn summary(&self) -> String {
        let mut out = format!(
            "checked {} of {} hidden ({:.1}%), {} cut",
            self.checked,
            self.checkable,
            self.percent(),
            self.cut
        );
        if self.tagged > 0 {
            out.push_str(&format!(", {} tagged", self.tagged));
        }
        out
    }

    /// Hidden UUIDs with no screen record yet.
    ///
    /// Saturating: a stale record set can report more `checked` than there are
    /// checkable neurons, and "minus three unchecked" is not a measurement.
    pub fn unchecked(&self) -> usize {
        self.checkable.saturating_sub(self.checked)
    }

    /// Multi-line commit-description block — the GRQ-facing artefact (Issue #40).
    ///
    /// ```text
    /// 🪒 Ockham neuron screening coverage
    /// checked:   1204 of 5013 hidden (24.0%)
    /// cut:       7 this run
    /// unchecked: 3809 remaining (~39 runs at 100/run)
    /// blocked:   412 checked but structurally unprunable
    /// tagged:    42 carry tags, screened like any other
    /// ```
    ///
    /// Line-oriented and stable: GRQ pastes it into a `git commit` description.
    /// `candidates` is the configured `--candidates` batch size; the
    /// runs-remaining clause is **omitted** rather than rendering `inf` when
    /// the batch size is zero, and there is nothing to estimate once coverage
    /// is complete. The `tagged:` line is omitted when nothing is tagged, and
    /// says only how many neurons carry tags — cutting one needs no declaration
    /// (Issue #87). The `blocked:` line is likewise omitted when nothing is
    /// blocked, and says how much of `checked` was reached by a visit the razor
    /// could propose nothing for (Issue #93). No trailing newline —
    /// [`write_files`] adds one.
    pub fn description(&self, candidates: usize) -> String {
        let unchecked = self.unchecked();
        let runs = if candidates > 0 && unchecked > 0 {
            let n = unchecked.div_ceil(candidates);
            let unit = if n == 1 { "run" } else { "runs" };
            format!(" (~{n} {unit} at {candidates}/run)")
        } else {
            String::new()
        };
        let mut out = String::from("🪒 Ockham neuron screening coverage\n");
        out.push_str(&format!(
            "{:<11}{} of {} hidden ({:.1}%)\n",
            "checked:",
            self.checked,
            self.checkable,
            self.percent()
        ));
        out.push_str(&format!("{:<11}{} this run\n", "cut:", self.cut));
        out.push_str(&format!("{:<11}{unchecked} remaining{runs}", "unchecked:"));
        if self.blocked > 0 {
            out.push_str(&format!(
                "\n{:<11}{} checked but structurally unprunable",
                "blocked:", self.blocked
            ));
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
#[derive(Debug, Default)]
pub(crate) struct ScreenProgress {
    opening: HashSet<String>,
    added: HashSet<String>,
}

impl ScreenProgress {
    /// Snapshot what the fleet had already screened when the run opened.
    pub(crate) fn new(screens: &[Screened]) -> Self {
        Self {
            opening: screens.iter().map(|s| s.uuid.clone()).collect(),
            added: HashSet::new(),
        }
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
/// A run that adds nothing to the screened set while unchecked neurons remain
/// has not done the job it exists to do, whatever else it reported. The line
/// names both figures so a plateau is legible from one run's log rather than
/// only by diffing two commits.
pub(crate) fn zero_progress_warning(newly_screened: usize, unchecked: usize) -> Option<String> {
    (newly_screened == 0 && unchecked > 0).then(|| {
        format!(
            "no progress: 0 newly checked uuid(s) this run while {unchecked} hidden neuron(s) \
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

/// The commit-description artefact: coverage, plus the run's winner economics.
///
/// [`Coverage`] is flattened, so a consumer that deserialises `coverage.json`
/// straight into [`Coverage`] keeps working — serde ignores the extra keys.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
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
}

impl CoverageReport {
    /// Coverage with no per-run progress and no winner figures.
    pub fn new(coverage: Coverage) -> Self {
        Self {
            coverage,
            newly_screened: 0,
            winners: None,
        }
    }

    /// The full commit-description block: coverage, progress, then winners.
    ///
    /// The `progress:` line is always rendered, zero included — a plateau is
    /// only visible by reading two consecutive commits if the figure is there
    /// in both.
    pub fn description(&self, candidates: usize) -> String {
        let mut out = self.coverage.description(candidates);
        out.push_str(&format!(
            "\n{:<11}{} newly checked this run",
            "progress:", self.newly_screened
        ));
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

/// Count coverage of `creature` from `screens`; `tagged` is counted, not excluded.
///
/// Every hidden neuron is in the denominator, tagged ones included (#74) —
/// `tagged` only says how many of them carry tags.
///
/// `cut` is the hidden neurons removed this run, carried through rather than
/// derived, because the creature in hand no longer holds them.
pub fn coverage(
    creature: &CreatureExport,
    tagged: &HashSet<String>,
    screens: &[Screened],
    cut: usize,
) -> Coverage {
    let hidden_uuids: Vec<&str> = creature
        .neurons
        .iter()
        .filter(|n| n.neuron_type == "hidden")
        .map(|n| n.uuid.as_str())
        .collect();
    let hidden = hidden_uuids.len();
    let tagged_hidden = hidden_uuids
        .iter()
        .filter(|uuid| tagged.contains(**uuid))
        .count();
    let hidden_uuids: HashSet<&str> = hidden_uuids.into_iter().collect();
    // A map of the screened UUIDs, so a uuid screened many times — by this host
    // or another — still counts once. The value is "every record so far was a
    // skipped visit": one real screen anywhere in the fleet's history clears it
    // permanently, so `blocked` never over-reports (Issue #93).
    let mut checked: std::collections::HashMap<&str, bool> = std::collections::HashMap::new();
    for s in screens
        .iter()
        .filter(|s| hidden_uuids.contains(s.uuid.as_str()))
    {
        let only_skips = checked.entry(s.uuid.as_str()).or_insert(true);
        *only_skips &= s.is_skipped();
    }
    Coverage {
        hidden,
        tagged: tagged_hidden,
        checkable: hidden,
        checked: checked.len(),
        blocked: checked.values().filter(|only_skips| **only_skips).count(),
        cut,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixtures::{creature, neuron, synapse};
    use crate::learnings::{SCREENS_FORMAT_VERSION, ScreenOutcomeKind};

    /// Creature with `n` parallel hidden IDENTITY neurons `h0..h{n-1}`.
    fn hidden_creature(n: usize) -> CreatureExport {
        let mut neurons: Vec<_> = (0..n)
            .map(|i| neuron("hidden", &format!("h{i}"), 0.0, Some("IDENTITY")))
            .collect();
        neurons.push(neuron("output", "output-0", 0.0, Some("IDENTITY")));
        let mut synapses = Vec::new();
        for i in 0..n {
            synapses.push(synapse("input-0", &format!("h{i}"), 1.0));
            synapses.push(synapse(&format!("h{i}"), "output-0", 1.0));
        }
        creature(1, 1, neurons, synapses)
    }

    fn screen(uuid: &str, unix_secs: u64) -> Screened {
        Screened {
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

    fn tags(uuids: &[&str]) -> HashSet<String> {
        uuids.iter().map(|u| (*u).to_string()).collect()
    }

    /// Rewritten for Issue #74: tagged neurons stay in the denominator and are
    /// still reported separately.
    #[test]
    fn tagged_neurons_stay_in_the_denominator_and_are_reported_separately() {
        let creature = hidden_creature(10);
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
        let creature = hidden_creature(6);
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
        let creature = hidden_creature(4);
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
        let creature = hidden_creature(4);
        let screens = [screen("h0", 1), screen("gone-a", 2), screen("gone-b", 3)];
        let cov = coverage(&creature, &HashSet::new(), &screens, 2);
        assert_eq!(cov.hidden, 4, "pruned neurons are not on the incumbent");
        assert_eq!(cov.checked, 1, "only still-present UUIDs are checked");
        assert_eq!(cov.cut, 2);
    }

    #[test]
    fn a_uuid_screened_many_times_counts_once() {
        let creature = hidden_creature(4);
        let screens = [screen("h0", 1), screen("h0", 2), screen("h0", 3)];
        let cov = coverage(&creature, &HashSet::new(), &screens, 0);
        assert_eq!(cov.checked, 1);
        assert_eq!(cov.percent(), 25.0);
    }

    /// Rewritten for Issue #74: a screened tagged uuid raises `checked`.
    #[test]
    fn a_screened_tagged_uuid_counts_as_checked() {
        let creature = hidden_creature(4);
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
        let creature = hidden_creature(4);
        let screens = [screen("h0", 1), visit("h1", 2), visit("h2", 3)];
        let cov = coverage(&creature, &HashSet::new(), &screens, 0);
        assert_eq!(cov.checked, 3, "every visited uuid counts as checked");
        assert_eq!(cov.blocked, 2, "two of them were never scored");
        assert_eq!(cov.percent(), 75.0);
        assert_eq!(cov.unchecked(), 1, "only h3 was never reached");
    }

    /// `blocked` is about the uuid, not the record: one real screen anywhere in
    /// the fleet's history says the razor could propose something for it.
    #[test]
    fn one_real_screen_clears_blocked_however_many_visits_surround_it() {
        let creature = hidden_creature(2);
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
        let creature = hidden_creature(2);
        let screens = [visit("gone", 1), visit("h0", 2)];
        let cov = coverage(&creature, &HashSet::new(), &screens, 1);
        assert_eq!(cov.checked, 1);
        assert_eq!(cov.blocked, 1, "the departed uuid is not on the creature");
    }

    /// Intended behaviour, not a bug: evolution adds hidden neurons, they start
    /// unchecked, and the percentage falls. Coverage describes the creature in
    /// front of us — do not "fix" this by remembering departed UUIDs.
    #[test]
    fn newly_evolved_neurons_lower_the_percentage() {
        let screens = [screen("h0", 1), screen("h1", 2)];
        let before = coverage(&hidden_creature(2), &HashSet::new(), &screens, 0);
        assert_eq!(before.percent(), 100.0);
        let after = coverage(&hidden_creature(4), &HashSet::new(), &screens, 0);
        assert_eq!(after.percent(), 50.0, "two new neurons start unchecked");
        assert!(after.percent() < before.percent());
    }

    /// Rewritten for Issue #74: an all-tagged creature has a real denominator
    /// now, so the zero-denominator guard is the no-hidden-neurons case below.
    #[test]
    fn nothing_checked_yields_zero_percent_without_panicking() {
        let creature = hidden_creature(2);
        let cov = coverage(&creature, &tags(&["h0", "h1"]), &[], 0);
        assert_eq!(cov.checkable, 2);
        assert_eq!(cov.checked, 0);
        assert_eq!(cov.percent(), 0.0);
        assert_eq!(
            cov.summary(),
            "checked 0 of 2 hidden (0.0%), 0 cut, 2 tagged"
        );
    }

    /// The only zero denominator left: no hidden neurons at all.
    #[test]
    fn an_empty_denominator_yields_zero_percent_without_panicking() {
        let cov = coverage(&hidden_creature(0), &HashSet::new(), &[], 0);
        assert_eq!(cov.checkable, 0);
        assert_eq!(cov.percent(), 0.0);
        assert_eq!(cov.summary(), "checked 0 of 0 hidden (0.0%), 0 cut");
    }

    #[test]
    fn percent_never_exceeds_one_hundred() {
        let cov = Coverage {
            hidden: 3,
            tagged: 0,
            checkable: 3,
            checked: 9,
            blocked: 0,
            cut: 0,
        };
        assert_eq!(cov.percent(), 100.0);
    }

    #[test]
    fn summary_omits_the_tagged_clause_when_nothing_is_tagged() {
        let creature = hidden_creature(4);
        let screens = [screen("h0", 1)];
        let cov = coverage(&creature, &HashSet::new(), &screens, 3);
        assert_eq!(cov.summary(), "checked 1 of 4 hidden (25.0%), 3 cut");
        assert!(!cov.summary().contains("tagged"));
    }

    #[test]
    fn summary_appends_the_tagged_clause_when_neurons_carry_tags() {
        let creature = hidden_creature(6);
        let screens = [screen("h2", 1), screen("h3", 2)];
        let cov = coverage(&creature, &tags(&["h0"]), &screens, 1);
        assert_eq!(
            cov.summary(),
            "checked 2 of 6 hidden (33.3%), 1 cut, 1 tagged"
        );
    }

    /// The fleet-scale example from Issue #40, rendered exactly.
    fn fleet_coverage() -> Coverage {
        Coverage {
            hidden: 5013,
            tagged: 42,
            checkable: 5013,
            checked: 1204,
            blocked: 0,
            cut: 7,
        }
    }

    #[test]
    fn the_description_block_renders_exactly_as_grq_will_paste_it() {
        assert_eq!(
            fleet_coverage().description(100),
            concat!(
                "🪒 Ockham neuron screening coverage\n",
                "checked:   1204 of 5013 hidden (24.0%)\n",
                "cut:       7 this run\n",
                "unchecked: 3809 remaining (~39 runs at 100/run)\n",
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
        for text in [cov.description(100), cov.summary()] {
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
            cov.description(100),
            concat!(
                "🪒 Ockham neuron screening coverage\n",
                "checked:   1204 of 5013 hidden (24.0%)\n",
                "cut:       7 this run\n",
                "unchecked: 3809 remaining (~39 runs at 100/run)\n",
                "blocked:   412 checked but structurally unprunable\n",
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
        let block = fleet_coverage().description(100);
        assert!(!block.contains("blocked:"), "{block}");
    }

    #[test]
    fn the_description_omits_the_tagged_line_when_nothing_is_tagged() {
        let cov = Coverage {
            tagged: 0,
            ..fleet_coverage()
        };
        let block = cov.description(100);
        assert!(!block.contains("tagged"), "{block}");
        assert!(
            block.ends_with("unchecked: 3809 remaining (~39 runs at 100/run)"),
            "{block}"
        );
    }

    /// `--candidates 0` must not produce a division by zero — the clause goes.
    /// A single remaining run reads `~1 run`, not `~1 runs`.
    #[test]
    fn the_last_remaining_run_is_singular() {
        let cov = Coverage {
            hidden: 4,
            tagged: 0,
            checkable: 4,
            checked: 2,
            blocked: 0,
            cut: 0,
        };
        assert!(
            cov.description(100)
                .ends_with("unchecked: 2 remaining (~1 run at 100/run)"),
            "{}",
            cov.description(100)
        );
    }

    #[test]
    fn a_zero_batch_size_drops_the_runs_clause_rather_than_rendering_inf() {
        let block = fleet_coverage().description(0);
        assert!(block.contains("unchecked: 3809 remaining\n"), "{block}");
        assert!(!block.contains("runs at"), "{block}");
        assert!(!block.contains("inf") && !block.contains("NaN"), "{block}");
    }

    #[test]
    fn complete_coverage_drops_the_runs_clause_because_nothing_is_left() {
        let cov = Coverage {
            hidden: 4,
            tagged: 0,
            checkable: 4,
            checked: 4,
            blocked: 0,
            cut: 1,
        };
        assert_eq!(
            cov.description(100),
            concat!(
                "🪒 Ockham neuron screening coverage\n",
                "checked:   4 of 4 hidden (100.0%)\n",
                "cut:       1 this run\n",
                "unchecked: 0 remaining"
            )
        );
    }

    #[test]
    fn more_records_than_checkable_neurons_never_renders_a_negative_remainder() {
        let cov = Coverage {
            hidden: 3,
            tagged: 0,
            checkable: 3,
            checked: 9,
            blocked: 0,
            cut: 0,
        };
        assert_eq!(cov.unchecked(), 0);
        assert!(cov.description(10).contains("unchecked: 0 remaining"));
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
    #[test]
    fn a_pre_93_coverage_json_reads_as_nothing_blocked() {
        let pre_93 = r#"{"hidden":5013,"tagged":42,"checkable":5013,"checked":1204,"cut":7}"#;
        let old: Coverage = serde_json::from_str(pre_93).unwrap();
        assert_eq!(old, fleet_coverage());
        assert_eq!(old.blocked, 0);
    }

    #[test]
    fn the_cut_count_is_carried_through_rather_than_derived() {
        let cov = coverage(&hidden_creature(4), &tags(&["h0"]), &[], 3);
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
        };
        assert_eq!(
            report.description(100),
            concat!(
                "🪒 Ockham neuron screening coverage\n",
                "checked:   1204 of 5013 hidden (24.0%)\n",
                "cut:       7 this run\n",
                "unchecked: 3809 remaining (~39 runs at 100/run)\n",
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
                cov.description(100)
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

    #[test]
    fn a_creature_with_no_hidden_neurons_is_complete_and_empty() {
        let cov = coverage(&hidden_creature(0), &HashSet::new(), &[], 7);
        assert_eq!(cov.hidden, 0);
        assert_eq!(cov.checkable, 0);
        assert_eq!(cov.percent(), 0.0);
        assert_eq!(cov.cut, 7);
    }
}
