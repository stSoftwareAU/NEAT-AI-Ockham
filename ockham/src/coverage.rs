//! Screening coverage over the **current** incumbent (Issue #37).
//!
//! One place answers "how far has Ockham got through this creature?", so the
//! `ockham` tag, the commit description and `--report` can never disagree.
//!
//! Two rules make the denominator honest:
//!
//! - **The current incumbent is the whole world.** A screen record for a uuid
//!   that is no longer on the creature is ignored entirely — it neither raises
//!   `checked` nor `hidden`.
//! - **Tagged neurons are not counted as checkable.** They are excluded from
//!   the denominator and reported separately. Selection no longer exempts them
//!   (#63): Ockham proposes tagged neurons like any other, so this denominator
//!   now *undercounts* the true one until the coverage child of #63 lands.
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
    /// Hidden neurons carrying GRQ-provenance tags Ockham must not touch.
    pub tagged: usize,
    /// `hidden - tagged`: the denominator.
    pub checkable: usize,
    /// Checkable UUIDs with at least one screen record.
    pub checked: usize,
    /// Hidden neurons removed this run.
    pub cut: usize,
}

impl Coverage {
    /// `checked / checkable * 100`; `0.0` when nothing is checkable.
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
    /// `checked 1204 of 4971 hidden (24.2%), 7 cut, 42 tagged skipped`. The
    /// `X of Y` denominator is [`Self::checkable`], so it always agrees with
    /// the percentage. The tagged clause is omitted when nothing is tagged.
    pub fn summary(&self) -> String {
        let mut out = format!(
            "checked {} of {} hidden ({:.1}%), {} cut",
            self.checked,
            self.checkable,
            self.percent(),
            self.cut
        );
        if self.tagged > 0 {
            out.push_str(&format!(", {} tagged skipped", self.tagged));
        }
        out
    }

    /// Checkable UUIDs with no screen record yet.
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
    /// checked:   1204 of 4971 hidden (24.2%)
    /// cut:       7 this run
    /// unchecked: 3767 remaining (~38 runs at 100/run)
    /// skipped:   42 tagged (GRQ provenance, outside the denominator)
    /// ```
    ///
    /// Line-oriented and stable: GRQ pastes it into a `git commit` description.
    /// `candidates` is the configured `--candidates` batch size; the
    /// runs-remaining clause is **omitted** rather than rendering `inf` when
    /// the batch size is zero, and there is nothing to estimate once coverage
    /// is complete. The `skipped:` line is omitted when nothing is tagged.
    /// No trailing newline — [`write_files`] adds one.
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
        if self.tagged > 0 {
            out.push_str(&format!(
                "\n{:<11}{} tagged (GRQ provenance, outside the denominator)",
                "skipped:", self.tagged
            ));
        }
        out
    }
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
/// straight into [`Coverage`] keeps working — serde ignores the extra key.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CoverageReport {
    /// Screening coverage of the final incumbent.
    #[serde(flatten)]
    pub coverage: Coverage,
    /// What the run tried and kept; absent when nothing was screened.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub winners: Option<Winners>,
}

impl CoverageReport {
    /// Coverage with no winner figures — the pre-Issue-#59 artefact.
    pub fn new(coverage: Coverage) -> Self {
        Self {
            coverage,
            winners: None,
        }
    }

    /// The full commit-description block: coverage lines, then winner lines.
    ///
    /// With no winners this is byte-identical to [`Coverage::description`].
    pub fn description(&self, candidates: usize) -> String {
        let mut out = self.coverage.description(candidates);
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

/// Count coverage of `creature` from `screens`, excluding `tagged` UUIDs.
///
/// `cut` is the number of hidden neurons removed this run — it is carried
/// through rather than derived, because the creature in hand no longer holds
/// the neurons that were cut.
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
    let checkable_uuids: HashSet<&str> = hidden_uuids
        .into_iter()
        .filter(|uuid| !tagged.contains(*uuid))
        .collect();
    // A HashSet of the screened UUIDs, so a uuid screened many times — by this
    // host or another — still counts once.
    let checked: HashSet<&str> = screens
        .iter()
        .map(|s| s.uuid.as_str())
        .filter(|uuid| checkable_uuids.contains(uuid))
        .collect();
    Coverage {
        hidden,
        tagged: tagged_hidden,
        checkable: hidden - tagged_hidden,
        checked: checked.len(),
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
        }
    }

    fn tags(uuids: &[&str]) -> HashSet<String> {
        uuids.iter().map(|u| (*u).to_string()).collect()
    }

    #[test]
    fn tagged_neurons_leave_the_denominator_and_are_reported_separately() {
        let creature = hidden_creature(10);
        let screens = [screen("h2", 1), screen("h3", 2), screen("h4", 3)];
        let cov = coverage(&creature, &tags(&["h0", "h1"]), &screens, 0);
        assert_eq!(cov.hidden, 10);
        assert_eq!(cov.tagged, 2);
        assert_eq!(cov.checkable, 8);
        assert_eq!(cov.checked, 3);
        assert_eq!(cov.percent(), 37.5);
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

    #[test]
    fn a_screened_tagged_uuid_never_counts_as_checked() {
        let creature = hidden_creature(4);
        let screens = [screen("h0", 1), screen("h1", 2)];
        let cov = coverage(&creature, &tags(&["h1"]), &screens, 0);
        assert_eq!(cov.checkable, 3);
        assert_eq!(cov.checked, 1, "a tagged uuid is outside the coverage set");
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

    #[test]
    fn nothing_checkable_yields_zero_percent_without_panicking() {
        let creature = hidden_creature(2);
        let cov = coverage(&creature, &tags(&["h0", "h1"]), &[], 0);
        assert_eq!(cov.checkable, 0);
        assert_eq!(cov.checked, 0);
        assert_eq!(cov.percent(), 0.0);
        assert_eq!(
            cov.summary(),
            "checked 0 of 0 hidden (0.0%), 0 cut, 2 tagged skipped"
        );
    }

    #[test]
    fn percent_never_exceeds_one_hundred() {
        let cov = Coverage {
            hidden: 3,
            tagged: 0,
            checkable: 3,
            checked: 9,
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
    fn summary_appends_the_tagged_clause_when_neurons_are_skipped() {
        let creature = hidden_creature(6);
        let screens = [screen("h2", 1), screen("h3", 2)];
        let cov = coverage(&creature, &tags(&["h0"]), &screens, 1);
        assert_eq!(
            cov.summary(),
            "checked 2 of 5 hidden (40.0%), 1 cut, 1 tagged skipped"
        );
    }

    /// The fleet-scale example from Issue #40, rendered exactly.
    fn fleet_coverage() -> Coverage {
        Coverage {
            hidden: 5013,
            tagged: 42,
            checkable: 4971,
            checked: 1204,
            cut: 7,
        }
    }

    #[test]
    fn the_description_block_renders_exactly_as_grq_will_paste_it() {
        assert_eq!(
            fleet_coverage().description(100),
            concat!(
                "🪒 Ockham neuron screening coverage\n",
                "checked:   1204 of 4971 hidden (24.2%)\n",
                "cut:       7 this run\n",
                "unchecked: 3767 remaining (~38 runs at 100/run)\n",
                "skipped:   42 tagged (GRQ provenance, outside the denominator)"
            )
        );
    }

    #[test]
    fn the_description_omits_the_skipped_line_when_nothing_is_tagged() {
        let cov = Coverage {
            tagged: 0,
            checkable: 5013,
            ..fleet_coverage()
        };
        let block = cov.description(100);
        assert!(!block.contains("skipped"), "{block}");
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
        assert!(block.contains("unchecked: 3767 remaining\n"), "{block}");
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
        assert_eq!(text, format!("{}\n", cov.description(100)));

        let json = std::fs::read_to_string(dir.join(COVERAGE_JSON_FILE)).unwrap();
        let back: Coverage = serde_json::from_str(&json).unwrap();
        assert_eq!(back, cov, "the machine-readable contract must round-trip");
        assert!(json.contains("\"checkable\": 4971"), "{json}");
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
            winners: Some(fleet_winners()),
        };
        assert_eq!(
            report.description(100),
            concat!(
                "🪒 Ockham neuron screening coverage\n",
                "checked:   1204 of 4971 hidden (24.2%)\n",
                "cut:       7 this run\n",
                "unchecked: 3767 remaining (~38 runs at 100/run)\n",
                "skipped:   42 tagged (GRQ provenance, outside the denominator)\n",
                "winners:   38 screened · 22 confirmed · 1 applied · 21 carried\n",
                "bundles:   9 plans · best 14 cuts (Δ +1.2e-4) · 3 skipped\n",
                "dropped:   12 entries over budget (est 18s/creature)"
            )
        );
    }

    /// The block is pasted into every fleet host's check-in commit, so a run
    /// with nothing to say must add no empty lines and no `0 of 0` filler.
    #[test]
    fn a_run_with_no_winners_renders_exactly_todays_block() {
        let cov = fleet_coverage();
        let report = CoverageReport::new(cov);
        assert_eq!(report.description(100), cov.description(100));
        assert!(!report.description(100).contains("winners:"));
        assert!(!report.description(100).contains("bundles:"));
        assert!(!report.description(100).contains("dropped:"));
        assert!(!Winners::default().has_any());
    }

    #[test]
    fn each_winner_line_is_omitted_when_it_has_nothing_to_report() {
        let report = CoverageReport {
            coverage: fleet_coverage(),
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
            winners: Some(fleet_winners()),
        };
        write_files(&dir, &report, 100).unwrap();

        let json = std::fs::read_to_string(dir.join(COVERAGE_JSON_FILE)).unwrap();
        let old: Coverage = serde_json::from_str(&json).unwrap();
        assert_eq!(old, fleet_coverage(), "existing fields must not move");
        let back: CoverageReport = serde_json::from_str(&json).unwrap();
        assert_eq!(back, report, "the new key must round-trip");
        assert!(json.contains("\"checkable\": 4971"), "{json}");
        assert!(json.contains("\"winners\": {"), "{json}");

        let text = std::fs::read_to_string(dir.join(COVERAGE_TEXT_FILE)).unwrap();
        assert_eq!(text, format!("{}\n", report.description(100)));
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
