#!/usr/bin/env python3
from pathlib import Path


def replace_once(path: Path, old: str, new: str) -> None:
    text = path.read_text()
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{path}: expected one match, found {count}: {old[:80]!r}")
    path.write_text(text.replace(old, new, 1))

coverage = Path("ockham/src/coverage.rs")
run = Path("ockham/src/run.rs")

# #161/#163: count every visit attempt (not only distinct UUIDs), preserve a
# neuron/synapse split, and keep distinct coverage semantics separate.
replace_once(
    coverage,
    '''#[derive(Debug, Default)]
pub(crate) struct ScreenProgress {
    opening: HashSet<String>,
    added: HashSet<String>,
    visited: HashSet<String>,
}
''',
    '''/// Run-level visit counts, split by the kind of thing the sweep reached.
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
    visited: HashSet<String>,
    visits: VisitCounts,
    revisits: VisitCounts,
}
''')

replace_once(
    coverage,
    '''        Self {
            opening: screens.iter().map(|s| s.uuid.clone()).collect(),
            added: HashSet::new(),
            visited: HashSet::new(),
        }
''',
    '''        Self {
            opening: screens.iter().map(|s| s.uuid.clone()).collect(),
            added: HashSet::new(),
            visited: HashSet::new(),
            visits: VisitCounts::default(),
            revisits: VisitCounts::default(),
        }
''')

replace_once(
    coverage,
    '''    pub(crate) fn visit(&mut self, uuid: &str) {
        self.visited.insert(uuid.to_string());
    }

    /// Distinct hidden UUIDs the sweep reached this run, revisits included.
    pub(crate) fn visited(&self) -> usize {
        self.visited.len()
    }

    /// How many of those the fleet had already checked when the run opened.
    ///
    /// Useful work that added no unique coverage — reported beside `progress:`,
    /// never inside it.
    pub(crate) fn revisited(&self) -> usize {
        self.visited
            .iter()
            .filter(|uuid| self.opening.contains(*uuid))
            .count()
    }
''',
    '''    pub(crate) fn visit(&mut self, uuid: &str) {
        self.visits.observe(uuid);
        if self.opening.contains(uuid) {
            self.revisits.observe(uuid);
        }
        self.visited.insert(uuid.to_string());
    }

    /// Distinct visit keys the sweep reached this run, retained for coverage
    /// diagnostics that still care about uniqueness.
    pub(crate) fn visited(&self) -> usize {
        self.visited.len()
    }

    /// Distinct visit keys already checked when the run opened.
    pub(crate) fn revisited(&self) -> usize {
        self.visited
            .iter()
            .filter(|uuid| self.opening.contains(*uuid))
            .count()
    }

    /// Every visit attempt this run, including repeats after sweep rebuilds.
    pub(crate) fn visit_counts(&self) -> VisitCounts {
        self.visits
    }

    /// Visit attempts over keys already checked when this run opened.
    pub(crate) fn revisit_counts(&self) -> VisitCounts {
        self.revisits
    }
''')

# Add topology-tolerant counters to Passes without changing the existing JSON
# keys or old constructor used by compatibility tests/report fixtures.
replace_once(
    coverage,
    '''    pub revisited_run: usize,
}

impl Passes {
''',
    '''    pub revisited_run: usize,
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
''')

replace_once(
    coverage,
    '''        Self {
            sweep_restarts_run,
            sweeps_completed_epoch,
            current_pass: sweeps_completed_epoch + 1,
            visited_run,
            revisited_run,
        }
    }

    /// The description lines: the pass counters, then this run's visits.
''',
    '''        Self {
            sweep_restarts_run,
            sweeps_completed_epoch,
            current_pass: sweeps_completed_epoch + 1,
            visited_run,
            revisited_run,
            visits_run: visited_run,
            neuron_visits_run: visited_run,
            synapse_visits_run: 0,
            revisit_attempts_run: revisited_run,
            neuron_revisits_run: revisited_run,
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
''')

replace_once(
    coverage,
    '''    fn lines(&self) -> Vec<String> {
        let mut out = vec![format!(
            "{:<11}{} complete this epoch · {} this run · pass {} in progress",
            "passes:", self.sweeps_completed_epoch, self.sweep_restarts_run, self.current_pass
        )];
        if self.visited_run > 0 {
            out.push(format!(
                "{:<11}{} hidden neurons visited this run · {} revisited",
                "visits:", self.visited_run, self.revisited_run
            ));
        }
        out
    }
''',
    '''    fn lines(&self) -> Vec<String> {
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
        }
        out
    }
''')

# Use the measured constructor at the real run boundary. No other run logic is
# changed: the sweep already calls `progress.visit()` for candidates and skips.
replace_once(
    run,
    '''        let passes = crate::coverage::Passes::new(
            restarts,
            completed_epoch,
            progress.visited(),
            progress.revisited(),
        );
''',
    '''        let passes = crate::coverage::Passes::measured(
            restarts,
            completed_epoch,
            progress.visit_counts(),
            progress.revisit_counts(),
            cov.checkable,
        );
''')

# Keep the operator log explicit about strict-vs-equivalent semantics.
replace_once(
    run,
    '''            "{} · epoch corpus {} · pass {} ({} complete this epoch)",
            cov.summary(),
            crate::coverage::short_epoch(&corpus.identity),
            passes.current_pass,
            passes.sweeps_completed_epoch
''',
    '''            "{} · epoch corpus {} · pass {} ({} strict complete; {:.2} equivalent this run)",
            cov.summary(),
            crate::coverage::short_epoch(&corpus.identity),
            passes.current_pass,
            passes.sweeps_completed_epoch,
            passes.equivalent_passes_run
''')

print("applied #161/#163 rescan telemetry patch")
