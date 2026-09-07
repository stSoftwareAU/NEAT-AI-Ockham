#!/usr/bin/env python3
from pathlib import Path
import re


def one(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{label}: expected one match, found {count}: {old[:120]!r}")
    return text.replace(old, new, 1)


coverage = Path("ockham/src/coverage.rs")
text = coverage.read_text()

text = one(
    text,
    '#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]\n#[serde(rename_all = "camelCase")]\npub struct Passes {',
    '#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]\n#[serde(rename_all = "camelCase")]\npub struct Passes {',
    "Passes derive",
)

for fragment in (
    '    visited: HashSet<String>,\n',
    '            visited: HashSet::new(),\n',
    '        self.visited.insert(uuid.to_string());\n',
):
    text = one(text, fragment, "", "obsolete distinct visit state")

pattern = (
    r'    /// Distinct visit keys the sweep reached this run, retained for coverage\n'
    r'.*?'
    r'(?=    /// Every visit attempt this run, including repeats after sweep rebuilds\.)'
)
text, count = re.subn(pattern, "", text, count=1, flags=re.S)
if count != 1:
    raise SystemExit(f"obsolete distinct helper block: expected one match, found {count}")

# Passes::new remains the compatibility constructor used by old report fixtures.
# It has no population denominator, so it must not manufacture a 0.00
# creature-equivalent measurement. The production run uses Passes::measured.
text = one(
    text,
    '''            visits_run: visited_run,
            neuron_visits_run: visited_run,
            synapse_visits_run: 0,
            revisit_attempts_run: revisited_run,
            neuron_revisits_run: revisited_run,
            synapse_revisits_run: 0,
            equivalent_passes_run: 0.0,
''',
    '''            visits_run: 0,
            neuron_visits_run: 0,
            synapse_visits_run: 0,
            revisit_attempts_run: 0,
            neuron_revisits_run: 0,
            synapse_revisits_run: 0,
            equivalent_passes_run: 0.0,
''',
    "compatibility Passes constructor",
)

measured_block = '''        if self.visits_run > 0 {
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
'''
measured_with_fallback = measured_block[:-2] + ''' else if self.visited_run > 0 {
            out.push(format!(
                "{:<11}{} hidden neurons visited this run · {} revisited",
                "visits:", self.visited_run, self.revisited_run
            ));
        }
'''
text = one(text, measured_block, measured_with_fallback, "Passes rendering")

for old, new in (
    (
        '    fn visits_count_every_uuid_reached_while_progress_counts_only_new_ones() {',
        '    fn visit_attempts_count_repeat_work_while_progress_counts_only_new_keys() {',
    ),
    ('        assert_eq!(progress.visited(), 0);', '        assert_eq!(progress.visit_counts().total, 0);'),
    (
        '            progress.visited(),\n            3,\n            "a uuid is visited once, however often"',
        '            progress.visit_counts().total,\n            4,\n            "repeat visits are real work and stay counted"',
    ),
    (
        '        assert_eq!(progress.revisited(), 2, "h_a and h_b were already checked");',
        '        assert_eq!(progress.revisit_counts().total, 3, "two h_a visits plus h_b were revisits");',
    ),
    (
        '        assert_eq!(progress.revisited(), 2, "the new uuid is not a revisit");',
        '        assert_eq!(progress.revisit_counts().total, 3, "the new uuid is not a revisit");',
    ),
):
    text = one(text, old, new, "attempt semantics test")

# Exact-render tests now identify permutation completions as the strict metric.
for old, new in (
    (
        'passes:    7 complete this epoch · 1 this run · pass 8 in progress',
        'passes:    7 strict complete this epoch · 1 strict this run · pass 8 in progress',
    ),
    (
        'passes:    0 complete this epoch · 0 this run · pass 1 in progress',
        'passes:    0 strict complete this epoch · 0 strict this run · pass 1 in progress',
    ),
):
    if old not in text:
        raise SystemExit(f"missing coverage pass fixture: {old}")
    text = text.replace(old, new)

coverage.write_text(text)

run = Path("ockham/src/run.rs")
text = run.read_text()
text = one(
    text,
    'passes:    3 complete this epoch · 3 this run · pass 4 in progress',
    'passes:    3 strict complete this epoch · 3 strict this run · pass 4 in progress',
    "run pass fixture",
)
text = one(
    text,
    '        assert!(text.contains("visits:"), "{text}");\n',
    '''        assert!(
            text.contains("rescan:    3.33 creature-equivalent this run · 20 visit attempts"),
            "{text}"
        );
        assert!(
            text.contains("visits:    neurons 8 (0 revisits) · synapses 12 (0 revisits)"),
            "{text}"
        );
''',
    "run measured telemetry assertions",
)
run.write_text(text)

print("finalized #161/#163 compatibility and acceptance fixtures")
