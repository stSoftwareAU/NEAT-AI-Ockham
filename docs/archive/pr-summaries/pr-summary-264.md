# PR summary — Issue #264

## Summary

The audit command, its controls, the paper's FPS formula and the fixture tests
were merged in #269, #276 and #277. The one remaining gap is the
representative run, and with it the keep-or-remove decision. This host still
has no private corpus, mature creature or `rust_scorer`, and arxiv.org still
answers `CONNECT tunnel failed, response 403`, so the run cannot be made here.

This PR fixes that run's ground rules in advance. A new section in
`docs/preservation-audit.md`, "Representative run and retention rule (private,
not yet performed)", states:

- what a run must contain to count as representative, including a minimum of
  20 validation children with at least 5 better and 5 not;
- how `criterion.status` decides keep (`met`), remove (`notMet`) or no verdict
  (`notEvaluable`);
- the private build, freeze and run commands;
- which report fields may be published.

The README section links to it. No code, interface or default changes.
Part of #264.

## Spec

### Intent and Rationale

- The issue requires the keep-or-remove decision to rest on representative
  runs. The retention rule and the representative bar are fixed now, before
  any data exists, so the decision cannot be fitted to the result.
- Documentation only: the command already accepts every input such a run
  needs, so there is nothing to add to the code until the run exists.

### Essential Design Decisions

- The command does not check the representative bar. Whether a run is
  representative is the operator's own declaration, and the report's
  `evidence` field stays `"fixture-or-unverified"` whatever the inputs. The
  doc says so.
- The first complete representative run decides. A later run with a changed
  manifest must report the earlier one, so a bad result cannot be rerun away.
- `met` keeps the command as a report-only diagnostic and changes no
  acceptance or default. `notMet` removes it, as the #262 soft-gate code was
  removed.
- The minimum of 20 validation children with at least 5 on each side is an
  assumption chosen here. The issue asks for a predeclared sample size but
  gives no number.

### Undiscoverable Facts

- On 2026-11-16, `curl` to `arxiv.org` and `export.arxiv.org` failed with
  `CONNECT tunnel failed, response 403`, and `which rust_scorer` found nothing.
- A `--candidate-log` row carries `sampleDelta` and `fullDelta`
  (`ockham/src/telemetry.rs:543-544`) but no creature. The doc therefore says
  each child's creature file must come from the operator's own record of the
  cut.

## Evidence

This is a docs-only change, with no UI files.

- Docs updated: `docs/preservation-audit.md` ("Purpose and status" now links
  the new final section) and `README.md` (section "Preservation audit (opt-in,
  report-only)").
- The rule names in the retention table (`met`, `notMet`, `notEvaluable`)
  match `ockham/src/preservation.rs:1255-1268`. The publishable fields match
  the serialised `GateReport`, `Costs` and coverage structs in the same file
  (camelCase).
- `#262` is cited as provenance; `gh issue view 262` gives "Experiment: Learn
  one soft-gate ranking for nested pruning budgets", whose runtime was removed
  in #278.

## Acceptance Criteria

The issue has no Acceptance Criteria or Definition of done heading. These
parts of it remain undone:

- **missing** — "Deliver a reproducible command/manifest and machine-readable results with paired distributions, uncertainty, failures and actual coverage. Fixtures alone do not demonstrate an optimisation benefit" — reason: only fixture results exist; a representative run needs a private corpus, a mature creature and `rust_scorer`, none of which are on this host
- **missing** — "Compare diagnostic value and possible screening benefit with current sample error, measuring false rejection of genuinely better children" — reason: the comparison has run on fixture children only, not on real children
- **missing** — "Record negative findings and remove optional code that does not earn its maintenance cost. Do not claim success before representative runs" — reason: no success is claimed and the retention rule is now predeclared, but the keep-or-remove decision itself waits for a representative run

## Test Plan

- No test added or edited: the change is documentation only.
- `cargo test -p neat_ai_ockham --test readme_contract < /dev/null`: 10 passed.
- markdownlint-cli2 on both files: 0 issues. codespell: clean.
- `./quality.sh < /dev/null` on the docs commit: exit 0, "All quality checks
  passed!".
- Docs sweep: grepped `README.md` and `docs/preservation-audit.md` for
  `representative`. The other hits name other experiments, and the two hits for
  this audit now link the new section.

## Merge verification

CI and pull-request review have not been observed yet.

Not done, and still a risk:

- the representative run and the keep-or-remove decision;
- the documented `cargo build --release` command was not run here; the gate
  built and tested the debug profile;
- macOS was not checked.
