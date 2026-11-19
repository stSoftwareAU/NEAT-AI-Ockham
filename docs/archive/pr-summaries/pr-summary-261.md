## Summary

The controlled ranking benchmark (#267, #274) and its synthetic pilot (#286)
are on Develop. #286 left the candidate-feature criterion partial because no
availability or cost case was made for the features the v1 vector lacks. This
PR adds that case to `docs/experiments/ranking-v1.md`, one item per feature,
each checked against the code it cites. Output variance, compiled latency and
gradient/Fisher signal are not cheap proxies here, so they stay out. The
functional-path sum is cheap and available but is not in the v1 schema, and
the archived pilot does not test it. Part of #261.

## Spec

### Intent and Rationale

- The issue names six candidate features; v1 uses sample error
  (`partialDelta`) and connections (`logFanIn`/`logFanOut`). For the rest, the
  doc now says why each is in or out, instead of the one-line "not added
  without an availability/cost case".

### Essential Design Decisions

- Docs only: the 15-column feature schema and the archived pilot are unchanged.
  Adding the path sum would change the schema, so it needs a new protocol
  version and a new pilot run with the real scorer.

### Undiscoverable Facts

- This host has no `rust_scorer` binary (`which rust_scorer` finds nothing), so
  the pilot could not be re-run with an extra feature in this run.

## Evidence

Docs-only change. Each claim was checked against the head code:

- `wall_clock_probe` (`ockham/src/strata/bench.rs:161`) calls
  `compile_creature` inside the `LATENCY_REPETITIONS = 3` loop, so each
  candidate costs three compiles and three timed passes.
- `SensitivityIndex` (`ockham/src/sensitivity.rs`) is `O(neurons + synapses)`,
  relaxes cycles conservatively, and is used by `LowOutputSensitivity` and
  `LowEstimatedEffect` in `ockham/src/ordering.rs`.
- `CandidateFeatures::downstream_sensitivity` (`ockham/src/features.rs:90`) is
  the one-hop `mean_abs × outgoing_weight`.
- `partialDelta` is the first pairwise column
  (`ockham/examples/priority_ordering_bench/evaluation.rs:14`).

Issue numbers cited: #105: Rank pruning candidates by downstream output
sensitivity.

## Acceptance Criteria

- **missing** — Candidate features: sample error, output variance, functional
  paths, connections, compiled latency and gradient/Fisher signal only if
  inexpensive/available — reason: the availability/cost case is now recorded,
  but functional paths are cheap and available and are not in the v1 vector or
  the archived pilot; testing them needs a v2 schema and a real-scorer pilot.
- **missing** — count all preparation/search costs — reason: unchanged from
  #286; binary builds, manifest work after its timing cutoff and result
  serialisation are still outside `totalResearchMs`.
- **missing** — Record negative findings and remove optional code that does
  not earn its maintenance cost — reason: unchanged from #286; the pairwise and
  transfer arms are kept until the representative run under the predeclared
  removal rule.

## Test Plan

- No test was added or edited; the change is one Markdown file.
- `markdownlint-cli2 docs/experiments/ranking-v1.md`: 0 issues.
- `codespell docs/experiments/ranking-v1.md`: no findings.
- `./quality.sh < /dev/null` on commit `5996989`: exit 0, "All quality checks passed!".
