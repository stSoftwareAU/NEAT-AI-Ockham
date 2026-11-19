## Summary

The controlled ranking benchmark (#267, #274) and its synthetic pilot (#286)
are on Develop. #286 left the candidate-feature criterion partial because no
availability or cost case was made for the features the v1 vector lacks. This
PR adds that case to `docs/experiments/ranking-v1.md`, one item per feature,
each checked against the code it cites. Output variance, compiled latency and
gradient/Fisher signal are not cheap proxies here, so they stay out. The
functional-path sum is cheap and available but is not in the v1 schema, and
the archived pilot does not test it. The PR also charges two costs that were
uncounted before. One is an operator-recorded `buildMs`, reported as
`buildCostMs`. The other is the measured serialisation of the evaluation
report, `serialisationCostMs`. Both are added to `totalResearchMs`. Part of
#261.

## Spec

### Intent and Rationale

- The issue names six candidate features; v1 uses sample error
  (`partialDelta`) and connections (`logFanIn`/`logFanOut`). For the rest, the
  doc now says why each is in or out, instead of the one-line "not added
  without an availability/cost case".

### Essential Design Decisions

- The 15-column feature schema and the archived pilot are unchanged. Adding the
  path sum would change the schema, so it needs a new protocol version and a new
  pilot run with the real scorer.
- `buildMs` follows the `preparationMs` pattern. It is optional, defaults to
  zero and is left out of the serialised output when zero, so existing manifest
  identities hold. `ranking_pilot manifest` reads it from an optional
  `<dir>/build-ms` through the same contained, size-capped `recorded_ms` reader
  as `inputs-ms`.
- `evaluation::serialise_charged` times one serialisation, rounded up to at
  least 1 ms. It then writes a second serialisation carrying that charge, so the
  measured pass stands in for the written one. The disk write is not charged.

### Undiscoverable Facts

- This host has no `rust_scorer` binary (`which rust_scorer` finds nothing), so
  the pilot could not be re-run with an extra feature in this run.

## Evidence

- `buildMs` → `buildCostMs`: `ockham/examples/priority_ordering_bench/manifest.rs`
  and `evaluation.rs`. Tested by
  `manifest_build_time_is_reported_and_added_to_the_research_total`.
- `serialisationCostMs`: `evaluation::serialise_charged`, called from the
  `evaluate` arm of `experiment.rs`. Tested by
  `result_serialisation_is_charged_to_the_research_total`.
- Red-check: removing each addend from the total made exactly its own test
  fail (24 passed, 2 failed); restoring it made them pass.
- `ranking_pilot`: tested by
  `recorded_build_time_is_written_as_build_ms_and_absent_is_zero`. The template
  gains `"buildMs": 0`, so `manifest_keys_match_the_template` still holds.

The feature-availability claims in the docs were checked against the head code:

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

<!-- vibe-spec-review inputs="diff+issue-body" -->

- **partial** — Candidate features: sample error, output variance, functional
  paths, connections, compiled latency and gradient/Fisher signal only if
  inexpensive/available — evidence: `docs/experiments/ranking-v1.md` (feature
  assessment) — reviewer: partial — reason: every named feature now has an
  availability/cost case, but functional paths are cheap and available and are
  still not in the v1 vector or the comparison. Adding them needs a v2 schema
  and a real-scorer pilot.
- **partial** — count all preparation/search costs — evidence:
  `ockham/examples/priority_ordering_bench/tests.rs::manifest_build_time_is_reported_and_added_to_the_research_total`,
  `ockham/examples/priority_ordering_bench/tests.rs::result_serialisation_is_charged_to_the_research_total`,
  `ockham/examples/ranking_pilot.rs::recorded_build_time_is_written_as_build_ms_and_absent_is_zero`
  — reviewer: partial — reason: build time and result serialisation are now
  charged, but `buildMs` is hand-recorded. The final file write, the manifest
  step after its first write and the source searches' hand-taken timing are
  still uncounted, and the archived pilot predates both new fields.
- **partial** — Record negative findings and remove optional code that does
  not earn its maintenance cost — evidence: `docs/experiments/ranking-v1.md`
  (Synthetic pilot) — reviewer: partial — reason: the synthetic negative finding
  and a predeclared removal rule are recorded, but no code is removed. The
  pairwise and transfer arms wait for the representative run, which needs
  private data not available on this host.

## Standards Review

<!-- vibe-standards-review inputs="diff+CODING-STANDARDS.md" -->

- **violation** — commit-message prefix (`CONTRIBUTING.md`, "Commit
  messages": Ockham uses 🪒) — evidence: commits `5996989`, `c90de36`,
  `d22c058` — reason: the earlier commit subjects lack the prefix and are not
  rewritten. This summary commit uses it. The repository has no
  CODING-STANDARDS.md, so `CONTRIBUTING.md` was the standard checked.
- **clean** — rustfmt, clippy with the gate's lints, codespell, markdownlint,
  template JSON validity, rustdoc on new public items, no version bump needed
  for example-only changes, and manifest identity preserved for zero `buildMs`.

## Test Plan

- `cargo test --example priority_ordering_bench < /dev/null`: 26 passed.
- `cargo test --example ranking_pilot < /dev/null`: 26 passed.
- Red-check of the two cost tests: described under Evidence.
- `markdownlint-cli2 docs/experiments/ranking-v1.md`: 0 issues.
- `codespell` on the changed files: no findings.
- `./quality.sh < /dev/null` on commit `d22c058`: exit 0, "All quality checks passed!".
