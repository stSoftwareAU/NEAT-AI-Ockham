## Summary

The ranking benchmark (#267, #274), its synthetic pilot (#286) and the
feature-availability case plus the build and serialisation costs (#288) are on
Develop. #288 found the functional-path signal cheap and available but left it
out of the comparison. This PR adds it to the opt-in bench as protocol v2.

`audit::path_sensitivities` computes an all-paths output sensitivity for each
neuron and synapse visit from the existing `SensitivityIndex` (Issue #105). It
runs once per parent in the audit's feature stage, so `features_ms` charges it.
The audit row records it as `pathSensitivity`. The pairwise arms read it as two
appended columns, `pathSensitivityAvailable` and `logPathSensitivity`. The
library's 15-column schema, telemetry and every ordinary code path are
unchanged.

No v2 pilot has been run, the remaining cost gaps are still open, and the
removal decision still waits for a representative run. Part of #261.

## Spec

### Intent and Rationale

- The issue asks for functional paths "only if inexpensive/available". The
  index is `O(neurons + synapses)` per creature and already used by two
  library orderings, so the bench reuses it rather than adding a new signal.

### Essential Design Decisions

- The path columns go only into the pairwise inputs, after the 15 library
  columns. The composite control keeps its fixed, untuned weights, and the
  partial and random arms are unchanged.
- Source telemetry has no path value, so historical rows get both columns as
  zero. The transfer arm can therefore learn the column's weight only from
  target calibration rows. Computing it for sources would mean loading each
  source parent at evaluation time.
- Persisted shape: the audit and pilot `formatVersion` goes from 1 to 2.
  `freeze` refuses a version 1 pilot and `evaluate` refuses a version 1 audit,
  so a v1 file is never read with the v2 vector. The manifest protocol is now
  `ockham-ranking-v2` in the bench, in `ranking_pilot` and in the template.
- The protocol doc keeps its `ranking-v1.md` file name so that existing links
  still work.

### Undiscoverable Facts

- This host has no `rust_scorer` binary (`which rust_scorer` finds nothing),
  so no v2 pilot could be run.

## Evidence

Backend and research CLI only; there is no UI change.

- `path_sensitivity_sums_every_downstream_path_not_just_one_hop`: on the chain
  input → h1 →(2) h2 →(3) output, h1 scores `0.5 × 6` and the h1→h2 synapse
  scores `0.5 × 2 × 3`. A neuron that reaches no output scores 0. h1's value
  differs from its one-hop `downstream_sensitivity`.
- `path_sensitivity_columns_are_appended_and_match_their_names`: a present
  value gives `[1, ln(1 + p)]`, an absent one gives `[0, 0]`, and the vector
  length equals `feature_names()`.
- `evaluation_refuses_a_version_one_audit_and_accepts_version_two`: the same
  audit is refused as version 1 and evaluated as version 2.
- The refusal of negative or non-finite path values and the `freeze`
  version check have the tests listed in the Test Plan.

Issue numbers cited: #105: Rank pruning candidates by downstream output
sensitivity.

## Acceptance Criteria

- **missing** — Candidate features: sample error, output variance, functional
  paths, connections, compiled latency and gradient/Fisher signal only if
  inexpensive/available — reason: functional paths are now in the v2 pairwise
  inputs (`ockham/examples/priority_ordering_bench/audit.rs`,
  `evaluation.rs`), but no v2 pilot has run them, because this host has no
  scorer binary. Source histories carry no path value.
- **missing** — count all preparation/search costs — reason: three costs
  are still not charged: writing the final results file, the manifest step
  after its first write, and the source searches' timing, which the operator
  records by hand.
- **missing** — Record negative findings and remove optional code that does
  not earn its maintenance cost — reason: the predeclared rule removes the
  pairwise and transfer arms only if the representative run, which needs
  private data, also shows no benefit.

## Test Plan

- Added in `ockham/examples/priority_ordering_bench/tests.rs`: the three tests
  under Evidence, plus the refusal tests listed below.
- Fixture updates in the same file (no assertion removed or weakened): the
  manifest fixture protocol is now `ockham-ranking-v2`, `artificial_audit` is
  `format_version: 2`, and the two `AuditRow` literals gain
  `path_sensitivity: None`.
