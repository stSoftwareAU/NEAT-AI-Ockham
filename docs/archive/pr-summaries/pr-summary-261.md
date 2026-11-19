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

The costs #288 left uncharged are now measured:

- Searches and builds: `ranking_pilot search <dir> <source-task> --
  <program> [args...]` times a source search into `acquisition-ms`, which the
  operator used to time by hand. `ranking_pilot time <new-file> -- <program>
  [args...]` times any other command the same way, such as the build behind
  `build-ms` or a private manifest's search.
- Phase files: each one the bench writes (preflight, pilot, frozen, audit,
  results) gets a `.write-ms` sidecar, because a file cannot hold the time of
  writing itself. The sidecar records the time of serialising and writing it.
- The manifest step's work after its first write goes to
  `<dir>/manifest-tail-ms`.
- Charging: `freeze` adds the pilot's sidecar to `pilotCostMs`. `evaluate`
  charges the preflight, freeze and audit sidecars, plus the manifest tail when
  present, as the new `recordedWriteCostMs` inside `totalResearchMs`. Only the
  results file's own write stays outside, in `results.json.write-ms`.

The archived v1 pilot's Reproduce section is now pinned to protocol-v1 revisions,
`aa41403` up to and including `a4f0d274` (PR #288).

No v2 pilot has been run, and the removal decision still waits for a
representative run. Part of #261.

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
- Write times go into sidecars, and the next stage that reads a file charges
  that file's sidecar, so all but the last write land in `totalResearchMs`. A
  run's whole measured cost is `totalResearchMs` plus `results.json.write-ms`.
  Only the writes of the sidecars' few bytes themselves are untimed.
- `evaluate` refuses a missing preflight, freeze or audit sidecar, because the
  bench always writes one. `manifest-tail-ms` is optional, because a private
  manifest the generator did not write has none.
- Every sidecar and timing record is created exclusively. An existing one is
  refused before the file it times is written, or before the timed program
  runs. The manifest step therefore still writes no history and no manifest
  when it refuses. `search` and `time` record nothing when the program fails.
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
- `audit_scores_sampled_losers_individually_and_preserves_parent` (extended)
  runs `collect_pool` on `shortcut_edge_creature`. Every featured row carries a
  finite, non-negative `pathSensitivity`, and a featureless row carries none.
  Each hidden-neuron row equals `mean_abs × importance(uuid)` and each synapse
  row equals `mean_abs × |weight| × importance(target)`, both from a
  `SensitivityIndex` on the same parent. The serialised pool's JSON rows carry
  `pathSensitivity` exactly when the row has one. Red check: with `collect_pool`
  recording `None` instead, the test fails with "has no path sensitivity".
- `search_times_the_ordinary_run_into_acquisition_ms`: `sh -c "sleep 0.05"`
  records at least 50 ms, and the manifest step's reader returns that value. A
  second run is refused ("already exists") and leaves the record unchanged. A
  failing program (`exit 3`) records nothing, and so does a non-source task.
- `manifest_step_records_its_tail_after_the_first_write`: the step writes a
  `manifest-tail-ms` of at least 1 ms. An existing one is refused before
  `manifest.json` or any `history.jsonl` is written, and is left unchanged.
- `results_write_time_is_recorded_in_a_sidecar_the_results_cannot_hold`:
  `write_timed` leaves the results bytes unchanged. It records the returned
  milliseconds (at least 1) in `results.json.write-ms` and refuses an existing
  results file. An existing sidecar is refused before the results file is
  created. A phase file written through `write` gets a sidecar of at least 1 ms.
- `read_ms_refuses_a_missing_malformed_oversized_or_symlinked_sidecar`:
  missing, non-numeric, 65-byte, symlinked and dangling sidecars are refused,
  and `12` reads as 12.
- `path_sensitivities_refuse_a_non_finite_value_and_take_the_first_duplicate_edge`:
  NaN, infinite and negative `mean_abs` are refused with "non-finite or
  negative path sensitivity". A twice-listed edge takes its first weight (2,
  not 5).
- `earlier_phase_write_times_are_summed_and_the_manifest_tail_is_optional`:
  `evaluate`'s reader refuses a missing audit sidecar. It sums 3 + 5 + 7 = 15
  with no manifest tail, and 26 once `manifest-tail-ms` holds 11.
- `recorded_write_cost_is_reported_and_added_to_the_research_total_once`:
  charging 4,321 ms sets `recordedWriteCostMs` and raises `totalResearchMs` by
  exactly 4,321. No other field changes.
- `freezing_a_pilot_file_charges_its_recorded_write_time`: `freeze_file` refuses
  a pilot with no sidecar. With a 250 ms sidecar, its `pilotCostMs` is the
  direct freeze's plus 250.
- `time_records_any_successful_command_into_a_new_file`: a 50 ms command records
  at least 50 ms. An existing record, or a record whose directory is missing,
  is refused before the program runs (its marker file is never created). A
  failing or missing program records nothing, and the manifest step writes a
  measured `build-ms` as `buildMs`.
- The refusal of negative or non-finite path values and the `freeze`
  version check have the tests listed in the Test Plan.

Issue numbers cited: #105: Rank pruning candidates by downstream output
sensitivity.

## Acceptance Criteria

<!-- vibe-spec-review inputs="diff+issue-body" -->

- **met** — Candidate features: sample error, output variance, functional
  paths, connections, compiled latency and gradient/Fisher signal only if
  inexpensive/available — evidence:
  `ockham/examples/priority_ordering_bench/tests.rs::path_sensitivity_sums_every_downstream_path_not_just_one_hop`,
  `::path_sensitivity_columns_are_appended_and_match_their_names`,
  `::audit_scores_sampled_losers_individually_and_preserves_parent`,
  `docs/experiments/ranking-v1.md` (feature assessment) — reviewer: met
- **met** — count all preparation/search costs — evidence:
  `ockham/examples/ranking_pilot.rs::search_times_the_ordinary_run_into_acquisition_ms`,
  `::time_records_any_successful_command_into_a_new_file`,
  `::manifest_step_records_its_tail_after_the_first_write`,
  `ockham/examples/priority_ordering_bench/tests.rs::earlier_phase_write_times_are_summed_and_the_manifest_tail_is_optional`,
  `::recorded_write_cost_is_reported_and_added_to_the_research_total_once`,
  `::freezing_a_pilot_file_charges_its_recorded_write_time` — reviewer: met
- **partial** — Record negative findings and remove optional code that does
  not earn its maintenance cost — evidence: `docs/experiments/ranking-v1.md`
  (synthetic negative finding, predeclared removal rule now covering the path
  columns), `docs/research/ranking-pilot-v1/README.md` (maintenance decision)
  — reviewer: partial — reason: whether the pairwise and transfer arms and the
  new path columns earn their cost needs the representative v2 run, which needs
  private data and a scorer that this host does not have.

## Standards Review

<!-- vibe-standards-review inputs="diff+CODING-STANDARDS.md" -->

- **violation** — `write_manifest`'s `# Errors` doc omitted the
  `manifest-tail-ms` refusal — evidence: `ockham/examples/ranking_pilot.rs`
  (`write_manifest` doc) — reason: fixed in this diff
- **violation** — results file written before its sidecar was checked, leaving
  a results file with no write-time record — evidence:
  `ockham/examples/priority_ordering_bench/experiment.rs` (`write_timed`) —
  reason: fixed in this diff
- **violation** — `time` ran the program before checking that the record's
  directory exists — evidence: `ockham/examples/ranking_pilot.rs` (`run_timed`)
  — reason: fixed in this diff
- **violation** — sidecars charged without a digest, and the doc did not say so
  — evidence: `docs/experiments/ranking-v1.md` (cost reporting) — reason: fixed
  in this diff
- **violation** — no test for `path_sensitivities`' own refusal and
  duplicate-edge rule, or for a symlinked sidecar — evidence:
  `ockham/examples/priority_ordering_bench/tests.rs` — reason: fixed in this diff
- **violation** — test name and `read_ms` doc did not match the function —
  evidence: `ockham/examples/priority_ordering_bench/tests.rs`, `experiment.rs`
  — reason: fixed in this diff
- **clean** — exclusive creation, refusal before first write, error strings
  naming paths, doc comments, version gates, test determinism (loose timing
  bounds only), docs against code, contribution rules

## Test Plan

- Added in `ockham/examples/priority_ordering_bench/tests.rs` and
  `ockham/examples/ranking_pilot.rs`: the tests under Evidence, plus
  `search_subcommand_parses_the_program_after_a_separator` (`search` and `time`
  need the `--` separator and a program), plus
  `labelled_refuses_a_non_finite_or_negative_path_sensitivity` (`-1` and NaN
  are refused with "non-finite or negative path sensitivity", and `0` is
  accepted) and `freeze_refuses_a_version_one_pilot_and_accepts_version_two`
  (the same pilot is refused at version 1 and accepted at version 2). The
  path tests first failed to compile against the unchanged bench, because the
  field and the helper did not exist yet.
- `cargo test --example priority_ordering_bench < /dev/null`: 37 passed.
- `cargo test --example ranking_pilot < /dev/null`: 30 passed.
- `cargo test --test ranking_pilot_archive < /dev/null`: 1 passed.
- `markdownlint-cli2` and `codespell` on `docs/experiments/ranking-v1.md` and
  `docs/research/ranking-pilot-v1/README.md`: no findings.
- `./quality.sh < /dev/null` on the final code: exit 0, "All quality checks
  passed!".
- Not checked: CI, macOS, and a v2 pilot with the real scorer.
- Spec and standards reviewer sub-agents ran over the diff. Their findings are
  in the two blocks above, and every standards finding is fixed here.
- Fixture updates in the same file (no assertion removed or weakened): the
  manifest fixture protocol is now `ockham-ranking-v2`, `artificial_audit` is
  `format_version: 2`, and the two `AuditRow` literals gain
  `path_sensitivity: None`.
