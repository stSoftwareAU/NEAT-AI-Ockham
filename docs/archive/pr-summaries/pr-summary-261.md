## Summary

Protocol v2 of the opt-in ranking bench (#288, #289) adds the functional-path
columns and measures every cost, but no v2 pilot had been run. This PR runs one:
a synthetic, non-representative pilot with the real scorer at code revision
`7cdabe6` (Develop after #289). It archives the run as
`docs/research/ranking-pilot-v2/` and adds a test that checks the archive
against its own digests and costs.

The verdict is `no-demonstrated-benefit`. In the primary cell,
transfer-pairwise minus partial was −0.611 [−0.75, −0.333]. The audit pools are
identical to v1's, so at history fraction 0 the path columns are the only
changed input. There they changed no evaluation-budget-5 cell, and at budget 2
target-pairwise minus random fell from 0.0 to −0.083. The path columns did not
help on this pilot. Under the predeclared rule, removal waits for the
representative v2 run, which needs private data. Part of #261.

No production code changes. The diff is the archive, the test and three docs.

## Spec

### Intent and Rationale

- The bench, the path columns and the cost accounting are already on Develop.
  The gap left was evidence: no v2 run existed. Running it with the documented
  v2 commands (`ranking_pilot search` and `time`) also exercises those commands
  end to end.
- The v2 archive reuses v1's design, inputs and scorer, so the only intended
  change between the two runs is the protocol.

### Essential Design Decisions

- The corpora are not recommitted. They are byte-identical to the v1 archive's
  `records.bin`, and the test asserts each manifest `corpusSha256` against
  those committed files.
- The v1 archive test is unchanged. Its helpers now delegate to generic
  helpers that take the archive name.
- The removal rule predeclared in `docs/experiments/ranking-v1.md` is not
  changed after the fact. A synthetic run does not trigger it, so the pairwise
  arms and path columns stay research-only.

### Undiscoverable Facts

- Run on 19 November 2026 on aarch64 with rustc 1.99.1. NEAT-AI-scorer
  `742c5df` built to the same executable digest as v1 (`d057a334…`).
- The cargo target directory already held earlier build output, so the
  47,570 ms build charge is partly incremental. The scorer's own build (about
  95 s) is not charged, as in v1.
- The source searches were re-run, so the histories differ from v1's. The raw
  row counts are the same (365 and 383).

## Evidence

Backend and research-archive change only, so no screenshot is needed.

```mermaid
flowchart LR
    B[bootstrap build<br/>shell clock] --> I[ranking_pilot inputs]
    I --> T[ranking_pilot time<br/>build-ms]
    T --> S[ranking_pilot search<br/>source-1, source-2]
    S --> M[ranking_pilot manifest]
    M --> C[check] --> P[pilot] --> F[freeze] --> A[audit] --> E[evaluate]
    E --> R[results.json<br/>no-demonstrated-benefit]
```

- `ockham/tests/ranking_pilot_archive.rs::archived_v2_ranking_pilot_matches_its_digests_costs_and_path_signal`
  checks:
  - the protocol, verdict and manifest identity across results, freeze and
    preflight;
  - the audit, pilot and history digests;
  - each corpus digest against the v1 files, and `formatVersion` 2;
  - that all 632 featured audit rows carry a finite, non-negative
    `pathSensitivity` and the 8 failure rows carry none;
  - the primary-cell sign, coverage (640 selected, 632 fully scored, 0 missing
    features), and that the ten charged stages sum to 50,574 ms;
  - `buildCostMs` = `build-ms` + `bootstrap-build-ms` = 47,570, and the
    recorded writes total 6 ms.
- Path-column attribution: a row-by-row comparison of the decompressed v1 and
  v2 audits found the same candidate digests, kinds, full and sampled deltas
  and feature vectors in all 640 rows. 104 of the 120 paired cells match v1
  exactly. The 16 that differ are all at evaluation budget 2 and calibration
  budget 8.
- `docs/research/ranking-pilot-v2/README.md` records the pins, costs, results,
  path-column effect, maintenance decision, checksums and the exact commands.

## Acceptance Criteria

<!-- vibe-spec-review inputs="diff+issue-body" -->

- **partial** — Record negative findings and remove optional code that does
  not earn its maintenance cost. — evidence:
  `docs/research/ranking-pilot-v2/README.md`,
  `docs/research/ranking-pilot-v1/README.md`, `docs/experiments/ranking-v1.md`,
  `ockham/tests/ranking_pilot_archive.rs::archived_v2_ranking_pilot_matches_its_digests_costs_and_path_signal`
  — reviewer: partial — reason: the v1 and v2 negative synthetic findings are
  recorded, but nothing is removed yet. Under the predeclared rule, the
  pairwise arms, the path columns and `audit::path_sensitivities` are removed
  only if the representative v2 run also shows no benefit over `partial`. That
  run needs private corpora and creatures this host does not have, and
  removing the code first would make the run impossible.

## Standards Review

<!-- vibe-standards-review inputs="diff+CODING-STANDARDS.md" -->

- **clean** — reviewed against `CONTRIBUTING.md`, because the repository has
  no `CODING-STANDARDS.md`. Checked: no version bump is needed (only a test and
  docs change), the new test only reads the archive and leaves the production
  path alone, no changelog entry was added, and `rustfmt`, `clippy -D warnings`,
  the `ranking_pilot_archive` tests (2 passed), `codespell` and
  `markdownlint-cli2` all pass.

## Test Plan

- Added `archived_v2_ranking_pilot_matches_its_digests_costs_and_path_signal`
  in `ockham/tests/ranking_pilot_archive.rs`. The v1 test
  `archived_ranking_pilot_matches_its_digests_and_negative_finding` is
  byte-identical to its text on Develop. Only its helpers delegate to the new
  generic helpers, and no assertion is removed.
- `cargo test --locked --test ranking_pilot_archive < /dev/null`: 2 passed.
- `cargo clippy --locked --all-targets -- -D warnings` and `cargo fmt --check`:
  no findings.
- `markdownlint-cli2` and `scripts/spell-check.sh` on
  `docs/experiments/ranking-v1.md`, `docs/research/ranking-pilot-v1/README.md`
  and `docs/research/ranking-pilot-v2/README.md`: no findings.
- Docs read for text this run makes false: the functional-paths bullet and the
  pilot and "Current finding" sections of `docs/experiments/ranking-v1.md`, and
  the v1 README's maintenance decision. All three now name the v2 pilot. A grep
  of `README.md` and `docs/` (excluding `docs/archive/`) for "v2 pilot" and
  "ranking-pilot-v1" found nothing else stale.
- Not checked: CI, macOS, and a representative v2 run.
