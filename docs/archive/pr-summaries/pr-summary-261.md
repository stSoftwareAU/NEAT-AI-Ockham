## Summary

The controlled proxy/history ranking benchmark merged in #267 and #274. This
PR adds a synthetic, non-representative pilot of that protocol, run with the
real scorer. It brings back the generator from the closed PR #279 and fixes
the blocking review finding that stalled #279. It also re-runs the pilot at
this branch's code (`fad4e74`), replacing #279's two archives, and commits the
raw search logs, so the curation can be re-run and checked byte for byte. Input
generation and the manifest step's own time are now charged in
`totalResearchMs`.
`ranking_pilot manifest` read its inputs with no byte limit and followed task
paths through symlinks. It now caps every read and digest, refuses symlinks
and paths that escape the pilot directory, creates its outputs exclusively,
and finishes every read before its first write. The synthetic finding is
negative: partial-score ordering alone beats the composite and pairwise arms,
and sparse-history transfer shows negative transfer. A representative run
still needs private corpora and histories this host does not have.
Part of #261.

## Spec

### Intent and Rationale

- The issue asks for a reproducible command and manifest, and for
  machine-readable paired results with uncertainty, failures and coverage. The
  synthetic pilot exercises that whole path end to end with the real scorer,
  and records its negative finding without claiming a benefit.
- The #279 reviewer found that `write_manifest` read raw logs, parents,
  corpora and the scorer with no limit. It also found that a dangling
  `history.jsonl` symlink let `telemetry::append` write outside the pilot
  directory. Bounding and confining those reads fixes that finding at its
  source and leaves the bench unchanged.

### Essential Design Decisions

- Limits live in one `Limits` struct, and `main` passes `Limits::DEFAULT`:
  parent 16 MiB, raw log 64 MiB (the bench's own history cap), `acquisition-ms`
  64 bytes, corpus 100,000,000 bytes and scorer 1 GiB. The manifest's
  `maxCorpusBytes` is written from the same `corpus_bytes`, so the cap the
  manifest step checked is the cap the bench enforces. Tests pass smaller
  limits through the same parameter.
- Pilot-owned paths are refused when they are a symlink or resolve outside the
  canonical pilot root. Outputs are refused when anything, including a dangling
  symlink, exists at the path, and are then created with `create_new`. The
  scorer is a separately chosen input: it may be a symlink and may live
  anywhere, but it must be a regular file, and its digest is streamed and
  capped.
- `telemetry::load_from_reader` holds the existing parse and version check, and
  `telemetry::load` calls it. The capped raw-log bytes are parsed in place
  rather than read again from disk.
- For valid inputs the output bytes are unchanged. History lines are
  serialised the way `telemetry::append` writes them, and the history digest is
  taken from the bytes written.

### Undiscoverable Facts

- PR #279 was closed by the worker's stall handling. Its CI was green, but the
  file-bounds review finding was never addressed. Its archives, made at
  `a27d638` and `e02ef9f`, are not on this branch and are replaced here. The
  run at `fad4e74` reproduced their paired results, coverage and verdict; only
  the timings differ.
- The scorer was rebuilt at the pinned revision on 18 November 2026. Its
  executable digest matched the archived pin, `d057a334…`.
- This host still has no representative NEAT-AI corpus, candidate history or
  parent (`docs/experiments/ranking-v1-availability.json`). The pilot's corpora
  are generated and homogeneous.
- Containment is checked by path and then the file is opened again, with no
  no-follow open. A concurrent swap of a path inside the pilot directory is not
  covered. This is accepted for a local, operator-run research example.

## Evidence

- Backend and research example only; no UI file changed.
- `docs/research/ranking-pilot-v1/README.md` is the pilot record: design,
  versions and pins, results, costs, coverage and failures, the cost recount,
  checksums and reproduction steps. Its new `Manifest step input boundaries`
  subsection documents the limits.
- `docs/experiments/ranking-v1.md` gains the `Synthetic pilot` section, and
  `Current finding` now says the synthetic pilot ran but is not
  representative.
- `ockham/examples/priority_ordering_bench/audit.rs`: `audit::ms` now rounds
  up (`ceil_ms`), so feature, fit, adaptation and ranking work under 1 ms is
  charged rather than counted as 0.
- The bench manifest gains an optional `preparationMs` (default 0, not
  serialised when 0, so older manifests keep their identity). The report gains
  `preparationCostMs`, which is added to `totalResearchMs`.
  `ranking_pilot inputs` writes `inputs-ms`. `ranking_pilot manifest` requires
  that file and writes `preparationMs` as its value plus the manifest step's own
  time, excluding curation, which `acquisitionMs` already charges. In the run
  this is 39 ms of 11,726 ms.
- `docs/research/ranking-pilot-v1/` holds one archive from `fad4e74`: the
  manifest, preflight, pilot, freeze, gzipped audit and results, plus
  `inputs-ms`, `source-N/acquisition-ms` and the gzipped raw logs
  `source-N/raw-candidates.jsonl.gz`. The old archive and `cost-recount/` were
  removed, which is about 15,700 lines of superseded JSON. The README records a
  maintenance decision with a predeclared removal rule for the pairwise and
  transfer arms.
- #261: Experiment: Compare cheap proxy ranking with transfer from sparse search histories.
- #267 and #274: the merged PRs that delivered the benchmark.
- #279: the closed PR whose pilot this restores.

```mermaid
flowchart LR
    I[ranking_pilot inputs] --> S[ordinary searches with --candidate-log]
    S --> M[ranking_pilot manifest]
    M -->|caps, symlink and root checks, all reads first| W[history.jsonl and manifest.json, create-new]
    W --> B[priority_ordering_bench: check, pilot, freeze, audit, evaluate]
    B --> R[results.json: no-demonstrated-benefit]
```

## Acceptance Criteria

<!-- vibe-spec-review inputs="diff+issue-body" -->

- **met** — Extend or benchmark existing features, model and telemetry rather
  than create another ranker; compare partial score alone, multiple cheap
  proxies and a pairwise ranker adapted from history — evidence:
  `docs/research/ranking-pilot-v1/results.json` (five arms) — reviewer: met
- **met** — Candidate features only where inexpensive and available —
  evidence: `results.json` `featureSchema` (17 existing features; no latency
  or Fisher signal) — reviewer: met
- **met** — Separate parent lineages and corpus versions between fitting and
  testing — evidence: `docs/research/ranking-pilot-v1/manifest.json` (disjoint
  lineage and version per task) — reviewer: met
- **met** — Audit a randomly selected fully scored pool to measure true
  top-5% recall, false rejection and cold-start/negative-transfer behaviour —
  evidence: `docs/research/ranking-pilot-v1/results.json` (16 seeded random
  pools, 640 selected, 632 generated and fully scored, all complete; recall and
  false rejection in all 360 trials; `negativeTransferTasks` validation-2 in the
  primary cell and validation-1/-2 at calibration 0),
  `ockham/tests/ranking_pilot_archive.rs` — reviewer: met
- **met** — Count feature and training overhead before permitting rejection —
  evidence:
  `ockham/examples/priority_ordering_bench/tests.rs::cost_milliseconds_round_up_so_sub_millisecond_work_is_counted`
  — reviewer: met
- **met** — Inspect current code and reuse existing mechanisms — evidence:
  `ockham/examples/ranking_pilot.rs` reuses `telemetry`, `corpus_info` and the
  #267/#274 bench — reviewer: met
- **met** — Opt-in only; preserve ordinary behaviour, interfaces, immutable
  parents and full acceptance — evidence: `ockham/src/telemetry.rs` (additive
  `load_from_reader`) — reviewer: met
- **met** — Predeclare controls, seeds, sample size, limits and criteria;
  identify versions; separate calibration from validation; count all costs —
  evidence: `docs/research/ranking-pilot-v1/manifest.json` (control, seeds,
  visits, compute limits, thresholds, code `fad4e74`, scorer and data digests),
  `results.json` `preparationCostMs` 39 inside `totalResearchMs` 11,726,
  `priority_ordering_bench/tests.rs::manifest_preparation_time_is_reported_and_added_to_the_research_total`,
  `ranking_pilot.rs::pilot_without_inputs_ms_is_refused_naming_it` —
  reviewer: met. The reviewer noted that the bench's preflight `check` stage is
  not timed. The docs now say so instead of claiming that every stage is
  counted. That cost is of the order of the 47 ms the pilot stage charges for
  the same checks.
- **met** — Deterministic positive/negative tests; normal quality checks;
  research runs separate from CI — evidence:
  `ockham/examples/ranking_pilot.rs` tests,
  `ockham/tests/ranking_pilot_archive.rs` — reviewer: met
- **met** — Deliver a reproducible command/manifest and machine-readable
  results with paired distributions, uncertainty, failures and actual
  coverage. Fixtures alone do not demonstrate an optimisation benefit —
  evidence: archive re-run at `fad4e74` with committed raw logs,
  `ranking_pilot.rs::committed_raw_logs_recurate_to_the_committed_histories`,
  `results.json` (`trials`, `paired[].taskDeltas`, `recallDeltaCi95`,
  `coverage[].failures`, `historyCoverage`), README `Reproduce`, verdict
  `no-demonstrated-benefit` — reviewer: met
- **partial** — Record negative findings and remove optional code that does
  not earn its maintenance cost. Do not claim success before representative
  runs — evidence: README `Finding` and `Maintenance decision`,
  `docs/experiments/ranking-v1.md` `Synthetic pilot`, the removed #279 archive
  and `cost-recount/` — reviewer: partial — reason: the pairwise and transfer
  arms did not earn their cost on the synthetic evidence but are kept. Their
  removal waits on a predeclared rule tied to the representative run, which has
  not happened.
- **met** — Keep public implementation and fixtures application-agnostic —
  evidence: `ockham/examples/ranking_pilot.rs` (generic synthetic creatures and
  corpora) — reviewer: met
- **unrequested** — Input bounds, symlink and containment checks in
  `ranking_pilot manifest` (`Limits`, `contained`, `read_bounded`,
  `refuse_existing`) and their 11 tests — reviewer: unrequested — reason:
  fixes the blocking #279 review finding on the pilot generator.
- **unrequested** — Public `telemetry::load_from_reader` — reviewer:
  unrequested — reason: lets the manifest step parse the capped raw-log bytes
  in place; `telemetry::load` delegates to it.

## Standards Review

<!-- vibe-standards-review inputs="diff+CODING-STANDARDS.md" -->

- **clean** — There is no `CODING-STANDARDS.md`, so the reviewer used
  `CONTRIBUTING.md`, which names no rule as enforced by review only. It checked
  containment and size limits in `write_manifest`, which `corpus_info` reads,
  the scorer exception, exclusive outputs, and agreement between the corpus
  digest and the bench. It also checked doc–code agreement, that every named
  test exists, that no stub stands in for another repository's binary, and
  that principles 1–7 hold (parent immutable, acceptance path untouched). Its
  optional notes on a symlinked task directory, a dangling `manifest.json`
  and the dangling-history positive control are covered by the tests added in
  this diff.

## Test Plan

- Added to `ockham/examples/ranking_pilot.rs`: tests that go through
  `write_manifest`. Each refusal test starts from a valid fixture, asserts the
  specific cause, asserts that no history and no manifest were written, and
  then has a positive control:
  - `oversized_raw_log_is_refused_at_the_boundary`
  - `oversized_corpus_is_refused_at_the_boundary`, which also asserts that
    `maxCorpusBytes` equals the limit
  - `oversized_parent_is_refused`
  - `oversized_acquisition_ms_is_refused`
  - `oversized_scorer_is_refused`
  - `symlinked_raw_log_is_refused`
  - `symlinked_corpus_bin_is_refused`
  - `symlinked_task_directory_is_refused`
  - `dangling_history_symlink_is_refused_and_its_target_not_created`
  - `dangling_manifest_symlink_is_refused_and_its_target_not_created`
  - `scorer_may_be_a_symlink`
- Added for preparation cost and reproducibility:
  - `ranking_pilot.rs`: `inputs_record_their_generation_time`,
    `manifest_preparation_includes_the_recorded_inputs_time`,
    `pilot_without_inputs_ms_is_refused_naming_it`,
    `symlinked_inputs_ms_is_refused`, and
    `committed_raw_logs_recurate_to_the_committed_histories`. The last one
    re-runs the manifest step on the committed raw logs and checks that the
    histories are byte-identical. Its negative control cuts the log to a
    single unlabelled row, which is refused.
  - `priority_ordering_bench/tests.rs::manifest_preparation_time_is_reported_and_added_to_the_research_total`,
    plus `preparationCostMs` in the existing audit-time sum test.
  - `ockham/tests/ranking_pilot_archive.rs` now also checks the research-cost
    sum, `preparationCostMs`, the acquisition sum and the coverage totals.
- Restored from #279: the other tests in `ockham/examples/ranking_pilot.rs`,
  `ockham/tests/ranking_pilot_archive.rs` (archive digests, verdict and
  coverage), and
  `ockham/examples/priority_ordering_bench/tests.rs::cost_milliseconds_round_up_so_sub_millisecond_work_is_counted`.
  Restored tests changed only to pass `&Limits::DEFAULT` or the corpus limit;
  no assertion was removed.
- `telemetry::load` now delegates to `load_from_reader`. The existing
  telemetry tests cover its messages and pass unchanged.
- With stdin from `/dev/null`: `--example ranking_pilot` 23 passed,
  `--example priority_ordering_bench` 18 passed, `--test ranking_pilot_archive`
  1 passed. Clippy with `-D warnings` and `cargo fmt --check` are clean.
  `./quality.sh < /dev/null` passed (exit 0) on `4108066`, the last code
  commit; only this summary line changed after it.
- No existing test on `Develop` was edited, except in `tests.rs`: the
  `ceil_ms` test, and `preparationCostMs` added to the audit-time sum test.

## Not performed and remaining risk

No representative pilot, frozen private manifest, validation scoring or
paired representative benefit and cost assessment, because no private
corpora, histories or representative parents exist on this host. All evidence
here is synthetic and does not demonstrate an optimisation benefit. The
keep-or-remove decision on the pairwise and transfer arms waits on that run,
under the rule predeclared in the README. The bench's preflight `check` stage
is not timed. The run commit `fad4e74` is on this branch, but a squash merge
would leave it reachable only through the PR head. The manifests carry absolute
scratch paths from the generating host. A concurrent path swap is not guarded.
macOS was not checked. CI and PR review have not been observed.
