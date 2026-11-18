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
This replaces the stalled PR #284 and answers its review:
`scorer_may_be_a_symlink` now asserts the recorded scorer digest and path,
the preparation-time test can no longer pass with the inputs term dropped,
and "count all preparation/search costs" is recorded as `partial`, naming the
stages still uncounted.
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
  `source-N/raw-candidates.jsonl.gz`. #279's two archives and its
  `cost-recount/` are not carried over; they were never on `Develop`. The README records a
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

- **met** — Extend or benchmark existing features/model/telemetry rather than
  create another ranker — evidence: `ockham/examples/ranking_pilot.rs` only
  generates inputs and a manifest for the existing `priority_ordering_bench`,
  reusing `telemetry`, `corpus_info` and `pairwise::FitConfig` — reviewer: met
- **partial** — Compare partial-data score alone, multiple cheap proxies, and
  a pairwise ranker adapted from historical corpus/task versions — evidence:
  `docs/research/ranking-pilot-v1/results.json` (five arms) — reviewer:
  partial — reason: all arms run, but 186 of 237 synthetic validation
  candidates carry no feature vector, so the proxy and pairwise arms are
  compared mostly on tied candidates; a representative run is still needed.
- **partial** — Candidate features: sample error, output variance, functional
  paths, connections, compiled latency and gradient/Fisher signal only if
  inexpensive/available — evidence: `results.json` `featureSchema` (17
  existing features) — reviewer: partial — reason: output variance,
  functional paths and compiled latency are not added, and no cost case for
  them is measured; `docs/experiments/ranking-v1.md` records why.
- **met** — Separate parent lineages and corpus versions between fitting and
  testing — evidence: `docs/research/ranking-pilot-v1/manifest.json`
  (disjoint lineage and version per task) — reviewer: met
- **met** — Audit a randomly selected fully scored pool to measure true top-5%
  recall, false rejection and cold-start/negative-transfer behaviour —
  evidence: `results.json` (640 selected, 632 generated and fully scored, all
  pools complete; `negativeTransferTasks` non-empty; calibration budget 0 is
  the cold start), `ockham/tests/ranking_pilot_archive.rs` — reviewer: met
- **met** — Count feature and training overhead before permitting rejection —
  evidence:
  `ockham/examples/priority_ordering_bench/tests.rs::cost_milliseconds_round_up_so_sub_millisecond_work_is_counted`
  — reviewer: met
- **met** — First inspect current code, merged work and repository
  instructions. Reuse existing mechanisms — evidence: `ranking_pilot.rs`
  reuses the #267/#274 bench; `telemetry::load` shares `load_from_reader` —
  reviewer: met
- **met** — Opt-in experiment only; preserve ordinary behaviour, interfaces,
  immutable parents and full authoritative acceptance. No automatic deployment
  or default change — evidence: changes confined to `ockham/examples/`, docs
  and tests, plus the additive `telemetry::load_from_reader` — reviewer: met
- **met** — Predeclare controls, seeds, sample size, compute limits and
  success/failure criteria — evidence: `manifest.json` (control, seeds 7 and
  19, visits, `maxTaskMs`, `maxRunMs`, thresholds) — reviewer: met
- **met** — Identify code/model/data versions — evidence: `manifest.json`
  `codeRevision` `fad4e74`, `scorerSha256`, parent, corpus and history
  digests;
  `ranking_pilot.rs::scorer_may_be_a_symlink` asserts the recorded scorer
  digest and path — reviewer: met
- **met** — separate calibration from untouched validation — evidence:
  disjoint calibration and validation tasks in `manifest.json`; pilot and
  freeze open calibration only — reviewer: met
- **partial** — count all preparation/search costs — evidence:
  `results.json` `preparationCostMs` 39 inside `totalResearchMs` 11,726,
  `priority_ordering_bench/tests.rs::manifest_preparation_time_is_reported_and_added_to_the_research_total`,
  `ranking_pilot.rs::manifest_preparation_includes_the_recorded_inputs_time`
  — reviewer: partial — reason: still uncounted, as the README and
  `docs/experiments/ranking-v1.md` say: the bench's preflight `check` stage,
  building the binaries, writing the final results, the manifest step after
  its first write, and source search time, which the operator times by hand.
- **met** — Provide deterministic positive/negative correctness tests for new
  behaviour and pass the repository's normal quality checks. Research runs are
  separate from routine feature CI — evidence: `ranking_pilot.rs` refusal
  tests with positive controls, `ockham/tests/ranking_pilot_archive.rs`,
  `./quality.sh` exit 0 on `cd5d26c` — reviewer: met
- **partial** — Deliver a reproducible command/manifest and machine-readable
  results with paired distributions, uncertainty, failures and actual coverage
  — evidence: README `Reproduce`, `results.json` (`paired`,
  `recallDeltaCi95`, `coverage[].failures`, `historyCoverage`),
  `ranking_pilot.rs::committed_raw_logs_recurate_to_the_committed_histories`
  — reviewer: partial — reason: only the curation step reproduces byte for
  byte; the committed manifest names absolute scratch paths, so a re-run from
  scratch gives new digests and only the verdict and signs compare.
- **met** — Fixtures alone do not demonstrate an optimisation benefit —
  evidence: README and `ranking-v1.md` say so; verdict
  `no-demonstrated-benefit` — reviewer: met
- **partial** — Record negative findings and remove optional code that does
  not earn its maintenance cost — evidence: README `Finding` and `Maintenance
  decision`, `ranking-v1.md` `Synthetic pilot` — reviewer: partial — reason:
  the negative finding is recorded, but nothing is removed; the pairwise and
  transfer arms are kept until the representative run, under the README's
  predeclared removal rule.
- **met** — Do not claim success before representative runs — evidence:
  README states no success is claimed — reviewer: met
- **met** — Keep public implementation and fixtures application-agnostic;
  private corpora/results and operational integration remain private —
  evidence: generic synthetic creatures and corpora in `ranking_pilot.rs` —
  reviewer: met. The reviewer noted that the archive records the generating
  host name and scratch paths; these name no private data.
- **met** — Leave unassigned until an implementer actively starts. `work-on`
  authorises the investigation, not skipping prerequisite verification —
  evidence: README records the scorer rebuild at the pinned revision and its
  digest match — reviewer: missing — reason: the reviewer could not judge
  assignment from a diff; assignment is the worker's lifecycle step, and the
  prerequisite verification is recorded in the README.
- **unrequested** — Input bounds, symlink and containment checks in
  `ranking_pilot manifest` (`Limits`, `contained`, `read_bounded`,
  `hash_capped`, `refuse_existing`) and their tests — reviewer: unrequested —
  reason: fixes the blocking #279 review finding on the pilot generator.
- **unrequested** — Public `telemetry::load_from_reader` — reviewer:
  unrequested — reason: lets the manifest step parse the capped raw-log bytes
  in place; `telemetry::load` delegates to it.
- **unrequested** — Crate version 0.1.103 to 0.1.104 in `ockham/Cargo.toml`
  and `Cargo.lock` — reviewer: unrequested — reason: the worker's automatic
  version commit for the library change in `telemetry.rs`.
- **unrequested** — `"preparationMs": 0` in
  `docs/experiments/ranking-v1.template.json` and `audit::ms` rounding up for
  every bench timing — reviewer: unrequested — reason: both serve "count all
  preparation/search costs"; the field is optional and omitted at 0, so older
  manifests keep their identity.
- **unrequested** — Rewrite of `docs/archive/pr-summaries/pr-summary-261.md` —
  reviewer: unrequested — reason: this PR's required summary replaces the
  earlier PR's copy.

## Standards Review

<!-- vibe-standards-review inputs="diff+CODING-STANDARDS.md" -->

- **clean** — There is no `CODING-STANDARDS.md`, so the reviewer used
  `CONTRIBUTING.md`, which names no rule as enforced by review only. No
  violations. It checked principles 1–8 (parent never written, acceptance
  path untouched, optional and isolated, version bump), that no new test
  asserts only success or presence (`scorer_may_be_a_symlink` now asserts the
  digest and path), that the one edited existing test
  (`comparison_reports_audit_time_outside_every_pool_ledger`) still guards its
  sum, that every named test exists, that no stub stands in for another
  repository's binary, and doc–code agreement on limits, refusals and the
  `preparationMs` formula. Optional notes not chased: a FIFO at a pilot path
  blocks rather than being refused, unchecked `+` on an operator-supplied
  `preparationMs`, and tests that need `gzip` on `PATH`.

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
  - `scorer_may_be_a_symlink`, which also asserts that `scorerSha256` is the
    digest of the link target's bytes and that `scorer` is the absolute link
    path, not the target (added after the PR #284 review)
- Added for preparation cost and reproducibility:
  - `ranking_pilot.rs`: `inputs_record_their_generation_time`,
    `manifest_preparation_includes_the_recorded_inputs_time`,
    `pilot_without_inputs_ms_is_refused_naming_it`,
    `symlinked_inputs_ms_is_refused`, and
    `committed_raw_logs_recurate_to_the_committed_histories`. The preparation
    test records 1,000,000,000 ms in `inputs-ms` and bounds `preparationMs`
    above it and below it plus 600,000 ms, so a dropped inputs term fails
    whatever the step's own time (tightened after the PR #284 review). The last one
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
  `./quality.sh < /dev/null` passed (exit 0) on `cd5d26c`, the last code
  commit; only this summary changed after it.
- The only existing test on `Develop` that the diff edits is
  `priority_ordering_bench/tests.rs::comparison_reports_audit_time_outside_every_pool_ledger`:
  it gains `preparationCostMs == 0` and that term in its cost sum. No
  assertion is removed. The `ceil_ms` test is new.

## Not performed and remaining risk

No representative pilot, frozen private manifest, validation scoring or
paired representative benefit and cost assessment, because no private
corpora, histories or representative parents exist on this host. All evidence
here is synthetic and does not demonstrate an optimisation benefit. The
keep-or-remove decision on the pairwise and transfer arms waits on that run,
under the rule predeclared in the README. Uncounted costs remain: the bench's
preflight `check` stage, building the binaries, writing the final results,
the manifest step after its first write, and hand-timed source searches.
Output variance, functional paths and compiled latency were not added as
features. The run commit `fad4e74` is on this branch, but a squash merge
would leave it reachable only through the PR head. The manifests carry absolute
scratch paths from the generating host. A concurrent path swap is not guarded.
macOS was not checked. CI and PR review have not been observed.
