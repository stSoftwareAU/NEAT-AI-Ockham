## Summary

The controlled proxy/history ranking benchmark merged in #267 and #274. This
PR adds a synthetic, non-representative pilot of that protocol, run with the
real scorer. It brings back the archived evidence and generator from the
closed PR #279, and fixes the blocking review finding that stalled #279.
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
  file-bounds review finding was never addressed. The revisions `a27d638` and
  `e02ef9f` named in the archive are reachable through
  `git fetch origin pull/279/head`.
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
  charged rather than counted as 0. The `cost-recount/` outputs come from that
  code.
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
- **partial** — Audit a randomly selected fully scored pool to measure true
  top-5% recall, false rejection and cold-start/negative-transfer behaviour —
  evidence: `ockham/examples/priority_ordering_bench/audit.rs`,
  `results.json` (16 complete pools, calibration-0 cold start,
  `negativeTransferTasks`) — reviewer: partial — reason: only synthetic,
  homogeneous pools were audited; no representative pool exists on this host.
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
- **partial** — Predeclare controls, seeds, sample size, limits and criteria;
  identify versions; separate calibration from validation; count all costs —
  evidence: `docs/research/ranking-pilot-v1/manifest.json`, `frozen.json` —
  reviewer: partial — reason: input generation and curation time (a few ms) is
  described in the README but not included in the archived `totalResearchMs`.
- **met** — Deterministic positive/negative tests; normal quality checks;
  research runs separate from CI — evidence:
  `ockham/examples/ranking_pilot.rs` tests,
  `ockham/tests/ranking_pilot_archive.rs` — reviewer: met
- **partial** — Deliver a reproducible command/manifest and machine-readable
  results with paired distributions, uncertainty, failures and actual
  coverage. Fixtures alone do not demonstrate an optimisation benefit —
  evidence: `results.json` (`paired`, `recallDeltaCi95`, `coverage[].failures`,
  `historyCoverage`), `docs/research/ranking-pilot-v1/README.md` — reviewer:
  partial — reason: no representative paired results exist; the archive was
  produced at `a27d638`/`e02ef9f` (reachable only via `pull/279/head`), HEAD's
  manifest step would not reproduce it byte for byte, the raw candidate logs
  are not committed, and the pilot was not re-run after the input bounds.
- **partial** — Record negative findings and remove optional code that does
  not earn its maintenance cost. Do not claim success before representative
  runs — evidence: README `Finding`, `docs/experiments/ranking-v1.md`
  `Synthetic pilot` — reviewer: partial — reason: the negative finding is
  recorded and no success is claimed, but the keep-or-remove decision on the
  pairwise and transfer arms is deferred; the synthetic result is too
  confounded (186 of 237 validation candidates are featureless synapse cuts,
  3 validation groups) to justify removing them.
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
- Restored from #279: the other tests in `ockham/examples/ranking_pilot.rs`,
  `ockham/tests/ranking_pilot_archive.rs` (archive digests, verdict and
  coverage), and
  `ockham/examples/priority_ordering_bench/tests.rs::cost_milliseconds_round_up_so_sub_millisecond_work_is_counted`.
  Restored tests changed only to pass `&Limits::DEFAULT` or the corpus limit;
  no assertion was removed.
- `telemetry::load` now delegates to `load_from_reader`. The existing
  telemetry tests cover its messages and pass unchanged.
- `cargo test --locked -p neat_ai_ockham --example ranking_pilot`: 18 passed
  on the final head. `./quality.sh < /dev/null` passed ("All quality checks
  passed!") on the final head.
- No existing test on `Develop` was edited, except that `audit.rs` and
  `tests.rs` gained the `ceil_ms` hunk and its test.

## Not performed and remaining risk

No representative pilot, frozen private manifest, validation scoring or
paired representative benefit and cost assessment, because no private
corpora, histories or representative parents exist on this host. The
synthetic pilot was not re-run after the input boundaries were added; for
valid inputs those boundaries do not change the output bytes, which the tests
exercise on generated inputs. The archive was produced at `a27d638` and
`e02ef9f`, not at this head, and its raw candidate logs are not committed, so
it cannot be regenerated byte for byte from this branch. Its manifests carry
absolute scratch paths from the generating host. A concurrent path swap is not guarded. macOS was
not checked. CI and PR review have not been observed.
