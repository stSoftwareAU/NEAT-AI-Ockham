## Summary

Added an opt-in controlled proxy/history comparison to the existing Rust benchmark,
with random full-score audits, isolated pairwise fits and paired metrics. Ordinary
ordering and acceptance remain unchanged. Representative research is outstanding.
Part of #261.

## Spec

### Intent and Rationale

- Compare ranking quality and complete acquisition/search costs without promoting
  an unverified heuristic into pruning acceptance.
- Reuse the existing feature extraction, composite weights, model numerics,
  telemetry reader, pruning sweep and benchmark entry point.

### Essential Design Decisions

- Research commands never apply candidates. Sampled losers receive individual
  full scores; generation failures are retained without replacement.
- Source logs precede the declared cutoff. Lineages, corpus/task versions and
  content identities cannot cross partitions; uncertainty groups also connect
  related target families. Undisclosed ancestry cannot be inferred from JSON.
- The fixed composite control and linear pairwise adaptation are deliberately
  smaller adaptations of the papers, not CAST or SHINE reproductions. Existing
  neuron features have explicit availability flags for synapse candidates.
- Private manifests/results use separate versioned formats. Pilot, manifest and
  executable digests bind the freeze. Missing labels stay unlabelled; incomplete
  scoring, unknown historical acquisition costs and insufficient groups prevent
  a positive research verdict.

### Undiscoverable Facts

- Neither representative inputs nor a real scorer were supplied or found in the
  standard local locations inspected. No research benefit has been measured.
- Both paper method sections were retrieved on 13 November 2026. Their
  applicability and deviations are recorded in `docs/experiments/ranking-v1.md`.

## Evidence

- #261: Experiment: Compare cheap proxy ranking with transfer from sparse search histories.
- Merged PR #118, commit `63e1b8c734b7accc03d304a2ef955d38e5f179e7`, supplied the
  existing features/model/telemetry and simulated benchmark reused here.
- Protocol, illustrative manifest and machine-readable availability finding:
  `docs/experiments/ranking-v1.md`, `docs/experiments/ranking-v1.template.json`
  and `docs/experiments/ranking-v1-availability.json`. The template is not a
  frozen representative manifest; availability metrics are explicitly null.
- CLI help exited 0. A schema-valid template with the committed code identity
  exited 1 on preflight, wrote `inconclusive`, and reported 15 missing-resource
  failures, starting with the absent scorer. Scratch outputs were removed.
- Backend/CLI only. The bounded-scorer fixture checks the existing adapter's
  stem-keyed JSON, sample flags, record counts and non-zero exits, matching the
  [scorer's documented contract](https://github.com/stSoftwareAU/NEAT-AI-scorer/blob/Develop/README.md).
  A real scorer execution was not performed.

```mermaid
flowchart LR
    M[Declared private manifest] --> P[Calibration-only pilot]
    P --> F[Digest-bound freeze]
    F --> A[Random audit visits]
    A --> S[Existing pruning and individual full scoring]
    S --> E[Isolated fits and paired evaluation]
    E --> R[Private JSON metrics and explicit omissions]
```

## Acceptance Criteria

- **missing** — Predeclare controls, seeds, sample size, compute limits and success/failure criteria. Identify code/model/data versions; separate calibration from untouched validation and count all preparation/search costs. — reason: the template and enforcement machinery are delivered, but no representative calibration pilot or final private frozen manifest exists.
- **missing** — Audit a randomly selected fully scored pool to measure true top-5% recall, false rejection and cold-start/negative-transfer behaviour. — reason: audit correctness is tested with deterministic fixtures; representative parents, corpora, independently identified histories and the scorer are unavailable.
- **missing** — Deliver a reproducible command/manifest and machine-readable results with paired distributions, uncertainty, failures and actual coverage. Fixtures alone do not demonstrate an optimisation benefit. — reason: commands, a manifest template, JSON reporting and an inconclusive availability record are delivered; paired representative results remain unmeasured.

## Test Plan

- `./quality.sh < /dev/null` passed on the final implementation: shell contracts,
  spelling, Markdown, Actions lint, dependency audit, formatting, all-target
  Clippy, workspace tests and rustdoc. Earlier Clippy and README-contract
  findings were fixed; no check remains failing.
- `ockham/examples/priority_ordering_bench/tests.rs`: ten passing contracts cover
  split leakage, random selection, ties/small pools, unknown/non-finite inputs,
  sampled losers, missing/incomplete/failed scoring, deterministic budget expiry,
  immutable parents, historical cutoffs, freeze integrity, cold starts, negative
  transfer, validation-label isolation, reproducibility, cost charges and grouped
  confidence intervals. These run in the normal gate through the example target.
- `ockham/src/model/pairwise.rs`: three passing tests cover pairwise ordering,
  adaptation reversal, empty/tied supervision, invalid inputs, reproducibility
  and invariance to positive label rescaling. The existing pointwise model tests
  also passed; initial new API tests failed to compile before implementation.
- `ockham/tests/bounded_scorer.rs`: the timeout adapter's argument order, full and
  sampled results, invalid bounds and scorer failures passed. Existing
  `ockham/tests/cli.rs`, `ockham/tests/readme_contract.rs` and pruning/acceptance
  regressions passed in the full gate. No existing test assertions were removed.
- Ordinary scorer callers in baseline, promote, reentry, run, screening and sweep
  retain their interface and share `score_command` with the bounded adapter.
  No twinned experiment rule was found. Documentation sweep covered the README
  ranking/benchmark section, CONTRIBUTING prerequisites and model/scorer docs.
- Security: explicit schemas/digests and finite inputs checked; argument-vector
  process execution; exclusive output creation; no new network endpoint,
  dependency, privileged operation or staged secret. No path-confinement claim
  or destructive state replacement is introduced.

## Merge verification

CI and independent code review are not yet verified.

## Not performed and remaining risk

Representative pilot, frozen private data/model identities and validation runs,
real-scorer integration, and macOS execution were not performed. Group provenance
and measured historical acquisition costs still require real input records.
The finding is inconclusive; no optimisation benefit or deployment is claimed.
