## Summary

Added the opt-in controlled proxy/history benchmark, retaining ordinary ordering
and authoritative acceptance. Fixed its audit-to-history corpus identity mismatch
and verified bounded scoring against a real scorer in earlier branch work.
The completion retry fixes understated strategy costs: recorded incomplete
calibration pools retain their acquisition cost, and validation preparation
includes residual measured pool work. The review round fixes the clean-source
guard, which now holds when the command runs from `ockham/`, and adds tests for
the verdict decision rule. Representative comparison results remain unavailable.
Part of #261.

## Spec

### Intent and Rationale

- Reuse the existing feature, composite, model, telemetry and pruning mechanisms
  to compare random, partial, composite, target-only and transfer ordering.
- Separate implementation correctness from evidence of a representative benefit.

### Essential Design Decisions

- Research runs never apply cuts; audit sampled losers individually and keep
  source, calibration and validation partitions separate.
- Earlier v1 audit telemetry remains readable: normalise its digest-valued corpus
  identity only for `random-audit` rows matching the manifest's corpus digest.
  New audit telemetry writes the ordinary corpus cache identity. Other provenance
  and cutoff checks remain in place; no persisted format version changes.
- Missing representative data leaves metrics unknown and the verdict inconclusive.
  Absent measurements do not justify removing the optional ranker.
- The clean-source guard uses repository-root pathspecs (`:/ockham`,
  `:/Cargo.toml`, `:/Cargo.lock`, `:/rust-toolchain.toml`) and `git -C`, so
  uncommitted edits are detected from any directory of the checkout.
- Charge recorded calibration work even when its incomplete labels cannot train
  a model. Retain residual pool wall time as preparation; never charge less than
  the sum of individually timed stages.
- The linear adaptation and fixed composite are smaller components inspired by
  the papers, not reproductions of CAST or SHINE.

### Undiscoverable Facts

- The mounted checkout search found no corpus `.bin` files, candidate histories
  or experiment journals outside hidden directories and build/dependency caches.
  No representative manifest was supplied; availability on other hosts is unknown.
  The completion retry repeated the search across `/home/vibe/auto-issue-work`,
  excluding hidden directories, `target`, `node_modules`, `vendor` and
  `lost+found`: all three input counts were zero, with no search errors.
- The scorer built from public source revision
  `742c5df30fc98ece3afe784be1aca18f925b47b3` and ran successfully on fixtures.
  This was a development build on a shared aarch64 Linux host.
- Both paper identities and method sections were fetched again on 13 November
  2026; their adaptations and limits are documented in the protocol.

## Evidence

- #261: Experiment: Compare cheap proxy ranking with transfer from sparse search histories.
- Merged PR #118, commit `63e1b8c734b7accc03d304a2ef955d38e5f179e7`,
  supplies the existing features/model/telemetry reused by the benchmark.
- `docs/experiments/ranking-v1.md` describes the methods, partitions, cost
  accounting and commands. `docs/experiments/ranking-v1.template.json` remains
  illustrative, not a frozen representative manifest.
- `docs/experiments/ranking-v1-availability.json` records zero representative
  trials and null metrics, alongside fresh integration results, code/scorer/
  lockfile/executable identities and reproducibility commands from earlier branch
  verification. Its separate `completionRetryVerification` records this retry's
  input search, cost regression and fresh quality gate; it is fixture evidence.
- Preflight with the real scorer and illustrative template paths exited 1 with
  14 unavailable parent/corpus tasks. It did not run a calibration pilot.
- Backend/CLI only. Real scorer full and sampled results agreed through ordinary
  and bounded adapters: 32 full records and one sampled record at rate 0.05,
  phase 7. Fixture correctness does not establish a ranking benefit.

```mermaid
flowchart LR
    M[Private declared manifest] --> P[Calibration-only pilot]
    P --> F[Digest-bound freeze]
    F --> A[Random fully scored audit]
    A --> E[Isolated fits and paired evaluation]
    E --> R[Private results and explicit omissions]
```

## Acceptance Criteria

<!-- vibe-spec-review inputs="diff+issue-body" -->

- **met** — First inspect current code, merged work and repository instructions. Reuse existing mechanisms; if functionality already exists, this issue still owns the missing controlled comparison rather than a second implementation. — evidence: `docs/experiments/ranking-v1.md`, `ockham/examples/priority_ordering_bench/audit.rs` reuses features, Sweep and telemetry; `ockham/src/model/pairwise.rs` reuses numerical helpers — reviewer: met
- **met** — Opt-in experiment only; preserve ordinary behaviour, interfaces, immutable parents and full authoritative acceptance. No automatic deployment or default change. — evidence: `ockham/examples/priority_ordering_bench.rs` experiment dispatch; `ockham/examples/priority_ordering_bench/tests.rs::audit_scores_sampled_losers_individually_and_preserves_parent` — reviewer: met
- **partial** — Predeclare controls, seeds, sample size, compute limits and success/failure criteria. Identify code/model/data versions; separate calibration from untouched validation and count all preparation/search costs. — evidence: `ockham/examples/priority_ordering_bench/manifest.rs`, `ockham/examples/priority_ordering_bench/experiment.rs::freeze`, `ockham/examples/priority_ordering_bench/tests.rs::comparison_charges_incomplete_calibration_and_unitemised_preparation` — reviewer: partial — reason: representative provenance, pilot and frozen manifest remain unavailable; build/final output costs are excluded and some shared preparation is only in actual research totals.
- **partial** — Audit a randomly selected fully scored pool to measure true top-5% recall, false rejection and cold-start/negative-transfer behaviour. — evidence: `ockham/examples/priority_ordering_bench/audit.rs`, `ockham/examples/priority_ordering_bench/tests.rs::comparison_reports_cold_start_negative_transfer_and_validation_is_not_fitted` — reviewer: partial — reason: deterministic audit and metric contracts pass, but no representative parent/corpus/history inputs or fully scored representative pools are available.
- **met** — Provide deterministic positive/negative correctness tests for new behaviour and pass the repository's normal quality checks. Research runs are separate from routine feature CI. — evidence: `ockham/examples/priority_ordering_bench/tests.rs::verdict_requires_recall_false_rejection_and_cost_thresholds`, `ockham/examples/priority_ordering_bench/tests.rs::clean_source_guard_is_independent_of_the_working_directory`, `ockham/examples/priority_ordering_bench/tests.rs::comparison_charges_incomplete_calibration_and_unitemised_preparation`, 15 experiment tests and fresh `./quality.sh < /dev/null` — reviewer: met
- **partial** — Deliver a reproducible command/manifest and machine-readable results with paired distributions, uncertainty, failures and actual coverage. Fixtures alone do not demonstrate an optimisation benefit. — evidence: `docs/experiments/ranking-v1.md`, `docs/experiments/ranking-v1.template.json`, `ockham/examples/priority_ordering_bench/evaluation.rs`, `docs/experiments/ranking-v1-availability.json` — reviewer: partial — reason: commands and result machinery exist, but the manifest remains illustrative and representative paired distributions, uncertainty, coverage and benefit/cost results remain unmeasured.
- **partial** — Record negative findings and remove optional code that does not earn its maintenance cost. Do not claim success before representative runs. — evidence: `docs/experiments/ranking-v1-availability.json`, protocol current finding — reviewer: partial — reason: input unavailability is recorded and no success claimed; representative maintenance-value assessment remains unmeasured.
- **met** — Keep public implementation and fixtures application-agnostic; private corpora/results and operational integration remain private. — evidence: illustrative `docs/experiments/ranking-v1.template.json` and application-agnostic fixture tests; no private research artifacts in diff — reviewer: met
- **met** — Leave unassigned until an implementer actively starts. `work-on` authorises the investigation, not skipping prerequisite verification. — evidence: active issue investigation, merged-work and method verification recorded in protocol — reviewer: met

The Spec reviewer reviewed the issue body and base-branch diff, then re-reviewed
the new accounting diff and tests. The three retry criteria remain partial;
none is relabelled met. The later automated review found that the verdict
rule and the clean-source guard had no tests; both now do, as described below.

## Standards Review

<!-- vibe-standards-review inputs="diff+CODING-STANDARDS.md" -->

- **clean** — Standards reviewer checked the branch and retry diff against `CONTRIBUTING.md` and local Rust conventions: immutable parents, authoritative acceptance, opt-in isolation, reused mechanisms, existing version bump, cost-label separation, regression coverage and accurate protocol limits. No new dependency or interface; `git diff --check` passed.

No `CODING-STANDARDS.md` exists in this repository. The reviewer explicitly
recorded that absence and reviewed the available contributing principles;
this does not claim review against a nonexistent document.

## Test Plan

- Fresh completion-retry `./quality.sh < /dev/null` passed:
  shell contracts, spelling, Markdown, Actions lint, dependency audit, formatting,
  all-target Clippy, workspace tests (707 library and 15 experiment tests) and
  rustdoc. This was rerun after the review fixes below. The real scorer is absent on this host, so its optional integration
  test returned early. Earlier real-scorer evidence is retained separately.
- `cargo test --locked --example priority_ordering_bench < /dev/null`: 15 passed.
  The new `comparison_charges_incomplete_calibration_and_unitemised_preparation`
  first failed with zero charged calibration milliseconds versus 6,000 expected,
  then passed after the fix. It verifies incomplete labels never enter fitting,
  residual validation preparation is charged, and the verdict stays inconclusive.
  `cargo clippy --locked --example priority_ordering_bench -- -D warnings < /dev/null`
  also passed.
- Review fix, verdict rule: `verdict_requires_recall_false_rejection_and_cost_thresholds`
  drives `evaluation::evaluate` with three independent validation groups, each
  with a same-family calibration task, one source history and non-zero
  acquisition cost. Its fixture hides the signal from the partial score and
  calibration labels, so only transfer ranks well. It asserts
  `supported-for-further-research` when all gates pass, and
  `no-demonstrated-benefit` with the exact `predeclared thresholds` reason when
  only the recall, only the false-rejection, or only the cost threshold fails.
  `verdict_is_inconclusive_when_fewer_than_three_groups_validate` keeps the
  inconclusive path. Mutation checks confirmed that inverting the recall
  comparison, dropping the cost term, and loosening the false-rejection or cost
  threshold each fail the first test.
- Review fix, clean-source guard: `clean_source_guard_is_independent_of_the_working_directory`
  creates a temporary repository and checks `source_is_clean` from the root,
  `ockham/` and `ockham/src`. Edits to a source file, `Cargo.toml`, `Cargo.lock`,
  `rust-toolchain.toml` and a staged edit are all detected, and a clean tree
  passes. With the earlier unanchored pathspecs the test fails from the
  subdirectories. No existing assertions were removed or weakened.
- Retained branch regression evidence:
  `ockham/examples/priority_ordering_bench/tests.rs` extends
  `audit_scores_sampled_losers_individually_and_preserves_parent` to export real
  audit telemetry and reload it as later history. The round trip failed with
  `historical parent/corpus provenance mismatch` against checkpoint `bf65e57`
  before the correction and now passes. This research code is absent on Develop;
  the observed failure was against the checkpointed implementation.
- Added `historical_loader_normalises_legacy_audit_identity_and_rejects_other_corpora`
  in that file: legacy records initially failed, then passed after correction.
  Ordinary identities and legacy digests reach transfer evaluation; unrelated
  corpora and digest-valued ordinary records receive the specific provenance error.
- `cargo test --locked --lib model::pairwise`: three passed, covering ordering,
  adaptation, cold starts, invalid inputs and deterministic fits.
- Earlier branch `cargo test --locked --test real_scorer --test bounded_scorer -- --nocapture`:
  both passed with `NEAT_SCORER_BIN` pointing at the real binary; this retry did
  not repeat real scoring.
  `ockham/tests/real_scorer.rs::real_scorer_scores_the_incumbent_baseline` adds
  bounded full/sample comparisons; it pins supported behaviour already present
  before this change. `ockham/tests/bounded_scorer.rs` retains the negative
  argument/bound/error contracts.
- No existing assertions were removed. The importer caller `audit::collect` and
  evaluator `historical_rows` retain provenance checks; imported legacy rows are
  normalised before the evaluator's existing check. The latter path is exercised
  by the new compatibility test. No second audit-telemetry writer was found.
- Documentation sweep covered the README controlled-comparison and candidate-log
  sections, CONTRIBUTING prerequisites, and the complete experiment protocol.
  Updated the protocol's data/separation and current-finding sections and its
  machine-readable availability record.
  The retry clarifies that the visit population includes output-neuron skips,
  documents incomplete-pool and residual cost charges, and records the remaining
  early-failure/shared-preparation/build/final-output accounting limits.
- Security: no new endpoint, dependency or privileged operation; argument-vector
  scorer execution and existing digest/finite-value checks retained. Staged paths
  contain no secrets or hidden files. No destructive operation or path guard added.

## Merge verification

Independent Spec and Standards sub-agent reviews covered the retry diff. The
later review-fix diff (clean-source guard and verdict tests) has had only the
automated PR review that requested it; CI and merge status are not yet verified.

## Not performed and remaining risk

Representative pilot, frozen private manifest, validation scoring and paired
benefit/cost assessment were not performed because their inputs were unavailable.
Actual lineage independence and historical acquisition costs remain unverified.
Early collection failures have no per-pool strategy ledger; shared evaluation
preparation is only in actual totals, and build/final result writing is excluded.
macOS execution was not checked. No optimisation benefit or deployment is claimed.
