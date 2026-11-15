## Summary

Removed the optional soft-gate runtime because representative benefit had not
earned its maintenance cost. Preserve the investigation, negative synthetic
findings, original failure and a tested historical replay command. The
representative controlled comparison remains unperformed. Part of #262.

## Spec

### Intent and Rationale

- Investigate one frozen-parent hidden-neuron ranking across nested
  full/90%/75%/50% budgets, compared with independent pruning searches.
- Reuse core inference/pruning, corpus streaming, activation statistics and
  the full directory scorer. The historical experiment remains available at
  commit `91e0d56ee27627382989a09dbc31b54b64f90d20`.
- Remove the gate fitter, family builder, manifest runtime, public API, CLI,
  generator and their feature tests from the maintained tree. Runtime source
  under `ockham/src/` now matches `origin/Develop`; no second pruning engine
  remains in the current library or executable.

### Essential Design Decisions

- Preserve the original manifests/results unchanged. A small replay script
  exports the pinned historical source, builds with `--locked`, regenerates
  checksum-matched synthetic inputs, records fresh scorer/build/hardware pins
  and locks a new manifest before fitting or evaluation.
- Keep the maintenance decision separate from the scientific verdict:
  `soft-gate-retirement-v1.json` records removal and inconclusive efficacy.
  No optimisation benefit or representative negative result is claimed.
- The historical numerical-difference method supports only forward-only,
  untyped IDENTITY/LOGISTIC/TANH networks. Its manifest records calibration,
  untouched validation, paired seeds, family limits and success criteria;
  full results record actual sizes, costs, failures, error and latency.

### Undiscoverable Facts

- No suitable representative mature-creature/corpus pair was supplied or
  identified locally. Actual coverage is zero representative creatures/pairs.
- The historical paper/method pin and reference-source checksums remain in the
  protocol/manifests. No raw-paper checksum is claimed.
- An earlier attempt raised no PR because the Vibe Coder pre-push gate
  blocked it, not because of a branch defect. Because the branch adds a `.sh`
  file, the gate runs `ci.yml`'s runlib refresh step outside Actions, without
  its `env:` block. In this run, `runAffectedShellcheckChecks` failed on
  `RUNNER_TEMP: unbound variable` with that script changed, and passed with
  `README.md` changed.

## Evidence

- `docs/experiments/soft-gate-smoke-results-v1.json` preserves ten paired seeds
  and 20 completed families over 512 synthetic records: 64 calibration and
  128 validation records. Achieved hidden/synapse/growth counts were 20/60/26,
  18/54/23.4, 15/45/19.5 and 10/30/13.
- The paired mean score difference was −0.0001831, with conditional 95%
  interval [−0.0002441, −0.0001222]. Held-out-error and latency criteria also
  failed. This is synthetic wiring evidence on a shared host.
- `scripts/replay-soft-gate-smoke.sh` completed with the real scorer in this
  retry: ten pairs, 20 families and zero failures. The new replay manifest,
  compressed full results and verification JSON are committed under
  `docs/experiments/soft-gate-replay-*-v1.*`. The paired score distribution and
  interval match the original; the executable hash also matches, while the
  scorer hash and timing samples differ.
- Both full-output archives retain rankings, identities, latency samples,
  costs and rewrite trails. Their decompressed checksums and manifest links
  were verified. The original failed preflight remains preserved separately.
- `docs/experiments/soft-gate-retirement-v1.json` checksum-links the evidence,
  records the negative findings and explicitly leaves the representative
  comparison outstanding. README and protocol now describe an archive.

## Acceptance Criteria

<!-- vibe-spec-review inputs="diff+issue-body" -->

- **met** — First inspect current code, merged work and repository instructions.
  Reuse existing mechanisms; if functionality already exists, this issue still
  owns the missing controlled comparison rather than a second implementation. —
  evidence: `docs/experiments/soft-gate-v1.md`; final runtime source matches
  `origin/Develop`, and the representative comparison remains acknowledged —
  reviewer: met
- **met** — Opt-in experiment only; preserve ordinary behaviour, interfaces,
  immutable parents and full authoritative acceptance. No automatic deployment
  or default change. — evidence:
  `ockham/tests/soft_gate_archive.rs::retired_experiment_is_absent_from_the_installed_command`;
  historical replay builds separate source and emits diagnostic outputs —
  reviewer: met
- **met** — Predeclare controls, seeds, sample size, compute limits and
  success/failure criteria. Identify code/model/data versions; separate
  calibration from untouched validation and count all preparation/search costs. —
  evidence: original/replay manifests and compressed full results record pins,
  splits, limits and preparation/search/evaluation costs for the performed smoke
  protocol — reviewer: met
- **met** — Provide deterministic positive/negative correctness tests for new
  behaviour and pass the repository's normal quality checks. Research runs are
  separate from routine feature CI. — evidence:
  `ockham/tests/soft_gate_archive.rs` (four tests); full `./quality.sh` passed
  with the real scorer enabled; research replay runs outside CI — reviewer: met
- **partial** — Deliver a reproducible command/manifest and machine-readable
  results with paired distributions, uncertainty, failures and actual coverage.
  Fixtures alone do not demonstrate an optimisation benefit. — evidence:
  `scripts/replay-soft-gate-smoke.sh`, archived manifests, failure, full results
  and `docs/experiments/soft-gate-replay-verification-v1.json` — reviewer: partial —
  reason: the representative controlled comparison remains unperformed; actual
  representative coverage is zero and fixtures cannot establish benefit.
- **met** — Record negative findings and remove optional code that does not
  earn its maintenance cost. Do not claim success before representative runs. —
  evidence: optional CLI/API/fitter/family/manifest runtime removed;
  `docs/experiments/soft-gate-retirement-v1.json` records negative synthetic
  findings and distinguishes maintenance removal from inconclusive science —
  reviewer: met
- **met** — Keep public implementation and fixtures application-agnostic;
  private corpora/results and operational integration remain private. —
  evidence: committed manifests/results concern only the generated synthetic
  TANH fixture; no private corpus or operational integration is introduced —
  reviewer: met
- **met** — Leave unassigned until an implementer actively starts. `work-on`
  authorises the investigation, not skipping prerequisite verification. —
  evidence: #262 is assigned to the active implementer; the representative
  prerequisite remains explicitly outstanding — reviewer: met

## Standards Review

<!-- vibe-standards-review inputs="diff+CODING-STANDARDS.md" -->

- **clean** — The Standards reviewer checked CONTRIBUTING.md principles,
  unchanged ordinary runtime and dependency/core pins, private historical
  replay, archive consistency/refusal tests and documentation. No
  `CODING-STANDARDS.md` exists in this repository; the applicable written
  standards were `CONTRIBUTING.md` and `quality.sh`.

## Test Plan

- `cargo test -p neat_ai_ockham --test soft_gate_archive --test readme_contract < /dev/null`:
  passed, 14 tests. Archive tests cover negative findings/coverage/hash links,
  original/replay distributions, installed-command retirement and refusal of
  invalid inputs/existing outputs without changing them.
- `NEAT_SCORER_BIN=/var/tmp/vibe-cargo-target/NEAT-AI-scorer-560c793c/release/rust_scorer ./quality.sh < /dev/null`:
  passed, including shellcheck, codespell, markdownlint, actionlint,
  cargo-deny, formatting, clippy, workspace tests and rustdoc.
- Retry run: `./quality.sh < /dev/null` re-run on the unchanged code at
  `03cdc00` without `NEAT_SCORER_BIN` set: exit 0, "All quality checks passed!".
- Historical replay against that real scorer: exit 0, all 20 families complete,
  no failures, matched declared costs and zero representative coverage.
  Full-output hashes, uncertainty arithmetic and evidence links were verified.
- Spec reviewer independently passed all four archive tests and verified replay
  distributions/checksums. Standards reviewer independently checked replay
  privacy/refusal, source equality, documentation and archive integrity.
- The retired runtime's tests remain at the pinned historical revision. No
  existing base-branch test assertions were changed.

## Merge verification

CI and ordinary pull-request review have not yet been observed. Representative
mature-creature performance remains untested; this summary intentionally keeps
that acceptance criterion partial and the issue open.
