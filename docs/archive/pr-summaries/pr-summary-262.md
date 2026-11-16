## Summary

PR #278 delivered and retired the #262 soft-gate experiment. Its synthetic
smoke results were negative, and it left one gap: the representative
comparison on a mature creature. No mature creature and corpus was found on
this host in the places listed under Undiscoverable Facts, so this PR still
cannot run that comparison. Instead it documents the steps a data holder
needs to run it privately. Every documented flag and refusal was checked
against the pinned historical source
`91e0d56ee27627382989a09dbc31b54b64f90d20`, and the steps listed under
Evidence were also run against that binary. No code, interface or default
changes. Part of #262.

## Spec

### Intent and Rationale

- `docs/experiments/soft-gate-v1.md` explained only how to replay the
  synthetic smoke study. It never said how to run the comparison on private
  data. The new **Representative run (private, not yet performed)** section
  gives the build, `soft-gate prepare`, declare, lock and `soft-gate run`
  steps, so the outstanding work no longer waits on rediscovering the
  retired CLI.
- The runtime stays retired. Running the study from the history tag reuses
  the archived implementation, so there is no second one.

### Essential Design Decisions

- The doc states that `"representative": true` is only the operator's own
  declaration. The historical runtime sets `representativeCreatures` from that
  flag and checks only the manifest's shape and pinned inputs. Calling the
  flag a verified property would invite a false representative claim.
- The study's manifest and results stay private. A public record may state
  only the verdict, the paired summary and the coverage counts, as the issue's
  public/private split requires.

### Undiscoverable Facts

- On this host, no mature creature/corpus pair was found in the checkout, the
  environment, `/tmp`, `/var/tmp` or the pinned `neat-core` git checkouts. The
  only `creature.json` files are three 198-byte single-neuron test fixtures.
  Data belonging to other repositories' worktrees was deliberately not used.
- The historical `Manifest::validate` (`manifest.rs:378-384` at `91e0d56`)
  refuses a manifest whose recorded `hardware` or `scorerEnvironment` differs
  from the current process. `scorerEnvironment` is the five `NEAT_SCORER_*`
  variables, not the host. That is why the doc says to run every step on the
  study host and to keep those variables unchanged between `prepare` and
  `run`.

## Evidence

Docs-only change; no UI file is touched. Updated docs: `README.md` (the
`## Soft-gate experiment archive` paragraph links the new section) and
`docs/experiments/soft-gate-v1.md` (new **Representative run (private, not
yet performed)** section). Contract check, run in a scratch
directory against the historical binary built with
`cargo build --release --locked -p neat_ai_ockham --bin neat_ai_ockham` at
`91e0d56`. The inputs were the regenerated public synthetic fixture and a
stand-in scorer (a copy of `/bin/true`). Nothing from this check is archived
as evidence:

- `soft-gate prepare <parent.json> <data> --scorer … --out manifest.json`
  (record counts left at their default of 256, which the doc passes
  explicitly): exit 0. It wrote `"representative": false`, the provenance `Unclassified
  input: smoke study until provenance is reviewed` and 256/256 calibration
  and validation records.
- `soft-gate run` with the pre-edit SHA after editing the manifest: exit 1,
  `manifest checksum mismatch`, no output directory created.
- `soft-gate run` on a locked manifest with blank `dataProvenance`: exit 1,
  `invalid executable checksum, hardware or data provenance`, no output
  directory created.
- `soft-gate run` on a locked manifest declaring `representative: true`:
  validation passed, both families failed at scoring (`scorer output
  malformed: EOF while parsing a value at line 1 column 0`), exit 1. The run
  wrote `study/results.json` with `representativeCreatures: 1`,
  `criteria.representativeData: true`, `completePairs: false` and
  `retentionDecision: undecided-no-completed-representative-comparison`.
- The retention condition named in the doc matches the historical
  `study.rs`:
  `comparable = manifest.representative && complete && matched_cost && n >= 10`.
- The documented flags (`--scorer`, `--out`, `--calibration-records`,
  `--validation-records`, `--manifest-sha256`) exist in `main.rs:222-245` at
  `91e0d56`. The `NEAT_SCORER_*` refusal was read from the source, not run.

## Acceptance Criteria

<!-- vibe-spec-review inputs="diff+issue-body" -->

- **met** — Freeze ordinary weights; fit hidden-neuron gates on a calibration
  sample across multiple capacity targets. — evidence:
  `docs/experiments/soft-gate-v1.md`, `soft-gate-smoke-results-v1.json` (from
  #278) — reviewer: met
- **met** — Produce nested full/90%/75%/50% candidates using canonical rewrites
  and actual achieved sizes. — evidence: `achievedCapacity` in
  `soft-gate-smoke-results-v1.json`, hidden counts 20/18/15/10 — reviewer: met
- **partial** — Compare independent pruning searches under equal total cost.
  Measure full/held-out error, latency and Pareto frontier. — evidence:
  `soft-gate-smoke-results-v1.json` (cost-matched pairs, `full.error`,
  `validationMse`, `medianLatencyNs`, Pareto flags),
  `docs/experiments/soft-gate-v1.md#representative-run-private-not-yet-performed`
  — reviewer: partial — reason: measured only on a synthetic TANH fixture; the
  comparison on one existing mature creature is documented but not run,
  because no such creature and corpus is available on this host.
- **met** — Non-differentiable activations need an explicit supported method or
  exclusion; no silent fictitious gradients. — evidence:
  `docs/experiments/soft-gate-v1.md` (IDENTITY/LOGISTIC/TANH only, others
  refused) — reviewer: met
- **met** — Read and pin the full method. — evidence: method-pin table in
  `docs/experiments/soft-gate-v1.md` — reviewer: met
- **partial** — Inspect current code and reuse existing mechanisms; the issue
  owns the missing controlled comparison. — evidence: the new section reuses
  the archived binary at `91e0d56` rather than a second implementation —
  reviewer: partial — reason: reuse is met, but the representative comparison
  the issue owns is still not performed.
- **met** — Opt-in only; preserve ordinary behaviour, interfaces, immutable
  parents; no deployment or default change. — evidence: `ockham/src/`
  unchanged;
  `ockham/tests/soft_gate_archive.rs::retired_experiment_is_absent_from_the_installed_command`
  — reviewer: met
- **met** — Predeclare controls, seeds, sample size, compute limits and
  success/failure criteria; identify versions; separate calibration from
  untouched validation; count all costs. — evidence:
  `docs/experiments/soft-gate-smoke-manifest-v1.json` — reviewer: met (for the
  synthetic run; no representative manifest exists)
- **met** — Deterministic positive/negative tests; normal quality checks;
  research runs separate from CI. — evidence:
  `ockham/tests/soft_gate_archive.rs` (4 tests pass) — reviewer: met
- **met** — Deliver a reproducible command/manifest and machine-readable
  results with paired distributions, uncertainty, failures and actual
  coverage. — evidence: `soft-gate-retirement-v1.json` `replayCommand`,
  `soft-gate-replay-verification-v1.json`, `pairedScoreDeltas`,
  `conservative95Interval`, the preflight failure record,
  `representativeCreatures: 0`, and the representative-run commands in
  `docs/experiments/soft-gate-v1.md` — reviewer: met — the results report the
  actual coverage, zero representative creatures; no representative results
  exist and none are claimed.
- **met** — Record negative findings and remove optional code that does not
  earn its maintenance cost. — evidence: `soft-gate-retirement-v1.json`
  `negativeFindings`/`removed` — reviewer: met
- **met** — Do not claim success before representative runs. — evidence:
  `soft-gate-retirement-v1.json` `scientificVerdict:
  inconclusive-no-representative-comparison`; `README.md` ("supports no
  benefit claim"); `docs/experiments/soft-gate-v1.md` (no benefit claimed; the
  new section says only a complete representative run can support the
  retention verdict) — reviewer: met — the prohibition is honoured; the
  representative run itself is tracked under the comparison criterion above.
- **met** — Keep public implementation and fixtures application-agnostic;
  private corpora/results stay private. — evidence: manifest `dataProvenance`
  (public synthetic fixture); the new section keeps the manifest and results
  in `$PRIVATE_DIR` — reviewer: met

## Standards Review

<!-- vibe-standards-review inputs="diff+CODING-STANDARDS.md" -->

- **violation** — Pending CI belongs in `## Merge verification` — evidence:
  `docs/archive/pr-summaries/pr-summary-262.md` — reason: fixed in this diff;
  the section is restored.
- **violation** — Prose must name every condition the code checks — evidence:
  `docs/experiments/soft-gate-v1.md` (Representative run) — reason: fixed in
  this diff; the doc now says to keep the five `NEAT_SCORER_*` variables
  unchanged between `prepare` and `run`. It also notes that the hardware
  record includes the usable CPU count, and that an external `timeout` kill
  may leave no `results.json`.
- **violation** — No over-claiming: "No mature creature and corpus exists" and
  "Every documented step was checked" — evidence:
  `docs/archive/pr-summaries/pr-summary-262.md` Summary — reason: fixed in
  this diff; both sentences are scoped to what was searched and run.
- **violation** — Evidence must name the updated doc files — evidence:
  `docs/archive/pr-summaries/pr-summary-262.md` Evidence — reason: fixed in
  this diff.
- **violation** — Accuracy of the acceptance criteria ("Do not claim success"
  marked missing) — evidence: `docs/archive/pr-summaries/pr-summary-262.md`
  — reason: fixed in this diff; the reviewer's verdicts are recorded above.
- **violation** — Completeness of the documented command (`$PRIVATE_DIR`
  never created, no `git worktree remove`, no GNU tools note) — evidence:
  `docs/experiments/soft-gate-v1.md` — reason: fixed in this diff.
- **clean** — Command flags and refusals against `91e0d56`, stdin from
  `/dev/null`, privacy (placeholders only), markdownlint and codespell, the
  README anchor, and placement of the new section.

## Test Plan

- No test added or edited. The change is documentation only, and its command
  claims were checked against the historical binary (see Evidence).
- `cargo test -p neat_ai_ockham --test soft_gate_archive < /dev/null`: passed,
  4 tests, after the final doc edit.
- `./quality.sh < /dev/null`: exit 0, "All quality checks passed!", on the
  tree with the scorer-environment and command fixes. A last two-sentence
  wording edit to `docs/experiments/soft-gate-v1.md` (CPU allowance, external
  `timeout`) was followed by markdownlint-cli2 (0 issues) and codespell. The
  worker re-runs the full quality gate.
- Docs sweep: grepped `README.md` and `docs/` for `soft-gate prepare`,
  `soft-gate run` and `representative`. The README's `## Soft-gate experiment
  archive` paragraph now links the new section. No other surface described a
  representative run.

## Not done

- The representative comparison itself, and therefore the retention or
  removal decision on representative evidence.
- The documented `soft-gate run` was not taken to completion with a real
  scorer in this run. #278's replay already verified that path on synthetic
  data.
- macOS was not checked.

## Merge verification

CI and ordinary pull-request review have not yet been observed.
