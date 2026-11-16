## Summary

PR #278 delivered and retired the #262 soft-gate experiment. Its synthetic
smoke results were negative, and it left one gap: the representative
comparison on a mature creature. No mature creature and corpus exists on this
host, so this PR still cannot run that comparison. Instead it documents the
steps a data holder needs to run it privately. Every documented step was
checked against the pinned historical source
`91e0d56ee27627382989a09dbc31b54b64f90d20`. No code, interface or default
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
  environment, `/tmp`, `/var/tmp` or the pinned `neat-core` git checkouts. Data
  belonging to other repositories' worktrees was deliberately not used.
- The historical `Manifest::validate` refuses a manifest whose recorded
  `hardware` or `scorerEnvironment` differs from the current host. That is why
  the doc says to run every step on the study host.

## Evidence

Docs-only change; no UI file is touched. Contract check, run in a scratch
directory against the historical binary built with
`cargo build --release --locked -p neat_ai_ockham --bin neat_ai_ockham` at
`91e0d56`. The inputs were the regenerated public synthetic fixture and a
stand-in scorer (a copy of `/bin/true`). Nothing from this check is archived
as evidence:

- `soft-gate prepare <parent.json> <data> --scorer … --out manifest.json`:
  exit 0. It wrote `"representative": false`, the provenance `Unclassified
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

## Acceptance Criteria

- **missing** — Compare independent pruning searches under equal total cost.
  Measure full/held-out error, latency and Pareto frontier. — reason: the
  comparison on an existing representative creature still needs a private
  mature creature and corpus, and none is available on this host. The
  commands to run it are documented in `docs/experiments/soft-gate-v1.md`.
- **missing** — Deliver a reproducible command/manifest and machine-readable
  results with paired distributions, uncertainty, failures and actual
  coverage. — reason: the command is now documented, but representative
  results cannot exist until a data holder runs it. The synthetic results from
  #278 remain the only machine-readable results.
- **missing** — Do not claim success before representative runs. — reason:
  no representative run has been performed. No success is claimed, and the
  retention decision stays undecided.

## Test Plan

- No test added or edited. The change is documentation only, and its command
  claims were checked against the historical binary (see Evidence).
- `cargo test -p neat_ai_ockham --test soft_gate_archive < /dev/null`: passed,
  4 tests, on the final head.
- `./quality.sh < /dev/null`: exit 0, "All quality checks passed!", on the
  final head before this summary commit. The summary commit is docs-only.
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
- macOS was not checked. CI has not been observed.
