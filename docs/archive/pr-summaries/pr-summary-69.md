# Soft-budget contract and stuck-process kill in `docs/grq-integration.md`

## Summary

`docs/grq-integration.md` described GRQ's process wrapper as a hard wall-clock
backstop at `timeout_sec + 120` and recorded rc 124/143/137 as a routine "OS
wall-clock backstop". Both contradict the contract settled in #53: the
`--timeout-seconds` budget is **soft**, and the only hard kill is a
stuck-process judgment with hours of deliberate headroom. This change states
the contract and corrects the wrapper and exit-code wording. Documentation
only — no code, no GRQ-side change. Closes #69.

What changed in section 2 ("Budget"):

- The `grq_ockham_timeout_seconds` resolution, the `GRQ_TASK_DEADLINE_EPOCH`
  trim, the 180 s check-in reserve and the rc 2 "budget too small" skip are
  left exactly as they were.
- A new paragraph states the soft-budget contract: the budget gates the
  **start** of new work, is consulted only at safe checkpoints (top of the
  batch loop; before a full cohort is launched, where `CostModel::cohort_budget`
  trims or declines it), an in-flight scorer call is never aborted, and a run
  may exit past its budget by up to one full-corpus cohort. It also records
  that the clock starts at the optimisation loop, so the baseline score and the
  activation scan are spent before it starts counting.
- The `timeout_sec + 120` description is replaced by what the wrapper is *for*
  — an hours-scale stuck-process kill with deliberate headroom for slow
  networks and retries — with the reason a budget-tracking value loses a cohort,
  and a note that the number itself is owned and changed on the GRQ side.
- A new paragraph says what an overrun looks like in the log and that it is
  expected: the countdown on the last `batch N:` line, the `screen:`/`full:`
  lines continuing past it, `stop reason=timeout` or `stop reason=budget`,
  exit code 0, and `elapsed_ms` in the `stop` record of `experiments.jsonl`.
- A Mermaid flowchart shows the ordering: pre-budget phases → budget → deadline
  → in-flight cohort finishes → hours of headroom → wrapper SIGTERM.

The exit-code table (this document's **section 6**, "Surface contract" — the
issue refers to it as section 8) now reads rc 124/143/137 as "the wrapper judged
the process **stuck** and killed it — an incident to investigate, not routine
budget enforcement … whatever cohort was in flight is lost", keeping the
existing "`best.json` is used if present".

## Evidence

No web interface — this is a documentation-only change, so there is nothing to
screenshot. Every behavioural claim added to the doc was verified against the
source rather than inferred:

| Claim in the doc | Verified at |
|---|---|
| Deadline checked at the top of the batch loop → `stop reason=timeout` | `ockham/src/run.rs:613-615` |
| Cohort sizing/refusal before launch → `stop reason=budget`, `full: Ns left, est …ms/creature — too small for a cohort; stopping` | `ockham/src/run.rs:1003-1017`, `CostModel::cohort_budget` at `ockham/src/run.rs:318-329` |
| No deadline check inside or after a launched scorer call | `ockham/src/run.rs:1019-1095` — `evaluate_full` result handling has no deadline branch |
| Budget clock starts at the loop, after baseline + activation scan | `ockham/src/run.rs:120-160`, `588` |
| `Ns remaining` on the batch line floors at `0s` | `ockham/src/run.rs:874-879` (`saturating_duration_since`) |
| `stop reason=…` line and `elapsed_ms` in the `stop` record | `ockham/src/run.rs:1191-1207`, `ockham/src/journal.rs:188` |
| `--min-improvement` is the flag name | `ockham/src/main.rs:59-61` |

```mermaid
flowchart LR
    A[budget starts] --> B[deadline passes]
    B --> C[in-flight cohort finishes, rc 0]
    C -.->|hours of headroom| D[wrapper SIGTERM: judged stuck]
```

## Acceptance Criteria

- **met** — Section 2 states the budget is soft, gates only the start of new work, and may be exceeded by up to one scorer call — evidence: `docs/grq-integration.md` "The budget is soft: it gates the start of new work, not total runtime (Issue #53)" paragraph.
- **met** — The `timeout_sec + 120` description is gone, replaced by the stuck-process kill with hours-scale headroom, and the value is stated to be GRQ-owned — evidence: `docs/grq-integration.md` "The wrapper is a stuck-process kill, not budget enforcement" paragraph.
- **met** — The rc 124/143/137 table entry describes a stuck-process kill and no longer says "wall-clock backstop" — evidence: `docs/grq-integration.md` section 6 surface-contract table, `Exit code 124/143/137` row.
- **met** — The doc says how an overrun appears in the log and that it is expected — evidence: `docs/grq-integration.md` "What an overrun looks like in the log — and that it is expected" paragraph.
- **met** — The rc 2 "budget too small" skip and the check-in reserve are documented unchanged — evidence: `docs/grq-integration.md` first "Budget." paragraph is byte-identical up to the removed backstop sentence; `git diff` shows no edit to it.
- **met** — No behaviour change and no GRQ-side change — evidence: the diff touches `docs/grq-integration.md` and this summary only.
- **partial** — `./quality.sh` passes — reason: the codespell preflight cannot run in this container (no `pip`/`pipx`/`ensurepip`, and codespell is not packaged), so that one stage aborts before the rest; every other stage was run in the foreground and passed — bash syntax, shellcheck, neat-core version gate, `markdownlint-cli2` (0 issues), `actionlint`, `cargo deny check` (advisories/bans/licenses/sources ok), `cargo fmt --check`, clippy with the repo's deny list, `cargo test --workspace --all-features` (all green), and rustdoc with `-D warnings`. CI runs codespell for real.

## Test Plan

No tests added or modified — the change is prose in `docs/grq-integration.md`
and there is no test surface for an integration document. The checks that do
apply were run:

- `markdownlint-cli2` — 0 issues in 15 files.
- `cargo test --workspace --all-features -- --test-threads=2` — all suites pass,
  confirming no code was touched.
- `cargo fmt --check`, clippy, `cargo deny check`, rustdoc `-D warnings` — clean.
