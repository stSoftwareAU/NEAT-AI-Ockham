# How GRQ drives Ockham today

This is an **audit**, not a proposal. It records how the GRQ fleet actually
invokes `neat_ai_ockham` right now, so that anyone reading this public repo can
see which of Ockham's surfaces are load-bearing before changing them.

Every claim below cites the GRQ file and function it came from, so it can be
re-verified. All citations are against `stSoftwareAU/GRQ` `Develop` at commit
`6ad319f` (2026-08-30). GRQ is a private repository; the paths are given in full
so a reader with access can check each one.

Two things in this document correct assumptions that were true earlier and are
not true at `6ad319f` — the commit **description** (section 5) and the meaning
of **exit code 2** (section 6). Both are flagged where they appear.

One thing is **retired on Ockham's side and pending on GRQ's**: since Issue #87
neuron tags are informational metadata only, so Ockham no longer writes the
pruned-tag declaration GRQ's check-in guard consumed under GRQ's #78. That half
is fixed on the GRQ branch `retire-pruned-provenance-declaration-ockham-89`
(Issue #89) but is **still live on `Develop`** until that branch merges, so the
merge must land **before** GRQ adopts the Ockham release carrying #87 —
otherwise a run that cuts a tagged neuron has its check-in refused. Section 5a
records both halves and the ordering.

## At a glance

```mermaid
sequenceDiagram
    participant N as worker/node.sh
    participant R as worker/Ockham/run.sh
    participant S as GRQ-sampler
    participant L as GRQ-Ockham (learnings)
    participant O as neat_ai_ockham

    N->>R: task "Ockham"
    R->>S: model_fetch.sh GRQ-sampler
    R->>R: select fittest samples/*.json → work dir copy
    R->>L: grq_ockham_learnings_prepare (pull)
    R->>O: grq_ockham_run (flags, --learnings-dir)
    O-->>R: out/best.json (+ coverage.txt / coverage.json)
    R->>L: grq_ockham_learnings_publish (push, before the gate)
    R->>R: score gates · rebase · lineage guard · check-in gate
    R->>S: commit subject = ockham tag, description = coverage.txt
```

## 1. Invocation

**Dispatch.** `worker/node.sh` runs the general-population worker from its
`"Ockham")` case (`./Ockham/run.sh`), and the island variant from its
`"team-ockham")` case (`./teams/team-ockham.sh`).

**The worker.** `worker/Ockham/run.sh`:

| Value | Source |
|---|---|
| `SAMPLER_REPO` | `${GRQ_SAMPLER_DIR:-<parent>/GRQ-sampler}` |
| `SAMPLES_DIR` | `${SAMPLER_REPO}/samples` |
| `WORK_DIR` | `${GRQ_OCKHAM_WORK_DIR:-${REPO_DIR}/.ockham-sampler}` |
| `OCKHAM_OUT_DIR` | `${WORK_DIR}/out` — Ockham's `--output-dir` |
| `HOST` | `uname -n`, first dot-separated label |
| `OUT_REL` | `samples/${HOST}-ockham.json` |

Order of operations, all in `worker/Ockham/run.sh`:

1. `worker/model_fetch.sh GRQ-sampler` runs to completion **before** selection —
   a concurrent clone empties the working tree, so selecting during it would
   pick from a half-populated `samples/`.
2. Training data: `GRQ_OCKHAM_DATA_DIR` when set; otherwise
   `worker/trainDataStocks.sh --mode=ockham --current` followed by
   `grq_terminal_resolve_data_dir` (`worker/shared/terminal.sh`). Failure to
   resolve is fatal.
3. Scorer: `ensure_neat_ai_native_scorer`
   (`worker/shared/ensure_neat_ai_native_scorer.sh`) when
   `NEAT_AI_RUST_SCORER_BINARY_PATH` is unset.
4. `grq_ockham_select_fittest_sample` (`worker/shared/ockham.sh`) walks
   `samples/*.json`, reads each `score` tag with `grq_ockham_read_score`, and
   prints `"<abs-path>\t<score>"` for the highest. No scored sample is fatal.
5. The work dir is deleted and recreated, and the winner is copied to
   `${WORK_DIR}/source.json`.

**Ockham never writes the source creature and never runs git.** It reads the
copy under the work dir; `worker/Ockham/run.sh` owns every commit. This is
stated in the script's own header comment and holds throughout — the only `git`
calls in the Ockham path are in `worker/Ockham/run.sh` and
`worker/shared/ockham_learnings.sh`.

**The binary.** `grq_ockham_ensure_binary` (`worker/shared/ockham.sh`) honours
`GRQ_OCKHAM_BINARY` when it is executable; otherwise it syncs the sibling
`NEAT-AI-Ockham` and `NEAT-AI-core` checkouts through `grq_ockham_fetch_repo`
(which calls `worker/model_fetch.sh`) on **every** call, and rebuilds with
`cargo build --release -q -p neat_ai_ockham` unless the marker
`target/release/.neat_ai_ockham.version` already equals the stamp
`v<crate version>+<NEAT-AI-Ockham HEAD>+<NEAT-AI-core HEAD>`
(`_grq_ockham_build_stamp`). The resolved binary's `--version` is logged by
`_grq_ockham_log_binary_version`, so a run's log names the build it used.

## 2. Flags

The full argument vector is built in `grq_ockham_run`
(`worker/shared/ockham.sh`). Positional first, then:

| Flag | Value GRQ passes | Env override |
|---|---|---|
| *(positional 1)* | `${WORK_DIR}/source.json` | — |
| *(positional 2)* | resolved 100% training data dir | `GRQ_OCKHAM_DATA_DIR` |
| `--output-dir` | `${WORK_DIR}/out` | `GRQ_OCKHAM_WORK_DIR` |
| `--timeout-seconds` | `grq_ockham_timeout_seconds` | see below |
| `--candidates` | `100` | `GRQ_OCKHAM_CANDIDATES` |
| `--screen-sample-rate` | `0.01` | `GRQ_OCKHAM_SCREEN_SAMPLE_RATE` |
| `--max-full` | **not passed**; the companion GRQ change unsets `GRQ_OCKHAM_MAX_FULL` | `GRQ_OCKHAM_MAX_FULL` |
| `--learnings-dir` | `${GRQ_OCKHAM_LEARNINGS_DIR}` | set by section 3 |
| `--learnings-host` | `grq_ockham_learnings_host` | `GRQ_OCKHAM_LEARNINGS_HOST` |
| `--learnings-replay` | only when set | `GRQ_OCKHAM_LEARNINGS_REPLAY` |
| `--seed` | only when set | `GRQ_OCKHAM_SEED` |
| `--max-experiments` | only when set | `GRQ_OCKHAM_MAX_EXPERIMENTS` |
| `--scorer` | arg 4, else `NEAT_AI_RUST_SCORER_BINARY_PATH` | both must be executable |
| `--scorer-arg` | `--gpu=<value>` only when set | `GRQ_OCKHAM_SCORER_GPU` |

Notes that matter to anyone changing Ockham's CLI:

- **`--global-champion` is never passed** — deliberate. Ockham checks in a local
  prune of this run's own source even when Forests is ahead, because breeding
  reads every `samples/*.json`.
- **`--ordering`, `--ordering-random-quota`, `--unchecked-first` and
  `--old-corpus-first` are never passed today.** `grq_ockham_run` builds no such
  arguments, so Ockham's own defaults apply. For unchecked-first that means
  `OckhamConfig::unchecked_first_enabled` (`ockham/src/config.rs`) decides, and
  it follows `--learnings-dir` — so in production the flag is **on** whenever
  the shared cache is reachable, and off when it is not.
  `old_corpus_first_enabled` follows exactly the same rule, so a production run
  with the cache reachable also reads the sibling `corpus-*` directories of
  earlier corpora and checks their still-present wins first (#88), and replays
  those confirmed winners early as hypotheses (#101). No GRQ-side change is
  needed: those directories are already under the learnings root GRQ passes, and
  the verdicts are read as **evidence** only — never as verdicts, so a replayed
  winner is re-scored against the corpus in hand and nothing from another corpus
  can suppress or accept a cut.
- **`--max-accepts` no longer exists (#96).** The cap is gone from the CLI: a
  **search** accept restarts the sweep instead of ending the search, so a run
  that is not replaying keeps searching until its budget ends. (A **replay**
  accept still ends the search and turns the rest of the budget over to screen
  coverage — that is #91 and is unchanged.) Passing the flag now fails the run
  with an unknown-flag error, so the companion GRQ change drops it — and must
  land **before** the fleet builds an Ockham that rejects it.
- **`--max-full` caps individual scoring only.** Since Issue #54 it no longer
  gates bundle construction: every screened winner reaches `bundle_plans`
  whatever the cap, so setting `GRQ_OCKHAM_MAX_FULL` again cannot re-create the
  state Issue #45 was raised about, where 30 of 38 winners were discarded before
  any combination was built. The companion GRQ change unsets the variable, so
  production runs with no cap at all.
- The three `--learnings-*` flags are only appended when
  `GRQ_OCKHAM_LEARNINGS_DIR` is non-empty; an unreachable cache means the whole
  block is absent, not empty.
- The scorer path is checked with `-x` before `--scorer` is added. A
  non-executable path is silently not passed, and Ockham falls back to its own
  scorer resolution.

**Budget.** `grq_ockham_timeout_seconds` (`worker/shared/ockham.sh`) starts at
`GRQ_OCKHAM_RUN_TIMEOUT_SECONDS`, else `GRQ_OCKHAM_TIMEOUT_SECONDS`, else
`2700` (45 minutes). When `GRQ_TASK_DEADLINE_EPOCH` is set it is trimmed to the
remaining task time less `GRQ_OCKHAM_CHECKIN_RESERVE_SEC` (default 180); under
120 seconds remaining it returns 1, and `grq_ockham_run` turns that into rc 2 —
a clean skip (section 6).

**The budget is soft: it gates the start of new work, not total runtime
(Issue #53).** `--timeout-seconds` is the window in which Ockham may *start*
work; it is not a bound on how long the process runs. The deadline is consulted
only at safe checkpoints — the top of the batch loop, and again before a
full-corpus cohort is launched, where `CostModel::cohort_budget`
(`ockham/src/run.rs`) trims the cohort to what the remaining clock can pay for
or declines to start one at all. **Ockham never aborts a scorer call it has
already launched.** A full-corpus cohort still running when the deadline passes
finishes, and its results are processed exactly as an in-budget cohort's are:
the winner is applied if it clears `--min-improvement`, the learnings are filed,
`best.json` is written. A run may therefore exit *after* its budget, typically
by up to the length of one full-corpus cohort — minutes, not hours. Two phases
sit outside the budget altogether: the clock starts when the optimisation loop
starts, so the authoritative baseline score and the hidden-neuron activation
scan (several minutes on a large corpus) are already spent before the first
second of it is counted.

```mermaid
flowchart LR
    A["baseline + activation scan<br/>(before the clock starts)"] --> B["budget starts<br/>--timeout-seconds"]
    B --> C["batches — deadline checked<br/>at safe checkpoints only"]
    C --> D["deadline passes:<br/>no new work starts"]
    D --> E["in-flight cohort finishes,<br/>results kept, rc 0"]
    E -.->|hours of headroom| F["wrapper SIGTERM:<br/>process judged stuck"]
```

**The wrapper is a stuck-process kill, not budget enforcement.** GRQ runs the
binary under `timeout`/`gtimeout` `-s SIGTERM`. That wrapper exists to kill a
process the fleet judges **stuck** — wedged on a network mount, a scorer that
never returns — and nothing else. Its value therefore belongs on the order of
**hours**, with deliberate headroom over the budget to absorb slow networks,
retries and one overrunning cohort; it must not track the budget. A SIGTERM
shortly after the deadline lands in the middle of an in-flight cohort, and that
cohort's several minutes of full-corpus scoring — and any winner in it — are
discarded, leaving a run that reports no improvement. That is the rc 124 in
Issue #53's fleet log, and it is why "budget plus a couple of minutes" is the
wrong shape for this value. The wrapper's number lives on the GRQ side and is
changed there; this document records only the contract it must satisfy.

**What an overrun looks like in the log — and that it is expected.** A run that
finishes past its budget is normal operation:

- the last `batch N: … candidates, … hidden left, Ns remaining` line shows a
  small remaining figure (it floors at `0s`), and that batch's `screen:` and
  `full:` lines continue past the deadline;
- the run ends with `stop reason=timeout` (the deadline was seen at the top of
  the next batch) or `stop reason=budget` (the remaining clock could not pay for
  another cohort, preceded by the warning `full: Ns left, est …ms/creature —
  too small for a cohort; stopping`);
- the exit code is **0**, `best.json` is written, and the run's `stop` record in
  `<output-dir>/experiments.jsonl` carries `elapsed_ms` — the number to compare
  against the budget when judging how far the run ran over.

An rc 124/143/137 is *not* what an overrun looks like: that is the wrapper's
stuck-process kill (section 6), and it means work was thrown away.

**Stdout discipline.** The binary is invoked with `>&2`. `grq_ockham_run`'s own
stdout *is its return value* — the caller reads the `best.json` path out of a
command substitution — so anything Ockham prints to stdout must not be mixed
into it. This is the `#4463` regression, where a run summary on stdout made
`BEST` a 51,000-line string and every accepted prune on the fleet was discarded.

## 3. Learnings cache

`worker/shared/ockham_learnings.sh` is the whole GRQ side of the shared cache.
The optimiser never runs git; GRQ moves the files.

- **Repo** — `grq_ockham_learnings_repo`: `${GRQ_OCKHAM_LEARNINGS_REPO}`, else
  `<parent>/GRQ-Ockham` beside the other data repos.
- **Directory** — `grq_ockham_learnings_dir`: `learnings` for the general
  population, `learnings-team-<island>` for an island. An island name that is
  not `^[A-Za-z0-9][A-Za-z0-9._-]*$` is refused, because the name reaches both a
  directory path and a `git add` pathspec.
- **Prepare** — `grq_ockham_learnings_prepare "" "Ockham"`, called from
  `worker/Ockham/run.sh`, syncs the clone via `grq_ockham_learnings_sync` →
  `grq_ockham_fetch_repo` → `worker/model_fetch.sh`, creates the directory, and
  prints it. The worker exports it as `GRQ_OCKHAM_LEARNINGS_DIR`. Every fault
  prints nothing, warns loudly, and the run continues **without** the cache.
- **Health** — each outcome is recorded against the `ockham-learnings`
  capability (`grq_ockham_learnings_capability`, `grq_capability_degraded` /
  `grq_capability_available` in `worker/shared/capability_health.sh`), so a
  cache that is off on every run stops being a warning nobody reads.
- **Publish** — `grq_ockham_learnings_publish` commits the run's appended
  records as `🪒 Ockham learnings from <host> (<rel>) (#4401)` and pushes,
  retrying `git_push_safe` / `git_pull_safe --rebase` up to
  `GRQ_OCKHAM_LEARNINGS_PUSH_ATTEMPTS` (default 5) times. Nothing staged is a
  silent success; a git fault is a warning and never a stage failure.

**Publish happens before the improvement gate.** `worker/Ockham/run.sh` calls
`grq_ockham_learnings_publish` immediately after `grq_ockham_run` returns and
*before* it inspects the score, so **REFUSED verdicts are published too** — a
refused cut is exactly as valuable to the fleet as an accepted one, because it
is a full-corpus scorer call nobody has to pay again.

**Layout.** From the file's own header:

```text
learnings/corpus-<identity>/<host>.jsonl                verdicts, general population
learnings/screens-<identity>/<host>.jsonl               screen coverage, pre-#76 (read only)
learnings/screens/<host>.jsonl                          screen coverage, general population
learnings-team-<island>/corpus-<identity>/<host>.jsonl  verdicts, one island
learnings-team-<island>/screens/<host>.jsonl            screen coverage, one island
```

Append-only, and **each host appends only to its own `<host>.jsonl`**, so two
hosts never touch the same file: no merge driver is needed and a concurrent push
resolves as a fast-forward or a clean rebase. A record names a **neuron UUID**,
not a portable patch, so a replayed verdict is only useful while that UUID is
still on the incumbent — the crate's own store does that filtering, and GRQ
changes nothing about it.

**A visit that scored nothing is written at screen-record version 3** (#93).
Version 2 records are candidates the scorer screened. A pre-#93 binary accepts
only versions 1 and 2 and *skips* anything else, which is deliberate: it has no
notion of a visit with nothing to score and would otherwise publish a coverage
percentage far above what it had screened. Mixed-version fleets therefore
degrade to the old figures on old hosts rather than to inflated ones.

**Screen records are not keyed by the corpus path; screen *coverage* is scoped
to the corpus** (#76, #100). GRQ regenerates the training corpus
(`worker/trainDataStocks.sh --mode=ockham --current`), so a new identity —
therefore a new `corpus-<identity>/` — is routine. Keying the screen *path* on
it was wrong: each identity saw its own slice and re-screened neurons another
identity had already checked, and a record written under an identity nothing
came back to was stranded. Screens therefore live in one stable `screens/`
directory with the identity carried on the record (`corpusIdentity`, screen
format version 2), and the old `screens-<identity>/` directories are still read
— their records stamped with the identity the directory name carries — so no
fleet history is lost.

What the identity on the record then decides is the **screening epoch** (#100).
A run counts only the records measured against the corpus in front of it, so
`sweep X/X checked (100.0% of epoch)` means the sweep finished *that* corpus,
not that Ockham is done — and since Issue #102 every surface says so in those
words: the check-in subject reads `sweep X/Y (Z% of epoch <short-id>)`, and a
finished sweep is reported as `sweep complete for this epoch`. A corpus that is
repacked with identical content hashes the same and keeps its coverage; a corpus
that is extended starts a fresh epoch at `0 / hidden`, with every hidden neuron
— `blocked` and `known-failure` included — eligible again. Nothing is deleted, so a host that returns to an earlier identity finds
that epoch intact. `coverage.json` and the journal `coverage` record carry
`corpusIdentity`, and the commit-description block gains an `epoch:` line.
**Read `checked` across an identity change as a new epoch, not as lost
coverage.** Nothing else in the GRQ path changes: publish still stages the
whole learnings directory.

Islands are `CLOSED_BIOSPHERE=1` by design and therefore get their own
directory: an island stops paying twice for its own refusals without ever seeing
a verdict the general population discovered.

## 4. Check-in path

In `worker/Ockham/run.sh`, in order. Every "discard" below is `exit 0` — a clean
no-improvement outcome is a success, not a host failure.

1. **Run outcome.** rc 2 → `skipped (budget too small)`, exit 0. Any other
   non-zero → `grq_fail_loud_fatal`.
2. **Path guard.** `grq_assert_candidate_path "Ockham" "${BEST}"`
   (`worker/shared/candidate_path.sh`) — `BEST` must be a readable regular file,
   checked before anything can mistake a run summary for a filename.
3. **Pre-rebase improvement gate (`#4528`).** `grq_ockham_read_score "${BEST}"`
   must beat `SOURCE_SCORE` under `grq_ockham_score_beats` (strict `>`, awk
   float). A readable losing score discards here, before the rebase's cargo
   build, sampler pull and full-corpus scoring are paid for. An *unreadable*
   score falls through to the existing post-rebase fatal.
4. **Rebase re-entry (`#4431`).** When `grq_rebase_enabled`
   (`worker/shared/rebase.sh`), the sampler is pulled and `grq_rebase_candidate`
   runs with direction **`onto-candidate`** — not `onto-champion`. Ockham's
   contribution is *removed* neurons, which NEAT-AI-Rebase v1 cannot harvest out
   of a published creature, so `BEST` must be the base or the run's work is
   silently dropped. On rc 0 with a file, `BEST` becomes the rebased creature and
   `REBASE_BASE` / `REBASE_BASE_SCORE` are read from `grq_rebase_base_path` /
   `grq_rebase_base_score`. Every other outcome keeps `BEST`.
5. **Score readable.** `CANDIDATE_SCORE` empty → fatal
   (`Ockham best.json has no numeric score tag`).
6. **Gate baseline (`#4532`).** The improvement gate judges against the rebase
   base's score when it is readable, otherwise against the source, with a
   warning. A rebased creature was never scored against the sample the run
   opened on, and judging it there discarded creatures the scorer had just
   preferred to the fleet champion.
7. **Lineage guard.** `grq_creature_guard_checkin_lineage "Ockham" <source>
   <candidate-before-rebase> <best> <rebase-base> <pruned-provenance.json>`
   (`worker/shared/creature_provenance_guard.sh`) walks the publication lineage
   one hop at a time: creature-level tag *names* must survive (except `score`
   and `dataSha`, which NEAT-AI sheds at the mutation site), per-neuron `tags`
   must survive on every neuron that survives, and `uuid` / `memetic` must be
   absent. A refusal is a **skipped check-in**, exit 0 — not a stage failure.
   The sixth argument is GRQ's Issue #78: for a pruning optimiser the "a TAGGED
   neuron was cut" case was judged against Ockham's declaration of what it
   removed. **That argument is still live on GRQ `Develop` and Ockham no longer
   writes the file** — the retirement is committed on the GRQ branch
   `retire-pruned-provenance-declaration-ockham-89` and not yet merged (#89,
   section 5a).
8. **Check-in gate.** `grq_ockham_validate_for_checkin` (`worker/shared/ockham.sh`)
   delegates to `grq_validate_for_checkin <file> ockham 🪒`
   (`worker/shared/validate_for_checkin.sh`), which asks the TypeScript engine —
   `src/IntelligentDesign/ValidateForCheckin.ts` — whether the general
   population can read the creature: `Creature.validate()` passes, and
   `fromJSON` → `exportJSON` returns every neuron and synapse. It **refuses; it
   never repairs**, because repair rewrites the creature that was measured. A
   refusal is `exit 1`.
9. **Publish subshell**, inside `SAMPLER_REPO`: `enforce_never_merge_json`,
   `sampler_pull_with_auto_resolve "ockham-sampler" "ours"`, then the second
   half of the tip gate — the candidate must also beat the score on the current
   tip `samples/${HOST}-ockham.json`, or it is discarded. Then
   `format_creature_json` writes `OUT_REL`, `grq_ockham_verify_written_score`
   reads the score back and refuses a mismatch, `safe_git_add` stages,
   `scan_json_merge_markers` refuses merge markers, `grq_ockham_git_commit`
   commits, and `git_push_safe` pushes with up to 12 pull-rebase-recommit
   retries (the retry subject gains `, attempt N` and carries the same
   description).

`GRQ_OCKHAM_SAMPLER_CHECKIN_HOOK` short-circuits the whole check-in for
hermetic tests, as `GRQ_OCKHAM_TEST_MODE=1` + `GRQ_OCKHAM_HOOK` does for the
binary.

## 5. Commit message

**The subject is the creature's `ockham` tag.** `grq_ockham_read_message`
(`worker/shared/ockham.sh`) reads
`[.tags[]? | select(.name == "ockham") | .value] | .[0]` out of `best.json` and
`worker/Ockham/run.sh` uses it verbatim as `MSG`, the first `-m` of the commit.
GRQ deliberately reads no other program's tag — Forests, Lamarck and Intelligent
Design stamp their own.

That tag is written by `CreatureMeta::stamp_acceptance` in `ockham/src/tags.rs`,
which upserts `score`, `error` and `ockham`, the last rendered by
`ockham_progress_message` in the same file.

Since Issue #91 the tag is stamped **twice** on a run that keeps screening after
its accept: once when the cut is applied and `best.json` is written, and again
at the end of the run with the coverage and batch count the run finished on.
Only the tag text changes — the creature `best.json` carries is the one the
accept produced — so the subject's `sweep X/Y` agrees with the `coverage.txt`
block GRQ relays beside it instead of freezing at the figure at the cut.

GRQ adds exactly two things to the subject:

- the razor prefix `🪒` (with a trailing space) when the tag does not already
  start with it; and
- the rebase outcome marker from `grq_rebase_commit_note`
  (`worker/shared/rebase_message.sh`) — `[🪢 rebase: applied, …]`, `attempted`,
  or `failed (rc N)`, and **no marker at all** when the rebase block never ran.

When the tag is empty (a creature with no `ockham` tag), the worker falls back
to `🪒 Ockham improvement of ${OUT_NAME}, ${SCORE_MSG}` with `SCORE_MSG` from
`grq_format_score_message`. GRQ never appends a second score clause to a real
tag subject — the tag already carries one.

**Correction to the premise of this audit's issue.** Issue #34 asked this
document to record that "there is no commit description assembly today —
`git commit -m "${MSG}"` is subject-only". That was true when #34 was written;
it is **not** true at GRQ `6ad319f`. GRQ `#4525` wired the description:

- `grq_ockham_read_coverage "${OCKHAM_OUT_DIR}"` (`worker/shared/ockham.sh`)
  reads `coverage.txt` from Ockham's `--output-dir`;
- `grq_ockham_git_commit "${MSG}" "${COVERAGE}"` (same file) issues
  `git commit -m "<subject>" -m "<coverage block>"`, one helper shared by the
  first commit and the push-retry commit so the two can never disagree.

GRQ computes nothing here — Ockham does all the measuring and renders the block
(`ockham/src/coverage.rs::write_files`), and GRQ relays it verbatim, **beside**
and never inside the `ockham` tag subject. Absent or whitespace-only
`coverage.txt` — an older binary, or a run with no learnings dir — prints
nothing, returns 0, and produces the byte-identical subject-only commit GRQ made
before. The gap that sub-issues of #33 exist to close is therefore closed on the
GRQ side; what remains is Ockham's own reporting quality, not the relay.

## 5a. The pruned-tag declaration: Ockham retired, GRQ pending (Issues #75, #78, #87, #89)

Neuron tags are **informational metadata only**. They describe where a neuron
came from; they never change what Ockham may prune, and cutting one needs no
permission and no declaration. Ockham therefore publishes no declaration
artefact: the file it wrote beside `best.json` under Issue #75 is gone.

**The GRQ half is written but not yet merged.** On the GRQ branch
`retire-pruned-provenance-declaration-ockham-89` (this repo's #89):

- `worker/shared/creature_provenance_guard.sh` loses the declaration parser and
  the declared / undeclared / forgiven prune helpers, so
  `grq_creature_guard_checkin` is one three-argument function every check-in path
  and both lineage hops share;
- `worker/Ockham/run.sh` passes no sixth argument;
- `worker/teams/README.md` documents the retirement in place of declared mode.

On GRQ `Develop` the sixth argument is **still live** and the guard still fails
closed when it cannot read a declaration.

What still holds, unchanged, is the tag round-trip: **a neuron that survives the
run keeps its `tags` byte-for-byte**, and a neuron that is cut takes its tags
with it rather than leaving a `tags` entry for a neuron that no longer exists.
That is enforced by `ockham/src/tags.rs`
(`a_cut_tagged_neuron_leaves_no_tags_entry_and_the_survivors_keep_theirs`) and
by `ockham/src/sweep.rs`, where a tag never skips a neuron.

**Release ordering — the hazard until the GRQ PR merges.** GRQ's guard on
`Develop` refuses a check-in on which a tagged neuron is missing undeclared, and
an absent declaration grants no relaxation. Ockham stopped writing that file at
Issue #87, so the retirement branch must **merge before** GRQ adopts the Ockham
release carrying #87. Until it does, a run that legitimately cuts a tagged
neuron has its check-in refused — exit 0, silently skipped — and the fleet stops
making progress on tagged creatures. A human releases and merges that GRQ PR;
this repo cannot.

**Re-pin when it merges.** This document audits GRQ `Develop` at `6ad319f`, so
the branch name above is a temporary pointer. Once the retirement lands on
`Develop`, four places describe the old state and must be re-pinned to the new
commit: this section, the summary in the header, step 7 of section 4 (which
still gives the six-argument call form live on `Develop` today) and the two
`pruned-provenance` rows of the section 6 surface contract.

```mermaid
flowchart LR
    O["Ockham cuts a hidden neuron"] --> T{"Did it carry tags?"}
    T -- yes --> C["Cut like any other —<br/>its tags leave with it"]
    T -- no --> C
    C --> S["Survivors keep their tags<br/>byte-for-byte"]
    S --> G["GRQ check-in"]
```

## 6. Surface contract

What GRQ reads out of Ockham. Changing any row breaks a live fleet worker.

| Surface | Read by | Contract |
|---|---|---|
| `<output-dir>/best.json` | `grq_ockham_run` | Must exist after the run, or the helper returns 1. It is the helper's printed return value. |
| `score` creature tag | `grq_ockham_read_score` | Numeric (`^-?\d+(\.\d+)?([eE][-+]?\d+)?$`). Drives both score gates and `grq_ockham_verify_written_score`. |
| `ockham` creature tag | `grq_ockham_read_message` | The commit **subject**, used verbatim. Must already carry its own single score clause. |
| Other creature tag *names* | `grq_creature_guard_checkin_lineage` | Every source tag name must survive on the candidate, except `score` / `dataSha`. `error` is stamped by `stamp_acceptance` and matters here as a name. |
| Per-neuron `tags` | `grq_creature_guard_checkin_lineage` | Must survive on every neuron that **survives**, byte-for-byte. A neuron that is cut takes its tags with it. GRQ `Develop` still refuses an undeclared cut of a tagged neuron (its Issue #78); since Ockham's Issue #87 that refusal has nothing left to read, and the retirement is committed but unmerged on the GRQ branch `retire-pruned-provenance-declaration-ockham-89` (#89, section 5a). |
| `<output-dir>/pruned-provenance.json` | `grq_creature_guard_checkin_lineage`, sixth argument | **No longer written** since Issue #87 — neuron tags are informational, so a tagged cut is declared to nobody. GRQ `Develop` still passes the path and fails closed on the absent file; the branch that stops requiring it must merge before GRQ adopts an Ockham release ≥ the one carrying #87 (#89). |
| `uuid` / `memetic` keys | `grq_creature_guard_checkin_lineage` | Must be **absent** from the written creature. |
| `<output-dir>/coverage.txt` | `grq_ockham_read_coverage` | Line-oriented block relayed verbatim as the commit description. Since Issue #59 it may carry `winners:` / `bundles:` / `dropped:` lines after the coverage lines; each is omitted when it has nothing to report, and a run that screened nothing renders the coverage lines alone. Since Issue #74 the tagged line reads `tagged:    N carry tags, screened like any other` — it replaced the `skipped:` line, and is still omitted when no neuron is tagged. Issue #87 removed the `declared:` line that followed it. Issue #93 added a `blocked:   N checked with no cut proposed` line before the tagged line — omitted when nothing is blocked — and the `progress:` line now reads `newly checked` rather than `newly screened`. Issue #100 added an `epoch:     corpus <identity> — coverage counts this corpus only` line naming the corpus the figures were measured against. Issue #102 made the block epoch-aware throughout: the first line is now `sweep:     N of M hidden (Z% of epoch)`, the `epoch:` line moved directly under it and carries the first eight characters of the identity, `unchecked:` reads `N remaining this epoch` — or `0 remaining — sweep complete for this epoch` when the sweep has finished — and a `history:   N of M ever checked across K corpus epochs` line after `progress:` reports cumulative coverage, omitted when the store holds no records. Absent or blank is a supported no-op. |
| `<output-dir>/coverage.json` | *nothing in GRQ today* | Written by Ockham; GRQ reads only `coverage.txt`. Since Issue #59 it carries an additive `winners` object beside the existing coverage fields. Issue #74 changed no key: `checkable` now counts **every** hidden neuron, tagged ones included, so the percentage no longer overstates progress. Issue #93 added an additive `blocked` key — the checked UUIDs no cut was proposed for — and a pre-#93 file still deserialises. Issue #100 added an additive `corpusIdentity` key naming the screening epoch — in full, however short the prose form is — and a pre-#100 file still deserialises. Issue #102 added an additive `history` object (`checkedEver`, `epochs`) carrying cumulative coverage across every epoch, kept out of the current-epoch percentage; a pre-#102 file still deserialises. |
| Process stdout | *nothing* | Redirected to stderr by `grq_ockham_run`. `grq_ockham_run`'s **own** stdout must carry the `best.json` path and nothing else. |
| Exit code 0 | `grq_ockham_run` | Success; `best.json` must be present. |
| Exit code 124/143/137 | `grq_ockham_run` | The wrapper judged the process **stuck** and killed it — an incident to investigate, not routine budget enforcement (section 2): whatever cohort was in flight is lost. Warned, and `best.json` is used if present. |
| Any other non-zero | `grq_ockham_run` | Reported as `neat_ai_ockham exited <rc>` and turned into helper rc **1**, a fatal run fault. |

**Correction: exit code 2.** `worker/Ockham/run.sh` treats rc 2 from
`grq_ockham_run` as "budget too small; nothing to check in". That rc 2 is
generated **inside `grq_ockham_run`** when `grq_ockham_timeout_seconds` reports
under 120 seconds left — it is not the binary's exit code. The binary's own
`ExitCode::from(2)` (`ockham/src/main.rs`, config validation failure) reaches
GRQ through the "any other non-zero" row above and is a **fault**, not a clean
skip. An invalid flag is therefore reported loudly rather than mistaken for a
budget skip.

## The island variant

`worker/shared/island_ockham.sh::grq_island_ockham_run`, dispatched by
`worker/teams/team-ockham.sh`, is the same optimiser wired for one island. It
shares `grq_ockham_run`, `grq_ockham_learnings_prepare` / `_publish` and the
same rc-2 skip, and differs in four ways:

- it runs only for an island with `ISLAND_OCKHAM=1`
  (`grq_island_ockham_enabled`, `worker/shared/island_select.sh`);
- the source is the island-wide fittest `*-100.json`
  (`grq_island_select_fittest`), not a GRQ-sampler sample;
- the wall clock is fitted into the remaining task hour through the one-shot
  `GRQ_OCKHAM_RUN_TIMEOUT_SECONDS` override
  (`grq_island_task_fit_budget_seconds`), never the durable default; and
- acceptance goes through `grq_island_accept_candidate <team> ockham …`
  (`worker/shared/island_accept.sh`) to `${HOST}-ockham-100.json`, so there is
  no GRQ-sampler commit and none of section 5 applies.

`GRQ_OCKHAM_LEARNINGS_DIR` is exported for the island's own
`learnings-team-<island>` directory and unset again immediately after the run,
so one island's cache can never bleed into the next island of the same dispatch.
