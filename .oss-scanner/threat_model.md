# Threat model

Disclosure channels and scope are in [SECURITY.md](../SECURITY.md); the fleet
integration that feeds Ockham its inputs is described in
[docs/grq-integration.md](../docs/grq-integration.md) and
[docs/population-entry.md](../docs/population-entry.md). This file is the
short brief for the scanner.

## What this project does and where untrusted input enters

NEAT-AI-Ockham is an experimental, offline Rust CLI (`neat_ai_ockham`) that
prunes structure from an already-fit NEAT-AI creature and keeps a cut only when
the external NEAT-AI-scorer binary (`rust_scorer`) says the result is fitter.
It opens no network connections and holds no credentials. Treat as untrusted:

- creature JSON (`--creature`, `--global-champion`), which in production is
  pulled from a repository shared by many machines;
- the training corpus directory of headerless little-endian `f32` `.bin` files
  (`--training-data`), streamed by `src/corpus.rs`;
- the fleet-shared learnings cache (`--learnings-dir`, `src/learnings.rs`),
  which other hosts write, and `experiments.jsonl` journals or candidate logs
  read back by `report` and `train-ordering`, plus a fitted `--ordering-model`;
- the scorer's stdout and result files, parsed in `src/scorer.rs`.

The scorer path, `--scorer-arg` values and the output directory are chosen by
the operator and are trusted.

## Components that matter most / least

Most important: creature loading and validation (`src/incumbent.rs`), corpus
streaming, learnings-cache parsing, the prune rewrites (`src/prune.rs`,
`src/collapse.rs`, `src/merge.rs`) and every place a file name is built from
input data (UUIDs, corpus identity, host) under the output or learnings
directory. Lower priority: `src/report.rs`, telemetry and the benches under
`ockham/examples/`. `scripts/` and `.github/` are CI tooling.

## How to exercise it

From `/src`: `cargo test --workspace --all-features` runs the suite;
`ockham/src/fixtures.rs` builds small creatures and `corpus::write_bin_file`
writes a corpus, which `ockham/tests/cli.rs` uses to drive the binary end to
end. `target/debug/neat_ai_ockham --help` lists the subcommands. `rust_scorer`
is on `PATH` when its optional install succeeded.

## How you rate severity

This is a research tool run by its own maintainers on their own hosts, so there
is no privileged boundary to cross. The crate has no `unsafe` code of its own.

- High: memory unsafety reachable from input, writing outside the output or
  learnings directory (path traversal), or running a command other than the
  operator's scorer.
- Medium: panics, unbounded memory or CPU, or hangs on malformed creature
  JSON, corpus files, journals or cache entries; a poisoned learnings entry
  that makes a host accept a candidate the scorer did not verify.
- Low: wrong statistics in reports, or issues that need the operator's own
  arguments to be hostile.

## Anything to leave alone

- Bugs in `neat-core` (NEAT-AI-core) or `rust_scorer` belong in those
  repositories.
- Lossy or surprising pruning is a fitness question, not a security one,
  unless it bypasses the scorer gate.
