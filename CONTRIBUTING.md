# Contributing to NEAT-AI-Ockham

## Repository layout

```text
parent/
├── NEAT-AI-scorer/    # build it for integration tests: cargo build --release
└── NEAT-AI-Ockham/
```

`neat-core` is a git-tag pin on NEAT-AI-core in `ockham/Cargo.toml`
(Issue #210): Cargo fetches it like any other dependency, so nothing needs to
sit beside the repository for the workspace to build. The pin moves only
through this repository's own PRs — the `version-increment` job runs the
canonical `scripts/family-pins.sh`, which rewrites the tag to core's newest
released `v*` and lets `Cargo.lock` follow, in the same commit as the crate
bump — so a breaking core release shows up as a red build on the PR that moved
the pin, and is fixed there. CI installs the pinned Rust toolchain via
`.github/actions/setup-rust-workspace`, the shared preamble every Cargo job
runs after its own checkout, so a toolchain bump is one edit.

To build against a local core checkout — an unreleased change, say — override
the pin from *outside* the repository. Cargo reads config from parent
directories too, so a `.cargo/config.toml` in the directory that holds both
clones does it (a path there is relative to that directory):

```toml
# parent/.cargo/config.toml
[patch."https://github.com/stSoftwareAU/NEAT-AI-core"]
neat-core = { path = "NEAT-AI-core/neat-core" }
```

Never put that `[patch]` in a tracked file — this repository's own
`.cargo/config.toml` is tracked — and never in a manifest: core's
downstream-consumer gate refuses a consumer carrying one, because it could not
then prove which core it compiled.

## Prerequisites

- Rust pinned by `rust-toolchain.toml` (rustup resolves it automatically).
- `shellcheck`, `codespell` (`pip install --user codespell`),
  `cargo install cargo-deny --locked`; optionally `markdownlint-cli2` and
  `actionlint` (CI runs them regardless).
- For `ockham/tests/real_scorer.rs`: a built `rust_scorer` at
  `../NEAT-AI-scorer/target/release/rust_scorer` or `NEAT_SCORER_BIN=…`.
  The tests print a skip notice and pass when it is absent.

## Commit messages

Ockham uses **🪒** as its commit-message prefix
([#23](https://github.com/stSoftwareAU/NEAT-AI-Ockham/issues/23)). The razor is
project identity, not a Conventional Commits taxonomy — the descriptive text
still matters:

```text
🪒 add full-corpus activation statistics
🪒 prune redundant hidden neuron
🪒 collapse exact IDENTITY path
🪒 add sampled pruning sweep
```

Keep the convention lightweight. Do not add commit-lint CI that rejects
otherwise valid commits solely because the emoji is absent.

## Local gate

```bash
./quality.sh < /dev/null
```

mirrors CI: shell syntax + shellcheck, the canonical-script contract tests,
codespell,
markdownlint, actionlint, cargo-deny, `cargo fmt --check`, clippy with
`-D warnings -D clippy::filter_next -D clippy::collapsible_if`,
`cargo test --all-features`, rustdoc with `-D warnings`.

## Principles every change must keep

1. The supplied creature is never written to.
2. Version 1 accepts forward-only creatures only.
3. Candidate generation may be approximate; acceptance may not be.
4. Only a full-corpus NEAT-AI-scorer result can accept a candidate.
5. Completed structural candidates must pass NEAT-AI-core `creature.validate()`.
6. Ockham remains optional and isolated from the production evolutionary path.
7. No TypeScript runtime or implementation dependency.
8. Bump `ockham/Cargo.toml` `version` for binary-affecting changes. Git history
   is the record; do not maintain a changelog.
