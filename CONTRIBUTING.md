# Contributing to NEAT-AI-Ockham

## Repository layout

```text
parent/
├── NEAT-AI-core/      # sibling clone; ockham/Cargo.toml depends on ../../NEAT-AI-core/neat-core
├── NEAT-AI-scorer/    # build it for integration tests: cargo build --release
└── NEAT-AI-Ockham/
```

CI checks NEAT-AI-core out beside the workspace via
`.github/actions/setup-neat-core`. `neat-core.expected-version` records the
last handled neat-core version; `scripts/check-neat-core-version.sh` fails on
an unhandled breaking bump.

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

mirrors CI: shell syntax + shellcheck, neat-core version gate, codespell,
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
