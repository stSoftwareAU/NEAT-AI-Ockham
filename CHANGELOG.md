# Changelog

All notable changes to NEAT-AI-Ockham are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- Bootstrap the pure-Rust workspace, CLI skeleton, quality gates and CI
  family, following NEAT-AI-Forests / NEAT-AI-Lamarck / NEAT-AI-scorer
  conventions ([#1](https://github.com/stSoftwareAU/NEAT-AI-Ockham/issues/1)).
  `neat_ai_ockham` reports configuration (45-minute default budget, 100-wide
  5% screen knobs, external `rust_scorer` path) and does not yet prune.

- Establish an immutable forward-only incumbent, isolated workspace copy and
  fail-closed full-corpus NEAT-AI-scorer baseline
  ([#2](https://github.com/stSoftwareAU/NEAT-AI-Ockham/issues/2)).
  Recurrent creatures are rejected; the source file is never written;
  pruning is not attempted unless this gate passes.
