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

- Stream full-corpus hidden-neuron activation statistics through the
  NEAT-AI-core compiled forward pass
  ([#3](https://github.com/stSoftwareAU/NEAT-AI-Ockham/issues/3)).
  Mean, variance, mean-abs, min and max are accumulated in `f64` with
  memory bounded by hidden-neuron count, cached by creature checksum +
  corpus identity + format version, and never used as an acceptance score.

- Mean-activation neuron ablation with recursive exact cleanup
  ([#4](https://github.com/stSoftwareAU/NEAT-AI-Ockham/issues/4)).
  A hidden neuron is removed from an incumbent clone after folding
  `mean × weight` into ordinary downstream biases; dead-output chains and
  known-squash constant neurons then fold exactly. Typed/aggregate cases
  fail closed. The incumbent is never mutated; only `creature.validate()`
  candidates are emitted.

- Exact cost-aware IDENTITY neuron collapse
  ([#5](https://github.com/stSoftwareAU/NEAT-AI-Ockham/issues/5)).
  A hidden IDENTITY neuron is rewritten into downstream biases and
  bypass synapses (`bias_z += bias_y * b`, `x_k → z` weight `+= a_k * b`)
  on an incumbent clone. Parallel synapses merge; automatic collapses
  that raise NEAT growth units are skipped unless an experimental
  override is set.

- Seeded random-without-replacement sweep and 100-wide 5% scorer screening
  ([#6](https://github.com/stSoftwareAU/NEAT-AI-Ockham/issues/6)).
  Hidden neurons are shuffled once, visited at most once per permutation,
  and screened with the incumbent in the same sampled scorer cohort.
  Sampled winners are returned for later full scoring; they cannot become
  `best.json`.

- Full-score every sampled winner plus grouped pruning bundles
  ([#7](https://github.com/stSoftwareAU/NEAT-AI-Ockham/issues/7)).
  Ranked prefixes of sampled winners are rebuilt from the same incumbent
  and scored with the individuals in one full-corpus call. The highest
  strict full-score win — including a tiny one — may become the next
  Ockham parent. Sampled wins never write `best.json`.

- Wire the 45-minute iterative Ockham loop and experiment journal
  ([#8](https://github.com/stSoftwareAU/NEAT-AI-Ockham/issues/8)).
  After the baseline gate, hidden neurons are swept without replacement
  until the wall-clock budget, an empty permutation, or consecutive
  scorer failures. An accepted local win recomputes activation statistics
  and restarts the sweep. The global champion is not consulted here.

- Re-score Ockham best against the current global champion before population
  re-entry ([#9](https://github.com/stSoftwareAU/NEAT-AI-Ockham/issues/9)).
  A fresh same-call full-corpus comparison writes
  `population-candidate.json` only when Ockham has positive authoritative
  headroom. Local `best.json` is always preserved.
