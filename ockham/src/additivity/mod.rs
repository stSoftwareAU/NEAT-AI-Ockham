//! Cached-output bundle-ranking experiment (Issue #260).
//!
//! The experiment asks whether single-cut output deltas, measured once per
//! hidden neuron and cached, rank pruning bundles more economically than the
//! current Ockham selection. The hypothesis comes from "When Does Blockwise
//! Additivity Break?" and is not validated for Ockham: an additive prediction
//! of a bundle's output is only a guess until the full-corpus scorer says
//! otherwise.
//!
//! The experiment is opt-in and disabled in ordinary runs. Nothing here
//! accepts a creature: predictions only rank candidates for evaluation, and
//! only the full-corpus scorer can accept one. The sole cost the experiment
//! reports is the mean squared error through
//! `neat_core::loss::mse_record`, the canonical per-record cost shared with
//! the rest of the family.
//!
//! The pieces are: [`manifest`] (the predeclared protocol), [`cache`] (keyed,
//! byte-bounded single-cut outputs), [`predict`] (additive loss and output
//! prediction), [`rank`] (arms and ranking audits) and [`bench`] (the driver
//! that ties them to the existing pruning, corpus and scorer code, reporting
//! through [`bench_report`]). The `bench-additivity` command runs it; see
//! `docs/additivity-experiment.md`.

pub mod bench;
pub mod bench_report;
pub mod cache;
pub mod manifest;
pub mod predict;
pub mod rank;
