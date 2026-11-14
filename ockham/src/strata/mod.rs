//! Opt-in stratum-guided pruning experiment (issue #263).
//!
//! `bench-strata` asks one question: when several hidden neurons are removed
//! together, does choosing the bundle by its *worst* stratum avoid the
//! regressions an *aggregate* choice walks into? Four arms are compared on
//! the same seeds:
//!
//! - `aggregate` — bundles drawn towards neurons whose weighted-average
//!   normalised removal cost is low;
//! - `worst-stratum` — bundles drawn towards neurons whose worst normalised
//!   removal cost over the strata is low;
//! - `pareto` — candidate bundles from both samplers, kept as a non-dominated
//!   archive on predicted worst-stratum cost, synapse count and latency;
//! - `random` — the control: uniform draws.
//!
//! Authority is strict. Predictions (the importance statistics, the
//! temperatures, the Pareto objectives) only *order* candidates. The
//! full-corpus scorer alone decides which bundle wins, judged on the
//! calibration strata. The validation strata are scored once per seed, after
//! every arm has selected, and never feed back into any choice.
//!
//! The bench is opt-in and writes nothing outside its own scratch directory
//! (`<report>.work/`, removed at the end of the run). It never accepts,
//! publishes or writes a creature anywhere else, and never touches the
//! journal, the learnings store or the stats cache.
//!
//! The module is split into: [`manifest`] (the experiment description),
//! [`sampler`] (scales, statistics, sampling and temperature choice),
//! [`pareto`] (the non-dominated archive), [`report`] (the JSON report types,
//! distributions, bootstrap intervals and the predeclared criterion) and
//! [`mod@bench`] (the driver that ties them to the scorer).

pub mod bench;
pub mod manifest;
pub mod pareto;
pub mod report;
pub mod sampler;

/// The arms, in report order. Names are fixed: they appear in the report and
/// in scratch file names.
pub const ARMS: [&str; 4] = ["aggregate", "worst-stratum", "pareto", "random"];
