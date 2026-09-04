//! NEAT-AI-Ockham — experimental pruning optimiser for already-fit NEAT-AI
//! creatures.
//!
//! > **Every neuron must earn its keep — prune freely, trust only the scorer.**
//!
//! This crate is organised as a pipeline. Cheap search only ever *proposes*
//! candidates. Only a full-corpus NEAT-AI-scorer result can *accept* one.
//! Issue #1 lands the workspace, CLI and external-scorer invocation. Issue #2
//! establishes the immutable forward-only incumbent and authoritative baseline.
//! Later issues fill the remaining stages:
//!
//! | Stage | Module | Issue |
//! |---|---|---|
//! | run configuration | [`config`] | #1 |
//! | external NEAT-AI-scorer judge | [`scorer`] | #1 |
//! | immutable incumbent + checksum | [`incumbent`] | #2 |
//! | authoritative baseline | [`baseline`], [`run`] | #2 |
//! | corpus identity / streaming | [`corpus`] | #2 |
//! | sampled activation statistics | [`stats`] | #3, #44 |
//! | mean-activation ablation + cleanup | [`ablation`] | #4 |
//! | exact IDENTITY collapse | [`collapse`] | #5 |
//! | seeded sampled sweep | [`sweep`] | #6 |
//! | full scoring + bundles | [`promote`] | #7 |
//! | iterative 45-minute loop | [`run`], [`journal`] | #8 |
//! | population re-entry | [`reentry`] | #9 |
//! | economics report | [`report`] | #10 |
//! | GRQ check-in tags | [`tags`] | #25 |
//! | fleet learnings store + replay | [`learnings`] | #27 |
//! | named candidate orderings | [`ordering`] | #11 |
//! | screening coverage over the incumbent | [`mod@coverage`] | #37 |
//! | GRQ commit-description coverage files | [`mod@coverage`], [`run`] | #40 |

#![warn(missing_docs)]

pub mod ablation;
pub mod baseline;
pub mod blocked;
pub mod cancel;
pub mod collapse;
pub mod config;
pub mod corpus;
pub mod coverage;
pub mod fixtures;
pub mod incumbent;
pub mod journal;
pub mod learnings;
pub mod log;
pub mod ordering;
pub mod promote;
pub mod reentry;
pub mod report;
pub mod run;
pub mod scorer;
pub mod stats;
pub mod substitute;
pub mod sweep;
pub mod tags;

pub use ablation::{Ablation, AblationSkip, TransformClass, ablate_mean};
pub use baseline::{AuthoritativeBaseline, establish_baseline};
pub use blocked::{BlockedBreakdown, BlockedReason};
pub use cancel::CancelToken;
pub use collapse::{CollapseOptions, CollapseSkip, IdentityCollapse, collapse_identity};
pub use config::{
    ConfigReport, DEFAULT_CANDIDATE_COUNT, DEFAULT_MIN_IMPROVEMENT, DEFAULT_SCREEN_SAMPLE_RATE,
    DEFAULT_SCREEN_THRESHOLD, DEFAULT_TIMEOUT_SECONDS, OckhamConfig,
};
pub use corpus::{CorpusInfo, RecordRange, corpus_info};
pub use coverage::{
    COVERAGE_JSON_FILE, COVERAGE_TEXT_FILE, Coverage, coverage, write_files as write_coverage_files,
};
pub use incumbent::{Incumbent, IncumbentError, load_incumbent};
pub use ordering::{Ordering, OrderingConfig, hidden_order};
pub use promote::{FullOutcome, evaluate_full};
pub use report::{Report, summarise};
pub use run::{BaselineRun, establish_run};
pub use scorer::{DirectoryScorer, ExternalScorer, ScoreResult, ScorerError, ScorerMode};
pub use stats::{ActivationStats, NeuronStats, SampleSpec, ensure_activation_stats};
pub use substitute::{ConstantSubstitution, SubstitutionSkip, substitute_constant};
pub use sweep::{
    ScreenConfig, ScreenOutcome, ScreenedLoser, Sweep, SweepCandidate, draw_seed, screen_batch,
};
pub use tags::{CreatureMeta, OckhamProgress, ockham_progress_message};

/// Crate version from `ockham/Cargo.toml`.
pub fn crate_version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[cfg(test)]
mod tests {
    use neat_core::parse_creature_json;

    #[test]
    fn neat_core_parses_a_minimal_forward_only_creature() {
        let json = r#"{
            "semanticVersion": "4.0.0",
            "forwardOnly": true,
            "input": 1,
            "output": 1,
            "neurons": [
                {"type": "output", "uuid": "o1", "bias": 0.0, "squash": "IDENTITY"}
            ],
            "synapses": []
        }"#;
        let creature = parse_creature_json(json).expect("minimal creature must parse");
        assert!(creature.forward_only);
        assert_eq!(creature.input, 1);
        assert_eq!(creature.output, 1);
        assert_eq!(creature.neurons.len(), 1);
    }
}
