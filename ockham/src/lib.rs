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
//! | full-corpus activation statistics | [`stats`] | #3 |
//! | mean-activation ablation + cleanup | later | #4 |
//! | exact IDENTITY collapse | later | #5 |
//! | seeded sampled sweep | later | #6 |
//! | full scoring + bundles | later | #7 |
//! | iterative 45-minute loop | later | #8 |
//! | population re-entry | later | #9 |
//! | economics report | later | #10 |

#![warn(missing_docs)]

pub mod baseline;
pub mod cancel;
pub mod config;
pub mod corpus;
pub mod fixtures;
pub mod incumbent;
pub mod log;
pub mod run;
pub mod scorer;
pub mod stats;

pub use baseline::{AuthoritativeBaseline, establish_baseline};
pub use cancel::CancelToken;
pub use config::{
    ConfigReport, DEFAULT_CANDIDATE_COUNT, DEFAULT_MIN_IMPROVEMENT, DEFAULT_SCREEN_SAMPLE_RATE,
    DEFAULT_SCREEN_THRESHOLD, DEFAULT_TIMEOUT_SECONDS, OckhamConfig,
};
pub use corpus::{CorpusInfo, corpus_info};
pub use incumbent::{Incumbent, IncumbentError, load_incumbent};
pub use run::{BaselineRun, establish_run};
pub use scorer::{DirectoryScorer, ExternalScorer, ScoreResult, ScorerError, ScorerMode};
pub use stats::{ActivationStats, NeuronStats, ensure_activation_stats};

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
