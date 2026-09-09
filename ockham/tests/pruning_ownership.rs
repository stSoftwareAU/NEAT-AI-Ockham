//! Pruning-rewrite ownership, as a contract (Issue #172).
//!
//! Canonical pruning rewrites belong in `NEAT-AI-core`; Ockham owns candidate
//! choice, screening and scoring. That boundary is a design decision the
//! repository has to carry in its own documentation and in an executable check,
//! because the migration it governs (#182) outlives the issue thread.
//!
//! The first half asserts the design principles against the engine that now
//! owns them, so a core release that dropped one fails here rather than in a
//! reviewer's memory. The second half reads the checked-in documents, in the
//! `readme_contract.rs` idiom, so a principle cannot quietly leave the docs.

use std::path::{Path, PathBuf};

use neat_core::{
    MAX_SUPPORT_CONSTANTS, ProtectedKind, PruneError, PruneStats, SUPPORT_CONSTANT_BIAS,
    SynapseKey, SynapseType, TransformClass, parse_creature_json, prune_neuron, prune_synapse,
    validate_creature_topology,
};

/// One observation, one hidden neuron, one output, and a direct edge that keeps
/// the output fed once the hidden neuron goes.
const CREATURE: &str = r#"{
  "input":1,"output":1,"forwardOnly":true,
  "neurons":[
    {"type":"hidden","uuid":"h-1","bias":0.1,"squash":"LOGISTIC"},
    {"type":"output","uuid":"output-0","bias":0.25,"squash":"IDENTITY"}
  ],
  "synapses":[
    {"weight":1.0,"fromUUID":"input-0","toUUID":"h-1"},
    {"weight":1.0,"fromUUID":"input-0","toUUID":"output-0"},
    {"weight":0.2,"fromUUID":"h-1","toUUID":"output-0"}
  ]
}"#;

// ---------------------------------------------------------------------------
// The principles, asserted against the engine that owns them.
// ---------------------------------------------------------------------------

/// "constants are support nodes … at most three constants, all bias=1".
#[test]
fn core_owns_the_constant_support_invariants() {
    assert_eq!(MAX_SUPPORT_CONSTANTS, 3, "at most three support constants");
    assert!(
        (SUPPORT_CONSTANT_BIAS - 1.0).abs() < f64::EPSILON,
        "every support constant carries bias 1"
    );
}

/// "observation/input and output neurons are protected from direct deletion".
#[test]
fn core_refuses_to_delete_protected_neurons() {
    let creature = parse_creature_json(CREATURE).expect("fixture parses");
    for (uuid, expected) in [
        ("input-0", ProtectedKind::Observation),
        ("output-0", ProtectedKind::Output),
    ] {
        match prune_neuron(&creature, uuid, None) {
            Err(PruneError::Protected { kind, .. }) => assert_eq!(kind, expected, "{uuid}"),
            other => panic!("{uuid} must be protected from direct deletion, got {other:?}"),
        }
    }
}

/// "hidden neurons … are pruning candidates", and the rewrite reaches a valid
/// fixed point before anything can screen or score it.
#[test]
fn pruning_a_hidden_neuron_returns_a_validated_fixed_point() {
    let creature = parse_creature_json(CREATURE).expect("fixture parses");
    let stats = PruneStats::mean(0.5);
    let result =
        prune_neuron(&creature, "h-1", Some(&stats)).expect("hidden neuron is a candidate");

    assert_eq!(result.removed_neuron.as_deref(), Some("h-1"));
    assert!(result.passes >= 1, "the cleanup ran to a fixed point");
    assert_eq!(result.transform, TransformClass::Approximate);
    assert!(
        !result.creature.neurons.iter().any(|n| n.uuid == "h-1"),
        "the pruned neuron is gone from the result"
    );
    validate_creature_topology(&result.creature).expect("no invalid candidate reaches the scorer");
}

/// "… and synapses are pruning candidates" — including an edge out of an
/// observation neuron, which is a candidate even though the neuron is not.
#[test]
fn pruning_a_synapse_out_of_an_observation_returns_a_validated_creature() {
    let creature = parse_creature_json(CREATURE).expect("fixture parses");
    let key = SynapseKey {
        from_uuid: "input-0".to_string(),
        to_uuid: "output-0".to_string(),
        role: SynapseType::Standard,
    };
    let result = prune_synapse(&creature, &key, Some(&PruneStats::mean(0.5)))
        .expect("an edge out of an observation is a candidate");

    assert!(
        !result
            .creature
            .synapses
            .iter()
            .any(|s| s.from_uuid == "input-0" && s.to_uuid == "output-0"),
        "the requested edge is gone from the result"
    );
    assert_eq!(
        result.creature.input, creature.input,
        "the observation width itself survives the cut"
    );
    validate_creature_topology(&result.creature).expect("no invalid candidate reaches the scorer");
}

/// "invalid candidates never reach the scorer" — an unsupported request returns
/// no creature at all rather than one for Ockham to screen.
#[test]
fn an_unsupported_request_returns_no_creature() {
    let creature = parse_creature_json(CREATURE).expect("fixture parses");
    match prune_neuron(&creature, "no-such-neuron", None) {
        Err(PruneError::UnknownNeuron { uuid }) => assert_eq!(uuid, "no-such-neuron"),
        other => panic!("an unknown neuron must yield no candidate, got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// The principles, as the checked-in design reference records them.
// ---------------------------------------------------------------------------

fn read(relative: &str) -> String {
    let path: PathBuf = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join(relative);
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

fn assert_contains_all(text: &str, needles: &[&str], what: &str) {
    let missing: Vec<&str> = needles
        .iter()
        .copied()
        .filter(|n| !text.contains(n))
        .collect();
    assert!(missing.is_empty(), "{what} omits: {missing:?} (#172)");
}

/// The eight preserved design principles, each by a phrase that cannot survive
/// the principle being dropped.
const PRESERVED_PRINCIPLES: &[&str] = &[
    "hidden neurons and synapses are pruning candidates",
    "observation/input and output neurons are protected from direct deletion",
    "constants are support nodes",
    "at most three constants, all bias=1",
    "remove them automatically when unreferenced",
    "valid fixed point before screening/scoring",
    "invalid candidates never reach the scorer",
    "`validation-failed` after a supported prune is a rewrite-engine bug",
    "Ockham owns candidate choice/scoring, not structural-rewrite semantics",
];

#[test]
fn design_reference_preserves_every_pruning_principle() {
    assert_contains_all(
        &read("docs/pruning-ownership.md"),
        PRESERVED_PRINCIPLES,
        "docs/pruning-ownership.md",
    );
}

#[test]
fn design_reference_names_the_owning_repository_and_the_migration() {
    assert_contains_all(
        &read("docs/pruning-ownership.md"),
        &[
            "NEAT-AI-core",
            "stSoftwareAU/NEAT-AI-core#587",
            "#182",
            "#173",
            "#180",
        ],
        "docs/pruning-ownership.md",
    );
}

#[test]
fn readme_points_at_the_design_reference() {
    let readme = read("README.md");
    assert!(
        readme.contains("docs/pruning-ownership.md"),
        "README.md must link the pruning-ownership design reference (#172)"
    );
    let start = readme.find("## Repository layout").expect("layout section");
    let section = &readme[start..];
    let end = section[3..].find("\n## ").map_or(section.len(), |i| i + 3);
    assert!(
        section[..end].contains("pruning-ownership.md"),
        "README repository layout omits docs/pruning-ownership.md (#172)"
    );
}

#[test]
fn blocked_reasons_links_validation_failed_to_the_design_reference() {
    assert!(
        read("docs/blocked-reasons.md").contains("pruning-ownership.md"),
        "docs/blocked-reasons.md must link the design reference from \
         `validation-failed` (#172)"
    );
}
