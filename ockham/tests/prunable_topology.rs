//! There is no unsafe topology — a contract (Issue #192).
//!
//! Every hidden neuron the incumbent carries, and every edge it lists, is a
//! pruning target: the shared NEAT-AI-core engine rewrites typed roles,
//! aggregate targets, `IF` structure and observation-incident edges alike
//! (#182). A visit that produces no candidate is missing a *value*, or is a
//! defect in the request — never a topology the razor is not allowed to touch.
//!
//! `blocked-reasons.md` says `unsafe-topology` and — since Issue #200 —
//! `aggregate-squash` are retired. This is the executable half: the shapes that
//! used to earn either code are pruned here, and no refusal any Ockham
//! transform can report carries a retired code any more.

use neat_ai_ockham::AblationSkip;
use neat_ai_ockham::blocked::BlockedReason;
use neat_ai_ockham::collapse::CollapseSkip;
use neat_ai_ockham::fixtures::{creature, neuron, synapse, typed_synapse};
use neat_ai_ockham::merge::{LinearRelation, MergeSkip};
use neat_ai_ockham::prune::{prune_edge, prune_hidden_neuron};
use neat_ai_ockham::substitute::SubstitutionSkip;
use neat_core::CreatureExport;

/// Every shape the retired code used to cover, on one valid creature.
///
/// `h_cond` is an IDENTITY feeding a `condition` role — the typed-synapse case;
/// `h_if` is the `IF` that reads it; `h_mean` is an aggregate target that
/// cannot absorb a bias fold; `constant-0` and the implicit `input-0` are the
/// non-hidden sources whose outgoing edges used to be refused outright.
fn adversarial() -> CreatureExport {
    creature(
        1,
        1,
        vec![
            neuron("constant", "constant-0", 1.0, None),
            neuron("hidden", "h_cond", 0.0, Some("IDENTITY")),
            neuron("hidden", "h_arm", 0.0, Some("TANH")),
            neuron("hidden", "h_if", 0.0, Some("IF")),
            neuron("hidden", "h_mean", 0.0, Some("MEAN")),
            neuron("output", "output-0", 0.0, Some("IDENTITY")),
        ],
        vec![
            synapse("input-0", "h_cond", 1.0),
            synapse("input-0", "h_arm", 1.0),
            typed_synapse("h_cond", "h_if", 1.0, "condition"),
            typed_synapse("h_arm", "h_if", 1.0, "positive"),
            typed_synapse("constant-0", "h_if", -1.0, "negative"),
            synapse("h_if", "h_mean", 1.0),
            synapse("input-0", "h_mean", 1.0),
            synapse("h_mean", "output-0", 1.0),
        ],
    )
}

/// Every hidden neuron on that creature is prunable, whatever it is wired into.
#[test]
fn every_hidden_neuron_is_a_pruning_candidate() {
    let incumbent = adversarial();
    neat_ai_ockham::incumbent::validate_creature(&incumbent).expect("the fixture is valid");
    let hidden: Vec<&str> = incumbent
        .neurons
        .iter()
        .filter(|n| n.neuron_type == "hidden")
        .map(|n| n.uuid.as_str())
        .collect();
    assert_eq!(hidden, vec!["h_cond", "h_arm", "h_if", "h_mean"]);
    for uuid in hidden {
        let built = prune_hidden_neuron(&incumbent, uuid, Some(0.25), None)
            .unwrap_or_else(|e| panic!("`{uuid}` must be prunable, got: {e}"));
        neat_ai_ockham::incumbent::validate_creature(&built.creature)
            .unwrap_or_else(|e| panic!("`{uuid}` produced an invalid candidate: {e}"));
        assert!(
            built.after.hidden_neurons < built.before.hidden_neurons,
            "`{uuid}` was not removed: {:?}",
            built.detail
        );
    }
}

/// Every edge it lists is one too — typed roles, aggregate targets, and the
/// observation- and constant-sourced edges that used to fail closed.
#[test]
fn every_listed_edge_is_a_pruning_candidate() {
    let incumbent = adversarial();
    let mut pairs: Vec<(&str, &str)> = incumbent
        .synapses
        .iter()
        .map(|s| (s.from_uuid.as_str(), s.to_uuid.as_str()))
        .collect();
    pairs.dedup();
    assert_eq!(pairs.len(), 8, "every edge is visited: {pairs:?}");
    for (from, to) in pairs {
        let built = prune_edge(&incumbent, from, to, Some(0.25), None)
            .unwrap_or_else(|e| panic!("`{from}`→`{to}` must be cuttable, got: {e}"));
        neat_ai_ockham::incumbent::validate_creature(&built.creature)
            .unwrap_or_else(|e| panic!("`{from}`→`{to}` produced an invalid candidate: {e}"));
        // The count is not the check: cleanup may hand an `IF` back a
        // zero-weight support edge for the role the cut emptied (core rule 12),
        // so what must be true is that the requested edge went.
        assert!(
            built
                .detail
                .removed_synapses
                .iter()
                .any(|s| s.from_uuid == from && s.to_uuid == to),
            "`{from}`→`{to}` was not removed: {:?}",
            built.detail
        );
    }
}

/// No refusal any Ockham transform can report carries a retired code.
///
/// Every variant of every skip enum is constructed and asked for its reason, so
/// a variant added later that reached for `unsafe-topology` or
/// `aggregate-squash` fails here.
#[test]
fn no_transform_refusal_reports_a_retired_code() {
    let uuid = || "h_x".to_string();
    let reasons: Vec<(&str, BlockedReason)> = vec![
        (
            "ablation:unknown",
            AblationSkip::UnknownNeuron(uuid()).blocked_reason(),
        ),
        (
            "ablation:not-hidden",
            AblationSkip::NotHidden {
                uuid: uuid(),
                neuron_type: "output".into(),
            }
            .blocked_reason(),
        ),
        (
            "ablation:aggregate-neuron",
            AblationSkip::AggregateNeuron {
                uuid: uuid(),
                squash: "MEAN".into(),
            }
            .blocked_reason(),
        ),
        (
            "ablation:non-finite",
            AblationSkip::NonFiniteMean(f64::NAN).blocked_reason(),
        ),
        (
            "ablation:typed",
            AblationSkip::TypedSynapse {
                from_uuid: uuid(),
                to_uuid: "h_if".into(),
                synapse_type: "condition".into(),
            }
            .blocked_reason(),
        ),
        (
            "ablation:unknown-synapse",
            AblationSkip::UnknownSynapse {
                from_uuid: uuid(),
                to_uuid: "h_if".into(),
            }
            .blocked_reason(),
        ),
        (
            "ablation:aggregate-target",
            AblationSkip::AggregateTarget {
                uuid: uuid(),
                squash: "MEAN".into(),
            }
            .blocked_reason(),
        ),
        (
            "ablation:unknown-squash",
            AblationSkip::UnknownSquash {
                uuid: uuid(),
                squash: String::new(),
            }
            .blocked_reason(),
        ),
        (
            "ablation:invalid",
            AblationSkip::Invalid("no".into()).blocked_reason(),
        ),
        (
            "ablation:empty-group",
            AblationSkip::EmptyGroup.blocked_reason(),
        ),
        (
            "collapse:unknown",
            CollapseSkip::UnknownNeuron(uuid()).blocked_reason(),
        ),
        (
            "collapse:not-hidden",
            CollapseSkip::NotHidden {
                uuid: uuid(),
                neuron_type: "output".into(),
            }
            .blocked_reason(),
        ),
        (
            "collapse:not-identity",
            CollapseSkip::NotIdentity {
                uuid: uuid(),
                squash: "TANH".into(),
            }
            .blocked_reason(),
        ),
        (
            "collapse:typed",
            CollapseSkip::TypedSynapse {
                from_uuid: uuid(),
                to_uuid: "h_if".into(),
                synapse_type: "condition".into(),
            }
            .blocked_reason(),
        ),
        (
            "collapse:aggregate-target",
            CollapseSkip::AggregateTarget {
                uuid: uuid(),
                squash: "MEAN".into(),
            }
            .blocked_reason(),
        ),
        (
            "collapse:self-loop",
            CollapseSkip::SelfLoop { uuid: uuid() }.blocked_reason(),
        ),
        (
            "collapse:cost",
            CollapseSkip::CostIncrease {
                before: 1.0,
                after: 2.0,
            }
            .blocked_reason(),
        ),
        (
            "collapse:invalid",
            CollapseSkip::Invalid("no".into()).blocked_reason(),
        ),
        (
            "merge:unknown",
            MergeSkip::UnknownNeuron(uuid()).blocked_reason(),
        ),
        (
            "merge:not-hidden",
            MergeSkip::NotHidden {
                uuid: uuid(),
                neuron_type: "output".into(),
            }
            .blocked_reason(),
        ),
        ("merge:same", MergeSkip::SameNeuron(uuid()).blocked_reason()),
        (
            "merge:non-finite",
            MergeSkip::NonFiniteRelation(LinearRelation {
                scale: f64::NAN,
                offset: 0.0,
            })
            .blocked_reason(),
        ),
        (
            "merge:no-outgoing",
            MergeSkip::NoOutgoing(uuid()).blocked_reason(),
        ),
        (
            "merge:typed",
            MergeSkip::TypedSynapse {
                from_uuid: uuid(),
                to_uuid: "h_if".into(),
                synapse_type: "condition".into(),
            }
            .blocked_reason(),
        ),
        (
            "merge:aggregate-target",
            MergeSkip::AggregateTarget {
                uuid: uuid(),
                squash: "MEAN".into(),
            }
            .blocked_reason(),
        ),
        (
            "merge:self-loop",
            MergeSkip::SelfLoop { uuid: uuid() }.blocked_reason(),
        ),
        (
            "merge:not-forward",
            MergeSkip::NotForward {
                survivor_uuid: uuid(),
                to_uuid: "h_if".into(),
            }
            .blocked_reason(),
        ),
        (
            "merge:cost",
            MergeSkip::CostIncrease {
                before: 1.0,
                after: 2.0,
            }
            .blocked_reason(),
        ),
        (
            "merge:invalid",
            MergeSkip::Invalid("no".into()).blocked_reason(),
        ),
        (
            "substitute:unknown",
            SubstitutionSkip::UnknownNeuron(uuid()).blocked_reason(),
        ),
        (
            "substitute:not-hidden",
            SubstitutionSkip::NotHidden {
                uuid: uuid(),
                neuron_type: "output".into(),
            }
            .blocked_reason(),
        ),
        (
            "substitute:non-finite",
            SubstitutionSkip::NonFiniteMean(f64::NAN).blocked_reason(),
        ),
        (
            "substitute:no-outgoing",
            SubstitutionSkip::NoOutgoing(uuid()).blocked_reason(),
        ),
        (
            "substitute:invalid",
            SubstitutionSkip::Invalid("no".into()).blocked_reason(),
        ),
    ];
    for (name, reason) in reasons {
        assert!(
            !reason.is_retired(),
            "{name} reports the retired `{}`",
            reason.code()
        );
    }
}

/// Core's own refusals are reported under live codes too, so nothing reaches
/// the reason tally as `unsafe-topology` through the engine boundary.
#[test]
fn core_refusals_are_reported_under_live_codes() {
    let incumbent = adversarial();
    let refusals = [
        prune_hidden_neuron(&incumbent, "nope", Some(0.25), None).unwrap_err(),
        prune_hidden_neuron(&incumbent, "output-0", Some(0.25), None).unwrap_err(),
        prune_hidden_neuron(&incumbent, "constant-0", Some(0.25), None).unwrap_err(),
        prune_hidden_neuron(&incumbent, "h_cond", Some(f64::NAN), None).unwrap_err(),
        prune_edge(&incumbent, "h_cond", "output-0", Some(0.25), None).unwrap_err(),
    ];
    for refusal in refusals {
        assert!(
            !refusal.blocked_reason().is_retired(),
            "{refusal} reports the retired `{}`",
            refusal.blocked_reason().code()
        );
    }
}

/// A retired code still *reads*, so fleet history deserialises unchanged.
///
/// `aggregate-squash` joined `unsafe-topology` in retirement under Issue #200:
/// core converts a one-edge aggregate target, folds a zero-edge one into its
/// bias and otherwise drops the term as an approximate transform, so no binary
/// files it any more.
#[test]
fn the_retired_codes_are_still_read_off_a_record() {
    for (code, expected) in [
        ("unsafe-topology", BlockedReason::UnsafeTopology),
        ("aggregate-squash", BlockedReason::AggregateSquash),
    ] {
        let reason = BlockedReason::from_code(code);
        assert_eq!(reason, expected);
        assert!(reason.is_retired(), "{code} must read as retired");
    }
    assert_eq!(
        BlockedReason::ALL
            .into_iter()
            .filter(|r| r.is_retired())
            .collect::<Vec<_>>(),
        vec![
            BlockedReason::AggregateSquash,
            BlockedReason::UnsafeTopology
        ],
        "the retired set is the contract a reader drops records against"
    );
}
