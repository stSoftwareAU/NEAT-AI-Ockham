use neat_ai_ockham::baseline::fake::ScriptedScorer;
use neat_ai_ockham::fixtures::{creature, neuron, synapse};
use neat_ai_ockham::promote::evaluate_full_proposal;

fn parent() -> neat_core::CreatureExport {
    creature(
        2,
        1,
        vec![neuron("output", "output-0", 0.0, Some("IDENTITY"))],
        vec![
            synapse("input-0", "output-0", 1.0),
            synapse("input-1", "output-0", 0.1),
        ],
    )
}

#[test]
fn a_prebuilt_proposal_uses_the_strict_full_corpus_win_rule() {
    let tmp = tempfile::tempdir().unwrap();
    let parent = parent();
    let proposal =
        neat_ai_ockham::prune::prune_edge(&parent, "input-1", "output-0", Some(0.0), None)
            .unwrap()
            .creature;
    let scorer = ScriptedScorer {
        baseline_score: 0.5,
        candidate_score: Some(0.625),
        ..Default::default()
    };
    let (measured, winner) = evaluate_full_proposal(
        &scorer,
        tmp.path(),
        &parent,
        &proposal,
        0.125,
        &tmp.path().join("tie"),
    )
    .unwrap();
    assert_eq!(measured.delta, 0.125);
    assert_eq!(winner, None, "an equal threshold is not a strict win");
    assert_eq!(
        scorer.last_mode.get(),
        Some(neat_ai_ockham::ScorerMode::Full)
    );
    let scorer = ScriptedScorer {
        candidate_score: Some(0.75),
        ..scorer
    };
    let (measured, winner) = evaluate_full_proposal(
        &scorer,
        tmp.path(),
        &parent,
        &proposal,
        0.125,
        &tmp.path().join("win"),
    )
    .unwrap();
    assert_eq!(measured.after.synapses, 1);
    assert_eq!(winner.unwrap().creature, proposal);
    assert_eq!(parent.synapses.len(), 2);
}

#[test]
fn failed_malformed_and_non_finite_proposals_never_return_a_winner() {
    let tmp = tempfile::tempdir().unwrap();
    let parent = parent();
    for (index, scorer, cause) in [
        (
            0,
            ScriptedScorer {
                fail_with: Some("invalid record width".into()),
                ..Default::default()
            },
            "invalid record width",
        ),
        (
            1,
            ScriptedScorer {
                raw_output: Some(r#"{"baseline":{"score":0.5,"error":0}}"#.into()),
                ..Default::default()
            },
            "no proposal",
        ),
        (
            2,
            ScriptedScorer {
                candidate_score: Some(f64::NAN),
                ..Default::default()
            },
            "non-finite",
        ),
    ] {
        let error = evaluate_full_proposal(
            &scorer,
            tmp.path(),
            &parent,
            &parent,
            0.01,
            &tmp.path().join(index.to_string()),
        )
        .unwrap_err();
        assert!(error.contains(cause), "{error}");
    }
    let scorer = ScriptedScorer::ok(0.5, 0.0);
    let error = evaluate_full_proposal(
        &scorer,
        tmp.path(),
        &parent,
        &parent,
        0.0,
        &tmp.path().join("invalid"),
    )
    .unwrap_err();
    assert_eq!(
        error,
        "proposal: min_improvement must be finite and positive"
    );
    let existing = tmp.path().join("existing");
    std::fs::create_dir(&existing).unwrap();
    std::fs::write(existing.join("owner.txt"), "retained").unwrap();
    assert!(
        evaluate_full_proposal(&scorer, tmp.path(), &parent, &parent, 0.01, &existing)
            .unwrap_err()
            .contains("new proposal cohort")
    );
    assert_eq!(
        std::fs::read_to_string(existing.join("owner.txt")).unwrap(),
        "retained"
    );
    let mut invalid = parent.clone();
    invalid.forward_only = false;
    let error = evaluate_full_proposal(
        &scorer,
        tmp.path(),
        &parent,
        &invalid,
        0.01,
        &tmp.path().join("topology"),
    )
    .unwrap_err();
    assert!(error.contains("forward"), "{error}");
}
