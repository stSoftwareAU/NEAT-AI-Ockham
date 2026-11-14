use std::cell::RefCell;
use std::collections::BTreeMap;
use std::path::Path;
use std::time::Duration;

use neat_ai_ockham::baseline::fake::ScriptedScorer;
use neat_ai_ockham::corpus::write_bin_file;
use neat_ai_ockham::fixtures::{creature, neuron, synapse};
use neat_ai_ockham::incumbent::load_incumbent;
use neat_ai_ockham::prune::prune_edge_by_key;
use neat_ai_ockham::scorer::{DirectoryScorer, ScoreResult, ScorerError, ScorerMode};
use neat_ai_ockham::stats::{ActivationStats, NeuronStats};
use neat_ai_ockham::{Clock, ManualClock};
use neat_core::{SynapseKey, SynapseType};

use super::allocation::{Arm, Constraints, Plan, quotas, ranked_edges};
use super::manifest::{
    Frozen, Spec, Subject, file_sha256, freeze, pin_inputs, read_json, write_new,
};
use super::measurement::{Metered, Usage};
use super::pilot::specification;
use super::runner::search_one;

#[test]
fn quotas_waterfill_caps_and_round_in_stable_region_order() {
    let capacities = BTreeMap::from([("a".into(), 1), ("b".into(), 4), ("c".into(), 4)]);
    assert_eq!(
        quotas(&capacities, 5).unwrap(),
        BTreeMap::from([("a".into(), 1), ("b".into(), 2), ("c".into(), 2),])
    );
    assert_eq!(quotas(&capacities, 4).unwrap()["b"], 2);
    assert_eq!(quotas(&capacities, 0).unwrap().values().sum::<usize>(), 0);
    assert_eq!(quotas(&capacities, 9).unwrap(), capacities);
    assert_eq!(
        quotas(&capacities, 10).unwrap_err(),
        "target exceeds capacity"
    );
}

fn parent() -> neat_core::CreatureExport {
    creature(
        2,
        1,
        vec![
            neuron("hidden", "a", 0.0, Some("TANH")),
            neuron("hidden", "b", 0.0, Some("TANH")),
            neuron("output", "output-0", 0.0, Some("IDENTITY")),
        ],
        vec![
            synapse("input-0", "a", 0.1),
            synapse("input-1", "a", 0.2),
            synapse("input-0", "b", 1.0),
            synapse("input-1", "b", 2.0),
            synapse("a", "output-0", 1.0),
            synapse("b", "output-0", 1.0),
            synapse("input-0", "output-0", 3.0),
        ],
    )
}

fn stats() -> ActivationStats {
    let mut stats = ActivationStats::empty();
    for uuid in ["input-0", "input-1", "a", "b"] {
        let value = NeuronStats {
            uuid: uuid.into(),
            neuron_index: 0,
            count: 4,
            mean: 0.0,
            variance: 1.0,
            std_dev: 1.0,
            mean_abs: 1.0,
            min: -1.0,
            max: 1.0,
        };
        if uuid.starts_with("input") {
            stats.inputs.push(value);
        } else {
            stats.neurons.push(value);
        }
    }
    stats
}

#[test]
fn allocation_arms_use_the_same_importance_but_different_eligible_regions() {
    let parent = parent();
    let ranked = ranked_edges(&parent, &stats(), 7).unwrap();
    assert_eq!(ranked[0].key.from_uuid, "input-0");
    assert_eq!(ranked[0].key.to_uuid, "a");
    let global = Plan::new(&parent, Arm::Global, 4).unwrap();
    let neuron = Plan::new(&parent, Arm::PerNeuron, 4).unwrap();
    let depth = Plan::new(&parent, Arm::PerDepth, 4).unwrap();
    // Region a has two edges and receives quota two; global may still cut it.
    assert!(global.eligible(&parent, &ranked[0].key));
    assert!(!neuron.eligible(&parent, &ranked[0].key));
    assert!(depth.eligible(&parent, &ranked[0].key));
    assert_eq!(depth.depths["output-0"], 2);
}

#[test]
fn stable_ties_are_reproducible_and_missing_measurements_fail_loudly() {
    let mut parent = parent();
    for edge in &mut parent.synapses {
        edge.weight = 1.0;
    }
    let a = ranked_edges(&parent, &stats(), 42).unwrap();
    let b = ranked_edges(&parent, &stats(), 42).unwrap();
    assert_eq!(
        a.iter().map(|e| &e.key).collect::<Vec<_>>(),
        b.iter().map(|e| &e.key).collect::<Vec<_>>()
    );
    assert_eq!(a.len(), parent.synapses.len());
    let error = ranked_edges(&parent, &ActivationStats::empty(), 42).unwrap_err();
    assert!(error.contains("unmeasured source"), "{error}");
}

#[test]
fn usefulness_refuses_losing_an_input_and_does_not_count_zero_weight_paths() {
    let parent = parent();
    let constraints = Constraints::new(&parent).unwrap();
    constraints.check(&parent).unwrap();
    let mut disconnected = parent.clone();
    disconnected.synapses.retain(|e| e.from_uuid != "input-1");
    assert_eq!(
        constraints.check(&disconnected).unwrap_err(),
        "lost useful input input-1"
    );
    for edge in &mut disconnected.synapses {
        edge.weight = 0.0;
    }
    assert!(
        constraints
            .check(&disconnected)
            .unwrap_err()
            .starts_with("output lacks input path")
    );
}

#[test]
fn cleanup_cascades_count_actual_connections_and_cannot_undercut_a_budget_or_quota() {
    let parent = parent();
    let key = SynapseKey {
        from_uuid: "a".into(),
        to_uuid: "output-0".into(),
        role: SynapseType::Standard,
    };
    let cut = prune_edge_by_key(&parent, &key, Some(0.0), None).unwrap();
    assert_eq!(cut.creature.synapses.len(), 4);
    assert!(!cut.creature.neurons.iter().any(|n| n.uuid == "a"));
    assert_eq!(
        Plan::new(&parent, Arm::Global, 6)
            .unwrap()
            .check(&parent, &cut.creature)
            .unwrap_err(),
        "cleanup would overshoot target"
    );
    Plan::new(&parent, Arm::Global, 4)
        .unwrap()
        .check(&parent, &cut.creature)
        .unwrap();
    Constraints::new(&parent)
        .unwrap()
        .check(&cut.creature)
        .unwrap();
    assert_eq!(
        Plan::new(&parent, Arm::PerNeuron, 4)
            .unwrap()
            .check(&parent, &cut.creature)
            .unwrap_err(),
        "cleanup would undercut quota a"
    );
}

struct Fixture {
    tmp: tempfile::TempDir,
    spec: Spec,
}

impl Fixture {
    fn new() -> Self {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        let parent_path = root.join("parent.json");
        write_new(&parent_path, &parent()).unwrap();
        let rows: Vec<_> = (0..4)
            .map(|i| (vec![i as f32, (i + 1) as f32], vec![0.0]))
            .collect();
        for name in ["training", "calibration", "validation"] {
            std::fs::create_dir(root.join(name)).unwrap();
            let data = if name == "validation" {
                (10..14)
                    .map(|i| (vec![i as f32, (i + 1) as f32], vec![0.0]))
                    .collect::<Vec<_>>()
            } else {
                rows.clone()
            };
            write_bin_file(&root.join(name).join("0.bin"), &data).unwrap();
        }
        let scorer = root.join("scorer");
        std::fs::write(&scorer, b"identity-pinned test scorer").unwrap();
        let subject = Subject {
            id: "subject".into(),
            independent_unit: "unit".into(),
            provenance: "deterministic correctness fixture".into(),
            creature: parent_path,
            training: root.join("training"),
            calibration: root.join("calibration"),
            validation: root.join("validation"),
        };
        let mut spec = specification(vec![subject], &scorer);
        spec.connection_targets = vec![5, 6];
        Self { tmp, spec }
    }

    fn scorer() -> ScriptedScorer {
        ScriptedScorer {
            baseline_score: 0.5,
            candidate_score: Some(0.6),
            record_count: 4,
            ..Default::default()
        }
    }

    fn phases() -> Phases {
        Phases {
            screen: ScriptedScorer {
                record_count: 2,
                ..Self::scorer()
            },
            full: Self::scorer(),
        }
    }
}

#[test]
fn real_runner_proposes_recalculates_and_accepts_only_through_full_promotion() {
    let fixture = Fixture::new();
    let parent = parent();
    let clock = ManualClock::new();
    let source_before = std::fs::read(&fixture.spec.creatures[0].creature).unwrap();
    for arm in Arm::ALL {
        let dir = fixture.tmp.path().join(format!("{arm:?}"));
        let scorer = Fixture::phases();
        let record = search_one(
            &fixture.spec,
            &fixture.spec.creatures[0],
            &parent,
            arm,
            5,
            42,
            &scorer,
            &clock,
            &dir,
        )
        .unwrap();
        assert_eq!(record.error, None, "{arm:?}");
        assert!(record.achieved_connections >= 5);
        let best = load_incumbent(&dir.join("best.json")).unwrap();
        let proposal = load_incumbent(&dir.join("proposal.json")).unwrap();
        assert_eq!(best.creature.synapses.len(), record.accepted_connections);
        assert_eq!(
            proposal.creature.synapses.len(),
            record.achieved_connections
        );
        assert!(record.search.reported_record_evaluations >= 12, "{arm:?}");
        assert_eq!(scorer.full.last_mode.get(), Some(ScorerMode::Full));
        Constraints::new(&parent)
            .unwrap()
            .check(&best.creature)
            .unwrap();
        if arm == Arm::Global {
            assert_eq!(record.achieved_connections, 5);
            assert_eq!(record.accepted_connections, 5);
            assert_eq!(
                record.search.statistics_records, 8,
                "two topology-specific calibration scans"
            );
        }
    }
    assert_eq!(
        std::fs::read(&fixture.spec.creatures[0].creature).unwrap(),
        source_before
    );
    assert_eq!(parent, self::parent());
}

struct Phases {
    screen: ScriptedScorer,
    full: ScriptedScorer,
}

impl DirectoryScorer for Phases {
    fn score_directory(
        &self,
        dir: &Path,
        corpus: &Path,
        mode: ScorerMode,
    ) -> Result<BTreeMap<String, ScoreResult>, ScorerError> {
        if mode.is_authoritative() {
            self.full.score_directory(dir, corpus, mode)
        } else {
            self.screen.score_directory(dir, corpus, mode)
        }
    }
    fn identity(&self) -> String {
        "scripted phase outcomes".into()
    }
}

#[test]
fn a_calibration_win_cannot_accept_a_full_corpus_loss() {
    let fixture = Fixture::new();
    let full = ScriptedScorer {
        candidate_score: Some(0.4),
        ..Fixture::scorer()
    };
    let scorer = Phases {
        screen: Fixture::phases().screen,
        full,
    };
    let dir = fixture.tmp.path().join("trial");
    let record = search_one(
        &fixture.spec,
        &fixture.spec.creatures[0],
        &parent(),
        Arm::Global,
        6,
        7,
        &scorer,
        &ManualClock::new(),
        &dir,
    )
    .unwrap();
    assert_eq!(record.achieved_connections, 6);
    assert_eq!(record.accepted_connections, 7);
    assert_eq!(record.search.scorer_calls, 2);
    assert_eq!(
        load_incumbent(&dir.join("best.json")).unwrap().creature,
        parent()
    );
}

#[test]
fn full_scorer_failure_preserves_the_parent_and_records_the_cause() {
    let fixture = Fixture::new();
    let full = ScriptedScorer {
        fail_with: Some("invalid training record length".into()),
        ..Fixture::scorer()
    };
    let scorer = Phases {
        screen: Fixture::phases().screen,
        full,
    };
    let dir = fixture.tmp.path().join("trial");
    let record = search_one(
        &fixture.spec,
        &fixture.spec.creatures[0],
        &parent(),
        Arm::Global,
        6,
        7,
        &scorer,
        &ManualClock::new(),
        &dir,
    )
    .unwrap();
    assert_eq!(record.accepted_connections, 7);
    assert_eq!(record.search.failed_calls, 1);
    assert!(
        record
            .error
            .unwrap()
            .contains("invalid training record length")
    );
    assert_eq!(
        load_incumbent(&dir.join("best.json")).unwrap().creature,
        parent()
    );
}

#[test]
fn compute_reservations_refuse_before_calling_and_deadlines_cannot_accept_late_scores() {
    let mut fixture = Fixture::new();
    fixture.spec.limits.scorer_record_evaluations = 7;
    let scorer = Fixture::phases();
    let limited = search_one(
        &fixture.spec,
        &fixture.spec.creatures[0],
        &parent(),
        Arm::Global,
        6,
        7,
        &scorer,
        &ManualClock::new(),
        &fixture.tmp.path().join("limited"),
    )
    .unwrap();
    assert_eq!(limited.stop_reason, "compute-limit");
    assert_eq!(limited.search.scorer_calls, 0);
    assert_eq!(limited.accepted_connections, 7);
    assert_eq!(scorer.screen.last_mode.get(), None);

    fixture.spec.limits.scorer_record_evaluations = 1000;
    fixture.spec.limits.search_seconds = 1;
    let clock = ManualClock::new();
    let slow = Phases {
        screen: ScriptedScorer {
            delay_per_creature: Duration::from_secs(1),
            clock: Some(clock.clone()),
            ..Fixture::phases().screen
        },
        full: Fixture::scorer(),
    };
    let late = search_one(
        &fixture.spec,
        &fixture.spec.creatures[0],
        &parent(),
        Arm::Global,
        6,
        7,
        &slow,
        &clock,
        &fixture.tmp.path().join("late"),
    )
    .unwrap();
    assert_eq!(late.search.timeout_exits, 1);
    assert_eq!(late.accepted_connections, 7);
    assert!(
        late.error
            .unwrap()
            .contains("completed after search deadline")
    );
    let allowed = search_one(
        &fixture.spec,
        &fixture.spec.creatures[0],
        &parent(),
        Arm::Global,
        6,
        7,
        &Fixture::phases(),
        &ManualClock::new(),
        &fixture.tmp.path().join("allowed"),
    )
    .unwrap();
    assert_eq!(allowed.accepted_connections, 6);
}

#[test]
fn missing_scorer_entries_and_omitted_counts_are_errors_not_empty_coverage() {
    let fixture = Fixture::new();
    let dir = fixture.tmp.path().join("cohort");
    std::fs::create_dir(&dir).unwrap();
    write_new(&dir.join("baseline.json"), &parent()).unwrap();
    write_new(&dir.join("proposal.json"), &parent()).unwrap();
    let clock = ManualClock::new();
    for (body, expected) in [
        (
            r#"{"baseline":{"score":0.5,"error":0.0,"recordCount":4}}"#,
            "cohort result keys",
        ),
        (
            r#"{"baseline":{"score":0.5,"error":0.0},"proposal":{"score":0.6,"error":0.0}}"#,
            "recordCount",
        ),
    ] {
        let scorer = ScriptedScorer {
            raw_output: Some(body.into()),
            ..Fixture::scorer()
        };
        let usage = RefCell::new(Usage::default());
        let meter = Metered {
            inner: &scorer,
            limits: &fixture.spec.limits,
            usage: &usage,
            clock: &clock,
            started: clock.now(),
            widths: (2, 1),
        };
        let error = meter
            .score_directory(&dir, &fixture.spec.creatures[0].training, ScorerMode::Full)
            .unwrap_err();
        assert!(error.to_string().contains(expected), "{error}");
        assert_eq!(usage.borrow().failed_calls, 1);
    }
    let scorer = Fixture::scorer();
    let usage = RefCell::new(Usage::default());
    let meter = Metered {
        inner: &scorer,
        limits: &fixture.spec.limits,
        usage: &usage,
        clock: &clock,
        started: clock.now(),
        widths: (2, 1),
    };
    assert_eq!(
        meter
            .score_directory(&dir, &fixture.spec.creatures[0].training, ScorerMode::Full)
            .unwrap()["proposal"]
            .score,
        0.6
    );
}

#[test]
fn freeze_checks_full_content_and_refuses_overlap_and_input_or_scorer_drift() {
    let fixture = Fixture::new();
    let spec = fixture.tmp.path().join("spec.json");
    let manifest = fixture.tmp.path().join("manifest.json");
    write_new(&spec, &fixture.spec).unwrap();
    freeze(&spec, &manifest).unwrap();
    let frozen: Frozen = read_json(&manifest).unwrap();
    frozen.verify().unwrap();
    assert!(freeze(&spec, &manifest).unwrap_err().contains("exists"));
    std::fs::write(&fixture.spec.scorer, b"changed scorer").unwrap();
    assert_eq!(
        frozen.verify().unwrap_err(),
        "code, executable or scorer changed after freeze"
    );
    std::fs::write(&fixture.spec.scorer, b"identity-pinned test scorer").unwrap();
    let path = fixture.spec.creatures[0].training.join("0.bin");
    let mut bytes = std::fs::read(&path).unwrap();
    bytes[13] ^= 1;
    std::fs::write(&path, bytes).unwrap();
    assert!(frozen.verify().unwrap_err().contains("subset"));
    let mut overlap = fixture.spec.clone();
    overlap.creatures[0].validation = overlap.creatures[0].training.clone();
    overlap.creatures[0].calibration = overlap.creatures[0].training.clone();
    assert_eq!(
        pin_inputs(&overlap).unwrap_err(),
        "validation inputs overlap full training data"
    );
    assert_ne!(
        file_sha256(&path).unwrap(),
        frozen.inputs[0].training.sha256
    );
}
