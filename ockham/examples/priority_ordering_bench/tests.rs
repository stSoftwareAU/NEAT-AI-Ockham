use super::*;
use manifest::Manifest;
use serde_json::json;

fn manifest() -> Manifest {
    let tasks = json!([
      {"id":"source", "family":"family", "lineage":"l1", "corpusVersion":"v1", "split":"source", "parent":"p1", "parentSha256":"1".repeat(64), "corpus":"c1", "corpusSha256":"4".repeat(64), "corpusIdentity":"c1", "history":{"path":"/private/history.jsonl","sha256":"7".repeat(64),"acquisitionMs":1000}},
      {"id":"cal", "family":"family", "lineage":"l2", "corpusVersion":"v2", "split":"calibration", "parent":"p2", "parentSha256":"2".repeat(64), "corpus":"c2", "corpusSha256":"5".repeat(64), "corpusIdentity":"c2", "history":null},
      {"id":"val", "family":"family", "lineage":"l3", "corpusVersion":"v3", "split":"validation", "parent":"p3", "parentSha256":"3".repeat(64), "corpus":"c3", "corpusSha256":"6".repeat(64), "corpusIdentity":"c3", "history":null}
    ]);
    serde_json::from_value(json!({
        "formatVersion": 1, "protocol": "ockham-ranking-v1", "codeRevision": "a".repeat(40),
        "scorer": "/private/rust_scorer", "scorerSha256": "b".repeat(64), "scorerArgs": ["--gpu", "off"],
        "historicalCutoffUnix": 100, "seeds": [7,19], "pilotVisits": 4,
        "auditVisitsCeiling": 40, "minimumCompletePool": 20, "minimumIndependentGroups": 3,
        "maxTaskMs": 60000, "maxRunMs": 600000, "scorerTimeoutSecs": 10,
        "statsRecords": 128, "maxCorpusBytes": 1000000,
        "evaluationBudgets": [1,5], "calibrationBudgets": [0,4], "historyFractions": [0.0,0.1,1.0],
        "fit": {"epochs": 20, "learningRate": 0.1, "l2": 0.001, "maxPairs": 100},
        "bootstrapReplicates": 200, "minimumRecallGain": 0.05, "maximumFalseRejection": 0.1,
        "maximumCostRatio": 1.0, "minimumImprovement": 0.000001,
        "tasks": tasks
    })).unwrap()
}

#[test]
fn split_leakage_checks_lineages_versions_and_content_not_only_names() {
    let base = manifest();
    base.validate().unwrap();
    for field in ["lineage", "corpusVersion", "parentSha256", "corpusSha256"] {
        let mut value = serde_json::to_value(&base).unwrap();
        value["tasks"][2][field] = value["tasks"][0][field].clone();
        let changed: Manifest = serde_json::from_value(value).unwrap();
        assert!(
            changed.validate().unwrap_err().contains("split leakage"),
            "{field}"
        );
    }
}

#[test]
fn metrics_count_ties_small_pools_and_improvements_outside_budget() {
    let scores = [1.0, 1.0, -1.0, 0.1];
    let result = evaluation::metrics(&scores, &[2, 3, 0, 1], 2, 0.0).unwrap();
    assert_eq!(
        result.top_count, 2,
        "include the entire tie at the ceil(5%) boundary"
    );
    assert_eq!(result.top_recall, 0.0);
    assert_eq!(result.false_rejection, Some(2.0 / 3.0));
    let all = evaluation::metrics(&scores, &[0, 1, 2, 3], 99, 0.0).unwrap();
    assert_eq!(all.top_recall, 1.0);
    assert_eq!(all.false_rejection, Some(0.0));
    assert!(
        evaluation::metrics(&[], &[], 1, 0.0)
            .unwrap_err()
            .contains("empty")
    );
    assert!(
        evaluation::metrics(&[f64::NAN], &[0], 1, 0.0)
            .unwrap_err()
            .contains("finite")
    );
    assert!(
        evaluation::metrics(&scores, &[0, 0, 2, 3], 1, 0.0)
            .unwrap_err()
            .contains("permutation")
    );
    assert_eq!(
        evaluation::metrics(&[-1.0], &[0], 1, 0.0)
            .unwrap()
            .false_rejection,
        None
    );
}

#[test]
fn seeded_selection_is_reproducible_and_independent_of_input_order() {
    let items = vec!["a".to_string(), "b".into(), "c".into(), "d".into()];
    let a = manifest::seeded_order(&items, 7);
    let mut reversed = items.clone();
    reversed.reverse();
    assert_eq!(a, manifest::seeded_order(&reversed, 7));
    let orders: std::collections::BTreeSet<_> =
        (0..10).map(|s| manifest::seeded_order(&items, s)).collect();
    assert!(orders.len() > 1);
    assert_eq!(a.len(), 4);
}

#[test]
fn missing_scores_are_unlabelled_and_cold_calibration_is_empty() {
    let rows = vec![audit::AuditRow {
        visit: "a".into(),
        kind: Some("ablation".into()),
        sample_delta: Some(0.2),
        full_delta: None,
        features: None,
        telemetry: None,
        failure: Some("scorer failed".into()),
        candidate_sha256: None,
        sample_ms: 1,
        full_ms: 1,
    }];
    assert_eq!(evaluation::labelled(&rows).unwrap().len(), 0);
    let mut scored = rows.clone();
    scored[0].full_delta = Some(0.3);
    assert_eq!(evaluation::labelled(&scored).unwrap()[0].score, 0.3);
    let mut invalid = scored.clone();
    invalid[0].features = Some(neat_ai_ockham::CandidateFeatures {
        variance: f64::NAN,
        ..Default::default()
    });
    assert!(
        evaluation::labelled(&invalid)
            .unwrap_err()
            .contains("non-finite")
    );
    scored[0].sample_delta = None;
    assert!(
        evaluation::labelled(&scored)
            .unwrap_err()
            .contains("partial")
    );
}

#[test]
fn audit_scores_sampled_losers_individually_and_preserves_parent() {
    use neat_ai_ockham::scorer::{DirectoryScorer, ScoreResult, ScorerError, ScorerMode};
    use std::collections::BTreeMap;
    use std::path::Path;
    struct Judge {
        omit_candidate: bool,
        fail: bool,
    }
    impl DirectoryScorer for Judge {
        fn identity(&self) -> String {
            "test-judge".into()
        }
        fn score_directory(
            &self,
            dir: &Path,
            _: &Path,
            mode: ScorerMode,
        ) -> Result<BTreeMap<String, ScoreResult>, ScorerError> {
            assert!(dir.join("baseline.json").is_file());
            assert!(dir.join("candidate.json").is_file());
            if self.fail {
                return Err(ScorerError::Spawn("deliberate missing binary".into()));
            }
            let count = if mode.is_authoritative() { 32 } else { 2 };
            let result = |score| ScoreResult {
                score,
                error: 1.0 - score,
                complexity_penalty: 0.0,
                record_count: count,
                sample_rate: None,
                gpu_backend: None,
                cost_name: None,
                time_taken: 0.0,
            };
            let mut map = BTreeMap::from([("baseline".into(), result(0.5))]);
            if !self.omit_candidate {
                map.insert(
                    "candidate".into(),
                    result(if mode.is_authoritative() { 0.6 } else { 0.4 }),
                );
            }
            Ok(map)
        }
    }
    let tmp = tempfile::tempdir().unwrap();
    let train = tmp.path().join("train");
    std::fs::create_dir(&train).unwrap();
    neat_ai_ockham::corpus::write_bin_file(
        &train.join("0.bin"),
        &(0..32)
            .map(|i| (vec![i as f32 / 32.0], vec![0.0]))
            .collect::<Vec<_>>(),
    )
    .unwrap();
    let creature = neat_ai_ockham::fixtures::shortcut_edge_creature();
    let text = neat_core::creature_to_json(&creature).unwrap();
    let parent = tmp.path().join("parent.json");
    std::fs::write(&parent, &text).unwrap();
    let mut m = manifest();
    m.tasks[1].parent = parent.clone();
    m.tasks[1].corpus = train;
    m.tasks[1].parent_sha256 = neat_ai_ockham::incumbent::sha256_hex(text.as_bytes());
    m.tasks[1].corpus_sha256 =
        audit::corpus_digest(&m.tasks[1].corpus, m.max_corpus_bytes).unwrap();
    m.tasks[1].corpus_identity = neat_ai_ockham::corpus::corpus_info(
        &m.tasks[1].corpus,
        &neat_core::training_data::TrainingDataConfig::new(1, 1),
    )
    .unwrap()
    .identity;
    let pool = audit::collect_pool(
        &neat_ai_ockham::SystemClock,
        &m,
        &m.tasks[1],
        7,
        40,
        &Judge {
            omit_candidate: false,
            fail: false,
        },
        &tmp.path().join("ok"),
    )
    .unwrap();
    assert_eq!(pool.rows.len(), pool.selected.len());
    assert!(
        pool.rows
            .iter()
            .any(|r| r.kind.as_deref() == Some("synapse"))
    );
    let scored: Vec<_> = pool.rows.iter().filter(|r| r.kind.is_some()).collect();
    assert!(!scored.is_empty());
    for row in scored {
        assert!(row.sample_delta.unwrap() < 0.0);
        assert!(row.full_delta.unwrap() > 0.0);
        assert_eq!(row.failure, None);
    }
    assert_eq!(std::fs::read_to_string(&parent).unwrap(), text);
    let records: Vec<_> = pool
        .rows
        .iter()
        .filter_map(|row| row.telemetry.clone())
        .collect();
    assert!(!records.is_empty());
    assert!(
        records
            .iter()
            .all(|row| row.corpus_identity == m.tasks[1].corpus_identity)
    );
    let history_path = tmp.path().join("audit-history.jsonl");
    neat_ai_ockham::telemetry::append(&history_path, &records).unwrap();
    let mut source = m.tasks[1].clone();
    source.split = manifest::Split::Source;
    source.history = Some(manifest::HistorySpec {
        path: history_path.clone(),
        sha256: hash_file(&history_path).unwrap(),
        acquisition_ms: 1,
    });
    let mut later = m.clone();
    later.historical_cutoff_unix = records.iter().map(|row| row.unix_secs).max().unwrap() + 1;
    let history = audit::load_history(&later, &source).unwrap();
    assert_eq!(history.records.len(), records.len());
    for row in &history.records {
        assert_eq!(row.corpus_identity, source.corpus_identity);
        assert_eq!(row.full_delta, Some(0.6 - 0.5));
    }
    let bad = audit::collect_pool(
        &neat_ai_ockham::SystemClock,
        &m,
        &m.tasks[1],
        7,
        40,
        &Judge {
            omit_candidate: true,
            fail: false,
        },
        &tmp.path().join("missing"),
    )
    .unwrap();
    for row in bad.rows.iter().filter(|r| r.kind.is_some()) {
        assert!(row.failure.as_ref().unwrap().contains("missing candidate"));
        assert_eq!(row.full_delta, None);
    }
    let bad = audit::collect_pool(
        &neat_ai_ockham::SystemClock,
        &m,
        &m.tasks[1],
        7,
        40,
        &Judge {
            omit_candidate: false,
            fail: true,
        },
        &tmp.path().join("failed"),
    )
    .unwrap();
    assert!(bad.rows.iter().filter(|r| r.kind.is_some()).all(|r| {
        r.failure
            .as_ref()
            .unwrap()
            .contains("deliberate missing binary")
    }));
    struct AlteredJudge {
        inner: Judge,
        clock: Option<neat_ai_ockham::ManualClock>,
        non_finite: bool,
    }
    impl DirectoryScorer for AlteredJudge {
        fn identity(&self) -> String {
            self.inner.identity()
        }
        fn score_directory(
            &self,
            dir: &Path,
            corpus: &Path,
            mode: ScorerMode,
        ) -> Result<BTreeMap<String, ScoreResult>, ScorerError> {
            let mut result = self.inner.score_directory(dir, corpus, mode)?;
            if let Some(clock) = &self.clock {
                if !mode.is_authoritative() {
                    clock.advance(std::time::Duration::from_millis(60_000));
                }
            } else if mode.is_authoritative() {
                if self.non_finite {
                    result.get_mut("candidate").unwrap().score = f64::NAN;
                } else {
                    result.get_mut("candidate").unwrap().record_count = 31;
                }
            }
            Ok(result)
        }
    }
    for non_finite in [false, true] {
        let altered = AlteredJudge {
            inner: Judge {
                omit_candidate: false,
                fail: false,
            },
            clock: None,
            non_finite,
        };
        let bad = audit::collect_pool(
            &neat_ai_ockham::SystemClock,
            &m,
            &m.tasks[1],
            7,
            40,
            &altered,
            &tmp.path().join(format!("invalid-{non_finite}")),
        )
        .unwrap();
        let scored: Vec<_> = bad.rows.iter().filter(|r| r.kind.is_some()).collect();
        assert!(!scored.is_empty());
        for row in scored {
            assert!(row.sample_delta.is_some());
            assert_eq!(row.full_delta, None);
            assert!(row.failure.as_ref().unwrap().contains(if non_finite {
                "non-finite"
            } else {
                "incomplete scoring"
            }));
        }
    }
    let clock = neat_ai_ockham::ManualClock::new();
    let spending = AlteredJudge {
        inner: Judge {
            omit_candidate: false,
            fail: false,
        },
        clock: Some(clock.clone()),
        non_finite: false,
    };
    let limited = audit::collect_pool(
        &clock,
        &m,
        &m.tasks[1],
        7,
        40,
        &spending,
        &tmp.path().join("limited"),
    )
    .unwrap();
    assert!(limited.rows.iter().any(|r| r.sample_delta.is_some()));
    assert!(
        limited.rows.iter().filter(|r| r.kind.is_some()).all(
            |r| r.full_delta.is_none() && r.failure.as_ref().unwrap().contains("compute limit")
        )
    );
    let mut changed = m.clone();
    changed.tasks[1].parent_sha256 = "f".repeat(64);
    assert!(
        audit::collect_pool(
            &neat_ai_ockham::SystemClock,
            &changed,
            &changed.tasks[1],
            7,
            40,
            &Judge {
                omit_candidate: false,
                fail: false
            },
            &tmp.path().join("changed")
        )
        .unwrap_err()
        .contains("parent digest")
    );
}

fn artificial_pool(task: &manifest::Task, seed: u64) -> audit::Pool {
    let ids: Vec<_> = (0..40).map(|i| format!("h{i}")).collect();
    let selected = manifest::seeded_order(&ids, seed);
    let rows = selected
        .iter()
        .map(|id| {
            let i = id[1..].parse::<usize>().unwrap();
            let features = neat_ai_ockham::CandidateFeatures {
                measured: true,
                mean_abs: i as f64 / 40.0,
                outgoing_weight: 1.0,
                cascade_growth_units: 1.0,
                ..Default::default()
            };
            audit::AuditRow {
                visit: id.clone(),
                kind: Some("ablation".into()),
                candidate_sha256: Some("a".repeat(64)),
                features: Some(features),
                sample_delta: Some(i as f64 / 40.0),
                full_delta: Some(i as f64 / 40.0 - 0.5),
                sample_ms: 2,
                full_ms: 10,
                telemetry: None,
                failure: None,
            }
        })
        .collect();
    audit::Pool {
        task: task.id.clone(),
        seed,
        population: 40,
        selected,
        rows,
        preparation_ms: 1,
        features_ms: 1,
        generation_ms: 1,
        elapsed_ms: 483,
    }
}

fn artificial_audit(m: &Manifest) -> audit::Audit {
    let task = &m.tasks[0];
    let stamp = neat_ai_ockham::RunStamp {
        host: "test".into(),
        corpus_identity: task.corpus_identity.clone(),
        creature_checksum: task.parent_sha256.clone(),
        ordering: "random".into(),
        seed: 7,
    };
    let records = (0..40)
        .map(|i| {
            let f = neat_ai_ockham::CandidateFeatures {
                measured: true,
                mean_abs: i as f64 / 40.0,
                outgoing_weight: 1.0,
                cascade_growth_units: 1.0,
                ..Default::default()
            };
            let mut row = neat_ai_ockham::CandidateRecord::new(
                &stamp,
                &format!("h{i}"),
                "ablation",
                &f,
                neat_ai_ockham::CandidateOutcome::Rejected,
            );
            row.unix_secs = 90;
            row.sample_delta = Some(i as f64 / 40.0);
            row.full_delta = Some(0.5 - i as f64 / 40.0);
            row
        })
        .collect();
    audit::Audit {
        format_version: 1,
        manifest_sha256: m.identity().unwrap(),
        freeze_sha256: Some("b".repeat(64)),
        executable_sha256: hash_file(&std::env::current_exe().unwrap()).unwrap(),
        phase: "audit".into(),
        protocol_preparation_ms: 1,
        pilot_cost_ms: 1,
        freezing_cost_ms: 1,
        pools: m
            .tasks
            .iter()
            .flat_map(|t| m.seeds.iter().map(|s| artificial_pool(t, *s)))
            .collect(),
        histories: vec![audit::History {
            task: task.id.clone(),
            records,
            unlabelled: 2,
            missing_partial: 0,
            future: 1,
            acquisition_ms: 400,
            loading_ms: 1,
        }],
        failures: vec![],
        elapsed_ms: 3000,
    }
}

#[test]
fn comparison_reports_cold_start_negative_transfer_and_validation_is_not_fitted() {
    let mut m = manifest();
    // Seed 43 gives the random cold start a true top candidate in this pool.
    m.seeds = vec![7, 43];
    let a = artificial_audit(&m);
    let output =
        serde_json::to_value(evaluation::evaluate(&m, &a, "c".repeat(64), 0).unwrap()).unwrap();
    assert_eq!(
        output["verdict"], "inconclusive",
        "one lineage group cannot establish a benefit"
    );
    let trials = output["trials"].as_array().unwrap();
    let trial = |strategy: &str| {
        trials
            .iter()
            .find(|t| {
                t["strategy"] == strategy
                    && t["seed"] == 7
                    && t["calibrationBudget"] == 0
                    && t["historyFraction"] == 1.0
                    && t["evaluationBudget"] == 5
            })
            .unwrap()
    };
    assert_eq!(
        trial("target-pairwise")["metrics"],
        trial("random")["metrics"]
    );
    assert_eq!(trial("target-pairwise")["calibrationObserved"], 0);
    assert_eq!(trial("transfer-pairwise")["historicalScoringMs"], 401);
    assert_eq!(trial("target-pairwise")["historicalScoringMs"], 0);
    assert_eq!(trial("transfer-pairwise")["preparationMs"], 5);
    assert!(
        trial("transfer-pairwise")["metrics"]["topRecall"]
            .as_f64()
            .unwrap()
            < trial("partial")["metrics"]["topRecall"].as_f64().unwrap()
    );
    assert!(output["paired"].as_array().unwrap().iter().any(|p| {
        p["strategy"] == "transfer-pairwise"
            && p["control"] == "target-pairwise"
            && p["negativeTransferTasks"]
                .as_array()
                .is_some_and(|v| !v.is_empty())
    }));
    for cal in [0, 4] {
        let empty_history = |strategy: &str| {
            trials
                .iter()
                .find(|t| {
                    t["strategy"] == strategy
                        && t["seed"] == 7
                        && t["calibrationBudget"] == cal
                        && t["historyFraction"] == 0.0
                        && t["evaluationBudget"] == 5
                })
                .unwrap()
        };
        assert_eq!(
            empty_history("target-pairwise")["modelSha256"],
            empty_history("transfer-pairwise")["modelSha256"]
        );
    }
    let mut changed = a.clone();
    for pool in changed.pools.iter_mut().filter(|p| p.task == "val") {
        for row in &mut pool.rows {
            row.full_delta = row.full_delta.map(|v| -v);
        }
    }
    let altered =
        serde_json::to_value(evaluation::evaluate(&m, &changed, "d".repeat(64), 0).unwrap())
            .unwrap();
    let signature = |value: &serde_json::Value| {
        value["trials"]
            .as_array()
            .unwrap()
            .iter()
            .map(|t| t["modelSha256"].clone())
            .collect::<Vec<_>>()
    };
    assert_eq!(
        signature(&output),
        signature(&altered),
        "validation labels cannot change a fitted model"
    );
    assert_ne!(
        output["trials"][1]["metrics"],
        altered["trials"][1]["metrics"]
    );
    let repeated =
        serde_json::to_value(evaluation::evaluate(&m, &a, "c".repeat(64), 0).unwrap()).unwrap();
    assert_eq!(signature(&output), signature(&repeated));
    assert_eq!(output["paired"], repeated["paired"]);
}

#[test]
fn freeze_uses_only_complete_calibration_and_rejects_changed_manifest() {
    let m = manifest();
    let mut pilot = artificial_audit(&m);
    pilot.phase = "pilot".into();
    pilot.freeze_sha256 = None;
    pilot.histories.clear();
    pilot.pools.retain(|p| p.task == "cal");
    for pool in &mut pilot.pools {
        pool.selected.truncate(m.pilot_visits);
        pool.rows.truncate(m.pilot_visits);
        pool.elapsed_ms = 10;
    }
    let frozen = freeze(&m, &pilot, "d".repeat(64)).unwrap();
    assert_eq!(frozen.audit_visits, 40);
    check_freeze(&m, &frozen).unwrap();
    let mut changed = m.clone();
    changed.seeds.push(23);
    assert!(
        check_freeze(&changed, &frozen)
            .unwrap_err()
            .contains("mismatch")
    );
    pilot.pools[0].rows[0].full_delta = None;
    assert!(
        freeze(&m, &pilot, "d".repeat(64))
            .unwrap_err()
            .contains("complete")
    );
    pilot.pools[0] = artificial_pool(&m.tasks[2], 7);
    assert!(
        freeze(&m, &pilot, "d".repeat(64))
            .unwrap_err()
            .contains("forbidden")
    );
}

#[test]
fn historical_loader_never_labels_unscored_or_future_records() {
    let mut m = manifest();
    let mut a = artificial_audit(&m);
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("history.jsonl");
    let mut missing = a.histories[0].records[0].clone();
    missing.uuid = "missing".into();
    missing.full_delta = None;
    let mut future = missing.clone();
    future.uuid = "future".into();
    future.full_delta = Some(100.0);
    future.unix_secs = 100;
    a.histories[0].records.extend([missing, future]);
    neat_ai_ockham::telemetry::append(&path, &a.histories[0].records).unwrap();
    let spec = m.tasks[0].history.as_mut().unwrap();
    spec.path = path.clone();
    spec.sha256 = hash_file(&path).unwrap();
    let history = audit::load_history(&m, &m.tasks[0]).unwrap();
    assert_eq!(history.records.len(), 40);
    assert_eq!(history.unlabelled, 1);
    assert_eq!(history.future, 1);
    let mut wrong = a.histories[0].records[0].clone();
    wrong.creature_checksum = "x".into();
    wrong.uuid = "wrong".into();
    neat_ai_ockham::telemetry::append(&path, &[wrong]).unwrap();
    m.tasks[0].history.as_mut().unwrap().sha256 = hash_file(&path).unwrap();
    assert!(
        audit::load_history(&m, &m.tasks[0])
            .unwrap_err()
            .contains("provenance")
    );
}

#[test]
fn historical_loader_normalises_legacy_audit_identity_and_rejects_other_corpora() {
    let mut m = manifest();
    let mut row = artificial_audit(&m).histories.remove(0).records.remove(0);
    let tmp = tempfile::tempdir().unwrap();
    for (index, identity, ordering, accepted) in [
        (0, m.tasks[0].corpus_identity.clone(), "random", true),
        (1, m.tasks[0].corpus_sha256.clone(), "random-audit", true),
        (2, "unrelated-corpus".into(), "random-audit", false),
        (3, m.tasks[0].corpus_sha256.clone(), "random", false),
    ] {
        row.corpus_identity = identity;
        row.ordering = ordering.into();
        let path = tmp.path().join(format!("history-{index}.jsonl"));
        neat_ai_ockham::telemetry::append(&path, std::slice::from_ref(&row)).unwrap();
        let spec = m.tasks[0].history.as_mut().unwrap();
        spec.path = path.clone();
        spec.sha256 = hash_file(&path).unwrap();
        let result = audit::load_history(&m, &m.tasks[0]);
        if accepted {
            let history = result.unwrap();
            assert_eq!(history.records.len(), 1);
            assert_eq!(
                history.records[0].corpus_identity,
                m.tasks[0].corpus_identity
            );
            assert_eq!(history.records[0].full_delta, row.full_delta);
            let mut a = artificial_audit(&m);
            a.histories = vec![history];
            let output =
                serde_json::to_value(evaluation::evaluate(&m, &a, "c".repeat(64), 0).unwrap())
                    .unwrap();
            assert!(
                output["trials"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .any(|t| t["strategy"] == "transfer-pairwise" && t["sourceObserved"] == 1)
            );
        } else {
            assert_eq!(
                result.unwrap_err(),
                "historical parent/corpus provenance mismatch"
            );
        }
    }
}

#[test]
fn confidence_intervals_resample_independent_groups() {
    assert_eq!(
        evaluation::cluster_interval(&[vec![1.0], vec![2.0]], 200),
        None
    );
    assert_eq!(
        evaluation::cluster_interval(&[vec![0.25, 0.25], vec![0.25], vec![0.25]], 200),
        Some([0.25, 0.25])
    );
    let values = vec![vec![-1.0; 20], vec![0.0], vec![1.0]];
    let result = evaluation::cluster_interval(&values, 200).unwrap();
    assert!(result[0] < 0.0 && result[1] > 0.0);
    assert_eq!(evaluation::cluster_interval(&values, 200), Some(result));
}

#[test]
fn uncertainty_keeps_related_families_and_lineages_together_transitively() {
    let m = manifest();
    let mut tasks = Vec::new();
    for i in 0..4 {
        let mut t = m.tasks[2].clone();
        t.id = format!("task-{i}");
        t.family = format!("family-{i}");
        t.lineage = format!("lineage-{i}");
        t.corpus_version = format!("version-{i}");
        t.parent_sha256 = format!("{i:064x}");
        t.corpus_sha256 = format!("{:064x}", i + 100);
        tasks.push(t);
    }
    tasks[1].family = tasks[0].family.clone();
    tasks[2].lineage = tasks[1].lineage.clone();
    let groups = evaluation::groups(&tasks.iter().collect::<Vec<_>>());
    assert_eq!(
        groups,
        vec![
            vec!["task-2".to_string(), "task-1".into(), "task-0".into()],
            vec!["task-3".into()]
        ]
    );
}

#[test]
fn comparison_charges_incomplete_calibration_and_unitemised_preparation() {
    let m = manifest();
    let mut a = artificial_audit(&m);
    let cal = a
        .pools
        .iter_mut()
        .find(|p| p.task == "cal" && p.seed == 7)
        .unwrap();
    cal.rows[0].full_delta = None;
    cal.rows[0].failure = Some("full scorer failed after spending time".into());
    // A failed pool is unusable supervision, but its acquisition is not free.
    cal.elapsed_ms = 6_000;
    let validation = a
        .pools
        .iter_mut()
        .find(|p| p.task == "val" && p.seed == 7)
        .unwrap();
    // Includes candidate serialisation, directory creation and telemetry work
    // outside the named feature/scoring stages.
    validation.elapsed_ms = 5_000;
    let output =
        serde_json::to_value(evaluation::evaluate(&m, &a, "c".repeat(64), 0).unwrap()).unwrap();
    let trials = output["trials"].as_array().unwrap();
    let trial = |strategy: &str, cal: usize| {
        trials
            .iter()
            .find(|t| {
                t["strategy"] == strategy
                    && t["seed"] == 7
                    && t["calibrationBudget"] == cal
                    && t["historyFraction"] == 0.0
                    && t["evaluationBudget"] == 5
            })
            .unwrap()
    };
    let target = trial("target-pairwise", 4);
    assert_eq!(
        target["calibrationScoringMs"], 6_000,
        "a failed calibration pool retains its entire measured acquisition cost"
    );
    assert_eq!(target["calibrationObserved"], 0);
    assert_eq!(
        target["modelSha256"],
        trial("target-pairwise", 0)["modelSha256"],
        "incomplete calibration labels must not enter the fit"
    );
    assert_eq!(
        target["preparationMs"], 4_522,
        "charge residual pool work plus protocol, pilot and freeze costs"
    );
    assert_eq!(trial("random", 4)["calibrationScoringMs"], 0);
    assert_eq!(trial("random", 4)["preparationMs"], 4_522);
    assert_eq!(output["verdict"], "inconclusive");
}
