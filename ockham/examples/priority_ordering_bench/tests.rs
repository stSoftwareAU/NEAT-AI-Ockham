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
        "formatVersion": 1, "protocol": "ockham-ranking-v2", "codeRevision": "a".repeat(40),
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
        path_sensitivity: None,
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
    // Synapse candidates carry a vector too, not a missing-feature tie (#261).
    for row in pool.rows.iter().filter(|r| r.kind.is_some()) {
        let features = row
            .features
            .unwrap_or_else(|| panic!("{:?} has no features", row.visit));
        if row.kind.as_deref() == Some("synapse") {
            assert_eq!(
                (features.fan_in, features.fan_out),
                (0, 1),
                "{:?}",
                row.visit
            );
        }
    }
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
                path_sensitivity: None,
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
        format_version: 2,
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
        serde_json::to_value(evaluation::evaluate(&m, &a, "c".repeat(64), 0, 0).unwrap()).unwrap();
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
        serde_json::to_value(evaluation::evaluate(&m, &changed, "d".repeat(64), 0, 0).unwrap())
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
        serde_json::to_value(evaluation::evaluate(&m, &a, "c".repeat(64), 0, 0).unwrap()).unwrap();
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

#[cfg(unix)]
#[test]
fn history_stream_longer_than_its_reported_length_is_cut_at_the_cap() {
    // /dev/zero reports length 0 but never ends, so only a bounded read returns.
    let mut m = manifest();
    m.tasks[0].history.as_mut().unwrap().path = "/dev/zero".into();
    let err = audit::load_history(&m, &m.tasks[0]).unwrap_err();
    assert!(err.contains("exceeds 67108864 bytes"), "{err}");
}

#[test]
fn history_one_byte_over_the_cap_is_refused_and_at_the_cap_is_accepted() {
    let mut m = manifest();
    let a = artificial_audit(&m);
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("history.jsonl");
    neat_ai_ockham::telemetry::append(&path, &a.histories[0].records).unwrap();
    let spec = m.tasks[0].history.as_mut().unwrap();
    spec.path = path.clone();
    spec.sha256 = hash_file(&path).unwrap();
    let len = std::fs::metadata(&path).unwrap().len();
    let err = audit::load_history_capped(&m, &m.tasks[0], len - 1).unwrap_err();
    assert!(err.contains(&format!("exceeds {} bytes", len - 1)), "{err}");
    let history = audit::load_history_capped(&m, &m.tasks[0], len).unwrap();
    assert_eq!(history.records.len(), 40);
    m.tasks[0].history.as_mut().unwrap().sha256 = "0".repeat(64);
    let err = audit::load_history_capped(&m, &m.tasks[0], len).unwrap_err();
    assert!(err.contains("digest mismatch"), "{err}");
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
                serde_json::to_value(evaluation::evaluate(&m, &a, "c".repeat(64), 0, 0).unwrap())
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
        serde_json::to_value(evaluation::evaluate(&m, &a, "c".repeat(64), 0, 0).unwrap()).unwrap();
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

#[test]
fn comparison_reports_audit_time_outside_every_pool_ledger() {
    let m = manifest();
    let mut a = artificial_audit(&m);
    // Stage sums are 483 ms (1 + 1 + 1 + 40 * 2 + 40 * 10), so five pools are
    // charged their full 100_000 ms.
    for pool in a.pools.iter_mut() {
        pool.elapsed_ms = 100_000;
    }
    // The first pool's raw elapsed time is below its stage sum, so it is charged
    // the 483 ms stage sum. Raw-elapsed subtraction would therefore give 2_983.
    a.pools[0].elapsed_ms = 0;
    // An early collection failure produces no pool, so it has no ledger.
    a.failures.push("val seed 9: maxRunMs exhausted".into());
    // Protocol setup and history loading sit inside the audit wall time but
    // trial ledgers already charge them, so they must not be reported as
    // unpooled. Distinct large values make each subtraction observable.
    a.protocol_preparation_ms = 700;
    a.histories[0].loading_ms = 3_000;
    // Five pools at 100_000 ms, one charged 483 ms, setup 700 ms, history
    // loading 3_000 ms, plus 2_500 ms unpooled.
    a.elapsed_ms = 502_983 + 700 + 3_000;
    // A distinct preflight time makes its place in the total observable.
    let output =
        serde_json::to_value(evaluation::evaluate(&m, &a, "c".repeat(64), 9_000, 0).unwrap())
            .unwrap();
    assert_eq!(output["unpooledAuditMs"], 2_500);
    assert_eq!(output["auditCostMs"], 506_683);
    assert_eq!(
        output["preparationCostMs"], 0,
        "a manifest without preparationMs records no preparation charge"
    );
    assert_eq!(output["preflightCostMs"], 9_000);
    assert_eq!(
        output["totalResearchMs"].as_u64().unwrap() - output["evaluationCostMs"].as_u64().unwrap(),
        output["auditCostMs"].as_u64().unwrap()
            + output["pilotCostMs"].as_u64().unwrap()
            + output["freezingCostMs"].as_u64().unwrap()
            + output["historicalAcquisitionMs"].as_u64().unwrap()
            + output["preparationCostMs"].as_u64().unwrap()
            + output["preflightCostMs"].as_u64().unwrap(),
        "unpooled audit time is already inside auditCostMs and must not be added twice"
    );
    assert!(
        output["reasons"]
            .as_array()
            .unwrap()
            .iter()
            .any(|r| r == "val seed 9: maxRunMs exhausted")
    );

    // Measured audit time below the pool sum must saturate at zero, not wrap.
    a.elapsed_ms = 1_000;
    let output =
        serde_json::to_value(evaluation::evaluate(&m, &a, "c".repeat(64), 0, 0).unwrap()).unwrap();
    assert_eq!(output["unpooledAuditMs"], 0);
}

#[test]
fn manifest_preparation_time_is_reported_and_added_to_the_research_total() {
    let cost = |m: &Manifest| {
        let output = serde_json::to_value(
            evaluation::evaluate(m, &artificial_audit(m), "c".repeat(64), 0, 0).unwrap(),
        )
        .unwrap();
        let get = |key: &str| output[key].as_u64().unwrap();
        let others = get("auditCostMs")
            + get("pilotCostMs")
            + get("freezingCostMs")
            + get("historicalAcquisitionMs")
            + get("preflightCostMs")
            + get("evaluationCostMs");
        (get("preparationCostMs"), get("totalResearchMs"), others)
    };

    let absent = manifest();
    assert_eq!(absent.preparation_ms, 0, "the field is optional");
    let (preparation, total, others) = cost(&absent);
    assert_eq!(preparation, 0);
    assert_eq!(total, others);
    assert!(
        serde_json::to_value(&absent)
            .unwrap()
            .get("preparationMs")
            .is_none(),
        "a zero preparation time keeps the identity of manifests without the field"
    );

    let mut value = serde_json::to_value(manifest()).unwrap();
    value["preparationMs"] = json!(1_234_567);
    let present: Manifest = serde_json::from_value(value).unwrap();
    present.validate().unwrap();
    let (preparation, total, others) = cost(&present);
    assert_eq!(preparation, 1_234_567);
    assert!(total >= 1_234_567, "total {total} must include preparation");
    assert_eq!(total, others + 1_234_567);
}

#[test]
fn manifest_build_time_is_reported_and_added_to_the_research_total() {
    let net = |m: &Manifest| {
        let output = serde_json::to_value(
            evaluation::evaluate(m, &artificial_audit(m), "c".repeat(64), 0, 0).unwrap(),
        )
        .unwrap();
        let get = |key: &str| output[key].as_u64().unwrap();
        (
            get("buildCostMs"),
            get("totalResearchMs") - get("evaluationCostMs"),
        )
    };

    let absent = manifest();
    assert_eq!(absent.build_ms, 0, "the field is optional");
    assert!(
        serde_json::to_value(&absent)
            .unwrap()
            .get("buildMs")
            .is_none(),
        "a zero build time keeps the identity of manifests without the field"
    );
    let (build, base) = net(&absent);
    assert_eq!(build, 0);

    let mut value = serde_json::to_value(manifest()).unwrap();
    value["buildMs"] = json!(87_654);
    let present: Manifest = serde_json::from_value(value).unwrap();
    present.validate().unwrap();
    let (build, total) = net(&present);
    assert_eq!(build, 87_654);
    assert_eq!(
        total,
        base + 87_654,
        "the build charge is added exactly once"
    );
}

#[test]
fn result_serialisation_is_charged_to_the_research_total() {
    let m = manifest();
    let report = evaluation::evaluate(&m, &artificial_audit(&m), "c".repeat(64), 0, 0).unwrap();
    let before = serde_json::to_value(&report).unwrap();
    assert_eq!(
        before["serialisationCostMs"], 0,
        "evaluate leaves it uncharged"
    );

    let written: serde_json::Value =
        serde_json::from_slice(&evaluation::serialise_charged(report).unwrap()).unwrap();
    let charge = written["serialisationCostMs"].as_u64().unwrap();
    assert!(charge >= 1, "measured work is rounded up to at least 1 ms");
    assert_eq!(
        written["totalResearchMs"].as_u64().unwrap(),
        before["totalResearchMs"].as_u64().unwrap() + charge,
        "the serialisation charge is added to the total exactly once"
    );
    let mut rest = written.clone();
    rest["serialisationCostMs"] = json!(0);
    rest["totalResearchMs"] = before["totalResearchMs"].clone();
    assert_eq!(rest, before, "only the two cost fields change");
}

#[test]
fn preflight_check_time_is_reported_and_added_to_the_research_total() {
    let m = manifest();
    let a = artificial_audit(&m);
    // Evaluation wall time is measured, so compare totals net of it.
    let charged = |preflight_ms: u64| {
        let output = serde_json::to_value(
            evaluation::evaluate(&m, &a, "c".repeat(64), preflight_ms, 0).unwrap(),
        )
        .unwrap();
        let get = |key: &str| output[key].as_u64().unwrap();
        (
            get("preflightCostMs"),
            get("totalResearchMs") - get("evaluationCostMs"),
        )
    };
    let (without, base) = charged(0);
    let (with, total) = charged(1_234);
    assert_eq!(without, 0);
    assert_eq!(with, 1_234);
    assert_eq!(total - base, 1_234);
}

#[test]
fn preflight_validation_refuses_mismatched_identity_and_missing_check_time() {
    let identity = "a".repeat(64);
    let valid = json!({
        "formatVersion": 1,
        "phase": "preflight",
        "manifestSha256": identity,
        "status": "available",
        "failures": [],
        "representativeBenefitDemonstrated": false,
        "checkMs": 4_321,
    });
    assert_eq!(preflight_check_ms(&valid, &identity), Ok(4_321));

    let mut mismatched = valid.clone();
    mismatched["manifestSha256"] = json!("b".repeat(64));
    assert!(
        preflight_check_ms(&mismatched, &identity)
            .unwrap_err()
            .contains("manifest mismatch")
    );

    let mut missing = valid.clone();
    missing.as_object_mut().unwrap().remove("checkMs");
    assert!(
        preflight_check_ms(&missing, &identity)
            .unwrap_err()
            .contains("checkMs")
    );
    for bad in [json!(-1), json!(1.5), json!("12")] {
        let mut wrong = valid.clone();
        wrong["checkMs"] = bad;
        assert!(
            preflight_check_ms(&wrong, &identity)
                .unwrap_err()
                .contains("checkMs missing or not a u64"),
            "{wrong}"
        );
    }

    let mut inconclusive = valid.clone();
    inconclusive["status"] = json!("inconclusive");
    assert!(
        preflight_check_ms(&inconclusive, &identity)
            .unwrap_err()
            .contains("phase/status/manifest mismatch")
    );
    let mut other_phase = valid;
    other_phase["phase"] = json!("pilot");
    assert!(
        preflight_check_ms(&other_phase, &identity)
            .unwrap_err()
            .contains("phase/status/manifest mismatch")
    );
}

fn synapse_stats() -> neat_ai_ockham::stats::ActivationStats {
    use neat_ai_ockham::stats::{ActivationStats, NeuronStats};
    let stat = |uuid: &str, variance: f64, mean_abs: f64, min: f64, max: f64| NeuronStats {
        uuid: uuid.into(),
        neuron_index: 0,
        count: 10,
        mean: 0.0,
        variance,
        std_dev: variance.sqrt(),
        mean_abs,
        min,
        max,
    };
    // Hidden `a` and wire `input-0` measured; hidden `b` deliberately not.
    ActivationStats {
        inputs: vec![stat("input-0", 0.09, 0.5, 0.0, 1.0)],
        neurons: vec![stat("a", 0.25, 0.4, -1.0, 2.0)],
        ..ActivationStats::empty()
    }
}

#[test]
fn every_synapse_visit_gets_a_source_signal_feature_vector() {
    use neat_ai_ockham::ablation::growth_units;
    use neat_ai_ockham::sweep::{parse_synapse_key, present_visits, synapse_key};
    let mut creature = neat_ai_ockham::fixtures::shortcut_edge_creature();
    let shortcut = creature
        .synapses
        .iter_mut()
        .find(|edge| edge.from_uuid == "a" && edge.to_uuid == "output-0")
        .unwrap();
    shortcut.weight = -0.3;
    let stats = synapse_stats();
    let got = audit::audit_features(&creature, &stats).unwrap();
    let neurons = neat_ai_ockham::features::extract(
        &creature,
        &stats,
        &neat_ai_ockham::features::PriorEvidence::new(),
    );
    let edges: Vec<_> = present_visits(&creature)
        .into_iter()
        .filter(|v| parse_synapse_key(v).is_some())
        .collect();
    assert_eq!(edges.len(), creature.synapses.len());
    for synapse in &creature.synapses {
        let f = got[&synapse_key(&synapse.from_uuid, &synapse.to_uuid)];
        assert_eq!(f.outgoing_weight, synapse.weight.abs());
        assert_eq!((f.fan_in, f.fan_out), (0, 1));
        assert_eq!(f.direct_growth_units, growth_units(0, 1));
        assert_eq!(f.cascade_growth_units, growth_units(0, 1));
        assert!(!f.identity && !f.blocked);
        assert_eq!((f.prior_wins, f.prior_failures), (0, 0));
    }
    let edge = |from: &str, to: &str| got[&synapse_key(from, to)];
    // Hidden source: its own stats and its own depth.
    let a_b = edge("a", "b");
    assert!(a_b.measured);
    assert_eq!((a_b.variance, a_b.mean_abs, a_b.range), (0.25, 0.4, 3.0));
    assert_eq!(a_b.depth_fraction, neurons["a"].depth_fraction);
    assert_eq!(edge("a", "output-0").mean_abs, 0.4);
    assert_eq!(edge("a", "output-0").outgoing_weight, 0.3);
    // Input source: the wire's stats, depth zero.
    let in_a = edge("input-0", "a");
    assert!(in_a.measured);
    assert_eq!((in_a.variance, in_a.mean_abs, in_a.range), (0.09, 0.5, 1.0));
    assert_eq!(in_a.depth_fraction, 0.0);
    // Unmeasured source: flagged, zero signal, structure still present.
    let b_out = edge("b", "output-0");
    assert!(!b_out.measured);
    assert_eq!(
        (b_out.variance, b_out.mean_abs, b_out.range),
        (0.0, 0.0, 0.0)
    );
    assert_eq!(b_out.outgoing_weight, 1.0);
    assert_eq!(b_out.depth_fraction, neurons["b"].depth_fraction);
}

#[test]
fn synapse_features_leave_neuron_vectors_untouched_and_unknown_keys_absent() {
    use neat_ai_ockham::sweep::synapse_key;
    let creature = neat_ai_ockham::fixtures::shortcut_edge_creature();
    let stats = synapse_stats();
    let neurons = neat_ai_ockham::features::extract(
        &creature,
        &stats,
        &neat_ai_ockham::features::PriorEvidence::new(),
    );
    let got = audit::audit_features(&creature, &stats).unwrap();
    assert_eq!(got.len(), neurons.len() + creature.synapses.len());
    for (uuid, features) in &neurons {
        assert_eq!(got[uuid], *features, "{uuid} changed by the merge");
    }
    // The synapse vector differs from its source neuron's.
    assert_ne!(got[&synapse_key("a", "b")], neurons["a"]);
    assert_eq!(got[&synapse_key("a", "b")].fan_out, 1);
    assert_eq!(neurons["a"].fan_out, 2);
    for unknown in [
        synapse_key("b", "a"),
        synapse_key("input-0", "b"),
        "output-0".to_string(),
        "missing".to_string(),
    ] {
        assert!(!got.contains_key(&unknown), "{unknown:?}");
    }
}

fn git(dir: &std::path::Path, args: &[&str]) {
    let status = std::process::Command::new("git")
        .arg("-C")
        .arg(dir)
        .args([
            "-c",
            "user.name=test",
            "-c",
            "user.email=test@example.invalid",
            "-c",
            "commit.gpgsign=false",
        ])
        .args(args)
        .stdin(std::process::Stdio::null())
        .status()
        .unwrap();
    assert!(status.success(), "git {args:?} failed");
}

#[test]
fn clean_source_guard_is_independent_of_the_working_directory() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path();
    git(root, &["init", "--quiet"]);
    let package = root.join("ockham");
    std::fs::create_dir_all(package.join("src")).unwrap();
    std::fs::write(package.join("src/lib.rs"), "// v1\n").unwrap();
    for name in ["Cargo.toml", "Cargo.lock", "rust-toolchain.toml"] {
        std::fs::write(root.join(name), "v1\n").unwrap();
    }
    git(root, &["add", "."]);
    git(root, &["commit", "--quiet", "-m", "initial"]);
    // Run from the repository root, the package directory and a nested one.
    let places = [root.to_path_buf(), package.clone(), package.join("src")];
    for place in &places {
        assert!(source_is_clean(place).unwrap(), "clean from {place:?}");
        assert_eq!(require_clean_source(place), Ok(()), "clean from {place:?}");
    }
    for file in [
        package.join("src/lib.rs"),
        root.join("Cargo.lock"),
        root.join("rust-toolchain.toml"),
        root.join("Cargo.toml"),
    ] {
        let original = std::fs::read(&file).unwrap();
        std::fs::write(&file, "edited\n").unwrap();
        for place in &places {
            assert!(
                !source_is_clean(place).unwrap(),
                "edit to {file:?} must be detected from {place:?}"
            );
            assert_eq!(
                require_clean_source(place),
                Err("experiment requires committed source and lockfile".to_string()),
                "edit to {file:?} must be refused from {place:?}"
            );
        }
        std::fs::write(&file, original).unwrap();
        assert!(source_is_clean(&package).unwrap());
    }
    // Staged but uncommitted edits are also uncommitted.
    std::fs::write(package.join("src/lib.rs"), "staged\n").unwrap();
    git(root, &["add", "."]);
    assert!(!source_is_clean(&package).unwrap());
    assert_eq!(
        require_clean_source(&package),
        Err("experiment requires committed source and lockfile".to_string())
    );
}

/// Three independent validation groups, each with a same-family calibration
/// task, plus one source task. Source history and validation pools carry a
/// signal that only the learned features can see: the partial score is
/// constant and calibration labels are uninformative, so random, partial and
/// target-only ordering are all chance-level. Six of forty candidates
/// improve; the budget of five therefore rejects one of them.
fn verdict_manifest(gain: f64, false_rejection: f64, cost_ratio: f64) -> Manifest {
    let digest = |n: usize| format!("{n:064x}");
    let mut tasks = vec![json!({
        "id":"source", "family":"fs", "lineage":"ls", "corpusVersion":"vs", "split":"source",
        "parent":"ps", "parentSha256":digest(1), "corpus":"cs", "corpusSha256":digest(2),
        "corpusIdentity":"cs",
        "history":{"path":"/private/history.jsonl","sha256":digest(3),"acquisitionMs":1000}
    })];
    for n in 0..3 {
        for (kind, split) in [("cal", "calibration"), ("val", "validation")] {
            let offset = 10 + n * 10 + if kind == "cal" { 0 } else { 5 };
            tasks.push(json!({
                "id":format!("{kind}{n}"), "family":format!("family{n}"),
                "lineage":format!("lineage-{kind}{n}"), "corpusVersion":format!("version-{kind}{n}"),
                "split":split, "parent":format!("p-{kind}{n}"), "parentSha256":digest(offset),
                "corpus":format!("c-{kind}{n}"), "corpusSha256":digest(offset + 1),
                "corpusIdentity":format!("c-{kind}{n}"), "history":null
            }));
        }
    }
    let mut m = manifest();
    m.tasks = serde_json::from_value(json!(tasks)).unwrap();
    m.seeds = vec![7, 19];
    m.minimum_recall_gain = gain;
    m.maximum_false_rejection = false_rejection;
    m.maximum_cost_ratio = cost_ratio;
    m.validate().unwrap();
    m
}

fn verdict_signal(i: usize) -> f64 {
    if i >= 34 {
        0.1 + i as f64 * 0.01
    } else {
        -0.5 + i as f64 * 0.001
    }
}

fn verdict_audit(m: &Manifest) -> audit::Audit {
    let mut a = artificial_audit(m);
    for pool in &mut a.pools {
        let informative = pool.task.starts_with("val");
        for row in &mut pool.rows {
            let i = row.visit[1..].parse::<usize>().unwrap();
            row.sample_delta = Some(0.0);
            row.full_delta = Some(if informative || pool.task == "source" {
                verdict_signal(i)
            } else {
                -0.5
            });
        }
    }
    let source = &m.tasks[0];
    let stamp = neat_ai_ockham::RunStamp {
        host: "test".into(),
        corpus_identity: source.corpus_identity.clone(),
        creature_checksum: source.parent_sha256.clone(),
        ordering: "random".into(),
        seed: 7,
    };
    a.histories[0].task = source.id.clone();
    a.histories[0].records = (0..40)
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
            row.sample_delta = Some(0.0);
            row.full_delta = Some(verdict_signal(i));
            row
        })
        .collect();
    a
}

fn verdict(m: &Manifest) -> (String, Vec<String>) {
    let report = serde_json::to_value(
        evaluation::evaluate(m, &verdict_audit(m), "c".repeat(64), 0, 0).unwrap(),
    )
    .unwrap();
    let reasons = report["reasons"]
        .as_array()
        .unwrap()
        .iter()
        .map(|r| r.as_str().unwrap().to_string())
        .collect();
    (report["verdict"].as_str().unwrap().to_string(), reasons)
}

#[test]
fn verdict_requires_recall_false_rejection_and_cost_thresholds() {
    let (verdict_name, reasons) = verdict(&verdict_manifest(0.05, 0.2, 100.0));
    assert_eq!(
        verdict_name, "supported-for-further-research",
        "{reasons:?}"
    );
    assert!(reasons.is_empty(), "{reasons:?}");

    for (m, expected) in [
        // The gain cannot be exceeded when recall is at most one.
        (
            verdict_manifest(1.0, 0.2, 100.0),
            "recall=false, false rejection=true, total cost=true",
        ),
        // Five evaluations miss one of six improvements: 1/6 > 0.1.
        (
            verdict_manifest(0.05, 0.1, 100.0),
            "recall=true, false rejection=false, total cost=true",
        ),
        // Transfer also pays for history and calibration, so it cannot be
        // no more expensive than the partial-score control.
        (
            verdict_manifest(0.05, 0.2, 1.0),
            "recall=true, false rejection=true, total cost=false",
        ),
    ] {
        let (verdict_name, reasons) = verdict(&m);
        assert_eq!(verdict_name, "no-demonstrated-benefit", "{reasons:?}");
        assert_eq!(reasons, vec![format!("predeclared thresholds: {expected}")]);
    }
}

#[test]
fn verdict_is_inconclusive_when_fewer_than_three_groups_validate() {
    let mut m = verdict_manifest(0.05, 0.2, 100.0);
    m.tasks.retain(|t| !t.id.ends_with('2'));
    m.validate().unwrap();
    let (verdict_name, reasons) = verdict(&m);
    assert_eq!(verdict_name, "inconclusive");
    assert!(
        reasons
            .iter()
            .any(|r| r.contains("independent validation groups"))
    );
}

#[test]
fn cost_milliseconds_round_up_so_sub_millisecond_work_is_counted() {
    use std::time::Duration;
    assert_eq!(audit::ceil_ms(Duration::ZERO), 0, "no work costs nothing");
    assert_eq!(
        audit::ceil_ms(Duration::from_nanos(1)),
        1,
        "sub-millisecond feature or fit work is charged, not rounded to 0"
    );
    assert_eq!(
        audit::ceil_ms(Duration::from_micros(999)),
        1,
        "just under 1 ms"
    );
    assert_eq!(
        audit::ceil_ms(Duration::from_millis(3)),
        3,
        "whole milliseconds"
    );
    assert_eq!(
        audit::ceil_ms(Duration::from_micros(3_001)),
        4,
        "partial ms rounds up"
    );
    assert_eq!(
        audit::ceil_ms(Duration::from_secs(u64::MAX)),
        u64::MAX,
        "saturates instead of overflowing"
    );
}

#[test]
fn path_sensitivity_sums_every_downstream_path_not_just_one_hop() {
    use neat_ai_ockham::fixtures::{creature, neuron, synapse};
    use neat_ai_ockham::sweep::synapse_key;
    // input-0 → h1 →(2) h2 →(3) output-0, plus h3 fed by the input and reaching no output.
    let c = creature(
        1,
        1,
        vec![
            neuron("hidden", "h1", 0.0, Some("TANH")),
            neuron("hidden", "h2", 0.0, Some("TANH")),
            neuron("hidden", "h3", 0.0, Some("TANH")),
            neuron("output", "output-0", 0.0, Some("IDENTITY")),
        ],
        vec![
            synapse("input-0", "h1", 1.0),
            synapse("h1", "h2", 2.0),
            synapse("h2", "output-0", 3.0),
            synapse("input-0", "h3", 1.0),
        ],
    );
    let feature = |mean_abs: f64, outgoing_weight: f64| neat_ai_ockham::CandidateFeatures {
        measured: true,
        mean_abs,
        outgoing_weight,
        ..Default::default()
    };
    let mut features = std::collections::HashMap::new();
    features.insert("h1".to_string(), feature(0.5, 2.0));
    features.insert("h2".to_string(), feature(0.25, 3.0));
    features.insert("h3".to_string(), feature(0.75, 1.0));
    features.insert(synapse_key("input-0", "h1"), feature(0.1, 1.0));
    features.insert(synapse_key("h1", "h2"), feature(0.5, 2.0));
    features.insert(synapse_key("h2", "output-0"), feature(0.25, 3.0));
    features.insert(synapse_key("input-0", "h3"), feature(0.1, 1.0));
    let got = audit::path_sensitivities(&c, &features).unwrap();
    assert_eq!(got["h1"], 0.5 * 6.0, "h1 reaches the output through 2 × 3");
    assert_eq!(got["h2"], 0.25 * 3.0);
    assert_eq!(got[&synapse_key("h1", "h2")], 0.5 * 2.0 * 3.0);
    assert_eq!(got["h3"], 0.0, "a neuron reaching no output scores zero");
    assert_ne!(
        got["h1"],
        features["h1"].downstream_sensitivity(),
        "the all-paths value is not the one-hop value"
    );
}

#[test]
fn path_sensitivity_columns_are_appended_and_match_their_names() {
    let row = |path: Option<f64>| audit::AuditRow {
        visit: "v".into(),
        kind: Some("ablation".into()),
        candidate_sha256: None,
        features: None,
        sample_delta: Some(0.2),
        full_delta: Some(0.3),
        sample_ms: 1,
        full_ms: 1,
        telemetry: None,
        failure: None,
        path_sensitivity: path,
    };
    let names = evaluation::feature_names();
    let n = names.len();
    assert_eq!(names[n - 2], "pathSensitivityAvailable");
    assert_eq!(names[n - 1], "logPathSensitivity");
    let with = evaluation::labelled(&[row(Some(std::f64::consts::E - 1.0))]).unwrap();
    let without = evaluation::labelled(&[row(None)]).unwrap();
    assert_eq!(with[0].features.len(), n);
    assert_eq!(without[0].features.len(), n);
    assert!((with[0].features[n - 2] - 1.0).abs() < 1e-12);
    assert!((with[0].features[n - 1] - 1.0).abs() < 1e-12);
    assert_eq!(&without[0].features[n - 2..], &[0.0, 0.0]);
}

#[test]
fn evaluation_refuses_a_version_one_audit_and_accepts_version_two() {
    let m = manifest();
    let mut a = artificial_audit(&m);
    assert_eq!(a.format_version, 2);
    a.format_version = 1;
    assert_eq!(
        evaluation::evaluate(&m, &a, "c".repeat(64), 0, 0).unwrap_err(),
        "audit protocol/manifest/freeze mismatch"
    );
    a.format_version = 2;
    assert!(evaluation::evaluate(&m, &a, "c".repeat(64), 0, 0).is_ok());
}

#[test]
fn labelled_refuses_a_non_finite_or_negative_path_sensitivity() {
    let row = |path: Option<f64>| audit::AuditRow {
        visit: "v".into(),
        kind: Some("ablation".into()),
        candidate_sha256: None,
        features: None,
        sample_delta: Some(0.2),
        full_delta: Some(0.3),
        sample_ms: 1,
        full_ms: 1,
        telemetry: None,
        failure: None,
        path_sensitivity: path,
    };
    for bad in [-1.0, f64::NAN] {
        assert_eq!(
            evaluation::labelled(&[row(Some(bad))]).err().as_deref(),
            Some("non-finite or negative path sensitivity"),
            "refused: {bad}"
        );
    }
    assert!(
        evaluation::labelled(&[row(Some(0.0))]).is_ok(),
        "zero is an accepted control"
    );
}

#[test]
fn freeze_refuses_a_version_one_pilot_and_accepts_version_two() {
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
    pilot.format_version = 1;
    assert_eq!(
        freeze(&m, &pilot, "d".repeat(64)).err().as_deref(),
        Some("pilot incomplete or manifest mismatch")
    );
    pilot.format_version = 2;
    assert!(freeze(&m, &pilot, "d".repeat(64)).is_ok());
}
