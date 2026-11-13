use super::*;
use manifest::{Manifest, Split};
use serde_json::json;

fn manifest() -> Manifest {
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
        "tasks": [
          {"id":"source", "family":"family", "lineage":"l1", "corpusVersion":"v1", "split":"source", "observedUnix":90, "parent":"p1", "parentSha256":"1".repeat(64), "corpus":"c1", "corpusSha256":"4".repeat(64)},
          {"id":"cal", "family":"family", "lineage":"l2", "corpusVersion":"v2", "split":"calibration", "observedUnix":101, "parent":"p2", "parentSha256":"2".repeat(64), "corpus":"c2", "corpusSha256":"5".repeat(64)},
          {"id":"val", "family":"family", "lineage":"l3", "corpusVersion":"v3", "split":"validation", "observedUnix":102, "parent":"p3", "parentSha256":"3".repeat(64), "corpus":"c3", "corpusSha256":"6".repeat(64)}
        ]
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
        assert!(changed.validate().unwrap_err().contains("split leakage"), "{field}");
    }
    let mut late = base.clone();
    late.tasks[0].observed_unix = 100;
    assert!(late.validate().unwrap_err().contains("historical cutoff"));
    late.tasks[0].observed_unix = 99;
    late.validate().unwrap();
}

#[test]
fn metrics_count_ties_small_pools_and_improvements_outside_budget() {
    let scores = [1.0, 1.0, -1.0, 0.1];
    let result = evaluation::metrics(&scores, &[2,3,0,1], 2, 0.0).unwrap();
    assert_eq!(result.top_count, 2, "include the entire tie at the ceil(5%) boundary");
    assert_eq!(result.top_recall, 0.0);
    assert_eq!(result.false_rejection, Some(2.0/3.0));
    let all = evaluation::metrics(&scores, &[0,1,2,3], 99, 0.0).unwrap();
    assert_eq!(all.top_recall, 1.0);
    assert_eq!(all.false_rejection, Some(0.0));
    assert!(evaluation::metrics(&[], &[], 1, 0.0).unwrap_err().contains("empty"));
    assert!(evaluation::metrics(&[f64::NAN], &[0], 1, 0.0).unwrap_err().contains("finite"));
    assert!(evaluation::metrics(&scores, &[0,0,2,3], 1, 0.0).unwrap_err().contains("permutation"));
    assert_eq!(evaluation::metrics(&[-1.0], &[0], 1, 0.0).unwrap().false_rejection, None);
}

#[test]
fn seeded_selection_is_reproducible_and_independent_of_input_order() {
    let items = vec!["a".to_string(), "b".into(), "c".into(), "d".into()];
    let a = manifest::seeded_order(&items, 7);
    let mut reversed = items.clone(); reversed.reverse();
    assert_eq!(a, manifest::seeded_order(&reversed, 7));
    let orders: std::collections::BTreeSet<_> = (0..10).map(|s| manifest::seeded_order(&items,s)).collect();
    assert!(orders.len() > 1);
    assert_eq!(a.len(), 4);
}

#[test]
fn missing_scores_are_unlabelled_and_cold_calibration_is_empty() {
    let rows = vec![audit::AuditRow { visit:"a".into(), kind:Some("ablation".into()), sample_delta:Some(0.2), full_delta:None, features:None, telemetry:None, failure:Some("scorer failed".into()), candidate_sha256:None, sample_ms:1, full_ms:1 }];
    assert_eq!(evaluation::labelled(&rows).unwrap().len(), 0);
    let mut scored = rows.clone(); scored[0].full_delta = Some(0.3);
    assert_eq!(evaluation::labelled(&scored).unwrap()[0].score, 0.3);
    scored[0].sample_delta = None;
    assert!(evaluation::labelled(&scored).unwrap_err().contains("partial"));
    assert_eq!(manifest().tasks.iter().filter(|t| t.split == Split::Calibration).count(), 1);
}

#[test]
fn audit_scores_sampled_losers_individually_and_preserves_parent() {
    use neat_ai_ockham::scorer::{DirectoryScorer, ScorerError, ScorerMode, ScoreResult};
    use std::collections::BTreeMap;
    use std::path::Path;
    struct Judge { omit_candidate: bool, fail: bool }
    impl DirectoryScorer for Judge {
        fn identity(&self) -> String { "test-judge".into() }
        fn score_directory(&self, dir: &Path, _: &Path, mode: ScorerMode) -> Result<BTreeMap<String, ScoreResult>, ScorerError> {
            assert!(dir.join("baseline.json").is_file());
            assert!(dir.join("candidate.json").is_file());
            if self.fail { return Err(ScorerError::Spawn("deliberate missing binary".into())); }
            let count = if mode.is_authoritative() { 32 } else { 2 };
            let result = |score| ScoreResult { score, error:1.0-score, complexity_penalty:0.0, record_count:count, sample_rate:None, gpu_backend:None, cost_name:None, time_taken:0.0 };
            let mut map = BTreeMap::from([("baseline".into(), result(0.5))]);
            if !self.omit_candidate { map.insert("candidate".into(), result(if mode.is_authoritative(){0.6}else{0.4})); }
            Ok(map)
        }
    }
    let tmp = tempfile::tempdir().unwrap();
    let train = tmp.path().join("train"); std::fs::create_dir(&train).unwrap();
    neat_ai_ockham::corpus::write_bin_file(&train.join("0.bin"), &(0..32).map(|i| (vec![i as f32 / 32.0],vec![0.0])).collect::<Vec<_>>()).unwrap();
    let creature = neat_ai_ockham::fixtures::shortcut_edge_creature();
    let text = neat_core::creature_to_json(&creature).unwrap();
    let parent = tmp.path().join("parent.json"); std::fs::write(&parent,&text).unwrap();
    let mut m = manifest();
    m.tasks[1].parent = parent.clone(); m.tasks[1].corpus = train;
    m.tasks[1].parent_sha256 = neat_ai_ockham::incumbent::sha256_hex(text.as_bytes());
    m.tasks[1].corpus_sha256 = audit::corpus_digest(&m.tasks[1].corpus, m.max_corpus_bytes).unwrap();
    let pool = audit::collect_pool(&m, &m.tasks[1], 7, 40, &Judge { omit_candidate:false,fail:false }, &tmp.path().join("ok")).unwrap();
    assert_eq!(pool.rows.len(), pool.selected.len());
    assert!(pool.rows.iter().any(|r| r.kind.as_deref() == Some("synapse")));
    let scored: Vec<_> = pool.rows.iter().filter(|r| r.kind.is_some()).collect();
    assert!(!scored.is_empty());
    for row in scored { assert!(row.sample_delta.unwrap() < 0.0); assert!(row.full_delta.unwrap() > 0.0); assert_eq!(row.failure, None); }
    assert_eq!(std::fs::read_to_string(&parent).unwrap(),text);
    let bad = audit::collect_pool(&m, &m.tasks[1], 7, 40, &Judge {omit_candidate:true,fail:false}, &tmp.path().join("missing")).unwrap();
    for row in bad.rows.iter().filter(|r| r.kind.is_some()) {
        assert!(row.failure.as_ref().unwrap().contains("missing candidate")); assert_eq!(row.full_delta,None);
    }
    let bad = audit::collect_pool(&m, &m.tasks[1], 7, 40, &Judge {omit_candidate:false,fail:true}, &tmp.path().join("failed")).unwrap();
    assert!(bad.rows.iter().filter(|r| r.kind.is_some()).all(|r| r.failure.as_ref().unwrap().contains("deliberate missing binary")));
    let mut changed = m.clone(); changed.tasks[1].parent_sha256 = "f".repeat(64);
    assert!(audit::collect_pool(&changed, &changed.tasks[1],7,40,&Judge{omit_candidate:false,fail:false},&tmp.path().join("changed")).unwrap_err().contains("parent digest"));
}
