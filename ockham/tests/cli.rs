//! CLI: help/version plus the Issue #2 baseline gate.

use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::process::Command;

use neat_ai_ockham::corpus::write_bin_file;
use neat_ai_ockham::fixtures::{
    hidden_identity_creature, identity_creature_json, recurrent_flagged_creature_json,
};

fn bin() -> Command {
    Command::new(env!("CARGO_BIN_EXE_neat_ai_ockham"))
}

fn stdout(out: &std::process::Output) -> String {
    String::from_utf8_lossy(&out.stdout).into_owned()
}

fn stderr(out: &std::process::Output) -> String {
    String::from_utf8_lossy(&out.stderr).into_owned()
}

fn fake_scorer(dir: &Path, body: &str) -> PathBuf {
    let path = dir.join("fake_scorer");
    let script = format!("#!/bin/sh\ncat <<'EOF'\n{body}\nEOF\n");
    std::fs::write(&path, script).unwrap();
    let mut perms = std::fs::metadata(&path).unwrap().permissions();
    perms.set_mode(0o755);
    std::fs::set_permissions(&path, perms).unwrap();
    path
}

fn write_training(dir: &Path, n: usize) {
    let recs: Vec<(Vec<f32>, Vec<f32>)> =
        (0..n).map(|i| (vec![i as f32], vec![i as f32])).collect();
    write_bin_file(&dir.join("0.bin"), &recs).unwrap();
}

#[test]
fn help_and_version_succeed() {
    let help = bin().arg("--help").output().unwrap();
    assert!(help.status.success(), "{}", stderr(&help));
    let help_text = stdout(&help);
    assert!(help_text.contains("neat_ai_ockham"));
    assert!(help_text.contains("[CREATURE]"));
    assert!(help_text.contains("[TRAINING_DATA]"));
    assert!(help_text.contains("--timeout-seconds"));
    assert!(help_text.contains("--version"));

    let version = bin().arg("--version").output().unwrap();
    assert!(version.status.success(), "{}", stderr(&version));
    assert!(stdout(&version).contains(env!("CARGO_PKG_VERSION")));
}

#[test]
fn missing_positionals_print_usage() {
    let out = bin().output().unwrap();
    assert!(!out.status.success());
    assert!(stderr(&out).contains("usage: neat_ai_ockham"));
}

#[test]
fn baseline_gate_writes_workspace_without_pruning() {
    let tmp = tempfile::tempdir().unwrap();
    let creature = tmp.path().join("creature.json");
    let text = identity_creature_json(1, 1);
    std::fs::write(&creature, &text).unwrap();
    let train = tmp.path().join("train");
    std::fs::create_dir(&train).unwrap();
    write_training(&train, 4);
    let out_dir = tmp.path().join("out");
    let scorer = fake_scorer(
        tmp.path(),
        r#"{"baseline":{"score":0.91,"error":0.09,"complexityPenalty":1e-8,"recordCount":4,"costName":"MSE","timeTaken":0.01}}"#,
    );

    let out = bin()
        .arg(&creature)
        .arg(&train)
        .arg("--output-dir")
        .arg(&out_dir)
        .arg("--scorer")
        .arg(&scorer)
        .output()
        .unwrap();
    assert!(out.status.success(), "{}", stderr(&out));
    let report: serde_json::Value = serde_json::from_str(&stdout(&out)).expect("json");
    assert_eq!(report["optimisation"], "complete");
    assert_eq!(report["baseline"]["score"], 0.91);
    assert!(report["baseline"]["score"].as_f64().unwrap() > 0.0);
    assert_eq!(std::fs::read_to_string(&creature).unwrap(), text);
    assert_eq!(
        std::fs::read_to_string(out_dir.join("best.json")).unwrap(),
        text
    );
    assert!(out_dir.join("workspace/incumbent.json").exists());
    assert!(out_dir.join("workspace/incumbent.meta.json").exists());
    assert!(out_dir.join("workspace/baseline.json").exists());
}

#[test]
fn stats_sample_records_bounds_the_activation_scan_and_zero_restores_the_full_one() {
    let tmp = tempfile::tempdir().unwrap();
    let creature = tmp.path().join("creature.json");
    std::fs::write(
        &creature,
        neat_core::creature_to_json_pretty(&hidden_identity_creature(0.0, 1.0)).unwrap(),
    )
    .unwrap();
    let train = tmp.path().join("train");
    std::fs::create_dir(&train).unwrap();
    write_training(&train, 4_000);
    let scorer = fake_scorer(
        tmp.path(),
        r#"{"baseline":{"score":0.5,"error":0.5,"complexityPenalty":1e-8,"recordCount":4000,"costName":"MSE","timeTaken":0.01}}"#,
    );

    let activation = |sample: &str| -> serde_json::Value {
        let out = bin()
            .arg(&creature)
            .arg(&train)
            .arg("--output-dir")
            .arg(tmp.path().join(format!("out-{sample}")))
            .arg("--scorer")
            .arg(&scorer)
            .arg("--stats-sample-records")
            .arg(sample)
            .arg("--timeout-seconds")
            .arg("1")
            .output()
            .unwrap();
        assert!(out.status.success(), "{}", stderr(&out));
        serde_json::from_str::<serde_json::Value>(&stdout(&out)).expect("json")["activation"]
            .clone()
    };

    let sampled = activation("500");
    assert_eq!(sampled["corpusRecordCount"], 4_000);
    assert!(
        sampled["recordCount"].as_u64().unwrap() <= 500,
        "sampled scan must stay inside the cap: {sampled}"
    );
    assert!(sampled["recordCount"].as_u64().unwrap() > 0);
    assert_eq!(sampled["sample"]["maxRecords"], 500);
    assert_eq!(sampled["neurons"][0]["count"], sampled["recordCount"]);

    let full = activation("0");
    assert_eq!(full["recordCount"], 4_000);
    assert_eq!(full["sample"]["maxRecords"], 0);
}

#[test]
fn non_forward_only_is_rejected_and_source_untouched() {
    let tmp = tempfile::tempdir().unwrap();
    let creature = tmp.path().join("creature.json");
    let text = recurrent_flagged_creature_json(1, 1);
    std::fs::write(&creature, &text).unwrap();
    let train = tmp.path().join("train");
    std::fs::create_dir(&train).unwrap();
    write_training(&train, 2);
    let scorer = fake_scorer(tmp.path(), r#"{"baseline":{"score":1,"error":0}}"#);

    let out = bin()
        .arg(&creature)
        .arg(&train)
        .arg("--output-dir")
        .arg(tmp.path().join("out"))
        .arg("--scorer")
        .arg(&scorer)
        .output()
        .unwrap();
    assert!(!out.status.success());
    assert!(
        stderr(&out).contains("forward-only") || stderr(&out).contains("forwardOnly"),
        "{}",
        stderr(&out)
    );
    assert_eq!(std::fs::read_to_string(&creature).unwrap(), text);
    assert!(!tmp.path().join("out/best.json").exists());
}

#[test]
fn fake_scorer_failure_aborts_without_best() {
    let tmp = tempfile::tempdir().unwrap();
    let creature = tmp.path().join("creature.json");
    std::fs::write(&creature, identity_creature_json(1, 1)).unwrap();
    let train = tmp.path().join("train");
    std::fs::create_dir(&train).unwrap();
    write_training(&train, 2);
    let scorer = fake_scorer_fail(tmp.path());
    let out_dir = tmp.path().join("out");

    let out = bin()
        .arg(&creature)
        .arg(&train)
        .arg("--output-dir")
        .arg(&out_dir)
        .arg("--scorer")
        .arg(&scorer)
        .output()
        .unwrap();
    assert!(!out.status.success(), "{}", stdout(&out));
    assert!(!out_dir.join("best.json").exists());
}

fn fake_scorer_fail(dir: &Path) -> PathBuf {
    let path = dir.join("fake_scorer_fail");
    std::fs::write(&path, "#!/bin/sh\necho boom >&2\nexit 1\n").unwrap();
    let mut perms = std::fs::metadata(&path).unwrap().permissions();
    perms.set_mode(0o755);
    std::fs::set_permissions(&path, perms).unwrap();
    path
}

#[test]
fn invalid_candidates_names_the_flag() {
    let out = bin()
        .arg("creature.json")
        .arg("training")
        .arg("--candidates")
        .arg("0")
        .output()
        .unwrap();
    assert_eq!(out.status.code(), Some(2));
    assert!(stderr(&out).contains("--candidates"));
}

#[test]
fn an_unknown_ordering_names_the_valid_strategies() {
    let out = bin()
        .arg("creature.json")
        .arg("training")
        .arg("--ordering")
        .arg("cleverest")
        .output()
        .unwrap();
    assert!(!out.status.success());
    let err = stderr(&out);
    assert!(err.contains("cleverest"), "{err}");
    assert!(err.contains("low-variance"), "{err}");
    assert!(err.contains("random"), "{err}");
}

#[test]
fn an_out_of_range_ordering_random_quota_names_the_flag() {
    let out = bin()
        .arg("creature.json")
        .arg("training")
        .arg("--ordering-random-quota")
        .arg("1.0")
        .output()
        .unwrap();
    assert_eq!(out.status.code(), Some(2));
    assert!(stderr(&out).contains("--ordering-random-quota"));
}

#[test]
fn help_lists_the_ordering_flags() {
    let help = bin().arg("--help").output().unwrap();
    assert!(help.status.success(), "{}", stderr(&help));
    let text = stdout(&help);
    assert!(text.contains("--ordering"), "{text}");
    assert!(text.contains("--ordering-random-quota"), "{text}");
}

/// Issue #96: the accept cap is gone — a run stops on its budget, never on an
/// accept — so the flag is rejected outright rather than quietly ignored.
#[test]
fn max_accepts_is_gone_from_the_cli() {
    let help = bin().arg("--help").output().unwrap();
    assert!(help.status.success(), "{}", stderr(&help));
    let text = stdout(&help);
    assert!(!text.contains("--max-accepts"), "{text}");

    let out = bin()
        .arg("creature.json")
        .arg("training")
        .arg("--max-accepts")
        .arg("1")
        .output()
        .unwrap();
    assert!(!out.status.success());
    assert!(stderr(&out).contains("--max-accepts"), "{}", stderr(&out));
}

/// Issue #77: a run that ends without adding a single uuid to the screened set
/// while unchecked neurons remain must say so on stderr. The overnight plateau
/// of #63 produced no signal at all; after this, eight silent runs are eight
/// warnings.
#[test]
fn a_run_that_screens_nothing_warns_that_it_advanced_no_coverage() {
    let tmp = tempfile::tempdir().unwrap();
    let creature = tmp.path().join("creature.json");
    std::fs::write(
        &creature,
        neat_core::creature_to_json_pretty(&hidden_identity_creature(0.0, 1.0)).unwrap(),
    )
    .unwrap();
    let train = tmp.path().join("train");
    std::fs::create_dir(&train).unwrap();
    write_training(&train, 8);
    let scorer = fake_scorer(
        tmp.path(),
        r#"{"baseline":{"score":0.5,"error":0.5,"complexityPenalty":1e-8,"recordCount":8,"costName":"MSE","timeTaken":0.01}}"#,
    );

    let out = bin()
        .arg(&creature)
        .arg(&train)
        .arg("--output-dir")
        .arg(tmp.path().join("out"))
        .arg("--scorer")
        .arg(&scorer)
        .arg("--learnings-dir")
        .arg(tmp.path().join("learnings"))
        // Stops the loop before a single batch is filled.
        .arg("--max-experiments")
        .arg("0")
        .arg("--timeout-seconds")
        .arg("30")
        .output()
        .unwrap();

    assert!(out.status.success(), "{}", stderr(&out));
    let log = stderr(&out);
    assert!(
        log.contains("no progress: 0 newly checked uuid(s) this run"),
        "a zero-progress run must warn: {log}"
    );
    assert!(
        log.contains("hidden neuron(s) remain unchecked"),
        "the warning must name the unchecked figure too: {log}"
    );
    let summary: serde_json::Value = serde_json::from_str(&stdout(&out)).expect("json");
    assert_eq!(summary["newlyScreened"], 0, "{summary}");
}

/// Issue #104: a malformed or contradictory ladder is a configuration fault,
/// refused with exit 2 before any scorer is spawned — never a silently ignored
/// flag that leaves the run screening at some other rate.
#[test]
fn a_bad_screening_ladder_is_refused_with_exit_two() {
    let tmp = tempfile::tempdir().unwrap();
    let creature = tmp.path().join("creature.json");
    std::fs::write(&creature, identity_creature_json(1, 1)).unwrap();
    let train = tmp.path().join("train");
    std::fs::create_dir(&train).unwrap();
    write_training(&train, 4);

    let refuse = |args: &[&str]| -> String {
        let out = bin()
            .arg(&creature)
            .arg(&train)
            .arg("--output-dir")
            .arg(tmp.path().join("out"))
            .args(args)
            .output()
            .unwrap();
        assert_eq!(out.status.code(), Some(2), "{}", stderr(&out));
        stderr(&out)
    };

    assert!(refuse(&["--screen-stages", "0.05,0.01"]).contains("ascend"));
    assert!(refuse(&["--screen-stages", "0.0025:0,0.05"]).contains("more evidence"));
    assert!(refuse(&["--screen-stages", "0.0025,2.0"]).contains("(0, 1)"));
    assert!(
        refuse(&[
            "--screen-stages",
            "0.0025,0.05",
            "--screen-sample-rate",
            "0"
        ])
        .contains("--screen-stages")
    );
    // A margin with no ladder to apply it to would otherwise be a silent no-op.
    assert!(
        refuse(&["--screen-reject-margin", "0.5"]).contains("--screen-reject-margin has no effect")
    );
}

/// Issue #107: the learned ranker is fitted offline from the candidate logs a
/// run writes, so the CLI has to close that loop end to end — log rows in,
/// versioned model out, and a model the sweep will then accept.
#[test]
fn train_ordering_fits_a_model_from_candidate_logs_and_reports_its_ranking_quality() {
    let tmp = tempfile::tempdir().unwrap();
    let log = tmp.path().join("candidates.jsonl");
    let stamp = neat_ai_ockham::RunStamp {
        host: "GRQ-1".into(),
        corpus_identity: "corpus-a".into(),
        creature_checksum: "abc".into(),
        ordering: "composite".into(),
        seed: 7,
    };
    // Quiet neurons are the wins; loud ones are not. Nothing here decides
    // anything — these are outcomes a scorer already returned.
    let rows: Vec<neat_ai_ockham::CandidateRecord> = (0..40)
        .map(|i| {
            let quiet = i % 2 == 0;
            let features = neat_ai_ockham::CandidateFeatures {
                measured: true,
                mean_abs: if quiet { 0.01 } else { 3.0 },
                outgoing_weight: 1.0,
                cascade_growth_units: 2.0,
                ..Default::default()
            };
            let mut record = neat_ai_ockham::CandidateRecord::new(
                &stamp,
                &format!("h{i}"),
                "ablation",
                &features,
                if quiet {
                    neat_ai_ockham::CandidateOutcome::Accepted
                } else {
                    neat_ai_ockham::CandidateOutcome::Rejected
                },
            );
            record.full_delta = Some(if quiet { 0.02 } else { -0.02 });
            record
        })
        .collect();
    neat_ai_ockham::telemetry::append(&log, &rows).unwrap();

    let model = tmp.path().join("model.json");
    let out = bin()
        .arg("train-ordering")
        .arg(&log)
        .arg("--out")
        .arg(&model)
        .output()
        .unwrap();
    assert!(out.status.success(), "{}", stderr(&out));
    let report: serde_json::Value = serde_json::from_str(&stdout(&out)).unwrap();
    assert_eq!(report["records"], 40);
    assert_eq!(report["skippedRecords"], 0);
    assert_eq!(report["evaluatedOn"], "holdout");
    assert_eq!(report["corpora"][0], "corpus-a");
    let auc = report["evaluation"]["auc"].as_f64().unwrap();
    assert!(
        auc > 0.9,
        "the model must rank the signal it was given: {auc}"
    );
    assert!(report["coefficients"]["logMeanAbs"].as_f64().unwrap() < 0.0);

    // The model the trainer wrote is the model the sweep accepts.
    let loaded = neat_ai_ockham::PriorityModel::load(&model).unwrap();
    assert_eq!(
        loaded.format_version(),
        neat_ai_ockham::PRIORITY_MODEL_FORMAT_VERSION
    );
    assert_eq!(loaded.training().rows, 32, "every fifth row is held out");
    assert_eq!(loaded.training().config.corpora, ["corpus-a"]);
}

/// Helper: a candidate log whose quiet rows are the wins.
fn write_candidate_log(path: &Path) {
    let stamp = neat_ai_ockham::RunStamp {
        host: "GRQ-1".into(),
        corpus_identity: "corpus-a".into(),
        creature_checksum: "abc".into(),
        ordering: "composite".into(),
        seed: 7,
    };
    let rows: Vec<neat_ai_ockham::CandidateRecord> = (0..40)
        .map(|i| {
            let quiet = i % 2 == 0;
            let features = neat_ai_ockham::CandidateFeatures {
                measured: true,
                mean_abs: if quiet { 0.01 } else { 3.0 },
                outgoing_weight: 1.0,
                cascade_growth_units: 2.0,
                ..Default::default()
            };
            let mut record = neat_ai_ockham::CandidateRecord::new(
                &stamp,
                &format!("h{i}"),
                "ablation",
                &features,
                if quiet {
                    neat_ai_ockham::CandidateOutcome::Accepted
                } else {
                    neat_ai_ockham::CandidateOutcome::Rejected
                },
            );
            record.full_delta = Some(if quiet { 0.02 } else { -0.02 });
            record
        })
        .collect();
    neat_ai_ockham::telemetry::append(path, &rows).unwrap();
}

/// `--holdout-every 0` is the documented "no holdout" setting, and the report
/// must say which rows the numbers came from rather than implying a holdout.
#[test]
fn train_ordering_without_a_holdout_says_it_evaluated_on_the_training_rows() {
    let tmp = tempfile::tempdir().unwrap();
    let log = tmp.path().join("candidates.jsonl");
    write_candidate_log(&log);
    let out = bin()
        .arg("train-ordering")
        .arg(&log)
        .arg("--out")
        .arg(tmp.path().join("model.json"))
        .arg("--holdout-every")
        .arg("0")
        .output()
        .unwrap();
    assert!(out.status.success(), "{}", stderr(&out));
    let report: serde_json::Value = serde_json::from_str(&stdout(&out)).unwrap();
    assert_eq!(report["evaluatedOn"], "train");
    assert_eq!(report["holdoutRows"], 0);
    assert_eq!(report["trainingRows"], 40);
}

/// Holding out every row leaves nothing to fit; refused by name rather than
/// quietly reinterpreted.
#[test]
fn train_ordering_refuses_a_holdout_of_every_row() {
    let tmp = tempfile::tempdir().unwrap();
    let log = tmp.path().join("candidates.jsonl");
    write_candidate_log(&log);
    let model = tmp.path().join("model.json");
    let out = bin()
        .arg("train-ordering")
        .arg(&log)
        .arg("--out")
        .arg(&model)
        .arg("--holdout-every")
        .arg("1")
        .output()
        .unwrap();
    assert!(!out.status.success());
    assert!(stderr(&out).contains("--holdout-every"), "{}", stderr(&out));
    assert!(!model.exists(), "a refused fit must write no model");
}

#[test]
fn train_ordering_without_logs_prints_usage() {
    let tmp = tempfile::tempdir().unwrap();
    let out = bin()
        .arg("train-ordering")
        .arg("--out")
        .arg(tmp.path().join("model.json"))
        .output()
        .unwrap();
    assert!(!out.status.success());
    assert!(stderr(&out).contains("train-ordering"), "{}", stderr(&out));
}

/// A `learned` run with no model must stop rather than rank by something else.
#[test]
fn learned_ordering_without_a_model_names_the_flag() {
    let tmp = tempfile::tempdir().unwrap();
    let creature = tmp.path().join("creature.json");
    std::fs::write(&creature, identity_creature_json(1, 1)).unwrap();
    let train = tmp.path().join("training");
    std::fs::create_dir_all(&train).unwrap();
    write_training(&train, 4);
    let out = bin()
        .arg(&creature)
        .arg(&train)
        .arg("--ordering")
        .arg("learned")
        .arg("--output-dir")
        .arg(tmp.path().join("out"))
        .output()
        .unwrap();
    assert!(!out.status.success());
    assert!(
        stderr(&out).contains("--ordering-model"),
        "{}",
        stderr(&out)
    );
}
