//! CLI: help/version plus the Issue #2 baseline gate.

use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::process::Command;

use neat_ai_ockham::corpus::write_bin_file;
use neat_ai_ockham::fixtures::{identity_creature_json, recurrent_flagged_creature_json};

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
    assert_eq!(report["optimisation"], "deferred");
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
