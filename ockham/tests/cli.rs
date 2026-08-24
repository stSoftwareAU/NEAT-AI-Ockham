//! CLI skeleton: help, version, usage, and configuration reporting.

use std::path::Path;
use std::process::Command;

fn bin() -> Command {
    Command::new(env!("CARGO_BIN_EXE_neat_ai_ockham"))
}

fn stdout(out: &std::process::Output) -> String {
    String::from_utf8_lossy(&out.stdout).into_owned()
}

fn stderr(out: &std::process::Output) -> String {
    String::from_utf8_lossy(&out.stderr).into_owned()
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
fn reports_configuration_without_optimising() {
    let tmp = tempfile::tempdir().unwrap();
    let creature = tmp.path().join("creature.json");
    let training = tmp.path().join("training");
    std::fs::write(&creature, "{}").unwrap();
    std::fs::create_dir(&training).unwrap();

    let out = bin()
        .arg(&creature)
        .arg(&training)
        .arg("--timeout-seconds")
        .arg("2700")
        .arg("--candidates")
        .arg("100")
        .arg("--screen-sample-rate")
        .arg("0.05")
        .arg("--seed")
        .arg("7")
        .output()
        .unwrap();
    assert!(out.status.success(), "{}", stderr(&out));
    let err = stderr(&out);
    assert!(err.contains("configuration only"));
    assert!(err.contains("not attempted yet"));

    let report: serde_json::Value = serde_json::from_str(&stdout(&out)).expect("json report");
    assert_eq!(report["optimisation"], "deferred");
    assert_eq!(report["timeoutSeconds"], 2700);
    assert_eq!(report["candidates"], 100);
    assert_eq!(report["screenSampleRate"], 0.05);
    assert_eq!(report["seed"], 7);
    assert_eq!(
        Path::new(report["creature"].as_str().unwrap()),
        creature.as_path()
    );
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
