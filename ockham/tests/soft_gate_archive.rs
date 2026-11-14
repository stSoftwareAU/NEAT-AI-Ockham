//! The retired #262 experiment keeps evidence, not an installed runtime feature.

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use neat_ai_ockham::incumbent::sha256_hex;
use serde_json::Value;

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("..")
}

fn evidence(name: &str) -> Value {
    serde_json::from_slice(&std::fs::read(root().join("docs/experiments").join(name)).unwrap())
        .unwrap()
}

#[test]
fn archived_negative_result_and_coverage_match_the_retirement_decision() {
    let decision = evidence("soft-gate-retirement-v1.json");
    let results = evidence("soft-gate-smoke-results-v1.json");
    let manifest = evidence("soft-gate-smoke-manifest-v1.json");
    let preflight = evidence("soft-gate-smoke-preflight-results-v1.json");
    assert_eq!(
        decision["maintenanceDecision"],
        "remove-optional-runtime-implementation"
    );
    assert_eq!(decision["representativeCreatures"], 0);
    assert_eq!(decision["representativePairs"], 0);
    assert_eq!(results["representative"], false);
    assert_eq!(results["coverage"]["representativeCreatures"], 0);
    assert_eq!(results["corpusRecords"], decision["syntheticRecords"]);
    assert_eq!(
        results["coverage"]["calibrationRecords"],
        decision["calibrationRecords"]
    );
    assert_eq!(
        results["coverage"]["validationRecords"],
        decision["validationRecords"]
    );
    assert_eq!(
        results["pairedSummary"]["completePairs"],
        decision["completedSyntheticPairs"]
    );
    assert_eq!(results["pairedSeedsRequested"], manifest["seeds"]);
    let deltas = results["pairedSummary"]["pairedScoreDeltas"]
        .as_array()
        .unwrap();
    assert_eq!(deltas.len(), manifest["seeds"].as_array().unwrap().len());
    let mean = deltas.iter().map(|v| v.as_f64().unwrap()).sum::<f64>() / deltas.len() as f64;
    assert!((mean - results["pairedSummary"]["meanScoreDelta"].as_f64().unwrap()).abs() < 1e-12);
    assert!(
        results["pairedSummary"]["conservative95Interval"][1]
            .as_f64()
            .unwrap()
            < 0.0
    );
    for criterion in ["heldOutError", "latencyReduction", "representativeData"] {
        assert_eq!(results["pairedSummary"]["criteria"][criterion], false);
    }
    assert_eq!(preflight["status"], "failed");
    assert!(!preflight["failures"].as_array().unwrap().is_empty());
    for item in decision["evidence"].as_array().unwrap() {
        let bytes = std::fs::read(
            root()
                .join("docs/experiments")
                .join(item["path"].as_str().unwrap()),
        )
        .unwrap();
        assert_eq!(item["sha256"], sha256_hex(&bytes));
    }
}

#[test]
fn historical_replay_preserves_the_paired_distribution_and_reports_zero_real_coverage() {
    let replay = evidence("soft-gate-replay-verification-v1.json");
    let original = evidence("soft-gate-smoke-results-v1.json");
    let decision = evidence("soft-gate-retirement-v1.json");
    assert_eq!(
        replay["experimentSourceCommit"],
        decision["experimentSourceCommit"]
    );
    assert_eq!(replay["observedExitCode"], 0);
    assert_eq!(replay["status"], "completed");
    assert_eq!(replay["representative"], false);
    assert_eq!(replay["coverage"], original["coverage"]);
    assert_eq!(replay["coverage"]["completedFamilies"], 20);
    assert_eq!(replay["failures"], original["failures"]);
    for field in [
        "pairedScoreDeltas",
        "completePairs",
        "conservative95Interval",
    ] {
        assert_eq!(
            replay["pairedSummary"][field],
            original["pairedSummary"][field]
        );
    }
    for (path, checksum) in [
        ("soft-gate-smoke-manifest-v1.json", "archivedManifestSha256"),
        ("soft-gate-replay-manifest-v1.json", "replayManifestSha256"),
    ] {
        let bytes = std::fs::read(root().join("docs/experiments").join(path)).unwrap();
        assert_eq!(replay[checksum], sha256_hex(&bytes));
    }
}

#[test]
fn retired_experiment_is_absent_from_the_installed_command() {
    let output = Command::new(env!("CARGO_BIN_EXE_neat_ai_ockham"))
        .arg("--help")
        .stdin(Stdio::null())
        .output()
        .unwrap();
    assert!(output.status.success());
    let help = String::from_utf8(output.stdout).unwrap();
    assert!(!help.contains("soft-gate"));
    assert!(help.contains("train-ordering"));
    assert!(help.contains("report"));
}

#[test]
fn historical_replay_refuses_invalid_inputs_without_changing_outputs() {
    let temp = tempfile::tempdir().unwrap();
    let script = root().join("scripts/replay-soft-gate-smoke.sh");
    let run = |args: &[&std::ffi::OsStr]| {
        Command::new("bash")
            .arg(&script)
            .args(args)
            .stdin(Stdio::null())
            .output()
            .unwrap()
    };
    assert_eq!(run(&[]).status.code(), Some(2));
    let missing = temp.path().join("missing-scorer");
    let output = temp.path().join("new-output");
    let result = run(&[missing.as_os_str(), output.as_os_str()]);
    assert_eq!(result.status.code(), Some(2));
    assert!(!output.exists());
    std::fs::create_dir(&output).unwrap();
    std::fs::write(output.join("existing"), "keep").unwrap();
    let result = run(&[Path::new("/bin/false").as_os_str(), output.as_os_str()]);
    assert_eq!(result.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&result.stderr).contains("output already exists"));
    assert_eq!(
        std::fs::read_to_string(output.join("existing")).unwrap(),
        "keep"
    );
}
