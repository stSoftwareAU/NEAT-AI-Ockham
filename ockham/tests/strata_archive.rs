//! The retired #263 experiment keeps evidence, not an installed runtime feature.

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use neat_ai_ockham::incumbent::sha256_hex;
use serde_json::{Value, json};

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("..")
}

fn repo_json(path: &str) -> Value {
    serde_json::from_slice(&std::fs::read(root().join(path)).unwrap()).unwrap()
}

fn record() -> Value {
    repo_json("docs/experiments/strata-retirement-v1.json")
}

fn archived_report() -> Value {
    repo_json("docs/research/strata-pilot-v1/report.json")
}

fn installed(args: &[&str]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_neat_ai_ockham"))
        .args(args)
        .stdin(Stdio::null())
        .output()
        .unwrap()
}

#[test]
fn every_evidence_file_exists_and_matches_its_recorded_sha256() {
    let decision = record();
    let items = decision["evidence"].as_array().unwrap();
    let paths: Vec<&str> = items
        .iter()
        .map(|item| item["path"].as_str().unwrap())
        .collect();
    for required in [
        "docs/research/strata-pilot-v1/manifest.json",
        "docs/research/strata-pilot-v1/parent.json",
        "docs/research/strata-pilot-v1/report.json",
    ] {
        assert!(paths.contains(&required), "evidence omits {required}");
    }
    assert!(!paths.contains(&"docs/research/strata-pilot-v1/report.md"));
    let records = paths
        .iter()
        .filter(|path| path.ends_with("/records.bin"))
        .count();
    assert_eq!(records, 8);
    assert_eq!(items.len(), 11);
    for item in items {
        let bytes = std::fs::read(root().join(item["path"].as_str().unwrap())).unwrap();
        assert_eq!(item["sha256"], sha256_hex(&bytes));
    }
}

#[test]
fn archived_report_has_no_representative_coverage_and_tied_informed_arms() {
    let decision = record();
    let report = archived_report();
    assert_eq!(decision["representativeStrata"], 0);
    assert_eq!(decision["scientificVerdict"], "inconclusive-synthetic-only");
    assert_eq!(decision["pilots"][0]["verdict"], "inconclusive");
    assert!(
        report["evidence"]
            .as_str()
            .unwrap()
            .starts_with("fixture-or-unreviewed")
    );

    let criteria = report["criteria"].as_array().unwrap();
    for arm in ["worst-stratum", "pareto"] {
        let criterion = criteria.iter().find(|c| c["arm"] == arm).unwrap();
        assert_eq!(criterion["met"], false, "criterion for {arm}");
        assert_eq!(criterion["regressionReduction"], 0, "criterion for {arm}");
    }

    let runs = report["runs"].as_array().unwrap();
    assert_eq!(runs.len(), 40);
    for seed in 1_u64..=10 {
        for arm in ["aggregate", "worst-stratum", "pareto", "random"] {
            let matching = runs
                .iter()
                .filter(|run| run["arm"] == arm && run["seed"] == seed)
                .count();
            assert_eq!(matching, 1, "arm {arm} on seed {seed}");
        }
        for arm in ["aggregate", "worst-stratum", "pareto"] {
            let run = runs
                .iter()
                .find(|run| run["arm"] == arm && run["seed"] == seed)
                .unwrap();
            assert_eq!(
                run["selected"],
                json!(["d0", "d1", "d2", "d3"]),
                "arm {arm} on seed {seed}"
            );
        }
    }
}

#[test]
fn retired_command_is_absent_from_the_installed_binary() {
    let help = installed(&["--help"]);
    assert!(help.status.success());
    let text = String::from_utf8(help.stdout).unwrap();
    assert!(!text.contains("bench-strata"));
    assert!(text.contains("audit-preservation"));
}

#[test]
fn replay_names_the_pinned_commit_and_the_retired_command_is_gone_from_head() {
    let decision = record();
    let commit = decision["experimentSourceCommit"].as_str().unwrap();
    assert_eq!(commit.len(), 40);
    assert!(
        commit
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    );
    let replay = decision["replayCommand"].as_str().unwrap();
    assert!(replay.contains(commit));
    assert!(replay.contains("bench-strata"));
    assert!(!root().join("ockham/src/strata").exists());
}
