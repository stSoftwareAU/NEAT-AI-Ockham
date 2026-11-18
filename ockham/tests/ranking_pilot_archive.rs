//! The archived #261 ranking-v1 synthetic pilot keeps its evidence, not a runtime feature.

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use neat_ai_ockham::incumbent::sha256_hex;
use serde_json::Value;

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("docs/research/ranking-pilot-v1")
}

fn bytes(path: &str) -> Vec<u8> {
    std::fs::read(root().join(path)).unwrap()
}

fn evidence(path: &str) -> Value {
    serde_json::from_slice(&bytes(path)).unwrap()
}

/// The archived audit is gzipped to keep the repository small. Decompress it for its digest.
fn decompressed_audit() -> Vec<u8> {
    let output = Command::new("gzip")
        .arg("-dc")
        .arg(root().join("audit.json.gz"))
        .stdin(Stdio::null())
        .output()
        .unwrap();
    assert!(output.status.success());
    output.stdout
}

#[test]
fn archived_ranking_pilot_matches_its_digests_and_negative_finding() {
    let manifest = evidence("manifest.json");
    let preflight = evidence("preflight.json");
    let frozen = evidence("frozen.json");
    let results = evidence("results.json");

    assert_eq!(results["verdict"], "no-demonstrated-benefit");
    assert_eq!(results["manifestSha256"], frozen["manifestSha256"]);
    assert_eq!(results["manifestSha256"], preflight["manifestSha256"]);
    assert_eq!(
        results["auditSha256"],
        sha256_hex(&decompressed_audit()).as_str()
    );
    assert_eq!(
        frozen["pilotSha256"],
        sha256_hex(&bytes("pilot.json")).as_str()
    );

    for task in manifest["tasks"].as_array().unwrap() {
        if task["split"] != "source" {
            continue;
        }
        let id = task["id"].as_str().unwrap();
        let history = bytes(&format!("{id}/history.jsonl"));
        assert_eq!(
            task["history"]["sha256"],
            sha256_hex(&history).as_str(),
            "history digest of {id}"
        );
    }

    let primary = results["paired"]
        .as_array()
        .unwrap()
        .iter()
        .find(|cell| {
            cell["strategy"] == "transfer-pairwise"
                && cell["control"] == "partial"
                && cell["calibrationBudget"] == 8
                && cell["historyFraction"] == 1.0
                && cell["evaluationBudget"] == 5
        })
        .expect("the primary transfer-pairwise versus partial cell is archived");
    assert!(primary["meanRecallDelta"].as_f64().unwrap() < 0.0);
    assert!(primary["recallDeltaCi95"][1].as_f64().unwrap() < 0.0);

    for coverage in results["coverage"].as_array().unwrap() {
        assert_eq!(coverage["complete"], true, "{coverage}");
    }
}
