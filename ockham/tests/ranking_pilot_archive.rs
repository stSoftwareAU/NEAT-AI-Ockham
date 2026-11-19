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

    // Preparation (input generation plus the manifest step) is charged to the total.
    let preparation = uint(&results["preparationCostMs"]);
    assert_eq!(preparation, uint(&manifest["preparationMs"]));
    assert!(preparation > 0, "preparation cost is charged");
    assert!(
        uint(&manifest["preparationMs"]) >= recorded_ms("inputs-ms"),
        "preparationMs includes the recorded input generation time"
    );
    let charged: u64 = [
        "auditCostMs",
        "pilotCostMs",
        "freezingCostMs",
        "historicalAcquisitionMs",
        "preparationCostMs",
        "preflightCostMs",
        "evaluationCostMs",
    ]
    .iter()
    .map(|key| uint(&results[*key]))
    .sum();
    assert_eq!(uint(&results["totalResearchMs"]), charged);

    // The preflight `check` stage records its own time, and evaluation charges it.
    let preflight_ms = uint(&results["preflightCostMs"]);
    assert_eq!(preflight_ms, uint(&preflight["checkMs"]));
    assert!(preflight_ms > 0, "preflight cost is charged");

    // Each source's acquisition is its recorded search time plus curation.
    let mut acquisition = 0;
    for task in manifest["tasks"].as_array().unwrap() {
        if task["split"] != "source" {
            continue;
        }
        let id = task["id"].as_str().unwrap();
        let charged = uint(&task["history"]["acquisitionMs"]);
        let recorded = recorded_ms(&format!("{id}/acquisition-ms"));
        assert!(
            charged >= recorded,
            "{id} acquisitionMs {charged} includes the recorded {recorded} ms"
        );
        acquisition += charged;
    }
    assert_eq!(uint(&results["historicalAcquisitionMs"]), acquisition);

    // Neither `frozen.json` nor `preflight.json` carries a code revision, so only
    // the manifest's is checked.
    let revision = manifest["codeRevision"].as_str().unwrap();
    assert_eq!(revision.len(), 40, "codeRevision {revision}");
    assert!(
        revision
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)),
        "codeRevision {revision} is lowercase hex"
    );

    let coverage = results["coverage"].as_array().unwrap();
    let total = |key: &str| -> u64 { coverage.iter().map(|cell| uint(&cell[key])).sum() };
    assert_eq!(total("selected"), 640);
    assert_eq!(total("generated"), 632);
    assert_eq!(total("fullyScored"), 632);
    // Synapse candidates carry their own feature vectors, so no generated
    // candidate is ranked without features.
    assert_eq!(total("missingFeatures"), 0);
    let synapses: u64 = coverage
        .iter()
        .map(|cell| uint(&cell["kinds"]["synapse"]))
        .sum();
    assert_eq!(synapses, 496, "synapse candidates audited with features");
}

/// A non-negative integer field of the archive.
fn uint(value: &Value) -> u64 {
    value
        .as_u64()
        .unwrap_or_else(|| panic!("{value} is not a u64"))
}

/// The integer recorded in the archived text file `path`.
fn recorded_ms(path: &str) -> u64 {
    String::from_utf8(bytes(path))
        .unwrap()
        .trim()
        .parse()
        .unwrap()
}
