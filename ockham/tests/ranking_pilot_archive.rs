//! The archived #261 ranking-v1 and ranking-v2 synthetic pilots keep their evidence, not a runtime feature.

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use neat_ai_ockham::incumbent::sha256_hex;
use serde_json::Value;

const V1: &str = "ranking-pilot-v1";
const V2: &str = "ranking-pilot-v2";

/// The directory of the archive called `name` under `docs/research`.
fn archive_root(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("docs/research")
        .join(name)
}

fn archive_bytes(name: &str, path: &str) -> Vec<u8> {
    std::fs::read(archive_root(name).join(path)).unwrap()
}

fn archive_evidence(name: &str, path: &str) -> Value {
    serde_json::from_slice(&archive_bytes(name, path)).unwrap()
}

/// The archived audit is gzipped to keep the repository small. Decompress it for its digest.
fn decompressed_audit_in(name: &str) -> Vec<u8> {
    let output = Command::new("gzip")
        .arg("-dc")
        .arg(archive_root(name).join("audit.json.gz"))
        .stdin(Stdio::null())
        .output()
        .unwrap();
    assert!(output.status.success());
    output.stdout
}

fn bytes(path: &str) -> Vec<u8> {
    archive_bytes(V1, path)
}

fn evidence(path: &str) -> Value {
    archive_evidence(V1, path)
}

fn decompressed_audit() -> Vec<u8> {
    decompressed_audit_in(V1)
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

#[test]
fn archived_v2_ranking_pilot_matches_its_digests_costs_and_path_signal() {
    let manifest = archive_evidence(V2, "manifest.json");
    let preflight = archive_evidence(V2, "preflight.json");
    let frozen = archive_evidence(V2, "frozen.json");
    let results = archive_evidence(V2, "results.json");
    let pilot = archive_evidence(V2, "pilot.json");
    let audit_bytes = decompressed_audit_in(V2);
    let audit: Value = serde_json::from_slice(&audit_bytes).unwrap();

    assert_eq!(manifest["protocol"], "ockham-ranking-v2");
    assert_eq!(results["verdict"], "no-demonstrated-benefit");
    assert_eq!(
        results["manifestSha256"],
        "767bfd5f57f610c78b57c7a4bd748926ea2fd5bc9f76b800dca49c045850a17c"
    );
    assert_eq!(results["manifestSha256"], frozen["manifestSha256"]);
    assert_eq!(results["manifestSha256"], preflight["manifestSha256"]);
    assert_eq!(results["auditSha256"], sha256_hex(&audit_bytes).as_str());
    assert_eq!(
        frozen["pilotSha256"],
        sha256_hex(&archive_bytes(V2, "pilot.json")).as_str()
    );

    let revision = manifest["codeRevision"].as_str().unwrap();
    assert_eq!(revision, "7cdabe6b89b65ee22a24151c7618b88c598136d9");
    assert!(
        revision.len() == 40
            && revision
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)),
        "codeRevision {revision} is 40 lowercase hex digits"
    );

    // Both the audit and the pilot are format version 2, the path-sensitivity protocol.
    assert_eq!(audit["formatVersion"], 2);
    assert_eq!(pilot["formatVersion"], 2);

    // The corpora are byte-identical to the v1 archive's committed records, and the source
    // histories match their manifest digests.
    let tasks = manifest["tasks"].as_array().unwrap();
    assert_eq!(tasks.len(), 8);
    for task in tasks {
        let id = task["id"].as_str().unwrap();
        if task["split"] == "source" {
            let history = archive_bytes(V2, &format!("{id}/history.jsonl"));
            assert_eq!(
                task["history"]["sha256"],
                sha256_hex(&history).as_str(),
                "history digest of {id}"
            );
        }
        let corpus = archive_bytes(V1, &format!("{id}/corpus/records.bin"));
        assert_eq!(
            task["corpusSha256"],
            sha256_hex(&corpus).as_str(),
            "corpus digest of {id}"
        );
    }

    // Every audit row with features carries a finite, non-negative path sensitivity. The
    // rows without features are the 8 generation failures and carry none.
    let rows: Vec<&Value> = audit["pools"]
        .as_array()
        .unwrap()
        .iter()
        .flat_map(|pool| pool["rows"].as_array().unwrap().iter())
        .collect();
    assert_eq!(rows.len(), 640);
    let mut with_path = 0;
    for &row in &rows {
        if row["features"].is_null() {
            assert!(row.get("pathSensitivity").is_none(), "{row}");
            continue;
        }
        let path = row["pathSensitivity"]
            .as_f64()
            .unwrap_or_else(|| panic!("{row} lacks a pathSensitivity"));
        assert!(path.is_finite() && path >= 0.0, "{row}");
        with_path += 1;
    }
    assert_eq!(with_path, 632);

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

    let coverage = results["coverage"].as_array().unwrap();
    assert_eq!(coverage.len(), 16);
    for cell in coverage {
        assert_eq!(cell["complete"], true, "{cell}");
    }
    let total = |key: &str| -> u64 { coverage.iter().map(|cell| uint(&cell[key])).sum() };
    assert_eq!(total("selected"), 640);
    assert_eq!(total("generated"), 632);
    assert_eq!(total("fullyScored"), 632);
    assert_eq!(total("missingFeatures"), 0);

    // The ten charged stages sum to the research total (50,574 ms in this archive).
    let stages = [
        "auditCostMs",
        "pilotCostMs",
        "freezingCostMs",
        "historicalAcquisitionMs",
        "preparationCostMs",
        "preflightCostMs",
        "evaluationCostMs",
        "buildCostMs",
        "serialisationCostMs",
        "recordedWriteCostMs",
    ];
    let charged: u64 = stages.iter().map(|key| uint(&results[*key])).sum();
    assert_eq!(uint(&results["totalResearchMs"]), charged);
    assert_eq!(charged, 50_574);

    // The build is the bootstrap build of `ranking_pilot` plus the timed remaining build.
    let build = uint(&results["buildCostMs"]);
    assert_eq!(build, uint(&manifest["buildMs"]));
    assert_eq!(
        build,
        recorded_ms_in(V2, "build-ms") + recorded_ms_in(V2, "bootstrap-build-ms")
    );
    assert_eq!(build, 47_570);

    // The bench charges the preflight, freeze and audit sidecars and the manifest tail.
    let written = recorded_ms_in(V2, "preflight.json.write-ms")
        + recorded_ms_in(V2, "frozen.json.write-ms")
        + recorded_ms_in(V2, "audit.json.write-ms")
        + recorded_ms_in(V2, "manifest-tail-ms");
    assert_eq!(uint(&results["recordedWriteCostMs"]), written);
    assert_eq!(written, 6);

    // The pilot cost is the freeze's copy and includes the pilot file's own write.
    let pilot_ms = uint(&results["pilotCostMs"]);
    assert_eq!(pilot_ms, uint(&frozen["pilotCostMs"]));
    assert!(pilot_ms >= recorded_ms_in(V2, "pilot.json.write-ms"));

    assert_eq!(
        uint(&results["preflightCostMs"]),
        uint(&preflight["checkMs"])
    );

    // Each source's acquisition is its recorded search time plus curation.
    let mut acquisition = 0;
    for task in tasks {
        if task["split"] != "source" {
            continue;
        }
        let id = task["id"].as_str().unwrap();
        let charged = uint(&task["history"]["acquisitionMs"]);
        let recorded = recorded_ms_in(V2, &format!("{id}/acquisition-ms"));
        assert!(
            charged >= recorded,
            "{id} acquisitionMs {charged} includes the recorded {recorded} ms"
        );
        acquisition += charged;
    }
    assert_eq!(uint(&results["historicalAcquisitionMs"]), acquisition);
}

/// A non-negative integer field of the archive.
fn uint(value: &Value) -> u64 {
    value
        .as_u64()
        .unwrap_or_else(|| panic!("{value} is not a u64"))
}

/// The integer recorded in the archived text file `path`.
fn recorded_ms(path: &str) -> u64 {
    recorded_ms_in(V1, path)
}

/// The integer recorded in the text file `path` of the archive called `name`.
fn recorded_ms_in(name: &str, path: &str) -> u64 {
    String::from_utf8(archive_bytes(name, path))
        .unwrap()
        .trim()
        .parse()
        .unwrap()
}
