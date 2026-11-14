//! The opt-in experiment's real command boundary.

use std::path::Path;
use std::process::{Command, Output};

use neat_ai_ockham::corpus::write_bin_file;
use neat_ai_ockham::fixtures::wide_creature;
use neat_ai_ockham::incumbent::sha256_hex;
use serde_json::Value;

fn cli(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_neat_ai_ockham"))
        .args(args)
        .stdin(std::process::Stdio::null())
        .output()
        .unwrap()
}

fn prepare(root: &Path) -> (std::path::PathBuf, String) {
    let creature = root.join("parent.json");
    std::fs::write(
        &creature,
        neat_core::creature_to_json(&wide_creature(1, 4, "TANH")).unwrap(),
    )
    .unwrap();
    let data = root.join("data");
    std::fs::create_dir(&data).unwrap();
    write_bin_file(
        &data.join("0.bin"),
        &(0..16)
            .map(|i| (vec![i as f32 / 16.0], vec![0.2]))
            .collect::<Vec<_>>(),
    )
    .unwrap();
    let manifest = root.join("manifest.json");
    let out = cli(&[
        "soft-gate", "prepare", creature.to_str().unwrap(), data.to_str().unwrap(),
        "--scorer", "/usr/bin/false", "--out", manifest.to_str().unwrap(),
        "--calibration-records", "8", "--validation-records", "4",
    ]);
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let report: Value = serde_json::from_slice(&out.stdout).unwrap();
    let bytes = std::fs::read(&manifest).unwrap();
    assert_eq!(report["manifestSha256"], sha256_hex(&bytes));
    let value: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(value["formatVersion"], 1);
    assert_eq!(value["targets"], serde_json::json!([1.0, 0.9, 0.75, 0.5]));
    (manifest, sha256_hex(&bytes))
}

#[test]
fn preparation_locks_inputs_and_run_refuses_manifest_drift() {
    let root = tempfile::tempdir().unwrap();
    let (manifest, digest) = prepare(root.path());
    let mut bytes = std::fs::read(&manifest).unwrap();
    bytes.push(b' ');
    std::fs::write(&manifest, bytes).unwrap();
    let out = cli(&[
        "soft-gate", "run", manifest.to_str().unwrap(), "--manifest-sha256", &digest,
        "--out", root.path().join("results").to_str().unwrap(),
    ]);
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("manifest checksum mismatch"));
    assert!(!root.path().join("results").exists());
}

#[test]
fn scorer_failure_is_a_failed_experiment_with_a_durable_record() {
    let root = tempfile::tempdir().unwrap();
    let (manifest, _) = prepare(root.path());
    let mut value: Value = serde_json::from_slice(&std::fs::read(&manifest).unwrap()).unwrap();
    value["seeds"] = serde_json::json!([7]);
    value["fit"]["steps"] = 1.into();
    let bytes = serde_json::to_vec_pretty(&value).unwrap();
    std::fs::write(&manifest, &bytes).unwrap();
    let out = cli(&[
        "soft-gate", "run", manifest.to_str().unwrap(), "--manifest-sha256", &sha256_hex(&bytes),
        "--out", root.path().join("results").to_str().unwrap(),
    ]);
    assert!(!out.status.success());
    let report: Value = serde_json::from_slice(
        &std::fs::read(root.path().join("results/results.json")).unwrap(),
    ).unwrap();
    assert_eq!(report["status"], "failed");
    assert!(report["failures"][0].as_str().unwrap().contains("scorer failed"));
    assert!(!root.path().join("results/best.json").exists());
}
