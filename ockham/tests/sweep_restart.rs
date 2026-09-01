//! Issue #77, end to end through the real binary: a run that exhausts its
//! sweep restarts it rather than idling out the rest of its budget.
//!
//! Asserted on the journalled batch records, never on wall-clock: a timing
//! assertion would pass on a fast machine and hide the spin this removes.
//!
//! Kept beside the in-crate `an_exhausted_sweep_restarts_rather_than_issuing_empty_batches`
//! rather than folded into it: this one is the reproduction, driving the shipped
//! binary and the real journal file end to end, which a unit test cannot.

use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::process::Command;

use neat_ai_ockham::corpus::write_bin_file;
use neat_ai_ockham::fixtures::{creature, neuron, synapse};

fn fake_scorer(dir: &Path) -> PathBuf {
    // Scores whatever creatures the run put in the cohort directory: the
    // incumbent baseline wins, every candidate loses, so nothing is accepted.
    let script = r#"#!/bin/sh
# The creature cohort directory is the second-to-last argument.
dir=""
prev=""
for arg in "$@"; do
  dir="$prev"
  prev="$arg"
done
printf '{'
first=1
for f in "$dir"/*.json; do
  stem=$(basename "$f" .json)
  case "$stem" in baseline) s=0.8; e=0.2;; *) s=0.1; e=0.9;; esac
  if [ $first -eq 0 ]; then printf ','; fi
  first=0
  printf '"%s":{"score":%s,"error":%s,"complexityPenalty":1e-8,"recordCount":2,"costName":"MSE","timeTaken":0.01}' "$stem" "$s" "$e"
done
printf '}\n'
"#;
    let path = dir.join("fake_scorer");
    std::fs::write(&path, script).unwrap();
    let mut perms = std::fs::metadata(&path).unwrap().permissions();
    perms.set_mode(0o755);
    std::fs::set_permissions(&path, perms).unwrap();
    path
}

#[test]
fn an_exhausted_sweep_restarts_rather_than_idling() {
    let tmp = tempfile::tempdir().unwrap();
    let c = creature(
        1,
        1,
        vec![
            neuron("hidden", "h_a", 0.0, Some("IDENTITY")),
            neuron("hidden", "h_b", 0.0, Some("IDENTITY")),
            neuron("output", "output-0", 0.0, Some("IDENTITY")),
        ],
        vec![
            synapse("input-0", "h_a", 1.0),
            synapse("input-0", "h_b", 1.0),
            synapse("h_a", "output-0", 1.0),
            synapse("h_b", "output-0", 1.0),
        ],
    );
    let path = tmp.path().join("creature.json");
    std::fs::write(&path, neat_core::creature_to_json_pretty(&c).unwrap()).unwrap();
    let train = tmp.path().join("train");
    std::fs::create_dir(&train).unwrap();
    write_bin_file(
        &train.join("0.bin"),
        &[(vec![1.0f32], vec![1.0f32]), (vec![2.0], vec![2.0])],
    )
    .unwrap();
    let out_dir = tmp.path().join("out");

    let out = Command::new(env!("CARGO_BIN_EXE_neat_ai_ockham"))
        .arg(&path)
        .arg(&train)
        .arg("--output-dir")
        .arg(&out_dir)
        .arg("--scorer")
        .arg(fake_scorer(tmp.path()))
        .arg("--learnings-dir")
        .arg(tmp.path().join("learnings"))
        .arg("--candidates")
        .arg("2")
        .arg("--max-experiments")
        .arg("4")
        .arg("--timeout-seconds")
        .arg("30")
        .arg("--seed")
        .arg("1")
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );

    let journal = std::fs::read_to_string(out_dir.join("experiments.jsonl")).unwrap();
    let records: Vec<serde_json::Value> = journal
        .lines()
        // A line that does not parse fails the test rather than vanishing from
        // the counts below.
        .map(|l| serde_json::from_str(l).expect("journal line is JSON"))
        .collect();
    let batches: Vec<&serde_json::Value> =
        records.iter().filter(|v| v["record"] == "batch").collect();
    assert_eq!(
        batches.len(),
        4,
        "the loop must keep filling batches: {batches:?}"
    );
    for b in &batches {
        assert_eq!(b["candidates"], 2, "an exhausted sweep must refill: {b}");
    }
    assert!(
        records.iter().any(|v| v["record"] == "sweepRestart"),
        "the sweep must journal its restart: {journal}"
    );
}
