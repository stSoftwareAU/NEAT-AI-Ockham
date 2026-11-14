//! CLI tests for `audit-preservation` (Issue #264).
//!
//! The audit is report-only: it reads creatures and corpora, writes one JSON
//! report, and must leave the parent creature byte-for-byte unchanged.

use std::path::Path;
use std::process::Command;

use neat_ai_ockham::corpus::write_bin_file;
use neat_ai_ockham::fixtures::{creature, identity_creature, neuron, synapse};

fn bin() -> Command {
    Command::new(env!("CARGO_BIN_EXE_neat_ai_ockham"))
}

fn stdout(out: &std::process::Output) -> String {
    String::from_utf8_lossy(&out.stdout).into_owned()
}

fn stderr(out: &std::process::Output) -> String {
    String::from_utf8_lossy(&out.stderr).into_owned()
}

/// Two inputs and one target per record; the parent copies `input-0`, so its
/// output varies across the corpus.
fn write_corpus(dir: &Path, offset: usize, records: usize) {
    std::fs::create_dir_all(dir).unwrap();
    let recs: Vec<(Vec<f32>, Vec<f32>)> = (0..records)
        .map(|i| {
            let i = i + offset;
            let x0 = (i % 9) as f32 / 4.0 - 1.0;
            let x1 = ((i * 7) % 5) as f32 - 2.0;
            (vec![x0, x1], vec![x0])
        })
        .collect();
    write_bin_file(&dir.join("0.bin"), &recs).unwrap();
}

fn write_creature(path: &Path, creature_value: &neat_core::CreatureExport) {
    let json = neat_core::creature_to_json_pretty(creature_value).unwrap();
    std::fs::write(path, json).unwrap();
}

#[test]
fn audit_preservation_help_lists_out_and_says_report_only() {
    let out = bin()
        .args(["audit-preservation", "--help"])
        .output()
        .unwrap();
    assert!(out.status.success(), "{}", stderr(&out));
    let text = stdout(&out);
    assert!(text.contains("--out"), "{text}");
    assert!(text.to_lowercase().contains("report-only"), "{text}");
}

#[test]
fn a_missing_manifest_exits_nonzero_and_names_the_path() {
    let tmp = tempfile::tempdir().unwrap();
    let out = bin()
        .arg("audit-preservation")
        .arg(tmp.path().join("no-such-manifest.json"))
        .arg("--out")
        .arg(tmp.path().join("report.json"))
        .output()
        .unwrap();
    assert!(!out.status.success());
    assert!(
        stderr(&out).contains("no-such-manifest.json"),
        "{}",
        stderr(&out)
    );
}

#[test]
fn an_end_to_end_audit_writes_a_report_and_leaves_the_parent_untouched() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path();

    // Parent: IDENTITY of input-0 (two inputs, one output).
    let parent_path = root.join("parent.json");
    let parent = identity_creature(2, 1);
    write_creature(&parent_path, &parent);
    let parent_before = std::fs::read(&parent_path).unwrap();

    // Identical copies of the parent: genuinely better children (positive delta).
    write_creature(&root.join("same-a.json"), &parent);
    write_creature(&root.join("same-b.json"), &parent);

    // Perturbed child: half the weight, so the output shape is kept but scaled.
    let perturbed = creature(
        2,
        1,
        vec![neuron("output", "output-0", 0.0, Some("IDENTITY"))],
        vec![synapse("input-0", "output-0", 0.5)],
    );
    write_creature(&root.join("perturbed.json"), &perturbed);

    // Collapsed child: zero weight, so the output is constant on every record.
    let collapsed = creature(
        2,
        1,
        vec![neuron("output", "output-0", 0.0, Some("IDENTITY"))],
        vec![synapse("input-0", "output-0", 0.0)],
    );
    write_creature(&root.join("collapsed.json"), &collapsed);

    write_corpus(&root.join("calibration"), 0, 16);
    write_corpus(&root.join("validation"), 100, 16);

    let children = [
        ("same-a", "same-a.json", 0.01, 0.002, "calibration"),
        ("perturbed", "perturbed.json", -0.03, -0.01, "calibration"),
        ("same-b", "same-b.json", 0.02, 0.001, "validation"),
        ("collapsed", "collapsed.json", -0.05, -0.02, "validation"),
    ];
    let children_json: Vec<serde_json::Value> = children
        .iter()
        .map(|(name, file, full, sample, role)| {
            serde_json::json!({
                "name": name,
                "creature": file,
                "fullDelta": full,
                "sampleDelta": sample,
                "role": role,
            })
        })
        .collect();
    let manifest = serde_json::json!({
        "parent": "parent.json",
        "children": children_json,
        "calibration": "calibration",
        "validation": "validation",
        "shuffleSeeds": [1, 2, 3],
        "bootstrapSeed": 7,
        "bootstrapResamples": 200,
        "maxRecords": 16,
        "sampleTolerance": 0.0,
        "minAucGain": 0.05,
    });
    let manifest_path = root.join("manifest.json");
    std::fs::write(&manifest_path, manifest.to_string()).unwrap();

    let report_path = root.join("report.json");
    let out = bin()
        .arg("audit-preservation")
        .arg(&manifest_path)
        .arg("--out")
        .arg(&report_path)
        .output()
        .unwrap();
    assert!(out.status.success(), "{}", stderr(&out));

    let report: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&report_path).unwrap()).unwrap();
    assert_eq!(report["schema"], "ockham-preservation-audit/1");
    assert_eq!(
        report["coverage"]["childrenScored"].as_u64(),
        Some(children.len() as u64),
        "{report}"
    );

    assert_eq!(
        std::fs::read(&parent_path).unwrap(),
        parent_before,
        "the audit must never modify the parent creature"
    );
}
