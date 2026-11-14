//! Writes application-agnostic synthetic inputs for a bench-strata pilot (issue #263). See docs/strata-experiment.md.
//!
//! Usage: `cargo run --release --example strata_pilot -- <new-directory>`

use std::fs::OpenOptions;
use std::io::Write;
use std::path::Path;

use neat_ai_ockham::corpus::write_bin_file;
use neat_ai_ockham::fixtures::{creature, neuron, synapse};
use neat_ai_ockham::strata::manifest::load;
use neat_core::{CreatureExport, compile_creature, creature_to_json_pretty};
use serde_json::json;
use sha2::{Digest, Sha256};

/// Calibration and validation stratum names, in manifest order.
const STRATA: [&str; 4] = ["alpha", "beta", "gamma", "rare"];

/// Records per split for each stratum in `STRATA`; `rare` is the minority.
const RECORDS: [usize; 4] = [192, 192, 192, 48];

/// `(directory, split index)` for calibration and validation.
const SPLITS: [(&str, usize); 2] = [("cal", 0), ("val", 1)];

fn draw(a: usize, b: usize, c: usize) -> f64 {
    let mut hasher = Sha256::new();
    hasher.update(b"strata-pilot-synthetic-v1");
    for value in [a, b, c] {
        hasher.update((value as u64).to_le_bytes());
    }
    let digest = hasher.finalize();
    f64::from(u32::from_le_bytes(digest[..4].try_into().unwrap())) / f64::from(u32::MAX) * 2.0 - 1.0
}

/// Parent creature: four inputs, one output, specialist, shared and dead-wood hidden neurons.
fn parent_creature() -> CreatureExport {
    let mut neurons = Vec::new();
    let mut edges = Vec::new();
    for k in 0..4 {
        let hk = format!("h{k}");
        neurons.push(neuron("hidden", &hk, draw(2, k, 0) * 0.1, Some("TANH")));
        for j in 0..4 {
            let weight = if j == k { 1.5 } else { draw(1, k, j) * 0.05 };
            edges.push(synapse(&format!("input-{j}"), &hk, weight));
        }
        let to_output = if k == 3 { 0.8 } else { 0.5 };
        edges.push(synapse(&hk, "output-0", to_output));
    }
    for s in 0..2 {
        let name = format!("s{s}");
        neurons.push(neuron("hidden", &name, 0.0, Some("TANH")));
        for j in 0..4 {
            edges.push(synapse(&format!("input-{j}"), &name, draw(3, s, j)));
        }
        edges.push(synapse(&name, "output-0", 0.3));
    }
    for d in 0..4 {
        let name = format!("d{d}");
        neurons.push(neuron("hidden", &name, 0.0, Some("TANH")));
        for j in 0..4 {
            edges.push(synapse(&format!("input-{j}"), &name, draw(4, d, j)));
        }
        edges.push(synapse(&name, "output-0", 1e-6 * (1 + d) as f64));
    }
    for j in 0..4 {
        edges.push(synapse(
            &format!("input-{j}"),
            "output-0",
            draw(5, j, 0) * 0.1,
        ));
    }
    neurons.push(neuron("output", "output-0", 0.0, Some("IDENTITY")));
    creature(4, 1, neurons, edges)
}

/// The parent with every near-zero synapse removed; its outputs are the targets.
fn teacher_of(parent: &CreatureExport) -> CreatureExport {
    let mut teacher = parent.clone();
    for edge in &mut teacher.synapses {
        if edge.weight.abs() < 1e-4 {
            edge.weight = 0.0;
        }
    }
    teacher
}

/// Input for record `row` of stratum `stratum` in split `split`: input `stratum` is full scale.
fn record_input(stratum: usize, split: usize, row: usize) -> Vec<f32> {
    (0..4)
        .map(|j| {
            let scale = if j == stratum { 1.0 } else { 0.1 };
            (draw(10 + stratum + 4 * split, row, j) * scale) as f32
        })
        .collect()
}

fn make_dir(path: &Path) -> Result<(), String> {
    std::fs::create_dir(path).map_err(|e| format!("cannot create '{}': {e}", path.display()))
}

/// Create `path` and write `bytes` into it, refusing to overwrite anything.
fn write_new_file(path: &Path, bytes: &[u8]) -> Result<(), String> {
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|e| format!("cannot create '{}': {e}", path.display()))?;
    file.write_all(bytes)
        .map_err(|e| format!("cannot write '{}': {e}", path.display()))
}

/// Write the synthetic pilot inputs into the new directory `dir`.
///
/// # Errors
///
/// Refuses an existing `dir` (nothing inside it is touched), and reports any
/// I/O, creature or manifest failure with the path at fault.
pub fn write(dir: &Path) -> Result<(), String> {
    make_dir(dir).map_err(|e| format!("new pilot directory: {e}"))?;

    let parent = parent_creature();
    let teacher = teacher_of(&parent);
    let mut net = compile_creature(&teacher).map_err(|e| e.to_string())?;
    let parent_json = creature_to_json_pretty(&parent).map_err(|e| e.to_string())?;
    write_new_file(&dir.join("parent.json"), parent_json.as_bytes())?;

    for (split_dir, split) in SPLITS {
        let split_path = dir.join(split_dir);
        make_dir(&split_path)?;
        for (stratum, name) in STRATA.iter().enumerate() {
            let stratum_path = split_path.join(name);
            make_dir(&stratum_path)?;
            let rows: Vec<(Vec<f32>, Vec<f32>)> = (0..RECORDS[stratum])
                .map(|row| {
                    let input = record_input(stratum, split, row);
                    let target = net.activate(&input, 1);
                    (input, target)
                })
                .collect();
            let bin_path = stratum_path.join("records.bin");
            write_bin_file(&bin_path, &rows)
                .map_err(|e| format!("cannot write '{}': {e}", bin_path.display()))?;
        }
    }

    let strata_for = |split_dir: &str| -> Vec<serde_json::Value> {
        STRATA
            .iter()
            .map(|name| json!({"name": name, "dir": format!("{split_dir}/{name}")}))
            .collect()
    };
    let seeds: Vec<u64> = (1..=10).collect();
    let manifest = json!({
        "parent": "parent.json",
        "calibration": strata_for("cal"),
        "validation": strata_for("val"),
        "seeds": seeds,
        "fitSeed": 9001,
        "temperatures": [0.25, 0.5, 1.0, 2.0, 4.0],
        "bundleSizes": [2, 4],
        "bundlesPerArm": 6,
        "paretoCandidates": 24,
        "archiveCap": 8,
        "latencyTolerance": 0.05,
        "latencyRecords": 48,
        "statsRecords": 0,
        "minImprovement": 0.0,
        "regressionTolerance": 0.000001,
        "success": {"maxFitnessLoss": 0.0, "minRegressionReduction": 1}
    });
    let manifest_text = serde_json::to_string_pretty(&manifest).map_err(|e| e.to_string())?;
    let manifest_path = dir.join("manifest.json");
    write_new_file(&manifest_path, format!("{manifest_text}\n").as_bytes())?;

    load(&manifest_path)
        .map(|_| ())
        .map_err(|e| format!("written manifest is invalid: {e}"))
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [directory] = args.as_slice() else {
        eprintln!("usage: cargo run --release --example strata_pilot -- <new-directory>");
        std::process::exit(2);
    };
    if let Err(message) = write(Path::new(directory)) {
        eprintln!("error: {message}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use neat_ai_ockham::corpus::{corpus_info, for_each_chunk};
    use neat_ai_ockham::strata::manifest::StratumDir;
    use neat_core::{parse_creature_json, training_data::TrainingDataConfig};

    #[test]
    fn two_fresh_directories_are_byte_identical_and_load() {
        let tmp = tempfile::tempdir().unwrap();
        let first = tmp.path().join("first");
        let second = tmp.path().join("second");
        write(&first).unwrap();
        write(&second).unwrap();

        let mut files = vec!["manifest.json".to_string(), "parent.json".to_string()];
        for (split_dir, _) in SPLITS {
            for name in STRATA {
                files.push(format!("{split_dir}/{name}/records.bin"));
            }
        }
        for file in &files {
            assert_eq!(
                std::fs::read(first.join(file)).unwrap(),
                std::fs::read(second.join(file)).unwrap(),
                "{file} differs between two writes"
            );
        }

        // The held-out split is generated from its own seeds, not copied from calibration.
        for name in STRATA {
            assert_ne!(
                std::fs::read(first.join(format!("val/{name}/records.bin"))).unwrap(),
                std::fs::read(first.join(format!("cal/{name}/records.bin"))).unwrap(),
                "val/{name}/records.bin is a copy of cal/{name}/records.bin"
            );
        }

        // The committed pilot record must still be exactly what the generator produces.
        let committed =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../docs/research/strata-pilot-v1");
        for file in ["manifest.json", "parent.json"] {
            assert_eq!(
                std::fs::read(first.join(file)).unwrap(),
                std::fs::read(committed.join(file)).unwrap(),
                "committed pilot record no longer matches the generator: {file} differs from docs/research/strata-pilot-v1/{file}"
            );
        }

        let loaded = load(&first.join("manifest.json")).unwrap();
        assert_eq!(loaded.manifest.calibration.len(), 4);
        let config = TrainingDataConfig::new(4, 1);
        let count_of = |strata: &[StratumDir], name: &str| {
            let stratum = strata
                .iter()
                .find(|s| s.name == name)
                .unwrap_or_else(|| panic!("no stratum '{name}'"));
            corpus_info(&stratum.dir, &config).unwrap().record_count
        };
        assert_eq!(count_of(&loaded.manifest.calibration, "rare"), 48);
        assert_eq!(count_of(&loaded.manifest.calibration, "alpha"), 192);
        for (label, strata) in [
            ("calibration", &loaded.manifest.calibration),
            ("validation", &loaded.manifest.validation),
        ] {
            for (name, expected) in [("alpha", 192), ("beta", 192), ("gamma", 192), ("rare", 48)] {
                assert_eq!(
                    count_of(strata, name),
                    expected,
                    "{label} '{name}' record count"
                );
            }
        }

        let parent_text = std::fs::read_to_string(first.join("parent.json")).unwrap();
        let parent = parse_creature_json(&parent_text).unwrap();
        let hidden = parent
            .neurons
            .iter()
            .filter(|n| n.neuron_type == "hidden")
            .count();
        assert_eq!(hidden, 10, "parent should have 10 hidden neurons");
        let dead_wood: Vec<_> = parent
            .synapses
            .iter()
            .filter(|s| s.to_uuid == "output-0" && (0..4).any(|d| s.from_uuid == format!("d{d}")))
            .collect();
        assert_eq!(dead_wood.len(), 4, "every d0..d3 should feed output-0");
        for edge in dead_wood {
            assert!(
                edge.weight.abs() < 1e-4,
                "dead-wood synapse {} -> {} has weight {}",
                edge.from_uuid,
                edge.to_uuid,
                edge.weight
            );
        }
        let h3 = parent
            .synapses
            .iter()
            .find(|s| s.from_uuid == "h3" && s.to_uuid == "output-0")
            .expect("h3 -> output-0 synapse");
        assert!(
            (h3.weight - 0.8).abs() < 1e-12,
            "h3 -> output-0 weight is {}, expected 0.8",
            h3.weight
        );
    }

    #[test]
    fn rare_calibration_is_full_scale_in_input_three_only() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join("pilot");
        write(&dir).unwrap();

        let mut max_abs = [0f32; 4];
        let visited = for_each_chunk(
            &dir.join("cal").join("rare"),
            &TrainingDataConfig::new(4, 1),
            16,
            |chunk| {
                for row in chunk.inputs.as_chunks::<4>().0 {
                    for (max, value) in max_abs.iter_mut().zip(row) {
                        *max = max.max(value.abs());
                    }
                }
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(visited, 48, "calibration rare record count");
        assert!(
            max_abs[3] > 0.5,
            "rare input 3 should be full scale, max |value| is {}",
            max_abs[3]
        );
        for (input, max) in max_abs.iter().enumerate().take(3) {
            assert!(
                *max <= 0.1,
                "rare input {input} should be scaled down, max |value| is {max}"
            );
        }
    }

    #[test]
    fn existing_directory_is_refused_and_left_untouched() {
        let tmp = tempfile::tempdir().unwrap();
        let existing = tmp.path().join("existing");
        std::fs::create_dir(&existing).unwrap();
        let sentinel = existing.join("keep.txt");
        std::fs::write(&sentinel, b"do not touch").unwrap();

        let err = write(&existing).expect_err("an existing directory must be refused");
        assert!(
            err.contains(&existing.display().to_string()),
            "error should name the path: {err}"
        );
        assert_eq!(std::fs::read(&sentinel).unwrap(), b"do not touch");
        assert!(!existing.join("manifest.json").exists());
    }

    /// Read every record of the corpus in `dir` as `(inputs, targets)`.
    fn read_records(dir: &Path) -> (Vec<Vec<f32>>, Vec<f32>) {
        let config = TrainingDataConfig::new(4, 1);
        let mut inputs: Vec<Vec<f32>> = Vec::new();
        let mut targets: Vec<f32> = Vec::new();
        for_each_chunk(dir, &config, 16, |chunk| {
            inputs.extend(
                chunk
                    .inputs
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|row| row.to_vec()),
            );
            targets.extend_from_slice(chunk.targets);
            Ok(())
        })
        .unwrap();
        (inputs, targets)
    }

    /// Largest difference allowed between a committed target and one computed
    /// on this host. Targets go through the host libm `tanhf`, whose last bits
    /// differ between libm builds. Every target is below 1.0 in magnitude, and
    /// an older glibc libm moved them by at most 1.2e-7 (about two ulps). In every
    /// stratum the largest teacher-versus-parent gap is at least 1.79e-6, above
    /// the 4 x TARGET_TOLERANCE (1.2e-6) the test requires.
    const TARGET_TOLERANCE: f32 = 3e-7;

    #[test]
    fn committed_corpora_match_the_record_and_regenerated_targets() {
        let committed =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../docs/research/strata-pilot-v1");
        let config = TrainingDataConfig::new(4, 1);

        // (a) Each committed records.bin has the sha256 that report.md gives it.
        let report_md = std::fs::read_to_string(committed.join("report.md")).unwrap();
        for (split_dir, _) in SPLITS {
            for name in STRATA {
                let file = format!("{split_dir}/{name}/records.bin");
                let row_prefix = format!("| `{file}` | `");
                let expected = report_md
                    .lines()
                    .find_map(|line| line.strip_prefix(row_prefix.as_str()))
                    .unwrap_or_else(|| panic!("no checksum row for {file} in report.md"))
                    .split('`')
                    .next()
                    .unwrap();
                let bytes = std::fs::read(committed.join(&file)).unwrap();
                let actual: String = Sha256::digest(&bytes)
                    .iter()
                    .map(|byte| format!("{byte:02x}"))
                    .collect();
                assert_eq!(
                    actual, expected,
                    "committed {file} sha256 differs from report.md"
                );
            }
        }

        // The committed manifest resolves to exactly these committed corpora.
        let loaded = load(&committed.join("manifest.json")).unwrap();
        for (split_dir, strata) in [
            ("cal", &loaded.manifest.calibration),
            ("val", &loaded.manifest.validation),
        ] {
            let names: Vec<&str> = strata.iter().map(|s| s.name.as_str()).collect();
            assert_eq!(names, STRATA, "committed manifest {split_dir} strata");
            for stratum in strata.iter() {
                assert_eq!(
                    std::fs::canonicalize(&stratum.dir).unwrap(),
                    std::fs::canonicalize(committed.join(split_dir).join(&stratum.name)).unwrap(),
                    "committed manifest {split_dir}/{} resolves elsewhere",
                    stratum.name
                );
            }
        }

        // (b) report.json names the committed corpora by identity and record count.
        let report_text = std::fs::read_to_string(committed.join("report.json")).unwrap();
        let report: serde_json::Value = serde_json::from_str(&report_text).unwrap();
        for (split_dir, key) in [("cal", "calibration"), ("val", "validation")] {
            let entries = report["versions"][key]
                .as_array()
                .unwrap_or_else(|| panic!("report.json versions.{key} is not an array"));
            for (index, name) in STRATA.iter().enumerate() {
                let entry = entries
                    .iter()
                    .find(|e| e["name"].as_str() == Some(*name))
                    .unwrap_or_else(|| panic!("report.json versions.{key} has no '{name}'"));
                let info = corpus_info(&committed.join(split_dir).join(name), &config).unwrap();
                assert_eq!(
                    info.identity,
                    entry["identity"].as_str().unwrap(),
                    "{key} '{name}' identity"
                );
                assert_eq!(
                    info.record_count,
                    entry["records"].as_u64().unwrap(),
                    "{key} '{name}' record count"
                );
                assert_eq!(
                    info.record_count, RECORDS[index] as u64,
                    "{key} '{name}' record count"
                );
            }
        }

        // (c) The generator still writes the committed records: inputs exactly
        // (they use no libm call), targets within TARGET_TOLERANCE.
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join("pilot");
        write(&dir).unwrap();
        let mut teacher = compile_creature(&teacher_of(&parent_creature())).unwrap();
        let mut parent_net = compile_creature(&parent_creature()).unwrap();
        for (split_dir, split) in SPLITS {
            for (stratum, name) in STRATA.iter().enumerate() {
                let label = format!("{split_dir}/{name}");
                let (inputs, targets) = read_records(&committed.join(split_dir).join(name));
                let (fresh_inputs, fresh_targets) = read_records(&dir.join(split_dir).join(name));
                assert_eq!(inputs.len(), RECORDS[stratum], "{label} input rows");
                assert_eq!(targets.len(), RECORDS[stratum], "{label} target rows");
                assert_eq!(fresh_inputs, inputs, "{label}: regenerated inputs differ");
                for (row, input) in inputs.iter().enumerate() {
                    assert_eq!(
                        *input,
                        record_input(stratum, split, row),
                        "{label}: row {row} input"
                    );
                }
                assert_eq!(
                    fresh_targets.len(),
                    targets.len(),
                    "{label} regenerated rows"
                );
                for (row, (&fresh, &target)) in fresh_targets.iter().zip(&targets).enumerate() {
                    assert!(
                        (fresh - target).abs() <= TARGET_TOLERANCE,
                        "{label}: row {row} regenerated target {fresh} differs from committed {target}"
                    );
                }

                let mut max_abs = [0f32; 4];
                for input in &inputs {
                    for (max, value) in max_abs.iter_mut().zip(input) {
                        *max = max.max(value.abs());
                    }
                }
                assert!(
                    max_abs[stratum] > 0.5,
                    "{label}: input {stratum} should be full scale, max |value| is {}",
                    max_abs[stratum]
                );
                for (input, max) in max_abs.iter().enumerate() {
                    if input != stratum {
                        assert!(
                            *max <= 0.1,
                            "{label}: input {input} should be scaled down, max |value| is {max}"
                        );
                    }
                }

                // Committed targets are the teacher's outputs, and clearly not the parent's.
                let mut largest_parent_gap = 0f32;
                for (input, &target) in inputs.iter().zip(&targets) {
                    let expected = teacher.activate(input, 1)[0];
                    assert!(
                        (target - expected).abs() <= TARGET_TOLERANCE,
                        "{label}: target {target} is not the teacher's output {expected} for input {input:?}"
                    );
                    let gap = (parent_net.activate(input, 1)[0] - target).abs();
                    largest_parent_gap = largest_parent_gap.max(gap);
                }
                assert!(
                    largest_parent_gap > 4.0 * TARGET_TOLERANCE,
                    "{label}: no target differs from the unmodified parent's output by more than \
                     4 x TARGET_TOLERANCE (largest gap {largest_parent_gap})"
                );
            }
        }
    }
}
