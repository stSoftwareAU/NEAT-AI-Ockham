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
    use neat_ai_ockham::corpus::corpus_info;
    use neat_core::training_data::TrainingDataConfig;

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

        let loaded = load(&first.join("manifest.json")).unwrap();
        assert_eq!(loaded.manifest.calibration.len(), 4);
        let config = TrainingDataConfig::new(4, 1);
        let count_of = |name: &str| {
            let stratum = loaded
                .manifest
                .calibration
                .iter()
                .find(|s| s.name == name)
                .unwrap_or_else(|| panic!("no calibration stratum '{name}'"));
            corpus_info(&stratum.dir, &config).unwrap().record_count
        };
        assert_eq!(count_of("rare"), 48);
        assert_eq!(count_of("alpha"), 192);
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
}
