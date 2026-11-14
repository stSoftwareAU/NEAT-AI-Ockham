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
                for row in chunk.inputs.chunks_exact(4) {
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
}
