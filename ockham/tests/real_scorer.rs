//! Integration with the real NEAT-AI-scorer binary (Issue #2).
//!
//! Locates `rust_scorer` via `NEAT_SCORER_BIN`, then the sibling
//! `../../NEAT-AI-scorer/target/release/rust_scorer`, then `$PATH`. When no
//! binary is available the test prints a skip notice and passes — CI for this
//! repo checks out NEAT-AI-core but not a built scorer.

use std::path::{Path, PathBuf};
use std::process::Command;

use neat_ai_ockham::corpus::{corpus_info, write_bin_file};
use neat_ai_ockham::fixtures::identity_creature_json;
use neat_ai_ockham::scorer::{DirectoryScorer, ExternalScorer, ScorerMode};
use neat_ai_ockham::{establish_baseline, load_incumbent};
use neat_core::training_data::TrainingDataConfig;

fn scorer_binary() -> Option<PathBuf> {
    if let Some(p) = std::env::var_os("NEAT_SCORER_BIN") {
        let p = PathBuf::from(p);
        return p.is_file().then_some(p);
    }
    let sibling = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../NEAT-AI-scorer/target/release/rust_scorer");
    if sibling.is_file() {
        return Some(sibling);
    }
    Command::new("rust_scorer")
        .arg("--help")
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|_| PathBuf::from("rust_scorer"))
}

#[test]
fn real_scorer_scores_the_incumbent_baseline() {
    let Some(bin) = scorer_binary() else {
        eprintln!("skipping: no rust_scorer binary (set NEAT_SCORER_BIN)");
        return;
    };
    let tmp = tempfile::tempdir().unwrap();
    let train = tmp.path().join("train");
    std::fs::create_dir(&train).unwrap();
    let recs: Vec<(Vec<f32>, Vec<f32>)> = (0..32)
        .map(|i| (vec![i as f32 / 32.0], vec![i as f32 / 32.0]))
        .collect();
    write_bin_file(&train.join("0.bin"), &recs).unwrap();
    let creature = tmp.path().join("creature.json");
    std::fs::write(&creature, identity_creature_json(1, 1)).unwrap();

    let scorer = ExternalScorer {
        binary: bin,
        extra_args: vec!["--gpu".into(), "off".into()],
    };
    let incumbent = load_incumbent(&creature).unwrap();
    let corpus = corpus_info(&train, &TrainingDataConfig::new(1, 1)).unwrap();
    let ws = tmp.path().join("ws");
    std::fs::create_dir_all(&ws).unwrap();
    let baseline = establish_baseline(
        &incumbent,
        &train,
        &corpus,
        &scorer,
        &scorer.extra_args,
        &ws,
    )
    .unwrap();
    assert!(baseline.score.is_finite());
    assert!(baseline.error.is_finite());
    assert_eq!(baseline.record_count, 32);
    assert_eq!(baseline.corpus_record_count, 32);
    assert_eq!(baseline.incumbent_checksum, incumbent.checksum);
    assert!(ws.join("baseline.json").exists());

    // Same-call directory score of the incumbent stem agrees with the baseline.
    let dir = tmp.path().join("cohort");
    std::fs::create_dir(&dir).unwrap();
    std::fs::write(
        dir.join("baseline.json"),
        neat_core::creature_to_json(&incumbent.creature).unwrap(),
    )
    .unwrap();
    let again = scorer
        .score_directory(&dir, &train, ScorerMode::Full)
        .unwrap();
    let delta = (again["baseline"].score - baseline.score).abs();
    assert!(
        delta < 1e-9,
        "same-call score {} vs baseline {} (|Δ|={delta})",
        again["baseline"].score,
        baseline.score
    );
}
