//! The ordinary external scorer's argument and JSON contract, checked against an executable fixture (Issue #261 review of PR #291).
#![cfg(unix)]

use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::path::PathBuf;

use neat_ai_ockham::scorer::{DirectoryScorer, ExternalScorer, ScorerError, ScorerMode};

/// Fake `rust_scorer`: checks the argument order and the directory arguments,
/// then prints a JSON object shaped like the real directory-mode output.
const FIXTURE: &str = r#"#!/bin/sh
if [ "$1" = "--sample-rate" ]; then
  [ "$2" = "0.05" ] && [ "$3" = "--sample-phase" ] && [ "$4" = "7" ] || exit 41
  records=2
  score=0.7
  shift 4
else
  records=32
  score=0.8
fi
[ "$1" = "--gpu" ] && [ "$2" = "off" ] || exit 42
shift 2
[ -d "$1" ] && [ -d "$2" ] && [ $# -eq 2 ] || exit 43
printf '{"baseline":{"score":0.5,"error":0.5,"recordCount":%s},"candidate":{"score":%s,"error":0.2,"recordCount":%s}}\n' "$records" "$score" "$records"
"#;

struct Fixture {
    _dir: tempfile::TempDir,
    binary: PathBuf,
    creatures: PathBuf,
    training: PathBuf,
}

fn fixture() -> Fixture {
    let dir = tempfile::tempdir().unwrap();
    let binary = dir.path().join("fake_scorer");
    fs::write(&binary, FIXTURE).unwrap();
    fs::set_permissions(&binary, fs::Permissions::from_mode(0o700)).unwrap();
    let creatures = dir.path().join("creatures");
    let training = dir.path().join("training");
    fs::create_dir(&creatures).unwrap();
    fs::create_dir(&training).unwrap();
    Fixture {
        _dir: dir,
        binary,
        creatures,
        training,
    }
}

#[test]
fn full_and_sampled_scoring_pass_sample_flags_extra_arguments_and_directories() {
    let fx = fixture();
    let scorer = ExternalScorer {
        binary: fx.binary.clone(),
        extra_args: vec!["--gpu".into(), "off".into()],
    };

    let full = scorer
        .score_directory(&fx.creatures, &fx.training, ScorerMode::Full)
        .unwrap();
    assert_eq!(full["candidate"].score, 0.8);
    assert_eq!(full["candidate"].record_count, 32);

    let sample = scorer
        .score_directory(
            &fx.creatures,
            &fx.training,
            ScorerMode::Sample {
                rate: 0.05,
                phase: 7,
            },
        )
        .unwrap();
    assert_eq!(sample["candidate"].score, 0.7);
    assert_eq!(sample["baseline"].record_count, 2);

    assert_ne!(full, sample);
}

#[test]
fn a_scorer_that_rejects_its_arguments_fails_loud_with_its_exit_status() {
    let fx = fixture();
    let correct = ExternalScorer {
        binary: fx.binary.clone(),
        extra_args: vec!["--gpu".into(), "off".into()],
    };
    let wrong = ExternalScorer {
        binary: fx.binary.clone(),
        extra_args: vec!["--wrong".into(), "off".into()],
    };
    let sample = ScorerMode::Sample {
        rate: 0.05,
        phase: 7,
    };

    // Positive control: the correct extra arguments and phase succeed.
    assert!(
        correct
            .score_directory(&fx.creatures, &fx.training, sample)
            .is_ok()
    );

    // Wrong extra arguments are rejected by the scorer with exit 42.
    match wrong.score_directory(&fx.creatures, &fx.training, sample) {
        Err(ScorerError::Failed { status, .. }) => {
            assert!(status.contains("42"), "unexpected status: {status}");
        }
        other => panic!("expected ScorerError::Failed, got {other:?}"),
    }

    // A different sample phase is rejected by the scorer with exit 41.
    let phase8 = ScorerMode::Sample {
        rate: 0.05,
        phase: 8,
    };
    match correct.score_directory(&fx.creatures, &fx.training, phase8) {
        Err(ScorerError::Failed { status, .. }) => {
            assert!(status.contains("41"), "unexpected status: {status}");
        }
        other => panic!("expected ScorerError::Failed, got {other:?}"),
    }

    // score_directory does not validate the training dir itself, so a
    // missing path reaches the scorer, which rejects it with exit 43.
    let missing = fx.creatures.join("no-such-training-dir");
    match correct.score_directory(&fx.creatures, &missing, ScorerMode::Full) {
        Err(ScorerError::Failed { status, .. }) => {
            assert!(status.contains("43"), "unexpected status: {status}");
        }
        other => panic!("expected ScorerError::Failed, got {other:?}"),
    }
}
