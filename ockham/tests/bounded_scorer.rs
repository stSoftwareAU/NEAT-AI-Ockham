//! The research time bound shares the ordinary scorer's argument and JSON contract.
#![cfg(unix)]

use std::os::unix::fs::PermissionsExt;
use std::path::Path;
use std::process::Command;

use neat_ai_ockham::scorer::{DirectoryScorer, ExternalScorer, ScorerError, ScorerMode};

#[test]
fn bounded_scoring_preserves_sample_flags_extra_arguments_and_loud_errors() {
    let bounding = ["timeout", "gtimeout"]
        .into_iter()
        .find(|tool| {
            Command::new(tool)
                .arg("--version")
                .output()
                .is_ok_and(|o| o.status.success())
        })
        .expect("research scorer test requires timeout or gtimeout");
    let tmp = tempfile::tempdir().unwrap();
    let binary = tmp.path().join("scorer");
    std::fs::write(&binary,r#"#!/bin/sh
set -eu
records=32
score=0.8
if [ "$1" = "--sample-rate" ]; then
    [ "$2" = "0.05" ] && [ "$3" = "--sample-phase" ] && [ "$4" = "7" ] || exit 41
    records=2
    score=0.7
    shift 4
fi
[ "$1" = "--gpu" ] && [ "$2" = "off" ] || exit 42
shift 2
[ -d "$1" ] && [ -d "$2" ] && [ "$#" = 2 ] || exit 43
printf '{"baseline":{"score":0.5,"error":0.5,"recordCount":%s},"candidate":{"score":%s,"error":0.2,"recordCount":%s}}' "$records" "$score" "$records"
"#).unwrap();
    std::fs::set_permissions(&binary, std::fs::Permissions::from_mode(0o700)).unwrap();
    let scorer = ExternalScorer {
        binary,
        extra_args: vec!["--gpu".into(), "off".into()],
    };
    let full = scorer
        .score_directory_bounded(
            tmp.path(),
            tmp.path(),
            ScorerMode::Full,
            Path::new(bounding),
            10,
        )
        .unwrap();
    assert_eq!(full["candidate"].score, 0.8);
    assert_eq!(full["candidate"].record_count, 32);
    let mode = ScorerMode::Sample {
        rate: 0.05,
        phase: 7,
    };
    let sample = scorer
        .score_directory_bounded(tmp.path(), tmp.path(), mode, Path::new(bounding), 10)
        .unwrap();
    assert_eq!(sample["candidate"].score, 0.7);
    assert_eq!(sample["baseline"].record_count, 2);
    assert_eq!(
        sample,
        scorer
            .score_directory(tmp.path(), tmp.path(), mode)
            .unwrap()
    );
    assert_eq!(
        scorer.score_directory_bounded(tmp.path(), tmp.path(), mode, Path::new(bounding), 0),
        Err(ScorerError::Malformed(
            "scorer timeout must be positive".into()
        ))
    );
    let wrong = ExternalScorer {
        extra_args: vec!["--wrong".into(), "off".into()],
        ..scorer
    };
    assert!(
        matches!(wrong.score_directory_bounded(tmp.path(),tmp.path(),mode,Path::new(bounding),10),Err(ScorerError::Failed{status,..}) if status.contains("42"))
    );
}
