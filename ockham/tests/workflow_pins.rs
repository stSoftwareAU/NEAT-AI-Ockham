//! Workflow-as-contract test: container images stay updater-trackable.
//!
//! A bare `image: repo@sha256:…` pin is immutable but invisible to Renovate
//! and Dependabot — both resolve the next version from the tag and rewrite the
//! digest beside it, so a digest with no tag freezes the image forever
//! (Issue #125). Every container pin must therefore carry both parts:
//! `repo:<tag>@sha256:<digest>`.

use std::path::{Path, PathBuf};

fn workflows_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../.github/workflows")
}

/// Container image references declared in the workflow files, as
/// `(file name, line number, image reference)`.
fn image_pins() -> Vec<(String, usize, String)> {
    let dir = workflows_dir();
    let entries = std::fs::read_dir(&dir).unwrap_or_else(|e| panic!("read {}: {e}", dir.display()));
    let mut pins = Vec::new();
    for entry in entries {
        let path = entry.expect("workflow dir entry").path();
        if !matches!(
            path.extension().and_then(|e| e.to_str()),
            Some("yml" | "yaml")
        ) {
            continue;
        }
        let name = path.file_name().unwrap().to_string_lossy().into_owned();
        let body = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
        for (index, line) in body.lines().enumerate() {
            let trimmed = line.trim();
            if let Some(reference) = trimmed.strip_prefix("image:") {
                pins.push((name.clone(), index + 1, reference.trim().to_string()));
            }
        }
    }
    pins
}

#[test]
fn every_container_pin_carries_a_tag_and_a_digest() {
    let pins = image_pins();
    assert!(
        !pins.is_empty(),
        "no `image:` pins found — did the workflow layout move?"
    );
    for (file, line, reference) in pins {
        let (repository, digest) = reference.split_once("@sha256:").unwrap_or_else(|| {
            panic!(".github/workflows/{file}:{line}: `{reference}` is not pinned by digest")
        });
        assert_eq!(
            digest.len(),
            64,
            ".github/workflows/{file}:{line}: `{reference}` has a malformed sha256 digest"
        );
        // The tag sits after the last `:`, and only when that colon follows the
        // final `/` — a registry host may carry a port (`host:5000/repo`).
        let tag = repository
            .rsplit_once('/')
            .map_or(repository, |(_, last)| last)
            .split_once(':')
            .map(|(_, tag)| tag);
        let tag = tag.unwrap_or_else(|| {
            panic!(
                ".github/workflows/{file}:{line}: `{reference}` pins a bare digest with no release \
                 tag — dependency updaters resolve bumps from the tag, so this image would never \
                 be updated again (Issue #125)"
            )
        });
        assert!(
            !tag.is_empty(),
            ".github/workflows/{file}:{line}: `{reference}` has an empty release tag"
        );
    }
}
