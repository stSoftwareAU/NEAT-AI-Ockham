//! Manifest-as-contract test: `neat-core` stays a released git-tag pin.
//!
//! Ockham consumes `neat-core` through a git dependency on a NEAT-AI-core
//! release tag, never through the `../../NEAT-AI-core/neat-core` sibling path
//! it used to carry (Issue #210). The pin moves in exactly one place — the
//! `version-increment` job runs `scripts/family-pins.sh`, which rewrites the
//! `tag = "v<semver>"` on that one line and lets `Cargo.lock` follow.
//!
//! Two things make that work, and both are easy to undo by accident: the
//! declaration must be the single-line inline form `family-pins.sh` rewrites
//! (it refuses a pin spread over several lines), and the tag must be a plain
//! `v<major>.<minor>.<patch>` release rather than a branch, a revision or a
//! pre-release.

use std::path::{Path, PathBuf};

fn manifest_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("Cargo.toml")
}

/// The one uncommented line of `ockham/Cargo.toml` declaring `neat-core`.
fn neat_core_line() -> String {
    let path = manifest_path();
    let body =
        std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    let matches: Vec<&str> = body
        .lines()
        .map(str::trim)
        .filter(|line| !line.starts_with('#'))
        .filter(|line| line.starts_with("neat-core"))
        .collect();
    assert_eq!(
        matches.len(),
        1,
        "{} must declare `neat-core` on exactly one uncommented line — found {:?}",
        path.display(),
        matches
    );
    matches[0].to_string()
}

#[test]
fn neat_core_is_a_git_dependency_not_a_sibling_path() {
    let line = neat_core_line();
    assert!(
        line.contains(r#"git = "https://github.com/stSoftwareAU/NEAT-AI-core""#),
        "neat-core must be pinned to the NEAT-AI-core repository so the workspace builds with \
         nothing checked out beside it (Issue #210) — found: {line}"
    );
    assert!(
        !line.contains("path ="),
        "neat-core must not be a path dependency: the sibling checkout is retired (Issue #210) \
         — found: {line}"
    );
}

#[test]
fn the_pin_is_on_one_line_so_family_pins_can_rewrite_it() {
    let line = neat_core_line();
    assert!(
        line.ends_with('}'),
        "`scripts/family-pins.sh` refuses a pin spread over several lines, so the whole \
         declaration must sit on one line — found: {line}"
    );
}

#[test]
fn the_pin_names_a_release_tag() {
    let line = neat_core_line();
    for forbidden in ["branch =", "rev =", "version ="] {
        assert!(
            !line.contains(forbidden),
            "neat-core must be pinned by release tag, not by `{forbidden}` — a pin \
             `family-pins.sh` cannot move would silently track something else: {line}"
        );
    }
    let tag = line
        .split(r#"tag = ""#)
        .nth(1)
        .and_then(|rest| rest.split('"').next())
        .unwrap_or_else(|| panic!("no `tag = \"…\"` in the neat-core declaration: {line}"));
    let semver = tag
        .strip_prefix('v')
        .unwrap_or_else(|| panic!("neat-core tag {tag} must start with `v`"));
    let parts: Vec<&str> = semver.split('.').collect();
    assert_eq!(
        parts.len(),
        3,
        "neat-core tag {tag} must be a released v<major>.<minor>.<patch>, not a pre-release: \
         `family-pins.sh` only ever moves a pin onto a released tag"
    );
    for part in parts {
        assert!(
            !part.is_empty() && part.bytes().all(|b| b.is_ascii_digit()),
            "neat-core tag {tag} must be a released v<major>.<minor>.<patch>, not a pre-release: \
             `family-pins.sh` only ever moves a pin onto a released tag"
        );
    }
}
