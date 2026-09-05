//! Workflow-as-contract test: the Cargo preamble stays in one place.
//!
//! Every Cargo job needs the same two things before it can run: the
//! NEAT-AI-core sibling checkout the `../../NEAT-AI-core/neat-core` path
//! dependency resolves through, and the pinned Rust toolchain. Both live in
//! `.github/actions/setup-rust-workspace`, so a toolchain bump is one edit
//! rather than six (Issue #126). A workflow that calls the composite action
//! and *also* installs the toolchain itself has re-introduced the copy-paste
//! this test exists to prevent.

use std::path::{Path, PathBuf};

const COMPOSITE_USES: &str = "uses: ./.github/actions/setup-rust-workspace";
const TOOLCHAIN_USES: &str = "uses: dtolnay/rust-toolchain@";

fn github_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../.github")
}

/// Every workflow file as `(file name, body)`.
fn workflows() -> Vec<(String, String)> {
    let dir = github_dir().join("workflows");
    let entries = std::fs::read_dir(&dir).unwrap_or_else(|e| panic!("read {}: {e}", dir.display()));
    let mut files = Vec::new();
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
        files.push((name, body));
    }
    files
}

/// Lines carrying `needle`, as `(line number, line)`.
fn hits(body: &str, needle: &str) -> Vec<(usize, String)> {
    body.lines()
        .enumerate()
        .filter(|(_, line)| line.trim().starts_with(needle))
        .map(|(index, line)| (index + 1, line.trim().to_string()))
        .collect()
}

#[test]
fn composite_callers_do_not_install_the_toolchain_themselves() {
    let workflows = workflows();
    let callers: Vec<_> = workflows
        .iter()
        .filter(|(_, body)| body.contains(COMPOSITE_USES))
        .collect();
    assert!(
        !callers.is_empty(),
        "no workflow calls {COMPOSITE_USES} — did the composite action move or get renamed?"
    );
    for (file, body) in callers {
        let toolchain = hits(body, TOOLCHAIN_USES);
        assert!(
            toolchain.is_empty(),
            ".github/workflows/{file}:{}: installs the Rust toolchain directly while also \
             calling `setup-rust-workspace`, which already installs it — fold the step into the \
             composite action instead of copy-pasting the pin (Issue #126)",
            toolchain[0].0
        );
    }
}

#[test]
fn the_toolchain_pin_has_one_home_per_call_site() {
    let action = github_dir().join("actions/setup-rust-workspace/action.yml");
    let body = std::fs::read_to_string(&action)
        .unwrap_or_else(|e| panic!("read {}: {e}", action.display()));
    assert_eq!(
        hits(&body, TOOLCHAIN_USES).len(),
        1,
        "{} must install the pinned Rust toolchain exactly once",
        action.display()
    );
    assert!(
        body.contains("components:"),
        "{} must expose a `components` input so callers needing rustfmt/clippy can share it",
        action.display()
    );
}

#[test]
fn no_workflow_references_the_retired_action_path() {
    for (file, body) in workflows() {
        assert!(
            !body.contains("setup-neat-core"),
            ".github/workflows/{file}: references `setup-neat-core`, which was renamed to \
             `setup-rust-workspace` (Issue #126) — the local action would fail to resolve"
        );
    }
}
