//! Workflow-as-contract test: `cargo fmt`/`clippy` run once per PR (Issue #243).
//!
//! `ci.yml`'s `quality` job already runs fmt and clippy on PRs into `Develop`
//! and `milestone/**`. `cargo-quality.yml` exists for every *other* base (the
//! feature-branch and stacked-PR case), so it must ignore exactly the branches
//! `ci.yml` gates — otherwise both fire on the same PR and burn runner minutes.

use std::path::Path;

/// The body of `.github/workflows/<name>`.
fn workflow(name: &str) -> String {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../.github/workflows")
        .join(name);
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
}

fn indent(line: &str) -> usize {
    line.len() - line.trim_start().len()
}

fn unquote(item: &str) -> String {
    item.trim().trim_matches('"').trim_matches('\'').to_string()
}

/// The patterns under `on.pull_request.<key>`, or `None` when the key is absent.
///
/// Reads both the inline (`key: [a, "b"]`) and block (`key:\n  - a`) forms.
fn pull_request_filter(body: &str, key: &str) -> Option<Vec<String>> {
    let lines: Vec<&str> = body.lines().collect();
    let start = lines.iter().position(|l| l.trim() == "pull_request:")?;
    let pr_indent = indent(lines[start]);
    let prefix = format!("{key}:");
    let mut rest = lines[start + 1..]
        .iter()
        .take_while(|l| l.trim().is_empty() || indent(l) > pr_indent);
    let line = rest.find(|l| l.trim().starts_with(&prefix))?;
    let value = line.trim()[prefix.len()..].trim();
    if let Some(inline) = value.strip_prefix('[') {
        let inline = inline.strip_suffix(']').expect("unterminated inline list");
        return Some(inline.split(',').map(unquote).collect());
    }
    assert!(value.is_empty(), "unsupported `{key}` form: {line}");
    let key_indent = indent(line);
    Some(
        rest.take_while(|l| l.trim().is_empty() || indent(l) > key_indent)
            .map(|l| l.trim())
            .filter(|l| l.starts_with("- "))
            .map(|l| unquote(&l[2..]))
            .collect(),
    )
}

#[test]
fn the_filter_parser_reads_both_list_forms() {
    let inline = "on:\n  pull_request:\n    branches-ignore: [Develop, \"milestone/**\"]\n";
    assert_eq!(
        pull_request_filter(inline, "branches-ignore"),
        Some(vec!["Develop".to_string(), "milestone/**".to_string()])
    );
    assert_eq!(pull_request_filter(inline, "branches"), None);
    let block = "on:\n  pull_request:\n    branches:\n      - Develop\n      # note\n      - \"milestone/**\"\n  workflow_dispatch:\n";
    assert_eq!(
        pull_request_filter(block, "branches"),
        Some(vec!["Develop".to_string(), "milestone/**".to_string()])
    );
    assert_eq!(pull_request_filter("on:\n  push:\n", "branches"), None);
}

#[test]
fn cargo_quality_skips_exactly_the_branches_ci_already_gates() {
    let ci = pull_request_filter(&workflow("ci.yml"), "branches")
        .expect("ci.yml must scope its pull_request trigger with `branches:`");
    let quality = workflow("cargo-quality.yml");
    assert_eq!(
        pull_request_filter(&quality, "branches"),
        None,
        "cargo-quality.yml must not list `branches:` — every base ci.yml does not gate needs \
         fmt/clippy coverage, so scope it with `branches-ignore:` instead (Issue #243)"
    );
    let mut ignored = pull_request_filter(&quality, "branches-ignore").expect(
        "cargo-quality.yml must ignore the branches ci.yml's quality job already gates, or \
         both run fmt + clippy on the same PR (Issue #243)",
    );
    let mut gated = ci;
    ignored.sort();
    gated.sort();
    assert_eq!(
        ignored, gated,
        "cargo-quality.yml's branches-ignore must match ci.yml's branches exactly: a branch \
         in ci.yml only is gated twice, a branch in branches-ignore only is not gated at all"
    );
}
