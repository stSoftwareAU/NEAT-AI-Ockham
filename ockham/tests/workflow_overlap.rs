//! Workflow-as-contract tests: `cargo fmt`/`clippy` run once per PR (Issue #243),
//! and every required status check reports on PRs into `Develop` (PR #247).
//!
//! `cargo-quality.yml` is the single home of fmt + clippy and runs on every PR
//! base; `ci.yml`'s `quality` job must not repeat them. A required check whose
//! workflow skips `Develop` never reports, which blocks every PR into it.

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

/// Required status checks of the `Develop` ruleset (22325798). Keep in step
/// with the ruleset: a context listed there but missing here goes unchecked.
const REQUIRED_CONTEXTS: &[&str] = &[
    "Cargo Format and Clippy",
    "Generate SBOM",
    "Gitleaks Secrets Detection",
    "Markdown Lint",
    "Quality Checks",
    "Shell Script Quality",
];

/// Every `(file name, body)` under `.github/workflows`.
fn all_workflows() -> Vec<(String, String)> {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../.github/workflows");
    let mut names: Vec<String> = std::fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("read {}: {e}", dir.display()))
        .map(|entry| entry.expect("workflow dir entry").file_name())
        .map(|name| name.to_string_lossy().into_owned())
        .filter(|name| name.ends_with(".yml") || name.ends_with(".yaml"))
        .collect();
    names.sort();
    names
        .into_iter()
        .map(|n| {
            let body = workflow(&n);
            (n, body)
        })
        .collect()
}

/// Whether a GitHub branch-filter pattern matches `branch`. Handles the forms
/// these workflows use: exact names, `**`, and a trailing `*` / `**` glob.
fn pattern_matches(pattern: &str, branch: &str) -> bool {
    if pattern == "**" {
        return true;
    }
    if let Some(prefix) = pattern.strip_suffix("**") {
        return branch.starts_with(prefix);
    }
    if let Some(prefix) = pattern.strip_suffix('*') {
        return branch.starts_with(prefix) && !branch[prefix.len()..].contains('/');
    }
    pattern == branch
}

/// Whether the workflow's `pull_request` trigger fires for PRs into `base`.
fn pull_request_runs_on(body: &str, base: &str) -> bool {
    if !body.lines().any(|l| l.trim() == "pull_request:") {
        return false;
    }
    let included = pull_request_filter(body, "branches")
        .is_none_or(|b| b.iter().any(|p| pattern_matches(p, base)));
    let ignored = pull_request_filter(body, "branches-ignore")
        .is_some_and(|b| b.iter().any(|p| pattern_matches(p, base)));
    included && !ignored
}

/// Whether the workflow's `pull_request` trigger has a path filter, which
/// skips the run (and so the required check) on some PRs.
fn pull_request_has_path_filter(body: &str) -> bool {
    pull_request_filter(body, "paths").is_some()
        || pull_request_filter(body, "paths-ignore").is_some()
}

#[test]
fn the_branch_matcher_follows_github_globs() {
    assert!(pattern_matches("**", "milestone/x"));
    assert!(pattern_matches("*", "Develop"));
    assert!(!pattern_matches("*", "milestone/x"));
    assert!(pattern_matches("milestone/**", "milestone/x/y"));
    assert!(!pattern_matches("milestone/*", "milestone/x/y"));
    assert!(pattern_matches("Develop", "Develop"));
    assert!(!pattern_matches("Develop", "Developer"));
    let skips = "on:\n  pull_request:\n    branches-ignore: [Develop]\n";
    assert!(!pull_request_runs_on(skips, "Develop"));
    assert!(pull_request_runs_on(skips, "feature"));
    assert!(!pull_request_runs_on("on:\n  push:\n", "Develop"));
    assert!(pull_request_runs_on("on:\n  pull_request:\n", "Develop"));
    assert!(pull_request_has_path_filter(
        "on:\n  pull_request:\n    paths: [\"src/**\"]\n"
    ));
}

#[test]
fn every_required_check_reports_on_develop_prs() {
    let bodies = all_workflows();
    for context in REQUIRED_CONTEXTS {
        let job = format!("name: {context}");
        let owners: Vec<&(String, String)> = bodies
            .iter()
            .filter(|(_, body)| body.lines().any(|l| l.trim() == job))
            .collect();
        assert!(
            !owners.is_empty(),
            "required status check {context:?} has no job in .github/workflows — it can \
             never report, so every PR into Develop is blocked"
        );
        for (file, body) in owners {
            assert!(
                pull_request_runs_on(body, "Develop"),
                "{file} defines required status check {context:?} but its pull_request \
                 trigger skips Develop — the check never reports and blocks every PR into \
                 Develop (PR #247)"
            );
            assert!(
                !pull_request_has_path_filter(body),
                "{file} defines required status check {context:?} but filters \
                 pull_request by path — PRs outside the filter never get the check"
            );
        }
    }
}

#[test]
fn cargo_quality_runs_on_every_base_ci_gates() {
    let ci = pull_request_filter(&workflow("ci.yml"), "branches")
        .expect("ci.yml must scope its pull_request trigger with `branches:`");
    let quality = workflow("cargo-quality.yml");
    for base in ci.iter().map(|b| b.replace("**", "x").replace('*', "x")) {
        assert!(
            pull_request_runs_on(&quality, &base),
            "cargo-quality.yml must run on PRs into {base:?}: it is the only fmt/clippy \
             gate, and `Cargo Format and Clippy` is a required check (Issue #243)"
        );
    }
    assert!(
        pull_request_runs_on(&quality, "feature/x"),
        "cargo-quality.yml must also cover bases ci.yml does not gate (Issue #66)"
    );
}

#[test]
fn ci_quality_job_does_not_repeat_fmt_or_clippy() {
    let ci = workflow("ci.yml");
    for needle in ["cargo fmt", "cargo clippy"] {
        let hit = ci
            .lines()
            .enumerate()
            .find(|(_, l)| !l.trim_start().starts_with('#') && l.contains(needle));
        assert!(
            hit.is_none(),
            ".github/workflows/ci.yml:{}: runs `{needle}`, which cargo-quality.yml already \
             runs on every PR — the duplicate burns runner minutes (Issue #243)",
            hit.map(|(i, _)| i + 1).unwrap_or(0)
        );
    }
}
