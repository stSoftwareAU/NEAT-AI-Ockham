//! Runbook-as-contract tests for the emergency dependency fast lane (#129).
//!
//! `docs/incident-response.md` tells a maintainer how to short-circuit the
//! weekly dependency-update cadence when a RUSTSEC advisory lands against a
//! pinned crate that is being actively exploited. A runbook is only useful if
//! it is *true*, so these tests check it against the repository rather than
//! against itself: every workflow it names must exist, every workflow it lists
//! as a manual trigger must really declare `workflow_dispatch`, every review
//! team it names must really own reviews in `CODEOWNERS`, and `SECURITY.md`
//! must point at it — otherwise the maintainer under pressure never finds it.

use std::path::{Path, PathBuf};

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("..")
}

fn read(relative: &str) -> String {
    let path = root().join(relative);
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

const RUNBOOK: &str = "docs/incident-response.md";

fn runbook() -> String {
    read(RUNBOOK)
}

/// Workflow file names (`something.yml`) mentioned anywhere in `text`.
///
/// A leading directory (`.github/workflows/ci.yml`) is stripped, so the
/// runbook may cite either spelling.
fn workflow_mentions(text: &str) -> Vec<String> {
    let mut names = Vec::new();
    for token in text.split(|c: char| !(c.is_ascii_alphanumeric() || "._-/".contains(c))) {
        if !token.ends_with(".yml") && !token.ends_with(".yaml") {
            continue;
        }
        let name = token.rsplit('/').next().unwrap_or(token).to_string();
        if !names.contains(&name) {
            names.push(name);
        }
    }
    names
}

/// The first cell of every Markdown table row in `text` that names a workflow
/// file — the runbook's table of manual triggers.
fn tabulated_workflows(text: &str) -> Vec<String> {
    let mut names = Vec::new();
    for line in text.lines() {
        let trimmed = line.trim();
        if !trimmed.starts_with('|') {
            continue;
        }
        let cell = trimmed
            .trim_start_matches('|')
            .split('|')
            .next()
            .unwrap_or("")
            .trim()
            .trim_matches('`')
            .trim();
        if (cell.ends_with(".yml") || cell.ends_with(".yaml")) && !names.contains(&cell.to_string())
        {
            names.push(cell.to_string());
        }
    }
    names
}

/// `@org/team` handles mentioned in `text`.
fn team_mentions(text: &str) -> Vec<String> {
    let mut teams = Vec::new();
    for token in text.split_whitespace() {
        let handle =
            token.trim_matches(|c: char| !(c.is_ascii_alphanumeric() || "@/-_".contains(c)));
        if handle.starts_with('@') && handle.contains('/') && !teams.contains(&handle.to_string()) {
            teams.push(handle.to_string());
        }
    }
    teams
}

/// The `on:` trigger block of a workflow declares `workflow_dispatch`.
fn declares_workflow_dispatch(body: &str) -> bool {
    body.lines()
        .map(str::trim_end)
        .any(|line| line.trim_start() == "workflow_dispatch:" && line.starts_with("  "))
}

#[test]
fn runbook_exists_and_states_its_purpose() {
    let text = runbook();
    for needle in [
        "# Emergency dependency fix",
        "## When to use this",
        "## The fast lane",
        "## After the incident",
    ] {
        assert!(
            text.contains(needle),
            "{RUNBOOK} must keep the {needle:?} section (#129)"
        );
    }
}

#[test]
fn security_policy_links_the_runbook() {
    let security = read("SECURITY.md");
    assert!(
        security.contains(RUNBOOK),
        "SECURITY.md must link {RUNBOOK} so a maintainer finds the fast lane (#129)"
    );
    assert!(
        security.contains("## Expediting a fix"),
        "SECURITY.md must keep the 'Expediting a fix' section that names the \
         emergency override (#129)"
    );
}

#[test]
fn every_workflow_the_runbook_names_exists() {
    let missing: Vec<String> = workflow_mentions(&runbook())
        .into_iter()
        .filter(|name| !root().join(".github/workflows").join(name).is_file())
        .collect();
    assert!(
        missing.is_empty(),
        "{RUNBOOK} names workflows that do not exist: {missing:?} (#129)"
    );
}

#[test]
fn every_tabulated_workflow_is_manually_dispatchable() {
    let tabulated = tabulated_workflows(&runbook());
    assert!(
        !tabulated.is_empty(),
        "{RUNBOOK} must tabulate the workflows a maintainer can dispatch by hand (#129)"
    );
    let undispatchable: Vec<String> = tabulated
        .into_iter()
        .filter(|name| {
            let path = root().join(".github/workflows").join(name);
            let body = std::fs::read_to_string(&path)
                .unwrap_or_else(|e| panic!("{}: {e}", path.display()));
            !declares_workflow_dispatch(&body)
        })
        .collect();
    assert!(
        undispatchable.is_empty(),
        "{RUNBOOK} offers these as manual triggers but they declare no \
         `workflow_dispatch`: {undispatchable:?} (#129)"
    );
}

#[test]
fn the_dependency_cadence_workflows_are_on_the_fast_lane() {
    let tabulated = tabulated_workflows(&runbook());
    for name in ["cargo-audit.yml", "cargo-upgrade.yml"] {
        assert!(
            tabulated.iter().any(|t| t == name),
            "{RUNBOOK} must name {name} as part of the fast lane — it is how the \
             normal dependency cadence is short-circuited (#129)"
        );
    }
}

#[test]
fn every_review_team_the_runbook_names_owns_reviews() {
    let codeowners = read(".github/CODEOWNERS");
    let teams = team_mentions(&runbook());
    assert!(
        !teams.is_empty(),
        "{RUNBOOK} must name the review team an expedited PR goes to (#129)"
    );
    let unknown: Vec<String> = teams
        .into_iter()
        .filter(|team| !codeowners.contains(team.as_str()))
        .collect();
    assert!(
        unknown.is_empty(),
        "{RUNBOOK} names review teams absent from .github/CODEOWNERS: {unknown:?} (#129)"
    );
}

#[test]
fn runbook_keeps_the_gate_and_never_sanctions_bypassing_it() {
    let text = runbook();
    for needle in ["ci-required", "quality.sh"] {
        assert!(
            text.contains(needle),
            "{RUNBOOK} must state that the expedited path still clears {needle:?} — \
             the fast lane shortens the wait, not the gate (#129)"
        );
    }
}

#[test]
fn workflow_mention_and_table_parsers_read_markdown() {
    let sample = "See `.github/workflows/ci.yml` and cargo-audit.yml.\n\
                  \n\
                  | Workflow | Why |\n\
                  |---|---|\n\
                  | `cargo-audit.yml` | re-run the advisory scan |\n\
                  | prose row | not a workflow |\n";
    assert_eq!(workflow_mentions(sample), vec!["ci.yml", "cargo-audit.yml"]);
    assert_eq!(tabulated_workflows(sample), vec!["cargo-audit.yml"]);
    assert_eq!(
        team_mentions("ping @stSoftwareAU/developers, not @someone."),
        vec!["@stSoftwareAU/developers"]
    );
    assert!(declares_workflow_dispatch("on:\n  workflow_dispatch:\n"));
    assert!(!declares_workflow_dispatch(
        "# mentions workflow_dispatch: in a comment\non:\n  push:\n"
    ));
}
