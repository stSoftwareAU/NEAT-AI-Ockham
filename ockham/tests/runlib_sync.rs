//! Workflow-as-contract test: the canonical `scripts/runlib.sh` stays synced.
//!
//! `scripts/runlib.sh` is owned by NEAT-AI-core (NEAT-AI-core#680) and copied
//! here byte-for-byte. The `version-increment` job — the one job that already
//! commits and pushes to the PR branch — refreshes that copy, so the refresh
//! rides the same single commit as the bump and never races it (Issue #209).
//!
//! Two things must hold, and neither is visible to a reviewer scanning YAML:
//! the refresh happens *before* the bump rewrites the manifest, and the
//! refreshed file is in the `git add` list that commit is built from. Drop
//! either and the fleet silently keeps a stale copy.

use std::path::{Path, PathBuf};

const CORE_RUNLIB_PATH: &str = "repos/${CORE_REPO}/contents/scripts/runlib.sh?ref=${CORE_REF}";
const CORE_REPO: &str = "CORE_REPO: stSoftwareAU/NEAT-AI-core";
const BUMP_INVOCATION: &str = "./scripts/auto-version.sh ockham/Cargo.toml";

fn ci_workflow_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../.github/workflows/ci.yml")
}

fn ci_workflow() -> String {
    let path = ci_workflow_path();
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
}

/// The body of the `version-increment` job: its lines, up to the next
/// two-space job key.
fn version_increment_job(workflow: &str) -> String {
    let mut lines = workflow.lines().skip_while(|l| *l != "  version-increment:");
    let header = lines.next().expect("ci.yml declares a version-increment job");
    let is_job_key = |line: &str| {
        line.starts_with("  ")
            && !line.starts_with("   ")
            && !line.trim_start().starts_with('#')
            && line.trim_end().ends_with(':')
    };
    let body: Vec<&str> = lines.take_while(|line| !is_job_key(line)).collect();
    format!("{header}\n{}", body.join("\n"))
}

#[test]
fn version_increment_refreshes_runlib_from_core_before_the_bump() {
    let workflow = ci_workflow();
    let job = version_increment_job(&workflow);

    assert!(
        job.contains(CORE_REPO),
        "the version-increment job must name stSoftwareAU/NEAT-AI-core as the owner of \
         scripts/runlib.sh; job body:\n{job}"
    );

    let fetch = job
        .find(CORE_RUNLIB_PATH)
        .expect("the version-increment job must fetch core's scripts/runlib.sh");
    let bump = job
        .find(BUMP_INVOCATION)
        .expect("the version-increment job must still bump ockham/Cargo.toml");
    assert!(
        fetch < bump,
        "the runlib refresh must run before the bump rewrites ockham/Cargo.toml, \
         so both land in one commit"
    );
}

#[test]
fn the_refreshed_runlib_rides_the_commit_the_job_pushes() {
    let workflow = ci_workflow();
    let job = version_increment_job(&workflow);

    let add = job
        .lines()
        .find(|line| line.trim_start().starts_with("git add "))
        .unwrap_or_else(|| panic!("the version-increment job must stage its changes; job:\n{job}"));
    assert!(
        add.contains("scripts/runlib.sh"),
        "the `git add` list must include scripts/runlib.sh so a refreshed copy is \
         committed with the bump, not left behind: {add}"
    );
    assert!(
        add.contains("ockham/Cargo.toml") && add.contains("Cargo.lock"),
        "widening the `git add` must not drop the manifest or the lock file: {add}"
    );
}

#[test]
fn a_failed_fetch_of_cores_runlib_fails_the_job() {
    let workflow = ci_workflow();
    let job = version_increment_job(&workflow);
    let refresh = job
        .split("- name:")
        .find(|step| step.contains(CORE_RUNLIB_PATH))
        .expect("a step fetches core's scripts/runlib.sh");
    assert!(
        refresh.contains("exit 1"),
        "an unfetchable scripts/runlib.sh must fail the step loudly rather than \
         leave a stale copy in place; step:\n{refresh}"
    );
}
