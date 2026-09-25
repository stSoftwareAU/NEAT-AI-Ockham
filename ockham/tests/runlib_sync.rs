//! Workflow-as-contract test: the canonical NEAT-AI-core scripts stay synced,
//! and the neat-core pin moves where it is meant to.
//!
//! `scripts/runlib.sh` (NEAT-AI-core#680) and `scripts/family-pins.sh`
//! (NEAT-AI-core#681) are owned by NEAT-AI-core and copied here byte-for-byte.
//! The `version-increment` job — the one job that already commits and pushes
//! to the PR branch — refreshes both copies, then runs `family-pins.sh` to move
//! the neat-core git-tag pin to core's newest release, so the refresh and the
//! moved pin ride the same single commit as the bump and never race it
//! (Issues #209 and #210).
//!
//! Three things must hold, and none is visible to a reviewer scanning YAML:
//! the refresh happens *before* the pin moves (a stale `family-pins.sh` must
//! not be the one that moves it) and both happen *before* the bump rewrites
//! the manifest, and every refreshed file is in the `git add` list that commit
//! is built from. Drop any of them and the fleet silently keeps a stale copy or
//! a stale pin. All are questions about where a step sits in the job, which is
//! why they are asserted here.
//!
//! What the step *does* — refuse an unfetchable file, refuse a body that is
//! not a script, refuse one that breaks the install contract, overwrite a
//! stale copy and leave an identical one alone — is asserted by executing the
//! step's own `run:` body against a `gh` shim in
//! `scripts/test-runlib-refresh.sh`. Behaviour belongs in the test that runs
//! it, not in a substring match on YAML.

use std::path::{Path, PathBuf};

const CORE_REPO: &str = "stSoftwareAU/NEAT-AI-core";
const RUNLIB_PATH: &str = "scripts/runlib.sh";
const FAMILY_PINS_PATH: &str = "scripts/family-pins.sh";
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
    let mut lines = workflow
        .lines()
        .skip_while(|l| *l != "  version-increment:");
    let header = lines
        .next()
        .expect("ci.yml declares a version-increment job");
    let is_job_key = |line: &str| {
        line.starts_with("  ")
            && !line.starts_with("   ")
            && !line.trim_start().starts_with('#')
            && line.trim_end().ends_with(':')
    };
    let body: Vec<&str> = lines.take_while(|line| !is_job_key(line)).collect();
    format!("{header}\n{}", body.join("\n"))
}

/// The step that refreshes `file` from core: the one step naming both the file
/// and the repository that owns it. How it fetches — `gh api`, `curl`, a
/// checkout — is the step's business, not this test's.
fn refresh_step<'a>(job: &'a str, file: &str) -> &'a str {
    job.split("- name:")
        .find(|step| step.contains(file) && step.contains(CORE_REPO))
        .unwrap_or_else(|| {
            panic!("a version-increment step must refresh {file} from {CORE_REPO}; job:\n{job}")
        })
}

/// Byte offset of the line that runs `family-pins.sh` for real — not its
/// `--help` smoke test inside the refresh step.
fn pin_move(job: &str) -> usize {
    let mut offset = 0;
    for line in job.lines() {
        let run = line.trim_start().trim_start_matches("if ! ");
        if run.starts_with("./scripts/family-pins.sh") && !run.contains("--help") {
            return offset;
        }
        offset += line.len() + 1;
    }
    panic!(
        "the version-increment job must run ./scripts/family-pins.sh to move the pin; job:\n{job}"
    )
}

#[test]
fn version_increment_refreshes_runlib_from_core_before_the_bump() {
    let workflow = ci_workflow();
    let job = version_increment_job(&workflow);

    let step = refresh_step(&job, RUNLIB_PATH);
    let fetch = job
        .find(step)
        .expect("the refresh step is part of the job it was taken from");
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
fn version_increment_refreshes_family_pins_then_moves_the_pin_then_bumps() {
    let workflow = ci_workflow();
    let job = version_increment_job(&workflow);

    let step = refresh_step(&job, FAMILY_PINS_PATH);
    let fetch = job
        .find(step)
        .expect("the refresh step is part of the job it was taken from");
    let moved = pin_move(&job);
    let bump = job
        .find(BUMP_INVOCATION)
        .expect("the version-increment job must still bump ockham/Cargo.toml");
    assert!(
        fetch < moved,
        "family-pins.sh must be refreshed from core before it moves the pin, so a stale \
         copy is never the one that decides where the pin goes"
    );
    assert!(
        moved < bump,
        "the pin must move before the bump rewrites ockham/Cargo.toml, so a moved pin \
         always lands with a bump in one commit (Issue #210)"
    );
}

#[test]
fn the_refreshed_scripts_ride_the_commit_the_job_pushes() {
    let workflow = ci_workflow();
    let job = version_increment_job(&workflow);

    let add = job
        .lines()
        .find(|line| line.trim_start().starts_with("git add "))
        .unwrap_or_else(|| panic!("the version-increment job must stage its changes; job:\n{job}"));
    for file in [RUNLIB_PATH, FAMILY_PINS_PATH] {
        assert!(
            add.contains(file),
            "the `git add` list must include {file} so a refreshed copy is committed with \
             the bump, not left behind: {add}"
        );
    }
    assert!(
        add.contains("ockham/Cargo.toml") && add.contains("Cargo.lock"),
        "widening the `git add` must not drop the manifest or the lock file: {add}"
    );
}
