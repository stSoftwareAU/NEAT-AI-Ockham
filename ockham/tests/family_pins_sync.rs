//! Workflow-as-contract test: the neat-core pin moves with the bump.
//!
//! `scripts/family-pins.sh` is owned by NEAT-AI-core (NEAT-AI-core#681) and
//! copied here byte-for-byte, exactly as `scripts/runlib.sh` is. It is what
//! moves the `neat-core` release-tag pin in `ockham/Cargo.toml`, and the
//! `version-increment` job — the one job that already commits and pushes to
//! the PR branch — both refreshes it and runs it (Issue #210).
//!
//! Three things must hold, and none is visible to a reviewer scanning YAML:
//! the copy is refreshed *before* it is run, the run happens *before* the bump
//! rewrites the manifest, and the moved pin is in the `git add` list that
//! commit is built from. Drop any of them and a moved pin either never lands or
//! lands on its own, without the version bump the unattended machines rebuild
//! from.
//!
//! What the steps *do* — refuse an unfetchable file, refuse a body that is not
//! a script, overwrite a stale copy, leave an identical one alone, record a
//! moved pin for the commit step — is asserted by executing their own `run:`
//! bodies against a `gh` shim in `scripts/test-family-pins-refresh.sh`.
//! Behaviour belongs in the test that runs it, not in a substring match on
//! YAML.

use std::path::{Path, PathBuf};

const CORE_REPO: &str = "stSoftwareAU/NEAT-AI-core";
const FAMILY_PINS_PATH: &str = "scripts/family-pins.sh";
const MOVE_INVOCATION: &str = "./scripts/family-pins.sh\n";
const BUMP_INVOCATION: &str = "./scripts/auto-version.sh ockham/Cargo.toml";

fn ci_workflow() -> String {
    let path: PathBuf = Path::new(env!("CARGO_MANIFEST_DIR")).join("../.github/workflows/ci.yml");
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

/// Offset of the step that refreshes `scripts/family-pins.sh` from core: the
/// one step naming both the file and the repository that owns it. How it
/// fetches — `gh api`, `curl`, a checkout — is the step's business.
fn refresh_offset(job: &str) -> usize {
    let step = job
        .split("- name:")
        .find(|step| step.contains(FAMILY_PINS_PATH) && step.contains(CORE_REPO))
        .unwrap_or_else(|| {
            panic!(
                "a version-increment step must refresh {FAMILY_PINS_PATH} from {CORE_REPO}; \
                 job:\n{job}"
            )
        });
    job.find(step)
        .expect("the refresh step is part of the job it was taken from")
}

/// Offset of the line that actually runs the script over this checkout.
fn move_offset(job: &str) -> usize {
    job.find(MOVE_INVOCATION).unwrap_or_else(|| {
        panic!("the version-increment job must run {FAMILY_PINS_PATH}; job:\n{job}")
    })
}

#[test]
fn the_pin_is_moved_by_the_copy_that_was_just_refreshed() {
    let workflow = ci_workflow();
    let job = version_increment_job(&workflow);
    assert!(
        refresh_offset(&job) < move_offset(&job),
        "the family-pins refresh must run before the script does, so the pin is moved by \
         core's current copy and not by a stale one"
    );
}

#[test]
fn the_pin_moves_before_the_bump_so_both_land_in_one_commit() {
    let workflow = ci_workflow();
    let job = version_increment_job(&workflow);
    let bump = job
        .find(BUMP_INVOCATION)
        .expect("the version-increment job must still bump ockham/Cargo.toml");
    assert!(
        move_offset(&job) < bump,
        "the pin must move before the bump rewrites ockham/Cargo.toml, so a moved pin always \
         lands with a version bump rather than on its own"
    );
}

#[test]
fn the_moved_pin_and_the_refreshed_script_ride_the_commit_the_job_pushes() {
    let workflow = ci_workflow();
    let job = version_increment_job(&workflow);

    let add = job
        .lines()
        .find(|line| line.trim_start().starts_with("git add "))
        .unwrap_or_else(|| panic!("the version-increment job must stage its changes; job:\n{job}"));
    assert!(
        add.contains(FAMILY_PINS_PATH),
        "the `git add` list must include {FAMILY_PINS_PATH} so a refreshed copy is committed \
         with the bump, not left behind: {add}"
    );
    assert!(
        add.contains("ockham/Cargo.toml") && add.contains("Cargo.lock"),
        "the moved pin lives in the manifest and the lock file, so neither may be dropped \
         from the `git add` list: {add}"
    );
}
