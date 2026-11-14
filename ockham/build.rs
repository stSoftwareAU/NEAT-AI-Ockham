//! Records the source commit in `OCKHAM_GIT_COMMIT` so reports can name the
//! exact code that produced them (Issue #264). Falls back to `unknown` when
//! git or the repository is unavailable, such as a packaged crate build.

use std::process::Command;

fn git(args: &[&str]) -> Option<String> {
    let out = Command::new("git").args(args).output().ok()?;
    if !out.status.success() {
        return None;
    }
    let text = String::from_utf8(out.stdout).ok()?.trim().to_string();
    (!text.is_empty()).then_some(text)
}

fn main() {
    // `-dirty` marks a build from uncommitted edits, so a report never names a
    // commit that did not produce it.
    let commit = git(&["describe", "--always", "--dirty", "--abbrev=40"])
        .unwrap_or_else(|| "unknown".into());
    println!("cargo:rustc-env=OCKHAM_GIT_COMMIT={commit}");
    // `logs/HEAD` changes on every commit and checkout in this worktree.
    for path in ["HEAD", "logs/HEAD"] {
        if let Some(p) = git(&["rev-parse", "--git-path", path]) {
            println!("cargo:rerun-if-changed={p}");
        }
    }
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=src");
}
