//! Workflow-as-contract test: the reusable `security.yml` never compiles
//! cargo-audit from source (Issue #244).
//!
//! `rustsec/audit-check` only runs `cargo install cargo-audit` when no
//! `cargo-audit` is already on `PATH`, so `security.yml` must install the
//! prebuilt binary *before* that step — exactly as `cargo-audit.yml` has since
//! Issue #208. The two installs must also stay identical (same action pin, same
//! `tool:` spec) so the two audit gates cannot drift onto different
//! cargo-audit versions and advisory coverage.

use std::path::Path;

const INSTALL_ACTION: &str = "taiki-e/install-action@";
const AUDIT_CHECK: &str = "rustsec/audit-check@";

fn workflow(name: &str) -> String {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../.github/workflows")
        .join(name);
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
}

/// Line index and full `uses:` value of the first step using `action`.
fn uses_line(body: &str, action: &str) -> Option<(usize, String)> {
    body.lines().enumerate().find_map(|(index, line)| {
        let value = line
            .trim()
            .trim_start_matches("- ")
            .strip_prefix("uses:")?
            .trim();
        value
            .starts_with(action)
            .then(|| (index, value.to_string()))
    })
}

/// The prebuilt cargo-audit install: `(line index, uses value, tool spec)`.
/// The `tool:` key is read from the same step, i.e. before the next `- ` item.
fn cargo_audit_install(body: &str) -> Option<(usize, String, String)> {
    let (index, uses) = uses_line(body, INSTALL_ACTION)?;
    let tool = body
        .lines()
        .skip(index + 1)
        .take_while(|line| !line.trim_start().starts_with("- "))
        .find_map(|line| line.trim().strip_prefix("tool:"))?
        .trim()
        .trim_matches('"')
        .trim_matches('\'')
        .to_string();
    tool.split('@')
        .next()
        .is_some_and(|name| name == "cargo-audit")
        .then_some((index, uses, tool))
}

#[test]
fn the_install_parser_reads_the_tool_of_its_own_step_only() {
    let body = "steps:\n  - name: Install\n    uses: taiki-e/install-action@abc  # v2\n    \
                with:\n      tool: 'cargo-audit@0.21.2'\n  - name: Other\n    tool: nope\n";
    assert_eq!(
        cargo_audit_install(body),
        Some((
            2,
            "taiki-e/install-action@abc  # v2".to_string(),
            "cargo-audit@0.21.2".to_string()
        ))
    );
    // A `tool:` that belongs to a later step is not this step's tool.
    let other_step = "  - uses: taiki-e/install-action@abc\n  - name: X\n    tool: cargo-audit\n";
    assert_eq!(cargo_audit_install(other_step), None);
    // Installing a different tool is not a cargo-audit install.
    let wrong_tool = "  - uses: taiki-e/install-action@abc\n    with:\n      tool: cargo-deny\n";
    assert_eq!(cargo_audit_install(wrong_tool), None);
    assert_eq!(cargo_audit_install(""), None);
}

#[test]
fn security_installs_prebuilt_cargo_audit_before_audit_check() {
    let body = workflow("security.yml");
    let (audit_check, _) =
        uses_line(&body, AUDIT_CHECK).expect("security.yml no longer runs rustsec/audit-check");
    let (install, _, _) = cargo_audit_install(&body).expect(
        "security.yml must install a prebuilt cargo-audit via taiki-e/install-action, or \
         rustsec/audit-check compiles it from source on every PR (Issue #244)",
    );
    assert!(
        install < audit_check,
        "the prebuilt cargo-audit install (line {}) must precede rustsec/audit-check (line {}), \
         or the action finds no binary and falls back to `cargo install`",
        install + 1,
        audit_check + 1
    );
}

#[test]
fn both_audit_gates_install_the_same_cargo_audit() {
    let (_, security_uses, security_tool) = cargo_audit_install(&workflow("security.yml"))
        .expect("security.yml has no prebuilt cargo-audit install");
    let (_, audit_uses, audit_tool) = cargo_audit_install(&workflow("cargo-audit.yml"))
        .expect("cargo-audit.yml has no prebuilt cargo-audit install");
    assert_eq!(
        security_uses, audit_uses,
        "security.yml and cargo-audit.yml must pin the same taiki-e/install-action"
    );
    assert_eq!(
        security_tool, audit_tool,
        "security.yml and cargo-audit.yml must install the same cargo-audit version"
    );
}
