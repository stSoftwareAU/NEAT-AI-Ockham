//! README-as-contract tests.
//!
//! The README documents the tool as built: every long flag the binary accepts
//! must appear in it, the README must not advertise flags the binary lacks,
//! the charter sections the project was founded on must survive, the published
//! prior art the razor implements must stay cited, and the repository-layout
//! tree must list every source file.

use std::collections::BTreeSet;
use std::path::Path;
use std::process::Command;

fn readme() -> String {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../README.md");
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

fn help() -> String {
    let out = Command::new(env!("CARGO_BIN_EXE_neat_ai_ockham"))
        .arg("--help")
        .output()
        .expect("run --help");
    assert!(out.status.success(), "--help failed");
    String::from_utf8(out.stdout).unwrap()
}

fn long_flags(text: &str) -> BTreeSet<String> {
    let bytes = text.as_bytes();
    let mut out = BTreeSet::new();
    let mut i = 0;
    while i + 2 < bytes.len() {
        let boundary = i == 0 || !(bytes[i - 1].is_ascii_alphanumeric() || bytes[i - 1] == b'-');
        if boundary && bytes[i] == b'-' && bytes[i + 1] == b'-' && bytes[i + 2].is_ascii_lowercase()
        {
            let mut j = i + 2;
            while j < bytes.len()
                && (bytes[j].is_ascii_lowercase() || bytes[j].is_ascii_digit() || bytes[j] == b'-')
            {
                j += 1;
            }
            let flag = text[i..j].trim_end_matches('-').to_string();
            out.insert(flag);
            i = j;
        } else {
            i += 1;
        }
    }
    out
}

/// Flags of other tools the README legitimately quotes.
const FOREIGN_FLAGS: &[&str] = &[
    "--sample-rate",
    "--sample-phase",
    "--cost",
    "--release",
    "--features",
    "--check",
    "--all-features",
    "--all-targets",
    "--workspace",
    "--example",
    "--no-deps",
    "--locked",
    "--print",
];

#[test]
fn readme_documents_every_cli_flag() {
    let documented = long_flags(&readme());
    let missing: Vec<String> = long_flags(&help())
        .into_iter()
        .filter(|f| !matches!(f.as_str(), "--help" | "--version"))
        .filter(|f| !documented.contains(f))
        .collect();
    assert!(
        missing.is_empty(),
        "README.md does not document these CLI flags: {missing:?}"
    );
}

#[test]
fn readme_mentions_no_unknown_flags() {
    let known = long_flags(&help());
    let unknown: Vec<String> = long_flags(&readme())
        .into_iter()
        .filter(|f| !known.contains(f))
        .filter(|f| !FOREIGN_FLAGS.contains(&f.as_str()))
        .collect();
    assert!(
        unknown.is_empty(),
        "README.md documents flags the binary does not accept: {unknown:?}"
    );
}

#[test]
fn charter_sections_survive() {
    let readme = readme();
    for needle in [
        "Every neuron must earn its keep — prune freely, trust only the scorer.",
        "## Safety invariants",
        "The supplied creature is immutable.",
        "The full scorer is king.",
        "`best.json` may never be worse than the opening authoritative baseline.",
        "experimental Rust optimiser for already-fit",
        "45-minute default run budget",
        "## Version-1 constraints",
        "A local Ockham winner is not automatically population-ready.",
        "## Implementation roadmap",
    ] {
        assert!(
            readme.contains(needle),
            "README lost charter text: {needle:?}"
        );
    }
}

const LITERATURE_HEADING: &str = "## Where this sits in the literature";

/// The README section citing the prior art the razor already implements (#30).
fn literature_section() -> String {
    let readme = readme();
    let start = readme
        .find(LITERATURE_HEADING)
        .unwrap_or_else(|| panic!("README lost the {LITERATURE_HEADING:?} section (#30)"));
    let section = &readme[start..];
    let end = section[3..].find("\n## ").map_or(section.len(), |i| i + 3);
    section[..end].to_string()
}

/// Prior art the literature section must name, by mechanism (#30).
const LITERATURE_CITATIONS: &[&str] = &[
    // Saliency-ranked structural pruning.
    "Optimal Brain Damage",
    "LeCun",
    "Optimal Brain Surgeon",
    "Hassibi",
    "Molchanov",
    // Downstream compensation after removal.
    "Nagel",
    "ThiNet",
    "Luo",
    // Redundant / identity unit folding.
    "Srinivas",
    "Babu",
    // Iterated pruning — the compounding hypothesis.
    "Frankle",
    "Carbin",
    "Dense-Sparse-Dense",
    "Han",
    // Minimum description length.
    "Rissanen",
    "van Camp",
    // Racing / sampled screening.
    "Maron",
    "Moore",
    "Birattari",
    "F-Race",
    "Jamieson",
    "Talwalkar",
    // Adaptive overfitting — the caveat.
    "Dwork",
    "Blum",
    "Hardt",
];

#[test]
fn literature_section_cites_the_pruning_prior_art() {
    let section = literature_section();
    let missing: Vec<&str> = LITERATURE_CITATIONS
        .iter()
        .copied()
        .filter(|c| !section.contains(c))
        .collect();
    assert!(
        missing.is_empty(),
        "{LITERATURE_HEADING:?} omits these citations: {missing:?} (#30)"
    );
}

#[test]
fn literature_section_connects_the_growth_gate_to_mdl() {
    let section = literature_section();
    for needle in [
        "minimum description length",
        "Rissanen",
        "growth_units",
        "costOfGrowth",
    ] {
        assert!(
            section.contains(needle),
            "{LITERATURE_HEADING:?} must connect the growth gate to MDL; missing {needle:?} (#30)"
        );
    }
}

#[test]
fn literature_section_states_the_compounding_hypothesis_and_its_failure_mode() {
    let section = literature_section();
    for needle in [
        "compound into a material improvement",
        "The reusable holdout",
        "The Ladder",
        "noise floor",
    ] {
        assert!(
            section.contains(needle),
            "{LITERATURE_HEADING:?} must state the compounding hypothesis beside its known \
             failure mode; missing {needle:?} (#30)"
        );
    }
}

#[test]
fn house_terminology_survives_the_literature_section() {
    let readme = readme();
    for needle in [
        "🪒",
        "Every neuron must earn its keep",
        "## The Ockham rule",
        "A tiny genuine local win is a stepping stone, not a failure.",
    ] {
        assert!(
            readme.contains(needle),
            "README lost house terminology: {needle:?} (#30)"
        );
    }
}

#[test]
fn repository_layout_lists_every_source_file() {
    let readme = readme();
    let start = readme.find("## Repository layout").expect("layout section");
    let section = &readme[start..];
    let end = section[3..].find("\n## ").map_or(section.len(), |i| i + 3);
    let tree = &section[..end];
    let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    for entry in std::fs::read_dir(&src).unwrap() {
        let name = entry.unwrap().file_name().to_string_lossy().to_string();
        assert!(
            tree.contains(&name),
            "README repository layout omits ockham/src/{name}"
        );
    }
}

#[test]
fn contributing_documents_the_razor_commit_prefix() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../CONTRIBUTING.md");
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    assert!(
        text.contains("🪒"),
        "CONTRIBUTING.md must identify 🪒 as the Ockham commit-message prefix (#23)"
    );
    assert!(
        text.contains("## Commit messages"),
        "CONTRIBUTING.md must keep a Commit messages section (#23)"
    );
}

#[test]
fn long_flags_extracts_flags_and_ignores_prose_dashes() {
    let flags = long_flags("use --seed 1 and --output-dir x -- not a—flag");
    assert!(flags.contains("--seed") && flags.contains("--output-dir"));
    assert_eq!(flags.len(), 2);
}
