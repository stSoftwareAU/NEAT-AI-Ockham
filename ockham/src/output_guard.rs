//! Shared output-path safety and atomic report writing for the report-only
//! subcommands (`bench-strata`, `audit-preservation`).
//!
//! Both subcommands rename their report into place, so a `--out` that names
//! the supplied creature would overwrite it, breaking the rule that the
//! creature is never written to. One copy of the guard means a fix to a missed
//! link case protects every subcommand.

use std::path::{Path, PathBuf};

use serde::Serialize;

/// Refuse a report path that would replace an input or land inside an input
/// corpus. `out` and its `<out>.tmp` sibling are both checked against every
/// input file (the same path, a symlink or a hard link) and every input
/// directory, because the report is renamed into place: `--out parent.json`
/// would otherwise overwrite the supplied creature, which is never written to.
pub(crate) fn refuse_input_overwrite(
    out: &Path,
    files: &[&Path],
    dirs: &[&Path],
) -> Result<(), String> {
    let mut tmp = out.as_os_str().to_owned();
    tmp.push(".tmp");
    for target in [out.to_path_buf(), PathBuf::from(tmp)] {
        let resolved = resolved_target(&target);
        for file in files {
            let same_path = resolved.is_some() && resolved == std::fs::canonicalize(file).ok();
            if same_path || same_inode(&target, file) {
                return Err(format!(
                    "report path '{}' would overwrite the input '{}'; choose another --out",
                    target.display(),
                    file.display()
                ));
            }
        }
        for dir in dirs {
            if let (Some(r), Ok(d)) = (&resolved, std::fs::canonicalize(dir))
                && r.starts_with(&d)
            {
                return Err(format!(
                    "report path '{}' is inside the input corpus '{}'; choose another --out",
                    target.display(),
                    dir.display()
                ));
            }
        }
    }
    Ok(())
}

/// The canonical form of `path`, or of its parent joined with its file name
/// when `path` does not exist yet.
fn resolved_target(path: &Path) -> Option<PathBuf> {
    if let Ok(found) = std::fs::canonicalize(path) {
        return Some(found);
    }
    let name = path.file_name()?;
    let parent = match path.parent() {
        Some(p) if !p.as_os_str().is_empty() => p,
        _ => Path::new("."),
    };
    std::fs::canonicalize(parent).ok().map(|p| p.join(name))
}

/// Whether two existing paths are the same file (catches hard links).
#[cfg(unix)]
fn same_inode(a: &Path, b: &Path) -> bool {
    use std::os::unix::fs::MetadataExt;
    match (std::fs::metadata(a), std::fs::metadata(b)) {
        (Ok(x), Ok(y)) => x.dev() == y.dev() && x.ino() == y.ino(),
        _ => false,
    }
}

/// Whether two existing paths are the same file (paths only off Unix).
#[cfg(not(unix))]
fn same_inode(_: &Path, _: &Path) -> bool {
    false
}

/// Write `report` as pretty JSON (plus a newline) to `<out>.tmp`, then rename
/// it over `out`, so a reader never sees a half-written report.
pub(crate) fn write_json_atomically<T: Serialize>(out: &Path, report: &T) -> Result<(), String> {
    let mut json = serde_json::to_string_pretty(report)
        .map_err(|e| format!("cannot serialise report: {e}"))?;
    json.push('\n');
    let mut tmp_name = out.file_name().unwrap_or_default().to_os_string();
    tmp_name.push(".tmp");
    let tmp = out.with_file_name(tmp_name);
    std::fs::write(&tmp, json)
        .map_err(|e| format!("cannot write report '{}': {e}", tmp.display()))?;
    std::fs::rename(&tmp, out)
        .map_err(|e| format!("cannot move report into place at '{}': {e}", out.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_report_path_that_names_an_input_is_refused() {
        let tmp = tempfile::tempdir().unwrap();
        let parent = tmp.path().join("parent.json");
        let corpus = tmp.path().join("corpus");
        std::fs::write(&parent, "{}").unwrap();
        std::fs::create_dir(&corpus).unwrap();
        let files = [parent.as_path()];
        let dirs = [corpus.as_path()];

        let same = refuse_input_overwrite(&parent, &files, &dirs).unwrap_err();
        assert!(same.contains("would overwrite the input"), "{same}");

        let tmp_sibling = tmp.path().join("parent");
        std::fs::rename(&parent, tmp.path().join("parent.tmp")).unwrap();
        let moved = tmp.path().join("parent.tmp");
        let sibling = refuse_input_overwrite(&tmp_sibling, &[moved.as_path()], &dirs).unwrap_err();
        assert!(sibling.contains("would overwrite the input"), "{sibling}");
        std::fs::rename(&moved, &parent).unwrap();

        #[cfg(unix)]
        {
            let link = tmp.path().join("link.json");
            std::os::unix::fs::symlink(&parent, &link).unwrap();
            let via_link = refuse_input_overwrite(&link, &files, &dirs).unwrap_err();
            assert!(via_link.contains("would overwrite the input"), "{via_link}");
            let hard = tmp.path().join("hard.json");
            std::fs::hard_link(&parent, &hard).unwrap();
            let via_hard = refuse_input_overwrite(&hard, &files, &dirs).unwrap_err();
            assert!(via_hard.contains("would overwrite the input"), "{via_hard}");
        }

        let inside = refuse_input_overwrite(&corpus.join("r.json"), &files, &dirs).unwrap_err();
        assert!(inside.contains("inside the input corpus"), "{inside}");

        let fine = tmp.path().join("report.json");
        refuse_input_overwrite(&fine, &files, &dirs).unwrap();
    }

    #[test]
    fn a_report_is_written_atomically_and_leaves_no_tmp_file() {
        let tmp = tempfile::tempdir().unwrap();
        let out = tmp.path().join("report.json");
        write_json_atomically(&out, &serde_json::json!({"a": 1})).unwrap();
        assert!(std::fs::read_to_string(&out).unwrap().ends_with("}\n"));
        assert!(!tmp.path().join("report.json.tmp").exists());
    }
}
