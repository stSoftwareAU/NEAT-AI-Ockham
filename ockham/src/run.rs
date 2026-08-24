//! Run entry: establish the fail-closed incumbent baseline (Issue #2).
//!
//! Pruning is not attempted. A later issue wires the 45-minute loop on top of
//! this gate.

use std::path::PathBuf;

use neat_core::training_data::TrainingDataConfig;
use serde::Serialize;

use crate::baseline::{AuthoritativeBaseline, establish_baseline};
use crate::config::OckhamConfig;
use crate::corpus::corpus_info;
use crate::incumbent::{IncumbentMeta, load_incumbent};
use crate::scorer::DirectoryScorer;
use crate::{crate_version, log};

/// Result of a baseline-only run.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct BaselineRun {
    /// Crate version.
    pub crate_version: String,
    /// Isolated workspace directory.
    pub workspace: PathBuf,
    /// Incumbent metadata written beside the byte-for-byte copy.
    pub incumbent: IncumbentMeta,
    /// Authoritative full-corpus scorer baseline. Larger `score` is better.
    pub baseline: AuthoritativeBaseline,
    /// Optimisation status (`deferred` until later issues).
    pub optimisation: &'static str,
}

/// Load the immutable incumbent, copy it, and score the full-corpus baseline.
///
/// Returns an error (fail closed) on invalid creatures, scorer failure or
/// checksum drift. Never writes to [`OckhamConfig::creature`].
pub fn establish_run(
    config: &OckhamConfig,
    scorer: &dyn DirectoryScorer,
) -> Result<BaselineRun, String> {
    let source = config.creature.clone();
    let source_before = std::fs::read(&source).map_err(|e| format!("{}: {e}", source.display()))?;
    let incumbent = load_incumbent(&source).map_err(|e| e.to_string())?;
    log::info(&format!(
        "incumbent {}  neurons={} synapses={} forwardOnly={}",
        incumbent.short_checksum(),
        incumbent.creature.neurons.len(),
        incumbent.creature.synapses.len(),
        incumbent.creature.forward_only
    ));

    let workspace = config.output_dir.join("workspace");
    let meta = incumbent
        .write_workspace(&workspace)
        .map_err(|e| e.to_string())?;

    let cfg = TrainingDataConfig::new(incumbent.creature.input, incumbent.creature.output);
    let corpus = corpus_info(&config.training_data, &cfg)?;
    log::detail(&format!(
        "corpus {}  {} records in {} files",
        corpus.identity, corpus.record_count, corpus.file_count
    ));

    let baseline = establish_baseline(
        &incumbent,
        &config.training_data,
        &corpus,
        scorer,
        &config.scorer_args,
        &workspace,
    )?;
    log::ok(&format!(
        "authoritative baseline score={} error={} (larger score is better)",
        baseline.score, baseline.error
    ));

    // Opening parent is the only verified creature so far. `best.json` must
    // never be worse than this baseline; later issues may replace it.
    std::fs::create_dir_all(&config.output_dir)
        .map_err(|e| format!("{}: {e}", config.output_dir.display()))?;
    std::fs::write(config.output_dir.join("best.json"), &incumbent.text)
        .map_err(|e| format!("best.json: {e}"))?;

    let source_after = std::fs::read(&source).map_err(|e| format!("{}: {e}", source.display()))?;
    if source_after != source_before {
        return Err(format!(
            "source creature {} was modified; aborting",
            source.display()
        ));
    }

    Ok(BaselineRun {
        crate_version: crate_version().to_string(),
        workspace,
        incumbent: meta,
        baseline,
        optimisation: "deferred",
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::baseline::fake::ScriptedScorer;
    use crate::corpus::write_bin_file;
    use crate::fixtures::identity_creature_json;
    use std::time::Duration;

    fn config(tmp: &std::path::Path) -> OckhamConfig {
        let creature = tmp.join("creature.json");
        std::fs::write(&creature, identity_creature_json(1, 1)).unwrap();
        let train = tmp.join("train");
        std::fs::create_dir(&train).unwrap();
        write_bin_file(
            &train.join("0.bin"),
            &[(vec![1.0f32], vec![1.0f32]), (vec![2.0], vec![2.0])],
        )
        .unwrap();
        OckhamConfig {
            creature,
            training_data: train,
            output_dir: tmp.join("out"),
            timeout: Duration::from_secs(60),
            ..OckhamConfig::default()
        }
    }

    #[test]
    fn baseline_gate_writes_workspace_and_does_not_prune() {
        let tmp = tempfile::tempdir().unwrap();
        let cfg = config(tmp.path());
        let before = std::fs::read(&cfg.creature).unwrap();
        let run = establish_run(&cfg, &ScriptedScorer::ok(0.9, 0.1)).unwrap();
        assert_eq!(run.optimisation, "deferred");
        assert_eq!(run.baseline.score, 0.9);
        assert!(cfg.output_dir.join("best.json").exists());
        assert!(run.workspace.join("incumbent.json").exists());
        assert!(run.workspace.join("baseline.json").exists());
        assert_eq!(std::fs::read(&cfg.creature).unwrap(), before);
        assert_eq!(
            std::fs::read(cfg.output_dir.join("best.json")).unwrap(),
            before
        );
    }

    #[test]
    fn scorer_failure_does_not_write_best() {
        let tmp = tempfile::tempdir().unwrap();
        let cfg = config(tmp.path());
        let failing = ScriptedScorer {
            fail_with: Some("nope".into()),
            ..Default::default()
        };
        assert!(establish_run(&cfg, &failing).is_err());
        assert!(!cfg.output_dir.join("best.json").exists());
    }
}
