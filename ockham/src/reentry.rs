//! Population re-entry gate against the latest global champion (Issue #9).
//!
//! A local Ockham win is never discarded. Population-ready status requires a
//! fresh same-call full-corpus comparison of Ockham `best.json` against the
//! **latest** global champion. Sample scores and stored scores from other
//! processes are not used.

use std::path::Path;

use neat_core::creature_to_json;
use serde::Serialize;

use crate::incumbent::{Incumbent, sha256_hex};
use crate::scorer::{DirectoryScorer, ScorerMode};

/// Result of the re-entry comparison. `best.json` is never deleted.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ReentryOutcome {
    /// Opening Ockham parent score (from this run's baseline).
    pub opening_score: f64,
    /// Final Ockham best score from the same-call comparison.
    pub ockham_score: f64,
    /// Latest global champion score from the same-call comparison.
    pub champion_score: f64,
    /// `ockham_score - opening_score`.
    pub cumulative_gain: f64,
    /// `champion_score - opening_score`.
    pub frontier_movement: f64,
    /// `ockham_score - champion_score`.
    pub population_headroom: f64,
    /// Ockham best checksum.
    pub ockham_checksum: String,
    /// Champion checksum.
    pub champion_checksum: String,
    /// True only when headroom exceeds `min_improvement`.
    pub population_ready: bool,
}

/// Compare Ockham best with `champion` in one full-corpus scorer call.
#[allow(clippy::too_many_arguments)]
pub fn compare_with_champion(
    scorer: &dyn DirectoryScorer,
    training_dir: &Path,
    ockham: &Incumbent,
    champion: &Incumbent,
    opening_score: f64,
    min_improvement: f64,
    dir: &Path,
    population_candidate_path: &Path,
) -> Result<ReentryOutcome, String> {
    std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    std::fs::write(dir.join("ockham.json"), &ockham.text)
        .map_err(|e| format!("ockham.json: {e}"))?;
    std::fs::write(dir.join("champion.json"), &champion.text)
        .map_err(|e| format!("champion.json: {e}"))?;
    // `baseline` is the reserved stem; the champion is the comparison parent.
    std::fs::write(dir.join("baseline.json"), &champion.text)
        .map_err(|e| format!("baseline.json: {e}"))?;

    let results = scorer
        .score_directory(dir, training_dir, ScorerMode::Full)
        .map_err(|e| e.to_string())?;
    let champion_score = results
        .get("baseline")
        .or_else(|| results.get("champion"))
        .ok_or_else(|| "re-entry: scorer returned no champion/baseline entry".to_string())?
        .score;
    let ockham_score = results
        .get("ockham")
        .ok_or_else(|| "re-entry: scorer returned no `ockham` entry".to_string())?
        .score;

    let population_headroom = ockham_score - champion_score;
    let population_ready = population_headroom > min_improvement;
    let outcome = ReentryOutcome {
        opening_score,
        ockham_score,
        champion_score,
        cumulative_gain: ockham_score - opening_score,
        frontier_movement: champion_score - opening_score,
        population_headroom,
        ockham_checksum: ockham.checksum.clone(),
        champion_checksum: champion.checksum.clone(),
        population_ready,
    };

    if population_ready {
        if let Some(parent) = population_candidate_path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| format!("{}: {e}", parent.display()))?;
        }
        let json = creature_to_json(&ockham.creature).map_err(|e| e.to_string())?;
        if sha256_hex(json.as_bytes()) != ockham.checksum {
            return Err("re-entry: exported Ockham JSON checksum drifted".into());
        }
        std::fs::write(population_candidate_path, json)
            .map_err(|e| format!("population-candidate.json: {e}"))?;
        let meta_path = population_candidate_path.with_extension("meta.json");
        let meta = serde_json::to_string_pretty(&outcome).map_err(|e| e.to_string())?;
        std::fs::write(&meta_path, meta).map_err(|e| format!("{}: {e}", meta_path.display()))?;
    }

    Ok(outcome)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::baseline::fake::ScriptedScorer;
    use crate::fixtures::hidden_identity_creature;
    use crate::incumbent::Incumbent;
    use std::collections::BTreeMap;

    #[test]
    fn ockham_ahead_of_champion_is_population_ready() {
        let ockham = Incumbent::from_creature(hidden_identity_creature(0.0, 1.0), "o").unwrap();
        let champion = Incumbent::from_creature(hidden_identity_creature(0.1, 1.0), "c").unwrap();
        let tmp = tempfile::tempdir().unwrap();
        let mut stem_scores = BTreeMap::new();
        stem_scores.insert("baseline".into(), 0.50);
        stem_scores.insert("champion".into(), 0.50);
        stem_scores.insert("ockham".into(), 0.80);
        let scorer = ScriptedScorer {
            stem_scores,
            ..ScriptedScorer::ok(0.50, 0.50)
        };
        let cand = tmp.path().join("population-candidate.json");
        let out = compare_with_champion(
            &scorer,
            tmp.path(),
            &ockham,
            &champion,
            0.40,
            1e-6,
            &tmp.path().join("reentry"),
            &cand,
        )
        .unwrap();
        assert!(out.population_ready);
        assert!(out.population_headroom > 0.0);
        assert!(cand.exists());
        assert!(
            tmp.path().join("population-candidate.meta.json").exists()
                || cand.with_extension("meta.json").exists()
        );
    }

    #[test]
    fn ockham_behind_champion_keeps_local_best_only() {
        let ockham = Incumbent::from_creature(hidden_identity_creature(0.0, 1.0), "o").unwrap();
        let champion = Incumbent::from_creature(hidden_identity_creature(0.1, 1.0), "c").unwrap();
        let tmp = tempfile::tempdir().unwrap();
        let mut stem_scores = BTreeMap::new();
        stem_scores.insert("baseline".into(), 0.90);
        stem_scores.insert("ockham".into(), 0.60);
        let scorer = ScriptedScorer {
            stem_scores,
            ..ScriptedScorer::ok(0.90, 0.10)
        };
        let cand = tmp.path().join("population-candidate.json");
        let out = compare_with_champion(
            &scorer,
            tmp.path(),
            &ockham,
            &champion,
            0.50,
            1e-6,
            &tmp.path().join("reentry"),
            &cand,
        )
        .unwrap();
        assert!(!out.population_ready);
        assert!(out.population_headroom < 0.0);
        assert!(!cand.exists(), "must not emit a stale population candidate");
    }
}
