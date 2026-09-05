//! Progressive adaptive screening (Issue #104).
//!
//! A fixed 5% screen pays the same scorer time for a candidate that is
//! catastrophically worse as for one that is a hair better. A ladder of
//! ascending sample rates spends almost nothing on the obvious losers:
//!
//! ```text
//! 0.25% corpus -> reject candidates clearly worse by a safe margin
//! 1% corpus    -> re-test the survivors on fresh records
//! 5% corpus    -> promote plausible winners
//! 100% corpus  -> authoritative acceptance (never here)
//! ```
//!
//! Two invariants hold at every stage:
//!
//! * **Sampling may reject or propose; only the full-corpus scorer accepts.**
//!   The ladder ends at the promotion stage — [`crate::promote`] still settles
//!   every survivor against the whole corpus.
//! * **A borderline candidate collects more evidence.** A non-final stage
//!   rejects only when the sampled Δ is at or below `-reject_margin`; anything
//!   uncertain — including a tiny improvement — is carried to the next, larger
//!   sample. That is why a non-final margin must be strictly positive.
//!
//! Every stage scores the incumbent alongside its candidates in one scorer
//! call, so the comparison stays apples-to-apples even though successive stages
//! sample at different phases. Stage sample phases are a pure function of the
//! batch index and the stage position, so a given seed and corpus reproduce the
//! same records exactly.
//!
//! The single-stage ladder — [`ScreenLadder::single`] — is the fixed-rate
//! control, and stays the default until benchmark evidence earns a change.

use std::path::{Path, PathBuf};

use neat_core::CreatureExport;
use serde::Serialize;

use crate::scorer::DirectoryScorer;
use crate::sweep::{
    SampledWinner, ScreenConfig, ScreenRejection, ScreenedLoser, SweepCandidate, screen_batch,
    screen_dir,
};

/// Default early-rejection margin for a stage that does not name its own.
///
/// Wide enough that only a candidate the sample calls clearly worse is dropped
/// at a small sample; a candidate within this band is re-tested rather than
/// rejected.
pub const DEFAULT_SCREEN_REJECT_MARGIN: f64 = 0.01;

/// One rung of the screening ladder.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ScreenStage {
    /// Scorer sample rate in `(0, 1)`.
    pub rate: f64,
    /// Sampled Δ at or below `-reject_margin` is rejected here.
    ///
    /// Ignored on the final stage, which promotes on `--screen-threshold`
    /// exactly as the fixed-rate control does.
    pub reject_margin: f64,
}

/// Ascending stages a candidate must survive to reach full scoring.
///
/// Every value is validated on construction, so a ladder that exists is a
/// ladder that may be run.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(transparent)]
pub struct ScreenLadder {
    stages: Vec<ScreenStage>,
}

impl ScreenLadder {
    /// The fixed-rate control: one stage, promotion by `--screen-threshold`.
    pub fn single(rate: f64) -> Result<Self, String> {
        Self::new(vec![ScreenStage {
            rate,
            reject_margin: 0.0,
        }])
    }

    /// Validate and build a ladder.
    ///
    /// Rejects an empty ladder, a rate outside `(0, 1)`, rates that do not
    /// strictly ascend, and a non-positive early-rejection margin on a
    /// non-final stage — that last one would let a tiny sampled loss kill a
    /// candidate at the smallest sample, which is precisely what the ladder
    /// must not do.
    pub fn new(stages: Vec<ScreenStage>) -> Result<Self, String> {
        if stages.is_empty() {
            return Err("--screen-stages must name at least one stage".into());
        }
        let last = stages.len() - 1;
        let mut previous: Option<f64> = None;
        for (i, stage) in stages.iter().enumerate() {
            if !(stage.rate > 0.0 && stage.rate < 1.0) {
                return Err(format!(
                    "--screen-stages: stage {i} rate {} must be in (0, 1)",
                    stage.rate
                ));
            }
            if let Some(prev) = previous
                && stage.rate <= prev
            {
                return Err(format!(
                    "--screen-stages: rates must strictly ascend; stage {i} rate {} follows {prev}",
                    stage.rate
                ));
            }
            previous = Some(stage.rate);
            if !stage.reject_margin.is_finite() {
                return Err(format!(
                    "--screen-stages: stage {i} reject margin must be finite"
                ));
            }
            if i != last && stage.reject_margin <= 0.0 {
                return Err(format!(
                    "--screen-stages: stage {i} reject margin must be > 0 so a borderline \
                     candidate collects more evidence instead of being rejected early"
                ));
            }
        }
        Ok(Self { stages })
    }

    /// Parse `rate[:margin],…` — e.g. `0.0025:0.02,0.01:0.005,0.05`.
    ///
    /// A stage that omits its margin takes `default_margin`.
    pub fn parse(spec: &str, default_margin: f64) -> Result<Self, String> {
        let mut stages = Vec::new();
        for (i, field) in spec.split(',').map(str::trim).enumerate() {
            if field.is_empty() {
                return Err(format!("--screen-stages: stage {i} is empty"));
            }
            let (rate, margin) = match field.split_once(':') {
                Some((rate, margin)) => (rate.trim(), Some(margin.trim())),
                None => (field, None),
            };
            let rate: f64 = rate
                .parse()
                .map_err(|e| format!("--screen-stages: stage {i} rate `{rate}`: {e}"))?;
            let reject_margin = match margin {
                Some(text) => text
                    .parse()
                    .map_err(|e| format!("--screen-stages: stage {i} margin `{text}`: {e}"))?,
                None => default_margin,
            };
            stages.push(ScreenStage {
                rate,
                reject_margin,
            });
        }
        Self::new(stages)
    }

    /// The stages, in ascending sample-rate order.
    pub fn stages(&self) -> &[ScreenStage] {
        &self.stages
    }

    /// `true` when the ladder has more than the one control stage.
    pub fn is_progressive(&self) -> bool {
        self.stages.len() > 1
    }

    /// Rate of the promotion stage — the largest sample the screen ever takes.
    pub fn promotion_rate(&self) -> f64 {
        self.stages[self.stages.len() - 1].rate
    }

    /// Sample phase for one stage of one batch.
    ///
    /// A pure function of the batch index and stage position, so the same seed
    /// and corpus replay the same records. Every stage of a batch gets its own
    /// phase, which is all this side of the boundary can promise: which records
    /// a phase selects is the scorer's stride, so how far two stages' slices
    /// overlap is its business, not the ladder's.
    pub fn phase(&self, batch: u64, stage: usize) -> u64 {
        batch
            .wrapping_mul(self.stages.len() as u64)
            .wrapping_add(stage as u64)
    }

    /// Cohort directory for one stage of one batch.
    ///
    /// The control keeps the historical `screen-<batch>` path; a progressive
    /// ladder gives each stage its own directory, because a stale candidate
    /// file left behind by an earlier stage would be scored again by the next.
    pub fn stage_dir(&self, workspace: &Path, batch: u64, stage: usize) -> PathBuf {
        let dir = screen_dir(workspace, batch);
        if self.is_progressive() {
            dir.join(format!("s{stage}"))
        } else {
            dir
        }
    }
}

/// What one stage cost and decided (Issue #104).
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct StageRecord {
    /// Position in the ladder (0-based).
    pub stage: usize,
    /// Sample rate this stage scored at.
    pub rate: f64,
    /// Deterministic sample phase.
    pub phase: u64,
    /// Candidates that entered the stage.
    pub entered: usize,
    /// Candidates the stage ended.
    pub rejected: usize,
    /// Candidates carried to the next stage, or promoted by the final one.
    pub promoted: usize,
    /// Records the scorer read, summed over the cohort including the incumbent.
    pub records_scored: u64,
    /// Sampled incumbent score this stage compared against.
    pub baseline_score: f64,
    /// Mean sampled Δ over the candidates that entered.
    pub mean_delta: f64,
    /// Scorer wall time (ms).
    pub ms: u64,
    /// Why survivors left the stage: `carried` or `promoted`.
    pub outcome: &'static str,
}

/// Result of running a whole ladder over one batch.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ProgressiveScreen {
    /// Candidates the promotion stage put forward for full scoring.
    pub winners: Vec<SampledWinner>,
    /// Candidates rejected, each carrying the stage and reason that ended it.
    pub losers: Vec<ScreenedLoser>,
    /// Per-stage economics, in ladder order.
    pub stages: Vec<StageRecord>,
    /// Sampled incumbent score at the last stage that ran.
    ///
    /// That is the promotion stage for a candidate set that reached it, and the
    /// rung the ladder stopped on otherwise; `NaN` when no stage ran at all,
    /// which only happens for an empty batch.
    pub baseline_score: f64,
    /// Total scorer wall time across every stage (ms).
    pub screen_ms: u64,
}

impl ProgressiveScreen {
    /// Records the scorer read across every stage.
    pub fn records_scored(&self) -> u64 {
        self.stages.iter().map(|s| s.records_scored).sum()
    }

    /// Creature-scores this batch cost, expressed at the ladder's promotion rate.
    ///
    /// The cost model tracks milliseconds per creature at one sample rate
    /// ([`crate::run`] seeds it with the promotion rate). A stage at a tenth of
    /// that rate is a tenth of a creature-score, so the ladder is converted
    /// rather than counted — counting rungs would tell the model each creature
    /// costs far less than a promotion-stage score really does.
    ///
    /// The rate comes from the ladder rather than the caller: `ScreenLadder`
    /// validates it into `(0, 1)`, so there is no degenerate rate to guard
    /// against here. Never zero, so a division by the result is safe.
    pub fn promotion_rate_creatures(&self, ladder: &ScreenLadder) -> usize {
        let promotion_rate = ladder.promotion_rate();
        let weighted: f64 = self
            .stages
            .iter()
            .map(|s| (s.entered + 1) as f64 * s.rate / promotion_rate)
            .sum();
        (weighted.round() as usize).max(1)
    }

    /// Candidates each rejection reason ended, for the run log (#104).
    pub fn rejection_tally(&self) -> (usize, usize) {
        let clearly_worse = self
            .losers
            .iter()
            .filter(|l| l.reason == ScreenRejection::ClearlyWorse)
            .count();
        (clearly_worse, self.losers.len() - clearly_worse)
    }
}

/// Parameters for [`screen_progressive`].
#[derive(Debug, Clone, Copy)]
pub struct ProgressiveConfig<'a> {
    /// Stages to run, ascending.
    pub ladder: &'a ScreenLadder,
    /// Sampled Δ the promotion stage requires (`--screen-threshold`).
    pub threshold: f64,
    /// Batch index — fixes every stage's sample phase.
    pub batch: u64,
    /// Unvisited neurons left after this batch (for the sweep ETA).
    pub remaining_after: usize,
    /// Run workspace; each stage scores in its own directory beneath it.
    pub workspace: &'a Path,
}

/// Run `candidates` up the ladder, returning the promotion stage's winners.
///
/// A stage that rejects everything ends the batch immediately — the whole point
/// of the ladder is that the larger samples are never paid for. Any scorer
/// failure aborts the batch, exactly as the single-stage screen does: a partial
/// ladder is not a verdict.
pub fn screen_progressive(
    scorer: &dyn DirectoryScorer,
    training_dir: &Path,
    incumbent: &CreatureExport,
    candidates: Vec<SweepCandidate>,
    cfg: ProgressiveConfig<'_>,
) -> Result<ProgressiveScreen, String> {
    let stages = cfg.ladder.stages();
    let last = stages.len() - 1;
    let mut survivors = candidates;
    let mut losers: Vec<ScreenedLoser> = Vec::new();
    let mut records = Vec::with_capacity(stages.len());
    let mut screen_ms = 0u64;
    let mut baseline_score = f64::NAN;
    let mut winners = Vec::new();

    for (index, stage) in stages.iter().enumerate() {
        if survivors.is_empty() {
            break;
        }
        let entered = survivors.len();
        let final_stage = index == last;
        // A non-final stage rejects only what the sample calls clearly worse;
        // `screen_batch` splits on `delta > threshold`, so `-reject_margin` is
        // that rule expressed in its terms. The final stage uses the promotion
        // threshold, which is what the fixed-rate control has always applied.
        let threshold = if final_stage {
            cfg.threshold
        } else {
            -stage.reject_margin
        };
        let phase = cfg.ladder.phase(cfg.batch, index);
        let outcome = screen_batch(
            scorer,
            training_dir,
            incumbent,
            survivors,
            ScreenConfig {
                sample_rate: stage.rate,
                sample_phase: phase,
                threshold,
                remaining_after: cfg.remaining_after,
                dir: &cfg.ladder.stage_dir(cfg.workspace, cfg.batch, index),
            },
        )?;
        screen_ms = screen_ms.saturating_add(outcome.screen_ms);
        baseline_score = outcome.baseline_score;
        let delta_sum: f64 = outcome.winners.iter().map(|w| w.delta).sum::<f64>()
            + outcome.losers.iter().map(|l| l.delta).sum::<f64>();
        records.push(StageRecord {
            stage: index,
            rate: stage.rate,
            phase,
            entered,
            rejected: outcome.losers.len(),
            promoted: outcome.winners.len(),
            records_scored: outcome.records_scored,
            baseline_score: outcome.baseline_score,
            mean_delta: if entered == 0 {
                0.0
            } else {
                delta_sum / entered as f64
            },
            ms: outcome.screen_ms,
            outcome: if final_stage { "promoted" } else { "carried" },
        });
        losers.extend(outcome.losers.into_iter().map(|mut l| {
            l.stage = index;
            if !final_stage {
                l.reason = ScreenRejection::ClearlyWorse;
            }
            l
        }));
        if final_stage {
            winners = outcome.winners;
            survivors = Vec::new();
        } else {
            survivors = outcome.winners.into_iter().map(|w| w.candidate).collect();
        }
    }

    Ok(ProgressiveScreen {
        winners,
        losers,
        stages: records,
        baseline_score,
        screen_ms,
    })
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::collections::BTreeMap;

    use super::*;
    use crate::fixtures::{creature, neuron, synapse};
    use crate::scorer::{ScoreResult, ScorerError, ScorerMode};
    use crate::sweep::CandidateKind;

    /// Scorer that answers from a per-stem score table and logs every call.
    #[derive(Default)]
    struct RecordingScorer {
        /// `stem -> score`; `baseline` included.
        scores: BTreeMap<String, f64>,
        /// Records claimed per creature.
        record_count: u64,
        /// `(rate, phase, stems)` of every call, in order.
        calls: RefCell<Vec<(f64, u64, Vec<String>)>>,
        /// When set, the call at this index fails.
        fail_call: Option<usize>,
    }

    impl DirectoryScorer for RecordingScorer {
        fn score_directory(
            &self,
            creature_dir: &Path,
            _training_dir: &Path,
            mode: ScorerMode,
        ) -> Result<BTreeMap<String, ScoreResult>, ScorerError> {
            let mut stems: Vec<String> = std::fs::read_dir(creature_dir)
                .map_err(|e| ScorerError::Spawn(e.to_string()))?
                .filter_map(|e| e.ok())
                .map(|e| e.path())
                .filter(|p| p.extension().is_some_and(|x| x == "json"))
                .filter_map(|p| p.file_stem().map(|s| s.to_string_lossy().to_string()))
                .collect();
            stems.sort();
            let (rate, phase) = match mode {
                ScorerMode::Sample { rate, phase } => (rate, phase),
                ScorerMode::Full => (1.0, 0),
            };
            if self.fail_call == Some(self.calls.borrow().len()) {
                return Err(ScorerError::Failed {
                    status: "exit 1".into(),
                    stderr: "scripted stage failure".into(),
                });
            }
            self.calls.borrow_mut().push((rate, phase, stems.clone()));
            Ok(stems
                .into_iter()
                .map(|stem| {
                    let score = self.scores.get(&stem).copied().unwrap_or(0.0);
                    (
                        stem,
                        ScoreResult {
                            score,
                            error: 1.0 - score,
                            complexity_penalty: 0.0,
                            record_count: self.record_count,
                            sample_rate: Some(rate),
                            gpu_backend: None,
                            cost_name: None,
                            time_taken: 0.0,
                        },
                    )
                })
                .collect())
        }

        fn identity(&self) -> String {
            "fake:recording".into()
        }
    }

    fn incumbent() -> CreatureExport {
        creature(
            1,
            1,
            vec![
                neuron("hidden", "h_a", 0.0, Some("IDENTITY")),
                neuron("output", "output-0", 0.0, Some("IDENTITY")),
            ],
            vec![
                synapse("input-0", "h_a", 1.0),
                synapse("h_a", "output-0", 1.0),
            ],
        )
    }

    fn candidate(stem: &str, uuid: &str) -> SweepCandidate {
        SweepCandidate {
            uuid: uuid.to_string(),
            permutation_index: 0,
            kind: CandidateKind::Ablation,
            stem: stem.to_string(),
            creature: incumbent(),
        }
    }

    fn ladder() -> ScreenLadder {
        ScreenLadder::parse("0.0025:0.01,0.01:0.005,0.05", DEFAULT_SCREEN_REJECT_MARGIN).unwrap()
    }

    fn scorer_with(scores: &[(&str, f64)]) -> RecordingScorer {
        RecordingScorer {
            scores: scores.iter().map(|(s, v)| ((*s).to_string(), *v)).collect(),
            record_count: 100,
            ..RecordingScorer::default()
        }
    }

    #[test]
    fn a_ladder_must_ascend_and_leave_room_for_evidence() {
        assert!(ScreenLadder::new(Vec::new()).is_err());
        assert!(
            ScreenLadder::parse("0.05,0.01", 0.01)
                .unwrap_err()
                .contains("ascend")
        );
        assert!(
            ScreenLadder::parse("0,0.05", 0.01)
                .unwrap_err()
                .contains("(0, 1)")
        );
        assert!(
            ScreenLadder::parse("0.01,1.0", 0.01)
                .unwrap_err()
                .contains("(0, 1)")
        );
        // A zero margin on a non-final stage would reject a candidate merely
        // for being a hair worse at the smallest sample.
        assert!(
            ScreenLadder::parse("0.0025:0,0.05", 0.01)
                .unwrap_err()
                .contains("more evidence")
        );
        assert!(ScreenLadder::parse("0.0025,,0.05", 0.01).is_err());
        assert!(ScreenLadder::parse("0.0025,x", 0.01).is_err());
        // The final stage promotes on the threshold, so its margin is free.
        let ok = ScreenLadder::parse("0.0025:0.02,0.05:0", 0.01).unwrap();
        assert_eq!(ok.promotion_rate(), 0.05);
        assert!(ok.is_progressive());
        let control = ScreenLadder::single(0.05).unwrap();
        assert!(!control.is_progressive());
        assert_eq!(control.promotion_rate(), 0.05);
        assert!(ScreenLadder::single(0.0).is_err());
    }

    #[test]
    fn a_stage_without_a_margin_takes_the_default() {
        let l = ScreenLadder::parse("0.0025, 0.01 : 0.5, 0.05", 0.02).unwrap();
        assert_eq!(l.stages()[0].reject_margin, 0.02);
        assert_eq!(l.stages()[1].reject_margin, 0.5);
        assert_eq!(l.stages().len(), 3);
    }

    #[test]
    fn phases_are_deterministic_and_distinct_per_stage() {
        let l = ladder();
        assert_eq!(l.phase(7, 0), ladder().phase(7, 0));
        let phases: Vec<u64> = (0..3).map(|s| l.phase(7, s)).collect();
        assert_eq!(phases, vec![21, 22, 23]);
        // No batch shares a phase with another batch's stage.
        assert!(!phases.contains(&l.phase(8, 0)));
        // Overflow near u64::MAX must wrap, not panic.
        let _ = l.phase(u64::MAX, 2);
    }

    #[test]
    fn an_obvious_loser_is_rejected_at_the_first_stage_and_never_scored_again() {
        let tmp = tempfile::tempdir().unwrap();
        let scorer = scorer_with(&[("baseline", 0.90), ("c000", 0.50), ("c001", 0.9001)]);
        let l = ladder();
        let out = screen_progressive(
            &scorer,
            tmp.path(),
            &incumbent(),
            vec![candidate("c000", "h_a"), candidate("c001", "h_b")],
            ProgressiveConfig {
                ladder: &l,
                threshold: 0.0,
                batch: 3,
                remaining_after: 0,
                workspace: tmp.path(),
            },
        )
        .unwrap();

        let calls = scorer.calls.borrow();
        assert_eq!(calls.len(), 3, "one call per stage");
        assert!(calls[0].2.contains(&"c000".to_string()));
        for call in calls.iter().skip(1) {
            assert!(
                !call.2.contains(&"c000".to_string()),
                "the obvious loser must not be scored after its rejection"
            );
            assert!(call.2.contains(&"c001".to_string()));
        }
        assert_eq!(
            calls.iter().map(|c| c.0).collect::<Vec<_>>(),
            vec![0.0025, 0.01, 0.05]
        );
        assert_eq!(
            calls.iter().map(|c| c.1).collect::<Vec<_>>(),
            vec![9, 10, 11]
        );

        assert_eq!(out.winners.len(), 1);
        assert_eq!(out.winners[0].candidate.stem, "c001");
        assert_eq!(out.losers.len(), 1);
        assert_eq!(out.losers[0].stage, 0);
        assert_eq!(out.losers[0].reason, ScreenRejection::ClearlyWorse);
        assert!(out.losers[0].delta < 0.0);
        assert_eq!(out.stages.len(), 3);
        assert_eq!(out.stages[0].entered, 2);
        assert_eq!(out.stages[0].rejected, 1);
        assert_eq!(out.stages[1].entered, 1);
        assert_eq!(out.stages[2].outcome, "promoted");
        // 2 candidates + incumbent at stage 0, 1 + incumbent at stages 1 and 2.
        assert_eq!(out.records_scored(), 100 * (3 + 2 + 2));
    }

    #[test]
    fn a_borderline_candidate_collects_more_evidence_instead_of_being_rejected() {
        let tmp = tempfile::tempdir().unwrap();
        // A hair worse than the incumbent — inside every stage's margin.
        let scorer = scorer_with(&[("baseline", 0.90), ("c000", 0.8999)]);
        let l = ladder();
        let out = screen_progressive(
            &scorer,
            tmp.path(),
            &incumbent(),
            vec![candidate("c000", "h_a")],
            ProgressiveConfig {
                ladder: &l,
                threshold: 0.0,
                batch: 0,
                remaining_after: 0,
                workspace: tmp.path(),
            },
        )
        .unwrap();
        assert_eq!(scorer.calls.borrow().len(), 3, "it reached the last stage");
        assert_eq!(out.stages[0].promoted, 1);
        assert_eq!(out.stages[1].promoted, 1);
        // Only the promotion stage applies `--screen-threshold`, so this is
        // where a candidate that never improves finally loses.
        assert!(out.winners.is_empty());
        assert_eq!(out.losers.len(), 1);
        assert_eq!(out.losers[0].stage, 2);
        assert_eq!(out.losers[0].reason, ScreenRejection::BelowThreshold);
    }

    #[test]
    fn a_stage_that_rejects_everything_stops_the_ladder() {
        let tmp = tempfile::tempdir().unwrap();
        let scorer = scorer_with(&[("baseline", 0.90), ("c000", 0.10)]);
        let l = ladder();
        let out = screen_progressive(
            &scorer,
            tmp.path(),
            &incumbent(),
            vec![candidate("c000", "h_a")],
            ProgressiveConfig {
                ladder: &l,
                threshold: 0.0,
                batch: 0,
                remaining_after: 0,
                workspace: tmp.path(),
            },
        )
        .unwrap();
        assert_eq!(
            scorer.calls.borrow().len(),
            1,
            "the larger samples are never paid for"
        );
        assert_eq!(out.stages.len(), 1);
        assert!(out.winners.is_empty());
        assert_eq!(out.losers.len(), 1);
        // A tenth of one promotion-stage creature-score, floored at 1.
        assert_eq!(out.promotion_rate_creatures(&l), 1);
    }

    /// The control must decide exactly what the single fixed screen decides.
    #[test]
    fn the_single_stage_control_reproduces_the_fixed_rate_screen() {
        let tmp = tempfile::tempdir().unwrap();
        let scorer = scorer_with(&[("baseline", 0.90), ("c000", 0.95), ("c001", 0.80)]);
        let l = ScreenLadder::single(0.05).unwrap();
        let out = screen_progressive(
            &scorer,
            tmp.path(),
            &incumbent(),
            vec![candidate("c000", "h_a"), candidate("c001", "h_b")],
            ProgressiveConfig {
                ladder: &l,
                threshold: 0.0,
                batch: 4,
                remaining_after: 0,
                workspace: tmp.path(),
            },
        )
        .unwrap();
        let calls = scorer.calls.borrow();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].0, 0.05);
        assert_eq!(calls[0].1, 4, "the control keeps the batch index as phase");
        assert_eq!(out.winners.len(), 1);
        assert_eq!(out.winners[0].candidate.stem, "c000");
        assert_eq!(out.losers.len(), 1);
        assert_eq!(out.losers[0].reason, ScreenRejection::BelowThreshold);
        assert_eq!(out.promotion_rate_creatures(&l), 3);
        // The control scores in the historical `screen-<batch>` directory.
        assert_eq!(l.stage_dir(tmp.path(), 4, 0), screen_dir(tmp.path(), 4));
    }

    #[test]
    fn a_stage_failure_aborts_the_batch() {
        let tmp = tempfile::tempdir().unwrap();
        let mut scorer = scorer_with(&[("baseline", 0.90), ("c000", 0.95)]);
        scorer.fail_call = Some(1);
        let l = ladder();
        let err = screen_progressive(
            &scorer,
            tmp.path(),
            &incumbent(),
            vec![candidate("c000", "h_a")],
            ProgressiveConfig {
                ladder: &l,
                threshold: 0.0,
                batch: 0,
                remaining_after: 0,
                workspace: tmp.path(),
            },
        )
        .unwrap_err();
        assert!(err.contains("scripted stage failure"), "{err}");
    }
}
