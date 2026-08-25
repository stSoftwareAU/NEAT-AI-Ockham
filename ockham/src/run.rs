//! Run entry: establish the fail-closed incumbent baseline (Issue #2).
//!
//! Pruning is not attempted. A later issue wires the 45-minute loop on top of
//! this gate.

use std::path::PathBuf;
use std::time::Instant;

use neat_core::training_data::TrainingDataConfig;
use serde::Serialize;

use crate::baseline::{AuthoritativeBaseline, establish_baseline};
use crate::cancel::CancelToken;
use crate::config::OckhamConfig;
use crate::corpus::corpus_info;
use crate::incumbent::{Incumbent, IncumbentMeta, load_incumbent};
use crate::journal::{self, Event};
use crate::promote::{FullConfig, evaluate_full};
use crate::scorer::DirectoryScorer;
use crate::stats::{ActivationStats, ensure_activation_stats};
use crate::sweep::{SampledWinner, ScreenConfig, Sweep, draw_seed, screen_batch, screen_dir};
use crate::tags::{CreatureMeta, OckhamProgress};
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
    /// Hidden-neuron activation statistics for the final incumbent.
    pub activation: ActivationStats,
    /// Effective RNG seed.
    pub seed: u64,
    /// Authoritative local accepts during this run.
    pub accepts: u64,
    /// Sweep batches attempted.
    pub experiments: u64,
    /// Why the loop stopped.
    pub stop_reason: String,
    /// Cumulative score gain from the opening parent.
    pub cumulative_delta: f64,
    /// Population re-entry comparison, when a global champion was supplied.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reentry: Option<crate::reentry::ReentryOutcome>,
    /// Optimisation status.
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
    let incumbent_meta = incumbent
        .write_workspace(&workspace)
        .map_err(|e| e.to_string())?;

    let cfg = TrainingDataConfig::new(incumbent.creature.input, incumbent.creature.output);
    let corpus = corpus_info(&config.training_data, &cfg)?;
    log::detail(&format!(
        "corpus {}  {} records in {} files",
        corpus.identity, corpus.record_count, corpus.file_count
    ));
    let creature_meta = CreatureMeta::from_json(&incumbent.text);
    log::detail(&format!(
        "provenance: {} creature tags, {} tagged neurons",
        creature_meta.tags.len(),
        creature_meta.neuron_tags.len()
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

    log::info("computing full-corpus hidden-neuron activation statistics");
    let activation = ensure_activation_stats(
        &incumbent,
        &config.training_data,
        &corpus,
        &workspace,
        crate::stats::DEFAULT_CHUNK_RECORDS,
    )?;
    log::detail(&format!(
        "activation stats: {} hidden neurons, {} records, {}ms{}",
        activation.neurons.len(),
        activation.record_count,
        activation.scan_ms,
        if activation.from_cache {
            " (cache)"
        } else {
            ""
        }
    ));

    std::fs::create_dir_all(&config.output_dir)
        .map_err(|e| format!("{}: {e}", config.output_dir.display()))?;
    std::fs::write(config.output_dir.join("best.json"), &incumbent.text)
        .map_err(|e| format!("best.json: {e}"))?;

    let cancel = CancelToken::new();
    let loop_out = ockham_loop(
        config,
        scorer,
        &corpus,
        incumbent,
        activation,
        baseline.score,
        creature_meta,
        &workspace,
        &cancel,
    )?;

    let reentry = if let Some(path) = &config.global_champion {
        log::info("re-scoring Ockham best against the supplied global champion");
        let best =
            load_incumbent(&config.output_dir.join("best.json")).map_err(|e| e.to_string())?;
        let champion = load_incumbent(path).map_err(|e| e.to_string())?;
        let outcome = crate::reentry::compare_with_champion(
            scorer,
            &config.training_data,
            &best,
            &champion,
            baseline.score,
            config.min_improvement,
            &workspace.join("reentry"),
            &config.output_dir.join("population-candidate.json"),
        )?;
        log::info(&format!(
            "re-entry population_ready={} headroom={:.3e} ockham={} champion={}",
            outcome.population_ready,
            outcome.population_headroom,
            outcome.ockham_score,
            outcome.champion_score
        ));
        Some(outcome)
    } else {
        None
    };

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
        incumbent: incumbent_meta,
        baseline,
        activation: loop_out.activation,
        seed: loop_out.seed,
        accepts: loop_out.accepts,
        experiments: loop_out.experiments,
        stop_reason: loop_out.stop_reason,
        cumulative_delta: loop_out.cumulative_delta,
        reentry,
        optimisation: "complete",
    })
}

struct LoopOut {
    activation: ActivationStats,
    seed: u64,
    accepts: u64,
    experiments: u64,
    stop_reason: String,
    cumulative_delta: f64,
}

#[allow(clippy::too_many_arguments)]
fn ockham_loop(
    config: &OckhamConfig,
    scorer: &dyn DirectoryScorer,
    corpus: &crate::corpus::CorpusInfo,
    mut incumbent: Incumbent,
    mut activation: ActivationStats,
    opening_score: f64,
    mut meta: CreatureMeta,
    workspace: &std::path::Path,
    cancel: &CancelToken,
) -> Result<LoopOut, String> {
    let journal_path = config.output_dir.join("experiments.jsonl");
    let seed = config.seed.unwrap_or_else(draw_seed);
    let mut sweep = Sweep::new(&incumbent.creature, seed);
    log::info(&format!(
        "loop seed={seed}  hidden={}  budget={}s  candidates={}",
        incumbent.hidden_neurons(),
        config.timeout.as_secs(),
        config.candidates
    ));
    journal::append(
        &journal_path,
        &Event::Start {
            seed,
            permutation_identity: sweep.permutation_identity.clone(),
            hidden: incumbent.hidden_neurons(),
            opening_score,
        },
    )?;
    let deadline = Instant::now() + config.timeout;
    let mut accepts = 0u64;
    let mut experiments = 0u64;
    let mut consecutive_fail = 0u32;
    let mut batch_idx = 0u64;
    let mut current_score = opening_score;
    let mut stop_reason = "exhausted".to_string();

    if incumbent.hidden_neurons() == 0 {
        stop_reason = "no-hidden".into();
    }

    while incumbent.hidden_neurons() > 0 {
        if cancel.is_cancelled() {
            stop_reason = "cancelled".into();
            break;
        }
        if Instant::now() >= deadline {
            stop_reason = "timeout".into();
            break;
        }
        if let Some(max) = config.max_experiments
            && experiments >= max
        {
            stop_reason = "max-experiments".into();
            break;
        }
        if sweep.exhausted() {
            stop_reason = "exhausted".into();
            break;
        }

        let (candidates, skips) =
            sweep.fill_batch(&incumbent.creature, &activation, config.candidates);
        let remaining_s = deadline.saturating_duration_since(Instant::now()).as_secs();
        log::info(&format!(
            "batch {batch_idx}: {} candidates, {} skipped, {} hidden left, {remaining_s}s remaining",
            candidates.len(),
            skips.len(),
            sweep.remaining()
        ));
        journal::append(
            &journal_path,
            &Event::Batch {
                batch: batch_idx,
                candidates: candidates.len(),
                skipped: skips.len(),
                remaining: sweep.remaining(),
            },
        )?;
        if candidates.is_empty() {
            batch_idx += 1;
            experiments += 1;
            continue;
        }

        let mut sampled = match config.screen_sample_rate {
            Some(rate) => {
                match screen_batch(
                    scorer,
                    &config.training_data,
                    &incumbent.creature,
                    candidates,
                    ScreenConfig {
                        sample_rate: rate,
                        sample_phase: batch_idx,
                        threshold: config.screen_threshold,
                        remaining_after: sweep.remaining(),
                        dir: &screen_dir(workspace, batch_idx),
                    },
                ) {
                    Ok(screen) => {
                        consecutive_fail = 0;
                        log::detail(&format!(
                            "screen: {} winners / {} losers in {}ms",
                            screen.winners.len(),
                            screen.losers.len(),
                            screen.screen_ms
                        ));
                        journal::append(
                            &journal_path,
                            &Event::Screen {
                                winners: screen.winners.len(),
                                losers: screen.losers.len(),
                                ms: screen.screen_ms,
                            },
                        )?;
                        screen.winners
                    }
                    Err(e) => {
                        consecutive_fail += 1;
                        log::warn(&format!("screen failed: {e}"));
                        if consecutive_fail >= config.max_consecutive_scorer_failures {
                            stop_reason = "scorer-failures".into();
                            break;
                        }
                        batch_idx += 1;
                        experiments += 1;
                        continue;
                    }
                }
            }
            None => candidates
                .into_iter()
                .map(|c| SampledWinner {
                    delta: 1.0,
                    score: 1.0,
                    baseline_score: 0.0,
                    candidate: c,
                })
                .collect(),
        };

        experiments += 1;
        batch_idx += 1;
        if sampled.is_empty() {
            log::detail("no sampled winners; continuing sweep");
            continue;
        }
        if let Some(cap) = config.max_full
            && sampled.len() > cap
        {
            sampled.sort_by(|a, b| {
                b.delta
                    .partial_cmp(&a.delta)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            log::detail(&format!(
                "keeping top {cap} of {} sampled winners by sample Δ for full scoring",
                sampled.len()
            ));
            sampled.truncate(cap);
        }

        log::detail(&format!(
            "full-scoring {} sampled winners plus bundles",
            sampled.len()
        ));
        match evaluate_full(
            scorer,
            &config.training_data,
            &incumbent.creature,
            &activation,
            &sampled,
            FullConfig {
                min_improvement: config.min_improvement,
                dir: &workspace.join(format!("full-{batch_idx}")),
                best_path: Some(&config.output_dir.join("best.json")),
            },
        ) {
            Ok(full) => {
                consecutive_fail = 0;
                log::detail(&format!(
                    "full: {} individuals, {} bundles, {}ms, accepted={}",
                    full.individuals.len(),
                    full.bundles.len(),
                    full.full_ms,
                    full.winner.is_some()
                ));
                journal::append(
                    &journal_path,
                    &Event::Full {
                        individuals: full.individuals.len(),
                        bundles: full.bundles.len(),
                        accepted: full.winner.is_some(),
                        score: full.winner.as_ref().map(|w| w.candidate.score),
                        delta: full.winner.as_ref().map(|w| w.candidate.delta),
                    },
                )?;
                if let Some(win) = full.winner {
                    let last = win.candidate.kind;
                    current_score = win.candidate.score;
                    accepts += 1;
                    meta.retain_neurons(&win.creature);
                    meta.stamp_acceptance(&OckhamProgress {
                        accepts,
                        experiments,
                        opening: opening_score,
                        score: current_score,
                        error: win.candidate.error,
                        last,
                    });
                    let tagged = meta
                        .serialize_with(&win.creature, true)
                        .map_err(|e| format!("tag best.json: {e}"))?;
                    std::fs::write(config.output_dir.join("best.json"), &tagged)
                        .map_err(|e| format!("best.json: {e}"))?;
                    let winners_dir = config.output_dir.join("winners");
                    if let Err(e) = std::fs::create_dir_all(&winners_dir).and_then(|_| {
                        std::fs::write(winners_dir.join(format!("{}.json", win.checksum)), &tagged)
                    }) {
                        log::warn(&format!("winners archive not written: {e}"));
                    }
                    incumbent = Incumbent::from_creature(win.creature, "ockham-best")
                        .map_err(|e| e.to_string())?;
                    log::ok(&format!(
                        "accepted local win score={current_score} Δ={:.3e} hidden={}",
                        win.candidate.delta,
                        incumbent.hidden_neurons()
                    ));
                    if let Some(max) = config.max_accepts
                        && accepts >= max
                    {
                        stop_reason = "max-accepts".into();
                        break;
                    }
                    activation = ensure_activation_stats(
                        &incumbent,
                        &config.training_data,
                        corpus,
                        workspace,
                        crate::stats::DEFAULT_CHUNK_RECORDS,
                    )?;
                    sweep = Sweep::new(&incumbent.creature, seed.wrapping_add(accepts));
                    log::detail(&format!(
                        "restarted sweep after accept; {} hidden remaining",
                        incumbent.hidden_neurons()
                    ));
                }
            }
            Err(e) => {
                consecutive_fail += 1;
                log::warn(&format!("full score failed: {e}"));
                if consecutive_fail >= config.max_consecutive_scorer_failures {
                    stop_reason = "scorer-failures".into();
                    break;
                }
            }
        }
    }

    journal::append(
        &journal_path,
        &Event::Stop {
            reason: stop_reason.clone(),
            accepts,
            experiments,
            final_score: current_score,
            cumulative_delta: current_score - opening_score,
        },
    )?;
    log::info(&format!(
        "stop reason={stop_reason}  accepts={accepts}  experiments={experiments}  Δ={:.3e}",
        current_score - opening_score
    ));

    Ok(LoopOut {
        activation,
        seed,
        accepts,
        experiments,
        stop_reason,
        cumulative_delta: current_score - opening_score,
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
        assert_eq!(run.optimisation, "complete");
        assert_eq!(run.stop_reason, "no-hidden");
        assert_eq!(run.baseline.score, 0.9);
        assert!(cfg.output_dir.join("best.json").exists());
        assert!(run.workspace.join("incumbent.json").exists());
        assert!(run.workspace.join("baseline.json").exists());
        assert_eq!(run.activation.record_count, 2);
        assert!(run.activation.neurons.is_empty());
        assert_eq!(std::fs::read(&cfg.creature).unwrap(), before);
        assert_eq!(
            std::fs::read(cfg.output_dir.join("best.json")).unwrap(),
            before
        );
        let journal = std::fs::read_to_string(cfg.output_dir.join("experiments.jsonl")).unwrap();
        assert!(journal.contains("\"record\":\"start\""));
        assert!(journal.contains("\"record\":\"stop\""));
    }

    #[test]
    fn hidden_creature_can_accept_a_local_win_and_restart_the_sweep() {
        let tmp = tempfile::tempdir().unwrap();
        let creature = tmp.path().join("creature.json");
        let c = crate::fixtures::creature(
            1,
            1,
            vec![
                crate::fixtures::neuron("hidden", "h1", 0.0, Some("IDENTITY")),
                crate::fixtures::neuron("output", "output-0", 0.0, Some("IDENTITY")),
            ],
            vec![
                crate::fixtures::synapse("input-0", "h1", 1.0),
                crate::fixtures::synapse("h1", "output-0", 1.0),
            ],
        );
        std::fs::write(&creature, neat_core::creature_to_json_pretty(&c).unwrap()).unwrap();
        let train = tmp.path().join("train");
        std::fs::create_dir(&train).unwrap();
        write_bin_file(
            &train.join("0.bin"),
            &[(vec![1.0f32], vec![1.0f32]), (vec![2.0], vec![2.0])],
        )
        .unwrap();
        let cfg = OckhamConfig {
            creature: creature.clone(),
            training_data: train,
            output_dir: tmp.path().join("out"),
            timeout: Duration::from_secs(30),
            max_experiments: Some(4),
            seed: Some(1),
            candidates: 8,
            ..OckhamConfig::default()
        };
        let before = std::fs::read(&creature).unwrap();
        let scorer = ScriptedScorer {
            baseline_score: 0.50,
            candidate_score: Some(0.80),
            ..ScriptedScorer::ok(0.50, 0.50)
        };
        let run = establish_run(&cfg, &scorer).unwrap();
        assert!(run.accepts >= 1, "accepts={}", run.accepts);
        assert!(run.cumulative_delta > 0.0);
        assert_eq!(std::fs::read(&creature).unwrap(), before);
        assert!(cfg.output_dir.join("experiments.jsonl").exists());
        assert!(
            cfg.output_dir
                .join("winners")
                .read_dir()
                .unwrap()
                .next()
                .is_some()
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
