//! Run entry: establish the fail-closed incumbent baseline (Issue #2).
//!
//! Pruning is not attempted. A later issue wires the 45-minute loop on top of
//! this gate.

use std::collections::HashSet;
use std::path::PathBuf;
use std::time::Instant;

use neat_core::CreatureExport;
use neat_core::training_data::TrainingDataConfig;
use serde::Serialize;

use crate::baseline::{AuthoritativeBaseline, establish_baseline};
use crate::cancel::CancelToken;
use crate::config::OckhamConfig;
use crate::corpus::corpus_info;
use crate::incumbent::{Incumbent, IncumbentMeta, load_incumbent};
use crate::journal::{self, Event};
use crate::learnings::{
    LearningsStore, Outcome, ReplayConfig, ScreenOutcomeKind, Screened, Verdict, default_host,
    file_screens, file_verdicts, known_failures, known_wins, oldest_screened_first, screened_uuids,
};
use crate::promote::{
    FullConfig, FullOutcome, LocalWinner, apply_available, evaluate_full, replay_prefix_plans,
};
use crate::scorer::DirectoryScorer;
use crate::stats::{ActivationStats, ensure_activation_stats};
use crate::sweep::{
    CandidateKind, SampledWinner, ScreenConfig, Sweep, SweepCandidate, draw_seed, propose,
    screen_batch, screen_dir,
};
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
    let ordering = config.ordering_config();
    let started = Instant::now();
    let opening_hidden = incumbent.hidden_neurons();
    let mut sweep = Sweep::with_ordering(&incumbent.creature, &activation, seed, ordering);
    log::info(&format!(
        "loop seed={seed}  ordering={}  randomQuota={}  hidden={}  budget={}s  candidates={}",
        ordering.strategy,
        ordering.random_quota,
        incumbent.hidden_neurons(),
        config.timeout.as_secs(),
        config.candidates
    ));

    let replay_cfg = ReplayConfig {
        max: config.learnings_replay,
        retry_after_secs: crate::learnings::DEFAULT_RETRY_AFTER_SECS,
    };
    let mut store = None;
    let mut known = Vec::new();
    let mut screens: Vec<Screened> = Vec::new();
    if let Some(dir) = &config.learnings_dir {
        let host = config.learnings_host.clone().unwrap_or_else(default_host);
        let s = LearningsStore::new(dir, corpus.identity.clone(), host.clone());
        match s.load() {
            Ok(records) => {
                log::info(&format!(
                    "learnings: {} record(s) from {} host={host}",
                    records.len(),
                    dir.display()
                ));
                known = records;
                store = Some(s);
            }
            Err(e) => log::warn(&format!(
                "learnings unreadable ({e}); continuing without cache"
            )),
        }
        // Coverage only — a screens fault must never stop pruning.
        if let Some(s) = store.as_ref() {
            match s.load_screens() {
                Ok(records) => {
                    log::info(&format!(
                        "screens: {} record(s) from {} host={host}",
                        records.len(),
                        dir.display()
                    ));
                    screens = records;
                }
                Err(e) => log::warn(&format!(
                    "screen coverage unreadable ({e}); continuing without it"
                )),
            }
        }
    }
    let store = store.as_ref();

    // Coverage is fleet state, so it reprioritises the sweep one layer above
    // the ordering strategies — after the identity above is already fixed.
    let unchecked_first = config.unchecked_first_enabled();
    if unchecked_first {
        prefer_unchecked(&mut sweep, &screens, &incumbent.creature);
    }
    journal::append(
        &journal_path,
        &Event::Start {
            seed,
            ordering: ordering.strategy,
            ordering_random_quota: ordering.random_quota,
            permutation_identity: sweep.permutation_identity.clone(),
            unchecked_first: sweep.unchecked_first,
            hidden: incumbent.hidden_neurons(),
            synapses: incumbent.creature.synapses.len(),
            opening_score,
        },
    )?;

    let deadline = Instant::now() + config.timeout;
    let mut accepts = 0u64;
    let mut search_accepts = 0u64;
    let mut experiments = 0u64;
    let mut consecutive_fail = 0u32;
    let mut batch_idx = 0u64;
    let mut current_score = opening_score;
    let mut stop_reason = "exhausted".to_string();
    let mut replay_done = false;
    let mut replay_skipped = HashSet::new();

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

        if !replay_done {
            let tagged: HashSet<String> = meta.neuron_tags.keys().cloned().collect();
            if !tagged.is_empty() {
                log::detail(&format!(
                    "replay: leaving {} tagged neuron(s) untouched (GRQ #4216)",
                    tagged.len()
                ));
            }
            let wins: Vec<String> = known_wins(&known, &incumbent.creature, replay_cfg)
                .into_iter()
                .filter(|u| !replay_skipped.contains(u) && !tagged.contains(u))
                .collect();
            if wins.is_empty() {
                replay_done = true;
                continue;
            }
            let (applied, _) = apply_available(&incumbent.creature, &activation, &wins);
            for u in &wins {
                if !applied.iter().any(|a| a == u) {
                    replay_skipped.insert(u.clone());
                }
            }
            if applied.is_empty() {
                replay_done = true;
                continue;
            }
            log::info(&format!(
                "replay: combining {} of {} known win(s) still on incumbent",
                applied.len(),
                wins.len()
            ));
            let plans = replay_prefix_plans(&applied);
            let sampled: Vec<SampledWinner> = if plans.is_empty() {
                match propose(&incumbent.creature, &activation, &applied[0]) {
                    Ok((kind, creature)) => vec![SampledWinner {
                        candidate: SweepCandidate {
                            uuid: applied[0].clone(),
                            permutation_index: 0,
                            kind,
                            stem: "r000".into(),
                            creature,
                        },
                        score: 0.0,
                        baseline_score: 0.0,
                        delta: 1.0,
                    }],
                    Err(reason) => {
                        log::detail(&format!("replay: skip {}: {reason}", applied[0]));
                        replay_skipped.insert(applied[0].clone());
                        continue;
                    }
                }
            } else {
                Vec::new()
            };
            let probe_n = config.max_full.unwrap_or(8).min(applied.len());
            experiments += 1;
            let mut extra_plans = plans;
            match evaluate_full(
                scorer,
                &config.training_data,
                &incumbent.creature,
                &activation,
                &sampled,
                FullConfig {
                    min_improvement: config.min_improvement,
                    dir: &workspace.join(format!("replay-{experiments}")),
                    best_path: Some(&config.output_dir.join("best.json")),
                    extra_plans: &extra_plans,
                },
            ) {
                Ok(full) => {
                    consecutive_fail = 0;
                    journal_full(&journal_path, &full, started)?;
                    if full.winner.is_none() && sampled.is_empty() && applied.len() > 1 {
                        log::info(&format!(
                            "replay: combined bundle missed; probing {probe_n} known win(s) individually"
                        ));
                        let mut probe = Vec::new();
                        for uuid in applied.iter().take(probe_n) {
                            match propose(&incumbent.creature, &activation, uuid) {
                                Ok((kind, creature)) => probe.push(SampledWinner {
                                    candidate: SweepCandidate {
                                        uuid: uuid.clone(),
                                        permutation_index: 0,
                                        kind,
                                        stem: "r000".into(),
                                        creature,
                                    },
                                    score: 0.0,
                                    baseline_score: 0.0,
                                    delta: 1.0,
                                }),
                                Err(reason) => {
                                    log::detail(&format!("replay: skip {uuid}: {reason}"));
                                    replay_skipped.insert(uuid.clone());
                                }
                            }
                        }
                        if probe.is_empty() {
                            continue;
                        }
                        experiments += 1;
                        extra_plans = Vec::new();
                        match evaluate_full(
                            scorer,
                            &config.training_data,
                            &incumbent.creature,
                            &activation,
                            &probe,
                            FullConfig {
                                min_improvement: config.min_improvement,
                                dir: &workspace.join(format!("replay-{experiments}")),
                                best_path: Some(&config.output_dir.join("best.json")),
                                extra_plans: &extra_plans,
                            },
                        ) {
                            Ok(probe_full) => {
                                consecutive_fail = 0;
                                journal_full(&journal_path, &probe_full, started)?;
                                file_full_outcome(store, &mut known, &probe, &probe_full);
                                if let Some(win) = probe_full.winner {
                                    apply_local_win(
                                        config,
                                        corpus,
                                        workspace,
                                        &mut meta,
                                        &mut incumbent,
                                        &mut activation,
                                        &mut accepts,
                                        experiments,
                                        opening_score,
                                        &mut current_score,
                                        win,
                                        "replay",
                                    )?;
                                    stop_reason = "replay-accepts".into();
                                    break;
                                }
                                continue;
                            }
                            Err(e) => {
                                consecutive_fail += 1;
                                log::warn(&format!("replay probe failed: {e}"));
                                if consecutive_fail >= config.max_consecutive_scorer_failures {
                                    stop_reason = "scorer-failures".into();
                                    break;
                                }
                                continue;
                            }
                        }
                    }
                    file_full_outcome(store, &mut known, &sampled, &full);
                    if let Some(win) = full.winner {
                        apply_local_win(
                            config,
                            corpus,
                            workspace,
                            &mut meta,
                            &mut incumbent,
                            &mut activation,
                            &mut accepts,
                            experiments,
                            opening_score,
                            &mut current_score,
                            win,
                            "replay",
                        )?;
                        stop_reason = "replay-accepts".into();
                        break;
                    }
                    continue;
                }
                Err(e) => {
                    consecutive_fail += 1;
                    log::warn(&format!("replay full score failed: {e}"));
                    if consecutive_fail >= config.max_consecutive_scorer_failures {
                        stop_reason = "scorer-failures".into();
                        break;
                    }
                    continue;
                }
            }
        }

        if sweep.exhausted() {
            stop_reason = "exhausted".into();
            break;
        }

        let avoid = known_failures(
            &known,
            &incumbent.creature,
            replay_cfg,
            crate::incumbent::now_unix(),
        );
        let tagged: HashSet<String> = meta.neuron_tags.keys().cloned().collect();
        if !avoid.is_empty() {
            log::detail(&format!(
                "learnings: skipping {} known failure(s)",
                avoid.len()
            ));
        }
        let (candidates, skips) = sweep.fill_batch_skipping(
            &incumbent.creature,
            &activation,
            config.candidates,
            &avoid,
            &tagged,
        );
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
                        let mut coverage: Vec<(&str, CandidateKind, ScreenOutcomeKind)> =
                            Vec::new();
                        for w in &screen.winners {
                            coverage.push((
                                w.candidate.uuid.as_str(),
                                w.candidate.kind,
                                ScreenOutcomeKind::Winner,
                            ));
                        }
                        for l in &screen.losers {
                            coverage.push((l.uuid.as_str(), l.kind, ScreenOutcomeKind::Loser));
                        }
                        file_batch_screens(
                            store,
                            &mut screens,
                            &coverage,
                            &journal_path,
                            batch_idx,
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
            None => {
                // Screening off: every candidate goes straight to full scoring,
                // so every candidate is checked and must leave a screen record.
                let coverage: Vec<(&str, CandidateKind, ScreenOutcomeKind)> = candidates
                    .iter()
                    .map(|c| (c.uuid.as_str(), c.kind, ScreenOutcomeKind::Winner))
                    .collect();
                file_batch_screens(store, &mut screens, &coverage, &journal_path, batch_idx)?;
                candidates
                    .into_iter()
                    .map(|c| SampledWinner {
                        delta: 1.0,
                        score: 1.0,
                        baseline_score: 0.0,
                        candidate: c,
                    })
                    .collect()
            }
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
            FullConfig::new(
                config.min_improvement,
                &workspace.join(format!("full-{batch_idx}")),
                Some(&config.output_dir.join("best.json")),
            ),
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
                journal_full(&journal_path, &full, started)?;
                file_full_outcome(store, &mut known, &sampled, &full);
                if let Some(win) = full.winner {
                    apply_local_win(
                        config,
                        corpus,
                        workspace,
                        &mut meta,
                        &mut incumbent,
                        &mut activation,
                        &mut accepts,
                        experiments,
                        opening_score,
                        &mut current_score,
                        win,
                        "search",
                    )?;
                    search_accepts += 1;
                    if let Some(max) = config.max_accepts
                        && search_accepts >= max
                    {
                        stop_reason = "max-accepts".into();
                        break;
                    }
                    sweep = Sweep::with_ordering(
                        &incumbent.creature,
                        &activation,
                        seed.wrapping_add(accepts),
                        ordering,
                    );
                    if unchecked_first {
                        prefer_unchecked(&mut sweep, &screens, &incumbent.creature);
                    }
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

    // Coverage is only meaningful with the screen store behind it; without one
    // there is no coverage state to report, so nothing is journalled.
    if store.is_some() {
        let tagged: HashSet<String> = meta.neuron_tags.keys().cloned().collect();
        let cov = crate::coverage::coverage(
            &incumbent.creature,
            &tagged,
            &screens,
            opening_hidden.saturating_sub(incumbent.hidden_neurons()),
        );
        log::info(&cov.summary());
        journal::append(
            &journal_path,
            &Event::Coverage {
                hidden: cov.hidden,
                tagged: cov.tagged,
                checkable: cov.checkable,
                checked: cov.checked,
                cut: cov.cut,
            },
        )?;
    }

    journal::append(
        &journal_path,
        &Event::Stop {
            reason: stop_reason.clone(),
            accepts,
            experiments,
            final_score: current_score,
            cumulative_delta: current_score - opening_score,
            final_hidden: incumbent.hidden_neurons(),
            final_synapses: incumbent.creature.synapses.len(),
            elapsed_ms: started.elapsed().as_millis() as u64,
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

/// Reorder the sweep's unvisited tail unchecked-first, stalest-first (Issue #38).
///
/// Selection only: nothing here removes a neuron from the sweep, so a run that
/// runs out of never-screened neurons recycles the stalest ones instead of
/// stopping. With no screen records the order is unchanged.
fn prefer_unchecked(sweep: &mut Sweep, screens: &[Screened], creature: &CreatureExport) {
    let screened = screened_uuids(screens, creature);
    let oldest = oldest_screened_first(screens, creature);
    let deferred = sweep.order[sweep.next..]
        .iter()
        .filter(|uuid| screened.contains(*uuid))
        .count();
    let unchecked = sweep.remaining() - deferred;
    sweep.prefer_unchecked(&screened, &oldest);
    log::info(&format!(
        "coverage: {unchecked} unchecked first, {deferred} already screened deferred"
    ));
}

fn journal_full(
    path: &std::path::Path,
    full: &FullOutcome,
    started: Instant,
) -> Result<(), String> {
    journal::append(
        path,
        &Event::Full {
            individuals: full.individuals.len(),
            bundles: full.bundles.len(),
            accepted: full.winner.is_some(),
            score: full.winner.as_ref().map(|w| w.candidate.score),
            delta: full.winner.as_ref().map(|w| w.candidate.delta),
            cuts: full.winner.as_ref().map_or(0, |w| w.candidate.uuids.len()),
            elapsed_ms: started.elapsed().as_millis() as u64,
        },
    )
}

/// File one screen-coverage record per checked candidate and journal the count.
///
/// Coverage only: nothing filed here accepts or rejects a prune. Store faults
/// warn inside [`file_screens`], so a learnings IO fault can never fail the run
/// — only the journal write, which is the run's own audit trail, can.
fn file_batch_screens(
    store: Option<&LearningsStore>,
    screens: &mut Vec<Screened>,
    coverage: &[(&str, CandidateKind, ScreenOutcomeKind)],
    journal_path: &std::path::Path,
    batch: u64,
) -> Result<(), String> {
    let n = file_screens(store, coverage, screens);
    if n > 0 {
        log::detail(&format!("screens: filed {n} screen record(s)"));
    }
    journal::append(journal_path, &Event::Screened { batch, screened: n })
}

fn file_full_outcome(
    store: Option<&LearningsStore>,
    known: &mut Vec<crate::learnings::Learning>,
    sampled: &[SampledWinner],
    full: &FullOutcome,
) {
    let win: HashSet<&str> = full
        .winner
        .as_ref()
        .map(|w| w.candidate.uuids.iter().map(String::as_str).collect())
        .unwrap_or_default();
    let mut verdicts = Vec::new();
    let mut seen: HashSet<&str> = HashSet::new();
    for w in sampled {
        if !full
            .individuals
            .iter()
            .any(|i| i.uuids.first() == Some(&w.candidate.uuid))
        {
            continue;
        }
        seen.insert(w.candidate.uuid.as_str());
        verdicts.push(Verdict {
            uuid: w.candidate.uuid.as_str(),
            kind: w.candidate.kind,
            outcome: if win.contains(w.candidate.uuid.as_str()) {
                Outcome::Accepted
            } else {
                Outcome::Rejected
            },
        });
    }
    if let Some(winner) = &full.winner {
        for uuid in &winner.candidate.uuids {
            if seen.contains(uuid.as_str()) {
                continue;
            }
            verdicts.push(Verdict {
                uuid: uuid.as_str(),
                kind: crate::sweep::CandidateKind::Ablation,
                outcome: Outcome::Accepted,
            });
        }
    }
    let n = file_verdicts(store, &verdicts, known);
    if n > 0 {
        log::detail(&format!("learnings: filed {n} full-corpus verdict(s)"));
    }
}

#[allow(clippy::too_many_arguments)]
fn apply_local_win(
    config: &OckhamConfig,
    corpus: &crate::corpus::CorpusInfo,
    workspace: &std::path::Path,
    meta: &mut CreatureMeta,
    incumbent: &mut Incumbent,
    activation: &mut ActivationStats,
    accepts: &mut u64,
    experiments: u64,
    opening_score: f64,
    current_score: &mut f64,
    win: LocalWinner,
    phase: &'static str,
) -> Result<(), String> {
    let last = win.candidate.kind;
    let cuts = win.candidate.uuids.len();
    let origin = if phase == "replay" && cuts > 1 {
        "replay-bundle"
    } else if phase == "replay" {
        "replay"
    } else {
        "search"
    };
    *current_score = win.candidate.score;
    *accepts += 1;
    meta.retain_neurons(&win.creature);
    meta.stamp_acceptance(&OckhamProgress {
        accepts: *accepts,
        experiments,
        opening: opening_score,
        score: *current_score,
        error: win.candidate.error,
        last,
        origin,
        cuts,
    });
    let tagged = meta
        .serialize_with(&win.creature, true)
        .map_err(|e| format!("tag best.json: {e}"))?;
    std::fs::write(config.output_dir.join("best.json"), &tagged)
        .map_err(|e| format!("best.json: {e}"))?;
    let winners_dir = config.output_dir.join("winners");
    if let Err(e) = std::fs::create_dir_all(&winners_dir)
        .and_then(|_| std::fs::write(winners_dir.join(format!("{}.json", win.checksum)), &tagged))
    {
        log::warn(&format!("winners archive not written: {e}"));
    }
    *incumbent =
        Incumbent::from_creature(win.creature, "ockham-best").map_err(|e| e.to_string())?;
    log::ok(&format!(
        "accepted local win score={} Δ={:.3e} hidden={}",
        *current_score,
        win.candidate.delta,
        incumbent.hidden_neurons()
    ));
    *activation = ensure_activation_stats(
        incumbent,
        &config.training_data,
        corpus,
        workspace,
        crate::stats::DEFAULT_CHUNK_RECORDS,
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::baseline::fake::ScriptedScorer;
    use crate::corpus::write_bin_file;
    use crate::fixtures::identity_creature_json;
    use neat_core::training_data::TrainingDataConfig;
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

    fn two_hidden_paths(tmp: &std::path::Path) -> (std::path::PathBuf, std::path::PathBuf) {
        hidden_paths(tmp, &["h_a", "h_b"])
    }

    /// Creature with one parallel hidden IDENTITY neuron per uuid.
    fn hidden_creature(uuids: &[&str]) -> CreatureExport {
        let mut neurons: Vec<neat_core::NeuronExport> = uuids
            .iter()
            .map(|u| crate::fixtures::neuron("hidden", u, 0.0, Some("IDENTITY")))
            .collect();
        neurons.push(crate::fixtures::neuron(
            "output",
            "output-0",
            0.0,
            Some("IDENTITY"),
        ));
        let mut synapses = Vec::new();
        for u in uuids {
            synapses.push(crate::fixtures::synapse("input-0", u, 1.0));
            synapses.push(crate::fixtures::synapse(u, "output-0", 1.0));
        }
        crate::fixtures::creature(1, 1, neurons, synapses)
    }

    /// [`hidden_creature`] written to disk, plus a training corpus.
    fn hidden_paths(
        tmp: &std::path::Path,
        uuids: &[&str],
    ) -> (std::path::PathBuf, std::path::PathBuf) {
        let creature = tmp.join("creature.json");
        let c = hidden_creature(uuids);
        std::fs::write(&creature, neat_core::creature_to_json_pretty(&c).unwrap()).unwrap();
        let train = tmp.join("train");
        std::fs::create_dir(&train).unwrap();
        write_bin_file(
            &train.join("0.bin"),
            &[(vec![1.0f32], vec![1.0f32]), (vec![2.0], vec![2.0])],
        )
        .unwrap();
        (creature, train)
    }

    /// Store pointed at the screen records a run under `train` would have filed.
    fn screens_store(learnings_dir: &std::path::Path, train: &std::path::Path) -> LearningsStore {
        let corpus = crate::corpus::corpus_info(train, &TrainingDataConfig::new(1, 1)).unwrap();
        LearningsStore::new(learnings_dir, corpus.identity, "t".into())
    }

    fn screened_uuids(store: &LearningsStore) -> Vec<String> {
        let mut uuids: Vec<String> = store
            .load_screens()
            .unwrap()
            .into_iter()
            .map(|s| s.uuid)
            .collect();
        uuids.sort();
        uuids
    }

    #[test]
    fn max_accepts_still_stops_new_discoveries() {
        let tmp = tempfile::tempdir().unwrap();
        let (creature, train) = two_hidden_paths(tmp.path());
        let cfg = OckhamConfig {
            creature,
            training_data: train,
            output_dir: tmp.path().join("out"),
            timeout: Duration::from_secs(30),
            max_experiments: Some(8),
            max_accepts: Some(1),
            seed: Some(1),
            candidates: 8,
            screen_sample_rate: None,
            ..OckhamConfig::default()
        };
        let scorer = ScriptedScorer {
            baseline_score: 0.50,
            candidate_score: Some(0.80),
            ..ScriptedScorer::ok(0.50, 0.50)
        };
        let run = establish_run(&cfg, &scorer).unwrap();
        assert_eq!(run.accepts, 1, "accepts={}", run.accepts);
        assert_eq!(run.stop_reason, "max-accepts");
    }

    #[test]
    fn replay_applies_every_known_win_ignoring_max_accepts() {
        let tmp = tempfile::tempdir().unwrap();
        let (creature, train) = two_hidden_paths(tmp.path());
        let learnings_dir = tmp.path().join("learnings");
        let td = TrainingDataConfig::new(1, 1);
        let corpus = crate::corpus::corpus_info(&train, &td).unwrap();
        let store = LearningsStore::new(&learnings_dir, corpus.identity.clone(), "t".into());
        for (uuid, secs) in [("h_b", 20u64), ("h_a", 10)] {
            store
                .append(&crate::learnings::Learning {
                    version: crate::learnings::LEARNINGS_FORMAT_VERSION,
                    uuid: uuid.into(),
                    kind: "identity".into(),
                    outcome: Outcome::Accepted,
                    unix_secs: secs,
                    host: "t".into(),
                })
                .unwrap();
        }
        let cfg = OckhamConfig {
            creature,
            training_data: train,
            output_dir: tmp.path().join("out"),
            timeout: Duration::from_secs(30),
            max_experiments: Some(8),
            max_accepts: Some(1),
            seed: Some(1),
            candidates: 8,
            screen_sample_rate: None,
            learnings_dir: Some(learnings_dir),
            learnings_host: Some("t".into()),
            ..OckhamConfig::default()
        };
        let scorer = ScriptedScorer {
            baseline_score: 0.50,
            candidate_score: Some(0.80),
            ..ScriptedScorer::ok(0.50, 0.50)
        };
        let run = establish_run(&cfg, &scorer).unwrap();
        assert_eq!(run.accepts, 1, "accepts={}", run.accepts);
        assert_eq!(run.stop_reason, "replay-accepts");
        let best = std::fs::read_to_string(cfg.output_dir.join("best.json")).unwrap();
        assert!(best.contains("replay-bundle"), "{best}");
        assert!(!best.contains("h_a"), "{best}");
        assert!(!best.contains("h_b"), "{best}");
    }

    #[test]
    fn replay_leaves_tagged_source_neurons_in_place() {
        let tmp = tempfile::tempdir().unwrap();
        let (creature, train) = two_hidden_paths(tmp.path());
        let mut v: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&creature).unwrap()).unwrap();
        if let Some(neurons) = v.get_mut("neurons").and_then(|n| n.as_array_mut()) {
            for n in neurons {
                if n.get("uuid").and_then(|u| u.as_str()) == Some("h_a") {
                    n.as_object_mut().unwrap().insert(
                        "tags".into(),
                        serde_json::json!([{"name":"discovered","value":"keep"}]),
                    );
                }
            }
        }
        std::fs::write(&creature, serde_json::to_string_pretty(&v).unwrap()).unwrap();
        let learnings_dir = tmp.path().join("learnings");
        let td = TrainingDataConfig::new(1, 1);
        let corpus = crate::corpus::corpus_info(&train, &td).unwrap();
        let store = LearningsStore::new(&learnings_dir, corpus.identity.clone(), "t".into());
        for uuid in ["h_a", "h_b"] {
            store
                .append(&crate::learnings::Learning {
                    version: crate::learnings::LEARNINGS_FORMAT_VERSION,
                    uuid: uuid.into(),
                    kind: "identity".into(),
                    outcome: Outcome::Accepted,
                    unix_secs: 10,
                    host: "t".into(),
                })
                .unwrap();
        }
        let cfg = OckhamConfig {
            creature,
            training_data: train,
            output_dir: tmp.path().join("out"),
            timeout: Duration::from_secs(30),
            max_experiments: Some(4),
            max_accepts: Some(1),
            seed: Some(1),
            candidates: 8,
            screen_sample_rate: None,
            learnings_dir: Some(learnings_dir),
            learnings_host: Some("t".into()),
            ..OckhamConfig::default()
        };
        let scorer = ScriptedScorer {
            baseline_score: 0.50,
            candidate_score: Some(0.80),
            ..ScriptedScorer::ok(0.50, 0.50)
        };
        let run = establish_run(&cfg, &scorer).unwrap();
        let best = std::fs::read_to_string(cfg.output_dir.join("best.json")).unwrap();
        assert!(best.contains("h_a"), "tagged neuron must survive: {best}");
        assert!(
            run.accepts >= 1 || !best.contains("h_b"),
            "untagged known win should prune or accept; accepts={} best={best}",
            run.accepts
        );
    }

    #[test]
    fn the_named_ordering_reaches_the_journal_and_the_report() {
        let tmp = tempfile::tempdir().unwrap();
        let (creature, train) = two_hidden_paths(tmp.path());
        let cfg = OckhamConfig {
            creature,
            training_data: train,
            output_dir: tmp.path().join("out"),
            timeout: Duration::from_secs(30),
            max_experiments: Some(4),
            seed: Some(1),
            candidates: 8,
            screen_sample_rate: None,
            ordering: crate::ordering::Ordering::LowVariance,
            ordering_random_quota: 0.25,
            ..OckhamConfig::default()
        };
        let run = establish_run(&cfg, &ScriptedScorer::ok(0.50, 0.50)).unwrap();
        assert_eq!(run.accepts, 0, "a flat scorer must not accept anything");
        let journal_path = cfg.output_dir.join("experiments.jsonl");
        let journal = std::fs::read_to_string(&journal_path).unwrap();
        assert!(
            journal.contains("\"ordering\":\"low-variance\""),
            "{journal}"
        );
        let report = crate::report::summarise(&[&journal_path]).unwrap();
        assert_eq!(
            report.ordering,
            Some(crate::ordering::Ordering::LowVariance)
        );
        assert_eq!(report.ordering_random_quota, Some(0.25));
        assert_eq!(report.seed, Some(1));
        assert!(report.elapsed_ms.is_some());
        assert_eq!(report.first_win_ms, None, "no win, so no time-to-first-win");
    }

    #[test]
    fn every_screened_candidate_is_filed_across_two_batches() {
        let tmp = tempfile::tempdir().unwrap();
        let (creature, train) = hidden_paths(tmp.path(), &["h_a", "h_b", "h_c", "h_d"]);
        let learnings_dir = tmp.path().join("learnings");
        let cfg = OckhamConfig {
            creature,
            training_data: train.clone(),
            output_dir: tmp.path().join("out"),
            timeout: Duration::from_secs(30),
            max_experiments: Some(2),
            seed: Some(1),
            candidates: 2,
            learnings_dir: Some(learnings_dir.clone()),
            learnings_host: Some("t".into()),
            ..OckhamConfig::default()
        };
        // Flat scorer: every candidate loses the screen, and losers are the
        // bulk of coverage.
        let run = establish_run(&cfg, &ScriptedScorer::ok(0.50, 0.50)).unwrap();
        assert_eq!(run.accepts, 0, "a flat scorer must not accept anything");

        let store = screens_store(&learnings_dir, &train);
        let records = store.load_screens().unwrap();
        assert_eq!(
            records.len(),
            4,
            "one record per candidate scored, no duplicates: {records:?}"
        );
        assert_eq!(
            screened_uuids(&store),
            vec!["h_a", "h_b", "h_c", "h_d"],
            "the union of both batches must be covered"
        );
        assert!(
            records
                .iter()
                .all(|r| r.outcome == crate::learnings::ScreenOutcomeKind::Loser
                    && r.kind == "identity"),
            "{records:?}"
        );

        let journal_path = cfg.output_dir.join("experiments.jsonl");
        let journal = std::fs::read_to_string(&journal_path).unwrap();
        assert_eq!(
            journal.matches(r#""record":"screened""#).count(),
            2,
            "one screened record per batch: {journal}"
        );
        let report = crate::report::summarise(&[&journal_path]).unwrap();
        assert_eq!(report.screened, 4);
    }

    /// Seed one already-screened record per uuid, dated `unix_secs`.
    fn seed_screens(store: &LearningsStore, uuids: &[(&str, u64)]) {
        for (uuid, unix_secs) in uuids {
            store
                .append_screen(&Screened {
                    version: crate::learnings::SCREENS_FORMAT_VERSION,
                    uuid: (*uuid).into(),
                    kind: "identity".into(),
                    outcome: ScreenOutcomeKind::Loser,
                    unix_secs: *unix_secs,
                    host: "t".into(),
                })
                .unwrap();
        }
    }

    /// UUIDs screened after `unix_secs` — i.e. by the run, not the fixture.
    fn screened_this_run(store: &LearningsStore, after: u64) -> Vec<String> {
        let mut uuids: Vec<String> = store
            .load_screens()
            .unwrap()
            .into_iter()
            .filter(|s| s.unix_secs > after)
            .map(|s| s.uuid)
            .collect();
        uuids.sort();
        uuids
    }

    fn unchecked_first_cfg(
        creature: std::path::PathBuf,
        train: std::path::PathBuf,
        out: std::path::PathBuf,
        learnings_dir: std::path::PathBuf,
        unchecked_first: Option<bool>,
    ) -> OckhamConfig {
        OckhamConfig {
            creature,
            training_data: train,
            output_dir: out,
            timeout: Duration::from_secs(30),
            max_experiments: Some(1),
            seed: Some(1),
            candidates: 2,
            learnings_dir: Some(learnings_dir),
            learnings_host: Some("t".into()),
            unchecked_first,
            ..OckhamConfig::default()
        }
    }

    #[test]
    fn the_run_screens_never_checked_neurons_before_stale_ones() {
        let tmp = tempfile::tempdir().unwrap();
        let (creature, train) = hidden_paths(tmp.path(), &["h_a", "h_b", "h_c", "h_d"]);
        let learnings_dir = tmp.path().join("learnings");
        let store = screens_store(&learnings_dir, &train);
        seed_screens(&store, &[("h_a", 10), ("h_b", 20)]);

        let cfg = unchecked_first_cfg(
            creature,
            train.clone(),
            tmp.path().join("out"),
            learnings_dir,
            None,
        );
        assert!(cfg.unchecked_first_enabled(), "--learnings-dir turns it on");
        establish_run(&cfg, &ScriptedScorer::ok(0.50, 0.50)).unwrap();

        assert_eq!(
            screened_this_run(&store, 20),
            vec!["h_c", "h_d"],
            "the one batch must advance coverage, not re-screen h_a/h_b"
        );
        let journal = std::fs::read_to_string(cfg.output_dir.join("experiments.jsonl")).unwrap();
        assert!(journal.contains(r#""unchecked_first":true"#), "{journal}");
    }

    #[test]
    fn unchecked_first_off_keeps_the_seeded_permutation() {
        let tmp = tempfile::tempdir().unwrap();
        let uuids = ["h_a", "h_b", "h_c", "h_d"];
        let (creature, train) = hidden_paths(tmp.path(), &uuids);
        let learnings_dir = tmp.path().join("learnings");
        let store = screens_store(&learnings_dir, &train);
        seed_screens(&store, &[("h_a", 10), ("h_b", 20)]);

        let cfg = unchecked_first_cfg(
            creature,
            train.clone(),
            tmp.path().join("out"),
            learnings_dir,
            Some(false),
        );
        establish_run(&cfg, &ScriptedScorer::ok(0.50, 0.50)).unwrap();

        let mut expected: Vec<String> =
            crate::ordering::random_order(&hidden_creature(&uuids), 1)[..2].to_vec();
        expected.sort();
        assert_eq!(
            screened_this_run(&store, 20),
            expected,
            "with the flag off the raw seeded permutation must be untouched"
        );
        let journal = std::fs::read_to_string(cfg.output_dir.join("experiments.jsonl")).unwrap();
        assert!(journal.contains(r#""unchecked_first":false"#), "{journal}");
    }

    #[test]
    fn the_run_journals_coverage_over_the_final_incumbent() {
        let tmp = tempfile::tempdir().unwrap();
        let (creature, train) = hidden_paths(tmp.path(), &["h_a", "h_b", "h_c", "h_d"]);
        let learnings_dir = tmp.path().join("learnings");
        let cfg = OckhamConfig {
            creature,
            training_data: train,
            output_dir: tmp.path().join("out"),
            timeout: Duration::from_secs(30),
            max_experiments: Some(1),
            seed: Some(1),
            candidates: 2,
            learnings_dir: Some(learnings_dir),
            learnings_host: Some("t".into()),
            ..OckhamConfig::default()
        };
        let run = establish_run(&cfg, &ScriptedScorer::ok(0.50, 0.50)).unwrap();
        assert_eq!(run.accepts, 0, "a flat scorer must not accept anything");

        let journal_path = cfg.output_dir.join("experiments.jsonl");
        let report = crate::report::summarise(&[&journal_path]).unwrap();
        assert_eq!(report.hidden, Some(4));
        assert_eq!(report.checked, Some(2), "one batch of two candidates");
        assert_eq!(report.coverage_percent, Some(50.0));
    }

    #[test]
    fn a_run_without_a_learnings_dir_journals_no_coverage() {
        let tmp = tempfile::tempdir().unwrap();
        let (creature, train) = two_hidden_paths(tmp.path());
        let cfg = OckhamConfig {
            creature,
            training_data: train,
            output_dir: tmp.path().join("out"),
            timeout: Duration::from_secs(30),
            max_experiments: Some(1),
            seed: Some(1),
            candidates: 2,
            ..OckhamConfig::default()
        };
        establish_run(&cfg, &ScriptedScorer::ok(0.50, 0.50)).unwrap();
        let journal_path = cfg.output_dir.join("experiments.jsonl");
        let journal = std::fs::read_to_string(&journal_path).unwrap();
        assert!(!journal.contains(r#""record":"coverage""#), "{journal}");
        let report = crate::report::summarise(&[&journal_path]).unwrap();
        assert_eq!(
            report.coverage_percent, None,
            "no screen store means no coverage state, not 0%"
        );
    }

    #[test]
    fn screening_disabled_still_files_a_record_for_every_candidate() {
        let tmp = tempfile::tempdir().unwrap();
        let (creature, train) = two_hidden_paths(tmp.path());
        let learnings_dir = tmp.path().join("learnings");
        let cfg = OckhamConfig {
            creature,
            training_data: train.clone(),
            output_dir: tmp.path().join("out"),
            timeout: Duration::from_secs(30),
            max_experiments: Some(1),
            seed: Some(1),
            candidates: 8,
            screen_sample_rate: None,
            learnings_dir: Some(learnings_dir.clone()),
            learnings_host: Some("t".into()),
            ..OckhamConfig::default()
        };
        let run = establish_run(&cfg, &ScriptedScorer::ok(0.50, 0.50)).unwrap();
        assert_eq!(run.accepts, 0);

        let store = screens_store(&learnings_dir, &train);
        assert_eq!(
            screened_uuids(&store),
            vec!["h_a", "h_b"],
            "candidates that reach full scoring are checked too"
        );

        let journal_path = cfg.output_dir.join("experiments.jsonl");
        let journal = std::fs::read_to_string(&journal_path).unwrap();
        assert!(
            !journal.contains(r#""record":"screen""#),
            "no sampled scorer call happened: {journal}"
        );
        let report = crate::report::summarise(&[&journal_path]).unwrap();
        assert_eq!(report.screen_calls, 0);
        assert_eq!(report.screened, 2);
    }

    #[test]
    fn a_failed_screen_files_no_screen_records() {
        let tmp = tempfile::tempdir().unwrap();
        let (creature, train) = hidden_paths(tmp.path(), &["h_a", "h_b", "h_c", "h_d", "h_e"]);
        let learnings_dir = tmp.path().join("learnings");
        let cfg = OckhamConfig {
            creature,
            training_data: train.clone(),
            output_dir: tmp.path().join("out"),
            timeout: Duration::from_secs(30),
            seed: Some(1),
            candidates: 1,
            learnings_dir: Some(learnings_dir.clone()),
            learnings_host: Some("t".into()),
            ..OckhamConfig::default()
        };
        let scorer = ScriptedScorer {
            fail_sample_with: Some("screen exploded".into()),
            ..ScriptedScorer::ok(0.50, 0.50)
        };
        let run = establish_run(&cfg, &scorer).unwrap();
        assert_eq!(run.stop_reason, "scorer-failures");

        let store = screens_store(&learnings_dir, &train);
        assert!(
            store.load_screens().unwrap().is_empty(),
            "candidates whose screen errored were never checked"
        );
        assert!(!store.screens_dir().exists());
        let journal = std::fs::read_to_string(cfg.output_dir.join("experiments.jsonl")).unwrap();
        assert!(!journal.contains(r#""record":"screened""#), "{journal}");
    }

    #[test]
    fn omitted_learnings_dir_files_no_screen_records() {
        let tmp = tempfile::tempdir().unwrap();
        let (creature, train) = two_hidden_paths(tmp.path());
        let cfg = OckhamConfig {
            creature,
            training_data: train,
            output_dir: tmp.path().join("out"),
            timeout: Duration::from_secs(30),
            max_experiments: Some(2),
            seed: Some(1),
            candidates: 8,
            ..OckhamConfig::default()
        };
        let run = establish_run(&cfg, &ScriptedScorer::ok(0.50, 0.50)).unwrap();
        assert_eq!(run.accepts, 0);
        let stray: Vec<std::path::PathBuf> = std::fs::read_dir(tmp.path())
            .unwrap()
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| {
                p.file_name()
                    .and_then(|n| n.to_str())
                    .is_some_and(|n| n.starts_with("screens-") || n == "learnings")
            })
            .collect();
        assert!(stray.is_empty(), "{stray:?}");
        assert!(!cfg.output_dir.join("learnings").exists());
    }

    #[test]
    fn omitted_learnings_dir_writes_nothing() {
        let tmp = tempfile::tempdir().unwrap();
        let cfg = config(tmp.path());
        let _ = establish_run(&cfg, &ScriptedScorer::ok(0.9, 0.1)).unwrap();
        assert!(!tmp.path().join("learnings").exists());
        assert!(
            !cfg.output_dir.join("learnings").exists(),
            "loop must not invent a learnings dir"
        );
    }
}
