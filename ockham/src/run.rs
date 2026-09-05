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
    HistoricalLearning, LearningsStore, Outcome, ReplayConfig, ScreenOutcomeKind, ScreenTry,
    Screened, Verdict, default_host, file_screens, file_verdicts, historical_replay,
    history_epochs, known_failures, oldest_screened_first, prior_corpus_priority, ranked_confirmed,
    replay_cap, screened_uuids,
};
use crate::promote::{
    BundleMember, FullConfig, FullOutcome, LocalWinner, REPLAY_PROBE_LIMIT, apply_available,
    evaluate_full, replay_plans,
};
use crate::scorer::DirectoryScorer;
use crate::screening::{ProgressiveConfig, screen_progressive};
use crate::stats::{ActivationStats, ensure_activation_stats};
use crate::sweep::{CandidateKind, SampledWinner, Sweep, SweepCandidate, draw_seed, propose};
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
    /// Distinct hidden UUIDs this run screened for the first time (Issue #77).
    pub newly_screened: usize,
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
        "tags: {} creature tags, {} tagged neurons",
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

    let sample = config.stats_sample_spec();
    log::info(&format!(
        "computing hidden-neuron activation statistics ({})",
        if sample.max_records == 0 {
            "full corpus".to_string()
        } else {
            format!("sampled, up to {} records", sample.max_records)
        }
    ));
    let activation = ensure_activation_stats(
        &incumbent,
        &config.training_data,
        &corpus,
        &workspace,
        crate::stats::DEFAULT_CHUNK_RECORDS,
        &sample,
    )?;
    log::detail(&format!(
        "activation stats: {} hidden neurons, {}/{} records, {}ms{}{}",
        activation.neurons.len(),
        activation.record_count,
        activation.corpus_record_count,
        activation.scan_ms,
        if activation.stopped_early {
            " (converged)"
        } else {
            ""
        },
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
        baseline.scorer_ms,
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
        newly_screened: loop_out.newly_screened,
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
    newly_screened: usize,
    cumulative_delta: f64,
}

/// Most confirmed winners carried between batches (Issue #56).
///
/// The pool costs nothing to score — its members join bundles only — but it
/// does feed the plan generator, so it is bounded rather than unbounded.
pub const MAX_CONFIRMED_POOL: usize = 64;

/// Weight of the newest cohort in the rolling cost estimate (Issue #58).
const FULL_COST_SMOOTHING: f64 = 0.3;

/// Safety multiple applied to a cost estimate derived from a screen (Issue #58).
const SCREEN_FALLBACK_SAFETY: f64 = 1.5;

/// Full-corpus multiple assumed when the screen sample rate is unknown.
const SCREEN_FALLBACK_MULTIPLE: f64 = 20.0;

/// Fraction of the remaining budget reserved for applying a win (Issue #58).
///
/// `apply_local_win` re-scans activation statistics and writes `best.json`
/// after the cohort returns; a cohort sized to the whole budget would leave
/// nothing to check the win in with.
const BUDGET_RESERVE_FRACTION: f64 = 0.25;

/// How big a full-corpus cohort the remaining wall clock can pay for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CohortBudget {
    /// Nothing measured yet — launch the cohort untrimmed.
    Unmeasured,
    /// At most this many entries besides the incumbent baseline.
    Entries(usize),
    /// Not even a minimal cohort fits; do not start a call that will overrun.
    TooSmall,
}

/// Rolling estimate of full-corpus scorer cost per creature (Issue #58).
///
/// Seeded from the first full cohort of the run and smoothed afterwards, so a
/// single anomalous cohort cannot permanently distort it. Before any full
/// cohort has run, the observed screen cost stands in: a screen scores the same
/// creatures over `sample_rate` of the corpus, so the full cost is roughly the
/// screen cost divided by that rate, with a safety multiple on top.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
struct CostModel {
    full_per_creature_ms: Option<f64>,
    screen_per_creature_ms: Option<f64>,
    baseline_per_creature_ms: Option<f64>,
    sample_rate: Option<f64>,
    /// What a whole screening batch costs relative to one pass at
    /// [`Self::sample_rate`] — `1.0` for the fixed-rate control (#104).
    batch_multiple: f64,
}

impl CostModel {
    fn new(sample_rate: Option<f64>, batch_multiple: f64) -> Self {
        Self {
            sample_rate,
            batch_multiple,
            ..Self::default()
        }
    }

    /// Record the opening authoritative baseline: one creature, full corpus.
    ///
    /// Kept apart from [`Self::full_per_creature_ms`] deliberately (Issue #77).
    /// It exists to size the screening reserve *before* any cohort has run,
    /// and cohort sizing keeps its own conservative estimate: a single-creature
    /// call is a weaker sample than a cohort, and folding it into the smoothed
    /// rolling estimate would drag every later cohort's sizing towards it.
    fn observe_baseline(&mut self, baseline_ms: u64) {
        if baseline_ms > 0 {
            self.baseline_per_creature_ms = Some(baseline_ms as f64);
        }
    }

    /// Milliseconds one screening batch of `candidates` is expected to cost.
    ///
    /// Measured screen cost first; otherwise a full-corpus estimate scaled by
    /// the sample rate, because a screen scores the same creatures over that
    /// fraction of the corpus. With screening disabled there is no cheap
    /// check — the batch *is* a full cohort — so the rate is 1. `None` while
    /// nothing at all has been measured.
    ///
    /// A progressive ladder runs several rungs per batch, so the estimate is
    /// scaled by [`Self::batch_multiple`]; without it a ladder batch is priced
    /// as though only the promotion stage ran, and the screening reserve of #77
    /// is sized against the wrong number (#104). Both figures the model keeps
    /// are per creature *at the promotion rate*, so the multiple applies either
    /// way. It prices the worst case — nothing rejected early — because a
    /// reserve that only covers the lucky batch is not a reserve.
    fn screen_batch_ms(&self, candidates: usize) -> Option<f64> {
        // The incumbent baseline is scored alongside every batch.
        let creatures = candidates as f64 + 1.0;
        if let Some(per_creature) = self.screen_per_creature_ms {
            return Some(per_creature * creatures * self.batch_multiple);
        }
        let full = self
            .full_per_creature_ms
            .or(self.baseline_per_creature_ms)?;
        let rate = match self.sample_rate {
            Some(rate) if rate > 0.0 => rate,
            _ => 1.0,
        };
        Some(full * rate * creatures * self.batch_multiple)
    }

    /// Record one sampled screen: `creatures` includes the incumbent.
    fn observe_screen(&mut self, screen_ms: u64, creatures: usize) {
        if creatures == 0 {
            return;
        }
        self.screen_per_creature_ms = Some(screen_ms as f64 / creatures as f64);
    }

    /// Record one full cohort: `entries` excludes the incumbent baseline.
    fn observe_full(&mut self, full_ms: u64, entries: usize) {
        let sample = full_ms as f64 / (entries + 1) as f64;
        self.full_per_creature_ms = Some(match self.full_per_creature_ms {
            None => sample,
            Some(prev) => prev * (1.0 - FULL_COST_SMOOTHING) + sample * FULL_COST_SMOOTHING,
        });
    }

    /// Best available milliseconds-per-creature estimate.
    fn per_creature_ms(&self) -> Option<f64> {
        if let Some(ms) = self.full_per_creature_ms {
            return Some(ms);
        }
        let screen = self.screen_per_creature_ms?;
        let multiple = match self.sample_rate {
            Some(rate) if rate > 0.0 => 1.0 / rate,
            _ => SCREEN_FALLBACK_MULTIPLE,
        };
        Some(screen * multiple * SCREEN_FALLBACK_SAFETY)
    }

    /// Entries the remaining wall clock can pay for, keeping a check-in reserve.
    fn cohort_budget(&self, remaining: std::time::Duration) -> CohortBudget {
        let Some(per_creature) = self.per_creature_ms().filter(|ms| *ms > 0.0) else {
            return CohortBudget::Unmeasured;
        };
        let usable = remaining.as_millis() as f64 * (1.0 - BUDGET_RESERVE_FRACTION);
        // The incumbent baseline is scored in every cohort and buys no cut.
        let creatures = (usable / per_creature).floor();
        if !creatures.is_finite() || creatures < 2.0 {
            return CohortBudget::TooSmall;
        }
        CohortBudget::Entries(creatures as usize - 1)
    }
}

/// Most of the run budget one screening batch may claim as its reserve (#77).
///
/// A batch that cannot fit in half the budget is not a reserve, it is the whole
/// plan; above this share the run keeps its pre-#77 behaviour rather than
/// standing the replay stage down for every pass.
const MAX_SCREEN_RESERVE_SHARE: f64 = 0.5;

/// Whether the wall clock left is down to this run's last screening batch.
///
/// The reserve of Issue #77, and its whole judgement call. The run's job is to
/// advance coverage, so once the budget has fallen to the cost of one screening
/// batch the replay stage stands down and the sweep takes what is left — inside
/// the budget, never past the deadline.
///
/// It is sized at exactly one batch, and it is claimed only when a run has
/// screened **nothing** so far. That is deliberately the smallest reserve that
/// can exist, because the cost falls on full-corpus scoring, which is where
/// accepts actually come from: reserve too much and the fleet screens
/// diligently while pruning nothing — rising coverage that reads as success for
/// weeks. Nothing is held back while the budget is healthy, and nothing is held
/// back at all once the run has advanced coverage once.
fn reserve_stands(
    cost: &CostModel,
    config: &OckhamConfig,
    remaining: std::time::Duration,
    screened_batches: u64,
) -> bool {
    if screened_batches > 0 {
        // This run has already advanced coverage; there is nothing left to
        // guarantee, and replay keeps every millisecond that remains.
        return false;
    }
    let Some(batch_ms) = cost.screen_batch_ms(config.candidates) else {
        // Nothing measured: guessing a reserve would cost replay a cohort for
        // a number we do not have.
        return false;
    };
    let cap = config.timeout.as_millis() as f64 * MAX_SCREEN_RESERVE_SHARE;
    batch_ms <= cap && remaining.as_millis() as f64 <= batch_ms
}

/// What a run tried, kept and rejected — the winners block of the commit
/// description (Issue #59).
#[derive(Debug, Clone, Default)]
struct WinnerTally {
    screened: usize,
    confirmed: HashSet<String>,
    applied: usize,
    plans: usize,
    skipped: usize,
    dropped: usize,
    best_cuts: usize,
    best_delta: f64,
}

impl WinnerTally {
    /// Fold one full-score cohort into the tally.
    fn observe(&mut self, full: &FullOutcome, min_improvement: f64) {
        self.plans += full.bundles.len();
        self.skipped += full.skipped_bundles;
        self.dropped += full.dropped();
        for cand in &full.individuals {
            if cand.delta > min_improvement
                && let Some(uuid) = cand.uuids.first()
            {
                self.confirmed.insert(uuid.clone());
            }
        }
        if let Some(win) = &full.winner {
            self.applied += win.candidate.uuids.len();
            self.confirmed.extend(win.candidate.uuids.iter().cloned());
            if win.candidate.uuids.len() > self.best_cuts {
                self.best_cuts = win.candidate.uuids.len();
                self.best_delta = win.candidate.delta;
            }
        }
    }

    /// Render for [`crate::coverage::Winners`], with the pool still standing.
    fn finish(&self, carried: usize, est_ms_per_creature: Option<f64>) -> crate::coverage::Winners {
        crate::coverage::Winners {
            screened: self.screened,
            confirmed: self.confirmed.len(),
            applied: self.applied,
            carried,
            plans: self.plans,
            skipped: self.skipped,
            best_cuts: self.best_cuts,
            best_delta: self.best_delta,
            dropped: self.dropped,
            est_ms_per_creature: est_ms_per_creature.unwrap_or(0.0).round() as u64,
        }
    }
}

/// The `full-scoring …` line for one batch (Issue #54).
///
/// The old line — `keeping top 8 of 38 sampled winners by sample Δ for full
/// scoring` — read as though bundles had been truncated to eight too, which is
/// exactly what was happening and exactly what #45 was raised about. These
/// numbers are now separate: how many winners are scored **individually**, and
/// how many are offered to bundle construction.
fn full_scoring_line(max_full: Option<usize>, sampled: usize, carried: usize) -> String {
    let individuals = max_full.map_or(sampled, |cap| cap.min(sampled));
    let bundled = sampled + carried;
    if individuals < sampled {
        return format!(
            "full-scoring {individuals} of {sampled} sampled winners individually; bundling all {bundled}"
        );
    }
    if carried == 0 {
        return format!("full-scoring {sampled} sampled winners plus bundles");
    }
    format!(
        "full-scoring {sampled} sampled winners individually; bundling all {bundled} ({carried} carried)"
    )
}

/// What a whole ladder batch costs relative to one promotion-stage pass (#104).
///
/// `1.0` for the fixed-rate control; `0.0025 + 0.01 + 0.05` over `0.05` — 1.25 —
/// for the documented ladder, the price when nothing is rejected early.
fn batch_cost_multiple(ladder: &crate::screening::ScreenLadder) -> f64 {
    let promotion = ladder.promotion_rate();
    ladder.stages().iter().map(|s| s.rate).sum::<f64>() / promotion
}

/// The per-stage ladder line for one progressive screen (Issue #104).
///
/// The saving is the point, so it is stated rather than left to be derived:
/// what each rung cost in records and how many candidates it ended.
fn screen_ladder_line(screen: &crate::screening::ProgressiveScreen, candidates: usize) -> String {
    let rungs: Vec<String> = screen
        .stages
        .iter()
        .map(|s| {
            format!(
                "{:.4}%: {} in, {} rejected, {}ms",
                s.rate * 100.0,
                s.entered,
                s.rejected,
                s.ms
            )
        })
        .collect();
    let per_candidate = if candidates == 0 {
        0
    } else {
        screen.records_scored() / candidates as u64
    };
    let (clearly_worse, below_threshold) = screen.rejection_tally();
    format!(
        "ladder: {} | {} records over {candidates} candidate(s) ({per_candidate}/candidate); \
         rejected {clearly_worse} clearly-worse, {below_threshold} below-threshold",
        rungs.join(" -> "),
        screen.records_scored()
    )
}

/// The cohort-trim line (Issue #58) — a silent cap reads as "we tried
/// everything" when we did not.
fn budget_trim_line(
    remaining: std::time::Duration,
    per_creature_ms: f64,
    full: &FullOutcome,
) -> String {
    format!(
        "full: budget {}s, est {:.0}s/creature → {} of {} entries; dropped {} ({} individual, {} bundle)",
        remaining.as_secs(),
        per_creature_ms / 1000.0,
        full.entries(),
        full.entries() + full.dropped(),
        full.dropped(),
        full.dropped_individuals,
        full.dropped_bundles
    )
}

/// Drop pool members the incumbent no longer carries, keeping ranked order.
///
/// A member is kept only when it still proposes **after** the members ranked
/// above it — the same order a bundle applies them in — so a plan built from
/// what survives can never be voided by a cut that has gone stale.
fn standing_pool(
    pool: &[BundleMember],
    incumbent: &CreatureExport,
    stats: &ActivationStats,
) -> Vec<BundleMember> {
    if pool.is_empty() {
        return Vec::new();
    }
    let uuids: Vec<String> = pool.iter().map(|m| m.uuid.clone()).collect();
    let (applied, _) = apply_available(incumbent, stats, &uuids);
    let kept: HashSet<&str> = applied.iter().map(String::as_str).collect();
    pool.iter()
        .filter(|m| kept.contains(m.uuid.as_str()))
        .cloned()
        .collect()
}

/// Fold one cohort's individual verdicts into the carried-winner pool.
///
/// The latest verdict wins: a uuid measured at or below `min_improvement`
/// leaves, and an applied cut leaves because it is no longer on the creature.
fn update_pool(
    pool: &mut Vec<BundleMember>,
    sampled: &[SampledWinner],
    full: &FullOutcome,
    min_improvement: f64,
) {
    let applied: HashSet<&str> = full
        .winner
        .as_ref()
        .map(|w| w.candidate.uuids.iter().map(String::as_str).collect())
        .unwrap_or_default();
    for cand in &full.individuals {
        let Some(uuid) = cand.uuids.first() else {
            continue;
        };
        pool.retain(|m| m.uuid != *uuid);
        if applied.contains(uuid.as_str()) || cand.delta <= min_improvement {
            continue;
        }
        let kind = sampled
            .iter()
            .find(|w| w.candidate.uuid == *uuid)
            .map_or(CandidateKind::Ablation, |w| w.candidate.kind);
        pool.push(BundleMember {
            uuid: uuid.clone(),
            kind,
            delta: cand.delta,
        });
    }
    pool.sort_by(|a, b| {
        b.delta
            .partial_cmp(&a.delta)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.uuid.cmp(&b.uuid))
    });
    if pool.len() > MAX_CONFIRMED_POOL {
        let dropped = pool.len() - MAX_CONFIRMED_POOL;
        pool.truncate(MAX_CONFIRMED_POOL);
        log::detail(&format!(
            "pool: dropped {dropped} weakest confirmed winner(s); {MAX_CONFIRMED_POOL} carried"
        ));
    }
}

#[allow(clippy::too_many_arguments)]
fn ockham_loop(
    config: &OckhamConfig,
    scorer: &dyn DirectoryScorer,
    corpus: &crate::corpus::CorpusInfo,
    mut incumbent: Incumbent,
    mut activation: ActivationStats,
    opening_score: f64,
    baseline_ms: u64,
    mut meta: CreatureMeta,
    workspace: &std::path::Path,
    cancel: &CancelToken,
) -> Result<LoopOut, String> {
    let journal_path = config.output_dir.join("experiments.jsonl");
    let seed = config.seed.unwrap_or_else(draw_seed);
    let ordering = config.ordering_config();
    let started = Instant::now();
    let opening_hidden = incumbent.hidden_neurons();
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
    // Indexed before the epoch filter below, so the cumulative figures survive
    // a corpus change that resets current-epoch coverage to zero (Issue #102).
    let mut screen_history = crate::coverage::ScreenHistory::default();
    // Deliberately its own set, never merged into `known` (Issues #88, #101): a
    // verdict from another corpus may reorder the sweep and be replayed as a
    // hypothesis, and nothing else — it can neither suppress nor accept a cut.
    let mut prior_records: Vec<HistoricalLearning> = Vec::new();
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
                    // Coverage is authoritative only for the corpus it was
                    // measured against (#100): the whole history is read, and
                    // the epoch of the corpus in hand is what counts as
                    // coverage. Extending the training data therefore opens a
                    // fresh epoch at 0/hidden without losing a single record.
                    let history = records.len();
                    screen_history = crate::coverage::ScreenHistory::new(&records);
                    let epoch = crate::learnings::current_epoch_screens(records, &corpus.identity);
                    log::info(&format!(
                        "screens: {} of {history} record(s) from {} are current-epoch \
                         coverage (corpus {}) host={host}",
                        epoch.len(),
                        dir.display(),
                        corpus.identity
                    ));
                    screens = epoch;
                }
                Err(e) => log::warn(&format!(
                    "screen coverage unreadable ({e}); continuing without it"
                )),
            }
        }
        // Evidence, so a fault here costs the priority and the replay
        // hypotheses, and nothing else (#88, #101).
        if let Some(s) = store.as_ref()
            && config.old_corpus_first_enabled()
        {
            match s.load_prior_corpora() {
                Ok(records) => {
                    // Each epoch is named, not just counted: a corpus change
                    // adds evidence rather than losing it, and that is only
                    // visible if the log says which corpus taught what (#101).
                    let epochs = history_epochs(&records);
                    log::info(&format!(
                        "prior corpora: {} verdict(s) from {} across {} historical epoch(s), \
                         read as priority and replay hypotheses",
                        records.len(),
                        dir.display(),
                        epochs.len()
                    ));
                    for (identity, n) in &epochs {
                        log::detail(&format!("history: corpus {identity} — {n} verdict(s)"));
                    }
                    prior_records = records;
                }
                Err(e) => log::warn(&format!(
                    "prior-corpus verdicts unreadable ({e}); continuing without the priority \
                     or the historical replay hypotheses it carries"
                )),
            }
        }
    }
    let store = store.as_ref();
    // On only where there is a cache to have read old corpora from: without a
    // store the hint has no records and nothing to say about them.
    let prior = if store.is_some() && config.old_corpus_first_enabled() {
        PriorHint {
            enabled: true,
            records: &prior_records,
            corpus_identity: &corpus.identity,
            min_improvement: config.min_improvement,
        }
    } else {
        PriorHint::none()
    };

    // What the composite and learned rankings read beyond the creature (#107):
    // the fleet's own history as a prior, and a fitted model when one was
    // configured. Built once, borrowed by every sweep this run rebuilds.
    // Historical epochs only (#88, #101): a verdict this corpus has already
    // returned is current truth the sweep acts on elsewhere, not a prior, and
    // folding the two into one counter would leave neither the ranking nor the
    // training rows able to tell them apart.
    let evidence =
        crate::features::PriorEvidence::from_history(&prior_records, config.min_improvement);
    let model = match &config.ordering_model {
        Some(path) => Some(crate::model::PriorityModel::load(path)?),
        None => None,
    };
    if let Some(model) = &model {
        log::info(&format!(
            "ordering model: {} row(s), {} win(s), fitted by {} — ranking only (#107)",
            model.training().rows,
            model.training().wins,
            model.training().crate_version
        ));
    }
    let priority = crate::priority::PriorityContext::with(evidence, model);
    // A `learned` run whose model never loaded stops here rather than ranking
    // by something it was not asked for.
    let ordering = ordering.with_priority(&priority);
    ordering.validate_ranker()?;
    let candidate_log =
        config
            .candidate_log
            .as_deref()
            .map(|path| crate::telemetry::CandidateLog {
                path,
                stamp: crate::telemetry::RunStamp {
                    host: config.learnings_host.clone().unwrap_or_else(default_host),
                    corpus_identity: corpus.identity.clone(),
                    // Stamped per row from the incumbent the features were read
                    // from; an accept moves the incumbent mid-run.
                    creature_checksum: incumbent.checksum.clone(),
                    ordering: ordering.effective_strategy().name().to_string(),
                    seed,
                },
                evidence: &priority.evidence,
            });

    // Coverage is fleet state, so it reprioritises the sweep one layer above
    // the ordering strategies — after the identity above is already fixed.
    let unchecked_first = config.unchecked_first_enabled();
    let mut progress = crate::coverage::ScreenProgress::new(&screens);
    let mut sweep = fresh_sweep(
        &incumbent.creature,
        &activation,
        seed,
        ordering,
        unchecked_first,
        &screens,
        &prior,
    );
    journal::append(
        &journal_path,
        &Event::Start {
            seed,
            ordering: ordering.strategy,
            ordering_random_quota: ordering.random_quota,
            permutation_identity: sweep.permutation_identity.clone(),
            unchecked_first: sweep.unchecked_first,
            old_corpus_first: sweep.old_corpus_first,
            hidden: incumbent.hidden_neurons(),
            synapses: incumbent.creature.synapses.len(),
            opening_score,
        },
    )?;

    let deadline = Instant::now() + config.timeout;
    let mut accepts = 0u64;
    let mut experiments = 0u64;
    let mut consecutive_fail = 0u32;
    let mut batch_idx = 0u64;
    let mut current_score = opening_score;
    // The loop can only fall out of its `while` when the creature has no hidden
    // neurons left, either because it opened that way or because accepts pruned
    // the last one. Every other exit sets its own reason and breaks. The
    // "exhausted" reason is retired (Issue #77): an exhausted sweep restarts,
    // so it can never be why a run stopped.
    let mut stop_reason = "no-hidden".to_string();
    // Sweep-restart bookkeeping (Issue #77). `pass_candidates` counts what the
    // current permutation proposed, so a pass that proposed nothing stops the
    // run instead of restarting into the same nothing.
    let mut restarts = 0u64;
    let mut pass_candidates = 0usize;
    let mut screened_batches = 0u64;
    let mut replay_done = false;
    let mut replay_skipped = HashSet::new();
    // A replay accept ends the search, not the run's coverage duty (Issue #91).
    // While this is set the run keeps screening on the budget the accept left
    // behind — no more replay, no more full scoring, no more accepts — and
    // `stop_reason` already names the accept that ended the search. A search
    // accept never sets this: since Issue #96 it restarts the sweep and the
    // search runs on to the budget.
    let mut coverage_tail = false;
    let mut tail_batches = 0u64;
    let mut tail_screened = 0usize;
    // What ended the tail, journalled separately because `stop_reason` keeps
    // naming the accept that ended the search (Issue #91). Empty until a
    // benign end is reached; a fault fills it from `stop_reason` instead.
    let mut tail_end = String::new();
    // The last accept's check-in tag, so a tail can re-stamp it with the
    // coverage the run finished on rather than the coverage at the cut (#91).
    let mut last_accept: Option<StampedAccept> = None;
    // In-run state, deliberately not seeded from the cache: cross-run memory is
    // the learnings store's job (Issues #56, #57).
    let mut pool: Vec<BundleMember> = Vec::new();
    // Neighbourhood memberships this run has already screened (Issue #108).
    // Generation is deterministic, so without this the same best-ranked groups
    // would be re-proposed and re-screened every batch until the deadline. It
    // is cleared on every accept: the incumbent those verdicts were measured
    // against no longer exists.
    let mut tried_groups: HashSet<String> = HashSet::new();
    // Resolved once: the ladder is the same every batch, and an unresolvable
    // one must stop the run rather than quietly screen at some other rate.
    let ladder = config.screen_ladder()?;
    // The promotion stage is the screen's dominant rate, and the ladder's cost
    // is converted to it, so the cost model has one rate to reason in. The
    // multiple is what a whole batch costs relative to one promotion-stage
    // pass, which is what sizes the pre-measurement screening reserve (#104).
    let mut cost = CostModel::new(
        ladder.as_ref().map(|l| l.promotion_rate()),
        ladder.as_ref().map_or(1.0, batch_cost_multiple),
    );
    // The opening baseline is a real full-corpus score of one creature, so the
    // screening reserve has a measurement to size itself from before the first
    // cohort has run (Issue #77).
    cost.observe_baseline(baseline_ms);
    let mut tally = WinnerTally::default();

    while incumbent.hidden_neurons() > 0 {
        if cancel.is_cancelled() {
            stop_reason = "cancelled".into();
            break;
        }
        // A coverage tail keeps the reason the replay accept set: running out
        // of budget or of experiments is how a tail is meant to end, and the
        // fact worth reporting as the run's stop is still the accept that
        // ended the search. What ended the tail is not lost — it is journalled
        // as its own `coverageTail` record. A fault — cancellation, a broken
        // scorer — overrides the stop reason as well.
        if Instant::now() >= deadline {
            if coverage_tail {
                tail_end = "timeout".into();
            } else {
                stop_reason = "timeout".into();
            }
            break;
        }
        if let Some(max) = config.max_experiments
            && experiments >= max
        {
            if coverage_tail {
                tail_end = "max-experiments".into();
            } else {
                stop_reason = "max-experiments".into();
            }
            break;
        }

        // One screening batch is reserved from the wall clock (Issue #77).
        // Replay spends the budget on the creature; the reserve spends it on
        // coverage, and once the budget is down to a single batch and this run
        // has screened nothing, coverage wins. See `reserve_stands` for why the
        // reserve is one batch and not a share of the budget.
        let reserving = reserve_stands(
            &cost,
            config,
            deadline.saturating_duration_since(Instant::now()),
            screened_batches,
        );
        // Silent in a coverage tail: the replay stage is already standing down
        // there, so the reserve has nothing left to stand down (#91).
        if reserving && !replay_done && !coverage_tail {
            log::info(&format!(
                "budget down to its last screening batch; standing the replay stage down so \
                 this run advances coverage (#77): {} newly screened so far",
                progress.count()
            ));
        }
        if !replay_done && !reserving && !coverage_tail {
            // A confirmed win on a tagged uuid replays like any other (#63):
            // a tag describes where a neuron came from, it does not earn it a
            // place.
            // Accepted cuts and confirmed-but-unapplied ones, best measured
            // delta first (Issue #57): the largest-first plans below then drop
            // the weakest members rather than the most recently filed.
            let mut ranked = ranked_confirmed(&known, &incumbent.creature, config.min_improvement);
            // Then the previous epochs' winners, as hypotheses (Issue #101): a
            // cut an older corpus confirmed is the best guess the fleet has
            // about the new one. They rank behind this corpus's own evidence
            // because that evidence was measured against the data in hand, and
            // every one of them is re-scored below before anything is accepted.
            let historical: HashSet<String> = if prior.enabled {
                let hypotheses = historical_replay(
                    prior.records,
                    &known,
                    &incumbent.creature,
                    config.min_improvement,
                );
                let uuids = hypotheses.iter().map(|c| c.uuid.clone()).collect();
                ranked.extend(hypotheses);
                uuids
            } else {
                HashSet::new()
            };
            let replayable: Vec<crate::learnings::ConfirmedWin> = ranked
                .into_iter()
                .filter(|c| !replay_skipped.contains(&c.uuid))
                .take(replay_cap(replay_cfg.max))
                .collect();
            let this_corpus: Vec<&crate::learnings::ConfirmedWin> = replayable
                .iter()
                .filter(|c| !historical.contains(&c.uuid))
                .collect();
            let accepted_only = this_corpus.iter().filter(|c| c.accepted).count();
            let confirmed_only = this_corpus.len() - accepted_only;
            let from_history = replayable.len() - this_corpus.len();
            let wins: Vec<String> = replayable.into_iter().map(|c| c.uuid).collect();
            // An accepted neighbourhood is replayed as the group it was
            // (Issue #108). Its members carry no individual verdict — the
            // scorer never judged them apart — so replaying them one at a time
            // asks the very question the group was proposed to get past. Only a
            // run that offers group cuts replays them: a control run must stay
            // a control run whatever the shared cache holds.
            let group_plans: Vec<Vec<String>> = if config.group_cuts {
                crate::learnings::confirmed_groups(
                    &known,
                    &incumbent.creature,
                    config.min_improvement,
                )
            } else {
                Vec::new()
            };
            if wins.is_empty() && group_plans.is_empty() {
                replay_done = true;
                continue;
            }
            let (applied, _) = apply_available(&incumbent.creature, &activation, &wins);
            for u in &wins {
                if !applied.iter().any(|a| a == u) {
                    replay_skipped.insert(u.clone());
                }
            }
            if applied.is_empty() && group_plans.is_empty() {
                replay_done = true;
                continue;
            }
            log::info(&format!(
                "replay: combining {} of {} known win(s) still on incumbent ({accepted_only} applied elsewhere, {confirmed_only} confirmed only, {from_history} from older corpus epochs — re-scored here)",
                applied.len(),
                wins.len()
            ));
            let plans = replay_plans(&applied);
            if plans.len() > 1 {
                log::detail(&format!(
                    "replay: combined plan plus {} shrink step(s), smallest {} cut(s)",
                    plans.len() - 1,
                    plans.last().map_or(0, Vec::len)
                ));
            }
            let sampled: Vec<SampledWinner> = if plans.is_empty() && !applied.is_empty() {
                match propose(&incumbent.creature, &activation, &applied[0]) {
                    Ok((kind, creature)) => vec![SampledWinner {
                        candidate: SweepCandidate {
                            members: vec![applied[0].clone()],
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
            // Replay is not sized by `--max-full`: after Issue #54 that flag
            // caps individual scoring in the search loop, and using it for a
            // replay probe was always a conflation. Nor is the replay cohort
            // trimmed to the wall clock (Issue #58): it runs first, with the
            // whole budget in front of it, and `MAX_REPLAY_PLANS` plus
            // `REPLAY_PROBE_LIMIT` already bound it.
            let probe_n = REPLAY_PROBE_LIMIT.min(applied.len());
            experiments += 1;
            let mut extra_plans = plans;
            if !group_plans.is_empty() {
                log::info(&format!(
                    "replay: {} accepted neighbourhood(s) still whole on the incumbent",
                    group_plans.len()
                ));
            }
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
                    group_plans: &group_plans,
                    max_individuals: None,
                    pool: &[],
                    max_entries: None,
                },
            ) {
                Ok(full) => {
                    consecutive_fail = 0;
                    cost.observe_full(full.full_ms, full.entries());
                    tally.observe(&full, config.min_improvement);
                    journal_full(&journal_path, &full, started)?;
                    if full.winner.is_none() && sampled.is_empty() && applied.len() > 1 {
                        log::info(&format!(
                            "replay: every plan missed; probing {probe_n} known win(s) individually"
                        ));
                        let mut probe = Vec::new();
                        for uuid in applied.iter().take(probe_n) {
                            match propose(&incumbent.creature, &activation, uuid) {
                                Ok((kind, creature)) => probe.push(SampledWinner {
                                    candidate: SweepCandidate {
                                        members: vec![uuid.clone()],
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
                                group_plans: &[],
                                max_individuals: None,
                                pool: &[],
                                max_entries: None,
                            },
                        ) {
                            Ok(probe_full) => {
                                consecutive_fail = 0;
                                cost.observe_full(probe_full.full_ms, probe_full.entries());
                                tally.observe(&probe_full, config.min_improvement);
                                journal_full(&journal_path, &probe_full, started)?;
                                // The probes are the honest per-uuid measurement
                                // of a replayed win: one that has stopped paying
                                // files a negative delta here and stops being
                                // replayed (Issue #57).
                                file_full_outcome(store, &mut known, &probe, &probe_full);
                                if let Some(win) = probe_full.winner {
                                    last_accept = Some(apply_local_win(
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
                                        store.map(|_| screens.as_slice()),
                                        opening_hidden,
                                    )?);
                                    stop_reason = "replay-accepts".into();
                                    if !open_coverage_tail(
                                        &incumbent,
                                        &activation,
                                        seed.wrapping_add(accepts).wrapping_add(restarts),
                                        ordering,
                                        unchecked_first,
                                        &screens,
                                        &prior,
                                        deadline,
                                        store.is_some(),
                                        ladder.is_some(),
                                        &stop_reason,
                                        &mut sweep,
                                        &mut pool,
                                        &mut pass_candidates,
                                    ) {
                                        break;
                                    }
                                    coverage_tail = true;
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
                        last_accept = Some(apply_local_win(
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
                            store.map(|_| screens.as_slice()),
                            opening_hidden,
                        )?);
                        stop_reason = "replay-accepts".into();
                        if !open_coverage_tail(
                            &incumbent,
                            &activation,
                            seed.wrapping_add(accepts).wrapping_add(restarts),
                            ordering,
                            unchecked_first,
                            &screens,
                            &prior,
                            deadline,
                            store.is_some(),
                            ladder.is_some(),
                            &stop_reason,
                            &mut sweep,
                            &mut pool,
                            &mut pass_candidates,
                        ) {
                            break;
                        }
                        coverage_tail = true;
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

        // An exhausted sweep is rebuilt, never idled on (Issue #77). Before
        // this it ended the run with the stop reason `exhausted`: the budget
        // left went unused, and a creature the fleet had worked all the way
        // through stopped being screened rather than recycling its stalest
        // neurons.
        //
        // `prefer_unchecked` already orders a fresh permutation unchecked
        // first, then stalest-screened first, so a creature that is 100%
        // screened rolls straight into re-screening its stalest neurons.
        if sweep.exhausted() {
            if pass_candidates == 0 {
                // A whole permutation that proposed nothing would restart into
                // exactly the same nothing. Stop with a reason instead.
                log::warn(&format!(
                    "sweep: a whole pass over {} hidden neuron(s) produced no candidate — every \
                     visit was skipped as a known failure or could not propose; stopping",
                    incumbent.hidden_neurons()
                ));
                if coverage_tail {
                    tail_end = "no-candidates".into();
                } else {
                    stop_reason = "no-candidates".into();
                }
                break;
            }
            restarts += 1;
            pass_candidates = 0;
            sweep = fresh_sweep(
                &incumbent.creature,
                &activation,
                seed.wrapping_add(accepts).wrapping_add(restarts),
                ordering,
                unchecked_first,
                &screens,
                &prior,
            );
            log::info(&format!(
                "sweep exhausted over {} hidden neuron(s); restart {restarts}, {} newly screened so far",
                incumbent.hidden_neurons(),
                progress.count()
            ));
            journal::append(
                &journal_path,
                &Event::SweepRestart {
                    restarts,
                    hidden: incumbent.hidden_neurons(),
                    newly_screened: progress.count(),
                },
            )?;
            continue;
        }

        let avoid = known_failures(
            &known,
            &incumbent.creature,
            replay_cfg,
            crate::incumbent::now_unix(),
            config.min_improvement,
        );
        if !avoid.is_empty() {
            log::detail(&format!(
                "learnings: skipping {} known failure(s)",
                avoid.len()
            ));
        }
        // Tagged neurons are candidates like any other (#63); `meta.neuron_tags`
        // is still read below for the informational coverage count.
        let (mut candidates, skips) =
            sweep.fill_batch_avoiding(&incumbent.creature, &activation, config.candidates, &avoid);
        pass_candidates += candidates.len();
        // Structural neighbourhood proposals ride the same batch (Issue #108):
        // a chain or a low-fan-out branch that no single-neuron cut can expose,
        // screened and scored exactly like every other candidate. They are
        // *extra* candidates, not sweep visits — the permutation and the
        // coverage it drives are untouched, because a group screen says nothing
        // about whether its members are removable one at a time.
        let mut group_candidates = 0usize;
        if config.group_cuts {
            let groups = crate::neighbourhood::group_batch(
                &incumbent.creature,
                &activation,
                config.neighbourhood_config(),
                &tried_groups,
            );
            if !groups.blocked.is_empty() {
                log::detail(&format!(
                    "groups: {} of {} proposal(s) refused: {}",
                    groups.blocked.len(),
                    groups.considered(),
                    groups.blocked.join("; ")
                ));
            }
            group_candidates = groups.candidates.len();
            for candidate in groups.candidates {
                tried_groups.insert(crate::neighbourhood::group_key(&candidate.members));
                log::detail(&format!(
                    "group: {} ({} neurons)",
                    candidate.members.join(" + "),
                    candidate.members.len()
                ));
                candidates.push(candidate);
            }
        }
        let candidates = candidates;
        let remaining_s = deadline.saturating_duration_since(Instant::now()).as_secs();
        log::info(&format!(
            "batch {batch_idx}: {} candidates ({group_candidates} group), {} skipped, \
             {} hidden left, {remaining_s}s remaining",
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
        // Every visit is coverage, candidate or not (Issue #93). A neuron the
        // razor cannot propose anything for — it feeds an aggregate squash, or
        // carries a typed synapse — is most of a forest-heavy creature, and
        // while those visits filed nothing the numerator was pinned to the
        // prunable minority and fell by one on every accepted cut.
        //
        // A uuid this run has already filed a record for is not filed again:
        // nothing was scored, so the second record would carry no new fact —
        // just another line in a log every host reads end to end on every run.
        // A record another host files concurrently is deduplicated by uuid when
        // coverage is counted, so the figures stand either way.
        if !skips.is_empty() {
            log::detail(&format!("skips: {}", skip_reason_tally(&skips)));
        }
        let visits: Vec<ScreenTry<'_>> = skips
            .iter()
            .filter(|s| !progress.seen(&s.uuid))
            .map(skip_try)
            .collect();
        if candidates.is_empty() {
            if !visits.is_empty() {
                file_batch_screens(
                    store,
                    &mut screens,
                    &mut progress,
                    &visits,
                    &journal_path,
                    batch_idx,
                )?;
            }
            batch_idx += 1;
            experiments += 1;
            continue;
        }

        let batch_size = candidates.len();
        let sampled = match &ladder {
            Some(ladder) => {
                match screen_progressive(
                    scorer,
                    &config.training_data,
                    &incumbent.creature,
                    candidates,
                    ProgressiveConfig {
                        ladder,
                        threshold: config.screen_threshold,
                        batch: batch_idx,
                        remaining_after: sweep.remaining(),
                        workspace,
                    },
                ) {
                    Ok(screen) => {
                        consecutive_fail = 0;
                        // The incumbent is scored alongside every stage, and a
                        // stage below the promotion rate is a fraction of a
                        // promotion-rate creature-score (#104).
                        cost.observe_screen(
                            screen.screen_ms,
                            screen.promotion_rate_creatures(ladder),
                        );
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
                        // The control is fully described by `screen` above; a
                        // ladder is not, so each rung says what it cost and
                        // what it ended (#104).
                        if ladder.is_progressive() {
                            log::detail(&screen_ladder_line(&screen, batch_size));
                            for stage in &screen.stages {
                                journal::append(
                                    &journal_path,
                                    &Event::ScreenStage {
                                        batch: batch_idx,
                                        stage: stage.stage,
                                        rate: stage.rate,
                                        phase: stage.phase,
                                        entered: stage.entered,
                                        rejected: stage.rejected,
                                        promoted: stage.promoted,
                                        records_scored: stage.records_scored,
                                        mean_delta: stage.mean_delta,
                                        ms: stage.ms,
                                        outcome: stage.outcome.to_string(),
                                    },
                                )?;
                            }
                        }
                        let mut coverage = visits.clone();
                        // A sampled winner is a lead, and only full scoring
                        // settles it. In a coverage tail nothing will score it,
                        // so filing it as checked would bury it: the record is
                        // the freshest in the store, `oldest_screened_first`
                        // sorts it last, and unchecked-first would defer it
                        // behind every never-screened neuron on the creature.
                        // It stays unchecked, so the next run screens *and*
                        // scores it (Issue #91).
                        // A group candidate files no screen record for the
                        // neuron it was keyed on (Issue #108): what was screened
                        // is the neighbourhood, and marking a member checked
                        // would claim coverage of a single cut nothing tried.
                        if !coverage_tail {
                            for w in screen.winners.iter().filter(|w| !w.candidate.is_group()) {
                                coverage.push(ScreenTry::scored(
                                    w.candidate.uuid.as_str(),
                                    w.candidate.kind,
                                    ScreenOutcomeKind::Winner,
                                ));
                            }
                        }
                        for l in screen
                            .losers
                            .iter()
                            .filter(|l| l.kind != crate::sweep::CandidateKind::Group)
                        {
                            coverage.push(ScreenTry::scored(
                                l.uuid.as_str(),
                                l.kind,
                                ScreenOutcomeKind::Loser,
                            ));
                        }
                        file_batch_screens(
                            store,
                            &mut screens,
                            &mut progress,
                            &coverage,
                            &journal_path,
                            batch_idx,
                        )?;
                        if let Some(log) = &candidate_log {
                            log.screened_out(
                                &incumbent.creature,
                                &activation,
                                &incumbent.checksum,
                                &screen.losers,
                                screen.screen_ms,
                                // Winners, losers and the incumbent: every
                                // creature the one screen call scored.
                                screen.winners.len() + screen.losers.len() + 1,
                            );
                        }
                        screen.winners
                    }
                    Err(e) => {
                        consecutive_fail += 1;
                        log::warn(&format!("screen failed: {e}"));
                        // The scorer lost the candidates, not the visits the
                        // sweep already made past them (#93) — filed before the
                        // failure limit is consulted, so the last batch's
                        // coverage is not thrown away with the run. A batch with
                        // nothing to file journals nothing: an empty screened
                        // record would claim coverage work that never happened.
                        if !visits.is_empty() {
                            file_batch_screens(
                                store,
                                &mut screens,
                                &mut progress,
                                &visits,
                                &journal_path,
                                batch_idx,
                            )?;
                        }
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
                let mut coverage = visits.clone();
                coverage.extend(candidates.iter().filter(|c| !c.is_group()).map(|c| {
                    ScreenTry::scored(c.uuid.as_str(), c.kind, ScreenOutcomeKind::Winner)
                }));
                file_batch_screens(
                    store,
                    &mut screens,
                    &mut progress,
                    &coverage,
                    &journal_path,
                    batch_idx,
                )?;
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
        screened_batches += 1;
        if coverage_tail {
            // The losers and the visits the razor could propose nothing for are
            // filed: those are finished business. The winners are not — nothing
            // in this run will score them — so they are left unchecked for the
            // next run to screen and full-score.
            tail_batches += 1;
            tail_screened += batch_size;
            if !sampled.is_empty() {
                log::detail(&format!(
                    "coverage: {} sampled winner(s) left unchecked for the next run to score",
                    sampled.len()
                ));
            }
            continue;
        }
        if sampled.is_empty() {
            log::detail("no sampled winners; continuing sweep");
            continue;
        }
        tally.screened += sampled.len();

        // Every screened winner is full-scored: `--max-full` caps how many are
        // scored *individually* and never which combinations are tried
        // (Issue #54). Confirmed winners from earlier batches join for bundle
        // membership only — they already have a verdict (Issue #56).
        pool = standing_pool(&pool, &incumbent.creature, &activation);
        log::detail(&full_scoring_line(
            config.max_full,
            sampled.len(),
            pool.len(),
        ));

        let remaining = deadline.saturating_duration_since(Instant::now());
        let max_entries = match cost.cohort_budget(remaining) {
            CohortBudget::Unmeasured => None,
            CohortBudget::Entries(n) => Some(n),
            CohortBudget::TooSmall => {
                // Starting a cohort that cannot finish overruns the deadline,
                // and GRQ runs Ockham on a schedule. The winners are already
                // in the screen cache, so the next run picks them up.
                log::warn(&format!(
                    "full: {}s left, est {:.0}ms/creature — too small for a cohort; stopping",
                    remaining.as_secs(),
                    cost.per_creature_ms().unwrap_or(0.0)
                ));
                stop_reason = "budget".into();
                break;
            }
        };
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
                extra_plans: &[],
                group_plans: &[],
                max_individuals: config.max_full,
                pool: &pool,
                max_entries,
            },
        ) {
            Ok(full) => {
                consecutive_fail = 0;
                cost.observe_full(full.full_ms, full.entries());
                log::detail(&format!(
                    "full: {} individuals, {} bundles, {} skipped, {}ms, accepted={}",
                    full.individuals.len(),
                    full.bundles.len(),
                    full.skipped_bundles,
                    full.full_ms,
                    full.winner.is_some()
                ));
                if full.dropped() > 0 {
                    log::detail(&budget_trim_line(
                        remaining,
                        cost.per_creature_ms().unwrap_or(0.0),
                        &full,
                    ));
                }
                if full.capped_plans > 0 {
                    log::detail(&format!(
                        "bundles: {} plan(s) beyond the cap of {} were not built",
                        full.capped_plans,
                        crate::promote::MAX_BUNDLE_PLANS
                    ));
                }
                journal_full(&journal_path, &full, started)?;
                journal::append(
                    &journal_path,
                    &Event::Budget {
                        est_ms_per_creature: cost.per_creature_ms().unwrap_or(0.0),
                        remaining_secs: remaining.as_secs(),
                        entries: full.entries(),
                        dropped_individuals: full.dropped_individuals,
                        dropped_bundles: full.dropped_bundles,
                    },
                )?;
                tally.observe(&full, config.min_improvement);
                if let Some(log) = &candidate_log {
                    log.judged(
                        &incumbent.creature,
                        &activation,
                        &incumbent.checksum,
                        &sampled,
                        &full,
                    );
                }
                file_full_outcome(store, &mut known, &sampled, &full);
                update_pool(&mut pool, &sampled, &full, config.min_improvement);
                if let Some(win) = full.winner {
                    last_accept = Some(apply_local_win(
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
                        store.map(|_| screens.as_slice()),
                        opening_hidden,
                    )?);
                    restart_after_accept(
                        &incumbent,
                        &activation,
                        seed.wrapping_add(accepts).wrapping_add(restarts),
                        ordering,
                        unchecked_first,
                        &screens,
                        &prior,
                        &mut sweep,
                        &mut pool,
                        &mut pass_candidates,
                    );
                    // The neighbourhoods this run screened were judged against
                    // an incumbent that no longer exists, so they are offered
                    // again on the new one (Issue #108).
                    tried_groups.clear();
                    log::detail(&format!(
                        "restarted sweep after accept; {} hidden remaining; {} confirmed winner(s) still standing",
                        incumbent.hidden_neurons(),
                        pool.len()
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

    if coverage_tail {
        // A fault ends the tail *and* the run, so it names both; anything else
        // filled `tail_end` on its way out.
        let ended = if tail_end.is_empty() {
            stop_reason.clone()
        } else {
            tail_end
        };
        log::info(&format!(
            "coverage tail: {tail_batches} batch(es), {tail_screened} candidate(s) screened after \
             the accept; {} newly checked this run; ended {ended} (#91)",
            progress.count()
        ));
        journal::append(
            &journal_path,
            &Event::CoverageTail {
                batches: tail_batches,
                screened: tail_screened,
                newly_checked: progress.count(),
                ended,
            },
        )?;
    }

    // What the run tried, kept and rejected, for the commit description and
    // `--report` (Issue #59). Journalled whenever anything was screened, so a
    // run with no learnings dir still reports its economics.
    let winners = tally.finish(pool.len(), cost.per_creature_ms());
    if winners.has_any() {
        log::info(&winners.summary());
        journal::append(
            &journal_path,
            &Event::Winners {
                screened: winners.screened,
                confirmed: winners.confirmed,
                applied: winners.applied,
                carried: winners.carried,
                plans: winners.plans,
                skipped: winners.skipped,
                best_cuts: winners.best_cuts,
                best_delta: winners.best_delta,
                dropped: winners.dropped,
                est_ms_per_creature: winners.est_ms_per_creature,
            },
        )?;
    }

    // How far the run itself moved the fleet (Issue #77). Computed with or
    // without a store: coverage state is only *reportable* with one, but a run
    // that advanced nothing must say so either way.
    let tagged: HashSet<String> = meta.neuron_tags.keys().cloned().collect();
    let cov = crate::coverage::coverage(
        &incumbent.creature,
        &tagged,
        &screens,
        opening_hidden.saturating_sub(incumbent.hidden_neurons()),
    );
    let newly_screened = progress.count();
    if let Some(warning) = crate::coverage::zero_progress_warning(newly_screened, cov.unchecked()) {
        log::warn(&warning);
    }

    // The accept published `best.json` before the tail screened anything, so
    // its `sweep X/Y` is the figure at the cut rather than the one the run
    // finished on. Re-stamp it, or the check-in subject would still report the
    // stalled coverage this issue is about (#91). Only the tag changes: the
    // creature published is the one the accept produced.
    if coverage_tail && let Some(stamp) = &last_accept {
        meta.stamp_acceptance(&OckhamProgress {
            accepts: stamp.accepts,
            experiments: stamp.experiments,
            opening: opening_score,
            score: stamp.score,
            error: stamp.error,
            last: stamp.last,
            origin: stamp.origin,
            cuts: stamp.cuts,
            coverage: store.map(|_| cov),
            epoch: store.map(|_| corpus.identity.as_str()),
        });
        publish_best(config, &meta, &incumbent.creature, &stamp.checksum)?;
        log::detail("coverage: re-stamped the check-in tag with the run's final coverage");
    }

    // Coverage is only meaningful with the screen store behind it; without one
    // there is no coverage state to report, so nothing is journalled.
    if store.is_some() {
        // The epoch travels with the figure, so a log read months later can
        // tell a fresh epoch from a collapse in coverage (Issue #102).
        log::info(&format!(
            "{} · epoch corpus {}",
            cov.summary(),
            crate::coverage::short_epoch(&corpus.identity)
        ));
        journal::append(
            &journal_path,
            &Event::Coverage {
                hidden: cov.hidden,
                tagged: cov.tagged,
                checkable: cov.checkable,
                checked: cov.checked,
                blocked: cov.blocked,
                blocked_by_reason: cov.blocked_by_reason,
                cut: cov.cut,
                corpus_identity: Some(corpus.identity.clone()),
            },
        )?;
        // The GRQ-facing commit-description artefacts (Issues #40, #59). A
        // write fault warns rather than failing the run, matching the learnings
        // cache: coverage is reporting, and reporting must never lose pruning.
        let report = crate::coverage::CoverageReport {
            coverage: cov,
            newly_screened,
            winners: winners.has_any().then_some(winners),
            corpus_identity: Some(corpus.identity.clone()),
            history: Some({
                // The records this run filed are history too, and its own
                // epoch must appear in its own history line (Issue #102).
                screen_history.merge(&screens);
                screen_history.over(&incumbent.creature)
            })
            .filter(crate::coverage::History::has_any),
        };
        match crate::coverage::write_files(&config.output_dir, &report, config.candidates) {
            Ok(()) => log::detail(&format!(
                "coverage: wrote {} and {}",
                crate::coverage::COVERAGE_TEXT_FILE,
                crate::coverage::COVERAGE_JSON_FILE
            )),
            Err(e) => log::warn(&format!("coverage files not written: {e}")),
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
            final_hidden: incumbent.hidden_neurons(),
            final_synapses: incumbent.creature.synapses.len(),
            elapsed_ms: started.elapsed().as_millis() as u64,
            newly_screened,
        },
    )?;
    log::info(&format!(
        "stop reason={stop_reason}  accepts={accepts}  experiments={experiments}  \
         newlyScreened={newly_screened}  restarts={restarts}  Δ={:.3e}",
        current_score - opening_score
    ));

    Ok(LoopOut {
        activation,
        seed,
        accepts,
        experiments,
        stop_reason,
        newly_screened,
        cumulative_delta: current_score - opening_score,
    })
}

/// Turn the run over to screening after a replay accept, or refuse when nothing
/// is owed.
///
/// A replayed known win used to end the run then and there, so the prune could
/// check in. It also ended the run's *other* job: nine consecutive GRQ-sampler
/// check-ins reported `progress: 0 newly screened this run` while the razor
/// kept cutting, because every one of those runs accepted before it screened
/// anything (Issue #91).
///
/// The replay accept still ends the search — `best.json` is already written,
/// and nothing after this replays, full-scores or accepts — but the budget it
/// leaves behind goes to coverage, one screening batch after another, so a run
/// that accepts early still checks in with the ~100-per-batch the fleet
/// expects. The sweep is rebuilt over the creature the accept just changed,
/// which re-applies unchecked-first selection (#38) against the records filed
/// so far. A *search* accept never comes here: since Issue #96 removed the
/// accept cap it restarts the sweep and keeps searching, so a run stops only
/// on its budget.
///
/// Returns `false` — stop now, as before — when there is nothing left to
/// screen: no hidden neurons, no budget, no screen store to file the coverage
/// in, or no sampled screen to check them with. With `--screen-sample-rate 0`
/// the only check available is a full-corpus cohort, and that is precisely the
/// search this accept ended; with no store there is no coverage to advance,
/// because the records would not outlive the run.
#[allow(clippy::too_many_arguments)]
fn open_coverage_tail(
    incumbent: &Incumbent,
    activation: &ActivationStats,
    seed: u64,
    ordering: crate::ordering::OrderingConfig<'_>,
    unchecked_first: bool,
    screens: &[Screened],
    prior: &PriorHint<'_>,
    deadline: Instant,
    has_store: bool,
    has_screen: bool,
    reason: &str,
    sweep: &mut Sweep,
    pool: &mut Vec<BundleMember>,
    pass_candidates: &mut usize,
) -> bool {
    let remaining = deadline.saturating_duration_since(Instant::now());
    if incumbent.hidden_neurons() == 0 || remaining.is_zero() || !has_store || !has_screen {
        return false;
    }
    restart_after_accept(
        incumbent,
        activation,
        seed,
        ordering,
        unchecked_first,
        screens,
        prior,
        sweep,
        pool,
        pass_candidates,
    );
    log::info(&format!(
        "{reason}: the search is over; spending the remaining {}s screening for coverage (#91)",
        remaining.as_secs()
    ));
    true
}

/// Rebuild the sweep and the carried pool against the creature an accept moved.
///
/// The one place a post-accept restart happens, so the search restart and the
/// coverage tail cannot drift apart: both must re-apply unchecked-first
/// selection to the *new* creature, and both must drop pool members the accept
/// removed — those were measured against an incumbent that has moved, so they
/// are candidates to re-try, never facts to re-apply.
#[allow(clippy::too_many_arguments)]
fn restart_after_accept(
    incumbent: &Incumbent,
    activation: &ActivationStats,
    seed: u64,
    ordering: crate::ordering::OrderingConfig<'_>,
    unchecked_first: bool,
    screens: &[Screened],
    prior: &PriorHint<'_>,
    sweep: &mut Sweep,
    pool: &mut Vec<BundleMember>,
    pass_candidates: &mut usize,
) {
    *sweep = fresh_sweep(
        &incumbent.creature,
        activation,
        seed,
        ordering,
        unchecked_first,
        screens,
        prior,
    );
    *pass_candidates = 0;
    *pool = standing_pool(pool, &incumbent.creature, activation);
}

/// Build a sweep over the current incumbent, unchecked-first when enabled.
///
/// The one place a sweep is created (Issue #77): the opening sweep, the
/// post-accept restart and the exhausted-sweep restart must all order the
/// permutation the same way, or a restart would silently drop the
/// coverage-driven selection the run started with.
fn fresh_sweep(
    creature: &CreatureExport,
    activation: &ActivationStats,
    seed: u64,
    ordering: crate::ordering::OrderingConfig<'_>,
    unchecked_first: bool,
    screens: &[Screened],
    prior: &PriorHint<'_>,
) -> Sweep {
    let mut sweep = Sweep::with_ordering(creature, activation, seed, ordering);
    if unchecked_first {
        prefer_unchecked(&mut sweep, screens, creature);
    }
    prefer_prior_corpus(&mut sweep, prior, screens, creature);
    sweep
}

/// Old-corpus verdicts, and what turns them into a priority queue (Issue #88).
///
/// Held by reference and rebuilt into a uuid list on every sweep, because the
/// list depends on the creature: an accept removes neurons, and a hint for a
/// uuid that is no longer there is not a hint.
struct PriorHint<'a> {
    /// Whether `--old-corpus-first` is in force for this run.
    ///
    /// Separate from an empty `records`, so a run with the priority **on** and
    /// nothing to prioritise still says `0` rather than saying nothing at all.
    enabled: bool,
    /// Verdicts loaded from sibling `corpus-*` directories, never from this one,
    /// each stamped with the epoch that established it (#101).
    records: &'a [HistoricalLearning],
    /// This run's corpus: a uuid already screened under it needs no priority.
    corpus_identity: &'a str,
    /// Measured delta above which an unapplied old win still counts as one.
    min_improvement: f64,
}

impl PriorHint<'_> {
    /// The hint switched off — the flag disabled, or no cache to read.
    fn none() -> Self {
        Self {
            enabled: false,
            records: &[],
            corpus_identity: "",
            min_improvement: 0.0,
        }
    }
}

/// Move old-corpus wins to the front of the sweep's unvisited tail (Issue #88).
///
/// A hidden neuron the fleet removed under earlier training data, still on the
/// incumbent and unchecked under this corpus, is the likeliest thing on the
/// creature to be removable again — so it is looked at before neurons with no
/// history at all. Selection only: [`Sweep::prefer`] reorders the same UUIDs,
/// every one of them still faces the sample screen and full-corpus scoring, and
/// nothing here changes how coverage is counted.
///
/// The count logged and recorded is what `prefer` actually **moved**, not what
/// the hint asked for: a uuid the sweep no longer holds was not prioritised.
fn prefer_prior_corpus(
    sweep: &mut Sweep,
    prior: &PriorHint<'_>,
    screens: &[Screened],
    creature: &CreatureExport,
) {
    if !prior.enabled {
        return;
    }
    let priority = prior_corpus_priority(
        prior.records,
        screens,
        creature,
        prior.corpus_identity,
        prior.min_improvement,
    );
    sweep.old_corpus_first = sweep.prefer(&priority);
    log::info(&format!(
        "coverage: {} neuron(s) prioritised from older corpus caches (#88)",
        sweep.old_corpus_first
    ));
}

/// The screen record one sweep skip files (Issues #93, #103).
///
/// A standing full-corpus verdict is the strongest check there is — the cut was
/// proposed, scored and judged — so it is filed as a known failure and carries
/// no blocked reason. Everything else is filed as a blocked visit.
///
/// The known-failure claim is made only for the skip that actually carries
/// [`crate::sweep::KNOWN_FAILURE_REASON`], never as a default: `SweepSkip` has
/// public fields, and a skip that named no reason code would otherwise be
/// upgraded to "the fleet scored and judged this cut" — a claim nothing made.
/// A skip with neither is blocked for an explicit-but-unknown reason, which is
/// what [`crate::blocked::BlockedReason::Other`] is for.
fn skip_try(skip: &crate::sweep::SweepSkip) -> ScreenTry<'_> {
    if skip.reason == crate::sweep::KNOWN_FAILURE_REASON && skip.blocked.is_none() {
        return ScreenTry::visited(&skip.uuid, crate::learnings::SCREEN_KIND_KNOWN_FAILURE);
    }
    ScreenTry::blocked(
        &skip.uuid,
        skip.blocked.unwrap_or(crate::blocked::BlockedReason::Other),
    )
}

/// `aggregate-squash: 41, known-failure: 3` — one batch's skips, by reason.
///
/// The kind filed against a skipped visit is only two buckets wide, so the
/// reason itself would otherwise be discarded: an unexpected skip — a
/// non-finite mean, a candidate that failed `creature.validate()` — would be
/// indistinguishable in the audit trail from the aggregate structure that
/// accounts for most of them, and a neuron the razor could prune on a later
/// pass would be silently filed alongside one it never can (Issue #93).
///
/// Counted by [`crate::blocked::BlockedReason`] code since #103, so the line
/// here, the record on disk and the coverage breakdown are the same categories
/// — the old word-prefix classifier could not be reconciled with either.
/// Commonest first, so the head of the line answers "why is this creature not
/// being pruned?".
fn skip_reason_tally(skips: &[crate::sweep::SweepSkip]) -> String {
    let mut counts: Vec<(&str, usize)> = Vec::new();
    for skip in skips {
        let class = skip
            .blocked
            .map(crate::blocked::BlockedReason::code)
            .unwrap_or(crate::sweep::KNOWN_FAILURE_REASON);
        match counts.iter_mut().find(|(seen, _)| *seen == class) {
            Some((_, n)) => *n += 1,
            None => counts.push((class, 1)),
        }
    }
    counts.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(b.0)));
    counts
        .iter()
        .map(|(class, n)| format!("{class}: {n}"))
        .collect::<Vec<_>>()
        .join(", ")
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
            groups: full.groups.len(),
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
///
/// `progress` sees only the records that were actually filed (Issue #77): a
/// record a store fault dropped is not coverage this run added.
fn file_batch_screens(
    store: Option<&LearningsStore>,
    screens: &mut Vec<Screened>,
    progress: &mut crate::coverage::ScreenProgress,
    coverage: &[ScreenTry<'_>],
    journal_path: &std::path::Path,
    batch: u64,
) -> Result<(), String> {
    let before = screens.len();
    let n = file_screens(store, coverage, screens);
    for filed in &screens[before..] {
        progress.observe(&filed.uuid);
    }
    if n > 0 {
        log::detail(&format!("screens: filed {n} screen record(s)"));
    }
    journal::append(journal_path, &Event::Screened { batch, screened: n })
}

/// File one verdict per individually scored uuid, plus the winner's cuts.
///
/// `outcome` still says which candidate won the cohort, but every individual
/// now carries the delta the scorer actually returned for it alone
/// (Issue #52). A cut that comfortably beat `min_improvement` and merely lost
/// its cohort is therefore recorded as the confirmed win it is, and stops being
/// suppressed fleet-wide for seven days as if it had failed.
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
        let Some(scored) = full
            .individuals
            .iter()
            .find(|i| i.uuids.first() == Some(&w.candidate.uuid))
        else {
            continue;
        };
        seen.insert(w.candidate.uuid.as_str());
        verdicts.push(Verdict {
            uuid: w.candidate.uuid.as_str(),
            kind: w.candidate.kind,
            outcome: if win.contains(w.candidate.uuid.as_str()) {
                Outcome::Accepted
            } else {
                Outcome::Rejected
            },
            full_delta: Some(scored.delta),
            group: None,
        });
    }
    if let Some(winner) = &full.winner {
        // A group cut is filed as the neighbourhood it was: every member
        // carries the whole membership so replay can rebuild it (Issue #108).
        // Without that, N members each look like a lone win that loses when it
        // is retried alone — which is exactly what the group was proposed to
        // get past.
        let group = (winner.candidate.kind == "group").then_some(&winner.candidate.uuids[..]);
        for uuid in &winner.candidate.uuids {
            if seen.contains(uuid.as_str()) {
                continue;
            }
            verdicts.push(Verdict {
                uuid: uuid.as_str(),
                kind: if group.is_some() {
                    crate::sweep::CandidateKind::Group
                } else {
                    crate::sweep::CandidateKind::Ablation
                },
                outcome: Outcome::Accepted,
                // Measured only inside the winning bundle or group, so its
                // individual contribution is unknown — never guess it.
                full_delta: None,
                group,
            });
        }
    }
    let n = file_verdicts(store, &verdicts, known);
    if n > 0 {
        log::detail(&format!("learnings: filed {n} full-corpus verdict(s)"));
    }
}

/// What an accept stamped on the `ockham` check-in tag.
///
/// Kept so a coverage tail can re-stamp the tag with the coverage the run
/// finished on (Issue #91): the accept publishes `best.json` before the tail
/// screens anything, and the tag's `sweep X/Y` is meant to agree with the
/// run's end-of-loop coverage rather than to freeze at the moment of the cut.
struct StampedAccept {
    accepts: u64,
    /// Search batches attempted when the accept landed.
    ///
    /// The re-stamp keeps this figure rather than the run's final count: the
    /// tag renders it as `N accepts / M batches`, and a coverage tail's batches
    /// bought no accept, so folding them in would inflate the fleet's own
    /// health signal (Issue #91).
    experiments: u64,
    score: f64,
    error: f64,
    last: &'static str,
    origin: &'static str,
    cuts: usize,
    checksum: String,
}

/// Journal what the cascade dry-run predicted beside what the accept removed.
///
/// The prediction is topology-only and the accepted creature is whatever the
/// scorer actually preferred — an IDENTITY collapse, a constant substitution
/// that keeps the edge, or a bundle — so the two legitimately differ. Recording
/// both is what makes the ranking signal auditable rather than assumed
/// (Issue #106).
fn journal_cascade(
    path: &std::path::Path,
    before: &CreatureExport,
    after: &CreatureExport,
    uuids: &[String],
    kind: &str,
) -> Result<(), String> {
    let estimate = crate::cascade::estimate_cut(before, uuids);
    let opening = crate::ablation::StructureSnapshot::of(before);
    let closing = crate::ablation::StructureSnapshot::of(after);
    let actual_growth_units = opening.growth_units - closing.growth_units;
    // Signed, so a transform that adds structure on one axis while its growth
    // units still fall is recorded as the addition it is (a collapse rewiring
    // a fan-in × fan-out neuron can do exactly that).
    let removed = |before: usize, after: usize| before as i64 - after as i64;
    log::detail(&format!(
        "cascade: {kind} estimated {:.1} growth units, removed {actual_growth_units:.1}",
        estimate.growth_units
    ));
    journal::append(
        path,
        &Event::Cascade {
            kind: kind.to_string(),
            cuts: uuids.len(),
            estimated_hidden: estimate.hidden_neurons(),
            estimated_synapses: estimate.synapses,
            estimated_growth_units: estimate.growth_units,
            actual_hidden: removed(opening.hidden_neurons, closing.hidden_neurons),
            actual_synapses: removed(opening.synapses, closing.synapses),
            actual_growth_units,
        },
    )
}

/// Write the tagged creature to `best.json` and to the winners archive.
fn publish_best(
    config: &OckhamConfig,
    meta: &CreatureMeta,
    creature: &CreatureExport,
    checksum: &str,
) -> Result<(), String> {
    let tagged = meta
        .serialize_with(creature, true)
        .map_err(|e| format!("tag best.json: {e}"))?;
    std::fs::write(config.output_dir.join("best.json"), &tagged)
        .map_err(|e| format!("best.json: {e}"))?;
    let winners_dir = config.output_dir.join("winners");
    if let Err(e) = std::fs::create_dir_all(&winners_dir)
        .and_then(|_| std::fs::write(winners_dir.join(format!("{checksum}.json")), &tagged))
    {
        log::warn(&format!("winners archive not written: {e}"));
    }
    Ok(())
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
    screens: Option<&[Screened]>,
    opening_hidden: usize,
) -> Result<StampedAccept, String> {
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
    // Before the incumbent is replaced: the estimate that ranked this cut was
    // made against the creature being cut, so it is checked against the same
    // one (Issue #106).
    journal_cascade(
        &config.output_dir.join("experiments.jsonl"),
        &incumbent.creature,
        &win.creature,
        &win.candidate.uuids,
        last,
    )?;
    meta.retain_neurons(&win.creature);
    // Coverage over the creature we are about to publish, so the tag agrees
    // with the run's end-of-loop coverage journal. Absent without a screen
    // store: there is no coverage state to report.
    let coverage = screens.map(|screens| {
        let tagged_uuids: HashSet<String> = meta.neuron_tags.keys().cloned().collect();
        let hidden = win
            .creature
            .neurons
            .iter()
            .filter(|n| n.neuron_type == "hidden")
            .count();
        crate::coverage::coverage(
            &win.creature,
            &tagged_uuids,
            screens,
            opening_hidden.saturating_sub(hidden),
        )
    });
    meta.stamp_acceptance(&OckhamProgress {
        accepts: *accepts,
        experiments,
        opening: opening_score,
        score: *current_score,
        error: win.candidate.error,
        last,
        origin,
        cuts,
        coverage,
        // Named only where there is coverage to scope: the clause qualifies the
        // percentage, so without one there is nothing to qualify (Issue #102).
        epoch: coverage.map(|_| corpus.identity.as_str()),
    });
    publish_best(config, meta, &win.creature, &win.checksum)?;
    let stamped = StampedAccept {
        accepts: *accepts,
        experiments,
        score: *current_score,
        error: win.candidate.error,
        last,
        origin,
        cuts,
        checksum: win.checksum.clone(),
    };
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
        &config.stats_sample_spec(),
    )?;
    Ok(stamped)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::baseline::fake::ScriptedScorer;
    use crate::corpus::write_bin_file;
    use crate::coverage::Coverage;
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
        // No hidden neurons, so there is nothing to measure and the scan is
        // skipped outright rather than streaming the corpus for an empty
        // result (#44).
        assert_eq!(run.activation.record_count, 0);
        assert_eq!(run.activation.corpus_record_count, 2);
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

    /// Replaces `max_accepts_still_stops_new_discoveries` (Issue #96): the cap
    /// is gone, so an accept restarts the sweep instead of ending the search —
    /// both hidden neurons are cut in the one run.
    #[test]
    fn an_accept_keeps_the_search_going() {
        let tmp = tempfile::tempdir().unwrap();
        let (creature, train) = two_hidden_paths(tmp.path());
        let cfg = OckhamConfig {
            creature,
            training_data: train,
            output_dir: tmp.path().join("out"),
            timeout: Duration::from_secs(30),
            max_experiments: Some(8),
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
        assert!(run.accepts > 1, "accepts={}", run.accepts);
        assert_eq!(
            run.stop_reason, "no-hidden",
            "the search runs on until the creature has nothing left to cut"
        );
    }

    #[test]
    fn replay_applies_every_known_win() {
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
                    full_delta: None,
                    group: None,
                })
                .unwrap();
        }
        let cfg = OckhamConfig {
            creature,
            training_data: train,
            output_dir: tmp.path().join("out"),
            timeout: Duration::from_secs(30),
            max_experiments: Some(8),
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

    /// `input-0 → a1 → a2 → output-0`, beside a lone `z → output-0`, on disk
    /// with a training corpus (Issue #108).
    fn chain_paths(tmp: &std::path::Path) -> (std::path::PathBuf, std::path::PathBuf) {
        let creature = tmp.join("creature.json");
        let c = crate::fixtures::creature(
            1,
            1,
            vec![
                crate::fixtures::neuron("hidden", "a1", 0.0, Some("TANH")),
                crate::fixtures::neuron("hidden", "a2", 0.0, Some("TANH")),
                crate::fixtures::neuron("hidden", "z", 0.0, Some("TANH")),
                crate::fixtures::neuron("output", "output-0", 0.0, Some("IDENTITY")),
            ],
            vec![
                crate::fixtures::synapse("input-0", "a1", 1.0),
                crate::fixtures::synapse("a1", "a2", 1.0),
                crate::fixtures::synapse("a2", "output-0", 0.01),
                crate::fixtures::synapse("input-0", "z", 1.0),
                crate::fixtures::synapse("z", "output-0", 1.0),
            ],
        );
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

    /// Issue #108: a chain whose members are all standing individual failures
    /// is exactly the dead wood one-neuron-at-a-time screening cannot reach.
    /// The group is proposed anyway, scored as a group, accepted by the full
    /// corpus, and filed with the membership replay needs.
    #[test]
    fn a_group_cut_the_single_neuron_sweep_cannot_reach_is_accepted_and_filed() {
        let tmp = tempfile::tempdir().unwrap();
        let (creature, train) = chain_paths(tmp.path());
        let learnings_dir = tmp.path().join("learnings");
        let corpus = crate::corpus::corpus_info(&train, &TrainingDataConfig::new(1, 1)).unwrap();
        let store = LearningsStore::new(&learnings_dir, corpus.identity.clone(), "t".into());
        // Fresh individual rejections: the sweep skips all three as known
        // failures, so the only candidate left to propose is the group.
        for uuid in ["a1", "a2", "z"] {
            store
                .append(&crate::learnings::Learning {
                    version: crate::learnings::LEARNINGS_FORMAT_VERSION,
                    uuid: uuid.into(),
                    kind: "ablation".into(),
                    outcome: Outcome::Rejected,
                    unix_secs: crate::incumbent::now_unix(),
                    host: "t".into(),
                    full_delta: None,
                    group: None,
                })
                .unwrap();
        }
        let cfg = OckhamConfig {
            creature,
            training_data: train,
            output_dir: tmp.path().join("out"),
            timeout: Duration::from_secs(30),
            max_experiments: Some(1),
            seed: Some(1),
            candidates: 8,
            screen_sample_rate: None,
            learnings_dir: Some(learnings_dir),
            learnings_host: Some("t".into()),
            group_cuts: true,
            ..OckhamConfig::default()
        };
        let scorer = ScriptedScorer {
            baseline_score: 0.50,
            candidate_score: Some(0.90),
            ..ScriptedScorer::ok(0.50, 0.50)
        };
        let run = establish_run(&cfg, &scorer).unwrap();
        assert_eq!(run.accepts, 1, "stop={}", run.stop_reason);

        let journal_path = cfg.output_dir.join("experiments.jsonl");
        let journal = std::fs::read_to_string(&journal_path).unwrap();
        assert!(journal.contains(r#""groups":1"#), "{journal}");
        assert!(journal.contains(r#""kind":"group""#), "{journal}");

        // The accepted creature lost the whole chain and kept the neuron that
        // was never in the group.
        let best = std::fs::read_to_string(cfg.output_dir.join("best.json")).unwrap();
        assert!(!best.contains("\"a1\""), "{best}");
        assert!(!best.contains("\"a2\""), "{best}");
        assert!(best.contains("\"z\""), "{best}");

        // Both members carry the whole membership, so replay can rebuild it.
        let filed: Vec<crate::learnings::Learning> = store
            .load()
            .unwrap()
            .into_iter()
            .filter(|l| l.outcome == Outcome::Accepted)
            .collect();
        assert_eq!(filed.len(), 2, "{filed:?}");
        for learning in &filed {
            assert_eq!(learning.kind, "group");
            assert_eq!(
                learning.group.as_deref(),
                Some(&["a1".to_string(), "a2".to_string()][..]),
                "{learning:?}"
            );
        }

        // A group screen is not coverage of its members: no screen record
        // claims either neuron was checked on its own by this cohort.
        assert!(
            store
                .load_screens()
                .unwrap()
                .iter()
                .all(|s| s.kind != "group"),
            "a group must file no per-neuron screen record"
        );

        let report = crate::report::summarise(&[&journal_path]).unwrap();
        assert_eq!(report.group_accepts, 1);
        assert_eq!(report.group_cuts_accepted, 2);
        assert_eq!(report.group_hidden_removed, 2);
        assert!(report.group_growth_units_removed > 0.0);
        assert_eq!(
            report.group_growth_units_per_accept,
            Some(report.group_growth_units_removed)
        );
    }

    /// Issue #108: a later run rebuilds an accepted neighbourhood from the
    /// membership its members recorded, even though each member's own latest
    /// verdict says the cut loses on its own — which is the whole reason the
    /// group was proposed. Without the membership the plan is unreconstructable
    /// and the fleet forgets a win it has already paid for.
    #[test]
    fn a_recorded_group_is_replayed_as_a_group_by_a_later_run() {
        let tmp = tempfile::tempdir().unwrap();
        let (creature, train) = chain_paths(tmp.path());
        let learnings_dir = tmp.path().join("learnings");
        let corpus = crate::corpus::corpus_info(&train, &TrainingDataConfig::new(1, 1)).unwrap();
        let store = LearningsStore::new(&learnings_dir, corpus.identity.clone(), "t".into());
        let now = crate::incumbent::now_unix();
        for uuid in ["a1", "a2"] {
            // The group that was accepted, then a fresher verdict rejecting the
            // same neuron on its own.
            store
                .append(&crate::learnings::Learning {
                    version: crate::learnings::LEARNINGS_FORMAT_VERSION,
                    uuid: uuid.into(),
                    kind: "group".into(),
                    outcome: Outcome::Accepted,
                    unix_secs: now - 100,
                    host: "t".into(),
                    full_delta: None,
                    group: Some(vec!["a1".into(), "a2".into()]),
                })
                .unwrap();
            store
                .append(&crate::learnings::Learning {
                    version: crate::learnings::LEARNINGS_FORMAT_VERSION,
                    uuid: uuid.into(),
                    kind: "ablation".into(),
                    outcome: Outcome::Rejected,
                    unix_secs: now,
                    host: "t".into(),
                    full_delta: Some(-0.2),
                    group: None,
                })
                .unwrap();
        }
        let cfg = OckhamConfig {
            creature,
            training_data: train,
            output_dir: tmp.path().join("out"),
            timeout: Duration::from_secs(30),
            max_experiments: Some(1),
            seed: Some(1),
            candidates: 8,
            screen_sample_rate: None,
            learnings_dir: Some(learnings_dir),
            learnings_host: Some("t".into()),
            group_cuts: true,
            ..OckhamConfig::default()
        };
        let scorer = ScriptedScorer {
            baseline_score: 0.50,
            candidate_score: Some(0.90),
            ..ScriptedScorer::ok(0.50, 0.50)
        };
        let run = establish_run(&cfg, &scorer).unwrap();
        assert_eq!(run.accepts, 1, "stop={}", run.stop_reason);
        assert_eq!(run.stop_reason, "replay-accepts");
        let best = std::fs::read_to_string(cfg.output_dir.join("best.json")).unwrap();
        assert!(!best.contains("\"a1\""), "{best}");
        assert!(!best.contains("\"a2\""), "{best}");
    }

    /// Issue #108: a group's verdict is about the neighbourhood, so it must
    /// never become a training row about one of its neurons — the ranker would
    /// learn that a neuron the scorer never judged alone is a confirmed cut.
    #[test]
    fn a_group_verdict_never_becomes_a_per_neuron_training_row() {
        let tmp = tempfile::tempdir().unwrap();
        let (creature, train) = chain_paths(tmp.path());
        let log = tmp.path().join("candidates.jsonl");
        let cfg = OckhamConfig {
            creature,
            training_data: train,
            output_dir: tmp.path().join("out"),
            timeout: Duration::from_secs(30),
            max_experiments: Some(1),
            seed: Some(1),
            candidates: 8,
            screen_sample_rate: None,
            candidate_log: Some(log.clone()),
            group_cuts: true,
            ..OckhamConfig::default()
        };
        // Every candidate loses, so the cohort is judged without an accept
        // rewriting the incumbent underneath the log.
        let scorer = ScriptedScorer {
            baseline_score: 0.50,
            candidate_score: Some(0.40),
            ..ScriptedScorer::ok(0.50, 0.50)
        };
        establish_run(&cfg, &scorer).unwrap();
        let records = crate::telemetry::load(&log).unwrap();
        assert!(!records.is_empty(), "the judged individuals must be logged");
        assert!(
            records.iter().all(|r| r.kind != "group"),
            "a group cut names no single neuron: {records:?}"
        );
        // Every logged uuid was scored on its own, so its delta is its own.
        assert!(records.iter().all(|r| r.full_delta.is_some()));
    }

    /// A control run must stay a control run: without the flag no group
    /// candidate is built, scored or counted (Issue #108).
    #[test]
    fn without_the_flag_a_run_proposes_no_group_at_all() {
        let tmp = tempfile::tempdir().unwrap();
        let (creature, train) = chain_paths(tmp.path());
        let cfg = OckhamConfig {
            creature,
            training_data: train,
            output_dir: tmp.path().join("out"),
            timeout: Duration::from_secs(30),
            max_experiments: Some(1),
            seed: Some(1),
            candidates: 8,
            screen_sample_rate: None,
            ..OckhamConfig::default()
        };
        let scorer = ScriptedScorer {
            baseline_score: 0.50,
            candidate_score: Some(0.40),
            ..ScriptedScorer::ok(0.50, 0.50)
        };
        establish_run(&cfg, &scorer).unwrap();
        let journal_path = cfg.output_dir.join("experiments.jsonl");
        let journal = std::fs::read_to_string(&journal_path).unwrap();
        assert!(!journal.contains(r#""kind":"group""#), "{journal}");
        assert!(journal.contains(r#""groups":0"#), "{journal}");
        assert_eq!(
            crate::report::summarise(&[&journal_path])
                .unwrap()
                .group_accepts,
            0
        );
    }

    /// Add a GRQ-style tag to each named neuron of a creature file.
    fn tag_neurons(creature: &std::path::Path, uuids: &[&str]) {
        let mut v: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(creature).unwrap()).unwrap();
        for n in v["neurons"].as_array_mut().unwrap() {
            let uuid = n["uuid"].as_str().unwrap_or_default().to_string();
            if uuids.contains(&uuid.as_str()) {
                n.as_object_mut().unwrap().insert(
                    "tags".into(),
                    serde_json::json!([{"name":"discovered","value":uuid}]),
                );
            }
        }
        std::fs::write(creature, serde_json::to_string_pretty(&v).unwrap()).unwrap();
    }

    /// Replaces `replay_leaves_tagged_source_neurons_in_place`: #63 reverses
    /// #26, so a confirmed win on a tagged uuid replays like any other.
    #[test]
    fn replay_cuts_a_confirmed_win_on_a_tagged_neuron() {
        let tmp = tempfile::tempdir().unwrap();
        let (creature, train) = two_hidden_paths(tmp.path());
        tag_neurons(&creature, &["h_a"]);
        let learnings_dir = tmp.path().join("learnings");
        let td = TrainingDataConfig::new(1, 1);
        let corpus = crate::corpus::corpus_info(&train, &td).unwrap();
        let store = LearningsStore::new(&learnings_dir, corpus.identity.clone(), "t".into());
        store
            .append(&crate::learnings::Learning {
                version: crate::learnings::LEARNINGS_FORMAT_VERSION,
                uuid: "h_a".into(),
                kind: "identity".into(),
                outcome: Outcome::Accepted,
                unix_secs: 10,
                host: "t".into(),
                full_delta: None,
                group: None,
            })
            .unwrap();
        let cfg = OckhamConfig {
            creature,
            training_data: train,
            output_dir: tmp.path().join("out"),
            timeout: Duration::from_secs(30),
            max_experiments: Some(4),
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
        assert_eq!(run.stop_reason, "replay-accepts");
        let best = std::fs::read_to_string(cfg.output_dir.join("best.json")).unwrap();
        assert!(
            !best.contains("h_a"),
            "a tagged known win must be replayable: {best}"
        );
        assert!(
            !best.contains("discovered"),
            "the cut neuron's tag must not survive it: {best}"
        );
    }

    #[test]
    fn a_tagged_hidden_neuron_that_improves_the_score_is_proposed_screened_and_accepted() {
        let tmp = tempfile::tempdir().unwrap();
        let (creature, train) = two_hidden_paths(tmp.path());
        tag_neurons(&creature, &["h_a", "h_b"]);
        let learnings_dir = tmp.path().join("learnings");
        let cfg = OckhamConfig {
            creature,
            training_data: train.clone(),
            output_dir: tmp.path().join("out"),
            timeout: Duration::from_secs(30),
            max_experiments: Some(4),
            seed: Some(1),
            candidates: 8,
            screen_sample_rate: Some(0.5),
            learnings_dir: Some(learnings_dir.clone()),
            learnings_host: Some("t".into()),
            ..OckhamConfig::default()
        };
        let scorer = ScriptedScorer {
            baseline_score: 0.50,
            candidate_score: Some(0.80),
            ..ScriptedScorer::ok(0.50, 0.50)
        };
        let run = establish_run(&cfg, &scorer).unwrap();
        assert!(
            run.accepts >= 1,
            "a tagged neuron must be acceptable; accepts={} stop={}",
            run.accepts,
            run.stop_reason
        );
        // Screened: every tagged candidate the batch scored left a screen record.
        let screened = screened_uuids(&screens_store(&learnings_dir, &train));
        for uuid in ["h_a", "h_b"] {
            assert!(
                screened.iter().any(|s| s == uuid),
                "{uuid} must reach the screen: {screened:?}"
            );
        }
        // Proposed: the batch emitted both tagged neurons and skipped neither.
        let journal = std::fs::read_to_string(cfg.output_dir.join("experiments.jsonl")).unwrap();
        assert!(
            journal.contains(r#""candidates":2,"skipped":0"#),
            "both tagged neurons must be proposed: {journal}"
        );
        let best = std::fs::read_to_string(cfg.output_dir.join("best.json")).unwrap();
        assert!(
            !best.contains("h_a") || !best.contains("h_b"),
            "a tagged neuron must have been cut: {best}"
        );
    }

    /// Issue #106: a cut is ranked on a topology-only prediction, so the run
    /// has to record what that prediction was worth once the scorer accepted
    /// it. Without the record the ranking signal can drift from the structure
    /// the razor really removes and nothing says so.
    #[test]
    fn an_accepted_cut_journals_the_estimated_cascade_beside_the_actual_saving() {
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
            screen_sample_rate: None,
            ordering: crate::ordering::Ordering::CascadeSaving,
            ..OckhamConfig::default()
        };
        let scorer = ScriptedScorer {
            baseline_score: 0.50,
            candidate_score: Some(0.80),
            ..ScriptedScorer::ok(0.50, 0.50)
        };
        let run = establish_run(&cfg, &scorer).unwrap();
        assert!(run.accepts >= 1, "stop={}", run.stop_reason);
        let journal_path = cfg.output_dir.join("experiments.jsonl");
        let journal = std::fs::read_to_string(&journal_path).unwrap();
        assert!(journal.contains(r#""record":"cascade""#), "{journal}");
        assert!(journal.contains("estimated_growth_units"), "{journal}");
        let report = crate::report::summarise(&[&journal_path]).unwrap();
        let estimated = report
            .cascade_estimated_growth_units
            .expect("an accept must record its estimate");
        let actual = report
            .cascade_actual_growth_units
            .expect("an accept must record what it removed");
        assert!(estimated > 0.0 && actual > 0.0, "{estimated} vs {actual}");
        // Under `1.0` because the winner was an exact IDENTITY collapse: it
        // rewires `input-0 → output-0` where the topology dry-run predicted
        // both synapses would go. Recording the difference is the point — the
        // estimate ranks candidates, it does not describe the transform the
        // scorer ends up accepting.
        let ratio = report.cascade_estimate_ratio.expect("estimate and actual");
        assert!(ratio > 0.5 && ratio <= 1.0, "{ratio}: {journal}");
    }

    /// Issue #107: the learned ranker is only as good as the rows it is fitted
    /// from, so a run must record one per candidate the scorer actually judged.
    #[test]
    fn a_candidate_log_records_the_features_and_the_verdict_of_every_judged_candidate() {
        let tmp = tempfile::tempdir().unwrap();
        let (creature, train) = two_hidden_paths(tmp.path());
        let log = tmp.path().join("telemetry").join("candidates.jsonl");
        let cfg = OckhamConfig {
            creature,
            training_data: train,
            output_dir: tmp.path().join("out"),
            timeout: Duration::from_secs(30),
            max_experiments: Some(2),
            seed: Some(1),
            candidates: 8,
            screen_sample_rate: None,
            ordering: crate::ordering::Ordering::Composite,
            candidate_log: Some(log.clone()),
            ..OckhamConfig::default()
        };
        let scorer = ScriptedScorer {
            baseline_score: 0.50,
            candidate_score: Some(0.80),
            ..ScriptedScorer::ok(0.50, 0.50)
        };
        let run = establish_run(&cfg, &scorer).unwrap();
        assert!(run.accepts >= 1, "stop={}", run.stop_reason);
        let records = crate::telemetry::load(&log).unwrap();
        assert!(
            !records.is_empty(),
            "the run judged candidates but logged none"
        );
        let accepted: Vec<_> = records
            .iter()
            .filter(|r| r.outcome == crate::telemetry::CandidateOutcome::Accepted)
            .collect();
        assert_eq!(
            accepted.len(),
            run.accepts as usize,
            "one accepted row per accept: {records:?}"
        );
        let win = accepted[0];
        assert!(win.full_delta.unwrap() > 0.0, "{win:?}");
        assert!(
            win.growth_units_removed > 0.0,
            "an accepted cut removed structure: {win:?}"
        );
        assert!(win.is_win(cfg.min_improvement));
        assert_eq!(win.ordering, "composite");
        assert_eq!(win.seed, 1);
        for name in crate::features::FEATURE_NAMES {
            assert!(win.features.contains_key(*name), "missing {name}: {win:?}");
        }
        // Every row is trainable against the current schema.
        let (rows, skipped) = crate::telemetry::training_rows(&records, cfg.min_improvement);
        assert_eq!(skipped, 0);
        assert_eq!(rows.len(), records.len());
    }

    /// A candidate the sampled screen threw out is training data too: it is the
    /// only evidence the ranker gets about what does *not* work.
    #[test]
    fn a_screened_out_candidate_is_logged_with_its_sampled_delta() {
        let tmp = tempfile::tempdir().unwrap();
        let (creature, train) = two_hidden_paths(tmp.path());
        let log = tmp.path().join("candidates.jsonl");
        let cfg = OckhamConfig {
            creature,
            training_data: train,
            output_dir: tmp.path().join("out"),
            timeout: Duration::from_secs(30),
            max_experiments: Some(1),
            seed: Some(1),
            candidates: 8,
            screen_sample_rate: Some(0.5),
            candidate_log: Some(log.clone()),
            ..OckhamConfig::default()
        };
        // Candidates that never beat the incumbent: every one is screened out.
        let scorer = ScriptedScorer {
            baseline_score: 0.50,
            candidate_score: Some(0.10),
            ..ScriptedScorer::ok(0.50, 0.50)
        };
        establish_run(&cfg, &scorer).unwrap();
        let records = crate::telemetry::load(&log).unwrap();
        assert!(
            !records.is_empty(),
            "screened-out candidates must be logged"
        );
        assert!(
            records
                .iter()
                .all(|r| r.outcome == crate::telemetry::CandidateOutcome::ScreenedOut),
            "{records:?}"
        );
        let row = &records[0];
        assert!(row.sample_delta.unwrap() < 0.0, "{row:?}");
        assert!(
            row.full_delta.is_none(),
            "nothing was fully scored: {row:?}"
        );
        assert_eq!(row.growth_units_removed, 0.0);
        assert!(!row.is_win(cfg.min_improvement));
    }

    /// The candidate log is evidence about the search, not part of it: a log
    /// that cannot be written warns and the run keeps pruning (#107).
    #[test]
    fn an_unwritable_candidate_log_warns_without_stopping_the_run() {
        let tmp = tempfile::tempdir().unwrap();
        let (creature, train) = two_hidden_paths(tmp.path());
        // A file where the log's parent directory should be.
        let blocker = tmp.path().join("blocked");
        std::fs::write(&blocker, "not a directory").unwrap();
        let cfg = OckhamConfig {
            creature,
            training_data: train,
            output_dir: tmp.path().join("out"),
            timeout: Duration::from_secs(30),
            max_experiments: Some(2),
            seed: Some(1),
            candidates: 8,
            screen_sample_rate: None,
            candidate_log: Some(blocker.join("candidates.jsonl")),
            ..OckhamConfig::default()
        };
        let scorer = ScriptedScorer {
            baseline_score: 0.50,
            candidate_score: Some(0.80),
            ..ScriptedScorer::ok(0.50, 0.50)
        };
        let run = establish_run(&cfg, &scorer).unwrap();
        assert!(run.accepts >= 1, "stop={}", run.stop_reason);
        assert!(
            !blocker.is_dir(),
            "the run must not have created the log directory"
        );
    }

    /// A run that asked to be ranked by a model must stop when the model is
    /// missing, never rank by something else and say nothing (#107).
    #[test]
    fn a_learned_run_without_a_readable_model_stops_rather_than_ranking_by_something_else() {
        let tmp = tempfile::tempdir().unwrap();
        let (creature, train) = two_hidden_paths(tmp.path());
        let cfg = OckhamConfig {
            creature,
            training_data: train,
            output_dir: tmp.path().join("out"),
            timeout: Duration::from_secs(30),
            max_experiments: Some(1),
            seed: Some(1),
            screen_sample_rate: None,
            ordering: crate::ordering::Ordering::Learned,
            ordering_model: Some(tmp.path().join("absent-model.json")),
            ..OckhamConfig::default()
        };
        let scorer = ScriptedScorer::ok(0.50, 0.50);
        let err = establish_run(&cfg, &scorer).unwrap_err();
        assert!(err.contains("absent-model.json"), "{err}");
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
    ///
    /// Filed under the store's own corpus identity: since #100 coverage counts
    /// the epoch of the corpus in hand, so a fixture naming another corpus
    /// would be read as history and seed no coverage at all.
    fn seed_screens(store: &LearningsStore, uuids: &[(&str, u64)]) {
        for (uuid, unix_secs) in uuids {
            store
                .append_screen(&Screened {
                    blocked_reason: Default::default(),
                    version: crate::learnings::SCREENS_FORMAT_VERSION,
                    uuid: (*uuid).into(),
                    kind: "identity".into(),
                    outcome: ScreenOutcomeKind::Loser,
                    unix_secs: *unix_secs,
                    host: "t".into(),
                    corpus_identity: Some(store.corpus_identity().to_string()),
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

    /// Config for the old-corpus priority tests: exactly one neuron per run.
    ///
    /// The same seeded, cache-backed single-batch run the unchecked-first tests
    /// use, narrowed to one candidate so the uuid the sweep reaches first is the
    /// only one screened.
    fn old_corpus_cfg(
        creature: std::path::PathBuf,
        train: std::path::PathBuf,
        out: std::path::PathBuf,
        learnings_dir: std::path::PathBuf,
        old_corpus_first: Option<bool>,
    ) -> OckhamConfig {
        OckhamConfig {
            candidates: 1,
            old_corpus_first,
            ..unchecked_first_cfg(creature, train, out, learnings_dir, None)
        }
    }

    /// File one verdict under a corpus identity this run will never load.
    fn seed_prior_corpus(
        learnings_dir: &std::path::Path,
        uuid: &str,
        outcome: Outcome,
        unix_secs: u64,
    ) {
        let store = LearningsStore::new(learnings_dir, "an-older-corpus".into(), "GRQ-23".into());
        store
            .append(&crate::learnings::Learning {
                version: crate::learnings::LEARNINGS_FORMAT_VERSION,
                uuid: uuid.into(),
                kind: "identity".into(),
                outcome,
                unix_secs,
                host: "GRQ-23".into(),
                full_delta: None,
                group: None,
            })
            .unwrap();
    }

    /// The one uuid a control run — same seed, no old-corpus hint — screens.
    fn control_screened(tmp: &std::path::Path, uuids: &[&str]) -> String {
        let (creature, train) = hidden_paths(tmp, uuids);
        let learnings_dir = tmp.join("learnings");
        let store = screens_store(&learnings_dir, &train);
        let cfg = old_corpus_cfg(creature, train, tmp.join("out"), learnings_dir, None);
        establish_run(&cfg, &ScriptedScorer::ok(0.50, 0.50)).unwrap();
        let screened = screened_this_run(&store, 0);
        assert_eq!(screened.len(), 1, "one batch of one: {screened:?}");
        screened.into_iter().next().unwrap()
    }

    /// Every uuid this run visited, in the order the records were filed.
    fn visits_in_order(store: &LearningsStore) -> Vec<String> {
        store
            .load_screens()
            .unwrap()
            .into_iter()
            .map(|s| s.uuid)
            .collect()
    }

    /// Issue #88: a neuron an older corpus removed is still on the incumbent and
    /// unchecked here, so it is the first thing this run looks at — ahead of the
    /// neurons the seeded permutation would otherwise have reached first.
    ///
    /// Since Issue #101 the run acts on that evidence twice, so the budget is
    /// two experiments rather than one: the replay stage re-scores the old
    /// winner against the corpus in hand first, and the sweep then visits it
    /// ahead of the neurons with no history. The flat scorer rejects the replay,
    /// so that first visit is filed as a `known-failure` skip rather than a
    /// screen — this corpus has just judged the uuid, and a visit is a visit.
    /// The assertion moved from "the one uuid screened" to "the first uuid
    /// visited" for that reason, and still discriminates: without the priority
    /// the sweep would have reached `reached` first, as the sibling test shows.
    #[test]
    fn an_old_corpus_win_is_screened_before_neurons_with_no_history() {
        let uuids = ["h_a", "h_b", "h_c", "h_d"];
        let control = tempfile::tempdir().unwrap();
        let reached = control_screened(control.path(), &uuids);
        let target = uuids
            .iter()
            .find(|u| **u != reached)
            .expect("a neuron the control run did not reach");

        let tmp = tempfile::tempdir().unwrap();
        let (creature, train) = hidden_paths(tmp.path(), &uuids);
        let learnings_dir = tmp.path().join("learnings");
        seed_prior_corpus(&learnings_dir, target, Outcome::Accepted, 10);
        let store = screens_store(&learnings_dir, &train);
        let cfg = OckhamConfig {
            max_experiments: Some(2),
            ..old_corpus_cfg(creature, train, tmp.path().join("out"), learnings_dir, None)
        };
        assert!(
            cfg.old_corpus_first_enabled(),
            "--learnings-dir turns it on"
        );
        establish_run(&cfg, &ScriptedScorer::ok(0.50, 0.50)).unwrap();

        assert_eq!(
            visits_in_order(&store).first().map(String::as_str),
            Some(*target),
            "the old-corpus win must be checked before neurons with no history"
        );
        // The reorder happens after `permutation_identity` is hashed, so the
        // journal must say it happened for the run to be reconstructable.
        let journal = std::fs::read_to_string(cfg.output_dir.join("experiments.jsonl")).unwrap();
        assert!(journal.contains(r#""old_corpus_first":1"#), "{journal}");
    }

    #[test]
    fn old_corpus_first_off_keeps_the_order_the_run_would_have_had() {
        let uuids = ["h_a", "h_b", "h_c", "h_d"];
        let control = tempfile::tempdir().unwrap();
        let reached = control_screened(control.path(), &uuids);
        let target = uuids.iter().find(|u| **u != reached).unwrap();

        let tmp = tempfile::tempdir().unwrap();
        let (creature, train) = hidden_paths(tmp.path(), &uuids);
        let learnings_dir = tmp.path().join("learnings");
        seed_prior_corpus(&learnings_dir, target, Outcome::Accepted, 10);
        let store = screens_store(&learnings_dir, &train);
        let cfg = old_corpus_cfg(
            creature,
            train,
            tmp.path().join("out"),
            learnings_dir,
            Some(false),
        );
        establish_run(&cfg, &ScriptedScorer::ok(0.50, 0.50)).unwrap();

        assert_eq!(
            screened_this_run(&store, 0),
            vec![reached],
            "with the priority off the seeded permutation stands"
        );
    }

    /// The near miss this must not make: old data is a hint, so a foreign-corpus
    /// **rejection** must not suppress the candidate the way this corpus's own
    /// fresh rejection does.
    #[test]
    fn a_prior_corpus_rejection_does_not_suppress_this_run() {
        let tmp = tempfile::tempdir().unwrap();
        let (creature, train) = hidden_paths(tmp.path(), &["h_a", "h_b"]);
        let learnings_dir = tmp.path().join("learnings");
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        for uuid in ["h_a", "h_b"] {
            seed_prior_corpus(&learnings_dir, uuid, Outcome::Rejected, now);
        }
        let store = screens_store(&learnings_dir, &train);
        let cfg = OckhamConfig {
            candidates: 2,
            ..old_corpus_cfg(creature, train, tmp.path().join("out"), learnings_dir, None)
        };
        establish_run(&cfg, &ScriptedScorer::ok(0.50, 0.50)).unwrap();

        let records = store.load_screens().unwrap();
        assert_eq!(records.len(), 2, "{records:?}");
        assert!(
            records
                .iter()
                .all(|s| s.kind != crate::learnings::SCREEN_KIND_KNOWN_FAILURE),
            "a foreign-corpus rejection must not suppress a candidate: {records:?}"
        );
    }

    /// Config for the coverage-tail tests: a seeded cache and screening on.
    ///
    /// `screen_threshold` is high enough that the scripted scorer's candidates
    /// lose the sampled screen, which is what a real batch mostly does — and a
    /// loser is the case the tail files coverage for. The winner case has its
    /// own test.
    fn coverage_tail_cfg(
        creature: std::path::PathBuf,
        train: std::path::PathBuf,
        out: std::path::PathBuf,
        learnings_dir: std::path::PathBuf,
        candidates: usize,
        max_experiments: u64,
    ) -> OckhamConfig {
        OckhamConfig {
            creature,
            training_data: train,
            output_dir: out,
            timeout: Duration::from_secs(30),
            max_experiments: Some(max_experiments),
            seed: Some(1),
            candidates,
            screen_sample_rate: Some(0.5),
            screen_threshold: 1.0,
            learnings_dir: Some(learnings_dir),
            learnings_host: Some("t".into()),
            ..OckhamConfig::default()
        }
    }

    /// Issue #91: an accept ended the run the moment it landed, so every run
    /// that replayed a known win screened nothing — nine consecutive
    /// GRQ-sampler check-ins reported `progress: 0 newly screened this run`
    /// while the razor kept cutting.
    #[test]
    fn a_replay_accept_still_spends_the_rest_of_the_budget_on_coverage() {
        let tmp = tempfile::tempdir().unwrap();
        let (creature, train) = hidden_paths(tmp.path(), &["h_a", "h_b", "h_c", "h_d", "h_e"]);
        let learnings_dir = tmp.path().join("learnings");
        let store = screens_store(&learnings_dir, &train);
        seed_verdicts(&store, &[("h_a", Outcome::Accepted, None, 10)]);
        let cfg = coverage_tail_cfg(creature, train, tmp.path().join("out"), learnings_dir, 2, 4);
        let scorer = ScriptedScorer {
            baseline_score: 0.50,
            candidate_score: Some(0.80),
            ..ScriptedScorer::ok(0.50, 0.50)
        };
        let run = establish_run(&cfg, &scorer).unwrap();

        assert_eq!(run.accepts, 1);
        assert_eq!(
            run.stop_reason, "replay-accepts",
            "the accept still names why the search ended"
        );
        // Deduplicated: a tail that runs out of unchecked neurons restarts the
        // sweep and re-screens the stalest, exactly as #77 intends.
        let mut screened = screened_this_run(&store, 10);
        screened.dedup();
        assert_eq!(
            screened,
            vec!["h_b", "h_c", "h_d", "h_e"],
            "the accept ends the search, not the run's coverage duty"
        );
        assert_eq!(run.newly_screened, 4);
        let best = std::fs::read_to_string(cfg.output_dir.join("best.json")).unwrap();
        assert!(
            best.contains("sweep 4/4"),
            "the check-in tag must report the coverage the run finished on, not the \
             coverage at the cut: {best}"
        );
        let journal = std::fs::read_to_string(cfg.output_dir.join("experiments.jsonl")).unwrap();
        assert!(
            journal.contains(r#""newly_screened":4"#),
            "the stop record carries the coverage the run advanced: {journal}"
        );
        assert!(
            journal.contains(r#""record":"coverageTail""#) && journal.contains(r#""ended":"#),
            "the tail's own end is journalled, not folded into the accept's stop reason: \
             {journal}"
        );
    }

    /// The tail honours unchecked-first selection (#38): it screens the neuron
    /// the fleet has never checked before recycling a stale one.
    #[test]
    fn the_coverage_tail_screens_the_unchecked_neuron_first() {
        let tmp = tempfile::tempdir().unwrap();
        let (creature, train) = hidden_paths(tmp.path(), &["h_a", "h_b", "h_c", "h_d", "h_e"]);
        let learnings_dir = tmp.path().join("learnings");
        let store = screens_store(&learnings_dir, &train);
        seed_verdicts(&store, &[("h_a", Outcome::Accepted, None, 10)]);
        seed_screens(&store, &[("h_c", 10), ("h_d", 20), ("h_e", 30)]);
        let cfg = coverage_tail_cfg(creature, train, tmp.path().join("out"), learnings_dir, 1, 2);
        let scorer = ScriptedScorer {
            baseline_score: 0.50,
            candidate_score: Some(0.80),
            ..ScriptedScorer::ok(0.50, 0.50)
        };
        let run = establish_run(&cfg, &scorer).unwrap();

        assert_eq!(run.stop_reason, "replay-accepts");
        assert_eq!(
            screened_this_run(&store, 30),
            vec!["h_b"],
            "h_b is the only never-screened neuron left, so it goes first"
        );
    }

    /// An accept that leaves nothing unchecked stops exactly as it always did.
    #[test]
    fn an_accept_that_leaves_no_hidden_neurons_stops_at_once() {
        let tmp = tempfile::tempdir().unwrap();
        let (creature, train) = two_hidden_paths(tmp.path());
        let learnings_dir = tmp.path().join("learnings");
        let store = screens_store(&learnings_dir, &train);
        seed_verdicts(
            &store,
            &[
                ("h_a", Outcome::Accepted, None, 10),
                ("h_b", Outcome::Accepted, None, 20),
            ],
        );
        let cfg = coverage_tail_cfg(creature, train, tmp.path().join("out"), learnings_dir, 2, 8);
        let scorer = ScriptedScorer {
            baseline_score: 0.50,
            candidate_score: Some(0.80),
            ..ScriptedScorer::ok(0.50, 0.50)
        };
        let run = establish_run(&cfg, &scorer).unwrap();
        assert_eq!(run.stop_reason, "replay-accepts");
        assert_eq!(
            screened_this_run(&store, 20),
            Vec::<String>::new(),
            "both hidden neurons were cut, so there is nothing left to screen"
        );
    }

    /// A sampled winner the tail finds is a lead nothing in this run will
    /// score, so filing it as checked would bury it behind every unchecked
    /// neuron on the next run's sweep (Issue #91). It stays unchecked.
    #[test]
    fn the_coverage_tail_leaves_its_sampled_winners_unchecked() {
        let tmp = tempfile::tempdir().unwrap();
        let (creature, train) = hidden_paths(tmp.path(), &["h_a", "h_b", "h_c"]);
        let learnings_dir = tmp.path().join("learnings");
        let store = screens_store(&learnings_dir, &train);
        seed_verdicts(&store, &[("h_a", Outcome::Accepted, None, 10)]);
        // Threshold 0: every scripted candidate beats the sampled incumbent,
        // so the tail's whole batch is winners.
        let cfg = OckhamConfig {
            screen_threshold: 0.0,
            ..coverage_tail_cfg(creature, train, tmp.path().join("out"), learnings_dir, 2, 3)
        };
        let scorer = ScriptedScorer {
            baseline_score: 0.50,
            candidate_score: Some(0.80),
            ..ScriptedScorer::ok(0.50, 0.50)
        };
        let run = establish_run(&cfg, &scorer).unwrap();

        assert_eq!(run.stop_reason, "replay-accepts");
        assert_eq!(
            screened_this_run(&store, 10),
            Vec::<String>::new(),
            "a winner nothing will score must not be filed as checked"
        );
        let journal = std::fs::read_to_string(cfg.output_dir.join("experiments.jsonl")).unwrap();
        assert!(
            journal.contains(r#""record":"coverageTail""#),
            "the tail still journals what it did: {journal}"
        );
    }

    /// Replaces `the_last_allowed_search_accept_still_screens_for_coverage`
    /// (Issue #96): with the accept cap gone a search accept opens no tail —
    /// it restarts the sweep and the run keeps searching and screening on the
    /// budget it has left.
    #[test]
    fn a_search_accept_keeps_searching_instead_of_opening_a_tail() {
        let tmp = tempfile::tempdir().unwrap();
        let (creature, train) = hidden_paths(tmp.path(), &["h_a", "h_b", "h_c", "h_d"]);
        let learnings_dir = tmp.path().join("learnings");
        let store = screens_store(&learnings_dir, &train);
        // Threshold 0 so the first batch's candidates clear the screen, reach
        // full scoring and accept — the search accept this test is about.
        let cfg = OckhamConfig {
            screen_threshold: 0.0,
            ..coverage_tail_cfg(creature, train, tmp.path().join("out"), learnings_dir, 1, 6)
        };
        let scorer = ScriptedScorer {
            baseline_score: 0.50,
            candidate_score: Some(0.80),
            ..ScriptedScorer::ok(0.50, 0.50)
        };
        let run = establish_run(&cfg, &scorer).unwrap();

        assert!(
            run.accepts > 1,
            "the search runs on past its first accept: {run:?}"
        );
        assert_eq!(
            run.stop_reason, "no-hidden",
            "no accept stops it: the search runs on until nothing is left to \
             cut, or the budget ends: {run:?}"
        );
        let journal = std::fs::read_to_string(cfg.output_dir.join("experiments.jsonl")).unwrap();
        assert!(
            !journal.contains(r#""record":"coverageTail""#),
            "a search accept opens no tail — the search itself screens on: {journal}"
        );
        assert!(
            !screened_this_run(&store, 0).is_empty(),
            "the run still advances coverage while it searches"
        );
    }

    /// The tail is refused when there is no sampled screen to check candidates
    /// with. The "no screen store" half of this test went with the accept cap
    /// (Issue #96): it drove the refusal through a *search* accept, and a
    /// search accept no longer opens a tail. The store guard itself stands —
    /// `known` grows in-run even without a store — it is simply no longer
    /// reachable from the run loop in a test.
    #[test]
    fn no_tail_without_a_sampled_screen() {
        let tmp = tempfile::tempdir().unwrap();
        let (creature, train) = hidden_paths(tmp.path(), &["h_a", "h_b", "h_c"]);
        let learnings_dir = tmp.path().join("learnings");
        let store = screens_store(&learnings_dir, &train);
        seed_verdicts(&store, &[("h_a", Outcome::Accepted, None, 10)]);
        let cfg = OckhamConfig {
            creature,
            training_data: train,
            output_dir: tmp.path().join("out"),
            timeout: Duration::from_secs(30),
            max_experiments: Some(4),
            seed: Some(1),
            candidates: 1,
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
        assert_eq!(run.stop_reason, "replay-accepts");
        let journal = std::fs::read_to_string(cfg.output_dir.join("experiments.jsonl")).unwrap();
        assert!(
            !journal.contains(r#""record":"coverageTail""#),
            "with no sampled screen there is nothing to screen with: {journal}"
        );
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

    /// A **repacked** corpus: the same records written to a fresh directory,
    /// so the authoritative content — and therefore the identity — is identical.
    fn repacked_corpus(tmp: &std::path::Path) -> std::path::PathBuf {
        let train = tmp.join("train-repacked");
        std::fs::create_dir(&train).unwrap();
        write_bin_file(
            &train.join("0.bin"),
            &[(vec![1.0f32], vec![1.0f32]), (vec![2.0], vec![2.0])],
        )
        .unwrap();
        train
    }

    /// An **extended** corpus: the fixture's records plus a third, which is what
    /// the fleet actually does to the training data every few days.
    fn extended_corpus(tmp: &std::path::Path) -> std::path::PathBuf {
        let train = tmp.join("train-extended");
        std::fs::create_dir(&train).unwrap();
        write_bin_file(
            &train.join("0.bin"),
            &[
                (vec![1.0f32], vec![1.0f32]),
                (vec![2.0], vec![2.0]),
                (vec![3.0], vec![3.0]),
            ],
        )
        .unwrap();
        train
    }

    /// Screen records in the store grouped by the corpus epoch they name.
    fn screens_by_epoch(
        learnings_dir: &std::path::Path,
    ) -> std::collections::HashMap<String, Vec<String>> {
        let any = LearningsStore::new(learnings_dir, "any".into(), "t".into());
        let mut out: std::collections::HashMap<String, Vec<String>> =
            std::collections::HashMap::new();
        for s in any.load_screens().unwrap() {
            out.entry(s.corpus_identity.clone().unwrap_or_else(|| "none".into()))
                .or_default()
                .push(s.uuid);
        }
        for uuids in out.values_mut() {
            uuids.sort();
        }
        out
    }

    /// Issue #76, kept as the case it was really about: GRQ regenerating the
    /// corpus must not cost the fleet its coverage. Rewritten for #100 — what
    /// makes it the *same* corpus is identical authoritative content, which
    /// hashes to the same identity, so the second run stays in the same epoch
    /// and its coverage is cumulative.
    #[test]
    fn a_repacked_corpus_with_identical_content_keeps_its_coverage() {
        let tmp = tempfile::tempdir().unwrap();
        let (creature, train) = hidden_paths(tmp.path(), &["h_a", "h_b", "h_c", "h_d"]);
        let learnings_dir = tmp.path().join("learnings");

        let first = coverage_files_cfg(
            creature.clone(),
            train.clone(),
            tmp.path().join("out-1"),
            Some(learnings_dir.clone()),
        );
        establish_run(&first, &ScriptedScorer::ok(0.50, 0.50)).unwrap();
        assert_eq!(
            coverage_json(&first.output_dir).checked,
            2,
            "one batch of two candidates"
        );

        let repacked = repacked_corpus(tmp.path());
        assert_eq!(
            corpus_identity(&repacked),
            corpus_identity(&train),
            "identical content must be identified as the same corpus"
        );
        let second = coverage_files_cfg(
            creature,
            repacked,
            tmp.path().join("out-2"),
            Some(learnings_dir),
        );
        establish_run(&second, &ScriptedScorer::ok(0.50, 0.50)).unwrap();
        assert_eq!(
            coverage_json(&second.output_dir).checked,
            4,
            "a repack is the same epoch: both batches must be counted"
        );
    }

    /// Issue #100: extending the training corpus starts a fresh screening
    /// epoch. Coverage reports from zero, every hidden neuron — the two the
    /// previous epoch checked included — is eligible to be visited again, and
    /// the previous epoch's records are still in the store, still named.
    #[test]
    fn an_extended_corpus_starts_a_new_epoch_and_keeps_the_old_records() {
        let tmp = tempfile::tempdir().unwrap();
        let (creature, train) = hidden_paths(tmp.path(), &["h_a", "h_b", "h_c", "h_d"]);
        let learnings_dir = tmp.path().join("learnings");

        let first = coverage_files_cfg(
            creature.clone(),
            train.clone(),
            tmp.path().join("out-1"),
            Some(learnings_dir.clone()),
        );
        establish_run(&first, &ScriptedScorer::ok(0.50, 0.50)).unwrap();
        let first_cov = coverage_json(&first.output_dir);
        assert_eq!(first_cov.checked, 2, "one batch of two candidates");

        let extended = extended_corpus(tmp.path());
        assert_ne!(
            corpus_identity(&extended),
            corpus_identity(&train),
            "an extended corpus is a different corpus"
        );
        let second = coverage_files_cfg(
            creature,
            extended.clone(),
            tmp.path().join("out-2"),
            Some(learnings_dir.clone()),
        );
        let second_run = establish_run(&second, &ScriptedScorer::ok(0.50, 0.50)).unwrap();

        let second_cov = coverage_json(&second.output_dir);
        assert_eq!(
            second_cov.checked, 2,
            "the new epoch counts its own batch alone, not the old corpus's"
        );
        assert_eq!(second_cov.checkable, 4, "every hidden neuron is checkable");
        assert_eq!(
            second_run.newly_screened, 2,
            "a uuid checked under the old corpus is new coverage under the new one"
        );

        let epochs = screens_by_epoch(&learnings_dir);
        assert_eq!(epochs.len(), 2, "{epochs:?}");
        let old = &epochs[&corpus_identity(&train)];
        let new = &epochs[&corpus_identity(&extended)];
        assert_eq!(old.len(), 2, "the previous epoch's records are still there");
        assert_eq!(
            old, new,
            "the neurons the old epoch checked are eligible again: {epochs:?}"
        );

        // The artefact names the epoch its figures belong to, so `100%` can be
        // read as "100% of this corpus".
        let json =
            std::fs::read_to_string(second.output_dir.join(crate::coverage::COVERAGE_JSON_FILE))
                .unwrap();
        let report: crate::coverage::CoverageReport = serde_json::from_str(&json).unwrap();
        assert_eq!(
            report.corpus_identity.as_deref(),
            Some(corpus_identity(&extended).as_str())
        );
    }

    /// Issue #102 end to end: a sweep that finished one epoch, then a corpus
    /// change, must publish fresh partial coverage — never a carried-forward
    /// `100%` — while the previous epoch's work stays visible as history.
    #[test]
    fn a_finished_sweep_then_a_corpus_change_publishes_fresh_epoch_coverage() {
        let tmp = tempfile::tempdir().unwrap();
        let (creature, train) = hidden_paths(tmp.path(), &["h_a", "h_b", "h_c", "h_d"]);
        let learnings_dir = tmp.path().join("learnings");

        // Two batches of two candidates: the whole creature, one epoch.
        let first = OckhamConfig {
            max_experiments: Some(2),
            ..coverage_files_cfg(
                creature.clone(),
                train.clone(),
                tmp.path().join("out-1"),
                Some(learnings_dir.clone()),
            )
        };
        establish_run(&first, &ScriptedScorer::ok(0.50, 0.50)).unwrap();
        let complete =
            std::fs::read_to_string(first.output_dir.join(crate::coverage::COVERAGE_TEXT_FILE))
                .unwrap();
        assert!(
            complete.contains("sweep:     4 of 4 hidden (100.0% of epoch)"),
            "{complete}"
        );
        assert!(
            complete.contains("unchecked: 0 remaining — sweep complete for this epoch"),
            "{complete}"
        );

        let extended = extended_corpus(tmp.path());
        let second = coverage_files_cfg(
            creature,
            extended.clone(),
            tmp.path().join("out-2"),
            Some(learnings_dir),
        );
        establish_run(&second, &ScriptedScorer::ok(0.50, 0.50)).unwrap();
        let fresh =
            std::fs::read_to_string(second.output_dir.join(crate::coverage::COVERAGE_TEXT_FILE))
                .unwrap();
        assert!(
            fresh.contains("sweep:     2 of 4 hidden (50.0% of epoch)"),
            "the new epoch reports its own coverage: {fresh}"
        );
        assert!(
            !fresh.contains("100.0%") && !fresh.contains("sweep complete"),
            "a corpus change cannot leave a misleading 100%: {fresh}"
        );
        assert!(
            fresh.contains(&format!(
                "epoch:     corpus {}",
                crate::coverage::short_epoch(&corpus_identity(&extended))
            )),
            "{fresh}"
        );
        assert!(
            fresh.contains("history:   4 of 4 ever checked across 2 corpus epochs"),
            "the previous epoch's work stays available, beside the percentage \
             rather than inside it: {fresh}"
        );

        let report = coverage_report_json(&second.output_dir);
        assert_eq!(report.coverage.percent(), 50.0);
        assert_eq!(
            report.history.expect("cumulative figures").checked_ever,
            4,
            "history is cumulative across epochs"
        );
    }

    /// The neurons a previous epoch could propose nothing for — `blocked`, and
    /// a standing full-corpus failure — are unchecked again under a new corpus,
    /// so a new epoch never opens already claiming coverage it has not measured.
    #[test]
    fn a_blocked_or_failed_neuron_is_eligible_again_in_the_new_epoch() {
        let tmp = tempfile::tempdir().unwrap();
        let (_, train) = hidden_paths(tmp.path(), &["h_a", "h_b"]);
        let learnings_dir = tmp.path().join("learnings");
        let old = screens_store(&learnings_dir, &train);
        for (uuid, kind) in [
            ("h_a", crate::learnings::SCREEN_KIND_SKIPPED),
            ("h_b", crate::learnings::SCREEN_KIND_KNOWN_FAILURE),
        ] {
            old.append_screen(&Screened {
                blocked_reason: Default::default(),
                version: crate::learnings::SCREENS_VISIT_FORMAT_VERSION,
                uuid: uuid.into(),
                kind: kind.into(),
                outcome: ScreenOutcomeKind::Loser,
                unix_secs: 10,
                host: "t".into(),
                corpus_identity: Some(old.corpus_identity().to_string()),
            })
            .unwrap();
        }
        let creature = hidden_creature(&["h_a", "h_b"]);
        let history = old.load_screens().unwrap();
        let before = crate::coverage::coverage(&creature, &HashSet::new(), &history, 0);
        assert_eq!(before.checked, 2, "both were visited under the old corpus");
        assert_eq!(before.blocked, 1);

        let extended = extended_corpus(tmp.path());
        let new = screens_store(&learnings_dir, &extended);
        let epoch = crate::learnings::current_epoch_screens(
            new.load_screens().unwrap(),
            new.corpus_identity(),
        );
        let after = crate::coverage::coverage(&creature, &HashSet::new(), &epoch, 0);
        assert_eq!(after.checked, 0, "neither is checked under the new corpus");
        assert_eq!(after.blocked, 0);
        assert_eq!(
            after.unchecked(),
            2,
            "both are eligible to be visited again"
        );
        assert_eq!(
            new.load_screens().unwrap().len(),
            2,
            "and both records are still readable"
        );
    }

    fn corpus_identity(train: &std::path::Path) -> String {
        crate::corpus::corpus_info(train, &TrainingDataConfig::new(1, 1))
            .unwrap()
            .identity
    }

    fn coverage_json(output_dir: &std::path::Path) -> Coverage {
        let json =
            std::fs::read_to_string(output_dir.join(crate::coverage::COVERAGE_JSON_FILE)).unwrap();
        serde_json::from_str(&json).unwrap()
    }

    /// The whole artefact, epoch and cumulative figures included (Issue #102).
    fn coverage_report_json(output_dir: &std::path::Path) -> crate::coverage::CoverageReport {
        let json =
            std::fs::read_to_string(output_dir.join(crate::coverage::COVERAGE_JSON_FILE)).unwrap();
        serde_json::from_str(&json).unwrap()
    }

    /// Config for the coverage-artefact tests: one batch over four neurons.
    fn coverage_files_cfg(
        creature: std::path::PathBuf,
        train: std::path::PathBuf,
        out: std::path::PathBuf,
        learnings_dir: Option<std::path::PathBuf>,
    ) -> OckhamConfig {
        OckhamConfig {
            creature,
            training_data: train,
            output_dir: out,
            timeout: Duration::from_secs(30),
            max_experiments: Some(1),
            seed: Some(1),
            candidates: 2,
            learnings_dir,
            learnings_host: Some("t".into()),
            ..OckhamConfig::default()
        }
    }

    #[test]
    fn the_run_writes_the_coverage_description_and_json_beside_best_json() {
        let tmp = tempfile::tempdir().unwrap();
        let (creature, train) = hidden_paths(tmp.path(), &["h_a", "h_b", "h_c", "h_d"]);
        let cfg = coverage_files_cfg(
            creature,
            train,
            tmp.path().join("out"),
            Some(tmp.path().join("learnings")),
        );
        establish_run(&cfg, &ScriptedScorer::ok(0.50, 0.50)).unwrap();

        let text =
            std::fs::read_to_string(cfg.output_dir.join(crate::coverage::COVERAGE_TEXT_FILE))
                .unwrap();
        assert!(cfg.output_dir.join("best.json").exists());
        let json =
            std::fs::read_to_string(cfg.output_dir.join(crate::coverage::COVERAGE_JSON_FILE))
                .unwrap();
        let cov: Coverage = serde_json::from_str(&json).unwrap();
        assert_eq!(cov.hidden, 4);
        assert_eq!(cov.checked, 2, "one batch of two candidates");
        assert_eq!(cov.checkable, 4);
        let report: crate::coverage::CoverageReport = serde_json::from_str(&json).unwrap();
        assert_eq!(report.newly_screened, 2, "the run's own progress (#77)");
        assert_eq!(
            text,
            format!("{}\n", report.description(cfg.candidates)),
            "the text block must render the JSON figures"
        );
        assert!(
            text.starts_with("🪒 Ockham neuron screening coverage\n"),
            "{text}"
        );
        assert!(
            text.contains("unchecked: 2 remaining this epoch (~1 run at 2/run)"),
            "{text}"
        );
        // Issue #102: the run names the epoch its percentage belongs to, in
        // both artefacts — compactly in the prose, in full in the JSON.
        let identity = report.corpus_identity.expect("the run names its epoch");
        assert_eq!(identity.len(), 16, "the JSON keeps the full identity");
        assert!(
            text.contains(&format!(
                "\nepoch:     corpus {} — coverage counts this corpus only\n",
                crate::coverage::short_epoch(&identity)
            )),
            "{text}"
        );
        assert!(text.contains("(50.0% of epoch)"), "{text}");
    }

    /// End-to-end detector for Issue #74: a fully tagged creature must report
    /// its tagged neurons *inside* the denominator and screened records against
    /// them as checked — the divergence that halves the fleet's percentage if
    /// only one half of the change lands.
    #[test]
    fn a_run_over_a_fully_tagged_creature_counts_every_hidden_neuron() {
        let tmp = tempfile::tempdir().unwrap();
        let uuids = ["h_a", "h_b", "h_c", "h_d"];
        let (creature, train) = hidden_paths(tmp.path(), &uuids);
        tag_neurons(&creature, &uuids);
        let cfg = coverage_files_cfg(
            creature,
            train,
            tmp.path().join("out"),
            Some(tmp.path().join("learnings")),
        );
        establish_run(&cfg, &ScriptedScorer::ok(0.50, 0.50)).unwrap();

        let json =
            std::fs::read_to_string(cfg.output_dir.join(crate::coverage::COVERAGE_JSON_FILE))
                .unwrap();
        let cov: Coverage = serde_json::from_str(&json).unwrap();
        assert_eq!(cov.hidden, 4);
        assert_eq!(cov.tagged, 4, "every hidden neuron carries a tag");
        assert_eq!(cov.checkable, 4, "tagged neurons stay in the denominator");
        assert_eq!(cov.checked, 2, "screened tagged UUIDs count as checked");
        assert_eq!(cov.percent(), 50.0);

        let text =
            std::fs::read_to_string(cfg.output_dir.join(crate::coverage::COVERAGE_TEXT_FILE))
                .unwrap();
        assert!(
            text.contains("tagged:    4 carry tags, screened like any other"),
            "{text}"
        );
        // The old `skipped:` coverage line, not the `bundles: … skipped`
        // clause, which is a legitimate winner figure.
        assert!(!text.contains("skipped:"), "{text}");
    }

    /// Creature the razor can only prune one third of.
    ///
    /// `h_agg` is a `MEAN` aggregate, so it can never be ablated, and `h_fed`
    /// feeds it, so it cannot be ablated either. Only `h_cut` is proposable —
    /// the shape of the production creature, where forests put an aggregate
    /// squash downstream of most hidden neurons (Issue #93).
    fn aggregate_blocked_paths(tmp: &std::path::Path) -> (std::path::PathBuf, std::path::PathBuf) {
        use crate::fixtures::{creature, neuron, synapse};
        let c = creature(
            1,
            1,
            vec![
                neuron("hidden", "h_cut", 0.0, Some("IDENTITY")),
                neuron("hidden", "h_fed", 0.0, Some("TANH")),
                neuron("hidden", "h_agg", 0.0, Some("MEAN")),
                neuron("output", "output-0", 0.0, Some("IDENTITY")),
            ],
            vec![
                synapse("input-0", "h_cut", 1.0),
                synapse("input-0", "h_fed", 1.0),
                synapse("input-0", "h_agg", 1.0),
                synapse("h_cut", "output-0", 1.0),
                synapse("h_fed", "h_agg", 1.0),
                synapse("h_agg", "output-0", 1.0),
            ],
        );
        let path = tmp.join("creature.json");
        std::fs::write(&path, neat_core::creature_to_json_pretty(&c).unwrap()).unwrap();
        let train = tmp.join("train");
        std::fs::create_dir(&train).unwrap();
        write_bin_file(
            &train.join("0.bin"),
            &[(vec![1.0f32], vec![1.0f32]), (vec![2.0], vec![2.0])],
        )
        .unwrap();
        (path, train)
    }

    /// Issue #93: the counter went backwards because a visit the razor could
    /// propose nothing for filed no coverage. `checked` was pinned to the
    /// prunable minority and fell by one on every accepted cut, so the fleet
    /// reported `1417/6969` one run and `1416/7005` the next while the sweep
    /// walked the same neurons over and over.
    ///
    /// The invariant that fixed it — every visit is coverage — is asserted here
    /// on the visit a standing verdict suppresses
    /// (`a_known_failure_skip_is_checked_without_being_called_unprunable`) and
    /// on a blocked visit in `coverage::tests`. What changed in #103 is that
    /// this creature's aggregate structure is no longer blocked at all, so the
    /// run-level assertion below is that it is *proposed*.
    #[test]
    fn every_hidden_neuron_of_an_aggregate_creature_is_checked() {
        let tmp = tempfile::tempdir().unwrap();
        let (creature, train) = aggregate_blocked_paths(tmp.path());
        let learnings_dir = tmp.path().join("learnings");
        let cfg = OckhamConfig {
            creature,
            training_data: train.clone(),
            output_dir: tmp.path().join("out"),
            timeout: Duration::from_secs(30),
            max_experiments: Some(1),
            seed: Some(1),
            candidates: 8,
            learnings_dir: Some(learnings_dir.clone()),
            learnings_host: Some("t".into()),
            ..OckhamConfig::default()
        };
        // Flat scorer: nothing is accepted, so the run's own coverage is the
        // only thing that moves.
        establish_run(&cfg, &ScriptedScorer::ok(0.50, 0.50)).unwrap();

        let store = screens_store(&learnings_dir, &train);
        assert_eq!(
            screened_uuids(&store),
            vec!["h_agg", "h_cut", "h_fed"],
            "one batch visited every hidden neuron, so every one is checked"
        );
        let cov = coverage_json(&cfg.output_dir);
        assert_eq!(cov.hidden, 3);
        assert_eq!(cov.checked, 3, "coverage must count the visits");
        assert_eq!(cov.percent(), 100.0);
        let text =
            std::fs::read_to_string(cfg.output_dir.join(crate::coverage::COVERAGE_TEXT_FILE))
                .unwrap();
        assert!(
            text.contains("progress:  3 newly checked this run"),
            "{text}"
        );

        // A second run re-visits the same neurons and files nothing new for the
        // ones it cannot advance: the record already says the sweep has been
        // there, and repeating it every pass would grow the fleet's shared log
        // without adding a fact.
        let again = OckhamConfig {
            output_dir: tmp.path().join("out-2"),
            ..cfg.clone()
        };
        establish_run(&again, &ScriptedScorer::ok(0.50, 0.50)).unwrap();
        let second = coverage_json(&again.output_dir);
        assert_eq!(second.checked, 3, "coverage held, it did not go backwards");
    }

    /// Issue #103: the category that blocked most of a forest-heavy creature is
    /// testable now. Every neuron of the fixture that used to file a `skipped`
    /// visit is proposed as a constant substitution, screened like any other
    /// candidate, and nothing is left blocked.
    #[test]
    fn aggregate_structure_that_was_blocked_is_proposed_as_a_constant() {
        let tmp = tempfile::tempdir().unwrap();
        let (creature, train) = aggregate_blocked_paths(tmp.path());
        let learnings_dir = tmp.path().join("learnings");
        let cfg = OckhamConfig {
            creature,
            training_data: train.clone(),
            output_dir: tmp.path().join("out"),
            timeout: Duration::from_secs(30),
            max_experiments: Some(1),
            seed: Some(1),
            candidates: 8,
            learnings_dir: Some(learnings_dir.clone()),
            learnings_host: Some("t".into()),
            ..OckhamConfig::default()
        };
        establish_run(&cfg, &ScriptedScorer::ok(0.50, 0.50)).unwrap();

        let store = screens_store(&learnings_dir, &train);
        let kinds: std::collections::HashMap<String, String> = store
            .load_screens()
            .unwrap()
            .into_iter()
            .map(|s| (s.uuid, s.kind))
            .collect();
        assert_eq!(kinds["h_cut"], "identity", "the exact path is unchanged");
        assert_eq!(
            kinds["h_fed"], "constant",
            "a neuron feeding an aggregate is proposable now, not blocked"
        );
        assert_eq!(
            kinds["h_agg"], "constant",
            "the aggregate neuron itself is proposable now, not blocked"
        );

        let cov = coverage_json(&cfg.output_dir);
        assert_eq!(cov.checked, 3);
        assert_eq!(cov.blocked, 0, "nothing is blocked on this creature now");
        assert_eq!(cov.blocked_by_reason.total(), 0);
        let text =
            std::fs::read_to_string(cfg.output_dir.join(crate::coverage::COVERAGE_TEXT_FILE))
                .unwrap();
        assert!(
            !text.contains("blocked:") && !text.contains("reasons:"),
            "nothing blocked renders neither line: {text}"
        );
    }

    /// A standing full-corpus verdict is the strongest check there is, so the
    /// visit it suppresses is filed as checked but never as blocked (#93).
    #[test]
    fn a_known_failure_skip_is_checked_without_being_called_unprunable() {
        let tmp = tempfile::tempdir().unwrap();
        let (creature, train) = two_hidden_paths(tmp.path());
        let learnings_dir = tmp.path().join("learnings");
        let corpus = crate::corpus::corpus_info(&train, &TrainingDataConfig::new(1, 1)).unwrap();
        let store = LearningsStore::new(&learnings_dir, corpus.identity, "t".into());
        store
            .append(&crate::learnings::Learning {
                version: crate::learnings::LEARNINGS_FORMAT_VERSION,
                uuid: "h_a".into(),
                kind: "identity".into(),
                outcome: Outcome::Rejected,
                unix_secs: crate::incumbent::now_unix(),
                host: "t".into(),
                full_delta: Some(-1.0),
                group: None,
            })
            .unwrap();
        let cfg = OckhamConfig {
            creature,
            training_data: train.clone(),
            output_dir: tmp.path().join("out"),
            timeout: Duration::from_secs(30),
            max_experiments: Some(1),
            seed: Some(1),
            candidates: 8,
            learnings_dir: Some(learnings_dir.clone()),
            learnings_host: Some("t".into()),
            ..OckhamConfig::default()
        };
        establish_run(&cfg, &ScriptedScorer::ok(0.50, 0.50)).unwrap();

        let screens = screens_store(&learnings_dir, &train)
            .load_screens()
            .unwrap();
        let filed: Vec<(&str, &str)> = screens
            .iter()
            .map(|s| (s.uuid.as_str(), s.kind.as_str()))
            .collect();
        assert!(
            filed.contains(&("h_a", crate::learnings::SCREEN_KIND_KNOWN_FAILURE)),
            "the suppressed visit must still be coverage: {filed:?}"
        );
        let cov = coverage_json(&cfg.output_dir);
        assert_eq!(cov.checked, 2, "both neurons were reached");
        assert_eq!(
            cov.blocked, 0,
            "a fully scored uuid is checked, not structurally unprunable"
        );

        // A visit that scored nothing files **one** record however many passes
        // see it (#93): the record already says the sweep has been there, and
        // repeating it every run would grow the fleet's shared append-only log
        // without adding a fact.
        let again = OckhamConfig {
            output_dir: tmp.path().join("out-2"),
            ..cfg.clone()
        };
        establish_run(&again, &ScriptedScorer::ok(0.50, 0.50)).unwrap();
        let screens = screens_store(&learnings_dir, &train)
            .load_screens()
            .unwrap();
        assert_eq!(
            screens.iter().filter(|s| s.uuid == "h_a").count(),
            1,
            "one visit record per suppressed uuid: {screens:?}"
        );
        assert_eq!(
            coverage_json(&again.output_dir).checked,
            2,
            "coverage held, it did not go backwards"
        );
    }

    #[test]
    fn a_run_without_a_learnings_dir_writes_no_coverage_files() {
        let tmp = tempfile::tempdir().unwrap();
        let (creature, train) = hidden_paths(tmp.path(), &["h_a", "h_b", "h_c", "h_d"]);
        let cfg = coverage_files_cfg(creature, train, tmp.path().join("out"), None);
        establish_run(&cfg, &ScriptedScorer::ok(0.50, 0.50)).unwrap();
        assert!(
            !cfg.output_dir
                .join(crate::coverage::COVERAGE_TEXT_FILE)
                .exists(),
            "no screen store means no coverage state to publish"
        );
        assert!(
            !cfg.output_dir
                .join(crate::coverage::COVERAGE_JSON_FILE)
                .exists()
        );
    }

    #[test]
    fn a_blocked_coverage_write_warns_rather_than_failing_the_run() {
        let tmp = tempfile::tempdir().unwrap();
        let (creature, train) = hidden_paths(tmp.path(), &["h_a", "h_b", "h_c", "h_d"]);
        let out = tmp.path().join("out");
        // A directory where coverage.txt belongs: the artefact cannot be
        // written, and the run must still complete.
        std::fs::create_dir_all(out.join(crate::coverage::COVERAGE_TEXT_FILE)).unwrap();
        let cfg = coverage_files_cfg(creature, train, out, Some(tmp.path().join("learnings")));
        let run = establish_run(&cfg, &ScriptedScorer::ok(0.50, 0.50)).unwrap();
        assert_eq!(run.optimisation, "complete");
        assert!(
            !cfg.output_dir
                .join(crate::coverage::COVERAGE_JSON_FILE)
                .exists(),
            "the failed text write stops before the json, and neither fails the run"
        );
        let journal = std::fs::read_to_string(cfg.output_dir.join("experiments.jsonl")).unwrap();
        assert!(
            journal.contains(r#""record":"coverage""#),
            "coverage is still journalled: {journal}"
        );
    }

    /// Issue #87: tags are informational only, so a run that cuts a tagged
    /// neuron declares nothing about it — no declaration artefact is written,
    /// and no rendered artefact mentions one.
    #[test]
    fn a_run_that_cut_a_tagged_neuron_writes_no_declaration_artefact() {
        let tmp = tempfile::tempdir().unwrap();
        let (creature, train) = two_hidden_paths(tmp.path());
        tag_neurons(&creature, &["h_a"]);
        let learnings_dir = tmp.path().join("learnings");
        known_wins(&learnings_dir, &train, &["h_a", "h_b"]);
        let cfg = replay_cfg(creature, train, tmp.path().join("out"), learnings_dir, None);

        let run = establish_run(&cfg, &improving_scorer()).unwrap();
        assert_eq!(run.stop_reason, "replay-accepts");
        let best = std::fs::read_to_string(cfg.output_dir.join("best.json")).unwrap();
        assert!(
            !best.contains("h_a") && !best.contains("h_b"),
            "the tagged neuron is an ordinary prune candidate: {best}"
        );
        assert!(
            !cfg.output_dir.join("pruned-provenance.json").exists(),
            "a declaration artefact must never be written again"
        );
        let text =
            std::fs::read_to_string(cfg.output_dir.join(crate::coverage::COVERAGE_TEXT_FILE))
                .unwrap();
        assert!(!text.contains("declared:"), "{text}");
        let json =
            std::fs::read_to_string(cfg.output_dir.join(crate::coverage::COVERAGE_JSON_FILE))
                .unwrap();
        assert!(!json.contains("taggedCut"), "{json}");
        let journal = std::fs::read_to_string(cfg.output_dir.join("experiments.jsonl")).unwrap();
        assert!(!journal.contains("tagged_cut"), "{journal}");
    }

    /// File one accepted learning per uuid, so replay cuts them as one bundle.
    fn known_wins(learnings_dir: &std::path::Path, train: &std::path::Path, uuids: &[&str]) {
        let store = screens_store(learnings_dir, train);
        let records: Vec<_> = uuids
            .iter()
            .enumerate()
            .map(|(i, uuid)| (*uuid, Outcome::Accepted, None, 10 + i as u64))
            .collect();
        seed_verdicts(&store, &records);
    }

    /// A scorer that prefers every candidate to the incumbent.
    fn improving_scorer() -> ScriptedScorer {
        ScriptedScorer {
            baseline_score: 0.50,
            candidate_score: Some(0.80),
            ..ScriptedScorer::ok(0.50, 0.50)
        }
    }

    /// Count the hidden neurons of a written `best.json`.
    fn hidden_neurons(best: &std::path::Path) -> usize {
        let v: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(best).unwrap()).unwrap();
        v["neurons"]
            .as_array()
            .expect("neurons")
            .iter()
            .filter(|n| n["type"] == "hidden")
            .count()
    }

    /// Extract the `ockham` creature tag from a written `best.json`.
    fn ockham_tag(best: &std::path::Path) -> String {
        let v: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(best).unwrap()).unwrap();
        v["tags"]
            .as_array()
            .expect("tags")
            .iter()
            .find(|t| t["name"] == "ockham")
            .expect("ockham tag")["value"]
            .as_str()
            .unwrap()
            .to_string()
    }

    #[test]
    fn an_accept_stamps_coverage_into_the_ockham_tag() {
        let tmp = tempfile::tempdir().unwrap();
        let (creature, train) = hidden_paths(tmp.path(), &["h_a", "h_b", "h_c", "h_d"]);
        let cfg = OckhamConfig {
            creature,
            training_data: train,
            output_dir: tmp.path().join("out"),
            timeout: Duration::from_secs(30),
            max_experiments: Some(8),
            seed: Some(1),
            candidates: 2,
            screen_sample_rate: None,
            learnings_dir: Some(tmp.path().join("learnings")),
            learnings_host: Some("t".into()),
            ..OckhamConfig::default()
        };
        let scorer = ScriptedScorer {
            baseline_score: 0.50,
            candidate_score: Some(0.80),
            ..ScriptedScorer::ok(0.50, 0.50)
        };
        let run = establish_run(&cfg, &scorer).unwrap();
        assert!(run.accepts >= 1, "accepts={}", run.accepts);
        let best = cfg.output_dir.join("best.json");
        let tag = ockham_tag(&best);
        assert!(tag.starts_with("🪒 Ockham"), "{tag}");
        assert!(tag.contains(" · sweep "), "{tag}");
        assert!(
            tag.contains("of epoch "),
            "the subject must scope its percentage to the epoch (#102): {tag}"
        );
        assert!(
            tag.contains(&format!("/{} (", hidden_neurons(&best))),
            "denominator must be every hidden neuron, tagged included (#74): {tag}"
        );
    }

    #[test]
    fn an_accept_without_a_learnings_dir_leaves_the_tag_coverage_free() {
        let tmp = tempfile::tempdir().unwrap();
        let (creature, train) = hidden_paths(tmp.path(), &["h_a", "h_b", "h_c", "h_d"]);
        let cfg = OckhamConfig {
            creature,
            training_data: train,
            output_dir: tmp.path().join("out"),
            timeout: Duration::from_secs(30),
            max_experiments: Some(8),
            seed: Some(1),
            candidates: 2,
            screen_sample_rate: None,
            ..OckhamConfig::default()
        };
        let scorer = ScriptedScorer {
            baseline_score: 0.50,
            candidate_score: Some(0.80),
            ..ScriptedScorer::ok(0.50, 0.50)
        };
        let run = establish_run(&cfg, &scorer).unwrap();
        assert!(run.accepts >= 1, "accepts={}", run.accepts);
        let tag = ockham_tag(&cfg.output_dir.join("best.json"));
        assert!(tag.starts_with("🪒 Ockham"), "{tag}");
        assert!(
            !tag.contains("sweep ") && !tag.contains("epoch"),
            "no screen store means no coverage clause, not 0/0: {tag}"
        );
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
                    .is_some_and(|n| n.starts_with("screens") || n == "learnings")
            })
            .collect();
        assert!(stray.is_empty(), "{stray:?}");
        assert!(!cfg.output_dir.join("learnings").exists());
    }

    // ---- Issue #54: the cap stops gating bundle construction ---------------

    #[test]
    fn the_full_scoring_line_no_longer_conflates_individuals_with_bundles() {
        assert_eq!(
            full_scoring_line(Some(8), 38, 0),
            "full-scoring 8 of 38 sampled winners individually; bundling all 38"
        );
        assert_eq!(
            full_scoring_line(None, 38, 0),
            "full-scoring 38 sampled winners plus bundles"
        );
        assert_eq!(
            full_scoring_line(None, 38, 21),
            "full-scoring 38 sampled winners individually; bundling all 59 (21 carried)"
        );
        // A cap larger than the winner list caps nothing.
        assert_eq!(
            full_scoring_line(Some(99), 38, 0),
            "full-scoring 38 sampled winners plus bundles"
        );
        for line in [
            full_scoring_line(Some(8), 38, 0),
            full_scoring_line(None, 38, 21),
        ] {
            assert!(!line.contains("keeping top"), "{line}");
        }
    }

    // ---- Issue #56: confirmed winners are carried between batches ----------

    fn member(uuid: &str, delta: f64) -> BundleMember {
        BundleMember {
            uuid: uuid.into(),
            kind: CandidateKind::Identity,
            delta,
        }
    }

    fn stats_of(creature: &CreatureExport) -> ActivationStats {
        crate::stats::ActivationStats {
            format_version: crate::stats::STATS_FORMAT_VERSION,
            creature_checksum: "t".into(),
            corpus_identity: "c".into(),
            record_count: 1,
            corpus_record_count: 1,
            sample: crate::stats::SampleSpec::full(),
            stopped_early: false,
            scan_ms: 0,
            from_cache: false,
            neurons: creature
                .neurons
                .iter()
                .enumerate()
                .filter(|(_, n)| n.neuron_type == "hidden")
                .map(|(i, n)| crate::stats::NeuronStats {
                    uuid: n.uuid.clone(),
                    neuron_index: i,
                    count: 1,
                    mean: 0.0,
                    variance: 0.0,
                    std_dev: 0.0,
                    mean_abs: 0.0,
                    min: 0.0,
                    max: 0.0,
                })
                .collect(),
        }
    }

    #[test]
    fn a_pool_member_the_incumbent_no_longer_carries_is_dropped() {
        let creature = hidden_creature(&["h_a", "h_b"]);
        let stats = stats_of(&creature);
        let pool = vec![
            member("h_a", 3e-6),
            member("gone", 9e-6),
            member("h_b", 1e-6),
        ];
        let standing = standing_pool(&pool, &creature, &stats);
        assert_eq!(
            standing.iter().map(|m| m.uuid.as_str()).collect::<Vec<_>>(),
            vec!["h_a", "h_b"],
            "a stale cut leaves rather than voiding every plan it joins"
        );
        assert!(standing_pool(&[], &creature, &stats).is_empty());
    }

    /// Build a `FullOutcome` whose individuals carry the given deltas.
    fn outcome_with(individuals: &[(&str, f64)], winner: Option<&str>) -> FullOutcome {
        let candidate = |uuid: &str, delta: f64| crate::promote::FullCandidate {
            stem: uuid.into(),
            kind: "individual",
            uuids: vec![uuid.into()],
            score: 0.5 + delta,
            error: 0.5,
            complexity_penalty: 0.0,
            after: crate::ablation::StructureSnapshot::of(&hidden_creature(&["h_a"])),
            delta,
        };
        FullOutcome {
            incumbent_score: 0.5,
            incumbent_error: 0.5,
            individuals: individuals
                .iter()
                .map(|(uuid, delta)| candidate(uuid, *delta))
                .collect(),
            bundles: Vec::new(),
            groups: Vec::new(),
            sample_false_positives: Vec::new(),
            winner: winner.map(|uuid| LocalWinner {
                candidate: candidate(uuid, 0.3),
                checksum: "c".into(),
                creature: hidden_creature(&["h_a"]),
            }),
            full_ms: 10,
            skipped_bundles: 0,
            dropped_individuals: 0,
            dropped_bundles: 0,
            capped_plans: 0,
        }
    }

    #[test]
    fn the_pool_keeps_confirmed_winners_and_forgets_the_rest() {
        let mut pool = Vec::new();
        update_pool(
            &mut pool,
            &[],
            &outcome_with(&[("h_a", 3e-6), ("h_b", 9e-6), ("h_c", -1.0)], Some("h_a")),
            1e-6,
        );
        assert_eq!(
            pool.iter().map(|m| m.uuid.as_str()).collect::<Vec<_>>(),
            vec!["h_b"],
            "the applied cut and the loser both leave; the confirmed one stays"
        );

        // A later cohort measures h_b as a loser: the latest verdict wins and
        // the old delta must not resurrect it.
        update_pool(&mut pool, &[], &outcome_with(&[("h_b", -1.0)], None), 1e-6);
        assert!(pool.is_empty(), "{pool:?}");
    }

    #[test]
    fn the_pool_holds_one_entry_per_uuid_and_drops_the_weakest_on_overflow() {
        let mut pool = Vec::new();
        let many: Vec<(String, f64)> = (0..MAX_CONFIRMED_POOL + 4)
            .map(|i| (format!("h{i:03}"), 1e-3 + i as f64 * 1e-4))
            .collect();
        let refs: Vec<(&str, f64)> = many.iter().map(|(u, d)| (u.as_str(), *d)).collect();
        update_pool(&mut pool, &[], &outcome_with(&refs, None), 1e-6);
        assert_eq!(pool.len(), MAX_CONFIRMED_POOL);
        assert!(
            pool.iter().all(|m| m.uuid != "h000"),
            "the weakest members are the ones dropped"
        );

        // Re-filing the same uuids must not duplicate them.
        update_pool(&mut pool, &[], &outcome_with(&refs[..4], None), 1e-6);
        let unique: HashSet<&str> = pool.iter().map(|m| m.uuid.as_str()).collect();
        assert_eq!(unique.len(), pool.len());
    }

    /// Hidden neurons in a written cohort creature.
    fn hidden_in(path: &std::path::Path) -> usize {
        let v: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
        v["neurons"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|n| n["type"] == "hidden")
            .count()
    }

    /// Cohort file stems of one kind (`i` or `b`) written under `dir`.
    fn cohort_stems(dir: &std::path::Path, prefix: char) -> Vec<String> {
        let mut out: Vec<String> = std::fs::read_dir(dir)
            .unwrap()
            .filter_map(|e| e.ok())
            .filter_map(|e| {
                e.path()
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .map(str::to_string)
            })
            .filter(|s| s.starts_with(prefix))
            .collect();
        out.sort();
        out
    }

    /// Two batches over parallel hidden neurons, one accept per batch.
    fn carried_winner_run(tmp: &std::path::Path) -> (OckhamConfig, BaselineRun) {
        let (creature, train) = hidden_paths(tmp, &["h_a", "h_b", "h_c", "h_d", "h_e", "h_f"]);
        let cfg = OckhamConfig {
            creature,
            training_data: train,
            output_dir: tmp.join("out"),
            timeout: Duration::from_secs(30),
            max_experiments: Some(2),
            seed: Some(1),
            candidates: 2,
            screen_sample_rate: None,
            ..OckhamConfig::default()
        };
        let scorer = ScriptedScorer {
            baseline_score: 0.50,
            candidate_score: Some(0.80),
            ..ScriptedScorer::ok(0.50, 0.50)
        };
        let run = establish_run(&cfg, &scorer).unwrap();
        (cfg, run)
    }

    #[test]
    fn a_confirmed_winner_from_batch_one_joins_batch_twos_bundles() {
        let tmp = tempfile::tempdir().unwrap();
        let (cfg, run) = carried_winner_run(tmp.path());
        assert!(run.accepts >= 1, "accepts={}", run.accepts);

        let second = run.workspace.join("full-2");
        let baseline = hidden_in(&second.join("baseline.json"));
        let widest = cohort_stems(&second, 'b')
            .iter()
            .map(|stem| baseline - hidden_in(&second.join(format!("{stem}.json"))))
            .max()
            .expect("batch two must build at least one bundle");
        assert!(
            widest > cfg.candidates,
            "a bundle wider than the batch can only come from the carried pool: \
             {widest} cuts from {} candidates",
            cfg.candidates
        );
    }

    #[test]
    fn carried_winners_are_never_re_scored_as_individuals() {
        let tmp = tempfile::tempdir().unwrap();
        let (cfg, _) = carried_winner_run(tmp.path());
        let workspace = cfg.output_dir.join("workspace");
        for batch in ["full-1", "full-2"] {
            let dir = workspace.join(batch);
            assert!(
                cohort_stems(&dir, 'i').len() <= cfg.candidates,
                "the individual cohort must stay this batch's fresh winners: {:?}",
                cohort_stems(&dir, 'i')
            );
        }
    }

    #[test]
    fn the_pool_starts_empty_so_the_first_batch_bundles_only_its_own_winners() {
        let tmp = tempfile::tempdir().unwrap();
        let (cfg, run) = carried_winner_run(tmp.path());
        let first = run.workspace.join("full-1");
        let baseline = hidden_in(&first.join("baseline.json"));
        for stem in cohort_stems(&first, 'b') {
            let cuts = baseline - hidden_in(&first.join(format!("{stem}.json")));
            assert!(
                cuts <= cfg.candidates,
                "cross-run memory is the learnings cache's job, not the pool's: {cuts}"
            );
        }
    }

    // ---- Issue #57: replay seeks the creature carrying the most winners ----

    /// Seed one learnings record per `(uuid, outcome, full_delta)`.
    fn seed_verdicts(store: &LearningsStore, records: &[(&str, Outcome, Option<f64>, u64)]) {
        for (uuid, outcome, full_delta, unix_secs) in records {
            store
                .append(&crate::learnings::Learning {
                    version: crate::learnings::LEARNINGS_FORMAT_VERSION,
                    uuid: (*uuid).into(),
                    kind: "identity".into(),
                    outcome: *outcome,
                    unix_secs: *unix_secs,
                    host: "t".into(),
                    full_delta: *full_delta,
                    group: None,
                })
                .unwrap();
        }
    }

    /// Config for the replay tests: two hidden neurons, a seeded cache.
    fn replay_cfg(
        creature: std::path::PathBuf,
        train: std::path::PathBuf,
        out: std::path::PathBuf,
        learnings_dir: std::path::PathBuf,
        max_full: Option<usize>,
    ) -> OckhamConfig {
        OckhamConfig {
            creature,
            training_data: train,
            output_dir: out,
            timeout: Duration::from_secs(30),
            max_experiments: Some(8),
            seed: Some(1),
            candidates: 8,
            screen_sample_rate: None,
            learnings_dir: Some(learnings_dir),
            learnings_host: Some("t".into()),
            max_full,
            ..OckhamConfig::default()
        }
    }

    #[test]
    fn a_confirmed_but_unapplied_cut_is_replayed() {
        let tmp = tempfile::tempdir().unwrap();
        let (creature, train) = two_hidden_paths(tmp.path());
        let learnings_dir = tmp.path().join("learnings");
        let store = screens_store(&learnings_dir, &train);
        // Neither was ever applied: both lost their cohort to a better
        // candidate, and before Issue #52 both were poison for seven days.
        seed_verdicts(
            &store,
            &[
                ("h_a", Outcome::Rejected, Some(2e-6), 10),
                ("h_b", Outcome::Rejected, Some(9e-6), 20),
            ],
        );
        let cfg = replay_cfg(creature, train, tmp.path().join("out"), learnings_dir, None);
        let scorer = ScriptedScorer {
            baseline_score: 0.50,
            candidate_score: Some(0.80),
            ..ScriptedScorer::ok(0.50, 0.50)
        };
        let run = establish_run(&cfg, &scorer).unwrap();
        assert_eq!(run.stop_reason, "replay-accepts");
        let best = std::fs::read_to_string(cfg.output_dir.join("best.json")).unwrap();
        assert!(best.contains("replay-bundle"), "{best}");
        assert!(!best.contains("h_a") && !best.contains("h_b"), "{best}");
    }

    /// Issue #101: a cut an older corpus epoch confirmed is replayed against the
    /// corpus in hand as a hypothesis, and the current scorer accepts it. The
    /// verdict filed is this corpus's own — the acceptance is a current-corpus
    /// result, never the old epoch's verdict carried forward.
    #[test]
    fn a_historical_winner_is_replayed_and_the_current_corpus_accepts_it() {
        let tmp = tempfile::tempdir().unwrap();
        let (creature, train) = two_hidden_paths(tmp.path());
        let learnings_dir = tmp.path().join("learnings");
        seed_prior_corpus(&learnings_dir, "h_a", Outcome::Accepted, 10);
        let store = screens_store(&learnings_dir, &train);
        assert!(
            store.load().unwrap().is_empty(),
            "the new epoch opens with no verdict of its own about h_a"
        );

        let cfg = replay_cfg(creature, train, tmp.path().join("out"), learnings_dir, None);
        let run = establish_run(&cfg, &improving_scorer()).unwrap();

        assert_eq!(run.stop_reason, "replay-accepts");
        assert!(
            run.workspace.join("replay-1").is_dir(),
            "the historical winner must be re-scored against this corpus"
        );
        let best = std::fs::read_to_string(cfg.output_dir.join("best.json")).unwrap();
        assert!(!best.contains("h_a"), "{best}");
        let filed = store.load().unwrap();
        assert!(
            filed
                .iter()
                .any(|l| l.uuid == "h_a" && l.outcome == Outcome::Accepted),
            "the accept is filed under the corpus that measured it: {filed:?}"
        );
    }

    /// The other half of Issue #101: the same historical winner is replayed, the
    /// current corpus scores it and says no. Nothing is cut, and this corpus's
    /// own rejection is what stands — history proposed, it did not decide.
    /// A historical hypothesis bundled with this corpus's own win is judged on
    /// current-corpus measurements, never on the old epoch's word: when the
    /// combined plan misses, the probe path measures every member individually
    /// against the corpus in hand and files it on that measurement (#57, #101).
    #[test]
    fn a_bundled_historical_hypothesis_is_measured_against_this_corpus() {
        let tmp = tempfile::tempdir().unwrap();
        let (creature, train) = two_hidden_paths(tmp.path());
        let learnings_dir = tmp.path().join("learnings");
        let store = screens_store(&learnings_dir, &train);
        seed_verdicts(&store, &[("h_a", Outcome::Accepted, None, 10)]);
        seed_prior_corpus(&learnings_dir, "h_b", Outcome::Accepted, 10);

        let cfg = replay_cfg(creature, train, tmp.path().join("out"), learnings_dir, None);
        let run = establish_run(&cfg, &ScriptedScorer::ok(0.50, 0.50)).unwrap();

        assert_eq!(run.accepts, 0, "a flat scorer accepts neither member");
        let filed = store.load().unwrap();
        let historical = filed
            .iter()
            .rfind(|l| l.uuid == "h_b")
            .expect("the historical member is judged here");
        assert_eq!(historical.outcome, Outcome::Rejected);
        assert!(
            historical.full_delta.is_some(),
            "its own delta was measured against this corpus: {historical:?}"
        );
    }

    #[test]
    fn a_historical_winner_the_current_corpus_rejects_is_not_cut() {
        let tmp = tempfile::tempdir().unwrap();
        let (creature, train) = two_hidden_paths(tmp.path());
        let learnings_dir = tmp.path().join("learnings");
        seed_prior_corpus(&learnings_dir, "h_a", Outcome::Accepted, 10);
        let store = screens_store(&learnings_dir, &train);

        let cfg = OckhamConfig {
            max_experiments: Some(1),
            ..replay_cfg(creature, train, tmp.path().join("out"), learnings_dir, None)
        };
        // A candidate that scores exactly the incumbent wins nothing.
        let run = establish_run(&cfg, &ScriptedScorer::ok(0.50, 0.50)).unwrap();

        assert_eq!(
            run.accepts, 0,
            "an old epoch's win cannot accept a cut here"
        );
        assert_ne!(run.stop_reason, "replay-accepts");
        assert!(
            run.workspace.join("replay-1").is_dir(),
            "it was still tried"
        );
        let filed = store.load().unwrap();
        assert!(
            filed
                .iter()
                .any(|l| l.uuid == "h_a" && l.outcome == Outcome::Rejected),
            "this corpus's own verdict is what the fleet keeps: {filed:?}"
        );
    }

    /// A historical **failure** is not a hypothesis, and it suppresses nothing:
    /// the run replays nothing and screens the neuron on its merits (#101).
    #[test]
    fn a_historical_failure_is_replayed_by_nothing_and_suppresses_nothing() {
        let tmp = tempfile::tempdir().unwrap();
        let (creature, train) = two_hidden_paths(tmp.path());
        let learnings_dir = tmp.path().join("learnings");
        seed_prior_corpus(&learnings_dir, "h_a", Outcome::Rejected, 10);
        let store = screens_store(&learnings_dir, &train);

        let cfg = OckhamConfig {
            max_experiments: Some(1),
            ..replay_cfg(creature, train, tmp.path().join("out"), learnings_dir, None)
        };
        let run = establish_run(&cfg, &ScriptedScorer::ok(0.50, 0.50)).unwrap();

        assert!(
            !run.workspace.join("replay-1").exists(),
            "an old failure is no hypothesis to replay"
        );
        assert!(
            screened_uuids(&store).contains(&"h_a".to_string()),
            "and it is still screened this epoch rather than skipped"
        );
    }

    /// `--old-corpus-first=false` turns the whole historical channel off, replay
    /// included: the run sees only what this corpus has measured.
    #[test]
    fn old_corpus_first_off_replays_no_history() {
        let tmp = tempfile::tempdir().unwrap();
        let (creature, train) = two_hidden_paths(tmp.path());
        let learnings_dir = tmp.path().join("learnings");
        seed_prior_corpus(&learnings_dir, "h_a", Outcome::Accepted, 10);

        let cfg = OckhamConfig {
            old_corpus_first: Some(false),
            ..replay_cfg(creature, train, tmp.path().join("out"), learnings_dir, None)
        };
        let run = establish_run(&cfg, &improving_scorer()).unwrap();

        assert!(
            !run.workspace.join("replay-1").exists(),
            "with the priority off nothing historical is replayed"
        );
    }

    #[test]
    fn a_measured_loss_is_still_not_replayed() {
        let tmp = tempfile::tempdir().unwrap();
        let (creature, train) = two_hidden_paths(tmp.path());
        let learnings_dir = tmp.path().join("learnings");
        let store = screens_store(&learnings_dir, &train);
        seed_verdicts(
            &store,
            &[
                (
                    "h_a",
                    Outcome::Rejected,
                    Some(-1.0),
                    crate::incumbent::now_unix(),
                ),
                ("h_b", Outcome::Rejected, None, crate::incumbent::now_unix()),
            ],
        );
        let cfg = OckhamConfig {
            max_experiments: Some(1),
            ..replay_cfg(creature, train, tmp.path().join("out"), learnings_dir, None)
        };
        let run = establish_run(&cfg, &ScriptedScorer::ok(0.50, 0.50)).unwrap();
        assert_ne!(run.stop_reason, "replay-accepts");
        let journal = std::fs::read_to_string(cfg.output_dir.join("experiments.jsonl")).unwrap();
        assert!(
            journal.contains(r#""record":"batch""#),
            "the run went to the sweep rather than replaying failures: {journal}"
        );
    }

    /// `--max-full` is an individual-scoring cap for the search loop; sizing a
    /// replay probe with it was always a conflation (Issue #57).
    #[test]
    fn a_replay_probe_is_not_sized_by_max_full() {
        let tmp = tempfile::tempdir().unwrap();
        let (creature, train) = hidden_paths(tmp.path(), &["h_a", "h_b", "h_c"]);
        let learnings_dir = tmp.path().join("learnings");
        let store = screens_store(&learnings_dir, &train);
        seed_verdicts(
            &store,
            &[
                ("h_a", Outcome::Accepted, None, 10),
                ("h_b", Outcome::Accepted, None, 20),
                ("h_c", Outcome::Accepted, None, 30),
            ],
        );
        let cfg = replay_cfg(
            creature,
            train,
            tmp.path().join("out"),
            learnings_dir,
            Some(1),
        );
        // Every plan scores exactly the incumbent, so the combined bundle and
        // every shrink step miss and the probes run.
        let run = establish_run(&cfg, &ScriptedScorer::ok(0.50, 0.50)).unwrap();
        assert_eq!(run.accepts, 0);
        let probe = run.workspace.join("replay-2");
        assert!(
            probe.is_dir(),
            "a missed replay cohort must fall back to individual probes"
        );
        assert_eq!(
            cohort_stems(&probe, 'i').len(),
            3,
            "all three known wins are probed despite --max-full 1"
        );
    }

    #[test]
    fn a_missed_replay_bundle_shrinks_before_it_probes() {
        let tmp = tempfile::tempdir().unwrap();
        let uuids: Vec<String> = (0..6).map(|i| format!("h_{i}")).collect();
        let refs: Vec<&str> = uuids.iter().map(String::as_str).collect();
        let (creature, train) = hidden_paths(tmp.path(), &refs);
        let learnings_dir = tmp.path().join("learnings");
        let store = screens_store(&learnings_dir, &train);
        let records: Vec<(&str, Outcome, Option<f64>, u64)> = refs
            .iter()
            .enumerate()
            .map(|(i, u)| (*u, Outcome::Accepted, None, 10 + i as u64))
            .collect();
        seed_verdicts(&store, &records);
        let cfg = replay_cfg(creature, train, tmp.path().join("out"), learnings_dir, None);
        establish_run(&cfg, &ScriptedScorer::ok(0.50, 0.50)).unwrap();

        let combined = cfg.output_dir.join("workspace").join("replay-1");
        let baseline = hidden_in(&combined.join("baseline.json"));
        let mut cuts: Vec<usize> = cohort_stems(&combined, 'b')
            .iter()
            .map(|stem| baseline - hidden_in(&combined.join(format!("{stem}.json"))))
            .collect();
        cuts.sort_unstable();
        assert_eq!(cuts.last(), Some(&6), "the combined plan is tried");
        assert!(
            cuts.len() > 1,
            "a shrink step must share the cohort with it: {cuts:?}"
        );
        assert!(cuts.len() <= crate::promote::MAX_REPLAY_PLANS, "{cuts:?}");
    }

    // ---- Issue #58: the cohort is sized to the wall clock ------------------

    #[test]
    fn the_cost_estimate_comes_from_observed_full_timings() {
        let mut cost = CostModel::new(Some(0.01), 1.0);
        assert_eq!(cost.per_creature_ms(), None, "nothing measured yet");
        assert_eq!(
            cost.cohort_budget(Duration::from_secs(600)),
            CohortBudget::Unmeasured
        );

        // A screen stands in until the first full cohort: 100ms per creature
        // over a 1% sample is roughly 10s per creature on the full corpus.
        cost.observe_screen(1_000, 10);
        let fallback = cost.per_creature_ms().unwrap();
        assert!((fallback - 15_000.0).abs() < 1.0, "{fallback}");
        assert_eq!(
            cost.cohort_budget(Duration::from_secs(600)),
            CohortBudget::Entries(29),
            "a first batch with no full timing still launches a bounded cohort"
        );

        // The first real cohort replaces it outright, later ones smooth in.
        cost.observe_full(4_000, 3);
        assert_eq!(cost.per_creature_ms(), Some(1_000.0));
        cost.observe_full(40_000, 3);
        let smoothed = cost.per_creature_ms().unwrap();
        assert!(
            smoothed > 1_000.0 && smoothed < 10_000.0,
            "one anomalous cohort must not become the estimate: {smoothed}"
        );
    }

    #[test]
    fn the_cohort_is_sized_to_the_budget_with_a_check_in_reserve() {
        let mut cost = CostModel::new(Some(0.01), 1.0);
        cost.observe_full(11_000, 10);
        assert_eq!(cost.per_creature_ms(), Some(1_000.0));
        // 100s left, a quarter reserved for applying the win: 75 creatures,
        // one of which is the incumbent baseline.
        assert_eq!(
            cost.cohort_budget(Duration::from_secs(100)),
            CohortBudget::Entries(74)
        );
        assert_eq!(
            cost.cohort_budget(Duration::from_secs(1)),
            CohortBudget::TooSmall,
            "a cohort that cannot finish must never be started"
        );
    }

    #[test]
    fn a_starved_run_stops_before_launching_a_cohort_it_cannot_finish() {
        let tmp = tempfile::tempdir().unwrap();
        let (creature, train) = hidden_paths(tmp.path(), &["h_a", "h_b", "h_c", "h_d"]);
        let cfg = OckhamConfig {
            creature,
            training_data: train,
            output_dir: tmp.path().join("out"),
            // Enough to score the baseline and one screen, nowhere near enough
            // for a full cohort at the cost that screen implies.
            timeout: Duration::from_secs(2),
            seed: Some(1),
            candidates: 4,
            screen_sample_rate: Some(0.01),
            ..OckhamConfig::default()
        };
        let scorer = ScriptedScorer {
            delay_per_creature: Duration::from_millis(100),
            baseline_score: 0.50,
            candidate_score: Some(0.80),
            ..ScriptedScorer::ok(0.50, 0.50)
        };
        let run = establish_run(&cfg, &scorer).unwrap();
        assert_eq!(run.stop_reason, "budget");
        let journal = std::fs::read_to_string(cfg.output_dir.join("experiments.jsonl")).unwrap();
        assert!(
            !journal.contains(r#""record":"full""#),
            "no cohort may be launched: {journal}"
        );
    }

    #[test]
    fn a_generous_budget_leaves_the_cohort_untrimmed() {
        let tmp = tempfile::tempdir().unwrap();
        let (creature, train) = hidden_paths(tmp.path(), &["h_a", "h_b", "h_c", "h_d"]);
        let cfg = OckhamConfig {
            creature,
            training_data: train,
            output_dir: tmp.path().join("out"),
            timeout: Duration::from_secs(600),
            max_experiments: Some(1),
            seed: Some(1),
            candidates: 4,
            ..OckhamConfig::default()
        };
        let scorer = ScriptedScorer {
            baseline_score: 0.50,
            candidate_score: Some(0.80),
            ..ScriptedScorer::ok(0.50, 0.50)
        };
        establish_run(&cfg, &scorer).unwrap();
        let journal = std::fs::read_to_string(cfg.output_dir.join("experiments.jsonl")).unwrap();
        assert!(journal.contains(r#""record":"budget""#), "{journal}");
        let report =
            crate::report::summarise(&[&cfg.output_dir.join("experiments.jsonl")]).unwrap();
        assert_eq!(report.budget_trims, 0, "the guard must be inert here");
        assert_eq!(report.budget_dropped, 0);
    }

    #[test]
    fn the_trim_line_names_what_was_dropped_and_why() {
        let mut full = outcome_with(&[("h_a", 1.0)], None);
        full.dropped_individuals = 24;
        full.dropped_bundles = 7;
        assert_eq!(
            budget_trim_line(Duration::from_secs(240), 18_000.0, &full),
            "full: budget 240s, est 18s/creature → 1 of 32 entries; dropped 31 (24 individual, 7 bundle)"
        );
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

    // ---------------------------------------------------------------------
    // Issue #77: per-run screening progress — restart, never spin, and report.
    // ---------------------------------------------------------------------

    /// Every journal record of `kind`, in the order they were written.
    ///
    /// A line that does not parse fails the test rather than being skipped: a
    /// serialisation this change broke must never look like a missing record.
    fn journal_records(out: &std::path::Path, kind: &str) -> Vec<serde_json::Value> {
        let text = std::fs::read_to_string(out.join("experiments.jsonl")).unwrap();
        text.lines()
            .map(|l| serde_json::from_str::<serde_json::Value>(l).expect("journal line is JSON"))
            .filter(|v| v["record"] == kind)
            .collect()
    }

    /// A scorer that prefers the incumbent to every candidate, so a run screens
    /// batch after batch without ever accepting and restarting on a win.
    fn losing_scorer() -> ScriptedScorer {
        ScriptedScorer {
            baseline_score: 0.80,
            candidate_score: Some(0.10),
            ..ScriptedScorer::ok(0.80, 0.20)
        }
    }

    /// Config that exhausts its sweep every batch: two neurons, two candidates.
    fn restart_cfg(
        creature: std::path::PathBuf,
        train: std::path::PathBuf,
        out: std::path::PathBuf,
        learnings_dir: Option<std::path::PathBuf>,
        max_experiments: Option<u64>,
    ) -> OckhamConfig {
        OckhamConfig {
            creature,
            training_data: train,
            output_dir: out,
            timeout: Duration::from_secs(30),
            max_experiments,
            seed: Some(1),
            candidates: 2,
            learnings_dir,
            learnings_host: Some("t".into()),
            ..OckhamConfig::default()
        }
    }

    /// What #77 removes: before this an exhausted sweep ended the run outright,
    /// so a creature the fleet had already worked through stopped being
    /// screened. Asserted on the batch records, never on wall-clock — a timing
    /// assertion would pass on a fast machine and prove nothing either way.
    #[test]
    fn an_exhausted_sweep_restarts_rather_than_issuing_empty_batches() {
        let tmp = tempfile::tempdir().unwrap();
        let (creature, train) = two_hidden_paths(tmp.path());
        let cfg = restart_cfg(
            creature,
            train,
            tmp.path().join("out"),
            Some(tmp.path().join("learnings")),
            Some(4),
        );
        let run = establish_run(&cfg, &losing_scorer()).unwrap();

        assert_eq!(run.stop_reason, "max-experiments");
        let batches = journal_records(&cfg.output_dir, "batch");
        assert_eq!(batches.len(), 4, "{batches:?}");
        for batch in &batches {
            assert_eq!(
                batch["candidates"], 2,
                "an exhausted sweep must refill, not issue an empty batch: {batch}"
            );
        }
        let restarts = journal_records(&cfg.output_dir, "sweepRestart");
        assert_eq!(
            restarts.len(),
            3,
            "one restart per exhausted pass: {restarts:?}"
        );
        assert_eq!(restarts[0]["restarts"], 1);
        assert_eq!(restarts[0]["hidden"], 2);
        assert_eq!(restarts[2]["newly_screened"], 2, "{restarts:?}");
    }

    /// The recycling half of the restart: with every neuron already screened,
    /// block A is empty and the fresh sweep is the stalest-first order itself.
    #[test]
    fn a_fully_screened_creature_refills_after_a_restart_stalest_first() {
        let creature = hidden_creature(&["h_a", "h_b", "h_c"]);
        let stats = stats_of(&creature);
        // Oldest screen first: h_c was looked at longest ago.
        let screens: Vec<Screened> = [("h_a", 30u64), ("h_b", 20), ("h_c", 10)]
            .into_iter()
            .map(|(uuid, unix_secs)| Screened {
                blocked_reason: Default::default(),
                version: crate::learnings::SCREENS_FORMAT_VERSION,
                uuid: uuid.into(),
                kind: "identity".into(),
                outcome: ScreenOutcomeKind::Loser,
                unix_secs,
                host: "t".into(),
                corpus_identity: None,
            })
            .collect();

        let mut sweep = fresh_sweep(
            &creature,
            &stats,
            7,
            crate::ordering::OrderingConfig::default(),
            true,
            &screens,
            &PriorHint::none(),
        );
        assert_eq!(
            sweep.order,
            vec!["h_c".to_string(), "h_b".to_string(), "h_a".to_string()],
            "a restart on a fully screened creature recycles stalest-first"
        );
        let (batch, skips) = sweep.fill_batch(&creature, &stats, 3);
        assert!(skips.is_empty(), "{skips:?}");
        assert_eq!(
            batch.iter().map(|c| c.uuid.as_str()).collect::<Vec<_>>(),
            vec!["h_c", "h_b", "h_a"],
            "the restart must still fill batches, in that order"
        );
    }

    /// The same recycling driven through the loop itself: a run over a
    /// creature the fleet has already screened end to end must keep filling
    /// batches across the restart, stalest neuron first each time.
    #[test]
    fn a_run_over_a_fully_screened_creature_recycles_stalest_first_across_a_restart() {
        let tmp = tempfile::tempdir().unwrap();
        let (creature, train) = two_hidden_paths(tmp.path());
        let learnings_dir = tmp.path().join("learnings");
        let store = screens_store(&learnings_dir, &train);
        // h_b was looked at longest ago, so it is the stalest of the two. Both
        // are filed under the run's own corpus, so they are current-epoch
        // coverage rather than history (#100).
        for (uuid, unix_secs) in [("h_a", 200u64), ("h_b", 100)] {
            store
                .append_screen(&Screened {
                    blocked_reason: Default::default(),
                    version: crate::learnings::SCREENS_FORMAT_VERSION,
                    uuid: uuid.into(),
                    kind: "identity".into(),
                    outcome: ScreenOutcomeKind::Loser,
                    unix_secs,
                    host: "seed".into(),
                    corpus_identity: Some(store.corpus_identity().to_string()),
                })
                .unwrap();
        }
        let cfg = OckhamConfig {
            candidates: 1,
            ..restart_cfg(
                creature,
                train,
                tmp.path().join("out"),
                Some(learnings_dir),
                Some(3),
            )
        };
        let run = establish_run(&cfg, &losing_scorer()).unwrap();

        assert_eq!(run.stop_reason, "max-experiments");
        assert_eq!(
            run.newly_screened, 0,
            "re-screening an already-screened creature advances no coverage"
        );
        // One record per batch, and the third comes after the restart.
        let filed: Vec<String> = store
            .load_screens()
            .unwrap()
            .into_iter()
            .filter(|s| s.host == "t")
            .map(|s| s.uuid)
            .collect();
        assert_eq!(
            filed[..2],
            ["h_b".to_string(), "h_a".to_string()],
            "the stalest of the two must be screened first"
        );
        assert_eq!(
            filed.len(),
            3,
            "the restart must refill rather than starve the run: {filed:?}"
        );
        // The third batch recycles one of the two again. Which one is not
        // asserted: both carry a record from this same second, and screen
        // records are timestamped in whole seconds, so their staleness is
        // genuinely equal by then.
        assert_eq!(journal_records(&cfg.output_dir, "sweepRestart").len(), 1);
    }

    /// A restart that would re-exhaust immediately must stop with a reason.
    /// Every neuron here is a fresh known failure, so no pass can ever propose.
    #[test]
    fn a_creature_that_can_never_propose_stops_instead_of_looping() {
        let tmp = tempfile::tempdir().unwrap();
        let (creature, train) = two_hidden_paths(tmp.path());
        let learnings_dir = tmp.path().join("learnings");
        let store = screens_store(&learnings_dir, &train);
        let now = crate::incumbent::now_unix();
        seed_verdicts(
            &store,
            &[
                ("h_a", Outcome::Rejected, Some(-1.0), now),
                ("h_b", Outcome::Rejected, Some(-1.0), now),
            ],
        );
        let cfg = restart_cfg(
            creature,
            train,
            tmp.path().join("out"),
            Some(learnings_dir),
            None,
        );
        let run = establish_run(&cfg, &losing_scorer()).unwrap();

        assert_eq!(run.stop_reason, "no-candidates");
        // Was 0 before Issue #93: a visit a standing verdict suppressed filed
        // nothing, so a barren pass reported no coverage at all. The pass still
        // stops — nothing could be proposed — but the two visits it made are
        // coverage, and pretending otherwise is what pinned `checked`.
        assert_eq!(run.newly_screened, 2);
        assert!(
            journal_records(&cfg.output_dir, "sweepRestart").is_empty(),
            "a barren pass must stop, not restart into the same nothing"
        );
    }

    /// Issue #93: the kind filed against a skip is two buckets wide, so the
    /// reason is logged. Issue #103 counts it by reason **code**, so the log
    /// line, the screen record and the coverage breakdown all say the same
    /// thing — and a class can never be one-per-neuron again, because it no
    /// longer comes from the message text.
    #[test]
    fn skip_reasons_are_tallied_by_code_not_by_neuron() {
        use crate::blocked::BlockedReason;
        use crate::sweep::SweepSkip;
        let skip = |uuid: &str, reason: String, blocked: Option<BlockedReason>| SweepSkip {
            uuid: uuid.into(),
            permutation_index: 0,
            reason,
            blocked,
        };
        let aggregate = |uuid: &str| {
            skip(
                uuid,
                format!("aggregate target `{uuid}-target` (`MEAN`); skipped"),
                Some(BlockedReason::AggregateSquash),
            )
        };
        let skips = vec![
            aggregate("h_a"),
            aggregate("h_b"),
            aggregate("h_c"),
            skip(
                "h_d",
                "typed synapse `h_d`→`h_if` (condition); skipped".into(),
                Some(BlockedReason::UnsafeTopology),
            ),
            skip("h_e", crate::sweep::KNOWN_FAILURE_REASON.into(), None),
            skip(
                "h_f",
                "non-finite mean NaN".into(),
                Some(BlockedReason::MissingActivation),
            ),
        ];
        assert_eq!(
            skip_reason_tally(&skips),
            "aggregate-squash: 3, known-failure: 1, missing-activation: 1, unsafe-topology: 1",
            "commonest first, then alphabetical, and no uuid in sight"
        );
    }

    /// A standing full-corpus verdict is checked in the strongest sense — the
    /// cut was proposed, scored and judged — so it is never counted as blocked,
    /// whatever the message on it says.
    #[test]
    fn a_known_failure_files_no_blocked_reason() {
        use crate::blocked::BlockedReason;
        use crate::sweep::SweepSkip;
        let known = SweepSkip {
            uuid: "h_known".into(),
            permutation_index: 0,
            reason: crate::sweep::KNOWN_FAILURE_REASON.into(),
            blocked: None,
        };
        let blocked = SweepSkip {
            uuid: "h_blocked".into(),
            permutation_index: 1,
            reason: "aggregate target `t` (`MEAN`); skipped".into(),
            blocked: Some(BlockedReason::AggregateSquash),
        };
        let known = skip_try(&known);
        assert_eq!(known.kind, crate::learnings::SCREEN_KIND_KNOWN_FAILURE);
        assert_eq!(known.blocked_reason, None);
        let blocked = skip_try(&blocked);
        assert_eq!(blocked.kind, crate::learnings::SCREEN_KIND_SKIPPED);
        assert_eq!(
            blocked.blocked_reason,
            Some(BlockedReason::AggregateSquash),
            "a blocked visit files the code that stopped it"
        );

        // Fail closed: a skip carrying neither the known-failure reason nor a
        // code is blocked for an unknown reason, never upgraded to a verdict
        // the fleet never reached.
        let unexplained = SweepSkip {
            uuid: "h_odd".into(),
            permutation_index: 2,
            reason: "something new".into(),
            blocked: None,
        };
        let unexplained = skip_try(&unexplained);
        assert_eq!(unexplained.kind, crate::learnings::SCREEN_KIND_SKIPPED);
        assert_eq!(unexplained.blocked_reason, Some(BlockedReason::Other));
    }

    /// Issue #77 point 3, the sizing rules, unit by unit. A measured screen is
    /// the best answer; a full-corpus estimate scaled by the sample rate is the
    /// fallback; screening disabled means the batch *is* a cohort.
    #[test]
    fn the_screening_reserve_is_sized_at_one_batch() {
        let mut cost = CostModel::new(Some(0.05), 1.0);
        assert_eq!(cost.screen_batch_ms(99), None, "nothing measured yet");

        // The opening baseline: one creature over the full corpus, 200ms.
        cost.observe_baseline(200);
        assert_eq!(
            cost.screen_batch_ms(99),
            Some(200.0 * 0.05 * 100.0),
            "a screen scores the same creatures over 5% of the corpus"
        );
        // A measured screen beats every estimate of one.
        cost.observe_screen(1_000, 100);
        assert_eq!(cost.screen_batch_ms(99), Some(10.0 * 100.0));

        let mut off = CostModel::new(None, 1.0);
        off.observe_baseline(200);
        assert_eq!(
            off.screen_batch_ms(9),
            Some(200.0 * 10.0),
            "with screening disabled the check is the full cohort itself"
        );
    }

    /// Issue #104: a ladder batch runs every rung, so the reserve must price
    /// every rung — not just the promotion stage it is measured in.
    #[test]
    fn the_reserve_prices_every_rung_of_a_ladder() {
        let ladder = crate::screening::ScreenLadder::parse("0.0025,0.01,0.05", 0.01).unwrap();
        let multiple = batch_cost_multiple(&ladder);
        assert!(
            (multiple - (0.0025 + 0.01 + 0.05) / 0.05).abs() < 1e-12,
            "multiple={multiple}"
        );
        assert_eq!(
            batch_cost_multiple(&crate::screening::ScreenLadder::single(0.05).unwrap()),
            1.0,
            "the control is exactly one pass"
        );

        let control = {
            let mut c = CostModel::new(Some(0.05), 1.0);
            c.observe_baseline(200);
            c.screen_batch_ms(99).unwrap()
        };
        let laddered = {
            let mut c = CostModel::new(Some(0.05), multiple);
            c.observe_baseline(200);
            c.screen_batch_ms(99).unwrap()
        };
        assert!(
            (laddered - control * multiple).abs() < 1e-9,
            "control={control} laddered={laddered}"
        );
    }

    /// The reserve is claimed only when the budget is down to one batch, and
    /// never when one batch would swallow half the run.
    #[test]
    fn the_reserve_stands_only_when_the_budget_is_down_to_one_batch() {
        let tmp = tempfile::tempdir().unwrap();
        let cfg = OckhamConfig {
            timeout: Duration::from_secs(10),
            candidates: 99,
            ..config(tmp.path())
        };
        let mut cost = CostModel::new(Some(0.05), 1.0);
        assert!(
            !reserve_stands(&cost, &cfg, Duration::from_millis(10), 0),
            "an unmeasured reserve must never cost replay a cohort"
        );

        cost.observe_screen(1_000, 100); // 10ms/creature → 1s a batch
        assert!(!reserve_stands(&cost, &cfg, Duration::from_secs(4), 0));
        assert!(reserve_stands(&cost, &cfg, Duration::from_millis(999), 0));
        assert!(reserve_stands(&cost, &cfg, Duration::ZERO, 0));
        assert!(
            !reserve_stands(&cost, &cfg, Duration::ZERO, 1),
            "a run that has already screened has nothing left to guarantee"
        );

        // A batch costing more than half the run is the whole plan, not a
        // reserve: the run keeps its pre-#77 behaviour.
        let tiny = OckhamConfig {
            timeout: Duration::from_secs(1),
            ..cfg
        };
        assert!(!reserve_stands(&cost, &tiny, Duration::from_millis(500), 0));
    }

    /// The behaviour those rules buy, end to end: a run whose budget has fallen
    /// to its last batch stands the replay stage down and screens instead.
    ///
    /// The one test here that depends on the wall clock, unavoidably: the
    /// reserve is a statement about time left, and the scorer's per-creature
    /// delay is how a test spends a budget. The margin is deliberately wide —
    /// the replay stage's 15 scored creatures nominally spend 1.5s of the 2s
    /// budget, so it takes better than 30% jitter to reach the deadline first.
    /// The assertions themselves are on what was screened and on record order,
    /// never on elapsed time.
    #[test]
    fn a_run_down_to_its_last_batch_screens_it_rather_than_replaying() {
        let tmp = tempfile::tempdir().unwrap();
        let uuids: Vec<String> = (0..12).map(|i| format!("h{i:02}")).collect();
        let names: Vec<&str> = uuids.iter().map(String::as_str).collect();
        let (creature, train) = hidden_paths(tmp.path(), &names);
        let learnings_dir = tmp.path().join("learnings");
        // Ten known wins: more replay work than one round of probing resolves,
        // so before #77 the replay stage spent every millisecond of what was
        // left and the run screened nothing at all.
        known_wins(&learnings_dir, &train, &names[..10]);
        let cfg = OckhamConfig {
            // The replay stage scores 15 creatures before it would come round
            // again (a combined-plan cohort, then a probe cohort, at 100ms a
            // creature). What is left of the budget is then below the 0.6s one
            // screening batch is estimated to cost, so the reserve stands and
            // the stage never gets its next round.
            timeout: Duration::from_millis(2_000),
            candidates: 11,
            screen_sample_rate: Some(0.5),
            ..restart_cfg(
                creature,
                train,
                tmp.path().join("out"),
                Some(learnings_dir),
                None,
            )
        };
        let scorer = ScriptedScorer {
            delay_per_creature: Duration::from_millis(100),
            ..losing_scorer()
        };
        let run = establish_run(&cfg, &scorer).unwrap();

        assert!(
            run.newly_screened > 0,
            "the reserve must buy this run a screening batch: {}",
            run.newly_screened
        );
        assert_eq!(
            coverage_json(&cfg.output_dir).checked,
            run.newly_screened,
            "every uuid the reserved batch screened must be counted as checked"
        );
        // The proof that the reserve, not luck, let the sweep in: the batch is
        // journalled after the replay stage's own cohorts.
        let text = std::fs::read_to_string(cfg.output_dir.join("experiments.jsonl")).unwrap();
        let full = text
            .rfind(r#""record":"full""#)
            .expect("replay scored a cohort");
        let batch = text
            .find(r#""record":"batch""#)
            .expect("a batch was filled");
        assert!(
            batch > full,
            "the batch must follow the replay stage: {text}"
        );
    }

    /// A zero-progress run must be self-reporting: the count is in the stop
    /// record, in the run summary and in the coverage description block.
    #[test]
    fn the_stop_record_and_run_summary_carry_the_uuids_newly_screened() {
        let tmp = tempfile::tempdir().unwrap();
        let (creature, train) = hidden_paths(tmp.path(), &["h_a", "h_b", "h_c", "h_d"]);
        let cfg = restart_cfg(
            creature,
            train,
            tmp.path().join("out"),
            Some(tmp.path().join("learnings")),
            Some(1),
        );
        let run = establish_run(&cfg, &losing_scorer()).unwrap();

        assert_eq!(run.newly_screened, 2, "one batch of two candidates");
        let stops = journal_records(&cfg.output_dir, "stop");
        assert_eq!(stops.len(), 1);
        assert_eq!(stops[0]["newly_screened"], 2, "{:?}", stops[0]);
        let text =
            std::fs::read_to_string(cfg.output_dir.join(crate::coverage::COVERAGE_TEXT_FILE))
                .unwrap();
        assert!(
            text.contains("progress:  2 newly checked this run"),
            "{text}"
        );
    }

    /// A run that stops before screening anything reports zero progress rather
    /// than the same well-formed block a full batch renders.
    #[test]
    fn a_run_that_screened_nothing_reports_zero_progress() {
        let tmp = tempfile::tempdir().unwrap();
        let (creature, train) = two_hidden_paths(tmp.path());
        let cfg = restart_cfg(
            creature,
            train,
            tmp.path().join("out"),
            Some(tmp.path().join("learnings")),
            Some(0),
        );
        let run = establish_run(&cfg, &losing_scorer()).unwrap();

        assert_eq!(run.stop_reason, "max-experiments");
        assert_eq!(run.newly_screened, 0);
        let cov = coverage_json(&cfg.output_dir);
        assert_eq!(cov.unchecked(), 2, "the figures the warning names");
        let text =
            std::fs::read_to_string(cfg.output_dir.join(crate::coverage::COVERAGE_TEXT_FILE))
                .unwrap();
        assert!(
            text.contains("progress:  0 newly checked this run"),
            "{text}"
        );
    }

    /// The contract of #63, asserted end to end: every run advances the checked
    /// count by the batch size, bounded by the unchecked remainder.
    #[test]
    fn two_successive_runs_advance_the_checked_count_by_the_batch_size() {
        let tmp = tempfile::tempdir().unwrap();
        let (creature, train) = hidden_paths(tmp.path(), &["h_a", "h_b", "h_c"]);
        let learnings_dir = tmp.path().join("learnings");

        let first = restart_cfg(
            creature.clone(),
            train.clone(),
            tmp.path().join("out-1"),
            Some(learnings_dir.clone()),
            Some(1),
        );
        let first_run = establish_run(&first, &losing_scorer()).unwrap();
        assert_eq!(first_run.newly_screened, 2, "a full batch");
        assert_eq!(coverage_json(&first.output_dir).checked, 2);

        let second = restart_cfg(
            creature,
            train,
            tmp.path().join("out-2"),
            Some(learnings_dir),
            Some(1),
        );
        let second_run = establish_run(&second, &losing_scorer()).unwrap();
        assert_eq!(
            second_run.newly_screened, 1,
            "bounded by the unchecked remainder, not the batch size"
        );
        let cov = coverage_json(&second.output_dir);
        assert_eq!(cov.checked, 3);
        assert_eq!(cov.unchecked(), 0);
    }

    // ---------------------------------------------------------------------
    // Issue #104: progressive adaptive screening.
    // ---------------------------------------------------------------------

    fn ladder_cfg(
        creature: std::path::PathBuf,
        train: std::path::PathBuf,
        out: std::path::PathBuf,
        stages: Option<&str>,
    ) -> OckhamConfig {
        OckhamConfig {
            creature,
            training_data: train,
            output_dir: out,
            timeout: Duration::from_secs(30),
            max_experiments: Some(4),
            seed: Some(1),
            candidates: 2,
            screen_stages: stages.map(|s| crate::screening::ScreenLadder::parse(s, 0.01).unwrap()),
            ..OckhamConfig::default()
        }
    }

    /// The ladder may only change *when* a candidate is dropped, never what the
    /// full scorer accepts: same scorer, same seed, same outcome.
    #[test]
    fn a_progressive_ladder_accepts_exactly_what_the_fixed_rate_control_accepts() {
        let tmp = tempfile::tempdir().unwrap();
        let (creature, train) = two_hidden_paths(tmp.path());
        let winning = || ScriptedScorer {
            baseline_score: 0.50,
            candidate_score: Some(0.80),
            ..ScriptedScorer::ok(0.50, 0.50)
        };

        let control = ladder_cfg(
            creature.clone(),
            train.clone(),
            tmp.path().join("control"),
            None,
        );
        let control_run = establish_run(&control, &winning()).unwrap();

        let ladder = ladder_cfg(
            creature,
            train,
            tmp.path().join("ladder"),
            Some("0.0025,0.01,0.05"),
        );
        let ladder_run = establish_run(&ladder, &winning()).unwrap();

        assert_eq!(control_run.accepts, ladder_run.accepts);
        assert_eq!(control_run.stop_reason, ladder_run.stop_reason);
        assert_eq!(control_run.cumulative_delta, ladder_run.cumulative_delta);
        assert_eq!(
            control_run.baseline.score, ladder_run.baseline.score,
            "the authoritative opening is untouched by how candidates are screened"
        );
        assert!(control_run.accepts > 0, "the fixture must accept something");
    }

    /// The whole claim of #104: an obvious loser is paid for at the smallest
    /// sample and never reaches the larger ones.
    #[test]
    fn an_obvious_loser_stops_at_the_first_rung_of_the_ladder() {
        let tmp = tempfile::tempdir().unwrap();
        let (creature, train) = two_hidden_paths(tmp.path());
        let cfg = ladder_cfg(
            creature,
            train,
            tmp.path().join("out"),
            Some("0.0025,0.01,0.05"),
        );
        let run = establish_run(&cfg, &losing_scorer()).unwrap();
        assert_eq!(run.stop_reason, "max-experiments");

        let stages = journal_records(&cfg.output_dir, "screenStage");
        assert!(!stages.is_empty(), "a progressive run journals its rungs");
        for stage in &stages {
            assert_eq!(
                stage["stage"], 0,
                "a candidate 70 points behind must never be re-tested at a larger sample: {stage}"
            );
            assert_eq!(stage["rate"], 0.0025);
            assert_eq!(stage["promoted"], 0);
            assert_eq!(stage["rejected"], stage["entered"]);
            assert_eq!(stage["outcome"], "carried");
        }
        // The batch still counts as screened: coverage is unaffected by which
        // rung ended the candidate.
        assert_eq!(run.newly_screened, 2);
    }

    /// The control's journal must not grow a record per batch it cannot use.
    #[test]
    fn the_fixed_rate_control_journals_no_stage_records() {
        let tmp = tempfile::tempdir().unwrap();
        let (creature, train) = two_hidden_paths(tmp.path());
        let cfg = ladder_cfg(creature, train, tmp.path().join("out"), None);
        establish_run(&cfg, &losing_scorer()).unwrap();
        assert!(journal_records(&cfg.output_dir, "screenStage").is_empty());
        assert!(!journal_records(&cfg.output_dir, "screen").is_empty());
    }

    /// A borderline candidate climbs the ladder instead of dying at the bottom.
    #[test]
    fn a_borderline_candidate_is_carried_to_the_larger_samples() {
        let tmp = tempfile::tempdir().unwrap();
        let (creature, train) = two_hidden_paths(tmp.path());
        let cfg = ladder_cfg(
            creature,
            train,
            tmp.path().join("out"),
            Some("0.0025,0.01,0.05"),
        );
        // A thousandth behind the incumbent — well inside the 0.01 margin.
        let scorer = ScriptedScorer {
            baseline_score: 0.80,
            candidate_score: Some(0.799),
            ..ScriptedScorer::ok(0.80, 0.20)
        };
        establish_run(&cfg, &scorer).unwrap();
        let stages = journal_records(&cfg.output_dir, "screenStage");
        let reached: Vec<i64> = stages.iter().filter_map(|s| s["stage"].as_i64()).collect();
        assert!(
            reached.contains(&2),
            "an uncertain candidate must collect more evidence: {stages:?}"
        );
        let promotion = stages
            .iter()
            .find(|s| s["stage"] == 2)
            .expect("promotion stage");
        assert_eq!(promotion["outcome"], "promoted");
        assert_eq!(
            promotion["promoted"], 0,
            "it still loses to `--screen-threshold`, which the ladder never relaxes"
        );
    }
}
