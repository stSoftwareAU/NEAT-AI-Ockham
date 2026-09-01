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
    file_screens, file_verdicts, known_failures, oldest_screened_first, ranked_confirmed,
    replay_cap, screened_uuids,
};
use crate::promote::{
    BundleMember, FullConfig, FullOutcome, LocalWinner, REPLAY_PROBE_LIMIT, apply_available,
    evaluate_full, replay_plans,
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
}

impl CostModel {
    fn new(sample_rate: Option<f64>) -> Self {
        Self {
            sample_rate,
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
    fn screen_batch_ms(&self, candidates: usize) -> Option<f64> {
        // The incumbent baseline is scored alongside every batch.
        let creatures = candidates as f64 + 1.0;
        if let Some(per_creature) = self.screen_per_creature_ms {
            return Some(per_creature * creatures);
        }
        let full = self
            .full_per_creature_ms
            .or(self.baseline_per_creature_ms)?;
        let rate = match self.sample_rate {
            Some(rate) if rate > 0.0 => rate,
            _ => 1.0,
        };
        Some(full * rate * creatures)
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
    // The provenance the run opened with (Issue #75). `meta` sheds a cut
    // neuron's tags at every accept, so only this snapshot can still say what
    // left; the declaration is a set difference against it, never a counter.
    let opening_meta = meta.clone();
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
    let mut progress = crate::coverage::ScreenProgress::new(&screens);
    let mut sweep = fresh_sweep(
        &incumbent.creature,
        &activation,
        seed,
        ordering,
        unchecked_first,
        &screens,
    );
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
    // In-run state, deliberately not seeded from the cache: cross-run memory is
    // the learnings store's job (Issues #56, #57).
    let mut pool: Vec<BundleMember> = Vec::new();
    let mut cost = CostModel::new(config.screen_sample_rate);
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
        if reserving && !replay_done {
            log::info(&format!(
                "budget down to its last screening batch; standing the replay stage down so \
                 this run advances coverage (#77): {} newly screened so far",
                progress.count()
            ));
        }
        if !replay_done && !reserving {
            // A confirmed win on a GRQ-tagged uuid replays like any other (#63):
            // provenance records where a neuron came from, not that it earns its
            // place.
            // Accepted cuts and confirmed-but-unapplied ones, best measured
            // delta first (Issue #57): the largest-first plans below then drop
            // the weakest members rather than the most recently filed.
            let replayable: Vec<crate::learnings::ConfirmedWin> =
                ranked_confirmed(&known, &incumbent.creature, config.min_improvement)
                    .into_iter()
                    .filter(|c| !replay_skipped.contains(&c.uuid))
                    .take(replay_cap(replay_cfg.max))
                    .collect();
            let accepted_only = replayable.iter().filter(|c| c.accepted).count();
            let confirmed_only = replayable.len() - accepted_only;
            let wins: Vec<String> = replayable.into_iter().map(|c| c.uuid).collect();
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
                "replay: combining {} of {} known win(s) still on incumbent ({accepted_only} applied elsewhere, {confirmed_only} confirmed only)",
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
            // Replay is not sized by `--max-full`: after Issue #54 that flag
            // caps individual scoring in the search loop, and using it for a
            // replay probe was always a conflation. Nor is the replay cohort
            // trimmed to the wall clock (Issue #58): it runs first, with the
            // whole budget in front of it, and `MAX_REPLAY_PLANS` plus
            // `REPLAY_PROBE_LIMIT` already bound it.
            let probe_n = REPLAY_PROBE_LIMIT.min(applied.len());
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
                                        store.map(|_| screens.as_slice()),
                                        opening_hidden,
                                        &opening_meta,
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
                            store.map(|_| screens.as_slice()),
                            opening_hidden,
                            &opening_meta,
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
                stop_reason = "no-candidates".into();
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
        // is still read below for coverage and the pruned-provenance manifest.
        let (candidates, skips) =
            sweep.fill_batch_avoiding(&incumbent.creature, &activation, config.candidates, &avoid);
        pass_candidates += candidates.len();
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

        let batch_size = candidates.len();
        let sampled = match config.screen_sample_rate {
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
                        // The incumbent is scored alongside the batch.
                        cost.observe_screen(screen.screen_ms, batch_size + 1);
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
                            &mut progress,
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
                file_full_outcome(store, &mut known, &sampled, &full);
                update_pool(&mut pool, &sampled, &full, config.min_improvement);
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
                        store.map(|_| screens.as_slice()),
                        opening_hidden,
                        &opening_meta,
                    )?;
                    search_accepts += 1;
                    if let Some(max) = config.max_accepts
                        && search_accepts >= max
                    {
                        stop_reason = "max-accepts".into();
                        break;
                    }
                    sweep = fresh_sweep(
                        &incumbent.creature,
                        &activation,
                        seed.wrapping_add(accepts),
                        ordering,
                        unchecked_first,
                        &screens,
                    );
                    pass_candidates = 0;
                    // The pool was measured against the incumbent that just
                    // moved, so its members are candidates to re-try, not facts
                    // to re-apply; the ones the accept removed go now.
                    pool = standing_pool(&pool, &incumbent.creature, &activation);
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

    // What provenance this run deliberately spent (Issue #75). Declared on
    // every run with an output dir, empty list included: GRQ's check-in guard
    // forgives a missing neuron tag only when its uuid is listed here, and must
    // fail closed on an absent file, so "nothing pruned" and "no declaration"
    // can never be the same artefact.
    let declaration = opening_meta.pruned_provenance(&incumbent.creature);
    match crate::tags::write_pruned_provenance(&config.output_dir, &declaration) {
        Ok(()) => log::detail(&format!(
            "provenance: declared {} pruned tagged neuron(s) in {}",
            declaration.pruned.len(),
            crate::tags::PRUNED_PROVENANCE_FILE
        )),
        Err(e) => log::warn(&format!(
            "{} not written: {e}; GRQ will refuse this check-in, which is correct",
            crate::tags::PRUNED_PROVENANCE_FILE
        )),
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
        declaration.pruned.len(),
    );
    let newly_screened = progress.count();
    if let Some(warning) = crate::coverage::zero_progress_warning(newly_screened, cov.unchecked()) {
        log::warn(&warning);
    }

    // Coverage is only meaningful with the screen store behind it; without one
    // there is no coverage state to report, so nothing is journalled.
    if store.is_some() {
        log::info(&cov.summary());
        journal::append(
            &journal_path,
            &Event::Coverage {
                hidden: cov.hidden,
                tagged: cov.tagged,
                checkable: cov.checkable,
                checked: cov.checked,
                cut: cov.cut,
                tagged_cut: cov.tagged_cut,
            },
        )?;
        // The GRQ-facing commit-description artefacts (Issues #40, #59). A
        // write fault warns rather than failing the run, matching the learnings
        // cache: coverage is reporting, and reporting must never lose pruning.
        let report = crate::coverage::CoverageReport {
            coverage: cov,
            newly_screened,
            winners: winners.has_any().then_some(winners),
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
    ordering: crate::ordering::OrderingConfig,
    unchecked_first: bool,
    screens: &[Screened],
) -> Sweep {
    let mut sweep = Sweep::with_ordering(creature, activation, seed, ordering);
    if unchecked_first {
        prefer_unchecked(&mut sweep, screens, creature);
    }
    sweep
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
///
/// `progress` sees only the records that were actually filed (Issue #77): a
/// record a store fault dropped is not coverage this run added.
fn file_batch_screens(
    store: Option<&LearningsStore>,
    screens: &mut Vec<Screened>,
    progress: &mut crate::coverage::ScreenProgress,
    coverage: &[(&str, CandidateKind, ScreenOutcomeKind)],
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
                // Measured only inside the winning bundle, so its individual
                // contribution is unknown — never guess it from the bundle.
                full_delta: None,
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
    screens: Option<&[Screened]>,
    opening_hidden: usize,
    opening_meta: &CreatureMeta,
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
            // Against the opening meta, for the same reason the end-of-run
            // declaration is (#75): `meta` has already forgotten what left.
            opening_meta.pruned_provenance(&win.creature).pruned.len(),
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
        &config.stats_sample_spec(),
    )?;
    Ok(())
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
                    full_delta: None,
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

    /// Add a GRQ-style provenance tag to each named neuron of a creature file.
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
            })
            .unwrap();
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
        assert_eq!(run.stop_reason, "replay-accepts");
        let best = std::fs::read_to_string(cfg.output_dir.join("best.json")).unwrap();
        assert!(
            !best.contains("h_a"),
            "a tagged known win must be replayable: {best}"
        );
        assert!(
            !best.contains("discovered"),
            "the cut neuron's provenance tag must not survive it: {best}"
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
            max_accepts: Some(1),
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
                    corpus_identity: Some("fixture-corpus".into()),
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

    /// A second training corpus over the same widths, so only the **identity**
    /// differs from the one [`hidden_paths`] wrote.
    fn regenerated_corpus(tmp: &std::path::Path) -> std::path::PathBuf {
        let train = tmp.join("train-regenerated");
        std::fs::create_dir(&train).unwrap();
        write_bin_file(
            &train.join("0.bin"),
            &[(vec![3.0f32], vec![3.0f32]), (vec![4.0], vec![4.0])],
        )
        .unwrap();
        train
    }

    /// Issue #76: coverage must survive GRQ regenerating the corpus between
    /// runs. Two runs over one learnings root, second corpus different — the
    /// second run's coverage must be strictly greater than the first's.
    #[test]
    fn a_second_run_against_a_regenerated_corpus_advances_fleet_coverage() {
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

        let regenerated = regenerated_corpus(tmp.path());
        assert_ne!(
            corpus_identity(&regenerated),
            corpus_identity(&train),
            "the fixture must actually change the corpus identity"
        );
        let second = coverage_files_cfg(
            creature,
            regenerated,
            tmp.path().join("out-2"),
            Some(learnings_dir),
        );
        establish_run(&second, &ScriptedScorer::ok(0.50, 0.50)).unwrap();
        let second_cov = coverage_json(&second.output_dir);

        assert!(
            second_cov.checked > first_cov.checked,
            "coverage reset when the corpus identity changed: {} then {}",
            first_cov.checked,
            second_cov.checked
        );
        assert_eq!(second_cov.checked, 4, "both batches must be counted");
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
            text.contains("unchecked: 2 remaining (~1 run at 2/run)"),
            "{text}"
        );
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
        assert_eq!(cov.tagged, 4, "every hidden neuron carries provenance");
        assert_eq!(cov.checkable, 4, "tagged neurons stay in the denominator");
        assert_eq!(cov.checked, 2, "screened tagged UUIDs count as checked");
        assert_eq!(cov.percent(), 50.0);

        let text =
            std::fs::read_to_string(cfg.output_dir.join(crate::coverage::COVERAGE_TEXT_FILE))
                .unwrap();
        assert!(
            text.contains("tagged:    4 carry GRQ provenance, screened like any other"),
            "{text}"
        );
        // The old `skipped:` coverage line, not the `bundles: … skipped`
        // clause, which is a legitimate winner figure.
        assert!(!text.contains("skipped:"), "{text}");
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

    /// Read the declaration a run wrote beside `best.json` (Issue #75).
    fn declaration(out: &std::path::Path) -> crate::tags::PrunedProvenance {
        let text = std::fs::read_to_string(out.join(crate::tags::PRUNED_PROVENANCE_FILE))
            .expect("written");
        serde_json::from_str(&text).expect("valid JSON")
    }

    /// The declared UUIDs, uuid-ordered.
    fn declared_uuids(out: &std::path::Path) -> Vec<String> {
        declaration(out)
            .pruned
            .into_iter()
            .map(|p| p.uuid)
            .collect()
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

    /// Issue #75, the load-bearing case: a bundle accept that cuts one tagged
    /// and one untagged neuron declares exactly the tagged one, with its tags.
    #[test]
    fn a_bundle_accept_declares_only_the_tagged_uuid_it_cut() {
        let tmp = tempfile::tempdir().unwrap();
        let (creature, train) = two_hidden_paths(tmp.path());
        tag_neurons(&creature, &["h_a"]);
        let learnings_dir = tmp.path().join("learnings");
        known_wins(&learnings_dir, &train, &["h_a", "h_b"]);
        let cfg = replay_cfg(creature, train, tmp.path().join("out"), learnings_dir, None);

        let run = establish_run(&cfg, &improving_scorer()).unwrap();
        assert_eq!(run.stop_reason, "replay-accepts");
        let best = std::fs::read_to_string(cfg.output_dir.join("best.json")).unwrap();
        assert!(!best.contains("h_a") && !best.contains("h_b"), "{best}");

        let decl = declaration(&cfg.output_dir);
        assert_eq!(decl.version, crate::tags::PRUNED_PROVENANCE_VERSION);
        assert_eq!(
            decl.pruned,
            vec![crate::tags::PrunedNeuron {
                uuid: "h_a".into(),
                tags: vec!["discovered".into()],
            }],
            "the untagged cut spent no provenance and must not be declared"
        );
    }

    /// The over-inclusive direction, end to end: a tagged neuron still on the
    /// final incumbent must not be in the list, or the guard stops checking a
    /// tag that is still supposed to be there.
    #[test]
    fn a_surviving_tagged_neuron_stays_out_of_the_declaration() {
        let tmp = tempfile::tempdir().unwrap();
        let (creature, train) = two_hidden_paths(tmp.path());
        tag_neurons(&creature, &["h_a", "h_b"]);
        let learnings_dir = tmp.path().join("learnings");
        known_wins(&learnings_dir, &train, &["h_a"]);
        let cfg = replay_cfg(creature, train, tmp.path().join("out"), learnings_dir, None);

        establish_run(&cfg, &improving_scorer()).unwrap();
        let best = std::fs::read_to_string(cfg.output_dir.join("best.json")).unwrap();
        assert!(best.contains("h_b"), "h_b must survive the run: {best}");
        assert_eq!(
            declared_uuids(&cfg.output_dir),
            vec!["h_a".to_string()],
            "only the neuron that actually left may be declared"
        );
    }

    /// Point 3 of Issue #75: the file is written on every run with an output
    /// dir — no learnings dir, no accepts, nothing tagged pruned — because a
    /// guard cannot tell "nothing pruned" from "no declaration" otherwise.
    #[test]
    fn a_run_that_pruned_nothing_tagged_still_declares_an_empty_list() {
        let tmp = tempfile::tempdir().unwrap();
        let uuids = ["h_a", "h_b", "h_c", "h_d"];
        let (creature, train) = hidden_paths(tmp.path(), &uuids);
        tag_neurons(&creature, &uuids);
        let cfg = coverage_files_cfg(creature, train, tmp.path().join("out"), None);

        let run = establish_run(&cfg, &ScriptedScorer::ok(0.50, 0.50)).unwrap();
        assert_eq!(run.accepts, 0, "a flat scorer accepts nothing");
        let decl = declaration(&cfg.output_dir);
        assert_eq!(decl.version, crate::tags::PRUNED_PROVENANCE_VERSION);
        assert!(
            decl.pruned.is_empty(),
            "every tagged neuron survived: {decl:?}"
        );
    }

    /// A blocked declaration is reporting, and reporting must never lose
    /// pruning — the run completes and `best.json` still lands.
    ///
    /// The same config runs twice: once with the path blocked, once clean. The
    /// clean run is the control — it proves the blocked run really did attempt
    /// the write it could not make, rather than never reaching it.
    #[test]
    fn a_blocked_declaration_write_warns_rather_than_failing_the_run() {
        let tmp = tempfile::tempdir().unwrap();
        let (creature, train) = hidden_paths(tmp.path(), &["h_a", "h_b", "h_c", "h_d"]);
        let blocked = tmp.path().join("blocked");
        // A directory where the declaration belongs: the write cannot succeed.
        std::fs::create_dir_all(blocked.join(crate::tags::PRUNED_PROVENANCE_FILE)).unwrap();
        let cfg = coverage_files_cfg(
            creature.clone(),
            train.clone(),
            blocked,
            Some(tmp.path().join("learnings")),
        );

        let run = establish_run(&cfg, &ScriptedScorer::ok(0.50, 0.50)).unwrap();
        assert_eq!(run.optimisation, "complete");
        assert!(cfg.output_dir.join("best.json").exists());
        assert!(
            cfg.output_dir
                .join(crate::tags::PRUNED_PROVENANCE_FILE)
                .is_dir(),
            "the blocker is still there, so nothing was declared"
        );
        assert!(
            cfg.output_dir
                .join(crate::coverage::COVERAGE_TEXT_FILE)
                .exists(),
            "the rest of the reporting still runs"
        );

        let control = coverage_files_cfg(
            creature,
            train,
            tmp.path().join("clean"),
            Some(tmp.path().join("learnings-control")),
        );
        establish_run(&control, &ScriptedScorer::ok(0.50, 0.50)).unwrap();
        assert!(
            control
                .output_dir
                .join(crate::tags::PRUNED_PROVENANCE_FILE)
                .is_file(),
            "the unblocked control run declares, so the blocked run tried to"
        );
    }

    /// The other removal path: a sweep accept, with no replayable known win.
    /// The declaration is a set difference over the final incumbent, so it
    /// cannot care which path cut the neuron — this pins that.
    #[test]
    fn a_sweep_accept_declares_the_tagged_neuron_it_cut() {
        let tmp = tempfile::tempdir().unwrap();
        let (creature, train) = two_hidden_paths(tmp.path());
        tag_neurons(&creature, &["h_a", "h_b"]);
        let cfg = OckhamConfig {
            creature,
            training_data: train,
            output_dir: tmp.path().join("out"),
            timeout: Duration::from_secs(30),
            max_experiments: Some(4),
            max_accepts: Some(1),
            seed: Some(1),
            candidates: 8,
            screen_sample_rate: Some(0.5),
            ..OckhamConfig::default()
        };

        let run = establish_run(&cfg, &improving_scorer()).unwrap();
        assert!(run.accepts >= 1, "stop={}", run.stop_reason);
        let best = std::fs::read_to_string(cfg.output_dir.join("best.json")).unwrap();
        let cut: Vec<String> = ["h_a", "h_b"]
            .iter()
            .filter(|u| !best.contains(**u))
            .map(|u| (*u).to_string())
            .collect();
        assert!(!cut.is_empty(), "the sweep must have cut something: {best}");
        assert_eq!(
            declared_uuids(&cfg.output_dir),
            cut,
            "a search accept declares exactly what it removed"
        );
    }

    /// The count reaches the commit description and `coverage.json` too, so the
    /// fleet history shows provenance being spent (Issue #75, point 5).
    #[test]
    fn the_coverage_artefacts_count_the_tagged_neuron_the_run_cut() {
        let tmp = tempfile::tempdir().unwrap();
        let (creature, train) = two_hidden_paths(tmp.path());
        tag_neurons(&creature, &["h_a"]);
        let learnings_dir = tmp.path().join("learnings");
        known_wins(&learnings_dir, &train, &["h_a"]);
        let cfg = replay_cfg(creature, train, tmp.path().join("out"), learnings_dir, None);

        establish_run(&cfg, &improving_scorer()).unwrap();
        let json =
            std::fs::read_to_string(cfg.output_dir.join(crate::coverage::COVERAGE_JSON_FILE))
                .unwrap();
        let cov: Coverage = serde_json::from_str(&json).unwrap();
        assert_eq!(cov.tagged_cut, 1);
        assert_eq!(cov.cut, 1);

        let text =
            std::fs::read_to_string(cfg.output_dir.join(crate::coverage::COVERAGE_TEXT_FILE))
                .unwrap();
        assert!(
            text.contains("declared:  1 tagged neuron cut, listed in pruned-provenance.json"),
            "{text}"
        );
        let journal = std::fs::read_to_string(cfg.output_dir.join("experiments.jsonl")).unwrap();
        assert!(journal.contains(r#""tagged_cut":1"#), "{journal}");
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
            max_accepts: Some(1),
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
        assert_eq!(run.accepts, 1);
        let best = cfg.output_dir.join("best.json");
        let tag = ockham_tag(&best);
        assert!(tag.starts_with("🪒 Ockham"), "{tag}");
        assert!(tag.contains(" · checked "), "{tag}");
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
            max_accepts: Some(1),
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
        assert_eq!(run.accepts, 1);
        let tag = ockham_tag(&cfg.output_dir.join("best.json"));
        assert!(tag.starts_with("🪒 Ockham"), "{tag}");
        assert!(
            !tag.contains("checked"),
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
            max_accepts: Some(1),
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
        let mut cost = CostModel::new(Some(0.01));
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
        let mut cost = CostModel::new(Some(0.01));
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
            max_accepts: Some(1),
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
        // h_b was looked at longest ago, so it is the stalest of the two.
        for (uuid, unix_secs) in [("h_a", 200u64), ("h_b", 100)] {
            store
                .append_screen(&Screened {
                    version: crate::learnings::SCREENS_FORMAT_VERSION,
                    uuid: uuid.into(),
                    kind: "identity".into(),
                    outcome: ScreenOutcomeKind::Loser,
                    unix_secs,
                    host: "seed".into(),
                    corpus_identity: None,
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
        assert_eq!(run.newly_screened, 0);
        assert!(
            journal_records(&cfg.output_dir, "sweepRestart").is_empty(),
            "a barren pass must stop, not restart into the same nothing"
        );
    }

    /// Issue #77 point 3, the sizing rules, unit by unit. A measured screen is
    /// the best answer; a full-corpus estimate scaled by the sample rate is the
    /// fallback; screening disabled means the batch *is* a cohort.
    #[test]
    fn the_screening_reserve_is_sized_at_one_batch() {
        let mut cost = CostModel::new(Some(0.05));
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

        let mut off = CostModel::new(None);
        off.observe_baseline(200);
        assert_eq!(
            off.screen_batch_ms(9),
            Some(200.0 * 10.0),
            "with screening disabled the check is the full cohort itself"
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
        let mut cost = CostModel::new(Some(0.05));
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
            text.contains("progress:  2 newly screened this run"),
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
            text.contains("progress:  0 newly screened this run"),
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
}
