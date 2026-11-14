//! The additivity benchmark driver (Issue #260).
//!
//! For each parent the driver proposes every single hidden-neuron cut, caches
//! its outputs on one deterministic audit core, then for every (budget, seed)
//! builds a pool of bundles, scores every unique bundle with the full-corpus
//! scorer (the audit), and lets each arm send its own top picks to a separate
//! full-scorer call. Predictions only order the picks; the best full score among
//! an arm's picks is its selection. Nothing here accepts a creature.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::fs;
use std::ops::ControlFlow;
use std::path::{Path, PathBuf};

use neat_core::training_data::TrainingDataConfig;
use neat_core::{CreatureExport, compile_creature, creature_to_json};

use super::bench_report::{
    self, AUTHORITY, ArmCost, ArmRow, AuditCost, BenchReport, BenchRow, BundleRow, CacheStats,
    DuplicateBundle, FailedProposal, ManifestEcho, ParentReport, Pick, REPORT_FORMAT_VERSION,
    Refusal, evidence_for,
};
use super::cache::{
    CACHE_FORMAT_VERSION, CacheKey, CacheRejection, OutputCache, core_pin, cut_entry_name,
    dataset_digest, ranges_digest,
};
use super::manifest::BenchManifest;
use super::predict::{LossSum, PredictError, add_delta, additive_loss_change, chunk_loss_sum};
use super::rank::{self, ARMS, Arm, AuditMetrics};
use crate::ablation::StructureSnapshot;
use crate::baseline::{AuthoritativeBaseline, establish_baseline};
use crate::clock::Clock;
use crate::corpus::{CorpusInfo, RecordChunk, RecordRange, corpus_info, for_each_selected_chunk};
use crate::incumbent::{Incumbent, load_incumbent, sha256_hex};
use crate::ordering::SplitMix64;
use crate::promote::{BundleMember, bundle_plans};
use crate::prune::{GroupMember, prune_hidden_group};
use crate::scorer::{DirectoryScorer, ScoreResult, ScorerMode, check_finite};
use crate::stats::{
    DEFAULT_CHUNK_RECORDS, DEFAULT_SAMPLE_RECORDS, SampleSpec, compute_activation_stats,
};
use crate::sweep::CandidateKind;

/// Meaning of a single-cut output entry, part of every cache key. Changing the
/// cut recipe must change this string so old entries are never served.
pub const CUT_SEMANTICS: &str = "prune_hidden_group single member, mean fold from stats sample tag";

/// Cache entry holding the audit core's targets.
const TARGETS_ENTRY: &str = "targets";

/// Cache entry holding the parent's outputs on the audit core.
const Y0_ENTRY: &str = "y0";

/// Stream tag for pool draws, keeping them apart from the random arm.
const POOL_STREAM: u64 = 1;

/// Stream tag for the random arm's permutation.
const RANDOM_STREAM: u64 = 2;

/// Random draws allowed per wanted pool bundle before the pool is left short.
const ATTEMPTS_PER_BUNDLE: usize = 50;

/// Chunk-sized buffers the bench keeps alive (targets, y0, build buffer, member).
/// The predicted-output accumulator is one cache entry, counted apart in
/// `ram_bound_bytes`.
const SCRATCH_BUFFERS: u64 = 4;

/// Run the additivity benchmark described by `manifest`.
///
/// Parents are processed in manifest order. `workspace` holds the scoring
/// directories, which are removed after use; the parent files are only read.
///
/// # Errors
///
/// A manifest or parent that cannot be read, a corpus that cannot be streamed,
/// or a full-scorer call that breaks its contract: missing or non-finite
/// results, or a reported cost or record count that disagrees with the manifest
/// or corpus. A cache rejection is never an error: it means no estimate.
pub fn run_bench(
    manifest: &BenchManifest,
    scorer: &dyn DirectoryScorer,
    clock: &dyn Clock,
    workspace: &Path,
) -> Result<BenchReport, String> {
    manifest.validate()?;
    fs::create_dir_all(workspace)
        .map_err(|e| format!("workspace '{}': {e}", workspace.display()))?;
    let mut parents = Vec::with_capacity(manifest.parents.len());
    let mut rows = Vec::new();
    for (index, path) in manifest.parents.iter().enumerate() {
        let outcome = ParentRun::prepare(manifest, scorer, clock, workspace, index, path).and_then(
            |mut run| {
                let mut parent_rows = Vec::new();
                run.rows(&mut parent_rows)?;
                Ok((run.report(), parent_rows))
            },
        );
        let _ = fs::remove_dir_all(workspace.join(format!("parent-{index}")));
        let (report, parent_rows) = outcome?;
        parents.push(report);
        rows.extend(parent_rows);
    }
    let representative = manifest.representative();
    let (aggregate, cases) = bench_report::aggregate(&rows, &manifest.budgets)?;
    Ok(BenchReport {
        format_version: REPORT_FORMAT_VERSION,
        manifest: ManifestEcho::of(manifest),
        representative,
        evidence: evidence_for(parents.len(), representative),
        authority: AUTHORITY.to_string(),
        parents,
        rows,
        aggregate,
        cases_beating_cached_output: cases,
    })
}

/// Why one entry could not be built or read. Every variant except `Stream`
/// means "no estimate for this cut".
enum Fault {
    Cache(CacheRejection),
    Predict(PredictError),
    Stream(String),
}

impl From<CacheRejection> for Fault {
    fn from(rejection: CacheRejection) -> Self {
        Self::Cache(rejection)
    }
}

impl From<PredictError> for Fault {
    fn from(error: PredictError) -> Self {
        Self::Predict(error)
    }
}

impl Fault {
    /// Stable kind name used in the report's rejection counts.
    fn kind(&self) -> &'static str {
        match self {
            Self::Cache(CacheRejection::NonFinite { .. })
            | Self::Predict(PredictError::NonFinite { .. }) => "nonFinite",
            Self::Cache(CacheRejection::Mismatched { .. })
            | Self::Predict(PredictError::Mismatched { .. }) => "mismatched",
            Self::Cache(CacheRejection::OverLimit { .. }) => "overLimit",
            Self::Cache(CacheRejection::Missing) => "missing",
            Self::Cache(CacheRejection::Io { .. }) => "io",
            Self::Stream(_) => "stream",
        }
    }

    fn describe(&self) -> String {
        match self {
            Self::Cache(rejection) => rejection.to_string(),
            Self::Predict(error) => error.to_string(),
            Self::Stream(message) => message.clone(),
        }
    }
}

/// What a cache entry is computed from.
enum Source<'c> {
    /// The audit core's targets, copied from the corpus.
    Targets,
    /// A creature's outputs on the audit core.
    Network(&'c CreatureExport),
}

/// The audit core: which records, how wide, and in what chunks.
struct Dataset {
    dir: PathBuf,
    cfg: TrainingDataConfig,
    ranges: Vec<RecordRange>,
    core_records: u64,
    chunk: usize,
}

/// Cache handle plus the bench-owned chunk buffers and counters.
struct Store {
    cache: OutputCache,
    tally: Tally,
    scratch: Scratch,
}

/// Counters for one parent's cache traffic.
#[derive(Default)]
struct Tally {
    hits: u64,
    builds: u64,
    rejections: BTreeMap<String, u64>,
    peak_bytes: u64,
    build_ms: u64,
    read_ms: u64,
}

/// The four chunk-sized buffers reused for every read and build, and the
/// predicted-output accumulator reused for every bundle.
struct Scratch {
    target: Vec<f32>,
    y0: Vec<f32>,
    acc: Vec<f32>,
    member: Vec<f32>,
    /// One cache entry's values, `core_records × outputs`. Grown once per parent
    /// on first use, then reused.
    predicted: Vec<f32>,
}

impl Scratch {
    fn new(len: usize) -> Self {
        Self {
            target: vec![0.0; len],
            y0: vec![0.0; len],
            acc: vec![0.0; len],
            member: vec![0.0; len],
            predicted: Vec::new(),
        }
    }
}

/// One proposable single cut. Its creature is not kept: only the cache entry
/// built from it outlives the proposal.
struct Single {
    uuid: String,
    mean: Option<f64>,
    entry: String,
    /// Mean loss of the single's outputs, when its entry is readable.
    loss: Option<f64>,
}

/// A materialised, unique pool bundle.
struct Scored {
    members: Vec<usize>,
    member_uuids: Vec<String>,
    creature: CreatureExport,
    before: StructureSnapshot,
    after: StructureSnapshot,
    requested: Vec<String>,
    cascade: Vec<String>,
    digest: String,
}

/// Everything the bench holds for one parent while its rows are computed.
struct ParentRun<'a> {
    manifest: &'a BenchManifest,
    scorer: &'a dyn DirectoryScorer,
    clock: &'a dyn Clock,
    index: usize,
    name: String,
    incumbent: Incumbent,
    parent_json: String,
    corpus: CorpusInfo,
    /// Record count of `validationData`, read once; `None` without held-out data.
    validation_records: Option<u64>,
    baseline: AuthoritativeBaseline,
    workdir: PathBuf,
    hidden: HashSet<String>,
    data: Dataset,
    store: Store,
    singles: Vec<Single>,
    failed: Vec<FailedProposal>,
    base_loss: Option<f64>,
    estimate_blocked: Option<String>,
    singles_ms: u64,
    setup_build_ms: u64,
    setup_read_ms: u64,
    setup_builds: u64,
}

impl<'a> ParentRun<'a> {
    /// Baseline, activation stats, audit core, cache handle and singles.
    fn prepare(
        manifest: &'a BenchManifest,
        scorer: &'a dyn DirectoryScorer,
        clock: &'a dyn Clock,
        workspace: &Path,
        index: usize,
        path: &Path,
    ) -> Result<Self, String> {
        let incumbent =
            load_incumbent(path).map_err(|e| format!("parent '{}': {e}", path.display()))?;
        let name = path
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default();
        let cfg = TrainingDataConfig::new(incumbent.creature.input, incumbent.creature.output);
        let corpus = corpus_info(&manifest.training_data, &cfg)?;
        let validation_records = match &manifest.validation_data {
            Some(dir) => Some(corpus_info(dir, &cfg)?.record_count),
            None => None,
        };
        let workdir = workspace.join(format!("parent-{index}"));
        fs::create_dir_all(&workdir).map_err(|e| format!("{}: {e}", workdir.display()))?;
        let baseline = establish_baseline(
            clock,
            &incumbent,
            &manifest.training_data,
            &corpus,
            scorer,
            &manifest.scorer.args,
            &workdir,
        )?;
        check_report(
            "baseline",
            baseline.cost_name.as_deref(),
            baseline.record_count,
            &manifest.cost,
            corpus.record_count,
        )?;

        // A fresh sample every run, with the same spec an ordinary run uses. The
        // single cuts fold these means in, so they are never read from a cache.
        let sample = SampleSpec::with_max_records(DEFAULT_SAMPLE_RECORDS);
        let activation = compute_activation_stats(
            &incumbent.creature,
            &incumbent.checksum,
            &manifest.training_data,
            &corpus,
            DEFAULT_CHUNK_RECORDS,
            &sample,
        )?;
        let means: HashMap<String, f64> = activation
            .neurons
            .iter()
            .map(|n| (n.uuid.clone(), n.mean))
            .collect();
        // The exact members handed to the pruning step, in hidden-neuron order.
        let hidden_members: Vec<GroupMember> = incumbent
            .creature
            .neurons
            .iter()
            .filter(|n| n.neuron_type == "hidden")
            .map(|n| GroupMember {
                uuid: n.uuid.clone(),
                mean: means.get(&n.uuid).copied(),
            })
            .collect();

        let wanted = ((manifest.core_fraction * corpus.record_count as f64).ceil() as u64).max(1);
        let ranges = SampleSpec::with_max_records(wanted).plan(&corpus);
        let core_records: u64 = ranges.iter().map(|r| r.len).sum();
        if core_records == 0 {
            return Err(format!("parent '{name}': the audit core holds no records"));
        }
        let chunk = manifest.limits.chunk_records;
        let (digest, visited) = dataset_digest(&manifest.training_data, &cfg, &ranges, chunk)?;
        if visited != core_records {
            return Err(format!(
                "parent '{name}': audit core visited {visited} records, planned {core_records}"
            ));
        }
        let key = CacheKey {
            format_version: CACHE_FORMAT_VERSION,
            parent_digest: incumbent.checksum.clone(),
            dataset_digest: digest,
            corpus_identity: corpus.identity.clone(),
            record_ranges_digest: ranges_digest(&ranges),
            records: core_records,
            input_count: cfg.num_inputs,
            output_count: cfg.num_outputs,
            cut_semantics: format!("{CUT_SEMANTICS} {}", sample.tag()),
            stats_digest: stats_digest(&hidden_members),
            core_pin: core_pin()?,
            ockham_version: env!("CARGO_PKG_VERSION").to_string(),
            scorer_identity: scorer.identity(),
            cost: manifest.cost.clone(),
            scorer_args: manifest.scorer.args.clone(),
        };
        let cache = OutputCache::open(
            &manifest.cache_dir.join("outputs"),
            key,
            manifest.limits.max_cache_bytes,
        )?;

        let parent_json = creature_to_json(&incumbent.creature).map_err(|e| e.to_string())?;
        let width = cfg.num_outputs;
        let mut run = Self {
            manifest,
            scorer,
            clock,
            index,
            name,
            incumbent,
            parent_json,
            corpus,
            validation_records,
            baseline,
            workdir,
            hidden: hidden_members.iter().map(|m| m.uuid.clone()).collect(),
            data: Dataset {
                dir: manifest.training_data.clone(),
                cfg,
                ranges,
                core_records,
                chunk,
            },
            store: Store {
                cache,
                tally: Tally::default(),
                scratch: Scratch::new(chunk * width),
            },
            singles: Vec::new(),
            failed: Vec::new(),
            base_loss: None,
            estimate_blocked: None,
            singles_ms: 0,
            setup_build_ms: 0,
            setup_read_ms: 0,
            setup_builds: 0,
        };
        run.build_setup()?;

        // Proposals in creature order. Each cut's entry is built, and its loss
        // read, straight after its proposal, so its candidate creature is dropped
        // before the next cut is proposed.
        let mut singles_ms = 0;
        for member in &hidden_members {
            let started = clock.now();
            let proposal =
                prune_hidden_group(&run.incumbent.creature, std::slice::from_ref(member));
            singles_ms += clock.ms_since(started);
            match proposal {
                Ok(candidate) => run.add_single(member, candidate.creature)?,
                Err(refusal) => run.failed.push(FailedProposal {
                    uuid: member.uuid.clone(),
                    reason: refusal.to_string(),
                }),
            }
        }
        run.singles_ms = singles_ms;
        run.setup_build_ms = run.store.tally.build_ms;
        run.setup_read_ms = run.store.tally.read_ms;
        run.setup_builds = run.store.tally.builds;
        Ok(run)
    }

    /// Build the targets and the parent's outputs, then read the parent's loss.
    /// A rejection leaves the whole parent without estimates.
    fn build_setup(&mut self) -> Result<(), String> {
        let clock = self.clock;
        let data = &self.data;
        match build_entry(&mut self.store, data, clock, TARGETS_ENTRY, Source::Targets) {
            Ok(()) => {}
            Err(Fault::Stream(message)) => return Err(message),
            Err(fault) => {
                self.estimate_blocked = Some(format!("{TARGETS_ENTRY}: {}", fault.describe()));
            }
        }
        if self.estimate_blocked.is_none() {
            let parent = &self.incumbent.creature;
            match build_entry(
                &mut self.store,
                data,
                clock,
                Y0_ENTRY,
                Source::Network(parent),
            ) {
                Ok(()) => {}
                Err(Fault::Stream(message)) => return Err(message),
                Err(fault) => {
                    self.estimate_blocked = Some(format!("{Y0_ENTRY}: {}", fault.describe()));
                }
            }
        }
        if self.estimate_blocked.is_none() {
            let read = mean_loss(&mut self.store, data, clock, Y0_ENTRY);
            match settle(&mut self.store.tally, read)? {
                Some(loss) if loss.is_finite() => self.base_loss = Some(loss),
                _ => {
                    self.estimate_blocked =
                        Some(format!("{Y0_ENTRY}: parent loss is not readable"));
                }
            }
        }
        Ok(())
    }

    /// Record one proposable single: build its cache entry and read its loss.
    /// The creature is taken by value and dropped on return. A rejection leaves
    /// that cut without an estimate.
    fn add_single(&mut self, member: &GroupMember, creature: CreatureExport) -> Result<(), String> {
        let clock = self.clock;
        let entry = cut_entry_name(&member.uuid);
        let mut loss = None;
        if self.base_loss.is_some() {
            match build_entry(
                &mut self.store,
                &self.data,
                clock,
                &entry,
                Source::Network(&creature),
            ) {
                Ok(()) => {
                    let read = mean_loss(&mut self.store, &self.data, clock, &entry);
                    if let Some(value) = settle(&mut self.store.tally, read)?
                        && value.is_finite()
                    {
                        loss = Some(value);
                    }
                }
                Err(Fault::Stream(message)) => return Err(message),
                Err(_) => {}
            }
        }
        self.singles.push(Single {
            uuid: member.uuid.clone(),
            mean: member.mean,
            entry,
            loss,
        });
        Ok(())
    }

    /// Run every (budget, seed) row for this parent.
    fn rows(&mut self, out: &mut Vec<BenchRow>) -> Result<(), String> {
        let manifest = self.manifest;
        for (budget_index, &budget) in manifest.budgets.iter().enumerate() {
            let k = requested_removals(budget, self.hidden.len());
            for &seed in &manifest.seeds {
                out.push(self.run_row(budget_index, budget, k, seed)?);
            }
        }
        Ok(())
    }

    /// One (budget, seed) row: pool, materialisation, audit, arms and validation.
    fn run_row(
        &mut self,
        budget_index: usize,
        budget: f64,
        k: usize,
        seed: u64,
    ) -> Result<BenchRow, String> {
        let manifest = self.manifest;
        let clock = self.clock;
        let mut row = BenchRow {
            parent: self.name.clone(),
            parent_index: self.index,
            budget,
            budget_index,
            seed,
            requested_removals: k,
            feasible: false,
            reason: None,
            eligible: self.singles.len(),
            pool_requested: 0,
            pool_size: 0,
            duplicates: Vec::new(),
            refused: Vec::new(),
            bundles: Vec::new(),
            audit: AuditCost {
                full_calls: 0,
                record_evaluations: 0,
                ms: 0,
            },
            pool_materialisation_ms: 0,
            estimate_ms: 0,
            validation_parent_score: None,
            arms: Vec::new(),
        };
        if k == 0 {
            row.reason = Some("parent has no hidden neurons".to_string());
            return Ok(row);
        }
        if row.eligible < k {
            row.reason = Some(format!(
                "{} proposable single(s) < {k} requested removal(s)",
                row.eligible
            ));
            return Ok(row);
        }

        // Pool: Ockham's ranked prefix first, then seeded distinct draws.
        let ranked = self.rank_singles()?;
        let started = clock.now();
        let pool_seed = derive_seed(&[seed, self.index as u64, budget_index as u64, POOL_STREAM]);
        let pool = build_pool(row.eligible, k, manifest.audit_pool, pool_seed);
        row.pool_requested = pool.len();
        let mut scored: Vec<Scored> = Vec::new();
        let mut digests: HashMap<String, usize> = HashMap::new();
        for positions in &pool {
            let members: Vec<usize> = positions.iter().map(|&p| ranked[p]).collect();
            let member_uuids: Vec<String> = members
                .iter()
                .map(|&i| self.singles[i].uuid.clone())
                .collect();
            let group: Vec<GroupMember> = members
                .iter()
                .map(|&i| GroupMember {
                    uuid: self.singles[i].uuid.clone(),
                    mean: self.singles[i].mean,
                })
                .collect();
            match prune_hidden_group(&self.incumbent.creature, &group) {
                Err(refusal) => row.refused.push(Refusal {
                    members: member_uuids,
                    reason: refusal.to_string(),
                }),
                Ok(candidate) => {
                    row.pool_size += 1;
                    let json = creature_to_json(&candidate.creature).map_err(|e| e.to_string())?;
                    let digest = sha256_hex(json.as_bytes());
                    if let Some(&first) = digests.get(&digest) {
                        row.duplicates.push(DuplicateBundle {
                            members: member_uuids,
                            duplicate_of: first,
                            creature_digest: digest,
                        });
                    } else {
                        digests.insert(digest.clone(), scored.len());
                        // Requested neurons first; the cascade and the neurons
                        // folded into constant support are both what the cut
                        // stranded, so both are named.
                        let cascade = candidate.detail.cascade_uuids();
                        scored.push(Scored {
                            members,
                            member_uuids,
                            creature: candidate.creature,
                            before: candidate.before,
                            after: candidate.after,
                            requested: candidate.detail.requested_neurons,
                            cascade,
                            digest,
                        });
                    }
                }
            }
        }
        row.pool_materialisation_ms = clock.ms_since(started);
        if scored.is_empty() {
            row.reason = Some("no pool bundle could be materialised".to_string());
            return Ok(row);
        }

        // Estimates, one per unique bundle.
        let started = clock.now();
        let mut additive = Vec::with_capacity(scored.len());
        let mut cached = Vec::with_capacity(scored.len());
        let mut additive_ms = 0;
        let mut cached_read_ms = 0;
        for bundle in &scored {
            let additive_started = clock.now();
            additive.push(self.additive_estimate(&bundle.members)?);
            additive_ms += clock.ms_since(additive_started);
            let read_started = clock.now();
            cached.push(self.cached_estimate(&bundle.members)?);
            cached_read_ms += clock.ms_since(read_started);
        }
        row.estimate_ms = clock.ms_since(started);

        // Audit: one full-scorer call over every unique bundle.
        let n = scored.len();
        let parent_score = self.baseline.score;
        let parent_error = self.baseline.error;
        let started = clock.now();
        let refs: Vec<(usize, &CreatureExport)> = scored
            .iter()
            .enumerate()
            .map(|(i, s)| (i, &s.creature))
            .collect();
        let audit_results = self.score(
            &format!("audit-b{budget_index}-s{seed}"),
            &manifest.training_data,
            self.corpus.record_count,
            &refs,
        )?;
        row.audit = AuditCost {
            full_calls: 1,
            record_evaluations: n as u64 * self.corpus.record_count,
            ms: clock.ms_since(started),
        };
        let truth: Vec<f64> = (0..n)
            .map(|i| audit_results[&bundle_name(i)].score)
            .collect();
        let best_truth = truth.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        // Full-corpus error reduction, in the prediction's units: the actual side
        // of the promotion audits. Ranking metrics stay on `truth`.
        let actual_gain: Vec<f64> = (0..n)
            .map(|i| parent_error - audit_results[&bundle_name(i)].error)
            .collect();
        row.bundles = scored
            .iter()
            .enumerate()
            .map(|(i, s)| {
                let result = &audit_results[&bundle_name(i)];
                let hidden_removed = s
                    .before
                    .hidden_neurons
                    .saturating_sub(s.after.hidden_neurons);
                BundleRow {
                    index: i,
                    members: s.member_uuids.clone(),
                    overlaps: s
                        .member_uuids
                        .iter()
                        .filter(|u| !s.requested.contains(u))
                        .cloned()
                        .collect(),
                    missing: s
                        .member_uuids
                        .iter()
                        .filter(|u| !self.hidden.contains(*u))
                        .cloned()
                        .collect(),
                    cascade_neurons: s.cascade.clone(),
                    hidden_removed,
                    synapses_removed: s.before.synapses.saturating_sub(s.after.synapses),
                    growth_units_removed: s.before.growth_units - s.after.growth_units,
                    meets_budget: hidden_removed >= k,
                    creature_digest: s.digest.clone(),
                    score: result.score,
                    error: result.error,
                    complexity_penalty: result.complexity_penalty,
                    truth_delta: result.score - parent_score,
                    additive_loss: additive[i],
                    cached_output: cached[i],
                }
            })
            .collect();

        // Arms: rank, audit the ranking, then search with the arm's own picks.
        let index = self.index as u64;
        let arm_seed = |stream: u64| derive_seed(&[seed, index, budget_index as u64, stream]);
        let needs_singles_setup = |arm: Arm| arm != Arm::Random;
        let mut arms = Vec::with_capacity(ARMS.len());
        for arm in ARMS {
            let started = clock.now();
            let ranking = match arm {
                Arm::Current => rank::rank_current(n),
                Arm::Random => rank::rank_random(n, arm_seed(RANDOM_STREAM)),
                Arm::AdditiveLoss => rank::rank_by_estimate(&additive),
                Arm::CachedOutput => rank::rank_by_estimate(&cached),
            };
            let ranking_ms = clock.ms_since(started);
            // `predicts()` is true only for the two predictor arms, so the `_`
            // branch is the additive one.
            let predicted: Option<&[Option<f64>]> =
                arm.predicts().then_some(if arm == Arm::CachedOutput {
                    cached.as_slice()
                } else {
                    additive.as_slice()
                });
            let metrics: AuditMetrics = rank::audit(
                &ranking,
                &truth,
                predicted,
                &actual_gain,
                manifest.min_improvement,
                manifest.top_k,
            )?;
            let inference_ms = match arm {
                Arm::AdditiveLoss => additive_ms,
                Arm::CachedOutput => cached_read_ms,
                Arm::Current | Arm::Random => 0,
            };
            let inference_latency_ms = predicted.and_then(|values| {
                let estimated = values.iter().flatten().filter(|v| v.is_finite()).count();
                (estimated > 0).then(|| inference_ms as f64 / estimated as f64)
            });

            let picks_n = ranking.len().min(manifest.full_evaluations);
            let picks_idx: Vec<usize> = ranking[..picks_n].to_vec();
            let started = clock.now();
            let picked: Vec<(usize, &CreatureExport)> = picks_idx
                .iter()
                .map(|&i| (i, &scored[i].creature))
                .collect();
            let search = self.score(
                &format!("search-{}-b{budget_index}-s{seed}", arm.label()),
                &manifest.training_data,
                self.corpus.record_count,
                &picked,
            )?;
            let scorer_ms = clock.ms_since(started);
            let picks: Vec<Pick> = picks_idx
                .iter()
                .map(|&i| Pick {
                    bundle: i,
                    score: search[&bundle_name(i)].score,
                })
                .collect();
            let best = picks.iter().fold(None::<&Pick>, |best, pick| match best {
                None => Some(pick),
                Some(b) => {
                    if pick.score > b.score || (pick.score == b.score && pick.bundle < b.bundle) {
                        Some(pick)
                    } else {
                        Some(b)
                    }
                }
            });
            let selected = best.map(|p| p.bundle);
            let chosen = selected.map(|i| &search[&bundle_name(i)]);
            let cost_read = if needs_singles_setup(arm) {
                self.setup_read_ms
                    + if arm == Arm::CachedOutput {
                        cached_read_ms
                    } else {
                        0
                    }
            } else {
                0
            };
            let mut cost = ArmCost {
                single_proposals: if needs_singles_setup(arm) {
                    self.singles_ms
                } else {
                    0
                },
                cache_build: if needs_singles_setup(arm) {
                    self.setup_build_ms
                } else {
                    0
                },
                cache_read_inference: cost_read,
                pool_materialisation: row.pool_materialisation_ms,
                ranking: ranking_ms,
                scorer: scorer_ms,
                total: 0,
            };
            cost.total = cost.single_proposals
                + cost.cache_build
                + cost.cache_read_inference
                + cost.pool_materialisation
                + cost.ranking
                + cost.scorer;
            arms.push(ArmRow {
                arm,
                ranking,
                audit: metrics,
                picks,
                selected_bundle: selected,
                selected_score: chosen.map(|r| r.score),
                selected_error: chosen.map(|r| r.error),
                selected_complexity_penalty: chosen.map(|r| r.complexity_penalty),
                selected_delta: chosen.map(|r| r.score - parent_score),
                selected_regret: selected.map(|i| best_truth - truth[i]),
                selected_hidden_removed: selected.map(|i| row.bundles[i].hidden_removed),
                selected_synapses_removed: selected.map(|i| row.bundles[i].synapses_removed),
                validation_score: None,
                validation_error: None,
                full_scorer_calls: 1,
                record_evaluations: picks_n as u64 * self.corpus.record_count
                    + if needs_singles_setup(arm) {
                        self.data.core_records * self.setup_builds
                    } else {
                        0
                    },
                cost_ms: cost,
                inference_latency_ms,
            });
        }

        if let (Some(validation), Some(validation_records)) =
            (&manifest.validation_data, self.validation_records)
        {
            let selected: BTreeSet<usize> = arms.iter().filter_map(|a| a.selected_bundle).collect();
            let refs: Vec<(usize, &CreatureExport)> =
                selected.iter().map(|&i| (i, &scored[i].creature)).collect();
            let held_out = self.score(
                &format!("validation-b{budget_index}-s{seed}"),
                validation,
                validation_records,
                &refs,
            )?;
            row.validation_parent_score = Some(held_out["baseline"].score);
            for arm in &mut arms {
                if let Some(i) = arm.selected_bundle {
                    let result = &held_out[&bundle_name(i)];
                    arm.validation_score = Some(result.score);
                    arm.validation_error = Some(result.error);
                }
            }
        }

        row.arms = arms;
        row.feasible = true;
        Ok(row)
    }

    /// Singles in Ockham's ranked order, as `promote::bundle_plans` ranks them.
    ///
    /// Every proposable single is a member with delta `L0 − Li`. A single with no
    /// loss gets `NEG_INFINITY`, so it sorts after every estimated single and ties
    /// by UUID. The order is the all-winners plan, the first plan `bundle_plans`
    /// emits for two or more members.
    ///
    /// # Errors
    ///
    /// When `bundle_plans` emits no all-winners plan for two or more members.
    fn rank_singles(&self) -> Result<Vec<usize>, String> {
        let members: Vec<BundleMember> = self
            .singles
            .iter()
            .map(|single| BundleMember {
                uuid: single.uuid.clone(),
                kind: CandidateKind::Ablation,
                delta: match (self.base_loss, single.loss) {
                    (Some(base), Some(loss)) => base - loss,
                    _ => f64::NEG_INFINITY,
                },
            })
            .collect();
        let ordered: Vec<String> = match members.len() {
            0 => Vec::new(),
            1 => vec![members[0].uuid.clone()],
            _ => bundle_plans(&members)
                .plans
                .into_iter()
                .next()
                .ok_or("bundle_plans emitted no all-winners plan")?,
        };
        let position: HashMap<&str, usize> = self
            .singles
            .iter()
            .enumerate()
            .map(|(i, single)| (single.uuid.as_str(), i))
            .collect();
        ordered
            .iter()
            .map(|uuid| {
                position
                    .get(uuid.as_str())
                    .copied()
                    .ok_or_else(|| format!("ranked uuid '{uuid}' is not a proposable single"))
            })
            .collect()
    }

    /// Diagnostic `-Σ(Li − L0)` over the members. `None` if any is unestimated;
    /// a fault in the sum is counted in the rejection tally.
    fn additive_estimate(&mut self, members: &[usize]) -> Result<Option<f64>, String> {
        let Some(base) = self.base_loss else {
            return Ok(None);
        };
        let Some(losses) = members
            .iter()
            .map(|&i| self.singles[i].loss)
            .collect::<Option<Vec<f64>>>()
        else {
            return Ok(None);
        };
        let change = additive_loss_change(base, &losses).map_err(Fault::from);
        Ok(settle(&mut self.store.tally, change)?.map(|change| -change))
    }

    /// Cached-output improvement `L0 − predicted loss`, streamed chunk by chunk.
    fn cached_estimate(&mut self, members: &[usize]) -> Result<Option<f64>, String> {
        let Some(base) = self.base_loss else {
            return Ok(None);
        };
        if members.iter().any(|&i| self.singles[i].loss.is_none()) {
            return Ok(None);
        }
        let names: Vec<String> = members
            .iter()
            .map(|&i| self.singles[i].entry.clone())
            .collect();
        let predicted = read_predicted_loss(&mut self.store, &self.data, self.clock, &names);
        Ok(settle(&mut self.store.tally, predicted)?
            .map(|loss| base - loss)
            .filter(|value| value.is_finite()))
    }

    /// Score `bundles` (plus the parent as `baseline`) in a fresh directory.
    fn score(
        &self,
        dir_name: &str,
        data: &Path,
        expected_records: u64,
        bundles: &[(usize, &CreatureExport)],
    ) -> Result<BTreeMap<String, ScoreResult>, String> {
        let dir = self.workdir.join(dir_name);
        let _ = fs::remove_dir_all(&dir);
        let outcome = self.write_and_score(&dir, data, expected_records, bundles);
        let _ = fs::remove_dir_all(&dir);
        outcome
    }

    fn write_and_score(
        &self,
        dir: &Path,
        data: &Path,
        expected_records: u64,
        bundles: &[(usize, &CreatureExport)],
    ) -> Result<BTreeMap<String, ScoreResult>, String> {
        fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
        fs::write(dir.join("baseline.json"), &self.parent_json)
            .map_err(|e| format!("{}: {e}", dir.display()))?;
        let mut expected = vec!["baseline".to_string()];
        for (index, creature) in bundles {
            let json = creature_to_json(creature).map_err(|e| e.to_string())?;
            let name = bundle_name(*index);
            fs::write(dir.join(format!("{name}.json")), json)
                .map_err(|e| format!("{}: {e}", dir.display()))?;
            expected.push(name);
        }
        let results = self
            .scorer
            .score_directory(dir, data, ScorerMode::Full)
            .map_err(|e| format!("full scorer: {e}"))?;
        if results.len() != expected.len() {
            return Err(format!(
                "full scorer returned {} result(s) for {} file(s)",
                results.len(),
                expected.len()
            ));
        }
        if let Some(missing) = expected.iter().find(|name| !results.contains_key(*name)) {
            return Err(format!("full scorer returned no result for '{missing}'"));
        }
        check_finite(&results).map_err(|e| format!("full scorer: {e}"))?;
        for (name, result) in &results {
            check_report(
                &format!("full scorer, result '{name}'"),
                result.cost_name.as_deref(),
                result.record_count,
                &self.manifest.cost,
                expected_records,
            )?;
        }
        Ok(results)
    }

    /// The parent's report: baseline, singles and cache counters.
    fn report(&self) -> ParentReport {
        let tally = &self.store.tally;
        let (ni, no) = (
            self.data.cfg.num_inputs as u64,
            self.data.cfg.num_outputs as u64,
        );
        let chunk = self.data.chunk as u64;
        // The predicted-output accumulator is one cache entry: core records × outputs × 4.
        let accumulator = self.data.core_records * no * 4;
        ParentReport {
            parent: self.name.clone(),
            parent_checksum: self.incumbent.checksum.clone(),
            parent_score: self.baseline.score,
            parent_error: self.baseline.error,
            parent_complexity_penalty: self.baseline.complexity_penalty,
            hidden_neurons: self.hidden.len(),
            core_records: self.data.core_records,
            proposable: self.singles.len(),
            estimated: self.singles.iter().filter(|s| s.loss.is_some()).count(),
            failed_proposals: self.failed.clone(),
            estimate_blocked: self.estimate_blocked.clone(),
            singles_ms: self.singles_ms,
            cache: CacheStats {
                hits: tally.hits,
                builds: tally.builds,
                rejections: tally.rejections.clone(),
                bytes_used: self.store.cache.used_bytes(),
                peak_bytes: tally.peak_bytes,
                max_cache_bytes: self.manifest.limits.max_cache_bytes,
                ram_bound_bytes: chunk * 4 * (ni + no + SCRATCH_BUFFERS * no) + accumulator,
                build_ms: tally.build_ms,
                read_ms: tally.read_ms,
            },
        }
    }
}

/// Refuses a scorer report whose cost or record count disagrees with the manifest
/// or the corpus. Mirrors `crate::baseline::establish_baseline`: a cost name is
/// compared case-insensitively, and a record count of zero means "not reported".
fn check_report(
    label: &str,
    cost_name: Option<&str>,
    record_count: u64,
    declared_cost: &str,
    corpus_records: u64,
) -> Result<(), String> {
    if let Some(reported) = cost_name
        && !reported.eq_ignore_ascii_case(declared_cost)
    {
        return Err(format!(
            "{label}: scorer reported cost '{reported}', but the manifest declares '{declared_cost}'"
        ));
    }
    if record_count != 0 && record_count != corpus_records {
        return Err(format!(
            "{label}: scorer saw {record_count} records but the corpus has {corpus_records}"
        ));
    }
    Ok(())
}

/// SHA-256 over the `(uuid, mean)` list handed to the pruning step, in order.
///
/// Each uuid is length-prefixed. A missing mean is one zero byte and a present
/// one is a one byte then the mean's `f64` bits, so the two never collide.
fn stats_digest(members: &[GroupMember]) -> String {
    let mut bytes = Vec::new();
    for member in members {
        bytes.extend_from_slice(&(member.uuid.len() as u64).to_le_bytes());
        bytes.extend_from_slice(member.uuid.as_bytes());
        match member.mean {
            None => bytes.push(0),
            Some(mean) => {
                bytes.push(1);
                bytes.extend_from_slice(&mean.to_bits().to_le_bytes());
            }
        }
    }
    sha256_hex(&bytes)
}

/// Turn a read fault into "no estimate", counting it by kind; a stream fault aborts.
fn settle<T>(tally: &mut Tally, result: Result<T, Fault>) -> Result<Option<T>, String> {
    match result {
        Ok(value) => Ok(Some(value)),
        Err(Fault::Stream(message)) => Err(message),
        Err(fault) => {
            *tally
                .rejections
                .entry(fault.kind().to_string())
                .or_insert(0) += 1;
            Ok(None)
        }
    }
}

/// Stateless name of a pool bundle's file and result key.
fn bundle_name(index: usize) -> String {
    format!("b{index}")
}

/// `ceil(budget × hidden)` with a tiny tolerance so `0.2 × 5` stays 1.
fn requested_removals(budget: f64, hidden: usize) -> usize {
    if hidden == 0 {
        return 0;
    }
    ((budget * hidden as f64) - 1e-9).ceil().max(0.0) as usize
}

/// Deterministic seed from a list of integers.
fn derive_seed(parts: &[u64]) -> u64 {
    let mut acc = 0x5EED_0260_u64;
    for part in parts {
        let mut mixer = SplitMix64(acc ^ part.wrapping_mul(0x9E37_79B9_7F4A_7C15));
        acc = mixer.next();
    }
    acc
}

/// Pool of distinct member sets, each a list of positions in rank order.
///
/// The first set is the ranked prefix `0..k`. The rest are seeded distinct
/// draws, kept in rank order and de-duplicated, with bounded attempts.
fn build_pool(eligible: usize, k: usize, audit_pool: usize, seed: u64) -> Vec<Vec<usize>> {
    let prefix: Vec<usize> = (0..k).collect();
    let mut seen: HashSet<Vec<usize>> = HashSet::from([prefix.clone()]);
    let mut pool = vec![prefix];
    let mut rng = SplitMix64(seed);
    let mut order: Vec<usize> = (0..eligible).collect();
    let max_attempts = audit_pool.saturating_mul(ATTEMPTS_PER_BUNDLE);
    let mut attempts = 0;
    while pool.len() < audit_pool && attempts < max_attempts {
        attempts += 1;
        for i in 0..k {
            let span = (eligible - i) as u64;
            let j = i + (rng.next() % span) as usize;
            order.swap(i, j);
        }
        let mut set = order[..k].to_vec();
        set.sort_unstable();
        if seen.insert(set.clone()) {
            pool.push(set);
        }
    }
    pool
}

/// Build one cache entry unless it already exists.
fn build_entry(
    store: &mut Store,
    data: &Dataset,
    clock: &dyn Clock,
    name: &str,
    source: Source<'_>,
) -> Result<(), Fault> {
    if store.cache.has(name) {
        store.tally.hits += 1;
        return Ok(());
    }
    let started = clock.now();
    let result = match source {
        Source::Targets => write_entry(store, data, name, |chunk, out| {
            out.copy_from_slice(chunk.targets);
            Ok(())
        }),
        Source::Network(creature) => match compile_creature(creature) {
            Ok(mut net) => {
                let (ni, no) = (data.cfg.num_inputs, data.cfg.num_outputs);
                write_entry(store, data, name, move |chunk, out| {
                    for r in 0..chunk.records {
                        net.activate_into(
                            &chunk.inputs[r * ni..(r + 1) * ni],
                            &mut out[r * no..(r + 1) * no],
                        );
                    }
                    Ok(())
                })
            }
            Err(error) => Err(Fault::Stream(error.to_string())),
        },
    };
    store.tally.build_ms += clock.ms_since(started);
    match result {
        Ok(()) => {
            store.tally.builds += 1;
            store.tally.peak_bytes = store.tally.peak_bytes.max(store.cache.used_bytes());
            Ok(())
        }
        Err(Fault::Stream(message)) => Err(Fault::Stream(message)),
        Err(fault) => {
            *store
                .tally
                .rejections
                .entry(fault.kind().to_string())
                .or_insert(0) += 1;
            Err(fault)
        }
    }
}

/// Stream the core records through `produce` into a new entry, chunk by chunk.
fn write_entry<F>(
    store: &mut Store,
    data: &Dataset,
    name: &str,
    mut produce: F,
) -> Result<(), Fault>
where
    F: FnMut(&RecordChunk<'_>, &mut [f32]) -> Result<(), String>,
{
    let mut writer = store.cache.writer(name)?;
    let mut refused: Option<CacheRejection> = None;
    let width = data.cfg.num_outputs;
    let buffer = &mut store.scratch.acc;
    let streamed =
        for_each_selected_chunk(&data.dir, &data.cfg, &data.ranges, data.chunk, |chunk| {
            let len = chunk.records * width;
            produce(&chunk, &mut buffer[..len])?;
            if let Err(rejection) = writer.push(&buffer[..len]) {
                refused = Some(rejection);
                return Err("cache entry refused".to_string());
            }
            Ok(ControlFlow::Continue(()))
        });
    if let Err(message) = streamed {
        return Err(match refused {
            Some(rejection) => Fault::Cache(rejection),
            None => Fault::Stream(message),
        });
    }
    writer.finish(&mut store.cache).map_err(Fault::Cache)
}

/// Mean loss of one entry against the core targets, chunk by chunk.
fn mean_loss(
    store: &mut Store,
    data: &Dataset,
    clock: &dyn Clock,
    entry: &str,
) -> Result<f64, Fault> {
    let started = clock.now();
    let result = read_mean_loss(store, data, entry);
    store.tally.read_ms += clock.ms_since(started);
    result
}

fn read_mean_loss(store: &mut Store, data: &Dataset, entry: &str) -> Result<f64, Fault> {
    let cache = &store.cache;
    let mut targets = cache.reader(TARGETS_ENTRY)?;
    let mut outputs = cache.reader(entry)?;
    let width = data.cfg.num_outputs;
    let scratch = &mut store.scratch;
    let mut total = LossSum::default();
    let mut remaining = data.core_records;
    while remaining > 0 {
        let n = remaining.min(data.chunk as u64) as usize;
        let len = n * width;
        targets.read_chunk(&mut scratch.target[..len])?;
        outputs.read_chunk(&mut scratch.member[..len])?;
        let sum = chunk_loss_sum(&scratch.target[..len], &scratch.member[..len], width)?;
        total.add(sum, n as u64);
        remaining -= n as u64;
    }
    total
        .mean()
        .ok_or_else(|| Fault::Stream("audit core is empty".to_string()))
}

/// Mean loss of `y0 + Σ(yi − y0)` against the core targets, chunk by chunk.
fn read_predicted_loss(
    store: &mut Store,
    data: &Dataset,
    clock: &dyn Clock,
    members: &[String],
) -> Result<f64, Fault> {
    let started = clock.now();
    let result = read_predicted(store, data, members);
    store.tally.read_ms += clock.ms_since(started);
    result
}

/// Mean loss of `y0 + Σ(yi − y0)` against the core targets.
///
/// The accumulator (one cache entry's values) starts as `y0`. Each member is then
/// folded in with its own reader and a fresh `y0` reader, both closed before the
/// next member, so at most two entry files are open at once. Targets are streamed
/// last. Every read is chunked, so RAM is the accumulator plus the chunk buffers.
fn read_predicted(store: &mut Store, data: &Dataset, members: &[String]) -> Result<f64, Fault> {
    let cache = &store.cache;
    let scratch = &mut store.scratch;
    let width = data.cfg.num_outputs;
    {
        let mut y0 = cache.reader(Y0_ENTRY)?;
        scratch
            .predicted
            .resize(data.core_records as usize * width, 0.0);
        for (start, n) in core_chunks(data) {
            let span = start * width..(start + n) * width;
            y0.read_chunk(&mut scratch.predicted[span])?;
        }
    }
    for name in members {
        let mut member = cache.reader(name)?;
        let mut y0 = cache.reader(Y0_ENTRY)?;
        for (start, n) in core_chunks(data) {
            let len = n * width;
            let span = start * width..start * width + len;
            member.read_chunk(&mut scratch.member[..len])?;
            y0.read_chunk(&mut scratch.y0[..len])?;
            add_delta(
                &mut scratch.predicted[span],
                &scratch.y0[..len],
                &scratch.member[..len],
            )?;
        }
    }
    let mut targets = cache.reader(TARGETS_ENTRY)?;
    let mut total = LossSum::default();
    for (start, n) in core_chunks(data) {
        let len = n * width;
        let span = start * width..start * width + len;
        targets.read_chunk(&mut scratch.target[..len])?;
        let sum = chunk_loss_sum(&scratch.target[..len], &scratch.predicted[span], width)?;
        total.add(sum, n as u64);
    }
    total
        .mean()
        .ok_or_else(|| Fault::Stream("audit core is empty".to_string()))
}

/// The audit core as `(first record, record count)` pairs, `chunk` records at a time.
fn core_chunks(data: &Dataset) -> impl Iterator<Item = (usize, usize)> {
    let total = data.core_records as usize;
    let chunk = data.chunk;
    (0..total)
        .step_by(chunk)
        .map(move |start| (start, chunk.min(total - start)))
}
