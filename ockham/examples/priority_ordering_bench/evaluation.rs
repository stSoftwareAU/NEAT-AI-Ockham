use std::collections::{BTreeMap, BTreeSet};
use std::time::Instant;

use neat_ai_ockham::features::FEATURE_NAMES;
use neat_ai_ockham::incumbent::sha256_hex;
use neat_ai_ockham::model::pairwise::{PairwiseModel, RankedRow};
use neat_ai_ockham::priority::{CompositeWeights, expected_pruning_value};
use serde::Serialize;

use super::audit::{Audit, AuditRow, History, Pool, ms};
use super::manifest::{Manifest, Split, Task, seeded_order};

pub fn feature_names() -> Vec<String> {
    ["partialDelta", "neuronFeaturesAvailable"]
        .into_iter()
        .chain(FEATURE_NAMES.iter().copied())
        .map(str::to_string)
        .collect()
}

fn vector(row: &AuditRow) -> Result<Vec<f64>, String> {
    let partial = row.sample_delta.ok_or("missing partial score")?;
    if let Some(f) = &row.features {
        super::audit::check_features(f)?;
    }
    let mut out = vec![partial, if row.features.is_some() { 1.0 } else { 0.0 }];
    out.extend(
        row.features
            .map_or_else(|| vec![0.0; FEATURE_NAMES.len()], |f| f.vector()),
    );
    if out.iter().any(|v| !v.is_finite()) {
        return Err("non-finite experimental feature".into());
    }
    Ok(out)
}

pub fn labelled(rows: &[AuditRow]) -> Result<Vec<RankedRow>, String> {
    rows.iter()
        .filter_map(|r| r.full_delta.map(|score| (r, score)))
        .map(|(row, score)| {
            if !score.is_finite() {
                return Err("non-finite full label".into());
            }
            Ok(RankedRow {
                features: vector(row)?,
                score,
            })
        })
        .collect()
}

#[derive(Debug, Clone, Serialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct Metrics {
    pub evaluated: usize,
    pub top_count: usize,
    pub top_recall: f64,
    pub improvements: usize,
    pub false_rejection: Option<f64>,
    pub best_full_delta: f64,
}

pub fn metrics(
    full: &[f64],
    order: &[usize],
    budget: usize,
    epsilon: f64,
) -> Result<Metrics, String> {
    if full.is_empty() {
        return Err("empty audit pool".into());
    }
    if full.iter().any(|v| !v.is_finite()) {
        return Err("non-finite full score".into());
    }
    if order.len() != full.len()
        || order.iter().copied().collect::<BTreeSet<_>>() != (0..full.len()).collect()
    {
        return Err("ranking is not a permutation".into());
    }
    let mut sorted = full.to_vec();
    sorted.sort_by(|a, b| b.total_cmp(a));
    let cutoff = sorted[full.len().div_ceil(20) - 1];
    let top_count = full.iter().filter(|s| **s >= cutoff).count();
    let selected = &order[..budget.min(order.len())];
    let found = selected.iter().filter(|i| full[**i] >= cutoff).count();
    let improvements = full.iter().filter(|s| **s > epsilon).count();
    let accepted = selected.iter().filter(|i| full[**i] > epsilon).count();
    Ok(Metrics {
        evaluated: selected.len(),
        top_count,
        top_recall: found as f64 / top_count as f64,
        improvements,
        false_rejection: (improvements > 0)
            .then(|| (improvements - accepted) as f64 / improvements as f64),
        best_full_delta: selected.iter().map(|i| full[*i]).fold(0.0, f64::max),
    })
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct Trial {
    task: String,
    lineage: String,
    corpus_version: String,
    seed: u64,
    calibration_budget: usize,
    calibration_observed: usize,
    history_fraction: f64,
    source_observed: usize,
    strategy: String,
    evaluation_budget: usize,
    metrics: Metrics,
    model_sha256: Option<String>,
    training_ms: u64,
    adaptation_ms: u64,
    ranking_ms: u64,
    preparation_ms: u64,
    historical_scoring_ms: u64,
    calibration_scoring_ms: u64,
    feature_ms: u64,
    partial_scoring_ms: u64,
    search_scoring_ms: u64,
    total_strategy_ms: u64,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct Coverage {
    task: String,
    seed: u64,
    population: usize,
    selected: usize,
    generated: usize,
    fully_scored: usize,
    missing_features: usize,
    complete: bool,
    kinds: BTreeMap<String, usize>,
    failures: Vec<String>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct Paired {
    strategy: String,
    control: String,
    calibration_budget: usize,
    history_fraction: f64,
    evaluation_budget: usize,
    groups: usize,
    task_deltas: BTreeMap<String, f64>,
    mean_recall_delta: f64,
    recall_delta_ci95: Option<[f64; 2]>,
    negative_transfer_tasks: Vec<String>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Report {
    format_version: u32,
    manifest_sha256: String,
    audit_sha256: String,
    feature_schema: Vec<String>,
    verdict: String,
    reasons: Vec<String>,
    audit_cost_ms: u64,
    evaluation_cost_ms: u64,
    history_coverage: Vec<serde_json::Value>,
    coverage: Vec<Coverage>,
    trials: Vec<Trial>,
    paired: Vec<Paired>,
}

fn retained(pool: &Pool) -> Vec<AuditRow> {
    pool.rows
        .iter()
        .filter(|r| r.kind.is_some())
        .cloned()
        .collect()
}
fn full_ms(pool: &Pool) -> u64 {
    pool.rows.iter().map(|r| r.full_ms).sum()
}
fn sample_ms(pool: &Pool) -> u64 {
    pool.rows.iter().map(|r| r.sample_ms).sum()
}

// Every connected lineage/task-family/corpus component is one bootstrap unit.
// Tasks sharing a grouping dimension stay together through transitive overlap.
pub(super) fn groups(tasks: &[&Task]) -> Vec<Vec<String>> {
    let mut groups: Vec<Vec<&Task>> = Vec::new();
    for task in tasks {
        let mut combined = vec![*task];
        loop {
            let found = groups.iter().position(|g| {
                g.iter().any(|a| {
                    combined.iter().any(|b| {
                        a.lineage == b.lineage
                            || a.family == b.family
                            || a.corpus_version == b.corpus_version
                            || a.corpus_sha256 == b.corpus_sha256
                            || a.parent_sha256 == b.parent_sha256
                    })
                })
            });
            if let Some(i) = found {
                combined.extend(groups.remove(i));
            } else {
                break;
            }
        }
        groups.push(combined);
    }
    groups
        .into_iter()
        .map(|g| g.into_iter().map(|t| t.id.clone()).collect())
        .collect()
}

pub fn cluster_interval(values: &[Vec<f64>], replicates: usize) -> Option<[f64; 2]> {
    if values.len() < 3 {
        return None;
    }
    let means: Vec<_> = values
        .iter()
        .map(|v| v.iter().sum::<f64>() / v.len() as f64)
        .collect();
    let mut draws = Vec::with_capacity(replicates);
    for draw in 0..replicates {
        let mut sum = 0.0;
        for slot in 0..means.len() {
            let hash = sha256_hex(format!("bootstrap-v1:{draw}:{slot}").as_bytes());
            let random = u64::from_str_radix(&hash[..16], 16).expect("SHA hex");
            sum += means[random as usize % means.len()];
        }
        draws.push(sum / means.len() as f64);
    }
    draws.sort_by(f64::total_cmp);
    Some([
        draws[replicates / 40],
        draws[(replicates * 39 / 40).min(replicates - 1)],
    ])
}

fn ordered(rows: &[AuditRow], keys: &[f64], seed: u64) -> Vec<usize> {
    let ids: Vec<_> = rows.iter().map(|r| r.visit.clone()).collect();
    let ties = seeded_order(&ids, seed ^ 0xa11d_17c0_2610_0001);
    let tie_keys: BTreeMap<_, _> = ties.iter().enumerate().map(|(i, id)| (id, i)).collect();
    let mut order: Vec<_> = (0..rows.len()).collect();
    order.sort_by(|a, b| {
        keys[*b]
            .total_cmp(&keys[*a])
            .then_with(|| tie_keys[&ids[*a]].cmp(&tie_keys[&ids[*b]]))
    });
    order
}

fn subset(rows: Vec<AuditRow>, n: usize, seed: u64) -> Vec<AuditRow> {
    let ids: Vec<_> = rows.iter().map(|r| r.visit.clone()).collect();
    let keys = seeded_order(&ids, seed);
    let mut by_id: BTreeMap<_, _> = rows.into_iter().map(|r| (r.visit.clone(), r)).collect();
    keys.into_iter()
        .take(n)
        .map(|id| by_id.remove(&id).expect("unique audit visit"))
        .collect()
}

fn historical_rows(
    history: &History,
    task: &Task,
    m: &Manifest,
    fraction: f64,
    seed: u64,
) -> Result<Vec<RankedRow>, String> {
    let ids: Vec<_> = history
        .records
        .iter()
        .map(|r| format!("{}:{}:{:?}", r.kind, r.uuid, r.merged_with))
        .collect();
    let domain = sha256_hex(format!("history-v1:{}:{seed}", task.id).as_bytes());
    let order = seeded_order(
        &ids,
        u64::from_str_radix(&domain[..16], 16).expect("SHA hex"),
    );
    if ids.iter().collect::<BTreeSet<_>>().len() != ids.len() {
        return Err("duplicate historical candidate".into());
    }
    let n = (order.len() as f64 * fraction).floor() as usize;
    order
        .iter()
        .take(n)
        .map(|id| {
            let r = &history.records[ids.iter().position(|x| x == id).expect("known history id")];
            if r.unix_secs >= m.historical_cutoff_unix
                || r.creature_checksum != task.parent_sha256
                || r.corpus_identity != task.corpus_identity
            {
                return Err("historical provenance/cutoff mismatch".into());
            }
            let mut features = vec![
                r.sample_delta.ok_or("missing historical partial score")?,
                1.0,
            ];
            let mut raw = r.vector().ok_or("invalid historical feature schema")?;
            // These counters have no individual observation timestamps in v1 logs.
            raw[13] = 0.0;
            raw[14] = 0.0;
            features.extend(raw);
            let score = r.full_delta.ok_or("missing historical full score")?;
            if !score.is_finite() || features.iter().any(|v| !v.is_finite()) {
                return Err("non-finite historical row".into());
            }
            Ok(RankedRow { features, score })
        })
        .collect()
}

fn training_pools<'a>(
    m: &'a Manifest,
    audit: &'a Audit,
    split: Split,
    family: Option<&str>,
    seed: u64,
) -> Vec<(&'a Task, &'a Pool)> {
    audit
        .pools
        .iter()
        .filter(|p| p.seed == seed && p.complete())
        .filter_map(|p| {
            let task = m.tasks.iter().find(|t| t.id == p.task)?;
            (task.split == split && family.is_none_or(|f| task.family == f)).then_some((task, p))
        })
        .collect()
}

pub fn evaluate(m: &Manifest, audit: &Audit, audit_sha256: String) -> Result<Report, String> {
    m.validate()?;
    if audit.format_version != 1
        || audit.phase != "audit"
        || audit.manifest_sha256 != m.identity()?
        || audit.freeze_sha256.is_none()
    {
        return Err("audit protocol/manifest/freeze mismatch".into());
    }
    let start = Instant::now();
    let mut report=Report {format_version:1,manifest_sha256:m.identity()?,audit_sha256,feature_schema:feature_names(),verdict:"inconclusive".into(),
        reasons:audit.failures.clone(),audit_cost_ms:audit.elapsed_ms,evaluation_cost_ms:0,history_coverage:audit.histories.iter().map(|h|serde_json::json!({"task":h.task,"labelled":h.records.len(),"unlabelled":h.unlabelled,"missingPartial":h.missing_partial,"futureExcluded":h.future,"acquisitionMs":h.acquisition_ms,"loadingMs":h.loading_ms})).collect(),coverage:Vec::new(),trials:Vec::new(),paired:Vec::new()};
    let mut seen = BTreeSet::new();
    for pool in &audit.pools {
        if !m.tasks.iter().any(|t| t.id == pool.task)
            || !m.seeds.contains(&pool.seed)
            || !seen.insert((&pool.task, pool.seed))
        {
            return Err("unknown or duplicate audit task/seed".into());
        }
        if pool.selected.len() != pool.rows.len()
            || pool
                .selected
                .iter()
                .zip(&pool.rows)
                .any(|(id, row)| *id != row.visit)
            || super::manifest::seeded_order(&pool.selected, pool.seed) != pool.selected
            || pool.selected.iter().collect::<BTreeSet<_>>().len() != pool.selected.len()
        {
            return Err("audit selected visit/outcome mismatch".into());
        }
        let mut kinds: BTreeMap<String, usize> = [
            "identity", "ablation", "constant", "group", "merge", "synapse",
        ]
        .into_iter()
        .map(|k| (k.into(), 0))
        .collect();
        for row in &pool.rows {
            if let Some(kind) = &row.kind {
                *kinds.get_mut(kind).ok_or("unknown candidate kind")? += 1;
            }
        }
        let rows = retained(pool);
        report.coverage.push(Coverage {
            task: pool.task.clone(),
            seed: pool.seed,
            population: pool.population,
            selected: pool.selected.len(),
            generated: rows.len(),
            fully_scored: rows.iter().filter(|r| r.full_delta.is_some()).count(),
            missing_features: rows.iter().filter(|r| r.features.is_none()).count(),
            complete: pool.complete(),
            kinds,
            failures: pool
                .rows
                .iter()
                .filter_map(|r| r.failure.as_ref().map(|f| format!("{}: {f}", r.visit)))
                .collect(),
        });
        if !pool.complete() {
            report.reasons.push(format!(
                "{} seed {} has incomplete scoring",
                pool.task, pool.seed
            ));
        }
    }
    if audit.histories.iter().any(|h| h.acquisition_ms == 0) {
        report
            .reasons
            .push("historical acquisition cost is unknown (zero charge)".into());
    }
    let expected_histories = m.tasks.iter().filter(|t| t.split == Split::Source).count();
    if audit.histories.len() != expected_histories
        || audit
            .histories
            .iter()
            .map(|h| &h.task)
            .collect::<BTreeSet<_>>()
            .len()
            != expected_histories
    {
        report
            .reasons
            .push("missing/duplicate source histories".into());
    }
    for task in &m.tasks {
        for seed in &m.seeds {
            if !seen.contains(&(&task.id, *seed)) {
                report
                    .reasons
                    .push(format!("missing pool {} seed {seed}", task.id));
            }
        }
    }
    let validation: Vec<_> = m
        .tasks
        .iter()
        .filter(|t| t.split == Split::Validation)
        .collect();
    let clusters = groups(&validation);
    if clusters.len() < m.minimum_independent_groups {
        report.reasons.push(format!(
            "{} independent validation groups; need {}",
            clusters.len(),
            m.minimum_independent_groups
        ));
    }
    'tasks: for task in &validation {
        for pool in audit.pools.iter().filter(|p| p.task == task.id) {
            let rows = retained(pool);
            if !pool.complete() || rows.len() < m.minimum_complete_pool {
                report.reasons.push(format!(
                    "{} seed {} below complete-pool requirement",
                    task.id, pool.seed
                ));
                continue;
            }
            let truth: Vec<_> = rows
                .iter()
                .map(|r| r.full_delta.ok_or("missing validation full score"))
                .collect::<Result<_, _>>()?;
            let source: Vec<_> = audit
                .histories
                .iter()
                .map(|h| {
                    let task = m
                        .tasks
                        .iter()
                        .find(|t| t.id == h.task && t.split == Split::Source)
                        .ok_or("unknown historical source task")?;
                    Ok((task, h))
                })
                .collect::<Result<_, String>>()?;
            let calibration =
                training_pools(m, audit, Split::Calibration, Some(&task.family), pool.seed);
            for fraction in &m.history_fractions {
                let source_tasks = source
                    .iter()
                    .map(|(t, h)| historical_rows(h, t, m, *fraction, pool.seed))
                    .collect::<Result<Vec<_>, _>>()?;
                let source_count = source_tasks.iter().map(Vec::len).sum();
                for cal_budget in &m.calibration_budgets {
                    if ms(start) >= m.max_run_ms {
                        report
                            .reasons
                            .push("evaluation compute limit exhausted".into());
                        break 'tasks;
                    }
                    // A single family-wide calibration budget, not one per parent.
                    let mut available: Vec<_> = calibration
                        .iter()
                        .flat_map(|(t, p)| {
                            retained(p).into_iter().map(|mut r| {
                                r.visit = format!("{}:{}", t.id, r.visit);
                                r
                            })
                        })
                        .collect();
                    available = subset(available, *cal_budget, pool.seed);
                    let cal_count = available.len();
                    if cal_count < *cal_budget {
                        report.reasons.push(format!(
                            "{} seed {} calibration budget {cal_budget} unavailable",
                            task.id, pool.seed
                        ));
                    }
                    // Restore task boundaries before constructing pairs.
                    let mut cal_groups: BTreeMap<String, Vec<AuditRow>> = BTreeMap::new();
                    for row in available {
                        let task = row
                            .visit
                            .split_once(':')
                            .ok_or("calibration task prefix")?
                            .0
                            .to_string();
                        cal_groups.entry(task).or_default().push(row);
                    }
                    let cal_tasks = cal_groups
                        .values()
                        .map(|r| labelled(r))
                        .collect::<Result<Vec<_>, _>>()?;
                    let normalisation: Vec<_> = source_tasks
                        .iter()
                        .chain(&cal_tasks)
                        .flatten()
                        .cloned()
                        .collect();
                    let cal_normalisation: Vec<_> = cal_tasks.iter().flatten().cloned().collect();
                    let tick = Instant::now();
                    let mut transfer = PairwiseModel::fit(
                        &source_tasks,
                        &normalisation,
                        feature_names().len(),
                        m.fit,
                    )?;
                    let pretrain_ms = ms(tick);
                    let tick = Instant::now();
                    transfer.adapt(&cal_tasks, m.fit)?;
                    let adapt_ms = ms(tick);
                    let tick = Instant::now();
                    let target = PairwiseModel::fit(
                        &cal_tasks,
                        &cal_normalisation,
                        feature_names().len(),
                        m.fit,
                    )?;
                    let target_ms = ms(tick);
                    for strategy in [
                        "random",
                        "partial",
                        "multiple-proxies",
                        "target-pairwise",
                        "transfer-pairwise",
                    ] {
                        let tick = Instant::now();
                        let keys = rows
                            .iter()
                            .map(|r| match strategy {
                                "random" => Ok(0.0),
                                "partial" => r.sample_delta.ok_or("missing partial score".into()),
                                "multiple-proxies" => Ok(r.features.map_or(0.0, |f| {
                                    let w = CompositeWeights::default();
                                    expected_pruning_value(w.survival_probability(&f), &f, &w)
                                })),
                                "target-pairwise" => target.score(&vector(r)?),
                                _ => transfer.score(&vector(r)?),
                            })
                            .collect::<Result<Vec<_>, String>>()?;
                        let order = ordered(&rows, &keys, pool.seed);
                        let ranking_ms = ms(tick);
                        let model_sha256 = match strategy {
                            "target-pairwise" => Some(sha256_hex(
                                &serde_json::to_vec(&target).map_err(|e| e.to_string())?,
                            )),
                            "transfer-pairwise" => Some(sha256_hex(
                                &serde_json::to_vec(&transfer).map_err(|e| e.to_string())?,
                            )),
                            _ => None,
                        };
                        for budget in &m.evaluation_budgets {
                            let metric = metrics(&truth, &order, *budget, m.minimum_improvement)?;
                            let is_transfer = strategy == "transfer-pairwise";
                            let is_target = strategy == "target-pairwise";
                            let uses_features =
                                is_transfer || is_target || strategy == "multiple-proxies";
                            let training_ms = if is_transfer {
                                pretrain_ms
                            } else if is_target {
                                target_ms
                            } else {
                                0
                            };
                            let adaptation_ms = if is_transfer { adapt_ms } else { 0 };
                            let historical_scoring_ms = if is_transfer {
                                source
                                    .iter()
                                    .map(|(_, h)| h.acquisition_ms + h.loading_ms)
                                    .sum()
                            } else {
                                0
                            };
                            let calibration_scoring_ms = if is_transfer || is_target {
                                calibration
                                    .iter()
                                    .map(|(_, p)| {
                                        full_ms(p)
                                            + sample_ms(p)
                                            + p.preparation_ms
                                            + p.features_ms
                                            + p.generation_ms
                                    })
                                    .sum()
                            } else {
                                0
                            };
                            let partial_scoring_ms =
                                if strategy == "random" || strategy == "multiple-proxies" {
                                    0
                                } else {
                                    sample_ms(pool)
                                };
                            let feature_ms = if uses_features { pool.features_ms } else { 0 };
                            let search_scoring_ms =
                                order.iter().take(*budget).map(|i| rows[*i].full_ms).sum();
                            let preparation_ms = pool.preparation_ms + pool.generation_ms;
                            let total_strategy_ms = preparation_ms
                                + historical_scoring_ms
                                + calibration_scoring_ms
                                + feature_ms
                                + partial_scoring_ms
                                + search_scoring_ms
                                + training_ms
                                + adaptation_ms
                                + ranking_ms;
                            report.trials.push(Trial {
                                task: task.id.clone(),
                                lineage: task.lineage.clone(),
                                corpus_version: task.corpus_version.clone(),
                                seed: pool.seed,
                                calibration_budget: *cal_budget,
                                calibration_observed: cal_count,
                                history_fraction: *fraction,
                                source_observed: source_count,
                                strategy: strategy.into(),
                                evaluation_budget: *budget,
                                metrics: metric,
                                model_sha256: model_sha256.clone(),
                                training_ms,
                                adaptation_ms,
                                ranking_ms,
                                preparation_ms,
                                historical_scoring_ms,
                                calibration_scoring_ms,
                                feature_ms,
                                partial_scoring_ms,
                                search_scoring_ms,
                                total_strategy_ms,
                            });
                        }
                    }
                }
            }
        }
    }
    for fraction in &m.history_fractions {
        for cal in &m.calibration_budgets {
            for budget in &m.evaluation_budgets {
                for strategy in [
                    "partial",
                    "multiple-proxies",
                    "target-pairwise",
                    "transfer-pairwise",
                ] {
                    for control in ["random", "partial", "target-pairwise"] {
                        if strategy == control {
                            continue;
                        }
                        let selected: Vec<_> = report
                            .trials
                            .iter()
                            .filter(|t| {
                                t.strategy == strategy
                                    && t.history_fraction == *fraction
                                    && t.calibration_budget == *cal
                                    && t.evaluation_budget == *budget
                            })
                            .collect();
                        let mut deltas: BTreeMap<String, Vec<f64>> = BTreeMap::new();
                        for trial in selected {
                            let reference = report
                                .trials
                                .iter()
                                .find(|c| {
                                    c.task == trial.task
                                        && c.seed == trial.seed
                                        && c.strategy == control
                                        && c.history_fraction == *fraction
                                        && c.calibration_budget == *cal
                                        && c.evaluation_budget == *budget
                                })
                                .ok_or("missing paired control")?;
                            deltas
                                .entry(trial.task.clone())
                                .or_default()
                                .push(trial.metrics.top_recall - reference.metrics.top_recall);
                        }
                        let task_deltas: BTreeMap<_, _> = deltas
                            .into_iter()
                            .map(|(k, v)| (k, v.iter().sum::<f64>() / v.len() as f64))
                            .collect();
                        let grouped: Vec<Vec<f64>> = clusters
                            .iter()
                            .map(|group| {
                                group
                                    .iter()
                                    .filter_map(|id| task_deltas.get(id).copied())
                                    .collect()
                            })
                            .filter(|v: &Vec<_>| !v.is_empty())
                            .collect();
                        if grouped.is_empty() {
                            continue;
                        }
                        let mean_recall_delta = grouped
                            .iter()
                            .map(|v| v.iter().sum::<f64>() / v.len() as f64)
                            .sum::<f64>()
                            / grouped.len() as f64;
                        let negative_transfer_tasks =
                            if strategy == "transfer-pairwise" && control == "target-pairwise" {
                                task_deltas
                                    .iter()
                                    .filter(|(_, v)| **v < 0.0)
                                    .map(|(k, _)| k.clone())
                                    .collect()
                            } else {
                                Vec::new()
                            };
                        report.paired.push(Paired {
                            strategy: strategy.into(),
                            control: control.into(),
                            calibration_budget: *cal,
                            history_fraction: *fraction,
                            evaluation_budget: *budget,
                            groups: grouped.len(),
                            task_deltas,
                            mean_recall_delta,
                            recall_delta_ci95: cluster_interval(&grouped, m.bootstrap_replicates),
                            negative_transfer_tasks,
                        });
                    }
                }
            }
        }
    }
    // A positive verdict requires the predeclared full-history, largest-budget
    // cell to pass both controls, cost and safety; the grid is diagnostic only.
    if report.reasons.is_empty() {
        let cal = *m
            .calibration_budgets
            .iter()
            .max()
            .expect("validated budgets");
        let budget = *m
            .evaluation_budgets
            .iter()
            .max()
            .expect("validated budgets");
        let quality = ["random", "partial", "target-pairwise"]
            .into_iter()
            .all(|control| {
                report.paired.iter().any(|p| {
                    p.strategy == "transfer-pairwise"
                        && p.control == control
                        && p.calibration_budget == cal
                        && p.history_fraction == 1.0
                        && p.evaluation_budget == budget
                        && p.recall_delta_ci95
                            .is_some_and(|ci| ci[0] > m.minimum_recall_gain)
                })
            });
        let transfer: Vec<_> = report
            .trials
            .iter()
            .filter(|t| {
                t.strategy == "transfer-pairwise"
                    && t.calibration_budget == cal
                    && t.history_fraction == 1.0
                    && t.evaluation_budget == budget
            })
            .collect();
        let safe = !transfer.is_empty()
            && transfer.iter().all(|t| {
                t.metrics
                    .false_rejection
                    .is_some_and(|v| v <= m.maximum_false_rejection)
            });
        let cheap = !transfer.is_empty()
            && transfer.iter().all(|t| {
                report
                    .trials
                    .iter()
                    .find(|c| {
                        c.task == t.task
                            && c.seed == t.seed
                            && c.strategy == "partial"
                            && c.calibration_budget == cal
                            && c.history_fraction == 1.0
                            && c.evaluation_budget == budget
                    })
                    .is_some_and(|c| {
                        c.total_strategy_ms > 0
                            && t.total_strategy_ms as f64 / c.total_strategy_ms as f64
                                <= m.maximum_cost_ratio
                    })
            });
        if quality && safe && cheap {
            report.verdict = "supported-for-further-research".into();
        } else {
            report.verdict = "no-demonstrated-benefit".into();
            report.reasons.push(format!("predeclared thresholds: recall={quality}, false rejection={safe}, total cost={cheap}"));
        }
    }
    report.evaluation_cost_ms = ms(start);
    Ok(report)
}
