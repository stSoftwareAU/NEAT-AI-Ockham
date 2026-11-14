use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use std::time::Instant;

use neat_ai_ockham::clock::Clock;
use neat_ai_ockham::corpus::{RecordRange, corpus_info, for_each_selected_chunk};
use neat_ai_ockham::scorer::{
    DirectoryScorer, ExternalScorer, ScoreResult, ScorerError, ScorerMode, check_finite,
};
use neat_core::training_data::TrainingDataConfig;
use neat_core::{CreatureExport, compile_creature};
use serde::{Deserialize, Serialize};

use super::manifest::{Latency, Limits, Spec};

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Usage {
    pub proposal_attempts: u64,
    pub statistics_records: u64,
    pub reserved_statistics_records: u64,
    pub scorer_calls: u64,
    pub reserved_record_evaluations: u64,
    pub reported_record_evaluations: u64,
    pub failed_calls: u64,
    pub timeout_exits: u64,
    pub refused_calls: u64,
    pub preparation_ms: f64,
    pub search_ms: f64,
    pub scorer_ms: f64,
    pub refusals: BTreeMap<String, u64>,
    pub visited_keys: BTreeSet<String>,
}

impl Usage {
    pub fn refuse(&mut self, reason: &str) {
        *self.refusals.entry(reason.into()).or_insert(0) += 1;
    }
}

pub struct Metered<'a> {
    pub inner: &'a dyn DirectoryScorer,
    pub limits: &'a Limits,
    pub usage: &'a RefCell<Usage>,
    pub clock: &'a dyn Clock,
    pub started: Instant,
    pub widths: (usize, usize),
}

impl Metered<'_> {
    pub fn expired(&self) -> bool {
        self.clock.since(self.started).as_secs_f64() >= self.limits.search_seconds as f64
    }

    pub fn statistics(&self, records: u64) -> Result<(), String> {
        let mut usage = self.usage.borrow_mut();
        if self.expired() {
            return Err("search-timeout".into());
        }
        if records
            > self
                .limits
                .statistics_records
                .saturating_sub(usage.reserved_statistics_records)
        {
            return Err("statistics-record-limit".into());
        }
        usage.reserved_statistics_records += records;
        Ok(())
    }
}

impl DirectoryScorer for Metered<'_> {
    fn score_directory(
        &self,
        dir: &Path,
        corpus: &Path,
        mode: ScorerMode,
    ) -> Result<BTreeMap<String, ScoreResult>, ScorerError> {
        let info = corpus_info(
            corpus,
            &TrainingDataConfig::new(self.widths.0, self.widths.1),
        )
        .map_err(ScorerError::Malformed)?;
        let mut stems = BTreeSet::new();
        for entry in std::fs::read_dir(dir).map_err(|e| ScorerError::Spawn(e.to_string()))? {
            let path = entry.map_err(|e| ScorerError::Spawn(e.to_string()))?.path();
            if path.extension().is_some_and(|e| e == "json") {
                stems.insert(
                    path.file_stem()
                        .and_then(|s| s.to_str())
                        .ok_or_else(|| ScorerError::Malformed("non-UTF8 cohort stem".into()))?
                        .to_owned(),
                );
            }
        }
        let reserve = info
            .record_count
            .checked_mul(stems.len() as u64)
            .ok_or_else(|| ScorerError::Malformed("record reservation overflow".into()))?;
        {
            let mut usage = self.usage.borrow_mut();
            if self.expired()
                || usage.scorer_calls >= self.limits.scorer_calls
                || reserve
                    > self
                        .limits
                        .scorer_record_evaluations
                        .saturating_sub(usage.reserved_record_evaluations)
            {
                usage.refused_calls += 1;
                return Err(ScorerError::Failed {
                    status: "experiment-compute-limit".into(),
                    stderr: "call refused before execution".into(),
                });
            }
            usage.scorer_calls += 1;
            usage.reserved_record_evaluations += reserve;
        }
        let started = Instant::now();
        let result = self
            .inner
            .score_directory(dir, corpus, mode)
            .and_then(|results| {
                check_finite(&results)?;
                if results.keys().cloned().collect::<BTreeSet<_>>() != stems {
                    return Err(ScorerError::Malformed(
                        "cohort result keys differ from requested stems".into(),
                    ));
                }
                for (key, result) in &results {
                    if result.record_count == 0
                        || result.record_count > info.record_count
                        || (mode.is_authoritative() && result.record_count != info.record_count)
                    {
                        return Err(ScorerError::Malformed(format!(
                            "{key}: recordCount does not cover requested corpus/mode"
                        )));
                    }
                }
                self.usage.borrow_mut().reported_record_evaluations +=
                    results.values().map(|r| r.record_count).sum::<u64>();
                if self.expired() {
                    return Err(ScorerError::Failed {
                        status: "search-timeout".into(),
                        stderr: "scorer completed after search deadline".into(),
                    });
                }
                Ok(results)
            });
        let mut usage = self.usage.borrow_mut();
        usage.scorer_ms += started.elapsed().as_secs_f64() * 1000.0;
        if let Err(error) = &result {
            usage.failed_calls += 1;
            if matches!(error, ScorerError::Failed { status, .. } if status == "exit status: 124" || status == "exit status: 137" || status == "search-timeout")
            {
                usage.timeout_exits += 1;
            }
        }
        result
    }

    fn identity(&self) -> String {
        self.inner.identity()
    }
}

pub struct TimedExternal {
    scorer: ExternalScorer,
    identity: String,
}

impl DirectoryScorer for TimedExternal {
    fn score_directory(
        &self,
        dir: &Path,
        corpus: &Path,
        mode: ScorerMode,
    ) -> Result<BTreeMap<String, ScoreResult>, ScorerError> {
        let mut scorer = self.scorer.clone();
        if let ScorerMode::Sample { rate, phase } = mode {
            scorer.extra_args.extend([
                "--sample-rate".into(),
                rate.to_string(),
                "--sample-phase".into(),
                phase.to_string(),
            ]);
        }
        scorer.score_directory(dir, corpus, ScorerMode::Full)
    }

    fn identity(&self) -> String {
        self.identity.clone()
    }
}

pub fn external(spec: &Spec) -> Result<TimedExternal, String> {
    let paths = std::env::var_os("PATH").ok_or("PATH is absent")?;
    let timeout = ["timeout", "gtimeout"]
        .into_iter()
        .find_map(|name| {
            std::env::split_paths(&paths)
                .map(|dir| dir.join(name))
                .find(|path| path.is_file())
        })
        .ok_or("experiment requires timeout or gtimeout to bound scorer processes")?;
    let mut args = vec![
        "--kill-after=2".into(),
        spec.limits.call_timeout_seconds.to_string(),
        spec.scorer.to_string_lossy().into_owned(),
    ];
    args.extend(spec.scorer_args.clone());
    Ok(TimedExternal {
        scorer: ExternalScorer {
            binary: timeout,
            extra_args: args,
        },
        identity: format!("sha256:{}", super::manifest::file_sha256(&spec.scorer)?),
    })
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Diagnostics {
    pub records: u64,
    pub per_output_mse_increase: Vec<f64>,
    pub worst_output_mse_increase: f64,
    pub stratum_mse_increase: Vec<Option<f64>>,
    pub stratum_records: Vec<u64>,
    pub parent_latency_ns_per_record: Vec<f64>,
    pub proposal_latency_ns_per_record: Vec<f64>,
    pub compilation_ms: f64,
}

// Evaluation reads validation only after every search has finished. Latency excludes IO and compilation.
pub fn diagnostics(
    parent: &CreatureExport,
    proposal: &CreatureExport,
    validation: &Path,
    latency: &Latency,
) -> Result<Diagnostics, String> {
    let cfg = TrainingDataConfig::new(parent.input, parent.output);
    let info = corpus_info(validation, &cfg)?;
    let started = Instant::now();
    let mut before = compile_creature(parent).map_err(|e| e.to_string())?;
    let mut after = compile_creature(proposal).map_err(|e| e.to_string())?;
    let compilation_ms = started.elapsed().as_secs_f64() * 1000.0;
    let mut output_delta = vec![0.0; parent.output];
    let mut stratum_delta = [0.0; 4];
    let mut strata_records = [0u64; 4];
    let mut probes = Vec::new();
    let records = for_each_selected_chunk(
        validation,
        &cfg,
        &[RecordRange {
            start: 0,
            len: info.record_count,
        }],
        4096,
        |chunk| {
            for record in 0..chunk.records {
                let input = &chunk.inputs[record * parent.input..(record + 1) * parent.input];
                if probes.len() < latency.records {
                    probes.push(input.to_vec());
                }
                let b = before.activate(input, parent.output);
                let a = after.activate(input, proposal.output);
                let stratum =
                    ((chunk.first_index + record as u64) * 4 / info.record_count) as usize;
                strata_records[stratum] += 1;
                for output in 0..parent.output {
                    let target = f64::from(chunk.targets[record * parent.output + output]);
                    let delta = (f64::from(a[output]) - target).powi(2)
                        - (f64::from(b[output]) - target).powi(2);
                    if !delta.is_finite() {
                        return Err("non-finite compiled validation output".into());
                    }
                    output_delta[output] += delta;
                    stratum_delta[stratum] += delta / parent.output as f64;
                }
            }
            Ok(std::ops::ControlFlow::Continue(()))
        },
    )?;
    for value in &mut output_delta {
        *value /= records as f64;
    }
    let stratum_mse_increase = stratum_delta
        .iter()
        .zip(strata_records)
        .map(|(value, count)| {
            if count == 0 {
                None
            } else {
                Some(*value / count as f64)
            }
        })
        .collect();
    let measure = |net: &mut neat_core::CompiledNetwork| -> Vec<f64> {
        let mut times = Vec::new();
        for pass in 0..latency.warmup_passes + latency.measured_passes {
            let start = Instant::now();
            for input in &probes {
                std::hint::black_box(net.activate(std::hint::black_box(input), parent.output));
            }
            if pass >= latency.warmup_passes {
                times.push(start.elapsed().as_nanos() as f64 / probes.len() as f64);
            }
        }
        times
    };
    let parent_latency = measure(&mut before);
    let proposal_latency = measure(&mut after);
    Ok(Diagnostics {
        records,
        worst_output_mse_increase: output_delta
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max),
        per_output_mse_increase: output_delta,
        stratum_mse_increase,
        stratum_records: strata_records.into(),
        parent_latency_ns_per_record: parent_latency,
        proposal_latency_ns_per_record: proposal_latency,
        compilation_ms,
    })
}
