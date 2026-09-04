//! Progressive screening economics against the fixed 5% control (Issue #104).
//!
//! The success metric is confirmed pruning gain per wall-clock hour, not sample
//! cleverness, so this measures the whole pipeline: screen, promote, full-score,
//! confirm. Both arms run the **real** [`screen_progressive`] over the **real**
//! cohort machinery — the only thing modelled is the scorer, because the honest
//! comparison needs a corpus far larger than a test fixture and a known truth to
//! score the missed-winner rate against.
//!
//! The scorer model is deliberately plain:
//!
//! * cost is linear in records read — `creatures × rate × corpus`, which is what
//!   a record-streaming scorer actually costs;
//! * a sampled score is the true score plus noise of `NOISE_SCALE / sqrt(n)`,
//!   the standard error of a mean over `n` sampled records;
//! * the full corpus is exact — it is the authority, and only it accepts.
//!
//! Run it with:
//!
//! ```bash
//! cargo run --release --example progressive_screen_bench
//! ```
//!
//! Wall-clock figures are the modelled scorer time, not this program's runtime:
//! a real scorer over a real corpus is minutes per full pass, and the point of
//! the ladder is the records it never reads.

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::path::Path;

use neat_ai_ockham::fixtures::{creature, neuron, synapse};
use neat_ai_ockham::scorer::{DirectoryScorer, ScoreResult, ScorerError, ScorerMode};
use neat_ai_ockham::screening::{ProgressiveConfig, ScreenLadder, screen_progressive};
use neat_ai_ockham::sweep::{CandidateKind, SweepCandidate};
use neat_core::CreatureExport;

/// Records in the modelled training corpus.
const CORPUS_RECORDS: f64 = 2_000_000.0;
/// Records the modelled scorer reads per millisecond, per creature.
const RECORDS_PER_MS: f64 = 20_000.0;
/// Standard error of a sampled score at one record; scales as `1/sqrt(n)`.
const NOISE_SCALE: f64 = 0.15;
/// Strict authoritative improvement, mirroring `--min-improvement`.
const MIN_IMPROVEMENT: f64 = 1e-6;
/// Screening batches simulated per arm.
const BATCHES: u64 = 300;
/// Candidates per batch.
const CANDIDATES: usize = 100;
/// Incumbent's true score.
const BASELINE: f64 = 0.900_000;

/// Deterministic 64-bit mixer — same inputs, same corpus slice, same noise.
fn mix(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Uniform `[0, 1)` from a seed.
fn unit(seed: u64) -> f64 {
    (mix(seed) >> 11) as f64 / (1u64 << 53) as f64
}

/// Standard normal from a seed (Box–Muller).
fn gauss(seed: u64) -> f64 {
    let u1 = unit(seed).max(1e-12);
    let u2 = unit(seed ^ 0xDEAD_BEEF_CAFE_F00D);
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

/// True full-corpus Δ of candidate `id` — the ground truth both arms are graded
/// against, and the thing neither arm can see before it pays for a full score.
///
/// A mature creature is mostly cuts that hurt: four in five are obvious losers,
/// most of the rest are close to neutral, and a few per cent genuinely help.
fn true_delta(id: u64) -> f64 {
    let bucket = unit(id ^ 0x1111);
    let spread = unit(id ^ 0x2222);
    if bucket < 0.80 {
        -0.05 + spread * 0.045
    } else if bucket < 0.98 {
        -0.002 + spread * 0.0025
    } else {
        1e-5 + spread * 0.002
    }
}

/// Scorer model: exact on the full corpus, noisy on a sample, priced by records.
#[derive(Default)]
struct ModelScorer {
    /// `stem -> candidate id`, refreshed per batch.
    ids: RefCell<BTreeMap<String, u64>>,
    /// Records read across every call so far.
    records: RefCell<f64>,
    /// Modelled scorer milliseconds across every call so far.
    ms: RefCell<f64>,
    /// Full-corpus creature scores paid for.
    full_scores: RefCell<u64>,
}

impl ModelScorer {
    fn charge(&self, creatures: usize, rate: f64) {
        let records = creatures as f64 * rate * CORPUS_RECORDS;
        *self.records.borrow_mut() += records;
        *self.ms.borrow_mut() += records / RECORDS_PER_MS;
    }

    /// One authoritative cohort: exact scores, full-corpus price.
    fn full_score(&self, ids: &[u64]) -> Vec<f64> {
        self.charge(ids.len() + 1, 1.0);
        *self.full_scores.borrow_mut() += ids.len() as u64 + 1;
        ids.iter().map(|id| true_delta(*id)).collect()
    }
}

impl DirectoryScorer for ModelScorer {
    fn score_directory(
        &self,
        creature_dir: &Path,
        _training_dir: &Path,
        mode: ScorerMode,
    ) -> Result<BTreeMap<String, ScoreResult>, ScorerError> {
        let (rate, phase) = match mode {
            ScorerMode::Sample { rate, phase } => (rate, phase),
            ScorerMode::Full => (1.0, 0),
        };
        let stems: Vec<String> = std::fs::read_dir(creature_dir)
            .map_err(|e| ScorerError::Spawn(e.to_string()))?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|x| x == "json"))
            .filter_map(|p| p.file_stem().map(|s| s.to_string_lossy().to_string()))
            .collect();
        self.charge(stems.len(), rate);
        let sampled = (rate * CORPUS_RECORDS).max(1.0);
        let se = NOISE_SCALE / sampled.sqrt();
        let ids = self.ids.borrow();
        let mut out = BTreeMap::new();
        for stem in stems {
            let (truth, id) = if stem == "baseline" {
                (BASELINE, 0)
            } else {
                let id = *ids.get(&stem).ok_or_else(|| {
                    ScorerError::Malformed(format!("bench: unknown candidate stem `{stem}`"))
                })?;
                (BASELINE + true_delta(id), id)
            };
            // The incumbent and every candidate share the sample context, so
            // the phase — not the creature — decides which records were read.
            let score = truth + se * gauss(mix(id ^ phase.wrapping_mul(0x5BF0_3635)));
            out.insert(
                stem,
                ScoreResult {
                    score,
                    error: 1.0 - score,
                    complexity_penalty: 0.0,
                    record_count: sampled as u64,
                    sample_rate: Some(rate),
                    gpu_backend: None,
                    cost_name: None,
                    time_taken: 0.0,
                },
            );
        }
        Ok(out)
    }

    fn identity(&self) -> String {
        "bench:model".into()
    }
}

/// Minimal valid creature; the model scorer keys off the file stem, not the
/// structure, so one shape serves for the incumbent and every candidate.
fn tiny() -> CreatureExport {
    creature(
        1,
        1,
        vec![
            neuron("hidden", "h_0", 0.0, Some("IDENTITY")),
            neuron("output", "output-0", 0.0, Some("IDENTITY")),
        ],
        vec![
            synapse("input-0", "h_0", 1.0),
            synapse("h_0", "output-0", 1.0),
        ],
    )
}

/// Candidate ids for one batch — identical across arms, so the two are graded
/// on the same population rather than on two different draws.
fn batch_ids(batch: u64) -> Vec<u64> {
    (0..CANDIDATES as u64)
        .map(|i| batch * CANDIDATES as u64 + i + 1)
        .collect()
}

/// What one arm achieved.
struct Arm {
    name: &'static str,
    screen_records: f64,
    screen_ms: f64,
    full_ms: f64,
    full_scores: u64,
    candidates: u64,
    promoted: u64,
    confirmed: u64,
    /// True winners the arm never sent to the full scorer.
    missed_winners: u64,
    /// True winners in the population it screened.
    true_winners: u64,
}

impl Arm {
    fn hours(&self) -> f64 {
        (self.screen_ms + self.full_ms) / 3_600_000.0
    }

    fn report(&self) {
        let hours = self.hours();
        println!("{}", self.name);
        println!(
            "  wall clock (modelled)     {:.2} h  ({:.2} h screen + {:.2} h full)",
            hours,
            self.screen_ms / 3_600_000.0,
            self.full_ms / 3_600_000.0
        );
        println!(
            "  candidates/hour           {:.0}",
            self.candidates as f64 / hours
        );
        println!(
            "  scorer-records/candidate  {:.0}",
            self.screen_records / self.candidates as f64
        );
        println!(
            "  full-scores/hour          {:.0}",
            self.full_scores as f64 / hours
        );
        println!(
            "  confirmed cuts/hour       {:.1}",
            self.confirmed as f64 / hours
        );
        println!(
            "  promoted/confirmed        {} / {}",
            self.promoted, self.confirmed
        );
        println!(
            "  missed-winner rate        {:.2}%  ({} of {} true winners)",
            100.0 * self.missed_winners as f64 / self.true_winners.max(1) as f64,
            self.missed_winners,
            self.true_winners
        );
    }
}

fn run_arm(name: &'static str, ladder: &ScreenLadder, workspace: &Path) -> Arm {
    let scorer = ModelScorer::default();
    let incumbent = tiny();
    let mut arm = Arm {
        name,
        screen_records: 0.0,
        screen_ms: 0.0,
        full_ms: 0.0,
        full_scores: 0,
        candidates: 0,
        promoted: 0,
        confirmed: 0,
        missed_winners: 0,
        true_winners: 0,
    };

    for batch in 0..BATCHES {
        let ids = batch_ids(batch);
        let mut candidates = Vec::with_capacity(ids.len());
        let mut stems = BTreeMap::new();
        for (i, id) in ids.iter().enumerate() {
            let stem = format!("c{i:03}");
            stems.insert(stem.clone(), *id);
            candidates.push(SweepCandidate {
                uuid: format!("h_{id}"),
                permutation_index: i,
                kind: CandidateKind::Ablation,
                stem,
                creature: incumbent.clone(),
            });
        }
        scorer.ids.replace(stems);
        arm.candidates += ids.len() as u64;
        arm.true_winners += ids
            .iter()
            .filter(|id| true_delta(**id) > MIN_IMPROVEMENT)
            .count() as u64;

        // Screen and full costs are attributed by bracketing each call, never
        // by subtracting one running total from another: the arithmetic that
        // "takes the full share back out" at the end silently drops the last
        // batch's cohort.
        let before_records = *scorer.records.borrow();
        let before_ms = *scorer.ms.borrow();
        let screen = screen_progressive(
            &scorer,
            workspace,
            &incumbent,
            candidates,
            ProgressiveConfig {
                ladder,
                threshold: 0.0,
                batch,
                remaining_after: 0,
                workspace: &workspace.join(name),
            },
        )
        .expect("model scorer never fails");
        arm.screen_records += *scorer.records.borrow() - before_records;
        arm.screen_ms += *scorer.ms.borrow() - before_ms;

        let promoted: Vec<u64> = screen
            .winners
            .iter()
            .map(|w| w.candidate.uuid.trim_start_matches("h_").parse().unwrap())
            .collect();
        arm.promoted += promoted.len() as u64;
        arm.missed_winners += ids
            .iter()
            .filter(|id| true_delta(**id) > MIN_IMPROVEMENT && !promoted.contains(id))
            .count() as u64;
        if promoted.is_empty() {
            continue;
        }
        // Only the full corpus may accept, in the benchmark exactly as in the
        // run: every promoted candidate is scored authoritatively.
        let before_full_ms = *scorer.ms.borrow();
        let deltas = scorer.full_score(&promoted);
        arm.full_ms += *scorer.ms.borrow() - before_full_ms;
        arm.confirmed += deltas.iter().filter(|d| **d > MIN_IMPROVEMENT).count() as u64;
    }
    arm.full_scores = *scorer.full_scores.borrow();
    arm
}

fn main() {
    let tmp = tempfile::tempdir().expect("workspace");
    println!(
        "modelled corpus {:.0} records, {:.0} records/ms, sampled SE = {NOISE_SCALE}/sqrt(n)",
        CORPUS_RECORDS, RECORDS_PER_MS
    );
    println!("{BATCHES} batches x {CANDIDATES} candidates per arm\n");

    let control = run_arm(
        "fixed 5% control",
        &ScreenLadder::single(0.05).unwrap(),
        tmp.path(),
    );
    let ladder = run_arm(
        "progressive 0.25% -> 1% -> 5%",
        &ScreenLadder::parse("0.0025:0.01,0.01:0.005,0.05", 0.01).unwrap(),
        tmp.path(),
    );

    control.report();
    println!();
    ladder.report();
    println!();
    let speed = control.hours() / ladder.hours();
    println!(
        "ladder vs control: {:.2}x the confirmed cuts per hour, {:.0}% of the screen records, \
         missed-winner rate {:.2}% vs {:.2}%",
        (ladder.confirmed as f64 / ladder.hours()) / (control.confirmed as f64 / control.hours()),
        100.0 * (ladder.screen_records / ladder.candidates as f64)
            / (control.screen_records / control.candidates as f64),
        100.0 * ladder.missed_winners as f64 / ladder.true_winners.max(1) as f64,
        100.0 * control.missed_winners as f64 / control.true_winners.max(1) as f64,
    );
    println!("wall-clock speed-up: {speed:.2}x");
}
