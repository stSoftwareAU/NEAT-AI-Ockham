//! Writes application-agnostic synthetic inputs for a ranking-v1 pilot (issue #261). See docs/experiments/ranking-v1.md.
//!
//! `inputs <dir>` writes eight tasks, each with its own parent and corpus.
//! `manifest <dir> --scorer <path> --code-revision <40-hex> [--cutoff-unix <u64>]`
//! curates each source task's `raw-candidates.jsonl` (written beforehand by
//! `neat_ai_ockham <parent> <corpus> --candidate-log ...`) into `history.jsonl`,
//! then writes `manifest.json`. Each source task also needs a `<task>/acquisition-ms`
//! file holding that ordinary run's wall time in milliseconds; the manifest refuses without it.
//! The manifest charges each source's `acquisitionMs` as that recorded time plus the
//! measured wall time of loading and curating its raw log, rounded up to whole
//! milliseconds, so curation is counted as a preparation cost of the history.
//! `inputs` also writes `<dir>/inputs-ms`, its own generation wall time rounded up to
//! whole milliseconds, and reports it on stderr. The manifest refuses without that file
//! and writes `preparationMs` as that time plus its own wall time up to its first write,
//! less the curation already charged to `acquisitionMs`, so the bench's
//! `totalResearchMs` counts this generator's preparation (the bench's own
//! preflight `check` stage is not timed).
//!
//! `manifest` treats the pilot directory as untrusted input. It resolves the directory once
//! and refuses any task path that is a symlink or whose resolved path leaves it. It also
//! refuses an input over its byte limit (see [`Limits`]), checked on the opened file and
//! again while reading, so a file that grows after the check is still caught. History and
//! manifest outputs are created exclusively, so an existing file or a dangling symlink
//! is refused rather than followed. The scorer is a separately authorised input: it may live
//! anywhere and may be a symlink, but must resolve to a regular file within its limit.
//! Every check and digest finishes before the first output is written.

use std::collections::{BTreeMap, BTreeSet};
use std::fs::OpenOptions;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use neat_ai_ockham::corpus::{corpus_info, write_bin_file};
use neat_ai_ockham::fixtures::{creature, neuron, synapse};
use neat_ai_ockham::incumbent::{now_unix, sha256_hex};
use neat_ai_ockham::model::pairwise::FitConfig;
use neat_ai_ockham::telemetry::{self, CandidateRecord};
use neat_core::training_data::TrainingDataConfig;
use neat_core::{CreatureExport, compile_creature, creature_to_json_pretty};
use serde_json::json;
use sha2::{Digest, Sha256};

/// Protocol name the bench's manifest must carry.
const PROTOCOL: &str = "ockham-ranking-v1";

/// Records per synthetic corpus.
const RECORDS: usize = 256;

/// Hidden-to-output synapses weaker than this are zeroed in the teacher.
const TEACHER_CUT: f64 = 0.06;

const USAGE: &str = "usage:\n  ranking_pilot inputs <new-directory>\n  ranking_pilot manifest <directory> --scorer <path> --code-revision <40-hex> [--cutoff-unix <u64>]";

/// `(id, split, family, lineage, corpus version)`. A task's index seeds its draws.
#[rustfmt::skip]
const TASKS: [(&str, &str, &str, &str, &str); 8] = [
    ("source-1", "source", "source-a", "lineage-1", "version-1"),
    ("source-2", "source", "source-b", "lineage-2", "version-2"),
    ("calibration-1", "calibration", "target-1", "lineage-3", "version-3"),
    ("calibration-2", "calibration", "target-2", "lineage-4", "version-4"),
    ("calibration-3", "calibration", "target-3", "lineage-5", "version-5"),
    ("validation-1", "validation", "target-1", "lineage-6", "version-6"),
    ("validation-2", "validation", "target-2", "lineage-7", "version-7"),
    ("validation-3", "validation", "target-3", "lineage-8", "version-8"),
];

/// Deterministic draw in [-1, 1] for task `task` and three indices.
fn draw(task: usize, a: usize, b: usize, c: usize) -> f64 {
    let mut hasher = Sha256::new();
    hasher.update(b"ranking-pilot-synthetic-v1");
    for value in [task, a, b, c] {
        hasher.update((value as u64).to_le_bytes());
    }
    let digest = hasher.finalize();
    let raw = u32::from_le_bytes(digest[..4].try_into().unwrap());
    f64::from(raw) / f64::from(u32::MAX) * 2.0 - 1.0
}

/// Parent for task `task`: four inputs, one output, useful, shared and dead-wood hidden neurons.
fn parent_creature(task: usize) -> CreatureExport {
    let (mut neurons, mut edges) = (Vec::new(), Vec::new());
    let input = |j: usize| format!("input-{j}");
    for k in 0..4 {
        let name = format!("u{k}");
        let bias = draw(task, 2, k, 0) * 0.1;
        neurons.push(neuron("hidden", &name, bias, Some("TANH")));
        for j in 0..4 {
            let weight = if j == k {
                1.5
            } else {
                draw(task, 3, k, j) * 0.05
            };
            edges.push(synapse(&input(j), &name, weight));
        }
        edges.push(synapse(&name, "output-0", 0.5));
    }
    for s in 0..2 {
        let name = format!("s{s}");
        neurons.push(neuron("hidden", &name, 0.0, Some("TANH")));
        for j in 0..4 {
            edges.push(synapse(&input(j), &name, 0.8 * draw(task, 4, s, j)));
        }
        edges.push(synapse(&name, "output-0", 0.3));
    }
    for d in 0..6 {
        let name = format!("d{d}");
        neurons.push(neuron("hidden", &name, 0.0, Some("TANH")));
        for j in 0..4 {
            edges.push(synapse(&input(j), &name, 0.5 * draw(task, 5, d, j)));
        }
        // Magnitude in [1e-3, 5e-2]: under the teacher's cut, yet not zero.
        let magnitude = 1e-3 + 0.049 * (draw(task, 6, d, 0) + 1.0) / 2.0;
        let sign = if draw(task, 7, d, 0) < 0.0 { -1.0 } else { 1.0 };
        edges.push(synapse(&name, "output-0", sign * magnitude));
    }
    for j in 0..4 {
        edges.push(synapse(&input(j), "output-0", 0.1 * draw(task, 8, j, 0)));
    }
    neurons.push(neuron("output", "output-0", 0.0, Some("IDENTITY")));
    creature(4, 1, neurons, edges)
}

/// The parent with every weak hidden-to-output synapse zeroed; its outputs are the corpus targets.
fn teacher_of(parent: &CreatureExport) -> CreatureExport {
    let mut teacher = parent.clone();
    for edge in &mut teacher.synapses {
        let from_hidden = !edge.from_uuid.starts_with("input-");
        if from_hidden && edge.to_uuid == "output-0" && edge.weight.abs() < TEACHER_CUT {
            edge.weight = 0.0;
        }
    }
    teacher
}

fn make_dir(path: &Path) -> Result<(), String> {
    std::fs::create_dir(path).map_err(|e| format!("cannot create '{}': {e}", path.display()))
}

/// Create `path` and write `bytes` into it, refusing to overwrite anything.
fn write_new_file(path: &Path, bytes: &[u8]) -> Result<(), String> {
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|e| format!("cannot create '{}': {e}", path.display()))?;
    file.write_all(bytes)
        .map_err(|e| format!("cannot write '{}': {e}", path.display()))
}

#[cfg(test)]
fn read(path: &Path) -> Result<Vec<u8>, String> {
    std::fs::read(path).map_err(|e| format!("cannot read '{}': {e}", path.display()))
}

/// Byte limits `write_manifest` enforces while reading and hashing its inputs.
#[derive(Clone, Copy, Debug)]
pub struct Limits {
    parent_bytes: u64,
    raw_log_bytes: u64,
    acquisition_bytes: u64,
    corpus_bytes: u64,
    scorer_bytes: u64,
}

impl Limits {
    /// Parent 16 MiB; raw log 64 MiB (the bench's history cap); acquisition and
    /// input-generation times (`acquisition-ms`, `inputs-ms`) 64 bytes;
    /// corpus 100 MB (also written to the manifest as `maxCorpusBytes`); scorer 1 GiB.
    const DEFAULT: Limits = Limits {
        parent_bytes: 16 * 1024 * 1024,
        raw_log_bytes: 64 * 1024 * 1024,
        acquisition_bytes: 64,
        corpus_bytes: 100_000_000,
        scorer_bytes: 1 << 30,
    };
}

/// Refuse `path` when anything exists at it, including a dangling symlink.
fn refuse_existing(path: &Path) -> Result<(), String> {
    if std::fs::symlink_metadata(path).is_ok() {
        return Err(format!(
            "'{}' already exists; refusing to overwrite",
            path.display()
        ));
    }
    Ok(())
}

/// Refuse a pilot-owned `path` that is missing, is a symlink, or resolves outside `root`.
fn contained(root: &Path, path: &Path) -> Result<PathBuf, String> {
    let meta = std::fs::symlink_metadata(path).map_err(|e| {
        if e.kind() == std::io::ErrorKind::NotFound {
            format!("required file '{}' is missing", path.display())
        } else {
            format!("cannot read '{}': {e}", path.display())
        }
    })?;
    if meta.file_type().is_symlink() {
        return Err(format!(
            "'{}' is a symlink; refusing to follow it",
            path.display()
        ));
    }
    let resolved = path
        .canonicalize()
        .map_err(|e| format!("cannot resolve '{}': {e}", path.display()))?;
    if !resolved.starts_with(root) {
        return Err(format!(
            "'{}' resolves outside the pilot directory '{}'",
            path.display(),
            root.display()
        ));
    }
    Ok(resolved)
}

fn over_limit(path: &Path, cap: u64) -> String {
    format!(
        "'{}' is larger than the {cap}-byte limit; refusing to read it",
        path.display()
    )
}

/// Read a contained regular file of at most `cap` bytes.
///
/// The opened file's length is checked before reading; the read itself is capped
/// at `cap + 1` bytes, so a file that grows after the check is also refused.
fn read_bounded(root: &Path, path: &Path, cap: u64) -> Result<Vec<u8>, String> {
    let resolved = contained(root, path)?;
    let file = std::fs::File::open(&resolved)
        .map_err(|e| format!("cannot read '{}': {e}", path.display()))?;
    let meta = file
        .metadata()
        .map_err(|e| format!("cannot read '{}': {e}", path.display()))?;
    if !meta.is_file() {
        return Err(format!("'{}' is not a regular file", path.display()));
    }
    if meta.len() > cap {
        return Err(over_limit(path, cap));
    }
    let mut bytes = Vec::new();
    file.take(cap.saturating_add(1))
        .read_to_end(&mut bytes)
        .map_err(|e| format!("cannot read '{}': {e}", path.display()))?;
    if bytes.len() as u64 > cap {
        return Err(over_limit(path, cap));
    }
    Ok(bytes)
}

/// Stream the file at `path` into `hasher` in 64 KiB chunks, adding to `total` and refusing past `cap`.
fn hash_capped(path: &Path, hasher: &mut Sha256, total: &mut u64, cap: u64) -> Result<(), String> {
    let mut file =
        std::fs::File::open(path).map_err(|e| format!("cannot read '{}': {e}", path.display()))?;
    let meta = file
        .metadata()
        .map_err(|e| format!("cannot read '{}': {e}", path.display()))?;
    if !meta.is_file() {
        return Err(format!("'{}' is not a regular file", path.display()));
    }
    if total.checked_add(meta.len()).is_none_or(|sum| sum > cap) {
        return Err(over_limit(path, cap));
    }
    let mut buffer = vec![0u8; 64 * 1024];
    loop {
        let n = file
            .read(&mut buffer)
            .map_err(|e| format!("cannot read '{}': {e}", path.display()))?;
        if n == 0 {
            return Ok(());
        }
        *total = total
            .checked_add(n as u64)
            .filter(|sum| *sum <= cap)
            .ok_or_else(|| over_limit(path, cap))?;
        hasher.update(&buffer[..n]);
    }
}

/// SHA-256 of the scorer, streamed. The scorer may be a symlink but must resolve to a regular file.
fn scorer_sha256(scorer: &Path, cap: u64) -> Result<String, String> {
    let mut hasher = Sha256::new();
    hash_capped(scorer, &mut hasher, &mut 0, cap)?;
    Ok(format!("{:x}", hasher.finalize()))
}

/// Absolute form of `path`, as text for the manifest.
fn absolute(path: &Path) -> Result<String, String> {
    std::path::absolute(path)
        .map(|p| p.display().to_string())
        .map_err(|e| format!("cannot make '{}' absolute: {e}", path.display()))
}

/// Write the synthetic inputs for every task into the new directory `dir`.
///
/// Also writes `<dir>/inputs-ms`: the generation wall time, rounded up to whole
/// milliseconds, which `manifest` charges as part of `preparationMs`. Returns
/// that time.
///
/// # Errors
///
/// Refuses an existing `dir` (nothing inside it is touched), and reports any
/// I/O or creature failure with the path at fault.
pub fn write_inputs(dir: &Path) -> Result<u64, String> {
    let started = Instant::now();
    make_dir(dir).map_err(|e| format!("new pilot directory: {e}"))?;
    for (task, (id, ..)) in TASKS.iter().enumerate() {
        let task_dir = dir.join(id);
        make_dir(&task_dir)?;
        let parent = parent_creature(task);
        let parent_json = creature_to_json_pretty(&parent).map_err(|e| e.to_string())?;
        write_new_file(&task_dir.join("parent.json"), parent_json.as_bytes())?;

        let mut net = compile_creature(&teacher_of(&parent)).map_err(|e| e.to_string())?;
        let rows: Vec<(Vec<f32>, Vec<f32>)> = (0..RECORDS)
            .map(|row| {
                let input: Vec<f32> = (0..4).map(|j| draw(task, 1, row, j) as f32).collect();
                let target = net.activate(&input, 1);
                (input, target)
            })
            .collect();
        let corpus_dir = task_dir.join("corpus");
        make_dir(&corpus_dir)?;
        let bin = corpus_dir.join("records.bin");
        write_bin_file(&bin, &rows)
            .map_err(|e| format!("cannot write '{}': {e}", bin.display()))?;
    }
    let elapsed_ms = ceil_ms(started.elapsed());
    write_new_file(&dir.join("inputs-ms"), format!("{elapsed_ms}\n").as_bytes())?;
    Ok(elapsed_ms)
}

/// Rows kept for one source task, with how many were dropped for each reason.
struct Curated {
    kept: Vec<CandidateRecord>,
    other_incumbent: usize,
    duplicate: usize,
}

/// Whether a row carries both a full-corpus Δ and a sampled Δ.
fn is_labelled(row: &CandidateRecord) -> bool {
    row.full_delta.is_some() && row.sample_delta.is_some()
}

/// Keep one row per `(uuid, kind, merged_with)` among rows naming the parent checksum.
///
/// The kept row is the first labelled one for its key, or the first row for the key
/// when none is labelled. An ordinary sweep re-screens the same candidate many times
/// and only promoted ones get a full score, so first-row-wins discards the labelled observation.
/// Kept rows stay in the order they had in the log.
fn curate(rows: Vec<CandidateRecord>, parent_sha: &str) -> Curated {
    let mut other_incumbent = 0;
    let mut parent_rows = 0;
    // Position in `rows` of the row kept for each key.
    let mut kept_at: BTreeMap<(String, String, Option<String>), usize> = BTreeMap::new();
    for (position, row) in rows.iter().enumerate() {
        if row.creature_checksum != parent_sha {
            other_incumbent += 1;
            continue;
        }
        parent_rows += 1;
        let key = (row.uuid.clone(), row.kind.clone(), row.merged_with.clone());
        match kept_at.get(&key).copied() {
            None => {
                kept_at.insert(key, position);
            }
            Some(current) if is_labelled(row) && !is_labelled(&rows[current]) => {
                kept_at.insert(key, position);
            }
            Some(_) => {}
        }
    }
    let kept_positions: BTreeSet<usize> = kept_at.into_values().collect();
    let duplicate = parent_rows - kept_positions.len();
    let kept = rows
        .into_iter()
        .enumerate()
        .filter(|(position, _)| kept_positions.contains(position))
        .map(|(_, row)| row)
        .collect();
    Curated {
        kept,
        other_incumbent,
        duplicate,
    }
}

/// SHA-256 over the concatenated `.bin` files of `corpus`, in sorted path order.
///
/// Refuses a symlinked entry, an empty corpus, and more than `max_bytes` in total.
fn corpus_sha256(corpus: &Path, max_bytes: u64) -> Result<String, String> {
    let entries = std::fs::read_dir(corpus)
        .map_err(|e| format!("cannot read '{}': {e}", corpus.display()))?;
    let mut bins = Vec::new();
    for entry in entries {
        let entry = entry.map_err(|e| format!("cannot read '{}': {e}", corpus.display()))?;
        let path = entry.path();
        let kind = entry
            .file_type()
            .map_err(|e| format!("cannot read '{}': {e}", path.display()))?;
        if kind.is_symlink() {
            return Err(format!(
                "'{}' is a symlink; refusing to follow it",
                path.display()
            ));
        }
        if path.extension().is_some_and(|ext| ext == "bin") {
            bins.push(path);
        }
    }
    if bins.is_empty() {
        return Err(format!("'{}' holds no .bin file", corpus.display()));
    }
    bins.sort();
    let mut hasher = Sha256::new();
    let mut total = 0u64;
    for bin in &bins {
        hash_capped(bin, &mut hasher, &mut total, max_bytes)?;
    }
    Ok(format!("{:x}", hasher.finalize()))
}

/// A recorded wall time in whole milliseconds, read from the contained file `path`.
///
/// The file is required: a missing file is an error naming it, so an unrecorded
/// cost never silently becomes 0.
fn recorded_ms(root: &Path, path: &Path, cap: u64) -> Result<u64, String> {
    let bytes = read_bounded(root, path, cap)?;
    let text = String::from_utf8(bytes).map_err(|e| format!("'{}': {e}", path.display()))?;
    text.trim()
        .parse::<u64>()
        .map_err(|e| format!("'{}': {e}", path.display()))
}

/// The operator-recorded wall time of the ordinary run, from `<task>/acquisition-ms`.
fn acquisition_ms(root: &Path, task_dir: &Path, cap: u64) -> Result<u64, String> {
    recorded_ms(root, &task_dir.join("acquisition-ms"), cap)
}

/// `duration` in whole milliseconds, rounded up so non-zero work is never 0 ms.
fn ceil_ms(duration: Duration) -> u64 {
    u64::try_from(duration.as_nanos().div_ceil(1_000_000)).unwrap_or(u64::MAX)
}

/// The acquisition cost charged to a source history: the recorded ordinary-run
/// time plus the curation wall time, rounded up so non-zero work is never 0 ms.
fn charged_acquisition_ms(recorded_ms: u64, curation: Duration) -> u64 {
    recorded_ms.saturating_add(ceil_ms(curation))
}

/// Everything known about one task before any output is written.
struct Prepared {
    parent_sha: String,
    corpus_sha: String,
    identity: String,
    /// Serialised curated history, and its acquisition cost, for a source task.
    history: Option<(Vec<u8>, u64)>,
}

/// Curate each source history, then write `<dir>/manifest.json` naming every task.
///
/// The manifest's `preparationMs` is the generation time recorded in `<dir>/inputs-ms`
/// plus this step's own wall time from entry up to just before its first write, less
/// the curation time already charged to the sources' `acquisitionMs`, rounded up to
/// whole milliseconds (saturating).
///
/// # Errors
///
/// Refuses, naming the path (and the task id for task files):
/// - an existing `manifest.json` or `history.jsonl`, including a dangling symlink;
/// - a symlinked task path, corpus entry, raw log or `inputs-ms`, and any pilot
///   path whose resolved location leaves the pilot directory;
/// - a `parent.json` over 16 MiB, a `raw-candidates.jsonl` over 64 MiB, an
///   `acquisition-ms` or `inputs-ms` over 64 bytes, a corpus over
///   `limits.corpus_bytes` in total (or with no `.bin` file), and a scorer over
///   `limits.scorer_bytes` or that is not a regular file (the scorer may itself be
///   a symlink);
/// - a source task with no fully scored row (full and sampled delta) for its parent,
///   a source task with no `acquisition-ms` file, and a pilot directory with no
///   `inputs-ms` file.
///
/// Every read, check and digest for all tasks finishes before the first write, so a
/// refusal writes no history and no manifest.
pub fn write_manifest(
    dir: &Path,
    scorer: &Path,
    code_revision: &str,
    cutoff_unix: u64,
    limits: &Limits,
) -> Result<(), String> {
    let started = Instant::now();
    let root = dir
        .canonicalize()
        .map_err(|e| format!("cannot resolve '{}': {e}", dir.display()))?;
    let manifest_path = dir.join("manifest.json");
    refuse_existing(&manifest_path)?;
    let inputs_ms = recorded_ms(&root, &dir.join("inputs-ms"), limits.acquisition_bytes)?;
    let scorer_sha = scorer_sha256(scorer, limits.scorer_bytes)?;

    let mut prepared = Vec::new();
    // Curation time already charged to the sources' `acquisitionMs`.
    let mut curation_total = Duration::ZERO;
    for (id, split, ..) in TASKS {
        let task_dir = dir.join(id);
        contained(&root, &task_dir).map_err(|e| format!("{id}: {e}"))?;
        let parent_bytes = read_bounded(&root, &task_dir.join("parent.json"), limits.parent_bytes)
            .map_err(|e| format!("{id}: {e}"))?;
        let parent_sha = sha256_hex(&parent_bytes);
        let corpus_dir = task_dir.join("corpus");
        contained(&root, &corpus_dir).map_err(|e| format!("{id}: {e}"))?;
        let corpus_sha =
            corpus_sha256(&corpus_dir, limits.corpus_bytes).map_err(|e| format!("{id}: {e}"))?;
        let identity = corpus_info(&corpus_dir, &TrainingDataConfig::new(4, 1))
            .map_err(|e| format!("{id}: {e}"))?
            .identity;

        let mut history = None;
        if split == "source" {
            refuse_existing(&task_dir.join("history.jsonl")).map_err(|e| format!("{id}: {e}"))?;
            let curation_started = Instant::now();
            let raw = task_dir.join("raw-candidates.jsonl");
            let raw_bytes = read_bounded(&root, &raw, limits.raw_log_bytes)
                .map_err(|e| format!("{id}: {e}"))?;
            let rows = telemetry::load_from_reader(&raw_bytes[..], &raw.display().to_string())?;
            let total = rows.len();
            let curated = curate(rows, &parent_sha);
            if !curated.kept.iter().any(is_labelled) {
                return Err(format!(
                    "{id}: no fully scored row (full and sampled delta) for the parent checksum in '{}'",
                    raw.display()
                ));
            }
            let mut buffer = String::new();
            for record in &curated.kept {
                buffer.push_str(&serde_json::to_string(record).map_err(|e| e.to_string())?);
                buffer.push('\n');
            }
            let curation = curation_started.elapsed();
            curation_total = curation_total.saturating_add(curation);
            let recorded = acquisition_ms(&root, &task_dir, limits.acquisition_bytes)
                .map_err(|e| format!("{id}: {e}"))?;
            let charged = charged_acquisition_ms(recorded, curation);
            let kept = curated.kept.len();
            eprintln!(
                "{id}: kept {kept}, dropped {} ({} other incumbent, {} duplicate) of {total} rows; \
                 acquisitionMs {charged} = recorded {recorded} + curation {} ms",
                total - kept,
                curated.other_incumbent,
                curated.duplicate,
                charged - recorded
            );
            history = Some((buffer.into_bytes(), charged));
        }
        prepared.push(Prepared {
            parent_sha,
            corpus_sha,
            identity,
            history,
        });
    }

    // This step's own work up to the first write, less the curation already charged.
    let manifest_step_ms = ceil_ms(started.elapsed().saturating_sub(curation_total));
    let preparation_ms = inputs_ms.saturating_add(manifest_step_ms);
    eprintln!(
        "preparationMs {preparation_ms} = inputs {inputs_ms} + manifest step {manifest_step_ms} ms \
         (excluding {} ms of curation charged to sources)",
        ceil_ms(curation_total)
    );

    let mut tasks = Vec::new();
    for ((id, split, family, lineage, version), task) in TASKS.into_iter().zip(&prepared) {
        let task_dir = dir.join(id);
        let history = match &task.history {
            Some((bytes, charged)) => {
                let path = task_dir.join("history.jsonl");
                write_new_file(&path, bytes)?;
                Some(json!({
                    "path": absolute(&path)?,
                    "sha256": sha256_hex(bytes),
                    "acquisitionMs": charged,
                }))
            }
            None => None,
        };
        tasks.push(json!({
            "id": id,
            "family": family,
            "lineage": lineage,
            "corpusVersion": version,
            "split": split,
            "parent": absolute(&task_dir.join("parent.json"))?,
            "parentSha256": task.parent_sha,
            "corpus": absolute(&task_dir.join("corpus"))?,
            "corpusSha256": task.corpus_sha,
            "corpusIdentity": task.identity,
            "history": history,
        }));
    }

    let fit = FitConfig {
        epochs: 200,
        learning_rate: 0.1,
        l2: 0.001,
        max_pairs: 1024,
    };
    let manifest = json!({
        "formatVersion": 1,
        "protocol": PROTOCOL,
        "codeRevision": code_revision,
        "scorer": absolute(scorer)?,
        "scorerSha256": scorer_sha,
        "scorerArgs": ["--gpu", "off", "--cost", "MSE"],
        "historicalCutoffUnix": cutoff_unix,
        "seeds": [7, 19],
        "pilotVisits": 8,
        "auditVisitsCeiling": 40,
        "minimumCompletePool": 20,
        "minimumIndependentGroups": 3,
        "maxTaskMs": 600_000,
        "maxRunMs": 2_400_000,
        "scorerTimeoutSecs": 60,
        "statsRecords": 512,
        "maxCorpusBytes": limits.corpus_bytes,
        "evaluationBudgets": [2, 5],
        "calibrationBudgets": [0, 8],
        "historyFractions": [0.0, 0.5, 1.0],
        "fit": serde_json::to_value(fit).map_err(|e| e.to_string())?,
        "bootstrapReplicates": 1000,
        "minimumRecallGain": 0.05,
        "maximumFalseRejection": 0.1,
        "maximumCostRatio": 1.0,
        "minimumImprovement": 1e-6,
        "preparationMs": preparation_ms,
        "tasks": tasks,
    });
    let text = serde_json::to_string_pretty(&manifest).map_err(|e| e.to_string())?;
    write_new_file(&manifest_path, format!("{text}\n").as_bytes())
}

/// A parsed subcommand. `Manifest` holds `(dir, scorer, code revision, cutoff)`.
enum Command {
    Inputs(PathBuf),
    Manifest(PathBuf, PathBuf, String, u64),
}

fn parse(args: &[String]) -> Result<Command, String> {
    let (command, rest) = args.split_first().ok_or("missing subcommand")?;
    match (command.as_str(), rest) {
        ("inputs", [dir]) => Ok(Command::Inputs(PathBuf::from(dir))),
        ("manifest", [dir, flags @ ..]) => {
            let (mut scorer, mut revision, mut cutoff) = (None, None, None);
            for pair in flags.chunks(2) {
                let [flag, value] = pair else {
                    return Err(format!("'{}' needs a value", pair[0]));
                };
                match flag.as_str() {
                    "--scorer" => scorer = Some(PathBuf::from(value)),
                    "--code-revision" => revision = Some(value.clone()),
                    "--cutoff-unix" => {
                        cutoff = Some(
                            value
                                .parse::<u64>()
                                .map_err(|e| format!("--cutoff-unix: {e}"))?,
                        );
                    }
                    other => return Err(format!("unknown argument '{other}'")),
                }
            }
            let scorer = scorer.ok_or("--scorer is required")?;
            let revision = revision.ok_or("--code-revision is required")?;
            if revision.len() != 40 || !revision.bytes().all(|b| b.is_ascii_hexdigit()) {
                return Err("--code-revision must be a 40-character hex digest".into());
            }
            let cutoff = cutoff.unwrap_or_else(now_unix);
            Ok(Command::Manifest(
                PathBuf::from(dir),
                scorer,
                revision,
                cutoff,
            ))
        }
        _ => Err("expected `inputs <dir>` or `manifest <dir> ...`".into()),
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let command = match parse(&args) {
        Ok(command) => command,
        Err(message) => {
            eprintln!("{message}\n{USAGE}");
            std::process::exit(2);
        }
    };
    let result = match command {
        Command::Inputs(dir) => write_inputs(&dir).map(|elapsed_ms| {
            eprintln!("inputs: generated in {elapsed_ms} ms");
        }),
        Command::Manifest(dir, scorer, revision, cutoff) => {
            write_manifest(&dir, &scorer, &revision, cutoff, &Limits::DEFAULT)
        }
    };
    if let Err(message) = result {
        eprintln!("error: {message}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const CUTOFF: u64 = 1_800_000_000;

    /// A `CandidateRecord` in the shape the scorer writes, for incumbent `checksum`.
    fn row(checksum: &str, uuid: &str, full_delta: Option<f64>) -> CandidateRecord {
        serde_json::from_value(json!({
            "version": telemetry::CANDIDATE_LOG_FORMAT_VERSION,
            "unixSecs": 1,
            "host": "test",
            "corpusIdentity": "corpus",
            "creatureChecksum": checksum,
            "ordering": "test",
            "seed": 7,
            "uuid": uuid,
            "kind": "ablation",
            "features": {},
            "fullDelta": full_delta,
            "outcome": "rejected",
            "growthUnitsRemoved": 0.0,
            "scorerMs": 1,
        }))
        .unwrap()
    }

    /// A fresh pilot directory from `write_inputs`, and a scorer file to name in the manifest.
    fn pilot() -> (tempfile::TempDir, PathBuf, PathBuf) {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join("pilot");
        write_inputs(&dir).unwrap();
        let scorer = tmp.path().join("rust_scorer");
        std::fs::write(&scorer, b"scorer bytes").unwrap();
        (tmp, dir, scorer)
    }

    fn parent_sha(dir: &Path, id: &str) -> String {
        sha256_hex(&read(&dir.join(id).join("parent.json")).unwrap())
    }

    fn write_raw(dir: &Path, id: &str, rows: &[CandidateRecord]) {
        telemetry::append(&dir.join(id).join("raw-candidates.jsonl"), rows).unwrap();
    }

    /// A row with both a full-corpus Δ and a sampled Δ, as a fully scored observation has.
    fn labelled_row(checksum: &str, uuid: &str) -> CandidateRecord {
        let mut record = row(checksum, uuid, Some(0.1));
        record.sample_delta = Some(0.5);
        record
    }

    /// Record the ordinary run's acquisition wall time (5 ms) for source task `id`.
    fn write_acquisition(dir: &Path, id: &str) {
        std::fs::write(dir.join(id).join("acquisition-ms"), "5").unwrap();
    }

    /// Give each source task a fully scored raw log for its own parent, and its acquisition time.
    fn write_labelled_sources(dir: &Path) {
        for id in ["source-1", "source-2"] {
            let parent = parent_sha(dir, id);
            write_raw(dir, id, &[labelled_row(&parent, "h0")]);
            write_acquisition(dir, id);
        }
    }

    fn keys(value: &serde_json::Value) -> BTreeSet<String> {
        value.as_object().unwrap().keys().cloned().collect()
    }

    #[test]
    fn inputs_are_deterministic_distinct_and_refuse_overwrite() {
        let tmp = tempfile::tempdir().unwrap();
        let (first, second) = (tmp.path().join("first"), tmp.path().join("second"));
        write_inputs(&first).unwrap();
        write_inputs(&second).unwrap();

        for (id, ..) in TASKS {
            for file in ["parent.json", "corpus/records.bin"] {
                let path = first.join(id).join(file);
                let twin = read(&second.join(id).join(file)).unwrap();
                assert_eq!(
                    read(&path).unwrap(),
                    twin,
                    "{} differs between writes",
                    path.display()
                );
            }
            let info = corpus_info(
                &first.join(id).join("corpus"),
                &TrainingDataConfig::new(4, 1),
            );
            assert_eq!(
                info.unwrap().record_count,
                RECORDS as u64,
                "{id} record count"
            );
        }

        let parents: Vec<Vec<u8>> = TASKS
            .iter()
            .map(|(id, ..)| read(&first.join(id).join("parent.json")).unwrap())
            .collect();
        for (i, a) in parents.iter().enumerate() {
            for (j, b) in parents.iter().enumerate().skip(i + 1) {
                assert_ne!(a, b, "parent bytes of tasks {i} and {j} are identical");
            }
        }

        let existing = tmp.path().join("existing");
        std::fs::create_dir(&existing).unwrap();
        std::fs::write(existing.join("keep.txt"), b"do not touch").unwrap();
        let err = write_inputs(&existing).expect_err("an existing directory must be refused");
        assert!(
            err.contains(&existing.display().to_string()),
            "error should name the path: {err}"
        );
        assert_eq!(read(&existing.join("keep.txt")).unwrap(), b"do not touch");
        assert!(!existing.join("source-1").exists());
    }

    #[test]
    fn curation_keeps_one_parent_row_per_key_and_writes_the_manifest() {
        let (_tmp, dir, scorer) = pilot();
        let other = sha256_hex(b"another incumbent");
        for id in ["source-1", "source-2"] {
            let parent = parent_sha(&dir, id);
            // An unlabelled screen comes first; the labelled re-screen of the same key follows.
            let mut labelled = row(&parent, "h0", Some(0.2));
            labelled.sample_delta = Some(0.5);
            let rows = [
                row(&parent, "h0", None),
                labelled,
                row(&other, "h1", Some(0.3)),
            ];
            write_raw(&dir, id, &rows);
            write_acquisition(&dir, id);
        }
        write_manifest(&dir, &scorer, &"a".repeat(40), CUTOFF, &Limits::DEFAULT).unwrap();
        assert!(
            write_manifest(&dir, &scorer, &"a".repeat(40), CUTOFF, &Limits::DEFAULT).is_err(),
            "a second manifest must be refused"
        );

        for id in ["source-1", "source-2"] {
            let history = telemetry::load(&dir.join(id).join("history.jsonl")).unwrap();
            assert_eq!(history.len(), 1, "{id} history rows");
            assert_eq!(
                history[0].creature_checksum,
                parent_sha(&dir, id),
                "{id} checksum"
            );
            assert_eq!(
                history[0].full_delta,
                Some(0.2),
                "{id} keeps the labelled observation"
            );
        }

        let manifest: serde_json::Value =
            serde_json::from_slice(&read(&dir.join("manifest.json")).unwrap()).unwrap();
        let tasks = manifest["tasks"].as_array().unwrap();
        assert_eq!(tasks.len(), 8, "manifest task count");
        for task in tasks {
            let id = task["id"].as_str().unwrap();
            let history = &task["history"];
            if task["split"].as_str() == Some("source") {
                let digest = sha256_hex(&read(&dir.join(id).join("history.jsonl")).unwrap());
                assert_eq!(digest.len(), 64, "{id} digest length");
                assert_eq!(
                    history["sha256"].as_str(),
                    Some(digest.as_str()),
                    "{id} sha256"
                );
                let charged = history["acquisitionMs"].as_u64().unwrap();
                assert!(
                    charged > 5,
                    "{id} acquisitionMs {charged} must add curation time to the recorded 5 ms"
                );
            } else {
                assert!(
                    history.is_null(),
                    "{id} is a target task but carries a history"
                );
            }
        }
    }

    #[test]
    fn charged_acquisition_adds_curation_rounded_up() {
        assert_eq!(
            charged_acquisition_ms(5, Duration::ZERO),
            5,
            "no curation work"
        );
        assert_eq!(
            charged_acquisition_ms(5, Duration::from_nanos(1)),
            6,
            "any curation work costs at least 1 ms"
        );
        assert_eq!(
            charged_acquisition_ms(5, Duration::from_millis(2)),
            7,
            "whole milliseconds are not rounded further"
        );
        assert_eq!(
            charged_acquisition_ms(5, Duration::from_micros(2_001)),
            8,
            "a partial millisecond rounds up"
        );
        assert_eq!(
            charged_acquisition_ms(u64::MAX, Duration::from_millis(1)),
            u64::MAX,
            "saturates instead of overflowing"
        );
    }

    #[test]
    fn manifest_keys_match_the_template() {
        let (_tmp, dir, scorer) = pilot();
        write_labelled_sources(&dir);
        write_manifest(&dir, &scorer, &"a".repeat(40), CUTOFF, &Limits::DEFAULT).unwrap();
        let template_path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../docs/experiments/ranking-v1.template.json");
        let template: serde_json::Value =
            serde_json::from_slice(&read(&template_path).unwrap()).unwrap();
        let written: serde_json::Value =
            serde_json::from_slice(&read(&dir.join("manifest.json")).unwrap()).unwrap();

        assert_eq!(keys(&written), keys(&template), "top-level manifest keys");
        assert_eq!(keys(&written["fit"]), keys(&template["fit"]), "fit keys");
        assert_eq!(
            keys(&written["tasks"][0]),
            keys(&template["tasks"][0]),
            "task keys"
        );
        assert_eq!(
            keys(&written["tasks"][0]["history"]),
            keys(&template["tasks"][0]["history"]),
            "history keys"
        );
        assert_eq!(written["protocol"].as_str(), Some(PROTOCOL));
    }

    #[test]
    fn source_without_fully_scored_parent_row_is_refused_naming_the_task() {
        let (_tmp, dir, scorer) = pilot();
        let parent = parent_sha(&dir, "source-1");
        let other = sha256_hex(b"another incumbent");
        // The row with a full-corpus Δ alone has no sampled Δ, so it is not fully scored.
        write_raw(
            &dir,
            "source-1",
            &[
                row(&parent, "h0", None),
                row(&parent, "h1", Some(0.3)),
                row(&other, "h2", Some(0.2)),
            ],
        );
        write_raw(
            &dir,
            "source-2",
            &[labelled_row(&parent_sha(&dir, "source-2"), "h0")],
        );
        write_acquisition(&dir, "source-1");
        write_acquisition(&dir, "source-2");

        let err = write_manifest(&dir, &scorer, &"a".repeat(40), CUTOFF, &Limits::DEFAULT)
            .expect_err("a source without a fully scored parent row must be refused");
        assert!(
            err.contains("source-1"),
            "error should name the task: {err}"
        );
        assert!(!dir.join("source-1").join("history.jsonl").exists());
        assert!(!dir.join("source-2").join("history.jsonl").exists());
        assert!(!dir.join("manifest.json").exists());
    }

    #[test]
    fn source_without_acquisition_ms_is_refused_naming_the_task() {
        let (_tmp, dir, scorer) = pilot();
        write_labelled_sources(&dir);
        std::fs::remove_file(dir.join("source-1").join("acquisition-ms")).unwrap();

        let err = write_manifest(&dir, &scorer, &"a".repeat(40), CUTOFF, &Limits::DEFAULT)
            .expect_err("a source without acquisition-ms must be refused");
        assert!(
            err.contains("source-1"),
            "error should name the task: {err}"
        );
        assert!(!dir.join("source-1").join("history.jsonl").exists());
        assert!(!dir.join("source-2").join("history.jsonl").exists());
        assert!(!dir.join("manifest.json").exists());

        // Positive control: the same fixture, with only the missing file added, is accepted.
        write_acquisition(&dir, "source-1");
        write_manifest(&dir, &scorer, &"a".repeat(40), CUTOFF, &Limits::DEFAULT).unwrap();
        assert!(dir.join("manifest.json").exists());
    }

    const REV: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";

    fn try_manifest(dir: &Path, scorer: &Path, limits: &Limits) -> Result<(), String> {
        write_manifest(dir, scorer, REV, CUTOFF, limits)
    }

    /// Assert a refusal naming `cause`, with nothing written by the refused run.
    fn assert_refused(dir: &Path, scorer: &Path, limits: &Limits, cause: &str) {
        let err = try_manifest(dir, scorer, limits).expect_err("must be refused");
        assert!(err.contains(cause), "error should contain '{cause}': {err}");
        for id in ["source-1", "source-2"] {
            let history = dir.join(id).join("history.jsonl");
            assert!(
                std::fs::symlink_metadata(&history).is_err()
                    || std::fs::symlink_metadata(&history)
                        .unwrap()
                        .file_type()
                        .is_symlink(),
                "{id} history must not be written"
            );
        }
        assert!(!dir.join("manifest.json").exists());
    }

    fn assert_accepted(dir: &Path, scorer: &Path, limits: &Limits) {
        try_manifest(dir, scorer, limits).unwrap();
        assert!(dir.join("manifest.json").exists());
    }

    fn file_len(path: &Path) -> u64 {
        std::fs::metadata(path).unwrap().len()
    }

    #[test]
    fn inputs_record_their_generation_time() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join("pilot");
        let returned = write_inputs(&dir).unwrap();
        let text = String::from_utf8(read(&dir.join("inputs-ms")).unwrap()).unwrap();
        let recorded: u64 = text.trim().parse().unwrap();
        assert!(recorded > 0, "generation work rounds up to at least 1 ms");
        assert_eq!(recorded, returned);
    }

    #[test]
    fn manifest_preparation_includes_the_recorded_inputs_time() {
        // Far above any plausible step time, so a dropped inputs term cannot hide behind it.
        const RECORDED: u64 = 1_000_000_000;
        let (_tmp, dir, scorer) = pilot();
        write_labelled_sources(&dir);
        std::fs::write(dir.join("inputs-ms"), RECORDED.to_string()).unwrap();
        assert_accepted(&dir, &scorer, &Limits::DEFAULT);
        let manifest: serde_json::Value =
            serde_json::from_slice(&read(&dir.join("manifest.json")).unwrap()).unwrap();
        let preparation = manifest["preparationMs"].as_u64().unwrap();
        assert!(
            preparation > RECORDED,
            "preparationMs {preparation} must include the recorded {RECORDED} ms \
             and the manifest step's own non-zero work"
        );
        assert!(
            preparation < RECORDED + 600_000,
            "preparationMs {preparation} must charge the recorded {RECORDED} ms once, \
             plus only the manifest step's own time"
        );
    }

    #[test]
    fn pilot_without_inputs_ms_is_refused_naming_it() {
        let (_tmp, dir, scorer) = pilot();
        write_labelled_sources(&dir);
        std::fs::remove_file(dir.join("inputs-ms")).unwrap();
        assert_refused(&dir, &scorer, &Limits::DEFAULT, "inputs-ms");

        // Positive control: the same fixture, with only the missing file added, is accepted.
        std::fs::write(dir.join("inputs-ms"), "3").unwrap();
        assert_accepted(&dir, &scorer, &Limits::DEFAULT);
    }

    #[cfg(unix)]
    #[test]
    fn symlinked_inputs_ms_is_refused() {
        use std::os::unix::fs::symlink;
        let (tmp, dir, scorer) = pilot();
        write_labelled_sources(&dir);
        let recorded = dir.join("inputs-ms");
        let outside = tmp.path().join("outside-inputs-ms");
        std::fs::rename(&recorded, &outside).unwrap();
        symlink(&outside, &recorded).unwrap();
        assert_refused(&dir, &scorer, &Limits::DEFAULT, "symlink");
        assert_refused(&dir, &scorer, &Limits::DEFAULT, "inputs-ms");

        std::fs::remove_file(&recorded).unwrap();
        std::fs::copy(&outside, &recorded).unwrap();
        assert_accepted(&dir, &scorer, &Limits::DEFAULT);
    }

    #[test]
    fn oversized_raw_log_is_refused_at_the_boundary() {
        let (_tmp, dir, scorer) = pilot();
        write_labelled_sources(&dir);
        let size = file_len(&dir.join("source-1").join("raw-candidates.jsonl"));
        let small = Limits {
            raw_log_bytes: size - 1,
            ..Limits::DEFAULT
        };
        assert_refused(&dir, &scorer, &small, &(size - 1).to_string());
        assert_refused(&dir, &scorer, &small, "source-1");
        let exact = Limits {
            raw_log_bytes: size,
            ..Limits::DEFAULT
        };
        assert_accepted(&dir, &scorer, &exact);
    }

    #[test]
    fn oversized_corpus_is_refused_at_the_boundary() {
        let (_tmp, dir, scorer) = pilot();
        write_labelled_sources(&dir);
        let size = file_len(&dir.join("source-1").join("corpus").join("records.bin"));
        let small = Limits {
            corpus_bytes: size - 1,
            ..Limits::DEFAULT
        };
        assert_refused(&dir, &scorer, &small, &(size - 1).to_string());
        let exact = Limits {
            corpus_bytes: size,
            ..Limits::DEFAULT
        };
        assert_accepted(&dir, &scorer, &exact);
        let manifest: serde_json::Value =
            serde_json::from_slice(&read(&dir.join("manifest.json")).unwrap()).unwrap();
        assert_eq!(manifest["maxCorpusBytes"].as_u64(), Some(size));
    }

    #[test]
    fn oversized_parent_is_refused() {
        let (_tmp, dir, scorer) = pilot();
        write_labelled_sources(&dir);
        let size = file_len(&dir.join("source-1").join("parent.json"));
        let limit = Limits {
            parent_bytes: size - 1,
            ..Limits::DEFAULT
        };
        assert_refused(&dir, &scorer, &limit, &(size - 1).to_string());
        assert_accepted(&dir, &scorer, &Limits::DEFAULT);
    }

    #[test]
    fn oversized_acquisition_ms_is_refused() {
        let (_tmp, dir, scorer) = pilot();
        write_labelled_sources(&dir);
        let long = format!("5{}", " ".repeat(64));
        std::fs::write(dir.join("source-1").join("acquisition-ms"), &long).unwrap();
        assert_eq!(long.len(), 65);
        assert_refused(&dir, &scorer, &Limits::DEFAULT, "64-byte");
        write_acquisition(&dir, "source-1");
        assert_accepted(&dir, &scorer, &Limits::DEFAULT);
    }

    #[test]
    fn oversized_scorer_is_refused() {
        let (_tmp, dir, scorer) = pilot();
        write_labelled_sources(&dir);
        let size = file_len(&scorer);
        let limit = Limits {
            scorer_bytes: size - 1,
            ..Limits::DEFAULT
        };
        assert_refused(&dir, &scorer, &limit, &(size - 1).to_string());
        let exact = Limits {
            scorer_bytes: size,
            ..Limits::DEFAULT
        };
        assert_accepted(&dir, &scorer, &exact);
    }

    #[cfg(unix)]
    #[test]
    fn symlinked_raw_log_is_refused() {
        use std::os::unix::fs::symlink;
        let (tmp, dir, scorer) = pilot();
        write_labelled_sources(&dir);
        let raw = dir.join("source-1").join("raw-candidates.jsonl");
        let outside = tmp.path().join("outside-raw.jsonl");
        std::fs::rename(&raw, &outside).unwrap();
        symlink(&outside, &raw).unwrap();
        assert_refused(&dir, &scorer, &Limits::DEFAULT, "symlink");

        std::fs::remove_file(&raw).unwrap();
        std::fs::copy(&outside, &raw).unwrap();
        assert_accepted(&dir, &scorer, &Limits::DEFAULT);
    }

    #[cfg(unix)]
    #[test]
    fn dangling_history_symlink_is_refused_and_its_target_not_created() {
        use std::os::unix::fs::symlink;
        let (tmp, dir, scorer) = pilot();
        write_labelled_sources(&dir);
        let target = tmp.path().join("outside-history.jsonl");
        symlink(&target, dir.join("source-1").join("history.jsonl")).unwrap();
        let err = try_manifest(&dir, &scorer, &Limits::DEFAULT).expect_err("must be refused");
        assert!(err.contains("already exists"), "{err}");
        assert!(err.contains("source-1"), "{err}");
        assert!(!target.exists(), "the outside target must not be created");
        assert!(std::fs::symlink_metadata(&target).is_err());
        assert!(
            std::fs::symlink_metadata(dir.join("source-2").join("history.jsonl")).is_err(),
            "source-2 history must not be written"
        );
        assert!(!dir.join("manifest.json").exists());

        std::fs::remove_file(dir.join("source-1").join("history.jsonl")).unwrap();
        assert_accepted(&dir, &scorer, &Limits::DEFAULT);
        assert!(!target.exists(), "the outside target must still not exist");
    }

    #[cfg(unix)]
    #[test]
    fn symlinked_task_directory_is_refused() {
        use std::os::unix::fs::symlink;
        let (tmp, dir, scorer) = pilot();
        write_labelled_sources(&dir);
        let task = dir.join("validation-2");
        let outside = tmp.path().join("outside-task");
        std::fs::rename(&task, &outside).unwrap();
        symlink(&outside, &task).unwrap();
        let err = try_manifest(&dir, &scorer, &Limits::DEFAULT).expect_err("must be refused");
        assert!(err.contains("symlink"), "{err}");
        assert!(err.contains("validation-2"), "{err}");
        assert_refused(&dir, &scorer, &Limits::DEFAULT, "symlink");

        std::fs::remove_file(&task).unwrap();
        std::fs::rename(&outside, &task).unwrap();
        assert_accepted(&dir, &scorer, &Limits::DEFAULT);
    }

    #[cfg(unix)]
    #[test]
    fn dangling_manifest_symlink_is_refused_and_its_target_not_created() {
        use std::os::unix::fs::symlink;
        let (tmp, dir, scorer) = pilot();
        write_labelled_sources(&dir);
        let target = tmp.path().join("outside-manifest.json");
        symlink(&target, dir.join("manifest.json")).unwrap();
        let err = try_manifest(&dir, &scorer, &Limits::DEFAULT).expect_err("must be refused");
        assert!(err.contains("already exists"), "{err}");
        assert!(!target.exists(), "the outside target must not be created");
        assert!(std::fs::symlink_metadata(&target).is_err());
        for id in ["source-1", "source-2"] {
            assert!(
                std::fs::symlink_metadata(dir.join(id).join("history.jsonl")).is_err(),
                "{id} history must not be written"
            );
        }

        std::fs::remove_file(dir.join("manifest.json")).unwrap();
        assert_accepted(&dir, &scorer, &Limits::DEFAULT);
        assert!(!target.exists(), "the outside target must still not exist");
    }

    #[cfg(unix)]
    #[test]
    fn symlinked_corpus_bin_is_refused() {
        use std::os::unix::fs::symlink;
        let (tmp, dir, scorer) = pilot();
        write_labelled_sources(&dir);
        let bin = dir.join("calibration-1").join("corpus").join("records.bin");
        let outside = tmp.path().join("outside.bin");
        std::fs::rename(&bin, &outside).unwrap();
        symlink(&outside, &bin).unwrap();
        assert_refused(&dir, &scorer, &Limits::DEFAULT, "symlink");

        std::fs::remove_file(&bin).unwrap();
        std::fs::rename(&outside, &bin).unwrap();
        assert_accepted(&dir, &scorer, &Limits::DEFAULT);
    }

    #[cfg(unix)]
    #[test]
    fn scorer_may_be_a_symlink() {
        use std::os::unix::fs::symlink;
        let (tmp, dir, scorer) = pilot();
        write_labelled_sources(&dir);
        let link = tmp.path().join("scorer-link");
        symlink(&scorer, &link).unwrap();
        assert_accepted(&dir, &link, &Limits::DEFAULT);
        let manifest: serde_json::Value =
            serde_json::from_slice(&read(&dir.join("manifest.json")).unwrap()).unwrap();

        // The digest is of the target's bytes, which `pilot` wrote to `scorer`.
        assert_eq!(
            manifest["scorerSha256"].as_str().unwrap(),
            sha256_hex(b"scorer bytes"),
            "scorerSha256 must hash the symlink's target"
        );

        // The manifest names the path passed in (the link), not the resolved target.
        let link_abs = std::path::absolute(&link).unwrap().display().to_string();
        let target_abs = std::path::absolute(&scorer).unwrap().display().to_string();
        assert_ne!(link_abs, target_abs, "the link and its target must differ");
        assert_eq!(
            manifest["scorer"].as_str().unwrap(),
            link_abs,
            "scorer must be the absolute link path, not the resolved target"
        );
        assert_ne!(
            manifest["scorer"].as_str().unwrap(),
            target_abs,
            "scorer must not be the resolved target"
        );
    }

    #[test]
    fn committed_pilot_inputs_match_the_generator() {
        let committed =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../docs/research/ranking-pilot-v1");
        let manifest: serde_json::Value =
            serde_json::from_slice(&read(&committed.join("manifest.json")).unwrap()).unwrap();
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join("pilot");
        write_inputs(&dir).unwrap();

        let tasks = manifest["tasks"].as_array().unwrap();
        assert_eq!(
            tasks.len(),
            TASKS.len(),
            "one committed entry per generated task"
        );
        for task in tasks {
            let id = task["id"].as_str().unwrap();
            let task_dir = dir.join(id);
            assert_eq!(
                task["parentSha256"].as_str().unwrap(),
                sha256_hex(&read(&task_dir.join("parent.json")).unwrap()),
                "parent digest of {id}"
            );
            // The committed corpus is the run's own bytes: its digest is the manifest's.
            let committed_corpus = committed.join(id).join("corpus");
            assert_eq!(
                task["corpusSha256"].as_str().unwrap(),
                corpus_sha256(&committed_corpus, Limits::DEFAULT.corpus_bytes).unwrap(),
                "committed corpus digest of {id}"
            );
            // A regenerated corpus is compared by value, not by digest: the
            // targets go through the host libm `tanhf`, whose last bits differ
            // between libm builds and CPU architectures.
            let committed_values = corpus_values(&committed_corpus.join("records.bin"));
            let regenerated = corpus_values(&task_dir.join("corpus").join("records.bin"));
            assert_eq!(
                committed_values.len(),
                regenerated.len(),
                "record count of {id}"
            );
            for (index, (kept, fresh)) in committed_values.iter().zip(&regenerated).enumerate() {
                let is_target = index % (INPUTS + 1) == INPUTS;
                let limit = if is_target { TARGET_TOLERANCE } else { 0.0 };
                assert!(
                    (kept - fresh).abs() <= limit,
                    "{id} value {index}: committed {kept}, regenerated {fresh}"
                );
            }
        }
    }

    /// The committed #261 synthetic pilot archive.
    fn committed_archive() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../docs/research/ranking-pilot-v1")
    }

    /// The decompressed bytes of a committed `.gz` file.
    fn gunzip(path: &Path) -> Vec<u8> {
        let output = std::process::Command::new("gzip")
            .arg("-dc")
            .arg(path)
            .stdin(std::process::Stdio::null())
            .output()
            .unwrap();
        assert!(output.status.success(), "gzip -dc {}", path.display());
        output.stdout
    }

    /// A fresh pilot carrying the committed raw logs, `acquisition-ms` and `inputs-ms`,
    /// with the first source's raw log replaced by `source_1_raw` when given.
    fn committed_raw_pilot(source_1_raw: Option<&[u8]>) -> (tempfile::TempDir, PathBuf, PathBuf) {
        let committed = committed_archive();
        let (tmp, dir, scorer) = pilot();
        for id in ["source-1", "source-2"] {
            let raw = gunzip(&committed.join(id).join("raw-candidates.jsonl.gz"));
            let raw = match source_1_raw {
                Some(replacement) if id == "source-1" => replacement.to_vec(),
                _ => raw,
            };
            std::fs::write(dir.join(id).join("raw-candidates.jsonl"), raw).unwrap();
            std::fs::copy(
                committed.join(id).join("acquisition-ms"),
                dir.join(id).join("acquisition-ms"),
            )
            .unwrap();
        }
        std::fs::copy(committed.join("inputs-ms"), dir.join("inputs-ms")).unwrap();
        (tmp, dir, scorer)
    }

    fn recorded_value(path: &Path) -> u64 {
        String::from_utf8(read(path).unwrap())
            .unwrap()
            .trim()
            .parse()
            .unwrap()
    }

    #[test]
    fn committed_raw_logs_recurate_to_the_committed_histories() {
        let committed = committed_archive();
        let archived: serde_json::Value =
            serde_json::from_slice(&read(&committed.join("manifest.json")).unwrap()).unwrap();
        let revision = archived["codeRevision"].as_str().unwrap();
        let cutoff = archived["historicalCutoffUnix"].as_u64().unwrap();

        let (_tmp, dir, scorer) = committed_raw_pilot(None);
        write_manifest(&dir, &scorer, revision, cutoff, &Limits::DEFAULT).unwrap();
        let regenerated: serde_json::Value =
            serde_json::from_slice(&read(&dir.join("manifest.json")).unwrap()).unwrap();

        let archived_tasks = archived["tasks"].as_array().unwrap();
        let regenerated_tasks = regenerated["tasks"].as_array().unwrap();
        assert_eq!(archived_tasks.len(), regenerated_tasks.len(), "task count");
        for (kept, fresh) in archived_tasks.iter().zip(regenerated_tasks) {
            let id = kept["id"].as_str().unwrap();
            assert_eq!(fresh["id"], kept["id"], "task order");
            assert_eq!(
                fresh["parentSha256"], kept["parentSha256"],
                "parent digest of {id}"
            );
            if kept["split"] != "source" {
                assert!(fresh["history"].is_null(), "{id} carries no history");
                continue;
            }
            assert_eq!(
                read(&dir.join(id).join("history.jsonl")).unwrap(),
                read(&committed.join(id).join("history.jsonl")).unwrap(),
                "regenerated history of {id} is byte-identical to the committed one"
            );
            assert_eq!(
                fresh["history"]["sha256"], kept["history"]["sha256"],
                "history digest of {id}"
            );
            let recorded = recorded_value(&committed.join(id).join("acquisition-ms"));
            let charged = fresh["history"]["acquisitionMs"].as_u64().unwrap();
            assert!(
                charged >= recorded,
                "{id} acquisitionMs {charged} must include the recorded {recorded} ms"
            );
        }

        // Negative control: the committed raw log's first row is a screen with no
        // full-corpus delta, so a log truncated to it leaves nothing fully scored
        // and the manifest step refuses it, writing no history and no manifest.
        let raw = gunzip(&committed.join("source-1/raw-candidates.jsonl.gz"));
        let first_row_end = raw.iter().position(|&byte| byte == b'\n').unwrap() + 1;
        let (_tmp, dir, scorer) = committed_raw_pilot(Some(&raw[..first_row_end]));
        let err = write_manifest(&dir, &scorer, revision, cutoff, &Limits::DEFAULT)
            .expect_err("a raw log truncated to its first row must be refused");
        assert!(
            err.starts_with("source-1: no fully scored row"),
            "refusal should name source-1 and the cause: {err}"
        );
        assert!(!dir.join("manifest.json").exists());
        for id in ["source-1", "source-2"] {
            assert!(!dir.join(id).join("history.jsonl").exists(), "{id} history");
        }
    }

    /// Inputs per record in a synthetic corpus; the target follows them.
    const INPUTS: usize = 4;

    /// Largest difference allowed between a committed target and one
    /// regenerated on this host. Targets are sums of a few `tanhf` terms, each
    /// off by at most a couple of ulps between libm builds, so this is far
    /// above any libm difference and far below the spread between tasks.
    const TARGET_TOLERANCE: f32 = 1e-5;

    /// Every little-endian `f32` in a `records.bin`, inputs and targets alike.
    fn corpus_values(bin: &Path) -> Vec<f32> {
        let bytes = read(bin).unwrap();
        bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|word| f32::from_le_bytes(*word))
            .collect()
    }
}
