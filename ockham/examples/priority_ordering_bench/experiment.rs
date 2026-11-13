//! Controlled, opt-in experiment alongside the existing simulated benchmark.

mod audit;
mod evaluation;
mod manifest;

#[cfg(test)]
mod tests;

use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use clap::{Parser, Subcommand};
use neat_ai_ockham::incumbent::sha256_hex;
use neat_ai_ockham::scorer::{
    DirectoryScorer, ExternalScorer, ScoreResult, ScorerError, ScorerMode,
};
use serde::{Serialize, de::DeserializeOwned};

use manifest::{Freeze, Manifest};

#[derive(Parser)]
#[command(about = "Opt-in fully scored pruning-order experiment; never applies a cut")]
struct Cli {
    #[command(subcommand)]
    command: Phase,
}

#[derive(Subcommand)]
enum Phase {
    /// Validate identities and availability without opening validation corpora.
    Check {
        #[arg(long)]
        manifest: PathBuf,
        #[arg(long)]
        out: PathBuf,
    },
    /// Calibration-only feasibility pilot. Output directory must not exist.
    Pilot {
        #[arg(long)]
        manifest: PathBuf,
        #[arg(long)]
        out: PathBuf,
    },
    /// Freeze affordable visit count and the exact manifest and pilot digests.
    Freeze {
        #[arg(long)]
        manifest: PathBuf,
        #[arg(long)]
        pilot: PathBuf,
        #[arg(long)]
        out: PathBuf,
    },
    /// Generate and fully score all declared partitions using the frozen limits.
    Audit {
        #[arg(long)]
        manifest: PathBuf,
        #[arg(long)]
        freeze: PathBuf,
        #[arg(long)]
        out: PathBuf,
    },
    /// Paired offline comparison; validation scores are read only as labels.
    Evaluate {
        #[arg(long)]
        manifest: PathBuf,
        #[arg(long)]
        freeze: PathBuf,
        #[arg(long)]
        audit: PathBuf,
        #[arg(long)]
        out: PathBuf,
    },
}

fn read<T: DeserializeOwned>(path: &Path) -> Result<T, String> {
    let meta = std::fs::metadata(path).map_err(|e| format!("{}: {e}", path.display()))?;
    if meta.len() > 64 * 1024 * 1024 {
        return Err("experiment input exceeds 64 MiB".into());
    }
    serde_json::from_reader(std::fs::File::open(path).map_err(|e| e.to_string())?)
        .map_err(|e| format!("{}: {e}", path.display()))
}
fn write(path: &Path, value: &impl Serialize) -> Result<(), String> {
    let bytes = serde_json::to_vec_pretty(value).map_err(|e| e.to_string())?;
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|e| format!("{}: {e}", path.display()))?;
    file.write_all(&bytes)
        .and_then(|()| file.write_all(b"\n"))
        .map_err(|e| e.to_string())
}
fn hash_file(path: &Path) -> Result<String, String> {
    Ok(sha256_hex(
        &std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?,
    ))
}

struct BoundedScorer {
    scorer: ExternalScorer,
    bounding: PathBuf,
    seconds: u64,
}
impl DirectoryScorer for BoundedScorer {
    fn identity(&self) -> String {
        self.scorer.identity()
    }
    fn score_directory(
        &self,
        dir: &Path,
        corpus: &Path,
        mode: ScorerMode,
    ) -> Result<std::collections::BTreeMap<String, ScoreResult>, ScorerError> {
        self.scorer
            .score_directory_bounded(dir, corpus, mode, &self.bounding, self.seconds)
    }
}

fn runtime(m: &Manifest) -> Result<BoundedScorer, String> {
    // Freeze against a clean source revision. Cargo's normal dependency tracking
    // rebuilds this example when its source or the library changes.
    let head = Command::new("git")
        .args(["rev-parse", "HEAD"])
        .stdin(Stdio::null())
        .output()
        .map_err(|e| e.to_string())?;
    if !head.status.success() {
        return Err("cannot resolve code revision".into());
    }
    if String::from_utf8_lossy(&head.stdout).trim() != m.code_revision {
        return Err("code revision mismatch".into());
    }
    let dirty = Command::new("git")
        .args([
            "diff",
            "--quiet",
            "HEAD",
            "--",
            "ockham",
            "Cargo.toml",
            "Cargo.lock",
            "rust-toolchain.toml",
        ])
        .stdin(Stdio::null())
        .status()
        .map_err(|e| e.to_string())?;
    if !dirty.success() {
        return Err("experiment requires committed source and lockfile".into());
    }
    if hash_file(&m.scorer)? != m.scorer_sha256 {
        return Err("scorer digest mismatch".into());
    }
    let bounding = ["timeout", "gtimeout"]
        .into_iter()
        .find(|name| {
            Command::new(name)
                .arg("--version")
                .stdin(Stdio::null())
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .status()
                .is_ok_and(|s| s.success())
        })
        .ok_or("experiment requires timeout or gtimeout to bound scorer calls")?;
    let scorer = ExternalScorer {
        binary: m.scorer.clone(),
        extra_args: m.scorer_args.clone(),
    };
    Ok(BoundedScorer {
        scorer,
        bounding: bounding.into(),
        seconds: m.scorer_timeout_secs,
    })
}

fn freeze(m: &Manifest, pilot: &audit::Audit, pilot_sha256: String) -> Result<Freeze, String> {
    if pilot.format_version != 1
        || pilot.phase != "pilot"
        || pilot.manifest_sha256 != m.identity()?
        || pilot.freeze_sha256.is_some()
        || !pilot.failures.is_empty()
    {
        return Err("pilot incomplete or manifest mismatch".into());
    }
    let mut slowest_ms = 1;
    let mut pools = std::collections::BTreeSet::new();
    for pool in &pilot.pools {
        let task = m
            .tasks
            .iter()
            .find(|t| t.id == pool.task)
            .ok_or("unknown pilot task")?;
        if task.split != manifest::Split::Calibration
            || !m.seeds.contains(&pool.seed)
            || !pools.insert((&pool.task, pool.seed))
        {
            return Err("pilot touched a forbidden/duplicate task or seed".into());
        }
        if pool.selected.len() != pool.population.min(m.pilot_visits)
            || pool.rows.len() != pool.selected.len()
        {
            return Err("pilot visit count mismatch".into());
        }
        let generated = pool.rows.iter().filter(|r| r.kind.is_some()).count();
        if !pool.complete() || generated == 0 {
            return Err("pilot has no complete generated candidates".into());
        }
        // Safety factor two, counting preparation and failed proposals too.
        slowest_ms = slowest_ms.max(pool.elapsed_ms.div_ceil(generated as u64));
    }
    let expected = m
        .tasks
        .iter()
        .filter(|t| t.split == manifest::Split::Calibration)
        .count()
        * m.seeds.len();
    if pools.len() != expected {
        return Err("pilot calibration pools are incomplete".into());
    }
    let per_pool = (m.max_task_ms / (2 * slowest_ms)).min(512) as usize;
    let total_pools = (m.tasks.len() * m.seeds.len()) as u64;
    let per_run = (m.max_run_ms / (2 * slowest_ms * total_pools)).min(512) as usize;
    let audit_visits = m.audit_visits_ceiling.min(per_pool).min(per_run);
    if audit_visits < m.minimum_complete_pool {
        return Err(format!(
            "calibration pilot cannot afford minimum pool: {audit_visits} visits"
        ));
    }
    Ok(Freeze {
        format_version: 1,
        manifest_sha256: m.identity()?,
        pilot_sha256,
        audit_visits,
    })
}

fn check_freeze(m: &Manifest, frozen: &Freeze) -> Result<(), String> {
    if frozen.format_version != 1
        || frozen.manifest_sha256 != m.identity()?
        || (frozen.pilot_sha256.len() != 64
            || !frozen.pilot_sha256.bytes().all(|b| b.is_ascii_hexdigit()))
        || frozen.audit_visits < m.minimum_complete_pool
        || frozen.audit_visits > m.audit_visits_ceiling
    {
        return Err("freeze protocol/manifest/limits mismatch".into());
    }
    Ok(())
}

pub fn main() -> Result<(), String> {
    let cli = Cli::try_parse_from(
        std::iter::once("priority_ordering_bench".to_string()).chain(std::env::args().skip(2)),
    )
    .map_err(|e| e.to_string())?;
    match cli.command {
        Phase::Check { manifest, out } => {
            let m: Manifest = read(&manifest)?;
            m.validate()?;
            let mut failures = Vec::new();
            if let Err(error) = runtime(&m) {
                failures.push(error);
            }
            for t in &m.tasks {
                if !t.parent.is_file()
                    || !t.corpus.is_dir()
                    || t.history.as_ref().is_some_and(|h| !h.path.is_file())
                {
                    failures.push(format!("{}: parent/corpus unavailable", t.id));
                }
            }
            write(
                &out,
                &serde_json::json!({"formatVersion":1,"phase":"preflight","manifestSha256":m.identity()?,"status":if failures.is_empty(){"available"}else{"inconclusive"},"failures":failures,"representativeBenefitDemonstrated":false}),
            )?;
            if !failures.is_empty() {
                return Err("preflight unavailable; see result file".into());
            }
        }
        Phase::Pilot { manifest, out } => {
            let m: Manifest = read(&manifest)?;
            m.validate()?;
            let scorer = runtime(&m)?;
            std::fs::create_dir(&out).map_err(|e| e.to_string())?;
            let result = audit::collect(&m, true, m.pilot_visits, None, &scorer, &out);
            write(&out.join("pilot.json"), &result)?;
            if !result.failures.is_empty() {
                return Err("calibration pilot failed; see pilot.json".into());
            }
        }
        Phase::Freeze {
            manifest,
            pilot,
            out,
        } => {
            let m: Manifest = read(&manifest)?;
            m.validate()?;
            runtime(&m)?;
            let result = freeze(&m, &read(&pilot)?, hash_file(&pilot)?)?;
            write(&out, &result)?;
        }
        Phase::Audit {
            manifest,
            freeze,
            out,
        } => {
            let m: Manifest = read(&manifest)?;
            m.validate()?;
            let scorer = runtime(&m)?;
            let frozen: Freeze = read(&freeze)?;
            check_freeze(&m, &frozen)?;
            std::fs::create_dir(&out).map_err(|e| e.to_string())?;
            let result = audit::collect(
                &m,
                false,
                frozen.audit_visits,
                Some(hash_file(&freeze)?),
                &scorer,
                &out,
            );
            write(&out.join("audit.json"), &result)?;
            if !result.failures.is_empty() {
                return Err("audit incomplete; see audit.json".into());
            }
        }
        Phase::Evaluate {
            manifest,
            freeze,
            audit,
            out,
        } => {
            let m: Manifest = read(&manifest)?;
            m.validate()?;
            runtime(&m)?;
            let frozen: Freeze = read(&freeze)?;
            check_freeze(&m, &frozen)?;
            let collected: audit::Audit = read(&audit)?;
            if collected.freeze_sha256.as_deref() != Some(hash_file(&freeze)?.as_str()) {
                return Err("audit/freeze digest mismatch".into());
            }
            let result = evaluation::evaluate(&m, &collected, hash_file(&audit)?)?;
            write(&out, &result)?;
        }
    }
    Ok(())
}
