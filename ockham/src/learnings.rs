//! Fleet-shared cache of full-corpus prune verdicts.
//!
//! Forests caches portable patches (feature indices). Ockham caches **hidden
//! neuron UUIDs**: a prune is only useful while that uuid is still in the
//! fittest creature. Known wins that remain are tried first (quick check-in).
//! Known full-corpus failures are skipped until [`DEFAULT_RETRY_AFTER_SECS`].
//!
//! Only full-corpus verdicts are stored. Sample-screen opinions are not —
//! they are wrong often enough to bury good prunes.
//!
//! Layout: `<root>/corpus-<identity>/<host>.jsonl` — one append-only file per
//! host, so a git-shared directory never conflicts. Nothing here talks to git.

use std::collections::{HashMap, HashSet};
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use neat_core::CreatureExport;
use serde::{Deserialize, Serialize};

use crate::sweep::CandidateKind;

/// Current learnings format version.
pub const LEARNINGS_FORMAT_VERSION: u32 = 1;

/// Failures older than this may be tried again (7 days).
pub const DEFAULT_RETRY_AFTER_SECS: u64 = 7 * 24 * 3600;

/// How a full-corpus candidate ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum Outcome {
    /// Authoritative scorer accepted the prune.
    Accepted,
    /// Fully scored and not good enough.
    Rejected,
}

/// One prune the fleet has already judged.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Learning {
    /// Format version.
    pub version: u32,
    /// Hidden neuron UUID that was pruned.
    pub uuid: String,
    /// `identity` or `ablation`.
    pub kind: String,
    /// Full-corpus outcome.
    pub outcome: Outcome,
    /// Unix seconds when filed.
    pub unix_secs: u64,
    /// Host that filed it (`GRQ-23`).
    pub host: String,
}

/// How many known wins to replay at the front of a sweep.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReplayConfig {
    /// Maximum known-win UUIDs to prefer.
    pub max: usize,
    /// Rejected records newer than this are skipped.
    pub retry_after_secs: u64,
}

impl Default for ReplayConfig {
    fn default() -> Self {
        Self {
            max: 32,
            retry_after_secs: DEFAULT_RETRY_AFTER_SECS,
        }
    }
}

/// Append-only per-host store under a corpus identity.
#[derive(Debug, Clone)]
pub struct LearningsStore {
    root: PathBuf,
    corpus_identity: String,
    host: String,
}

impl LearningsStore {
    /// `root/corpus-<identity>/<host>.jsonl`.
    pub fn new(root: impl Into<PathBuf>, corpus_identity: String, host: String) -> Self {
        Self {
            root: root.into(),
            corpus_identity,
            host,
        }
    }

    /// Directory holding this corpus's host files.
    pub fn corpus_dir(&self) -> PathBuf {
        self.root.join(format!("corpus-{}", self.corpus_identity))
    }

    /// This host's append-only file.
    pub fn host_path(&self) -> PathBuf {
        self.corpus_dir().join(format!("{}.jsonl", self.host))
    }

    /// Load every record for this corpus from every host file.
    pub fn load(&self) -> Result<Vec<Learning>, String> {
        let dir = self.corpus_dir();
        if !dir.is_dir() {
            return Ok(Vec::new());
        }
        let mut out = Vec::new();
        let entries = std::fs::read_dir(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
        for entry in entries {
            let path = entry.map_err(|e| format!("{}: {e}", dir.display()))?.path();
            if path.extension().and_then(|e| e.to_str()) != Some("jsonl") {
                continue;
            }
            let file = File::open(&path).map_err(|e| format!("{}: {e}", path.display()))?;
            for line in BufReader::new(file).lines() {
                let line = line.map_err(|e| format!("{}: {e}", path.display()))?;
                if line.trim().is_empty() {
                    continue;
                }
                match serde_json::from_str::<Learning>(&line) {
                    Ok(l) if l.version == LEARNINGS_FORMAT_VERSION => out.push(l),
                    Ok(_) => {}
                    Err(e) => return Err(format!("{}: {e}", path.display())),
                }
            }
        }
        Ok(out)
    }

    /// Append one record to this host's file.
    pub fn append(&self, learning: &Learning) -> Result<(), String> {
        let dir = self.corpus_dir();
        std::fs::create_dir_all(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
        let path = self.host_path();
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .map_err(|e| format!("{}: {e}", path.display()))?;
        let line = serde_json::to_string(learning).map_err(|e| e.to_string())?;
        writeln!(file, "{line}").map_err(|e| format!("{}: {e}", path.display()))?;
        Ok(())
    }
}

/// Host name for `--learnings-host` when unset.
pub fn default_host() -> String {
    hostname_unqualified().unwrap_or_else(|| "unknown".into())
}

fn hostname_unqualified() -> Option<String> {
    let raw = std::env::var("HOST").ok().filter(|s| !s.is_empty())?;
    Some(raw.split('.').next().unwrap_or(&raw).to_string())
}

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Kind label stored in the cache.
pub fn kind_label(kind: CandidateKind) -> &'static str {
    match kind {
        CandidateKind::Identity => "identity",
        CandidateKind::Ablation => "ablation",
    }
}

/// Latest outcome per uuid still present on `creature`.
fn latest_by_uuid<'a>(
    known: &'a [Learning],
    present: &HashSet<&str>,
) -> HashMap<&'a str, &'a Learning> {
    let mut latest: HashMap<&str, &Learning> = HashMap::new();
    for l in known {
        if !present.contains(l.uuid.as_str()) {
            continue;
        }
        latest
            .entry(l.uuid.as_str())
            .and_modify(|prev| {
                if l.unix_secs >= prev.unix_secs {
                    *prev = l;
                }
            })
            .or_insert(l);
    }
    latest
}

/// Known-win UUIDs still in the incumbent, most recent first, capped.
pub fn known_wins(known: &[Learning], creature: &CreatureExport, cfg: ReplayConfig) -> Vec<String> {
    let present: HashSet<&str> = creature.neurons.iter().map(|n| n.uuid.as_str()).collect();
    let mut wins: Vec<&Learning> = latest_by_uuid(known, &present)
        .into_values()
        .filter(|l| l.outcome == Outcome::Accepted)
        .collect();
    wins.sort_by_key(|a| std::cmp::Reverse(a.unix_secs));
    wins.into_iter()
        .take(cfg.max)
        .map(|l| l.uuid.clone())
        .collect()
}

/// UUIDs whose latest full-corpus verdict is a fresh rejection.
pub fn known_failures(
    known: &[Learning],
    creature: &CreatureExport,
    cfg: ReplayConfig,
    now: u64,
) -> HashSet<String> {
    let present: HashSet<&str> = creature.neurons.iter().map(|n| n.uuid.as_str()).collect();
    latest_by_uuid(known, &present)
        .into_values()
        .filter(|l| {
            l.outcome == Outcome::Rejected && now.saturating_sub(l.unix_secs) < cfg.retry_after_secs
        })
        .map(|l| l.uuid.clone())
        .collect()
}

/// One full-corpus verdict to file.
pub struct Verdict<'a> {
    /// Hidden UUID.
    pub uuid: &'a str,
    /// Sweep kind.
    pub kind: CandidateKind,
    /// Accepted or rejected.
    pub outcome: Outcome,
}

/// File verdicts onto `known` and the store. Returns how many were written.
pub fn file_verdicts(
    store: Option<&LearningsStore>,
    verdicts: &[Verdict<'_>],
    known: &mut Vec<Learning>,
) -> usize {
    let mut n = 0;
    let host = store.map(|s| s.host.clone()).unwrap_or_else(default_host);
    let unix_secs = now_secs();
    for v in verdicts {
        let learning = Learning {
            version: LEARNINGS_FORMAT_VERSION,
            uuid: v.uuid.to_string(),
            kind: kind_label(v.kind).to_string(),
            outcome: v.outcome,
            unix_secs,
            host: host.clone(),
        };
        if let Some(store) = store
            && let Err(e) = store.append(&learning)
        {
            crate::log::warn(&format!("learnings not written: {e}"));
            continue;
        }
        known.push(learning);
        n += 1;
    }
    n
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixtures::{creature, neuron, synapse};

    fn hidden(uuid: &str) -> neat_core::NeuronExport {
        neuron("hidden", uuid, 0.0, Some("IDENTITY"))
    }

    fn two_hidden() -> CreatureExport {
        creature(
            1,
            1,
            vec![
                hidden("h_a"),
                hidden("h_b"),
                neuron("output", "output-0", 0.0, Some("IDENTITY")),
            ],
            vec![
                synapse("input-0", "h_a", 1.0),
                synapse("h_a", "h_b", 1.0),
                synapse("h_b", "output-0", 1.0),
            ],
        )
    }

    fn rec(uuid: &str, outcome: Outcome, secs: u64) -> Learning {
        Learning {
            version: LEARNINGS_FORMAT_VERSION,
            uuid: uuid.into(),
            kind: "ablation".into(),
            outcome,
            unix_secs: secs,
            host: "t".into(),
        }
    }

    #[test]
    fn wins_still_in_creature_are_preferred() {
        let c = two_hidden();
        let known = vec![
            rec("h_a", Outcome::Accepted, 10),
            rec("gone", Outcome::Accepted, 20),
            rec("h_b", Outcome::Rejected, 30),
        ];
        let wins = known_wins(&known, &c, ReplayConfig::default());
        assert_eq!(wins, vec!["h_a".to_string()]);
    }

    #[test]
    fn fresh_failures_are_skipped_stale_ones_are_not() {
        let c = two_hidden();
        let now = 1_000_000;
        let known = vec![
            rec("h_a", Outcome::Rejected, now - 60),
            rec("h_b", Outcome::Rejected, now - DEFAULT_RETRY_AFTER_SECS - 1),
        ];
        let skip = known_failures(&known, &c, ReplayConfig::default(), now);
        assert!(skip.contains("h_a"));
        assert!(!skip.contains("h_b"));
    }

    #[test]
    fn latest_verdict_wins() {
        let c = two_hidden();
        let known = vec![
            rec("h_a", Outcome::Rejected, 1),
            rec("h_a", Outcome::Accepted, 2),
        ];
        let wins = known_wins(&known, &c, ReplayConfig::default());
        assert_eq!(wins, vec!["h_a".to_string()]);
        let skip = known_failures(&known, &c, ReplayConfig::default(), 3);
        assert!(!skip.contains("h_a"));
    }

    #[test]
    fn store_round_trips_appended_records() {
        let dir = tempfile::tempdir().unwrap();
        let store = LearningsStore::new(dir.path(), "corp".into(), "host-a".into());
        let l = rec("h_a", Outcome::Accepted, 9);
        store.append(&l).unwrap();
        let loaded = store.load().unwrap();
        assert_eq!(loaded, vec![l]);
    }
}
