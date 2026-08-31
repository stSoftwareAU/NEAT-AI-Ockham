//! Fleet-shared cache of full-corpus prune verdicts.
//!
//! Forests caches portable patches (feature indices). Ockham caches **hidden
//! neuron UUIDs**: a prune is only useful while that uuid is still in the
//! fittest creature. Known wins that remain are tried first (quick check-in).
//! Known full-corpus failures are skipped until [`DEFAULT_RETRY_AFTER_SECS`].
//!
//! Only full-corpus verdicts are stored as verdicts. Sample-screen opinions
//! never are — they are wrong often enough to bury good prunes.
//!
//! A cohort can apply only one of its candidates, so most full-corpus winners
//! lose to a better one and are filed [`Outcome::Rejected`]. [`Learning::full_delta`]
//! records what the scorer actually measured for each individual, and
//! [`confirmed_wins`] reads it back: *confirmed but not applied* is a candidate
//! to retry, not a failure to suppress (Issue #52).
//!
//! Layout: `<root>/corpus-<identity>/<host>.jsonl` — one append-only file per
//! host, so a git-shared directory never conflicts. Nothing here talks to git.
//!
//! Screen **tries** ([`Screened`]) live in the sibling
//! `<root>/screens-<identity>/<host>.jsonl`, deliberately outside the verdict
//! directory: they record *coverage* ("this uuid has been looked at") and feed
//! unchecked-first selection. A screen record is never a prune verdict, so
//! [`LearningsStore::load`], [`known_wins`] and [`known_failures`] never see
//! one, and a corrupt screen log can never break verdict loading.

use std::collections::{HashMap, HashSet};
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use neat_core::CreatureExport;
use serde::de::DeserializeOwned;
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
    /// Full-corpus `score - incumbent` for this uuid scored **alone**, when it
    /// was scored as an individual (Issue #52).
    ///
    /// `None` on records written before this field and on uuids that only ever
    /// appeared inside a bundle — their individual contribution was never
    /// measured.
    ///
    /// Only one candidate out of a cohort can be applied, because every one of
    /// them was scored from the same incumbent snapshot. A positive delta on a
    /// record whose [`Outcome`] is [`Outcome::Rejected`] therefore means
    /// *confirmed but not applied*, not *no good*: [`confirmed_wins`] replays
    /// it and [`known_failures`] stops suppressing it.
    ///
    /// This is deliberately an additive **field** rather than a new [`Outcome`]
    /// variant. The fleet runs mixed versions against one shared cache and
    /// loading treats an undeserialisable line as a hard error for the
    /// whole load; serde ignores unknown fields but rejects unknown enum
    /// variants, so a new variant would break learnings on every host still on
    /// the old binary the moment one upgraded host wrote a record.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub full_delta: Option<f64>,
}

/// Current screen-record format version.
pub const SCREENS_FORMAT_VERSION: u32 = 1;

/// Which side of a sampled screen a uuid landed on.
///
/// Informational only: it drives **coverage and selection**, never a prune
/// verdict. Sample screens are wrong often enough that only [`Outcome`] — a
/// full-corpus result — may accept or reject a prune.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum ScreenOutcomeKind {
    /// Beat the sampled incumbent on the screen.
    Winner,
    /// Did not beat the sampled incumbent on the screen.
    Loser,
}

/// One hidden neuron the fleet has screened at least once.
///
/// A coverage fact — "this uuid has been looked at" — used for
/// `checked X of Y hidden` reporting and unchecked-first selection. It is
/// **never** a prune verdict: only [`Learning`] carries those.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Screened {
    /// Format version.
    pub version: u32,
    /// Hidden neuron UUID that was screened.
    pub uuid: String,
    /// `identity` or `ablation`.
    pub kind: String,
    /// Which side of the screen it landed on — informational only.
    pub outcome: ScreenOutcomeKind,
    /// Unix seconds when filed.
    pub unix_secs: u64,
    /// Host that filed it (`GRQ-23`).
    pub host: String,
}

/// How many known wins to replay before the random sweep.
///
/// [`Self::max`] of `0` means every still-present known win (Forests caps
/// replay; Ockham does not — Forests leaps faster than new cuts are found).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReplayConfig {
    /// Maximum known-win UUIDs to replay (`0` = all still present).
    pub max: usize,
    /// Rejected records newer than this are skipped.
    pub retry_after_secs: u64,
}

impl Default for ReplayConfig {
    fn default() -> Self {
        Self {
            max: 0,
            retry_after_secs: DEFAULT_RETRY_AFTER_SECS,
        }
    }
}

/// Effective replay cap (`0` on the CLI means unlimited).
pub fn replay_cap(max: usize) -> usize {
    if max == 0 { usize::MAX } else { max }
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

    /// Directory holding this corpus's screen-coverage host files.
    ///
    /// A sibling of [`Self::corpus_dir`], never inside it: screen records are
    /// coverage facts, so a corrupt or oversized screen log must not be able
    /// to break verdict loading.
    pub fn screens_dir(&self) -> PathBuf {
        self.root.join(format!("screens-{}", self.corpus_identity))
    }

    /// This host's append-only screen-coverage file.
    pub fn screens_host_path(&self) -> PathBuf {
        self.screens_dir().join(format!("{}.jsonl", self.host))
    }

    /// Load every verdict for this corpus from every host file.
    ///
    /// Screen records live elsewhere, so this never returns one.
    pub fn load(&self) -> Result<Vec<Learning>, String> {
        load_jsonl(&self.corpus_dir(), |l: &Learning| {
            l.version == LEARNINGS_FORMAT_VERSION
        })
    }

    /// Append one verdict to this host's file.
    pub fn append(&self, learning: &Learning) -> Result<(), String> {
        append_jsonl(&self.corpus_dir(), &self.host_path(), learning)
    }

    /// Load every screen record for this corpus from every host file.
    ///
    /// Coverage and selection only — these are not prune verdicts.
    pub fn load_screens(&self) -> Result<Vec<Screened>, String> {
        load_jsonl(&self.screens_dir(), |s: &Screened| {
            s.version == SCREENS_FORMAT_VERSION
        })
    }

    /// Append one screen record to this host's screen file.
    ///
    /// Coverage and selection only — this is not a prune verdict.
    pub fn append_screen(&self, screened: &Screened) -> Result<(), String> {
        append_jsonl(&self.screens_dir(), &self.screens_host_path(), screened)
    }
}

/// Read every `*.jsonl` record under `dir` that `keep` accepts.
///
/// Records of an unknown format version are skipped; malformed JSON is an
/// error, so corruption is loud rather than silently partial.
fn load_jsonl<T: DeserializeOwned>(
    dir: &Path,
    keep: impl Fn(&T) -> bool,
) -> Result<Vec<T>, String> {
    if !dir.is_dir() {
        return Ok(Vec::new());
    }
    let mut out = Vec::new();
    let entries = std::fs::read_dir(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
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
            match serde_json::from_str::<T>(&line) {
                Ok(rec) if keep(&rec) => out.push(rec),
                Ok(_) => {}
                Err(e) => return Err(format!("{}: {e}", path.display())),
            }
        }
    }
    Ok(out)
}

/// Append one JSON line to `path`, creating `dir` if needed.
fn append_jsonl<T: Serialize>(dir: &Path, path: &Path, record: &T) -> Result<(), String> {
    std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .map_err(|e| format!("{}: {e}", path.display()))?;
    let line = serde_json::to_string(record).map_err(|e| e.to_string())?;
    writeln!(file, "{line}").map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(())
}

/// Host name for `--learnings-host` when unset.
///
/// Order: `$HOSTNAME`, `$HOST`, `hostname(1)`, then `unknown`. The result is
/// the unqualified label (`GRQ-23`, not `GRQ-23.local`).
pub fn default_host() -> String {
    env_nonempty("HOSTNAME")
        .or_else(|| env_nonempty("HOST"))
        .or_else(hostname_cmd)
        .map(|s| unqualified_host(&s))
        .unwrap_or_else(|| "unknown".into())
}

/// First DNS label of `raw`, or `"unknown"` if empty.
pub fn unqualified_host(raw: &str) -> String {
    raw.trim()
        .split('.')
        .next()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .unwrap_or("unknown")
        .to_string()
}

fn env_nonempty(key: &str) -> Option<String> {
    std::env::var(key)
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}

fn hostname_cmd() -> Option<String> {
    let out = std::process::Command::new("hostname").output().ok()?;
    if !out.status.success() {
        return None;
    }
    String::from_utf8(out.stdout)
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
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

/// Known-win UUIDs still in the incumbent, most recent first.
///
/// [`ReplayConfig::max`] of `0` means every still-present known win.
pub fn known_wins(known: &[Learning], creature: &CreatureExport, cfg: ReplayConfig) -> Vec<String> {
    let present: HashSet<&str> = creature.neurons.iter().map(|n| n.uuid.as_str()).collect();
    let mut wins: Vec<&Learning> = latest_by_uuid(known, &present)
        .into_values()
        .filter(|l| l.outcome == Outcome::Accepted)
        .collect();
    wins.sort_by_key(|a| std::cmp::Reverse(a.unix_secs));
    wins.into_iter()
        .take(replay_cap(cfg.max))
        .map(|l| l.uuid.clone())
        .collect()
}

/// UUIDs whose latest full-corpus verdict is a fresh rejection.
///
/// A record carrying a [`Learning::full_delta`] above `min_improvement` is
/// **not** a failure whatever its [`Outcome`] (Issue #52): it lost its cohort
/// to a better candidate, which says nothing about the cut itself.
pub fn known_failures(
    known: &[Learning],
    creature: &CreatureExport,
    cfg: ReplayConfig,
    now: u64,
    min_improvement: f64,
) -> HashSet<String> {
    let present: HashSet<&str> = creature.neurons.iter().map(|n| n.uuid.as_str()).collect();
    latest_by_uuid(known, &present)
        .into_values()
        .filter(|l| {
            l.outcome == Outcome::Rejected
                && !confirmed_positive(l, min_improvement)
                && now.saturating_sub(l.unix_secs) < cfg.retry_after_secs
        })
        .map(|l| l.uuid.clone())
        .collect()
}

/// Whether this record measured a full-corpus win for the uuid on its own.
fn confirmed_positive(l: &Learning, min_improvement: f64) -> bool {
    l.full_delta.is_some_and(|d| d > min_improvement)
}

/// One still-present uuid a previous full-corpus score spoke well of.
///
/// Either it was applied ([`Outcome::Accepted`]) or its own individual delta
/// beat `min_improvement` while another candidate won the cohort.
#[derive(Debug, Clone, PartialEq)]
pub struct ConfirmedWin {
    /// Hidden neuron UUID.
    pub uuid: String,
    /// Measured individual full-corpus delta, when one was recorded.
    pub full_delta: Option<f64>,
    /// Whether the latest record applied the cut.
    pub accepted: bool,
    /// Unix seconds of the latest record.
    pub unix_secs: u64,
}

/// Still-present uuids whose latest verdict confirmed the cut, best delta first.
///
/// The cross-run half of "remember every winner" (Issue #52): a cut that beat
/// `min_improvement` on the full corpus but lost its cohort is exactly the
/// candidate #45 asks Ockham to keep trying, and it was previously filed —
/// and suppressed — as a failure.
pub fn confirmed_wins(
    known: &[Learning],
    creature: &CreatureExport,
    min_improvement: f64,
) -> Vec<String> {
    ranked_confirmed(known, creature, min_improvement)
        .into_iter()
        .map(|c| c.uuid)
        .collect()
}

/// [`confirmed_wins`] with the ranking keys the replay path needs.
///
/// Accepted records come first — the fleet has already paid to apply them —
/// then confirmed-only ones. Within each group the best measured delta leads,
/// an unmeasured delta comes last, and recency then uuid break the remaining
/// ties so every host builds the same order.
pub fn ranked_confirmed(
    known: &[Learning],
    creature: &CreatureExport,
    min_improvement: f64,
) -> Vec<ConfirmedWin> {
    let present: HashSet<&str> = creature.neurons.iter().map(|n| n.uuid.as_str()).collect();
    let mut out: Vec<ConfirmedWin> = latest_by_uuid(known, &present)
        .into_values()
        .filter(|l| l.outcome == Outcome::Accepted || confirmed_positive(l, min_improvement))
        .map(|l| ConfirmedWin {
            uuid: l.uuid.clone(),
            full_delta: l.full_delta,
            accepted: l.outcome == Outcome::Accepted,
            unix_secs: l.unix_secs,
        })
        .collect();
    out.sort_by(|a, b| {
        b.accepted
            .cmp(&a.accepted)
            .then_with(|| {
                let (x, y) = (
                    a.full_delta.unwrap_or(f64::NEG_INFINITY),
                    b.full_delta.unwrap_or(f64::NEG_INFINITY),
                );
                y.partial_cmp(&x).unwrap_or(std::cmp::Ordering::Equal)
            })
            .then_with(|| b.unix_secs.cmp(&a.unix_secs))
            .then_with(|| a.uuid.cmp(&b.uuid))
    });
    out
}

/// One full-corpus verdict to file.
pub struct Verdict<'a> {
    /// Hidden UUID.
    pub uuid: &'a str,
    /// Sweep kind.
    pub kind: CandidateKind,
    /// Accepted or rejected.
    pub outcome: Outcome,
    /// Individual full-corpus delta when this uuid was scored alone.
    pub full_delta: Option<f64>,
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
            full_delta: v.full_delta,
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

/// Latest screen record per uuid, newest wins (ties keep the later record).
///
/// Coverage and selection only — a screen record is never a prune verdict.
pub fn latest_screen_by_uuid(screens: &[Screened]) -> HashMap<&str, &Screened> {
    let mut latest: HashMap<&str, &Screened> = HashMap::new();
    for s in screens {
        latest
            .entry(s.uuid.as_str())
            .and_modify(|prev| {
                if s.unix_secs >= prev.unix_secs {
                    *prev = s;
                }
            })
            .or_insert(s);
    }
    latest
}

/// UUIDs screened at least once and still present on `creature`.
///
/// This is the coverage set: "checked X of Y hidden". It says nothing about
/// whether a prune was any good.
pub fn screened_uuids(screens: &[Screened], creature: &CreatureExport) -> HashSet<String> {
    let present: HashSet<&str> = creature.neurons.iter().map(|n| n.uuid.as_str()).collect();
    screens
        .iter()
        .map(|s| s.uuid.as_str())
        .filter(|uuid| present.contains(uuid))
        .map(str::to_string)
        .collect()
}

/// Still-present screened UUIDs, least-recently screened first.
///
/// What recycling reads once coverage completes — selection only, never a
/// prune verdict. Equal screen times are broken by uuid so the order is
/// deterministic across hosts.
pub fn oldest_screened_first(screens: &[Screened], creature: &CreatureExport) -> Vec<String> {
    let present: HashSet<&str> = creature.neurons.iter().map(|n| n.uuid.as_str()).collect();
    let latest = latest_screen_by_uuid(screens);
    let mut still_present: Vec<&Screened> = latest
        .into_values()
        .filter(|s| present.contains(s.uuid.as_str()))
        .collect();
    still_present.sort_by(|a, b| {
        a.unix_secs
            .cmp(&b.unix_secs)
            .then_with(|| a.uuid.cmp(&b.uuid))
    });
    still_present.into_iter().map(|s| s.uuid.clone()).collect()
}

/// File screen tries onto `known` and the store. Returns how many were written.
///
/// Coverage and selection only — nothing filed here can accept or reject a
/// prune. A store write failure warns and skips that record rather than
/// failing the run: a cache fault must never stop pruning.
pub fn file_screens(
    store: Option<&LearningsStore>,
    uuids: &[(&str, CandidateKind, ScreenOutcomeKind)],
    known: &mut Vec<Screened>,
) -> usize {
    let mut n = 0;
    let host = store.map(|s| s.host.clone()).unwrap_or_else(default_host);
    let unix_secs = now_secs();
    for (uuid, kind, outcome) in uuids {
        let screened = Screened {
            version: SCREENS_FORMAT_VERSION,
            uuid: (*uuid).to_string(),
            kind: kind_label(*kind).to_string(),
            outcome: *outcome,
            unix_secs,
            host: host.clone(),
        };
        if let Some(store) = store
            && let Err(e) = store.append_screen(&screened)
        {
            crate::log::warn(&format!("screen coverage not written: {e}"));
            continue;
        }
        known.push(screened);
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
            full_delta: None,
        }
    }

    /// A rejected record that nonetheless measured `delta` on its own.
    fn confirmed(uuid: &str, delta: f64, secs: u64) -> Learning {
        Learning {
            full_delta: Some(delta),
            ..rec(uuid, Outcome::Rejected, secs)
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
        let skip = known_failures(&known, &c, ReplayConfig::default(), now, 1e-6);
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
        let skip = known_failures(&known, &c, ReplayConfig::default(), 3, 1e-6);
        assert!(!skip.contains("h_a"));
    }

    #[test]
    fn replay_cap_zero_means_all_still_present() {
        let c = two_hidden();
        let known = vec![
            rec("h_a", Outcome::Accepted, 10),
            rec("h_b", Outcome::Accepted, 20),
        ];
        let all = known_wins(&known, &c, ReplayConfig::default());
        assert_eq!(all, vec!["h_b".to_string(), "h_a".to_string()]);
        let capped = known_wins(
            &known,
            &c,
            ReplayConfig {
                max: 1,
                retry_after_secs: DEFAULT_RETRY_AFTER_SECS,
            },
        );
        assert_eq!(capped, vec!["h_b".to_string()]);
    }

    #[test]
    fn unqualified_host_strips_the_domain() {
        assert_eq!(unqualified_host("GRQ-23.local"), "GRQ-23");
        assert_eq!(unqualified_host("  GRQ-23  "), "GRQ-23");
        assert_eq!(unqualified_host(""), "unknown");
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

    fn screen(uuid: &str, outcome: ScreenOutcomeKind, secs: u64) -> Screened {
        Screened {
            version: SCREENS_FORMAT_VERSION,
            uuid: uuid.into(),
            kind: "ablation".into(),
            outcome,
            unix_secs: secs,
            host: "t".into(),
        }
    }

    #[test]
    fn screens_round_trip_through_the_store() {
        let dir = tempfile::tempdir().unwrap();
        let store = LearningsStore::new(dir.path(), "corp".into(), "host-a".into());
        let s = screen("h_a", ScreenOutcomeKind::Winner, 9);
        store.append_screen(&s).unwrap();
        assert_eq!(store.load_screens().unwrap(), vec![s]);
    }

    #[test]
    fn screens_live_outside_the_verdict_directory() {
        let dir = tempfile::tempdir().unwrap();
        let store = LearningsStore::new(dir.path(), "corp".into(), "host-a".into());
        store
            .append_screen(&screen("h_a", ScreenOutcomeKind::Loser, 9))
            .unwrap();

        assert_ne!(store.screens_dir(), store.corpus_dir());
        assert!(!store.screens_dir().starts_with(store.corpus_dir()));
        assert!(store.load().unwrap().is_empty());
    }

    #[test]
    fn a_screened_uuid_is_neither_a_known_win_nor_a_known_failure() {
        let dir = tempfile::tempdir().unwrap();
        let store = LearningsStore::new(dir.path(), "corp".into(), "host-a".into());
        let mut seen = Vec::new();
        let written = file_screens(
            Some(&store),
            &[
                ("h_a", CandidateKind::Ablation, ScreenOutcomeKind::Winner),
                ("h_b", CandidateKind::Identity, ScreenOutcomeKind::Loser),
            ],
            &mut seen,
        );
        assert_eq!(written, 2);
        assert_eq!(seen.len(), 2);

        let c = two_hidden();
        let known = store.load().unwrap();
        assert!(known_wins(&known, &c, ReplayConfig::default()).is_empty());
        assert!(known_failures(&known, &c, ReplayConfig::default(), 1_000_000, 1e-6).is_empty());
    }

    #[test]
    fn unknown_version_screen_lines_are_skipped() {
        let dir = tempfile::tempdir().unwrap();
        let store = LearningsStore::new(dir.path(), "corp".into(), "host-a".into());
        store
            .append_screen(&screen("h_a", ScreenOutcomeKind::Winner, 9))
            .unwrap();
        let mut future =
            serde_json::to_value(screen("h_b", ScreenOutcomeKind::Winner, 10)).unwrap();
        future["version"] = serde_json::json!(SCREENS_FORMAT_VERSION + 1);
        let mut file = OpenOptions::new()
            .append(true)
            .open(store.screens_host_path())
            .unwrap();
        writeln!(file, "{future}").unwrap();

        let loaded = store.load_screens().unwrap();
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].uuid, "h_a");
    }

    #[test]
    fn file_screens_warns_and_reduces_the_count_on_a_write_failure() {
        let dir = tempfile::tempdir().unwrap();
        let blocker = dir.path().join("not-a-dir");
        std::fs::write(&blocker, "").unwrap();
        let store = LearningsStore::new(&blocker, "corp".into(), "host-a".into());

        let mut seen = Vec::new();
        let written = file_screens(
            Some(&store),
            &[("h_a", CandidateKind::Ablation, ScreenOutcomeKind::Winner)],
            &mut seen,
        );
        assert_eq!(written, 0);
        assert!(seen.is_empty());
    }

    #[test]
    fn latest_screen_per_uuid_wins() {
        let known = vec![
            screen("h_a", ScreenOutcomeKind::Loser, 1),
            screen("h_a", ScreenOutcomeKind::Winner, 2),
            screen("h_b", ScreenOutcomeKind::Loser, 5),
        ];
        let latest = latest_screen_by_uuid(&known);
        assert_eq!(latest.len(), 2);
        assert_eq!(latest["h_a"].unix_secs, 2);
        assert_eq!(latest["h_a"].outcome, ScreenOutcomeKind::Winner);
        assert_eq!(latest["h_b"].unix_secs, 5);
    }

    #[test]
    fn screened_uuids_are_limited_to_still_present_neurons() {
        let c = two_hidden();
        let known = vec![
            screen("h_a", ScreenOutcomeKind::Winner, 1),
            screen("gone", ScreenOutcomeKind::Loser, 2),
        ];
        let seen = screened_uuids(&known, &c);
        assert_eq!(seen, HashSet::from(["h_a".to_string()]));
    }

    #[test]
    fn oldest_screened_uuid_comes_first() {
        let c = two_hidden();
        let known = vec![
            screen("h_b", ScreenOutcomeKind::Winner, 5),
            screen("h_a", ScreenOutcomeKind::Loser, 20),
            screen("h_a", ScreenOutcomeKind::Winner, 30),
            screen("gone", ScreenOutcomeKind::Winner, 1),
        ];
        assert_eq!(
            oldest_screened_first(&known, &c),
            vec!["h_b".to_string(), "h_a".to_string()]
        );
    }

    /// A line written before Issue #52 must still load, with no delta.
    #[test]
    fn a_record_written_before_full_delta_still_deserialises() {
        let line = r#"{"version":1,"uuid":"h_a","kind":"ablation","outcome":"rejected","unixSecs":9,"host":"t"}"#;
        let l: Learning = serde_json::from_str(line).unwrap();
        assert_eq!(l.full_delta, None);
        assert_eq!(l.uuid, "h_a");
        // And the field is skipped on the way out, so an old binary reading a
        // new binary's record sees the shape it already knows.
        assert_eq!(serde_json::to_string(&l).unwrap(), line);
    }

    #[test]
    fn a_file_mixing_both_record_shapes_loads_without_error() {
        let dir = tempfile::tempdir().unwrap();
        let store = LearningsStore::new(dir.path(), "corp".into(), "host-a".into());
        store.append(&rec("h_a", Outcome::Accepted, 9)).unwrap();
        store.append(&confirmed("h_b", 2e-6, 10)).unwrap();

        let loaded = store.load().unwrap();
        assert_eq!(loaded.len(), 2, "{loaded:?}");
        assert_eq!(loaded[0].full_delta, None);
        assert_eq!(loaded[1].full_delta, Some(2e-6));
        let text = std::fs::read_to_string(store.host_path()).unwrap();
        assert!(
            !text.lines().next().unwrap().contains("fullDelta"),
            "{text}"
        );
        assert!(
            text.lines().nth(1).unwrap().contains("\"fullDelta\":"),
            "{text}"
        );
    }

    #[test]
    fn a_confirmed_positive_is_a_win_even_though_it_was_filed_rejected() {
        let c = two_hidden();
        let known = vec![confirmed("h_a", 2e-6, 10), confirmed("h_b", 1e-9, 11)];
        assert_eq!(confirmed_wins(&known, &c, 1e-6), vec!["h_a".to_string()]);
        assert!(
            known_wins(&known, &c, ReplayConfig::default()).is_empty(),
            "nothing was applied, so nothing is an accepted win"
        );
    }

    #[test]
    fn confirmed_wins_are_still_present_and_best_delta_first() {
        let c = two_hidden();
        let known = vec![
            confirmed("h_a", 2e-6, 10),
            confirmed("h_b", 9e-6, 11),
            confirmed("gone", 9e-3, 12),
        ];
        assert_eq!(
            confirmed_wins(&known, &c, 1e-6),
            vec!["h_b".to_string(), "h_a".to_string()]
        );
    }

    #[test]
    fn an_accepted_record_outranks_a_confirmed_one_with_a_bigger_delta() {
        let c = two_hidden();
        let known = vec![
            confirmed("h_b", 9e-3, 11),
            Learning {
                full_delta: Some(1e-5),
                ..rec("h_a", Outcome::Accepted, 10)
            },
        ];
        let ranked = ranked_confirmed(&known, &c, 1e-6);
        assert_eq!(ranked[0].uuid, "h_a", "applied cuts replay first");
        assert!(ranked[0].accepted);
        assert_eq!(ranked[1].uuid, "h_b");
        assert!(!ranked[1].accepted);
    }

    /// Recency and measured delta deliberately disagree: replay must follow the
    /// delta, so its largest-first plans drop the weakest members (Issue #57).
    #[test]
    fn ranking_follows_the_measured_delta_not_the_filing_order() {
        let c = two_hidden();
        let known = vec![confirmed("h_a", 9e-3, 10), confirmed("h_b", 2e-6, 5_000)];
        let ranked = ranked_confirmed(&known, &c, 1e-6);
        assert_eq!(
            ranked.iter().map(|r| r.uuid.as_str()).collect::<Vec<_>>(),
            vec!["h_a", "h_b"],
            "the newest record is the weakest and must rank last"
        );
    }

    #[test]
    fn a_confirmed_positive_is_no_longer_suppressed_as_a_failure() {
        let c = two_hidden();
        let now = 1_000_000;
        let known = vec![
            confirmed("h_a", 2e-6, now - 60),
            confirmed("h_b", -1e-4, now - 60),
        ];
        let skip = known_failures(&known, &c, ReplayConfig::default(), now, 1e-6);
        assert!(!skip.contains("h_a"), "a measured win is not a failure");
        assert!(skip.contains("h_b"), "a measured loss still is");

        // A fresh rejection carrying no measurement stays suppressed too.
        let unmeasured = vec![rec("h_a", Outcome::Rejected, now - 60)];
        assert!(
            known_failures(&unmeasured, &c, ReplayConfig::default(), now, 1e-6).contains("h_a")
        );
    }

    #[test]
    fn loading_screens_from_an_absent_store_is_empty() {
        let dir = tempfile::tempdir().unwrap();
        let store = LearningsStore::new(dir.path(), "corp".into(), "host-a".into());
        assert!(store.load_screens().unwrap().is_empty());
    }
}
