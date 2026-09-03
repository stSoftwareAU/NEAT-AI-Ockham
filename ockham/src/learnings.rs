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
//! `<root>/screens/<host>.jsonl`, deliberately outside the verdict directory:
//! they record *coverage* ("this uuid has been looked at") and feed
//! unchecked-first selection. A screen record is never a prune verdict, so
//! [`LearningsStore::load`], [`known_wins`] and [`known_failures`] never see
//! one, and a corrupt screen log can never break verdict loading.
//!
//! That screen path carries **no corpus identity** (Issue #76). GRQ regenerates
//! the training corpus before every run, so a path keyed by the identity
//! pointed at a directory nothing had ever written and every run started from
//! zero coverage — re-screening the same neurons forever. Whether a uuid has
//! been looked at does not depend on what it was looked at against, so the
//! identity is recorded on the record ([`Screened::corpus_identity`]) instead,
//! preserving the information the path was carrying. A verdict is the opposite:
//! a full-corpus result genuinely is a claim about one corpus, so
//! `corpus-<identity>/` stays keyed exactly as it was.
//!
//! The pre-#76 `<root>/screens-<identity>/<host>.jsonl` directories are still
//! **read** — never written — so no fleet history is lost. Every `<root>` here
//! is whichever learnings root the caller passed, so an island's own root has
//! exactly the same shape.

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
///
/// Version 2 added [`Screened::corpus_identity`] and moved the records to the
/// stable `screens/` directory (Issue #76). Version 1 records — the fleet
/// history still sitting under `screens-<identity>/` — carry no identity and
/// are read unchanged; every other version is skipped.
pub const SCREENS_FORMAT_VERSION: u32 = 2;

/// The only older screen version this reads: the pre-#76 fleet history.
const LEGACY_SCREENS_FORMAT_VERSION: u32 = 1;

/// Version of a **visit-only** record: the sweep looked, nothing was scored (#93).
///
/// Deliberately a version an older binary does not accept. The fleet runs mixed
/// versions against one shared `screens/` directory, and a pre-#93 reader has no
/// notion of a visit that scored nothing: it would count these as ordinary
/// screens and publish a coverage percentage far above what it had actually
/// screened — the overstatement [`crate::coverage::Coverage::blocked`] exists to
/// prevent. Loading *skips* a record of an unknown version rather than
/// failing the load, so an old host simply carries on with the figures it can
/// justify until it is upgraded.
pub const SCREENS_VISIT_FORMAT_VERSION: u32 = 3;

/// [`Screened::kind`] of a visit no candidate could be proposed for (#93).
///
/// The razor cannot ablate a neuron that feeds an aggregate squash (`IF`,
/// `MEAN`, `MINIMUM`, …) or that carries a typed synapse, and on a
/// forest-heavy creature that is most of the hidden neurons. The visit is real
/// coverage; nothing was scored, so it never claims a screen.
///
/// Not necessarily *permanent*: an ablation can also fail on a non-finite
/// measured mean, and the same neuron may well propose a candidate on a later
/// pass. The record says what happened on that visit, nothing more — which is
/// why the reason is logged and the rendered line claims no cut was proposed
/// rather than that none ever could be.
pub const SCREEN_KIND_SKIPPED: &str = "skipped";

/// [`Screened::kind`] of a visit a standing full-corpus verdict suppressed.
///
/// Distinct from [`SCREEN_KIND_SKIPPED`]: the fleet has already **fully
/// scored** this uuid — that is why it is being skipped — so it is checked in
/// the strongest sense, with a cut that was proposed and judged.
pub const SCREEN_KIND_KNOWN_FAILURE: &str = "known-failure";

/// Directory holding screen-coverage host files, under the learnings root.
///
/// Stable across corpus identities on purpose — see the module docs.
const SCREENS_DIR: &str = "screens";

/// Prefix of the pre-#76 corpus-keyed screen directories.
///
/// Read for their fleet history, never written: dropping them would re-create,
/// once, exactly the coverage reset this change removes.
const LEGACY_SCREENS_PREFIX: &[u8] = b"screens-";

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

/// One hidden neuron the fleet has already looked at.
///
/// A coverage fact — "this uuid has been looked at" — used for
/// `checked X of Y hidden` reporting and unchecked-first selection. It is
/// **never** a prune verdict: only [`Learning`] carries those.
///
/// [`Self::kind`] says *what happened on the visit*: `identity` or `ablation`
/// for a candidate the scorer actually screened, and [`SCREEN_KIND_SKIPPED`] or
/// [`SCREEN_KIND_KNOWN_FAILURE`] for a visit that produced no candidate to
/// score (Issue #93). A visit that could not propose is still coverage — the
/// sweep has been there and there was nothing to try — and filing it is what
/// stops those neurons sitting in `unchecked` forever while the numerator only
/// ever falls. Such a record carries [`ScreenOutcomeKind::Loser`] so an older
/// host, which knows nothing of these kinds, reads it as ordinary coverage
/// rather than failing to deserialise the whole fleet's screen history.
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
    /// Corpus identity this screen was measured against (Issue #76).
    ///
    /// Carried on the record rather than in the path, so a regenerated corpus
    /// cannot reset coverage while anything wanting corpus-exact screening can
    /// still filter on it. `None` on version-1 records, where the identity
    /// lived in the directory name.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub corpus_identity: Option<String>,
}

impl Screened {
    /// True when the razor could propose nothing for this visit (Issue #93).
    ///
    /// Coverage counts the uuid as checked either way; this is what separates
    /// "looked at, nothing to try" from "the scorer screened a candidate".
    pub fn is_skipped(&self) -> bool {
        self.kind == SCREEN_KIND_SKIPPED
    }
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

    /// Directory holding the fleet's screen-coverage host files.
    ///
    /// A sibling of [`Self::corpus_dir`], never inside it: screen records are
    /// coverage facts, so a corrupt or oversized screen log must not be able
    /// to break verdict loading. Not keyed by corpus identity (Issue #76) —
    /// the identity rides on the record instead.
    pub fn screens_dir(&self) -> PathBuf {
        self.root.join(SCREENS_DIR)
    }

    /// Pre-#76 corpus-keyed screen directories, sorted, read-only.
    ///
    /// An unreadable root is an error rather than an empty list: silently
    /// reading no history is how the fleet loses months of screening.
    fn legacy_screens_dirs(&self) -> Result<Vec<PathBuf>, String> {
        if !self.root.is_dir() {
            return Ok(Vec::new());
        }
        let entries =
            std::fs::read_dir(&self.root).map_err(|e| format!("{}: {e}", self.root.display()))?;
        let mut dirs = Vec::new();
        for entry in entries {
            let entry = entry.map_err(|e| format!("{}: {e}", self.root.display()))?;
            let path = entry.path();
            // Matched on bytes, not `to_str()`: a name that is not valid UTF-8
            // is still a directory the fleet may have written history into,
            // and dropping it here would be silent.
            let is_legacy = path
                .file_name()
                .is_some_and(|n| n.as_encoded_bytes().starts_with(LEGACY_SCREENS_PREFIX));
            if is_legacy && path.is_dir() {
                dirs.push(path);
            }
        }
        dirs.sort();
        Ok(dirs)
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

    /// Load every screen record the fleet has filed, whatever corpus it used.
    ///
    /// The union of [`Self::screens_dir`] and every pre-#76
    /// `screens-<identity>/` directory, so the first run after the move starts
    /// from what the fleet already knows rather than from zero. Coverage and
    /// selection only — these are not prune verdicts.
    ///
    /// A fault in the live directory is an error, as it always was. A fault in
    /// a legacy directory is **warned and skipped** instead: nothing rewrites
    /// those files, so one truncated line in fleet history would otherwise
    /// empty the whole union on every host of every run — reinstating exactly
    /// the coverage reset this change removes (Issue #76). Loud, but contained
    /// to the directory that is broken.
    pub fn load_screens(&self) -> Result<Vec<Screened>, String> {
        let keep = |s: &Screened| {
            matches!(
                s.version,
                LEGACY_SCREENS_FORMAT_VERSION
                    | SCREENS_FORMAT_VERSION
                    | SCREENS_VISIT_FORMAT_VERSION
            )
        };
        let mut out = load_jsonl(&self.screens_dir(), keep)?;
        for legacy in self.legacy_screens_dirs()? {
            match load_jsonl(&legacy, keep) {
                Ok(records) => out.extend(records),
                Err(e) => crate::log::warn(&format!(
                    "legacy screen coverage unreadable ({e}); skipping {}",
                    legacy.display()
                )),
            }
        }
        Ok(out)
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

/// One visit to file: the uuid the sweep looked at, and what came of it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScreenTry<'a> {
    /// Hidden neuron UUID that was visited.
    pub uuid: &'a str,
    /// `identity`, `ablation`, [`SCREEN_KIND_SKIPPED`] or
    /// [`SCREEN_KIND_KNOWN_FAILURE`].
    pub kind: &'a str,
    /// Which side of the screen it landed on; a visit with no candidate to
    /// score is filed as [`ScreenOutcomeKind::Loser`] — see [`Screened`].
    pub outcome: ScreenOutcomeKind,
    /// [`SCREENS_FORMAT_VERSION`] for a scored candidate,
    /// [`SCREENS_VISIT_FORMAT_VERSION`] for a visit that scored nothing.
    pub version: u32,
}

impl<'a> ScreenTry<'a> {
    /// A candidate the scorer screened.
    pub fn scored(uuid: &'a str, kind: CandidateKind, outcome: ScreenOutcomeKind) -> Self {
        Self {
            uuid,
            kind: kind_label(kind),
            outcome,
            version: SCREENS_FORMAT_VERSION,
        }
    }

    /// A visit that produced no candidate to score (Issue #93).
    ///
    /// Filed at [`SCREENS_VISIT_FORMAT_VERSION`], so a host still running a
    /// pre-#93 binary skips it rather than mistaking it for a screen.
    pub fn visited(uuid: &'a str, kind: &'a str) -> Self {
        Self {
            uuid,
            kind,
            outcome: ScreenOutcomeKind::Loser,
            version: SCREENS_VISIT_FORMAT_VERSION,
        }
    }
}

/// File screen tries onto `known` and the store. Returns how many were written.
///
/// Coverage and selection only — nothing filed here can accept or reject a
/// prune. A store write failure warns and skips that record rather than
/// failing the run: a cache fault must never stop pruning.
pub fn file_screens(
    store: Option<&LearningsStore>,
    tries: &[ScreenTry<'_>],
    known: &mut Vec<Screened>,
) -> usize {
    let mut n = 0;
    let host = store.map(|s| s.host.clone()).unwrap_or_else(default_host);
    let unix_secs = now_secs();
    let corpus_identity = store.map(|s| s.corpus_identity.clone());
    for t in tries {
        let screened = Screened {
            version: t.version,
            uuid: t.uuid.to_string(),
            kind: t.kind.to_string(),
            outcome: t.outcome,
            unix_secs,
            host: host.clone(),
            corpus_identity: corpus_identity.clone(),
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
            corpus_identity: Some("corp".into()),
        }
    }

    /// A pre-#76 screen record, written straight into `screens-<identity>/`.
    fn write_legacy_screen(root: &Path, identity: &str, host: &str, line: &str) {
        let dir = root.join(format!("screens-{identity}"));
        std::fs::create_dir_all(&dir).unwrap();
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(dir.join(format!("{host}.jsonl")))
            .unwrap();
        writeln!(file, "{line}").unwrap();
    }

    /// The exact shape the fleet wrote before this change: version 1, no
    /// `corpusIdentity`, and the identity in the directory name instead.
    fn legacy_screen_line(uuid: &str, secs: u64) -> String {
        format!(
            r#"{{"version":1,"uuid":"{uuid}","kind":"ablation","outcome":"loser","unixSecs":{secs},"host":"legacy"}}"#
        )
    }

    #[test]
    fn screens_round_trip_through_the_store() {
        let dir = tempfile::tempdir().unwrap();
        let store = LearningsStore::new(dir.path(), "corp".into(), "host-a".into());
        let s = screen("h_a", ScreenOutcomeKind::Winner, 9);
        store.append_screen(&s).unwrap();
        assert_eq!(store.load_screens().unwrap(), vec![s]);
    }

    /// Issue #93: a visit that scored nothing is written at a version a pre-#93
    /// binary does not accept, so an old host in the mixed-version fleet skips
    /// it instead of counting it as a screen and publishing a coverage figure
    /// far above what it has actually screened. This host reads both.
    #[test]
    fn a_visit_only_record_is_written_at_a_version_older_hosts_skip() {
        let dir = tempfile::tempdir().unwrap();
        let store = LearningsStore::new(dir.path(), "corp".into(), "host-a".into());
        let mut seen = Vec::new();
        let written = file_screens(
            Some(&store),
            &[
                ScreenTry::scored("h_a", CandidateKind::Ablation, ScreenOutcomeKind::Loser),
                ScreenTry::visited("h_b", SCREEN_KIND_SKIPPED),
                ScreenTry::visited("h_c", SCREEN_KIND_KNOWN_FAILURE),
            ],
            &mut seen,
        );
        assert_eq!(written, 3);

        let pre_93_versions = [LEGACY_SCREENS_FORMAT_VERSION, SCREENS_FORMAT_VERSION];
        assert!(
            !pre_93_versions.contains(&SCREENS_VISIT_FORMAT_VERSION),
            "the visit version must be one a pre-#93 reader rejects"
        );

        let loaded = store.load_screens().unwrap();
        assert_eq!(loaded.len(), 3, "this host reads every one of them");
        let by_uuid: HashMap<&str, &Screened> =
            loaded.iter().map(|s| (s.uuid.as_str(), s)).collect();
        assert_eq!(by_uuid["h_a"].version, SCREENS_FORMAT_VERSION);
        assert!(!by_uuid["h_a"].is_skipped());
        for uuid in ["h_b", "h_c"] {
            assert_eq!(
                by_uuid[uuid].version, SCREENS_VISIT_FORMAT_VERSION,
                "{uuid} scored nothing, so an old host must skip its record"
            );
            assert_eq!(
                by_uuid[uuid].outcome,
                ScreenOutcomeKind::Loser,
                "{uuid} claims no win it never earned"
            );
        }
        assert!(by_uuid["h_b"].is_skipped(), "no cut could be proposed");
        assert!(
            !by_uuid["h_c"].is_skipped(),
            "a known failure was proposed and judged, so it is not blocked"
        );
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
                ScreenTry::scored("h_a", CandidateKind::Ablation, ScreenOutcomeKind::Winner),
                ScreenTry::scored("h_b", CandidateKind::Identity, ScreenOutcomeKind::Loser),
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

    /// Issue #76: coverage is a fact about the uuid, not about the corpus.
    #[test]
    fn screens_survive_a_corpus_identity_change() {
        let dir = tempfile::tempdir().unwrap();
        let before = LearningsStore::new(dir.path(), "corpus-a".into(), "host-a".into());
        let s = screen("h_a", ScreenOutcomeKind::Winner, 9);
        before.append_screen(&s).unwrap();

        let after = LearningsStore::new(dir.path(), "corpus-b".into(), "host-a".into());
        let loaded = after.load_screens().unwrap();
        assert_eq!(
            loaded.iter().map(|s| s.uuid.as_str()).collect::<Vec<_>>(),
            vec!["h_a"],
            "a regenerated corpus must not reset screen coverage"
        );
    }

    /// Issue #76: the fleet's pre-move history is read, not abandoned.
    #[test]
    fn legacy_corpus_keyed_screens_still_count_as_checked() {
        let dir = tempfile::tempdir().unwrap();
        write_legacy_screen(
            dir.path(),
            "old-corpus",
            "GRQ-23",
            &legacy_screen_line("h_a", 5),
        );
        let store = LearningsStore::new(dir.path(), "new-corpus".into(), "host-a".into());
        store
            .append_screen(&screen("h_b", ScreenOutcomeKind::Winner, 9))
            .unwrap();

        let loaded = store.load_screens().unwrap();
        assert_eq!(loaded.len(), 2, "{loaded:?}");
        let c = two_hidden();
        assert_eq!(
            screened_uuids(&loaded, &c),
            HashSet::from(["h_a".to_string(), "h_b".to_string()]),
            "legacy records are coverage the fleet already paid for"
        );
        let legacy = loaded.iter().find(|s| s.uuid == "h_a").unwrap();
        assert_eq!(legacy.version, 1);
        assert_eq!(legacy.corpus_identity, None, "the identity was in the path");
    }

    /// The near miss this fix must not make: a verdict is a claim about one
    /// corpus, so widening the screens path must not widen the verdict path.
    #[test]
    fn a_verdict_from_another_corpus_identity_is_still_not_loaded() {
        let dir = tempfile::tempdir().unwrap();
        let before = LearningsStore::new(dir.path(), "corpus-a".into(), "host-a".into());
        before.append(&rec("h_a", Outcome::Rejected, 9)).unwrap();

        let after = LearningsStore::new(dir.path(), "corpus-b".into(), "host-a".into());
        assert!(
            after.load().unwrap().is_empty(),
            "a foreign-corpus verdict must never suppress a candidate"
        );
        assert_eq!(
            before.load().unwrap().len(),
            1,
            "its own corpus still sees it"
        );
    }

    /// A corrupt screen line fails screens loudly and leaves verdicts alone.
    #[test]
    fn a_corrupt_screen_line_cannot_break_verdict_loading() {
        let dir = tempfile::tempdir().unwrap();
        let store = LearningsStore::new(dir.path(), "corp".into(), "host-a".into());
        store.append(&rec("h_a", Outcome::Accepted, 9)).unwrap();
        store
            .append_screen(&screen("h_b", ScreenOutcomeKind::Winner, 9))
            .unwrap();
        let mut file = OpenOptions::new()
            .append(true)
            .open(store.screens_host_path())
            .unwrap();
        writeln!(file, "{{not json").unwrap();

        assert!(store.load_screens().is_err(), "corruption must be loud");
        assert_eq!(store.load().unwrap().len(), 1, "verdicts are unaffected");
    }

    /// An unknown-version line in the *legacy* location is skipped too, so the
    /// migration cannot turn a forward-compatible record into a hard failure.
    #[test]
    fn unknown_version_legacy_screen_lines_are_skipped() {
        let dir = tempfile::tempdir().unwrap();
        write_legacy_screen(
            dir.path(),
            "old-corpus",
            "GRQ-23",
            &legacy_screen_line("h_a", 5),
        );
        write_legacy_screen(
            dir.path(),
            "old-corpus",
            "GRQ-24",
            &legacy_screen_line("h_b", 6).replace(
                r#""version":1"#,
                &format!(r#""version":{}"#, SCREENS_VISIT_FORMAT_VERSION + 1),
            ),
        );
        let store = LearningsStore::new(dir.path(), "new-corpus".into(), "host-a".into());
        let loaded = store.load_screens().unwrap();
        assert_eq!(loaded.len(), 1, "{loaded:?}");
        assert_eq!(loaded[0].uuid, "h_a");
    }

    /// A version older than the fleet history is unknown, not "legacy enough".
    #[test]
    fn a_below_legacy_screen_version_is_skipped() {
        let dir = tempfile::tempdir().unwrap();
        write_legacy_screen(
            dir.path(),
            "old-corpus",
            "GRQ-23",
            &legacy_screen_line("h_a", 5).replace(r#""version":1"#, r#""version":0"#),
        );
        let store = LearningsStore::new(dir.path(), "new-corpus".into(), "host-a".into());
        assert!(store.load_screens().unwrap().is_empty());
    }

    /// One corrupt line in a directory nothing writes to any more must not
    /// empty the fleet's whole coverage — that is the plateau itself.
    #[test]
    fn a_corrupt_legacy_screen_file_does_not_empty_the_union() {
        let dir = tempfile::tempdir().unwrap();
        write_legacy_screen(dir.path(), "corrupt-corpus", "GRQ-23", "{not json");
        write_legacy_screen(
            dir.path(),
            "good-corpus",
            "GRQ-24",
            &legacy_screen_line("h_a", 5),
        );
        let store = LearningsStore::new(dir.path(), "new-corpus".into(), "host-a".into());
        store
            .append_screen(&screen("h_b", ScreenOutcomeKind::Winner, 9))
            .unwrap();

        let loaded = store.load_screens().unwrap();
        let c = two_hidden();
        assert_eq!(
            screened_uuids(&loaded, &c),
            HashSet::from(["h_a".to_string(), "h_b".to_string()]),
            "a broken legacy directory must cost only its own records"
        );
    }

    #[test]
    fn a_screen_record_carries_the_corpus_it_was_measured_against() {
        let dir = tempfile::tempdir().unwrap();
        let store = LearningsStore::new(dir.path(), "corpus-a".into(), "host-a".into());
        let mut seen = Vec::new();
        file_screens(
            Some(&store),
            &[ScreenTry::scored(
                "h_a",
                CandidateKind::Ablation,
                ScreenOutcomeKind::Winner,
            )],
            &mut seen,
        );
        assert_eq!(seen[0].corpus_identity.as_deref(), Some("corpus-a"));
        assert_eq!(seen[0].version, SCREENS_FORMAT_VERSION);
        assert_eq!(
            store.load_screens().unwrap()[0].corpus_identity.as_deref(),
            Some("corpus-a"),
            "the path stopped carrying it, so the record must"
        );
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
        future["version"] = serde_json::json!(SCREENS_VISIT_FORMAT_VERSION + 1);
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
            &[ScreenTry::scored(
                "h_a",
                CandidateKind::Ablation,
                ScreenOutcomeKind::Winner,
            )],
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
