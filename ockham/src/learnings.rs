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
//!
//! What that identity on the record is *for* is Issue #100: coverage is only
//! authoritative for the corpus it was measured against. The training corpus is
//! extended every few days, and a screen taken against the old one says nothing
//! about how the neuron behaves under the new data. So the records are all
//! **kept and read**, and [`current_epoch_screens`] selects the ones filed under
//! the corpus in hand — the current *screening epoch*. A corpus change starts a
//! fresh epoch at `0 / hidden`; a repacked corpus with identical content hashes
//! the same and keeps its coverage; and a host that moves back to an identity it
//! screened before finds that epoch's records still there, because nothing was
//! ever cleared. Selecting rather than clearing is what keeps #76's fix intact
//! while the fleet sits on several live identities at once.
//!
//! Verdicts filed against **older** corpora are read too, as **evidence rather
//! than truth** (Issues #88, #101). [`LearningsStore::load_prior_corpora`] reads
//! every sibling `corpus-*` directory and returns each record stamped with the
//! epoch that established it ([`HistoricalLearning`]), so nothing is lost when a
//! corpus changes and every learning still names the corpus it came from.
//!
//! Two things read that history, and neither of them is a verdict:
//!
//! - [`prior_corpus_priority`] turns it into the still-present, still-unchecked
//!   uuids the fleet once removed under earlier training data. Those are checked
//!   first because they are the likeliest to be removable again;
//! - [`historical_replay`] turns it into the previous winners worth **replaying**
//!   against the corpus in hand (Issue #101) — a high-value hypothesis, scored
//!   from scratch by the current scorer before anything is accepted.
//!
//! What history may never do is decide. It never joins [`LearningsStore::load`],
//! so an old `Rejected` cannot suppress a candidate and an old `Accepted` cannot
//! accept a prune: every uuid it raises faces the sample screen and full-corpus
//! scoring exactly as it would have done last. Historical results are evidence;
//! the current scorer is truth.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use neat_core::CreatureExport;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

use crate::blocked::BlockedReason;
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
const LEGACY_SCREENS_PREFIX: &str = "screens-";

/// Prefix of the per-corpus verdict directories.
///
/// The remainder of the name is the corpus identity the verdicts inside were
/// measured against — the epoch a historical learning was established in (#101).
const CORPUS_DIR_PREFIX: &str = "corpus-";

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
/// `sweep X of Y hidden` reporting and unchecked-first selection. It is
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
    /// Carried on the record rather than in the path, so a repacked corpus
    /// cannot reset coverage while anything wanting corpus-exact screening can
    /// still filter on it — which is what the screening epoch of Issue #100
    /// does. `None` on version-1 records, where the identity lived in the
    /// directory name; [`LearningsStore::load_screens`] restores it from that
    /// name as it reads, so pre-#76 history sits in the epoch it was measured
    /// against rather than in none at all.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub corpus_identity: Option<String>,
    /// Why no candidate could be proposed, on a [`SCREEN_KIND_SKIPPED`] record.
    ///
    /// The per-neuron half of Issue #103: the breakdown every reporting surface
    /// counts is derived from these, so a blocked category can be inspected
    /// neuron by neuron and epoch by epoch rather than as one opaque total.
    ///
    /// `None` on every other kind — a scored candidate and a standing verdict
    /// are not blocked — and on a record filed before #103, which
    /// [`Self::blocked_category`] reads as
    /// [`BlockedReason::Unrecorded`] rather than guessing at. Carried as a
    /// **code string** so an unknown code from a newer host reads as `other`
    /// instead of failing the whole load (see [`crate::blocked`]).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub blocked_reason: Option<BlockedReason>,
}

impl Screened {
    /// True when the razor could propose nothing for this visit (Issue #93).
    ///
    /// Coverage counts the uuid as checked either way; this is what separates
    /// "looked at, nothing to try" from "the scorer screened a candidate".
    pub fn is_skipped(&self) -> bool {
        self.kind == SCREEN_KIND_SKIPPED
    }

    /// Why this visit was blocked, for a blocked record (Issue #103).
    ///
    /// A record filed before reasons existed reads as
    /// [`BlockedReason::Unrecorded`] rather than as a guess: the sum of the
    /// reasons must still equal the blocked total, so a missing reason is a
    /// category of its own instead of a dropped neuron.
    pub fn blocked_category(&self) -> Option<BlockedReason> {
        self.is_skipped()
            .then(|| self.blocked_reason.unwrap_or(BlockedReason::Unrecorded))
    }

    /// True when this record was measured against `corpus_identity` (#100).
    ///
    /// A record naming no corpus at all belongs to no epoch: it is history the
    /// fleet can still read, never current-epoch coverage. Pre-#76 records are
    /// not in that position — [`LearningsStore::load_screens`] stamps them with
    /// the identity their directory name carries.
    pub fn in_epoch(&self, corpus_identity: &str) -> bool {
        self.corpus_identity.as_deref() == Some(corpus_identity)
    }
}

/// The records of one screening epoch: those measured against this corpus (#100).
///
/// Consumes the loaded history and keeps the current epoch, so coverage and
/// unchecked-first selection see the corpus in hand and nothing else. Every
/// other record stays on disk — [`LearningsStore::load_screens`] still returns
/// it, tagged with the corpus it was measured against — because this invalidates
/// *coverage authority*, not history.
///
/// A neuron the previous epoch recorded as screened, `blocked`
/// ([`SCREEN_KIND_SKIPPED`]) or [`SCREEN_KIND_KNOWN_FAILURE`] is therefore
/// unchecked again under a new corpus, and eligible to be visited on its merits.
pub fn current_epoch_screens(screens: Vec<Screened>, corpus_identity: &str) -> Vec<Screened> {
    screens
        .into_iter()
        .filter(|s| s.in_epoch(corpus_identity))
        .collect()
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

    /// Identity of the corpus this store is reading and writing against.
    ///
    /// The screening epoch every record it appends is filed under (#100).
    pub fn corpus_identity(&self) -> &str {
        &self.corpus_identity
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

    /// Sibling directories of the root whose name starts with `prefix`, sorted.
    ///
    /// `skip` drops one path — the current corpus, which is never its own
    /// history. An unreadable root is an error rather than an empty list:
    /// silently reading no history is how the fleet loses months of screening.
    fn dirs_named(&self, prefix: &[u8], skip: Option<&Path>) -> Result<Vec<PathBuf>, String> {
        if !self.root.is_dir() {
            return Ok(Vec::new());
        }
        let entries =
            std::fs::read_dir(&self.root).map_err(|e| format!("{}: {e}", self.root.display()))?;
        let mut dirs = Vec::new();
        for entry in entries {
            let entry = entry.map_err(|e| format!("{}: {e}", self.root.display()))?;
            let path = entry.path();
            if skip == Some(path.as_path()) {
                continue;
            }
            // Matched on bytes, not `to_str()`: a name that is not valid UTF-8
            // is still a directory the fleet may have written history into,
            // and dropping it here would be silent.
            let matches = path
                .file_name()
                .is_some_and(|n| n.as_encoded_bytes().starts_with(prefix));
            if matches && path.is_dir() {
                dirs.push(path);
            }
        }
        dirs.sort();
        Ok(dirs)
    }

    /// Pre-#76 corpus-keyed screen directories, sorted, read-only.
    fn legacy_screens_dirs(&self) -> Result<Vec<PathBuf>, String> {
        self.dirs_named(LEGACY_SCREENS_PREFIX.as_bytes(), None)
    }

    /// Sibling `corpus-*` directories other than this run's, sorted (Issue #88).
    fn other_corpus_dirs(&self) -> Result<Vec<PathBuf>, String> {
        self.dirs_named(CORPUS_DIR_PREFIX.as_bytes(), Some(&self.corpus_dir()))
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

    /// Verdicts filed against **other** corpora — evidence, never truth (#88).
    ///
    /// GRQ regenerates the training corpus before every run, so a prune the
    /// fleet accepted under earlier training data is never consulted again by
    /// [`Self::load`], which reads this corpus alone. These records say only
    /// *this uuid was worth removing once*, which is why they may reorder the
    /// sweep ([`prior_corpus_priority`]) and be replayed as hypotheses
    /// ([`historical_replay`]), and may never be mixed into `load`: a
    /// foreign-corpus verdict must still not suppress or accept anything.
    ///
    /// Each record is returned stamped with the epoch it was established in —
    /// the identity in its directory name (Issue #101) — so nothing that reads
    /// this history has to guess which corpus produced it, and a corpus change
    /// never costs the fleet a learning.
    ///
    /// A fault in one foreign directory is **warned and skipped** rather than
    /// failing the load, matching [`Self::load_screens`]: nothing rewrites those
    /// directories, and one truncated line must not cost the evidence from every
    /// other corpus. An unreadable root is still an error.
    pub fn load_prior_corpora(&self) -> Result<Vec<HistoricalLearning>, String> {
        let keep = |l: &Learning| l.version == LEARNINGS_FORMAT_VERSION;
        let mut out = Vec::new();
        for dir in self.other_corpus_dirs()? {
            match load_jsonl(&dir, keep) {
                Ok(records) => out.extend(stamp_epoch(records, &dir)),
                Err(e) => crate::log::warn(&format!(
                    "prior-corpus verdicts unreadable ({e}); skipping {}",
                    dir.display()
                )),
            }
        }
        Ok(out)
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
    ///
    /// A legacy record names no corpus of its own — the identity was the
    /// directory name — so it is stamped with the identity from that name as it
    /// is read (Issue #100). Without that, screening epochs would read the
    /// fleet's pre-#76 coverage as belonging to no corpus at all, and a host
    /// that had not run since would re-screen a creature it had already
    /// finished under the very corpus in hand.
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
                Ok(records) => out.extend(stamp_legacy_identity(records, &legacy)),
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

/// One verdict an **older** corpus epoch established (Issue #101).
///
/// A [`Learning`] never carries a corpus identity of its own — the
/// `corpus-<identity>/` directory it sits in is what names the epoch — so
/// reading one out of that directory would otherwise strip the very fact that
/// makes it history rather than a current verdict. Stamping it on the way out
/// keeps every historical learning attributable to the corpus it was measured
/// against, which is what longitudinal reporting and future heuristics need and
/// what stops a foreign record being mistaken for a current one.
#[derive(Debug, Clone, PartialEq)]
pub struct HistoricalLearning {
    /// Corpus identity of the epoch that established this evidence.
    pub corpus_identity: String,
    /// The verdict exactly as it was filed.
    pub learning: Learning,
}

/// Stamp records read from `corpus-<identity>/` with that epoch (Issue #101).
///
/// A directory name that is not valid UTF-8 is carried across lossily: it can
/// never equal a live identity, so it labels the epoch for reporting without
/// ever being mistaken for the corpus in hand — and dropping the records
/// instead would lose fleet history, which is the one thing this must not do.
/// A path with no final component at all — which [`LearningsStore::other_corpus_dirs`]
/// cannot produce, since every entry it returns is a named directory — labels
/// its records with the empty identity for the same reason: an unnameable epoch
/// is still evidence, and it can match no corpus in hand.
fn stamp_epoch(records: Vec<Learning>, dir: &Path) -> Vec<HistoricalLearning> {
    let name = dir
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    let corpus_identity = name.strip_prefix(CORPUS_DIR_PREFIX).unwrap_or(&name);
    records
        .into_iter()
        .map(|learning| HistoricalLearning {
            corpus_identity: corpus_identity.to_string(),
            learning,
        })
        .collect()
}

/// Give pre-#76 records the corpus identity their directory name carries (#100).
///
/// `screens-<identity>/` **is** the record's corpus identity; it simply lived in
/// the path rather than on the record. Recovering it here is what keeps the
/// fleet's pre-#76 coverage inside the epoch it was measured against. A record
/// that already names a corpus is left alone, and a directory name that is not
/// valid UTF-8 names no identity we could match against, so those records stay
/// as they are — history, never current-epoch coverage.
fn stamp_legacy_identity(records: Vec<Screened>, dir: &Path) -> Vec<Screened> {
    let Some(identity) = dir
        .file_name()
        .and_then(|n| n.to_str())
        .and_then(|n| n.strip_prefix(LEGACY_SCREENS_PREFIX))
        .filter(|id| !id.is_empty())
    else {
        return records;
    };
    records
        .into_iter()
        .map(|mut s| {
            s.corpus_identity
                .get_or_insert_with(|| identity.to_string());
            s
        })
        .collect()
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
        CandidateKind::Constant => "constant",
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

/// Still-present uuids an **older** corpus once spoke well of (Issue #88).
///
/// The set the sweep checks first: a hidden neuron the fleet removed — or
/// measured a positive full-corpus delta for — under earlier training data,
/// that is still on the incumbent and has **not** been screened under
/// `corpus_identity` yet. It was removable under at least one corpus, so it is
/// a better first guess than a neuron with no history at all.
///
/// A hint, never a verdict. Every uuid returned here still faces the sample
/// screen and full-corpus scoring; nothing in `prior` may suppress, replay or
/// accept a cut.
///
/// Old rejections neither promote nor **demote**: the qualifying records are
/// selected *before* the latest-per-uuid collapse, so a corpus that later
/// rejected a uuid another corpus had accepted cannot cancel the hint. Filtering
/// after the collapse — as replay does, where the newest verdict about *this*
/// corpus is the only one that counts — would have made a foreign `Rejected`
/// deprioritise a neuron, which is exactly the per-corpus suppression this hint
/// must not reintroduce.
///
/// Within the qualifying records the ranking is [`ranked_confirmed`]'s, so a
/// host prioritising from the same records builds the same queue: applied cuts
/// first, then the best measured delta, with recency and uuid breaking ties.
///
/// Screening is cross-corpus (Issue #76) but this filter is not: a uuid already
/// looked at under *this* corpus needs no priority, while one screened only
/// under an older corpus — or by a pre-#76 record, which names no corpus — has
/// still not been tried against the data in hand.
pub fn prior_corpus_priority(
    prior: &[HistoricalLearning],
    screens: &[Screened],
    creature: &CreatureExport,
    corpus_identity: &str,
    min_improvement: f64,
) -> Vec<String> {
    let checked: HashSet<&str> = screens
        .iter()
        .filter(|s| s.corpus_identity.as_deref() == Some(corpus_identity))
        .map(|s| s.uuid.as_str())
        .collect();
    let qualifying = historical_wins(prior, min_improvement, |uuid| !checked.contains(uuid));
    ranked_confirmed(&qualifying, creature, min_improvement)
        .into_iter()
        .map(|win| win.uuid)
        .collect()
}

/// Historical records that spoke well of a cut, for uuids `keep_uuid` accepts.
///
/// Selected *before* the latest-per-uuid collapse [`ranked_confirmed`] applies,
/// so a corpus that later rejected what another corpus accepted cannot cancel
/// the evidence — per-corpus suppression must never cross corpora (#88).
fn historical_wins(
    prior: &[HistoricalLearning],
    min_improvement: f64,
    keep_uuid: impl Fn(&str) -> bool,
) -> Vec<Learning> {
    prior
        .iter()
        .map(|h| &h.learning)
        .filter(|l| l.outcome == Outcome::Accepted || confirmed_positive(l, min_improvement))
        .filter(|l| keep_uuid(l.uuid.as_str()))
        .cloned()
        .collect()
}

/// Previous winners worth replaying against the corpus in hand (Issue #101).
///
/// A cut an older epoch confirmed is the fleet's best hypothesis about the new
/// corpus — it was worth full-corpus scoring once — so it is replayed early
/// rather than waiting for the sweep to reach it. It is only ever a hypothesis:
/// the replay stage re-scores every uuid returned here against the **current**
/// corpus, and nothing is accepted without that current result.
///
/// A uuid this corpus has already judged is left out whatever it said: an
/// [`Outcome::Accepted`] or confirmed record is replayed by
/// [`ranked_confirmed`] over `known`, and a [`Outcome::Rejected`] one is this
/// corpus's own current verdict, which history does not get to overrule. That
/// is not suppression by history — a historical failure suppresses nothing, and
/// a uuid with no current-corpus verdict is screened on its merits regardless of
/// what an older epoch made of it.
///
/// Ranked by [`ranked_confirmed`], so every host builds the same replay queue:
/// applied cuts first, then the best measured delta, recency and uuid breaking
/// ties.
pub fn historical_replay(
    prior: &[HistoricalLearning],
    known: &[Learning],
    creature: &CreatureExport,
    min_improvement: f64,
) -> Vec<ConfirmedWin> {
    let judged: HashSet<&str> = known.iter().map(|l| l.uuid.as_str()).collect();
    let qualifying = historical_wins(prior, min_improvement, |uuid| !judged.contains(uuid));
    ranked_confirmed(&qualifying, creature, min_improvement)
}

/// How many verdicts each historical epoch established (Issue #101).
///
/// The longitudinal view of the cache: what the fleet has learnt, attributed to
/// the corpus it learnt it under, so a corpus change is visible as evidence
/// gained rather than evidence lost.
///
/// Ordered by corpus identity, not by age: an identity is a content hash
/// (`crate::corpus`), so it says nothing about when the epoch ran, and sorting
/// on it is only there to give every host the same stable list.
pub fn history_epochs(prior: &[HistoricalLearning]) -> Vec<(String, usize)> {
    let mut counts: BTreeMap<&str, usize> = BTreeMap::new();
    for h in prior {
        *counts.entry(h.corpus_identity.as_str()).or_default() += 1;
    }
    counts
        .into_iter()
        .map(|(identity, n)| (identity.to_string(), n))
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
/// This is the coverage set: "sweep X of Y hidden". It says nothing about
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
    /// Why the razor could propose nothing, on a blocked visit (Issue #103).
    pub blocked_reason: Option<BlockedReason>,
}

impl<'a> ScreenTry<'a> {
    /// A candidate the scorer screened.
    pub fn scored(uuid: &'a str, kind: CandidateKind, outcome: ScreenOutcomeKind) -> Self {
        Self {
            uuid,
            kind: kind_label(kind),
            outcome,
            version: SCREENS_FORMAT_VERSION,
            blocked_reason: None,
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
            blocked_reason: None,
        }
    }

    /// A visit the razor could propose nothing for, and why (Issue #103).
    ///
    /// The [`SCREEN_KIND_SKIPPED`] half of [`Self::visited`], with the reason
    /// carried onto the record so the blocked population can be broken down
    /// long after the run that filed it has gone.
    pub fn blocked(uuid: &'a str, reason: BlockedReason) -> Self {
        Self {
            uuid,
            kind: SCREEN_KIND_SKIPPED,
            outcome: ScreenOutcomeKind::Loser,
            version: SCREENS_VISIT_FORMAT_VERSION,
            blocked_reason: Some(reason),
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
            blocked_reason: t.blocked_reason,
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
            blocked_reason: Default::default(),
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

    /// Issue #76: a corpus change never costs the fleet a screen **record**.
    ///
    /// Since #100 those records are no longer current-epoch coverage under the
    /// new corpus — [`current_epoch_screens`] decides that — but they are still
    /// read, still attributed to the corpus they were measured against, and
    /// still there when a host returns to that corpus.
    #[test]
    fn screens_survive_a_corpus_identity_change() {
        let dir = tempfile::tempdir().unwrap();
        let before = LearningsStore::new(dir.path(), "corpus-a".into(), "host-a".into());
        let s = Screened {
            corpus_identity: Some("corpus-a".into()),
            ..screen("h_a", ScreenOutcomeKind::Winner, 9)
        };
        before.append_screen(&s).unwrap();

        let after = LearningsStore::new(dir.path(), "corpus-b".into(), "host-a".into());
        let loaded = after.load_screens().unwrap();
        assert_eq!(
            loaded.iter().map(|s| s.uuid.as_str()).collect::<Vec<_>>(),
            vec!["h_a"],
            "the record must remain readable across a corpus change"
        );
        assert!(
            current_epoch_screens(loaded.clone(), "corpus-b").is_empty(),
            "it is history under the new corpus, not current-epoch coverage"
        );
        assert_eq!(
            current_epoch_screens(loaded, "corpus-a"),
            vec![s],
            "and its own epoch is intact when the host comes back to it"
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
        assert_eq!(
            legacy.corpus_identity.as_deref(),
            Some("old-corpus"),
            "the identity was in the path, and is restored from it (#100)"
        );
    }

    /// Issue #100: the pre-#76 layout put the identity in the directory name,
    /// so a legacy record must land in *that* corpus's epoch. Reading it as
    /// belonging to no corpus would make a host that has not run since re-screen
    /// a creature it had already finished under the corpus in hand.
    #[test]
    fn a_legacy_record_lands_in_the_epoch_its_directory_named() {
        let dir = tempfile::tempdir().unwrap();
        write_legacy_screen(
            dir.path(),
            "corpus-a",
            "GRQ-23",
            &legacy_screen_line("h_a", 5),
        );
        write_legacy_screen(
            dir.path(),
            "corpus-b",
            "GRQ-23",
            &legacy_screen_line("h_b", 6),
        );
        let store = LearningsStore::new(dir.path(), "corpus-a".into(), "host-a".into());
        let loaded = store.load_screens().unwrap();

        let epoch = current_epoch_screens(loaded.clone(), "corpus-a");
        assert_eq!(
            epoch.iter().map(|s| s.uuid.as_str()).collect::<Vec<_>>(),
            vec!["h_a"],
            "only the directory that named this corpus is current-epoch coverage"
        );
        assert_eq!(
            current_epoch_screens(loaded, "corpus-b")
                .iter()
                .map(|s| s.uuid.as_str())
                .collect::<Vec<_>>(),
            vec!["h_b"],
        );
    }

    /// A record that already names a corpus keeps it, and a directory whose
    /// name carries no identity leaves the records exactly as they were.
    #[test]
    fn stamping_never_overwrites_an_identity_or_invents_one() {
        let named = screen("h_a", ScreenOutcomeKind::Loser, 5);
        let stamped =
            stamp_legacy_identity(vec![named.clone()], Path::new("/tmp/screens-other-corpus"));
        assert_eq!(
            stamped,
            vec![named.clone()],
            "an identity is never rewritten"
        );

        let anonymous = Screened {
            corpus_identity: None,
            ..named
        };
        for dir in ["/tmp/screens-", "/tmp/screens", "/tmp/corpus-x"] {
            assert_eq!(
                stamp_legacy_identity(vec![anonymous.clone()], Path::new(dir)),
                vec![anonymous.clone()],
                "{dir} names no legacy corpus identity"
            );
        }
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

    /// Issue #88: an old corpus's verdicts are read as a hint, and are still
    /// kept out of the verdict set that suppresses and replays.
    #[test]
    fn prior_corpus_verdicts_are_read_as_a_hint_and_never_as_verdicts() {
        let dir = tempfile::tempdir().unwrap();
        let old = LearningsStore::new(dir.path(), "corpus-a".into(), "GRQ-23".into());
        old.append(&rec("h_a", Outcome::Accepted, 10)).unwrap();
        old.append(&confirmed("h_b", 9e-6, 11)).unwrap();
        let now = LearningsStore::new(dir.path(), "corpus-b".into(), "host-a".into());

        assert!(
            now.load().unwrap().is_empty(),
            "a foreign-corpus verdict must never join this corpus's verdicts"
        );
        let loaded = now.load_prior_corpora().unwrap();
        assert!(
            loaded.iter().all(|h| h.corpus_identity == "corpus-a"),
            "every historical record names the epoch that established it: {loaded:?}"
        );
        let mut prior: Vec<String> = loaded.into_iter().map(|h| h.learning.uuid).collect();
        prior.sort();
        assert_eq!(prior, vec!["h_a".to_string(), "h_b".to_string()]);

        let c = two_hidden();
        assert!(
            known_failures(
                &now.load().unwrap(),
                &c,
                ReplayConfig::default(),
                1_000_000,
                1e-6
            )
            .is_empty(),
            "the hint cannot suppress a candidate"
        );
    }

    /// This run's own corpus is not its own prior corpus.
    #[test]
    fn the_current_corpus_is_excluded_from_the_prior_hint() {
        let dir = tempfile::tempdir().unwrap();
        let store = LearningsStore::new(dir.path(), "corp".into(), "host-a".into());
        store.append(&rec("h_a", Outcome::Accepted, 10)).unwrap();
        assert!(store.load_prior_corpora().unwrap().is_empty());
        assert_eq!(store.load().unwrap().len(), 1);
    }

    /// Screen records are not verdicts, so `screens/` is not a corpus directory.
    #[test]
    fn the_screens_directory_is_never_read_as_a_prior_corpus() {
        let dir = tempfile::tempdir().unwrap();
        let store = LearningsStore::new(dir.path(), "corp".into(), "host-a".into());
        store
            .append_screen(&screen("h_a", ScreenOutcomeKind::Winner, 9))
            .unwrap();
        write_legacy_screen(
            dir.path(),
            "old-corpus",
            "GRQ-23",
            &legacy_screen_line("h_b", 5),
        );
        assert!(store.load_prior_corpora().unwrap().is_empty());
    }

    /// One corrupt foreign directory must not cost the hint every other corpus
    /// carries — the same containment rule the legacy screen union follows.
    #[test]
    fn a_corrupt_prior_corpus_directory_costs_only_its_own_records() {
        let dir = tempfile::tempdir().unwrap();
        let good = LearningsStore::new(dir.path(), "corpus-good".into(), "GRQ-23".into());
        good.append(&rec("h_a", Outcome::Accepted, 10)).unwrap();
        let broken = dir.path().join("corpus-broken");
        std::fs::create_dir_all(&broken).unwrap();
        std::fs::write(broken.join("GRQ-24.jsonl"), "{not json\n").unwrap();

        let now = LearningsStore::new(dir.path(), "corpus-now".into(), "host-a".into());
        let prior = now.load_prior_corpora().unwrap();
        assert_eq!(prior.len(), 1, "{prior:?}");
        assert_eq!(prior[0].learning.uuid, "h_a");
        assert_eq!(prior[0].corpus_identity, "corpus-good");
    }

    #[test]
    fn a_prior_hint_from_an_absent_root_is_empty() {
        let dir = tempfile::tempdir().unwrap();
        let store = LearningsStore::new(dir.path().join("nothing-here"), "corp".into(), "h".into());
        assert!(store.load_prior_corpora().unwrap().is_empty());
    }

    /// Verdicts as one older epoch's history — the shape `load_prior_corpora`
    /// returns, with the corpus that established them stamped on (#101).
    fn history(identity: &str, records: Vec<Learning>) -> Vec<HistoricalLearning> {
        records
            .into_iter()
            .map(|learning| HistoricalLearning {
                corpus_identity: identity.into(),
                learning,
            })
            .collect()
    }

    #[test]
    fn the_prior_priority_is_still_present_unchecked_and_best_delta_first() {
        let c = two_hidden();
        let prior = history(
            "corpus-old",
            vec![
                confirmed("h_a", 2e-6, 10),
                confirmed("h_b", 9e-6, 11),
                confirmed("gone", 9e-3, 12),
                confirmed("h_a", 1e-9, 5), // older and weak: the latest record wins
            ],
        );
        assert_eq!(
            prior_corpus_priority(&prior, &[], &c, "corp", 1e-6),
            vec!["h_b".to_string(), "h_a".to_string()],
            "the best measured delta leads, and a departed uuid is not a hint"
        );
    }

    /// The filter that makes this a *priority* rather than a re-screen: a uuid
    /// already looked at under this corpus is dropped, one looked at only under
    /// an older corpus — or by a pre-#76 record naming none — is kept.
    #[test]
    fn a_uuid_already_screened_under_this_corpus_is_not_prioritised() {
        let c = two_hidden();
        let prior = history(
            "corpus-old",
            vec![
                rec("h_a", Outcome::Accepted, 10),
                rec("h_b", Outcome::Accepted, 11),
            ],
        );
        let this_corpus = screen("h_a", ScreenOutcomeKind::Loser, 12);
        assert_eq!(this_corpus.corpus_identity.as_deref(), Some("corp"));
        let older = Screened {
            corpus_identity: Some("corpus-old".into()),
            ..screen("h_b", ScreenOutcomeKind::Loser, 12)
        };
        assert_eq!(
            prior_corpus_priority(&prior, &[this_corpus.clone(), older], &c, "corp", 1e-6),
            vec!["h_b".to_string()]
        );

        let pre_76 = Screened {
            corpus_identity: None,
            ..screen("h_b", ScreenOutcomeKind::Loser, 12)
        };
        assert_eq!(
            prior_corpus_priority(&prior, &[this_corpus, pre_76], &c, "corp", 1e-6),
            vec!["h_b".to_string()],
            "a record naming no corpus was not measured against this one"
        );
    }

    /// Old rejections do not deprioritise: failure suppression stays per-corpus,
    /// so an old `Rejected` simply never enters the hint.
    #[test]
    fn an_old_rejection_is_neither_a_hint_nor_a_penalty() {
        let c = two_hidden();
        let prior = history(
            "corpus-old",
            vec![
                rec("h_a", Outcome::Rejected, 10),
                confirmed("h_b", -1e-4, 11),
            ],
        );
        assert!(prior_corpus_priority(&prior, &[], &c, "corp", 1e-6).is_empty());
    }

    /// The half of "rejections do not deprioritise" that is easy to lose: two
    /// old corpora disagree, and the newer one rejected. Collapsing to the
    /// latest record first would drop the win, reinstating per-corpus
    /// suppression across corpora — the one thing this hint must not do.
    #[test]
    fn a_later_corpus_rejecting_does_not_cancel_an_earlier_corpus_win() {
        let c = two_hidden();
        let records = vec![
            rec("h_a", Outcome::Accepted, 10),
            rec("h_a", Outcome::Rejected, 20),
            confirmed("h_b", 9e-6, 10),
            rec("h_b", Outcome::Rejected, 20),
        ];
        let prior = history("corpus-old", records.clone());
        assert_eq!(
            prior_corpus_priority(&prior, &[], &c, "corp", 1e-6),
            vec!["h_a".to_string(), "h_b".to_string()],
            "a uuid one corpus removed is still the likeliest to be removable"
        );
        assert!(
            known_wins(&records, &c, ReplayConfig::default()).is_empty(),
            "replay of *this* corpus still reads the latest verdict only"
        );
    }

    #[test]
    fn loading_screens_from_an_absent_store_is_empty() {
        let dir = tempfile::tempdir().unwrap();
        let store = LearningsStore::new(dir.path(), "corp".into(), "host-a".into());
        assert!(store.load_screens().unwrap().is_empty());
    }

    /// A screen record filed under another corpus, kept for history (#100).
    fn other_epoch(uuid: &str, identity: &str, secs: u64) -> Screened {
        Screened {
            corpus_identity: Some(identity.into()),
            ..screen(uuid, ScreenOutcomeKind::Loser, secs)
        }
    }

    /// Issue #100: only the records measured against the corpus in hand are
    /// current-epoch coverage — whatever they said, and however old they are.
    #[test]
    fn the_current_epoch_is_the_records_filed_under_this_corpus() {
        let history = vec![
            screen("h_a", ScreenOutcomeKind::Winner, 30),
            other_epoch("h_b", "corpus-old", 20),
            Screened {
                kind: SCREEN_KIND_SKIPPED.into(),
                version: SCREENS_VISIT_FORMAT_VERSION,
                ..other_epoch("h_c", "corpus-old", 21)
            },
            Screened {
                kind: SCREEN_KIND_KNOWN_FAILURE.into(),
                version: SCREENS_VISIT_FORMAT_VERSION,
                ..other_epoch("h_d", "corpus-old", 22)
            },
            Screened {
                corpus_identity: None,
                ..screen("h_e", ScreenOutcomeKind::Loser, 10)
            },
        ];
        let epoch = current_epoch_screens(history.clone(), "corp");
        assert_eq!(
            epoch.iter().map(|s| s.uuid.as_str()).collect::<Vec<_>>(),
            vec!["h_a"],
            "a screened winner, a blocked visit and a known failure of the \
             previous corpus are all outside this epoch"
        );
        assert_eq!(
            history.len(),
            5,
            "the history itself is untouched — this invalidates authority, not records"
        );
    }

    /// Coverage is *selected*, never cleared, so an identity a host screened
    /// before still has its epoch when the host comes back to it.
    #[test]
    fn returning_to_an_earlier_corpus_finds_that_epoch_intact() {
        let history = vec![
            screen("h_a", ScreenOutcomeKind::Loser, 10),
            other_epoch("h_b", "corpus-old", 20),
        ];
        for (identity, expected) in [("corp", "h_a"), ("corpus-old", "h_b")] {
            let epoch = current_epoch_screens(history.clone(), identity);
            assert_eq!(
                epoch.iter().map(|s| s.uuid.as_str()).collect::<Vec<_>>(),
                vec![expected],
                "epoch {identity}"
            );
        }
        assert!(
            current_epoch_screens(history, "corpus-never-seen").is_empty(),
            "an identity this host has not screened opens at zero coverage"
        );
    }

    /// Every record the store appends is filed under the epoch it is reading,
    /// so a run's own screens are always in its own epoch.
    #[test]
    fn records_a_store_files_belong_to_the_epoch_it_reads() {
        let dir = tempfile::tempdir().unwrap();
        let store = LearningsStore::new(dir.path(), "corp-9".into(), "host-a".into());
        assert_eq!(store.corpus_identity(), "corp-9");
        let mut known = Vec::new();
        let filed = file_screens(
            Some(&store),
            &[ScreenTry::scored(
                "h_a",
                CandidateKind::Ablation,
                ScreenOutcomeKind::Loser,
            )],
            &mut known,
        );
        assert_eq!(filed, 1);
        let loaded = store.load_screens().unwrap();
        assert_eq!(loaded.len(), 1);
        assert!(loaded[0].in_epoch("corp-9"), "{:?}", loaded[0]);
        assert_eq!(current_epoch_screens(loaded, "corp-9").len(), 1);
    }

    /// Issue #101: a new corpus epoch keeps every learning the fleet has, each
    /// still attributable to the epoch that established it. Nothing is deleted
    /// merely because the training data moved on.
    #[test]
    fn a_new_corpus_epoch_keeps_the_learnings_it_inherited() {
        let dir = tempfile::tempdir().unwrap();
        for (identity, uuid) in [("corpus-a", "h_a"), ("corpus-b", "h_b")] {
            LearningsStore::new(dir.path(), identity.into(), "GRQ-23".into())
                .append(&rec(uuid, Outcome::Accepted, 10))
                .unwrap();
        }

        let fresh = LearningsStore::new(dir.path(), "corpus-c".into(), "host-a".into());
        assert!(
            fresh.load().unwrap().is_empty(),
            "the new epoch opens with no verdicts of its own"
        );
        let prior = fresh.load_prior_corpora().unwrap();
        assert_eq!(prior.len(), 2, "{prior:?}");
        assert_eq!(
            history_epochs(&prior),
            vec![("corpus-a".to_string(), 1), ("corpus-b".to_string(), 1)],
            "each learning is still attributed to the corpus that established it"
        );
    }

    /// Issue #101: a historical failure is evidence, not a verdict, so it can
    /// neither suppress the uuid nor keep it out of the current epoch's work.
    #[test]
    fn a_historical_failure_leaves_the_uuid_eligible_again() {
        let c = two_hidden();
        let now = 1_000_000;
        let prior = history(
            "corpus-old",
            vec![
                rec("h_a", Outcome::Rejected, now - 60),
                confirmed("h_b", -1e-4, now - 60),
            ],
        );
        let this_corpus: Vec<Learning> = Vec::new();

        assert!(
            known_failures(&this_corpus, &c, ReplayConfig::default(), now, 1e-6).is_empty(),
            "an old failure must not suppress a current-epoch visit"
        );
        assert!(
            prior_corpus_priority(&prior, &[], &c, "corp", 1e-6).is_empty(),
            "nor does it earn a priority it never measured"
        );
        assert!(
            historical_replay(&prior, &this_corpus, &c, 1e-6).is_empty(),
            "and a failure is not a hypothesis worth replaying"
        );
    }

    /// Issue #101: a winner an older epoch confirmed is replayed against the
    /// corpus in hand — accepted cuts first, best measured delta next, and a
    /// uuid that has departed the creature is no hypothesis at all.
    #[test]
    fn historical_winners_are_replayed_best_evidence_first() {
        let c = two_hidden();
        let prior = [
            history("corpus-old", vec![confirmed("h_a", 9e-6, 10)]),
            history(
                "corpus-older",
                vec![
                    rec("h_b", Outcome::Accepted, 5),
                    rec("gone", Outcome::Accepted, 6),
                ],
            ),
        ]
        .concat();

        let replay = historical_replay(&prior, &[], &c, 1e-6);
        assert_eq!(
            replay.iter().map(|c| c.uuid.as_str()).collect::<Vec<_>>(),
            vec!["h_b", "h_a"],
            "an applied cut outranks a confirmed-only one, and a departed uuid is dropped"
        );
        assert!(replay[0].accepted);
        assert_eq!(replay[1].full_delta, Some(9e-6));
    }

    /// The current scorer is truth: once this corpus has judged a uuid, its own
    /// verdict stands and history stops proposing it — an accepted or confirmed
    /// one is replayed from `known`, a rejected one is this corpus's answer.
    #[test]
    fn a_verdict_from_this_corpus_settles_what_history_only_suggests() {
        let c = two_hidden();
        let prior = history(
            "corpus-old",
            vec![
                rec("h_a", Outcome::Accepted, 10),
                rec("h_b", Outcome::Accepted, 10),
            ],
        );
        let this_corpus = vec![rec("h_a", Outcome::Rejected, 20)];

        assert_eq!(
            historical_replay(&prior, &this_corpus, &c, 1e-6)
                .iter()
                .map(|c| c.uuid.as_str())
                .collect::<Vec<_>>(),
            vec!["h_b"],
            "the uuid this corpus scored is settled here; the untried one is not"
        );
        assert_eq!(
            prior_corpus_priority(&prior, &[], &c, "corp", 1e-6),
            vec!["h_a".to_string(), "h_b".to_string()],
            "ordering evidence is unaffected — it never decides anything"
        );
    }

    /// Two epochs disagree about the same uuid: the win still replays, because
    /// per-corpus suppression must not cross corpora (#88) and a historical
    /// failure suppresses nothing (#101).
    #[test]
    fn one_epoch_rejecting_does_not_withdraw_another_epoch_s_hypothesis() {
        let c = two_hidden();
        let prior = [
            history("corpus-old", vec![rec("h_a", Outcome::Accepted, 10)]),
            history("corpus-newer", vec![rec("h_a", Outcome::Rejected, 20)]),
        ]
        .concat();
        assert_eq!(
            historical_replay(&prior, &[], &c, 1e-6)
                .iter()
                .map(|c| c.uuid.as_str())
                .collect::<Vec<_>>(),
            vec!["h_a"],
        );
    }

    /// The epoch summary counts what each corpus taught, so a corpus change
    /// reads as evidence gained rather than evidence lost.
    #[test]
    fn the_epoch_summary_counts_the_verdicts_each_corpus_established() {
        let prior = [
            history(
                "corpus-b",
                vec![
                    rec("h_a", Outcome::Accepted, 1),
                    rec("h_b", Outcome::Rejected, 2),
                ],
            ),
            history("corpus-a", vec![rec("h_c", Outcome::Accepted, 3)]),
        ]
        .concat();
        assert_eq!(
            history_epochs(&prior),
            vec![("corpus-a".to_string(), 1), ("corpus-b".to_string(), 2)],
        );
        assert!(history_epochs(&[]).is_empty());
    }

    /// The identity lives in the directory name, so it must survive the read —
    /// including a name that carries no `corpus-` prefix at all.
    #[test]
    fn stamping_an_epoch_takes_the_identity_from_the_directory_name() {
        let l = rec("h_a", Outcome::Accepted, 10);
        assert_eq!(
            stamp_epoch(vec![l.clone()], Path::new("/tmp/learnings/corpus-6fc028da")),
            vec![HistoricalLearning {
                corpus_identity: "6fc028da".into(),
                learning: l.clone(),
            }],
        );
        assert_eq!(
            stamp_epoch(vec![l.clone()], Path::new("/tmp/learnings/odd-name"))[0].corpus_identity,
            "odd-name",
            "an unexpected name still labels its records rather than losing them"
        );
        assert!(stamp_epoch(Vec::new(), Path::new("/tmp/corpus-x")).is_empty());
    }
}
