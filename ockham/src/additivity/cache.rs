//! Keyed, bounded on-disk output cache for the additivity benchmark (Issue #260).
//!
//! An entry holds the network outputs for one set of records as raw
//! little-endian `f32` values, record-major (`records × outputs` values). Each
//! [`CacheKey`] owns its own directory, so an entry produced under one key is
//! never served under another. The cache is never authoritative: a rejected
//! entry means "no estimate", never "loss".

use std::ffi::OsStr;
use std::fmt;
use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Read, Write};
use std::ops::ControlFlow;
use std::path::{Path, PathBuf};

use neat_core::training_data::TrainingDataConfig;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::corpus::{RecordRange, for_each_selected_chunk};
use crate::incumbent::sha256_hex;

/// Version of the entry layout; part of every [`CacheKey`].
pub const CACHE_FORMAT_VERSION: u32 = 1;

/// Name of the file that records the key owning a cache directory.
const KEY_FILE: &str = "key.json";

/// Extension of a finished entry file.
const ENTRY_EXT: &str = "f32";

/// Identity of one cached output set. Its digest names the cache directory.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CacheKey {
    /// Entry layout version, see [`CACHE_FORMAT_VERSION`].
    pub format_version: u32,
    /// Digest of the parent creature the outputs were computed from.
    pub parent_digest: String,
    /// Content digest of the records, see [`dataset_digest`].
    pub dataset_digest: String,
    /// Corpus identity from `corpus_info`, recorded alongside the content digest.
    pub corpus_identity: String,
    /// Digest of the record ranges, see [`ranges_digest`].
    pub record_ranges_digest: String,
    /// Number of records covered by each entry.
    pub records: u64,
    /// Network input width.
    pub input_count: usize,
    /// Network output width; entries hold `records × output_count` values.
    pub output_count: usize,
    /// Semantics of the cut the outputs were taken at.
    pub cut_semantics: String,
    /// Pinned `neat-core` version line, see [`core_pin`].
    pub core_pin: String,
    /// Ockham crate version that produced the outputs.
    pub ockham_version: String,
    /// Identity of the external scorer.
    pub scorer_identity: String,
    /// Cost function name.
    pub cost: String,
    /// Arguments the scorer was run with.
    pub scorer_args: Vec<String>,
}

impl CacheKey {
    /// Hex SHA-256 of the key's canonical JSON; names the cache directory.
    pub fn digest(&self) -> String {
        let canonical = serde_json::to_string(self).expect("CacheKey always serialises");
        sha256_hex(canonical.as_bytes())
    }
}

/// The `neat-core` pin line from `ockham/Cargo.toml`, trimmed.
///
/// Returns `neat-core = <unknown>` if the line is absent.
pub fn core_pin() -> String {
    include_str!("../../Cargo.toml")
        .lines()
        .map(str::trim)
        .find(|line| line.starts_with("neat-core ="))
        .map(str::to_string)
        .unwrap_or_else(|| "neat-core = <unknown>".to_string())
}

/// Hex SHA-256 of the serialised, ordered record ranges.
pub fn ranges_digest(ranges: &[RecordRange]) -> String {
    let canonical = serde_json::to_string(ranges).expect("record ranges always serialise");
    sha256_hex(canonical.as_bytes())
}

/// Strong content identity of the selected records.
///
/// Streams the selected records and hashes, for each record in order, its input
/// values then its target values as little-endian `f32` bytes. Returns the hex
/// digest and the number of records visited. Unlike the head/tail fingerprint in
/// `corpus_info`, this changes when any selected record's value changes.
pub fn dataset_digest(
    dir: &Path,
    config: &TrainingDataConfig,
    ranges: &[RecordRange],
    chunk_records: usize,
) -> Result<(String, u64), String> {
    let mut hasher = Sha256::new();
    let (ni, no) = (config.num_inputs, config.num_outputs);
    let visited = for_each_selected_chunk(dir, config, ranges, chunk_records, |chunk| {
        for r in 0..chunk.records {
            for v in &chunk.inputs[r * ni..(r + 1) * ni] {
                hasher.update(v.to_le_bytes());
            }
            for v in &chunk.targets[r * no..(r + 1) * no] {
                hasher.update(v.to_le_bytes());
            }
        }
        Ok(ControlFlow::Continue(()))
    })?;
    Ok((format!("{:x}", hasher.finalize()), visited))
}

/// Entry file name for a cut, derived from its UUID.
///
/// `"cut-"` followed by the first 32 hex characters of `sha256(uuid)`.
pub fn cut_entry_name(uuid: &str) -> String {
    let digest = sha256_hex(uuid.as_bytes());
    format!("cut-{}", &digest[..32])
}

/// Why an entry could not be written or read. Each variant means "no estimate".
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "camelCase")]
pub enum CacheRejection {
    /// A NaN or infinite value was found.
    NonFinite {
        /// Absolute index of the offending value within the entry.
        index: u64,
    },
    /// The entry holds a different number of bytes than its key requires.
    Mismatched {
        /// Bytes the key requires.
        expected_bytes: u64,
        /// Bytes actually present or supplied.
        actual_bytes: u64,
    },
    /// Writing the entry would exceed the cache's byte limit.
    OverLimit {
        /// Bytes the cache would hold after the write.
        needed_bytes: u64,
        /// Configured byte limit.
        limit_bytes: u64,
    },
    /// No entry file exists under this name.
    Missing,
    /// A filesystem or naming error.
    Io {
        /// Human-readable description.
        detail: String,
    },
}

impl fmt::Display for CacheRejection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonFinite { index } => write!(f, "non-finite value at index {index}"),
            Self::Mismatched {
                expected_bytes,
                actual_bytes,
            } => write!(
                f,
                "entry is {actual_bytes} bytes, expected {expected_bytes}"
            ),
            Self::OverLimit {
                needed_bytes,
                limit_bytes,
            } => write!(
                f,
                "entry would take the cache to {needed_bytes} bytes, over the {limit_bytes}-byte limit"
            ),
            Self::Missing => write!(f, "entry is missing"),
            Self::Io { detail } => write!(f, "cache I/O error: {detail}"),
        }
    }
}

fn io_rejection(e: std::io::Error) -> CacheRejection {
    CacheRejection::Io {
        detail: e.to_string(),
    }
}

fn unusable_writer() -> CacheRejection {
    CacheRejection::Io {
        detail: "writer is unusable after a failed push".to_string(),
    }
}

/// True for names matching `[A-Za-z0-9_-]{1,64}`.
fn valid_entry_name(name: &str) -> bool {
    (1..=64).contains(&name.len())
        && name
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-')
}

/// Size of a file, or zero if it is absent or not a regular file.
fn existing_len(path: &Path) -> u64 {
    fs::metadata(path)
        .ok()
        .filter(|m| m.is_file())
        .map_or(0, |m| m.len())
}

/// Total size of the finished `*.f32` entries in a cache directory.
fn existing_entry_bytes(dir: &Path) -> Result<u64, String> {
    let list_err = |e: std::io::Error| format!("cannot list cache '{}': {e}", dir.display());
    let mut total = 0u64;
    for entry in fs::read_dir(dir).map_err(list_err)? {
        let entry = entry.map_err(list_err)?;
        let path = entry.path();
        if path.extension() == Some(OsStr::new(ENTRY_EXT)) {
            let meta = entry.metadata().map_err(list_err)?;
            if meta.is_file() {
                total = total.saturating_add(meta.len());
            }
        }
    }
    Ok(total)
}

/// A keyed, byte-bounded output cache rooted at one key's directory.
#[derive(Debug)]
pub struct OutputCache {
    dir: PathBuf,
    key: CacheKey,
    limit: u64,
    used: u64,
}

impl OutputCache {
    /// Open or create the cache directory for `key` under `root`.
    ///
    /// The directory is `root/<key digest>`. An existing `key.json` must equal
    /// `key`; a mismatch is refused rather than reused.
    pub fn open(root: &Path, key: CacheKey, max_bytes: u64) -> Result<Self, String> {
        let dir = root.join(key.digest());
        let key_path = dir.join(KEY_FILE);
        if key_path.is_file() {
            let text = fs::read_to_string(&key_path)
                .map_err(|e| format!("cannot read '{}': {e}", key_path.display()))?;
            let stored: CacheKey = serde_json::from_str(&text)
                .map_err(|e| format!("cannot parse '{}': {e}", key_path.display()))?;
            if stored != key {
                return Err(format!(
                    "cache key in '{}' does not match its directory; refusing to reuse it",
                    key_path.display()
                ));
            }
        } else {
            fs::create_dir_all(&dir)
                .map_err(|e| format!("cannot create cache '{}': {e}", dir.display()))?;
            let pretty = serde_json::to_string_pretty(&key)
                .map_err(|e| format!("cannot serialise cache key: {e}"))?;
            fs::write(&key_path, pretty)
                .map_err(|e| format!("cannot write '{}': {e}", key_path.display()))?;
        }
        let used = existing_entry_bytes(&dir)?;
        Ok(Self {
            dir,
            key,
            limit: max_bytes,
            used,
        })
    }

    /// Directory holding this key's entries.
    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// The key this cache was opened with.
    pub fn key(&self) -> &CacheKey {
        &self.key
    }

    /// Bytes currently held by finished entries.
    pub fn used_bytes(&self) -> u64 {
        self.used
    }

    /// Size of one complete entry: `records × outputs × 4`.
    pub fn entry_bytes(&self) -> u64 {
        self.key
            .records
            .saturating_mul(self.key.output_count as u64)
            .saturating_mul(4)
    }

    /// True when a finished entry named `name` exists with exactly [`Self::entry_bytes`].
    pub fn has(&self, name: &str) -> bool {
        self.entry_path(name)
            .ok()
            .and_then(|path| fs::metadata(path).ok())
            .is_some_and(|m| m.is_file() && m.len() == self.entry_bytes())
    }

    /// Begin writing entry `name`.
    ///
    /// Refused with [`CacheRejection::OverLimit`] before anything is written if
    /// the entry would exceed the byte limit. An existing entry of the same name
    /// is counted as released. Two writers for the same name must not be live at
    /// once.
    pub fn writer(&mut self, name: &str) -> Result<EntryWriter, CacheRejection> {
        let final_path = self.entry_path(name)?;
        let entry_bytes = self.entry_bytes();
        let needed = self
            .used
            .saturating_sub(existing_len(&final_path))
            .saturating_add(entry_bytes);
        if needed > self.limit {
            return Err(CacheRejection::OverLimit {
                needed_bytes: needed,
                limit_bytes: self.limit,
            });
        }
        let tmp_path = self.dir.join(format!("{name}.{ENTRY_EXT}.tmp"));
        let file = File::create(&tmp_path).map_err(io_rejection)?;
        Ok(EntryWriter {
            out: Some(BufWriter::new(file)),
            tmp_path,
            final_path,
            expected_values: self
                .key
                .records
                .saturating_mul(self.key.output_count as u64),
            written_values: 0,
            entry_bytes,
        })
    }

    /// Open entry `name` for sequential reading.
    ///
    /// [`CacheRejection::Missing`] if absent, [`CacheRejection::Mismatched`] if
    /// its size is not [`Self::entry_bytes`].
    pub fn reader(&self, name: &str) -> Result<EntryReader, CacheRejection> {
        let path = self.entry_path(name)?;
        let file = match File::open(&path) {
            Ok(file) => file,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                return Err(CacheRejection::Missing);
            }
            Err(e) => return Err(io_rejection(e)),
        };
        let actual_bytes = file.metadata().map_err(io_rejection)?.len();
        let expected_bytes = self.entry_bytes();
        if actual_bytes != expected_bytes {
            return Err(CacheRejection::Mismatched {
                expected_bytes,
                actual_bytes,
            });
        }
        Ok(EntryReader {
            inner: BufReader::new(file),
            total_values: self
                .key
                .records
                .saturating_mul(self.key.output_count as u64),
            read_values: 0,
        })
    }

    fn entry_path(&self, name: &str) -> Result<PathBuf, CacheRejection> {
        if !valid_entry_name(name) {
            return Err(CacheRejection::Io {
                detail: format!("entry name '{name}' must be 1 to 64 of [A-Za-z0-9_-]"),
            });
        }
        Ok(self.dir.join(format!("{name}.{ENTRY_EXT}")))
    }
}

/// Streams one entry to a temporary file; renamed into place by [`Self::finish`].
///
/// Dropping an unfinished writer removes its temporary file.
#[derive(Debug)]
pub struct EntryWriter {
    out: Option<BufWriter<File>>,
    tmp_path: PathBuf,
    final_path: PathBuf,
    expected_values: u64,
    written_values: u64,
    entry_bytes: u64,
}

impl EntryWriter {
    /// Append values to the entry.
    ///
    /// Any non-finite value is refused with [`CacheRejection::NonFinite`], and
    /// writing past the entry size is refused with [`CacheRejection::Mismatched`].
    /// A failed push leaves the writer unusable; drop it.
    pub fn push(&mut self, values: &[f32]) -> Result<(), CacheRejection> {
        let Some(out) = self.out.as_mut() else {
            return Err(unusable_writer());
        };
        let incoming = values.len() as u64;
        if self.written_values.saturating_add(incoming) > self.expected_values {
            self.out = None;
            return Err(CacheRejection::Mismatched {
                expected_bytes: self.expected_values.saturating_mul(4),
                actual_bytes: self
                    .written_values
                    .saturating_add(incoming)
                    .saturating_mul(4),
            });
        }
        if let Some(pos) = values.iter().position(|v| !v.is_finite()) {
            self.out = None;
            return Err(CacheRejection::NonFinite {
                index: self.written_values + pos as u64,
            });
        }
        for v in values {
            if let Err(e) = out.write_all(&v.to_le_bytes()) {
                self.out = None;
                return Err(io_rejection(e));
            }
        }
        self.written_values += incoming;
        Ok(())
    }

    /// Flush and rename the temporary file to the final entry name.
    ///
    /// Refused with [`CacheRejection::Mismatched`] if the number of values
    /// written is not `records × outputs`. The rename is also refused with
    /// [`CacheRejection::OverLimit`] if it would exceed the cache limit.
    pub fn finish(mut self, cache: &mut OutputCache) -> Result<(), CacheRejection> {
        if self.written_values != self.expected_values {
            return Err(CacheRejection::Mismatched {
                expected_bytes: self.expected_values.saturating_mul(4),
                actual_bytes: self.written_values.saturating_mul(4),
            });
        }
        let Some(mut out) = self.out.take() else {
            return Err(unusable_writer());
        };
        out.flush().map_err(io_rejection)?;
        let file = out.into_inner().map_err(|e| io_rejection(e.into_error()))?;
        drop(file);
        let used = cache
            .used
            .saturating_sub(existing_len(&self.final_path))
            .saturating_add(self.entry_bytes);
        if used > cache.limit {
            return Err(CacheRejection::OverLimit {
                needed_bytes: used,
                limit_bytes: cache.limit,
            });
        }
        fs::rename(&self.tmp_path, &self.final_path).map_err(io_rejection)?;
        cache.used = used;
        Ok(())
    }
}

impl Drop for EntryWriter {
    fn drop(&mut self) {
        // Best effort: an unfinished entry must not linger as a partial file.
        let _ = fs::remove_file(&self.tmp_path);
    }
}

/// Sequentially reads one entry, in chunks, with bounded memory.
#[derive(Debug)]
pub struct EntryReader {
    inner: BufReader<File>,
    total_values: u64,
    read_values: u64,
}

impl EntryReader {
    /// Fill `out` with the next `out.len()` values.
    ///
    /// Reading past the end is refused with [`CacheRejection::Mismatched`], and a
    /// non-finite value with [`CacheRejection::NonFinite`].
    pub fn read_chunk(&mut self, out: &mut [f32]) -> Result<(), CacheRejection> {
        let want = out.len() as u64;
        if self.read_values.saturating_add(want) > self.total_values {
            return Err(CacheRejection::Mismatched {
                expected_bytes: self.total_values.saturating_mul(4),
                actual_bytes: self.read_values.saturating_add(want).saturating_mul(4),
            });
        }
        let mut buf = [0u8; 4];
        for slot in out.iter_mut() {
            if let Err(e) = self.inner.read_exact(&mut buf) {
                return Err(if e.kind() == std::io::ErrorKind::UnexpectedEof {
                    CacheRejection::Mismatched {
                        expected_bytes: self.total_values.saturating_mul(4),
                        actual_bytes: self.read_values.saturating_mul(4),
                    }
                } else {
                    io_rejection(e)
                });
            }
            let v = f32::from_le_bytes(buf);
            if !v.is_finite() {
                return Err(CacheRejection::NonFinite {
                    index: self.read_values,
                });
            }
            *slot = v;
            self.read_values += 1;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::corpus::{corpus_info, write_bin_file};

    const LIMIT: u64 = 1 << 20;

    fn key() -> CacheKey {
        CacheKey {
            format_version: CACHE_FORMAT_VERSION,
            parent_digest: "parent-a".to_string(),
            dataset_digest: "dataset-a".to_string(),
            corpus_identity: "corpus-a".to_string(),
            record_ranges_digest: "ranges-a".to_string(),
            records: 3,
            input_count: 2,
            output_count: 2,
            cut_semantics: "cut-v1".to_string(),
            core_pin: "neat-core = v0".to_string(),
            ockham_version: "0.1.90".to_string(),
            scorer_identity: "scorer-a".to_string(),
            cost: "MSE".to_string(),
            scorer_args: vec!["--fast".to_string()],
        }
    }

    fn write_entry(
        cache: &mut OutputCache,
        name: &str,
        values: &[f32],
    ) -> Result<(), CacheRejection> {
        let mut writer = cache.writer(name)?;
        writer.push(values)?;
        writer.finish(cache)
    }

    fn read_all(cache: &OutputCache, name: &str) -> Result<Vec<f32>, CacheRejection> {
        let mut reader = cache.reader(name)?;
        let mut out = vec![0f32; (cache.entry_bytes() / 4) as usize];
        reader.read_chunk(&mut out)?;
        Ok(out)
    }

    fn six_values() -> Vec<f32> {
        vec![0.5, -1.0, 2.0, 3.25, -4.5, 6.0]
    }

    #[test]
    fn key_round_trips_and_entry_reads_back_identically() {
        let tmp = tempfile::tempdir().unwrap();
        let mut cache = OutputCache::open(tmp.path(), key(), LIMIT).unwrap();
        assert_eq!(cache.entry_bytes(), 24);
        write_entry(&mut cache, "y0", &six_values()).unwrap();
        assert_eq!(cache.used_bytes(), 24);
        drop(cache);

        let reopened = OutputCache::open(tmp.path(), key(), LIMIT).unwrap();
        assert_eq!(reopened.used_bytes(), 24);
        assert!(reopened.has("y0"));
        assert_eq!(read_all(&reopened, "y0").unwrap(), six_values());
    }

    #[test]
    fn changing_one_key_field_gives_a_fresh_directory_without_the_entry() {
        type Mutate = fn(&mut CacheKey);
        let mutations: [(&str, Mutate); 5] = [
            ("parent_digest", |k| {
                k.parent_digest = "parent-b".to_string()
            }),
            ("dataset_digest", |k| {
                k.dataset_digest = "dataset-b".to_string()
            }),
            ("record_ranges_digest", |k| {
                k.record_ranges_digest = "ranges-b".to_string()
            }),
            ("core_pin", |k| k.core_pin = "neat-core = v9".to_string()),
            ("scorer_identity", |k| {
                k.scorer_identity = "scorer-b".to_string()
            }),
        ];
        let tmp = tempfile::tempdir().unwrap();
        let mut base = OutputCache::open(tmp.path(), key(), LIMIT).unwrap();
        write_entry(&mut base, "y0", &six_values()).unwrap();

        for (field, mutate) in mutations {
            let mut other_key = key();
            mutate(&mut other_key);
            let other = OutputCache::open(tmp.path(), other_key, LIMIT).unwrap();
            assert_ne!(other.dir(), base.dir(), "{field} must change the directory");
            assert!(!other.has("y0"), "{field} must not reuse the entry");
            assert!(
                matches!(other.reader("y0"), Err(CacheRejection::Missing)),
                "{field} must read as Missing"
            );
        }
        assert!(base.has("y0"));
    }

    #[test]
    fn key_json_disagreeing_with_its_directory_is_refused() {
        let tmp = tempfile::tempdir().unwrap();
        let cache = OutputCache::open(tmp.path(), key(), LIMIT).unwrap();
        let key_path = cache.dir().join(KEY_FILE);
        let mut tampered = key();
        tampered.records = 99;
        fs::write(&key_path, serde_json::to_string(&tampered).unwrap()).unwrap();
        let err = OutputCache::open(tmp.path(), key(), LIMIT).unwrap_err();
        assert!(err.contains("does not match"), "error was: {err}");
    }

    #[test]
    fn non_finite_values_are_refused_and_leave_no_file() {
        let tmp = tempfile::tempdir().unwrap();
        let mut cache = OutputCache::open(tmp.path(), key(), LIMIT).unwrap();

        let mut writer = cache.writer("nan").unwrap();
        assert_eq!(
            writer.push(&[1.0, f32::NAN]),
            Err(CacheRejection::NonFinite { index: 1 })
        );
        drop(writer);

        let mut writer = cache.writer("inf").unwrap();
        assert_eq!(
            writer.push(&[f32::INFINITY]),
            Err(CacheRejection::NonFinite { index: 0 })
        );
        drop(writer);

        for name in ["nan", "inf"] {
            assert!(!cache.dir().join(format!("{name}.f32")).exists());
            assert!(!cache.dir().join(format!("{name}.f32.tmp")).exists());
            assert!(!cache.has(name));
        }
        assert_eq!(cache.used_bytes(), 0);
    }

    #[test]
    fn finishing_short_is_mismatched_and_leaves_no_file() {
        let tmp = tempfile::tempdir().unwrap();
        let mut cache = OutputCache::open(tmp.path(), key(), LIMIT).unwrap();
        let mut writer = cache.writer("short").unwrap();
        writer.push(&[1.0, 2.0, 3.0, 4.0]).unwrap();
        assert!(matches!(
            writer.finish(&mut cache),
            Err(CacheRejection::Mismatched {
                expected_bytes: 24,
                actual_bytes: 16
            })
        ));
        assert!(!cache.dir().join("short.f32").exists());
        assert!(!cache.dir().join("short.f32.tmp").exists());
    }

    #[test]
    fn truncated_entry_reads_as_mismatched() {
        let tmp = tempfile::tempdir().unwrap();
        let mut cache = OutputCache::open(tmp.path(), key(), LIMIT).unwrap();
        write_entry(&mut cache, "y0", &six_values()).unwrap();
        let path = cache.dir().join("y0.f32");
        fs::OpenOptions::new()
            .write(true)
            .open(&path)
            .unwrap()
            .set_len(10)
            .unwrap();
        assert!(!cache.has("y0"));
        assert!(matches!(
            cache.reader("y0"),
            Err(CacheRejection::Mismatched {
                expected_bytes: 24,
                actual_bytes: 10
            })
        ));
    }

    #[test]
    fn reading_past_the_end_is_mismatched() {
        let tmp = tempfile::tempdir().unwrap();
        let mut cache = OutputCache::open(tmp.path(), key(), LIMIT).unwrap();
        write_entry(&mut cache, "y0", &six_values()).unwrap();
        let mut reader = cache.reader("y0").unwrap();
        let mut out = [0f32; 7];
        assert!(matches!(
            reader.read_chunk(&mut out),
            Err(CacheRejection::Mismatched { .. })
        ));
    }

    #[test]
    fn over_limit_write_is_refused_and_writes_nothing() {
        let tmp = tempfile::tempdir().unwrap();
        // The limit holds exactly one 24-byte entry.
        let mut cache = OutputCache::open(tmp.path(), key(), 24).unwrap();
        write_entry(&mut cache, "y0", &six_values()).unwrap();
        assert!(matches!(
            cache.writer("y1"),
            Err(CacheRejection::OverLimit {
                needed_bytes: 48,
                limit_bytes: 24
            })
        ));
        assert!(!cache.dir().join("y1.f32").exists());
        assert!(!cache.dir().join("y1.f32.tmp").exists());
        assert_eq!(cache.used_bytes(), 24);

        // Replacing the existing entry releases its bytes first, so this fits.
        let replaced = [9.0, 9.5, 10.0, 10.5, 11.0, 11.5];
        write_entry(&mut cache, "y0", &replaced).unwrap();
        assert_eq!(cache.used_bytes(), 24);
        assert_eq!(read_all(&cache, "y0").unwrap(), replaced.to_vec());
    }

    #[test]
    fn unsafe_entry_names_are_refused() {
        let tmp = tempfile::tempdir().unwrap();
        let mut cache = OutputCache::open(tmp.path(), key(), LIMIT).unwrap();
        assert!(matches!(
            cache.writer("../escape"),
            Err(CacheRejection::Io { .. })
        ));
        assert!(!cache.has("../escape"));
    }

    #[test]
    fn dataset_digest_sees_a_middle_record_that_corpus_info_misses() {
        let tmp = tempfile::tempdir().unwrap();
        let cfg = TrainingDataConfig::new(1, 1);
        // 48 records of 8 bytes: the first and last 64 bytes cover records 0-7
        // and 40-47, so changing record 20 leaves the head/tail fingerprint alone.
        let mut records: Vec<(Vec<f32>, Vec<f32>)> = (0..48)
            .map(|i| (vec![i as f32 + 0.5], vec![(i * 3) as f32]))
            .collect();
        write_bin_file(&tmp.path().join("0.bin"), &records).unwrap();
        let ranges = [RecordRange { start: 0, len: 48 }];

        let info_before = corpus_info(tmp.path(), &cfg).unwrap();
        let (before, visited) = dataset_digest(tmp.path(), &cfg, &ranges, 5).unwrap();
        assert_eq!(visited, 48);

        records[20].1[0] = 999.0;
        write_bin_file(&tmp.path().join("0.bin"), &records).unwrap();

        let info_after = corpus_info(tmp.path(), &cfg).unwrap();
        let (after, _) = dataset_digest(tmp.path(), &cfg, &ranges, 5).unwrap();
        assert_eq!(info_before.identity, info_after.identity);
        assert_ne!(before, after);
    }

    #[test]
    fn helpers_produce_stable_well_formed_values() {
        let k = key();
        assert_eq!(k.digest(), key().digest());
        assert_eq!(k.digest().len(), 64);
        assert!(core_pin().starts_with("neat-core ="));
        let name = cut_entry_name("3f2a");
        assert_eq!(name.len(), 36);
        assert!(valid_entry_name(&name));
        assert_eq!(name, cut_entry_name("3f2a"));
        let short = [RecordRange { start: 0, len: 1 }];
        let long = [RecordRange { start: 0, len: 2 }];
        assert_eq!(ranges_digest(&short), ranges_digest(&short));
        assert_ne!(ranges_digest(&short), ranges_digest(&long));
    }
}
