//! Immutable incumbent creature (Issue #2).
//!
//! The supplied creature is read once, required to be forward-only, validated
//! through NEAT-AI-core `creature_validate`, checksummed, and copied
//! byte-for-byte into the run workspace. Nothing in Ockham ever writes to the
//! source path.

use std::fmt;
use std::path::{Path, PathBuf};

use neat_core::{
    CreatureExport, ValidateOptions, compile_creature, creature_to_json, creature_validate,
    parse_creature_json, validate_no_duplicate_synapses,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// Options every Ockham incumbent and candidate is validated with.
///
/// `forward_only: true` is the v1 contract: recurrent / self-connected
/// creatures are out of scope.
pub const OCKHAM_VALIDATE_OPTIONS: ValidateOptions = ValidateOptions {
    neurons: None,
    connections: None,
    feedback_loop: None,
    forward_only: true,
};

/// SHA-256 hex digest of a byte slice.
pub fn sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

/// Errors establishing the incumbent.
#[derive(Debug)]
pub enum IncumbentError {
    /// Filesystem failure.
    Io(PathBuf, std::io::Error),
    /// Version 1 rejects creatures that are not `forwardOnly: true`.
    NotForwardOnly,
    /// NEAT-AI-core rejected the creature.
    Creature(String),
    /// Canonical `creature_validate` failed.
    Invalid(String),
    /// Serialise→parse round trip did not reproduce the creature.
    RoundTrip(String),
    /// Checksum of the workspace copy differs from the source.
    CopyDrift {
        /// Expected (source) checksum.
        expected: String,
        /// Observed checksum of the copy.
        observed: String,
    },
}

impl fmt::Display for IncumbentError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(p, e) => write!(f, "{}: {e}", p.display()),
            Self::NotForwardOnly => write!(
                f,
                "v1 accepts forward-only creatures only; this creature has forwardOnly=false (recurrent/self-connected networks are out of scope)"
            ),
            Self::Creature(m) => write!(f, "creature rejected by NEAT-AI-core: {m}"),
            Self::Invalid(m) => write!(f, "creature.validate() failed: {m}"),
            Self::RoundTrip(m) => {
                write!(f, "creature does not round-trip through NEAT-AI-core: {m}")
            }
            Self::CopyDrift { expected, observed } => {
                write!(f, "workspace copy checksum {observed} != source {expected}")
            }
        }
    }
}

impl std::error::Error for IncumbentError {}

/// The immutable starting creature.
#[derive(Debug, Clone)]
pub struct Incumbent {
    /// Where it was read from (never written).
    pub source_path: PathBuf,
    /// Exact source bytes.
    pub text: String,
    /// SHA-256 of `text`.
    pub checksum: String,
    /// Parsed creature.
    pub creature: CreatureExport,
}

/// Metadata written beside the workspace copy.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct IncumbentMeta {
    /// SHA-256 of the source bytes.
    pub checksum: String,
    /// Source path as supplied.
    pub source_path: String,
    /// Input width.
    pub input: usize,
    /// Output width.
    pub output: usize,
    /// Listed (non-input) neuron count.
    pub neurons: usize,
    /// Hidden neuron count.
    pub hidden_neurons: usize,
    /// Synapse count.
    pub synapses: usize,
    /// `forwardOnly` flag.
    pub forward_only: bool,
    /// Unix seconds when the workspace copy was made.
    pub created_at_unix: u64,
    /// Version of Ockham that made the copy.
    pub ockham_version: String,
}

/// Validate a parsed creature: forward-only, compiles, `creature_validate`,
/// round-trips.
pub fn validate_creature(creature: &CreatureExport) -> Result<(), IncumbentError> {
    if !creature.forward_only {
        return Err(IncumbentError::NotForwardOnly);
    }
    compile_creature(creature).map_err(|e| IncumbentError::Creature(e.to_string()))?;
    validate_no_duplicate_synapses(creature)
        .map_err(|e| IncumbentError::Creature(e.to_string()))?;
    creature_validate(creature, &OCKHAM_VALIDATE_OPTIONS)
        .map_err(|e| IncumbentError::Invalid(e.to_string()))?;
    // serde_json's default float parser can move a weight by one ulp. The
    // contract is structural equality with a 1e-12 relative tolerance on
    // weights/biases — anything larger is a real defect.
    let json = creature_to_json(creature).map_err(|e| IncumbentError::RoundTrip(e.to_string()))?;
    let again = parse_creature_json(&json).map_err(|e| IncumbentError::RoundTrip(e.to_string()))?;
    if let Err(m) = creatures_equivalent(creature, &again) {
        return Err(IncumbentError::RoundTrip(m));
    }
    Ok(())
}

fn close(a: f64, b: f64) -> bool {
    a == b || (a - b).abs() <= 1e-12 * a.abs().max(b.abs())
}

/// Structural equality with a 1e-12 relative tolerance on weights and biases.
pub fn creatures_equivalent(a: &CreatureExport, b: &CreatureExport) -> Result<(), String> {
    if a.input != b.input
        || a.output != b.output
        || a.forward_only != b.forward_only
        || a.semantic_version != b.semantic_version
    {
        return Err("header differs".into());
    }
    if a.neurons.len() != b.neurons.len() || a.synapses.len() != b.synapses.len() {
        return Err("neuron/synapse counts differ".into());
    }
    for (x, y) in a.neurons.iter().zip(&b.neurons) {
        if x.uuid != y.uuid
            || x.neuron_type != y.neuron_type
            || x.squash != y.squash
            || !close(x.bias, y.bias)
        {
            return Err(format!("neuron `{}` differs", x.uuid));
        }
    }
    for (x, y) in a.synapses.iter().zip(&b.synapses) {
        if x.from_uuid != y.from_uuid
            || x.to_uuid != y.to_uuid
            || x.synapse_type != y.synapse_type
            || !close(x.weight, y.weight)
        {
            return Err(format!("synapse `{}`→`{}` differs", x.from_uuid, x.to_uuid));
        }
    }
    Ok(())
}

/// Load and validate the incumbent from `path` without modifying it.
pub fn load_incumbent(path: &Path) -> Result<Incumbent, IncumbentError> {
    let text =
        std::fs::read_to_string(path).map_err(|e| IncumbentError::Io(path.to_path_buf(), e))?;
    let creature =
        parse_creature_json(&text).map_err(|e| IncumbentError::Creature(e.to_string()))?;
    validate_creature(&creature)?;
    Ok(Incumbent {
        source_path: path.to_path_buf(),
        checksum: sha256_hex(text.as_bytes()),
        text,
        creature,
    })
}

impl Incumbent {
    /// Build an incumbent from an in-memory creature (used after acceptance).
    pub fn from_creature(creature: CreatureExport, label: &str) -> Result<Self, IncumbentError> {
        validate_creature(&creature)?;
        let text =
            creature_to_json(&creature).map_err(|e| IncumbentError::Creature(e.to_string()))?;
        Ok(Self {
            source_path: PathBuf::from(label),
            checksum: sha256_hex(text.as_bytes()),
            text,
            creature,
        })
    }

    /// Hidden (non-input, non-output, non-constant) neuron count.
    pub fn hidden_neurons(&self) -> usize {
        self.creature
            .neurons
            .iter()
            .filter(|n| n.neuron_type == "hidden")
            .count()
    }

    /// Short checksum prefix for file names and logs.
    pub fn short_checksum(&self) -> &str {
        &self.checksum[..12]
    }

    /// Copy the incumbent byte-for-byte into `workspace/incumbent.json` and
    /// write `incumbent.meta.json`. Verifies the copy's checksum.
    pub fn write_workspace(&self, workspace: &Path) -> Result<IncumbentMeta, IncumbentError> {
        std::fs::create_dir_all(workspace)
            .map_err(|e| IncumbentError::Io(workspace.to_path_buf(), e))?;
        let copy = workspace.join("incumbent.json");
        std::fs::write(&copy, &self.text).map_err(|e| IncumbentError::Io(copy.clone(), e))?;
        let observed =
            sha256_hex(&std::fs::read(&copy).map_err(|e| IncumbentError::Io(copy.clone(), e))?);
        if observed != self.checksum {
            return Err(IncumbentError::CopyDrift {
                expected: self.checksum.clone(),
                observed,
            });
        }
        let meta = IncumbentMeta {
            checksum: self.checksum.clone(),
            source_path: self.source_path.display().to_string(),
            input: self.creature.input,
            output: self.creature.output,
            neurons: self.creature.neurons.len(),
            hidden_neurons: self.hidden_neurons(),
            synapses: self.creature.synapses.len(),
            forward_only: self.creature.forward_only,
            created_at_unix: now_unix(),
            ockham_version: env!("CARGO_PKG_VERSION").to_string(),
        };
        let meta_path = workspace.join("incumbent.meta.json");
        let json = serde_json::to_string_pretty(&meta)
            .map_err(|e| IncumbentError::Creature(e.to_string()))?;
        std::fs::write(&meta_path, json).map_err(|e| IncumbentError::Io(meta_path, e))?;
        Ok(meta)
    }
}

/// Current unix time in seconds.
pub fn now_unix() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixtures::{identity_creature_json, recurrent_flagged_creature_json};

    #[test]
    fn source_is_untouched_and_copy_matches() {
        let tmp = tempfile::tempdir().unwrap();
        let src = tmp.path().join("creature.json");
        let text = identity_creature_json(2, 1);
        std::fs::write(&src, &text).unwrap();
        let before = std::fs::metadata(&src).unwrap().modified().unwrap();
        let inc = load_incumbent(&src).unwrap();
        let ws = tmp.path().join("ws");
        let meta = inc.write_workspace(&ws).unwrap();
        assert_eq!(std::fs::read_to_string(&src).unwrap(), text);
        assert_eq!(std::fs::metadata(&src).unwrap().modified().unwrap(), before);
        assert_eq!(
            std::fs::read_to_string(ws.join("incumbent.json")).unwrap(),
            text
        );
        assert_eq!(meta.checksum, sha256_hex(text.as_bytes()));
        assert_eq!(meta.input, 2);
        assert!(meta.forward_only);
        assert!(ws.join("incumbent.meta.json").exists());
    }

    #[test]
    fn non_forward_only_is_rejected_before_optimisation() {
        let tmp = tempfile::tempdir().unwrap();
        let src = tmp.path().join("creature.json");
        let text = recurrent_flagged_creature_json(1, 1);
        std::fs::write(&src, &text).unwrap();
        let err = load_incumbent(&src).unwrap_err();
        assert!(matches!(err, IncumbentError::NotForwardOnly), "{err}");
        assert_eq!(std::fs::read_to_string(&src).unwrap(), text);
    }

    #[test]
    fn malformed_creature_is_refused() {
        let tmp = tempfile::tempdir().unwrap();
        let src = tmp.path().join("bad.json");
        std::fs::write(&src, "{\"input\": 1}").unwrap();
        assert!(matches!(
            load_incumbent(&src),
            Err(IncumbentError::Creature(_))
        ));
    }
}
