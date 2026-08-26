//! Creature JSON tags for GRQ-sampler check-in (production).
//!
//! `neat_core::CreatureExport` does not round-trip `tags`. Forests and Lamarck
//! keep them in a sidecar [`CreatureMeta`] and re-attach on write. Ockham must
//! do the same: a better score that lost discovery / intelligent-design
//! provenance is refused by GRQ's check-in guard.
//!
//! Deliberately dropped on serialise: creature-level `uuid` and `memetic`
//! (the structure changed, so those identities would be a lie).

use std::collections::{BTreeMap, HashSet};

use neat_core::{CreatureExport, creature_to_json};
use serde_json::{Map, Value};

/// One `{ name, value }` tag.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Tag {
    /// Tag key (`score`, `error`, `ockham`, …).
    pub name: String,
    /// Tag value (always a string in the export format).
    pub value: String,
}

impl Tag {
    /// Convenience constructor.
    pub fn new(name: impl Into<String>, value: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            value: value.into(),
        }
    }
}

fn parse_tags(v: Option<&Value>) -> Vec<Tag> {
    let mut out = Vec::new();
    if let Some(Value::Array(tags)) = v {
        for t in tags {
            if let Value::Object(o) = t
                && let Some(Value::String(name)) = o.get("name")
            {
                let value = match o.get("value") {
                    Some(Value::String(s)) => s.clone(),
                    Some(v) => v.to_string(),
                    None => String::new(),
                };
                out.push(Tag {
                    name: name.clone(),
                    value,
                });
            }
        }
    }
    out
}

fn tags_value(tags: &[Tag]) -> Value {
    Value::Array(
        tags.iter()
            .map(|t| serde_json::json!({"name": t.name, "value": t.value}))
            .collect(),
    )
}

/// Metadata carried alongside a creature.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct CreatureMeta {
    /// Creature-level tags in original order.
    pub tags: Vec<Tag>,
    /// Per-neuron tags keyed by neuron uuid.
    pub neuron_tags: BTreeMap<String, Vec<Tag>>,
}

impl CreatureMeta {
    /// Parse creature-level and per-neuron tags from raw creature JSON.
    pub fn from_json(text: &str) -> Self {
        let mut out = Self::default();
        if let Ok(Value::Object(map)) = serde_json::from_str::<Value>(text) {
            out.tags = parse_tags(map.get("tags"));
            if let Some(Value::Array(neurons)) = map.get("neurons") {
                for n in neurons {
                    if let Value::Object(o) = n
                        && let Some(Value::String(uuid)) = o.get("uuid")
                    {
                        let tags = parse_tags(o.get("tags"));
                        if !tags.is_empty() {
                            out.neuron_tags.insert(uuid.clone(), tags);
                        }
                    }
                }
            }
        }
        out
    }

    /// Replace or append a creature-level tag.
    pub fn upsert(&mut self, name: &str, value: String) {
        if let Some(t) = self.tags.iter_mut().find(|t| t.name == name) {
            t.value = value;
        } else {
            self.tags.push(Tag {
                name: name.into(),
                value,
            });
        }
    }

    /// Drop per-neuron tags whose uuid is no longer in `creature`.
    pub fn retain_neurons(&mut self, creature: &CreatureExport) {
        let keep: HashSet<&str> = creature.neurons.iter().map(|n| n.uuid.as_str()).collect();
        self.neuron_tags
            .retain(|uuid, _| keep.contains(uuid.as_str()));
    }

    /// Update score/error and stamp a run-level Ockham summary for check-in.
    pub fn stamp_acceptance(&mut self, progress: &OckhamProgress<'_>) {
        self.upsert("score", format!("{}", progress.score));
        self.upsert("error", format!("{}", progress.error));
        self.upsert("ockham", ockham_progress_message(progress));
    }

    /// Serialise `creature` with creature-level and remaining per-neuron tags.
    ///
    /// Neuron tags whose uuid is not in the creature are dropped silently.
    /// Creature `uuid` / `memetic` are never written.
    pub fn serialize_with(
        &self,
        creature: &CreatureExport,
        pretty: bool,
    ) -> Result<String, String> {
        let text = creature_to_json(creature).map_err(|e| e.to_string())?;
        let mut value: Map<String, Value> =
            serde_json::from_str(&text).map_err(|e| e.to_string())?;
        if !self.tags.is_empty() {
            value.insert("tags".into(), tags_value(&self.tags));
        }
        if !self.neuron_tags.is_empty()
            && let Some(Value::Array(neurons)) = value.get_mut("neurons")
        {
            for n in neurons.iter_mut() {
                if let Value::Object(o) = n
                    && let Some(Value::String(uuid)) = o.get("uuid")
                    && let Some(tags) = self.neuron_tags.get(uuid)
                {
                    o.insert("tags".into(), tags_value(tags));
                }
            }
        }
        let v = Value::Object(value);
        if pretty {
            serde_json::to_string_pretty(&v)
        } else {
            serde_json::to_string(&v)
        }
        .map_err(|e| e.to_string())
    }
}

/// Fields for the run-level Ockham check-in summary.
#[derive(Debug, Clone, Copy)]
pub struct OckhamProgress<'a> {
    /// Authoritative local accepts so far.
    pub accepts: u64,
    /// Sweep batches attempted.
    pub experiments: u64,
    /// Opening-parent score.
    pub opening: f64,
    /// Current authoritative score.
    pub score: f64,
    /// Current authoritative error.
    pub error: f64,
    /// Last accepted strategy label (`individual`, `bundle`, …).
    pub last: &'a str,
    /// `replay-bundle`, `replay`, or `search` — flows through the population as the `ockham` tag.
    pub origin: &'a str,
    /// Hidden UUIDs removed in this accept.
    pub cuts: usize,
}

/// GRQ-sampler skim line. Becomes the sampler commit subject.
pub fn ockham_progress_message(progress: &OckhamProgress<'_>) -> String {
    let delta = if progress.score > progress.opening {
        format!(" (+{:.2e})", progress.score - progress.opening)
    } else {
        String::new()
    };
    match progress.origin {
        "search" => format!(
            "🪒 Ockham · search {} · {} accepts / {} batches · score: {:.6}{delta}",
            progress.last, progress.accepts, progress.experiments, progress.score
        ),
        other => format!(
            "🪒 Ockham · {other} · {} cuts · score: {:.6}{delta}",
            progress.cuts, progress.score
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixtures::{creature, neuron, synapse};

    const SRC: &str = r#"{
        "input":1,"output":1,"uuid":"abc","memetic":{"x":1},
        "tags":[{"name":"score","value":"0.5"},{"name":"name","value":"Frank"}],
        "neurons":[
            {"type":"hidden","uuid":"h1","bias":0,"squash":"IDENTITY",
             "tags":[{"name":"discovered","value":"ReLU6"}]},
            {"type":"output","uuid":"output-0","bias":0,"squash":"IDENTITY"}
        ],
        "synapses":[{"fromUUID":"input-0","toUUID":"h1","weight":1},
                    {"fromUUID":"h1","toUUID":"output-0","weight":1}]
    }"#;

    #[test]
    fn creature_and_neuron_tags_survive_uuid_and_memetic_do_not() {
        let mut meta = CreatureMeta::from_json(SRC);
        assert_eq!(meta.tags.len(), 2);
        assert_eq!(meta.neuron_tags["h1"].len(), 1);
        let pruned = creature(
            1,
            1,
            vec![neuron("output", "output-0", 0.0, Some("IDENTITY"))],
            vec![synapse("input-0", "output-0", 1.0)],
        );
        meta.retain_neurons(&pruned);
        meta.stamp_acceptance(&OckhamProgress {
            accepts: 1,
            experiments: 3,
            opening: 0.5,
            score: 0.500002,
            error: 0.499998,
            last: "collapse",
            origin: "search",
            cuts: 1,
        });
        let out = meta.serialize_with(&pruned, true).unwrap();
        let v: Value = serde_json::from_str(&out).unwrap();
        assert!(v.get("uuid").is_none());
        assert!(v.get("memetic").is_none());
        let tags = v["tags"].as_array().unwrap();
        let by_name: BTreeMap<&str, &str> = tags
            .iter()
            .filter_map(|t| Some((t["name"].as_str()?, t["value"].as_str()?)))
            .collect();
        assert_eq!(by_name["score"], "0.500002");
        assert_eq!(by_name["name"], "Frank");
        assert!(by_name["ockham"].starts_with("🪒 Ockham"));
        assert!(by_name["ockham"].contains("search"));
        assert!(by_name["ockham"].contains("(+"));
        assert!(v["neurons"][0].get("tags").is_none());
    }

    #[test]
    fn remaining_neuron_tags_are_reattached() {
        let meta = CreatureMeta::from_json(SRC);
        let keep = creature(
            1,
            1,
            vec![
                neuron("hidden", "h1", 0.0, Some("IDENTITY")),
                neuron("output", "output-0", 0.0, Some("IDENTITY")),
            ],
            vec![
                synapse("input-0", "h1", 1.0),
                synapse("h1", "output-0", 1.0),
            ],
        );
        let out = meta.serialize_with(&keep, false).unwrap();
        let v: Value = serde_json::from_str(&out).unwrap();
        assert_eq!(v["neurons"][0]["tags"][0]["name"], "discovered");
    }

    #[test]
    fn replay_bundle_tag_names_the_origin_and_cut_count() {
        let msg = ockham_progress_message(&OckhamProgress {
            accepts: 1,
            experiments: 1,
            opening: 0.39,
            score: 0.39001,
            error: 0.61,
            last: "bundle",
            origin: "replay-bundle",
            cuts: 12,
        });
        assert!(msg.contains("replay-bundle"));
        assert!(msg.contains("12 cuts"));
        assert!(msg.contains("(+"));
    }
}
