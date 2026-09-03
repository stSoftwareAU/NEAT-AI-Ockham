//! Creature JSON tags for GRQ-sampler check-in (production).
//!
//! Tags are **informational metadata only** (Issue #87): they describe a neuron,
//! they never change what Ockham does with it. `neat_core::CreatureExport` does
//! not round-trip `tags`, so Forests and Lamarck keep them in a sidecar
//! [`CreatureMeta`] and re-attach on write. Ockham does the same, so a surviving
//! neuron keeps its tags byte-for-byte.
//!
//! Deliberately dropped on serialise: creature-level `uuid` and `memetic`
//! (the structure changed, so those identities would be a lie).

use std::collections::{BTreeMap, HashSet};

use neat_core::{CreatureExport, creature_to_json};
use serde_json::{Map, Value};

use crate::coverage::Coverage;

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
    /// Screening coverage over the incumbent, when a screen store is configured.
    ///
    /// `None` without `--learnings-dir`: there is no coverage state, and
    /// `0/0 (0.0%)` would be a lie rather than a measurement.
    pub coverage: Option<Coverage>,
}

/// GRQ-sampler skim line. Becomes the sampler commit subject.
///
/// The coverage clause uses the compact `checked X/Y (Z%)` form — this is a
/// commit subject, so [`Coverage::summary`]'s fuller wording belongs in the
/// commit description instead.
pub fn ockham_progress_message(progress: &OckhamProgress<'_>) -> String {
    let delta = if progress.score > progress.opening {
        format!(" (+{:.2e})", progress.score - progress.opening)
    } else {
        String::new()
    };
    let coverage = progress.coverage.map_or_else(String::new, |c| {
        format!(
            " · checked {}/{} ({:.1}%)",
            c.checked,
            c.checkable,
            c.percent()
        )
    });
    match progress.origin {
        "search" => format!(
            "🪒 Ockham · search {} · {} accepts / {} batches · score: {:.6}{delta}{coverage}",
            progress.last, progress.accepts, progress.experiments, progress.score
        ),
        other => format!(
            "🪒 Ockham · {other} · {} cuts · score: {:.6}{delta}{coverage}",
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
            coverage: None,
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

    /// #63 lets Ockham cut tagged neurons, so the cut uuid's tags leave with
    /// it — a `tags` entry for a neuron that no longer exists is a lie. The
    /// survivors keep theirs verbatim, which is the whole of the tag contract
    /// (Issue #87).
    #[test]
    fn a_cut_tagged_neuron_leaves_no_tags_entry_and_the_survivors_keep_theirs() {
        const TWO_TAGGED: &str = r#"{
            "input":1,"output":1,
            "neurons":[
                {"type":"hidden","uuid":"h1","bias":0,"squash":"IDENTITY",
                 "tags":[{"name":"discovered","value":"ReLU6"}]},
                {"type":"hidden","uuid":"h2","bias":0,"squash":"IDENTITY",
                 "tags":[{"name":"discovered","value":"MEAN"},{"name":"design","value":"grq"}]},
                {"type":"output","uuid":"output-0","bias":0,"squash":"IDENTITY"}
            ],
            "synapses":[{"fromUUID":"input-0","toUUID":"h1","weight":1},
                        {"fromUUID":"input-0","toUUID":"h2","weight":1},
                        {"fromUUID":"h1","toUUID":"output-0","weight":1},
                        {"fromUUID":"h2","toUUID":"output-0","weight":1}]
        }"#;
        let mut meta = CreatureMeta::from_json(TWO_TAGGED);
        let before = meta.neuron_tags["h2"].clone();
        let pruned = creature(
            1,
            1,
            vec![
                neuron("hidden", "h2", 0.0, Some("IDENTITY")),
                neuron("output", "output-0", 0.0, Some("IDENTITY")),
            ],
            vec![
                synapse("input-0", "h2", 1.0),
                synapse("h2", "output-0", 1.0),
            ],
        );
        meta.retain_neurons(&pruned);
        assert!(
            !meta.neuron_tags.contains_key("h1"),
            "the cut uuid must leave the sidecar"
        );
        let v: Value = serde_json::from_str(&meta.serialize_with(&pruned, true).unwrap()).unwrap();
        let neurons = v["neurons"].as_array().unwrap();
        assert!(
            !neurons.iter().any(|n| n["uuid"] == "h1"),
            "the cut neuron itself is gone: {v}"
        );
        assert!(
            !serde_json::to_string(&v).unwrap().contains("ReLU6"),
            "no tags may be claimed for a neuron that no longer exists: {v}"
        );
        let survivor = neurons.iter().find(|n| n["uuid"] == "h2").unwrap();
        assert_eq!(
            survivor["tags"],
            serde_json::json!([
                {"name":"discovered","value":"MEAN"},
                {"name":"design","value":"grq"}
            ]),
            "the surviving tags must be written back byte-for-byte"
        );
        assert_eq!(meta.neuron_tags["h2"], before);
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
            coverage: None,
        });
        assert!(msg.contains("replay-bundle"));
        assert!(msg.contains("12 cuts"));
        assert!(msg.contains("(+"));
    }

    fn progress(origin: &'static str, coverage: Option<Coverage>) -> OckhamProgress<'static> {
        OckhamProgress {
            accepts: 3,
            experiments: 41,
            opening: 0.512225,
            score: 0.512345,
            error: 0.487655,
            last: "bundle",
            origin,
            cuts: 8,
            coverage,
        }
    }

    fn some_coverage() -> Option<Coverage> {
        Some(Coverage {
            hidden: 5013,
            tagged: 42,
            checkable: 5013,
            checked: 1204,
            blocked: 0,
            cut: 8,
        })
    }

    #[test]
    fn absent_coverage_leaves_the_search_message_exactly_as_it_was() {
        assert_eq!(
            ockham_progress_message(&progress("search", None)),
            "🪒 Ockham · search bundle · 3 accepts / 41 batches · score: 0.512345 (+1.20e-4)"
        );
    }

    #[test]
    fn absent_coverage_leaves_the_replay_message_exactly_as_it_was() {
        assert_eq!(
            ockham_progress_message(&progress("replay-bundle", None)),
            "🪒 Ockham · replay-bundle · 8 cuts · score: 0.512345 (+1.20e-4)"
        );
    }

    #[test]
    fn search_carries_the_compact_coverage_clause() {
        assert_eq!(
            ockham_progress_message(&progress("search", some_coverage())),
            "🪒 Ockham · search bundle · 3 accepts / 41 batches · score: 0.512345 (+1.20e-4) · checked 1204/5013 (24.0%)"
        );
    }

    #[test]
    fn replay_carries_the_same_compact_coverage_clause() {
        let msg = ockham_progress_message(&progress("replay", some_coverage()));
        assert_eq!(
            msg,
            "🪒 Ockham · replay · 8 cuts · score: 0.512345 (+1.20e-4) · checked 1204/5013 (24.0%)"
        );
        assert!(
            msg.starts_with("🪒 Ockham"),
            "GRQ's razor-prefix check must keep matching"
        );
        assert!(msg.contains("score: 0.512345"));
        assert!(msg.contains("(+1.20e-4)"));
    }

    /// The compact clause, not `Coverage::summary` — the verbose wording is for
    /// the commit description, not this subject line.
    #[test]
    fn the_clause_is_compact_rather_than_the_full_summary() {
        let msg = ockham_progress_message(&progress("search", some_coverage()));
        assert!(!msg.contains("hidden"), "{msg}");
        assert!(!msg.contains("tagged"), "{msg}");
    }

    /// An all-tagged creature has a real denominator since Issue #74, so the
    /// clause reports honest progress through it rather than `0/0`.
    #[test]
    fn an_all_tagged_creature_still_renders_an_honest_clause() {
        let msg = ockham_progress_message(&progress(
            "search",
            Some(Coverage {
                hidden: 2,
                tagged: 2,
                checkable: 2,
                checked: 1,
                blocked: 0,
                cut: 0,
            }),
        ));
        assert!(msg.contains("checked 1/2 (50.0%)"), "{msg}");
    }

    /// The only zero denominator left: a creature with no hidden neurons.
    #[test]
    fn nothing_checkable_still_renders_an_honest_clause_when_coverage_exists() {
        let msg = ockham_progress_message(&progress(
            "search",
            Some(Coverage {
                hidden: 0,
                tagged: 0,
                checkable: 0,
                checked: 0,
                blocked: 0,
                cut: 0,
            }),
        ));
        assert!(msg.contains("checked 0/0 (0.0%)"), "{msg}");
    }

    #[test]
    fn stamped_acceptance_puts_coverage_in_the_ockham_tag() {
        let mut meta = CreatureMeta::default();
        meta.stamp_acceptance(&progress("search", some_coverage()));
        let ockham = meta
            .tags
            .iter()
            .find(|t| t.name == "ockham")
            .expect("ockham tag");
        assert!(ockham.value.contains("checked 1204/5013 (24.0%)"));
    }
}
