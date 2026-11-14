use std::collections::{BTreeMap, BTreeSet, HashMap};

use neat_ai_ockham::SensitivityIndex;
use neat_ai_ockham::ordering::synapse_order;
use neat_ai_ockham::stats::{ActivationStats, source_value};
use neat_ai_ockham::sweep::synapse_key;
use neat_core::{CreatureExport, SynapseKey, parse_synapse_type};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Arm {
    Global,
    PerNeuron,
    PerDepth,
    Ockham,
}

impl Arm {
    pub const ALL: [Self; 4] = [Self::Global, Self::PerNeuron, Self::PerDepth, Self::Ockham];
}

// Capped equal shares of retained connections, with lexicographic remainder ties.
pub fn quotas(
    capacities: &BTreeMap<String, usize>,
    target: usize,
) -> Result<BTreeMap<String, usize>, String> {
    if target > capacities.values().sum() {
        return Err("target exceeds capacity".into());
    }
    let mut quotas: BTreeMap<_, _> = capacities.keys().map(|k| (k.clone(), 0)).collect();
    let mut left = target;
    while left > 0 {
        let active: Vec<_> = capacities
            .iter()
            .filter(|(k, cap)| quotas[*k] < **cap)
            .map(|(k, _)| k.clone())
            .collect();
        let share = (left / active.len()).max(1);
        for key in active {
            let add = share.min(capacities[&key] - quotas[&key]).min(left);
            *quotas.get_mut(&key).unwrap() += add;
            left -= add;
        }
    }
    Ok(quotas)
}

#[derive(Debug)]
pub struct RankedEdge {
    pub key: SynapseKey,
    pub importance: f64,
    tie: (usize, u8),
}

pub fn ranked_edges(
    creature: &CreatureExport,
    stats: &ActivationStats,
    seed: u64,
) -> Result<Vec<RankedEdge>, String> {
    let sensitivity = SensitivityIndex::new(creature);
    let pair_order: HashMap<_, _> = synapse_order(creature, seed)
        .into_iter()
        .enumerate()
        .map(|(index, key)| (key, index))
        .collect();
    let mut edges = Vec::new();
    for edge in &creature.synapses {
        let source = source_value(creature, stats, &edge.from_uuid)
            .ok_or_else(|| format!("unmeasured source {}", edge.from_uuid))?;
        let variance = stats
            .by_uuid(&edge.from_uuid)
            .or_else(|| stats.input_by_uuid(&edge.from_uuid));
        let scale = match source.kind {
            neat_ai_ockham::stats::SourceValueKind::Constant => 0.0,
            neat_ai_ockham::stats::SourceValueKind::Mean => {
                variance
                    .filter(|s| s.count >= 2 && s.std_dev.is_finite() && s.std_dev >= 0.0)
                    .ok_or_else(|| format!("unmeasured variance {}", edge.from_uuid))?
                    .std_dev
            }
        };
        let downstream = sensitivity
            .importance(&edge.to_uuid)
            .ok_or_else(|| format!("unknown target {}", edge.to_uuid))?;
        let importance = scale * edge.weight.abs() * downstream;
        if !importance.is_finite() {
            return Err("non-finite connection importance".into());
        }
        let role = parse_synapse_type(edge.synapse_type.as_deref());
        edges.push(RankedEdge {
            key: SynapseKey {
                from_uuid: edge.from_uuid.clone(),
                to_uuid: edge.to_uuid.clone(),
                role,
            },
            importance,
            tie: (
                pair_order[&synapse_key(&edge.from_uuid, &edge.to_uuid)],
                role as u8,
            ),
        });
    }
    edges.sort_by(|a, b| {
        a.importance
            .total_cmp(&b.importance)
            .then(a.tie.cmp(&b.tie))
    });
    Ok(edges)
}

pub fn depths(creature: &CreatureExport) -> Result<BTreeMap<String, usize>, String> {
    let mut depths: BTreeMap<_, _> = (0..creature.input)
        .map(|i| (format!("input-{i}"), 0))
        .collect();
    for neuron in &creature.neurons {
        let mut depth = 0;
        for edge in creature
            .synapses
            .iter()
            .filter(|e| e.to_uuid == neuron.uuid)
        {
            let source = depths
                .get(&edge.from_uuid)
                .ok_or("depth requires forward topological order")?;
            depth = depth.max(source + 1);
        }
        depths.insert(neuron.uuid.clone(), depth);
    }
    Ok(depths)
}

#[derive(Debug)]
pub struct Plan {
    arm: Arm,
    target: usize,
    pub depths: BTreeMap<String, usize>,
    pub quotas: BTreeMap<String, usize>,
}

impl Plan {
    pub fn new(parent: &CreatureExport, arm: Arm, target: usize) -> Result<Self, String> {
        if target > parent.synapses.len() {
            return Err("target exceeds parent connections".into());
        }
        let mut plan = Self {
            arm,
            target,
            depths: depths(parent)?,
            quotas: BTreeMap::new(),
        };
        if matches!(arm, Arm::PerNeuron | Arm::PerDepth) {
            plan.quotas = quotas(&plan.counts(parent)?, target)?;
        }
        Ok(plan)
    }

    fn region(&self, target: &str) -> Result<String, String> {
        match self.arm {
            Arm::PerNeuron => {
                if !self.depths.contains_key(target) {
                    return Err("cleanup created an unassigned region".into());
                }
                Ok(target.into())
            }
            Arm::PerDepth => self
                .depths
                .get(target)
                .map(|d| format!("depth-{d:08}"))
                .ok_or_else(|| "cleanup created an unassigned region".into()),
            Arm::Global | Arm::Ockham => Ok("global".into()),
        }
    }

    pub fn counts(&self, creature: &CreatureExport) -> Result<BTreeMap<String, usize>, String> {
        let mut counts = BTreeMap::new();
        for edge in &creature.synapses {
            *counts.entry(self.region(&edge.to_uuid)?).or_insert(0) += 1;
        }
        Ok(counts)
    }

    pub fn eligible(&self, creature: &CreatureExport, key: &SynapseKey) -> bool {
        if self.quotas.is_empty() {
            return true;
        }
        self.region(&key.to_uuid)
            .and_then(|region| {
                Ok(self.counts(creature)?.get(&region).copied().unwrap_or(0)
                    > self.quotas.get(&region).copied().unwrap_or(0))
            })
            .unwrap_or(false)
    }

    pub fn check(&self, before: &CreatureExport, after: &CreatureExport) -> Result<(), String> {
        if after.synapses.len() >= before.synapses.len() {
            return Err("no net connection saving after cleanup".into());
        }
        if after.synapses.len() < self.target {
            return Err("cleanup would overshoot target".into());
        }
        let counts = self.counts(after)?;
        for (region, quota) in &self.quotas {
            if counts.get(region).copied().unwrap_or(0) < *quota {
                return Err(format!("cleanup would undercut quota {region}"));
            }
        }
        Ok(())
    }
}

// Structural support, not a claim that an IF branch fires or a squash is unsaturated.
pub fn output_support(
    creature: &CreatureExport,
) -> Result<BTreeMap<String, BTreeSet<String>>, String> {
    let mut support: BTreeMap<_, _> = (0..creature.input)
        .map(|i| {
            let input = format!("input-{i}");
            (input.clone(), BTreeSet::from([input]))
        })
        .collect();
    for neuron in &creature.neurons {
        let mut sources = BTreeSet::new();
        for edge in creature
            .synapses
            .iter()
            .filter(|e| e.to_uuid == neuron.uuid && e.weight != 0.0)
        {
            sources.extend(
                support
                    .get(&edge.from_uuid)
                    .ok_or("support requires forward topology")?
                    .iter()
                    .cloned(),
            );
        }
        support.insert(neuron.uuid.clone(), sources);
    }
    Ok(creature
        .neurons
        .iter()
        .filter(|n| n.neuron_type == "output")
        .map(|n| (n.uuid.clone(), support[&n.uuid].clone()))
        .collect())
}

pub struct Constraints {
    useful_inputs: BTreeSet<String>,
}

impl Constraints {
    pub fn new(parent: &CreatureExport) -> Result<Self, String> {
        let useful_inputs = output_support(parent)?
            .values()
            .flatten()
            .cloned()
            .collect();
        let constraints = Self { useful_inputs };
        constraints.check(parent)?;
        Ok(constraints)
    }

    pub fn check(&self, creature: &CreatureExport) -> Result<(), String> {
        let support = output_support(creature)?;
        for (output, inputs) in &support {
            if inputs.is_empty() {
                return Err(format!("output lacks input path {output}"));
            }
        }
        let still_useful: BTreeSet<_> = support.values().flatten().cloned().collect();
        for input in &self.useful_inputs {
            if !still_useful.contains(input) {
                return Err(format!("lost useful input {input}"));
            }
        }
        Ok(())
    }
}
