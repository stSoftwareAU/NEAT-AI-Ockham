use std::collections::BTreeMap;

use neat_core::{CreatureExport, compile_creature, network::CompiledNetwork};
use serde::Serialize;

use super::{Budget, manifest::Fit};

#[derive(Clone)]
pub(super) struct Row {
    pub inputs: Vec<f32>,
    pub targets: Vec<f32>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub(super) struct Ranking {
    pub uuids: Vec<String>,
    pub log_scores: Vec<f64>,
    pub removal_order: Vec<String>,
}

pub(super) fn supported(parent: &CreatureExport) -> Result<(), String> {
    if !parent.forward_only || parent.input == 0 || parent.output == 0 {
        return Err("soft gates require a non-empty forward-only creature".into());
    }
    let hidden = parent.neurons.iter().filter(|n| n.neuron_type == "hidden").count();
    if !(1..=4096).contains(&hidden) {
        return Err("finite-difference experiment supports 1..4096 hidden neurons".into());
    }
    for n in &parent.neurons {
        if n.neuron_type != "constant" && !matches!(n.squash.as_deref(), Some("IDENTITY" | "LOGISTIC" | "TANH")) {
            return Err(format!("unsupported activation {:?} at {}: v1 supports IDENTITY, LOGISTIC and TANH only", n.squash, n.uuid));
        }
    }
    if parent.synapses.iter().any(|s| s.synapse_type.is_some()) {
        return Err("soft gates refuse typed synapses".into());
    }
    Ok(())
}

// A gate modulates every outgoing edge of its hidden source. Core still owns
// input limiting, accumulation, squash and output limiting, including rounding.
fn gated(parent: &CreatureExport, ids: &[String], masks: &[f64], budget: &mut Budget) -> Result<CompiledNetwork, String> {
    if ids.len() != masks.len() || masks.iter().any(|m| !m.is_finite()) {
        return Err("invalid hidden gate vector".into());
    }
    budget.compile()?;
    let gates: BTreeMap<_, _> = ids.iter().zip(masks).collect();
    let mut masked = parent.clone();
    for edge in &mut masked.synapses {
        if let Some(mask) = gates.get(&edge.from_uuid) {
            edge.weight *= **mask;
        }
    }
    compile_creature(&masked).map_err(|e| format!("compile gated creature: {e}"))
}

fn row_loss(net: &mut CompiledNetwork, row: &Row, outputs: usize) -> Result<f64, String> {
    if row.inputs.len() != net.num_inputs() || row.targets.len() != outputs
        || row.inputs.iter().chain(&row.targets).any(|x| !x.is_finite())
    { return Err("non-finite or wrong-width calibration record".into()); }
    let prediction = net.activate(&row.inputs, outputs);
    let error = prediction.iter().zip(&row.targets)
        .map(|(a, b)| (f64::from(*a) - f64::from(*b)).powi(2)).sum::<f64>() / outputs as f64;
    if !error.is_finite() { return Err("non-finite calibration loss".into()); }
    Ok(error)
}

pub(super) fn loss(parent: &CreatureExport, ids: &[String], masks: &[f64], rows: &[Row], budget: &mut Budget) -> Result<f64, String> {
    if rows.is_empty() { return Err("empty calibration sample".into()); }
    budget.records(rows.len())?;
    let mut net = gated(parent, ids, masks, budget)?;
    let sum = rows.iter().try_fold(0.0, |sum, row| row_loss(&mut net, row, parent.output).map(|x| sum + x))?;
    Ok(sum / rows.len() as f64)
}

fn mask_derivatives(parent: &CreatureExport, ids: &[String], masks: &[f64], rows: &[Row], step: f64, budget: &mut Budget) -> Result<Vec<f64>, String> {
    let mut scratch = masks.to_vec();
    let mut derivatives = Vec::with_capacity(ids.len());
    for i in 0..ids.len() {
        scratch[i] = masks[i] + step;
        let plus = loss(parent, ids, &scratch, rows, budget)?;
        scratch[i] = masks[i] - step;
        let minus = loss(parent, ids, &scratch, rows, budget)?;
        derivatives.push((plus - minus) / (2.0 * step));
        scratch[i] = masks[i];
    }
    Ok(derivatives)
}

fn initial_scores(parent: &CreatureExport, ids: &[String], rows: &[Row], step: f64, budget: &mut Budget) -> Result<Vec<f64>, String> {
    let mut masks = vec![1.0; ids.len()];
    let mut scores = vec![0.0; ids.len()];
    for i in 0..ids.len() {
        budget.records(2 * rows.len())?;
        masks[i] = 1.0 + step;
        let mut plus = gated(parent, ids, &masks, budget)?;
        masks[i] = 1.0 - step;
        let mut minus = gated(parent, ids, &masks, budget)?;
        for row in rows {
            let d = (row_loss(&mut plus, row, parent.output)? - row_loss(&mut minus, row, parent.output)?) / (2.0 * step);
            scores[i] += d * d / rows.len() as f64;
        }
        masks[i] = 1.0;
        scores[i] = scores[i].max(1e-12).ln();
    }
    let mut sorted = scores.clone();
    sorted.sort_by(f64::total_cmp);
    let median = (sorted[(sorted.len() - 1) / 2] + sorted[sorted.len() / 2]) / 2.0;
    for score in &mut scores { *score -= median; }
    Ok(scores)
}

// Unit costs are equal because the declared capacity axis is original hidden
// identities; final hard candidates report growth units and support separately.
fn threshold(scores: &[f64], keep: f64) -> f64 {
    let mut sorted = scores.to_vec();
    sorted.sort_by(|a, b| b.total_cmp(a));
    let kept = ((keep * scores.len() as f64).floor() as usize).max(1);
    sorted[kept - 1]
}

pub(super) fn fit(parent: &CreatureExport, rows: &[Row], config: &Fit, seed: u64, budget: &mut Budget) -> Result<Ranking, String> {
    supported(parent)?;
    if rows.is_empty() { return Err("empty calibration sample".into()); }
    let ids = crate::ordering::random_order(parent, seed);
    let mut scores = initial_scores(parent, &ids, rows, config.difference_step, budget)?;
    let mut first = vec![0.0; ids.len()];
    let mut second = vec![0.0; ids.len()];
    for iteration in 1..=config.steps {
        let offset = ((seed % rows.len() as u64) as usize + (iteration - 1) * 32) % rows.len();
        let batch = (0..rows.len().min(32)).map(|i| rows[(offset + i) % rows.len()].clone()).collect::<Vec<_>>();
        let mut gradient = vec![0.0; ids.len()];
        for keep in [0.9, 0.75, 0.5] {
            let cutoff = threshold(&scores, keep);
            let masks = scores.iter().map(|z| 1.0 / (1.0 + (-(z - cutoff) / config.temperature).exp())).collect::<Vec<_>>();
            // Threshold is held fixed: differentiate loss with respect to masks,
            // then apply the sigmoid derivative. No activation derivative guessed.
            let partials = mask_derivatives(parent, &ids, &masks, &batch, config.difference_step, budget)?;
            for i in 0..ids.len() {
                gradient[i] += partials[i] * masks[i] * (1.0 - masks[i]) / config.temperature / 3.0;
            }
        }
        for i in 0..ids.len() {
            first[i] = 0.9 * first[i] + 0.1 * gradient[i];
            second[i] = 0.999 * second[i] + 0.001 * gradient[i].powi(2);
            let m = first[i] / (1.0 - 0.9_f64.powi(iteration as i32));
            let v = second[i] / (1.0 - 0.999_f64.powi(iteration as i32));
            scores[i] -= config.learning_rate * m / (v.sqrt() + 1e-8);
            if !scores[i].is_finite() { return Err("non-finite gate fit".into()); }
        }
    }
    let mut rank: Vec<_> = (0..ids.len()).collect();
    rank.sort_by(|&a, &b| scores[a].total_cmp(&scores[b]).then(a.cmp(&b)));
    let removal_order = rank.into_iter().map(|i| ids[i].clone()).collect();
    Ok(Ranking { uuids: ids, log_scores: scores, removal_order })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixtures::{creature, neuron, synapse, wide_creature};
    use super::super::{Cost, manifest::Limits};

    fn budget() -> Budget {
        Budget::new(&Limits { record_evaluations: 1_000_000, compilations: 100_000,
            rewrites: 1000, io_bytes: 1_000_000, seconds: 60.0 }, &Cost::default()).unwrap()
    }

    #[test]
    fn gate_perturbations_use_core_inference_and_freeze_the_parent() {
        let parent = creature(1, 1, vec![
            neuron("hidden", "h", 0.0, Some("IDENTITY")),
            neuron("output", "output-0", 0.1, Some("IDENTITY")),
        ], vec![synapse("input-0", "h", 0.5), synapse("h", "output-0", 0.8)]);
        let before = neat_core::creature_to_json(&parent).unwrap();
        let mut canonical = neat_core::compile_creature(&parent).unwrap();
        let mut dense = gated(&parent, &["h".into()], &[1.0], &mut budget()).unwrap();
        assert_eq!(dense.activate(&[0.5], 1), canonical.activate(&[0.5], 1));
        let mut half = gated(&parent, &["h".into()], &[0.5], &mut budget()).unwrap();
        assert!((half.activate(&[0.5], 1)[0] - 0.2).abs() < 1e-6);
        let rows = vec![Row { inputs: vec![0.5], targets: vec![0.0] }];
        let derivative = mask_derivatives(&parent, &["h".into()], &[1.0], &rows, 0.001, &mut budget()).unwrap()[0];
        // d((0.2*m + 0.1)^2)/dm at m=1 is 0.12.
        assert!((derivative - 0.12).abs() < 1e-4, "{derivative}");
        assert_eq!(neat_core::creature_to_json(&parent).unwrap(), before);
    }

    #[test]
    fn supported_activations_and_fits_are_reproducible() {
        for squash in ["IDENTITY", "LOGISTIC", "TANH"] {
            let parent = wide_creature(1, 4, squash);
            supported(&parent).unwrap();
            let before = neat_core::creature_to_json(&parent).unwrap();
            let rows = (0..8).map(|i| Row { inputs: vec![i as f32 / 8.0], targets: vec![0.1] }).collect::<Vec<_>>();
            let config = Fit { steps: 2, learning_rate: 0.1, temperature: 0.2, difference_step: 0.001 };
            let a = fit(&parent, &rows, &config, 7, &mut budget()).unwrap();
            let b = fit(&parent, &rows, &config, 7, &mut budget()).unwrap();
            assert_eq!(a.log_scores, b.log_scores);
            assert_eq!(a.removal_order, b.removal_order);
            assert!(a.log_scores.iter().any(|s| s.abs() > 0.01));
            assert_eq!(neat_core::creature_to_json(&parent).unwrap(), before);
        }
    }

    #[test]
    fn discontinuous_aggregate_unknown_and_typed_semantics_are_refused() {
        for squash in ["STEP", "IF", "MINIMUM", "MAXIMUM", "ABSOLUTE", "invented"] {
            let parent = wide_creature(1, 4, squash);
            assert!(supported(&parent).unwrap_err().contains("unsupported activation"));
        }
        let mut parent = wide_creature(1, 4, "TANH");
        parent.synapses[0].synapse_type = Some("condition".into());
        assert!(supported(&parent).unwrap_err().contains("typed synapses"));
        parent.synapses[0].synapse_type = None;
        supported(&parent).unwrap();
    }

    #[test]
    fn budget_exhaustion_refuses_fitting_before_inference() {
        let parent = wide_creature(1, 4, "TANH");
        let mut b = budget();
        b.limits.record_evaluations = 1;
        let rows = vec![Row { inputs: vec![0.2], targets: vec![0.0] }; 2];
        let err = loss(&parent, &["h0".into()], &[1.0], &rows, &mut b).unwrap_err();
        assert!(err.contains("budget exhausted"));
    }
}
