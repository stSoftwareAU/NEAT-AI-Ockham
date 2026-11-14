//! Generate public wiring data, not evidence of a pruning benefit.
//!
//! `cargo run --example soft_gate_smoke -- /tmp/new-smoke-directory`

use std::path::PathBuf;

use neat_ai_ockham::corpus::write_bin_file;
use neat_ai_ockham::fixtures::wide_creature;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let root = PathBuf::from(std::env::args_os().nth(1).ok_or("expected a new output directory")?);
    std::fs::create_dir(&root)?;
    let mut parent = wide_creature(2, 20, "TANH");
    for (i, neuron) in parent.neurons.iter_mut().enumerate() {
        if neuron.neuron_type == "hidden" {
            neuron.bias = (i as f64 - 10.0) * 0.01;
        }
    }
    for (i, edge) in parent.synapses.iter_mut().enumerate() {
        edge.weight *= 0.5 + (i as f64 * 0.7).sin().abs();
    }
    std::fs::write(root.join("parent.json"), neat_core::creature_to_json(&parent)?)?;
    let data = root.join("data");
    std::fs::create_dir(&data)?;
    let rows = (0..512).map(|i| {
        let x = i as f32 / 255.5 - 1.0;
        let y = (i * 37 % 512) as f32 / 255.5 - 1.0;
        (vec![x, y], vec![0.15 * x + 0.03 * (3.0 * y).sin()])
    }).collect::<Vec<_>>();
    write_bin_file(&data.join("0.bin"), &rows)?;
    println!("{}", root.display());
    Ok(())
}
