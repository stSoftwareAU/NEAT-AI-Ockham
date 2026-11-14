//! Opt-in, versioned soft-gate research protocol. Ordinary pruning is unchanged.

mod family;
mod fit;
mod manifest;
mod study;

use std::time::Instant;

use serde::Serialize;

pub use manifest::prepare_manifest;
pub use study::run_manifest;

#[derive(Debug, Clone, Default, Serialize)]
#[serde(rename_all = "camelCase")]
struct Cost {
    record_evaluations: u64,
    compilations: u64,
    rewrites: u64,
    io_bytes: u64,
    preparation_seconds: f64,
    elapsed_seconds: f64,
}

struct Budget {
    limits: manifest::Limits,
    cost: Cost,
    started: Instant,
}

impl Budget {
    fn new(limits: &manifest::Limits, common: &Cost) -> Result<Self, String> {
        let budget = Self {
            limits: limits.clone(),
            cost: common.clone(),
            started: Instant::now(),
        };
        budget.check()?;
        Ok(budget)
    }

    fn check(&self) -> Result<(), String> {
        if self.cost.record_evaluations > self.limits.record_evaluations
            || self.cost.rewrites > self.limits.rewrites
            || self.cost.compilations > self.limits.compilations
            || self.cost.io_bytes > self.limits.io_bytes
            || self.started.elapsed().as_secs_f64() + self.cost.preparation_seconds
                > self.limits.seconds
        {
            return Err("family budget exhausted (includes preparation and final evaluation)".into());
        }
        Ok(())
    }

    fn records(&mut self, n: usize) -> Result<(), String> {
        self.cost.record_evaluations = self.cost.record_evaluations.saturating_add(n as u64);
        self.check()
    }

    fn compile(&mut self) -> Result<(), String> {
        self.cost.compilations += 1;
        self.check()
    }

    fn rewrite(&mut self) -> Result<(), String> {
        self.cost.rewrites += 1;
        self.check()
    }

    fn io(&mut self, n: u64) -> Result<(), String> {
        self.cost.io_bytes = self.cost.io_bytes.saturating_add(n);
        self.check()
    }

    fn finish(mut self) -> Result<Cost, String> {
        self.check()?;
        self.cost.elapsed_seconds = self.started.elapsed().as_secs_f64()
            + self.cost.preparation_seconds;
        Ok(self.cost)
    }
}
