# Stratum-guided pruning experiment (opt-in)

`bench-strata` is an offline benchmark of a pruning-order hypothesis. It is
not a pruning mode: it is opt-in, changes no default, and never accepts or
publishes a creature.

## Purpose

**This is a hypothesis to test, not an established improvement.** Removing
one hidden neuron may hurt a single calibration stratum far more than the
others while barely moving the average. Ordering cuts by their worst-stratum
importance, rather than their aggregate importance, might protect the weakest
stratum. The idea came from the 23 October research shortlist.

The source is arXiv:2610.27812. Only its abstract was available during the
audit, so the full method was not retrieved in this implementation. The
sampler reproduces the formula from the issue instead:

```text
p_i ∝ exp(−max_t I[t,i] / temperature)
```

The statistic is scale-normalised per stratum, so strata with different
effect sizes are comparable: `N[t][i] = I[t][i] / s_t`, not raw `I`.

## Predeclared controls

Arms are fixed and reported in this order: `aggregate`, `worst-stratum`,
`pareto`, `random`.

- **aggregate** is the control. It uses the same sampler on
  `a_i = Σ_t w_t N[t][i]`, with a temperature fitted the same way.
- **worst-stratum** is the hypothesis: the sampler on `m_i = max_t N[t][i]`.
- **pareto** draws candidates alternately from the aggregate and worst-stratum
  samplers, keeps a non-dominated archive and full-scores its members. Each
  sampler uses its own fitted temperature.
- **random** is the negative control: uniform draws, with equal weights and
  no temperature.

## Seeds, strata and sample size

- `seeds` is a non-empty list of distinct integers. Every arm runs under every
  seed, and the arms of one seed share that seed (common random numbers).
- `fitSeed` drives the temperature fit only. It must not appear in `seeds`.
- Calibration has three to five named strata. Validation names must be exactly
  the same set, and no validation directory may resolve to a calibration one.
- Paired intervals use a seeded percentile bootstrap: 2000 resamples, seeded
  with `fitSeed`. They are `null` with fewer than two seeds.

## Compute limits

- `bundlesPerArm` (default 6) is the number of full calibration evaluations
  each arm spends per seed. Pareto full-scores the first
  `min(bundlesPerArm, archive size)` members, so it may spend fewer.
- `paretoCandidates` (default 24) is the number of canonical builds per seed
  for the pareto arm. It must be at least `bundlesPerArm`.
- `bundleSizes` (default `[2, 4]`) cycles across bundles: bundle `j` has size
  `bundleSizes[j % len]`.
- `archiveCap` (default 8) bounds the Pareto archive.
- Scorer calls are counted per phase and per (seed, arm). Importance scores one
  work directory per calibration stratum. The temperature fit scores one
  directory per arm and temperature, per calibration stratum. Evaluation scores
  each full-scored bundle on every calibration stratum. Validation scores one
  directory per validation stratum.

## Success rule

The manifest's `success` block holds the predeclared thresholds:

```json
"success": {"maxFitnessLoss": 0.0, "minRegressionReduction": 1}
```

For each treatment arm (worst-stratum and pareto), compared with aggregate:

- `regressionReduction` is aggregate's minority-stratum regression count minus
  the arm's, counted across seeds.
- `meanFitnessDifference` is the arm's mean validation delta minus aggregate's.
- `met` is true **iff** `regressionReduction >= minRegressionReduction`
  **and** `meanFitnessDifference >= -maxFitnessLoss`.

A minority regression is a validation delta below `-regressionTolerance` on the
minority stratum: the one with the fewest validation records, ties broken by
name. The rule is written as text in each `criteria` entry of the report.

## Calibration and validation

Calibration alone drives every choice: the importance and scales, the sampler
statistics, the temperature fit and selection by full calibration scores.
Validation strata are not read during preparation. They are scored only after
every arm of a seed has made its selection, and no validation score feeds back
into a choice.

## Manifest

The manifest is JSON with camelCase keys, and unknown fields are refused. Its
file size is checked before reading (maximum 1 MiB). Relative paths resolve
against the manifest's directory. This is the full example from the spec:

```json
{
  "parent": "parent.json",
  "calibration": [{"name": "north", "dir": "cal/north"}, {"name": "south", "dir": "cal/south"}, {"name": "rare", "dir": "cal/rare"}],
  "validation":  [{"name": "north", "dir": "val/north"}, {"name": "south", "dir": "val/south"}, {"name": "rare", "dir": "val/rare"}],
  "seeds": [1, 2, 3, 4, 5],
  "fitSeed": 9001,
  "temperatures": [0.25, 0.5, 1.0, 2.0, 4.0],
  "bundleSizes": [2, 4],
  "bundlesPerArm": 6,
  "paretoCandidates": 24,
  "archiveCap": 8,
  "latencyRecords": 256,
  "statsRecords": 4096,
  "minImprovement": 0.0,
  "regressionTolerance": 0.000001,
  "success": {"maxFitnessLoss": 0.0, "minRegressionReduction": 1}
}
```

Required: `parent`, `calibration`, `validation`, `seeds`, `fitSeed`, `success`.

Defaults: `temperatures` `[0.25, 0.5, 1, 2, 4]`; `bundleSizes` `[2, 4]`;
`bundlesPerArm` 6; `paretoCandidates` 24; `archiveCap` 8; `latencyRecords`
256; `statsRecords` 4096 (0 means every record); `minImprovement` 0.0;
`regressionTolerance` 1e-6.

Each of these is refused with an error naming the field:

- calibration count outside 3 to 5; stratum names empty or duplicated;
- validation names not exactly the calibration set; any shared directory;
- `seeds` empty or duplicated; `fitSeed` present in `seeds`;
- `temperatures` empty, or any value not finite and above zero;
- `bundleSizes` empty, or containing 0; `bundlesPerArm` 0;
- `paretoCandidates` below `bundlesPerArm`; `archiveCap` 0; `latencyRecords` 0;
- `minImprovement`, `regressionTolerance` or `maxFitnessLoss` not finite, or
  negative.

## Method

1. **Load.** The parent is read with `incumbent::load_incumbent`, never
   written, and its sha256 is recorded. Every calibration stratum is inspected
   for record counts and identity. The weight `w_t` is the stratum's share of
   calibration records.
2. **Means.** Activation statistics are measured on each calibration stratum,
   over `statsRecords` records (every record when 0). A neuron's mean is the
   record-weighted average of the stratum means.
3. **Importance.** Each hidden neuron is removed alone. `I[t][i]` is the
   baseline score minus the single-cut score on stratum `t`, so a positive
   value means removal hurts. A refused cut is a failure at stage `importance`,
   and the neuron is excluded from every arm.
4. **Scales.** `s_t` is the median of `|I[t][i]|` over measured neurons, or
   1.0 when that median is not above 1e-12. `N = I / s_t`, and `a_i`, `m_i`
   follow as above.
5. **Sampler.** Each (seed, arm) has one SplitMix64 stream, seeded with the
   seed. A bundle is `k` distinct neurons, drawn without replacement. Each
   draw picks `i` with probability proportional to
   `exp(−(x_i − min remaining x) / τ)`, and `k` is clamped to the candidate
   count.
6. **Temperature fit.** On calibration only, with `fitSeed`, each candidate
   temperature for aggregate and worst-stratum draws `bundlesPerArm` bundles,
   builds them with grouped pruning and scores them once per stratum. The
   criterion is the mean over built bundles of `Σ_t w_t score_t`. A
   temperature with no built bundle scores −∞. The highest criterion wins;
   ties go to the earliest temperature in the list.
7. **Evaluation.** For each seed and arm, bundles are built canonically. A
   refused build is a failure at stage `evaluate` and still uses its slot. The
   calibration delta is `Σ_t w_t score_t(bundle) − Σ_t w_t score_t(parent)`.
   A win has a delta above `minImprovement`. The selection is the win with the
   largest delta, and ties go to the lowest index.
   Pareto builds `paretoCandidates` bundles and ranks them on three minimised
   objectives: predicted worst-stratum cost `max_t Σ_{i∈B} N[t][i]`, synapses
   after the cut, and measured latency. It keeps the non-dominated set, capped
   at `archiveCap` by (predicted cost, synapses, latency, members). Predictions
   only set the order; the selection comes from full scores alone.
8. **Validation.** Once every arm of a seed has selected, the baseline and each
   arm's selected creature are written to one subdirectory and scored once per
   validation stratum. An arm with no win reports `noChange`: delta 0, nothing
   removed, no regression.
9. **Latency.** The probe compiles each creature with
   `neat_core::compile_creature`, activates the first `latencyRecords` records
   of the first calibration stratum three times, and reports the median
   milliseconds per record.

## Phases

```mermaid
flowchart TD
    A["load parent + stratum counts"] --> B["calibration statistics"]
    B --> C["importance: single cuts<br/>scored per calibration stratum"]
    C --> D["scales s_t, statistics a_i and m_i"]
    D --> E["temperature fit (fitSeed)<br/>calibration scores only"]
    E --> F["per seed: aggregate, worst-stratum,<br/>pareto, random"]
    F --> G["select the best calibration win per arm"]
    G --> H["validation: each selection scored<br/>per validation stratum"]
    H --> I["report JSON, written atomically"]
```

## Report

The report is camelCase JSON. Its top-level fields are:

- `schema` is `ockham-strata-bench/1`, and `evidence` states that the report is
  data, not a claim.
- `method` holds a short string for each step, including the source note.
- `versions`, `manifest` and `fitted` record the identities, the effective
  manifest with defaults applied, the scales, the chosen temperatures and every
  fit criterion.
- `coverage` and `costs` record the neurons measured and refused, and the
  preparation time and scorer calls.
- `runs` holds one entry per seed and arm: bundles scored, refusals, wins,
  selection, calibration and validation deltas, per-stratum validation deltas,
  minority regression, removals, latency, timings, scorer calls and
  `winsPerHour`. `archiveSize` is set for pareto only.
- `arms` gives, per arm, a distribution of each metric as `Dist` (`n`, `mean`,
  `median`, `min`, `max`, `std`, all `null` when `n` is 0) plus counts.
- `paired` gives per-seed differences, means and 95% intervals for each
  treatment against its control.
- `criteria` gives the success result for each treatment arm.
- `failures` lists each refusal with its `stage`, `seed`, `arm`, `uuid` and
  `detail`.

## Cost accounting

- `prepMs` is the wall time of steps 1 to 6, which every arm shares.
- `winsPerHour` for a (seed, arm) is `wins / hours`, where
  `hours = (armMs + prepMs / (arms × seeds)) / 3.6e6`. Preparation is amortised
  over the four arms. It is `null` when `hours` is zero.
- Wall-clock figures are indicative only. They vary by host and are never
  asserted in tests.

## What the bench never does

- It never accepts, publishes or journals a creature. It never writes the
  learnings cache or the statistics cache.
- It never changes `OckhamConfig` or any default.
- It never modifies the parent, which is loaded immutably.
- It writes creatures only inside its work directory, `<REPORT>.work`. It
  refuses to run if that directory already exists, and removes it at the end
  of the run, on success and on error. A removal failure is reported, never
  ignored.
- The report is written atomically: a temporary file beside it, then a rename.

## Status

No representative run has been made. The fixtures do not demonstrate a
benefit, and the report's `evidence` field says so. Any optimisation claim
needs representative corpora, which this repository does not hold.

Negative findings are recorded in this section. If a representative run shows
that worst-stratum ordering does not earn its keep, the code is removed rather
than kept as an option.
