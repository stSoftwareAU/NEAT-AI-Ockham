# Stratum-guided pruning pilot v1: inconclusive

This is a synthetic pilot run with the real scorer, not evidence of a benefit
on evolved creatures. The fixture is application-agnostic and deliberately
simple, and no representative creature or corpus was available. It claims no
benefit, and it claims no representative negative result.

The [frozen manifest](manifest.json) and [parent creature](parent.json) are the
inputs. The [raw report](report.json) is the unedited output of
`bench-strata`. The method and its controls are described in
[the experiment protocol](../../strata-experiment.md).

## Versions and pins

- Ockham crate 0.1.93, built from commit
  `a112bc1db107dc4fb17feeec1865c4732b2b7017`. The example that generates the
  inputs, `ockham/examples/strata_pilot.rs`, is added on the issue #263
  branch after that commit. The `bench-strata` source is unchanged since
  `a112bc1`.
- `rust_scorer` built with `cargo build --release --locked -p rust_scorer` from
  NEAT-AI-scorer revision `742c5df30fc98ece3afe784be1aca18f925b47b3`, with
  sha256 `d057a3349da1c28e1319d7e235491597517581b17fda21ea1cf8f0dd55ffbbf7`.
  The report's `versions.scorerIdentity` is `sha256:d057a3349da1c28e`.
- Both binaries use neat-core `v0.23.4`
  (`e86c342e72e85172b2f9e557329aca3b15bf5229`).

## Fixture

- Four inputs, one output and ten hidden TANH neurons in the parent.
- Specialists `h0` to `h3`: `hk` reads input `k` at weight 1.5. `h3` is the
  specialist of the minority stratum `rare`.
- Shared neurons `s0` and `s1`.
- Dead wood `d0` to `d3`, each with an outgoing weight of 1e-6 to 4e-6.
- Targets are the parent's own outputs with every |weight| below 1e-4 zeroed.
- Strata `alpha`, `beta` and `gamma` each have 192 calibration and 192
  validation records. The minority stratum `rare` has 48 of each.
- In stratum `k`, input `k` is at full scale and the other inputs are scaled by
  0.1.
- Seeds 1 to 10, `fitSeed` 9001, temperature grid
  `[0.25, 0.5, 1.0, 2.0, 4.0]`, `bundlesPerArm` 6, `paretoCandidates` 24,
  `bundleSizes` `[2, 4]`, `statsRecords` 0, `latencyRecords` 48, and success
  criteria `maxFitnessLoss` 0 and `minRegressionReduction` 1.

## Observed coverage

- 10 of 10 hidden neurons were measured, with 0 refused single cuts and
  0 failures.
- The fitted temperature was 0.25 for both aggregate and worst-stratum, which is
  the lowest value in the grid.

## Results

| Arm | Seeds with a win | Selected bundle | Mean validation delta | Minority regressions | Hidden removed |
| --- | --- | --- | --- | --- | --- |
| aggregate | 10 of 10 | {d0, d1, d2, d3} | 5.947e-7 | 0 | 4 on every seed |
| worst-stratum | 10 of 10 | {d0, d1, d2, d3} | 5.947e-7 | 0 | 4 on every seed |
| pareto | 10 of 10 | {d0, d1, d2, d3} | 5.947e-7 | 0 | 4 on every seed |
| random | 4 of 10 (seeds 2, 7, 9, 10) | two dead-wood neurons where it won, nothing elsewhere | 1.192e-7 | 0 | 0.8 |

Every informed arm won on six of its six scored bundles on each seed. The
aggregate, worst-stratum and pareto arms selected the same bundle on every
seed, with a validation delta of 5.947e-7 on every seed. Each removed 4 hidden
neurons and 20 synapses. Pareto's archive held one entry on nine seeds and
three on seed 9 (where `{d2, d3}` and `{d1, d3}` were also archived); its
selection was still `{d0, d1, d2, d3}`. Archive size depends on wall-clock
latency, so it varied between runs, but the selection did not.

The random arm had a mean of 0.5 wins per seed and removed a mean of 4.0
synapses per seed. Every arm had 0 minority regressions and 0 regressed strata.

## Paired differences and criteria

- Worst-stratum minus aggregate, and pareto minus aggregate: every per-seed
  difference is 0 for validation delta, minority regression and hidden removed.
  The mean is 0 and the 95% interval is `[0, 0]`.
- Worst-stratum minus random, validation delta: mean 4.755e-7, 95% interval
  `[3.862e-7, 5.649e-7]`.
- Worst-stratum minus random, hidden removed: mean 3.2, 95% interval
  `[2.6, 3.8]`.
- Pareto minus random gives the same figures.
- The criteria for worst-stratum and pareto are both `met: false`. Each has
  `regressionReduction` 0, against a required minimum of 1, and
  `meanFitnessDifference` 0.

## Interpretation

The result is inconclusive. The fixture's dead wood is cheap on every stratum,
so at temperature 0.25 the informed arms' selections converge on the same four
neurons. The aggregate control never walked into a minority regression, so
there was nothing for worst-stratum guidance to avoid. The arms are tied, and
the success criterion is not met for either treatment.

This is synthetic evidence only. No representative creature or corpus was
available, and the coverage of representative pairs is zero. No benefit is
claimed, and no representative negative result is claimed.

## Costs

- 208 scorer invocations, from `report.json`: 8 preparation calls
  (`costs.prepScorerCalls`), 160 per-arm calls (4 per seed and arm, summed over
  `runs[].scorerCalls`) and 40 validation calls (`costs.validationScorerCalls`).
- `prepMs` was 35.2. Wall-clock times were taken on a shared host and are
  indicative only. They do not establish a speed-up.

## Input checksums

| File | sha256 |
| --- | --- |
| `parent.json` | `f6ddc1d650bbc1a2e9d5180bdc8446117c0b06737035d62748041e3f3c3f2013` |
| `manifest.json` | `054522f1447fea20ded7868285754d3a1487b5f378f224bf4f759a7092407259` |
| `cal/alpha/records.bin` | `dc1b60e0b43372c1425c410b69e342e649a441deed5784a55be154cfaa60e616` |
| `cal/beta/records.bin` | `4c7403101ffdb388dfd53b33b288a9e3a34d0694b95b6c25a4b97d383c255b02` |
| `cal/gamma/records.bin` | `bc0967497c2a0ee468aac2a8c302e0f4e5e25b2412555cb23f75e8df9e0980c8` |
| `cal/rare/records.bin` | `b071b229ed872ecfcf3fe85da8114db944db0247525573b63512b29db57083b7` |
| `val/alpha/records.bin` | `5f9157127628cdf2a91d2b4d1a6e9912fc9bf0d087c6b8c7478fff2a997a069a` |
| `val/beta/records.bin` | `d072205ab82ad8b48d93000aec62cf13487c78c8bef15d6b97eb59cb3fa79c2b` |
| `val/gamma/records.bin` | `119ff741ad4d63d1537044a61c59165ea2ce61b991e2fde01c566008622a7099` |
| `val/rare/records.bin` | `f37659fb95b57cbbdccd55fae68e2caaeeb625263b8d1591da485fbd75f1dfb6` |
| `report.json` (raw output) | `72a49801b90ff8e0f63de912cfc72a65fb47700d63d52349c5fdf0c2b150f369` |

The `.bin` files are not committed. The example regenerates them byte for byte.

## Reproduce

Build the scorer at the pinned revision, in a NEAT-AI-scorer checkout:

```sh
git checkout 742c5df30fc98ece3afe784be1aca18f925b47b3
cargo build --release --locked -p rust_scorer
```

The committed run was made from a working directory holding the generated
`pilot-v1/` directory (the report's `manifest` echo records
`pilot-v1/parent.json` and `pilot-v1/cal/...`), with `--out report-v1.json`,
later committed as `report.json`. From the root of an Ockham checkout, generate
the inputs in `runs/`, which is gitignored. `pilot-v1` must not already exist,
because the example refuses it:

```sh
mkdir -p runs && cd runs
cargo run --release --manifest-path ../Cargo.toml --example strata_pilot -- pilot-v1
```

The generated `manifest.json` and `parent.json` are byte-identical to the
committed copies, as the checksums above show. The manifest's relative paths
resolve inside the generated directory. The test
`two_fresh_directories_are_byte_identical_and_load` in
`ockham/examples/strata_pilot.rs` checks the committed copies against fresh
generator output.

Run the benchmark, replacing `/path/to/rust_scorer` with the built binary's
path:

```sh
neat_ai_ockham bench-strata pilot-v1/manifest.json \
  --out report-v1.json --scorer /path/to/rust_scorer \
  --scorer-arg=--gpu --scorer-arg=off --scorer-arg=--cost --scorer-arg=MSE
```

The run exits 0. The committed `report.json` is the unedited `report-v1.json`
of the original run, renamed. A fresh rerun, with the same scorer and
arguments, reproduces the selections, scores, paired results and criteria. It
does not reproduce archive membership, because Pareto's archive depends on
latency through `latencyTolerance`. Wall-clock fields (`prepMs`, `armMs`,
latency and `winsPerHour`) also vary, so the file checksum is not expected to
match.
