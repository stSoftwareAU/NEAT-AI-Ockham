# Synthetic ranking-v2 pilot (issue #261)

This is a synthetic pilot run with the real scorer, not evidence of a benefit
on representative data. The corpora are generated, homogeneous and
deliberately simple, and no representative creature or corpus was available.
Fixtures alone do not demonstrate an optimisation benefit. The pilot claims
no benefit, and operational rejection stays disabled.

This run is protocol `ockham-ranking-v2`. It repeats the design of the
[ranking-v1 pilot](../ranking-pilot-v1/README.md) and adds two pairwise path
columns, `pathSensitivityAvailable` and `logPathSensitivity`, computed from
the all-paths output sensitivity (`audit::path_sensitivities`).

The [manifest](manifest.json), [preflight](preflight.json),
[pilot](pilot.json), [freeze](frozen.json), [audit](audit.json.gz) and
[results](results.json) are the outputs of the run. The audit is gzipped
with `gzip -n -9` to keep the repository small. The two
[source histories](source-1/history.jsonl) are the curated inputs to the
transfer arms. They were curated from the committed raw search logs
`source-N/raw-candidates.jsonl.gz`, also gzipped with `gzip -n -9`. The
recorded timings are committed as `inputs-ms`, `build-ms`,
`bootstrap-build-ms`, `manifest-tail-ms` and `source-N/acquisition-ms`. The
write times of the phase files are committed as `*.write-ms`. The protocol is
described in [the ranking-v1 experiment](../../experiments/ranking-v1.md).

The corpora and parents are not committed. Each task's corpus is
byte-identical to the v1 archive's committed `<task>/corpus/records.bin`, and
the manifest's `corpusSha256` values equal those digests. The test checks this
against the v1 files. The parents match the `parentSha256` values in the
[v1 manifest](../ranking-pilot-v1/manifest.json). The v1 archive does not
commit `parent.json`, so the parents are matched by digest only.

## Versions and pins

- Code revision `7cdabe6b89b65ee22a24151c7618b88c598136d9`, the manifest's
  `codeRevision`. It is Develop after PR #289. Audit and pilot files carry
  `formatVersion` 2. Later commits on the branch add only this archive, its
  test and its documentation.
- NEAT-AI-scorer revision `742c5df30fc98ece3afe784be1aca18f925b47b3`, built
  with `cargo build --release --locked -p rust_scorer`. Executable sha256
  `d057a3349da1c28e1319d7e235491597517581b17fda21ea1cf8f0dd55ffbbf7`, the same
  as v1. The scorer was run with `--gpu off --cost MSE`.
- Release executables built at that revision: bench
  `517bc85290700fa8d6048099efdf07859139e6d5fa99f2ff6afeb546cd5dcf8b` (the
  freeze binds this digest), `ranking_pilot`
  `244552b788a2c1e63af79bde7ce58d8cf9bb58a22d2085c7015845b81ae58318` and
  `neat_ai_ockham`
  `c6af18d1c5ab9c993279aa5400715c125fc6fa540cae77f1dfcf35837ea20d82`.
- rustc 1.99.1 (58cf8f9c3 2026-10-13), target `aarch64-unknown-linux-gnu`.
- Run on 19 November 2026.
- Manifest identity `767bfd5f57f610c78b57c7a4bd748926ea2fd5bc9f76b800dca49c045850a17c`.
  The results, preflight and freeze files all carry this value.
- The paths in the files are absolute scratch paths under
  `/tmp/vibe-scratch/s261v2/`. A rerun writes its own paths and digests.

## Design

The design is identical to v1's, so it is not repeated here. See the
[v1 Design section](../ranking-pilot-v1/README.md#design). It has the same
eight tasks, seeds 7 and 19, pilot and audit visits, budgets, compute limits,
thresholds and primary cell (full history, calibration budget 8, evaluation
budget 5). The strategies and the cold-start cell are also the same.

## Source histories

The source searches were run again from scratch with the same flags as v1,
through `ranking_pilot search` (see Reproduce). They are not the v1 histories
and do not match them byte for byte.

- Raw logs: source-1 has 365 rows and source-2 has 383 rows. Each was curated
  to 12 candidates, three of them fully scored.
- Recorded search times, committed as `source-N/acquisition-ms`: 314 and 311
  ms. With curation, the charged `acquisitionMs` values are 316 and 313.

## Costs

`totalResearchMs` is 50,574 ms. It is the sum of the ten charged fields of
`results.json` below.

| Stage | ms |
| --- | ---: |
| Audit (`auditCostMs`) | 1,754 |
| Pilot | 158 |
| Freezing | 48 |
| Historical acquisition (316 + 313) | 629 |
| Preparation (inputs 7 plus manifest step 17, excluding 4 ms curation) | 24 |
| Preflight `check` (`preflightCostMs`) | 19 |
| Evaluation | 365 |
| Build (bootstrap 31,389 plus `ranking_pilot time` 16,181) | 47,570 |
| Serialisation | 1 |
| Recorded writes (preflight 1, freeze 1, audit 3, manifest tail 1) | 6 |
| Total | 50,574 |

The build dominates. The bootstrap build of `ranking_pilot` was timed by the
shell clock into `bootstrap-build-ms`. The remaining examples and binary were
timed by `ranking_pilot time` into `build-ms`. The builds used a cargo target
directory that already held earlier build output on this host, so they were
partly incremental. A cold build costs more. The scorer build (about 95 s) is
not charged, as in v1. The results file's own write (`results.json.write-ms`,
1 ms) is outside the total, since the file cannot hold its own write time.

## Results

The verdict is `no-demonstrated-benefit`, with the reason "predeclared
thresholds: recall=false, false rejection=false, total cost=false".

Paired top-5% recall differences in the primary cell, averaged over the three
validation groups, with 95% bootstrap intervals:

| Comparison | Mean difference | 95% interval |
| --- | ---: | --- |
| transfer-pairwise − partial | −0.611 | [−0.75, −0.333] |
| partial − random | +0.833 | [0.75, 1.0] |
| multiple-proxies − partial | −0.389 | [−0.5, −0.25] |
| transfer-pairwise − target-pairwise | 0.0 | [0.0, 0.0] |

Cold start (calibration budget 0, full history, evaluation budget 5):
transfer-pairwise minus target-pairwise is −0.167 [−0.25, 0.0], negative on
validation-1 and validation-2.

Coverage is complete on all 16 pools. There were 640 visits selected, 632
generated and fully scored, and 0 missing features. The eight generation
failures are the protected output neuron `output-0` on the seed 19 pool of
every task, as in v1. The 632 audit rows with features each carry a
`pathSensitivity`. The eight rows without features are those failures.

## Path-column effect

The audit pools are identical to v1's. The same 640 candidates have the same
digests, full and sampled deltas and feature vectors, row by row. Only
`pathSensitivity` is new. The effect of the path columns is as follows.

- At history fraction 0, neither pairwise arm sees source rows, so the path
  columns are the only changed input there. Source telemetry carries no path
  signal, so `historical_rows` sets both path columns to 0 for source rows.
  At history fractions 0.5 and 1, the transfer arm sees source rows with
  zero path columns.
- 104 of the 120 paired cells are identical to v1's, including every
  evaluation-budget-5 cell. The 16 that differ are all at evaluation budget 2
  and calibration budget 8. Target-pairwise minus random moved from 0.0 to
  −0.083 [−0.25, 0.25], and target-pairwise minus partial from −0.611 to
  −0.694 [−1.0, −0.333], at every history fraction. The transfer arms moved
  likewise at fraction 0.
- At fractions 0.5 and 1, transfer minus target-pairwise moved from −0.083 to
  0.0 [0.0, 0.0] and from −0.028 to +0.056 [0.0, 0.167]. The source histories
  were also re-run, so these two changes cannot be attributed to the columns
  alone.

The path columns did not help on this pilot.

## Finding

Partial-score ordering still dominates the primary cell. The transfer arm
ties the target-only fit there, and it is negative at cold start on two of
three tasks. No benefit is claimed. The representative run remains
outstanding.

## Maintenance decision

- The removal rule is unchanged. It is predeclared in the
  [ranking-v1 experiment](../../experiments/ranking-v1.md). The pairwise arms,
  the path columns and `audit::path_sensitivities` are removed only if the
  representative v2 run also returns `no-demonstrated-benefit` against
  `partial`. This synthetic run does not trigger it.
- The pairwise arms, path columns and `audit::path_sensitivities` stay
  research-only in the opt-in example bench. Operational rejection stays
  disabled.

## Test

`ockham/tests/ranking_pilot_archive.rs`, test
`archived_v2_ranking_pilot_matches_its_digests_costs_and_path_signal`, checks
the archive. It checks the verdict, the manifest identity across results,
freeze and preflight, and the audit and pilot digests. It checks the source
history digests and each corpus digest against the v1 files. It checks the
protocol, the code revision and `formatVersion` 2. It checks that every audit
row with features has a finite, non-negative `pathSensitivity` and that rows
without features have none (632 rows). It checks the primary-cell sign,
complete coverage and the coverage totals. It checks that the ten charged
stages sum to 50,574 ms, the build cost (47,570 ms), the recorded writes (6 ms),
the pilot and preflight costs, and the source acquisition costs.

## Checksums

| File | sha256 |
| --- | --- |
| `manifest.json` | `3a3784a7de8cfa256bfa7d5375305c2c198b802c2fc84bea0f3a96e1c6e52112` |
| `preflight.json` | `52f670fc1a6f9bad61c83f5b59b34d4b64ca8071e8de6df96902d1926a911456` |
| `pilot.json` | `7952391ce3b77cb74a290701568dc4d52c4d5f50ce9a7f74ef20cb06dcef504d` |
| `frozen.json` | `007632e85abdb380e6305782cfb1b22b49683979c6facc8c0db60b377af7361f` |
| `results.json` | `dfc935e1e40dccf2feeaf9e1c9876cc16ef5e9fad5f7090bc8ddff11a3bd931e` |
| `audit.json.gz` | `612157a0fa5856e159ea82c869d26862b7b95bee67292ff9d144549d70afe632` |
| `audit.json` (decompressed) | `9d340a94716f2bd1c0d828a412e46ad0a8c863b05cd431d5df3ee270b0fa1628` |
| `source-1/history.jsonl` | `e5e25e852292c9a15c538c28d66c613161bd579013e9fb8b15b7e2548dd9145b` |
| `source-2/history.jsonl` | `ce61f3319244d8b28f2173e10e24077afd2784cbe5d2f0aae4693285e711b062` |
| `source-1/raw-candidates.jsonl.gz` | `6979fd9912c546ee30fd0e7b1584e311404d0375b3743c01aa782292c5c9c2ee` |
| `source-1/raw-candidates.jsonl` (decompressed) | `eb713b96f09650b92d4b807f3955350f048f9f07e6b3fa53899579182b7c8b08` |
| `source-2/raw-candidates.jsonl.gz` | `65dece3d1ed2830a33e06cba1bab769386bdb08e16484a5df63a8e152a28d0d9` |
| `source-2/raw-candidates.jsonl` (decompressed) | `ab17d76e18df17990468d087586e548ed45b14aa7964a6beccd7825a47edb572` |

The archived audit and raw logs are gzipped with `gzip -n -9`. Decompressed,
they are byte-exact with the run's files. The other archived files are
byte-exact copies of the run's outputs.

## Reproduce

Build the scorer at the pinned revision, in a NEAT-AI-scorer checkout:

```sh
git checkout 742c5df30fc98ece3afe784be1aca18f925b47b3 < /dev/null
cargo build --release --locked -p rust_scorer < /dev/null
```

Build the Ockham binaries at a clean checkout of `<sha>`, which is
`7cdabe6b89b65ee22a24151c7618b88c598136d9` or a later protocol-v2 revision,
with `<target>` as the cargo target directory. The bench refuses a manifest
whose `codeRevision` is not the checkout's HEAD, so pass that HEAD to the
manifest step below. Time the bootstrap build of the generator with the shell
clock:

```sh
start=$(date +%s%N); cargo build --release --locked --example ranking_pilot < /dev/null; end=$(date +%s%N)
bootstrap_ms=$(( (end - start + 999999) / 1000000 ))
```

Generate the inputs into a new directory `<dir>`, which must not exist. Then
record the bootstrap time there:

```sh
<target>/release/examples/ranking_pilot inputs <dir> < /dev/null
echo "$bootstrap_ms" > <dir>/bootstrap-build-ms
```

Time the remaining build with `ranking_pilot time`:

```sh
<target>/release/examples/ranking_pilot time <dir>/build-ms -- \
  cargo build --release --locked --example ranking_pilot \
  --example priority_ordering_bench --bin neat_ai_ockham < /dev/null
```

Run the two source searches with `ranking_pilot search`. It times each search
and writes `<dir>/<task>/acquisition-ms`. It also writes the raw candidate log
through `--candidate-log`:

```sh
for t in source-1 source-2; do
  <target>/release/examples/ranking_pilot search <dir> $t -- \
    <target>/release/neat_ai_ockham <dir>/$t/parent.json <dir>/$t/corpus \
    --output-dir <run>-$t --scorer <scorer> \
    --scorer-arg=--gpu --scorer-arg=off --scorer-arg=--cost --scorer-arg=MSE \
    --timeout-seconds 180 --max-experiments 40 --seed 11 --no-exact-cleanup \
    --min-improvement 1000000 --candidate-log <dir>/$t/raw-candidates.jsonl \
    < /dev/null
done
```

Write the manifest, with `<scorer>` as an absolute path:

```sh
<target>/release/examples/ranking_pilot manifest <dir> --scorer <scorer> \
  --code-revision <sha> < /dev/null
```

Then run the five bench stages in order, writing into a new run directory
`<run>`. These are the commands from
[the ranking-v1 experiment](../../experiments/ranking-v1.md):

```sh
<target>/release/examples/priority_ordering_bench --experiment check \
  --manifest <dir>/manifest.json --out <run>/preflight.json < /dev/null
<target>/release/examples/priority_ordering_bench --experiment pilot \
  --manifest <dir>/manifest.json --out <run>/pilot < /dev/null
<target>/release/examples/priority_ordering_bench --experiment freeze \
  --manifest <dir>/manifest.json --pilot <run>/pilot/pilot.json \
  --out <run>/frozen.json < /dev/null
<target>/release/examples/priority_ordering_bench --experiment audit \
  --manifest <dir>/manifest.json --freeze <run>/frozen.json \
  --out <run>/audit < /dev/null
<target>/release/examples/priority_ordering_bench --experiment evaluate \
  --manifest <dir>/manifest.json --freeze <run>/frozen.json \
  --audit <run>/audit/audit.json --preflight <run>/preflight.json \
  --out <run>/results.json < /dev/null
```

A rerun's timings and history rows differ from this archive, so its digests
differ too. The verdict and the direction of the paired differences are the
comparable outputs.
