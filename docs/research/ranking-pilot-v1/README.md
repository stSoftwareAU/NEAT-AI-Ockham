# Synthetic ranking-v1 pilot (issue #261)

This is a synthetic pilot run with the real scorer, not evidence of a benefit
on representative data. The corpora are generated, homogeneous and
deliberately simple, and no representative creature or corpus was available.
Fixtures alone do not demonstrate an optimisation benefit. The pilot claims
no benefit, and operational rejection stays disabled.

The [manifest](manifest.json), [preflight](preflight.json),
[pilot](pilot.json), [freeze](frozen.json), [audit](audit.json.gz) and
[results](results.json) are the unedited outputs of the run. The two
[source histories](source-1/history.jsonl) are the curated inputs to the
transfer arms. The protocol is described in
[the ranking-v1 experiment](../../experiments/ranking-v1.md).

Each task's generated corpus is committed as `<task>/corpus/records.bin`. Its
SHA-256 over the concatenated `.bin` bytes is the manifest's `corpusSha256`.
The generator's targets go through the host libm `tanhf`, whose last bits differ
between libm builds, so the test regenerates the corpora and compares them by
value (inputs exactly, targets within `1e-5`) rather than by digest.

## Versions and pins

- Code revision `a27d638a209e78f24a1ce7d890521c83d058d180`, the manifest's
  `codeRevision`. The example that generates the inputs,
  `ockham/examples/ranking_pilot.rs`, and the bench,
  `ockham/examples/priority_ordering_bench.rs`, are both from this revision.
- NEAT-AI-scorer revision `742c5df30fc98ece3afe784be1aca18f925b47b3`, built with
  `cargo build --release --locked -p rust_scorer`. Executable sha256
  `d057a3349da1c28e1319d7e235491597517581b17fda21ea1cf8f0dd55ffbbf7`. The
  scorer was run with `--gpu off --cost MSE`.
- Bench executable sha256 `e02edb84f77af8326e82464e090f24bf225173b94efe41ae0c6fc4246b7fcbe5`,
  built in the release profile. The freeze binds this digest.
- rustc 1.99.1 (58cf8f9c3 2026-10-13), target `aarch64-unknown-linux-gnu`.
- Manifest identity `588be5c86f13bf1bd55f3e116f3c8e9009f7cf107b0f3ff8849456254db7bf23`.
  The results, preflight and freeze files all carry this value.
- The manifest's paths are the run's absolute scratch paths under
  `/tmp/vibe-scratch/r261/`. A rerun writes its own paths and digests.

## Design

The design was predeclared in the manifest before the pilot ran.

- Eight tasks: two source tasks (families `source-a` and `source-b`), three
  calibration tasks and three validation tasks (families `target-1` to
  `target-3`). Each task has its own lineage, corpus version, parent and
  256-record corpus.
- Seeds 7 and 19. Pilot visits: 8. Audit visits frozen at 40, with a ceiling
  of 40 and a minimum complete pool of 20.
- Evaluation budgets 2 and 5. Calibration budgets 0 and 8. History fractions
  0, 0.5 and 1.
- Minimum of 3 independent groups. Thresholds: `minimumRecallGain` 0.05,
  `maximumFalseRejection` 0.1 and `maximumCostRatio` 1.0.
- The primary cell is full history, calibration budget 8, evaluation budget 5.

The strategies are `random`, `partial` (partial-score ordering alone),
`multiple-proxies` (the existing fixed composite), `target-pairwise` (a pairwise
fit on the target's calibration records) and `transfer-pairwise` (a pairwise
fit that also uses the source histories).

## Sparse source histories

The source histories came from real ordinary `neat_ai_ockham` searches on the
two source parents, run with `--candidate-log`, `--no-exact-cleanup`,
`--seed 11`, `--max-experiments 40` and `--min-improvement 1000000`. The last
setting means no cut could be accepted, so every row stays on the immutable
parent. These are search-only histories.

- Raw logs: source-1 has 365 rows and source-2 has 383 rows.
- `ranking_pilot manifest` curated each log to 12 candidates, one observation
  per candidate, preferring the fully scored one. Three of the 12 are labelled
  (fully scored) per source. The other nine were screened out without a full
  score.
- Acquisition wall time was 436 ms for source-1 and 398 ms for source-2. These
  are charged without amortisation.

### Preparation costs outside the archived total

The archived `totalResearchMs` (3469 ms) charges the two searches' recorded
acquisition time, plus the pilot, freeze, audit and evaluation. It does not
include two preparation steps, which were measured afterwards with the release
`ranking_pilot` binary on the same host:

- Curation. Loading and curating the archived raw logs took 2 ms per source.
  The re-curated histories were byte-identical to the archived
  `history.jsonl` files. `ranking_pilot manifest` now adds this curation time,
  rounded up to whole milliseconds, to each source's `acquisitionMs`. A rerun
  therefore charges it to the transfer arm. The archived manifest predates
  that change and records 436 and 398 ms.
- Input generation. `ranking_pilot inputs` took 4 to 5 ms over three runs. It
  builds the synthetic parents and corpora that every arm shares, so it is a
  fixture cost and not a cost of any ranking method. The command now reports it
  on stderr.

Together these add about 9 ms, under 0.3% of the archived total. They do not
change any comparison or the verdict.

## Results

Paired top-5% recall differences in the primary cell, averaged over the three
validation groups, with 95% bootstrap intervals:

| Comparison | Mean difference | 95% interval |
| --- | --- | --- |
| partial − random | +0.833 | [0.75, 1.0] |
| multiple-proxies − random | +0.444 | [0.25, 0.583] |
| multiple-proxies − partial | −0.389 | [−0.5, −0.25] |
| target-pairwise − partial | −0.611 | [−0.75, −0.5] |
| transfer-pairwise − random | +0.139 | [0.0, 0.417] |
| transfer-pairwise − partial | −0.694 | [−0.75, −0.583] |
| transfer-pairwise − target-pairwise | −0.083 | [−0.25, 0.0] |

Transfer-pairwise minus target-pairwise was negative on validation-2 in the
primary cell.

Partial-score ordering recovered the whole true top set (two or three
candidates) in all six primary trials.

With calibration budget 0 and full history, transfer-pairwise minus
target-pairwise was −0.167 [−0.25, 0.0]. It was negative on validation-1 and
validation-2.

Mean per-trial strategy cost in the primary cell:

| Strategy | Mean cost (ms) |
| --- | --- |
| random | 275.5 |
| partial | 315.8 |
| multiple-proxies | 275.8 |
| target-pairwise | 469.3 |
| transfer-pairwise | 1303.3 |

Each arm's per-trial cost is its full strategy ledger, which includes the
preparation and scoring that every arm shares, so the "about four times" ratio
in the Finding compares totals, not arm-specific overhead.

The verdict is `no-demonstrated-benefit`, with the reason "predeclared
thresholds: recall=false, false rejection=false, total cost=false".

Costs, in milliseconds: pilot 169, freeze 34, historical acquisition 834
(436 plus 398), audit 2238 (unpooled 13), evaluation 194, and total research
3469. These timings were taken on a shared host and are indicative only.

## Coverage and failures

- 16 pools, including the four source pools, with 640 visits selected, 632
  generated and 632 fully scored. All 16 pools are complete.
- The eight generation failures are all the protected output neuron
  `output-0`, which is protected from direct removal, so generation refuses
  it. They occur on the seed 19 pool of every task.
- Validation: six pools with 237 candidates scored, of which 51 are ablation
  cuts and 186 are synapse cuts.
- The 186 validation synapse candidates have no neuron feature vector. The
  pairwise and composite arms therefore score them at zero, tying them. This is
  a large part of why the learned arms trail partial ordering.
- False rejection for partial ordering at evaluation budget 5 is 0.56 to 0.69.
  Each pool has 9 to 16 genuine improvements, so every arm fails the 0.1
  threshold at this budget.

## Finding

On this synthetic pilot, partial-score ordering alone dominates. Neither the
existing composite nor the pairwise fits beat it, and sparse-history transfer
does not either. Transfer shows negative transfer on validation-2 and costs
about four times partial's strategy cost.

The limits are:

- The corpora are synthetic and homogeneous. A partial 5% sample of 256 records
  tracks the full corpus well here, which is a property of these corpora and
  should not be assumed elsewhere.
- Only three validation groups were used.
- Each source history has only three labelled rows.
- No representative data was available.

The representative run remains outstanding.

## Cost recount with overhead rounded up

In the archived run, `featureMs`, `trainingMs`, `adaptationMs` and `rankingMs`
are 0 in all 360 trials. The bench's `audit::ms` rounded elapsed time down to
whole milliseconds, so feature extraction, fitting, adaptation and ranking
under 1 ms were charged as nothing. That cost was therefore not counted before
a rejection could be permitted. `audit::ms` now rounds up, so any non-zero work
costs at least 1 ms (`cost_milliseconds_round_up_so_sub_millisecond_work_is_counted`).

The five bench stages were re-run at code revision
`e02ef9f250c7169d7718ba6669b3030989aa6ea2`. The run used the same scorer
binary and the same generated parents and corpora. Its manifest was rewritten
from the same raw logs, and the re-curated histories were byte-identical. The
outputs are in [cost-recount](cost-recount/): the manifest, preflight, pilot,
freeze, gzipped audit and results. Their checksums are listed below. Manifest identity is
`df9353fdc595f3f8bde77d1c239aa2af86b4e6217c4bf8bfd91e34759f1928ee`, and the
charged `acquisitionMs` values are 438 and 400 (recorded time plus 2 ms of
curation each).

- Verdict: `no-demonstrated-benefit`, with the same three threshold failures.
- Coverage: 16 pools, 640 visits selected, 632 generated and fully scored, and
  every pool complete. These match the archive.
- Recall and false rejection are identical to the archive in every primary-cell
  trial. All primary-cell paired differences and intervals are identical too,
  including the negative transfer on validation-2.
- Every trial now charges ranking time. 216 of 360 trials charge feature time
  and 144 charge training time.

Mean per-trial overhead and total strategy cost in the primary cell:

| Strategy | Feature (ms) | Training (ms) | Adaptation (ms) | Ranking (ms) | Total (ms) |
| --- | --- | --- | --- | --- | --- |
| random | 0 | 0 | 0 | 1 | 240.0 |
| partial | 0 | 0 | 0 | 1 | 279.3 |
| multiple-proxies | 1 | 0 | 0 | 1 | 240.8 |
| target-pairwise | 1 | 1 | 0 | 1 | 382.2 |
| transfer-pairwise | 1 | 1 | 1 | 1 | 1223.2 |

These figures are upper bounds of about 1 ms each, because of the rounding.
On these 256-record synthetic corpora, the model's own feature and fit
overhead is negligible. The transfer arm's extra cost comes from historical
acquisition and calibration scoring. Total research cost was 2938 ms here
against 3469 ms archived. Both were measured on a shared host, so the
difference is noise, not an effect of the change. This recount does not replace
the archive, and it does not demonstrate a benefit either.

## Input checksums

| File | sha256 |
| --- | --- |
| `manifest.json` | `2042086f480acab0b4db9ba965482ae822c9c040a662e79e42c5dbc4efb9b258` |
| `preflight.json` | `29774d46a5c20edeac66d6e19e404869b0fa6f745ea3224568cbec64738f9863` |
| `pilot.json` | `de3897de42a88368aef756baaa63932b8a65a500fb4160b120ebba0a79aa5f41` |
| `frozen.json` | `bac33d3b6f0fe712f22505e3252e403680c25a1cac66a1ee4c45ddbfda487adc` |
| `results.json` | `02beb6a7986c9033ffc018bb96434686785156fde89938d31f700e71aa107de9` |
| `audit.json.gz` | `7b50ae6d9c021f3483ebda351bd417192caddb8969c7dbc25996f2b15bea8545` |
| `audit.json` (decompressed) | `687d7905d2fc839eee80119602902de29bd2c00a53edb1bd815fdbffb937be17` |
| `source-1/history.jsonl` | `814d8ca28dcac4ee4799fe5cc7d46feb3097b01a7406c180a1bac446e1f2263a` |
| `source-2/history.jsonl` | `deeacf066c3197b35b080612a92c1e7f69bca4ea8800f6a468213b3ca530f979` |
| `cost-recount/manifest.json` | `321f335edfac4a063d5a8944d4fb97216a0fe283ec2d5afaada50f9dca7989ad` |
| `cost-recount/preflight.json` | `71ab8cbda3a7c98084e9c1970623b69f016924a95c47599cd8cd8b49c3215ed5` |
| `cost-recount/pilot.json` | `3659b370b6db24cd1d1b82daf2ccd729dbc0f93b92b9f3e917299707521ad0d4` |
| `cost-recount/frozen.json` | `baf32530ed90b871f91d0a38ab58170cb3ae190dd6b5bdd934ff4e856e119de4` |
| `cost-recount/results.json` | `4047ebb8a9cc32a5be93a2e769735e7b132caf1d11c1222814f2c88a13c5b1ab` |
| `cost-recount/audit.json.gz` | `4b281a2b118a2b9a23f1d82d7ce24108ded4c5382b70d454775696393f233d09` |
| `cost-recount/audit.json` (decompressed) | `478c59ac19579ee53b4815c910af418bc6529c02ac6fe4032f48e35cf1444bae` |

The archived audit is gzipped with `gzip -n -9`. The decompressed file is
byte-exact with the run's `audit.json`. The other archived files are byte-exact
copies of the run's outputs. The integration test
`ockham/tests/ranking_pilot_archive.rs` recomputes the decompressed audit, pilot
and source-history digests from the archived bytes and checks them against the
digests recorded in `results.json`, `frozen.json` and the manifest. It also
checks that the manifest identity agrees across preflight, freeze and results,
the verdict, the primary-cell sign, and that every pool is complete. The
remaining rows of the table above are a record, not checked by the test.

## Reproduce

Build the scorer at the pinned revision, in a NEAT-AI-scorer checkout:

```sh
git checkout 742c5df30fc98ece3afe784be1aca18f925b47b3
cargo build --release --locked -p rust_scorer
```

Build the Ockham binaries at a clean checkout. The bench refuses a manifest
whose `codeRevision` is not the checkout's HEAD, so pass that HEAD to the
manifest step below.

- Revision `a27d638` reproduces the archive. There, `acquisition-ms` is
  optional and curation is not charged, and costs are rounded down.
- Revision `e02ef9f` or later reproduces the [cost recount](cost-recount/).
  There, `acquisition-ms` is required, curation is charged and costs are
  rounded up. The steps below describe this behaviour.

The `<target>` placeholder is the cargo target directory:

```sh
cargo build --release --locked --example ranking_pilot --example priority_ordering_bench --bin neat_ai_ockham
```

Generate the inputs into a new directory `<dir>`, which must not exist:

```sh
<target>/release/examples/ranking_pilot inputs <dir> < /dev/null
```

Run the two source searches. Time each one in milliseconds (for example with
`date +%s%3N` before and after), and write the figure to
`<dir>/$t/acquisition-ms`. The file is required: the manifest step refuses a
source task without it:

```sh
for t in source-1 source-2; do
  <target>/release/neat_ai_ockham <dir>/$t/parent.json <dir>/$t/corpus \
    --output-dir <run>-$t --scorer <scorer> \
    --scorer-arg=--gpu --scorer-arg=off --scorer-arg=--cost --scorer-arg=MSE \
    --timeout-seconds 180 --max-experiments 40 --seed 11 --no-exact-cleanup \
    --min-improvement 1000000 --candidate-log <dir>/$t/raw-candidates.jsonl < /dev/null
done
```

Write the manifest, with `<scorer>` as an absolute path and `<sha>` as the clean
HEAD commit:

```sh
<target>/release/examples/ranking_pilot manifest <dir> --scorer <scorer> --code-revision <sha> < /dev/null
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
  --audit <run>/audit/audit.json --out <run>/results.json < /dev/null
```

A rerun's timings and history rows differ from this archive, so its digests
will differ too. The verdict and the direction of the paired differences are
the comparable outputs.
