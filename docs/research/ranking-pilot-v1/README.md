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
transfer arms. They were curated from the committed raw search logs
`source-N/raw-candidates.jsonl.gz`. The recorded timings are committed as
`source-N/acquisition-ms` and `inputs-ms`. The protocol is described in
[the ranking-v1 experiment](../../experiments/ranking-v1.md).

Each task's generated corpus is committed as `<task>/corpus/records.bin`. Its
SHA-256 over the concatenated `.bin` bytes is the manifest's `corpusSha256`.
The generator's targets go through the host libm `tanhf`, whose last bits differ
between libm builds, so the test regenerates the corpora and compares them by
value (inputs exactly, targets within `1e-5`) rather than by digest.

## Versions and pins

- Code revision `fad4e742000261a02a530ea63b465ca6227ec639`, the manifest's
  `codeRevision`. It is a commit on this branch. The example that generates
  the inputs, `ockham/examples/ranking_pilot.rs`, and the bench,
  `ockham/examples/priority_ordering_bench.rs`, are both from this revision,
  and so are the input boundaries and the preparation-cost accounting
  described below. Later commits on the branch add only this archive, its
  tests and documentation.
- NEAT-AI-scorer revision `742c5df30fc98ece3afe784be1aca18f925b47b3`, built with
  `cargo build --release --locked -p rust_scorer`. Executable sha256
  `d057a3349da1c28e1319d7e235491597517581b17fda21ea1cf8f0dd55ffbbf7`. The
  scorer was run with `--gpu off --cost MSE`.
- Release executables built at that revision: bench
  `725aa37ee18ba3a240cad9cc95ccd1c6d82bd309a071df4eed441499512d8811` (the
  freeze binds this digest), `ranking_pilot`
  `3863ecc5e0f50f8ba908fba34ff76fc767a94f60e9e5e438a351b3dabada1788` and
  `neat_ai_ockham` `55b73d4516201aef52353ea4da73b8532950513b211ffda83d45cae5bf3d2b79`.
- rustc 1.99.1 (58cf8f9c3 2026-10-13), target `aarch64-unknown-linux-gnu`.
- Run on 18 November 2026.
- Manifest identity `a1f6de004715a90b9a25999d1256bb28a1198bdfa42ce34aa52d4c93df399b17`.
  The results, preflight and freeze files all carry this value.
- The manifest's paths are the run's absolute scratch paths under
  `/tmp/vibe-scratch/r261b/`. A rerun writes its own paths and digests.

This archive replaces the earlier run from the closed PR #279 (revisions
`a27d638` and `e02ef9f`) and its `cost-recount/` directory. Neither earlier
revision is on this branch, their raw candidate logs were never committed, and
they predate the input boundaries and the preparation-cost accounting. This
run, at the current code, reproduced every primary-cell and cold-start paired
difference and interval of the earlier run, as well as its verdict and coverage.
Only the timings differ. Keeping two superseded archives would add about 15,700
lines of JSON to maintain and no evidence this run lacks, so they were removed.

## Design

The design was predeclared in the manifest before the pilot ran. The bench
binds the manifest identity in the preflight, the freeze and the results.

- Eight tasks: two source tasks (families `source-a` and `source-b`), three
  calibration tasks and three validation tasks (families `target-1` to
  `target-3`). Each task has its own lineage, corpus version, parent and
  256-record corpus. Calibration fits and audit-visit freezing use only the
  calibration tasks. The validation tasks are touched only by the audit and
  the evaluation.
- Seeds 7 and 19. Pilot visits: 8. Audit visits frozen at 40, with a ceiling
  of 40 and a minimum complete pool of 20.
- Evaluation budgets 2 and 5. Calibration budgets 0 and 8. History fractions
  0, 0.5 and 1.
- Compute limits: 600,000 ms per task, 2,400,000 ms per run, a 60 s scorer
  timeout and a 100,000,000-byte corpus cap.
- Minimum of 3 independent groups. Thresholds: `minimumRecallGain` 0.05,
  `maximumFalseRejection` 0.1 and `maximumCostRatio` 1.0.
- The primary cell is full history, calibration budget 8, evaluation budget 5.

The strategies are `random` (the control), `partial` (partial-score ordering
alone), `multiple-proxies` (the existing fixed composite), `target-pairwise`
(a pairwise fit on the target's calibration records) and `transfer-pairwise`
(a pairwise fit that also uses the source histories). Calibration budget 0 is
the cold-start cell: the pairwise arms then see no target records.

## Sparse source histories

The source histories came from real ordinary `neat_ai_ockham` searches on the
two source parents, run with `--candidate-log`, `--no-exact-cleanup`,
`--seed 11`, `--max-experiments 40` and `--min-improvement 1000000`. The last
setting means no cut could be accepted, so every row stays on the immutable
parent. These are search-only histories.

- Raw logs: source-1 has 365 rows and source-2 has 383 rows. Both are
  committed, gzipped with `gzip -n -9`, as `source-N/raw-candidates.jsonl.gz`.
- `ranking_pilot manifest` curated each log to 12 candidates, one observation
  per candidate, preferring the fully scored one. Three of the 12 are labelled
  (fully scored) per source. The other nine were screened out without a full
  score.
- The recorded search wall times, committed as `source-N/acquisition-ms`, were
  922 ms and 955 ms. The manifest step added curation time of 3 and 4 ms, so
  the charged `acquisitionMs` values are 925 and 959. They are charged without
  amortisation.

## All costs counted

`totalResearchMs` is 11,726 ms. It sums these stages:

| Stage | ms |
| --- | --- |
| Preparation (`preparationCostMs`): input generation 9 (committed as `inputs-ms`) plus the manifest step's own 30 ms, excluding curation | 39 |
| Historical acquisition: two searches plus curation (925 + 959) | 1,884 |
| Pilot | 660 |
| Freeze | 71 |
| Audit (19 ms of it unpooled) | 8,715 |
| Evaluation | 357 |
| Total | 11,726 |

Three things are outside these totals: building the binaries, serialising and
writing the final results, and the bench's preflight `check` stage. The
preflight's wall time is neither recorded nor charged. It repeats the
revision, clean-source and scorer-digest checks that the pilot stage charges
as protocol preparation, which took 47 ms there. So the uncounted preflight
cost is of that order, well under 1% of the total. The timings were taken on a shared host and are
indicative only. This run's total is about three times the earlier #279 run's
3,469 ms. The growth is in protocol setup and scoring time, which the trial
ledgers share. The arms' own feature, fit, adaptation and ranking overhead is
the same in both runs.
The recall, false-rejection and paired results are identical, so the difference
is in timing only and changes no comparison or verdict.

Feature, fit, adaptation and ranking time is rounded up to whole milliseconds
(`audit::ms`), so non-zero work is never charged as 0. All 360 trials charge
ranking time, 216 charge feature time, 144 charge training time and 64 charge
adaptation time.

## Results

Paired top-5% recall differences in the primary cell, averaged over the three
validation groups, with 95% bootstrap intervals (1,000 replicates):

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
primary cell (`negativeTransferTasks`).

Partial-score ordering recovered the whole true top set (two or three
candidates) in all six primary trials.

Cold start (calibration budget 0, full history): transfer-pairwise minus
target-pairwise was −0.167 [−0.25, 0.0], negative on validation-1 and
validation-2. Transfer-pairwise minus random was −0.167 [−0.25, 0.0] and minus
partial was −1.0 [−1.0, −1.0]. Sparse history did not rescue the cold start.

Mean per-trial strategy cost and overhead in the primary cell:

| Strategy | Feature (ms) | Training (ms) | Adaptation (ms) | Ranking (ms) | Total (ms) |
| --- | --- | --- | --- | --- | --- |
| random | 0 | 0 | 0 | 1 | 878.7 |
| partial | 0 | 0 | 0 | 1 | 1125.8 |
| multiple-proxies | 1 | 0 | 0 | 1 | 878.0 |
| target-pairwise | 1 | 1 | 0 | 1 | 1635.0 |
| transfer-pairwise | 1 | 1 | 1 | 1 | 3522.8 |

Each arm's total is its full strategy ledger, including the preparation and
scoring every arm shares. The model's own feature and fit overhead is at most
about 1 ms per trial on these 256-record corpora. The transfer arm's extra cost
comes from historical acquisition and calibration scoring.

The verdict is `no-demonstrated-benefit`, with the reason "predeclared
thresholds: recall=false, false rejection=false, total cost=false".

## Coverage and failures

- 16 fully scored audit pools (each task × each seed, including the four
  source pools), drawn by seeded random visit selection: 640 visits selected,
  632 generated and 632 fully scored. All 16 pools are complete.
- The eight generation failures are all the protected output neuron
  `output-0`, which is protected from direct removal, so generation refuses
  it. They occur on the seed 19 pool of every task, and are listed per pool in
  `results.json` `coverage[].failures`.
- Validation: six pools with 237 candidates scored, of which 51 are ablation
  cuts and 186 are synapse cuts.
- The 186 validation synapse candidates have no neuron feature vector
  (`missingFeatures`). The pairwise and composite arms therefore score them at
  zero, tying them. This is a large part of why the learned arms trail partial
  ordering.
- Each pool has 9 to 16 genuine improvements. False rejection for partial
  ordering at evaluation budget 5 is 0.56 to 0.69, so every arm fails the 0.1
  threshold at this budget.

## Finding

On this synthetic pilot, partial-score ordering alone dominates. Neither the
existing composite nor the pairwise fits beat it, and sparse-history transfer
does not either. Transfer shows negative transfer on validation-2 in the
primary cell and on two of three tasks at cold start, and costs about three
times partial's strategy cost.

The limits are:

- The corpora are synthetic and homogeneous. A partial 5% sample of 256 records
  tracks the full corpus well here, which is a property of these corpora and
  should not be assumed elsewhere.
- Only three validation groups were used.
- Each source history has only three labelled rows.
- No representative data was available.

No success is claimed. The representative run remains outstanding.

## Maintenance decision

- Removed: the superseded #279 archive and its `cost-recount/` directory (see
  above). This run replaces them and can be reproduced from this branch.
- Kept, research-only: the pairwise and transfer arms of the example bench.
  They did not earn their cost on this synthetic pilot. But the pilot cannot
  separate method from fixture: 186 of 237 validation candidates have no
  feature vector, there are three validation groups and three labelled
  history rows per source. The arms are the subject of the representative run
  that this issue still owes. Nothing calls them outside the opt-in example
  bench, and operational rejection stays disabled.
- Predeclared removal rule: if the representative run also returns
  `no-demonstrated-benefit` for `target-pairwise` and `transfer-pairwise`
  against `partial` in its primary cell, remove both arms and their fit code
  from the bench. Keep `partial` and `multiple-proxies`.
- Kept: `ranking_pilot`. It is the reproducible generator for this archive and
  the tests that check the archive.

## Input checksums

| File | sha256 |
| --- | --- |
| `manifest.json` | `a0592ba9a7677aef7849fb3a20d62714958ea801e5fc3906842a2a49996b0a70` |
| `preflight.json` | `5389e932e3f6669fe9bfcebd612d59ee39ea68baf817a196ca9101d720965230` |
| `pilot.json` | `e3c7cac9a88ea249d1676ec0d56f8a165832a869cdcad45a2a536e8c0d54cb7e` |
| `frozen.json` | `e6c1a957452246d807bb244808e21743960c3dfaa6c4cc38cbebfb6881059f97` |
| `results.json` | `2bf582eea9a9ba1a427ffac25b940843d59852f6046b3914a60d7ce53a1f234a` |
| `audit.json.gz` | `7f757ea61eeef3d48d818699505ab8d8245f6fa25f1d96caf841db02c924063a` |
| `audit.json` (decompressed) | `4f2fd6f94fcc5b3ff38efbd504c33be0231c032cf0f28acd573433c675a1c902` |
| `source-1/history.jsonl` | `7f6009f2ef474fd4892801939dafb23d0401c5e50715ef2282513b66a6ab1e3c` |
| `source-2/history.jsonl` | `9ef60a6a94ea97634e38c4fca48c58431d5088e4ce42ea9d8221c5b844046c1b` |
| `source-1/raw-candidates.jsonl.gz` | `3e942874ba63051f36cbc8ec39f5f5a99fbcb7e2942181ef716e5bde5b653e4a` |
| `source-1/raw-candidates.jsonl` (decompressed) | `4ff45a278730d069e34d6dc4724edb621897f7f8ea421897d7f7507d31233950` |
| `source-2/raw-candidates.jsonl.gz` | `efee0e02a27310f183a04fdae83d49a941c7aa630c929daaa9d97010194891ca` |
| `source-2/raw-candidates.jsonl` (decompressed) | `8c845b49b4ee04e90579fd7a9a2ec32b74e76435277b1431cf97b9c0c24016e2` |

The archived audit and raw logs are gzipped with `gzip -n -9`. Decompressed,
they are byte-exact with the run's files. The other archived files are
byte-exact copies of the run's outputs.

Tests check the archive:

- `ockham/tests/ranking_pilot_archive.rs` recomputes the decompressed audit,
  pilot and source-history digests from the archived bytes, and checks them
  against the digests recorded in `results.json`, `frozen.json` and the
  manifest. It also checks the manifest identity across preflight, freeze and
  results, the verdict, the primary-cell sign, complete coverage, and the
  research-cost sum including `preparationCostMs`.
- `ockham/examples/ranking_pilot.rs::committed_raw_logs_recurate_to_the_committed_histories`
  re-runs the manifest step on the committed raw logs and timings. It checks
  that the curated histories are byte-identical to the committed ones.

## Reproduce

Build the scorer at the pinned revision, in a NEAT-AI-scorer checkout:

```sh
git checkout 742c5df30fc98ece3afe784be1aca18f925b47b3
cargo build --release --locked -p rust_scorer
```

Build the Ockham binaries at a clean checkout of revision
`fad4e742000261a02a530ea63b465ca6227ec639` or a later revision of this
code. The bench refuses a manifest whose `codeRevision` is not the checkout's
HEAD, so pass that HEAD to the manifest step below.

There are two levels of reproduction:

- Manifest step from the committed inputs. Run `ranking_pilot inputs <dir>`,
  then replace `<dir>/inputs-ms` with the committed `inputs-ms`. Decompress the
  committed `source-N/raw-candidates.jsonl.gz` into
  `<dir>/source-N/raw-candidates.jsonl`, and copy the committed
  `source-N/acquisition-ms`. Then run the manifest step below. The curated
  histories are byte-identical to the committed ones. The test
  `committed_raw_logs_recurate_to_the_committed_histories` does exactly this.
  The charged `acquisitionMs` and `preparationMs` add this run's own curation
  and manifest-step time.
- The whole pilot from scratch, with the steps below. The searches' rows and
  timings differ between runs, so the digests differ. The verdict and the
  direction of the paired differences are the comparable outputs.

The `<target>` placeholder is the cargo target directory:

```sh
cargo build --release --locked --example ranking_pilot --example priority_ordering_bench --bin neat_ai_ockham
```

Generate the inputs into a new directory `<dir>`, which must not exist:

```sh
<target>/release/examples/ranking_pilot inputs <dir> < /dev/null
```

This also writes `<dir>/inputs-ms`, the generation wall time. The manifest
step requires it and charges it in `preparationMs`.

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

### Manifest step input boundaries

The `ranking_pilot manifest` step hardens its inputs as follows.

- It resolves the pilot directory once. It refuses any symlink among the
  pilot-owned paths it reads (task directories, `parent.json`, the corpus
  directory and each `.bin` entry, `raw-candidates.jsonl`, `acquisition-ms`
  and `inputs-ms`), and any such path whose resolved location is outside the
  pilot directory.
- It refuses an existing `history.jsonl` or `manifest.json`, including a
  dangling symlink. It creates both exclusively (create-new), so it never
  writes through a symlink.
- Byte caps are checked before reading and again while reading: `parent.json`
  16 MiB, `raw-candidates.jsonl` 64 MiB (the bench's own history cap),
  `acquisition-ms` and `inputs-ms` 64 bytes each, each task's corpus 100,000,000 bytes in total
  (streamed digest; the same value is written to the manifest's
  `maxCorpusBytes`), and the scorer 1 GiB (streamed digest).
- The scorer is a separately chosen input. It may live anywhere and may be a
  symlink. Only its size is capped.
- Every read and digest finishes before the first write, so a refusal leaves no
  history and no manifest.
- This archive was produced with these boundaries in place.

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
