# Controlled pruning-order experiment, protocol v1

This is research infrastructure for issue #261, not evidence that a proxy or
transfer model improves pruning. Ordinary CLI defaults, the pointwise
`train-ordering` command and authoritative full-score acceptance are unchanged.
The research command never applies a cut or writes a population candidate.

## Verified methods and adaptations

The method sections were retrieved on 13 November 2026:

- [FENAS v1, sections 3.1–3.3](https://arxiv.org/html/2610.15307v1#S3):
  CAST aligns heterogeneous initialisation proxies through empirical copulas
  and constructs synergy features; a few-label predictor combines them.
  Its default is a tree ensemble with pointwise regression. This experiment
  instead reuses Ockham's fixed composite fusion and existing measured
  activation, connectivity, cascade and sensitivity features. It does not
  implement CAST, Fisher, NASWOT, SynFlow or a tree ensemble. Already-fit
  creatures and pruning deltas differ from untrained architecture accuracy;
  an initialisation proxy's usefulness cannot be assumed here.
- [Searching with Sparse Histories v1, sections 4.1–4.3](https://arxiv.org/html/2610.32179v1#S4):
  SHINE imputes source rankings for a champion portfolio and pretrains a
  multi-task pairwise encoder using sparse histories and proxy pseudo-tasks.
  It adapts on target pairs with source regularisation, then searches in a
  shrinking predicted region. Here a linear pairwise logistic fit shares the
  existing model's numerical helpers and adapts its weights once on a bounded
  calibration set. There is no imputation, portfolio, neural encoder,
  task-specific head ensemble, proxy pseudo-task, source regulariser or online
  acquisition loop. This isolates a cheaper transfer component; it is not a
  reproduction or a performance claim about SHINE.

The baseline implementation from merged PR #118 is reused: `features::extract`,
`model`, `telemetry::load`/`CandidateRecord`, the core-backed `Sweep::fill_batch`
and `priority_ordering_bench`. The positional holdout in the ordinary trainer
and the original simulated benchmark are not validation evidence for this study.

## Data and separation

Copy [the manifest template](ranking-v1.template.json) to private storage and
replace its illustrative paths, digests, timestamps and grouping declarations.
The template is not a frozen experiment. `codeRevision` is the clean checkout's
full commit SHA. `scorerSha256`, parent digests and historical-log digests cover
complete file bytes. Corpus SHA-256 covers the concatenated `.bin` bytes in
sorted path order; `corpusIdentity` is also checked using ordinary
`corpus::corpus_info`. File layout changes therefore require identity updates.

Each task declares a parent lineage, corpus/task version, target family and
partition. Lineages, versions, parent bytes and corpus bytes cannot cross
source, calibration and validation partitions. The declarations must reflect
actual ancestry and task provenance; the tool cannot infer undisclosed ancestry
from a creature JSON file. Several validation tasks sharing a lineage, target family or corpus identity
are one connected uncertainty group, including transitive overlaps.

Source tasks require digested existing candidate logs with their original
parent checksum and corpus identity. Logs are bounded to 64 MiB and 4,096 rows
per task. Only individual full-score deltas with partial scores and the current
named feature schema are eligible for fitting. Missing full scores are
unlabelled, never losses. Future records are counted and excluded using the
strict `unixSecs < historicalCutoffUnix` boundary. Duplicate observations of one
candidate are refused; curate one pre-target observation per candidate.
`acquisitionMs` records the measured historical scoring/preparation cost; zero
means no recorded charge, not proof that acquisition was free.

Audit rows with neuron features embed ordinary `CandidateRecord` telemetry. Its
`corpusIdentity` uses the manifest's corpus cache identity, so those records can
be exported as source history for a later, independently partitioned experiment.
The historical loader also reads earlier v1 `random-audit` records that used the
manifest's full corpus SHA-256 in that field, normalising them in memory to the
cache identity. Other ordering names do not receive this compatibility conversion;
the log digest, parent checksum and remaining provenance checks still apply.

Calibration uses different lineages **and** corpus/task versions from validation,
within a declared target family. Thus this v1 adaptation uses an earlier or
related target version, not labels from the untouched validation task itself.
The calibration budget is family-wide, sampled across its calibration pools.
Normalisation is fitted only on the permitted source subset and calibration
rows (target-only uses calibration alone). Prior-win/failure counters in source
vectors are zeroed because v1 telemetry cannot date each underlying observation.
Audit features read no learnings cache.

## Sampling, failures and coverage

Candidates are selected by seeded SHA-256 priorities over the complete hidden
neuron and synapse visit population, before feature extraction or proposal.
The selected visit list is then passed to the existing pruning sweep. Failed
proposals are recorded without replacement. Every generated candidate is scored
individually with its immutable parent as same-call baseline, on both the 5%
partial corpus and full corpus. A sampled loser or partial scorer failure does
not suppress its full-score attempt. Parents are never overwritten.

Missing results, non-finite values, inconsistent record counts, scorer failures
and exhausted budgets remain explicit. Full results must cover the declared
record count for both baseline and candidate. Failed scoring makes the pool
incomplete; evaluation omits that pool and reports the omission. It does not
infer a label or silently call the remaining pool representative. Generation
failures, population size, selected/generated/scored counts and feature absence
are reported separately.

The ordinary proposal ladder can produce identity, ablation, constant and
synapse candidates. Group and merge proposals are not enabled in this protocol;
their coverage is explicitly zero. Existing neuron telemetry cannot represent
synapse feature vectors. The audit ledger retains those candidates with a
feature-availability flag; pairwise inputs zero-fill the neuron columns only
behind that flag. The composite control ties featureless candidates at zero.
Activation variance is not output variance; sensitivity here is the existing
activation-times-outgoing-weight estimate, not a gradient. Functional-path,
compiled-latency and gradient/Fisher measurements are not added without an
availability/cost case.

## Pilot, freeze and commands

The manifest declares seeds, source density grid, family calibration budgets,
evaluation budgets, fit limits, sample requirements and thresholds **before**
the calibration pilot. Pilot collection opens calibration data only. Freeze
chooses an affordable audit visit count using twice the slowest observed pilot
milliseconds per generated candidate, constrained by per-pool and total limits.
It refuses fewer than `minimumCompletePool` visits. No validation scores tune
these limits. The freeze binds the manifest, pilot file and running executable
SHA-256. Rebuilding a different executable requires another pilot and freeze.

Run from the committed source revision with Rust and `timeout` (Linux) or
`gtimeout` (macOS). Keep manifest, parents, logs and result directories private.
All output paths are exclusive: an existing output file/directory is refused.
For example, with a prepared private directory `/private/ordering`:

```bash
cargo build --example priority_ordering_bench < /dev/null
cargo run --example priority_ordering_bench -- --experiment check \
  --manifest /private/ordering/manifest.json \
  --out /private/ordering/preflight.json < /dev/null
cargo run --example priority_ordering_bench -- --experiment pilot \
  --manifest /private/ordering/manifest.json \
  --out /private/ordering/pilot < /dev/null
cargo run --example priority_ordering_bench -- --experiment freeze \
  --manifest /private/ordering/manifest.json \
  --pilot /private/ordering/pilot/pilot.json \
  --out /private/ordering/frozen.json < /dev/null
cargo run --example priority_ordering_bench -- --experiment audit \
  --manifest /private/ordering/manifest.json \
  --freeze /private/ordering/frozen.json \
  --out /private/ordering/audit < /dev/null
cargo run --example priority_ordering_bench -- --experiment evaluate \
  --manifest /private/ordering/manifest.json \
  --freeze /private/ordering/frozen.json \
  --audit /private/ordering/audit/audit.json \
  --out /private/ordering/results.json < /dev/null
```

Each scorer invocation has a hard `scorerTimeoutSecs` bound through the existing
scorer adapter. Per-task and run wall limits are checked between stages/calls;
a running preparation stage or scorer call can overrun the remaining wall
allowance. The corpus byte cap, statistics-record cap, pair/epoch caps and
per-call timeout bound that work. An external run supervisor should impose the
site's hard process deadline. Pilot timings are feasibility estimates from a
shared host, not performance guarantees. Build time is outside reported research
runtime and is not hidden as a strategy speed-up.

```mermaid
flowchart LR
    M[Private declared manifest] --> P[Calibration-only pilot]
    P --> F[Digest-bound freeze]
    F --> A[Random visits without replacement]
    A --> G[Existing prune engine]
    G --> S[Individual partial and full scores]
    S --> E[Isolated fits and paired evaluation]
    E --> R[Private JSON results and explicit omissions]
```

## Comparisons and decision rule

Five strategies share each validation pool, seed and full-score budget:

1. Seeded random ordering.
2. Partial-corpus score delta alone.
3. Existing fixed composite proxy fusion, without tuning.
4. Target-only pairwise logistic fit.
5. Sparse-history pairwise fit followed by target calibration adaptation.

Fits compare unequal scores within one task only, with equal total weight per
non-empty task. Deterministic capped pair enumeration follows seeded row order.
Empty or tied supervision gives a zero-score cold start; ranking ties use the
same independent seeded priority as the random control. Pairwise scores are
uncalibrated ordering keys, never acceptance probabilities. Historical density
zero and calibration budget zero are mandatory controls. Transfer minus
target-only per-task differences expose negative transfer.

For N scored candidates, the true top set contains every candidate tied at the
`ceil(N / 20)`-th full-score rank. Small pools use at least one top candidate;
pools below the declared minimum cannot support a verdict. Evaluation budgets
are clamped to N. Top recall is recovered top members / top-set size. Shadow
false rejection is the fraction of genuine improvements (`fullDelta >
minimumImprovement`) outside that budget. With no improvements it is unknown,
not zero. The baseline remains available when reporting the best full delta.

Results retain per-task/seed/grid metrics and paired task differences. Seeds are
averaged within task; connected lineage/family/version components are resampled for
95% percentile bootstrap intervals, with equal component weight. Fewer than
three observed groups yields no interval. Repeated seeds are not independent
tasks. No seed-level significance claim or hyper-parameter selection is made.

The primary cell is full history, the largest declared calibration budget and
the largest evaluation budget. Support for further research requires its lower
recall-difference interval to exceed `minimumRecallGain` versus random, partial
and target-only controls; every primary transfer trial must also meet
`maximumFalseRejection` and `maximumCostRatio` versus partial ordering. No
missing pool, unknown historical acquisition cost, insufficient calibration or
group-count shortfall can support a
positive verdict. Other grid cells diagnose cold starts and sparsity, not pick
a favourable result after validation. An inconclusive or unsupported result
never enables deployment or real rejection.

Cost reporting retains pilot, freeze, historical acquisition, audit and
evaluation totals, including protocol identity checks. It separates the actual
audit wall time from counterfactual
strategy ledgers: preparation/generation, feature extraction, historical
acquisition/loading, calibration acquisition, partial scoring, fit, adaptation,
ranking and selected full scoring. Historical and calibration acquisition is
charged without amortisation (including fully collected calibration pools even
when a smaller label budget is exposed). This is a conservative upper charge,
not a measured amortised speed-up. The full audit scores losers for measurement;
those extra validation scores are in audit cost, not charged as search queries.
Output serialisation/writing and build time are outside these runtime totals.
Timings on this shared host are indicative, never correctness assertions.

## Current finding

[The availability record](ranking-v1-availability.json) records an inconclusive
result. No representative corpus, independently grouped history manifest or
real scorer was supplied or found in the standard local locations inspected.
No representative feasibility pilot, frozen validation manifest or validation
comparison has been run. Deterministic fixtures validate implementation
contracts only; they do not demonstrate an optimisation benefit. The private
research run remains outstanding under the same issue.
