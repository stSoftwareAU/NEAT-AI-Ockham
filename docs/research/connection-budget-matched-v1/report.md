# Matched connection-budget follow-up v1: inconclusive

This separately frozen synthetic study supplies actual matched comparisons with
equal per-neuron quotas, equal per-depth quotas and the configured current-Ockham
control. It does not establish an optimisation benefit on evolved creatures.
The original [whole-budget pilot](../connection-budget-v1/report.md) remains
unchanged, including its compute exclusions.

## Predeclared follow-up

The original pilot exposed different acceptance schedules: allocation arms judged
one complete budget proposal, while Ockham screened and promoted each removal.
The opt-in `acceptance: "per-removal"` protocol applies Ockham's existing
screening and full-corpus promotion to each ranked allocation removal too. A
rejected removal is skipped; topology/statistics change only after acceptance.
There is no padding, relaxed matching tolerance, new winner rule or default change.

The `matched-pilot` generator creates eight independently generated layered
networks on a zero-output synthetic task. Each has two inputs, two hidden TANH
neurons, four outputs and 20 connections across two genuine depths (4 and 16
connections), with six target-neuron regions. Four deliberately weak output
connections occupy different regions in quota-rounding order. This design
makes legal multi-cut proposals comparable; quota and global selections coincide
here, so it does not establish an advantage from unequal allocation. The original
pilot retains the heterogeneous allocation/compute result.

Seeds `7` and `19`, targets `18` and `16` (two and four removals), 10% maximum
relative mismatch in each of the five counted compute dimensions, all original
compute limits and success thresholds were frozen before any validation score.
Training/calibration/validation contain 256/64/128 records per subject; calibration
is a training subset and validation inputs are disjoint. Generated inputs and
targets are application-agnostic, without evolution or fitting.

The [manifest](manifest.json) pins input contents, scorer, executable and exact
source digest. `codeRevision` records the pre-existing branch head
`62d1a62b17b7bc58222d8dd14de929a97ea54151`; the new protocol was in its working
tree when frozen and is identified by `sourceSha256`
`ec315f3a95c1b9b9d062b1ce84ecb5016821fc580eb71f7d780bb06e7d7722d0`.
The scorer identity is unchanged from the original pilot.

## Actual coverage and paired differences

All 128 arm runs completed without failures, timeouts or target misses. Every
reduction was accepted through authoritative full-corpus promotion. Matched pairs
have identical reduced achieved/accepted counts; every recorded work dimension
satisfies the frozen tolerance.

| Target | Global versus | Planned | Matched | Independent units | Full-score mean gain | Held-out mean gain |
| --- | --- | --- | --- | --- | --- | --- |
| 18 | per-neuron | 16 | 16 | 8 | 0 | 0 |
| 18 | per-depth | 16 | 16 | 8 | 0 | 0 |
| 18 | ockham | 16 | 12 | 7 | -8.9343127e-06 | -8.59522989e-06 |
| 16 | per-neuron | 16 | 16 | 8 | 0 | 0 |
| 16 | per-depth | 16 | 16 | 8 | 0 | 0 |
| 16 | ockham | 16 | 11 | 6 | -1.82271132e-05 | -1.82359869e-05 |

The 87/96 matched pairs include 23 actual global/Ockham comparisons. Nine Ockham
pairs are excluded for unmatched compute, with their complete costs retained.
Ockham matched coverage is 75% and 68.75%, below the predeclared 80%, with fewer
than eight matched independent units. Global versus quotas has zero difference
and `[0, 0]` intervals at both targets. Global scores are lower than Ockham on this
zero-output task; preserving higher-importance connections is not advantageous for an
intentionally oversized positive predictor whose desired output is zero.

The clustered familywise 95% bootstrap intervals and every individual pair are
in [summary.json](summary.json). The full/held-out Ockham intervals are:

| Target | Full-score gain interval | Held-out gain interval |
| --- | --- | --- |
| 18 | [-1.27112102e-05, -6.24410372e-06] | [-1.16805684e-05, -5.29784431e-06] |
| 16 | [-2.72541814e-05, -1.41806091e-05] | [-2.72257515e-05, -1.00537463e-05] |

The conclusion remains **inconclusive**: representative coverage is zero, the
global allocation does not pass the score-gain criteria, and Ockham coverage is
underpowered. This is a completed controlled synthetic comparison, not a claim
that the pruning adaptation improves NEAT-AI or generalises beyond these fixtures.

## Costs and measurements

The run counted 395 proposal visits, 24,576 statistics records,
358,016 scorer-reported and 383,232 reserved record evaluations
(including final training/validation measurement). Diagnostics separately evaluated
32,768 records and latency measurements 393,216. Freeze and verification
hashed 3,584 and 7,168 records respectively.

[results.jsonl](results.jsonl) retains per-output and stratum damage, opening/final
neuron/depth distributions, actual counts, refusals, scores, compiled latency and
costs for every arm, including exclusions. Compiled latency is indicative on a
shared host and excludes IO, scorer execution and compilation; no speed-up is
claimed. The release example build took 20 seconds outside arm budgets.

## Reproduce

From the repository root with the separately built released real scorer:

```bash
cargo build --release --example connection_budget < /dev/null
cargo run --release --example connection_budget -- \
  matched-pilot runs/connection-budget-matched-pilot-v1 /absolute/path/to/rust_scorer < /dev/null
cargo run --release --example connection_budget -- \
  freeze runs/connection-budget-matched-pilot-v1/spec-v1.json \
  runs/connection-budget-matched-pilot-v1/frozen-v1.json < /dev/null
cargo run --release --example connection_budget -- \
  run runs/connection-budget-matched-pilot-v1/frozen-v1.json \
  runs/connection-budget-matched-results-v1 < /dev/null
```

Use fresh paths for a new run. Rebuilding on another machine requires a new freeze;
compare protocol/input identities rather than assuming binary hashes match.
Representative models, default-CLI economics, quiet-host latency and a PDF byte
checksum remain unverified.
