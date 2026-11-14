# Connection-budget pilot v1: inconclusive

This is a synthetic correctness and measurement pilot, not evidence of a benefit
on evolved creatures. No representative model/corpus pair was available in the
local checkout. The investigation remains incomplete on that acceptance scope.

The [frozen manifest](manifest.json) was written before validation scoring.
It identifies the code revision `a48160f85c30d571c12f687ff23dddca44f2cd42`, source,
example executable, scorer executable, input contents, controls and paper pin.
The scorer was built from NEAT-AI-scorer revision
`742c5df30fc98ece3afe784be1aca18f925b47b3` (its manifest declares 1.1.103),
using the same released core pin as this example.
The [protocol and reproduction command](../../connection-budgets.md) describe
the permanent-pruning assumptions and the one-candidate Ockham control.

## Observed coverage

Eight separately generated teachers, two seeds and two targets produced 128
arm runs. Every arm reached its requested count, and all 128 proposals were
authoritatively accepted against their opening parent. There were no failed
runs or timeout exits. These deliberately weak-edge fixtures establish neither
maturity nor generalisation to representative tasks.

| Target | Global versus | Planned pairs | Matched achieved count and compute | Independent units in matched set |
| --- | --- | --- | --- | --- |
| 30 | Per neuron | 16 | 0 | 0 |
| 30 | Per depth | 16 | 0 | 0 |
| 30 | Ockham | 16 | 0 | 0 |
| 34 | Per neuron | 16 | 0 | 0 |
| 34 | Per depth | 16 | 4 | 2 |
| 34 | Ockham | 16 | 0 | 0 |

All 92 exclusions were compute mismatches under the frozen 10% rule. Targets
and compute limits were not relaxed after seeing validation. The four matched
seed pairs had zero full-score and held-out-score difference; the bootstrap
intervals were `[0, 0]` for both metrics. Two units and 25% coverage in that
comparison miss the declared eight-unit and 80% thresholds. Other comparisons
have null uncertainty intervals rather than invented estimates. The study has
zero representative independent units and concludes **inconclusive**.

## Evidence and costs

[results.jsonl](results.jsonl) contains each arm's requested/achieved/accepted
counts, structural distributions, output support, scores, regional damage,
latency trials, refusals and resource accounting. [summary.json](summary.json)
contains the paired distributions, clustered uncertainty and exclusions.

The run recorded 1,594 proposal visits, 32,640 completed statistics records,
256,512 scorer-reported record evaluations and 267,264 reserved scorer record
evaluations. Compiled diagnostics evaluated 32,768 records; latency measurement
evaluated 393,216 records separately. Freeze and verification hashed 3,584 and
7,168 records respectively. These deterministic work counts are the primary
cost evidence.

Wall times and compiled latency samples were taken on a shared, parallel host;
they are indicative and do not establish a speed-up. Scorer execution is
accounted separately from compiled inference. Building the shared scorer took
about 68 seconds and building the example about 30 seconds, outside arm budgets.
No production ordering, CLI default, population entry or deployment changed.

Not performed: representative runs, a default-CLI economics comparison, quiet-host
latency validation, or a PDF byte checksum. The paper pin is explicitly an
extracted-text checksum. Private corpora and operational integration are absent
from these public artefacts.
