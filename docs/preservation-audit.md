# Preservation audit (report-only)

## Purpose and status

`audit-preservation` tests a hypothesis: a compressed child that keeps its
parent's behaviour on the same records is a better candidate for acceptance
than the sampled error screen alone suggests. The idea comes from
arXiv:2611.13681. Only the abstract was available when this was written; the
full method was not retrieved, so the formulas below are this crate's own
operationalisation, not a transcription of that paper.

No representative run exists yet. The audit therefore claims no benefit. Its
output is a diagnostic report, nothing more.

## What it never does

- It never changes acceptance, the default run path, or any default flag.
- It never writes a creature. The parent and children are read only.
- It never substitutes for the full scorer. Similarity is not authoritative
  fitness; `fullDelta` remains the only accept evidence.
- Its success rule needs representative runs before it means anything.

## Inputs

Each child is compared with the parent on the same records. A record is one
input vector and one target vector from a corpus of `.bin` files. For each
split, the first `R = min(maxRecords, corpus.record_count)` records are used,
where `K` is the number of outputs.

- `p_rk`, `c_rk`: parent and child outputs on record `r`, output `k`, from
  `neat_core::compile_creature` and `net.activate(inputs, creature.output)`.
- `t_rk`: the target from the corpus.
- `p̄_k`: the parent's mean output on the same split.

Every child must have the parent's input and output widths, and each corpus
must match them. A mismatch is refused, and the error names the child or split.

## Formulas

```text
E_child = (1 / RK) * sum_r sum_k (c_rk - p_rk)^2
E_mean  = (1 / RK) * sum_r sum_k (pbar_k - p_rk)^2
E_shuf(s) = (1 / RK) * sum_r sum_k (p_{pi_s(r),k} - p_rk)^2

preservationVsMean    = 1 - E_child / E_mean
preservationVsShuffle = 1 - E_child / mean_s(E_shuf(s))

taskLoss = (1 / RK) * sum_r sum_k (c_rk - t_rk)^2   (parent and child)
```

`pi_s` is a seeded Fisher–Yates permutation of the records, using the crate's
`SplitMix64`. Larger preservation scores mean the child stays closer to the
parent.

### Degenerate handling

- A denominator that is exactly `0.0`, or below `1e-24`, makes that score
  `null`. The reason is recorded in `degenerate`: `"zeroVarianceParent"` for the
  mean control, `"zeroShuffleError"` for the shuffle control.
- The audit never divides by zero and never emits `NaN` or infinity.
- `collapsedChild` is `true` when every child output column has zero variance on
  the split while the parent does not. Such a child has
  `preservationVsMean <= 0` by construction, and the report says so.
- A non-finite output on any record marks that child with
  `failure: "nonFiniteOutput at record N"`. The child is excluded from the
  comparison and listed in `failures`.

### Uncertainty

`ci95` is a percentile bootstrap 95% interval (2.5th and 97.5th percentiles) of
`preservationVsMean`. It resamples records, uses `bootstrapResamples`
resamples, and is seeded by `bootstrapSeed`. Per-record sums of squares for the
child and for the mean control are stored so that resampling needs no second
pass over the corpus.

`latencyMsPer1kRecords` is the wall-clock time of the activation loop. It is
indicative only and is never used to decide anything.

## Controls

- **Mean-output control.** The parent's mean output on the split is the
  baseline. `E_mean` measures how far the parent's outputs sit from their own
  mean.
- **Record-shuffled control, per seed.** The parent's outputs are paired with
  the wrong records by a seeded permutation. `E_shuf(s)` is computed once per
  seed in `shuffleSeeds`, and the report keeps each value. A child that does no
  better than a shuffled parent has learnt nothing about records.

## Predeclared parameters and compute limits

Fix these before the run. The manifest records them, so the report can be
checked against them afterwards.

| Parameter | Rule |
|---|---|
| `shuffleSeeds` | Required. Non-empty and unique. |
| `bootstrapSeed` | Default `0`. Must not appear in `shuffleSeeds`. |
| `bootstrapResamples` | Default `1000`. Range `100..=100000`. |
| `maxRecords` | Default `4096`. Range `1..=1000000`. Sample size per split. |
| `children` | Range `1..=256`. Names must be unique and non-empty. At least one child per `role`. |
| `children[].role` | Required: `"calibration"` or `"validation"`. |
| `children[].preparationMs` | Optional. Finite and `>= 0`. |
| `sampleTolerance` | Default `0.0`. Finite and `>= 0`. |
| `minAucGain` | Required. Finite. |

`fullDelta` and `sampleDelta` must be finite. `calibration` and `validation`
must not resolve to the same path.

## Calibration and validation

The threshold `theta` is fitted on the calibration split only. Both the records
and the children are split: every child carries a `role`. Calibration children,
scored on the calibration corpus, fit `theta`. Validation children, scored on the
validation corpus, are the only ones the gates count. A child is never in both.

1. Each calibration `preservationVsMean` score is a candidate `theta`.
2. The chosen `theta` maximises balanced accuracy, `(TPR + TNR) / 2`. A child is
   accepted when its score is at least `theta`, and a child is positive when it
   is genuinely better (`fullDelta > 0`).
3. Ties go to the smallest `theta`.
4. If the calibration split has no better child, or no worse child, or any score
   set is empty, `theta` is `null` and `failures` records
   `"threshold not fittable: <why>"`.

The validation split is not used until `theta` is fixed. On validation, the
preservation gate rejects a child when its score is below `theta`. A `null`
score counts as not covered.

## Gates and the success rule

Two gates are compared on validation: the sample gate and the preservation
gate. The sample gate rejects when `sampleDelta < -sampleTolerance`.

For each gate, the report gives:

- `rejected`: the number of children rejected.
- `falseRejections`: children rejected that are genuinely better.
- `falseRejectionRate`: `falseRejections / betterChildren`. It is `null` when no
  child is better.
- `AUC`: the Mann–Whitney area under the curve (reusing `model::auc`) of the gate's score for
  predicting "better". Ties count as 0.5. It is `null` when undefined.

Each child also appears as a paired row, `{name, role, better, fullDelta,
sampleDelta, preservationVsMean, sampleRejects, preservationRejects}`. The rows
hold the paired score distributions. `preservationVsMean` is the child's score
on its own role's corpus. `preservationRejects` is `null` for calibration
children. A bootstrap 95% interval over children (seeded by
`bootstrapSeed`) gives `aucGain = AUC_preservation - AUC_sample`.

The rule is fixed before the run:

- `met` when `aucGain >= minAucGain` **and** the preservation
  `falseRejectionRate` is no greater than the sample `falseRejectionRate`.
- `notEvaluable` when any input is `null`.

Even `met` is only a statement about the fixtures or runs supplied. It is not a
demonstration of benefit until representative runs exist.

## Known limitations and negative findings

- `fullDelta` and `sampleDelta` are supplied in the manifest from earlier scorer
  runs. The audit does not compute them, and it does not run the sampled screen
  in `promote.rs`.
- neat-core clamps non-finite activations, so a NaN or infinite corpus input
  produces a huge finite output, not a non-finite one. The `nonFiniteOutput`
  paths are defensive: the unit test drives them directly. A fixture with a NaN
  input showed that one clamped record dominates every error term. The child
  with a doubled weight still scored `preservationVsMean = 1.0`, the same as
  an exact copy. The score cannot tell children apart when the corpus holds
  such records.
- The arXiv full text could not be fetched here (no network access), so the
  method is still not pinned to the paper. `latencyMsPer1kRecords` times the whole streamed pass, so
it includes corpus reading and per-record scoring, not only the activation loop.

## Manifest example

```json
{
  "parent": "parent.json",
  "children": [
    { "name": "c1", "creature": "c1.json", "fullDelta": 0.01, "sampleDelta": -0.002,
      "role": "calibration", "preparationMs": 1200 },
    { "name": "c2", "creature": "c2.json", "fullDelta": -0.02, "sampleDelta": 0.001,
      "role": "validation", "preparationMs": 900 }
  ],
  "calibration": "calib-corpus-dir",
  "validation": "valid-corpus-dir",
  "shuffleSeeds": [1, 2, 3],
  "bootstrapSeed": 7,
  "bootstrapResamples": 1000,
  "maxRecords": 4096,
  "sampleTolerance": 0.0,
  "minAucGain": 0.05
}
```

Paths are resolved relative to the manifest's directory when they are relative.
The manifest uses `deny_unknown_fields`, and it is capped at 1 MiB, checked by
metadata before the file is read.

`preparationMs` is the declared cost of producing the child, such as pruning,
search and scorer time. The audit cannot measure that cost itself.

`fullDelta` is the authoritative full-scorer fitness change of the child against
the parent, larger is better, taken from an earlier full-scorer run such as an
Ockham journal. `sampleDelta` is the current sampled screen's change.

## Report fields

The report is written atomically: first to `<out>.tmp`, then renamed. Its
top-level fields are:

- `schema`: `"ockham-preservation-audit/1"`.
- `evidence`: `"fixture-or-unverified"`. It records the status of the
  evidence behind the report.
- `reportOnly`: `"report-only; not a demonstration of benefit"`. This fixed
  sentence is a separate top-level field, not part of `evidence`.
- `method`: the formula text and the source note, which says the abstract only
  was used.
- `versions`: the crate version, `gitCommit` (from `git rev-parse HEAD` at
  build time, or `"unknown"` without git; uncommitted edits are not detected),
  SHA-256 of the parent and of each child, and the
  identity of each corpus.
- `manifest`: an echo of the input manifest.
- `threshold`: the fitted `theta`, or `null`.
- `splits`: `calibration` and `validation`, each with per-child scores,
  `ci95`, `collapsedChild`, `degenerate`, `taskLoss` and
  `latencyMsPer1kRecords`. The split's `parent` entry carries `failure` (a
  non-finite parent output) and `null` task loss and control error when it
  fails.
- `gates`: `sample` and `preservation`.
- `paired`, `aucGain` (value and `ci95`).
- `criterion`: `status` and `rule`.
- `coverage`: children supplied and scored (in total and per role), and
  validation records.
- `costs`: calibration and validation activation milliseconds, `evaluationMs`
  (θ search, gates and the AUC-gain bootstrap), `totalMs` (whole audit wall
  clock), records activated, and `childPreparationMs`. That is the sum of the
  declared `preparationMs`, or `null` unless every child declares one, with
  `childrenWithPreparationCost`.
- `failures`: the refusals that stop a child or the threshold being used. It
  has these kinds of entry: `"<split> child '<name>': nonFiniteOutput at
  record N"` for a child with a non-finite output, `"<split> child '<name>':
  parent nonFiniteOutput at record N (<split> split)"` for every child on a split
  where the parent's output is non-finite (the report is still written), and `"threshold not fittable:
  <why>"` when `theta` cannot be fitted. Degenerate reasons and
  `collapsedChild` are not listed here. They appear on each child's entry under
  `splits`.

## Flow

```mermaid
flowchart TD
    A[Read manifest] --> B{Valid limits and paths?}
    B -- no --> X[Refuse, naming the field]
    B -- yes --> C[Check child and corpus widths]
    C --> D[Score calibration split]
    D --> E[Fit theta on calibration only]
    E --> F[Score validation split]
    F --> G[Apply gates to validation]
    G --> H[Write report atomically]
    H --> I{Criterion met?}
    I -- yes --> J[Report: met, still report-only]
    I -- no --> K[Report: not met or notEvaluable]
```

The flow ends in a report either way. Nothing in it touches the creatures or the
run path.
