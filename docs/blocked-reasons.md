# Blocked neurons, by reason — and what to do about each (Issue #103)

`blocked` counts the hidden neurons the sweep has **visited** and could propose
no cut for. It has never meant *not pruneable forever*: it means the current
proposal mechanism does not know how to test that neuron safely.

Until Issue #103 it was one number, and one number cannot be attacked. Every
blocked visit now carries a **reason code**, the code rides on the screen record
in `screens/<host>.jsonl`, and every reporting surface counts by it.

## The codes

| Code | What it means | Can the razor build a candidate? |
|---|---|---|
| `aggregate-squash` | The neuron, or something a bias fold would touch, uses an aggregate squash (`IF`, `MEAN`, `MINIMUM`, …) that does not sum its inputs. | **Yes, since #103** — constant substitution. The code is still recorded where no substitution was reachable, chiefly an IDENTITY collapse blocked by an aggregate target with no activation statistic to fall back on. |
| `unsafe-topology` | The transform cannot treat the neuron as an ordinary hidden unit — it is not hidden, or not on the creature at all. Since #103 the typed-synapse case this code used to cover is proposed rather than blocked, because constant substitution keeps the role-carrying edge. Since #108 an empty group cut is counted here too: a group with no members would remove nothing, and a candidate identical to the incumbent is refused rather than scored. | Case by case; the recorded cases are structural faults, not categories to build a path for. |
| `missing-activation` | No finite sampled activation statistic for the neuron. | No — there is no value to substitute. See below. |
| `validation-failed` | A candidate was built and NEAT-AI-core `creature.validate()` rejected it. | No — failing closed is the point. |
| `no-output-path` | The neuron feeds nothing, so no candidate could be built around it. | Not reachable on a valid incumbent (rule 18). |
| `other` | An explicit reason outside the codes above, including a code written by a newer binary than the one reading it. | Case by case. |
| `unrecorded` | The record was filed before #103 and carries no reason. | Unknown — it is counted separately rather than guessed at. |

The counts are over UUIDs and **sum to the `blocked` total exactly**, so the
breakdown is a partition of the blocked population rather than a sample of it.

## The dominant category, and the path built for it

On a forest-heavy GRQ creature roughly four hidden neurons in five feed an
aggregate squash or carry a typed synapse, so `aggregate-squash` (with
`unsafe-topology` behind it) dominated the blocked population. That is the
measurement #93 recorded and the code paths agree with it: before #103 every one
of those visits ended at the aggregate or typed-synapse check in
`ablation::ablate_mean`. A run against live GRQ data will now print the figure
under the codes themselves — the `reasons:` line — which is the first time the
split is measured rather than reasoned about. Those were the neurons worth a new
proposal path, and `ockham/src/substitute.rs` is it.

The mean-activation ablation removes the neuron and folds its mean into every
downstream **bias**. That is only valid where the target sums its inputs. An
aggregate target does not sum, and a bias cannot stand in for a synapse role, so
the fold fails closed.

Constant substitution keeps the **edge** and replaces the **source**:

```mermaid
flowchart LR
    subgraph before
        X[input] --> H["h (TANH)"]
        H -->|condition| I[IF]
        I --> O[output]
    end
    subgraph after
        C["h (constant = mean)"] -->|condition| I2[IF]
        I2 --> O2[output]
    end
```

- the hidden neuron becomes a `constant` neuron emitting its measured mean;
- its **incoming** synapses go, and whatever upstream structure that leaves
  feeding nothing cascades away with it — that is the pruning;
- its **outgoing** synapses stay untouched, weights and roles included, so the
  aggregate target still reads a value on the same edge, `MEAN` still averages
  the same arity and an `IF` still has one edge of each role (NEAT-AI-core
  rule 12).

The claim being made is the razor's usual one — *this neuron's output is close
enough to its mean* — and nothing downstream is rewritten. It is not, however,
a *weaker* claim than the bias fold: folding a mean into a summing target is
linear, whereas pinning a `MEAN`, `MINIMUM` or `IF` **condition** input to its
mean can change which branch the target takes. That is why this is a proposal
and not a simplification: the sampled screen and the full-corpus scorer are what
decide whether the approximation held. Where the neuron really is constant, the
substitution is exact.

It remains a **proposal**. Every substituted candidate passes
`creature.validate()`, the sampled screen and the authoritative full-corpus
scorer like any other, and only the scorer accepts a cut.

### What it counts as

A substituted candidate is scored under a third kind, `constant`, beside
`identity` and `ablation`, in `screens/<host>.jsonl` and in the journal. An
accepted one removes a hidden neuron and the structure that fed it, and leaves a
`constant` neuron in its place — so `cut:` counts it as a cut (the hidden neuron
is gone) while `neurons.len()` falls by one less than the cut count. The growth
proxy Ockham reports, `hidden + synapses / 10`, moves the same way the scorer's
own cost of growth does, and the scorer is what accepts the trade.

### What it costs

A blocked visit used to be nearly free: the sweep rejected it before cloning the
creature (#91), so a batch of 100 candidates could walk five hundred neurons.
Every one of those neurons now produces a candidate instead, so a batch reaches
fewer neurons per sampled screen call — the same one screen call per batch, over
more useful work. That is the trade Issue #103 asks for: coverage per hour buys
less, and scorer-verified cuts per hour buys more, because the neurons being
screened are ones nothing was ever going to prune before.

## The categories with no path yet

- **`missing-activation`** — there is no finite measured mean, so neither the
  bias fold nor a constant has a value to stand in for the neuron. A safe path
  would need a different statistic (a median, or a scan that reaches the neuron
  at all), and inventing a substitute value would be exactly the guess the razor
  must not make. Note that NEAT-AI-core clamps every activation to a finite
  range, so this is rare in practice; it is what a visit reports when a cached
  statistics scan does not cover the neuron.
- **`validation-failed`** — a candidate was built and NEAT-AI-core rejected it.
  This is the razor failing closed, and it is reported rather than retried: a
  candidate that cannot validate must never be silently replaced by a different
  transform, because the rejection is information about a shape the razor does
  not model.
- **`no-output-path`** — NEAT-AI-core rejects a hidden neuron with no outgoing
  edge (rule 18), so a validated incumbent never holds one. The code exists so a
  transform that would leave one refuses rather than emitting a candidate that
  cannot validate.

## Where to read the breakdown

| Surface | What it carries |
|---|---|
| `screens/<host>.jsonl` | `blockedReason` per neuron, per screening epoch. |
| `coverage.txt` | The `reasons:` line under `blocked:`, commonest first with each category's share. |
| `coverage.json` | `blockedByReason`, one fixed key per code. |
| `experiments.jsonl` | The `coverage` record carries `blockedByReason` beside `blocked`. |
| `report` | `blockedByReason` and `dominantBlockedReason` for the latest snapshot, plus `blockedEpochs` — one row per screening epoch, holding that epoch's freshest breakdown as counts and as a rendered `reasons` string with each category's share. |

`blockedEpochs` is the historical half: a corpus change opens a new screening
epoch, and the series says whether a category is growing, shrinking, or was
solved by a new proposal path. The rows appear in the order the journals name
the epochs, and a journal that carries a blocked total with no reasons — every
journal written before #103 — has the difference filed as `unrecorded`, so the
sum invariant holds on the replay path too rather than only on the live one.

Every artefact is additive. A `coverage.json`, journal or screen record written
before #103 still deserialises — as no reasons, and as the `unrecorded`
category — and a reason code from a newer host reads as `other` rather than
failing the load of every record beside it.
