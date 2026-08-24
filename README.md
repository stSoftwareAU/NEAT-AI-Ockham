# NEAT-AI-Ockham

> **Every neuron must earn its keep — prune freely, trust only the scorer.** ✂️🧠

NEAT-AI-Ockham is an isolated experimental Rust optimiser for already-fit
[NEAT-AI](https://github.com/stSoftwareAU/NEAT-AI) creatures.

It asks one deliberately narrow question:

> **Can an already highly evolved creature become fitter by removing structure
> that no longer earns the cost of keeping it?**

The motivating production creature has been evolved for years and contains more
than 3,000 neurons. Ockham does not replace normal evolution, redesign the
network, or retrain it from scratch. It takes a clone of the current fittest
creature, proposes small pruning/simplification experiments, and lets
[NEAT-AI-scorer](https://github.com/stSoftwareAU/NEAT-AI-scorer) decide whether
anything is genuinely better.

Ockham is deliberately isolated while experimenting. If it finds nothing, that
is a valid result. If the **full canonical scorer** says an Ockham creature is
fitter, that creature is legitimate again: it can be checked back into the
normal NEAT-AI population and continue ordinary evolution.

Ockham proposes and proves. Normal NEAT-AI owns the population.

## Core principle

```text
current fittest creature
        │
        ├── immutable clone
        ├── full authoritative baseline score
        └── full-corpus hidden-neuron activation statistics
        │
        ▼
seeded random permutation of hidden neurons
        │
        ▼
~100 single-neuron pruning candidates
        │
        ├── mean-activation bias compensation
        ├── exact deterministic cleanup where possible
        ├── cascading dead-structure removal
        └── NEAT-AI-core creature.validate()
        │
        ▼
NEAT-AI-scorer on ~5% sample
        │
        ▼
every apparent winner
        │
        ├── full-corpus individual candidates
        └── grouped/prefix combinations
        │
        ▼
full-corpus NEAT-AI-scorer
        │
        ├── no improvement → keep incumbent; continue sweep
        └── improvement    → new Ockham incumbent
                                │
                                ├── recompute statistics
                                ├── restart sweep
                                └── eligible for population re-entry
```

The default wall-clock budget is **45 minutes**. With roughly 3,200 hidden
neurons, the first implementation intentionally starts simple: if ~100 candidate
creatures can be screened together cheaply on 5% of the training corpus, a
complete no-win sweep is only about 30–35 scorer batches. We should measure that
before inventing clever search machinery.

## The Ockham rule

Ockham may **approximate when proposing a candidate**. It must never approximate
when judging one.

Replacing a removed neuron's behaviour with its average observed activation is
explicitly approximate. Cascading dead-code deletion, constant folding and some
IDENTITY simplifications can be mathematically exact. Neither distinction grants
any authority: only the full-corpus NEAT-AI-scorer may declare a better creature.

## Related repositories

- [NEAT-AI](https://github.com/stSoftwareAU/NEAT-AI) — evolutionary trainer and
  owner of the normal population.
- [NEAT-AI-core](https://github.com/stSoftwareAU/NEAT-AI-core) — canonical Rust
  creature/network implementation used for loading, inference, structural edits
  and `creature.validate()`.
- [NEAT-AI-scorer](https://github.com/stSoftwareAU/NEAT-AI-scorer) — authoritative
  judge. Sample scoring may screen; only full-corpus scoring may accept.
- [NEAT-AI-Forests](https://github.com/stSoftwareAU/NEAT-AI-Forests) — sibling
  experiment asking what useful structure can be **added** to a mature creature.
- [NEAT-AI-Lamarck](https://github.com/stSoftwareAU/NEAT-AI-Lamarck) — sibling
  experiment using statistical/backpropagation information to propose heritable
  improvements.

Ockham follows the Rust workspace, quality, journalling and scorer-gating
conventions of the existing Rust NEAT-AI family where practical, but remains an
independent experiment.

## Version-1 constraints

- **Pure Rust.**
- **Forward-only creatures only.** Recurrent/self-connected networks are out of
  scope initially.
- **The supplied creature is immutable.** Every candidate starts from a clone of
  a scorer-verified incumbent.
- **Every structural candidate must validate.** Call NEAT-AI-core
  `creature.validate()` after the complete atomic transformation.
- **Unknown semantics are skipped.** Aggregate/typed synapses are only transformed
  when their substitution semantics are explicitly proven and tested.
- **Sample scores only screen.** Default screen: about 100 candidates over 5% of
  records, with the incumbent in the same sample cohort.
- **Promote every sampled winner initially.** No top-K cap until evidence says it
  is needed.
- **The full scorer is king.** Highest full-corpus score wins, regardless of
  whether the creature is elegant.
- **45-minute default run budget.** The global champion is perishable while
  normal evolution continues elsewhere.

## Full-corpus activation statistics

For the current incumbent, stream the entire training corpus and accumulate for
every hidden neuron:

- mean post-activation value;
- variance / standard deviation;
- mean absolute activation;
- minimum and maximum activation;
- record count.

Use bounded memory and `f64` accumulators even where inference is `f32`. Do not
store all per-record activations. Cache only against the exact creature checksum,
corpus identity and statistics-format version.

The mean is required for the first pruning proposal. The extra statistics are
useful evidence for later dirty tricks such as nearly-constant or almost-always-
zero neurons, but none is an acceptance metric.

## Approximate single-neuron ablation

For selected hidden neuron `i`, let its measured mean post-activation be
`mean_i`. For each supported ordinary downstream connection `i -> j` with weight
`w_ij`, propose removing `i` while compensating the downstream bias:

```text
bias_j' = bias_j + mean_i * w_ij
```

This preserves the neuron's **average contribution** to the downstream
pre-activation, not its record-by-record contribution. It is therefore an
experimental approximation.

## Deterministic cleanup cascade

After a proposed removal, repeatedly simplify exact consequences until stable.

### No outgoing synapses

A hidden neuron with no outgoing synapses cannot affect an output. Remove it.
Its removal may make upstream hidden neurons newly useless, so recurse.

### No incoming synapses

Where the activation semantics are supported, a hidden neuron with no incoming
synapses is constant:

```text
constant = squash(bias)
```

Fold each ordinary outgoing contribution into the downstream bias:

```text
bias_j' = bias_j + constant * w_ij
```

then remove the constant neuron and continue the cascade.

The entire requested removal + cleanup is an atomic transformation on a clone.
Intermediate topology may be temporarily invalid; the final candidate must pass
`creature.validate()`.

Journal both the **requested removal** and every neuron/synapse actually removed.

## Exact IDENTITY collapse

IDENTITY neurons can sometimes be removed exactly.

For:

```text
y = bias_y + Σ(x_k * a_k)
```

and ordinary outgoing connection `y -> z` with weight `b`:

```text
bias_z += bias_y * b
x_k -> z weight += a_k * b
```

Merge parallel connections by adding their weights. With multiple inputs and
outputs this can create `incoming × outgoing` synapses, so the automatic
simplification should only be emitted when resulting NEAT growth cost is lower.

The historical TypeScript `Creature.compact()` logic is useful as a reference
and control, but Ockham implements canonical Rust transformations through
NEAT-AI-core.

## Sampled sweep

Create a seeded random permutation of eligible hidden-neuron UUIDs. Random order
without replacement gives exploration without accidentally testing the same
neuron repeatedly.

Initial defaults:

```text
candidate batch       100
screen sample rate    0.05
promotion policy      every sampled winner
run budget            2700 seconds
```

The incumbent and all candidates in a screen must use the same scorer sample
phase/context. A sample result can reject or promote; it can never accept.

## Full-corpus promotion and bundles

Every sampled winner gets an authoritative full-corpus score. Include the
incumbent in that same scorer context.

Also rank sampled winners by sample delta and try a few grouped candidates:

```text
best 2 together
best 4 together
best 8 together
best 16 together
all sampled winners together
```

Only generate prefixes that exist and deduplicate equivalent bundles. Each
bundle starts independently from the same incumbent, performs its full cleanup,
validates, and is judged as a complete creature.

Individual improvements are **not** assumed additive. Ten individually useful
removals may be terrible together; that is precisely why the scorer evaluates
the group.

The highest full-corpus score wins. No preference is given to an individual over
a bundle or to an elegant transformation over an ugly one.

## Iterative 45-minute loop

When the full scorer verifies an improvement:

1. write the winning candidate as Ockham's new experimental incumbent;
2. write `best.json` and preserve an intermediate under `winners/`;
3. invalidate all incumbent-specific activation statistics and sweep state;
4. recompute full-corpus activation statistics;
5. create a fresh random sweep for the new topology;
6. continue until the wall-clock budget expires.

A successful run may therefore never finish testing every neuron of its opening
creature. Once topology changes, old activation averages belong to a creature
that no longer exists.

## Re-entry into the general population

Isolation applies to **experimentation**, not to a scorer-proven result.

A winning creature is eligible to re-enter normal NEAT-AI evolution when:

1. it passes NEAT-AI-core `creature.validate()`;
2. it was scored over the full canonical corpus using the same scorer/cost
   configuration as its incumbent;
3. its authoritative score is strictly higher than the incumbent by the
   configured minimum threshold;
4. the exact exported JSON checksum matches the creature that was scored;
5. scorer/corpus/incumbent provenance is recorded.

Ockham v1 should not write directly into a live production population. Instead it
emits an explicit scorer-proven population candidate for the existing NEAT-AI
population path to ingest. No Ockham-specific runtime dependency should be
required once the creature returns to ordinary evolution.

## Outputs

| Path | Purpose |
|---|---|
| `best.json` | best full-scorer-verified creature found during the run |
| `population-candidate.json` | exact scorer-proven creature eligible for normal evolution |
| `population-candidate.meta.json` | checksum + scorer/corpus/provenance for population re-entry |
| `experiments.jsonl` | append-only experiment/candidate journal |
| `winners/` | every accepted intermediate incumbent |
| `workspace/incumbent.json` | isolated copy of the current run incumbent |
| activation-statistics cache | versioned by creature checksum + corpus identity |

The journal should record score/error, structure/growth cost, requested and
cascaded removals, bias compensation, exact/approximate transform type,
validation result, screen/full results, bundle membership, timings and random
seed/permutation state.

It matters whether a winner predicts better or predicts approximately the same
while becoming sufficiently cheaper to raise the total score. Record both
`error` and final `score`.

## Safety invariants

1. **Never modify the supplied creature in place.**
2. **Only forward-only creatures are accepted in v1.**
3. **Every candidate starts from a known full-scorer-verified incumbent clone.**
4. **Every completed structural candidate must pass `creature.validate()`.**
5. **Unknown activation/synapse/aggregate semantics are skipped, never guessed.**
6. **Mean-activation substitution is explicitly approximate.**
7. **Sample scoring can reject/rank/promote, but cannot accept.**
8. **Only the full-corpus NEAT-AI-scorer may replace the incumbent.**
9. **Scorer failure means no winner. Fail closed.**
10. **After an accepted structural change, activation statistics are stale and
    must be recomputed.**
11. **Bundle deltas are never assumed additive.**
12. **`best.json` may never be worse than the opening authoritative baseline.**
13. **Only an exact full-scored winner artefact may be offered back to the normal
    population.**

## Implementation roadmap

The issue list is the deliberately small project plan:

| Issue | Phase |
|---|---|
| [#1](https://github.com/stSoftwareAU/NEAT-AI-Ockham/issues/1) | Rust bootstrap and quality gates |
| [#2](https://github.com/stSoftwareAU/NEAT-AI-Ockham/issues/2) | immutable incumbent + authoritative baseline |
| [#3](https://github.com/stSoftwareAU/NEAT-AI-Ockham/issues/3) | full-corpus activation statistics |
| [#4](https://github.com/stSoftwareAU/NEAT-AI-Ockham/issues/4) | mean-activation ablation + recursive cleanup |
| [#5](https://github.com/stSoftwareAU/NEAT-AI-Ockham/issues/5) | exact cost-aware IDENTITY collapse |
| [#6](https://github.com/stSoftwareAU/NEAT-AI-Ockham/issues/6) | seeded 100-wide / 5% sampled sweep |
| [#7](https://github.com/stSoftwareAU/NEAT-AI-Ockham/issues/7) | full scoring + grouped bundles |
| [#8](https://github.com/stSoftwareAU/NEAT-AI-Ockham/issues/8) | iterative 45-minute optimiser + journal |
| [#9](https://github.com/stSoftwareAU/NEAT-AI-Ockham/issues/9) | verified winner export / population re-entry |
| [#10](https://github.com/stSoftwareAU/NEAT-AI-Ockham/issues/10) | reporting + production economics |
| [#11](https://github.com/stSoftwareAU/NEAT-AI-Ockham/issues/11) | smarter ordering only if evidence warrants it |

## What success looks like

The useful metric is not how many neurons Ockham deletes. It is:

> **scorer-verified improvement per wall-clock hour on a mature creature.**

A smaller network is interesting. A smaller network with a better authoritative
score — good enough to put back into the population and keep evolving — is the
experiment succeeding.

And if no neuron can be removed profitably, Ockham has still answered a useful
question about how efficiently evolution is already using its structure.
