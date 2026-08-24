# NEAT-AI-Ockham

> **Every neuron must earn its keep — prune freely, trust only the scorer.** ✂️🧠

NEAT-AI-Ockham is an isolated experimental Rust optimiser for already-fit
[NEAT-AI](https://github.com/stSoftwareAU/NEAT-AI) creatures.

It starts with the current fittest creature and asks a deliberately narrow
question:

> **Can we make an already highly evolved creature fitter by removing structure
> that no longer earns the cost of keeping it?**

The motivating production creature has been evolved for years and contains more
than 3,000 neurons. Ockham does **not** replace normal evolution, redesign the
network, or retrain it from scratch. It takes a clone of a known-good incumbent,
tries small pruning/simplification experiments, and lets the existing
[NEAT-AI-scorer](https://github.com/stSoftwareAU/NEAT-AI-scorer) decide whether
anything is genuinely better.

Ockham is intentionally expendable and isolated. If the experiment never finds
a useful improvement, that is a valid result. If it does improve the fittest
creature, excellent. The production evolutionary system must not depend on
Ockham being correct.

## Core principle

```text
fittest creature
      │
      ├── full authoritative baseline score
      │
      ├── full-corpus hidden-neuron activation statistics
      │
      ▼
seeded random permutation of hidden neurons
      │
      ▼
batches of ~100 single-neuron ablations
      │
      ├── compensate downstream biases using mean activation
      ├── exact deterministic cleanup where possible
      ├── cascade dead structure
      └── creature.validate()
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
      ├── no improvement → keep incumbent, continue sweep
      └── improvement    → new experimental incumbent
                              │
                              └── recompute statistics and restart
```

The default wall-clock budget is **45 minutes**. The current production creature
has roughly 3,200 hidden neurons, so the first implementation deliberately tries
the simple experiment before inventing clever search heuristics: approximately
100 candidates per sampled scorer batch means a complete sweep may only require
~30–35 screens if reality is kind.

## The Ockham rule

Ockham may **approximate when proposing a candidate**. It must never approximate
when judging one.

Removing a neuron by replacing its behaviour with its average observed
activation is deliberately approximate. Cascading dead-code deletion and some
IDENTITY simplifications can be mathematically exact. Neither distinction gives
a candidate any authority: only the full-corpus NEAT-AI-scorer can accept a new
incumbent.

## Related repositories

- [NEAT-AI](https://github.com/stSoftwareAU/NEAT-AI) — TypeScript evolutionary
  trainer and the source of the mature creatures Ockham experiments on.
- [NEAT-AI-core](https://github.com/stSoftwareAU/NEAT-AI-core) — canonical Rust
  creature/network implementation. Ockham uses it for loading, inference,
  structural editing and `creature.validate()`.
- [NEAT-AI-scorer](https://github.com/stSoftwareAU/NEAT-AI-scorer) — the
  authoritative judge. Sample scoring may screen candidates; only full-corpus
  scoring may accept one.
- [NEAT-AI-Forests](https://github.com/stSoftwareAU/NEAT-AI-Forests) — sibling
  experiment that searches for useful structure to **add** to a mature creature.
  Ockham asks the opposite question: what can safely be removed?
- [NEAT-AI-Lamarck](https://github.com/stSoftwareAU/NEAT-AI-Lamarck) — sibling
  experiment that uses acquired statistical/backpropagation information to
  propose heritable improvements.

Ockham follows the Rust workspace, quality, journalling and scorer-gating
conventions of the existing Rust NEAT-AI family where practical, but remains a
separate experiment.

## Scope and constraints

Version 1 is deliberately conservative:

- **Rust only.** No TypeScript implementation path.
- **Forward-only creatures only.** Recurrent/self-connected semantics make
  constant substitution and cascading removal much harder to reason about and
  are out of scope initially.
- **Source creature is immutable.** Every candidate is produced from a clone of
  the current experimental incumbent.
- **Structure changes must validate.** Every complete candidate calls
  NEAT-AI-core `creature.validate()` before it can reach the scorer.
- **Sample scores only screen.** The default screen uses about 5% of records and
  batches about 100 candidate creatures with the incumbent evaluated in the
  same sample context.
- **Every sampled winner is interesting.** Initially, every candidate that beats
  the sampled incumbent is promoted rather than imposing a top-K cap.
- **Full scorer is king.** The candidate with the highest authoritative score
  wins, whether it is a single removal or an ugly multi-neuron bundle.
- **45-minute default budget.** A currently fittest creature is perishable while
  normal evolution continues on other machines.

## Phase 1 — establish the incumbent

1. Load the supplied creature through NEAT-AI-core.
2. Require `forwardOnly: true`.
3. Validate and round-trip it.
4. Keep the supplied file byte-for-byte immutable and work from a private copy.
5. Score the incumbent over the full training corpus with NEAT-AI-scorer.
6. Record creature checksum, corpus identity, scorer/version/configuration,
   authoritative score/error and structural counts.

Any validation/scorer/parity failure aborts the experiment rather than guessing.

## Phase 2 — activation statistics

Run the current incumbent over the **complete training corpus** and accumulate
statistics for every hidden neuron. Start with:

- mean post-activation value — required for the ablation substitution;
- variance / standard deviation;
- mean absolute activation;
- minimum and maximum activation;
- record count.

Use bounded-memory streaming and `f64` accumulators even where inference values
are `f32`. Integrate accumulation into the inference pass rather than storing
per-record neuron activations.

The additional statistics are not authority. They are cheap evidence for future
"dirty tricks" such as nearly-constant, almost-always-zero or unusually stable
neurons.

## Phase 3 — approximate single-neuron ablation

For a selected hidden neuron `i`, let its measured mean post-activation be
`mean_i`. For every ordinary downstream weighted connection `i -> j` with weight
`w_ij`, propose removing `i` while compensating the downstream bias:

```text
bias_j' = bias_j + mean_i * w_ij
```

This preserves the removed neuron's **average contribution** to the downstream
pre-activation, not its record-by-record behaviour. It is therefore an
experimental approximation, not an algebraic simplification.

Typed/aggregate synapses must only be transformed when NEAT-AI-core can prove
that the same substitution semantics apply. Unsupported structural cases are
skipped and journalled rather than guessed.

## Phase 4 — deterministic cleanup cascade

After a proposed removal, repeatedly simplify consequences until stable:

### Hidden neuron with no outgoing synapses

It cannot influence an output. Remove it. This may make upstream hidden neurons
newly useless, so continue recursively.

### Hidden neuron with no incoming synapses

For a supported ordinary activation neuron, its output is constant:

```text
constant = squash(bias)
```

Fold each ordinary outgoing contribution into the downstream bias:

```text
bias_j' = bias_j + constant * w_ij
```

then remove the constant neuron and continue the cleanup cascade.

Only transformations whose semantics are understood exactly are folded. Any
activation/aggregate/synapse type whose constant behaviour cannot be established
safely remains untouched.

The entire requested removal + cleanup cascade is an atomic operation on a
clone. Intermediate states do not need to be valid; the completed candidate
must pass `creature.validate()`.

The journal records both the **requested neuron** and every neuron/synapse
actually removed by the cascade.

## Phase 5 — exact IDENTITY collapse

IDENTITY neurons deserve a separate exact simplification path.

For:

```text
y = bias_y + Σ(x_k * a_k)
```

and an ordinary outgoing connection `y -> z` with weight `b`, eliminating `y`
can be exact by applying:

```text
bias_z += bias_y * b
x_k -> z weight += a_k * b
```

Existing parallel connections should be merged by adding weights. With multiple
inputs and outputs this can create `incoming × outgoing` connections, so an
IDENTITY collapse is only attractive when its resulting NEAT growth cost is
lower (or when explicitly being tested as a candidate).

The TypeScript `Creature.compact()` implementation contains related historical
simplifications and is useful as a reference/control, but Ockham should implement
canonical Rust transformations through NEAT-AI-core rather than depend on the
TypeScript optimiser.

## Phase 6 — sampled screen

Create a seeded random permutation of hidden-neuron UUIDs. This gives random
search order without replacement and guarantees eventual coverage if the run is
long enough.

Default first experiment:

- candidate batch: `100` removals;
- scorer sample rate: `0.05`;
- candidate and incumbent use the **same sample phase/context**;
- promote **every** candidate whose sampled score is greater than the sampled
  incumbent by the configured threshold;
- validation failures and unsupported transformations are recorded and skipped.

The seed and permutation state are journalled for replay.

## Phase 7 — authoritative promotion and bundles

Every sampled winner is scored over the complete corpus. The same authoritative
cohort must contain the incumbent so scorer configuration/data differences cannot
masquerade as improvement.

Also construct a small number of combination candidates from sampled winners.
A useful first strategy is to rank removals by sampled delta and full-score:

```text
all individual sampled winners
best 2 together
best 4 together
best 8 together
best 16 together
all sampled winners together
```

Only prefixes that exist are generated, and duplicate bundles are removed. Each
bundle is built independently from the same incumbent and must apply its complete
cleanup/validation rules.

Interactions are expected: ten individually promising removals may be bad when
combined. Never add predicted deltas together and call that evidence.

The **highest full-corpus score wins**, regardless of whether it is an individual
or bundle. No elegance bonus.

## Phase 8 — iterative 45-minute loop

When the full scorer verifies an improvement:

1. atomically write the candidate as the experimental incumbent;
2. write `best.json` and preserve an intermediate under `winners/`;
3. invalidate all incumbent-specific activation statistics and sweep state;
4. recompute full-corpus activation statistics;
5. create a new seeded/randomised sweep for the new topology;
6. continue until the wall-clock budget expires.

A successful run therefore may never finish testing every neuron of the opening
creature. That is intentional: once the incumbent changes, stale activation
statistics belong to a creature that no longer exists.

## Outputs

Planned outputs follow the sibling experiments:

| Path | Purpose |
|---|---|
| `best.json` | best full-scorer-verified creature found during the run |
| `experiments.jsonl` | append-only experiment/candidate journal |
| `winners/` | every accepted intermediate incumbent |
| `workspace/incumbent.json` | private immutable copy of current run incumbent |
| activation-statistics cache | versioned by creature checksum + corpus identity |

Record at least:

- opening/current creature checksum;
- scorer and corpus identity;
- score/error before and after;
- sampled score/delta separately from authoritative score/delta;
- neurons/synapses and growth cost before/after;
- requested neuron UUID;
- mean activation and other recorded statistics;
- exact downstream bias compensation;
- every cascaded removal/constant fold/IDENTITY collapse;
- candidate validation result/reason;
- bundle membership;
- timings by activation scan, mutation, screen and full scoring;
- random seed/permutation position.

It matters whether a winner has lower prediction error, or merely approximately
the same error with enough structural cost removed to increase total score.
Report both.

## Safety invariants

1. **Never modify the supplied creature in place.**
2. **Only forward-only creatures are accepted in v1.**
3. **Every candidate starts from a known scorer-verified incumbent clone.**
4. **Every completed structural candidate must pass NEAT-AI-core
   `creature.validate()`.**
5. **Unknown activation/synapse/aggregate semantics are skipped, never guessed.**
6. **Mean-activation substitution is explicitly approximate.**
7. **Sample scoring can reject/rank/promote, but cannot accept.**
8. **Only a full-corpus NEAT-AI-scorer result may replace the incumbent.**
9. **Scorer failure means no winner. Fail closed.**
10. **After every accepted structural change, activation statistics are stale and
    must be recomputed.**
11. **A bundle is judged as a complete creature; individual deltas are never
    assumed additive.**
12. **`best.json` may never be worse than the opening full-scorer baseline.**

## What success looks like

The useful metric is not how many neurons Ockham deletes. It is:

> **scorer-verified improvement per wall-clock hour on a mature production
> creature.**

A smaller network is interesting. A smaller network with a better authoritative
score is the experiment succeeding.

And if no neuron can be removed profitably, Ockham has still answered a useful
question about how efficiently evolution is already using its structure.
