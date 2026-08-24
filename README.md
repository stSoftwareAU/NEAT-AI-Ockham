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

Ockham is deliberately isolated while experimenting. Tiny full-scorer-verified
improvements are allowed to become new Ockham incumbents and compound: the core
hypothesis is that **many small cuts may add up to a material improvement**.

Normal evolution and NEAT-AI-Forests continue moving the global frontier while
Ockham runs. Therefore a creature that beats its Ockham parent is a valid local
win, but it is only population-ready after a fresh full-corpus comparison against
the **latest global champion** at re-entry time.

Ockham proposes, accumulates and proves. Normal NEAT-AI owns the population.

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
full-corpus NEAT-AI-scorer vs current Ockham incumbent
        │
        ├── no local improvement → keep incumbent; continue sweep
        └── local improvement    → new Ockham incumbent
                                     │
                                     ├── keep even a tiny verified win
                                     ├── accumulate gain from opening parent
                                     ├── recompute statistics
                                     └── restart sweep

end/re-entry
        │
        ▼
latest global NEAT-AI champion + Ockham best
        │
        ▼
fresh same-call full-corpus scorer comparison
        │
        ├── Ockham loses → preserve local best; not population-ready
        └── Ockham wins  → population candidate
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

A second rule matters just as much:

> **A tiny genuine local win is a stepping stone, not a failure.**

Ockham keeps full-scorer-verified local improvements so they can compound over
multiple pruning steps. Population competitiveness is checked separately against
the moving global frontier.

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

## Usage

```bash
neat_ai_ockham <creature.json> <training-data-dir> [OPTIONS]
neat_ai_ockham --help
neat_ai_ockham --version
```

The supplied creature path is never written to. Training data is a directory of
NEAT-AI `.bin` records, scored by an external `rust_scorer` binary rather than
by Ockham itself.

| Flag | Default | Purpose |
|---|---|---|
| `--timeout-seconds` | `2700` | Wall-clock budget (45 minutes). |
| `--output-dir` | `.` | `best.json`, `experiments.jsonl`, `winners/`, `workspace/`. |
| `--scorer` | `rust_scorer` | NEAT-AI-scorer binary. |
| `--scorer-arg` | _(none)_ | Extra argument passed verbatim to the scorer (repeatable). |
| `--seed` | drawn | RNG seed; printed for replay when optimisation starts. |
| `--candidates` | `100` | Sampled-sweep batch size. |
| `--screen-sample-rate` | `0.05` | Scorer sample used only to screen; `0` disables screening. |
| `--screen-threshold` | `0` | Sampled Δscore required to promote a candidate. |
| `--min-improvement` | `1e-6` | Strict full-corpus improvement required to accept. |
| `--max-experiments` | _(none)_ | Optional cap in addition to the wall-clock budget. |
| `--max-consecutive-scorer-failures` | `3` | Abort after this many consecutive scorer failures. |

Issue #1 only reports this configuration as JSON on stdout. Pruning, screening
and promotion land in later issues; a run must not attempt optimisation until
those stages exist.

## Repository layout

```text
NEAT-AI-Ockham/
├── Cargo.toml                 # workspace
├── ockham/
│   ├── Cargo.toml
│   └── src/
│       ├── main.rs            # CLI
│       ├── lib.rs
│       ├── config.rs          # flags and defaults
│       ├── scorer.rs          # external rust_scorer judge
│       ├── log.rs
│       └── cancel.rs
├── quality.sh
├── rust-toolchain.toml
└── neat-core.expected-version
```

Sibling clones expected beside this repo: `NEAT-AI-core` (path dependency) and
`NEAT-AI-scorer` (the authoritative judge binary).

## Quality gates

```bash
./quality.sh < /dev/null
```

See [CONTRIBUTING.md](CONTRIBUTING.md). The local gate mirrors CI: rustfmt,
clippy `-D warnings`, tests, rustdoc, cargo-deny, shellcheck and spell-check.

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
- **The full scorer is king.** Highest full-corpus score wins locally, regardless
  of whether the delta is tiny or the creature is elegant.
- **Local and population wins are different.** Local wins compound inside Ockham;
  population re-entry requires beating the latest global champion.
- **45-minute default run budget.** The global champion is perishable while
  normal evolution and Forests continue elsewhere.

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
**current Ockham incumbent** in that same scorer context.

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

The highest full-corpus score above the current Ockham incumbent wins. No
preference is given to an individual over a bundle or to an elegant
transformation over an ugly one. A tiny strict full-scorer win is still a win:
it becomes the next Ockham parent so later cuts can build on it.

## Iterative 45-minute loop

The central Ockham hypothesis is cumulative:

> **Can many individually tiny scorer-verified pruning improvements compound into
> a material gain?**

When the full scorer verifies a local improvement:

1. write the winning candidate as Ockham's new experimental incumbent — even if
   the authoritative delta is tiny;
2. write `best.json` and preserve an intermediate under `winners/`;
3. record both the step delta and the **cumulative delta from the opening parent**;
4. record cumulative neurons/synapses/growth cost removed;
5. invalidate all incumbent-specific activation statistics and sweep state;
6. recompute full-corpus activation statistics;
7. create a fresh random sweep for the new topology;
8. continue until the wall-clock budget expires.

A successful run may therefore never finish testing every neuron of its opening
creature. Once topology changes, old activation averages belong to a creature
that no longer exists. The new topology may also expose pruning opportunities
that were not useful on the original parent.

The external/global champion does **not** gate these internal steps. Forests or
normal evolution may move ahead while Ockham accumulates its chain of local wins;
that moving-frontier comparison happens only when Ockham offers its result back.

## Re-entry into the general population

Isolation applies to experimentation. Population re-entry has a stricter,
**moving-frontier** requirement.

At re-entry time:

1. obtain the latest current global NEAT-AI champion;
2. validate Ockham's exact `best.json` and the latest champion;
3. score **both together** over the full canonical corpus using identical
   scorer/cost/data settings;
4. calculate:

```text
Ockham cumulative gain = OckhamBest - OckhamOpeningParent
frontier movement       = LatestGlobalChampion - OckhamOpeningParent
population headroom     = OckhamBest - LatestGlobalChampion
```

5. mark/export Ockham as population-ready only when authoritative population
   headroom is strictly positive by the configured minimum threshold.

A creature that beats its Ockham parent but loses to the latest global champion
remains a valid `best.json`; it simply does not enter the population yet. Do not
discard the experimental result or pretend it is globally competitive.

The global comparison must be fresh and same-call. Do not compare Ockham's score
to a stale score copied from another process.

Ockham v1 should not write directly into a live production population. Instead it
emits an explicit scorer-proven population candidate for the existing NEAT-AI
population path to ingest. No Ockham-specific runtime dependency should be
required once the creature returns to ordinary evolution.

## Outputs

| Path | Purpose |
|---|---|
| `best.json` | best local full-scorer-verified Ockham creature, always preserved |
| `population-candidate.json` | exact Ockham best emitted/marked ready only after beating the latest global champion |
| `population-candidate.meta.json` | opening-parent, frontier, checksum, scorer/corpus and population-headroom provenance |
| `experiments.jsonl` | append-only experiment/candidate journal |
| `winners/` | every accepted local intermediate, including tiny wins |
| `workspace/incumbent.json` | isolated copy of the current run incumbent |
| activation-statistics cache | versioned by creature checksum + corpus identity |

The journal should record score/error, per-step and cumulative deltas,
structure/growth cost, requested and cascaded removals, bias compensation,
exact/approximate transform type, validation result, screen/full results, bundle
membership, timings and random seed/permutation state.

When a frontier comparison is performed, also record the latest global champion
checksum/score/error, frontier movement since the Ockham opening parent, and
final population headroom.

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
8. **Only the full-corpus NEAT-AI-scorer may replace the local Ockham incumbent.**
9. **Tiny strict local wins are retained so they can compound.**
10. **Scorer failure means no winner. Fail closed.**
11. **After an accepted structural change, activation statistics are stale and
    must be recomputed.**
12. **Bundle deltas are never assumed additive.**
13. **`best.json` may never be worse than the opening authoritative baseline.**
14. **A local Ockham winner is not automatically population-ready.**
15. **Population re-entry requires a fresh same-call full-corpus win over the
    latest global champion.**

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
| [#7](https://github.com/stSoftwareAU/NEAT-AI-Ockham/issues/7) | local full scoring + grouped bundles |
| [#8](https://github.com/stSoftwareAU/NEAT-AI-Ockham/issues/8) | cumulative iterative 45-minute optimiser + journal |
| [#9](https://github.com/stSoftwareAU/NEAT-AI-Ockham/issues/9) | fresh global-frontier comparison + population re-entry |
| [#10](https://github.com/stSoftwareAU/NEAT-AI-Ockham/issues/10) | cumulative/frontier reporting + production economics |
| [#11](https://github.com/stSoftwareAU/NEAT-AI-Ockham/issues/11) | smarter ordering only if evidence warrants it |

## What success looks like

Ockham has two useful success measures.

Internally:

> **Can tiny scorer-verified pruning wins compound into significant cumulative
> improvement per wall-clock hour?**

For population impact:

> **Can that accumulated result catch and beat the moving global NEAT-AI
> frontier?**

A smaller network is interesting. A chain of smaller-and-fitter networks is much
more interesting. If the final chain overtakes the latest global champion, it
goes back into ordinary evolution as breeding stock.

And if Ockham improves its own parent but cannot catch a fast-moving Forests /
evolution frontier, that is still useful evidence: it tells us pruning works,
but not yet fast enough to win the race.
