# NEAT-AI-Ockham

[![NEAT-AI-Ockham social preview](https://raw.githubusercontent.com/stSoftwareAU/NEAT-AI/Develop/docs/brand/social-previews/neat-ai-ockham.png)](https://github.com/stSoftwareAU/NEAT-AI/blob/Develop/docs/brand/social-previews/neat-ai-ockham.png)

> **Every neuron must earn its keep — prune freely, trust only the scorer.** 🪒🧠

NEAT-AI-Ockham is an isolated experimental Rust optimiser for already-fit
[NEAT-AI](https://github.com/stSoftwareAU/NEAT-AI) creatures.

It asks a deliberately simple question:

> **Can an already highly evolved creature become fitter by removing structure
> that no longer earns the cost of keeping it?**

Ockham does not replace ordinary NEAT evolution and it does not redesign a
network from scratch. It starts from a known-good creature, removes or simplifies
small pieces of structure, and lets the existing
[NEAT-AI-scorer](https://github.com/stSoftwareAU/NEAT-AI-scorer) decide whether the
result is actually better.

The central hypothesis is cumulative: **many individually tiny, scorer-verified
pruning wins may compound into a material improvement**.

## Current state

Ockham is now an implemented experimental optimiser rather than a project plan.
The current Rust implementation includes:

- immutable loading and validation of a forward-only incumbent;
- authoritative full-corpus baseline scoring through `NEAT-AI-scorer`;
- full-corpus hidden-neuron activation statistics;
- mean-activation neuron ablation with downstream bias compensation;
- recursive removal/folding of newly redundant structure;
- exact, cost-aware `IDENTITY` neuron collapse;
- seeded random-without-replacement neuron sweeps;
- default batches of 100 candidates screened on a 5% scorer sample;
- full-corpus scoring of every sampled winner plus grouped pruning bundles;
- iterative acceptance of even tiny full-scorer local wins;
- a default 45-minute optimisation loop with append-only journalling;
- fresh re-entry comparison against a supplied current global champion;
- `population-candidate.json` only when Ockham wins that frontier comparison;
- a `report` command for cumulative pruning economics;
- normal Rust CI, security and quality gates.

The remaining work is experimental refinement: measure what works on mature
creatures and only then decide whether smarter pruning order or other dirty
tricks improve the rate of discovery.

## Application scope

NEAT-AI and its public `NEAT-AI-*` subprojects are intentionally
**application-agnostic**.

They provide general neural-network evolution, inference, scoring and optimisation
techniques. Specific downstream applications belong outside these public
libraries. Keeping that boundary clear makes the public projects useful for many
different problem domains without coupling their design or documentation to one
particular use case.

## How Ockham works

```text
known-good creature
        │
        ├── immutable clone
        ├── full authoritative baseline score
        └── full-corpus hidden-neuron activation statistics
        │
        ▼
seeded random permutation of hidden neurons
        │
        ▼
~100 pruning candidates
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
        ├── no local improvement → continue sweep
        └── local improvement    → new Ockham incumbent
                                     │
                                     ├── keep even a tiny verified win
                                     ├── recompute activation statistics
                                     └── restart on the new topology
```

At re-entry time, Ockham's best can optionally be compared with the **latest
external/global champion** in a fresh same-call full-corpus scorer comparison.
Only a genuine frontier win is emitted as population-ready.

## Two kinds of winning

Ockham deliberately distinguishes two different things.

### Local Ockham win

A candidate beats the current Ockham incumbent using the authoritative
full-corpus scorer.

That is enough to keep it and continue pruning from it. The improvement may be
very small; small verified wins are the stepping stones Ockham is trying to
accumulate.

### Population win

The final Ockham result beats the latest current global champion when re-entry is
attempted.

Normal evolution and other experiments may move the frontier while Ockham is
running, so beating the creature Ockham started from is not automatically enough
to re-enter the population.

Conceptually:

```text
Ockham cumulative gain = OckhamBest - OckhamOpeningParent
frontier movement       = LatestGlobalChampion - OckhamOpeningParent
population headroom     = OckhamBest - LatestGlobalChampion
```

Positive authoritative population headroom is required for re-entry.

## The Ockham rule

Ockham may **approximate when proposing a candidate**. It must never approximate
when judging one.

Removing a neuron by replacing its behaviour with its measured average activation
is deliberately approximate. Cascading dead-code removal, constant folding and
some `IDENTITY` simplifications can be mathematically exact. Neither distinction
makes a candidate trustworthy by itself.

**The scorer is the authority.**

A second rule matters just as much:

> **A tiny genuine local win is a stepping stone, not a failure.**

## Mean-activation ablation

For hidden neuron `i`, let its measured mean post-activation be `mean_i`. For each
supported ordinary downstream connection `i -> j` with weight `w_ij`, Ockham
proposes:

```text
bias_j' = bias_j + mean_i * w_ij
```

and removes neuron `i`.

This preserves the removed neuron's **average downstream contribution**, not its
record-by-record behaviour. It is therefore a proposal mechanism, not a proof of
equivalence.

Unsupported aggregate or typed-synapse semantics are skipped rather than guessed.

## Exact cleanup

After a proposed removal, Ockham repeatedly applies safe deterministic cleanup.

A hidden neuron with no outgoing synapses cannot affect an output and is removed.
This can recursively make upstream structure redundant.

A supported hidden neuron with no incoming synapses is constant. Its constant
output can be folded into downstream biases before removing it.

For a suitable `IDENTITY` neuron:

```text
y = bias_y + Σ(x_k * a_k)
```

with downstream weight `b`, Ockham can exactly replace it with:

```text
bias_z += bias_y * b
x_k -> z weight += a_k * b
```

Parallel synapses are merged. Automatic `IDENTITY` collapse is only attractive
when the resulting NEAT growth cost is lower.

Every completed structural candidate must pass NEAT-AI-core
`creature.validate()` before it reaches the scorer.

## Sampling and authoritative promotion

The current defaults are deliberately simple:

```text
candidate batch       100
screen sample rate    0.05
promotion policy      every sampled winner
run budget            2700 seconds
minimum full win      1e-6
```

The incumbent and all candidates in a sampled screen use the same scorer sample
context. Sampling may reject or promote candidates; it can never accept one.

Every sampled winner is full-scored. Ockham also tries small grouped prefixes of
promising removals because individual pruning effects are not assumed additive.

The highest strict full-corpus improvement becomes the next Ockham incumbent,
even if that improvement is tiny.

## Usage

```bash
neat_ai_ockham <creature.json> <training-data-dir> [OPTIONS]
neat_ai_ockham report <experiments.jsonl> [...]
neat_ai_ockham --help
```

Common options:

| Flag | Default | Purpose |
|---|---:|---|
| `--timeout-seconds` | `2700` | Wall-clock optimisation budget. |
| `--candidates` | `100` | Candidates per sampled sweep batch. |
| `--screen-sample-rate` | `0.05` | Sample rate used only for screening; `0` disables it. |
| `--screen-threshold` | `0` | Sampled Δscore required for promotion. |
| `--min-improvement` | `1e-6` | Strict authoritative improvement required locally. |
| `--seed` | drawn | Reproducible random sweep seed. |
| `--max-experiments` | none | Optional experiment cap in addition to timeout. |
| `--scorer` | `rust_scorer` | NEAT-AI-scorer binary. |
| `--scorer-arg` | none | Extra scorer argument; repeatable. |
| `--global-champion` | none | Latest champion JSON for the re-entry comparison. |
| `--output-dir` | `.` | Output workspace. |

The supplied creature is never modified in place.

## Outputs

| Path | Purpose |
|---|---|
| `best.json` | Best authoritative local Ockham result found during the run. |
| `experiments.jsonl` | Append-only experiment journal. |
| `winners/` | Accepted intermediate Ockham incumbents. |
| `workspace/` | Isolated run state, baseline and statistics caches. |
| `population-candidate.json` | Written only after beating the supplied current global champion. |

`best.json` remains useful even when the moving frontier has already passed it;
it records a genuine pruning result against its own lineage. It is not presented
as population-ready unless the fresh frontier comparison also succeeds.

## Reporting

```bash
neat_ai_ockham report path/to/experiments.jsonl
```

The report focuses on the experiment Ockham actually cares about: whether many
small verified cuts compound into useful improvement. Journals preserve per-step
and cumulative scorer deltas, accepted pruning steps, timing and structural
changes.

Useful measures include:

- cumulative scorer improvement from the opening parent;
- distribution of individual accepted win sizes;
- neurons and synapses removed;
- growth-cost reduction;
- sampled-screen false positives;
- time spent in activation analysis, screening and full scoring;
- individual versus bundled winners;
- population headroom when a global-champion comparison was performed.

## Safety invariants

1. The supplied creature is never modified in place.
2. Version 1 accepts forward-only creatures only.
3. Every candidate starts from a known authoritative incumbent clone.
4. Every completed structural candidate must pass `creature.validate()`.
5. Unknown activation/synapse/aggregate semantics are skipped, never guessed.
6. Mean-activation substitution is explicitly approximate.
7. Sample scoring can screen candidates but cannot accept them.
8. Only full-corpus NEAT-AI-scorer results may replace the Ockham incumbent.
9. Scorer failure means no winner; Ockham fails closed.
10. Activation statistics are recomputed after an accepted topology change.
11. Bundle deltas are never assumed additive.
12. A local Ockham win and a current-global-frontier win are reported separately.

## Related repositories

- [NEAT-AI](https://github.com/stSoftwareAU/NEAT-AI) — evolutionary neural-network library and trainer.
- [NEAT-AI-core](https://github.com/stSoftwareAU/NEAT-AI-core) — canonical Rust creature/network implementation.
- [NEAT-AI-scorer](https://github.com/stSoftwareAU/NEAT-AI-scorer) — authoritative scorer used by Ockham.
- [NEAT-AI-Forests](https://github.com/stSoftwareAU/NEAT-AI-Forests) — experimental search for useful structure to add.
- [NEAT-AI-Lamarck](https://github.com/stSoftwareAU/NEAT-AI-Lamarck) — experimental acquired-information optimisation.

## Development

The project is pure Rust and expects sibling clones of `NEAT-AI-core` and
`NEAT-AI-scorer` where required by the local development setup.

```bash
./quality.sh < /dev/null
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the local and CI quality gates.
Ockham commit messages use the **🪒** prefix.

## What success looks like

The useful metric is not simply how many neurons Ockham deletes. It is:

> **cumulative scorer-verified improvement per wall-clock hour on an already-fit
> creature.**

A smaller creature is interesting. A smaller creature that is fitter is much
more interesting. If enough tiny verified improvements accumulate to catch and
beat a moving evolutionary frontier, Ockham has done exactly what it was built to
do.

If nothing can be removed profitably, that is also useful evidence about how
efficiently evolution is already using its structure.
