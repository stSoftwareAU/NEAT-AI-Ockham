# Where pruning rewrites live (Issue #172)

Ockham does not own the semantics of a structural rewrite. The canonical
pruning engine lives in
[NEAT-AI-core](https://github.com/stSoftwareAU/NEAT-AI-core), so every member of
the NEAT-AI family prunes a creature the same way, and a rewrite bug is fixed
once rather than in each optimiser that grew its own copy.

This document is the checked-in design reference for that decision. The
decision itself was taken in Ockham issue #172; the engine was built under the
core project [stSoftwareAU/NEAT-AI-core#587](https://github.com/stSoftwareAU/NEAT-AI-core/issues/587)
(parity fixtures #588, fixed-point cleanup and constant invariants #589,
hidden-neuron pruning #590, synapse pruning with `IF`/typed-role rewrites #591,
WASM exposure #592). Ockham's own consumption of that engine — and the
retirement of the duplicate rewrite code it still carries — is issue #182.

Ockham issues #173 to #180 remain useful as scenario and design history. They
are superseded by the core project: none of them is an Ockham implementation
task any more.

## The boundary

```mermaid
flowchart LR
    subgraph Ockham
        A[sweep: visit a hidden neuron or a synapse] --> B[choose a candidate]
        B --> C[screen on a corpus sample]
        C --> D[full-corpus scorer]
        D --> E[accept or discard]
    end
    subgraph NEAT-AI-core
        R["prune_neuron / prune_synapse<br/>rewrite + cleanup to a fixed point"]
        R --> V["validated creature<br/>or PruneError — never both"]
    end
    B -->|"request: creature + target + statistics"| R
    V -->|"PruneResult, or no candidate at all"| C
```

Ockham decides **what to try and whether it was worth it**. Core decides
**what the graph becomes**. A candidate that core will not build is not a
candidate: nothing partially rewritten is passed on to be screened or scored.

## The preserved design principles

These are the principles the engine was built to, recorded here so they survive
the issue thread that stated them:

- hidden neurons and synapses are pruning candidates;
- observation/input and output neurons are protected from direct deletion;
- constants are support nodes, not direct optimisation targets;
- at most three constants, all bias=1; remove them automatically when unreferenced;
- graph rewrites must run to a valid fixed point before screening/scoring;
- invalid candidates never reach the scorer;
- `validation-failed` after a supported prune is a rewrite-engine bug, not a
  normal outcome;
- Ockham owns candidate choice/scoring, not structural-rewrite semantics.

Each one is enforced by named structure in core rather than by convention, and
`ockham/tests/pruning_ownership.rs` asserts them against the sibling neat-core
this repository compiles against — so a core release that dropped one fails
here rather than in a reviewer's memory:

| Principle | What enforces it in core |
|---|---|
| Hidden neurons and synapses are candidates | `prune_neuron` and `prune_synapse` — a synapse is named by its `(from, to, role)` `SynapseKey`, so an edge out of an observation is a candidate even though the observation neuron is not. |
| Observation and output neurons are protected | `PruneError::Protected { kind }`, with `ProtectedKind::Observation` / `ProtectedKind::Output`. |
| Constants are support nodes | `ProtectedKind::Constant` — a constant is removed as dead structure by the cleanup once nothing references it, never as a requested target. |
| At most three, all bias 1 | `MAX_SUPPORT_CONSTANTS` (3) and `SUPPORT_CONSTANT_BIAS` (1.0), applied by `cleanup_creature`. |
| A valid fixed point before screening | `cleanup_creature` runs to a fixed point; `PruneResult::passes` reports how many passes it took. |
| Invalid candidates never reach the scorer | Every `PruneError` variant returns **no** creature; a returned `PruneResult` carries a canonical, validated one. |

## What this means for Ockham today

Issue #182 cut the first three capabilities over and deleted the Ockham
implementation of each in the same migration:

| Capability | Where it lives now |
|---|---|
| Hidden-neuron prune | `prune::prune_hidden_neuron` → core `prune_neuron`. `ablation::ablate_mean` is gone. |
| Neighbourhood group cut (#108) | `prune::prune_hidden_group` → core `prune_neuron`, one member at a time against the creature the last request returned. `ablation::ablate_group` is gone. |
| Synapse prune (#133, #135) | `prune::prune_edge` → core `prune_synapse`, naming the `(from, to, role)` triple. `ablation::ablate_synapse` is gone. |

Ockham keeps what core does not own: which candidate to try, the statistics it
measures, the sampled screen, the full-corpus scoring, bundling and replay, the
reports — and the candidate kinds that are not prunes at all (the exact identity
collapse, the correlated merge, the Issue #103 constant substitution). Those
three still carry their own rewrites through `ablation::cleanup_cascade`; they
are the capabilities not yet migrated, and the same rule applies to them when
their turn comes.

Core's report travels with every candidate it built: `prune::PruneDetail` is
written into the sweep candidate, the screened-out record and the telemetry row,
so a run's evidence says whether a rewrite was exact or approximate, what its
cascade took, which `IF` structure it rewrote, and which targets it left
uncompensated.

The rule for new work is the simple half of the boundary: **a
change to what a prune does structurally belongs in core**, and a change to
which candidate is tried, how it is screened, scored, bundled, replayed or
reported belongs here. If core cannot yet express a rewrite Ockham needs,
improve core rather than growing a second implementation in Ockham.

## `validation-failed`, before and after the migration

[docs/blocked-reasons.md](blocked-reasons.md) counts `validation-failed` as a
blocked visit: a candidate was built, NEAT-AI-core `creature.validate()`
rejected it, and the razor failed closed. That is the right reading of
**Ockham's own** rewrites, which are known to be approximate about shapes they
do not model.

It is not the right reading of a core prune. Once a capability has moved to
core, a `validation-failed` following a *supported* core prune is a bug in the
rewrite engine: core promised a canonical validated creature or no creature at
all. Raise it in NEAT-AI-core with the creature and the request that produced
it — do not count it and move on.
