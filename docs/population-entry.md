# Getting Ockham cuts into the general population

Success is not a higher `best.json` on one host. It is a structural prune that
other machines then breed from: **hidden UUIDs we removed stay gone on the
fittest `samples/*.json`**, including files Ockham did not write.

An `ockham` tag on Forests/Lamarck is **not** proof. Tags copy through
crossover. The live Forests champion on 2026-08-26 still carried

`🪒 Ockham · 1 accepts / 2 batches · last: individual · score: 0.388922`

while scoring **0.393902**. That tag is ancestral baggage from a prune ~5e-3
ago. Judging by tags overstates adoption.

## What we measured on GRQ-23

- Forests / Lamarck leap the fittest creature faster than a random Ockham sweep
  discovers a new cut. With `--max-accepts 1` and no cache, each loop paid
  activation + one prune of a **new** champion, then Forests moved on.
- Historical `runs/prod-*/source.json` vs `best.json` diffs found **52 unique**
  successful hidden-UUID cuts. **50 of those UUIDs were still on the next
  Forests champion.** The cuts were not obsolete; we were not replaying them.
- Sequential one-UUID full scores of those 50 would take hours. Combined replay
  (apply every still-present known win, one full-corpus cohort, largest bundle
  first) is the path that can still finish before Forests moves.
- `samples/GRQ-23-ockham.json` at **0.393017** vs live Forests **0.393902** is
  ~8.8e-4 behind. Selection will not prefer offspring of that file. Check-in
  without staying near the frontier is a private notebook that happens to live
  in git.
- Direct structure test (2026-08-26 morning): of 52 known-win UUIDs, **50 were
  still present** on `GRQ-23-forests.json`. Only 2 were already gone. Ockham’s
  published creature had **0 neurons Forests lacked** — it was an older, smaller
  prune (4335 hidden vs Forests 4412), not a parent of the current fittest.
- 23 non-ockham samples carried an `ockham` tag. All showed the same stale
  `score: 0.388922` blurb. Tag inheritance ≠ cut adoption.

## Overnight approach (GRQ-23, 2026-08-26)

Forests is the competitor, not the gate. Tip-check still: beat **this run’s
source** and beat dest `${HOST}-ockham.json`. Do not wait to beat a moving
Forests champion *after* the run — by then it is too late. Instead:

1. Always start from the **current** fittest `samples/*.json` (usually
   `GRQ-23-forests.json`).
2. Combined replay of every still-present known win (prefixes 16/8/4 in the
   same full-corpus cohort). If it accepts, stop (`replay-accepts`) and check
   in immediately. That creature is a prune of *today’s* champion, so its
   score can sit within ~1e-6–1e-5 of the frontier.
3. If the giant bundle misses: probe up to 8 individuals; demote full-corpus
   losers from the success pile; remaining wins stay for the next loop.
4. Cap the whole run at **20 minutes** (`--timeout-seconds 1200`). A 45-minute
   random sweep on a champion Forests has already left is how we get beaten.
   After timeout, loop: pull sampler, take the new fittest, replay again.
5. `--max-accepts 1` still stops **new** search discoveries so a single fresh
   cut can check in the same hour if replay is dry.

Sampler commit subject is the creature `ockham` tag (🪒). Skim for
`replay-bundle` vs `search`.

## How to tell if we actually won

After a check-in, wait for the next Forests/Lamarck sample and ask:

- Are the UUIDs we just removed **absent** from that file?
- Is its score within a small gap of the previous fittest (not 1e-3 behind)?

If the `ockham` tag is present but those UUIDs are back, breeding kept the
label and discarded the prune.

## First competitive check-in (2026-08-26 08:44)

Combined replay on live `GRQ-23-forests.json` (0.393902, 4412 hidden):

- 50 known-win UUIDs still on the incumbent; 47 proposed; one full-corpus
  cohort of 5 creatures (all + prefixes).
- Accepted **8 cuts**, hidden 4412 → 4404, score **0.393928** (Δ +2.54e-5).
- Stopped `replay-accepts` (~13 minutes including activation).
- Checked in [GRQ-sampler `f6d9ed7`](https://github.com/stSoftwareAU/GRQ-sampler/commit/f6d9ed7)
  `samples/GRQ-23-ockham.json` with tag
  `🪒 Ockham · replay-bundle · 8 cuts · score: 0.393928 (+2.54e-5)`.

That file is **ahead of the Forests sample it was pruned from**. Selection can
actually see it. Whether Forests/Lamarck keep those 8 UUIDs gone is the next
proof.

The giant 47-cut bundle did not win; a prefix of 8 did. Combining everything
at once is too rough; prefixes in the same cohort are load-bearing.

Prod-83 replay-beat `Mac-Ultra-M2-forests.json` (+1.25e-5) but **check-in was
refused**: one tagged source neuron was in the bundle (GRQ #4216). Replay and
the random sweep now skip tagged UUIDs (journal reason `tagged`) so known
untagged cuts can still publish. Overnight loop
is replay-only (`--max-experiments 2`): re-apply the cache onto the current
fittest, check in, pull, repeat.

## What failed before

Publishing tiny prunes of **yesterday’s** Forests champion. Breeding reads
every `samples/*.json`, but fitness-proportional selection never promotes a
creature that is 8e-4 behind the live residual grafts. The general population
is judged by the fittest, not by whether our file exists.
