# Pre-registration: warm-starting EM at the constructed recency solution

Written before any checkpoint in `runs/warm/` exists. Tier 1, item 2 of the plan
following `AUDIT_2026-09-10.md`.

## The question

Is EM's recency deficit (single-`p0` EM 0.600 vs WM 0.975 at n=8; WM reaches
loss < 0.5 on 8/8 seeds, EM on 0/8) a failure to FIND a solution the architecture
can hold, or a limit of what it can hold? The audit's existence proof (a
single-`p0` kernel selects the answer 1423/1423 once the query token rewinds the
count) is only kernel-level: an argmax with an idealised content gate.

## Deviation from the plan, stated

The plan said "build EM weights implementing the rewind, a content gate and a value
readout, then train from there". Instead **only the POSITION PATHWAY is installed**
-- the rewind code in two embedding coordinates, `w_in`/`w_out`, `omega` and `p0`
at the per-head norm EM learns on recency (1.957) -- and the content branch starts
random. This is the stronger test: if the model then reaches WM-level accuracy it
has LEARNED a full solution around the rewind, which demonstrates representability
at the full-model level more convincingly than a hand-set content branch would, and
it isolates the rewind as the thing from-scratch training fails to find.

## Verified before launch (`model_em_warm.py`)

- Inside the model's own modules, the installed `A_P` selects the answer on every
  query: **8 construction seeds x T=1024 and T=2048, all 1.000** (see launch log).
- Architecture unchanged: `token_emb(t) = base(t) + latent(t)` with `base`'s two
  latent columns held at zero by a gradient mask -- one table with two columns
  held fixed, not a second embedding.
- The freeze holds bitwise under AdamW with weight decay (30-step smoke test: no
  frozen tensor moved; masked columns stayed 0.0); the trainable twin moves all.
- The `w_out` scale (x16) was chosen by a margin sweep BEFORE training: argmax exact
  at every scale x1-x32; min margin over kappa_max 3% at x1, ~58% at x16. It sets
  how hard the content gate's job is, not whether the kernel is correct.
- Stated limitation of the construction: filler keys between the answer and the next
  symbol, and the query's own key, TIE the answer on `A_P` on every query. The content
  branch must learn a token-type gate. That is part of what is being tested.

## Arms

`EMWarm_freeze` (position pathway frozen) and `EMWarm_train` (installed, all
trainable), seeds 0-7, recency recipe unchanged, no `--fast-attn`. Comparators are
existing deterministic runs on the same seeds: `VanillaEM_P0_r4` s0-7
(`runs/dof/recency`, from scratch) and `Vanilla_r4` s0-7 (`runs/recency_em`, WM).
The MagOnly batch re-trains `VanillaEM_P0_r4` s0 and checks it bitwise against the
stored checkpoint, which covers the determinism assumption for this comparator.

## Predictions

**W1 -- representability at the full-model level.** `EMWarm_freeze` reaches
accuracy **>= 0.95 at T=1024 on at least 7/8 seeds**. *Falsified if* it stays near
from-scratch EM (<= 0.75 mean): then even with the correct position kernel EM's
content side fails, and the deficit is not (only) the rewind.

**W2 -- the headline contrast.** `EMWarm_freeze - VanillaEM_P0_r4` (paired by seed)
is positive and clears its MDE.

**W3 -- does installed EM match WM?** `EMWarm_freeze - Vanilla_r4` is inside its MDE.
Registered as the expected outcome IF W1 holds; a detectable gap either way is
reported as such.

**W4 -- is the rewind an attractor under training?** `EMWarm_train` is within MDE of
`EMWarm_freeze`. If instead it falls toward from-scratch EM, the solution exists but
SGD moves away from it -- which would locate the problem in the loss landscape
around the solution, not only in finding it. Diagnostic: at the end of training,
the fraction of queries on which the trained `A_P` still selects the answer among
symbol keys.

**Not a rule-9 question.** The claim is that a solution is REACHED; accuracy and
loss will move together by construction and loss-matching would condition away the
effect being measured (audit finding 7).
