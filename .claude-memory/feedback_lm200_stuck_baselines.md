---
name: The lm200 era is retracted — a whole leaderboard ranked convergence
description: April lm200 checkpoints never converged, so the landmark leaderboard ranked training convergence rather than architecture. Four derived findings died with it. The lesson generalises past lm200.
metadata:
  type: feedback
---

**Everything measured on the lm200 (landmark) task before 2026-05-08 is void**, and
so is every claim built on it. This file replaces four separate memories that each
recorded one of those claims as a finding.

## What happened

Stored lm200 checkpoints trained 2026-04-22..24 never converged (final CE ~1.0
instead of ~0.005); ones trained 2026-05-08+ converged normally. The reported
leaderboard was **monotone in final training loss**, not in architecture:

| reported rank | acc | stored final loss |
|---|---|---|
| Vanilla | 0.716 | 1.22 (stuck) |
| Level15 | 0.819 | 1.01 (stuck) |
| NoDrop | 0.948 | 0.24 (partial) |
| GSF | 0.956 | 0.0007 (converged) |
| TEMFaithful | 0.969 | 0.0004 (converged) |

Retrained under current code, **Level15 reaches 0.996 and beats TEMFaithful (0.982)**.

## The four claims that died

- "TEMFaithful is the lm200 leader" — reversed.
- "Removing post-attention dropout buys +13pp on lm200" — the baseline was stuck.
- "GSF (K=8 multi-modal Bayes) closes 95% of the TEM gap" — there was no gap.
- "NoDrop and GSF are accuracy-substitutes but NLL-complements" — built on both.

**Do not cite any of them.** The post-attention-dropout mechanism story (block
dropout hurts rare-token retrieval) was plausible and may still be true; it has no
surviving evidence here.

## Scope, and why it is narrow

Clean and noise checkpoints retrain **bit-identically**, so those results are valid.
The root cause is the landmark-cell-selection RNG, which only runs when
`n_landmarks > 0`; lm200 training is basin-sensitive to the resulting layout.

## The general lesson

A leaderboard whose ordering tracks final training loss is measuring optimisation.
Check that correlation **before** reading the ranking — r(final loss, accuracy) has
reached **-0.999** in this project. This is standing rule 9, and it was bought here.
