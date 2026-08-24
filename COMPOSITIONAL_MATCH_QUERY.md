# Compositional Match-Query — honest writeup (n=6, with LR warmup)

Blind continuation in a repeated-motif world (see `environment_compositional_match_query.py`).
Explore with observations visible; then continue BLIND and predict the observation
at the current cell. Scored two ways, both oracle-answerable (=1.0) by construction:

- **exact** — cell exactly seen in explore → path-integration MATCHING alone.
- **cross** — cell not seen, room explored + motif seen in another copy →
  path integration AND motif abstraction TOGETHER (the compositional synergy target).

Chance = 1/16 = 0.0625. Trained T_explore=512, T_query=256, 200 epochs, LR
warmup 0.05, 5 variants × 6 seeds. Held-out env (seed=10000). This is the
**stabilised** rerun; the no-warmup sweep is in git history.

## Full per-seed results (nothing dropped) — @ T_query=256

| variant | metric | s0 | s1 | s2 | s3 | s4 | s5 |
|---|---|---|---|---|---|---|---|
| Vanilla | cross | 0.096 | 0.098 | 0.199 | 0.083 | 0.102 | 0.095 |
| Vanilla | exact | 0.584 | 0.106 | 0.717 | 0.680 | 0.171 | 0.119 |
| Hourglass_k2 | cross | 0.107 | 0.095 | 0.094 | 0.120 | 0.072 | 0.112 |
| Hourglass_k2 | exact | 0.228 | 0.148 | 0.177 | 0.772 | 0.760 | 0.689 |
| Hourglass_CoarseIdx | cross | 0.113 | **0.568** | 0.110 | 0.101 | 0.102 | 0.097 |
| Hourglass_CoarseIdx | exact | 0.212 | **1.000** | 0.190 | 0.183 | 0.161 | 0.151 |
| PlainFlat | cross | 0.123 | 0.127 | 0.110 | 0.112 | 0.119 | 0.114 |
| PlainFlat | exact | 0.241 | 0.176 | 0.197 | 0.192 | 0.177 | 0.176 |
| PlainHourglass | cross | 0.117 | 0.117 | 0.112 | 0.102 | 0.114 | 0.118 |
| PlainHourglass | exact | 0.221 | 0.164 | 0.194 | 0.178 | 0.173 | 0.168 |

## Hit-rate (pre-declared bars: exact > 0.40, cross > 0.20)

| variant | exact cleared | cross cleared | exact (converged mean) |
|---|---|---|---|
| Vanilla | 3/6 | 0/6 | 0.660 |
| Hourglass_k2 | 3/6 | 0/6 | 0.740 |
| Hourglass_CoarseIdx | 1/6 | **1/6** | 1.000 |
| PlainFlat | 0/6 | 0/6 | — |
| PlainHourglass | 0/6 | 0/6 | — |

## What the results honestly say

**1. Path integration is NECESSARY for blind continuation — robust, airtight.**
Plain transformers (no path-integration code) NEVER clear even the easy `exact`
column: 0/12 seeds, stable at ~0.20 exact / ~0.11 cross regardless of seed. A
plain model cannot localise itself when observations are withheld, so it cannot
match a cell it saw during explore. This is the clean negative.

**2. MapFormer learns exact-blind matching — capability confirmed, training bimodal.**
7/18 MapFormer seeds reach exact 0.58–1.0 (vs plain's ~0.20 ceiling); the other
11 stay stuck at the plain floor. So the capability is real and multi-seed, but
optimisation is unreliable and LR warmup did NOT fix the bimodality.

**3. The compositional cross-blind synergy is a FRONTIER, not a result.**
Exactly **1 run in 30** solved it: `Hourglass_CoarseIdx` s1, cross 0.568 with
exact 1.000. This is a genuine existence proof — the task is solvable, it is not
a leak (a cheat would push cross to 1.0 too), and it is the re-indexed-coarse
hierarchy that achieved it. But at 1/6 for the best variant and 0/6 for every
other, it is NOT a reliable or multi-seed result.

**4. The most informative finding is the DISSOCIATION.** `Hourglass_k2` reaches
exact 0.69–0.77 on 3/6 seeds yet NEVER clears cross. So the localisation step is
learnable; the localise→identify-template→reuse-motif COMPOSITION is the hard
part that is almost never learned. Only the re-indexed-coarse structure ever got
it (n=1). This says the bottleneck is the compositional binding, not the spatial
code — a real, if negative, mechanistic result.

## Conclusion and status

- **Publishable, honest claim:** *path integration is necessary for blind
  positional matching (plain 0/12); it is sufficient on convergent seeds for
  exact-blind recall (7/18); the compositional cross-blind extension is
  achievable (existence, 1/30) but not reliably learnable, and only by the
  re-indexed-coarse hierarchy.*
- **Not claimable:** any average-case superiority on cross, or that hierarchy
  reliably solves the compositional task. It does not, at this training budget.
- **Positioning:** this is a SECONDARY, frontier/negative-leaning result. The
  paper's spine remains the STABLE findings (visible-obs compositional
  `CoarseIdx` win at n=8; enwik8 efficiency; PoPE paper-OOD). Compositional
  Match-Query is the honest "we pushed to the blind regime; path integration is
  necessary, the synergy is achievable but training-limited" section.
- **If pursued further:** the fix is a curriculum (short→long T_query) to raise
  the good-basin hit-rate; warmup alone was insufficient. Not run here.

Raw aggregates in `COMPOSITIONAL_MATCH_QUERY_STAB.md`; per-seed checkpoints in
`runs/cmq_stab/`.
