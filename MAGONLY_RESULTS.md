# MagOnly: phase freedom survives the matched-optimiser control

Pre-registration: `MAGONLY_PREREG.md`. `EMDoF_magonly` seeds 0-23, compared with the
existing `AlignFree`, `AlignLock` and `VanillaEM_P0_r4` runs on the same seeds.

**Determinism check passed:** `AlignFree` s0 and `VanillaEM_P0_r4` s0 re-trained in
this batch are bitwise identical to the stored checkpoints (weights and loss
curves), so reusing the existing comparators is licensed by measurement.

| arm | phase DOF | magnitude parameterisation | final loss | acc T=1024 (n=24) |
|---|---|---|---|---|
| `AlignFree` | `n_b` | raw vector, ~0.02 | **0.746 +/- 0.362** | **0.818 +/- 0.127** |
| `MagOnly` | 0 | raw vector, ~0.02 (same as AlignFree) | 1.156 +/- 0.243 | 0.672 +/- 0.094 |
| `AlignLock` | 0 | scale `s`, init 1.0 | 1.239 +/- 0.388 | 0.654 +/- 0.141 |
| `P0` | 0 | shared vector | 1.143 +/- 0.320 | 0.687 +/- 0.127 |

## M1 CONFIRMED -- the decisive contrast

| contrast | delta | sd | MDE | seeds + | verdict |
|---|---|---|---|---|---|
| **`AlignFree - MagOnly`** (phase freedom, matched optimiser) | **+0.146** | 0.150 | 0.086 | **22/24** | **DETECTABLE** |

`MagOnly` is the same function as `AlignFree` at init, with the same parameter
count and the same optimiser scale. Only phase freedom differs, and it is worth
+0.146.

## M2 -- the confound was real in principle and ~0.02 in practice

    AlignFree - AlignLock (D1)  +0.165  =  +0.146 (phase)  +0.018 (parameterisation)

`MagOnly - AlignLock` is **+0.018** (MDE 0.089, 13/24, unmeasured). Audit finding 8
correctly identified a confound; it accounts for about a tenth of D1.

## M3 -- magnitude freedom is null even when the optimiser can move it

`MagOnly - P0` = **-0.015** (MDE 0.083, 12/24). `MagOnly`'s magnitudes move ~59% in
200 steps against 1% from weight decay (pre-launch manipulation check, M4), so this is
the properly controlled version of D5's null, not a parameter that barely moved.

## M5 -- replicates on fresh seeds, and n=8 overestimated again

| seeds | `AlignFree - MagOnly` | seeds + | MDE | verdict |
|---|---|---|---|---|
| 0-7 | +0.213 | 8/8 | 0.110 | DETECTABLE |
| **8-23 (fresh)** | **+0.113** | **14/16** | 0.111 | **DETECTABLE (just)** |

The sign holds on fresh seeds. The fresh-seed size is about half the first eight
seeds', and it clears its MDE by 0.002 -- report +0.146 pooled, with the fresh-seed
estimate beside it.

## M6 -- as registered

`r(final loss, acc) = -0.978` over 96 runs; `AlignFree` fits better than `MagOnly`
on 23/24 seeds (loss -0.410, MDE 0.233). After audit finding 7 that does not, by
itself, say "optimisation"; `WARM_RESULTS.md` supplies the stronger argument.

## Post-hoc probe (not registered): phase freedom does NOT work through the rewind

`_REWIND_PROBE.json`. The existence proof solves recency with a **rewind** (`q_k`
carrying `-(k-1)` symbol steps). **No from-scratch EM arm learns one**: pooled rewind
slope 0.000 +/- 0.02 in every arm (exact rewind = -1), and at the block level only
**1 of ~2,550** (head, block) pairs across the 40 runs has slope below -0.5. Their
position kernels alone pick the answer on only 11-16% of queries.

Yet `AlignFree` reaches 0.868 on these seeds, with per-offset accuracy 0.76 at
`k=64` against `P0`'s 0.37. So phase freedom improves recency through some OTHER,
partial mechanism -- one that handles small `k` without a rewind and degrades more
slowly with `k`. **What that mechanism is has not been identified.** The link I
expected between the two Tier-1 results ("phase freedom helps EM find the rewind")
is refuted.
