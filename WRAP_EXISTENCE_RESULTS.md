# Can a non-negative rotary step hold the ring map? (rule 14 existence) -- results (2026-10-13)

Pre-registration `WRAP_EXISTENCE_PREREG.md` + Amendment 1 (independent check `docs/audits/2026-10-13/wrap_existence_audit.md`).
Batch 15:29-18:02 at 5b577bb on both GPUs; ring 32 cells, p_plus 0.5, T = 128, 300 epochs (+300 for continuations), seeds
600-607 (n = 8). Registered output `WRAP_EXISTENCE_ANALYSIS.txt`, `WRAP_EXISTENCE_VERDICTS.json`. Models: SignedAO_r4 /
AbsAO_r4 = Signed_r4 / Abs_r4 with observation steps 0 by construction (only the two moves have steps).

| arm | what | acc T = 128 | SOLVED | LOST in training | acc T = 512 (OOD) |
|---|---|---|---|---|---|
| SO | signed, from scratch | 1.000 | 8/8 | -- | 0.996 |
| AO | non-negative, from scratch | 0.921 (0.85-0.95) | **0/8** (all STALLED) | -- | 0.830 |
| SO_bal | SO rebuilt at the balanced gauge (signed), trained 300 more | 1.000 | 8/8 | 0/8 | 0.998 |
| **AO_bal** | SO converted exactly to non-negative steps (balanced construction), trained 300 more | **1.000** | **8/8** | **0/8** | 0.978 |
| SO_cont | SO trained 300 more as is | 1.000 | 8/8 | 0/8 | 0.996 |
| AO_wrap | non-negative, min-norm construction (W_out up to 16,000x the signed scale) | 1.000 | 7/8 | 8/8 | 0.914 |
| AO_wrapg | non-negative, per-channel-gauge construction | 1.000 | 8/8 | 8/8 | 0.930 |
Every constructed model starts at accuracy 1.000, equal to its source (frozen). Floor 0.554.

## Registered
- **P0:** the signed observation-free model solves 8/8, so E1 / E2 are read on all 8 seeds.
- **E1 (exists, frozen): EXISTS for every construction** -- frozen accuracy equals the source's on 8/8 seeds; max angle
  error 1.2e-6 rad (balanced), max logit difference 5e-4 at T = 128.
- **E2 (primary, AO_bal vs SO_bal): HELD** -- 8/8 SOLVED, 0/8 LOST (max per-epoch loss 0.024-0.154 vs the control's
  0.015-0.112), accuracy 1.000 vs 1.000. Descriptive arms (ill-conditioned constructions, against SO_cont): **NOT HELD** by the
  LOST rule (8/8 lose the solution in the warm-up, max loss 0.6-2.0; Fisher p 0.0002), and then re-find it (7/8 and 8/8
  SOLVED by the end). Per the pre-stated reading, NOT HELD under an ill-conditioned construction says nothing about the class.
- **E3 (observation-free, from scratch): SIGNED LEARNS BETTER** -- non-negative 0/8 SOLVED, 0.921 vs signed 8/8, 1.000
  (d -0.079, perm p 0.0002; Fisher p 0.0002).

## Secondaries (declared; no verdict)
- **The task is held, the wrapped code is not.** At construction 87% of AO_bal's channels cancel opposite moves mod 2 pi
  (as in its signed source); after 300 epochs, 18% (8/8 seeds, 0.15-0.22). The signed control moves the other way: 87% ->
  98%. Training keeps the non-negative model solved but moves it off the cancelling code, and it ends a little worse out of
  distribution (T = 512: 0.978 vs 0.998).
- From scratch, the non-negative model never reaches the cancelling code (6% of channels vs 87% signed) and learns slower
  on 8/8 paired seeds (loss AUC +0.45, sign-flip p 0.0078).
- **Fragility, frozen, at a matched gauge:** multiplying omega by (1 + 0.01 N(0,1)) leaves the signed solution at 0.998 but
  drops the wrapped one to 0.802 (min 0.681); W_out the same way: 0.975 vs 0.815. At 1e-3 both stay at 1.000. The wrapped
  solution is narrower to perturbations of the same relative size (a probe of frozen models, not a measured basin).
- Dropout re-score changes nothing (all constructed arms 1.000; AO 0.921). Cross-batch, labelled: removing observation steps
  did not help the non-negative step (0/8 here vs RING_DIAL's Abs_r4 with observation steps 2/8).

## Post hoc (`docs/audits/2026-10-13/wrap_existence_channels.py`, `_out.txt`; CPU, after the registered analysis)
Where the cancelling code goes in AO_bal: the share of the total action angle carried by cancelling channels falls from 0.79
to 0.15 (SO_bal: 0.79 -> 0.97), in slow and fast channels alike (fast channels |omega| > 1: 0.67 -> 0.16), and angle moves
from small-angle channels to large-angle ones (channels with |angle| > 0.5 rad: 23 -> 42). Not identified: which non-cancelling
code the solved model then uses.

## What it means
- **Rule 14 is now met for the learnability claim.** In the rotary slot, a non-negative step model that solves the ring
  EXISTS at a trained scale (weights within ~5x of the signed model's) and training KEEPS it solved (8/8). From scratch the
  same class never solves (0/8). So the sign cost on the ring is search: training does not find a solution that exists and
  is stable in task terms.
- **Stable in task terms, not in code terms.** The wrapped (cancelling) code itself is not where training settles under the
  non-negative constraint: started on it, training drifts away (87% -> 18% of channels) while staying solved, and the wrapped
  code is more fragile to small parameter perturbations than the signed one. So the non-negative step has solving regions
  that training can stay in but cannot reach from scratch, and the exact map code is not an attractor inside them.
- Observation steps were not the obstacle (E3 replicates the sign cost with them removed).
- An aside the check caught: the naive construction (keep W_in, solve W_out) is exact but ill-conditioned (solved signed
  models have antiparallel action latents, cos -0.987..-0.998) and loses the solution under Adam within ~10 epochs; the pilot
  reading of that as an intrinsic "path-length fragility" was confounded and is withdrawn.

## Caveats
- One task (1D ring, p_plus 0.5), 1 layer, 300 + 300 epochs, one optimiser recipe (AdamW, lr 1e-3, cosine re-warmed);
  observation-step-free classes (the RING_DIAL models have observation steps). The balanced construction is one gauge
  choice (omega' = max(|omega|, 1), orthogonal action latents).
- "Held" is task performance (accuracy, SOLVED, no loss spike above the LOST thresholds); the code drifted (secondary).
- The fragility probe is on frozen models and two perturbation sizes; it is not a measured basin width.
