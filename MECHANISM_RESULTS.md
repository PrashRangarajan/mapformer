# Why a non-negative rotary step does not learn the map -- results (2026-10-14)

Pre-registration `MECHANISM_PREREG.md` + Amendment 1 (independent check `docs/audits/2026-10-14/mechanism_audit.md`). Batch
01:12-05:07 at 85be14d on both GPUs (alongside GENERALITY); ring 32 cells, p_plus 0.5, T = 128, 300 epochs, observation-step-
free models (SignedAO_r4 / AbsAO_r4), seeds 800-807, n = 8 per arm. Registered output `MECHANISM_ANALYSIS.txt`,
`MECHANISM_VERDICTS.json`. Floor 0.554. Arms differ only in the start of the two move angles (every other weight identical at a
seed); "balanced gauge" = the move latents re-solved orthogonal, omega' = max(|omega|, 1) (the gauge controls show it changes
nothing measurable).

| arm | step | start of the two move angles | acc T = 128 | SOLVED | acc 4T |
|---|---|---|---|---|---|
| SO | signed | default (zero-mean, small) | 1.000 | 8/8 | 0.999 |
| SO_same | signed | the same angles, balanced gauge | 1.000 | 8/8 | 0.995 |
| **SO_absmag** | signed | **|default|: both moves rotate the same way, small (the |.| start)** | **0.983** | **7/8** | 0.981 |
| SO_high | signed | large, iid U(1.7, 4.5) rad (sums > pi) | 1.000 | 8/8 | 0.996 |
| AO | |.| | default (= |SO's angles|) | 0.919 | 0/8 | 0.733 |
| AO_absmag | |.| | the same angles, balanced gauge | 0.892 | 0/8 | 0.773 |
| AO_mid | |.| | large but sums < pi, U(0.9, 1.5) | 0.944 | 2/8 | 0.980 |
| **AO_high** | |.| | **large, sums > pi (SO_high's exact angles)** | **1.000** | **8/8** | 0.995 |

## Registered
- **M2: a start whose move-angle sums exceed pi RESCUES the non-negative step** -- AO_high 8/8 vs AO_absmag 0/8, +0.109
  [95% CI +0.085, +0.135] (perm p 0.0002; Fisher p 0.0002).
- **M2b: it is the sum, not just size** -- AO_high 8/8 vs AO_mid (large but sums below pi) 2/8, +0.056 [+0.030, +0.085]
  (perm p 0.0014; Fisher p 0.0070). AO_mid's angles move ~5x more per Adam step than the default's, AO_high's 13-22x (the
  check's measurement), so the mobility control is partial.
- **M6: from the large start the sign makes no difference** -- AO_high 8/8 vs SO_high 8/8 (NO EFFECT DETECTED; both 1.000).
- **M5: from the small same-direction start the sign DOES matter** -- AO_absmag 0/8 vs SO_absmag 7/8, -0.092 [-0.135,
  -0.045] (perm p 0.0030; Fisher p 0.0014): SIGN COST AT THE |.| START.
- **M1: the |.| start barely hurts a signed model** -- SO_absmag 7/8 (0.983) vs SO_same 8/8 (1.000), -0.017 [-0.040, 0.000]:
  UNMEASURED (prediction HURTS not met).
- **M3: the balanced gauge changes nothing detected** (SO_same = SO; AO_absmag vs AO -0.028, p 0.16). **M4 replicates the sign
  cost** in this batch: AO 0/8 vs SO 8/8 (p 0.0002).

## Secondaries (declared) and one post hoc check
- Wrap share at the end: SO 0.86, SO_same 0.96, SO_absmag 0.89; AO 0.07, AO_absmag 0.04, AO_mid 0.05; AO_high 0.23, SO_high
  0.34 (the large-start models solve with a minority of cancelling channels).
- Post hoc (`docs/audits/2026-10-14/mech_escape.py`, `_out.txt`): **how the cancelling channels cancel.** Signed models from
  the default or the |.| start end with 98-99% of channels having opposite-sign move angles and cancel THROUGH ZERO (0% of
  their cancelling channels wrap) -- SO_absmag started with 0% opposite-sign channels. From the large start both signs cancel
  BY WRAPPING (SO_high 88%, AO_high 100% of cancelling channels have |raw sum| > pi).
- Dropout re-score changes nothing (all arms within 0.001). r(initial opposite-sign share, loss AUC) over SO's seeds -0.30.

## What it means
- **The non-negative constraint removes the path through zero.** A signed rotary step reaches the cancelling map by moving
  one move's angle through zero (it does so even when started with both moves rotating the same way: 7/8). A non-negative
  step cannot cross zero; its only cancelling solution is the wrap at a(+1) + a(-1) = 2 pi, and training reaches it when the
  start already has sums past pi (8/8) and not from a small start (0/8) or a large one whose sums stay below pi (2/8).
- **So the learnability cost is the constraint acting on the start, and a start fixes it completely**: from the large start
  the non-negative step learns the map as well as the signed one (M6). Together with WRAP_EXISTENCE (an exact non-negative
  solution is held under training) this explains the sign result: the non-negative solution exists and is stable, but from a
  small start the only route to it would pass through zero.
- **Design implication (one task, untested elsewhere):** a monotone content-dependent rotary method (Case II) can learn a map
  if its per-move angles are initialised so that opposite moves' angles sum past pi -- i.e. near the wrap -- rather than near
  zero (CARoPE and similar initialise at index RoPE, small steps).
- **The toy's stronger claim did not transfer.** In the toy (exploratory) a signed model started at the |.| angles failed
  (0/8); the transformer escapes (7/8). The toy's signed default also solved only 6/12. The exploratory "the cost is the start,
  not the constraint" is replaced by the registered "the constraint blocks the path through zero; a start past pi avoids it".

## Caveats
- One task (1D ring, p_plus 0.5), observation-step-free models, 1 layer, 300 epochs, one optimiser recipe; n = 8.
- The large start is also a more mobile start under Adam (13-22x per step); AO_mid separates sum from size only partly (~5x).
- The large-start models solve with few cancelling channels (23-34%); what the remaining channels do is not identified.
- No torus version of the start manipulation yet.
