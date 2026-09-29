# Context-step pilot 2 (two cue conditions) -- the double dissociation did NOT happen (2026-09-29)

Pilot, 2 seeds, not a registered result. Runs `runs/ctxstep2_pilot`; task `environment_textworld_ctx2.py`
(gated: the move/decoy class is unpredictable from the matched side); held-out map, T=1024, p_decoy 0.3.

## Accuracy (s0 / s1)
| arm | leading cue | trailing cue | predicted |
|---|---|---|---|
| CF context-free | 0.810 / 0.826 | 0.803 / 0.747 | fail / fail -- as predicted |
| CG context gate | 1.000 / 1.000 | 0.960 / 0.999 | solve / **fail** -- WRONG on trailing |
| SR Selective-RoPE generator | 1.000 / 1.000 | 1.000 / 1.000 | **fail** / solve -- WRONG on leading |
| HS hidden-state step | 1.000 / 0.874 | 0.941 / 0.880 | solve / solve |

## Swap test (`docs/audits/2026-09-27/swap_ctxstep2.py`): angle change from north <-> south, decoy / move
CF 1.00 / 1.00 (both conditions); CG 0.00-0.01 (lead), 0.02-0.03 (trail); SR 0.19-0.20 (lead),
0.07-0.10 (trail); HS 0.00 (lead), 0.03-0.04 (trail). Every context arm suppresses decoys in both
conditions (SR only partly on leading cues).

## Which words they read (inline check, 2026-09-29 05:10)
- SR, leading: replace the cue word just before the decoy's direction ("go" / "going") by "walked"
  and the decoy's angle change rises 0.012-0.014 -> 0.054-0.061 (real moves 0.062-0.071). SR reads the
  LEADING cue.
- CG, trailing: replace the cue just after ("no" / "but" / "--") by "and": 0.004-0.009 -> 0.137-0.240
  (real moves ~0.25). CG reads the TRAILING cue.

## Why the design's argument was wrong (the second correction)
Both generators are nonlinear within a 4-token window, and that suffices for either side:
- CG: `Delta_t = g_t * Delta(x_t)`. The direction word's own step cannot be gated by what follows, but
  the tokens AFTER it can: their gates read the direction word inside their window, so a cue token
  can carry a direction-specific cancelling step (e.g. "no" steps -north only when north precedes it).
- SR: `w_t = g(x_t) * sum_k a_k u(x_{t-k})` per channel. The direction word's per-channel gate
  multiplies the LAGGED cue terms, so with channels specialised by direction the cue's lagged
  contribution can cancel the direction's own step in exactly the channels that word writes to.
Lead-versus-trail is not what separates these mechanisms. What does is how far away the cue may be:
CG and SR see a fixed window of 4 tokens; HS reads its context through attention, without a limit.
