---
name: project-mappope-asymmetry
description: PoPE's encoding helps the path row on Bach (MapPoPE never detectably worse than MapWM); "path integration added to PoPE hurts on clock tasks" is UNMEASURED after the code full-val rescore; the Dyck +0.050 is depth-OOD.
metadata:
  type: project
---

> **CORRECTED 2026-10-03.** (1) Code, rescored on the full val file (`CODE_FULLVAL_RESULTS.md`, n=3, t-test):
> MapPoPE - MapWM -0.0054 (p 0.086) and MapPoPE - PoPE +0.0034 (p 0.19) are both UNMEASURED; the "detectable"
> labels below were the house |t| > 2.8 rule on `best_val_bpc`. (2) The Dyck rows are the D4-trained ladder read at
> D12, i.e. depth extrapolation; trained at D12 (`DYCK_MDEPTH_RESULTS.md`) MapWM and MapPoPE are both 0.996-1.000.
> What survives: MapPoPE - MapWM on Bach -0.0165 (5/5, MDE 0.0111). The clock half (path integration hurts PoPE) is
> unmeasured everywhere; do not claim it.

**SEPARATED 2026-10-05 (`MAPPOPE_PAIR_RESULTS.md`, registered).** Our MapPoPE differed from MapWM in the score rule AND
the frequency count (64 per-element angles vs 32 per pair). On the paper torus (T=128, r2, n=16) the score rule carries
the whole gain (+0.0243, p 0.012; 16/16 vs 10/16 SOLVED); the angle count adds +0.0002, CI [-0.0004, +0.0008]. It also
does not explain MapPoPE's small rank-4 gain. Read MapPoPE vs MapWM as a score-rule comparison on this task. Open:
does PoPE's score rescue rank 2 at T=1024 (MapWM r2 0/8 there)?

Encoding effect on the index row (PoPE - RoPE) vs the path row (MapPoPE - MapWM), 2026-09-23:

| task | PoPE - RoPE | MapPoPE - MapWM |
|---|---|---|
| Bach NLL | -0.032 (5/5) | -0.0165 (5/5, MDE 0.0111) |
| Dyck L32D12 1L / 2L (Hewitt acc) | +0.010 / +0.026 (8/8) | +0.073 (7/8) / +0.050 (8/8) |
| Dyck 3L / 4L | +0.017 / +0.026 | -0.013 / +0.025, unmeasured |
| code trained+tested 2048, bpc | -0.0008 unmeasured | -0.0052 (3/3, MDE 0.0036) |

Sources: `DYCK_LADDER_RESULTS.json`, `runs/code2048/*.json` (trainer best_val_bpc), `JSB_RESULTS.md`.

Orderings: Dyck MapPoPE >= MapWM > PoPE > RoPE; Bach PoPE > MapPoPE > MapWM ~ RoPE; code PoPE ~ RoPE >
MapPoPE > MapWM. MapPoPE - PoPE on clock tasks: code +0.0033 (detectable), Bach +0.0111 (just inside MDE).

**Why it matters:** the two ideas are not symmetric partners. PoPE's encoding is a fairly general
improvement to how an angle is used; path integration is a specialist tool for where the angle comes
from -- worth it only when position is genuinely signed (navigation, bracket depth).

**How to apply:** adding PoPE's encoding to path integration is good on Bach and never detectably bad
(if a task needs MapFormer, use MapPoPE); on clock-like tasks path integration buys nothing measured
(code position main +0.0056 bpc, t p 0.024, n=3, is a cost of path integration averaged over encodings). Whether the path-row effect is
genuinely LARGER is the 2x2 interaction and is NOT established -- do not claim it.
