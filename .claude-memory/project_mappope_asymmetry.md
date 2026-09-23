---
name: project-mappope-asymmetry
description: PoPE's encoding helps MapFormer wherever it helps at all (MapPoPE never detectably worse than MapWM); path integration added to PoPE hurts on clock-like tasks.
metadata:
  type: project
---

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

**How to apply:** adding PoPE's encoding to path integration is good (if a task needs MapFormer, use
MapPoPE); adding path integration to PoPE on a clock-like task is not. Whether the path-row effect is
genuinely LARGER is the 2x2 interaction and is NOT established -- do not claim it.
