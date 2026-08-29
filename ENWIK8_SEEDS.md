# enwik8 — does the PoPE x path-integration combination beat RoPE? (n=3)

36k iters, seq 512, deterministic val (fixed generator -- identical batches for every arm and checkpoint), r=4 for the path-integrating arm, param-matched to 0.03%. Values are the mean of the last 5 checkpoints. **Lower is better.**

| seed | RoPE | MapPoPE-Flat r4 | delta |
|---|---|---|---|
| 0 | 1.3864 | 1.3786 | **-0.0078** |
| 1 | 1.3837 | 1.3795 | **-0.0043** |
| 2 | 1.3840 | 1.3804 | **-0.0036** |

- mean **-0.0052**, sd 0.0023
- sign-consistent: **YES** (3/3 favour the combination)

> All seeds favour the combination. At this scale (295M tokens vs the
> paper's 100B) the effect is small but consistent -- the two orthogonal
> mechanisms compose on language, as they do on navigation.

## CORRECTION: "composes" is NOT established — only "beats the baseline"

| arm | mean last-5 | ckpt sd | seeds |
|---|---|---|---|
| MapPoPE-Flat r4 | 1.3786 | 0.0054 | **3** |
| PoPE-Flat | 1.3806 | 0.0051 | **1** |
| Vanilla r4 | 1.3841 | 0.0069 | **1** |
| RoPE | 1.3864 | 0.0032 | **3** |

MapPoPE is numerically lowest, but the claim "the two mechanisms COMPOSE" requires it
to beat BOTH COMPONENTS, and that comparison is the weak one:
- vs PoPE-Flat: **-0.0020 at n=1** -- less than HALF PoPE-Flat's own checkpoint sd
  (0.0051). Indistinguishable from zero.
- vs Vanilla_r4: -0.0056 at n=1.
- Only vs RoPE has 3 seeds (-0.0052, 3/3 sign-consistent).

**ESTABLISHED:** MapPoPE-Flat r4 beats the plain RoPE baseline, 3/3 seeds.
**NOT ESTABLISHED:** that it beats its own components, i.e. that PoPE and path
integration compose on language. The seeds are on the wrong comparison for that claim.

**TO ESTABLISH IT:** seeds 1,2 for PoPE-Flat and Vanilla_r4 (4 arms, ~2.5h). Then the
composition claim can be evaluated with the same 3-seed sign-consistency test, and
the vs-PoPE-Flat margin (-0.002) can be read against seed variance rather than
checkpoint variance. On current evidence that margin is unlikely to survive.
