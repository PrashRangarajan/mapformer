# Paper's own OOD protocol (Table 2, 1D-2D grid navigation)

Appendix B verbatim: trained at l=128, lgrid=64, pempty=0.5; **OOD-d**: l=64, pempty=0.2, lgrid=32; **OOD-s**: l=512, pempty=0.8, lgrid=128.
'dense'/'sparse' = OBJECT density (pempty), not visit density.

The Table 2 caption instead gives OOD-s as l=256; both lengths are reported since the paper is internally inconsistent.

Paper's 2D results -- MapWM: IID 0.99, OOD-d 0.99, OOD-s 0.96. MapEM-os: IID 1.0, OOD-d 0.99, OOD-s 0.97.

| variant | IID  l=128 g=64  pe=0.5 | OOD-d l=64 g=32  pe=0.2 | OOD-s l=256 g=128 pe=0.8 | OOD-s l=512 g=128 pe=0.8 | ext-s l=1024 g=128 pe=0.8 | ext-s l=2048 g=128 pe=0.8 |
|---|---|---|---|---|---|---|
| Vanilla | 0.969 ± 0.037 | 0.958 ± 0.048 | 0.978 ± 0.018 | 0.943 ± 0.033 | 0.893 ± 0.034 | 0.854 ± 0.026 |
| VanillaEM_P0 | 0.987 ± 0.009 | 0.983 ± 0.010 | 0.988 ± 0.007 | 0.978 ± 0.011 | 0.963 ± 0.014 | 0.939 ± 0.020 |
| MapPoPE-Flat | 0.994 ± 0.015 | 0.991 ± 0.010 | 0.995 ± 0.011 | 0.992 ± 0.015 | 0.985 ± 0.019 | 0.970 ± 0.028 |

## n=3 -> n=8: the effect survives, one claim does not

`PAPER_OOD_EXTENDED.md` reported this table at n=3. Standing rule 6 says three
seeds is not a point estimate, so all eight seeds were retrained fresh in one
batch (rule 3 — none of the n=3 checkpoints were reused).

| ext-s l=2048 | n=3 | n=8 |
|---|---|---|
| MapWM (Vanilla) | 0.860 ± 0.040 | 0.854 ± 0.026 |
| MapEM-os (VanillaEM_P0) | 0.939 ± 0.009 | 0.939 ± 0.020 |
| **MapPoPE-Flat** | **0.978 ± 0.019** | **0.970 ± 0.028** |

**The effect holds.** MapPoPE-Flat over the two paper models at l=2048:

| comparison | diff | se | t | verdict |
|---|---|---|---|---|
| vs MapWM | **+0.116** | 0.0135 | **8.59** | decisive |
| vs MapEM-os | **+0.031** | 0.0122 | **2.55** | p ≈ 0.02 |

**But the "non-overlapping seed ranges" claim does not.** At n=3 the ±1sd bands
for MapPoPE-Flat and MapEM-os were disjoint. At n=8 they overlap:

    MapPoPE-Flat   [0.942, 0.998]
    MapEM-os       [0.919, 0.959]   <- overlaps
    MapWM          [0.828, 0.880]

The extra seeds widened MapPoPE-Flat's spread from ±0.019 to ±0.028. The gap over
MapEM-os is real and significant on a t-test, but it is a **+3.1pp effect with
overlapping bands**, not the clean separation three seeds suggested. Against
MapWM the separation is genuine and large.

## The reproduction match to the paper was partly a small-sample artifact

At n=3 our Vanilla reproduced the paper's IID figure almost exactly (0.988 vs a
reported 0.99), which is what licensed extending the protocol. At n=8:

| | paper reports | n=3 | n=8 |
|---|---|---|---|
| MapWM IID | 0.99 | 0.988 ± 0.011 | **0.969 ± 0.037** |
| MapWM OOD-s l=512 | 0.96 | 0.962 ± 0.026 | 0.943 ± 0.033 |
| MapEM-os IID | 1.0 | 0.989 ± 0.010 | 0.987 ± 0.009 |
| MapEM-os OOD-s l=512 | 0.97 | 0.976 ± 0.010 | 0.978 ± 0.011 |

MapWM's near-exact match at n=3 was lucky seeds; at n=8 it sits ~2pp below the
published figure, with sd tripling from ±0.011 to ±0.037. The paper's 0.99 is
still inside one sd, so the reproduction is *consistent* with the published
number — but "we reproduce it exactly" was an overstatement that eight seeds
removed. MapEM-os is unaffected and stable.

This matters for how the `ext-s` rows are framed: they extend a reproduction that
is consistent with the paper, not one that lands on it.
