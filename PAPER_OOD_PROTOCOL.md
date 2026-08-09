# Paper's own OOD protocol (Table 2, 1D-2D grid navigation)

Appendix B verbatim: trained at l=128, lgrid=64, pempty=0.5; **OOD-d**: l=64, pempty=0.2, lgrid=32; **OOD-s**: l=512, pempty=0.8, lgrid=128.
'dense'/'sparse' = OBJECT density (pempty), not visit density.

The Table 2 caption instead gives OOD-s as l=256; both lengths are reported since the paper is internally inconsistent.

Paper's 2D results -- MapWM: IID 0.99, OOD-d 0.99, OOD-s 0.96. MapEM-os: IID 1.0, OOD-d 0.99, OOD-s 0.97.

| variant | IID  l=128 g=64  pe=0.5 | OOD-d l=64 g=32  pe=0.2 | OOD-s l=256 g=128 pe=0.8 | OOD-s l=512 g=128 pe=0.8 |
|---|---|---|---|---|
| Vanilla | 0.989 ± 0.010 | 0.981 ± 0.012 | 0.988 ± 0.011 | 0.962 ± 0.027 |
| VanillaEM | 0.900 ± 0.103 | 0.860 ± 0.164 | 0.955 ± 0.031 | 0.936 ± 0.035 |
| VanillaEM_P0 | 0.986 ± 0.014 | 0.986 ± 0.006 | 0.987 ± 0.009 | 0.976 ± 0.010 |

## Measured vs paper (2D)

| model | IID | OOD-d | OOD-s |
|---|---|---|---|
| paper MapWM | 0.99 | 0.99 | 0.96 |
| **ours WM** | **0.989** | **0.981** | **0.962** (l=512) |
| paper MapEM-os | 1.0 | 0.99 | 0.97 |
| **ours EM single-p_0** | **0.986** | **0.986** | **0.976** (l=512) |
| ours EM separate q0/k0 | 0.900 | 0.860 | 0.936 |

Every corrected cell is within 0.014 of the paper. The separate-q0/k0 EM is not.

## The ordering claim does NOT replicate (paired, n=3)

The paper has MapEM-os >= MapWM in all three 2D columns. Per-seed differences
(EM_P0 minus WM, same seed):

| condition | s0 | s1 | s2 | mean | EM wins |
|---|---|---|---|---|---|
| IID l=128 | -0.0002 | -0.0141 | +0.0035 | -0.0036 | 1/3 |
| OOD-d l=64 | -0.0018 | -0.0019 | +0.0169 | +0.0044 | 1/3 |
| OOD-s l=256 | -0.0026 | -0.0088 | +0.0092 | -0.0008 | 1/3 |
| OOD-s l=512 | -0.0005 | -0.0005 | +0.0409 | +0.0133 | 1/3 |

EM wins exactly 1 of 3 seeds in EVERY condition. The apparent +1.3pp mean
advantage at OOD-s l=512 -- which would have matched the paper's +1pp -- comes
entirely from seed 2, where WM alone degraded (0.934 vs 0.987/0.966). On seeds
0 and 1 the two are identical to within 0.0005.

So: absolute numbers replicate, the EM>WM ordering does not resolve at n=3.
What IS consistent is lower spread for EM at the hardest condition (std 0.010
vs WM 0.027 at OOD-s l=512) -- EM's worst seed is much better than WM's worst.
That is a robustness difference, not an accuracy difference, and n=3 is thin
evidence for a variance claim. More seeds would be needed to settle either.
