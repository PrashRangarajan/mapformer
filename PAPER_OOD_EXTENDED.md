# Paper's own OOD protocol (Table 2, 1D-2D grid navigation)

Appendix B verbatim: trained at l=128, lgrid=64, pempty=0.5; **OOD-d**: l=64, pempty=0.2, lgrid=32; **OOD-s**: l=512, pempty=0.8, lgrid=128.
'dense'/'sparse' = OBJECT density (pempty), not visit density.

The Table 2 caption instead gives OOD-s as l=256; both lengths are reported since the paper is internally inconsistent.

Paper's 2D results -- MapWM: IID 0.99, OOD-d 0.99, OOD-s 0.96. MapEM-os: IID 1.0, OOD-d 0.99, OOD-s 0.97.

| variant | IID  l=128 g=64  pe=0.5 | OOD-d l=64 g=32  pe=0.2 | OOD-s l=256 g=128 pe=0.8 | OOD-s l=512 g=128 pe=0.8 | ext-s l=1024 g=128 pe=0.8 | ext-s l=2048 g=128 pe=0.8 |
|---|---|---|---|---|---|---|
| Vanilla | 0.988 ± 0.011 | 0.987 ± 0.010 | 0.988 ± 0.011 | 0.962 ± 0.026 | 0.904 ± 0.044 | 0.860 ± 0.040 |
| VanillaEM_P0 | 0.989 ± 0.010 | 0.984 ± 0.008 | 0.987 ± 0.008 | 0.976 ± 0.010 | 0.959 ± 0.009 | 0.939 ± 0.009 |
| MapPoPE-Flat | 0.999 ± 0.001 | 0.995 ± 0.001 | 0.999 ± 0.002 | 0.997 ± 0.006 | 0.991 ± 0.014 | 0.978 ± 0.019 |

## The comparison, stated plainly

The published benchmark saturates at 0.96-1.0, so no method can be shown "best"
on it -- a 1-3pp claim on a ceiling separates nothing. The `ext-s` columns hold
the paper's OOD-s condition fixed (grid 128, p_empty 0.8) and extend LENGTH only.
Headroom opens, and the ordering becomes decisive.

| l (OOD-s condition) | Vanilla = MapWM | VanillaEM_P0 = MapEM-os | **MapPoPE-Flat** | gap vs best paper model |
|---|---|---|---|---|
| 512 (paper protocol) | 0.962 | 0.976 | **0.997** | +2.1 pp |
| 1024 (ours) | 0.904 | 0.959 | **0.991** | +3.2 pp |
| 2048 (ours) | 0.860 | 0.939 | **0.978** | **+3.9 pp** |

At l=2048 the +/-1sd seed ranges do not overlap with EITHER paper model:
MapPoPE-Flat [0.959, 0.997] vs MapEM-os [0.930, 0.948] vs MapWM [0.820, 0.900].
Against MapWM the gap is **+11.8 pp**.

### Why the extension is licensed

The reproduction matches the paper's reported figures on the protocol rows,
including the two that matter most here:

| | paper reports | we measure |
|---|---|---|
| MapWM OOD-s | 0.96 | 0.962 +/- 0.026 |
| MapEM-os OOD-s | 0.97 | 0.976 +/- 0.010 |
| MapWM IID | 0.99 | 0.988 +/- 0.011 |
| MapEM-os IID | 1.0 | 0.989 +/- 0.010 |

So the `ext-s` rows extend a reproduction that lands on the published numbers,
rather than a divergent implementation.

### What this table is NOT

- **The ext-s rows have no published counterpart.** They compare MapPoPE-Flat to
  OUR reproductions of the paper's two models, trained in the same batch (rule
  3). No one else has reported l=1024/2048 on this condition, so "beats the
  paper" is true only for the protocol rows, where the margin is small.
- **n=3.** The seed ranges separate, but three seeds is not a point estimate
  (rule 6) and this needs n>=8 before it carries a paper.
- **One benchmark, one task family.** Everything here is the 64->128 torus grid.
- **MapPoPE-Flat is not a new mechanism.** It is PoPE (arXiv:2509.10534) as the
  Q/K modulation, driven by MapFormer's path-integrated angle instead of the
  sequence index. The contribution is the combination and the demonstration that
  the angle's SOURCE, not the encoding, is what carries the result -- PoPE with
  index position sits on the blank floor at 0.509.
