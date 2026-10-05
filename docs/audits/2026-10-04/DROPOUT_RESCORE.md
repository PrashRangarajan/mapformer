# Re-score of every committed registered batch for the attention-dropout scale (2026-10-04)

Plan step 1 of `docs/theory/2026-10-04/00_PLAN.md`. Mechanism (verified, `dropout_scale_check_out.txt`): models trained
with inverted dropout on the attention probabilities (p 0.1) can come to expect the 1/(1-p) scale of the typical
training-time attention output; eval mode then under-reports them. Method: `rescore_hook.py` multiplies every attention
layer's o_proj input (softmax(scores) @ V) by 1/(1-p) at eval, and the batch's OWN registered eval code is re-run on its
own stream. Every batch was also re-run with no hook: **all reproduce the registered numbers exactly** (torus JSONs,
24/24 text world, 32/32 TW_NORMSTEP, 128/128 H3, 24/24 leak, RANK_ND, RANK_WRAP, and the Dyck position main effects
+0.353 / +0.130 / +0.045 / +0.024). Scripts: `rescore_torus.sh` + `rescore_torus_compare.py`, `rescore_nd.sh`,
`rescore_trainer_evals.py` + `rescore_other_compare.py`, `rescore_dyck.py`; outputs `*_out.txt` here, raw in `runs_rescore/`.

## Verdict: no registered verdict changes
| batch | registered contrast | registered | re-scored |
|---|---|---|---|
| paper torus (T=128) | MapWM r2 - RoPE / MapPoPE r2 - PoPE | +0.166 / +0.320 | +0.155 / +0.307 (index arms gain a little) |
| rank_mi / rank_sep / rank3 | r4 - r2, D - C_bd, r3 - r2 | +0.104, +0.051, +0.102 (p 0.027) | +0.103, +0.051, +0.102 (p 0.027) |
| loop_rank (H1) | loop - r2, 4L - r2 | +0.079, +0.096 | +0.071, +0.078 (see caveat) |
| e1800 | r4 - r2 | +0.086 | +0.086 |
| sign matched | Abs - Signed | -0.177 | -0.177 |
| rank_nd | D+1 - D, 2D / 3D | +0.230 / +0.047 (p 0.50) | +0.233 / +0.049 (p 0.48) |
| rank_wrap | 3L - 3H | +0.153 (p 0.0085) | +0.152 (p 0.012) |
| Dyck matched depth | position main, 1-4 layers | +0.353 / +0.130 / +0.045 / +0.024 | +0.352 / +0.129 / +0.050 / +0.030 |
| H3 | index 1L a1(p); k(p) | 0.717 / 0.676 / 0.825 / 1.000; k 3,3,3,1 | 0.725 / 0.686 / 0.820 / 1.000; k 3,3,3,1 (p 0.9 knife-edge G2 0.0115 -> 0.0201) |
| leak | ActOnly / NormStep - MapWM | +0.0107 / +0.0107 | +0.0107 / +0.0108 |
| text world | path 1L - RoPE 1L | +0.464 (path 0.969) | **+0.488** (path 0.993) |
| TW_NORMSTEP | NormStep - MapWM | +0.006 (p 0.47) | +0.001 (p 0.90) |

## Where it matters
- **Only the text-world path arms move**: MapWM 0.969 -> 0.993 (text world), 0.973 -> 0.997 (TW_NORMSTEP), NormStep
  0.979 -> 0.997, NormStepNB 0.941 -> 0.994; up to +0.17 on single seeds. Torus path models move <= 0.002; the leak
  task 0.0001. The effect belongs to the text world's runs below ceiling, not to path integration in general.
- **One secondary changes**: TW_NORMSTEP DirOnly - MapWM, -0.0006 (p 1.00) registered -> **-0.024 (p 0.0002)** re-scored.
  Rescaled, every learned-step arm (0.994-0.997) beats the direction-words-only oracle (0.972-0.973), which does not
  move: the post hoc aside finding (the oracle is capped by asides) becomes a detectable accuracy gap.
- Index models gain slightly (paper torus RoPE +0.011, PoPE +0.013; sign-batch RoPE +0.004): all path-vs-index gaps
  shrink by ~0.01 and stay p 0.0002.

## Caveat: the correction is a one-layer correction
In multi-layer and looped models the deterministic x1/(1-p) in every layer LOWERS accuracy (loop x4 0.973 -> 0.965,
4 layers 0.990 -> 0.973; Dyck 3-4-layer PoPE -0.009 / -0.011): the per-layer scales compound, so those rows are not
under-reported in eval mode and their re-scored numbers are not a better estimate. The H1 loop / depth contrasts stay
significant either way (p <= 0.006).
