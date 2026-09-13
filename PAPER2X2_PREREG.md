# Pre-registration: the paper-task 2x2 under the converged recipe (PAPER2X2)

Written 2026-09-13, before any arm is trained.

## Why

The report's headline table (`INDEX_BASELINE_PAPER_TASK_n8.md`, `BASELINE_TABLE.md`) puts the
position main effect at **+0.461** and the encoding main effect at **+0.003** (MDE 0.029), n=8. It
was trained with the paper's recipe: 16 epochs, linear decay from step one, lr 3e-4. Under the
converged recipe every other claim in the report uses (300 epochs, 5% warmup + cosine, lr 1e-3),
index RoPE reaches 0.799 +/- 0.018 at training length (`SIGN_ABLATION.md`) against 0.530 in the
headline table. A referee can therefore say the position effect at training length is a recipe
artefact. This batch measures the 2x2 under the converged recipe.

## Arms: 6 x 8 seeds (0-7), one batch

| arm | encoding | position | rank |
|---|---|---|---|
| `RoPE` | RoPE | index | -- |
| `PoPE-Flat` | PoPE | index | -- |
| `Vanilla` | RoPE | path-integrated | r=2 (paper) |
| `MapPoPE-Flat` | PoPE | path-integrated | r=2 (paper) |
| `Vanilla_r4` | RoPE | path-integrated | r=4 |
| `MapPoPE_r4` | PoPE | path-integrated | r=4 |

Recipe copied from `run_sign.sh`: torus grid 64, T=128, 1 layer, 2 heads, d=128, 300 epochs x 98
batches x 128, cosine, lr 1e-3, no landmarks. Evaluation copied from `run_sign.sh`:
`eval_noise_refine`, held-out map (env-seed 10000), T = 128 / 512 / 1024, 100 trials.

## Contrasts (paired by seed; MDE = 2.8 sd / sqrt(n); below MDE = "unmeasured")

Per rank r in {2, 4} (the two index arms are shared between the two 2x2s), per length:

- **Position main effect** = mean over encodings of (path-integrated - index).
- **Encoding main effect** = mean over positions of (PoPE - RoPE).
- **Interaction** = (MapPoPE - PoPE) - (MapWM - RoPE).

Each is reported raw and loss-matched (one acc ~ final-loss fit pooled over all six arms at that
length), with rule 9's r(loss, acc) per length. Primary cell: **r=2, T=128, raw** -- the cell the
headline table reports.

## Predictions and what each outcome does to the report

- **H1**: the position main effect at r=2, T=128 is detectable and positive.
  - Detectable and >= +0.15: the headline claim stands under the converged recipe; the report
    uses this table.
  - Detectable but < +0.15: the claim stands at reduced size; the report states that most of the
    16-epoch gap at training length was recipe.
  - Not detectable: the headline claim at training length is withdrawn; the report's Claim 1 is
    carried by the OOD lengths and the other tasks only.
- **H2**: the position main effect grows with length (T=1024 > T=128), both detectable.
- **H3**: the encoding main effect is not detectably larger than +0.05 in absolute value at
  r=2, T=128. Falsified if |encoding| is detectable and exceeds the position effect at any length.
- No prediction for the interaction or for r=4 vs r=2 on the PoPE arms (`MAPPOPE_R4_RESULTS.md`:
  +0.019, unmeasured).

## Checks

- Every arm 8/8 checkpoints before evaluation; the evaluator's JSON must exist before the marker.
- Convergence reported per arm (final-loss mean and range) beside every table.
- `Vanilla_r4` seeds 0-7 and `RoPE` seeds 0-7 exist in `runs/sign/p0` under this recipe; seed 0 of
  each is compared bitwise with this batch as a determinism check (not replication, rule 27). The
  batch's own arms are what the verdicts use.
