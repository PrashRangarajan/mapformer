# EM on Match-Query — does single-p_0 help where the task is MATCHING?

All three arms trained in ONE batch, TE=512 TQ=256, 200 epochs, n=3.
Held-out env (seed=10000). **Chance 0.0625.** Gates: `MATCH_QUERY_GATES.md`.

Match-Query is the only task in this repo whose shortcuts are gated, and it
tests position MATCHING — which is what A_P is for. So it is the most
informative place to ask whether the q0/k0 parameterisation matters.

| variant | T_query=256 | T_query=512 (OOD) |
|---|---|---|
| MapWM-Flat (WM) | 0.888 ± 0.140 | 0.902 ± 0.117 |
| MapEM separate q0/k0 (paper-faithful) | 0.450 ± 0.332 | 0.385 ± 0.323 |
| MapEM single p_0 (ablation) | 0.808 ± 0.168 | 0.789 ± 0.188 |

## Per-seed (T_query=256)

| variant | s0 | s1 | s2 |
|---|---|---|---|
| MapWM-Flat (WM) | 0.731 | 1.000 | 0.934 |
| MapEM separate q0/k0 (paper-faithful) | 0.107 | 0.769 | 0.475 |
| MapEM single p_0 (ablation) | 0.736 | 1.000 | 0.689 |

Reference: the earlier 6-variant sweep gave Vanilla 0.888 ± 0.140 under
identical settings. The Vanilla arm here is the reproducibility check.

## Verdict

**Largest version of the single-p_0 effect measured anywhere.** Paired per seed,
single-p_0 minus separate-q0/k0: **+0.629 / +0.231 / +0.214, mean +0.358, 3/3.**
Seed 0 of the paper-faithful form collapses to **0.107**, barely above the 0.0625
floor.

Effect size by task, for context:

| task | single-p_0 advantage |
|---|---|
| paper task (held-out revisit accuracy) | +0.089 |
| compositional (cross_nb) | +0.167 |
| **Match-Query (gated + ablated)** | **+0.358** |

The ordering is not arbitrary: Match-Query is pure position MATCHING -- retrieve
what you saw at the cell you now occupy -- which is exactly what A_P exists to do.
The more the task leans on A_P, the more the parameterisation matters.

**Control passes:** the WM arm reproduces the sweep's 0.888 to three decimals, so
this batch is comparable and the EM rows can be read against it.

**The paper's conjecture is refuted on four independent tasks.** App. A.4 states
the separation "would create sparser attention values" and flags it as a
suspicion, never measured. Every measurement runs the other way.

### Still unexplained, and deliberately not explained

The A_P kernel-geometry account was FALSIFIED on a pre-registered test
(`AP_KERNEL_DIAGNOSTIC.md`): the configuration with the worst kernel -- negative
at zero displacement on 100% of revisit pairs -- is the one where EM beats WM on
3/3 seeds at n_obs=16. Single-p_0 reliably wins and the reason is unknown. Three
separate mathematically-real properties failed to predict performance today
(kernel geometry, theta drift, non-commutativity), so no fourth mechanism is
proposed here.

### EM still never beats WM

0.808 (single-p_0) vs 0.888 (WM). The correction repairs EM to near-parity, not
superiority. The paper's EM >= WM ordering does not reproduce on any task we
have measured.
