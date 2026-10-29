# Do the forget gate and PoPE bound the accumulator?

Pre-registered in `probe_accumulator.py`. Eval-only.

`alpha` is the exponent of `range ~ T^alpha` between the two lengths;
0.5 is a diffusive random walk, 1.0 is ballistic drift.

| arm | range(S) T=128 | T=1024 | **alpha(S)** | range(theta) T=1024 | extra |
|---|---|---|---|---|---|
| `Vanilla (forget batch)` | 60.1 | 263.1 | **0.710** | 237.3 | — |
| `Forget` | 53.0 | 205.6 | **0.652** | 178.9 | sum log-gamma 0.38->2.75 (alpha +0.956) |
| `Vanilla (L15 batch)` | 50.7 | 188.2 | **0.631** | 248.9 | — |
| `Level15` | 62.8 | 228.9 | **0.622** | 283.9 | theta_hat 78.6->285.6 (alpha +0.621); vs theta_path 283.9 |
| `Vanilla (PoPE batch)` | 60.1 | 263.1 | **0.710** | 237.3 | — |
| `MapPoPE` | 48.9 | 204.7 | **0.689** | 161.6 | — |

---

# Verdict

Per-seed exponents, paired within batch (`alpha` from `range(S)` at T=128 vs 1024):

| contrast | delta alpha | sd | MDE | n | verdict |
|---|---|---|---|---|---|
| `Forget` − `Vanilla` | −0.051 | 0.081 | 0.092 | 6 | unmeasured |
| `MapPoPE` − `Vanilla` | +0.006 | 0.155 | 0.177 | 6 | unmeasured |
| `Level15` − `Vanilla` | +0.009 | 0.152 | 0.191 | 5 | unmeasured |

For scale, the spread the growth law was built on is **0.518 → 0.943 = 0.425**.
These three are 5–70x smaller than that and every one is inside its MDE.

## P1 — the POSITIVE CONTROL FAILED, and it takes a claim with it

`Level15`'s corrected angle is **not** bounded relative to its path angle. At
T=1024, `range(theta_hat) = 285.6` against `range(theta_path) = 283.9`, and its
exponent is `alpha = 0.621` — indistinguishable from the uncorrected arm.

The reason is structural and should have been read off the code before the claim was
made: the InEKF wraps the **innovation** `atan2(sin(z − theta), cos(z − theta))`, not
`theta_hat`. So `theta_hat = theta_path + (bounded correction)` and its range tracks
`theta_path` exactly. **"A wrapped filter bounds the accumulator" is false** and is
withdrawn from both documents, where it appeared twice.

## P2, P3 — neither the forget gate nor PoPE changes the accumulator

Both unmeasured against their own batch's control. Note also that the two `Vanilla`
baselines differ across batches by 0.057 in `alpha` (0.665 vs 0.608), which is larger
than either effect — so between-batch variation alone would swamp them.

## P4 — the forget gate has a SECOND accumulator, and it is ballistic

`sum(log gamma)` grows 0.38 → 2.75 from T=128 to T=1024, `alpha = +0.956`. The gate
adds an unbounded magnitude accumulator alongside the unchanged phase accumulator.
Whether that is what its +0.086 buys is not established here.

## What this does to the synthesis

**The growth law explains the sign result and the rank result. It does not explain
the forget gate, PoPE, or the InEKF**, all three of which show the same OOD-length
signature while leaving `alpha` untouched. "One quantity predicts almost everything"
is too strong; the correct statement is that one quantity accounts for two of the
four mechanisms, and the OOD axis still has at least one other cause.

The account also loses its proposed intervention. Nothing measured here bounds the
accumulator, so the prediction "bounding it should help at length" remains untested
rather than confirmed — the arm that was supposed to demonstrate it does not do it.
