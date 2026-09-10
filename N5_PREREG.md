# Pre-registration: N5 -- setting the position kernel's coherence

Written before any checkpoint in `runs/n5_phase/` exists. N5 in
`THEORY_KERNEL.md`, replacing N4 (tested, null: `r(rho_init, final_loss)` = +0.142
and +0.029). N4 was a correlation and `rho` neither predicts a seed's fate nor
self-corrects during training, so the causal test is to **set** `rho`.

## The intervention

EM's position kernel is `kappa(dS) = sum_i a_i cos(omega_i dS + phi_i)` with
coherence `rho = kappa(0)/sum_i a_i = (sum_i a_i cos phi_i)/sum_i a_i`.
`model_em_phase.py` builds `k_0` as a per-block ROTATION of `q_0`, so

    a_i = |q_0i|^2   is IDENTICAL across conditions;   only phi_i differs.

Verified at construction: `sum_i a_i` agrees to 6 decimal places across all four
arms, per-block magnitudes match to 1e-6, and `rho` hits its target exactly.
Both vectors are FROZEN, so `rho` stays at its set point (a trainable pair drifts
0.25-0.27, `probe_ap_coherence.py`).

**Why magnitude-matching is the whole design.** The recency gate ablation
pre-registered a condition that collapsed to 0.086 -- and a control changing only
`theta`'s SCALE collapsed just as hard (0.110). Nothing was established until a
magnitude-matched pair was run. Same trap here: a bad `rho` also changes how big
`A_P` is unless `k_0` is built as a rotation.

| arm | `phi_i` | `rho` | reading |
|---|---|---|---|
| `EMPhase_plus_r4` | 0 | **+1** | `kappa` peaked at `dS = 0` -- matched filter for "same place" |
| `EMPhase_zero_r4` | `pi/2` | **0** | coherent quarter turn: isolates the phase VALUE from its RANDOMNESS |
| `EMPhase_minus_r4` | `pi` | **-1** | `kappa` MINIMISED at `dS = 0` |
| `EMPhase_rand_r4` | `U(0,2pi)` | `~0` | the paper-faithful draw, now magnitude-matched |

4 arms x 2 tasks x 8 seeds = 64 runs, each task in ONE batch (rule 3).

- **torus paper task** (map, `delta == 0` constant): `train_variant`, 300 ep,
  98 batches of 128, `T=128`, eval `T` in {128, 512, 1024}, lr 1e-3 cosine, 1
  layer, d=128 -- the `run_sign.sh` recipe.
- **recency** (clock, `delta = k(t)` per query): `train_recency`, `k_max=64`,
  `T=1024`, eval {1024, 2048}, 300 ep, 48 batches of 16, lr 1e-3 cosine -- the
  `run_recency_em.sh` recipe, no `--fast-attn`.

## Predictions

**P1 (torus, map).** `rho = +1` best, `rho = -1` worst:
`plus > zero ~ rand > minus`. Zero displacement is the retrieval target, so a
kernel peaked there helps and a kernel minimised there is exactly wrong.
*Falsified if* `plus - minus` on the torus is inside its MDE, or negative.

**P2 (the claim -- the INTERACTION).** `rho`'s effect differs by task kind:

    (plus - minus)_torus  -  (plus - minus)_recency   >  0, clearing its MDE.

This is the confident prediction and the one the theory stands on. It says
coherence is not a quality axis but a *choice of what the kernel is tuned to* --
the same structure as signed-vs-monotone on the accumulator, one level up.
*Falsified if* the interaction is inside its MDE, or negative.

**P3 (recency, clock) -- deliberately WEAK.** The theory says a clock task needs
`kappa` injective over the range, NOT peaked at zero; it does not say `rho = -1`
is optimal. Registered only as: **`plus` is not the best arm on recency.** I am
not predicting the internal ordering of `zero`/`rand`/`minus`.
*Falsified if* `plus` is best on recency by more than its MDE.

**P4 (coherent vs incoherent zero).** `zero` and `rand` both have `rho ~ 0` but
`zero` is coherent (every block a quarter turn) and `rand` is not. If only `rho`
matters they tie; if the *shape* of `kappa` matters beyond `kappa(0)` they do not.
No registered direction -- this is the one exploratory cell, and it is the cheapest
available probe of whether `rho` is a sufficient summary of the kernel.

## Power

Recency `EM` arms had seed sd 0.126 -> **MDE 0.125** at n=8. Torus EM arms have
historically run sd 0.01-0.10; at sd 0.08 the MDE is 0.079. The interaction MDE is
larger than either main effect's (independent batches, so variances add):
expect **~0.15**. If `plus - minus` on the torus is not at least ~0.15 this test
cannot resolve P2, and that will be reported as unmeasured rather than null.

**Ceiling check** (rule 11, and the error that cost P3 in the last batch): the
torus paper task runs at 0.95-1.00 for healthy arms at `T=128`, so the informative
readout there is `T=1024`, where the sign ablation's arms spanned 0.75-0.95. A
verdict cell that could not have gone the other way is not a test.
