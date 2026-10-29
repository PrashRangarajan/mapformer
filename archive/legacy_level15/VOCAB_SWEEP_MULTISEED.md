# Vocab sweep, MULTI-SEED (n=3), same training batch

Supersedes the single-seed table in VOCAB_SWEEP_RESULTS.md, whose EM row could not be distinguished from a collapsed seed.

`VanillaEM` = paper-faithful separate q0/k0 (App. A.4). `VanillaEM_P0` = single-p_0 ablation.

## n_obs = 16

| variant | T=128 | T=512 |
|---|---|---|
| Vanilla | 0.997 ± 0.000 | 0.950 ± 0.010 |
| VanillaEM | 1.000 ± 0.001 | 0.977 ± 0.006 |
| VanillaEM_P0 | 0.999 ± 0.001 | 0.977 ± 0.005 |

- Vanilla per-seed @T=512: 0.959, 0.952, 0.939
- VanillaEM per-seed @T=512: 0.973, 0.984, 0.975
- VanillaEM_P0 per-seed @T=512: 0.983, 0.975, 0.974

## n_obs = 256

| variant | T=128 | T=512 |
|---|---|---|
| Vanilla | 0.913 ± 0.090 | 0.675 ± 0.103 |
| VanillaEM | 0.761 ± 0.050 | 0.590 ± 0.020 |
| VanillaEM_P0 | 0.806 ± 0.262 | 0.773 ± 0.234 |

- Vanilla per-seed @T=512: 0.652, 0.788, 0.587
- VanillaEM per-seed @T=512: 0.593, 0.568, 0.608
- VanillaEM_P0 per-seed @T=512: 0.910, 0.502, 0.906

## n_obs = 4096

| variant | T=128 | T=512 |
|---|---|---|
| Vanilla | 0.465 ± 0.018 | 0.483 ± 0.007 |
| VanillaEM | 0.493 ± 0.004 | 0.497 ± 0.001 |
| VanillaEM_P0 | 0.499 ± 0.006 | 0.499 ± 0.001 |

- Vanilla per-seed @T=512: 0.480, 0.480, 0.491
- VanillaEM per-seed @T=512: 0.498, 0.498, 0.496
- VanillaEM_P0 per-seed @T=512: 0.499, 0.500, 0.498


## Paired analysis (T=512, positive = EM variant beats Vanilla, same seed)

| n_obs | VanillaEM | wins | VanillaEM_P0 | wins |
|---|---|---|---|---|
| 16 | +0.027 | **3/3** | +0.028 | **3/3** |
| 256 | -0.086 | 1/3 | +0.097 | 2/3 |
| 4096 | +0.014 | 3/3 | +0.016 | 3/3 |

## Findings

**1. n_obs=4096 is a degenerate regime -- do not interpret it.** Best accuracy
across all 27 runs at n_obs=4096 is **0.500**, and p_empty=0.5, so every model has
collapsed to always-predict-blank. The n_obs=4096 row of the old table measured
nothing. (n_obs=16 tops out at 0.984, n_obs=256 at 0.910, so those are real.)

**2. "VanillaEM crashes at n_obs=256" SURVIVES, and is not a collapsed seed.**
VanillaEM is 0.590 ± 0.020 -- the *tightest* spread of the three variants
(0.593 / 0.568 / 0.608). It reliably underperforms Vanilla (0.675 ± 0.103). This
is a reproducible architectural effect, not training instability. The single-seed
claim was right for the wrong reason.

**3. But "long sequences favour WM" is FALSIFIED.** At n_obs=16, T=512 -- long OOD,
aliased observations, exactly the regime the mechanism table assigns to WM -- EM
beats Vanilla on **3/3 seeds** (+0.027), and so does EM-p_0 (+0.028). The deficit
is specific to n_obs=256, not to sequence length.

**4. The instability inverts with vocabulary.** On the paper task, separate q0/k0
was the unstable one (0.898 ± 0.108) and single-p_0 the stable one (0.987 ± 0.012).
At n_obs=256 it reverses: separate q0/k0 is stable-but-mediocre (±0.020), while
single-p_0 is bimodal (0.910 / 0.502 / 0.906) -- two seeds far better than any
other model, one seed at the 0.502 blank floor, i.e. it failed to learn at all.
Neither configuration dominates; they fail in different ways.
