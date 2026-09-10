# N5: coherence is load-bearing, but it does NOT invert by task

4 arms x 2 tasks x 8 seeds, each task one batch. Pre-registration: `N5_PREREG.md`.
Arms differ ONLY in the phases of the position kernel: `k_0` is a per-block
rotation of `q_0`, so `sum_i a_i` is identical to 6 d.p. across conditions, and
both vectors are frozen (verified to hold `rho` to 1e-6 through training).

| arm | `rho` | torus T=1024 (map) | recency T=1024 (clock) |
|---|---|---|---|
| `EMPhase_plus_r4` | **+1** | **0.959 +/- 0.025** | 0.309 +/- 0.152 |
| `EMPhase_minus_r4` | **-1** | **0.943 +/- 0.029** | 0.350 +/- 0.246 |
| `EMPhase_zero_r4` | 0 (coherent) | 0.638 +/- 0.132 | 0.118 +/- 0.020 |
| `EMPhase_rand_r4` | ~0 (random) | 0.679 +/- 0.101 | 0.182 +/- 0.110 |

Torus at T=128 is 1.000 / 1.000 / 0.976 / 0.986 -- at ceiling, exactly as the
pre-registration anticipated, which is why T=1024 was named the readout in advance.

## P1 REFUTED, for a reason I could have derived before spending the GPU

Predicted `plus > zero ~ rand > minus`. Measured:

| contrast (torus, T=1024) | delta | sd | MDE | seeds + | verdict |
|---|---|---|---|---|---|
| `plus - minus` | **+0.016** | 0.034 | 0.034 | 6/8 | **unmeasured** |
| `plus - zero` | **+0.321** | 0.112 | 0.110 | 8/8 | **DETECTABLE** |
| `minus - zero` | **+0.304** | 0.119 | 0.117 | 8/8 | **DETECTABLE** |

`rho = -1` is as good as `rho = +1`. The algebra says it must be:

    kappa_minus(dS) = sum_i a_i cos(omega_i dS + pi) = - kappa_plus(dS)

and EM's score is `A_X (*) A_P` with `A_X` LEARNED, so
`A_X (*) (-kappa) = (-A_X) (*) kappa`. **The sign of the kernel is absorbable by
the content branch; the coherence MAGNITUDE is not.** `rho = -1` was a relabelling,
not an ablation. This is the same error as the sign-ablation discriminator: a
verdict cell that could not have gone the other way, here for an algebraic reason
sitting in plain view.

**So the axis is `|rho|`, not `rho`.** Restated and measured:

| contrast | delta | sd | MDE | seeds + | verdict |
|---|---|---|---|---|---|
| torus, `\|rho\|=1 - \|rho\|=0` | **+0.292** | 0.064 | 0.063 | **8/8** | **DETECTABLE** |
| recency, `\|rho\|=1 - \|rho\|=0` | +0.180 | 0.186 | 0.184 | 7/8 | unmeasured |

## P2 -- the registered central claim -- NOT SUPPORTED

    (plus - minus)_torus - (plus - minus)_recency = +0.058,  MDE 0.182  UNMEASURED

and restated on the axis that actually moves:

    (|rho|=1 - |rho|=0)_torus - (same)_recency  = +0.113,  MDE 0.195  UNMEASURED

**There is no inversion.** Coherence helps on BOTH tasks, with the same sign. The
theory's headline corollary -- that coherence is a choice of what the kernel is
tuned to, inverting between map and clock the way signed-vs-monotone does on the
accumulator -- is refuted by the test built to check it.

## P3 -- not falsified, but the recency half is degenerate

`plus` is not the best recency arm (`minus` 0.350 vs `plus` 0.309), so P3 as
written survives. It should not be credited. **Every frozen arm is far below the
trainable EM arms on recency** -- best frozen 0.350 against `VanillaEM_r4` 0.837,
`VanillaEM_P0_r4` 0.600 and WM 0.975 -- with final losses 2.02-2.64 against a
chance of 2.77. These arms barely left chance. Rule 10: do not compare unconverged
arms. The recency half of N5 carries little information, and the `|rho|` row there
(MDE 0.184) reflects that.

Freezing costs almost nothing on the torus (0.959 at `|rho|=1`) and is
catastrophic on recency. That contrast is itself interesting, but this batch
contains no trainable arm, so it cannot be tested here without violating rule 3.

## P4 -- `rho` is a sufficient summary at this resolution

`zero` (coherent quarter turn) vs `rand` (random phases), same `rho ~ 0`:
torus **-0.040** (MDE 0.170), recency **-0.064** (MDE 0.105). Both unmeasured. The
*shape* of an incoherent kernel does not matter beyond `kappa(0)`; only whether it
is a matched filter at all.

## What this does to the theory

**Theorem 2 survives in modified form and is now the best-supported part of the
frame.** `|rho|` is load-bearing on the map task at 8/8 seeds, established by a
magnitude-matched intervention rather than a correlation -- which is more than N4
delivered. The correction is that `rho` enters through its magnitude, because
`kappa`'s sign is a gauge the content branch can absorb.

**The corollary the theory was built to explain is dead.** Coherence does not
invert by task kind. So the original observation that motivated all of this --
single `p_0` beats separate `q_0/k_0` on three map tasks (+0.089/+0.167/+0.358) and
LOSES on recency (-0.237) -- is *not* explained by coherence. Both forms would want
`|rho|` high, and this batch shows high `|rho|` helping on both tasks.

**The surviving candidate, and it is third-generation and post-hoc:** the two forms
differ in *degrees of freedom*, not in coherence. Single `p_0` has `q_0 = k_0`
permanently and therefore NO phase freedom; the separate form has `n_b` phases it
can move. On a map task the matched filter is already correct so freedom is worth
nothing; on a clock task the kernel must be reshaped and freedom is worth a lot.
N5's own freeze result is consistent -- freezing is free on the torus and
catastrophic on recency -- but consistency is not evidence, this batch has no
trainable control, and I have now been wrong about this mechanism twice. It needs
its own pre-registered test with trainable and frozen arms of both forms in one
batch, and it should not be written into either document until it has one.
