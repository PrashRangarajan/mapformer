# MapPoPE at r=4, and the P4 repair

4 arms x 8 seeds torus + 2 arms x 8 seeds recency, one batch each. Recipes verbatim
from `run_sign.sh` / `run_recency.sh`. Pre-registration: `MAPPOPE_R4_PREREG.md`.

| arm (torus) | final loss | flat | T=512 | T=1024 |
|---|---|---|---|---|
| `MapPoPE_r4` | **0.00021** | 8/8 | **0.982 ± 0.018** | **0.941 ± 0.028** |
| `MapPoPE-Flat` (r=2) | 0.00635 | 6/8 | 0.974 ± 0.011 | 0.921 ± 0.023 |
| `Gated_r2` | 0.03954 | 5/8 | 0.962 ± 0.048 | 0.887 ± 0.105 |
| `Vanilla` (r=2) | 0.08863 | **2/8** | 0.915 ± 0.052 | 0.777 ± 0.094 |

r(final loss, accuracy) = **-0.81 / -0.68**, so every contrast is given raw and
loss-matched.

## Q-A: r=4 helps MapPoPE far less than it helps MapWM

| contrast | raw | loss-matched | seeds + | verdict |
|---|---|---|---|---|
| `MapPoPE_r4 - MapPoPE-Flat`, T=512 | +0.009 | +0.006 | 5/8 | unmeasured |
| `MapPoPE_r4 - MapPoPE-Flat`, T=1024 | +0.019 | +0.015 | 5/8 | unmeasured |

**Reference: the same upgrade on the MapWM family is +0.085 at T=1024 (8/8, t=3.57).**
Here it is a quarter of that and does not clear its own MDE on 5 of 8 seeds.

This is the pre-registered alternative, written before the run: *"if it does NOT
reproduce, the rank effect is specific to the RoPE-style pairing and does not
transfer to PoPE's one-frequency-per-element layout."* That reading is now live, and
there is a mechanism for it in the construction: PoPE uses `n_blocks = d_head = 64`
against MapWM's 32, so it already has twice the frequency channels and the r=2
bottleneck is correspondingly less binding.

**The practical consequence is the opposite of what motivated the run.** MapPoPE was
flagged as "the best arm, never given the free rank upgrade". The upgrade is real in
direction but small, so MapPoPE's standing does not change and `r=4` should not be
quoted as a general recommendation -- it is a recommendation for the **MapWM
family**, where it was measured.

## The within-batch result that does matter: PoPE on top of path integration

| contrast | raw | loss-matched | seeds + |
|---|---|---|---|
| `MapPoPE-Flat - Vanilla`, T=512 | **+0.058** (8/8) DETECTABLE | +0.024 (MDE 0.034) | 6/8 |
| `MapPoPE-Flat - Vanilla`, T=1024 | **+0.144** (8/8) DETECTABLE | +0.082 (MDE 0.104) | 6/8 |

Raw, this is 8/8 at both lengths and the cleanest confirmation yet that PoPE's
magnitude pays on top of a path-integrated phase -- previously supported across
batches, now within one. **But `Vanilla` is the worst-converging arm in the set**
(final loss 0.089, flat on 2/8 seeds against MapPoPE's 6/8), and loss-matched the
contrast drops to +0.024/+0.082 and stops clearing its MDE. So the effect is real in
direction and **partly a convergence gap**: PoPE makes this configuration easier to
train, and how much is left after that is not resolved at n=8.

## P4: directionally against my own recorded prior, and still unmeasured

`GATED_PREREG` asked whether an explicit gate helps `r=2` more than `r=4`, and the
gated batch could not answer it because `Vanilla_r2` was missing. Repaired:

| | gate effect | verdict |
|---|---|---|
| torus T=512, `Gated_r2 - Vanilla` | +0.046 raw / +0.026 matched (5-7/8) | unmeasured |
| torus T=1024, `Gated_r2 - Vanilla` | +0.110 raw / +0.073 matched (5-6/8) | unmeasured |
| recency T=2048, `Gated_r2 - Vanilla` | +0.015 (3/8) | unmeasured |
| **interaction** (r=2 effect minus r=4 effect, recency) | **+0.054** | cross-batch |

**Every cell is unmeasured, and every cell points the way P4 predicted** -- the gate
helps at r=2 (+0.046, +0.110, +0.015) where it did nothing at r=4 (+0.004, +0.003,
-0.039). I recorded a negative prior for this before the run, and the data is mildly
against that prior rather than confirming it.

Two reasons not to promote it. The effect is largest exactly where the baseline
converges worst (`Vanilla` at 2/8 flat), and loss-matching cuts it by a third; that
is the signature of an optimisation effect, which is what nearly everything in this
project has turned out to be. And the r=4 half of the interaction comes from a
different batch, so the interaction is not a single within-batch measurement.

**Verdict: the gate substituting for an impaired separator is consistent with the
data and not established by it.** The clean test would be all four cells in one
batch, which is 32 runs.
