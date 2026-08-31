# Does a loop help where path integration is NOT sufficient?

The looped pilot found recursion substitutes for depth in the INDEX arm
(+0.363, sd 0.018, 3/3 seeds) but nothing on top of PATH INTEGRATION
(+0.046, sd 0.074, MDE 0.120, one seed negative). That null was
uninterpretable -- MapFormer already scored 0.948 on the torus, leaving 0.052
of headroom. A ceiling cannot distinguish 'the loop adds nothing' from 'there
was nothing to add'.

Match-Query 128^2 leaves **0.177** of headroom (published
path-integrated 0.823, index 0.192, chance 0.0625), so the loop has
room to show an effect if it has one. All arms retrained in ONE batch with
warmup+cosine and fast-attn, so the published number is a reference, not a
baseline. n=3, scored at TQ=256.

| arm | params | accuracy | per-seed |
|---|---|---|---|
| path integration, no loop | 204,630 | **0.456** ± 0.220 | 0.398, 0.449, 0.800, 0.320, 0.349, 0.705, 0.110, 0.520 |
| path integration + LOOP x4 | 204,630 | **0.870** ± 0.099 | 0.766, 1.000, 0.772, 0.838, 1.000, 0.868, 0.942, 0.777 |
| index, no loop | 204,182 | **0.108** ± 0.025 | 0.063, 0.148, 0.114, 0.093, 0.126, 0.103, 0.103, 0.115 |
| index + LOOP x4 | 204,182 | **0.207** ± 0.032 | 0.215, 0.203, 0.149, 0.227, 0.198, 0.261, 0.193, 0.210 |
| path integration, 3 REAL layers | 601,174 | **0.771** ± 0.263 | 0.836, 0.812, 0.143, 0.809, 0.834, 0.920, 0.814, 1.000 |

chance = 0.0625, perfect = 1.000

## The four pre-registered questions

**Q1 sanity.** path integration, no loop − index, no loop = **+0.348** (sd 0.218, MDE 0.215, n=8, per-seed +0.335, +0.302, +0.686, +0.227, +0.223, +0.602, +0.007, +0.405, 8/8 positive)

Path integration is necessary on this task, as published. The rest is readable.

**Q2 THE QUESTION.** path integration + LOOP x4 − path integration, no loop = **+0.414** (sd 0.279, MDE 0.277, n=8, per-seed +0.368, +0.551, -0.029, +0.518, +0.651, +0.163, +0.832, +0.257, 7/8 positive)

**The loop COMPLEMENTS path integration once there is headroom.** The torus null was a ceiling artifact, and a recursive MapFormer is worth building.

**Q3 the reverse.** index + LOOP x4 − index, no loop = **+0.099** (sd 0.045, MDE 0.045, n=8, per-seed +0.152, +0.055, +0.035, +0.133, +0.072, +0.158, +0.090, +0.095, 8/8 positive)

A loop helps the arm with headroom regardless of position code.

**Q4 loop vs real depth.** path integration + LOOP x4 − path integration, 3 REAL layers = **+0.099** (sd 0.255, MDE 0.252, n=8, per-seed -0.070, +0.188, +0.629, +0.029, +0.166, -0.052, +0.128, -0.223, 5/8 positive)

Looping does not beat real depth here; check the sign and the flat arm above to see whether it matches or falls short.

Headroom actually available to the loop was 0.544 (1.0 − 0.456); it captured 76% of it.

## Scope

One task, one map size, one loop count (4), n=3. The loop here is the
conservative form: a shared block with no per-iteration depth embedding, and
theta computed once rather than refined per iteration. A negative on Q2 does
NOT rule out the refine-theta variant, which is a different model.

---

## Final, all five arms at n=8

| arm | params | mean | sd | min | per-seed |
|---|---|---|---|---|---|
| index, no loop | 204,182 | 0.108 | 0.025 | 0.06 | 0.06 0.15 0.11 0.09 0.13 0.10 0.10 0.12 |
| index + loop×4 | 204,182 | 0.207 | 0.032 | 0.15 | 0.22 0.20 0.15 0.23 0.20 0.26 0.19 0.21 |
| path-int, 1 layer | 204,630 | 0.456 | 0.220 | **0.11** | 0.40 0.45 0.80 0.32 0.35 0.71 0.11 0.52 |
| **path-int + loop×4** | **204,630** | **0.870** | **0.099** | **0.77** | 0.77 1.00 0.77 0.84 1.00 0.87 0.94 0.78 |
| path-int, 3 real layers | 601,174 | 0.771 | 0.263 | **0.14** | 0.84 0.81 0.14 0.81 0.83 0.92 0.81 1.00 |

Chance 0.0625. TQ=256, 300 ep warmup+cosine, fast-attn, all arms in one batch.

| question | effect | sd | MDE | seeds | verdict |
|---|---|---|---|---|---|
| Q2 loop on path integration | **+0.414** | 0.279 | 0.277 | 7/8 | **DETECTABLE** |
| Q3 loop on index | **+0.099** | 0.045 | 0.045 | 8/8 | **DETECTABLE** |
| Q4 loop vs 3 real layers | +0.099 | 0.255 | 0.252 | 5/8 | underpowered |
| 2×2 interaction | **+0.315** | 0.283 | 0.281 | — | **DETECTABLE** |

### What this establishes

**Path integration and looping both help, and they compose super-additively.** The
loop is worth +0.414 on the path-integrated arm and +0.099 on the index arm — an
interaction of +0.315, detectable. Neither ingredient alone comes close: index+loop
reaches 0.207, path-int alone 0.456, both together **0.870**.

This is the case the torus could not show, because 1-layer path integration already
scored 0.948 there. Headroom was the missing ingredient, not a different mechanism.

### RETRACTED from the n=3 read: "the loop BEATS three real layers by +0.273"

At n=3, PI_L3 was 0.836 / 0.812 / **0.143** and I reported the loop beating depth.
At n=8 the failure rate is **1/8**, L3's mean is 0.771, and Q4 is **+0.099,
underpowered** (5/8 positive). **The loop MATCHES real depth, it does not beat it** —
at a third of the parameters.

(Earlier still, at n=2, I read L3's 0.824 as "reproduces the published 0.823". Three
successive claims off n≤3 in one session, each dissolved by more seeds.)

### The most robust finding is STABILITY, not the mean

The loop arm never fails: **8/8 seeds ≥ 0.77, sd 0.099**. Both alternatives do fail —
1-layer path integration ranges 0.11–0.80 (sd 0.220) and three real layers 0.14–1.00
(sd 0.263). So the loop's contribution is mostly to the FLOOR: it converts an
unreliable model into a reliable one at constant parameters, rather than raising the
best case. The single seed where the 1-layer baseline trained well (0.800) is the one
seed where the loop did not help (−0.029).

That is also why Q2 clears its MDE only narrowly despite a large mean — the variance
being removed is the baseline's, so the paired differences inherit it.
