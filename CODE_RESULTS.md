# Code modelling: the out-of-distribution result is RETRACTED

> **RETRACTED 2026-09-21 by its own registered control (C1).** Trained AND tested
> at 2048 (matched tokens/step, `runs/code2048`, seed 0, all four arms complete):
>
> | arm | position | encoding | val bpc @2048 |
> |---|---|---|---|
> | **RoPE** | index | RoPE | **0.6997** |
> | PoPE | index | PoPE | 0.7035 |
> | MapPoPE | path-int | PoPE | 0.7077 |
> | MapWM | path-int | RoPE | 0.7111 |
>
> Spread **0.0114**, RoPE BEST, MapWM last. `CODE_PREREG.md` Amendment 2 fixed
> this branch in advance: "if every arm ceilings at 2048 the way they did at 512,
> the code OOD result is an artifact of the train/test mismatch and **the framing
> is retracted**." It is. The -3.694 encoding effect and the -0.102
> "MapPoPE beats both components" both live entirely inside the extrapolation
> artifact. **Train the arms at the length you test them at and the result is gone.**
>
> The measurements below are not wrong -- they are the cost of extrapolating past
> a short training context, which is a real (and well-known) phenomenon with cheap
> standard fixes. They are NOT evidence that one encoding models code better.
>
> Process failure worth recording: this batch finished with a `.done` marker and
> four complete JSONs, and I reported "the two path-integrated arms haven't
> started" after reading two `.partial.json` files. An agent audit found it.
> **Check the completion marker, not the partial files.**


## C1 CLOSED at the registered n=3 (2026-09-23) -- the retraction becomes a reversal

All four arms trained AND tested at 2048, 3 seeds each, one batch (two runs were
lost to CUDA OOM from a blind round-robin GPU picker and re-run with an
occupancy-aware one).

| arm | s0 | s1 | s2 | mean |
|---|---|---|---|---|
| RoPE | 0.6997 | 0.7113 | 0.7069 | 0.7060 |
| **PoPE** | 0.7035 | 0.7058 | 0.7063 | **0.7052** |
| MapPoPE | 0.7077 | 0.7103 | 0.7074 | 0.7085 |
| MapWM | 0.7111 | 0.7149 | 0.7150 | 0.7137 |

| contrast | at matched length | when extrapolating 512 -> 2048 |
|---|---|---|
| ENCODING main | -0.0030 (MDE 0.0046, 2/3) **unmeasured** | -3.694 DETECTABLE |
| POSITION main | **+0.0055 (MDE 0.0033, 0/3) DETECTABLE -- against** | -- |
| MapPoPE - PoPE | **+0.0033 (MDE 0.0031, 0/3) DETECTABLE -- against** | -0.102 DETECTABLE |

**C1a is confirmed at the registered power.** The encoding effect is ~1,200x
smaller at matched length and inside its MDE. And the retracted claim does not
merely vanish -- it REVERSES: at matched training length path integration is
detectably WORSE on code, and MapPoPE is detectably behind PoPE. This agrees with
C2 (`CODE_DECAY_RESULTS.md`), where both axes are detectably worse with the
baselines repaired. Two independent matched-length batches, same verdict.

## Further corrections found in the same audit

- **"Position main effect is zero" was bucket-specific.** By bucket: 0-512
  +0.0061 (MDE 0.0067, unmeasured); **512-1024 -0.5523 (MDE 0.2127, 3/3)
  DETECTABLE, path integration HELPS**; 1024-2048 +0.0080 (MDE 0.2606,
  unmeasured -- MDE 33x the effect). Only the last cell was quoted. Rule 11:
  "unmeasured", never "zero".
- **The code 2x2 is NOT additive.** Detectable interactions in three cells, all
  against composition: bpc 512-1024 +1.024 (0/3), acc 129-512 -0.090 (0/3),
  acc 513-1024 -0.106 (0/3).
- **alpha = 1.000 was a rounded mean.** Per seed: MapWM 1.016/1.004/0.999,
  MapPoPE 1.014/1.013/**0.864**. And Dyck's MapPoPE -- the best Dyck arm -- has
  alpha 0.784 +/- 0.189 with a per-seed max of 1.01, so alpha ~ 1 does NOT
  preclude winning on Dyck. The clock/map reading of alpha was over-stated.


> **n=3 seeds, complete.** MDEs in `CODE_RESULTS_OOD.md`. Two claims from the
> n=1 read did NOT survive the extra seeds and are corrected in place below.

Pre-registration `CODE_PREREG.md` + Amendment 1 (the OOD readout was registered
before any OOD number was computed). Gates `CODE_GATES.md`. Corpus: 100.1 MB of
local Python, split by file, brackets from CPython's own tokenizer.

## 1. In distribution: a CEILING, and therefore uninformative

Trained at seq 512, evaluated at seq 512. Registered primary cell d5-8/x33-128
(floor 0.272): RoPE **1.000**, MapWM 0.996, PoPE 0.965, MapPoPE 0.959.
**13 of 17 strata sit at or above 0.98 for every arm**; overall closer accuracy
is 0.997-0.998 and full-val bpc spans 0.8826-0.8908.

**Verdict: the in-distribution readout cannot discriminate.** MapPoPE is
numerically last at the primary cell and that is a ceiling difference, NOT the
falsifier firing (rule 11). The finding that IS real: a 9-layer model on 90 MB
of Python solves bracket matching inside a 512-byte window essentially perfectly
whatever its positional encoding. Real code, at this scale, in distribution, is
not hard enough to separate these mechanisms.

## 2. Beyond the training context: the ceiling lifts and the arms separate

Same checkpoints, crop length **2048** = 4x the training context. All four load
at 2048 with zero missing or unexpected state-dict keys.

### O-A  val bpc by absolute position (lower better)

| arm | position | encoding | 0-512 | 512-1024 | 1024-2048 |
|---|---|---|---|---|---|
| **MapPoPE** | path-int | PoPE | 0.8801 | **0.7550** | **0.7727** |
| PoPE | index | PoPE | 0.8808 | 0.7977 | 0.8680 |
| RoPE | index | RoPE | 0.8834 | 2.7698 | 4.3373 |
| MapWM | path-int | RoPE | 0.8876 | 1.9939 | **4.8073** |

All four agree to 0.008 inside the training context, so the OOD spread is not a
baseline difference.

### O-B  closer-identity accuracy by bracket distance, floor re-measured on the scored set

| arm | 0-32 | 33-128 | 129-512 | 513-1024 |
|---|---|---|---|---|
| MapPoPE | 0.999 | 0.986 | 0.927 | 0.862 |
| **PoPE** | 0.999 | **0.991** | **0.953** | **0.886** |
| RoPE | 0.936 | 0.893 | 0.749 | 0.634 |
| MapWM | 0.872 | 0.852 | 0.780 | 0.659 |
| *no-stack floor* | *0.890* | *0.756* | *0.758* | *0.675* |

Distances above 512 were never seen in training by any arm. The 1025+ bin holds
under 100 scored positions and is not reported.

## What holds at n=3

All 12 runs trained in one batch. Contrasts, MDEs and sign counts in
`CODE_RESULTS_OOD.md`; seed means below.

### O-A  val bpc by position, seed means (lower better)

| arm | position | encoding | 0-512 | 512-1024 | 1024-2048 |
|---|---|---|---|---|---|
| **MapPoPE** | path-int | PoPE | 0.8828 | **0.7587** | **0.7776** |
| PoPE | index | PoPE | 0.8785 | 0.7991 | 0.8792 |
| RoPE | index | RoPE | 0.8755 | 2.6713 | 4.4641 |
| MapWM | path-int | RoPE | 0.8834 | 1.6070 | 4.5815 |

### O-B  closer accuracy by distance, seed means (higher better)

| arm | 0-32 | 33-128 | 129-512 | 513-1024 |
|---|---|---|---|---|
| MapPoPE | 0.999 | 0.986 | 0.919 | 0.848 |
| PoPE | 0.999 | 0.990 | 0.946 | 0.854 |
| RoPE | 0.927 | 0.861 | 0.738 | **0.550** |
| MapWM | 0.909 | 0.868 | 0.801 | **0.650** |
| *no-stack floor* | *0.890* | *0.756* | *0.758* | *0.675* |

### ESTABLISHED (detectable, 3/3 seeds)

- **The encoding is what survives the training context.** PoPE - RoPE =
  **-3.585 bpc** at 2-4x (MDE 0.166). The RoPE-encoding arms blow up
  (4.46 / 4.58) and the PoPE-encoding arms do not (0.88 / 0.78).
- **MapPoPE beats BOTH of its components at 2-4x**: **-0.102 against PoPE**
  (MDE 0.052, 3/3) and **-3.804 against MapWM** (MDE 0.382, 3/3). This is the
  composition claim that `ENWIK8_SEEDS.md` had the seeds on the wrong comparison
  to test, and it fires here on the axis PoPE's own paper heads with.
- **MapPoPE - MapWM on closer accuracy: +0.117 / +0.118 / +0.198** at distances
  33-128 / 129-512 / 513-1024, every one detectable, 3/3.
- **Both RoPE-encoding arms fall BELOW the no-stack floor at the longest
  distance**: RoPE 0.550 and MapWM 0.650 against a floor of 0.675. A model with
  no memory at all beats them there.

### NOT established

- **MapPoPE - PoPE at 512-1024**: -0.0404 against an MDE of 0.0429. 3/3 seeds
  favour it and it still misses. Unmeasured.
- **MapPoPE - PoPE on closer accuracy**: -0.005 / -0.027 / -0.005, sign AGAINST
  MapPoPE, 1/3, 0/3, 1/3, all unmeasured. So the bpc win does NOT reproduce on
  the stack-sensitive metric; it does not reverse either.
- **MapWM - RoPE at 1024-2048**: +0.118 against an MDE of 0.529, 1/3.

### CORRECTED from the n=1 read

- **"O1 refuted and inverted -- MapWM extrapolates worse than plain RoPE" does
  NOT survive seeds.** At n=3, MapWM - RoPE is **-1.064 at 512-1024 (MDE 0.445,
  3/3) DETECTABLE**, i.e. path integration HELPS there, and the 1024-2048
  difference is unmeasured. Seed 0's 4.81-vs-4.34 was one seed. O1 is partly
  confirmed at the nearer bucket, not inverted.
- **The metric disagreement is weaker than n=1 suggested.** At n=1 PoPE led on
  closer accuracy at every distance; at n=3 that lead is unmeasured everywhere.
  The honest statement is that the bpc win is not corroborated by the
  stack-sensitive metric, not that the two rank the arms oppositely.

### Rule 9 and loss overlap

- r(final train bpc, OOD bpc at 1024-2048) = **-0.097** over 12 runs. The OOD
  effect is NOT the training loss in disguise -- unusual for this project, where
  that correlation is normally near -0.99, and it holds because the OOD metric
  is measured outside the regime the loss was computed in.
- MapPoPE and PoPE final training losses OVERLAP, so a loss-matched residual
  would be permissible here; it is not needed given r.


## The advantage is NOT a stack effect (`probe_code_depth0.py`, registered before running)

Partitioning every predicted byte in the far bucket by the nesting depth at that
byte, reconstructed from genuine bracket pairs (so brackets inside strings and
comments never count):

| MapPoPE - PoPE, bpc at 1024-2048 | effect | verdict |
|---|---|---|
| all far positions | -0.1016 (MDE 0.0521) | DETECTABLE 3/3 |
| **depth 0 -- NO open bracket** | **-0.0998 (MDE 0.0510)** | DETECTABLE 3/3 |
| depth >=1 | -0.1047 (MDE 0.0560) | DETECTABLE 3/3 |
| depth >=3 | -0.0936 (MDE 0.0120) | DETECTABLE 3/3 |

Flat across depth; **98% of the advantage survives where no bracket is open at
all** (depth 0 is 1,919,002 of 2,993,152 scored bytes). Same for the encoding
gap: MapPoPE - MapWM is -3.70 at depth 0 and -3.99 at depth >=1.

**THE EXPERIMENT'S RATIONALE IS NOT SUPPORTED.** The reason for running a code
corpus was that Dyck's positional variable is signed push/pop and real code
nests, so MapPoPE's Dyck win should transfer. Three measurements say otherwise:
1. the direct test of nesting (in-distribution bracket accuracy) is a CEILING;
2. the accumulator learns **alpha = 1.000** -- a clock, not the signed map
   (alpha 0.609) MapWM learns on Dyck (`probe_code_accum.py`);
3. the advantage is **depth-independent**, surviving in full at depth 0.

What remains is a pure CONTEXT-LENGTH effect. It is real and detectable, but no
code corpus was needed to find it, and the nesting account gets no support here.
Fifth failed Dyck -> elsewhere generalisation in this line.

## Caveats that must travel with these numbers

- **n=3.** Small for this project. The MapPoPE-over-PoPE margin clears its MDE
  at 2-4x but not at 1-2x, and one n=1 claim in this very file dissolved under
  the extra two seeds -- treat +/-0.05 bpc contrasts here as provisional until n=8.
- **Budget-limited.** Every arm still has a negative val slope at 36k
  (-0.0015 to -0.0142 per 3k iters), so the ordering is not settled. Same
  condition as the enwik8 36k runs.
- **P1 missed, marginally.** Registered "overall bpc separates by less than
  0.01"; the best-val spread is 0.0139 on the trainer's 40-batch estimator
  (0.0082 on the full-val estimator the eval uses). Recorded as a miss.
- RoPE's blow-up past its training context qualitatively reproduces PoPE's own
  Figure 2, which is a check on the setup rather than a new result.
