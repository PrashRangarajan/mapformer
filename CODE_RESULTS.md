# Code modelling: MapPoPE beats both components BEYOND the training context

> **PROVISIONAL, n=1 SEED.** Seed 0 only; no MDE, nothing here is established.
> Seeds 1 and 2 are training (`runs/code`). Do not cite a number from this file
> until the MDE columns exist. Recorded now so a compaction loses nothing.

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

## What the seed-0 numbers say, and what they contradict

- **O2 resolves toward Dyck, against Bach.** MapPoPE does not collapse past its
  training context -- it is the best arm on bpc in both OOD buckets and **beats
  BOTH components**: -0.043 and -0.095 against PoPE with the gap growing, and
  0.7727 against MapWM's 4.8073. `JSB_LENGTH_RESULTS.md` had MapPoPE collapsing
  to 4.616 in the equivalent bucket; that does not reproduce here.
- **O1 REFUTED, and inverted.** The registered prediction was the JSB shape:
  path-integrated arms degrading less than index arms. On the RoPE row the
  opposite happens -- **MapWM (4.8073) extrapolates WORSE than plain RoPE
  (4.3373)**. Path integration helps carried on PoPE's encoding and hurts
  carried on RoPE's. On Bach, MapWM - RoPE was -0.662, i.e. it helped.
- **The two metrics disagree.** bpc ranks MapPoPE first; closer-identity accuracy
  ranks PoPE first at every distance. Both are reported, neither dropped. This is
  the third such disagreement in the project (see `CROSS_RESULTS.md`).
- **Both RoPE-encoding arms fall to or below the no-stack floor out of
  distribution**: MapWM 0.872 at distance 0-32 against a floor of 0.890, and
  0.659 at 513-1024 against 0.675.

## Caveats that must travel with these numbers

- **n=1.** No MDE. The bpc effects are a factor of six and unlikely to be seed
  noise, but the MapPoPE-over-PoPE margin (-0.043 / -0.095) is exactly the size
  that has dissolved under seeds repeatedly in this project.
- **Budget-limited.** Every arm still has a negative val slope at 36k
  (-0.0015 to -0.0142 per 3k iters), so the ordering is not settled. Same
  condition as the enwik8 36k runs.
- **P1 missed, marginally.** Registered "overall bpc separates by less than
  0.01"; the best-val spread is 0.0139 on the trainer's 40-batch estimator
  (0.0082 on the full-val estimator the eval uses). Recorded as a miss.
- RoPE's blow-up past its training context qualitatively reproduces PoPE's own
  Figure 2, which is a check on the setup rather than a new result.
