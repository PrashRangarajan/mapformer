# bf16 autocast: NOT licensed -- keep fp32

`check_bf16.py` (criteria registered in its docstring BEFORE running), output `BF16_CHECK.json`
(per-step losses), log `runs/bf16_check.log` (gitignored; its full text is reproduced below).
2026-09-23.

## Question

Can the planned MAESTRO batch (`MAESTRO_PLAN.md`) use bf16 autocast? bf16 cannot be row-exact,
so the registered question is narrower: **does bf16 change the DIFFERENCES BETWEEN ARMS?**
Path integration cumulatively sums per-token increments over 2048 positions, so bf16 error could
bite the path-integrated arms harder than the index arms -- a confound between arms, not noise.

## Design

All four 2x2 arms (RoPE, PoPE-Flat, Vanilla = MapWM, MapPoPE-Flat) at the MAESTRO architecture
(d=384, 6 layers, 8 heads, seq 2048, batch 4, r=4), 400 AdamW steps (lr 6e-4, 50-step warmup,
wd 0.01, beta2 0.99, clip 1.0), fp32 and bf16-autocast from the SAME init on the SAME batches.
Data: real bytes from `data/code_train.bin` (MAESTRO is not tokenised yet). One run per
(arm, precision).

License only if, for every arm: (a) mean |loss_fp32 - loss_bf16| over the last 100 steps
< 0.02 nats; (b) the between-arm gap (arm - RoPE, last-100 mean) moves < 0.005 nats, well under
the 0.015 published PoPE-RoPE MAESTRO effect; (c) init logit error similar on every arm.

## Result

| arm | last-100 fp32 | last-100 bf16 | shift | mean \|dloss\| (a) | gap vs RoPE moved (b) | speedup |
|---|---|---|---|---|---|---|
| RoPE | 1.8533 | 1.8501 | -0.0032 | 0.0094 | -- | 1.22x |
| PoPE-Flat | 1.7731 | 1.7743 | +0.0012 | 0.0065 | 0.0044 OK | 1.29x |
| Vanilla (MapWM) | 2.1011 | 2.1095 | +0.0085 | 0.0148 | **0.0117 FAILS** | 1.22x |
| MapPoPE-Flat | 1.8784 | 1.8761 | -0.0023 | 0.0183 | 0.0010 OK | 1.28x |

(a) passes on every arm; (c) passes (init logit mean |diff| 0.00163 / 0.00165 / 0.00206 /
0.00180, spread 1.3x). **(b) fails on MapWM: its gap to RoPE moves 0.0117 nats**, 2.3x the
registered threshold and 78% of the 0.015 effect the MAESTRO batch would be measuring.

**Mean speedup is only 1.25x**, so the cost of keeping fp32 is small. **Verdict: keep fp32.**

## What this does and does not show

- It shows bf16 is not safe *by the registered criterion* for a 0.015-sized between-arm effect.
- It does NOT establish that the cumulative sum is the precision-sensitive part. MapPoPE also
  path-integrates and moved only 0.0010; MapWM has the largest init logit error (0.00206) but
  that is 1.3x the others, not a separate regime. With one run per cell the 0.0117 could be
  trajectory divergence after 400 steps rather than a systematic penalty. Either way the
  criterion fails and a 1.25x speedup does not justify resolving it.

## Log (verbatim)

```
RoPE          init logit |diff| max 0.0120 mean 0.00163 | last-100 |dloss| 0.0094 | fp32 7.51 it/s  bf16 9.17 it/s (1.22x)
PoPE-Flat     init logit |diff| max 0.0117 mean 0.00165 | last-100 |dloss| 0.0065 | fp32 6.27 it/s  bf16 8.10 it/s (1.29x)
Vanilla       init logit |diff| max 0.0145 mean 0.00206 | last-100 |dloss| 0.0148 | fp32 7.39 it/s  bf16 9.00 it/s (1.22x)
MapPoPE-Flat  init logit |diff| max 0.0128 mean 0.00180 | last-100 |dloss| 0.0183 | fp32 6.17 it/s  bf16 7.92 it/s (1.28x)

BETWEEN-ARM GAPS (arm - RoPE, mean loss over last 100 steps):
  PoPE-Flat     fp32 -0.0802  bf16 -0.0758  moved 0.0044 OK
  Vanilla       fp32 +0.2477  bf16 +0.2594  moved 0.0117 *** FAILS (b) ***
  MapPoPE-Flat  fp32 +0.0251  bf16 +0.0260  moved 0.0010 OK

init logit mean |diff| across arms: RoPE 0.00163, PoPE-Flat 0.00165, Vanilla 0.00206, MapPoPE-Flat 0.00180  (spread 1.3x)
mean speedup 1.25x

VERDICT: bf16 NOT licensed -- keep fp32
```
