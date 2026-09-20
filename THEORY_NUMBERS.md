# Theory numbers, recomputed from committed code (2026-09-19)

Supersedes the inline values in `THEORY_MAPPOPE.md` and `T3GEN_RESULTS.md`, which were taken at 1-4 seeds or on 2 pieces.

## (a) Full-context control: 2048-trained checkpoints, bucket NLL

| arm | 0-512 | 512-1024 | 1024-2048 | seeds |
|---|---|---|---|---|
| RoPE | 0.6533 | 0.6852 | 0.7550 | 5 |
| PoPE | 0.6206 | 0.6670 | 0.7336 | 5 |
| MapWM_r2 | 0.5592 | 0.5838 | 0.6239 | 5 |
| MapPoPE_r2 | 0.5396 | 0.5649 | 0.6120 | 5 |

## (b) Omega compression at EVAL only (512-trained checkpoints)

| arm | scale | 0-512 | 512-1024 | 1024-2048 |
|---|---|---|---|---|
| MapWM_r2 | 1.00 | 0.5382 | 0.7835 | 1.3969 |
| MapWM_r2 | 0.50 | 3.3740 | 3.5225 | 3.1168 |
| MapWM_r2 | 0.25 | 2.8826 | 3.0196 | 2.8393 |
| MapPoPE_r2 | 1.00 | 0.5235 | 1.9536 | 4.6158 |
| MapPoPE_r2 | 0.50 | 3.6832 | 3.8095 | 3.7286 |
| MapPoPE_r2 | 0.25 | 2.9145 | 3.0763 | 2.9134 |

## (c) Learned per-token phase magnitude (mean |d^q|, |d^k|, radians)

| task | arm | phase |
|---|---|---|
| Bach | MapPoPE_T3_r2 | 0.412 +/- 0.026 |
| Bach | MapPoPE_T3_r2_pi0.1 | 1.118 +/- 0.003 |
