# Does recursion buy attention horizon cheaply?

Weight-shared looped block (1 block x4, ALBERT-style, no per-iteration
depth embedding) against 1 real layer and 4 real layers. Torus paper task,
d=128, 2 heads, 300 epochs, 5% warmup + cosine, n=3 seeds.

**All three configs were retrained in this batch.** The published horizon
table used 16 epochs of LinearLR; this uses 300 of warmup+cosine, so those
numbers are not a valid baseline for these (rules 3 and 10).

Parameter parity is exact: Looped 207,457 = L1 207,457, against L4 802,273.

## INDEX (RoPE)

| config | 1-2 | 3-4 | 5-8 | 9-16 | 17-32 | 33-64 | 65+ | horizon |
|---|---|---|---|---|---|---|---|---|
| RoPE L1 (204K, depth 1) | 0.985 | 0.948 | 0.805 | 0.615 | 0.492 | 0.496 | 0.499 | **9-16** |
| RoPE L4 (802K, depth 4) | 1.000 | 1.000 | 0.999 | 0.995 | 0.878 | 0.523 | 0.508 | **17-32** |
| RoPE Looped x4 (204K, depth 4) | 1.000 | 1.000 | 0.999 | 0.992 | 0.855 | 0.512 | 0.481 | **17-32** |

## PATH-INTEGRATED (MapFormer-WM)

| config | 1-2 | 3-4 | 5-8 | 9-16 | 17-32 | 33-64 | 65+ | horizon |
|---|---|---|---|---|---|---|---|---|
| MapFormer L1 (204K, depth 1) | 0.960 | 0.959 | 0.959 | 0.962 | 0.959 | 0.951 | 0.945 | **65+** |
| MapFormer L4 (802K, depth 4) | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.999 | 0.997 | **65+** |
| MapFormer Looped x4 (204K, depth 4) | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.999 | 0.989 | **65+** |

## Verdict

**Q1 index arm.** horizon: L1 9-16, L4 17-32, Looped x4 17-32.

Recursion buys DEPTH'S HORIZON AT A QUARTER OF THE PARAMETERS. Effective depth, not layer specialisation, is what the depth grid was measuring.

**Q2 path-integrated arm**, mean over 33-64/65+: L1 0.948, L4 0.998, Looped x4 0.994.

Depth did NOT hurt at long range under this schedule and budget, which does not reproduce the earlier grid's non-monotonicity. That grid ran at 16 epochs of LinearLR, so the earlier finding was plausibly an optimisation artifact -- as its own caveat allowed. Q2 is answered by dissolving it.

## Scope

One task (torus, T=128), one width, one loop count (4), n=3. A shared block
with no depth embedding is the most conservative form of recursion; Universal
Transformer's per-iteration timestep embedding and recomputing theta each
iteration (iterative position refinement) are both untested here and are the
natural follow-ons IF this pilot is positive.
