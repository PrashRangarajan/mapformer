# Testing the accounts behind the sign result -- results (2026-10-15)

Pre-registration `ACCOUNTS_PREREG.md` + Amendment 1 (independent check `docs/audits/2026-10-14/accounts_audit.md`). GPU batch
2026-10-14 19:54 -> 10-15 01:39 at adf3c08 on both GPUs; T3 (CPU toy) alongside. Registered output `ACCOUNTS_ANALYSIS.txt`,
`ACCOUNTS_VERDICTS.json`; toy `runs/accounts/t3/summary.txt`.

| test | account tested | registered verdict | prediction |
|---|---|---|---|
| T2 | the signed escape from the |.| start is collective (one latent direction cancels every channel) | **SUPPORTED** | SUPPORTED |
| T1 | weight decay drives the drift off the wrapped code | D1 NO EFFECT DETECTED (and the drift itself is in unused channels) | drives the drift (wrong) |
| T4 | CoPE fails because of the ring's wrap (gate lattice) | L1 NO EFFECT DETECTED; L2 SIGNED LEARNS BETTER | line easier (wrong) |
| T3 | the toy fails because it rewards cell, not observation, retrieval | (toy) not the difference | signed escapes (wrong) |

## T2: the signed escape uses the latent route (ring, seeds 1000-1007)
| arm | start | move latents | acc | SOLVED | opposite-sign channels at the end |
|---|---|---|---|---|---|
| SO_same | its default angles (balanced gauge) | free | 1.000 | 8/8 | 1.00 |
| SO_same_fz | the same | frozen | 0.991 | 7/8 | 0.54 |
| SO_absmag | |default| (the |.| start) | free | 0.985 | 7/8 | 0.82 |
| SO_absmag_fz | the same | **frozen** | **0.910** | **1/8** | **0.15** |
- **F1: WORSE** -- freezing the two move latents blocks the escape from the |.| start: 1/8 vs 7/8 (Fisher p 0.010), -0.075
  [-0.130, -0.020] (perm p 0.009). **F2: NO EFFECT DETECTED** -- from the default start freezing costs nothing (7/8 vs 8/8).
  **Reading: SUPPORTED.** With W_out per channel and omega free, the per-channel route alone rarely crosses zero (15% of
  channels end opposite-signed); with the latents free the model moves one move's latent toward the negative of the other's
  and cancels the channels together (82%).

## T1: the "drift off the cancelling code" was in channels the model does not use (ring, seeds 600-607)
| arm | usage-weighted cancelling share: construction -> end | unweighted | acc (4T) | SOLVED |
|---|---|---|---|---|
| AO_bal (wd 0.05) | 0.99 -> **0.98** | 0.87 -> 0.18 | 1.000 (0.978) | 8/8 |
| AO_bal (wd 0) | 0.99 -> 0.98 | 0.87 -> 0.24 | 1.000 (0.985) | 8/8 |
| SO_bal (wd 0.05) | 0.99 -> 1.00 | 0.87 -> 0.98 | 1.000 (0.998) | 8/8 |
| SO_bal (wd 0) | 0.99 -> 0.97 | 0.87 -> 0.38 | 1.000 (0.989) | 8/8 |
- **D1: NO EFFECT DETECTED** (AO_bal wd 0 - wd 0.05: -0.003); **D2: NO EFFECT DETECTED** (-0.024, |d| < 0.10).
- **Weighted by use, the held non-negative model keeps 98% of its cancelling code.** WRAP_EXISTENCE's "87% -> 18%" counted
  every channel equally; the channels that leave the code are the ones attention does not read. The secondary claim "training
  moves the held model off the cancelling code" is withdrawn. Weight decay changes only the unused channels (unweighted
  shares move both ways: signed 0.98 with decay vs 0.38 without).
- Prop. 2 (rescaling sensitivity) is unaffected as mathematics; the drift it was offered to explain does not exist in the used
  channels.

## T4: CoPE fails on a line that never wraps too (seeds 1100-1115 for CoPE, 1100-1107 otherwise)
| arm | task | acc | SOLVED |
|---|---|---|---|
| CoPEF | line (256 cells, never wraps) | 0.914 (0.62-1.00) | 1/16 |
| CoPEF | ring (32 cells) | 0.873 (0.67-1.00) | 0/16 |
| Signed_r4 | line | 1.000 | 8/8 |
| RoPE (index) | line | 0.772 | 0/8 |
Predict-blank floor on the line 0.503.
- **L1: NO EFFECT DETECTED** (1/16 vs 0/16): removing the wrap does not make CoPE learnable. **L2: SIGNED LEARNS BETTER** on the
  line (8/8 vs 1/16, Fisher p < 1e-4; d -0.086, p 0.023). The index model fails the line (0.772), so the line needs a map.
- The diagnosis's prediction failed. The theory appendix's later correction anticipated part of it: on a line the gate
  difference still has to resolve D_max + 1 displacements, a margin of at most P / (D_max + 1) ~ 0.016 -- looser than the
  ring's lattice (~1e-3) but still fine. CoPE's obstacle is the precision its count needs, not the wrap specifically. Several
  CoPE seeds reach 0.99-1.00 accuracy on both tasks without meeting the loss criterion (bimodal, as in WRAP_COPE).

## T3 (toy, exploratory): the observation target is not the toy's difference
Signed toy from the |.| start: 0/8 MAP SOLVED with the observation target (predicted >= 5/8), as with the cell target (0/8);
opposite-sign channels 0.27 vs 0.23. The observation target raises accuracy (0.41 -> 0.66) but does not open the escape. The
toy and transformer remain different for an unidentified reason; T2 shows the transformer escapes through the latent route.

## What it means
- **The mechanism gains a tested component:** the signed model leaves a same-direction start through a collective latent
  rotation (z(-1) -> -z(+1)), not channel by channel; |.| has no such route (|W z(-1)| = |W z(+1)| would be a clock). With
  T1, the held non-negative solution stays on the cancelling code where it matters.
- **Two accounts are refuted and are dropped:** weight-decay drift (there is no drift in used channels) and "the wrap is
  CoPE's obstacle".

## Caveats
Ring only for T1 / T2; n = 8 (16 for CoPE); the usage weights are an eval-only proxy (query x key norms per channel).
