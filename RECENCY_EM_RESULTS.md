# EM vs WM on recency (k-back): two of three pre-registered predictions REFUTED

3 arms x 8 seeds, one batch, no `--fast-attn`. Pre-registration: `REC_EM_PREREG.md`.
Config as `RECENCY_RESULTS.md` (`k_max=64`, `p_filler=0.5`, `min_gap=64`, train
`T=1024`, 300 ep cosine, lr 1e-3, 1 layer, d=128). Chance **0.0625**, most-recent
shortcut floor **0.0771**.

| arm | final loss | T=1024 | T=2048 | worst seed (T=1024) |
|---|---|---|---|---|
| `Vanilla_r4` (WM) | 0.008-0.380 | **0.975 +/- 0.072** | 0.947 +/- 0.078 | 0.797 |
| `VanillaEM_r4` (paper-faithful EM) | 0.430-1.136 | 0.837 +/- 0.081 | 0.728 +/- 0.106 | 0.689 |
| `VanillaEM_P0_r4` (single `p_0` EM) | 0.924-1.885 | **0.600 +/- 0.126** | 0.510 +/- 0.111 | 0.365 |

| contrast | delta | sd | MDE | seeds + | verdict |
|---|---|---|---|---|---|
| `EM_P0_r4 - Vanilla_r4` | **-0.375** | 0.156 | 0.154 | **0/8** | **DETECTABLE** |
| `EM_r4 - Vanilla_r4` | **-0.137** | 0.108 | 0.106 | 1/8 | **DETECTABLE** |
| `EM_P0_r4 - EM_r4` | **-0.237** | 0.154 | 0.152 | **0/8** | **DETECTABLE** |

## P1 CONFIRMED, and more strongly than registered

[11] names N-back as its one exception to "EM learns faster". Registered: EM does
not reach threshold sooner than WM. Measured: EM does not learn the task at all --
final loss 0.92-1.89 against WM's 0.008-0.044, and 0/8 seeds positive.

## P2 REFUTED

Registered: `|EM - WM|` below the 0.050 MDE, on the argument that MapEM has no
separate memory network and should simply match WM. Measured **-0.375**, seven
times the MDE. **This is the first task in the project where EM and WM differ at
all** -- the four previous cells were +0.000 to +0.004. The structural argument in
`TALE_OF_TWO_ALGORITHMS.md` predicted a null and got the largest EM-vs-WM effect
we have measured. It was incomplete, not wrong: see `THEORY_KERNEL.md` Sec 5.

## P3 REFUTED, SIGN INVERTED -- and this is the informative one

Registered: single `p_0` beats separate `q_0^p`/`k_0^p`, "a LARGE gap here" because
`A_P` carries the whole task. Measured **-0.237, 0/8 seeds**: the paper-faithful
separate form is BETTER, decisively.

Five tasks now:

| task | kind | `EM_P0 - EM_sep` |
|---|---|---|
| paper task (torus) | map | +0.089 |
| compositional | map | +0.167 |
| Match-Query | map | +0.358 |
| MiniGrid allocentric | map | collapse removed (gap 0.137 -> 0.002) |
| **recency (k-back)** | **clock** | **-0.237** |

The sign tracks the task, not the architecture. `THEORY_KERNEL.md` Sec 4 gives the
account: a single `p_0` forces the position kernel's coherence to `rho = 1`, i.e. a
matched filter peaked exactly at zero displacement. That is what a map task wants
and precisely what a k-back task does not.

**Stated plainly: the theory in `THEORY_KERNEL.md` was written AFTER this batch
landed and retrodicts this row. It did not predict it -- my pre-registration
predicted the opposite sign.** Sec 8 of that document lists what it predicts that
has not yet been measured.

## The per-offset curve says where EM fails

Mean over 8 seeds, T=1024:

| arm | k=1 | k=2 | k=4 | k=8 | k=16 | k=32 | k=48 | k=64 |
|---|---|---|---|---|---|---|---|---|
| `Vanilla_r4` | 0.99 | 0.98 | 0.98 | 1.00 | 1.00 | 0.98 | 0.98 | 0.97 |
| `VanillaEM_r4` | 1.00 | 0.98 | 1.00 | 0.79 | 0.85 | 0.77 | 0.71 | 0.73 |
| `VanillaEM_P0_r4` | 0.98 | 1.00 | 0.89 | 0.76 | 0.60 | 0.79 | 0.72 | **0.37** |

Both EM arms are at WM's level for `k <= 2` and fall away with `k`; the coherent
arm falls furthest, to 0.37 at `k=64`. WM is flat. So EM is not globally weaker --
it is weaker *exactly where the required retrieval offset is far from zero*, which
is the mechanism claim.

## Caveat

`Vanilla_r4` here is 0.975 +/- 0.072 where the published `Signed_r4` row is
1.000 +/- 0.000; seed 7 (final loss 0.380) accounts for it. The published arm used
`--fast-attn` and a different construction path. Every contrast above is
within-batch, so this does not touch them, but the WM arm is not at ceiling here
and the published row should not be mixed with this table.
