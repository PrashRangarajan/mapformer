# TEM on recency (k-back): pilot

Written 2026-09-14, before training. A pilot, not a registered result.

**Question.** Does TEM, whose memory is queried by a structural code alone, show MapEM's recency
deficit?

**Why the answer matters for MapEM.** With g in rotation blocks, TEM's retrieval kernel equals
single-origin MapEM's shared kernel. The difference is that TEM's per-query transform is a full
orthogonal matrix per token, not MapFormer's rank-4 bottleneck scaled by frequency.
- If TEM learns recency easily, EM's deficit is likely the bottleneck parameterisation.
- If TEM struggles too, it is likely the shared-kernel design (offsets must be written into position,
  not selected by content).

**Model:** `model_tem_recency.py`. Its adaptations are listed in the module docstring.

| arm | seeds | what it is |
|---|---|---|
| `TEMRecency` | 0, 1 | faithful: the query is the state; a query token's transition also moves the state |
| `TEMRecency_Query` | 0, 1 | a separate per-token query transform (non-committing lookup) |
| `TEMRecency_Query_Installed` | 0 | existence check: exact rewind installed and frozen; only content, binding and decoder train |

Construction check before training: with the installed transforms, the raw structural kernel ranks
the correct k-back symbol first among earlier symbols on 58/58 test queries.

**Recipe.** Identical to the MONOTONE and PAIRORIGIN recency batches: k_max 64, T 1024, evaluation at
T 1024 and 2048, 300 epochs x 48 x 16, cosine, lr 1e-3.

**References, not same-batch:** `MONOTONE_RESULTS.md` has the same recipe and seeds, with
VanillaEM_P0_r4 0.641 +/- 0.136 and Signed_r4 (MapWM) 0.994 +/- 0.020 at T=1024. A pre-registered
same-batch comparison follows only if the pilot shows TEM in range.

**Parameter counts** are not matched: TEMRecency 376,307, TEMRecency_Query 740,851, MapEM 222,361.
TEM's extra parameters are all in the per-token transition matrices. A TEM deficit therefore cannot
be blamed on capacity; a TEM success could partly be capacity.

## Results

Same recipe as `MONOTONE_RESULTS.md` (chance 0.0625; most-recent floor 0.0771; ln 16 = 2.77).

| arm | seed | acc T=1024 | acc T=2048 | final loss | first epoch with loss < 0.5 |
|---|---|---|---|---|---|
| `TEMRecency_Query_Installed` (rewind installed, frozen) | 0 | **1.000** | **1.000** | 0.000 | 8 |
| `TEMRecency_Query` | 0 | 0.118 | 0.104 | 2.695 | never |
| `TEMRecency_Query` | 1 | 0.119 | 0.099 | 2.700 | never |
| `TEMRecency` (faithful, committing) | 0 | 0.094 | 0.091 | 2.729 | never |
| `TEMRecency` | 1 | 0.099 | 0.107 | 2.735 | never |
| *reference, other batch:* VanillaEM_P0_r4 / Signed_r4 (MapWM) | 0-11 | 0.641 / 0.994 | 0.543 / 0.898 | 1.262 / 0.032 | -- |

## Reading (pilot; not a registered result)

- **The exact solution is representable and trivially holdable in TEM's form.**
  - With the rewind installed and frozen, TEM scores 1.000 at both lengths.
  - The content-side weights learn to use it within 8 epochs.
- **From scratch, TEM learns essentially nothing.**
  - All four runs sit barely above chance (0.094-0.119).
  - Training loss stays at 2.70-2.74 against 2.77 for uniform guessing.
  - This is worse than MapEM (0.641), not comparable to it.
- **This does NOT answer the bottleneck question.** TEM failing more completely than MapEM does not
  show that a full per-query transform is harder to learn than a rank-4 one. From scratch, TEM must
  learn the counter itself (a transition per token id) as well as the rewind. It never gets below
  chance loss, which looks like a failure to start learning, not a failure at the rewind.
- **The likely cause is untested.** The transitions start near the identity, so every structural code
  is nearly the same, retrieval is uniform, and there may be little gradient to break the symmetry.
  The installed arm rules out the readout and binding; it does not test the start.

## Diagnostics before any registered TEM comparison

1. **k_max = 1 or a fixed small k.** If TEM cannot learn "the most recent symbol" either, the problem
   is the adaptation's optimisation, not recency.
2. **Install only the counter** (symbol transition and filler identity), leave the query transforms
   trainable. If TEM then learns the rewinds, the counter was the bottleneck. If it does not, the
   per-query rewind search is.
3. **A larger initialisation scale** for the transitions, which breaks the near-identity symmetry.
