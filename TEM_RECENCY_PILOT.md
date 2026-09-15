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
