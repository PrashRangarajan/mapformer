# MONOTONE results: the sign ablation on MapFormer-EM and on Selective RoPE's generator

Pre-registration: `MONOTONE_PREREG.md` (committed 92d6eaa before training). Mechanical report with
every table: `MONOTONE_RAW.md` / `MONOTONE_RAW.json` (`analyze_monotone.py`). 96 runs, one batch,
12 seeds per arm. Registered verdicts first, exactly as the rules fire; exploratory analyses after,
labelled.

## Manipulation checks: all pass

- **M-C1**: every constrained checkpoint's per-channel increments are >= 0 on all 12 seeds (min
  +1.1e-09); every signed checkpoint's go negative (max over seeds -0.86 or lower).
- **M-C2**: at init the signed twin is bitwise `VanillaEM_P0_r4`; both constrained arms share every
  parameter and RNG draw with their parents.
- **M-C3**: `Signed_r4` and `Abs_r4` seed 0 retrained here are **bitwise identical** to
  `runs/sign/p0` (24/24 tensors, loss curves equal). Determinism, not replication; it licenses the
  stored sign-batch arms as same-pipeline.

## Experiment 1: does MapFormer-EM need subtraction on recency?

| arm | acc T=1024 | acc T=2048 | final loss |
|---|---|---|---|
| `VanillaEM_P0_r4` (EM, signed) | 0.641 +/- 0.136 | 0.543 +/- 0.114 | 1.262 |
| `EM_P0_Abs_r4` (EM, monotone) | 0.443 +/- 0.117 | 0.375 +/- 0.089 | 1.753 |
| `Signed_r4` (WM, signed) | 0.994 +/- 0.020 | 0.898 +/- 0.058 | 0.032 |
| `Abs_r4` (WM, monotone) | 0.934 +/- 0.098 | 0.883 +/- 0.108 | 0.156 |

| contrast | delta | sd | MDE | seeds + | verdict |
|---|---|---|---|---|---|
| **P1** EM: Abs - signed, T=1024 | **-0.198** | 0.192 | 0.155 | 2/12 | DETECTABLE |
| EM: Abs - signed, T=2048 | -0.169 | 0.120 | 0.097 | 1/12 | DETECTABLE |
| **P2** WM: Abs - signed, T=1024 | -0.060 | 0.104 | 0.084 | 1/12 | unmeasured |
| **P3** interaction P1 - P2 | -0.138 | se 0.065 | 0.181 | -- | unmeasured |
| P1 loss-matched (reported, not read) | -0.043 | 0.063 | 0.051 | 3/12 | unmeasured |

Rule 9 over the 72 recency runs: r(final loss, acc) = -0.987.

**P4, route of solved cells (acc >= 0.9, k >= 8):**

| arm | seeds with solved cells | solved cells / seed | rewind-route fraction |
|---|---|---|---|
| EM signed | 12/12 | 23.9 | 0.953 |
| EM monotone | 11/12 | 10.4 | 0.936 |

**Registered verdict: UNRESOLVED.**
- SUBTRACTION NEEDED required P3 to be detectable, and it is not (MDE 0.181).
- WRAP SUBSTITUTES required P1 not to be detectably negative, and it is.

**Reading.** This is not a verdict; it describes the numbers above.
- **Cost.** Removing the sign costs EM 0.198 on recency, on 10 of 12 seeds. For WM the same
  constraint is not measurable at T=1024 (-0.060).
- **Route.** Monotone EM that solves a cell still solves it through the query token's own step, in
  0.936 of solved cells. Its increments are >= 0, so that step is a forward, wrapped one. The
  rewind route does not require subtraction.
- **Count.** What the constraint changes is how many tokens find that step: 10.4 against 23.9
  solved cells per seed (exploratory contrast below, -13.5, MDE 11.5).
- **Fit to the search account.** A constraint that leaves the solution expressible and the route
  intact, but shrinks how often training finds it, is what `SEARCH_RESULTS.md` would predict.
  Nothing here shows a representational limit.
- **Caveat.** Rule 9 is -0.987, so the raw P1 is also a loss gap. At matched loss P1 is -0.043,
  which is unmeasured.

## Experiment 2: does the sign account transfer to Selective RoPE's generator?

| arm | torus T=128 | T=512 | T=1024 | final loss |
|---|---|---|---|---|
| `SRoPEGen` | 1.000 +/- 0.000 | 0.984 +/- 0.008 | 0.899 +/- 0.051 | 0.0003 |
| `SRoPEGen_Abs` | 0.925 +/- 0.028 | 0.638 +/- 0.029 | 0.545 +/- 0.029 | 0.3148 |

| arm | recency T=1024 | T=2048 | final loss |
|---|---|---|---|
| `SRoPEGen` | 0.944 +/- 0.064 | 0.859 +/- 0.071 | 0.181 |
| `SRoPEGen_Abs` | 0.877 +/- 0.068 | 0.806 +/- 0.050 | 0.409 |

| contrast | delta | sd | MDE | seeds + | verdict |
|---|---|---|---|---|---|
| torus Abs - signed, T=1024, raw | -0.355 | 0.057 | 0.046 | 0/12 | DETECTABLE |
| **Q1** torus Abs - signed, T=1024, loss-matched (2-arm pool) | -0.056 | 0.102 | 0.083 | 5/12 | unmeasured |
| **Q2** recency Abs - signed, T=1024 | **-0.068** | 0.067 | 0.054 | 2/12 | DETECTABLE |
| **Q3** crossover torus(raw) - recency | **-0.287** | se 0.025 | 0.071 | -- | DETECTABLE |

**Registered verdicts:**
- **Q1: FALSIFIED as registered.**
- **Q2: FALSIFIED.** On recency the monotone constraint costs SRoPE a small, detectable amount.
- **Q3: MET.**

**Q1's registered form could not detect the effect it was testing.** The design error is mine.
- **Why the fit absorbs the effect.** The pre-registration loss-matched over only the two SRoPE
  arms. Their final losses do not overlap: 0.0002-0.0006 against 0.20-0.47. With two clusters and
  essentially no within-arm spread in one of them, the acc ~ loss fit is identified by the
  between-arm gap itself, and the residual removes that gap by construction.
- **What `SIGN_ABLATION.md` did instead.** Its -0.280 pooled six arms whose losses span the range,
  RoPE and the partially monotone arms among them.
- **The general point.** A loss-matched contrast needs a pool in which loss varies independently
  of the manipulation. When the manipulation itself causes the loss gap, it measures nothing.

## Exploratory (not registered)

| contrast | delta | sd | MDE | seeds + | verdict |
|---|---|---|---|---|---|
| SRoPE torus Abs - signed, T=1024, loss-matched, **8-arm pool** (6 stored sign arms + 2 SRoPE) | -0.188 | 0.072 | 0.059 | 0/12 | DETECTABLE |
| MapWM torus Abs - signed, T=1024, loss-matched, same 8-arm pool | -0.273 | 0.083 | 0.067 | 0/12 | DETECTABLE |
| SRoPE torus Abs - signed, T=512, loss-matched, 8-arm pool | -0.172 | 0.047 | 0.038 | 0/12 | DETECTABLE |
| MapWM torus raw Abs - signed, T=1024 (stored) | -0.363 | 0.030 | 0.025 | 0/12 | DETECTABLE |
| SRoPE sign cost - MapWM sign cost, torus raw T=1024 (unpaired) | +0.009 | se 0.019 | 0.052 | -- | unmeasured |
| WM recency Abs - signed, T=2048 | -0.015 | 0.120 | 0.097 | 7/12 | unmeasured |
| SRoPE recency Abs - signed, T=2048 | -0.052 | 0.094 | 0.076 | 3/12 | unmeasured |
| EM solved cells (k >= 8) per seed, Abs - signed | -13.5 | 14.2 | 11.5 | 3/12 | DETECTABLE |

The 8-arm pool is licensed only by M-C3's bitwise match and remains post hoc. Pool r(loss, acc) is
-0.742 at T=1024 and -0.853 at T=512.

Per-seed caveat on P4: the route fraction for monotone EM averages over very few solved cells on
some seeds (seed 7: 1 cell, seed 8: 3, seed 10: none).

## What this changes

- **The sign cost on navigation is not specific to MapFormer's generator.** Selective RoPE's
  generator, which has a gate, a conv and no rank bottleneck, pays the same raw cost when its
  increment is forced non-negative: -0.355 against MapWM's -0.363, difference unmeasured with an
  MDE of 0.052. At matched loss in the 8-arm pool it pays -0.188, which is exploratory.
- **"Monotone is free on a counting task" must be softened.** On recency, monotone increments cost
  SRoPE -0.068 (detectable) and WM -0.060 (unmeasured). The torus cost is about five times larger
  (crossover -0.287, detectable). The defensible form is: **large on a map task, small on a
  counting task**, not "decisive vs free".
- **MapFormer-EM does not need a negative increment to rewind.** Monotone EM keeps the rewind route
  through a forward, wrapped step. Its detectable cost comes with finding that step for fewer
  tokens, which fits the search account. Whether EM is hurt more than WM (P3) is unmeasured.
