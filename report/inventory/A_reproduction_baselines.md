# Inventory A -- Reproduction, paper task and baselines

## Overview

This line asks three questions. (1) Does our reimplementation reproduce MapFormer's own task,
Table 2 (1D-2D grid navigation, 2D columns), including its OOD-d / OOD-s protocol? (2) On that
task and a few others, what does the paper's model set look like against non-MapFormer
baselines: index-position transformers (RoPE, PlainFlat, PoPE-Flat), LSTM, CoPE, a Mamba-style
SSM, two TEM formulations, and the paper's own proposed extensions (separate q0/k0,
MapEM-NC on a family tree)? (3) What do the EM-vs-WM comparisons the paper frames (Table 2,
Fig. 4c vocab scaling) show once they are re-measured?

What survives:
- The reproduction is consistent with paper **v1** (WM 0.99/0.99/0.96, EM-os 1.0/0.99/0.97)
  within about one sd. It sits systematically **below v4** (1.00 throughout). WM's IID shortfall
  does not move with budget (0.969 at 16 ep LinearLR, 0.968 at 50 ep cosine).
- On the paper task, **path integration, not the encoding, is the axis**. At n=8 in one batch
  the position main effect is +0.461 and the encoding main effect +0.003, which is unmeasured
  (MDE 0.029). Index models sit on the measured blank floor. Their small excess is out-and-back
  retraces at recurrence interval 1-2.
- The paper-task maps are **used**: destroying the action stream drops accuracy below the
  floor, with worse-than-uniform NLL.
- At extended length on the paper's OOD-s condition, **EM degrades more slowly than WM**.
  Floor-normalised, EM - WM is +0.186 at l=1024 and +0.287 at l=2048, 8/8 each, in a
  pre-registered 50-epoch rerun whose convergence gate failed. **MapPoPE-Flat beats MapWM by
  +0.116 at l=2048.** Its lead over EM is unmeasured.
- Two paper conjectures fail on map tasks. (a) Separate q0/k0 is worse than single p0 on
  four map tasks (n=3; the sign reverses on recency). (b) Non-commutativity buys +0.005 to
  +0.014 on a family tree whose non-commutativity is 1.000, and costs 14.4x length scaling
  in training time.
- The **parallel-scan cost claim** holds on wall clock: fwd+bwd growth over L=128->2048 is
  2.5x/3.9x/2.7x for the parallel models, against 14.4x for MapEM-NC and 120.4x for
  TEMFaithful.
- On the April clean batch, the Mamba-style SSM, CoPE, LSTM and RoPE all trail MapFormer on
  fresh maps (n=3, exploratory).
- The paper's quadratic-capacity claim for EM gets **no support** from a healthy-EM vocab
  sweep: every contrast is unmeasured at n=8, and the only gain is one collapsed seed.
- Most TEM comparisons (cross-scale, cross-topology, multi-env, long-T, sparse landmarks,
  TEM scaling) were run with lm200 landmarks and are void. Surviving TEM numbers are
  single-env clean/noise at n=3.

---

### A1 Paper-task reproduction, first pass (n=3)
- Dates: first commit 2026-08-09 (PAPER_VALIDATION.md); PAPER_TASK_ACCURACY.md last touched
  2026-09-11 (a correction block only); EM_P0_PAPER.md 2026-08-09.
- Question: Does the reimplementation reach the paper's Table 2 IID accuracy? Which EM
  parameterisation does?
- Task / environment: torus 64x64, 16 obs types, p_empty 0.5, 1 layer, 2 heads, d=128, T=128,
  batch 128, 200K sequences (16 epochs x 98 batches). Held-out revisit accuracy is measured on
  the same map and on a fresh map. Floor: the always-blank rate of 0.506, measured later in
  PAPER_TASK_ABLATION.md.
- Arms: `Vanilla` (MapWM), `VanillaEM` (paper-faithful separate q0/k0), `VanillaEM_P0`
  (single p0, `MapFormerEM_SingleP0` in model_em_fixed.py), `VanillaEM_Fixed` (paper eq.13,
  Hadamard on probabilities).
- Seeds / batch: n=3; LinearLR default recipe.
- Validity gates: none beyond same-map == fresh-map. That equality ("to 3 decimals for every
  variant") shows the map is built in context rather than memorised.
- Result (PAPER_VALIDATION.md, PAPER_TASK_ACCURACY.md):

  | variant | same-map | fresh-map |
  |---|---|---|
  | Vanilla (WM) | 0.989 +/- 0.010 | 0.989 +/- 0.010 |
  | VanillaEM (separate q0/k0) | 0.898 +/- 0.108 | 0.901 +/- 0.102 |
  | VanillaEM_P0 (single p0) | 0.987 +/- 0.012 | 0.987 +/- 0.012 |

  - Separate-q0/k0 per-seed values: 0.778 / 0.931 / 0.986. VanillaEM_Fixed final losses were
    1.1963 / 1.2021 / 1.1840, so it did not train. EM_P0 final losses: 0.1053 / 0.1508 /
    0.1014 (EM_P0_PAPER.md).
- Status: EXPLORATORY (n=3). The WM and EM_P0 values are superseded by A3/A4/A5 at n=8. The
  separate-q0/k0 row exists only at n=3.
- Pre-registered? no.
- Caveats the report must carry:
  - PAPER_VALIDATION originally cited a 0.955/0.999 target that appears in no paper table.
    It was corrected in place.
  - Paper version: our numbers match v1. v4 reports 1.00 throughout (CLAUDE.md).
  - Our EM is the paper's MapEM-os.
  - Separate q0/k0 is paper-faithful; single p0 is an ablation of the paper's stated
    suspicion. PAPER_VALIDATION cites App. A.4; CLAUDE.md corrects this to App. A.7 (line 1527).
- Sources: PAPER_VALIDATION.md, PAPER_TASK_ACCURACY.md, EM_P0_PAPER.md.
- Bears on: reproduction; EM parameterisation (q0/k0); the where/what factorisation (in-context
  map, same-map == fresh-map).

### A2 Paper's own OOD protocol, first pass (n=3; superseded)
- Dates: PAPER_OOD_PROTOCOL.md 2026-08-09; PAPER_OOD_WITH_POPE.md 2026-08-18;
  PAPER_OOD_EXTENDED.md 2026-08-19.
- Question: Do the paper's OOD-d and OOD-s numbers reproduce? Does MapPoPE-Flat hold there?
- Task / environment: the paper's App. B protocol:
  - IID: l=128, g=64, pe=0.5.
  - OOD-d: l=64, g=32, pe=0.2.
  - OOD-s: pe=0.8, g=128, at l=512 (App. B) and at l=256 (the Table 2 caption; the paper is
    internally inconsistent). Both lengths are reported.
  - Extended: ext-s at l=1024 and l=2048.
  - The blank floor tracks p_empty.
- Arms: Vanilla, VanillaEM, VanillaEM_P0, MapPoPE-Flat, RoPE, PlainFlat, PoPE-Flat.
- Seeds / batch: n=3, 16-epoch paper recipe.
- Validity gates: the floor reading. Every index model tracks the blank rate in every
  condition: RoPE 0.513 / 0.271 / 0.803 / 0.802.
- Result: superseded by A4 (n=8) and A5. The one n=3 observation that matters is that the
  paired "EM >= WM" ordering did not resolve: EM won exactly 1 of 3 seeds in every condition
  (PAPER_OOD_PROTOCOL.md).
- Status: EXPLORATORY. Superseded.
- Pre-registered? no.
- Caveats the report must carry:
  - The n=3 "non-overlapping +/-1sd ranges at l=2048" claim was withdrawn at n=8
    (PAPER_OOD_EXTENDED_n8.md).
  - "Reproduces exactly" (WM IID 0.988) was lucky seeds.
- Sources: PAPER_OOD_PROTOCOL.md, PAPER_OOD_WITH_POPE.md, PAPER_OOD_EXTENDED.md.
- Bears on: reproduction; floors.

### A3 Paper task: position x encoding 2x2, n=8, one batch
- Dates: INDEX_BASELINE_PAPER_TASK.md 2026-08-17 (n=3); INDEX_BASELINE_PAPER_TASK_n8.md
  2026-08-20; BASELINE_TABLE.md section A/headline (last edit 2026-09-07, corrected
  2026-09-06); AUDIT_HEADLINE.md 2026-08-23.
- Question: On the paper's own task, is the axis where the angle comes from (index vs
  path-integrated) or how it is applied (RoPE vs PoPE)?
- Task / environment: as A1. Fresh obs_map, T=128. Measured always-blank floor 0.506
  (PAPER_TASK_ABLATION.md).
- Arms:

  | arm | encoding | position |
  |---|---|---|
  | `Vanilla` = MapWM-Flat | RoPE | path-integrated |
  | `MapPoPE-Flat` | PoPE | path-integrated |
  | `RoPE` | index RoPE, MapFormer architecture | index |
  | `PlainFlat` | index RoPE | index |
  | `PoPE-Flat` | PoPE | index |
  | `VanillaEM_P0` | -- | path-integrated |

  Parameters are matched within 0.4% (~204k).
- Seeds / batch: n=8, one batch (`runs/paper_task_n8`), 16 epochs on the paper recipe
  (LinearLR). The batch directory and its 48 training logs have since been deleted
  (PAPERTASK_PREREG.md), so rule 9 cannot be run on it. Its numbers reproduce from committed
  per-seed JSON.
- Validity gates:
  - AUDIT_HEADLINE.md re-derives all five fresh-map A values from per-seed JSON, 5/5 PASS
    (e.g. Vanilla 0.9670 vs 0.9674).
  - Context destruction: A6.
  - Residual explanation: A7.
- Result (INDEX_BASELINE_PAPER_TASK_n8.md; BASELINE_TABLE.md A):

  | model | fresh-map acc (n=8) |
  |---|---|
  | MapPoPE-Flat | 0.994 +/- 0.017 |
  | MapEM-os (VanillaEM_P0) | 0.987 +/- 0.009 |
  | MapWM-Flat (Vanilla) | 0.967 +/- 0.039 |
  | Plain-Flat | 0.534 +/- 0.040 |
  | RoPE | 0.530 +/- 0.043 |
  | PoPE-Flat | 0.509 +/- 0.001 |

  - Paired n=8 main effects (BASELINE_TABLE.md headline): **position +0.461**; encoding
    +0.003, unmeasured (MDE 0.029, 5/8 seeds). No sd or MDE is printed for the position
    effect.
  - At n=3 (INDEX_BASELINE_PAPER_TASK.md) the 2x2 cells were: RoPE/index 0.514 +/- 0.004,
    RoPE/PI 0.989 +/- 0.011, PoPE/index 0.509 +/- 0.004, PoPE/PI 1.000 +/- 0.001.
- Status: CITABLE for the position effect. The encoding effect is unmeasured.
- Pre-registered? no.
- Caveats the report must carry:
  - Every MapPoPE number is at `bottleneck_r=2` (RESULTS_INDEX.md caveat). MapPoPE at r=4 is
    another line and gives +0.019, unmeasured.
  - The ~0.5 floor makes 1.000 a partial ceiling.
  - RESULTS_INDEX.md's headline table still quotes the n=3 cells. The n=8 values above
    supersede them.
  - The BASELINE_TABLE encoding cell once read +0.011 with a "40x" ratio. That was the n=3
    path-integrated row, not the main effect; corrected 2026-09-06.
- Sources: INDEX_BASELINE_PAPER_TASK.md, INDEX_BASELINE_PAPER_TASK_n8.md, BASELINE_TABLE.md,
  AUDIT_HEADLINE.md, N3_AUDIT.md (base-rate table).
- Bears on: where/what (the angle source is the load-bearing axis); the environment line
  (torus side of the torus-vs-MiniGrid contrast); PoPE line.

### A4 Paper OOD protocol plus length extension, n=8, 16 epochs
- Dates: PAPER_OOD_EXTENDED_n8.md 2026-08-20; BASELINE_TABLE.md section B.
- Question: At n=8, does the reproduction hold under OOD-d / OOD-s? Does the ordering open up
  when length is extended at the OOD-s condition?
- Task / environment: as A2. Same checkpoints as A3. The floors were measured later in
  PAPER_TASK_FLOORS.md (see A5).
- Arms: Vanilla (MapWM), VanillaEM_P0 (MapEM-os), MapPoPE-Flat.
- Seeds / batch: n=8, one batch, all retrained fresh (no n=3 checkpoints reused). 16 epochs
  LinearLR.
- Validity gates: floors (A5). Rule 9 is impossible because the logs were deleted.
- Result (PAPER_OOD_EXTENDED_n8.md):

  | variant | IID | OOD-d | OOD-s l=256 | OOD-s l=512 | ext-s l=1024 | ext-s l=2048 |
  |---|---|---|---|---|---|---|
  | Vanilla | 0.969 +/- 0.037 | 0.958 +/- 0.048 | 0.978 +/- 0.018 | 0.943 +/- 0.033 | 0.893 +/- 0.034 | 0.854 +/- 0.026 |
  | VanillaEM_P0 | 0.987 +/- 0.009 | 0.983 +/- 0.010 | 0.988 +/- 0.007 | 0.978 +/- 0.011 | 0.963 +/- 0.014 | 0.939 +/- 0.020 |
  | MapPoPE-Flat | 0.994 +/- 0.015 | 0.991 +/- 0.010 | 0.995 +/- 0.011 | 0.992 +/- 0.015 | 0.985 +/- 0.019 | 0.970 +/- 0.028 |

  - MapPoPE-Flat vs MapWM at l=2048: **+0.116**, se 0.0135, t 8.59.
  - MapPoPE-Flat vs MapEM-os at l=2048: +0.031, se 0.0122, t 2.55 (p ~0.02), and the +/-1sd
    bands overlap.
  - EM_P0 - WM, raw, paired, same batch (EM_WM_THEORY.md 2a; EM_WM_STATE.md): +0.035
    (l=512, 7/8), +0.070 (l=1024, 8/8), +0.085 (l=2048, 8/8), MDE 0.031-0.034. It survives
    IID-matching (+0.0856).
  - Reproduction against the paper:

    | | paper | n=3 | n=8 |
    |---|---|---|---|
    | MapWM IID | 0.99 | 0.988 +/- 0.011 | 0.969 +/- 0.037 |
    | MapWM OOD-s l=512 | 0.96 | 0.962 +/- 0.026 | 0.943 +/- 0.033 |
    | MapEM-os IID | 1.0 | 0.989 +/- 0.010 | 0.987 +/- 0.009 |
    | MapEM-os OOD-s l=512 | 0.97 | 0.976 +/- 0.010 | 0.978 +/- 0.011 |

- Status: CITABLE (MapPoPE vs WM; EM vs WM at l=1024/2048). DIRECTIONAL (MapPoPE vs EM: by the
  repo's MDE standard, t 2.55 is below 2.8). Reproduction: consistent with v1 within one sd.
- Pre-registered? no.
- Caveats the report must carry:
  - The paper columns are **v1**. v4 lists 1.00 for MapWM-r2, MapEM-os and MapEM-s in the
    IID and OOD-d columns, and 0.99/1.00/1.00 for OOD-s. Our WM is ~2-3pp below v4.
  - The ext-s columns have no published counterpart.
  - 16-epoch LinearLR recipe; logs deleted (see A5 for the rerun).
  - Floors are high at pe=0.8 (A5).
  - The MapPoPE r=2 caveat applies.
- Sources: PAPER_OOD_EXTENDED_n8.md, BASELINE_TABLE.md, EM_WM_THEORY.md, EM_WM_STATE.md.
- Bears on: reproduction; EM vs WM; PoPE; "helps at OOD length" as a universal unexplained
  signature.

### A5 PAPERTASK rerun: does EM's length advantage survive a converged recipe? (pre-registered)
- Dates: PAPER_TASK_FLOORS.md 2026-09-12; PAPERTASK_PREREG.md 2026-09-12 (commit 415e96e);
  PAPERTASK_RESULTS.md 2026-09-12; PAPER_OOD_RERUN.md 2026-09-12.
- Question:
  - Is the extended-length EM > WM gap of A4 real, or an artefact of the 16-epoch LinearLR
    recipe?
  - Is MapPoPE - EM detectable?
- Task / environment: as A4. The primary readout is floor-normalised accuracy,
  (acc - floor) / (1 - floor).

  Floors are measured with no model on exactly the scored events (PAPER_TASK_FLOORS.md, fresh
  map, env seed 10000, 8x32):

  | condition | always-blank floor | scored n |
  |---|---|---|
  | IID l=128 | 0.522 | 7,342 |
  | OOD-d | 0.216 | 3,212 |
  | OOD-s l=256 | 0.803 | 16,663 |
  | OOD-s l=512 | 0.799 | 34,793 |
  | ext-s l=1024 | 0.801 | 74,919 |
  | ext-s l=2048 | 0.802 | 162,768 |

  The best constant equals the blank floor in each condition; 1/vocab is 0.0476.
- Arms: `Vanilla` (WM), `VanillaEM_P0`, `MapPoPE-Flat`.
- Seeds / batch: n=8 each, one batch, trained fresh into `runs/paper_task_rerun/`. 50 epochs,
  cosine (5% warmup, decay to 10%), 98 batches x 128, 1 layer / 2 heads / d=128. Logs kept.
- Validity gates:
  - Pre-registered convergence gate (every arm IID >= 0.99): **FAILED**. WM reached 0.968 and
    EM 0.985; MapPoPE passed at 1.000.
  - Rule 9: r(final loss, acc) = **-0.461** over 24 runs (acc = 0.942 - 0.152*loss, resid sd
    0.036), so the contrast is not a loss gap.
  - Mean final loss: WM 0.1306, EM 0.0762, MapPoPE 0.0168.
- Result:
  - Raw accuracy (PAPER_OOD_RERUN.md):

    | variant | IID | OOD-d | OOD-s l=256 | OOD-s l=512 | ext l=1024 | ext l=2048 |
    |---|---|---|---|---|---|---|
    | Vanilla | 0.968 +/- 0.051 | 0.943 +/- 0.076 | 0.984 +/- 0.020 | 0.964 +/- 0.029 | 0.927 +/- 0.034 | 0.886 +/- 0.035 |
    | VanillaEM_P0 | 0.985 +/- 0.023 | 0.980 +/- 0.031 | 0.988 +/- 0.012 | 0.978 +/- 0.015 | 0.964 +/- 0.016 | 0.942 +/- 0.024 |
    | MapPoPE-Flat | 1.000 +/- 0.001 | 0.993 +/- 0.004 | 0.998 +/- 0.002 | 0.991 +/- 0.003 | 0.978 +/- 0.004 | 0.963 +/- 0.005 |

  - Floor-normalised, paired (PAPERTASK_RESULTS.md):

    | contrast | delta | MDE | seeds + | verdict | at 16 ep |
    |---|---|---|---|---|---|
    | EM - WM, l=512 | +0.067 | 0.168 | 4/8 | unmeasured | +0.174 |
    | EM - WM, l=1024 | **+0.186** | 0.185 | 8/8 | DETECTABLE | +0.352 |
    | EM - WM, l=2048 | **+0.287** | 0.194 | 8/8 | DETECTABLE | +0.430 |
    | MapPoPE - EM, l=512 | +0.066 | 0.079 | 6/8 | unmeasured | +0.070 |
    | MapPoPE - EM, l=1024 | +0.073 | 0.090 | 6/8 | unmeasured | +0.112 |
    | MapPoPE - EM, l=2048 | +0.102 | 0.128 | 6/8 | unmeasured | +0.159 |

  - Floor-normalised means from the 16-epoch batch (EM_WM_THEORY.md 2a):

    | condition | WM | EM | MapPoPE |
    |---|---|---|---|
    | OOD-s l=512 | 0.715 | 0.889 | 0.959 |
    | l=1024 | 0.460 | 0.812 | 0.924 |
    | l=2048 | 0.261 | 0.692 | 0.851 |

- Status: CITABLE, with the gate caveat. The EM - WM gap at l=1024/2048 is detectable in two
  independent n=8 batches, and at 50 epochs it is about two thirds of its 16-epoch size.
  MapPoPE - EM is DIRECTIONAL.
- Pre-registered? yes (PAPERTASK_PREREG.md).
  - P1 (EM - WM >= +0.20 and detectable): **NOT READ**, because the gate failed. The numbers
    would satisfy it at l=2048.
  - P2 (still detectable after rule 9): r = -0.461, so no loss-matching was needed. The
    contrast is not a loss gap.
  - P3 (MapPoPE - EM stays unmeasured): **held**.
  - P4 (falsifier): did not fire.
  - The file itself judges the gate mis-set, because WM's shortfall is systematic across
    budgets.
- Caveats the report must carry:
  - The axis is LENGTH. The effect is monotone in l (+0.067 / +0.186 / +0.287) and does not
    support "a shared kernel is better when the offset is fixed". Counterexamples on other
    fixed-offset tasks (MiniGrid, vocab, Match-Query, compositional, family tree) stand
    (EM_WM_THEORY.md 2a).
  - Floor of about 0.80 at pe=0.8.
  - One task.
- Sources: PAPERTASK_PREREG.md, PAPERTASK_RESULTS.md, PAPER_OOD_RERUN.md, PAPER_TASK_FLOORS.md,
  EM_WM_THEORY.md.
- Bears on: EM vs WM (the only map-task EM advantage); reproduction (WM's IID shortfall vs
  paper is systematic, not budget); floors methodology.

### A6 Context-destruction ablation on the paper task
- Dates: PAPER_TASK_ABLATION.md 2026-08-17.
- Question:
  - Does the paper task leave a content route to localisation open, making it less diagnostic
    than Match-Query?
  - Do trained models actually use the action stream?
- Task / environment: T=128, fresh obs_map (env seed 10000), 20x128 sequences per cell, scored
  at revisited observations. Measured blank rate 0.506. Vocab 21, so uniform NLL = ln(21) =
  3.04 nats.
- Arms: Vanilla, VanillaEM_P0 (A1 checkpoints).
- Seeds / batch: n=3.
- Validity gates: this experiment is itself the gate. `resample` (an on-manifold stream from
  an independent episode) is the trustworthy column; `shuffle` was reported first.
- Result:

  | variant | intact | shuffle actions | resample actions | shuffle obs | resample obs |
  |---|---|---|---|---|---|
  | Vanilla | 0.9889 +/- 0.0102 | 0.2314 +/- 0.0119 | 0.1783 +/- 0.0137 | 0.2915 +/- 0.0021 | 0.3092 +/- 0.0020 |
  | VanillaEM_P0 | 0.9873 +/- 0.0118 | 0.3831 +/- 0.0184 | 0.3617 +/- 0.0449 | 0.2680 +/- 0.0131 | 0.2877 +/- 0.0122 |

  - Every destroyed condition falls below the 0.506 floor. NLL in destroyed conditions is
    3.68-4.94 (actions resampled) and 4.50-5.64 (obs resampled), against 0.006-0.100 intact.
    The models are overconfident, not hedging.
  - Resample is more destructive than shuffle for Vanilla (0.178 vs 0.231, 3/3 seeds; NLL
    4.89 vs 4.63).
  - EM retains more than WM under action resampling (0.362 vs 0.178, 3/3). No mechanism is
    claimed.
- Status: EXPLORATORY by n (n=3). Each drop is ~50x its per-arm sd, and the gate verdict (the
  models use the path) is unambiguous.
- Pre-registered? The prediction was stated in the file: destroying actions should cost the
  paper task much less than Match-Query's -0.842. **FAILED.** The "Match-Query closes a route
  the paper task leaves open" framing is withdrawn. The author's own explanation for the
  below-floor collapse (the shuffle is off-manifold) was also falsified by the resample
  column.
- Caveats the report must carry:
  - This shows the models USE the action stream. It does not show the task is unsolvable
    without it; A3's index arms address that.
  - On other tasks shuffle and resample order the other way (A12, A21), so they are not
    interchangeable.
- Sources: PAPER_TASK_ABLATION.md.
- Bears on: validity of the paper task as a path-integration measurement; where/what.

### A7 Revisit accuracy by recurrence interval: why index models exceed the floor
- Dates: REVISIT_DISTANCE.md 2026-08-17.
- Question: Index models beat the marginal in likelihood (train loss 1.59-1.68 vs 2.079 nats)
  while sitting at the floor in accuracy. Where does that likelihood come from?
- Task / environment: paper task; revisits binned by steps since the cell was last visited.
  The per-bucket blank rate is the floor.
- Arms: Vanilla, RoPE, PlainFlat.
- Seeds / batch: not stated in the file ("n per seed" row empty); the A1/A3-era checkpoints.
- Validity gates: per-bucket floor.
- Result (REVISIT_DISTANCE.md):

  | variant | 1-2 | 3-4 | 5-8 | 9-16 | 17-32 | 33-64 | 65+ |
  |---|---|---|---|---|---|---|---|
  | Vanilla | 0.995 | 0.996 | 0.996 | 0.993 | 0.980 | 0.937 | 0.881 |
  | RoPE | 0.557 | 0.516 | 0.505 | 0.503 | 0.500 | 0.495 | 0.507 |
  | PlainFlat | 0.575 | 0.518 | 0.505 | 0.501 | 0.477 | 0.475 | 0.488 |
  | blank rate | 0.504 | 0.514 | 0.514 | 0.509 | 0.484 | 0.504 | 0.533 |
  | share of events | 19.3% | 16.3% | 24.7% | 24.6% | 8.9% | 4.5% | 1.7% |

  - Index models exceed the floor only at interval 1-2, by +0.05 to +0.07. Share-weighted
    this is +0.012, matching their aggregate +0.007 to +0.010 over the floor.
  - Verdict in the file: the retrace hypothesis is CONFIRMED.
- Status: EXPLORATORY (seed count not stated). The explanation is consistent with
  PAPER_OOD_WITH_POPE.md's OOD-d excess (+0.07 for RoPE over a ~0.20 floor).
- Pre-registered? no (hypothesis stated before the table).
- Caveats the report must carry: the likelihood-vs-accuracy reconciliation is arithmetic
  illustration, not a measurement.
- Sources: REVISIT_DISTANCE.md, PAPER_OOD_WITH_POPE.md.
- Bears on: where/what (index codes learn only local retraces from action tokens as content;
  no map).

### A8 Separate q0/k0 vs single p0 across map tasks (paper App. A.4/A.7 conjecture)
- Dates: PAPER_VALIDATION.md 2026-08-09; EM_P0_COMP.md, EM_COMP_SAMEBATCH.md 2026-08-09;
  EM_FIX_COMP.md 2026-08-08; MATCH_QUERY_EM.md 2026-08-15.
- Question: The paper suspects separate k0p/q0p is "beneficial" and "would create sparser
  attention values". Is it?
- Task / environment:
  - Paper task (A1).
  - Compositional motif task (cross_nb_acc, fresh env seed 10000; floor ~0.072 per
    BASELINE_TABLE D).
  - Match-Query (TE=512, TQ=256, 200 epochs, chance 0.0625, gates MATCH_QUERY_GATES.md).
- Arms: MapEM-Flat / VanillaEM (separate), VanillaEM_P0 (single p0), VanillaEM_Fixed (eq.13,
  compositional only), MapWM-Flat (control).
- Seeds / batch: n=3 throughout. EM_COMP_SAMEBATCH is one 3-arm batch. Match-Query is one
  3-arm batch whose WM control reproduces the earlier sweep's 0.888 to three decimals; this is
  determinism of the pipeline, not replication.
- Validity gates: the Match-Query gates and context destruction (Match-Query line).
- Result:
  - Compositional, cross_nb_acc at T=256 (EM_COMP_SAMEBATCH.md): MapWM-Flat 0.270 +/- 0.030,
    MapEM-Flat (separate) 0.097 +/- 0.013, VanillaEM_P0 0.264 +/- 0.025. VanillaEM_Fixed 0.158
    +/- 0.056 (EM_FIX_COMP.md). exact_acc at T=2048: EM_P0 0.769 +/- 0.083 vs MapWM-Flat
    0.646 +/- 0.046.
  - Match-Query (MATCH_QUERY_EM.md):

    | variant | TQ=256 | TQ=512 (OOD) | per-seed TQ=256 |
    |---|---|---|---|
    | MapWM-Flat | 0.888 +/- 0.140 | 0.902 +/- 0.117 | 0.731 / 1.000 / 0.934 |
    | MapEM separate | 0.450 +/- 0.332 | 0.385 +/- 0.323 | 0.107 / 0.769 / 0.475 |
    | MapEM single p0 | 0.808 +/- 0.168 | 0.789 +/- 0.188 | 0.736 / 1.000 / 0.689 |

    Paired single-p0 minus separate: +0.629 / +0.231 / +0.214, mean **+0.358**, 3/3.
  - Effect by task (MATCH_QUERY_EM.md): paper task +0.089, compositional +0.167, Match-Query
    +0.358. The file counts the conjecture refuted "on four independent tasks".
  - EM (either form) never beats WM on these tasks: 0.808 vs 0.888.
- Status: EXPLORATORY (n=3 on every task). The direction is consistent across four map tasks.
- Pre-registered? no.
- Caveats the report must carry:
  - **Scope-limited by AUDIT_2026-09-10.md #3 and RESULTS_INDEX.md.** On recency (k-back) the
    separate form is BETTER: +0.128 at n=24, but +0.073 on fresh seeds alone (9/16, MDE
    0.130, unmeasured). "Refuted" therefore holds only for map tasks, and the sign is
    task-dependent.
  - The Match-Query arms (other than MapWM-Flat) were never extended beyond n=3. MapWM-Flat
    itself fell from 0.888 to 0.730 at n=5 (N3_AUDIT.md).
  - Match-Query is not reproducible across batches (CLAUDE.md 2026-09-05/06).
  - No mechanism: the kernel-geometry account was falsified (A9).
- Sources: PAPER_VALIDATION.md, EM_P0_COMP.md, EM_FIX_COMP.md, EM_COMP_SAMEBATCH.md,
  MATCH_QUERY_EM.md, AUDIT_2026-09-10.md.
- Bears on: EM vs WM; q0/k0 phase-freedom work in the EM/WM line; paper-conjecture tests.

### A9 A_P kernel geometry at zero displacement (pre-registered falsification)
- Dates: AP_KERNEL_DIAGNOSTIC.md 2026-08-09.
- Question: Does the sign or peakedness of EM's position kernel at zero displacement explain
  when EM wins or loses?
- Task / environment: same-cell revisit pairs on vocab-16, vocab-256 and paper-task
  checkpoints (A1, A10).
- Arms: EM-sep and EM-p0 per config, including one good and one failed EM-p0 seed at vocab 256.
- Seeds / batch: per-checkpoint probe (1071-1294 pairs each), not a seed study.
- Validity gates: none.
- Result (AP_KERNEL_DIAGNOSTIC.md):

  | model | same-cell A_P | % negative | outcome |
  |---|---|---|---|
  | EM-sep vocab16 | -0.0884 | 100.0% | EM WINS (+0.027, 3/3) |
  | EM-sep vocab256 | -0.0424 | 46.4% | EM LOSES (-0.086) |
  | EM-p0 vocab16 | +0.5617 | 0.0% | -- |
  | EM-p0 vocab256 s0 (good 0.910) | +0.6543 | 0.0% | -- |
  | EM-p0 vocab256 s1 (failed 0.502) | +0.3443 | 27.0% | -- |
  | EM-sep paper task | -0.1677 | 77.3% | -- |
  | EM-p0 paper task | +0.3974 | 0.0% | -- |

  - The parameterisation hypothesis is **FALSIFIED**: kernel quality is inversely related to
    accuracy across configs.
  - The measurement cleanly separates the two parameterisations (separate form negative at
    zero displacement, single p0 positive).
- Status: EXPLORATORY (probe). The falsification stands.
- Pre-registered? yes, in the script. The falsification condition ("equally bad kernels where
  EM wins") fired, more strongly than stated.
- Caveats the report must carry:
  - The kernel's sign is a **gauge** (A_X absorbs kappa's sign; AUDIT_2026-09-10.md #5;
    EM_WM_THEORY.md 1a), which explains why a negative-at-zero kernel can still retrieve.
    That is consistent with the file's own closing reading.
  - EM_WM_THEORY.md P7 lists whether this diagnostic "needs narrowing" as open.
- Sources: AP_KERNEL_DIAGNOSTIC.md, AUDIT_2026-09-10.md, EM_WM_THEORY.md.
- Bears on: EM vs WM mechanism (withdrawn AND/OR-gate regime story); rule "representational
  property != performance".

### A10 Vocab sweep, old recipe, n=3 (paper Fig. 4c axis)
- Dates: VOCAB_SWEEP_MULTISEED.md 2026-08-09. VOCAB_SWEEP_RESULTS.md (single seed) is
  superseded.
- Question: Does EM scale better than WM with observation vocabulary, as the paper's Fig. 4c
  claims at l=16?
- Task / environment: clean torus, n_obs in {16, 256, 4096}. Train T=128, eval T=128 and
  T=512 on a fresh map. p_empty 0.5, so the floor is ~0.50.
- Arms: Vanilla, VanillaEM (separate), VanillaEM_P0.
- Seeds / batch: n=3, same batch, LinearLR at lr 3e-4 (the published recipe).
- Validity gates: floor check. n_obs=4096 is degenerate: the best of 27 runs is 0.500, i.e.
  always-blank.
- Result (VOCAB_SWEEP_MULTISEED.md), T=512:

  | n_obs | Vanilla | VanillaEM | VanillaEM_P0 | EM - WM (wins) | EM_P0 - WM (wins) |
  |---|---|---|---|---|---|
  | 16 | 0.950 +/- 0.010 | 0.977 +/- 0.006 | 0.977 +/- 0.005 | +0.027 (3/3) | +0.028 (3/3) |
  | 256 | 0.675 +/- 0.103 | 0.590 +/- 0.020 | 0.773 +/- 0.234 | -0.086 (1/3) | +0.097 (2/3) |
  | 4096 | 0.483 +/- 0.007 | 0.497 +/- 0.001 | 0.499 +/- 0.001 | degenerate | degenerate |

  - VanillaEM_P0 per-seed at n_obs=256, T=512: 0.910 / 0.502 / 0.906 (bimodal).
  - "Long sequences favour WM" is FALSIFIED (EM wins 3/3 at n_obs=16, T=512).
  - The paper's Fig. 4c direction is not reproduced at l=128/T=512.
- Status: EXPLORATORY (n=3; old recipe). Superseded for the capacity question by A11.
- Pre-registered? no.
- Caveats the report must carry:
  - Fig. 4c is at l=16 with vocabularies up to 10,000; this sweep does not test that regime.
  - The single-seed file's "VanillaEM crashes at 256 (0.562)" and "correction rescues both"
    readings belong to the withdrawn regime narrative.
- Sources: VOCAB_SWEEP_MULTISEED.md, VOCAB_SWEEP_RESULTS.md (superseded).
- Bears on: EM vs WM (paper Fig. 4c); withdrawn regime table.

### A11 Vocab sweep with a healthy EM arm, n=8 (pre-registered)
- Dates: VOCAB_EM_PREREG.md 2026-09-09; VOCAB_EM.md 2026-09-09 (last 2026-09-11).
- Question:
  - Does the paper's quadratic-capacity claim for EM hold once the init pathology (separate
    q0/k0) and rank r=2 are removed?
  - Was the n_obs=256 collapse a matter of recipe or of init?
- Task / environment: torus, held-out map, n_obs in {16, 64, 256} (4096 excluded as
  degenerate). T=128 is primary; T=512 is also reported.
- Arms: `Vanilla` (WM r2), `Vanilla_r4`, `VanillaEM`, `VanillaEM_P0`, `VanillaEM_P0_r4`.
- Seeds / batch: 5 arms x 3 vocab x 8 seeds = 120 runs in one batch, cosine at lr 1e-3 (not
  the published recipe, by design).
- Validity gates: a ceiling check in advance (n_obs=16 and 64 are at ceiling, so 256 is the
  primary cell). The reading order (per-seed min, then worst-to-second gap, then mean) was
  fixed in advance.
- Result (VOCAB_EM.md), n_obs=256, T=128:

  | arm | min | gap | sd | mean |
  |---|---|---|---|---|
  | Vanilla (WM r2) | 0.990 | 0.004 | 0.003 | 0.997 |
  | Vanilla_r4 | 0.518 | 0.482 | 0.171 | 0.940 |
  | VanillaEM | 0.796 | 0.031 | 0.036 | 0.849 |
  | VanillaEM_P0 | 0.504 | 0.225 | 0.171 | 0.849 |
  | VanillaEM_P0_r4 | 0.995 | 0.005 | 0.002 | 0.999 |

  - VanillaEM_P0_r4 scores 1.000 / 1.000 / 0.999 across n_obs 16 / 64 / 256.
  - EM_P0_r4 - Vanilla_r4 is -0.000 / +0.000 / +0.060. After dropping each arm's worst seed it
    is +0.0001 (n_obs 64) and **+0.0000** (n_obs 256). The whole +0.060 is one collapsed
    Vanilla_r4 seed.
  - Every contrast is unmeasured (MDE 0.169 at n_obs=256).
  - VanillaEM_P0 at n_obs=256, T=512, per-seed: 0.491 0.654 0.755 0.812 0.875 0.949 0.955
    0.957. The low seed survives the better recipe.
  - Which fix removes collapse is task-dependent. On this torus task it is rank; on MiniGrid
    it is the origin.
- Status: EXPLORATORY. All contrasts are unmeasured at n=8, and the capacity claim has no
  support here.
- Pre-registered? yes (VOCAB_EM_PREREG.md).
  - P1 (the gap grows with vocab): the predicted shape appeared, but the gap is one seed.
    Refuted on the trimmed reading.
  - P2 (no collapse for EM_P0_r4): held (gap 0.005).
  - P3 (the recipe explains the VanillaEM_P0 collapse): not supported. The collapse persists,
    and the author's init attribution was also wrong here.
- Caveats the report must carry:
  - The recipe differs from A10, so the numbers are not comparable.
  - The file's line "EM is worse remains dead" was refuted the next day by recency (EM_P0 - WM
    = -0.375, 0/8; RECENCY_EM_RESULTS.md).
  - One task at l=128.
- Sources: VOCAB_EM_PREREG.md, VOCAB_EM.md.
- Bears on: EM vs WM (capacity claim); rank line (r=4 recommendation is a stabiliser for EM
  here); reliability vs capacity.

### A12 Family tree, depth 5: non-commutativity vs path integration (paper App. B.2.2)
- Dates: FAMILY_TREE_GATES.md 2026-08-16 (gate corrected 2026-08-09 per banner);
  FAMILY_TREE_RESULTS.md 2026-08-15; FAMILY_TREE_WM_GAP.md 2026-09-07 (power caveat added);
  ABLATE_FAMILY_TREE.md 2026-08-17.
- Question: The paper motivates MapEM-NC with a family tree (mother and father do not
  commute) and never runs it. Does non-commutative machinery beat a commutative model on one?
- Task / environment: ancestor tree depth 5 (63 nodes), 8 observation types, 8 relational
  actions, scored at revisited nodes. Train T=64, OOD T=128. Chance 0.125.
  - Floors and gates (FAMILY_TREE_GATES.md): **hub-node floor 0.1628**; marginal 0.1280;
    last-obs 0.1240 (corrected from 0.1580); n-gram o1/o2/o3/o5 0.1235 / 0.1202 / 0.1241 /
    0.1283; oracle 1.0000.
  - Well-posedness: revisit rate 0.195; measured non-commutativity **1.000**.
- Arms: `MapEM-NC-L`, `MapEM-NC-NL` (non-commutative, paper B.2.2: K=n(n-1)/2 skew
  generators, sequential product), `MapEM-os` / VanillaEM_P0 (commutative control),
  `Plain-Flat` (index), `MapWM-Flat`, `Level15`.
- Seeds / batch: n=3. FAMILY_TREE_WM_GAP retrained five arms in one batch: 2 layers, 100
  epochs.
- Validity gates:
  - All shortcut gates at chance.
  - Context destruction (ABLATE_FAMILY_TREE.md) PASSES. MapEM_NC_NL falls 0.7282 -> 0.2486
    (shuffle actions) / 0.2740 (resample actions) / **0.1626** (shuffle obs), and the last
    lands on the hub floor to three decimals. VanillaEM_P0 0.7126 -> 0.2477 / 0.2709 /
    0.1720; PlainFlat 0.6123 -> 0.2432 / 0.2615 / 0.1852.
  - For destroyed actions the "no context" floor is nearer 0.25, because the observation
    history is intact.
- Result:
  - From FAMILY_TREE_WM_GAP.md:

    | model | T=64 | T=128 | per-seed T=64 |
    |---|---|---|---|
    | Level15 | 0.843 +/- 0.015 | 0.789 +/- 0.027 | 0.830 / 0.859 / 0.841 |
    | MapWM-Flat | 0.805 +/- 0.072 | 0.746 +/- 0.080 | 0.835 / 0.858 / 0.724 |
    | MapEM-NC-NL | 0.729 +/- 0.010 | 0.672 +/- 0.012 | 0.720 / 0.740 / 0.726 |
    | MapEM-os (commutative) | 0.715 +/- 0.008 | 0.659 +/- 0.015 | 0.712 / 0.725 / 0.709 |
    | Plain-Flat (index) | 0.601 +/- 0.011 | 0.550 +/- 0.031 | 0.589 / 0.610 / 0.603 |

    MapEM-NC-L (FAMILY_TREE_RESULTS.md only): 0.720 +/- 0.011 at T=64, 0.671 +/- 0.006 at
    T=128.
  - Paired (FAMILY_TREE_RESULTS.md): NC-L - commutative +0.005 (3/3); NC-NL - commutative
    +0.014 (3/3); commutative - index +0.115 (3/3).
  - Contrasts computed in N3_AUDIT.md from the per-seed data:

    | contrast | delta | sd | MDE | seeds + | verdict |
    |---|---|---|---|---|---|
    | MapWM-Flat - MapEM-NC-NL | +0.077 | 0.068 | 0.111 | 2/3 | unmeasured |
    | MapWM-Flat - MapEM-os | +0.090 | 0.065 | 0.106 | 3/3 | unmeasured |
    | Level15 - MapWM-Flat | +0.038 | 0.069 | 0.111 | 2/3 | unmeasured |
    | MapEM-NC-NL - MapEM-os | +0.013 | 0.005 | 0.008 | 3/3 | DETECTABLE |
    | MapWM-Flat - Plain-Flat | +0.205 | 0.073 | 0.118 | 3/3 | DETECTABLE |

- Status: EXPLORATORY (n=3, never extended).
  - Within n=3: path integration over index (+0.205) and the paper's own non-commutativity
    axis (+0.013) both clear their MDE.
  - The non-commutativity gain is small beside path integration, and it carries 14.4x (vs
    2.5x) length-scaling cost (A14).
- Pre-registered? no.
- Caveats the report must carry:
  - "Plain WM beats every published variant" is unmeasured (N3_AUDIT.md). Its whole margin is
    one MapWM seed at 0.724. Do not write "beats" below n=8.
  - The "batch reproduces the published numbers exactly" claim (three arms to three decimals)
    is almost certainly determinism of same-seed reruns, not replication (rule C27).
  - Level15's +0.038 is variance reduction driven by one seed (t ~0.89).
  - The paper's group-theory claim is correct. Representational necessity does not translate
    into a performance gap here.
- Sources: FAMILY_TREE_GATES.md, FAMILY_TREE_GATES.json, FAMILY_TREE_RESULTS.md,
  FAMILY_TREE_WM_GAP.md, ABLATE_FAMILY_TREE.md, N3_AUDIT.md, BASELINE_TABLE.md F.
- Bears on: non-commutativity (paper extension); where/what on a non-spatial relational
  structure; correction line (variance reduction).

### A13 Family tree, depth 7 (memorisation caveat test)
- Dates: FAMILY_TREE_D7_GATES.md, FAMILY_TREE_D7_RESULTS.md 2026-08-16.
- Question: Was the depth-5 null a memorisation artefact? Does a bigger tree give
  non-commutativity room?
- Task / environment: depth 7 (255 nodes), 8 obs types, 8 actions.
  - Gates: chance 0.1250, **hub floor 0.1442**, marginal 0.1290, last-obs 0.1581, n-gram
    o1/o2/o3/o5 0.1285 / 0.1238 / 0.1265 / 0.1294.
  - Revisit rate 0.215; non-commutativity 1.000.
- Arms: MapEM-NC-L, MapEM-NC-NL, commutative control, Plain-Flat.
- Seeds / batch: n=3 (implied by the 2/3 counts).
- Validity gates: shortcut gates at chance. No context destruction at depth 7.
- Result (FAMILY_TREE_D7_RESULTS.md), paired effects:

  | effect | depth 5 | depth 7 |
  |---|---|---|
  | NC-L - commutative | +0.005, 3/3 | +0.0017, 2/3 (one seed -0.003) |
  | NC-NL - commutative | +0.014, 3/3 | +0.0037, 2/3 (one seed -0.006) |
  | path integration - index | +0.115 | +0.180 |

  - Non-commutativity shrank with scale and lost consistency. Path integration grew.
- Status: EXPLORATORY (n=3). Per-arm accuracies are not recorded in the file.
- Pre-registered? no (the prediction was written as a caveat in FAMILY_TREE_RESULTS.md).
- Caveats the report must carry:
  - No absolute accuracies are given.
  - The NC deltas have no sd or MDE.
  - The file's "34x less at L=2048" refers to TIMING_BENCHMARK's L=2048 wall-clock gap
    (Vanilla vs MapEM-NC).
- Sources: FAMILY_TREE_D7_GATES.md, FAMILY_TREE_D7_GATES.json, FAMILY_TREE_D7_RESULTS.md.
- Bears on: non-commutativity; path integration on non-spatial structure.

### A14 Wall-clock scaling with length: parallel scan vs sequential models
- Dates: TIMING_BENCHMARK.md first 2026-08-15, re-measured with `torch.no_grad()` (banner
  2026-08-09 wording; last commit 2026-08-16).
- Question: Does MapFormer's parallel cumsum scan scale better in L than sequential
  alternatives (MapEM-NC's matrix product, TEM's RNN)?
- Task / environment: batch 4, d_model 128, n_layers 2. Median of 15 reps after 3 warmups,
  with CUDA synchronize around each timed region. IQR typically <0.5% of the median. Units ms.
- Arms (params):

  | arm | params |
  |---|---|
  | Vanilla | 405,472 |
  | VanillaEM_P0 | 405,600 |
  | PlainFlat | 405,024 |
  | MapEM_NC_L | 430,016 |
  | TEMFaithful | 20,705 |

  Models are not parameter-matched; only scaling is compared.
- Seeds / batch: n/a (timing).
- Validity gates: the first run timed autograd graph construction in its forward-only rows,
  which biased results toward the claim. Those rows were corrected.
- Result (TIMING_BENCHMARK.md), forward+backward:

  | variant | L=128 | L=2048 | growth |
  |---|---|---|---|
  | Vanilla | 4.8 | 12.1 | 2.5x |
  | VanillaEM_P0 | 4.3 | 16.7 | 3.9x |
  | PlainFlat | 4.2 | 11.4 | 2.7x |
  | MapEM_NC_L | 28.5 | 410.1 | 14.4x |
  | TEMFaithful | 163.2 | 19646.3 | 120.4x |

  - At L=2048 Vanilla is 34x faster than MapEM-NC and 1624x faster than TEMFaithful
    (fwd+bwd).
  - Forward-only (corrected) at L=2048: Vanilla 4.7, PlainFlat 4.6, MapEM_NC_L 88.3,
    TEMFaithful 1803.3. Inference gaps are 18.8x (vs MapEM-NC) and 384x (vs TEMFaithful). As
    first published they were 26.9x and 478x, overstated by about a third.
  - Forward-only scaling: 3.5-3.9x (parallel) vs 13.2x (MapEM-NC) vs 44.5x (TEMFaithful).
- Status: CITABLE (a deterministic engineering measurement, not seed-based).
- Pre-registered? no.
- Caveats the report must carry:
  - Parallel models are overhead-dominated below L~1024; most of their growth is 1024 -> 2048.
  - TEM's constant includes Python-loop overhead. Only the SHAPE is architectural (CLAUDE.md).
  - Not parameter-matched.
  - The mechanical explanation in RESULTS_INDEX.md ("TEM's cost is the missing group law") is
    an argument from the review, not a measurement.
- Sources: TIMING_BENCHMARK.md.
- Bears on: the paper's parallelism claim; the cost side of non-commutativity (A12/A13) and
  TEM comparisons.

### A15 April clean/noise baseline batch: RoPE, LSTM, CoPE, MambaLike vs MapWM/MapEM
- Dates: generated 2026-04-26 (RESULTS_PAPER.md, LONG_SEQ_clean.md, PER_VISIT_clean.md,
  ZERO_SHOT_TRANSFER_clean.md); ZERO_SHOT_TRANSFER_clean_brokeninit.md 2026-04-23;
  NOISE_CLEAN_REVALIDATION.md 2026-07-16.
- Question: Can generic sequence models (index RoPE, LSTM, CoPE, a diagonal-A Mamba-style SSM)
  do the aliased torus task? How do the paper's two backbones compare on fresh maps, at length
  and under action noise?
- Task / environment:
  - Torus 64x64, 16 obs, p_empty 0.5 (so the blank floor is ~0.50; 0.506 measured at T=128 in
    PAPER_TASK_ABLATION.md).
  - Configs: `clean`, and `noise` (10% action replacement, i.e. stochastic transitions).
  - Train T=128. Eval on a fresh obs_map at T=128 and T=512; LONG_SEQ extends to T=2048.
  - PER_VISIT bins by visit count at T=512.
  - ZERO_SHOT axis 2 tests biased action distributions at T=512.
- Arms: Vanilla (MapWM), VanillaEM (separate q0/k0), RoPE, LSTM, CoPE, MambaLike, plus
  correction-line arms (Level1, Level15, Level15EM, PC, Level15PC; other line).
- Seeds / batch: n=3 (seeds 0,1,2). 50 epochs x 156 batches x 128 (~1M sequences), LinearLR
  default.
- Validity gates:
  - NOISE_CLEAN_REVALIDATION.md: fresh retrains under current code are bit-identical to the
    stored clean/noise checkpoints. That rules out the lm200 RNG defect for these configs. It
    is determinism, not replication.
  - No context destruction, rule 9 or convergence check on this batch.
- Result:
  - RESULTS_PAPER.md, OOD (fresh map):

    | variant | clean T=128 | clean T=512 | noise T=128 | noise T=512 |
    |---|---|---|---|---|
    | Vanilla | 0.992 +/- 0.006 | 0.913 +/- 0.037 | 0.954 +/- 0.008 | 0.739 +/- 0.062 |
    | VanillaEM | 1.000 +/- 0.000 | 0.972 +/- 0.003 | 0.957 +/- 0.009 | 0.765 +/- 0.138 |
    | RoPE | 0.635 +/- 0.072 | 0.463 +/- 0.026 | 0.608 +/- 0.070 | 0.469 +/- 0.027 |
    | LSTM | 0.860 +/- 0.010 | 0.800 +/- 0.004 | 0.798 +/- 0.018 | 0.743 +/- 0.010 |
    | CoPE | 0.741 +/- 0.037 | 0.679 +/- 0.024 | 0.687 +/- 0.032 | 0.633 +/- 0.022 |
    | MambaLike | 0.591 +/- 0.005 | 0.573 +/- 0.007 | 0.586 +/- 0.003 | 0.568 +/- 0.004 |

  - Clean length sweep (LONG_SEQ_clean.md), T=128 / 512 / 1024 / 2048:

    | variant | T=128 | T=512 | T=1024 | T=2048 |
    |---|---|---|---|---|
    | Vanilla | 0.990 | 0.921 | 0.795 | 0.635 |
    | VanillaEM | 1.000 | 0.976 | 0.918 | 0.827 |
    | LSTM | 0.880 | 0.831 | 0.788 | 0.739 |
    | MambaLike | 0.602 | 0.584 | 0.571 | 0.558 |

    At T=2048 LSTM exceeds Vanilla (0.739 vs 0.635).
  - Per-visit (PER_VISIT_clean.md), k=2 first-revisit accuracy: Vanilla 0.910 +/- 0.046,
    VanillaEM 0.979 +/- 0.006, LSTM 0.809 +/- 0.011, MambaLike 0.577 +/- 0.006.
  - Biased actions at T=512 (ZERO_SHOT_TRANSFER_clean.md): Vanilla uniform 0.917 /
    mostly_east 0.757 / mostly_NS 0.940 / diagonal_NE 0.860; VanillaEM 0.976 / 0.866 / 0.974 /
    0.899; LSTM 0.808 / 0.743 / 0.803 / 0.822; MambaLike 0.587 / 0.563 / 0.576 / 0.584.
    RoPE (brokeninit file only; its RoPE checkpoints are unaffected by the Level15EM init
    change) 0.471 / 0.462 / 0.432 / 0.456.
- Status: EXPLORATORY (n=3, LinearLR, no gates on this batch).
- Pre-registered? no.
- Caveats the report must carry:
  - Floors are not stated in these files. RoPE at T=512 (0.463) is at or below a ~0.5 blank
    floor, and MambaLike (0.573) sits just above it.
  - "Reproduces the paper's Mamba failure (Table 3)" is qualitative only. The paper's Table 3
    is at l=16 on a related task, and the v4 numbers changed (Mamba 0.38/0.66/0.30,
    CLAUDE.md).
  - MambaLike is a diagonal-A approximation; MAmPa was never implemented.
  - CoPE is a reimplementation.
  - "VanillaEM > Vanilla at T=512" is consistent with A4/A5 but from a different recipe.
  - "Vanilla is the paper-faithful separate-q0/k0 EM" applies to the VanillaEM row, unlike A3's
    EM_P0.
  - lm200 columns in the same files are void (see Excluded).
  - BASELINE_TABLE.md's coverage-gap line says "TEM / Mamba / LSTM exist only in the lm200
    column". That is inaccurate for LSTM and MambaLike, which also have these clean/noise rows,
    though not on the 16-epoch paper-task batch of A3.
- Sources: RESULTS_PAPER.md, LONG_SEQ_clean.md, PER_VISIT_clean.md, ZERO_SHOT_TRANSFER_clean.md,
  ZERO_SHOT_TRANSFER_clean_brokeninit.md, NOISE_CLEAN_REVALIDATION.md, DETAILED_RESULTS.md
  (narrative).
- Bears on: non-MapFormer baselines; EM vs WM at length (older evidence); stochastic-transition
  framing.

### A16 TEM-t (transformer-formulation TEM) and TEM-GRU baselines
- Dates: TEM_RESULTS.md, TEM_T_RESULTS.md, TEM_T_MULTISEED.md (banners 2026-08-09; runs
  May 2026).
- Question: How does Whittington 2022's TEM-t (sequential `e_t = ReLU(e_{t-1} W_{a_t})`,
  parameter-matched to MapFormer-EM at ~250K) compare with MapFormer's parallel angle on
  clean/noise?
- Task / environment: torus clean and noise (A15 configs). Fresh-map OOD at T=128 and T=512.
- Arms: `TEM_T` (with two LayerNorms added to fix NaN; `e_in_rnn` deviates from the paper),
  `TEM` (GRU + factorised g/x + Hebbian memory; "TEM-Lite", not a faithful TEM), Vanilla,
  VanillaEM, Level15, Level15EM.
- Seeds / batch: TEM_T n=3 (TEM_T_MULTISEED.md). TEM-GRU is single seed.
- Validity gates: none. The clean/noise sections are covered by NOISE_CLEAN_REVALIDATION.
- Result:
  - TEM_T_MULTISEED.md, clean: TEM_T 0.946 +/- 0.003 (T=128), 0.856 +/- 0.006 (T=512), NLL
    0.576. Compare Vanilla 0.993 +/- 0.007 / 0.911 +/- 0.035 and VanillaEM 1.000 +/- 0.000 /
    0.972 +/- 0.003.
  - TEM_T_MULTISEED.md, noise: TEM_T 0.759 +/- 0.011 / 0.668 +/- 0.022. Compare Vanilla
    0.757 +/- 0.013 / 0.638 +/- 0.035 and VanillaEM 0.755 +/- 0.007 / 0.640 +/- 0.081.
  - TEM-GRU single seed (TEM_RESULTS.md): clean 0.772 / 0.692; noise 0.662 / 0.584.
- Status: EXPLORATORY (n=3 / n=1).
- Pre-registered? no.
- Caveats the report must carry:
  - Pre-bug-fix TEMFaithful rows were removed from these files.
  - The Vanilla / VanillaEM values here differ slightly from RESULTS_PAPER.md (clean T=512
    0.911 vs 0.913) because they come from a separate evaluation pass.
  - TEM_T's lm200 "win" (0.847) is void.
- Sources: TEM_RESULTS.md, TEM_T_RESULTS.md, TEM_T_MULTISEED.md, RESULTS_SUMMARY_2026-05-10.md
  Part I (bug fixes).
- Bears on: TEM baselines; sequential vs parallel position (cost in A14).

### A17 TEMFaithful (per-action orthogonal W_a + Hopfield memory): clean, noise, FFN test
- Dates: TEM_BACKGROUND_BASELINES.md 2026-05-15; TEM_NOISE_FFN_RESULTS.md (banner 2026-08-09).
- Question:
  - Where does a faithful TEM stand on the single-env clean and noise torus?
  - Does adding a per-position FFN close its clean-regime lag?
- Task / environment: torus, n_landmarks=0 (clean) or action noise p=0.10. T=128 and T=512
  OOD.
- Arms: `TEMFaithful` (W_a = exp(skew(A_a)), post-fix predict-then-update order),
  `TEMFaithful_FFN`. References: Level15, Vanilla.
- Seeds / batch: n=3 each. References are from other batches.
- Validity gates: the predict-then-update bug fix (the old order queried memory with the
  pre-action g). NOISE_CLEAN_REVALIDATION has TEMFaithful in its noise rows (bit-identical).
- Result:
  - Clean single env (TEM_BACKGROUND_BASELINES.md): TEMFaithful 1.000 +/- 0.000 at T=128;
    **0.966 +/- 0.008** at T=512; NLL 0.182 +/- 0.049.
  - TEMFaithful_FFN clean (TEM_NOISE_FFN_RESULTS.md): 1.000 +/- 0.000; **0.969 +/- 0.002**;
    NLL 0.145 +/- 0.003. Level15 reference 0.993, NLL 0.039.
  - Noise (TEM_NOISE_FFN_RESULTS.md): TEMFaithful 0.762 +/- 0.003 (T=128), **0.709 +/- 0.002**
    (T=512), NLL 1.216 +/- 0.010. Cross-batch references: Vanilla 0.638, Level15 0.702.
  - Reading in REPORT_v2.md 7.2: the FFN does not close the clean lag (0.966 -> 0.969, within
    noise), so the missing-FFN hypothesis is largely falsified. NLL improves.
- Status: EXPLORATORY (n=3; cross-batch references).
- Pre-registered? The decision rules are stated in the file. The "FFN does not help accuracy"
  branch fired.
- Caveats the report must carry:
  - TEM_BACKGROUND_BASELINES.md also has a "multi-env held-out CLEAN" row (0.970 +/- 0.008).
    That experiment family (MULTIENV_CLEAN_2x2.md) was archived as void wholesale, so do not
    cite it.
  - NOISE_CLEAN_REVALIDATION.md lists TEMFaithful noise acc 0.907 at T=512 (env seed 0, 120
    trials, one seed). That is a different evaluation from the 0.709 above (see
    disagreements).
  - The TEMFaithful_FFN lm200 rows are void.
  - Parameters: TEMFaithful ~20-45K vs MapFormer ~250K (DETAILED_RESULTS.md parameter budget;
    TIMING 20,705).
- Sources: TEM_BACKGROUND_BASELINES.md, TEM_NOISE_FFN_RESULTS.md, REPORT_v2.md sec 7.2-7.3,
  REPORT.md 5.4.
- Bears on: TEM baseline (factorised where/what with explicit memory); cost (A14).

### A18 CSCG stitching negative control, ported to attention
- Dates: STITCH_ATTENTION.md 2026-08-16.
- Question: CSCG's stitching negative control asks whether the model distinguishes two cells
  that emit the same symbol after a shared prefix. It also asks whether the model retrieves a
  room-B memory reached through room A's frame. Does MapFormer's attention show both?
- Task / environment: two-room stitching episodes. Paired statistic `join_share` (join arm
  minus confound arm) at layer 0; its floor is exactly 0. Also a transitive-tail B-share and a
  retrieval-concentration ratio (1.0 = no retrieval).
- Arms: MapWM-Flat (600,660 params), PlainFlat (600,212).
- Seeds / batch: n=3 seeds x 200 episodes each.
- Validity gates: paired design (visit count, recency and token identity cancel). A bootstrap
  over episodes is also reported but understates uncertainty; the per-seed spread is primary.
- Result (STITCH_ATTENTION.md), layer 0:
  - MapWM-Flat paired diff **+0.1306 +/- 0.0242** (per-seed +0.1033 / +0.1495 / +0.1388).
    Transitive tail diff +0.0210 +/- 0.0155. Room-B-only retrieval concentration 2.98x
    (per-seed 2.59 / 1.56 / 4.79x) against a within-room yardstick of 3.32x.
  - PlainFlat paired diff **-0.0053 +/- 0.0158**. Tail +0.0007 +/- 0.0026. Concentration 1.16x.
  - Held-out revisit accuracy: MapWM-Flat 0.971-0.974; PlainFlat 0.57-0.61.
- Status: EXPLORATORY (n=3). Every MapWM seed is above the PlainFlat range.
- Pre-registered? no.
- Caveats the report must carry (from the file):
  - This does not show the discrimination is specifically path integration; the arms also
    differ in approach-path content.
  - It is not a clean architecture comparison, because PlainFlat did not learn the map
    (0.57-0.61).
  - The transitive magnitude is unstable.
  - CSCG reports no number, so this is a port of their evaluation style, not a reproduction.
- Sources: STITCH_ATTENTION.md.
- Bears on: where/what (disambiguating aliased cells by position); CSCG comparison.

### A19 OOD grid with the omega-rescaling trick (older protocol; superseded)
- Dates: OOD_GRID_RESULTS.md 2026-05-11; OMEGA_RESCALE_clean.md 2026-04-24.
- Question: An earlier implementation of Table 2's OOD-d/OOD-s. At eval time trained omega is
  multiplied by N/N' (called "the paper's omega-rescaling trick" in the file).
- Task / environment: the April clean checkpoints (A15). Held-out map, env seed 1000. IID
  64x64 T=128; OOD-d 32x32 pe=0.2 T=64; OOD-s 128x128 pe=0.8 T=512.
- Arms: RoPE, Vanilla, Level15, Level15GSF_NoDrop, TEMFaithful.
- Seeds / batch: n=3, 30 trials each.
- Validity gates: none.
- Result:
  - OOD_GRID_RESULTS.md, with omega rescaled:

    | | Vanilla | RoPE | TEMFaithful |
    |---|---|---|---|
    | IID | 0.993 +/- 0.007 | 0.645 +/- 0.071 | 1.000 |
    | OOD-d | 0.367 +/- 0.185 | 0.472 +/- 0.110 | 0.989 +/- 0.008 |
    | OOD-s | 0.799 +/- 0.074 | 0.741 +/- 0.042 | 0.979 +/- 0.003 |

  - OMEGA_RESCALE_clean.md shows the rescaling itself is harmful for Vanilla. At test grid 32,
    Vanilla scores 0.954 +/- 0.015 with omega untouched and 0.310 +/- 0.108 rescaled; at grid
    128, 0.988 +/- 0.013 vs 0.681 +/- 0.199. VanillaEM at grid 32: 0.970 vs 0.938.
- Status: EXPLORATORY. **Superseded** for reproduction by A4/A5, which do not rescale omega
  and give Vanilla OOD-d 0.958 / 0.943.
- Pre-registered? no.
- Caveats the report must carry:
  - Whether the paper rescales omega at test time was not verified in the files read.
  - Its OOD-d floor is ~0.2, so Vanilla's 0.367 with rescaling is far below its unrescaled
    level, not a model failure.
- Sources: OOD_GRID_RESULTS.md, OMEGA_RESCALE_clean.md.
- Bears on: reproduction protocol; omega scale-coupling.

### A20 NumberLine: arithmetic as 1D navigation
- Dates: NUMBERLINE_RESULTS.md 2026-05-22; CAPACITY_PERREGIME.md NumberLine section
  (first 2026-05-22).
- Question: On a 1D additive torus where theta literally computes (a+b+...) mod N, does the
  correction line extrapolate to longer chains? Is any gain capacity?
- Task / environment: N=64, 6 ops (+/-1, +/-2, +/-3), predict the obs token at a revisited
  value. Train chain 128 ops, OOD chain 512. n_landmarks 0 (run_numberline.sh). No floor
  measured.
- Arms: Vanilla, Level15, Vanilla_ExtraHead (capacity control).
- Seeds / batch: n=3; 50 epochs x 156 batches (run_numberline.sh).
- Validity gates: none (no n-gram gates, context destruction or floor).
- Result (CAPACITY_PERREGIME.md):

  | variant | in-dist T=128 | OOD T=512 | T=512 NLL |
  |---|---|---|---|
  | Vanilla | 0.925 +/- 0.009 | 0.633 +/- 0.071 | 2.024 +/- 0.666 |
  | Vanilla_ExtraHead | 0.986 +/- 0.000 | 0.662 +/- 0.130 | 2.542 +/- 1.324 |
  | Level15 | 0.902 +/- 0.023 | 0.841 +/- 0.056 | 0.521 +/- 0.292 |

- Status: EXPLORATORY (n=3, ungated, cross-batch capacity arm).
- Pre-registered? no.
- Caveats the report must carry:
  - The file notes both in-distribution accuracies are low and "revisit-prediction needs work".
  - No floor has been measured.
  - Given the later correction-line findings (L15_ABLATION: no component load-bearing; benefit
    does not scale with drift), the "self-correcting accumulator" interpretation is not
    licensed.
- Sources: NUMBERLINE_RESULTS.md, CAPACITY_PERREGIME.md, REPORT_v2.md sec 6.
- Bears on: correction line (OOD-length stabilisation signature); non-spatial path
  integration.

### A21 Context-destruction ablation on the compositional task (gate for another line)
- Dates: ABLATE_COMPOSITIONAL.md 2026-08-17.
- Question: Is compositional cross_nb_acc read off the path?
- Task / environment: compositional motif task; metric cross_nb_acc. There is no analytic
  floor; the destroyed rows are the empirical floor.
- Arms: Vanilla (MapWM-Flat), Hourglass_k2 (MapWM-Hier), PlainFlat.
- Seeds / batch: n=3.
- Validity gates: this is the gate.
- Result (ABLATE_COMPOSITIONAL.md):
  - Vanilla 0.2708 -> 0.0234 (shuffle actions) / 0.0658 (resample actions).
  - Hourglass_k2 0.4279 +/- 0.1808 -> 0.0294 / 0.0854.
  - PlainFlat 0.2162 -> 0.0075 / 0.0724.
  - Headroom lost: 76% / 80% / 67%. PASSES.
  - Resample is less destructive than shuffle here, the opposite of the paper task.
- Status: EXPLORATORY (n=3 gate). The verdict is PASS.
- Pre-registered? no.
- Caveats the report must carry: Hourglass_k2's wide intact sd reflects the known seed-1
  outlier.
- Sources: ABLATE_COMPOSITIONAL.md.
- Bears on: hierarchy line (validity of the compositional task); method (shuffle vs resample).

---

## Excluded

| file / claim | reason | killed by |
|---|---|---|
| All lm200 sections of RESULTS_PAPER.md, TEM_RESULTS.md, TEM_T_RESULTS.md, TEM_T_MULTISEED.md, TEM_NOISE_FFN_RESULTS.md, DETAILED_RESULTS.md headline column | non-converged April checkpoints; ranking tracks convergence | CLAUDE.md RETRACTION 2026-07-16; CORRECTED_LM200_LEADERBOARD.md; NOISE_CLEAN_REVALIDATION.md |
| TEM_T lm200 "win" (0.847 vs Level15 0.819) | lm200 | same |
| Cross-scale TEM comparison (TEM dominates size 32), in TEM_CROSSSCALE_DIAGNOSTIC.md, EM_HOPFIELD_CROSSSCALE.md, HOPFIELD_NOMAINAP_RESULTS.md, PERSCALE_OMEGA_RESULTS.md, GENERALIZATION_REPORT.md sec 3/5/6, REPORT_v2.md sec 2.3/3 | training scripts use `--n-landmarks 200` (run_em_perscale_and_hopfield.sh, run_hopfield_nomainap.sh, run_perscale_omega.sh); source tables MULTISIZE_RESULTS.md, TEM_NOVEL_ENV_RESULTS.md, SINGLE_SIZE_CONTROL.md are in archive/void | archive/void banners (lm200) |
| TOPOLOGY_RESULTS.md and GENERALIZATION_REPORT sec 2 (cross-topology) | `run_topology.sh --n-landmarks 200`; single seed; the multi-seed version is in archived TEM_NOVEL_ENV_RESULTS.md | archive/void/TEM_NOVEL_ENV_RESULTS.md |
| Multi-env held-out (GENERALIZATION_REPORT sec 1, both lm200 and CLEAN halves; TEM_BACKGROUND_BASELINES multi-env row; RESULTS_SUMMARY Part IX) | archived as void wholesale | archive/void/MULTIENV_CLEAN_2x2.md, MULTIENV_RESULTS.md |
| Cross-class torus+DoorKey (MULTICLASS_MULTISEED_RESULTS.md, GENERALIZATION_REPORT sec 4) | torus envs built with `torus_n_landmarks=200` default (environment_multiclass.py:67), not overridden | lm200 retraction (inferred from code; not banner-marked) |
| Long-T lm200 table, sparse landmarks lm10/lm50, TEM parameter-scaling (GENERALIZATION_REPORT sec 7-8, RESULTS_SUMMARY Part IX, REPORT_v2 sec 7.1, README "Match or beat TEMFaithful") | lm200 / landmark checkpoints | archive/void/LONGT_EVAL_RESULTS.md, SPARSE_LANDMARKS_RESULTS.md, TEM_SCALING_RESULTS.md |
| "TEMFaithful is worst baseline" (pre-fix 0.42) and all pre-fix TEMFaithful rows | predict-then-update bug | TEM_RESULTS.md / TEM_T_RESULTS.md banners |
| WM-vs-EM regime table (AND-gate vs OR-gate; "long OOD favours WM"; "landmarks favour WM") in REPORT.md 3.2, REPORT_v2.md 9.2, RESULTS_SUMMARY Part IV, VOCAB_SWEEP_RESULTS.md preamble | contradicted by own clean row and vocab n=3; landmark rows void; MapWM is not additive | RESULTS_SUMMARY Part IV retraction; VOCAB_SWEEP_MULTISEED.md; AUDIT_2026-09-10.md #1 |
| A_P kernel geometry as mechanism for EM deficit | pre-registered falsification | AP_KERNEL_DIAGNOSTIC.md |
| "Paper target WM 0.955 / EM 0.999" | numbers in no paper table | PAPER_VALIDATION.md correction; CLAUDE.md 2026-08-09 |
| "Our reproduction matches the paper exactly" (WM IID 0.988 at n=3) | n=8 gives 0.969 +/- 0.037; v4 target is 1.00 | PAPER_OOD_EXTENDED_n8.md; CLAUDE.md v4 note |
| "MapPoPE-Flat and MapEM-os have non-overlapping ranges at l=2048" | n=8 bands overlap | PAPER_OOD_EXTENDED_n8.md |
| "Match-Query closes a content route the paper task leaves open" | prediction failed | PAPER_TASK_ABLATION.md |
| "Shuffle is off-manifold, hence below-floor collapse" | resample gives same collapse | PAPER_TASK_ABLATION.md |
| "EM and WM tie within 0.004 on map tasks" | false at extended length | EM_WM_THEORY.md; RESULTS_INDEX.md |
| "Separate q0/k0 refuted" as unconditional | sign reverses on recency; fresh-seed replication unmeasured | AUDIT_2026-09-10.md #3 |
| "Plain MapWM beats every published family-tree variant" | +0.077 vs MDE 0.111 at n=3 | N3_AUDIT.md; FAMILY_TREE_WM_GAP.md banner |
| "EM is worse remains dead" (VOCAB_EM.md) | recency EM_P0 - WM = -0.375 | VOCAB_EM.md strike; RECENCY_EM_RESULTS.md |
| VOCAB_SWEEP_RESULTS.md single-seed table; n_obs=4096 rows anywhere | superseded; degenerate always-blank | VOCAB_SWEEP_MULTISEED.md |
| Timing forward-only numbers as first published (26.9x, 478x) and pre-remeasure scaling (2.6-3.3x, 14.5x, 120.2x, 1632x) still in README.md / RESULTS_INDEX.md | autograd graph construction timed | TIMING_BENCHMARK.md re-measure |
| Hopfield-head / ExtraHead cross-scale "capacity" verdict (REPORT_v2 sec 3.4-3.6) | built on lm200 cross-scale runs | as cross-scale row above |
| NumberLine "self-correcting accumulator extrapolates" as a mechanism | mechanism not licensed after correction-line findings | L15_ABLATION.md, MQ_NOISE_2X2*.md (correction line) |
| DETAILED_RESULTS.md "MapFormer-WM (0.96) / EM (1.00) leave no headroom" framing and its headline table | mixes lm200 column; paper numbers misquoted | lm200 retraction; A4 |
| EM_WM_THEORY 2a v1 "shared kernel is better when offset fixed, EM wins" | demoted; axis is length | EM_WM_THEORY.md 2a; PAPERTASK_RESULTS.md |

## Cross-line dependencies

- **Environment line (torus vs MiniGrid; knob sweep; map-extent threshold).** A3's torus
  position effect (+0.461, n=8) is the torus half of BASELINE_TABLE's headline contrast. The
  knob-sweep baseline (+0.438) is a separate batch.
- **EM/WM kernel line.** A4/A5 are the only map-task EM > WM evidence, cited in
  EM_WM_THEORY.md 2a and EM_WM_STATE.md. A8 (q0/k0 sign on map tasks) is the counterpart to
  D5/recency. A9 relates to the sign-gauge finding. A11's EM_P0_r4 arm is the "healthy EM"
  used as a tie cell.
- **PoPE line.** MapPoPE-Flat as the best paper-task arm (A3-A5). The r=2 caveat links to
  MAPPOPE_R4_RESULTS.md.
- **Rank line.** A11 (r=4 stabilises EM at n_obs=256; Vanilla_r4 collapses one seed) and the
  "which fix binds is task-dependent" finding.
- **Match-Query line.** A6 compares against Match-Query's -0.842 ablation. A8's Match-Query EM
  rows are n=3 arms flagged in N3_AUDIT.md.
- **Hierarchy / compositional line.** A21's gate licenses the compositional task. A8's
  compositional EM rows.
- **Correction (Level 1.5) line.** Level15 rows in A12 (variance reduction), A15, A16, A20
  (NumberLine), and TEM references in A17.
- **Sign / clock-map line.** A7's retrace residual and A3's index-at-floor are the navigation
  anchor for "an index code cannot build a map".
- **Timing (A14)** is the cost evidence for non-commutativity (A12/A13) and for TEM,
  referenced in positional_review.pdf sec 7.

### Source disagreements recorded
1. **Blank floor at IID.** 0.506 (PAPER_TASK_ABLATION.md, 20x128 sequences) vs 0.522
   (PAPER_TASK_FLOORS.md, 8x32, the OOD-evaluator's event set). These are different event
   samples. Use 0.522 for A5's floor-normalised numbers and 0.506 where the A3/A6 files quote
   it.
2. **Paper-task 2x2 values.** RESULTS_INDEX.md headline quotes n=3 (e.g. RoPE 0.514,
   MapPoPE 1.000). The n=8 values (0.530, 0.994) supersede them.
3. **Timing.** README.md and RESULTS_INDEX.md quote pre-re-measure numbers (2.6-3.3x, 14.5x,
   120x, 1632x). TIMING_BENCHMARK.md's current fwd+bwd numbers are 2.5x/3.9x/2.7x, 14.4x,
   120.4x, 1624x.
4. **App. reference for q0/k0 separation.** PAPER_VALIDATION.md says A.4; CLAUDE.md
   (corrected 2026-09-11) says A.7, line 1527. Not re-checked against the PDF here.
5. **OOD-d Vanilla.** 0.367 (OOD_GRID_RESULTS.md, omega rescaled) vs 0.958 (A4) / 0.943 (A5),
   no rescaling. The protocol difference explains it (OMEGA_RESCALE_clean.md). Whether the
   paper rescales omega is unresolved.
6. **TEMFaithful noise T=512.** 0.709 +/- 0.002 (TEM_NOISE_FFN_RESULTS.md, n=3, fresh map) vs
   0.907 (NOISE_CLEAN_REVALIDATION.md, seed 0, env seed 0, 120 trials). The evaluations
   differ, and which map the latter used is not fully specified.
7. **RESULTS_INDEX.md lists TOPOLOGY_RESULTS.md, EM_HOPFIELD_CROSSSCALE.md,
   HOPFIELD_NOMAINAP_RESULTS.md, TEM_CROSSSCALE_DIAGNOSTIC.md, PERSCALE_OMEGA_RESULTS.md and
   MULTICLASS_MULTISEED_RESULTS.md as "other current".** Their run scripts or environment
   defaults use 200 landmarks, the same design as the archived void files. Excluded here as
   lm200. The report writer may want this confirmed against the retraction scope: the
   retraction names April checkpoints, while the archive also voids May lm200 runs.
8. **BASELINE_TABLE.md coverage gap** says LSTM/Mamba exist only in the lm200 column. A15's
   clean/noise rows contradict that, although they are not on the A3 paper-task batch.
9. **Family-tree "reproduces to three decimals"** (FAMILY_TREE_WM_GAP.md) is presented as an
   independent retrain. Under rule C27 it is most likely same-seed determinism.

## Files read

INVENTORY_BRIEF.md, CLAUDE.md (provided), README.md (sections), RESULTS_INDEX.md, N3_AUDIT.md,
KNOWN_BUGS.md, archive/void/README.md, archive/void/ listing and banners of
TEM_NOVEL_ENV_RESULTS.md, MULTISIZE_RESULTS.md, MULTIENV_CLEAN_2x2.md, TEM_SCALING_RESULTS.md,
PAPER_VALIDATION.md, PAPER_TASK_ACCURACY.md, PAPER_TASK_ABLATION.md, PAPER_TASK_FLOORS.md,
PAPER_OOD_PROTOCOL.md, PAPER_OOD_EXTENDED.md, PAPER_OOD_EXTENDED_n8.md, PAPER_OOD_WITH_POPE.md,
PAPER_OOD_RERUN.md, PAPERTASK_PREREG.md, PAPERTASK_RESULTS.md, EM_WM_THEORY.md,
EM_WM_STATE.md (section), AUDIT_2026-09-10.md (findings), INDEX_BASELINE_PAPER_TASK.md,
INDEX_BASELINE_PAPER_TASK_n8.md, BASELINE_TABLE.md, AUDIT_HEADLINE.md, TIMING_BENCHMARK.md,
FAMILY_TREE_GATES.md, FAMILY_TREE_RESULTS.md, FAMILY_TREE_D7_GATES.md,
FAMILY_TREE_D7_RESULTS.md, FAMILY_TREE_WM_GAP.md, ABLATE_FAMILY_TREE.md,
ABLATE_COMPOSITIONAL.md, NUMBERLINE_RESULTS.md, CAPACITY_PERREGIME.md (NumberLine section),
TOPOLOGY_RESULTS.md, OOD_GRID_RESULTS.md, OMEGA_RESCALE_clean.md (part), VOCAB_SWEEP_RESULTS.md,
VOCAB_SWEEP_MULTISEED.md, VOCAB_EM_PREREG.md, VOCAB_EM.md, EM_P0_PAPER.md, EM_P0_COMP.md,
EM_FIX_COMP.md, EM_COMP_SAMEBATCH.md, MATCH_QUERY_EM.md, AP_KERNEL_DIAGNOSTIC.md, TEM_RESULTS.md,
TEM_T_RESULTS.md, TEM_T_MULTISEED.md, TEM_NOISE_FFN_RESULTS.md, TEM_BACKGROUND_BASELINES.md,
TEM_CROSSSCALE_DIAGNOSTIC.md, EM_HOPFIELD_CROSSSCALE.md, HOPFIELD_NOMAINAP_RESULTS.md,
PERSCALE_OMEGA_RESULTS.md (part), MULTICLASS_MULTISEED_RESULTS.md, LONG_SEQ_clean.md,
PER_VISIT_clean.md, ZERO_SHOT_TRANSFER_clean.md, ZERO_SHOT_TRANSFER_clean_brokeninit.md,
REVISIT_DISTANCE.md, DETAILED_RESULTS.md (sections on framing, headline, models, parameter
budgets, empirical findings, Mamba, zero-shot, ablations), RESULTS_PAPER.md,
NOISE_CLEAN_REVALIDATION.md, REPORT.md (sections 1 header, 3.2, 5.4, 6.1-6.5), REPORT_v2.md
(sections 0-1, 6-9), REPORT_ADDENDUM.md (banner, sec 6), GENERALIZATION_REPORT.md,
RESULTS_SUMMARY_2026-05-10.md (Parts I-IV, IX-X), STITCH_ATTENTION.md. Code/scripts checked:
run_topology.sh, run_perscale_omega.sh, run_em_perscale_and_hopfield.sh,
run_hopfield_nomainap.sh, run_numberline.sh, environment_multiclass.py, train_multiclass.py,
run_paper_validation.sh, model_em_fixed.py, environment.py (GridWorld defaults).

## Files in scope not covered

None missing. Partially read (sections outside this line skipped): DETAILED_RESULTS.md
(InEKF/PC/hippocampal sections), REPORT.md and REPORT_v2.md (goal-directed, NoDrop/GSF,
behavioural sections), RESULTS_SUMMARY_2026-05-10.md Parts V-VIII (goal-directed, GSF, DoorKey
BC, DAgger), REPORT_ADDENDUM.md sections 1-5 (active inference, single-size, per-scale, SR aux,
cwd bug). All belong to other lines or are lm200/void.
