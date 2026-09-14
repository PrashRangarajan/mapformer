# Inventory F: EM vs WM and the position kernel (2026-09-08 .. 2026-09-12)

## Overview

The line asked how MapFormer's two variants combine the where with the what. MapEM scores
`softmax(A_X (*) A_P)` with one content-independent position kernel `A_P = q0^T R(dtheta) k0`
shared by every query-key pair (the Hadamard product is exactly TEM's `g (x) x` conjunction);
MapWM rotates content-derived Q,K, so its kernel has per-pair amplitudes and phases (MapWM is NOT
additive; `AUDIT_2026-09-10.md` #1). Starting from Whittington et al. (Neuron 2025, [11]), it found
the only detectable EM/WM accuracy gap on a fixed-budget comparison: varying-k recency, single-`p0`
EM - WM = -0.375 (0/8). What survived: (1) the gap is not expressivity (rewind installed and frozen
1.000, 8/8) and not holdability (installed at 8x scale with the content->Delta leak closed, 1.000,
8/8); (2) from scratch EM finds the rewind per query token, wrapped modulo each block's period, peak
or trough under the sign gauge; one shared k=64 is found on 7/8; (3) queries per token is the
currency at fixed budget (m4 - m64 +0.665, 8/8); (4) a near-perfect rewind is available in the
rank-4 subspace for failed tokens as for solved ones (T1), so failures are search; (5) phase freedom
in `k0` is real (+0.146 vs a matched-optimiser control, fresh seeds +0.113), mechanism unidentified;
(6) per-pair origins recover most of the gap at n=48 (total +0.215 = pathway +0.124 + freedom +0.091),
and only per-pair freedom abandons the per-token rewind route. On map tasks EM ties WM once its init
is fixed (MiniGrid, vocab), and degrades less with length on the paper task (floor-normalised +0.287 at
l=2048, 8/8) but that batch's pre-registered convergence gate failed, so it is formally not read.
Every recency-line contrast runs at r(loss, acc) = -0.936 to -0.986: these are statements about what
training finds and fits.

Common recency recipe (unless stated): `RecencyWorld`, `k_max=64`, `p_filler=0.5`, `min_gap=64`,
train T=1024, eval T in {1024, 2048}, 300 epochs cosine, lr 1e-3, 48 batches x 16, 1 layer,
d=128, 2 heads, no `--fast-attn`. Chance 0.0625; most-recent shortcut floor 0.0771; chance loss 2.77.
Gates `RECENCY_GATES_K64.md` pass. Held-out episodes. (Recency has no observation map; "transfer"
here means held-out symbol sequences.) Source: `REC_EM_PREREG.md`, `EM_WM_STATE.md` Sec 3.

---

### F1 Recency EM vs WM (the k-back batch)
- Dates: prereg 2026-09-09; results 2026-09-09, audit block 2026-09-10.
- Question: does [11]'s "EM learns faster except on N-back" exception appear in MapFormer; does EM match WM; does single `p0` beat separate `q0/k0` (fifth replication)?
- Task / environment: recency recipe above. Chance 0.0625, most-recent floor 0.0771.
- Arms: `Vanilla_r4` (WM, 222,233 params), `VanillaEM_P0_r4` (single p0, 222,361), `VanillaEM_r4` (paper-faithful separate q0/k0, 222,489).
- Seeds / batch: 8 per arm, one batch, recipe above, no `--fast-attn` on any arm.
- Validity gates: recency gates pass; within-batch only (published `Signed_r4` 1.000 used `--fast-attn` and is not mixed). Rule 9 not reported for this batch in the file; later batches on the same task give r ~ -0.98. Registered primary readout (epochs to loss threshold) computed only in the audit.
- Result (`RECENCY_EM_RESULTS.md`, `AUDIT_2026-09-10.md` #9):

  | arm | final loss | T=1024 | T=2048 | worst seed T=1024 |
  |---|---|---|---|---|
  | `Vanilla_r4` | 0.008-0.380 | 0.975 +/- 0.072 | 0.947 +/- 0.078 | 0.797 |
  | `VanillaEM_r4` | 0.430-1.136 | 0.837 +/- 0.081 | 0.728 +/- 0.106 | 0.689 |
  | `VanillaEM_P0_r4` | 0.924-1.885 | 0.600 +/- 0.126 | 0.510 +/- 0.111 | 0.365 |

  | contrast (T=1024) | delta | sd | MDE | seeds+ | verdict |
  |---|---|---|---|---|---|
  | EM_P0 - WM | -0.375 | 0.156 | 0.154 | 0/8 | DETECTABLE |
  | EM_sep - WM | -0.137 | 0.108 | 0.106 | 1/8 | DETECTABLE |
  | EM_P0 - EM_sep | -0.237 | 0.154 | 0.152 | 0/8 | DETECTABLE (n=8; superseded, see F7) |

  At T=2048 (registered length for P2/P3): -0.437 / -0.218. Registered primary: epochs to loss < 0.5: WM 8/8 (median 70.5), P0 0/8, sep 1/8 (epoch 232); < 0.1: WM 7/8 (median 118), P0 0/8, sep 0/8.
  Per-offset mean T=1024: WM 0.99/0.98/0.98/1.00/1.00/0.98/0.98/0.97 at k=1/2/4/8/16/32/48/64; sep 1.00/0.98/1.00/0.79/0.85/0.77/0.71/0.73; P0 0.98/1.00/0.89/0.76/0.60/0.79/0.72/0.37.
- Status: CITABLE (EM_P0 - WM -0.375 and EM_sep - WM -0.137). The sep-vs-P0 size is superseded by F7 (pooled n=24 +0.128; fresh seeds alone +0.073, unmeasured).
- Pre-registered? yes (`REC_EM_PREREG.md`). P1 (EM not faster than WM) CONFIRMED on the registered readout. P2 (|EM - WM| < 0.050) REFUTED; the audit reads it as learnability, not representation. P3 (P0 > sep) REFUTED with sign inverted.
- Caveats: WM here 0.975 not at ceiling (seed 7, loss 0.380). "EM does not learn the task at all" (file text) is overstated: P0 0.600 vs chance 0.0625. The file's WM loss range "0.008-0.044" in its prose is wrong (0.008-0.380). The file's retrodictive coherence account (THEORY_KERNEL Thm 2 corollary) was later refuted (F5). The varying-k task is NOT [11]'s N-back (which is fixed N with no filler, `tale_two_algorithms.txt:1672-1674`); fixed k is (F13).
- Sources: `REC_EM_PREREG.md`, `RECENCY_EM_RESULTS.md`, `AUDIT_2026-09-10.md`.
- Bears on: EM vs WM (the only detectable fixed-budget accuracy gap); learnability vs expressivity.

### F2 Kernel frame and existence construction (theory)
- Dates: `THEORY_KERNEL.md` 2026-09-09 (audited 2026-09-10/11); construction in `AUDIT_2026-09-10.md` 2026-09-10.
- Question: what object does attention read, and can single-`p0` EM's kernel represent recency at all?
- Task / environment: algebra; construction evaluated on recency episodes, 3 draws.
- Arms: construction: symbols `Delta=(1,0)`, query `q_k` `Delta=(-(k-1),0)`, MASK `(0,1)`, filler 0; single-`p0` kernel (rho = 1) with the real omega schedule, random magnitudes and projections; argmax of kappa over symbol keys with an idealised content gate.
- Seeds / batch: 3 construction draws; no training.
- Validity gates: n/a (construction). Without the rewind the same kernel scores 0.0857 (most-recent floor 0.077).
- Result: frame `A_P[t,s] = sum_i a_i cos(omega_i (S_t - S_s) + phi_i) =: kappa(dS)`, three axes (argument / kernel / composition). Construction selects the answer **1423/1423, 1417/1417, 1415/1415**. Theorem status after audit (`EM_WM_STATE.md` Sec 2): frame is exact algebra for EM, holds for WM only with per-pair amplitudes and phases; Thm 1 (cancellation exclusive) scoped to a scalar or fully constrained accumulator; Thm 2 algebra exact, init sd of rho measured 0.147 vs predicted 0.160 (16 head-values; mean -0.059); Thm 3 and its corollary WITHDRAWN.
- Status: CITABLE as algebra and as an existence construction at kernel level (upgraded to full model by F9).
- Pre-registered? no (theory is an explicit retrodiction of F1; stated in file).
- Caveats: kernel-level only, idealised content gate. Coherence sd shorthand "~1/sqrt(2 n_b)" is off by 4/pi; exact expression 0.159. The +0.123/+0.195 labelled "signed vs monotone" in THEORY_KERNEL are signed vs INDEX (signed vs monotone is -0.215/-0.280 loss-matched).
- Sources: `THEORY_KERNEL.md`, `AUDIT_2026-09-10.md` #1, #2, stale list.
- Bears on: EM vs WM (shared vs per-pair kernel); "the retrieval offset is chosen by the model, not the task"; sign axis (Thm 1 scoping).

### F3 N4: does rho at init predict a seed's fate?
- Dates: 2026-09-09 (in `THEORY_KERNEL.md` Sec 8).
- Question: correlation of initial kernel coherence rho with final loss.
- Task / environment: probe (`probe_ap_coherence.py`) on existing checkpoints: `runs/recency_em` (recency) and `runs/minigrid_em_fix` (MiniGrid).
- Arms: separate-q0/k0 EM checkpoints (MiniGrid probe ran on `VanillaEM_r4`, audit note).
- Seeds / batch: 8 + 8.
- Validity gates: none needed beyond probe; MiniGrid outcome range 0.307-0.357 (almost no dynamic range).
- Result: r(rho_init, final loss) = +0.142 (recency, loss range 0.430-1.136), +0.029 (MiniGrid). Unregistered: rho does not converge toward 1 in training; mean per-seed absolute change in head-averaged rho 0.2495 / 0.2663 (signed shifts -0.048 / -0.172); recency -0.059 -> -0.107, MiniGrid -0.055 -> -0.227.
- Status: EXPLORATORY (correlational, n=8; recency row a null-shaped correlation, MiniGrid unmeasured by range).
- Pre-registered? stated as prediction N4 in THEORY_KERNEL; falsifier fired ("not established").
- Caveats: MiniGrid probe ran on the r=4 separate arm (gap 0.098), not the r=2 arm with gap 0.137 quoted beside it.
- Sources: `THEORY_KERNEL.md` Sec 8, `AUDIT_2026-09-10.md` (Overruled section, stale list).
- Bears on: whether kernel coherence is causal (motivates N5).

### F4 Phase-spread probe (per-pair kernel phase)
- Dates: 2026-09-11 (`EM_WM_THEORY.md` v2, after three probe bug fixes).
- Question: does WM actually vary its kernel phase across pairs, and EM not?
- Task / environment: stored recency checkpoints, 29,056 pairs per model, eval-only (`probe_phase_spread.py`, `_PHASE_SPREAD.json`).
- Arms: WM trained (`Vanilla_r4`), WM untrained (3 inits), EM single p0, EM separate q0/k0.
- Seeds / batch: stored runs; WM trained seeds span 1.83-2.08.
- Validity gates: finite-sample uniform null computed (3.267); untrained WM control.
- Result: circular sd of phase across pairs: WM trained 1.947 (amplitude-weighted 1.816; per-offset mean 1.656; 64/64 live blocks); WM untrained 2.722 (2.603; 2.032); EM p0 0.000 (25/64 live blocks, i.e. 39 of 64 dead, amplitudes to 5e-86); EM sep 0.000 (phases non-zero, mean 1.603, identical every pair; 64/64 live); null 3.267.
- Status: CITABLE as description (EM 0.000 is algebraic; WM can reshape per pair). NOT evidence that WM's advantage comes from reshaping (training reduces spread).
- Pre-registered? no.
- Caveats: v1 numbers (2.003) VOID from three probe bugs. PAIRORIGIN's rerun gives WM 1.966 and null 3.274 on its batch.
- Sources: `EM_WM_THEORY.md` 1b and "What v1 got wrong".
- Bears on: shared vs per-pair kernel; input to F17-F19 and T1 (pruning).

### F5 N5: setting kernel coherence by construction (frozen, magnitude-matched)
- Dates: prereg 2026-09-09; results 2026-09-10.
- Question: does coherence rho causally matter, and does its effect invert between a map task and a clock task?
- Task / environment: (a) torus paper task, 300 ep, 98 x 128, T=128 train, eval 128/512/1024, lr 1e-3 cosine (readout T=1024; T=128 at ceiling); (b) recency recipe. Held-out map on torus.
- Arms: `EMPhase_plus_r4` (phi=0, rho=+1), `EMPhase_zero_r4` (phi=pi/2, rho=0), `EMPhase_minus_r4` (phi=pi, rho=-1), `EMPhase_rand_r4` (U(0,2pi)); k0 a per-block rotation of q0, both frozen; sum a_i equal to 6 d.p.
- Seeds / batch: 4 arms x 8 seeds per task, each task one batch.
- Validity gates: magnitude matching verified; rho held to 1e-6.
- Result (`N5_RESULTS.md`, `_N5_TORUS_RAW.md`): torus T=1024 plus 0.959 +/- 0.025, minus 0.943 +/- 0.029, zero 0.638 +/- 0.132, rand 0.679 +/- 0.101 (T=512: 0.993/0.990/0.814/0.830). Recency T=1024: plus 0.309 +/- 0.152, minus 0.350 +/- 0.246, zero 0.118 +/- 0.020, rand 0.182 +/- 0.110.
  - torus plus - minus +0.016 (sd 0.034, MDE 0.034, 6/8) unmeasured [gauge, expectation zero].
  - torus |rho|=1 - |rho|=0 **+0.292** (sd 0.064, MDE 0.063, 8/8) DETECTABLE.
  - torus plus - zero +0.321 (MDE 0.110, 8/8); minus - zero +0.304 (MDE 0.117, 8/8).
  - recency |rho| +0.180 (MDE 0.184, 7/8) unmeasured; recency plus - zero **+0.191** (7/8, MDE 0.158) DETECTABLE (omitted from results file, reported by audit).
  - interaction (|rho| form) +0.113 (MDE 0.195) unmeasured; (plus-minus form) +0.058 (MDE 0.182).
  - P4 zero - rand: torus -0.040 (MDE 0.170), recency -0.064 (MDE 0.105), both unmeasured.
- Status: |rho| torus effect CITABLE but scoped: established only for a frozen kernel at peak amplitude ~0.003 (learned kernels 0.06-0.08 torus, 0.27-0.50 recency). Inversion corollary: refuted-direction but interaction unmeasured (not a powered negative).
- Pre-registered? yes (`N5_PREREG.md`). P1 (plus > minus on torus) REFUTED by gauge; P2 (interaction) NOT SUPPORTED; P3 survives but degenerate (recency frozen arms barely left chance by arm means 2.02-2.64); P4 unmeasured. The |rho| axis was chosen after the data.
- Caveats: P1/P2 had expectation exactly zero ((W_q,b_q) -> (-W_q,-b_q) maps minus onto plus). "Freezing is catastrophic on recency" confounded with amplitude. "3 of 8 seeds start with every head at rho<0" line withdrawn. Per run, one minus seed reached loss 0.789 / acc 0.776.
- Sources: `N5_PREREG.md`, `N5_RESULTS.md`, `_N5_TORUS_RAW.md`, `AUDIT_2026-09-10.md` #4, #5.
- Bears on: kernel shape (axis B); sign gauge.

### F6 DOF: phase degrees of freedom, n=8 (torus half still current)
- Dates: prereg and results 2026-09-10.
- Question: at matched initial coherence (rho=1), does phase freedom help on the clock task and cost on the map task?
- Task / environment: recency recipe; torus paper task 300 ep, 98x128, T=128 train, eval {128,512,1024}, readout T=1024.
- Arms: `VanillaEM_P0_r4` (0 phase DOF, 222,361), `VanillaEM_r4` (sep, random rho, 222,489), `EMDoF_alignfree` (rho=1 at init, n_b phase DOF, 222,489), `EMDoF_alignlock` (k0_i = s_i q0_i, 0 phase DOF, free per-block scale, 222,425).
- Seeds / batch: 4 x 8 x 2 tasks, each task one batch; P0 and sep retrained in-batch.
- Validity gates: pre-launch check AlignLock holds rho 1.000000 while AlignFree drifts to 0.21 (40 steps lr 1e-2). Rule 9: recency r(loss,acc) = -0.985 (acc = 1.104 - 0.373*loss); torus r = -0.160 (all arms 0.00018-0.00028 train loss).
- Result (`DOF_RESULTS.md`, `_DOF_TORUS_RAW.md`):
  - Recency (n=8; superseded by F7): AlignFree 0.868 +/- 0.079, sep 0.837, AlignLock 0.720, P0 0.600. D1 AlignFree - AlignLock +0.148 (MDE 0.089, 8/8); loss-matched -0.014 (MDE 0.031). D4 sep - P0 +0.237 (MDE 0.152, 8/8) -- determinism, not replication (16/16 bitwise-identical checkpoints). D5 AlignLock - P0 +0.120 (MDE 0.134) unmeasured.
  - Torus T=1024: AlignLock 0.963 +/- 0.022, P0 0.962 +/- 0.024, AlignFree 0.875 +/- 0.112, sep 0.809 +/- 0.128 (T=512: 0.994/0.993/0.979/0.961; T=128 all 1.000). D2 AlignFree - AlignLock -0.088 (sd 0.119, MDE 0.118, 1/8) unmeasured. sep - P0 **-0.154** (sd 0.131, MDE 0.130, 0/8) DETECTABLE. Final rho: AlignFree 0.363 +/- 0.198, sep 0.173 +/- 0.196; within freed arms r(rho_final, acc) = +0.158 (n=16).
  - D3 interaction +0.236 (se 0.053, MDE 0.148); +0.253 with n=24 recency term.
- Status: torus sep - P0 -0.154 CITABLE but n=8, not extended ("provisional" per `D5_RESULTS.md`, `EM_WM_STATE.md`). D2 DIRECTIONAL (against freedom). D3 not a valid mechanism test (see caveat). Recency half superseded by F7/F8.
- Pre-registered? yes (`DOF_PREREG.md`). D1 confirmed raw (n=8); D2 unmeasured as registered; D3 "detectable" but invalid; D4 "reproduced" = determinism; D5 unmeasured.
- Caveats: D3 subtracts OOD torus accuracy (8x train length) from in-distribution recency accuracy and clears MDE only via an unmeasured opposite-sign term (audit #6). The n=8 additive decomposition (+0.148 / +0.120 / -0.031) is WITHDRAWN. "Final losses order monotonically in freedom" false at n=24. AlignLock vs AlignFree also differ in optimiser dynamics (audit #8; resolved by F8).
- Sources: `DOF_PREREG.md`, `DOF_RESULTS.md`, `_DOF_TORUS_RAW.md`, `AUDIT_2026-09-10.md` #6, #8, #10.
- Bears on: map side: freedom off the matched filter costs at OOD length; EM init (q0/k0) task dependence.

### F7 D5: DOF recency extended to n=24
- Dates: prereg and results 2026-09-10.
- Question: is magnitude freedom worth anything; do the n=8 sizes survive fresh seeds?
- Task / environment: recency recipe; `runs/dof/recency` extended with seeds 8-23.
- Arms: AlignFree, sep, P0, AlignLock (as F6).
- Seeds / batch: 4 x 24 (seeds 0-7 from F6, 8-23 new batch; pooling of identical code).
- Validity gates: AlignLock scales never negative over 512 block-scales (so it is pure magnitude freedom). Rule 9: r(loss,acc) = -0.986 over 96 runs.
- Result (`D5_RESULTS.md`, audit block):

  | arm | final loss | acc T=1024 |
  |---|---|---|
  | AlignFree | 0.746 +/- 0.362 | 0.818 +/- 0.127 |
  | sep | 0.775 +/- 0.299 | 0.814 +/- 0.106 |
  | P0 | 1.143 +/- 0.320 | 0.687 +/- 0.127 |
  | AlignLock | 1.239 +/- 0.388 | 0.654 +/- 0.141 |

  | contrast n=24 | delta | sd | MDE | seeds+ | verdict | fresh seeds 8-23 alone |
  |---|---|---|---|---|---|---|
  | sep - P0 (total) | +0.128 | 0.190 | 0.108 | 17/24 | DETECTABLE | +0.073 (9/16, MDE 0.130) unmeasured |
  | AlignFree - AlignLock (phase) | **+0.165** | 0.155 | 0.088 | 21/24 | DETECTABLE | **+0.173 (13/16, MDE 0.127)** DETECTABLE |
  | AlignLock - P0 (magnitude) | -0.033 | 0.203 | 0.116 | 9/24 | unmeasured | acc -0.109 (3/16) |
  | sep - AlignFree (init coherence) | -0.004 | 0.133 | 0.076 | 10/24 | unmeasured | -- |

  Loss-matched residuals: AlignLock - P0 +0.001 (MDE 0.021); AlignFree - AlignLock -0.009 (MDE 0.015). Loss AlignLock - P0: n=8 -0.292, fresh +0.290 (13/16), pooled +0.096 (MDE 0.310).
- Status: phase freedom CITABLE (replicates on fresh seeds), as an effect on fit. sep - P0 DIRECTIONAL (pooled detectable, carried by seeds 0-7; fresh seeds unmeasured). Magnitude and initial coherence unmeasured here (magnitude re-tested properly in F8).
- Pre-registered? yes (`D5_PREREG.md`). E1, E2 REFUTED (sign flips on fresh seeds; E4 fired); E3 CONFIRMED (loss-matched zero); E4 fired as falsifier; E5 held.
- Caveats: pooling licence cited in prereg (same-seed "reproduction") was void, pooling itself fine. "Magnitude freedom buys nothing" here was a parameter the optimiser barely moved (weight decay predicts s -> 0.674; measured median ~0.70); loss-matching conditions on a mediator (audit #7), so the "optimisation" label rests on existence (F9), not rule 9. Scale mean 0.794 is seeds 0-7; n=24 gives 0.786.
- Sources: `D5_PREREG.md`, `D5_RESULTS.md`, `AUDIT_2026-09-10.md` #3, #7, #8.
- Bears on: EM q0/k0 parameterisation; effect sizes shrink on fresh seeds.

### F8 MagOnly: phase freedom vs a matched-optimiser control
- Dates: prereg and results 2026-09-10.
- Question: is D1 phase freedom, or AlignLock's parameterisation (scale init 1.0 barely moves under Adam)?
- Task / environment: recency recipe.
- Arms: new `EMDoF_magonly` (k0 stored as raw vector u init = q0, k0_i = |u_i| q0_i/|q0_i|; 222,489 params, same function as AlignFree at init, logits agree to 2.4e-7, rho=1) vs stored AlignFree, AlignLock, P0.
- Seeds / batch: 24 new runs (seeds 0-23) + 2 in-batch determinism re-trains (AlignFree s0, P0 s0), bitwise identical to stored (weights and loss curves), licensing reuse.
- Validity gates: determinism PASS; manipulation check M4: MagOnly magnitudes move ~59% in 200 steps vs 1% from weight decay. Rule 9 r(loss,acc) = -0.978 over 96 runs.
- Result (`MAGONLY_RESULTS.md`): MagOnly final loss 1.156 +/- 0.243, acc 0.672 +/- 0.094 (n=24).
  - M1 AlignFree - MagOnly **+0.146** (sd 0.150, MDE 0.086, 22/24) DETECTABLE; seeds 0-7 +0.213 (8/8, MDE 0.110); **fresh seeds 8-23 +0.113 (14/16, MDE 0.111, clears by 0.002)**.
  - M2 MagOnly - AlignLock +0.018 (MDE 0.089, 13/24) unmeasured: confound ~a tenth of D1.
  - M3 MagOnly - P0 -0.015 (MDE 0.083, 12/24) unmeasured.
  - AlignFree fits better than MagOnly on 23/24 seeds (loss -0.410, MDE 0.233).
- Status: phase freedom CITABLE (+0.146 pooled, +0.113 fresh). Mechanism unidentified. Magnitude freedom unmeasured (MDE 0.083 is below the retracted n=8 +0.120 estimate, so against that specific estimate it is a POWERED NEGATIVE; the file calls it null).
- Pre-registered? yes (`MAGONLY_PREREG.md`). M1 CONFIRMED; M2 as expected; M3 exploratory null-shaped; M4 passed; M5 replication sign holds; M6 as registered.
- Caveats: fresh-seed effect half of seeds 0-7 and clears MDE by 0.002. On fit (r ~ -0.98). Post-hoc linear rewind probe in this file ("no from-scratch arm learns a rewind", 0/40) is WITHDRAWN by F14.
- Sources: `MAGONLY_PREREG.md`, `MAGONLY_RESULTS.md`, `_MAGONLY.json`.
- Bears on: EM kernel phase freedom; rule 31 (parameterisation is optimiser).

### F9 WARM: existence at full-model level (installed rewind, frozen vs trainable)
- Dates: prereg 2026-09-10; results 2026-09-10, corrected 2026-09-11.
- Question: is EM's recency deficit a failure to find a representable solution?
- Task / environment: recency recipe.
- Arms: `EMWarm_freeze` (single-p0 EM, position pathway installed with the rewind code in two latent embedding coordinates, w_in/w_out, omega, p0 at learned norm 1.957; frozen; content branch random; w_out scale x16 chosen by margin sweep before training), `EMWarm_train` (same install, all trainable). Comparators: stored `VanillaEM_P0_r4` s0-7, `Vanilla_r4` s0-7.
- Seeds / batch: 2 x 8 (seeds 0-7); comparators deterministic stored runs (determinism re-verified by F8).
- Validity gates: installed A_P selects the answer on every query, 8 construction seeds x T=1024/2048 (1.000); freeze holds bitwise under AdamW; not a rule-9 question.
- Result (`WARM_RESULTS.md`):

  | arm | T=1024 | T=2048 | final loss | worst |
  |---|---|---|---|---|
  | EMWarm_freeze | **1.000 +/- 0.000** | **1.000 +/- 0.000** | 0.007 | 1.000 |
  | Vanilla_r4 | 0.975 +/- 0.072 | -- | 0.064 | 0.797 |
  | EMWarm_train | 0.642 +/- 0.266 | 0.582 +/- 0.249 | 1.216 | 0.172 |
  | VanillaEM_P0_r4 | 0.600 +/- 0.126 | -- | 1.340 | 0.365 |

  W2 freeze - P0 **+0.400** (sd 0.126, MDE 0.125, 8/8) DETECTABLE. W3 freeze - WM +0.025 (MDE 0.071, 1/8) unmeasured. W4 train - freeze -0.358 (MDE 0.263, 0/8) detectable. Frozen per-offset 1.00 at every k 1-64. Trainable twin: pooled slope -0.055, kernel picks answer on 15% of queries.
- Status: W1/W2 CITABLE (EM represents recency exactly, 8/8, both lengths). W4 number stands; its interpretation WITHDRAWN (F10).
- Pre-registered? yes (`WARM_PREREG.md`). W1 CONFIRMED; W2 CONFIRMED; W3 unmeasured (matches WM); W4 falsifier fired, reading later withdrawn.
- Caveats: deviation from plan (only position pathway installed; content learned). Filler keys and the query's own key tie the answer on A_P; content must learn a token-type gate. "Weight decay excluded" (factor 0.992 over warmup). Early-window hypothesis in this file refuted (F10). "0 of 40" clauses withdrawn.
- Sources: `WARM_PREREG.md`, `WARM_RESULTS.md`, `AUDIT_2026-09-10.md` Tier 1.
- Bears on: expressivity vs learnability; EM's factorised where/what can represent the clock-like task.

### F10 UNFREEZE: early window vs install scale
- Dates: prereg 2026-09-10; results 2026-09-11 (with same-day correction block).
- Question: does the rewind survive release after content has trained, and was W4 an artefact of install scale?
- Task / environment: recency recipe; rewind state recorded every epoch into checkpoints.
- Arms: `EMUnf_0` (release step 0, install 1/64), `EMUnf_5` (epoch 5), `EMUnf_30` (epoch 30, peak lr), `EMUnf_100` (epoch 100), `EMUnf_0_e8` (step 0, install 1/8 = 8x; same Delta via w_in gain). Warmup 720 steps.
- Seeds / batch: 5 x 8 (seeds 0-7), one batch.
- Validity gates: U1 `EMUnf_0` s0 bitwise identical to stored `EMWarm_train` s0 (27 tensors + loss curve): recording inert.
- Result (`UNFREEZE_RESULTS.md`):

  | arm | T=1024 | T=2048 | >=0.95 | eff. slope | coord-0 slope | leak |
  |---|---|---|---|---|---|---|
  | EMUnf_0 | 0.642 +/- 0.266 | 0.582 | 0/8 | -0.054 | -0.095 | 0.421 |
  | EMUnf_5 | 0.609 +/- 0.142 | 0.575 | 0/8 | +0.013 | -0.229 | 0.393 |
  | EMUnf_30 | 0.835 +/- 0.282 | 0.758 | 3/8 | -0.143 | -0.445 | 0.375 |
  | EMUnf_100 | 0.605 +/- 0.370 | 0.531 | 1/8 | -0.125 | -0.203 | 0.308 |
  | EMUnf_0_e8 | **0.941 +/- 0.063** | 0.895 | 4/8 | -0.604 | -0.989 | 0.726 |

  Correction block (final weights, full latent pathway): EMUnf_0 eff -0.055 / pathway -0.074 / coord0 -0.095; EMUnf_0_e8 -0.602 / **-0.866** (seeds -1.10..-0.43) / -0.989; EMUnf_30 -0.142 / -0.192 / -0.445.
  U3 e8 - e64 **+0.298** (sd 0.272, MDE 0.270, 7/8) DETECTABLE, 84% of the gap to frozen. U2 paired contrasts vs EMUnf_0 unmeasured (MDE 0.33-0.44). Epoch effective slope first above -0.5: EMUnf_0 9-17; EMUnf_30 31-36, i.e. **1-6 epochs after release**; EMUnf_100 101-119, **1-19 after release**.
- Status: U3 CITABLE. U2 (early window) REFUTED by trajectories (descriptive, every seed). U4 exhaustiveness WITHDRAWN.
- Pre-registered? yes (`UNFREEZE_PREREG.md`). U1 passed; U2 not supported (3/8, slope -0.143; hard falsifier <=0.75 did not fire, mean 0.835); U3 no direction registered, 8x install mostly holds; U4 descriptive, its "only two channels" claim false.
- Caveats: stated confound at 8x (content sees larger latent coordinates). "Erosion tracks the learning rate" measured on coordinate 0 only. Trajectory values are recorded at the start of the last epoch (differ from final-weight values by ~0.001).
- Sources: `UNFREEZE_PREREG.md`, `UNFREEZE_RESULTS.md`.
- Bears on: holdability; rule 33 (install at training scale).

### F11 NOLEAK: closing the content -> Delta leak
- Dates: prereg and results 2026-09-11.
- Question: is content leaking into Delta through w_in the residual at 8x?
- Task / environment: recency recipe.
- Arms: 2x2 {install 1/64, 1/8} x {leak open = stored `EMUnf_0`, `EMUnf_0_e8`; leak closed = new `EMNoLeak_e64`, `EMNoLeak_e8` (w_in content columns held at zero, gradient-masked)}.
- Seeds / batch: 2 new arms x 8 (seeds 0-7) + determinism re-check.
- Validity gates: `EMUnf_0_e8` s0 retrain bitwise identical (30 tensors + loss curve). Manipulation check: max |traj_leak| = 0 and max |traj_slope - traj_latpath| = 0 at every epoch in both closed arms. Analysis script committed before results.
- Result (`NOLEAK_RESULTS.md`):

  | cell | T=1024 | T=2048 | >=0.95 | final loss | eff. slope | latent pathway | leak |
  |---|---|---|---|---|---|---|---|
  | 1/64 open | 0.642 +/- 0.266 | 0.582 | 0/8 | 1.216 | -0.055 | -0.074 | 0.421 |
  | 1/64 closed | 0.784 +/- 0.281 | 0.709 | 3/8 | 0.832 | -0.008 | -0.008 | 0 |
  | 8x open | 0.941 +/- 0.063 | 0.895 | 4/8 | 0.247 | -0.602 | -0.866 | 0.728 |
  | **8x closed** | **1.000 +/- 0.000** | **0.991** | **8/8** | 0.017 | -0.859 | -0.859 | 0 |

  Paired 8x closed - open +0.059 (MDE 0.063, 6/8) unmeasured (ceiling, anticipated; seed count named as readout: 8/8 vs 4/8). L1b latent pathway +0.007 (MDE 0.309) unmeasured. L3: scale effect leak closed +0.216 (MDE 0.278, 7/8) unmeasured; interaction -0.082 (MDE 0.387); closing leak at 1/64 +0.142 (MDE 0.374).
- Status: CITABLE on the registered seed-count readout (8/8 at 1.000 vs 4/8); paired accuracy delta unmeasured.
- Pre-registered? yes (`NOLEAK_PREREG.md`, with a pre-launch revision recorded before any checkpoint). L1 split: accuracy half met, slope half not (-0.859 vs <= -0.95). L1b: pathway unchanged. L2 split: slope half met (-0.008, rewind erased at 1/64 with zero leak), accuracy half not (0.784 > 0.75).
- Caveats: "holds" is one task, one config, n=8, with w_in content columns pinned (a point in EM's weight space the unconstrained optimiser drifts from). A slope near -0.86 with 1.000 accuracy means the slope statistic is not a sufficient summary; do not read -0.85..-1 as degrees of failure.
- Sources: `NOLEAK_PREREG.md`, `NOLEAK_RESULTS.md`, `_NOLEAK.json`.
- Bears on: holdability; "EM's recency deficit is entirely a search problem".

### F12 SEARCH S1: retrieval anatomy from scratch (eval-only)
- Dates: prereg and results 2026-09-11.
- Question: do from-scratch EM arms find the rewind in wrapped form, and what does phase freedom do?
- Task / environment: held-out `RecencyWorld(seed=10000)`, 256 episodes per checkpoint at T=1024 (deviation: prereg said 128; 256 matches its stated ~28 queries per k). `em_forward` asserted to reproduce `model(x)` (tol 1e-3).
- Arms: stored `VanillaEM_P0_r4`, `EMDoF_alignlock`, `EMDoF_magonly`, `EMDoF_alignfree`, `VanillaEM_r4` (24 seeds each); WM for attention readouts.
- Seeds / batch: 5 x 24 stored checkpoints.
- Validity gates: manipulation check PASSES (every rho=1 head peaks at symbol distance n=0, 48/48 in P0, AlignLock, MagOnly).
- Result (`SEARCH_RESULTS.md`):
  - H-wrap, cells k>=8 (1368 per arm): solved (acc>=0.9) / fraction with sel_max - sel0 >= 0.5 / failed (acc<=0.3) / fraction with it / r(acc, sel-sel0): P0 595/0.459/433/0.000/+0.460; AlignLock 595/0.395/496/0.000/+0.440; MagOnly 600/0.425/456/0.000/+0.455; AlignFree 1009/0.445/283/0.000/+0.382; sep 958/0.522/274/0.000/+0.422.
  - Exploratory route table (head with most attention on answer), peak-rewind / trough-rewind / peak-static / trough-static: P0 0.427/0.541/0.000/0.032; AlignLock 0.371/0.576/0.000/0.045 (other 0.007); MagOnly 0.398/0.557/0.000/0.045; AlignFree 0.424/0.515/0.028/0.033; sep 0.501/0.432/0.035/0.031. A_X > 0 on peak routes 1.000, on trough routes 0.000 in every arm. 93-97% of solved large-k cells are query-token rewinds.
  - All 48 AlignFree heads peak off zero (n = 2..63); all 48 sep heads (n = 4..78). Route fraction AlignFree 0.445 vs MagOnly 0.425 (+0.020).
  - AlignFree - MagOnly by k bin (n=24): 1-16 +0.124 (MDE 0.084, 21/24) DETECTABLE; 17-32 +0.263 (MDE 0.110, 22/24) DETECTABLE; 33-48 +0.077 (MDE 0.133) unmeasured; 49-64 +0.130 (MDE 0.141) unmeasured.
  - Success by distance from kernel peak d (0-3/4-7/8-15/16-31/32-64): P0 0.972/0.740/0.458/0.352/0.470; MagOnly 0.944/0.833/0.547/0.333/0.463; AlignFree 0.812/0.742/0.770/0.753/0.722; sep 0.732/0.688/0.733/0.723/0.802.
- Status: CITABLE as description (failed cells 0.000 carry a rewind, 5 arms x 24 seeds). Route table and distance account EXPLORATORY (unregistered).
- Pre-registered? yes (`SEARCH_PREREG.md`). H-wrap PARTIAL (failed half met exactly; solved half 0.459 vs >= 0.70, because more than half of solved cells use the trough). H-phase route prediction REFUTED (+0.020 vs <= -0.15); manipulation check passed.
- Caveats: phase freedom's mechanism remains unidentified: moves peaks off zero, does not change route or shorten shift, raises per-token success at every distance.
- Sources: `SEARCH_PREREG.md`, `SEARCH_RESULTS.md`, `_ANATOMY.json`, `_ANATOMY_EXT.json`.
- Bears on: withdraws "0/40"; rule 34 (readouts must respect symmetries); sign gauge visible inside trained models.

### F13 SEARCH S2: gradient at initialisation and path ruggedness (eval-only)
- Dates: 2026-09-11.
- Question: is the per-token landscape multi-modal, and does the init gradient point to the rewind?
- Task / environment: P0, AlignFree, MagOnly constructed as `train_recency` does, seeds 0-7; loss on first 8 training batches; ruggedness also on trained P0 s0-7.
- Arms: as above (AlignFree and MagOnly identical to P0 in function at init by construction).
- Seeds / batch: 8.
- Validity gates: ruggedness counted with prominence >= 1% of kappa(0) (added after first run, before verdicts; the registered strict count read float noise: 15.5 "maxima" at k<=4 on flat paths).
- Result: position-pathway gradient 1.7e-4 to 4.3e-4 of content branch's (8/8); rms A_P 6-9e-4 vs rms A_X 0.31-0.34. Linear-rewind rate |t| < 2 on 7/8 seeds, negative on 4/8. cos(g, dJ) -0.042..+0.023. Peaks along straight path to rewind (k<=4 / 8-16 / 32 / 60-64): init 0/0/1/2; trained P0 0/2/8/15.5. Kernel at rewind > kernel now for k>=32: init 0.727, trained 1.000.
- Status: EXPLORATORY (two snapshots; "window" account untested).
- Pre-registered? yes. H-rugged (ii) MET; (i) NOT MET at init (median 2 vs >= 5), MET at trained P0 (15.5).
- Caveats: nothing times barrier appearance against gradient growth.
- Sources: `SEARCH_RESULTS.md` S2, `runs/search/S2_report.md` (not opened), `_INIT_GRAD.json` (not opened).
- Bears on: why search is hard (multiplicative saddle at init).

### F14 SEARCH S3: fixed k and k curriculum (training)
- Dates: 2026-09-11.
- Question: is a large constant rewind findable; does a k curriculum find the linear rewind?
- Task / environment: recency recipe; fixed-k gates (`--k-fixed 64`): o1 0.063, o3 0.079, marginal 0.080, most-recent 0.057, oracle 1.000; default stream MD5-identical to pre-edit environment.
- Arms: `P0_fix64` (VanillaEM_P0_r4, every k=64), `P0_fix16`, `WM_fix64` (Vanilla_r4, k=64, positive control), `P0_cur` (k from 1..2*2^(epoch//30), full 64 from epoch 150), `P0_repro` (default, s0).
- Seeds / batch: 4 arms x 8, one batch; curriculum comparator = stored P0 s0-7 (licensed by repro).
- Validity gates: `P0_repro` s0 bitwise identical to stored (25/25 tensors, loss curves).
- Result (`SEARCH_RESULTS.md` S3):

  | arm | T=1024 | >=0.9 | >=0.95 | T=2048 | final loss | epochs to loss<0.5 |
  |---|---|---|---|---|---|---|
  | P0_fix64 | 0.985 +/- 0.043 | 7/8 | 7/8 | 0.946 | 0.065 | 54 (8/8) |
  | P0_fix16 | 0.994 +/- 0.018 | 8/8 | 7/8 | 0.966 | 0.015 | 25 (8/8) |
  | WM_fix64 | 0.947 +/- 0.120 | 7/8 | 6/8 | 0.755 | 0.119 | 99 (7/8) |
  | P0_cur | 0.727 +/- 0.038 | 0/8 | 0/8 | 0.668 | 1.045 | 11 (8/8, not comparable) |

  P0_cur - P0 **+0.127** (sd 0.117, MDE 0.116, 7/8) DETECTABLE; by k bin 1-16 +0.300 (MDE 0.191, 8/8), 17-32 +0.301 (MDE 0.221, 6/8) detectable; 33-48 -0.080, 49-64 -0.091 unmeasured. Linear slopes -0.015..+0.009.
  Exploratory: EM - WM fixed k=64 T=1024 +0.038 (MDE 0.133, 3/8) unmeasured; T=2048 **+0.191** (sd 0.152, MDE 0.150, 7/8) detectable; P0_fix16 - P0_fix64 +0.009 (MDE 0.048).
  Mechanism: fix16 peak-rewind 8/8 (two seeds linear, -15.04/-14.80; six wrapped). fix64 peak-rewind 6/8, trough 1/8, none visible 1/8; linear ratios +0.32..+1.42 vs -63: 0/8 linear, 7/8 wrapped.
- Status: findability of one shared k CITABLE (7/8); curriculum +0.127 CITABLE; EM - WM at fixed k T=2048 EXPLORATORY (unregistered, labelled exploratory; T=1024 unmeasured).
- Pre-registered? yes. S3-P1 MET (WM 6/8 >= 0.95). S3-P2 REFUTED (fix64 7/8 solved, <= 2/8 predicted). S3-P3 MET (curriculum does not close the gap), gain detectable.
- Caveats: this fixed-k condition is the one matching [11]'s N-back; there EM learns faster than WM. One task, n=8.
- Sources: `SEARCH_PREREG.md`, `SEARCH_RESULTS.md`, `runs/search/S3_report.md` (not opened).
- Bears on: EM vs WM on [11]'s N-back analogue; search spread vs size.

### F15 SPREAD: number of offsets at fixed budget (ceiling-limited exposure half)
- Dates: prereg and results 2026-09-11.
- Question: is the search limited by queries per token or by number of query tokens?
- Task / environment: recency, k drawn from a set of size m; primary = mean acc over k in {4,16,64} at T=1024 (k=1 excluded because most-recent floor = 1/m + chance(1-1/m): 0.297 / 0.121 / 0.077 at m=4/16/64). Gates at 800 episodes (`RECENCY_GATES_K4SET.md`, `RECENCY_GATES_K16SET.md`): n-gram columns at chance; most-recent 0.2958 (m=4), 0.1123 (m=16).
- Arms: all `VanillaEM_P0_r4`: m4_e300 (~403k queries/token), m16_e300 (~101k), m16_e1200 (~403k), m64_e1200 (~101k), m64_e300 (~25k, stored P0 s0-7). Cosine stretched to each arm's budget.
- Seeds / batch: 4 new arms x 8, one batch + repro (bitwise identical to stored s0).
- Validity gates: determinism PASS; rule 9 r(loss,acc) = -0.945 over 40 runs (acc = 1.040 - 0.338*loss, resid sd 0.059).
- Result (`SPREAD_RESULTS.md`): primary m4_e300 1.000 +/- 0.000; m16_e1200 1.000 +/- 0.000; m16_e300 0.996 +/- 0.010; m64_e1200 0.928 +/- 0.131 (two seeds 0.724/0.707); m64_e300 0.578 +/- 0.123. Final losses 0.048/0.071/0.245/0.359/1.340.
  m4 - m64 (300 ep) **+0.422** (sd 0.123, MDE 0.122, 8/8) DETECTABLE; m16 - m64 +0.418 (MDE 0.124, 8/8) DETECTABLE; m4 - m16 +0.004 (MDE 0.010) unmeasured; exposure-matched m16_e1200 - m4_e300 +0.000 (MDE 0.000, ceiling, uninformative); m64_e1200 - m16_e300 -0.068 (MDE 0.132, 1/8) unmeasured.
- Status: fixed-budget gradient CITABLE (superseded in size by F16 at lower budget). Exposure half: uninformative. 4x budget 0.578 -> 0.928: EXPLORATORY (unregistered; no contrast statistic given).
- Pre-registered? yes (`SPREAD_PREREG.md`). S4-P2 MET. S4-P1 printed CONFIRMED but half-vacuous (ceiling). S4-P3 NOT MET: r(rewind fraction, primary) = +0.573 vs >= 0.70 (m4_e300 at 1.000 with rewind fractions 0.25-0.75).
- Caveats: WM at 1200 epochs never run, so -0.375 remains the fair shared-budget comparison. Budget = more steps and slower decay.
- Sources: `SPREAD_PREREG.md`, `SPREAD_RESULTS.md`, gate files.
- Bears on: per-token search account; step efficiency.

### F16 SPREAD2: exposure test off the ceiling
- Dates: prereg and results 2026-09-12.
- Question: at matched queries per token, does the number of offsets still matter?
- Task / environment: as F15, 60-epoch base budget; calibration (`runs/spread_cal/`, n=2): m4 at 20/40/80 ep = 0.605/0.697/0.984.
- Arms: all `VanillaEM_P0_r4`: m4_e60 (80,600 q/token), m16_e240 (80,600, exposure-matched), m16_e60 (20,200), m64_e60 (5,000).
- Seeds / batch: 4 x 8, one batch.
- Validity gates: design check m4_e60 in 0.60-0.95: PASSES (0.913). Rule 9 r(loss,acc) = -0.936 over 32 runs.
- Result (`SPREAD2_RESULTS.md`): primary m4_e60 0.913 +/- 0.110 (loss 0.306); m16_e240 0.957 +/- 0.098 (0.295); m16_e60 0.590 +/- 0.190 (1.161); m64_e60 0.248 +/- 0.175 (2.502).
  exposure-matched m16_e240 - m4_e60 +0.045 (sd 0.168, MDE 0.166, 5/8) unmeasured; m4 - m16 (60) +0.323 (MDE 0.241, 7/8) DETECTABLE; m16 - m64 +0.342 (MDE 0.249, 7/8) DETECTABLE; m4 - m64 **+0.665** (sd 0.215, MDE 0.213, 8/8) DETECTABLE.
- Status: fixed-budget m4 - m64 CITABLE. Exposure-matched contrast unmeasured (MDE 0.166); "queries per token is the currency" is DIRECTIONAL/consistent, not shown equal.
- Pre-registered? yes (`SPREAD2_PREREG.md`). S5-P1 CONFIRMED as registered (below MDE, both off ceiling). S5-P2 not confirmed. S5-P3 MET.
- Caveats: m16_e240 at 0.957 near band edge; queries per token and optimiser steps per token not separated (varied via budget); fit contrasts.
- Sources: `SPREAD2_PREREG.md`, `SPREAD2_RESULTS.md`.
- Bears on: per-token search currency; explains SPREAD's 4x-budget jump.

### F17 PAIRORIGIN: per-pair position origins for EM (n=8)
- Dates: prereg 2026-09-11; results 2026-09-12.
- Question: does per-pair kernel freedom, holding Hadamard composition, rank and depth fixed, recover EM's varying-k deficit?
- Task / environment: recency recipe.
- Arms: `EMPair_r4` (q^p_t = p0 + W^q_out W^q_in x_t, likewise k; W_out zero-init so exactly `VanillaEM_P0_r4` at step 0; +2,048 params, +0.92%; 224,409 params per PAIRCONST), `VanillaEM_P0_r4`, `Vanilla_r4`.
- Seeds / batch: 3 x 8, one batch, all trained fresh.
- Validity gates: same function at init max |logit diff| 0.000e+00; origin pathway moved 0.327-0.841 over 16 tensors; per-pair spread 0.000 -> 1.448. Rule 9 r(loss,acc) = -0.983 over 24 runs. Two probe failures caught (anatomy assert max diff 13.8; phase-spread probe first reported 0.000 for EMPair), both fixed.
- Result (`PAIRORIGIN_RESULTS.md`): WM 0.975 +/- 0.072 (T=2048 0.947, loss 0.064); EMPair 0.880 +/- 0.136 (0.784, 0.422); P0 0.600 +/- 0.126 (0.510, 1.340). EMPair - P0 +0.280 (sd 0.218, MDE 0.216, 7/8) detectable at n=8; at T=2048 +0.273 (MDE 0.231, 7/8); EMPair - WM -0.095 (MDE 0.135, 0/8) unmeasured. Mechanism: phase spread P0 0.000 / EMPair 1.448 (seeds 1.28-1.58) / WM 1.966 (null 3.274); solved cells via per-token rewind P0 0.962 / EMPair 0.185.
- Status: accuracy numbers SUPERSEDED by F19 (n=48). Mechanism readouts CITABLE (intervention, same function at init). EMPair - WM unmeasured (not "recovered").
- Pre-registered? yes (`PAIRORIGIN_PREREG.md`). P1 CONFIRMED at n=8; P2 "met" in registered sense (inside MDE, 0/8 positive); P3 not confirmed; P4 (rewind fraction lower) held.
- Caveats: capacity confound (resolved by F18/F19). Within-EM test; does not turn EM into WM. Does not bear on fixed-offset case. Does not show WM wins through per-pair freedom.
- Sources: `PAIRORIGIN_PREREG.md`, `PAIRORIGIN_RESULTS.md`, `_PAIRORIGIN.json`.
- Bears on: kernel sharing as the varying-offset cost; where/what composition.

### F18 PAIRCONST: capacity control (n=8)
- Dates: prereg and results 2026-09-12.
- Question: is PAIRORIGIN's gain per-pair freedom or 2,048 parameters?
- Task / environment: recency recipe.
- Arms: `EMPairConst_r4` (same pathway reading a learned constant; origins identical for every token; 224,537 params, 128 MORE than EMPair; same function as P0 at init, max |logit diff| 0.000e+00), vs stored EMPair_r4 and P0.
- Seeds / batch: 8 new + EMPair s0 determinism re-check (bitwise, 29/29 tensors).
- Validity gates: determinism PASS; origin spread across tokens 0.000 (EMPairConst) vs 0.117 (EMPair).
- Result (`PAIRCONST_RESULTS.md`): EMPairConst 0.782 +/- 0.159 (T=2048 0.659, loss 0.755). EMPair - EMPairConst +0.098 (sd 0.211, MDE 0.209, 7/8) unmeasured; EMPairConst - P0 +0.182 (sd 0.190, MDE 0.188, 6/8) unmeasured. Mechanism: phase spread P0 0.000 / EMPairConst 0.000 / EMPair 1.448; solved cells via per-token rewind 0.964 / **0.948** / **0.189**; solved cells (k>=8) 23.5 / 36.8 / 30.8.
- Status: mechanism attribution CITABLE (abandoning the rewind tracks per-pair freedom, not capacity; control with more parameters keeps the route). Accuracy split superseded by F19.
- Pre-registered? yes (`PAIRCONST_PREREG.md`). C1 not confirmed; C2 "met by the letter" (0.006 under MDE) but reported unmeasured; C3 not confirmed; unresolved at n=8.
- Caveats: mechanism readout is n=8 descriptive.
- Sources: `PAIRCONST_PREREG.md`, `PAIRCONST_RESULTS.md`.
- Bears on: per-pair freedom vs pathway capacity.

### F19 PAIRSPLIT: freedom vs pathway at n=48
- Dates: prereg and results 2026-09-12.
- Question: resolve the accuracy split.
- Task / environment: recency recipe.
- Arms: `EMPair_r4`, `EMPairConst_r4` seeds 8-47 new (seeds 0-7 reused); `VanillaEM_P0_r4` seeds 0-23 stored, later extended to 48.
- Seeds / batch: 80 new runs in one batch (+ P0 extension); seeds 0-7 reused under in-batch bitwise determinism re-check ("P0 extended, determinism bitwise").
- Validity gates: determinism re-check; fresh-seed replication split registered; rule 9 r(loss,acc) = -0.954 over 96 runs.
- Result (`PAIRSPLIT_RESULTS.md`):

  | term | n | delta | sd | MDE | seeds+ | verdict |
  |---|---|---|---|---|---|---|
  | freedom EMPair - EMPairConst | 48 | **+0.091** | 0.164 | 0.066 | 34/48 | DETECTABLE |
  | freedom, fresh seeds 8-47 alone | 40 | **+0.089** | -- | 0.069 | -- | DETECTABLE |
  | freedom, first 8 | 8 | +0.098 | -- | 0.209 | -- | unmeasured |
  | pathway EMPairConst - P0 | 24 | +0.100 | 0.197 | 0.113 | 14/24 | unmeasured |
  | total EMPair - P0 | 24 | +0.191 | 0.175 | 0.100 | 20/24 | DETECTABLE |
  | **total EMPair - P0** | 48 | **+0.215** | -- | 0.068 | 42/48 | DETECTABLE (100%) |
  | **pathway EMPairConst - P0** | 48 | **+0.124** | -- | 0.071 | 34/48 | DETECTABLE (58%) |
  | **freedom** | 48 | **+0.091** | -- | 0.066 | 34/48 | DETECTABLE (42%) |

- Status: CITABLE (n=48, freedom replicates on fresh seeds).
- Pre-registered? yes (`PAIRSPLIT_PREREG.md`). S1 CONFIRMED (0.05-0.15 band); S2, S3 not confirmed. C2 predicted detectable at n=24: not (+0.100, MDE 0.113); detectable at n=48.
- Caveats: fit contrasts. First eight seeds inflated the total (+0.280 -> +0.215) and pathway (+0.182 -> +0.124), not the freedom term. Quote n=48.
- Sources: `PAIRSPLIT_PREREG.md`, `PAIRSPLIT_RESULTS.md`.
- Bears on: shared vs per-pair kernel (varying-offset side only).

### F20 PAPERTASK rerun: EM vs WM at extended length on the paper task
- Dates: floors 2026-09-12; prereg and results 2026-09-12.
- Question: does EM's extended-length advantage on the paper task (from a 16-epoch LinearLR batch whose logs were deleted) survive a converged recipe?
- Task / environment: torus paper task, `train_variant --epochs 50 --schedule cosine --n-batches 98 --batch-size 128 --n-steps 128 --n-layers 1 --n-heads 2 --d-model 128 --n-landmarks 0`. Eval `eval_paper_ood --extended --n-batches 8 --batch-size 32`: IID l=128 g=64 pe=0.5; OOD-d l=64 g=32 pe=0.2; OOD-s l=256 and l=512 g=128 pe=0.8; ext-s l=1024, l=2048. Held-out map (env seed 10000). Measured floors (`PAPER_TASK_FLOORS.md`, always-blank = best constant): 0.522 / 0.216 / 0.803 / 0.799 / 0.801 / 0.802; 1/vocab 0.0476. Primary = floor-normalised (acc - floor)/(1 - floor) at l=2048.
- Arms: `Vanilla` (WM, r=2 default), `VanillaEM_P0`, `MapPoPE-Flat`.
- Seeds / batch: 3 x 8, one batch, all fresh, logs kept.
- Validity gates: registered convergence gate IID >= 0.99 FAILED (WM 0.968, EM 0.985; MapPoPE 1.000). Rule 9 r(final loss, acc) = -0.461 over 24 runs (acc = 0.942 - 0.152*loss, resid sd 0.036). Mean final loss WM 0.1306, EM 0.0762, MapPoPE 0.0168.
- Result: raw (`PAPER_OOD_RERUN.md`), IID / OOD-d / OOD-s256 / OOD-s512 / ext1024 / ext2048: Vanilla 0.968 +/- 0.051 / 0.943 +/- 0.076 / 0.984 +/- 0.020 / 0.964 +/- 0.029 / 0.927 +/- 0.034 / 0.886 +/- 0.035; VanillaEM_P0 0.985 +/- 0.023 / 0.980 +/- 0.031 / 0.988 +/- 0.012 / 0.978 +/- 0.015 / 0.964 +/- 0.016 / 0.942 +/- 0.024; MapPoPE-Flat 1.000 +/- 0.001 / 0.993 +/- 0.004 / 0.998 +/- 0.002 / 0.991 +/- 0.003 / 0.978 +/- 0.004 / 0.963 +/- 0.005.
  Floor-normalised (`PAPERTASK_RESULTS.md`): EM - WM l=512 +0.067 (MDE 0.168, 4/8) unmeasured; l=1024 **+0.186** (MDE 0.185, 8/8) detectable; l=2048 **+0.287** (MDE 0.194, 8/8) detectable. MapPoPE - EM l=512 +0.066 (MDE 0.079, 6/8); l=1024 +0.073 (MDE 0.090, 6/8); l=2048 +0.102 (MDE 0.128, 6/8), all unmeasured. 16-epoch values for comparison: EM - WM +0.174/+0.352/+0.430; MapPoPE - EM +0.070/+0.112/+0.159.
- Status: EXPLORATORY (detectable at l=1024/2048, 8/8, and not a loss gap by rule 9; but the pre-registered gate failed so P1 is formally NOT READ).
- Pre-registered? yes (`PAPERTASK_PREREG.md`). Gate failed -> P1 not read; P2 (rule 9) run: r = -0.461, loss-matched residual not reported; P3 held (MapPoPE - EM unmeasured); P4 did not fire.
- Caveats: gate probably mis-set (WM 0.969 at 16 ep vs 0.968 at 50 ep: systematic shortfall vs paper's 0.99). Effect is monotone in LENGTH, the project's universal unexplained OOD signature; other fixed-offset tasks tie or reverse (F21-F24), so it does not support "a shared kernel is better when the offset is fixed". WM arm is r=2 (Vanilla), EM arm has no rank suffix: rank as in code defaults. Raw 16-epoch n=8 batch (`PAPER_OOD_EXTENDED_n8.json`, EM_P0 - WM +0.035/+0.070/+0.085, MDE 0.031-0.034) is superseded; its logs and checkpoints are deleted.
- Sources: `PAPERTASK_PREREG.md`, `PAPERTASK_RESULTS.md`, `PAPER_OOD_RERUN.md`, `PAPER_TASK_FLOORS.md`, `EM_WM_THEORY.md` 2a.
- Bears on: EM vs WM on map tasks; OOD-length axis; MapPoPE as strongest paper-task arm (cross-line).

### F21 VOCAB_EM: capacity claim with a healthy EM arm
- Dates: prereg 2026-09-09; results 2026-09-09 (annotated 2026-09-11).
- Question: does EM's claimed quadratic capacity (paper Fig. 11) show as an EM - WM gain growing with vocabulary once the init pathology is removed; recipe vs init for the n_obs=256 collapse.
- Task / environment: torus paper task, held-out map, cosine / lr 1e-3 (NOT the published recipe); n_obs in {16, 64, 256} (4096 excluded: all arms at 0.50 blank floor); primary T=128 at n_obs=256.
- Arms: `Vanilla` (WM r2), `Vanilla_r4`, `VanillaEM`, `VanillaEM_P0`, `VanillaEM_P0_r4`.
- Seeds / batch: 5 x 3 x 8 = 120 runs, one batch.
- Validity gates: reading rule fixed in advance (per-seed min and worst-to-second gap before mean). No rule-9 or floor beyond the 0.50 blank note.
- Result (`VOCAB_EM.md`), n_obs=256 T=128, min / gap / sd / mean: Vanilla 0.990/0.004/0.003/0.997; Vanilla_r4 0.518/0.482/0.171/0.940; VanillaEM 0.796/0.031/0.036/0.849; VanillaEM_P0 0.504/0.225/0.171/0.849; **VanillaEM_P0_r4 0.995/0.005/0.002/0.999**. VanillaEM_P0_r4 mean 1.000 at n_obs 64 and 16.
  EM_P0_r4 - Vanilla_r4: -0.000 / +0.000 / +0.060 (n_obs 16/64/256); dropping each arm's worst seed: +0.0001 (64), **+0.0000** (256). MDE at n_obs=256 0.169; all contrasts unmeasured. VanillaEM_P0 n_obs=256 T=512 per seed 0.491 0.654 0.755 0.812 0.875 0.949 0.955 0.957.
- Status: EXPLORATORY/descriptive (every contrast unmeasured; profile rests on one collapsed WM seed). Direction: no EM capacity advantage.
- Pre-registered? yes (`VOCAB_EM_PREREG.md`). P1 REFUTED (gain is one collapsed Vanilla_r4 seed); P2 held for EM_P0_r4 (gap 0.005); P3: low seed persists under better recipe, so neither recipe nor shared p0 explains that collapse.
- Caveats: n_obs 16/64 at ceiling for r=4 arms. Not comparable to `VOCAB_SWEEP_MULTISEED.md`. Paper's scaling is at l=16 up to vocab 10,000. The file's "EM is worse remains dead" bullet was REFUTED next day by F1. A WM arm (Vanilla_r4) is the worst collapser here.
- Sources: `VOCAB_EM_PREREG.md`, `VOCAB_EM.md`, `VOCAB_EM.json`.
- Bears on: EM capacity claim; which init fix binds is task-dependent.

### F22 MINIGRID_EM: EM vs WM x rank on allocentric MiniGrid
- Dates: prereg and results 2026-09-08.
- Question: does path integration beat index once rotation is handled (allocentric) and does EM beat WM at matched rank?
- Task / environment: MiniGrid-DoorKey-16x16, egocentric obj_color observation, allocentric action recoding, 50 epochs, 25K cached buffer, no `--fast-attn`. Measured floor T=128 0.635, T=512 0.536, T=1024 0.490.
- Arms: `RoPE` (index, ~614K), `Vanilla` (WM r2, 614,538), `Vanilla_r4` (614,922), `VanillaEM` (r2, 614,794), `VanillaEM_r4` (615,178); spread < 0.11%.
- Seeds / batch: 5 x 8, one batch.
- Validity gates: floors measured; ceiling note (~0.18 headroom at T=1024). Rule 9 and convergence checks promised in prereg not reported in results.
- Result (`MINIGRID_EM.md`), T=512 / T=1024 / sd / min@1024: RoPE 0.819 / 0.788 / 0.019 / 0.754; Vanilla 0.817 / 0.778 / 0.045 / 0.693; **Vanilla_r4 0.843 / 0.822 / 0.015 / 0.798**; VanillaEM 0.797 / 0.763 / 0.081 / 0.572; VanillaEM_r4 0.826 / 0.803 / 0.056 / 0.666.
  Vanilla_r4 - RoPE **+0.034** (MDE 0.014, 8/8) DETECTABLE; Vanilla - RoPE -0.010 (MDE 0.056, 4/8); VanillaEM - RoPE -0.026 (MDE 0.087); VanillaEM_r4 - RoPE +0.015 (MDE 0.062, 7/8). EM - WM at r2 -0.020 (3/8) / -0.016 (4/8); at r4 -0.018 (2/8) / -0.019 (4/8), all unmeasured. Rank interaction -0.003 (MDE 0.128). Trimmed (drop each arm's worst): r2 -0.001, r4 -0.003. Worst-to-2nd gap: WM r4 0.007; EM r4 0.144 (0.666 -> 0.810); WM r2 0.031; EM r2 0.167.
- Status: Vanilla_r4 - RoPE CITABLE (cross-line: environment/rank). EM vs WM unmeasured; EM's deficit is one collapsed seed (descriptive).
- Pre-registered? yes (`MINIGRID_EM_PREREG.md`). P1 CONFIRMED only with r=4; P2 (EM >= WM) REFUTED direction, all unmeasured; P3 unmeasured; P4 (EM collapses rather than degrades) CONFIRMED descriptively.
- Caveats: the file's AND-gate/"no additive fallback" mechanism language rests on the withdrawn additive-WM premise; the collapse was later traced to separate q0/k0 (F23). One environment, flat models, n=8.
- Sources: `MINIGRID_EM_PREREG.md`, `MINIGRID_EM.md`.
- Bears on: EM vs WM on realistic navigation; allocentric + r=4 (cross-line).

### F23 MINIGRID_EM_FIX: the 2x2 of both EM init fixes
- Dates: prereg 2026-09-08; results 2026-09-09 (annotated 2026-09-11).
- Question: is EM's MiniGrid collapse the separate-q0/k0 origin or r=2?
- Task / environment: as F22 (floor T=1024 0.490).
- Arms: `Vanilla_r4` (reference), `VanillaEM`, `VanillaEM_r4`, `VanillaEM_P0`, `VanillaEM_P0_r4` (new); spread 0.08%.
- Seeds / batch: 5 x 8, one batch.
- Validity gates: reading rule (min and gap before mean) fixed in advance.
- Result (`MINIGRID_EM_FIX.md`), T=1024 min / 2nd / gap / sd: Vanilla_r4 0.798/0.814/0.016/0.014; VanillaEM 0.602/0.739/0.137/0.073; VanillaEM_r4 0.712/0.810/0.098/0.040; VanillaEM_P0 0.797/0.813/0.016/0.014; **VanillaEM_P0_r4 0.816/0.818/0.002/0.011**. EM_P0_r4 means 0.847 (T=512) / 0.830 (T=1024). EM_P0_r4 - EM_r4 +0.0139 / +0.0212 (MDE 0.044) unmeasured (predicted +0.019). Shared p0 buys +0.050 at r2, +0.021 at r4; interaction -0.029 (MDE 0.104). **EM_P0_r4 - Vanilla_r4 +0.0012 (4/8) / +0.0035 (6/8)**, unmeasured (no MDE stated).
- Status: DIRECTIONAL/descriptive (gap ordering 0.137/0.098/0.016/0.002 unambiguous as description; all contrasts unmeasured). EM ties WM once init is fixed.
- Pre-registered? yes (`MINIGRID_EM_FIX_PREREG.md`). P1 CONFIRMED (no collapsed seed, gap 0.002); P2 point estimate on prediction, unmeasured; P3 direction as predicted, unmeasured.
- Caveats: not a universal fix (on vocab n_obs=256 VanillaEM_P0 still collapses; there rank was needed, F21). "Separate q0/k0 refuted" holds on map tasks only; on recency the separate form is directionally better (F1, F7). App. A.4 citation is App. A.7 per CLAUDE.md correction.
- Sources: `MINIGRID_EM_FIX_PREREG.md`, `MINIGRID_EM_FIX.md`.
- Bears on: EM init pathology; EM = WM on map tasks.

### F24 Pre-line EM single-p0 context: paper task and Match-Query (n=3)
- Dates: `EM_P0_PAPER.md` 2026-08-09; `MATCH_QUERY_EM.md` 2026-08-15.
- Question: does single p0 (paper eq. 3) help over separate q0/k0?
- Task / environment: paper task (final losses only); Match-Query TE=512 TQ=256, 200 ep, held-out env seed 10000, chance 0.0625 (gated, `MATCH_QUERY_GATES.md`).
- Arms: Vanilla, VanillaEM, VanillaEM_P0 (paper); MapWM-Flat, MapEM sep, MapEM single p0 (Match-Query).
- Seeds / batch: n=3 each; Match-Query one batch.
- Validity gates: Match-Query WM control reproduces sweep 0.888 to three decimals.
- Result: paper-task final loss Vanilla 0.0700/0.1362/0.1379; VanillaEM 0.3908/1.0234/0.0832; VanillaEM_P0 0.1053/0.1508/0.1014 (`EM_P0_PAPER.md`). Match-Query TQ=256: WM 0.888 +/- 0.140, sep 0.450 +/- 0.332, p0 0.808 +/- 0.168; TQ=512: 0.902 / 0.385 / 0.789; p0 - sep per seed +0.629/+0.231/+0.214, mean +0.358, 3/3 (`MATCH_QUERY_EM.md`). Cited elsewhere: paper-task held-out +0.089 (VanillaEM 0.898 +/- 0.108 vs P0 0.987 +/- 0.012, `PAPER_TASK_ACCURACY.md`), compositional +0.167 (`EM_P0_COMP.md`, n=3).
- Status: EXPLORATORY (n=3; `N3_AUDIT.md` lists all MQ EM arms at n=3; MapWM-Flat fell 0.888 -> 0.730 when extended to n=5).
- Pre-registered? no.
- Caveats: "refuted on four tasks" is map-task-only (sign reverses on recency). The file's kernel-geometry falsification (`AP_KERNEL_DIAGNOSTIC.md`) is outside this line.
- Sources: `EM_P0_PAPER.md`, `MATCH_QUERY_EM.md`, `N3_AUDIT.md`; `PAPER_TASK_ACCURACY.md` and `EM_P0_COMP.md` grepped only.
- Bears on: EM init; motivates F1 P3.

### F25 T1: availability of rewinds per token (Diophantine framing, eval-only)
- Dates: theory and results 2026-09-12 (`THEORY_SEARCH_AND_LENGTH.md`, written before either test).
- Question: for tokens EM fails, does a good wrapped rewind exist in the model's own rank-4 subspace (existence) or is it not found (search)? Is frequency pruning the strategy?
- Task / environment: recency; `probe_achievable.py` on 8 stored P0 checkpoints; multi-start optimisation of z in R^4 against real episode samples (filler included); readout |A| under sign gauge.
- Arms: `VanillaEM_P0_r4` s0-7.
- Seeds / batch: 8 checkpoints; cells k>=8.
- Validity gates: none beyond probe; no inferential contrast.
- Result: SOLVED cells (acc >= 0.9, n=166): available Q 0.995, achieved |A| 0.544, |A| >= 0.5 0.590, trough 0.331, peak 0.259. FAILED (acc <= 0.3, n=181): Q 0.992, |A| 0.359, 0.276, 0.160, 0.116. r(accuracy, |A|) = +0.355 over 431 cells. P1a: r(dead-block fraction, tokens solved) = **-0.546** (n=8; dead fractions 0.41-0.77, tokens solved 5-29).
- Status: EXPLORATORY-descriptive (strong eval-only description, 347 cells, 8 checkpoints; not an inferential contrast). Core claim (failures are search, not existence) held; pruning limb refuted in direction (underpowered).
- Pre-registered? stated in-file before the test (no separate prereg). Core CONFIRMED; P1a REFUTED (sign inverted); P1b-P1d not run.
- Caveats: alternative reading (pruning broadens kernel, symptom of failure) untested. Successes sit far below available optimum.
- Sources: `THEORY_SEARCH_AND_LENGTH.md` T1, `_ACHIEVABLE.json` (not opened).
- Bears on: search vs existence at per-token grain.

### F26 T2: collisions and the OOD-length axis (eval-only)
- Dates: 2026-09-12.
- Question: is failure at length a pigeonhole collision on the position kernel, and does length act only through collisions?
- Task / environment: paper task; `probe_collisions.py` on `VanillaEM_P0` paper-task (rerun) checkpoints; N_coll = prior keys at a different cell ranked at least as high as the correct key by the model's own kernel; ~9k-20k scored events per length.
- Arms: `VanillaEM_P0` only (WM has no position-only score).
- Seeds / batch: 4 seeds x 4 lengths.
- Validity gates: no free parameters.
- Result: length 256/512/1024/2048: collision rate 0.058/0.085/0.132/0.174; accuracy 0.974/0.974/0.954/0.933; predicted from rate alone 0.962/0.957/0.948/0.941; acc | 0 coll 0.983/0.989/0.981/0.964; acc | >=1 coll 0.825/0.817/0.777/0.786. Pooled acc 0.973 (no collision) vs 0.787 (any). Dose bins: 1-2 0.743, 3-8 0.782, 9-32 0.796, 33-128 0.794, 129+ 0.797. Collision rate accounts for 53% of the 0.041 drop from 256 to 2048.
- Status: EXPLORATORY (association within one arm, 4 seeds).
- Pre-registered? stated in-file before the test. Collisions as dominant identified mechanism: CONFIRMED (as association). Pigeonhole dose form: REFUTED (a switch, not a dose). "Length acts only through collisions": REFUTED (53%). Cross-arm half (resolution delta, R(T) ordering Vanilla/EM/MapPoPE): UNTESTED.
- Caveats: one arm, one task; association not intervention.
- Sources: `THEORY_SEARCH_AND_LENGTH.md` T2, `_COLLISIONS.json` (keys checked only).
- Bears on: OOD-length axis (rank, InEKF, forget gate, PoPE, EM); cross-line.

### F27 Neuron 2025 "tale of two algorithms" comparison (literature)
- Dates: 2026-09-09, audited 2026-09-10; corrected reading in `EM_WM_THEORY.md` Sec 3 (2026-09-11) and `EM_WM_STATE.md` Sec 1.
- Question: what does [11] (Whittington, Dorrell, Behrens, Ganguli, El-Gaby, Neuron 113(2):321-333, 2025) predict for MapFormer?
- Task / environment: literature reading (`papers/txt/tale_two_algorithms.txt`).
- Arms: n/a.
- Result: [11] claims EM and WM solutions are equivalent once trained (same generalisation), learning dynamics differ, EM faster "except on N-back"; capacity advantage of EM at fixed RNN neurons from a separate synaptic memory network. Surviving argument: MapEM has no separate memory network (Hadamard = rank-one contraction of the product space), so [11]'s capacity result does not transfer and both MapFormers are WM models in [11]'s sense. [11]'s N-back is fixed N with no filler (l.1672-1674), matching the fixed-k condition (F14, where EM learns faster, 54 vs 99 epochs), not varying-k recency. [11] l.599 notes its discriminator degenerates on N-back; its learning-speed evidence (Fig. S2G) is not in the corpus. Measured ties: MiniGrid allocentric r=4 EM - WM +0.0035; vocab trimmed +0.0000. Untested: position decodable from EM vs activity slots from WM (Fig. 2G/2I; may be ill-posed); parallel planning.
- Status: PRIOR-ART comparison (argument; capacity-non-transfer stands).
- Pre-registered? no.
- Caveats: file's headline is STALE (predates recency batch); "WM = sum / OR-like" wrong; "separate q0/k0 refuted four times" map-task-only; "App. A.4" should be A.7 (CLAUDE.md). MapFormer's Fig. 11 capacity claim cites [11] for support it does not provide.
- Sources: `TALE_OF_TWO_ALGORITHMS.md`, `EM_WM_THEORY.md` Sec 3, `EM_WM_STATE.md` Sec 1.
- Bears on: EM vs WM framing; capacity claim.

---

## Excluded

| file / claim | reason | killed by |
|---|---|---|
| "MapWM is additive / OR-gate; EM AND-gate" (CLAUDE.md 2026-05-10, TALE, THEORY_KERNEL Sec 5, MINIGRID_EM mechanism prose) | MapWM rotates content Q,K; per-pair kernel | `AUDIT_2026-09-10.md` #1 |
| Thm 3 (`d score/d A_X = 1`) and corollary "EM ~ WM at constant offset, EM << WM when offset varies" | WM not additive; offset chosen by model; construction 1423/1423 | `AUDIT_2026-09-10.md` #1, #2; `WARM_RESULTS.md` W1 |
| "EM's recency deficit is a function-class limit" | frozen install 1.000 | `WARM_RESULTS.md` |
| Thm 2 corollary: coherence inverts by task (clock/map) | same sign on both tasks; interaction unmeasured | `N5_RESULTS.md` |
| N5 sign "strengthening" (rho=-1 as good as +1 as a finding); "3 of 8 seeds start at rho<0, down-weighting" | gauge, expectation zero | `AUDIT_2026-09-10.md` #5 |
| "Freezing is catastrophic on recency" (N5) | confounded with ~100x lower amplitude | `AUDIT_2026-09-10.md` #4 |
| N2 (MapPoPE rho lottery) | ill-posed, MapPoPE has no q0/k0 | `AUDIT_2026-09-10.md` |
| N1 as a test of Thm 3 corollary | tests a dead corollary; restated as learnability (effectively run as F14) | `AUDIT_2026-09-10.md` |
| D4 "reproduced +0.237 to three decimals" | 16/16 bitwise-identical checkpoints (determinism) | `AUDIT_2026-09-10.md` #10; `D5_RESULTS.md` |
| n=8 additive decomposition +0.148 / +0.120 / -0.031 | at n=24 one component and two zeros; magnitude flips sign on fresh seeds | `D5_RESULTS.md` |
| "Final losses order monotonically in total freedom" | P0 1.143 < AlignLock 1.239 at n=24 | `AUDIT_2026-09-10.md` stale list |
| D5's "magnitude freedom buys nothing" (as first stated) | parameter barely moved; re-established by MagOnly M3 | `AUDIT_2026-09-10.md` #8; `MAGONLY_RESULTS.md` |
| D3 as a test of the mechanism (+0.236) | subtracts OOD from in-distribution accuracy; clears MDE via unmeasured opposite-sign term | `AUDIT_2026-09-10.md` #6 |
| "Optimisation label follows from rule 9" | loss-matching conditions on a mediator (label kept on existence grounds) | `AUDIT_2026-09-10.md` #7 |
| "Phase freedom helps EM find the rewind" | route unchanged (0.445 vs 0.425) | `MAGONLY_RESULTS.md` probe; `SEARCH_RESULTS.md` S1 |
| "0 of 40 from-scratch EM runs find a rewind" (linear-slope probe, `_REWIND_PROBE.json`) | linear readout blind to wrapped rewinds | `SEARCH_RESULTS.md` |
| W4 "training dismantles the solution; landscape property"; AUDIT Tier-1 "sharpened" bullet | ~84% was install scale (8x install 0.941) | `UNFREEZE_RESULTS.md` U3 |
| Early-window mechanism (random content gate dismantles rewind) | late release breaks within 1-6 / 1-19 epochs | `UNFREEZE_RESULTS.md` U2 |
| "Erosion tracks the learning rate" as a whole-code claim | measured on coordinate 0 only | `UNFREEZE_RESULTS.md` top block |
| U4 "two recorded channels exhaustive" / "at 8x the code holds; leakage does the damage" | two latent coordinates; full pathway -0.866 | `UNFREEZE_RESULTS.md` top block; `NOLEAK_PREREG.md` revision |
| "EM does not learn the task at all"; WM loss range "0.008-0.044" | P0 0.600 vs chance; range 0.008-0.380 | `AUDIT_2026-09-10.md` stale list |
| "Separate q0/k0 refuted four times" as general | sign reverses on recency | `AUDIT_2026-09-10.md` #3; `RECENCY_EM_RESULTS.md` |
| "Our recency task is [11]'s N-back" | [11]'s N-back is fixed N, no filler | `EM_WM_STATE.md` Sec 1 (CORRECTED 2026-09-11) |
| "EM wins wherever the offset is fixed" (`EM_WM_THEORY.md` v1 2a) | one 16-epoch batch; monotone in length; best arm per-pair; other tasks tie/reverse | `EM_WM_THEORY.md` v2 2a |
| v1 phase-spread numbers (2.003; "~1.97 near-uniform") | three probe bugs; null is 3.267 | `EM_WM_THEORY.md` "What v1 got wrong" |
| "On map tasks EM and WM tie within 0.004" (EM_WM_STATE Sec 5 point 1, RESULTS_INDEX) | false at extended length on paper task | `EM_WM_THEORY.md` 2a; `PAPERTASK_RESULTS.md` |
| 16-epoch paper-task n=8 cell (`PAPER_OOD_EXTENDED_n8.json`) as headline | LinearLR 16 ep, logs deleted; superseded by 50-ep rerun | `PAPERTASK_PREREG.md`, `PAPERTASK_RESULTS.md` |
| PAPERTASK P1 verdict (EM - WM >= +0.20 at l=2048) | convergence gate failed; not read | `PAPERTASK_RESULTS.md` |
| PAIRORIGIN n=8 accuracy attribution (+0.280 as freedom) | split pathway/freedom; superseded at n=48 | `PAIRCONST_RESULTS.md`, `PAIRSPLIT_RESULTS.md` |
| SPREAD S4-P1 "CONFIRMED" | ceiling-vs-ceiling, could not fire | `SPREAD_RESULTS.md`; re-run F16 |
| T1 "frequency pruning is the search strategy" | r = -0.546, opposite sign | `THEORY_SEARCH_AND_LENGTH.md` T1 |
| T2 pigeonhole / dose form | no dose-response above one collision | `THEORY_SEARCH_AND_LENGTH.md` T2 |
| T2 "length acts only through collisions" | collisions explain 53% | `THEORY_SEARCH_AND_LENGTH.md` T2 |
| VOCAB_EM "EM is worse remains dead" | recency -0.375 next day | `VOCAB_EM.md` annotation; `RECENCY_EM_RESULTS.md` |
| TALE headline ("N-back never run", "no EM/WM performance story") | stale | `AUDIT_2026-09-10.md` |

## Cross-line dependencies

- **Paper-task floors** (`PAPER_TASK_FLOORS.md`: 0.522 IID, 0.216 OOD-d, ~0.80 at pe=0.8) are needed by any line reporting the paper's OOD protocol.
- **MapPoPE-Flat is directionally best on the paper task at extended length** (F20: raw 0.963 at l=2048; MapPoPE - EM unmeasured) and has 1.000 IID with the lowest final loss (0.0168): relevant to the PoPE / encoding line and to rank (MapPoPE at r=2 default).
- **Allocentric MiniGrid: path integration beats index only with r=4** (F22: Vanilla_r4 - RoPE +0.034, MDE 0.014, 8/8; r=2 -0.010 unmeasured): relied on by the environment / rotation / allocentric-recoding line and the rank line.
- **VanillaEM_P0_r4 as the preferred EM configuration** (F21, F23: no collapse on vocab or MiniGrid): relevant to any line using an EM arm; the separate-q0/k0 "pathology" is map-task-only.
- **The recency task's clock/map crossover and the rewind construction** (F2; scope correction "recency does not need a clock"): the sign/clock-vs-map line (`RECENCY_RESULTS.md`, `SIGN_ABLATION.md`) must carry this scoping; Thm 1 is scoped to scalar accumulators.
- **Rule 9 on the paper task** (F20: r = -0.461) vs recency (-0.936..-0.986): relevant to any line interpreting loss-matched residuals.
- **The OOD-length axis** (F26 T2: collisions account for 53% of EM's own length drop; cross-arm untested): bears on rank, InEKF (correction), forget gate and PoPE lines' shared "helps at OOD length" signature.
- **Fresh-seed shrinkage pattern** (F7, F8, F19: first eight seeds overestimated 1.9x, ~1.9x, and 30% on totals): methodological dependency for any n=8 claim elsewhere.
- **[11] capacity non-transfer** (F27) bears on any claim citing MapFormer's EM capacity (Fig. 11).

## Files read

EM_WM_STATE.md, AUDIT_2026-09-10.md, THEORY_KERNEL.md, EM_WM_THEORY.md, THEORY_SEARCH_AND_LENGTH.md,
TALE_OF_TWO_ALGORITHMS.md, REC_EM_PREREG.md, RECENCY_EM_RESULTS.md, DOF_PREREG.md, DOF_RESULTS.md,
_DOF_TORUS_RAW.md, N5_PREREG.md, N5_RESULTS.md, _N5_TORUS_RAW.md, D5_PREREG.md, D5_RESULTS.md,
MAGONLY_PREREG.md, MAGONLY_RESULTS.md, WARM_PREREG.md, WARM_RESULTS.md, UNFREEZE_PREREG.md,
UNFREEZE_RESULTS.md, NOLEAK_PREREG.md, NOLEAK_RESULTS.md, SEARCH_PREREG.md, SEARCH_RESULTS.md,
SPREAD_PREREG.md, SPREAD_RESULTS.md, SPREAD2_PREREG.md, SPREAD2_RESULTS.md, PAIRORIGIN_PREREG.md,
PAIRORIGIN_RESULTS.md, PAIRCONST_PREREG.md, PAIRCONST_RESULTS.md, PAIRSPLIT_PREREG.md,
PAIRSPLIT_RESULTS.md, PAPERTASK_PREREG.md, PAPERTASK_RESULTS.md, PAPER_OOD_RERUN.md,
PAPER_TASK_FLOORS.md, VOCAB_EM_PREREG.md, VOCAB_EM.md, MINIGRID_EM.md, MINIGRID_EM_PREREG.md,
MINIGRID_EM_FIX.md, MINIGRID_EM_FIX_PREREG.md, EM_P0_PAPER.md, MATCH_QUERY_EM.md, THEORY_NARRATIVE.md
(map/cross-check only), RESULTS_INDEX.md, archive/void/README.md, INVENTORY_BRIEF.md.
Partially: N3_AUDIT.md (Match-Query section, grep), KNOWN_BUGS.md (grep), CLAUDE.md (top and grep for
CORRECTED/WITHDRAWN), EM_P0_COMP.md (head), PAPER_TASK_ACCURACY.md (grep), _COLLISIONS.json (keys).

## Files in scope not covered

None of the listed files is missing. Not opened (referenced only): `runs/search/S2_report.md`,
`runs/search/S3_report.md`, `_ACHIEVABLE.json`, `_ANATOMY*.json`, `_INIT_GRAD.json`, `_REWIND_PROBE.json`,
`_PHASE_SPREAD.json`, `_PAIRORIGIN.json`, `_NOLEAK.json`, `_MAGONLY.json`, `PAPER_OOD_EXTENDED_n8.json`,
`papers/txt/tale_two_algorithms.txt`, `RECENCY_GATES_K4SET.md`, `RECENCY_GATES_K16SET.md`,
`RECENCY_GATES_K64.md`, `EM_FIX_COMP.md`, `EM_COMP_SAMEBATCH.md`. Numbers attributed to them are copied
from the results files that quote them.

### Unresolved or noted source disagreements
1. `EM_WM_STATE.md` top PAIRSPLIT block garbles the decomposition ("total +0.191 = pathway +0.100 +0.124 + freedom +0.091 = total +0.215"); `PAIRSPLIT_RESULTS.md` is authoritative: n=24 total +0.191 / pathway +0.100 (unmeasured); n=48 total +0.215 / pathway +0.124 / freedom +0.091, all detectable.
2. Per-token-rewind fraction for P0 solved cells: 0.459 (`SEARCH_RESULTS.md` H-wrap, any-head peak readout), 0.427 peak + 0.541 trough (route table, most-attended head), 0.962 (`PAIRORIGIN_RESULTS.md`), 0.964 (`PAIRCONST_RESULTS.md`). Different readouts/probe runs; not reconciled in sources.
3. WM phase spread 1.947 / null 3.267 (`EM_WM_THEORY.md`) vs 1.966 / 3.274 (`PAIRORIGIN_RESULTS.md`): different batches of the probe.
4. `Vanilla_r4` on MiniGrid: worst-to-2nd 0.798 -> 0.805 gap 0.007 (`MINIGRID_EM.md`) vs 0.798 -> 0.814 gap 0.016 (`MINIGRID_EM_FIX.md`): two batches; within-batch comparisons unaffected.
5. `EM_WM_STATE.md` Sec 5 and `RESULTS_INDEX.md` still carry "map tasks tie within 0.004" in places, contradicted by the paper-task extended-length result; the later `EM_WM_THEORY.md`/`PAPERTASK_RESULTS.md` win.
6. `THEORY_NARRATIVE.md` ledger #13 labels MiniGrid/vocab ties "no MDE quoted"; `VOCAB_EM.md` gives MDE 0.169 at n_obs=256 and `MINIGRID_EM_FIX.md` gives MDE 0.044 for EM_P0_r4 - EM_r4 but no MDE for EM_P0_r4 - Vanilla_r4.
7. Paper appendix for separate q0/k0: "App. A.4" in TALE, REC_EM_PREREG, MINIGRID_EM_FIX, MATCH_QUERY_EM vs "App. A.7, line 1527" (CLAUDE.md correction 2026-09-11).
