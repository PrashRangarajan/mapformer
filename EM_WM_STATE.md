# EM vs WM and the position-kernel theory -- current state (2026-09-11)

> **LEAKAGE TEST LANDED (2026-09-11) -- `NOLEAK_RESULTS.md`.** With `w_in`'s content columns
> held at zero, the 8x-installed TRAINABLE rewind scores **1.000 on 8/8 seeds (0.991 at 2x
> length)** -- identical to the frozen install, against 0.941 with the leak open. Leakage is
> the entire accuracy residual; the latent pathway settles near -0.86 with or without it and
> that costs nothing. At the 1/64 scale the rewind is erased with zero leakage (-0.008), so
> the channels are separable. **EM's recency deficit is entirely a search problem**: the
> solution exists (1.000), can be held (1.000), and is never found from scratch (0/40)
> [CORRECTED by SEARCH, next block: it IS found, per token and wrapped, for ~half the k].
> Registered verdicts split: L1 accuracy half met / slope half not (-0.859 vs -0.95); L2
> slope half met / accuracy half not (0.784 vs <= 0.75). A slope of -0.86 with perfect
> accuracy means the slope statistic is not a sufficient summary of the rewind.

> **SEARCH LANDED (2026-09-11) -- `SEARCH_RESULTS.md`. "Never found (0/40)" is WITHDRAWN.**
> It came from a LINEAR slope, and the rewind only has to hold modulo each block's period. From
> scratch, EM finds it per query token, wrapped: every solved large-k cell in all five EM arms
> is a rewind of the query token (~40% to the kernel's peak with A_X > 0, ~55% to its trough
> with A_X < 0), and failed cells never carry one. With ONE k, EM finds a 63-symbol rewind on
> 7/8 seeds (0.985), faster than WM (54 vs 99 epochs), and beats WM at 2x length (+0.191, 7/8,
> exploratory). A k curriculum gives +0.127 (detectable) without closing the gap, all of it at
> k <= 32. **The search problem is spread across 64 per-token rewinds, not the size of any
> one.** Phase freedom moves every head's kernel peak off zero (48/48) but does not change the
> route or shorten the shift. It raises each token's success rate at every distance.

> **THEORY SYNTHESIS (2026-09-11): `EM_WM_THEORY.md`.** Written over this file, the graveyard
> audit and the literature read first-hand. Its claim: EM applies ONE kernel to every pair and
> content can only rescale it; WM's kernel is reshaped per pair (measured: WM phase spread
> 2.003 rad across pairs, EM exactly 0.000). EM therefore WINS at a structurally fixed offset
> and loses only when the offset varies per query.

**This is the single current account of the EM/WM line (2026-09-09..11).** It summarises;
it does not replace the source files. Where a source file carries a CORRECTED or AUDIT block
at its top, that block supersedes the text beneath it. Where this file and a source disagree,
the source wins and this file is the bug. Every number here is copied from the file named
beside it.

Reading order if you need the detail: `AUDIT_2026-09-10.md` -> `MAGONLY_RESULTS.md` ->
`WARM_RESULTS.md` (top block) -> `UNFREEZE_RESULTS.md` (top block) -> `NOLEAK_RESULTS.md`.
`THEORY_KERNEL.md` is the theory as first written, with inline withdrawal markers.

**In flight (2026-09-11): SPREAD** (`SPREAD_PREREG.md`, `runs/spread/`) -- does EM's per-token
search limit track queries per token or the number of query tokens? Nothing else.

---

## 1. Question and framing

**The project's goal** (stated 2026-09-08): positional encoding as the mechanism by which a
model learns a relational *where* kept separate from the *what* -- the TEM claim that this
factorisation is what buys transfer. MapEM's score `A_X (*) A_P` is TEM's `g (x) x`
conjunction (tensor-lift identity, CLAUDE.md 2026-09-07/08). So EM vs WM is a question
about how the where meets the what.

**The source.** Whittington et al., *Neuron* 2025 (MapFormer's ref [11];
`papers/txt/tale_two_algorithms.txt`). It says EM and WM **solutions** are equivalent once
trained, their **learning dynamics** differ, and EM learns faster **"except on N-back"**.
`TALE_OF_TWO_ALGORITHMS.md`: its capacity result does not carry over to MapFormer, because
MapEM has no separate memory network. Both MapFormers are WM models in the Neuron paper's
sense.

**What we had measured before this line.** ~~EM - WM was +0.000 to +0.004 across four map
cells~~ -- **incomplete (`EM_WM_THEORY.md`): on the paper task at extended length EM is AHEAD
by +0.070 (l=1024) and +0.085 (l=2048), 8/8 seeds, one batch, MDE 0.034, computed from
`PAPER_OOD_EXTENDED_n8.json`; no results file stated it.** The cells behind the tie
(MiniGrid allocentric r=4 +0.0035; vocab sweep trimmed at n_obs=256 +0.0000 --
`TALE_OF_TWO_ALGORITHMS.md`, `RECENCY_EM_RESULTS.md:38`). ~~Our recency task is a k-back
task, i.e. the one exception [11] names.~~ **CORRECTED 2026-09-11: it is not.** [11]'s N-back
is a FIXED N with no filler (`tale_two_algorithms.txt:1672-1674`), so it corresponds to our
FIXED-k condition -- where EM is FASTER than WM (54 vs 99 epochs) and better at 2x length.
Our varying-k task is a condition [11] never ran.

**The architectural contrast, corrected** (`AUDIT_2026-09-10.md` finding 1, verified in
`model.py:224-232`):

- **MapEM**: `softmax(A_X (*) A_P)`, `A_P = q0^T R(dtheta) k0`. One position kernel,
  **shared by every query-key pair**, independent of content.
- **MapWM**: `Q_t^T R(theta_s - theta_t) K_s = sum_b |Q_b||K_b| cos(omega_b dS + phi_b(q,k))`.
  The position kernel's amplitudes AND phases are **set per pair by content**.
  **WM is not additive. There is no OR-gate.**

## 2. The theory as it stands now

### The frame (`THEORY_KERNEL.md` Sec 1)

    A_P[t,s] = q0^T R(theta_s - theta_t) k0 = sum_i a_i cos(omega_i (S_t - S_s) + phi_i) =: kappa(dS)

where `S_t = sum_{u<=t} Delta_u` is the accumulator, `a_i = |q0_i||k0_i|` and `phi_i` the
per-block phase. Three design axes:

| axis | sets | knobs |
|---|---|---|
| **A. argument** | what `dS` measures | signed vs monotone `Delta`; rank r; index (`Delta == 1`) |
| **B. kernel** | shape of `kappa` | phases (coherence `rho`); magnitudes `a`; frequencies `omega` |
| **C. composition** | how `kappa` meets content | EM: one shared `kappa` times `A_X`. WM: a per-pair kernel whose amplitudes and phases content sets. (Originally written "product vs sum"; the "sum" is wrong.) |

The frame is exact algebra for EM. For WM it holds only with per-pair amplitudes and phases.

### Status of each claim

| claim | status | reason | decided by |
|---|---|---|---|
| `A_P = kappa(dS)`, axes A/B/C | **established (algebra)**, with C corrected | WM's kernel is per-pair | `AUDIT_2026-09-10.md` #1 |
| **Thm 1** -- cancellation is exclusive: no single accumulator is both a map and a clock | **scoped** | true only for a scalar or fully constrained accumulator. The model's is rank-r per head, so one subspace can cancel while another counts. Recency does not need a clock at all (rewind construction below) | `AUDIT_2026-09-10.md` #2 and "Stale" list |
| Thm 1's data (crossover: monotone costs -0.280 torus, 12/12; -0.004 recency, inside MDE) | **data stands; the theorem is a retrodiction of it** | the recency half shows a monotone increment is *harmless* there, not that recency *needs* one | `RECENCY_RESULTS.md`; `AUDIT` #2 |
| **Thm 2** -- coherence `rho = kappa(0)/sum a_i`; single `p0` forces `rho = 1`, separate `q0/k0` gives `E[rho] = 0` | **algebra exact; init sd verified** (measured 0.147 vs predicted 0.160 over 16 head-values) | the "~1/sqrt(2 n_b)" shorthand is off by 4/pi; the exact expression gives 0.159 | `THEORY_KERNEL.md` Sec 4; `AUDIT` "Stale" list |
| Thm 2 in `\|rho\|` form: `\|rho\|=1 - \|rho\|=0` = +0.292 on the torus, 8/8 | **established only for a FROZEN kernel at peak amplitude ~0.003**. Learned kernels reach 0.06-0.08 (torus) and 0.27-0.50 (recency) | the axis was chosen after the data. The sign is a gauge (see next row) | `N5_RESULTS.md`; `AUDIT` #4, #5 |
| Sign of `kappa` matters (`rho = +1` vs `-1`) | **withdrawn -- a gauge** | `A_X (*) (-kappa) = (-A_X) (*) kappa`; `(W_q,b_q) -> (-W_q,-b_q)` maps `minus` onto `plus` exactly. P1/P2 had expectation zero before any run. The "3 of 8 seeds start at rho < 0, the kernel down-weights the same place" line is withdrawn | `AUDIT` #5 |
| Thm 2 corollary -- coherence is a clock/map choice and **inverts by task** | **refuted** | coherence helps on both tasks with the same sign. Interaction +0.113, MDE 0.195, unmeasured | `N5_RESULTS.md` |
| `rho` at init predicts which seed suffers (N4) | **not established** | r(rho_init, final loss) +0.142 on recency (n=8, a null); +0.029 on MiniGrid (unmeasured: the outcome has almost no range). `rho` also does not converge toward 1 in training | `THEORY_KERNEL.md` Sec 8 |
| **Thm 3** -- composition: EM gates content by `kappa`, WM is `A_X + kappa` with `d score/d A_X = 1` | **withdrawn** | MapWM is not additive | `AUDIT` #1 |
| Thm 3 corollary -- `EM ~ WM` when the retrieval offset is constant, `EM << WM` when it varies per query | **withdrawn** | the offset is chosen by the model, not the task. A single-`p0` EM kernel solves recency exactly once the query token rewinds the count | `AUDIT` #2; `WARM_RESULTS.md` W1 |
| EM half of Thm 3 restated (`d score/d A_P = A_X`: a random content branch feeds the position pathway noise and dismantles the rewind early) | **refuted** | released at epoch 30 or 100, after content has trained, the rewind still breaks within 1-6 / 1-19 epochs on every seed | `UNFREEZE_RESULTS.md` U2 |
| **Phase degrees of freedom** (the third proposed mechanism for `sep` vs `P0`) | **established as an effect on what training finds** on recency (+0.146 vs a matched-optimiser control, 22/24). **Does not act through the rewind.** Map-side half (freedom costs on the torus) n=8 only, provisional | see Sec 3 | `MAGONLY_RESULTS.md`; `D5_RESULTS.md`; `DOF_RESULTS.md` |
| The phenomenon phase freedom was introduced to explain (`sep - P0` on recency) | **does not replicate at detectable size on fresh seeds** (+0.073, 9/16, MDE 0.130). Pooled n=24 is +0.128, carried by seeds 0-7 | the mechanism is better supported than the phenomenon | `AUDIT` #3 |
| WM > EM-sep > EM-P0 on recency tracks phase freedom (per pair > `n_b` global > none) | **post-hoc hypothesis, untested** | consistent with findings 1-3, rule 9, DOF, D5 -- and that is all | `AUDIT` "Where the account now stands" |

Predictions listed in `THEORY_KERNEL.md` Sec 8:

- **N1** (fixed-k recency): needs restating. With a fixed k the rewind is a constant, so N1
  now tests whether a simpler rewind can be learned. Not run.
- **N2**: withdrawn as ill-posed. `MapFormerWM_PoPE` has no `q0/k0`.
- **N3** (`k0 = q0` at init, both free): run in effect as `AlignFree` in DOF, never scored
  as N3. Arm means only: torus T=1024 `AlignFree` 0.875 vs `P0` 0.962 vs `sep` 0.809 (n=8);
  recency `AlignFree` 0.818 vs `sep` 0.814 (n=24). Initial coherence is null on recency
  (-0.004).
- **N4**: tested, not established (table above).
- **N5**: run; see Sec 3.

### What the theory does NOT explain

- **The EM search failure itself.** The frame describes a function class. EM's recency
  deficit sits outside that class (the solution is inside it). The frame has nothing to
  say about why training never finds the solution.
- **What `AlignFree` does instead of a rewind.** It reaches 0.76 at k=64 against `P0`'s
  0.37 (seeds 0-7) with no rewind anywhere (`MAGONLY_RESULTS.md`).
- **Optimisation effects in general.** On recency every origin-vector contrast is an effect
  on FIT: r(loss, acc) = -0.986 over 96 runs, and loss-matched residuals are ~0 (`D5_RESULTS.md`).
- Carried over from `THEORY_KERNEL.md` Sec 7: the map-size threshold (flat at 32 and 128
  occupied cells, +0.305 at 512); loop/recursion; "helps at OOD length" as the shared
  signature of rank, InEKF, forget gate and PoPE; the InEKF (it changes S but not alpha).

---

## 3. Experiments

All on the recency task unless marked: `k_max=64`, `p_filler=0.5`, `min_gap=64`, train
T=1024, 300 ep cosine, lr 1e-3, 1 layer, d=128, no `--fast-attn`. Chance 0.0625,
most-recent floor 0.0771, chance loss 2.77. The torus runs use the paper task, 300 ep,
T=128 train, readout T=1024. MDE = 2.8*sd/sqrt(n) on paired differences.

| experiment | question | arms | n | headline (current numbers) | verdict | file |
|---|---|---|---|---|---|---|
| **Recency EM** | does [11]'s N-back exception appear? | `Vanilla_r4` (WM), `VanillaEM_r4` (sep), `VanillaEM_P0_r4` | 3x8, one batch | T=1024: WM 0.975+/-0.072, sep 0.837+/-0.081, P0 0.600+/-0.126. **P0 - WM -0.375** (MDE 0.154, 0/8); sep - WM -0.137 (MDE 0.106, 1/8). Registered readout, epochs to loss <0.5: WM 8/8 (median 70.5), P0 0/8, sep 1/8; to <0.1: WM 7/8 (median 118), P0 0/8, sep 0/8. At T=2048: -0.437 / -0.218. Per-offset k=64: WM 0.97, sep 0.73, P0 0.37 | P1 confirmed. P2 refuted, read as a **learnability** result. P3 refuted with the sign inverted: the separate form is better | `RECENCY_EM_RESULTS.md`, `AUDIT` #9 |
| **Existence construction** | can a single-`p0` EM kernel solve recency? | symbols `Delta=(1,0)`, `q_k` `Delta=(-(k-1),0)`, MASK `(0,1)` | 3 draws | 1423/1423, 1417/1417, 1415/1415. Without the rewind: 0.0857 | **yes, at kernel level** (argmax with an idealised content gate). Upgraded to full model by W1 | `AUDIT` #2 |
| **N4** | does `rho` at init predict a seed's fate? | probe on existing checkpoints | 8 + 8 | r = +0.142 (recency), +0.029 (MiniGrid) | not established | `THEORY_KERNEL.md` Sec 8 |
| **N5** | set coherence by construction (frozen, magnitude-matched) | `EMPhase_{plus,minus,zero,rand}_r4` | 4x8 per task | Torus T=1024: plus 0.959, minus 0.943, zero 0.638, rand 0.679. **`\|rho\|` torus +0.292** (MDE 0.063, 8/8). Recency `\|rho\|` +0.180 (MDE 0.184) unmeasured. Recency plus - zero **+0.191** (7/8, MDE 0.158, detectable; left out of the original write-up). Interaction +0.113 (MDE 0.195). zero vs rand: -0.040 / -0.064, both unmeasured | inversion **refuted**; `\|rho\|` holds only for a frozen kernel at ~0.003 amplitude; sign is a gauge | `N5_RESULTS.md`, `AUDIT` #4-5 |
| **DOF, torus half** | does phase freedom cost where the kernel is already right? | `AlignFree`, `sep`, `AlignLock`, `P0` | 4x8 | T=1024: AlignLock 0.963, P0 0.962, AlignFree 0.875, sep 0.809. D2 AlignFree - AlignLock -0.088 (MDE 0.118, 1/8) unmeasured; sep - P0 **-0.154** (MDE 0.130, 0/8) detectable. All arms reach ~2.5e-4 training loss; r(loss,acc) = -0.160. Final rho: AlignFree 0.363+/-0.198, sep 0.173+/-0.196 | direction against freedom. **n=8, not extended -- provisional** | `DOF_RESULTS.md` |
| **DOF D3** | interaction (recency - torus) | as above | 8 | +0.236 (se 0.053, MDE 0.148); +0.253 with the n=24 recency term | **reproduces the `sep`/`P0` sign flip, not the mechanism**. It subtracts OOD torus accuracy from in-distribution recency accuracy, and it clears its MDE only because of an unmeasured term of opposite sign | `AUDIT` #6 |
| **D5 (DOF recency, n=24)** | phase vs magnitude vs initial coherence | `AlignFree`, `sep`, `P0`, `AlignLock` | 4x24 | T=1024: AlignFree 0.818+/-0.127, sep 0.814+/-0.106, P0 0.687+/-0.127, AlignLock 0.654+/-0.141. **Phase (AlignFree - AlignLock) +0.165** (MDE 0.088, 21/24); fresh seeds 8-23 alone +0.173 (13/16, MDE 0.127). sep - P0 +0.128 (MDE 0.108, 17/24); fresh alone +0.073 (9/16, MDE 0.130) unmeasured. Magnitude -0.033 (MDE 0.116); init coherence -0.004 (MDE 0.076). r(loss,acc) -0.986; loss-matched residuals +0.001 / -0.009 | phase freedom is the whole effect; magnitude and init coherence null. The n=8 sizes were ~1.9x high | `D5_RESULTS.md`, `AUDIT` #3 |
| **MagOnly** | is D1 phase freedom, or AlignLock's parameterisation (scale `s` init 1.0 barely moves under Adam)? | new `EMDoF_magonly` (same function as AlignFree at init, same optimiser scale, phases pinned) vs existing arms | 24 new + 2 determinism checks | **AlignFree - MagOnly +0.146** (sd 0.150, MDE 0.086, 22/24); fresh seeds 8-23 +0.113 (14/16, MDE 0.111, clears by 0.002). MagOnly - AlignLock +0.018 (MDE 0.089). MagOnly - P0 -0.015 (MDE 0.083) with magnitudes moving ~59% vs 1% from weight decay. AlignFree fits better on 23/24 seeds. Determinism check bitwise identical | **phase freedom survives its control**; the confound was ~a tenth of D1 | `MAGONLY_RESULTS.md` |
| **Rewind probe** (post-hoc) | do from-scratch EM arms learn a rewind? | 40 from-scratch runs (5 parameterisations x 8 seeds) | 40 | pooled rewind slope 0.000 +/- 0.02 (exact rewind = -1); 1 of ~2,550 (head, block) pairs below -0.5; the position kernel alone picks the answer on 11-16% of queries | **no arm learns a rewind.** Phase freedom helps some other way | `MAGONLY_RESULTS.md`, `_REWIND_PROBE.json` |
| **Warm-start** | can the full EM model hold the rewind? | `EMWarm_freeze` (position pathway installed + frozen), `EMWarm_train` (installed, all trainable) | 2x8 | freeze **1.000+/-0.000** at T=1024 and T=2048 (8/8, every k). train 0.642+/-0.266. W2 freeze - P0 +0.400 (MDE 0.125, 8/8). W3 freeze - WM +0.025 (MDE 0.071) unmeasured. W4 train - freeze -0.358 (MDE 0.263, 0/8) | **W1 confirmed**: the full model represents recency exactly. W4's "the landscape rejects the solution" reading is **withdrawn** (next row) | `WARM_RESULTS.md` (read its top block) |
| **Freeze-then-unfreeze** | early window, or install scale? | `EMUnf_0/5/30/100` (released at step 0 / epoch 5/30/100, install 1/64), `EMUnf_0_e8` (step 0, install 1/8 = 8x) | 5x8 | T=1024: 0.642 / 0.609 / 0.835 (3/8 >= 0.95) / 0.605 / **0.941+/-0.063** (4/8 >= 0.95). **U3 e8 - e64 +0.298** (sd 0.272, MDE 0.270, 7/8), 84% of the gap to frozen. Slope first above -0.5: 9-17 epochs from step 0; **1-6 epochs after release at 30; 1-19 after release at 100**. At 8x, endpoint latent-pathway slope -0.866 (coord 0 alone -0.989), effective -0.602 | **early window refuted; install scale dominant.** U4's "two channels are exhaustive" withdrawn | `UNFREEZE_RESULTS.md` (read its top block) |
| **Leakage test** | is content leaking into Delta through `w_in` the residual at 8x? | `EMNoLeak_e8`, `EMNoLeak_e64` (w_in content columns held at 0) against existing `EMUnf_0_e8`, `EMUnf_0`; + determinism re-check (bitwise PASS) | 2x8 new | leak closed: **8x 1.000+/-0.000 (8/8), 0.991 at T=2048**; 1/64 0.784+/-0.281 (3/8). Leak open: 8x 0.941, 1/64 0.642. Latent pathway 8x -0.859 closed vs -0.866 open (L1b +0.007, MDE 0.309); 1/64 closed -0.008. Manipulation check exact (leak 0, identity 0) | **leakage is the entire accuracy residual; EM's recency deficit is entirely search.** L1: accuracy half met, slope half not (-0.859 vs -0.95). L2: rewind erased at 1/64 with zero leak (channels separable), accuracy 0.784 above the 0.75 collapse line | `NOLEAK_RESULTS.md` |

---

## 4. Withdrawn and corrected claims

Each is recorded in the file named. Do not revive any of them.

- **"WM is additive / an OR-gate; EM is an AND-gate."** MapWM rotates content Q,K, so its
  kernel is per-pair. The idea came from a 2026-05-10 summary and was never checked against
  the code (`AUDIT` #1).
- **Thm 3** (composition theorem, `d score/d A_X = 1` for WM). Same reason.
- **Thm 3's corollary** ("EM << WM when the retrieval offset varies per query"). The offset
  is chosen by the model; a rewinding query token makes it zero (`AUDIT` #2).
- **"EM's recency deficit is a function-class limit."** The frozen install scores 1.000 (`WARM_RESULTS.md` W1).
- **The coherence inversion** (Thm 2's corollary: coherence is a clock/map choice). N5
  finds the same sign on both tasks (`N5_RESULTS.md`).
- **The sign "strengthening"** (`rho = -1` as good as `+1` presented as a finding). It is
  a gauge with expectation zero, and "3 of 8 seeds start at rho < 0, down-weighting the same
  place" is withdrawn with it (`AUDIT` #5).
- **"Freezing is catastrophic on recency" (N5).** Confounded with amplitude: every frozen
  kernel sits ~100x below learned amplitude (`AUDIT` #4).
- **N2.** Ill-posed; MapPoPE has no `q0/k0` (`AUDIT` "Stale" list).
- **D4's "three-decimal reproduction" of +0.237.** The same computation twice: 16/16
  bitwise-identical checkpoints. At n=24 the contrast is +0.128, and +0.073 on fresh seeds
  (`AUDIT` #10, `D5_RESULTS.md`).
- **The additive decomposition at n=8** (+0.148 phase / +0.120 magnitude / -0.031
  coherence). At n=24 it is one component and two zeros. The magnitude term flipped sign on
  fresh seeds (`D5_RESULTS.md` E4).
- **"Final losses order monotonically in total freedom."** False at n=24: P0 1.143 <
  AlignLock 1.239 (`AUDIT` "Stale" list).
- **"Magnitude freedom buys nothing" as D5 first stated it.** That was a parameter the
  optimiser barely moved (weight decay alone predicts s -> 0.674; measured median ~0.70).
  **MagOnly re-establishes the null properly**: -0.015 with magnitudes that do move (`AUDIT` #8, `MAGONLY_RESULTS.md` M3).
- **D3 as a test of the mechanism.** It compares in-distribution with OOD accuracy (`AUDIT` #6).
- **"The optimisation label follows from rule 9."** Loss-matching conditions on a mediator.
  The label is right, but because of existence (the locked arms CAN represent the task),
  not because of rule 9 (`AUDIT` #7).
- **"Phase freedom helps EM find the rewind."** No from-scratch arm learns one (`MAGONLY_RESULTS.md` probe).
- **W4's landscape reading** ("training dismantles the solution; the correct position code
  is not an attractor"). About 84% of it was the scale at which the rewind was installed:
  0.642 -> 0.941 at 8x (`UNFREEZE_RESULTS.md` U3). Also withdrawn: the "sharpened" Tier-1
  bullet in `AUDIT_2026-09-10.md` and the matching text in `THEORY_KERNEL.md`'s header.
- **The early-window mechanism** (a random content gate dismantles the rewind before
  content trains). A late release is dismantled within 1-6 epochs (`UNFREEZE_RESULTS.md` U2).
- **"Erosion tracks the learning rate"** as a statement about the whole latent code. It was
  measured on coordinate 0 only (`UNFREEZE_RESULTS.md` top block).
- **U4's "two recorded channels are exhaustive" / "at 8x the code holds; leakage does the
  damage."** The latent code has two coordinates and trainable per-token rows. The full
  latent pathway degrades to -0.866 at 8x, so leakage is the larger share of the SLOPE
  residual (`UNFREEZE_RESULTS.md` top block). In ACCURACY terms leakage is the whole residual:
  closing it gives 1.000 while the pathway still sits near -0.86 (`NOLEAK_RESULTS.md`).
- **"MapEM's separate q0/k0 is refuted four times" / an initialisation pathology.** On recency the
  separate form is better (directionally; not replicated at detectable size on fresh seeds).
  The sign depends on the task (`AUDIT` "Stale" list, `RECENCY_EM_RESULTS.md` P3).
- **"EM does not learn the task at all"** (`RECENCY_EM_RESULTS.md:31`). Overstated: P0
  reaches 0.600 against a chance of 0.0625. The WM loss range there is 0.008-0.380, not 0.044.
- **The +0.123/+0.195 described as signed vs MONOTONE** in `THEORY_KERNEL.md`. Those numbers are
  signed vs INDEX; signed vs monotone is -0.215/-0.280 loss-matched (`AUDIT` "Stale" list).

---

## 5. Current best account, in plain words

1. **EM and WM differ on exactly one task we have measured: the k-back recency task.**
   There, single-`p0` EM trails WM by 0.375 and never gets its training loss below 0.5.
   On every map task they tie to within 0.004.
2. **This is a SEARCH problem, not a capacity problem.** Three parts, each measured:
   - *The solution exists.* With the rewind installed and frozen, EM scores 1.000 on 8/8
     seeds at T=1024 and T=2048, matching WM.
   - *It can be held.* Installed at a weight scale Adam does not erode (8x), the trainable
     model keeps 0.941; with the content -> Delta leak also closed it keeps **1.000 on 8/8
     (0.991 at 2x length)**, the frozen install's level. Leakage was the entire accuracy
     residual; the latent pathway's drift to ~-0.86 costs nothing (`NOLEAK_RESULTS.md`).
   - *It is found per token, not as a code.* ~~0 of 40 from-scratch EM runs learn a rewind~~
     (linear readout; WITHDRAWN). Each query token finds a wrapped rewind or not; ~45% of k in
     P0, more with phase freedom. One shared k is found on 7/8 seeds (`SEARCH_RESULTS.md`).
3. **Phase freedom in the origin vectors is real and does not work through the rewind.**
   Letting `k0`'s per-block phases move is worth +0.146 against a matched control (22/24;
   +0.113 on fresh seeds). It helps at large k, but the mechanism is unidentified. On this
   task it acts on fit: r(loss, acc) is about -0.98.
4. **On map tasks the matched filter (`rho = 1`) is already the right kernel.** Freedom
   there is directionally a cost at OOD length (n=8, provisional).
5. **The kernel frame is a correct description of the function class.** It is not an
   explanation of EM's deficit, because the deficit is not in the function class.

---

## 6. Open questions, ranked, each with its cheapest decisive experiment

1. ~~Is leakage the entire residual at 8x?~~ **RESOLVED 2026-09-11** (`NOLEAK_RESULTS.md`): in
   accuracy terms yes -- leak closed gives 1.000 (8/8). Registered verdicts split (L1 accuracy
   half met, slope half not; L2 slope half met, accuracy half not). A slope near -0.86 with
   perfect accuracy shows the slope statistic is not a sufficient summary of the rewind.
2. **[LARGELY ANSWERED by SEARCH_RESULTS.md: found per token, wrapped; the obstacle is
   spread across 64 tokens, not rewind size. Next: k from a small set at fixed queries per
   token; per-epoch recording of per-token rewind status.]** Original text follows.
   Why does from-scratch search never find the rewind (0/40)? This is now *the*
   question. Candidate from `UNFREEZE_RESULTS.md`: the rewind needs `Delta(q_k)` spread over
   a 64:1 range along one direction, and nothing in a random init points there. Cheapest
   decisive steps, in order:
   (a) a no-training probe of the gradient at init and in the first steps, projected onto
   the rewind direction;
   (b) N1 restated -- recency with k fixed, or a small-to-large `k_max` curriculum -- to ask
   whether a constant rewind is findable. One environment flag, 8 seeds.
3. **What does phase freedom do, if not rewind?** Eval-only on existing checkpoints
   (`runs/dof/recency`, the MagOnly runs): per-offset A_P argmax and the learned `kappa`
   shape for AlignFree vs MagOnly. No training.
4. **Does WM win recency through per-pair phase freedom?** First step is eval-only:
   measure the spread of WM's content-set phases `phi_b(q,k)` on `runs/recency_em`
   `Vanilla_r4`. A control arm needs design.
5. **Map-side DOF at n=24.** Extend the DOF torus arms with seeds 8-23 (4 arms x 16) and
   report fresh seeds alone beside the pooled figure (the E4/M5 guard).
6. **`|rho|` at learned amplitude.** Re-run N5 torus plus vs zero, frozen, with the kernel
   scaled to the learned peak amplitude (0.06-0.08). 2 arms x 8.
7. **Untested [11] predictions:** position decodable from EM vs activity slots from WM
   (Fig. 2G/2I), and parallel planning. The first may not be well posed here, since both
   MapFormers carry an explicit theta (`TALE_OF_TWO_ALGORITHMS.md`).

---

## 7. Method lessons from this line

Rules 27-33 are numbered in `CLAUDE.md` (Session 2026-09-09/10). Note: `RESULTS_INDEX.md`
uses 27 and 28 for two different rules (git add, grep case); the numbering collides.

- **27. A same-seed rerun is determinism, not replication.** D4 "matched" +0.237 to three
  decimals from bitwise-identical checkpoints; n=24 gave +0.128.
- **28. Check for a sign or scale GAUGE before registering a contrast.** N5's `rho = +1`
  vs `-1` had expectation zero because `A_X` absorbs the kernel's sign.
- **29. Existence before mechanism.** Before explaining a deficit as a function-class
  limit, construct a solution in that class. A 30-line construction killed Thm 3's
  corollary; no training run could have.
- **30. Report the registered primary readout even when the verdict is obvious.** REC_EM's
  epochs-to-threshold went uncomputed until the audit.
- **31. A parameterisation change is an optimiser change.** A scale initialised at 1.0
  under Adam moves ~50x slower in relative terms than a vector initialised at 0.02.
- **32. Existence, then stability.** Warm-start a constructed solution twice, frozen AND
  trainable.
- **33. Install a warm start at the scale training would use.** A solution stored in tiny
  weights tests whether Adam can erode it, not whether the landscape holds it.

Also shown by the files, not yet numbered in `CLAUDE.md`:

- **Loss-matching conditions on a mediator** (`AUDIT` #7). A zero loss-matched residual
  cannot by itself separate optimisation from representation. Existence decides.
- **Guard every pooled estimate with the fresh seeds alone** (D5 E4, MagOnly M5). The first
  eight seeds overestimated twice in this line: `sep - P0` +0.237 against +0.128 at n=24
  (1.9x), and MagOnly's +0.213 on seeds 0-7 against +0.113 on fresh seeds 8-23.
- **An interaction must subtract like with like** (`AUDIT` #6). D3 subtracted OOD accuracy
  from in-distribution accuracy.
- **Reuse a stored arm only after an in-batch bitwise determinism re-check** (MagOnly,
  NoLeak). This refines rule 3 and does not replace it.
- **Record the mechanism's state into the checkpoint every epoch.** W4's "destroyed early"
  was inferred from a loss curve. The unfreeze recordings refuted the inference.
- **An exhaustiveness claim needs every trainable route enumerated.** A probe that reads one
  coordinate of a two-coordinate code is not a reading of the code (U4).
- **Say when a theory is a retrodiction.** `THEORY_KERNEL.md` was written after the batch
  it explains, and its pre-registration had predicted the opposite sign
  (`RECENCY_EM_RESULTS.md` P3).
