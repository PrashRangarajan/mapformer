# A positional encoding for a cognitive map, from brain constraints: scorecard, mechanism, design, plan (2026-10-05)

Theory note plus three post hoc CPU checks. Nothing here is registered; no GPU job was run. Builds on
`neuro_positional.md` (same folder; its ~70 citations and grades are used, not repeated) and on
`docs/theory/2026-10-04/` (T1-T7). Every number of ours is copied from the file named beside it.

Labels on every claim. **EVID** = our measurement, with file and status (**REG** registered, **PH** post hoc,
**PIL** pilot). **LIT** = published finding, abstract at least read (for this note or `neuro_positional.md` /
`02_literature.md`). **DER** = derivation (follows from stated assumptions). **CONJ** = conjecture with a test.

New CPU checks (eval-only, committed checkpoints `runs/mappope_pair/p0`, all PH):
`docs/audits/2026-10-05/neuro_rank2_mech.py` / `_out.txt` / `.json` (head geometry, retrieval, content-phase and
delta lesions), `neuro_samecell_phase.py` / `_out.txt` / `.json` (same-cell kernel coherence by revisit gap),
`neuro_strip_obs.py` / `_out.txt` (content-phase lesion on observation keys only). Rebuilt forwards reproduce
`model(tokens)` exactly (max |dlogit| 0.0 on all 72 checkpoints).

## 0. Verdict

1. **Scorecard winner: MapPoPE, pairwise form (MapPoPE-Pair).** Highest brain score of any built encoding
   (30/46; tied with 64-angle MapPoPE, which is the same model with twice the angles and no measured benefit,
   COUNT +0.0002, CI [-0.0004, +0.0008], REG) and the best registered performance on the map task
   (16/16 vs 10/16 SOLVED at rank 2, `MAPPOPE_PAIR_RESULTS.md`, REG). **Runner-up: MapWM + NormStep** (28/46;
   +0.0107 in distribution on the new-object task, p 0.0002, REG, `LEAK_RESULTS.md`). MapEM ties on the brain score
   (the purest gain field) but carries a registered search cost (EM - WM -0.375 on recency). Ranking is stable
   under equal weights (MapPoPE 14/24, MapWM+NormStep 13, MapEM 12).
2. **The two winners fix disjoint constraints** (PoPE: content may gain, not shift, recall; NormStep: content may
   not set a gain on the velocity read-in). Their composition, unbuilt, scores 34/46 and is the proposed **core**.
   Everything else (modules, noise + memory reset, a time channel) must earn its place by ablation; section 3 says
   which prior negatives argue against each.
3. **Remap probe, in neural terms (PH):** MapEM = pure rate remapping (by construction, not learned); MapPoPE =
   rate remapping plus field-width change, stronger fields narrower in 3 of 4 MapPoPE arms (r -0.41 to -0.68; the
   pairwise rank-2 arm is the exception, +0.09); MapWM = gain plus skew, the field's phase centre displaced by a
   nearly content-free content phase. Field *location* is fixed for every converged path model.
4. **Why PoPE's score rescues rank 2 (new checks, PH, n = 16 seeds per arm):** not by making the rank-2 phase code
   clean -- MapPoPE r2 ends with drifting or collapsed heads on 6/16 (pairwise) and 9/16 (64-angle) seeds, and all of
   them solve, while MapWM r2 solves 6/11 such seeds. MapWM fits a defective code by *shifting* its kernel (a shared
   content phase, on which it depends: removing it on observation keys costs -0.155 at rank 2, -0.022 at rank 4);
   PoPE cannot shift (delta at 0 on 89-100% of channels, setting it to 0 changes accuracy by <= 0.010), so its only
   lever is non-negative gain, and its kernels stay centred on the revisited cell at every gap (same-cell coherence
   0.98 vs MapWM 0.78 at 65-127-move gaps). The claim "non-negative weights on cosines cannot create a peak above
   phase difference 0, only ties" is correct (DER) and verified in the code; what it buys is a readout that tolerates
   a defective where-code, not a better code.

## 1. The scorecard

### 1.1 Constraints, evidence, weights

Weight 3 = core to a path-integrating cognitive map with converging physiological + lesion evidence; 2 = strong
evidence, narrower role; 1 = true of brains but weakly discriminating between encodings or weakly evidenced. Score
per encoding: 2 satisfies, 1 partial / not applicable, 0 violates.

| id | constraint | evidence (LIT unless marked) | w | 2 / 0 means |
|---|---|---|---|---|
| a | position from self-motion (path integration) | grid phase = coordinate on a module torus preserved across environments and sleep (Gardner et al. 2022, Nature 602:123); passive transport abolishes grid patterns (Winter et al. 2015, Curr Biol 25:2493); path-integration gain recalibrated by landmarks (Jayakumar et al. 2019, Nature 566:533). EVID: position effect +0.243 at training length (`PAPER2X2_RESULTS.md`, REG), Match-Query 0.730 vs 0.154 (`MATCH_QUERY_SCALE.md`) | 3 | phase = integral of a movement signal / token count |
| b | signed velocity, not speed, drives the phase | VCO phase minus baseline integrates velocity (Burgess, Barry & O'Keefe 2007, Hippocampus 17:801); band cells along a few directions (Krupic, Burgess & O'Keefe 2012, Science 337:853); speed cells are a separate, context-invariant signal (Kropff et al. 2015, Nature 523:419). EVID: signed 8/8 vs monotone 0/8 at matched length, -0.177 (`SIGN_MATCHED_RESULTS.md`, REG) | 3 | signed / monotone or none |
| c | few discrete periodic modules, scale ratio ~1.4 | discrete modules (Stensola et al. 2012, Nature 492:72); residue-code capacity (Fiete, Burak & Brookings 2008, J Neurosci 28:6858; Mathis, Herz & Stemmler 2012); graded, finite place-field scale along the dorsoventral axis (Kjelstrup et al. 2008, Science 321:140). EVID: angle count adds nothing (COUNT +0.0002, REG); frequency learning negligible (`FREQ_CONTROL.md`, n=3, unmeasured) | 2 | modular / no multi-scale code or pooled time instead |
| d | phase from MEC, content gain from LEC, conjunction at recall; rate remapping keeps field location | rate remapping (Leutgeb et al. 2005, Science 309:619); stable grids under rate remapping (Fyhn et al. 2007, Nature 446:190); LEC needed for rate remapping (Lu et al. 2013, Nat Neurosci 16:1085); grid rates redistributed across fixed fields (Diehl et al. 2017, Neuron 94:83). EVID: SCORE +0.024, 16/16 vs 10/16 (REG) | 3 | content scales recall but cannot move it / content shifts or position not from movement |
| e | non-negative rates; activity/energy constraints | non-negativity + energy -> factorised codes (Whittington et al. 2023, arXiv 2210.01768); non-negative actionable codes -> modules (Dorrell et al. 2023, arXiv 2209.15563). Weakly discriminating: any rotary block can be read by a non-negative ring (`neuro_positional.md` 3.2) | 1 | non-negative per-channel gains / signed gain that inverts the kernel (MapEM) |
| f | multiplicative gain fields | Salinas & Abbott 1995 (J Neurosci 15:6461), 1996 (PNAS 93:11956). EVID: a gain on one shared kernel restores 1.000 on converged path models; additive is worse than none (`docs/WHAT_WHERE_CHECKS.md`, PH) | 2 | content x position is a gain / content phase |
| g | divisive normalisation of inputs to the integrator | canonical computation (Carandini & Heeger 2012, Nat Rev Neurosci 13:51); context-invariant speed cells (Kropff 2015). EVID: NormStep +0.0107 (REG); text world no difference (`TW_NORMSTEP_RESULTS.md`, REG) | 2 | step invariant to input intensity / step linear in the raw embedding |
| h | noisy integration corrected by memory / landmarks | drift without correction (Burak & Fiete 2009, PLoS Comput Biol 5:e1000291); boundary-driven error correction (Hardcastle, Ganguli & Giocomo 2015, Neuron 86:827); landmark vs self-motion gain (Campbell et al. 2018, Nat Neurosci 21:1096); hippocampal drive needed for grids (Bonnevie et al. 2013, Nat Neurosci 16:309). EVID: every correction line negative (`L15_ABLATION.md`, `MQ_NOISE_2X2.md`, `NOISE_REFINE.md`) | 2 | noise present and corrected from memory / exact integrator, no correction |
| i | velocity input gated by context (active motion, heading) | passive transport (Winter 2015; Terrazas et al. 2005, J Neurosci 25:8085). EVID: context steps PIL only (`CTXSTEP_*`) | 1 | step depends on context / token only |
| j | objects move the phase only as corrections | MEC weakly object-driven, LEC strongly (Deshmukh & Knierim 2011; Keene et al. 2016; Hargreaves et al. 2005, Science 308:1792); object-anchored position is a separate population (Hoydal et al. 2019, Nature 568:400). Counter-evidence ours: language asides need non-motion steps (`TW_NORMSTEP_RESULTS.md`, PH) | 2 | object steps 0 or correction-only / every token ticks |
| k | recurrence, bounded state, finite precision | CAN drift and capacity (Burak & Fiete 2009; Fiete 2008). Weak discriminator: every attention model stores all keys | 1 | recurrent settling / -- |
| l | time as a separate channel | time cells (MacDonald et al. 2011, Neuron 71:737); LEC experience time (Tsao et al. 2018, Nature 561:57); Laplace time code (Howard et al. 2014, J Neurosci 34:4692). EVID: rank >= D+1 lets a clock coexist with the map (03_formal Theorem 1b, DER); signed steps also learn recency (`RECENCY_RESULTS.md`, 1.000) | 1 | separate time code / time mixed into the map or no map |

Max 46. Weights are a judgement (DER at best); section 1.3 checks sensitivity.

### 1.2 Scores and measured performance

| encoding | a b c d e f g h i j k l | brain /46 | measured (status) |
|---|---|---|---|
| MapPoPE-Pair (32 angles) | 2 2 1 2 2 2 0 0 0 1 1 1 | **30** | rank 2, T=128: 0.9995, 16/16 SOLVED vs MapWM 10/16 (REG, `MAPPOPE_PAIR_RESULTS.md`) |
| MapPoPE (64 angles) | 2 2 1 2 2 2 0 0 0 1 1 1 | 30 | 0.9997, 16/16 (REG); torus 0.999 (`PAPER2X2_RESULTS.md`); Bach -0.0165 NLL vs MapWM (5/5); code "path hurts PoPE" UNMEASURED |
| MapPoPE + index decay envelope | 2 2 1 2 2 2 0 0 0 1 1 2 | 31 | envelope metric +0.281 on Dyck index row; path version collapses 2-D to 1-D; confounded with convergence (`CROSS_RESULTS.md`); never on navigation |
| MapWM + NormStep | 2 2 1 1 1 1 2 0 0 1 1 1 | 28 | +0.0107 new-object, 16/16 SOLVED vs MapWM descending (REG, `LEAK_RESULTS.md`); text world +0.006, unmeasured (REG) |
| MapWM + ActOnly | 2 2 1 1 1 1 1 0 0 2 1 1 | 28 | +0.0107 (REG) but needs an oracle action label; capped at 0.972 by asides in language (DirOnly, PH) |
| MapEM | 2 2 1 2 0 2 0 0 0 1 1 1 | 28 | torus fine; recency EM - WM -0.375, search (REG, `EM_WM_STATE.md`) |
| loop x4 (on MapWM) | 2 2 1 1 1 1 0 1 0 1 2 1 | 27 | Match-Query +0.346 unpaired, not reproducible paired (`REFINE_RESULTS.md`); under action noise +0.121 (8/8, `MQ_NOISE_2X2.md`) |
| grid variant (3 bands at 60 deg per module) | 2 2 2 1 1 1 0 0 0 1 1 1 | 26 | no hexagons (live negative); never run at matched long length |
| context step (CG / HSR) | 2 2 1 1 1 1 0 0 2 1 1 1 | 26 | PIL, n=1-2 per cell |
| MapWM | 2 2 1 1 1 1 0 0 0 1 1 1 | 24 | torus r2 0.971, r4 1.000 (REG); rank 2 per head 0-2/8 at T=1024 (REG) |
| Level 1.5 (InEKF) | 2 2 1 1 1 1 0 1 0 0 1 1 | 24 | stabilisation, not inference; flat under 13-cell drift (`L15_ABLATION.md`, `MQ_NOISE_2X2.md`) |
| hierarchy / hourglass | 2 2 0 1 1 1 0 0 0 1 1 1 | 22 | efficiency only (`ENWIK8_HIERARCHY.md`) |
| MapEM PosOnly (TEM-t-like) | 2 2 1 1 1 0 0 0 0 1 1 1 | 22 | new-object 0.976 (PIL, n=1) |
| monotone step (Abs / softplus) | 1 0 1 1 1 1 0 0 0 1 1 1 | 15 | 0/8, 1/8 at matched length (REG); fine as a clock (recency 0.96-1.00) |
| PoPE (index) | 0 0 1 1 2 2 1 0 0 0 1 1 | 15 | torus 0.679, below RoPE (-0.126) |
| RoPE (index) | 0 0 1 0 1 0 1 0 0 0 1 1 | 7 | torus 0.805 |
| *core proposal (MapPoPE-Pair + NormStep), unbuilt* | 2 2 1 2 2 2 2 0 0 1 1 1 | *34* | -- |
| *full proposal (section 3), unbuilt* | 2 2 2 2 2 2 2 2 1 2 1 2 | *44* | -- |

Scoring notes. MapEM e = 0: its content gain A_X is signed and inverts the kernel (4 of 16 heads all-negative,
`WHAT_WHERE_ANALYSIS.md` sec. 6). Index models j = 0: every token, objects included, advances the phase. MapWM d = 1:
the architecture lets content shift recall, trained models mostly do not move the peak (but see section 2.3). Level 1.5
j = 0: token identity sets an absolute phase, which on a redrawn map has no memory to stand on (`neuro_positional.md`
sec. 2). Loops h = 1: iteration is the only thing in this repo that helped under drift.

### 1.3 Reading
- **Sensitivity.** Equal weights (max 24): decay+MapPoPE 15, MapPoPE 14, MapWM+NormStep 13, loop 13, MapEM 12, MapWM 11.
  The ordering of built single encodings is unchanged. The decay row's lead is the time column only, for an
  envelope never run on navigation; it is not a measured encoding of a map and is not ranked.
- **What the scorecard cannot decide.** Four of twelve constraints (h, i, k, l) score 0-1 for every built encoding:
  the repo has no noisy integrator, no context-gated velocity on navigation, no bounded recurrent state and no
  separate time code. These are where a brain-faithful design must add parts, and where the evidence that they
  help *performance* is weakest or negative (section 3.3).
- **Brain score and performance agree at the top** (MapPoPE best on both); they disagree at MapEM (brain-pure gain,
  measured search cost) and at ActOnly (anatomically segregated, but wrong for language asides). Neither is a reason
  to doubt the ranking; both are reasons not to equate "biological" with "better".

## 2. The remap probe in neural terms, and why PoPE rescues rank 2

### 2.1 What the probe is (and is not)
`docs/audits/2026-10-05/remap_probe.py` (PH): pre-softmax score of the 4 action queries x 17 observation keys over
torus displacements d in [-16, 16]^2, phase built from the model's own action steps along a minimal path (intervening
observation steps excluded), medians over 8 seeds x 2 heads. The content x position interaction is split into
**gain** (scaling one shared kernel: rate remapping), **even** residual (width / shape about the same centre) and **odd**
residual (skew / shift). Caveats: pre-softmax (a field is post-softmax competition, so widths here are not attention
widths); minimal-path kernels are not what real sequences produce when steps do not cancel exactly (2.3 measures that
directly); **MapEM's gain share 1.000 and width_cv 0.000 are architectural** (A_X(a,o) x A_P(d) is exactly rank 1 in
content x position), not a learned property.

### 2.2 Neural reading per model (ANALOGY, PH)

| model | probe | neural counterpart | supported by |
|---|---|---|---|
| MapEM r4 | gain 1.000, peak at d = 0 on all pairs, height cv 0.34, width fixed | **pure rate remapping**: one place field, content scales its rate (and, being signed, can invert it -- no rate analogue) | Leutgeb 2005; Fyhn 2007 (fields stay); the sign inversion has no counterpart |
| MapPoPE r2 / r4 (64) | gain 0.69 / 0.50, even 0.23 / 0.40, odd <= 0.08; peaks at 0 on 100%; r(height, width) -0.59 / -0.68 | **rate remapping with width change**: content re-weights spatial scales, so a stronger field is a narrower one (weight moved to fine scales raises the peak above the window median and narrows it; DER) | partial: under directional rate remapping "average field sizes did not change very much (about 35%) ... compared to a 50% change in firing rates" (Navratilova et al. 2012, Front Neural Circuits 6:6, body sentence read via the article page) -- sizes do change, less than rates; the sign of the rate-size covariance is not reported there |
| MapPoPE-Pair r2 | gain 0.75, even 0.19, width cv 0.08, r(height, width) +0.09 | close to pure gain: at rank 2 with 32 angles the scale-reweighting freedom is barely used | -- |
| MapPoPE-Pair r4 | gain 0.57, even 0.37, r -0.41 | as MapPoPE | -- |
| MapWM r4 | gain 0.77, even 0.13, **odd 0.11** (largest), peak at 0 on 100% | **rate remapping plus field skew** (asymmetric fields that keep their location) | place fields become negatively skewed with experience, "symmetric at the beginning of a session, ... highly asymmetric with experience" (Mehta, Quirk & Wilson 2000, Neuron 25:707); there the skew is direction-dependent and plastic, MapWM's is nearly content-free (2.3) |
| MapWM r2 | peak at d = 0 on 7% of pairs, mean 13.7 cells away, width 199 cells | not a remapping type: the rank-2 frame is collinear or drifting (03_formal T1), so the minimal-path kernel aliases the origin | none; a learning failure |

A fourth type none of our models has: **module rescaling**. Novelty expands grid scale and, less, place fields, which
shrink back with familiarity (Barry, Ginzberg, O'Keefe & Burgess 2012, PNAS 109:17687). MapPoPE re-weights fixed scales;
Barry's data says the scales themselves move (omega plasticity per module). Place-field scale is graded along the
hippocampal long axis (Kjelstrup et al. 2008): per-scale gain is then "which dorsoventral level answers".

**Refined neural prediction N1** (extends `neuro_positional.md` N1; DER from the probe). Under rate remapping in a
fixed enclosure: MapEM-type predicts width unchanged and rate changes only; MapPoPE-type predicts rate and width change
together with *negative* covariance (rate up, field narrower) and the width changes shared by cells driven by the same
modules; MapWM-type predicts skew changes at fixed peak location. Published: sizes change less than rates
(Navratilova 2012); the covariance itself was not found (two searches; possibly new as an analysis of existing data).

### 2.3 Mechanism check: why PoPE's score rescues rank 2 (PH, new)
Checkpoints: the registered MAPPOPE_PAIR batch (paper torus, T=128, 300 ep, seeds 10-25 for rank 2, 10-17 rank 4).
Walks: 60 held-out trajectories (env seed 10000). Head class as in `03_formal.md` 1.3 without the lattice criterion
(T=128 has almost no wraps): A = drift (per-move common phase clk >= 0.05), B = collapsed frame (cond <= 0.25), ok.

| arm | heads ok / A / B | seeds with no ok head | of those SOLVED | retrieval top-1 (median, min) | same-cell coherence C, gaps 1-4 / 65-127 (median) | C0 (code alone), 65-127 |
|---|---|---|---|---|---|---|
| MapWM r2 | 10 / 10 / 12 | 11 | **6/11** | 0.940, 0.785 | 0.856 / 0.780 | 0.881 |
| MapPoPE-Pair r2 | 20 / 4 / 8 | 6 | **6/6** | 0.977, 0.960 | 1.000 / 0.978 | 0.979 |
| MapPoPE r2 (64) | 14 / 11 / 7 | 9 | **9/9** | 0.991, 0.959 | 0.999 / 0.978 | 0.979 |
| MapWM r4 | 16 / 0 / 0 | 0 | -- | 0.961, 0.939 | 0.875 / 0.869 | 0.947 |

(top-1 = fraction of revisit queries whose highest-scoring observation key is in the query's cell, best head;
C = sum_c A_c cos(kernel phase_c) / sum_c A_c over all same-cell query/key pairs, A = the pair's content amplitudes;
C0 = the same with the content phase psi, or PoPE's delta, removed.)

Findings (all PH, one batch, eval-only):
1. **PoPE does not rescue rank 2 by learning a clean code.** Defective heads are common in both PoPE arms
   (12/32 and 18/32 heads; MapWM 22/32). Seeds with at least one clean head: 10/16 vs 5/16 (Fisher p 0.16),
   7/16 vs 5/16 (p 0.72): not separated. The rescue is that **every defective PoPE seed solves** (15/15 pooled)
   where MapWM's solve on 6/11 (Fisher p 0.007; conditioned on a post hoc class, descriptive only).
2. **Retrieval is the failure.** Within MapWM r2, held-out accuracy tracks top-1 retrieval (Spearman +0.92, n = 16).
3. **MapWM shifts its kernel with a content phase that is almost content-free.** Amplitude-weighted dispersion of
   psi(a, o) across the 68 pairs is <= 0.08 on 30/32 rank-2 heads and <= 0.02 on all rank-4 heads: one offset per
   channel, shared by all contents -- a learned phase shift of the kernel, not content-specific remapping. It moves the
   kernel off the revisited cell's phase (C < C0 at every gap, r2 and r4) yet the models depend on it: removing it on
   observation keys only (action-key scores intact) costs -0.155 mean at rank 2 (14/16 seeds down, up to -0.44) and
   -0.022 at rank 4 (7/8 down) (`neuro_strip_obs_out.txt`). Removing it on all keys costs -0.426 at rank 2 but also
   breaks the action/observation type gating, so that number is not used.
4. **PoPE cannot shift, and does not need to.** delta sits at its bound 0 on 89-100% of channels; forcing every
   delta to 0 at eval changes accuracy by 0.000 on 31/32 rank-2 seeds and -0.010 on one. Its kernels stay centred on
   the revisited cell at all gaps (C = C0 = 0.90-1.00 up to 127 moves; median 0.98), including on drifting and collapsed heads.
5. **The derivation, checked against the code.** PoPE's score (`model_pope.py:61-70`) is
   `sum_c mu^q_c mu^k_c cos(theta_k,c - theta_q,c + delta_c)`, mu = softplus >= 0. With delta = 0 each term is
   <= mu^q_c mu^k_c, with equality iff theta_k,c = theta_q,c (mod 2 pi), so the sum is maximal at phase difference 0 and
   any other key reaches that value only if every weighted channel realigns (a tie). RoPE's
   `|q_c||k_c| cos(dtheta_c + psi_c)` peaks at dtheta_c = -psi_c. DER, exact. Two qualifications the earlier wording
   missed: the guarantee is in *phase* space -- a drifting code puts the true cell at a non-zero phase difference,
   so PoPE gains nothing there unless gains move to drift-free channels -- and exact ties (a collapsed frame) are not
   broken by PoPE either.

**Reading (CONJ).** At rank 2 the learned code is often defective (T1). MapWM has two levers to fit the training
loss -- per-channel gain and a per-channel phase shift -- and uses the shift to sculpt a skewed kernel that separates
the true cell from aliases; this works on some seeds and stalls on others (the shift is a fragile, finely tuned
solution). PoPE removes the shift; the remaining lever is non-negative gain over channels, which can only *select*
channels coherent at revisits, and selection is enough at T=128. In brain terms: **select scales, never shift them**
-- rate remapping as the only form of content influence makes the map easier to learn, which is the "anatomy as prior"
reading of `neuro_positional.md` 3.2, now with a mechanism candidate. Test: E2 below (scalar vs per-channel gain;
signed magnitudes), and SCORE_RANK at T=1024 (being run separately), where long gaps should expose drift that gain
selection cannot remove (prediction: PoPE rescues collapsed-frame seeds less than drifting ones only if drift-free
channels exist in the head; a head-class census of SCORE_RANK checkpoints decides it, CPU, minutes).

## 3. Design: the gain-phase map (GPM)

### 3.1 Equations
Per head h, channel c (32 per head, pairwise angles), tokens x_t with embedding e_t:

- **Read-in (NormStep):** `Delta_t = W_out^h W_in^h LN(e_t)` (`model_codes.py:138` MapWM_NormStep). Rank
  r_h = D + 1 per head by default; r_h = D if SCORE_RANK shows PoPE's score solves rank D at T=1024 (biology's velocity
  input is D-dimensional, head direction x speed, and a module lives on a D-torus, Gardner 2022: D+1 is a learnability
  margin, not a brain feature; `02_literature.md` 2a).
- **Phase (signed path integral):** `theta_t,c = omega_c * sum_{u<=t} Delta_u,c` (`model.py` PathIntegrator).
- **Score (PoPE, pairwise, delta frozen at 0):** `a_ts = sum_c g^q_t,c g^k_s,c cos(theta_t,c - theta_s,c) / sqrt(d_h)`,
  `g = softplus(W LN(x))` (`model_pope_pair.py` with `pope_delta` frozen). Values, softmax, FFN unchanged.

That is the **core** (34/46). Optional components, each entering only through its ablation:

- **Modules (c):** channels grouped into M = 4-6 modules, `omega_c = omega_m`, `omega_m = omega_1 rho^-(m-1)`, rho ~ 1.4,
  each module a D-dim velocity frame read by 3 band directions 60 deg apart (`model_grid.py`: ActionToLie2D,
  GridPathIntegrator; `model_fixed_omega.py` for frozen omega). Per-module learned omega allowed (Barry 2012 rescaling).
- **Gain granularity (d, f):** per-channel g (core) vs per-module g vs one scalar g per token pair (pure gain,
  MapEM-like but non-negative). The brain prior (rates change more than sizes) favours the coarser forms.
- **Noise + memory reset (h):** train and evaluate with action-record noise (`p_action_noise`, the `NOISE_REFINE.md`
  regime) or wall bumps (`boundary="wall"`: a recorded move that does not happen, a native path-integration error
  with a biological counterpart, border-driven correction, Hardcastle 2015). Reset: at observation tokens, the layer's
  own PoPE attention restricted to earlier observation keys gives
  `phi_t,c = arg sum_s alpha_ts exp(i theta_s,c)`, innovation `eps_t,c = wrap(phi_t,c - theta_t,c)`, confidence
  `kappa_t = sigmoid(w . [max_s alpha_ts, entropy(alpha_t)])` (init near 0), and the correction **enters the step**:
  `Delta'_t,c = Delta_t,c + kappa_t eps_t,c / omega_c`, so it propagates to every later phase (an integrator reset,
  not a per-token offset). Two passes (integrate, retrieve, re-integrate) keep the parallel scan.
- **Time channel (l):** one head with an index (or monotone-step) phase; content otherwise identical.
- **Context-gated step (i):** HSR (`model_context_step.py`) for language only; navigation gives action tokens.

### 3.2 Minimal ablation set (each component: brain constraint with strong evidence AND not hurting, or shown to help)

| component | brain case | performance case today | keep / test | ablation that would justify it |
|---|---|---|---|---|
| signed path phase | a, b (w 3+3) | REG: +0.243 position, sign -0.177 | keep | done |
| PoPE score | d, f, e | REG: SCORE +0.024, 16/16 vs 10/16 | keep | split the bundle (E2) |
| delta frozen at 0 | d (peak invariance exact) | PH: delta inert (<= 0.010) | keep (simplifies; no cost expected) | included in E2 as the arms' default; one arm with free delta |
| NormStep read-in | g, j | REG: +0.0107 new-object; text world null | keep | compose with PoPE score (E4) |
| rank D+1 | none (D is biological) | REG: rank 2 0-2/8 vs 4 8/8 at T=1024 | keep until SCORE_RANK | SCORE_RANK (other agent) |
| modules at 1.4 | c (w 2) | angle count null (REG); frequency learning null (n=3) | test | E6, non-wrapping arena |
| per-module / scalar gain | d (Navratilova) | none | test | E2 |
| noise + memory reset | h (w 2, strong LIT) | four negatives | test, expectation low | E5 |
| time channel | l (w 1) | signed steps already learn recency | drop on navigation | E8 only on a map + time task |
| context step | i (w 1) | PIL | language only | outside this plan |

### 3.3 What our negatives say about the optional parts
- **Correction (Level 1.5 / InEKF, PC, LoopedRefine).** Three separate failures. (i) Level 1.5's measurement is a fixed
  function of token identity; on a redrawn map a token carries no absolute position, so it cannot localise (DER,
  `neuro_positional.md`); it was flat under 13 cells of drift (`MQ_NOISE_2X2.md`). (ii) LoopedRefine corrected theta
  from the hidden state under action noise and gained nothing (refine - fixed -0.011 to +0.006, `NOISE_REFINE.md`);
  its correction was a bounded per-token offset, deliberately NOT propagated. (iii) PC and Kalman are duals (no new
  computation). Where the reset differs: its target comes from retrieved memory (an earlier phase at which the same
  content was stored), and it propagates like a biological reset of the integrator. Where it does not: attention
  already re-localises implicitly (the loop's +0.121 under noise), so a reset may add nothing. Honest prior: the
  "no noise -> nothing to correct" excuse is already refuted for Level 1.5 (noise present, still flat); it survives
  only for memory-sourced, propagated resets, which have never been built. Expect a negative; register a kill.
- **Modules / hex emergence.** No lattice emerged (live negative), consistent with a signed code (Dorrell, Sorscher
  conditions absent). Architected modules make no emergence claim. One new risk (DER): fixed omega at ratio 1.4 is
  generically off the 2 pi / N lattice that wrap-only revisits demand on an N-torus (`03_formal.md` Theorem 2), so modules
  should hurt on the paper torus and be neutral on a bounded or large non-wrapping arena. The torus is the
  unbiological part; test on walls.
- **Hierarchy.** Temporal pooling is not multi-scale space (grade C/D); omitted.
- **Time channel.** Recency is solved by signed steps alone (`RECENCY_RESULTS.md`), so on these tasks a separate
  clock has no performance case. The rank theorem gives one: at rank >= D+1 a clock can live off the map axes;
  at rank D it must be zero (T1). A dedicated time head is the clean way to let rank D hold.

## 4. Ranked experiment plan (cheapest first)

Costs scaled from: T=128 paper torus 72 runs ~2.5 h on two 4090s (~2.1 min per run); T=1024 ~80 min per run at 8
concurrent (8 runs per 80 min); T=512 assumed ~half that. Power from the pair batch: MapWM r2 sd 0.039, PoPE arms sd
<= 0.001. Every batch: pilot on outside seeds, a blind code-verification agent, amendments before reading (rule 29).

| # | experiment | arms; task; length; n | cost | registered readout | what would change the conclusion |
|---|---|---|---|---|---|
| E1 | **CPU** extensions of this note's checks: (a) head-class census + same-cell coherence on SCORE_RANK checkpoints when they exist; (b) the same on `runs/leak` (NormStep vs MapWM, T=1024); (c) kernel grid score per frequency group (M5) | eval-only | minutes each | descriptive | (a) if PoPE's T=1024 rank-2 solves have only clean heads, "gain tolerates a defective code" is wrong at long T |
| E2 | **Split the score by brain ingredient** at the sensitive point (rank 2, T=128): MapWM r2 (control, expect ~10/16); MapPoPE-Pair softplus per-channel (core score); Pair with signed magnitudes (NoSigma: content phase in {0, pi}); Pair with ONE non-negative gain per token pair (`g^q_t g^k_s sum_c cos`, pure gain) | paper torus, T=128 train = test, 300 ep cosine, 16 seeds each | 64 runs, ~2.2 h + port of `model_pope_ablate.py` arms | SOLVED (Fisher vs MapWM and vs core) primary; T=128 accuracy (perm) | NoSigma = core: the operative ingredient is "no continuous content phase", not non-negativity; scalar gain = core: per-scale width freedom unused, adopt pure gain (closer to rates > sizes); scalar gain < core: width change is needed (supports MapPoPE-type N1) |
| E3 | SCORE_RANK (separate agent): does PoPE's score rescue rank 2 at T=1024 | -- | -- | as registered there | fires: core uses rank D (biological); null: keep D+1 and say rank is a learnability margin |
| E4 | **Compose the core** on the new-object task: MapPoPE-Pair r4; Pair r4 + NormStep (core); MapWM r4 + NormStep (reproduces LEAK) | new-object, T=1024, 900 ep, 8 seeds each | 24 runs, ~4 h | unseen-object accuracy x1 (perm), SOLVED | core < Pair + 0: NormStep and PoPE interfere; core = both: fixes are redundant (both act through convergence speed, r(loss, acc) -0.94); core > both: composition justified |
| E5 | **Noise x memory reset**: core; core + reset; core + loop x4 (the known partial remedy); at p_action_noise 0 and 0.1 | paper torus, T=128, 8 seeds | 48 runs, ~1.7 h (+ build and verify the reset, ~1 day) | interaction (reset - core at 0.1) - (reset - core at 0) > MDE (~0.03 at n=8, from NOISE_REFINE sds); secondary reset - loop | interaction <= 0 or reset <= loop: a fifth correction negative; drop h from the design and state that attention re-localisation makes an explicit reset unnecessary in silico. A positive is the first correction gain in the project; then repeat at T=512 (~3.5 h) and with wall bumps |
| E6 | **Modules on a non-wrapping arena**: core with dense learned ladder; fixed modules (M = 5, rho 1.4, 3 bands); learned per-module omega | `boundary="wall"` grid 64 or large torus with near-zero wrap share (CPU gate first: wrap share, n-gram floor), T=512, 8 seeds | 24 runs, ~2-4 h | held-out accuracy at training length (perm) | fixed modules within MDE of the dense ladder: adopt (fewer parameters, brain-faithful); worse: modules are a capacity cost at this scale, keep the ladder and say so. Run the same arms on the 64-torus as a secondary: predicted loss on wrap-only revisits (Theorem 2) |
| E7 | gain granularity, if E2's scalar arm loses: per-module gain | as E2, 16 seeds, 2 arms | 32 runs, ~1.1 h | as E2 | per-module = per-channel: module-level gain suffices |
| E8 | time head, only on a task mixing return-to-place and recency (not built) | -- | design first | -- | -- |
| N1 | rate-remapping data: covariance of peak-rate change and field-width change, per cell, in an existing rate-remapping dataset | no GPU | analysis | sign of r(rate change, width change) | negative: MapPoPE-type; ~0: MapEM-type pure gain; centroid shifts: MapWM-type |

## 5. Novelty and what not to claim
- The full design is close to an attention form of **Vector-HaSH** (fixed modular grid scaffold, velocity-updated
  phases, hippocampal recall that error-corrects; Chandra et al. 2025, Nature 638, abstract per `docs/lit/LIT_WHAT_WHERE.md`)
  and **TEM-t** (Whittington, Warren & Behrens 2022). What may be new: a learned, path-integrated rotary phase with a
  gain-only (PoPE) score as the rate-remapping step; the mechanism in 2.3 (shift vs gain as the two levers for fitting a
  defective code). Not found in `docs/lit/*.md` or `papers/` (grep), not a literature search.
- Do not claim: that MapPoPE is a circuit model; that PoPE's score improves the learned code (2.3 says it does not,
  at T=128); that the brain needs rank D+1; hexagon emergence; any of section 2.2's neural correspondences as
  supported by our data (they are analogies); the defective-seed Fisher p 0.007 as a test (post hoc conditioning).

## Sources read for this note (abstract at least, via Europe PMC / publisher pages)
Navratilova, Hoang, Schwindel, Tatsuno & McNaughton 2012, Front Neural Circuits 6:6 (abstract; one body sentence on
field size vs rate); Mehta, Quirk & Wilson 2000, Neuron 25:707-715 (abstract); Barry, Ginzberg, O'Keefe & Burgess 2012,
PNAS 109:17687-17692 (abstract); Kjelstrup et al. 2008, Science 321:140-143 (abstract); Solstad, Yousif & Sejnowski
2014, PLoS Comput Biol 10:e1003648 (abstract; CA3 attractor model of rate remapping, not cited above). All other
citations are those of `neuro_positional.md` (Sources) and `02_literature.md`.
