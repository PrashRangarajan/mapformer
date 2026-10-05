# Formal theory for the measured effects (2026-10-04, review agent 3 of 5: formal)

Read-only review. Nothing here is registered; every CHECK is post hoc, eval-only, on committed checkpoints (CPU),
except one toy training batch (section 1.9) on synthetic data. Scripts and raw outputs are in the appendix.

Status labels, used on every claim:
- **THEOREM** -- proved here (all are elementary linear algebra / inequalities; no claim of novelty for the math itself).
- **DERIVATION** -- follows under the stated assumptions; the assumptions are the weak point.
- **CHECK** -- measured here on committed checkpoints; post hoc, n as stated, no pre-registration.
- **CONJECTURE** -- a guess with a test attached.

Prior art used, not re-derived: the cumsum/abelian frame and the clock-vs-map table are `mapformer_math.tex` (sec. 2, 6.2);
content-dependent rotation is published (Mamba-3 2603.15569 Prop. 3, Selective RoPE 2511.17388, GRAPE 2512.07805);
sign = cancellation is Sarrof 2405.17394 / Grazzi et al.; "counting needs a layer" results are in the formal-language
literature (Bhattamishra et al. 2020; Weiss et al. 2021 RASP; Yao et al. 2021 Dyck; Sanford, Hsu, Telgarsky 2023/2024
one-layer limits). **The last group is NOT in `papers/`; check before citing.** The S^1-vs-S^2 synchronisation
analogy in 1.8 (Markdahl et al. 2018; Wiley-Strogatz-Girvan 2006) is likewise from memory, not the corpus.

## 0. Results in one table

| # | claim | status | what it predicts | cheapest test |
|---|---|---|---|---|
| 1 | At per-head rank r = D, any per-move common step c is, in every channel, a phantom DRIFT of the decoded position; a drift-free head must either have c = 0 exactly or lose a spatial dimension. At r >= D+1 a full-rank drift-free channel set exists for any generic c. | THEOREM | rank-D failures are of two kinds only (drift / collapsed frame); failure is per head; independent of D, N, channel count | done here on 80 checkpoints (below); GPU: rank 2 with odd action steps and zero observation steps (a CPU toy of this went the wrong way, 1.9) |
| 2 | "SOLVED iff at least one head is clean (full-rank frame, no drift, on-lattice)" classifies 79/80 matched-length rank runs; rank-D heads are 57 drift (A), 13 collapsed (B), 2 off-lattice (C), 8 clean. In every B head the latent frame collapsed (median cond 0.00) with c small (median 0.04 of the frame). | CHECK | -- | -- |
| 3 | The rank-2 defect already exists at T=128 (32/32 heads drifting or collapsed, at training loss 0.001-0.39) while r=4 learns full-rank, drift-free frames (off-lattice only: no wraps in training). | CHECK | robustness and matched-length deficits share one cause | done |
| 4 | Clean rank-D+1 heads at T=1024 end with c REMOVED (|c|/|U| 0.002-0.004), not parked off the map plane; Theorem 1b's parking route is at most transient. | CHECK (partly negative for my own mechanism) | trajectories should show parking mid-training | toy / snapshot run |
| 5 | Head-count model P(solve) ~ 1 - (1-q)^H, q = per-head clean rate (rank D: 8/80 = 0.10; pooled rank-D solve rate predicted 0.19, observed 6/40) | DERIVATION | at per-head rank 2: H=4 -> ~0.34, H=8 -> ~0.57 (upper bounds: heads co-fail) | GPU, 3 arms x 8 seeds |
| 6 | Wrap-only revisits are exactly the revisits that close a non-contractible loop; they are the only ones that constrain each channel's frequency to the lattice (2 pi / N) Z^D, and the only ones with gap >= N moves, so they carry both the lattice constraint and the largest drift exposure. | THEOREM | -- | -- |
| 7 | The small-grid failures (2H memorised, 3H partial) are a fixed-map memorisation race, not a wrap limit. | DERIVATION | with the map redrawn per sequence, rank D+1 solves the 10-per-side tori | GPU, 2 cells x 8 seeds |
| 8 | The text-world eval/train-mode gap is the SCALE of inverted attention dropout: deterministic x1/(1-p) recovers it on all 6 gap runs (e.g. 0.828 -> 0.993), dropout noise without the scale makes it worse (0.75). Registered text-world path accuracy is under-reported (40-walk mean 0.971 -> 0.994); RoPE arms unaffected. | THEOREM + CHECK | any path model with peaked attention and small residual margin | re-score committed evals with x1/(1-p) (minutes) |
| 9 | Leak cost is a gap-accumulated phase error: 0 below 128 moves, 0.003-0.018 at 128-511, 0.047-0.122 at >= 512 moves (8/8 MapWM seeds). | DERIVATION + CHECK | the leak (and NormStep's in-distribution gain) vanish at short T | done; GPU: leak arms at T=256 |
| 10 | Asides: with zero steps on non-direction words a 1-layer model cannot tell an aside noun from a located noun at the same cell by phase; learned-step models separate them by a small coherent phase offset (0.06-0.33 rad) that the steep kernel turns into a 1.4-11 logit gap. The registered "per-word clock" of TW_NORMSTEP (verdict B) is almost entirely the aside sentences (aside-only drift 0.03-0.18 rad vs adverbs/fillers 0.008-0.035). | THEOREM + CHECK | grammar whose asides use only shared words defeats context-free steps | GPU text-world variant |
| 11 | Depth exchange: one path layer = a free prefix-sum (counter) + mod-N phase; an index transformer must spend >= 1 layer on counting (uniform attention, resolution 1/t) and an MLP on the mod. | THEOREM (upper bound) + CONJECTURE (lower bound) | index-2L shortfall grows with T/N; path-1L flat | GPU sweep of T at fixed N |
| 12 | A minimal CPU toy (map redrawn per sequence) does NOT reproduce the rank split (r=2/3/4 solve 3/5/3 of 8), and forcing c = 0 at rank 2 made it worse (1/8): no causal support for Theorem 1 from the toy. | CHECK (negative) | -- | -- |

---

## 1. Rank D versus D+1 on a D-torus

### 1.1 Setup (exact for a 1-layer MapWM)
Per head h, latent z_t = sum_{u<=t} u(x_u) in R^r (u(x) = W_in^h e(x)); channel i has frequency row f_i = omega_i w_i in R^r
(w_i a row of W_out^h); phase phi_i(t) = <f_i, z_t> (`model.py:83,120`; `mapformer_math.tex` eq. factor). Torus Z_N^D,
actions +-e_j, strict alternation action/observation.
Per-move latent increment = u(a) + u(o) = U e_a + c + eta(o), where
- U in R^{r x D}, U_j = (u(+e_j) - u(-e_j))/2: the antisymmetric ("map") frame,
- c = mean over actions of u(a) + E_o u(o): the per-move common component ("clock"),
- eta(o) = u(o) - E u(o): the leak (set to 0 in this section; section 4).
After n moves with unwrapped displacement p: z = U p + n c. Channel i reads phi_i = <k_i, p> + n tau_i with
k_i = U^T f_i (phase per unit displacement) and tau_i = <f_i, c> (phase per move).

**Remark 1.0 (THEOREM: no expressivity difference on actions).** The action-only code is the matrix K = [k_i] = F U
(F = rows f_i). For r >= D, every K in R^{nb x D} is reachable (U invertible on its range, F free). So r = D and r = D+1
parameterise the same set of action codes; any difference must come through c (or eta). This is why a rank-2 projection
of a solved r=4 model works (0.9955, `RANK_PROJ_RESULTS.md`): solved r=4 heads have c ~ 0 (CHECK 1.3, `drift_park.py`), so nothing is lost.

### 1.2 Theorem 1 (clock visibility)
Call channel i *drift-free* if tau_i = 0, and a head *clean* if its drift-free channels resolve all D directions
(rank of {k_i : tau_i = 0} is D).

(a) **r = D.** If U is invertible then c = U beta for the unique beta = U^{-1} c, and tau_i = <k_i, beta> for every i:
in every channel the clock is a phantom displacement of beta cells per move, the same beta for all channels.
The drift-free channels satisfy <k_i, beta> = 0, so they span at most beta-perp, a (D-1)-dim subspace, unless beta = 0.
Hence: **a rank-D head is clean iff c = 0 exactly.** If c != 0 the head is either
(A) full-rank but drifting (every localising channel set reads beta), or
(B) drift-free but collapsed (its channels resolve only beta-perp: rank <= D-1 code; in 2D, collinear axes).
If U is singular the head is already collapsed.

(b) **r >= D+1.** If c is not in range(U), the hyperplane H = c-perp in R^r satisfies U^T(H) = R^D
(proof: ker U^T = range(U)-perp is not inside H because c is not in range(U), so H + ker U^T = R^r and
U^T(H) = U^T(R^r) = R^D). So drift-free channels f_i in H can realise ANY full-rank K: a clean head exists with the
clock still present, and the clock can live in other channels as a separate time coordinate.

Proof: linear algebra as written. Per-HEAD because each head's channels read only that head's r-dim image; total latent
size does not enter. This is exactly the RANK_SEP split (B and C_bd: per-head 2, fail; C and D: per-head 4, solve;
total latent 4 on both sides).

Corollaries (all THEOREM given the setup):
- the obstruction does not depend on D, on N, or on the number of channels per head;
- at rank D the identity tau = K beta holds exactly, so the part of tau outside span(K) is 0 for every rank-D head
  (used below as a probe self-check: printed 0.000 on all rank-D heads);
- monotone steps (Abs, softplus) make u(+a) = u(-a), so U = 0 and the code is all clock: the sign result
  (`SIGN_MATCHED_RESULTS.md`) is the degenerate end of the same algebra (published as Sarrof/Grazzi; not new).

### 1.3 CHECK: the three failure types on every matched-length rank checkpoint
Probe (`probe_winding2.py`): per head, phase steps phi(x) = omega * Delta(x); frame K (nb x D) weighted by the
channel's content amplitude w_i ~ mean_{a,o} |q_b(a)||k_b(o)|; cond = s_min/s_max of the weighted frame;
clk = |w^.5 tau| / |w^.5 K|_F; wind = weighted mean distance of N k_ij / 2 pi to the nearest integer (random 0.25).
Class (thresholds fixed before the cross-tab): clean = wind < 0.05 and cond > 0.25 and clk < 0.05; else A if
clk >= 0.05, else B if cond <= 0.25, else C (off-lattice).

| set (T=1024, 900 ep) | per-head rank | SOLVED | heads clean / A / B / C | per seed (head0 head1, * = SOLVED) |
|---|---|---|---|---|
| torus G64, shared r=2 | 2 = D | 0/8 | 1 / 13 / 1 / 1 | AA Cc AA BA AA AA AA AA |
| torus G64, per-head r=2 | 2 = D | 2/8 | 3 / 9 / 4 / 0 | AA Ac* cc* AA AA BB BB AA |
| torus G64, block-diag r=4 | 2 = D | 2/8 | 2 / 7 / 7 / 0 | AA BA cB* AB AB BA cA* BB |
| torus G64, per-head r=3 | 3 = D+1 | 6/8 | 11 / 3 / 1 / 1 | AA cc* cc* AB cc* cC* cc* cc* |
| torus G64, shared r=4 | 4 | 8/8 | 15 / 0 / 0 / 1 | all with a clean head |
| torus G64, per-head r=4 | 4 | 8/8 | 14 / 0 / 0 / 2 | all with a clean head |
| ND D2 G32, r=2 | D | 1/8 | 1 / 14 / 1 / 0 | AA AA cA* AA AA AA BA AA |
| ND D2 G32, r=3 | D+1 | 8/8 | 15 / 0 / 0 / 1 | all with a clean head |
| ND D3 G10, r=3 | D | 1/8 | 1 / 14 / 0 / 1 | AA Ac* AA AA AA AA AA CA |
| ND D3 G10, r=4 | D+1 | 4/8 | 5 / 10 / 1 / 0 | Ac* AA AA Ac* cc* AB Ac* AA |

- **Rule "SOLVED iff some head is clean": 79/80 runs.** The exception is shared r=2 s1 (final loss 0.051, on the 0.05
  line, one clean head). This is close to a definition of solving, so its value is the decomposition of failure, not
  the rule.
- **Rank D: 57 drift heads, 13 collapsed, 2 off-lattice, 8 clean (80 heads).** Exactly Theorem 1(a)'s two alternatives.
- **Anatomy of B (`drift_B.py`), 13/13 heads:** latent frame cond(U) 0.00-0.22 (median 0.00): both axes on one latent
  direction; readout cond 0.02-0.16; |c|/|U| median 0.04 (small); the residual c lies along the readout's least-read
  latent direction (|cos| 1.00 on 13/13). CAUTION, found on re-reading: this alignment is what gradient pruning gives
  after a collapse of ANY cause (the component of c along the read direction is a visible drift and is removed; the
  unread component gets no gradient), so it does not show that the clock CAUSED the collapse. B is Theorem 1(a)'s
  branch as a state, not as a mechanism. In A heads c is large (|c|/|U| median 0.62) and also sits mostly in the
  least-read direction (|cos| median 0.92): the visible part has been pruned and the rest cannot be hidden at rank D.
- **Rank D+1 failures in 3D G10 are A heads with part of c outside the map plane** (`drift_park.py`: |c|/|U| 0.51,
  outside-plane 0.16, readout still sees c at 0.48 of the map): the parking Theorem 1(b) allows exists but was not taken.

### 1.4 CHECK: the defect predates long training (T=128 checkpoints)
`classify_t128.py` on `runs/paper2x2/p0` and `runs/rank_sweep/p0` (T=128, grid 64, essentially no wraps):

| set | heads clean / A / B / C | final training loss |
|---|---|---|
| paper2x2 r=2 | 0 / 7 / 9 / 0 | 0.002-0.385 |
| rank_sweep r=2 | 0 / 6 / 10 / 0 | 0.001-0.089 |
| paper2x2 r=4 | 1 / 0 / 0 / 15 | 0.000 |
| rank_sweep r=4 | 0 / 0 / 0 / 16 | 0.000 |

Rank 2 at T=128 is drifting or collapsed on 32/32 heads while fitting the training task; rank 4 is full-rank and
drift-free on 31/32 and only off-lattice (nothing at T=128 asks for the lattice). So the old "r=4 helps only out of
distribution" (`RANK_SWEEP.md`) and the matched-length deficit are **one defect read at two lengths**: a drifting or
collinear code is harmless while gaps are short and fatal when they are long (1.6).

### 1.5 CHECK (negative): initial conditioning does not predict success
Rebuilding every run's initial weights (`probe_init.py`; `torch.manual_seed(seed)` then the constructor, as
`train_variant.py:499,550`; env construction draws no torch RNG): initial per-head frame cond and clock ratio do not
separate solved from unsolved seeds (e.g. per-head r=2 s1: cond 0.69/0.47, solved on the G64 torus and unsolved on ND G32
from the same init). The random-matrix argument "square frames are near-singular with density linear at 0, r = D+1
quadratic" is therefore not the account. The defect is made by training.

### 1.6 DERIVATION: why the defect costs only at long gaps
Assume one retrieval head, drift beta per move, revisit gap g. The score of the true key relative to its value at
beta = 0 is sum_i A_i [cos(g <k_i, beta>) - 1] ~ -(g^2/2) sum_i A_i <k_i, beta>^2 for small g|k||beta|; a collapsed
head aliases cells along beta-perp whose separation in the 1-D code scales like the packing bound of
`mapformer_math.tex` (exponent D/rho with rho the LEARNED effective rank, 1 in 2D B heads -- the geometric account
was withdrawn for the architecture's rank, but it applies to the learned one). Both errors grow with the
displacement/gap range, i.e. with T. At T=128 the gap range is short; at T=1024 it is not. Consistent with:
rank-2 short-gap revisits fine and gap >= 128 / wrap-only revisits near floor (`RANK_MATCHED_RESULTS.md`: 0.671 / 0.557
vs r=4 0.988 / 0.970). Not derived: why training at rank D does not drive c to 0 (open; 1.9).

### 1.7 DERIVATION: head count
If each head ends clean independently with probability q, P(solve) = 1 - (1 - q)^H. Measured q (clean heads / heads):
rank D 8/80 = 0.10 (0.06-0.19 per set) -> predicted 0.19 at H=2, observed 6/40 = 0.15; torus r=3 11/16 -> 0.90 vs 6/8;
ND D2 r=3 15/16 -> 1.00 vs 8/8; ND D3 r=4 5/16 -> 0.53 vs 4/8; r=4 on 2D 29/32 -> 0.99 vs 16/16. Heads co-fail (torus
r=3: 2 seeds with both heads bad where independence predicts 0.8), so the formula is an upper bound.
**Prediction:** at per-head rank 2 (d_model 128 fixed, so nb per head shrinks; Theorem 1 does not depend on nb),
H = 4 solves at most ~1 - 0.9^4 = 0.34 and H = 8 at most ~0.57, against 0.15 at H = 2. A flat result at H = 8 would say
the per-head failures are driven by a shared cause (embeddings, data) rather than by each head's search.

### 1.8 What Theorem 1 suggests about the search, and its limits (CONJECTURE)
Theorem 1(b) offers a route at r >= D+1: build a clean map in c-perp while the clock is still there, then let the clock's
channels lose amplitude. `drift_park.py` shows every clean D+1 head at T=1024 has c removed (|c|/|U| 0.002-0.004, readout sees
0.000 of it), so if the route is used it is transient. The analogy I would test: synchronisation on S^1 has
stable twisted states that cannot unwind (pi_1 = Z) while on S^n, n >= 2, they can (Markdahl et al.; check before
citing). Here the "extra dimension" lets a head move between codes without passing through a drifting or collapsed
one. Test: snapshots every 25 epochs of 4 r=3 and 4 r=2 seeds (the rank_proj review already did this for 2 seeds),
reading clk and the out-of-plane part of c over training: the route predicts out-of-plane c early in solved r=3 runs.

### 1.9 Toy check of the causal claim (synthetic, CPU)
Minimal 1-layer model (`toy2.py`: no FFN, d 32, 2 heads, 8 channels per head, per-head rank r, map REDRAWN per
sequence so memorisation is impossible, D2 torus N 16, T 192 moves, wrap-only share 0.39, 8000 Adam steps, 8 seeds).
Each run took ~9 min of CPU (over the 5-min guideline; I let them finish rather than kill them).

| arm | SOLVED (final loss < 0.05) | acc | wrap-only | other |
|---|---|---|---|---|
| r=2 | 3/8 | 0.934 | 0.868 | 0.978 |
| r=2, odd action steps + zero observation steps (c = 0 by construction) | 1/8 | 0.823 | 0.671 | 0.923 |
| r=3 | 5/8 | 0.938 | 0.890 | 0.970 |
| r=4 | 3/8 | 0.937 | 0.888 | 0.969 |

**Negative on both counts.** (i) The toy does NOT reproduce the rank split at this scale and budget (r=3 vs r=2 5/8
vs 3/8, Fisher p 0.62; r=4 3/8), so it cannot test Theorem 1 as a cause; every arm is budget-limited and fails on
wrap-only revisits. (ii) Removing the common component by construction did not help rank 2 -- it was worse
(1/8 vs 3/8, Fisher p 0.57, unmeasured), and its failures are off-lattice frames (winding residual 0.17-0.29),
not drift. Confounds: the arm also deletes observation steps (and with them the per-move gauge) and changes the
gradient path through the step map. Read: in this toy the rank-D obstruction of Theorem 1 is not what limits
learning; the lattice is. Whether it limits the full model at T=1024 is open, and prediction 1.10.1 should be run
with that expectation lowered. (An earlier 3000-step version with the repo's MapFormerWM layer, d 32, map redrawn,
`toy.py`, gave r=2 2/8, r=3 3/8, r=4 3/8: also no split.)

### 1.10 Predictions and cheapest tests (ranked)
1. **Odd action steps, no observation steps at per-head rank 2** (c = 0 by construction: u(-a) := -u(+a), u(o) := 0;
   the oracle part is the observation mask, as ActOnly). Theorem 1 says the drift branch vanishes; if rank 2 then
   solves at the r >= 3 rate, the account is confirmed as cause; if not, the obstruction is elsewhere (frame
   learning, lattice). The toy (1.9) went the wrong way (1/8 vs 3/8), so expect a negative. 16 GPU runs, T=1024, 900 ep.
   A cleaner variant that keeps the per-move gauge: penalise only |c| (item 2).
2. **Penalty lambda |c|^2** on the per-move common component at rank 2 (no oracle). Same prediction, weaker.
3. **Heads** (1.7).
4. **D-independence:** rank D per head should fail at a similar per-head clean rate (~0.1) at D = 4, 5 on large tori;
   rank D+1 should succeed whenever the torus is large enough that wrap-only revisits are a minority. Consistent with
   the two D's measured (q = 0.06 at D2 G32 and D3 G10).
5. **Channel count:** varying d_head at fixed per-head rank should not move the rank-D failure rate.
6. What is NOT explained: why loop x4 and 4 real layers partly rescue rank 2 (`LOOP_RANK_RESULTS.md`): the phase is
   computed once from embeddings, so depth cannot remove drift from it; a later layer must compensate in the content
   path. Unmeasured; a head-type census of those checkpoints is the cheap first look.

---

## 2. Wrap-only revisits, small grids, memorisation

**Theorem 2 (what each revisit constrains).** A revisit (s, t) has unwrapped displacement N m, m in Z^D.
Non-wrap (m = 0): the phase difference is sum over the interval of the per-move increments = (t - s) tau (drift) only;
it is satisfied by ANY frame K. Wrap-only (m != 0, the cell never seen before at the same unwrapped position): the
interval is a loop with non-zero class in H_1(T^D) = Z^D, and the phase difference is N <k_i, m> + (t - s) tau_i, which
vanishes for every m only if N k_i in 2 pi Z^D (integer winding per channel and axis) and tau_i = 0. Also t - s >= N |m|_1.
So (i) wrap-only revisits are the ONLY source of the lattice constraint; (ii) they are the longest-gap revisits, with
drift error >= N |m|_1 |tau_i|. Both failure mechanisms of section 1 load on them. (Proof: as written.)

Consequences. At T=128 on grid 64 the lattice is never asked for, so r=4 learns off-lattice frames (type C at 31/32
heads, 1.4) and everything wrap-only fails out of distribution -- the T=128 "wrap" stratum below floor for both arms
(`RANK_MATCHED_RESULTS.md` item 4) is Theorem 2(i). The tolerance on the lattice is |k_i - lattice| < ~1/(N|m|) per
channel: the lattice spacing and the tolerance both scale as 1/N, so grid size alone does not make alignment harder.

**DERIVATION 2.1 (memorisation race).** Training uses one fixed map per seed (`train_variant.py` constructs the env
with `seed=args.seed`; ND env the same). Two exact minima of the training loss exist: the in-context map (transfers)
and the memorised map plus localisation (does not). Rough rates: visits per cell per epoch
V = 98 x 16 x 1024 / n_cells; in-context reward available before the lattice is learned ~ (1 - w) rho
(w = wrap-only share, rho = revisit rate).

| cell | n_cells | V per epoch | w | own-map minus held-out |
|---|---|---|---|---|
| 2H (D2 G10) | 100 | 16,000 | 0.68 | +0.71 (memorised) |
| 3H (D3 G10) | 1000 | 1,600 | 0.71 | +0.09 |
| 2L (D2 G32) | 1024 | 1,570 | 0.37 | 0.00 |
| 3L (D3 G18) | 5832 | 275 | 0.35 | 0.00 |

Memorisation tracks high V together with high w (3H and 2L have equal V and differ in w). **Prediction:** with the map
redrawn per sequence (as `environment_cancel.py` already does) memorisation is impossible, and per Theorem 2 a small
grid then helps alignment (more wrap events, shorter wrap gaps); rank D+1 should solve 2H and 3H in-context.
Cheapest test: ND env with per-trajectory map redraw, rank D+1, grid 10, D = 2 and 3, 8 seeds each. If 3H still fails
in-context, wrap-heavy alignment is hard for its own reason and the wrap account stands.

---

## 3. Clock versus map; why eval mode under-reports

**Theorem 3 (when a clock is free).** Write each channel's phase over a key at displacement Delta and gap g as
<k_i, Delta> + g tau_i. Partition channels into map channels (tau_i = 0) and the rest. If the map channels alone give a
margin M = S_map(0) - max_{Delta != 0} S_map(Delta) and the other channels have total amplitude A_c = sum A_i, then the
true key wins against every distractor whenever M > 2 A_c (the clock part of the score lies in [-A_c, A_c]).
So a clock is free when its channels are weak relative to the map margin. By Theorem 1 this partition requires
per-head rank >= D+1. **Prediction (cheap, sharp):** the text world at per-head rank 2 (D = 2) loses the clock seeds:
with r = 4 the clock seeds (4/8 MapWM) solve like map seeds (Fisher 1.00, `TEXTWORLD_RESULTS.md`); at per-head rank 2 a
clock must be a drift (Theorem 1a), so seeds that acquire one should fail. 16 GPU runs.

**Theorem 4 (inverted dropout preserves the mean, not the mode).** Attention-probability dropout returns
o = (1/(1-p)) sum_s m_s a_s v_s, E o = o_eval. If a fraction 1 - eps of the mass is on keys that are kept together
(peaked attention), then with probability ~(1-p) the training-time output is o_eval / (1-p) + O(sqrt(eps p)), and with
probability ~p it is far smaller. The downstream map (residual add, FFN, LayerNorm, argmax) is fit to that mixture, and
o_eval sits off its support (between ~0 and 1/(1-p)). Proof: direct.
**DERIVATION (when it bites).** The pre-norm vector is x + s o + f with s ~ 1/(1-p) in training and 1 at eval; the
direction of LN(.) changes with s by a term proportional to |x| / |o|. A model whose margin is small relative to a 10%
change of the ratio fails at eval; a clean solution with a large margin does not.

**CHECK (`dropout_scale.py`, `dropout_scale_lean.py`, `tw_scale_all.py`; same 40 walks as
`docs/audits/2026-10-03/dropout_mode_check.py`):**

| run | eval | train-mode attn dropout | **eval, attention x1/(1-p)** | dropout, renormalised (noise, no scale) | x1.05 / x1.2 / x1.4 |
|---|---|---|---|---|---|
| tw_normstep MapWM s12 | 0.828 | 0.991 | **0.993** | 0.750 | 0.990 / 0.909 / 0.580 |
| tw_normstep NormStepNB s12 | 0.828 | 0.994 | **0.995** | 0.741 | 0.960 / 0.992 / 0.659 |

All 32 tw_normstep runs, eval -> x1.111: the six runs with a gap all recover (MapWM s12 0.828 -> 0.993, s14
0.965 -> 0.995; NormStepNB s10 0.818 -> 0.977, s12 0.828 -> 0.995, s16 0.896 -> 0.997; NormStep s16 0.836 -> 0.982);
no run at ceiling moves by more than 0.001; DirOnly (diffuse attention, median max weight 0.09-0.17) moves by
-0.004..+0.006. The six gap runs have the lowest |attention output| / |residual| medians (1.37-2.58; ceiling runs
2.3-4.3), as the derivation requires. Registered text world (`runs/textworld/p0`): path s0 0.884 -> 0.974, s7
0.896 -> 0.988, other path seeds unchanged (40-walk mean 0.971 -> 0.994); RoPE 1L and 2L change by -0.015..+0.004.
So the eval-mode gap is not specific to clock seeds; it is a calibration artefact of runs below the clean solution,
and **the text-world path arm is under-reported, not over-reported**. Fixes: score with attention x1/(1-p) (equivalent
to non-inverted dropout at eval) or train without attention dropout. Post hoc; the registered numbers stand as
registered.

---

## 4. The leak, NormStep, and asides

**DERIVATION 4.1 (leak cost accumulates with the gap).** With object steps eta(o) (mean zero over objects), the phase
error between two visits is sum of eta over the objects seen in between: a random walk with variance
~ n_obj(g) sigma_eta^2, n_obj proportional to the gap g. Retrieval fails once the error exceeds the kernel's tolerance,
so the leak's cost is a threshold in sqrt(g): zero at short gaps, rising at long ones.
**CHECK (`leak_gap.py`, `runs/leak/p0` MapWM, 40 eval sequences of the registered stream, test pool, x1;
leak = steps-zeroed minus intact accuracy):**

| gap (moves) | 1-7 | 8-31 | 32-127 | 128-511 | 512-1024 |
|---|---|---|---|---|---|
| n targets | 2467 | 2275 | 1117 | 2583 | 889 |
| leak, 8 seeds | 0.0000-0.0004 | 0.0000-0.0004 | 0.0000 | 0.0027-0.0178 | **0.047-0.122** |

8/8 seeds. **Prediction:** the leak and NormStep's +0.0107 in-distribution gain (`LEAK_RESULTS.md`) disappear at
T <= 256 and grow with T. Cheapest test: re-score the committed leak runs at T = 256 / 2048 (eval-only).

**4.2 NormStep and speed (what is provable, what is not).** THEOREM: in MapWM the content path reads LN(e) and is
invariant to the object-embedding scale s, while the step is linear in s; so the training loss has a direction
(object-embedding scale) along which only the leak changes. NormStep deletes that direction. THEOREM (negative for the
obvious story): along that direction the leak loss is ~ a s^2 and gradient flow gives s(t) ~ exp(-2 a t); NormStep must
remove the leak by rotating W's rows off the object subspace, a quadratic problem with the same curvature scale a.
So no rate advantage follows from curvature alone; `NORMSTEP_NOTES.md` sec. 4's "optimisation effect" is not derived
here. CONJECTURE: the difference is Adam's per-coordinate normalisation -- the object-embedding coordinates carry large,
noisy content gradients that dominate their second-moment estimate and shrink the effective step along the scale
direction; W's coordinates do not. Test (eval-only, minutes): from the saved optimiser states, the ratio
|exp_avg| / sqrt(exp_avg_sq) projected on the scale direction of object embeddings (MapWM) vs on W's object-subspace
rows (NormStep).

**4.3 Asides.**
THEOREM (phase-blind confusion). In a 1-layer path model the key at a token is a function of the token and the phase
only. If every non-direction word has zero step (DirOnly), an aside noun told at cell p has the phase of p, so its key
equals that of the same noun located at p, and differs from a located noun of another type only by the token's own
gain. Retrieval at a cell with both a located and an aside mention is then a token-gain-weighted vote.
CHECK (`aside_attn.py`, `aside_vote.py`, 40 walks): DirOnly s10 gives each located mention 0.38 and each aside mention
0.21 of the attention (confusable targets n = 217); the count-vote idealisation predicts 0.936 accuracy, observed
0.958-0.968 (located "nothing" outweighs aside objects: token gain). Learned-step models: aside mention 0.000-0.16,
located 0.24-0.61 (MapWM s16 0.06 vs 0.58). The three weakest separations (aside 0.12-0.16 per mention: MapWM s11,
s13, NormStep s11) are all per-move-clock seeds (30-41 of 64 drifting channels), at no accuracy cost (0.987-0.999).
**Mechanism (CHECK, `aside_attn2.py`):** the aside noun's phase differs from the located object's by only 0.06-0.33 rad
(mean over channels; MapWM s16 0.069), yet the logit gap is 1.4-11.3 (DirOnly 0.92): the learned kernel is steep and
the offset is coherent across channels, so a small phase bracket separates the mentions.
**THEOREM (bracket conditions).** With context-free steps sigma(w), an aside "w_1 .. w_j NOUN .. w_m" is separated and
the map preserved iff (i) the prefix sum b = sum_{i<j} sigma(w_i) is off the code's lattice of cell phases by more
than the kernel's width, and (ii) the closure sum_{i<=m} sigma(w_i) is 0 modulo the code (else the aside moves the
agent). With context-free steps (i) and (ii) are jointly feasible iff some aside-only words precede the noun (true in
this grammar: thought / remembered / story / about). If every pre-noun word is shared with movement clauses whose sums
are pinned, b is pinned to 0 and a context-dependent step (a gate) is required.
**CHECK (`drift_split.py`): the registered per-word clock is the aside residual.** Splitting the registered
`drift_opt` readout (same walks; it reproduces the registered values, e.g. MapWM s10 0.097, NormStep s16 0.042):

| arm | all optional words (registered) | aside sentences only | adverbs + fillers only |
|---|---|---|---|
| MapWM, 8 seeds | 0.037-0.118 | 0.032-0.115 | 0.008-0.035 |
| NormStep, 8 seeds | 0.042-0.192 | 0.038-0.183 | 0.009-0.025 |

So TW_NORMSTEP's verdict B ("word-count clock") measures the imperfect closure of the aside bracket, not a tick per
word. Negative: per seed, the aside logit gap does not correlate with the aside drift (Spearman 0.12, n = 16), so the
residual is not the price of the separation; it is leftover.
**CHECK of prediction (a) (`aside_zero.py`, 6 solved learned-step runs, 40 walks):** zeroing the steps at aside-sentence
positions only drops accuracy from 0.998-1.000 to 0.987-0.995, and 33 of the 64 new errors name an aside object at the
cell (DirOnly: 100%). Partly confirmed: the aside steps carry part of the separation; the rest must sit in core-word
steps (the step clause's closing '.' precedes every aside and is not masked here). **Prediction (b):** a grammar whose asides start with movement-clause words only ("she walked about a cat") defeats every
context-free step arm, and a context step (HSR/CG) restores it (GPU).

---

## 5. Depth substitution

**THEOREM 5.1 (construction).** Ring of N cells, steps +-1, any p_plus, any T: one MapWM layer with per-head rank 1,
step u(+) = 1, u(-) = -1, u(o) = 0, one channel omega = 2 pi / N, constant q = k, and inverse temperature beta solves
revisit prediction exactly as beta -> infinity (score cos(2 pi (x_t - x_s)/N) is maximal exactly on same-cell keys,
all of which carry the target). Dyck-k with bounded depth (construction sketch): rank 2 suffices -- one channel family on depth
(signed cancelling step) and one monotone clock channel of frequency eps < pi/T for "most recent at this depth";
bracket type is matched by content (the rank-r accumulator that
cancels in one subspace and counts in another, `mapformer_math.tex` sec. 6.2).
**THEOREM 5.2 (simulation, upper bound).** An index transformer with L + 1 layers and an MLP simulates an L-layer MapWM
whose steps are token functions: layer 1 with uniform causal attention over value u(x_s) returns z_t / t; the MLP
multiplies by t (t is available from the index encoding) and outputs cos/sin of omega <w, z_t>; these then play the
rotary role. Cost: the decoding of z_t mod (2 pi / omega) from z_t / t needs resolution 1/t and ~T/N oscillations of
the MLP. So the exchange rate is at least 1 layer in principle; the measured ~2-3 (Dyck, `DYCK_MDEPTH_RESULTS.md`;
ring, `CANCEL_RESULTS.md`) is consistent with the precision/MLP cost making the 2-layer route hard to learn
(index 2L STALLED at 0.954-0.988).
**CONJECTURE 5.3 (lower bound).** A 1-layer index model (score a function of tokens and lag) cannot solve the ring at
p_plus in (0, 1) with width and precision polylog(T), by a one-round communication argument of the
Sanford-Hsu-Telgarsky type: the query's information is (token, t) only, while the target cell depends on a prefix
sum of the other party's tokens. Not proved here.
**CHECK (negative) (`lagmix.py`):** the best fixed "lag-mixture" (score = f(query action, key blank?, lag), copy the
attended observation) reaches 0.50 / 0.51 / 0.67 / 1.00 (two fits: 0.497/0.506/0.672 and 0.498/0.505/0.670) at p_plus 0.5 / 0.75 / 0.9 / 1.0, below the trained index
1-layer models (0.717 / 0.676 / 0.825 / 1.000). The trained models use more than lag statistics (plausibly a
superposed copy decoded by the FFN), and the registered non-monotone dip at 0.75 is not explained by lag statistics.
**Prediction (5.2):** at fixed ring size the index-2L shortfall grows with T / N (more counter wraps per mod period),
while path-1L stays at ceiling; at p_plus = 1 the counter is the index and 1 layer suffices (already observed).
Cheapest test: CANCEL cell p_plus 0.5, index 2L and path 1L, T in {64, 128, 256}, N = 32, 8 seeds.

---

## 6. Ranked to-do (cheapest first)
1. Eval-only, minutes: re-score committed evaluations of every dropout-trained path model with attention x1/(1-p)
   (section 3); report beside the registered numbers.
2. Eval-only: leak by gap at T = 256 / 2048 (4.1); aside-word step ablation on learned-step text-world runs (4.3a);
   head-type census on loop/depth rank-2 checkpoints (1.10.6).
3. GPU, 16 runs: rank 2 with odd action steps and zero observation steps vs plain rank 2 (1.10.1) -- the direct test of
   Theorem 1 as the cause.
4. GPU, 16 runs: text world at per-head rank 2 (section 3 prediction).
5. GPU, 16 runs: ND tori at grid 10 with per-sequence map redraw (section 2).
6. GPU, 24 runs: heads H in {2, 4, 8} at per-head rank 2 (1.7).
7. GPU: CANCEL T-sweep for the exchange rate (5).
Each needs pre-registration and a blind code audit per rules 7 and 29.

---

## Appendix: scripts and raw outputs

All scripts ran from /home/prashr with PYTHONPATH=/home/prashr, CPU, torch 2.6, read-only on runs/. They lived in a
scratch directory; the paths below assume a copy at the same names. Each file is reproduced verbatim, followed by its
output. Note: `classify.py` has no __main__ guard, so the scripts that import it (drift_B, drift_park, classify_t128)
also print the classification table; the duplicate lines are omitted from their outputs here.

### probe_winding2.py

```python
"""Per-head phase-code geometry of trained torus checkpoints (read-only, CPU). See 03_formal.md S1.
phi_h(x) = omega_h * Delta_h(x) (rad per token), Delta from the model's own action_to_lie.
k_j = (phi(+e_j) - phi(-e_j))/2 per channel (phase per unit move on axis j); tau = per-move common
phase = mean(action phis) + E[obs phi] (p_empty 0.5, 16 objects). Channel weight w_b ~ mean_{a,o} |q_b(a)||k_b(o)|.
wind  = w-weighted mean over channels and axes of dist(G k_j / 2pi, Z)   (0 = exactly periodic; random 0.25)
cond  = s_min/s_max of the w-weighted nb x D frame K                     (0 = collinear axes)
clk   = |w^.5 tau| / |w^.5 K|_F ; clk_perp = part of w^.5 tau outside span(w^.5 K), same scale
dir   = top right-singular vector of the weighted frame (which spatial axis the head resolves best)
"""
import sys, numpy as np, torch
torch.set_num_threads(4)
sys.path.insert(0, '/home/prashr')
from mapformer.train_variant import VARIANT_MAP

def probe(path, arm):
    b = torch.load(path, map_location='cpu', weights_only=False); c = b['config']
    m = VARIANT_MAP[arm](vocab_size=c['vocab_size'], d_model=c['d_model'], n_heads=c['n_heads'],
                         n_layers=c['n_layers'], grid_size=c['grid_size']).eval()
    m.load_state_dict(b['model_state_dict']); G = c['grid_size']; Dd = c.get('n_dims', 2)
    L = np.array(b['losses']); fl = L[-max(1, len(L)//20):].mean()
    if c.get('env', 'torus') != 'nd':
        plus, minus = [1, 3], [0, 2]           # GridWorld: N=-x S=+x W=-y E=+y
    else:
        plus, minus = [2 * i for i in range(Dd)], [2 * i + 1 for i in range(Dd)]
    nA = 2 * Dd
    with torch.no_grad():
        ids = torch.arange(c['vocab_size']); x = m.token_emb(ids)
        Dl = m.action_to_lie(x[None])[0].numpy()
        om = m.path_integrator.omega.detach().numpy()
        lay = m.layers[0]; h = lay.norm1(x)
        H, dh = c['n_heads'], c['d_model'] // c['n_heads']
        q = lay.q_proj(h).view(-1, H, dh).numpy(); k = lay.k_proj(h).view(-1, H, dh).numpy()
    phi = Dl * om[None]
    qa = np.hypot(q[..., 0::2], q[..., 1::2]); ka = np.hypot(k[..., 0::2], k[..., 1::2])
    obs = list(range(nA, c['vocab_size'])); K0 = len(obs) - 1
    wobs = np.array([0.5 / K0] * K0 + [0.5])
    out = []
    for hh in range(H):
        P = phi[:, hh]
        K = np.stack([(P[plus[j]] - P[minus[j]]) / 2 for j in range(Dd)], 1)      # nb x D
        tau = P[:nA].mean(0) + (wobs[:, None] * P[obs]).sum(0)
        A = qa[:nA, hh].mean(0) * (wobs[:, None] * ka[obs, hh]).sum(0); w = A / A.sum()
        res = np.abs(G * K / (2 * np.pi) - np.round(G * K / (2 * np.pi)))
        sw = np.sqrt(w)[:, None]; Kw = K * sw; tw = tau * sw[:, 0]
        U, sv, Vt = np.linalg.svd(Kw, full_matrices=False)
        beta, *_ = np.linalg.lstsq(Kw, tw, rcond=None)
        perp = tw - Kw @ beta
        fn = np.linalg.norm(Kw)
        out.append(dict(wind=float((w[:, None] * res).sum() / Dd), cond=float(sv[-1] / sv[0]),
                        clk=float(np.linalg.norm(tw) / fn), clkp=float(np.linalg.norm(perp) / fn),
                        dir=np.round(np.abs(Vt[0]), 2).tolist()))
    return fl, out

if __name__ == '__main__':
    runs, arm = sys.argv[1], sys.argv[2]; pre = sys.argv[3] if len(sys.argv) > 3 else ''
    for s in range(8):
        p = f'{runs}/{arm}_s{s}/{arm}.pt'
        try:
            fl, out = probe(p, arm)
        except FileNotFoundError:
            continue
        hs = ' | '.join(f"wind {o['wind']:.3f} cond {o['cond']:.2f} clk {o['clk']:.3f} perp {o['clkp']:.3f} dir {o['dir']}" for o in out)
        print(f"{pre}{arm} s{s} loss {fl:.3f} {'SOLVED' if fl < 0.05 else '      '} {hs}")
```

(Used as a module by classify.py and the others; its per-head table is reproduced through classify.py.)

### classify.py

```python
"""Classify each head of each trained rank checkpoint: clean / A drift (clock) / B collinear / C off-lattice,
and test the rule 'SOLVED iff some head is clean'. Thresholds fixed before looking at the cross-tab:
clean = wind < 0.05 and cond > 0.25 and clk < 0.05; else A if clk >= 0.05, else B if cond <= 0.25, else C."""
import sys, collections
sys.path.insert(0, '/tmp/claude-1002/-home-prashr-mapformer/224d7c23-b570-4513-8742-cf8a41acc4a3/scratchpad')
from probe_winding2 import probe
R = '/home/prashr/mapformer/runs'
SETS = [('torus D2 G64 rank2 (shared)', f'{R}/rank_mi/p0', 'Vanilla'), ('torus D2 G64 rank2 (per-head)', f'{R}/rank_mi/p0', 'Vanilla_r2ph'),
        ('torus D2 G64 rank2 (block-diag)', f'{R}/rank_sep/p0', 'Vanilla_r4mibd'), ('torus D2 G64 rank3', f'{R}/rank3/p0', 'Vanilla_r3ph'),
        ('torus D2 G64 rank4 (shared)', f'{R}/rank_mi/p0', 'Vanilla_r4mi'), ('torus D2 G64 rank4 (per-head)', f'{R}/rank_sep/p0', 'Vanilla_r4ph'),
        ('nd D2 G32 rank2', f'{R}/rank_nd/D2', 'Vanilla_r2ph'), ('nd D2 G32 rank3', f'{R}/rank_nd/D2', 'Vanilla_r3ph'),
        ('nd D3 G10 rank3', f'{R}/rank_nd/D3', 'Vanilla_r3ph'), ('nd D3 G10 rank4', f'{R}/rank_nd/D3', 'Vanilla_r4ph')]
def cls(o):
    if o['wind'] < 0.05 and o['cond'] > 0.25 and o['clk'] < 0.05: return 'clean'
    if o['clk'] >= 0.05: return 'A'
    if o['cond'] <= 0.25: return 'B'
    return 'C'
tab = collections.Counter(); agree = n = 0
for name, runs, arm in SETS:
    types = collections.Counter(); solved = 0; rule = 0; rows = []
    for s in range(8):
        try: fl, out = probe(f'{runs}/{arm}_s{s}/{arm}.pt', arm)
        except FileNotFoundError: continue
        c = [cls(o) for o in out]; sv = fl < 0.05; pred = 'clean' in c
        solved += sv; rule += pred; agree += (sv == pred); n += 1
        for x in c: types[x] += 1
        rows.append(f"s{s}:{''.join(x[0] for x in c)}{'*' if sv else ''}")
    print(f"{name:32s} SOLVED {solved}/8  rule {rule}/8  heads: clean {types['clean']:2d} A {types['A']:2d} B {types['B']:2d} C {types['C']:2d}   " + ' '.join(rows), flush=True)
print(f"rule 'SOLVED iff some head clean' agrees on {agree}/{n} runs")
```

Output:
```
torus D2 G64 rank2 (shared)      SOLVED 0/8  rule 1/8  heads: clean  1 A 13 B  1 C  1   s0:AA s1:Cc s2:AA s3:BA s4:AA s5:AA s6:AA s7:AA
torus D2 G64 rank2 (per-head)    SOLVED 2/8  rule 2/8  heads: clean  3 A  9 B  4 C  0   s0:AA s1:Ac* s2:cc* s3:AA s4:AA s5:BB s6:BB s7:AA
torus D2 G64 rank2 (block-diag)  SOLVED 2/8  rule 2/8  heads: clean  2 A  7 B  7 C  0   s0:AA s1:BA s2:cB* s3:AB s4:AB s5:BA s6:cA* s7:BB
torus D2 G64 rank3               SOLVED 6/8  rule 6/8  heads: clean 11 A  3 B  1 C  1   s0:AA s1:cc* s2:cc* s3:AB s4:cc* s5:cC* s6:cc* s7:cc*
torus D2 G64 rank4 (shared)      SOLVED 8/8  rule 8/8  heads: clean 15 A  0 B  0 C  1   s0:cc* s1:cc* s2:cc* s3:cc* s4:cc* s5:cc* s6:cc* s7:cC*
torus D2 G64 rank4 (per-head)    SOLVED 8/8  rule 8/8  heads: clean 14 A  0 B  0 C  2   s0:cc* s1:cC* s2:cc* s3:cc* s4:cc* s5:Cc* s6:cc* s7:cc*
nd D2 G32 rank2                  SOLVED 1/8  rule 1/8  heads: clean  1 A 14 B  1 C  0   s0:AA s1:AA s2:cA* s3:AA s4:AA s5:AA s6:BA s7:AA
nd D2 G32 rank3                  SOLVED 8/8  rule 8/8  heads: clean 15 A  0 B  0 C  1   s0:cC* s1:cc* s2:cc* s3:cc* s4:cc* s5:cc* s6:cc* s7:cc*
nd D3 G10 rank3                  SOLVED 1/8  rule 1/8  heads: clean  1 A 14 B  0 C  1   s0:AA s1:Ac* s2:AA s3:AA s4:AA s5:AA s6:AA s7:CA
nd D3 G10 rank4                  SOLVED 4/8  rule 4/8  heads: clean  5 A 10 B  1 C  0   s0:Ac* s1:AA s2:AA s3:Ac* s4:cc* s5:AB s6:Ac* s7:AA
rule 'SOLVED iff some head clean' agrees on 79/80 runs
```

### classify_t128.py

```python
import sys, collections
sys.path.insert(0, '/tmp/claude-1002/-home-prashr-mapformer/224d7c23-b570-4513-8742-cf8a41acc4a3/scratchpad')
from probe_winding2 import probe
from classify import cls
R = '/home/prashr/mapformer/runs'
for name, runs, arm in [('T=128 paper2x2 rank2', f'{R}/paper2x2/p0', 'Vanilla'), ('T=128 paper2x2 rank4', f'{R}/paper2x2/p0', 'Vanilla_r4'),
                        ('T=128 rank_sweep rank2', f'{R}/rank_sweep/p0', 'Vanilla'), ('T=128 rank_sweep rank4', f'{R}/rank_sweep/p0', 'Vanilla_r4')]:
    rows = []; types = collections.Counter()
    for s in range(8):
        try: fl, out = probe(f'{runs}/{arm}_s{s}/{arm}.pt', arm)
        except FileNotFoundError: continue
        c = [cls(o) for o in out]
        for x in c: types[x] += 1
        rows.append(f"s{s}:{''.join(x[0] for x in c)}(L{fl:.3f},clk {max(o['clk'] for o in out):.2f})")
    print(f"{name:24s} heads clean {types['clean']} A {types['A']} B {types['B']} C {types['C']}  " + ' '.join(rows), flush=True)
```

Output:
```
T=128 paper2x2 rank2     heads clean 0 A 7 B 9 C 0  s0:AA(L0.002,clk 0.47) s1:BB(L0.058,clk 0.01) s2:BB(L0.010,clk 0.01) s3:BB(L0.009,clk 0.01) s4:BB(L0.015,clk 0.01) s5:AA(L0.164,clk 0.36) s6:BA(L0.060,clk 0.07) s7:AA(L0.385,clk 0.48)
T=128 paper2x2 rank4     heads clean 1 A 0 B 0 C 15  s0:CC(L0.000,clk 0.01) s1:CC(L0.000,clk 0.00) s2:CC(L0.000,clk 0.01) s3:Cc(L0.000,clk 0.00) s4:CC(L0.000,clk 0.00) s5:CC(L0.000,clk 0.00) s6:CC(L0.000,clk 0.01) s7:CC(L0.000,clk 0.00)
T=128 rank_sweep rank2   heads clean 0 A 6 B 10 C 0  s0:AA(L0.001,clk 0.76) s1:BB(L0.011,clk 0.00) s2:BB(L0.009,clk 0.00) s3:BB(L0.009,clk 0.01) s4:AA(L0.002,clk 0.44) s5:BB(L0.012,clk 0.04) s6:BB(L0.010,clk 0.00) s7:AA(L0.089,clk 0.29)
T=128 rank_sweep rank4   heads clean 0 A 0 B 0 C 16  s0:CC(L0.000,clk 0.01) s1:CC(L0.000,clk 0.00) s2:CC(L0.000,clk 0.01) s3:CC(L0.000,clk 0.00) s4:CC(L0.000,clk 0.00) s5:CC(L0.000,clk 0.00) s6:CC(L0.000,clk 0.01) s7:CC(L0.000,clk 0.00)
```

### drift_B.py

```python
"""Type-B (collinear) rank-D heads: is the phase frame degenerate because the LATENT frame U collapsed, or because
the head's readout W_h (nb x r, omega-scaled) killed a latent direction? If the readout killed it, does the killed
direction carry the per-move common component c (drift avoidance, Lemma 1 alternative)?"""
import sys, numpy as np, torch
torch.set_num_threads(4)
sys.path.insert(0, '/home/prashr'); sys.path.insert(0, '/tmp/claude-1002/-home-prashr-mapformer/224d7c23-b570-4513-8742-cf8a41acc4a3/scratchpad')
from mapformer.train_variant import VARIANT_MAP
from probe_winding2 import probe
from classify import cls
R = '/home/prashr/mapformer/runs'
def parts(path, arm):
    b = torch.load(path, map_location='cpu', weights_only=False); c = b['config']
    m = VARIANT_MAP[arm](vocab_size=c['vocab_size'], d_model=c['d_model'], n_heads=c['n_heads'],
                         n_layers=c['n_layers'], grid_size=c['grid_size']).eval()
    m.load_state_dict(b['model_state_dict']); H = c['n_heads']; Dd = c.get('n_dims', 2)
    a2l = m.action_to_lie; om = m.path_integrator.omega.detach().numpy()
    with torch.no_grad():
        x = m.token_emb(torch.arange(c['vocab_size'])); z = a2l.w_in(x).numpy()
        q = m.layers[0].q_proj(m.layers[0].norm1(x)); k = m.layers[0].k_proj(m.layers[0].norm1(x))
    nb = om.shape[1]
    if arm == 'Vanilla': lat = [z, z]; Wo = a2l.w_out.weight.detach().numpy(); W = [Wo[:nb], Wo[nb:]]
    elif arm == 'Vanilla_r4mibd':
        Wo = (a2l.w_out.weight * a2l.mask).detach().numpy(); lat = [z[:, 0:2], z[:, 2:4]]; W = [Wo[:nb, 0:2], Wo[nb:, 2:4]]
    else:
        r = z.shape[1] // H; lat = [z[:, h*r:(h+1)*r] for h in range(H)]; W = [a2l.w_out[h].weight.detach().numpy() for h in range(H)]
    if c.get('env', 'torus') != 'nd': plus, minus = [1, 3], [0, 2]
    else: plus, minus = [2*i for i in range(Dd)], [2*i+1 for i in range(Dd)]
    nA = 2 * Dd; K0 = c['vocab_size'] - nA - 1; wobs = np.array([0.5 / K0] * K0 + [0.5])
    dh = c['d_model'] // H; qn = q.view(-1, H, dh).numpy(); kn = k.view(-1, H, dh).numpy()
    out = []
    for h in range(H):
        L = lat[h]; U = np.stack([(L[plus[j]] - L[minus[j]]) / 2 for j in range(Dd)], 1)
        cc = L[:nA].mean(0) + (wobs[:, None] * L[nA:]).sum(0)
        qa = np.hypot(qn[:nA, h, 0::2], qn[:nA, h, 1::2]).mean(0); ka = (wobs[:, None] * np.hypot(kn[nA:, h, 0::2], kn[nA:, h, 1::2])).sum(0)
        wgt = np.sqrt(qa * ka / (qa * ka).sum())
        Wh = (om[h][:, None] * W[h]) * wgt[:, None]                    # weighted, omega-scaled readout nb x r
        su = np.linalg.svd(U, compute_uv=False); _, sw, Vw = np.linalg.svd(Wh, full_matrices=False)
        null = Vw[-1]                                                   # latent direction the readout reads least
        out.append(dict(condU=su[-1] / su[0], condW=sw[-1] / sw[0], c_over_U=np.linalg.norm(cc) / np.linalg.norm(U),
                        cos_c_null=abs(null @ cc) / max(np.linalg.norm(cc), 1e-12)))
    return out
SETS = [(f'{R}/rank_mi/p0', 'Vanilla'), (f'{R}/rank_mi/p0', 'Vanilla_r2ph'), (f'{R}/rank_sep/p0', 'Vanilla_r4mibd'),
        (f'{R}/rank_nd/D2', 'Vanilla_r2ph'), (f'{R}/rank_nd/D3', 'Vanilla_r3ph')]
agg = {}
for runs, arm in SETS:
    for s in range(8):
        p = f'{runs}/{arm}_s{s}/{arm}.pt'; fl, out = probe(p, arm); pr = parts(p, arm)
        for o, g in zip(out, pr):
            t = cls(o); agg.setdefault(t, []).append(g)
            if t == 'B':
                print(f"B {runs.split('runs/')[1]:12s} {arm:15s} s{s}: cond(U latent) {g['condU']:.2f}  cond(readout) {g['condW']:.3f}  |c|/|U| {g['c_over_U']:.2f}  |cos(c, readout-null dir)| {g['cos_c_null']:.2f}")
for t, gs in agg.items():
    f = lambda k: np.median([g[k] for g in gs])
    print(f"type {t:5s} n={len(gs):2d}: median cond(U) {f('condU'):.2f}  cond(readout) {f('condW'):.3f}  |c|/|U| {f('c_over_U'):.2f}  |cos(c, null)| {f('cos_c_null'):.2f}")
```

Output:
```
B rank_mi/p0   Vanilla         s3: cond(U latent) 0.08  cond(readout) 0.021  |c|/|U| 0.79  |cos(c, readout-null dir)| 1.00
B rank_mi/p0   Vanilla_r2ph    s5: cond(U latent) 0.00  cond(readout) 0.092  |c|/|U| 0.04  |cos(c, readout-null dir)| 1.00
B rank_mi/p0   Vanilla_r2ph    s5: cond(U latent) 0.00  cond(readout) 0.132  |c|/|U| 0.03  |cos(c, readout-null dir)| 1.00
B rank_mi/p0   Vanilla_r2ph    s6: cond(U latent) 0.00  cond(readout) 0.111  |c|/|U| 0.03  |cos(c, readout-null dir)| 1.00
B rank_mi/p0   Vanilla_r2ph    s6: cond(U latent) 0.00  cond(readout) 0.024  |c|/|U| 0.05  |cos(c, readout-null dir)| 1.00
B rank_sep/p0  Vanilla_r4mibd  s1: cond(U latent) 0.00  cond(readout) 0.083  |c|/|U| 0.04  |cos(c, readout-null dir)| 1.00
B rank_sep/p0  Vanilla_r4mibd  s2: cond(U latent) 0.00  cond(readout) 0.080  |c|/|U| 0.02  |cos(c, readout-null dir)| 1.00
B rank_sep/p0  Vanilla_r4mibd  s3: cond(U latent) 0.05  cond(readout) 0.018  |c|/|U| 0.34  |cos(c, readout-null dir)| 1.00
B rank_sep/p0  Vanilla_r4mibd  s4: cond(U latent) 0.01  cond(readout) 0.106  |c|/|U| 0.10  |cos(c, readout-null dir)| 1.00
B rank_sep/p0  Vanilla_r4mibd  s5: cond(U latent) 0.00  cond(readout) 0.038  |c|/|U| 0.09  |cos(c, readout-null dir)| 1.00
B rank_sep/p0  Vanilla_r4mibd  s7: cond(U latent) 0.01  cond(readout) 0.082  |c|/|U| 0.02  |cos(c, readout-null dir)| 1.00
B rank_sep/p0  Vanilla_r4mibd  s7: cond(U latent) 0.22  cond(readout) 0.147  |c|/|U| 0.01  |cos(c, readout-null dir)| 1.00
B rank_nd/D2   Vanilla_r2ph    s6: cond(U latent) 0.00  cond(readout) 0.157  |c|/|U| 0.07  |cos(c, readout-null dir)| 1.00
type A     n=57: median cond(U) 0.19  cond(readout) 0.516  |c|/|U| 0.62  |cos(c, null)| 0.92
type C     n= 2: median cond(U) 0.80  cond(readout) 0.622  |c|/|U| 0.02  |cos(c, null)| 0.47
type clean n= 8: median cond(U) 0.83  cond(readout) 0.834  |c|/|U| 0.00  |cos(c, null)| 0.53
type B     n=13: median cond(U) 0.00  cond(readout) 0.083  |c|/|U| 0.04  |cos(c, null)| 1.00
```

### drift_park.py

```python
"""Type-B (collinear) rank-D heads: is the phase frame degenerate because the LATENT frame U collapsed, or because
the head's readout W_h (nb x r, omega-scaled) killed a latent direction? If the readout killed it, does the killed
direction carry the per-move common component c (drift avoidance, Lemma 1 alternative)?"""
import sys, numpy as np, torch
torch.set_num_threads(4)
sys.path.insert(0, '/home/prashr'); sys.path.insert(0, '/tmp/claude-1002/-home-prashr-mapformer/224d7c23-b570-4513-8742-cf8a41acc4a3/scratchpad')
from mapformer.train_variant import VARIANT_MAP
from probe_winding2 import probe
from classify import cls
R = '/home/prashr/mapformer/runs'
def parts(path, arm):
    b = torch.load(path, map_location='cpu', weights_only=False); c = b['config']
    m = VARIANT_MAP[arm](vocab_size=c['vocab_size'], d_model=c['d_model'], n_heads=c['n_heads'],
                         n_layers=c['n_layers'], grid_size=c['grid_size']).eval()
    m.load_state_dict(b['model_state_dict']); H = c['n_heads']; Dd = c.get('n_dims', 2)
    a2l = m.action_to_lie; om = m.path_integrator.omega.detach().numpy()
    with torch.no_grad():
        x = m.token_emb(torch.arange(c['vocab_size'])); z = a2l.w_in(x).numpy()
        q = m.layers[0].q_proj(m.layers[0].norm1(x)); k = m.layers[0].k_proj(m.layers[0].norm1(x))
    nb = om.shape[1]
    if arm == 'Vanilla_r4mi': lat = [z, z]; Wo = a2l.w_out.weight.detach().numpy(); W = [Wo[:nb], Wo[nb:]]
    elif arm == 'Vanilla': lat = [z, z]; Wo = a2l.w_out.weight.detach().numpy(); W = [Wo[:nb], Wo[nb:]]
    elif arm == 'Vanilla_r4mibd':
        Wo = (a2l.w_out.weight * a2l.mask).detach().numpy(); lat = [z[:, 0:2], z[:, 2:4]]; W = [Wo[:nb, 0:2], Wo[nb:, 2:4]]
    else:
        r = z.shape[1] // H; lat = [z[:, h*r:(h+1)*r] for h in range(H)]; W = [a2l.w_out[h].weight.detach().numpy() for h in range(H)]
    if c.get('env', 'torus') != 'nd': plus, minus = [1, 3], [0, 2]
    else: plus, minus = [2*i for i in range(Dd)], [2*i+1 for i in range(Dd)]
    nA = 2 * Dd; K0 = c['vocab_size'] - nA - 1; wobs = np.array([0.5 / K0] * K0 + [0.5])
    dh = c['d_model'] // H; qn = q.view(-1, H, dh).numpy(); kn = k.view(-1, H, dh).numpy()
    out = []
    for h in range(H):
        L = lat[h]; U = np.stack([(L[plus[j]] - L[minus[j]]) / 2 for j in range(Dd)], 1)
        cc = L[:nA].mean(0) + (wobs[:, None] * L[nA:]).sum(0)
        qa = np.hypot(qn[:nA, h, 0::2], qn[:nA, h, 1::2]).mean(0); ka = (wobs[:, None] * np.hypot(kn[nA:, h, 0::2], kn[nA:, h, 1::2])).sum(0)
        wgt = np.sqrt(qa * ka / (qa * ka).sum())
        Wh = (om[h][:, None] * W[h]) * wgt[:, None]                    # weighted, omega-scaled readout nb x r
        su = np.linalg.svd(U, compute_uv=False); _, sw, Vw = np.linalg.svd(Wh, full_matrices=False)
        null = Vw[-1]                                                   # latent direction the readout reads least
        Pu = U @ np.linalg.pinv(U); cperp = cc - Pu @ cc                   # part of c outside the map plane
        out.append(dict(condU=su[-1] / su[0], condW=sw[-1] / sw[0], c_over_U=np.linalg.norm(cc) / np.linalg.norm(U),
                        cperp_over_U=np.linalg.norm(cperp) / np.linalg.norm(U),
                        read_c=np.linalg.norm(Wh @ cc) / np.linalg.norm(Wh @ U),           # how much the readout sees c, rel. to map
                        cos_c_null=abs(null @ cc) / max(np.linalg.norm(cc), 1e-12)))
    return out
SETS = [(f'{R}/rank_mi/p0', 'Vanilla_r4mi'), (f'{R}/rank_sep/p0', 'Vanilla_r4ph'), (f'{R}/rank3/p0', 'Vanilla_r3ph'),
        (f'{R}/rank_nd/D2', 'Vanilla_r3ph'), (f'{R}/rank_nd/D3', 'Vanilla_r4ph')]
for runs, arm in SETS:
    rows = []
    for s in range(8):
        p = f'{runs}/{arm}_s{s}/{arm}.pt'; fl, out = probe(p, arm); pr = parts(p, arm)
        for o, g in zip(out, pr): rows.append((cls(o), g))
    for t in ('clean', 'A', 'C'):
        G = [g for tt, g in rows if tt == t]
        if not G: continue
        f = lambda k: np.median([g[k] for g in G])
        print(f"{runs.split('runs/')[1]:12s} {arm:13s} type {t:5s} n={len(G):2d}: |c|/|U| {f('c_over_U'):.3f}  |c outside map plane|/|U| {f('cperp_over_U'):.3f}  readout sees c (|Wc|/|WU|) {f('read_c'):.3f}  cond(U) {f('condU'):.2f}")

```

Output:
```
rank_mi/p0   Vanilla_r4mi  type clean n=15: |c|/|U| 0.003  |c outside map plane|/|U| 0.003  readout sees c (|Wc|/|WU|) 0.000  cond(U) 0.94
rank_mi/p0   Vanilla_r4mi  type C     n= 1: |c|/|U| 0.003  |c outside map plane|/|U| 0.003  readout sees c (|Wc|/|WU|) 0.001  cond(U) 0.99
rank_sep/p0  Vanilla_r4ph  type clean n=14: |c|/|U| 0.002  |c outside map plane|/|U| 0.002  readout sees c (|Wc|/|WU|) 0.000  cond(U) 0.88
rank_sep/p0  Vanilla_r4ph  type C     n= 2: |c|/|U| 0.003  |c outside map plane|/|U| 0.003  readout sees c (|Wc|/|WU|) 0.000  cond(U) 0.80
rank3/p0     Vanilla_r3ph  type clean n=11: |c|/|U| 0.003  |c outside map plane|/|U| 0.002  readout sees c (|Wc|/|WU|) 0.000  cond(U) 0.86
rank3/p0     Vanilla_r3ph  type A     n= 3: |c|/|U| 1.633  |c outside map plane|/|U| 0.413  readout sees c (|Wc|/|WU|) 1.335  cond(U) 0.35
rank3/p0     Vanilla_r3ph  type C     n= 1: |c|/|U| 0.004  |c outside map plane|/|U| 0.004  readout sees c (|Wc|/|WU|) 0.000  cond(U) 0.60
rank_nd/D2   Vanilla_r3ph  type clean n=15: |c|/|U| 0.002  |c outside map plane|/|U| 0.002  readout sees c (|Wc|/|WU|) 0.000  cond(U) 0.89
rank_nd/D2   Vanilla_r3ph  type C     n= 1: |c|/|U| 0.014  |c outside map plane|/|U| 0.013  readout sees c (|Wc|/|WU|) 0.001  cond(U) 0.94
rank_nd/D3   Vanilla_r4ph  type clean n= 5: |c|/|U| 0.004  |c outside map plane|/|U| 0.004  readout sees c (|Wc|/|WU|) 0.000  cond(U) 0.61
rank_nd/D3   Vanilla_r4ph  type A     n=10: |c|/|U| 0.509  |c outside map plane|/|U| 0.161  readout sees c (|Wc|/|WU|) 0.484  cond(U) 0.21
```

### probe_init.py

```python
"""Reconstruct each run's INITIAL weights (torch.manual_seed(seed); cls(...), as train_variant.py:499,550;
env construction draws no torch RNG) and measure the initial per-head phase frame. Compare with outcome."""
import sys, numpy as np, torch
torch.set_num_threads(4)
sys.path.insert(0, '/home/prashr')
from mapformer.train_variant import VARIANT_MAP
from scipy.stats import mannwhitneyu

def init_geom(arm, seed, c):
    torch.manual_seed(seed)
    m = VARIANT_MAP[arm](vocab_size=c['vocab_size'], d_model=c['d_model'], n_heads=c['n_heads'],
                         n_layers=c['n_layers'], grid_size=c['grid_size']).eval()
    Dd = c.get('n_dims', 2)
    if c.get('env', 'torus') != 'nd': plus, minus = [1, 3], [0, 2]
    else: plus, minus = [2*i for i in range(Dd)], [2*i+1 for i in range(Dd)]
    nA = 2 * Dd
    with torch.no_grad():
        x = m.token_emb(torch.arange(c['vocab_size']))
        P = (m.action_to_lie(x[None])[0] * m.path_integrator.omega[None]).numpy()   # V,H,nb
    res = []
    for h in range(c['n_heads']):
        K = np.stack([(P[plus[j], h] - P[minus[j], h]) / 2 for j in range(Dd)], 1)
        sv = np.linalg.svd(K, compute_uv=False)
        tau = P[:nA, h].mean(0) + P[nA:, h].mean(0)
        res.append((sv[-1] / sv[0], np.linalg.norm(tau) / np.linalg.norm(K)))
    return res, m

def trained_state(path):
    b = torch.load(path, map_location='cpu', weights_only=False)
    L = np.array(b['losses']); return b, L[-max(1, len(L)//20):].mean()

rows = []
for runs, arm in [(a.split(':')[0], a.split(':')[1]) for a in sys.argv[1:]]:
    for s in range(8):
        p = f'{runs}/{arm}_s{s}/{arm}.pt'
        try: b, fl = trained_state(p)
        except FileNotFoundError: continue
        g, m = init_geom(arm, s, b['config'])
        # sanity: an init weight that the trained model shares exactly would be ideal; report omega init vs trained diff instead
        best_cond = max(x[0] for x in g); min_clk = min(x[1] for x in g)
        rows.append((arm, runs.split('/')[-2] + '/' + runs.split('/')[-1], s, fl < 0.05, best_cond, min_clk))
        print(f"{runs.split('runs/')[-1]:18s} {arm:15s} s{s} {'SOLVED' if fl<0.05 else '      '} init per-head cond " +
              ' '.join(f'{x[0]:.2f}' for x in g) + '  clk ' + ' '.join(f'{x[1]:.2f}' for x in g))
```

Output:
```
rank_mi/p0         Vanilla         s0        init per-head cond 0.30 0.14  clk 0.37 0.19
rank_mi/p0         Vanilla         s1        init per-head cond 0.62 0.51  clk 0.48 0.38
rank_mi/p0         Vanilla         s2        init per-head cond 0.50 0.63  clk 0.75 0.86
rank_mi/p0         Vanilla         s3        init per-head cond 0.22 0.47  clk 0.16 0.27
rank_mi/p0         Vanilla         s4        init per-head cond 0.04 0.03  clk 1.03 0.83
rank_mi/p0         Vanilla         s5        init per-head cond 0.22 0.15  clk 0.56 0.39
rank_mi/p0         Vanilla         s6        init per-head cond 0.02 0.02  clk 1.18 0.86
rank_mi/p0         Vanilla         s7        init per-head cond 0.43 0.50  clk 1.02 0.88
rank_mi/p0         Vanilla_r2ph    s0        init per-head cond 0.37 0.17  clk 0.50 0.53
rank_mi/p0         Vanilla_r2ph    s1 SOLVED init per-head cond 0.69 0.47  clk 0.84 1.13
rank_mi/p0         Vanilla_r2ph    s2 SOLVED init per-head cond 0.36 0.40  clk 0.15 0.28
rank_mi/p0         Vanilla_r2ph    s3        init per-head cond 0.20 0.32  clk 2.44 0.23
rank_mi/p0         Vanilla_r2ph    s4        init per-head cond 0.37 0.01  clk 2.91 0.87
rank_mi/p0         Vanilla_r2ph    s5        init per-head cond 0.57 0.18  clk 0.60 0.91
rank_mi/p0         Vanilla_r2ph    s6        init per-head cond 0.55 0.11  clk 0.52 2.92
rank_mi/p0         Vanilla_r2ph    s7        init per-head cond 0.05 0.37  clk 0.21 0.73
rank_sep/p0        Vanilla_r4mibd  s0        init per-head cond 0.36 0.09  clk 0.60 0.29
rank_sep/p0        Vanilla_r4mibd  s1        init per-head cond 0.92 0.60  clk 0.76 1.10
rank_sep/p0        Vanilla_r4mibd  s2 SOLVED init per-head cond 0.37 0.20  clk 0.15 0.30
rank_sep/p0        Vanilla_r4mibd  s3        init per-head cond 0.22 0.33  clk 2.41 0.24
rank_sep/p0        Vanilla_r4mibd  s4        init per-head cond 0.21 0.01  clk 1.93 0.73
rank_sep/p0        Vanilla_r4mibd  s5        init per-head cond 0.62 0.25  clk 0.68 1.28
rank_sep/p0        Vanilla_r4mibd  s6 SOLVED init per-head cond 0.64 0.16  clk 0.49 2.58
rank_sep/p0        Vanilla_r4mibd  s7        init per-head cond 0.06 0.33  clk 0.21 0.73
rank_nd/D2         Vanilla_r2ph    s0        init per-head cond 0.37 0.16  clk 0.49 0.48
rank_nd/D2         Vanilla_r2ph    s1        init per-head cond 0.69 0.47  clk 0.84 1.14
rank_nd/D2         Vanilla_r2ph    s2 SOLVED init per-head cond 0.37 0.42  clk 0.15 0.27
rank_nd/D2         Vanilla_r2ph    s3        init per-head cond 0.20 0.34  clk 2.42 0.24
rank_nd/D2         Vanilla_r2ph    s4        init per-head cond 0.36 0.02  clk 2.84 0.87
rank_nd/D2         Vanilla_r2ph    s5        init per-head cond 0.57 0.18  clk 0.61 0.92
rank_nd/D2         Vanilla_r2ph    s6        init per-head cond 0.53 0.12  clk 0.53 2.89
rank_nd/D2         Vanilla_r2ph    s7        init per-head cond 0.06 0.39  clk 0.21 0.73
rank_nd/D3         Vanilla_r3ph    s0        init per-head cond 0.26 0.14  clk 0.32 0.80
rank_nd/D3         Vanilla_r3ph    s1 SOLVED init per-head cond 0.24 0.16  clk 1.10 0.37
rank_nd/D3         Vanilla_r3ph    s2        init per-head cond 0.02 0.07  clk 0.17 1.01
rank_nd/D3         Vanilla_r3ph    s3        init per-head cond 0.28 0.04  clk 0.26 0.38
rank_nd/D3         Vanilla_r3ph    s4        init per-head cond 0.08 0.03  clk 0.55 0.25
rank_nd/D3         Vanilla_r3ph    s5        init per-head cond 0.20 0.22  clk 1.05 0.58
rank_nd/D3         Vanilla_r3ph    s6        init per-head cond 0.19 0.02  clk 0.38 0.14
rank_nd/D3         Vanilla_r3ph    s7        init per-head cond 0.32 0.03  clk 0.36 0.20
rank3/p0           Vanilla_r3ph    s0        init per-head cond 0.42 0.56  clk 0.61 0.44
rank3/p0           Vanilla_r3ph    s1 SOLVED init per-head cond 0.42 0.40  clk 0.99 0.65
rank3/p0           Vanilla_r3ph    s2 SOLVED init per-head cond 0.58 0.35  clk 0.19 0.41
rank3/p0           Vanilla_r3ph    s3        init per-head cond 0.25 0.33  clk 0.67 0.19
rank3/p0           Vanilla_r3ph    s4 SOLVED init per-head cond 0.50 0.56  clk 1.32 1.11
rank3/p0           Vanilla_r3ph    s5 SOLVED init per-head cond 0.70 0.30  clk 0.78 1.61
rank3/p0           Vanilla_r3ph    s6 SOLVED init per-head cond 0.78 0.38  clk 1.03 0.61
rank3/p0           Vanilla_r3ph    s7 SOLVED init per-head cond 0.37 0.32  clk 0.60 0.22
rank_nd/D3         Vanilla_r4ph    s0 SOLVED init per-head cond 0.31 0.35  clk 0.68 0.45
rank_nd/D3         Vanilla_r4ph    s1        init per-head cond 0.26 0.48  clk 1.09 0.32
rank_nd/D3         Vanilla_r4ph    s2        init per-head cond 0.17 0.13  clk 0.19 0.82
rank_nd/D3         Vanilla_r4ph    s3 SOLVED init per-head cond 0.23 0.21  clk 0.24 0.42
rank_nd/D3         Vanilla_r4ph    s4 SOLVED init per-head cond 0.14 0.30  clk 0.40 0.40
rank_nd/D3         Vanilla_r4ph    s5        init per-head cond 0.24 0.16  clk 0.54 0.48
rank_nd/D3         Vanilla_r4ph    s6 SOLVED init per-head cond 0.32 0.43  clk 0.36 0.13
rank_nd/D3         Vanilla_r4ph    s7        init per-head cond 0.41 0.16  clk 0.35 0.33
```

### dropout_scale.py

```python
"""Is the eval/train-mode gap a SCALE effect of inverted dropout? Same 40 walks and forward as
docs/audits/2026-10-03/dropout_mode_check.py, with extra modes on the attention probabilities a:
  eval: a;  attn: dropout(a) (3 seeds);  scale:s  a*s deterministic;  renorm: dropout(a) / rowsum (noise, no scale);
  topdrop: zero the single largest weight with prob 0.1 per query, no rescale."""
import sys, math, numpy as np, torch, torch.nn.functional as F
torch.set_num_threads(4)
sys.path.insert(0, '/home/prashr')
from mapformer.environment_textworld import TextWorld
from mapformer.analyze_textworld_secondary import load as load_tw
from mapformer.tw_normstep_readouts import load as load_ns
from mapformer.model import _apply_rope

def walks(n=40, T=1024):
    te = TextWorld(size=64, seed=10000); np.random.seed(10**6)
    return [te.generate_trajectory(T) for _ in range(n)]

STATS = {}
@torch.no_grad()
def fwd(m, tok, mode):
    x = m.token_emb(tok)
    d = m.step(tok, x) if hasattr(m, "step") else m.action_to_lie(x)
    cos_a, sin_a = m.path_integrator(d)
    T = tok.shape[1]; cm = torch.triu(torch.ones(T, T, dtype=torch.bool), 1)
    for L in m.layers:
        B = x.shape[0]; h = L.norm1(x)
        Q = _apply_rope(L.q_proj(h).view(B, T, L.n_heads, L.d_head).transpose(1, 2), cos_a, sin_a)
        K = _apply_rope(L.k_proj(h).view(B, T, L.n_heads, L.d_head).transpose(1, 2), cos_a, sin_a)
        V = L.v_proj(h).view(B, T, L.n_heads, L.d_head).transpose(1, 2)
        a = F.softmax((Q @ K.transpose(-1, -2) / math.sqrt(L.d_head)).masked_fill(cm, float("-inf")), -1)
        if mode == 'attn': a = F.dropout(a, 0.1, training=True)
        elif mode.startswith('scale:'): a = a * float(mode[6:])
        elif mode == 'renorm':
            a = F.dropout(a, 0.1, training=True); a = a / a.sum(-1, keepdim=True).clamp_min(1e-9)
        o = L.o_proj((a @ V).transpose(1, 2).reshape(B, T, L.d_model))
        if mode == 'eval':
            STATS.setdefault('ratio', []).append((o.norm(dim=-1) / x.norm(dim=-1)).flatten())
            STATS.setdefault('amax', []).append(a.max(-1).values.flatten(1))
        x = x + o
        x = x + L.ffn[2](L.ffn[1](L.ffn[0](L.norm2(x))))
    return m.out_proj(m.out_norm(x))

@torch.no_grad()
def score(m, W, mode, dseed=0):
    torch.manual_seed(dseed); ok = tot = 0; nll = 0.0
    for tok, _o, rev in W:
        lg = fwd(m, tok[None, :-1], mode)[0]; msk = rev[1:]
        lp = F.log_softmax(lg, -1); ok += int((lg.argmax(-1)[msk] == tok[1:][msk]).sum()); tot += int(msk.sum())
        nll += float(-lp[msk].gather(1, tok[1:][msk][:, None]).sum())
    return ok / tot, nll / tot

if __name__ == '__main__':
    W = walks()
    for ck in sys.argv[1:]:
        m = (load_ns(ck)[0] if "tw_normstep" in ck else load_tw(ck, "cpu")).eval()
        tok = W[0][0][None, :-1]; assert (fwd(m, tok, "eval") - m(tok)).abs().max() < 1e-4
        STATS.clear()
        r = {'eval': score(m, W, 'eval')}
        ratio = torch.cat(STATS['ratio']).median().item(); amax = torch.cat([z.flatten() for z in STATS['amax']]).median().item()
        r['attn'] = tuple(np.mean([score(m, W, 'attn', s) for s in range(3)], 0))
        r['renorm'] = tuple(np.mean([score(m, W, 'renorm', s) for s in range(3)], 0))
        for s in ('1.05', '1.111', '1.2', '1.4'):
            r['x' + s] = score(m, W, 'scale:' + s)
        print(ck.split('/runs/')[-1], ' '.join(f"{k} {a:.4f}/{n:.3f}" for k, (a, n) in r.items()),
              f"| eval median |attn out|/|resid| {ratio:.2f} median max-attn {amax:.2f}", flush=True)
```

Output:
```
tw_normstep/p0/MapWM_s12/MapWM.pt eval 0.8278/0.547 attn 0.9911/0.039 renorm 0.7501/1.101 x1.05 0.9897/0.048 x1.111 0.9931/0.027 x1.2 0.9092/0.200 x1.4 0.5801/3.709 | eval median |attn out|/|resid| 2.58 median max-attn 0.81
tw_normstep/p0/NormStepNB_s12/NormStepNB.pt eval 0.8278/0.753 attn 0.9943/0.026 renorm 0.7412/1.285 x1.05 0.9597/0.085 x1.111 0.9949/0.020 x1.2 0.9923/0.030 x1.4 0.6590/3.389 | eval median |attn out|/|resid| 1.37 median max-attn 0.51
```

Lean variant (`dropout_scale_lean.py`: modes eval, one train-mode dropout seed, x1.111 only; num_threads 2), all 32 tw_normstep runs:
```
tw_normstep/p0/DirOnly_s10/DirOnly.pt eval 0.9632/0.051 attn 0.9674/0.052 x1.111 0.9666/0.050 | eval median |attn out|/|resid| 5.16 median max-attn 0.17
tw_normstep/p0/DirOnly_s11/DirOnly.pt eval 0.9649/0.050 attn 0.9623/0.055 x1.111 0.9657/0.049 | eval median |attn out|/|resid| 5.85 median max-attn 0.13
tw_normstep/p0/DirOnly_s12/DirOnly.pt eval 0.9666/0.049 attn 0.9632/0.058 x1.111 0.9666/0.049 | eval median |attn out|/|resid| 4.14 median max-attn 0.09
tw_normstep/p0/DirOnly_s13/DirOnly.pt eval 0.9640/0.049 attn 0.9640/0.054 x1.111 0.9683/0.048 | eval median |attn out|/|resid| 4.85 median max-attn 0.12
tw_normstep/p0/DirOnly_s14/DirOnly.pt eval 0.9632/0.049 attn 0.9597/0.052 x1.111 0.9692/0.048 | eval median |attn out|/|resid| 4.54 median max-attn 0.11
tw_normstep/p0/DirOnly_s15/DirOnly.pt eval 0.9683/0.049 attn 0.9666/0.053 x1.111 0.9726/0.048 | eval median |attn out|/|resid| 5.52 median max-attn 0.09
tw_normstep/p0/DirOnly_s16/DirOnly.pt eval 0.9674/0.049 attn 0.9666/0.052 x1.111 0.9657/0.049 | eval median |attn out|/|resid| 4.28 median max-attn 0.14
tw_normstep/p0/DirOnly_s17/DirOnly.pt eval 0.9580/0.051 attn 0.9623/0.057 x1.111 0.9614/0.051 | eval median |attn out|/|resid| 5.00 median max-attn 0.09
tw_normstep/p0/MapWM_s10/MapWM.pt eval 1.0000/0.000 attn 1.0000/0.001 x1.111 1.0000/0.000 | eval median |attn out|/|resid| 3.54 median max-attn 0.94
tw_normstep/p0/MapWM_s11/MapWM.pt eval 0.9871/0.055 attn 0.9854/0.073 x1.111 0.9889/0.053 | eval median |attn out|/|resid| 2.69 median max-attn 0.50
tw_normstep/p0/MapWM_s12/MapWM.pt eval 0.8278/0.547 attn 0.9889/0.044 x1.111 0.9931/0.027 | eval median |attn out|/|resid| 2.58 median max-attn 0.81
tw_normstep/p0/MapWM_s13/MapWM.pt eval 0.9923/0.016 attn 0.9914/0.024 x1.111 0.9949/0.015 | eval median |attn out|/|resid| 3.14 median max-attn 0.47
tw_normstep/p0/MapWM_s14/MapWM.pt eval 0.9649/0.086 attn 0.9923/0.024 x1.111 0.9949/0.019 | eval median |attn out|/|resid| 1.77 median max-attn 0.61
tw_normstep/p0/MapWM_s15/MapWM.pt eval 0.9957/0.041 attn 0.9949/0.045 x1.111 0.9957/0.042 | eval median |attn out|/|resid| 2.67 median max-attn 0.61
tw_normstep/p0/MapWM_s16/MapWM.pt eval 1.0000/0.000 attn 0.9991/0.002 x1.111 1.0000/0.000 | eval median |attn out|/|resid| 3.77 median max-attn 0.74
tw_normstep/p0/MapWM_s17/MapWM.pt eval 0.9983/0.005 attn 0.9983/0.005 x1.111 0.9983/0.005 | eval median |attn out|/|resid| 3.43 median max-attn 0.74
tw_normstep/p0/NormStepNB_s10/NormStepNB.pt eval 0.8175/0.641 attn 0.9709/0.131 x1.111 0.9769/0.105 | eval median |attn out|/|resid| 1.49 median max-attn 0.59
tw_normstep/p0/NormStepNB_s11/NormStepNB.pt eval 0.9931/0.027 attn 0.9923/0.038 x1.111 0.9940/0.026 | eval median |attn out|/|resid| 2.28 median max-attn 0.62
tw_normstep/p0/NormStepNB_s12/NormStepNB.pt eval 0.8278/0.753 attn 0.9940/0.028 x1.111 0.9949/0.020 | eval median |attn out|/|resid| 1.37 median max-attn 0.51
tw_normstep/p0/NormStepNB_s13/NormStepNB.pt eval 1.0000/0.000 attn 1.0000/0.000 x1.111 1.0000/0.000 | eval median |attn out|/|resid| 3.43 median max-attn 0.75
tw_normstep/p0/NormStepNB_s14/NormStepNB.pt eval 0.9957/0.024 attn 0.9957/0.022 x1.111 0.9957/0.023 | eval median |attn out|/|resid| 4.26 median max-attn 0.40
tw_normstep/p0/NormStepNB_s15/NormStepNB.pt eval 1.0000/0.000 attn 1.0000/0.000 x1.111 1.0000/0.000 | eval median |attn out|/|resid| 3.01 median max-attn 0.67
tw_normstep/p0/NormStepNB_s16/NormStepNB.pt eval 0.8963/0.501 attn 0.9957/0.021 x1.111 0.9974/0.015 | eval median |attn out|/|resid| 1.91 median max-attn 0.60
tw_normstep/p0/NormStepNB_s17/NormStepNB.pt eval 1.0000/0.000 attn 1.0000/0.000 x1.111 1.0000/0.000 | eval median |attn out|/|resid| 3.26 median max-attn 0.77
tw_normstep/p0/NormStep_s10/NormStep.pt eval 1.0000/0.000 attn 1.0000/0.001 x1.111 1.0000/0.000 | eval median |attn out|/|resid| 3.96 median max-attn 0.81
tw_normstep/p0/NormStep_s11/NormStep.pt eval 0.9991/0.004 attn 0.9957/0.017 x1.111 0.9991/0.004 | eval median |attn out|/|resid| 2.34 median max-attn 0.49
tw_normstep/p0/NormStep_s12/NormStep.pt eval 1.0000/0.000 attn 0.9991/0.001 x1.111 1.0000/0.000 | eval median |attn out|/|resid| 3.87 median max-attn 0.82
tw_normstep/p0/NormStep_s13/NormStep.pt eval 1.0000/0.000 attn 0.9991/0.002 x1.111 1.0000/0.000 | eval median |attn out|/|resid| 2.95 median max-attn 0.75
tw_normstep/p0/NormStep_s14/NormStep.pt eval 1.0000/0.000 attn 1.0000/0.000 x1.111 1.0000/0.000 | eval median |attn out|/|resid| 3.88 median max-attn 0.76
tw_normstep/p0/NormStep_s15/NormStep.pt eval 0.9991/0.001 attn 0.9991/0.002 x1.111 0.9991/0.002 | eval median |attn out|/|resid| 3.12 median max-attn 0.70
tw_normstep/p0/NormStep_s16/NormStep.pt eval 0.8355/0.574 attn 0.9751/0.086 x1.111 0.9820/0.063 | eval median |attn out|/|resid| 2.13 median max-attn 0.46
tw_normstep/p0/NormStep_s17/NormStep.pt eval 0.9991/0.010 attn 0.9983/0.011 x1.111 0.9991/0.010 | eval median |attn out|/|resid| 3.65 median max-attn 0.81
```

### tw_scale_all.py

```python
"""Text-world (registered batch runs/textworld/p0): eval-mode accuracy vs eval with attention probabilities scaled by
1/(1-p) = 1.111 (mode-matched to inverted dropout), every arm, same 40 walks as dropout_mode_check.py.
Implemented by wrapping mapformer.model.F.softmax (the attention softmax of model.py's layer, which the RoPE baseline reuses)."""
import sys, types, numpy as np, torch
torch.set_num_threads(2); sys.path.insert(0, '/home/prashr')
import mapformer.model as MM
from mapformer.environment_textworld import TextWorld
from mapformer.analyze_textworld_secondary import load as load_tw
orig = MM.F
class _FW(types.ModuleType):
    scale = 1.0
    def __getattr__(self, k): return getattr(orig, k)
    def softmax(self, *a, **k): return orig.softmax(*a, **k) * self.scale
FW = _FW('Fw'); MM.F = FW
te = TextWorld(size=64, seed=10000); np.random.seed(10**6); W = [te.generate_trajectory(1024) for _ in range(40)]
@torch.no_grad()
def acc(m):
    ok = tot = 0
    for tok, _o, rev in W:
        lg = m(tok[None, :-1])[0]; msk = rev[1:]; ok += int((lg.argmax(-1)[msk] == tok[1:][msk]).sum()); tot += int(msk.sum())
    return ok / tot
for ck in sys.argv[1:]:
    m = load_tw(ck, 'cpu').eval(); FW.scale = 1.0; a = acc(m); FW.scale = 1 / 0.9; b = acc(m); FW.scale = 1.0
    print(f"{ck.split('/')[-2]:18s} eval {a:.4f}  x1.111 {b:.4f}  diff {b - a:+.4f}", flush=True)
```

Output (runs/textworld/p0, all 24 runs):
```
RoPE_L1_s0         eval 0.5090  x1.111 0.5090  diff +0.0000
RoPE_L1_s1         eval 0.5107  x1.111 0.5099  diff -0.0009
RoPE_L1_s2         eval 0.5021  x1.111 0.5056  diff +0.0034
RoPE_L1_s3         eval 0.5064  x1.111 0.5099  diff +0.0034
RoPE_L1_s4         eval 0.5030  x1.111 0.5013  diff -0.0017
RoPE_L1_s5         eval 0.5133  x1.111 0.5167  diff +0.0034
RoPE_L1_s6         eval 0.5064  x1.111 0.5039  diff -0.0026
RoPE_L1_s7         eval 0.5030  x1.111 0.5004  diff -0.0026
RoPE_L2_s0         eval 0.7121  x1.111 0.6975  diff -0.0146
RoPE_L2_s1         eval 0.7541  x1.111 0.7498  diff -0.0043
RoPE_L2_s2         eval 0.9374  x1.111 0.9366  diff -0.0009
RoPE_L2_s3         eval 0.7232  x1.111 0.7164  diff -0.0069
RoPE_L2_s4         eval 0.9297  x1.111 0.9340  diff +0.0043
RoPE_L2_s5         eval 0.7258  x1.111 0.7104  diff -0.0154
RoPE_L2_s6         eval 0.7806  x1.111 0.7789  diff -0.0017
RoPE_L2_s7         eval 0.7369  x1.111 0.7386  diff +0.0017
Vanilla_r4_L1_s0   eval 0.8835  x1.111 0.9743  diff +0.0908
Vanilla_r4_L1_s1   eval 0.9991  x1.111 0.9991  diff +0.0000
Vanilla_r4_L1_s2   eval 0.9983  x1.111 0.9983  diff +0.0000
Vanilla_r4_L1_s3   eval 0.9949  x1.111 0.9949  diff +0.0000
Vanilla_r4_L1_s4   eval 0.9974  x1.111 0.9974  diff +0.0000
Vanilla_r4_L1_s5   eval 1.0000  x1.111 1.0000  diff +0.0000
Vanilla_r4_L1_s6   eval 0.9991  x1.111 0.9991  diff +0.0000
Vanilla_r4_L1_s7   eval 0.8955  x1.111 0.9880  diff +0.0925
```

### leak_gap.py

```python
"""Leak cost by revisit gap (new-object task, MapWM). Derivation: object steps eta_o add a random-walk phase error
over the interval, Var ~ (#objects in interval) * sigma_eta^2, so the cost of the leak (zeroed - intact accuracy)
should GROW with the gap since the last visit. Same eval stream as leak_eval.py (test pool, x1), first N_SEQ sequences."""
import sys, numpy as np, torch
torch.set_num_threads(1); sys.path.insert(0, '/home/prashr')
from mapformer.leak_eval import load, Probe, SIZE, P, EVAL_SEED, ENV_SEED, T_STEPS
from mapformer.environment_newobj import NewObjectWorld, N_SPECIAL
from mapformer.model_codes import set_pool
ck, arm, nseq = sys.argv[1], sys.argv[2], int(sys.argv[3])
m, b = load(ck, arm, 'cpu'); probe = Probe(m); set_pool(m, 'test')
env = NewObjectWorld(size=SIZE, seed=ENV_SEED, pool_size=P, pool='test'); np.random.seed(EVAL_SEED)
bins = [1, 8, 32, 128, 512, 1025]; acc = {k: [[0, 0] for _ in bins[:-1]] for k in ('intact', 'zobj')}
lo = N_SPECIAL + P
with torch.no_grad():
    for _ in range(nseq):
        tok, _o, rev = env.generate_trajectory(T_STEPS); locs = env.visited_locations
        last = {}; gap = np.zeros(len(tok), int)
        for k, L in enumerate(locs):          # object slot of move k is token 2k+1
            L = tuple(L); gap[2 * k + 1] = k - last[L] if L in last else 0; last[L] = k
        inp = tok[None, :-1]
        for mode in ('intact', 'zobj'):
            probe.keep = (inp < N_SPECIAL) if mode == 'zobj' else None
            pred = m(inp)[0, :, lo:lo + P].argmax(-1) + lo; probe.keep = None
            tgt = tok[1:]; msk = (rev[1:] & (tgt >= N_SPECIAL)).numpy(); ok = (pred == tgt).numpy(); g = gap[1:]
            for i in range(len(bins) - 1):
                sel = msk & (g >= bins[i]) & (g < bins[i + 1]); acc[mode][i][0] += int(ok[sel].sum()); acc[mode][i][1] += int(sel.sum())
row = []
for i in range(len(bins) - 1):
    a = acc['intact'][i]; z = acc['zobj'][i]
    if a[1]: row.append(f"gap [{bins[i]},{bins[i+1]}): n {a[1]:5d} intact {a[0]/a[1]:.4f} leak {z[0]/z[1]-a[0]/a[1]:+.4f}")
print(ck.split('/')[-2], ' | '.join(row), flush=True)
```

Output (run with num_threads 1, 40 sequences, 8 MapWM seeds):
```
MapWM_s0 gap [1,8): n  2467 intact 1.0000 leak +0.0000 | gap [8,32): n  2275 intact 1.0000 leak +0.0000 | gap [32,128): n  1117 intact 1.0000 leak +0.0000 | gap [128,512): n  2583 intact 0.9973 leak +0.0027 | gap [512,1025): n   889 intact 0.9359 leak +0.0630
MapWM_s1 gap [1,8): n  2467 intact 1.0000 leak +0.0000 | gap [8,32): n  2275 intact 1.0000 leak +0.0000 | gap [32,128): n  1117 intact 1.0000 leak +0.0000 | gap [128,512): n  2583 intact 0.9954 leak +0.0046 | gap [512,1025): n   889 intact 0.9494 leak +0.0506
MapWM_s2 gap [1,8): n  2467 intact 1.0000 leak +0.0000 | gap [8,32): n  2275 intact 1.0000 leak +0.0000 | gap [32,128): n  1117 intact 1.0000 leak +0.0000 | gap [128,512): n  2583 intact 0.9903 leak +0.0097 | gap [512,1025): n   889 intact 0.9168 leak +0.0832
MapWM_s3 gap [1,8): n  2467 intact 1.0000 leak +0.0000 | gap [8,32): n  2275 intact 1.0000 leak +0.0000 | gap [32,128): n  1117 intact 1.0000 leak +0.0000 | gap [128,512): n  2583 intact 0.9903 leak +0.0097 | gap [512,1025): n   889 intact 0.9021 leak +0.0979
MapWM_s4 gap [1,8): n  2467 intact 0.9996 leak +0.0004 | gap [8,32): n  2275 intact 1.0000 leak +0.0000 | gap [32,128): n  1117 intact 1.0000 leak +0.0000 | gap [128,512): n  2583 intact 0.9822 leak +0.0178 | gap [512,1025): n   889 intact 0.8785 leak +0.1215
MapWM_s5 gap [1,8): n  2467 intact 1.0000 leak +0.0000 | gap [8,32): n  2275 intact 1.0000 leak +0.0000 | gap [32,128): n  1117 intact 1.0000 leak +0.0000 | gap [128,512): n  2583 intact 0.9957 leak +0.0043 | gap [512,1025): n   889 intact 0.9123 leak +0.0866
MapWM_s6 gap [1,8): n  2467 intact 1.0000 leak +0.0000 | gap [8,32): n  2275 intact 1.0000 leak +0.0000 | gap [32,128): n  1117 intact 1.0000 leak +0.0000 | gap [128,512): n  2583 intact 0.9919 leak +0.0081 | gap [512,1025): n   889 intact 0.9460 leak +0.0506
MapWM_s7 gap [1,8): n  2467 intact 1.0000 leak +0.0000 | gap [8,32): n  2275 intact 0.9996 leak +0.0004 | gap [32,128): n  1117 intact 1.0000 leak +0.0000 | gap [128,512): n  2583 intact 0.9961 leak +0.0039 | gap [512,1025): n   889 intact 0.9449 leak +0.0472
```

### aside_vote.py

```python
"""Predict DirOnly's ceiling from the grammar alone. With zero steps on every non-direction word, an aside noun
sits at the phase of the cell where the aside was told, and a 1-layer key depends on the token only, so located
and aside mentions of the cell are indistinguishable by score. Two readout idealisations:
 count-vote: the object with the most mentions at the cell wins (ties split);  recency: the latest mention wins.
Same eval walks as the registered readout (TextWorld(size=64, seed=10000), np seed 10**6, 40 walks, T=1024)."""
import sys, numpy as np, collections
sys.path.insert(0, '/home/prashr')
import mapformer.environment_textworld as E
te = E.TextWorld(size=64, seed=10000); np.random.seed(10**6)
obj = set(te.obj_ids); dirw = {i for v in te.dir_ids.values() for i in v}
cv = rc = n = conf = 0
for _ in range(40):
    tok, om, rev = te.generate_trajectory(1024); tok = tok.tolist(); om = om.tolist(); rev = rev.tolist()
    locs = te.visited_locations; k = 0; cell = None
    ment = collections.defaultdict(list)            # cell -> [(time, object, is_located)]
    # walk tokens: the cell changes at each object slot (one per rendered step); asides follow the object slot
    for t, w in enumerate(tok):
        if om[t]:
            cell = locs[k]; k += 1
            if rev[t]:
                M = ment[cell]; tgt = w
                cnt = collections.Counter(o for _, o, _ in M); best = max(cnt.values())
                winners = [o for o, c in cnt.items() if c == best]
                cv += (tgt in winners) / len(winners); rc += (M[-1][1] == tgt); n += 1
                conf += any((not loc) and o != tgt for _, o, loc in M)
            ment[cell].append((t, w, True))
        elif w in obj and cell is not None:          # an aside noun (objects appear only in object slots or asides)
            ment[cell].append((t, w, False))
print(f"revisit targets {n}; with a different aside object at the cell {conf/n:.3f}")
print(f"count-vote predicted accuracy {cv/n:.4f};  recency predicted accuracy {rc/n:.4f}")
```

Output:
```
revisit targets 1167; with a different aside object at the cell 0.186
count-vote predicted accuracy 0.9362;  recency predicted accuracy 0.8380
```

### aside_attn.py

```python
"""DirOnly: attention mass (eval mode, mean over heads) per mention at the queried cell: located object tokens vs
aside nouns, on confusable revisit targets. Scores of the two kinds are identical if keys depend on the token
and the phase only (1 layer, zero steps on non-direction words)."""
import sys, math, numpy as np, torch, torch.nn.functional as F
torch.set_num_threads(4); sys.path.insert(0, '/home/prashr')
from mapformer.environment_textworld import TextWorld
from mapformer.tw_normstep_readouts import load
from mapformer.model_textstep import step_of
from mapformer.model import _apply_rope
ck = sys.argv[1]; m, arm = load(ck)
te = TextWorld(size=64, seed=10000); np.random.seed(10**6)
obj = set(te.obj_ids); L = m.layers[0]
loc_w, asd_w, loc_n, asd_n, other = [], [], [], [], []
for _ in range(40):
    tok, om, rev = te.generate_trajectory(1024); locs = te.visited_locations
    x = tok[None, :-1]
    with torch.no_grad():
        e = m.token_emb(x); d = step_of(m, x); cos_a, sin_a = m.path_integrator(d)
        T = x.shape[1]; h = L.norm1(e)
        Q = _apply_rope(L.q_proj(h).view(1, T, L.n_heads, L.d_head).transpose(1, 2), cos_a, sin_a)
        K = _apply_rope(L.k_proj(h).view(1, T, L.n_heads, L.d_head).transpose(1, 2), cos_a, sin_a)
        A = F.softmax((Q @ K.transpose(-1, -2) / math.sqrt(L.d_head)).masked_fill(torch.triu(torch.ones(T, T, dtype=torch.bool), 1), float('-inf')), -1)[0].mean(0).numpy()
    t = tok.tolist(); omv = om.tolist(); rv = rev.tolist(); k = 0; cell = None; ment = {}
    for i, w in enumerate(t):
        if omv[i]:
            cell = locs[k]; k += 1
            if rv[i] and i - 1 < T:
                M = ment.get(cell, [])
                if any(not lo and o != w for _, o, lo in M):
                    a = A[i - 1]
                    lw = [a[j] for j, o, lo in M if lo]; aw = [a[j] for j, o, lo in M if not lo]
                    loc_w += lw; asd_w += aw; loc_n.append(sum(lw)); asd_n.append(sum(aw))
            ment.setdefault(cell, []).append((i, w, True))
        elif w in obj and cell is not None:
            ment.setdefault(cell, []).append((i, w, False))
print(f"{arm}: confusable targets {len(loc_n)}; mean attention per located mention {np.mean(loc_w):.4f}, per aside mention {np.mean(asd_w):.4f}; "
      f"total on located {np.mean(loc_n):.3f}, on asides {np.mean(asd_n):.3f}")
```

Output (DirOnly s10, MapWM s16, NormStep s12, MapWM s10 as first run):
```
DirOnly: confusable targets 217; mean attention per located mention 0.3799, per aside mention 0.2069; total on located 0.527, on asides 0.215
MapWM s16: confusable targets 217; mean attention per located mention 0.5827, per aside mention 0.0626; total on located 0.808, on asides 0.065
NormStep s12: confusable targets 217; mean attention per located mention 0.6067, per aside mention 0.0362; total on located 0.842, on asides 0.038
MapWM s10: confusable targets 217; mean attention per located mention 0.4635, per aside mention 0.0171; total on located 0.643, on asides 0.018
```

`aside_attn2.py` = aside_attn.py plus, per confusable target, the mean over channels of |wrapped phase(aside noun) - phase(located object)| and the layer-1 head-mean pre-softmax score difference (located - aside); num_threads 2. The added lines:
```python
torch.set_num_threads(2); sys.path.insert(0, '/home/prashr')
loc_w, asd_w, loc_n, asd_n, other = [], [], [], [], []; dphi = []; dsc = []
        SC = (Q @ K.transpose(-1, -2) / math.sqrt(L.d_head)); TH = (torch.cumsum(d, 1) * m.path_integrator.omega)[0].numpy(); A = F.softmax(SC.masked_fill(torch.triu(torch.ones(T, T, dtype=torch.bool), 1), float('-inf')), -1)[0].mean(0).numpy()
                    sc = SC[0].mean(0).numpy()[i - 1]
                    for jl, ol, lol in M:
                        if not lol: continue
                        for ja, oa, loa in M:
                            if loa: continue
                            dphi.append(np.abs((TH[ja] - TH[jl] + np.pi) % (2*np.pi) - np.pi).mean()); dsc.append(sc[jl] - sc[ja])
print(f"mean |phase(aside noun) - phase(located obj)| over channels {np.mean(dphi):.3f} rad (median {np.median(dphi):.3f}); score(located) - score(aside) mean {np.mean(dsc):.2f}"); print(f"{arm}: confusable targets {len(loc_n)}; mean attention per located mention {np.mean(loc_w):.4f}, per aside mention {np.mean(asd_w):.4f}; "
```
Output (16 runs; DirOnly s10 run separately: phase 0.038 rad (median 0.000), score gap 0.92):
```
MapWM_s10: mean |phase(aside noun) - phase(located obj)| over channels 0.109 rad (median 0.048); score(located) - score(aside) mean 6.37 MapWM: confusable targets 217; mean attention per located mention 0.4635, per aside mention 0.0171; total on located 0.643, on asides 0.018 
MapWM_s11: mean |phase(aside noun) - phase(located obj)| over channels 0.334 rad (median 0.014); score(located) - score(aside) mean 1.37 MapWM: confusable targets 217; mean attention per located mention 0.4525, per aside mention 0.1498; total on located 0.628, on asides 0.156 
MapWM_s12: mean |phase(aside noun) - phase(located obj)| over channels 0.092 rad (median 0.034); score(located) - score(aside) mean 11.34 MapWM: confusable targets 217; mean attention per located mention 0.2731, per aside mention 0.0000; total on located 0.379, on asides 0.000 
MapWM_s13: mean |phase(aside noun) - phase(located obj)| over channels 0.253 rad (median 0.015); score(located) - score(aside) mean 1.40 MapWM: confusable targets 217; mean attention per located mention 0.4854, per aside mention 0.1593; total on located 0.673, on asides 0.166 
MapWM_s14: mean |phase(aside noun) - phase(located obj)| over channels 0.263 rad (median 0.015); score(located) - score(aside) mean 5.67 MapWM: confusable targets 217; mean attention per located mention 0.3484, per aside mention 0.0051; total on located 0.483, on asides 0.005 
MapWM_s15: mean |phase(aside noun) - phase(located obj)| over channels 0.302 rad (median 0.039); score(located) - score(aside) mean 2.96 MapWM: confusable targets 217; mean attention per located mention 0.5609, per aside mention 0.0674; total on located 0.778, on asides 0.070 
MapWM_s16: mean |phase(aside noun) - phase(located obj)| over channels 0.069 rad (median 0.048); score(located) - score(aside) mean 3.26 MapWM: confusable targets 217; mean attention per located mention 0.5827, per aside mention 0.0626; total on located 0.808, on asides 0.065 
MapWM_s17: mean |phase(aside noun) - phase(located obj)| over channels 0.116 rad (median 0.048); score(located) - score(aside) mean 3.82 MapWM: confusable targets 217; mean attention per located mention 0.5484, per aside mention 0.0492; total on located 0.761, on asides 0.051 
NormStep_s10: mean |phase(aside noun) - phase(located obj)| over channels 0.076 rad (median 0.055); score(located) - score(aside) mean 4.52 NormStep: confusable targets 217; mean attention per located mention 0.6039, per aside mention 0.0372; total on located 0.838, on asides 0.039 
NormStep_s11: mean |phase(aside noun) - phase(located obj)| over channels 0.273 rad (median 0.028); score(located) - score(aside) mean 1.72 NormStep: confusable targets 217; mean attention per located mention 0.4698, per aside mention 0.1227; total on located 0.652, on asides 0.128 
NormStep_s12: mean |phase(aside noun) - phase(located obj)| over channels 0.064 rad (median 0.048); score(located) - score(aside) mean 4.24 NormStep: confusable targets 217; mean attention per located mention 0.6067, per aside mention 0.0362; total on located 0.842, on asides 0.038 
NormStep_s13: mean |phase(aside noun) - phase(located obj)| over channels 0.113 rad (median 0.065); score(located) - score(aside) mean 4.63 NormStep: confusable targets 217; mean attention per located mention 0.5817, per aside mention 0.0392; total on located 0.807, on asides 0.041 
NormStep_s14: mean |phase(aside noun) - phase(located obj)| over channels 0.136 rad (median 0.078); score(located) - score(aside) mean 4.18 NormStep: confusable targets 217; mean attention per located mention 0.5904, per aside mention 0.0511; total on located 0.819, on asides 0.053 
NormStep_s15: mean |phase(aside noun) - phase(located obj)| over channels 0.124 rad (median 0.076); score(located) - score(aside) mean 3.08 NormStep: confusable targets 217; mean attention per located mention 0.5689, per aside mention 0.0725; total on located 0.789, on asides 0.076 
NormStep_s16: mean |phase(aside noun) - phase(located obj)| over channels 0.263 rad (median 0.018); score(located) - score(aside) mean 5.68 NormStep: confusable targets 217; mean attention per located mention 0.2424, per aside mention 0.0230; total on located 0.336, on asides 0.024 
NormStep_s17: mean |phase(aside noun) - phase(located obj)| over channels 0.097 rad (median 0.070); score(located) - score(aside) mean 4.09 NormStep: confusable targets 217; mean attention per located mention 0.5906, per aside mention 0.0436; total on located 0.819, on asides 0.045 
```

### drift_split.py

```python
"""Split the registered per-word drift (tw_normstep_readouts.drift, 'opt') into aside-sentence positions vs the other
optional positions (adverbs, fillers). Same walks (map 10000, np seed 0, 40 walks, T=1024) and same statistic: mean over
revisit pairs of |wrapped dtheta| per channel, then mean over the 64 channels (rad). Also: attention-score gap
located-minus-aside on confusable targets is measured elsewhere (aside_attn2.py)."""
import sys, numpy as np, torch
torch.set_num_threads(2); sys.path.insert(0, '/home/prashr')
from mapformer.environment_textworld import TextWorld, VERBS
from mapformer.tw_normstep_readouts import load, core_mask
from mapformer.model_textstep import step_of
from mapformer.train_tw_normstep import HELDOUT
wrap = lambda x: (x + np.pi) % (2 * np.pi) - np.pi
for ck in sys.argv[1:]:
    m, arm = load(ck); te = TextWorld(size=64, seed=HELDOUT); om = m.path_integrator.omega.detach().numpy()
    verbs = set(te.idx[w] for w in VERBS); dot = te.idx['.']
    np.random.seed(0); dif = {"opt": [], "aside": [], "other_opt": []}
    with torch.no_grad():
        for _ in range(40):
            tok, obs, _ = te.generate_trajectory(1024); locs = te.visited_locations
            d = step_of(m, tok[None])[0].numpy(); t = tok.numpy()
            slots = np.nonzero(obs.numpy())[0][:len(locs)]; core = core_mask(te, t, slots)
            aside = np.zeros(len(t), bool)
            for i in slots:
                j = i + 2
                if j < len(t) and t[j] not in verbs:
                    while j < len(t):
                        aside[j] = True
                        if t[j] == dot: break
                        j += 1
            masks = {"opt": ~core, "aside": aside & ~core, "other_opt": ~core & ~aside}
            ths = {k: np.cumsum(d * v[:, None, None], 0) * om[None] for k, v in masks.items()}
            first = {}
            for k, i in enumerate(slots):
                L = tuple(locs[k])
                if L in first:
                    for key, th in ths.items(): dif[key].append(wrap(th[i] - th[first[L]]))
                else: first[L] = i
    r = {k: float(np.abs(np.array(v)).mean(0).mean()) for k, v in dif.items()}
    print(f"{arm:10s} {ck.split('/')[-2]:14s} per-word drift (rad): all optional {r['opt']:.3f}  aside sentences {r['aside']:.3f}  adverbs/fillers {r['other_opt']:.3f}", flush=True)
```

Output:
```
MapWM      MapWM_s10      per-word drift (rad): all optional 0.097  aside sentences 0.099  adverbs/fillers 0.031
MapWM      MapWM_s11      per-word drift (rad): all optional 0.037  aside sentences 0.032  adverbs/fillers 0.009
MapWM      MapWM_s12      per-word drift (rad): all optional 0.071  aside sentences 0.072  adverbs/fillers 0.035
MapWM      MapWM_s13      per-word drift (rad): all optional 0.040  aside sentences 0.034  adverbs/fillers 0.010
MapWM      MapWM_s14      per-word drift (rad): all optional 0.046  aside sentences 0.033  adverbs/fillers 0.022
MapWM      MapWM_s15      per-word drift (rad): all optional 0.111  aside sentences 0.091  adverbs/fillers 0.028
MapWM      MapWM_s16      per-word drift (rad): all optional 0.118  aside sentences 0.115  adverbs/fillers 0.008
MapWM      MapWM_s17      per-word drift (rad): all optional 0.092  aside sentences 0.103  adverbs/fillers 0.020
NormStep   NormStep_s10   per-word drift (rad): all optional 0.130  aside sentences 0.128  adverbs/fillers 0.009
NormStep   NormStep_s11   per-word drift (rad): all optional 0.072  aside sentences 0.064  adverbs/fillers 0.014
NormStep   NormStep_s12   per-word drift (rad): all optional 0.116  aside sentences 0.113  adverbs/fillers 0.011
NormStep   NormStep_s13   per-word drift (rad): all optional 0.159  aside sentences 0.149  adverbs/fillers 0.019
NormStep   NormStep_s14   per-word drift (rad): all optional 0.192  aside sentences 0.183  adverbs/fillers 0.013
NormStep   NormStep_s15   per-word drift (rad): all optional 0.187  aside sentences 0.173  adverbs/fillers 0.020
NormStep   NormStep_s16   per-word drift (rad): all optional 0.042  aside sentences 0.038  adverbs/fillers 0.025
NormStep   NormStep_s17   per-word drift (rad): all optional 0.169  aside sentences 0.162  adverbs/fillers 0.013
```

### aside_zero.py

```python
"""Prediction 4.3a: zero the steps at ASIDE-SENTENCE positions only (eval). Learned-step models should then (i) make
DirOnly-type errors (an aside object at the cell) and (ii) lose most of the registered drift_opt. 40 walks (map 10000,
np seed 10**6), eval mode; masking as tw_normstep_readouts.ablate (hook on action_to_lie, or m.step override)."""
import sys, numpy as np, torch
torch.set_num_threads(2); sys.path.insert(0, '/home/prashr')
from mapformer.environment_textworld import TextWorld, VERBS
from mapformer.tw_normstep_readouts import load
ck = sys.argv[1]; m, arm = load(ck)
te = TextWorld(size=64, seed=10000); np.random.seed(10**6); W = [te.generate_trajectory(1024) + (list(te.visited_locations),) for _ in range(40)]
verbs = set(te.idx[w] for w in VERBS); dot = te.idx['.']; obj = set(te.obj_ids)
keep = {}
orig = m.step if hasattr(m, 'step') else None
if orig is None:
    m.action_to_lie.register_forward_hook(lambda mod, i, o: o if 'k' not in keep else o * keep['k'][..., None, None].to(o.dtype))
else:
    m.step = lambda tokens, x: orig(tokens, x) * (keep['k'][..., None, None].to(x.dtype) if 'k' in keep else 1)
def aside_mask(t, om):
    a = np.zeros(len(t), bool); slots = np.nonzero(om)[0]
    for i in slots:
        j = i + 2
        if j < len(t) and t[j] not in verbs:
            while j < len(t):
                a[j] = True
                if t[j] == dot: break
                j += 1
    return a
res = {}
with torch.no_grad():
    for mode in ('intact', 'aside_zeroed'):
        ok = tot = asd_err = 0
        for tok, om, rev, locs in W:
            t = tok.numpy(); inp = tok[None, :-1]
            if mode == 'aside_zeroed': keep['k'] = torch.tensor(~aside_mask(t, om.numpy())[:-1])[None]
            else: keep.pop('k', None)
            pred = m(inp)[0].argmax(-1).numpy(); msk = rev[1:].numpy(); tg = t[1:]
            # aside objects told at each cell before each target
            k = 0; cell = None; ment = {}
            for i in range(len(t)):
                if om[i]:
                    cell = tuple(locs[k]); k += 1
                    if rev[i] and i >= 1:
                        tot += 1; ok += pred[i - 1] == t[i]
                        asd_err += (pred[i - 1] != t[i]) and (pred[i - 1] in ment.get(cell, set()))
                elif t[i] in obj and cell is not None:
                    ment.setdefault(cell, set()).add(t[i])
        keep.pop('k', None); res[mode] = (ok / tot, asd_err, tot - ok)
print(f"{arm:9s} {ck.split('/')[-2]:14s} " + '  '.join(f"{k}: acc {a:.4f} errors {e} of which aside-object {s}" for k, (a, s, e) in res.items()), flush=True)
```

Output:
```
MapWM     MapWM_s10      intact: acc 1.0000 errors 0 of which aside-object 0  aside_zeroed: acc 0.9880 errors 14 of which aside-object 9
MapWM     MapWM_s16      intact: acc 1.0000 errors 0 of which aside-object 0  aside_zeroed: acc 0.9906 errors 11 of which aside-object 2
MapWM     MapWM_s17      intact: acc 0.9983 errors 2 of which aside-object 0  aside_zeroed: acc 0.9914 errors 10 of which aside-object 8
NormStep  NormStep_s10   intact: acc 1.0000 errors 0 of which aside-object 0  aside_zeroed: acc 0.9949 errors 6 of which aside-object 4
NormStep  NormStep_s12   intact: acc 1.0000 errors 0 of which aside-object 0  aside_zeroed: acc 0.9871 errors 15 of which aside-object 2
NormStep  NormStep_s13   intact: acc 1.0000 errors 0 of which aside-object 0  aside_zeroed: acc 0.9931 errors 8 of which aside-object 8
```

### lagmix.py

```python
"""Best 'lag-mixture' predictor on the H3 ring task (CANCEL): a 1-layer index-attention idealisation.
Score s(query action, key is blank?, lag L) over previous observation keys + one sink with a learned
output distribution; prediction = softmax-weighted mixture of copied one-hots. Fit by Adam on NLL over
simulated revisits (environment_cancel's walk, T=128, ring 32), report argmax accuracy on fresh walks."""
import sys, numpy as np, torch, torch.nn.functional as F
torch.set_num_threads(4)
sys.path.insert(0, '/home/prashr')
from mapformer.environment_cancel import GridWorldCancel

def data(p, n, T=128, seed=0):
    np.random.seed(seed); env = GridWorldCancel(size=32, p_plus=p, seed=0)
    A, O, R = [], [], []
    for _ in range(n):
        tok, _, rev = env.generate_trajectory(T)
        A.append(tok[0::2].numpy()); O.append(tok[1::2].numpy() - env.obs_offset); R.append(rev[1::2].numpy())
    return np.array(A), np.array(O), np.array(R)

def fit_eval(p, T=128, ntr=3000, nte=1000, steps=1500, K=17):
    A, O, R = data(p, ntr, T, 1); At, Ot, Rt = data(p, nte, T, 2)
    S = torch.zeros(2, 2, T, requires_grad=True)        # action, key-blank, lag(1..T-1)
    sink = torch.zeros(2, 1, requires_grad=True); prior = torch.zeros(2, K, requires_grad=True)
    opt = torch.optim.Adam([S, sink, prior], lr=0.05)
    def batch(A, O, R):
        b, t = np.nonzero(R); a = torch.tensor(A[b, t]); tgt = torch.tensor(O[b, t])
        lags = torch.arange(1, T)                                       # L
        src = torch.tensor(t)[:, None] - lags[None]                     # n, L
        valid = src >= 0
        ko = torch.tensor(O)[torch.tensor(b)[:, None], src.clamp_min(0)]   # n, L
        return a, tgt, lags, valid, ko
    def logp(a, lags, valid, ko):
        sc = S[a[:, None], (ko == K - 1).long(), lags[None] - 1].masked_fill(~valid, -1e9)   # n, L
        sc = torch.cat([sc, sink[a]], 1); w = sc.softmax(1)
        prob = torch.zeros(len(a), K).scatter_add_(1, ko * valid, w[:, :-1] * valid)
        prob = prob + w[:, -1:] * prior[a].softmax(1)
        return (prob + 1e-6).log()
    tr = batch(A, O, R); te = batch(At, Ot, Rt)
    for i in range(steps):
        idx = torch.randint(0, len(tr[0]), (4096,))
        lp = logp(tr[0][idx], tr[2], tr[3][idx], tr[4][idx])
        loss = F.nll_loss(lp, tr[1][idx]); opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        lp = logp(te[0], te[2], te[3], te[4]); acc = (lp.argmax(1) == te[1]).float().mean().item()
        # best single lag (oracle choice per query action), for reference
        best = 0
        for a in (0, 1):
            m = te[0] == a
            if m.sum() == 0: continue
            eq = ((te[4][m] == te[1][m][:, None]) & te[3][m]).float().mean(0)
            best += eq.max().item() * m.float().mean().item()
    return acc, best

if __name__ == '__main__':
    for p in (0.5, 0.75, 0.9, 1.0):
        acc, best = fit_eval(p)
        print(f"p_plus {p}: lag-mixture acc {acc:.3f}   best single lag {best:.3f}", flush=True)
```

Output:
```
p_plus 0.5: lag-mixture acc 0.498   best single lag 0.352
p_plus 0.75: lag-mixture acc 0.505   best single lag 0.414
p_plus 0.9: lag-mixture acc 0.670   best single lag 0.648
p_plus 1.0: lag-mixture acc 1.000   best single lag 1.000
```

### toy.py

```python
"""Small-scale MapWM on a D-torus, CPU. Uses the repo's MapFormerWM layer code with a per-head rank-r step map.
Map redrawn per sequence (in-context map). Loss on revisited observations only (paper)."""
import sys, math, time, json, argparse, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, '/home/prashr')
from mapformer.model import MapFormerWM
from mapformer.model_rank_perhead import ActionToLieAlgebraPerHead

def walks(rng, B, T, D, N, K=16, p_empty=0.5):
    nA = 2 * D
    deltas = np.zeros((nA, D), int)
    for i in range(D): deltas[2*i, i] = 1; deltas[2*i+1, i] = -1
    acts = np.empty((B, T), int)
    for b in range(B):
        t = 0
        while t < T:
            a = rng.integers(nA); k = rng.integers(1, 11)
            acts[b, t:t+k] = a; t += k
    unw = np.cumsum(deltas[acts], 1) + rng.integers(0, N, (B, 1, D))      # unwrapped positions
    pos = unw % N
    cell = np.ravel_multi_index(tuple(pos[..., i] for i in range(D)), (N,) * D)   # B,T
    maps = np.where(rng.random((B, N ** D)) < p_empty, K, rng.integers(0, K, (B, N ** D)))
    obs = np.take_along_axis(maps, cell, 1)
    tok = np.empty((B, 2 * T), int); tok[:, 0::2] = acts; tok[:, 1::2] = obs + nA
    rev = np.zeros((B, T), bool); wrap = np.zeros((B, T), bool)
    for b in range(B):
        seen_c, seen_u = set(), set()
        for t in range(T):
            c = cell[b, t]; u = tuple(unw[b, t])
            if c in seen_c:
                rev[b, t] = True; wrap[b, t] = u not in seen_u
            seen_c.add(c); seen_u.add(u)
    return torch.tensor(tok), torch.tensor(rev), torch.tensor(wrap)

class AntiSym(nn.Module):
    """Step map with an exact antisymmetric action code and zero observation steps (no clock possible)."""
    def __init__(self, base, D):
        super().__init__(); self.base = base; self.D = D
    def forward(self, x):
        d = self.base(x)                     # B,T,H,nb (x is the embedding sequence)
        return d
def build(args, vocab):
    m = MapFormerWM(vocab, d_model=args.d, n_heads=args.H, n_layers=1, dropout=0.0, grid_size=args.N)
    m.action_to_lie = ActionToLieAlgebraPerHead(args.d, args.H, m.n_blocks, args.r)
    return m

def run(args):
    torch.manual_seed(args.seed); rng = np.random.default_rng(1000 + args.seed)
    D, N, T = args.D, args.N, args.T; nA = 2 * D; vocab = nA + 17
    m = build(args, vocab)
    if args.antisym:
        # step(token) := A(e_token) - A(e_partner) for actions (partner = opposite), 0 for observations
        a2l = m.action_to_lie; partner = torch.arange(vocab)
        for i in range(D): partner[2*i] = 2*i+1; partner[2*i+1] = 2*i
        isact = (torch.arange(vocab) < nA).float()
        emb = m.token_emb
        orig = a2l.forward
        def fwd(x, _o=orig):
            return _o(x)
        m._antisym = (partner, isact)
    opt = torch.optim.AdamW(m.parameters(), lr=args.lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.05,
                                                anneal_strategy='cos')
    def forward(tok):
        if not args.antisym: return m(tok)
        partner, isact = m._antisym
        x = m.token_emb(tok); xp = m.token_emb(partner[tok])
        delta = (m.action_to_lie(x) - m.action_to_lie(xp)) * 0.5 * isact[tok][..., None, None]
        cos_a, sin_a = m.path_integrator(delta)
        L = tok.shape[1]; mask = torch.triu(torch.ones(L, L, dtype=torch.bool), 1)
        for layer in m.layers: x = layer(x, cos_a, sin_a, mask)
        return m.out_proj(m.out_norm(x))
    hist = []
    for step in range(args.steps):
        tok, rev, _ = walks(rng, args.B, T, D, N)
        logits = forward(tok)[:, 0::2]                     # at action tokens -> predict next obs
        tgt = tok[:, 1::2]
        loss = F.cross_entropy(logits[rev], tgt[rev])
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        hist.append(loss.item())
    m.eval(); erng = np.random.default_rng(99)
    with torch.no_grad():
        tok, rev, wrap = walks(erng, 64, T, D, N)
        pred = forward(tok)[:, 0::2].argmax(-1); ok = pred == tok[:, 1::2]
    fl = float(np.mean(hist[-max(1, args.steps // 20):]))
    return dict(seed=args.seed, r=args.r, D=D, N=N, T=T, antisym=args.antisym, final_loss=fl,
                acc=float(ok[rev].float().mean()), acc_wrap=float(ok[rev & wrap].float().mean()),
                acc_other=float(ok[rev & ~wrap].float().mean()), wrap_share=float((rev & wrap).sum() / rev.sum()),
                curve=[float(np.mean(hist[i:i+100])) for i in range(0, args.steps, 100)])

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    for k, v in dict(D=2, N=12, T=192, r=2, H=2, d=64, B=32, steps=3000, lr=3e-3, seed=0, antisym=0).items():
        ap.add_argument('--' + k, type=type(v), default=v)
    args = ap.parse_args(); torch.set_num_threads(2)
    t0 = time.time(); res = run(args); res['sec'] = time.time() - t0
    print(json.dumps(res))
```

(Its walks generator is used by toy2.py; its own 3000-step batch is summarised in 1.9.)

### toy2.py

```python
"""Minimal 1-layer path-integrated attention on a D-torus (CPU). No FFN. Map redrawn per sequence, interleaved
[a1,o1,a2,o2,...], loss on revisited observations (predicted at the preceding action token), per-head rank r step map
Delta_h = omega_h * W_out^h W_in^h e(token). Arms: plain; --antisym 1 (action steps exactly odd, observation steps 0:
no per-move common component can exist; Lemma 1's obstruction removed by construction)."""
import sys, math, time, json, argparse, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, '/tmp/claude-1002/-home-prashr-mapformer/224d7c23-b570-4513-8742-cf8a41acc4a3/scratchpad')
from toy import walks

class Toy(nn.Module):
    def __init__(s, V, D, d, H, r, N, antisym):
        super().__init__(); s.H, s.dh, s.nb, s.r, s.D, s.antisym = H, d // H, d // H // 2, r, D, antisym
        s.emb = nn.Embedding(V, d); s.win = nn.Linear(d, H * r, bias=False)
        s.wout = nn.Parameter(torch.empty(H, s.nb, r).uniform_(-1 / math.sqrt(r), 1 / math.sqrt(r)))
        om = torch.tensor([2 * math.pi * (1 / N) ** (i / max(s.nb - 1, 1)) for i in range(s.nb)])
        s.omega = nn.Parameter(om.repeat(H, 1)); s.ln1 = nn.LayerNorm(d)
        s.q = nn.Linear(d, d); s.k = nn.Linear(d, d); s.v = nn.Linear(d, d); s.o = nn.Linear(d, d)
        s.lno = nn.LayerNorm(d); s.out = nn.Linear(d, V)
        partner = torch.arange(V)
        for i in range(D): partner[2*i] = 2*i+1; partner[2*i+1] = 2*i
        s.register_buffer('partner', partner); s.register_buffer('isact', (torch.arange(V) < 2 * D).float())
    def steps(s, tok):
        z = s.win(s.emb(tok)).view(*tok.shape, s.H, s.r)
        if s.antisym:
            zp = s.win(s.emb(s.partner[tok])).view(*tok.shape, s.H, s.r)
            z = (z - zp) * 0.5 * s.isact[tok][..., None, None]
        return torch.einsum('bthr,hnr->bthn', z, s.wout) * s.omega          # B,T,H,nb (radians)
    def forward(s, tok):
        B, T = tok.shape; x = s.emb(tok); th = torch.cumsum(s.steps(tok), 1).transpose(1, 2)   # B,H,T,nb
        c, sn = torch.cos(th), torch.sin(th); h = s.ln1(x)
        def rot(z):
            z = z.view(B, T, s.H, s.dh).transpose(1, 2); a, b = z[..., 0::2], z[..., 1::2]
            return torch.cat([a * c - b * sn, a * sn + b * c], -1)
        Q, K = rot(s.q(h)), rot(s.k(h)); Vv = s.v(h).view(B, T, s.H, s.dh).transpose(1, 2)
        sc = (Q @ K.transpose(-1, -2)) / math.sqrt(s.dh)
        sc = sc.masked_fill(torch.triu(torch.ones(T, T, dtype=torch.bool), 1), float('-inf'))
        o = (sc.softmax(-1) @ Vv).transpose(1, 2).reshape(B, T, -1)
        return s.out(s.lno(x + s.o(o)))

def geometry(m, D, N):
    """per head: cond of the phase frame, clock ratio (common per-move phase / frame), winding residual (Sec 1 probe)."""
    with torch.no_grad():
        V = m.emb.num_embeddings; P = m.steps(torch.arange(V)[None])[0].numpy()     # V,H,nb
    out = []
    for h in range(m.H):
        K = np.stack([(P[2*j, h] - P[2*j+1, h]) / 2 for j in range(D)], 1)
        tau = P[:2*D, h].mean(0) + P[2*D:, h].mean(0)
        sv = np.linalg.svd(K, compute_uv=False)
        res = np.abs(N * K / (2 * np.pi) - np.round(N * K / (2 * np.pi))).mean()
        out.append(dict(cond=float(sv[-1] / sv[0]), clk=float(np.linalg.norm(tau) / np.linalg.norm(K)), wind=float(res)))
    return out

def run(a):
    torch.manual_seed(a.seed); rng = np.random.default_rng(1000 + a.seed); D, N, T = a.D, a.N, a.T
    m = Toy(2 * D + 17, D, a.d, a.H, a.r, N, a.antisym)
    opt = torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=0.05)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.steps, pct_start=0.05)
    hist = []; snaps = []
    for step in range(a.steps):
        tok, rev, _ = walks(rng, a.B, T, D, N)
        lg = m(tok)[:, 0::2]; loss = F.cross_entropy(lg[rev], tok[:, 1::2][rev])
        opt.zero_grad(); loss.backward(); opt.step(); sch.step(); hist.append(loss.item())
        if (step + 1) % (a.steps // 5) == 0: snaps.append(geometry(m, D, N))
    m.eval(); erng = np.random.default_rng(99)
    with torch.no_grad():
        tok, rev, wrap = walks(erng, 64, T, D, N); ok = m(tok)[:, 0::2].argmax(-1) == tok[:, 1::2]
    return dict(seed=a.seed, r=a.r, H=a.H, antisym=a.antisym, D=D, N=N, T=T, steps=a.steps,
                final_loss=float(np.mean(hist[-a.steps // 20:])), acc=float(ok[rev].float().mean()),
                acc_wrap=float(ok[rev & wrap].float().mean()), acc_other=float(ok[rev & ~wrap].float().mean()),
                geom=snaps, curve=[float(np.mean(hist[i:i + 250])) for i in range(0, a.steps, 250)])

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    for k, v in dict(D=2, N=16, T=192, r=2, H=2, d=32, B=16, steps=10000, lr=3e-3, seed=0, antisym=0).items():
        ap.add_argument('--' + k, type=type(v), default=v)
    a = ap.parse_args(); torch.set_num_threads(4); t0 = time.time(); res = run(a); res['sec'] = time.time() - t0
    print(json.dumps(res), flush=True)
```

Output (32 runs, final geometry is UNWEIGHTED by channel amplitude, so less informative than the checkpoint probe):
```
r2 0 loss 0.121 acc 0.951 wrap 0.899 other 0.985 sec 544 [cond 0.09 clk 0.30 wind 0.10] [cond 0.03 clk 0.05 wind 0.11]
r2 1 loss 0.629 acc 0.834 wrap 0.681 other 0.934 sec 556 [cond 0.02 clk 0.51 wind 0.12] [cond 0.03 clk 0.70 wind 0.14]
r2 2 loss 0.037 acc 0.986 wrap 0.974 other 0.994 sec 569 [cond 0.08 clk 0.52 wind 0.15] [cond 0.17 clk 0.37 wind 0.16]
r2 3 loss 0.000 acc 1.000 wrap 1.000 other 1.000 sec 548 [cond 0.75 clk 0.00 wind 0.01] [cond 0.00 clk 0.03 wind 0.21]
r2 4 loss 0.001 acc 1.000 wrap 0.999 other 1.000 sec 526 [cond 0.00 clk 0.04 wind 0.22] [cond 0.16 clk 0.00 wind 0.06]
r2 5 loss 0.339 acc 0.886 wrap 0.771 other 0.961 sec 521 [cond 0.17 clk 3.07 wind 0.05] [cond 0.11 clk 0.62 wind 0.08]
r2 6 loss 0.580 acc 0.855 wrap 0.654 other 0.988 sec 466 [cond 0.12 clk 0.72 wind 0.17] [cond 0.00 clk 0.02 wind 0.16]
r2 7 loss 0.090 acc 0.964 wrap 0.964 other 0.964 sec 505 [cond 0.40 clk 1.25 wind 0.10] [cond 0.00 clk 0.21 wind 0.01]
r2 odd-steps 0 loss 0.670 acc 0.802 wrap 0.637 other 0.910 sec 560 [cond 0.16 clk 0.00 wind 0.19] [cond 0.30 clk 0.00 wind 0.27]
r2 odd-steps 1 loss 0.747 acc 0.779 wrap 0.554 other 0.927 sec 585 [cond 0.06 clk 0.00 wind 0.27] [cond 0.11 clk 0.00 wind 0.20]
r2 odd-steps 2 loss 0.843 acc 0.782 wrap 0.505 other 0.964 sec 589 [cond 0.45 clk 0.00 wind 0.23] [cond 0.08 clk 0.00 wind 0.29]
r2 odd-steps 3 loss 0.508 acc 0.876 wrap 0.691 other 0.999 sec 586 [cond 0.24 clk 0.00 wind 0.25] [cond 0.26 clk 0.00 wind 0.22]
r2 odd-steps 4 loss 0.853 acc 0.748 wrap 0.594 other 0.849 sec 555 [cond 0.29 clk 0.00 wind 0.25] [cond 0.02 clk 0.00 wind 0.17]
r2 odd-steps 5 loss 0.000 acc 1.000 wrap 1.000 other 1.000 sec 540 [cond 0.10 clk 0.00 wind 0.04] [cond 0.18 clk 0.00 wind 0.05]
r2 odd-steps 6 loss 1.019 acc 0.705 wrap 0.536 other 0.817 sec 546 [cond 0.65 clk 0.00 wind 0.23] [cond 0.17 clk 0.00 wind 0.19]
r2 odd-steps 7 loss 0.365 acc 0.891 wrap 0.850 other 0.919 sec 528 [cond 0.38 clk 0.00 wind 0.19] [cond 0.43 clk 0.00 wind 0.13]
r3 0 loss 0.001 acc 1.000 wrap 1.000 other 1.000 sec 557 [cond 0.44 clk 0.11 wind 0.18] [cond 0.66 clk 0.01 wind 0.07]
r3 1 loss 0.031 acc 0.982 wrap 0.967 other 0.992 sec 557 [cond 0.27 clk 0.01 wind 0.02] [cond 0.06 clk 0.66 wind 0.13]
r3 2 loss 0.023 acc 0.987 wrap 0.977 other 0.994 sec 535 [cond 0.07 clk 0.00 wind 0.13] [cond 0.28 clk 2.32 wind 0.24]
r3 3 loss 0.002 acc 1.000 wrap 0.999 other 1.000 sec 530 [cond 0.30 clk 0.00 wind 0.04] [cond 0.15 clk 0.07 wind 0.12]
r3 4 loss 0.019 acc 0.989 wrap 0.978 other 0.996 sec 511 [cond 0.02 clk 0.29 wind 0.10] [cond 0.13 clk 0.10 wind 0.13]
r3 5 loss 0.150 acc 0.950 wrap 0.938 other 0.958 sec 555 [cond 0.18 clk 0.38 wind 0.11] [cond 0.30 clk 0.10 wind 0.11]
r3 6 loss 0.847 acc 0.752 wrap 0.596 other 0.856 sec 515 [cond 0.13 clk 0.26 wind 0.14] [cond 0.04 clk 0.17 wind 0.13]
r3 7 loss 0.567 acc 0.846 wrap 0.666 other 0.965 sec 501 [cond 0.35 clk 0.10 wind 0.13] [cond 0.19 clk 0.02 wind 0.16]
r4 0 loss 0.001 acc 1.000 wrap 1.000 other 1.000 sec 573 [cond 0.02 clk 0.58 wind 0.18] [cond 0.25 clk 0.28 wind 0.18]
r4 1 loss 0.429 acc 0.874 wrap 0.744 other 0.959 sec 584 [cond 0.06 clk 0.04 wind 0.14] [cond 0.21 clk 0.76 wind 0.17]
r4 2 loss 0.574 acc 0.810 wrap 0.646 other 0.918 sec 548 [cond 0.02 clk 0.40 wind 0.02] [cond 0.36 clk 2.72 wind 0.12]
r4 3 loss 0.061 acc 0.981 wrap 0.955 other 0.999 sec 537 [cond 0.08 clk 0.34 wind 0.14] [cond 0.38 clk 0.33 wind 0.16]
r4 4 loss 0.001 acc 1.000 wrap 0.999 other 1.000 sec 548 [cond 0.62 clk 0.00 wind 0.04] [cond 0.38 clk 0.08 wind 0.05]
r4 5 loss 0.243 acc 0.903 wrap 0.860 other 0.932 sec 516 [cond 0.45 clk 0.14 wind 0.17] [cond 0.10 clk 0.10 wind 0.09]
r4 6 loss 0.084 acc 0.959 wrap 0.936 other 0.975 sec 523 [cond 0.54 clk 0.02 wind 0.15] [cond 0.15 clk 0.02 wind 0.13]
r4 7 loss 0.046 acc 0.968 wrap 0.966 other 0.969 sec 498 [cond 0.62 clk 0.03 wind 0.10] [cond 0.30 clk 0.01 wind 0.07]

r2             SOLVED 3/8  acc 0.934  wrap 0.868  other 0.978
r2 odd-steps   SOLVED 1/8  acc 0.823  wrap 0.671  other 0.923
r3             SOLVED 5/8  acc 0.938  wrap 0.890  other 0.970
r4             SOLVED 3/8  acc 0.937  wrap 0.888  other 0.969
```
