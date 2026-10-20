# Why per-head rank decides whether the map is learned: hiding the clock, and parking it

Expansion note for `docs/paper/MAIN_DRAFT1.md` Sec. 6.1 (2026-10-20). Labels: **ESTABLISHED** (registered results, files
named), **ANALYSIS** (computed here on stored checkpoints or CPU pilots, script named; post hoc unless said), **THEORY**
(statement + assumptions + the check that could break it), **SPECULATION**. Scripts and outputs:
`docs/paper/expansion/scripts/rank/`. Nothing here is registered; nothing was committed; no GPU was used.

## Summary

**The account that survives (THEORY, end-state checks pass, dynamics partly checked): rank above D gives every channel a
map-free way to cancel the per-move clock; at rank D there is none.** Every move adds a step common to all moves (a
"clock" c) to the latent. A channel can stop seeing it in two ways only (Prop. 1): through the latent (shrink c: one
global coordinate, removed at a rate proportional to the head's own resolution along the drift direction) or through its
own readout. At per-head rank r > D the readout can turn away from the part of c that lies outside the map plane
**without touching the map** ("parking"). At r = D the clock lies inside the map plane, c = U beta, and a readout that
stops seeing it must stop resolving the drift direction ("hiding"), which in turn slows the global route (Props. 2-3).
At D = 1 hiding kills the channel and leaves the drift-to-map ratio unchanged, so only the global route acts: rank 1
learns the ring (7/8).

Strongest evidence (ANALYSIS, post hoc; 1-3 on the stored checkpoints of registered batches, 4 a CPU pilot):
1. **Solved rank > D heads park the clock.** The off-plane share |c_perp| / |c| of their leftover latent clock has medians
   0.96-1.00 in nine sets of solved rank > D heads (0.81 in the tenth, 3D grid 18); a random direction gives 0.50-0.71; in those nine sets the readout sees it at 0.09-0.27 of the map's gain.
   Unsolved rank > D heads hold it in the plane and read it (torus r3 off-plane 0.27, lifted r4 0.08; readgain
   0.61-0.78). In solved rank-D heads the clock is simply gone (lam <= 0.001). `summarise_final_out.txt`.
2. **Escape test, re-read.** A fresh rank-4 step map on a stalled network removes the drift in 16/16 heads, from initial
   drift up to 5.9 cells per move, every one by parking (off-plane 0.91-1.00); rank 2 continued removes it in 2/16 heads.
   The lifted rank-4 runs start with an off-plane share of only 0.01-0.14, and the three that solve are exactly the three with
   the largest initial off-plane share (0.09-0.14 vs 0.01-0.06; exact rank-sum p 0.018, n 8). `escape_start_out.txt`.
3. **Rank-D drift sits on the least-resolved axis**: 57/62 rank-D runs with a drifting head have it at r_beta < 0.5
   against 28.7 expected for a random drift direction (P < 5e-5); median r_beta 0.087 over 116 drifting heads.
   `c1_runlevel_out.txt`. (Also predicted by "gradient pruning after collapse": not discriminating on its own.)
4. Pilot dynamics (T = 128, CPU, n = 3 per arm; `pilot_t128/`): rank 4 parks (off-plane share 0.55-0.81 -> ~1.0 while
   the visible drift drops 10x); rank 2 removes c through the latent in lockstep with the visible drift, or not at all.
   The one rank-2 seed that failed started with a near-collinear frame (indep 0.03) and kept its drift on the
   unresolved axis (r_beta 0.00, loss 0.23); **the same initial weights plus a per-channel clock slot (rank 2 map +
   a fixed extra latent coordinate) solved (loss 0.006, frame indep 0.44)**. n = 1 matched pair: suggestive only.

Not supported or not tested: the positive feedback in which hiding *creates* the weak axis (P2) was not seen -- the
pilot failure's weak axis was there at initialisation, and the readout gradient at initialisation shows no hiding
direction (drift there is ~1 cell per move, outside the small-drift regime; `grad_hiding_out.txt`, NEGATIVE).
Initial frame conditioning does not predict rank-D outcomes (p 0.19, as `03_formal.md` 1.5); initial drift weakly does
(pooled p 0.012 with shared seeds; independent score_rank seeds p 0.073). The measure-zero account (b) fails the 1D test.
MapFormer's 5D failure did not reproduce under our recipe at T = 16 (all ranks 0.999-1.000; RoPE 0.95), so it says
nothing here. **Decisive test (GPU, ~8 h): rank 2 + clock slot on the T = 1024 torus** (Sec. 6, G1): the account
predicts rank-3/4 behaviour; an over-parameterisation account predicts rank-2 behaviour.

---

## 0. What a theory has to explain (ESTABLISHED; files are authoritative)

| fact | numbers | file |
|---|---|---|
| per-head rank D = 2 fails on the T = 1024 torus | shared r2 0/8, per-head r2 2/8 (fresh seeds 1/6), block-diagonal r4 2/8; score_rank shared r2 3/12 | `RANK_SEP_RESULTS.md`, `RANK_MI_RESULTS.md`, `SCORE_RANK_RESULTS.md` |
| rank D + 1 and D + 2 succeed | per-head r3 6/8, r4 8/8 (shared or per head) | `RANK3_RESULTS.md`, `RANK_SEP_RESULTS.md` |
| it is per-head rank, not latent size | D - C_bd 8/8 vs 2/8 (Fisher p 0.0070); B and C both 4 latent dims, opposite sides | `RANK_SEP_RESULTS.md` |
| search, not capacity | rank-2 projection of solved r4: 0.9955 frozen, held under training 7/8 | `RANK_PROJ_RESULTS.md` |
| budget does not fix it | 1800 epochs: r2 0/8 vs r4 7/8 (5/8 r2 still descending) | `LOOP_RANK_E1800_P1_RESULTS.md` |
| rank = D is hard in 2D and 3D, not in 1D | 2D G32 r2 1/8 vs r3 8/8; 3D G10 r3 1/8, r4 4/8; 3D G18 r4 8/8; ring rank 1 7/8 vs rank 2 8/8 | `RANK_ND_RESULTS.md`, `RANK_WRAP_RESULTS.md`, `RANK_TESTS_RESULTS.md` |
| failing runs end with a drifting head | "SOLVED iff a clean head" 39/40 (post hoc thresholds, close to circular: CORRECTED block) | `SCORE_RANK_RESULTS.md` |
| forbidding the drift speeds rank 1 on the ring | R1DF 8/8, median 24 epochs vs R1 58; R1AO (observation steps 0) 52 | `RANK_TESTS_RESULTS.md` |
| escape | stalled r2 lifted to r4: 3/8 in 600 ep vs r2 continued 1/8, fresh r4 step map 2/8 (unmeasured); unsolved lifted runs keep a CLOCK head (5/5), fresh step maps end CLEAN (8/8) | `RANK_TESTS_RESULTS.md` |
| failure signature | rank 2 keeps short-gap revisits, loses long-gap (0.671) and wrap-only (0.557) | `MAIN_DRAFT1.md` Sec. 6.3, `RANK_MATCHED_RESULTS.md` |
| MapFormer's own Table 6 | inner rank = D: 3D (T 256, grid 16) MapWM 1.00 IID; 5D (T 16, grid 5) 0.75, "on par with RoPE" | `papers/txt/mapformer.txt` l. 2385-2455 |

Withdrawn accounts that are NOT revived here: "r = 2 loses because its basis is skewed" (skew is a symptom), "the r = 4
gap is training speed", packing geometry (`DXR_RANK_THRESHOLD.md`). The end-state lemma of `docs/theory/2026-10-04/03_formal.md`
(Theorem 1, "T1") is used and extended; its causal reading was explicitly left open there (toy negative, 1.9).

---

## 1. Setup and notation (exact for a one-layer MapWM)

Per head h, every token x adds a latent step u(x) = W_in^h e(x) in R^r (r = per-head rank; for a shared latent, the
head's read subspace, i.e. the row space of W_out^h). Channel i (i = 1..n_b) has readout row f_i (a row of
diag(omega_h) W_out^h) and phase phi_i(t) = <f_i, z_t>, z_t = sum_{s <= t} u(x_s) (`model.py:83,120`). The attention
score between a query at t and a key at s is sum_i A_i cos(phi_i(t) - phi_i(s) + psi_i), A_i the content amplitudes.

On a D-torus with actions +-e_j and strictly alternating action / observation tokens, one move adds
    Delta z = U a + c + eta(o),   U_j = (u(+e_j) - u(-e_j)) / 2,   c = mean_a u(a) + E_o u(o),
U in R^{r x D} the **map frame**, c the **per-move clock** (common to every move), eta the zero-mean observation leak
(ignored here; `LEAK_RESULTS.md`). Channel i reads a phase per unit displacement k_i = U^T f_i in R^D and a **drift**
tau_i = <f_i, c> per move. A revisit at displacement 0 after g moves carries a phase error g tau_i in channel i.

- At r = D with U invertible, c = U beta for a unique **phantom displacement** beta = U^{-1} c (cells per move), and
  tau_i = <k_i, beta> in every channel (T1). Write P = sum_i A_i k_i k_i^T (the head's positional precision, D x D)
  and r_beta = (bhat^T P bhat) / (tr P / D): the head's resolution along its own drift direction, relative to its mean
  (1 = isotropic or unaligned).
- At r > D, split c = c_par + c_perp with c_par in range(U) and c_perp orthogonal to it (latent metric); c_perp is the
  part of the clock **outside the map plane**.

Readouts used below (`rank_geom.py`): kappa (visible drift per move in units of one move, wrap-aware, channel-weighted
as `analyze_score_rank.head_stats`: verified identical on the stored checkpoints), indep (frame conditioning), beta,
r_beta, lam = |c| / |U| in the native latent (the clock whether read or not), off = |c_perp| / |c|, and readgain (how
strongly the readout sees the leftover clock relative to the map; ~1 = read like the map, ~0 = invisible).

---

## 2. Theory

### 2.1 Proposition 1 (two routes to a drift-free channel; THEORY, proved -- linear algebra)
A change of parameters lowers channel i's drift tau_i = <f_i, c> only by
(G) the **global route**: changing c (shared by every channel of the head; for a shared latent, by every head), or
(L) the **local route**: changing the channel's readout f_i by delta with <delta, c> = -tau_i.
The change L makes to the channel's map is delta k_i = U^T delta. Among all delta that cancel tau_i:
(a) if c_perp != 0 (possible only at r > D), delta = -tau_i c_perp / |c_perp|^2 cancels it with **U^T delta = 0 exactly**:
    the map is untouched to all orders ("parking": the clock stays in the latent, off the map plane, and the readout
    stops seeing it);
(b) if c in range(U) (always at r = D), <delta, c> = <U^T delta, beta>, so every cancelling delta changes k_i, by at
    least |tau_i| / |beta|, and the minimum-norm one sets k_i <- k_i - (k_i . bhat) bhat: **it removes the channel's
    resolution along the drift direction** ("hiding": the channel stops seeing the drift by no longer resolving bhat).
(c) at D = 1, bhat-perp = {0}: hiding sets k_i = 0 (the channel dies). More strongly, in 1D the drift a channel sees,
    measured in cells, is tau_i / k_i = beta for every channel and every readout: **no readout change alters the
    drift-to-map ratio**. Only the global route (shrinking c, or growing U) reduces the phantom displacement.
Proof: (a) take delta along c_perp; U^T c_perp = 0 by definition. (b) c = U beta gives <delta, c> = <U^T delta, beta>;
Cauchy-Schwarz; the minimiser is delta k_i parallel to beta. (c) D = 1: k_i = U f_i, tau_i = c f_i, ratio c / U. QED.
Per head, because each head's channels read only that head's r-dim image (T1's per-head argument): a block-diagonal
rank-4 latent gives each head rank 2 and no c_perp, which is exactly the RANK_SEP split.

### 2.2 Proposition 2 (at rank D >= 2 hiding is free to first order; at D = 1 it is not; THEORY)
Assumptions (A1) small drift: for the gaps that carry gradient, g |tau_i| << 1, so the drift loss is
L_d = (1/2) sum_i a_i tau_i^2 with a_i = gamma A_i E[g^2; g < g_c] > 0 (gamma: curvature of the retrieval loss in the
score); (A2) the map loss depends on the frame only through the spectrum of P, L_m = Phi(P) = sum_j f(lambda_j(P)) with
f decreasing and convex (more resolution helps, with diminishing returns), plus a norm cost on readout rows (weight
decay); (A3) U near-isometric on the readout's span, so rotating k_i preserves |f_i|.
Claim. (i) D >= 2, P isotropic (P = lambda I): rotating any channel's k_i by a small angle eps from bhat towards a
direction v orthogonal to bhat changes L_m by 0 at first order (dPhi = f'(lambda) tr(dP) and a rotation leaves
tr P unchanged), and changes L_d by -eps a_i |k_i|^2 |beta|^2 sin(2 phi_i) / 2 < 0 for every channel not already
aligned with or orthogonal to bhat. So the gradient flow on the readout moves resolution **off the drift axis** at once,
for any beta != 0; the equilibrium anisotropy solves f'(lambda_perp) - f'(lambda_beta) = (a/2) |beta|^2 (D = 2, a_i = a A_i), growing
with the drift. (ii) D = 1: the only readout move that lowers L_d is shrinking |k_i|, whose first-order map cost is
-f'(lambda) A_i > 0; it is taken only above the threshold a_i |beta|^2 > -2 f'(lambda) A_i, and even then it does not
change beta (Prop. 1c). (iii) r >= D + 1 with c_perp != 0: parking (Prop. 1a) lowers L_d at rate
a_i |c_perp|^2 per channel with zero change in L_m at all orders.
Derivation: (i) k_i = |k_i|(cos phi bhat + sin phi v); tau_i = |k_i||beta| cos phi; d/deps of (a_i/2) tau_i^2 =
-a_i |k_i|^2 |beta|^2 cos phi sin phi. (ii)-(iii) as stated. The step is a statement about the gradient, so it also holds
for sign-based (Adam) steps when the gradient component is consistent across steps.

### 2.3 Proposition 3 (hiding starves the global route: a positive feedback; THEORY)
Under (A1), the gradient of L_d with respect to the phantom displacement is P_a beta (P_a = sum_i a_i k_i k_i^T), so the
global route removes the drift along bhat at a rate proportional to the head's own (drift-weighted) resolution along
bhat, bhat^T P_a bhat. Hiding (Prop. 2) lowers exactly that quantity. Hence at rank D >= 2:
drift -> hiding (resolution leaves bhat) -> slower global removal -> drift persists -> more hiding.
The clean state (beta = 0, P isotropic) remains a stable fixed point (it is the projection that RANK_PROJ installs and
training holds), but a run that starts with, or develops, a drift reaches it only after a plateau. In a reduced model
(f(lambda) = k0 / lambda; quasi-static hiding) the hidden equilibrium is lambda_beta ~ sqrt(2 k0 / a) / |beta| and the
drift then shrinks only linearly, d|beta|/dt = -eta_G sqrt(2 k0 a): a plateau whose length grows with the initial drift
and shrinks with the latent learning rate -- slow, not absorbing (SPECULATION as to the exact law; the qualitative
statement follows from Props. 1-2).
Two refinements. (R1) Long gaps carry the loss but not the gradient: once g |tau_i| > 1 the cosine term averages out
(Riemann-Lebesgue over the gap distribution), so drift removal is driven by short gaps while the failure shows on long
gaps and wrap-only revisits (which are long-gap by construction, T5 of `03_formal.md`). At T = 128 the hidden drift is
nearly harmless: rank-2 heads drift or collapse on 32/32 while fitting the training task (`03_formal.md` 1.4); at
T = 1024 it is the failure. (R2) With hiding the head stays near the "COLLAPSE" corner (T1's class B) when hiding wins
completely, and in the "CLOCK" class (A) when it is partial: both are points on one curve, not two mechanisms.

### 2.4 Corollaries and predictions
P1 (rank D >= 2). Drifting heads have their drift along their least-resolved direction: r_beta << 1.
P2 (rank D >= 2, dynamics). The anisotropy builds up along an early, persistent drift direction (rather than the drift
    turning onto an axis that was already weak).
P3 (rank >= D + 1, end state). In solved heads the leftover latent clock lies outside the map plane (off -> 1, above
    the random-direction value sqrt((r - D) / r)) and the readout is nearly blind to it (readgain << 1): parked, not
    removed. Its size then decays only through weight decay, so it should shrink with training length.
P4 (rank >= D + 1, dynamics). Parking precedes removal: the visible drift kappa falls well before the latent clock lam.
P5 (rank >= D + 1 failures). Unsolved D + 1 heads have their clock in the map plane (off small) and read (readgain ~1):
    the free route existed in principle and was not open.
P6 (1D). Rank D = 1 is not trapped (no direction to hide along); its failures, if any, show a large drift that the
    global route is still removing, and forbidding the drift only speeds it up.
P7 (clock slot). Rank D plus a per-channel learned clock rate (a fixed extra latent coordinate that every token steps by
    1; the map stays rank D) opens Prop. 1(a) without adding any map freedom. Prediction: it behaves like rank D + 1
    (drift-free, well-conditioned frames), and on the T = 1024 torus solves like rank 3-4. An over-parameterisation
    (Burer-Monteiro) account, in which the extra rank helps by adding factorisation freedom to the map, predicts no
    rescue: this arm discriminates.
P8 (escape). A stalled rank-2 model lifted to rank 4 can park only if its clock acquires an off-plane part; the lift
    adds new W_in rows at the default scale beside large trained rows, so the clock starts almost entirely in-plane.
    Lifted runs should solve in proportion to their initial off-plane share.
P9 (D + k). P(clean) should rise with the expected off-plane share sqrt((r - D) / r) of a random clock: 0.71 (r 4, D 2)
    > 0.58 (r 3, D 2) > 0.50 (r 4, D 3); rank 5 in 3D should beat rank 4 on the wrap-heavy grid.

### 2.5 The alternatives (and where each stands)
- (T1 alone, end-state lemma) "a rank-D head is clean iff c = 0": correct but static; it does not say why training
  leaves c != 0 at rank D and removes it at D = 1. Wrap note: exactly, a rank-D head is drift-free iff
  beta in N Z^D (a whole number of laps per move), which is irrelevant in practice.
- (b) measure-zero / symmetry: the clean set at rank D is {c = 0} (codimension r in c), at rank D + 1 it also contains
  {F orthogonal to c, c outside range U}, which is reachable through W_out alone. This is the static shadow of
  Prop. 1(a). On its own it fails the 1D test: at rank 1 the clean set is also {c = 0}, yet rank 1 solves 7/8.
- Burer-Monteiro / over-parameterisation (red team, `00_PLAN.md`): extra rank smooths the landscape of the map
  factorisation. Distinguished by P3 (what the extra direction holds) and decisively by P7 (clock slot).
- "Rank D is hard because of the exactly periodic code" (SCORE_RANK secondary, CORRECTED as not separated from long
  gaps): in this theory wrap-only revisits fail because they are long-gap revisits read through a hidden drift (R1);
  RANK_NOWRAP would separate them.

---

## 3. Checks

All scripts: `docs/paper/expansion/scripts/rank/`, run from `/home/prashr` with `PYTHONPATH=/home/prashr`. Geometry
readout `rank_geom.py` (verified: kappa and indep equal `analyze_score_rank.head_stats` to 4 decimals on rank_mi
Vanilla s0, r4mi s0, r2ph s1, r4mibd s2). Final-checkpoint sweep `sweep_final.py` -> `sweep_final_out.txt` /
`sweep_final.json` (25 sets, every stored matched-length rank checkpoint plus the escape arms and the T = 128
rank_sweep); summaries `summarise_final.py` -> `summarise_final_out.txt`. Note: `runs/rank_wrap/N10/D3` and
`runs/rank_nd/D3` r4 (and N32/D2 vs rank_nd D2 r3) are bit-identical reruns (`RANK_WRAP_RESULTS.md`), counted once.
SOLVED is the registered rule (final-5% training loss < 0.05, `stats_core.classify_run`).

### C1. P1: drifting rank-D heads drift along their least-resolved axis -- PASS (ANALYSIS; not discriminating)
| set (rank D, D >= 2) | drifting heads | median r_beta | share < 0.5 |
|---|---|---|---|
| torus G64 shared r2 (rank_mi) | 16 | 0.060 | 0.88 |
| torus G64 per-head r2 | 9 | 0.194 | 0.89 |
| torus G64 block-diagonal r4 | 8 | 0.106 | 0.75 |
| torus G64 shared r2 (score_rank) | 20 | 0.195 | 0.80 |
| torus G64 shared r2, 1800 ep | 12 | 0.214 | 0.67 |
| ND D2 G32 r2 | 15 | 0.005 | 1.00 |
| ND D3 G10 r3 | 14 | 0.088 | 0.86 |
| escape C (r2 continued) | 14 | 0.087 | 1.00 |
| T = 128 rank_sweep r2 | 8 | 0.096 | 0.75 |
Pooled 116 heads: median 0.087, 85% below 0.5 (random direction given each head's own frame: 29%). Run level
(`c1_runlevel.py`): 57/62 runs vs 28.7 expected, P < 5e-5 (20,000 simulations). Clean heads of rank > D sit at
r_beta ~1 (0.93-1.05 medians). Caveat: this is what Prop. 3 predicts, but also what pruning of the visible drift after
a collapse of any cause predicts (`03_formal.md` 1.3 CAUTION); it fixes the geometry, not the order of events.

### C2. P3: solved rank > D heads park the clock -- PASS (ANALYSIS)
| set (solved heads) | r - D | leftover lam = abs(c)/abs(U) | off-plane share (random) | readgain |
|---|---|---|---|---|
| torus G64 per-head r3 | 1 | 0.004 | 0.978 (0.58) | 0.15 |
| torus G64 shared r4 (rank_mi) | 2 | 0.004 | 0.996 (0.71) | 0.11 |
| torus G64 per-head r4 | 2 | 0.003 | 0.988 (0.71) | 0.16 |
| torus G64 shared r4 (score_rank) | 2 | 0.003 | 0.998 (0.71) | 0.09 |
| ND D2 G32 r3 | 1 | 0.003 | 0.987 (0.58) | 0.09 |
| ND D3 G10 r4 | 1 | 0.008 | 0.964 (0.50) | 0.23 |
| ND D3 G18 r4 | 1 | 0.001 | 0.810 (0.50) | 0.57 |
| ring R2 (1D, rank 2), 300 ep | 1 | 0.025 | 0.989 (0.71) | 0.21 |
| T = 128 rank_sweep r4, 300 ep | 2 | 0.021 | 0.988 (0.71) | 0.27 |
| escape F (fresh r4 step map, 600 ep) | 2 | 0.003-0.004 | 0.98 (0.71) | 0.18 |
For comparison, solved rank-D heads: lam 0.000-0.001 (ring R1, torus r2 solved): there the clock is removed, because it
cannot be parked. The parked remainder is ~6x larger after 300 epochs than after 600-900 (0.021-0.025 vs 0.003-0.004),
of the order AdamW's decoupled decay alone gives an unread direction over a cosine schedule (factor exp(-wd * sum lr):
0.44 at 300 epochs vs 0.09 at 900, ratio ~5). Consistent, not a test (different tasks).

### C3. P5: unsolved rank > D heads hold the clock in the map plane and read it -- PARTIAL PASS (ANALYSIS)
Torus per-head r3, 4 unsolved heads: lam 1.50, off-plane share 0.27 (random 0.58), readgain 0.61. Escape E unsolved,
10 heads: off-plane 0.08 (random 0.71), readgain 0.76. ND D3 G10 r4, 8 unsolved heads: off-plane 0.53 (random 0.50),
readgain 0.78 -- here the clock is not especially in-plane, but it is read: the free route existed and was not taken
(as `03_formal.md` 1.3 found, |c_perp| / |c| 0.16 by its metric). So "in-plane" holds in 2 of 3 sets, "read" in 3 of 3.

### C4. P6: the 1D exemption -- consistent (ANALYSIS + ESTABLISHED)
In 1D r_beta = 1 identically and every channel sees the same phantom drift beta (Prop. 1c). Ring R1 (`RANK_TESTS`):
14 clean heads end at lam <= 0.001 (the clock removed, not parked); the 3 drifting heads (s20 h0, s22 h0, s26 h1) have
the largest map steps |U| (5.0, 2.8, 3.25 vs 1.6-2.6 for clean heads): the only other global way to shrink
beta = c / U. R1DF (drift forbidden) solves in 24 epochs vs 58 (ESTABLISHED), as Prop. 1c says the drift has to be
removed through the latent in 1D. Not a test of the mechanism: the ring task also differs (T = 128) -- see G5.

### C5. P8: the escape test re-read -- PASS (ANALYSIS, post hoc; `escape_start.py` -> `escape_start_out.txt`)
| arm (600 ep, same 8 trained networks) | runs with a drift-free head at the end | heads drift-free | off-plane share at end |
|---|---|---|---|
| F: fresh rank-4 step map (start kappa 0.28-5.86) | 8/8 | 16/16 | 0.91-1.00 (all) |
| E: stalled rank 2 lifted to rank 4 | 3/8 | 6/16 | 1.00 in the 3 solved; 0.01-0.31 in four unsolved runs, 0.66 in s4 (unsolved) |
| C: rank 2 continued | 1/8 | 2/16 | -- (rank 2) |
F vs E on runs: Fisher p 0.026. Within E, the full-latent off-plane share of the clock at the START (the lift adds two
W_in rows at default scale next to large trained rows): solved s1 0.09, s2 0.14, s3 0.11; unsolved s0 0.03, s4 0.06,
s5 0.01, s6 0.01, s7 0.06 -- the three largest are the three solved (exact one-sided rank-sum p 0.018). Confound: the
three solved sources were "descending" at 900 epochs (so was s4, unsolved). F's solves are only 2/8 (`RANK_TESTS`):
parking removes the drift in every F run; learning the rest of the map within 600 epochs is a separate matter.

### C6. P2, P4, P7 dynamics -- PARTIAL (ANALYSIS, CPU pilot; `train_snap.py`, `analyze_snap.py`, `pilot_t128/`)
Paper torus T = 128 (grid 64), batch 32 (a quarter of RANK_SWEEP's), 40 epochs cosine, lr 1e-3, `--sdpa`; seeds
3000-3002 (outside every batch); arms shared r2, shared r4 (r2's base), R2Clock (= shared r2 at init plus a per-channel
learned clock rate: a fixed extra latent coordinate every token steps by 1; `rank_geom.MapFormerWM_r2Clock`).
Snapshots every 49 steps.
| arm | seed | final loss | acc T=128 / 512 (eval mode) | heads: kappa, frame indep (end) | latent clock |
|---|---|---|---|---|---|
| r2 | 3000 | 0.005 | 1.000 / 0.937 | 0.004, 0.76 / 0.004, 0.67 | removed in lockstep with kappa (lam 0.63 -> 0.004) |
| r2 | 3001 | **0.230** | 0.922 / 0.836 | 0.043, **0.00** / 0.039, **0.00** | kept, on the unresolved axis (r_beta 0.00) |
| r2 | 3002 | 0.003 | 1.000 / 0.981 | 0.002, 0.68 / 0.001, 0.50 | removed in lockstep |
| r4 | 3000 | 0.005 | 1.000 / 0.954 | 0.003, 0.38 / 0.004, 0.51 | parked: off-plane 0.62 -> 0.96 |
| r4 | 3001 | 0.028 | 0.990 / 0.959 | 0.004, 0.57 / 0.005, 0.91 | parked: 0.55 -> 0.98 |
| r4 | 3002 | 0.005 | 1.000 / 0.946 | 0.032, 0.48 / 0.037, 0.78 | parked: 0.81 -> 1.00, lam 0.09 vs kappa 0.03 |
| R2Clock | 3000 | 0.003 | 1.000 / 0.949 | 0.020, 0.71 / 0.018, 0.48 | in the slot (off-plane 1.00) |
| R2Clock | 3001 | **0.006** | **1.000** / 0.941 | 0.045, **0.44** / 0.049, **0.36** | in the slot |
| R2Clock | 3002 | 0.004 | 1.000 / 0.959 | 0.025, 0.58 / 0.023, 0.53 | in the slot |
- P4 (parking first): the visible drift fell 10x before the latent clock did in 4/6 rank-4 heads (the other 2 at the
  same snapshot) and 4/6 R2Clock heads; in rank 2 the two fall together (lam ~ kappa throughout). PASS (n = 3).
- P2 (feedback creates the weak axis): NOT SHOWN. The failing r2 seed 3001 started with a near-collinear frame
  (indep 0.03-0.04 at step 0) and its drift stayed on that axis (|cos(beta@200, beta_end)| 1.00); the anisotropy was
  inherited, not created. What it does show is Prop. 3's static half: drift on an unresolved axis is not removed.
- P7 (clock slot): at the SAME initial weights as the failing r2 seed, R2Clock solved (loss 0.006 vs 0.230, held-out
  1.000 vs 0.922 at T = 128) and repaired its frame (indep 0.03 -> 0.44 / 0.36). One matched pair: suggestive only.
  R2Clock does not drive its visible drift below 0.01 at T = 128 (0.02-0.05), where a drift of that size is harmless.
- Budget and batch differ from RANK_SWEEP (40 vs 300 epochs, batch 32 vs 128): a dynamics testbed, not a reproduction.

### C7. Initial conditions at T = 1024 -- MIXED (ANALYSIS; `init_frame.py` -> `init_frame_out.txt`)
Rebuilt initial weights (train_variant's seeding) for 36 rank-D runs and 8 rank-3 runs. Initial frame conditioning
does not separate solved from unsolved rank-D runs (best head indep: solved 0.17-0.84 vs unsolved median 0.35;
one-sided p 0.19), agreeing with `03_formal.md` 1.5. Initial drift (min over heads of kappa) is lower in the runs that
solve: solved 0.28-1.25 (5 of 8 below 0.5) vs unsolved median 1.11 (n 28), p 0.012 pooled (seeds 0-7 share bases across
sets: not independent); score_rank alone (independent seeds 10-21) 3 vs 9, p 0.073. At rank 3 there is no such
relation (solved runs start at 0.28-1.45). Consistent with Prop. 3 (the rank-D plateau grows with the initial drift;
rank D + 1 parks whatever the start); weak and post hoc.

### C8. Prop. 2 at initialisation (readout gradient direction) -- NEGATIVE / uninformative (`grad_hiding.py`)
On the real task (T = 1024, 2 batches of 16, seeds 0-3, shared r2 and r4), a small readout step along -grad or
-sign(grad) moves frame resolution toward or away from the drift axis with no consistent sign (hide index from -1.00 to
+1.00 in both ranks) and changes kappa by < 0.4%. At initialisation the drift is 0.4-2.7 cells per move, so g |tau| >> 1
for almost every gap: assumption A1 (small drift) does not hold and the drift gradient is incoherent (R1). Prop. 2 is a
statement about the small-drift phase (after the first ~200 steps in the pilot, where kappa has fallen to 0.01-0.2);
it was not tested there.

### C9. MapFormer's 5D failure under our recipe -- UNINFORMATIVE for rank (`train_nd_cpu.py`, `pilot_5d/`)
5D torus, grid 5, T = 16 (the paper's Table 6 setting; our directed walk and recipe, 100 epochs, batch 128, seeds
4000-4002): per-head rank 5 (= D), 6, and 5 + clock slot all SOLVED 3/3 (final loss 0.005-0.012; held-out 0.999-1.000
at T = 16, 0.996-0.998 at T = 32); index RoPE STALLED (0.95 / 0.84-0.88). The paper's 0.75 for MapWM does not reproduce
here (recipe and walk differ; the paper's 5D recipe is not stated). Every MapWM head in every arm ends drifting
(kappa 0.2-0.9) while solving: at T = 16 a drift is harmless (R1), and nothing removes it at any rank. So "SOLVED iff a
clean head" is a long-walk statement, and the 5D short-walk task cannot test this theory.


---

## 4. Which theory survives

| candidate | what it says | verdict |
|---|---|---|
| (a) clock / free-route account (Props. 1-3), as stated in the Summary | rank > D lets each channel cancel the clock off the map plane; at rank D the clock can only leave through the latent, at a rate set by the resolution along the drift | **SURVIVES** every end-state check (C1, C2, C3 partly, C5) and the pilot dynamics' parking (C6 P4); one matched clock-slot rescue (C6 P7). THEORY, not established: no registered test |
| (a') its feedback half (hiding creates the weak axis, Prop. 2) | the readout rotates resolution off the drift axis because that is free to first order | **NOT SHOWN**: C6's failure inherited its weak axis; C8 uninformative (drift too large at init). Keep as a derivation with its assumptions, not as a finding |
| T1 end-state lemma (`03_formal.md`) | clean at rank D iff c = 0 | true, kept as Lemma 0; explains nothing about training by itself |
| (b) measure-zero / symmetry of the rank-D solution set | the rank-D clean set is a thin target | **FAILS** the 1D test (rank 1's clean set is the same codimension, yet 7/8 solve) and does not predict C1, C2 or C5 |
| (c) 1D exemption by "a scalar clock separates by sign" | -- | not needed: Prop. 1(c) gives the exemption (no hiding direction; the readout cannot change the drift in cells) |
| Burer-Monteiro / over-parameterisation of the map factorisation | spare rank smooths the map's landscape | **NOT RULED OUT**, but C2 shows the spare direction ends up holding the clock, not the map (the map is rank D: RANK_PROJ's top-2 projection keeps 0.9955), and the n = 1 clock-slot pair favours (a). G1 decides |
| "the rank defect is the periodic (wrap) code" (SCORE_RANK secondary) | -- | subsumed as a consequence (R1: wrap-only revisits are long-gap revisits read through a drift); not separated -- RANK_NOWRAP would |
| "skew is the mechanism" (WITHDRAWN), "training speed" (WITHDRAWN), packing (WITHDRAWN) | -- | not revived. In this account skew (collinear axes, indep -> 0) is where a drift along an unresolved axis leaves a rank-D head: a symptom, as the withdrawal says |

**Status to carry into the paper:** the account is a THEORY with ANALYSIS support on stored checkpoints (post hoc) and a
CPU pilot (n = 3); the one causal prediction that would make it a finding (G1) is untested.

---

## 5. Proposed paper text

### 5.1 Addition to Sec. 6.1 (after "Rank against the map's dimension"), ~260 words

**Why the extra rank helps (theory and analysis).** Every move adds to the latent a step common to all moves, a clock c,
on top of the displacement U a. A channel stops seeing the clock either through the latent (c shrinks, one coordinate
shared by the head) or through its own readout. When the per-head rank exceeds D, part of c lies outside the plane the
map occupies, and the readout can turn away from that part without changing the map; at rank D, c = U beta lies inside
the map plane, so a channel can stop seeing it only by stopping to resolve the drift direction, and the latent removes
it at a rate proportional to the head's resolution along that direction (App. M, Props. 1-3). In 1D there is no
direction to turn into, so only the latent route exists, and rank 1 learns the ring (7/8). The stored checkpoints show
the predicted signatures (analysis, post hoc). Solved heads of rank above D keep a small leftover clock almost entirely
outside the map plane (off-plane share 0.96-1.00 in nine of ten sets, against 0.50-0.71 for a random direction) and
read it at a tenth to a quarter of the map's gain; failing heads hold it in the plane. Drifting rank-D heads drift along their
least-resolved axis (57 of 62 runs, against 29 expected). A fresh rank-4 step map on the stalled rank-2 networks
removes the drift in 16 of 16 heads, each time by parking it off the plane, and of the stalled runs lifted to rank 4 the
three that solve are the three whose clock started with the largest off-plane share (rank-sum p 0.018). Whether the
extra rank helps by giving the clock somewhere to go, rather than by adding freedom to the map, is decided by a rank-2
model with a fixed clock coordinate (App. M; untested at T = 1024).

### 5.2 Appendix M (new): the clock and the rank

M1 Setup (Sec. 1 of this note, verbatim). M2 Lemma 0 (T1, with the wrap remark). M3 Proposition 1 with proof.
M4 Proposition 2 with assumptions A1-A3 and the derivation; state plainly that it was not observed at initialisation
(C8) and that the pilot failure inherited its weak axis (C6). M5 Proposition 3 and refinements R1-R2; the reduced
plateau law marked as a cartoon. M6 Checks C1-C5 as tables (C1, C2, C5 tables above); C6 table with the caveat
"CPU pilot, n = 3, quarter batch, 40 epochs"; C7 and C9 as one paragraph each. M7 Predictions P1-P9 with which were
checked. Figure M1 (suggested, from `pilot_t128/*.json`): kappa and off-plane share against step for rank 2, rank 4 and
R2Clock (one panel each, three seeds), showing lockstep removal at rank 2 and parking at rank 4.

### 5.3 Wording constraints
- Say "parks the clock outside the map plane" only with the off-plane share and its random baseline beside it.
- Do not say "rank fails because of drift" (causal); say "failing rank-D heads drift along their least-resolved axis;
  rank above D gives the drift a map-free exit" until G1 is run.
- Keep the per-head caveat: our shared latent vs the paper's per-head latent (CLAUDE.md invariants).

---

## 6. GPU experiments that would decide it

Throughput from SCORE_RANK / RANK_TESTS: ~2,700 run-epochs per hour on the two 4090s at the T = 1024 recipe
(RANK_MI: batch 16, lr 1e-3, cosine, 900 epochs, 1 layer, 2 heads, d 128). Each would need a pre-registration, a pilot
on outside seeds and an independent code audit (rules 7, 29).

| # | experiment | arms x seeds x epochs | run-epochs | wall | what decides it |
|---|---|---|---|---|---|
| **G1** | **Clock slot at T = 1024 (decisive).** Shared r2 (`Vanilla`), r2 + clock slot (`MapFormerWM_r2Clock`: identical to Vanilla at init, +64 parameters), shared r4 (`Vanilla_r4mi`), fresh seeds; snapshots every 10 epochs (geometry only) | 3 x 8 x 900 | 21,600 | ~8 h | Primary SOLVED, slot vs r2 (Fisher). Account predicts slot >= 6/8 (like r3/r4); over-parameterisation predicts <= 2/8. Branches as RANK_TESTS. The snapshots test P2/P4 at full scale for free |
| G1+ | add r2 with drift forbidden (odd action steps, zero observation steps) and its ActOnly control (observation steps 0, actions free) | +2 x 8 x 900 | +14,400 | +5.3 h | separates the clock from the observation leak (the R1DF / R1AO logic of RANK_TESTS) |
| G2 | **Seeded escape.** The 8 stalled rank-2 RANK_MI runs lifted to rank 4 with new W_in rows set (still function-preserving, W_out new columns 0) so that the clock's off-plane share is ~0.7, vs the default lift (share 0.01-0.14) | 2 x 8 x 600 | 9,600 | ~3.6 h | account predicts seeded >= 6/8 vs default ~3/8 (C5) |
| G3 | rank D + 2 in 3D on the wrap-heavy grid (per-head r5, grid 10, vs r4 in the same batch) | 2 x 8 x 900 | 14,400 | ~5.3 h | P9: expected off-plane share 0.63 vs 0.50 predicts r5 > r4's 4/8; a flat result says the 3D-G10 shortfall is the grid (memorisation / lattice), not the clock |
| G4 | 1D at long walks: ring rank 1 vs rank 2 at T = 1024 (ring 32 cells) | 2 x 8 x 300 | 4,800 | ~2 h | Prop. 1(c) predicts rank 1 still learns the ring at long T (no hiding direction); if rank 1 then fails like rank 2 in 2D, the 1D exemption was the short task (RANK_TESTS caveat) and the account loses its D-dependence |

Minimum to settle the account: G1 (8 h). With G2 (3.6 h) the mechanism and the escape reading are both registered.

---

## 7. Reproduce (all CPU; `cd /home/prashr; export PYTHONPATH=/home/prashr`; D = `mapformer/docs/paper/expansion/scripts/rank`)
- `python3 $D/sweep_final.py` (~3 min) -> `sweep_final_out.txt`, `sweep_final.json`; `python3 $D/summarise_final.py`;
  `cd $D && python3 c1_runlevel.py` (C1-C4).
- `python3 $D/escape_start.py` (C5); `python3 $D/init_frame.py` (C7); `python3 $D/grad_hiding.py` (C8, ~10 min).
- C6: `python3 $D/train_snap.py --arm {Vanilla,Vanilla_r4mi,R2Clock} --seed {3000,3001,3002} --epochs 40 --batch-size 32
  --sdpa --threads 2 --out OUT` (~85 min each on a loaded 32-core host), then `python3 $D/analyze_snap.py OUT`;
  stored outputs `pilot_t128/`.
- C9: `python3 $D/train_nd_cpu.py --arm {PH5,PH6,PH5C,RoPE} --seed {4000,4001,4002} --epochs 100 --out OUT`
  (~55 min each); stored outputs `pilot_5d/`.
- Pilot seeds (3000-3002, 4000-4002) lie outside every registered batch (rule 29). Checkpoints of the pilots were kept
  in the session scratchpad only (not in `runs/`).
