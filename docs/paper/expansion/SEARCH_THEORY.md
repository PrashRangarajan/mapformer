# Search theory for the two failure axes (sign and rank)

*CORRECTED 2026-10-21: one symmetric clock / map definition replaces S1's and A5's (`scripts/sign/clock_table.py`), S5 is a bound on the position factor rather than a signature theorem, and `step_basin.py` is not deterministic (`docs/paper/review_draft5/ANALYSIS_SCRIPT_AUDIT.md` B3, M-SG1-M-SG7). The body below is unchanged.*

*2026-10-20. Expansion note for the main paper (`docs/paper/MAIN_DRAFT1.md` Sec. 6, 8, 9; theory appendix
`docs/paper/APPENDIX_theory_v3.md`). CPU only; nothing here is registered and nothing changes a registered verdict. Every
statement carries one label: **ESTABLISHED** (a registered result, cited), **ANALYSIS** (a post hoc or exploratory
computation on stored checkpoints or a CPU run, script named), **THEORY** (proved here or in the appendix; proof given or
pointed to), **SPECULATION** (an account or prediction not yet tested). Scripts and outputs:
`docs/paper/expansion/scripts/sign/` (every `*.py` has a `*_out.txt` beside it). Notation follows the appendix: channel
c = (head, block), angle a_c(x) = omega_c * step_c(x), step = W_out W_in e (pre-|.|), latent z_x = W_in e_x, readout row w_c.*

---

## 0. Summary

**The unifying statement (THEORY for the static part, ANALYSIS for the end states, SPECULATION for the dynamics).**
Both failure axes are failures to remove the *common mode* of the move code -- the part of a move's angle that an
opposite move does not undo, i.e. a clock riding on the map. Write, for an inverse pair (x, y) in channel c, the clock
k_c = a_c(x) + a_c(y) (the phase a back-and-forth adds at zero displacement) and the map m_c = a_c(y) (the phase per unit
displacement). A map code needs k_c = 0 mod 2 pi in every channel attention reads. There are two ways to get there from a
generic small start:

- **R1, the collective route**: a linear subspace of the move latents on which every channel's clock vanishes at once and
  the map part is untouched (u = z_x + z_y = 0). Along it every channel's clock shrinks in proportion, and in a quadratic
  surrogate the latent gradient flow converges to it from every start (Proposition S2).
- **R2, per-channel decoupling**: in each channel separately, a readout row with zero clock and full map rank, reachable
  without crossing a fold or losing a map dimension.

| class | R1 | R2 (per channel; its success depends on how many channels start near their target) | ring (1D) | torus (2D) |
|---|---|---|---|---|
| signed, per-head rank >= D + 1 | yes (S2) | yes (S6) | 8/8 (every batch) | 8/8 (rank 3: 6/8) |
| signed, per-head rank = D | yes (S2) | **no**: a clock-free channel loses a map dimension (S6) | rank 1: 7/8 | rank 2: 0-2/8; 3D rank 3: 1/8 |
| signed, rank 4, move latents frozen (ACCOUNTS T2) | **no** (frozen) | yes, through zero per channel | default start (half the channels start opposite-signed): 7/8; \|.\| start (none do): 1/8 | not run |
| \|.\|, rank 4, small start | **no**: the latent route ends at a clock; no latent choice wraps > 2r channels at a generic readout (S3) | only to the wrap sheets: an unwrapped \|.\| channel has clock >= map (S1), its zero-clock points lie at \|k\| = 2 pi n, across \|k\| = pi (P3); every channel starts below pi | 0-2/8 (4/8 at lr x3) | 0-2/8 |
| \|.\|, start past pi (MECHANISM, TORUS_START) | no | per-channel wrap targets near the start | 8/8 | 0/8 (signed from the same start: 2/8) |

Proved: the R1/R2 inventory of each class (S1, S2, S3, S6). R1 does not depend on the start (S2(c)); R2 does, and the
measured counts follow that: R2 alone succeeds when many channels start near their per-channel target (signed frozen
latents from the default start 7/8; \|.\| past pi on the ring 8/8) and fails when none do (1/8; 0/8). Measured (ANALYSIS, `sign_clock.py`, `rank_clock.py`):
every failing arm, on both axes, ends with a usage-weighted clock as large as its map (clock/map 1.1-1.9 for \|.\| from
small starts, 1.39 for frozen-latent signed, 0.5-1.9 for unsolved rank-2 heads), and every success from a small start ends
clock-free (0.001-0.065), by R1 (latent clock 0.045-0.065 of the map at rank 3/4; u = z_x + z_y at 0.044-0.053 of the
latent norm for signed ring models). Theory (S5): any code with clock rate k serves revisits with gap g < pi / k and loses
the rest, which is the single failure signature of Sec. 6.3. **What the inventory alone does not explain**: why R1 alone
suffices on the ring (rank 1: 7/8) but not on the 2D torus (rank 2: 0-2/8). The parallel rank note
(`RANK_THEORY.md`, Props. 2-3) supplies it: at rank D >= 2 a free local move ("hiding" the drift by turning channels'
resolution off its direction) starves the global route; in 1D there is no such move. So, in one frame (S7): **sign
removes the global route; rank D >= 2 keeps it but lets hiding starve it; rank >= D + 1 adds a local move (parking) that
costs no map and does not starve it.**

**The sign axis in particular.**
1. *Why |.| fails from small starts* (THEORY + ANALYSIS). An unwrapped |.| channel satisfies |k_c| >= |m_c| and
   |k_c| >= 2 |delta_c| (the cone; S1): it is at least as much clock as map, and its only zero-clock non-blind points are
   the wrap sheets k_c = 2 pi n. The latent move that cancels every signed channel (R1) produces a(y) = a(x) under |.|, a
   clock (T1 / Remark R), and at a generic readout no latent choice wraps more than 2r of the C channels (S3, proved by an
   omega-scaling argument: the wrap targets in step units, 2 pi n / omega_c, differ across the omega ladder). Measured:
   34% (unweighted) / 56% (usage-weighted) of the change |.| needs at initialisation lies in the latent-reachable
   subspace, against exactly 100% for signed (`latent_reach.py`). Along straight paths with the content of a solved model
   held fixed, the |.| path to the wrapped solution rises above its running minimum by 0.6-1.9 nats in every seed, the
   signed path by 0.04-0.48 (complete separation, 8 vs 8); the signed collective latent route alone takes accuracy from
   0.30 to 0.72, the same route under |.| leaves it at 0.30 (`path_profiles.py`).
2. *Why the wall* (THEORY + ANALYSIS). In the quadratic surrogate, the |.| latent flow that removes the clock is a
   contraction to the blind origin (S4, the +1/-1 paragraph); at fixed clock the largest map a |.| channel can carry sits on the cone's
   boundary, where one move's step is zero (a "one-sided counter"). Trained |.| ring models put 0.30-0.36 of their sub-pi
   channels within 10% of that boundary (chance at initialisation 0.056) and end with 19-22 one-sided and 14-16 blind
   channels of 64 (signed: 2-4 and 8). The reanalysis's MIXED wall verdict is consistent with this: collapse is the
   boundary optimum of a clock-map trade-off, not only a sticky kink.
3. *Why |.| "fails" on +1/+2 although no zero crossing is needed* (ANALYSIS, `latent_geometry.py`, `lattice_p.py`).
   It does not fail in the step code. With moves +1/+2 the |.| class keeps a collective route (z_2 = +-2 z_1; two mirror
   copies, S4) and the |.| models find it: the relation a(+2) = 2 a(+1) holds in >= 0.98 of the used channels in 8/8 |.|
   runs (signed 7/8 at >= 0.99), the latent relation in 5/8 |.| runs (one through the mirror copy z_2 = -2 z_1, cos -0.98)
   against 7/8 signed. The runs that stay unsolved (|.| 6/8, signed 3/8) have the relation in place; what they miss is not
   visible in the step code. So GAUGE G1's UNMEASURED cell is not evidence of a sign cost without an inverse pair, and
   nothing here points to one; the +1/+2 task is harder for both classes for a reason outside the step map.
4. *"Exists, held, not found" as a statement about the landscape* (THEORY definitional + ANALYSIS, S8, Sec. 3.4). The
   three facts together say the obstruction is where the initialisation distribution sits relative to the basin of the
   solution set. With the content of a solved model frozen and only the step map trained (`step_basin.py`, 2000 steps),
   the start decides: signed from the default start reaches 0.83 accuracy (3/8 >= 0.9), while from the same-direction
   start signed (0.47), frozen-latent signed (0.40) and |.| (0.36) all stay near the start (0.29), and |.| from past pi
   reaches 0.64 (p 0.0007 against its small start); started halfway along the straight segment to the exact |.| solution,
   the |.| step map reaches 0.61 and returns to the solution in 0/8 runs (the solution itself is held). The signed escape from the same-direction start that full training
   shows (7/8) is NOT reproduced by the step map alone: the collective route needs the content to co-adapt, or more steps.

**What does not survive.** "The |.| failure needs an inverse pair" (no: the +1/+2 shortfall is not in the step code, and
G1 is unmeasured); "R1 is the whole story for rank" (no: rank 2 has R1 and fails on the torus); the strong form of the
requested unification ("the failing class lacks a linear route to cancellation") holds for sign and for frozen latents
but is FALSE for rank D, which has the linear route.

---

## 1. Setting

One layer, observation-step-free models (only the moves step) unless stated. Ring: moves +1 / -1 (inverse pair) or
+1 / +2 (GAUGE). Torus: four moves, two inverse pairs. Per channel c and inverse pair (x, y):

- clock k_c = a_c(x) + a_c(y), map m_c = a_c(y), delta_c = (a_c(y) - a_c(x)) / 2.
- Over an interval with n_x moves x and n_y moves y, displacement d = n_y - n_x, the phase is
  phi_c = m_c d + k_c n_x (exact: a_y n_y + a_x n_x = a_y (d + n_x) + a_x n_x).
- A *map code* has k_c = 0 mod 2 pi in every used channel; a *cancelling* channel has k_c = 0 mod 2 pi with m_c != 0 mod
  2 pi; a *blind* channel has m_c = 0 = a_c(x) mod 2 pi.
- Latents z_x = W_in e_x in R^r; signed step w_c . z_x; |.| step |w_c . z_x|. The relation latent u = z_x + z_y.
- Usage (attention's reading of a channel) = mean |q_c| over move tokens x mean |k_c| over observation tokens at layer 0,
  exact per token because layer 0's input is the embedding (`model_wrap_exist.usage_weights` up to token frequencies).

---

## 2. Theory statements

### S1. The clock floor of the |.| class (THEORY, proved)

**Proposition S1.** In the |.| class, for an inverse pair in channel c (a_c(x) = omega_c |w_c . z_x|,
a_c(y) = omega_c |w_c . z_y|, one omega_c of either sign):
(a) a_c(x) and a_c(y) have the same sign, so |k_c| = |a_c(x)| + |a_c(y)| >= max(|m_c|, 2 |delta_c|), with
|k_c| = |m_c| iff a_c(x) = 0 and |k_c| = 2 |delta_c| iff one of the two angles is 0.
(b) As real numbers, k_c = 0 iff the channel is blind. So the non-blind zero-clock points of the channel are exactly the
wrap sheets |k_c| in 2 pi Z_{>=1}.
(c) On a zero-displacement revisit with n_x = g / 2 backward moves (g the number of moves in the interval), the channel's
phase is k_c g / 2, with |k_c| g / 2 >= |m_c| g / 2. If |k_c| < 2 pi (unwrapped), the phase is within eps of 0 mod 2 pi
only for g <= 2 eps / |k_c| <= 2 eps / |m_c| (or g near a multiple of 4 pi / |k_c|).

*Proof.* (a) same sign of omega_c times non-negative numbers; triangle equality cases. (b) from (a). (c) phi = m d + k n_x
with d = 0 and n_x = g / 2. []

*Reading.* An unwrapped |.| channel is at least as much clock as map, and the better it resolves neighbouring cells
(|m_c| large), the shorter the backtracks it can read. The signed class has k_c = 0 with any m_c (through zero). The
appendix's P3 (the cut circle) is the geometric form of (b): the near-side cancelling point is the blind origin, the
non-blind ones lie at |k| = 2 pi, across |k| = pi.

*Check (ANALYSIS, `sign_clock.py`).* Usage-weighted clock/map ratio rms(wrap k) / rms(wrap m) at the end of training:
\|.\| from small starts 1.10-1.68 per arm mean (ring MECH AO 1.68, AO_absmag 1.63, GAUGE AO_m 1.55; torus AO4 1.22), signed
0.001-0.018 (SO, SO_absmag, SO_m, SO4). The inequality itself holds with zero violations on raw angles (it is an identity: 0
violations in 4,059 sub-pi |.| channels over all arms; a first version of the script compared signed sums and counted
channels with omega < 0 as violations, corrected before this run).

### S2. The collective route of the signed class (THEORY, proved)

**Proposition S2.** Signed class, inverse pair, readout W in R^{C x r} of full column rank, omega_c != 0.
(a) The clock vector is k = diag(omega) W u with u = z_x + z_y. The latent pairs at which every channel has zero clock form
the r-dimensional linear subspace {u = 0}; the map part delta = (z_y - z_x)/2 is free on it, so it contains non-blind
points (any z_y not in ker W).
(b) Along z_y(tau) = (1 - tau) z_y - tau z_x (W, omega fixed), u(tau) = (1 - tau) u(0) and every channel's clock is
(1 - tau) k_c(0): all channels shrink together, monotonically.
(c) Quadratic surrogate R(z) = 1/2 sum_c q_c k_c^2 (q_c >= 0 usage weights). Gradient flow on (z_x, z_y) gives
du/dt = -2 H u with H = W^T diag(omega) diag(q) diag(omega) W, positive definite when the used channels' rows span R^r, and
d(delta)/dt = 0. So u(t) = exp(-2 H t) u(0) -> 0 from every start, the map part is unchanged, and {u = 0} is the only
stationary set.
(d) Generic uniqueness. For Lebesgue-almost every (W, omega), {u = 0} is the only latent set at which every channel cancels
mod 2 pi (no latent u != 0 makes diag(omega) W u a non-zero vector of 2 pi Z^C when C > r).

*Proof.* (a)-(c) linear algebra: k_c = omega_c (w_c . z_x + w_c . z_y); dR/dz_x = dR/dz_y = H u. (d) diag(omega) W u in
2 pi Z^C iff the vector (2 pi n_c / omega_c)_c lies in col(W); for fixed n != 0 the map omega -> (2 pi n_c / omega_c) is a
diffeomorphism of (R \ 0)^C onto itself, so its preimage of the r-dimensional subspace col(W) is null when r < C; union over
countably many n. []

ACCOUNTS T2 (ESTABLISHED) is the causal test of (b): freezing the two move latents blocks the escape from the |.| start
(1/8 vs 7/8, Fisher p 0.010; 15% vs 82% opposite-signed channels). Trained signed ring models end on {u = 0}:
|u| / (|z_x| + |z_y|) = 0.044-0.053, cos(z_x, z_y) = -0.994 to -0.995 (ANALYSIS, `latent_geometry.py`; MECH SO, SO_absmag,
GAUGE SO_m); MapFormer's Fig. 3d / 9b describe the same antipodal end state.

### S3. The |.| class has no collective route for an inverse pair (THEORY, proved)

**Proposition S3.** |.| class, inverse pair, C channels, rank r.
(a) The latent route of S2(b) ends, under |.|, at a(y) = a(x) in every channel: |w . (-z_x)| = |w . z_x|. It is a clock.
(b) For Lebesgue-almost every (W, omega), at that fixed (W, omega) no latent pair (z_x, z_y) makes more than r channels of
each sign-coherence class (pre-|.| steps of equal or of opposite sign), hence more than 2r channels, cancel non-blindly.
(c) Hence every |.| map code with more than 2r cancelling channels has its readout (W, omega) tuned channel by channel: the
solution set is a countable union of sheets indexed by the wrap counts n in Z^{C'} of the C' used channels, and on each
sheet the channel-c condition, in step units, is |w_c . z_x| + |w_c . z_y| = 2 pi n_c / |omega_c|, an offset that differs
across the omega ladder (omega_max / omega_min = grid size: 32 on the ring, 64 on the torus).

*Proof.* (a) evenness of |.|. (b) Fix a set S of 2r + 1 channels, a sign pattern (e_c(x), e_c(y)) per channel and
non-zero integers n_c (a non-blind |.| channel cancels only by wrapping, S1(b)). On the sign region the angle sum is
k_c = omega_c (e_c(x) w_c . z_x + e_c(y) w_c . z_y), linear in zeta = (z_x, z_y) in R^{2r}: k = diag(omega) N zeta with N
the (2r + 1) x 2r matrix of rows (e_c(x) w_c, e_c(y) w_c). The conditions k_c = 2 pi n_c for c in S hold for some zeta iff
v(omega) = (2 pi n_c / omega_c)_{c in S} lies in col(N), a subspace of dimension <= 2r < 2r + 1 that does not depend on
omega. As in S2(d), omega -> v(omega) is a diffeomorphism of (R \ 0)^S, so the set of omega_S for which it holds is null; by
Fubini the set of (W, omega) is null. (Within one coherence class the sum depends on zeta only through z_x + z_y or
z_x - z_y, an r-vector, so the same argument with r + 1 channels of one class gives the per-class bound.) Countable union over S, the sign patterns and n. (c) from (b): at any (W, omega)
outside the null set, more than 2r cancelling channels requires moving (W, omega); the channel-c equation is the |.| sum
omega_c (|w_c . z_x| + |w_c . z_y|) = 2 pi n_c written in step units. []

*Reading.* The signed class reaches every channel's zero clock by r numbers (u = 0), whatever the readout. The |.| class
needs the readout itself to carry per-channel integer offsets. The constructed solution AO_bal is such a point: its latents
are orthogonal (c e1, c e2) and its cancellation sits entirely in W_out (W_out[c, 0] + W_out[c, 1] = 2 pi n_c / (omega'_c c)).
This is the precise sense in which the wrapped code needs "coordinated, channel-specific" configurations.

*How many latent directions?* (ANALYSIS, `aobal_geometry.py`, the held AO_bal models after 300 epochs, seeds 600-607.)
Both codes are carried by ONE latent combination: in the balanced construction every pre-|.| step is >= 0 (same-sign
share 0.77-0.86 after training), so on that sign region the |.| clock is diag(omega) W u with u = z_x + z_y, exactly as in
the signed class. The difference is the target: the signed code needs W u = 0, which every readout contains; the wrapped
code needs diag(omega) W u = 2 pi n (here n_c = 1 on every cancelling channel), which a readout drawn at initialisation
carries 0.3-11% of (share of the target in col(diag(omega) W_init); random-subspace baseline 4/64 = 6%). The held models
cancel in only 9-13 channels, which carry 92-98% of the usage, so S3(b)'s bound of 2r = 8 latent-cancellable channels
leaves a handful of channels to tune: S3 alone is a modest obstruction; the larger ones are S3(a) (the collective move
makes a clock) and S1 + P3 (each channel's target lies at k = 2 pi, across pi, from a small start).

*Check (THEORY check, `prop_checks.py` (1)).* At the default readouts of seeds 800-807, least-squares latents for random
channel sets, sign regions and wrap counts n in {1, 2, 3} (4,000 trials per size): 9 channels never cancel exactly
(smallest max error 0.027 rad); 8 channels cancel exactly (1e-14) only when they split 4 / 4 between the coherence classes
(median error 7.6 rad over random splits), as the per-class bound says.

*Check (ANALYSIS, `latent_reach.py`).* At the default initialisation of MECHANISM seeds 800-807, the share of the needed
change of the angle sums that a latent move can produce: signed (to zero) 1.000 in every seed (by S2(a)); |.| (to the
nearest wrap) 0.339 unweighted, 0.561 usage-weighted (random-subspace baseline 8/64 = 0.125); at the |.| (absmag) start in
the balanced gauge 0.285 / 0.463 (that start's latent-reachable space has rank 2). A latent move does help |.| partly (it
scales all sums up together), but the residual 44-66% must come from per-channel readout changes.

### S4. Moves +1 / +2: mirror copies and frustration (THEORY in the surrogate; ANALYSIS)

**Proposition S4.** Relation n = (n_x, n_y) (the code needs n_x a(x) + n_y a(y) = 0 per channel: +1/-1 is n = (1, 1);
+1/+2 is n = (-2, 1)).
(a) Signed: the residual is omega_c w_c . u with u = n_x z_x + n_y z_y; one linear zero set {u = 0} (as S2).
(b) |.|: on a sign region the residual is omega_c w_c . (n_x e_c(x) z_x + n_y e_c(y) z_y). If n_x n_y < 0 (no inverse
pair) the class contains two collective zero sets, {n_x z_x + n_y z_y = 0} and its image under z_y -> -z_y (the |.| class is
invariant under that reflection; S3(a)), and both contain non-blind points. If n_x n_y > 0 (an inverse pair) neither is
non-blind (S3).
(c) Frustration. In the quadratic surrogate R = 1/2 sum_c q_c (residual_c)^2 restricted to a sign region in which both
coherence classes (e_c(x) e_c(y) = +1 and = -1) contain channels whose rows span R^r, the only latent stationary point is
the blind origin zeta = 0, and R is a positive-definite quadratic in zeta there, so the latent flow contracts toward it until
some channel's step reaches zero (the wall).

*Proof.* (a), (b) as S2, S3. (c) On the region, residual = A_+ (n_x z_x + n_y z_y) for the coherent channels and
A_- (n_x z_x - n_y z_y) for the others (rows omega_c w_c, sign absorbed); dR/dz_x = n_x (H_+ u_+ + H_- u_-),
dR/dz_y = n_y (H_+ u_+ - H_- u_-), with H_+- positive definite; both vanish iff u_+ = u_- = 0 iff z_x = z_y = 0. []

*Check (`prop_checks.py` (3)).* +1/+2 surrogate R = 1/2 mean_c (omega_c (step_c(+2) - 2 step_c(+1)))^2 with z_1 fixed at
unit norm, Adam on z_2 from random starts, real readouts of seeds 800-807: signed reaches the collective zero z_2 = 2 z_1
from 80/80 starts; |.| reaches one of its two collective zeros from 30/80 (16 of them the mirror copy z_2 = -2 z_1) and
stalls elsewhere from 50/80. Frustration costs the |.| class the collective route from most random starts even with no
inverse pair; in the trained transformers the per-channel route then supplies the relation (below).

*For +1/-1 the same computation gives (b) of the summary:* every channel's residual is the clock k_c >= 0, so on any
region R is a positive-definite quadratic with minimum at the blind origin: **the |.| latent gradient that removes the clock
is a contraction toward the wall**, while the signed one (S2(c)) projects onto u = 0 and leaves the map alone. This is the
surrogate-level mechanism of the collapse the reanalysis measured. *Check (`prop_checks.py` (2)):* gradient descent on the
clock surrogate from the real initial latents and readouts (4 seeds): signed ends at u = 0 exactly (|u| / |z| < 1e-3) with
the map part unchanged (relative change 0.000) and mean |map angle| 0.12-0.50; |.| ends with both latents at 0 (norm ratio
0.000), map part gone, mean |map angle| 0.000 -- the blind origin. Trained models show the same collapse in part (`collapse_wall_out.md`: 24-42% of |.| move steps on the
ring, 60-70% on the torus; signed 10-16%, 34-40%).

*Check (ANALYSIS; GAUGE seeds 1300-1307, `latent_geometry_out.txt`, `lattice_p_out*.txt`).*
- At initialisation the pre-|.| sign coherence is 0.52-0.77 (mixed: both classes populated).
- +1/+2, latent relation (min over the two mirror copies of |z_2 -+ 2 z_1| / (|z_2| + 2 |z_1|) < 0.1): |.| 5/8 (s1307 via
  the mirror copy: cos(z_1, z_2) = -0.980, ratio 1.99, same-sign share 0.02), signed 7/8. The 3 |.| runs without it stay
  mixed (coherence 0.53-0.67, residual 0.47-0.73): frustrated.
- In the channels attention uses the |.| models satisfy the relation anyway (usage share with |wrap(a(+2) - 2 a(+1))| <
  0.05: 0.98-1.00 in 8/8 |.| runs; signed 0.99-1.00 in 7/8): where the collective route is frustrated the per-channel route
  supplies it, because on +1/+2 the per-channel targets lie on the near side (no wall, no far wrap).
- Unsolved runs of both classes have the relation in place (|.| s1300-s1304, s1306; signed s1304, s1306); the lap lattice
  32 a(+1) = 0 mod 2 pi does not separate them either (usage share within 0.05 rad: solved |.| runs 0.15 and 0.38, unsolved
  |.| runs 0.00-0.99, four of them >= 0.75). What the unsolved +1/+2 runs miss is not visible in the step code (ANALYSIS;
  the usage proxy is per token, not per context; content and readout not examined).
- Inverse pair: the same readout gives 1-collapsed / 2-collapsed channels (of 64) 19.4 / 13.9 (MECH AO), 22.2 / 16.0
  (AO_absmag), 18.8 / 13.2 (GAUGE AO_m), against 2.1-4.1 / 8.1-8.9 signed and 15.6 / 5.0 at initialisation. Sub-pi |.|
  channels within 10% of the cone boundary (|delta| >= 0.9 k / 2): 0.30 (AO), 0.36 (AO_absmag), 0.25 (AO_m); 0.056 at
  initialisation; 0.00-0.02 for the large starts.

### S5. A clock produces the failure signature (THEORY, proved; ANALYSIS for the measured clocks)

**Proposition S5.** Let a channel's code have clock k_c and map m_c (any class), and let a revisit at displacement 0 have
n_x backward moves. Its phase is k_c n_x. For a population of channels with usage weights q_c (sum 1), the same-cell
score's position factor F(n_x) = sum_c q_c cos(k_c n_x) equals 1 at n_x = 0; if a usage share s of the channels has
kappa <= |k_c| <= K (unwrapped), then F(n_x) <= 1 - s (1 - cos(kappa n_x)) for n_x <= pi / K. Index RoPE is the extreme case:
every token, observations included, advances every channel, so the whole code is clock.

*Proof.* phi = m d + k n_x with d = 0. For n_x <= pi / K, kappa n_x <= |k_c| n_x <= pi, and cos is decreasing on [0, pi],
so cos(k_c n_x) <= cos(kappa n_x) on the share s and <= 1 elsewhere. []

So any code with a non-zero clock serves short-gap revisits and loses long-gap ones; wrap-only revisits also need the
lattice N m_c = 0 mod 2 pi, which a code with a large clock cannot read either (its long-gap revisits are already lost).
This is the common failure signature of the main draft Sec. 6.3 (index RoPE 0.784 / 0.462 / 0.471, shared rank 2 0.950 /
0.671 / 0.557, |.| 0.870 / 0.630 / 0.515 for short / long / wrap-only; ESTABLISHED), derived rather than observed.

*Measured clocks (ANALYSIS, `sign_clock.py`, `rank_clock.py`; usage-weighted rms clock / rms map).*

| arm | SOLVED | clock / map |
|---|---|---|
| ring signed (MECH SO, SO_absmag; GAUGE SO_m) | 8/8, 7/8, 8/8 | 0.015-0.018 |
| ring \|.\| from small starts (MECH AO, AO_absmag; GAUGE AO_m) | 0/8, 0/8, 1/8 | 1.55-1.68 |
| ring signed, move latents frozen, \|.\| start (T2 SO_absmag_fz) | 1/8 | 1.39 (the one solved run 0.06) |
| ring \|.\| from past pi (MECH AO_high) | 8/8 | 0.31 (wrapped; usage on cancelling channels 0.82) |
| torus signed SO4 / \|.\| AO4 (TORUS_START) | 8/8 / 1/8 | 0.001 / 1.22 (the solved AO4 run 0.02) |
| torus rank 2 heads, unsolved (shared, per-head, block-diagonal) | 0/8, 2/8, 2/8 | 0.51-1.91 (latent clock 1.6-3.2) |
| torus rank 3 / 4 heads, solved | 6/8, 8/8, 8/8 | 0.043-0.051 (latent clock 0.045-0.065) |

Every arm that fails from a small start ends with a clock about as large as its map; every arm that succeeds from a small
start removes it, and at rank 3 / 4 it is removed in the latent (R1), not parked. (Rank 2 heads that solved keep a
moderate clock, 0.26-0.40; large-start arms carry wrapped codes whose rms clock is not small although they cancel in the
used channels: the ratio is a coarse summary there.)

### S6. The rank analogue: clock and map share every channel at rank D (THEORY, proved; it restates the rank theory T1)

**Proposition S6.** One head, latent dimension r, move latents z_m in R^r; for a D-dimensional map write the common mode
c (the sum of the latents of each inverse pair, stacked) and the map frame F = [d_1 ... d_D] in R^{r x D} (half-differences),
rank F = D. Channel c's clock along pair j is omega_c w_c . c_j and its map wavevector is omega_c F^T w_c.
(a) r = D: if c_j != 0 then c_j = F alpha_j with alpha_j != 0, and channel c's clock equals its wavevector's component
along alpha_j: omega_c w_c . c_j = (omega_c F^T w_c) . alpha_j. A channel is clock-free iff its wavevector is orthogonal to
every alpha_j, i.e. it resolves at most D - (number of independent alpha_j) dimensions. So with a non-zero common mode,
no channel is both clock-free and a full D-dimensional map (no R2), and the only way to a clean code is c = 0 (R1).
(b) r >= D + 1: if the components of the clock vectors outside col(F) are linearly independent (possible only for
q <= r - D independent clock directions), every channel can be clock-free with any wavevector (choose the part of w_c
orthogonal to col(F) to cancel every w_c . c_j). R1 and R2 both exist. On the torus a single common clock (all four moves
advancing alike, the index-like start) needs r >= 3; two independent pair clocks need r >= 4.

*Proof.* (a) c_j in R^D = col(F) when F is invertible. (b) w = F (F^T F)^{-1} kappa + w_perp with w_perp in col(F)^perp
free; w . c_j = kappa . (F^T F)^{-1} F^T c_j + w_perp . c_j,perp for j = 1..q, a linear system in w_perp with linearly
independent coefficient vectors c_j,perp, hence solvable. []

*Reading.* At rank D a clock is a uniform drift of the decoded position (alpha_j per move pair): the rank theory's T1
("a rank-D head cannot place a clock off the map"). In route terms, rank D keeps R1 (u = 0 is linear at every rank) but
loses R2. The measured rank-3 / 4 successes nevertheless end on R1 (latent clock 0.045-0.065 of the map), and the rank-2
failures end clock-coupled (latent clock 1.6-3.2): **the end states match the sign axis; the route that rank 2 lacks is R2,
not R1**.

### S7. The unification, stated once (THEORY: inventory; ANALYSIS: end states; SPECULATION: dynamics)

**Proposition U (route inventory).** For an inverse-pair map task, a class reaches the zero-clock set from a generic small
start through R1 iff its clock is a linear function of one latent vector whose zero set contains non-blind points, and
through R2 iff each channel has a clock-free, full-map readout reachable without crossing a fold. Then:
signed r >= D + 1: R1 and R2 (S2, S6(b)); signed r = D: R1 only (S6(a)); signed with frozen move latents: R2 only (the
latents cannot move; each channel can still cross zero through W_out); |.|: neither from a small start (S3(a, b): the
latent route ends at a clock, and no latent choice wraps more than 2r channels; S1 + P3: per channel the zero-clock points
are at |k| = 2 pi n, across |k| = pi).

**Empirical claim (ANALYSIS + ESTABLISHED counts).** R1 and R2 together: 6/8-8/8 in every batch measured (ring and
torus). Neither (|.| from small starts): 0-2/8 in every ring and torus recipe but one (ROBUST lr x3 on the ring, 4/8, unmeasured against 8/8). R2 only depends on the start, as it should:
signed with frozen move latents solves 7/8 from the default start (about half the channels start opposite-signed, i.e.
near their through-zero target) and 1/8 from the |.| start (none do); |.|, whose per-channel targets are the wrap sheets,
solves 8/8 on the ring from a start past pi and 0-2/8 from small starts. R1 only: 7/8 on the 1D ring (rank 1), 0-2/8 on the
2D torus (rank 2), 1/8 on a small 3D torus (rank 3). Every failure ends clock-coupled (S5 table). So: **from a small start
whose channels are not already near their per-channel targets, learning needs the collective route; on the 2D and 3D
torus it has needed per-channel decoupling as well**.

**Why R1 alone fails in 2D: the rank note's answer (THEORY for the gradient statements, SPECULATION for the dynamics;
`docs/paper/expansion/RANK_THEORY.md`, written in parallel, read-only here).** RANK_THEORY Prop. 1 is the same two-route
split as S6 (its "global" route = R1, its "local" route = R2; at r > D the local route "parks" the clock off the map plane
at zero map cost, at r = D it "hides" the drift by removing the channel's resolution along the drift direction beta). Its
Props. 2-3 supply the missing dynamics: at r = D >= 2, hiding costs nothing to first order when the head's resolution is
isotropic (rotating a channel's wavevector off beta leaves tr P unchanged), so the readout takes it at once; but the
global route removes the drift at a rate proportional to the head's resolution along beta, which hiding has just lowered
-- a positive feedback that leaves the drift in place (a plateau, or the CLOCK / COLLAPSE end states). At D = 1 there is no
direction to hide along (hiding kills the channel and leaves the drift-to-map ratio unchanged), so the global route is not
starved: rank 1 solves 7/8. This explains the one anomaly of Proposition U.

**The sign axis in the same terms (THEORY).** For |.| the local route to a zero clock *below pi* is the cone's: shrinking
k_c shrinks the largest map the channel can carry (S1(a), |m_c| <= |k_c|), so it is "hiding" with no free direction, as
at D = 1 -- the channel dies (blind) or becomes a one-sided counter -- and the global route does not exist (S3(a)). The two
axes therefore fail for different reasons inside one frame: **sign removes the global route; rank D >= 2 keeps it but lets
a free local move (hiding) starve it; per-head rank >= D + 1 adds a free local move (parking) that does not starve it**.
The common end state (a clock as large as the map, S5 table) and the common signature (S5) follow.

*Consistency with the measured end states (ANALYSIS, `rank_clock.py`).* Unsolved rank-2 heads keep a latent clock 1.6-3.2x
the map frame with the clock pattern inside the map span (in-span 0.91-0.98, wrapped); solved rank-3/4 heads have a latent
clock of 0.045-0.065 of the frame (and a read clock 0.043-0.051), i.e. by the end the clock is mostly removed, not merely
parked; whether it was parked first (RANK_THEORY P4) needs mid-training snapshots (Sec. 6, item 3). RANK_THEORY P3 (the
leftover clock lies off-plane) is compatible with these numbers (a small leftover), not tested here.

Predictions shared by the two notes: (i) a rank-2 torus model whose move latents are tied to cancel (drift forbidden, as
RANK_TESTS' R1DF) solves like rank 4; (ii) rank D plus a learned per-channel clock slot (RANK_THEORY P7) behaves like
rank D + 1; (iii) on the sign axis, a model with a non-negative readout but a signed latent (global route restored, Sec. 6
item 4) loses the sign cost.

### S8. "A solution exists, is held, and is not found" as a statement about basins (THEORY, definitional; ANALYSIS)

Let Phi_B be the stochastic training map for budget B, S_eps = {theta : L(theta) <= eps} the solution set of a class, and
P_0 the initialisation distribution. Define the basin Bas = {theta : P(Phi_B(theta) in S_eps) >= 1/2}. Then:
- *exists*: S_eps is non-empty in the class (Proposition 1 / P1(a), constructed);
- *held*: the constructed points lie in Bas (ESTABLISHED: ring 8/8, torus re-found 8/8, rank-2 projection 7/8);
- *not found*: P_0(Bas) is small (ESTABLISHED: SOLVED fractions 0-2/8 estimate it, with binomial error).
Together they say the obstruction is the position of supp(P_0) relative to Bas, not the emptiness or instability of
S_eps. Two cheap measurements of that position follow (Sec. 3.3, 3.4): the loss profile along straight paths from the start
to a constructed solution (barriers), and the basin's extent along such a segment (train from (1 - tau) theta* + tau
theta_0 and record success).

---

## 3. Analyses run (CPU; scripts in `docs/paper/expansion/scripts/sign/`)

### 3.1 Latent geometry and per-channel patterns (`latent_geometry.py` -> `latent_geometry_out.txt`, `_rows.json`)
Eval-only on GAUGE (+1/-1 and +1/+2, seeds 1300-1307, initialisation rebuilt exactly and checked bit-exact against
MECHANISM's stored default angles: max difference 0.0) and MECHANISM (seeds 800-807). Results quoted in S2, S4.

### 3.2 Clock / map ratios (`sign_clock.py`, `rank_clock.py`) and the reachable share (`latent_reach.py`)
Results in S1, S3, S5, S6. `rank_clock.py` reads RANK_MI / RANK_SEP / RANK3 checkpoints (full models: observations step);
the observation clock (mean observation step) is 0.015-0.018 of the map in solved rank-3/4 heads and 0.18-0.52 in unsolved
rank-2 / rank-3 heads.

### 3.3 Loss along straight paths with the content of a solved model fixed (`path_profiles.py` -> `path_profiles_out.txt`)
Ring, MECHANISM SO seeds 800-807; content (everything but W_in, W_out, omega) from the solved signed model; balanced gauge,
so a straight line in W_out is a straight line in angle space. 64 fixed walks, T = 128, eval mode. The start loss is high
(8.8-10.4 nats) because the solved content is confident.

| path | loss at tau = 0, 0.2, ..., 1 (mean of 8) | accuracy at tau = 0 -> 1 | max rise above running minimum, per seed |
|---|---|---|---|
| S_lin: signed, \|default\| -> signed solution | 9.46 9.18 8.65 7.50 6.41 0.00 | 0.30 -> 1.00 | 0.27 0.04 0.30 0.40 0.48 0.38 0.05 0.36 |
| A_lin: \|.\|, \|default\| -> wrapped solution | 9.46 8.23 7.87 7.96 7.44 0.00 | 0.30 -> 1.00 | 1.92 0.87 1.41 0.60 1.09 1.01 1.14 0.74 |
| S_lat: signed, collective latent route only (W_out, omega at the start) | 9.46 9.39 9.49 9.25 8.35 2.98 | 0.30 -> 0.72 | 0.03 0.00 0.00 1.36 2.15 1.44 0.21 0.02 |
| A_lat: \|.\|, the same latent route | 9.46 9.41 9.66 9.90 9.68 9.21 | 0.30 -> 0.30 | 0.17 1.11 0.18 2.40 2.23 1.68 1.33 0.51 |

Reading (ANALYSIS, n = 8, exploratory): the |.| straight path to its exact solution is rugged (every seed's largest rise
exceeds every signed seed's: complete separation, exact two-sided permutation p = 2 / C(16, 8) = 1.6e-4 for "all 8 |.|
rises above all 8 signed"), the signed one nearly monotone. Rotating one move latent toward the negative of the other,
with nothing else changed, takes the signed model from 0.30 to 0.72 accuracy (monotone in 5/8 seeds) and leaves the |.|
model at 0.30 (it ends at a clock, S3(a)). This is the transformer version of the review's toy `path_cut` (where the route
started on a plateau); here it descends from the start in 5/8 seeds.

### 3.4 The step map's basin with the content frozen (`step_basin.py`, driver `run_basin.sh` -> `runs_basin/`)
Ring, MECHANISM seeds 800-807. Content (embeddings, attention, FFN, readout) from the solved signed model, FROZEN; only
W_in, W_out, omega train (AdamW lr 1e-3, wd 0.05, cosine, 2000 steps of batch 32, eval mode: about 2% of the samples of a
full 300-epoch run). Both classes have an exact solution with this content (signed_balanced / balanced construction). The
start loss is 7-10 nats (confident content, wrong step map). No run reaches the house SOLVED loss (0.05) in this budget,
so the readout is fresh-stream accuracy (100 walks, T = 128). `basin_summary_out.txt`; exploratory, n = 8.

| start (balanced gauge) | class | accuracy start -> end (mean) | per seed (end) | acc >= 0.9 |
|---|---|---|---|---|
| default angles | signed (S_def) | 0.33 -> **0.83** | 0.89 0.86 0.95 0.70 0.98 0.75 0.96 0.56 | 3/8 |
| \|default\| (the \|.\| start) | signed (S_abs) | 0.29 -> 0.47 | 0.37 0.94 0.34 0.75 0.28 0.37 0.35 0.32 | 1/8 |
| \|default\|, move latents frozen | signed (S_abs_fz) | 0.29 -> 0.40 | 0.36 0.32 0.33 0.76 0.33 0.35 0.37 0.36 | 0/8 |
| \|default\| | \|.\| (A_abs) | 0.29 -> 0.36 | 0.33 0.37 0.35 0.55 0.28 0.35 0.36 0.31 | 0/8 |
| a(+1), a(-1) iid U(1.7, 4.5) (past pi) | \|.\| (A_high) | 0.36 -> 0.64 | 0.81 0.50 0.74 0.56 0.92 0.53 0.55 0.53 | 1/8 |
| the exact wrapped solution (pilot, seed 800 only, 300 steps) | \|.\| (A_sol) | 1.00 -> 1.00 | held | -- |

Contrasts (permutation p on accuracy): default vs |.| start, signed: +0.365 (p 0.005); past-pi vs small start, |.|: +0.280
(p 0.0007); signed vs |.| from the |.| start: +0.104 (p 0.32, not separated); signed free vs frozen latents from the |.|
start: +0.069 (p 0.56, not separated).

Reading (ANALYSIS, exploratory). (i) With the content fixed, the START decides the step map's search: from the default
start the signed step map recovers most of the map in 2000 steps; from the same-direction (|.|) start neither class does,
and the past-pi start helps the |.| class (the MECHANISM direction, reproduced in the step map alone). (ii) **The signed
escape from the |.| start (7/8 in full training, MECHANISM SO_absmag) is NOT reproduced when only the step map trains**:
2/8 signed runs move off the start (0.94, 0.75) against 1/8 frozen-latent and 1/8 |.| runs. So the collective route,
although a descent direction along the straight latent path (Sec. 3.3: 0.30 -> 0.72 with nothing else changed), is not
what gradient descent on the step map alone takes from that start within this budget; in full training the escape also
involves the content weights (or more steps). This qualifies S2(c): the surrogate's guarantee is about the clock term
alone. (iii) In the language of S8: the default signed start lies in (or near) the step map's basin of its solution, the
same-direction start lies outside it for both classes, and the past-pi start is closer for |.|. The basin's extent along
the segment from the |.| start (tau = 1) to the wrapped solution (tau = 0), angles (1 - tau) a* + tau |a_0|, |.| class:

| tau | 1 (A_abs) | 0.75 | 0.5 | 0 (A_sol pilot) |
|---|---|---|---|---|
| accuracy at the start of training | 0.29 | 0.32 | 0.32 | 1.00 |
| accuracy after 2000 step-map steps (mean of 8) | 0.36 | 0.51 | 0.61 | 1.00 (held, n = 1) |
| runs >= 0.9 | 0/8 | 0/8 | 0/8 | -- |

(permutation p against tau = 1: 0.012 at 0.75, 0.0017 at 0.5; tau = 0.25 and the signed segment were dropped to save CPU
time.) Halfway along the straight segment the |.| step map recovers part of the map but returns to the solution in no
run, while the signed step map from its default start (a generic point, not on any segment) reaches 0.83 (p 0.013 against
tau = 0.5). In S8's terms the wrapped solution's basin, for the step map with this content and budget, does not reach
halfway from the |.| start; midpoints of the segment score like the start itself (0.32), consistent with the rugged
straight path of Sec. 3.3.

---

## 4. What survives

- THEORY, proved: S1 (clock floor of |.|), S2 (signed collective route; uniqueness), S3 (no collective route for |.| on an
  inverse pair; at most 2r latent-cancellable channels at a generic readout; channel-specific wrap offsets), S4 (mirror
  copies; frustration and contraction in the quadratic surrogate), S5 (a clock gives the short-gap-only signature), S6 (rank
  D: no clock-free full-map channel while the common mode is non-zero), U (the route inventory).
- ANALYSIS: every failing arm on both axes ends clock-coupled (clock/map 0.5-1.9), every small-start success clock-free
  (<= 0.065) by R1; |.| ends with one-sided and blind channels at the cone boundary (5-6x the initial rate); on +1/+2 the
  |.| step code is learned (relation in 8/8 runs) and the unsolved runs fail elsewhere; the signed collective route is a
  descent direction on the real model with solved content (0.30 -> 0.72), the |.| one is not; the |.| straight path to its
  solution is rugged, the signed one is not; with the content frozen the start decides the step map's search for both
  classes, and the signed escape from the same-direction start is not reproduced by the step map alone.
- Not surviving: the strong unification "the failing class has no linear route" (false for rank D); "|.| fails without
  an inverse pair" (no step-code evidence); the wall as the sole account (it is the boundary optimum of the cone trade-off
  plus a contraction, and the reanalysis's end-of-training gradients are MIXED).
- Open: whether hiding (RANK_THEORY Props. 2-3) is what starves R1 at rank 2 in training (its gradient statements are proved there; the dynamics and the predictions P2, P4, P7 are untested); what the unsolved +1/+2 runs miss; whether per-channel
  capture or a collective effect produces the past-pi rescue on the ring and why it fails on the torus.

---

## 5. Proposed paper text

### 5.1 Main text: a theory subsection (~1 page; for Sec. 6, after 6.2, replacing the Limitations line "why the
non-negative parameterisations stall from small starts is not identified")

**6.3 Why search fails: one common mode, two missing routes.** A map code must cancel opposite moves: in each channel the
clock k = a(x) + a(y), the phase a back-and-forth adds at zero displacement, must vanish mod 2 pi while the map m = a(y)
does not. Any non-zero clock makes a revisit's phase grow with the number of moves since the last visit, so the code reads
short-gap revisits and loses long-gap and wrap-only ones (Proposition S5) -- the single failure signature of every failing
model (Sec. 6.4). The question is how a class removes its clock from a small random start. There are two routes. In the
*collective* route the clock vector is diag(omega) W_out (z_x + z_y), so the r-dimensional latent condition z_y = -z_x
zeroes every channel at once and leaves the map untouched; in a quadratic surrogate the latent gradient converges to it from
every start (Proposition S2). In the *per-channel* route each channel's readout is tuned to a clock-free row of full map
rank. A signed step at rank D + 1 has both. A non-negative step has neither from a small start: under |.| the collective
move produces a(y) = a(x), a clock, and at a generic readout no choice of latents makes more than 2r of the C channels wrap
(Proposition S3: the wrap targets 2 pi n / omega_c differ across the frequency ladder); per channel, an unwrapped |.| angle
pair has clock at least as large as its map (|k| >= |m|, Proposition S1), and its clock-free non-blind points lie at
|k| = 2 pi, across |k| = pi. At rank D the collective route remains, but no channel can be clock-free and a full map while
the common mode is non-zero (Proposition S6, the rank theory's drift). The trained models match this inventory: every model
that fails from a small start, on either axis, ends with a usage-weighted clock as large as its map (ratio 0.5-1.9;
non-negative, frozen-latent signed and rank-2 alike), and every success ends clock-free (<= 0.065) through the collective
route (Table X). Freezing the move latents of a signed model removes the collective route and leaves it where the
non-negative step ends (1/8; clock ratio 1.39 against 1.6). With the content of a solved model held fixed, rotating one
move latent toward the negative of the other takes the signed model from 0.30 to 0.72 accuracy with nothing else changed,
and leaves the non-negative one at 0.30 (gradient descent on the step map alone, with that content frozen, rarely takes
this route from the same-direction start; in full training the content co-adapts). The two axes fail differently inside this frame. The non-negative step has no collective route. At rank D the
collective route exists, but for D >= 2 a channel can stop seeing the drift at no first-order cost by turning its
resolution away from the drift direction, and that lowers exactly the resolution that drives the collective removal --
a feedback that, we conjecture, keeps the drift in place (the gradient statements are proved; the dynamics are not
tested); in 1D there is no direction to turn to, and rank 1 learns the ring (7/8). One more latent dimension
gives a per-channel move that hides the clock off the map plane at no cost to the map.

### 5.2 Appendix material (for App. E, after P5 / Remark R)

- **E.x Clock and map.** Definitions of Sec. 1; Proposition S1 with proof; the cone-boundary readout (S4 check).
- **E.y The collective route.** Proposition S2 (a)-(d) with proofs; ACCOUNTS T2 as the causal test; latent end states.
- **E.z No collective route under |.|.** Proposition S3 with the omega-scaling proof; the reachable-share check; the
  remark that AO_bal is a point of a channel-tuned sheet.
- **E.w Same-direction moves.** Proposition S4; the GAUGE +1/+2 readouts (relation learned in 8/8 |.| runs; mirror copy in
  s1307; unsolved runs fail outside the step code). This replaces the main draft's "why |.| also struggles on +1/+2 is open"
  with "the |.| step code is learned on +1/+2; the remaining failures, shared with signed, lie elsewhere".
- **E.v Rank.** Proposition S6 (restating the rank theory's T1 as a route statement) and Proposition U, with the clock
  table of S5 and the explicit open problem.
- **E.u Basins.** The definitional statement of S8 and the measurements of Sec. 3.3-3.4.

Remark R of the appendix (the |.| side "open, fits loosely") can then say: the |.| failure from small starts is accounted
for at the level of routes (S1, S3) and end states (S5 table); the per-channel dynamics remain underived (per-channel flow
captures only near 2 pi, review `mech_capture_out.txt`).

---

## 6. Decisive GPU experiments (when GPUs return)

Costs from the batch logs: ring observation-step-free run (T = 128, 300 epochs) ~10-12 min on one GPU slot (GAUGE: 40 runs
in 1 h 40 min on two GPUs); torus T = 1024, 900 epochs ~45-60 min per run per slot (TORUS_START: 40 runs in 6 h 10 min on
two GPUs).

1. **R1 alone vs R1 + R2 on the torus: drift-forbidden rank 2** (tests S7's central claim). Torus T = 1024, 900 epochs,
   observation-step-free, per-head rank 2 with the inverse-pair latents tied (z_W = -z_E, z_S = -z_N: the clock forbidden by
   construction, as R1DF), against per-head rank 2 untied and rank 4 untied; n = 8 each, 24 runs, ~6 h on two GPUs.
   Prediction: tied rank 2 solves like rank 4 (>= 6/8) if the rank-2 failure is a failure to remove the clock; if tied
   rank 2 still fails, the rank axis has a second obstacle (the S6 account is incomplete).
2. **R2 only on the torus: frozen move latents** (completes the 2 x 2). Signed rank 4, torus, move latents frozen at the
   default and at the |.| start, n = 8 each, 16 runs, ~4 h. Prediction: fails like |.| (<= 2/8) from the |.| start.
3. **Parking (S7 prediction ii), eval of snapshots.** Rerun 4 rank-4 and 4 rank-2 torus seeds with checkpoints every 50
   epochs (8 runs, ~2 h incl. snapshots); eval-only readout of the clock's share outside span{map} over training.
   Prediction: rank 4 parks then removes; rank 2 cannot park and keeps it.
4. **A |.| class with a collective route** (tests S3 constructively). |.| applied after a learned sign-free reparameterisation
   that keeps an odd latent: e.g. steps |w . z_x| for x and the same with an inverse-pair tie z_y := -z_x (the clock is then
   omega (|w . z| + |w . z|) = 2 omega |w . z|, never zero): predicted to fail; vs a "signed-latent, |.|-readout" hybrid
   a(x) = omega |w| . z_x elementwise (readout non-negative, latent signed: collective route restored) predicted to succeed.
   Ring, n = 8 per arm, 16 runs, ~1.5 h. This separates "non-negative readout" from "no collective route".
5. **+1/+2 with the content examined.** GAUGE AO_p / SO_p at 900 epochs (16 runs, ~3 h) plus an eval-only read of what the
   unsolved runs miss (content: same-observation key retrieval; readout). Decides whether G1 is a budget effect.

Cheapest decisive item: 4 (1.5 h); most informative for the unification: 1 (6 h).

---

## 7. Reproduce

From `docs/paper/expansion/scripts/sign/` (CPU; paths are absolute): `python3 latent_geometry.py > latent_geometry_out.txt`
(~2 min); `python3 sign_clock.py > sign_clock_out.txt`; `python3 rank_clock.py > rank_clock_out.txt`;
`python3 latent_reach.py > latent_reach_out.txt`; `python3 aobal_geometry.py > aobal_geometry_out.txt`;
`python3 -u prop_checks.py > prop_checks_out.txt` (~10 min); `python3 lattice_p.py > lattice_p_out.txt` and
`python3 lattice_p.py 0.05 > lattice_p_out_tol005.txt`; `python3 path_profiles.py > path_profiles_out.txt` (~15 min, 4
threads); `./run_basin.sh` (56 step-only runs of 2000 steps, 10 at a time, ~3 h on a host at load ~80; the 16 runs of
A_seg_t0.25 / S_seg_t0.75 were dropped by writing `{"skipped": true}` placeholders, which the summary skips; pilot in
`runs_basin_pilot/`), then
`python3 basin_summary.py > basin_summary_out.txt`.
