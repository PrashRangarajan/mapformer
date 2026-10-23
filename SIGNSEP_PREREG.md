# SIGNSEP: is the |.| learnability cost the missing collective route, or the wrap barrier? -- pre-registration

Drafted 2026-10-23 on kalman, before any batch run; amended the same day by Amendment 1 (the independent results-blind
code check, rule 29, and its re-check of the amended code), Amendment 2 (tau exempt from weight decay) and Amendment 3
(C1's tau threshold), both from pilots; pilots in Sec. 9 (seeds outside the batch, never pooled). NOT launched: commit
first (Sec. 11). Code: `model_signsep.py`, `train_signsep.py`, `analyze_signsep.py` (`--selftest`), `construct_signsep.py` (existence and
account checks, output `runs/signsep_pilot/construct_signsep_out.txt` / `.json`), `test_signsep.py` (9/9),
`run_signsep.sh` (batch), `run_signsep_pilot.sh` (pilots).

## 1. Question

On the ring (32 cells, moves +1/-1) the non-negative step |.| solves 0-2/8 against the signed step's 8/8 (LRSWEEP, at each
step's best rate; SIGNROUTE in-batch at lr 1e-3 / 3e-3). SIGNROUTE's hybrid (readout matrix |W_out|, signed increment)
solved 16/16 and showed that the cost is carried by the non-negative INCREMENT, but its zero clock sits at k = 0, so it
removed both remaining accounts at once (SIGNROUTE Amendment 1, finding 2):

- **(A) Missing collective route** (`docs/paper/expansion/SEARCH_THEORY.md` S3; ACCOUNTS T2). The signed class zeroes
  every channel's clock k_c = a_c(+1) + a_c(-1) through r numbers (u = z_x + z_y = 0, S2). Under |.| that latent move ends
  at a(y) = a(x), a clock (S3(a)), and at a generic readout no latent choice wraps more than 2r of the C = 64 channels
  (S3(b)): cancellation must be found channel by channel. Stated so that it can fail: *from a small start, learning needs a
  direction that moves many channels' clocks at once; per-channel decoupling at each channel's own rate does not reach the
  map in the budget.*
- **(B) Wrap barrier** (S1, appendix P3, P4). A sign-coherent channel's only non-blind cancelling points are
  |k_c| = 2 pi n; any path to one from |k_c| < pi crosses |k_c| = pi (P3(d)), and the near side ends at the kink at zero
  (P4, the sticky wall). Stated so that it can fail: *a class whose cancelling targets lie across pi from the start does not
  reach them, route or no route; a class with near-side targets does, route or no route.*

SIGNSEP builds the two missing cells of the 2x2 and reads them against in-batch anchors from ONE common start.

## 2. Arms

| arm | step (angle a_c(x) = omega_c * step_c(x)) | collective route | barrier | role |
|---|---|---|---|---|
| signed (S) | w_c . z_x | yes (S2) | no | anchor (G0, G1) |
| abs (A) | \|w_c . z_x\| | no (S3) | yes | anchor (G1) |
| **abs_wrap (W)** | \|w_c . z_x + tau_x pi / \|omega_c\|\| | **yes (W2)** | **yes (W1)** | **NEW: route without removing the barrier** |
| **signed_frozen (F)** | w_c . z̄_x, z̄ frozen at the start | **no (F2)** | **no (F1)** | **NEW: no barrier without a route** |
| abs_wrap_past (P) | W's class, tau_x = tau_y = 0.75 at the start | yes | crossed at the start | control C1 |

tau_x, tau_y: one learnable scalar per move (zeros at init, so no RNG is consumed). In F, W_in gets no gradient and the
step path reads a buffer copy of the move embeddings (`model_signroute.freeze_move_latents`, ACCOUNTS T2's code path).

**The common start** (`model_signsep.build`). The seed's default init, then MECHANISM's balanced-gauge |.| start
(`train_wrap_mech.set_move_angles(m, |default angles|)`: move latents c e1, c e2, omega' = max(|omega|, 1),
W_out[:, :2] = |angle| / (omega' c), W_out[:, 2:] = 0). Every pre-|.| step is >= 0 there, so S, A, W (tau = 0) and F are
FUNCTION-IDENTICAL at init: logits bitwise equal at 3 seeds (`test_signsep.py`); S and A are bitwise MECHANISM's
SO_absmag / AO_absmag start (MECHANISM: 7/8 and 0/8 at lr 1e-3 on `environment_cancel`). P differs only by tau. This is
NOT the LRSWEEP / SIGNROUTE default start: a 2x2 needs one start for all four cells, W's route is clean only where the
pre-|.| steps are non-negative, and on the |.| class the default start and this one are the same angles (MECHANISM M3: no
difference detected). At the start (8 check seeds 2380-2387): median |k_c(0)| 0.21-0.57 rad, 0-12.5% of channels past pi
(`start_stats`); P: every channel past pi (min 4.73 rad), 94-100% in (pi, 3 pi), median 4.9-5.3.

Parameters (measured, vocab 19): S, A, F 204,243 (F trainable 203,731: W_in's 512 frozen); W, P 204,245 (+ tau).
(SIGNROUTE's prereg quoted 209,640 for the same SignedAO_r4; measured here it is 204,243 -- a note, not a change.)

## 3. Which account each arm removes (proofs; numerical checks in `construct_signsep.py`, seeds 2380-2387, init only)

Notation as SEARCH_THEORY Sec. 1: channel c, x = +1, y = -1, clock k_c = a_c(x) + a_c(y), map m_c = a_c(y),
u = z_x + z_y, T = tau_x + tau_y, P := {every pre-|.| step >= 0}.

**W1 (W keeps the barrier; proved).** For all parameters a_c(x) a_c(y) >= 0: both angles are omega_c times a
non-negative number. So S1 holds verbatim (its proof uses only this): |k_c| >= |m_c|, and the non-blind zero-clock points
are |k_c| in 2 pi Z_{>=1}. P3(d) holds verbatim: along any continuous path |k_c| = |a_c(x)| + |a_c(y)| is continuous, so a
path from |k_c| < pi to a non-blind cancelling point crosses |k_c| = pi (intermediate value theorem). The pre-|.| step
s = w . z + tau pi / |omega| has the kink of P4. []
*Check:* at random parameters (random W, z, tau of either sign, omega of either sign) 100% of channels are sign-coherent
and carry omega's sign; signed at the same draw is not (`test_abs_wrap_is_sign_coherent`).

**W2 (W has a collective route to the wrap; proved).** On P, a_c(x) = omega_c w_c . z_x + pi sign(omega_c) tau_x, so
  k_c = omega_c w_c . u + pi sign(omega_c) T,
linear in the r + 1 route coordinates (u, T).
(a) The affine set {u = 0, T = 2} within P cancels EVERY channel at once by wrapping, k_c = 2 pi sign(omega_c), for any
readout (W, omega), and it is non-blind (m_c = omega_c w_c . z_y + pi sign(omega_c) tau_y is free through z_y). This is
what S3(c) says a generic |.| readout lacks: the fixed column pi / |omega_c| (step units) is the wrap-target vector
2 pi n / omega_c with n = 1 in every channel.
(b) From the common start, the straight route z_y(t) = (1 - t) z_y0 - t z_x0, tau = (0, 2t) gives
k_c(t) = (1 - t) k_c(0) + 2 pi t sign(omega_c): every channel's distance to its wrap target shrinks in proportion,
monotonically, and every channel that starts below pi crosses pi. It stays in P iff |a_c(x)(0)| <= 2 pi (pre_y(1) =
-w z_x0 + 2 pi / |omega| >= 0). Its endpoint, (a_x0, 2 pi sign(omega) - a_x0), equals mod 2 pi the endpoint (a_x0, -a_x0)
of S's collective route from the same start (S2(b)): **W and S have routes to the same function; S's shrinks every clock
to 0, W's grows every clock to 2 pi through pi.**
(c) In the quadratic surrogate centred on the wrap, R = 1/2 sum_c q_c (k_c - 2 pi sign omega_c)^2 on P, the residual is
J [u; T - 2] with J = [diag(omega) W, pi sign(omega)]. For almost every (W, omega) J has full column rank (sign(omega) is
not in col(diag(omega) W) generically, as in S2(d)); at the common start W_out[:, 2:] = 0, so J has rank 3 and the
statement holds on (u_1, u_2, T) (Amendment 1 finding 9). Gradient flow converges to {u = 0, T = 2} from every start in
P, as S2(c).
The task's loss near the start is NOT this surrogate: it is periodic and its near well is k = 0 (W1), which is the barrier.
*Proof.* Substitution on P (|.| is the identity there); (b), (c) linear algebra as in S2. []
*Checks (8 seeds):* route formula = model code to <= 1.0e-6 rad at t = 0, 1/2, 1 (a measurement); along S's route every
|k_c| is non-increasing and reaches 0 (8/8 seeds); along W's route every channel's distance to 2 pi is non-increasing at
7/8 seeds (at seed 2387 one channel starts with a(x) = 6.63 > 2 pi, leaves P and ends 0.69 rad from the target), and the
route stays in P in 63-64 of 64 channels; at the route's end the clock equals 2 pi sign(omega) in every channel to 7e-7
rad, while abs at the same latents does not cancel (median |wrap k| 0.17; `test_collective_wrap_point`). Reach (share of
the clock change to the collective target reachable in the route coordinates' tangent space): **A 0.33-0.64** (latents
only, target 2 pi; S3) is the measurement; S's and W's 1.000 are identities (k0 = J u0; W's J contains the ones vector),
"every channel below pi crosses pi" follows from k(1) = 2 pi, and F's 0 is by definition (Amendment 1 finding 7).

**F1 (F has no barrier; proved).** F is signed: per channel, the straight segment of W_out[c, :] to the projection onto
{W[c, 0] + W[c, 1] = 0} (latents fixed and orthogonal, so (a_c(x), a_c(y)) = omega_c c (W[c, 0], W[c, 1]) are free
per-channel coordinates) keeps |k_c(t)| = (1 - t) |k_c(0)|: no crossing of pi, no kink (P3(d), signed half). []

**F2 (F has no collective direction; proved).** With W_in frozen and the step path reading a frozen buffer, k_c depends on
the trainable parameters only through (W_out[c, :], omega_c): the Jacobian of the clock vector is block-diagonal by
channel. []
*Checks:* trainable step-path scalars, token embedding included, that move >= 2 channels' clocks (autograd Jacobian):
**F 0**; S and A 512 (256 W_in entries feeding latent dims 1-2 -- dims 3-4 read zero W_out columns at the balanced start --
and the two move-embedding rows); W 514 (+ tau) (Amendment 1 finding 6). One AdamW step
leaves F's latents unchanged (0.0) and moves W_out; channel 5's clock depends on W_out row 5 only
(`test_frozen_has_no_collective_direction`). Note: in the quadratic clock surrogate F converges too (per channel, at rate
q_c omega_c^2 c^2): account A's content -- the pooling of the gradient across channels -- is not in the surrogate. ACCOUNTS
T2 is its causal precedent: from this start, freezing the latents gave 1/8 against 7/8 (Fisher p 0.010).

**The barrier operationally (analysis).** The same-cell phase surrogate (mean over 19,399 revisit pairs of the task and
the 64 channels of 1 - cos(phase); 0 = every revisit's phase matches) along the routes: **S's route descends from the
first step (slope -0.26 to -0.11, max rise 0); W's route first ascends (slope +3.9 to +11.7, max rise 0.14-0.40 at
t = 0.26-0.61) before descending to the same end value (0.09-0.14); P's route (tau_sum 1.5 -> 2) is flat to descending
(max rise <= 0.007).** This is a surrogate of the attractive term only (uniform channel weights, no content, no competing
keys); it shows the barrier exists on W's route, not that training cannot cross it.

So, as designed: **W keeps (B) and removes (A); F removes (B) and keeps (A) -- removes the route; P keeps W's class and
removes the barrier by its start.**

**What the manipulations also change (Amendment 1 findings 3, 15, 18; rule 15).** Per-channel clock change per unit
sign-step of each coordinate at the common start (seeds 2380-2382, the reviewer's measurement): W_in entries median 56-72
rad (max 740-980), a W_out row 0.9-2.2, a move-embedding row 0.27-0.52, omega_c 0.12-0.29, tau 6.28. So (i) freezing the
latents in F also removes the angles' fastest Adam coordinates, not only the collective direction: F A-LIKE licenses
"freezing the latent coordinates blocks learning from this start", and reads as account A only together with ACCOUNTS
T2's other arm (frozen latents from the default start, SO_same_fz: 7/8, i.e. the frozen class can learn when channels start
near their per-channel targets); (ii) W's added route (tau) is about 10x less mobile per channel than S's latent route, so
a W failure is read beside C1 (which now requires P's tau to move, Sec. 7). Weight decay acts on tau and pulls it toward
T = 0 (across pi from the sheet): lr wd tau = 5e-5 tau per step against an Adam step of ~lr, small; the global grad clip
(1.0) is active (grad norm at init 4.2-4.5, tau's share < 0.5%). The balanced start clamps omega' = max(|omega|, 1), so 30
of 64 channels sit at omega = 1 (ladder ratio 6.3, not 32), identically in every arm. The barrier binds only on channels
attention uses (in held |.| models 9-13 channels carry 92-98% of usage, SEARCH_THEORY S3).

## 4. The 2x2 logic

Per rate, after the gates (Sec. 7), W and F are each classed S-LIKE / A-LIKE / PARTIAL / UNMEASURED against the anchors.

| W (route, barrier) | F (no route, no barrier) | reading | licenses | does not license |
|---|---|---|---|---|
| S-LIKE (sheet) | A-LIKE | **ROUTE** (account A) | the route is what \|.\| lacks; with one, the barrier is crossed | "the barrier never matters" (one task, one start) |
| A-LIKE (BLOCKS) | S-LIKE | **BARRIER** (account B) | near-side targets suffice per channel; a route does not get past pi | "routes are irrelevant" |
| A-LIKE (BLOCKS) | A-LIKE | **BOTH** | each missing piece alone blocks learning from this start | a ranking of the two; for F, "pooling as such" (finding 3) |
| S-LIKE (sheet) | S-LIKE | **NEITHER ALONE** | either remedy suffices; \|.\| fails because it lacks both | either account as THE account |
| A-LIKE, P < 6/8 | any | **C1 FAILED** | nothing about B: W's class does not learn even from past pi | -- |
| A-LIKE, P >= 6/8 but P's tau did not move | any | **BARRIER OR PARAMETERISATION** | nothing about B (tau may be unusable, rule 15) | -- |
| A-LIKE with a diverged W run | any | **W DIVERGED** | nothing about B | -- |
| S-LIKE off the sheet | any | **W SOLVED OFF THE SHEET** | W learned, but not through the route it was built to add | ROUTE or NEITHER ALONE |
| otherwise | | PARTIAL / UNMEASURED | the half verdict of the definite arm, printed | a reading |

"S-LIKE (sheet)": the SOLVED W runs end on a collective wrap sheet (median |tau_sum| >= 1.5 and within 0.5 of 2 Z \ {0});
"A-LIKE (BLOCKS)": P >= 6/8, P's median final tau_sum >= 1.55 (moved toward the sheet from 1.5; Amendment 3), and no W
run diverged at that rate (Amendment 1 findings 1, 2, 10). The ROUTE row also does not license "pooling across channels as
such": F's manipulation removes the latents' Adam mobility as well (finding 3).

Half verdicts are registered and printed beside the reading: F A-LIKE = ROUTE NEEDED (no barrier is not enough); F S-LIKE =
ROUTE NOT NEEDED; W A-LIKE with C1 (amended) = BARRIER BLOCKS (a route is not enough); W S-LIKE on the sheet = BARRIER
CROSSED.

Prior evidence, other batches (not part of this test): F's cell is ACCOUNTS T2's SO_absmag_fz (1/8, `environment_cancel`,
lr 1e-3); MECHANISM's AO_high (|.| from past pi, no route) solved 8/8, so a start past pi suffices without a route on the
ring; nothing has measured W's cell.

## 5. Existence before mechanism (rule 14)

Source: a trained SignedAO_r4 on this recipe, LRSWEEP `runs/lrsweep/lr1e-03/SO_s1500/SO.pt` (acc 1.0000 at T = 128;
used as a source of a solution only, never as a comparison arm). `construct_signsep.py` (frozen, eval on the batch's
fresh-map stream, 400 trials):

| arm | construction | frozen acc (source 1.0000) | max angle err (rad) | max logit diff | weight scale (source: \|w_in\| 0.166, \|w_out\| 0.395, \|omega\| 2.81) |
|---|---|---|---|---|---|
| A | balanced P1(a) (`wrap_construct`) | 1.0000 | 5.4e-7 | 3.0e-4 | 0.211 / 1.42 / 2.81 |
| **W** | **on the collective wrap sheet**: latents kept, omega'_c = sign(A_c)\|omega_c\|, w'_c = sign(A_c omega_c) w_c, tau = (0, 2): every channel wraps through tau_y (raw clocks 6.19-6.43, all > pi), every pre-\|.\| step >= 0.007 | **1.0000** | 6.1e-7 | 2.8e-4 | 0.166 / 0.395 / 2.81 (= source) |
| F (source scale) | signed balanced (orthogonal latents c e1, c e2 with the source's c = 4.42), then frozen | 1.0000 | 2.4e-7 | 8.9e-5 | 0.211 / 0.446 / 2.81 |
| **F (F's scale)** | the same, latents rescaled to the norm F's frozen latents have at seed 2384's start (c = 0.99; batch seeds 0.42-1.60) and W_out by the inverse (Amendment 1 finding 5) | **1.0000** | 2.4e-7 | 7.7e-5 | 0.047 / **1.99** / 2.81 |
| P | same class as W | -- | -- | -- | -- |

Trainable (pilot warm starts, 300 epochs, lr 1e-3, data stream offset 10^6): Sec. 9 -- W from its sheet construction, F
at the source scale and at F's own scale. (A draft note here reported a loss jump after the first AdamW step of a warm
start; it was measured at constant peak lr, whereas the trainer warms up over 5% of the steps, and at the trainer's
first-step lr S, W and F all stay at 0.0000 (reviewer). Withdrawn, Amendment 1 finding 11.)

## 6. Batch

Ring 32 cells, moves +1/-1 (`GridWorldSteps`, p_first 0.5), the LRSWEEP / SIGNROUTE recipe verbatim: T = 128, batch 128,
98 batches, 300 epochs, cosine, AdamW wd 0.05, 1 layer, 2 heads, d 128, rank 4, 3 data workers; evaluation
`train_cancel.evaluate`, fresh stream (np seed 10^6, env seed 10000, maps redrawn), 400 trials, T = 128 (primary) and 512.
**5 arms x 2 rates (1e-3, 3e-3) x 8 fresh seeds 2300-2307 = 80 runs.** Seeds: no `_s23xx` run dir and no `23xx` seed
range anywhere in the repo (grep, 2026-10-23). (The first choice, 2100-2107 / pilots 2180-2187, collided with the
concurrent SSM_SIGN session's registered seeds 2100-2107 and its running pilots 2180-2183, and was moved before any
SIGNSEP run.) Pilots 2380-2387, never pooled.
Host kalman, 2x RTX 4090, torch 2.10.0+cu128. GPU policy (user, 2026-10-23): both GPUs carry another user's jobs; ours
run beside them at <= 4 per GPU (`DRV_MAXPG=4`; lib_driver counts every `mapformer.train_*` process of this user on a
device, so the SSM_SIGN batch and this one share the budget). The driver refuses modified tracked files and any
uncommitted SIGNSEP file, checks its directory tags against `analyze_signsep.rate_tag`, md5-guards every imported
mapformer module plus driver, lib and this file before training and before the analysis, requires all 160 artifacts
before the analysis, and sets `.signsep_done` only after `SIGNSEP_ANALYSIS.txt` and `SIGNSEP_VERDICTS.json` exist.

## 7. Registered analysis (`analyze_signsep.py`)

SOLVED = `stats_core.classify_run` (final-5% training loss < 0.05); a run with any non-finite loss is DIVERGED, never
SOLVED, accuracy 0; recipe and len(losses) = 300 asserted per run. better(a, b) = (Fisher < .05 and k_a > k_b) or
(permutation p < .05 and acc_a - acc_b >= +0.05) (`analyze_signroute.better`). Per rate, n = 8:

- **G0** k_S >= 6, else UNREADABLE at that rate.
- **G1** k_A <= 3 and better(S, A), else NO COST TO EXPLAIN at that rate.
- **Class of X in {W, F}:** S-LIKE = better(X, A) and not better(S, X) and k_X >= k_S - 1; A-LIKE = better(S, X) and not
  better(X, A) and k_X <= k_A + 1; PARTIAL = better(X, A) or better(S, X) but neither; UNMEASURED otherwise.
- **C1** (amended) k_P >= 6 and median final tau_sum of P >= 1.55 (Amendment 3; 1.75 in Amendment 1).
- **Sheet use** (amended) W's SOLVED runs: median |tau_sum| >= 1.5 and median distance of |tau_sum| to 2 Z \ {0} <= 0.5.
- **Reading** per Sec. 4 (amended rows). **Overall**: UNREADABLE (no rate passes G0 and G1); MIXED if W or F is S-LIKE at
  one readable rate and A-LIKE at the other (amended, finding 4); the reading if both rates give the same definite reading;
  MIXED if two different definite readings; "<reading> AT ONE RATE" if exactly one readable rate is definite; else C1
  FAILED, BARRIER OR PARAMETERISATION, W DIVERGED, W SOLVED OFF THE SHEET, PARTIAL, UNMEASURED in that precedence. Scope
  "at lr 1e-3 and 3e-3", never "at best rates".
- The classes use SOLVED counts plus `better`'s accuracy arm; there is no separate accuracy tolerance as in SIGNROUTE's
  "full" (finding 14): an arm at 1/8 whose unsolved runs reach 0.99 reads PARTIAL through the accuracy test (checked), and
  at k_A = 3 an S-LIKE reading on SOLVED alone needs 8/8 (7 vs 3, p 0.119) unless the accuracy test fires.
- Diverged runs per arm x rate are printed; a W A-LIKE with any diverged W run reads W DIVERGED (finding 10).
- **Descriptive (registered as such):** medians per arm x rate of u_ratio, cos(z_x, z_y), wrapped rms clock, cancelling
  share split through-zero / by-wrap, tau_sum, pre-|.| < 0 share, start median |k|; tau_sum per W / P seed (did solved W
  runs use the collective sheet |tau_sum| ~ 2, or the per-channel route?); r(log10 final loss, acc) at T and 4T (rule 2);
  per-seed accuracy and run class; floors blank 0.50, n-gram 0.554, retrace 0.860.

**Power (rule 5; Fisher two-sided, `stats_core.fisher_solved`, n = 8 vs 8).** 8 vs 0 p .0002, 8 vs 1 .0014, 8 vs 2 .007,
8 vs 3 .026, 8 vs 4 .077; 7 vs 0 .0014, 7 vs 1 .010, 7 vs 2 .041, 7 vs 3 .119; 6 vs 0 .007, 6 vs 1 .041, 6 vs 2 .132.
So with S at 7-8 and A at 0-1: W or F reads S-LIKE at >= k_S - 1 (6-7 of 8), A-LIKE at <= k_A + 1 (1-2 of 8). An arm at
3-5 of 8 reads PARTIAL or UNMEASURED on the SOLVED test alone (the accuracy test can still fire at d >= 0.05) -- by
design: it is evidence for neither account's sharp form. If A reaches 2-3 (G1 still passes), A-LIKE allows up to 3-4 but
needs better(S, X), which on SOLVED requires X <= 3 at S = 8. The "or" of two tests inflates alpha somewhat (house rule,
as SIGNROUTE). Seeds are paired by design (same init and data order per seed across arms); the tests are unpaired
(conservative). G0 risk: at a true 7/8 the chance of k_S < 6 is 7% per rate.

## 8. Predictions (stated before the batch)

- Anchors: S >= 6/8 at both rates (MECHANISM SO_absmag 7/8 at 1e-3; 3e-3 unmeasured from this start); A <= 2/8.
- F: **A-LIKE** at 1e-3 (T2's 1/8 from this start); no prior at 3e-3.
- W: **A-LIKE** -- the route has to climb against the gradient (W's route rises 0.14-0.40 in the surrogate before it
  descends; the loss near the start pulls every clock toward the near well at 0, i.e. the wall), and P: **S-LIKE** (>= 6/8;
  MECHANISM AO_high 8/8 from past pi). Low confidence: tau is two scalars with leverage on all 64 channels, and a
  coordinated climb may be easier than 64 separate ones.
- C1: P >= 6/8 with its tau moving toward the sheet (median >= 1.55, Amendment 3).
- Predicted reading: **BOTH** (each missing piece alone blocks). ROUTE is the main alternative (W S-LIKE). These
  predictions were written before the pilots; the pilots (Sec. 9) agree on one seed per cell and change none of them.

## 9. Pilots (seeds 2380-2387, outside the batch; never pooled; one GPU beside the other user's jobs, <= 4 of ours)

Wave 1 (`run_signsep_pilot.sh`, `runs/signsep_pilot/`; the abs_wrap arms here still had tau under weight decay, i.e.
BEFORE Amendment 2). Wave 2 (`run_signsep_pilot2.sh`, `runs/signsep_pilot2/`): the abs_wrap pilots rerun on the same
seeds with tau exempt from decay (the registered setting). Accuracy on fresh maps at T = 128 / 512; class by
`classify_run`; cancel = share of channels with |wrap a(x)| > 0.05 and |wrap k| < 0.1, split through zero (raw |k| < pi;
in a |.| class these are near-blind one-sided channels, not cancellation) / by wrap.

| wave | seed | arm, lr | acc 128 | acc 512 | final loss | class | tau_sum end (start) | u_ratio, cos(z_x, z_y) | cancel (zero / wrap) |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 2380 | W, 1e-3 (tau decayed) | 0.931 | 0.900 | 0.356 | DESCENDING | -0.00 (0) | 0.42, -0.66 | 0.00 |
| 1 | 2381 | F, 1e-3 | 0.908 | 0.867 | 0.345 | STALLED | -- | 0.71, 0.00 (frozen) | 0.12 / 0 |
| 1 | 2382 | P, 1e-3 (tau decayed) | 0.969 | 0.963 | 0.087 | DESCENDING | 0.89 (1.5) | 0.81, +0.03 | 0 / 0.16 |
| 1 | 2383 | W warm from the sheet (tau decayed) | 1.000 | 1.000 | 0.000 | SOLVED | 1.60 (2.0) | 0.72, -0.04 | 0 / 0.17 |
| 1 | 2384 | F warm, source scale (c 4.42) | 1.000 | 0.995 | 0.000 | SOLVED | -- | -- | 0.45 / 0 |
| 1 | 2384 | F warm, F's scale (c 0.99) | 1.000 | 0.997 | 0.000 | SOLVED | -- | -- | 0.79 / 0 |
| 1 | 2385 | S, 3e-3 | 1.000 | 1.000 | 0.000 | SOLVED | -- | 0.001, -1.000 | 1.00 / 0 |
| 1 | 2386 | W, 3e-3 (tau decayed) | 0.949 | 0.950 | 0.214 | STALLED | 0.36 (0) | 0.76, +0.14 | 0.00 |
| 1 | 2387 | F, 3e-3 | 0.932 | 0.779 | 0.257 | STALLED | -- | 0.71, 0.00 (frozen) | 0.06 / 0 |
| **2** | 2380 | **W, 1e-3** | 0.932 | 0.917 | 0.322 | STALLED | **0.01** (0) | 0.62, -0.29 | 0.06 / 0 |
| **2** | 2382 | **P, 1e-3** | **1.000** | 0.998 | 0.002 | **SOLVED** | **1.62** (1.5) | 0.50, -0.50 | 0 / 0.12 |
| **2** | 2383 | **W warm from the sheet** | **1.000** | 1.000 | 0.000 | **SOLVED** | **2.00** (2.0; tau = 0.07, 1.93) | 0.023, -0.999 | 0 / **0.94** |
| **2** | 2386 | **W, 3e-3** | 0.962 | 0.785 | 0.161 | STALLED | 0.54 (0) | 0.76, +0.15 | 0.02 / 0 |

Reading of the pilots (anecdote, one seed per condition; nothing here is evidence for the batch's verdict):
- **Rule 14, trainable, holds for both new classes.** W warm-started on the collective sheet stays solved for 300 epochs and,
  with tau exempt from decay, stays ON the sheet (tau_sum 2.00, antipodal latents u 0.023, 94% of channels cancelling by
  wrap) -- the collective wrapped solution is held, not only re-found. With decay (wave 1) it stayed solved but slid off
  the sheet (tau_sum 1.60; the per-channel readout took the wrap over): this is what prompted Amendment 2. F warm-started
  holds at the source scale and at its own start scale (c 0.99, |w_out| 1.99; Amendment 1 finding 5).
- **From the common start W did not use its route** at either rate or decay setting: tau_sum ends at -0.00 / 0.01 (1e-3)
  and 0.36 / 0.54 (3e-3), the runs stall at 0.93-0.96 with no wrapped channels; F stalls at 0.91 / 0.93; S at 3e-3 solves
  (antipodal latents, every channel through zero). This is the BOTH pattern of Sec. 4, predicted in Sec. 8 -- on one seed.
- **C1:** P solves from past pi (1.000) and with decay off its tau moves TOWARD the sheet (1.50 -> 1.62), with decay on
  away from it (-> 0.89). Amendment 1's threshold (median >= 1.75) would have read this pilot's tau as unused; Amendment 3.
- W's failure is not divergence: no pilot run diverged.


## 10. Wall time

Measured on kalman beside the other user's jobs: 5.1-5.2 s/epoch with 4 of our ring runs on one GPU (6.0 s with 4-5),
i.e. ~26.5 min per run including the evaluation (launch to `DONE`: 26.8 min, s2385). The batch is 80 runs at <= 4 per GPU:
- both GPUs free of our other jobs: 8 concurrent, 10 waves x ~27 min = **~4.5 h** (seed-major order, so a partial batch
  is balanced);
- one GPU (`DRV_GPUS="1"`), or both GPUs shared half-and-half with another batch of ours (lib_driver counts every
  `mapformer.train_*` process against the same 4-per-GPU budget): 20 waves = **~9 h**.
GPU 0's per-epoch time beside the other user was not measured (the pilots ran on GPU 1 only); assume it is the same.
Disk: ~0.83 MB per checkpoint plus eval.json and log, ~80 MB for the batch.


## 11. Before launch

1. Fill Sec. 9 and 10 FIRST (this file is md5-guarded by the batch driver; editing it after launch aborts the analysis).
2. Commit the SIGNSEP files (the driver refuses otherwise). 3. `cd /home/prashr/mapformer && setsid nohup bash
run_signsep.sh > /dev/null 2>&1 &`.

## Amendment 1 -- independent results-blind code check (rule 29), 2026-10-23

Reviewer (a separate agent, blind to every pilot run, log and eval.json) read this file, SEARCH_THEORY, the appendix, the
precedents and all SIGNSEP code and imports; ran `test_signsep` (8/8 then), the analysis selftest, the existence and account
checks, a dry copy of the driver, and scratch scripts. **Verified:** S, A, W (tau = 0), F bitwise-identical logits at init
in eval AND train mode (same dropout RNG) at the batch seeds 2300-2307 and at 2390-2392, P differing only by tau; at the
batch seeds every pre-|.| step is positive (min 2e-4 to 1e-2), so S, A, W take identical first gradients except for tau;
S / A bitwise MECHANISM's SO_absmag / AO_absmag start; parameter counts; F's W_in gets no grad or decay and the step path's
leverage of the move embedding is 0; the Fisher table; existence and all quoted init numbers reproduce; the recipe equals
`train_signroute`; proofs W1, W2(a)(b), F1, F2 correct; no path for W to a non-blind cancelling code avoids |k| = pi
(|a_c| is continuous in every parameter, omega crossing 0 and the clamp included; negative tau only reaches the mirror
sheet tau_sum = -2, also across pi; the exemptions are channels already past pi at the start, 0-12.5%, as for A); driver
ordering, tag check, md5 coverage (104 files), 80 launches seed-major, 160 artifacts, fail-closed marker, rules 23 / 27.

| # | sev | finding | disposition |
|---|---|---|---|
| 1 | MAJOR | C1 = k_P >= 6 cannot detect an unusable tau: P starts in MECHANISM AO_high's regime, where \|.\| learns with no route | FIXED: C1 also needs P's median final tau_sum >= 1.75 (1.55 after Amendment 3); else a W A-LIKE reads BARRIER OR PARAMETERISATION |
| 2 | MAJOR | ROUTE / NEITHER never check that W used the route (W = A + two scalars) | FIXED: S-LIKE W must end on a sheet (Sec. 7); else W SOLVED OFF THE SHEET (not definite) |
| 3 | MAJOR | Adam-leverage confound on both route cells (freezing removes the fastest coordinates; tau ~10x less mobile per channel than the latents) | ACCEPTED: numbers in Sec. 3, "does not license" column amended; F's reading leans on T2's SO_same_fz 7/8 |
| 4 | MAJOR | "<reading> AT ONE RATE" printed when the other rate contradicts a half verdict | FIXED: MIXED if W or F is S-LIKE at one rate and A-LIKE at the other; two selftest cases |
| 5 | MAJOR | F existence not at F's training weight scale (source c 4.42 vs F's 0.42-1.60) | FIXED: `--frozen-c-seed`; F built at seed 2384's c = 0.99 (\|w_out\| 1.99), frozen 1.0000; its warm start added to the pilot |
| 6 | MINOR | collective-parameter count omitted the move-embedding rows | FIXED: S / A 512, W 514, F 0 |
| 7 | MINOR | reach for S and W and F's 0 cannot fail | FIXED: labelled identities in the code and Sec. 3 |
| 8 | MINOR | W monotone check printed "all False" for 7/8 | FIXED: counts printed; Sec. 3 says 7/8 and names the channel |
| 9 | MINOR | J rank 3 of 5 at the start | FIXED: W2(c) reworded |
| 10 | MINOR | diverged runs count as accuracy 0 in `better` | FIXED: printed per arm; W A-LIKE with a diverged W run reads W DIVERGED |
| 11 | MINOR | first-AdamW-step note measured at constant lr, not the trainer's warmup | FIXED: withdrawn (Sec. 5) |
| 12 | MINOR | stale parameter count in `model_signsep.py` | FIXED |
| 13 | NOTE | selftest comment said one-sided Fisher | FIXED (two-sided) |
| 14 | NOTE | no accuracy tolerance in the classes | recorded in Sec. 7 |
| 15 | NOTE | weight decay on tau; grad clip | recorded in Sec. 3 |
| 16 | NOTE | this file is md5-guarded | Sec. 11 step 1 |
| 17 | NOTE | pilot driver md5 list missed model_sign.py, data_parallel.py | FIXED for the restarted pilot driver |
| 18 | NOTE | omega clamp (30/64 channels at 1); barrier binds only on used channels | recorded in Sec. 2 / 3 |

Pilot operations noted here: the first pilot driver was stopped after one launch (abs_wrap s2380, which kept running) to
apply this amendment; `model_signsep.py` changed only in its docstring and `construct_signsep.py` gained the F-scale
construction while s2380 ran (its trainer had already imported the code; the first md5 record is kept as
`runs/signsep_pilot/code_md5_first_launch.txt`). The restarted pilot driver uses a second lock file because the first is
held by the s2380 trainer, which inherited the driver's flock fd 9 (lib_driver behaviour: a stopped driver cannot be
restarted on the same lock until its children exit); its lock line lost `|| exit 1` to a trailing comment (harmless: the
lock was acquired) -- fixed after the pilot, not during, so the running driver's md5 guard still matches.

## Amendment 2 -- tau is exempt from weight decay (2026-10-23, from the pilots, before the batch)

Pilot evidence (Sec. 9, wave 1, tau under the recipe's AdamW decay 0.05): from the past-pi start (P, s2382) tau_sum
FELL from 1.50 to 0.89 over 300 epochs, and from the sheet construction (W warm, s2383) from 2.00 to 1.60 while the model
stayed solved (the per-channel readout took the wrap over). Pure decoupled decay alone would take tau to exp(-sum_t lr_t
wd) = 0.44 of its start over this schedule (sum of lr over 29,400 cosine steps ~16.2, x 0.05 = 0.81); the gradient on tau
is weak on the plateau, so decay is a systematic pull of the collective coordinate toward T = 0 -- the NEAR side of the
barrier -- that S's route coordinates do not suffer (S's route end u = 0 is compatible with decay). Under the amended C1
this would make W's cell unreadable by construction. **Registered change:** in the abs_wrap arms tau is exempt from
weight decay (`train_signsep --tau-decay off`, the default; implemented by `exempt_from_decay`, which undoes AdamW's
decoupled decay on tau after each step, exact up to float32 rounding (~1e-6): `test_tau_decay_exemption_is_exact`, hooked = a wd-0 parameter group to
1e-5, against a decay effect of 3.5e-3 in 5 steps; end to end through `train()` 1.7e-6 on tau, reviewer). Re-checked
results-blind by the Amendment 1 reviewer: no BLOCKER / MAJOR; M1 (the test's all-parameter threshold barely caught a
hook that undid decay everywhere) fixed by tightening it to 3e-5; M2 (ROUTE row lacked "(sheet)") fixed; N1 ("exactly" ->
to float rounding) fixed here, and in the trainer's docstring after the pilot (it is md5-guarded by the running pilot). Every other parameter of every arm keeps wd 0.05; the analysis asserts
`tau_decay == "off"` on every run. The W pilots were rerun on the same seeds under the change (`run_signsep_pilot2.sh`,
`runs/signsep_pilot2/`), so each pairs with its decay-on run.

## Amendment 3 -- C1's tau threshold (2026-10-23, from the wave-2 pilot, before the batch)

With tau exempt from decay, the P pilot (s2382) SOLVED and moved tau_sum from 1.50 to 1.62 in 300 epochs (with decay: to
0.89): P's start is on the loss plateau, so tau moves slowly even when it moves the right way. Amendment 1's threshold
(median >= 1.75, "halfway to the sheet") would have read this run as "tau unused" and made a W failure unreadable by a
criterion the control could not meet in the budget. **Registered change:** C1's tau condition is median final tau_sum of
P >= 1.55 -- tau moved toward the sheet -- which separates the two pilot settings (1.62 vs 0.89) and still fails for a
tau that does not move or decays. Analysis, selftest and Sec. 4 / 7 updated; nothing else changes. Disclosed as a
pilot-informed threshold.
