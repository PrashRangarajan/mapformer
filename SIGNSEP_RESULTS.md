# Is the |.| learnability cost the missing collective route or the wrap barrier? -- results (SIGNSEP, registered, `SIGNSEP_PREREG.md` + Amendments 1-3)

Batch 2026-10-23 22:03 -> 10-24 06:48 on kalman, 80 runs: five observation-step-free rank-4 arms x lr 1e-3, 3e-3 x fresh
seeds 2300-2307, ring 32 cells, moves +1/-1, the LRSWEEP / SIGNROUTE recipe verbatim (T = 128, batch 128, 98 batches, 300
epochs, cosine, AdamW wd 0.05; tau exempt from decay, Amendment 2), every arm from ONE common start (MECHANISM's
balanced-gauge |.| start, at which signed, |.|, W and F are function-identical). Code `ffa34cd`; the md5 guard over 104 files
passed at every launch and matches the tree today (`runs/signsep/p0/code_md5.txt`); torch 2.10.0+cu128. Independent
results-blind check before launch (Amendment 1: 5 MAJOR, 7 MINOR, 6 NOTE, folded in; re-checked after Amendment 2).
Analysis `analyze_signsep.py` -> `SIGNSEP_ANALYSIS.txt`, `SIGNSEP_VERDICTS.json` (written by the driver before
`.signsep_done`). Pilots (seeds 2380-2387) are in the prereg Sec. 9 and are not pooled. Floors (ring 32, +1/-1): blank
0.50, n-gram 0.554, retrace 0.860.

**Operations.** The first supervisor (commit `0367dec`, clean tree) ran on GPU 0 from 22:03. Another user's jobs moved onto
GPU 0, so at 00:38 the supervisor was restarted pinned to GPU 1 (commit `fb7a028`, clean; the same 104 md5s); the driver
skipped the 20 finished and 4 running runs, and the first supervisor's lock file is kept as
`.run_signsep.lock.stale_20261024` (a restart attempt at 00:37 was refused while that lock was held). 24 runs trained on GPU
0 (seeds 2300-2301, four runs of 2302), 56 on GPU 1; 80 launches in `signsep.log`, none duplicated; every run log has one
`DONE`. Both GPUs are RTX 4090s; device placement is not a registered factor.

**Reading (registered): UNMEASURED.** At lr 1e-3 the 2x2 is UNREADABLE: the signed anchor solved 5/8, below the G0 gate
(>= 6). At lr 3e-3 the gates pass and the four cells read: **W (|.| with a collective route to the wrap, barrier kept)
A-LIKE, 0/8** against signed 8/8; **F (signed with frozen move latents: no barrier, no collective route) UNMEASURED, 4/8**;
control C1 passed (P, W's class started past pi, 7/8, median final tau_sum 2.00). The Sec. 4 table has no definite row for
(A-LIKE, UNMEASURED), so the rate reads UNMEASURED, and with no definite rate the overall reading is UNMEASURED. Registered
half-readings at 3e-3: **barrier "BARRIER BLOCKS"** (W A-LIKE with C1 passed: a route is not enough), **route UNMEASURED**
(F neither S-LIKE nor A-LIKE). Scope (registered): ring 32, +1/-1, observation-step-free rank 4, the common |.| start, 300
epochs, at lr 1e-3 and 3e-3; ROUTE / BARRIER / BOTH / NEITHER ALONE are statements about these two constructed classes
against the two anchors, nothing wider. Predicted (Sec. 8): BOTH; not reached.

## Arms (prereg Sec. 2)

| arm | step | collective route | wrap barrier | role |
|---|---|---|---|---|
| signed (S) | w_c . z_x | yes | no | anchor (G0, G1) |
| abs (A) | \|w_c . z_x\| | no | yes | anchor (G1) |
| abs_wrap (W) | \|w_c . z_x + tau_x pi / \|omega_c\|\|, tau = 0 at the start | yes: u = 0, tau_x + tau_y = 2 cancels every channel by wrapping | yes | new cell |
| signed_frozen (F) | w_c . z̄_x, move latents frozen | no | no | new cell |
| abs_wrap_past (P) | W's class, tau_x = tau_y = 0.75 at the start (every channel past pi) | yes | crossed at the start | control C1 |

## Arm table per rate (n = 8 per cell; none diverged)

Mean held-out accuracy on fresh maps (400 trials; T = 128 registered, T = 512 descriptive). Classes S / T / D = SOLVED /
STALLED / DESCENDING (`stats_core.classify_run`, final-5% training loss < 0.05 for SOLVED).

| lr | arm | SOLVED | S / T / D | acc T = 128 | acc T = 512 | final loss, median |
|---|---|---|---|---|---|---|
| 1e-3 | signed | **5/8** | 5 / 3 / 0 | 0.981 | 0.971 | 0.006 |
| 1e-3 | abs | 0/8 | 0 / 8 / 0 | 0.930 | 0.823 | 0.224 |
| 1e-3 | abs_wrap (W) | 2/8 | 2 / 6 / 0 | 0.932 | 0.826 | 0.286 |
| 1e-3 | signed_frozen (F) | 3/8 | 3 / 5 / 0 | 0.939 | 0.902 | 0.245 |
| 1e-3 | abs_wrap_past (P) | 5/8 | 5 / 3 / 0 | 0.960 | 0.958 | 0.001 |
| 3e-3 | signed | **8/8** | 8 / 0 / 0 | 1.000 | 0.994 | 0.000 |
| 3e-3 | abs | 3/8 | 3 / 4 / 1 | 0.973 | 0.930 | 0.088 |
| 3e-3 | abs_wrap (W) | **0/8** | 0 / 7 / 1 | 0.953 | 0.889 | 0.172 |
| 3e-3 | signed_frozen (F) | **4/8** | 4 / 3 / 1 | 0.971 | 0.974 | 0.088 |
| 3e-3 | abs_wrap_past (P) | **7/8** | 7 / 0 / 1 | 0.985 | 0.973 | 0.000 |

## Registered decisions (lr 3e-3; better(a, b) = Fisher < .05 with k_a > k_b, or permutation p < .05 with d >= +0.05)

| decision | numbers | outcome |
|---|---|---|
| G0: k_S >= 6 | 8/8 | pass |
| G1: k_A <= 3 and better(S, A) | 3/8; S vs A Fisher p 0.026, perm p 0.026, d +0.027 | pass (k_A at the bound; the SOLVED test fires) |
| W class | better(S, W): Fisher p 0.0002, perm p 0.0002, d +0.048 -- true; better(W, A): Fisher p 0.20, perm p 0.10, d -0.021 -- false; k_W 0 <= k_A + 1 | **A-LIKE** |
| F class | better(F, A): Fisher p 1.0, perm p 0.88, d -0.002 -- false; better(S, F): Fisher p 0.077, perm p 0.026 but d +0.029 < 0.05 -- false | **UNMEASURED** |
| C1: k_P >= 6 and median final tau_sum of P >= 1.55 | 7/8; median 1.999 | pass |
| W diverged runs | 0 | -- |
| sheet use (W's SOLVED runs) | no W run solved at 3e-3 | not applicable |
| half-reading, barrier | W A-LIKE, C1 passed, no divergence | **BARRIER BLOCKS** |
| half-reading, route | F UNMEASURED | **UNMEASURED** |
| reading at 3e-3 | (A-LIKE, UNMEASURED): no definite Sec. 4 row | UNMEASURED |
| reading at 1e-3 | G0 fails (5/8 < 6) | UNREADABLE |
| overall | no definite rate | **UNMEASURED** |

The analysis was re-checked against the prereg: every gate, class rule, C1 threshold (Amendment 3), sheet rule, half-verdict
and the overall precedence (Amendment 1 finding 4) are implemented as registered; the verdicts JSON and the text agree.
No discrepancy found.

## What "BARRIER BLOCKS" with route UNMEASURED licenses (prereg Sec. 4)

- **Licensed (half-reading, lr 3e-3 only):** *a route is not enough.* A |.| class that has a collective route to a point
  where every channel cancels by wrapping (W2: u = 0, tau_x + tau_y = 2, any readout) does not learn the ring map from the
  common small start: 0/8 against signed 8/8. The control rules out the two readings Amendment 1 named: the same class
  learns from past pi (P 7/8) and its tau moves onto the collective sheet there (median 2.00), so tau is usable and the
  class is not broken (C1); and no W run diverged.
- **Not licensed:** the BARRIER reading of the 2x2 (account B as THE account), which needs F S-LIKE as well -- whether
  removing the barrier without a route suffices is unmeasured (F 4/8, between the anchors). Nor "routes are irrelevant"
  (the BARRIER row's own exclusion), nor anything at lr 1e-3, nor anything beyond this start, task and budget.
- **Amendment 1 finding 3 still applies to W:** tau is about 10x less mobile per channel per Adam step than the latents.
  C1 is the registered answer to that objection (tau moved fully to the sheet in P at 3e-3); it does not make W's and S's
  route coordinates equally fast.

## The signed anchor at 1e-3 (G0 failed)

Signed solved 5/8 at 1e-3 (unsolved: 0.968, 0.925, 0.956, all STALLED, not descending) and 8/8 at 3e-3. Earlier signed
observation-step-free ring results at 1e-3 (other batches, never pooled into this test):

| batch | start | seeds | signed at 1e-3 |
|---|---|---|---|
| LRSWEEP | default (small, signs independent) | 1500-1507 | 8/8, 1.000 |
| SIGNROUTE | default | 1800-1807 | 8/8, 1.000 |
| MECHANISM (SO_absmag) | the \|.\| start, balanced gauge (bitwise this batch's start construction) | 800-807 | 7/8, 0.983 |
| ACCOUNTS T2 (SO_absmag) | the same | 1000-1007 | 7/8, 0.985 |
| **SIGNSEP** | the same (the common start) | 2300-2307 | **5/8, 0.981** |

What differs from LRSWEEP / SIGNROUTE is the start: a 2x2 needs one start for all four cells, and W's route is clean only
where every pre-|.| step is non-negative, so the anchors start where both moves rotate the same way (with omega clamped at
>= 1 in 30 of 64 channels). From that start the signed step has lost runs in all three batches that used it (19/24 pooled,
against 16/16 from the default start at 1e-3 in LRSWEEP and SIGNROUTE). Task generator, recipe and start construction match
MECHANISM and ACCOUNTS (`GridWorldSteps` with +1/-1 is `GridWorldCancel`'s generator); the seeds differ. 5/8 against the
earlier 14/16 from the same start: Fisher p 0.29 (post hoc, across batches; descriptive only). The prereg put the chance of
k_S < 6 at a true 7/8 at 7% per rate. So the 1e-3 failure reads as seed variation on top of a start that costs the signed
step something at 1e-3, not as a broken anchor; at 3e-3 the same start costs it nothing (8/8). The prereg predicted
S >= 6/8 at both rates.

## Descriptive diagnostics (registered as such; medians per arm x rate)

u_ratio = |z_x + z_y| / (|z_x| + |z_y|) of the move latents; cancel = share of channels with |wrap a(x)| > 0.05 and
|wrap k| < 0.1, split through zero (raw |k| < pi) / by wrap. In a |.| class the "through zero" share counts near-blind
one-sided channels, not cancellation (prereg Sec. 9). k0 = median |k_c| at the start.

| lr | arm | u_ratio | cos(z_x, z_y) | clock rms | cancel | zero | wrap | tau_sum | pre-\|.\| < 0 | k0 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1e-3 | signed | 0.304 | -0.738 | 0.661 | 0.504 | 0.504 | 0.000 | -- | -- | 0.516 |
| 1e-3 | abs | 0.730 | +0.058 | 1.272 | 0.027 | 0.027 | 0.000 | -- | -- | 0.516 |
| 1e-3 | W | 0.744 | +0.104 | 1.276 | 0.043 | 0.041 | 0.000 | 0.062 | 0.113 | 0.516 |
| 1e-3 | F | 0.707 | -0.000 | 1.157 | 0.080 | 0.080 | 0.000 | -- | -- | 0.516 |
| 1e-3 | P | 0.606 | -0.279 | 0.939 | 0.188 | 0.000 | 0.188 | 1.684 | 0.000 | 5.228 |
| 3e-3 | signed | 0.002 | -1.000 | 0.001 | 1.000 | 1.000 | 0.000 | -- | -- | 0.516 |
| 3e-3 | abs | 0.878 | +0.536 | 1.228 | 0.051 | 0.032 | 0.000 | -- | -- | 0.516 |
| 3e-3 | W | 0.782 | +0.210 | 1.653 | 0.016 | 0.000 | 0.000 | 0.547 | 0.059 | 0.516 |
| 3e-3 | F | 0.707 | -0.000 | 0.962 | 0.182 | 0.170 | 0.013 | -- | -- | 0.516 |
| 3e-3 | P | 0.036 | -0.997 | 0.011 | 1.000 | 0.000 | 1.000 | 1.999 | 0.000 | 5.228 |

tau_sum at the end per seed (2300-2307; S = SOLVED; the collective wrap sheet is tau_sum = +-2):

```
1e-3 W   +0.07  +0.04  +0.09  +0.06S +0.11  +0.04  +0.11S +0.01
1e-3 P   +1.68  +1.69S +1.69S +1.63  +1.68S +1.69S +1.71  +1.66S
3e-3 W   +0.37  +0.17  +0.55  +0.55  +0.55  +0.63  +0.70  +0.13
3e-3 P   +2.00S +2.00S +2.00S +2.00S +2.00S +2.00S +2.00  +2.00S
```

- **P at 3e-3 ends at the end point of W's collective route** (W2(b)): tau_sum 2.00 on every seed, antipodal latents
  (u 0.036, cos -0.997), every counted channel cancelling by wrap, clock rms 0.011. Started past pi, the class takes exactly
  the collective wrapped solution the route was built to reach. At 1e-3 P's tau moved from 1.5 to 1.63-1.71 (all eight
  above C1's 1.55; the count gate missed there, 5/8, but 1e-3 is unreadable anyway).
- **W never got near the sheet.** At 3e-3 its tau_sum moved toward it but stopped at 0.13-0.70 on every seed, below 1, the
  value at which the offset term pi T alone reaches pi; at 1e-3 it stayed at 0.01-0.11. No W run cancels by wrap (wrap share
  0.000 at both rates). W's two SOLVED runs at 1e-3 (tau_sum 0.06, 0.11) are off the sheet, at the unreadable rate.
- Signed at 3e-3 ends antipodal and cancels every channel through zero (u 0.002, cos -1.000), as in SIGNROUTE (0.001,
  -1.000). abs and W keep same-direction latents (cos +0.54, +0.21).
- F's latents are frozen (u 0.707, cos 0 = the start, by construction); its cancelling channels are mostly through zero
  (0.170 of 0.182 at 3e-3), i.e. the per-channel route the frozen class has (F1).

## Predictions (prereg Sec. 8) against outcomes

| prediction | outcome |
|---|---|
| S >= 6/8 at both rates | 3e-3 8/8 (met); 1e-3 5/8 (missed: G0 fails) |
| A <= 2/8 | 1e-3 0/8 (met); 3e-3 3/8 (missed; G1 still passes at its bound k_A <= 3) |
| F A-LIKE at 1e-3; no prior at 3e-3 | 1e-3 unreadable (raw 3/8; ACCOUNTS T2's same cell 1/8); 3e-3 UNMEASURED (4/8) |
| W A-LIKE | 3e-3 A-LIKE, 0/8 (met); 1e-3 unreadable (raw 2/8, off the sheet) |
| P S-LIKE (>= 6/8) | 3e-3 7/8 (met); 1e-3 5/8 |
| C1 passes (P >= 6/8, median tau_sum >= 1.55) | 3e-3 pass (7/8, 2.00); 1e-3 tau 1.68 but 5/8 |
| reading BOTH (ROUTE the main alternative) | UNMEASURED: W came out as predicted, F landed between the anchors |

## Checks on the reading

- **Rule 2:** r(log10 final loss, acc) = -0.782 at T = 128 and -0.603 at T = 512 over 80 runs (0 diverged). The decisive
  W cell is a SOLVED contrast (0/8 vs 8/8) that also fires on accuracy (perm p 0.0002).
- **Rule 3 / convergence:** `python3 -m mapformer.experiment_audit --runs-dir mapformer/runs/signsep/p0` from /home/prashr
  (read-only) exits 2: no (json, pt) pairs with key 512 in its layout, so it runs check [1] only, pooling both rates (n = 16
  per arm). Its windowed classes agree with the registered per-seed classes in every arm: signed 13 SOLVED + 3 STALLED; abs 3
  + 12 + 1 DESCENDING; abs_wrap 2 + 13 + 1; signed_frozen 7 + 8 + 1; abs_wrap_past 12 + 3 + 1 ("NOT ALL FLAT", 13/16
  flat). Its "not all arms converged" flag concerns one DESCENDING run in each of abs, W, F and P, all at 3e-3. W's
  0/8 at 3e-3 is 7 STALLED + 1 DESCENDING: read at this budget.
- **Power (prereg Sec. 7):** with S at 8/8 and A at 3/8, F at 4/8 can read neither S-LIKE (needs k_F >= 7, or the accuracy
  test) nor A-LIKE (needs better(S, F), which on SOLVED requires k_F <= 3); its accuracy gap to S (+0.029) is below the
  0.05 the accuracy test requires. The UNMEASURED is what the registered design returns for a cell between the anchors.

## Per seed (accuracy at T = 128 and class: S SOLVED, T STALLED, D DESCENDING; none diverged)

```
lr     arm            2300   2301   2302   2303   2304   2305   2306   2307
1e-3   signed         0.968T 1.000S 0.925T 0.956T 1.000S 1.000S 1.000S 1.000S
1e-3   abs            0.859T 0.906T 0.943T 0.936T 0.947T 0.971T 0.988T 0.891T
1e-3   abs_wrap       0.876T 0.865T 0.929T 1.000S 0.923T 0.948T 1.000S 0.917T
1e-3   signed_frozen  0.901T 0.927T 0.989S 1.000S 0.869T 1.000S 0.913T 0.915T
1e-3   abs_wrap_past  0.899T 1.000S 1.000S 0.874T 1.000S 1.000S 0.911T 1.000S
3e-3   signed         1.000S 1.000S 1.000S 1.000S 1.000S 1.000S 1.000S 1.000S
3e-3   abs            0.934T 0.960T 1.000S 0.963T 1.000S 0.985D 1.000S 0.945T
3e-3   abs_wrap       0.948T 0.972D 0.963T 0.959T 0.957T 0.974T 0.933T 0.913T
3e-3   signed_frozen  0.960T 0.916T 1.000S 1.000S 0.955D 1.000S 0.983S 0.956T
3e-3   abs_wrap_past  1.000S 1.000S 1.000S 1.000S 1.000S 1.000S 0.881D 1.000S
```

## Reading for the paper

SIGNROUTE localised the |.| cost to the non-negative increment but removed both candidate obstacles at once. SIGNSEP built
the two missing cells from one start. Giving the |.| step a collective route to a wrapped point that cancels every channel
at once does not let it learn the ring map from a small start at lr 3e-3 (0/8 against signed 8/8), although the same class
learns from past pi (7/8) and ends exactly on that point: the route does not get the step across |k| = pi (registered
half-reading BARRIER BLOCKS). Whether the barrier alone is the obstacle is unmeasured: a signed step with its move latents
frozen, which has no barrier and no collective route, solved 4/8, between the anchors. The registered 2x2 therefore reads
UNMEASURED; the barrier account gains one registered half, and the route account is neither confirmed nor excluded. The
mechanism remains an account.

## Scope

Ring 32, moves +1/-1, observation-step-free rank 4, one layer, the common |.| start (not the default start), 300 epochs,
lr 1e-3 (unreadable) and 3e-3 (readable); n = 8 per cell. The half-reading is a statement about the constructed W class
against the anchors; tau's lower per-channel mobility (Amendment 1 finding 3) is answered by C1, not removed. The |.|
anchor at 3e-3 (3/8) sits within earlier |.| ring counts at that rate (LRSWEEP 2/8, SIGNROUTE 2/8, ROBUST lr x3 4/8, all
from the default start); G1 passed at its bound. Not run: the
torus, other starts, other rates, a longer budget.
