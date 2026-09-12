# EM vs WM: what the evidence supports (2026-09-11, v2 after adversarial review)

**v1 of this file made a claim that did not survive review. This version records what broke and
why.** `EM_WM_STATE.md` is the numbers of record. The review is summarised in "What v1 got
wrong" at the bottom; nothing there is quietly absorbed.

---

## 1. What is established

### 1a. The architectural asymmetry (from the code; my probe adds nothing to this)

`model.py:295-296, 375-379` (EM) and `:224-232` (WM):

- **MapEM**: `A_P[t,s] = q0^T R(dtheta) k0`, built from two LEARNED VECTORS. One kernel for every
  query-key pair. `A_X` multiplies it -- a per-pair scalar gain. EM's content branch is never
  rotated, so content cannot reshape the kernel.
- **MapWM**: `Q_t^T R(theta_s - theta_t) K_s` on CONTENT-DERIVED Q,K, i.e.
  `sum_b |Q_b||K_b| cos(dtheta_b + phi_b(q,k))` -- amplitude and phase per block, per pair.

**Important qualification the first version missed.** "Content can only rescale" understates
EM's function class, because the scalar gain carries a SIGN. Under the gauge
`A_X (*) (-kappa) = (-A_X) (*) kappa`, a negative gain converts peak retrieval into trough
retrieval, and `SEARCH_RESULTS.md` measures **55% of solved large-k cells taking the trough
route with `A_X < 0`** (the sign tracks the route 1.000/0.000). Content moves the EFFECTIVE
target on more than half of all successful retrievals. It still cannot give two pairs different
kernel SHAPES; that is the real limit.

### 1b. The phase measurement, after three bug fixes

`probe_phase_spread.py`, stored recency checkpoints, 29,056 pairs per model:

| arm | circ sd of phase across pairs | amplitude-weighted | per-offset mean phase | live blocks |
|---|---|---|---|---|
| WM, trained | 1.947 (seeds 1.83-2.08) | 1.816 | 1.656 | 64/64 |
| **WM, UNTRAINED (control)** | **2.722** (3 inits) | 2.603 | 2.032 | 64/64 |
| EM, single `p0` | **0.000** | 0.000 | 0.000 | **25/64** |
| EM, separate `q0/k0` | **0.000** (phases non-zero, mean 1.603, identical every pair) | 0.000 | 0.000 | 64/64 |
| uniform ("no structure") null at this N | 3.267 | -- | -- | -- |

Two readings, and only the first is licensed:

- **WM CAN reshape its kernel per pair; EM cannot.** EM's 0.000 is now computed over the same
  pairs, not asserted.
- **Nothing here shows WM's advantage comes from reshaping.** An untrained WM -- which cannot do
  the task -- is MORE spread (2.722) than a trained one (1.947), so training REDUCES per-pair
  spread. The statistic tracks the parameterisation, not a learned capability. Any claim of the
  form "WM wins because its kernel is per-pair" needs different evidence than this.

Incidental but real: the trained single-`p0` kernel leaves **39 of 64 blocks dead** (amplitudes
to 5e-86). It prunes frequencies rather than using the whole bank.

### 1c. The varying-offset account -- the part that needs no kernel premise

With `k` drawn per query and one query token per offset (`environment_recency.py:140-146`):

- EM_P0 - WM = **-0.375** (0/8, MDE 0.154); by the registered readout WM reaches loss < 0.5 on
  8/8 seeds, EM_P0 on 0/8.
- Not representational: installed frozen 1.000 (8/8); installed trainable at 8x with the leak
  closed, 1.000 (8/8).
- Every solved large-k cell rewinds via the query token (peak or trough); failed cells never do.
- With ONE shared k, a 63-symbol rewind is found on 7/8 seeds -- all wrapped, 0/8 linear -- and
  EM is FASTER than WM (54 vs 99 median epochs to loss < 0.5).
- A k curriculum gives +0.127 (7/8) and closes nothing; its gain is confined to k <= 32.
- **Budget alone moves the full task 0.578 -> 0.928** (4x epochs, 6/8 seeds ~1.000;
  `SPREAD_RESULTS.md`), so the deficit is substantially step-efficiency -- what a per-token
  search account predicts. Caveat: WM at that budget was never run, so the -0.375 headline
  remains the fair SHARED-budget comparison.

**So the deficit is a per-token search cost: 64 independent wrapped rewinds, each trained by
1/64 of the queries.** This is the strongest thing in the line, and it does not depend on
anything in 1a/1b -- with one shared k, the same shared kernel wins.

### 1d. Per-pair origins recover the varying-offset deficit (2026-09-12)

The one architectural test that varies SHARING alone, holding composition, rank, depth and the
init function fixed (`PAIRORIGIN_RESULTS.md`, 3 arms x 8 seeds, one batch):

| readout | single `p0` | **EMPair** | WM |
|---|---|---|---|
| accuracy (varying k) | 0.600 | **0.880** | 0.975 |
| kernel phase spread across pairs | 0.000 | **1.448** | 1.966 |
| solved cells via a per-token rewind | 0.962 | **0.185** | -- |

`EMPair - P0` = **+0.280** (7/8, MDE 0.216); `EMPair - WM` = -0.095, inside MDE. So on the
VARYING-offset side the sharing account is supported by intervention, not correlation: remove
the sharing constraint and the model stops rewinding query tokens, reshapes the kernel, and
recovers most of the gap. Two caveats that keep this from being finished: r(loss, acc) = -0.983
(a statement about what training finds, which is the right frame given the existence
construction), and the +2,048-parameter confound, whose control is queued.

**QUALIFIED by the capacity control (`PAIRCONST_RESULTS.md`, 2026-09-12).** A constant-origin
arm with 128 MORE parameters scores 0.782, between P0 and EMPair, so the accuracy gain splits
+0.182 (pathway) and +0.098 (freedom) with NEITHER detectable at n=8. The mechanism split is
clean and goes the other way: the control KEEPS the per-token rewind route (0.948 against P0's
0.964) while EMPair abandons it (0.189). Read 1d as: per-pair freedom changes WHAT THE MODEL
DOES; how much of the accuracy it buys is unresolved.

**This does not revive 2a.** It is evidence about the varying-offset case only.

---

## 2. What is NOT established

### 2a. "When the offset is fixed by structure, a shared kernel is better, and EM wins"

v1's headline. **Demoted to: on one batch, EM degrades less with LENGTH than WM.** The numbers
reproduce exactly (paired, one batch, parameter-matched within 0.16%):
+0.035 (l=512, 7/8), +0.070 (l=1024, 8/8), +0.085 (l=2048, 8/8), MDE 0.031-0.034, and they
survive IID-matching (+0.0856). Four reasons they do not support the claim:

1. ~~**The loser is near the floor.**~~ **ANSWERED, AND IT INVERTS (2026-09-12).** The floors
   are now MEASURED per condition on exactly the scored events (`PAPER_TASK_FLOORS.md`):
   0.522 IID, 0.216 OOD-d, 0.799 (l=512), 0.801 (l=1024), 0.802 (l=2048). Being near the floor
   COMPRESSES the raw scale, so normalising by it makes the gap LARGER, not smaller. Fraction of
   the available range used, mean over the same 8 seeds:

   | condition | WM | EM_P0 | MapPoPE |
   |---|---|---|---|
   | OOD-s l=512 | 0.715 | 0.889 | 0.959 |
   | ext-s l=1024 | 0.460 | 0.812 | 0.924 |
   | ext-s l=2048 | **0.261** | **0.692** | 0.851 |

   Paired, floor-normalised: EM - WM = **+0.174** (l=512, 7/8), **+0.352** (l=1024, 8/8),
   **+0.430** (l=2048, 8/8), all DETECTABLE. So this objection does not demote the effect; it
   was the one objection I could check cheaply and it came out against itself.
2. **The best arm in that batch is a PER-PAIR kernel -- but the contrast is UNMEASURED.**
   `MapPoPE-Flat` (a `MapFormerWM` subclass) is ahead of EM_P0 at all six cells, and
   floor-normalised the gap is +0.070 / +0.112 / +0.159 at l=512/1024/2048 -- **7/8 seeds but
   inside its MDE at every length**. It is a directional counterexample, not a detectable one,
   and v2 overstated it by calling it a counterexample flatly.
3. **The pattern is monotone in LENGTH, not in offset-fixedness** (+0.018 IID, +0.035, +0.070,
   +0.085). "Helps at OOD length" is this project's universal unexplained signature, shared by
   rank, the InEKF, the forget gate and PoPE. EM is a fifth instance of an unexplained axis.
4. **Counterexamples on other fixed-offset (dS = 0) tasks**: MiniGrid n=8 one batch, EM - WM
   +0.0035 and -0.016..-0.020 (ties/loses); vocab n=8 trimmed +0.0000; Match-Query EM below WM;
   compositional and family tree below MapWM-Flat.

Also: `runs/paper_task_n8/` and its 48 training logs are deleted, so rule 9 cannot be applied
to the one load-bearing cell, and the batch ran 16 epochs on LinearLR -- the recipe rule 10
exists to warn about.

**What survives of the correction it forced:** the summary line "on map tasks EM and WM tie to
within 0.004" is still wrong, but the right replacement is narrow -- *in one 16-epoch batch at
pe=0.8, EM is ahead at extended length, near the blank floor* -- not "EM wins at fixed offsets".

### 2b. The freedom ladder (WM per-pair > EM-sep > EM-`p0`)

Post-hoc, and contradicted by MiniGrid, vocab and Match-Query orderings. Phase freedom itself is
real (+0.146 vs a matched control, 22/24) but clears its fresh-seed MDE by 0.002, and its first
eight seeds ran 1.9x high -- the third time in this line.

### 2c. Relation to the withdrawn Thm 3 corollary -- stated, not hidden

`AUDIT_2026-09-10.md` #2 withdrew "EM ~ WM at constant offset, EM << WM when the offset varies",
because the offset is chosen by the MODEL (a rewinding query token makes it zero) and EM can
represent the task. **Section 1c resembles that corollary and must not be read as reinstating
it.** The difference: the corollary was a claim about what EM can REPRESENT, and it is dead
(frozen install 1.000). 1c is a claim about what training FINDS, on evidence the corollary never
had. Where they overlap in wording, the withdrawal stands.

---

## 3. Literature, corrected

- **[11] (Whittington 2025)** distinguishes memories in synapses (EM, a separate fast-weight
  store) from memories in activity (WM). MapEM has no such store, so the capacity result does
  not transfer -- and by `TALE_OF_TWO_ALGORITHMS.md`'s own argument **both MapFormers are WM
  models in [11]'s sense**, which means [11]'s EM/WM predictions have no clean referent here.
  v1 called our fixed-k result a "tension" with [11]'s N-back exception; that overstated it.
- **[11]'s N-back is fixed-N with no filler** (`tale_two_algorithms.txt:1672-1674`), so it maps
  to our FIXED-k condition, not our varying-k task. [11] also notes (l.599) its own EM/WM
  discriminator degenerates on N-back. Its learning-speed evidence is Fig. S2G, **not in this
  corpus**.
- **PaTH, corrected.** v1 called its transform "shared by every pair spanning that interval",
  which is vacuous as written and false as intended: `H_t = I - beta_t w_t w_t^T` accumulated
  between i and j varies with both endpoints and with the tokens between them
  (`path.txt:115-131`). PaTH sits on the per-pair side of the axis, not EM's. What survives is
  narrower: PaTH solves MQRAR-N-back for N < 4 (`path.txt:968`) where FoX -- a shared monotone
  scalar decay -- fails, so a content-dependent interval operator can do contextual-offset
  retrieval. It says nothing about shared-vs-per-pair, because it never varies that.
- **No paper in the corpus isolates shared vs per-pair position**, and no survey names the axis
  (`survey_longctx.txt:1396`). That is this corpus's answer, not the field's.

---

## 4. What would decide it

| # | test | cost | status |
|---|---|---|---|
| P1 | Matched queries-per-token across m = 4/16/64 | done | **PARTLY ANSWERED** (`SPREAD_RESULTS.md`): at matched budget fewer offsets is better, +0.422 (8/8). The exposure-matched cells landed on the CEILING, so exposure-vs-token-count is still open; re-run with the budget set so the m4 arm lands near 0.8 |
| P2 | EM with PER-PAIR origin vectors, Hadamard composition kept | done | **CONFIRMED** (`PAIRORIGIN_RESULTS.md`): +0.280 over single-`p0` (7/8, MDE 0.216), within MDE of WM; phase spread 0.000 -> 1.448 and the per-token-rewind route 0.962 -> 0.185. Capacity control `PAIRCONST_PREREG.md` queued |
| P3 | Port PaTH's MQRAR-N-back; run MapEM, MapWM, PaTH | published task | the only external benchmark that discriminates |
| P4 | Remove the per-token handle: encode k as a count, not one token per offset | env flag | kernel account predicts EM degrades further; capacity account predicts nothing |
| P5 | Vary `n_symbols` at fixed `k_max` | env flag | [11]'s task-size axis vs the kernel axis |
| P6 | Re-run the extended-length cell with a converged recipe, logs kept, floor reported | 2 arms x 8 | the only way to rescue or bury 2a |
| P7 | `\|rho\|` at LEARNED kernel amplitude, not the frozen 0.003 | 2 arms x 8 | also decides whether `AP_KERNEL_DIAGNOSTIC.md` needs narrowing |

---

## What v1 of this file got wrong (2026-09-11, same day)

An adversarial review plus my own re-checks found:

1. **Three bugs in `probe_phase_spread.py`, all mine.** (a) The EM rows were hardcoded `0.0`
   literals under a table captioned "measured, not asserted" -- rule 23. (b) `_flat` used
   `reshape(-1, H, nb)` on a (pairs, heads, candidates, blocks) array, interleaving the candidate
   axis across heads and blocks; every reported column was a scramble, including the 2.003 first
   published. (c) The amplitude-weighted column divided by a `clip(..., 1e-12)` floor, so dead
   blocks (5e-86) produced |z| ~ 1e-69 and read as maximal spread. Bugs (b) and (c) were caught
   by outputs that contradicted their own algebra, not by the review.
2. **"~1.97 is near-uniform" was wrong** -- the finite-sample uniform null is 3.267 at this N.
3. **The untrained control was never run**, and it inverts the interpretation (above).
4. **The floor was omitted** beside a headline whose loser sits 0.05 above it, breaking rule 4.
5. **`MapPoPE-Flat` was omitted** from a table drawn from the batch it wins.
6. **A row mixed arms and batches**: the +0.192 long-sequence cell is separate-`q0/k0` from an
   April stored batch, inside a one-batch EM_P0 table (rules 3 and 6).
7. **The fixed-k EM - WM win was quoted at T=2048 only** (+0.191); at T=1024 the same contrast
   is +0.038, 3/8, unmeasured, and the whole comparison is exploratory.
8. **PaTH was miscited and mischaracterised** (section 3).
