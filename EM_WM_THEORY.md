# EM vs WM: the best current theory (2026-09-11)

Synthesis over the whole EM/WM line: every measurement (repo ledger), every withdrawn claim
(graveyard audit), and the literature read first-hand. `EM_WM_STATE.md` remains the numbers of
record; this file is the account they support, and it names what would break it.

---

## The claim

**MapEM applies ONE position kernel to every query-key pair, and content can only rescale it.
MapWM builds a kernel whose amplitudes and phases the pair's content sets, so it can reshape
per pair. Everything measured follows from which of those two a task needs.**

- **When the retrieval offset is fixed by the task's structure, one shared kernel is the better
  object, and EM wins** -- decisively at length.
- **When the offset varies per query, a shared kernel cannot be reshaped per query.** EM's only
  remaining handle is the query TOKEN's own step: one independent solution per query token,
  each trained by a fraction of the queries. That is where EM loses, and the loss is a search
  cost, not a representational one.

### The structural half, verified in code and measured

`model.py:295-296, 375-379` (EM) and `:224-232` (WM):

| | position kernel | what content can do to it |
|---|---|---|
| MapEM | `A_P[t,s] = q0^T R(dtheta) k0` from two LEARNED VECTORS | `A_X` multiplies it -- a per-pair scalar GAIN. The content branch is never rotated, so content cannot move the kernel's peak |
| MapWM | `Q_t^T R(theta_s - theta_t) K_s` on CONTENT-DERIVED Q,K | amplitude and phase per block, per pair: `sum_b \|Q_b\|\|K_b\| cos(dtheta_b + phi_b(q,k))` |

Measured, not asserted (`probe_phase_spread.py`, stored recency checkpoints, n=8):

| arm | circular sd of kernel phase across pairs | amplitude-weighted | spread of per-offset mean phase |
|---|---|---|---|
| WM | **2.003** (R ~ 0.13, near-uniform) | 1.838 | **0.918** |
| EM, single `p0` | **0.000** | 0.000 | 0.000 |
| EM, separate `q0/k0` | **0.000** (phases non-zero, mean 1.603 rad, but identical for every pair) | 0.000 | 0.000 |

This settles `EM_WM_STATE.md` open question 4 and answers a literature sweep that argued MapWM's
kernel is also shared: the rotation FAMILY is shared in both (one `omega` per head/block); the
realised kernel is not. WM's also varies systematically with the query's offset.

### The fixed-offset half: EM is BETTER, and this was never written down

| task | contrast | n | result |
|---|---|---|---|
| torus paper task, ext-s l=1024 | EM_P0 - WM | 8, one batch | **+0.070, 8/8, MDE 0.034 DETECTABLE** |
| torus paper task, ext-s l=2048 | EM_P0 - WM | 8, one batch | **+0.085, 8/8, MDE 0.034 DETECTABLE** |
| torus paper task, OOD-s l=512 | EM_P0 - WM | 8 | +0.035, 7/8, MDE 0.031 DETECTABLE |
| recency, ONE fixed k=64, T=2048 | EM - WM | 8, one batch | **+0.191, 7/8, MDE 0.150 DETECTABLE** |
| recency, ONE fixed k=64, learning speed | epochs to loss < 0.5 | 8 | EM median **54** vs WM **99** (WM 7/8 reach it, EM 8/8) |
| torus clean, T=2048 | EM - WM | 3, no MDE | +0.192 (unpowered, same direction) |

The first three are computed here from the committed per-seed file `PAPER_OOD_EXTENDED_n8.json`
(means in `BASELINE_TABLE.md:83`); **no results file states them**, and they falsify the
standing summary line "on map tasks the two tie to within 0.004" (see Corrections below).

### The varying-offset half: one solution per query token

With `k` drawn per query, the recency vocabulary gives each offset its own token
(`environment_recency.py:140-146`). Then:

- EM_P0 - WM = **-0.375** (0/8, MDE 0.154); EM_sep - WM = -0.137 (1/8). By the registered
  readout WM reaches loss < 0.5 on 8/8 seeds, EM_P0 on 0/8.
- The deficit is **not** representational: the rewind installed and frozen scores 1.000 (8/8);
  installed trainable at 8x scale with the content leak closed, also 1.000 (8/8).
- Every solved large-k cell in all five EM arms goes through the query token's own step -- 40%
  to the kernel's peak with `A_X > 0`, 55% to its trough with `A_X < 0` (the sign gauge), 3-6%
  static. Failed cells never carry one.
- With ONE shared k, a 63-symbol rewind is found on 7/8 seeds; all wrapped, 0/8 linear.
- A k curriculum gives +0.127 (7/8) and closes nothing; its whole gain sits at k <= 32, the
  tokens it introduces early.

So the failure is per token, and the obstacle is spread across 64 of them rather than the size
of any one rewind. **Why only about half succeed is the open mechanism** (`SPREAD_PREREG.md` is
in flight; the timing account -- a token must find its rewind before the kernel sharpens -- is
consistent with S2's smooth-at-init / rugged-after-training result but untested).

### The intermediate rung

Phase freedom in `q0/k0` gives `n_b` GLOBAL phases -- still pair-independent (measured 0.000
spread), but no longer pinned to zero. It is worth **+0.146** against a matched-optimiser
control (22/24; +0.113 on fresh seeds, clearing its MDE by 0.002), moves every head's kernel
peak off zero (48/48 heads, n = 2..63), and does NOT change the retrieval route or shorten the
shift a token must make. The ladder WM (per-pair) > EM-sep (`n_b` global) > EM-`p0` (none)
matches every ordering measured, and remains **untested as a causal claim**.

---

## Where this sits in the literature

- **Whittington et al., Neuron 2025 [11]** distinguishes EM (memories in synapses, a separate
  fast-weight store) from WM (memories in activity slots), proves them equivalent in
  computation, and predicts EM scales better per RNN neuron. **MapEM has no such store** -- its
  memory is the same attention WM uses -- so the capacity result does not transfer. MapFormer
  cites [11] for its own Fig. 11 scaling claim (`mapformer.txt:2264-2269`), which transfers a
  statement about RNN state budget to one about attention factorisation.
- **[11]'s N-back is FIXED-N with no filler** (`tale_two_algorithms.txt:1672-1674`), so it
  corresponds to our FIXED-k condition, not our varying-k task. [11] says N-back is the one
  place EM's faster learning disappears; we find EM faster there (54 vs 99 epochs) and better
  at 2x length. **The tension is unresolvable in this corpus**: the supporting panel (Fig. S2G)
  is not in `papers/`. [11] also notes (l.599) that on N-back its own EM/WM discriminator
  degenerates, because past sequences and slot sequences coincide.
- **No paper in the corpus isolates shared vs per-pair position as a variable**, and no survey
  names the axis (`survey_longctx.txt:1396` opens "Content-Aware Position Embedding" with CoPE
  and DAPE and does not subdivide). Treat "unstudied" as this corpus's answer, not the field's.
- **PaTH (`path.txt:891-905`) is the sharpest external check and it refines the claim.** Its
  transform is a cumulative product of data-dependent Householders between i and j -- content
  set, but SHARED by every pair spanning that interval -- and it succeeds at MQRAR-N-back for
  N < 4 where FoX (a shared monotone scalar decay) fails. So "shared kernels fail at contextual
  offsets" is too strong. The failing property is narrower: **a fixed EVEN kernel `kappa(dS)`
  of a scalar displacement**, which is what MapEM has.

---

## What would break this account

| # | prediction | status |
|---|---|---|
| P1 | Matched queries-per-token collapses the m=4/16/64 arms onto one curve | **SPREAD, in flight** |
| P2 | MapEM fails MQRAR-N-back for N>1 while MapWM and PaTH pass | untested; the one published task that discriminates |
| P3 | EM with per-pair origin vectors (`q0`,`k0` low-rank functions of content, Hadamard composition kept) recovers WM's varying-k performance | untested; the decisive architectural arm |
| P4 | Removing the per-token handle (encode k CoPE-style as a count, not one token per offset) hurts EM further; [11]'s capacity account predicts no change | untested, environment-only |
| P5 | Varying `n_symbols` at fixed `k_max` moves the gap ([11]'s task-size axis) or does not (kernel axis) | untested, one flag |
| P6 | `\|rho\|` = +0.292 survives at LEARNED kernel amplitude, not only at the frozen 0.003 | untested; also decides whether `AP_KERNEL_DIAGNOSTIC.md`'s 2026-08-09 falsification needs narrowing |

**Known weaknesses, carried openly.** Phase freedom clears its fresh-seed MDE by 0.002 and its
first eight seeds ran 1.9x high -- the third time in this line. The DOF torus half (freedom
costs at OOD length) is n=8 and never extended. The ladder is post-hoc. "EM - WM = -0.375" is a
property of 64 distinct offsets, not of EM: the sign reverses at one fixed offset. And the
existence construction that killed Thm 3's corollary exists only as prose -- no script, no json
reproduces its 1423/1423.

---

## Corrections this synthesis forces

1. **"On map tasks EM and WM tie to within 0.004" is FALSE at extended length.** EM is ahead by
   +0.070 (l=1024) and +0.085 (l=2048), 8/8 seeds, one batch, MDE 0.034. The "four map cells"
   behind the tie were never enumerated in any file, and the set excludes this batch.
2. **"Found from scratch: 0/40" is withdrawn** but still asserted in `NOLEAK_RESULTS.md:70`,
   `WARM_RESULTS.md:9,61-73`, `UNFREEZE_RESULTS.md:116,119`, `MAGONLY_RESULTS.md:57-70` and
   `CLAUDE.md:2864`.
3. **WM-is-additive is still asserted in `axes_measured.tex:84`**, the presentable results
   paper, and in ~15 other files (full list in the graveyard audit).
4. **Citation errors**: the separate-`k0p`/`q0p` passage is in the paper's **App. A.7**
   (line 1414 header), not A.4; "two separate pools of neurons" is **App. C.3** (line 2228),
   not Sec 5.4. Both are miscited in `CLAUDE.md` and `TALE_OF_TWO_ALGORITHMS.md`.
5. **`PAPER_TASK_ACCURACY.md:4` quotes paper numbers (0.955 / 0.999) that appear in no table of
   the paper**, and the file is on the VERIFIED list.
6. **`README.md:115-117` "Refuted on four tasks"** carries no recency caveat, where the sign
   reverses and the fresh-seed replication is unmeasured (+0.073, 9/16).
