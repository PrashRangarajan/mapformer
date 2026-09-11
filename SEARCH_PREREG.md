# SEARCH -- why from-scratch EM never finds the recency rewind, and what phase freedom does instead

Pre-registered 2026-09-11, before any readout below was computed. `EM_WM_STATE.md` Sec 6 items 2 and 3.

## Where this starts

- The rewind (query token `q_k` carries `Delta = -(k-1)` symbol steps, so the inclusive cumsum
  lands `S_query` on the answer's `S`) is inside single-`p0` EM's function class (installed + frozen
  1.000), holdable (installed at 8x, leak closed, 1.000), and "never found" from scratch: pooled
  LINEAR rewind slope 0.000 +/- 0.02 over 40 runs (`MAGONLY_RESULTS.md`, `_REWIND_PROBE.json`).
- Data already seen before writing this: `VanillaEM_P0_r4` s0's per-offset accuracy is nearly
  BINARY and not monotone in k (k11 0.00, k12 1.00, k15 0.00, k17 1.00, k54 1.00, k60 0.00). No
  aggregated per-k table has been computed for any arm.

## A gap in "never found" that this pre-registration must test first

`theta = omega * cumsum(Delta)` enters only through cos/sin, so block i's rewind is defined only
MODULO `2 pi / omega_i`. A token with `Delta(q_k)_i = -(k-1) Delta_sym_i + 2 pi m_i / omega_i` is an
exact rewind, and gradient descent from a random start would reach the nearest such point, not the
linear one. The linear slope statistic cannot see a wrapped rewind. The A_P-selection readout can,
and it read 11-16% for from-scratch arms against a uniform floor far below that. The binary per-k
profile is also what independent per-token successes would look like. So "never found" may be
"found token by token, wrapped, for some k".

## S1 -- retrieval anatomy on existing checkpoints (eval-only, no training)

Checkpoints: `runs/dof/recency` `VanillaEM_P0_r4`, `EMDoF_alignlock`, `EMDoF_magonly`,
`EMDoF_alignfree`, `VanillaEM_r4` (24 seeds each); `runs/recency_em` `Vanilla_r4` (WM, 8 seeds) for
the attention readouts only. Held-out episodes: `RecencyWorld(seed=10000)`, private
`RandomState(7000 + seed)`, 128 episodes at T=1024 (~28 queries per k per checkpoint).

Per (checkpoint, k), EM arms:
- `acc_k` -- the model's own argmax prediction.
- `sel_k` -- A_P selection: fraction of queries whose A_P (summed over heads) argmax over PRIOR
  SYMBOL keys is the answer. Also per head, and `sel_max_k` = max over heads.
- `sel0_k` -- the same with `Delta(q_k)` replaced by 0 at the query position (the counterfactual
  without the query token's own step). **`sel_k - sel0_k` is the per-token rewind's contribution**,
  wrapped or not.
- `att_k` -- attention weight on the answer key, max over heads (the real softmax(A_X (*) A_P)).
- `gain_k` -- sign of A_X at the answer key in the head with the largest `att_k`.

**H-wrap (S1-a).** In `VanillaEM_P0_r4`, among (seed, k) cells with `k >= 8` (below that a peak at
`dS ~ 0` reaches the answer without a rewind):
- PREDICTION: cells with `acc_k >= 0.9` have `sel_max_k - sel0_k >= 0.5` in at least 70% of cases,
  and cells with `acc_k <= 0.3` have it in at most 20%.
- CONFIRMED if both hold. "0/40 never find a rewind" is then restated as "find it per token,
  wrapped, for a subset of k". REFUTED if solved cells are under 30%: the solved large-k cells go
  through some other route (for example A_X's sign selecting a kernel trough), recorded via
  `gain_k`. Anything between is reported as partial.
- Also reported for every EM arm: r(acc_k, sel_max_k - sel0_k) over (seed, k) cells with `k >= 8`.

**H-phase (S1-b).** What does phase freedom (`AlignFree - MagOnly` +0.146) do?
- Per-k gain curve, paired by seed (n=24), in four k bins (1-16, 17-32, 33-48, 49-64), with MDE.
  Descriptive; no direction is registered.
- Route: for each arm, of the solved cells (`acc_k >= 0.9`, `k >= 8`), the fraction with
  `sel_max_k - sel0_k >= 0.5` (per-token rewind route).
- PREDICTION: AlignFree's extra solved cells go through a NON-rewind route. The fraction of solved
  cells on the rewind route is lower in AlignFree than in MagOnly by at least 0.15. Reason: with
  free phases a head's kernel peak can sit at `dS != 0` (with `rho = 1` it is at `dS = 0` exactly,
  since `kappa(0) = sum a_i >= kappa(anything)`), so a shift of the peak can stand in for a shift
  of the query.
- Kernel-peak readout, and the MANIPULATION CHECK for the reason: per head, the argmax over symbol
  distance n in 0..80 of the mean A_P profile computed with `Delta(q) = 0`. MagOnly / P0 / AlignLock
  must peak at the smallest n (they have `rho = 1`). If they do not, the readout is broken and
  H-phase is not read. For AlignFree and sep, report the fraction of heads that peak at n >= 2.

## S2 -- the gradient at initialisation (no training)

Arms `VanillaEM_P0_r4`, `EMDoF_alignfree`, `EMDoF_magonly`, seeds 0-7, constructed exactly as
`train_recency` does (`torch.manual_seed(seed)` then `VARIANT_MAP`). Loss = the training loss on
8 batches (16 x T=1024) from `RandomState(seed)`, i.e. the first 8 training batches.

- Function-space gradient `g_k = -dL/dDelta(q_k)`, with the Delta table as a leaf.
- **Linear-rewind component.** `y_k = <g_k, s> / <s, s>` with `s` = mean symbol Delta at init.
  The rate of change of the rewind slope under gradient flow is the slope of `y_k` on k. Reported
  pooled per seed, with t over the 8 batches.
- **Kernel-at-answer component.** `cos(g_k, dA_P(query, answer)/dDelta(q_k))`, averaged over
  queries. Does the loss gradient at init point toward raising the kernel at the answer?
- **Scale.** `||dL/d(position pathway)|| / ||dL/d(content branch)||` (w_in, w_out, omega, origin
  vectors against q/k_content, v, o, ffn) and `|A_P|` / `|A_X|` at init.
- **Ruggedness** (per k, at init AND at the trained P0 checkpoints). For query token k, the number
  of local maxima of the mean-kernel value at the answer along the straight line from the current
  `Delta(q_k)` to the exact linear rewind `-(k-1) s`, over 2001 points.

**H-rugged.** PREDICTIONS: (i) local maxima on the path grow with k, median >= 5 at k = 64 against
<= 1 at k <= 4, at init and at trained P0; (ii) at init the linear-rewind rate is not consistent:
|t| < 2 on at least 6 of 8 seeds for P0. Either failing weakens the account that the search fails
because the landscape each query token faces is multi-modal. (i) holding with (ii) failing would
mean the gradient points to the rewind but gets trapped on the way.

## S3 -- training: is a constant rewind findable, and does a k curriculum find the linear one?

Recipe as every recency batch: `k_max 64, min_gap 64, T 1024, 300 ep x 48 batches x 16, cosine,
lr 1e-3, 1 layer, d 128, h 2`, no `--fast-attn`. Output `runs/search/`.

| arm | variant | task |
|---|---|---|
| `P0_fix64` | `VanillaEM_P0_r4` | every query k = 64 (train and eval) |
| `P0_fix16` | `VanillaEM_P0_r4` | every query k = 16 |
| `WM_fix64` | `Vanilla_r4` | every query k = 64: the positive control |
| `P0_cur` | `VanillaEM_P0_r4` | k drawn from 1..2*2^(epoch//30), full 64 from epoch 150; eval on 1..64 |
| `P0_repro` | `VanillaEM_P0_r4` | default task, seed 0: must be BITWISE identical to `runs/dof/recency/VanillaEM_P0_r4_s0` (checks the edits left the default task unchanged) |

Seeds 0-7 per arm. Gates for fixed k = 64 (`validate_recency --k-fixed 64`, min_gap 64, T 1024,
200 episodes) pass: o1 0.063, o3 0.079, marginal 0.080, most-recent 0.057, oracle 1.000, against a
chance of 0.0625. The default stream is MD5-identical to the pre-edit environment.

Readouts: held-out accuracy at T=1024 (registered primary) and T=2048; epochs to loss < 0.5; for
the fixed arms, `sel` and `sel - sel0` for the single query token, and the linear ratio
`<Delta(q_K), s>/<s, s>` against `-(K-1)`; for `P0_cur`, the S1 readouts and the linear slope.

- **S3-P1 (positive control).** `WM_fix64` >= 0.95 on >= 6/8 seeds. If this fails, the fixed-k
  arms are not interpreted.
- **S3-P2.** `P0_fix16` solves (>= 0.9 on >= 6/8) and `P0_fix64` does not (>= 0.9 on <= 2/8).
  CONFIRMED: search difficulty grows with the SIZE of the rewind, consistent with H-rugged. If
  `P0_fix64` also solves, the standard task's failure is not about finding one large rewind but
  about k-many tokens under a shared bottleneck, and the curriculum is the relevant arm.
- **S3-P3.** `P0_cur` does NOT close the gap: `P0_cur - P0` (stored `runs/dof/recency`
  `VanillaEM_P0_r4` s0-7, reuse licensed only if `P0_repro` is bitwise identical) is below its
  MDE, or `P0_cur` stays under 0.9. Reason: rank-4 `W_out W_in` lets every `q_k` embedding move
  independently, so nothing learned at small k transfers to large k. If `P0_cur >= 0.95` on
  >= 6/8 with a linear slope <= -0.5, the curriculum finds the linear rewind and the prediction is
  refuted.

## Analysis discipline

`stats_guard.paired` for every contrast (MDE = 2.8 sd / sqrt(n); "unmeasured", never "null");
`stats_guard.rule9` before any loss-matched reading; `ckpt_guard.require_checkpoints` for every
checkpoint set; `ckpt_guard.compare_checkpoints` for `P0_repro`. Every registered verdict is
reported as met, split or refuted, including when it is obvious.
