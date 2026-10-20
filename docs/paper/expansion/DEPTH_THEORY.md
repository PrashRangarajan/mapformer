# The depth exchange rate: what index attention computes, layer by layer (expansion note, 2026-10-20)

Scope: main draft Section 5 (`docs/paper/MAIN_DRAFT1.md`), outline claims 5.1-5.4 (`docs/paper/OUTLINE_main_v2.md`).
Every statement carries one label:

- **ESTABLISHED** -- a registered result in a results file (the results file is authoritative; quoted, not re-derived).
- **ANALYSIS** -- computed here, post hoc, eval-only on stored checkpoints (CPU), after the batches were read. Not
  pre-registered; descriptive. Each number names its script and output file.
- **THEORY** -- an argument (a construction, an upper bound, or a cited theorem applied to our setting). Not a measurement.
- **SPECULATION** -- a mechanism the data are consistent with but do not establish.

Scripts and outputs: `docs/paper/expansion/scripts/depth/` (outputs in `out/`). Every stored accuracy was reproduced
before any new readout was taken (Section 4.0). No GPU was used; nothing was trained.

---

## 1. Summary

1. **ESTABLISHED (the rates).** One path layer is worth about three index layers on Dyck-2 at matched depth and on the
   biased ring wherever a step can be undone; one index layer suffices when the walk never reverses (p_plus 1.0); on the
   torus at T = 1024 four index layers or a loop of four fall far short (0.860 / 0.864 vs 1.000), and depth helps only
   short-gap revisits.
2. **THEORY (why path integration needs one layer).** A revisit at move t must retrieve an observation at any earlier s with
   P_s = P_t (mod N), P the prefix sum of the moves. In the path model the score depends on omega (P_t - P_s) directly: the
   integrator supplies the prefix sum, exactly and for any gap, before attention. Index attention sees position only as
   t - s, so the predicate "the moves in (s, t] sum to zero" must be computed in the residual stream.
3. **THEORY (two routes for index attention).** (A) *Global count*: uniform causal attention averages the moves, giving
   P_t / n_t; rescaled and matched, this is a depth-2 solution in principle (Liu et al. 2023, Lemma 6; Yao et al. 2021's
   depth-2 Dyck construction), but it needs a normaliser and precision 1/n_t, its MLP width and weights grow with N x T in
   the known construction, and the target is highly sensitive (every one of the g moves since the last visit decides it;
   Hahn and Rofin 2024). Its cost grows with T, not with depth. (B) *Local return set*: a layer can realise rules
   whose return offset depends on a bounded number of recent direction runs (*segments*): undoing the last run
   (retrace), or a constant offset (a clock). Its cost grows with the number of segments a revisit spans, and depth
   extends that number.
4. **ANALYSIS (route B is what deep index attention learned on the torus).** Of the torus's short-gap revisits, 82% are
   one-level retraces; four index layers or the loop solve them (0.996 / 1.000), and on the rest accuracy falls with the
   number of segments spanned: 0.93 / 0.98 at 3-4 segments, 0.74 / 0.65 at 5-8, 0.56 / 0.50 at 9-16, floor beyond. At a
   fixed gap of 32-127 moves the same fall appears (0.87 / 1.00 at 2 segments to 0.49 / 0.50 at 17+), so segments, more than the gap in moves, limit the deep
   models (Section 4.4 finds a smaller residual gap effect). This accounts exactly for "just above retrace": 0.943 = (0.680 x 0.996 + 0.148 x
   0.699) / 0.828. Their last layer puts 0.55-0.58 of its attention on the last visit's observation for retrace queries,
   0.22 for other short gaps, and 0.00-0.01 for long-gap and wrap-only revisits (path model: 0.81 / 0.74 on long gaps).
   Linear probes find no global count in their residual stream (unwrapped displacement R^2 <= 0.10; normalised count
   P_t / t <= 0.21), and a displacement code over the recent window that degrades with the window (R^2 0.96 at 4
   moves, 0.77 at 64, 0.37 at 256).
5. **ANALYSIS (route A is what three index layers learned on the ring).** On the ring (T = 128, fresh seeds 2-7), one
   index layer learns exactly two rules: retrace (0.68-0.80) and, when the walk is biased, the one-lap clock (0.987-1.000);
   every other revisit is at the floor (0.49-0.59). Two layers add segment composition that decays with segments (0.89 /
   0.80 / 0.68 at 3-4 / 5-8 / 9-16 segments, p_plus 0.5). Three layers are flat in segments (0.997 / 0.993 / 0.972) and
   their residual stream carries a global count (P_t / t R^2 0.69, P_t 0.60, against 0.18 / 0.09 at one layer): the
   count route is found on the ring and not on the torus.
6. **ANALYSIS (the 0.75 dip, descriptive only).** One-layer index accuracy across p_plus is the share-weighted sum of
   those two rules and the floor: a1 = 0.709 / 0.678 at p_plus 0.5 / 0.75 is reproduced from class shares and
   class accuracies. The dip is the retrace share falling (0.72 -> 0.49) before the clock's share is large (0.18). The
   pre-registration attached no mechanism sentence to the dip; this is a post hoc decomposition, not a verdict.
7. **ANALYSIS (Dyck).** One index layer predicts a closer only when its opener is the previous bracket (0.998), and is
   near chance (0.61-0.64) once one balanced block intervenes; each further layer extends the number of blocks it skips
   (two blocks: 0.78 / 0.89 / 0.93 at 2 / 3 / 4 layers), with a residual dependence on distance. Dyck has a BOS token, so
   the count route's normaliser is available; the data are a mixture and do not separate the routes cleanly.
8. **Proposed paper text** (Section 5) and **five GPU experiments** that test the account (Section 6). The decisive one varies
   the walk's run length at fixed T: route B predicts accuracy is a function of segments spanned, route A a function of
   T and the gap. A CPU eval-only version on the stored models
   (Section 4.4) agrees: retraces are solved at any gap, and at a fixed gap of 16-63 moves 4L scores 0.98 / 0.87-0.93 /
   0.72-0.80 / 0.64-0.70 at 2 / 3-4 / 5-8 / 9-16 segments, with a smaller residual dependence on the gap.

---

## 2. What is being explained (ESTABLISHED)

| task (trained and tested at the same length / depth) | path, 1 layer | index 1 / 2 / 3 / 4 layers | rate | file |
|---|---|---|---|---|
| Dyck-2, L32 D12, A2f (floor 0.594) | 0.996 | 0.648 / 0.889 / 0.964 / 0.982 (RoPE) | ~3 | `DYCK_MDEPTH_RESULTS.md` |
| ring 32, T = 128, p_plus 0.5 (fresh seeds 2-7) | 1.000 | 0.710 / 0.949 / 0.998 | 3 | `CANCEL_RESULTS.md`, `docs/audits/2026-09-27/fresh_seeds.txt` |
| ring, p_plus 0.75 / 0.9 | 1.000 | 0.677 / 0.962 / 0.998; 0.824 / 0.987 / 0.999 | 3 (0.9 knife-edge) | same |
| ring, p_plus 1.0 | 1.000 | 1.000 | 1 | same |
| torus 64x64, T = 1024 (n = 4) | 1.000 | 0.719 / -- / -- / 0.860; loop x4 0.864 | > 4 | `FACTORIAL_RESULTS.md` |

Torus strata (registered secondary; floors post hoc, `docs/paper/figs/fig_factorial_strata_out.txt`): index 1 layer /
4 layers / loop: short gap 0.769 / 0.943 / 0.938 (retrace floor 0.912), long gap 0.480 / 0.458 / 0.507, wrap-only
0.482 / 0.457 / 0.509 (constant floors 0.515 / 0.517). Every index run STALLED; budget-scoped (900 epochs torus,
300 ring, 1x Dyck ladder budget; at 3x budget four index layers reach Dyck's ceiling, 0.997-0.998).

---

## 3. Theory

### 3.1 What a revisit prediction requires (THEORY, elementary)

Tokens interleave moves and observations, [a_1, o_1, a_2, o_2, ...]; o_t is predicted at the token a_t. Let P_t be the
unwrapped displacement after move t (an integer in 1D, a pair in 2D). The map is redrawn per evaluation, so the only
source of o_t is an earlier observation token o_s at the same cell: P_s = P_t (mod N per axis). Because the map is fixed
within a walk, *any* earlier visit will do; no recency tie-break is needed.

A causal attention score between query t and key s in an index-RoPE layer is

    score(t, s) = q(h_t)^T R((t - s) Omega) k(h_s),

so position enters only through the offset t - s, and the content h_t, h_s are the residual streams. The target predicate
P_t - P_s = 0 (mod N) is a property of the *interval* (s, t]. There are two ways to make the score depend on it:

- **(A) absolute code**: write a function of P_t into every h_t, so that q(h_t)^T k(h_s) peaks at P_s = P_t (mod N).
  This requires the prefix sum P_t in the residual stream.
- **(B) relative return set**: write into h_t a description of the set of offsets d <= W at which the recent path returns
  to the current cell, and let the RoPE term select those offsets: score(t, t - d) = sum_c Re(alpha_c(t) e^{i omega_c d})
  approximates the indicator of the return set on [1, W]. This needs only the last W moves, but the query must encode
  their return structure.

### 3.2 Path integration supplies the prefix sum (THEORY, exact)

MapWM rotates queries and keys by theta_t = omega * cumsum(Delta(x)). With observation steps near zero and move steps
a fixed vector per move (MapFormer's learned solution), theta_t = omega P_t and score(t, s) = q^T R(omega (P_t - P_s)) k:
route (A) with the prefix sum computed outside attention, exactly (a float32 cumsum), for every gap and every T. In a channel
whose frequency times the learned move step is a multiple of 2 pi / N, the score is N-periodic in P, and wrap-around costs
nothing (the slowest default frequency has period N moves at unit step; Prop. 1 of the draft concerns which step
tables represent this). One layer suffices; the stored path models show
it (same-cell attention 0.81 / 0.74 on long-gap torus revisits, 0.66 / 0.58 on wrap-only; Section 4.2).

### 3.3 Route A for index attention: depth 2 in principle, precision in practice (THEORY)

*Construction (upper bound on depth).* Layer 1: one head with constant query and key attends uniformly to the prefix; its
value carries the move vector on move tokens, so its output is P_t / n_t with n_t = 2t the prefix length. A second
feature proportional to 1 / n_t is available in the same layer (for example, a head with a fixed self-attention bonus
beta has self-weight e^beta / (e^beta + n_t - 1)). The MLP rescales to P_t and forms periodic features cos(2 pi k P_t / N),
sin(2 pi k P_t / N). Layer 2 matches q(h_t)^T k(h_s) on those features. This is Liu et al.'s (2023) Lemma 6 (one layer
simulates a mod-n counter: prefix average + MLP mod) followed by a match, and Yao et al.'s (2021) depth-2 Dyck construction
(counter in layer 1, proximity-based depth matching in layer 2; as summarised in Liu et al., App. A.5). In RASP terms
the task has depth 2 in 1D and in 2D alike (two counts are two value dimensions of one head) and with or without wrap
(the mod is in the MLP). **So the measured rates (3, 3, > 4) are not representational depth bounds.**

*What the construction costs.*
- *A normaliser.* Lemma 6 uses a padding token and an absolute position channel gamma_t = log(2T - t) to undo the 1/t of
  causal averaging. Our torus and ring sequences have neither (no BOS; RoPE is relative); Dyck has a BOS. The model must
  improvise the 1/n_t feature (it does: 1/t is linearly decodable at R^2 0.67-0.89 at the best pass of every multi-layer index model,
  Sections 4.3, 4.5).
- *Precision.* Adjacent cells differ by 1/n_t in the averaged code: 1/256 at the end of a ring walk (T = 128), 1/2048 per
  axis at the end of a torus walk (T = 1024).
- *Width and weights.* Lemma 6's mod gate uses MLP width 4 n T and weights up to 4 n T + 2 (width O(n) only with
  periodic activations). Read as an upper-bound construction: 32 x 256 for the ring, 64 x 2048 per axis for the torus
  (16x per axis, two axes); our MLP has width 512.
- *Sensitivity.* For a revisit at gap g, flipping any one of the g moves in (s, t] moves the target cell, while flips before
  s do not; the prediction's sensitivity grows with g. Hahn and Rofin (2024, Cor. 5 and Thm. 6) show that representing a
  function whose average sensitivity grows faster than sqrt(n) log n needs parameter norms or layer-norm blow-up that grow
  with input length, and that such minima are sharp. Long-gap torus revisits (g in the hundreds against sqrt(n) ln n ~ 345 at
  n = 2048 tokens) approach that regime; short-gap revisits (g < 16 moves) are far from it. This cost does not by itself
  separate the ring from the torus (ring gaps reach 128 moves against sqrt(n) ln n ~ 89 at n = 256); precision, width and
  the second axis do. Their theorem is for Boolean inputs under the uniform distribution; we use it for direction only.

*Prediction.* Route A is found when T and the dimension are small and a normaliser is easy (ring at T = 128; Dyck with a
BOS at length 32) and not when T is large (torus at T = 1024). Where it is found, accuracy does not depend on how many
segments a revisit spans.

### 3.4 Route B: bounded return rules, extended by depth (THEORY, with an empirical reach)

*One layer.* Two rules need no prefix sum.
- *Retrace.* While the current run of moves reverses the previous run, the walk returns along its own track: after j
  steps of reversal the cell is the one seen 2j moves (4j tokens) earlier. The query must encode j (the current run's
  length, a local average over the last <= 10 moves) and rotate its RoPE phases by the matching offset (4j tokens). This is a bounded-window rule
  (runs are at most 10 moves); its sensitivity is bounded by the two runs involved.
- *Clock.* When the walk never reverses (p_plus 1.0) P_t = t, and the cell visited N moves earlier is the current cell:
  a constant offset, which RoPE represents in one head. Position equals index; there is nothing to integrate.

*Depth.* A return that spans k segments (maximal runs of one action) is a function of those k segments' directions and
lengths. To select its offset the query needs a summary of k segments; segment boundaries sit at content-dependent
offsets, so fetching the summary of the segment before the last is a pointer hop. With two heads a layer can make a
bounded number of hops; stacking layers extends the reach. Two bounds bracket the reach: hops compose linearly if each
layer adds one segment, and logarithmically if layers double their pointers (Liu et al.'s Theorem 1 gives O(log T)
depth for any semiautomaton by such doubling; for an Abelian group like ours Lemma 6's depth-1 route exists, which is
route A). We have only L = 1 and 4 on the torus, so the shape (linear or logarithmic) is open (experiment E2).

*Prediction.* At fixed depth, accuracy falls with the number of segments spanned, and depends on T or the gap only
through them. The depth needed to solve a task is set by the segment counts of its revisits: on a 2D walk at T = 1024,
long-gap revisits span 17 or more segments in 98.5% of cases (median 37; wrap-only: 99%, median 65; Section 4.1),
far beyond what 4 layers reach.

### 3.5 What the two routes predict for each rate (THEORY, checked in Section 4)

| task | route B covers | route A | predicted rate | checked |
|---|---|---|---|---|
| ring, p_plus 1.0 | everything (clock: constant offset) | not needed | 1 | 4.5: lap class 1.000 at 1 layer |
| ring, p_plus 0.5-0.9, T = 128 | retrace + clock at 1 layer; 3-8 segments partly at 2 | ring is small: count learnable | 3 (count found at 2-3) | 4.5: 3 layers flat in segments, count decodable |
| Dyck-2, L32 D12 | adjacent opener at 1 layer; blocks to skip grow with depth | BOS available | ~3-4 | 4.6: block dependence shrinks with depth |
| torus, T = 1024 | retrace; ~4 segments at 4 layers | not found | > 4, growing with segments | 4.1-4.4 |

### 3.6 Assumptions and limits of the theory

- Route A's depth-2 construction is an upper bound on the depth needed and an existence statement; it says nothing about
  what SGD finds. Route B's reach is measured, not derived; the per-layer hop model is a SPECULATION about the
  mechanism, consistent with the attention and probe data.
- The two routes are not exhaustive (a model can mix them, as on the ring at 2 layers and on Dyck).
- The sensitivity argument transfers Hahn and Rofin's Boolean, uniform-input theorem to structured walks; it predicts a
  direction (long gaps harder), not a threshold.
- Linear probes bound what attention can read linearly from LN(h); a nonlinear code could exist undetected. The torus
  models' retrieval failure at long gaps (attention mass 0.00-0.01 on the cell) does not depend on the probes.

---

## 4. Checks run (ANALYSIS; post hoc, eval-only, CPU)

### 4.0 Reproduction (every check passed before any new readout)

- Torus (`torus_depth.py`): the FACTORIAL eval stream re-generated (env seed 10000, np seed 1234 + s, 100 walks); overall
  accuracy per run equals `FACTORIAL_{L1,L4,LOOP}_STRATA.json` "all" to 4 decimals for all 16 runs; the re-implemented
  forward (to expose attention and residuals) equals the model's logits (max |diff| 0.0).
- Ring (`ring_depth.py`): the CANCEL eval stream (train_cancel.evaluate: np seed 0, env seed 10000, 200 walks); overall
  accuracy equals each run's `eval.json` "128" for all 128 runs (asserted < 2e-3; printed values equal to 4 decimals).
  Only fresh seeds 2-7 are read below.
- Dyck (`dyck_distance.py`): A2f recomputed per run; arm means equal `DYCK_MDEPTH_RESULTS.md` (RoPE 0.648 / 0.889 / 0.964 /
  0.982, PoPE 0.640 / 0.850 / 0.946 / 0.971, MapWM 0.996 / 1.000).

### 4.1 Torus: which revisits depth buys (`torus_depth.py`, `summarize_torus.py` -> `out/torus_depth_out.txt`)

Per revisit: s = the last visit to the cell; *retrace* = the current run reverses the previous one and has not overshot it
(then s = t - 2 cur, the one-level retrace rule of the floors); *segments* = maximal single-action runs among moves
s+1..t. Mean of per-seed accuracies (n = 4 seeds, range in brackets); share = fraction of all revisits.

| revisit class | share | index 1L | index 4L | loop x4 | path 1L |
|---|---|---|---|---|---|
| short gap, retrace | 0.680 | 0.833 (0.79-0.86) | 0.996 (0.99-1.00) | 1.000 | 1.000 |
| short gap, not retrace | 0.148 | 0.475 | 0.699 (0.51-0.77) | 0.655 | 1.000 |
| ... 3-4 segments | 0.031 | 0.477 | 0.926 (0.74-0.99) | 0.977 | 1.000 |
| ... 5-8 segments | 0.056 | 0.476 | 0.735 (0.52-0.81) | 0.645 | 1.000 |
| ... 9-16 segments | 0.052 | 0.474 | 0.561 (0.38-0.64) | 0.499 | 1.000 |
| ... 17+ segments | 0.009 | 0.472 | 0.493 | 0.502 | 1.000 |
| long gap (>= 128) | 0.090 | 0.480 | 0.458 | 0.507 | 1.000 |
| wrap-only | 0.083 | 0.482 | 0.457 | 0.509 | 1.000 |

Gap in moves (plain revisits): index 4L / loop 1.000 / 1.000 at 2-7, 0.978 / 0.986 at 8-31, 0.614 / 0.551 at 32-127,
0.458 / 0.506 at 128-511. One layer: 0.935 at 2-7, 0.681 at 8-31, floor beyond.

*Segments at a fixed gap* (plain revisits, gap 32-127 moves; pooled over seeds):

| segments spanned | n | index 1L | index 4L | loop x4 |
|---|---|---|---|---|
| 2 | 240 | 0.412 | 0.874 | 0.996 |
| 3-4 | 503 | 0.483 | 0.832 | 0.911 |
| 5-8 | 4565 | 0.464 | 0.688 | 0.575 |
| 9-16 | 6569 | 0.474 | 0.560 | 0.499 |
| 17+ | 1202 | 0.472 | 0.493 | 0.502 |

At gap 8-31 the same: 4L 0.993 / 0.940 / 0.818, loop 1.000 / 0.987 / 0.768 at 2 / 3-4 / 5-8 segments.

*Accounting for "just above retrace".* The short-gap retrace floor is (0.680 x 1 + 0.148 x ~0.5) / 0.828 = 0.911
(measured 0.912). Four index layers: (0.680 x 0.996 + 0.148 x 0.699) / 0.828 = 0.943 (measured 0.943). Depth adds the exact
retrace rule plus partial returns over 3-8 segments; that is the whole of its short-gap margin over retrace.

*Long-gap and wrap-only revisits span many segments*: 98.5% of long-gap revisits span 17 or more (median 37, minimum
11), and 99% of wrap-only revisits (median 65). They are *below the constant floor* (0.458 / 0.457 vs 0.515 / 0.517) with high NLL (2.1-4.8 in
`FACTORIAL_L4_STRATA.json`): the deep models predict a specific wrong object rather than abstaining to blank.

### 4.2 Torus: where index attention looks (same script)

Attention at the revisit query (token a_t), per pass and head (h0 / h1), mean over revisits and seeds. *same* = mass on
earlier observation tokens of the same cell (uniform-attention chance 0.001-0.003); *last* = mass on the last visit's
observation and arrival tokens.

| model, final pass | retrace: same / last | short, not retrace: same | long gap: same | wrap-only: same |
|---|---|---|---|---|
| index 1L | 0.05 / 0.10 (h0), 0.06 / 0.12 (h1) | 0.00 | 0.00 | 0.00 |
| index 4L, layer 4 | 0.55 / 0.56, 0.58 / 0.58 | 0.22 / 0.22 | 0.01 / 0.01 | 0.00 / 0.00 |
| loop x4, pass 4 | 0.50 / 0.52, 0.49 / 0.50 | 0.20 / 0.20 | 0.00 / 0.00 | 0.00 / 0.00 |
| path 1L | 0.83 / 0.70, 0.76 / 0.65 | 0.82 / 0.76 | 0.81 / 0.74 | 0.66 / 0.58 |

In the deep index models retrieval happens in the last pass only (layers 1-2: same-cell mass <= 0.04 everywhere; layer 3
0.21-0.33 for retrace), and on retraces all of it is on the last visit (same = last): they find the mirror point, not
the cell. Split by outcome, the final pass's max-over-heads same-cell mass is 0.75 on correct and 0.30 on wrong retrace
predictions (4L), 0.45 vs 0.21 on other short gaps, and 0.02 vs 0.02 on long gaps: at long gaps there is no retrieval to
get right or wrong.

### 4.3 Torus: no global count (`torus_probe.py` -> `out/torus_probe_out.txt`; `torus_depth.py` probes)

Ridge probes from the residual stream at move tokens (train walks 0-29, test 30-39, t >= 16; R^2 out of sample, mean of
4 seeds, x and y averaged). Best pass shown.

| target | index 1L | index 4L | loop x4 | (ring index 3L, p_plus 0.5, for comparison; 4.5) |
|---|---|---|---|---|
| unwrapped displacement P_t | -0.06 | 0.10 | -0.01 | 0.60 |
| normalised count P_t / (t+1) | -0.03 | 0.21 | 0.06 | 0.69 |
| 1 / (t+1) | -0.01 | 0.78 | 0.74 | 0.89 |
| displacement over the last 4 moves | 0.90 | 0.96 | 0.95 | 0.97 |
| ... last 16 | 0.47 | 0.93 | 0.86 | 0.86 |
| ... last 64 | 0.09 | 0.77 | 0.56 | 0.70 |
| ... last 256 | -0.01 | 0.37 | 0.14 | -- |
| torus phase cos / sin (2 pi P / 64) | -0.02 | -0.01 | -0.01 | (ring phase 0.04-0.10) |

The deep torus models know elapsed time (1/t) but carry no normalised count; they carry a recent-displacement code whose
error in cells grows with the window (rmse = sd x sqrt(1 - R^2) with the window sds of `out/walk_stats_out.txt`: 0.5 cells
at 4 moves, 1.8 at 16, 7.1 at 64, 25 at 256 for 4L). That is route B's signature. (The path model's residual stream
carries no position either, as expected: its position is in the rotation.)

### 4.4 Torus: run length moved at evaluation (`torus_runlen.py` -> `out/torus_runlen_{L1,L4,LOOP,PATH}_out.txt`)

The stored FACTORIAL models re-evaluated (no training) on walks whose run length is fixed at 2, 5 or 10 moves instead
of drawn from 1-10 (within the training support of run lengths, but a shift in walk statistics; held-out map; 60 walks
per seed for 1 layer, 30 for 4L / loop, 10 for path). Plain revisits, pooled over the 4 seeds; cells with >= 50 revisits.
Rows: segments spanned; columns: gap in moves (2-15 / 16-63 / 64-127 / 128+). Index 4 layers (loop in brackets):

| condition | 2 segments (retrace) | 3-4 | 5-8 | 9-16 | 17+ |
|---|---|---|---|---|---|
| training walk (1-10) | 0.999 / 0.978 (1.000 / 1.000) | 0.978 / 0.925 (0.969 / 0.978) | 0.875 / 0.759 / 0.527 (0.838 / 0.675 / 0.418) | -- / 0.635 / 0.539 (0.512 / 0.505) | 0.440 / 0.459 (0.481 / 0.509) |
| runs of 2 | 0.999 (1.000) | 0.934 / 0.898 (0.884 / 0.966) | 0.836 / 0.783 (0.744 / 0.621) | 0.702 (0.526) | 0.580 / 0.524 / 0.461 (0.483 / 0.489 / 0.493) |
| runs of 5 | 0.997 / 0.979 (1.000 / 1.000) | 0.997 / 0.912 (0.999 / 0.982) | 0.803 / 0.819 (0.683 / 0.511) | 0.701 / 0.654 (0.517 / 0.503) | 0.551 / 0.465 (0.509 / 0.509) |
| runs of 10 | 1.000 / 0.978 (1.000 / 0.995) | 0.867 / 0.662 (0.939 / 0.382) | 0.715 / 0.605 (0.556 / 0.503) | 0.557 / 0.449 (0.511 / 0.494) | 0.453 (0.506) |

(Each cell lists the populated gap columns left to right.) Overall plain accuracy of 4L: 0.901 on the training walk,
0.843 / 0.858 / 0.841 with runs of 2 / 5 / 10; path 1.000 in every cell of every condition.

- *Two-segment returns are gap-invariant for deep index models*: retraces at gaps of 16-63 moves (runs of 10) score
  0.978 / 0.995. One index layer's retrace is window-limited instead: 0.83-0.93 at gaps of 2-15 moves and 0.44-0.46 at
  16-63 in every condition (`torus_runlen_L1_out.txt`).
- *Segments set the level, the gap trims it*: down a column of fixed gap (16-63 moves) 4L falls from 0.98 (2 segments) to
  0.87-0.93 (3-4), 0.72-0.80 (5-8), 0.64-0.70 (9-16) and 0.58 (17+, runs of 2); within a segment bin across gaps it moves
  by 0.02-0.2 (0.35 in one cell of 55 revisits). The loop is sharper: 0.88-1.00 up to 4 segments at gaps below 64, at or near the floor (0.50-0.53) from
  9 segments at any gap.
- Shorter runs put more segments into each gap and lower overall accuracy (0.843 with runs of 2), as route B predicts.
  Runs of 10 also lower it (0.841), through 3-8 segment returns at gaps of 16-127 moves (4L 0.60-0.87), so the
  gap is not irrelevant.
- Reading: route B with a reach of about 4 segments at 4 layers (partial to 8), plus a weaker gap dependence. Because
  the conditions shift the walk distribution, this is suggestive; E1 (Section 6) trains on each condition.

### 4.5 Ring: rules, segments and a count (`ring_depth.py`, `summarize_ring.py` -> `out/ring_depth_out.txt`)

Classes: *retrace* as on the torus; *lap* = the displacement over the last 32 moves is 0 mod 32 (a fixed one-lap offset
lands on the cell; the clock); *neither* = the rest, split by segments. Fresh seeds 2-7, mean (range).

p_plus 0.5:

| class | share | index 1L | index 2L | index 3L | path 1L |
|---|---|---|---|---|---|
| retrace | 0.716 | 0.797 | 0.987 | 1.000 | 1.000 |
| lap, not retrace | 0.035 | 0.497 | 0.977 | 1.000 | 1.000 |
| neither, 3-4 segments | 0.127 | 0.487 | 0.891 (0.72-0.94) | 0.997 | 1.000 |
| neither, 5-8 | 0.097 | 0.489 | 0.802 (0.55-0.87) | 0.993 | 1.000 |
| neither, 9-16 | 0.025 | 0.485 | 0.678 (0.48-0.75) | 0.972 | 1.000 |
| wrap-only | 0.118 | 0.503 | 0.819 | 0.989 | 1.000 |

p_plus 0.75 / 0.9 (index 1L): retrace 0.692 / 0.677, lap 0.987 / 0.994, neither 0.487 / 0.592. p_plus 1.0: every revisit
is a lap revisit, 1.000 at every depth.

- One index layer learns exactly retrace and (when it is common) the clock, and nothing else.
- Two layers compose segments, decaying with their number (route B).
- Three layers are nearly flat in segments, and the residual stream at layer 2 carries a global count: P_t R^2 0.60,
  P_t / t 0.69, 1/t 0.89 (one layer: 0.09 / 0.18 / 0.14; two: 0.40 / 0.60 / 0.67). At p_plus 0.5 there is no drift, so P_t
  is not a proxy for t here. Route A is found on the ring.
- Its last-layer attention reaches the cell on wrap-only revisits (same-cell mass 0.36 / 0.38) and other non-local
  revisits (0.17 / 0.18), where 1-2 layers do not (0.00-0.14). The probes find no linear phase code (cos / sin R^2 <= 0.10),
  so how the count is matched across the wrap is not identified.

*The non-monotone a1(p) (descriptive).* With per-class accuracies at each p, a1 = sum over classes of share x accuracy:
p_plus 0.5: 0.716 x 0.797 + 0.035 x 0.497 + 0.249 x 0.487 = 0.709 (observed 0.710); 0.75: 0.493 x 0.692 + 0.179 x
0.987 + 0.329 x 0.487 = 0.678 (0.677); 0.9: 0.214 x 0.677 + 0.530 x 0.994 + 0.256 x 0.592 = 0.823 (0.824). The dip at
0.75 is retrace's share falling faster than the clock's share rises (an identity given the class accuracies; the
explanatory content is that one layer implements these two rules and only these). Registered as "no mechanism sentence";
this is a post hoc decomposition and should be presented as such or not at all.

### 4.6 Dyck-2: blocks to skip (`dyck_distance.py`, `dyck_crosstab.py` -> `out/dyck_distance_out.txt`, `out/dyck_crosstab_out.txt`)

Closing accuracy (A2f's per-prefix renormalised legal-closer probability, closer-feasible prefixes of the registered L32
D12 cell) by the number of complete balanced blocks between the top opener and the prefix; mean over 8 seeds.

| blocks to skip (n prefixes) | RoPE 1L | 2L | 3L | 4L | MapWM 1L |
|---|---|---|---|---|---|
| 0 (3682) | 0.998 | 0.999 | 0.999 | 0.999 | 0.999 |
| 1 (5839) | 0.640 | 0.925 | 0.984 | 0.991 | 0.997 |
| 2 (572) | 0.641 | 0.778 | 0.885 | 0.934 | 0.993 |
| 3-4 (161) | 0.613 | 0.760 | 0.860 | 0.925 | 0.993 |

Cross-tab (`dyck_crosstab_out.txt`): at a fixed distance of 16-31 tokens, 3 layers score 0.966 / 0.896 / 0.853 at 1 / 2
/ 3-4 blocks; at a fixed 1 block, 2 layers score 0.970 / 0.948 / 0.835 at distance 2-7 / 8-15 / 16-31. Both blocks and
distance matter for the index models; the path model is flat in both (0.987-0.998). Reading: a mixture of the routes,
consistent with but not specific to the account. 93% of scored prefixes need at most one block, which is why three
layers already reach 0.964 in aggregate. By stack depth, one layer falls from 0.88 at depth 1 to 0.60 at depth 9-10;
two layers or more are nearly flat.

---

## 5. Proposed paper text (Section 5)

Replace the paragraph "What depth buys index attention" and the closing positioning paragraph with the following (labels
for the editor: the first two sentences restate registered numbers; the rest is post hoc analysis and theory and must be
marked "(analysis)" in the paper as the draft already does for floors):

> **Why the rate differs by task.** A revisit asks for an earlier observation at the same cell, that is, for a key s
> whose moves since s sum to zero. Path integration puts the prefix sum into the rotation, so one layer compares
> P_t - P_s directly at any gap. Index attention sees only t - s, and must compute the predicate in its residual stream.
> It can do so in two ways. A global count (uniform attention over the moves, rescaled, then matched) needs only two
> layers in principle (Liu et al. 2023; Yao et al. 2021), but its precision must grow with the walk length and the target
> is sensitive to every move since the last visit (Hahn and Rofin 2024). A local rule needs no count but only reaches
> returns that span a few direction runs: undoing the last run (retrace) or, when the walk never reverses, a fixed
> one-lap offset (a clock). Each extra layer reaches returns across more runs. The stored models show both routes
> (analysis; App. F). One index layer learns exactly retrace and the clock: on the ring it scores 0.68-0.80 on retraces,
> 0.99-1.00 on one-lap returns once they are common, and the floor on everything else, so at p_plus 1.0, where every
> revisit is a one-lap return, it matches the path layer. Three index layers on the ring are flat in the number of runs
> a revisit spans and carry a global count in the residual stream (linear-probe R^2 0.69 for P_t / t, against 0.18 at
> one layer). On the torus at T = 1024 no index model carries a count (R^2 <= 0.21); four layers or the loop solve
> retraces (0.996 / 1.000) and decay with the number of runs spanned (0.93 / 0.98 at 3-4 runs, 0.74 / 0.65 at 5-8,
> 0.56 / 0.50 at 9-16), at a fixed gap as well as overall, and their last layer attends to the remembered cell for
> retraces (0.55-0.58 of its mass) but not at long gaps (0.00-0.01; path layer 0.81). Nearly all long-gap revisits span 17 or more
> runs (median 37). Depth therefore buys index attention a local reach measured in direction runs; the torus exchange rate is
> "more than four" because its long-range revisits lie beyond that reach and the global count is not found.

Figure suggestion (Fig. 2a companion or App. F): accuracy against runs spanned (2, 3-4, 5-8, 9-16, 17+) for index 1L,
4L, loop and path on the torus, with the ring's 1L / 2L / 3L beside it; caption numbers from Sections 4.1 and 4.5.

Two sentences for Limitations: "The explanation of the exchange rate is post hoc: the class splits, attention masses and
probes were computed on stored checkpoints after the registered results were read, and the reach of route B was measured
only at one and four layers on the torus. The routes are distinguished by probes and by accuracy against runs spanned,
not by an intervention."

---

## 6. Proposed GPU experiments (each with a registered prediction)

Costs from stored logs: torus T = 1024 at 900 epochs ~3.1 s/epoch per index layer (L1 50 min, L4 3.2 h per run, two
jobs per GPU); ring T = 128 at 300 epochs 9 min (1 layer) to 25 min (3 layers).

**E1. Run length at fixed T (decisive between routes; cheapest).** Torus T = 1024, walks with run length fixed at 2, 5
or 10 moves (or drawn 1-3 / 8-10), index RoPE 4 layers and path 1 layer, trained and tested on each. Prediction (route
B): the accuracy-against-runs-spanned curve of the index model is the same in every condition, so short runs (more
segments per gap) lower overall accuracy and long runs raise it; route A predicts dependence on T and gap only. Path
flat at 1.000. 3 conditions x 2 arms x 4 seeds: ~12 x 3.2 h + 12 x 0.8 h ~ 48 GPU-h; at T = 512 ~ 24 GPU-h. A CPU
eval-only version on the stored models is Section 4.4 (distribution-shifted, so it can only suggest).

**E2. Reach against depth.** Torus T = 1024, index RoPE at 1, 2, 3, 4, 6, 8 layers, n = 4, all retrained in one batch
(rule 12). Readout: the largest runs-spanned bin with accuracy >= 0.9, per depth. Linear reach predicts ~L + 1 runs at L
layers; doubling predicts ~2^L. Decides whether any practical depth reaches the torus (long gaps need 17+). Cost: 24
layer-units per seed x 47 min x 4 seeds ~ 75 GPU-h; trimmed to {2, 4, 8} ~ 44 GPU-h.

**E3. Give index attention a normaliser.** Add a BOS token and Lemma 6's absolute channel log(2T - t) (or t / T) to
index RoPE at 2 and 4 layers on the torus. If route A is blocked by the missing normaliser, two layers should then solve
long-gap and wrap-only revisits, and probes should find P_t / t; if precision or sensitivity block it, nothing changes.
2 arms x 2 depths x 4 seeds ~ 16 runs ~ 35 GPU-h. A cheap first step: the ring at T = 128 with and without BOS at 2
layers (does the count appear at 2 instead of 3? ~3 GPU-h).

**E4. The ring as a function of T.** Ring p_plus 0.5, T in {128, 256, 512, 1024} at matched length, with the ring size
scaled so the wrap-only share is fixed (N ~ sd of P_T), index 1-4 (6) layers, n = 4. Route A predicts the rate grows with
T (precision 1/T, sensitivity ~ gap) and the count probe weakens; route B predicts the rate grows with runs spanned. With
E1 this separates the two. ~60 GPU-h full; T in {128, 512} x L in {2, 3, 4, 6} ~ 15 GPU-h.

**E5. Dyck at matched depth across D.** Dyck at L32 / L64 with D in {4, 8, 12, 16}, index 1-4 layers: does the rate
rise with D (blocks to skip) at fixed length? ~20 GPU-h. Lower priority: Dyck mixes the routes (Section 4.6).

Priority: E1 (cheap, decisive, ties to the paper's main exchange-rate claim), then E3's ring step (3 GPU-h), then E2.
Each needs a pre-registration and a results-blind code check (rule 29) before launch; pilots on seeds outside the batch.

---

## 7. Caveats

- Everything in Sections 4.1-4.6 is post hoc and eval-only; it describes stored models, all of whose index runs are
  STALLED at their budget. "Reach" is the reach of these budgets.
- n = 4 seeds on the torus (complete separations only in the registered contrasts); seed-to-seed spread in the 4-layer
  model is large on non-retrace classes (0.51-0.77).
- Segment and gap are correlated in natural walks; the fixed-gap tables and Section 4.4 reduce but do not remove that.
- Linear probes can miss a nonlinear code; the attention readout (no mass on the cell at long gaps) does not rely on them.
- The ring eval stream overlaps training batch 0 of seed-0 runs (`CANCEL_RESULTS.md` correction); only seeds 2-7 are read.
- Liu et al.'s and Yao et al.'s constructions use absolute positions or a padding token, dense state supervision (Liu), and
  hard or near-hard attention; our task supervises the state only through retrieval. They bound depth; they do not predict
  what is learned. Yao et al. (2021) is cited through Liu et al.'s App. A.5 (not in `papers/`).
- Hahn and Rofin's bounds are for Boolean functions under uniform inputs; we use them for direction only.
