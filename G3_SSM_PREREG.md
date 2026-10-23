# G3_SSM (G-3): the sign test beyond attention -- pre-registration

**Status: PREPARED, NOT REGISTERED, NOT LAUNCHED (2026-10-23).** Before launch: (1) an independent results-blind code check
(rule 29) whose findings go into an Amendment 1 here; (2) the user's decisions in Sec. 10; (3) commit every file in Sec. 0
(the driver refuses an uncommitted or modified tree). The driver md5-guards this file, so it is final at launch.

Source of the design: `docs/RUNBOOK_NEXT_EXPERIMENTS.md` row G-3, `docs/paper/expansion/PAPER_LEVEL.md` #1,
`docs/paper/expansion/SYNTHESIS_DRAFT2_PLAN.md` row G-3. Code (all new; no existing module edited):

| file | what |
|---|---|
| `model_ssm_sign.py` | one-layer diagonal complex SSM, rotary transition, step classes, sequential reference recurrence |
| `train_ssm_sign.py` | trainer (the attention ring recipe verbatim), diagnostics, `--init-built` rule-14 continuations |
| `construct_ssm_sign.py` | exact non-negative constructions (Prop. 3 (a), (f), (g)) from a trained signed model, frozen checks |
| `gate_ssm_sign.py` | task gate (rule 11): vocabulary, chance, action-only and full-stream n-grams, retrace rule |
| `analyze_ssm_sign.py` | registered analysis; `--selftest` reaches every label (written eval.json files, through `load()`) |
| `run_ssm_sign.sh` | driver: `lib_driver.sh` (flock, slot picker, md5 guard, artifacts-then-marker), clean-tree checks |
| `test_ssm_sign.py` | unit checks, 39/39 PASS (CPU) |

## 1. Question, and what the batch can and cannot decide

In rotary ATTENTION a non-negative phase step costs learnability, not representability: on the ring (32 cells, +1/-1,
observation-step-free, T = 128, 300 epochs) |.| solves 2/8 at its best rate vs signed 8/8 (`LRSWEEP_RESULTS.md`), exp and
softplus 1/8 at lr 1e-3 (`SIGNROUTE_RESULTS.md`), while an exact non-negative solution exists and is held under training
(`WRAP_EXISTENCE_RESULTS.md`). Grazzi et al. (Thm. 1-3) and Mamba-3 (Sec. 3.2) state only representability for
state-space models (SSMs): real non-negative eigenvalues cannot track parity / Z_m, complex (rotation) eigenvalues can, and
rotations represent Z_m with non-negative angles (2 pi w / m). Mamba-3's angle is Delta_t theta_t with Delta_t > 0 and
theta_t signed (`papers/txt/mamba3.txt:154, 637`), so its sign sits in theta_t; nobody has asked whether a non-negative
rotary step is LEARNED in an SSM.

**G-3 asks: does the learnability cost of a non-negative phase step appear in a one-layer diagonal complex SSM with an
input-dependent rotation?** Two outcomes are informative:
- cost present (|.| fails where signed solves, an exact |.| solution exists and is held): the cost is a property of
  content-dependent rotary phase parameterisations, not of softmax attention;
- no cost: the cost is attention-specific -- a scope boundary (PAPER_LEVEL #1: "the recurrence gives gradient through time").

It cannot decide WHY (the route/wrap accounts of `SEARCH_THEORY.md` are not separated here, as SIGNROUTE Amendment 1 found
for attention), nor anything beyond one layer, one task, two fixed rates and one budget.

## 2. Model as built (`model_ssm_sign.py`)

Per head (2 heads, d_head 64, N = 32 complex channels, value width P = 64):

    H_t = diag(exp(-i omega Delta(x_t))) H_{t-1} + k_t v_t^T,      y_t = Re(conj(q_t)^T H_t),

with k_t, q_t in C^N the (even, odd) pairs of the real projections (Mamba's B_t, C_t; Mamba-3 Prop. 2 writes the same
complex SSM as a real one with 2 x 2 rotation blocks) and v_t the value (Mamba's x_t). The spec's
h_t = exp(i omega Delta(x_t)) h_{t-1} + B x_t is this with omega -> -omega (omega's sign is a free gauge in every arm).
Readout is from Re and Im parts through q_t. y is RMS-normalised per head (Mamba-2/3's norm before the output projection;
without it y grows with the number of past tokens), then o_proj, residual, and the attention model's FFN, LayerNorms and
output head.

**Computation.** `forward` uses the cumsum-of-angles ("RoPE trick", Mamba-3 Prop. 3) form: rotate q and k by their
cumulative angle Phi = omega * cumsum(Delta), take the causal UNNORMALISED product (current token included). It is exact,
not an approximation; `sequential_reference` runs the recurrence step by step in complex128. Rule 19 (`test_ssm_sign.py`):
parallel = sequential to **1.1e-12 relative in float64** over all 5 step classes x 2 step modes x 2 decay settings; float32
parallel vs float64 recurrence **1.3e-4** relative at T = 48 and **8.7e-5 at the 4T evaluation length (1023 tokens, max
|Phi| 7,788 rad)**. Causality and the memoryless arm's independence from the past are checked (exact 0).

**What differs from the attention arms (`model_signroute.ARMS`), and nothing else does.** softmax(QK^T / sqrt(d)) with
attention dropout is replaced by the causal linear kernel; y is RMS-normalised per head (+128 parameters: 204,371 vs
204,243 at vocabulary 19). Embedding, q/k/v/o shapes, FFN, out head, the step map (`SignRouteActionToLie`, the attention
arms' own readout code) and `PathIntegrator` (learned omega, geometric init 2 pi .. 2 pi / grid_size, grid_size = m) are
the attention model's, drawn in the same RNG order: **at a seed every shared tensor (24 of 24) starts bitwise equal to the
attention arm's** (`test_shared_init_matches_attention_arm`, all four step classes).

**Decay: none (|a| = 1), and why.** Mamba-3's transition is e^{Delta A} R(Delta theta) with a separate real decay
e^{Delta A} in (0, 1). It is left out of the registered arms because (i) the attention anchors have no envelope, so a decay
would add a forgetting channel to a cross-architecture comparison; (ii) the question (Prop. 1 / Prop. 3, Grazzi's
representability statement) concerns the unit-modulus rotation part; (iii) retrieval after a long revisit gap (up to 128
moves) competes with any forgetting, so a decay arm's failure would not be attributable to the sign. A Mamba-style
data-dependent decay a_t = exp(-softplus(w . x_t + b)) per head (init a = 0.999) is built (`--decay learned`) and passes
the same parallel = sequential check; it is not in the batch. **Also not Mamba-faithful, declared:** no output gate
silu(z); the step Delta enters only the phase (Mamba's Delta_t also scales the input and the decay).

**Step classes** (applied to s = W_out W_in e, rank 4, as the attention arms): signed s; abs |s|; softplus(s) (Mamba's
Delta parameterisation); exp(s) (LRU-style positive phase); index (every token steps 1: an input-independent rotary SSM,
S4D/LRU-like) as the integration gate. **Steps: `moves`** (registered primary) -- only the two move tokens step, every
other token's Delta is 0 by construction, as in the attention ring anchors (LRSWEEP, SIGNROUTE incl. its exp / softplus
arms); **`all`** (block B) -- every token steps, Mamba-3-faithful; observation tokens are then "stay" inputs that must learn
Delta = 0 (signed, abs) or a full turn (softplus, exp). Moves-only is primary because the inverse-pair algebra of
Prop. 1 applies exactly, the attention numbers it is read against are of this kind, and under `all` exp / softplus carry a
second burden (a full turn for 17 stay tokens) unrelated to the sign.

## 3. Task, and the gate (rule 11)

**Task: retrieval at revisits on the ring Z_m** -- `environment_steps.GridWorldSteps` (moves +1 / -1, p_first 0.5, runs of
1-10, 16 object types + blank with p_empty 0.5, the map REDRAWN per walk), interleaved [a1, o1, a2, o2, ...], loss on
revisited observations only (`train.train`), evaluation `train_cancel.evaluate` on a fresh stream (np seed 10^6, env seed
10000, 400 walks). **Chosen over pure state tracking (predict position mod m) because** it keeps the comparison closest to
the attention ring task -- same environment, tokens, loss mask and evaluation, so only the sequence mixer differs -- and
because it is origin-free and demands the inverse-pair cancellation whose learnability is at stake; position tracking needs
an absolute origin and a decoder over m classes and has no attention anchor on record. "Stay" moves are not added to the
primary task (none exist in the attention anchors); in block B every observation token is a stay input for the phase.

Sizes m = 32 (primary; the attention anchors' ring) and m = 8 (secondary); training T = 128 moves, evaluation T = 128
(matched, registered) and 512 (descriptive, out of distribution).

**Gate (`gate_ssm_sign.py`, CPU, calls the task code; n-grams fit on 2,000 walks np seed 4242, scored on the registered
evaluation stream):**

| m | T | scored | revisit share | chance (blank) | action-only n-gram 1-5 | full-stream n-gram 1-5 | retrace | best floor |
|---|---|---|---|---|---|---|---|---|
| 32 | 128 | 38,847 | 0.759 | 0.496 | 0.496 x5 | 0.496 0.496 0.496 0.554 0.552 | **0.864** | **0.864** |
| 32 | 512 | 192,000 | 0.938 | 0.498 | 0.498 x5 | 0.498 0.498 0.498 0.544 0.542 | 0.809 | 0.809 |
| 8 | 128 | 48,000 | 0.938 | 0.498 | 0.498 x5 | 0.498 0.498 0.498 0.544 0.541 | **0.792** | **0.792** |
| 8 | 512 | 201,600 | 0.984 | 0.487 | 0.487 x5 | 0.487 0.487 0.487 0.502 0.502 | 0.791 | 0.791 |

Token ids all in vocabulary (19); the majority label is blank in every cell. The m = 32 numbers reproduce the floors on record
(n-gram 0.554, retrace 0.860, `docs/audits/2026-10-16/ring_floors_out.txt`). **Needs integration** (pilot, outside seeds):
the memoryless arm scores 0.496 (= chance; 100 epochs, loss 2.08) and the index arm (input-independent rotation) 0.755 at
m = 32 (final loss 0.96, STALLED) -- below the retrace rule -- against signed 0.979-1.000 (Sec. 9). The index arm is also
in the batch at both m (gate GI, Sec. 6).

## 4. Existence before mechanism (rule 14)

Constructions (`construct_ssm_sign.py`; MAIN_APPENDIX2.md E2 = Prop. 3, APPENDIX_theory_v3.md P1), from a TRAINED signed
model, with Theta = omega Delta its angle table and rem = Theta mod 2 pi:
- latents: W_in' sends move j to c0 u_j (c0 = the source's mean move-latent norm; D3's balanced construction, so W_in stays
  near its trained scale) and, with steps = all, every observation embedding to 0 (min-norm solve against the full embedding
  matrix; residual <= 2.7e-15);
- **abs (a):** omega' = max(|omega|, 1), W_out' = rem / omega' / c0;
- **softplus (f):** omega' = 2 pi / ln 2, W_out' = softplus^-1((rem + 2 pi) / omega') / c0 -- an annihilated token steps
  ln 2, exactly one turn;
- **exp (g, audit CPU-7 -- built here for the first time in a model):** omega' = 2 pi, W_out' = ln((rem + 2 pi) / 2 pi) / c0
  -- an annihilated token steps exp(0) = 1, exactly one turn;
- **signed control:** the same latents and omega' = max(|omega|, 1) with the UNWRAPPED angle, so it differs from the abs
  construction only in the wrap. One control serves the three classes.

**Checked (unit tests, random perturbed signed models, both step modes and moves -> all):** angle error mod 2 pi <= 2.2e-6
rad on every token, constraint held (abs >= 0; softplus, exp > 0 on every stepping token), logits equal to the source
(moves; moves -> all) or the signed control (all -> all) to <= 1.2e-4; annihilated tokens step 0.6931 (softplus) / 1.0000
(exp), a full turn to <= 7.8e-7.

**Balance (rule 14's scale clause).** A trained signed SSM has nearly antiparallel move embeddings, so orthogonal latents
need a large W_in: unbalanced, pilot s2183 had max|W_in| 7.6 against the source's 0.50. The construction therefore rescales
W_in -> lam W_in, W_out -> W_out / lam (every step unchanged, checked to <= 9.5e-7) so that max|W_in| = max|W_out|.

**Checked on trained sources (pilots, CPU, the registered evaluation stream):**

| source | class | angle err vs source (rad) | min move step | frozen acc = source | max logit diff | max\|W_in\| = max\|W_out\| (src) | max\|omega\| (src) |
|---|---|---|---|---|---|---|---|
| s2183 signed, 3e-3, acc 1.0000 | abs | 4.2e-7 | 0.0000 | 1.0000 = 1.0000 | 6.6e-4 (scale 21.5) | 1.596 (0.496 / 0.340) | 1.00 (0.68) |
| | softplus | 1.3e-6 | 0.6931 | 1.0000 = 1.0000 | 3.4e-3 | 0.667 | 9.06 |
| | exp | 2.3e-6 | 1.0000 | 1.0000 = 1.0000 | 5.0e-3 | 0.530 | 6.28 |
| | signed control | 7.2e-7 | -4.210 | 1.0000 = 1.0000 | 2.7e-4 | 1.307 | 1.00 |
| s2180 signed, 1e-3, acc 0.9792 | abs | 1.0e-6 | 0.0000 | 0.9792 = 0.9792 | 7.2e-3 (scale 27.5) | 0.526 (0.186 / 0.419) | 2.92 (2.92) |
| | softplus | 1.8e-6 | 0.6931 | 0.9792 = 0.9792 | 1.9e-2 | 0.220 | 9.06 |
| | exp | 1.5e-6 | 1.0000 | 0.9792 = 0.9792 | 1.1e-2 | 0.175 | 6.28 |
| | signed control | 4.8e-7 | -1.764 | 0.9792 = 0.9792 | 3.5e-4 | 0.309 | 2.92 |

The logit differences are float32 cos/sin of larger cumulative angles (softplus / exp moves turn up to 4 pi per step), not
construction error (angle errors <= 2.3e-6). W_in / W_out within 3.2x / 4.7x of the source's (D3's standard: "within about
5x"). **Declared (rule 15):** (f) and (g) fix omega' at 2 pi / ln 2 = 9.06 and 2 pi, far above the trained source's
omega (0.68 at 3e-3; training shrinks it from the 2 pi init); with steps = moves any omega' would do, but the batch uses
the appendix's (f) / (g) as stated.

**Full-step classes (block B).** A trained full-step signed source (pilot s2182, steps = all, 1e-3, acc 0.9992) USES its
observation steps (wrapped |angle| up to 0.595 rad), so zeroing them -- what every construction must do -- loses the map
(frozen 0.4190 in all four classes, the signed control included; exact against each other, angle err <= 1.7e-6). Existence
in the full-step classes is therefore shown, as D4 does for attention, by building a MOVES-only signed code into the
every-token-steps classes (`--steps all`, observation embeddings annihilated): from s2183, frozen 1.0000 = source in abs,
softplus, exp and the signed control; angle err <= 2.2e-6 on every token; annihilated tokens at 0 mod 2 pi to <= 7.8e-7
(stepping 0, ln 2, 1, 0); max logit diff <= 5.0e-3. Block B has no held arm (declared).

**Trainable (pilot, s2183's constructions, 100 epochs, fresh stream; one seed each, anecdote):**

| continuation | lr | max train loss (epoch) | final loss | acc T=128 / 512 | class | D6 LOST? |
|---|---|---|---|---|---|---|
| signed control | 3e-3 | 0.018 (6) | 0.0000 | 1.0000 / 0.9995 | SOLVED | -- |
| abs | 3e-3 | 0.674 (5) | 0.0004 | 0.9998 / 0.9394 | SOLVED | yes (re-found) |
| softplus | 3e-3 | 1.256 (28) | 0.0760 | 0.9830 / 0.9317 | DESCENDING | yes |
| exp | 3e-3 | 1.696 (9) | 0.1665 | 0.9266 / 0.7317 | DESCENDING | yes |
| signed control | 1e-3 | 0.002 | 0.0000 | 1.0000 / 0.9995 | SOLVED | -- |
| abs | 1e-3 | 0.075 | 0.0001 | 1.0000 / 0.9894 | SOLVED | yes, by the 10x rule (0.075 > 10 x 0.002 and > 0.05) |
| softplus | 1e-3 | 0.108 | 0.0003 | 1.0000 / 0.9567 | SOLVED | yes, by the 10x rule |
| exp | 1e-3 | 0.132 | 0.0006 | 1.0000 / 0.9900 | SOLVED | yes, by the 10x rule |

So at the batch rate (3e-3) the warm-up knocks every non-negative construction off its solution while the signed control
stays; at 1e-3 all three are only mildly perturbed and end SOLVED. This is why the 1e-3 continuation (block H2) is in the
batch: at 3e-3 a from-scratch failure could be "the class cannot STAY at a solution at this rate", not only "cannot find
it". On one seed each the D6 rule would read PERTURBED AND RE-FOUND at 1e-3 (it needs n_s - 1 SOLVED and <= 1 LOST for
HELD), so a SEARCH COST joint reading is not assured at either rate.

**In the batch (blocks H and H2, registered):** each signed m = 32 source of block A (lr 3e-3) is constructed into abs,
softplus, exp and the signed control (CPU, by the driver), each evaluated FROZEN before any update (E1) and then trained 100
epochs on a fresh walk stream (data_seed_offset 10^6), cosine with 5% warmup, at lr 3e-3 (H) and, from the SAME built
checkpoints, at lr 1e-3 (H2) (E2). 100 epochs, not WRAP_EXISTENCE's 300, to save cost: LOST is a warm-up event in every
continuation on record (`WRAP_EXISTENCE_RESULTS.md`: lost by epoch 10; here by epoch 5-28).

## 5. Batch

Seeds **2100-2107** (fresh; no run dir in the repo carries them); pilots **2180-2189**, never pooled. Recipe = the attention
ring anchors' (`train_signroute.py` / `train_gauge.py`): T = 128, batch 128, 98 batches, 300 epochs, cosine with 5% warmup
to 10% of peak, AdamW wd 0.05, 1 layer, 2 heads, d 128, rank 4, 3 data workers, dropout 0.1.

**Rates.** The primary rate **3e-3** was fixed from the SIGNED pilots before any non-signed 3e-3 pilot was read: signed at
1e-3 (s2180) was still at loss 0.064 after 300 epochs (STALLED, not SOLVED: G0 at risk), at 3e-3 (s2183) at 0.003 by epoch
70. 3e-3 is also |.|'s best rate in attention (LRSWEEP). **Then the non-signed 3e-3 pilots showed exp unstable at 3e-3**
(s2183: training loss rising from 1.3 to 2.8, above the memoryless 2.08, ending at 0.504 accuracy) **while exp at 1e-3
solved** (s2180), and the 3e-3 continuations of the constructions were knocked off their solutions (Sec. 4). The second
rate 1e-3 (blocks A2 at m = 32 and H2) was added for that reason -- a design change made after reading pilot outcomes of
non-signed arms, on seeds outside the batch, disclosed here. Claims are scoped **"at lr 3e-3"** and **"at lr 1e-3"**,
never "at best rates" (no SSM rate sweep).

| block | arms | m | steps | lr | runs | epochs |
|---|---|---|---|---|---|---|
| A (primary) | signed, abs, softplus, exp, index | 32, 8 | moves | 3e-3 | 80 | 300 |
| A2 (secondary; droppable with H2) | signed, abs, softplus, exp, index | 32 | moves | 1e-3 | 40 | 300 |
| B (secondary; droppable) | signed, abs, softplus, exp | 32 | all | 3e-3 | 32 | 300 |
| H (rule 14) | abs, softplus, exp constructions + signed control, from A's signed m = 32 | 32 | moves | 3e-3 | 32 | 100 |
| H2 (rule 14) | the same constructions | 32 | moves | 1e-3 | 32 | 100 |

Seed-major order. Driver `run_ssm_sign.sh`: refuses a modified tracked file or any uncommitted G-3 file (at start, after
stage 1 is launched and before the final wait); md5-guards every imported mapformer module plus driver, lib and this file
(before training, before the constructions, before the analysis); `flock` single instance; `DRV_GPUS` (default 1),
`DRV_MAXPG` 4 (`G3_MAXPG`), `G3_BLOCK_A2` and `G3_BLOCK_B` (default 1); artifacts counted before the done marker
(`.ssm_sign_done`). Dry-run end to end in a scratch clone (DRV_DRYRUN=1 with synthetic eval.json files in the batch layout):
216 launches listed (80 + 40 + 32 + 32 + 32), fails closed at stage 2 without sources, with them counts every artifact,
re-checks the md5 guard, runs the analysis and sets the marker; the same with `G3_BLOCK_B=0`.
Host kalman, 2x RTX 4090 shared with another user's jobs, torch 2.10.0+cu128.

## 6. Registered analysis (`analyze_ssm_sign.py`)

SOLVED = `stats_core.classify_run` (final-5% training loss < 0.05); a run with any non-finite loss is DIVERGED, never SOLVED,
accuracy counted 0. Every run's recipe (incl. lr, epochs, init_built) and len(losses) are asserted. `better(a, b)` = (Fisher <
.05 and k_a > k_b) or (permutation p < .05 and acc_a - acc_b >= +0.05). Contrast labels are `analyze_robust.cell_label`'s
thresholds (SIGNED LEARNS BETTER / NO SIGN COST / NON-NEGATIVE LEARNS BETTER / UNMEASURED).

Block A, per m (n = 8):
- **G0** k_signed >= 6, else UNREADABLE at m.
- **GI** k_index <= 1 and mean acc_signed - mean acc_index >= 0.05, else TASK DOES NOT NEED INTEGRATION at m.
- **P (primary: m = 32; secondary: m = 8)** abs vs signed, raw p's.
- **BR** softplus and exp vs signed at each m, Holm per test over the 4 cells, a label kept only if the test that fired
  survives Holm.
- **READING** from P at m = 32: UNREADABLE / TASK DOES NOT NEED INTEGRATION / SIGN COST IN THE SSM (= SIGNED LEARNS BETTER) /
  NO SIGN COST IN THE SSM / NON-NEGATIVE LEARNS BETTER / UNMEASURED; with "; m = 8: <label>" and
  "; lr 1e-3 (m = 32): <label>" (block A2, same gates) appended.

Block A2 (lr 1e-3, m = 32): G0, GI as above; abs vs signed raw p's; softplus, exp vs signed Holm over the 3 cells per test.

Blocks H (3e-3) and H2 (1e-3), each on the n_s sources SOLVED in block A at m = 32 (UNREADABLE if n_s < 6):
- **E1** frozen |acc - source acc| <= 0.005 on every source and move-angle error <= 1e-4 rad; else NO EXACT CONSTRUCTION.
- **E2** LOST (MAIN_APPENDIX2 D6): max per-epoch training loss in the continuation > 0.3, or > 10x the control's at that
  seed and > 0.05. HELD = SOLVED on >= n_s - 1, mean acc >= control - 0.01, LOST on <= 1; PERTURBED AND RE-FOUND = the same
  but LOST on >= 2; NOT HELD otherwise.
- **Joint** per class and rate: SIGNED LEARNS BETTER from scratch at m = 32 (A for H, A2 for H2) + HELD at the same rate ->
  "SEARCH COST (exists, kept, not found)". A from-scratch cost with PERTURBED AND RE-FOUND or NOT HELD is reported as a cost
  whose solution exists but is not stable under training at that rate -- not as search alone.

Block B: G0 on B's signed; abs / softplus / exp vs signed, Holm over the 3 cells per test.

Descriptive: r(log10 final loss, acc) at T and 4T (rule 2); per-seed accuracy and run class; medians of cos(z_+, z_-), the
wrap-invariant cancellation share of non-blind channels, the share cancelling by a wrap (|sum| > 1 rad); observation-angle
residual in block B; the attention anchors on record beside the table (never pooled). Selftest
(`python3 -m mapformer.analyze_ssm_sign --selftest`): every READING branch, all four contrast labels, every existence label
incl. the n_s < 6 gate and the joint SEARCH COST, divergence handling, block B labels, the recipe guard, and breadth_label ==
cell_label on raw p.

## 7. Power (rule 5; `stats_core.fisher_solved`)

With signed 8/8: SOLVED fires for abs <= 3/8 (p 0.026; 4/8 p 0.077); with signed 7/8: abs <= 2/8 (p 0.041); with signed 6/8:
abs <= 1/8 (p 0.041). Otherwise only the accuracy test (permutation p < .05 and |d| >= 0.05) can fire. NO SIGN COST needs
k_abs >= k_signed - 1 and acc within 0.01. Anything between reads UNMEASURED. Seeds are paired by design (same init and data
per seed across arms); the tests are unpaired (conservative).

## 8. Predictions

- From attention (LRSWEEP, SIGNROUTE at 3e-3: |.| 2/8, exp 1/8, softplus 3/8 vs signed 8/8): **SIGN COST IN THE SSM** at
  m = 32 for abs at both rates; BR SIGNED LEARNS BETTER for exp and softplus; E1 PASS for every class. HELD for abs at 1e-3
  (as WRAP_EXISTENCE on the ring); the pilot suggests PERTURBED AND RE-FOUND at 3e-3.
- Rival (PAPER_LEVEL #1): the recurrence gives gradient through time and the linear kernel has no softmax competition ->
  NO SIGN COST.
- **No registered prior for exp**: its init (exp(0) = 1 on every channel, omega 2 pi .. 2 pi / 32) puts 13 of 32 channels per
  head (omega > pi / 2) at move-angle sums > pi, the "past-pi start" that rescued |.| on the ring in attention (`MECHANISM_RESULTS.md` M2);
  in attention exp still solved 1/8. At 1e-3 one pilot seed solved, at 3e-3 one was unstable (Sec. 9). An exp success would
  be read as that start, not as "non-negativity is free"; an exp failure at 3e-3 as instability unless A2 agrees.
- m = 8: no registered prior (small ring, retrace floor 0.792; with grid_size 8, 21 of 32 channels per head start past pi for exp).

## 9. Pilot (outside seeds, never pooled)

One seed per condition: anecdote, not evidence. Pilots s2180-2182 used the model before the init-order fix (the step map was
drawn second instead of last, Sec. 2); the function is unchanged.

| run | arm | steps | m | lr | epochs | acc T=128 | acc T=512 | final loss | class | cos(z+, z-) | cancel share |
|---|---|---|---|---|---|---|---|---|---|---|---|
| s2180 | signed | moves | 32 | 1e-3 | 300 | 0.9792 | 0.5710 | 0.0644 | STALLED | +0.094 | 0.35 |
| s2183 | signed | moves | 32 | 3e-3 | 300 | **1.0000** | 0.9994 | 0.0001 | **SOLVED** | -1.000 | 1.00 |
| s2180 | abs | moves | 32 | 1e-3 | 300 | 0.7363 | 0.4093 | 0.8164 | STALLED | -0.008 | 0.06 |
| s2183 | abs | moves | 32 | 3e-3 | 300 | **0.8098** | 0.3801 | 0.6507 | **STALLED** | +0.009 | 0.06 |
| s2180 | softplus | moves | 32 | 1e-3 | 300 | 0.9050 | 0.4121 | 0.3563 | STALLED | +0.365 | 0.07 |
| s2183 | softplus | moves | 32 | 3e-3 | 300 | 0.9640 | 0.5460 | 0.1616 | DESCENDING | +0.885 | 0.16 |
| s2180 | exp | moves | 32 | 1e-3 | 300 | 0.9986 | 0.8324 | 0.0045 | SOLVED | +0.508 | 0.24 |
| s2183 | exp | moves | 32 | 3e-3 | 300 | 0.5040 | 0.4815 | 1.7716 | STALLED (loss rose to 2.84) | -1.000 | 0.00 |
| s2181 | index | moves | 32 | 1e-3 | 300 | 0.7554 | 0.4057 | 0.9603 | STALLED | -- | -- |
| s2181 | none (memoryless) | moves | 32 | 1e-3 | 100 | 0.4959 | 0.4981 | 2.0816 | STALLED | -- | -- |
| s2180 | signed | moves | 8 | 1e-3 | 300 | 1.0000 | 0.8246 | 0.0000 | SOLVED | +0.067 | 0.17 |
| s2183 | signed | moves | 8 | 3e-3 | 300 | 1.0000 | 0.9979 | 0.0000 | SOLVED | -1.000 | 1.00 |
| s2180 | abs | moves | 8 | 1e-3 | 300 | 0.8992 | 0.4725 | 0.2851 | STALLED | +0.151 | 0.04 |
| s2182 | signed | all | 32 | 1e-3 | 300 | 0.9992 | 0.8996 | 0.0031 | SOLVED | -0.504 | 0.19 |

Reading (anecdote): the signed SSM learns the ring map in one layer (at 3e-3 it ends with antipodal move latents and every
non-blind channel cancelling, as trained signed attention models do); |.| stalls at both rates at m = 32 (0.736, 0.810:
BELOW the retrace floor 0.864 and near the index arm 0.755) and at m = 8 (0.899); softplus stalls or descends slowly; exp
solves at 1e-3 and is unstable at 3e-3. The task needs integration: index 0.755 and memoryless 0.496 against the 0.864
retrace floor. "cancel share" = share of non-blind channels whose move-angle sum is within 0.1 rad of 0 mod 2 pi.

## 10. Cost and decisions for the user

Measured on kalman beside the other user's jobs (both GPUs at 100% utilisation from them): one of our runs alone 2.1
s/epoch; 4 of ours per GPU 5.0-6.1 s/epoch each (`s_per_epoch` in every pilot eval.json), i.e. 1.25-1.52 s per run-epoch.
Packing 4 per GPU is 1.4-1.7x the throughput of one at a time. Constructions: < 1 min CPU per seed. Disk: ~0.85 MB per
run, ~0.2 GB for the batch.

| configuration | run-epochs | one shared 4090 (4 per GPU) | both GPUs |
|---|---|---|---|
| registered default: A + A2 + B + H + H2 (216 runs) | 52,000 | 18-22 h | 9-11 h |
| without B (184 runs) | 42,400 | 15-18 h | 7.5-9 h |
| A + H only (112 runs; no second rate, no full-step block) | 27,200 | 9.4-11.5 h | 4.7-5.7 h |
| minimal: A at m = 32 only + H (72 runs) | 15,200 | 5.3-6.4 h | 2.6-3.2 h |

(PAPER_LEVEL's 5-8 GPU-h assumed an unshared GPU and 64-96 runs; the minimal row is that design.)

Decisions before launch: (1) which configuration (blocks B and A2 + H2 are droppable by `G3_BLOCK_B=0`, `G3_BLOCK_A2=0`;
the m = 8 cell and the minimal row would need a one-line edit of SIZES in the driver and the analysis, i.e. a new
registration); (2) one GPU (default GPU 1) or both (`DRV_GPUS="0 1"`); (3) the rule-29 verification agent and Amendment 1;
(4) commit, then `cd /home/prashr/mapformer && setsid nohup bash run_ssm_sign.sh > /dev/null 2>&1 &`.
