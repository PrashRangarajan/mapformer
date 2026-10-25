# G3_SSM (G-3): the sign test beyond attention -- pre-registration

**Status: PREPARED AND AMENDED, NOT LAUNCHED (2026-10-23).** The independent results-blind code check (rule 29,
`docs/audits/2026-10-23/g3_ssm_audit.md`: no BLOCKER, 6 MAJOR, 11 MINOR) is folded in as Amendment 1 (end of file; the
sections below are updated where marked [A1]). Before launch: the user's choice of configuration (Sec. 10) and a commit
of every file in the table below (the driver refuses an uncommitted or modified tree). The driver md5-guards this file,
so it is final at launch.

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
| `test_ssm_sign.py` | unit checks, 40/40 PASS (CPU) |

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
- cost present (|.| fails where signed solves, an exact |.| solution exists and is held): the cost is **not specific to
  softmax attention** -- it appears in a one-layer diagonal complex SSM with an input-dependent rotation, equivalently
  unnormalised rotary linear attention [A1.5];
- no cost: the cost is **not shown without softmax normalisation** -- a scope boundary at softmax, NOT "attention-specific"
  [A1.5]. The model is computed as unnormalised rotary linear attention (Mamba-3 Prop. 3), the same function and gradients
  as the recurrence, so "the recurrence gives gradient through time" is not a mechanism that differs between the two forms
  and is dropped as the rival's account; the rival is "softmax competition is the source of the cost".

It cannot decide WHY (the route/wrap accounts of `SEARCH_THEORY.md` are not separated here, as SIGNROUTE Amendment 1 found
for attention), nor anything beyond one layer, one task, the fixed rate(s) run and one budget.

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
checkpoints, at lr 1e-3 (H2) (E2). 100 epochs, not WRAP_EXISTENCE's 300, to save cost (warm-up 5 epochs instead of 15 to the
same peak). [A1.7] This is a third of the recovery budget: the pilot's softplus at 3e-3 peaked at epoch 28 and softplus /
exp at 3e-3 were still DESCENDING at 100, so PERTURBED AND RE-FOUND vs NOT HELD is set partly by the budget; the joint
distinguishes them (Sec. 6) and no registered SEARCH COST depends on the line between them.

## 5. Batch

Seeds **2100-2107** (fresh; no run dir in the repo carries them); pilots **2180-2189**, never pooled. Recipe = the attention
ring anchors' (`train_signroute.py` / `train_gauge.py`): T = 128, batch 128, 98 batches, 300 epochs, cosine with 5% warmup
to 10% of peak, AdamW wd 0.05, 1 layer, 2 heads, d 128, rank 4, 3 data workers, dropout 0.1.

**Rates.** The primary rate **3e-3** was fixed from the SIGNED pilots before any non-signed 3e-3 pilot was read: signed at
1e-3 (s2180) was still at loss 0.064 after 300 epochs (STALLED, not SOLVED: G0 at risk), at 3e-3 (s2183) at 0.003 by epoch
70. 3e-3 is also |.|'s best rate in attention (LRSWEEP). **Disclosure [A1.7]:** the 1e-3 non-signed pilots (abs 0.736,
softplus 0.905, exp SOLVED 0.9986) had finished and WERE read (printed in the same table as the signed 1e-3 result) before
3e-3 was fixed. The selection was made on the reference arm (G0); abs's pilot later scored higher at 3e-3 (0.810) than at
1e-3, so the choice did not favour the primary contrast. **Then the non-signed 3e-3 pilots showed exp unstable at 3e-3**
(s2183: training loss rising from 1.3 to 2.8, above the memoryless 2.08, ending at 0.504 accuracy) **while exp at 1e-3
solved** (s2180), and the 3e-3 continuations of the constructions were knocked off their solutions (Sec. 4). The second
rate 1e-3 (blocks A2 at m = 32 and H2) was added for that reason -- a design change made after reading pilot outcomes of
non-signed arms, on seeds outside the batch, disclosed here. Claims are scoped **"at lr 3e-3"** and **"at lr 1e-3"**,
never "at best rates" (no SSM rate sweep).

| block | arms | m | steps | lr | runs | epochs |
|---|---|---|---|---|---|---|
| A (primary) | signed, abs, softplus, exp, index | 32 (all configurations), 8 (core, full) | moves | 3e-3 | 40 / 80 | 300 |
| A2 (full only; repeats the index gate at the second rate) | signed, abs, softplus, exp, index | 32 | moves | 1e-3 | 40 | 300 |
| B (full only) | signed, abs, softplus, exp | 32 | all | 3e-3 | 32 | 300 |
| H (rule 14) | abs, softplus, exp constructions + signed control, from A's signed m = 32 | 32 | moves | 3e-3 | 32 | 100 |
| H2 (rule 14; minimal_h2, full) | the same constructions | 32 | moves | 1e-3 | 32 | 100 |

Seed-major order. Driver `run_ssm_sign.sh`: at start refuses any modified tracked file or any uncommitted G-3 file;
mid-batch (after stage 1 is launched, before the final wait) checks only the batch's files and the guarded modules
against HEAD [A1.7]; md5-guards every imported mapformer module plus driver, lib and this file (before training, before
the constructions, before the analysis); `flock` single instance; `G3_CONFIG` (Sec. 10), `DRV_GPUS` (default 1),
`DRV_MAXPG` 4 (`G3_MAXPG`); passes `--config`, `--root` and `--out` to the analysis; artifacts counted before the done
marker (`.ssm_sign_done`). Dry runs: see Amendment 1.
Host kalman, 2x RTX 4090 shared with another user's jobs, torch 2.10.0+cu128.

## 6. Registered analysis (`analyze_ssm_sign.py`)

SOLVED = `stats_core.classify_run` (final-5% training loss < 0.05); a run with any non-finite loss is DIVERGED, never SOLVED,
accuracy counted 0. Every run's recipe (incl. lr, epochs, init_built) and len(losses) are asserted. `better(a, b)` = (Fisher <
.05 and k_a > k_b) or (permutation p < .05 and acc_a - acc_b >= +0.05). Contrast labels are `analyze_robust.cell_label`'s
thresholds (SIGNED LEARNS BETTER / NO SIGN COST / NON-NEGATIVE LEARNS BETTER / UNMEASURED).

Block A, per m (n = 8):
- **G0** k_signed >= 6, else UNREADABLE at m.
- **GI** k_index <= 1 and mean acc_signed - mean acc_index >= 0.05, else TASK DOES NOT NEED INTEGRATION at m.
- **P (primary: m = 32; secondary: m = 8 when run)** abs vs signed, raw p's; the stored label is the GATED one (G0 / GI)
  [A1.1]. NON-NEGATIVE LEARNS BETTER is practically unreachable under G0 (Fisher cannot fire for abs above a signed arm at
  >= 6/8; only the accuracy branch can) [A1.7].
- **RUNAWAY** [A1.4] a run whose final move_step_absmax > 1e3 or with any post-warm-up epoch loss > 2.08 (memoryless);
  counted per cell; a non-negative cell with k_RUNAWAY >= 2 keeps its label with "(k RUNAWAY)", licenses no learnability
  sentence and forms no joint (blocks A, A2, B).
- **BR** softplus and exp vs signed at each m, Holm per test over the 2 x (number of sizes) cells, a label kept only if
  the test that fired survives Holm.
- **READING** from P at m = 32: UNREADABLE / TASK DOES NOT NEED INTEGRATION / SIGN COST IN THE SSM (= SIGNED LEARNS BETTER) /
  NO SIGN COST IN THE SSM / NON-NEGATIVE LEARNS BETTER / UNMEASURED; with "; m = 8: <label>" (core, full) and
  "; lr 1e-3 (m = 32): <label>" (block A2, full) appended. Wording per Sec. 1 [A1.5].

Block A2 (lr 1e-3, m = 32): G0, GI as above; abs vs signed raw p's; softplus, exp vs signed Holm over the 3 cells per test.

Blocks H (3e-3) and H2 (1e-3), each on the n_s sources SOLVED in block A at m = 32 (UNREADABLE if n_s < 6):
- **E1** frozen |acc - source acc| <= 0.005 on every source, for the class and for the signed control [A1.7], and
  move-angle error <= 1e-4 rad; else NO EXACT CONSTRUCTION.
- **E2** LOST (MAIN_APPENDIX2 D6): max per-epoch training loss in the continuation > 0.3, or > 10x the control's at that
  seed and > 0.05. HELD = SOLVED on >= n_s - 1, mean acc >= control - 0.01, LOST on <= 1; PERTURBED AND RE-FOUND = the same
  but LOST on >= 2; NOT HELD = neither (declared: WRAP_EXISTENCE A4's statistical NOT HELD is not used) [A1.7].
- **E2'** (declared secondary) [A1.3] the same with LOST' = max loss > 0.3 only, printed with each class's max-loss range
  beside the attention ranges (AO_bal 0.024-0.154, SO_bal 0.015-0.112): the SSM signed control peaks at 0.002-0.018, so the
  registered 10x clause binds where it did not in attention.
- **softplus / exp** [A1.2] any E2 other than HELD is read as a property of constructions (f) / (g) at omega' = 2 pi /
  ln 2 and 2 pi, not of the class; the class-level E2 statement is made for abs only. Each construction's move-angle
  sensitivity is recorded in report.json and printed (pilot s2183: control 30, abs 116, softplus 191, exp 238; attention
  AO_bal 87-96, SO_bal 15-18).
- **Joint** per class: gated, licensed SIGNED LEARNS BETTER from scratch at m = 32 plus HELD -> SEARCH COST (exists, kept,
  not found); plus PERTURBED AND RE-FOUND -> COST, SOLUTION PERTURBED AND RE-FOUND; plus NOT HELD -> COST, SOLUTION NOT HELD
  [A1.7: distinguished; with 100-epoch continuations the line between the last two is set by the budget]. H joins A
  (3e-3); H2 joins A2 in `full`, otherwise A's 3e-3 label as a declared CROSS-RATE joint.

Block B: G0 on B's signed; abs / softplus / exp vs signed, Holm over the 3 cells per test.

Descriptive: r(log10 final loss, acc) at T and 4T over blocks A / A2 / B (not the continuations) [A1.7]; per-seed accuracy and run class; medians of cos(z_+, z_-), the
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
- Rival [A1.5]: softmax competition is the source of the cost -> NO SIGN COST here, read "not shown without softmax
  normalisation".
- softplus also starts past pi on 9 of 32 channels per head at m = 32 (15.6 at m = 8) [A1.7].
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

## 10. Cost and configuration [A1.6]

Measured on kalman beside the other user's jobs (both GPUs at 100% utilisation from them): one of our runs alone 2.1
s/epoch; 4 of ours per GPU 5.0-6.1 s/epoch each (`s_per_epoch` in every pilot eval.json), i.e. 1.25-1.52 s per run-epoch.
Packing 4 per GPU is 1.4-1.7x the throughput of one at a time. Constructions: < 1 min CPU per seed. Disk: ~0.85 MB per
run, ~0.2 GB for the batch.

The configuration is a **launch-time choice**, `G3_CONFIG`, whose presets live in `analyze_ssm_sign.CONFIGS` (the driver
reads them from there, so driver and analysis cannot disagree). The first launch writes `runs/ssm_sign/p0/config.txt`; a
relaunch with another configuration is refused. All three are registered; the readings each can produce are those of
Sec. 6 restricted to its blocks.

| G3_CONFIG | blocks | runs | run-epochs | one shared 4090 (4 per GPU) | both GPUs |
|---|---|---|---|---|---|
| **minimal_h2 (recommended default)** | A at m = 32 (40) + H (32) + H2 (32) | 104 | 18,400 | 6.4-7.8 h | 3.2-3.9 h |
| core | A at m = 32 and 8 (80) + H (32) | 112 | 27,200 | 9.4-11.5 h | 4.7-5.7 h |
| full | A at 32, 8 (80) + A2 (40) + B (32) + H (32) + H2 (32) | 216 | 52,000 | 18-22 h | 9-11 h |

Why minimal_h2 (the audit's recommendation): the primary (abs vs signed at m = 32, lr 3e-3, G0, GI, E1, E2) needs A at
m = 32 and H only; H2 (3,200 run-epochs, readable without A2 since its sources are block A's) adds the gentler rate at which
the pilot's constructions were only mildly perturbed. A2 carries a real G0 risk (signed at 1e-3 stalled on the one pilot
seed), m = 8 has no prior, and B is a separate Mamba-faithfulness question carrying the full-turn burden. In minimal_h2 the
H2 joint is CROSS-RATE (from scratch at 3e-3, kept at 1e-3), declared as such. Both-GPU timing (8 runs x 3 data workers
beside the other user's jobs on 32 cores) is unmeasured; slots are shared with this user's other batches (the SIGNSEP pilot
held 4 slots on cuda:1 during the check), so a launch may wait for them.

Launch (after commit):

    cd /home/prashr/mapformer && G3_CONFIG=minimal_h2 setsid nohup bash run_ssm_sign.sh > /dev/null 2>&1 &
    cd /home/prashr/mapformer && G3_CONFIG=core setsid nohup bash run_ssm_sign.sh > /dev/null 2>&1 &

(add `DRV_GPUS="0 1"` for both GPUs).

## Amendment 1 -- independent results-blind code check (rule 29), 2026-10-23, before launch

From `docs/audits/2026-10-23/g3_ssm_audit.md` (check scripts in `docs/audits/2026-10-23/g3_ssm_check/`). No batch result
exists (`runs/ssm_sign/` absent). No BLOCKER; the recurrence, step classes, init identity, task, gate, constructions,
analysis and driver were re-derived and verified. Every MAJOR and MINOR is applied below; the sections above carry the
changes, marked [A1.n].

1. **Joint reading respects the gates (MAJOR 1, code bug).** The joint read the RAW block-A abs label, so a GI failure at
   m = 32 could still read SEARCH COST (audit `c4_joint.py`). `analyze_ssm_sign` now stores and reads the GATED label for
   abs (as it already did for softplus / exp); no joint forms when G0 or GI fails. Selftest case added (index 2/8 at
   m = 32: READING = TASK DOES NOT NEED INTEGRATION, every joint None, at both H and H2).
2. **Construction conditioning (MAJOR 2).** A non-HELD E2 for softplus / exp is read as a property of constructions (f) /
   (g) at omega' = 2 pi / ln 2 and 2 pi, not of the class; the class-level E2 statement is made for abs only, and the
   analysis prints that note beside any non-HELD softplus / exp label and any joint built from one. `construct_ssm_sign`
   now records each construction's move-angle sensitivity (L1 norm of d angle / d parameter over the move embeddings,
   W_in, W_out, omega; mean and max over the move angles; the audit's definition) in report.json and in the built
   checkpoint's construction record, which the trainer copies into eval.json's `frozen`. Reproduced on pilot s2183:
   control 29.9, abs 115.5, softplus 191.2, exp 238.0 (source 20.1), equal to the audit's 30 / 116 / 191 / 238; attention
   AO_bal 87-96, SO_bal 15-18.
3. **Declared secondary E2' (MAJOR 3).** The registered LOST rule is kept. E2' uses LOST' = max per-epoch continuation
   loss > 0.3 only, with HELD' / PERTURBED' / NOT HELD' as in E2; each class's max-loss range is printed beside the
   attention ranges (AO_bal 0.024-0.154, SO_bal 0.015-0.112), because the SSM signed control peaks at 0.002-0.018 and the 10x
   clause then binds. A joint read on E2' is a secondary. Selftest: a 0.1 peak against a 0.001 control reads PERTURBED AND
   RE-FOUND under E2 and HELD under E2'.
4. **RUNAWAY (MAJOR 4).** A run is RUNAWAY if its final move_step_absmax > 1e3 or any epoch after the 5% warm-up has
   training loss > 2.08 (the memoryless arm's). Pilot s2183 exp at 3e-3 had move_step_absmax 79,713 and loss 2.84 and would
   otherwise read as an ordinary STALLED run. Every cell prints k_RUNAWAY and each seed carries a "!" mark; a contrast whose
   non-negative cell has k_RUNAWAY >= 2 keeps its label with "(k RUNAWAY)", licenses no learnability sentence and forms no
   joint. Applies to blocks A, A2 and B (exp in B included). Selftest: both criteria detected; a 2-runaway abs cell reads
   "SIGN COST IN THE SSM (2 RUNAWAY)" with no joint; block B exp "(3 RUNAWAY)".
5. **Framing (MAJOR 5).** The model is computed as unnormalised rotary linear attention, the same function and gradients
   as the recurrence. SIGN COST IN THE SSM reads "not specific to softmax attention"; NO SIGN COST IN THE SSM reads "not
   shown without softmax normalisation (a scope boundary at softmax)", never "attention-specific". "The recurrence gives
   gradient through time" is dropped as the rival's mechanism; the rival is "softmax competition is the source of the
   cost". The analysis prints the wording beside the reading.
6. **Configuration as a launch-time choice (MAJOR 6).** `G3_CONFIG` in {minimal_h2 (default, recommended), core, full},
   presets in `analyze_ssm_sign.CONFIGS` and read from there by the driver; H2 is decoupled from A2; sizes may be m = 32
   only, and the reading then has no m = 8 part; BR's Holm family is 2 x (number of sizes) cells. Without A2 the H2 joint
   is a declared CROSS-RATE joint (from scratch at 3e-3, kept at 1e-3). The first launch writes `runs/ssm_sign/p0/
   config.txt` and a relaunch with another configuration is refused. The analysis requires `--config` and takes `--root`
   and `--out` (MINOR 10); the driver passes all three.
7. **Smaller items (MINORs).**
   - Rule-2 r(loss, acc) runs over blocks A / A2 / B only; it previously included H2 (MINOR 1).
   - E1 also requires the signed control's frozen accuracy within 0.005 of the source; otherwise every class reads NO
     EXACT CONSTRUCTION (MINOR 2; selftest case).
   - NOT HELD is declared as "neither HELD nor PERTURBED"; WRAP_EXISTENCE A4's statistical NOT HELD is not used (MINOR 3).
   - The joint distinguishes SEARCH COST, COST WITH SOLUTION PERTURBED AND RE-FOUND, and COST WITH SOLUTION NOT HELD. The
     line between the last two depends on the 100-epoch budget, which is stated in Sec. 4 (MINOR 4).
   - Mid-batch tree checks cover only the batch's files and the guarded modules (`git diff --quiet HEAD -- ...`), so
     another session's edits elsewhere in the repo cannot fail a running batch. The start check is unchanged (MINOR 5).
   - Slots are shared with this user's other batches, and the both-GPU timing is unmeasured; both are stated in Sec. 10
     (MINOR 6).
   - Disclosure: the 1e-3 non-signed pilots had finished and WERE read before 3e-3 was fixed (Sec. 5) (MINOR 7).
   - NON-NEGATIVE LEARNS BETTER at P is reachable only through the accuracy branch (MINOR 8).
   - softplus also starts past pi on 9 of 32 channels per head at m = 32 (15.6 at m = 8) (MINOR 9).
   - A2's index arm repeats A's gate at the second rate, costing 2,400 run-epochs; kept in `full` only (MINOR 11).

**Checks after the amendment (CPU):** `test_ssm_sign` 40/40 PASS, including a new sensitivity check;
`analyze_ssm_sign --selftest` PASS, covering:
- every READING branch in all three configurations;
- all four contrast labels and the GI-joint case;
- every existence label, including the control-E1 failure, E2 vs E2', the joint variants and the cross-rate H2 joint;
- RUNAWAY by both criteria;
- divergence, block B with a RUNAWAY suffix, and the recipe guard.

**Driver dry runs** were made in a scratch clone with the batch files committed there (`DRV_DRYRUN=1`):
- (i) minimal_h2 without sources: 40 stage-1 launches, all lr 3e-3, none at m = 8; fails closed at stage 2.
- A relaunch as core into the same run dir is REFUSED (config.txt).
- (i) with synthetic sources and built checkpoints: 64 stage-2 launches (32 H at 3e-3, 32 H2 at 1e-3, all
  `--init-built`); fails closed without results.
- (i) with synthetic results: 104 runs found; artifacts counted; the md5 guard re-checked; the analysis run with
  `--config minimal_h2`; the marker is set.
- (ii) core: 80 stage-1 launches, 40 of them at m = 8.
- full: 152 stage-1 launches, 40 of them at 1e-3.
- An unknown `G3_CONFIG` aborts.

## Amendment 2 -- block A2 launch (2026-10-25)

**Not results-blind.** Blocks A, H and H2 (configuration `minimal_h2`, koopman, 2026-10-23, `G3_SSM_RESULTS.md`) were
read before this amendment was written: signed 8/8, |.| 0/8, softplus 1/8 (5 of 7 unsolved DESCENDING), exp 0/8 with all
8 RUNAWAY, index 0/8; READING SIGN COST IN THE SSM. Block A2 is launched BECAUSE of that reading: Sec. 8 registers "an exp
failure at 3e-3 as instability unless A2 agrees", and with exp 8/8 RUNAWAY at 3e-3 the exp cell licenses no learnability
sentence without A2. The decision to run A2 therefore depends on block A's outcome; A2's design and analysis do not.

**Design: unchanged from Sec. 5** (table row A2). Arms signed, abs, softplus, exp, index; m = 32; steps = moves; lr 1e-3;
300 epochs; the block-A recipe (asserted per run by `analyze_ssm_sign.RECIPE`); seeds **2100-2107**, the registered batch
seeds, so each A2 run has the same initial weights and walk stream as its block-A twin and differs only in the rate
(paired by design, tested unpaired, Sec. 7). Seed labels 2100-2107 were also used by RANKWD (torus attention); no run,
checkpoint or contrast is shared, the overlap is in labels only. Layout `runs/ssm_sign/p0/A2/<arm>_moves_m32_lr1e-3_s<seed>`,
inside the minimal_h2 root (rsynced from koopman to kalman; 104 eval.json, 136 checkpoints).

**Analysis: unchanged from Sec. 6** ("Block A2" line), as already implemented and audited for `full`: G0 on A2's own
signed arm (k >= 6), GI on A2's own index arm (k <= 1 and signed - index >= 0.05); abs vs signed at raw p; softplus, exp vs
signed; RUNAWAY as Amendment 1.4 (a cell with k_RUNAWAY >= 2 keeps its label with the suffix, licenses no learnability
sentence, forms no joint); READING suffix "; lr 1e-3 (m = 32): <gated abs label>"; rule-2 r(loss, acc) over A and A2.
- **Holm family -- a text / code discrepancy found while preparing this amendment.** Sec. 6 says "softplus, exp vs signed
  Holm over the 3 cells per test"; the audited code (`analyse`, block A2) adjusts over the **2** cells softplus and exp, with
  abs at raw p (as block A's P / BR split, where BR's family is softplus and exp x sizes). Declared before A2 is run: the
  code's family governs the registered label (it is the one the Amendment 1 check verified, and the text's "3" matches no
  split that keeps abs at raw p); the results file will also state whether any softplus / exp label would differ under
  Holm over the 3 cells (abs, softplus, exp). Flagged for the independent check.

**Readings that depend on block A, and how.**
- The READING string is "<block A label>; lr 1e-3 (m = 32): <A2 label>"; its first part is already known (SIGN COST IN THE
  SSM). No A2 label is conditioned on block A: G0 and GI at 1e-3 are read on A2's own signed and index arms.
- **The H2 joint changes kind.** In this configuration H2 joins A2 at the same rate (1e-3), as `full` registers; the
  CROSS-RATE H2 joint reported in `G3_SSM_RESULTS.md` (minimal_h2) is superseded by it in the A2 output and kept in the
  minimal_h2 output, both reported. H2's E2 labels are fixed by data already read (|.|, softplus, exp: PERTURBED AND
  RE-FOUND; E2' HELD), so the same-rate H2 joint can only be "COST, SOLUTION PERTURBED AND RE-FOUND" (A2 label SIGNED
  LEARNS BETTER, gated, licensed) or none; the registered SEARCH COST joint cannot form at H2 whatever A2 shows.
- **"Unless A2 agrees" (Sec. 8), made operational here, before A2 is run** (interpretation only; no code): A2 agrees iff
  A2 passes G0 and GI and A2's exp vs signed label is SIGNED LEARNS BETTER without a RUNAWAY suffix. Then: "exp learns
  worse than signed at lr 1e-3" (a learnability statement scoped at 1e-3), while the 3e-3 cell stays instability. If A2's
  exp cell has k_RUNAWAY >= 2: instability at both rates, no exp learnability sentence. If A2's exp label is NO SIGN COST
  or UNMEASURED: no exp cost shown at 1e-3, and exp's 3e-3 failure stays instability. If A2 fails G0 or GI: A2 is
  UNREADABLE / TASK DOES NOT NEED INTEGRATION and exp stays unread. Softplus is read the same way at 1e-3, with its own
  budget scope (DESCENDING runs counted as at 3e-3). Any exp result is read beside Sec. 8's caveat: exp starts past pi on
  13 of 32 channels per head, so an exp success is that start, not "non-negativity is free".
- Block A ran on koopman and A2 runs on kalman (both torch 2.10.0+cu128, RTX 4090). Every registered A2 contrast is
  within A2, so the host does not enter any contrast; the descriptive rule-2 r pools A and A2 across hosts.

**Code diff (additive; no change to the model, trainer, constructions or registered analysis logic).**
- `analyze_ssm_sign.py`: a fourth preset `CONFIGS["a2_after_minimal_h2"] = {sizes (32,), a2 True, b False, h2 True,
  base "minimal_h2"}` (the analysis reads it exactly as any configuration with blocks A at 32, A2, H, H2; the `base` key is
  read only by the driver); two docstring lines; one selftest case (READING with the A2 suffix, A2 exp "(3 RUNAWAY)" with
  no joint, H2 joint kind "same rate (1e-3)").
- `run_ssm_sign.sh`: an A2-only mode, entered when the preset has a `base`. It (i) requires `p0/config.txt` = the base
  (`minimal_h2`) and keeps this launch's `config.txt` and `code_md5.txt` in `p0/A2/` (the base root's own provenance files
  are never written); (ii) requires every guarded file to be the set recorded in `p0/code_md5.txt` with the recorded md5,
  except `analyze_ssm_sign.py`, `run_ssm_sign.sh` and this file -- so A2 trains with byte-identical model, trainer, task and
  library code to block A (checked: 104 guarded files, 0 mismatches); (iii) requires all block-A, H and H2 artifacts and
  never re-runs them (no block-A launches, no constructions, no continuations); (iv) launches the 40 A2 runs, then the
  unchanged artifact count (320), md5 re-check and analysis with `--config a2_after_minimal_h2 --root runs/ssm_sign/p0`;
  (v) writes `G3_SSM_A2_ANALYSIS.txt`, `G3_SSM_A2_VERDICTS.json`, marker `.ssm_sign_a2_done`, log `ssm_sign_a2.log`, so the
  minimal_h2 outputs are not overwritten. For the three original presets the driver's behaviour is unchanged (same run
  dir, provenance files, outputs and marker).

**Host and settings.** kalman, **GPU 0 only** (`DRV_GPUS=0`; GPU 1 is reserved for another batch), `DRV_MAXPG` 4
(`G3_MAXPG` default), `DRV_MINFREE` 2500 MiB, `DRV_SPACING` 5 s, seed-major order, 3 data workers and `OMP_NUM_THREADS=2`
per run.

**Cost.** 40 runs x 300 epochs = **12,000 run-epochs**. Measured in the pilot below: 3 concurrent runs on an otherwise
idle GPU 0, 2.85-2.89 s/epoch each = 0.96 s per run-epoch, so about **3.2 h** at 4 per GPU if GPU 0 stays free of other
jobs; at the Sec. 10 shared-GPU rate (1.25-1.52 s per run-epoch) 4.2-5.1 h. Analysis < 1 min CPU. Disk ~35 MB.

**Pilot (seed 2184, in the registered pilot range 2180-2189 and used by no run in the repo; never pooled; one seed per
arm: anecdote).** Full registered recipe and length (300 epochs at lr 1e-3; the 3e-3 runaways set in at epochs 21-104, so
a shortened schedule would not have tested stability), run concurrently on GPU 0, 2026-10-24. Signed added to the
requested exp / softplus pair because of A2's G0 risk (Sec. 10, audit MAJOR 6).

| run | arm | acc T=128 | acc T=512 | final loss | class (final-5% < 0.05) | max loss after warm-up | move_step_absmax | RUNAWAY | cos(z+, z-) | cancel share | s/epoch |
|---|---|---|---|---|---|---|---|---|---|---|---|
| s2184 | exp | 0.9984 | 0.8278 | 0.0055 | SOLVED | 1.233 (epoch 19) | 4.93 | no | +0.551 | 0.21 | 2.85 |
| s2184 | softplus | 0.8669 | 0.4407 | 0.4386 | not solved (drifting down, 0.46 -> 0.44 over the last 60 epochs) | 1.04 | 2.42 | no | +0.633 | 0.07 | 2.89 |
| s2184 | signed | 0.9961 | 0.4574 | 0.0131 | SOLVED | 0.94 | 1.74 | no | -0.482 | 0.35 | 2.88 |

Reading (anecdote): at 1e-3 exp is stable on this seed (no epoch above the memoryless 2.08, steps bounded by 5) and
solves, as pilot s2180 did; softplus ends at 0.867, at the retrace floor (0.864); signed solves on loss but with a partial
code (T=512 0.457; s2180 signed at 1e-3 was STALLED at 0.064), so G0 at 1e-3 remains a real risk. Pilot run dirs:
`runs/ssm_sign_pilot/{exp,softplus,signed}_moves_m32_lr1e-3_s2184`.

**Checks after the amendment (CPU unless stated).** `test_ssm_sign` 40/40 PASS; `analyze_ssm_sign --selftest` PASS
(incl. the new case); `analyze_ssm_sign --config minimal_h2` on the real root reproduces `G3_SSM_VERDICTS.json` exactly.
Driver dry runs (`DRV_DRYRUN=1`) in a scratch clone with these files committed there and a copy of the real p0 root
(eval.json copied with paths rewritten, checkpoints linked): (i) `a2_after_minimal_h2` without A2: 40 launches, all
`--lr 1e-3 --device cuda:0` into `p0/A2/`, seed-major, no block-A / H / H2 launch, provenance 0 mismatches, fails closed
on the 80 missing A2 artifacts (none outside A2); (ii) with one H2 eval.json removed: REFUSED (base incomplete); (iii) with
the trainer changed by one comment line and committed: REFUSED (provenance mismatch `train_ssm_sign.py`); (iv)
`minimal_h2` into the same root: ABORT (md5 guard: driver / analysis changed); `full`: REFUSED (config.txt); (v) with
synthetic A2 results (block-A eval.json relabelled lr 1e-3, plumbing only): 40 skips, 320 artifacts, md5 re-check,
analysis with the A2 suffix and the same-rate H2 joint, `.ssm_sign_a2_done` set. The base root's `config.txt` and
`code_md5.txt` were unchanged by every dry run.

**Launch (after a commit of every file, by the main session; the driver refuses any modified tracked file anywhere in
the repo at start, and md5-guards this file in `p0/A2/code_md5.txt`, so this file must not be edited again until the
batch has finished):**

    cd /home/prashr/mapformer && G3_CONFIG=a2_after_minimal_h2 DRV_GPUS=0 setsid nohup bash run_ssm_sign.sh > /dev/null 2>&1 &

## Amendment 3 -- independent check of the A2 launch (rule 29), 2026-10-25, before A2 is run

From `docs/audits/2026-10-25/g3_a2_audit.md`. The check is results-blind for A2: no A2 run exists (`runs/ssm_sign/p0/A2`
absent). Verdict: launchable after an amendment. No BLOCKER; 1 MAJOR, 3 MINOR, 6 NOTE, all applied below. Where this
amendment and Amendment 2 differ, this one governs.

1. **Holm family for block A2 (MAJOR 1; supersedes Amendment 2's Holm paragraph).**
   - The registered text governs (Sec. 6, "Block A2"). The softplus and exp labels use Holm per test over the 3 cells
     {abs, softplus, exp}. abs's own label uses raw p.
   - Amendment 2 claimed the text "matches no split that keeps abs at raw p". That was wrong: abs is the unadjusted
     primary, and the two secondaries are adjusted within a family that also counts abs, which is the literal reading.
   - Amendment 1 did not change the A2 family; A1.6 concerns block A's BR line.
   - The audit found the family matters in the G0-risk regime (signed 6-7/8): 80 of 4,374 softplus / exp cells change
     verdict between the two families (Fisher-only scan).
   - `analyze_ssm_sign.analyse` now matches the text. The A2 block computes the 3-cell family and then replaces abs's
     entry with its raw-p label. This changes `full`'s A2 in the same way; no `full` run exists.
   - The 2-cell labels (Holm over {softplus, exp}) are printed as a declared sensitivity line, marked "[DIFFERS from the
     registered label]" where they differ. They are stored as `A2_sensitivity_2cell` in the verdicts JSON and are never a
     registered label.
   - A new selftest case separates the families: signed 6/8, abs 1/8, softplus 0/8, exp 1/8, with accuracies close enough
     that the permutation branch cannot fire. exp's Fisher p is 0.0811 adjusted over 3 cells (UNMEASURED) and 0.0406
     over 2 cells (SIGNED LEARNS BETTER, sensitivity only); abs stays at raw p.
   - The change is in `analyze_ssm_sign.py`, one of the three files exempt from the A2 provenance check, and nothing
     imports it. Model, trainer, task and library code are unchanged.

2. **"Unless A2 agrees" (MINOR 1; supersedes Amendment 2's definition).**
   - **The 3e-3 exp cell stays instability whatever A2 shows.** Amendment 1.4 binds it: 8/8 RUNAWAY, so it licenses no
     learnability sentence.
   - **"A2 agrees"** means: A2 passes G0 and GI, and A2's exp label is SIGNED LEARNS BETTER without a RUNAWAY suffix.
     The exp label is the registered one under item 1's 3-cell family.

   The branches for A2's exp cell:
   - **SIGNED LEARNS BETTER, licensed (agrees):** "exp learns worse than signed at lr 1e-3", a learnability statement
     scoped at 1e-3.
   - **NO SIGN COST:** "no sign cost for exp at lr 1e-3". This is a positive registered statement, read with Sec. 8's
     caveat: exp starts past pi on 13 of 32 channels per head, so the start, not "non-negativity is free".
   - **UNMEASURED:** "no exp cost shown at 1e-3". This is not "no cost".
   - **NON-NEGATIVE LEARNS BETTER:** read as the past-pi start (Sec. 8), never as "non-negativity is free". It is
     reachable only through the accuracy branch, e.g. when signed passes G0 with partial codes (pilot s2184: exp 0.998,
     signed 0.996).
   - **k_RUNAWAY >= 2 (any label):** instability at both rates; no exp learnability sentence.
   - **G0 or GI fails:** A2 is UNREADABLE / TASK DOES NOT NEED INTEGRATION, and exp stays unread.

   Softplus is read by the same branches at 1e-3, with its own budget scope (DESCENDING runs counted as at 3e-3).

3. **Which H2 joint is cited (NOTE 4, NOTE 5).**
   - In the A2 output, the same-rate H2 joint is the registered one. The minimal_h2 CROSS-RATE joint (`G3_SSM_RESULTS.md`)
     is reported as superseded.
   - The same holds when A2 is UNREADABLE (G0) or TASK DOES NOT NEED INTEGRATION (GI): then no H2 joint is registered,
     and the cross-rate joint is NOT promoted back.
   - The same-rate joint combines an A2 label (kalman) with H2 existence labels (koopman). That is a conjunction of
     labels, not a cross-host contrast.
   - Amendment 2 says each A2 run starts from the same weights and walk stream as its block-A twin. Across hosts that is
     unverified, and nothing uses it: every test is unpaired.

4. **Driver (MINOR 2, NOTE 6).**
   - The base-root precheck now also requires every H and H2 continuation checkpoint (`H/<cls>_built_moves_m32_s<seed>/<cls>.pt`,
     `H2/<cls>_built_moves_m32_lr1e-3_s<seed>/<cls>.pt`). The final artifact count already required them, so an
     incomplete root is now refused before training rather than ~3 h later.
   - A2-only mode refuses a launch without an explicit `DRV_GPUS`, because the default `1` is GPU index 1. The launch
     line sets `DRV_GPUS=0`.
   - `p0/A2` is no longer created before the checks pass; `drv_md5_guard` creates it.
   - A lock refusal in A2-only mode is logged to `ssm_sign.log`, because the log switches to `ssm_sign_a2.log` after the
     lock is taken.

5. **Selftest (MINOR 3).** Added in the `a2_after_minimal_h2` configuration:
   - the Holm-family case of item 1;
   - A2 G0 failure (signed 5/8): READING suffix `UNREADABLE (signed SOLVED 5/8 < 6)`, every A2 label gated, every H2
     joint None;
   - A2 GI failure (index 2/8): suffix `TASK DOES NOT NEED INTEGRATION (index SOLVED 2/8, ...)`, every H2 joint None;
   - A2 abs UNMEASURED (abs 5/8 vs signed 8/8): suffix `UNMEASURED`, no abs H2 joint;
   - A2 exp NON-NEGATIVE LEARNS BETTER (signed 6/8 at 0.93 / 0.60, exp 8/8): label reached, no exp H2 joint.

6. **Stated (NOTES 2, 3).**
   - **Reproduction is content-equal, not byte-equal.** The minimal_h2 re-analysis equals `G3_SSM_VERDICTS.json` as
     parsed JSON. The summary dicts `solved` / `runaway` iterated a set, so their key order followed the hash seed;
     they are now built over sorted keys (exempt file, same content). "Reproduces exactly" in Amendment 2 means
     content-equal.
   - **Sec. 5's seed statement is stale.** Its "no run dir in the repo carries them" no longer holds: RANKWD (torus
     attention, koopman) used the labels 2100-2107. No run, checkpoint or contrast is shared.

7. **Freeze rule.**
   - Launch only with `DRV_GPUS=0`.
   - From launch to `.ssm_sign_a2_done`, no guarded file may be edited, this pre-registration included. Its md5 sits in
     `p0/A2/code_md5.txt`, and the driver re-checks it before the analysis.
   - The start check needs a tree with no modified tracked file anywhere. Other sessions' work (e.g. FACTORIAL_N8) must
     be committed or untracked at launch.

**Checks after this amendment (CPU).**
- `test_ssm_sign`: 40/40 PASS.
- `analyze_ssm_sign --selftest`: PASS, including the five new cases above.
- `--config minimal_h2` on the real root: equal to `G3_SSM_VERDICTS.json` as parsed JSON. The text output is equal
  apart from a SyntaxWarning line from `model_gated.py`, which goes to stderr.
- Driver dry runs (`DRV_DRYRUN=1`, `CUDA_VISIBLE_DEVICES=""`), in a fresh scratch clone with the three files committed
  there. The copied p0 has eval.json with paths rewritten and empty placeholder checkpoints.
  - No `DRV_GPUS`: REFUSED.
  - One H continuation checkpoint removed: REFUSED before launch. The same for one H2 checkpoint.
  - `DRV_GPUS=0` without A2:
    - provenance: 104 files, 0 mismatches;
    - 40 launches, all `--lr 1e-3 --device cuda:0` into `p0/A2/`, seed-major;
    - fails closed on 80 missing artifacts, none outside A2; no marker.
  - Synthetic A2 (block-A eval.json relabelled lr 1e-3; plumbing only):
    - 40 skips and 320 artifacts;
    - analysis with the 3-cell labels, the sensitivity line, the A2 suffix and the same-rate H2 joint;
    - `.ssm_sign_a2_done` set; `.ssm_sign_done` not touched.
- The real `runs/ssm_sign/p0` is untouched: 104 eval.json, 136 checkpoints, `config.txt` = minimal_h2, `config.txt` and
  `code_md5.txt` still dated 2026-10-23 18:24. No `p0/A2`, no `G3_SSM_A2_*`, no `.ssm_sign_a2_done`.

**Launch (after a commit of every file):**

    cd /home/prashr/mapformer && G3_CONFIG=a2_after_minimal_h2 DRV_GPUS=0 setsid nohup bash run_ssm_sign.sh > /dev/null 2>&1 &
