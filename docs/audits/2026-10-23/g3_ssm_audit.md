# G3_SSM (G-3) independent code check (rule 29), 2026-10-23, before launch

Scope: `G3_SSM_PREREG.md` (read in full), `model_ssm_sign.py`, `train_ssm_sign.py`, `construct_ssm_sign.py`,
`gate_ssm_sign.py`, `analyze_ssm_sign.py`, `run_ssm_sign.sh`, `test_ssm_sign.py` at HEAD 6b42679; precedents
`model_signroute.py` / `train_signroute.py` (task, recipe, step readout), `model_wrap_exist.py` / `WRAP_EXISTENCE_*`
(construction and LOST rule), `docs/paper/MAIN_APPENDIX2.md` D6. Blind to G-3 batch results (none exist; `runs/ssm_sign/`
absent and not created). The pilot (seeds 2180-2183, outside the batch) is in the pre-registration and was read, as were
its eval.json files and built checkpoints. CPU only; scratch checks on seeds 2170-2179 plus the pilot checkpoints. Check
scripts and outputs: `docs/audits/2026-10-23/g3_ssm_check/`. **No code edited.**

**No BLOCKER.** The model is the stated recurrence, the classes and step modes do what they say, the init is the attention
arm's bitwise, the task and gate reproduce, the constructions are exact, the analysis selftest passes and the driver
fails closed. Six MAJOR points (one a code bug in the joint reading, one an undetected exp failure mode, two about what the
existence block measures, one framing, one cost) and eleven MINOR.

## Verified (no action)

1. **Recurrence (rule 19), re-derived independently** (`c1_recurrence.py`). An auditor-written numpy complex128 loop
   computed from the raw state dict: steps s = E W_in^T W_out^T, the class function, the moves mask, angle = omega * Delta,
   H_t = diag(exp(-i angle_t)) H_{t-1} + k_t v_t^T, y_t = Re(conj(q_t)^T H_t) per head, per-head RMS, o_proj, residual,
   FFN (exact-erf GELU), out head. Against `SignSSM.forward` in float64, 5 classes x 2 step modes, steps of O(1) rad:
   max relative error 2.5e-13 (signed / abs / exp / index), 2.3e-12 (softplus moves), 5.3e-11 / 6.0e-10 (exp / softplus
   with every token stepping: cumulative angles of hundreds of rad). The stated gauge holds exactly: exp(-i omega Delta)
   with omega equals exp(+i omega Delta) with -omega (difference 0.0). The rotation direction alone at fixed omega is not
   a gauge (1.5), so the sentence "omega -> -omega" is the right one. The 39 unit checks rerun: 39/39 PASS (2.7 s CPU).
2. **Step classes and gates.** moves: only the two move tokens step, every observation's Delta is exactly 0 and receives
   no gradient through the step path; all: every token steps. index: every token steps 1 regardless of `steps`, so the
   angle is omega * token index (an LTI rotary SSM); the unused step map still decays under AdamW (irrelevant). none: mixer
   output 0, memoryless (pilot only, not in the batch).
3. **Init identity at trainer level** (`c1b_init.py`). The trainers' own order (seed, env, model), seeds 2177-2178, m 32
   and 8, signed / abs / softplus / exp: 24 of 24 shared tensors bitwise equal to `model_signroute.ARMS[arm]`, step tables
   equal (0.0), only `layers.0.y_norm.weight` extra. The index and memoryless arms share the signed attention arm's
   tensors too. `GridWorldSteps` draws its fixed map from its own RandomState, so building the evaluation env before
   training (train_ssm_sign does, train_signroute does not) changes no stream. The "past-pi start" count is right for exp
   (12.9 of 32 channels per head at m 32, 21.2 at m 8).
4. **Task = the attention ring task.** Same environment class and arguments, `train.train` (loss on revisits,
   ParallelBatchGenerator -> env.generate_batch, base seed torch.initial_seed() = seed), cosine 5% warmup to 10%, AdamW wd
   0.05, clip 1.0, 3 workers, `train_cancel.evaluate` on np seed 10^6 / env seed 10000 / 400 walks at T and 4T.
   **Gate rerun reproduces the table exactly** (`gate_out.txt`): all four rows, chance, every n-gram order, retrace
   (0.8636 / 0.8085 / 0.7917 / 0.7905), token ids in vocabulary; the pilot index 0.7554 and memoryless 0.4959.
5. **Constructions on the pilot checkpoints, recomputed from raw weights** (`c3_built.py`). built_s2183, built_s2180,
   built_s2183_all: only w_in, w_out, omega differ from the source; angle error mod 2 pi <= 8.3e-7 rad on every token
   (observations against 0 in the all-mode builds); constraints hold; frozen accuracy on CPU equals the source exactly
   (1.0000; 0.9792) and the stored value. The built checkpoint's `construction.angle_err_moves` reaches the analysis
   through train_ssm_sign's `frozen` record (checked on a CPU smoke continuation).
6. **The (f) / (g) omega' choice is legitimate and disclosed**: with steps = moves no token is annihilated, any omega'
   reproduces the map, and Sec. 4 declares the choice (rule 15). Its consequence is MAJOR 2.
7. **Analysis.** Selftest PASS (every READING branch, all four contrast labels, HELD / PERTURBED AND RE-FOUND / NOT HELD /
   NO EXACT CONSTRUCTION, n_s < 6, divergence, block B, the recipe guard). Recipe, lr, epochs, init_built and len(losses)
   asserted per run; the init_built string the analysis expects equals the one the driver passes. Power numbers in Sec. 7
   check (Fisher 0.026 / 0.077 / 0.041 / 0.041). Holm per test over the stated families.
8. **Cost arithmetic** checks: 52,000 / 42,400 / 27,200 / 15,200 run-epochs; pilot `s_per_epoch` 5.0-6.1 s at 4 per GPU
   (1.25-1.52 s per run-epoch), 2.1 alone.
9. **Driver.** flock single instance; clean tree at start; md5 guard over the 101 imported mapformer modules plus
   driver, lib and prereg (104 files) at three points; DRV_MAXPG / DRV_MINFREE / DRV_SPACING assigned after sourcing (rule 27); `running()` matches the exact
   output dir; eval.json skip; constructions on CPU, skipped only when report.json exists; NWANT arithmetic correct for
   every switch setting; artifacts before the marker. **Dry run** (scratch copy of the tracked top-level files in its own
   git repo, DRV_DRYRUN=1, DRV_GPUS=0, synthetic sources = the pilot s2183 signed model for every seed): 208 launches (the 8 synthetic sources
   skipped), by block A 72 / A2 40 / B 32 / H 32 / H2 32, lr 3e-3 / 1e-3 / 3e-3 / 3e-3 / 1e-3, epochs 300 / 300 / 300 /
   100 / 100, steps moves except B, no duplicate output dir; all 8 constructions ran on CPU (about 1 min in total) and
   wrote four built checkpoints each; the final check listed 416 missing artifacts (464 wanted, 48 present) and failed
   closed without the marker. Without sources, stage 2 fails closed (`drv_require`, read from the code). Trainer smoke
   (CPU, 2 epochs) from scratch and from a built checkpoint writes the eval.json the analysis expects.

## MAJOR

**MAJOR 1 (code bug): the joint SEARCH COST ignores the GI gate for abs.** In `analyse`, `prim[m]` is gated but
`v["A"]["abs|32"]["label"]` keeps the raw label, and the joint reads that. Synthetic case (`c4_joint.py`: index 2/8 at m
32): READING = "TASK DOES NOT NEED INTEGRATION ..." while `joint["abs"]` = "SEARCH COST". (softplus / exp are gated
correctly; G0 failure is caught via n_s < 6.) **Fix (one line):** after `prim[m] = ...` set `lab["label"] = prim[m]`, or
require `v["gate"][32] == "pass"` in the joint. Add the case to the selftest.

**MAJOR 2: the softplus / exp constructions are worse conditioned than the abs one by construction choice, so their E2
labels are not class statements.** Move-angle sensitivity to an Adam-sized step (L1 norm of d angle / d parameter over
move embeddings, W_in, W_out, omega, averaged over the 128 move angles; a proxy and an upper bound, since Adam moves each
coordinate by at most about lr; `c3_built.py`): s2183 signed control 30, abs 116, softplus 191, exp 238; s2180 19, 50,
102, 136. Calibration on WRAP_EXISTENCE's attention constructions rebuilt with the committed `wrap_construct`
(`c3b_attn_calib.py`, s600-602): SO_bal 15-18, AO_bal 87-96 (ratio 4.9-5.8), and AO_bal was HELD at 1e-3. So the abs
construction is on the precedent's footing (the wrap itself costs ~4x: angles rem in [0, 2 pi), mean pi, vs the control's
|Theta| mean 1.16), while softplus / exp add another 1.7-2.7x from omega' = 9.06 / 6.28 on every channel (trained source
0.68) and target angles rem + 2 pi (mean 3 pi). The pilot's maximum continuation losses follow the same order at both
rates (abs < softplus < exp). **Fix:** register that a non-HELD E2 for softplus / exp is read as a property of
constructions (f) / (g) at that omega', not of the class (WRAP_EXISTENCE's own rule: "NOT HELD under an ill-conditioned
construction says nothing about the class"); print the sensitivity per construction in report.json. Optional: a moves-only
construction of softplus / exp at the abs construction's omega' (16-32 continuation runs).

**MAJOR 3: the LOST rule's 10x clause binds on an unusually stable SSM control, so E2 is not on the attention footing.**
Pilot max per-epoch continuation loss: SSM signed control 0.002 (1e-3) / 0.018 (3e-3) vs the attention SO_bal 0.015-0.112.
At 1e-3 the SSM abs construction peaks at 0.075, softplus 0.108, exp 0.132: LOST by "> 10x control and > 0.05", although
the attention AO_bal peaked at up to 0.154 and read not LOST (its control also spiked). Every LOST on record in
WRAP_EXISTENCE met the absolute clause (0.6-2.0). Transported unchanged, the rule makes "SSM PERTURBED vs attention HELD"
partly a statement about the controls. **Fix:** keep the registered rule; add a declared secondary E2' with LOST = max
loss > 0.3 only, and print each class's max-loss range beside the attention ranges (AO_bal 0.024-0.154, SO_bal
0.015-0.112). A joint read on E2' is a secondary.

**MAJOR 4: exp at 3e-3 can fail by a runaway the analysis does not see.** Pilot s2183 exp: move_step_absmax 79,713 (every
other pilot <= 11.8), move latents antipodal (cos -0.99995), loss 1.30 -> 2.84 (above the memoryless 2.08), final 1.77,
classified STALLED, finite. The code would count such seeds as ordinary unsolved runs and BR exp would read SIGNED LEARNS
BETTER; Sec. 8's reading ("an exp failure at 3e-3 as instability unless A2 agrees") is prose only, and A2 may be dropped or
UNREADABLE (MAJOR 6). The runaway also follows the signed arm's collective route (z_- = -z_+), which under exp sends one
step to e^{+a} and the other to e^{-a} with no cancelling fixed point, so it is neither clean "instability" nor clean
"search". **Fix:** register RUNAWAY = move_step_absmax > 10^3 or a post-warm-up epoch loss above 2.08 (memoryless); `_row`
reads `move_step_absmax`; every cell prints k_RUNAWAY; a cell with k_RUNAWAY >= 2 keeps its label with the suffix "(k
RUNAWAY)" and licenses no learnability sentence. Applies to block B's exp too.

**MAJOR 5 (framing): this SSM is unnormalised rotary linear attention, so a null cannot mean "attention-specific" or
"the recurrence gives gradient through time".** The code computes the model as the causal product of rotated q and k
(Mamba-3 Prop. 3); the recurrence is the same function with the same gradients. What differs from the anchors is softmax
(with attention dropout and 1/sqrt(d)) replaced by an unnormalised kernel plus per-head RMSNorm. **Fix (Sec. 1, Sec. 8):**
cost present -> "the cost is not specific to softmax attention: it appears in a one-layer diagonal complex SSM with an
input-dependent rotation (equivalently, unnormalised rotary linear attention)"; no cost -> "the cost is not shown without
softmax normalisation; a scope boundary at softmax", not "attention-specific". Drop "the recurrence gives gradient through
time" as the rival's mechanism, or name a mechanism that differs between the two forms (none does here).

**MAJOR 6 (cost): A2 is at real risk of G0 failure, and H2 is tied to it.** The one signed pilot at 1e-3 (s2180) ended at
loss 0.064 > 0.05, STALLED, acc 0.979, latent cos +0.09 (not the cancelling code), and exp at 1e-3 SOLVED where it did
not. If that is typical, A2 (12,000 run-epochs, 23% of the full batch) reads UNREADABLE and the 1e-3 joint cannot form.
H2 (3,200 run-epochs) is readable without A2 (its sources are block A's) and is the cheap half that answers "kept at a
gentler rate" when the pilot already says every non-negative construction is LOST at 3e-3. `G3_BLOCK_A2=0` drops both.
**Fix:** a separate switch `G3_BLOCK_H2` (driver loop and REQ, NWANT, `load(need_h2=...)`); see the recommendation.

## MINOR

1. Rule-2 r(loss, acc) includes H2 continuations (`k[0] != "H"`); the comment says blocks A and B. Fix:
   `k[0] in ("A", "A2", "B")`.
2. The signed control's E1 is not checked (WRAP_EXISTENCE A3 gated E2 on the control's E1 too). Fix: require
   |frozen - source| <= 0.005 for the control as well.
3. NOT HELD here is "everything not HELD or PERTURBED"; WRAP_EXISTENCE A4 had a statistical NOT HELD and PARTIAL otherwise.
   Declare the difference.
4. Continuations are 100 epochs (WRAP_EXISTENCE 300): warm-up 5 epochs instead of 15 to the same peak, a third of the
   recovery budget. "LOST is a warm-up event" is contradicted by the pilot's softplus at 3e-3 (maximum at epoch 28), and
   softplus / exp at 3e-3 were still DESCENDING at 100 (0.076, 0.167), so PERTURBED vs NOT HELD is set by the budget. The
   joint reads both the same way, so no registered label depends on it; say so.
5. The mid-batch `clean_tree` calls trip on ANY modified tracked file (8,530, incl. `.claude-memory/` and docs) while
   another session (SIGNSEP) works in this repo; children inherit the flock fd (verified in scratch), so after such a
   `drv_fail` a relaunch is REFUSED until every running child exits. The md5 guard already covers the code. Fix: restrict
   the mid-batch checks to `git diff --quiet HEAD -- "${FILES[@]}" "${GUARD[@]}"` (precedent drivers check only at start).
6. Slots are shared with the same user's other batches: during this check the SIGNSEP pilot held 4 slots on cuda:1, so a
   DRV_GPUS=1 launch would have waited. The both-GPU timing (8 runs x 3 data workers beside the other user's jobs on 32
   cores) is unmeasured.
7. Sec. 5 disclosure: the 1e-3 non-signed pilots (abs 0.736, softplus 0.905, exp SOLVED 0.9986) finished at 03:07-03:08,
   before the signed 3e-3 pilot finished (03:15), so they were available when 3e-3 was fixed; say whether they were read.
   Bias to the primary is small: abs's pilot was also better at 3e-3 (0.810 vs 0.736), the selection is on the reference
   arm (G0), and claims are scoped "at lr 3e-3".
8. NON-NEGATIVE LEARNS BETTER is practically unreachable at P under G0 (Fisher cannot fire for abs above a signed arm at
   >= 6/8; only the accuracy branch, which needs SOLVED signed runs scoring 0.05 below abs). Say so.
9. softplus also starts past pi on 9.4 of 32 channels per head at m 32 (15.6 at m 8); Sec. 8 mentions exp only.
10. The analysis reads a hard-coded ROOT and writes G3_SSM_VERDICTS.json to a hard-coded REPO; the driver passes no
    `--root`. Harmless in production; pass `--root "$R"`.
11. A2 repeats the index arm (2,400 run-epochs); GI at a second rate adds little if A2 is kept.

## Recommendation on the configuration

The primary (abs vs signed at m = 32, lr 3e-3, with G0, GI, E1, E2) needs block A at m = 32 (40 runs) and H (32): the
pre-registration's minimal row, 72 runs, 15,200 run-epochs, 5.3-6.4 h on one shared 4090 (2.6-3.2 h on both). It needs
`SIZES = (32,)` in driver and analysis, and `reading()` must stop requiring m = 8: an edit, so it goes in Amendment 1 and
the selftest and the dry run are rerun.

**Recommended: minimal + H2, 104 runs, 18,400 run-epochs, 6.4-7.8 h on one shared GPU (3.2-3.9 h on both, if the CPUs
keep up)**, with H2 decoupled from A2 (MAJOR 6). The pilot predicts LOST at 3e-3 for every non-negative construction, so
H alone answers "kept?" only at a rate that perturbs; H2 adds the gentler rate for 3,200 run-epochs. Drop A2 (G0 risk;
restores the 1e-3 joint only if signed solves at 1e-3), m = 8 (no prior) and B (a separate Mamba-faithfulness question
that carries the exp runaway and the full-turn burden). If one add-on is affordable, take A2: it is the only readable
from-scratch exp if exp runs away at 3e-3 (MAJOR 4). Whatever the configuration, MAJOR 1-5 apply.

## Proposed Amendment 1 (text for G3_SSM_PREREG.md)

> **Amendment 1 (2026-10-23, before launch, from the independent check `docs/audits/2026-10-23/g3_ssm_audit.md`; no
> batch result exists).**
> 1. *Joint reading (code fix).* The joint SEARCH COST uses the GATED block-A label: if G0 or GI fails at m = 32 no joint
>    is formed. `analyze_ssm_sign.analyse` stores the gated label for abs; a selftest case covers GI failure.
> 2. *Construction conditioning.* E2 for softplus and exp is read as a property of constructions (f) / (g) at
>    omega' = 2 pi / ln 2 and 2 pi, not of the class, unless HELD; the class-level E2 statement is made for abs only.
>    report.json records each construction's move-angle sensitivity (L1 norm of d angle / d parameter; pilot s2183:
>    control 30, abs 116, softplus 191, exp 238; attention AO_bal 87-96).
> 3. *Declared secondary E2'.* LOST' = maximum per-epoch continuation loss > 0.3 (the absolute clause alone). HELD' /
>    PERTURBED' / NOT HELD' as E2 with LOST'. Reported beside E2 with each class's max-loss range and the attention ranges
>    (AO_bal 0.024-0.154, SO_bal 0.015-0.112), because the SSM control peaks at 0.002-0.018 and the 10x clause then binds.
>    E2 stays registered.
> 4. *RUNAWAY.* A run is RUNAWAY if its final move_step_absmax > 10^3 or any epoch after the warm-up has loss > 2.08 (the
>    memoryless arm). Every cell prints k_RUNAWAY; a contrast whose non-negative cell has k_RUNAWAY >= 2 keeps its label
>    with "(k RUNAWAY)" and licenses no learnability sentence for that cell.
> 5. *Framing.* The model is computed as unnormalised rotary linear attention, which is the same function as the
>    recurrence. SIGN COST IN THE SSM reads "not specific to softmax attention"; NO SIGN COST reads "not shown without
>    softmax normalisation (a scope boundary)", not "attention-specific". The rival is stated as "softmax competition is
>    the source of the cost".
> 6. *Configuration:* [the user's choice]; if minimal + H2: `SIZES = (32,)` in driver and analysis, `reading()` without
>    m = 8, Holm over the 2 BR cells, a `G3_BLOCK_H2` switch, then the selftest and a dry run again before commit.
> 7. *Smaller items.* Rule-2 r over blocks A / A2 / B only; the signed control's E1 also checked; NOT HELD declared as
>    "not HELD and not PERTURBED" (WRAP_EXISTENCE's statistical NOT HELD is not used); with 100-epoch continuations
>    PERTURBED vs NOT HELD depends on the budget and the joint treats them alike; mid-batch tree checks restricted to the
>    batch's files; the 1e-3 non-signed pilots had finished before 3e-3 was fixed [state whether they were read];
>    NON-NEGATIVE LEARNS BETTER at P is reachable only through the accuracy branch; softplus starts past pi on 9 of 32
>    channels per head.
