# DELTA_FREE independent code check (rule 29), 2026-10-20, before launch

Scope: `DELTA_FREE_PREREG.md`, `model_pope_delta.py`, `analyze_delta_free.py`, `run_delta_free.sh`, the `train_variant.py`
additions, at HEAD 7dc669a; account under test `docs/paper/expansion/MAPPOPE.md`; recipe precedent `run_paper2x2.sh` /
`PAPER2X2_PREREG.md`. Blind to DELTA_FREE batch results (none exist: `runs/delta_free/` absent, not created). The pilot
(seed 1690, outside the batch) is already in the pre-registration and was read; its checkpoints were re-read for the
weight-decay check below. Test runs in the scratchpad (`delta_free_check/`) on seeds 1670-1677, CPU only, no GPU used.
Check scripts and outputs copied to `docs/audits/2026-10-20/delta_free_check/`. No code edited.

**No BLOCKER.** The code does what the pre-registration says: the arms are the stated classes and command lines, the
pinned arms reproduce the parent commit bitwise, twins at a seed differ only in delta, the frozen delta never moves and
gets no decay, the clamp is gone from every place the free / frozen arms use, the analysis uses the effective delta
everywhere and its selftest passes. Five MAJOR points, none a code bug:
1. the "path helped" clause has no size threshold;
2. weight decay, not learning, moves the free deltas (pilot);
3. the manipulation is in practice the init, not the clamp;
4. the account's path-side prediction has no registered readout;
5. the licensed paper sentence ("freeing delta reverses the interaction") says more than I measures.

## Verified (no action)

1. **Arms and command lines.** `RECIPE` in `run_delta_free.sh` equals `run_paper2x2.sh`'s `train_variant` line flag for
   flag: 300 epochs x 98 batches x 128, T 128, 1 layer, 2 heads, d 128, `--n-landmarks 0`, `--schedule cosine`,
   `--lr 1e-3`, serial data, no `--fast-attn`, fp32. The pilot checkpoint's saved config agrees (epochs 300, lr 0.001,
   cosine, n_steps 128, batch 128, n_batches 98, data_workers 0, fast_attn False, deterministic False, 300 losses).
   `VARIANT_MAP`: `PoPE-Flat` -> `MapFormerWM_RoPEIndex_PoPE`, `MapPoPE-Flat` -> `MapFormerWM_PoPE` (both unchanged),
   the three new names -> the three new classes. Eval = PAPER2X2's (`eval_noise_refine`, env seed 10000, np seed
   1234 + s, 100 trials, T 128 / 512 / 1024).
2. **Pinned arms = parent commit, bitwise (CPU, independent of the author's GPU check).** af839b3 exported by
   `git archive` (import path verified to resolve there; its `VARIANT_MAP` lacks the new arms) vs HEAD, `train_variant`
   seed 1674, 3 epochs x 4 batches x 16, T 64, the batch's other flags, `--device cpu`: per-epoch losses identical and
   every state-dict tensor `torch.equal` (PoPE-Flat 23/23, MapPoPE-Flat 25/25).
3. **Twins differ only in delta, through training** (`twin_check.py` T1, seeds 1670-1671, both phases). The free arm's
   draw, moved into the clamp's interior ([-2 pi + 0.3, -0.3]), was written into both the pinned and the free model, which
   were then trained 2 epochs x 4 batches. Losses and every weight came out bitwise identical, and no entry reached the
   clamp. So the init, the data stream, the dropout masks and the optimiser are paired, and the clamp is the only
   difference in the forward. The real twins (zero init vs the draw) do diverge (T3), so the check is not vacuous.
   Selftest: only `pope_delta` differs at init, building a free arm leaves the global RNG untouched, and FI / FP / ZP
   draw the same delta at a seed.
4. **Frozen arm** (T2, seeds 1672-1673): after training `pope_delta == pope_delta_init` exactly. It has no AdamW state
   entry and its grad is None, so it gets no update and no decay. `clip_grad_norm_` skips None grads. Its init equals
   the free arm's draw. The analysis also prints CODE CHECK FAILED if a frozen delta moved.
5. **The clamp is removed everywhere it is used.** By grep, `pope_delta` is clamped only in the forward passes of
   `model_pope.py`, `model_pope_decay.py`, `model_pope_t3.py` and `model_pope_ablate.py`, and in post hoc scripts, none
   of which these classes or this batch use. Nothing in `train.py` or `train_variant.py` clamps in place.
   `eval_noise_refine` and `analyze_delta_free.load_run` rebuild from `VARIANT_MAP` and load strictly, including the
   `pope_delta_init` buffer, so they run the unclamped forward. The analysis reads delta through `delta_eff` in
   `forward_manual`, `coherence` and `kernel_checks`. `deadzone_share` deliberately reads the raw value.
6. **Parameterisation.** `pope_delta` has shape (n_heads, d_head) = (2, 64): one value per (head, frequency), as in
   `model_pope` (PoPE eq. 6 has delta_c per component, and we keep one per head). Channel c is aligned with frequency c,
   ordered fast to slow (index: theta_c = 10000^(-c/64); path: omega decreasing).
7. **What "pinned" means.** `torch.clamp` passes the gradient at exactly 0 and at -2 pi, and gives 0 outside. So the zero
   init does get a gradient at step 1. An entry pushed above 0 is then absorbed: its gradient is 0, Adam momentum keeps
   pushing it up, and decoupled decay is multiplicative and cannot cross 0 (T4: never back to <= 0 in 3000 steps). On the
   PAPER2X2 checkpoints, PoPE-Flat has 98.4-100% of raw entries >= 0 (max raw 0.004), so its effective delta is exactly 0.
   MapPoPE-Flat has 93.8-97.7% (minimum raw down to -1.59), so the path arm is mostly but not fully pinned (pilot 89%).
   **Representational note (for the wording):** [-2 pi, 0] spans a full period, so the pinned class can represent every
   free solution exactly (delta mod 2 pi). The pin is an optimisation effect (zero init sitting on an absorbing edge),
   not a limit on capacity. Any verdict is a learnability statement, as in the sign result's 2026-10-12 correction.
8. **Weight-decay factor.** The sum of lr over the schedule is 16.097, so decay alone multiplies by
   prod(1 - lr_k x 0.05) = 0.4471 = exp(-0.805). This confirms the prereg's 0.447.
9. **Analysis.** `primary_label` implements the six labels in the registered order. I = dP - dI, paired by seed.
   `signflip_p` is exact (256 patterns) and handles ties correctly: zeros contribute nothing. `ub(dI)` is the upper end
   of the 90% test-inversion CI. SOLVED = `classify_run` "registered" (final 5% < 0.05). Fisher on path SOLVED. The
   verdict JSON is written before the secondaries. A missing eval entry at T = 128 -> SystemExit; a missing checkpoint
   -> FileNotFoundError, not dropped. The floors regex parses the eight batch seeds: mean 0.9381 = the prereg's.
   `--selftest` passes (every primary and S1 label, pairing, RNG, rebuild 0.0e+00, `delta_eff`, kernel_probe agreement).
   **End to end without the driver** (`e2e_run.sh`, seeds 1670-1677, tiny CPU recipe, 3 eval walks): 40 trainings,
   eval and analysis ran. Rebuild max |dlogit| 0.0e+00, probe accuracy = eval accuracy, frozen moved 0.000, primary
   UNREADABLE (untrained path arm), as it should be.
10. **Driver.** `flock -n` (one lock for all modes). In batch mode, the clean-tree and committed-file checks run before any
    wait. `DRV_GPUS` is required. The md5 guard covers 92 imported modules plus `lib_driver.sh` and the driver, and is
    rechecked before the eval and before the analysis. Seed outer, arm inner. `DRV_MAXPG=4` and the other knobs are
    assigned after sourcing (rule 27). Slot counting uses every `mapformer.train_*` on the device (rule-23 awk). Resume
    skips finished runs (`<arm>.pt` is written only at the end of training) and running ones (exact `--output-dir`
    match). `drv_wait_dir "$R/p0/"` cannot match the pilot's directory. All 40 checkpoints are required before the eval.
    Artifacts come before the marker. The eval runs after training on the first `DRV_GPUS` device; the analysis runs on
    CPU.
11. **Seeds 1600-1607** are unused by any run directory, driver or other prereg. The only other `16xx` string in a driver is
    a sequence count. Smoke 1680-1687 and pilot 1690 are outside the batch.

## Findings

### MAJOR 1 -- "path helped" has no size threshold: a ceiling-level +0.001 can flip SUPPORTED to NOT SUPPORTED (helps both)

`path_helped = (mean dP > 0 and p_sf(dP) < .05) or (FP solves more, Fisher p < .05)`. Its mirror, `path_hurt`, requires
mean dP <= -TAU_P. At the ceiling, FP >= PP on every seed with at least 6 seeds strictly above gives p_sf = 0.031 for a
mean dP of about +0.001. Example: PP 0.9983 / 0.9961 / 0.9993 on six seeds, FP 1.000 everywhere. PAPER2X2's MapPoPE-Flat
was below 1.000 on 3/8 seeds (0.9961-0.9993). Simulation (`ceiling_flip_sim.py`; PP resampled from those values, FP
1.000 on every seed, true dI +0.08 / +0.12): **3.6-3.9% of batches read NOT SUPPORTED (helps both)** where they would
otherwise read SUPPORTED. With FP distributed like PP it is 0.2-0.4%. Under the prereg's own decision table, NOT
SUPPORTED withdraws the account, so a 0.001 difference below the noise floor (the prereg sets TAU_P = 0.01 as "a real
loss, not ceiling noise") must not decide it.
**Fix (analysis, one condition):** path helped = (mean dP >= +TAU_P and p_sf(dP) < .05) or (FP solves more, Fisher
p < .05), the mirror of path cost. State that with PP at ~0.999, "helps both" is unreachable by accuracy (it stays
reachable by SOLVED and in the selftest's PP = 0.96 case). Add a selftest case: FP 1.000 on all seeds, PP 0.998 on six
-> SUPPORTED.

### MAJOR 2 -- weight decay, not learning, moves the free deltas; "free" vs "frozen" therefore differ by decay as well

Pilot checkpoints (`pilot_decay_check.py`, seed 1690): the free deltas moved mean |delta - init| = 1.74 (index) / 1.82
(path). Decay alone (delta_T = 0.4471 x init) predicts 0.553 x mean|init| = 1.74. The residual from the decay-only
trajectory is **0.169 (index) / 0.221 (path)**, median 0.066 / 0.175. Learning sits mostly in the fastest quartile of
channels (|residual| 0.41 / 0.35, vs 0.09 / 0.18 in the other three). Free-delta values agree with the init at
correlation 0.934 / 0.985. Consequences:
- The prereg's pilot reading, "free arms moved far from their draw ... and did NOT return to the pin", describes decay.
  Under decay alone the end distribution is U(-2.81, 0), whose re-pinned share (|wrap| < 0.1) is 0.036. The pilot
  measured 0.016 (index) and 0.109 (path), so the path arm's learning moved some entries toward the pin, three times
  the decay-only rate (n = 1, descriptive).
- Decay is not neutral on the circle. An entry drawn in (-2 pi, -pi) (wrapped peak in (0, pi)) is carried through
  +/- pi into (-2.81, -1.40). On slow channels, where the kernel is cos(theta D - delta) with theta D small, that turns
  an anti-recency kernel into a recency one. So over training, decay pulls FI's slow channels toward the pinned arm's
  recency shape. For the primary this is conservative: it pulls FI toward PI. But S3 (FI's OOD cost) and S5 (kernel
  shape) partly read a schedule artefact.
- **S1b (ZP - FP, "does the route back matter") is confounded.** FP's deltas shrink by 0.447 and ZP's do not, so ZP vs
  FP is "decay + weak learning vs neither", not "learning vs none". S1 (ZP - PP) is unaffected, because ZP is cleanly
  "random and fixed".

**Fix, option B (recommended; no GPU, small code change, design content, so the author's call):** exclude `pope_delta`
from weight decay in the free arms. Delta's decay is then a delta-only property: twins still differ only in delta, and
FP - ZP isolates learning. It needs `train.py` to build AdamW with a no-decay group for parameters carrying a flag,
default path unchanged, and a bitwise re-check of PoPE-Flat / MapPoPE-Flat and one non-PoPE variant (as in item 2).
The pinned arms keep decay, which is inert on their absorbed entries. **Option A (acceptable):** keep the recipe and
fix the interpretation instead. Define "free = U(-2 pi, 0) init, unclamped, learnable under AdamW decay (x0.447 over
the run)". Reword S1b as "decay + learning vs neither". Add to S4 the residual |delta - 0.4471 x init| by frequency
quartile (the learned part), with the decay-only re-pinned share 0.036 printed beside the re-pinned share.

### MAJOR 3 -- the manipulation is in practice the init, not the clamp

At the end of the pilot, 99.2% (index) / 96.1% (path) of free entries lie inside [-2 pi, 0], the range from -4.17 to
+0.10. For almost every entry the clamp would have been inactive. So PoPE-FreeDelta is nearly "PoPE with the PoPE
paper's alternative U(-2 pi, 0) init". The account says delta is "pinned near 0 by its clamp", and the paper paragraph
would inherit that. The batch cannot tell "zero init on the clamp's absorbing edge" apart from "the clamp". The PoPE
paper itself says zero init "is important for length generalization", so S3's predicted OOD index cost is
attributable to the init, not to removing the clamp.
**Fix (wording, required):**
- pinned = "zero init on the absorbing edge of the [-2 pi, 0] clamp";
- free = "U(-2 pi, 0) init, clamp removed (inactive for >= 96% of entries in the pilot)";
- SUPPORTED licenses "the pin (zero-init delta held at 0 by the clamp's dead zone) causes ...", not "the clamp causes".
**Optional arm (user decision):** `PoPE-ZeroFree` (zero init, no clamp, learnable). It isolates the clamp: if
ZeroFree ~ FI, the clamp is the pin; if ZeroFree ~ PI, the zero init plus weak delta gradients is the pin (see MAJOR 2:
delta learns little outside fast channels). Cost: 8 index runs, 2 waves x ~20 min = ~40 min wall at 4/GPU on one shared
GPU (+20%). Register it as a secondary (ZeroFree - PI with the dI thresholds), not a primary.

### MAJOR 4 -- the account's path-side prediction has no registered readout, and the analysis has no ZP-PP contrast beyond T = 128

MAPPOPE.md predicts "free or frozen-random delta hurts or slows MapPoPE, since target coherence C falls below C0". At
T = 128 every path arm is at ceiling: PAPER2X2 MapPoPE-Flat 0.9961-1.000, and all three pilot path arms are at 1.000. So
the primary's (path cost) qualifier and S1 will almost surely read "unchanged" / NO COST, as the prereg itself says.
Noise floors for the candidate readouts (PAPER2X2 MapPoPE-Flat, n = 8) beside the pilot (n = 1, outside seeds):

| readout | PAPER2X2 PP range (sd) | pilot PP / FP / ZP | usable? |
|---|---|---|---|
| T = 128 accuracy | 0.9961-1.000 (0.0014) | 1.000 / 1.000 / 1.000 | ceiling |
| T = 128 held-out NLL | 0.000-0.015 (0.005), 4/8 exactly 0 | 3e-7 / 4e-7 / 7e-4 | descriptive only |
| loss AUC | 0.026-0.097 (0.033), bimodal; first epoch < 0.05 at 4-87 | 0.031 / 0.036 / 0.094 | ZP's 0.094 is INSIDE PP's range: not a registered readout |
| T = 512 accuracy | 0.965-0.998 (0.011) | 0.985 / 0.969 / 0.937 | declared |
| T = 1024 accuracy | 0.901-0.976 (0.023) | 0.939 / 0.926 / **0.801** | yes (OOD: robustness, rule 10) |

S3 has dP at T = 512 / 1024 but no ZP contrast, so the pilot's largest path signal (ZP 0.801 vs 0.939) appears in no
contrast.
**Fix (amendment, fixed before the batch; motivated by the pilot on outside seeds, which is legitimate):** add a
registered secondary **P1**, which does not change the primary. Per X in {ZP, FP}, c = X - PP at T = 1024, exact
sign-flip, Holm over the 2:
- **REMOVING THE PIN COSTS MAPPOPE'S LENGTH ROBUSTNESS** if mean <= -0.03 and Holm p < .05;
- **NO OOD COST** if the 90% CI's lower end is > -0.03;
- otherwise UNMEASURED.

TAU_P1024 = 0.03 is about 1.3 x PAPER2X2's sd and a quarter of the pilot's ZP gap. Declared beside it: the same
contrasts at T = 512, AUC and T = 128 NLL with a bimodality caveat. Wording allowed: robustness, not capability ("the
pin keeps MapPoPE's readout robust to length"). It may not say "the path phase uses the pin to solve the task", because
that is untestable at T = 128 in this batch. Under option A of MAJOR 2, an FP reading also carries the decay
qualifier.

### MAJOR 5 -- the licensed sentence "freeing delta reverses the interaction" says more than I measures

I = dP - dI is the delta x phase interaction. The +0.154 it is set beside is the score x phase interaction (PoPE vs RoPE
score). Saying "freeing delta reverses the interaction" invites the reading that the score x phase interaction changes
sign under free delta. That needs (FP - MapWM) - (FI - RoPE) < 0, which needs RoPE and MapWM (Vanilla) arms in the batch
(rule 12; the pilot's FI 0.816 vs PAPER2X2's RoPE 0.805 is cross-batch). Also, at the ceiling I ~ -dI + dP with
|dP| <= ~0.004, so "I <= -TAU_I with p < .05" follows from "dI >= 0.054 with p < .05": the interaction clause is the index
gain restated, not independent evidence.
**Fix (wording, required):** SUPPORTED licenses "freeing delta raises the index model by dI [CI], concentrated in R3+
(S2), and leaves the path model at ceiling; the delta x phase interaction is I". Drop "reverses the interaction" unless
RoPE + Vanilla are added. **Optional (user decision):** add `RoPE` and `Vanilla` (16 runs, ~4 waves x 20 min = ~1.4 h
wall on one shared GPU, +40%). The score x phase interaction under free delta then becomes a declared secondary.

### MINOR 1 -- an alternative mechanism for dI: on slow channels delta is a signed content gain

On channels where theta_c x 128 < 0.5 rad (index: c > 38, 25 of 64 per head), the kernel is flat in offset and equals
cos(delta_c). Pinned, it is +1: a non-negative softplus x softplus content term. Free, it has a random sign: a signed
content-only term. Freeing delta therefore also lifts PoPE's content non-negativity, whatever happens to the peak
location. S5's K(3) > K(1) pools channels and cannot separate the two. **Fix:** declared eval-only secondary **S7**, run
on CPU with the existing rebuild. In each FI checkpoint, reset delta to 0 on (a) the fast half (c < 32) and (b) the slow
half (c >= 32), and read T = 128 accuracy. Peak pinning predicts that (a) removes most of dI and (b) little. This is
diagnostic only, since the network was trained with its delta. The pilot's learning is concentrated in the fastest
quartile, which fits, but n = 1.

### MINOR 2 -- NaN accuracies are not refused

`load_raw` maps an eval `None` to NaN. Every comparison with NaN is False, so a NaN seed falls through to a label (e.g.
UNMEASURED) instead of failing. **Fix:** before `primary_label`, SystemExit unless every T = 128 value of every arm and
seed is finite.

### MINOR 3 -- the analysis does not assert the recipe of each checkpoint

ROBUST's precedent asserts it. **Fix:** in `analyse`, assert `config` epochs 300, lr 1e-3, schedule cosine, n_steps 128,
batch_size 128, n_batches 98, n_layers 1, n_heads 2, d_model 128, n_landmarks 0, `args.data_workers` 0, fast_attn False,
`variant` = arm, `seed` = s, and len(losses) = 300.

### MINOR 4 -- dropout-scale rescore

Both index arms are below ceiling. `DROPOUT_RESCORE` moved PAPER2X2's index arms by +0.011 / +0.013, and FI's kernel
differs from PI's. **Fix:** declare dI under the 1/(1-p) rescale as descriptive (eval-only), or state that none is run
(the current text says no re-score is declared, which is acceptable).

### MINOR 5 -- small hygiene

- The md5 guard does not cover the floors file the analysis reads. It is committed and the tree is checked clean at
  start, so the risk is a mid-run edit only. Add `docs/audits/2026-10-20/delta_free_floors_out.txt` to `GUARD`.
- The selftest does not cover "path helped by SOLVED", NaN input, or the MAJOR 1 ceiling case.
- `sf_ci` uses a 0.0005 grid, so with ties (ceiling path contrasts) its ends are approximate to one step. This is
  negligible against TAU 0.01 / 0.05.
- In S4, print the decay-only expectation beside the re-pinned share (0.036; MAJOR 2).

### Not findings (checked)

- Frozen parameters count in `params=` (204,053 for FreeDelta) because `model.parameters()` includes them; this is
  cosmetic.
- `clip_grad_norm_` sees delta's gradient in the free arms and not in the pinned arm's dead zone, so clipping differs
  between twins. This is inherent to "free" and not a confound.
- Building the model in eval consumes the private generator with an arbitrary `initial_seed`; the state dict (including
  `pope_delta_init`) overwrites it.
- Power table: reproduced logic (`power()` uses the registered `primary_label`). The thresholds are sensible.
  "NOT SUPPORTED (helps neither)" under a true null of 47-79% is a strength.

## Proposed amendment text (Amendment 1, before launch)

> **Amendment 1 (2026-10-20, before launch; independent code check `docs/audits/2026-10-20/delta_free_audit.md`, blind to
> batch results).** The code was verified: pinned arms bitwise equal to the parent commit (CPU), twins bitwise identical
> through training when delta is held in the clamp interior, frozen delta unmoved with no optimiser state, clamp absent
> from every path the new arms use, selftest and a CPU end-to-end run pass. Changes, fixed before any batch result:
> 1. *Path helped* (primary, branch 4/5) now requires mean dP >= +TAU_P with p_sf < .05 (or more SOLVED, Fisher p < .05),
>    the mirror of *path cost*. With PP at ~0.999, "helps both" is unreachable by accuracy. (A sub-noise +0.001 path
>    difference flipped 3.6-3.9% of simulated SUPPORTED batches to NOT SUPPORTED.)
> 2. *What free means.* [Option B:] `pope_delta` is excluded from weight decay in the free arms (FI, FP); the frozen arm
>    already has none; the pinned arms are unchanged. FP - ZP is then learning alone. [Or option A:] Free = U(-2 pi, 0)
>    init, unclamped, learnable under AdamW decay (x0.447 over the run). In the pilot, decay accounts for ~90% of the free
>    deltas' movement (residual 0.17 / 0.22 rad), and it carries draws in (-2 pi, -pi) through +/- pi. S1b reads "decay +
>    learning vs neither". S4 adds the residual from the decay-only trajectory by frequency quartile and prints the
>    decay-only re-pinned share (0.036).
> 3. *Wording.* Pinned = zero init on the absorbing edge of the [-2 pi, 0] clamp (the clamp passes the gradient at 0;
>    an entry pushed above 0 never returns). Free = U(-2 pi, 0) init with the clamp removed; in the pilot the clamp would
>    have been inactive for >= 96% of entries. So the batch tests the pin (zero init held by the dead zone), not the clamp
>    alone. The pinned class can represent every free solution (delta mod 2 pi), so any verdict is about learnability.
>    S3's index OOD cost may be the zero-init effect the PoPE paper reports. [Optional: add arm `PoPE-ZeroFree` (zero
>    init, no clamp), 8 runs, secondary ZeroFree - PI with the dI thresholds.]
> 4. *Registered path-side secondary P1* (motivated by the pilot on an outside seed; no primary decision depends on it):
>    for X in {MapPoPE-FrozenDelta, MapPoPE-FreeDelta}, X - MapPoPE-Flat T = 1024 accuracy, exact sign-flip, Holm over 2.
>    REMOVING THE PIN COSTS MAPPOPE'S LENGTH ROBUSTNESS if mean <= -0.03 and Holm p < .05; NO OOD COST if the 90% CI
>    lower end > -0.03; else UNMEASURED. This is a robustness result (rule 10). The claim that the path phase uses the pin
>    in distribution stays untested at T = 128 (ceiling). Declared beside it: the same contrasts at T = 512, loss AUC and
>    T = 128 held-out NLL (AUC is bimodal across PAPER2X2 seeds, 0.026-0.097, so it is descriptive only).
> 5. *Licensed wording on SUPPORTED:* "freeing delta raises the index model by dI [CI], concentrated in R3+, and leaves
>    the path model at ceiling; the delta x phase interaction is I". This replaces "freeing delta reverses the
>    interaction". At the ceiling I ~ -dI, so the interaction clause is not independent evidence. The score x phase
>    interaction under free delta is not measured. [Optional: add RoPE and Vanilla, 16 runs, as a declared secondary.]
> 6. *Declared secondary S7* (eval-only, CPU): FI checkpoints with delta reset to 0 on the fast half (c < 32) and on the
>    slow half (c >= 32) of the channels, T = 128 accuracy. This separates peak location (fast channels) from the signed
>    content gain that a free delta gives slow channels.
> 7. *Analysis hygiene:* refuse non-finite accuracies before labelling; assert each checkpoint's recipe, variant, seed and
>    300 losses; add the floors file to the md5 guard; selftest cases for items 1 and 7. The smoke is re-run after these
>    edits and before the commit.

## Commands (CPU; scripts and outputs in `docs/audits/2026-10-20/delta_free_check/`)

- `PYTHONPATH=/home/prashr python3 twin_check.py` (T1 twins bitwise in training with interior delta, T2 frozen, T3 real
  twins differ, T4 absorbing dead zone) -> `twin_check_out.txt`: ALL OK.
- Parent-vs-HEAD: `git archive af839b3 -- '*.py'` into the scratchpad, then `python3 -m mapformer.train_variant --variant
  {PoPE-Flat,MapPoPE-Flat} --seed 1674 --epochs 3 --n-batches 4 --batch-size 16 --n-steps 64 ... --device cpu` from both
  trees: losses and state dicts identical.
- `python3 -m mapformer.analyze_delta_free --selftest`: SELFTEST PASSED.
- `e2e_run.sh` (seeds 1670-1677, tiny CPU recipe, train -> eval -> analysis) -> `e2e_ANALYSIS.txt`.
- `pilot_decay_check.py` (pilot checkpoints, read-only) -> `pilot_decay_check_out.txt`.
- `PYTHONPATH=/home/prashr python3 ceiling_flip_sim.py` -> `ceiling_flip_sim_out.txt`.
