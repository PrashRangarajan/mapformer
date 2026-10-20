# Is PoPE's peak pinning what flips its effect between an index and a path phase? (DELTA_FREE) -- pre-registration

Written 2026-10-20, before any batch result. The pilot (seed 1690, outside the batch, never pooled) was launched after this
file's registered decisions were fixed; its numbers are appended in "Pilot" below and change no decision.

## Question
PAPER2X2 (registered, `PAPER2X2_RESULTS.md`, T = 128, r = 2, n = 8) found a score x phase interaction of **+0.154**
(MDE 0.076, 8/8): PoPE's polar score lowers the index model (PoPE 0.679 vs RoPE 0.805) and raises the path-integrated one
(MapPoPE 0.999 vs MapWM 0.971). `docs/paper/expansion/MAPPOPE.md` (THEORY + post hoc ANALYSIS, 2026-10-20) accounts for it
by **peak pinning**: a PoPE score is `sum_c softplus(q_c) softplus(k_c) cos(dphi_c - delta_c)`; content sets only the
amplitudes, and with delta = 0 every head's kernel peaks at zero phase difference whatever the content (Proposition 1). In
the faithful model delta is zero-initialised and clamped to [-2 pi, 0]; on the PAPER2X2 PoPE-Flat runs 98-100% of entries
sit in the clamp's dead zone (>= 0, gradient 0), so delta is effectively 0.
- Path phase: zero phase difference = the same cell = the revisit target. The pin aims the kernel at the target
  (target coherence C 0.995 vs MapWM 0.70, post hoc).
- Index phase: zero phase difference = the query's own token. The one-layer torus solution needs kernel peaks at fixed past
  offsets (retrace targets at token offset 4j - 1); a pinned kernel decreases monotonically over offsets 0..3 for every
  content pair (K(3) > K(1) in 0% of PoPE pairs vs 53-100% for RoPE, post hoc), and 0.110 of PoPE's 0.126 deficit is on
  multi-step reversals (R3+).

The account is a description of stored checkpoints. This batch is its causal test: remove the pin and nothing else.

## Design
Paper torus (64 x 64, 16 observation types, p_empty 0.5), T = 128, rank 2, **PAPER2X2's recipe exactly**
(`run_paper2x2.sh`: `train_variant` 300 epochs x 98 batches x 128, `--schedule cosine` (5% warmup, cosine to 0.1 x lr),
`--lr 1e-3`, AdamW weight decay 0.05, gradient clip 1.0, 1 layer, 2 heads, d 128, `--n-landmarks 0`, serial data, explicit
attention path, fp32). One batch, every arm retrained (rule 12).

| arm | phase | delta | class |
|---|---|---|---|
| `PoPE-Flat` (PI) | index | **pinned**: zero init + clamp [-2 pi, 0] (unchanged faithful PoPE) | `MapFormerWM_RoPEIndex_PoPE` |
| `PoPE-FreeDelta` (FI) | index | **free**: init U(-2 pi, 0), no clamp, learnable | `MapFormerWM_RoPEIndex_PoPE_FreeDelta` |
| `MapPoPE-Flat` (PP) | path, r = 2 | pinned | `MapFormerWM_PoPE` |
| `MapPoPE-FreeDelta` (FP) | path, r = 2 | free | `MapFormerWM_PoPE_FreeDelta` |
| `MapPoPE-FrozenDelta` (ZP) | path, r = 2 | **frozen** at its U(-2 pi, 0) draw, no clamp: no route back to the pin | `MapFormerWM_PoPE_FrozenDelta` |

The 5th arm (ZP) is **included**: it costs 8 runs (+25%) and is the only arm in which the path side's dependence on the pin
can show (FP can learn its way back to delta = 0, and at T = 128 the path arm is at ceiling).

**5 arms x 8 seeds = 40 runs. Seeds 1600-1607** (checked 2026-10-20: no directory under `runs/` matches `*_s16[0-9][0-9]`, no
`run_*.sh` names a seed in 1600-1699, no prereg uses one as a seed). Pilot seed 1690; smoke seeds 1680-1687 (CPU, scratchpad).

**Pairing.** At a seed every arm sees the same training map (GridWorld seed = --seed) and the same training stream (serial
path, global numpy RNG seeded by --seed). Within a phase, the free / frozen arm's every non-delta initial weight is bitwise its
pinned twin's: the new classes build the parent unchanged, re-class the PoPE layer in place (no module constructed, no global
torch RNG consumed) and draw delta from a private generator seeded `torch.initial_seed() + 1_000_000_007`; so dropout's stream
also starts identical. FI, FP and ZP start from the same delta values at a seed. (Checked in `analyze_delta_free --selftest`:
only `pope_delta` differs; the global RNG is untouched; FI and ZP draw the same delta.) The contrasts are therefore paired by
seed; the index and path arms differ in architecture, so the interaction is a difference of two paired differences.

**Evaluation** (PAPER2X2's, unchanged): `eval_noise_refine`, held-out map (env seed 10000), np seed 1234 + s, 100
trajectories at T = 128 / 512 / 1024, accuracy on revisited observation targets, noise 0.

## Code (new flags / classes only; no default changes)
- `model_pope_delta.py` (new): `WMTransformerLayer_PoPE_FreeDelta` = `WMTransformerLayer_PoPE.forward` with the clamp
  removed (one line differs) plus `delta_eff()`; the three model classes above; the initial draw is stored as a buffer
  `pope_delta_init` (so a checkpoint records how far delta moved); the frozen arm sets `requires_grad False` (AdamW skips a
  parameter whose grad is None, so it gets neither gradient nor weight decay; `clip_grad_norm_` ignores it).
- `train_variant.py`: one import and three `VARIANT_MAP` entries (`PoPE-FreeDelta`, `MapPoPE-FreeDelta`,
  `MapPoPE-FrozenDelta`). `model_pope.py`, `train.py`, `environment.py`, `model.py` and `eval_noise_refine.py` are unchanged.
- **Does removing the clamp need any other change?** Checked by grep (2026-10-20): `pope_delta` is read with the clamp in
  `model_pope.py` (the pinned forward, unchanged), `model_pope_ablate.py`, `model_pope_decay.py`, `model_pope_t3.py` (other
  subclasses, untouched by these classes), and in post hoc analysis scripts (`analyze_score_rank.py`,
  `analyze_gain_grain.py`, `kernel_probe.py`, `docs/audits/2026-10-20/causal_whatwhere_T1024.py`). None of those scripts is
  used here; **any future probe of a free / frozen checkpoint must use the effective (raw) delta**: `model_pope_delta.delta_eff`
  does that and `analyze_delta_free.py` uses it everywhere. No init code elsewhere touches `pope_delta` for `train_variant`
  (the U(-2 pi, 0) inits in `train_jsb.py` / `train_indirect.py` are other trainers). `eval_noise_refine` rebuilds each model
  from `VARIANT_MAP` and loads its state dict strictly, so it runs the unclamped forward for the free / frozen arms. No
  rescore hook or `ckpt_guard` path special-cases the PoPE layer class. Unclamped delta enters only through cos / sin.
- **Weight decay acts on the free delta** (the recipe is identical across arms, as in PAPER2X2). With this schedule AdamW's
  decoupled decay alone multiplies a parameter by exp(-sum lr_t x 0.05) = **0.447** over the run, i.e. it pulls raw delta
  toward 0 (not toward 0 mod 2 pi) unless the gradient holds it. This is part of what "free" means here; the frozen arm has
  no decay and no gradient. S4 reads where the free deltas end.

### Bitwise check (rule 19)
`PoPE-Flat` and `MapPoPE-Flat`, seed 1699, the batch's flags with `--epochs 3` (98 batches each), `cuda:1`, run from HEAD
(af839b3; its top-level modules exported to the scratchpad by `git archive`, import path verified to resolve there and its
`VARIANT_MAP` verified to lack the new arms) and from this working tree: **final state dicts bitwise equal (23 / 25 tensors)
and per-epoch loss lists identical** for both arms (PoPE-Flat 2.11256 / 1.87927 / 1.84078; MapPoPE-Flat 1.60706 / 0.12578 /
0.03298). Also checked on CPU: with delta set inside [-2 pi, 0] (or to 0) the free and pinned forwards are bitwise equal, and
they differ once delta > 0 (the clamp is the only difference).

## Registered decisions (`analyze_delta_free.py`)
Per seed s: dI_s = FI_s - PI_s, dP_s = FP_s - PP_s, **I_s = dP_s - dI_s** (T = 128 held-out accuracy). I has PAPER2X2's
orientation (a treatment's effect on the path phase minus its effect on the index phase; the PoPE score's was +0.154), so the
peak-pinning account predicts **I < 0**: the interaction reverses sign relative to the pinned pair's.
Tests: exact sign-flip over the 8 paired seeds (`stats_core.signflip_p`, 256 sign patterns, min p 0.0078), alpha 0.05; t-test
p, house and exact-t MDE and the unpaired permutation p printed beside (rule 5). ub(dI) = upper end of the 90% sign-flip
test-inversion CI (a one-sided 95% bound). SOLVED = `stats_core.classify_run` (final 5% of training loss < 0.05); Fisher on
SOLVED for the path arms (the index arms cannot reach loss 0.05: PAPER2X2's PoPE-Flat ended at 0.73-0.96).

Thresholds, against the noise floor and the retrace floor:
- **TAU_I = 0.05** (index gain / interaction size). PAPER2X2's PoPE-Flat between-seed sd is 0.058 (RoPE 0.012), so a paired
  dI has sd ~0.05-0.08 and exact-t MDE ~0.06-0.09 at n = 8; 0.05 is 40% of the 0.126 PoPE - RoPE gap the account attributes to
  the pin, and 20% of the distance from PoPE-Flat (0.679) to the retrace floor.
- **TAU_P = 0.01** (path loss). MapPoPE-Flat's sd at T = 128 is 0.0014, so 0.01 is ~7 sd: a real loss, not ceiling noise.
- Retrace floor (the one-layer rule "copy the observation 2j steps back while a run reverses the previous one"): 0.934 on
  PAPER2X2's stream; **0.9381 on this batch's own eval stream** (seeds 1600-1607, range 0.928-0.948; blank 0.510, best
  n-gram 0.603; wrap-only share 0.000-0.003), `docs/audits/2026-10-20/delta_free_floors_out.txt`
  (`docs/audits/2026-10-16/sign_floors.py` with `FLOOR_SEEDS`, `FLOOR_T=128`; computed before any run). Both index arms are
  expected below it; a path arm falling below it has lost the map beyond the retrace heuristic.

**Primary (T = 128), labels in this order:**
1. **UNREADABLE** if mean PP < 0.95 (the pinned path arm did not learn: nothing to hurt) or mean PI >= 0.9381 (no index deficit
   to repair). Nothing is read.
2. **NOT SUPPORTED (freeing delta hurts the index arm)** if mean dI < 0 with p(dI) < .05.
3. **NOT SUPPORTED (helps neither)** if ub(dI) < TAU_I: a gain of 0.05 on the index arm is excluded. (The path arm cannot be
   helped detectably at ceiling, see Power, so "neither" turns on the index arm.)
4. **PEAK-PINNING SUPPORTED** if mean dI >= TAU_I with p(dI) < .05 (the registered contrast), AND mean I <= -TAU_I with
   p(I) < .05 (the interaction reverses), AND the path arm is not helped (not: mean dP > 0 with p(dP) < .05, nor FP solving more
   with Fisher p < .05). Qualifier **(path cost)** if mean dP <= -TAU_P with p(dP) < .05 or FP solves fewer with Fisher p < .05;
   else **(path unchanged within noise)**.
5. **NOT SUPPORTED (helps both)** if the index arm is helped as in 4 and the path arm is helped (as defined in 4).
6. **UNMEASURED** otherwise (e.g. an index gain whose CI spans TAU_I without p < .05).

**What each verdict does to the paper** (App. L / MAPPOPE.md proposed paragraph):
- SUPPORTED: the pin is causal for the index-side loss; Proposition 1 and Corollary 1c become a tested mechanism; the
  paragraph may say "freeing delta reverses the interaction". With (path cost) it may also say the pin is what the path phase
  uses; with (path unchanged) the path side stays a description (the frozen arm S1 then carries it, as a secondary).
- NOT SUPPORTED (hurts / neither / both): the peak-pinning account of the index-side loss is withdrawn; MAPPOPE.md's Corollary
  1c and "80% of the interaction is the index side" stay as post hoc description only, and the paper text drops the causal
  wording ("the pin produces ...").
- UNMEASURED: the account stays labelled THEORY + post hoc; the paper says the causal test was run and was inconclusive,
  with dI and its CI.

**Power** (`analyze_delta_free --power`, synthetic: dI ~ N(m, sd) per seed, PP = 0.999, FP = PP + N(0, 0.0015)):

| true dI | sd 0.05: SUPPORTED / UNMEASURED / NOT SUPP. | sd 0.08 |
|---|---|---|
| 0 | 0.001 / 0.21 / 0.79 | 0.02 / 0.52 / 0.47 |
| +0.03 | 0.11 / 0.63 / 0.26 | 0.11 / 0.75 / 0.15 |
| +0.05 | 0.47 / 0.47 / 0.05 | 0.30 / 0.66 / 0.04 |
| +0.08 | 0.93 / 0.07 / 0.004 | 0.64 / 0.36 / 0.01 |
| +0.126 (full recovery to RoPE) | 1.00 / 0 / 0.004 | 0.96 / 0.04 / 0.01 |

So the design detects a recovery of most of the PoPE - RoPE gap; a recovery of ~0.05 or less will most often read
UNMEASURED; with no true effect it reads NOT SUPPORTED 47-79% of the time and SUPPORTED <= 2%.
**Ceiling.** MapPoPE pinned is at 0.999 (sd 0.0014) at T = 128: a path-arm DEcrease is detectable (TAU_P = 0.01 is ~7 sd), an
INcrease is not (at most +0.001). Branch 5 ("helps both") is therefore effectively unreachable, and branch 4's "path not
helped" clause is met almost by construction: the discriminating parts of the test are the index gain, the interaction's
sign, and the (path cost) qualifier. A path-side speed-up would show only in S6 (loss AUC).

## Declared secondaries (no verdict depends on them; post hoc wording for any reading of them)
- **S1 frozen delta** (ZP - PP, T = 128; sign-flip on accuracy, Fisher on SOLVED): **COSTS THE PATH ARM** if mean <= -0.01 with
  p < .05 or ZP solves fewer with Fisher p < .05; HELPS if the mirror; **NO COST** if the 90% CI's lower end > -0.01 and ZP
  solves no fewer; else UNMEASURED. Prediction: costs (C falls below C0). Also ZP - FP (does the route back matter).
  Code check: every ZP checkpoint must have `pope_delta == pope_delta_init` exactly (printed as CODE CHECK FAILED otherwise).
- **S2 strata** (T = 128 eval stream, layer 0 rebuilt by hand with the effective delta, verified against the model's own
  logits; per-target classes from `kernel_probe.retrace_class`, checked equal): accuracy in R1 / R2 / R3+ / non-retrace per
  arm; dI per stratum (sign-flip, Holm over the 4) and its share-weighted contribution to dI. Prediction: the index gain is
  concentrated in R3+.
- **S3 out of distribution**: dI, dP and I at T = 512 and 1024 (trained at 128: robustness, rule 10; Holm over the 6).
  Prediction (MAPPOPE.md): free delta costs the index arm at T = 1024 (the pinned recency kernel is length-agnostic).
- **S4 learned delta**: dead-zone share (raw >= 0) for the pinned arms; for free / frozen: re-pinned share
  (|wrap(delta)| < 0.1 rad), mean |wrap(delta)|, mean |delta - init|, share outside [-2 pi, 0].
- **S5 kernel**: index arms, share of content pairs (head, action query, observation key) with K(3) > K(1) and the pooled argmax
  offset over D in 1..16 (prediction: free > 0%, a comb near 4j - 1); attention on the newest observation; path arms, target
  coherence C and C0 on the most recent same-cell observation key (prediction: ZP's C < C0).
- **S6 training**: loss AUC (mean per-epoch loss), final-5% loss, run classes, SOLVED per arm; AUC contrasts FI - PI, FP - PP,
  ZP - PP; r(final loss, acc) per length (rule 2).
- Descriptive: runs below the retrace floor (path arms) / above it (index arms); I beside PAPER2X2's +0.154 (cross-batch: a
  comparison of sizes only; I <= -0.154 would mean freeing delta undoes at least the whole pinned interaction). No RoPE / MapWM
  arms are in this batch, so the score x phase interaction under free delta is NOT measured; adding them (16 runs) is the
  amendment that would.

## What this can and cannot show
One task (paper torus), one length (T = 128) in distribution, one layer, rank 2, one recipe and budget, n = 8. It tests
whether the pin causes the index-side deficit and whether the path side depends on the pin; it does not test PoPE's other
ingredients (softplus magnitudes, the doubled frequency count), multi-layer models (where content can carry position), or
language. "Free" includes AdamW's pull toward raw 0 (see Code). The eval-mode accuracy under-reports one-layer runs below
ceiling (attention-dropout scale, `docs/audits/2026-10-04/DROPOUT_RESCORE.md`); both index arms are read the same way; no
re-score is declared.

## Process, cost and launch
- `analyze_delta_free.py` (new): registered analysis; `--selftest` reaches every primary label (UNREADABLE x 2, hurts, neither,
  SUPPORTED with and without path cost, by accuracy and by SOLVED, helps both, UNMEASURED) and every S1 label from
  deterministic synthetic seeds, and checks on tiny models: pairing (only delta differs), global RNG untouched, same delta draw,
  frozen requires_grad, the hand rebuild equals the model's logits for all five arms (0.0e+00), `delta_eff` = clamp for pinned
  and raw for free, and `retrace_class` / coherence / K(3) > K(1) equal `kernel_probe.py`'s on pinned models. `--power` prints
  the table above. The verdict JSON is written before the secondaries.
- `run_delta_free.sh` (new; sources `lib_driver.sh`): `flock` lock (one lock for every mode); MODE=batch refuses a dirty tree
  and uncommitted `analyze_delta_free.py` / `run_delta_free.sh` / `model_pope_delta.py` / this file BEFORE any wait; refuses to
  start without `DRV_GPUS`; md5 guard over every `mapformer` module the trainer, evaluator and analysis import (92 files, from
  `sys.modules`; `model_pope_delta.py` asserted in it) plus `lib_driver.sh` and the driver, re-checked before the eval and the
  analysis; seed outer, arm inner; `DRV_MAXPG=4` trainers per GPU (every `mapformer.train_*` of the user counts); skips finished
  and running runs; artifacts-then-marker (`.delta_free_done` after `DELTA_FREE_ANALYSIS.txt`, `DELTA_FREE_VERDICTS.json` and
  `_DELTA_FREE_RAW.json`; all 40 checkpoints required before the eval). MODE=pilot (seed 1690, `runs/delta_free_pilot`, eval
  only) and MODE=smoke (CPU, tiny recipe, seeds 1680-1687, eval with 3 trials, analysis with the probe on the same 3 walks).
- **Smoke** (2026-10-20, CPU, `<scratchpad>/delta_free_smoke`): all 40 tiny runs, eval, analysis and probe ran end to end;
  probe accuracy equals eval accuracy exactly on every run, rebuild max |dlogit| 0.0e+00, frozen deltas unmoved (0.000), free
  deltas moved; the primary read UNREADABLE (untrained path arm), as it should.
- Launch rules: (1) an independent code-verification agent, blind to results, before launch (rule 29); its findings go into an
  amendment; (2) commit the new files and the `train_variant.py` change (the batch refuses otherwise); (3)
  `cd /home/prashr/mapformer && DRV_GPUS="1" setsid nohup bash run_delta_free.sh > /dev/null 2>&1 &` (one shared GPU, <= 4
  runs, per the user's approval; `DRV_GPUS="0 1"` halves the wall time if both GPUs are allowed).
- Cost: see "Pilot" (measured while sharing).

## Pilot (outside seeds, never pooled)
Seed 1690, all five arms, full recipe (300 epochs), the driver's exact command line (`MODE=pilot DRV_GPUS="1"`), on
`cuda:1` shared with another user's job (~7-8 GB, 100% utilisation throughout), <= 4 of our runs at once; code = HEAD af839b3
+ this working tree (md5 guard in `runs/delta_free_pilot/code_md5.txt`). Eval as registered (100 walks, own stream; retrace
floor of seed 1690's stream 0.9455). Descriptive only (n = 1); every number changes no registered decision.

| arm | T=128 | T=512 | T=1024 | class (final-5% loss) | loss AUC | R1 / R2 / R3+ / none (T=128) |
|---|---|---|---|---|---|---|
| PoPE-Flat | 0.671 | 0.623 | 0.599 | STALLED (0.943) | 1.099 | 0.949 / 0.826 / 0.549 / 0.533 |
| PoPE-FreeDelta | **0.816** | 0.525 | 0.401 | STALLED (0.826) | 0.950 | 0.993 / 0.947 / **0.774** / 0.535 |
| MapPoPE-Flat | 1.000 | 0.985 | 0.939 | SOLVED (0.0001) | 0.031 | 1 / 1 / 1 / 1 |
| MapPoPE-FreeDelta | 1.000 | 0.969 | 0.926 | SOLVED (0.0002) | 0.036 | 1 / 1 / 1 / 1 |
| MapPoPE-FrozenDelta | 1.000 | 0.937 | 0.801 | SOLVED (0.0118) | 0.094 | 1 / 1 / 1 / 1 |

Probe (eval-only, the analysis's own functions; rebuild max |dlogit| 0.0e+00 and probe accuracy = eval accuracy on all five):
- delta: PoPE-Flat dead-zone share 1.000 (MapPoPE-Flat 0.891, re-pinned 0.930) -- the pin reproduces; free arms moved far
  from their draw (mean |delta - init| 1.74 index, 1.82 path) and did NOT return to the pin (re-pinned share 0.016 / 0.109);
  frozen moved 0.000 (code check passes).
- index kernel: K(3) > K(1) in **0% of PoPE-Flat content pairs vs 100% of PoPE-FreeDelta's**; argmax offsets 1 (all 136) vs
  3 / 8 / 11 / 16; attention on the newest observation 0.354 vs 0.001.
- path: target coherence C / C0 = 0.999 / 0.999 pinned, 0.675 / 0.998 free, 0.665 / 0.990 frozen (C falls below C0 once the
  pin is removed, as predicted), yet all three are at 1.000 accuracy.

Reading for the batch (not a decision): the index side moved in the predicted direction and size (+0.145, R3+ +0.225,
OOD cost at T = 1024 -0.198); the path side is at ceiling in all three path arms at T = 128, so (as stated under Power) the
(path cost) qualifier and S1 will most likely read "unchanged" / "NO COST" on accuracy, and the path side's dependence on
the pin, if any, will show only in S6 (frozen: loss AUC 0.094 vs 0.031, final loss 0.012 vs 0.0001) and in S3 (T = 1024
0.801 vs 0.939). If the user wants a registered path-side readout, the place to add it is an amendment before launch
(e.g. a label on ZP - PP loss AUC, or on T = 512 accuracy); it was not added here because the pilot motivated it.

**Timing while sharing** (`cuda:1`, the other user's job running): 3.9 s/epoch (index arms) to 4.2-4.4 s/epoch (path arms)
at 4 of our runs at once; 1.7 s/epoch for one run alone. Wall per run 19.2-21.3 min at 4 concurrent (9.5 min alone). Our
4 runs used ~5.1 GB together. Eval of 5 checkpoints x 3 lengths < 1.5 min.
**Projected batch:** 40 runs / 4 slots = 10 waves x ~20.5 min = **~3.4 h** on one shared GPU, + eval (~8 min for 40
checkpoints) + CPU analysis with probe (~5 min): **~3.6 h wall** (~1.8 h if both GPUs are allowed, `DRV_GPUS="0 1"`, 4 per
GPU). If the other user's job ends, expect it shorter (one run alone: 1.7 s/epoch).
