# Gain granularity between MapEM and MapPoPE -- pre-registration (2026-10-05, before any run of the batch)

Written, with the analysis (`analyze_gain_grain.py`), its smoke test and the power table, BEFORE the pilot was launched;
the pilot and cost sections were appended after it (see "Pilot" for what was read).

## Question
MapEM and MapPoPE are two ends of one family. With the same path machinery in every arm (MapWM's rank-2
`ActionToLieAlgebra` + `PathIntegrator`, 32 angles per head, omega 2pi .. 2pi/64), every score here is

    score_ts = sum_c G_c(x_t, x_s) * A_c * cos(dtheta_c + delta_c),      dtheta_c = theta_t,c - theta_s,c

| arm | G_c (content) | A_c | delta_c |
|---|---|---|---|
| MapEM (`VanillaEM`) | ONE SIGNED scalar for all c: A_X = q_t . k_s / sqrt(d_h) | \|q0_c\|\|k0_c\| / sqrt(d_h), learned | arg q0_c - arg k0_c, learned |
| MapPoPE-Pair | softplus(q_t,e) softplus(k_s,e) >= 0 per element, two elements per angle | 1 (absorbed by the gains' biases) | per element, learned in [-2pi, 0] (sits at 0) |
| **scalar gain** (`GainScalar`, new) | mu^q_t mu^k_s >= 0, ONE scalar per token per head (softplus of a scalar projection) | softplus(a_c) >= 0, learned | 0 |
| **per-module gain** (`GainMod4`, new) | mu^q_t,m mu^k_s,m >= 0, one per token per head per module; M = 4 modules of 8 contiguous channels in omega order (fine -> coarse scale bands) | softplus(a_c) >= 0, learned | 0 |
| **MapEM, non-negative** (`VanillaEM_NonNeg`, new) | softplus(A_X) >= 0, one scalar for all c | as MapEM | as MapEM |

The scalar gain is a pure gain field: content scales ONE fixed kernel whose maximum is at dtheta = 0 (pure rate
remapping). The per-module gain lets content choose which band of spatial scales answers. MapPoPE-Pair lets content
re-weight every frequency. **(a) Does MapPoPE's per-frequency freedom matter, or does a scalar (or per-module)
non-negative gain do as well? (b) Does MapEM's SIGNED scalar gain cost relative to a non-negative one?**

Why now. On this task at rank 2, PoPE's score carries MapPoPE's whole gain over MapWM (`MAPPOPE_PAIR_RESULTS.md`, REG:
+0.024, 16/16 vs 10/16 SOLVED). The post hoc mechanism (`docs/theory/2026-10-05/neuro_design.md` 2.3) is "select
scales, never shift them": PoPE's delta sits at its bound 0 on 89-100% of channels (forcing it to 0 changes accuracy by
0.000 on 31/32 rank-2 seeds), so its remaining lever is non-negative gain over channels. The remap probe reads
MapPoPE-Pair r2 as close to pure gain (gain share 0.75, width cv 0.08). The design note's E2 / E7 ask exactly this:
scalar gain = per-channel -> adopt pure gain; scalar < per-channel -> width change is needed. Existence is not the
question: on converged MapPoPE r2 checkpoints, replacing the score with a shared kernel x content gain keeps 0.988
(`docs/WHAT_WHERE_CHECKS.md` 1, post hoc). Whether training FINDS a solution in the coarser classes is.

## Arms (one batch, all retrained, rule 12; seeds 26-45, n = 20 per arm)
| label | variant | class | params | initial weights shared at a seed (checked) |
|---|---|---|---|---|
| W | `Vanilla` | `model.MapFormerWM` | 204,373 | embeddings, bottleneck, omega, readout with P |
| P | `MapPoPE-Pair` | `model_pope_pair.MapFormerWM_PoPEPair` (unchanged) | 204,501 | -- |
| S | `GainScalar` | `model_em_pope.MapFormerWM_GainScalar` (new) | 171,929 | everything with P except the score's content projections |
| M | `GainMod4` | `model_em_pope.MapFormerWM_GainMod4` (new) | 173,477 | everything with P except the score's content projections |
| E | `VanillaEM` | `model.MapFormerEM` (MapEM-os: separate q0_pos / k0_pos) | 204,629 | -- |
| N | `VanillaEM_NonNeg` | `model_em_pope.MapFormerEM_NonNeg` (new) | 204,629 | ALL of E's (bit for bit; no new parameter) |

Every arm: 1 layer, 2 heads, d 128, per-head d_h 64, **rank 2 (asserted), 32 angles per head, identical initial omega**
(`gain_grain_equiv_out.txt` 5). MapEM needed no change: `VanillaEM` already has n_blocks = d_h / 2 = 32 per head and
r = 2 by default. Entry points: `train_gain_grain.py` (= `train_variant.main()` with `model_em_pope.register` adding
MapPoPE-Pair, GainScalar, GainMod4, VanillaEM_NonNeg to VARIANT_MAP; `train_variant.py` and every existing model file are
not edited) and `eval_gain_grain.py` (runs `eval_noise_refine` unchanged via runpy, same registration).

### Design choices, each justified
- **delta = 0 in the new gain arms.** PoPE's delta is measured to sit at 0 and to be removable at no cost on this task
  (above). Fixing it makes the gain arms' kernels peak at dtheta = 0 by construction (DER: each term <= its amplitude,
  equality iff dtheta_c = 0 mod 2pi; confirmed on untrained models, `gain_grain_remap_untrained_out.txt`, shift0 1.000).
  So S / M vs P differ in granularity AND in P's (unused) delta freedom; the latter is not separated here.
- **A_c = softplus(a_c) >= 0 learned per head per channel, init 1; score scale 2 / sqrt(d_h).** With a scalar or
  module gain, a learned non-negative spectrum is what lets the model choose which scales form the kernel (MapPoPE-Pair
  does this through its per-element biases). The factor 2 makes the M = 32, A_c = 1 limit equal MapPoPE-Pair (each
  angle drives two elements there) and matches P's expected initial score scale (64 x E[softplus]^2 / 8 ~ 4.5 at the
  same phase vs 32 x 2 x ~0.56 / 8 ~ 4.5).
- **Gains = softplus(Linear(LN x))**, PoPE's own magnitude map, H x M outputs. Consequence: S and M have ~31-33k fewer
  parameters than P (the score's content projection shrinks from 64 to 1 or 4 outputs per head). The per-frequency
  freedom IS those parameters; a WORSE verdict cannot separate "granularity" from "content-projection width", an AS
  GOOD verdict says the width is not needed.
- **Modules: M = 4 contiguous blocks of 8 channels in omega order**; at init the bands span periods (in units of the
  learned unit step) 1.0-2.6, 2.9-7.5, 8.6-21.9, 25-64. omega stays learnable in every arm (as in all arms of the
  project), so band membership is by channel index, not frequency. These are scale bands, not single-frequency grid
  modules (neuro_design 3.1: "M = 4-6 modules"); 4 divides 32 and gives bands of a factor ~2.6 in scale.
- **The sign arm N (included).** S vs E would change three things at once (sign, factorised vs bilinear content
  form, learned offsets). N changes only the sign: softplus applied to A_X, identical parameters and initial weights
  (bit for bit). Caveat stated now: a non-negative map of A_X necessarily has a positive mean (softplus(0) = 0.69 at
  init vs A_X ~ 0), so N vs E cannot separate "non-negative" from "positive-mean gain". The bundle S - E and the content
  form S - N are secondaries.
- **Seeds 26-45, fresh for every arm.** MAPPOPE_PAIR used 10-25 for W and P and those results were read; through this
  wrapper W and P at 10-25 would be bitwise reruns (its pilot reproduced paper2x2 bit for bit), i.e. determinism, not
  replication (rule 6), costing 32 runs for no information. Fresh seeds also give a second, independent fresh-seed
  estimate of MAPPOPE_PAIR's SCORE contrast (P - W, secondary and headroom control). The wrapper is checked against a
  stored run instead (pilot, below). Seeds 26-45 are fresh relative to this question.
- **n = 20** from the power table below (not 16): at n = 16 a coarser gain that behaves like MapWM is called WORSE with
  0.81, at n = 20 with 0.94; a half-sized deficit is wrongly called AS GOOD with 0.14 at n = 16 and 0.05 at n = 20; the
  headroom control passes with 0.81 vs 0.90. Cost: +24 runs (~1 h, below).

## Recipe (= `run_mappope_pair.sh` exactly; diffed)
Torus 64x64, 16 observation types, p_empty 0.5, no landmarks; 300 epochs x 98 batches, batch 128, T=128, 1 layer,
2 heads, d 128, AdamW lr 1e-3 wd 0.05, cosine, `--data-workers` 0, explicit attention path. Driver `run_gain_grain.sh`
(lib_driver: flock, least-loaded slot picker at 4 jobs/GPU, setsid/nohup, md5 guard over every imported mapformer
module (92 files) + both wrappers + analysis + stats_core + the remap secondary and its helpers, re-checked before eval
and before analysis; done marker only after the artifacts). Dry run (scratch copy, DRV_DRYRUN): 120 launches, 6 arms x
seeds 26-45, flags identical to MAPPOPE_PAIR's.

## Evaluation
`eval_noise_refine` via `eval_gain_grain`: noise 0, held-out map (env seed 10000), 100 walks per run (np seed 1234 + s),
eval mode, T = 128 (REGISTERED, matched length), 512 and 1024 (extrapolation, no verdict). Floors (paper torus,
`docs/WHAT_WHERE_CHECKS.md`): best n-gram 0.598, always-blank 0.507. SOLVED = `stats_core.classify_run` (final-5%
training loss < 0.05).

## Registered primaries (T = 128, n = 20 per arm), `analyze_gain_grain.py`
Shared rules: accuracy fires if two-sided permutation p < .05 (`perm2_p`, 200k Monte Carlo relabellings) AND |d| >= 0.01;
SOLVED fires if two-sided Fisher p < .05. **Both readouts count** in this batch (unlike MAPPOPE_PAIR): the non-inferiority
claim is about both, so a cost on either must count against it.

**(a) GRANULARITY -- non-inferiority against P**, separately for S and M (`granularity_state`), evaluated in order:
1. **CONFLICT** -- accuracy and SOLVED fire in opposite directions.
2. **WORSE** -- accuracy or SOLVED fires negative (which one is printed).
3. **BETTER** -- accuracy or SOLVED fires positive.
4. **AS GOOD** -- non-inferior on accuracy at margin 0.01 (one-sided permutation shift test, H0: d <= -0.01, p < .05;
   with equal n the permutation distribution is symmetric, so one-sided p = two-sided p / 2 on the favourable side)
   AND SOLVED(x) >= SOLVED(P) - 1. If both arms are >= 0.999 on every seed it is labelled "(at ceiling)".
5. **UNDETERMINED** -- none of the above (why is printed).
Margin 0.01: the house materiality floor, ~40% of MapPoPE-Pair's registered gain over MapWM (+0.024). The 90%
permutation CI of d is printed (its lower end is the one-sided 95% bound).

Headline (`granularity_verdict`, exhaustive over the 5 x 5 label pairs, all printed in `gain_grain_smoke_out.txt`):
| S vs P | M vs P | REGISTERED (a) |
|---|---|---|
| AS GOOD / BETTER | AS GOOD / BETTER or UNDETERMINED | **SCALAR GAIN SUFFICES** (per-frequency freedom not needed on this task) |
| AS GOOD / BETTER | WORSE | **NON-MONOTONE** (anomalous; the module arm's cost is not granularity) |
| WORSE | AS GOOD / BETTER | **MODULE GAIN SUFFICES, SCALAR DOES NOT** |
| WORSE | WORSE | **PER-FREQUENCY GAIN NEEDED** |
| WORSE | UNDETERMINED | SCALAR COSTS; per-module undetermined |
| UNDETERMINED | AS GOOD / BETTER | MODULE GAIN SUFFICES; scalar undetermined |
| UNDETERMINED | WORSE | MODULE COSTS; scalar undetermined |
| UNDETERMINED | UNDETERMINED | UNDETERMINED |
| CONFLICT in either | | CONFLICT (reported as it falls) |

**Headroom qualifier.** P - W (MAPPOPE_PAIR's SCORE contrast on fresh seeds) is the task-difficulty control: if neither
its accuracy nor its SOLVED readout fires positive, verdict (a) carries "NO HEADROOM: MapWM r2 is not detectably below
MapPoPE-Pair on these seeds, so an AS GOOD verdict does not show that a coarser gain avoids a cost this task can reveal".
A WORSE verdict stays informative either way.

**(b) SIGN -- N vs E, two-sided** (`sign_state`), in order: **CEILING** (both >= 0.999 on every seed: undetermined);
**CONFLICT**; **NON-NEGATIVE BETTER (MapEM's signed gain costs)** (accuracy or SOLVED fires positive; which is
printed); **NON-NEGATIVE WORSE (the sign freedom helps)**; **NO DIFFERENCE DETECTED** (with the MDE and the 95% CI).

Multiplicity: three registered decisions, each at its own .05 (they answer different questions; no family-wise claim).
Each has two readouts; a one-readout firing is labelled with its readout. Under the null of equality the resampled
false-firing rate is 0.00 (granularity, the reference sits near ceiling) and 0.05-0.06 (sign; power table).

Every branch of every function, and `main()` end to end on 120 synthetic checkpoints, is exercised by
`docs/audits/2026-10-05/gain_grain_smoke.py` / `_out.txt` (PASS).

## Power (`docs/audits/2026-10-05/gain_grain_power.py` / `_out.txt`; computed, and n fixed, before the pilot)
Resampling stored per-seed (accuracy, SOLVED) pairs through the registered functions: P-like = MAPPOPE_PAIR's
MapPoPE-Pair r2 (0.9995 +/- 0.0010, 16/16), W-like = its MapWM r2 (0.9752 +/- 0.0393, 10/16), E-like = em_fig4's MapEM r2
at this recipe (0.9311 +/- 0.1258, 5/8; evaluated for this purpose, `gain_grain_emfig4_T128.json`; trained with
`--data-workers 3`, a different data stream).
| n | x like P (null): AS GOOD / WORSE | x like W: WORSE / AS GOOD | x = 50/50 W,P: WORSE / AS GOOD / UNDET | x like E: WORSE | N like P: BETTER | N = 50/50 E,P: BETTER | N like E (null): any firing | headroom passes |
|---|---|---|---|---|---|---|---|---|
| 12 | 1.00 / 0.00 | 0.59 / 0.01 | 0.15 / 0.17 / 0.68 | 0.99 | 0.98 | 0.20 | 0.05 | 0.59 |
| 16 | 1.00 / 0.00 | 0.81 / 0.00 | 0.27 / 0.14 / 0.60 | 0.99 | 0.99 | 0.24 | 0.06 | 0.81 |
| **20** | **1.00 / 0.00** | **0.94 / 0.00** | **0.37 / 0.05 / 0.58** | **1.00** | **1.00** | **0.25** | **0.06** | **0.90** |
Reading: a coarser gain as bad as MapWM is caught (0.94) and never passes as AS GOOD; one as bad as MapEM is always
caught; a half-sized deficit is mostly UNDETERMINED (0.58) and passes as AS GOOD with 0.05. A non-negative MapEM that
behaves like MapPoPE-Pair is caught (1.00); half of that, 0.25 (unmeasured, and so labelled).

## Declared secondaries (no verdict)
- P - W (fresh-seed replication of MAPPOPE_PAIR's SCORE); the bundle S - E; content form S - N; module vs scalar M - S;
  E - W; N - P. Accuracy (perm) and SOLVED (Fisher) each.
- T = 512 / 1024 (rule 10: robustness, not capability) for S - P, M - P, N - E, P - W.
- T = 128 revisit NLL for the three primaries (continuous, less ceiling-bound).
- r(final-5% loss, acc@128) over 120 runs and within arm; run classes; epoch at which the 10-epoch running loss first
  falls below 0.05 (speed, descriptive).
- Basins (theory T1): `analyze_score_rank`'s per-head (kappa, indep) classification with its thresholds unchanged (set
  at T = 1024; T = 128 has few wraps, so this is descriptive), "SOLVED iff CLEAN" per arm. Declared channel weights:
  W and P as SCORE_RANK; E and N: |q0_c||k0_c| (content is one scalar for all channels); S and M: mean action-query gain
  x mean observation-key gain of channel c's module x A_c.
- Remap probe (`docs/audits/2026-10-05/gain_grain_remap.py`, run by the driver after the registered artifacts):
  gain / even / odd shares, peak at d = 0, width cv, r(height, width) per arm; the learned spectrum of S and M (share of
  sum A_c per 8-channel band, fine -> coarse) and the spread of their observation-key gains; MapEM's share of (action,
  observation) pairs with A_X < 0. **By construction** (as for MapEM in `remap_probe_out.txt`): S, E and N are exactly
  rank 1 in content x position (gain share 1.000, width cv 0), and S, M peak at d = 0; only M's and P's shares are
  learned quantities. Pipeline verified on untrained models: every rebuilt forward reproduces the logits (0.0).
- Not run: the dropout-scale re-score (`rescore_hook` does not know the new layers; on this task paper-torus path arms
  move <= 0.0007 under it, `DROPOUT_RESCORE.md`).

## Construction and equivalence checks (`docs/audits/2026-10-05/gain_grain_equiv.py` / `_out.txt`, all PASS; every positive control differs)
1. Nesting, on 4 held-out walks (255 tokens), eval mode: **M = 32 at A_c = 1 equals MapPoPE-Pair with each angle's two
   element gains tied and delta = 0** (max |dlogit| 6.6e-07, float32 summation order); GainMod4 equals M = 32 with gain
   rows tied within contiguous modules (6.0e-07); GainScalar equals GainMod4 with all module rows equal (8.3e-07).
   Positive controls: untied Pair 7.4e-02, Pair with delta -0.5 6.0e-02, A_c = 1.5 1.8e-01, interleaved modules 2.7e-01,
   one module bias moved 1.7e-01.
2. **GainScalar with gains fixed at 1 equals MapEM with A_X fixed at 1 and q0_c = k0_c = (sqrt(2 A_c), 0)**: a pure
   position kernel, MapEM-PosOnly-like with zero offsets (7.2e-07); a non-zero q0 phase offset differs (3.0e-02).
3. VanillaEM_NonNeg with its content map set to identity equals VanillaEM exactly (0.0); its initial state_dict equals
   VanillaEM's bit for bit at seeds 26, 45, 100.
4. Matched initialisation at every batch seed (26-45) and pilot seed (100, 101): S and M share all 20 non-score tensors
   with P; W and P share embeddings, bottleneck, omega and readout.
5. All six arms causal (leak 0.0), 32 angles per head, initial omega 6.2832 .. 0.0982 identical, rank 2.

## Void
Any of 120 checkpoints missing or not 300 epochs; md5 guard trips (at launch, before eval, before analysis); the pilot
reproduction check failing.

## What it can and cannot show
- Can: on the paper torus at T = 128, rank 2, 1 layer, whether training finds the map as reliably with one (or four)
  non-negative content gains per token as with MapPoPE-Pair's 64; whether making MapEM's content factor non-negative
  changes how reliably MapEM finds it.
- Cannot: whether per-frequency gains matter on longer walks (SCORE_RANK: wrap-only revisits at T = 1024 are a code
  problem no score fixes), on other tasks, or with depth; separate granularity from content-projection width (S/M) or
  non-negativity from a positive mean gain (N); say anything about MapPoPE's unused delta beyond the prior post hoc
  lesion. The remap-probe readouts of S, E and N are architectural, not findings.

## Pilot (`docs/audits/2026-10-05/gain_grain_pilot.sh`; readout `gain_grain_pilot_check.py` / `_out.txt`; seeds 100-101 and 10)
Launched AFTER the prereg, analysis, branches, power table and n were committed (d332718); nothing in them was changed
after it. Full recipe, 8 concurrent jobs (4 per GPU), the batch's flags.
(a) **Reproduction** (pass = bitwise-equal per-epoch losses): `MapPoPE-Pair` s10 through `train_gain_grain` (which also
imports `model_em_pope`) against the stored `runs/mappope_pair/p0/MapPoPE-Pair_s10` -- **PASS: 300/300 epochs bitwise
equal, final weights equal (max diff 0.0)**. The wrapper changes nothing on the existing path.
(b) **Timing** at 8 concurrent: GainScalar / GainMod4 2.8-3.1 s/epoch, VanillaEM 3.1, VanillaEM_NonNeg 3.5,
MapPoPE-Pair 3.5 (MAPPOPE_PAIR's logs: 3.2 for its arms at the same load). Wall time of the 8-job wave: 16.4 min.
(c) **Outcome, READ** (seeds outside the batch; disclosed): GainScalar s100/s101, GainMod4 s100/s101,
VanillaEM_NonNeg s100/s101 and VanillaEM s100 all SOLVED (final-5% loss 0.0002-0.0005), T=128 held-out accuracy 1.000
on every run. The per-epoch training-loss prints were also seen while it ran (every new arm below 0.03 by epoch 45).
Training is sane; the eval path (eval_gain_grain -> eval_noise_refine, ckpt_guard layout) works. It suggests AS GOOD
for (a) and, if MapEM solves as often here as s100 did (em_fig4: 5/8 at another data stream), little headroom for (b),
whose CEILING / NO DIFFERENCE branches exist for that case. n = 1-2 per arm: no inference.

## Cost (measured)
120 runs at 8 concurrent = 15 waves x ~16.5 min (pilot wave 16.4 min, mixed arms) -> **~4.1 h of training**, + launch
spacing (8 s per job, overlapping) and queue tail ~0.2 h; eval (120 runs x T = 128 / 512 / 1024 x 100 walks) ~10 min;
analysis (permutation tests, 240 basin classifications) and the remap secondary ~15 min. **ETA ~4.5 h from launch.**
n = 16 (96 runs) would be ~3.6 h.

## Launch (not done; rule 29: independent code audit first, findings as Amendment 1 before launch)
    cd /home/prashr/mapformer && setsid nohup bash run_gain_grain.sh > /dev/null 2>&1 &

## Amendment 1 (2026-10-05, after an independent code audit, BEFORE the batch was launched)
The audit (read-only, blind to pilot outcomes) found no bug. Re-verified: the score classes (incl. dropout placement in
train mode: tied GainMod32 vs MapPoPE-Pair max |dlogit| 7.2e-07); the equivalence chain (byte-identical re-run; every
positive control fails); NonNeg = VanillaEM at identical init with content map = identity (0.0); all six arms 32
angles, rank 2, identical omega, causal; recipe and eval equal to MAPPOPE_PAIR (Pair s10 reproduced bitwise, 300/300
epochs and final tensors); seeds 26-45 fresh; md5 GUARD covers every imported module; analysis directions; power.
Changes, all before launch:
1. **D1 -- AS GOOD calibration.** Outcomes are bimodal, so the shift-permutation non-inferiority test is miscalibrated:
   with SOLVED slack 1 the realised false-AS-GOOD rate at the 0.01 margin was ~0.10 (audit simulation). SOLVED_SLACK is
   now 0 (AS GOOD needs SOLVED(x) >= SOLVED(P)): realised size at the margin ~0.03; an arm identical to P still passes
   (1.00); a half-margin deficit passes 0.22. Cost: a coarse arm with a true 5% / 10% failure rate passes AS GOOD less
   often (it was 0.59 / 0.30 with slack 1). Read AS GOOD as "no failure the reference did not also have".
2. **D2 -- verdict (b) relabelled** "NON-NEGATIVE / POSITIVE-MEAN BETTER": softplus(A_X) is ~0.69 for every pair at init,
   so q0/k0 get a coherent position gradient from step one; non-negativity and a positive mean gain are not separated.
3. **D3 -- WORSE headlines carry their confound** (fewer content-projection parameters, ~32k; no delta; learned A_c; tied
   pairs). AS GOOD is unaffected. The separating follow-up, if needed, is a trained GainMod32.
4. **D4 -- dropout-scale re-score added as a declared secondary** after the marker
   (`docs/audits/2026-10-05/gain_grain_rescore.py`): rescore_hook with the two new layer classes registered (both feed
   o_proj with attention @ V; dry check: all six arms hooked, none skipped). No verdict.
5. N1: sign_state checks SOLVED before declaring CEILING. N2: "(at ceiling)" is carried into the (a) headline. N3: the
   driver logs `git rev-parse HEAD` and whether the tree is clean at launch.
The analysis smoke test (`gain_grain_smoke.py`) now reports one FAIL, the expected-label string for "N solid, E
bimodal", which was written before the relabel in item 2; that case still routes to the intended branch.
