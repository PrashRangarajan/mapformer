# The sign test in a state-space model at the second rate (G-3 block A2, lr 1e-3) -- results (registered, `G3_SSM_PREREG.md` + Amendments 1-3)

Batch 2026-10-24 23:32 -> 2026-10-25 02:37 (3 h 05 min) on kalman GPU 0 (up to 4 runs at a time), configuration
`a2_after_minimal_h2` (A2-only mode on the `minimal_h2` root of `G3_SSM_RESULTS.md`), 40 runs, code `ab872fb` (clean; A2
provenance: 104 guarded files equal to block A's record, 3 exempt, 0 mismatches), torch 2.10.0+cu128. Run dirs
`runs/ssm_sign/p0/A2/<arm>_moves_m32_lr1e-3_s<seed>`. Registered outputs `G3_SSM_A2_ANALYSIS.txt`, `G3_SSM_A2_VERDICTS.json`
(marker `.ssm_sign_a2_done`); re-run here read-only from the run dirs into a scratch path: text identical, JSON equal.

**Design (prereg Sec. 5 row A2, Amendment 2).** Block A's model, task and recipe at lr 1e-3 instead of 3e-3: one-layer
diagonal complex SSM with an input-dependent rotation and no decay (= unnormalised rotary linear attention), ring of 32
cells, moves +1/-1, only the moves step, T = 128, 300 epochs, cosine with 5% warm-up, AdamW wd 0.05, rank 4, 1 layer, 2
heads, d 128. Arms signed, |.| (abs), softplus, exp, index (every token steps 1: the integration gate). Seeds 2100-2107,
block A's labels (paired by design, tested unpaired). **Not results-blind:** A2 was launched because block A's exp cell was
8/8 RUNAWAY at 3e-3 (Amendment 2); its design and analysis were fixed before it ran, and the independent check
(Amendment 3, `docs/audits/2026-10-25/g3_a2_audit.md`: no BLOCKER, 1 MAJOR, 3 MINOR) set the Holm family (3 cells {abs,
softplus, exp}, abs at raw p) and made "unless A2 agrees" operational before launch. **Floors** (gate, registered stream):
blank 0.496, best n-gram 0.554, retrace 0.864 at T = 128.

## Reading (registered)

**SIGN COST IN THE SSM at lr 1e-3 for |.| and softplus; NO SIGN COST for exp.** READING string: "SIGN COST IN THE SSM; lr
1e-3 (m = 32): SIGN COST IN THE SSM". At lr 1e-3 the gates pass (signed 7/8 >= 6; index 0/8, signed - index +0.285), and:

- **|.| vs signed: SIGNED LEARNS BETTER** -- 0/8 vs 7/8 SOLVED (Fisher p 0.0014), accuracy 0.756 vs 0.994 (permutation
  p 0.0002, d -0.238). The primary cost of block A replicates at the second rate.
- **softplus vs signed: SIGNED LEARNS BETTER** -- 0/8 vs 7/8 (Fisher 0.0014, Holm over 3 cells 0.0042), accuracy 0.938
  (permutation 0.0003, Holm 0.0006; d -0.056). Per Amendment 3.2 this licenses "softplus learns worse than signed at
  lr 1e-3". All 8 softplus runs are STALLED (none descending), so unlike block A this cell is not budget-scoped by
  descending runs.
- **exp vs signed: NO SIGN COST** -- 6/8 vs 7/8 SOLVED (Fisher 1.0, Holm 1.0), accuracy 0.985 vs 0.994 (permutation 0.72,
  d -0.009); no RUNAWAY (0/8; largest final move step 7.8).

**What the registered text licenses about exp (Amendment 3.2, which supersedes Amendment 2's definition).** "A2 agrees"
required exp's label to be SIGNED LEARNS BETTER without a RUNAWAY suffix; it is NO SIGN COST, so A2 does not agree, and the
branch that applies is: **"no sign cost for exp at lr 1e-3"**, a positive registered statement, read with prereg Sec. 8's
caveat -- exp starts past pi on 13 of 32 channels per head, so the registered reading of the success is that start, not
"non-negativity is free". **The 3e-3 exp cell stays instability** (8/8 RUNAWAY; Amendment 1.4 and 3.2: it licenses no
learnability sentence whatever A2 shows). So the sign claim in the SSM is a claim about **|.| and softplus increments**; it
is not a claim about every non-negative parameterisation.

The Holm family matters for neither label: the declared 2-cell sensitivity ({softplus, exp}) gives the same labels
(softplus Fisher adj 0.0028, permutation adj 0.0006; exp 1.0 / 0.72).

## Block A2 table (lr 1e-3, m = 32, n = 8 per arm)

| arm | SOLVED | classes (S / T / D / R) | acc T = 128 | acc T = 512 (4x, descriptive) | RUNAWAY | vs signed (registered) |
|---|---|---|---|---|---|---|
| signed | **7/8** | 7 / 1 / 0 / 0 | **0.994** +/- 0.008 | 0.764 +/- 0.273 | 0 | -- (G0 passes) |
| exp | **6/8** | 6 / 2 / 0 / 0 | **0.985** +/- 0.032 | 0.753 +/- 0.137 | 0 | **NO SIGN COST** (Fisher 1.0, perm 0.72) |
| softplus | 0/8 | 0 / 8 / 0 / 0 | 0.938 +/- 0.033 | 0.533 +/- 0.111 | 0 | SIGNED LEARNS BETTER (Holm: Fisher 0.0042, perm 0.0006) |
| \|.\| (abs) | **0/8** | 0 / 8 / 0 / 0 | 0.756 +/- 0.041 | 0.431 +/- 0.044 | 0 | SIGNED LEARNS BETTER (raw: Fisher 0.0014, perm 0.0002) |
| index (gate GI) | 0/8 | 0 / 6 / 1 / 1 | 0.709 +/- 0.018 | 0.415 +/- 0.050 | 0 | integration gate passes |

Mean +/- sd over seeds; none diverged. Beside block A (3e-3, `G3_SSM_RESULTS.md`): signed 8/8 -> 7/8, |.| 0/8 -> 0/8,
softplus 1/8 -> 0/8, exp 0/8 (8 RUNAWAY) -> 6/8 (0 RUNAWAY), index 0/8 -> 0/8. |.| at 1e-3 sits below the retrace floor
(0.756 vs 0.864) and near the index arm; softplus above retrace but never solving; exp and signed solve.

**Signed at 1e-3 solves on loss with partial codes on half the seeds (descriptive).** Four signed runs end antipodal
(cos(z_+, z_-) -1.000, every non-blind channel cancelling, T = 512 0.995-1.000); the other four (s2100, 2102, 2103, 2105)
have cos -0.46 to +0.11, 32-48% of channels cancelling and T = 512 0.43-0.78 (s2100 is the STALLED run, 0.977). The pilot
(Amendment 2, s2184) had flagged this. So at 1e-3 the signed anchor is weaker than at 3e-3 (8/8, every run antipodal), and
exp's T = 512 mean (0.753) equals signed's (0.764).

## Registered joints at the same rate (H2 joins A2; Amendment 3.3)

H2's existence labels were fixed by data already read (`G3_SSM_RESULTS.md`; |.|, softplus, exp: PERTURBED AND RE-FOUND,
E2' HELD). Same-rate joints (1e-3), which supersede the minimal_h2 CROSS-RATE joint:
- |.|: **COST, SOLUTION PERTURBED AND RE-FOUND**;
- softplus: COST, SOLUTION PERTURBED AND RE-FOUND (construction (f), omega' = 2 pi / ln 2: a property of the construction);
- exp: none (no cost from scratch at 1e-3).

The registered SEARCH COST joint (needs HELD under E2) did not form at either rate; on E2' (declared secondary) |.| and
softplus at 1e-3 read exists, kept, not found from scratch, and exp reads exists, kept, and found (6/8).

## Mechanism (descriptive; registered readouts from `eval.json`, plus (analysis) items computed post hoc)

Registered readouts, medians per arm (u-ratio = |z_+ + z_-| / (|z_+| + |z_-|) of the move latents; cancel share = share of
non-blind channels whose move-angle sum is within 0.1 rad of 0 mod 2 pi; "by wrap" = cancelling with |raw sum| > 1 rad;
move steps = Delta before omega):

| arm (lr 1e-3) | u-ratio | cos(z_+, z_-) | cancel share (by wrap) | max raw move-angle sum (rad) | move step min / absmax |
|---|---|---|---|---|---|
| signed | 0.261 | -0.728 | 0.74 (0.00) | 2.7 | -2.01 / 2.01 |
| exp | 0.895 | +0.601 | **0.22 (0.21)** | **15.9** | 0.04 / 5.2 |
| softplus | 0.887 | +0.569 | 0.12 (0.11) | 10.1 | 0.005 / 3.0 |
| \|.\| | 0.640 | -0.228 | 0.05 (0.01) | 5.4 | 0.0 / 1.7 |

**exp solves by wrapping, not through zero.** Its move latents are not antipodal (cos +0.60; signed -0.73, -1.000 on the
four clean seeds), every step stays positive (minimum 0.004-0.33 per run), and the channels that cancel do so almost all by
a wrap (0.21 of 0.22). Per seed exp cancels on 10-14 of 59-63 non-blind channels (s2102, STALLED, 6); the largest final
move step is 4.1-7.8, against 10^3 to 3.6e24 for the RUNAWAY runs at 3e-3.

(analysis; `docs/audits/2026-10-25/g3_a2_exp_mech.py` -> `g3_a2_exp_mech_out.txt`, the 32 non-index A2 checkpoints read
through `construct_ssm_sign.load_source`; post hoc, not a registered readout) Per channel, against the initial omega (omega_0 = 2 pi .. 2 pi / 32; exp's initial step is exp(0) = 1,
so a channel starts past pi iff 2 omega_0 > pi, 13 of 32 per head, 26 per model):
- **90 of exp's 95 cancelling channels (over 8 seeds) are channels that started past pi**; 92 of 95 cancel by a wrap,
  at 1-3 full turns (k = round(sum / 2 pi) in {1, 2, 3}). Per model about 12 of the 26 past-pi-start channels end
  cancelling. exp therefore learns the map the way the past-pi |.| start did in attention (MECHANISM M2: 8/8 by
  wrapping): on the channels its initialisation already put past pi.
- omega shrinks to 0.44-0.49 of its initial value (median per run) in every arm (weight decay), so exp's steps grow to keep
  those channels past pi (median raw sum over non-blind channels 1.2-2.3 rad, maximum 8.5-22 rad).
- softplus shows the same signature on fewer channels and does not solve: 58 of its 63 cancelling channels lie in its own
  past-pi start set (2 ln 2 omega_0 > pi), 5-12 per model. |.| cancels on 1-4 channels per model, 7 of 20 by a wrap.
  Signed cancels through zero (8 of 329 cancelling channels by a wrap).

**What this does and does not separate.** The registered caveat reads exp's success as its past-pi start. The batch does not
test that: exp's start, its multiplicative parameterisation (d Delta / d s = Delta, so steps grow geometrically) and the
mixer are not varied. One cross-batch fact argues against "the start alone": in softmax attention the same exp
parameterisation, with the same initialisation scheme, solved 1/8 at lr 1e-3 (`SIGNROUTE_RESULTS.md`, ring, seeds
1800-1807; never pooled, other seeds). So exp's no-cost is shown in this SSM at 1e-3 only; in attention exp still pays.

## Checks on the reading

- **experiment_audit** (`python3 -m mapformer.experiment_audit --runs-dir runs/ssm_sign/p0/A2`, read-only): exit 2, as on
  this layout (no (json, pt) pairs; check [1] only). Its windowed classes equal the registered ones in every cell: signed 7
  SOLVED + 1 STALLED, exp 6 + 2, softplus 8 STALLED, |.| 8 STALLED, index 6 STALLED + 1 DESCENDING + 1 RISING. Its "not all
  arms converged" flag is the index arm (5/8 flat); every non-negative and signed cell is flat at 300 epochs.
- **Rule 2:** r(log10 final loss, acc) -0.689 at T = 128 and -0.891 at T = 512 over the 80 block-A and A2 runs (pooled
  across hosts: A on koopman, A2 on kalman; descriptive). The exp verdict rests on SOLVED counts and accuracy together.
- **Power (prereg Sec. 7):** with signed 7/8, SOLVED fires for a non-negative arm at <= 2/8; NO SIGN COST needs k >=
  k_signed - 1 and accuracy within 0.01, which exp meets (6/8, -0.009). exp's label could have gone the other way.
- **Every number** in this file was re-read from `G3_SSM_A2_ANALYSIS.txt` / `G3_SSM_A2_VERDICTS.json` or the 40 `eval.json`
  files; the (analysis) counts from the checkpoints.

## Predictions vs outcomes

| registered prediction (Sec. 8, Amendments 2-3) | outcome |
|---|---|
| SIGN COST for |.| at both rates | yes: 0/8 vs 7/8 at 1e-3 (0/8 vs 8/8 at 3e-3) |
| BR SIGNED LEARNS BETTER for softplus | yes at 1e-3 (0/8, Holm-surviving) |
| BR SIGNED LEARNS BETTER for exp (from attention) | **no**: NO SIGN COST at 1e-3 (the 3e-3 cell was RUNAWAY) |
| no registered prior for exp; "an exp success would be read as that start" | exp 6/8, cancelling by wrapping on channels that started past pi (analysis); the start is not separated from the parameterisation or the mixer |
| G0 at 1e-3 "a real risk" (pilot) | passed at 7/8, with partial signed codes on 4 seeds |

## Reading for the paper

At the second rate the sign cost in the SSM holds for the two parameterisations that start small or flat at zero: |.|
learns the ring map on 0/8 seeds and softplus on 0/8 against 7/8 for the signed step. The LRU-style exp step does not pay at
this rate: 6/8 against 7/8 (registered NO SIGN COST), stable, and it reaches the map by wrapping, with almost all of its
cancelling channels among those its initialisation already put past pi. That identifies which non-negative
parameterisations pay in this model, |.| and softplus, and narrows the claim: the learnability cost is a property of the
|.| and softplus increments (and of exp in softmax attention, 1/8 at both reference rates), not of non-negativity as such.
Why exp escapes is open: softplus also starts some channels past pi (9 of 32 per head, prereg) and cancels by wrapping on
them, on fewer channels, without solving; exp's start, its multiplicative step and the mixer are not separated.

## Scope

One layer, one task (ring of 32 cells, moves only stepping), two rates from scratch (3e-3, 1e-3; no sweep), 300 epochs,
no decay or output gate, Delta entering only the phase (not Mamba-faithful). exp at 3e-3 is instability, not a learnability
statement; exp's no-cost is at 1e-3 in this SSM only and is not separated from its past-pi start (Sec. 8 caveat). The signed
anchor at 1e-3 solves 7/8 with partial codes on four seeds. m = 8 and block B (every token steps) were not run. A and A2
ran on different hosts; no contrast crosses hosts.
