# WRAP_EXISTENCE: independent code check (rule 29), 2026-10-13

Scope: `WRAP_EXISTENCE_PREREG.md`, `model_wrap_exist.py` (including the coordinator's mid-review edit: step-space
remainder and `gauge=True`), `train_wrap_exist.py` (including `AO_wrapg`), `analyze_wrap_existence.py`,
`run_wrap_existence.sh`. All checks ran on the CPU (`CUDA_VISIBLE_DEVICES=""`). Scripts are in the session scratchpad
(`t1`-`t6.py`). I read the pilot `runs/wrap_existence_pilot` (seed 690) and RING_DIAL checkpoints without writing to them.
No repo file was edited except this one.

Verdict: the classes, the trainer plumbing and the stats calls are correct. E2 as registered is not fit to launch, for
two reasons: the construction's weight scale decides the outcome (findings 1-2), and "held" is read only at the end
(finding 3).

## Findings

**1. BLOCKER: E2 measures the construction's parameterisation, not whether a non-negative solution is stable (rules 14, 15).**
On the pilot SO (seed 690, SOLVED, acc 1.000) the two action latents are nearly antiparallel: cos(lat+, lat-) = -0.993,
singular values 7.33 / 0.44, condition number 16.6. On RING_DIAL Signed_r4 s30-33 the cosine is -0.996 to -0.999 and the
condition number 22-42. This is expected: a cancelling code with a full-rank W_out needs lat+ ~ -lat-. With W_in kept, the
min-norm W_out has to give two non-negative targets from antiparallel latents, so it is inflated by about 1/sv_min.
Measured:

| construction (pilot s690) | max abs W_out | max step | 400 AdamW steps at lr 1e-3 (acc T=128, 100 trials, at steps 0/25/100/400) |
|---|---|---|---|
| signed source (SO_cont) | 0.39 | 2.76 | 1.000 / 0.995 / 1.000 / 1.000 |
| `gauge=False` (AO_wrap) | 1825 | 1199 | 1.000 / 0.691 / 0.889 / 0.926 (wd 0: 1.000 / 0.502 / 0.868 / 0.908) |
| `gauge=True` (AO_wrapg, current code) | 8.4 | 2.76 | 1.000 / 0.493 / 0.489 / 0.514 |
| balanced (my construction below) | 1.21 | 6.28 | 1.000 / 0.997 / 1.000 / 1.000 |
| signed at the balanced gauge (SO_bal) | 0.53 | -- | 1.000 / 0.998 / 1.000 / 1.000 |

The GPU pilot agrees. AO_wrap's per-epoch loss went 0.053 / 1.05 / 0.34 / 0.35 at epochs 5-20, while SO_cont stayed at
0.015 / 0.0005 / 0.0009 / 0.023. The collapse happens during warm-up (lr <= 6.7e-4), and AO_wrap with weight decay 0
collapses too. So this is not weight decay. It is the Adam relative step: a W_in update of about lr per entry moves a step
by lr x |W_out| (rule 15). The coordinator's `gauge=True` rescales omega and W_out per channel but keeps W_in. Its W_out is
still 22x the signed one, and it collapses faster than the min-norm construction. Any NOT HELD under either construction
would be a parameterisation artefact.

Fix. In AO, W_in acts only on the two action embeddings, so W_in is a free gauge. The balanced construction has three steps:
- Re-solve W_in = (pinv(e_act) @ L)^T, with L = c [e1; e2] (orthogonal latents, c = the source's mean latent norm).
- Set omega' = max(|omega|, 1) per channel.
- Set steps s = remainder(omega D, 2 pi) / omega', and W_out[:, 0:2] = s^T / c.

Results: exact (angle error 6e-7 rad), frozen acc 1.000, max |W_out| 1.21, |W_in| 0.27 (source 0.20), and it held for
400 steps. Make it the registered E2 construction. Also add a signed control at the same gauge (SO_bal: the same W_in, L
and omega' with the signed angle). Then AO_bal vs SO_bal differs only in the wrap. SO_cont stays as the re-warm control for
the source gauge. The coordinator's `k` reference scale (global max |source step|) is reasonable, but without
orthogonalising the latents it does not fix the scale.

**2. BLOCKER (now fixed by the coordinator's edit; regression test): negative and near-zero omega.**
Trained omega crosses zero. Pilot omega min is -0.0123. RING_DIAL s31 / s32 / s33 have -1.5e-4, -0.011 and -0.0045.
Synthetic test of the original `remainder(om D, 2pi)/om`, with omega -0.5 on one channel: angle error 2.46 rad (E1 would
read CONSTRUCTION FAILED). With omega 1e-5 on a dead channel: step 6.3e5, W_out 6.9e5. The current
`remainder(D, 2 pi/|om|)` gives angle error 0.0 on the pilot and 1e-5 on s31. Keep `n_omega_negative` in the report. One
residual issue: a dead channel whose D is slightly negative wraps to about 2 pi/|omega|, and the gauge then makes it a live
high-frequency channel (pilot: omega' -2.28 where the source had -0.012). The function is identical but sensitivity is about
190x. This is harmless in the balanced probe, but state it.

**3. MAJOR: "HELD" is read only at the end, so lost-and-refound reads as held.** The pilot AO_wrap lost the solution by
epoch 10 (loss 1.05) and was descending again at epoch 180 (0.089). If it ends below 0.05 it is SOLVED, and E2 can read
HELD. The per-epoch losses are already saved, so register a hold readout from them. Proposal: LOST if the max per-epoch
loss over the continuation is above 0.3, or above 10x SO_cont's max at that seed. HELD needs not LOST on >= n_s - 1
seeds, in addition to the current rule. Also register the wrap-aware angle distance from the construction at the end (as
in the probe's dA) as a secondary.

**4. MAJOR: register E2 separately for each construction, and say in advance what each outcome means.** Recommended
primary: the balanced construction vs SO_bal. Report min-norm and k-gauge as descriptive, since finding 1 already shows
they collapse for a parameterisation reason. Wording to register now: HELD under any exact construction shows that a
stable non-negative solution exists (the deficit is search). NOT HELD under the balanced construction, with SO_bal held,
is "unstable at the scale training uses". NOT HELD only under an ill-conditioned construction means nothing. Gate E2 on E1
EXISTS for that construction. The code currently computes E2 even when E1 fails.

**5. MAJOR: E3 NO SIGN COST is reachable vacuously.** The smoke (3 epochs, both arms at floor 0.506, 0/2 vs 0/2) printed
`E3: NO SIGN COST`. If SO solves few seeds, E3 reads "no sign cost" at the floor. Gate E3 on SO SOLVED >= 4/8 (else
UNREADABLE), or require mean SO > floor + 0.1.

**6. MINOR: the power of E2's NOT HELD branches depends on n_s.** Sign-flip min p is 0.125 at n_s = 4 and 0.0625 at
n_s = 5, so the accuracy branch cannot fire below n_s = 6. Fisher needs 0/4 vs 4/4 (p 0.029) at n_s = 4, and at n_s = 5
it needs 0/5 vs 5/5, 1/5 vs 5/5 or 0/5 vs 4/5 (p <= 0.048). State this in Amendment 1.

**7. MINOR: `wrap_share` is not invariant to wrapping.** It selects channels by raw |omega step(+1)| > 0.05. The same
angles mod 2 pi give source 0.103 vs constructed 0.148 (smoke s9001) and 0.796 vs 0.817 (pilot). Select on
|wrap(angle(+1))| > 0.05.

**8. MINOR (verified OK):** these items need no change.
- At one seed SignedAO and AbsAO initial weights are bit-identical (max diff 0.0).
- Observation and blank steps are exactly 0 by construction (`step_table` zero rows, so no gradient path).
- Abs table = |signed| (0.0).
- `forward` matches MapFormerWM.forward with masked deltas to 6e-7.
- `act` = (0, 1) is (+1, -1) in environment_cancel's vocabulary (19 tokens: 0/1 actions, 2-17 obs, 18 blank).
- No other repo caller of get_position_state / delta_of is affected.
- The data stream: base_seed = torch.initial_seed() + 10^6, identical for SO_cont / AO_wrap / AO_wrapg at a seed and
  distinct from SO's.
- `--init` assert works.
- Fisher argument order and direction checks are right. `signflip_p` / `perm2_p` return dicts with "p".
- The E1 / E2 branches match the prereg text and are disjoint.

Two items to disclose rather than fix: AO_wrap builds one extra model before training, so its torch RNG (dropout) stream
differs from SO_cont's; and the optimizer state is fresh in all continuations.

**9. MINOR: analysis output.** With n_s = 0, E1 is set but not printed. `AO_wrapg` is missing from `CLS`, from the per-seed
table, from E1 / E2 and from the driver (coordinator aware). For E1, also record the max logit difference: pilot 3e-4 at
T=128 and 5e-3 at T=512 (RING_DIAL s32 0.023 / 0.077 under min-norm). Accuracy within 0.002 of a source at 1.000 is a
weak test.

**10. MINOR: driver.** Knobs are set after sourcing (correct). The phase-2 wait loop fails closed if SO dies, and SO.pt is
written before eval.json, so the launch never reads a partial checkpoint. A hung SO would make the loop wait forever, which
is acceptable. The remaining problems:
- `[ ${#REQ[@]} -eq 64 ]` is true by construction. Count the existing files instead, and update it to 80 when AO_wrapg
  (and SO_bal) are added.
- GUARD misses `train_variant.py`, which train_cancel imports at module level, and `lie_groups.py`, which model.py imports.
- The five files are untracked or modified, so the logged "clean" is wrong (`git diff --quiet` ignores untracked files).
  Commit before launch (rule 7).
- Library-wide: the lock fd 9 is inherited by the setsid children, so a relaunch after `drv_fail` is REFUSED until the
  jobs end.
- 12 jobs x 3 data workers is about 48 processes on 32 cores. Expect slower epochs than the pilot's 2.0 s (3.1 s with
  the pilot's 2 jobs plus my CPU probes).

## Suggested Amendment 1 (summary)
(a) Primary construction = balanced: orthogonal action latents, omega' = max(|omega|, 1), steps in [0, 2 pi). Add a
signed control SO_bal at the same gauge. Keep min-norm and k-gauge as descriptive arms.
(b) E2 per construction, gated on that construction's E1, with the LOST readout from per-epoch losses.
(c) E3 gated on SO SOLVED >= 4.
(d) Wrap-aware `wrap_share` selection.
(e) Log the logit difference in E1.
(f) Driver: 80 or 96 artifacts, an extended GUARD, commit first.
(g) State the n_s-dependent reachability of NOT HELD.
