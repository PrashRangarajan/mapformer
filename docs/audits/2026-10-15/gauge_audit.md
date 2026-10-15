# GAUGE independent code check (rule 29), 2026-10-15, before launch

Scope: `GAUGE_PREREG.md`, `environment_steps.py`, `train_gauge.py`, `analyze_gauge.py`, `run_gauge.sh` at HEAD 4c04bde.
Blind to results: the running pilot (`runs/gauge_pilot/`, seed 1390) was not read. CPU only.
Scripts: scratchpad `envcheck.py`, `floor2.py`; smoke in scratchpad `gauge_smoke/`.

No BLOCKER. The code does what the pre-registration says. Two MAJOR points are about interpretation, not code.

## Verified (no action)
- **environment_steps reproduces environment_cancel for (1, -1) bitwise**: tokens, obs mask, revisit mask and
  `visited_locations` identical at np seeds 0, 1300, 1307, 10**6, batch 64 x T 128 and 8 x T 512.
- **(1, 2) bookkeeping**: positions recomputed from the action tokens match `visited_locations` on 200 trajectories;
  `is_revisit` = "cell seen before" exactly; action positions never masked.
- **Pairing bonus**: at one np seed the action token stream is identical for (1, -1) and (1, 2) (the RNG calls do not
  depend on deltas), and SO/AO/m/p share initial weights at a seed (AO differs only by |.|). The eval stream (np seed 10**6)
  is the same 400 trajectories for every run.
- **Premise is a map task**: (1, 2) revisits have net displacement != 0 in 100% of cases (laps only); (1, -1) 87.5% net 0.
  Revisit rate 0.759 (m) vs 0.751 (p), as stated. Trivial floor (predict blank) 0.500 / 0.501; the best fixed-lag copy
  rule is below it (0.35); a 1-layer index model cannot count +1 vs +2, so RoPE_p is the right reference.
- **Labels**: identical to WRAP_EXISTENCE E3 / MECHANISM M4. All four reachable (SIGNED LEARNS BETTER via Fisher e.g.
  0/8 vs 5/8 p 0.026; perm min p 1.6e-4 at 8 vs 8). Fisher and perm are two-sided; d = AO - SO.
- **Same-sign secondary is gauge-invariant**: a T1 flip negates both move angles of a channel (via the W_out row or via
  omega, which `action_angles` includes). Checked: flipping omega on 32 random channels leaves the share unchanged (0.703).
- **Driver**: `--deltas="$D"` parses "1,-1" to (1, -1) (smoke logs print `deltas=(1, -1)`); `${C%_*}` gives the arm and
  the checkpoint name `SO.pt` / `AO.pt` / `RoPE.pt`, matching `train_gauge`; 8 x 5 x 2 = 80 artifacts; GUARD covers the
  whole executed import path (`model_wrap_exist -> model_sign -> model -> lie_groups`; RoPE is
  `model_baseline_rope.MapFormerWM_RoPE`; `train`, `data_parallel`, `environment*`, `train_cancel`, `stats_core`); tracked
  tree clean; `running()`/`drv_wait_dir` use the rule-23 awk. Analysis defaults (root, seeds 1300-1307) match the driver.
- **Smoke** (5 cells x seeds 1300-1301, 2 epochs x 2 batches, CPU, 20 trials, driver layout): all 10 runs wrote
  `eval.json` and the checkpoint; `analyze_gauge --root ... --seeds 1300-1301 --no-json` ran end to end (both gates
  UNREADABLE as expected at the floor) and wrote no JSON.

## Findings

1. **MAJOR -- the two tasks differ in revisit structure, not only in relative sign.** Gap (steps since last visit):
   (1, -1) median 12, 46% <= 10 steps, 87.5% net-zero (backtrack) revisits; (1, 2) median 23, 0% <= 10, 0% net-zero,
   q10 16. Every (1, 2) revisit needs the lap condition 32 a(+1) = 0 mod 2 pi in both arms; (1, -1) offers the signed arm
   short backtrack revisits that need no wrap. So G1-vs-G2 confounds relative sign with "all revisits are laps". Also the
   gate risk: SO_p may solve less often than SO_m (lap-only, no short-gap curriculum) -> UNREADABLE.
   Fix (amendment, eval-only, no new training): (a) disclose the gap / net-displacement numbers above; (b) declare a
   stratified secondary: accuracy on net-zero vs lap revisits for SO_m / AO_m, and by gap bin (<= 10, 11-20, >= 21) for all
   cells (does AO_m fail on the net-zero stratum specifically, as the sign account says?); (c) read the pilot's SO on
   (1, 2) before launch; if it is not SOLVED at 300 epochs, extend the budget now (rule 4), not after an UNREADABLE.

2. **MAJOR -- the reading overstates what the test can confirm or refute.** T1 and T1(i) are representational (proved);
   the appendix itself says the corollary "says nothing about learning speed" (P4 kink at s = 0). The batch tests the
   added hypothesis that the +1/-1 learnability cost is caused by the excluded through-zero (opposite-sign) branch.
   "REFUTED" cannot refute T1; it would refute that hypothesis (P4's sticky wall is a sign-independent alternative cause).
   Also CONFIRMED could arise because SO is slowed on laps-only (finding 1) rather than AO freed.
   Fix: rename to "GAUGE ACCOUNT SUPPORTED / NOT SUPPORTED"; state the hypothesis; add to CONFIRMED the declared
   condition AO_p SOLVED > AO_m SOLVED (same seeds, identical init and action stream), and report SO_p vs SO_m likewise.

3. **MINOR -- the same-sign secondary is not wrap-invariant, not usage-weighted, and has no baseline.** Gauge-invariant
   (verified), but a(+2) and a(+2) - 2 pi (step shifted by 2 pi / omega) are the same model with opposite sign
   (rule 8): omega spans 2 pi/32 .. 2 pi, so high-frequency channels wrap readily. Unweighted channel shares were
   withdrawn in ACCOUNTS (drift in unused channels). The init share is already 0.70 (seed 1300) and is identical for
   m and p. Fix: report (i) share with |wrap(A1 - 2 A0)| < 0.1 (encodes the map mod 2 pi), (ii) among those, the share
   with |A1 - 2 A0| < 0.1 unwrapped (coherent representative), (iii) both weighted by `usage_weights`, (iv) the init value.

4. **MINOR -- Reading prints "AS FALLS" when a gate fails** (smoke: both UNREADABLE -> "AS FALLS"). Fix: if either G is
   UNREADABLE, Reading = "UNREADABLE (G1/G2 gate)"; also say AS FALLS when G2 is not SIGNED LEARNS BETTER (G1 uninformative).

5. **MINOR -- floor not reported (rule 1).** Fix: print the predict-blank floor of the eval stream per task (0.500 / 0.501
   on 3000 trajectories) beside the cell means; flag RoPE_p if any seed is SOLVED or its mean exceeds floor + 0.05.

6. **MINOR -- NO SIGN COST is a point-estimate label, not an equivalence test** (AO one SOLVED fewer and -0.01 pass).
   Kept for consistency with E3/M4; in the results file quote the perm CI and write "no cost detected" when its lower end
   is below -0.05.

7. **MINOR -- premise wording.** "a(+2) = 2 a(+1) mod 2 pi has both angles of one sign" is false for the mod-2pi class
   (a representative can have either sign). Precise: every channel admits the unwrapped representative a(+2) = 2 a(+1),
   which is sign-coherent and so in the |.| class by T1(i) (omega' = D omega), near init (lowest channel omega = 2 pi/32:
   steps 1 and 2); on (+1, -1) the near-init representative a(-1) = -a(+1) is excluded and the |.| class needs
   a(-1) = 2 pi - a(+1) (T1(iii)). Whether SO lands on the coherent representative does not affect G1; it affects only
   the secondary (finding 3).

8. **MINOR -- accuracy at 4T** is out of distribution (rule 10; 24 laps on p): label it so. `train_gauge` docstring's
   example dir `AO_d12_s1300` differs from the driver's `AO_p_s1300` (cosmetic).
