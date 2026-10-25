# G-3 block A2 launch -- independent results-blind code check (rule 29)

Scope: the uncommitted diff to `analyze_ssm_sign.py` (preset `a2_after_minimal_h2`, selftest case), `run_ssm_sign.sh`
(A2-only mode) and `G3_SSM_PREREG.md` Amendment 2, against the registration (Sec. 5, 6, 8, 10), Amendment 1 and
`G3_SSM_RESULTS.md`. No A2 run exists (`runs/ssm_sign/p0/A2` absent; no `G3_SSM_A2_*`, `.ssm_sign_a2_done`,
`ssm_sign_a2.log`). The checker launched no training, committed nothing in the repository and edited no file under review.

**Verdict: launchable after an amendment (Amendment 3 below), with the one-line code fix of MAJOR 1 if Amendment 3 adopts
it.** No BLOCKER. 1 MAJOR, 3 MINOR, 6 NOTE.

## What was run (CPU only)

| check | result |
|---|---|
| `python3 -m mapformer.test_ssm_sign` | 40/40 PASS |
| `python3 -m mapformer.analyze_ssm_sign --selftest` | PASS, incl. the new `a2_after_minimal_h2` case |
| `analyze_ssm_sign --config minimal_h2 --root runs/ssm_sign/p0` into scratch vs committed `G3_SSM_VERDICTS.json` | parsed JSON **equal**; bytes differ only in key order of `solved` / `runaway` (NOTE 2); `G3_SSM_ANALYSIS.txt` differs only by a SyntaxWarning line from `model_gated.py` on stderr |
| standalone provenance check (current kalman files vs `p0/code_md5.txt`) | 104 guarded files, set difference empty, changed = exactly the 3 exempt files |
| who imports `analyze_ssm_sign` | no module (so the exempt analysis file cannot change training) |
| real root | 104 eval.json, 136 checkpoints; `config.txt` = `minimal_h2` |
| torch on kalman / in block-A eval.json | 2.10.0+cu128 / 2.10.0+cu128 |

Driver dry runs: a scratch git clone (`scratchpad/clone/mapformer`, the three files committed there), a copied p0 with
eval.json path-rewritten and every `.pt` replaced by an **empty placeholder** (no link to a real checkpoint),
`DRV_DRYRUN=1 DRV_GPUS=0 CUDA_VISIBLE_DEVICES=""`, under `timeout`. Script `scratchpad/dryrun.sh`.

| dry run | result |
|---|---|
| 1. `a2_after_minimal_h2`, no A2 results | 40 DRYRUN lines, **0 real launches**, all `--lr 1e-3 --device cuda:0`, all into `p0/A2/`, seed-major; provenance 0 mismatches; md5 guard written to `p0/A2/code_md5.txt`; fails closed: MISSING 80 (all in A2, none outside); no marker |
| 2. synthetic A2 (block-A eval.json relabelled lr 1e-3; plumbing only) | 40 skips, 0 launches, 320 artifacts, analysis with suffix "; lr 1e-3 (m = 32): ..." and "joint, same rate (1e-3)"; `.ssm_sign_a2_done` set; `.ssm_sign_done` not touched |
| 3. non-exempt guarded file (`environment_steps.py`) changed and committed | provenance mismatch `['environment_steps.py']`, REFUSED, before any launch |
| 4. default `G3_CONFIG=minimal_h2` into the base root | ABORT (md5 guard) |
| every run | base `config.txt` and `code_md5.txt` md5-unchanged; real `runs/ssm_sign/p0` unchanged; 0 `train_ssm_sign` processes |

Also probed through `_write` / `load` / `analyse` in the `a2_after_minimal_h2` config: A2 signed 5/8 -> READING suffix
`UNREADABLE (signed SOLVED 5/8 < 6)`, every A2 label gated, every H2 joint None; A2 index 2/8 -> `TASK DOES NOT NEED
INTEGRATION`, every H2 joint None.

## MAJOR

**MAJOR 1 -- Holm family for block A2: the registered text governs, and the code does not implement it.**
`G3_SSM_PREREG.md:255` (Sec. 6, unchanged by Amendment 1): "abs vs signed raw p's; softplus, exp vs signed Holm over the 3
cells per test". `analyze_ssm_sign.py:328-329` adjusts over 2 (softplus, exp) and gives abs raw p. Amendment 2
(`G3_SSM_PREREG.md:461-466`) declares the code governing, on the ground that "the text's 3 matches no split that keeps abs
at raw p".
- Ruling. That ground is wrong: "Holm over {abs, softplus, exp}, used for the softplus and exp labels, with abs's own
  label at raw p" is a coherent split (abs as the unadjusted primary, the two secondaries adjusted within a family that
  also counts abs) and is the literal reading of line 255. Amendment 1 does not touch the A2 family: A1.6 sets "BR's Holm
  family is 2 x (number of sizes) cells", which is block A's BR line (Sec. 6, line 249), not A2's. The Amendment 1 audit
  states only "Holm per test over the stated families" (`docs/audits/2026-10-23/g3_ssm_audit.md:51`) and did not compare
  the A2 family with the text. So per the registered text and Amendment 1 the family is the 3 cells; the code is the
  deviation.
- Material, in the regime A2 is expected to be in. Fisher-only scan over k_signed in {6, 7, 8} and all k_abs, k_softplus,
  k_exp: 80 of 4,374 softplus / exp cells change verdict between the two families, all at k_signed 6-7 -- the G0-risk
  regime the pilots point to (s2180 signed STALLED at 1e-3, s2184 solved with a partial code). Example: signed 6/8, abs
  1/8, softplus 0/8, exp 1/8: exp Fisher adjusted 0.041 (code, fires) vs 0.081 (text, does not); the permutation branch may
  or may not rescue it. With abs the smallest p (as in block A) the two families agree.
- Choosing the less conservative family after block A was read is not data-dependent for A2, but it is the
  anti-conservative choice and contradicts the registered text; Amendment 2's promise to report both does not settle which
  label is registered.
- Fix (preferred; a two-line change to an exempt file, so provenance is unaffected):

      labs2 = contrast_labels(C2, C2["signed"], ["abs", "softplus", "exp"], n, True, g2)   # Holm over the 3 cells (Sec. 6)
      labs2["abs"] = contrast_labels(C2, C2["signed"], ["abs"], n, False, g2)["abs"]      # abs: raw p (Sec. 6)

  plus a selftest case that separates the families (the case above: signed 6/8, abs 1/8, softplus 0/8, exp 1/8 with
  accuracies chosen so the permutation branch cannot fire; assert exp's `fisher_p_adj` = 0.0811). This changes `full`'s A2
  equally (no full run exists). Report the 2-cell label as a declared sensitivity line.
- Alternative (acceptable only if stated as a deviation): keep the code, and Amendment 3 says "deviation from the
  registered text, chosen after block A was read; the 3-cell labels are reported beside it and any disagreement is
  read UNMEASURED". The preferred fix is simpler to defend.

## MINOR

**MINOR 1 -- "Unless A2 agrees" (`G3_SSM_PREREG.md:476-483`): faithful and made before A2 data, but incomplete.**
Made before any A2 run exists (verified) and after block A and pilot s2184 were read (disclosed). Keeping the 3e-3 exp
cell as instability even when A2 agrees is narrower than Sec. 8 line 302 literally allows. It is nonetheless right,
because Amendment 1.4 (k_RUNAWAY >= 2 licenses no learnability sentence) binds the 3e-3 cell (8/8 RUNAWAY). The amendment
should cite A1.4 as the reason. Gaps:
- (a) NON-NEGATIVE LEARNS BETTER for exp at 1e-3 has no branch. It is reachable through the accuracy branch when signed
  passes G0 with partial codes, and pilot s2184 had exp 0.998 vs signed 0.996.
- (b) NO SIGN COST and UNMEASURED are merged into one sentence, though NO SIGN COST is a positive registered statement
  ("no sign cost for exp at lr 1e-3").
- (c) "agrees" depends on the Holm family of MAJOR 1; it should name the family.

**MINOR 2 -- base-root completeness check is narrower than the final artifact count** (`run_ssm_sign.sh:98-106` vs
`174-177`). `BASEREQ` omits `H/<C>_built_moves_m32_s<S>/<C>.pt` and `H2/.../<C>.pt`, which the final `REQ` requires. If
one were missing, the batch would spend ~3 h and then fail closed at "missing runs" (no wrong output, no marker). Not a
risk on the real root (136 checkpoints = 40 A + 32 built + 32 H + 32 H2). Fix: add both to `BASEREQ`.

**MINOR 3 -- selftest coverage of A2 labels** (`analyze_ssm_sign.py:583-595`). The new case asserts COST, the RUNAWAY
suffix on A2 exp, the absent exp joint and "same rate (1e-3)". The assertions are exact string comparisons, so a wrong
reading or joint kind prints BAD. Earlier `full` cases add A2 NO SIGN COST and A2 UNREADABLE. Gaps:
- no case reaches A2 TASK DOES NOT NEED INTEGRATION, UNMEASURED or NON-NEGATIVE LEARNS BETTER;
- no case asserts the H2 joints under an A2 gate failure;
- no case separates the two Holm families.

The checker's probe shows the G0 and GI paths behave correctly, but nothing in the selftest guards them. Fix: add the A2
G0-fail and GI-fail cases (assert every `joint_lr1e-3` is None) and the Holm case of MAJOR 1.

## NOTE

1. **Driver safety (item 4): verified.**
   - The heredoc `python3 - ... <<'PYEOF' || { ...; exit 1; }` parses correctly in bash. A failing check (dry run 3) and
     a crash inside the check (non-zero exit) both refuse.
   - The exempt list covers files that never reach training: nothing imports `analyze_ssm_sign`, and the driver and the
     prereg are not imported. A changed module set also fails (set difference).
   - NWANT = 8 x (2x1x5 + 3x4) + 80 (A2) + 64 (H2) = 320, equal to the listed REQ (80 A + 80 A2 + 96 H + 64 H2).
   - Artifacts are counted, the md5 guard re-checked, then the analysis runs, then `drv_done` requires both outputs
     before the marker.
   - Block A / H / H2 can be neither re-run nor overwritten: the stage-1 A loop and stage 2 are skipped; `launch` skips
     any dir with eval.json; provenance files go to `p0/A2/`; outputs, log and marker have A2 names. `minimal_h2` is
     ABORTed and `core` / `full` are REFUSED on the base root.
2. **"Reproduces `G3_SSM_VERDICTS.json` exactly"** (Amendment 2 checks) holds for content, not bytes. `v["solved"]` and
   `v["runaway"]` (`analyze_ssm_sign.py:419-422`) iterate a set, so key order follows PYTHONHASHSEED. Suggest `sorted(...)`
   at the next code change. Not a launch issue.
3. **Seeds (item 5).** Driver `SEEDS` = 2100-2107 = Sec. 5 = Amendment 2. No run dir on kalman outside `runs/ssm_sign/`
   carries them. RANKWD registered the same labels (`run_rankwd.sh:22`; its run dirs are not on kalman), so Sec. 5's
   "fresh; no run dir in the repo carries them" (`G3_SSM_PREREG.md:201`) is stale. Amendment 2 discloses the label
   overlap. No run, checkpoint or contrast is shared.
4. **G0 risk (item 6).** A G0 failure gives UNREADABLE on every A2 label and in the READING suffix, and no H2 joint. A GI
   failure gives TASK DOES NOT NEED INTEGRATION in the same way. No false label can form. Under G0 failure the A2 output
   carries no H2 joint while the minimal_h2 output keeps its CROSS-RATE one; Amendment 3 should say which is cited (see
   text).
5. **Hosts (item 7).** All registered A2 contrasts are within A2 (kalman), and torch and code are byte-identical by the
   provenance check. Two caveats:
   - the same-rate H2 joint combines a kalman A2 label with koopman H2 existence labels. That is a conjunction of labels,
     not a contrast, so it is acceptable, but it should be said.
   - "each A2 run has the same initial weights and walk stream as its block-A twin" is unverified across hosts. No test
     uses the pairing (tests are unpaired), so nothing depends on it.
6. **Operational.**
   - `DRV_GPUS` defaults to `1`, which is GPU *index* 1 (`run_ssm_sign.sh:26`), the GPU reserved for another batch.
     The launch line sets `DRV_GPUS=0`; a launch without it would land on GPU 1. Optionally make A2-only mode refuse an
     unset `DRV_GPUS`.
   - The prereg is md5-guarded in `p0/A2/code_md5.txt`, so Amendment 3 must be committed before launch and the file
     left untouched until `.ssm_sign_a2_done`.
   - The launch also needs a tree with no modified tracked file anywhere, so the parallel FACTORIAL_N8 work must be
     committed or untracked.

The H2 joint's change from cross-rate to same-rate (item 3) is faithful. The module docstring
(`analyze_ssm_sign.py:51`) registers "H2 joins A2 when A2 is run", and `full` was the only configuration with A2 when
Sec. 6 was written. Amendment 2's claim that only "COST, SOLUTION PERTURBED AND RE-FOUND" or no joint can form at H2 is
correct given H2's fixed E2 labels: the existence computation reads block A's sources only and does not depend on the
configuration, and dry run 2 printed exactly those joints.

## Proposed Amendment 3 text

    ## Amendment 3 -- independent check of the A2 launch (rule 29), 2026-10-25, before A2 is run

    From `docs/audits/2026-10-25/g3_a2_audit.md` (results-blind for A2; no A2 run exists). No BLOCKER.

    1. **Holm family for block A2 (MAJOR 1; supersedes Amendment 2's Holm paragraph).** The registered text (Sec. 6, "Block
       A2") governs: the softplus and exp labels use Holm per test over the 3 cells abs, softplus, exp; abs's own label
       uses raw p. Amendment 2's statement that the text "matches no split that keeps abs at raw p" was wrong. Amendment 1
       did not change the A2 family (A1.6 concerns block A's BR). `analyze_ssm_sign` is changed to match (two lines in the
       A2 block). A selftest case separates the two families. The 2-cell labels are printed as a declared sensitivity line.
       The change is in an exempt file of the A2 provenance check; model, trainer, task and library code are unchanged.
    2. **"Unless A2 agrees" (MINOR 1).** The 3e-3 exp cell stays instability whatever A2 shows, because Amendment 1.4 binds
       it (8/8 RUNAWAY). "A2 agrees" uses the A2 exp label of item 1. Two branches are added:
       - A2 exp NON-NEGATIVE LEARNS BETTER: read as the past-pi start (Sec. 8), never "non-negativity is free";
       - A2 exp NO SIGN COST: "no sign cost for exp at lr 1e-3" (with the same caveat), kept distinct from UNMEASURED
         ("no exp cost shown at 1e-3").
    3. **Which H2 joint is cited.** In the A2 output the same-rate H2 joint is the registered one. The minimal_h2 CROSS-RATE
       joint is reported as superseded, including when A2 is UNREADABLE (G0) or TASK DOES NOT NEED INTEGRATION (GI). In
       those cases no H2 joint is registered, and the cross-rate one is not promoted back. The same-rate joint combines an A2
       label (kalman) with H2 existence labels (koopman): a conjunction of labels, not a cross-host contrast.
    4. **Driver (MINOR 2).** The base-root check also requires every H and H2 continuation checkpoint, which the final
       artifact count already requires, so an incomplete root is refused before training rather than after.
    5. **Selftest (MINOR 3).** Added in the `a2_after_minimal_h2` configuration:
       - A2 G0 failure and A2 GI failure, each asserting the READING suffix and every H2 joint None;
       - the Holm-family case of item 1.
    6. **Stated.**
       - Sec. 5's "no run dir in the repo carries them" is stale: RANKWD used the labels 2100-2107; no run is shared.
       - Cross-host pairing of A2 with its block-A twin is unverified and unused (the tests are unpaired).
       - The minimal_h2 re-analysis equals `G3_SSM_VERDICTS.json` as parsed JSON (key order of two summary dicts varies
         with the hash seed).
       - Launch with `DRV_GPUS=0` (the default `1` is GPU index 1).

    **Checks after this amendment:**
    - `test_ssm_sign`, `analyze_ssm_sign --selftest` (incl. the new cases);
    - `--config minimal_h2` re-analysis equal as parsed JSON;
    - a DRV_DRYRUN scratch-clone run of `a2_after_minimal_h2`: 40 launches at 1e-3 into `p0/A2/`, fail-closed on 80
      missing artifacts.
