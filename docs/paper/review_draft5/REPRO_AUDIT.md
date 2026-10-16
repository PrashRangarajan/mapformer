# Reproduction audit of the registered analyses behind DRAFT5_workshop.md

2026-10-16, HEAD 08a9802, torch 2.10.0+cu128. Independent re-run from stored artifacts; nothing in `runs/` or any
committed `*_ANALYSIS.txt` / `*_VERDICTS.json` was written. Re-run outputs live in the session scratchpad
(`.../scratchpad/repro/out/`, `reeval/`); they are ephemeral, the commands below regenerate them.

## Verdict per batch

| batch | analysis script | ANALYSIS.txt | VERDICTS.json | run dirs | spot-check (2 runs) | verdict |
|---|---|---|---|---|---|---|
| SIGN_MATCHED | analyze_sign_matched.py | byte-identical | (none written) | 4 arms x 8, seeds 0-7, 900 ep | classes match; acc re-eval GPU bitwise, CPU within 7e-4 | **REPRODUCED** |
| RING_DIAL | analyze_ring_dial.py | byte-identical | byte-identical | 6 cells x 8, seeds 30-37, 300 ep | classes match; CPU re-eval exact | **REPRODUCED** |
| RING_BUDGET | analyze_ring_budget.py | byte-identical | byte-identical | 2 cells x 8, seeds 40-47, 900 / 300 ep | classes match; CPU re-eval exact | **REPRODUCED** |
| WRAP_EXISTENCE | analyze_wrap_existence.py | byte-identical | byte-identical | 7 arms x 8, seeds 600-607, 300 ep | classes match; CPU re-eval exact | **REPRODUCED** |
| MECHANISM | analyze_wrap_mech.py | byte-identical | byte-identical | 8 arms x 8, seeds 800-807, 300 ep | classes match; CPU re-eval exact | **REPRODUCED** |
| GENERALITY | analyze_generality.py | byte-identical | byte-identical | ring 4 x 8, torus 9 x 8, seeds 700-707 | classes match; ring CPU exact; torus GPU bitwise | **REPRODUCED** |
| WRAP_COPE | analyze_wrap_cope.py (`--root runs/wrap_cope/p0 --seeds 900-907`, as the driver) | byte-identical | byte-identical | 4 arms x 8, seeds 900-907, 300 ep | classes match; CPU re-eval exact | **REPRODUCED** |
| ACCOUNTS | analyze_accounts.py (t1, t2, t4) + runs/accounts/t3/summarize_t3.py | byte-identical (t3 summary.txt too) | byte-identical | t1 4 x 8 (600-607), t2 4 x 8 (1000-1007), t3 6 x 8 (10-17), t4 CoPE 2 x 16 (1100-1115) + 2 x 8 (1100-1107) | classes match; CPU re-eval exact | **REPRODUCED** (provenance note) |
| TORUS_START | analyze_torus_start.py | byte-identical | byte-identical | 5 arms x 8, seeds 1200-1207, 900 ep | classes match; CPU re-eval within 1e-4 | **REPRODUCED** (provenance note) |
| GAUGE | analyze_gauge.py | byte-identical | byte-identical | 5 cells x 8, seeds 1300-1307, 300 ep | classes match; CPU re-eval exact | **REPRODUCED** (provenance note) |
| ROBUST | analyze_robust.py (`--json-out` to scratch) | byte-identical | byte-identical | ring 4 recipes x 2 x 8, torus 2 x 8, seeds 1400-1407 | classes match; ring CPU exact; torus GPU bitwise | **REPRODUCED** (provenance note) |

No batch DIFFERS; none COULD NOT RUN. Every analysis ran on CPU (`CUDA_VISIBLE_DEVICES=""`), exit 0, empty stderr,
2 s (SIGN_MATCHED) to 2 min 39 s (WRAP_EXISTENCE). The ring analyses re-score every model (dropout-scale hook) inside the
script, so their re-scored columns are a full CPU re-evaluation, not a read-back.

## What was done

1. **Re-run.** Each script copied to the scratchpad with only the VERDICTS output path changed (`sed` on the one
   `open(f"{REPO}/X_VERDICTS.json", "w")`; ROBUST via its `--json-out`), run from `/home/prashr` with
   `PYTHONPATH=/home/prashr`, every other import from the repo at HEAD; stdout+stderr diffed against the committed files.
   For each batch the analysis script at its ANALYSIS commit equals HEAD (`git diff <commit> HEAD -- analyze_X.py` empty),
   and its last change is an ancestor of that commit. Dependencies (stats_core, model_wrap_exist, rescore_hook,
   train_variant, ...) changed after several ANALYSIS commits; the byte-identical re-runs show those changes are
   output-preserving for these analyses.
2. **Eval JSONs behind the torus readouts** (the analyses read them, they do not re-evaluate):
   `eval_noise_refine` re-run on all 32 SIGN_MATCHED models, GENERALITY SO4 / AO4 (16) and ROBUST torus SO / AO (16) at
   T = 1024 and 2048, 100 trials. **On GPU: bitwise identical to `SIGN_MATCHED.json`, `GENERALITY_TORUS_{SO4,AO4}.json`,
   `ROBUST_TORUS_{SO,AO}.json` (every seed, both lengths).** On CPU: per-seed |d| <= 6.9e-4 at T = 1024 and <= 5.2e-3 at
   2048 (argmax ties on unsolved runs); SIGN_MATCHED's analysis fed the CPU numbers gives the same primary (-0.177, perm
   p 0.0002, 0/8 vs 8/8, SIGN IS CAPABILITY); only 2048 secondaries move in the third decimal (-0.244 -> -0.243).
3. **Integrity scan** of all 648 `*.pt` under the eleven run roots: each loaded, checked for seed (dir name vs
   checkpoint), arm, `len(losses) == epochs`, `output_dir` == its own directory, `eval.json` present, one recipe per cell
   (args identical up to the seed), and a SHA-1 of the state dict and of the loss curve compared across all 648.
   Driver logs: one start each, launch counts equal the registered run counts (32 / 48 / 16 / 56 / 64 / 104 / 32 / 112 /
   40 / 40 / 80), no skip / abort / fail / traceback lines; every `.X_done` marker exists, no `_FAILED` marker.
4. **Spot-check, 2 runs per batch** (26 runs): `stats_core.classify_run` on the stored losses, accuracy read from
   eval.json / the eval JSON, and an independent CPU re-evaluation with the trainer's protocol (ring:
   `train_cancel.evaluate`, env seed 10000, np seed 10^6, 400 trajectories, T = 128, no hook; torus:
   `eval_noise_refine.evaluate`, env seed 10000, seed 1234 + s, 100 trials, T = 1024). All classes equal the per-seed
   classes the ANALYSIS files print; 24/26 re-evaluations exact, the other 2 within 1.3e-4 (CPU vs GPU argmax ties, torus: sign_matched Abs_r4_s5, torus_start AO4_high_s1201).
   Runs: ring_dial Signed_r4_p0.5_s30, Abs_r4_p0.9_s37; ring_budget e900_s41, e300_s46; wrap_existence AO_s603,
   SO_bal_s605; wrap_mech AO_high_s802, SO_s806; generality ring CoPEF_s701, Signed_r4_s704, torus SO4 s704, AO4 s705;
   wrap_cope CoPEF_Hi_s900, CoPEF_LoS_s907; accounts t1 AO_bal_wd0_s602, t2 SO_absmag_s1003; gauge AO_p_s1303,
   SO_m_s1305; robust ring lrx3 AO_s1404, wd0 SO_s1401, torus AO s1402, SO s1407; sign_matched Signed_r4_s3, Abs_r4_s5;
   torus_start AO4_high_s1201, SO4_s1206.
5. **Code versions.** Every md5 in the run roots' `code_md5.txt` resolves to a committed version of its file. Files
   changed since their batch (all reviewed by diff; all additive, defaults preserved):
   `train_variant.py` (new VARIANT_MAP entries), `rescore_hook.py` (CoPEAttn added to the known layers),
   `train_wrap_exist.py` / `train_gauge.py` (`--weight-decay`, default 0.05 = the old implicit value),
   `model_wrap_exist.py` (`wrap_share` gains `pairs=((0, 1),)` defaulting to the old behaviour; new classes and helpers),
   `train_wrap_mech.py`, `train_ring_variant.py`, `model_mono_baselines.py` (new arms only). WRAP_COPE, ACCOUNTS,
   TORUS_START, ROBUST: every recorded md5 equals HEAD; GAUGE: all but `train_gauge.py` (above). The analysis scripts
   themselves, where recorded, equal HEAD. SIGN_MATCHED's driver records training code only.
6. **`python3 -m mapformer.test_guards`** (from /home/prashr, CPU): **13/13 passed in 10.9 s.** Note that these guards
   test generic infrastructure (stats edge cases, determinism, layout guard, probes); none is specific to these batches.

Seeds match the pre-registrations (as amended: RING_DIAL n = 8 seeds 30-37, p_plus 0.5 / 0.75 / 0.9; ACCOUNTS T4 CoPE
16 seeds). No pilot seed (130, 690, 990, 1390, 1490-1496) appears in any batch directory. Recipes per cell as
registered (e.g. ROBUST torus d 256 / 2 layers; lrd3 3.33e-4, lrx3 3e-3, wd0 0; GENERALITY torus continuations lr 1e-4,
`_hi` 1e-3, 300 epochs from `runs/generality/torus/init/s7xx`).

## Integrity notes (none changes a registered verdict)

1. **Untracked artifacts behind four committed analyses.** ACCOUNTS (`runs/accounts/{t1,t2,t3,t4}`, its `code_md5.txt`),
   TORUS_START, GAUGE and ROBUST (`runs/{torus_start,gauge,robust}/`, including their `code_md5.txt`) are untracked in
   git; the analyses read their `eval.json` files (ROBUST torus reads the tracked top-level JSONs). Earlier batches
   (RING_DIAL ... WRAP_COPE) committed their eval.json and code_md5.txt. The stored files reproduce today, but the
   committed record does not carry them. Recommend committing the eval.json, code_md5.txt and t3 JSON/summary files.
2. **ACCOUNTS T1's wd 0.05 arms are bit-identical to WRAP_EXISTENCE.** `accounts/t1/{AO,SO}_bal_wd_s60x` (retrained
   2026-10-14, logs show full training) have state dicts and loss curves identical to `wrap_existence/p0/{AO,SO}_bal_s60x`
   (2026-10-13), all 16. This is determinism, not a copy -- and by rule 6 not a replication: the T1 control arm adds no
   independent evidence for WRAP_EXISTENCE's E2 HELD. No other checkpoint or loss curve is duplicated among the 648.
3. **Loss-based SOLVED vs held-out accuracy disagree for CoPE.** SOLVED is the train-mode (dropout on) final-5% loss
   < 0.05. ACCOUNTS T4: 5/16 CoPEF_line runs score 0.9997-1.0000 held-out accuracy (NLL <= 8e-4) with loss tails
   0.062-0.089, so they count as unsolved, while the one SOLVED line run (s1104) scores 0.988; CoPEF_ring has 2/16 at
   >= 0.9999, 0/16 SOLVED. At an accuracy threshold of 0.999 the line would read 5/16 vs 2/16, not 1/16 vs 0/16. The
   draft ("On the line CoPE still fails at the registered criterion (1/16 vs 0/16)") is literally correct but the
   criterion is fragile for this model family; ACCOUNTS_RESULTS.md gives only the range "0.62-1.00". Same pattern, single
   runs: wrap_cope CoPEF_Hi s904 (0.9997, tail 0.055), CoPEF_LoS s902 (0.9992, 0.059), robust ring lrx3 AO s1400 (1.000,
   0.138), wrap_existence AO_wrap s604 (0.9998, 0.162). Converse: wrap_mech AO_mid s800 is SOLVED (tail < 0.05) at
   0.968 held-out, one of AO_mid's 2/8 in M2b.
4. **Knife-edge SOLVED.** SIGN_MATCHED Pos_r4 s7 is the arm's only SOLVED run at tail 0.049 (threshold 0.05); the Pos_r4
   contrast fires on accuracy (perm p 0.0003) regardless.
5. **Attention path differs across torus batches.** SIGN_MATCHED trained with `fast_attn False`; GENERALITY and ROBUST
   torus with `--fast-attn` (rule 19: logits agree ~1e-6, training not bit-identical). Within-batch contrasts are
   unaffected; cross-batch torus comparisons (draft line 123: -0.120 GENERALITY, -0.104 TORUS_START; ROBUST's
   "a quarter of the one-layer gap") mix attention paths and model classes (Signed_r4 vs SignedAO4_r4).
6. `runs/wrap_existence/init/` holds 8 untrained Signed_r4 checkpoints built by `docs/audits/2026-10-13/wrap_existence_build.py`
   from an earlier design; no analysis reads them.

## Draft numbers checked against the reproduced outputs

Lines 108-130, 141-145, 155-173, 185-187, 199-200 and 219-221 of DRAFT5_workshop.md were checked against the reproduced
ANALYSIS files: -0.177 (0.821 vs 0.998, p 0.0002, 0/8 vs 8/8, 3/8 descending), -0.218 (1/8); ring 2/8, 1/8 (Fisher
0.007, 0.0014), 0.9 unmeasured (4/8, -0.046); 3x budget 3/8; MECHANISM 0/8 vs 7/8 (-0.092 [-0.135, -0.045]), +0.109
[+0.085, +0.135], 2/8 (0.007); GENERALITY torus LOST 6/8, re-solves 8/8 (lr 1e-4) and 6/8 (1e-3), from scratch 0/8 (-0.120);
TORUS_START 0/8 vs 1/8 (+0.005 [-0.060, +0.070]), SO4_high 2/8, 1/8 vs 8/8 (Fisher 0.0014, -0.104); GAUGE +1/-1 1.000 8/8 vs
0.931 1/8 (-0.069 [-0.095, -0.045]), +1/+2 0.947 5/8 vs 0.883 2/8 (Fisher 0.31); ROBUST d256L2 2/8 vs 8/8 (0.007), 0.967
vs 0.999 (p 0.0005), 6/8 descending, 5/8 vs 8/8 at loss 0.1; CoPE line 1/16 vs ring 0/16. All match.

## Reproduce

```
cd /home/prashr; export CUDA_VISIBLE_DEVICES="" PYTHONPATH=/home/prashr
# per batch: copy analyze_X.py, point its VERDICTS path elsewhere, then
python3 -u <copy>/analyze_X.py [driver flags] > X_ANALYSIS.txt 2>&1 && diff mapformer/X_ANALYSIS.txt X_ANALYSIS.txt
python3 mapformer/runs/accounts/t3/summarize_t3.py | diff mapformer/runs/accounts/t3/summary.txt -
python3 -m mapformer.eval_noise_refine --runs-dir mapformer/runs/sign_matched --variants Signed_r4 Abs_r4 Pos_r4 RoPE \
  --noises 0.0 --seeds 0 1 2 3 4 5 6 7 --lengths 1024 2048 --n-trials 100 --device cuda:0 --title x --out /tmp/x/SIGN_MATCHED.md
python3 -m mapformer.test_guards
```
