# RANKDECAY: which parameters must weight decay act on for shared rank 4 to learn the T = 1024 torus? -- pre-registration

Registered 2026-10-24 before any batch run. Origin: RANKWD (`RANKWD_RESULTS.md`, registered): shared rank 4 with AdamW weight
decay 0.05 on ALL parameters solves 8/8, without decay 1/8 (p 0.0014). Its Amendment 1 noted the reading could be credited
only to "decay on everything", and its descriptive table showed the frequency ladder `path_integrator.omega` ending at
~0.15 of its initial size with decay against ~0.96 without -- non-overlapping across seeds, and the parameter decay changes
most among those the map is built from. The user chose this test over G-4 (2026-10-24). Code: `train.py`
(`build_optimizer`: optional AdamW parameter groups), `train_variant.py` (`--wd-select`, `--wd-names`),
`analyze_rankdecay.py`, `run_rankdecay.sh`; checks `test_wd_groups.py` (5/5) and `analyze_rankdecay.py --selftest` (12).
**This file is md5-guarded by the driver: it is final at launch.**

## 1. Question and arms

A 2 x 2: decay on omega (yes / no) x decay on every other parameter (yes / no), at weight decay 0.05 wherever it is on.
Everything else is the RANKWD recipe verbatim (`Vanilla_r4mi`, torus 64 x 64, T = 1024, 900 epochs, lr 1e-3 cosine with 5%
warm-up, 98 batches of 16, 1 layer, 2 heads, d 128, no landmarks, 3 data workers).

| arm | decay on omega | decay on the rest | flags | role |
|---|---|---|---|---|
| all | yes | yes | `--weight-decay 0.05` (single-group AdamW, unchanged path) | anchor = RANKWD's 8/8 arm |
| none | no | no | `--weight-decay 0` | anchor = RANKWD's 1/8 arm |
| **noomega** | no | yes | `--weight-decay 0.05 --wd-select exclude --wd-names path_integrator.omega` | is omega decay NECESSARY? |
| **omega** | yes | no | `--weight-decay 0.05 --wd-select only --wd-names path_integrator.omega` | is omega decay SUFFICIENT? |

At a seed all four arms have bitwise-identical initial weights and data; only the set of decayed parameters differs.
Fresh seeds **2600-2607** (unused anywhere); pilots 2690-2691 and warm starts from seed 1990, never pooled. 32 runs, GPU 1
only, two at a time, ~17 h. Geometry snapshots at init and every 10 epochs.

**The groups are verified to change nothing but the decay (`test_wd_groups.py`, rule 19):** (1) with no `--wd-select` the
code path is the original single-group AdamW and the exact pre-edit command reproduces bitwise (losses and all 24 tensors);
(2) two groups both decayed at 0.05 equal the single group bitwise over 5 real steps, for both "only" and "exclude", so the
grouping itself is neutral; (3) with zero gradients one AdamW step scales EXACTLY the selected parameters by (1 - lr * wd) and
leaves every other parameter bit-identical ("only": 1 decayed / 23 untouched; "exclude": 23 / 1); (4) unknown names, empty
names and unknown modes are refused, and the checkpoint records the setting. The shared suites (`test_epoch_callback`,
`test_rankclock`, `test_signroute`, `test_guards`) give unchanged results after the change.

**What the readings can and cannot say.** "Omega" here is the single learnable frequency vector `path_integrator.omega`
(64 values). A NECESSARY / SUFFICIENT reading is about AdamW's decoupled decay acting on that tensor at 0.05 within 900 epochs.
It would not identify the route by which shrinking omega helps (fewer wraps per cell, a slower phase response that changes
Adam's effective step on the path code, or an interaction with the rest of the map); the norm growth is printed to inform,
not decide.

## 2. Existence before mechanism (rule 14)

A rank-4 solution exists and is found under "all" (RANKWD 8/8; G-1 8/8). Trainable half for the two new settings: the solved
r4 (seed 1990) warm-started for 60 cosine epochs at peak lr 1e-3 under "noomega" and under "omega" (RANKWD already showed it
HOLDS under "all" and "none"). HELD = still SOLVED and frozen accuracy within 0.02 of the start.

Run 2026-10-24 on GPU 1 (`runs/rankdecay_pilot/warm_exclude`, `warm_only`), source the solved r4 pilot (seed 1990, frozen
1.0000), scored frozen with `eval_noise_refine.evaluate` on fresh maps at T = 1024 (60 trials):

| warm start (60 cosine epochs, peak lr 1e-3, wd 0.05 where on) | after | loss first / max / final-5% | class | |
|---|---|---|---|---|
| noomega (decay on all but omega) | 1.0000 (nll 0.0000) | 0.0180 / 0.2636 / 0.0010 | SOLVED | **HELD** |
| omega (decay on omega only) | 1.0000 (nll 0.0000) | 0.0241 / 0.2604 / 0.0070 | SOLVED | **HELD** |

With RANKWD's checks (all: HELD, peak 0.269; none: HELD, peak 0.254) every arm's setting holds a solution: knocked off by
the fresh optimiser and re-found. A failure of any arm in the batch is a failure to FIND the solution. As in RANKWD, the warm
start keeps a solution trained WITH decay at its weight scale, so it does not rule out a norm-growth route.

## 3. Registered analysis (`analyze_rankdecay.py`)

SOLVED = `stats_core.classify_run` (final-5% training loss < 0.05); non-finite loss = DIVERGED, never SOLVED. Each run's recipe,
weight decay, `wd_select`, `wd_names` and len(losses) = 900 are asserted; a missing evaluation row raises.

- **G0** all SOLVED >= 6/8, else UNREADABLE.
- **G1** none SOLVED <= 2/8 and Fisher(all, none) < .05, else NO DECAY EFFECT TO EXPLAIN (RANKWD must replicate in-batch).
- **P1 (omega decay NECESSARY?)** noomega vs all: k_noomega <= 2 and Fisher p < .05 -> OMEGA DECAY NECESSARY;
  k_noomega >= k_all - 1 and mean acc within 0.01 -> OMEGA DECAY NOT NECESSARY; else PARTIAL.
- **P2 (omega decay SUFFICIENT?)** omega vs none: k_omega >= 6 and Fisher p < .05 -> OMEGA DECAY SUFFICIENT;
  k_omega <= 2 -> OMEGA DECAY NOT SUFFICIENT; else PARTIAL.
- **Overall:** NECESSARY + SUFFICIENT -> **OMEGA CARRIES THE DECAY EFFECT**; NOT NECESSARY + NOT SUFFICIENT -> **THE DECAY
  EFFECT IS NOT IN OMEGA**; NECESSARY + NOT SUFFICIENT -> OMEGA DECAY NECESSARY BUT NOT SUFFICIENT; NOT NECESSARY + SUFFICIENT ->
  EITHER ROUTE SUFFICES; any PARTIAL -> PARTIAL.
- **Descriptive:** accuracy at T = 512 / 1024 / 2048 with the unpaired permutation p and the paired sign-flip p beside P1 and
  P2; run classes per seed; norm growth end / init of omega, token_emb, W_in, W_out (snapshots) and the LayerNorm gains
  (checkpoint, init 1); r(log10 final loss, acc) per length (rule 2).

**Power (Fisher, n = 8).** NECESSARY needs noomega <= 2/8 against all: with all at 8/8, p 0.0002 / 0.0014 / 0.0070 for 0 / 1 / 2;
at 6/8, 2/8 gives p 0.13 (-> PARTIAL). SUFFICIENT needs omega >= 6/8 against none: with none at 1/8, 6/8 gives p 0.041, 7/8
p 0.010; with none at 2/8, 6/8 gives p 0.13 (-> PARTIAL). The asymmetry of RANKWD's rule applies: NOT NECESSARY means "equal"
(within one solve and 0.01 accuracy); NECESSARY / SUFFICIENT need a significant difference; the rest reads PARTIAL.

**Budget (rules 3, 4).** "Within 900 epochs". RANKWD's no-decay failures were mostly still DESCENDING; a slow-but-eventual
solution under any arm is outside this batch's claim.

## 4. Predictions

- From RANKWD's descriptive omega shrink: OMEGA CARRIES THE DECAY EFFECT (noomega fails like none, omega-only solves like all).
- The rival -- decay acts through the rest of the network (embeddings, LayerNorm gains, the step map) -- predicts THE DECAY
  EFFECT IS NOT IN OMEGA. Both are registered; neither is assumed.

## 5. Pilots (outside the batch; never pooled)

PILOT_PLACEHOLDER

## Amendment 1 (independent results-blind code check, rule 29)

AMENDMENT_PLACEHOLDER
