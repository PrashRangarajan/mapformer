# RANKWD: is weight decay necessary for shared rank 4 to learn the T = 1024 torus? -- pre-registration

Registered 2026-10-23 before any batch run. Origin: a POST-HOC observation in G-1 (`RANKCLOCK_RESULTS.md`, "Not registered"):
shared rank 4 at weight decay 0 solved 1/8 against 8/8 at the default 0.05 (Fisher p 0.0014), in an arm that was there to
read a different question (X2). Every rank-4 result in the repo was trained at weight decay 0.05, so if decay is what makes
rank 4 work, the rank story -- and the parking/removal mechanism -- changes. Scope chosen by the user 2026-10-23 ("weight-decay
test only"; the 1800-epoch clock-slot test and G-4 were dropped for now). Code: `analyze_rankwd.py` (imports
`analyze_rankclock`'s loader, geometry and X2 readouts), `run_rankwd.sh`; selftest `analyze_rankwd.py --selftest` (12 checks, incl. the three Amendment 1 cases).
**This file is md5-guarded by the driver: it is final at launch.**

## 1. Question and arms

One factor, two levels, everything else the RANK_MI / RANKCLOCK recipe verbatim (torus 64 x 64, T = 1024, 900 epochs,
lr 1e-3, cosine with 5% warm-up, 98 batches of 16, AdamW, 1 layer, 2 heads, d 128, no landmarks, 3 data workers):

| arm | variant | weight decay |
|---|---|---|
| wd | Vanilla_r4mi (shared rank 4, built from r2's base) | 0.05 (train()'s default; every earlier rank-4 run) |
| wd0 | Vanilla_r4mi | 0 |

At a seed both arms have bitwise-identical initial weights and data (only AdamW's decay term differs; `--weight-decay`
default-preserving, verified for G-1 by the reviewer through the real command line). Fresh seeds **2100-2107** (unused
anywhere; 2000 is avoided because it is an evaluation seed elsewhere). 16 runs, GPU 1 only, two at a time, ~9 h.
Geometry snapshots at init and every 10 epochs (the bitwise-neutral hook, `test_epoch_callback.py`).

**What a DECAY NEEDED reading would and would not say (Amendment 1, finding 1).** `train.py` builds
`AdamW(model.parameters(), weight_decay=...)` with no parameter groups, so decay acts on EVERY parameter -- the step map, the
learnable frequency ladder `path_integrator.omega`, every LayerNorm gain and bias, the embeddings and `out_proj`. Over this
cosine schedule decay alone shrinks a parameter the gradient does not hold to **0.089** of its size (`passive_decay_factor()`,
computed from the schedule). So the reading can be credited only to **"AdamW decay 0.05 on all parameters, at lr 1e-3
cosine, within 900 epochs"** -- not to the step map or the clock. Candidate mechanisms, none tested here: (a) removing an
unread, parked clock direction; (b) capping weight-norm growth, which at fixed lr shrinks Adam's RELATIVE step (rule 15);
(c) pulling omega toward lower frequencies, i.e. fewer wraps (rule 8); (d) shrinking the LayerNorm / output gains, which
sets the logit scale; (e) generic regularisation. The norm growth of W_in, W_out, token_emb, omega and the LayerNorm gains
and the clock geometry are printed (descriptive) to inform, not decide, between them.

## 2. Existence before mechanism (rule 14)

A rank-4 solution exists and is found at wd 0.05 (G-1 8/8; pilot seed 1990 frozen 1.0000). The trainable half for the wd-0
condition: warm-start that solved r4 (seed 1990) for 60 cosine epochs at peak lr 1e-3 with weight decay 0, and with 0.05 as
the control, on GPU 0 (a few minutes, the user's allowance for short jobs). HELD = still SOLVED and frozen accuracy within
0.02 of the start.

Run 2026-10-23 on GPU 0 (`runs/rankwd_pilot/warm_wd0`, `warm_wd005`), source the solved r4 pilot (seed 1990, G-1, frozen
1.0000), scored frozen with `eval_noise_refine.evaluate` on fresh maps at T = 1024 (60 trials):

| warm start (60 cosine epochs, peak lr 1e-3) | after | loss first / max / final-5% | class | |
|---|---|---|---|---|
| weight decay 0 | 1.0000 (nll 0.0000) | 0.0232 / 0.2537 / 0.0006 | SOLVED | **HELD** |
| weight decay 0.05 (control) | 0.9999 (nll 0.0001) | 0.0170 / 0.2686 / 0.0016 | SOLVED | **HELD** |

Without decay a solved rank-4 model is knocked off its solution by the fresh optimiser (peak loss 0.25) and re-finds it, as
with decay. So a wd-0 failure in the batch is a failure to FIND the solution, not to hold one. (Amendment 1, finding 10: the
warm start keeps a solution trained WITH decay, at its small weight scale; wd-0 training reaches larger norms, where mechanism
(b) acts, so this check does not rule (b) out. It is an existence check -- decay is not part of the function class.)

## 3. Registered analysis (`analyze_rankwd.py`)

SOLVED = `stats_core.classify_run` (final-5% training loss < 0.05); non-finite loss = DIVERGED, never SOLVED; each run's
recipe, weight decay and len(losses) = 900 asserted; a missing evaluation row raises (RANKCLOCK's loader).

- **G0** wd SOLVED >= 6/8, else UNREADABLE.
- **P (primary)** k_wd0 <= 2 and Fisher(wd, wd0) < .05 -> **DECAY NEEDED**; k_wd0 >= k_wd - 1 and mean acc_wd0 >= mean
  acc_wd - 0.01 (the house no-cost closeness) -> **DECAY NOT NEEDED**; otherwise **PARTIAL**. Accuracy at T = 1024 with its
  permutation p printed beside.
- **S1 (secondary)** RANKCLOCK's X2 park-vs-remove readout on SOLVED wd0 runs (needs >= 4, else X2 UNREADABLE) and the X2b
  snapshot ordering, exactly as registered there (per-inverse-pair clock u = z_+x + z_-x, wrapped angles, self-normalised to
  each run's e0000: REMOVAL if median |u_end| / |u_init| <= 0.2; PARKING if >= 0.5 with readgain_end / readgain_init <= 0.1;
  DILUTED if u_ratio <= 0.15; else UNRESOLVED). **Limits (Amendment 1, finding 2):** if P reads DECAY NEEDED (k_wd0 <= 2), X2 is
  UNREADABLE by construction and X2b runs on <= 2 runs, where a sign test cannot reach p < .05 (smallest p 0.5) -- so S1 is
  readable only when P misses its prediction, and then on a SELECTED subset of solved runs (rule 5). The wd arm's values are
  printed as a reference that is **non-diagnostic of removal**: passive decay alone gives 0.089, below the 0.2 REMOVAL bar
  (G-1 measured 0.134). Geometry medians over ALL runs per arm are printed too.
- **Descriptive:** accuracy at T = 512 / 1024 / 2048, with the unpaired permutation p AND the paired sign-flip p (the arms
  share init, data stream and evaluation trajectories at a seed; finding 5); run classes per seed; norm growth end / init of
  W_in, W_out, token_emb, omega and the LayerNorm / out_norm gains (RMS, init 1); geometry medians over all runs per arm;
  r(log10 final loss, acc) per length (rule 2).

**Power (Fisher, n = 8; measured, Amendment 1 finding 3).** DECAY NEEDED needs k_wd0 <= 2 AND p < .05:

| k_wd | wd0 0/8 | wd0 1/8 | wd0 2/8 |
|---|---|---|---|
| 8 | 0.0002 | 0.0014 | 0.0070 |
| 7 | 0.0014 | 0.0101 | 0.0406 |
| 6 | 0.0070 | 0.0406 | 0.1319 (-> PARTIAL) |

**Asymmetry (finding 4), stated:** DECAY NOT NEEDED needs k_wd0 >= k_wd - 1 AND mean accuracy within 0.01 of wd's; with one
unsolved wd0 seed that seed must score >= 0.92 / 0.84 / 0.81 (k_wd = 8 / 7 / 6) -- G-1's unsolved wd0 runs averaged ~0.76 --
so in practice NOT NEEDED means "equal", while NEEDED needs a significant difference; everything between reads PARTIAL.

**Budget (rules 3, 4).** "Within 900 epochs"; a slow-but-eventual wd-0 solution is outside this batch's claim. No extension is
registered. (Amendment 1, finding 9: G-1's 6/8 STALLED wd-0 runs are NOT evidence that they would not solve later -- a flat
final window read after the learning rate has decayed is not convergence, rule 3.)

## 4. Predictions

- From G-1's post-hoc 1/8 vs 8/8: DECAY NEEDED. This is the replication being registered.
- No prediction is registered for S1 or the norm growth.

## 5. Pilots (outside every batch; never pooled)

| pilot | arm | status at launch |
|---|---|---|
| 1993 | Vanilla_r4mi, weight decay 0, 900 epochs (GPU 1) | **still training at launch** -- last log line: `Epoch 800/900 | Loss: 0.0004 | LR: 1.30e-04 | 2.0s` |
| 1990 (warm starts) | solved r4 continued 60 epochs at wd 0 / wd 0.05 | both HELD (Sec. 2) |

Seed 1993 was still running when this file was frozen (the user asked to launch once the review was in; this file is
md5-guarded and cannot change after launch). Its final outcome is reported in RANKWD_RESULTS.md, labelled as a pilot. One seed
outside the batch: an anecdote, never pooled, and it changes nothing registered here. Timing: 2.0 s/epoch alone, 4.2 s/epoch
with two sharing GPU 1 -- no throughput gain from packing; ~9 h for 16 runs.

## Amendment 1 (independent results-blind code check, rule 29)

Reviewer read this file, RANKCLOCK_RESULTS.md, analyze_rankwd + the analyze_rankclock code it imports, the driver and its
precedent, train.py / train_variant.py, the r4 model, the evaluator and stats_core, blind to `runs/` and logs; ran both
selftests and a 2-epoch CPU run through the real command line at each weight decay (bitwise-equal e0000 snapshots; saved args
differ only in weight_decay). Found the recipe identical to RANKCLOCK's r4 arms, the layout consistent with the evaluator and
analyzer, the module-global swap exception-safe and on the real loader, G0 and P matching this file, and the Fisher numbers
right. Findings and dispositions:

| # | sev | finding | disposition |
|---|---|---|---|
| 1 | MAJOR | AdamW decays EVERY parameter (omega, LayerNorm gains and biases, embeddings, out_proj); passive shrink 0.089 over the schedule; mechanism list omitted omega and the gains | ACCEPTED: Sec. 1 credits a DECAY NEEDED reading only to "decay 0.05 on all parameters at lr 1e-3 cosine"; mechanisms (c) omega and (d) gains added; omega and LayerNorm-gain growth printed; factor computed from the schedule |
| 2 | MAJOR | S1 unreadable in the predicted outcome (X2 needs >= 4 solved; X2b cannot reach p < .05 on <= 2); the wd reference cannot tell removal from passive decay | ACCEPTED: limits stated in Sec. 3; reference labelled non-diagnostic with the 0.089 factor printed; all-run geometry medians restored |
| 3 | MINOR | power wording assumed wd 8/8 | FIXED: measured table for k_wd = 6 / 7 / 8 |
| 4 | MINOR | the k_wd0 >= k_wd - 1 clause of NOT NEEDED rarely matters; selftest never reached it | FIXED: asymmetry stated; selftest case (wd0 7/8 with its unsolved seed at 0.995) reads DECAY NOT NEEDED |
| 5 | MINOR | accuracy permutation test ignores the pairing | FIXED: stats_core.signflip_p on paired differences printed beside |
| 6 | MINOR | selftest did not check norm growth or wrong-root rejection; "every label" overstated | FIXED: hand-computed W_in ratio checked (0.6793), a wd-0 checkpoint in the wd root raises; docstring scoped |
| 7 | MINOR | ls-files covered 7 files; only e0900 snapshots counted though X2b reads all | FIXED: ls-files over the whole guard list; 16 x 91 = 1456 snapshots required |
| 8 | NOTE | tree not clean (environment_schema.py, run_pope_redo.sh); new files untracked; placeholders must be filled before launch | the user's two changes set aside in a named stash for the batch; files committed; this file final at launch |
| 9 | NOTE | budget wording leaned on STALLED after LR decay | FIXED in Sec. 3 |
| 10 | NOTE | the warm start does not rule out mechanism (b) | recorded in Sec. 2 |
