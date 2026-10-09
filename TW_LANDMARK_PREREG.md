# Landmarks vs path integration in words -- pre-registration (2026-10-08, before any GPU run)

## Question
In the text world (`TEXTWORLD_RESULTS.md`: path 1L 0.969 vs RoPE 1L 0.505, +0.464, p 0.0002) nothing in the text
identifies a place: position must be integrated from the direction words. Natural text names places (Jericho's room
names; `docs/theory/2026-10-04/04_language.md` condition U, its top proposal). Here the text sometimes NAMES the place
("reached n417"), with per-walk random names, so an observation identifies location directly the way a landmark does.
(1) When names are available, does the path model stop path-integrating -- do names overshadow the map (cue
competition; Zhao & Warren 2015 in humans; `docs/theory/2026-10-05/neuro_positional.md`: landmark correction of grid
phase is the biological counterpart, absent from our integrator)? (2) What happens with names absent at test?
(3) How much of the path advantage over an index model survives when names are there?

## Task (`environment_tw_landmark.py`)
`TextWorld` unchanged (64x64 torus, K=16, p_empty 0.5, the directed walk, 58 words, asides, T = 1024 words) plus a
name pool of 512 single-token names and a mark word: vocabulary 571 at every rate. Per walk, each distinct cell is a
LANDMARK with probability r (one uniform per cell); a landmark gets a name drawn without replacement, fresh per walk,
so names cannot be memorised across sequences and must be bound in context. Every arrival at a landmark carries
`reached <name>` just before the seeing phrase (`walked north then paused reached n417 and saw lamp .`): the name is
always 3 tokens before the object slot, the easiest layout for an index model's induction circuit (the index arm's
best shot at name lookup). Loss: the object word at revisited cells, as in the text world.
Draw discipline: walk and rendering words are drawn from the global RNG exactly as `TextWorld` draws them; names come
from a private RNG seeded by a CRC32 of the walk. So **rate 0 is the text world byte for byte** (tokens, masks, RNG
stream; check C1), every eval condition of a walk is the same walk with only the names changed, rates are coupled
(rate-0.5 landmarks are a subset of rate-1 landmarks with the same names), and one seed sees the same walks at
every rate. Training rates r in {0, 0.5, 1}.

**Why the path arm needs two layers (a design decision; the request said path 1L).** In a one-layer model the
prediction at the query (the last seeing word) is the query token plus a mix of values v(x_s) weighted by
a(x_t, x_s, code(s, t)). Name lookup needs the weight to depend on whether the current name (at t-2) equals the name
before the earlier object (at s-3). Neither x_t nor x_s is a name; RoPE's code(s, t) = t - s ignores identities;
MapWM's code is omega * M * (word counts in (s, t]), which contains the current name's step but never the earlier
name's (it lies before s), so it cannot compare them; and a key at a name carries the name, not the object, as value.
**A one-layer model of either kind is name-blind** (beyond knowing that a cell is a landmark). The question "do names
overshadow the map" is therefore decided by construction at one layer; it needs the path model at two layers (P2),
compared with the index model at two layers (R2, matched depth; 2 layers is also the minimum for an induction
circuit). The one-layer path model (P1) stays as the attribution control: same data, no name route. The pilot checks
the argument (RoPE 1L at r = 1 at the floor, P1's name benefit ~0).

## Gate (rule 11; `docs/audits/2026-10-08/tw_landmark/gate_tw_landmark.py` / `_out.txt`, calls the task code)
Eval stream (the batch's): held-out map 10000, np seed 10**6, 200 walks, scored targets = revisit object slots of the
moves that fit in the rate-1 rendering (4617 targets, 23.1 per walk; identical across every condition and cell).
| training rate r | object slots / walk | revisit fraction | scored revisits whose place is named before |
|---|---|---|---|
| 0 | 131.8 | 0.232 | 0.000 |
| 0.5 | 116.7 | 0.225 | 0.505 |
| 1 | 104.7 | 0.220 | 1.000 |

Floors on the scored targets (best trivial predictor per condition = max of these):
| eval condition | constant | reversal-copy | name lookup (copy the object last seen with the same name) | name lookup, else reversal-copy | word n-gram 1-5 (fit on 1500 training walks, same rendering) |
|---|---|---|---|---|---|
| names stripped / uninformative | 0.508 | **0.603** | 0.508 | 0.603 | 0.508 / 0.508 / <=0.508 / <=0.504 / <=0.447 |
| names at r = 0.5 (consistent) | 0.508 | 0.603 | 0.764 | **0.810** | <= 0.508 |
| names at r = 1 (consistent) | 0.508 | 0.603 | **1.000** | 1.000 | <= 0.508 |
| cue conflict (r = 1) | 0.508 | **0.603** | 0.598 | 0.598 | <= 0.508 |
No answer leak: best constant on landmark vs unnamed targets 0.494 / 0.522 (overall 0.508, same top answer); a
name -> object table fit across 1500 training walks scores 0.5077 = the constant (names carry nothing across walks);
direction words only inside movement clauses (0 exceptions in 200 walks); every id inside the vocabulary.
**Ceiling consequence:** at r = 1 a perfect name-lookup scores 1.000, so the index arm CAN reach the ceiling; I1 has a
"both at ceiling" branch.

## Arms (`train_tw_landmark.py`; models unchanged: `Vanilla_r4` = MapWM r=4 shared, and `RoPE`)
| cell | model | layers | training name rates |
|---|---|---|---|
| P2 | MapWM (path) | 2 | 0, 0.5, 1 |
| R2 | RoPE (index) | 2 | 0, 0.5, 1 |
| P1 | MapWM (path) | 1 | 0, 1 |
8 cells x **6 seeds (50-55, fresh; never used on this task)** = 48 runs, one batch, seed outer, cell inner. One seed has
the same initial weights at every rate (same vocabulary; check C8) and the same walks.
Recipe = the text world's: T = 1024 words, batch 16, 900 epochs x 98 batches, lr 1e-3, warmup + cosine, d 128, 2 heads,
`--data-workers 3`; eval mode for the registered readouts.
- **RoPE 1L is not in the batch**: it is name-blind by the argument above and sat at the constant floor on 8/8 text-world
  seeds (0.505 +/- 0.001); the floor is measured on CPU instead (rule 1) and the pilot runs it once at r = 1 to check.
- **NormStep: not added.** Its registered benefit is the what-to-where leak on UNSEEN objects (`LEAK_RESULTS.md`); here
  every content token (objects, names) is a trained word, and on the text world it was unmeasured (+0.006, MDE 0.096).
  Whether the 512 name tokens leak into the step is measured directly in MapWM (secondary: name step and its
  per-name part relative to a direction step); a large leak would motivate a NormStep follow-up. +3 path cells
  (~5.6 GPU-h) would not answer the question asked.
- **GainScalar score: not added.** Its gain was at rank 2, T = 128 (`GAIN_GRAIN_RESULTS.md`); at rank 4 every arm is at
  1.000. The question here is what the phase integrates when a second retrieval route exists, not the score rule; the
  gain-score x phase interaction is being measured by GAIN_PHASE (running).

## Readouts (`tw_landmark_eval.py`, per run, from the checkpoint, CPU)
Same 200 walks rendered under each condition (r = the run's training rate):
- `own` names at rate r, consistent (in-distribution); `strip` no names; `uninf` names at rate r but every arrival's
  name FRESH (training form, names carry no information; = strip at r = 0); `named` rate 1, consistent; `conf` rate 1,
  and each revisit to a landmark shows, w.p. 0.5, the name of a DIFFERENT landmark visited earlier (cue conflict).
- acc_<cond>: accuracy on the 4617 scored targets. acc_own split by landmark / unnamed cell.
- **Theta reliance** rel_<cond> (path models): acc - acc with every direction word's step replaced by the mean step of
  the 12 direction words, so theta carries no displacement; the per-move common component (clock), aside / role offsets
  and name steps are untouched (they are not path integration; `04_language.md` 2.1). Validated on CPU on this eval
  stream (`reliance_validate.py` / `_out.txt`): committed text-world path models 0.39-0.51 (median 0.492), TW_NORMSTEP
  MapWM 0.35-0.57 (median 0.498), untrained 1- and 2-layer models 0.0000-0.0002; with each direction word mapped to its
  own step the hook changes logits by <= 1e-2 on trained and ~1e-6 on untrained models (float rounding of a
  recomputed step). Secondary rel_all: names and mark also replaced by their mean.
- name benefit = acc_named - acc_uninf (same walks, same form; saturates when the map is perfect).
- conflict: on conflict targets whose true and conflicting objects differ and are both non-blank, share of predictions
  naming the PATH object (true cell) and the NAME object (the other cell); dominance = name share - path share. (With a
  blank on either side, base rates alone made a constant 'nothing' predictor read 0.31 path / 0.52 name in the
  end-to-end test; restricted to two objects a constant predictor reads ~0 / ~0.)
- run class (`stats_core.classify_run`), final-5% loss; step table (omega-scaled: move ratio, opposition minus common
  component, name step / direction step and its per-name part, mark step); drift channels (40 walks, own rendering).

## PRIMARY O: do names overshadow the map? (P2; O1: r = 1 vs r = 0; O2: r = 0.5 vs r = 0)
Two readouts: **rel_uninf** (does the model still use theta when names are present but useless) and **acc_strip**
(names absent at test). Each contrast: exact permutation test (unpaired, n = 6 vs 6) on per-seed values; FIRES if
p < 0.05 AND |d| >= 0.05; 95% permutation CI. Gate: median rel_uninf of P2(r=0) >= 0.2, else
**O UNMEASURED: P2 trained without names does not path-integrate**. Branches (exhaustive):
- both fire downward: **NAMES OVERSHADOW THE MAP** if the r* cell uses names (median name benefit >= 0.05), else
  **MAP LOST, NAMES NOT USED EITHER (a learning failure, not cue competition)**.
- rel_uninf down only: **NAMES OVERSHADOW WHEN PRESENT** (map intact when names are stripped).
- acc_strip down only: **MAP LOST ONLY WHEN NAMES ARE STRIPPED** (form shift; theta still used with names present).
- any upward firing, none downward: **NAMES STRENGTHEN THE MAP**. One up, one down: **MIXED**.
- none fires and both 95% CIs lie above -0.10: **PATH INTEGRATION PERSISTS** (+ "both cells at ceiling" when both
  medians of acc_strip >= 0.98); none fires otherwise: **UNMEASURED (a 95% CI reaches below -0.10)**.
- Qualifier on PERSISTS / STRENGTHEN / UNMEASURED (r* cell, cue conflict): "names dominate" (median dominance > 0) or
  "the path dominates"; at dominance <= -0.8: "names essentially unused by the path model: no cue competition took
  place" (then PERSISTS says only that the path model did not learn the name route at this budget).
- Attribution qualifier on O1's overshadow / map-lost branches, from P1 (name-blind): P1 reliance r=1 - r=0 by the same
  rule: fires downward -> "names ALSO lower 1-layer path integration: not cue competition alone"; else (P1(r=0) median
  reliance >= 0.2) -> "the 1-layer path model keeps its map: names in the stream do not by themselves block step
  learning"; else "P1 qualifier unmeasured".

## PRIMARY I: path vs index at matched depth (P2 - R2)
I0 (r = 0), I05 (r = 0.5), I1 (r = 1) on acc_own; **I05u**: r = 0.5 on acc_own over UNNAMED revisits only (the
prediction: path integration helps where no name exists). Exact permutation test; FIRES if p < 0.05 and |d| >= 0.02.
- **PATH WINS** / **INDEX WINS** -- fires (+ " (BOTH AT CEILING)" if both medians >= 0.98).
- **SOLVED RATE HIGHER / LOWER FOR PATH (convergence; accuracy unmeasured below max(MDE, 0.02))** -- accuracy does not
  fire but Fisher on SOLVED (final-5% loss < 0.05) does.
- **NO DETECTABLE DIFFERENCE** (+ ceiling label) -- otherwise, unmeasured below max(MDE, 0.02).
Holm-adjusted p over the four I tests is printed; a firing that does not survive Holm says so in its label.
**Composite** (no new test): **NAMES CLOSE THE PATH ADVANTAGE** if I0 is PATH WINS and I1 is not;
**PATH ADVANTAGE SURVIVES NAMES** if both are PATH WINS; otherwise NO COMPOSITE.

## Declared secondaries (no verdict)
R2 name benefit per rate (did the index model learn name lookup; the r = 0 cell is never trained on names);
acc_own landmark vs unnamed per cell; paired sign-flip by seed for P2 r* - r = 0 (same init, same walks); step table and
drift per path cell (do names get steps? do names leak into theta?); rel_all; cue-conflict shares P2 vs R2 at r = 1
(which cue wins when they disagree); rel_named and rel_own (reliance with informative names present: cue dominance);
r(final loss, acc) over 48 runs (rule 2); **dropout re-score** (`rescore_hook --scale auto`, the same readouts,
`TW_LANDMARK_RESCORED.json`): verdicts recomputed and any flip flagged. It is a ONE-LAYER correction
(`docs/audits/2026-10-04/DROPOUT_RESCORE.md`: it compounds and lowers multi-layer models), so for P2 / R2 a flip is
reported, not adopted; the registered readouts stay eval mode.

## Void
Any of the 48 runs missing; md5 guard trips (checked at launch, before the readouts, before the analysis); scored
target sets differ across runs; reliance missing on a path run or present on an index run.

## Power (`docs/audits/2026-10-08/tw_landmark/tw_landmark_power.py` / `_out.txt`; registered decision rules)
Per-seed proxy for P2(r=0): the 16 committed one-layer path runs on this eval stream (reliance 0.485 +/- 0.049,
stripped acc 0.972 +/- 0.055; no 2-layer path model has been trained on words -- the pilot replaces this proxy);
RoPE 2L: the 8 text-world runs (0.772 +/- 0.097). Probabilities at n = 6 per cell (n = 8 in brackets):
| scenario (r* cell vs r = 0 cell) | reliance down fires | both fire (overshadow-type) | PERSISTS |
|---|---|---|---|
| null (unchanged) | 0.011 [0.000] | 0.001 [0.000] | 0.76 [0.93] |
| every seed loses the map | 1.00 [1.00] | 1.00 [1.00] | 0 |
| each seed loses it w.p. 0.5 | 0.42 [0.63] | 0.33 [0.55] | 0.01 |
| every seed shifted down 0.10 | 0.85 [0.96] | 0.77 [0.89] | 0.003 |
I0 (path - RoPE 2L at r = 0) fires PATH WINS with probability 0.95 at n = 6 (0.98 at n = 8).
**n = 6** detects complete or uniform overshadowing and can conclude PERSISTS under the null 3 times in 4; overshadowing
of HALF the seeds (seed-bimodal, plausible here) is detected only 1 time in 3 and otherwise reads UNMEASURED. n = 8
would cost +16 runs (~+2.2 h wall); not chosen within the budget, stated as a scope limit.

## Cost (measured where possible; the pilot measures the rest)
Measured: MapWM 1L on the text world 1.5 s/epoch at 2 jobs/GPU, 2.0 s/epoch at 3/GPU (~0.17 GPU-h per 900-epoch run);
RoPE 2L 2.5 s/epoch at 2/GPU (~0.31 GPU-h). MapWM 2L: not measured; taken as RoPE 2L's (pilot measures). Per seed:
6 two-layer runs x 0.31 + 2 one-layer runs x 0.17 = 2.2 GPU-h; **48 runs ~13.3 GPU-h, ~6.7 h wall on the two 4090s**
if GPU-bound at 8 slots (memory may cap 2-layer jobs at 3/GPU; DRV_MINFREE set from the pilot). Readouts on CPU:
~2.7 min per run per pass (8 threads, measured on a loaded CPU), 48 runs x 2 passes in 4 shards ~1.1 h. **Total ~7.5-8 h
wall.** This is above the ~4-6 h asked for; the reasons: the question needs two layers in both main arms (the
one-layer argument), and 6 cells is the minimum for 2 arms x 3 rates at n = 6. Cheaper variants and what they lose:
drop P1 (-2 GPU-h, ~-1 h wall: no attribution qualifier); n = 5 (-2.2 GPU-h: PERSISTS-under-null 0.70, shift power
0.74, paired secondaries cannot reach p < 0.05); drop the r = 0.5 cells (-3.7 GPU-h: no O2, I05, I05u).

## What it can and cannot show
- Can: whether a 2-layer path model trained with named places keeps using its path-integrated phase (reliance with
  useless names; accuracy with names gone), whether names close the path advantage, whether the path still helps on
  unnamed places when half are named, and which cue wins in conflict.
- Cannot: partial (seed-bimodal) overshadowing at n = 6 (reads UNMEASURED); anything about noisy integration -- our
  integrator is exact, so names can only substitute for the map, never correct it (biological landmark correction is
  absent by construction; `neuro_positional.md`); multi-word or ambiguous names; names stated without arrival (Jericho's
  "You are in the kitchen" is this layout). The name route is easier for RoPE (fixed-offset heads) than for MapWM
  (needs a content+recency head, i.e. a clock component in the phase); a PERSISTS with "names essentially unused"
  means the path model did not learn the second route, not that it resisted it.
- Fresh names (`uninf`) keep the training form at r > 0 but are themselves a novelty (a revisit with a never-seen
  name); `strip` removes the form at r = 1 (a shift). O requires both readouts for OVERSHADOW and names the
  one-readout cases separately for this reason.

## Construction checks (`docs/audits/2026-10-08/tw_landmark/tw_landmark_checks.py` / `_out.txt`: ALL PASS)
C1 rate 0 == TextWorld (tokens, masks, RNG state, `generate_batch`, 3 map seeds x 50 walks); C2 every eval condition of
a walk consumes identical draws (asserted in `render_conditions`); C3 named rendering minus name clauses == stripped
prefix; C4 names injective and constant per cell, at object-3 with the mark at -4, fresh names never repeat, a conflict
name belongs to a different earlier landmark and the recorded alt object is that cell's; C5 coupling across rates;
C6 landmark share 0.4967 at r = 0.5 (18058 cells); C7 names redrawn per walk; C8 identical initial weights across rates,
layer counts, reliance hook inert under the identity substitution (|logit diff| ~1e-6), untrained reliance 0.0000;
C9 CPU trainer determinism (losses and weights bitwise, data workers on).
Analysis smoke test over every branch on synthetic data (`analyze_smoke.py` / `_out.txt`: ALL BRANCHES PASS):
overshadow (+ the three attribution outcomes), learning failure, when-present, form shift, strengthen, mixed, gate
failure, unmeasured, persists (+ ceiling, + both conflict qualifiers); I: path wins, index wins, Fisher-only, no
difference, both at ceiling, Holm non-survival; the three composites; the three voids.
End-to-end driver test (CPU, 48 tiny CPU-trained runs, 6 walks): `driver_e2e_out.txt`.

## Reproduction plan
- Bitwise on CPU (done): data stream at rate 0 equals the text world's (C1); trainer losses and weights bitwise across
  two runs (C9); readouts are deterministic (CPU, fixed eval seed).
- Bitwise on GPU: **deferred to the pilot** (CPU cannot check GPU kernels): MapWM 2L r=1 seed 151, 3 epochs, launched
  twice; losses and weights compared bitwise (`run_tw_landmark_pilot.sh`). The text-world batch was GPU-deterministic
  (its seeds 0-1 were byte-identical to the pilot's).
- Not reproducible bitwise against the committed text-world runs: the vocabulary (571 vs 58) changes the
  initialisation, by design (same init across rates).

## Pilot (GPU, before launch; seed 150, outside the batch; `run_tw_landmark_pilot.sh`)
P2 r=0, P2 r=1, R2 r=1, P1 r=1, RoPE 1L r=1, 900 epochs, plus the GPU reproduction check. It must show, before
launch: (a) P2 at r = 0 path-integrates in words at 900 epochs (reliance >= 0.2; else O cannot be asked at this
budget -- redesign, e.g. 1800 epochs); (b) R2 at r = 1 learns name lookup (name benefit clearly > 0; else the name
route does not exist at this budget and the question is moot); (c) RoPE 1L at r = 1 at the floor (<= 0.61) and P1's
name benefit ~0 (the one-layer argument); (d) s/epoch and per-process GPU memory for the cost line and DRV_MINFREE;
(e) the readout pipeline on GPU-trained checkpoints. Pilot results go into an amendment; none is reused.
Then (rule 29) an independent read-only code-verification agent, blind to results; its findings become an amendment
before the batch's results are read.

## Launch (NOT done)
    cd /home/prashr/mapformer && setsid nohup bash run_tw_landmark_pilot.sh > /dev/null 2>&1 &     # pilot first
    cd /home/prashr/mapformer && setsid nohup bash run_tw_landmark.sh > /dev/null 2>&1 &           # the batch
Outputs: `runs/tw_landmark/p0/<arm>_L<layers>_r<rate>_s<seed>`, `TW_LANDMARK.json`, `TW_LANDMARK_RESCORED.json`,
`TW_LANDMARK_ANALYSIS.txt`, `TW_LANDMARK_VERDICTS.json`, marker `.tw_landmark_done`; log `tw_landmark.log`.
