# Navigation told in words -- pre-registration (2026-09-28, before any run of the batch)

## Question
MapFormer's step is a function of the token, `Delta_w = W_out W_in emb(w)`. On the torus every
action token IS a movement. In language, a few words carry movement among many that do not. Render
the torus walk as English and ask: (1) does path integration still beat index RoPE when the actions
are buried in prose, at the training length; (2) does the learned step table pick out the movement
words by itself -- direction words move the phase, every other word does not, opposite directions
cancel, synonyms share one step?

## Task (`environment_textworld.py`, gated `docs/audits/2026-09-27/gate_textworld.py`)
GridWorld's 64x64 torus walk (K=16, p_empty 0.5, directed walk), each step rendered as
`[verb] [adverb?] [direction] [filler?] [seeing phrase] [object] .`, direction words with three
synonyms each (north/up/northward, south/down/southward, west/left/westward, east/right/eastward),
plus movement-free asides naming objects ("she remembered a cat ."). 58-word vocabulary; T = 1024
words (~132 steps). Loss and readout: the object word at revisited cells; held-out map (seed 10000).
Gate on the held-out map: revisit fraction 0.231 of object slots; floor 0.512 (best constant, and
no word n-gram of order 1-5 beats it). Direction words never appear outside a movement clause
(Delta is context-free, so that case would be decided by construction).

## Pilot (`runs/textworld_pilot`, 2 seeds, NOT reused)
900 epochs, batch 16, lr 1e-3, warmup + cosine, 1 layer: RoPE 0.504 / 0.504 at T=1024 (at the floor,
loss still creeping, 1.64-1.67); path (r=4 shared) 0.998 (SOLVED, loss 0.004) and 0.881 (loss 0.097,
still descending). Step probe: the solved seed -- non-direction words move 0.058x as much as
direction words, opposition 0.083, synonym cosine 1.000; the unsolved seed -- synonym cosine 1.000
but opposition 1.730 and move ratio 0.277 (the non-cancelling basin the rank-2 torus runs stall in).

## Batch (`run_textworld.sh`, `train_textworld.py`; one batch, 8 seeds, all retrained)
Arms: path `Vanilla_r4` 1 layer; index `RoPE` 1 layer; index `RoPE` 2 layers. 24 runs. Recipe as the
pilot: T=1024 words, batch 16, 900 epochs x 98 batches, lr 1e-3, warmup + cosine, d 128, 2 heads,
`--data-workers 3`. Eval: 200 held-out-map trials at T=1024 (registered) and 2048 (extrapolation,
no verdict).

## Readouts and branches
Primary A (accuracy): path 1L - RoPE 1L at T=1024, exact permutation test on per-seed accuracy and
Fisher on SOLVED (final-5% loss < 0.05); FIRES if either p < 0.05.
- **PATH WINS IN WORDS** -- A fires with path above RoPE.
- **UNMEASURED** otherwise.
Primary B (the step table, `probe_textworld.py`, path arm only), per seed:
move ratio (mean ||Delta|| of non-direction words / of direction words), opposition (north x south,
west x east, over synonym pairs; 0 = cancel, 2 = identical), synonym cosine.
- **FINDS THE ACTION WORDS** -- on >= 6/8 seeds: move ratio < 0.2 AND opposition < 0.3 AND synonym
  cosine > 0.9.
- **FINDS SYNONYMS ONLY** -- synonym cosine > 0.9 on >= 6/8, but the full criterion on <= 2/8.
- Anything else: reported as it falls.
Secondary (no verdict): RoPE 2L vs RoPE 1L and vs path; per-seed link between SOLVED and the
probe criterion (Fisher); |cos(N,E)|; the 12 largest-step words per seed; T=2048; r(loss, acc).
Void: any of the 24 runs missing; md5 guard trips.
Scope: one rendering grammar, 58 words, context-free steps, T=1024, 1 layer, r=4 shared, 900 epochs.
Cost: 1.5 s/epoch solo, ~25-35 min per run; 24 runs over 4 slots ~3 h.

---

## Amendment 1 (2026-09-28, 20:15, after an independent audit, BEFORE any batch result was read)

An audit (all checks CPU-only, blind to `runs/textworld/p0` results) found no bug that would make
verdict A or B wrong, and four things that change how B must be read. The registered computations
and branches above are UNCHANGED; the following are added as declared secondaries
(`analyze_textworld_secondary.py`, output `TEXTWORLD_SECONDARY.json`), run after the batch:

1. **The pilot's s0 was misread above.** Its raw opposition 1.730 is not a non-cancelling map. Every
   movement clause holds one verb and one direction word, so a vector can move between the verbs and
   the direction words without changing any object-slot phase (an exact gauge, rule 8). s0 carries a
   per-step clock on both: the common component of the four direction steps is 0.711 of |north| and
   has cosine 1.000 with the mean verb step; with it removed, north+south and west+east cancel
   (0.008 / 0.006; s1 0.003 / 0.003). Verified by us on the pilot weights. Added: S1, opposition and
   |cos NE| of the direction steps minus their common component, |c|/|N|, cos(c, verb step).
2. **The raw step table is gauge-dependent** ((Delta*k, omega/k) is exact). Added: S2, the table on
   Delta*omega; S3, the functional version of "finds the action words": held-out accuracy with Delta
   zeroed for all non-direction words, and for the direction words. (The auditor found zeroing the
   non-direction words costs 0.999 -> 0.903 in pilot s1 and 0.885 -> 0.195 in s0: a small move ratio
   does not mean the other words carry no used phase.)
3. **Synonym cosine > 0.9 barely discriminates**: every exchangeable class collapses (verbs,
   seeing-phrase words, in both pilot seeds). Added: S4, synonym cosine read against the mean |cos|
   between different directions and within the verb and seeing-phrase classes.
4. **The floor.** 0.512 was the gate's own sample; the constant floor on the registered eval set is
   0.505, and a reversal-copy rule (copy the object from two steps back when a move reverses the
   previous one) that word n-grams cannot express scores ~0.597. Added: S5, both floors recomputed on
   the eval set. RoPE's pilot 0.504 sits at the constant floor, below the reversal-copy rule; any RoPE
   "cannot" is budget-scoped (its loss was still creeping).
Also noted: verdict A ORs two tests (family-wise alpha up to ~0.10); `model_rank.py` (defines
`Vanilla_r4`, unchanged since 2026-09-04) is missing from the md5 list; eval seed 0 shares its walk
stream with training batch 0 of seed-0 runs (different map, so no answer leak).

---

## Post-hoc correction (2026-09-30, after the results; changes no registered computation)
- "Pilot ... NOT reused" was false: the pilot's seeds 0 and 1 are byte-identical to the batch's s0/s1.
  Fresh-seed-only numbers are reported in `TEXTWORLD_RESULTS.md`.
- Amendment 1 called the verb/direction shift "an exact gauge". It is a gauge only for a vector moved
  between the two classes with opposite sign; the learned common component is parallel to the verb step
  (cos +1.000), i.e. a per-step clock, not a gauge (`docs/audits/2026-09-27/tw_clock_probe.py`).
