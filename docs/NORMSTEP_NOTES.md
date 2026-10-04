# NormStep: what it is, what is provable, where it applies (2026-10-03)

Context: `LEAK_RESULTS.md` (registered: both remedies REMEDY; in distribution +0.0107 = MapWM's leak, p 0.0002).
Check behind section 2-3: `docs/audits/2026-10-03/normstep_gauge.py` / `_out.txt`.

## Definition
e = token embedding (objects: e = A c, fixed code c, learned encoder A); W = W_out W_in (rank r = 4).
MapWM: Delta(e) = W e. NormStep: Delta(e) = W LN(e), LN(e) = gamma * (e - mean(e)) / std(e) + beta.
Every token still goes through the same learned step map; nothing tells the model which tokens are actions
(unlike ActOnly, the oracle reference). This is MapFormer's premise kept intact.

## 1. Robustness to embedding scale -- a theorem (by construction)
LN(s e) = LN(e) for s > 0 (up to eps), so Delta_NormStep(s e) = Delta_NormStep(e), while Delta_MapWM(s e) = s W e; and
||LN(e)|| <= ||gamma|| sqrt(d) + ||beta|| bounds NormStep's step for any input. Hence x4 codes cost MapWM 0.13 and
NormStep 0. Not evidence of anything learned.

## 2. Why zeroing NormStep's object steps hurts (-0.19): a per-move gauge, checked on the weights
Observation step = c_bar (shared by all observations) + delta(object). With exactly one action and one observation
per move, (a -> a - c_bar, observation -> + c_bar) leaves every per-move phase change, hence every score, unchanged.
NormStep uses it: opposite actions alone cancel to 0.006-0.007 of an action step, and to 0.001 once the two shared
observation steps are added (MapWM: 0.007 -> 0.004). Zeroing observation steps at eval removes c_bar but leaves the
actions' offset, so phase drifts every move. Zeroing is therefore the wrong leak readout for NormStep.

## 3. The true leak (object-identity dependence) is ~5x smaller in NormStep
Spread of delta(object) across objects, relative to an action step: MapWM 0.0038-0.0041, NormStep 0.0007-0.0009
(shared part c_bar: MapWM 0.0012, NormStep 0.0028-0.0035, absorbed). Seeds 0 and 3.

## 4. Not provable: that NormStep must reach a smaller leak
MapWM can reach zero leak too: W has rank 4 in d = 128 (124-dim null space) and A's image is 64-dim, so W A = 0 is
feasible. The gain is an optimisation effect within the budget (MapWM still descending at 900 epochs; r(loss, acc)
-0.94). Mechanism hypothesis: in MapWM the encoder scale serves both the content path and the step, and the cheap way
to shrink the leak is to shrink A (MapWM object embeddings ~12x smaller than RoPE's, 2026-10-02 audit); in NormStep
the scale no longer touches the step, leaving only a direction to learn. Tests: train MapWM longer; penalise ||W A||.

## 5. Scope (predicted; tested only on the new-object task, strict alternation, 1 layer, r=4, 900 epochs)
- Should help: observation tokens with varied/large/drifting embedding norms; strictly alternating action/observation
  streams (c_bar is then a harmless gauge).
- Risk 1, language: with a variable number of tokens per move (text world), the shared beta-step is a per-TOKEN tick,
  i.e. a word-count clock leaking into "where"; the gauge argument no longer applies. Must be learned away (beta into
  W's null space), not guaranteed. Test: NormStep and a bias-free NormStep on the text world (~3 h).
- Risk 2, graded/continuous actions: normalising removes magnitude carried by the input's norm (fine for discrete
  tokens, which keep per-token steps through direction; breaks continuous-magnitude inputs).
- Orthogonal: context ("did not go north") is not addressed -- combine with the context-aware step (HSR).
- Probably little to gain where the leak is already small (paper torus, 16 learned observation types).
