# State changes told in words -- results (2026-10-09)

Pre-registration `TW_STATECHANGE_PREREG.md` (+ Amendment 1 from the independent code audit, before any run; Amendment 2
the pilot, before the batch). Runs `runs/tw_statechange/p0` (32 runs, one batch, seeds 50-57 fresh, n = 8 per arm;
launched 12:21 at a106cfe, clean, on GPU 0 only; done 18:05; md5 guard passed before readouts and analysis). Registered
output `TW_STATECHANGE_ANALYSIS.txt`, verdicts `TW_STATECHANGE_VERDICTS.json`, readouts `TW_STATECHANGE.json` /
`_READOUTS.txt`. Every accuracy below is dropout-scale re-scored (registered); eval-mode values are in the analysis file.

The text world (a torus walk told in English, T = 1024 words, 1 layer) with state-change sentences added: after a move
the agent may pick up the object at the cell or drop one there ("She picked up the cat."), each with p = 0.4. Strata of
the scored revisits: **T1** cell never changed (pure location); **T2drop** first return after a drop -- the answer differs
from the last thing seen there, so it needs the location AND the state sentence; T2take (answered by the constant
'nothing'); T3 later returns.

| arm | step | all | T1 | T2drop | T3 | SOLVED | T = 2048 |
|---|---|---|---|---|---|---|---|
| MapWM | learned, raw | **0.994** | 0.995 | **0.983** | 0.991 | 7/8 | 0.964 |
| NormStep | learned, LN(emb) | 0.961 | 0.970 | 0.979 | 0.970 | 5/8 | 0.918 |
| DirOnly (oracle) | only the 12 direction words | 0.970 | 0.970 | 0.926 | 0.997 | 8/8 | 0.969 |
| RoPE, 1 layer | index | 0.516 | 0.514 | 0.011 | 0.493 | 0/8 | 0.523 |
| floors | | constant 0.514 | constant 0.512, reversal-copy 0.607 | F1 0.211, **F2 (last dropped) 0.649**, stale map 0 | | | |

## Registered
- **A PATH NEEDED FOR LOCATION.** MapWM - RoPE on T1 +0.481 (perm p 0.0002); SOLVED 7/8 vs 0/8 (Fisher p 0.0014). RoPE
  sits within 0.01 of the constant floor on all 8 seeds. The text-world result holds with state sentences in the stream.
- **C STATE BOUND TO PLACE (MapWM and NormStep).** T2drop beats the best location-free rule (F2 0.649) by +0.334 /
  +0.331 (sign-flip p 0.0078 each, the minimum at n = 8). A one-layer path-integrated model learns from prediction alone
  that a dropped object is now at the place where it was dropped, and retrieves it there on return: 0.904-1.000 per seed.
- **B STATE VERBS STAY OFF THE MAP PLANE (both learned-step arms), not distinguished from asides.** A whole state sentence
  moves the position estimate by 0.028 moves (MapWM, mean over seeds; max 0.055) and 0.027 (NormStep; max 0.084):
  OFF on 8/8 and 6/8 (2 NormStep seeds undefined: the two axis steps nearly collinear, |cos| > 0.99). State sentences
  displace no more than aside sentences (state - aside +0.0075, p 0.33; +0.0012, p 0.98).
- **SUMMARY: PREDICTION HOLDS** -- location needs path integration; learned steps keep state verbs off the map plane,
  within the aside range, while the state is used.
- **D1 NormStep - MapWM: NO DIFFERENCE** (-0.033, p 0.23; unmeasured below 0.085). SOLVED 5/8 vs 7/8.
- **D2 DirOnly - MapWM: WORSE** (-0.0235, p 0.0014), with a FLAG: in eval mode without the dropout-scale correction it
  reads NO DIFFERENCE. Replicates TW_NORMSTEP's re-scored -0.024: learned steps beat 'only direction words move'.

## Secondaries (no verdict)
- **Where the oracle loses.** DirOnly is at 1.000 on T1clean (cells with no aside ever told) but 0.970 on T1 and 0.926 on
  T2drop, on every seed (sd 0.004). On T1 this is the aside confusion of TW_NORMSTEP: with no step on any other word, an
  aside's object sits at the place's position. T2drop -0.057 vs MapWM (p 0.0014). The T2drop errors' composition was not
  examined.
- **NormStep's two unconverged seeds.** s55 (training tail 0.45) answers T2drop at 0.984 but T2take at 0.007: it learned
  that a drop puts an object at the place but not that a take removes it. s56 (tail 0.32) is ~0.92 on every stratum. The
  other six are 0.991-1.000. MapWM's one unconverged seed (s56, tail 0.15) is 0.966. NormStep was also bimodal in
  TW_NORMSTEP; at n = 8 the difference from MapWM is unmeasured.
- Functional check (net state-sentence step removed, role offsets kept): no cost (MapWM -0.006, NormStep -0.002; both
  unmeasured below ~0.01). Zeroing every state-word step costs 0.064 / 0.075: the within-sentence offsets are used, the
  net displacement is not.
- Step sizes by word class (relative to a direction half-step), MapWM: direction verbs 0.81, 'saw' 0.55, take/drop verbs
  0.33 / 0.42, objects 0.23, adverbs and fillers 0.02. The state verbs get sizeable steps that cancel within the
  sentence (net 0.02 moves), the same pattern as asides.
- r(final-5% loss, acc) over 32 runs -0.997.

## What it means
- **Language world, beyond the walk.** A one-layer model with path-integrated position, trained only to predict the
  next word, tracks both where the agent is and what has changed at each place. It keeps the sentences that change the
  world from moving its map, to within 0.02-0.03 of a move. Index RoPE at the same depth gets neither (constant floor).
- **What and where, in language.** The change is stored as content at a position, not as a position change. This is
  the factorisation TEM claims, now with state changes. It does not distinguish state sentences from asides: both are
  handled as non-moving sentences with offsets that cancel.
- **Learned steps keep beating the oracle.** Letting only direction words move is worse, not better, because other words
  need small offsets to keep mentioned-but-absent objects off the place.

## Caveats
- Scripted grammar; one task, T = 1024, 1 layer, 900 epochs, n = 8; GPU 0 only (placement does not change the
  computation: the bitwise reproduction passed on cuda:0).
- C's p 0.0078 is the floor of an 8-seed sign-flip test; every seed of both arms is above F2.
- B cannot separate "off the map because the sentence is optional" from "off the map because it is a state change"
  (stated in the prereg); asides behave the same way.
- D2 depends on the registered dropout-scale re-score (flagged).
- The pilot (seed 150) had MapWM still descending at 0.936; the batch has it 7/8 SOLVED. The pilot is not part of any
  number above.
