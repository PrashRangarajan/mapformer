# Retracted / corrected / withdrawn claims (compiled from CLAUDE.md top blocks, RESULTS_INDEX, project_state)
Use this to decide whether a claim in a document is still presented as a finding after it was retracted.

R1  Code modelling OOD (retracted 2026-09-21 by matched-length control runs/code2048): encoding effect -3.694 / PoPE-RoPE -3.585 at 2-4x; MapPoPE-PoPE -0.102 bpc; MapPoPE-MapWM -3.804; "MapPoPE beats both components on code"; "RoPE-encoding arms fall below the no-stack floor"; "the ENCODING is what survives the context"; composition -0.102. At matched length the encoding effect is -0.0030 (MDE 0.0046) and the claim REVERSES (position +0.0055, MapPoPE-PoPE +0.0033 detectable AGAINST path integration).
R2  "Depth substitutes for path integration on Dyck, 4.4x drop" -- withdrawn (width confound: 1L arms d=64, 2L d=128). Superseded by fixed-width ladder (DYCK_LADDER_RESULTS.md).
R3  Dyck 2x2 on the F1 metric as a headline: forbidden; Hewitt position main effect +0.136 not +0.306 (F1 inflates 2.3x); interaction +0.065 not +0.009.
R4  "PoPE's Table 5 does not replicate" / "anomalous NoSigma seed" -- WRONG (stale checkpoint + 40-window val). 3/4 cells agree, 4th unmeasured.
R5  Non-negativity (bound) account of PoPE extrapolation -- REFUTED (NoSigma does not blow up).
R6  Clock/map mechanistic account of the MapPoPE collapse repairs (out-of-range accumulator + kernel) -- WITHDRAWN 2026-09-20 (T2_RESULTS). PoPE-decoupling corollary of clock/map WITHDRAWN.
R7  "MapWM is additive / OR-gate" -- wrong (rotates content Q,K). Thm 3 + corollary withdrawn. "Map tasks tie EM vs WM within 0.004" false at extended length.
R8  "EM never finds the rewind from scratch (0/40)" -- WITHDRAWN (found per query token, wrapped).
R9  "The InEKF's wrap bounds the accumulator" -- FALSE (wraps the innovation).
R10 Level15 decomposition: "token-type gate load-bearing / ConstR worse than nothing" RETRACTED (sign inverted at n=5); "Level15 does not reduce to clamping theta" WITHDRAWN (unmeasured).
R11 Level 1.5 as inference / "measurement-driven"; lm200 wins (+24.8pp, +11pp) -- lm200 all void pre-2026-07; interpretation withdrawn (ExtraHead control ties).
R12 "Rank gap at matched length is training speed" (loss-matched residual +0.002) -- WITHDRAWN 2026-09-23.
R13 MiniWorld "position effect scales with aliasing" -- FALSIFIED, sign inverted.
R14 "Scale HURTS the path-integrated model at long range" (HORIZON) -- RETRACTED; horizon table budget-limited.
R15 "Loop BEATS three real layers by +0.273" -- RETRACTED (matches at n=8).
R16 hier-goal "MapFormer x hierarchy super-additive synergy" -- VOID; planner tasks void; frozen-probe +7.5pp void.
R17 "Distinct cells visited" hypothesis -- falsified.
R18 Match-Query base 0.888 (n=3) -> 0.730 +/- 0.247 (n=5). "No OOD degradation" -> degrades gracefully.
R19 Torus 2x2 encoding "+0.011 / 40x ratio" -> +0.003 main effect, no ratio.
R20 Refine-theta gate "inconsistent sign 4/8" was Match-Query run; action-noise run is 7/9 positive.
R21 SELECTIVE_ROPE per-knob attribution confounded (every single-knob arm deletes path_integrator.omega).
R22 D x r Table 6 geometry account WITHDRAWN; optimisation half rests on one detectable cell.
R23 "Use r=4" general -> MapWM-family only (MapPoPE r4 +0.019 unmeasured).
R24 "no survey covers this" -- wrong (Zhang et al. 2503.17407).
R25 Forget gate transient-aid story REFUTED; mechanism unidentified; forget-gate-as-clock batch DELETED (must re-run).
R26 Separate q0/k0 "refuted" holds only on the 4 map tasks (recency: separate form better, +0.128 n=24; fresh seeds unmeasured).
R27 alpha as independently controllable / "vary alpha" -- malformed; alpha ~ opposition collinear (r=0.9995).
R28 Critical-dimension (low-frequency channels) account imported and REFUTED (LOCALISATION.md).
R29 "Gate is token suppressor" (Selective RoPE) FALSIFIED.
R30 Hierarchy on compositional: +0.13/+0.136 never powered (directional, n=8, unmeasured). Hierarchy on text: null on bpc.
R31 "Level15 +8pp clean OOD T=512 / +11pp noise" etc. early-era numbers -- Level15 effect confined to OOD length, loss-matched +0.062/+0.124 at T=512/1024; no matched-length control ever.
R32 v4 "+3.4pp" -- lm200 retracted; RNG-control byte identical; no surviving v4 win.
R33 DoG hex test (DOG_RESULTS.md) vacuous (all-zero targets).
R34 Indirect Indexing "0.965 replicates paper's 0.948": 0.965 is mean AMONG SOLVERS vs paper all-run mean.
R35 "MapWM extrapolates worse than RoPE on code" (n=1) -- corrected at n=3 (it HELPS at 512-1024).
R36 Code "position main effect zero" -- bucket-specific (detectable -0.552 at 512-1024); code 2x2 NOT additive; "alpha=1.000" rounded mean.
R37 Dyck "The paper's OOD levels sit AT the no-stack n-gram floor 0.884" -- F1 metric floor; not a model failure per se.
R38 "Hierarchy helps compositional transfer 0.415 vs 0.285" -- recipe-limited (COMP_HEADROOM +0.160 from recipe alone); hierarchy +0.136 unmeasured.
R39 "Kalman win is stabilisation + token-type gating" -- token-type gating part retracted (R10).
R40 PAPER2X2 converged recipe: position +0.243 at training length, NOT the +0.461 (old LinearLR recipe, index arm at floor).  [CHECK]
