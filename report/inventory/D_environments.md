# D. ENVIRONMENTS -- when does path-integrated position beat an index code, and what decides it

## Overview

The line asks whether the torus headline (path-integrated position +0.461 over an index code,
index arms on the 0.506 floor) is a property of MapFormer or of the torus, and which property of an
environment decides it. It spans the torus knob sweep (five torus-vs-MiniGrid differences turned one
at a time), MiniGrid DoorKey-16x16 factorials, allocentric action recoding (torus H=4, H=12,
MiniGrid), a Habitat build, and a MiniWorld (3D, OneRoom) programme that went through fixed-map,
fresh-map, oracle-recode, grid-sweep, convergence, aliasing-controlled and visits-per-cell stages.

What survived: (1) rotation-based actions are the dominant single property that removes the
position effect on the torus (+0.438 -> +0.050, n=8), and recording the realised absolute
displacement instead of turn/forward restores it (+0.488, n=8); the direction holds at 12 headings
(budget curve non-monotone). (2) On MiniGrid DoorKey-16x16 with commanded actions, index codes are
the best arms and MapWM-Flat is last; with allocentric recoding AND r=4, path integration beats index
8/8 (+0.034, MDE 0.014). (3) On MiniWorld, once both arms converge, path integration wins at grid 32
(+0.178, n=5, detectable) and there is no crossover at grid 8 (-0.010); the "scales with aliasing"
story is falsified with the sign inverted, distinct-cells-visited is falsified, and map extent
(threshold between 128 and 512 occupied cells) survives from a post-hoc pooled analysis at n=3.
Most MiniWorld mechanism claims (reconstruction fidelity as sign-setter, attention horizon,
crossover, hierarchy amplification, fresh-map "liability") were withdrawn.

Environment-level floors (copy alongside any table): torus paper task 0.506 (knob baseline
0.505/0.506; rotate/allocentric 0.508; allcombined 0.552-0.558); torus H=12 0.508-0.509; MiniGrid
DK-16x16 T=128/512/1024 0.642/0.536/0.495 (n=3 batches) and 0.635/0.536/0.490 (n=8 batches);
MiniWorld non-blank marginal 0.077 (fresh, n_obs=16, grid 8), 0.070 (oracle ablation), 0.073-0.079
(grids 16-32), 0.031 (n_obs=64), 0.013 (n_obs=256); chance 1/n_obs; measured MiniWorld run-to-run
noise floor 0.150.

---

### D1 DoorKey-8x8 single-seed series: Vanilla vs Level15, cached buffer, long-T, RoPE diagnostic
- Dates: 2026-05-01 (all four files)
- Question: does Level 1.5 beat Vanilla on DoorKey-8x8 (landmarks key/door/goal), with and without 10% action noise; does RoPE underperform (i.e. does the env exercise path integration); does the gap grow with T?
- Task / environment: MiniGrid-DoorKey-8x8, `obj_color` tokenisation (66 obs types), forward-biased random policy (65% fwd, 30% turn, 5% other); egocentric, commanded turn/forward actions. Eval T=128/512 (in-dist and OOD), long-T to 2048. No floor reported. "OOD" = held-out.
- Arms: Vanilla (MapWM), Level15, RoPE.
- Seeds / batch: n=1 (s0). Live-gym run (MINIGRID_DOORKEY_RESULTS) and 25K cached buffer run (MINIGRID_DOORKEY_CACHED; ~35x faster).
- Validity gates: none (no n-gram gate, no floor, no ablation).
- Result (cached, OOD acc): primary T=512 Vanilla 0.916 / Level15 0.900 / RoPE 0.834; noise T=512 0.789 / 0.887 / 0.812. Long-T primary T=2048 Vanilla 0.823, Level15 0.833, RoPE 0.699; noise T=2048 0.669 / 0.831 / 0.664. Live run primary T=512 OOD Vanilla 0.904, Level15 0.921.
- Status: EXPLORATORY (n=1, no floor).
- Pre-registered? Decision rules stated in-file (RoPE within ~2pp -> not differentiating; >5pp -> exercises PI). RoPE underperformed by >5pp at T=512 primary.
- Caveats: n=1; no measured floor; cached and live differ by ~2pp; commanded rotation actions (later shown to defeat the cumsum, D10/D11).
- Sources: MINIGRID_DOORKEY_RESULTS.md, MINIGRID_DOORKEY_CACHED.md, MINIGRID_DOORKEY_LONGT.md, MINIGRID_DOORKEY_ROPE_DIAG.md
- Bears on: correction (stabilisation under noise), environment (MiniGrid discriminability).

### D2 DoorKey-16x16 single-seed scaling test
- Dates: 2026-05-01
- Question: does Level15's advantage emerge with a 4x larger map?
- Task / environment: MiniGrid-DoorKey-16x16 (256 cells), T=128/512/1024, primary and 10% noise. No floor.
- Arms: Vanilla, Level15, RoPE.
- Seeds / batch: n=1.
- Validity gates: none.
- Result: primary T=512 Vanilla 0.754, Level15 0.844, RoPE 0.877; T=1024 0.608 / 0.795 / 0.757.
- Status: EXPLORATORY; the RoPE-beats-Vanilla inversion was re-measured at n=3 in D3.
- Pre-registered? no.
- Caveats: n=1, no floor. Superseded by D3/D6.
- Sources: MINIGRID_DK16_RESULTS.md
- Bears on: environment.

### D3 MiniGrid 2x2 {RoPE,PoPE} x {index, path-int}, 1 layer, n=3
- Dates: 2026-08-21
- Question: does the torus 2x2 (position decides, encoding irrelevant) survive an egocentric, rotation-actioned external benchmark?
- Task / environment: MiniGrid-DoorKey-16x16, egocentric (cell in front), commanded actions, 25K cached buffer. Measured floor (most common scored target): T=128 0.642, T=512 0.536, T=1024 0.495.
- Arms: MapWM-Flat (RoPE+PI), MapPoPE-Flat (PoPE+PI), RoPE (index), PoPE-Flat (index); n_layers=1.
- Seeds / batch: n=3, one batch, 50 epochs.
- Validity gates: floor measured; no n-gram gate recorded.
- Result: T=512 MapWM-Flat 0.800 +/- 0.065, MapPoPE-Flat 0.944 +/- 0.017, RoPE 0.851 +/- 0.015, PoPE-Flat 0.906 +/- 0.011; T=1024 0.676 / 0.917 / 0.746 / 0.882. Factor means: encoding +0.100 (T=512) / +0.189 (T=1024); position -0.007 / -0.017.
- Status: EXPLORATORY (n=3); superseded by the 3-layer n=8 factorial (D6).
- Pre-registered? yes, in-file prediction that rotation actions make path integration actively misleading: PARTLY REFUTED (MapPoPE-Flat, path-integrated, best at every length).
- Caveats: the torus reference row in this file quotes encoding +0.011, later corrected to +0.003 (BASELINE_TABLE). 1-layer vs hierarchy scaffold confound motivated D4.
- Sources: MINIGRID_2X2.md
- Bears on: environment; encoding vs position.

### D4 MiniGrid 7-cell factorial, 3 layers, n=3
- Dates: 2026-08-22 (file last touched 2026-09-06 for the torus correction)
- Question: does the index>MapWM inversion survive depth; does hierarchy help; how do factors rank?
- Task / environment: as D3; floors 0.642/0.536/0.495.
- Arms: MapWM-Flat, MapWM-Hier, MapPoPE-Flat, MapPoPE-Hier, RoPE-Flat, RoPE-Hier, PoPE-Flat (no PoPE-Hier yet); n_layers=3, hierarchy arms parameter-matched at 614K.
- Seeds / batch: n=3, one batch, 50 epochs, 156 batches.
- Validity gates: floors only.
- Result: T=1024 MapWM-Flat 0.792 +/- 0.077, RoPE-Flat 0.860 +/- 0.022, PoPE-Flat 0.951 +/- 0.004, MapPoPE-Hier 0.948 +/- 0.014. Factor means T=512 / T=1024: encoding +0.038 / +0.085, hierarchy +0.033 / +0.069, position -0.012 / -0.037. "18 of 18 paired hierarchy comparisons positive".
- Status: EXPLORATORY; superseded by D6 (the 18/18 was n=3 luck, 27/32 at n=8).
- Pre-registered? no.
- Caveats: 7 of 8 cells; n=3.
- Sources: MINIGRID_2X2X2.md; N3_AUDIT.md (base-rate table)
- Bears on: environment, hierarchy.

### D5 Frequency control: MapWM-Flat with omega frozen
- Dates: 2026-08-21
- Question: path-integrated arms learn omega (nn.Parameter), index arms use fixed buffers -- is "position effect" confounded with frequency learning?
- Task / environment: MiniGrid-DK-16x16 as D4, floors 0.642/0.536/0.495.
- Arms: MapWM-Flat (614,538 trainable), Vanilla_FixedOmega (614,474 trainable; same init, 64 omega frozen), RoPE-Flat.
- Seeds / batch: n=3, same 3-layer batch/recipe as D4.
- Validity gates: none beyond floor.
- Result: frequency learning (Vanilla - FixedOmega) +0.004 (T=512), -0.008 (T=1024), paired -0.003/+0.013/+0.001 and -0.006/+0.010/-0.028. Pure position (FixedOmega - RoPE) -0.042 / -0.060, negative on 3/3 seeds at both lengths.
- Status: EXPLORATORY (n=3, no MDE); direction of the pure-position contrast 3/3.
- Pre-registered? no.
- Caveats: FixedOmega and RoPE still use different frequency SCHEDULES; learned-omega null tested only on MiniGrid, not the torus. Commanded actions.
- Sources: FREQ_CONTROL.md; BASELINE_TABLE.md sec. H
- Bears on: position-vs-frequency confound for every "position effect" in the repo.

### D6 MiniGrid full 8-cell factorial, 3 layers, n=8 (the published MiniGrid table)
- Dates: 7-cell n=8 batch 2026-08-22; 8th cell (PoPE-Hier) + repro control 2026-08-23
- Question: complete {encoding} x {position} x {hierarchy} factorial at n=8.
- Task / environment: MiniGrid-DK-16x16, egocentric, commanded turn/forward, cached 25K buffer. Floors T=128 0.635, T=512 0.536, T=1024 0.490.
- Arms: MapWM-Flat, MapWM-Hier, MapPoPE-Flat, MapPoPE-Hier, RoPE-Flat, RoPE-Hier, PoPE-Flat, PoPE-Hier; n_layers=3 (parameter-matched).
- Seeds / batch: n=8; 7 cells one batch; PoPE-Hier in a second batch with PoPE-Flat retrained as reproducibility control ("matched its previous n=8 figures exactly (0.963 / 0.953)"). 50 epochs, 156 batches (train_variant default schedule).
- Validity gates: floors measured; repro control exact. No n-gram gate recorded for MiniGrid.
- Result (T=512 / T=1024): PoPE-Hier 0.964 +/- 0.002 / 0.955 +/- 0.003; PoPE-Flat 0.963 +/- 0.002 / 0.953 +/- 0.003; MapPoPE-Hier 0.966 +/- 0.009 / 0.942 +/- 0.017; RoPE-Hier 0.950 +/- 0.004 / 0.924 +/- 0.006; MapPoPE-Flat 0.959 +/- 0.012 / 0.919 +/- 0.026; MapWM-Hier 0.945 +/- 0.014 / 0.893 +/- 0.023; RoPE-Flat 0.914 +/- 0.019 / 0.827 +/- 0.044; MapWM-Flat 0.902 +/- 0.058 / 0.823 +/- 0.088. Main effects (BASELINE_TABLE): T=512 encoding +0.035, hierarchy +0.022, position -0.005; T=1024 encoding +0.076, hierarchy +0.048, position -0.021. Hierarchy gain at T=1024 by pair: RoPE+index +0.096 (8/8), RoPE+PI +0.070 (7/8), PoPE+PI +0.023 (7/8), PoPE+index +0.002 (5/8); 27/32 paired positive overall.
- Status: arm means CITABLE as n=8 levels; main-effect contrasts DIRECTIONAL (no sd/MDE recorded in any source).
- Pre-registered? no.
- Caveats: commanded rotation actions (scope later narrowed by D8: with allocentric + r=4 path integration wins); every arm far above floor (spread 0.823-0.955 at T=1024), little headroom; one MiniGrid env, one tokenisation; the n=8 files carry the eval template header "(n=3)" and "four arms" -- the header is stale. Torus comparison row in these files (RoPE index 0.530 / PI 0.967; PoPE 0.509 / 0.994, floor 0.506) comes from INDEX_BASELINE_PAPER_TASK_n8.md (another line).
- Sources: MINIGRID_2X2X2_n8.md, MINIGRID_FULL_2X2X2.md, BASELINE_TABLE.md (top table and sec. H), N3_AUDIT.md
- Bears on: environment decides the ingredient ranking; hierarchy as compensation.

### D7 MiniGrid allocentric 8-cell factorial, n=3
- Dates: 7-cell 2026-08-25/26; 8th cell + PoPE-Flat repro control 2026-08-26
- Question: does allocentric recoding (realised grid displacement, 5 world-fixed classes) flip the MiniGrid position effect?
- Task / environment: MiniGrid-DK-16x16, egocentric obs, allocentric action record; floors 0.642/0.536/0.495.
- Arms: the 8 cells of D6, n_layers=3, 50 epochs.
- Seeds / batch: n=3; PoPE-Hier trained later alongside a PoPE-Flat repro control.
- Validity gates: repro control PoPE-Flat 0.871 / 0.828 / 0.807 reproduced exactly (MINIGRID_REPRO_CONTROL.md).
- Result (T=128/512/1024): best path-int MapPoPE-Hier 0.877 / 0.840 / 0.825; MapWM-Flat 0.874 / 0.833 / 0.809; PoPE-Hier 0.873 / 0.831 / 0.817; RoPE-Flat 0.873 / 0.811 / 0.781. Corrected 8-cell position effect: raw -0.005 / -0.021 vs allocentric +0.013 / +0.020 (T=512 / T=1024) -- "the FLIP SURVIVES". The ordering claim "all path-int arms outrank all index arms" is FALSIFIED (PoPE-Hier 0.817 > MapWM-Flat 0.809 at T=1024). Absolute scores are lower than raw (best 0.825 vs 0.953).
- Status: EXPLORATORY (n=3, effects ~0.02, no MDE); the sign flip is directional only.
- Pre-registered? no (the 7-cell file predicted parity-to-slight-win and got it).
- Caveats: raw comparison is the n=8 D6 batch vs this n=3 batch -- cross-batch; absolute drop vs raw because allocentric levels a content-solvable task; r=2 throughout. MINIGRID_ALLO_8THCELL.md (untracked) reports the control MISSING and the comparison invalid -- a stale generator output contradicted by the RESOLVED note and MINIGRID_REPRO_CONTROL.md.
- Sources: MINIGRID_ALLOCENTRIC_2X2X2.md, MINIGRID_ALLOCENTRIC_8CELL.md, MINIGRID_REPRO_CONTROL.md, MINIGRID_ALLO_8THCELL.md
- Bears on: rotation mechanism transfer to an external benchmark.

### D8 EM vs WM x rank on allocentric MiniGrid-DK-16x16
- Dates: 2026-09-08
- Question: with rotation handled, does path integration beat index (P1); EM >= WM (P2); r=4 helps EM more (P3); EM failures are collapses (P4)?
- Task / environment: DK-16x16, egocentric obs, allocentric actions, obj_color. Floors T=128 0.635, T=512 0.536, T=1024 0.490.
- Arms: RoPE (index, ~614K), Vanilla (WM r=2, 614,538), Vanilla_r4 (614,922), VanillaEM (r=2, 614,794), VanillaEM_r4 (615,178); spread <0.11%; no --fast-attn.
- Seeds / batch: n=8, one batch, 50 epochs, 156 batches, n_layers=3, 25K cached buffer.
- Validity gates: floors measured; parameter parity; no n-gram gate; convergence/r(loss,acc) listed as pre-registered checks but not reported in the results file.
- Result (T=1024): RoPE 0.788 +/- 0.019 (min 0.754); Vanilla 0.778 +/- 0.045 (min 0.693); Vanilla_r4 0.822 +/- 0.015 (min 0.798); VanillaEM 0.763 +/- 0.081 (min 0.572); VanillaEM_r4 0.803 +/- 0.056 (min 0.666). Contrasts: Vanilla - RoPE -0.010 (MDE 0.056, 4/8, unmeasured); **Vanilla_r4 - RoPE +0.034 (MDE 0.014, 8/8, DETECTABLE)**; VanillaEM - RoPE -0.026 (MDE 0.087, 3/8); VanillaEM_r4 - RoPE +0.015 (MDE 0.062, 7/8). EM - WM: r=2 -0.020 (T=512, 3/8) / -0.016 (T=1024, 4/8); r=4 -0.018 (2/8) / -0.019 (4/8), all unmeasured. Rank x backbone interaction -0.003 (MDE 0.128). Worst-to-2nd-worst gap: WM r=4 0.007, EM r=4 0.144, WM r=2 0.031, EM r=2 0.167; symmetric trimming gives EM - WM -0.001 (r=2), -0.003 (r=4).
- Status: CITABLE for Vanilla_r4 - RoPE; all EM contrasts unmeasured (DIRECTIONAL/EXPLORATORY).
- Pre-registered? yes (MINIGRID_EM_PREREG.md). P1 CONFIRMED only with r=4; P2 REFUTED as stated (all four point estimates negative, unmeasured); P3 unmeasured; P4 CONFIRMED descriptively (one collapsed EM seed).
- Caveats: one env, one tokenisation, flat models only; trimming analysis is post-hoc descriptive; results file says "App." references are the pre-registration's.
- Sources: MINIGRID_EM_PREREG.md, MINIGRID_EM.md
- Bears on: environment (MiniGrid scope narrowing), rank (r=4), EM vs WM.

### D9 EM collapse fix: shared p0 x r=4 on allocentric MiniGrid
- Dates: prereg 2026-09-08; results 2026-09-11
- Question: is EM's one-seed collapse the separate q0/k0 origin (unpeaked A_P) or the r=2 basis?
- Task / environment: as D8; floor T=1024 0.490.
- Arms: Vanilla_r4 (reference), VanillaEM, VanillaEM_r4, VanillaEM_P0, VanillaEM_P0_r4 (new); parameter spread 0.08%.
- Seeds / batch: n=8, one batch, same recipe as D8.
- Validity gates: parameter parity; reading rule fixed in advance (min and gap before means).
- Result (T=1024): min / 2nd / gap / sd -- Vanilla_r4 0.798 / 0.814 / 0.016 / 0.014; VanillaEM 0.602 / 0.739 / 0.137 / 0.073; VanillaEM_r4 0.712 / 0.810 / 0.098 / 0.040; VanillaEM_P0 0.797 / 0.813 / 0.016 / 0.014; VanillaEM_P0_r4 0.816 / 0.818 / 0.002 / 0.011. EM_P0_r4 - EM_r4 +0.0139 (T=512), +0.0212 (T=1024; predicted +0.019; MDE 0.044, unmeasured). Shared p0 buys +0.050 at r=2, +0.021 at r=4; interaction -0.029 (MDE 0.104). EM_P0_r4 - Vanilla_r4 +0.0012 (4/8) / +0.0035 (6/8), unmeasured. VanillaEM_P0_r4 means 0.847 / 0.830.
- Status: DIRECTIONAL (every contrast unmeasured; the gap ordering is descriptive).
- Pre-registered? yes (MINIGRID_EM_FIX_PREREG.md). P1 held (gap 0.002 < 0.03); P2 point estimate on prediction, unmeasured; P3 direction as predicted, unmeasured.
- Caveats: "EM matches WM" is an unmeasured contrast, not a powered equivalence; not a universal fix (torus vocab n_obs=256 VanillaEM_P0 still had a collapsed seed 0.910/0.502/0.906 at r=2); flat models, 50 epochs, one env. File cites paper "App. A.4" for separate q0/k0; CLAUDE.md corrects the location to App. A.7.
- Sources: MINIGRID_EM_FIX_PREREG.md, MINIGRID_EM_FIX.md
- Bears on: EM vs WM; separate-q0/k0 refutation on map tasks.

### D10 Knob sweep, n=3 (single-knob environment properties on the torus)
- Dates: 2026-08-20/21
- Question: which of five torus-vs-MiniGrid properties (rotate, ego, wall, small, richobs) removes the position effect?
- Task / environment: torus paper task code, knobs turned one at a time. Floors per condition: baseline 0.505, rotate 0.504 (void run) / 0.508 (redo), ego 0.508, wall 0.502, small 0.502, richobs 0.505, allcombined 0.558.
- Arms: Vanilla (PI) vs RoPE (index), same batch per condition.
- Seeds / batch: n=3, 16 epochs, 98 batches, batch 128, T=128, 1 layer, 2 heads, d=128. Rotate redo: standard budget and matched budget (392 batches, 4x).
- Validity gates: answer-stream n-gram gate run AFTER training (200 episodes): rotate VOID (o1 0.932, o3 0.913 vs marginal 0.508); allcombined caveat (o3 0.634 vs 0.536 marginal; both arms exceed it); others clean. Rotate redo gated BEFORE training: o1-5 0.501/0.472/0.440/0.462 vs 0.507 marginal (PASS), using revisit keyed on obs-determining state plus `--score-moves-only` (scored rate 0.727 -> 0.054; fix 1 alone left o1 at 0.899).
- Result: position effect baseline +0.478; ego +0.265; wall +0.251; small +0.324; richobs +0.299; allcombined -0.084. Rotate standard budget +0.004 (both arms on floor: 0.512 / 0.508); matched budget Vanilla 0.557 +/- 0.023 vs RoPE 0.508 +/- 0.007, +0.049 (paired +0.063/+0.053/+0.031). Reduction from baseline: rotate -0.429, small -0.154, richobs -0.179, ego -0.213, wall -0.227.
- Status: EXPLORATORY for the four n=3-only knobs (wall, ego, richobs, small); baseline/rotate/allcombined superseded by D11 at n=8.
- Pre-registered? yes, in-file: "aliasing and size drive it, embodiment knobs do not" -- REFUTED. The intermediate "every knob contributes about equally" conclusion was WITHDRAWN once rotate was redone.
- Caveats: 16-epoch 1-layer budget (the rule-5 budget: rotate itself moved +0.004 -> +0.050); small and richobs each carry one unstable Vanilla seed (small +0.038/+0.526/+0.408; richobs +0.085/+0.410/+0.403); decomposition not additive; "twice the next knob" margin rests on n=3 runners-up (N3_AUDIT #4) -- cite dominance, not the multiple. The "allcombined -0.084 lands on MiniGrid -0.060" comparison uses FREQ_CONTROL's 3-layer 50-epoch pure-position T=1024 number, a different recipe.
- Sources: KNOB_SWEEP.md, N3_AUDIT.md, BASELINE_TABLE.md sec. I
- Bears on: environment (rotation actions), sign/fixed-per-token delta mechanism.

### D11 Knob sweep n=8 and allocentric action recoding (H=4)
- Dates: n=3 allocentric 2026-08-20; n=8 table 2026-08-22; ALLOCENTRIC_RECODING.md last edited 2026-09-07
- Question: is rotate's collapse a representation mismatch? Record the realised absolute displacement (or STAY) instead of turn/forward; dynamics identical.
- Task / environment: torus, rotate mode, 4 headings, score-moves-only. Floors: baseline 0.506, allcombined 0.552, rotate 0.508, allocentric 0.508.
- Arms: Vanilla vs RoPE. Allocentric adds one STAY token (vocab 21 -> 22).
- Seeds / batch: n=3 (ALLOCENTRIC_RECODING.md, matched budget 392 batches) and n=8 (KNOB_SWEEP_n8.md; the n=8 file does not restate recipe -- assumed the KNOB_SWEEP recipe with rotate/allocentric at matched budget).
- Validity gates: answer-stream gates identical to three decimals under both records (o1 0.501 / o2 0.472 / o3 0.440 / o5 0.462 vs 0.507 marginal).
- Result n=8: baseline Vanilla 0.967 +/- 0.039, RoPE 0.529 +/- 0.044, +0.438; allcombined 0.708 +/- 0.068 vs 0.785 +/- 0.022, -0.076; rotate 0.558 +/- 0.026 vs 0.508 +/- 0.006, +0.050; **allocentric 0.996 +/- 0.005 vs 0.508 +/- 0.006, +0.488**. n=3: rotate commanded +0.049; allocentric 0.994 +/- 0.008 vs 0.508 +/- 0.007, +0.485 (paired +0.483/+0.490/+0.484). BASELINE_TABLE: rotate accounts for -0.388 of the -0.438 swing.
- Status: CITABLE (n=8, effect ~0.49 with index at the floor; no MDE printed, but sds 0.005-0.044).
- Pre-registered? The n=3 KNOB_SWEEP "Still open" section predicted recovery if the mis-specification account holds; confirmed.
- Caveats: index arm at the floor in both records, so the effect is about PI recovery, not index; STAY token breaks exact embedding parity; discrete 4-direction displacement only; ALLOCENTRIC_RECODING.md's "Limits: n=3" is stale (n=8 exists). Mechanism (fixed per-token delta cannot represent heading-dependent displacement) established by intervention on the record.
- Sources: KNOB_SWEEP_n8.md, ALLOCENTRIC_RECODING.md, KNOB_SWEEP.md, BASELINE_TABLE.md sec. I
- Bears on: environment; the MapFormer path integrator's well-specification condition; Habitat/MiniGrid remedy.

### D12 H=12 headings: continuous position, actuation noise, and the budget curve
- Dates: CONTINUOUS_ALLOC 2026-08-20 (corrections 2026-08-23); H12_BUDGET_CURVE 2026-08-23
- Question: does allocentric recoding survive Habitat's 12 headings (real-valued position) and actuation noise; was partial recovery undertraining?
- Task / environment: torus GridWorld, action_mode rotate, n_headings=12, score_moves_only, held-out env seed 10000, T=128, 32 eval batches x 64. Floor 0.508-0.509. Scored rate 0.022 (vs 0.225 baseline).
- Arms: Vanilla vs RoPE; conditions commanded / allocentric / allocnoise (0.15 rad Gaussian on each executed turn).
- Seeds / batch: n=3; 16 epochs, linear decay; budgets 980, 2000, 4000 batches (one batch per budget).
- Validity gates: floor measured; r(final loss, acc) = -0.996 over 18 runs (Spearman -0.953).
- Result: at 980 batches commanded +0.110, allocentric +0.263 (Vanilla 0.772 +/- 0.099), allocnoise +0.230 (noise cost -0.033). Budget curve (allocentric): 980 Vanilla 0.772 +/- 0.101 / RoPE 0.508 +/- 0.006 / +0.264; 2000 0.891 +/- 0.005 / 0.508 +/- 0.006 / +0.383; 4000 0.837 +/- 0.060 / 0.551 +/- 0.007 / +0.286. Per seed Vanilla 980: 0.661/0.798/0.858 (loss 1.357/0.881/0.629); 2000: 0.885/0.893/0.894 (0.552/0.527/0.507); 4000: 0.807/0.799/0.906 (0.834/0.815/0.422). RoPE 4000: 0.542/0.555/0.555. Weakest PI seed anywhere 0.661 vs strongest index seed 0.555.
- Status: DIRECTION CITABLE as an existence statement (every PI seed above every index seed at every budget, n=3 x 3 budgets); magnitude EXPLORATORY (n=3, bimodal, non-monotone).
- Pre-registered? no. "Partial recovery at H=12" WITHDRAWN; "recovers once budget is adequate / still climbing" FALSIFIED by nb=4000.
- Caveats: bimodal basin selection unexplained (more steps at high LR untested); index arm leaves floor at nb=4000; allocnoise -0.033 measured only at the undertrained 980 budget; direction quantised with fixed magnitude -- does not model Habitat's continuous-magnitude slides (D13); N3_AUDIT #6 and BASELINE_TABLE say more seeds needed.
- Sources: CONTINUOUS_ALLOC.md, H12_BUDGET_CURVE.md, BASELINE_TABLE.md coverage gaps, N3_AUDIT.md
- Bears on: allocentric remedy under finer heading quantisation; rules 5/9.

### D13 Habitat build and simulator measurements (no model trained)
- Dates: 2026-08-19/23
- Question: can the allocentric setup be ported to Habitat, and what does the real simulator do?
- Task / environment: habitat-sim 0.3.3 headless, py3.9 conda env; habitat_test_scenes (van-gogh-room 9 m^2, apartment_1 53 m^2, skokloster-castle 227 m^2).
- Arms: none.
- Seeds / batch: n/a; 361 forward attempts per scene.
- Validity gates: 8 unit tests PASS -- turn exactly 30 deg with zero displacement; forward exactly 0.25 m; displacement depends on accumulated heading; 12x30 deg closes the circle; navmesh loads (9.2 / 52.9 / 226.7 m^2); walls block 12.7-28.8% of forward attempts; renders vary with position.
- Result: library default turn is 10.0 deg vs PointNav spec 30.0 deg (adapter must set it). Forward moves producing exact 0.25 m: 19.7% / 30.7% / 8.6%; partial slide 51.5% / 48.5% / 78.7%; fully blocked 28.8% / 20.8% / 12.7%. So 69-91% of forward moves do not produce the commanded displacement.
- Status: EXPLORATORY (engineering record of measured simulator properties; no learning result).
- Pre-registered? no.
- Caveats: decision NOT to port in this framing (continuous-magnitude displacement; published Habitat numbers are RL recurrent policies; ~130k tokens/episode at 256 image tokens/frame). Observation tokenisation unresolved. Realistic PointNav without GPS+Compass is the only setting with headroom (Partsey et al. 2022).
- Sources: HABITAT_BUILD.md, CLAUDE.md 2026-08-19/20 section, PERCEPTION_EXPERIMENT_PLAN.md
- Bears on: environment scope of the allocentric remedy.

### D14 MiniWorld fixed-map factorial {PI, index} x {raw, allocentric}
- Dates: 2026-08-25
- Question: does the MiniGrid allocentric flip reproduce in continuous 3D, on a known (fixed per seed) map?
- Task / environment: MiniWorld-OneRoom-v0 used as a geometry generator with location-keyed aliased tokens (not RGB); grid 8; n_obs=16; p_empty 0.5; cross-cell-revisit scoring; non-blank accuracy, chance 0.0625, oracle 1.0; non-blank marginal 0.150 (fixed-map gate). T=512 and T=1024; fixed obs_map, novel walks. Raw 3-action macros vs allocentric 24/25-bin displacement direction.
- Arms: Vanilla, MapPoPE-Flat (PI); RoPE, PoPE-Flat (index). d=256, 4 layers, 3.18M params.
- Seeds / batch: n=3, 24 arms, 100 epochs, all converged (train loss < 0.16).
- Validity gates: MINIWORLD_GATES_FIXED (raw, grids 8 and 6) all PASS: G4 action n-gram best 0.427 vs marginal 0.419, non-blank best 0.155 vs 0.150; G6 median revisit lag 49; G7 oracle 1.0000. MINIWORLD_GATES_ALLO (allo stream): non-blank n-gram 0.140 < marginal 0.150, accuracy decreases with order. (GATES_ALLO header still reads "allocentric=False" -- template text.)
- Result: T=512 raw Vanilla 0.653, MapPoPE-Flat 0.662, RoPE 0.620, PoPE-Flat 0.655; allocentric 0.801 / 0.819 / 0.798 / 0.807. Position effect paired: raw +0.020 +/- 0.013, allocentric +0.008 +/- 0.009 (T=512); raw +0.010 +/- 0.017, allocentric -0.021 +/- 0.013 (T=1024). Allocentric raises all arms by ~+0.15.
- Status: EXPLORATORY; position effects are below the measured 0.150 noise floor (unmeasured, not null).
- Pre-registered? no.
- Caveats: fixed map deliberately dilutes the need for path integration; n=3; the T=1024 comparison is out of training length for index arms. The "attention path-integrates, the SO(2) code is an inductive bias" interpretation is a reading, not tested here.
- Sources: MINIWORLD_FIXED_RESULTS.md, MINIWORLD_FIXED_RESULTS_T1024.md, MINIWORLD_FIXED_FINDINGS.md, MINIWORLD_GATES_FIXED.md, MINIWORLD_GATES_ALLO.md, MINIWORLD_TODO.md
- Bears on: environment (allocentric recoding as better input vs PI fix).

### D15 MiniWorld fresh-map factorial {PI, index} x {raw, allocentric}, 100 epochs
- Dates: 2026-08-26
- Question: in the in-context regime (new map each episode), does path integration become load-bearing and does allocentric recoding flip it?
- Task / environment: OneRoom, grid 8, n_obs=16, fresh obs_map per episode, held-out new map. Chance 0.0625; non-blank marginal 0.077 (fresh gates), 0.076 in ablation table.
- Arms: Vanilla, MapPoPE-Flat, RoPE, PoPE-Flat; d=256, 4 layers.
- Seeds / batch: n=3, 24 arms one batch, 100 epochs (LinearLR default of train_miniworld).
- Validity gates: fresh-map n-gram gates PASS both encodings (raw non-blank best 0.050, allo 0.059 vs 0.077); G7 oracle 1.0000 (after fixing a stale-obs_map validator bug, KNOWN_BUGS.md). Context destruction PASS on all 24 arms (intact 0.168-0.512 -> obs-shuffle 0.007-0.088, action-shuffle 0.016-0.064, marginal 0.076). Aggregator flagged two arms loss>1.5 (Vanilla_raw_s2 1.53, MapPoPE-Flat_allo_s0 1.60); findings say they plateaued.
- Result: T=512 raw Vanilla 0.303, MapPoPE-Flat 0.308, RoPE 0.398, PoPE-Flat 0.384; allo 0.284 / 0.232 / 0.501 / 0.364. Position effect raw -0.086 +/- 0.021, allo -0.174 +/- 0.035 (T=512); raw -0.051 +/- 0.020, allo -0.184 +/- 0.034 (T=1024). 40-epoch probe (Vanilla-raw 0.093) was undertraining (0.356 at 100 epochs).
- Status: EXPLORATORY. The "path integration is a LIABILITY" reading must not be cited: it was never re-run under the warmup+cosine budget that converged both arms, and the parallel grid-8 oracle anchor at this same recipe (-0.529) collapsed to -0.010 once converged (D19); CROSSOVER_CONVERGED states there is no regime where index is genuinely better.
- Pre-registered? Validity guardrails F1-F5 (MINIWORLD_TODO.md) pre-specified; hypothesis (flip) not supported.
- Caveats: aggregator's verdict "INCOMPLETE / SUSPECT -- verdict withheld"; results tables carry a stale "fixed-map" title; n=3.
- Sources: MINIWORLD_FRESH_RESULTS.md, MINIWORLD_FRESH_RESULTS_T1024.md, MINIWORLD_FRESH_ABLATION.md, MINIWORLD_FRESH_GATES_RAW.md, MINIWORLD_FRESH_GATES_ALLO.md, MINIWORLD_FRESH_FINDINGS.md, MINIWORLD_TODO.md
- Bears on: environment; validity methodology for in-context map tasks.

### D16 MiniWorld oracle exact-cell recode vs 24-bin allocentric (fidelity test)
- Dates: 2026-08-26 (correction 2026-08-27)
- Question: if the action token determines displacement exactly (R^2 -> 1), does path integration flip positive?
- Task / environment: fresh-map OneRoom grid 8, n_obs=16; oracle recode emits exact integer cell transition (clamp rate 0). Non-blank marginal 0.070 (oracle ablation). Forensic R^2 of per-step displacement from token id (200 trajectories): torus 1.0000, MiniGrid allo 0.9994, MiniWorld allo 0.5506, MiniWorld raw 0.0000; MiniWorld allo forward-step magnitude CV 0.49.
- Arms: Vanilla, MapPoPE-Flat, RoPE, PoPE-Flat.
- Seeds / batch: n=3, one batch with the allocentric arms, 100 epochs.
- Validity gates: oracle gates PASS (non-blank n-gram best 0.049 vs 0.077); context destruction PASS all 24 (RoPE oracle 0.982/0.981/0.969 -> obs-shuffle 0.083/0.080/0.087).
- Result: T=512 allo -> oracle: Vanilla 0.284 -> 0.448 (per seed +0.260, +0.132, +0.099), MapPoPE-Flat 0.232 -> 0.324 (+0.176, +0.071, +0.030), RoPE 0.501 -> 0.977 (+0.471, +0.496, +0.463), PoPE-Flat 0.364 -> 0.938 (+0.585, +0.596, +0.542). Position effect allo -0.174 -> oracle -0.571 +/- 0.042 (T=512); -0.184 -> -0.409 +/- 0.072 (T=1024).
- Status: EXPLORATORY (n=3). Surviving narrow claim (in-file correction): exact integrand improves PI by +0.164 within batch, 3/3 seeds, but does not set the sign.
- Pre-registered? yes, in MINIWORLD_FRESH_FINDINGS ("oracle recode -> PI flips positive"): the ordering prediction FAILED. "H1 REFUTED" was then itself CORRECTED as overshooting.
- Caveats: index arms hit ceiling (0.977/0.938) so -0.571 is not a mechanism size; 100-epoch recipe, grid-8 Vanilla later shown unconverged (loss 0.93-1.14; D19), so between-arm numbers are superseded; the R^2-vs-effect correlation is confounded.
- Sources: MINIWORLD_ORACLE_RESULTS_T512.md, MINIWORLD_ORACLE_RESULTS_T1024.md, MINIWORLD_ORACLE_ABLATION.md, MINIWORLD_ORACLE_GATES.md, MINIWORLD_FRESH_FINDINGS.md
- Bears on: integrand fidelity as a driver of PI performance.

### D17 MiniWorld run-to-run noise floor (GateDeltaCtl)
- Dates: 2026-08-27
- Question: what is the seed-to-seed variance of this setup? (GateDeltaCtl = GateDelta params 3,206,682 with gate multiplied out: max|diff| 0.00e+00 vs Vanilla, zero gate gradient -- a second Vanilla seed.)
- Task / environment: fresh-map oracle recode, grids 16/24/32, T=512 and T=1024.
- Arms: Vanilla, GateDeltaCtl, GateDelta.
- Seeds / batch: n=3 per grid (9 pairs), 100 epochs.
- Validity gates: function identity verified.
- Result: Ctl vs Vanilla mean|delta| 0.150 (T=512) / 0.163 (T=1024), sd 0.198 / 0.230, range -0.228..+0.410. GateDelta - Ctl pooled +0.081 (T=512) / +0.131 (T=1024); all-converged n=1 rows are not numbers.
- Status: CITABLE as a method measurement (the noise floor used for every MiniWorld claim); the GateDelta effect is unmeasured.
- Pre-registered? no.
- Caveats: measured on mostly unconverged 100-epoch runs (only 1 of 9 triples all converged); converged conditions later show much smaller sds (e.g. 0.031-0.094 in D21), yet 0.150 continued to be applied as the floor there.
- Sources: MINIWORLD_GATE_CONTROL.md, CLAUDE.md rule 8
- Bears on: every MiniWorld effect; rule 8.

### D18 MiniWorld Selective-RoPE components (ConvDelta, GateDelta) and NoPE probe
- Dates: 2026-08-27
- Question: do SRoPE's conv1d-before-cumsum or sigmoid gate help MapFormer on navigation; is index-RoPE a straw man (NoPE)?
- Task / environment: fresh-map oracle recode, grids 8 (probe only)/16/24/32; non-blank marginal 0.070-0.079 (not 1/16).
- Arms: Vanilla, RoPE, NoPE (probe), ConvDelta, GateDelta.
- Seeds / batch: n=3 for components (9 grid-seed pairs), n=1 probe (NoPE seed 0 only); 100 epochs.
- Validity gates: convergence flags per run.
- Result: T=512 pooled ConvDelta - Vanilla +0.035, GateDelta - Vanilla +0.053; T=1024 +0.040 / +0.120. Paired-delta sd 0.176 / 0.203, MDE 0.165 / 0.190 (n=9). Probe: NoPE 0.045 (grid 8) to 0.114 (grid 32) at T=512, loss still descending (1.693@80 -> 1.680@90 -> 1.662@100 at g32).
- Status: DIRECTIONAL/unmeasured (explicitly "UNMEASURED, not null"); NoPE probe EXPLORATORY (n=1).
- Pre-registered? no.
- Caveats: all effects inside 0.150 floor; "NoPE collapses to chance" and "NoPE < RoPE so RoPE is not a straw man" were CORRECTED (non-sequitur; RoPE base 10000 never tuned).
- Sources: MINIWORLD_SROPE_COMPONENTS.md, MINIWORLD_PROBE3.md
- Bears on: Selective RoPE line (cross-domain), index-baseline strength.

### D19 Converged position effect at grid 32 and the withdrawn crossover (grid 8)
- Dates: ROPE_CONVERGE / POSITION_EFFECT_CONVERGED / CROSSOVER_CONVERGED 2026-08-28
- Question: at 100 epochs linear decay RoPE never converged at grid >= 16 (0/9) and Vanilla never at grid 8 (0/3) -- is the grid-size crossover representation or optimisation?
- Task / environment: fresh-map oracle recode, n_obs=16; grid 32 (32 cells/token) and grid 8 (2 cells/token); T=512 in-distribution (T=1024 is 2048 tokens, out of training length for index RoPE).
- Arms: Vanilla vs RoPE.
- Seeds / batch: n=3 each grid; 400 epochs, 5% warmup + cosine. Grid-32 Vanilla trained in a later batch than RoPE with RoPE s0 repro control (0.725 vs stored 0.725, drift +0.000).
- Validity gates: convergence (slope over final 10%): grid 32 6/6 flat (slopes -0.0001 to -0.0004/ep); grid 8 6/6 flat; repro exact.
- Result: ROPE_CONVERGE: RoPE mean final loss 0.4460, nb_acc 0.754 (0.725/0.789/0.748), vs 0.615 at 100 epochs. Grid 32: Vanilla 1.000 / 1.000 / 0.781 (loss 0.0038 / 0.0268 / 0.4302), mean 0.927 vs RoPE 0.754, **effect +0.173**, per seed +0.275 / +0.211 / +0.033. Grid 8: Vanilla 0.992 / 0.996 / 0.982, RoPE 1.000 x3, **effect -0.010 (sd 0.007)**, per seed -0.008 / -0.003 / -0.018.
- Status: grid-32 +0.173 EXPLORATORY at n=3 (extended to n=5 as +0.178 in D20, which is CITABLE); grid-8 -0.010 is a both-solve ceiling cell (EXPLORATORY). The CROSSOVER is WITHDRAWN.
- Pre-registered? The "what would complete it" decision rule in POSITION_EFFECT_CONVERGED was stated before grid 8 ran; it fired "no crossover".
- Caveats: grid-32 ranges overlap (RoPE max 0.789 > Vanilla min 0.781), Vanilla bimodal; CROSSOVER_CONVERGED's own "surviving claim: monotone in aliasing" was later FALSIFIED (D20). Also withdraws the 100-epoch -0.529 grid-8 anchor (Vanilla 0.448 -> 0.990).
- Sources: ROPE_CONVERGE.md, POSITION_EFFECT_CONVERGED.md, CROSSOVER_CONVERGED.md, MINIWORLD_GRID_SWEEP.md (audit caveat)
- Bears on: rule 10; environment (index never genuinely better).

### D20 Aliasing controlled at fixed grid 32 (n_obs 16/64/256), with 800-epoch extension
- Dates: gates 2026-08-29; results 2026-08-30
- Question: holding map size fixed, does lowering aliasing (cells per token) shrink the position effect?
- Task / environment: fresh-map oracle recode, grid 32, ~512 occupied cells, T=512; n_obs 16 / 64 / 256 = 32 / 8 / 2 cells per token. Non-blank marginal (floor) 0.077 / 0.031 / 0.013; chance 1/16, 1/64, 1/256.
- Arms: Vanilla vs RoPE.
- Seeds / batch: 400 epochs warmup+cosine: n_obs=16 n=5 (3 reused from D19 + 2 new, RoPE s0 repro control 0.725 vs 0.725), n_obs=64 n=4, n_obs=256 n=5. Follow-up (run_alias_followup/waveb): fast-attn control at n_obs=256 400 ep (+0.392 vs +0.374 reference, licensing fast-attn) and n_obs=256 budget extension to 800 ep, n=3.
- Validity gates: ALIASING_GATES all PASS except G2 WARN at n_obs=256 (non-blank marginal 0.013 vs chance 0.004); G4 non-blank best 0.055 / 0.019 / 0.000 vs marginals; G5 label mass 50.4/traj and G6 median lag 33 identical across conditions; G8 vocab range PASS. Convergence per arm; rule 9 loss table.
- Result (400 ep, ALIASING_CONTROLLED.md): n_obs=16 Vanilla 0.936 / RoPE 0.758 / **+0.178** (per seed +0.274, +0.210, +0.033, +0.230, +0.142; all flat; sd 0.094, MDE 0.118, detectable); n_obs=64 0.999 / 0.689 / +0.310 (sd 0.031, MDE 0.043; not all flat); n_obs=256 0.981 / 0.607 / +0.374 (sd 0.046, MDE 0.057; not all flat). Loss gaps -0.318 / -0.534 / -0.591 (ranges do not overlap at 64 and 256, so no loss-matched residual). File verdict: "NOT ALL ARMS CONVERGED. Nothing here is interpretable". 800 ep (CLAUDE.md 2026-08-30; VISITS_TEST.md pooled table): n_obs=256 **+0.305**, n=3, 6/6 flat; index-arm acc 0.676 (VISITS_TEST). CLAUDE.md endpoint contrast +0.178 vs +0.305: +0.127, t=2.52.
- Status: n_obs=16 effect CITABLE (+0.178, n=5, MDE 0.118). The aliasing hypothesis is FALSIFIED with sign inverted -- CITABLE as a negative in direction (pre-registered outcome B fired: effect > 0.150 at all three n_obs), but the converged endpoint +0.305 is n=3 with no sd/MDE recorded (N3_AUDIT #5) and the t=2.52 compares n=5 vs n=3.
- Pre-registered? yes (run_alias_sweep.sh header, outcomes A-D fixed before runs; verdict thresholds hard-coded in the aggregator). Outcome A (aliasing) refuted; B (map size) fired; C (floor collapse) ruled out.
- Caveats: the 800-epoch +0.305 has no generated results file (raw JSON only in runs/alias_follow/n256_800/); n_obs=64 was never converged (6/8 flat per CLAUDE.md) so "monotone the wrong way" rests on the endpoints; the both-flat conditioning argument pointed the wrong direction (400 ep +0.374 -> 800 ep +0.305); T/aliasing also move via p_empty and map size, held fixed here; loss ranges do not overlap, so the study measures "optimises better", not "represents better at equal fit".
- Sources: ALIASING_GATES.md, ALIASING_CONTROLLED.md, VISITS_TEST.md, CLAUDE.md (2026-08-29/30 section), N3_AUDIT.md, run_alias_sweep.sh, run_alias_followup.sh
- Bears on: which environment property decides the position effect; rule 5/10/11.

### D21 Visits test and the map-extent threshold
- Dates: 2026-08-30
- Question: at matched aliasing (2.0 cells/token), is the effect driven by distinct cells visited, prior visits per scored cell, or map extent (occupied cells)?
- Task / environment: fresh-map oracle recode. Conditions: A grid 32, n_obs=256, T=128 (48 distinct, 1.95 prior, 512 occupied); B grid 16, n_obs=64, T=1024 (153 distinct, 6.20 prior, 128 occupied). Trained and evaluated at training length. Measured prior-visit counts (probe_visits_per_cell.py, VISITS_PER_CELL.json) 8.64 / 4.61 / 3.05 at grid 8/16/32, T=512 (not the 16/4/1 arithmetic); per-grid ranges do not overlap (grid 8 5.67-18.35; grid 32 1.95-4.13 over T=128..2048).
- Arms: Vanilla vs RoPE.
- Seeds / batch: n=3 per condition; 400 epochs warmup+cosine, fast-attn (licensed). Grid 16 @ n_obs=64, T=512 point: 400 ep, n=3 (run_alias_followup waveB).
- Validity gates: convergence A 3/3 flat, B 1/3 flat (both arms 0.98-1.00, ceiling); noise floor 0.150.
- Result: A **+0.275** (per seed +0.192, +0.345, +0.288) vs grid 8 T=512 reference -0.010; B **+0.010** (+0.006, +0.005, +0.018) vs grid 32 T=512 reference +0.374 (400-ep value). Condition B index arm 0.9888 / 0.9893 / 0.9824 vs PI 0.9949 / 0.9945 / 1.0000. Pooled table (post-hoc): grid 8 (32 occ, T=512, prior 8.64) -0.010; grid 16 (128, T=512, 4.61) +0.015; grid 16 (128, T=1024, 6.20) +0.010 [index 0.987]; grid 32 (512, T=128, 1.95) +0.275 [index 0.674]; grid 32 (512, T=512, 3.05) +0.305 [index 0.676]. Within map size the effect moves 0.005 (grid 16) and 0.030 (grid 32); across map sizes 0.285.
- Status: distinct-cells-visited FALSIFIED (pre-registered pairwise test) -- CITABLE as a falsification at n=3 given effect sizes ~2x the floor. Map extent over visits-per-cell: EXPLORATORY (post-hoc pooling; the pre-registered pairwise verdict was "cannot separate"). Threshold between 128 and 512 occupied cells: EXPLORATORY (n=3 endpoints, no sd recorded).
- Pre-registered? yes for the pairwise design (run_visits_test.sh; outcomes enumerated in agg_visits.py): A large -> not distinct cells (held); B ~zero -> distinct ruled out (held); prior visits vs map extent pre-registered as unseparated. The pooling is NOT pre-registered.
- Caveats: T changes sequence length as well as visit statistics; prior visits varied only 1.34x/1.56x at fixed extent against a 4.4x total range; map extent and revisit frequency cannot be crossed in MiniWorld (only grid 32 @ T=2048 separates them; not run); the three threshold points use different budgets (grid 32 point is 800 ep, others 400 ep); grid 16 @ n_obs=64 T=512 (+0.015) has no generated results file (raw JSON runs/alias_follow/g16/); the grid 8 point is n_obs=16 at 400 ep without fast-attn (D19). The pre-registered grid-16 band ("between -0.010 and +0.374") fired "graded" mechanically and was overruled (rule 15).
- Sources: VISITS_TEST.md, CLAUDE.md (2026-08-30 sections 2 and 5), .claude-memory/project_miniworld_flip_negative.md, run_visits_test.sh, N3_AUDIT.md #5
- Bears on: the headline boundary "effect tracks map extent, not aliasing" (RESULTS_INDEX).

### D22 MiniWorld hierarchy: grid sweep hier pair and pooling ablation
- Dates: 2026-08-26/27
- Question: does hierarchy shift the crossover / amplify path integration on MiniWorld?
- Task / environment: fresh-map oracle recode, grids 8/16/24/32, T=512/1024, chance 0.0625.
- Arms: MapWM-Hier (pooled k=2) vs Plain-Hier (index+hier), both 2.38M; ablation MapWM-Hier vs MapWM-FlatHG (identical scaffold, 2,384,026 params, differ only in pooling).
- Seeds / batch: n=3, 100 epochs; ablation retrained both arms in one batch (MapWM-Hier reproduced prior-batch values exactly, 0.683/0.943/0.978/0.986).
- Validity gates: convergence conditioning.
- Result: hier pair T=512 effect -0.292 / +0.346 / +0.393 / +0.388 (grids 8-32). Pooling effect means T=512 +0.335 / +0.064 / +0.151 / +0.231 with per-seed sign disagreement at grids >= 16; among both-converged pairs mean +0.001 (threshold 0.2) or -0.022 (threshold 0.4). Runs with final loss > 0.4: MapWM-FlatHG 7 of 12, MapWM-Hier 2-3 of 12.
- Status: pooling accuracy effect: unmeasured (converged pairs ~0); convergence reliability difference EXPLORATORY (n=3 per grid). Hier-pair crossover: not established (unconverged 100-ep recipe; flat-pair crossover withdrawn in D19).
- Pre-registered? no.
- Caveats: "hierarchy adds +0.283 to path integration" RETRACTED (scaffold/param mismatch 2.38M vs 3.17M and convergence failures); the in-file threshold was misreported (0.2 vs 0.4) and CORRECTED; HIER_ABLATION's "STILL TRUE: the crossover replicates in both pairs" predates and is contradicted by CROSSOVER_CONVERGED.
- Sources: MINIWORLD_GRID_SWEEP_HIER.md, MINIWORLD_HIER_ABLATION.md
- Bears on: hierarchy (trainability, not capability).

### D23 RoPE frequency schedule: repo n_b-1 vs canonical
- Dates: 2026-09-04
- Question: is the index baseline's frequency schedule load-bearing?
- Task / environment: parity (train_algorithmic), L=16..256.
- Arms: RoPE repo schedule `base^(-c/(n_b-1))` vs canonical `base^(-c/n_b)`.
- Seeds / batch: n=16, one batch.
- Validity gates: paired t, sign test, MDE per length.
- Result: canonical - repo L=16 -0.0425 (MDE 0.1480, 7/16); L=32 -0.0199 (MDE 0.0685); L=64 -0.0083 (MDE 0.0339); L=128 -0.0035 (MDE 0.0169, 8/16); L=256 -0.0023 (MDE 0.0086, 7/16). All inside MDE.
- Status: POWERED NEGATIVE at the stated MDEs (tightest 0.0086 at L=256); code switched to canonical.
- Pre-registered? yes (run_rope_canonical.sh header decision rule); "inside MDE at every length" branch fired.
- Caveats: one task (parity), one width; inv_freq is a registered buffer so pre-2026-09-04 RoPE checkpoints keep the n_b-1 schedule; index RoPE base 10000 never tuned.
- Sources: ROPE_CANONICAL.md, run_rope_canonical.sh, CLAUDE.md 2026-09-04
- Bears on: validity of every index-RoPE baseline in the repo.

### D24 Test-time omega rescaling for grid-size generalisation (April, clean config)
- Dates: 2026-04-24
- Question: does multiplying omega by train_size/test_size at eval help transfer to a different grid size?
- Task / environment: torus, clean config, trained grid 64, T=128; test grids 32/48/64/96/128; revisit accuracy; fresh test seeds. No floor reported.
- Arms: Vanilla, VanillaEM, Level1, Level15, Level15EM, PC (LSTM, MambaLike rows empty).
- Seeds / batch: 3 model seeds x 3 fresh test seeds = 9 runs per cell; April recipe.
- Validity gates: none.
- Result (orig / rescaled): Vanilla grid 32 0.954 +/- 0.015 / 0.310 +/- 0.108; grid 128 0.988 +/- 0.013 / 0.681 +/- 0.199. VanillaEM grid 32 0.970 / 0.938; grid 128 0.999 / 0.870. Level15 grid 32 0.964 / 0.990; grid 128 1.000 / 0.886. Level15EM grid 32 0.969 / 0.698. Unrescaled models transfer across grid size (Vanilla 0.954-0.992).
- Status: EXPLORATORY (n=3 model seeds, April recipe, no floor). Clean-config rows of that era are valid (not lm200).
- Pre-registered? no.
- Caveats: pre-dates the convergence/recipe rules; no measured floor; OMEGA_RESCALE_lm200.md is archived void.
- Sources: OMEGA_RESCALE_clean.md
- Bears on: whether omega encodes map scale; grid-size transfer.

### D25 Dissociation sweep: hierarchy vs path integration over n_templates (compositional task)
- Dates: 2026-08-19
- Question: is which ingredient pays a property of the task structure?
- Task / environment: compositional rooms task, 64x64 grid, room size 8, n_templates 2/4/8/16 (motif count); cross_nb_acc @T=256 (floor 0.072 / 0.072 / 0.071 / 0.081) and exact_acc @T=1024 (no floor given). Each model evaluated on the env it was trained on.
- Arms: MapWM-Flat, MapWM-Hier, Plain-Flat, Plain-Hier.
- Seeds / batch: n=3, one batch (runs/dissociation); published compositional recipe (LinearLR from step one, lr 3e-4, 50 epochs -- per COMP_HEADROOM.md).
- Validity gates: floors; aliasing covariate measured (ALIASING_COVARIATE.md: ~1100 cells share an observation at every point, H(obs) 2.91-3.00 bits; run_8 disambiguation 20.43 -> 7.40 -> 3.35 -> 2.14).
- Result: hierarchy effect on cross_nb +0.163 / +0.135 / +0.034 / +0.006; path-int effect +0.081 / +0.078 / +0.174 / +0.114. exact_acc hierarchy +0.120 / +0.076 / +0.014 / -0.015; path-int +0.060 / +0.158 / +0.251 / +0.154. At nt=16 the hierarchy +0.006 averages -0.072 (MapWM) and +0.084 (Plain).
- Status: DIRECTIONAL/EXPLORATORY (n=3, no MDE, MapWM sds up to +/-0.216).
- Pre-registered? yes (sweep_dissociation.py): P1 (hierarchy advantage falls) CONFIRMED; P2 (PI exact_acc advantage flat) REFUTED; P3 (crossover) CONFIRMED. The proposed aliasing explanation for P2 was ruled out by ALIASING_COVARIATE (wrong sign); the "position-aware ceiling rises" account is untested.
- Caveats: recipe later shown to be a recipe limit on this task (COMP_HEADROOM C - A +0.160); backbone divergence at nt=16; one env.
- Sources: DISSOCIATION_SWEEP.md, ALIASING_COVARIATE.md, BASELINE_TABLE.md sec. E, COMP_HEADROOM.md
- Bears on: hierarchy line; task structure decides the ingredient.

### D26 D-dimensional torus gates (for the D x r rank test)
- Dates: 2026-09-04
- Question: are D=2/3/5 torus tasks shortcut-free before the rank test?
- Task / environment: T=128, 200 trajectories, n_obs 16. D=2 grid 32 (1024 cells, 4 actions), D=3 grid 10 (1000 cells, 6 actions), D=5 grid 4 (1024 cells, 10 actions).
- Arms: none (gates).
- Seeds / batch: seed 0.
- Validity gates: chance (majority) 0.5064 / 0.5227 / 0.5260; best action n-gram 0.513 / 0.531 / 0.536; revisit rate 0.233 / 0.283 / 0.620; scored/traj 29.8 / 36.3 / 79.3; G5 min separation r=2 0.0472 / 0.0330 / 0.0198. All PASS.
- Result: gates only.
- Status: gate record (supports DXR_RANK_THRESHOLD.md in the rank line).
- Pre-registered? n/a.
- Caveats: D=5 revisit rate 0.620 is much higher than D=2, so floors and label mass differ across D.
- Sources: ND_GATES.md
- Bears on: rank line (DxR).

---

## Excluded

| file / claim | reason | killed by |
|---|---|---|
| MINIWORLD_RESULTS.md (40-epoch MiniWorld factorial) | INCONCLUSIVE: no arm learned (0.08-0.23 non-blank vs 1.0 oracle) | in-file banner |
| MINIWORLD_GRID_SWEEP.md "attention substitutability" mechanism and grid-size CROSSOVER (-0.529 at grid 8 -> positive at >= 16) | mechanism falsified by gate G6 (lags 47/43/38/33 shorten; within-32 fraction rises 0.43->0.50); crossover was a convergence crossover (RoPE 0/9 converged at g>=16, Vanilla 0/3 at g8); converged grid 8 = -0.010 | MINIWORLD_FRESH_FINDINGS.md (retraction block), MINIWORLD_GRID_SWEEP.md audit, CROSSOVER_CONVERGED.md |
| MINIWORLD_GRID_SWEEP_HIER.md "hierarchy adds +0.283 to PI" | scaffold/param mismatch + convergence failures | MINIWORLD_HIER_ABLATION.md |
| MINIWORLD_FRESH_FINDINGS "reconstruction fidelity sets the sign" and later "H1 REFUTED" | first refuted by oracle recode; the refutation itself corrected as overshoot | MINIWORLD_FRESH_FINDINGS.md (both correction blocks) |
| MINIWORLD_FRESH_FINDINGS "path integration is a LIABILITY on fresh-map" | 100-ep LinearLR recipe; the matched grid-8 anchor inverted once converged; never re-run converged | CROSSOVER_CONVERGED.md (by implication) |
| CROSSOVER_CONVERGED.md "surviving claim: monotone in aliasing" (-0.010 / +0.173 / +0.461) | aliasing co-varied with map size; fixed-grid manipulation gives opposite ordering | ALIASING_CONTROLLED.md, VISITS_TEST.md, CLAUDE.md 2026-08-30 |
| BASELINE_TABLE / CLAUDE.md aliasing explanation of the environment table (grid 8 -0.010, grid 32 +0.173, torus +0.461) | same | CLAUDE.md CORRECTED 2026-08-30 block |
| "Distinct cells visited" hypothesis | condition A +0.275 vs grid 8 -0.010 at matched distinct cells | VISITS_TEST.md |
| KNOB_SWEEP.md original rotate row (+0.000) | void, order-1 answer shortcut 0.932 | KNOB_SWEEP.md gate section |
| KNOB_SWEEP.md "every knob contributes about equally" | withdrawn after rotate redo | KNOB_SWEEP.md "ROTATE, REDONE" |
| KNOB_SWEEP.md pre-registered "aliasing and size drive it" | refuted | KNOB_SWEEP.md |
| CONTINUOUS_ALLOC.md "partial recovery at 12 headings" | undertraining | CONTINUOUS_ALLOC.md correction, H12_BUDGET_CURVE.md |
| CONTINUOUS_ALLOC.md / CLAUDE.md "recovers once budget adequate, still climbing (+0.264 -> +0.383)" | nb=4000 +0.286, bimodal | H12_BUDGET_CURVE.md, CLAUDE.md CORRECTED 2026-08-23 |
| MINIGRID_2X2X2.md "18/18 hierarchy positive, every seed" | n=3 luck; 27/32 at n=8 | BASELINE_TABLE.md sec. H, N3_AUDIT.md |
| MINIGRID_ALLOCENTRIC_2X2X2.md "all four PI arms outrank all index arms" | PoPE-Hier 0.817 > MapWM-Flat 0.809 | MINIGRID_ALLOCENTRIC_2X2X2.md RESOLVED note |
| MINIGRID_ALLO_8THCELL.md "control MISSING, comparison NOT valid" | stale generator output; control present and exact | MINIGRID_REPRO_CONTROL.md, MINIGRID_ALLOCENTRIC_2X2X2.md |
| MINIGRID_2X2.md pre-registered "rotation makes PI actively misleading" (strong form) | MapPoPE-Flat (PI) best arm | MINIGRID_2X2.md |
| Torus encoding main effect +0.011 and "40x" ratio | was the PI row at n=3, not the n=8 main effect; corrected to +0.003 (MDE 0.029, 5/8) | BASELINE_TABLE.md, MINIGRID_2X2X2.md correction, N3_AUDIT.md |
| MINIGRID_EM.md "index codes win on MiniGrid" (as a general statement) | scope narrowed: with allocentric + r=4 PI wins 8/8 | MINIGRID_EM.md |
| MINIWORLD_PROBE3.md "NoPE collapses to chance; RoPE not a straw man" | n=1, still descending, non-sequitur | MINIWORLD_PROBE3.md correction |
| MINIWORLD_GATE_CONTROL "all-converged +0.022" as a number | n=1 | MINIWORLD_GATE_CONTROL.md correction |
| MINIWORLD_SROPE_COMPONENTS "null" | unmeasured (MDE 0.165/0.190) | in-file correction |
| MINIWORLD_HIER_ABLATION converged-pair numbers at "0.4" threshold | were computed at 0.2 | in-file CORRECTED 2026-08-27 |
| CLOCK_SCAN.md (modular-clock navigation, n=8) | SUSPECT: planner-demonstration task never re-validated with action-only n-gram; that family voided (0.650-0.969 shortcuts) | CLOCK_SCAN.md banner, PLANNER_TASK_AUDIT.md |
| OMEGA_RESCALE_lm200.md | lm200 era, void | archive/void/ |
| MiniWorld Habitat premise "allocentric recoding rescues PI in continuous 3D" | falsified in both MiniWorld regimes at the tested recipe | MINIWORLD_FIXED_FINDINGS.md, MINIWORLD_FRESH_FINDINGS.md |
| PERCEPTION_EXPERIMENT_PLAN.md, PUBLICATION_VENUES.md, MINIWORLD_TODO.md | not experiments (plan, venue list, engineering notes) | n/a |

## Cross-line dependencies

- **Headline boundaries in RESULTS_INDEX** ("effect tracks map extent, not aliasing"; "does not survive rotation-based actions, restored by allocentric recoding") rest on D11, D20, D21.
- **Torus 2x2 headline** (another line) is cited in every MiniGrid file as the comparison row (0.530/0.967/0.509/0.994, floor 0.506, INDEX_BASELINE_PAPER_TASK_n8.md); the encoding main effect +0.003 correction touches both lines.
- **Rank line**: D8 (Vanilla_r4 - RoPE +0.034 8/8; r=2 -0.010) is an external-benchmark instance of "use r=4"; D9 compounds rank with the p0 fix; D26 gates DXR_RANK_THRESHOLD.md.
- **EM/WM line**: D8/D9 are the only EM runs on a MiniGrid/MiniWorld environment; D9 adds the fourth map-task refutation of separate q0/k0 and the statement that prior "EM is worse" results used the pathological arm (EM_WM_STATE.md cross-reference on recency sign reversal).
- **Sign / clock-map line**: D11's mechanism (fixed per-token delta must name a displacement) is the navigation-side counterpart of the signed-increment requirement.
- **Hierarchy line**: D6 (hierarchy as compensation, 27/32), D22 (pooling = trainability), D25 (dissociation over n_templates) feed the hierarchy-negative memory.
- **Selective RoPE line**: D18 (ConvDelta/GateDelta unmeasured on navigation).
- **Correction line**: D1/D2 (Level15 on DoorKey, n=1) and D24 (omega rescale across Level15 variants).
- **Method rules**: D17 (noise floor 0.150, rule 8), D12/D19 (rule 9 r=-0.996, rule 10 LinearLR), D10 (gate before training), D20 (budget extension beats conditioning), D23 (validity of all index-RoPE baselines).

## Files read

MINIGRID_2X2.md, MINIGRID_2X2X2.md, MINIGRID_2X2X2_n8.md, MINIGRID_FULL_2X2X2.md, MINIGRID_ALLO_8THCELL.md, MINIGRID_ALLOCENTRIC_2X2X2.md, MINIGRID_ALLOCENTRIC_8CELL.md, MINIGRID_DK16_RESULTS.md, MINIGRID_DOORKEY_CACHED.md, MINIGRID_DOORKEY_LONGT.md, MINIGRID_DOORKEY_RESULTS.md, MINIGRID_DOORKEY_ROPE_DIAG.md, MINIGRID_EM.md, MINIGRID_EM_PREREG.md, MINIGRID_EM_FIX.md, MINIGRID_EM_FIX_PREREG.md, MINIGRID_REPRO_CONTROL.md, MINIWORLD_RESULTS.md, MINIWORLD_TODO.md, MINIWORLD_PROBE3.md, MINIWORLD_FIXED_FINDINGS.md, MINIWORLD_FIXED_RESULTS.md, MINIWORLD_FIXED_RESULTS_T1024.md, MINIWORLD_FRESH_ABLATION.md, MINIWORLD_FRESH_FINDINGS.md, MINIWORLD_FRESH_GATES_ALLO.md, MINIWORLD_FRESH_GATES_RAW.md, MINIWORLD_FRESH_RESULTS.md, MINIWORLD_FRESH_RESULTS_T1024.md, MINIWORLD_GATE_CONTROL.md, MINIWORLD_GATES_ALLO.md, MINIWORLD_GATES_FIXED.md, MINIWORLD_GRID16_GATES.md, MINIWORLD_GRID24_GATES.md, MINIWORLD_GRID32_GATES.md, MINIWORLD_GRID_SWEEP.md, MINIWORLD_GRID_SWEEP_HIER.md, MINIWORLD_HIER_ABLATION.md, MINIWORLD_ORACLE_ABLATION.md, MINIWORLD_ORACLE_GATES.md, MINIWORLD_ORACLE_RESULTS_T512.md, MINIWORLD_ORACLE_RESULTS_T1024.md, MINIWORLD_SROPE_COMPONENTS.md, ALIASING_CONTROLLED.md, ALIASING_COVARIATE.md, ALIASING_GATES.md, ALLOCENTRIC_RECODING.md, VISITS_TEST.md, KNOB_SWEEP.md, KNOB_SWEEP_n8.md, HABITAT_BUILD.md, POSITION_EFFECT_CONVERGED.md, ROPE_CONVERGE.md, ROPE_CANONICAL.md, FREQ_CONTROL.md, H12_BUDGET_CURVE.md, CROSSOVER_CONVERGED.md, DISSOCIATION_SWEEP.md, ND_GATES.md, CLOCK_SCAN.md, PERCEPTION_EXPERIMENT_PLAN.md, PUBLICATION_VENUES.md.
Also: report/INVENTORY_BRIEF.md, RESULTS_INDEX.md, N3_AUDIT.md, KNOWN_BUGS.md, archive/void/README.md, BASELINE_TABLE.md (top, sec. E, H, I, coverage gaps), CONTINUOUS_ALLOC.md, OMEGA_RESCALE_clean.md, COMP_HEADROOM.md (head), .claude-memory/project_miniworld_flip_negative.md, CLAUDE.md (relevant sections), headers of run_alias_sweep.sh, run_alias_gates.sh, run_alias_followup.sh, run_alias_finish.sh, run_alias_waveb.sh, run_visits_test.sh, run_rope_canonical.sh, run_rope_converge.sh, run_knob_sweep.sh, run_minigrid_2x2x2.sh, run_minigrid_em.sh, run_minigrid_em_fix.sh, run_mw_fresh_factorial.sh; raw JSONs runs/alias_follow/{n256_800,g16,ctl_n256}/s*/{Vanilla,RoPE}_oracle.json (inspected only to confirm the undocumented +0.305 and +0.015 points exist as data; no numbers derived from them are reported above).

## Files in scope not covered

None. MINIGRID_ALLO_8THCELL.md is untracked in git (no commit date). README.md was not reopened for this line (orientation only via CLAUDE.md/RESULTS_INDEX.md). PERSCALE_OMEGA_RESULTS.md, INDEX_BASELINE_PAPER_TASK_n8.md and VISITS_PER_CELL.json were referenced but not read in full (outside the listed scope).
