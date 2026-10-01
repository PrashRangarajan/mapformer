# Context-dependent step -- pre-registration (2026-09-30, before any run of the batch)

## Question
MapFormer's step is a function of the word alone, so a direction word used WITHOUT moving ("she never,
after a long pause, walked north") moves the phase like a real move. Which context-dependent steps can
suppress such decoys, and does it depend on how far away the cue is? Pilots (`CTXSTEP_PILOT1/2/3.md`,
`CTXSTEP_HS_RECIPE.md`, `CTXSTEP_HSR_PILOT.md`; design history in `CONTEXT_STEP_DESIGN.md`, including two
predictions that failed) suggest: steps that see a fixed 4-token window (a context gate, the
Selective-RoPE generator) use a cue on either side but only within the window; a step computed through
attention reaches any distance, and trains reliably only if it keeps the word's own step and adds
context as a correction.

## Task (`environment_textworld_ctx3.py`, gated `docs/audits/2026-09-27/gate_ctxstep3.py`)
The text world (torus walk rendered as English) with decoys after 30% of steps. A single cue word marks
move vs decoy, LEADING ("then/soon/finally" vs "never/nearly/almost") or TRAILING ("and/so/then ... saw"
vs "but/yet/though ... she stayed"). NEAR: the cue is 1-3 tokens from the direction word; FAR: 7-12
(leading) or 6-10 (trailing), with the same pad moved to the other side so tokens per step are equal.
Gate: with the cue far, move vs decoy is at base rate from the 4 tokens before and the 3 after the
direction word; near, the cue side decides it. T = 2048 words (~100 steps), held-out map (seed 10000),
eval stream seed 10**6. Floors (gate, eval set): best constant 0.509 / 0.483, reversal-copy 0.616 / 0.590
(lead / trail).

## Arms and cells (one batch, every run retrained, 1800 epochs each, batch 8, lr 1e-3, warmup + cosine,
## d 128, 2 heads, --data-workers 3; seeds 6-13 -- seeds 0, 1, 2, 3, 5 were used by the pilots)
| arm | step | layers | far cue (lead, trail) | near cue (lead, trail) |
|---|---|---|---|---|
| CF | context-free (`Vanilla_r4`) | 1 | 8 + 8 | -- (suppression impossible by construction) |
| CG | context gate, 4-token causal conv (`CtxGateWM`) | 1 | 8 + 8 | 8 + 8 |
| SR | Selective-RoPE generator at MapFormer's placement (`MapFormerWM_SRoPEGen`) | 1 | 8 + 8 | 8 + 8 |
| HSR | word step + alpha * attention context, alpha init 0 (`HiddenStepResWM`) | 2 | 8 + 8 | -- |
96 runs. SR is NOT a one-knob arm: besides its window it has no rank bottleneck and no omega (stated in
`CTXSTEP_PILOT3.md`); its window reading is read together with CG's, not alone. HSR has 2 layers; CF, CG
and SR 1 -- the reach contrast is a claim about the design, not about matched depth.

## Readouts, per run
Accuracy at T=2048 (registered); SOLVED (final-5% loss < 0.05); decoy/move ratio from the committed swap
test (`docs/audits/2026-09-27/swap_test.py --offset 15`, 30 held-out sequences, 50 direction words per
class): the change in accumulated angle 15 tokens after a direction word when it is switched north <->
south, decoys over real moves (0 = decoys ignored, 1 = treated as moves); "learned a step" = move change
>= 0.01.

## Hypotheses and branches (fixed now; tests: exact permutation on accuracy, two-sided, `stats_core`)
**H-W, the window limit.** For CG and SR separately, on each cue side: near - far accuracy, and the
median decoy/move ratio near and far.
- **WINDOW-LIMITED** -- in all 4 (arm x cue) cases: near - far fires (p < 0.05, near higher) AND median
  ratio <= 0.3 near AND >= 0.7 far.
- **PARTLY WINDOW-LIMITED** -- the same holds in 2 or 3 of the 4 cases (each case named).
- Otherwise: reported as it falls.
**H-R, reach.** On each cue side, far: HSR - max(CG, SR) accuracy, and HSR's median ratio.
- **ATTENTION REACHES** -- on both sides: HSR - max(CG, SR) fires (HSR higher) AND HSR median ratio <= 0.3
  AND HSR learned a step on >= 7/8 seeds.
- **REACHES ON ONE SIDE** -- the same on exactly one side.
- Otherwise: reported as it falls.
Secondary (no verdict): HSR - CF far; CF ratio (1.00 by construction, a check); SOLVED counts; HSR's alpha;
r(final loss, accuracy) per arm; T=4096 accuracy (extrapolation).
Void: any run missing; the md5 guard trips; a CF ratio differing from 1.00 by more than 0.01 (wiring).
Scope: one scripted grammar, single-word cues, window 4 for CG/SR, T=2048, 1800 epochs, n=8 per cell.
Cost estimate: ~1-1.5 days on both GPUs (two-layer HSR runs are ~2x the one-layer ones).
