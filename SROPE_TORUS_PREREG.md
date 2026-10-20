# Does faithful Selective RoPE learn the 2D torus map? -- pre-registration (2026-10-20, before the batch)

Written before any batch run exists. The pilot (seed 1790, outside the batch, never pooled) was launched while this file was
being written; the predictions below were fixed when the pilots stood at epoch ~150 of 900 (training loss 0.58-0.66, nothing
evaluated). The pilot section was added after it finished and changes no registered decision.

## Question
The paper (`docs/paper/MAIN_DRAFT1.md`, `docs/paper/expansion/SYNTHESIS_DRAFT2_PLAN.md`) claims that encodings whose phase comes
from content through **signed** steps, with **per-head rank of the content-to-angle map above the map's dimension**, learn the
torus map, and that monotone ones (CoPE, CARoPE: `GENERALITY_RESULTS.md`) and index RoPE do not. Selective RoPE (Movahedi et al.,
ICLR 2026, arXiv:2511.17388) is a published content phase that is signed and, at its best configuration, per-head rank 32 (one
full-rank projection per head) -- far above D = 2. The theory therefore predicts it learns the map, **unless its other
ingredients obstruct it**: a causal conv that mixes four neighbouring tokens' angle inputs before the cumsum, a sigmoid phase
gate, SiLU after the conv (bounded below at -0.278, so negative steps are capped while positive ones are not), and a fixed
language-model frequency spectrum (500000^(-2c/64): only 6 of 32 channels per head have temperature >= 0.1 (1, .66, .44, .29, .19, .13)). That is
the registered question. It is a test of the theory on a published method it did not design, not a reproduction of the
Selective RoPE paper (which has no navigation task).

## The faithfulness audit of `model_selective.py::SRoPEGen` (read against the paper and the authors' code)
Sources: the paper's Fig. 4 pseudocode, Sec. 4.1 implementation paragraph, Table 2 ablations (`papers/txt/srope.txt` l. 785-1000,
1150); its Appendix B code listing is NOT in the text we hold, so the authors' released code was read in full:
github.com/timurcarstensen/selective-rope at d12e825 (2026-03-01), `src/selective_rope/modules/rotary.py::SelectiveRoPE`, the
attention layer that calls it (`layers/attn.py`: it receives the block's normed input, per layer), the HF init
(`models/transformer/modeling.py::_init_weights`), the configs, and the two flash-linear-attention primitives it uses (fla rev
64248fc: `ShortConvolution` = depthwise causal `nn.Conv1d`, padding k-1, then SiLU if `activation`; `l2_norm` = x / sqrt(sum x^2
+ 1e-6)). The best language-model configuration (`configs/language_modeling/language_modeling.yaml`): `phi_parametrization:
full-rank`, conv (`d_conv` 4) with `conv_act: silu`, `use_gate: true`, `gate_parametrization: standard`, `use_bias: false`;
temperature from the base config: `temp_type: rope`, `temp_theta: 500000`, `temp_max: 1.0`, `temp_grad: false`.

| ingredient | paper / authors' code (best LM config) | `SRoPEGen` (model_selective.py, 2026-09) | material? |
|---|---|---|---|
| input to the angle map | the attention block's **normed** input x (not q: Table 2 says x is more stable), **l2-normalised per head chunk** (d_model / H = 64 dims per head) | raw token embedding, no norm | yes |
| projection | per head, `ChunkedLinear`: head h reads ONLY its 64-dim chunk -> 32 channels; **weight norm** per output column with ONE learnable scale g per head (init 1, exempt from weight decay); kaiming-uniform (a = 0); no bias | one `Linear(128 -> 64)` from the whole x, PyTorch `weight_norm` (a scale per output channel, init = row norm), **with bias**; default init | yes |
| conv | depthwise causal, kernel 4, no bias, **init N(0, 0.02)** (HF `_init_weights` on `nn.Conv1d`), **SiLU after** | depthwise causal, kernel 4, no bias, default init U(-0.5, 0.5), no activation | yes |
| gate | `sigmoid(Linear(128 -> H))` on the l2-normed input: **one gate per head** and position; init N(0, 0.02), bias 0 (gate 0.5) | `sigmoid(Linear(128 -> 64))` on the raw embedding: one gate per channel; default init | yes |
| temperature / frequencies | **fixed** per-channel vector `500000^(-2c/d_head)`, c = 0..31 (1 down to 3e-6) | ONE learnable scalar exp(log_temp), init 1, shared by all 64 channels (no frequency spectrum) | **yes (largest)** |
| bias term | off in the best config (`exp(phi_bias)`, a positive per-step drift, when on) | none | no |
| order | proj -> conv -> SiLU -> (bias) -> gate -> cumsum -> x temperature | proj -> conv -> gate -> cumsum -> x temp | -- |
| placement | per attention layer, from that layer's normed input | once, before the blocks, from the embedding; shared by all layers | at 1 layer only via the norm |
| rotation pairs | GPT-NeoX halves (c, c + 32) | interleaved (2c, 2c+1) | no (a fixed permutation of q/k rows; the paper says so) |

`SRoPEGen` is therefore NOT faithful: it is "a full-rank, gated, convolved content step" without Selective RoPE's input
normalisation, per-head structure, weight-norm scale, init, SiLU or frequency spectrum. It also keeps the confound recorded in
`SELECTIVE_ROPE.md` (no learned per-channel frequencies). It differs materially, so it stays in the batch **as is**, as a
secondary arm. Prior evidence on it: trained at T = 128 on the torus it reads 1.000 at 128 and 0.899 at 1024 (`MONOTONE_RESULTS.md`,
out of distribution); it has never been trained at T = 1024.

## The faithful arm (`model_srope.py`, new; no default changed anywhere)
`SelectiveRoPEAngle` reproduces `SelectiveRoPE`'s PyTorch path line for line, with the authors' init (ChunkedLinear kaiming a = 0
then normalised, g = 1; conv and gate N(0, 0.02), gate bias 0; temperature a fixed buffer). `MapFormerWM_SRoPE` is MapFormer-WM
with `action_to_lie` / `path_integrator` deleted and one generator per layer reading that layer's `norm1` output; the scaffold
(embedding, block, readout) is built first, so its initial weights equal `MapFormerWM`'s at the same seed. `MapFormerWM_SRoPE_NoSiLU`
is the same with `conv_act=None` (a single-knob arm: the step is as free to be negative as positive; whether Table 2 ran exactly this combination is ambiguous, its rows may be cumulative).
Per-head rank of the content-to-angle map: 32 (SRoPE), 32 (SRoPEGen, over 128 inputs), 4 (MapWM `Signed_r4`, shared latent).

**Verified against the authors' module** (`scratchpad/tests/verify_srope_official.py`; their `rotary.py` executed with the two
fla primitives stubbed from fla source; weights copied, randomised to N(0, 0.7) so every path is exercised; B 3, T 37, H 2, d_head
64): rotated q, k agree to <= 6.1e-6 (fp32) and the angle agrees with a literal transcription of Fig. 4 (omega = W xhat; conv1d;
[SiLU]; [+ bias]; x gate; temp x cumsum) to <= 3.8e-6, for conv_act in {silu, None} x bias in {off, on}. The NeoX -> interleaved
pairing was checked by permuting q, k into MapFormer's `_apply_rope` and back.

Deviations of the faithful arm (unavoidable or chosen, all disclosed): MapFormer's block (LayerNorm, GELU FFN, dropout 0.1)
instead of their RMSNorm / SwiGLU HF block; our `train()` puts every parameter in one AdamW group, so g gets weight decay 0.05
(theirs is exempt); PyTorch path, not their Triton kernel; the frequency spectrum is their language-model one (theta 500000), not
re-tuned to a 64-cell torus -- that is part of "faithful", and if S1 reads FAILS the spectrum is the first named suspect, not
something this batch can separate.

## Design
Torus 64 x 64, 16 observation types, p_empty 0.5, T = 1024 (2048 tokens), 900 epochs; GENERALITY's torus recipe exactly
(`run_generality.sh` `torus`): `train_variant`, lr 1e-3, batch 16, 98 batches per epoch, cosine schedule, `--fast-attn`,
`--data-workers 3`, OMP 2, AdamW weight decay 0.05; 1 layer, 2 heads, d_model 128. Training loss on revisited observations only.

| arm | variant | role | params |
|---|---|---|---|
| SRoPE | `SRoPE` (`model_srope.MapFormerWM_SRoPE`) | faithful Selective RoPE, authors' best LM config: primary | 208,537 |
| SRoPE_NoSiLU | `SRoPE_NoSiLU` | the same without SiLU (signed-symmetric step) | 208,537 |
| SRoPE_ST | `SRoPE_ST` (Amendment 1) | the authors' state-tracking config (low-rank 32, no SiLU, no gate, bias on, all temperatures 1) | 210,389 |
| SRoPEGen | `SRoPEGen` (`model_selective`, unchanged) | the earlier, non-faithful generator swap | 220,758 |
| MapWM | `Signed_r4` | in-batch positive reference (signed, rank 4; GENERALITY's torus reference, 8/8 SOLVED) | 204,757 |
| RoPE | `RoPE` | in-batch negative reference (index RoPE) | 203,925 |

**Seeds** 1700-1707 (fresh: no run dir, driver or prereg uses 1700-1799). At a seed every arm sees the same training stream;
the scaffold's initial weights are shared by SRoPE, SRoPE_NoSiLU and SRoPEGen (built before the generator). Pilot seed 1790;
smoke seeds 1780-1781 (CPU, scratch); bitwise check seed 1789. 6 arms x 8 seeds = 48 runs (SRoPE_ST added in Amendment 1).

**Evaluation** (after all training): `eval_noise_refine` (held-out map, env seed 10000, 100 trajectories, np seed 1234 + s) at
T = 1024 (primary) and 2048; `eval_rank_strata` at 1024, checked against the former. Eval mode, no re-score (none declared).

## Registered decisions (`analyze_srope_torus.py`)
SOLVED = `stats_core.classify_run` (mean training loss over the final 5% of epochs < 0.05). Fisher exact on SOLVED
(`fisher_solved`), unpaired permutation on accuracy at T = 1024 (`perm2_p`, CI `perm2_ci`). n = 8 per arm.

**S1 (primary): SRoPE vs MapWM**, rule in this order:
- **UNREADABLE** if MapWM solves < 6/8 (the positive reference failed under this recipe; nothing is read);
- **FAILS** if Fisher p < .05 with MapWM ahead, or permutation p < .05 with d = SRoPE - MapWM <= -0.05;
- **LEARNS THE MAP** if SRoPE SOLVED >= 6/8 and d >= -0.01 (within 0.01 of MapWM, or better);
- **UNMEASURED** otherwise.
(LEARNS and FAILS cannot both hold at n = 8: SOLVED >= 6/8 against <= 8/8 gives Fisher p >= 0.467, and d >= -0.01 excludes the
accuracy route.) LEARNS deliberately ignores a significant but sub-0.01 accuracy deficit: "learns the map" means solves it, not
"identical to MapWM".

**S2 (primary, a gate on what S1 FAILS means): SRoPE vs index RoPE.** **BEATS INDEX** if permutation p < .05 with d > 0, or
Fisher p < .05 with SRoPE ahead; **WORSE THAN INDEX** if permutation p < .05 with d < 0, or Fisher p < .05 with RoPE ahead;
**NOT DISTINGUISHED** otherwise.

**Reading** (`analyze_srope_torus.reading`): UNREADABLE if S1 is; **SELECTIVE ROPE LEARNS THE TORUS MAP** if S1 = LEARNS; if
S1 = FAILS: **FAILS THE MAP BUT USES ITS CONTENT PHASE** (S2 BEATS INDEX) or **FAILS THE MAP AND IS NOT DISTINGUISHED FROM /
WORSE THAN INDEX ROPE**; **UNMEASURED** otherwise (S2 named).

Power (Fisher, n = 8): with MapWM 8/8, SRoPE <= 3/8 reads FAILS by SOLVED (p 0.026; 4/8 gives 0.077), and 4-5/8 can read FAILS
only through the accuracy route; 6-8/8 cannot read FAILS by SOLVED. GENERALITY's index RoPE sat at 0.722 (sd 0.02), CARoPE at
0.667: a failing content phase lands far below MapWM's 1.000 and the accuracy route has ample power at -0.05.

**Declared secondary contrasts** (labelled by the same functions, no family-wise correction, not part of the reading):
SiLU = SRoPE_NoSiLU vs SRoPE (**SiLU COSTS** if Fisher p < .05 with NoSiLU ahead or permutation p < .05 with d >= +0.05; **SiLU
HELPS** mirrored; **NO DIFFERENCE SHOWN** otherwise); SRoPE_NoSiLU vs MapWM and SRoPEGen vs MapWM (S1 rule); SRoPEGen vs RoPE and
SRoPE_NoSiLU vs RoPE (S2 rule).

**Declared secondary readouts (descriptive):** accuracy at T = 2048 (2x the training length; OOD); loss AUC (mean per-epoch
loss); run classes; SOLVED at loss 0.1 (threshold sensitivity, with Fisher p for SRoPE vs MapWM and NoSiLU vs SRoPE); revisit
strata (lag < 128, lag >= 128, wrap-only) from `eval_rank_strata`; learned angle statistics on 8 eval-stream walks per seed
(layer-1 angle, per-position increment = first difference of the angle): per token class (each move, observation, blank) the
share of |increment| mass that is negative; wrap-aware cancellation of opposite moves (pairs (0,1), (2,3): over channels with
|wrap(mean step)| > 0.05, share with |wrap(mean a + mean b)| < 0.1 rad -- GENERALITY's `wrap_share` on empirical means); context
dependence (median std / |mean| of a move's increment across positions: 0 for a context-free step, the conv makes it non-zero);
observation-to-move step ratio; per-head effective rank (participation ratio) and top-2 energy of the 4 x 32 matrix of mean
move increments. Index RoPE has no content angle and is skipped.

## Floors (this batch's own evaluation stream, seeds 1700-1707, before any run; `docs/audits/2026-10-16/sign_floors.py`)
`docs/audits/2026-10-20/srope_torus_floors_out.txt`: T = 1024: always-blank 0.5087, best n-gram (orders 0-5, fitted on 16
training maps) 0.5779, retrace 0.8454 (best trivial predictor; range over seeds 0.840-0.851), wrap-only share of revisits 0.080.
T = 2048: blank 0.5086, n-gram 0.5663, retrace 0.7913, wrap-only 0.186. Every arm is reported beside retrace. (GENERALITY's index
RoPE, 0.722, sat BELOW the retrace floor: index attention at one layer does not even implement retrace on this task.)

## Predictions (fixed before data)
- **S1: LEARNS THE MAP** (signed, per-head rank 32 > D = 2). Confidence moderate, not high: the obstacles are learnability ones
  (all are representable: a lag-0 conv tap, zero observation steps or a closed gate, |steps| below SiLU's floor or wrapping),
  and the authors' init starts with near-zero steps (conv and gate N(0, 0.02): ~1e-3 rad per token on the top channel), so the
  model must grow its phase from nothing. MapWM starts at ~0.5 rad.
- **S2: BEATS INDEX.**
- SiLU: **NO DIFFERENCE SHOWN** (the top channels carry the map with steps ~2 pi k / 64 <= 0.278 x temperature, inside SiLU's
  negative range).
- SRoPEGen vs MapWM: **LEARNS THE MAP** (it solved T = 128 in MONOTONE; at T = 1024 it is untested).
- Angle statistics, if S1 LEARNS: wrap-aware cancel share > 0.8 in the channels that move; observation / move step ratio < 0.2;
  context CV < 0.2 (the conv learns a near-lag-0 kernel); effective rank of the move map ~2 per head (top-2 energy > 0.9).

## What this can and cannot show
It tests one published content phase in one configuration (the authors' best language-model one, plus its no-SiLU row) on one
task, one size (1 layer, 2 heads, d 128), one recipe (GENERALITY's), one budget (900 epochs). A LEARNS reading adds a third
published method to the theory's positive side and shows Selective RoPE's conv / gate / SiLU / spectrum do not obstruct the map
at this budget; it does not show they are harmless elsewhere. A FAILS reading is a learnability result for this configuration
in MapFormer's scaffold: it does not separate the four ingredients (the NoSiLU arm separates one), and the language-model
spectrum is the first suspect. The scaffold is MapFormer's, not the authors' transformer; the parity of the comparison (same
scaffold, recipe, stream for every arm) is what the batch controls. Budget-scoped throughout. Eval-mode accuracy under-reports
one-layer runs below ceiling (`DROPOUT_RESCORE.md`) for every arm alike; SOLVED does not depend on it.

## Process, code and cost
- **Files.** `model_srope.py` (new), `train_variant.py` (+2 `VARIANT_MAP` entries, `SRoPE`, `SRoPE_NoSiLU`; nothing else),
  `analyze_srope_torus.py`, `run_srope_torus.sh` (sources `lib_driver.sh`), this file. `model_selective.py` unchanged.
- **Bitwise default check** (rule: no default changed). `Signed_r4`, seed 1789, the batch's flags plus `--deterministic`, 3
  epochs x 98 batches, cuda:1: HEAD a83e24a (a `git archive` copy) vs the edited tree -- per-epoch losses identical
  (2.146843180364492, 1.9537722261584536, 1.7583196771388152) and all 24 state-dict tensors `torch.equal`.
- **`--fast-attn`** is valid for every arm: all five use the plain `model.WMTransformerLayer` (the trainer's check passes), and
  eval-mode logits with SDPA vs the explicit path agree to <= 1.9e-6 for SRoPE, SRoPE_NoSiLU, SRoPEGen, Signed_r4 and RoPE (CPU,
  T = 256 walks, perturbed weights). Not bit-identical in training (dropout RNG): a new series only, as in GENERALITY.
- **Driver** `run_srope_torus.sh`: `flock` lock; refuses a dirty tree (tracked files) and uncommitted `analyze_srope_torus.py` /
  `run_srope_torus.sh` / `model_srope.py` / this file BEFORE any wait; md5 guard over every `mapformer` module the trainer,
  evaluators and analysis import (from `sys.modules`: 95 files incl. `lib_driver.sh` and the driver), re-checked before the
  evaluation; seed-outer, arm-inner launches; at most 4 of our trainers per GPU (`DRV_MAXPG=4`: the policy's cap on a GPU shared
  with another user's job, and GENERALITY's normal concurrency for this recipe, so the cap is the same whether or not the GPU is
  shared); `DRV_MINFREE` 6000 MiB; evaluations on the first GPU of `DRV_GPUS` after every checkpoint exists (48 required, counted);
  analysis on CPU; artifacts-then-marker (`.srope_torus_done` after `SROPE_TORUS_ANALYSIS.txt` and `SROPE_TORUS_VERDICTS.json`).
  `MODE=smoke` runs the same command lines on CPU (2 seeds, 2 epochs x 2 batches, 3 eval trials) into a scratch dir.
- **Smoke** (`MODE=smoke`, `scratchpad/srope_smoke`, seeds 1780-1781): all 10 runs train on CPU, all 5 arms evaluate (noise
  refine T 1024 / 2048 and strata, strata 'all' checked against the former), `analyze_srope_torus` reads the tree end to end
  (UNREADABLE at n = 2, as it must), marker set after the artifacts. **Labels:** `--selftest` reaches every S1 label (LEARNS,
  FAILS by Fisher, FAILS by accuracy, UNMEASURED, UNREADABLE, LEARNS when ahead), every S2 label (incl. BEATS by Fisher), every
  SiLU label and every reading; two synthetic 8-seed trees built from the smoke checkpoints (rewritten seeds, losses,
  accuracies) read through the real file path: scenario A -> S1 LEARNS THE MAP, S2 BEATS INDEX, reading LEARNS; scenario B ->
  S1 FAILS, S2 BEATS INDEX, reading FAILS BUT USES ITS CONTENT PHASE, SiLU COSTS, SRoPEGen WORSE THAN INDEX.
- **Cost** (measured 2026-10-20 14:38-14:45; both RTX 4090s carry another user's job at ~100% utilisation, 7-9 GB): one of our
  runs beside it 2.5-3.0 s/epoch; four of ours on one GPU 4.1-5.1 s/epoch each (SRoPEGen / Signed_r4 / RoPE probes, 15 epochs,
  seeds 1791-1793, beside the SRoPE pilot). At 4 per GPU on 2 GPUs: 8 concurrent runs x 900 epochs x ~4.6 s = ~69 min per wave,
  5 waves = **~5.8 h training**, evaluations ~20 min, analysis ~5 min: **~6-6.5 h wall** with the other user's jobs running
  (GENERALITY's runs took 4-6 s/epoch at 4 per GPU, consistent). If the other user's jobs end, ~4.5-5 h.
  Trimmed alternative: drop SRoPEGen (secondary, non-faithful; 32 runs, ~4.8 h).

## Pilot (outside seeds, never pooled)
Seed 1790, the driver's exact command (900 epochs), one run per GPU beside the other user's job, 2026-10-20 14:38-15:20
(`runs/srope_torus_pilot/`); evaluated with the batch's evaluators (100 walks) and the analysis' angle / zero-angle functions
(`PILOT_*.json`, `PILOT_ANGLES.txt`). Own-stream floors for seed 1790: blank 0.5085, n-gram 0.5791, retrace 0.8415 (T 1024);
retrace 0.7889 (T 2048). n = 1: no estimate, no registered decision changes.

| | SRoPE (best LM config) | SRoPE_NoSiLU |
|---|---|---|
| run class (final 5% loss) | DESCENDING (0.268; loss 0.67 / 0.47 / 0.56 / 0.39 / 0.27 at epochs 100 / 200 / 300 / 500 / 900) | **SOLVED** (0.015; < 0.05 from epoch 595) |
| accuracy T 1024 / 2048 | 0.929 / 0.752 | 0.9985 / 0.983 |
| strata lag < 128 / >= 128 / wrap-only | 0.980 / 0.835 / 0.504 (floor 0.495) | 1.000 / 0.988 / 0.996 |
| zero-angle accuracy (T 1024) | 0.503 | 0.508 |
| negative-mass share, moves 0-3 | 0.50 / 0.69 / 0.64 / 0.46 | 0.12 / 0.93 / 0.92 / 0.11 |
| cancel share pair / token-class | 0.75 / 0.71 | 0.87 / 1.00 |
| context CV; obs / move step ratio | 0.058; 0.99 | 0.18; 0.86 |
| pair move-map participation ratio (top-2 energy) per head | 1.68 (0.98), 2.02 (0.94) | 2.02 (0.99), 1.97 (1.00) |
| g per head; mean conv |tap| lag 3..0 | 0.85, 1.14; 0.10 / 0.11 / 0.17 / 0.16 | 1.01, 0.90; 0.13 / 0.15 / 0.19 / 0.19 |

Reading of one seed, descriptive only: both arms use their phase (zeroing it drops each to the blank floor). Without SiLU the
step map is the signed, opposite-cancelling, rank-2 code the theory describes (moves 0/1 and 2/3 carry opposite signs, 0.12 vs
0.93 negative mass) and the run solves, wrap-only revisits included. With SiLU the run had not solved by epoch 900: it is above
retrace on short and long-gap revisits but at the floor on wrap-only revisits, and its moves carry mixed signs. The
observation / move step ratio near 1 in both is the conv spreading each move's step over the next tokens (taps of similar size
at all four lags), which is why the pair versions were added. **Risk for the registered primary:** if the pilot is typical, S1
reads FAILS by the accuracy route (0.929 vs MapWM ~1.000) or UNMEASURED, and a FAILS here would be budget-scoped (the run was
still DESCENDING). No change is made on the strength of one seed; the batch keeps the registered 900-epoch budget, the same as
every other arm. If the user wants the budget question settled in the same batch, the option is to also run SRoPE and
SRoPE_NoSiLU at 1800 epochs (16 more runs, ~4.6 h more); not registered here.

## Deviations from the brief (each with its reason)
1. The brief's "omega = W_omega @ q (or the l2-normalised input x)": the faithful arm uses x (the layer's normed input, l2-normalised
   per head chunk), because the authors' released code does, at every configuration, and Table 2 names it the stable choice.
2. The brief named weight norm "on the projection": in the authors' code it is a per-HEAD scale on column-normalised per-head
   weights (`ChunkedLinear`), not PyTorch's per-output `weight_norm`; implemented as theirs.
3. The gate is per head (their `standard` parametrisation, used by the best config), not per frequency.
4. Temperature: the brief listed "temperature"; the authors' best config fixes a per-frequency vector (RoPE-type, theta 500000),
   not a learned scalar; implemented as theirs. The paper's Appendix A.3.1 tan(phi/2) schedule ("rfa") is in their code but not
   in any config we found; not used. [CORRECTED, Amendment 1: `configs/copying/copying.yaml` uses it (rfa, theta 0.9999, temp_max 0.25).]
5. Arms: SRoPEGen kept (it differs materially, audit above). Considered and NOT added: the authors' state-tracking config
   (`configs/state_tracking/state_tracking.yaml`: low-rank 32 shared across heads, conv without SiLU, no gate, bias exp(.) on,
   all temperatures 1.0) -- it is their GLA sweep for S2 parity, not a softmax-transformer best config, and it would add 8 runs
   (~1.2 h); named as the first follow-up if S1 reads FAILS, together with a torus-scaled spectrum. [SUPERSEDED, Amendment 1:
   the arm is added (SRoPE_ST); the reason given here was wrong, see there.]
6. S2 and the reading branch on S2 were added so that a FAILS has a defined meaning (fails the map but uses content, vs no
   better than index); the brief called S2 a gate.

## Amendment 1 (2026-10-20, after the independent check; before launch, no batch result exists)
Written while the pilot was still training (epochs ~600-850, nothing evaluated); the prediction for SRoPE_ST at the end was
added after the pilot section, and says so.
The rule-29 check (a separate agent, blind to every result incl. the pilot; it re-ran the official-module equivalence, the
selftest, the parameter counts, the scaffold-init equality and a gradient-reaches-every-parameter check) found no code bug that
would make a registered verdict wrong, and five text / scope problems. Changes:
- **MAJOR 1 (scope of "faithful").** The language-model configuration is the authors' GLA sweep file
  (`configs/language_modeling/language_modeling.yaml`, `defaults: model: gla`); the paper's Table 2 ablations are on GLA, and the
  paper says the same best configuration is used for FoX (a softmax-attention Forgetting Transformer) in Table 3. Deviation 5's
  reason for excluding the state-tracking configuration ("a GLA sweep, not a softmax-transformer config") therefore applied to
  the LM config as well, and the state-tracking setting is the one where the paper reports a 1-layer softmax Transformer with
  Selective RoPE (Fig. 7; `configs/state_tracking/model/attn.yaml` has 1 layer). The theta-500000 spectrum, named here as the
  first suspect for a FAILS, appears only in the LM config; the synthetic-task configs use spectra where every channel is
  substantial (state tracking: all temperatures 1.0; copying: rfa, temp_max 0.25). Changes: (a) **arm SRoPE_ST added**
  (`model_srope.MapFormerWM_SRoPE_ST`, variant `SRoPE_ST`): low-rank 32 projection shared across heads (their `low-rank`
  option, N(0, 0.02) init), conv k = 4 WITHOUT SiLU, no gate, learnable bias exp(phi_bias) (+1 rad per token per channel at
  init: an index-like drift the content step must cancel), temperature 1.0 in every channel; per-head rank of the content map
  <= 32. Verified against the authors' module (`scratchpad/tests/verify_srope_st.py`: rotated q, k agree to 1.4e-5 at angles up
  to 68 rad, fp32). SRoPE_ST has 210,389 params (vocab 21; measured in the smoke). Secondary, labelled by the S1 rule
  (vs MapWM) and the S2 rule (vs index RoPE). Cost +8 runs (~1.2 h). (b) The primary stays the best-LM-configuration arm (the
  brief fixed it), and **every reading now names its scope**: "SELECTIVE ROPE (BEST LM CONFIG) ...". (c) Deviation 4 corrected
  (rfa is used by the copying config; not adopted: copying is not a state-tracking task and its config is the GLA sweep).
- **MINOR 2.** SRoPE_NoSiLU is a single-knob arm; its attribution to a Table 2 row is withdrawn (the rows may be cumulative).
- **MINOR 3 (what "uses its content phase" means).** Beating index RoPE (0.722, below the retrace floor) does not show that the
  phase is used: an SRoPE with near-zero steps or a closed gate is NoPE, and this repo has no NoPE number on this torus at T =
  1024. Added, eval-only (no training cost): **phase ablation** -- each trained SRoPE-family / SRoPEGen / MapWM model is re-scored
  with its rotary angle forced to 0 (SRoPE*: temperature 0; SRoPEGen: exp(log_temp) = 0; MapWM: omega 0) on eval_noise_refine's
  exact walks (held-out map, env seed 10000, np seed 1234 + s, 100 walks, T = 1024; CPU). **USES ITS PHASE** if mean(ablated -
  intact) <= -0.05 with sign-flip p < .05 (paired over the 8 seeds; min p 0.0078), else PHASE NOT SHOWN USED. The reading's
  S1 = FAILS & S2 = BEATS INDEX branch now splits: **FAILS THE MAP BUT USES ITS CONTENT PHASE** (phase used) / **FAILS THE MAP;
  IT BEATS INDEX ROPE WITHOUT A SHOWN USE OF ITS PHASE**. Other branches are unchanged. Reported for every content arm
  (secondary for all but SRoPE).
- **MINOR 4 (label order).** S2 and SiLU labels are evaluated in the order written (BEATS before WORSE; COSTS before HELPS);
  a simultaneous permutation-one-way / Fisher-other-way conflict is practically unreachable and resolves to the first.
  "Fisher p >= 0.47" corrected to 0.467.
- **MINOR 5 (descriptive angle readouts).** neg_share's channel cutoff (mean |increment| at that class > 1e-3 rad) is now
  stated. The participation ratio is now the standard (sum s^2)^2 / sum s^4 (it was (sum s)^2 / sum s^2). The cancellation
  share and the effective rank are now computed on **pair increments** (theta at observation k minus theta at observation k-1,
  keyed by action k), robust to the conv placing a move's step one token late; the token-class cancellation is kept beside it
  (`cancel_tok`). The effective rank is on raw (not wrapped) means and says so: a wrapping solution reads higher. "~5 of 32
  channels" corrected to 6.
- **MINOR 6 (GPU sharing).** The brief's policy (user's instruction relayed by the launching session) allows up to 4 of our
  runs on a GPU shared with another user's job; the memory (`feedback_gpu_sharing_and_cost.md`) asks for the user's go-ahead
  before launching beside another user. The batch is NOT launched by this preparation; the launch decision is the user's.
- Re-verified after the changes: `--selftest` reaches every S1 / S2 / SiLU / phase label and every reading (incl. both split
  FAILS branches); a second smoke (`scratchpad/srope_smoke2`, 6 arms x 2 seeds, CPU) ran end to end and set its marker; the two
  synthetic 8-seed trees read through the real file path give A: S1 LEARNS, S2 BEATS, phase USES, reading LEARNS; B: S1 FAILS,
  S2 BEATS, phase USES, reading FAILS BUT USES ITS CONTENT PHASE (the "without a shown use" branch is reached in the selftest);
  SRoPE_ST reads UNMEASURED / FAILS vs MapWM and BEATS vs index in A / B.
- Cost with SRoPE_ST: 48 runs, 6 waves of 8 at ~69 min = **~6.9 h training**, evaluations ~25 min, analysis (now incl. the CPU
  phase ablation, 6 x 8 x 100 forwards) ~15 min: **~7.5 h wall** beside the other user's jobs. Trimmed: drop SRoPEGen (40 runs,
  ~6.3 h).
- **Prediction for SRoPE_ST** (written AFTER the pilot of the other two arms had been read): vs MapWM **UNMEASURED or LEARNS**
  (signed, no SiLU, all channels at temperature 1, but the +1 rad per token bias must be cancelled by the content step on
  every token: a learnability load the other arms do not carry); vs index **BEATS INDEX**; phase **USES ITS PHASE**.

