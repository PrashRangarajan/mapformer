# Does faithful Selective RoPE learn the torus map? -- results (registered, `SROPE_TORUS_PREREG.md` + Amendment 1)

Batch 2026-10-20 16:02 -> 10-21 07:54 on GPU 1 (shared with another user, <= 4 runs), 48 runs, seeds 1700-1707 (fresh).
Torus 64 x 64, T = 1024, 900 epochs, GENERALITY's torus recipe, 1 layer, 2 heads, d 128. Arms: SRoPE (faithful Selective
RoPE, Movahedi et al. ICLR 2026, the authors' best language-model config, module reproduced from their code to 6e-6),
SRoPE_NoSiLU (the same without the SiLU), SRoPE_ST (the authors' state-tracking config: low-rank 32, no SiLU, no gate, a
learnable bias of +1 rad per token per channel at init, temperature 1), SRoPEGen (our earlier generator swap), MapWM
(Signed_r4, in-batch positive reference), RoPE (index, negative reference). Independent check before launch (Amendment 1);
pilot on seed 1790. Analysis `analyze_srope_torus.py` -> `SROPE_TORUS_ANALYSIS.txt`, `SROPE_TORUS_VERDICTS.json`.
`experiment_audit`: windowed classes agree. Own-stream floors: blank 0.509, n-gram 0.578, retrace 0.845.

**Reading (registered): SELECTIVE ROPE (BEST LM CONFIG) FAILS THE MAP BUT USES ITS CONTENT PHASE** -- it beats index
RoPE by +0.236 and zeroing its angle costs 0.454, but it solves 0/8 against MapWM's 8/8 within 900 epochs. Every SRoPE run
was still DESCENDING at 900 epochs, so the failure is budget-scoped.

| arm | accuracy T = 1024 | SOLVED | classes | short / long / wrap-only revisits | T = 2048 |
|---|---|---|---|---|---|
| MapWM (signed, rank 4) | 1.000 | 8/8 | all SOLVED | 1.000 / 1.000 / 0.999 | 0.992 |
| SRoPEGen (earlier swap) | 0.985 | 5/8 | 5 SOLVED, 2 DESC, 1 STAL | 0.997 / 0.975 / 0.865 | 0.951 |
| SRoPE_NoSiLU | 0.969 | 2/8 | 2 SOLVED, 6 DESC | 0.987 / 0.952 / 0.805 | 0.861 |
| **SRoPE (faithful, best LM config)** | **0.961** | **0/8** | all DESCENDING | 0.992 / 0.937 / 0.659 | 0.807 |
| SRoPE_ST (state-tracking config) | 0.769 | 0/8 | 7 STAL, 1 DESC | 0.788 / 0.705 / 0.628 | 0.724 |
| index RoPE | 0.725 | 0/8 | all STALLED | 0.775 / 0.470 / 0.465 | 0.561 |

## Registered decisions
- **S1 SRoPE vs MapWM: FAILS** (0.961 vs 1.000, d -0.039 [-0.050, -0.030], perm p 0.0002; 0/8 vs 8/8, Fisher p 0.0002).
- **S2 SRoPE vs index RoPE: BEATS INDEX** (+0.236, p 0.0002).
- **Phase check: USES ITS PHASE** (angle set to zero: 0.961 -> 0.507, p 0.0078).

## Declared secondaries (no family-wise correction; not part of the reading)
- **SiLU:** no difference shown (NoSiLU 0.969, 2/8 vs SRoPE 0.961, 0/8; p 0.57). NoSiLU vs MapWM also FAILS (2/8, p 0.007).
- **SRoPEGen vs MapWM:** UNMEASURED (5/8 vs 8/8, Fisher p 0.20; accuracy -0.015).
- **SRoPE_ST vs MapWM: FAILS** (0.769, 0/8); vs index RoPE NOT DISTINGUISHED (+0.044, p 0.23).
- Every arm uses its phase (zeroing it drops each to the floor).

## What the learned angles show (declared secondary readouts, descriptive)
- **Observations move the phase.** The ratio of an observation's step to a move's step is 0.027 in MapWM and 0.86-0.96 in
  SRoPE / NoSiLU (0.45 in SRoPEGen): Selective RoPE reads its angle from the query of every token, so what is seen moves
  where the model thinks it is -- the what-to-where leak, at full size.
- **Partial cancellation.** Opposite moves cancel (wrap-aware) in 0.75-0.80 of channel pairs for SRoPE / NoSiLU against 1.00
  for MapWM; the angles also vary with context (context CV 0.11-0.13, MapWM 0.00) through the causal convolution.
- **The state-tracking config never takes a negative step.** Its learned per-token angles have negative-mass share 0.00 for
  every token class and cancel in 0.00 of channels: the +1 rad bias dominates the content step, so the step is effectively
  non-negative -- and the arm fails like the non-negative parameterisations (`SIGN_MATCHED_RESULTS.md`).
- Threshold sensitivity: SRoPE vs MapWM still 0/8 vs 8/8 at a loss threshold of 0.1.

## Reading for the paper
Selective RoPE belongs to the signed, content-dependent family and does build a phase it relies on (it beats index RoPE by
+0.24), but in its published configurations it learns the torus map more slowly or not at all within the registered
budget: the best-LM config solves 0/8 (all still descending), the state-tracking config 0/8. Two ingredients separate it
from MapFormer in the readouts: every token, not only the moves, steps the phase (observations step 0.9x as far as moves),
and the angle depends on context through a convolution. The state-tracking config's bias makes every step positive and it
fails like a non-negative step. Scope: one layer, 2 heads, the torus at T = 1024, 900 epochs; mechanism readouts
descriptive.
