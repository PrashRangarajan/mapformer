# CLAUDE.md — Project Memory for MapFormer

**Purpose of this file:** concise context for Claude when resuming work on this
project in a fresh session. Read the README for the full picture; this file
focuses on state + lessons learned + what to do next.

## Project in one sentence

Faithful reproduction of Rambaud et al. (2025) *MapFormer* (arXiv:2511.19279)
plus three experimental extensions that add explicit state-correction
mechanisms to the path-integration circuit: a **parallel Invariant EKF**, a
**sequential InEKF** (for reference), and a **predictive-coding** variant.

## Current state (what's implemented and working)

1. **Paper reproduction** — `model.py`, `environment.py`, `main.py`.
   MapFormer-WM / EM reach paper-level accuracy (0.955 / 0.999) on 200K
   sequences at the paper's exact hyperparameters (Appendix B). Checkpoints
   in `figures_v6/`.
2. **Parallel InEKF** — `model_inekf_parallel.py`, `main_inekf_parallel.py`.
   Steady-state gain + FFT scan. Same speed as vanilla. Checkpoint in
   `figures_inekf_parallel_v2/`.
3. **Sequential InEKF (wrapped)** — `model_inekf_proper.py`,
   `main_inekf_proper.py`. ~2.5× slower but same final accuracy as parallel.
   Checkpoint in `figures_inekf_topology_fix/`.
4. **Predictive-Coding MapFormer** — `model_predictive_coding.py`,
   `main_predictive_coding.py`. Forward model + error-driven corrections.
   Checkpoint in `figures_predictive_coding/` after training completes.
5. **Evaluation tools** — `noise_test.py`, `gaussian_noise_test.py`,
   `diagnose.py`. Handle all variants above.

## Things that are true (verified) and must be preserved

**Paper-faithfulness invariants** (breaking any of these regresses to
broken states we debugged through):

- `environment.py`: torus grid (`(x+dx) % N`), interleaved token stream
  `[a1, o1, a2, o2, …]`, `revisit_mask` returned per trajectory
- `train.py`: loss masked to **revisited** observation positions only
- `model.py::PathIntegrator`: ω initialized monotonically decreasing in `i`
  (paper eq. 17 has a sign typo; the correct formula is
  `ω_i = ω_max · (1/Δ_max)^(i/(n_b-1))`)
- `model.py::MapFormerEM`: attention is Hadamard product `A_X ⊙ A_P`, not
  additive; uses separate learnable `q_0^p` and `k_0^p`
- `model.py::ActionToLieAlgebra`: low-rank factorization `W_out · W_in`
  with bottleneck `r=2`
- `model.py`: path integration via cumsum of angles, not prefix product of
  rotation matrices
- Default hyperparameters: 1 layer, 2 heads, h=64, d_model=128, lr=3e-4,
  AdamW, wd=0.05, linear LR decay, batch 128, grid 64, T=128 steps,
  K=16 obs types, p_empty=0.5, 200K sequences

## Architectural choices that matter

- **Feed `content_emb` only to the InEKF measurement head.** Adding
  position features `(cos θ, sin θ)` creates a degenerate optimum
  `z ≈ θ` → zero innovation → filter does nothing. We learned this
  the hard way.
- **Wrap innovations modulo 2π** via `atan2(sin(z - θ̂), cos(z - θ̂))`.
  Without this, length generalization breaks (θ̂ grows unboundedly, bounded
  z can't express the large "error" geometrically).
- **Steady-state Kalman gain from closed-form scalar DARE.** Enables
  FFT-conv based parallel affine scan, preserving MapFormer's O(log T)
  property.
- **Markovic et al. (2017) proves:** on SO(2), wrapped-innovation EKF equals
  Lie-Group EKF. So the simple wrapping *is* the correct Lie-group filter.
- **Predictive coding uses a forward model** `g(cos θ, sin θ) → ô` and
  computes error in *embedding space*, masked at observation positions only
  (action positions have unpredictable content conditioned on θ alone).
  Includes an auxiliary loss coefficient to force the forward model to
  actually model observations.

## Landmark experiment (added in latest session)

Added `n_landmarks` parameter to `environment.py`: sets N cells to emit
unique single-use tokens instead of aliased obs. Retrained all three
variants with 200 landmarks (~5% density).

Ran `landmark_eval.py` at T=128 and T=512, accuracy + NLL per cell type
(landmark / regular / blank):

- **At T=128:** PC best overall (87.7% acc, 0.591 NLL). But only InEKF
  predicts landmarks well (18% vs 1.5% for others).
- **At T=512:** InEKF is decisively best (78.5% vs 64% vanilla, 62% PC).
  Degrades only -7pp from T=128 to T=512 vs -18 to -26 for the others.

**Key finding: the three architectures are complementary, not alternatives:**
- Vanilla attention → clean aliased tasks
- PC MapFormer → matched-noise drift correction on aliased obs (best at
  training length)
- Parallel InEKF → true landmarks + long OOD (bounded-error stability)

This is the regime where Kalman filtering earns its theoretical guarantees
empirically. A 15pp overall accuracy gap at T=512 with landmarks.

Checkpoints:
- `figures_vanilla_noise_lm200/MapFormer_WM_noise.pt`
- `figures_inekf_parallel_lm200/MapFormer_WM_ParallelInEKF.pt`
- `figures_pc_lm200/MapFormer_WM_PredictiveCoding.pt`

## Clone-structure analysis (added in latest session)

`clone_analysis.py` runs 300 trajectories from a fixed start, records model
state at observation positions, and measures per-obs-type separation of
(x, y) cells in feature space (two metrics: linear-regression R² and
cosine-distance separation score).

Result:
- **PC MapFormer has the best θ̂ separation score (0.619 vs 0.573 vs 0.395).**
  Its prediction-error correction mechanism most cleanly clusters per-cell
  representations — closest to the CSCG (Clone-Structured Cognitive Graph)
  hypothesis from neuroscience.
- InEKF has more continuous (higher R²) but less clustered θ̂.
- Hidden features are similar across all models (attention blends position
  and content uniformly).

## Main empirical finding

On the paper's aliased-observation task, **vanilla attention + noise
augmentation beats all Kalman-style variants on raw next-token prediction**.
Reasons, in order of importance:

1. Attention already implements soft associative retrieval — implicit
   Bayesian pattern completion — which is what the Kalman update was meant
   to add.
2. Aliased observations (16 obs types / 4096 cells = ~128 cells per type)
   mean Kalman measurements can't produce sharp corrections. The Gaussian
   assumption of Kalman is violated; the true posterior is multimodal.
3. Innovation wrapping, required for length generalization, slows training
   by bounding per-step corrections.

**Where the Kalman / PC framework should win:**

- Tasks with **true landmarks** (5% of cells emit unique IDs found nowhere
  else). Not yet tested. Predicted to be where Kalman dominates.
- Very long sequences (T >> 2048) where attention becomes infeasible.
- Scenarios needing calibrated uncertainty (InEKF tracks σ², attention
  doesn't).
- External sensor fusion with known Q, R matrices.

## Things that didn't work / why

- **Uncertainty-modulated attention** (`model_kalman.py`, the first-pass
  InEKF): redundant with softmax attention's natural behavior. Kept for
  reference comparison.
- **Unwrapped InEKF innovations**: trained faster (0.66 vs 0.88 final loss)
  but broke at T=512 OOD — the measurement head extrapolated badly outside
  the short-sequence θ range it was trained on.
- **Adding position features to InEKF head**: degenerate optimum — the
  filter turns into the identity function.
- **Multi-layer MapFormer with shared θ**: not tested empirically. The
  paper only runs 1-layer MapFormer. If you need more layers for position
  correction, Option 1 (per-layer θ correction) is the natural extension
  but requires validation that multi-layer MapFormer trains stably first.

## Open questions / natural next experiments

1. **Evaluate the predictive-coding variant** against InEKF + vanilla on
   `gaussian_noise_test.py` at T=128 and T=512, noise_std 0.00 – 1.00.
   (This is running as of the last commit — check
   `figures_predictive_coding/`.)
2. **Add true landmarks** to `environment.py`: reserve ~5% of cells to
   emit unique high-info tokens (beyond the 16 standard types). Retrain +
   compare. This is the Kalman framework's home turf.
3. **Level 2 InEKF** (time-varying R_t from heteroscedastic head,
   parallelisable via Möbius-matrix associative scan). Theoretical sketch
   lives in the chat transcript; not yet implemented.
4. **Calibration metrics.** NLL or ECE would show whether InEKF's tracked
   σ² is a useful confidence estimate even when point accuracy doesn't
   improve.
5. **Multi-layer MapFormer ablation.** Paper only runs 1 layer. Test 2/3/4
   layers at vanilla + InEKF-augmented configurations to see whether depth
   helps at all in this architecture.
6. **Scaling.** Paper acknowledges they didn't scale model/data. 4 layers,
   4 heads, d=256, 10M sequences would be a natural next step.

## Quick reproducibility commands

```bash
# Re-verify paper reproduction:
python3 -m mapformer.main --device cuda --epochs 16 --n-batches 98

# Train each variant under 10% action-noise augmentation:
python3 -m mapformer.main_vanilla_noise --device cuda --epochs 50 --n-batches 156
python3 -m mapformer.main_inekf_parallel --device cuda --epochs 50 --n-batches 156 \
  --p-action-noise 0.10
python3 -m mapformer.main_predictive_coding --device cuda --epochs 50 --n-batches 156 \
  --p-action-noise 0.10 --aux-coef 0.1

# Head-to-head evaluation under Gaussian Δ noise:
python3 -m mapformer.gaussian_noise_test \
    --checkpoints \
      figures_v6/MapFormer_WM.pt \
      figures_vanilla_noise/MapFormer_WM_noise.pt \
      figures_inekf_parallel_v2/MapFormer_WM_ParallelInEKF.pt \
      figures_predictive_coding/MapFormer_WM_PredictiveCoding.pt \
    --device cuda --n-steps 128 --n-trials 200
# Then with --n-steps 512 for OOD length.

# Diagnostics on any trained model:
python3 -m mapformer.diagnose --checkpoint figures_v6/MapFormer_WM.pt --device cuda
```

## Filesystem map

- `figures_v6/` — paper-faithful MapFormer-WM and EM (reference)
- `figures_vanilla_noise/` — vanilla + noise-aug baseline
- `figures_inekf_parallel_v2/` — parallel InEKF main result
- `figures_inekf_topology_fix/` — sequential wrapped InEKF (same model class)
- `figures_inekf_proper/` — **stale**: sequential unwrapped InEKF
  (model code has since been updated; loading this checkpoint with current
  code is incorrect)
- `figures_kalman/` — first-pass fake InEKF (kept for comparison)
- `figures_predictive_coding/` — PC MapFormer (being populated)
- `figures_2M/`, `figures_constlr/`, `figures_v3/`, `figures_v4*/` — older
  runs from earlier debug sessions; can safely be deleted

## Authoring style / preferences

- No emojis in source code or commit messages
- No `Co-Authored-By` lines; single-author commits
- README is the primary documentation, this file is a memory-aid for Claude
- Honest reporting: if an experiment didn't work, write that down with the
  reason, don't bury it

## Level 2 InEKF results (autonomous addition)

Level 2 (heteroscedastic R_t) training completed. See RESULTS_LEVEL2.md
for full evaluation (per-cell-type accuracy, NLL, robustness, R_t / K_t
distribution by token category).

Checkpoints:
- figures_inekf_level2_lm200/MapFormer_WM_Level2InEKF.pt (with landmarks)
- figures_inekf_level2/MapFormer_WM_Level2InEKF.pt (no landmarks)

## Level 1.5 InEKF (compromise between Level 1 and Level 2)

Level 1.5 = constant Pi (learnable scalar, not DARE-derived) + per-token R_t.
Key insight: Level 2 diagnostic showed Pi only varied ~4x across tokens;
replacing Pi dynamics with a constant while keeping R_t/K_t dynamics
should recover most of Level 2's benefit at Level 1's cost.

Empirically: 60x faster training than Level 2, best landmark-training loss
of any variant (0.8124 vs Level 2's 1.133). Result in RESULTS_LEVEL15.md.

## Open for next session: matched-compute vanilla retest

**What to verify when resuming:**
- Whether Level 1.5's clean-task advantage over vanilla MapFormer is
  architectural or just due to more training compute.
- Vanilla paper-repro (`figures_v6/`) trained 200K sequences × 16 epochs.
- Level 1.5 clean (`figures_inekf_level15_clean/`) trained 1M sequences
  × 50 epochs (5× more compute).

**Running in the background (launched end of last session):**
- `figures_vanilla_50ep/MapFormer_WM.pt` — vanilla MapFormer at 50 epochs
  clean training (matched compute to Level 1.5 clean).
- Training script: `main.py --epochs 50 --n-batches 156 --output-dir figures_vanilla_50ep`
- PID was 3318458 when launched. Check process or checkpoint file to
  confirm completion. Logs at `/home/prashr/mapformer/vanilla_50ep_run.log`.

**When it's done — compare:**
```bash
python3 -u << 'PYEOF'
import torch, torch.nn.functional as F
from mapformer.environment import GridWorld
from mapformer.model import MapFormerWM
from mapformer.model_inekf_level15 import MapFormerWM_Level15InEKF

env = GridWorld(size=64, n_obs_types=16, p_empty=0.5, n_landmarks=0, seed=42)
def build(path, cls):
    m = cls(vocab_size=env.unified_vocab_size, d_model=128, n_heads=2, n_layers=1, grid_size=64)
    m.load_state_dict(torch.load(path, map_location="cuda", weights_only=False)["model_state_dict"])
    return m.cuda().eval()

vanilla_16ep = build("mapformer/figures_v6/MapFormer_WM.pt", MapFormerWM)
vanilla_50ep = build("mapformer/figures_vanilla_50ep/MapFormer_WM.pt", MapFormerWM)
l15_clean    = build("mapformer/figures_inekf_level15_clean/MapFormer_WM_Level15InEKF.pt", MapFormerWM_Level15InEKF)

def eval_m(model, T, n_trials=300, seed=0):
    import numpy as np; torch.manual_seed(seed); np.random.seed(seed)
    c=tot=0; nll=0.0
    with torch.no_grad():
        for _ in range(n_trials):
            tokens, om, rm = env.generate_trajectory(T)
            tt = tokens.unsqueeze(0).cuda()
            logits = model(tt[:,:-1])
            lp = F.log_softmax(logits, dim=-1)
            preds = lp.argmax(-1)[0]; tgts = tt[0,1:]; mask = rm[1:].cuda()
            if mask.sum()==0: continue
            c += (preds[mask]==tgts[mask]).sum().item(); tot += mask.sum().item()
            nll += -lp[0, torch.arange(lp.shape[1], device="cuda")[mask], tgts[mask]].sum().item()
    return c/tot, nll/tot

for T in [128, 256, 512, 1024]:
    print(f"T={T}")
    for name, m in [("vanilla 16ep", vanilla_16ep), ("vanilla 50ep", vanilla_50ep), ("L1.5 clean", l15_clean)]:
        a, nll = eval_m(m, T)
        print(f"  {name:>14s}: acc={a:.4f}  nll={nll:.3f}")
PYEOF
```

**Expected outcomes and interpretation:**
- If vanilla 50ep matches L1.5 on accuracy: the clean-task accuracy gap
  was training-compute, not architecture. Keep L1.5's NLL/calibration
  win + its landmark + noise robustness as the real architectural
  contribution.
- If L1.5 still wins on accuracy even at matched compute: Level 1.5 is
  architecturally strictly superior on the clean task too.
- Either way, L1.5 is the recommended drop-in replacement for MapFormer
  given its advantages on noise, landmarks, and NLL.

After verifying, update REPORT.md accordingly and push.
