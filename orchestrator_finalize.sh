#!/bin/bash
# Runs after orchestrator.py finishes all trainings. Produces the
# paper-ready evaluation tables and commits to GitHub.

set -e
cd /home/prashr

RESULTS=/home/prashr/mapformer/RESULTS_PAPER.md

echo "[$(date)] Starting final evaluation..."

python3 -u << 'PYEOF' > "$RESULTS"
"""Aggregate all multi-seed results into paper-ready markdown tables."""
import torch, torch.nn.functional as F, numpy as np, statistics as st
from pathlib import Path
from mapformer.environment import GridWorld
from mapformer.model import MapFormerWM, MapFormerEM
from mapformer.model_inekf_parallel import MapFormerWM_ParallelInEKF
from mapformer.model_inekf_level15 import MapFormerWM_Level15InEKF
from mapformer.model_inekf_level2 import MapFormerWM_Level2InEKF
from mapformer.model_predictive_coding import MapFormerWM_PredictiveCoding
from mapformer.model_baseline_rope import MapFormerWM_RoPE
from mapformer.model_ablations import ABLATIONS

VARIANT_CLS = {
    "Vanilla": MapFormerWM, "Level1": MapFormerWM_ParallelInEKF,
    "Level15": MapFormerWM_Level15InEKF, "Level2": MapFormerWM_Level2InEKF,
    "PC": MapFormerWM_PredictiveCoding, "RoPE": MapFormerWM_RoPE,
    **ABLATIONS,
}

RUNS = Path("mapformer/runs")
SEEDS = [0, 1, 2]

def build_model(variant, ckpt_path, env):
    c = torch.load(ckpt_path, map_location="cuda", weights_only=False)
    cls = VARIANT_CLS[variant]
    m = cls(vocab_size=env.unified_vocab_size, d_model=128, n_heads=2, n_layers=1, grid_size=64)
    m.load_state_dict(c["model_state_dict"])
    return m.cuda().eval(), c

def eval_revisit(model, env, T, n_trials, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    c = tot = 0; nll_sum = 0.0
    with torch.no_grad():
        for _ in range(n_trials):
            tokens, om, rm = env.generate_trajectory(T)
            tt = tokens.unsqueeze(0).cuda()
            logits = model(tt[:, :-1])
            lp = F.log_softmax(logits, dim=-1)
            preds = lp.argmax(-1)[0]
            tgts = tt[0, 1:]; mask = rm[1:].cuda()
            if mask.sum() == 0: continue
            c += (preds[mask] == tgts[mask]).sum().item()
            tot += mask.sum().item()
            idx = torch.arange(lp.shape[1], device="cuda")[mask]
            nll_sum += -lp[0, idx, tgts[mask]].sum().item()
    return (c / tot if tot else 0, nll_sum / tot if tot else float("inf"))

def summarize(metrics_by_seed):
    vals = list(metrics_by_seed.values())
    if not vals: return "N/A"
    return f"{st.mean(vals):.3f} ± {st.pstdev(vals):.3f}" if len(vals) > 1 else f"{vals[0]:.3f}"

print(f"# Paper-Ready Results\n\nGenerated: {__import__('datetime').datetime.now()}\n")
print(f"All trainings used 50 epochs × 156 batches × 128 batch size = ~1M sequences. "
      f"Multi-seed results below: mean ± std over 3 training seeds (0, 1, 2).\n")

configs = [
    ("clean",  0),
    ("noise",  0),
    ("lm200",  200),
]
variants_main = ["Vanilla", "RoPE", "Level1", "Level15", "PC"]

for cfg_tag, n_lm in configs:
    env = GridWorld(size=64, n_obs_types=16, p_empty=0.5, n_landmarks=n_lm, seed=42)
    print(f"## Config: {cfg_tag} (n_landmarks={n_lm})\n")
    print(f"| Variant | T=128 acc | T=128 NLL | T=512 acc | T=512 NLL |")
    print(f"|---------|-----------|-----------|-----------|-----------|")
    for variant in variants_main:
        per_seed = {"acc128": {}, "nll128": {}, "acc512": {}, "nll512": {}}
        for seed in SEEDS:
            ckpt = RUNS / f"{variant}_{cfg_tag}" / f"seed{seed}" / f"{variant}.pt"
            if not ckpt.exists(): continue
            try:
                m, _ = build_model(variant, ckpt, env)
                a128, n128 = eval_revisit(m, env, 128, 200, seed=seed)
                a512, n512 = eval_revisit(m, env, 512, 100, seed=seed)
                per_seed["acc128"][seed] = a128
                per_seed["nll128"][seed] = n128
                per_seed["acc512"][seed] = a512
                per_seed["nll512"][seed] = n512
            except Exception as e:
                print(f"# error evaluating {variant} {cfg_tag} seed{seed}: {e}")
        print(f"| {variant} | {summarize(per_seed['acc128'])} | {summarize(per_seed['nll128'])} | "
              f"{summarize(per_seed['acc512'])} | {summarize(per_seed['nll512'])} |")
    print("")

# Ablations (seed 0 only)
print("## Level 1.5 Ablations (single seed)\n")
for cfg_tag, n_lm in [("clean", 0), ("lm200", 200)]:
    env = GridWorld(size=64, n_obs_types=16, p_empty=0.5, n_landmarks=n_lm, seed=42)
    print(f"### Config: {cfg_tag}\n")
    print(f"| Variant | T=128 acc | T=128 NLL | T=512 acc | T=512 NLL |")
    print(f"|---------|-----------|-----------|-----------|-----------|")
    # Include Level15 as baseline reference
    for v in ["Level15", "L15_ConstR", "L15_NoMeas", "L15_NoCorr", "L15_DARE"]:
        ckpt = RUNS / f"{v}_{cfg_tag}" / f"seed0" / f"{v}.pt"
        if not ckpt.exists(): continue
        try:
            m, _ = build_model(v, ckpt, env)
            a128, n128 = eval_revisit(m, env, 128, 200, seed=0)
            a512, n512 = eval_revisit(m, env, 512, 100, seed=0)
            print(f"| {v} | {a128:.3f} | {n128:.3f} | {a512:.3f} | {n512:.3f} |")
        except Exception as e:
            print(f"# error: {e}")
    print("")

print("\n---\n*Auto-generated by orchestrator_finalize.sh.*\n")
PYEOF

echo "[$(date)] Results saved to $RESULTS"
echo "[$(date)] Committing to git..."

cd /home/prashr/mapformer
git add RESULTS_PAPER.md model_ablations.py model_baseline_rope.py train_variant.py orchestrator.py orchestrator_finalize.sh runs/
git commit -m "Paper-ready multi-seed ablation + baseline experiment suite

Includes:
- Multi-seed (3 seeds) training runs for 5 main variants × 3 configs
  (clean / noise / landmarks)
- 4 Level 1.5 ablations (L15_ConstR, L15_NoMeas, L15_NoCorr, L15_DARE)
- RoPE baseline (standard transformer with fixed position rotations)
- Unified train_variant.py for consistent training across all models
- Orchestrator that schedules everything on 2 GPUs
- Aggregated results in RESULTS_PAPER.md"
git push origin main 2>&1 | tail -5
echo "[$(date)] Done."
