"""Long-T eval on existing lm200 checkpoints.

Reuses 3 seeds of each variant from runs/. Evaluates revisit accuracy + NLL
at T ∈ {512, 1024, 2048, 4096} on a fresh obs_map. Tests whether the gap
between architectures grows with sequence length — the bounded-error
Kalman claim should make Level15* degrade more gracefully than Vanilla,
and GSF's multi-modal posterior should shine as the trajectory becomes
more ambiguous.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from mapformer.environment import GridWorld
from mapformer.model import MapFormerWM
from mapformer.model_inekf_level15 import MapFormerWM_Level15InEKF
from mapformer.model_inekf_level15_nodrop import MapFormerWM_Level15NoDrop
from mapformer.model_inekf_gsf import MapFormerWM_Level15GSF
from mapformer.model_inekf_gsf_nodrop import MapFormerWM_Level15GSF_NoDrop
from mapformer.model_tem_faithful import TEMFaithful
from mapformer.model_baseline_rope import MapFormerWM_RoPE


# Minimal "cognitive-map necessity + improvement" set:
# - RoPE      : standard transformer baseline (no path-integration mechanism)
# - Vanilla   : MapFormer-WM, the cognitive-map inductive bias
# - Level15   : MapFormer + correction
# - Level15GSF_NoDrop : our best stacked variant
# - TEMFaithful : strong cognitive-map baseline (post-fix)
# TODO before paper submission: backfill RoPE/LSTM/MambaLike on all
# within-family tables (LEVEL15BETA, VOCAB_SWEEP, NODROP_PARETO, etc.)
VARIANT_CLS = {
    "RoPE":               MapFormerWM_RoPE,
    "Vanilla":            MapFormerWM,
    "Level15":            MapFormerWM_Level15InEKF,
    "Level15GSF_NoDrop":  MapFormerWM_Level15GSF_NoDrop,
    "TEMFaithful":        TEMFaithful,
}


def build(variant, ckpt):
    c = torch.load(ckpt, map_location="cuda", weights_only=False)
    cfg = c["config"]; cls = VARIANT_CLS[variant]
    kw = dict(vocab_size=cfg["vocab_size"], d_model=cfg["d_model"],
              n_heads=cfg["n_heads"], n_layers=cfg["n_layers"],
              grid_size=cfg["grid_size"])
    if variant in ("Level15GSF", "Level15GSF_NoDrop"): kw["n_modes"] = 8
    m = cls(**kw); m.load_state_dict(c["model_state_dict"])
    return m.cuda().eval()


def eval_revisit(model, env, T, n_trials, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    correct = total = 0; nll = 0.0
    with torch.no_grad():
        for _ in range(n_trials):
            tokens, _, rm = env.generate_trajectory(T)
            tt = tokens.unsqueeze(0).cuda()
            try: logits = model(tt[:, :-1])
            except Exception: return None, None
            lp = F.log_softmax(logits, dim=-1)
            preds = lp.argmax(-1)[0]; tgts = tt[0, 1:]; mask = rm[1:].cuda()
            if mask.sum() == 0: continue
            correct += (preds[mask] == tgts[mask]).sum().item()
            total += mask.sum().item()
            idx = torch.arange(lp.shape[1], device="cuda")[mask]
            nll += -lp[0, idx, tgts[mask]].sum().item()
    return (correct / total if total else None, nll / total if total else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-md", default="LONGT_EVAL_RESULTS.md")
    ap.add_argument("--T-values", nargs="+", type=int, default=[512, 1024, 2048, 4096])
    ap.add_argument("--n-trials", type=int, default=50)
    args = ap.parse_args()

    env = GridWorld(size=64, n_obs_types=16, p_empty=0.5, n_landmarks=200, seed=1000)

    rows = []  # (variant, T, mean_acc, std_acc, mean_nll, n_seeds)
    for variant in ["RoPE", "Vanilla", "Level15", "Level15GSF_NoDrop", "TEMFaithful"]:
        for T in args.T_values:
            accs, nlls = [], []
            for s in [0, 1, 2]:
                ckpt = Path(f"mapformer/runs/{variant}_lm200/seed{s}/{variant}.pt")
                if not ckpt.exists(): continue
                try:
                    m = build(variant, ckpt)
                except Exception as e:
                    print(f"[skip {variant} s{s}: {e}]")
                    continue
                t0 = time.time()
                acc, nll = eval_revisit(m, env, T, args.n_trials, seed=2000 + s)
                dt = time.time() - t0
                if acc is not None:
                    accs.append(acc); nlls.append(nll)
                    print(f"  {variant:18s} s{s} T={T:5d}: acc {acc:.3f} nll {nll:.3f} ({dt:.1f}s)")
                else:
                    print(f"  {variant:18s} s{s} T={T:5d}: crash")
                del m; torch.cuda.empty_cache()
            if accs:
                rows.append((variant, T, float(np.mean(accs)), float(np.std(accs)),
                             float(np.mean(nlls)), len(accs)))

    # Markdown report
    lines = []
    lines.append("# Long-T eval — does the gap grow with sequence length?\n")
    lines.append("All 3 seeds of each lm200 prediction-trained checkpoint, evaluated")
    lines.append(f"at T ∈ {args.T_values} on a fresh obs_map (seed=1000), {args.n_trials} trials each.\n")
    lines.append("**Cognitive-map mechanism predictions:**")
    lines.append("- RoPE (standard transformer): no path-integration mechanism; should collapse fastest.")
    lines.append("- Vanilla MapFormer: cognitive-map bias but no drift correction; gradual decay.")
    lines.append("- Level15: bounded-error wrap; should degrade gracefully.")
    lines.append("- Level15GSF_NoDrop: stacked fixes; should hold even further.")
    lines.append("- TEMFaithful: per-action W_a is exact rotation; baseline upper bound.\n")
    lines.append("Note: this is the MINIMAL cognitive-map necessity comparison.")
    lines.append("LSTM / MambaLike / VanillaEM / Level15EM / NoDrop / GSF (single chain)")
    lines.append("will be backfilled before paper submission.\n")

    for T in args.T_values:
        lines.append(f"## T = {T}\n")
        lines.append("| Variant | acc (mean ± std) | NLL | n |")
        lines.append("|---|---|---|---|")
        for variant, T2, acc, std, nll, n in rows:
            if T2 != T: continue
            lines.append(f"| **{variant}** | {acc:.3f} ± {std:.3f} | {nll:.3f} | {n} |")
        lines.append("")
    lines.append("*Auto-generated by eval_long_t.py*\n")
    with open(args.output_md, "w") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
