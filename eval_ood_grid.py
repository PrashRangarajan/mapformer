"""OOD-d / OOD-s evaluation: paper-faithful grid-size + density generalization.

Tests the MapFormer paper's Table 2 OOD claims with our 5-variant minimal set.

Training: 64×64 grid, p_empty=0.5, l=128 (the standard setup our `_clean` checkpoints use)

OOD configs:
  OOD-d (dense):  32×32 grid, p_empty=0.2, l=64
  OOD-s (sparse): 128×128 grid, p_empty=0.8, l=512

We optionally apply the paper's omega-rescaling trick: multiply trained
omega by N/N' (training_size / test_size) before evaluation. This
compensates for the cumulative path-integration angle scaling with grid size.

TEMFaithful has no omega (uses per-action W_a); we evaluate it without
rescaling.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from mapformer.environment import GridWorld
from mapformer.model import MapFormerWM
from mapformer.model_inekf_level15 import MapFormerWM_Level15InEKF
from mapformer.model_inekf_gsf_nodrop import MapFormerWM_Level15GSF_NoDrop
from mapformer.model_tem_faithful import TEMFaithful
from mapformer.model_baseline_rope import MapFormerWM_RoPE


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
    if variant == "Level15GSF_NoDrop": kw["n_modes"] = 8
    m = cls(**kw); m.load_state_dict(c["model_state_dict"])
    return m.cuda().eval()


def rescale_omega(model, factor: float):
    """Apply the paper's omega-rescaling trick. Multiply omega by `factor`.

    Returns the original omega tensor so caller can restore it after eval."""
    if hasattr(model, "path_integrator"):
        orig = model.path_integrator.omega.data.clone()
        model.path_integrator.omega.data.mul_(factor)
        return ("path_integrator", orig)
    return None


def restore_omega(model, snapshot):
    if snapshot is None: return
    kind, orig = snapshot
    if kind == "path_integrator":
        model.path_integrator.omega.data.copy_(orig)


def eval_revisit(model, env, T, n_trials, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    correct = total = 0; nll = 0.0
    with torch.no_grad():
        for _ in range(n_trials):
            tokens, _, rm = env.generate_trajectory(T)
            tt = tokens.unsqueeze(0).cuda()
            try:
                logits = model(tt[:, :-1])
            except Exception as e:
                print(f"  eval crash: {e}")
                return None, None
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
    ap.add_argument("--output-md", default="OOD_GRID_RESULTS.md")
    ap.add_argument("--n-trials", type=int, default=30)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    args = ap.parse_args()

    # Three configs: IID + OOD-d + OOD-s. The IID config is the training one;
    # OOD-d and OOD-s match the paper's Table 2 definitions.
    CONFIGS = [
        ("IID",   dict(size=64,  p_empty=0.5, n_obs_types=16), 128, 1.0),
        ("OOD-d", dict(size=32,  p_empty=0.2, n_obs_types=16), 64,  64/32),
        ("OOD-s", dict(size=128, p_empty=0.8, n_obs_types=16), 512, 64/128),
    ]

    rows = {}  # (variant, config_name) -> list of (acc, nll) per seed

    for variant in VARIANT_CLS:
        for cfg_name, env_kwargs, T, omega_factor in CONFIGS:
            for s in args.seeds:
                ckpt = Path(f"mapformer/runs/{variant}_clean/seed{s}/{variant}.pt")
                if not ckpt.exists():
                    continue
                try:
                    m = build(variant, ckpt)
                except Exception as e:
                    print(f"  skip {variant} s{s}: {e}")
                    continue

                # Apply omega rescaling (no-op for TEMFaithful)
                snap = None
                if cfg_name != "IID" and variant != "TEMFaithful":
                    snap = rescale_omega(m, omega_factor)

                env = GridWorld(**env_kwargs, n_landmarks=0, seed=1000)
                t0 = time.time()
                acc, nll = eval_revisit(m, env, T, args.n_trials, seed=2000 + s)
                dt = time.time() - t0

                # Restore omega
                restore_omega(m, snap)

                if acc is not None:
                    rows.setdefault((variant, cfg_name), []).append((acc, nll))
                    print(f"  {variant:20s} s{s} {cfg_name:6s} T={T:4d} acc={acc:.3f} nll={nll:.3f} ({dt:.1f}s)")
                else:
                    print(f"  {variant:20s} s{s} {cfg_name:6s} CRASH ({dt:.1f}s)")

                del m; torch.cuda.empty_cache()

    # Generate markdown report
    lines = []
    lines.append("# OOD-d / OOD-s: grid-size + density generalization (paper-faithful)\n")
    lines.append("Tests the MapFormer paper's Table 2 OOD claims. Models trained on the")
    lines.append("standard clean config (64×64, p_empty=0.5, l=128) are evaluated on:\n")
    lines.append("- **IID:** 64×64, p_empty=0.5, T=128 (training distribution, fresh obs_map)")
    lines.append("- **OOD-d (dense, smaller):** 32×32, p_empty=0.2, T=64")
    lines.append("- **OOD-s (sparse, larger):** 128×128, p_empty=0.8, T=512\n")
    lines.append("MapFormer variants use the paper's omega-rescaling trick (multiply trained")
    lines.append("omega by N/N' where N=64, N'=test grid size). TEMFaithful has no omega")
    lines.append("(per-action W_a is content-independent); no rescaling applied.\n")
    lines.append("Held-out obs_map (env seed=1000). Multi-seed (n=3 from the prediction-trained")
    lines.append(f"checkpoints), {args.n_trials} trials per (variant, seed, config).\n")

    for cfg_name, env_kwargs, T, omega_factor in CONFIGS:
        lines.append(f"## {cfg_name} ({env_kwargs['size']}×{env_kwargs['size']}, p_empty={env_kwargs['p_empty']}, T={T})\n")
        lines.append("| Variant | acc (mean ± std) | NLL | n |")
        lines.append("|---|---|---|---|")
        for variant in VARIANT_CLS:
            key = (variant, cfg_name)
            if key not in rows:
                lines.append(f"| {variant} | — | — | 0 |"); continue
            accs = [r[0] for r in rows[key]]
            nlls = [r[1] for r in rows[key]]
            lines.append(f"| **{variant}** | {np.mean(accs):.3f} ± {np.std(accs):.3f} | "
                         f"{np.mean(nlls):.3f} | {len(accs)} |")
        lines.append("")

    lines.append("## Interpretation\n")
    lines.append("- IID should be near-ceiling for MapFormer family; RoPE collapses.")
    lines.append("- OOD-d tests denser-smaller-shorter generalization (32×32, p_empty=0.2, T=64). Should be relatively easy.")
    lines.append("- OOD-s tests sparser-larger-longer (128×128, p_empty=0.8, T=512). The hardest setting: cumulative angle grows further, fewer informative observations, more drift.")
    lines.append("- The omega-rescaling trick (×2 for 32×32, ×0.5 for 128×128) should make MapFormer variants scale-robust. Without it, the model assumes a 64×64 world during path integration.\n")
    lines.append("*Auto-generated by eval_ood_grid.py, evaluating CLEAN-trained checkpoints*\n")

    with open(args.output_md, "w") as f:
        f.write("\n".join(lines))
    print("\n--- Result file written: " + args.output_md)
    print("\n".join(lines[-30:]))


if __name__ == "__main__":
    main()
