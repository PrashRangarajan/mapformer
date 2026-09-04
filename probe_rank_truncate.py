"""Are the extra directions in an unconstrained angle map LOAD-BEARING?

The unconstrained maps do not collapse to rank 2: effective rank 7.5-8.4, top-2
energy 39-43% (LEARNED_RANK.md). Two readings, and they make opposite predictions:

  (a) the extra directions carry something real -> truncating to rank 2 COSTS
      accuracy, and MapFormer's displacement-dimensionality justification is
      incomplete for this task.
  (b) they are unused capacity the optimiser never pruned -> truncating to rank 2
      is FREE, MapFormer's bias is correct, and Selective RoPE's torus win is
      optimisation (a full-rank map is better conditioned) rather than capacity.

Truncate the TRAINED weight by SVD and re-evaluate. No retraining, so the
comparison isolates what the learned map's spectrum is actually used for.
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from mapformer.environment import GridWorld
from mapformer.train_variant import VARIANT_MAP
from mapformer.eval_noise_refine import evaluate


def effective_W(sd, prefix="angle.proj"):
    g = sd.get(f"{prefix}.parametrizations.weight.original0")
    v = sd.get(f"{prefix}.parametrizations.weight.original1")
    if g is not None and v is not None:
        return g * v / v.norm(dim=1, keepdim=True).clamp_min(1e-12)
    w0, w1 = sd.get(f"{prefix}.0.weight"), sd.get(f"{prefix}.1.weight")
    if w0 is not None and w1 is not None:
        return w1 @ w0
    return sd.get(f"{prefix}.weight")


def truncate(W, k):
    U, S, Vh = torch.linalg.svd(W.float(), full_matrices=False)
    S = S.clone(); S[k:] = 0.0
    return (U * S) @ Vh


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="NoBottleneck")
    ap.add_argument("--runs-dir", default="mapformer/runs/selective/torus/p0")
    ap.add_argument("--ranks", nargs="+", type=int, default=[1, 2, 3, 4, 8, 16, 64])
    ap.add_argument("--lengths", nargs="+", type=int, default=[128, 512])
    ap.add_argument("--n-trials", type=int, default=60)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="RANK_TRUNCATION.md")
    a = ap.parse_args()
    dev = torch.device(a.device)

    res = {}
    for s in range(8):
        cp = Path(a.runs_dir) / f"{a.arm}_s{s}" / f"{a.arm}.pt"
        if not cp.exists():
            continue
        blob = torch.load(cp, map_location="cpu", weights_only=False)
        c = blob["config"]
        W = effective_W(blob["model_state_dict"])
        bias = blob["model_state_dict"].get("angle.proj.bias")
        for k in a.ranks:
            m = VARIANT_MAP[a.arm](vocab_size=c["vocab_size"], d_model=c["d_model"],
                                   n_heads=c["n_heads"], n_layers=c["n_layers"],
                                   grid_size=c["grid_size"])
            m.load_state_dict(blob["model_state_dict"])
            lin = nn.Linear(W.shape[1], W.shape[0], bias=bias is not None)
            with torch.no_grad():
                lin.weight.copy_(truncate(W, k))
                if bias is not None:
                    lin.bias.copy_(bias)
            m.angle.proj = lin                      # replace the parameterised map
            m = m.to(dev).eval()
            env = GridWorld(size=c["grid_size"], n_obs_types=c.get("n_obs_types", 16),
                            p_empty=c.get("p_empty", 0.5), seed=10000)
            for L in a.lengths:
                acc, _nll = evaluate(m, env, L, a.n_trials, 0.0, dev, seed=1234 + s)
                res.setdefault((k, L), []).append(acc)
            del m; torch.cuda.empty_cache()
        print(f"seed {s} done", flush=True)

    o = [f"# Are the extra directions load-bearing? ({a.arm}, trained weights)", "",
         "The unconstrained angle map does not collapse to rank 2 (effective rank",
         "7.5-8.4, top-2 energy 39-43%). This truncates the TRAINED weight by SVD",
         "and re-evaluates -- no retraining, so it isolates what the spectrum is",
         "actually used for. MapFormer's own map is rank 2 by construction.", "",
         "| rank kept | " + " | ".join(f"T={L}" for L in a.lengths) + " |",
         "|---" * (len(a.lengths) + 1) + "|"]
    for k in a.ranks:
        cells = []
        for L in a.lengths:
            xs = res.get((k, L), [])
            cells.append(f"{np.mean(xs):.3f} ± {np.std(xs, ddof=1):.3f}"
                         if len(xs) > 1 else "—")
        o.append(f"| {k}{' (full)' if k >= 64 else ''} | " + " | ".join(cells) + " |")
    o.append("")
    full = {L: np.mean(res.get((max(a.ranks), L), [np.nan])) for L in a.lengths}
    r2 = {L: np.mean(res.get((2, L), [np.nan])) for L in a.lengths}
    o += ["## Verdict", ""]
    drops = {L: full[L] - r2[L] for L in a.lengths}
    o.append("Truncating to rank 2 costs: "
             + ", ".join(f"{d:+.3f} at T={L}" for L, d in drops.items()) + ".")
    o.append("")
    if all(abs(d) < 0.02 for d in drops.values()):
        o += ["**The extra directions are not load-bearing.** A trained "
              "unconstrained map can be projected to rank 2 at no cost, so the "
              "spectrum above rank 2 is unused capacity rather than information. "
              "MapFormer's displacement-dimensionality justification for r=2 holds "
              "on this task, and Selective RoPE's torus win is therefore NOT extra "
              "representational capacity -- optimisation (a full-rank map is better "
              "conditioned than a rank-2 one) is the remaining candidate.", ""]
    else:
        o += ["**The extra directions are load-bearing IN THIS MODEL** -- which "
              "is NOT the same as rank 2 being insufficient for the task. A map "
              "trained without the constraint spreads its solution over many "
              "directions and cannot be projected back afterwards; that says "
              "nothing about whether a rank-2 solution exists. It does: `Vanilla` "
              "is rank 2 by construction and solves the same task. Post-hoc "
              "truncation is not a sufficiency test, the way pruning a dense "
              "network to 2% says little about training one sparse at 2%. "
              "Separating optimisation from real extra signal needs models TRAINED "
              "at each rank -- see the rank sweep.", ""]
    o += ["Inference only, 8 seeds, held-out map. Truncation is exact SVD "
          "projection of the trained weight; the bias is left untouched."]
    Path(a.out).write_text("\n".join(o) + "\n")
    print("\n".join(o))


if __name__ == "__main__":
    main()
