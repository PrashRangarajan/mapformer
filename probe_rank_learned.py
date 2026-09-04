"""Does an unconstrained angle map REDISCOVER MapFormer's rank-2 bottleneck?

MapFormer's W_Delta = W_out W_in is rank r=2 BY CONSTRUCTION, and the paper's
justification is structural: r is meant to be the dimensionality of the action
space, so on a 2D grid Delta_in is literally the displacement vector (App. A.7,
A.1). Every head and every frequency then reads the same 2-dimensional quantity.

Selective RoPE has no such constraint -- its W_omega is full rank, so each of the
H*n_b channels can be an independent function of the token. On language that is
the right call, since there is no displacement vector to name. On a 2D grid it
means the bias has to be LEARNED rather than given.

So: measure the effective rank of the learned map on the torus. If it collapses
toward 2, the unconstrained model rediscovers the inductive bias and the
bottleneck is a shortcut rather than a restriction. If it stays high, the model is
solving the task some other way, and "MapFormer's bias is correct for 2D" would
need rethinking.

Reported two ways, because a hard rank threshold is arbitrary:
  - fraction of spectral energy in the top 2 singular values
  - participation ratio  (sum s_i^2)^2 / sum s_i^4, a continuous effective rank

Inference only, on existing checkpoints.
"""
import argparse
from pathlib import Path

import numpy as np
import torch

from mapformer.train_variant import VARIANT_MAP


def spectrum(W):
    s = torch.linalg.svdvals(W.float()).cpu().numpy()
    e = s ** 2
    top2 = e[:2].sum() / e.sum()
    pr = (e.sum() ** 2) / (e ** 2).sum()
    return top2, pr, s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default="mapformer/runs/selective/torus/p0")
    ap.add_argument("--out", default="LEARNED_RANK.md")
    a = ap.parse_args()

    o = ["# Does an unconstrained angle map rediscover the rank-2 bottleneck?", "",
         "MapFormer's `W_Delta = W_out W_in` is rank 2 **by construction**, and the",
         "paper's reason is structural: `r` is the dimensionality of the action",
         "space, so on a 2D grid `Delta_in` is the displacement vector itself.",
         "Selective RoPE's `W_omega` is full rank -- each of the H*n_b channels can",
         "be an independent function of the token -- so on this task it would have",
         "to learn the bias rather than be given it.", "",
         "Effective rank of the learned map, torus checkpoints, 8 seeds:", "",
         "| arm | matrix | top-2 energy | participation ratio | max possible |",
         "|---|---|---|---|---|"]

    def effective_W(sd, prefix="angle.proj"):
        """Reconstruct the actual (out, in) weight, whatever the parameterisation.

        Weight norm stores magnitude `original0` (out,1) and direction `original1`
        (out,in); the weight is g * v/||v||. A first pass of this probe read
        `original0` -- a (64,1) column -- and duly reported "100% of the energy in
        the top 2 singular values", which is true of any rank-1 object and means
        nothing. Reconstruct it properly, and for the low-rank arms multiply the
        two factors so the product's rank is what gets measured.
        """
        g = sd.get(f"{prefix}.parametrizations.weight.original0")
        v = sd.get(f"{prefix}.parametrizations.weight.original1")
        if g is not None and v is not None:
            return g * v / v.norm(dim=1, keepdim=True).clamp_min(1e-12)
        w0, w1 = sd.get(f"{prefix}.0.weight"), sd.get(f"{prefix}.1.weight")
        if w0 is not None and w1 is not None:
            return w1 @ w0                       # (out, r) @ (r, in) -> (out, in)
        return sd.get(f"{prefix}.weight")

    rows = {}
    for arm in ("NoBottleneck", "SRoPEGen", "GateAngle", "ConvAngle"):
        t2, pr, shp = [], [], None
        for s in range(8):
            cp = Path(a.runs_dir) / f"{arm}_s{s}" / f"{arm}.pt"
            if not cp.exists():
                continue
            W = effective_W(torch.load(cp, map_location="cpu",
                                       weights_only=False)["model_state_dict"])
            if W is None or W.ndim != 2:
                continue
            shp = tuple(W.shape)
            x, y, _ = spectrum(W)
            t2.append(x); pr.append(y)
        if t2:
            rows[arm] = (float(np.mean(t2)), float(np.mean(pr)))
            o.append(f"| {arm} | W_omega {shp} | {np.mean(t2):.3f} | "
                     f"{np.mean(pr):.2f} | {min(shp)} |")

    # MapFormer's own map, for the reference point
    m = VARIANT_MAP["Vanilla"](vocab_size=21, d_model=128, n_heads=2, n_layers=1,
                               grid_size=64)
    W = (m.action_to_lie.w_out.weight @ m.action_to_lie.w_in.weight).detach()
    x, y, _ = spectrum(W)
    o.append(f"| Vanilla (r=2, by construction) | W_out W_in {tuple(W.shape)} | "
             f"{x:.3f} | {y:.2f} | 2 |")
    o.append("")

    o += ["## Verdict", ""]
    if rows:
        # judge on the arms that are genuinely UNCONSTRAINED, not the low-rank ones
        free = {k: v for k, v in rows.items() if k in ("NoBottleneck", "SRoPEGen")}
        best = max(free.values(), key=lambda v: v[0])[0] if free else 0.0
        worst_pr = max(v[1] for v in free.values()) if free else float("nan")
        if best > 0.9:
            o += [f"**The bias is rediscovered.** The unconstrained maps put "
                  f"{best:.1%} of their spectral energy in the top two singular "
                  f"values, with a participation ratio of {worst_pr:.2f} against a "
                  f"maximum of 64. An unconstrained "
                  f"model given a 2D environment converges on a 2-dimensional "
                  f"latent, which is exactly MapFormer's stated justification for "
                  f"r=2 -- so the bottleneck is a shortcut to a solution the model "
                  f"reaches anyway, not a restriction it needs.", ""]
        else:
            o += [f"**The bias is NOT rediscovered.** Top-2 energy is only "
                  f"{best:.1%} and the participation ratio is {worst_pr:.2f}, so "
                  f"the unconstrained map spreads across many directions. Either "
                  f"the extra directions carry something real -- which would "
                  f"explain the torus win and contradict the "
                  f"displacement-dimensionality story -- or they are unused "
                  f"capacity the optimiser never pruned. The two are "
                  f"distinguishable by whether zeroing the tail costs accuracy.", ""]
    o += ["Inference only. Participation ratio is `(sum s^2)^2 / sum s^4`, a "
          "continuous effective rank: it equals k for k equal singular values and "
          "1 for a rank-1 map."]
    Path(a.out).write_text("\n".join(o) + "\n")
    print("\n".join(o))


if __name__ == "__main__":
    main()
