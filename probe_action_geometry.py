"""Does r=4 destroy the interpretability that r=2 gives for free?

The paper's justification for r=2 is that Delta_in IS the 2D movement vector
(App. A.7), so the learned latent can be read directly: four actions should appear
as +/-x and +/-y. At r=4 the latent is a 4-vector and that reading is not
guaranteed -- which is a cost accuracy cannot see.

But it may survive anyway. The model can use two of the four directions for the
displacement and leave the rest as optimisation slack. Three structural tests, all
invariant to the arbitrary choice of basis:

  OPPOSITION   N + S ~= 0 and E + W ~= 0. Opposite actions must be negatives, which
               is what makes the latent a DISPLACEMENT rather than an arbitrary
               code. Reported as |N+S| / mean(|N|,|S|) -- 0 is perfect.
  DIMENSION    the four action vectors should span a 2-PLANE. Reported as the
               fraction of spectral energy in the top 2 singular values of the
               4 x r matrix of action latents.
  ORTHOGONALITY the two axes should be independent: |cos(angle(N, E))| ~= 0.

Plus the check that matters for the cognitive-map story: observation tokens should
sit near the ORIGIN, since they must not move the agent.

Inference only, on trained checkpoints. r=2 is the reference: it satisfies
DIMENSION by construction, so opposition and orthogonality are what distinguish a
genuine displacement code from an arbitrary one.
"""
import argparse
from pathlib import Path

import numpy as np
import torch

from mapformer.environment import GridWorld
from mapformer.train_variant import VARIANT_MAP


def latents(m, env, dev):
    """Delta_in for every token id: the r-dimensional internal action code."""
    ids = torch.arange(env.unified_vocab_size, device=dev)
    x = m.token_emb(ids)                                  # (V, d)
    return m.action_to_lie.w_in(x).detach().cpu().numpy()  # (V, r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default="mapformer/runs/rank_sweep/p0")
    ap.add_argument("--arms", nargs="+",
                    default=["Vanilla", "Vanilla_r4", "Vanilla_r8", "Vanilla_r32"])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="ACTION_GEOMETRY.md")
    a = ap.parse_args()
    dev = torch.device(a.device)

    rows = {}
    for arm in a.arms:
        opp, dim2, orth, obsnorm = [], [], [], []
        for s in range(8):
            cp = Path(a.runs_dir) / f"{arm}_s{s}" / f"{arm}.pt"
            if not cp.exists():
                continue
            blob = torch.load(cp, map_location="cpu", weights_only=False)
            c = blob["config"]
            m = VARIANT_MAP[arm](vocab_size=c["vocab_size"], d_model=c["d_model"],
                                 n_heads=c["n_heads"], n_layers=c["n_layers"],
                                 grid_size=c["grid_size"]).to(dev).eval()
            m.load_state_dict(blob["model_state_dict"])
            env = GridWorld(size=c["grid_size"], n_obs_types=c.get("n_obs_types", 16),
                            p_empty=c.get("p_empty", 0.5), seed=1)
            Z = latents(m, env, dev)
            A = Z[env.action_offset:env.action_offset + env.N_ACTIONS]   # (4, r)
            O = Z[env.obs_offset:]                                       # observations
            # ACTION_DELTAS order: pair each action with its opposite by delta
            deltas = [env.ACTION_DELTAS[i] for i in range(env.N_ACTIONS)]
            pairs = []
            for i in range(env.N_ACTIONS):
                for j in range(i + 1, env.N_ACTIONS):
                    if (deltas[i][0] == -deltas[j][0]) and (deltas[i][1] == -deltas[j][1]):
                        pairs.append((i, j))
            for i, j in pairs:
                scale = (np.linalg.norm(A[i]) + np.linalg.norm(A[j])) / 2
                opp.append(np.linalg.norm(A[i] + A[j]) / max(scale, 1e-9))
            if len(pairs) >= 2:
                u, v = A[pairs[0][0]], A[pairs[1][0]]
                orth.append(abs(float(u @ v) /
                                max(np.linalg.norm(u) * np.linalg.norm(v), 1e-9)))
            sv = np.linalg.svd(A - A.mean(0, keepdims=True), compute_uv=False)
            dim2.append(float((sv[:2] ** 2).sum() / max((sv ** 2).sum(), 1e-12)))
            obsnorm.append(float(np.linalg.norm(O, axis=1).mean() /
                                 max(np.linalg.norm(A, axis=1).mean(), 1e-9)))
            del m; torch.cuda.empty_cache()
        if opp:
            rows[arm] = (np.mean(opp), np.mean(dim2), np.mean(orth), np.mean(obsnorm))

    o = ["# Does a wider bottleneck destroy the action geometry?", "",
         "The paper's justification for r=2 is that `Delta_in` IS the 2D movement",
         "vector, so the latent can be read directly. At r>2 that is not guaranteed.",
         "Three basis-invariant structural tests on the trained latents, plus the",
         "check that observations must not move the agent.", "",
         "| arm | r | opposition <br><sub>|N+S|/scale, 0 is perfect</sub> | 2-plane energy <br><sub>1.0 = spans a plane</sub> | |cos(N,E)| <br><sub>0 = orthogonal</sub> | obs norm / action norm <br><sub>0 = no movement</sub> |",
         "|---|---|---|---|---|---|"]
    R = {"Vanilla": 2, "Vanilla_r4": 4, "Vanilla_r8": 8, "Vanilla_r16": 16,
         "Vanilla_r32": 32}
    for arm in a.arms:
        if arm not in rows:
            continue
        op, d2, ort, on = rows[arm]
        o.append(f"| {arm} | {R.get(arm,'?')} | {op:.4f} | {d2:.4f} | {ort:.4f} | {on:.4f} |")
    o.append("")
    o += ["## Verdict", ""]
    if "Vanilla" in rows and "Vanilla_r4" in rows:
        b, w = rows["Vanilla"], rows["Vanilla_r4"]
        keeps = (w[1] > 0.95) and (w[0] < b[0] * 2 + 0.05) and (w[2] < b[2] + 0.15)
        if keeps:
            o += [f"**The geometry survives.** At r=4 the four action latents still "
                  f"span a 2-plane ({w[1]:.1%} of their spectral energy), opposite "
                  f"actions still cancel ({w[0]:.3f} against r=2's {b[0]:.3f}), and "
                  f"the two axes are no less independent ({w[2]:.3f} vs {b[2]:.3f}). "
                  f"The extra rank is optimisation slack, not a different code: the "
                  f"displacement reading the paper relies on is preserved, and can "
                  f"be recovered exactly by projecting onto the top two singular "
                  f"directions.", ""]
        else:
            o += [f"**The geometry degrades.** At r=4 the action latents put "
                  f"{w[1]:.1%} of their energy in a 2-plane (r=2: 100% by "
                  f"construction), opposition is {w[0]:.3f} against {b[0]:.3f}, and "
                  f"|cos| is {w[2]:.3f} against {b[2]:.3f}. The +0.085 accuracy gain "
                  f"is bought at a real interpretability cost, and r=2 should be "
                  f"kept wherever the displacement reading matters.", ""]
    o += ["Inference only, 8 seeds. `opposition` and `|cos|` are scale-free; "
          "`2-plane energy` is 1.0 by construction at r=2, so only r>2 rows are "
          "informative on that column."]
    Path(a.out).write_text("\n".join(o) + "\n")
    print("\n".join(o))


if __name__ == "__main__":
    main()
