"""Does a skewed basis inject drift that accumulates with length?

THE CLAIM UNDER TEST. MapFormer's state is an exact sum over tokens,
s_t = sum_a n_a(t) v_a with v_a = W_in emb[a]. If the learned basis were perfect
-- opposite actions cancelling, observations at zero -- then s_t would be an exact
linear function of the NET DISPLACEMENT alone, two numbers, no matter how long the
walk. It is not: ACTION_GEOMETRY measures |v_N + v_S| at 0.495 of the action scale
at r=2 against 0.092 at r=4, and observations at 0.139 / 0.042.

Decompose v_a into its antisymmetric part (which carries net displacement) and its
symmetric part (which does not). The symmetric part multiplies n_+ + n_-, the PATH
LENGTH along that axis, which grows linearly in t while net displacement does not.
So a skewed basis leaks path length into a code that should carry only
displacement, and the leak accumulates.

    residual_t  =  s_t  -  (V . d_t + c)      d_t = unwrapped net displacement

PREDICTION, stated before running: the residual grows with t, and the RATIO of
growth rates between r=2 and r=4 tracks the ratio of their opposition errors,
0.495 / 0.092 = 5.4x. If the residual is flat, the non-cancellation account is
wrong and the length-dependence of the rank effect needs another explanation.

No forward pass is needed -- s_t is a cumsum of a linear map of the embeddings --
so this is exact and costs no GPU.
"""
import argparse, json, glob
import numpy as np
import torch
from pathlib import Path

from mapformer.environment import GridWorld


def increments(cp):
    """v[token_id] = W_in @ emb[token_id], the per-token increment in state space."""
    b = torch.load(cp, map_location="cpu", weights_only=False)
    sd = b["model_state_dict"]
    W = sd["action_to_lie.w_in.weight"].numpy()          # (r, d)
    E = sd["token_emb.weight"].numpy()                   # (V, d)
    return E @ W.T, float(np.mean(b["losses"][-5:]))     # (V, r)


def run(arm, runs_dir, env, T, n_traj, seed):
    rows = []
    for cp in sorted(glob.glob(f"{runs_dir}/{arm}_s*/{arm}.pt")):
        v, _loss = increments(cp)
        S, D = [], []
        np.random.seed(seed)
        for _ in range(n_traj):
            tok, _m, _r = env.generate_trajectory(T)
            ids = tok.numpy()
            s = np.cumsum(v[ids], axis=0)                # (2T, r) the state
            # unwrapped net displacement, from the ACTION tokens only
            d = np.zeros((len(ids), 2))
            step = np.zeros(2)
            for i, tid in enumerate(ids):
                if tid < env.N_ACTIONS:
                    step = step + np.array(env.ACTION_DELTAS[int(tid)], float)
                d[i] = step
            S.append(s); D.append(d)
        S = np.concatenate(S); D = np.concatenate(D)
        # least squares fit s ~ V d + c, pooled over trajectories and positions
        X = np.concatenate([D, np.ones((len(D), 1))], 1)
        coef, *_ = np.linalg.lstsq(X, S, rcond=None)
        res = np.linalg.norm(S - X @ coef, axis=1)       # per position
        scale = np.linalg.norm(v[:env.N_ACTIONS], axis=1).mean()
        res = res.reshape(n_traj, -1).mean(0) / scale    # normalised, vs position
        rows.append(res)
    return np.array(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default="/home/prashr/mapformer/runs/rank_sweep/p0")
    ap.add_argument("--arms", nargs="+", default=["Vanilla", "Vanilla_r4"])
    ap.add_argument("--lengths", nargs="+", type=int, default=[128, 512, 1024, 2048])
    ap.add_argument("--n-traj", type=int, default=24)
    ap.add_argument("--env-seed", type=int, default=10000)
    ap.add_argument("--out", default="/home/prashr/mapformer/DRIFT_PROBE.md")
    a = ap.parse_args()

    env = GridWorld(size=64, n_obs_types=16, seed=a.env_seed)
    o = ["# Does a skewed basis inject drift that accumulates with length?", "",
         "State residual after removing everything net displacement can explain,",
         "normalised by the action scale. `s_t` is exact (a cumsum of a linear map",
         "of the embeddings), so no forward pass is involved.", "",
         "| arm | " + " | ".join(f"t={T}" for T in a.lengths) + " | growth |",
         "|---|" + "---|" * (len(a.lengths) + 1)]
    store = {}
    for arm in a.arms:
        ends = []
        for T in a.lengths:
            R = run(arm, a.runs_dir, env, T, a.n_traj, a.env_seed)
            ends.append((R[:, -1].mean(), R[:, -1].std(ddof=1), len(R)))
        store[arm] = ends
        growth = ends[-1][0] / max(ends[0][0], 1e-9)
        o.append(f"| `{arm}` | " +
                 " | ".join(f"{m:.3f} ± {s:.3f}" for m, s, _ in ends) +
                 f" | **{growth:.2f}x** |")
    o += ["", f"n = {store[a.arms[0]][0][2]} seeds, {a.n_traj} trajectories each, "
          f"env seed {a.env_seed}.", ""]

    if len(a.arms) == 2:
        x, y = a.arms
        o += ["## The pre-registered comparison", ""]
        for i, T in enumerate(a.lengths):
            rx, ry = store[x][i][0], store[y][i][0]
            o.append(f"- t={T}: `{x}` {rx:.3f} vs `{y}` {ry:.3f} "
                     f"-> **{rx/max(ry,1e-9):.2f}x**")
        o += ["", "Predicted ratio from the opposition errors measured in "
              "`ACTION_GEOMETRY.md` (0.495 vs 0.092): **5.4x**.", ""]
    Path(a.out).write_text("\n".join(o) + "\n")
    print("\n".join(o))


if __name__ == "__main__":
    main()
