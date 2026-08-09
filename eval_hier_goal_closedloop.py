"""Closed-loop navigation eval for the hierarchical goal task.

Unlike train_hier_goal's teacher-forced action-MATCH accuracy, here the model
DRIVES: at each navigate step it picks an action from its OWN rollout, we step
the torus, feed the result back, and repeat. Success = the agent reaches the
target cell within the step budget. This measures whether MapWM-Hier's
teacher-forced OOD advantage cashes out in actual navigation behaviour.

Batched over trials (all run in lockstep for T_navigate steps; 'reached' is
latched). Evaluated at multiple OOD explore lengths.
"""
import argparse
import json
import statistics as st
from pathlib import Path

import numpy as np
import torch

from mapformer.environment_hier_goal import HierGoalGridWorld
from mapformer.environment_goal import bfs_torus
from mapformer.train_variant import VARIANT_MAP

DISPLAY = {"Vanilla": "MapWM-Flat", "Hourglass_k2": "MapWM-Hier",
           "PoPE": "PoPE-Flat", "MapPoPE_Hier": "MapPoPE-Hier",
           "PlainFlat": "Plain-Flat", "PlainHourglass": "Plain-Hier",
           "Hourglass_CoarseIdx": "MapWM-Hier-CoarseIdx"}


@torch.no_grad()
def rollout(model, env, T_explore, T_navigate, n_trials, device, seed):
    rng = np.random.RandomState(seed)
    N, S, A = n_trials, env.size, env.N_ACTIONS
    obs = np.stack([env._draw_obs(rng) for _ in range(N)])            # (N,S,S)
    room = rng.randint(0, env.n_rooms, N); local = rng.randint(0, env.n_local, N)
    cells = [env.room_local_to_cell(int(room[i]), int(local[i])) for i in range(N)]
    gx = np.array([c[0] for c in cells]); gy = np.array([c[1] for c in cells])
    x = np.zeros(N, dtype=np.int64); y = np.zeros(N, dtype=np.int64)
    DA = env.ACTION_DELTAS

    toks = torch.zeros(N, 2 + 2 * T_explore, dtype=torch.long, device=device)
    toks[:, 0] = torch.tensor(env.room_tok0 + room, device=device)
    toks[:, 1] = torch.tensor(env.local_tok0 + local, device=device)
    for i in range(T_explore):                                       # explore (random)
        a = rng.randint(0, A, N)
        dx = np.array([DA[int(v)][0] for v in a]); dy = np.array([DA[int(v)][1] for v in a])
        x = (x + dx) % S; y = (y + dy) % S
        toks[:, 2 + 2 * i] = torch.tensor(a + env.action_offset, device=device)
        toks[:, 2 + 2 * i + 1] = torch.tensor(obs[np.arange(N), x, y] + env.obs_offset, device=device)

    obs_t = torch.tensor(obs, device=device); ar = torch.arange(N, device=device)
    adx = torch.tensor([DA[k][0] for k in range(A)], device=device)
    ady = torch.tensor([DA[k][1] for k in range(A)], device=device)
    xt = torch.tensor(x, device=device); yt = torch.tensor(y, device=device)
    gxt = torch.tensor(gx, device=device); gyt = torch.tensor(gy, device=device)
    reached = torch.zeros(N, dtype=torch.bool, device=device)
    for _ in range(T_navigate):                                      # model drives
        a = model(toks)[:, -1, :A].argmax(-1)                        # restrict to action tokens
        xt = (xt + adx[a]) % S; yt = (yt + ady[a]) % S
        o = obs_t[ar, xt, yt]
        toks = torch.cat([toks, torch.stack([a + env.action_offset, o + env.obs_offset], 1)], 1)
        reached |= (xt == gxt) & (yt == gyt)
    dist = (torch.minimum((xt - gxt) % S, (gxt - xt) % S)
            + torch.minimum((yt - gyt) % S, (gyt - yt) % S)).float()
    return reached.float().mean().item(), dist.mean().item()


def rollout_reference(env, T_explore, T_navigate, n_trials, seed, policy):
    """Model-free reference policies, same harness/geometry as rollout().

    'random'  uniform actions -- the floor. Without this, a success rate is
              uninterpretable.
    'bfs'     shortest-path oracle -- must come out ~1.0. If it does not, the
              harness is broken and no model number from it means anything.
    """
    rng = np.random.RandomState(seed)
    N, S, A = n_trials, env.size, env.N_ACTIONS
    DA = env.ACTION_DELTAS
    room = rng.randint(0, env.n_rooms, N); local = rng.randint(0, env.n_local, N)
    cells = [env.room_local_to_cell(int(room[i]), int(local[i])) for i in range(N)]
    gx = np.array([c[0] for c in cells]); gy = np.array([c[1] for c in cells])
    x = np.zeros(N, dtype=np.int64); y = np.zeros(N, dtype=np.int64)
    for _ in range(T_explore):
        a = rng.randint(0, A, N)
        x = (x + np.array([DA[int(v)][0] for v in a])) % S
        y = (y + np.array([DA[int(v)][1] for v in a])) % S
    reached = np.zeros(N, dtype=bool)
    paths = None
    if policy == "bfs":
        paths = [bfs_torus((int(x[i]), int(y[i])), (int(gx[i]), int(gy[i])), S)
                 for i in range(N)]
    for t in range(T_navigate):
        if policy == "random":
            a = rng.randint(0, A, N)
        else:
            a = np.array([paths[i][t] if t < len(paths[i]) else 0 for i in range(N)])
        x = (x + np.array([DA[int(v)][0] for v in a])) % S
        y = (y + np.array([DA[int(v)][1] for v in a])) % S
        reached |= (x == gx) & (y == gy)
    dist = (np.minimum((x - gx) % S, (gx - x) % S)
            + np.minimum((y - gy) % S, (gy - y) % S)).astype(float)
    return float(reached.mean()), float(dist.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, required=True)
    ap.add_argument("--variants", nargs="+", required=True)
    ap.add_argument("--explore-lengths", nargs="+", type=int, default=[64, 128, 192, 256])
    ap.add_argument("--T-navigate", type=int, default=96)
    ap.add_argument("--n-trials", type=int, default=200)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="HIERGOAL_CLOSEDLOOP.md")
    args = ap.parse_args()

    env = HierGoalGridWorld(size=64, room_size=8, seed=10000)
    succ = {v: {t: [] for t in args.explore_lengths} for v in args.variants}
    for ref in ("random", "bfs"):
        succ[ref] = {t: [] for t in args.explore_lengths}
        for te in args.explore_lengths:
            sr, d = rollout_reference(env, te, args.T_navigate, args.n_trials, 7, ref)
            succ[ref][te].append(sr)
            print(f"[REFERENCE {ref}] T_exp={te}: success={sr:.3f} final_dist={d:.1f}", flush=True)
    for v in args.variants:
        for s in args.seeds:
            cp = Path(args.runs_dir) / f"seed{s}" / f"{v}_hiergoal.pt"
            if not cp.exists():
                print(f"MISSING {cp}"); continue
            c = torch.load(cp, map_location=args.device, weights_only=False)
            model = VARIANT_MAP[v](vocab_size=c["vocab_size"], d_model=c["d_model"],
                                   n_heads=c["n_heads"], n_layers=c["n_layers"],
                                   grid_size=64).to(args.device).eval()
            model.load_state_dict(c["model_state"])
            for te in args.explore_lengths:
                sr, d = rollout(model, env, te, args.T_navigate, args.n_trials,
                                args.device, seed=2000 + s)
                succ[v][te].append(sr)
                print(f"[{DISPLAY.get(v,v)} seed{s}] T_exp={te}: success={sr:.3f} final_dist={d:.1f}")

    def ms(xs): return (st.mean(xs), st.pstdev(xs) if len(xs) > 1 else 0.0) if xs else (float('nan'), 0)
    lines = ["# Hierarchical goal nav — CLOSED-LOOP success (model drives its own rollout)\n",
             f"Fixed start, held-out env. Success = reached target cell within "
             f"T_navigate={args.T_navigate} steps. n_trials={args.n_trials}, seeds={args.seeds}.\n",
             "| variant | " + " | ".join(f"T_exp={t}" for t in args.explore_lengths) + " |",
             "|" + "---|" * (len(args.explore_lengths) + 1)]
    for v in args.variants:
        cells = [f"{ms(succ[v][t])[0]:.3f} ± {ms(succ[v][t])[1]:.3f}" for t in args.explore_lengths]
        lines.append(f"| {DISPLAY.get(v,v)} | " + " | ".join(cells) + " |")
    Path(args.out).write_text("\n".join(lines) + "\n")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
