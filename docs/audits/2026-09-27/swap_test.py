"""Decoy swap test for the context-step pilots (replaces the inline scripts behind CTXSTEP_PILOT1/3 and
CTXSTEP_HS_RECIPE; audit 2026-09-30 B1).

For each checkpoint: on held-out-map sequences, at every direction word of a class (real move / decoy),
set the word to "north" in one copy and "south" in another, and measure the mean |difference| of the
accumulated angle `offset` tokens later. ratio = decoy / move (0 = decoys ignored, 1 = treated as moves).
Optionally replace the token at `--replace-at` (relative to the direction word) by `--replace-with` in
both copies (the cue-replacement check).

    python3 -m mapformer.docs... (run as a file with PYTHONPATH=/home/prashr)
    swap_test.py --task ctx|ctx2|ctx3 [--cue lead|trail] [--dist near|far] --ckpt PATH --arm CF|CG|SR|HS
                 --layers N --offset K [--replace-at J --replace-with WORD] [--n-seq 30 --n 50 --seed 321]
Prints one line per checkpoint; --json appends a record to a file.
"""
import argparse
import json

import numpy as np
import torch


def env_and_arms(task, cue, dist):
    if task == "ctx":
        from mapformer.environment_textworld_ctx import TextWorldCtx as E
        from mapformer.train_ctxstep import ARMS
        return E(seed=10000, p_decoy=0.3), ARMS
    if task == "ctx2":
        from mapformer.environment_textworld_ctx2 import TextWorldCtx2 as E
        from mapformer.train_ctxstep2 import ARMS
        return E(seed=10000, cue=cue), ARMS
    from mapformer.environment_textworld_ctx3 import TextWorldCtx3 as E
    from mapformer.train_ctxstep3 import ARMS
    return E(seed=10000, cue=cue, dist=dist), ARMS


def theta_fn(m):
    def th(tok):
        with torch.no_grad():
            if hasattr(m, "angle"):
                return m.angle(m.token_emb(tok))
            d = m.step(tok)[0] if hasattr(m, "step") else m.action_to_lie(m.token_emb(tok))
            return torch.cumsum(d, 1) * m.path_integrator.omega
    return th


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["ctx", "ctx2", "ctx3"])
    ap.add_argument("--cue", default="lead"); ap.add_argument("--dist", default="far")
    ap.add_argument("--ckpt", required=True); ap.add_argument("--arm", required=True)
    ap.add_argument("--layers", type=int, default=1); ap.add_argument("--offset", type=int, required=True)
    ap.add_argument("--replace-at", type=int, default=None); ap.add_argument("--replace-with", default=None)
    ap.add_argument("--n-seq", type=int, default=30); ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=321); ap.add_argument("--json", default=None)
    a = ap.parse_args()
    env, ARMS = env_and_arms(a.task, a.cue, a.dist)
    V = env.unified_vocab_size; N, S = env.idx["north"], env.idx["south"]
    blob = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    m = ARMS[a.arm](vocab_size=V, d_model=128, n_heads=2, n_layers=a.layers, grid_size=64)
    m.load_state_dict(blob["model_state_dict"]); m.eval(); th = theta_fn(m)
    np.random.seed(a.seed); data = []
    for _ in range(a.n_seq):
        t = env.generate_trajectory(1024)[0]; data.append((t, list(env.ctx)))
    res = {}
    for want in ("move", "decoy"):
        out = []
        for tok, ctx in data:
            for i, k in ctx:
                if (k == "move") != (want == "move") or i + a.offset + 2 >= len(tok) or i < 5:
                    continue
                x = tok.clone(); y = tok.clone()
                if a.replace_at is not None and want == "decoy":
                    x[i + a.replace_at] = env.idx[a.replace_with]; y[i + a.replace_at] = env.idx[a.replace_with]
                x[i] = N; y[i] = S
                out.append((th(x[None])[0, i + a.offset] - th(y[None])[0, i + a.offset]).abs().mean().item())
                if len(out) >= a.n:
                    break
            if len(out) >= a.n:
                break
        res[want] = float(np.mean(out)); res[f"n_{want}"] = len(out)
    res["ratio"] = res["decoy"] / max(res["move"], 1e-9)
    tag = f" replace[{a.replace_at:+d}]={a.replace_with}" if a.replace_at is not None else ""
    print(f"{a.ckpt}: move {res['move']:.3f} decoy {res['decoy']:.3f} ratio {res['ratio']:.2f}{tag}")
    if a.json:
        with open(a.json, "a") as f:
            f.write(json.dumps({**vars(a), **res}) + "\n")


if __name__ == "__main__":
    main()
