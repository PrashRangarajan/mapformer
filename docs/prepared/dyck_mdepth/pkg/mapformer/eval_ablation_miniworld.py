"""Context-destruction ablation (validity gate F2) for the fresh-map factorial.

The trust gate that made Match-Query citable (0.918 -> 0.074 under shuffle) and
VOIDED hier-goal (0.912 -> 0.913). A genuine in-context cognitive map must
COLLAPSE when its context is destroyed. For each trained model we eval on the
same held-out set three ways:

  intact         the real held-out sequences (baseline nb_acc).
  obs-shuffle    permute the OBSERVATION token values across obs positions within
                 each sequence -> the location->token map becomes inconsistent
                 (a cell emits different tokens on different visits), so no map
                 can be retrieved. A real map-user must fall to ~marginal.
  action-shuffle permute the ACTION token values across action positions -> the
                 agent's path is scrambled, so position can't be integrated and
                 revisits can't be matched to first visits. A path-integration /
                 attention localiser must fall to ~marginal.

PASS = both shuffles collapse nb_acc to near the marginal / chance floor. If an
arm keeps high accuracy under shuffle, it is exploiting a non-map shortcut and
its headline number is void.
"""
import argparse
import json
import os
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F

from mapformer.miniworld_env import MiniWorldWorld
from mapformer.train_miniworld import build_or_load_eval_buffer
from mapformer.train_variant import VARIANT_MAP

_REPO = os.path.dirname(os.path.abspath(__file__))


def _shuffle_positions(tok, positions, rng):
    """Return a copy of tok with the VALUES at `positions` independently permuted
    within each row (breaks the map / path while preserving the token alphabet)."""
    out = tok.copy()
    for r in range(out.shape[0]):
        vals = out[r, positions]
        out[r, positions] = vals[rng.permutation(len(positions))]
    return out


@torch.no_grad()
def _score(model, tok, rev, blank, dev, bs=16):
    model.eval()
    ok_nb = tot_nb = 0
    for s in range(0, tok.shape[0], bs):
        t = torch.from_numpy(tok[s:s + bs]).to(dev)
        m = torch.from_numpy(rev[s:s + bs, 1:]).to(dev)
        inp, tgt = t[:, :-1], t[:, 1:]
        if m.sum() == 0:
            continue
        pred = model(inp).argmax(-1)
        nb = m & (tgt != blank)
        ok_nb += int((pred[nb] == tgt[nb]).sum()); tot_nb += int(nb.sum())
    return ok_nb / max(tot_nb, 1)


def _marginal(tok, rev, blank):
    """Most-common non-blank target on the scored set = the no-map floor."""
    tgt = tok[:, 1:][rev[:, 1:]]
    nb = tgt[tgt != blank]
    if len(nb) == 0:
        return float("nan")
    return Counter(nb.tolist()).most_common(1)[0][1] / len(nb)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default=os.path.join(_REPO, "runs", "miniworld_fresh"))
    ap.add_argument("--variants", nargs="+", default=["Vanilla", "RoPE", "MapPoPE-Flat", "PoPE-Flat"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--encodings", nargs="+", default=["raw", "allo"])
    ap.add_argument("--length", type=int, default=512)
    ap.add_argument("--eval-trials", type=int, default=128)
    ap.add_argument("--grid-size", type=int, default=8)
    ap.add_argument("--n-workers", type=int, default=16)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=os.path.join(_REPO, "MINIWORLD_FRESH_ABLATION.md"))
    args = ap.parse_args()
    dev = torch.device(args.device)
    rng = np.random.RandomState(999)

    lines = ["# MiniWorld fresh-map — context-destruction ablation (validity gate F2)", "",
             "A genuine in-context map must COLLAPSE when context is destroyed. "
             f"chance nb_acc = 1/16 = 0.0625. Held-out T={args.length}.", "",
             "| variant | enc | seed | intact | obs-shuffle | action-shuffle | marginal | verdict |",
             "|---|---|---|---|---|---|---|---|"]
    obs_idx = np.arange(1, 2 * args.length, 2)
    act_idx = np.arange(0, 2 * args.length, 2)
    for enc in args.encodings:
        env_test = MiniWorldWorld(grid_size=args.grid_size, seed=10000,
                                  allocentric=(enc == "allo"),
                                  oracle=(enc == "oracle"), fixed_map=False)
        blank = env_test.obs_offset + env_test.blank_token
        et, er = build_or_load_eval_buffer(env_test, args.length, args.eval_trials,
                                           n_workers=args.n_workers)
        marg = _marginal(et, er, blank)
        obs_sh = _shuffle_positions(et, obs_idx, rng)
        act_sh = _shuffle_positions(et, act_idx, rng)
        for v in args.variants:
            for s in args.seeds:
                pt = os.path.join(args.runs_dir, f"s{s}", f"{v}_{enc}.pt")
                if not os.path.exists(pt):
                    lines.append(f"| {v} | {enc} | {s} | — | — | — | {marg:.3f} | MISSING |")
                    continue
                ck = torch.load(pt, map_location="cpu")
                model = VARIANT_MAP[v](vocab_size=ck["vocab_size"], d_model=ck["d_model"],
                                       n_heads=ck["n_heads"], n_layers=ck["n_layers"],
                                       grid_size=args.grid_size).to(dev)
                model.load_state_dict(ck["model_state"])
                a0 = _score(model, et, er, blank, dev)
                a_obs = _score(model, obs_sh, er, blank, dev)
                a_act = _score(model, act_sh, er, blank, dev)
                # PASS if both shuffles fall near the marginal floor (within 0.05)
                floor = marg + 0.05
                good = (a_obs <= floor and a_act <= floor)
                # only meaningful if the arm actually learned something intact
                learned = a0 > marg + 0.10
                verdict = "PASS" if (good or not learned) else "SHORTCUT?"
                lines.append(f"| {v} | {enc} | {s} | {a0:.3f} | {a_obs:.3f} | "
                             f"{a_act:.3f} | {marg:.3f} | {verdict} |")
                print(lines[-1], flush=True)
    open(args.out, "w").write("\n".join(lines) + "\n")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
