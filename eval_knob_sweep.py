"""Which environment property flips the sign of the position effect?

Reads the knob sweep and reports, per condition, the POSITION EFFECT --
Vanilla (path-integrated) minus RoPE (index) -- against the MEASURED floor for
that condition. Floors move a lot across these conditions (the blank rate
depends on p_empty and the observation mode), so an accuracy is meaningless
without its own floor: standing rule 4.
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from mapformer.environment import GridWorld
from mapformer.train_variant import VARIANT_MAP

_REPO = Path(__file__).resolve().parent
CONDS = {
    "baseline":    dict(size=64, n_obs_types=16),
    "rotate":      dict(size=64, n_obs_types=16, action_mode="rotate"),
    "ego":         dict(size=64, n_obs_types=16, obs_mode="ego"),
    "wall":        dict(size=64, n_obs_types=16, boundary="wall"),
    "small":       dict(size=16, n_obs_types=16),
    "richobs":     dict(size=64, n_obs_types=64),
    "allcombined": dict(size=16, n_obs_types=64, action_mode="rotate",
                        obs_mode="ego", boundary="wall"),
}


@torch.no_grad()
def score(model, env, n_batches, bs, n_steps, dev, seed):
    model.eval()
    np.random.seed(seed)
    torch.manual_seed(seed)
    ok = tot = 0
    marg = Counter()
    for _ in range(n_batches):
        tokens, _om, rev, *_ = env.generate_batch(bs, n_steps)
        tokens = tokens.to(dev)
        m = rev[:, 1:].to(dev)
        if not m.any():
            continue
        pred = model(tokens[:, :-1]).argmax(-1)
        tgt = tokens[:, 1:]
        ok += int((pred[m] == tgt[m]).sum())
        tot += int(m.sum())
        marg.update(tgt[m].tolist())
    return ok / max(tot, 1), (max(marg.values()) / tot if tot else float("nan")), tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default=str(_REPO / "runs/knob_sweep"))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--n-batches", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--n-steps", type=int, default=128)
    ap.add_argument("--env-seed", type=int, default=10000)
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--out", default=str(_REPO / "KNOB_SWEEP.md"))
    args = ap.parse_args()
    dev = torch.device(args.device)

    res, floors = {}, {}
    for lbl, kw in CONDS.items():
        for v in ("Vanilla", "RoPE"):
            for s in args.seeds:
                ck = Path(args.runs_dir) / lbl / f"{v}_s{s}" / f"{v}.pt"
                if not ck.exists():
                    print(f"MISSING {ck}", flush=True)
                    continue
                blob = torch.load(ck, map_location="cpu", weights_only=False)
                cfg = blob["config"]
                env = GridWorld(p_empty=0.5, n_landmarks=0, seed=args.env_seed, **kw)
                m = VARIANT_MAP[v](vocab_size=cfg["vocab_size"], d_model=cfg["d_model"],
                                   n_heads=cfg["n_heads"], n_layers=cfg["n_layers"],
                                   grid_size=cfg["grid_size"]).to(dev)
                m.load_state_dict(blob["model_state_dict"])
                a, fl, n = score(m, env, args.n_batches, args.batch_size,
                                 args.n_steps, dev, 5000 + s)
                res.setdefault((lbl, v), []).append(a)
                floors.setdefault(lbl, []).append(fl)
                print(f"{lbl:12s} {v:8s} s{s} acc={a:.4f} floor={fl:.3f} n={n}", flush=True)
                del m
                torch.cuda.empty_cache()

    lines = ["# Which environment property flips the position effect?", "",
             "Position effect = **Vanilla (path-integrated) − RoPE (index)**, both "
             "trained in the same batch at every condition (rule 3), n=3, "
             "16 epochs, 1 layer. Knobs turned ONE AT A TIME from the torus "
             "baseline.", "",
             "Floors are **measured per condition** — they move a lot here, so an "
             "accuracy without its column is meaningless (rule 4).", "",
             "| condition | floor | Vanilla | RoPE (index) | **position effect** |",
             "|---|---|---|---|---|"]
    summary = {}
    for lbl in CONDS:
        v = np.array(res.get((lbl, "Vanilla"), []))
        r = np.array(res.get((lbl, "RoPE"), []))
        if not len(v) or not len(r):
            lines.append(f"| {lbl} | — | — | — | — |")
            continue
        fl = float(np.mean(floors[lbl]))
        eff = v.mean() - r.mean()
        summary[lbl] = {"floor": fl, "vanilla": v.tolist(), "rope": r.tolist(),
                        "effect": float(eff)}
        lines.append(f"| {lbl} | {fl:.3f} | {v.mean():.3f} ± {v.std(ddof=1):.3f} "
                     f"| {r.mean():.3f} ± {r.std(ddof=1):.3f} | **{eff:+.3f}** |")
    lines += ["", "Reference points: torus paper task **+0.461** (n=8), "
              "MiniGrid-DoorKey-16x16 **−0.060** (n=3, frequency-controlled).", ""]
    Path(args.out).write_text("\n".join(lines) + "\n")
    json.dump(summary, open(str(args.out).replace(".md", ".json"), "w"), indent=2)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
