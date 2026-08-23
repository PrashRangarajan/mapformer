"""The torus 2x2 evaluated on MiniGrid-DoorKey-16x16, at n=3.

The torus version of this grid (`INDEX_BASELINE_PAPER_TASK_n8.md`) is
unambiguous: index-position arms sit on the measured blank floor (0.509-0.534
against 0.506) while path-integrated arms score 0.967-0.994. `MINIGRID_DK16_
RESULTS.md`, at n=1, reports the opposite ordering at long horizon -- Vanilla
0.754 vs RoPE 0.877 at T=512. This tests that at n=3 with all four arms trained
in one batch.

Mechanism hypothesis, stated before the numbers: MiniGrid's actions are
0=turn-left, 1=turn-right, 2=forward -- rotations plus a HEADING-DEPENDENT
translation. MapFormer's path integrator maps each action token to a fixed
Lie-algebra element and cumsums it, which assumes actions ARE translations. On
MiniGrid "forward" displaces you in a direction set by the accumulated history of
turns, so the assumption is violated and the position code should be actively
misleading rather than merely unhelpful.

A FLOOR IS MEASURED HERE, which the original inline evaluation did not do
(standing rule 4). On this task the scored targets are observation tokens whose
marginal is dominated by whatever the egocentric view most often shows; without
that number an accuracy of 0.877 cannot be read at all.
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from mapformer.minigrid_env import MiniGridWorld
from mapformer.train_variant import VARIANT_MAP

_REPO = Path(__file__).resolve().parent
DISPLAY = {"Vanilla": "MapWM-Flat (RoPE + path int.)",
           "Hourglass_k2": "MapWM-Hier (RoPE + path int. + hier)",
           "MapPoPE-Flat": "MapPoPE-Flat (PoPE + path int.)",
           "MapPoPE-Hier": "MapPoPE-Hier (PoPE + path int. + hier)",
           "RoPE": "RoPE-Flat (index)",
           "PlainHourglass": "RoPE-Hier (index + hier)",
           "PoPE-Flat": "PoPE-Flat (PoPE + index)",
           "PoPE-Hier": "PoPE-Hier (PoPE + index + hier)",
           "Vanilla_FixedOmega": "MapWM-Flat, omega FROZEN"}


@torch.no_grad()
def evaluate(model, env, T, n_trials, device, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    ok = tot = 0
    nll = 0.0
    marg = Counter()
    for _ in range(n_trials):
        tokens, _om, rm = env.generate_trajectory(T)
        tt = tokens.unsqueeze(0).to(device)
        lp = F.log_softmax(model(tt[:, :-1]), dim=-1)
        pred = lp.argmax(-1)[0]
        tgt = tt[0, 1:]
        m = rm[1:].to(device)
        if m.sum() == 0:
            continue
        ok += int((pred[m] == tgt[m]).sum())
        tot += int(m.sum())
        idx = torch.arange(lp.shape[1], device=device)[m]
        nll += float(-lp[0, idx, tgt[m]].sum())
        marg.update(tgt[m].tolist())
    floor = (max(marg.values()) / tot) if tot else float("nan")
    return (ok / max(tot, 1), nll / max(tot, 1), floor, tot)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default=str(_REPO / "runs/minigrid_2x2"))
    ap.add_argument("--variants", nargs="+",
                    default=["Vanilla", "MapPoPE-Flat", "RoPE", "PoPE-Flat"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--lengths", nargs="+", type=int, default=[128, 512, 1024])
    ap.add_argument("--env-seed", type=int, default=10000)
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--out", default=str(_REPO / "MINIGRID_2X2.md"))
    args = ap.parse_args()
    dev = torch.device(args.device)
    N = {128: 100, 512: 50, 1024: 25, 2048: 12}

    acc = {}
    floors = {}
    for v in args.variants:
        for s in args.seeds:
            ck = Path(args.runs_dir) / f"{v}_s{s}" / f"{v}.pt"
            if not ck.exists():
                print(f"MISSING {ck}", flush=True)
                continue
            blob = torch.load(ck, map_location="cpu", weights_only=False)
            cfg = blob["config"]
            m = VARIANT_MAP[v](vocab_size=cfg["vocab_size"],
                               d_model=cfg.get("d_model", 128),
                               n_heads=cfg.get("n_heads", 2),
                               n_layers=cfg.get("n_layers", 1),
                               grid_size=cfg.get("grid_size", 16)).to(dev)
            m.load_state_dict(blob["model_state_dict"])
            m.eval()
            for T in args.lengths:
                env = MiniGridWorld(env_name="MiniGrid-DoorKey-16x16-v0",
                                    tokenization="obj_color", seed=args.env_seed)
                a, nl, fl, n = evaluate(m, env, T, N.get(T, 25), dev, 2000 + s)
                acc.setdefault((v, T), []).append(a)
                floors.setdefault(T, []).append(fl)
                print(f"{v:14s} s{s} T={T:5d}  acc={a:.4f} nll={nl:.3f} "
                      f"floor={fl:.3f} n={n}", flush=True)
            del m
            torch.cuda.empty_cache()

    fl = {T: float(np.mean(floors[T])) for T in floors}
    lines = ["# The torus 2x2 on MiniGrid-DoorKey-16x16 (n=3)", "",
             "External, published benchmark; egocentric observation (the cell in "
             "front of the agent). All four arms trained in ONE batch (rule 3), "
             "50 epochs, 25K cached trajectory buffer.", "",
             "**Measured floor** (most common scored target) per length: "
             + ", ".join(f"T={T}: **{fl[T]:.3f}**" for T in args.lengths)
             + ". The original n=1 evaluation reported no floor at all.", "",
             "| model | position | " + " | ".join(f"T={T}" for T in args.lengths) + " |",
             "|---" * (len(args.lengths) + 2) + "|"]
    summary = {}
    for v in args.variants:
        cells = []
        summary[v] = {}
        for T in args.lengths:
            a = np.array(acc.get((v, T), []))
            if len(a) == 0:
                cells.append("—")
                continue
            summary[v][T] = {"mean": float(a.mean()),
                             "sd": float(a.std(ddof=1)) if len(a) > 1 else 0.0,
                             "per_seed": a.tolist()}
            cells.append(f"{a.mean():.3f} ± {a.std(ddof=1) if len(a)>1 else 0:.3f}")
        name = DISPLAY.get(v, v)
        pos = ("path-integrated" if v in ("Vanilla", "MapPoPE-Flat",
               "Hourglass_k2", "MapPoPE-Hier",
               "Vanilla_FixedOmega") else "**index**")
        lines.append(f"| {name} | {pos} | " + " | ".join(cells) + " |")
    lines += ["", "| *measured floor* | | "
              + " | ".join(f"*{fl[T]:.3f}*" for T in args.lengths) + " |", "",
              "## The torus comparison", "", "For reference, the identical 2x2 on "
              "the 64x64 torus paper task at n=8 "
              "(`INDEX_BASELINE_PAPER_TASK_n8.md`), floor 0.506:", "",
              "| | index | path-integrated |", "|---|---|---|",
              "| RoPE encoding | 0.530 | **0.967** |",
              "| PoPE encoding | 0.509 | **0.994** |", "",
              "Both index arms sit ON the floor there. Whether that survives on "
              "an egocentric, rotation-actioned environment is what this file "
              "measures."]
    Path(args.out).write_text("\n".join(lines) + "\n")
    json.dump({"acc": summary, "floors": fl},
              open(str(args.out).replace(".md", ".json"), "w"), indent=2)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
