"""Aggregate MiniWorld factorial: non-blank accuracy per (variant, encoding),
and the POSITION EFFECT (path-integrated - index) under raw vs allocentric.
The headline question: does the position effect flip from <=0 (raw rotation
actions) to positive (allocentric displacement), as on MiniGrid?"""
import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np

_REPO = os.path.dirname(os.path.abspath(__file__))
PATHINT = {"Vanilla", "MapPoPE-Flat"}          # path-integrated arms
INDEX = {"RoPE", "PoPE-Flat"}                   # sequence-index arms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default=os.path.join(_REPO, "runs", "miniworld"))
    ap.add_argument("--variants", nargs="+",
                    default=["Vanilla", "MapPoPE-Flat", "RoPE", "PoPE-Flat"])
    ap.add_argument("--length", type=int, default=512)
    ap.add_argument("--out", default=os.path.join(_REPO, "MINIWORLD_RESULTS.md"))
    args = ap.parse_args()

    # data[variant][enc] = list of nb_acc over seeds
    data = defaultdict(lambda: defaultdict(list))
    for f in glob.glob(os.path.join(args.runs_dir, "s*", "*_*.json")):
        base = os.path.basename(f)[:-5]           # strip .json
        variant, enc = base.rsplit("_", 1)         # <variant>_<raw|allo>
        if variant not in args.variants or enc not in ("raw", "allo"):
            continue
        r = json.load(open(f))
        key = str(args.length)
        if key in r:
            data[variant][enc].append(r[key]["nb_acc"])

    def cell(v, e):
        a = np.array(data[v][e])
        return f"{a.mean():.3f} ± {a.std(ddof=1):.3f}" if len(a) > 1 else (f"{a.mean():.3f}" if len(a) else "—")

    def grp(names, e):
        vals = [x for v in names for x in data[v][e]]
        return np.array(vals)

    lines = [f"# MiniWorld continuous-3D — non-blank accuracy at T={args.length}", "",
             "Held-out fresh obs_map. chance (non-blank) = 1/16 = 0.0625. Path-integrated "
             "= {Vanilla, MapPoPE-Flat}; index = {RoPE, PoPE-Flat}.", "",
             "| variant | position | raw | allocentric |",
             "|---|---|---|---|"]
    for v in args.variants:
        pos = "path-int" if v in PATHINT else "index"
        lines.append(f"| {v} | {pos} | {cell(v,'raw')} | {cell(v,'allo')} |")

    praw, pallo = grp(PATHINT, "raw"), grp(PATHINT, "allo")
    iraw, iallo = grp(INDEX, "raw"), grp(INDEX, "allo")
    eff_raw = praw.mean() - iraw.mean() if len(praw) and len(iraw) else float("nan")
    eff_allo = pallo.mean() - iallo.mean() if len(pallo) and len(iallo) else float("nan")
    lines += ["", "## Position effect (path-integrated − index)", "",
              "| encoding | path-int mean | index mean | position effect |",
              "|---|---|---|---|",
              f"| raw (turn/forward) | {praw.mean():.3f} | {iraw.mean():.3f} | **{eff_raw:+.3f}** |",
              f"| allocentric (displacement) | {pallo.mean():.3f} | {iallo.mean():.3f} | **{eff_allo:+.3f}** |",
              "",
              f"**Flip:** raw {eff_raw:+.3f} → allocentric {eff_allo:+.3f} "
              f"({'CONFIRMS the MiniGrid result — allocentric restores path integration in continuous 3D' if eff_allo > eff_raw + 0.02 else 'no clear flip'})."]
    open(args.out, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
