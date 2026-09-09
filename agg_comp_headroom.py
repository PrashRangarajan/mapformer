"""Aggregate the compositional headroom batch (COMP_HEADROOM_PREREG.md).

`agg_comp_multiseed.py` cannot be reused: it expects `seed<N>/<Variant>.pt` and
labels rows by the checkpoint filename, but three of the four arms here are the SAME
variant at different recipes, so the filename does not identify the arm. This calls
`eval_compositional.eval_ckpt` -- the same evaluator, fresh held-out env at
seed=10000 -- so the metrics are identical to the published table and only the
row labelling differs.
"""
import argparse, json, os
import numpy as np
import torch

from mapformer.eval_compositional import eval_ckpt

import os as _os
if _os.environ.get("HIER_RECHECK"):
    ARMS = [("Hourglass_k2", "Hourglass_k2", "MapWM-Hier, cosine/1e-3/150ep"),
            ("HourglassFlat3", "HourglassFlat3", "MapWM-FlatHG, cosine/1e-3/150ep")]
    _SUB = "hier_recheck"
else:
    ARMS = [("A", "Hourglass_k2", "published recipe: linear, 3e-4, 50ep"),
            ("B", "Hourglass_k2", "cosine, 1e-3, 50ep (budget held)"),
            ("C", "Hourglass_k2", "cosine, 1e-3, 150ep"),
            ("D", "LoopedHourglass", "loop, cosine, 1e-3, 150ep (1/3 the params)")]
    _SUB = "comp_headroom"


REPO = "/home/prashr/mapformer"   # absolute: `python3 -m mapformer.X` runs from the
                                  # PARENT dir, so relative paths resolve there and
                                  # silently find nothing. This has cost four
                                  # debugging rounds in this project.


def final_loss(tag, seed):
    f = f"{REPO}/runs/{_SUB}/logs/{tag}_s{seed}.log"
    if not os.path.exists(f):
        return float("nan")
    v = [l for l in open(f) if "final_loss=" in l]
    return float(v[-1].split("final_loss=")[1].split()[0]) if v else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=list(range(8)))
    ap.add_argument("--lengths", nargs="+", type=int, default=[256, 512])
    ap.add_argument("--n-traj", type=int, default=200)   # MATCHES agg_comp_multiseed, so arm A is comparable to the published table
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="COMP_HEADROOM.md")
    a = ap.parse_args()

    res, loss = {}, {}
    for tag, var, _ in ARMS:
        for T in a.lengths:
            res[(tag, T)] = []
        loss[tag] = []
        for s in a.seeds:
            ck = f"{REPO}/runs/{_SUB}/{tag}_s{s}/{var}.pt"
            if not os.path.exists(ck):
                print(f"  missing {ck}", flush=True); continue
            _variant, r = eval_ckpt(ck, a.lengths, a.n_traj, a.device)  # returns (variant, results)
            for T in a.lengths:
                res[(tag, T)].append(r[T]["cross_nb_acc"])
            loss[tag].append(final_loss(tag, s))
            print(f"{tag} s{s}: " + " ".join(f"T{T}={r[T]['cross_nb_acc']:.3f}"
                                            for T in a.lengths), flush=True)

    def arr(t, T): return np.array(res[(t, T)], dtype=float)
    L = ["# Compositional headroom: is 0.415 a capability limit or a recipe limit?", "",
         "`cross_nb` on a fresh held-out environment (seed 10000), the same evaluator "
         "as the published table. Floor ~0.072, ceiling 1.0. "
         "Pre-registration: `COMP_HEADROOM_PREREG.md`.", "",
         "| arm | recipe | final loss | " + " | ".join(f"cross_nb T={T}" for T in a.lengths) + " | n |",
         "|---|---|---|" + "---|" * (len(a.lengths) + 1)]
    for tag, var, desc in ARMS:
        cells = " | ".join(
            f"**{arr(tag,T).mean():.3f} ± {arr(tag,T).std(ddof=1):.3f}**" if len(arr(tag,T)) > 1 else "—"
            for T in a.lengths)
        lm = np.nanmean(loss[tag]) if loss[tag] else float("nan")
        L.append(f"| **{tag}** | {desc} | {lm:.4f} | {cells} | {len(arr(tag,a.lengths[0]))} |")

    L += ["", "## Pre-registered contrasts", "",
          "| contrast | tests | delta | sd | MDE | seeds + | verdict |",
          "|---|---|---|---|---|---|---|"]
    for T in a.lengths:
        for nm, x, y, what in [("C - A", "C", "A", "P1: the whole recipe"),
                               ("B - A", "B", "A", "P2: schedule + lr alone"),
                               ("C - B", "C", "B", "P2: budget alone"),
                               ("D - C", "D", "C", "P4: the loop, at 1/3 params")]:
            u, v = arr(x, T), arr(y, T)
            if len(u) < 2 or len(v) < 2 or len(u) != len(v):
                continue
            d = u - v; sd = d.std(ddof=1); mde = 2.8 * sd / np.sqrt(len(d))
            L.append(f"| `{nm}` @ T={T} | {what} | {d.mean():+.3f} | {sd:.3f} | {mde:.3f} | "
                     f"{int((d>0).sum())}/{len(d)} | "
                     f"{'**DETECTABLE**' if abs(d.mean())>mde else 'unmeasured'} |")

    L += ["", "## P3: does the recipe compress spread?", "",
          "| arm | sd at T=%d |" % a.lengths[0], "|---|---|"]
    for tag, _, _ in ARMS:
        u = arr(tag, a.lengths[0])
        if len(u) > 1:
            L.append(f"| {tag} | {u.std(ddof=1):.3f} |")
    L += ["", "Pre-registered: the recipe should take sd from $0.096$ to below $0.05$. "
          "A mean gain with unchanged variance means something other than optimisation.",
          "", "**A is the reproduction control.** If it does not land near $0.415$ the "
          "batch is not comparable to the published table and nothing above is readable."]
    open(a.out, "w").write("\n".join(L) + "\n")
    json.dump({f"{k[0]}|{k[1]}": v for k, v in res.items()}, open(a.out.replace(".md", ".json"), "w"), indent=2)
    print("\n".join(L))


if __name__ == "__main__":
    main()
