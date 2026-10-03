"""Readouts for LEAK_PREREG.md."""
import json
import numpy as np
import torch

from mapformer.leak_eval import evaluate_run
from mapformer.stats_core import classify_run, perm2_p

REPO = "/home/prashr/mapformer"; R = f"{REPO}/runs/leak/p0"; S = range(8); ARMS = ("MapWM", "ActOnly", "NormStep")


def main():
    res, cls = {}, {}
    for a in ARMS:
        res[a], cls[a] = [], []
        for s in S:
            ck = f"{R}/{a}_s{s}/{a}.pt"
            res[a].append(evaluate_run(ck, a))
            c = classify_run(torch.load(ck, map_location="cpu", weights_only=False)["losses"]); cls[a].append(c)
    json.dump({"res": res, "classes": {a: [c["cls"] for c in cls[a]] for a in ARMS}}, open(f"{REPO}/LEAK.json", "w"), indent=1)
    g = lambda a, key, f="intact": [r[key][f] for r in res[a]]
    print("== object-identity accuracy, test pool (mean +/- sd over 8 seeds); leak L = zeroed - intact (median) ==")
    for a in ARMS:
        print(f"  {a:8s} " + " | ".join(f"x{s}: {np.mean(g(a, f'test|x{s}')):.4f} +/- {np.std(g(a, f'test|x{s}'), ddof=1):.4f} "
                                       f"L {np.median(g(a, f'test|x{s}', 'leak')):+.4f}" for s in (1, 2, 4))
              + f" | train x1 {np.mean(g(a, 'train|x1')):.4f} | classes " + " ".join(f"{c['cls'][:4]}({c['tail']:.3f})" for c in cls[a]))
    for R_ in ("ActOnly", "NormStep"):
        out = {}
        for s in (1, 2, 4):
            a, b = g(R_, f"test|x{s}"), g("MapWM", f"test|x{s}")
            out[s] = (np.mean(a) - np.mean(b), perm2_p(b, a)["p"])
        d4, p4 = out[4]; d1, p1 = out[1]; L4 = np.median(g(R_, "test|x4", "leak"))
        x4_fires = p4 < 0.05 and d4 >= 0.02
        cost = p1 < 0.05 and d1 <= -0.01
        if x4_fires and L4 <= 0.01 and not cost:
            v = "REMEDY"
        elif x4_fires and L4 <= 0.01 and cost:
            v = "REMEDY WITH A COST"
        elif not x4_fires and L4 > 0.05:
            v = "NO REMEDY"
        else:
            v = "no registered branch -- reported as it falls"
        print(f"\n  {R_} - MapWM: x1 {d1:+.4f} (p {p1:.4f}) | x2 {out[2][0]:+.4f} (p {out[2][1]:.4f}) | x4 {d4:+.4f} (p {p4:.4f}) | "
              f"median L(x4) {L4:+.4f}\n  REGISTERED {R_}: {v}")
    Lact = [abs(x) for s in (1, 2, 4) for x in g("ActOnly", f"test|x{s}", "leak")]
    print(f"\n== void check: ActOnly max |L| {max(Lact):.4f} (must be <= 0.002) -> {'OK' if max(Lact) <= 0.002 else 'VOID'}")


if __name__ == "__main__":
    main()
