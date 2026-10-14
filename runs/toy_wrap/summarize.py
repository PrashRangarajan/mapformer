"""Summarise every toy run (exploratory): per (mode, init, scale, clamp) the SOLVED count (retrieval acc >= 0.99) and the
attention-weighted share of channels whose two move angles cancel mod 2 pi (< 0.1 rad)."""
import json, glob, collections, numpy as np
g = collections.defaultdict(list)
for f in sorted(glob.glob("/home/prashr/mapformer/runs/toy_wrap/*/*.json")):
    r = json.load(open(f)); a = r["args"]
    if "w" not in r:
        continue
    A = np.array(r["angles"]); w = np.array(r["w"])
    sig = np.abs(np.angle(np.exp(1j * (A[0] + A[1])))); big = np.abs(np.angle(np.exp(1j * A[0]))) > 0.05; ww = w * big
    key = (a["mode"], a["init"], a.get("init_scale", 1.0), not a.get("no_omega_clamp", False))
    g[key].append((a["seed"], r["final_acc"], (ww * (sig < 0.1)).sum() / ww.sum()))
for k, v in sorted(g.items()):
    v = sorted(set(v)); acc = [x[1] for x in v]
    print(f"{k[0]:6s} {k[1]:8s} x{k[2]} clamp={k[3]!s:5s} n={len(v):2d} SOLVED {sum(x >= 0.99 for x in acc):2d}/{len(v):2d} "
          f"acc median {np.median(acc):.3f} [{min(acc):.2f}, {max(acc):.2f}] weighted cancel {np.mean([x[2] for x in v]):.2f}")
