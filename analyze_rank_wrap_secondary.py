"""Amendment-1 readouts for RANK_WRAP_PREREG.md (written before any result was read). Reads the registered
outputs (RANK_WRAP.json, runs/rank_wrap/N*/EVAL_D*.json) and the checkpoints; adds the INTERACTION branch,
the |d| >= 0.02 rule for accuracy-only firing, within-revisit-type tests, reweighting, floor-relative
accuracy, per-seed classes, reversals and the enforced determinism void."""
import json
import numpy as np
import torch

from mapformer.stats_core import classify_run, perm2_p, fisher_solved

REPO = "/home/prashr/mapformer"; R = f"{REPO}/runs/rank_wrap"; S = list(range(8))
CELLS = {"2L": (2, 32, "Vanilla_r3ph"), "2H": (2, 10, "Vanilla_r3ph"), "3L": (3, 18, "Vanilla_r4ph"),
         "3H": (3, 10, "Vanilla_r4ph")}
RETRACE = {"2L": 0.750, "2H": 0.589, "3L": 0.818, "3H": 0.653}


def main():
    W = json.load(open(f"{REPO}/RANK_WRAP.json")); sec = W["secondary"]
    acc = {k: W["acc"][k] for k in CELLS}; solved = W["solved"]
    print("== per seed: class(final-5% loss) held-out acc ==")
    for k, (D, N, v) in CELLS.items():
        cl = [classify_run(torch.load(f"{R}/N{N}/D{D}/{v}_s{s}/{v}.pt", map_location="cpu", weights_only=False)["losses"]) for s in S]
        flag = [s for s, c in zip(S, cl) if c["registered"] == "SOLVED" and acc[k][s] < 0.95]
        rel = [(a - RETRACE[k]) / (1 - RETRACE[k]) for a in acc[k]]
        rw = [0.5 * r["wrap"] + 0.5 * r["other"] for r in sec[k]]
        print(f"  {k}: " + " ".join(f"{c['cls'][:4]}({c['tail']:.3f}) {a:.3f}" for c, a in zip(cl, acc[k])))
        print(f"      floor-relative {np.mean(rel):+.3f} | reweighted to wrap 0.5 {np.mean(rw):.3f} | "
              f"SOLVED-but-held<0.95 {flag}")
        if k in ("2L", "3H"):
            diffs = [r.get("determinism_max_diff") for r in sec[k]]
            bad = [s for s, d in zip(S, diffs) if d is None or d != 0.0]
            print(f"      determinism vs RANK_ND per seed {diffs} -> {'VOID: ' + str(bad) if bad else 'OK'}")

    def contrast(better, worse):
        pf = fisher_solved(solved[worse], 8, solved[better], 8); pp = perm2_p(acc[worse], acc[better])["p"]
        d = float(np.mean(acc[better]) - np.mean(acc[worse]))
        by_solved = pf < 0.05 and solved[better] > solved[worse]
        by_acc = pp < 0.05 and d >= 0.02
        rev = (pf < 0.05 and solved[better] < solved[worse]) or (pp < 0.05 and d < 0)
        ties = sum(a == 1.0 for a in acc[better]), sum(a == 1.0 for a in acc[worse])
        out = []
        for typ in ("wrap", "other"):
            a = [r[typ] for r in sec[better]]; b = [r[typ] for r in sec[worse]]
            out.append(f"{typ} {np.mean(a) - np.mean(b):+.3f} (p {perm2_p(b, a)['p']:.4f})")
        f = by_solved or by_acc
        print(f"  {better} over {worse}: SOLVED {solved[better]}/8 vs {solved[worse]}/8 p {pf:.4f} | acc {d:+.3f} p {pp:.4f} "
              f"| at 1.000: {ties[0]} vs {ties[1]} | within type: {', '.join(out)} | "
              f"{'FIRES' if f else ('REVERSED' if rev else 'unmeasured')}")
        return f
    print("\n== amended contrasts (accuracy-only firing needs |d| >= 0.02) ==")
    w2, w3 = contrast("2L", "2H"), contrast("3L", "3H")
    dl, dh = contrast("2L", "3L"), contrast("2H", "3H")
    if w2 and w3 and not dl and not dh:
        v = "WRAP DRIVES IT"
    elif dl and dh and not w2 and not w3:
        v = "DIMENSION DRIVES IT"
    elif w2 and w3 and dl and dh:
        v = "BOTH"
    elif w3 and dh and not w2 and not dl:
        v = "INTERACTION (only the 3D high-wrap cell is impaired)"
    else:
        v = "no branch"
    print(f"\n== AMENDED READING (Amendment 1, not the registered verdict): {v}")


if __name__ == "__main__":
    main()
