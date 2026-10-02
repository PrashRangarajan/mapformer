"""Readouts for RANK_WRAP_PREREG.md: dimension x wrap share at per-head rank D+1."""
import json
import numpy as np
import torch

from mapformer.analyze_rank_nd_secondary import strat_acc
from mapformer.environment_nd import GridWorldND
from mapformer.stats_core import classify_run, perm2_p, fisher_solved
from mapformer.train_variant import VARIANT_MAP

REPO = "/home/prashr/mapformer"; R = f"{REPO}/runs/rank_wrap"; S = list(range(8))
CELLS = {"2L": (2, 32, "Vanilla_r3ph"), "2H": (2, 10, "Vanilla_r3ph"), "3L": (3, 18, "Vanilla_r4ph"),
         "3H": (3, 10, "Vanilla_r4ph")}
PRIOR = {"2L": f"{REPO}/runs/rank_nd/D2/Vanilla_r3ph_s{{s}}/Vanilla_r3ph.pt",
         "3H": f"{REPO}/runs/rank_nd/D3/Vanilla_r4ph_s{{s}}/Vanilla_r4ph.pt"}


def main():
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    solved, acc, out = {}, {}, {}
    print("== cells (SOLVED = final-5% loss < 0.05); T=1024 held-out acc; [T=2048]; secondary: wrap-only / other, own map ==")
    for k, (D, N, v) in CELLS.items():
        J = json.load(open(f"{R}/N{N}/EVAL_D{D}.json"))[f"D{D}"]["acc"][v]
        acc[k] = [J["1024"][str(s)] for s in S]; a2 = np.mean([J["2048"][str(s)] for s in S])
        cl, sec = [], []
        held, rows = GridWorldND(dims=D, size=N, seed=10000), []
        for s in S:
            b = torch.load(f"{R}/N{N}/D{D}/{v}_s{s}/{v}.pt", map_location="cpu", weights_only=False)
            cl.append(classify_run(b["losses"])); c = b["config"]
            m = VARIANT_MAP[v](vocab_size=c["vocab_size"], d_model=c["d_model"], n_heads=c["n_heads"],
                               n_layers=c["n_layers"], grid_size=c["grid_size"])
            m.load_state_dict(b["model_state_dict"]); m.to(dev).eval()
            h, ht = strat_acc(m, held, 1024, 40, 10**6 + s, dev)
            o, ot = strat_acc(m, GridWorldND(dims=D, size=N, seed=s), 1024, 40, 10**6 + s, dev)
            oall = (o["wrap"] * ot["wrap"] + o["near"] * ot["near"]) / max(ot["wrap"] + ot["near"], 1)
            rows.append({"seed": s, "wrap": h["wrap"], "other": h["near"], "own": oall,
                         "wrap_share": ht["wrap"] / max(ht["wrap"] + ht["near"], 1)})
            if k in PRIOR:
                p = np.array(torch.load(PRIOR[k].format(s=s), map_location="cpu", weights_only=False)["losses"])
                rows[-1]["determinism_max_diff"] = float(np.abs(np.array(b["losses"]) - p).max())
        solved[k] = sum(c["registered"] == "SOLVED" for c in cl); out[k] = rows
        mean = lambda f: np.mean([r[f] for r in rows])
        det = f" | determinism max loss diff vs RANK_ND {max(r['determinism_max_diff'] for r in rows):.1e}" if k in PRIOR else ""
        print(f"  {k} D={D} N={N:2d} {v}: SOLVED {solved[k]}/8 acc {np.mean(acc[k]):.3f} +/- {np.std(acc[k], ddof=1):.3f} "
              f"[{a2:.3f}] | wrap share {mean('wrap_share'):.2f}, wrap-only {mean('wrap'):.3f}, other {mean('other'):.3f}, "
              f"own map {mean('own'):.3f}{det}")
    def fires(better, worse):
        pf = fisher_solved(solved[worse], 8, solved[better], 8); pp = perm2_p(acc[worse], acc[better])["p"]
        d = np.mean(acc[better]) - np.mean(acc[worse])
        f = (pf < 0.05 and solved[better] > solved[worse]) or (pp < 0.05 and d > 0)
        print(f"  {better} over {worse}: SOLVED {solved[better]}/8 vs {solved[worse]}/8 Fisher p {pf:.4f} | acc {d:+.3f} perm p {pp:.4f} | {'FIRES' if f else 'does not fire'}")
        return f
    print("\n== wrap effect within D (low wrap better) ==")
    w2, w3 = fires("2L", "2H"), fires("3L", "3H")
    print("== dimension effect within wrap level (2D better) ==")
    dl, dh = fires("2L", "3L"), fires("2H", "3H")
    if w2 and w3 and not dl and not dh:
        v = "WRAP DRIVES IT"
    elif dl and dh and not w2 and not w3:
        v = "DIMENSION DRIVES IT"
    elif w2 and w3 and dl and dh:
        v = "BOTH"
    else:
        v = "no registered branch -- reported as it falls"
    print(f"\n== REGISTERED: {v}")
    json.dump({"solved": solved, "acc": acc, "secondary": out}, open(f"{REPO}/RANK_WRAP.json", "w"), indent=1)


if __name__ == "__main__":
    main()
