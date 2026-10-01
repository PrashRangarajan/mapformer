"""Readouts for CTXSTEP_PREREG.md."""
import json
import numpy as np
import torch

from mapformer.stats_core import classify_run, perm2_p

REPO = "/home/prashr/mapformer"; R = f"{REPO}/runs/ctxstep/p0"; S = range(6, 14)


def main():
    sw = {}
    for l in open(f"{REPO}/CTXSTEP_SWAP.jsonl"):
        r = json.loads(l); k = r["ckpt"].rstrip("/").split("/")[-2]; sw[k] = r
    acc, ood, ratio, step, solved, tail, alpha = {}, {}, {}, {}, {}, {}, {}
    cells = [(c, d, a) for c in ("lead", "trail") for d, a in
             (("far", "CF"), ("far", "CG"), ("far", "SR"), ("far", "HSR"), ("near", "CG"), ("near", "SR"))]
    for c, d, a in cells:
        key = (c, d, a); acc[key], ood[key], ratio[key], step[key], solved[key], tail[key] = [], [], [], [], 0, []
        for s in S:
            run = f"{c}_{d}_{a}_s{s}"; e = json.load(open(f"{R}/{run}/eval.json"))
            acc[key].append(e["eval"]["2048"]["acc"]); ood[key].append(e["eval"]["4096"]["acc"])
            ratio[key].append(sw[run]["ratio"]); step[key].append(sw[run]["move"] >= 0.01)
            blob = torch.load(f"{R}/{run}/{a}.pt", map_location="cpu", weights_only=False)
            cl = classify_run(blob["losses"]); solved[key] += cl["registered"] == "SOLVED"; tail[key].append(cl["tail"])
            if a == "HSR":
                alpha.setdefault(key, []).append(float(blob["model_state_dict"]["ctx_alpha"]))
    print("== per cell: T=2048 acc mean +/- sd | SOLVED | median decoy/move ratio | learned a step | [T=4096] ==")
    for k in cells:
        print(f"  {k[0]:5s} {k[1]:4s} {k[2]:3s} {np.mean(acc[k]):.3f} +/- {np.std(acc[k], ddof=1):.3f} | {solved[k]}/8 | "
              f"{np.median(ratio[k]):.2f} | {sum(step[k])}/8 | [{np.mean(ood[k]):.3f}]   r(loss,acc) "
              f"{np.corrcoef(tail[k], acc[k])[0, 1] if np.std(acc[k]) > 0 else float('nan'):+.2f}")
    print("\n== H-W: window limit (near - far, per window arm and cue) ==")
    hw = []
    for a in ("CG", "SR"):
        for c in ("lead", "trail"):
            n, f = (c, "near", a), (c, "far", a)
            d = np.mean(acc[n]) - np.mean(acc[f]); p = perm2_p(acc[f], acc[n])["p"]
            ok = p < 0.05 and d > 0 and np.median(ratio[n]) <= 0.3 and np.median(ratio[f]) >= 0.7
            hw.append(ok)
            print(f"  {a} {c}: near - far {d:+.3f} perm p {p:.4f}; median ratio near {np.median(ratio[n]):.2f} far "
                  f"{np.median(ratio[f]):.2f} -> {'holds' if ok else 'does not hold'}")
    v = "WINDOW-LIMITED" if all(hw) else ("PARTLY WINDOW-LIMITED" if sum(hw) >= 2 else "no registered branch")
    print(f"  REGISTERED H-W: {v} ({sum(hw)}/4)")
    print("\n== H-R: reach (far: HSR vs the better window arm) ==")
    hr = []
    for c in ("lead", "trail"):
        h = (c, "far", "HSR"); best = max(("CG", "SR"), key=lambda a: np.mean(acc[(c, "far", a)])); b = (c, "far", best)
        d = np.mean(acc[h]) - np.mean(acc[b]); p = perm2_p(acc[b], acc[h])["p"]
        ok = p < 0.05 and d > 0 and np.median(ratio[h]) <= 0.3 and sum(step[h]) >= 7
        hr.append(ok)
        pc = perm2_p(acc[(c, "far", "CF")], acc[h])["p"]
        print(f"  {c}: HSR - {best} {d:+.3f} perm p {p:.4f}; HSR median ratio {np.median(ratio[h]):.2f}, step {sum(step[h])}/8 "
              f"-> {'holds' if ok else 'does not hold'}   [secondary HSR - CF {np.mean(acc[h]) - np.mean(acc[(c, 'far', 'CF')]):+.3f} p {pc:.4f}; "
              f"alpha {np.round(alpha[h], 2).tolist()}]")
    v = "ATTENTION REACHES" if all(hr) else ("REACHES ON ONE SIDE" if sum(hr) == 1 else "no registered branch")
    print(f"  REGISTERED H-R: {v}")
    cf = [r for c in ("lead", "trail") for r in ratio[(c, "far", "CF")]]
    print(f"\n== void check: CF ratios {min(cf):.3f}-{max(cf):.3f} (must be 1.00 +/- 0.01) "
          f"-> {'OK' if all(abs(x - 1) <= 0.01 for x in cf) else 'VOID'}")


if __name__ == "__main__":
    main()
