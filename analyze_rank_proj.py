"""Readouts for RANK_PROJ_PREREG.md: frozen existence, trainable stability vs the r=4 control."""
import json
import numpy as np
import torch
from scipy.stats import fisher_exact

from mapformer.analyze_rank_matched import classify, perm

REPO = "/home/prashr/mapformer"; S = list(range(8))


def runs(d, v):
    out = {}
    for s in S:
        b = torch.load(f"{REPO}/runs/{d}/p0/{v}_s{s}/{v}.pt", map_location="cpu", weights_only=False)
        l = np.asarray(b["losses"], float)
        below = np.nonzero(l < 0.05)[0]
        peak = int(l.argmax())
        rec = next((int(i) for i in below if i > peak), None)
        out[s] = dict(cls=classify(l)[0], tail=classify(l)[1], peak=float(l.max()), peak_ep=peak + 1,
                      recover_ep=None if rec is None else rec + 1)
    return out


def acc(js, v, T=1024):
    J = json.load(open(js)); return [dict((x[0], x) for x in J[f"0.0|{v}|{T}"])[s][1] for s in S]


def main():
    print("== FROZEN: projected r=2 vs its source r=4, T=1024 ==")
    fr = acc(f"{REPO}/RANK_PROJ_FROZEN.json", "Vanilla"); src = acc(f"{REPO}/RANK_MATCHED_e900.json", "Vanilla_r4")
    for s in S:
        print(f"  s{s}  r4 source {src[s]:.4f}  -> projected r2 {fr[s]:.4f}  ({fr[s] - src[s]:+.4f})")
    print(f"  mean {np.mean(src):.4f} -> {np.mean(fr):.4f}; projected >= 0.95 on {sum(x >= 0.95 for x in fr)}/8")

    T = runs("rank_proj_train", "Vanilla"); C = runs("rank_matched_e900c", "Vanilla_r4")
    R = runs("rank_matched_e900c", "Vanilla")
    print("\n== TRAINABLE (r2 from projection) vs CONTROL (r4 from its own solution) vs r2 continuation ==")
    for name, X in (("TRAINABLE r2-proj", T), ("CONTROL r4-cont", C), ("REFERENCE r2-cont", R)):
        n = {k: sum(X[s]["cls"] == k for s in S) for k in ("SOLVED", "STALLED", "DESCENDING")}
        print(f"  {name:18s} {n}")
        for s in S:
            x = X[s]
            print(f"    s{s} {x['cls']:10s} tail {x['tail']:.4f}  peak {x['peak']:.3f} @ep{x['peak_ep']}"
                  f"  back below 0.05 @ep{x['recover_ep']}")
    st = sum(T[s]["cls"] == "SOLVED" for s in S); sc = sum(C[s]["cls"] == "SOLVED" for s in S)
    pf = fisher_exact([[st, 8 - st], [sc, 8 - sc]])[1]
    print(f"\n== primary: SOLVED trainable {st}/8 vs control {sc}/8, Fisher p {pf:.4f}")
    if sc < 6:
        verdict = "UNINFORMATIVE -- the control does not re-solve; run the gentle version"
    elif st >= 6:
        verdict = "S1 STABLE -- r=2 holds and recovers the solution; the from-scratch failure is search"
    elif st <= 2:
        verdict = "S2 UNSTABLE -- r=2 does not keep the solution under this recipe; run the gentle version"
    else:
        verdict = "MIXED"
    print(f"== BRANCH: {verdict}")
    at = acc(f"{REPO}/RANK_PROJ_TRAIN.json", "Vanilla"); ac = acc(f"{REPO}/RANK_MATCHED_e900c.json", "Vanilla_r4")
    ar = acc(f"{REPO}/RANK_MATCHED_e900c.json", "Vanilla")
    print(f"== T=1024 accuracy: trainable {np.mean(at):.4f}, control {np.mean(ac):.4f} "
          f"(perm p {perm(ac, at):.4f}), r2-cont {np.mean(ar):.4f} (trainable vs r2-cont perm p {perm(ar, at):.4f})")


if __name__ == "__main__":
    main()
