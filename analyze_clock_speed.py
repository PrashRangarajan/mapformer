"""Registered analysis for CLOCK_SPEED_PREREG.md.  python3 -m mapformer.analyze_clock_speed [--smoke] [--runs DIR --seeds ...]"""
import argparse, json, re, sys
import numpy as np
import torch

from mapformer.stats_core import perm2_p, perm2_ci, signflip_p, classify_run

REPO = "/home/prashr/mapformer"; R = f"{REPO}/runs/clock_speed/p0"
ARMS = ("Signed_r4", "Abs_r4", "CARoPE_r4"); T = 1024; K_MAX = 128; EPOCHS = 300
SEEDS = {"Signed_r4": list(range(210, 226)), "Abs_r4": list(range(210, 226)), "CARoPE_r4": list(range(210, 218))}  # Amendment 1


def signflip_ci(dd, level=0.95, lo=-2.0, hi=2.0, step=0.005):
    """Test-inversion CI for the paired mean shift: every s with sign-flip p(d - s) >= 1 - level, on a fixed grid."""
    ok = [x for x in np.arange(lo, hi + 1e-9, step) if signflip_p([v - x for v in dd])["p"] >= 1 - level]
    return (min(ok), max(ok)) if ok else (None, None)


def branch(signed_auc, other_auc):
    """PRIMARY (paired by seed): d_i = AUC(other) - AUC(signed), exact sign-flip test on mean(d); NO DIFFERENCE needs the
    95% sign-flip CI within [-0.10, +0.10] (Amendment 1, audit D1)."""
    dd = [o - g for o, g in zip(other_auc, signed_auc)]; d = float(np.mean(dd)); p = signflip_p(dd)["p"]
    lo, hi = signflip_ci(dd)
    tail = (f"paired mean d {d:+.3f} (exact sign-flip p {p:.4f}, 95% CI [{lo:+.3f}, {hi:+.3f}]; "
            f"{sum(x < 0 for x in dd)}/{len(dd)} seeds faster than Signed)")
    if d < 0 and p < .05 and -d >= 0.10:
        return f"NON-NEGATIVE LEARNS THE CLOCK FASTER: {tail}"
    if d > 0 and p < .05 and d >= 0.10:
        return f"SIGNED LEARNS THE CLOCK FASTER: {tail}"
    if lo is not None and lo >= -0.10 and hi <= 0.10:
        return f"NO DIFFERENCE: {tail}"
    return f"UNMEASURED: {tail}"


def unpaired(signed_auc, other_auc):
    d = float(np.mean(other_auc) - np.mean(signed_auc)); p = perm2_p(signed_auc, other_auc)["p"]
    ci = perm2_ci(signed_auc, other_auc, grid=(-2.0, 2.0), step=0.005)
    return f"unpaired d {d:+.3f} (perm p {p:.4f}, 95% CI [{ci['lo']:+.3f}, {ci['hi']:+.3f}])"


def first(l, thr):
    rm = np.convolve(l, np.ones(10) / 10, "valid"); h = np.nonzero(rm < thr)[0]; return int(h[0]) + 10 if len(h) else EPOCHS + 1


def smoke():
    ok = True
    s = [2.2, 2.3, 2.1, 2.25, 2.2, 2.15, 2.3, 2.2]
    for other, want in (([x - 0.4 for x in s], "NON-NEGATIVE LEARNS"), ([x + 0.4 for x in s], "SIGNED LEARNS"),
                        ([x + (0.003 if i % 2 else -0.003) for i, x in enumerate(s)], "NO DIFFERENCE"),
                        ([x + d for x, d in zip(s, [-0.5, 0.4, -0.3, 0.2, -0.4, 0.3, -0.2, 0.1])], "UNMEASURED")):
        g = branch(s, other); ok &= want in g; print(f"  {'ok ' if want in g else 'BAD'} {g}")
    print("SMOKE", "ALL PASS" if ok else "FAILED"); return ok


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--runs", default=R); ap.add_argument("--seeds", nargs="+", type=int, default=None)
    ap.add_argument("--out", default=f"{REPO}/CLOCK_SPEED_VERDICTS.json"); a = ap.parse_args()
    if a.smoke:
        sys.exit(0 if smoke() else 1)
    auc, rows = {}, {}
    for arm in ARMS:
        auc[arm], rows[arm] = [], []
        for sd in (a.seeds or SEEDS[arm]):
            b = torch.load(f"{a.runs}/{arm}_s{sd}/{arm}_recency.pt", map_location="cpu", weights_only=False)
            res = json.load(open(f"{a.runs}/{arm}_s{sd}/{arm}_recency.json")); L = np.asarray(b["losses"], float)
            assert b["variant"] == arm and b["seed"] == sd and b["k_max"] == K_MAX and b["min_gap"] == K_MAX and len(L) == EPOCHS, (arm, sd)
            hdr = re.search(r"T=(\d+) k_max=(\d+) min_gap=(\d+)", open(f"{a.runs}/{arm}_s{sd}.log").read())
            assert hdr and tuple(int(x) for x in hdr.groups()) == (T, K_MAX, K_MAX), (arm, sd)
            auc[arm].append(float(L.mean()))
            rows[arm].append(dict(seed=sd, auc=float(L.mean()), e2=first(L, 2.0), e1=first(L, 1.0), l100=float(L[99]),
                                  l200=float(L[199]), l300=float(L[299]), acc=res[str(T)]["acc"], acc2=res[str(2 * T)]["acc"],
                                  cls=classify_run(L)["registered"]))
    print(f"== CLOCK_SPEED: k-back recency, k_max {K_MAX}, T {T}, {EPOCHS} epochs; AUC = mean training loss (chance 2.77) ==")
    for arm in ARMS:
        r = rows[arm]
        print(f"  {arm:10s} AUC {np.mean(auc[arm]):.3f} +/- {np.std(auc[arm]):.3f} | epoch to <2.0 median {np.median([x['e2'] for x in r]):.0f}, "
              f"<1.0 {np.median([x['e1'] for x in r]):.0f} | loss 100/200/300 {np.median([x['l100'] for x in r]):.2f}/"
              f"{np.median([x['l200'] for x in r]):.2f}/{np.median([x['l300'] for x in r]):.2f} | acc {np.mean([x['acc'] for x in r]):.3f} "
              f"(2T {np.mean([x['acc2'] for x in r]):.3f}) | SOLVED {sum(x['cls'] == 'SOLVED' for x in r)}/{len(r)}")
    n2 = len(auc["CARoPE_r4"])
    print(f"  REGISTERED S1, PRIMARY (Abs - Signed, the sign alone, n = {len(auc['Abs_r4'])}): {branch(auc['Signed_r4'], auc['Abs_r4'])}")
    print(f"  REGISTERED S2, secondary contrast (CARoPE - Signed, first {n2} seeds): {branch(auc['Signed_r4'][:n2], auc['CARoPE_r4'])}")
    for arm in ("Abs_r4", "CARoPE_r4"):
        print(f"  secondary: {arm} - Signed AUC {unpaired(auc['Signed_r4'][:len(auc[arm])], auc[arm])}")
    allauc = [x for arm in ARMS for x in auc[arm]]; allacc = [x["acc"] for arm in ARMS for x in rows[arm]]
    print(f"  secondary: r(AUC, accuracy) over {len(allauc)} runs {np.corrcoef(allauc, allacc)[0, 1]:+.3f}")
    json.dump({"auc": auc, "rows": rows}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
