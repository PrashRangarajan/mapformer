"""Registered analysis for RING_DIAL_PREREG.md.
  python3 -m mapformer.analyze_ring_dial            # from /home/prashr, after the batch
  python3 -m mapformer.analyze_ring_dial --smoke"""
import argparse, json, sys
import numpy as np
import torch

from mapformer.stats_core import perm2_p, perm2_ci, classify_run, fisher_solved

REPO = "/home/prashr/mapformer"; R = f"{REPO}/runs/ring_dial/p0"
ARMS = ("Signed_r4", "Abs_r4", "RoPE"); PS = (0.5, 0.9, 1.0); SEEDS = list(range(30, 36)); EPOCHS = 300


def contrast(a, b):
    """d = mean(b) - mean(a); fires if perm p < .05 and |d| >= 0.05."""
    d = float(np.mean(b) - np.mean(a)); p = perm2_p(a, b)["p"]; ci = perm2_ci(a, b)
    return d, p, ci, (p < .05 and abs(d) >= 0.05)


def dm(signed, mono, name):
    d, p, ci, fires = contrast(signed, mono)
    tail = f"d {d:+.4f} (perm p {p:.4f}, 95% CI [{ci['lo']:+.4f}, {ci['hi']:+.4f}])"
    return (f"{name}: {tail}" if fires and d < 0 else f"NOT SHOWN: {tail}"), (fires and d < 0)


def dc(signed, mono):
    d, p, ci, fires = contrast(signed, mono)
    tail = f"d {d:+.4f} (perm p {p:.4f}, 95% CI [{ci['lo']:+.4f}, {ci['hi']:+.4f}])"
    if ci["lo"] is not None and ci["lo"] >= -0.02 and ci["hi"] <= 0.02:
        ceil = " (both at ceiling)" if min(signed) >= 0.999 and min(mono) >= 0.999 else ""
        return f"MONOTONE MATCHES ON THE CLOCK{ceil}: {tail}", True
    if fires and d < 0:
        return f"MONOTONE WORSE ON THE CLOCK: {tail}", False
    return f"UNMEASURED: {tail}", False


def smoke():
    ok = True
    def chk(got, want):
        nonlocal ok
        ok &= want in got; print(f"  {'ok ' if want in got else 'BAD'} -> {got}")
    one = [1.0] * 6; low = [0.80, 0.82, 0.79, 0.81, 0.80, 0.83]
    chk(dm(one, low, "MONOTONE FAILS ON THE MAP")[0], "MONOTONE FAILS ON THE MAP")
    chk(dm(one, [0.999, 1.0, 0.998, 1.0, 1.0, 0.999], "X")[0], "NOT SHOWN")
    chk(dc(one, one)[0], "MATCHES ON THE CLOCK (both at ceiling)")
    chk(dc(one, low)[0], "WORSE ON THE CLOCK")
    chk(dc([0.9, 0.6, 1.0, 0.7, 0.95, 0.8], [0.7, 0.9, 0.6, 1.0, 0.85, 0.75])[0], "UNMEASURED")
    print("SMOKE", "ALL PASS" if ok else "FAILED"); return ok


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--smoke", action="store_true"); a = ap.parse_args()
    if a.smoke:
        sys.exit(0 if smoke() else 1)
    acc = {}; acc4 = {}; sol = {}
    for p in PS:
        for arm in ARMS:
            acc[p, arm], acc4[p, arm], sol[p, arm] = [], [], []
            for s in SEEDS:
                d = f"{R}/{arm}_p{p}_s{s}"; b = torch.load(f"{d}/{arm}.pt", map_location="cpu", weights_only=False)
                ar = b["config"]["args"]; ev = json.load(open(f"{d}/eval.json"))["eval"]
                assert ar["arm"] == arm and ar["p_plus"] == p and b["seed"] == s and len(b["losses"]) == EPOCHS, (d, ar)
                acc[p, arm].append(ev["128"]["acc"]); acc4[p, arm].append(ev["512"]["acc"])
                sol[p, arm].append(classify_run(b["losses"])["registered"] == "SOLVED")
    print("== RING_DIAL: accuracy at T = 128 (eval mode), mean (min) | SOLVED | acc T = 512 ==")
    for p in PS:
        print(f"  p_plus {p}: " + " | ".join(f"{arm} {np.mean(acc[p, arm]):.4f} ({min(acc[p, arm]):.4f}) {sum(sol[p, arm])}/6 "
                                             f"[{np.mean(acc4[p, arm]):.3f}]" for arm in ARMS))
    l_m, f_m = dm(acc[0.5, "Signed_r4"], acc[0.5, "Abs_r4"], "MONOTONE FAILS ON THE MAP")
    l_d, _ = dm(acc[0.9, "Signed_r4"], acc[0.9, "Abs_r4"], "MONOTONE COSTS WHERE BACKWARD STEPS ARE RARE")
    l_c, f_c = dc(acc[1.0, "Signed_r4"], acc[1.0, "Abs_r4"])
    print(f"  REGISTERED DM (p 0.5, Abs - Signed): {l_m}")
    print(f"  REGISTERED DD (p 0.9, Abs - Signed): {l_d}")
    print(f"  REGISTERED DC (p 1.0, Abs - Signed): {l_c}")
    print(f"  REGISTERED DIAL: {'HOWARD CASE II PATTERN HOLDS' if f_m and f_c else 'DOES NOT HOLD AS STATED'}")
    for p in PS:
        d, pp, ci, _ = contrast(acc[p, "Signed_r4"], acc[p, "RoPE"])
        print(f"  REGISTERED DI (p {p}, RoPE - Signed): d {d:+.4f} (perm p {pp:.4f}); SOLVED {sum(sol[p, 'RoPE'])}/6 vs "
              f"{sum(sol[p, 'Signed_r4'])}/6 (Fisher p {fisher_solved(sum(sol[p, 'RoPE']), 6, sum(sol[p, 'Signed_r4']), 6):.4f})")
    json.dump({f"{p}|{arm}": {"acc": acc[p, arm], "acc512": acc4[p, arm], "solved": sol[p, arm]} for p in PS for arm in ARMS},
              open(f"{REPO}/RING_DIAL_VERDICTS.json", "w"), indent=1)


if __name__ == "__main__":
    main()
