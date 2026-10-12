"""Registered analysis for CLOCK_SIDE_PREREG.md (Howard Case II vs III on a content-clock task below ceiling).
  python3 -m mapformer.analyze_clock_side            # from /home/prashr, after the batch
  python3 -m mapformer.analyze_clock_side --smoke    # branch tests on synthetic values
T and K_MAX are set by Amendment 1 (the calibration); the analysis asserts them against every run's saved results."""
import argparse, json, sys
import numpy as np
import torch

from mapformer.stats_core import perm2_p, perm2_ci, classify_run, fisher_solved

REPO = "/home/prashr/mapformer"
R = f"{REPO}/runs/clock_side/p0"
T = 1024; K_MAX = 128                       # set by Amendment 1 (calibration)
ARMS = ("Signed_r4", "CARoPE_r4", "Pos_r4", "RoPE"); SEEDS = list(range(200, 208))


def d1(signed, mono):
    """signed, mono: per-seed accuracies (mono = mean of CARoPE and Pos per seed)."""
    d = float(np.mean(mono) - np.mean(signed)); p = perm2_p(signed, mono)["p"]; ci = perm2_ci(signed, mono)
    tail = f"d {d:+.4f} (perm p {p:.4f}, 95% CI [{ci['lo']:+.4f}, {ci['hi']:+.4f}])"
    if d > 0 and p < .05 and d >= 0.02:
        return f"CASE II ADVANTAGE (double dissociation): {tail}"
    if d < 0 and p < .05 and -d >= 0.02:
        return f"SIGNED BETTER: {tail}"
    if ci["lo"] is not None and ci["lo"] >= -0.03 and ci["hi"] <= 0.03:
        return f"NO DIFFERENCE (signed matches the clock): {tail}"
    return f"UNMEASURED: {tail}"


def d2(signed, rope):
    d = float(np.mean(rope) - np.mean(signed)); p = perm2_p(signed, rope)["p"]
    if d <= -0.10 and p < .05:
        return f"INDEX FAILS (d {d:+.4f}, perm p {p:.4f})"
    return f"AS IT FALLS: d {d:+.4f}, perm p {p:.4f}"


def smoke():
    ok = True
    rng = np.random.default_rng(0)
    def chk(got, want):
        nonlocal ok
        ok &= want in got; print(f"  {'ok ' if want in got else 'BAD'} -> {got}")
    s = 0.70 + 0.01 * rng.standard_normal(8)
    chk(d1(s, s + 0.10), "CASE II ADVANTAGE")
    chk(d1(s, s - 0.10), "SIGNED BETTER")
    chk(d1(s, s + 0.002 * rng.standard_normal(8)), "NO DIFFERENCE")
    chk(d1(0.5 + 0.2 * rng.standard_normal(8), 0.5 + 0.2 * rng.standard_normal(8)), "UNMEASURED")
    chk(d2(s, np.full(8, 0.25) + 0.01 * rng.standard_normal(8)), "INDEX FAILS")
    chk(d2(s, s), "AS IT FALLS")
    print("SMOKE", "ALL PASS" if ok else "FAILED"); return ok


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--smoke", action="store_true"); a = ap.parse_args()
    if a.smoke:
        sys.exit(0 if smoke() else 1)
    assert T is not None and K_MAX is not None, "T / K_MAX not set (Amendment 1)"
    acc = {}; acc2 = {}; solved = {}
    print(f"== CLOCK_SIDE: k-back recency, k_max {K_MAX}, trained and tested at T = {T} (eval mode) ==")
    for arm in ARMS:
        acc[arm], acc2[arm], solved[arm] = [], [], []
        for s in SEEDS:
            b = torch.load(f"{R}/{arm}_s{s}/{arm}_recency.pt", map_location="cpu", weights_only=False)
            res = json.load(open(f"{R}/{arm}_s{s}/{arm}_recency.json"))
            assert b["variant"] == arm and b["seed"] == s and b["k_max"] == K_MAX and len(b["losses"]) == 300, (arm, s)
            assert set(res) == {str(T), str(2 * T)}, (arm, s, list(res))
            acc[arm].append(res[str(T)]["acc"]); acc2[arm].append(res[str(2 * T)]["acc"])
            solved[arm].append(classify_run(b["losses"])["registered"] == "SOLVED")
        print(f"  {arm:10s} acc T {np.mean(acc[arm]):.4f} +/- {np.std(acc[arm]):.4f} (min {min(acc[arm]):.4f}) | SOLVED "
              f"{sum(solved[arm])}/8 | acc 2T (OOD) {np.mean(acc2[arm]):.4f}")
    mono = [(c + p) / 2 for c, p in zip(acc["CARoPE_r4"], acc["Pos_r4"])]
    print(f"  REGISTERED D1 CLOCK SIDE (mean(CARoPE, Pos) - Signed): {d1(acc['Signed_r4'], mono)}")
    print(f"  REGISTERED D2 INDEX CHECK (RoPE - Signed): {d2(acc['Signed_r4'], acc['RoPE'])}")
    for arm in ("CARoPE_r4", "Pos_r4", "RoPE"):
        print(f"  secondary: SOLVED {arm} {sum(solved[arm])}/8 vs Signed {sum(solved['Signed_r4'])}/8, "
              f"Fisher p {fisher_solved(sum(solved[arm]), 8, sum(solved['Signed_r4']), 8):.4f}; "
              f"acc {arm} - Signed {np.mean(acc[arm]) - np.mean(acc['Signed_r4']):+.4f}")
    json.dump({"acc": acc, "acc2T": acc2, "solved": solved}, open(f"{REPO}/CLOCK_SIDE_VERDICTS.json", "w"), indent=1)


if __name__ == "__main__":
    main()
