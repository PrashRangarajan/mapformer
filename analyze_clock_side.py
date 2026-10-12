"""Registered analysis for CLOCK_SIDE_PREREG.md as amended (Amendment 2, after the independent audit).
  python3 -m mapformer.analyze_clock_side            # from /home/prashr, after the batch and eval_clock_side
  python3 -m mapformer.analyze_clock_side --smoke    # branch tests on synthetic values
Primary readout: accuracy at the training length T = 1024 on the larger re-evaluation (`eval_clock_side.py`,
CLOCK_SIDE_EVAL.json, ~1,700 targets per run), eval mode; the registered branch must agree under the dropout-scale
re-score, else it carries a FLAG. Primary contrast D1: Abs_r4 - Signed_r4 (the sign alone); CARoPE_r4 and Pos_r4 are
each contrasted with Signed_r4 separately (D1c, D1p)."""
import argparse, json, re, sys
import numpy as np
import torch

from mapformer.stats_core import perm2_p, perm2_ci, classify_run, fisher_solved, signflip_p

REPO = "/home/prashr/mapformer"
R = f"{REPO}/runs/clock_side/p0"
T = 1024; K_MAX = 128; MIN_GAP = 128; EPOCHS = 300; FLOOR = 0.072   # Amendments 1-2 (8,000-episode gate)
ARMS = ("Signed_r4", "Abs_r4", "CARoPE_r4", "Pos_r4", "RoPE"); SEEDS = list(range(200, 208))


def branch(signed, mono):
    """d = mean(mono) - mean(signed)."""
    d = float(np.mean(mono) - np.mean(signed)); p = perm2_p(signed, mono)["p"]; ci = perm2_ci(signed, mono)
    tail = f"d {d:+.4f} (perm p {p:.4f}, 95% CI [{ci['lo']:+.4f}, {ci['hi']:+.4f}])"
    if d > 0 and p < .05 and d >= 0.02:
        return "CASE II ADVANTAGE", tail
    if d < 0 and p < .05 and -d >= 0.02:
        return "SIGNED BETTER", tail
    if ci["lo"] is not None and ci["lo"] >= -0.03 and ci["hi"] <= 0.03:
        return "NO DIFFERENCE (signed matches the clock)", tail
    return "UNMEASURED", tail


def d1(signed, mono, signed_rs, mono_rs, n_desc_mono):
    lab, tail = branch(signed, mono); lab_rs, tail_rs = branch(signed_rs, mono_rs)
    s = f"{lab}: {tail}"
    if lab == "CASE II ADVANTAGE":
        s += " [search / training speed: the signed arm can represent every monotone solution (RECENCY amendment)]"
        s += f" [monotone runs still DESCENDING: {n_desc_mono}/8]"
    if lab_rs != lab:
        s += f" [FLAG: under the dropout-scale re-score this reads {lab_rs}: {tail_rs}]"
    return s


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
        ok &= want in got; print(f"  {'ok ' if want in got else 'BAD'} -> {got[:150]}")
    s = 0.70 + 0.01 * rng.standard_normal(8)
    chk(d1(s, s + 0.10, s, s + 0.10, 2), "CASE II ADVANTAGE")
    chk(d1(s, s + 0.10, s, s + 0.10, 2), "search / training speed")
    chk(d1(s, s - 0.10, s, s - 0.10, 0), "SIGNED BETTER")
    chk(d1(s, s + 0.002 * rng.standard_normal(8), s, s, 0), "NO DIFFERENCE")
    chk(d1(0.5 + 0.2 * rng.standard_normal(8), 0.5 + 0.2 * rng.standard_normal(8), s, s, 0), "UNMEASURED")
    chk(d1(s, s + 0.10, s, s, 0), "FLAG: under the dropout-scale re-score")
    chk(d2(s, np.full(8, 0.25)), "INDEX FAILS")
    print("SMOKE", "ALL PASS" if ok else "FAILED"); return ok


def header(path):
    m = re.search(r"T=(\d+) k_max=(\d+) min_gap=(\d+)", open(path).read())
    return tuple(int(x) for x in m.groups()) if m else None


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--smoke", action="store_true"); a = ap.parse_args()
    if a.smoke:
        sys.exit(0 if smoke() else 1)
    E = json.load(open(f"{REPO}/CLOCK_SIDE_EVAL.json"))
    acc, acc_rs, acc_small, n_eval, solved, cls, tail = {}, {}, {}, {}, {}, {}, {}
    print(f"== CLOCK_SIDE: k-back recency, k_max {K_MAX}, trained and tested at T = {T}; floor (best trivial) {FLOOR} ==")
    for arm in ARMS:
        for dct in (acc, acc_rs, acc_small, n_eval, solved, cls, tail):
            dct[arm] = []
        for s in SEEDS:
            b = torch.load(f"{R}/{arm}_s{s}/{arm}_recency.pt", map_location="cpu", weights_only=False)
            res = json.load(open(f"{R}/{arm}_s{s}/{arm}_recency.json"))
            assert b["variant"] == arm and b["seed"] == s and b["k_max"] == K_MAX and b["min_gap"] == MIN_GAP, (arm, s)
            assert len(b["losses"]) == EPOCHS and set(res) == {str(T), str(2 * T)}, (arm, s)
            assert header(f"{R}/{arm}_s{s}.log") == (T, K_MAX, MIN_GAP), (arm, s, header(f"{R}/{arm}_s{s}.log"))
            e = E[f"{arm}|{s}"]
            acc[arm].append(e["acc"]); acc_rs[arm].append(e["acc_rs"]); n_eval[arm].append(e["n"])
            acc_small[arm].append(res[str(T)]["acc"])
            c = classify_run(b["losses"]); cls[arm].append(c["registered"]); tail[arm].append(c["tail"])
            solved[arm].append(c["registered"] == "SOLVED")
        print(f"  {arm:10s} acc {np.mean(acc[arm]):.4f} +/- {np.std(acc[arm]):.4f} (min {min(acc[arm]):.4f}; re-scored "
              f"{np.mean(acc_rs[arm]):.4f}; in-training eval {np.mean(acc_small[arm]):.4f}) | n/run {int(np.mean(n_eval[arm]))} | "
              f"SOLVED {sum(solved[arm])} STALLED {cls[arm].count('STALLED')} DESCENDING {cls[arm].count('DESCENDING')}")
    sig, sig_rs = acc["Signed_r4"], acc_rs["Signed_r4"]
    print(f"  REGISTERED D1 (Abs - Signed, the sign alone): "
          f"{d1(sig, acc['Abs_r4'], sig_rs, acc_rs['Abs_r4'], cls['Abs_r4'].count('DESCENDING'))}")
    print(f"  REGISTERED D1c (CARoPE - Signed): {d1(sig, acc['CARoPE_r4'], sig_rs, acc_rs['CARoPE_r4'], cls['CARoPE_r4'].count('DESCENDING'))}")
    print(f"  REGISTERED D1p (Pos - Signed): {d1(sig, acc['Pos_r4'], sig_rs, acc_rs['Pos_r4'], cls['Pos_r4'].count('DESCENDING'))}")
    print(f"  REGISTERED D2 INDEX CHECK (RoPE - Signed): {d2(sig, acc['RoPE'])}")
    for arm in ("Abs_r4", "CARoPE_r4", "Pos_r4", "RoPE"):
        dd = [m - g for m, g in zip(acc[arm], sig)]
        print(f"  secondary: {arm} - Signed paired by seed {np.mean(dd):+.4f} (sign-flip p {signflip_p(dd)['p']:.4f}; "
              f"{sum(x > 0 for x in dd)}/8 seeds > 0); SOLVED {sum(solved[arm])}/8 vs {sum(solved['Signed_r4'])}/8 "
              f"(Fisher p {fisher_solved(sum(solved[arm]), 8, sum(solved['Signed_r4']), 8):.4f})")
    allt = [x for arm in ARMS for x in tail[arm]]; alla = [x for arm in ARMS for x in acc[arm]]
    print(f"  secondary: r(final-5% loss, acc) over 40 runs {np.corrcoef(allt, alla)[0, 1]:+.3f}; per arm "
          + ", ".join(f"{arm} {np.corrcoef(tail[arm], acc[arm])[0, 1]:+.2f}" for arm in ARMS if np.std(acc[arm]) > 0))
    json.dump({"acc": acc, "acc_rs": acc_rs, "acc_small": acc_small, "solved": solved, "cls": cls},
              open(f"{REPO}/CLOCK_SIDE_VERDICTS.json", "w"), indent=1)


if __name__ == "__main__":
    main()
